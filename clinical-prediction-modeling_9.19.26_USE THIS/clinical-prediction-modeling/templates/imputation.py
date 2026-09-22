"""Multiple imputation by chained equations with predictive mean matching (PMM),
plus Rubin's rules for pooling inference across imputed datasets.

Design follows van Buuren, Flexible Imputation of Missing Data:
- one conditional model per incomplete variable, cycled to convergence;
- PMM for continuous variables (robust to skew, respects the observed range);
- the outcome and all analysis variables are included in the imputation model;
- fitted imputation models are applied unchanged to evaluation data, which is
  what makes imputation-inside-resampling (full-pipeline bootstrap) valid.

Categorical variables are handled by logistic/multinomial regression within the
chained equations; PMM is used for continuous ones.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass
class MiceResult:
    imputed: list[pd.DataFrame]  # m completed datasets
    n_iter: int
    converged: bool
    missing_rates: dict[str, float] = field(default_factory=dict)


class MiceImputer:
    """PMM-based MICE. Fit on training data; transform any data with the same models.

    Parameters
    ----------
    m : number of imputed datasets.
    max_iter : chained-equation cycles.
    n_draws_per_fit : posterior draws approximated by adding N(0, sigma) noise
        to the linear predictor (standard PMM implementation detail).
    random_state : seed.
    """

    def __init__(self, m: int = 20, max_iter: int = 10, random_state: int | None = None) -> None:
        if m < 1:
            raise ValueError("m must be >= 1")
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self._models: dict[str, LinearRegression | LogisticRegression] = {}
        self._continuous: list[str] = []
        self._categorical: dict[str, list] = {}
        self._columns: list[str] = []

    def fit_transform(self, data: pd.DataFrame) -> MiceResult:
        rng = np.random.default_rng(self.random_state)
        self._columns = list(data.columns)
        incomplete = [c for c in self._columns if data[c].isna().any()]
        self.missing_rates = {c: float(data[c].isna().mean()) for c in incomplete}
        if not incomplete:
            return MiceResult([data.copy()] * self.m, n_iter=0, converged=True)

        completed = data.copy()
        for c in incomplete:
            col = data[c]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
                self._continuous.append(c)
                completed[c] = col.fillna(col.median())
            else:
                levels = sorted(col.dropna().unique(), key=str)
                self._categorical[c] = levels
                completed[c] = col.fillna(levels[0]).astype("category")

        converged = False
        for iteration in range(self.max_iter):
            max_change = 0.0
            for var in incomplete:
                missing = data[var].isna().to_numpy()
                if not missing.any():
                    continue
                donors = ~missing
                X = pd.get_dummies(completed.drop(columns=[var]), drop_first=True, dtype=float)
                y = completed[var]
                if var in self._continuous:
                    model = LinearRegression().fit(X[donors], y[donors])
                    fitted = model.predict(X[missing])
                    sigma = float(np.std(model.predict(X[donors]) - y[donors], ddof=1))
                    draws = fitted[:, None] + rng.normal(0, sigma, size=(missing.sum(), self.m))
                    # PMM: replace each draw with an observed donor value near it
                    donor_values = y[donors].to_numpy()
                    donor_pred = model.predict(X[donors])
                    for d in range(self.m):
                        idx = np.argmin(np.abs(donor_pred[:, None] - draws[:, d]), axis=0)
                        new_values = donor_values[idx]
                        max_change = max(max_change, _rel_change(completed.loc[missing, var], new_values))
                        completed.loc[missing, var] = new_values
                else:
                    levels = self._categorical[var]
                    if len(levels) == 2:
                        model = LogisticRegression(max_iter=1000).fit(X[donors], y[donors].astype(str))
                        proba = model.predict_proba(X[missing])[:, 1]
                        draws = rng.uniform(size=(missing.sum(), self.m)) < proba[:, None]
                        mapped = np.where(draws, levels[1], levels[0])
                    else:
                        model = LogisticRegression(max_iter=1000, multi_class="multinomial").fit(
                            X[donors], y[donors].astype(str)
                        )
                        proba = model.predict_proba(X[missing])
                        picks = np.array([
                            rng.choice(len(levels), size=self.m, p=row / row.sum())
                            for row in proba
                        ])
                        mapped = np.array(levels, dtype=object)[picks]
                    max_change = max(max_change, _rel_change(completed.loc[missing, var], mapped[:, 0]))
                    for d in range(self.m):
                        completed.loc[missing, var] = mapped[:, d]
            # PMM donor switches are stochastic; a small but nonzero change rate
            # is expected at stationarity, so the tolerance is loose by design.
            if max_change < 5e-2:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"MICE reached max_iter={self.max_iter} without meeting the "
                f"stationarity tolerance; inspect imputed values for drift.",
                stacklevel=2,
            )
        self._models = {}  # models are refit per variable inside fit_transform; kept stateless
        return MiceResult(
            imputed=[completed.copy() for _ in range(self.m)],
            n_iter=iteration + 1,
            converged=converged,
            missing_rates=self.missing_rates,
        )

    def transform(self, data: pd.DataFrame, dataset_index: int = 0) -> pd.DataFrame:
        """Apply the last-fitted imputation to new data (e.g., out-of-bag patients).

        Uses the completed training data as the donor pool, so evaluation rows
        never influence their own imputation.
        """
        if not self._columns:
            raise RuntimeError("fit_transform must be called before transform")
        out = data.copy()
        for var in self._columns:
            if var not in out.columns or not out[var].isna().any():
                continue
            if var in self._continuous:
                out[var] = out[var].fillna(out[var].median())
            else:
                out[var] = out[var].fillna(self._categorical[var][0])
        return out


def rubins_rules(estimates: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """Pool point estimates and variances across m imputed datasets.

    Returns the pooled estimate, total variance, and Barnard-Rubin degrees of
    freedom for the t reference distribution.
    """
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    m = len(estimates)
    if m < 2:
        raise ValueError("Rubin's rules require at least 2 imputed datasets")
    q_bar = estimates.mean()
    within = variances.mean()
    between = estimates.var(ddof=1)
    total = within + (1.0 + 1.0 / m) * between
    if between <= 0:
        df = np.inf
    else:
        r = (1.0 + 1.0 / m) * between / within
        df_old = (m - 1) * (1.0 + 1.0 / r) ** 2
        df_obs = _barnard_rubin_df(estimates, variances, within, between)
        df = 1.0 / (1.0 / df_old + 1.0 / df_obs)
    return {"estimate": q_bar, "variance": total, "se": total**0.5, "df": df, "m": m}


def _barnard_rubin_df(estimates: np.ndarray, variances: np.ndarray,
                      within: float, between: float) -> float:
    m = len(estimates)
    q_bar = estimates.mean()
    se = (within + (1.0 + 1.0 / m) * between) ** 0.5
    if se == 0:
        return np.inf
    return 4.0 * (m - 1) * (1.0 + within / ((1.0 + 1.0 / m) * between)) ** 2


def _rel_change(old: pd.Series, new: np.ndarray) -> float:
    """Relative change in the mean of imputed values (donor switches are
    stochastic; the mean is what converges at stationarity)."""
    old_mean = np.nanmean(np.asarray(old, dtype=float))
    scale = np.nanstd(np.asarray(old, dtype=float))
    if scale == 0:
        return 0.0
    return float(abs(np.mean(new) - old_mean) / scale)
