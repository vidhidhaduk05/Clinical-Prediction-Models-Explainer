"""Binary-outcome pipeline: logistic regression with restricted cubic splines,
multiple imputation inside resampling, and marginal standardization for
adjusted risks. See ch03, ch06, ch07, ch11.

The pipeline object returned by `fit_binary_pipeline` is self-contained: it
carries its own imputer, spline knots, and scaler, so applying it to
out-of-bag patients never leaks information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .imputation import MiceImputer
from .splines import rcs_design, rcs_design_with_knots


@dataclass
class BinaryPipeline:
    continuous_splines: dict[str, int]          # predictor -> n_knots
    linear_continuous: list[str]
    categorical: list[str]
    m_imputation: int = 20
    seed: int | None = None
    _imputer: MiceImputer | None = None
    _models: list = field(default_factory=list)  # one statsmodels fit per imputed dataset
    _design_columns: list[str] = field(default_factory=list)
    _knots: dict[str, list[float]] = field(default_factory=dict)  # fit-time knots
    _fill_values: dict[str, float] = field(default_factory=dict)  # training-derived fills

    def predict_proba(self, data: pd.DataFrame, dataset_index: int = 0) -> np.ndarray:
        design = self._design(data)
        model = self._models[dataset_index % len(self._models)]
        return np.asarray(model.predict(sm.add_constant(design, has_constant="add")))

    def _design(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col, value in self._fill_values.items():
            data[col] = data[col].fillna(value)
        parts = []
        for predictor, n_knots in self.continuous_splines.items():
            knots = self._knots[predictor]
            parts.append(rcs_design_with_knots(data[predictor].to_numpy(), knots, name=predictor))
        for predictor in self.linear_continuous:
            parts.append(pd.DataFrame({predictor: data[predictor].to_numpy(dtype=float)}))
        if self.categorical:
            dummies = pd.get_dummies(data[self.categorical], drop_first=True, dtype=float)
            parts.append(dummies)
        design = pd.concat(parts, axis=1)
        # align to training columns (handles unseen levels / column order)
        design = design.reindex(columns=self._design_columns, fill_value=0)
        return design


def fit_binary_pipeline(
    data: pd.DataFrame,
    outcome: str,
    *,
    spline_predictors: dict[str, int],
    linear_continuous: list[str],
    categorical: list[str],
    m_imputation: int = 20,
    seed: int | None = None,
    trace=None,
) -> BinaryPipeline:
    """Fit the full pipeline: MICE -> spline design -> logistic model per dataset."""
    imputer = MiceImputer(m=m_imputation, random_state=seed)
    result = imputer.fit_transform(data)

    pipeline = BinaryPipeline(
        continuous_splines=spline_predictors,
        linear_continuous=linear_continuous,
        categorical=categorical,
        m_imputation=m_imputation,
        seed=seed,
        _imputer=imputer,
    )

    first = result.imputed[0]
    probe = _build_design(first, spline_predictors, linear_continuous, categorical)
    pipeline._design_columns = list(probe.columns)
    for predictor, n_knots in spline_predictors.items():
        pipeline._knots[predictor] = list(rcs_design(first[predictor].to_numpy(), n_knots,
                                                     name=predictor).attrs["knots"])
    for predictor in [*spline_predictors, *linear_continuous]:
        pipeline._fill_values[predictor] = float(first[predictor].median())
    for predictor in categorical:
        pipeline._fill_values[predictor] = first[predictor].mode().iloc[0]

    for completed in result.imputed:
        design = _build_design(completed, spline_predictors, linear_continuous, categorical)
        model = sm.Logit(completed[outcome].astype(float), sm.add_constant(design, has_constant="add")).fit(disp=0)
        pipeline._models.append(model)

    if trace is not None:
        trace.log(
            "binary_pipeline",
            chosen={
                "spline_predictors": spline_predictors,
                "linear_continuous": linear_continuous,
                "categorical": categorical,
                "m_imputation": m_imputation,
                "n_observations": len(data),
            },
            alternatives=["complete-case analysis (rejected: outcome-associated missingness)",
                          "median-split exposure (rejected: display only)"],
            rationale="MICE inside pipeline; RCS default for continuous predictors",
            stage="modeling",
        )
    return pipeline


def _build_design(data: pd.DataFrame, spline_predictors: dict[str, int],
                  linear_continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    parts = []
    for predictor, n_knots in spline_predictors.items():
        parts.append(rcs_design(data[predictor].to_numpy(), n_knots, name=predictor))
    for predictor in linear_continuous:
        parts.append(pd.DataFrame({predictor: data[predictor].to_numpy(dtype=float)}))
    if categorical:
        parts.append(pd.get_dummies(data[categorical], drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


def marginal_standardization(
    pipeline: BinaryPipeline,
    data: pd.DataFrame,
    exposure: str,
    values: tuple[float, ...] = (0.0, 1.0),
) -> dict[float, float]:
    """Adjusted absolute risks at given exposure values, averaged over the cohort.

    Predicts every patient's risk with the exposure fixed at each value and
    averages — the communicable counterpart of the adjusted OR (ch07).
    """
    risks: dict[float, float] = {}
    for value in values:
        counterfactual = data.copy()
        counterfactual[exposure] = value
        proba = pipeline.predict_proba(counterfactual)
        risks[value] = float(np.mean(proba))
    return risks
