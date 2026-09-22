"""Continuous-outcome pipelines: linear models with RCS, ANCOVA adjustment,
and a random-intercept mixed model for repeated measures. See ch09.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .imputation import MiceImputer
from .metrics import oob_r_squared
from .splines import rcs_design, likelihood_ratio_test


def fit_linear_pipeline(
    data: pd.DataFrame,
    outcome: str,
    *,
    spline_predictors: dict[str, int] | None = None,
    linear_continuous: list[str] | None = None,
    categorical: list[str] | None = None,
    m_imputation: int = 20,
    seed: int | None = None,
    trace=None,
) -> list:
    """Fit a linear model in each imputed dataset; returns statsmodels results.

    Compare nested models with likelihood_ratio_test(); report out-of-sample
    R-squared from validation.py, never in-sample R-squared.
    """
    spline_predictors = spline_predictors or {}
    linear_continuous = linear_continuous or []
    categorical = categorical or []

    imputer = MiceImputer(m=m_imputation, random_state=seed)
    result = imputer.fit_transform(data)

    models = []
    for completed in result.imputed:
        design = _design(completed, spline_predictors, linear_continuous, categorical)
        model = sm.OLS(completed[outcome].astype(float), sm.add_constant(design)).fit()
        models.append(model)

    if trace is not None:
        trace.log(
            "linear_pipeline",
            chosen={"spline_predictors": spline_predictors,
                    "linear_continuous": linear_continuous,
                    "categorical": categorical},
            rationale="OLS with RCS; nested comparisons by LR test",
            stage="modeling",
        )
    return models


def ancova(data: pd.DataFrame, outcome_change: str, baseline: str,
           covariates: list[str], group: str) -> sm.regression.linear_model.RegressionResults:
    """Change-score model adjusted for the baseline value (guards regression to
    the mean). Covariates follow the case study: baseline measure, age, severity,
    volume, center. (ch09)
    """
    formula = f"{outcome_change} ~ {group} + {baseline} + " + " + ".join(covariates)
    return smf.ols(formula, data=data).fit()


def repeated_measures(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    group: str,
    patient_id: str,
    covariates: list[str] | None = None,
) -> sm.regression.mixed_linear_model.MixedLMResults:
    """Random-intercept mixed model: outcome ~ time * group + covariates.

    With exactly two timepoints this is algebraically a difference-in-differences
    on the increment and does NOT estimate trajectory shape — say so in the
    paper (ch09, ch16). Random slopes require >= 3 timepoints.
    """
    covariates = covariates or []
    formula = f"{outcome} ~ {time} * {group}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    return smf.mixedlm(formula, data=data, groups=data[patient_id]).fit()


def _design(data: pd.DataFrame, spline_predictors: dict[str, int],
            linear_continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    parts = []
    for predictor, n_knots in spline_predictors.items():
        parts.append(rcs_design(data[predictor].to_numpy(), n_knots, name=predictor))
    for predictor in linear_continuous:
        parts.append(pd.DataFrame({predictor: data[predictor].to_numpy(dtype=float)}))
    if categorical:
        parts.append(pd.get_dummies(data[categorical], drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


__all__ = [
    "fit_linear_pipeline", "ancova", "repeated_measures",
    "oob_r_squared", "likelihood_ratio_test",
]
