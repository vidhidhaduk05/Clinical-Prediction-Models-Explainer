"""Survival pipeline: Cox proportional hazards with RCS predictors, Schoenfeld
residual PH tests, concordance, and time-horizon Brier scoring. See ch08.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from .imputation import MiceImputer
from .splines import rcs_design


def fit_cox_pipeline(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    *,
    spline_predictors: dict[str, int] | None = None,
    linear_continuous: list[str] | None = None,
    categorical: list[str] | None = None,
    m_imputation: int = 20,
    seed: int | None = None,
    trace=None,
) -> tuple[list[CoxPHFitter], list[pd.DataFrame]]:
    """Fit a Cox model in each of m imputed datasets.

    Returns (models, design_frames) — the design frames are needed for
    Schoenfeld PH tests. Pool coefficients with rubins_rules() from
    imputation.py; test the PH assumption with ph_diagnostics() (report
    per-covariate tests, ch08).
    """
    spline_predictors = spline_predictors or {}
    linear_continuous = linear_continuous or []
    categorical = categorical or []

    imputer = MiceImputer(m=m_imputation, random_state=seed)
    result = imputer.fit_transform(data)

    models, frames = [], []
    for completed in result.imputed:
        design = _cox_design(completed, spline_predictors, linear_continuous, categorical)
        frame = design.copy()
        frame[duration_col] = completed[duration_col].to_numpy()
        frame[event_col] = completed[event_col].to_numpy()
        cph = CoxPHFitter()
        cph.fit(frame, duration_col=duration_col, event_col=event_col)
        models.append(cph)
        frames.append(frame)

    if trace is not None:
        trace.log(
            "cox_pipeline",
            chosen={"spline_predictors": spline_predictors,
                    "linear_continuous": linear_continuous,
                    "categorical": categorical, "m_imputation": m_imputation},
            alternatives=["binary outcome model (rejected: censoring present)"],
            rationale="Cox PH with RCS; PH assumption tested separately",
            stage="modeling",
        )
    return models, frames


def ph_diagnostics(model: CoxPHFitter, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-covariate Schoenfeld residual tests for the PH assumption.

    Significant P values indicate time-varying effects; report time-averaged
    effects with a caveat or fit time-varying coefficients (ch08).
    """
    test = proportional_hazard_test(model, frame, time_transform="rank")
    return test.summary


def concordance(models: list[CoxPHFitter], frames: list[pd.DataFrame]) -> float:
    """Pooled concordance across imputed datasets (mean of per-dataset C-indices)."""
    return float(np.mean([m.concordance_index_ for m in models]))


def km_curve(durations: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate; returns the fitted KaplanMeierFitter."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    return kmf


def _cox_design(data: pd.DataFrame, spline_predictors: dict[str, int],
                linear_continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    parts = []
    for predictor, n_knots in spline_predictors.items():
        parts.append(rcs_design(data[predictor].to_numpy(), n_knots, name=predictor))
    for predictor in linear_continuous:
        parts.append(pd.DataFrame({predictor: data[predictor].to_numpy(dtype=float)}))
    if categorical:
        parts.append(pd.get_dummies(data[categorical], drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)
