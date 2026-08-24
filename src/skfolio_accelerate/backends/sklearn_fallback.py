"""Eligibility checks and sklearn GridSearchCV fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from skfolio.model_selection import BaseCombinatorialCV
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex import ObjectiveFunction
from skfolio import RiskMeasure

from skfolio_accelerate.classify import grid_has_non_executable

SUPPORTED_OBJECTIVES = {
    ObjectiveFunction.MINIMIZE_RISK,
    ObjectiveFunction.MAXIMIZE_UTILITY,
}

SUPPORTED_RISK_MEASURES = {
    RiskMeasure.VARIANCE,
    RiskMeasure.CVAR,
}


def _nonzero(value: Any) -> bool:
    arr = np.asarray(value, dtype=float)
    return bool(np.any(np.abs(arr) > 0))


def _grid_values(param_grid, key: str) -> list[Any]:
    if isinstance(param_grid, list):
        values: list[Any] = []
        for grid in param_grid:
            values.extend(grid.get(key, []))
        return values
    return list(param_grid.get(key, [])) if isinstance(param_grid, dict) else []


def acceleration_blocked_reason(
    estimator,
    param_grid,
    cv,
    scoring=None,
) -> str | None:
    del scoring
    if not isinstance(estimator, MeanRisk):
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    solver = getattr(estimator, "solver", "CLARABEL")
    solver_values = _grid_values(param_grid, "solver") or [solver]
    if any(str(value).upper() != "CLARABEL" for value in solver_values):
        return "solver is not CLARABEL"
    for attr, label in (
        ("cardinality", "cardinality (MIP)"),
        ("group_cardinalities", "group_cardinalities (MIP)"),
        ("threshold_long", "threshold_long (MIP)"),
        ("threshold_short", "threshold_short (MIP)"),
        ("add_constraints", "add_constraints callables"),
        ("add_objective", "add_objective callables"),
        ("overwrite_expected_return", "overwrite_expected_return"),
        ("efficient_frontier_size", "efficient_frontier_size"),
        ("mu_uncertainty_set_estimator", "mu uncertainty sets"),
        ("covariance_uncertainty_set_estimator", "covariance uncertainty sets"),
    ):
        if getattr(estimator, attr, None) is not None:
            return f"{label} are not accelerable"
    objective = getattr(
        estimator, "objective_function", ObjectiveFunction.MINIMIZE_RISK
    )
    objectives = _grid_values(param_grid, "objective_function") or [objective]
    if any(item not in SUPPORTED_OBJECTIVES for item in objectives):
        return "objective_function is not accelerable"
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    risks = _grid_values(param_grid, "risk_measure") or [risk]
    if any(item not in SUPPORTED_RISK_MEASURES for item in risks):
        return "risk_measure is not accelerable in v0.1"
    if grid_has_non_executable(param_grid):
        return "param_grid contains non-executable hyperparameters"
    if _nonzero(getattr(estimator, "transaction_costs", 0.0)) or _nonzero(
        getattr(estimator, "management_fees", 0.0)
    ):
        return "transaction costs / management fees are not accelerable in v0.1"
    return None


def sklearn_grid_search(
    estimator,
    param_grid,
    cv,
    scoring,
    n_jobs,
    refit,
    error_score,
    verbose,
    pre_dispatch: str | int = "2*n_jobs",
    return_train_score: bool = False,
):
    if isinstance(cv, BaseCombinatorialCV):
        raise TypeError(
            "sklearn GridSearchCV cannot consume CombinatorialPurgedCV splits. "
            "Use backend='python'/'rust'/'auto' or a standard splitter."
        )
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        error_score=error_score,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        return_train_score=return_train_score,
    )
