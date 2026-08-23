"""Classify MeanRisk hyperparameters as structural, numerical, data, or fallback."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import ParameterGrid

from skfolio_accelerate.ir import ParameterClass

NUMERICAL_PARAMS = {
    "l1_coef",
    "l2_coef",
    "risk_aversion",
    "min_return",
    "cvar_beta",
}

STRUCTURAL_PARAMS = {
    "risk_measure",
    "objective_function",
    "solver",
    "cardinality",
    "group_cardinalities",
    "threshold_long",
    "threshold_short",
    "add_constraints",
    "add_objective",
    "overwrite_expected_return",
    "efficient_frontier_size",
    "min_weights",
    "max_weights",
    "budget",
    "transaction_costs",
    "management_fees",
}

DATA_PARAMS = {
    "prior_estimator",
    "mu_uncertainty_set_estimator",
    "covariance_uncertainty_set_estimator",
}

DATA_PREFIXES = ("prior_estimator__",)


def classify_param_name(name: str) -> ParameterClass:
    if name in NUMERICAL_PARAMS:
        return ParameterClass.NUMERICAL
    if name in STRUCTURAL_PARAMS:
        return ParameterClass.STRUCTURAL
    if name in DATA_PARAMS or name.startswith(DATA_PREFIXES):
        return ParameterClass.DATA
    return ParameterClass.NON_EXECUTABLE


def classify_grid(
    param_grid: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, ParameterClass]:
    keys: set[str] = set()
    for combo in ParameterGrid(param_grid):
        keys.update(combo)
    return {key: classify_param_name(key) for key in sorted(keys)}


def classify_param_grid(
    estimator,
    param_grid: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, ParameterClass]:
    del estimator
    return classify_grid(param_grid)


def grid_has_non_executable(param_grid: dict[str, Any] | list[dict[str, Any]]) -> bool:
    classes = classify_grid(param_grid)
    return any(kind is ParameterClass.NON_EXECUTABLE for kind in classes.values())


def data_fingerprint(params: dict[str, Any]) -> str:
    items = [
        (key, repr(value))
        for key, value in sorted(params.items())
        if key == "prior_estimator" or key.startswith("prior_estimator__")
    ]
    return repr(items)
