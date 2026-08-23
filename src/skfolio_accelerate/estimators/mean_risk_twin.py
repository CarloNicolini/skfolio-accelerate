"""DPP-parameterized MeanRisk twins that mirror skfolio cone formulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.moments import FoldMoments


def _resolved(estimator, params: dict[str, Any], name: str, default=None):
    if name in params:
        return params[name]
    return getattr(estimator, name, default)


def structure_key(
    estimator,
    params: dict[str, Any],
    *,
    n_observations: int,
    n_assets: int,
) -> str:
    risk_measure = _resolved(estimator, params, "risk_measure")
    objective = _resolved(estimator, params, "objective_function")
    include_min_return = _resolved(estimator, params, "min_return") is not None
    include_l1 = float(_resolved(estimator, params, "l1_coef", 0.0) or 0.0) != 0.0
    train_len = n_observations if risk_measure is RiskMeasure.CVAR else 0
    return (
        f"{risk_measure}|{objective}|T={train_len}|n={n_assets}|"
        f"min_return={include_min_return}|l1={include_l1}"
    )


def build_twin_from_estimator(
    estimator,
    params: dict[str, Any],
    *,
    n_observations: int,
    n_assets: int,
) -> TwinProblem:
    risk_measure = _resolved(estimator, params, "risk_measure")
    objective = _resolved(
        estimator, params, "objective_function", ObjectiveFunction.MINIMIZE_RISK
    )
    return build_mean_risk_twin(
        n_assets,
        risk_measure=risk_measure,
        objective_function=objective,
        n_observations=n_observations,
        min_weights=_resolved(estimator, params, "min_weights", 0.0),
        max_weights=_resolved(estimator, params, "max_weights", 1.0),
        budget=float(_resolved(estimator, params, "budget", 1.0) or 1.0),
        include_min_return=_resolved(estimator, params, "min_return") is not None,
        include_l1=float(_resolved(estimator, params, "l1_coef", 0.0) or 0.0) != 0.0,
        scale_objective=_resolved(estimator, params, "scale_objective"),
        scale_constraints=_resolved(estimator, params, "scale_constraints"),
    )


def bind_from_estimator(
    twin: TwinProblem,
    moments: FoldMoments,
    estimator,
    params: dict[str, Any],
) -> None:
    bind_twin_values(
        twin,
        moments,
        l1_coef=float(_resolved(estimator, params, "l1_coef", 0.0) or 0.0),
        l2_coef=float(_resolved(estimator, params, "l2_coef", 0.0) or 0.0),
        risk_aversion=float(_resolved(estimator, params, "risk_aversion", 1.0) or 1.0),
        min_return=_resolved(estimator, params, "min_return"),
        cvar_beta=float(_resolved(estimator, params, "cvar_beta", 0.95) or 0.95),
    )


@dataclass
class TwinProblem:
    problem: cp.Problem
    weights: cp.Variable
    parameters: dict[str, cp.Parameter]
    risk_measure: RiskMeasure
    n_assets: int
    n_observations: int | None
    scale_objective: float
    scale_constraints: float


def _as_1d(value: float | NDArray[np.float64], n: int) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr.reshape(n)


def default_scales(risk_measure: RiskMeasure) -> tuple[float, float]:
    if risk_measure in {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.CVAR,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.CDAR,
        RiskMeasure.ULCER_INDEX,
    }:
        return 1e-1, 1e2
    if risk_measure is RiskMeasure.EVAR:
        return 1.0, 1e-2
    if risk_measure is RiskMeasure.EDAR:
        return 1.0, 1e2
    return 1.0, 1.0


def build_mean_risk_twin(
    n_assets: int,
    *,
    risk_measure: RiskMeasure,
    objective_function: ObjectiveFunction = ObjectiveFunction.MINIMIZE_RISK,
    n_observations: int | None = None,
    min_weights: float | NDArray[np.float64] = 0.0,
    max_weights: float | NDArray[np.float64] = 1.0,
    budget: float = 1.0,
    include_min_return: bool = False,
    include_l1: bool = False,
    scale_objective: float | None = None,
    scale_constraints: float | None = None,
) -> TwinProblem:
    if risk_measure not in {RiskMeasure.VARIANCE, RiskMeasure.CVAR}:
        raise ValueError(f"Unsupported risk_measure {risk_measure}")
    if objective_function not in {
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_UTILITY,
    }:
        raise ValueError(f"Unsupported objective_function {objective_function}")
    if risk_measure is RiskMeasure.CVAR and n_observations is None:
        raise ValueError("CVaR twin requires n_observations")

    if scale_objective is None or scale_constraints is None:
        default_obj, default_con = default_scales(risk_measure)
        if scale_objective is None:
            scale_objective = default_obj
        if scale_constraints is None:
            scale_constraints = default_con

    w = cp.Variable(n_assets)
    parameters: dict[str, cp.Parameter] = {}
    constraints: list[cp.Constraint] = []

    l2 = cp.Parameter(nonneg=True, name="l2_coef")
    l2.value = 0.0
    parameters["l2_coef"] = l2
    regularization = l2 * cp.sum_squares(w)
    if include_l1:
        l1 = cp.Parameter(nonneg=True, name="l1_coef")
        l1.value = 0.0
        parameters["l1_coef"] = l1
        regularization = l1 * cp.norm(w, 1) + regularization

    mu = cp.Parameter(n_assets, name="mu")
    mu.value = np.zeros(n_assets)
    parameters["mu"] = mu
    expected_return = mu @ w

    if risk_measure is RiskMeasure.VARIANCE:
        v = cp.Variable(nonneg=True)
        L = cp.Parameter((n_assets, n_assets), name="L")
        L.value = np.eye(n_assets)
        parameters["L"] = L
        risk = cp.square(v)
        constraints.append(
            cp.SOC(v * scale_constraints, L.T @ w * scale_constraints)
        )
        n_observations = None
    else:
        assert n_observations is not None
        alpha = cp.Variable()
        u = cp.Variable(n_observations, nonneg=True)
        R = cp.Parameter((n_observations, n_assets), name="R")
        R.value = np.zeros((n_observations, n_assets))
        cvar_coef = cp.Parameter(nonneg=True, name="cvar_coef")
        cvar_coef.value = 1.0 / (n_observations * 0.05)
        parameters["R"] = R
        parameters["cvar_coef"] = cvar_coef
        risk = alpha + cvar_coef * cp.sum(u)
        constraints.append(
            R @ w * scale_constraints
            + alpha * scale_constraints
            + u * scale_constraints
            >= 0
        )

    min_w = _as_1d(min_weights, n_assets)
    max_w = _as_1d(max_weights, n_assets)
    constraints.append(w * scale_constraints >= min_w * scale_constraints)
    constraints.append(w * scale_constraints <= max_w * scale_constraints)
    constraints.append(
        cp.sum(w) * scale_constraints == float(budget) * scale_constraints
    )

    if include_min_return:
        min_return = cp.Parameter(name="min_return")
        min_return.value = 0.0
        parameters["min_return"] = min_return
        constraints.append(
            expected_return * scale_constraints >= min_return * scale_constraints
        )

    if objective_function is ObjectiveFunction.MINIMIZE_RISK:
        objective = cp.Minimize(
            risk * scale_objective + regularization * scale_objective
        )
    else:
        risk_aversion = cp.Parameter(nonneg=True, name="risk_aversion")
        risk_aversion.value = 1.0
        parameters["risk_aversion"] = risk_aversion
        objective = cp.Minimize(
            -expected_return * scale_objective
            + risk_aversion * risk * scale_objective
            + regularization * scale_objective
        )

    problem = cp.Problem(objective, constraints)
    if not problem.is_dcp(dpp=True):
        raise ValueError("MeanRisk twin is not DPP-compliant")
    return TwinProblem(
        problem=problem,
        weights=w,
        parameters=parameters,
        risk_measure=risk_measure,
        n_assets=n_assets,
        n_observations=n_observations,
        scale_objective=float(scale_objective),
        scale_constraints=float(scale_constraints),
    )


def bind_twin_values(
    twin: TwinProblem,
    moments: FoldMoments,
    *,
    l1_coef: float = 0.0,
    l2_coef: float = 0.0,
    risk_aversion: float = 1.0,
    min_return: float | None = None,
    cvar_beta: float = 0.95,
) -> None:
    params = twin.parameters
    if "l1_coef" in params:
        params["l1_coef"].value = float(l1_coef)
    params["l2_coef"].value = float(l2_coef)
    params["mu"].value = np.asarray(moments.mu, dtype=float)
    if "L" in params:
        params["L"].value = np.asarray(moments.cholesky, dtype=float)
    if "R" in params:
        returns = np.asarray(moments.returns, dtype=float)
        expected_t = twin.n_observations
        if expected_t is None:
            raise ValueError("CVaR twin is missing n_observations")
        if returns.shape[0] != expected_t:
            raise ValueError(
                f"CVaR returns length {returns.shape[0]} != template T={expected_t}"
            )
        params["R"].value = returns
        params["cvar_coef"].value = 1.0 / (expected_t * (1.0 - float(cvar_beta)))
    if "risk_aversion" in params:
        params["risk_aversion"].value = float(risk_aversion)
    if "min_return" in params and min_return is not None:
        params["min_return"].value = float(min_return)
