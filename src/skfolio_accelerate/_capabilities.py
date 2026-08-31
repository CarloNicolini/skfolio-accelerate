"""Eligibility gates for amortized ``cross_val_predict`` backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from skfolio import RiskMeasure
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    Random,
)
from skfolio.optimization.convex import ObjectiveFunction
from sklearn.pipeline import Pipeline

from skfolio_accelerate.compact import estimator_spec
from skfolio_accelerate.compact._util import SCENARIO_RISKS
from skfolio_accelerate.linear_lp import continuation_unhelpful_reason, is_highs_lp_risk
from skfolio_accelerate.mean_risk_problem import ParametricMeanRisk
from skfolio_accelerate.moments import is_default_empirical

BackendName = Literal[
    "osqp",
    "highs",
    "clarabel",
    "max-return",
    "cvxpy-sequential",
    "closed-form",
    "fit-assemble",
    "sklearn",
    "compact-grid",
]

_OBJECTIVES = frozenset(
    {
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_RETURN,
        ObjectiveFunction.MAXIMIZE_UTILITY,
    }
)
_RISKS = SCENARIO_RISKS | {RiskMeasure.VARIANCE, RiskMeasure.STANDARD_DEVIATION}
_COMPACT_NONE = (
    "min_budget max_budget max_short max_long cardinality group_cardinalities "
    "threshold_long threshold_short previous_weights target_weights groups "
    "linear_constraints left_inequality right_inequality add_constraints "
    "add_objective overwrite_expected_return efficient_frontier_size "
    "mu_uncertainty_set_estimator covariance_uncertainty_set_estimator "
    "min_return max_tracking_error max_turnover max_mean_absolute_deviation "
    "max_first_lower_partial_moment max_variance max_standard_deviation "
    "max_semi_variance max_semi_deviation max_worst_realization max_cvar "
    "max_evar max_max_drawdown max_average_drawdown max_cdar max_edar "
    "max_ulcer_index max_gini_mean_difference solver_params scale_objective "
    "scale_constraints portfolio_params fallback"
).split()
_CLOSED_FORM = (EqualWeighted, InverseVolatility, Random)
_CLOSED_FORM_TYPES = _CLOSED_FORM
_ROUTED = frozenset({"factors"})
_MEAN_RISK = {MeanRisk, ParametricMeanRisk}
_OSQP_LINEAR = frozenset(
    {
        "linear_constraints",
        "groups",
        "left_inequality",
        "right_inequality",
        "min_return",
        "min_budget",
        "max_budget",
    }
)


def _variance_osqp(estimator) -> bool:
    return (
        type(estimator) in _MEAN_RISK
        and estimator.risk_measure is RiskMeasure.VARIANCE
        and estimator.objective_function in _OBJECTIVES
        and estimator.objective_function is not ObjectiveFunction.MAXIMIZE_RETURN
    )


@dataclass(frozen=True, slots=True)
class CallCapabilities:
    compact_reason: str | None
    assemble_reason: str | None
    sequential_reason: str | None = None

    @property
    def can_compact(self) -> bool:
        return self.compact_reason is None

    @property
    def can_sequential(self) -> bool:
        return self.sequential_reason is None

    @property
    def can_assemble(self) -> bool:
        return self.assemble_reason is None

    def auto_backend(self, estimator) -> BackendName:
        if self.can_compact:
            if type(estimator) in _CLOSED_FORM:
                return "closed-form"
            return compact_backend_name(estimator)
        if self.can_sequential:
            return "cvxpy-sequential"
        if self.can_assemble:
            return "fit-assemble"
        return "sklearn"


def compact_backend_name(estimator) -> BackendName:
    if type(estimator) in _CLOSED_FORM:
        return "closed-form"
    if estimator.objective_function is ObjectiveFunction.MAXIMIZE_RETURN:
        return "max-return"
    if estimator.risk_measure is RiskMeasure.VARIANCE:
        return "osqp"
    if is_highs_lp_risk(estimator_spec(estimator)):
        return "highs"
    return "clarabel"


_compact_backend_name = compact_backend_name


def _nonzero(value: Any) -> bool:
    return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 0))


def _call_blocked(
    *,
    method: str,
    params: dict | None,
    column_indices,
    entry_rebalancing_params,
    cv,
    n_jobs,
    verb: str,
    allow_routed: bool,
) -> str | None:
    if method != "predict":
        return f"only method='predict' is {verb}"
    if params:
        if not allow_routed or set(params) - _ROUTED:
            return "fit params use skfolio cross_val_predict"
    if column_indices is not None:
        return "column_indices uses skfolio cross_val_predict"
    if entry_rebalancing_params is not None:
        return "entry_rebalancing_params uses skfolio cross_val_predict"
    if n_jobs is not ... and n_jobs not in (None, 1):
        return "n_jobs!=1 uses skfolio cross_val_predict"
    if getattr(cv, "shuffle", False) is True:
        return "shuffled CV uses skfolio cross_val_predict"
    return None


def blocked_reason(estimator) -> str | None:
    match estimator:
        case Pipeline():
            return "pipelines use skfolio cross_val_predict"
        case EqualWeighted() | InverseVolatility() | Random():
            if estimator.fallback is not None:
                return "fallback estimator is not compacted"
            if estimator.previous_weights is not None:
                return "previous weights are not compacted"
            if estimator.raise_on_failure is not True:
                return "raise_on_failure=False is not compacted"
            if estimator.portfolio_params is not None:
                return "estimator portfolio parameters are not compacted"
            if type(estimator) is InverseVolatility and not is_default_empirical(estimator):
                return "custom prior is not compacted"
            return None
        case MeanRisk():
            if type(estimator) not in _MEAN_RISK:
                return "MeanRisk subclasses are not compacted"
            state = estimator.__dict__
            osqp = _variance_osqp(estimator)
            for name in _COMPACT_NONE:
                if state[name] is None:
                    continue
                if osqp and name in _OSQP_LINEAR:
                    if name in {"min_budget", "max_budget"} and estimator.budget is not None:
                        return f"{name} is not compacted"
                    continue
                return f"{name} is not compacted"
            if estimator.budget is None and not (
                osqp and (estimator.min_budget is not None or estimator.max_budget is not None)
            ):
                return "an unspecified equality budget is not compacted"
            if estimator.needs_previous_weights:
                return "sequential previous_weights (costs, turnover, or fallback)"
            if estimator.objective_function not in _OBJECTIVES:
                return "objective_function is not compacted"
            risk = estimator.risk_measure
            if (
                estimator.objective_function is not ObjectiveFunction.MAXIMIZE_RETURN
                and risk not in _RISKS
            ):
                return "risk_measure is not compacted"
            allowed = (
                {"CLARABEL", "OSQP"}
                if risk is RiskMeasure.VARIANCE
                and estimator.objective_function is not ObjectiveFunction.MAXIMIZE_RETURN
                else {"CLARABEL"}
            )
            if estimator.solver not in allowed:
                return f"solver {estimator.solver!r} is not compacted for {risk.name}"
            if _nonzero(estimator.l1_coef) and not osqp:
                return "l1_coef is not compacted"
            if type(estimator.min_weights) is dict or type(estimator.max_weights) is dict:
                if not osqp:
                    return "dict weight bounds are not compacted"
            if type(estimator.min_acceptable_return) is dict:
                return "dict minimum acceptable returns are not compacted"
            if osqp and estimator.min_return is not None:
                if type(estimator.min_return) is dict or np.ndim(estimator.min_return) > 0:
                    return "min_return is not compacted"
            for attr, msg in (
                ("transaction_costs", "transaction costs are not compacted"),
                ("management_fees", "management fees are not compacted"),
                ("risk_free_rate", "a non-zero risk-free rate is not compacted"),
            ):
                if _nonzero(getattr(estimator, attr)):
                    return msg
            if estimator.raise_on_failure is not True:
                return "raise_on_failure=False is not compacted"
            if estimator.save_problem:
                return "saved CVXPY problem state is not compacted"
            if not is_default_empirical(estimator):
                return "custom prior is not compacted"
            return None
        case _:
            return f"estimator {type(estimator).__name__} is not MeanRisk"


def compact_blocked_reason(estimator, *, y=None, method="predict", params=None, column_indices=None, entry_rebalancing_params=None, n_jobs=None, cv=None):
    del y
    return _call_blocked(
        method=method, params=params, column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params, cv=cv, n_jobs=n_jobs,
        verb="compacted", allow_routed=False,
    ) or continuation_unhelpful_reason(estimator, cv) or blocked_reason(estimator)


def assemble_blocked_reason(estimator, *, method="predict", params=None, column_indices=None, entry_rebalancing_params=None, n_jobs=None, cv=None):
    if reason := _call_blocked(
        method=method, params=params, column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params, cv=cv, n_jobs=n_jobs,
        verb="assembled from weights", allow_routed=True,
    ):
        return reason
    match estimator:
        case Pipeline():
            return "pipelines use skfolio cross_val_predict"
        case BaseOptimization():
            if estimator.needs_previous_weights:
                return "sequential previous_weights (costs, turnover, or fallback)"
            if estimator.raise_on_failure is not True:
                return "raise_on_failure=False uses skfolio cross_val_predict"
            if type(estimator) in _MEAN_RISK and estimator.efficient_frontier_size is not None:
                return "efficient_frontier_size uses skfolio cross_val_predict"
            return continuation_unhelpful_reason(estimator, cv)
        case _:
            return f"estimator {type(estimator).__name__} is not a portfolio optimizer"


def sequential_blocked_reason(estimator, *, method="predict", params=None, column_indices=None, entry_rebalancing_params=None, n_jobs=None, cv=None):
    reason = assemble_blocked_reason(
        estimator, method=method, params=params, column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params, n_jobs=n_jobs, cv=cv,
    )
    if reason:
        return reason
    if type(estimator) not in _MEAN_RISK:
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    if estimator.objective_function is ObjectiveFunction.MAXIMIZE_RATIO:
        return "MAXIMIZE_RATIO homogenization is not parameterized"
    for attr, msg in (
        ("add_constraints", "add_constraints uses fit-assemble"),
        ("add_objective", "add_objective uses fit-assemble"),
        ("overwrite_expected_return", "overwrite_expected_return uses fit-assemble"),
        ("mu_uncertainty_set_estimator", "mu uncertainty sets use fit-assemble"),
        ("covariance_uncertainty_set_estimator", "covariance uncertainty sets use fit-assemble"),
        ("max_tracking_error", "tracking error is not parameterized"),
    ):
        if getattr(estimator, attr) is not None:
            return msg
    if estimator.fallback not in (None, "previous_weights"):
        return "fallback estimator uses skfolio cross_val_predict"
    return continuation_unhelpful_reason(estimator, cv)


def classify_call(estimator, *, y=None, method="predict", params=None, column_indices=None, entry_rebalancing_params=None, n_jobs=None, cv=None):
    kw = dict(method=method, params=params, column_indices=column_indices, entry_rebalancing_params=entry_rebalancing_params, cv=cv)
    return CallCapabilities(
        compact_reason=compact_blocked_reason(estimator, y=y, n_jobs=n_jobs, **kw),
        assemble_reason=assemble_blocked_reason(estimator, n_jobs=n_jobs, **kw),
        sequential_reason=sequential_blocked_reason(estimator, n_jobs=n_jobs, **kw),
    )
