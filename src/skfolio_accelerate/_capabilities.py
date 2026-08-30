"""Eligibility gates for amortized ``cross_val_predict`` backends.

Capability checks are the only gate; numerical engines assume they have already
been passed. This module does not import solvers or the predict orchestrator.
"""

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
from skfolio_accelerate.linear_lp import (
    continuation_unhelpful_reason,
    is_highs_lp_risk,
)
from skfolio_accelerate.mean_risk_problem import ParametricMeanRisk
from skfolio_accelerate.moments import is_default_empirical

BackendName = Literal[
    "osqp",
    "highs",
    "clarabel",
    "max-return",
    "cosmo",
    "cvxpy-sequential",
    "closed-form",
    "fit-assemble",
    "sklearn",
    "compact-grid",
]

_SUPPORTED_OBJECTIVES = frozenset(
    {
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_RETURN,
        ObjectiveFunction.MAXIMIZE_UTILITY,
    }
)
_SUPPORTED_RISKS = frozenset(
    {
        RiskMeasure.VARIANCE,
        RiskMeasure.STANDARD_DEVIATION,
        RiskMeasure.SEMI_VARIANCE,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.CVAR,
        RiskMeasure.EVAR,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
        RiskMeasure.EDAR,
    }
)
_COMPACT_NONE_ATTRS = (
    "min_budget",
    "max_budget",
    "max_short",
    "max_long",
    "cardinality",
    "group_cardinalities",
    "threshold_long",
    "threshold_short",
    "previous_weights",
    "target_weights",
    "groups",
    "linear_constraints",
    "left_inequality",
    "right_inequality",
    "add_constraints",
    "add_objective",
    "overwrite_expected_return",
    "efficient_frontier_size",
    "mu_uncertainty_set_estimator",
    "covariance_uncertainty_set_estimator",
    "min_return",
    "max_tracking_error",
    "max_turnover",
    "max_mean_absolute_deviation",
    "max_first_lower_partial_moment",
    "max_variance",
    "max_standard_deviation",
    "max_semi_variance",
    "max_semi_deviation",
    "max_worst_realization",
    "max_cvar",
    "max_evar",
    "max_max_drawdown",
    "max_average_drawdown",
    "max_cdar",
    "max_edar",
    "max_ulcer_index",
    "max_gini_mean_difference",
    "solver_params",
    "scale_objective",
    "scale_constraints",
    "portfolio_params",
    "fallback",
)

_CLOSED_FORM_TYPES = (EqualWeighted, InverseVolatility, Random)


@dataclass(frozen=True, slots=True)
class CallCapabilities:
    """What this ``cross_val_predict`` call is allowed to skip.

    ``compact_reason``, ``sequential_reason``, and ``assemble_reason`` are
    independent. Auto uses compact OSQP/HiGHS/Clarabel when possible, otherwise
    Parameterized MeanRisk reuse, otherwise fit-assemble, otherwise skfolio.
    """

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
        """Engine ``backend="auto"`` would run.

        Parameters
        ----------
        estimator : estimator instance
            Portfolio optimization estimator.

        Returns
        -------
        backend : str
            Compact OSQP/HiGHS/Clarabel, sequential CVXPY, fit-assemble, or sklearn.
        """
        if self.can_compact:
            if type(estimator) in _CLOSED_FORM_TYPES:
                return "closed-form"
            return _compact_backend_name(estimator)
        if self.can_sequential:
            return "cvxpy-sequential"
        if self.can_assemble:
            return "fit-assemble"
        return "sklearn"


def _nonzero(value: Any) -> bool:
    return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 0))


def _compact_backend_name(estimator) -> BackendName:
    """OSQP, HiGHS, Clarabel, max-return, or COSMO for a compact estimator."""
    if type(estimator) in _CLOSED_FORM_TYPES:
        return "closed-form"
    from skfolio_accelerate._cosmo import uses_cosmo_solver

    if uses_cosmo_solver(estimator):
        return "cosmo"
    if estimator.objective_function is ObjectiveFunction.MAXIMIZE_RETURN:
        return "max-return"
    if estimator.risk_measure is RiskMeasure.VARIANCE:
        return "osqp"
    if is_highs_lp_risk(estimator_spec(estimator)):
        return "highs"
    return "clarabel"


def _first_set_attr(obj, attrs: tuple[str, ...]) -> str | None:
    state = obj.__dict__
    for name in attrs:
        if state[name] is not None:
            return name
    return None


def _closed_form_blocked(estimator) -> str | None:
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


def _mean_risk_compact_blocked(estimator: MeanRisk) -> str | None:
    if type(estimator) not in {MeanRisk, ParametricMeanRisk}:
        return "MeanRisk subclasses are not compacted"
    if name := _first_set_attr(estimator, _COMPACT_NONE_ATTRS):
        return f"{name} is not compacted"
    if estimator.budget is None:
        return "an unspecified equality budget is not compacted"
    if estimator.needs_previous_weights:
        return "sequential previous_weights (costs, turnover, or fallback)"
    if estimator.objective_function not in _SUPPORTED_OBJECTIVES:
        return "objective_function is not compacted"
    risk = estimator.risk_measure
    if (
        estimator.objective_function is not ObjectiveFunction.MAXIMIZE_RETURN
        and risk not in _SUPPORTED_RISKS
    ):
        return "risk_measure is not compacted"
    solver_name = str(estimator.solver or "")
    if solver_name.upper() in {"COSMO", "COSMO_RS", "COSMO_RUST"}:
        from skfolio_accelerate._cosmo import (
            cosmo_available,
            cosmo_cv_blocked_reason,
        )

        if not cosmo_available():
            return "COSMO.rs is not installed"
        if reason := cosmo_cv_blocked_reason(risk):
            return reason
        if estimator.objective_function is ObjectiveFunction.MAXIMIZE_RETURN:
            return "COSMO compact path does not cover analytic maximum-return"
        if risk is RiskMeasure.STANDARD_DEVIATION:
            return "COSMO compact path does not cover standard deviation yet"
    else:
        allowed = (
            {"CLARABEL", "OSQP"}
            if risk is RiskMeasure.VARIANCE
            and estimator.objective_function is not ObjectiveFunction.MAXIMIZE_RETURN
            else {"CLARABEL"}
        )
        if estimator.solver not in allowed:
            return f"solver {estimator.solver!r} is not compacted for {risk.name}"
    if _nonzero(estimator.l1_coef):
        return "l1_coef is not compacted"
    if type(estimator.min_weights) is dict or type(estimator.max_weights) is dict:
        return "dict weight bounds are not compacted"
    if type(estimator.min_acceptable_return) is dict:
        return "dict minimum acceptable returns are not compacted"
    if _nonzero(estimator.transaction_costs):
        return "transaction costs are not compacted"
    if _nonzero(estimator.management_fees):
        return "management fees are not compacted"
    if _nonzero(estimator.risk_free_rate):
        return "a non-zero risk-free rate is not compacted"
    if estimator.raise_on_failure is not True:
        return "raise_on_failure=False is not compacted"
    if estimator.save_problem:
        return "saved CVXPY problem state is not compacted"
    if not is_default_empirical(estimator):
        return "custom prior is not compacted"
    return None


def blocked_reason(estimator) -> str | None:
    """Why the compact engine cannot run this estimator, or ``None`` if it can.

    Compact eligibility is intentionally narrow: the engine must reproduce the
    boxed MeanRisk problem skfolio builds with CVXPY. Options that would change
    that problem return a short reason instead of silently approximating.

    See Also
    --------
    compact_blocked_reason, assemble_blocked_reason, classify_call
    """
    match estimator:
        case Pipeline():
            return "pipelines use skfolio cross_val_predict"
        case EqualWeighted() | InverseVolatility() | Random():
            return _closed_form_blocked(estimator)
        case MeanRisk():
            return _mean_risk_compact_blocked(estimator)
        case _:
            return f"estimator {type(estimator).__name__} is not MeanRisk"


def _call_options_blocked(
    *,
    method: str,
    params: dict | None,
    column_indices,
    entry_rebalancing_params: dict | None,
    cv=None,
    n_jobs: int | None | object = ...,
    verb: str,
) -> str | None:
    """Shared call-level gates for compact / assemble / sequential paths."""
    if method != "predict":
        return f"only method='predict' is {verb}"
    if params:
        return "fit params use skfolio cross_val_predict"
    if column_indices is not None:
        return "column_indices uses skfolio cross_val_predict"
    if entry_rebalancing_params is not None:
        return "entry_rebalancing_params uses skfolio cross_val_predict"
    if n_jobs is not ... and n_jobs not in (None, 1):
        return "n_jobs!=1 uses skfolio cross_val_predict"
    try:
        if cv.shuffle is True:
            return "shuffled CV uses skfolio cross_val_predict"
    except AttributeError:
        pass
    return None


def compact_blocked_reason(
    estimator,
    *,
    y=None,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    n_jobs: int | None = None,
    cv=None,
) -> str | None:
    """Why this call cannot use the compact engine."""
    del y  # API compatibility with skfolio
    if reason := _call_options_blocked(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
        n_jobs=n_jobs,
        verb="compacted",
    ):
        return reason
    return continuation_unhelpful_reason(estimator, cv) or blocked_reason(estimator)


def assemble_blocked_reason(
    estimator,
    *,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    n_jobs: int | None = None,
    cv=None,
) -> str | None:
    """Why this call cannot fit natively and assemble from ``weights_``."""
    if reason := _call_options_blocked(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
        n_jobs=n_jobs,
        verb="assembled from weights",
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
            if (
                type(estimator) in {MeanRisk, ParametricMeanRisk}
                and estimator.efficient_frontier_size is not None
            ):
                return "efficient_frontier_size uses skfolio cross_val_predict"
            return continuation_unhelpful_reason(estimator, cv)
        case _:
            return f"estimator {type(estimator).__name__} is not a portfolio optimizer"


def sequential_blocked_reason(
    estimator,
    *,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    n_jobs: int | None = None,
    cv=None,
) -> str | None:
    """Why this call cannot reuse a Parameterized MeanRisk CVXPY problem."""
    reason = assemble_blocked_reason(
        estimator,
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        n_jobs=n_jobs,
        cv=cv,
    )
    if reason is not None:
        return reason
    if type(estimator) not in {MeanRisk, ParametricMeanRisk}:
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    if estimator.objective_function is ObjectiveFunction.MAXIMIZE_RATIO:
        return "MAXIMIZE_RATIO homogenization is not parameterized"
    if estimator.add_constraints is not None:
        return "add_constraints uses fit-assemble"
    if estimator.add_objective is not None:
        return "add_objective uses fit-assemble"
    if estimator.overwrite_expected_return is not None:
        return "overwrite_expected_return uses fit-assemble"
    if estimator.mu_uncertainty_set_estimator is not None:
        return "mu uncertainty sets use fit-assemble"
    if estimator.covariance_uncertainty_set_estimator is not None:
        return "covariance uncertainty sets use fit-assemble"
    if estimator.max_tracking_error is not None:
        return "tracking error is not parameterized"
    if estimator.fallback not in (None, "previous_weights"):
        return "fallback estimator uses skfolio cross_val_predict"
    return continuation_unhelpful_reason(estimator, cv)


def classify_call(
    estimator,
    *,
    y=None,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    n_jobs: int | None = None,
    cv=None,
) -> CallCapabilities:
    """Map estimator and call options to compact / sequential / assemble eligibility."""
    call_kw = dict(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
    )
    return CallCapabilities(
        compact_reason=compact_blocked_reason(estimator, y=y, n_jobs=n_jobs, **call_kw),
        assemble_reason=assemble_blocked_reason(estimator, n_jobs=n_jobs, **call_kw),
        sequential_reason=sequential_blocked_reason(
            estimator, n_jobs=n_jobs, **call_kw
        ),
    )
