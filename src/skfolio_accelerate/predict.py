"""Drop-in for ``skfolio.model_selection.cross_val_predict``.

A call is classified once, then executed as:

    CV definition → compiled :class:`~skfolio_accelerate.cv_plan.CVPlan`
    → backend (compact / sequential CVXPY / closed-form / fit-assemble / native)
    → fold weights → assembled Portfolio objects

Compact OSQP/Clarabel kernels accelerate a subset of MeanRisk. EqualWeighted,
Random, and default InverseVolatility use closed-form weights. Remaining
serial estimators still call native ``fit``, then assemble test portfolios
from ``weights_`` so they skip joblib, train/test copies, and ``predict()``.

The accelerator never reinterprets an unsupported estimator as a nearby
compact problem. Capability checks are the only gate; numerical engines assume
they have already been passed.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    Random,
)
from skfolio.optimization.convex import ObjectiveFunction
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio
from skfolio.utils.stats import rand_weights_dirichlet
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from skfolio_accelerate._arrays import as_float_2d, as_float_array
from skfolio_accelerate.compact import EngineCache, MeanRiskSpec, estimator_spec
from skfolio_accelerate.cv_plan import (
    CVPlan,
    FoldSpec,
    chains_previous_weights,
    compile_cv_plan,
)
from skfolio_accelerate.mean_risk_problem import (
    ParametricMeanRisk,
    SequentialProblemCache,
)
from skfolio_accelerate.moments import (
    PathMomentSession,
    is_default_empirical,
    path_moment_session,
)
from skfolio_accelerate.scoring import assemble_prediction, window_view

BackendName = Literal[
    "osqp",
    "clarabel",
    "cvxpy-sequential",
    "closed-form",
    "fit-assemble",
    "sklearn",
    "compact-grid",
    "sequential-grid",
]

_SUPPORTED_OBJECTIVES = frozenset(
    {
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_UTILITY,
    }
)
_SUPPORTED_RISKS = frozenset(
    {
        RiskMeasure.VARIANCE,
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
_UNSUPPORTED_IF_SET = (
    ("min_budget", "minimum budget"),
    ("max_budget", "maximum budget"),
    ("max_short", "maximum short exposure"),
    ("max_long", "maximum long exposure"),
    ("cardinality", "cardinality (MIP)"),
    ("group_cardinalities", "group_cardinalities (MIP)"),
    ("threshold_long", "threshold_long (MIP)"),
    ("threshold_short", "threshold_short (MIP)"),
    ("previous_weights", "previous weights"),
    ("target_weights", "target weights"),
    ("groups", "groups"),
    ("linear_constraints", "linear constraints"),
    ("left_inequality", "left inequality"),
    ("right_inequality", "right inequality"),
    ("add_constraints", "add_constraints"),
    ("add_objective", "add_objective"),
    ("overwrite_expected_return", "overwrite_expected_return"),
    ("efficient_frontier_size", "efficient_frontier_size"),
    ("mu_uncertainty_set_estimator", "mu uncertainty sets"),
    ("covariance_uncertainty_set_estimator", "covariance uncertainty sets"),
    ("min_return", "minimum return"),
    ("max_tracking_error", "maximum tracking error"),
    ("max_turnover", "maximum turnover"),
    ("max_mean_absolute_deviation", "maximum mean absolute deviation"),
    ("max_first_lower_partial_moment", "maximum first lower partial moment"),
    ("max_variance", "maximum variance"),
    ("max_standard_deviation", "maximum standard deviation"),
    ("max_semi_variance", "maximum semi-variance"),
    ("max_semi_deviation", "maximum semi-deviation"),
    ("max_worst_realization", "maximum worst realization"),
    ("max_cvar", "maximum CVaR"),
    ("max_evar", "maximum EVaR"),
    ("max_max_drawdown", "maximum drawdown"),
    ("max_average_drawdown", "maximum average drawdown"),
    ("max_cdar", "maximum CDaR"),
    ("max_edar", "maximum EDaR"),
    ("max_ulcer_index", "maximum ulcer index"),
    ("max_gini_mean_difference", "maximum Gini mean difference"),
    ("solver_params", "custom solver parameters"),
    ("scale_objective", "custom objective scaling"),
    ("scale_constraints", "custom constraint scaling"),
    ("portfolio_params", "estimator portfolio parameters"),
    ("fallback", "fallback estimator"),
)

_CLOSED_FORM_TYPES = (EqualWeighted, InverseVolatility, Random)
_PORTFOLIO_ATTRS = (
    "transaction_costs",
    "management_fees",
    "previous_weights",
    "risk_free_rate",
)


@dataclass
class AccelerationReport:
    """Diagnostics for one :func:`cross_val_predict` or :func:`grid_search` call.

    Returned when ``return_report=True``. The ``backend`` field identifies which
    execution path ran; ``fallback_reason`` explains native skfolio or
    fit-assemble fallbacks.

    Attributes
    ----------
    backend : str
        Selected execution backend. One of ``"osqp"``, ``"clarabel"``,
        ``"cvxpy-sequential"``, ``"closed-form"``, ``"fit-assemble"``,
        ``"sklearn"``, ``"compact-grid"``, or ``"sequential-grid"``.

        * ``"osqp"`` — compact mean-variance QP.
        * ``"clarabel"`` — compact scenario LP / QP / SOCP / exponential cone.
        * ``"cvxpy-sequential"`` — skfolio MeanRisk CVXPY graph reused across
          folds via Parameters (full constraint set).
        * ``"closed-form"`` — EqualWeighted, Random, or InverseVolatility.
        * ``"fit-assemble"`` — native ``fit`` with portfolio assembly from
          ``weights_``.
        * ``"sklearn"`` — unmodified skfolio ``cross_val_predict``.
        * ``"compact-grid"`` — shared compact path inside :func:`grid_search`.
        * ``"sequential-grid"`` — Parameterized MeanRisk grid inside
          :func:`grid_search`.

    n_solves : int, default=0
        Number of fold solves (compact, closed-form, sequential, or native
        ``fit``).

    n_prior_fits : int, default=0
        Cold Gram / sufficient-statistic computations in the moment cache.

    n_prior_updates : int, default=0
        Rank-k sliding-window or CPCV block updates that avoided a full
        ``X.T @ X``.

    n_warm_starts : int, default=0
        Successful OSQP / Clarabel / CVXPY warm starts across folds.

    n_rebuilds : int, default=0
        CVXPY graphs compiled by the sequential MeanRisk adapter.

    is_dpp : bool or None, default=None
        Whether the last compiled sequential problem reported as DPP.

    reason : str or None, default=None
        Why ``backend`` was selected. Under ``backend="auto"`` this is the
        policy decision (for example boxed variance uses OSQP; a ratio
        objective uses Parameterized CVXPY because it is outside the compact
        subset).

    fallback_reason : str or None, default=None
        Human-readable reason when a preferred engine failed and the call
        retried on fit-assemble or skfolio, or ``None`` when no fallback
        occurred.

    moments_s, solve_s, eval_s, wall_s : float, default=0.0
        Wall-clock seconds spent on moments, solves, portfolio assembly, and
        the full call respectively.

    baseline_s : float, default=0.0
        Optional native baseline time filled by benchmark helpers.

    speedup : float, default=nan
        ``baseline_s / wall_s`` when a baseline was recorded.

    Examples
    --------
    >>> pred, report = cross_val_predict(
    ...     MeanRisk(), X, cv=cv, return_report=True
    ... )  # doctest: +SKIP
    >>> print(report.backend)  # doctest: +SKIP
    osqp
    """

    backend: BackendName | str
    n_solves: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_warm_starts: int = 0
    n_rebuilds: int = 0
    is_dpp: bool | None = None
    reason: str | None = None
    fallback_reason: str | None = None
    moments_s: float = 0.0
    solve_s: float = 0.0
    eval_s: float = 0.0
    wall_s: float = 0.0
    baseline_s: float = 0.0
    speedup: float = float("nan")

    def __str__(self) -> str:
        lines = [
            f"Backend: {self.backend}",
            f"Reason: {self.reason or 'none'}",
            f"Solves: {self.n_solves}",
            f"Moment fits: {self.n_prior_fits}",
            f"Moment updates: {self.n_prior_updates}",
            f"Warm starts: {self.n_warm_starts}",
            f"Rebuilds: {self.n_rebuilds}",
            f"moments {self.moments_s:.4f}s  solve {self.solve_s:.4f}s  "
            f"eval {self.eval_s:.4f}s  wall {self.wall_s:.4f}s",
            f"DPP: {self.is_dpp}",
            f"Fallback: {self.fallback_reason or 'none'}",
        ]
        if self.baseline_s > 0:
            lines.append(
                f"Baseline {self.baseline_s:.4f}s  speedup {self.speedup:.2f}×"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CallCapabilities:
    """What this ``cross_val_predict`` call is allowed to skip.

    ``compact_reason``, ``sequential_reason``, and ``assemble_reason`` are
    independent. A MeanRisk configuration may be ineligible for the cone
    engines yet still eligible for Parameterized CVXPY reuse or for serial
    native ``fit`` plus weight assembly.
    """

    compact_reason: str | None
    sequential_reason: str | None
    assemble_reason: str | None

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
        """Engine ``backend="auto"`` would run for this estimator.

        Order: compact OSQP / Clarabel (or closed-form), then Parameterized
        MeanRisk CVXPY, then fit-assemble, then unmodified skfolio.

        Parameters
        ----------
        estimator : estimator instance
            Portfolio optimization estimator.

        Returns
        -------
        backend : str
            Concrete engine name stored on :class:`AccelerationReport`.
        """
        if self.can_compact:
            return compact_engine_name(estimator)
        if self.can_sequential:
            return "cvxpy-sequential"
        if self.can_assemble:
            return "fit-assemble"
        return "sklearn"

    def choice_reason(self, backend: str, *, fallback: str | None = None) -> str:
        """Plain-language explanation of ``backend`` for :class:`AccelerationReport`.

        Parameters
        ----------
        backend : str
            Engine name stored on the report.

        fallback : str, optional
            Override used when a preferred engine failed and the call retried.

        Returns
        -------
        reason : str
            Policy decision, or ``fallback`` when a retry overrode the choice.
        """
        if fallback is not None:
            return fallback
        if backend == "osqp":
            return "boxed MeanRisk variance; compact OSQP"
        if backend == "clarabel":
            return "boxed MeanRisk scenario risk; compact Clarabel"
        if backend == "closed-form":
            return "closed-form weights; no solver"
        if backend == "cvxpy-sequential":
            if self.compact_reason:
                return f"MeanRisk outside the compact subset ({self.compact_reason})"
            return "Parameterized MeanRisk CVXPY reuse"
        if backend == "compact-grid":
            return "boxed MeanRisk grid; shared compact engines"
        if backend == "sequential-grid":
            extra = f" ({self.compact_reason})" if self.compact_reason else ""
            return "MeanRisk grid outside the compact subset" + extra
        if backend == "fit-assemble":
            return (
                self.sequential_reason
                or self.compact_reason
                or "native fit; assemble from weights_"
            )
        if backend == "sklearn":
            return (
                self.assemble_reason
                or self.sequential_reason
                or self.compact_reason
                or "unmodified skfolio"
            )
        return backend


def compact_engine_name(estimator) -> BackendName:
    """OSQP, Clarabel, or closed-form name for a compact-eligible estimator.

    Parameters
    ----------
    estimator : estimator instance
        Closed-form type or boxed :class:`~skfolio.optimization.MeanRisk`.

    Returns
    -------
    name : {"osqp", "clarabel", "closed-form"}
        Engine name ``backend="auto"`` would record on
        :class:`AccelerationReport`.
    """
    if type(estimator) in _CLOSED_FORM_TYPES:
        return "closed-form"
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    if risk is RiskMeasure.VARIANCE:
        return "osqp"
    return "clarabel"


def resolve_backend(
    requested: str,
    capabilities: CallCapabilities,
    estimator,
) -> BackendName:
    """Map ``backend="auto"|...`` plus eligibility to one execution engine.

    ``auto`` is the user-facing policy: compact OSQP / Clarabel when the
    boxed MeanRisk (or closed-form) problem applies, Parameterized CVXPY
    reuse for other MeanRisk configurations, then fit-assemble, then
    unmodified skfolio. Explicit ``compact`` / ``cvxpy-sequential`` /
    ``sklearn`` override that order.

    Parameters
    ----------
    requested : {"auto", "compact", "cvxpy-sequential", "sklearn"}
        Value of the ``backend`` argument.

    capabilities : CallCapabilities
        Output of :func:`classify_call`.

    estimator : estimator instance
        Used to distinguish OSQP, Clarabel, and closed-form when compact.

    Returns
    -------
    backend : str
        Concrete engine name stored on :class:`AccelerationReport`.

    Raises
    ------
    ValueError
        If ``requested`` is unknown or an override is ineligible.
    """
    if requested not in {"auto", "compact", "cvxpy-sequential", "sklearn"}:
        raise ValueError(f"Unknown backend {requested!r}")
    if requested == "sklearn":
        return "sklearn"
    if requested == "compact":
        if not capabilities.can_compact:
            raise ValueError(
                f"backend={requested!r} cannot compact this predict: "
                f"{capabilities.compact_reason}"
            )
        return compact_engine_name(estimator)
    if requested == "cvxpy-sequential":
        if not capabilities.can_sequential:
            raise ValueError(
                f"backend={requested!r} cannot reuse this MeanRisk problem: "
                f"{capabilities.sequential_reason}"
            )
        return "cvxpy-sequential"
    return capabilities.auto_backend(estimator)


@dataclass(slots=True)
class FoldBatchResult:
    """Weights and accounting for one path batch (or the whole plan).

    Attributes
    ----------
    weights : dict[int, ndarray of shape (n_assets,)]
        Mapping from ``fold_id`` to portfolio weights.

    moments_s, solve_s : float, default=0.0
        Seconds spent on moments and solves.

    n_solves, n_warm_starts, n_prior_fits, n_prior_updates : int, default=0
        Accounting counters aggregated into :class:`AccelerationReport`.

    n_rebuilds : int, default=0
        Sequential CVXPY graphs compiled in this batch.

    is_dpp : bool or None, default=None
        DPP flag reported by the sequential adapter, when available.

    previous_weights : dict[int, ndarray or None]
        ``previous_weights`` used when solving each ``fold_id``, for portfolio
        assembly that charges transaction costs.
    """

    weights: dict[int, NDArray[np.float64]]
    moments_s: float = 0.0
    solve_s: float = 0.0
    n_solves: int = 0
    n_warm_starts: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_rebuilds: int = 0
    is_dpp: bool | None = None
    previous_weights: dict[int, Any] = field(default_factory=dict)


def _cap_native_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _nonzero(value: Any) -> bool:
    return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 0))


def blocked_reason(estimator) -> str | None:
    """Why the compact engine cannot run this estimator, or ``None`` if it can.

    Compact eligibility is intentionally narrow. The engine must reproduce the
    same boxed MeanRisk problem that skfolio builds with CVXPY. Any option that
    would change that problem (MIP constraints, custom priors, risk limits,
    sequential ``previous_weights``, pipelines, ...) returns a short reason
    string instead of silently approximating.

    Parameters
    ----------
    estimator : estimator instance
        Portfolio optimization estimator to inspect. Closed-form types
        (:class:`~skfolio.optimization.EqualWeighted`,
        :class:`~skfolio.optimization.Random`,
        :class:`~skfolio.optimization.InverseVolatility`) are accepted when
        they use default settings. Otherwise the estimator must be
        :class:`~skfolio.optimization.MeanRisk`.

    Returns
    -------
    reason : str or None
        ``None`` when the estimator is eligible for the compact OSQP / Clarabel
        path. Otherwise a short English phrase describing the blocking option.

    Notes
    -----
    This function only inspects the estimator object. Call-level options such
    as ``n_jobs``, ``method``, or shuffled CV are checked by
    :func:`compact_blocked_reason`.

    See Also
    --------
    compact_blocked_reason
    assemble_blocked_reason
    classify_call
    """
    if isinstance(estimator, Pipeline):
        return "pipelines use skfolio cross_val_predict"
    if type(estimator) in _CLOSED_FORM_TYPES:
        if getattr(estimator, "fallback", None) is not None:
            return "fallback estimator is not compacted"
        if getattr(estimator, "previous_weights", None) is not None:
            return "previous weights are not compacted"
        if getattr(estimator, "raise_on_failure", True) is not True:
            return "raise_on_failure=False is not compacted"
        if getattr(estimator, "portfolio_params", None) is not None:
            return "estimator portfolio parameters are not compacted"
        if type(estimator) is InverseVolatility and not is_default_empirical(estimator):
            return "custom prior is not compacted"
        return None
    if not isinstance(estimator, MeanRisk):
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    for attr, label in _UNSUPPORTED_IF_SET:
        if getattr(estimator, attr, None) is not None:
            return f"{label} is not compacted"
    if getattr(estimator, "budget", 1.0) is None:
        return "an unspecified equality budget is not compacted"
    if getattr(estimator, "needs_previous_weights", False):
        return "sequential previous_weights (costs, turnover, or fallback)"
    objective = getattr(
        estimator, "objective_function", ObjectiveFunction.MINIMIZE_RISK
    )
    if objective not in _SUPPORTED_OBJECTIVES:
        return "objective_function is not compacted"
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    if risk not in _SUPPORTED_RISKS:
        return "risk_measure is not compacted"
    solver = getattr(estimator, "solver", "CLARABEL")
    if risk is RiskMeasure.VARIANCE:
        if solver not in {"CLARABEL", "OSQP"}:
            return f"solver {solver!r} is not compacted for {risk.name}"
    elif solver != "CLARABEL":
        return f"solver {solver!r} is not compacted for {risk.name}"
    if _nonzero(getattr(estimator, "l1_coef", 0.0)):
        return "l1_coef is not compacted"
    if isinstance(getattr(estimator, "min_weights", 0.0), dict) or isinstance(
        getattr(estimator, "max_weights", 1.0), dict
    ):
        return "dict weight bounds are not compacted"
    if isinstance(getattr(estimator, "min_acceptable_return", None), dict):
        return "dict minimum acceptable returns are not compacted"
    if _nonzero(getattr(estimator, "transaction_costs", 0.0)):
        return "transaction costs are not compacted"
    if _nonzero(getattr(estimator, "management_fees", 0.0)):
        return "management fees are not compacted"
    if _nonzero(getattr(estimator, "risk_free_rate", 0.0)):
        return "a non-zero risk-free rate is not compacted"
    if getattr(estimator, "raise_on_failure", True) is not True:
        return "raise_on_failure=False is not compacted"
    if getattr(estimator, "save_problem", False):
        return "saved CVXPY problem state is not compacted"
    if not is_default_empirical(estimator):
        return "custom prior is not compacted"
    return None


def _call_options_blocked(
    *,
    method: str,
    params: dict | None,
    column_indices,
    entry_rebalancing_params: dict | None,
    cv=None,
    n_jobs: int | None = None,
    require_serial: bool = False,
    method_label: str,
) -> str | None:
    """Shared call-level gates for compact, sequential, and assemble."""
    if method != "predict":
        return f"only method='predict' is {method_label}"
    if params:
        return "fit params use skfolio cross_val_predict"
    if column_indices is not None:
        return "column_indices uses skfolio cross_val_predict"
    if entry_rebalancing_params is not None:
        return "entry_rebalancing_params uses skfolio cross_val_predict"
    if require_serial and n_jobs not in (None, 1):
        return "n_jobs!=1 uses skfolio cross_val_predict"
    if getattr(cv, "shuffle", False) is True:
        return "shuffled CV uses skfolio cross_val_predict"
    return None


def compact_blocked_reason(
    estimator,
    *,
    y=None,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    cv=None,
) -> str | None:
    """Why this call cannot use the compact engine.

    Combines estimator checks from :func:`blocked_reason` with call-level
    options that force the native skfolio path.

    Parameters
    ----------
    estimator : estimator instance
        Portfolio optimization estimator.

    y : array-like of shape (n_observations,) or (n_observations, n_assets), optional
        Target passed through for API compatibility. Unused by the compact gate.

    method : str, default="predict"
        Prediction method. Only ``"predict"`` is compacted.

    params : dict, optional
        Extra ``fit`` parameters. Any non-empty mapping disables compaction.

    column_indices : array-like, optional
        Per-split asset subsets outside MultipleRandomizedCV. Not compacted.

    entry_rebalancing_params : dict, optional
        Entry-rebalancing metadata. Not compacted.

    cv : int, cross-validation generator or an iterable, optional
        Splitter. Shuffled CV (``shuffle=True``) is not compacted.

    Returns
    -------
    reason : str or None
        ``None`` when compact OSQP / Clarabel may run; otherwise a short reason.

    See Also
    --------
    blocked_reason
    assemble_blocked_reason
    classify_call
    """
    blocked = _call_options_blocked(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
        method_label="compacted",
    )
    if blocked:
        return blocked
    return blocked_reason(estimator)


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
    """Why this call cannot fit natively and assemble from ``weights_``.

    The fit-assemble path still calls the estimator's native ``fit``, then builds
    test portfolios from ``weights_``. It skips joblib, train/test copies, and
    ``predict()`` construction for serial ``n_jobs`` in ``{None, 1}``.

    Parameters
    ----------
    estimator : estimator instance
        Must be a :class:`~skfolio.optimization.BaseOptimization` (not a
        :class:`~sklearn.pipeline.Pipeline`).

    method : str, default="predict"
        Only ``"predict"`` is assembled from weights.

    params : dict, optional
        Extra ``fit`` parameters. Any non-empty mapping disables assembly.

    column_indices : array-like, optional
        External column routing. Not assembled.

    entry_rebalancing_params : dict, optional
        Entry-rebalancing metadata. Not assembled.

    n_jobs : int or None, default=None
        Parallelism. Values other than ``None`` or ``1`` use native skfolio.

    cv : int, cross-validation generator or an iterable, optional
        Splitter. Shuffled CV is not assembled.

    Returns
    -------
    reason : str or None
        ``None`` when fit-assemble is allowed; otherwise a short reason.

    See Also
    --------
    compact_blocked_reason
    classify_call
    """
    blocked = _call_options_blocked(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
        n_jobs=n_jobs,
        require_serial=True,
        method_label="assembled from weights",
    )
    if blocked:
        return blocked
    if isinstance(estimator, Pipeline):
        return "pipelines use skfolio cross_val_predict"
    if not isinstance(estimator, BaseOptimization):
        return f"estimator {type(estimator).__name__} is not a portfolio optimizer"
    if getattr(estimator, "needs_previous_weights", False):
        return "sequential previous_weights (costs, turnover, or fallback)"
    if getattr(estimator, "raise_on_failure", True) is not True:
        return "raise_on_failure=False uses skfolio cross_val_predict"
    if getattr(estimator, "efficient_frontier_size", None) is not None:
        return "efficient_frontier_size uses skfolio cross_val_predict"
    return None


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
    """Why this call cannot reuse a Parameterized MeanRisk CVXPY problem.

    Sequential reuse keeps skfolio's full constraint set. It is blocked for
    pipelines, MeanRisk subclasses, non-MeanRisk estimators, custom CVXPY
    hooks that close over a window, fallback estimators other than
    ``"previous_weights"``, and the same call-level options that force native
    skfolio (parallel ``n_jobs``, shuffled CV, ``efficient_frontier_size``,
    ...).

    Transaction costs, turnover, and ``previous_weights`` are allowed: they
    are economic state updated as Parameters, not a reason to leave the
    sequential path.

    Parameters
    ----------
    estimator : estimator instance
        Portfolio optimization estimator.

    method : str, default="predict"
        Only ``"predict"`` is sequential.

    params : dict, optional
        Extra ``fit`` parameters. Any non-empty mapping disables sequential
        reuse.

    column_indices : array-like, optional
        External column routing. Not sequential.

    entry_rebalancing_params : dict, optional
        Entry-rebalancing metadata. Not sequential.

    n_jobs : int or None, default=None
        Parallelism. Values other than ``None`` or ``1`` use native skfolio.

    cv : int, cross-validation generator or an iterable, optional
        Splitter. Shuffled CV is not sequential.

    Returns
    -------
    reason : str or None
        ``None`` when Parameterized MeanRisk reuse may run; otherwise a short
        reason.
    """
    blocked = _call_options_blocked(
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        cv=cv,
        n_jobs=n_jobs,
        require_serial=True,
        method_label="sequential",
    )
    if blocked:
        return blocked
    if isinstance(estimator, Pipeline):
        return "pipelines use skfolio cross_val_predict"
    if type(estimator) not in {MeanRisk, ParametricMeanRisk}:
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    if getattr(estimator, "raise_on_failure", True) is not True:
        return "raise_on_failure=False uses skfolio cross_val_predict"
    if getattr(estimator, "efficient_frontier_size", None) is not None:
        return "efficient_frontier_size uses skfolio cross_val_predict"
    if getattr(estimator, "add_constraints", None) is not None:
        return "add_constraints uses fit-assemble"
    if getattr(estimator, "add_objective", None) is not None:
        return "add_objective uses fit-assemble"
    if getattr(estimator, "overwrite_expected_return", None) is not None:
        return "overwrite_expected_return uses fit-assemble"
    if getattr(estimator, "mu_uncertainty_set_estimator", None) is not None:
        return "mu uncertainty sets use fit-assemble"
    if getattr(estimator, "covariance_uncertainty_set_estimator", None) is not None:
        return "covariance uncertainty sets use fit-assemble"
    fallback = getattr(estimator, "fallback", None)
    if fallback not in (None, "previous_weights"):
        return "fallback estimator uses skfolio cross_val_predict"
    return None


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
    """Map estimator and call options to compact / sequential / assemble
    eligibility.

    The three gates are independent. A MeanRisk configuration may be ineligible
    for the cone engines yet still eligible for Parameterized CVXPY reuse or
    for serial native ``fit`` plus weight assembly.

    Parameters
    ----------
    estimator : estimator instance
        Portfolio optimization estimator.

    y, method, params, column_indices, entry_rebalancing_params, n_jobs, cv
        Same meaning as in :func:`cross_val_predict`.

    Returns
    -------
    capabilities : CallCapabilities
        Independent compact, sequential, and assemble reasons.

    See Also
    --------
    resolve_backend : Apply ``backend="auto"`` or an override to capabilities.

    Examples
    --------
    >>> caps = classify_call(MeanRisk(), cv=cv)  # doctest: +SKIP
    >>> caps.auto_backend(MeanRisk())  # doctest: +SKIP
    'osqp'
    """
    return CallCapabilities(
        compact_reason=compact_blocked_reason(
            estimator,
            y=y,
            method=method,
            params=params,
            column_indices=column_indices,
            entry_rebalancing_params=entry_rebalancing_params,
            cv=cv,
        ),
        sequential_reason=sequential_blocked_reason(
            estimator,
            method=method,
            params=params,
            column_indices=column_indices,
            entry_rebalancing_params=entry_rebalancing_params,
            n_jobs=n_jobs,
            cv=cv,
        ),
        assemble_reason=assemble_blocked_reason(
            estimator,
            method=method,
            params=params,
            column_indices=column_indices,
            entry_rebalancing_params=entry_rebalancing_params,
            n_jobs=n_jobs,
            cv=cv,
        ),
    )


def merge_batch_results(parts: Sequence[FoldBatchResult]) -> FoldBatchResult:
    """Merge per-path :class:`FoldBatchResult` objects into one aggregate.

    Parameters
    ----------
    parts : sequence of FoldBatchResult
        Results from each MRC path batch (or a single batch for other CV kinds).

    Returns
    -------
    merged : FoldBatchResult
        Combined weight map and summed timing / accounting fields.
    """
    weights: dict[int, NDArray[np.float64]] = {}
    previous_weights: dict[int, Any] = {}
    moments_s = solve_s = 0.0
    n_solves = n_warm = n_fits = n_updates = n_rebuilds = 0
    is_dpp: bool | None = None
    for part in parts:
        weights.update(part.weights)
        previous_weights.update(part.previous_weights)
        moments_s += part.moments_s
        solve_s += part.solve_s
        n_solves += part.n_solves
        n_warm += part.n_warm_starts
        n_fits += part.n_prior_fits
        n_updates += part.n_prior_updates
        n_rebuilds += part.n_rebuilds
        if part.is_dpp is not None:
            is_dpp = part.is_dpp if is_dpp is None else bool(is_dpp and part.is_dpp)
    return FoldBatchResult(
        weights=weights,
        moments_s=moments_s,
        solve_s=solve_s,
        n_solves=n_solves,
        n_warm_starts=n_warm,
        n_prior_fits=n_fits,
        n_prior_updates=n_updates,
        n_rebuilds=n_rebuilds,
        is_dpp=is_dpp,
        previous_weights=previous_weights,
    )


def solve_compact_folds(
    session: PathMomentSession,
    folds: Sequence[FoldSpec],
    spec: MeanRiskSpec,
    *,
    engines: EngineCache | None = None,
) -> FoldBatchResult:
    """Solve one path batch with a shared moment cache and compact engine.

    Parameters
    ----------
    session : PathMomentSession
        Moment cache bound to one MRC asset subset or the full universe.

    folds : sequence of FoldSpec
        Compiled folds for this path batch.

    spec : MeanRiskSpec
        Numeric MeanRisk configuration already validated by
        :func:`blocked_reason`.

    engines : EngineCache, optional
        Reused solver cache. When ``None``, a fresh cache is created for this
        batch.

    Returns
    -------
    result : FoldBatchResult
        Fold weights keyed by ``fold_id`` plus timing and warm-start counts.

    Notes
    -----
    The first fold of a batch is solved cold (``warm=False``). Subsequent folds
    reuse the OSQP / Clarabel workspace when the topology is unchanged.
    """
    engines = EngineCache(spec=spec) if engines is None else engines
    warm_before = int(getattr(engines.engine, "n_warm_starts", 0))
    weights: dict[int, NDArray[np.float64]] = {}
    moments_s = 0.0
    solve_s = 0.0
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        moments = session.get(fold)
        moments_s += time.perf_counter() - t0
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if spec.needs_returns() else None,
        )
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
    n_warm = int(getattr(engines.engine, "n_warm_starts", 0)) - warm_before
    return FoldBatchResult(
        weights=weights,
        moments_s=moments_s,
        solve_s=solve_s,
        n_solves=len(folds),
        n_warm_starts=n_warm,
        n_prior_fits=int(session.cache.n_fits),
        n_prior_updates=int(session.cache.n_updates),
    )


def _train_slice(X, x_arr: np.ndarray, fold: FoldSpec):
    """Training window, preserving DataFrame columns when ``X`` has ``iloc``."""
    if hasattr(X, "iloc"):
        rows = X.iloc[np.asarray(fold.train_idx)]
        if fold.asset_idx is not None:
            return rows.iloc[:, np.asarray(fold.asset_idx)]
        return rows
    return window_view(x_arr, fold.train_idx, fold.asset_idx)


def solve_sequential_folds(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: Sequence[FoldSpec],
    *,
    cache: SequentialProblemCache | None = None,
    path_id: int = 0,
    chain_previous_weights: bool = False,
) -> FoldBatchResult:
    """Solve one path batch with a Parameterized MeanRisk CVXPY problem.

    The first fold compiles skfolio's full constraint graph. Later folds with
    the same topology update ``cp.Parameter`` values and warm-start.
    """
    cache = SequentialProblemCache(estimator) if cache is None else cache
    adapter = cache.get(path_id)
    warm_before = adapter.n_warm_starts
    rebuild_before = adapter.n_rebuilds
    weights: dict[int, NDArray[np.float64]] = {}
    previous_used: dict[int, Any] = {}
    solve_s = 0.0
    previous = getattr(adapter, "previous_weights", None)
    chain = chain_previous_weights and bool(
        getattr(adapter, "needs_previous_weights", False)
    )
    n_assets = int(x_arr.shape[1])
    for fold in folds:
        previous_used[fold.fold_id] = previous
        if chain:
            adapter.set_params(previous_weights=previous)
        x_train = _train_slice(X, x_arr, fold)
        y_train = _train_target(y_arr, fold, n_assets)
        started = time.perf_counter()
        if y_train is None:
            adapter.fit(x_train)
        else:
            adapter.fit(x_train, y_train)
        fitted = np.asarray(adapter.weights_, dtype=np.float64)
        if fitted.ndim != 1:
            raise ValueError("2-dimensional weights_ cannot be assembled")
        weights[fold.fold_id] = np.ascontiguousarray(fitted)
        if chain:
            previous = weights[fold.fold_id]
        solve_s += time.perf_counter() - started
    return FoldBatchResult(
        weights=weights,
        solve_s=solve_s,
        n_solves=len(folds),
        n_warm_starts=adapter.n_warm_starts - warm_before,
        n_rebuilds=adapter.n_rebuilds - rebuild_before,
        is_dpp=adapter.is_dpp_,
        previous_weights=previous_used,
    )


def closed_form_weights(
    X: NDArray[np.float64],
    folds: Sequence[FoldSpec],
    estimator,
    *,
    fold_blocks: Sequence[NDArray[np.intp]] | None = None,
) -> FoldBatchResult:
    """EqualWeighted / Random / InverseVolatility weights for one path batch.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_assets)
        Asset returns (full universe).

    folds : sequence of FoldSpec
        Compiled folds for this path batch.

    estimator : EqualWeighted or Random or InverseVolatility
        Closed-form portfolio estimator.

    fold_blocks : sequence of ndarray of shape (n_block,), optional
        CPCV fold-block row indices used by InverseVolatility moment reuse.

    Returns
    -------
    result : FoldBatchResult
        Weight vectors of shape ``(n_assets,)`` (or the MRC subset size) keyed
        by ``fold_id``.

    Notes
    -----
    EqualWeighted and Random never touch the return matrix. InverseVolatility
    reuses the same overlapping-moment cache as compact MeanRisk variance.
    """
    inverse_vol = type(estimator) is InverseVolatility
    session = (
        path_moment_session(X, folds, keep_returns=False, fold_blocks=fold_blocks)
        if inverse_vol
        else None
    )
    if session is not None:
        n_assets = session.x_work.shape[1]
    elif folds and folds[0].asset_idx is not None:
        n_assets = int(folds[0].asset_idx.size)
    else:
        n_assets = int(X.shape[1])
    weights: dict[int, NDArray[np.float64]] = {}
    moments_s = 0.0
    draw_random = type(estimator) is Random
    for fold in folds:
        if session is None:
            if draw_random:
                weights[fold.fold_id] = rand_weights_dirichlet(n=n_assets)
            else:
                weights[fold.fold_id] = np.full(n_assets, 1.0 / n_assets)
            continue
        started = time.perf_counter()
        moments = session.get(fold)
        moments_s += time.perf_counter() - started
        inverse_volatility = 1.0 / np.sqrt(np.diag(moments.covariance))
        weights[fold.fold_id] = inverse_volatility / inverse_volatility.sum()
    return FoldBatchResult(
        weights=weights,
        moments_s=moments_s,
        n_prior_fits=0 if session is None else session.cache.n_fits,
        n_prior_updates=0 if session is None else session.cache.n_updates,
    )


def _segment_params(estimator) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    own = getattr(estimator, "portfolio_params", None)
    if own:
        extra.update(own)
    for name in _PORTFOLIO_ATTRS:
        if name not in extra and hasattr(estimator, name):
            extra[name] = getattr(estimator, name)
    extra.pop("name", None)
    extra.pop("check_observations_order", None)
    extra.pop("fallback_chain", None)
    return extra


def _train_target(y_arr: np.ndarray | None, fold: FoldSpec, n_assets: int):
    if y_arr is None:
        return None
    cols = fold.asset_idx
    if y_arr.ndim == 1 or cols is None or y_arr.shape[-1] != n_assets:
        return window_view(y_arr, fold.train_idx)
    return window_view(y_arr, fold.train_idx, cols)


def fit_native_weights(
    estimator,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: Sequence[FoldSpec],
) -> FoldBatchResult:
    """Clone, native ``fit``, and collect 1-D ``weights_`` for each fold.

    Parameters
    ----------
    estimator : BaseOptimization
        Unfitted portfolio optimizer. Each fold receives a fresh
        :func:`~sklearn.base.clone`.

    x_arr : ndarray of shape (n_observations, n_assets)
        Contiguous float64 returns.

    y_arr : ndarray of shape (n_observations,) or (n_observations, n_assets) or None
        Optional target passed to ``fit``.

    folds : sequence of FoldSpec
        Compiled train/test splits.

    Returns
    -------
    result : FoldBatchResult
        One weight vector per fold. Two-dimensional ``weights_`` (efficient
        frontiers) raise :class:`ValueError` because assembly expects a single
        portfolio per fold.

    Notes
    -----
    This is the shared serial path for estimators outside the compact subset
    (HRP, risk budgeting, ratio objectives, ...). Portfolio objects are not
    built here; see :func:`~skfolio_accelerate.scoring.assemble_prediction`.
    """
    n_assets = int(x_arr.shape[1])
    weights: dict[int, NDArray[np.float64]] = {}
    solve_s = 0.0
    for fold in folds:
        started = time.perf_counter()
        fitted = clone(estimator)
        x_train = window_view(x_arr, fold.train_idx, fold.asset_idx)
        y_train = _train_target(y_arr, fold, n_assets)
        if y_train is None:
            fitted.fit(x_train)
        else:
            fitted.fit(x_train, y_train)
        weights_ = np.asarray(fitted.weights_, dtype=np.float64)
        if weights_.ndim != 1:
            raise ValueError("2-dimensional weights_ cannot be assembled")
        weights[fold.fold_id] = np.ascontiguousarray(weights_)
        solve_s += time.perf_counter() - started
    return FoldBatchResult(
        weights=weights,
        solve_s=solve_s,
        n_solves=len(folds),
    )


def _fit_assemble_prediction(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    cv_plan: CVPlan,
    portfolio_params: dict | None,
) -> tuple[Any, FoldBatchResult, float]:
    merged = merge_batch_results(
        [
            fit_native_weights(estimator, x_arr, y_arr, batch)
            for batch in cv_plan.path_batches()
        ]
    )
    started = time.perf_counter()
    pred = assemble_prediction(
        X,
        cv_plan,
        merged.weights,
        name=type(estimator).__name__,
        portfolio_params=portfolio_params,
        segment_params=_segment_params(estimator),
    )
    return pred, merged, time.perf_counter() - started


def _skfolio_predict(
    estimator,
    X,
    y,
    cv,
    *,
    n_jobs,
    method,
    verbose,
    params,
    pre_dispatch,
    column_indices,
    portfolio_params,
    entry_rebalancing_params,
):
    return skfolio_cross_val_predict(
        estimator,
        X,
        y=y,
        cv=cv,
        n_jobs=n_jobs,
        method=method,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        column_indices=column_indices,
        portfolio_params=portfolio_params,
        entry_rebalancing_params=entry_rebalancing_params,
    )


def _report_from_batch(
    backend: BackendName | str,
    merged: FoldBatchResult,
    *,
    eval_s: float = 0.0,
    reason: str | None = None,
    fallback_reason: str | None = None,
    wall_s: float,
) -> AccelerationReport:
    return AccelerationReport(
        backend=backend,
        n_solves=int(merged.n_solves),
        n_prior_fits=int(merged.n_prior_fits),
        n_prior_updates=int(merged.n_prior_updates),
        n_warm_starts=int(merged.n_warm_starts),
        n_rebuilds=int(merged.n_rebuilds),
        is_dpp=merged.is_dpp,
        reason=reason,
        fallback_reason=fallback_reason,
        moments_s=float(merged.moments_s),
        solve_s=float(merged.solve_s),
        eval_s=eval_s,
        wall_s=wall_s,
    )


def cross_val_predict(
    estimator,
    X,
    y=None,
    cv=None,
    n_jobs: int | None = None,
    method: str = "predict",
    verbose: int = 0,
    params: dict | None = None,
    pre_dispatch: str = "2*n_jobs",
    column_indices=None,
    portfolio_params: dict | None = None,
    entry_rebalancing_params: dict | None = None,
    *,
    backend: str = "auto",
    return_report: bool = False,
) -> (
    MultiPeriodPortfolio
    | Population
    | tuple[MultiPeriodPortfolio | Population, AccelerationReport]
):
    """Generate cross-validated portfolio predictions with amortized backends.

    Drop-in replacement for :func:`skfolio.model_selection.cross_val_predict`.
    The call signature matches skfolio, with two additional keyword-only
    arguments: ``backend`` and ``return_report``.

    For single-path cross-validation such as ``KFold`` or
    :class:`~skfolio.model_selection.WalkForward`, the output is a
    :class:`~skfolio.portfolio.MultiPeriodPortfolio`. For combinatorial or
    multi-path splitters
    (:class:`~skfolio.model_selection.CombinatorialPurgedCV`,
    :class:`~skfolio.model_selection.MultipleRandomizedCV`), the output is a
    :class:`~skfolio.population.Population` of multi-period portfolios.

    Internally the call is compiled once into a
    :class:`~skfolio_accelerate.cv_plan.CVPlan`, then executed by the engine
    ``backend="auto"`` selects:

    1. compact OSQP / Clarabel, or closed-form weights, when the boxed
       problem applies,
    2. Parameterized CVXPY reuse for other MeanRisk configurations that keep
       a fixed problem shape (full constraint set),
    3. native ``fit`` plus assembly from ``weights_`` for other serial
       optimizers,
    4. unmodified skfolio when options or estimators require it.

    Leave ``backend`` at ``"auto"``. Pass ``return_report=True`` and read
    ``report.backend`` / ``report.reason`` if you need to see which engine
    ran. The other ``backend`` values are escape hatches for tests.

    Parameters
    ----------
    estimator : BaseOptimization or Pipeline
        Portfolio optimization estimator or pipeline whose last step is an
        optimization estimator.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.

    y : array-like of shape (n_observations,) or (n_observations, n_assets), optional
        Target relative to ``X`` for estimators that support it.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Compatible with
        sklearn splitters and skfolio's WalkForward, CombinatorialPurgedCV, and
        MultipleRandomizedCV.

    n_jobs : int, default=None
        Number of jobs for the native skfolio fallback. The amortized paths
        require ``n_jobs in {None, 1}``.

    method : str, default="predict"
        Invokes the given estimator method. Only ``"predict"`` is accelerated.

    verbose : int, default=0
        Verbosity level forwarded to native skfolio when used.

    params : dict, optional
        Parameters to pass to the ``fit`` method of the estimator. Any non-empty
        mapping disables amortization.

    pre_dispatch : int or str, default="2*n_jobs"
        Controls the number of jobs dispatched during parallel execution in the
        native fallback.

    column_indices : array-like, optional
        Column indices routing. Disables amortization when set outside the
        MultipleRandomizedCV asset subsets already encoded in the CV plan.

    portfolio_params : dict, optional
        Parameters passed to the portfolio constructor during assembly.

    entry_rebalancing_params : dict, optional
        Entry-rebalancing metadata. Disables amortization when set.

    backend : {"auto", "compact", "cvxpy-sequential", "sklearn"}, default="auto"
        Execution policy. ``"auto"`` (the default) selects OSQP, Clarabel,
        sequential CVXPY, closed-form, fit-assemble, or skfolio from the
        estimator and call options. The other values force a path and raise
        if that path is ineligible (except ``"sklearn"``, which always runs
        native skfolio).

    return_report : bool, default=False
        If ``True``, also return an :class:`AccelerationReport`.

    Returns
    -------
    predictions : MultiPeriodPortfolio or Population
        Result of calling ``predict`` on each test fold, assembled into the
        same container types as skfolio.

    report : AccelerationReport, optional
        Present only when ``return_report=True``.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, or an override (``"compact"``,
        ``"cvxpy-sequential"``) is requested for an ineligible estimator / call.

    Notes
    -----
    Compact numerical failure does not return an accelerator-only error when
    fit-assemble is allowed: the call retries with native ``fit`` and the same
    compiled CV plan. When even assembly is unavailable, the original splitter
    deepcopy is passed to skfolio so mutable ``RandomState`` objects are not
    double-consumed.

    Examples
    --------
    >>> from skfolio.model_selection import WalkForward
    >>> from skfolio.optimization import MeanRisk
    >>> from skfolio_accelerate import cross_val_predict
    >>> cv = WalkForward(train_size=252, test_size=21)  # doctest: +SKIP
    >>> prediction = cross_val_predict(MeanRisk(), X, cv=cv)  # doctest: +SKIP
    >>> prediction, report = cross_val_predict(
    ...     MeanRisk(), X, cv=cv, return_report=True
    ... )  # doctest: +SKIP
    >>> print(report.backend, report.reason)  # doctest: +SKIP
    osqp boxed MeanRisk variance; compact OSQP

    See Also
    --------
    grid_search : Compact or sequential MeanRisk hyperparameter search.
    classify_call : Inspect auto engine selection without running.
    """
    _cap_native_threads()
    t_wall = time.perf_counter()
    capabilities = classify_call(
        estimator,
        y=y,
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        n_jobs=n_jobs,
        cv=cv,
    )
    chosen = resolve_backend(backend, capabilities, estimator)

    def _return(pred, report):
        return (pred, report) if return_report else pred

    if chosen == "sklearn":
        pred = _skfolio_predict(
            estimator,
            X,
            y,
            cv,
            n_jobs=n_jobs,
            method=method,
            verbose=verbose,
            params=params,
            pre_dispatch=pre_dispatch,
            column_indices=column_indices,
            portfolio_params=portfolio_params,
            entry_rebalancing_params=entry_rebalancing_params,
        )
        report = AccelerationReport(
            backend="sklearn",
            reason=capabilities.choice_reason(
                "sklearn",
                fallback="backend=sklearn" if backend == "sklearn" else None,
            ),
            wall_s=time.perf_counter() - t_wall,
        )
        return _return(pred, report)

    estimator = clone(estimator)
    x_arr = as_float_2d(X)
    y_arr = None if y is None else as_float_array(y)
    # A compact numerical failure must retry the exact original split plan.
    # Some splitters accept mutable RandomState objects and advance them in split().
    fallback_cv = copy.deepcopy(cv)
    cv_plan = compile_cv_plan(cv, X, y)

    def _after_engine_failure(fail_reason: str):
        if capabilities.can_assemble:
            pred, merged, eval_s = _fit_assemble_prediction(
                estimator,
                X,
                x_arr,
                y_arr,
                cv_plan,
                portfolio_params,
            )
            report = _report_from_batch(
                "fit-assemble",
                merged,
                eval_s=eval_s,
                reason=capabilities.choice_reason("fit-assemble"),
                fallback_reason=fail_reason,
                wall_s=time.perf_counter() - t_wall,
            )
            return _return(pred, report)
        pred = _skfolio_predict(
            estimator,
            X,
            y,
            fallback_cv,
            n_jobs=n_jobs,
            method=method,
            verbose=verbose,
            params=params,
            pre_dispatch=pre_dispatch,
            column_indices=column_indices,
            portfolio_params=portfolio_params,
            entry_rebalancing_params=entry_rebalancing_params,
        )
        report = AccelerationReport(
            backend="sklearn",
            reason=capabilities.choice_reason("sklearn"),
            fallback_reason=fail_reason,
            wall_s=time.perf_counter() - t_wall,
        )
        return _return(pred, report)

    if chosen == "closed-form":
        merged = merge_batch_results(
            [
                closed_form_weights(
                    x_arr,
                    batch,
                    estimator,
                    fold_blocks=cv_plan.fold_blocks,
                )
                for batch in cv_plan.path_batches()
            ]
        )
        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged.weights,
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
        )
        report = _report_from_batch(
            "closed-form",
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=capabilities.choice_reason("closed-form"),
            wall_s=time.perf_counter() - t_wall,
        )
        return _return(pred, report)

    if chosen in {"osqp", "clarabel"}:
        spec = estimator_spec(estimator)
        keep_returns = spec.needs_returns()
        # MRC paths have the same number of assets. Reuse one OSQP topology across
        # paths, while deliberately disabling the first warm start of each path.
        # Clarabel workspaces are not shared across paths: there is no supported
        # cold-start reset, so a leftover interior point can leak between subsets.
        shared_engines = (
            EngineCache(spec=spec)
            if cv_plan.kind == "mrc" and not spec.needs_returns()
            else None
        )
        try:
            merged = merge_batch_results(
                [
                    solve_compact_folds(
                        path_moment_session(
                            x_arr,
                            batch,
                            keep_returns=keep_returns,
                            fold_blocks=cv_plan.fold_blocks,
                        ),
                        batch,
                        spec,
                        engines=shared_engines,
                    )
                    for batch in cv_plan.path_batches()
                ]
            )
        except (RuntimeError, ValueError) as error:
            fail_reason = (
                f"compact {spec.risk_measure.name} solve failed: "
                f"{type(error).__name__}: {error}"
            )
            return _after_engine_failure(fail_reason)

        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged.weights,
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
        )
        report = _report_from_batch(
            chosen,
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=capabilities.choice_reason(chosen),
            wall_s=time.perf_counter() - t_wall,
        )
        return _return(pred, report)

    if chosen == "cvxpy-sequential":
        cache = SequentialProblemCache(estimator)
        chain_prev = chains_previous_weights(cv_plan)
        try:
            merged = merge_batch_results(
                [
                    solve_sequential_folds(
                        estimator,
                        X,
                        x_arr,
                        y_arr,
                        batch,
                        cache=cache,
                        path_id=path_index,
                        chain_previous_weights=chain_prev,
                    )
                    for path_index, batch in enumerate(cv_plan.path_batches())
                ]
            )
        except Exception as error:
            fail_reason = (
                f"cvxpy-sequential solve failed: {type(error).__name__}: {error}"
            )
            return _after_engine_failure(fail_reason)

        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged.weights,
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
            segment_params=_segment_params(estimator),
            fold_segment_params={
                fold_id: {"previous_weights": prev}
                for fold_id, prev in merged.previous_weights.items()
            },
        )
        report = _report_from_batch(
            "cvxpy-sequential",
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=capabilities.choice_reason("cvxpy-sequential"),
            wall_s=time.perf_counter() - t_wall,
        )
        return _return(pred, report)

    pred, merged, eval_s = _fit_assemble_prediction(
        estimator,
        X,
        x_arr,
        y_arr,
        cv_plan,
        portfolio_params,
    )
    report = _report_from_batch(
        "fit-assemble",
        merged,
        eval_s=eval_s,
        reason=capabilities.choice_reason("fit-assemble"),
        wall_s=time.perf_counter() - t_wall,
    )
    return _return(pred, report)


massive_cross_val_predict = cross_val_predict
