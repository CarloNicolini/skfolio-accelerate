"""Drop-in for ``skfolio.model_selection.cross_val_predict``.

    A call is classified once, then executed as:

    CV definition → compiled :class:`~skfolio_accelerate.cv_plan.CVPlan`
    → backend (compact / sequential CVXPY / closed-form / fit-assemble / native)
    → fold weights → assembled Portfolio objects

Compact OSQP, HiGHS, and Clarabel kernels accelerate a subset of MeanRisk.
EqualWeighted, Random, and default InverseVolatility use closed-form weights.
Remaining serial estimators still call native ``fit``, then assemble test
portfolios from ``weights_`` so they skip joblib, train/test copies, and
``predict()``.

The accelerator never reinterprets an unsupported estimator as a nearby
compact problem. Capability checks are the only gate; numerical engines assume
they have already been passed.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
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
from skfolio_accelerate.cv_plan import CVPlan, FoldSpec, compile_cv_plan
from skfolio_accelerate.mean_risk_problem import (
    ParametricMeanRisk,
    SequentialProblemCache,
)
from skfolio_accelerate.moments import (
    PathMomentSession,
    is_default_empirical,
    path_moment_session,
)
from skfolio_accelerate.linear_lp import (
    continuation_unhelpful_reason,
    is_highs_lp_risk,
)
from skfolio_accelerate.scoring import assemble_prediction, window_view

BackendName = Literal[
    "osqp",
    "highs",
    "clarabel",
    "cvxpy-sequential",
    "closed-form",
    "fit-assemble",
    "sklearn",
    "compact-grid",
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
_PORTFOLIO_ATTRS = (
    "transaction_costs",
    "management_fees",
    "previous_weights",
    "risk_free_rate",
)


class AccelerationWarning(UserWarning):
    """Issued when ``backend="auto"`` skips an engine that would not speed up.

    CombinatorialPurgedCV with boxed MAD / FLPM falls back to native skfolio:
    those LPs are large and not rolling, so a persistent HiGHS basis is slower
    than Clarabel. Filter with ``warnings.filterwarnings`` if the notice is
    noisy in a batch job.
    """


@dataclass
class AccelerationReport:
    """Diagnostics for one :func:`cross_val_predict` call.

    Returned when ``return_report=True``. ``backend`` is the execution
    path; ``fallback_reason`` explains native or fit-assemble fallbacks.
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
            f"moments {self.moments_s:.4f}s  solve {self.solve_s:.4f}s  "
            f"eval {self.eval_s:.4f}s  wall {self.wall_s:.4f}s",
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
    independent. Auto uses compact OSQP/Clarabel when possible, otherwise
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
            Compact OSQP/Clarabel, sequential CVXPY, fit-assemble, or sklearn.
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


def _compact_backend_name(estimator) -> BackendName:
    """OSQP, HiGHS, or Clarabel for a compact-eligible estimator."""
    if type(estimator) in _CLOSED_FORM_TYPES:
        return "closed-form"
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
    if name := _first_set_attr(estimator, _COMPACT_NONE_ATTRS):
        return f"{name} is not compacted"
    if estimator.budget is None:
        return "an unspecified equality budget is not compacted"
    if estimator.needs_previous_weights:
        return "sequential previous_weights (costs, turnover, or fallback)"
    if estimator.objective_function not in _SUPPORTED_OBJECTIVES:
        return "objective_function is not compacted"
    risk = estimator.risk_measure
    if risk not in _SUPPORTED_RISKS:
        return "risk_measure is not compacted"
    allowed = {"CLARABEL", "OSQP"} if risk is RiskMeasure.VARIANCE else {"CLARABEL"}
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
        compact_reason=compact_blocked_reason(estimator, y=y, **call_kw),
        assemble_reason=assemble_blocked_reason(estimator, n_jobs=n_jobs, **call_kw),
        sequential_reason=sequential_blocked_reason(
            estimator, n_jobs=n_jobs, **call_kw
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
    moments_s = solve_s = 0.0
    n_solves = n_warm = n_fits = n_updates = n_rebuilds = 0
    is_dpp: bool | None = None
    for part in parts:
        weights.update(part.weights)
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
    weights: dict[int, NDArray[np.float64]] = {}
    moments_s = solve_s = 0.0
    warm_before = 0
    engine = None
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        moments = session.get(fold)
        moments_s += time.perf_counter() - t0
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if spec.needs_returns() else None,
        )
        if i == 0:
            warm_before = engine.n_warm_starts
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
    n_warm = 0 if engine is None else engine.n_warm_starts - warm_before
    return FoldBatchResult(
        weights=weights,
        moments_s=moments_s,
        solve_s=solve_s,
        n_solves=len(folds),
        n_warm_starts=n_warm,
        n_prior_fits=int(session.cache.n_fits),
        n_prior_updates=int(session.cache.n_updates),
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
    extra = dict(estimator.portfolio_params or {})
    state = estimator.__dict__
    for name in _PORTFOLIO_ATTRS:
        if name not in extra and name in state:
            extra[name] = state[name]
    extra.pop("name", None)
    extra.pop("check_observations_order", None)
    extra.pop("fallback_chain", None)
    return extra


def _train_slice(X, x_arr: np.ndarray, fold: FoldSpec):
    """Training window, keeping DataFrame columns when ``X`` supports ``iloc``."""
    try:
        rows = X.iloc[np.asarray(fold.train_idx)]
    except AttributeError:
        return window_view(x_arr, fold.train_idx, fold.asset_idx)
    if fold.asset_idx is not None:
        return rows.iloc[:, np.asarray(fold.asset_idx)]
    return rows


def solve_sequential_folds(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: Sequence[FoldSpec],
    *,
    cache: SequentialProblemCache | None = None,
    path_id: int = 0,
) -> FoldBatchResult:
    """Solve one path batch by Parameterizing MeanRisk and reusing ``cp.Problem``.

    Parameters
    ----------
    estimator : MeanRisk
        Estimator whose parameters are copied onto the sequential adapter.

    X : array-like
        Full return matrix (DataFrame columns are preserved for named constraints).

    x_arr : ndarray
        Float view of ``X``.

    y_arr : ndarray or None
        Optional target.

    folds : sequence of FoldSpec
        Compiled folds for this path.

    cache : SequentialProblemCache, optional
        Per-path adapter cache.

    path_id : int, default=0
        MRC path identifier.

    Returns
    -------
    result : FoldBatchResult
        Fold weights and reuse counters.
    """
    cache = SequentialProblemCache(estimator) if cache is None else cache
    adapter = cache.get(path_id)
    warm_before = adapter.n_warm_starts
    rebuild_before = adapter.n_rebuilds
    weights: dict[int, NDArray[np.float64]] = {}
    solve_s = 0.0
    n_assets = int(x_arr.shape[1])
    for fold in folds:
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
        solve_s += time.perf_counter() - started
    return FoldBatchResult(
        weights=weights,
        solve_s=solve_s,
        n_solves=len(folds),
        n_warm_starts=adapter.n_warm_starts - warm_before,
        n_rebuilds=adapter.n_rebuilds - rebuild_before,
        is_dpp=adapter.is_dpp_,
    )


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


def _choice_reason(backend: str, capabilities: CallCapabilities) -> str:
    match backend:
        case "osqp":
            return "boxed MeanRisk variance; compact OSQP"
        case "highs":
            return "boxed MeanRisk LP; persistent HiGHS simplex"
        case "clarabel":
            return "boxed MeanRisk scenario risk; compact Clarabel"
        case "closed-form":
            return "closed-form weights; no solver"
        case "cvxpy-sequential":
            if capabilities.compact_reason:
                return (
                    f"MeanRisk outside the compact subset "
                    f"({capabilities.compact_reason})"
                )
            return "Parameterized MeanRisk CVXPY reuse"
        case "fit-assemble":
            return (
                capabilities.sequential_reason
                or capabilities.compact_reason
                or "native fit; assemble from weights_"
            )
        case "sklearn":
            return (
                capabilities.assemble_reason
                or capabilities.sequential_reason
                or capabilities.compact_reason
                or "unmodified skfolio"
            )
        case _:
            return backend


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


def _after_solve_failure(
    *,
    fail_reason: str,
    capabilities: CallCapabilities,
    estimator,
    X,
    y,
    x_arr,
    y_arr,
    cv_plan: CVPlan,
    fallback_cv,
    portfolio_params,
    n_jobs,
    method,
    verbose,
    params,
    pre_dispatch,
    column_indices,
    entry_rebalancing_params,
    t_wall: float,
) -> tuple[Any, AccelerationReport]:
    """Native fit-assemble or skfolio fallback after a compact/sequential failure."""
    if capabilities.can_assemble:
        pred, merged, eval_s = _fit_assemble_prediction(
            estimator, X, x_arr, y_arr, cv_plan, portfolio_params
        )
        report = _report_from_batch(
            "fit-assemble",
            merged,
            eval_s=eval_s,
            reason=_choice_reason("fit-assemble", capabilities),
            fallback_reason=fail_reason,
            wall_s=time.perf_counter() - t_wall,
        )
        return pred, report
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
        reason=_choice_reason("sklearn", capabilities),
        fallback_reason=fail_reason,
        wall_s=time.perf_counter() - t_wall,
    )
    return pred, report


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
    :class:`~skfolio_accelerate.cv_plan.CVPlan`, then executed by the first
    eligible backend:

    1. compact OSQP (variance), HiGHS (scenario LPs), or Clarabel
       (scenario cones) for a narrow
       :class:`~skfolio.optimization.MeanRisk` subset,
    2. Parameterized CVXPY reuse for other MeanRisk configurations with a
       fixed problem shape,
    3. closed-form weights for default EqualWeighted / Random /
       InverseVolatility,
    4. native ``fit`` plus assembly from ``weights_`` for other serial
       optimizers,
    5. unmodified skfolio when options or estimators require it.

    Leave ``backend`` at ``"auto"``. Read ``report.backend`` /
    ``report.reason`` if you need to see which engine ran.

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
        Execution policy. ``"auto"`` selects OSQP, HiGHS, Clarabel, sequential
        CVXPY, closed-form, fit-assemble, or skfolio. The other values are
        test/debug overrides.

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
        If ``backend`` is unknown, or ``backend="compact"`` is requested for an
        ineligible estimator / call.

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
    >>> print(report.backend)  # doctest: +SKIP
    osqp

    See Also
    --------
    grid_search : Compact MeanRisk hyperparameter search.
    classify_call : Inspect compact / assemble eligibility without running.
    """
    _cap_native_threads()
    t_wall = time.perf_counter()
    if backend not in {"auto", "compact", "cvxpy-sequential", "sklearn"}:
        raise ValueError(f"Unknown backend {backend!r}")

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
    if backend == "auto":
        native_lp = continuation_unhelpful_reason(estimator, cv)
        if native_lp:
            warnings.warn(
                native_lp + ". Falling back to native skfolio cross_val_predict.",
                AccelerationWarning,
                stacklevel=2,
            )
    if backend == "cvxpy-sequential" and not capabilities.can_sequential:
        raise ValueError(
            f"backend={backend!r} cannot reuse this MeanRisk problem: "
            f"{capabilities.sequential_reason}"
        )
    if backend == "sklearn" or (
        backend == "auto"
        and not capabilities.can_compact
        and not capabilities.can_sequential
        and not capabilities.can_assemble
    ):
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
            reason=_choice_reason("sklearn", capabilities),
            fallback_reason=(
                "backend=sklearn"
                if backend == "sklearn"
                else (
                    capabilities.assemble_reason
                    or capabilities.sequential_reason
                    or capabilities.compact_reason
                )
            ),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred
    if backend == "compact" and not capabilities.can_compact:
        raise ValueError(
            f"backend={backend!r} cannot compact this predict: "
            f"{capabilities.compact_reason}"
        )

    estimator = clone(estimator)
    x_arr = as_float_2d(X)
    y_arr = None if y is None else as_float_array(y)
    # A compact numerical failure must retry the exact original split plan.
    # Some splitters accept mutable RandomState objects and advance them in split().
    fallback_cv = copy.deepcopy(cv)
    cv_plan = compile_cv_plan(cv, X, y)

    if capabilities.can_compact and type(estimator) in _CLOSED_FORM_TYPES:
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
            reason=_choice_reason("closed-form", capabilities),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

    if capabilities.can_compact and backend != "cvxpy-sequential":
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
            pred, report = _after_solve_failure(
                fail_reason=fail_reason,
                capabilities=capabilities,
                estimator=estimator,
                X=X,
                y=y,
                x_arr=x_arr,
                y_arr=y_arr,
                cv_plan=cv_plan,
                fallback_cv=fallback_cv,
                portfolio_params=portfolio_params,
                n_jobs=n_jobs,
                method=method,
                verbose=verbose,
                params=params,
                pre_dispatch=pre_dispatch,
                column_indices=column_indices,
                entry_rebalancing_params=entry_rebalancing_params,
                t_wall=t_wall,
            )
            return (pred, report) if return_report else pred

        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged.weights,
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
        )
        backend_name: BackendName = _compact_backend_name(estimator)
        report = _report_from_batch(
            backend_name,
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=_choice_reason(backend_name, capabilities),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

    if capabilities.can_sequential:
        cache = SequentialProblemCache(estimator)
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
                    )
                    for path_index, batch in enumerate(cv_plan.path_batches())
                ]
            )
        except (RuntimeError, ValueError) as error:
            fail_reason = (
                f"cvxpy-sequential solve failed: {type(error).__name__}: {error}"
            )
            pred, report = _after_solve_failure(
                fail_reason=fail_reason,
                capabilities=capabilities,
                estimator=estimator,
                X=X,
                y=y,
                x_arr=x_arr,
                y_arr=y_arr,
                cv_plan=cv_plan,
                fallback_cv=fallback_cv,
                portfolio_params=portfolio_params,
                n_jobs=n_jobs,
                method=method,
                verbose=verbose,
                params=params,
                pre_dispatch=pre_dispatch,
                column_indices=column_indices,
                entry_rebalancing_params=entry_rebalancing_params,
                t_wall=t_wall,
            )
            return (pred, report) if return_report else pred
        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged.weights,
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
            segment_params=_segment_params(estimator),
        )
        report = _report_from_batch(
            "cvxpy-sequential",
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=_choice_reason("cvxpy-sequential", capabilities),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

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
        reason=_choice_reason("fit-assemble", capabilities),
        fallback_reason=capabilities.compact_reason,
        wall_s=time.perf_counter() - t_wall,
    )
    return (pred, report) if return_report else pred

