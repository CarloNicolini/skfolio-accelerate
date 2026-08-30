"""Drop-in for ``skfolio.model_selection.cross_val_predict``.

    A call is classified once, then executed as:

    CV definition → compiled :class:`~skfolio_accelerate.cv_plan.CVPlan`
    → weights (compact / sequential CVXPY / native ``fit`` / trivial formula)
    → assembled Portfolio objects

Compact OSQP, HiGHS, and Clarabel kernels accelerate a subset of MeanRisk.
Every other serial estimator shares the compiled plan, contiguous training
slices, and assembly from ``weights_``, which skips joblib, ``safe_split``
copies, and ``predict()``. That bookkeeping saving is independent of the
estimator: EqualWeighted happens to skip ``fit`` because the weights are
trivial; HRP still calls native ``fit`` and then uses the same assembly.

The accelerator never reinterprets an unsupported estimator as a nearby
compact problem. Capability checks are the only gate; numerical engines assume
they have already been passed.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio
from sklearn.base import clone

from skfolio_accelerate._capabilities import (
    _CLOSED_FORM_TYPES,
    BackendName,
    CallCapabilities,
    _compact_backend_name,
    assemble_blocked_reason,
    blocked_reason,
    classify_call,
    compact_blocked_reason,
    sequential_blocked_reason,
)
from skfolio_accelerate._solvers import (
    FoldBatchResult,
    _segment_params,
    closed_form_weights,
    fit_native_weights,
    merge_batch_results,
    solve_compact_folds,
    solve_sequential_folds,
)
from skfolio_accelerate.compact import EngineCache, estimator_spec
from skfolio_accelerate.cv_plan import CVPlan, compile_cv_plan
from skfolio_accelerate.linear_lp import continuation_unhelpful_reason
from skfolio_accelerate.mean_risk_problem import SequentialProblemCache
from skfolio_accelerate.moments import path_moment_session
from skfolio_accelerate.scoring import assemble_prediction

# Public names for this module. Capability helpers remain importable for
# diagnostics; solver internals live in ``_solvers``.
__all__ = [
    "AccelerationReport",
    "AccelerationWarning",
    "CallCapabilities",
    "assemble_blocked_reason",
    "blocked_reason",
    "classify_call",
    "compact_blocked_reason",
    "cross_val_predict",
    "sequential_blocked_reason",
]


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


def _cap_native_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _fit_assemble_prediction(
    estimator,
    X,
    x_arr,
    y_arr,
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


def _native_solver_estimator(estimator):
    """Rewrite COSMO solver names so a native fallback can still run."""
    solver = str(getattr(estimator, "solver", "") or "")
    if solver.upper() in {"COSMO", "COSMO_RS", "COSMO_RUST"}:
        return clone(estimator).set_params(solver="CLARABEL")
    return estimator


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
        case "max-return":
            return "boxed maximum-return MeanRisk; analytic L2 projection"
        case "cosmo":
            return "boxed MeanRisk; persistent COSMO.rs ADMM"
        case "closed-form":
            return "trivial weights; shared serial CV assembly"
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
            _native_solver_estimator(estimator),
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
            fallback_reason=fail_reason,
            wall_s=time.perf_counter() - t_wall,
        )
        return pred, report
    pred = _skfolio_predict(
        _native_solver_estimator(estimator),
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
    :class:`~skfolio_accelerate.cv_plan.CVPlan`. Compact engines amortize the
    MeanRisk *solver*. Serial estimators that are not in that subset still
    share the compiled plan and assemble test portfolios from fold weights
    (native ``fit`` unless the weights are a trivial formula):

    1. compact OSQP (variance), HiGHS (scenario LPs), or Clarabel
       (scenario cones) for a narrow
       :class:`~skfolio.optimization.MeanRisk` subset,
    2. Parameterized CVXPY reuse for other MeanRisk configurations with a
       fixed problem shape,
    3. serial assembly from ``weights_`` for any other
       :class:`~skfolio.optimization.BaseOptimization` (``fit`` is skipped
       only when weights are closed-form),
    4. unmodified skfolio when options or estimators require it.

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

    backend : str, default="auto"
        One of ``"auto"``, ``"compact"``, ``"cvxpy-sequential"``,
        ``"sklearn"``, or ``"cosmo"``. ``"auto"`` selects a compact solver,
        sequential CVXPY, serial assembly from ``weights_``, or skfolio.
        ``"cosmo"`` forces the optional persistent COSMO.rs compact engine
        (not selected by ``"auto"``). The other values are test/debug
        overrides.

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
        If ``backend`` is unknown, or ``backend="compact"`` / ``"cosmo"`` is
        requested for an ineligible estimator / call.

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
    if backend not in {"auto", "compact", "cvxpy-sequential", "sklearn", "cosmo"}:
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "cosmo":
        from skfolio_accelerate._cosmo import (
            cosmo_available,
            cosmo_cv_blocked_reason,
        )

        if not cosmo_available():
            raise ImportError(
                "backend='cosmo' requires COSMO.rs. Install the optional extra "
                "skfolio-accelerate[cosmo] or build "
                "https://github.com/CarloNicolini/COSMO.rs with maturin."
            )
        risk = getattr(estimator, "risk_measure", None)
        if reason := cosmo_cv_blocked_reason(risk):
            raise ValueError(reason)

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
            _native_solver_estimator(estimator),
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
    if backend in {"compact", "cosmo"} and not capabilities.can_compact:
        raise ValueError(
            f"backend={backend!r} cannot compact this predict: "
            f"{capabilities.compact_reason}"
        )
    if backend == "cosmo" and type(estimator) in _CLOSED_FORM_TYPES:
        raise ValueError("backend='cosmo' requires a compact MeanRisk estimator")

    estimator = clone(estimator)
    x_arr = np.ascontiguousarray(X, dtype=np.float64)
    y_arr = None if y is None else np.ascontiguousarray(y, dtype=np.float64)
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
        if backend == "cosmo":
            spec = replace(spec, solver="COSMO")
        keep_returns = spec.needs_returns()
        # MRC paths have the same number of assets. Reuse one OSQP topology across
        # paths, while deliberately disabling the first warm start of each path.
        # Clarabel workspaces are not shared across paths: there is no supported
        # cold-start reset, so a leftover interior point can leak between subsets.
        # COSMO workspaces are also path-local (different asset subsets / ADMM state).
        shared_engines = (
            EngineCache(spec=spec)
            if (
                cv_plan.kind == "mrc"
                and not spec.needs_returns()
                and not spec.uses_cosmo()
            )
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
                            keep_covariance=not keep_returns,
                            fold_blocks=cv_plan.fold_blocks,
                        ),
                        batch,
                        spec,
                        engines=shared_engines,
                    )
                    for batch in cv_plan.path_batches()
                ]
            )
        except (RuntimeError, ValueError, ImportError) as error:
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
        backend_name: BackendName = (
            "cosmo" if spec.uses_cosmo() else _compact_backend_name(estimator)
        )
        report = _report_from_batch(
            backend_name,
            merged,
            eval_s=time.perf_counter() - t_eval,
            reason=_choice_reason(backend_name, capabilities),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

    if capabilities.can_sequential:
        cache = SequentialProblemCache(_native_solver_estimator(estimator))
        try:
            merged = merge_batch_results(
                [
                    solve_sequential_folds(
                        _native_solver_estimator(estimator),
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
        _native_solver_estimator(estimator),
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
