"""Drop-in for ``skfolio.model_selection.cross_val_predict``."""

from __future__ import annotations

import copy
import os
import time
import warnings
from dataclasses import dataclass
import numpy as np
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
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
    """``backend="auto"`` skipped an engine that would not speed up."""


@dataclass
class AccelerationReport:
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
        extra = f"\nBaseline {self.baseline_s:.4f}s  speedup {self.speedup:.2f}×" if self.baseline_s > 0 else ""
        return (
            f"Backend: {self.backend}\nReason: {self.reason or 'none'}\n"
            f"Solves: {self.n_solves}\nMoment fits: {self.n_prior_fits}\n"
            f"Moment updates: {self.n_prior_updates}\nWarm starts: {self.n_warm_starts}\n"
            f"moments {self.moments_s:.4f}s  solve {self.solve_s:.4f}s  "
            f"eval {self.eval_s:.4f}s  wall {self.wall_s:.4f}s\n"
            f"Fallback: {self.fallback_reason or 'none'}{extra}"
        )


_REASONS = {
    "osqp": "boxed MeanRisk variance; compact OSQP",
    "highs": "boxed MeanRisk LP; persistent HiGHS simplex",
    "clarabel": "boxed MeanRisk scenario risk; compact Clarabel",
    "max-return": "boxed maximum-return MeanRisk; analytic L2 projection",
    "closed-form": "trivial weights; shared serial CV assembly",
}


def _choice_reason(backend: str, capabilities: CallCapabilities) -> str:
    if backend in _REASONS:
        return _REASONS[backend]
    if backend == "cvxpy-sequential":
        if capabilities.compact_reason:
            return f"MeanRisk outside the compact subset ({capabilities.compact_reason})"
        return "Parameterized MeanRisk CVXPY reuse"
    if backend == "fit-assemble":
        return (
            capabilities.sequential_reason
            or capabilities.compact_reason
            or "native fit; assemble from weights_"
        )
    if backend == "sklearn":
        return (
            capabilities.assemble_reason
            or capabilities.sequential_reason
            or capabilities.compact_reason
            or "unmodified skfolio"
        )
    return backend


def _cap_native_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _skfolio_predict(estimator, X, y, cv, **kw):
    return skfolio_cross_val_predict(estimator, X, y=y, cv=cv, **kw)


def _report(backend, merged: FoldBatchResult, *, eval_s=0.0, reason=None, fallback_reason=None, wall_s: float):
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


def _assemble(X, cv_plan, merged, name, *, portfolio_params=None, segment_params=None):
    started = time.perf_counter()
    pred = assemble_prediction(
        X, cv_plan, merged.weights, name=name,
        portfolio_params=portfolio_params, segment_params=segment_params,
    )
    return pred, time.perf_counter() - started


def _after_failure(*, fail_reason, capabilities, estimator, X, y, x_arr, y_arr, cv_plan, fallback_cv, portfolio_params, n_jobs, method, verbose, params, pre_dispatch, column_indices, entry_rebalancing_params, t_wall):
    if capabilities.can_assemble:
        merged = merge_batch_results(
            [fit_native_weights(estimator, X, x_arr, y_arr, batch, params=params) for batch in cv_plan.path_batches()]
        )
        pred, eval_s = _assemble(X, cv_plan, merged, type(estimator).__name__, portfolio_params=portfolio_params, segment_params=_segment_params(estimator))
        return pred, _report("fit-assemble", merged, eval_s=eval_s, reason=_choice_reason("fit-assemble", capabilities), fallback_reason=fail_reason, wall_s=time.perf_counter() - t_wall)
    pred = _skfolio_predict(
        estimator, X, y, fallback_cv, n_jobs=n_jobs, method=method, verbose=verbose,
        params=params, pre_dispatch=pre_dispatch, column_indices=column_indices,
        portfolio_params=portfolio_params, entry_rebalancing_params=entry_rebalancing_params,
    )
    return pred, AccelerationReport(
        backend="sklearn", reason=_choice_reason("sklearn", capabilities),
        fallback_reason=fail_reason, wall_s=time.perf_counter() - t_wall,
    )


def cross_val_predict(
    estimator, X, y=None, cv=None, n_jobs=None, method="predict", verbose=0,
    params=None, pre_dispatch="2*n_jobs", column_indices=None, portfolio_params=None,
    entry_rebalancing_params=None, *, backend="auto", return_report=False,
):
    """Amortized drop-in for skfolio ``cross_val_predict`` (same positional API)."""
    _cap_native_threads()
    t_wall = time.perf_counter()
    if backend not in {"auto", "compact", "cvxpy-sequential", "sklearn"}:
        raise ValueError(f"Unknown backend {backend!r}")
    capabilities = classify_call(
        estimator, y=y, method=method, params=params, column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params, n_jobs=n_jobs, cv=cv,
    )
    if backend == "auto":
        native_lp = continuation_unhelpful_reason(estimator, cv)
        if native_lp:
            warnings.warn(
                native_lp + ". Falling back to native skfolio cross_val_predict.",
                AccelerationWarning, stacklevel=2,
            )
    if backend == "cvxpy-sequential" and not capabilities.can_sequential:
        raise ValueError(
            f"backend={backend!r} cannot reuse this MeanRisk problem: {capabilities.sequential_reason}"
        )
    native_kw = dict(
        n_jobs=n_jobs, method=method, verbose=verbose, params=params,
        pre_dispatch=pre_dispatch, column_indices=column_indices,
        portfolio_params=portfolio_params, entry_rebalancing_params=entry_rebalancing_params,
    )
    if backend == "sklearn" or (
        backend == "auto" and not capabilities.can_compact
        and not capabilities.can_sequential and not capabilities.can_assemble
    ):
        pred = _skfolio_predict(estimator, X, y, cv, **native_kw)
        report = AccelerationReport(
            backend="sklearn",
            reason=_choice_reason("sklearn", capabilities),
            fallback_reason="backend=sklearn" if backend == "sklearn" else (
                capabilities.assemble_reason or capabilities.sequential_reason or capabilities.compact_reason
            ),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred
    if backend == "compact" and not capabilities.can_compact:
        raise ValueError(f"backend={backend!r} cannot compact this predict: {capabilities.compact_reason}")

    estimator = clone(estimator)
    x_arr = np.ascontiguousarray(X, dtype=np.float64)
    y_arr = None if y is None else np.ascontiguousarray(y, dtype=np.float64)
    fallback_cv = copy.deepcopy(cv)
    cv_plan = compile_cv_plan(cv, X, y)
    name = type(estimator).__name__
    fail_kw = dict(
        capabilities=capabilities, estimator=estimator, X=X, y=y, x_arr=x_arr, y_arr=y_arr,
        cv_plan=cv_plan, fallback_cv=fallback_cv, t_wall=t_wall, **native_kw,
    )

    if capabilities.can_compact and type(estimator) in _CLOSED_FORM_TYPES:
        merged = merge_batch_results(
            [closed_form_weights(x_arr, batch, estimator, fold_blocks=cv_plan.fold_blocks) for batch in cv_plan.path_batches()]
        )
        pred, eval_s = _assemble(X, cv_plan, merged, name, portfolio_params=portfolio_params)
        report = _report("closed-form", merged, eval_s=eval_s, reason=_choice_reason("closed-form", capabilities), wall_s=time.perf_counter() - t_wall)
        return (pred, report) if return_report else pred

    if capabilities.can_compact and backend != "cvxpy-sequential":
        spec = estimator_spec(
            estimator,
            names=tuple(map(str, X.columns)) if hasattr(X, "columns") else None,
        )
        keep_returns = spec.needs_returns()
        share = (
            cv_plan.kind == "mrc"
            and not keep_returns
            and spec.linear_constraints is None
            and spec.groups is None
        )
        shared = EngineCache(spec=spec) if share else None
        try:
            merged = merge_batch_results(
                [
                    solve_compact_folds(
                        path_moment_session(
                            x_arr, batch, keep_returns=keep_returns,
                            keep_covariance=not keep_returns, fold_blocks=cv_plan.fold_blocks,
                        ),
                        batch, spec, engines=shared,
                    )
                    for batch in cv_plan.path_batches()
                ]
            )
        except (RuntimeError, ValueError) as error:
            pred, report = _after_failure(
                fail_reason=f"compact {spec.risk_measure.name} solve failed: {type(error).__name__}: {error}",
                **fail_kw,
            )
            return (pred, report) if return_report else pred
        pred, eval_s = _assemble(X, cv_plan, merged, name, portfolio_params=portfolio_params)
        backend_name: BackendName = _compact_backend_name(estimator)
        report = _report(backend_name, merged, eval_s=eval_s, reason=_choice_reason(backend_name, capabilities), wall_s=time.perf_counter() - t_wall)
        return (pred, report) if return_report else pred

    if capabilities.can_sequential:
        cache = SequentialProblemCache(estimator)
        try:
            merged = merge_batch_results(
                [
                    solve_sequential_folds(
                        estimator, X, x_arr, y_arr, batch, cache=cache, path_id=path_index, params=params,
                    )
                    for path_index, batch in enumerate(cv_plan.path_batches())
                ]
            )
        except (RuntimeError, ValueError) as error:
            pred, report = _after_failure(
                fail_reason=f"cvxpy-sequential solve failed: {type(error).__name__}: {error}",
                **fail_kw,
            )
            return (pred, report) if return_report else pred
        pred, eval_s = _assemble(
            X, cv_plan, merged, name, portfolio_params=portfolio_params, segment_params=_segment_params(estimator),
        )
        report = _report("cvxpy-sequential", merged, eval_s=eval_s, reason=_choice_reason("cvxpy-sequential", capabilities), wall_s=time.perf_counter() - t_wall)
        return (pred, report) if return_report else pred

    merged = merge_batch_results(
        [fit_native_weights(estimator, X, x_arr, y_arr, batch, params=params) for batch in cv_plan.path_batches()]
    )
    pred, eval_s = _assemble(
        X, cv_plan, merged, name, portfolio_params=portfolio_params, segment_params=_segment_params(estimator),
    )
    report = _report(
        "fit-assemble", merged, eval_s=eval_s, reason=_choice_reason("fit-assemble", capabilities),
        fallback_reason=capabilities.compact_reason, wall_s=time.perf_counter() - t_wall,
    )
    return (pred, report) if return_report else pred
