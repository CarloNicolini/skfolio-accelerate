"""Amortized drop-in for ``skfolio.model_selection.cross_val_predict``."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone

from skfolio import RiskMeasure
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compact import EngineCache, estimator_spec
from skfolio_accelerate.cv_plan import FoldSpec, compile_cv_plan, cpcv_fold_blocks
from skfolio_accelerate.moments import (
    OverlapMomentCache,
    as_float_2d,
    is_default_empirical,
)
from skfolio_accelerate.scoring import assemble_prediction

_SUPPORTED_OBJECTIVES = {
    ObjectiveFunction.MINIMIZE_RISK,
    ObjectiveFunction.MAXIMIZE_UTILITY,
}
_SUPPORTED_RISKS = {RiskMeasure.VARIANCE, RiskMeasure.CVAR}
_BLOCKED_ATTRS = (
    ("cardinality", "cardinality (MIP)"),
    ("group_cardinalities", "group_cardinalities (MIP)"),
    ("threshold_long", "threshold_long (MIP)"),
    ("threshold_short", "threshold_short (MIP)"),
    ("add_constraints", "add_constraints"),
    ("add_objective", "add_objective"),
    ("overwrite_expected_return", "overwrite_expected_return"),
    ("efficient_frontier_size", "efficient_frontier_size"),
    ("mu_uncertainty_set_estimator", "mu uncertainty sets"),
    ("covariance_uncertainty_set_estimator", "covariance uncertainty sets"),
)


@dataclass
class AccelerationReport:
    backend: str
    n_solves: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_warm_starts: int = 0
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


def _nonzero(value: Any) -> bool:
    return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 0))


def blocked_reason(estimator) -> str | None:
    """Why the compact engine cannot run this estimator, or None if it can."""
    if not isinstance(estimator, MeanRisk):
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    for attr, label in _BLOCKED_ATTRS:
        if getattr(estimator, attr, None) is not None:
            return f"{label} is not accelerated"
    objective = getattr(
        estimator, "objective_function", ObjectiveFunction.MINIMIZE_RISK
    )
    if objective not in _SUPPORTED_OBJECTIVES:
        return "objective_function is not accelerated"
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    if risk not in _SUPPORTED_RISKS:
        return "risk_measure is not accelerated"
    if getattr(estimator, "min_return", None) is not None:
        return "min_return is not accelerated"
    if _nonzero(getattr(estimator, "l1_coef", 0.0)):
        return "l1_coef is not accelerated"
    if isinstance(getattr(estimator, "min_weights", 0.0), dict) or isinstance(
        getattr(estimator, "max_weights", 1.0), dict
    ):
        return "dict weight bounds are not accelerated"
    if _nonzero(getattr(estimator, "transaction_costs", 0.0)) or _nonzero(
        getattr(estimator, "management_fees", 0.0)
    ):
        return "transaction costs / management fees are not accelerated"
    if not is_default_empirical(estimator):
        return "custom prior is not accelerated"
    return None


def _path_groups(folds: list[FoldSpec]) -> list[list[FoldSpec]]:
    buckets: dict[int, list[FoldSpec]] = defaultdict(list)
    for fold in folds:
        buckets[fold.path_id].append(fold)
    return [buckets[key] for key in sorted(buckets)]


def _run_fold_batch(
    X: np.ndarray,
    folds: list[FoldSpec],
    spec: dict[str, Any],
    *,
    keep_returns: bool,
    fold_blocks: list[np.ndarray] | None,
) -> dict[str, Any]:
    first_assets = folds[0].asset_idx if folds else None
    if first_assets is not None:
        x_work = X[:, np.asarray(first_assets, dtype=np.intp)]
        blocks = None
    else:
        x_work = X
        blocks = fold_blocks
    cache = OverlapMomentCache(x_work, keep_returns=keep_returns, fold_blocks=blocks)
    engines = EngineCache(spec=spec)
    weights: dict[int, np.ndarray] = {}
    moments_s = 0.0
    solve_s = 0.0
    n_warm = 0
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        moments = cache.get(fold, path_key=fold.path_id)
        moments_s += time.perf_counter() - t0
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if keep_returns else None,
        )
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
        n_warm = int(getattr(engine, "n_warm_starts", 0))
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": solve_s,
        "n_solves": len(folds),
        "n_warm_starts": n_warm,
        "n_prior_fits": int(cache.n_fits),
        "n_prior_updates": int(cache.n_updates),
    }


def _merge_batch_results(parts: list[dict[str, Any]]) -> dict[str, Any]:
    weights: dict[int, np.ndarray] = {}
    moments_s = solve_s = 0.0
    n_solves = n_warm = n_fits = n_updates = 0
    for part in parts:
        weights.update(part["weights"])
        moments_s += part["moments_s"]
        solve_s += part["solve_s"]
        n_solves += part["n_solves"]
        n_warm += part["n_warm_starts"]
        n_fits += part["n_prior_fits"]
        n_updates += part["n_prior_updates"]
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": solve_s,
        "n_solves": n_solves,
        "n_warm_starts": n_warm,
        "n_prior_fits": n_fits,
        "n_prior_updates": n_updates,
    }


def cross_val_predict(
    estimator,
    X,
    cv=None,
    *,
    y=None,
    n_jobs: int | None = None,
    backend: str = "auto",
    portfolio_params: dict | None = None,
    return_report: bool = False,
):
    """Drop-in for ``skfolio.model_selection.cross_val_predict``.

    Reuses overlapping empirical moments and a compact QP/LP with warm starts
    for ``MeanRisk`` on WalkForward, CombinatorialPurgedCV, MultipleRandomizedCV,
    and KFold. Other estimators and unsupported MeanRisk options fall back to
    skfolio. ``n_jobs`` is forwarded only to that fallback; the compact engine
    is sequential (tiny QPs lose to thread overhead).
    """
    _cap_native_threads()
    t_wall = time.perf_counter()
    estimator = clone(estimator)
    blocked = blocked_reason(estimator)
    if backend not in {"auto", "compact", "sklearn"}:
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "sklearn" or (backend == "auto" and blocked is not None):
        pred = skfolio_cross_val_predict(
            estimator,
            X,
            y=y,
            cv=cv,
            n_jobs=n_jobs,
            portfolio_params=portfolio_params,
        )
        report = AccelerationReport(
            backend="sklearn",
            fallback_reason=blocked or "backend=sklearn",
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred
    if blocked is not None:
        raise ValueError(f"backend={backend!r} cannot accelerate this predict: {blocked}")

    spec = estimator_spec(estimator)
    keep_returns = spec["risk_measure"] is RiskMeasure.CVAR
    x_arr = as_float_2d(X)
    cv_plan = compile_cv_plan(cv, X, y)
    fold_blocks = None
    if cv_plan.kind == "cpcv":
        fold_blocks = cpcv_fold_blocks(x_arr.shape[0], int(getattr(cv, "n_folds")))

    batches = _path_groups(cv_plan.folds) if cv_plan.kind == "mrc" else [cv_plan.folds]
    merged = _merge_batch_results(
        [
            _run_fold_batch(
                x_arr,
                batch,
                spec,
                keep_returns=keep_returns,
                fold_blocks=fold_blocks,
            )
            for batch in batches
        ]
    )

    t_eval = time.perf_counter()
    pred = assemble_prediction(
        X,
        cv_plan,
        merged["weights"],
        name=type(estimator).__name__,
        portfolio_params=portfolio_params,
    )
    eval_s = time.perf_counter() - t_eval
    backend_name = (
        "osqp" if spec["risk_measure"] is RiskMeasure.VARIANCE else "clarabel"
    )
    report = AccelerationReport(
        backend=backend_name,
        n_solves=int(merged["n_solves"]),
        n_prior_fits=int(merged["n_prior_fits"]),
        n_prior_updates=int(merged["n_prior_updates"]),
        n_warm_starts=int(merged["n_warm_starts"]),
        moments_s=float(merged["moments_s"]),
        solve_s=float(merged["solve_s"]),
        eval_s=eval_s,
        wall_s=time.perf_counter() - t_wall,
    )
    return (pred, report) if return_report else pred


massive_cross_val_predict = cross_val_predict
