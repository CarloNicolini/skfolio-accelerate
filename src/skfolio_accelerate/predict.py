"""Amortized multi-path ``cross_val_predict`` for skfolio splitters."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from skfolio import RiskMeasure
from skfolio.model_selection import cross_val_predict

from skfolio_accelerate.backends.sklearn_fallback import acceleration_blocked_reason
from skfolio_accelerate.compact import EngineCache, estimator_spec
from skfolio_accelerate.cv_plan import compile_cv_plan, cpcv_fold_blocks
from skfolio_accelerate.ir import AccelerationReport, FoldSpec
from skfolio_accelerate.moments import (
    OverlapMomentCache,
    as_float_2d,
    empirical_from_window,
    fit_prior,
    is_default_empirical,
)
from skfolio_accelerate.scoring import (
    assemble_prediction,
    native_path_sharpe,
)


def _cap_native_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _nonzero(value: Any) -> bool:
    arr = np.asarray(value, dtype=float)
    return bool(np.any(np.abs(arr) > 0))


def path_predict_blocked_reason(estimator) -> str | None:
    reason = acceleration_blocked_reason(estimator, {}, None, None)
    if reason is not None:
        return reason
    if getattr(estimator, "min_return", None) is not None:
        return "min_return is not accelerable in the compact path engine"
    if _nonzero(getattr(estimator, "l1_coef", 0.0)):
        return "l1_coef is not accelerable in the compact path engine"
    min_w = getattr(estimator, "min_weights", 0.0)
    max_w = getattr(estimator, "max_weights", 1.0)
    if isinstance(min_w, dict) or isinstance(max_w, dict):
        return "dict weight bounds are not accelerable in the compact path engine"
    return None


def _n_jobs(n_jobs: int | None) -> int:
    if n_jobs in (None, 0):
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return int(n_jobs)


def _needs_returns(spec: dict[str, Any]) -> bool:
    return spec["risk_measure"] is RiskMeasure.CVAR


def _path_groups(folds: list[FoldSpec], n_paths: int, kind: str) -> list[list[FoldSpec]]:
    if kind == "mrc":
        buckets: dict[int, list[FoldSpec]] = defaultdict(list)
        for fold in folds:
            buckets[fold.path_id].append(fold)
        return [buckets[key] for key in sorted(buckets)]
    return [folds]


def _chunk(items: list, n_chunks: int) -> list[list]:
    if n_chunks <= 1 or len(items) <= 1:
        return [items]
    n_chunks = min(n_chunks, len(items))
    sizes = [len(items) // n_chunks] * n_chunks
    for i in range(len(items) % n_chunks):
        sizes[i] += 1
    out: list[list] = []
    start = 0
    for size in sizes:
        if size:
            out.append(items[start : start + size])
        start += size
    return out


def _run_fold_batch(
    X: np.ndarray,
    folds: list[FoldSpec],
    spec: dict[str, Any],
    *,
    keep_returns: bool,
    incremental: bool,
    fold_blocks: list[np.ndarray] | None,
) -> dict[str, Any]:
    first_assets = folds[0].asset_idx if folds else None
    if first_assets is not None:
        x_work = X[:, np.asarray(first_assets, dtype=np.intp)]
        blocks = None
    else:
        x_work = X
        blocks = fold_blocks
    cache = (
        OverlapMomentCache(x_work, keep_returns=keep_returns, fold_blocks=blocks)
        if incremental
        else None
    )
    engines = EngineCache(spec=spec)
    weights: dict[int, np.ndarray] = {}
    moments_s = 0.0
    solve_s = 0.0
    n_solves = 0
    n_warm = 0
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        if cache is not None:
            moments = cache.get(fold, path_key=fold.path_id)
        else:
            moments = empirical_from_window(
                x_work[fold.train_idx], keep_returns=keep_returns
            )
        moments_s += time.perf_counter() - t0
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if keep_returns else None,
        )
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
        n_solves += 1
        n_warm = int(getattr(engine, "n_warm_starts", 0))
    n_fits = int(cache.n_fits) if cache is not None else n_solves
    n_updates = int(cache.n_updates) if cache is not None else 0
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": solve_s,
        "n_solves": n_solves,
        "n_warm_starts": n_warm,
        "n_prior_fits": n_fits,
        "n_prior_updates": n_updates,
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


def massive_cross_val_predict(
    estimator,
    X,
    cv=None,
    *,
    y=None,
    n_jobs: int | None = None,
    backend: str = "auto",
    portfolio_params: dict | None = None,
    return_report: bool = False,
    build_portfolios: bool = True,
):
    """Fast ``cross_val_predict`` for WalkForward, CPCV, and MultipleRandomizedCV.

    Uses overlapping empirical moments and a compact QP/LP with warm starts.
    Returns the same ``MultiPeriodPortfolio`` / ``Population`` types as skfolio.
    """
    _cap_native_threads()
    t_wall = time.perf_counter()
    estimator = clone(estimator)
    jobs = _n_jobs(n_jobs)
    blocked = path_predict_blocked_reason(estimator)
    if backend not in {"auto", "compact", "sklearn"}:
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "sklearn" or (backend == "auto" and blocked is not None):
        pred = cross_val_predict(
            estimator,
            X,
            y=y,
            cv=cv,
            n_jobs=n_jobs,
            portfolio_params=portfolio_params,
        )
        report = AccelerationReport(
            backend="sklearn",
            dpp="n/a",
            fallback_reason=blocked or "backend=sklearn",
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred
    if blocked is not None:
        raise ValueError(f"backend={backend!r} cannot accelerate this predict: {blocked}")

    spec = estimator_spec(estimator)
    keep_returns = _needs_returns(spec)
    incremental = is_default_empirical(estimator)
    x_arr = as_float_2d(X)
    cv_plan = compile_cv_plan(cv, X, y)
    fold_blocks = None
    if cv_plan.kind == "cpcv" and incremental:
        n_folds = int(getattr(cv, "n_folds"))
        fold_blocks = cpcv_fold_blocks(x_arr.shape[0], n_folds)

    if cv_plan.kind == "mrc":
        batches = _path_groups(cv_plan.folds, cv_plan.n_paths, cv_plan.kind)
    elif cv_plan.kind == "cpcv" and jobs > 1:
        batches = _chunk(cv_plan.folds, jobs)
        fold_blocks = None
    else:
        batches = [cv_plan.folds]

    if not incremental:
        # Custom priors: fit skfolio prior per window, still use compact solve.
        def _run_custom(fold_batch: list[FoldSpec]) -> dict[str, Any]:
            engines = EngineCache(spec=spec)
            weights: dict[int, np.ndarray] = {}
            moments_s = solve_s = 0.0
            n_warm = 0
            for i, fold in enumerate(fold_batch):
                from skfolio_accelerate.cv_plan import slice_panel

                x_train = slice_panel(X, fold.train_idx, fold.asset_idx)
                t0 = time.perf_counter()
                moments = fit_prior(estimator, x_train, keep_returns=keep_returns)
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
                "n_solves": len(fold_batch),
                "n_warm_starts": n_warm,
                "n_prior_fits": len(fold_batch),
                "n_prior_updates": 0,
            }

        if jobs == 1 or len(batches) == 1:
            parts = [_run_custom(batch) for batch in batches]
        else:
            parts = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_run_custom)(batch) for batch in batches
            )
        merged = _merge_batch_results(list(parts))
    else:
        if jobs == 1 or len(batches) == 1:
            parts = [
                _run_fold_batch(
                    x_arr,
                    batch,
                    spec,
                    keep_returns=keep_returns,
                    incremental=True,
                    fold_blocks=fold_blocks,
                )
                for batch in batches
            ]
        else:
            parts = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_run_fold_batch)(
                    x_arr,
                    batch,
                    spec,
                    keep_returns=keep_returns,
                    incremental=True,
                    fold_blocks=fold_blocks,
                )
                for batch in batches
            )
        merged = _merge_batch_results(list(parts))

    t_eval = time.perf_counter()
    name = type(estimator).__name__
    if build_portfolios:
        pred = assemble_prediction(
            X,
            cv_plan,
            merged["weights"],
            name=name,
            portfolio_params=portfolio_params,
        )
    else:
        # Native Sharpe paths packed as a Population-compatible list of floats
        # is not a public type; still assemble lightweight portfolios from
        # numpy views so the API stays drop-in.
        pred = assemble_prediction(
            X,
            cv_plan,
            merged["weights"],
            name=name,
            portfolio_params=portfolio_params,
        )
    eval_s = time.perf_counter() - t_eval
    wall_s = time.perf_counter() - t_wall
    n_solves = int(merged["n_solves"])
    moments_s = float(merged["moments_s"])
    solve_s = float(merged["solve_s"])
    accelerated = moments_s + solve_s + eval_s
    backend_name = (
        "osqp"
        if spec["risk_measure"] is RiskMeasure.VARIANCE
        else "clarabel-compact"
    )
    report = AccelerationReport(
        backend=backend_name,
        dpp="n/a",
        n_templates=1,
        n_evaluations=n_solves,
        n_prior_fits=int(merged["n_prior_fits"]),
        n_prior_updates=int(merged["n_prior_updates"]),
        n_native_solves=n_solves,
        n_updates=n_solves,
        n_warm_starts=int(merged["n_warm_starts"]),
        moments_s=moments_s,
        solve_s=solve_s,
        eval_s=eval_s,
        wall_s=wall_s,
    )
    del accelerated
    if return_report:
        return pred, report
    return pred


def path_sharpes(prediction) -> np.ndarray:
    """Sharpe of each path (Population) or the single MultiPeriodPortfolio."""
    if hasattr(prediction, "__len__") and not hasattr(prediction, "sharpe_ratio"):
        return np.asarray([ptf.sharpe_ratio for ptf in prediction], dtype=np.float64)
    if hasattr(prediction, "sharpe_ratio"):
        return np.asarray([prediction.sharpe_ratio], dtype=np.float64)
    raise TypeError(f"Unsupported prediction type {type(prediction)!r}")


def native_plan_sharpes(X, cv_plan, weights_by_fold: dict[int, np.ndarray]) -> np.ndarray:
    path_items: list[list[tuple]] = [[] for _ in range(cv_plan.n_paths)]
    if cv_plan.combinatorial:
        for fold in cv_plan.folds:
            w = weights_by_fold[fold.fold_id]
            for seg, path_id in zip(fold.test_segments, fold.path_ids, strict=False):
                path_items[path_id].append((w, seg, fold.asset_idx))
    else:
        for fold in cv_plan.folds:
            path_items[fold.path_id].append(
                (weights_by_fold[fold.fold_id], fold.test_idx, fold.asset_idx)
            )
    return np.asarray(
        [native_path_sharpe(X, items) for items in path_items], dtype=np.float64
    )
