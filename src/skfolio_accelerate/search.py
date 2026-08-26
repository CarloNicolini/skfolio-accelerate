"""Hyperparameter search that shares CV plans and empirical moments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from skfolio import RiskMeasure
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from skfolio_accelerate.compact import EngineCache, estimator_spec
from skfolio_accelerate.cv_plan import compile_cv_plan, cpcv_fold_blocks
from skfolio_accelerate.moments import OverlapMomentCache, as_float_2d
from skfolio_accelerate.predict import AccelerationReport, compact_blocked_reason
from skfolio_accelerate.scoring import (
    assemble_prediction,
    path_sharpes_from_weights,
)


@dataclass
class GridSearchResult:
    """Result of :func:`grid_search`."""

    best_params_: dict[str, Any]
    best_score_: float
    best_index_: int
    best_prediction_: Any
    cv_results_: dict[str, Any]
    acceleration_report_: AccelerationReport


def _path_batches(cv_plan):
    if cv_plan.kind != "mrc":
        return [cv_plan.folds]
    batches: list[list] = [[] for _ in range(cv_plan.n_paths)]
    for fold in cv_plan.folds:
        batches[fold.path_id].append(fold)
    return batches


def grid_search(estimator, X, param_grid, cv=None, *, y=None) -> GridSearchResult:
    """Select compact MeanRisk parameters with one shared CV/moment pass.

    Scores are mean out-of-sample path Sharpe ratios. All candidates must be
    eligible for the compact engine. For general estimator searches, use
    skfolio's ``OnlineGridSearch`` or sklearn's ``GridSearchCV``.
    """
    started = time.perf_counter()
    params = list(ParameterGrid(param_grid))
    if not params:
        raise ValueError("param_grid produced no candidates")

    candidates = []
    specs = []
    for candidate_params in params:
        candidate = clone(estimator).set_params(**candidate_params)
        reason = compact_blocked_reason(candidate, y=y)
        if reason is not None:
            raise ValueError(
                "grid_search only supports compact MeanRisk candidates; "
                f"{candidate_params!r} is unsupported: {reason}"
            )
        candidates.append(candidate)
        specs.append(estimator_spec(candidate))

    x_arr = as_float_2d(X)
    cv_plan = compile_cv_plan(cv, X, y)
    keep_returns = any(
        spec["risk_measure"] is not RiskMeasure.VARIANCE for spec in specs
    )
    fold_blocks = None
    if cv_plan.kind == "cpcv":
        fold_blocks = cpcv_fold_blocks(x_arr.shape[0], int(cv.n_folds))

    weights = [dict() for _ in candidates]
    engines = [EngineCache(spec=spec) for spec in specs]
    moments_s = 0.0
    solve_s = 0.0
    n_prior_fits = 0
    n_prior_updates = 0
    n_warm_starts = 0

    for path_index, folds in enumerate(_path_batches(cv_plan)):
        if path_index:
            # Clarabel has no explicit cold-start reset. Keep OSQP workspaces
            # across MRC paths, but rebuild CVaR engines at path boundaries.
            for candidate_id, spec in enumerate(specs):
                if spec["risk_measure"] not in {
                    RiskMeasure.VARIANCE,
                    RiskMeasure.SEMI_VARIANCE,
                }:
                    engines[candidate_id] = EngineCache(spec=spec)
        asset_idx = folds[0].asset_idx if folds else None
        if asset_idx is None:
            x_work = x_arr
            blocks = fold_blocks
        else:
            x_work = x_arr[:, asset_idx]
            blocks = None
        cache = OverlapMomentCache(
            x_work, keep_returns=keep_returns, fold_blocks=blocks
        )
        warm_before = [
            int(getattr(engine.engine, "n_warm_starts", 0)) for engine in engines
        ]

        for fold_index, fold in enumerate(folds):
            t0 = time.perf_counter()
            moments = cache.get(fold, path_key=fold.path_id)
            moments_s += time.perf_counter() - t0
            for candidate_id, (spec, engine_cache) in enumerate(
                zip(specs, engines, strict=True)
            ):
                observations = (
                    moments.n_observations
                    if spec["risk_measure"] is not RiskMeasure.VARIANCE
                    else None
                )
                engine = engine_cache.get(int(moments.mu.size), observations)
                t1 = time.perf_counter()
                weights[candidate_id][fold.fold_id] = engine.solve(
                    moments, warm=fold_index > 0
                )
                solve_s += time.perf_counter() - t1

        n_prior_fits += cache.n_fits
        n_prior_updates += cache.n_updates
        n_warm_starts += sum(
            int(getattr(engine.engine, "n_warm_starts", 0)) - before
            for engine, before in zip(engines, warm_before, strict=True)
        )

    t_eval = time.perf_counter()
    path_scores = np.vstack(
        [
            path_sharpes_from_weights(X, cv_plan, candidate_weights)
            for candidate_weights in weights
        ]
    )
    mean_scores = np.mean(path_scores, axis=1)
    best_index = int(np.nanargmax(mean_scores))
    best_prediction = assemble_prediction(
        X,
        cv_plan,
        weights[best_index],
        name=type(candidates[best_index]).__name__,
    )
    eval_s = time.perf_counter() - t_eval

    report = AccelerationReport(
        backend="compact-grid",
        n_solves=len(cv_plan.folds) * len(candidates),
        n_prior_fits=n_prior_fits,
        n_prior_updates=n_prior_updates,
        n_warm_starts=n_warm_starts,
        moments_s=moments_s,
        solve_s=solve_s,
        eval_s=eval_s,
        wall_s=time.perf_counter() - started,
    )
    return GridSearchResult(
        best_params_=params[best_index],
        best_score_=float(mean_scores[best_index]),
        best_index_=best_index,
        best_prediction_=best_prediction,
        cv_results_={
            "params": params,
            "mean_test_score": mean_scores,
            "path_scores": path_scores,
        },
        acceleration_report_=report,
    )
