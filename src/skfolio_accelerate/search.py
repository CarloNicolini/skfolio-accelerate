"""Hyperparameter search that shares CV plans and empirical moments.

Candidate evaluation scores compact MeanRisk grids from fold weights (mean
out-of-sample path Sharpe). Only the winning parameter set is materialized
into Portfolio objects. For estimators outside the compact MeanRisk subset,
use skfolio's ``OnlineGridSearch`` or sklearn's ``GridSearchCV``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from skfolio_accelerate._arrays import as_float_2d
from skfolio_accelerate.compact import EngineCache, MeanRiskSpec, estimator_spec
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.moments import path_moment_session
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


def _candidate_specs(
    estimator, param_grid, *, y=None, cv=None
) -> tuple[
    list[dict[str, Any]],
    list[MeanRiskSpec],
]:
    params = list(ParameterGrid(param_grid))
    if not params:
        raise ValueError("param_grid produced no candidates")
    specs: list[MeanRiskSpec] = []
    for candidate_params in params:
        candidate = clone(estimator).set_params(**candidate_params)
        reason = compact_blocked_reason(candidate, y=y, cv=cv)
        if reason is not None:
            raise ValueError(
                "grid_search only supports compact MeanRisk candidates; "
                f"{candidate_params!r} is unsupported: {reason}"
            )
        specs.append(estimator_spec(candidate))
    return params, specs


def grid_search(estimator, X, param_grid, cv=None, *, y=None) -> GridSearchResult:
    """Select compact MeanRisk parameters with one shared CV/moment pass.

    Scores are mean out-of-sample path Sharpe ratios. All candidates must be
    eligible for the compact engine. For general estimator searches, use
    skfolio's ``OnlineGridSearch`` or sklearn's ``GridSearchCV``.
    """
    started = time.perf_counter()
    params, specs = _candidate_specs(estimator, param_grid, y=y, cv=cv)

    x_arr = as_float_2d(X)
    cv_plan = compile_cv_plan(cv, X, y)
    keep_returns = any(spec.needs_returns() for spec in specs)

    weights: list[dict[int, np.ndarray]] = [dict() for _ in specs]
    engines = [EngineCache(spec=spec) for spec in specs]
    moments_s = 0.0
    solve_s = 0.0
    n_prior_fits = 0
    n_prior_updates = 0
    n_warm_starts = 0

    for path_index, folds in enumerate(cv_plan.path_batches()):
        if path_index:
            # Clarabel has no explicit cold-start reset. Keep OSQP workspaces
            # across MRC paths, but rebuild scenario engines at path boundaries.
            for candidate_id, spec in enumerate(specs):
                if spec.needs_returns():
                    engines[candidate_id] = EngineCache(spec=spec)
        session = path_moment_session(
            x_arr,
            folds,
            keep_returns=keep_returns,
            fold_blocks=cv_plan.fold_blocks,
        )
        warm_before = [
            int(getattr(engine.engine, "n_warm_starts", 0)) for engine in engines
        ]

        for fold_index, fold in enumerate(folds):
            t0 = time.perf_counter()
            moments = session.get(fold)
            moments_s += time.perf_counter() - t0
            for candidate_id, (spec, engine_cache) in enumerate(
                zip(specs, engines, strict=True)
            ):
                observations = moments.n_observations if spec.needs_returns() else None
                engine = engine_cache.get(int(moments.mu.size), observations)
                t1 = time.perf_counter()
                weights[candidate_id][fold.fold_id] = engine.solve(
                    moments, warm=fold_index > 0
                )
                solve_s += time.perf_counter() - t1

        n_prior_fits += session.cache.n_fits
        n_prior_updates += session.cache.n_updates
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
        name=type(estimator).__name__,
    )
    eval_s = time.perf_counter() - t_eval

    report = AccelerationReport(
        backend="compact-grid",
        n_solves=len(cv_plan.folds) * len(specs),
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
