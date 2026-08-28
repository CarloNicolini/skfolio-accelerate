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

from skfolio_accelerate._arrays import as_float_2d, as_float_array
from skfolio_accelerate.compact import EngineCache, MeanRiskSpec, estimator_spec
from skfolio_accelerate.cv_plan import chains_previous_weights, compile_cv_plan
from skfolio_accelerate.mean_risk_problem import SequentialProblemCache
from skfolio_accelerate.moments import path_moment_session
from skfolio_accelerate.predict import (
    AccelerationReport,
    classify_call,
    solve_sequential_folds,
)
from skfolio_accelerate.scoring import (
    assemble_prediction,
    path_sharpes_from_weights,
)


@dataclass
class GridSearchResult:
    """Result of :func:`grid_search`.

    Attributes
    ----------
    best_params_ : dict
        Parameter combination with the highest mean path Sharpe.

    best_score_ : float
        Mean out-of-sample path Sharpe of ``best_params_``.

    best_index_ : int
        Index into ``cv_results_["params"]``.

    best_prediction_ : MultiPeriodPortfolio or Population
        Fully assembled prediction for the winning parameters only.

    cv_results_ : dict
        Search diagnostics with keys:

        * ``"params"`` — list of candidate parameter dicts,
        * ``"mean_test_score"`` — mean path Sharpe per candidate,
        * ``"path_scores"`` — ndarray of shape ``(n_candidates, n_paths)``.

    acceleration_report_ : AccelerationReport
        Timing and warm-start accounting for the shared compact-grid pass.
    """

    best_params_: dict[str, Any]
    best_score_: float
    best_index_: int
    best_prediction_: Any
    cv_results_: dict[str, Any]
    acceleration_report_: AccelerationReport


def _classify_grid(
    estimator, param_grid, *, y=None, cv=None
) -> tuple[list[dict[str, Any]], list[MeanRiskSpec | None], str]:
    params = list(ParameterGrid(param_grid))
    if not params:
        raise ValueError("param_grid produced no candidates")
    specs: list[MeanRiskSpec | None] = []
    compact_ok = True
    sequential_ok = True
    first_failure: tuple[dict[str, Any], str] | None = None
    for candidate_params in params:
        candidate = clone(estimator).set_params(**candidate_params)
        caps = classify_call(candidate, y=y, cv=cv)
        if caps.can_compact:
            specs.append(estimator_spec(candidate))
        else:
            compact_ok = False
            specs.append(None)
        if not caps.can_sequential:
            sequential_ok = False
            if first_failure is None:
                first_failure = (
                    candidate_params,
                    caps.sequential_reason or "unsupported MeanRisk",
                )
    if compact_ok:
        return params, specs, "compact"
    if sequential_ok:
        return params, specs, "sequential"
    if first_failure is None:
        raise ValueError("grid_search produced no supported MeanRisk candidates")
    failed_params, reason = first_failure
    raise ValueError(
        "grid_search only supports compact or sequential MeanRisk candidates; "
        f"{failed_params!r} is unsupported: {reason}"
    )


def grid_search(estimator, X, param_grid, cv=None, *, y=None) -> GridSearchResult:
    """Select MeanRisk parameters with one shared CV plan.

    Compact-eligible grids reuse OSQP / Clarabel engines and empirical moments.
    Other MeanRisk grids (ratio objectives, risk limits, linear constraints,
    ...) reuse Parameterized CVXPY problems. Scores are mean out-of-sample
    path Sharpe ratios computed from fold weights. Only the winning parameter
    set is materialized.

    Parameters
    ----------
    estimator : MeanRisk
        Base estimator. Each grid point is applied via
        :meth:`~sklearn.base.BaseEstimator.set_params` on a clone.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.

    param_grid : dict or list of dict
        Parameter grid as accepted by
        :class:`~sklearn.model_selection.ParameterGrid`.

    cv : int, cross-validation generator or an iterable, default=None
        Cross-validation splitting strategy. Compiled once and shared across
        all candidates.

    y : array-like, optional
        Target relative to ``X`` for API compatibility.

    Returns
    -------
    result : GridSearchResult
        Best parameters, scores, assembled prediction, and acceleration report.

    Raises
    ------
    ValueError
        If the grid is empty or any candidate is outside compact and sequential
        MeanRisk support.

    Notes
    -----
    For general estimators (pipelines, HRP, ...), use skfolio's
    ``OnlineGridSearch`` or sklearn's ``GridSearchCV``.
    """
    started = time.perf_counter()
    params, specs, kind = _classify_grid(estimator, param_grid, y=y, cv=cv)

    x_arr = as_float_2d(X)
    cv_plan = compile_cv_plan(cv, X, y)

    if kind == "sequential":
        return _sequential_grid_search(
            estimator,
            X,
            x_arr,
            y,
            params,
            cv_plan,
            started,
        )

    compact_specs = [spec for spec in specs if spec is not None]
    keep_returns = any(spec.needs_returns() for spec in compact_specs)
    specs = compact_specs

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
        reason="boxed MeanRisk grid; shared compact engines",
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


def _sequential_grid_search(
    estimator,
    X,
    x_arr,
    y,
    params: list[dict[str, Any]],
    cv_plan,
    started: float,
) -> GridSearchResult:
    y_arr = None if y is None else as_float_array(y)
    weights: list[dict[int, np.ndarray]] = [dict() for _ in params]
    caches = [
        SequentialProblemCache(clone(estimator).set_params(**candidate_params))
        for candidate_params in params
    ]
    solve_s = 0.0
    n_warm_starts = 0
    n_rebuilds = 0
    is_dpp: bool | None = None
    for path_index, folds in enumerate(cv_plan.path_batches()):
        for candidate_id, cache in enumerate(caches):
            result = solve_sequential_folds(
                estimator,
                X,
                x_arr,
                y_arr,
                folds,
                cache=cache,
                path_id=path_index,
                chain_previous_weights=chains_previous_weights(cv_plan),
            )
            weights[candidate_id].update(result.weights)
            solve_s += result.solve_s
            n_warm_starts += result.n_warm_starts
            n_rebuilds += result.n_rebuilds
            if result.is_dpp is not None:
                is_dpp = (
                    result.is_dpp if is_dpp is None else bool(is_dpp and result.is_dpp)
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
        backend="sequential-grid",
        reason="MeanRisk grid outside the compact subset",
        n_solves=len(cv_plan.folds) * len(params),
        n_warm_starts=n_warm_starts,
        n_rebuilds=n_rebuilds,
        is_dpp=is_dpp,
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
