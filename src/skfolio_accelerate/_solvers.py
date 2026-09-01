"""Backend solvers that produce fold weights."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skfolio.optimization import InverseVolatility, Random
from skfolio.utils.stats import rand_weights_dirichlet
from skfolio.utils.tools import _check_method_params
from sklearn.base import clone

from skfolio_accelerate.compact import EngineCache, MeanRiskSpec
from skfolio_accelerate.cv_plan import FoldSpec
from skfolio_accelerate.mean_risk_problem import SequentialProblemCache
from skfolio_accelerate.moments import is_default_empirical, path_moment_session
from skfolio_accelerate.scoring import window_view

_PORTFOLIO_ATTRS = (
    "transaction_costs",
    "management_fees",
    "previous_weights",
    "risk_free_rate",
)


@dataclass(slots=True)
class FoldBatchResult:
    weights: dict[int, NDArray[np.float64]]
    moments_s: float = 0.0
    solve_s: float = 0.0
    n_solves: int = 0
    n_warm_starts: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_rebuilds: int = 0
    is_dpp: bool | None = None


def merge_batch_results(parts: Sequence[FoldBatchResult]) -> FoldBatchResult:
    weights: dict[int, NDArray[np.float64]] = {}
    moments_s = solve_s = 0.0
    n_solves = n_warm = n_fits = n_updates = n_rebuilds = 0
    is_dpp = None
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
        weights,
        moments_s,
        solve_s,
        n_solves,
        n_warm,
        n_fits,
        n_updates,
        n_rebuilds,
        is_dpp,
    )


def solve_compact_folds(
    session, folds, spec: MeanRiskSpec, *, engines=None
) -> FoldBatchResult:
    engines = EngineCache(spec=spec) if engines is None else engines
    weights = {}
    moments_s = solve_s = 0.0
    warm_before, engine = 0, None
    names = spec.asset_names
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        moments = session.get(fold)
        moments_s += time.perf_counter() - t0
        fold_names = names
        if names is not None and fold.asset_idx is not None:
            fold_names = tuple(names[int(j)] for j in fold.asset_idx)
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if spec.needs_returns() else None,
            names=fold_names,
        )
        if i == 0:
            warm_before = engine.n_warm_starts
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
    return FoldBatchResult(
        weights,
        moments_s,
        solve_s,
        len(folds),
        0 if engine is None else engine.n_warm_starts - warm_before,
        int(session.cache.n_fits),
        int(session.cache.n_updates),
    )


def closed_form_weights(X, folds, estimator, *, fold_blocks=None) -> FoldBatchResult:
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
    weights, moments_s = {}, 0.0
    draw_random = type(estimator) is Random
    for fold in folds:
        if session is None:
            weights[fold.fold_id] = (
                rand_weights_dirichlet(n=n_assets)
                if draw_random
                else np.full(n_assets, 1.0 / n_assets)
            )
            continue
        started = time.perf_counter()
        moments = session.get(fold)
        moments_s += time.perf_counter() - started
        inv = 1.0 / np.sqrt(np.diag(moments.covariance))
        weights[fold.fold_id] = inv / inv.sum()
    return FoldBatchResult(
        weights,
        moments_s,
        n_prior_fits=0 if session is None else session.cache.n_fits,
        n_prior_updates=0 if session is None else session.cache.n_updates,
    )


def _segment_params(estimator) -> dict:
    extra = dict(estimator.portfolio_params or {})
    state = estimator.__dict__
    for name in _PORTFOLIO_ATTRS:
        if name not in extra and name in state:
            extra[name] = state[name]
    extra.pop("name", None)
    extra.pop("check_observations_order", None)
    extra.pop("fallback_chain", None)
    return extra


def _train_slice(X, x_arr, fold: FoldSpec):
    try:
        rows = X.iloc[np.asarray(fold.train_idx)]
    except AttributeError:
        return window_view(x_arr, fold.train_idx, fold.asset_idx)
    return (
        rows.iloc[:, np.asarray(fold.asset_idx)] if fold.asset_idx is not None else rows
    )


def _train_target(y_arr, fold: FoldSpec, n_assets: int):
    if y_arr is None:
        return None
    cols = fold.asset_idx
    if y_arr.ndim == 1 or cols is None or y_arr.shape[-1] != n_assets:
        return window_view(y_arr, fold.train_idx)
    return window_view(y_arr, fold.train_idx, cols)


def _fold_fit_params(X, params, fold: FoldSpec) -> dict:
    if not params:
        return {}
    fit_params = dict(params)
    if fold.asset_idx is not None:
        fit_params = _check_method_params(
            X, params=fit_params, indices=np.asarray(fold.asset_idx), axis=1
        )
    return _check_method_params(
        X, params=fit_params, indices=np.asarray(fold.train_idx), axis=0
    )


def _fit_weights(fitted, X, x_arr, y_arr, fold, n_assets, params):
    x_train = _train_slice(X, x_arr, fold)
    y_train = _train_target(y_arr, fold, n_assets)
    fit_params = _fold_fit_params(X, params, fold)
    if y_train is None:
        fitted.fit(x_train, **fit_params)
    else:
        fitted.fit(x_train, y_train, **fit_params)
    weights = np.asarray(fitted.weights_, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("2-dimensional weights_ cannot be assembled")
    return np.ascontiguousarray(weights)


def solve_sequential_folds(
    estimator, X, x_arr, y_arr, folds, *, cache=None, path_id=0, params=None
):
    cache = SequentialProblemCache(estimator) if cache is None else cache
    adapter = cache.get(path_id)
    warm_before, rebuild_before = adapter.n_warm_starts, adapter.n_rebuilds
    weights, solve_s, n_assets = {}, 0.0, int(x_arr.shape[1])
    moment_session = (
        path_moment_session(x_arr, folds, keep_returns=True, keep_covariance=True)
        if is_default_empirical(estimator) and not params
        else None
    )
    for fold in folds:
        started = time.perf_counter()
        reused = moment_session is not None and adapter.fit_from_moments(
            moment_session.get(fold)
        )
        if reused:
            fitted = np.asarray(adapter.weights_, dtype=np.float64)
            if fitted.ndim != 1:
                raise ValueError("2-dimensional weights_ cannot be assembled")
            weights[fold.fold_id] = np.ascontiguousarray(fitted)
        else:
            weights[fold.fold_id] = _fit_weights(
                adapter, X, x_arr, y_arr, fold, n_assets, params
            )
        solve_s += time.perf_counter() - started
    return FoldBatchResult(
        weights,
        solve_s=solve_s,
        n_solves=len(folds),
        n_warm_starts=adapter.n_warm_starts - warm_before,
        n_rebuilds=adapter.n_rebuilds - rebuild_before,
        is_dpp=adapter.is_dpp_,
    )


def fit_native_weights(estimator, X, x_arr, y_arr, folds, *, params=None):
    n_assets, weights, solve_s = int(x_arr.shape[1]), {}, 0.0
    for fold in folds:
        started = time.perf_counter()
        weights[fold.fold_id] = _fit_weights(
            clone(estimator), X, x_arr, y_arr, fold, n_assets, params
        )
        solve_s += time.perf_counter() - started
    return FoldBatchResult(weights, solve_s=solve_s, n_solves=len(folds))
