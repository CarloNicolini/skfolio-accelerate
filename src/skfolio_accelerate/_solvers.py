"""Backend solvers that produce fold weights for amortized ``cross_val_predict``.

Does not import capability gates or the predict orchestrator. Portfolio assembly
lives in :mod:`skfolio_accelerate.scoring`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skfolio.optimization import InverseVolatility, Random
from skfolio.utils.stats import rand_weights_dirichlet
from skfolio.utils.tools import _check_method_params
from sklearn.base import clone

from skfolio_accelerate.compact import EngineCache, MeanRiskSpec
from skfolio_accelerate.cv_plan import FoldSpec
from skfolio_accelerate.mean_risk_problem import SequentialProblemCache
from skfolio_accelerate.moments import (
    PathMomentSession,
    is_default_empirical,
    path_moment_session,
)
from skfolio_accelerate.scoring import window_view

_PORTFOLIO_ATTRS = (
    "transaction_costs",
    "management_fees",
    "previous_weights",
    "risk_free_rate",
)


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
    This is the skip-``fit`` case of the shared serial assembly path.
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


def _train_target(y_arr: np.ndarray | None, fold: FoldSpec, n_assets: int):
    if y_arr is None:
        return None
    cols = fold.asset_idx
    if y_arr.ndim == 1 or cols is None or y_arr.shape[-1] != n_assets:
        return window_view(y_arr, fold.train_idx)
    return window_view(y_arr, fold.train_idx, cols)


def _fold_fit_params(X, params: dict | None, fold: FoldSpec) -> dict[str, Any]:
    """Slice routed fit metadata to this fold (same rules as skfolio)."""
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


def solve_sequential_folds(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: Sequence[FoldSpec],
    *,
    cache: SequentialProblemCache | None = None,
    path_id: int = 0,
    params: dict | None = None,
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

    params : dict, optional
        Routed fit metadata (e.g. ``factors``) sliced per fold.

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
    # Moment-session skip of prior.fit is only valid for default empirical priors.
    # Factor priors must be re-fit every fold; nonempty ``params`` also disables it.
    moment_session = (
        path_moment_session(
            x_arr,
            folds,
            keep_returns=True,
            keep_covariance=True,
        )
        if is_default_empirical(estimator) and not params
        else None
    )
    for fold in folds:
        started = time.perf_counter()
        reused = moment_session is not None and adapter.fit_from_moments(
            moment_session.get(fold)
        )
        if not reused:
            x_train = _train_slice(X, x_arr, fold)
            y_train = _train_target(y_arr, fold, n_assets)
            fit_params = _fold_fit_params(X, params, fold)
            if y_train is None:
                adapter.fit(x_train, **fit_params)
            else:
                adapter.fit(x_train, y_train, **fit_params)
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


def fit_native_weights(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: Sequence[FoldSpec],
    *,
    params: dict | None = None,
) -> FoldBatchResult:
    """Clone, native ``fit``, and collect 1-D ``weights_`` for each fold.

    Parameters
    ----------
    estimator : BaseOptimization
        Unfitted portfolio optimizer. Each fold receives a fresh
        :func:`~sklearn.base.clone`.

    X : array-like
        Full return matrix (DataFrame columns preserved when available).

    x_arr : ndarray of shape (n_observations, n_assets)
        Contiguous float64 returns.

    y_arr : ndarray of shape (n_observations,) or (n_observations, n_assets) or None
        Optional target passed to ``fit``.

    folds : sequence of FoldSpec
        Compiled train/test splits.

    params : dict, optional
        Routed fit metadata (e.g. ``factors``) sliced per fold.

    Returns
    -------
    result : FoldBatchResult
        One weight vector per fold. Two-dimensional ``weights_`` (efficient
        frontiers) raise :class:`ValueError` because assembly expects a single
        portfolio per fold.

    Notes
    -----
    This is the shared serial path for estimators outside the compact subset
    (HRP, risk budgeting, ratio objectives, ...). The same compiled CV plan
    and ``weights_`` assembly is used when ``fit`` is skipped. Portfolio
    objects are not built here; see
    :func:`~skfolio_accelerate.scoring.assemble_prediction`.
    """
    n_assets = int(x_arr.shape[1])
    weights: dict[int, NDArray[np.float64]] = {}
    solve_s = 0.0
    for fold in folds:
        started = time.perf_counter()
        fitted = clone(estimator)
        x_train = _train_slice(X, x_arr, fold)
        y_train = _train_target(y_arr, fold, n_assets)
        fit_params = _fold_fit_params(X, params, fold)
        if y_train is None:
            fitted.fit(x_train, **fit_params)
        else:
            fitted.fit(x_train, y_train, **fit_params)
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
