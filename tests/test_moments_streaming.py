"""Streaming / mergeable empirical moments."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from skfolio_accelerate.cv_plan import FoldSpec, compile_cv_plan
from skfolio_accelerate.moments import (
    OverlapMomentCache,
    merge_states,
    state_from_window,
    unmerge_state,
)
from tests.helpers import synthetic_returns


def _ref_cov(window: np.ndarray) -> np.ndarray:
    cov = np.cov(window, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return 0.5 * (cov + cov.T)


def test_state_from_window_matches_numpy():
    X = synthetic_returns(80, 6, seed=41)
    state = state_from_window(X)
    np.testing.assert_allclose(state.mu, X.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.covariance(), _ref_cov(X), rtol=1e-12, atol=1e-12)


def test_chan_merge_and_unmerge_match_numpy():
    X = synthetic_returns(90, 5, seed=42)
    left = state_from_window(X[:40])
    right = state_from_window(X[40:])
    merged = merge_states(left, right)
    np.testing.assert_allclose(merged.mu, X.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(merged.covariance(), _ref_cov(X), rtol=1e-12, atol=1e-12)

    recovered_right = unmerge_state(merged, left)
    np.testing.assert_allclose(recovered_right.mu, right.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered_right.m2, right.m2, rtol=1e-12, atol=1e-12)


def test_chan_merge_is_associative():
    X = synthetic_returns(60, 4, seed=43)
    a = state_from_window(X[:20])
    b = state_from_window(X[20:40])
    c = state_from_window(X[40:])
    left = merge_states(merge_states(a, b), c)
    right = merge_states(a, merge_states(b, c))
    np.testing.assert_allclose(left.mu, right.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(left.m2, right.m2, rtol=1e-12, atol=1e-12)


def test_welford_singleton_merge_matches_batch():
    X = synthetic_returns(40, 5, seed=44)
    state = state_from_window(X[:1])
    for row in X[1:]:
        state = merge_states(state, state_from_window(row.reshape(1, -1)))
    batch = state_from_window(X)
    np.testing.assert_allclose(state.mu, batch.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.m2, batch.m2, rtol=1e-10, atol=1e-12)


def test_prefix_cache_matches_numpy_and_avoids_refits():
    X = synthetic_returns(50, 6, seed=45)
    cache = OverlapMomentCache(X, keep_returns=False)
    for t in range(10, 50):
        fold = FoldSpec(
            fold_id=t,
            train_idx=np.arange(t, dtype=np.intp),
            test_idx=np.array([t], dtype=np.intp),
        )
        moments = cache.get(fold)
        np.testing.assert_allclose(
            moments.mu, X[:t].mean(axis=0), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            moments.covariance, _ref_cov(X[:t]), rtol=1e-12, atol=1e-12
        )
    assert cache.n_fits == 1
    assert cache.n_updates == 39


def test_sliding_window_cache_matches_numpy():
    X = synthetic_returns(60, 5, seed=46)
    width = 20
    cache = OverlapMomentCache(X, keep_returns=False)
    for stop in range(width, 60, 5):
        start = stop - width
        fold = FoldSpec(
            fold_id=stop,
            train_idx=np.arange(start, stop, dtype=np.intp),
            test_idx=np.array([stop % X.shape[0]], dtype=np.intp),
        )
        moments = cache.get(fold)
        window = X[start:stop]
        np.testing.assert_allclose(
            moments.mu, window.mean(axis=0), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            moments.covariance, _ref_cov(window), rtol=1e-12, atol=1e-12
        )
    assert cache.n_fits == 1
    assert cache.n_updates >= 1


def test_timeseries_split_reuses_expanding_prefixes():
    X = synthetic_returns(80, 4, seed=47)
    plan = compile_cv_plan(TimeSeriesSplit(n_splits=4), X)
    cache = OverlapMomentCache(X, keep_returns=False)
    for fold in plan.folds:
        moments = cache.get(fold)
        window = X[fold.train_idx]
        np.testing.assert_allclose(
            moments.mu, window.mean(axis=0), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            moments.covariance, _ref_cov(window), rtol=1e-12, atol=1e-12
        )
    assert cache.n_fits == 1
    assert cache.n_updates == 3


def test_centered_state_is_stable_for_large_means():
    rng = np.random.default_rng(48)
    noise = rng.normal(0.0, 0.01, size=(200, 6))
    X = 1_000.0 + noise
    state = state_from_window(X)
    np.testing.assert_allclose(state.covariance(), _ref_cov(X), rtol=1e-10, atol=1e-12)

    sum_vec = X.sum(axis=0)
    gram = X.T @ X
    uncentered = (gram - np.outer(sum_vec, sum_vec) / X.shape[0]) / (X.shape[0] - 1)
    assert np.max(np.abs(state.covariance() - _ref_cov(X))) < np.max(
        np.abs(uncentered - _ref_cov(X))
    )
