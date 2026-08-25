"""Fit skfolio priors once per fold, plus overlapping-window empirical moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from skfolio.prior import EmpiricalPrior
from skfolio.prior._base import ReturnDistribution

from skfolio_accelerate.ir import FoldSpec


@dataclass
class FoldMoments:
    mu: NDArray[np.float64]
    covariance: NDArray[np.float64]
    cholesky: NDArray[np.float64]
    returns: NDArray[np.float64]
    sample_weight: NDArray[np.float64] | None = None
    n_observations: int = 0


def _as_float(x: Any) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def as_float_2d(X: Any) -> NDArray[np.float64]:
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(copy=False)
    else:
        arr = np.asarray(X)
    return np.ascontiguousarray(arr, dtype=np.float64)


def moments_from_distribution(
    dist: ReturnDistribution, *, keep_returns: bool = True
) -> FoldMoments:
    cov = _as_float(dist.covariance)
    chol = getattr(dist, "cholesky", None)
    if chol is None:
        chol = np.linalg.cholesky(cov)
    else:
        chol = _as_float(chol)
    sample_weight = getattr(dist, "sample_weight", None)
    if sample_weight is not None:
        sample_weight = _as_float(sample_weight)
    n_observations = int(np.asarray(dist.returns).shape[0])
    returns = (
        _as_float(dist.returns)
        if keep_returns
        else np.empty((0, 0), dtype=np.float64)
    )
    return FoldMoments(
        mu=_as_float(dist.mu),
        covariance=cov,
        cholesky=chol,
        returns=returns,
        sample_weight=sample_weight,
        n_observations=n_observations,
    )


def fit_prior(estimator, X_train, y_train=None, *, keep_returns: bool = True) -> FoldMoments:
    """Fit the estimator's prior on a training window and return moments."""
    prior = getattr(estimator, "prior_estimator", None)
    if prior is None:
        prior = EmpiricalPrior()
    else:
        prior = clone(prior)
    if y_train is None:
        prior.fit(X_train)
    else:
        prior.fit(X_train, y_train)
    return moments_from_distribution(
        prior.return_distribution_, keep_returns=keep_returns
    )


def is_default_empirical(estimator) -> bool:
    """True when moments are sample mean + sample covariance (ddof=1)."""
    prior = getattr(estimator, "prior_estimator", None)
    if prior is None:
        return True
    if type(prior).__name__ != "EmpiricalPrior":
        return False
    if getattr(prior, "is_log_normal", False):
        return False
    if getattr(prior, "investment_horizon", None) is not None:
        return False
    mu = getattr(prior, "mu_estimator", None)
    cov = getattr(prior, "covariance_estimator", None)
    if mu is not None and type(mu).__name__ != "EmpiricalMu":
        return False
    if cov is not None and type(cov).__name__ != "EmpiricalCovariance":
        return False
    if mu is not None and getattr(mu, "window_size", None) is not None:
        return False
    if cov is not None and getattr(cov, "window_size", None) is not None:
        return False
    if cov is not None and int(getattr(cov, "ddof", 1)) != 1:
        return False
    return True


def empirical_from_window(
    window: NDArray[np.float64], *, keep_returns: bool, ddof: int = 1
) -> FoldMoments:
    t, n = window.shape
    mu = np.mean(window, axis=0)
    if n == 1:
        cov = np.var(window, axis=0, ddof=ddof).reshape(1, 1)
    else:
        cov = np.cov(window, rowvar=False, ddof=ddof)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    cov = np.ascontiguousarray(cov, dtype=np.float64)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.eye(n)
        cov = cov + jitter
        chol = np.linalg.cholesky(cov)
    returns = window if keep_returns else np.empty((0, 0), dtype=np.float64)
    return FoldMoments(
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        covariance=cov,
        cholesky=np.ascontiguousarray(chol, dtype=np.float64),
        returns=returns,
        n_observations=t,
    )


def empirical_from_stats(
    n_obs: int,
    sum_vec: NDArray[np.float64],
    gram: NDArray[np.float64],
    *,
    returns: NDArray[np.float64] | None,
    keep_returns: bool,
    ddof: int = 1,
) -> FoldMoments:
    t = int(n_obs)
    mu = sum_vec / t
    cov = (gram - np.outer(sum_vec, sum_vec) / t) / (t - ddof)
    cov = np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
    n = cov.shape[0]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-12 * np.eye(n)
        chol = np.linalg.cholesky(cov)
    ret = (
        returns
        if keep_returns and returns is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    return FoldMoments(
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        covariance=cov,
        cholesky=np.ascontiguousarray(chol, dtype=np.float64),
        returns=ret,
        n_observations=t,
    )


class FoldCache:
    """Cache fitted priors keyed by (fold_id, data_param_fingerprint)."""

    def __init__(self, *, keep_returns: bool = True) -> None:
        self._store: dict[tuple[int, str], FoldMoments] = {}
        self.n_fits = 0
        self.keep_returns = keep_returns

    def get(
        self,
        fold_id: int,
        estimator,
        X_train,
        y_train=None,
        data_key: str = "",
    ) -> FoldMoments:
        key = (fold_id, data_key)
        cached = self._store.get(key)
        if cached is not None:
            return cached
        moments = fit_prior(
            estimator, X_train, y_train, keep_returns=self.keep_returns
        )
        self._store[key] = moments
        self.n_fits += 1
        return moments


def _contiguous_bounds(idx: NDArray[np.intp]) -> tuple[int, int] | None:
    if idx.ndim != 1 or idx.size == 0:
        return None
    start = int(idx[0])
    stop = int(idx[-1]) + 1
    if stop - start != idx.size or start < 0:
        return None
    if idx.size > 2 and int(idx[idx.size // 2]) != start + idx.size // 2:
        return None
    if idx.size > 1 and int(idx[1]) != start + 1:
        return None
    return start, stop


@dataclass
class _BlockStats:
    n_obs: int
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64]
    rows: NDArray[np.intp]


@dataclass
class _SlideState:
    start: int
    stop: int
    n_obs: int
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64]


class OverlapMomentCache:
    """Empirical moments with sliding-window and CPCV fold-block reuse.

    ``n_fits`` counts cold Gram computations. ``n_updates`` counts rank-k slides
    or block additions that avoid a full ``X.T @ X`` on the train window.
    """

    def __init__(
        self,
        X: NDArray[np.float64],
        *,
        keep_returns: bool,
        ddof: int = 1,
        fold_blocks: list[NDArray[np.intp]] | None = None,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.keep_returns = keep_returns
        self.ddof = ddof
        self.n_fits = 0
        self.n_updates = 0
        self._slide: dict[tuple[int, tuple[int, ...]], _SlideState] = {}
        self._blocks: list[_BlockStats] | None = None
        if fold_blocks:
            self._blocks = [self._stats_from_rows(rows) for rows in fold_blocks]
            self.n_fits += len(self._blocks)

    def _stats_from_rows(self, rows: NDArray[np.intp]) -> _BlockStats:
        window = self.X[rows]
        return _BlockStats(
            n_obs=int(rows.size),
            sum_vec=window.sum(axis=0),
            gram=window.T @ window,
            rows=np.asarray(rows, dtype=np.intp),
        )

    def get(
        self,
        fold: FoldSpec,
        *,
        path_key: int = 0,
        asset_idx: NDArray[np.intp] | None = None,
    ) -> FoldMoments:
        cols = tuple(int(v) for v in asset_idx) if asset_idx is not None else ()
        x = self.X if not cols else self.X[:, np.asarray(cols, dtype=np.intp)]
        if self._blocks is not None and fold.train_block_ids:
            return self._from_blocks(fold, x if cols else self.X, cols)

        bounds = _contiguous_bounds(fold.train_idx)
        if bounds is None:
            window = x[fold.train_idx]
            self.n_fits += 1
            return empirical_from_window(window, keep_returns=self.keep_returns, ddof=self.ddof)

        start, stop = bounds
        key = (path_key, cols)
        prev = self._slide.get(key)
        if prev is None or prev.stop <= start or prev.start >= stop:
            window = x[start:stop]
            state = _SlideState(
                start=start,
                stop=stop,
                n_obs=stop - start,
                sum_vec=window.sum(axis=0),
                gram=window.T @ window,
            )
            self._slide[key] = state
            self.n_fits += 1
            returns = window if self.keep_returns else None
            return empirical_from_stats(
                state.n_obs,
                state.sum_vec,
                state.gram,
                returns=returns,
                keep_returns=self.keep_returns,
                ddof=self.ddof,
            )

        gram = prev.gram
        sum_vec = prev.sum_vec
        if start > prev.start:
            drop = x[prev.start : start]
            gram = gram - drop.T @ drop
            sum_vec = sum_vec - drop.sum(axis=0)
        elif start < prev.start:
            add = x[start : prev.start]
            gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        if stop > prev.stop:
            add = x[prev.stop : stop]
            gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        elif stop < prev.stop:
            drop = x[stop : prev.stop]
            gram = gram - drop.T @ drop
            sum_vec = sum_vec - drop.sum(axis=0)
        self.n_updates += 1
        state = _SlideState(
            start=start,
            stop=stop,
            n_obs=stop - start,
            sum_vec=sum_vec,
            gram=gram,
        )
        self._slide[key] = state
        returns = x[start:stop] if self.keep_returns else None
        return empirical_from_stats(
            state.n_obs,
            state.sum_vec,
            state.gram,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )

    def _from_blocks(
        self,
        fold: FoldSpec,
        x: NDArray[np.float64],
        cols: tuple[int, ...],
    ) -> FoldMoments:
        assert self._blocks is not None
        n_obs = 0
        sum_vec = None
        gram = None
        row_parts: list[NDArray[np.intp]] = []
        for block_id in fold.train_block_ids:
            block = self._blocks[block_id]
            n_obs += block.n_obs
            if cols:
                idx = np.asarray(cols, dtype=np.intp)
                sv = block.sum_vec[idx]
                g = block.gram[np.ix_(idx, idx)]
            else:
                sv = block.sum_vec
                g = block.gram
            sum_vec = sv if sum_vec is None else sum_vec + sv
            gram = g if gram is None else gram + g
            row_parts.append(block.rows)
        self.n_updates += 1
        returns = None
        if self.keep_returns:
            rows = np.concatenate(row_parts)
            returns = x[rows] if cols else self.X[rows]
        return empirical_from_stats(
            n_obs,
            sum_vec,
            gram,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )
