"""Empirical moments from overlapping CV training windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skfolio_accelerate.cv_plan import FoldSpec

__all__ = [
    "FoldMoments",
    "OverlapMomentCache",
    "PathMomentSession",
    "empirical_from_stats",
    "empirical_from_window",
    "is_default_empirical",
    "path_moment_session",
]


def _contiguous_row_slice(rows: NDArray[np.intp]) -> slice | None:
    if rows.ndim != 1 or rows.size == 0:
        return None
    start, stop = int(rows[0]), int(rows[-1]) + 1
    if start < 0 or stop - start != rows.size:
        return None
    if rows.size > 1 and int(rows[1]) != start + 1:
        return None
    if rows.size > 2 and int(rows[rows.size // 2]) != start + rows.size // 2:
        return None
    return slice(start, stop)


@dataclass(slots=True)
class FoldMoments:
    mu: NDArray[np.float64]
    covariance: NDArray[np.float64]
    returns: NDArray[np.float64]
    n_observations: int = 0


def is_default_empirical(estimator) -> bool:
    prior = estimator.prior_estimator
    if prior is None:
        return True
    if type(prior).__name__ != "EmpiricalPrior":
        return False
    if (
        prior.is_log_normal
        or prior.investment_horizon is not None
        or prior.max_history is not None
    ):
        return False
    mu, cov = prior.mu_estimator, prior.covariance_estimator
    if mu is not None and (
        type(mu).__name__ != "EmpiricalMu" or mu.window_size is not None
    ):
        return False
    if cov is not None and (
        type(cov).__name__ != "EmpiricalCovariance"
        or cov.window_size is not None
        or int(cov.ddof) != 1
        or cov.assume_centered
    ):
        return False
    return True


def _finish_moments(mu, cov, returns, n_observations, *, keep_returns) -> FoldMoments:
    return FoldMoments(
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        covariance=(
            np.empty((0, 0), dtype=np.float64)
            if cov is None
            else np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
        ),
        returns=returns
        if keep_returns and returns is not None
        else np.empty((0, 0), dtype=np.float64),
        n_observations=int(n_observations),
    )


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
    return _finish_moments(
        mu, cov, window if keep_returns else None, t, keep_returns=keep_returns
    )


def empirical_from_stats(
    n_obs, sum_vec, gram, *, returns, keep_returns, ddof: int = 1
) -> FoldMoments:
    t = int(n_obs)
    mu = sum_vec / t
    cov = None if gram is None else (gram - np.outer(sum_vec, sum_vec) / t) / (t - ddof)
    return _finish_moments(mu, cov, returns, t, keep_returns=keep_returns)


@dataclass(slots=True)
class _BlockStats:
    n_obs: int
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64] | None


@dataclass(slots=True)
class _SlideState:
    start: int
    stop: int
    n_obs: int
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64] | None


@dataclass(slots=True)
class _IndexState:
    rows: NDArray[np.intp]
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64] | None


class OverlapMomentCache:
    def __init__(
        self, X, *, keep_returns, keep_covariance=True, ddof=1, fold_blocks=None
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.keep_returns = keep_returns
        self.keep_covariance = keep_covariance
        self.ddof = ddof
        self.n_fits = 0
        self.n_updates = 0
        self._slide: dict[int, _SlideState] = {}
        self._indexed: dict[int, _IndexState] = {}
        self._blocks = (
            [self._stats_from_rows(rows) for rows in fold_blocks]
            if fold_blocks
            else None
        )
        if self._blocks:
            self.n_fits += len(self._blocks)

    def _stats_from_rows(self, rows) -> _BlockStats:
        window = self.X[rows]
        return _BlockStats(
            n_obs=int(np.asarray(rows).size),
            sum_vec=window.sum(axis=0),
            gram=window.T @ window if self.keep_covariance else None,
        )

    def _shift(self, sum_vec, gram, lo, hi, sign):
        if lo >= hi:
            return sum_vec, gram
        block = self.X[lo:hi]
        sum_vec = sum_vec + sign * block.sum(axis=0)
        if gram is not None:
            gram = gram + sign * (block.T @ block)
        return sum_vec, gram

    def get(self, fold: FoldSpec, *, path_key: int = 0) -> FoldMoments:
        if self._blocks is not None and fold.train_block_ids:
            return self._from_blocks(fold)
        bounds = _contiguous_row_slice(fold.train_idx)
        if bounds is None or path_key in self._indexed:
            return self._from_index_rows(fold.train_idx, path_key)
        start, stop = bounds.start, bounds.stop
        prev = self._slide.get(path_key)
        if prev is None or prev.stop <= start or prev.start >= stop:
            window = self.X[start:stop]
            state = _SlideState(
                start=start,
                stop=stop,
                n_obs=stop - start,
                sum_vec=window.sum(axis=0),
                gram=window.T @ window if self.keep_covariance else None,
            )
            self._slide[path_key] = state
            self.n_fits += 1
            return empirical_from_stats(
                state.n_obs,
                state.sum_vec,
                state.gram,
                returns=window if self.keep_returns else None,
                keep_returns=self.keep_returns,
                ddof=self.ddof,
            )
        sum_vec, gram = prev.sum_vec, prev.gram
        if start > prev.start:
            sum_vec, gram = self._shift(sum_vec, gram, prev.start, start, -1)
        elif start < prev.start:
            sum_vec, gram = self._shift(sum_vec, gram, start, prev.start, 1)
        if stop > prev.stop:
            sum_vec, gram = self._shift(sum_vec, gram, prev.stop, stop, 1)
        elif stop < prev.stop:
            sum_vec, gram = self._shift(sum_vec, gram, stop, prev.stop, -1)
        self.n_updates += 1
        self._slide[path_key] = _SlideState(start, stop, stop - start, sum_vec, gram)
        return empirical_from_stats(
            stop - start,
            sum_vec,
            gram,
            returns=self.X[start:stop] if self.keep_returns else None,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )

    def _from_index_rows(self, rows, path_key) -> FoldMoments:
        previous = self._indexed.get(path_key)
        if previous is None:
            slide = self._slide.get(path_key)
            if slide is not None:
                previous = _IndexState(
                    rows=np.arange(slide.start, slide.stop, dtype=np.intp),
                    sum_vec=slide.sum_vec,
                    gram=slide.gram,
                )
        if previous is not None:
            removed = np.setdiff1d(previous.rows, rows, assume_unique=True)
            added = np.setdiff1d(rows, previous.rows, assume_unique=True)
            if removed.size + added.size < rows.size:
                sum_vec, gram = previous.sum_vec, previous.gram
                if removed.size:
                    values = self.X[removed]
                    sum_vec = sum_vec - values.sum(axis=0)
                    if gram is not None:
                        gram = gram - values.T @ values
                if added.size:
                    values = self.X[added]
                    sum_vec = sum_vec + values.sum(axis=0)
                    if gram is not None:
                        gram = gram + values.T @ values
                self._indexed[path_key] = _IndexState(rows, sum_vec, gram)
                self.n_updates += 1
                return empirical_from_stats(
                    int(rows.size),
                    sum_vec,
                    gram,
                    returns=self.X[rows] if self.keep_returns else None,
                    keep_returns=self.keep_returns,
                    ddof=self.ddof,
                )
        window = self.X[rows]
        state = _IndexState(
            rows,
            window.sum(axis=0),
            window.T @ window if self.keep_covariance else None,
        )
        self._indexed[path_key] = state
        self.n_fits += 1
        return empirical_from_stats(
            int(rows.size),
            state.sum_vec,
            state.gram,
            returns=window if self.keep_returns else None,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )

    def _from_blocks(self, fold: FoldSpec) -> FoldMoments:
        n_obs, sum_vec, gram = 0, None, None
        for block_id in fold.train_block_ids:
            block = self._blocks[block_id]
            n_obs += block.n_obs
            sum_vec = block.sum_vec if sum_vec is None else sum_vec + block.sum_vec
            if self.keep_covariance:
                gram = block.gram if gram is None else gram + block.gram
        if fold.train_excluded_idx.size:
            excluded = self.X[fold.train_excluded_idx]
            n_obs -= int(excluded.shape[0])
            sum_vec = sum_vec - excluded.sum(axis=0)
            if gram is not None:
                gram = gram - excluded.T @ excluded
        self.n_updates += 1
        return empirical_from_stats(
            n_obs,
            sum_vec,
            gram,
            returns=self.X[fold.train_idx] if self.keep_returns else None,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )


@dataclass(slots=True)
class PathMomentSession:
    cache: OverlapMomentCache
    x_work: NDArray[np.float64]

    def get(self, fold: FoldSpec) -> FoldMoments:
        return self.cache.get(fold, path_key=fold.path_id)


def path_moment_session(
    X, folds, *, keep_returns, keep_covariance=True, fold_blocks=None
) -> PathMomentSession:
    asset_idx = folds[0].asset_idx if folds else None
    if asset_idx is None:
        x_work, blocks = X, fold_blocks
    else:
        x_work, blocks = X[:, np.asarray(asset_idx, dtype=np.intp)], None
    return PathMomentSession(
        cache=OverlapMomentCache(
            x_work,
            keep_returns=keep_returns,
            keep_covariance=keep_covariance,
            fold_blocks=blocks,
        ),
        x_work=x_work,
    )
