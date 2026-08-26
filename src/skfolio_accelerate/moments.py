"""Empirical moments with sliding-window and CPCV fold-block reuse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skfolio_accelerate.cv_plan import FoldSpec


@dataclass
class FoldMoments:
    mu: NDArray[np.float64]
    covariance: NDArray[np.float64]
    returns: NDArray[np.float64]
    n_observations: int = 0


def as_float_2d(X) -> NDArray[np.float64]:
    arr = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
    return np.ascontiguousarray(arr, dtype=np.float64)


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


def _pack(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    returns: NDArray[np.float64] | None,
    n_observations: int,
    *,
    keep_returns: bool,
) -> FoldMoments:
    cov = np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
    ret = (
        returns
        if keep_returns and returns is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    return FoldMoments(
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        covariance=cov,
        returns=ret,
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
    returns = window if keep_returns else None
    return _pack(mu, cov, returns, t, keep_returns=keep_returns)


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
    return _pack(mu, cov, returns, t, keep_returns=keep_returns)


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


@dataclass
class _SlideState:
    start: int
    stop: int
    n_obs: int
    sum_vec: NDArray[np.float64]
    gram: NDArray[np.float64]


@dataclass
class _IndexState:
    rows: NDArray[np.intp]
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
        self._slide: dict[int, _SlideState] = {}
        self._indexed: dict[int, _IndexState] = {}
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
        )

    def get(self, fold: FoldSpec, *, path_key: int = 0) -> FoldMoments:
        if self._blocks is not None and fold.train_block_ids:
            return self._from_blocks(fold)

        bounds = _contiguous_bounds(fold.train_idx)
        if bounds is None or path_key in self._indexed:
            return self._from_index_rows(fold.train_idx, path_key)

        start, stop = bounds
        prev = self._slide.get(path_key)
        if prev is None or prev.stop <= start or prev.start >= stop:
            window = self.X[start:stop]
            state = _SlideState(
                start=start,
                stop=stop,
                n_obs=stop - start,
                sum_vec=window.sum(axis=0),
                gram=window.T @ window,
            )
            self._slide[path_key] = state
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
            drop = self.X[prev.start : start]
            gram = gram - drop.T @ drop
            sum_vec = sum_vec - drop.sum(axis=0)
        elif start < prev.start:
            add = self.X[start : prev.start]
            gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        if stop > prev.stop:
            add = self.X[prev.stop : stop]
            gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        elif stop < prev.stop:
            drop = self.X[stop : prev.stop]
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
        self._slide[path_key] = state
        returns = self.X[start:stop] if self.keep_returns else None
        return empirical_from_stats(
            state.n_obs,
            state.sum_vec,
            state.gram,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )

    def _from_index_rows(
        self,
        rows: NDArray[np.intp],
        path_key: int,
    ) -> FoldMoments:
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
                sum_vec = previous.sum_vec
                gram = previous.gram
                if removed.size:
                    values = self.X[removed]
                    sum_vec = sum_vec - values.sum(axis=0)
                    gram = gram - values.T @ values
                if added.size:
                    values = self.X[added]
                    sum_vec = sum_vec + values.sum(axis=0)
                    gram = gram + values.T @ values
                state = _IndexState(
                    rows=rows,
                    sum_vec=sum_vec,
                    gram=gram,
                )
                self._indexed[path_key] = state
                self.n_updates += 1
                returns = self.X[rows] if self.keep_returns else None
                return empirical_from_stats(
                    int(rows.size),
                    sum_vec,
                    gram,
                    returns=returns,
                    keep_returns=self.keep_returns,
                    ddof=self.ddof,
                )

        window = self.X[rows]
        state = _IndexState(
            rows=rows,
            sum_vec=window.sum(axis=0),
            gram=window.T @ window,
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
        assert self._blocks is not None
        n_obs = 0
        sum_vec = None
        gram = None
        for block_id in fold.train_block_ids:
            block = self._blocks[block_id]
            n_obs += block.n_obs
            sum_vec = block.sum_vec if sum_vec is None else sum_vec + block.sum_vec
            gram = block.gram if gram is None else gram + block.gram
        if fold.train_excluded_idx.size:
            excluded = self.X[fold.train_excluded_idx]
            n_obs -= int(excluded.shape[0])
            sum_vec = sum_vec - excluded.sum(axis=0)
            gram = gram - excluded.T @ excluded
        self.n_updates += 1
        returns = None
        if self.keep_returns:
            returns = self.X[fold.train_idx]
        return empirical_from_stats(
            n_obs,
            sum_vec,
            gram,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )
