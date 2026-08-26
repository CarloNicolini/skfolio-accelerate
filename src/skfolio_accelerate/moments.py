"""Empirical moments with mergeable sufficient statistics.

The cache stores Chan/Welford state ``(n, μ, M₂)`` rather than covariance
matrices. Adjacent walk-forward windows, KFold overlaps, and CPCV fold blocks
are then merged or unmerged in ``O(d²)`` (plus a BLAS Gram of the rows that
actually arrived or left). Prefix covariances cost ``O(T d²)`` total instead of
recomputing each ``X[:t]`` from scratch.

``M₂`` is the sum of centered outer products. Sample covariance is
``M₂ / (n - ddof)``. Block construction uses a two-pass Gram; combining blocks
uses Chan's formula, which reduces to Welford's rank-one update when a block
has one row.
"""

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


@dataclass
class MomentState:
    """Mergeable mean / second-moment summary of a set of rows.

    ``m2`` is ``Σ_i (x_i - μ)(x_i - μ)ᵀ``. Two disjoint summaries combine with
    Chan's formula; removing a subset is the same formula run backwards.
    """

    n_obs: int
    mu: NDArray[np.float64]
    m2: NDArray[np.float64]

    def copy(self) -> MomentState:
        return MomentState(
            n_obs=int(self.n_obs),
            mu=np.array(self.mu, dtype=np.float64, copy=True),
            m2=np.array(self.m2, dtype=np.float64, copy=True),
        )

    def covariance(self, ddof: int = 1) -> NDArray[np.float64]:
        denom = int(self.n_obs) - int(ddof)
        if denom <= 0:
            raise ValueError(
                f"need n_obs > ddof to form covariance, got n_obs={self.n_obs} "
                f"ddof={ddof}"
            )
        cov = self.m2 / denom
        return np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)


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


def empty_state(n_assets: int) -> MomentState:
    d = int(n_assets)
    return MomentState(
        n_obs=0,
        mu=np.zeros(d, dtype=np.float64),
        m2=np.zeros((d, d), dtype=np.float64),
    )


def state_from_window(window: NDArray[np.float64]) -> MomentState:
    """Two-pass ``(n, μ, M₂)`` for a block of rows."""
    data = np.ascontiguousarray(window, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_obs, n_assets = data.shape
    if n_obs == 0:
        return empty_state(n_assets)
    mu = data.mean(axis=0)
    centered = data - mu
    m2 = centered.T @ centered
    if m2.ndim == 0:
        m2 = np.asarray(m2, dtype=np.float64).reshape(1, 1)
    return MomentState(
        n_obs=int(n_obs),
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        m2=np.ascontiguousarray(m2, dtype=np.float64),
    )


def merge_states(left: MomentState, right: MomentState) -> MomentState:
    """Chan merge of two disjoint summaries.

    For a singleton ``right`` this is Welford's rank-one update.
    """
    if left.n_obs == 0:
        return right.copy()
    if right.n_obs == 0:
        return left.copy()
    n_left = int(left.n_obs)
    n_right = int(right.n_obs)
    n_obs = n_left + n_right
    mu = (n_left * left.mu + n_right * right.mu) / n_obs
    delta = right.mu - left.mu
    m2 = left.m2 + right.m2 + (n_left * n_right / n_obs) * np.outer(delta, delta)
    return MomentState(
        n_obs=n_obs,
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        m2=np.ascontiguousarray(m2, dtype=np.float64),
    )


def unmerge_state(total: MomentState, part: MomentState) -> MomentState:
    """Remove ``part`` from ``total`` when ``part`` is a subset of ``total``."""
    if part.n_obs == 0:
        return total.copy()
    n_obs = int(total.n_obs) - int(part.n_obs)
    if n_obs < 0:
        raise ValueError("cannot unmerge more observations than the total holds")
    if n_obs == 0:
        return empty_state(int(total.mu.size))
    mu = (total.n_obs * total.mu - part.n_obs * part.mu) / n_obs
    delta = part.mu - mu
    coeff = (n_obs * part.n_obs) / total.n_obs
    m2 = total.m2 - part.m2 - coeff * np.outer(delta, delta)
    return MomentState(
        n_obs=n_obs,
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        m2=np.ascontiguousarray(m2, dtype=np.float64),
    )


def empirical_from_window(
    window: NDArray[np.float64], *, keep_returns: bool, ddof: int = 1
) -> FoldMoments:
    state = state_from_window(window)
    returns = window if keep_returns else None
    return _pack(
        state.mu,
        state.covariance(ddof=ddof),
        returns,
        state.n_obs,
        keep_returns=keep_returns,
    )


def _fold_from_state(
    state: MomentState,
    *,
    returns: NDArray[np.float64] | None,
    keep_returns: bool,
    ddof: int,
) -> FoldMoments:
    return _pack(
        state.mu,
        state.covariance(ddof=ddof),
        returns,
        state.n_obs,
        keep_returns=keep_returns,
    )


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
class _SlideState:
    start: int
    stop: int
    moments: MomentState


@dataclass
class _IndexState:
    rows: NDArray[np.intp]
    moments: MomentState


class OverlapMomentCache:
    """Empirical moments with sliding-window and CPCV fold-block reuse.

    ``n_fits`` counts cold Gram computations. ``n_updates`` counts Chan merges
    or unmerges that avoid a full centered Gram on the train window.
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
        self._blocks: list[MomentState] | None = None
        if fold_blocks:
            self._blocks = [self._stats_from_rows(rows) for rows in fold_blocks]
            self.n_fits += len(self._blocks)

    def _stats_from_rows(self, rows: NDArray[np.intp]) -> MomentState:
        return state_from_window(self.X[rows])

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
                moments=state_from_window(window),
            )
            self._slide[path_key] = state
            self.n_fits += 1
            returns = window if self.keep_returns else None
            return _fold_from_state(
                state.moments,
                returns=returns,
                keep_returns=self.keep_returns,
                ddof=self.ddof,
            )

        moments = prev.moments
        if start > prev.start:
            moments = unmerge_state(
                moments, state_from_window(self.X[prev.start : start])
            )
        elif start < prev.start:
            moments = merge_states(
                moments, state_from_window(self.X[start : prev.start])
            )
        if stop > prev.stop:
            moments = merge_states(moments, state_from_window(self.X[prev.stop : stop]))
        elif stop < prev.stop:
            moments = unmerge_state(
                moments, state_from_window(self.X[stop : prev.stop])
            )
        self.n_updates += 1
        state = _SlideState(start=start, stop=stop, moments=moments)
        self._slide[path_key] = state
        returns = self.X[start:stop] if self.keep_returns else None
        return _fold_from_state(
            moments,
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
                    moments=slide.moments,
                )
        if previous is not None:
            removed = np.setdiff1d(previous.rows, rows, assume_unique=True)
            added = np.setdiff1d(rows, previous.rows, assume_unique=True)
            if removed.size + added.size < rows.size:
                moments = previous.moments
                if removed.size:
                    moments = unmerge_state(moments, state_from_window(self.X[removed]))
                if added.size:
                    moments = merge_states(moments, state_from_window(self.X[added]))
                state = _IndexState(rows=rows, moments=moments)
                self._indexed[path_key] = state
                self.n_updates += 1
                returns = self.X[rows] if self.keep_returns else None
                return _fold_from_state(
                    moments,
                    returns=returns,
                    keep_returns=self.keep_returns,
                    ddof=self.ddof,
                )

        window = self.X[rows]
        state = _IndexState(rows=rows, moments=state_from_window(window))
        self._indexed[path_key] = state
        self.n_fits += 1
        return _fold_from_state(
            state.moments,
            returns=window if self.keep_returns else None,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )

    def _from_blocks(self, fold: FoldSpec) -> FoldMoments:
        assert self._blocks is not None
        moments = empty_state(self.X.shape[1])
        for block_id in fold.train_block_ids:
            moments = merge_states(moments, self._blocks[block_id])
        if fold.train_excluded_idx.size:
            moments = unmerge_state(
                moments, state_from_window(self.X[fold.train_excluded_idx])
            )
        self.n_updates += 1
        returns = self.X[fold.train_idx] if self.keep_returns else None
        return _fold_from_state(
            moments,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )
