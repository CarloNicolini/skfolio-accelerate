"""Empirical moments from overlapping CV training windows.

The cache stores sufficient statistics

    n  = number of observations
    s  = sum of rows                          (first moment × n)
    G  = XᵀX                                  (second raw moment)

and forms the unbiased sample covariance only when a solver needs it:

    μ = s / n
    Σ = (G − s sᵀ / n) / (n − ddof)     with ddof=1

This is algebraically the same estimator as ``numpy.cov(..., ddof=1)`` and as
skfolio's default :class:`~skfolio.moments.EmpiricalCovariance` *before* the
optional nearest-PD projection. Rolling WalkForward windows and CPCV fold
blocks are applied as exact rank-k updates of ``(s, G)``, not as a different
statistical estimator.

``keep_returns=True`` additionally stores the training window itself for
scenario-based risks. Those arrays are views into the parent returns matrix
when the training rows are contiguous. ``keep_covariance=False`` skips Gram
matrix updates when an engine only consumes scenarios and their mean.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """Return ``slice(start, stop)`` when ``rows`` is a contiguous increasing range."""
    if rows.ndim != 1 or rows.size == 0:
        return None
    start = int(rows[0])
    stop = int(rows[-1]) + 1
    if start < 0 or stop - start != rows.size:
        return None
    if rows.size > 1 and int(rows[1]) != start + 1:
        return None
    if rows.size > 2 and int(rows[rows.size // 2]) != start + rows.size // 2:
        return None
    return slice(start, stop)


@dataclass(slots=True)
class FoldMoments:
    """Moments of one training window.

    ``covariance`` is empty when the caller does not require second moments.
    ``returns`` is empty when the caller asked not to keep scenarios (variance /
    inverse-volatility).
    """

    mu: NDArray[np.float64]
    covariance: NDArray[np.float64]
    returns: NDArray[np.float64]
    n_observations: int = 0


def is_default_empirical(estimator) -> bool:
    """True when moments are sample mean + sample covariance (ddof=1).

    Custom ``EmpiricalPrior`` options that change the statistical estimator
    (log-normal projection, investment horizon, non-default mu/covariance
    classes, rolling ``window_size``, ``assume_centered``, ``ddof != 1``,
    ``max_history``) must not use the compact path.

    Default ``EmpiricalCovariance(nearest=True)`` projects to the nearest PD
    matrix after ``numpy.cov``. The compact Gram formula skips that
    projection. The two coincide when the sample covariance is already PD,
    which is the intended compact regime (typically ``T > n_assets``).
    """
    prior = getattr(estimator, "prior_estimator", None)
    if prior is None:
        return True
    if type(prior).__name__ != "EmpiricalPrior":
        return False
    if getattr(prior, "is_log_normal", False):
        return False
    if getattr(prior, "investment_horizon", None) is not None:
        return False
    if getattr(prior, "max_history", None) is not None:
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
    if cov is not None and getattr(cov, "assume_centered", False):
        return False
    return True


def _finish_moments(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64] | None,
    returns: NDArray[np.float64] | None,
    n_observations: int,
    *,
    keep_returns: bool,
) -> FoldMoments:
    covariance = (
        np.empty((0, 0), dtype=np.float64)
        if cov is None
        else np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
    )
    ret = (
        returns
        if keep_returns and returns is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    return FoldMoments(
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        covariance=covariance,
        returns=ret,
        n_observations=int(n_observations),
    )


def empirical_from_window(
    window: NDArray[np.float64], *, keep_returns: bool, ddof: int = 1
) -> FoldMoments:
    """Empirical mean and covariance from an explicit training window.

    Parameters
    ----------
    window : ndarray of shape (n_observations, n_assets)
        Training returns.

    keep_returns : bool
        If ``True``, store ``window`` on the returned :class:`FoldMoments` for
        scenario-based risks.

    ddof : int, default=1
        Delta degrees of freedom for the covariance (matches
        ``numpy.cov(..., ddof=1)`` and skfolio's default empirical covariance).

    Returns
    -------
    moments : FoldMoments
        Mean, symmetrized covariance, optional returns, and observation count.
    """
    t, n = window.shape
    mu = np.mean(window, axis=0)
    if n == 1:
        cov = np.var(window, axis=0, ddof=ddof).reshape(1, 1)
    else:
        cov = np.cov(window, rowvar=False, ddof=ddof)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    returns = window if keep_returns else None
    return _finish_moments(mu, cov, returns, t, keep_returns=keep_returns)


def empirical_from_stats(
    n_obs: int,
    sum_vec: NDArray[np.float64],
    gram: NDArray[np.float64] | None,
    *,
    returns: NDArray[np.float64] | None,
    keep_returns: bool,
    ddof: int = 1,
) -> FoldMoments:
    """Moments from ``(n, sum, XᵀX)``; omit covariance when ``gram`` is ``None``."""
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
    """Empirical moments with sliding-window and CPCV fold-block reuse.

    ``n_fits`` counts cold statistics computations. ``n_updates`` counts slides
    or block additions that avoid recomputing statistics on the full train
    window.

    When covariance is retained, cancellation can in principle erode
    positive-definiteness after many Gram updates; OSQP then retries with a
    small diagonal jitter. No additional approximation is introduced.
    """

    def __init__(
        self,
        X: NDArray[np.float64],
        *,
        keep_returns: bool,
        keep_covariance: bool = True,
        ddof: int = 1,
        fold_blocks: Sequence[NDArray[np.intp]] | None = None,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.keep_returns = keep_returns
        self.keep_covariance = keep_covariance
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
            gram=window.T @ window if self.keep_covariance else None,
        )

    def get(self, fold: FoldSpec, *, path_key: int = 0) -> FoldMoments:
        """Return empirical moments for ``fold``, reusing prior statistics.

        Parameters
        ----------
        fold : FoldSpec
            Compiled train/test split.

        path_key : int, default=0
            Cache key for sliding / indexed state (typically ``fold.path_id``).

        Returns
        -------
        moments : FoldMoments
            Mean, covariance, and optional scenario returns for the training
            window.
        """
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
            if gram is not None:
                gram = gram - drop.T @ drop
            sum_vec = sum_vec - drop.sum(axis=0)
        elif start < prev.start:
            add = self.X[start : prev.start]
            if gram is not None:
                gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        if stop > prev.stop:
            add = self.X[prev.stop : stop]
            if gram is not None:
                gram = gram + add.T @ add
            sum_vec = sum_vec + add.sum(axis=0)
        elif stop < prev.stop:
            drop = self.X[stop : prev.stop]
            if gram is not None:
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
                    if gram is not None:
                        gram = gram - values.T @ values
                if added.size:
                    values = self.X[added]
                    sum_vec = sum_vec + values.sum(axis=0)
                    if gram is not None:
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
            gram=window.T @ window if self.keep_covariance else None,
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
        if self._blocks is None:
            raise RuntimeError("CPCV fold-block statistics were not compiled")
        n_obs = 0
        sum_vec: NDArray[np.float64] | None = None
        gram: NDArray[np.float64] | None = None
        for block_id in fold.train_block_ids:
            block = self._blocks[block_id]
            n_obs += block.n_obs
            sum_vec = block.sum_vec if sum_vec is None else sum_vec + block.sum_vec
            if self.keep_covariance:
                if block.gram is None:
                    raise RuntimeError("CPCV block covariance statistics are missing")
                gram = block.gram if gram is None else gram + block.gram
        if sum_vec is None:
            raise ValueError("CPCV training window has no fold blocks")
        if fold.train_excluded_idx.size:
            excluded = self.X[fold.train_excluded_idx]
            n_obs -= int(excluded.shape[0])
            sum_vec = sum_vec - excluded.sum(axis=0)
            if gram is not None:
                gram = gram - excluded.T @ excluded
        self.n_updates += 1
        returns = self.X[fold.train_idx] if self.keep_returns else None
        return empirical_from_stats(
            n_obs,
            sum_vec,
            gram,
            returns=returns,
            keep_returns=self.keep_returns,
            ddof=self.ddof,
        )


@dataclass(slots=True)
class PathMomentSession:
    """Moment cache bound to one MRC asset subset or the full universe.

    Attributes
    ----------
    cache : OverlapMomentCache
        Underlying sliding-window / CPCV-block cache.

    x_work : ndarray of shape (n_observations, n_assets_work)
        Returns used by the cache (full universe or one MRC subset).
    """

    cache: OverlapMomentCache
    x_work: NDArray[np.float64]

    def get(self, fold: FoldSpec) -> FoldMoments:
        """Moments for ``fold`` using ``fold.path_id`` as the cache key."""
        return self.cache.get(fold, path_key=fold.path_id)


def path_moment_session(
    X: NDArray[np.float64],
    folds: Sequence[FoldSpec],
    *,
    keep_returns: bool,
    keep_covariance: bool = True,
    fold_blocks: Sequence[NDArray[np.intp]] | None = None,
) -> PathMomentSession:
    """Slice MRC assets once, then reuse overlapping train-window statistics.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_assets)
        Full-universe returns.

    folds : sequence of FoldSpec
        Folds for one path batch. When the first fold carries ``asset_idx``,
        columns are sliced once for the whole session.

    keep_returns : bool
        Forwarded to :class:`OverlapMomentCache`.

    keep_covariance : bool, default=True
        Compute and update second moments. Scenario-only compact engines disable
        this because they consume returns and their mean, not covariance.

    fold_blocks : sequence of ndarray, optional
        CPCV fold blocks. Ignored for MRC asset subsets (blocks are defined on
        the full universe and would not align after column slicing).

    Returns
    -------
    session : PathMomentSession
        Cache bound to the working returns matrix.
    """
    asset_idx = folds[0].asset_idx if folds else None
    if asset_idx is None:
        x_work = X
        blocks = fold_blocks
    else:
        x_work = X[:, np.asarray(asset_idx, dtype=np.intp)]
        blocks = None
    return PathMomentSession(
        cache=OverlapMomentCache(
            x_work,
            keep_returns=keep_returns,
            keep_covariance=keep_covariance,
            fold_blocks=blocks,
        ),
        x_work=x_work,
    )
