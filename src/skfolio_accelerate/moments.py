"""Fit skfolio priors once per fold (and per prior hyperparameter set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from skfolio.prior import EmpiricalPrior
from skfolio.prior._base import ReturnDistribution


@dataclass
class FoldMoments:
    mu: NDArray[np.float64]
    covariance: NDArray[np.float64]
    cholesky: NDArray[np.float64]
    returns: NDArray[np.float64]
    sample_weight: NDArray[np.float64] | None = None


def _as_float(x: Any) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def moments_from_distribution(dist: ReturnDistribution) -> FoldMoments:
    cov = _as_float(dist.covariance)
    chol = getattr(dist, "cholesky", None)
    if chol is None:
        chol = np.linalg.cholesky(cov)
    else:
        chol = _as_float(chol)
    sample_weight = getattr(dist, "sample_weight", None)
    if sample_weight is not None:
        sample_weight = _as_float(sample_weight)
    return FoldMoments(
        mu=_as_float(dist.mu),
        covariance=cov,
        cholesky=chol,
        returns=_as_float(dist.returns),
        sample_weight=sample_weight,
    )


def fit_prior(estimator, X_train, y_train=None) -> FoldMoments:
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
    return moments_from_distribution(prior.return_distribution_)


class FoldCache:
    """Cache fitted priors keyed by (fold_id, data_param_fingerprint)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, str], FoldMoments] = {}
        self.n_fits = 0

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
        moments = fit_prior(estimator, X_train, y_train)
        self._store[key] = moments
        self.n_fits += 1
        return moments
