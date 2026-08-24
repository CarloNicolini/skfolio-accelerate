"""Portfolio scores aligned with skfolio estimator.score / path Sharpe."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from skfolio import RatioMeasure
from skfolio.measures import (
    cvar,
    get_drawdowns,
    max_drawdown,
    mean,
    semi_deviation,
    standard_deviation,
    variance,
)
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio


def _as_matrix(X) -> NDArray[np.float64]:
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(copy=False)
    else:
        arr = np.asarray(X)
    if arr.dtype != np.float64:
        return np.asarray(arr, dtype=np.float64)
    return arr


def portfolio_returns(X, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return _as_matrix(X) @ np.asarray(weights, dtype=np.float64)


def sharpe_ratio(returns: NDArray[np.float64], risk_free_rate: float = 0.0) -> float:
    excess = mean(returns) - risk_free_rate
    denom = standard_deviation(returns, biased=False)
    if denom == 0:
        return float("nan")
    return float(excess / denom)


def sortino_ratio(returns: NDArray[np.float64], risk_free_rate: float = 0.0) -> float:
    excess = mean(returns) - risk_free_rate
    denom = semi_deviation(returns)
    if denom == 0:
        return float("nan")
    return float(excess / denom)


def score_returns(returns: NDArray[np.float64], scoring: Any) -> float:
    if scoring is None or scoring in {"sharpe_ratio", RatioMeasure.SHARPE_RATIO}:
        return sharpe_ratio(returns)
    if scoring in {"sortino_ratio", RatioMeasure.SORTINO_RATIO}:
        return sortino_ratio(returns)
    if scoring == "mean":
        return float(mean(returns))
    if scoring == "variance":
        return float(-variance(returns, biased=False))
    if scoring == "volatility":
        return float(-standard_deviation(returns, biased=False))
    if scoring == "cvar":
        return float(-cvar(returns))
    if scoring == "max_drawdown":
        return float(-max_drawdown(get_drawdowns(returns, compounded=False)))
    raise TypeError(f"Unsupported native scoring {scoring!r}")


def _rows(X, idx: NDArray[np.intp]):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def attach_weights(
    estimator,
    params: dict[str, Any],
    weights: NDArray[np.float64],
    X_test,
    scratch=None,
):
    est = scratch if scratch is not None else clone(estimator)
    if params:
        est.set_params(**params)
    est.weights_ = np.asarray(weights, dtype=np.float64)
    est.n_features_in_ = int(_as_matrix(X_test).shape[1])
    if hasattr(X_test, "columns"):
        est.feature_names_in_ = np.asarray(X_test.columns)
    return est


def score_with_estimator(
    estimator,
    params: dict[str, Any],
    weights: NDArray[np.float64],
    X_test,
    scoring: Any,
    scratch=None,
) -> float:
    """Score a test window. ``scoring=None`` uses the estimator's native score."""
    if isinstance(scoring, (str, RatioMeasure)):
        return score_returns(portfolio_returns(X_test, weights), scoring)
    est = attach_weights(estimator, params, weights, X_test, scratch=scratch)
    if scoring is None:
        return float(est.score(X_test))
    if callable(scoring):
        try:
            return float(scoring(est, X_test))
        except TypeError:
            return float(scoring(est, X_test, None))
    raise TypeError(f"Unsupported scoring object {type(scoring)!r}")


def path_portfolios(
    X,
    weights: NDArray[np.float64],
    segments: list[NDArray[np.intp]],
) -> list[Portfolio]:
    return [
        Portfolio(X=_rows(X, idx), weights=weights)
        for idx in segments
        if len(idx) > 0
    ]


def score_multi_period(portfolios: list[Portfolio], scoring: Any = None) -> float:
    if not portfolios:
        return float("nan")
    if len(portfolios) == 1:
        mpp = portfolios[0]
    else:
        mpp = MultiPeriodPortfolio(
            portfolios=portfolios,
            name="path",
            check_observations_order=False,
        )
    if scoring is None or scoring in {"sharpe_ratio", RatioMeasure.SHARPE_RATIO}:
        return float(mpp.sharpe_ratio)
    if callable(scoring) and hasattr(scoring, "_score_func"):
        sign = getattr(scoring, "_sign", 1)
        kwargs = getattr(scoring, "_kwargs", {})
        return float(sign * scoring._score_func(mpp, **kwargs))
    return float(mpp.sharpe_ratio)


def multi_period_sharpe(
    X,
    weights: NDArray[np.float64],
    segments: list[NDArray[np.intp]],
) -> float:
    return score_multi_period(path_portfolios(X, weights, segments))
