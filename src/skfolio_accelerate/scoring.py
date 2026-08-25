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


def native_segment_returns(
    X,
    weights: NDArray[np.float64],
    idx: NDArray[np.intp],
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    matrix = _as_matrix(X)
    w = np.asarray(weights, dtype=np.float64)
    rows = np.asarray(idx, dtype=np.intp)
    if cols is None:
        return matrix[rows] @ w
    return matrix[np.ix_(rows, np.asarray(cols, dtype=np.intp))] @ w


def native_path_returns(
    X,
    items: list[tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp] | None]],
) -> NDArray[np.float64]:
    """Concatenate test-window portfolio returns for one path."""
    parts = [
        native_segment_returns(X, weights, idx, cols)
        for weights, idx, cols in items
        if np.asarray(idx).size
    ]
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def native_path_sharpe(
    X,
    items: list[tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp] | None]],
) -> float:
    return sharpe_ratio(native_path_returns(X, items))


def make_segment_portfolio(
    X,
    weights: NDArray[np.float64],
    idx: NDArray[np.intp],
    cols: NDArray[np.intp] | None = None,
    *,
    name: str = "MeanRisk",
) -> Portfolio:
    from skfolio_accelerate.cv_plan import slice_panel

    x_test = slice_panel(X, idx, cols)
    return Portfolio(X=x_test, weights=np.asarray(weights, dtype=np.float64), name=name)


def assemble_prediction(
    X,
    cv_plan,
    weights_by_fold: dict[int, NDArray[np.float64]],
    *,
    name: str = "MeanRisk",
    portfolio_params: dict | None = None,
    build_portfolios: bool = True,
):
    """Build a skfolio MultiPeriodPortfolio or Population from fold weights.

    When ``build_portfolios`` is False, path Sharpes are still computed natively
    and a Population of empty-named MultiPeriodPortfolios is not returned; the
    caller should use ``native_path_sharpe``. This function always builds
    Portfolio objects when ``build_portfolios`` is True so the return type
    matches ``cross_val_predict``.
    """
    del build_portfolios
    extra = {} if portfolio_params is None else dict(portfolio_params)
    extra.setdefault("check_observations_order", False)

    if cv_plan.combinatorial:
        path_lists: list[list[Portfolio]] = [[] for _ in range(cv_plan.n_paths)]
        for fold in cv_plan.folds:
            w = weights_by_fold[fold.fold_id]
            for seg, path_id in zip(fold.test_segments, fold.path_ids, strict=False):
                if len(seg) == 0:
                    continue
                path_lists[path_id].append(
                    make_segment_portfolio(X, w, seg, fold.asset_idx, name=name)
                )
        pop_name = extra.pop("name", "path")
        extra.pop("check_observations_order", None)
        from skfolio.population import Population

        return Population(
            [
                MultiPeriodPortfolio(
                    name=f"{pop_name}_{i}",
                    portfolios=path_lists[i],
                    check_observations_order=False,
                    **extra,
                )
                for i in range(cv_plan.n_paths)
            ]
        )

    if cv_plan.multi_path:
        path_lists = [[] for _ in range(cv_plan.n_paths)]
        for fold in cv_plan.folds:
            w = weights_by_fold[fold.fold_id]
            path_lists[fold.path_id].append(
                make_segment_portfolio(
                    X, w, fold.test_idx, fold.asset_idx, name=name
                )
            )
        pop_name = extra.pop("name", "path")
        extra.pop("check_observations_order", None)
        from skfolio.population import Population

        return Population(
            [
                MultiPeriodPortfolio(
                    name=f"{pop_name}_{i}",
                    portfolios=path_lists[i],
                    check_observations_order=False,
                    **extra,
                )
                for i in range(cv_plan.n_paths)
            ]
        )

    ordered = sorted(cv_plan.folds, key=lambda f: int(f.test_idx[0]) if f.test_idx.size else 0)
    portfolios = [
        make_segment_portfolio(
            X,
            weights_by_fold[fold.fold_id],
            fold.test_idx,
            fold.asset_idx,
            name=name,
        )
        for fold in ordered
        if fold.test_idx.size
    ]
    extra.pop("name", None)
    return MultiPeriodPortfolio(portfolios=portfolios, **extra)
