"""Assemble skfolio MultiPeriodPortfolio / Population from fold weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio


def _as_matrix(X) -> NDArray[np.float64]:
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(copy=False)
    else:
        arr = np.asarray(X)
    if arr.dtype != np.float64:
        return np.asarray(arr, dtype=np.float64)
    return arr


def path_sharpes(prediction) -> np.ndarray:
    """Sharpe of each path (Population) or the single MultiPeriodPortfolio."""
    if hasattr(prediction, "__len__") and not hasattr(prediction, "sharpe_ratio"):
        return np.asarray([ptf.sharpe_ratio for ptf in prediction], dtype=np.float64)
    if hasattr(prediction, "sharpe_ratio"):
        return np.asarray([prediction.sharpe_ratio], dtype=np.float64)
    raise TypeError(f"Unsupported prediction type {type(prediction)!r}")


def path_sharpes_from_weights(X, cv_plan, weights_by_fold) -> np.ndarray:
    """Compute path Sharpes without constructing thousands of Portfolio objects."""
    matrix = _as_matrix(X)
    path_returns: list[list[NDArray[np.float64]]] = [[] for _ in range(cv_plan.n_paths)]
    for fold in cv_plan.folds:
        weights = weights_by_fold[fold.fold_id]
        if cv_plan.combinatorial:
            segments = zip(fold.test_segments, fold.path_ids, strict=False)
        else:
            segments = ((fold.test_idx, fold.path_id),)
        for rows, path_id in segments:
            if fold.asset_idx is None:
                returns = matrix[rows] @ weights
            else:
                returns = matrix[np.ix_(rows, fold.asset_idx)] @ weights
            path_returns[path_id].append(returns)

    sharpes = np.empty(cv_plan.n_paths, dtype=np.float64)
    for path_id, parts in enumerate(path_returns):
        returns = np.concatenate(parts)
        volatility = np.std(returns, ddof=1)
        sharpes[path_id] = np.mean(returns) / volatility if volatility else np.nan
    return sharpes


def make_segment_portfolio(
    X,
    weights: NDArray[np.float64],
    idx: NDArray[np.intp],
    cols: NDArray[np.intp] | None = None,
    *,
    name: str = "MeanRisk",
    x_np: NDArray[np.float64] | None = None,
) -> Portfolio:
    matrix = _as_matrix(X) if x_np is None else x_np
    rows = np.asarray(idx, dtype=np.intp)
    w = np.asarray(weights, dtype=np.float64)
    if cols is None:
        x_test = matrix[rows]
    else:
        x_test = matrix[np.ix_(rows, np.asarray(cols, dtype=np.intp))]
    return Portfolio(X=x_test, weights=w, name=name)


def assemble_prediction(
    X,
    cv_plan,
    weights_by_fold: dict[int, NDArray[np.float64]],
    *,
    name: str = "MeanRisk",
    portfolio_params: dict | None = None,
):
    """Build a skfolio MultiPeriodPortfolio or Population from fold weights."""
    extra = {} if portfolio_params is None else dict(portfolio_params)
    extra.setdefault("check_observations_order", False)
    x_np = _as_matrix(X)

    if cv_plan.combinatorial or cv_plan.multi_path:
        path_lists: list[list[Portfolio]] = [[] for _ in range(cv_plan.n_paths)]
        for fold in cv_plan.folds:
            w = weights_by_fold[fold.fold_id]
            if cv_plan.combinatorial:
                pairs = zip(fold.test_segments, fold.path_ids, strict=False)
            else:
                pairs = ((fold.test_idx, fold.path_id),)
            for seg, path_id in pairs:
                if len(seg) == 0:
                    continue
                path_lists[path_id].append(
                    make_segment_portfolio(
                        X, w, seg, fold.asset_idx, name=name, x_np=x_np
                    )
                )
        pop_name = extra.pop("name", "path")
        extra.pop("check_observations_order", None)
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

    ordered = sorted(
        cv_plan.folds, key=lambda f: int(f.test_idx[0]) if f.test_idx.size else 0
    )
    portfolios = [
        make_segment_portfolio(
            X,
            weights_by_fold[fold.fold_id],
            fold.test_idx,
            fold.asset_idx,
            name=name,
            x_np=x_np,
        )
        for fold in ordered
        if fold.test_idx.size
    ]
    extra.pop("name", None)
    return MultiPeriodPortfolio(portfolios=portfolios, **extra)
