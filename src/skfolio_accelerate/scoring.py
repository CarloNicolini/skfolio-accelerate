"""Assemble skfolio MultiPeriodPortfolio / Population from fold weights.

Ranking helpers treat numerical closeness as a possible tie. Matching Sharpe
values to machine precision does not by itself prove that two rankings are
the same; callers should pass ``score_tolerance`` when solver noise matters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio

from skfolio_accelerate._arrays import as_float_array, contiguous_row_slice
from skfolio_accelerate.cv_plan import CVPlan


def path_sharpes(prediction) -> np.ndarray:
    """Sharpe of each path (Population) or the single MultiPeriodPortfolio."""
    if hasattr(prediction, "__len__") and not hasattr(prediction, "sharpe_ratio"):
        return np.asarray([ptf.sharpe_ratio for ptf in prediction], dtype=np.float64)
    if hasattr(prediction, "sharpe_ratio"):
        return np.asarray([prediction.sharpe_ratio], dtype=np.float64)
    raise TypeError(f"Unsupported prediction type {type(prediction)!r}")


def _ranking_inputs(reference, observed) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.ndim != 1 or obs.ndim != 1 or ref.shape != obs.shape:
        raise ValueError(
            "reference and observed must be one-dimensional with equal size"
        )
    if ref.size == 0 or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(obs)):
        raise ValueError("ranking scores must be non-empty and finite")
    return ref, obs


def ranking_precision_at_k(
    reference,
    observed,
    *,
    k: int,
    score_tolerance: float = 0.0,
) -> float:
    """Top-k precision, without penalizing swaps tied in the native scores."""
    ref, obs = _ranking_inputs(reference, observed)
    if not 1 <= k <= ref.size:
        raise ValueError(f"k must be between 1 and {ref.size}")
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be non-negative")
    reference_order = np.argsort(-ref, kind="stable")
    threshold = ref[reference_order[k - 1]] - score_tolerance
    ref_top = np.flatnonzero(ref >= threshold)
    obs_top = np.argsort(-obs, kind="stable")[:k]
    return float(np.intersect1d(ref_top, obs_top, assume_unique=True).size / k)


def _tolerant_ranks(values: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance == 0:
        return rankdata(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        group_min = values[order[start]]
        while stop < values.size and values[order[stop]] - group_min <= tolerance:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman_rank_correlation(
    reference,
    observed,
    *,
    score_tolerance: float = 0.0,
) -> float:
    """Spearman correlation with optional numerical tie grouping."""
    ref, obs = _ranking_inputs(reference, observed)
    if ref.size < 2:
        raise ValueError("at least two portfolios are required")
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be non-negative")
    ranked_ref = _tolerant_ranks(ref, score_tolerance)
    ranked_obs = _tolerant_ranks(obs, score_tolerance)
    if np.ptp(ranked_ref) == 0 or np.ptp(ranked_obs) == 0:
        return float("nan")
    return float(np.corrcoef(ranked_ref, ranked_obs)[0, 1])


def path_sharpes_from_weights(
    X, cv_plan: CVPlan, weights_by_fold: dict[int, NDArray[np.float64]]
) -> np.ndarray:
    """Compute path Sharpes without constructing thousands of Portfolio objects.

    Uses the same sample standard deviation (``ddof=1``) as skfolio's default
    Sharpe, with a zero risk-free rate. Fold order matches :func:`assemble_prediction`.
    """
    matrix = as_float_array(X)
    path_returns: list[list[NDArray[np.float64]]] = [[] for _ in range(cv_plan.n_paths)]
    for fold in cv_plan.folds:
        weights = weights_by_fold[fold.fold_id]
        if cv_plan.combinatorial:
            segments = zip(fold.test_segments, fold.path_ids, strict=False)
        else:
            segments = ((fold.test_idx, fold.path_id),)
        for rows, path_id in segments:
            row_selector = contiguous_row_slice(rows)
            if row_selector is None:
                row_selector = rows
            if fold.asset_idx is None:
                returns = matrix[row_selector] @ weights
            else:
                returns = matrix[row_selector][:, fold.asset_idx] @ weights
            path_returns[path_id].append(returns)

    sharpes = np.empty(cv_plan.n_paths, dtype=np.float64)
    for path_id, parts in enumerate(path_returns):
        returns = np.concatenate(parts)
        volatility = np.std(returns, ddof=1)
        sharpes[path_id] = np.mean(returns) / volatility if volatility else np.nan
    return sharpes


def window_view(
    matrix: NDArray,
    rows: NDArray[np.intp],
    cols: NDArray[np.intp] | None = None,
) -> NDArray:
    """Row (and optional column) slice, using a view when the rows are contiguous."""
    rows = np.asarray(rows, dtype=np.intp)
    row_selector = contiguous_row_slice(rows)
    if row_selector is None:
        row_selector = rows
    if cols is None:
        return matrix[row_selector]
    return matrix[row_selector][:, np.asarray(cols, dtype=np.intp)]


def _is_default_range(labels) -> bool:
    return (
        type(labels).__name__ == "RangeIndex"
        and getattr(labels, "start", None) == 0
        and getattr(labels, "step", None) == 1
    )


def _keep_frame_labels(X) -> bool:
    """True when ``X`` carries timestamps or asset names worth preserving."""
    index = getattr(X, "index", None)
    columns = getattr(X, "columns", None)
    if index is None or columns is None:
        return False
    return not (_is_default_range(index) and _is_default_range(columns))


def _test_observations(
    X,
    idx: NDArray[np.intp],
    cols: NDArray[np.intp] | None,
    *,
    x_np: NDArray[np.float64],
):
    """Numeric test-fold view, wrapping a labeled frame only when needed."""
    rows = np.asarray(idx, dtype=np.intp)
    values = window_view(x_np, rows, cols)
    if not _keep_frame_labels(X):
        return values
    selector = contiguous_row_slice(rows)
    row_sel: slice | NDArray[np.intp] = selector if selector is not None else rows
    columns = X.columns if cols is None else X.columns[np.asarray(cols, dtype=np.intp)]
    import pandas as pd

    return pd.DataFrame(values, index=X.index[row_sel], columns=columns)


def make_segment_portfolio(
    X,
    weights: NDArray[np.float64],
    idx: NDArray[np.intp],
    cols: NDArray[np.intp] | None = None,
    *,
    name: str = "MeanRisk",
    x_np: NDArray[np.float64] | None = None,
    segment_params: dict | None = None,
) -> Portfolio:
    matrix = as_float_array(X) if x_np is None else x_np
    x_test = _test_observations(X, idx, cols, x_np=matrix)
    extra = {} if segment_params is None else dict(segment_params)
    extra.pop("name", None)
    extra.pop("check_observations_order", None)
    return Portfolio(
        X=x_test,
        weights=np.asarray(weights, dtype=np.float64),
        name=name,
        **extra,
    )


def assemble_prediction(
    X,
    cv_plan: CVPlan,
    weights_by_fold: dict[int, NDArray[np.float64]],
    *,
    name: str = "MeanRisk",
    portfolio_params: dict | None = None,
    segment_params: dict | None = None,
) -> MultiPeriodPortfolio | Population:
    """Build a skfolio MultiPeriodPortfolio or Population from fold weights."""
    extra = {} if portfolio_params is None else dict(portfolio_params)
    extra.setdefault("check_observations_order", False)
    x_np = as_float_array(X)

    if cv_plan.combinatorial or cv_plan.multi_path:
        path_lists: list[list[Portfolio]] = [[] for _ in range(cv_plan.n_paths)]
        for fold in cv_plan.folds:
            w = weights_by_fold[fold.fold_id]
            if cv_plan.combinatorial:
                pairs = zip(fold.test_segments, fold.path_ids, strict=True)
            else:
                pairs = ((fold.test_idx, fold.path_id),)
            for seg, path_id in pairs:
                if len(seg) == 0:
                    continue
                path_lists[path_id].append(
                    make_segment_portfolio(
                        X,
                        w,
                        seg,
                        fold.asset_idx,
                        name=name,
                        x_np=x_np,
                        segment_params=segment_params,
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
            segment_params=segment_params,
        )
        for fold in ordered
        if fold.test_idx.size
    ]
    extra.pop("name", None)
    return MultiPeriodPortfolio(portfolios=portfolios, **extra)
