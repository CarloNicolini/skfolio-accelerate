"""Assemble skfolio MultiPeriodPortfolio / Population from fold weights.

This module turns a compiled :class:`~skfolio_accelerate.cv_plan.CVPlan` and a
mapping ``fold_id -> weights`` into the same
:class:`~skfolio.portfolio.MultiPeriodPortfolio` /
:class:`~skfolio.population.Population` containers that skfolio's
``cross_val_predict`` returns.

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
    """Sharpe ratio of each path in a prediction container.

    Parameters
    ----------
    prediction : MultiPeriodPortfolio or Population
        Output of :func:`~skfolio_accelerate.predict.cross_val_predict`. A
        :class:`~skfolio.population.Population` yields one Sharpe per path. A
        single :class:`~skfolio.portfolio.MultiPeriodPortfolio` yields a
        length-1 array.

    Returns
    -------
    sharpes : ndarray of shape (n_paths,)
        Path Sharpe ratios using skfolio's default computation on each
        multi-period portfolio.

    Raises
    ------
    TypeError
        If ``prediction`` is neither a Population-like sequence of portfolios
        nor an object exposing ``sharpe_ratio``.

    Examples
    --------
    >>> sharpes = path_sharpes(prediction)  # doctest: +SKIP
    >>> float(np.mean(sharpes))  # doctest: +SKIP
    """
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
    """Top-k precision without penalizing swaps tied in the native scores.

    Parameters
    ----------
    reference : array-like of shape (n_portfolios,)
        Reference scores (for example native skfolio path Sharpes). Higher is
        better.

    observed : array-like of shape (n_portfolios,)
        Observed scores to compare (for example accelerated path Sharpes).

    k : int
        Number of top portfolios to retain. Must satisfy ``1 <= k <= n``.

    score_tolerance : float, default=0.0
        Absolute score gap treated as a tie in the reference ranking. Portfolios
        within ``score_tolerance`` of the k-th reference score are all counted
        as belonging to the reference top set. Must be non-negative.

    Returns
    -------
    precision : float
        Fraction of the observed top-``k`` that intersects the (tie-expanded)
        reference top set. Always in ``[0, 1]``.

    Raises
    ------
    ValueError
        If shapes differ, scores are empty/non-finite, ``k`` is out of range, or
        ``score_tolerance`` is negative.

    Notes
    -----
    Use this when validating that accelerated rankings preserve the native
    best set. Numerical closeness of scores alone does not imply identical
    orderings; pass a positive ``score_tolerance`` when solver noise creates
    artificial swaps among near-ties.

    Examples
    --------
    >>> ranking_precision_at_k([0.5, 0.4, 0.1], [0.49, 0.41, 0.1], k=2)
    1.0

    See Also
    --------
    spearman_rank_correlation
    """
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
    """Spearman rank correlation with optional numerical tie grouping.

    Parameters
    ----------
    reference : array-like of shape (n_portfolios,)
        Reference scores. Higher is better for ranking comparisons, but any
        finite scores are accepted.

    observed : array-like of shape (n_portfolios,)
        Observed scores of the same length.

    score_tolerance : float, default=0.0
        Absolute gap within which scores receive the same average rank. Must be
        non-negative.

    Returns
    -------
    correlation : float
        Spearman correlation in ``[-1, 1]``, or ``nan`` when either ranking is
        constant after tie grouping.

    Raises
    ------
    ValueError
        If fewer than two portfolios are provided, shapes differ, scores are
        non-finite, or ``score_tolerance`` is negative.

    See Also
    --------
    ranking_precision_at_k
    """
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
    Sharpe, with a zero risk-free rate. Fold order matches
    :func:`assemble_prediction`.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    cv_plan : CVPlan
        Compiled cross-validation plan.

    weights_by_fold : dict[int, ndarray of shape (n_assets,)]
        Weight vector for each ``fold_id``.

    Returns
    -------
    sharpes : ndarray of shape (n_paths,)
        Per-path Sharpe ratios. Paths with zero volatility yield ``nan``.

    Notes
    -----
    Intended for :func:`~skfolio_accelerate.search.grid_search`, where only the
    winning parameter set is materialized into Portfolio objects.
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
    """Row (and optional column) slice, using a view when the rows are contiguous.

    Parameters
    ----------
    matrix : ndarray of shape (n_observations, n_assets)
        Parent returns matrix.

    rows : ndarray of shape (n_rows,)
        Observation indices.

    cols : ndarray of shape (n_cols,), optional
        Asset indices. When ``None``, all columns are kept.

    Returns
    -------
    view : ndarray
        ``matrix[rows]`` or ``matrix[rows][:, cols]``. Contiguous WalkForward /
        CPCV blocks become a slice view instead of an advanced-index copy.
    """
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
    """Build one skfolio :class:`~skfolio.portfolio.Portfolio` for a test segment.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Full return matrix (used for index/column labels when present).

    weights : ndarray of shape (n_assets,)
        Portfolio weights for this segment.

    idx : ndarray of shape (n_test,)
        Test observation indices.

    cols : ndarray of shape (n_subset,), optional
        Asset subset for MultipleRandomizedCV paths.

    name : str, default="MeanRisk"
        Portfolio name.

    x_np : ndarray of shape (n_observations, n_assets), optional
        Precomputed float64 view of ``X``.

    segment_params : dict, optional
        Extra keyword arguments forwarded to
        :class:`~skfolio.portfolio.Portfolio` (transaction costs, fees, ...).

    Returns
    -------
    portfolio : Portfolio
        Out-of-sample portfolio on the selected test observations.
    """
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


def _merged_segment_params(segment_params, fold_segment_params, fold_id: int) -> dict:
    extra = {} if segment_params is None else dict(segment_params)
    if fold_segment_params:
        extra.update(fold_segment_params.get(fold_id, {}))
    return extra


def assemble_prediction(
    X,
    cv_plan: CVPlan,
    weights_by_fold: dict[int, NDArray[np.float64]],
    *,
    name: str = "MeanRisk",
    portfolio_params: dict | None = None,
    segment_params: dict | None = None,
    fold_segment_params: dict[int, dict] | None = None,
) -> MultiPeriodPortfolio | Population:
    """Build a skfolio MultiPeriodPortfolio or Population from fold weights.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    cv_plan : CVPlan
        Compiled cross-validation plan that defines paths and test segments.

    weights_by_fold : dict[int, ndarray of shape (n_assets,)]
        Weight vector for each ``fold_id``.

    name : str, default="MeanRisk"
        Name stamped on each segment portfolio.

    portfolio_params : dict, optional
        Parameters for the outer
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` /
        :class:`~skfolio.population.Population` constructors.

    segment_params : dict, optional
        Parameters forwarded to each segment
        :class:`~skfolio.portfolio.Portfolio`.

    fold_segment_params : dict[int, dict], optional
        Per-``fold_id`` overrides merged on top of ``segment_params``. Used to
        stamp the ``previous_weights`` that were in force when that fold was
        solved (transaction costs, turnover).

    Returns
    -------
    prediction : MultiPeriodPortfolio or Population
        Same container types as skfolio ``cross_val_predict``. Combinatorial and
        multi-path plans yield a Population; single-path plans yield a
        MultiPeriodPortfolio ordered by first test index.

    See Also
    --------
    path_sharpes_from_weights
    """
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
                        segment_params=_merged_segment_params(
                            segment_params, fold_segment_params, fold.fold_id
                        ),
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
            segment_params=_merged_segment_params(
                segment_params, fold_segment_params, fold.fold_id
            ),
        )
        for fold in ordered
        if fold.test_idx.size
    ]
    extra.pop("name", None)
    return MultiPeriodPortfolio(portfolios=portfolios, **extra)
