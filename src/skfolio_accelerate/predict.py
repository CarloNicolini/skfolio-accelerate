"""Drop-in for ``skfolio.model_selection.cross_val_predict``.

Compact OSQP/Clarabel kernels accelerate a subset of MeanRisk. EqualWeighted,
Random, and default InverseVolatility use closed-form weights. Remaining
estimators still call native ``fit``, but reuse the compiled CV plan and
assemble portfolios from ``weights_`` so they skip joblib, train/test copies,
and ``predict()`` construction when ``n_jobs`` is serial.
"""

from __future__ import annotations

import copy
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    Random,
)
from skfolio.optimization.convex import ObjectiveFunction
from skfolio.utils.stats import rand_weights_dirichlet
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from skfolio_accelerate.compact import EngineCache, estimator_spec
from skfolio_accelerate.cv_plan import FoldSpec, compile_cv_plan, cpcv_fold_blocks
from skfolio_accelerate.moments import (
    OverlapMomentCache,
    as_float_2d,
    is_default_empirical,
)
from skfolio_accelerate.scoring import assemble_prediction, window_view

_SUPPORTED_OBJECTIVES = {
    ObjectiveFunction.MINIMIZE_RISK,
    ObjectiveFunction.MAXIMIZE_UTILITY,
}
_SUPPORTED_RISKS = {
    RiskMeasure.VARIANCE,
    RiskMeasure.SEMI_VARIANCE,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.WORST_REALIZATION,
    RiskMeasure.CVAR,
    RiskMeasure.EVAR,
    RiskMeasure.MAX_DRAWDOWN,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.CDAR,
    RiskMeasure.EDAR,
}
_UNSUPPORTED_IF_SET = (
    ("min_budget", "minimum budget"),
    ("max_budget", "maximum budget"),
    ("max_short", "maximum short exposure"),
    ("max_long", "maximum long exposure"),
    ("cardinality", "cardinality (MIP)"),
    ("group_cardinalities", "group_cardinalities (MIP)"),
    ("threshold_long", "threshold_long (MIP)"),
    ("threshold_short", "threshold_short (MIP)"),
    ("previous_weights", "previous weights"),
    ("target_weights", "target weights"),
    ("groups", "groups"),
    ("linear_constraints", "linear constraints"),
    ("left_inequality", "left inequality"),
    ("right_inequality", "right inequality"),
    ("add_constraints", "add_constraints"),
    ("add_objective", "add_objective"),
    ("overwrite_expected_return", "overwrite_expected_return"),
    ("efficient_frontier_size", "efficient_frontier_size"),
    ("mu_uncertainty_set_estimator", "mu uncertainty sets"),
    ("covariance_uncertainty_set_estimator", "covariance uncertainty sets"),
    ("min_return", "minimum return"),
    ("max_tracking_error", "maximum tracking error"),
    ("max_turnover", "maximum turnover"),
    ("max_mean_absolute_deviation", "maximum mean absolute deviation"),
    ("max_first_lower_partial_moment", "maximum first lower partial moment"),
    ("max_variance", "maximum variance"),
    ("max_standard_deviation", "maximum standard deviation"),
    ("max_semi_variance", "maximum semi-variance"),
    ("max_semi_deviation", "maximum semi-deviation"),
    ("max_worst_realization", "maximum worst realization"),
    ("max_cvar", "maximum CVaR"),
    ("max_evar", "maximum EVaR"),
    ("max_max_drawdown", "maximum drawdown"),
    ("max_average_drawdown", "maximum average drawdown"),
    ("max_cdar", "maximum CDaR"),
    ("max_edar", "maximum EDaR"),
    ("max_ulcer_index", "maximum ulcer index"),
    ("max_gini_mean_difference", "maximum Gini mean difference"),
    ("solver_params", "custom solver parameters"),
    ("scale_objective", "custom objective scaling"),
    ("scale_constraints", "custom constraint scaling"),
    ("portfolio_params", "estimator portfolio parameters"),
    ("fallback", "fallback estimator"),
)

_CLOSED_FORM_TYPES = (EqualWeighted, InverseVolatility, Random)
_PORTFOLIO_ATTRS = (
    "transaction_costs",
    "management_fees",
    "previous_weights",
    "risk_free_rate",
)


@dataclass
class AccelerationReport:
    backend: str
    n_solves: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_warm_starts: int = 0
    fallback_reason: str | None = None
    moments_s: float = 0.0
    solve_s: float = 0.0
    eval_s: float = 0.0
    wall_s: float = 0.0
    baseline_s: float = 0.0
    speedup: float = float("nan")

    def __str__(self) -> str:
        lines = [
            f"Backend: {self.backend}",
            f"Solves: {self.n_solves}",
            f"Moment fits: {self.n_prior_fits}",
            f"Moment updates: {self.n_prior_updates}",
            f"Warm starts: {self.n_warm_starts}",
            f"moments {self.moments_s:.4f}s  solve {self.solve_s:.4f}s  "
            f"eval {self.eval_s:.4f}s  wall {self.wall_s:.4f}s",
            f"Fallback: {self.fallback_reason or 'none'}",
        ]
        if self.baseline_s > 0:
            lines.append(
                f"Baseline {self.baseline_s:.4f}s  speedup {self.speedup:.2f}×"
            )
        return "\n".join(lines)


def _cap_native_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _nonzero(value: Any) -> bool:
    return bool(np.any(np.abs(np.asarray(value, dtype=float)) > 0))


def blocked_reason(estimator) -> str | None:
    """Why the compact engine cannot run this estimator, or None if it can."""
    if isinstance(estimator, Pipeline):
        return "pipelines use skfolio cross_val_predict"
    if type(estimator) in _CLOSED_FORM_TYPES:
        if getattr(estimator, "fallback", None) is not None:
            return "fallback estimator is not compacted"
        if getattr(estimator, "previous_weights", None) is not None:
            return "previous weights are not compacted"
        if getattr(estimator, "raise_on_failure", True) is not True:
            return "raise_on_failure=False is not compacted"
        if getattr(estimator, "portfolio_params", None) is not None:
            return "estimator portfolio parameters are not compacted"
        if type(estimator) is InverseVolatility and not is_default_empirical(estimator):
            return "custom prior is not compacted"
        return None
    if not isinstance(estimator, MeanRisk):
        return f"estimator {type(estimator).__name__} is not MeanRisk"
    for attr, label in _UNSUPPORTED_IF_SET:
        if getattr(estimator, attr, None) is not None:
            return f"{label} is not compacted"
    if getattr(estimator, "budget", 1.0) is None:
        return "an unspecified equality budget is not compacted"
    if getattr(estimator, "needs_previous_weights", False):
        return "sequential previous_weights (costs, turnover, or fallback)"
    objective = getattr(
        estimator, "objective_function", ObjectiveFunction.MINIMIZE_RISK
    )
    if objective not in _SUPPORTED_OBJECTIVES:
        return "objective_function is not compacted"
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    if risk not in _SUPPORTED_RISKS:
        return "risk_measure is not compacted"
    solver = getattr(estimator, "solver", "CLARABEL")
    if risk is RiskMeasure.VARIANCE:
        if solver not in {"CLARABEL", "OSQP"}:
            return f"solver {solver!r} is not compacted for {risk.name}"
    elif solver != "CLARABEL":
        return f"solver {solver!r} is not compacted for {risk.name}"
    if _nonzero(getattr(estimator, "l1_coef", 0.0)):
        return "l1_coef is not compacted"
    if isinstance(getattr(estimator, "min_weights", 0.0), dict) or isinstance(
        getattr(estimator, "max_weights", 1.0), dict
    ):
        return "dict weight bounds are not compacted"
    if isinstance(getattr(estimator, "min_acceptable_return", None), dict):
        return "dict minimum acceptable returns are not compacted"
    if _nonzero(getattr(estimator, "transaction_costs", 0.0)):
        return "transaction costs are not compacted"
    if _nonzero(getattr(estimator, "management_fees", 0.0)):
        return "management fees are not compacted"
    if _nonzero(getattr(estimator, "risk_free_rate", 0.0)):
        return "a non-zero risk-free rate is not compacted"
    if getattr(estimator, "raise_on_failure", True) is not True:
        return "raise_on_failure=False is not compacted"
    if getattr(estimator, "save_problem", False):
        return "saved CVXPY problem state is not compacted"
    if not is_default_empirical(estimator):
        return "custom prior is not compacted"
    return None


def compact_blocked_reason(
    estimator,
    *,
    y=None,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
) -> str | None:
    """Why this call cannot use the compact engine (estimator or call options)."""
    if method != "predict":
        return "only method='predict' is compacted"
    if params:
        return "fit params use skfolio cross_val_predict"
    if column_indices is not None:
        return "column_indices uses skfolio cross_val_predict"
    if entry_rebalancing_params is not None:
        return "entry_rebalancing_params uses skfolio cross_val_predict"
    return blocked_reason(estimator)


def assemble_blocked_reason(
    estimator,
    *,
    method: str = "predict",
    params: dict | None = None,
    column_indices=None,
    entry_rebalancing_params: dict | None = None,
    n_jobs: int | None = None,
    cv=None,
) -> str | None:
    """Why this call cannot fit natively and assemble from ``weights_``."""
    if method != "predict":
        return "only method='predict' is assembled from weights"
    if params:
        return "fit params use skfolio cross_val_predict"
    if column_indices is not None:
        return "column_indices uses skfolio cross_val_predict"
    if entry_rebalancing_params is not None:
        return "entry_rebalancing_params uses skfolio cross_val_predict"
    if n_jobs not in (None, 1):
        return "n_jobs!=1 uses skfolio cross_val_predict"
    if getattr(cv, "shuffle", False) is True:
        return "shuffled CV uses skfolio cross_val_predict"
    if isinstance(estimator, Pipeline):
        return "pipelines use skfolio cross_val_predict"
    if not isinstance(estimator, BaseOptimization):
        return f"estimator {type(estimator).__name__} is not a portfolio optimizer"
    if getattr(estimator, "needs_previous_weights", False):
        return "sequential previous_weights (costs, turnover, or fallback)"
    if getattr(estimator, "raise_on_failure", True) is not True:
        return "raise_on_failure=False uses skfolio cross_val_predict"
    if getattr(estimator, "efficient_frontier_size", None) is not None:
        return "efficient_frontier_size uses skfolio cross_val_predict"
    return None


def _path_groups(folds: list[FoldSpec]) -> list[list[FoldSpec]]:
    buckets: dict[int, list[FoldSpec]] = defaultdict(list)
    for fold in folds:
        buckets[fold.path_id].append(fold)
    return [buckets[key] for key in sorted(buckets)]


def _run_fold_batch(
    X: np.ndarray,
    folds: list[FoldSpec],
    spec: dict[str, Any],
    *,
    keep_returns: bool,
    fold_blocks: list[np.ndarray] | None,
    engines: EngineCache | None = None,
) -> dict[str, Any]:
    first_assets = folds[0].asset_idx if folds else None
    if first_assets is not None:
        x_work = X[:, np.asarray(first_assets, dtype=np.intp)]
        blocks = None
    else:
        x_work = X
        blocks = fold_blocks
    cache = OverlapMomentCache(x_work, keep_returns=keep_returns, fold_blocks=blocks)
    engines = EngineCache(spec=spec) if engines is None else engines
    warm_before = int(getattr(engines.engine, "n_warm_starts", 0))
    weights: dict[int, np.ndarray] = {}
    moments_s = 0.0
    solve_s = 0.0
    for i, fold in enumerate(folds):
        t0 = time.perf_counter()
        moments = cache.get(fold, path_key=fold.path_id)
        moments_s += time.perf_counter() - t0
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if keep_returns else None,
        )
        t1 = time.perf_counter()
        weights[fold.fold_id] = engine.solve(moments, warm=i > 0)
        solve_s += time.perf_counter() - t1
    n_warm = int(getattr(engines.engine, "n_warm_starts", 0)) - warm_before
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": solve_s,
        "n_solves": len(folds),
        "n_warm_starts": n_warm,
        "n_prior_fits": int(cache.n_fits),
        "n_prior_updates": int(cache.n_updates),
    }


def _merge_batch_results(parts: list[dict[str, Any]]) -> dict[str, Any]:
    weights: dict[int, np.ndarray] = {}
    moments_s = solve_s = 0.0
    n_solves = n_warm = n_fits = n_updates = 0
    for part in parts:
        weights.update(part["weights"])
        moments_s += part["moments_s"]
        solve_s += part["solve_s"]
        n_solves += part["n_solves"]
        n_warm += part["n_warm_starts"]
        n_fits += part["n_prior_fits"]
        n_updates += part["n_prior_updates"]
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": solve_s,
        "n_solves": n_solves,
        "n_warm_starts": n_warm,
        "n_prior_fits": n_fits,
        "n_prior_updates": n_updates,
    }


def _run_closed_form_batch(
    X: np.ndarray,
    folds: list[FoldSpec],
    estimator,
    *,
    fold_blocks: list[np.ndarray] | None,
) -> dict[str, Any]:
    first_assets = folds[0].asset_idx if folds else None
    if first_assets is None:
        x_work = X
        blocks = fold_blocks
    else:
        x_work = X[:, np.asarray(first_assets, dtype=np.intp)]
        blocks = None
    cache = None
    if type(estimator) is InverseVolatility:
        cache = OverlapMomentCache(x_work, keep_returns=False, fold_blocks=blocks)
    weights: dict[int, np.ndarray] = {}
    moments_s = 0.0
    draw_random = type(estimator) is Random
    for fold in folds:
        if cache is None:
            n_assets = x_work.shape[1]
            if draw_random:
                weights[fold.fold_id] = rand_weights_dirichlet(n=n_assets)
            else:
                weights[fold.fold_id] = np.full(n_assets, 1.0 / n_assets)
            continue
        started = time.perf_counter()
        moments = cache.get(fold, path_key=fold.path_id)
        moments_s += time.perf_counter() - started
        inverse_volatility = 1.0 / np.sqrt(np.diag(moments.covariance))
        weights[fold.fold_id] = inverse_volatility / inverse_volatility.sum()
    return {
        "weights": weights,
        "moments_s": moments_s,
        "solve_s": 0.0,
        "n_solves": 0,
        "n_warm_starts": 0,
        "n_prior_fits": 0 if cache is None else cache.n_fits,
        "n_prior_updates": 0 if cache is None else cache.n_updates,
    }


def _as_float_any(data) -> np.ndarray | None:
    if data is None:
        return None
    arr = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
    if arr.dtype != np.float64:
        arr = np.asarray(arr, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _segment_params(estimator) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    own = getattr(estimator, "portfolio_params", None)
    if own:
        extra.update(own)
    for name in _PORTFOLIO_ATTRS:
        if name not in extra and hasattr(estimator, name):
            extra[name] = getattr(estimator, name)
    extra.pop("name", None)
    extra.pop("check_observations_order", None)
    extra.pop("fallback_chain", None)
    return extra


def _train_target(y_arr: np.ndarray | None, fold: FoldSpec, n_assets: int):
    if y_arr is None:
        return None
    cols = fold.asset_idx
    if y_arr.ndim == 1 or cols is None or y_arr.shape[-1] != n_assets:
        return window_view(y_arr, fold.train_idx)
    return window_view(y_arr, fold.train_idx, cols)


def _run_fit_assemble_batch(
    estimator,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    folds: list[FoldSpec],
) -> dict[str, Any]:
    n_assets = int(x_arr.shape[1])
    weights: dict[int, np.ndarray] = {}
    solve_s = 0.0
    for fold in folds:
        started = time.perf_counter()
        fitted = clone(estimator)
        x_train = window_view(x_arr, fold.train_idx, fold.asset_idx)
        y_train = _train_target(y_arr, fold, n_assets)
        if y_train is None:
            fitted.fit(x_train)
        else:
            fitted.fit(x_train, y_train)
        weights_ = np.asarray(fitted.weights_, dtype=np.float64)
        if weights_.ndim != 1:
            raise ValueError("2-dimensional weights_ cannot be assembled")
        weights[fold.fold_id] = np.ascontiguousarray(weights_)
        solve_s += time.perf_counter() - started
    return {
        "weights": weights,
        "moments_s": 0.0,
        "solve_s": solve_s,
        "n_solves": len(folds),
        "n_warm_starts": 0,
        "n_prior_fits": 0,
        "n_prior_updates": 0,
    }


def _fit_assemble_prediction(
    estimator,
    X,
    x_arr: np.ndarray,
    y_arr: np.ndarray | None,
    cv_plan,
    portfolio_params: dict | None,
) -> tuple[Any, dict[str, Any], float]:
    batches = _path_groups(cv_plan.folds) if cv_plan.kind == "mrc" else [cv_plan.folds]
    merged = _merge_batch_results(
        [_run_fit_assemble_batch(estimator, x_arr, y_arr, batch) for batch in batches]
    )
    started = time.perf_counter()
    pred = assemble_prediction(
        X,
        cv_plan,
        merged["weights"],
        name=type(estimator).__name__,
        portfolio_params=portfolio_params,
        segment_params=_segment_params(estimator),
    )
    return pred, merged, time.perf_counter() - started


def _skfolio_predict(
    estimator,
    X,
    y,
    cv,
    *,
    n_jobs,
    method,
    verbose,
    params,
    pre_dispatch,
    column_indices,
    portfolio_params,
    entry_rebalancing_params,
):
    return skfolio_cross_val_predict(
        estimator,
        X,
        y=y,
        cv=cv,
        n_jobs=n_jobs,
        method=method,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        column_indices=column_indices,
        portfolio_params=portfolio_params,
        entry_rebalancing_params=entry_rebalancing_params,
    )


def cross_val_predict(
    estimator,
    X,
    y=None,
    cv=None,
    n_jobs: int | None = None,
    method: str = "predict",
    verbose: int = 0,
    params: dict | None = None,
    pre_dispatch: str = "2*n_jobs",
    column_indices=None,
    portfolio_params: dict | None = None,
    entry_rebalancing_params: dict | None = None,
    *,
    backend: str = "auto",
    return_report: bool = False,
):
    """Drop-in for ``skfolio.model_selection.cross_val_predict``.

    Call signature matches skfolio (plus ``backend`` and ``return_report``).
    Compact OSQP/Clarabel is used only when it is equivalent to MeanRisk.
    Closed-form estimators skip ``fit`` entirely. Other serial calls still use
    native ``fit`` but assemble test portfolios from ``weights_``.
    """
    _cap_native_threads()
    t_wall = time.perf_counter()
    if backend not in {"auto", "compact", "sklearn"}:
        raise ValueError(f"Unknown backend {backend!r}")

    compact_reason = compact_blocked_reason(
        estimator,
        y=y,
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
    )
    assemble_reason = assemble_blocked_reason(
        estimator,
        method=method,
        params=params,
        column_indices=column_indices,
        entry_rebalancing_params=entry_rebalancing_params,
        n_jobs=n_jobs,
        cv=cv,
    )
    if backend == "sklearn" or (
        backend == "auto" and compact_reason is not None and assemble_reason is not None
    ):
        pred = _skfolio_predict(
            estimator,
            X,
            y,
            cv,
            n_jobs=n_jobs,
            method=method,
            verbose=verbose,
            params=params,
            pre_dispatch=pre_dispatch,
            column_indices=column_indices,
            portfolio_params=portfolio_params,
            entry_rebalancing_params=entry_rebalancing_params,
        )
        report = AccelerationReport(
            backend="sklearn",
            fallback_reason=(compact_reason or assemble_reason or "backend=sklearn"),
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred
    if backend == "compact" and compact_reason is not None:
        raise ValueError(
            f"backend={backend!r} cannot compact this predict: {compact_reason}"
        )

    estimator = clone(estimator)
    x_arr = as_float_2d(X)
    y_arr = _as_float_any(y)
    # A compact numerical failure must retry the exact original split plan.
    # Some splitters accept mutable RandomState objects and advance them in split().
    fallback_cv = copy.deepcopy(cv)
    cv_plan = compile_cv_plan(cv, X, y)
    fold_blocks = None
    if cv_plan.kind == "cpcv":
        fold_blocks = cpcv_fold_blocks(x_arr.shape[0], int(cv.n_folds))

    batches = _path_groups(cv_plan.folds) if cv_plan.kind == "mrc" else [cv_plan.folds]
    if compact_reason is None and type(estimator) in _CLOSED_FORM_TYPES:
        merged = _merge_batch_results(
            [
                _run_closed_form_batch(
                    x_arr,
                    batch,
                    estimator,
                    fold_blocks=fold_blocks,
                )
                for batch in batches
            ]
        )
        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged["weights"],
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
        )
        report = AccelerationReport(
            backend="closed-form",
            n_prior_fits=int(merged["n_prior_fits"]),
            n_prior_updates=int(merged["n_prior_updates"]),
            moments_s=float(merged["moments_s"]),
            eval_s=time.perf_counter() - t_eval,
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

    if compact_reason is None:
        spec = estimator_spec(estimator)
        keep_returns = spec["risk_measure"] is not RiskMeasure.VARIANCE
        # MRC paths have the same number of assets. Reuse one solver topology across
        # paths, while deliberately disabling the first warm start of each path.
        shared_engines = (
            EngineCache(spec=spec)
            if cv_plan.kind == "mrc" and spec["risk_measure"] is RiskMeasure.VARIANCE
            else None
        )
        try:
            merged = _merge_batch_results(
                [
                    _run_fold_batch(
                        x_arr,
                        batch,
                        spec,
                        keep_returns=keep_returns,
                        fold_blocks=fold_blocks,
                        engines=shared_engines,
                    )
                    for batch in batches
                ]
            )
        except (RuntimeError, ValueError) as error:
            fail_reason = (
                f"compact {spec['risk_measure'].name} solve failed: "
                f"{type(error).__name__}: {error}"
            )
            if assemble_reason is None:
                pred, merged, eval_s = _fit_assemble_prediction(
                    estimator,
                    X,
                    x_arr,
                    y_arr,
                    cv_plan,
                    portfolio_params,
                )
                report = AccelerationReport(
                    backend="fit-assemble",
                    n_solves=int(merged["n_solves"]),
                    solve_s=float(merged["solve_s"]),
                    eval_s=eval_s,
                    fallback_reason=fail_reason,
                    wall_s=time.perf_counter() - t_wall,
                )
                return (pred, report) if return_report else pred
            pred = _skfolio_predict(
                estimator,
                X,
                y,
                fallback_cv,
                n_jobs=n_jobs,
                method=method,
                verbose=verbose,
                params=params,
                pre_dispatch=pre_dispatch,
                column_indices=column_indices,
                portfolio_params=portfolio_params,
                entry_rebalancing_params=entry_rebalancing_params,
            )
            report = AccelerationReport(
                backend="sklearn",
                fallback_reason=fail_reason,
                wall_s=time.perf_counter() - t_wall,
            )
            return (pred, report) if return_report else pred

        t_eval = time.perf_counter()
        pred = assemble_prediction(
            X,
            cv_plan,
            merged["weights"],
            name=type(estimator).__name__,
            portfolio_params=portfolio_params,
        )
        eval_s = time.perf_counter() - t_eval
        backend_name = (
            "osqp" if spec["risk_measure"] is RiskMeasure.VARIANCE else "clarabel"
        )
        report = AccelerationReport(
            backend=backend_name,
            n_solves=int(merged["n_solves"]),
            n_prior_fits=int(merged["n_prior_fits"]),
            n_prior_updates=int(merged["n_prior_updates"]),
            n_warm_starts=int(merged["n_warm_starts"]),
            moments_s=float(merged["moments_s"]),
            solve_s=float(merged["solve_s"]),
            eval_s=eval_s,
            wall_s=time.perf_counter() - t_wall,
        )
        return (pred, report) if return_report else pred

    pred, merged, eval_s = _fit_assemble_prediction(
        estimator,
        X,
        x_arr,
        y_arr,
        cv_plan,
        portfolio_params,
    )
    report = AccelerationReport(
        backend="fit-assemble",
        n_solves=int(merged["n_solves"]),
        solve_s=float(merged["solve_s"]),
        eval_s=eval_s,
        fallback_reason=compact_reason,
        wall_s=time.perf_counter() - t_wall,
    )
    return (pred, report) if return_report else pred


massive_cross_val_predict = cross_val_predict
