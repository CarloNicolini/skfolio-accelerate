"""CV splitters, timed execution, and correctness validation."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from benchmark.config import BenchmarkConfig
from benchmark.estimators import EstimatorSpec
from benchmark.metrics import nanmean, timing_summary
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.predict import classify_call
from skfolio_accelerate.scoring import path_sharpes


class BenchmarkTimeout(Exception):
    """Raised when a single ``cross_val_predict`` call exceeds ``timeout_s``."""


def apply_thread_limits(config: BenchmarkConfig) -> None:
    """Pin OpenMP/BLAS threads. Call before heavy numeric work in this process.

    ``n_jobs != 1`` on :func:`skfolio_accelerate.cross_val_predict` selects
    unmodified skfolio, so the canonical comparison keeps ``n_jobs=1`` and
    uses ``workers`` / ``thread_limit`` only as explicit thread caps.
    """
    import os

    limit = str(config.thread_limit)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = limit


def make_cv(kind: str, config: BenchmarkConfig):
    from skfolio.model_selection import (
        CombinatorialPurgedCV,
        MultipleRandomizedCV,
        WalkForward,
    )

    if kind == "walk-forward":
        return WalkForward(train_size=config.train_size, test_size=config.test_size)
    if kind == "multiple-randomized":
        return MultipleRandomizedCV(
            walk_forward=WalkForward(
                train_size=config.train_size, test_size=config.test_size
            ),
            n_subsamples=config.mrc_n_subsamples,
            asset_subset_size=config.mrc_asset_subset_size,
            window_size=config.mrc_window_size,
            random_state=config.mrc_random_state,
        )
    if kind == "purged-cpcv":
        return CombinatorialPurgedCV(
            n_folds=config.cpcv_n_folds,
            n_test_folds=config.cpcv_n_test_folds,
            purged_size=config.cpcv_purged_size,
            embargo_size=config.cpcv_embargo_size,
        )
    raise ValueError(f"unknown cv kind {kind!r}")


def fold_index_fingerprint(X, cv) -> tuple[tuple[int, ...], ...]:
    """Stable train/test index tuples for equality checks across methods."""
    plan = compile_cv_plan(cv, X)
    folds = []
    for fold in plan.folds:
        train = tuple(int(i) for i in np.asarray(fold.train_idx))
        test = tuple(int(i) for i in np.asarray(fold.test_idx))
        assets = (
            tuple(int(i) for i in np.asarray(fold.asset_idx))
            if fold.asset_idx is not None
            else ()
        )
        folds.append((train, test, assets))
    return tuple(folds)


def skip_extra_for_cv(spec: EstimatorSpec, cv_kind: str) -> bool:
    """Match sequential benchmark: extras are skipped on MultipleRandomizedCV."""
    return (
        bool(spec.extra) and spec.extra != "l2_0" and cv_kind == "multiple-randomized"
    )


def _call_with_timeout(timeout_s: float | None, fn: Callable[[], Any]) -> Any:
    if timeout_s is None:
        return fn()

    def _handle(_signum, _frame):
        raise BenchmarkTimeout(f"exceeded {timeout_s}s")

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _native_predict(estimator, X, cv, n_jobs: int):
    from skfolio.model_selection import cross_val_predict as skfolio_cv_predict

    return skfolio_cv_predict(clone(estimator), X, cv=cv, n_jobs=n_jobs)


def _accelerated_predict(estimator, X, cv, n_jobs: int):
    from skfolio_accelerate import cross_val_predict

    return cross_val_predict(
        clone(estimator), X, cv=cv, n_jobs=n_jobs, return_report=True
    )


def collect_segment_weights(prediction) -> list[np.ndarray]:
    """Flatten MultiPeriodPortfolio / Population segment weights."""
    if prediction is None:
        return []
    if type(prediction).__name__ == "Population" or (
        hasattr(prediction, "__len__") and not hasattr(prediction, "sharpe_ratio")
    ):
        weights: list[np.ndarray] = []
        for path in prediction:
            weights.extend(collect_segment_weights(path))
        return weights
    portfolios = getattr(prediction, "portfolios", None)
    if portfolios is not None:
        return [np.asarray(port.weights, dtype=np.float64) for port in portfolios]
    if hasattr(prediction, "weights"):
        return [np.asarray(prediction.weights, dtype=np.float64)]
    return []


def validate_prediction(
    prediction,
    *,
    report=None,
    reference_prediction=None,
) -> dict[str, Any]:
    """Detect NaNs, empty outputs, and optional weight disagreement vs native."""
    sharpes = np.asarray(path_sharpes(prediction), dtype=np.float64)
    weights = collect_segment_weights(prediction)
    n_invalid = int(np.size(sharpes) - np.isfinite(sharpes).sum())
    n_nonfinite_w = sum(int(np.size(w) - np.isfinite(w).sum()) for w in weights)
    n_failed = 0
    status = "ok"
    if report is not None and getattr(report, "fallback_reason", None):
        status = "ok_with_fallback"
    if n_invalid or n_nonfinite_w:
        status = "invalid_output"
    if prediction is None:
        status = "invalid_output"
        n_failed = 1
    max_abs_w = float("nan")
    max_abs_sharpe = float("nan")
    if reference_prediction is not None:
        ref_s = np.asarray(path_sharpes(reference_prediction), dtype=np.float64)
        if ref_s.shape == sharpes.shape and ref_s.size:
            max_abs_sharpe = float(np.nanmax(np.abs(sharpes - ref_s)))
        ref_w = collect_segment_weights(reference_prediction)
        if ref_w and len(ref_w) == len(weights):
            diffs = [
                float(np.nanmax(np.abs(a - b)))
                for a, b in zip(weights, ref_w, strict=True)
                if a.shape == b.shape
            ]
            if diffs:
                max_abs_w = max(diffs)
    solver_status = "unknown"
    if report is not None:
        solver_status = str(report.backend)
        if report.fallback_reason:
            solver_status = f"{report.backend}:fallback"
    validation_ok = status in {"ok", "ok_with_fallback"} and n_invalid == 0
    return {
        "mean_sharpe": nanmean(sharpes),
        "n_failed_folds": n_failed,
        "n_invalid_outputs": n_invalid,
        "n_nonfinite_weights": n_nonfinite_w,
        "max_abs_weight_diff": max_abs_w,
        "max_abs_sharpe_diff": max_abs_sharpe,
        "solver_status": solver_status,
        "validation_ok": validation_ok,
        "status": status if validation_ok or status != "ok" else status,
    }


def run_method_cell(
    *,
    method: str,
    estimator,
    X: pd.DataFrame,
    cv_factory: Callable[[], Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Warm-up, validate (untimed), then time ``repetitions`` isolated calls."""
    n_jobs = config.n_jobs
    probe_cv = cv_factory()
    if method == "native":

        def invoke():
            return _native_predict(estimator, X, cv_factory(), n_jobs)

    elif method == "accelerated":

        def invoke():
            return _accelerated_predict(estimator, X, cv_factory(), n_jobs)

    else:
        raise ValueError(f"unknown method {method!r}")

    expected_backend = None
    if method == "accelerated":
        expected_backend = classify_call(
            estimator, cv=probe_cv, n_jobs=n_jobs
        ).auto_backend(estimator)

    for _ in range(config.warmups):
        _call_with_timeout(config.timeout_s, invoke)

    prediction = None
    report = None
    try:
        validated = _call_with_timeout(config.timeout_s, invoke)
    except BenchmarkTimeout as error:
        return {
            "method": method,
            "status": "timeout",
            "error": str(error),
            "raw_times": [],
            "backend": expected_backend,
            **timing_summary([]),
            "validation_ok": False,
            "cache_warning": False,
        }
    except Exception as error:  # noqa: BLE001 — cell isolation
        return {
            "method": method,
            "status": f"{type(error).__name__}: {str(error).splitlines()[0]}",
            "error": str(error).splitlines()[0],
            "raw_times": [],
            "backend": expected_backend,
            **timing_summary([]),
            "validation_ok": False,
            "cache_warning": False,
        }

    if method == "accelerated":
        prediction, report = validated
    else:
        prediction = validated

    diagnostics = validate_prediction(prediction, report=report)
    if not diagnostics["validation_ok"]:
        return {
            "method": method,
            "prediction": prediction,
            "report": report,
            "backend": getattr(report, "backend", expected_backend or "sklearn"),
            "reason": getattr(report, "reason", None),
            "fallback_reason": getattr(report, "fallback_reason", None),
            "n_solves": getattr(report, "n_solves", None),
            "n_warm_starts": getattr(report, "n_warm_starts", None),
            "n_rebuilds": getattr(report, "n_rebuilds", None),
            "n_prior_fits": getattr(report, "n_prior_fits", None),
            "n_prior_updates": getattr(report, "n_prior_updates", None),
            "raw_times": [],
            **timing_summary([]),
            **diagnostics,
            "cache_warning": False,
            "error": None,
        }

    raw_times: list[float] = []
    timed_error = None
    for _ in range(config.repetitions):
        started = time.perf_counter()
        try:
            _call_with_timeout(config.timeout_s, invoke)
        except Exception as error:  # noqa: BLE001
            timed_error = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            break
        raw_times.append(time.perf_counter() - started)

    cache_warning = any(t < 1e-9 for t in raw_times)
    summary = timing_summary(raw_times)
    status = "ok"
    if timed_error:
        status = timed_error
    return {
        "method": method,
        "prediction": prediction,
        "report": report,
        "backend": getattr(
            report, "backend", "sklearn" if method == "native" else expected_backend
        ),
        "reason": getattr(report, "reason", None),
        "fallback_reason": getattr(report, "fallback_reason", None),
        "n_solves": getattr(report, "n_solves", None),
        "n_warm_starts": getattr(report, "n_warm_starts", None),
        "n_rebuilds": getattr(report, "n_rebuilds", None),
        "n_prior_fits": getattr(report, "n_prior_fits", None),
        "n_prior_updates": getattr(report, "n_prior_updates", None),
        "raw_times": raw_times,
        **summary,
        **diagnostics,
        "cache_warning": cache_warning,
        "status": status,
        "error": timed_error,
        "expected_backend": expected_backend,
    }


def empty_row_base(
    *,
    timestamp: str,
    git_sha: str | None,
    git_branch: str | None,
    dataset,
    cv_kind: str,
    n_folds: int,
    spec: EstimatorSpec,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    from benchmark.config import SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "git_branch": git_branch,
        "dataset": dataset.name,
        "n_observations": int(dataset.X.shape[0]),
        "n_assets": int(dataset.X.shape[1]),
        "cv": cv_kind,
        "n_folds": n_folds,
        "estimator": spec.name,
        "objective": spec.objective,
        "risk": spec.risk,
        "extra": spec.extra,
        "solver": config.solver,
        "workers": config.workers,
        "thread_limit": config.thread_limit,
        "n_jobs": config.n_jobs,
    }
