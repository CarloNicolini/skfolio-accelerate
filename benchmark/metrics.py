"""Timing aggregates, speed-up, Sharpe comparisons, and result schema helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCHEMA_FIELDS = [
    "schema_version",
    "timestamp",
    "git_sha",
    "git_branch",
    "dataset",
    "n_observations",
    "n_assets",
    "cv",
    "n_folds",
    "estimator",
    "objective",
    "risk",
    "extra",
    "method",
    "backend",
    "reason",
    "fallback_reason",
    "solver",
    "n_solves",
    "n_warm_starts",
    "n_rebuilds",
    "n_prior_fits",
    "n_prior_updates",
    "time_s",
    "time_s_mean",
    "time_s_std",
    "time_s_min",
    "time_s_max",
    "n_repetitions",
    "raw_times_s",
    "delta_time_s",
    "relative_time",
    "speedup",
    "mean_sharpe",
    "delta_sharpe",
    "relative_sharpe_error",
    "max_abs_sharpe_diff",
    "n_failed_folds",
    "n_invalid_outputs",
    "n_nonfinite_weights",
    "max_abs_weight_diff",
    "solver_status",
    "validation_ok",
    "cache_warning",
    "workers",
    "thread_limit",
    "n_jobs",
    "status",
    "error",
]


def median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    ordered = sorted(finite)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def stdev(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    mu = sum(finite) / len(finite)
    var = sum((x - mu) ** 2 for x in finite) / (len(finite) - 1)
    return float(math.sqrt(var))


def timing_summary(raw_times: list[float]) -> dict[str, float | int]:
    """Median (reported time), mean, sample std, min, max, and repetition count."""
    finite = [float(v) for v in raw_times if math.isfinite(v)]
    return {
        "time_s": median(finite),
        "time_s_mean": mean(finite),
        "time_s_std": stdev(finite),
        "time_s_min": min(finite) if finite else float("nan"),
        "time_s_max": max(finite) if finite else float("nan"),
        "n_repetitions": len(finite),
    }


def speedup(native_time: float, accelerated_time: float) -> float:
    """``native_time / accelerated_time``. Values ``> 1`` are speed-ups."""
    if (
        not math.isfinite(native_time)
        or not math.isfinite(accelerated_time)
        or accelerated_time <= 0
    ):
        return float("nan")
    return float(native_time / accelerated_time)


def delta_time(native_time: float, accelerated_time: float) -> float:
    """``accelerated_time - native_time`` (negative means faster)."""
    if not math.isfinite(native_time) or not math.isfinite(accelerated_time):
        return float("nan")
    return float(accelerated_time - native_time)


def relative_time(native_time: float, accelerated_time: float) -> float:
    """``accelerated_time / native_time``. Values ``< 1`` are faster."""
    if (
        not math.isfinite(native_time)
        or native_time <= 0
        or not math.isfinite(accelerated_time)
    ):
        return float("nan")
    return float(accelerated_time / native_time)


def delta_sharpe(native_sharpe: float, accelerated_sharpe: float) -> float:
    if not math.isfinite(native_sharpe) or not math.isfinite(accelerated_sharpe):
        return float("nan")
    return float(accelerated_sharpe - native_sharpe)


def relative_sharpe_error(native_sharpe: float, accelerated_sharpe: float) -> float:
    """``(accelerated - native) / |native|``; undefined when native is 0."""
    if (
        not math.isfinite(native_sharpe)
        or not math.isfinite(accelerated_sharpe)
        or native_sharpe == 0.0
    ):
        return float("nan")
    return float((accelerated_sharpe - native_sharpe) / abs(native_sharpe))


def relative_delta_pct(base_time: float, head_time: float) -> float:
    """Percentage change: ``100 * (head_time - base_time) / base_time``.

    Positive values mean the head commit is slower. Undefined when ``base_time``
    is missing, non-finite, or zero.
    """
    if not math.isfinite(base_time) or not math.isfinite(head_time) or base_time == 0.0:
        return float("nan")
    return float(100.0 * (head_time - base_time) / base_time)


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def attach_native_comparisons(
    row: dict[str, Any], native: dict[str, Any] | None
) -> dict[str, Any]:
    """Fill Δ time / speed-up / Sharpe deltas versus the native row."""
    updated = dict(row)
    if native is None or row.get("method") == "native":
        if row.get("method") == "native":
            ok = str(row.get("status")) == "ok"
            updated["delta_time_s"] = 0.0 if ok else float("nan")
            updated["relative_time"] = 1.0 if ok else float("nan")
            updated["speedup"] = 1.0 if ok else float("nan")
            updated["delta_sharpe"] = 0.0 if ok else float("nan")
            updated["relative_sharpe_error"] = 0.0 if ok else float("nan")
        return updated
    native_t = _as_float(native.get("time_s", float("nan")))
    acc_t = _as_float(row.get("time_s", float("nan")))
    native_s = _as_float(native.get("mean_sharpe", float("nan")))
    acc_s = _as_float(row.get("mean_sharpe", float("nan")))
    updated["delta_time_s"] = delta_time(native_t, acc_t)
    updated["relative_time"] = relative_time(native_t, acc_t)
    updated["speedup"] = speedup(native_t, acc_t)
    updated["delta_sharpe"] = delta_sharpe(native_s, acc_s)
    updated["relative_sharpe_error"] = relative_sharpe_error(native_s, acc_s)
    return updated


def parse_raw_times(value: Any) -> list[float]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        inner = text.strip("[]")
        if not inner:
            return []
        return [float(part) for part in inner.split(",") if part.strip()]
    return [float(part) for part in text.split("|") if part.strip()]


def format_raw_times(values: list[float]) -> str:
    return "|".join(f"{v:.9g}" for v in values)


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
