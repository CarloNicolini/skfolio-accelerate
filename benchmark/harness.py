"""CV splitters, timed cells, comparisons, and result files."""

import csv
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.base import clone

from benchmark.config import SCHEMA_VERSION, BenchmarkConfig
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.predict import classify_call
from skfolio_accelerate.scoring import path_sharpes

SCHEMA_FIELDS = """
schema_version timestamp git_sha git_branch dataset n_observations n_assets
cv n_folds estimator objective risk extra method backend reason
fallback_reason solver n_solves n_warm_starts n_rebuilds n_prior_fits
n_prior_updates time_s time_s_mean time_s_std time_s_min time_s_max
n_repetitions raw_times_s delta_time_s relative_time speedup mean_sharpe
delta_sharpe relative_sharpe_error max_abs_sharpe_diff n_failed_folds
n_invalid_outputs n_nonfinite_weights max_abs_weight_diff solver_status
validation_ok cache_warning workers thread_limit n_jobs status error
""".split()
REPORT_KEYS = (
    "backend reason fallback_reason n_solves n_warm_starts n_rebuilds "
    "n_prior_fits n_prior_updates"
).split()
SUMMARY_COLUMNS = [
    ("dataset", "Dataset"),
    ("estimator", "Estimator"),
    ("method", "Method"),
    ("time_s", "Time (s)"),
    ("delta_time_s", "Δ Time (s)"),
    ("relative_time", "Relative Time"),
    ("speedup", "Speed-up"),
    ("mean_sharpe", "Mean Sharpe"),
    ("delta_sharpe", "Δ Sharpe"),
    ("relative_sharpe_error", "Relative Sharpe Error"),
]


class BenchmarkTimeout(Exception):
    pass


def apply_thread_limits(config: BenchmarkConfig) -> None:
    limit = str(config.thread_limit)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = limit


def make_cv(kind: str, config: BenchmarkConfig, n_observations: int):
    from skfolio.model_selection import (
        CombinatorialPurgedCV,
        MultipleRandomizedCV,
        WalkForward,
    )

    test = config.test_size
    folds = min(
        config.target_folds,
        max(1, (n_observations - config.min_train_size) // test),
    )
    if kind == "walk-forward":
        return WalkForward(train_size=n_observations - folds * test, test_size=test)
    if kind == "multiple-randomized":
        return MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=config.mrc_train_size, test_size=test),
            n_subsamples=config.mrc_n_subsamples,
            asset_subset_size=config.mrc_asset_subset_size,
            window_size=min(config.mrc_window_size, n_observations),
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


def fold_index_fingerprint(X, cv) -> tuple:
    folds = []
    for fold in compile_cv_plan(cv, X).folds:
        train = tuple(int(i) for i in np.asarray(fold.train_idx))
        test = tuple(int(i) for i in np.asarray(fold.test_idx))
        assets = (
            tuple(int(i) for i in np.asarray(fold.asset_idx))
            if fold.asset_idx is not None
            else ()
        )
        folds.append((train, test, assets))
    return tuple(folds)


def as_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def timing_summary(raw_times: list[float]) -> dict:
    finite = [float(v) for v in raw_times if math.isfinite(v)]
    nan = float("nan")
    if not finite:
        return {
            "time_s": nan,
            "time_s_mean": nan,
            "time_s_std": nan,
            "time_s_min": nan,
            "time_s_max": nan,
            "n_repetitions": 0,
        }
    return {
        "time_s": statistics.median(finite),
        "time_s_mean": statistics.mean(finite),
        "time_s_std": statistics.stdev(finite) if len(finite) > 1 else nan,
        "time_s_min": min(finite),
        "time_s_max": max(finite),
        "n_repetitions": len(finite),
    }


def speedup(native_time: float, accelerated_time: float) -> float:
    if not math.isfinite(native_time) or accelerated_time <= 0:
        return float("nan")
    return native_time / accelerated_time


def delta_time(native_time: float, accelerated_time: float) -> float:
    if not math.isfinite(native_time) or not math.isfinite(accelerated_time):
        return float("nan")
    return accelerated_time - native_time


def relative_time(native_time: float, accelerated_time: float) -> float:
    if native_time <= 0 or not math.isfinite(accelerated_time):
        return float("nan")
    return accelerated_time / native_time


def delta_sharpe(native_sharpe: float, accelerated_sharpe: float) -> float:
    if not math.isfinite(native_sharpe) or not math.isfinite(accelerated_sharpe):
        return float("nan")
    return accelerated_sharpe - native_sharpe


def relative_sharpe_error(native_sharpe: float, accelerated_sharpe: float) -> float:
    if (
        not math.isfinite(native_sharpe)
        or not math.isfinite(accelerated_sharpe)
        or native_sharpe == 0.0
    ):
        return float("nan")
    return (accelerated_sharpe - native_sharpe) / abs(native_sharpe)


def relative_delta_pct(base_time: float, head_time: float) -> float:
    if not math.isfinite(base_time) or not math.isfinite(head_time) or base_time == 0.0:
        return float("nan")
    return 100.0 * (head_time - base_time) / base_time


def attach_native_comparisons(row: dict, native: dict | None) -> dict:
    updated = dict(row)
    if native is None or row.get("method") == "native":
        if row.get("method") == "native":
            ok = str(row.get("status")) == "ok"
            fill = 0.0 if ok else float("nan")
            updated["delta_time_s"] = 0.0 if ok else float("nan")
            updated["relative_time"] = 1.0 if ok else float("nan")
            updated["speedup"] = 1.0 if ok else float("nan")
            updated["delta_sharpe"] = fill
            updated["relative_sharpe_error"] = fill
        return updated
    native_t, acc_t = as_float(native.get("time_s")), as_float(row.get("time_s"))
    native_s = as_float(native.get("mean_sharpe"))
    acc_s = as_float(row.get("mean_sharpe"))
    updated["delta_time_s"] = delta_time(native_t, acc_t)
    updated["relative_time"] = relative_time(native_t, acc_t)
    updated["speedup"] = speedup(native_t, acc_t)
    updated["delta_sharpe"] = delta_sharpe(native_s, acc_s)
    updated["relative_sharpe_error"] = relative_sharpe_error(native_s, acc_s)
    return updated


def parse_raw_times(value) -> list[float]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        return [float(part) for part in text.strip("[]").split(",") if part.strip()]
    return [float(part) for part in text.split("|") if part.strip()]


def format_raw_times(values: list[float]) -> str:
    return "|".join(f"{v:.9g}" for v in values)


def git_metadata() -> dict:
    def git(*args):
        try:
            done = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return done.stdout.strip() or None if done.returncode == 0 else None

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_sha_short": git("rev-parse", "--short", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": git("status", "--porcelain"),
    }


def collect_environment(config: BenchmarkConfig) -> dict:
    packages = {}
    for name in (
        "skfolio",
        "skfolio-accelerate",
        "numpy",
        "scipy",
        "pandas",
        "cvxpy",
        "clarabel",
        "osqp",
        "highspy",
        "scs",
        "scikit-learn",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **git_metadata(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu_model": platform.processor() or None,
        "logical_cores": os.cpu_count(),
        "configured_workers": config.workers,
        "thread_limit": config.thread_limit,
        "n_jobs": config.n_jobs,
        "packages": packages,
        "config": config.to_dict(),
        "speedup_definition": "native_time / accelerated_time",
        "reported_time": "median of raw repetitions (warm-ups excluded)",
    }


def _call_with_timeout(timeout_s: float | None, fn):
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


def _segment_weights(prediction) -> list:
    if prediction is None:
        return []
    portfolios = getattr(prediction, "portfolios", None)
    if portfolios is not None:
        return [np.asarray(port.weights, dtype=np.float64) for port in portfolios]
    weights = getattr(prediction, "weights", None)
    if weights is not None:
        return [np.asarray(weights, dtype=np.float64)]
    return [w for path in prediction for w in _segment_weights(path)]


def _report_fields(report, backend) -> dict:
    if report is None:
        return {"backend": backend, **{k: None for k in REPORT_KEYS if k != "backend"}}
    return {k: getattr(report, k) for k in REPORT_KEYS}


def validate_prediction(prediction, *, report=None, reference_prediction=None) -> dict:
    sharpes = np.asarray(path_sharpes(prediction), dtype=np.float64)
    weights = _segment_weights(prediction)
    n_invalid = int(np.size(sharpes) - np.isfinite(sharpes).sum())
    n_nonfinite_w = sum(int(np.size(w) - np.isfinite(w).sum()) for w in weights)
    status = (
        "invalid_output" if n_invalid or n_nonfinite_w or prediction is None else "ok"
    )
    max_abs_w = max_abs_sharpe = float("nan")
    if reference_prediction is not None:
        ref_s = np.asarray(path_sharpes(reference_prediction), dtype=np.float64)
        if ref_s.shape == sharpes.shape and ref_s.size:
            max_abs_sharpe = float(np.nanmax(np.abs(sharpes - ref_s)))
        ref_w = _segment_weights(reference_prediction)
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
        if report.fallback_reason and report.backend in {"sklearn", "fit-assemble"}:
            solver_status = f"{report.backend}:{report.fallback_reason}"
    finite = sharpes[np.isfinite(sharpes)]
    return {
        "mean_sharpe": float(np.mean(finite)) if finite.size else float("nan"),
        "n_failed_folds": 1 if prediction is None else 0,
        "n_invalid_outputs": n_invalid,
        "n_nonfinite_weights": n_nonfinite_w,
        "max_abs_weight_diff": max_abs_w,
        "max_abs_sharpe_diff": max_abs_sharpe,
        "solver_status": solver_status,
        "validation_ok": status == "ok",
        "status": status,
    }


def run_method_cell(
    *, method: str, estimator, X, cv_kind: str, config: BenchmarkConfig
) -> dict:
    n_jobs, n_obs = config.n_jobs, X.shape[0]
    from skfolio.model_selection import cross_val_predict as skfolio_cv_predict

    from skfolio_accelerate import cross_val_predict as accel_cv_predict

    if method == "native":
        expected_backend = "sklearn"

        def invoke():
            return skfolio_cv_predict(
                clone(estimator), X, cv=make_cv(cv_kind, config, n_obs), n_jobs=n_jobs
            )

    elif method == "accelerated":
        expected_backend = classify_call(
            estimator, cv=make_cv(cv_kind, config, n_obs), n_jobs=n_jobs
        ).auto_backend(estimator)

        def invoke():
            return accel_cv_predict(
                clone(estimator),
                X,
                cv=make_cv(cv_kind, config, n_obs),
                n_jobs=n_jobs,
                return_report=True,
            )

    else:
        raise ValueError(f"unknown method {method!r}")

    try:
        for _ in range(config.warmups):
            _call_with_timeout(config.timeout_s, invoke)
        validated = _call_with_timeout(config.timeout_s, invoke)
    except Exception as error:
        status = (
            "timeout"
            if isinstance(error, BenchmarkTimeout)
            else f"{type(error).__name__}: {str(error).splitlines()[0]}"
        )
        return {
            "method": method,
            "status": status,
            "error": str(error).splitlines()[0],
            "raw_times": [],
            "prediction": None,
            "report": None,
            **_report_fields(None, expected_backend),
            **timing_summary([]),
            "validation_ok": False,
            "cache_warning": False,
        }

    prediction, report = validated if method == "accelerated" else (validated, None)
    diagnostics = validate_prediction(prediction, report=report)
    fields = _report_fields(report, expected_backend)
    if not diagnostics["validation_ok"]:
        return {
            "method": method,
            "prediction": prediction,
            "report": report,
            "raw_times": [],
            "cache_warning": False,
            "error": None,
            **fields,
            **timing_summary([]),
            **diagnostics,
        }

    raw_times, timed_error = [], None
    for _ in range(config.repetitions):
        started = time.perf_counter()
        try:
            _call_with_timeout(config.timeout_s, invoke)
        except Exception as error:
            timed_error = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            break
        raw_times.append(time.perf_counter() - started)
    return {
        "method": method,
        "prediction": prediction,
        "report": report,
        "raw_times": raw_times,
        "cache_warning": any(t < 1e-9 for t in raw_times),
        "error": timed_error,
        **fields,
        **timing_summary(raw_times),
        **diagnostics,
        "status": timed_error or diagnostics["status"],
    }


def apply_comparisons(rows: list[dict]) -> list[dict]:
    def key(row):
        return tuple(
            row.get(k)
            for k in ("dataset", "cv", "estimator", "objective", "risk", "extra")
        )

    natives = {key(row): row for row in rows if row.get("method") == "native"}
    return [
        attach_native_comparisons(
            row, natives.get(key(row)) if row.get("method") != "native" else row
        )
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if "raw_times" in payload and "raw_times_s" not in payload:
                payload["raw_times_s"] = payload["raw_times"]
            writer.writerow(
                {
                    k: (
                        format_raw_times([float(v) for v in payload[k]])
                        if isinstance(payload.get(k), list)
                        else ("" if payload.get(k) is None else payload.get(k))
                    )
                    for k in SCHEMA_FIELDS
                }
            )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_summary_md(path: Path, *, rows: list[dict], environment: dict) -> None:
    packages = environment.get("packages") or {}
    numeric = {
        k for k, _ in SUMMARY_COLUMNS if k not in {"dataset", "estimator", "method"}
    }
    header = " | ".join(title for _, title in SUMMARY_COLUMNS)
    lines = [
        "# cross_val_predict benchmark summary",
        "",
        f"- timestamp: `{environment.get('timestamp')}`",
        f"- git: `{environment.get('git_sha')}` (`{environment.get('git_branch')}`)",
        f"- skfolio: `{packages.get('skfolio')}`",
        f"- skfolio-accelerate: `{packages.get('skfolio-accelerate')}`",
        f"- python: `{environment.get('python_version')}`",
        f"- cpu: `{environment.get('cpu_model')}`",
        (
            f"- workers / threads: `{environment.get('configured_workers')}`"
            f" / `{environment.get('thread_limit')}`"
        ),
        "- speed-up: `native_time / accelerated_time` (median, no warm-ups)",
        "",
        f"| {header} |",
        f"| {' | '.join('---' for _ in SUMMARY_COLUMNS)} |",
    ]
    for row in rows:
        cells = []
        for key, _title in SUMMARY_COLUMNS:
            value = row.get(key)
            if key not in numeric:
                cells.append("" if value is None else str(value))
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                cells.append("" if value is None else str(value))
                continue
            digits = 3 if key == "speedup" else 4
            cells.append("" if number != number else f"{number:.{digits}f}")
        lines.append(f"| {' | '.join(cells)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_results_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_run_dir(root: Path, git_sha_short: str | None) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha = git_sha_short or "unknown"
    path, suffix = root / f"{day}_{sha}", 2
    while path.exists():
        path = root / f"{day}_{sha}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_in_run_rows(base_rows: list[dict], head_rows: list[dict]) -> list[dict]:
    def key(row):
        return tuple(
            str(row.get(k) or "") for k in ("dataset", "cv", "estimator", "method")
        )

    base_map, head_map = {key(r): r for r in base_rows}, {key(r): r for r in head_rows}
    compared = []
    for k in sorted(set(base_map) | set(head_map)):
        base, head = base_map.get(k, {}), head_map.get(k, {})
        base_t, head_t = as_float(base.get("time_s")), as_float(head.get("time_s"))
        base_s = as_float(base.get("mean_sharpe"))
        head_s = as_float(head.get("mean_sharpe"))
        compared.append(
            {
                "dataset": k[0],
                "cv": k[1],
                "estimator": k[2],
                "method": k[3],
                "base_time_s": base_t,
                "head_time_s": head_t,
                "delta_time_s": (
                    head_t - base_t
                    if math.isfinite(base_t) and math.isfinite(head_t)
                    else float("nan")
                ),
                "delta_pct": relative_delta_pct(base_t, head_t),
                "base_speedup": as_float(base.get("speedup")),
                "head_speedup": as_float(head.get("speedup")),
                "base_mean_sharpe": base_s,
                "head_mean_sharpe": head_s,
                "delta_sharpe": delta_sharpe(base_s, head_s),
                "base_status": base.get("status") or "missing",
                "head_status": head.get("status") or "missing",
            }
        )
    return compared


def write_relative_artifacts(
    out_dir: Path, rows: list[dict], payload: dict, refs: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = """
dataset cv estimator method base_time_s head_time_s delta_time_s delta_pct
base_speedup head_speedup base_mean_sharpe head_mean_sharpe delta_sharpe
base_status head_status
""".split()
    with (out_dir / "delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "delta.json", payload)

    def cell(value, digits):
        number = as_float(value)
        return "" if not math.isfinite(number) else f"{number:.{digits}f}"

    lines = [
        "# In-run relative benchmark",
        "",
        f"- base (timed first): `{refs['base_ref']}` (`{refs['base_sha']}`)",
        f"- head (timed second): `{refs['head_ref']}` (`{refs['head_sha']}`)",
        "- same host, same Python environment, identical CLI flags",
        "- Δ% = `100 * (head_time - base_time) / base_time` (positive = head slower)",
        "",
        "| Dataset | Estimator | Method | Base (s) | Head (s) | Δ Time (s) | Δ% |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['estimator']} | {row['method']} | "
            f"{cell(row['base_time_s'], 4)} | {cell(row['head_time_s'], 4)} | "
            f"{cell(row['delta_time_s'], 4)} | {cell(row['delta_pct'], 2)} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
