"""Repeated isolated compatibility, timing, and peak-RSS benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from benchmark_coverage import _cases, _cv_cases
from profile_process import measure_call, run_worker
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict

from skfolio_accelerate import (
    cross_val_predict,
    path_sharpes,
    ranking_precision_at_k,
    spearman_rank_correlation,
)
from skfolio_accelerate.flagship import factor_returns


def _native_predict(estimator, X, y, cv, n_jobs):
    return skfolio_cross_val_predict(
        estimator,
        X,
        y=y,
        cv=cv,
        n_jobs=n_jobs,
    )


def _accelerated_predict(estimator, X, y, cv):
    return cross_val_predict(
        estimator,
        X,
        y=y,
        cv=cv,
        n_jobs=1,
        return_report=True,
    )


def _dispersion(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array))))


def _worker(args) -> None:
    case = _cases()[args.case]
    cv_factory = _cv_cases(args.quick)[args.cv][1]
    n_observations = 120 if args.quick else 20 * 252
    n_assets = 6 if args.quick else 20
    X = factor_returns(n_observations, n_assets, seed=42)
    times: list[float] = []
    peaks: list[int] = []
    scores = None
    report = None
    try:
        for _ in range(args.repeats):
            np.random.seed(44)
            estimator = case.factory()
            y = X.iloc[:, 0] if case.needs_target else None
            cv = cv_factory()
            if args.implementation == "native":
                call = partial(
                    _native_predict,
                    estimator,
                    X,
                    y,
                    cv,
                    args.native_n_jobs,
                )
                prediction, wall_s, peak_rss = measure_call(call)
            else:
                call = partial(
                    _accelerated_predict,
                    estimator,
                    X,
                    y,
                    cv,
                )
                result, wall_s, peak_rss = measure_call(call)
                prediction, report = result
            times.append(wall_s)
            peaks.append(peak_rss)
            if scores is None:
                scores = path_sharpes(prediction).tolist()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {str(error).splitlines()[0]}",
                }
            )
        )
        return

    print(
        json.dumps(
            {
                "status": "ok",
                "times_s": times,
                "peak_rss_bytes": peaks,
                "scores": scores,
                "report": None if report is None else asdict(report),
            }
        )
    )


def _format_float(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--native-n-jobs", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cv", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--implementation",
        choices=["native", "accelerated"],
        default="native",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    header = [
        "case",
        "cv",
        "status",
        "backend",
        "fallback_reason",
        "native_median_s",
        "native_mad_s",
        "accelerated_median_s",
        "accelerated_mad_s",
        "speedup",
        "native_peak_rss_mib",
        "accelerated_peak_rss_mib",
        "memory_ratio",
        "n_solves",
        "moment_fits",
        "moment_updates",
        "warm_starts",
        "max_sharpe_difference",
        "precision_at_k",
        "spearman",
    ]
    writer = csv.writer(sys.stdout)
    writer.writerow(header)
    script = Path(__file__).resolve()
    for case_index, case in enumerate(_cases()):
        for cv_index, (cv_name, _) in enumerate(_cv_cases(args.quick)):
            common = [
                "--case",
                str(case_index),
                "--cv",
                str(cv_index),
                "--repeats",
                str(args.repeats),
                "--native-n-jobs",
                str(args.native_n_jobs),
            ]
            if args.quick:
                common.append("--quick")
            native = run_worker(script, [*common, "--implementation", "native"])
            if native["status"] != "ok":
                writer.writerow(
                    [case.name, cv_name, "native-error", "", native["error"]]
                )
                continue
            accelerated = run_worker(
                script, [*common, "--implementation", "accelerated"]
            )
            if accelerated["status"] != "ok":
                writer.writerow(
                    [
                        case.name,
                        cv_name,
                        "accelerated-error",
                        "",
                        accelerated["error"],
                    ]
                )
                continue

            native_time = float(np.median(native["times_s"]))
            accelerated_time = float(np.median(accelerated["times_s"]))
            native_peak = float(np.median(native["peak_rss_bytes"])) / 2**20
            accelerated_peak = float(np.median(accelerated["peak_rss_bytes"])) / 2**20
            reference = np.asarray(native["scores"], dtype=np.float64)
            observed = np.asarray(accelerated["scores"], dtype=np.float64)
            max_difference = float(np.max(np.abs(reference - observed)))
            k = max(1, min(5, reference.size))
            tolerance = max(1e-8, max_difference)
            precision = ranking_precision_at_k(
                reference,
                observed,
                k=k,
                score_tolerance=tolerance,
            )
            spearman = (
                float("nan")
                if reference.size < 2
                else spearman_rank_correlation(
                    reference,
                    observed,
                    score_tolerance=tolerance,
                )
            )
            report = accelerated["report"]
            writer.writerow(
                [
                    case.name,
                    cv_name,
                    "ok",
                    report["backend"],
                    report["fallback_reason"] or "",
                    _format_float(native_time),
                    _format_float(_dispersion(native["times_s"])),
                    _format_float(accelerated_time),
                    _format_float(_dispersion(accelerated["times_s"])),
                    _format_float(native_time / accelerated_time),
                    _format_float(native_peak),
                    _format_float(accelerated_peak),
                    _format_float(accelerated_peak / native_peak),
                    report["n_solves"],
                    report["n_prior_fits"],
                    report["n_prior_updates"],
                    report["n_warm_starts"],
                    _format_float(max_difference),
                    _format_float(precision),
                    _format_float(spearman),
                ]
            )


if __name__ == "__main__":
    main()
