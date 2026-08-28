"""Time native skfolio vs sequential MeanRisk for every objective and risk."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction

from skfolio_accelerate import cross_val_predict


def synthetic_returns(n_observations: int, n_assets: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0005, scale=0.01, size=(n_observations, n_assets))


_NON_ANNUALIZED = [rm for rm in RiskMeasure if not rm.is_annualized]


def _time(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-observations", type=int, default=160)
    parser.add_argument("--n-assets", type=int, default=6)
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/sequential_mean_risk_speedups.csv"),
    )
    args = parser.parse_args()
    X = synthetic_returns(args.n_observations, args.n_assets, seed=21)
    cv = WalkForward(train_size=args.train_size, test_size=args.test_size)
    rows = []
    print(
        f"{'objective':<18} {'risk':<28} {'n_folds':>7} "
        f"{'native_s':>9} {'auto_s':>9} {'seq_s':>9} "
        f"{'auto_x':>7} {'seq_x':>7} {'auto':<16} {'warm':>5} {'rebuild':>7}"
    )
    for objective in ObjectiveFunction:
        for risk in _NON_ANNUALIZED:
            estimator = MeanRisk(
                objective_function=objective,
                risk_measure=risk,
                l2_coef=1e-5,
            )
            try:
                native_s = _time(
                    lambda est=estimator: skfolio_cv_predict(est, X, cv=cv, n_jobs=1),
                    args.repeats,
                )
                pred_auto, report_auto = cross_val_predict(
                    estimator, X, cv=cv, n_jobs=1, return_report=True
                )
                auto_s = _time(
                    lambda est=estimator: cross_val_predict(est, X, cv=cv, n_jobs=1),
                    args.repeats,
                )
                pred_seq, report_seq = cross_val_predict(
                    estimator,
                    X,
                    cv=cv,
                    n_jobs=1,
                    backend="cvxpy-sequential",
                    return_report=True,
                )
                seq_s = _time(
                    lambda est=estimator: cross_val_predict(
                        est,
                        X,
                        cv=cv,
                        n_jobs=1,
                        backend="cvxpy-sequential",
                    ),
                    args.repeats,
                )
            except Exception as error:
                print(
                    f"{objective.name:<18} {risk.name:<28} "
                    f"ERROR {type(error).__name__}: {str(error).splitlines()[0]}"
                )
                rows.append(
                    {
                        "objective": objective.name,
                        "risk": risk.name,
                        "status": (
                            f"{type(error).__name__}: {str(error).splitlines()[0]}"
                        ),
                    }
                )
                continue
            n_folds = int(report_seq.n_solves)
            auto_x = native_s / auto_s if auto_s > 0 else float("nan")
            seq_x = native_s / seq_s if seq_s > 0 else float("nan")
            print(
                f"{objective.name:<18} {risk.name:<28} {n_folds:7d} "
                f"{native_s:9.3f} {auto_s:9.3f} {seq_s:9.3f} "
                f"{auto_x:7.2f} {seq_x:7.2f} {report_auto.backend:<16} "
                f"{report_seq.n_warm_starts:5d} {report_seq.n_rebuilds:7d}"
            )
            rows.append(
                {
                    "objective": objective.name,
                    "risk": risk.name,
                    "n_folds": n_folds,
                    "native_s": native_s,
                    "auto_s": auto_s,
                    "sequential_s": seq_s,
                    "auto_speedup": auto_x,
                    "sequential_speedup": seq_x,
                    "auto_backend": report_auto.backend,
                    "n_warm_starts": report_seq.n_warm_starts,
                    "n_rebuilds": report_seq.n_rebuilds,
                    "is_dpp": report_seq.is_dpp,
                    "status": "ok",
                }
            )
            del pred_auto, pred_seq
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
