"""Native serial vs ``n_jobs=-1`` vs serial ``backend="auto"``.

MRC paths and CPCV combinations are independent, so native skfolio can use
joblib. The amortized engines stay serial (warm starts / Parameter reuse) and
already pin solver threads to 1. This script reports whether that serial
engine still beats a multi-core native run, and records the two practical
tips: cap solver threads when using ``n_jobs=-1``, and relax Clarabel
tolerances on exploratory hyperparameter searches.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from skfolio_accelerate import cross_val_predict, grid_search
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.predict import _cap_native_threads, classify_call

_FIELDNAMES = [
    "cv",
    "case",
    "n_solves",
    "n_cpus",
    "native_serial_s",
    "native_parallel_s",
    "auto_s",
    "auto_backend",
    "speedup_vs_serial",
    "speedup_vs_parallel",
    "parallel_vs_serial",
    "status",
]


def _cv_factories(*, quick: bool) -> dict[str, Callable[[], object]]:
    train_size = 40 if quick else 252
    test_size = 20 if quick else 21
    return {
        "walk-forward": lambda: WalkForward(train_size=train_size, test_size=test_size),
        "multiple-randomized": lambda: MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=train_size, test_size=test_size),
            n_subsamples=3 if quick else 20,
            asset_subset_size=4 if quick else 12,
            window_size=100 if quick else 756,
            random_state=43,
        ),
        "purged-cpcv": lambda: CombinatorialPurgedCV(
            n_folds=4,
            n_test_folds=2,
            purged_size=1,
            embargo_size=1,
        ),
        "purged-cpcv-wide": lambda: CombinatorialPurgedCV(
            n_folds=5 if quick else 10,
            n_test_folds=2,
            purged_size=1,
            embargo_size=1,
        ),
    }


def _cases() -> list[dict[str, object]]:
    return [
        {
            "case": "MINIMIZE_RISK/VARIANCE",
            "estimator": MeanRisk(l2_coef=1e-5),
        },
        {
            "case": "MINIMIZE_RISK/CVAR",
            "estimator": MeanRisk(risk_measure=RiskMeasure.CVAR, l2_coef=1e-5),
        },
        {
            "case": "MINIMIZE_RISK/STANDARD_DEVIATION",
            "estimator": MeanRisk(
                risk_measure=RiskMeasure.STANDARD_DEVIATION, l2_coef=1e-5
            ),
        },
        {
            "case": "MAXIMIZE_RATIO/VARIANCE",
            "estimator": MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                l2_coef=1e-5,
            ),
        },
    ]


def _warmup_parallel(X) -> None:
    n_rows = min(len(X), 60)
    n_cols = min(X.shape[1], 4)
    tiny = X.iloc[:n_rows, :n_cols]
    cv = WalkForward(train_size=20, test_size=10)
    skfolio_cv_predict(MeanRisk(l2_coef=1e-5), tiny, cv=cv, n_jobs=-1)


def _time_native(estimator, X, cv, n_jobs: int) -> float:
    started = time.perf_counter()
    skfolio_cv_predict(clone(estimator), X, cv=cv, n_jobs=n_jobs)
    return time.perf_counter() - started


def _time_native_subprocess(
    *,
    quick: bool,
    n_jobs: int,
    thread_cap: int,
) -> float:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(thread_cap)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "mrc-variance",
        "--n-jobs",
        str(n_jobs),
    ]
    if quick:
        cmd.append("--quick")
    completed = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        if line.startswith("WORKER_S="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(
        f"worker produced no timing\n{completed.stdout}\n{completed.stderr}"
    )


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        shutil.copy(path, artifacts / path.name)


def _run_cv_matrix(*, quick: bool, selected: list[str]) -> list[dict[str, object]]:
    n_observations = 120 if quick else 20 * 252
    n_assets = 6 if quick else 20
    n_cpus = os.cpu_count() or 1
    X = factor_returns(n_observations, n_assets, seed=42)
    X.columns = [f"A{i}" for i in range(n_assets)]
    factories = _cv_factories(quick=quick)
    rows: list[dict[str, object]] = []
    print(f"data: {X.shape[0]} days × {X.shape[1]} assets; cpus={n_cpus}", flush=True)
    _warmup_parallel(X)
    for cv_name in selected:
        cv = factories[cv_name]()
        print(f"== {cv_name}  n_splits≈{cv.get_n_splits(X)} ==", flush=True)
        for spec in _cases():
            estimator = spec["estimator"]
            expected = classify_call(estimator, cv=cv).auto_backend(estimator)
            try:
                serial_s = _time_native(estimator, X, cv, 1)
                parallel_s = _time_native(estimator, X, cv, -1)
                started = time.perf_counter()
                _, report = cross_val_predict(
                    clone(estimator),
                    X,
                    cv=cv,
                    n_jobs=1,
                    return_report=True,
                )
                auto_s = time.perf_counter() - started
            except Exception as error:
                status = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                print(f"{cv_name:<22} {spec['case']:<36} ERROR {status}", flush=True)
                rows.append(
                    {
                        "cv": cv_name,
                        "case": spec["case"],
                        "n_cpus": n_cpus,
                        "auto_backend": expected,
                        "status": status,
                    }
                )
                continue
            vs_serial = serial_s / auto_s if auto_s > 0 else float("nan")
            vs_parallel = parallel_s / auto_s if auto_s > 0 else float("nan")
            par_vs_serial = serial_s / parallel_s if parallel_s > 0 else float("nan")
            print(
                f"{cv_name:<22} {spec['case']:<36} {report.n_solves:4d} "
                f"{serial_s:7.3f} {parallel_s:7.3f} {auto_s:7.3f}  "
                f"{vs_serial:5.2f}×ser {vs_parallel:5.2f}×par {report.backend}",
                flush=True,
            )
            rows.append(
                {
                    "cv": cv_name,
                    "case": spec["case"],
                    "n_solves": report.n_solves,
                    "n_cpus": n_cpus,
                    "native_serial_s": serial_s,
                    "native_parallel_s": parallel_s,
                    "auto_s": auto_s,
                    "auto_backend": report.backend,
                    "speedup_vs_serial": vs_serial,
                    "speedup_vs_parallel": vs_parallel,
                    "parallel_vs_serial": par_vs_serial,
                    "status": "ok",
                }
            )
            if report.backend != expected:
                raise SystemExit(
                    f"auto {report.backend!r} != policy {expected!r} "
                    f"for {cv_name}/{spec['case']}"
                )
    return rows


def _run_worker(*, quick: bool, n_jobs: int) -> None:
    n_observations = 120 if quick else 20 * 252
    n_assets = 6 if quick else 20
    train = 40 if quick else 252
    test = 20 if quick else 21
    X = factor_returns(n_observations, n_assets, seed=42)
    X.columns = [f"A{i}" for i in range(n_assets)]
    _warmup_parallel(X)
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(train_size=train, test_size=test),
        n_subsamples=3 if quick else 20,
        asset_subset_size=4 if quick else 12,
        window_size=100 if quick else 756,
        random_state=43,
    )
    elapsed = _time_native(MeanRisk(l2_coef=1e-5), X, cv, n_jobs)
    print(f"WORKER_S={elapsed:.6f}", flush=True)


def _run_tips(*, quick: bool) -> list[dict[str, object]]:
    n_observations = 120 if quick else 20 * 252
    n_assets = 6 if quick else 20
    n_cpus = os.cpu_count() or 1
    X = factor_returns(n_observations, n_assets, seed=42)
    X.columns = [f"A{i}" for i in range(n_assets)]
    train = 40 if quick else 252
    test = 20 if quick else 21
    rows: list[dict[str, object]] = []

    def record(kind: str, label: str, elapsed: float, **extra: object) -> None:
        row = {
            "kind": kind,
            "label": label,
            "n_cpus": n_cpus,
            "time_s": elapsed,
            **extra,
        }
        rows.append(row)
        print(f"{kind:<12} {label:<44} {elapsed:8.3f}s", flush=True)

    record(
        "threads",
        "MRC variance native n_jobs=-1 threads=1",
        _time_native_subprocess(quick=quick, n_jobs=-1, thread_cap=1),
        n_jobs=-1,
        thread_cap=1,
    )
    record(
        "threads",
        f"MRC variance native n_jobs=-1 threads={n_cpus}",
        _time_native_subprocess(quick=quick, n_jobs=-1, thread_cap=n_cpus),
        n_jobs=-1,
        thread_cap=n_cpus,
    )

    window = X.iloc[: (100 if quick else 756)]
    n_rep = 4 if quick else 8
    default = MeanRisk(risk_measure=RiskMeasure.CVAR, l2_coef=1e-5)
    relaxed = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        l2_coef=1e-5,
        solver_params={"tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4},
    )
    clone(default).fit(window)
    clone(relaxed).fit(window)
    started = time.perf_counter()
    weights_default = [clone(default).fit(window).weights_ for _ in range(n_rep)]
    default_s = (time.perf_counter() - started) / n_rep
    started = time.perf_counter()
    weights_relaxed = [clone(relaxed).fit(window).weights_ for _ in range(n_rep)]
    relaxed_s = (time.perf_counter() - started) / n_rep
    delta = float(
        np.max(np.abs(np.asarray(weights_default) - np.asarray(weights_relaxed)))
    )
    record("tolerance", "CVaR MeanRisk.fit default Clarabel", default_s, n_rep=n_rep)
    record(
        "tolerance",
        "CVaR MeanRisk.fit tol_gap=1e-4",
        relaxed_s,
        n_rep=n_rep,
        max_abs_weight_delta=delta,
        speedup=default_s / relaxed_s if relaxed_s else float("nan"),
    )

    cv = WalkForward(train_size=train, test_size=test)
    param_grid = {"l2_coef": np.logspace(-5, -1, 4 if quick else 8)}
    started = time.perf_counter()
    for params in ParameterGrid(param_grid):
        skfolio_cv_predict(MeanRisk(**params), X, cv=cv, n_jobs=-1)
    native_grid_s = time.perf_counter() - started
    started = time.perf_counter()
    grid_search(MeanRisk(), X, param_grid, cv=cv)
    compact_grid_s = time.perf_counter() - started
    record(
        "grid",
        "WalkForward ParameterGrid n_jobs=-1",
        native_grid_s,
        n_candidates=len(list(ParameterGrid(param_grid))),
    )
    record(
        "grid",
        "WalkForward compact grid_search",
        compact_grid_s,
        n_candidates=len(list(ParameterGrid(param_grid))),
        speedup=native_grid_s / compact_grid_s if compact_grid_s else float("nan"),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--cv",
        action="append",
        default=[],
        help="walk-forward, multiple-randomized, purged-cpcv, purged-cpcv-wide",
    )
    parser.add_argument("--skip-tips", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--tips-csv", type=Path, default=None)
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--n-jobs", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == "mrc-variance":
        _run_worker(quick=args.quick, n_jobs=args.n_jobs)
        return
    _cap_native_threads()
    factories = _cv_factories(quick=args.quick)
    selected = args.cv or list(factories)
    unknown = [name for name in selected if name not in factories]
    if unknown:
        raise SystemExit(f"unknown --cv values: {unknown}")
    args.csv = args.csv or Path(
        "benchmarks/parallel_cv_speedups_quick.csv"
        if args.quick
        else "benchmarks/parallel_cv_speedups.csv"
    )
    args.tips_csv = args.tips_csv or Path(
        "benchmarks/solver_tips_quick.csv"
        if args.quick
        else "benchmarks/solver_tips.csv"
    )
    rows = _run_cv_matrix(quick=args.quick, selected=selected)
    _write(args.csv, _FIELDNAMES, rows)
    print(f"Wrote {args.csv}  ({sum(r.get('status') == 'ok' for r in rows)} ok)")
    if not args.skip_tips:
        tips = _run_tips(quick=args.quick)
        _write(
            args.tips_csv,
            [
                "kind",
                "label",
                "n_cpus",
                "time_s",
                "n_jobs",
                "thread_cap",
                "n_rep",
                "n_candidates",
                "max_abs_weight_delta",
                "speedup",
            ],
            tips,
        )
        print(f"Wrote {args.tips_csv}")


if __name__ == "__main__":
    main()
