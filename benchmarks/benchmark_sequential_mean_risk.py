"""Native skfolio vs ``backend="auto"`` across WalkForward, MRC, and CPCV.

Times every non-annualized ``ObjectiveFunction`` × ``RiskMeasure`` pair, plus a
few extra MeanRisk options (risk limits, linear constraints, fees, L1).
Gini is omitted by default: a year-long training window is a ~20-minute LP
per side and stays sequential at ~1×.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.base import clone

from skfolio_accelerate import cross_val_predict
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.predict import _cap_native_threads, classify_call

_SKIP_DEFAULT = {RiskMeasure.GINI_MEAN_DIFFERENCE}
_FIELDNAMES = [
    "cv",
    "case",
    "objective",
    "risk",
    "extra",
    "n_solves",
    "n_warm_starts",
    "n_rebuilds",
    "native_s",
    "auto_s",
    "auto_speedup",
    "auto_backend",
    "reason",
    "fallback_reason",
    "status",
]


def _gmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.prod(values) ** (1.0 / len(values))


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
    }


def _cases(*, include_gini: bool, include_extras: bool) -> list[dict[str, object]]:
    risks = [
        risk
        for risk in RiskMeasure
        if not risk.is_annualized and (include_gini or risk not in _SKIP_DEFAULT)
    ]
    cases: list[dict[str, object]] = []
    for objective in ObjectiveFunction:
        for risk in risks:
            cases.append(
                {
                    "case": f"{objective.name}/{risk.name}",
                    "objective": objective.name,
                    "risk": risk.name,
                    "extra": "",
                    "estimator": MeanRisk(
                        objective_function=objective,
                        risk_measure=risk,
                        l2_coef=1e-5,
                    ),
                }
            )
    if include_extras:
        cases.extend(
            [
                {
                    "case": "MINIMIZE_RISK/VARIANCE+min_return",
                    "objective": "MINIMIZE_RISK",
                    "risk": "VARIANCE",
                    "extra": "min_return",
                    "estimator": MeanRisk(min_return=1e-5, l2_coef=1e-5),
                },
                {
                    "case": "MINIMIZE_RISK/VARIANCE+linear_constraints",
                    "objective": "MINIMIZE_RISK",
                    "risk": "VARIANCE",
                    "extra": "linear_constraints",
                    "estimator": MeanRisk(
                        linear_constraints=["A0 <= 0.45"], l2_coef=1e-5
                    ),
                },
                {
                    "case": "MINIMIZE_RISK/VARIANCE+management_fees",
                    "objective": "MINIMIZE_RISK",
                    "risk": "VARIANCE",
                    "extra": "management_fees",
                    "estimator": MeanRisk(management_fees=1e-4, l2_coef=1e-5),
                },
                {
                    "case": "MINIMIZE_RISK/VARIANCE+l1_coef",
                    "objective": "MINIMIZE_RISK",
                    "risk": "VARIANCE",
                    "extra": "l1_coef",
                    "estimator": MeanRisk(l1_coef=1e-3, l2_coef=1e-5),
                },
                {
                    "case": "MINIMIZE_RISK/CVAR+min_return",
                    "objective": "MINIMIZE_RISK",
                    "risk": "CVAR",
                    "extra": "min_return",
                    "estimator": MeanRisk(
                        risk_measure=RiskMeasure.CVAR,
                        min_return=1e-6,
                        l2_coef=1e-5,
                    ),
                },
            ]
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-observations", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--include-gini", action="store_true")
    parser.add_argument("--skip-extras", action="store_true")
    parser.add_argument(
        "--cv",
        action="append",
        default=[],
        help="walk-forward, multiple-randomized, purged-cpcv (repeatable; default all)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    _cap_native_threads()
    n_observations = args.n_observations or (120 if args.quick else 20 * 252)
    n_assets = args.n_assets or (6 if args.quick else 20)
    X = factor_returns(n_observations, n_assets, seed=42)
    X.columns = [f"A{i}" for i in range(n_assets)]
    factories = _cv_factories(quick=args.quick)
    selected = args.cv or list(factories)
    unknown = [name for name in selected if name not in factories]
    if unknown:
        raise SystemExit(f"unknown --cv values: {unknown}")
    args.csv = args.csv or Path(
        "benchmarks/sequential_mean_risk_speedups_quick.csv"
        if args.quick
        else "benchmarks/sequential_mean_risk_speedups.csv"
    )
    cases = _cases(include_gini=args.include_gini, include_extras=not args.skip_extras)
    rows: list[dict[str, object]] = []
    print(
        f"data: {X.shape[0]} days × {X.shape[1]} assets; cvs={selected}",
        flush=True,
    )
    for cv_name in selected:
        cv = factories[cv_name]()
        print(f"== {cv_name}  n_splits≈{cv.get_n_splits(X)} ==", flush=True)
        for spec in cases:
            if spec["extra"] and cv_name == "multiple-randomized":
                # Asset subsets drop named columns; min_return can be infeasible.
                continue
            estimator = spec["estimator"]
            expected = classify_call(estimator, cv=cv).auto_backend(estimator)
            native_s = auto_s = float("nan")
            report = None
            try:
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    skfolio_cv_predict(clone(estimator), X, cv=cv, n_jobs=1)
                    native_s = time.perf_counter() - started
                    started = time.perf_counter()
                    _, report = cross_val_predict(
                        clone(estimator), X, cv=cv, n_jobs=1, return_report=True
                    )
                    auto_s = time.perf_counter() - started
            except Exception as error:
                status = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                print(
                    f"{cv_name:<22} {spec['case']:<44} ERROR {status}",
                    flush=True,
                )
                rows.append(
                    {
                        "cv": cv_name,
                        "case": spec["case"],
                        "objective": spec["objective"],
                        "risk": spec["risk"],
                        "extra": spec["extra"],
                        "auto_backend": expected,
                        "status": status,
                    }
                )
                continue
            auto_x = native_s / auto_s if auto_s > 0 else float("nan")
            print(
                f"{cv_name:<22} {spec['case']:<44} {report.n_solves:4d} "
                f"{native_s:8.3f} {auto_s:8.3f} {auto_x:6.2f} {report.backend}",
                flush=True,
            )
            rows.append(
                {
                    "cv": cv_name,
                    "case": spec["case"],
                    "objective": spec["objective"],
                    "risk": spec["risk"],
                    "extra": spec["extra"],
                    "n_solves": report.n_solves,
                    "n_warm_starts": report.n_warm_starts,
                    "n_rebuilds": report.n_rebuilds,
                    "native_s": native_s,
                    "auto_s": auto_s,
                    "auto_speedup": auto_x,
                    "auto_backend": report.backend,
                    "reason": report.reason,
                    "fallback_reason": report.fallback_reason,
                    "status": "ok",
                }
            )
            compact_retry = (
                expected in {"osqp", "clarabel"}
                and report.backend in {"fit-assemble", "sklearn"}
                and report.fallback_reason
            )
            if report.backend != expected and not compact_retry:
                raise SystemExit(
                    f"auto {report.backend!r} != policy {expected!r} "
                    f"for {cv_name}/{spec['case']}"
                )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        shutil.copy(args.csv, artifacts / args.csv.name)
    ok = [row for row in rows if row.get("status") == "ok"]
    print(f"Wrote {args.csv}  ({len(ok)} ok / {len(rows)})", flush=True)
    for cv_name in selected:
        speedups = [float(row["auto_speedup"]) for row in ok if row["cv"] == cv_name]
        print(f"  {cv_name} gmean: {_gmean(speedups):.2f}×  n={len(speedups)}")
    backends = sorted({str(row["auto_backend"]) for row in ok})
    for backend in backends:
        speedups = [
            float(row["auto_speedup"]) for row in ok if row["auto_backend"] == backend
        ]
        print(f"  {backend} gmean: {_gmean(speedups):.2f}×  n={len(speedups)}")


if __name__ == "__main__":
    main()
