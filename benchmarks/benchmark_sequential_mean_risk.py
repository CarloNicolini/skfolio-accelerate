"""Native skfolio vs ``backend="auto"`` on a 20-year WalkForward.

Times the default policy (OSQP / Clarabel / sequential CVXPY) against native
``cross_val_predict``. Gini is omitted by default: a 228-fold year-long
training window is a 20-minute LP per side and does not change the auto
decision (it stays sequential at ~1×).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction

from skfolio_accelerate import cross_val_predict
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.predict import _cap_native_threads, classify_call

_SKIP_DEFAULT = {RiskMeasure.GINI_MEAN_DIFFERENCE}
_FIELDNAMES = [
    "objective",
    "risk",
    "n_folds",
    "native_s",
    "auto_s",
    "auto_speedup",
    "auto_backend",
    "reason",
    "status",
]


def _gmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.prod(values) ** (1.0 / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-observations", type=int, default=20 * 252)
    parser.add_argument("--n-assets", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--include-gini", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/sequential_mean_risk_speedups.csv"),
    )
    args = parser.parse_args()
    _cap_native_threads()
    X = factor_returns(args.n_observations, args.n_assets, seed=21)
    cv = WalkForward(train_size=args.train_size, test_size=args.test_size)
    risks = [
        rm
        for rm in RiskMeasure
        if not rm.is_annualized and (args.include_gini or rm not in _SKIP_DEFAULT)
    ]
    rows: list[dict[str, object]] = []
    print(
        f"data: {X.shape[0]} days × {X.shape[1]} assets; "
        f"WalkForward(train={args.train_size}, test={args.test_size})",
        flush=True,
    )
    for objective in ObjectiveFunction:
        for risk in risks:
            estimator = MeanRisk(
                objective_function=objective, risk_measure=risk, l2_coef=1e-5
            )
            expected = classify_call(estimator, cv=cv).auto_backend(estimator)
            try:
                started = time.perf_counter()
                skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
                native_s = time.perf_counter() - started
                started = time.perf_counter()
                _, report = cross_val_predict(
                    estimator, X, cv=cv, n_jobs=1, return_report=True
                )
                auto_s = time.perf_counter() - started
            except Exception as error:
                status = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                print(f"{objective.name} {risk.name} ERROR {status}", flush=True)
                rows.append(
                    {
                        "objective": objective.name,
                        "risk": risk.name,
                        "auto_backend": expected,
                        "status": status,
                    }
                )
                continue
            auto_x = native_s / auto_s if auto_s > 0 else float("nan")
            print(
                f"{objective.name:<18} {risk.name:<28} {report.n_solves:4d} "
                f"{native_s:8.3f} {auto_s:8.3f} {auto_x:6.2f} {report.backend}",
                flush=True,
            )
            rows.append(
                {
                    "objective": objective.name,
                    "risk": risk.name,
                    "n_folds": report.n_solves,
                    "native_s": native_s,
                    "auto_s": auto_s,
                    "auto_speedup": auto_x,
                    "auto_backend": report.backend,
                    "reason": report.reason,
                    "status": "ok",
                }
            )
            if report.backend != expected:
                raise SystemExit(
                    f"auto {report.backend!r} != policy {expected!r} "
                    f"for {objective.name}/{risk.name}"
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
    if ok:
        speedups = [float(row["auto_speedup"]) for row in ok]
        print(f"geometric mean speedup: {_gmean(speedups):.2f}×")


if __name__ == "__main__":
    main()
