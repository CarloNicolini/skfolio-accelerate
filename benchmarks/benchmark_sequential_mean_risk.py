"""Native skfolio vs ``backend="auto"`` on a 20-year WalkForward.

The user-facing API does not pick an engine. This script times that default
policy against native ``cross_val_predict`` for every
``ObjectiveFunction`` × non-annualized ``RiskMeasure``. Short synthetic
windows hide the policy behind per-fold copy overhead; 20 years of daily
returns with a 1-year / 1-month WalkForward does not.
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

from skfolio_accelerate import classify_call, cross_val_predict
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.predict import _cap_native_threads

_LATE_RISKS = (
    RiskMeasure.CVAR,
    RiskMeasure.CDAR,
    RiskMeasure.EVAR,
    RiskMeasure.EDAR,
    RiskMeasure.ULCER_INDEX,
    RiskMeasure.GINI_MEAN_DIFFERENCE,
)
_FIELDNAMES = [
    "objective",
    "risk",
    "n_folds",
    "native_s",
    "auto_s",
    "auto_speedup",
    "auto_backend",
    "expected_backend",
    "reason",
    "n_warm_starts",
    "n_rebuilds",
    "is_dpp",
    "status",
]


def _ordered_risks(selected: list[str] | None) -> list[RiskMeasure]:
    risks = [rm for rm in RiskMeasure if not rm.is_annualized]
    if selected:
        wanted = {name.upper() for name in selected}
        risks = [rm for rm in risks if rm.name in wanted]
        missing = wanted - {rm.name for rm in risks}
        if missing:
            raise SystemExit(f"unknown risk measure(s): {sorted(missing)}")
    late = set(_LATE_RISKS)
    early = [rm for rm in risks if rm not in late]
    tail = [rm for rm in risks if rm in late]
    tail.sort(key=lambda rm: rm is RiskMeasure.GINI_MEAN_DIFFERENCE)
    return early + tail


def _ordered_objectives(selected: list[str] | None) -> list[ObjectiveFunction]:
    objectives = list(ObjectiveFunction)
    if not selected:
        return objectives
    wanted = {name.upper() for name in selected}
    chosen = [obj for obj in objectives if obj.name in wanted]
    missing = wanted - {obj.name for obj in chosen}
    if missing:
        raise SystemExit(f"unknown objective(s): {sorted(missing)}")
    return chosen


def _median(samples: list[float]) -> float:
    return float(sorted(samples)[len(samples) // 2])


def _gmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.prod(values) ** (1.0 / len(values))


def _time_repeats(fn, repeats: int) -> float:
    return _median([_once(fn) for _ in range(repeats)])


def _once(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-observations", type=int, default=20 * 252)
    parser.add_argument("--n-assets", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--risks", nargs="*", default=None)
    parser.add_argument("--objectives", nargs="*", default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/sequential_mean_risk_speedups.csv"),
    )
    args = parser.parse_args()
    _cap_native_threads()
    X = factor_returns(args.n_observations, args.n_assets, seed=21)
    cv = WalkForward(train_size=args.train_size, test_size=args.test_size)
    objectives = _ordered_objectives(args.objectives)
    risks = _ordered_risks(args.risks)
    rows: list[dict[str, object]] = []
    _log(
        f"data: {X.shape[0]} days × {X.shape[1]} assets; "
        f"WalkForward(train={args.train_size}, test={args.test_size}); "
        f"{len(objectives)} objectives × {len(risks)} risks; "
        f"repeats={args.repeats}"
    )
    _log(
        f"{'objective':<18} {'risk':<28} {'n_folds':>7} "
        f"{'native_s':>9} {'auto_s':>9} {'auto_x':>7} {'auto':<16} "
        f"{'warm':>5} {'rebuild':>7}"
    )
    for risk in risks:
        for objective in objectives:
            estimator = MeanRisk(
                objective_function=objective,
                risk_measure=risk,
                l2_coef=1e-5,
            )
            expected = classify_call(estimator, cv=cv).auto_backend(estimator)
            try:
                native_s = _time_repeats(
                    lambda est=estimator: skfolio_cv_predict(est, X, cv=cv, n_jobs=1),
                    args.repeats,
                )
            except Exception as error:
                status = f"native {type(error).__name__}: {str(error).splitlines()[0]}"
                _log(f"{objective.name:<18} {risk.name:<28} ERROR {status}")
                rows.append(
                    {
                        "objective": objective.name,
                        "risk": risk.name,
                        "expected_backend": expected,
                        "status": status,
                    }
                )
                continue
            try:
                auto_samples: list[float] = []
                pred = None
                report = None
                for index in range(args.repeats):
                    started = time.perf_counter()
                    if index == 0:
                        pred, report = cross_val_predict(
                            estimator, X, cv=cv, n_jobs=1, return_report=True
                        )
                    else:
                        cross_val_predict(estimator, X, cv=cv, n_jobs=1)
                    auto_samples.append(time.perf_counter() - started)
                auto_s = _median(auto_samples)
            except Exception as error:
                status = f"auto {type(error).__name__}: {str(error).splitlines()[0]}"
                _log(f"{objective.name:<18} {risk.name:<28} ERROR {status}")
                rows.append(
                    {
                        "objective": objective.name,
                        "risk": risk.name,
                        "native_s": native_s,
                        "expected_backend": expected,
                        "status": status,
                    }
                )
                continue
            assert report is not None
            n_folds = int(report.n_solves)
            auto_x = native_s / auto_s if auto_s > 0 else float("nan")
            _log(
                f"{objective.name:<18} {risk.name:<28} {n_folds:7d} "
                f"{native_s:9.3f} {auto_s:9.3f} {auto_x:7.2f} "
                f"{report.backend:<16} {report.n_warm_starts:5d} "
                f"{report.n_rebuilds:7d}"
            )
            rows.append(
                {
                    "objective": objective.name,
                    "risk": risk.name,
                    "n_folds": n_folds,
                    "native_s": native_s,
                    "auto_s": auto_s,
                    "auto_speedup": auto_x,
                    "auto_backend": report.backend,
                    "expected_backend": expected,
                    "reason": report.reason,
                    "n_warm_starts": report.n_warm_starts,
                    "n_rebuilds": report.n_rebuilds,
                    "is_dpp": report.is_dpp,
                    "status": "ok",
                }
            )
            if report.backend != expected:
                raise SystemExit(
                    f"auto backend {report.backend!r} != policy {expected!r} "
                    f"for {objective.name}/{risk.name}"
                )
            del pred
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        shutil.copy(args.csv, artifacts / args.csv.name)
    ok = [row for row in rows if row.get("status") == "ok"]
    _log(f"\nWrote {args.csv}  ({len(ok)} ok / {len(rows)} combos)")
    if ok:
        by_backend: dict[str, list[float]] = {}
        for row in ok:
            backend = str(row["auto_backend"])
            by_backend.setdefault(backend, []).append(float(row["auto_speedup"]))
        all_speedups = [float(row["auto_speedup"]) for row in ok]
        _log(f"geometric mean speedup  all: {_gmean(all_speedups):.2f}×")
        for backend, values in sorted(by_backend.items()):
            _log(f"  {backend}: {_gmean(values):.2f}×  n={len(values)}")


if __name__ == "__main__":
    main()
