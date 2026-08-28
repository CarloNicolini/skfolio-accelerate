"""Native skfolio vs compact HiGHS on boxed MeanRisk LPs across CV families.

Measures wall time and mean path Sharpe for MAD, FLPM, CVaR, and worst
realization (``l2_coef=0``) on WalkForward, MultipleRandomizedCV, and
CombinatorialPurgedCV.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
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
from skfolio.optimization import MeanRisk
from sklearn.base import clone

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.predict import _cap_native_threads

_LP_RISKS = (
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.CVAR,
    RiskMeasure.WORST_REALIZATION,
)

_FIELDNAMES = [
    "cv",
    "risk",
    "n_solves",
    "n_warm_starts",
    "backend",
    "native_s",
    "auto_s",
    "speedup",
    "native_mean_sharpe",
    "auto_mean_sharpe",
    "mean_sharpe_diff",
    "max_abs_sharpe_diff",
    "status",
]


def _gmean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    if not finite:
        return float("nan")
    return math.prod(finite) ** (1.0 / len(finite))


def _cv_factories(*, quick: bool) -> dict[str, Callable[[], object]]:
    train_size = 40 if quick else 252
    test_size = 20 if quick else 21
    return {
        "walk-forward": lambda: WalkForward(train_size=train_size, test_size=test_size),
        "multiple-randomized": lambda: MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=train_size, test_size=test_size),
            n_subsamples=3 if quick else 12,
            asset_subset_size=4 if quick else 12,
            window_size=100 if quick else 756,
            random_state=43,
        ),
        "purged-cpcv": lambda: CombinatorialPurgedCV(
            n_folds=4 if quick else 6,
            n_test_folds=2,
            purged_size=1,
            embargo_size=1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-observations", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    _cap_native_threads()
    n_observations = args.n_observations or (120 if args.quick else 5 * 252)
    n_assets = args.n_assets or (6 if args.quick else 20)
    X = factor_returns(n_observations, n_assets, seed=42)
    X.columns = [f"A{i}" for i in range(n_assets)]
    factories = _cv_factories(quick=args.quick)
    args.csv = args.csv or Path(
        "benchmarks/lp_cv_speedups_quick.csv"
        if args.quick
        else "benchmarks/lp_cv_speedups.csv"
    )
    rows: list[dict[str, object]] = []
    print(
        f"data: {X.shape[0]} days × {X.shape[1]} assets; l2_coef=0; "
        f"cvs={list(factories)}",
        flush=True,
    )
    for cv_name, factory in factories.items():
        n_splits = factory().get_n_splits(X)
        print(f"== {cv_name}  n_splits={n_splits} ==", flush=True)
        for risk in _LP_RISKS:
            estimator = MeanRisk(risk_measure=risk)
            native_s = auto_s = float("nan")
            report = None
            native_sharpes = auto_sharpes = None
            try:
                started = time.perf_counter()
                native_pred = skfolio_cv_predict(
                    clone(estimator), X, cv=factory(), n_jobs=1
                )
                native_s = time.perf_counter() - started
                native_sharpes = path_sharpes(native_pred)
                started = time.perf_counter()
                auto_pred, report = cross_val_predict(
                    clone(estimator),
                    X,
                    cv=factory(),
                    n_jobs=1,
                    return_report=True,
                )
                auto_s = time.perf_counter() - started
                auto_sharpes = path_sharpes(auto_pred)
            except Exception as error:
                status = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                print(f"{cv_name:<22} {risk.name:<32} ERROR {status}", flush=True)
                rows.append(
                    {
                        "cv": cv_name,
                        "risk": risk.name,
                        "status": status,
                    }
                )
                continue
            if native_sharpes.shape != auto_sharpes.shape:
                status = f"sharpe shape {native_sharpes.shape} != {auto_sharpes.shape}"
                print(f"{cv_name:<22} {risk.name:<32} ERROR {status}", flush=True)
                rows.append(
                    {
                        "cv": cv_name,
                        "risk": risk.name,
                        "status": status,
                    }
                )
                continue
            speedup = native_s / auto_s if auto_s > 0 else float("nan")
            mean_n = float(np.mean(native_sharpes))
            mean_a = float(np.mean(auto_sharpes))
            max_abs = float(np.max(np.abs(native_sharpes - auto_sharpes)))
            print(
                f"{cv_name:<22} {risk.name:<32} {report.n_solves:4d} "
                f"{native_s:8.3f}s {auto_s:8.3f}s {speedup:6.2f}×  "
                f"{report.backend:<8}  sharpe {mean_n:.6f} vs {mean_a:.6f}  "
                f"Δmean={mean_a - mean_n:+.2e}  max|Δ|={max_abs:.2e}",
                flush=True,
            )
            rows.append(
                {
                    "cv": cv_name,
                    "risk": risk.name,
                    "n_solves": report.n_solves,
                    "n_warm_starts": report.n_warm_starts,
                    "backend": report.backend,
                    "native_s": native_s,
                    "auto_s": auto_s,
                    "speedup": speedup,
                    "native_mean_sharpe": mean_n,
                    "auto_mean_sharpe": mean_a,
                    "mean_sharpe_diff": mean_a - mean_n,
                    "max_abs_sharpe_diff": max_abs,
                    "status": "ok",
                }
            )
            if report.backend != "highs":
                native_ok = (
                    cv_name == "purged-cpcv"
                    and risk
                    in {
                        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
                        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
                    }
                    and report.backend == "sklearn"
                )
                if not native_ok:
                    raise SystemExit(
                        f"expected highs, got {report.backend!r} "
                        f"for {cv_name}/{risk.name}"
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
    print(
        f"{'cv':<22} {'gmean speedup':>14} {'max |Δ Sharpe|':>16} {'n':>4}",
        flush=True,
    )
    for cv_name in factories:
        subset = [row for row in ok if row["cv"] == cv_name]
        speedups = [float(row["speedup"]) for row in subset]
        diffs = [float(row["max_abs_sharpe_diff"]) for row in subset]
        print(
            f"{cv_name:<22} {_gmean(speedups):14.2f}× "
            f"{(max(diffs) if diffs else float('nan')):16.2e} {len(subset):4d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
