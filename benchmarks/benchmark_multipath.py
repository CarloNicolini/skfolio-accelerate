"""Flagship multi-path backtest vs skfolio cross_val_predict.

Run with::

    PYTHONPATH=src python benchmarks/benchmark_multipath.py
    PYTHONPATH=src python benchmarks/benchmark_multipath.py --quick
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior
from sklearn.base import clone

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.flagship import FLAGSHIP_MRC, SMOKE_CPCV, make_cpcv, make_mrc


def _cap_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _profile_one_fit(estimator, X_window, n_rep: int = 8) -> dict[str, float]:
    clone(estimator).fit(X_window)  # warmup
    EmpiricalPrior().fit(X_window)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        clone(estimator).fit(X_window)
    fit_s = (time.perf_counter() - t0) / n_rep
    t0 = time.perf_counter()
    for _ in range(n_rep):
        EmpiricalPrior().fit(X_window)
    prior_s = (time.perf_counter() - t0) / n_rep
    return {
        "prior_s": prior_s,
        "fit_s": fit_s,
        "solve_proxy_s": max(fit_s - prior_s, 0.0),
    }


def _print_report(
    title: str,
    baseline_s: float,
    report,
    pred,
    ref=None,
    *,
    baseline_fit_s: float = 0.0,
    baseline_prior_s: float = 0.0,
) -> None:
    speedup = baseline_s / report.wall_s if report.wall_s else float("nan")
    report.baseline_s = baseline_s
    report.speedup = speedup
    # Compact-path phase share of *this* run (should be most of compact wall).
    compact_core = report.moments_s + report.solve_s + report.eval_s
    compact_frac = compact_core / report.wall_s if report.wall_s else float("nan")
    # Baseline Amdahl: prior+QP inside MeanRisk.fit, estimated from a sample fit.
    if baseline_fit_s > 0:
        prior_share = baseline_prior_s / baseline_fit_s
        solve_share = max(0.0, 1.0 - prior_share)
        accelerated_frac = prior_share + solve_share
    else:
        accelerated_frac = float("nan")
    print(title)
    print(
        f"  baseline {baseline_s:.4f}s  compact {report.wall_s:.4f}s  "
        f"speedup {speedup:.2f}×"
    )
    print(
        f"  compact phases  moments {report.moments_s:.4f}s  "
        f"solve {report.solve_s:.4f}s  "
        f"eval {report.eval_s:.4f}s  ({100 * compact_frac:.1f}% of compact wall)"
    )
    print(
        f"  n_solves {report.n_solves}  n_prior_fits {report.n_prior_fits}  "
        f"n_prior_updates {report.n_prior_updates}  "
        f"n_warm_starts {report.n_warm_starts}"
    )
    if baseline_fit_s > 0:
        print(
            f"  baseline MeanRisk.fit sample {baseline_fit_s * 1000:.1f}ms  "
            f"(prior {baseline_prior_s * 1000:.1f}ms)  "
            f"accelerated kernels ≈ {100 * accelerated_frac:.0f}% of each fit"
        )
    if ref is not None:
        d = np.max(np.abs(path_sharpes(pred) - path_sharpes(ref)))
        print(f"  max |Δ path Sharpe| {d:.3e}")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    _cap_threads()
    n_jobs = os.cpu_count() or 1
    print("cpu_count", n_jobs)

    spec = dict(FLAGSHIP_MRC)
    if args.quick:
        spec.update(
            {
                "n_obs": 756,
                "n_assets": 30,
                "n_subsamples": 12,
                "asset_subset_size": 12,
                "window_size": 504,
            }
        )

    X, cv = make_mrc(spec)
    n_splits = cv.get_n_splits(X)
    print(
        f"MRC n_obs={spec['n_obs']} n_assets={spec['n_assets']} "
        f"subsamples={spec['n_subsamples']} splits={n_splits}"
    )

    estimator = MeanRisk()
    window = X.iloc[: spec["train_size"], : spec["asset_subset_size"]]
    sample = _profile_one_fit(estimator, window)
    print(
        f"single MeanRisk.fit {sample['fit_s'] * 1000:.1f}ms  "
        f"(prior {sample['prior_s'] * 1000:.1f}ms  "
        f"rest {sample['solve_proxy_s'] * 1000:.1f}ms)"
    )

    baseline_s = 0.0
    ref = None
    if not args.skip_baseline:
        t0 = time.perf_counter()
        ref = skfolio_cv_predict(MeanRisk(), X, cv=cv, n_jobs=-1)
        baseline_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    # wall already in report; t0 unused except sanity
    del t0
    _print_report(
        "FLAGSHIP MRC VARIANCE",
        baseline_s,
        report,
        pred,
        ref,
        baseline_fit_s=sample["fit_s"],
        baseline_prior_s=sample["prior_s"],
    )

    # Secondary: CPCV smoke-shaped unless quick skips it
    X2, cv2 = make_cpcv(SMOKE_CPCV)
    if not args.skip_baseline:
        t0 = time.perf_counter()
        ref2 = skfolio_cv_predict(MeanRisk(), X2, cv=cv2, n_jobs=-1)
        base2 = time.perf_counter() - t0
    else:
        ref2, base2 = None, 0.0
    pred2, report2 = cross_val_predict(MeanRisk(), X2, cv=cv2, return_report=True)
    _print_report("SMOKE CPCV VARIANCE", base2, report2, pred2, ref2)

    # CVaR kernel sample on walk-forward (smaller)
    wf = WalkForward(train_size=spec["train_size"], test_size=spec["test_size"])
    Xw = X.iloc[: spec["window_size"], : spec["asset_subset_size"]]
    cvar = MeanRisk(risk_measure=RiskMeasure.CVAR)
    if not args.skip_baseline:
        t0 = time.perf_counter()
        ref3 = skfolio_cv_predict(cvar, Xw, cv=wf, n_jobs=1)
        base3 = time.perf_counter() - t0
    else:
        ref3, base3 = None, 0.0
    pred3, report3 = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.CVAR),
        Xw,
        cv=wf,
        return_report=True,
    )
    _print_report("WALKFORWARD CVaR", base3, report3, pred3, ref3)


if __name__ == "__main__":
    main()
