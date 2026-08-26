"""Twenty-year cross-validation and parameter-search benchmark."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import (
    cross_val_predict as skfolio_cross_val_predict,
)
from skfolio.optimization import MeanRisk
from sklearn.model_selection import ParameterGrid

from skfolio_accelerate import cross_val_predict, grid_search, path_sharpes
from skfolio_accelerate.flagship import factor_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    X = factor_returns(20 * 252, 120, seed=21)
    mrc = MultipleRandomizedCV(
        walk_forward=WalkForward(train_size=2 * 252, test_size=21),
        n_subsamples=200,
        asset_subset_size=40,
        window_size=10 * 252,
        random_state=21,
    )

    reference = None
    baseline_s = float("nan")
    if not args.skip_baseline:
        started = time.perf_counter()
        reference = skfolio_cross_val_predict(MeanRisk(), X, cv=mrc, n_jobs=-1)
        baseline_s = time.perf_counter() - started

    started = time.perf_counter()
    prediction, report = cross_val_predict(MeanRisk(), X, cv=mrc, return_report=True)
    accelerated_s = time.perf_counter() - started
    print("20-year MultipleRandomizedCV")
    print(f"  data: {X.shape[0]} days × {X.shape[1]} assets")
    print(f"  paths: 200; assets/path: 40; solves: {report.n_solves:,}")
    if reference is not None:
        print(f"  skfolio: {baseline_s:.3f}s")
    print(f"  accelerated: {accelerated_s:.3f}s")
    if reference is not None:
        delta = np.max(np.abs(path_sharpes(prediction) - path_sharpes(reference)))
        print(f"  speedup: {baseline_s / accelerated_s:.2f}×")
        print(f"  max |path Sharpe difference|: {delta:.3e}")

    cpcv = CombinatorialPurgedCV(
        n_folds=10,
        n_test_folds=2,
        purged_size=5,
        embargo_size=5,
    )
    cpcv_reference = None
    cpcv_baseline_s = float("nan")
    if not args.skip_baseline:
        started = time.perf_counter()
        cpcv_reference = skfolio_cross_val_predict(MeanRisk(), X, cv=cpcv, n_jobs=-1)
        cpcv_baseline_s = time.perf_counter() - started

    started = time.perf_counter()
    cpcv_prediction, cpcv_report = cross_val_predict(
        MeanRisk(), X, cv=cpcv, return_report=True
    )
    cpcv_accelerated_s = time.perf_counter() - started
    print("20-year purged CPCV")
    print("  folds: 10; test folds/split: 2; solves: 45")
    if cpcv_reference is not None:
        print(f"  skfolio: {cpcv_baseline_s:.3f}s")
    print(f"  accelerated: {cpcv_accelerated_s:.3f}s")
    print(
        f"  moment fits: {cpcv_report.n_prior_fits}; "
        f"moment updates: {cpcv_report.n_prior_updates}"
    )
    if cpcv_reference is not None:
        delta = np.max(
            np.abs(path_sharpes(cpcv_prediction) - path_sharpes(cpcv_reference))
        )
        print(f"  speedup: {cpcv_baseline_s / cpcv_accelerated_s:.2f}×")
        print(f"  max |path Sharpe difference|: {delta:.3e}")

    grid_X = X.iloc[:, :100]
    walk_forward = WalkForward(train_size=2 * 252, test_size=21)
    param_grid = {"l2_coef": np.logspace(-5, -1, 16)}
    baseline_scores = []
    baseline_grid_s = float("nan")
    if not args.skip_baseline:
        started = time.perf_counter()
        for params in ParameterGrid(param_grid):
            pred = skfolio_cross_val_predict(
                MeanRisk(**params), grid_X, cv=walk_forward, n_jobs=-1
            )
            baseline_scores.append(pred.sharpe_ratio)
        baseline_grid_s = time.perf_counter() - started

    started = time.perf_counter()
    result = grid_search(MeanRisk(), grid_X, param_grid, cv=walk_forward)
    accelerated_grid_s = time.perf_counter() - started
    print("20-year WalkForward parameter search")
    print("  candidates: 16; assets: 100")
    if baseline_scores:
        print(f"  skfolio repeated CV: {baseline_grid_s:.3f}s")
    print(f"  shared-moment search: {accelerated_grid_s:.3f}s")
    if baseline_scores:
        score_delta = np.max(
            np.abs(np.asarray(baseline_scores) - result.cv_results_["mean_test_score"])
        )
        print(f"  speedup: {baseline_grid_s / accelerated_grid_s:.2f}×")
        print(f"  max |Sharpe difference|: {score_delta:.3e}")
    best_l2 = float(result.best_params_["l2_coef"])
    print(f"  best l2_coef: {best_l2:.3g}")


if __name__ == "__main__":
    main()
