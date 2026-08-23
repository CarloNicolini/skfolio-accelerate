"""Prior/moment cache: one fit per fold vs fit per (fold, param)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import KFold

from skfolio.optimization import MeanRisk

from skfolio_accelerate.cv_plan import compile_cv_plan, slice_rows
from skfolio_accelerate.moments import FoldCache, fit_prior


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0005, scale=0.01, size=(150, 20))
    cv = KFold(n_splits=5, shuffle=False)
    plan = compile_cv_plan(cv, X)
    l2_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    estimator = MeanRisk()

    t0 = time.perf_counter()
    cache = FoldCache()
    for fold in plan.folds:
        X_train = slice_rows(X, fold.train_idx)
        for _ in l2_grid:
            cache.get(fold.fold_id, estimator, X_train)
    cached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_fits = 0
    for fold in plan.folds:
        X_train = slice_rows(X, fold.train_idx)
        for _ in l2_grid:
            fit_prior(estimator, X_train)
            n_fits += 1
    naive_s = time.perf_counter() - t0
    print(
        "cached fits",
        cache.n_fits,
        cached_s,
        "naive fits",
        n_fits,
        naive_s,
        "speedup",
        naive_s / cached_s,
    )


if __name__ == "__main__":
    main()
