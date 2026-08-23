"""CombinatorialPurgedCV search: compiled engine vs repeated MeanRisk.fit."""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone

from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0005, scale=0.01, size=(80, 10))
    cv = CombinatorialPurgedCV(n_folds=5, n_test_folds=2)
    params = {"l2_coef": [1e-4, 1e-3, 1e-2]}

    t0 = time.perf_counter()
    acc = MassiveGridSearchCV(
        MeanRisk(), params, cv=cv, backend="python", refit=False
    ).fit(X)
    compiled_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for train, _tests in cv.split(X):
        for l2 in params["l2_coef"]:
            clone(MeanRisk()).set_params(l2_coef=l2).fit(X[train])
    naive_s = time.perf_counter() - t0
    print("compiled CPCV", compiled_s, "naive fits", naive_s, "speedup", naive_s / compiled_s)
    print(acc.acceleration_report_)


if __name__ == "__main__":
    main()
