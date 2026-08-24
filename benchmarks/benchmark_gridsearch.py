"""sklearn GridSearchCV vs MassiveGridSearchCV wall-clock on a small grid."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0005, scale=0.01, size=(180, 25))
    params = {"l2_coef": [1e-4, 1e-3, 1e-2, 1e-1]}
    cv = KFold(n_splits=5, shuffle=False)

    t0 = time.perf_counter()
    GridSearchCV(MeanRisk(), params, cv=cv).fit(X)
    sklearn_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = MassiveGridSearchCV(
        MeanRisk(), params, cv=cv, backend="python", refit=True
    ).fit(X)
    python_s = time.perf_counter() - t0

    print("sklearn", sklearn_s)
    print("python-clarabel", python_s, "speedup", sklearn_s / python_s)
    print(acc.acceleration_report_)


if __name__ == "__main__":
    main()
