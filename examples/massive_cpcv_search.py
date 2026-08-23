"""Massive CombinatorialPurgedCV search with the compiled engine."""

from __future__ import annotations

import numpy as np

from skfolio import RiskMeasure
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0005, scale=0.01, size=(252, 20))
    search = MassiveGridSearchCV(
        MeanRisk(),
        param_grid={
            "risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR],
            "l2_coef": [1e-4, 1e-3, 1e-2],
        },
        cv=CombinatorialPurgedCV(n_folds=6, n_test_folds=2),
        backend="auto",
        n_jobs=1,
        solver_threads=1,
    )
    search.fit(X)
    print(search.acceleration_report_)
    print("best_params_", search.best_params_)
    print("best_score_", search.best_score_)


if __name__ == "__main__":
    main()
