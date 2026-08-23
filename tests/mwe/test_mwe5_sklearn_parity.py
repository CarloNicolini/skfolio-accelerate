"""MWE 5: MassiveGridSearchCV KFold parity vs sklearn GridSearchCV."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV
from tests.helpers import synthetic_returns


def test_massive_engine_matches_sklearn():
    X = synthetic_returns(n_observations=90, n_assets=8, seed=42)
    params = {"l2_coef": [1e-3, 1e-2]}
    cv = KFold(n_splits=3, shuffle=False)
    ref = GridSearchCV(MeanRisk(), params, cv=cv, error_score="raise")
    acc = MassiveGridSearchCV(
        MeanRisk(),
        params,
        cv=cv,
        backend="python",
        refit=True,
    )
    ref.fit(X)
    acc.fit(X)
    np.testing.assert_allclose(
        acc.cv_results_["mean_test_score"],
        ref.cv_results_["mean_test_score"],
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(acc.best_score_, ref.best_score_, rtol=1e-4, atol=1e-6)
    assert acc.best_params_ == ref.best_params_
    assert acc.acceleration_report_.n_prior_fits == cv.get_n_splits()
    assert acc.acceleration_report_.n_prior_fits < cv.get_n_splits() * len(
        params["l2_coef"]
    )
