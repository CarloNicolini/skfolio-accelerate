"""MWE 6: CombinatorialPurgedCV path scores vs cross_val_predict."""

from __future__ import annotations

import numpy as np

from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict
from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV
from tests.helpers import synthetic_returns


def test_cpcv_path_scores_match_cross_val_predict():
    X = synthetic_returns(n_observations=48, n_assets=6, seed=11)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
    estimator = MeanRisk(l2_coef=1e-3)
    population = cross_val_predict(estimator, X, cv=cv)
    ref_scores = np.array([ptf.sharpe_ratio for ptf in population])
    ref_mean = float(np.mean(ref_scores))

    search = MassiveGridSearchCV(
        MeanRisk(),
        {"l2_coef": [1e-3]},
        cv=cv,
        backend="python",
        refit=False,
    )
    search.fit(X)
    np.testing.assert_allclose(search.best_score_, ref_mean, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"][0], ref_mean, rtol=1e-4, atol=1e-6
    )
    assert search.acceleration_report_.n_prior_fits == cv.get_n_splits()
