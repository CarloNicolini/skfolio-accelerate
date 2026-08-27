"""Moment-update contracts: sufficient statistics, not a different estimator."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
from skfolio.moments import EmpiricalCovariance
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior

from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.moments import (
    OverlapMomentCache,
    empirical_from_stats,
    empirical_from_window,
    is_default_empirical,
)
from skfolio_accelerate.predict import blocked_reason, cross_val_predict
from tests.helpers import synthetic_returns


def test_gram_formula_matches_numpy_cov():
    window = synthetic_returns(40, 6, seed=7)
    stats = empirical_from_stats(
        window.shape[0],
        window.sum(axis=0),
        window.T @ window,
        returns=None,
        keep_returns=False,
    )
    direct = empirical_from_window(window, keep_returns=False)
    np.testing.assert_allclose(stats.mu, np.mean(window, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        stats.covariance, np.cov(window, rowvar=False, ddof=1), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(stats.covariance, direct.covariance, rtol=0, atol=1e-12)


def test_single_asset_matches_sample_variance():
    window = synthetic_returns(30, 1, seed=8)
    moments = empirical_from_window(window, keep_returns=False)
    np.testing.assert_allclose(
        moments.covariance, np.var(window, axis=0, ddof=1).reshape(1, 1)
    )


def test_walkforward_slides_match_full_window_recompute():
    X = synthetic_returns(80, 5, seed=9)
    cv = WalkForward(train_size=30, test_size=10)
    plan = compile_cv_plan(cv, X)
    cache = OverlapMomentCache(X, keep_returns=False)
    for fold in plan.folds:
        observed = cache.get(fold, path_key=0)
        window = X[fold.train_idx]
        np.testing.assert_allclose(
            observed.mu, np.mean(window, axis=0), rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            observed.covariance,
            np.cov(window, rowvar=False, ddof=1),
            rtol=0,
            atol=1e-10,
        )
    assert cache.n_fits == 1
    assert cache.n_updates == plan.n_splits - 1


def test_purged_cpcv_blocks_match_training_rows():
    X = synthetic_returns(60, 4, seed=10)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2, purged_size=1, embargo_size=1)
    plan = compile_cv_plan(cv, X)
    assert plan.fold_blocks is not None
    cache = OverlapMomentCache(X, keep_returns=True, fold_blocks=plan.fold_blocks)
    for fold in plan.folds:
        observed = cache.get(fold)
        window = X[fold.train_idx]
        np.testing.assert_allclose(observed.mu, np.mean(window, axis=0), atol=1e-10)
        np.testing.assert_allclose(
            observed.covariance,
            np.cov(window, rowvar=False, ddof=1),
            atol=1e-10,
        )
        np.testing.assert_allclose(observed.returns, window)
    assert cache.n_fits == cv.n_folds
    assert cache.n_updates == cv.get_n_splits()


def test_assume_centered_prior_is_not_compacted():
    estimator = MeanRisk(
        prior_estimator=EmpiricalPrior(
            covariance_estimator=EmpiricalCovariance(assume_centered=True)
        )
    )
    assert not is_default_empirical(estimator)
    assert blocked_reason(estimator) == "custom prior is not compacted"
    X = synthetic_returns(48, 4, seed=11)
    _, report = cross_val_predict(
        estimator, X, cv=WalkForward(train_size=24, test_size=8), return_report=True
    )
    assert report.backend == "fit-assemble"


def test_compiled_plan_is_immutable():
    X = synthetic_returns(40, 3, seed=12)
    plan = compile_cv_plan(WalkForward(train_size=20, test_size=5), X)
    with pytest.raises(AttributeError):
        plan.kind = "mrc"
    with pytest.raises(AttributeError):
        plan.folds[0].fold_id = 99
