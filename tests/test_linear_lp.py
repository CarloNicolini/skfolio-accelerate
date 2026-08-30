"""HiGHS parametric LP engines for MAD / CVaR / FLPM."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk
from sklearn.model_selection import KFold

from skfolio_accelerate import AccelerationWarning, cross_val_predict, path_sharpes
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.linear_lp import LinearHighs, rolling_shift
from skfolio_accelerate.moments import empirical_from_window
from tests.helpers import synthetic_returns


def test_rolling_shift_detects_walk_forward_step():
    previous = np.arange(12, dtype=float).reshape(4, 3)
    current = np.vstack([previous[1:], previous[-1] + 10])
    assert rolling_shift(previous, current) == 1
    shuffled = previous[[0, 2, 1, 3]]
    assert rolling_shift(previous, shuffled) is None


def test_mad_highs_matches_skfolio_and_reuses_basis():
    X = synthetic_returns(96, 6, seed=11)
    estimator = MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION)
    spec = estimator_spec(estimator)
    engine = make_compact_engine(spec, n_assets=6, n_observations=40)
    assert isinstance(engine, LinearHighs)
    first = empirical_from_window(X[:40], keep_returns=True)
    second = empirical_from_window(X[10:50], keep_returns=True)
    w0 = engine.solve(first, warm=False)
    w1 = engine.solve(second, warm=True)
    assert engine.n_warm_starts >= 1
    assert engine.n_model_passes == 1
    np.testing.assert_allclose(w0.sum(), 1.0, atol=1e-8)
    reference = MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION).fit(X[10:50])
    np.testing.assert_allclose(w1, reference.weights_, atol=5e-5)


def test_auto_picks_highs_for_boxed_mad():
    X = synthetic_returns(84, 5, seed=3)
    _, report = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION),
        X,
        cv=WalkForward(train_size=36, test_size=12),
        n_jobs=1,
        return_report=True,
    )
    assert report.backend == "highs"
    assert report.n_warm_starts >= 1
    assert "HiGHS" in (report.reason or "")


def test_cvar_highs_matches_skfolio():
    X = synthetic_returns(70, 5, seed=4)
    estimator = MeanRisk(risk_measure=RiskMeasure.CVAR)
    engine = make_compact_engine(
        estimator_spec(estimator), n_assets=5, n_observations=70
    )
    assert isinstance(engine, LinearHighs)
    moments = empirical_from_window(X, keep_returns=True)
    observed = engine.solve(moments, warm=False)
    reference = estimator.fit(X).weights_
    np.testing.assert_allclose(observed, reference, atol=5e-5)


def test_disjoint_kfold_still_solves_mad():
    X = synthetic_returns(90, 4, seed=8)
    pred, report = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION),
        X,
        cv=KFold(n_splits=3, shuffle=False),
        n_jobs=1,
        return_report=True,
    )
    assert report.backend == "highs"
    assert len(pred) == 3


def test_cpcv_mad_falls_back_to_native_skfolio():
    """Non-rolling MAD LPs are slower with HiGHS; auto uses native skfolio."""
    X = synthetic_returns(96, 6, seed=12)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2, purged_size=1, embargo_size=1)
    with pytest.warns(AccelerationWarning, match="native skfolio"):
        _, report = cross_val_predict(
            MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION),
            X,
            cv=cv,
            n_jobs=1,
            return_report=True,
        )
    assert report.backend == "sklearn"
    assert "CombinatorialPurgedCV" in (report.reason or report.fallback_reason or "")


def test_cpcv_cvar_stays_on_highs():
    X = synthetic_returns(96, 6, seed=13)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2, purged_size=1, embargo_size=1)
    _, report = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.CVAR),
        X,
        cv=cv,
        n_jobs=1,
        return_report=True,
    )
    assert report.backend == "highs"


@pytest.mark.parametrize(
    "cv",
    [
        WalkForward(train_size=36, test_size=12),
        MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=36, test_size=12),
            n_subsamples=2,
            asset_subset_size=4,
            window_size=72,
            random_state=14,
        ),
        CombinatorialPurgedCV(
            n_folds=4,
            n_test_folds=2,
            purged_size=1,
            embargo_size=1,
        ),
    ],
    ids=["walk-forward", "multiple-randomized", "purged-cpcv"],
)
def test_highs_cvar_matches_native_across_cv(cv):
    X = synthetic_returns(96, 6, seed=15)
    estimator = MeanRisk(risk_measure=RiskMeasure.CVAR)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator,
        X,
        cv=cv,
        n_jobs=1,
        return_report=True,
    )

    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=1e-6,
        atol=1e-8,
    )
    assert report.backend == "highs"
