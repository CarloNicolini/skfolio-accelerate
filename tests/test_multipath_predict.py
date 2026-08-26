"""Parity tests for skfolio_accelerate.cross_val_predict vs skfolio."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict as skfolio_cv_predict,
)
from skfolio.optimization import MeanRisk

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.flagship import SMOKE_CPCV, SMOKE_MRC, make_cpcv, make_mrc
from skfolio_accelerate.moments import empirical_from_window, is_default_empirical
from skfolio_accelerate.predict import blocked_reason
from tests.helpers import synthetic_returns


def test_compact_variance_matches_mean_risk():
    X = synthetic_returns(80, 8, seed=4)
    est = MeanRisk(l2_coef=1e-4)
    w_sk = est.fit(X).weights_
    moments = empirical_from_window(np.asarray(X, dtype=np.float64), keep_returns=False)
    engine = make_compact_engine(
        estimator_spec(MeanRisk(l2_coef=1e-4)), n_assets=8, n_observations=None
    )
    w = engine.solve(moments, warm=False)
    np.testing.assert_allclose(w, w_sk, atol=5e-4, rtol=0)


def test_compact_cvar_matches_mean_risk():
    X = synthetic_returns(60, 6, seed=5)
    est = MeanRisk(risk_measure=RiskMeasure.CVAR)
    w_sk = est.fit(X).weights_
    moments = empirical_from_window(np.asarray(X, dtype=np.float64), keep_returns=True)
    engine = make_compact_engine(
        estimator_spec(MeanRisk(risk_measure=RiskMeasure.CVAR)),
        n_assets=6,
        n_observations=60,
    )
    w = engine.solve(moments, warm=False)
    np.testing.assert_allclose(w, w_sk, atol=1e-6, rtol=0)


def test_walkforward_path_matches_skfolio():
    X = synthetic_returns(120, 8, seed=6)
    cv = WalkForward(train_size=40, test_size=10)
    ref = skfolio_cv_predict(MeanRisk(), X, cv=cv)
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(pred.sharpe_ratio, ref.sharpe_ratio, rtol=1e-3, atol=1e-4)
    assert report.n_solves == cv.get_n_splits(X)
    assert report.n_prior_fits < report.n_solves
    assert report.n_warm_starts >= 1
    assert report.n_prior_updates >= 1


def test_cpcv_population_matches_skfolio():
    X = synthetic_returns(48, 6, seed=11)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
    ref = path_sharpes(skfolio_cv_predict(MeanRisk(l2_coef=1e-3), X, cv=cv))
    pred, report = cross_val_predict(
        MeanRisk(l2_coef=1e-3), X, cv=cv, return_report=True
    )
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=1e-3, atol=1e-4)
    assert report.n_solves == cv.get_n_splits()
    assert report.n_prior_fits <= cv.n_folds
    assert report.n_prior_fits < report.n_solves


def test_smoke_mrc_matches_skfolio():
    X, cv = make_mrc(SMOKE_MRC)
    ref = path_sharpes(skfolio_cv_predict(MeanRisk(), X, cv=cv, n_jobs=1))
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=2e-3, atol=1e-4)
    assert report.n_prior_fits < report.n_solves
    assert report.n_warm_starts >= SMOKE_MRC["n_subsamples"]


def test_smoke_cpcv_flagship_helper():
    X, cv = make_cpcv(SMOKE_CPCV)
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    ref = path_sharpes(skfolio_cv_predict(MeanRisk(), X, cv=cv, n_jobs=1))
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=2e-3, atol=1e-4)
    assert report.n_prior_fits < report.n_solves


def test_kfold_still_works():
    X = synthetic_returns(90, 5, seed=8)
    cv = KFold(n_splits=3, shuffle=False)
    ref = skfolio_cv_predict(MeanRisk(), X, cv=cv)
    pred = cross_val_predict(MeanRisk(), X, cv=cv)
    np.testing.assert_allclose(pred.sharpe_ratio, ref.sharpe_ratio, rtol=1e-3, atol=1e-4)


def test_default_empirical_and_blocked_mip():
    assert is_default_empirical(MeanRisk())
    assert blocked_reason(MeanRisk(cardinality=3)) is not None


def test_compile_mrc_records_assets():
    X, cv = make_mrc(SMOKE_MRC)
    plan = compile_cv_plan(cv, X)
    assert plan.kind == "mrc"
    assert plan.multi_path
    assert plan.folds[0].asset_idx is not None
    assert plan.n_paths == SMOKE_MRC["n_subsamples"]
