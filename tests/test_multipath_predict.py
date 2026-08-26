"""Parity tests for skfolio_accelerate.cross_val_predict vs skfolio."""

from __future__ import annotations

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
)
from skfolio.model_selection import (
    cross_val_predict as skfolio_cv_predict,
)
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex import ObjectiveFunction
from sklearn.model_selection import KFold

from skfolio_accelerate import cross_val_predict, grid_search, path_sharpes
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.cvxpy_engine import CvxpyParamEngine, uses_cvxpy_params
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


def test_cvxpy_params_match_mean_risk_with_min_return_and_l1():
    X = synthetic_returns(80, 8, seed=22)
    estimators = [
        MeanRisk(min_return=1e-5),
        MeanRisk(l1_coef=1e-3, l2_coef=1e-4),
        MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            min_return=1e-5,
            l1_coef=1e-4,
        ),
    ]
    for estimator in estimators:
        spec = estimator_spec(estimator)
        assert uses_cvxpy_params(spec)
        moments = empirical_from_window(
            np.asarray(X, dtype=np.float64), keep_returns=False
        )
        engine = make_compact_engine(spec, n_assets=8, n_observations=None)
        assert isinstance(engine, CvxpyParamEngine)
        w = engine.solve(moments, warm=False)
        w_sk = estimator.fit(X).weights_
        np.testing.assert_allclose(w, w_sk, atol=5e-4, rtol=0)


def test_walkforward_cvxpy_params_reuse_compiled_problem():
    X = synthetic_returns(120, 8, seed=23)
    cv = WalkForward(train_size=40, test_size=10)
    estimator = MeanRisk(min_return=1e-5, l1_coef=1e-4)
    ref = skfolio_cv_predict(estimator, X, cv=cv)
    pred, report = cross_val_predict(estimator, X, cv=cv, return_report=True)
    np.testing.assert_allclose(
        pred.sharpe_ratio, ref.sharpe_ratio, rtol=1e-3, atol=1e-4
    )
    assert report.backend == "cvxpy"
    assert report.n_solves == cv.get_n_splits(X)
    assert report.n_prior_fits == 1
    assert report.n_prior_updates >= 1
    assert report.n_warm_starts >= 1


def test_walkforward_path_matches_skfolio():
    X = synthetic_returns(120, 8, seed=6)
    cv = WalkForward(train_size=40, test_size=10)
    ref = skfolio_cv_predict(MeanRisk(), X, cv=cv)
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(
        pred.sharpe_ratio, ref.sharpe_ratio, rtol=1e-3, atol=1e-4
    )
    assert report.n_solves == cv.get_n_splits(X)
    assert report.n_prior_fits < report.n_solves
    assert report.n_warm_starts >= 1
    assert report.n_prior_updates >= 1
    assert np.shares_memory(pred.portfolios[0].X, X)


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


def test_purged_cpcv_reuses_blocks_and_matches_skfolio():
    X = synthetic_returns(120, 6, seed=21)
    cv = CombinatorialPurgedCV(
        n_folds=6,
        n_test_folds=2,
        purged_size=3,
        embargo_size=2,
    )
    ref = path_sharpes(skfolio_cv_predict(MeanRisk(), X, cv=cv))
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=2e-3, atol=1e-4)
    assert report.n_prior_fits == cv.n_folds
    assert report.n_prior_updates == cv.get_n_splits()


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
    X = synthetic_returns(100, 5, seed=8)
    cv = KFold(n_splits=5, shuffle=False)
    ref = skfolio_cv_predict(MeanRisk(), X, cv=cv)
    pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(
        pred.sharpe_ratio, ref.sharpe_ratio, rtol=1e-3, atol=1e-4
    )
    assert report.n_prior_fits == 1
    assert report.n_prior_updates == 4


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


def test_grid_search_shares_moments_and_matches_repeated_predict():
    X = synthetic_returns(100, 6, seed=18)
    cv = WalkForward(train_size=40, test_size=10)
    values = [0.0, 1e-3, 1e-2]
    result = grid_search(MeanRisk(), X, {"l2_coef": values}, cv=cv)

    expected = np.asarray(
        [
            cross_val_predict(MeanRisk(l2_coef=value), X, cv=cv).sharpe_ratio
            for value in values
        ]
    )
    np.testing.assert_allclose(
        result.cv_results_["mean_test_score"], expected, rtol=1e-12, atol=1e-12
    )
    assert result.best_index_ == int(np.argmax(expected))
    assert result.best_params_ == {"l2_coef": values[result.best_index_]}
    assert result.acceleration_report_.n_prior_fits == 1
    assert result.acceleration_report_.n_solves == len(values) * cv.get_n_splits(X)


def test_cpcv_grid_search_matches_repeated_predict():
    X = synthetic_returns(48, 5, seed=19)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
    values = [0.0, 1e-2]
    result = grid_search(MeanRisk(), X, {"l2_coef": values}, cv=cv)
    expected = np.asarray(
        [
            np.mean(path_sharpes(cross_val_predict(MeanRisk(l2_coef=value), X, cv=cv)))
            for value in values
        ]
    )
    np.testing.assert_allclose(
        result.cv_results_["mean_test_score"], expected, rtol=1e-12, atol=1e-12
    )


def test_grid_search_rejects_options_not_in_compact_problem():
    X = synthetic_returns(48, 5, seed=20)
    cv = WalkForward(train_size=24, test_size=8)
    with np.testing.assert_raises_regex(ValueError, "maximum variance"):
        grid_search(
            MeanRisk(max_variance=1.0),
            X,
            {"l2_coef": [0.0, 1e-2]},
            cv=cv,
        )


def test_mrc_grid_search_matches_repeated_predict():
    X, cv = make_mrc(SMOKE_MRC)
    values = [0.0, 1e-2]
    result = grid_search(MeanRisk(), X, {"l2_coef": values}, cv=cv)
    expected = np.asarray(
        [
            np.mean(path_sharpes(cross_val_predict(MeanRisk(l2_coef=value), X, cv=cv)))
            for value in values
        ]
    )
    np.testing.assert_allclose(
        result.cv_results_["mean_test_score"], expected, rtol=1e-12, atol=1e-12
    )
