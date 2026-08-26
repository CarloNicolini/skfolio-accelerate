"""Compatibility: any skfolio estimator / option / splitter matches skfolio."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.prior import EmpiricalPrior
from sklearn.model_selection import TimeSeriesSplit

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.flagship import SMOKE_MRC, make_mrc
from skfolio_accelerate.predict import blocked_reason
from tests.helpers import synthetic_returns


def _assert_same_paths(pred, ref, *, rtol=2e-3, atol=1e-4):
    np.testing.assert_allclose(
        path_sharpes(pred), path_sharpes(ref), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "estimator",
    [
        InverseVolatility(),
        EqualWeighted(),
        MeanRisk(risk_measure=RiskMeasure.SEMI_VARIANCE),
        MeanRisk(risk_measure=RiskMeasure.MEAN_ABSOLUTE_DEVIATION),
        MeanRisk(risk_measure=RiskMeasure.MAX_DRAWDOWN),
        MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO),
        MeanRisk(min_return=1e-5),
        MeanRisk(l1_coef=1e-3),
        MeanRisk(max_variance=1.0),
        MeanRisk(management_fees=1e-4),
        MeanRisk(budget=None, min_budget=0.9, max_budget=1.1),
        MeanRisk(risk_free_rate=1e-4),
        MeanRisk(solver_params={"max_iter": 1_000}),
        MeanRisk(prior_estimator=EmpiricalPrior()),
        HierarchicalRiskParity(),
        RiskBudgeting(),
    ],
)
def test_estimators_and_mean_risk_options_match_skfolio(estimator):
    X = synthetic_returns(90, 5, seed=12)
    cv = WalkForward(train_size=40, test_size=10)
    ref = skfolio_cv_predict(estimator, X, cv=cv)
    pred, report = cross_val_predict(estimator, X, cv=cv, return_report=True)
    _assert_same_paths(pred, ref)
    if type(estimator).__name__ != "MeanRisk" or blocked_reason(estimator) is not None:
        assert report.backend == "sklearn"


def test_default_mean_risk_stays_compact():
    X = synthetic_returns(90, 5, seed=13)
    cv = WalkForward(train_size=40, test_size=10)
    _, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    assert report.backend == "osqp"


def test_time_series_split_and_integer_cv():
    X = synthetic_returns(60, 4, seed=14)
    ref = skfolio_cv_predict(InverseVolatility(), X, cv=TimeSeriesSplit(n_splits=3))
    pred = cross_val_predict(InverseVolatility(), X, cv=TimeSeriesSplit(n_splits=3))
    _assert_same_paths(pred, ref)

    ref2 = skfolio_cv_predict(MeanRisk(), X, cv=3)
    pred2, report = cross_val_predict(MeanRisk(), X, cv=3, return_report=True)
    _assert_same_paths(pred2, ref2)
    assert report.backend == "osqp"


def test_cpcv_other_estimator():
    X = synthetic_returns(48, 4, seed=15)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
    ref = path_sharpes(skfolio_cv_predict(EqualWeighted(), X, cv=cv))
    pred, report = cross_val_predict(EqualWeighted(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=1e-8, atol=1e-10)
    assert report.backend == "sklearn"


def test_skfolio_kwargs_are_accepted():
    X = synthetic_returns(40, 3, seed=16)
    pred = cross_val_predict(
        EqualWeighted(),
        X,
        y=None,
        cv=2,
        n_jobs=1,
        method="predict",
        verbose=0,
        params=None,
        portfolio_params={"name": "eq"},
    )
    assert len(pred) == 2


def test_mrc_and_pipeline_use_skfolio_path():
    from sklearn.pipeline import Pipeline

    X, cv = make_mrc(SMOKE_MRC)
    ref = path_sharpes(skfolio_cv_predict(InverseVolatility(), X, cv=cv))
    pred, report = cross_val_predict(InverseVolatility(), X, cv=cv, return_report=True)
    np.testing.assert_allclose(path_sharpes(pred), ref, rtol=1e-8, atol=1e-10)
    assert report.backend == "sklearn"

    X2 = synthetic_returns(60, 4, seed=17)
    pipe = Pipeline([("opt", MeanRisk())])
    ref2 = skfolio_cv_predict(pipe, X2, cv=3)
    pred2, report2 = cross_val_predict(pipe, X2, cv=3, return_report=True)
    _assert_same_paths(pred2, ref2)
    assert report2.backend == "sklearn"
