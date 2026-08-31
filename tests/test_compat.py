"""Compatibility: any skfolio estimator / option / splitter matches skfolio."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import skfolio.optimization as optimization
from skfolio import RiskMeasure
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
    Random,
    RiskBudgeting,
)
from skfolio.prior import EmpiricalPrior
from sklearn.model_selection import KFold, TimeSeriesSplit

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.flagship import SMOKE_MRC, make_mrc
from skfolio_accelerate.predict import blocked_reason
from tests.helpers import synthetic_returns


def _direct_public_estimators():
    cases = []
    for name in dir(optimization):
        estimator_type = getattr(optimization, name)
        if (
            name.startswith("Base")
            or estimator_type is MeanRisk
            or not inspect.isclass(estimator_type)
            or not issubclass(estimator_type, BaseOptimization)
        ):
            continue
        signature = inspect.signature(estimator_type)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ]
        if not required:
            cases.append(pytest.param(estimator_type, id=name))
    return cases


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
    if blocked_reason(estimator) is None:
        assert report.backend in {"osqp", "highs", "clarabel", "closed-form"}
    elif getattr(estimator, "needs_previous_weights", False):
        assert report.backend == "sklearn"
    elif isinstance(estimator, MeanRisk):
        assert report.backend in {"cvxpy-sequential", "fit-assemble"}
    else:
        assert report.backend == "fit-assemble"


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
    assert report.backend == "closed-form"


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
    assert report.backend == "closed-form"

    X2 = synthetic_returns(60, 4, seed=17)
    pipe = Pipeline([("opt", MeanRisk())])
    ref2 = skfolio_cv_predict(pipe, X2, cv=3)
    pred2, report2 = cross_val_predict(pipe, X2, cv=3, return_report=True)
    _assert_same_paths(pred2, ref2)
    assert report2.backend == "sklearn"


@pytest.mark.parametrize("estimator_type", _direct_public_estimators())
def test_directly_constructible_public_estimators_match_native(estimator_type):
    X = synthetic_returns(72, 5, seed=24)
    y = synthetic_returns(72, 1, seed=25).ravel()
    cv = WalkForward(train_size=36, test_size=12)
    np.random.seed(26)
    try:
        reference = skfolio_cv_predict(
            estimator_type(),
            X,
            y=y,
            cv=cv,
            n_jobs=1,
        )
    except Exception as error:
        pytest.skip(f"native skfolio limitation: {type(error).__name__}: {error}")
    np.random.seed(26)
    observed, report = cross_val_predict(
        estimator_type(),
        X,
        y=y,
        cv=WalkForward(train_size=36, test_size=12),
        n_jobs=1,
        return_report=True,
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=1e-8,
        atol=1e-10,
    )
    constructed = estimator_type()
    if type(constructed) in {EqualWeighted, InverseVolatility, Random}:
        assert report.backend == "closed-form"
    else:
        assert report.backend in {"fit-assemble", "sklearn"}


def test_random_uses_closed_form():
    X = synthetic_returns(90, 5, seed=30)
    cv = WalkForward(train_size=40, test_size=10)
    np.random.seed(31)
    reference = skfolio_cv_predict(Random(), X, cv=cv, n_jobs=1)
    np.random.seed(31)
    pred, report = cross_val_predict(Random(), X, cv=cv, n_jobs=1, return_report=True)
    np.testing.assert_allclose(
        path_sharpes(pred), path_sharpes(reference), rtol=0, atol=0
    )
    assert report.backend == "closed-form"


def test_random_closed_form_skips_fit(monkeypatch):
    X = synthetic_returns(60, 4, seed=32)
    cv = WalkForward(train_size=30, test_size=10)
    calls = {"n": 0}
    original = Random.fit

    def counting_fit(self, X, y=None, **fit_params):
        calls["n"] += 1
        return original(self, X, y, **fit_params)

    monkeypatch.setattr(Random, "fit", counting_fit)
    np.random.seed(33)
    pred, report = cross_val_predict(Random(), X, cv=cv, return_report=True)
    assert report.backend == "closed-form"
    assert calls["n"] == 0
    assert len(pred) == cv.get_n_splits(X)


def test_fallback_estimators_assemble_from_native_fit():
    X = synthetic_returns(90, 5, seed=34)
    cv = WalkForward(train_size=40, test_size=10)
    for estimator, expected in (
        (HierarchicalRiskParity(), "fit-assemble"),
        (RiskBudgeting(), "fit-assemble"),
        (MeanRisk(min_return=1e-5), "osqp"),
        (MeanRisk(management_fees=1e-4), "cvxpy-sequential"),
        (MeanRisk(risk_measure=RiskMeasure.STANDARD_DEVIATION), "clarabel"),
    ):
        ref = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
        pred, report = cross_val_predict(
            estimator, X, cv=cv, n_jobs=1, return_report=True
        )
        _assert_same_paths(pred, ref)
        assert report.backend == expected
        assert report.n_solves == cv.get_n_splits(X)


def test_parallel_n_jobs_keeps_native_fallback():
    X = synthetic_returns(72, 4, seed=35)
    cv = WalkForward(train_size=36, test_size=12)
    ref = skfolio_cv_predict(HierarchicalRiskParity(), X, cv=cv, n_jobs=1)
    pred, report = cross_val_predict(
        HierarchicalRiskParity(),
        X,
        cv=cv,
        n_jobs=-1,
        return_report=True,
    )
    _assert_same_paths(pred, ref)
    assert report.backend == "sklearn"

    compact, compact_report = cross_val_predict(
        MeanRisk(),
        X,
        cv=cv,
        n_jobs=-1,
        return_report=True,
    )
    compact_ref = skfolio_cv_predict(MeanRisk(), X, cv=cv, n_jobs=1)
    _assert_same_paths(compact, compact_ref)
    assert compact_report.backend == "sklearn"


def test_transaction_costs_use_native_skfolio():
    X = synthetic_returns(72, 4, seed=36)
    cv = WalkForward(train_size=36, test_size=12)
    estimator = MeanRisk(transaction_costs=1e-4)
    ref = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    pred, report = cross_val_predict(estimator, X, cv=cv, n_jobs=1, return_report=True)
    _assert_same_paths(pred, ref)
    assert report.backend == "sklearn"
    assert "previous_weights" in (report.fallback_reason or "")


def test_shuffled_kfold_is_rejected_like_skfolio():
    X = synthetic_returns(60, 4, seed=37)
    shuffled = KFold(n_splits=3, shuffle=True, random_state=0)
    with pytest.raises(ValueError, match="shuffle"):
        skfolio_cv_predict(MeanRisk(), X, cv=shuffled)
    with pytest.raises(ValueError, match="shuffle"):
        cross_val_predict(
            MeanRisk(), X, cv=KFold(n_splits=3, shuffle=True, random_state=0)
        )
