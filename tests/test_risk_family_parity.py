"""Parity and eligibility checks for every MeanRisk family."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.model_selection import BaseCrossValidator, KFold, TimeSeriesSplit

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.moments import empirical_from_window
from skfolio_accelerate.predict import blocked_reason
from tests.helpers import synthetic_returns

COMPACT_RISKS = {
    RiskMeasure.VARIANCE,
    RiskMeasure.SEMI_VARIANCE,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.WORST_REALIZATION,
    RiskMeasure.CVAR,
    RiskMeasure.EVAR,
    RiskMeasure.MAX_DRAWDOWN,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.CDAR,
    RiskMeasure.EDAR,
}


def _walk_forward():
    return WalkForward(train_size=36, test_size=12)


def _kfold():
    return KFold(n_splits=3, shuffle=False)


def _time_series():
    return TimeSeriesSplit(n_splits=3)


def _cpcv():
    return CombinatorialPurgedCV(
        n_folds=4,
        n_test_folds=2,
        purged_size=2,
        embargo_size=1,
    )


def _mrc():
    return MultipleRandomizedCV(
        walk_forward=WalkForward(train_size=36, test_size=12),
        n_subsamples=2,
        asset_subset_size=4,
        window_size=84,
        random_state=91,
    )


@pytest.mark.parametrize(
    "cv_factory",
    [
        pytest.param(_walk_forward, id="walk-forward"),
        pytest.param(_kfold, id="kfold"),
        pytest.param(_time_series, id="time-series"),
        pytest.param(_cpcv, id="purged-cpcv"),
        pytest.param(_mrc, id="multiple-randomized"),
    ],
)
@pytest.mark.parametrize("risk_measure", list(RiskMeasure), ids=lambda risk: risk.name)
def test_all_risk_measures_match_native_across_cv(risk_measure, cv_factory):
    X = synthetic_returns(96, 6, seed=90)
    estimator = MeanRisk(risk_measure=risk_measure, l2_coef=1e-5)
    cv = cv_factory()
    try:
        reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    except Exception as error:
        pytest.skip(f"native skfolio limitation: {type(error).__name__}: {error}")

    observed, report = cross_val_predict(
        estimator,
        X,
        cv=cv_factory(),
        n_jobs=1,
        return_report=True,
    )
    reference_scores = path_sharpes(reference)
    observed_scores = path_sharpes(observed)
    assert observed_scores.shape == reference_scores.shape
    np.testing.assert_allclose(
        observed_scores,
        reference_scores,
        rtol=3e-3,
        atol=2e-4,
    )
    if estimator.risk_measure in COMPACT_RISKS:
        assert report.backend in {
            "osqp",
            "highs",
            "clarabel",
            "fit-assemble",
            "sklearn",
        }
        if report.backend in {"sklearn", "fit-assemble"}:
            assert report.fallback_reason is not None
    else:
        assert report.backend in {"cvxpy-sequential", "fit-assemble"}
        if report.backend == "fit-assemble":
            assert report.fallback_reason is not None


@pytest.mark.parametrize(
    "risk_measure",
    sorted(COMPACT_RISKS, key=lambda risk: risk.name),
    ids=lambda risk: risk.name,
)
@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_UTILITY,
    ],
)
def test_compact_family_weights_and_feasibility(risk_measure, objective):
    X = synthetic_returns(80, 5, seed=92)
    estimator = MeanRisk(
        risk_measure=risk_measure,
        objective_function=objective,
        min_weights=0.05,
        max_weights=0.6,
        l2_coef=1e-5,
    )
    reference = estimator.fit(X).weights_
    moments = empirical_from_window(
        np.asarray(X, dtype=np.float64),
        keep_returns=risk_measure is not RiskMeasure.VARIANCE,
    )
    engine = make_compact_engine(
        estimator_spec(estimator),
        n_assets=X.shape[1],
        n_observations=(None if risk_measure is RiskMeasure.VARIANCE else X.shape[0]),
    )
    observed = engine.solve(moments, warm=False)

    if risk_measure is RiskMeasure.VARIANCE:
        tolerance = 5e-4
    elif risk_measure in {RiskMeasure.EVAR, RiskMeasure.EDAR}:
        tolerance = 2e-4
    else:
        tolerance = 2e-5
    np.testing.assert_allclose(observed, reference, rtol=0, atol=tolerance)
    assert observed.sum() == pytest.approx(1.0, abs=2e-7)
    assert np.min(observed) >= 0.05 - 2e-7
    assert np.max(observed) <= 0.6 + 2e-7


@pytest.mark.parametrize("l2_coef", [0.0, 1e-5, 0.1])
def test_analytic_max_return_matches_mean_risk(l2_coef):
    X = synthetic_returns(80, 5, seed=93)
    estimator = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        risk_measure=RiskMeasure.CVAR,
        min_weights=-0.2,
        max_weights=0.6,
        l2_coef=l2_coef,
    )
    reference = estimator.fit(X).weights_
    moments = empirical_from_window(X, keep_returns=False)
    engine = make_compact_engine(
        estimator_spec(estimator),
        n_assets=X.shape[1],
        n_observations=None,
    )
    observed = engine.solve(moments, warm=False)
    np.testing.assert_allclose(observed, reference, rtol=0, atol=2e-5)
    assert observed.sum() == pytest.approx(1.0, abs=1e-12)


def test_max_return_ignores_risk_topology_and_uses_analytic_backend():
    X = synthetic_returns(96, 6, seed=94)
    cv = WalkForward(train_size=36, test_size=12)
    estimator = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        risk_measure=RiskMeasure.ULCER_INDEX,
        l2_coef=1e-5,
    )
    reference = skfolio_cv_predict(estimator, X, cv=cv)
    observed, report = cross_val_predict(estimator, X, cv=cv, return_report=True)
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=1e-6,
        atol=1e-8,
    )
    assert report.backend == "max-return"


@pytest.mark.parametrize(
    "option",
    [
        {"transaction_costs": 1e-4},
        {"management_fees": 1e-4},
        {"previous_weights": np.full(5, 0.2)},
        {"linear_constraints": ["x0 <= 0.5"]},
        {"solver_params": {"max_iter": 1000}},
        {"solver": "SCS"},
        {"scale_objective": 2.0},
        {"scale_constraints": 2.0},
        {"save_problem": True},
    ],
)
def test_compact_families_do_not_drop_unsupported_options(option):
    estimator = MeanRisk(risk_measure=RiskMeasure.CDAR, **option)
    assert blocked_reason(estimator) is not None


def test_compact_solver_failure_retries_native(monkeypatch):
    import skfolio_accelerate.compact as compact

    class FailingEngine:
        n_warm_starts = 0

        def solve(self, moments, *, warm=True):
            raise RuntimeError("deliberate compact failure")

    monkeypatch.setattr(
        compact, "make_compact_engine", lambda *args, **kwargs: FailingEngine()
    )
    X = synthetic_returns(72, 5, seed=93)
    cv = WalkForward(train_size=36, test_size=12)
    reference = skfolio_cv_predict(
        MeanRisk(risk_measure=RiskMeasure.EVAR), X, cv=cv, n_jobs=1
    )
    observed, report = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.EVAR),
        X,
        cv=cv,
        n_jobs=1,
        return_report=True,
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=0,
        atol=0,
    )
    assert report.backend == "fit-assemble"
    assert "deliberate compact failure" in report.fallback_reason


def test_compact_failure_preserves_mutable_randomized_cv_plan(monkeypatch):
    import skfolio_accelerate.compact as compact

    class FailingEngine:
        n_warm_starts = 0

        def solve(self, moments, *, warm=True):
            raise RuntimeError("deliberate randomized failure")

    monkeypatch.setattr(
        compact, "make_compact_engine", lambda *args, **kwargs: FailingEngine()
    )

    class MutableRandomizedCV(BaseCrossValidator):
        shuffle = False

        def __init__(self, seed):
            self.random_state = np.random.RandomState(seed)

        def get_n_splits(self, X=None, y=None, groups=None):
            return 3

        def split(self, X, y=None, groups=None):
            permutation = self.random_state.permutation(len(X))
            for test in np.array_split(permutation, self.get_n_splits()):
                test = np.sort(test)
                train = np.setdiff1d(np.arange(len(X)), test)
                yield train, test

    X = synthetic_returns(96, 6, seed=94)
    cv = MutableRandomizedCV(seed=95)
    reference = skfolio_cv_predict(
        MeanRisk(risk_measure=RiskMeasure.CVAR),
        X,
        cv=copy.deepcopy(cv),
        n_jobs=1,
    )
    observed, report = cross_val_predict(
        MeanRisk(risk_measure=RiskMeasure.CVAR),
        X,
        cv=cv,
        n_jobs=1,
        return_report=True,
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=0,
        atol=0,
    )
    assert report.backend == "fit-assemble"
