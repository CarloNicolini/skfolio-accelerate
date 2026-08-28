"""Parameterized MeanRisk CVXPY reuse: parity, topology, and eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.model_selection import TimeSeriesSplit

from skfolio_accelerate import cross_val_predict, grid_search, path_sharpes
from skfolio_accelerate.mean_risk_problem import ParametricMeanRisk
from skfolio_accelerate.predict import sequential_blocked_reason
from tests.helpers import synthetic_returns

_NON_ANNUALIZED = [rm for rm in RiskMeasure if not rm.is_annualized]


def _walk_forward():
    return WalkForward(train_size=36, test_size=12)


def test_parametric_second_fold_reuses_problem_identity():
    X = synthetic_returns(80, 5, seed=1)
    adapter = ParametricMeanRisk(min_return=1e-5, l2_coef=1e-5)
    adapter.fit(X[:40])
    first_id = id(adapter.last_problem)
    assert adapter.last_problem is not None
    assert adapter.is_dcp_ is True
    assert adapter.n_rebuilds == 1
    reference = MeanRisk(min_return=1e-5, l2_coef=1e-5).fit(X[10:50])
    adapter.fit(X[10:50])
    assert id(adapter.last_problem) == first_id
    assert adapter.n_warm_starts >= 1
    np.testing.assert_allclose(adapter.weights_, reference.weights_, rtol=0, atol=5e-5)


def test_forced_sequential_backend_on_boxed_variance():
    X = synthetic_returns(72, 5, seed=2)
    cv = _walk_forward()
    reference = skfolio_cv_predict(MeanRisk(l2_coef=1e-5), X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        MeanRisk(l2_coef=1e-5),
        X,
        cv=cv,
        n_jobs=1,
        backend="cvxpy-sequential",
        return_report=True,
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=3e-3,
        atol=2e-4,
    )
    assert report.backend == "cvxpy-sequential"
    assert report.n_warm_starts >= 1
    assert report.n_rebuilds == 1


@pytest.mark.parametrize("objective", list(ObjectiveFunction), ids=lambda o: o.name)
@pytest.mark.parametrize("risk_measure", _NON_ANNUALIZED, ids=lambda r: r.name)
def test_all_objectives_and_risks_match_native(objective, risk_measure):
    X = synthetic_returns(96, 5, seed=3)
    estimator = MeanRisk(
        objective_function=objective,
        risk_measure=risk_measure,
        l2_coef=1e-5,
        min_return=1e-6 if objective is not ObjectiveFunction.MAXIMIZE_RETURN else None,
    )
    cv = _walk_forward()
    try:
        reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    except Exception as error:
        pytest.skip(f"native skfolio limitation: {type(error).__name__}: {error}")
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=5e-3,
        atol=5e-4,
    )
    compact = report.backend in {"osqp", "clarabel"}
    sequential = report.backend == "cvxpy-sequential"
    assert compact or sequential
    if sequential:
        assert report.n_rebuilds >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_return": 1e-5},
        {"max_cvar": 0.5},
        {"l1_coef": 1e-3},
        {"budget": None, "min_budget": 0.9, "max_budget": 1.1},
        {"risk_free_rate": 1e-4},
        {"management_fees": 1e-4},
        {
            "objective_function": ObjectiveFunction.MAXIMIZE_RATIO,
            "risk_measure": RiskMeasure.STANDARD_DEVIATION,
        },
        {"risk_measure": RiskMeasure.ULCER_INDEX},
        {"risk_measure": RiskMeasure.GINI_MEAN_DIFFERENCE},
        {"transaction_costs": 1e-4},
    ],
    ids=lambda kwargs: next(iter(kwargs)),
)
def test_previously_blocked_options_use_sequential(kwargs):
    X = synthetic_returns(84, 5, seed=4)
    cv = _walk_forward()
    estimator = MeanRisk(l2_coef=1e-5, **kwargs)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=5e-3,
        atol=5e-4,
    )
    assert report.backend == "cvxpy-sequential"
    assert report.n_solves == cv.get_n_splits(X)


def test_linear_constraints_on_named_assets():
    raw = synthetic_returns(84, 4, seed=5)
    X = pd.DataFrame(raw, columns=["A0", "A1", "A2", "A3"])
    cv = _walk_forward()
    estimator = MeanRisk(linear_constraints=["A0 <= 0.45"], l2_coef=1e-5)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=3e-3,
        atol=2e-4,
    )
    assert report.backend == "cvxpy-sequential"


def test_expanding_window_rebuilds_scenario_topology():
    X = synthetic_returns(80, 4, seed=6)
    cv = TimeSeriesSplit(n_splits=3)
    estimator = MeanRisk(risk_measure=RiskMeasure.CVAR, min_return=1e-6)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=5e-3,
        atol=5e-4,
    )
    assert report.backend == "cvxpy-sequential"
    assert report.n_rebuilds == cv.get_n_splits(X)
    assert report.n_warm_starts == 0


def test_custom_hooks_stay_on_fit_assemble():
    estimator = MeanRisk(add_constraints=lambda w: [w[0] >= 0])
    assert sequential_blocked_reason(estimator) == "add_constraints uses fit-assemble"
    X = synthetic_returns(60, 4, seed=7)
    _, report = cross_val_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, return_report=True
    )
    assert report.backend == "fit-assemble"


def test_sequential_grid_search_ratio_and_limits():
    X = synthetic_returns(72, 4, seed=8)
    cv = WalkForward(train_size=36, test_size=12)
    result = grid_search(
        MeanRisk(l2_coef=1e-5),
        X,
        {"min_return": [1e-6, 5e-6], "max_cvar": [0.8, None]},
        cv=cv,
    )
    assert result.acceleration_report_.backend == "sequential-grid"
    assert result.best_params_ in (
        {"min_return": 1e-6, "max_cvar": 0.8},
        {"min_return": 1e-6, "max_cvar": None},
        {"min_return": 5e-6, "max_cvar": 0.8},
        {"min_return": 5e-6, "max_cvar": None},
    )
    assert np.isfinite(result.best_score_)
