"""Parameterized MeanRisk CVXPY reuse: identity, auto policy, and a few parity cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.model_selection import TimeSeriesSplit

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.mean_risk_problem import ParametricMeanRisk
from skfolio_accelerate.predict import sequential_blocked_reason
from tests.helpers import synthetic_returns


def _walk_forward():
    return WalkForward(train_size=36, test_size=12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_return": 1e-5, "l2_coef": 1e-5},
        {
            "risk_measure": RiskMeasure.SEMI_VARIANCE,
            "min_return": 1e-6,
            "l2_coef": 1e-5,
        },
        {"risk_measure": RiskMeasure.CVAR, "min_return": 1e-6, "l2_coef": 1e-5},
    ],
    ids=["variance-limit", "semi-variance-limit", "cvar-limit"],
)
def test_parametric_disjoint_windows_match_native(kwargs):
    X = synthetic_returns(80, 5, seed=1)
    adapter = ParametricMeanRisk(**kwargs)
    adapter.fit(X[:40])
    first_id = id(adapter.last_problem)
    assert adapter.last_problem is not None
    reference = MeanRisk(**kwargs).fit(X[40:80])
    adapter.fit(X[40:80])
    assert id(adapter.last_problem) == first_id
    assert adapter.n_warm_starts >= 1
    np.testing.assert_allclose(adapter.weights_, reference.weights_, rtol=0, atol=5e-5)
    assert adapter.n_rebuilds == 1


def test_auto_picks_osqp_for_boxed_variance():
    X = synthetic_returns(72, 5, seed=9)
    _, report = cross_val_predict(
        MeanRisk(l2_coef=1e-5), X, cv=_walk_forward(), n_jobs=1, return_report=True
    )
    assert report.backend == "osqp"
    assert report.reason == "boxed MeanRisk variance; compact OSQP"


def test_auto_picks_sequential_for_risk_limits():
    X = synthetic_returns(84, 5, seed=4)
    estimator = MeanRisk(min_return=1e-5, l2_coef=1e-5)
    cv = _walk_forward()
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed), path_sharpes(reference), rtol=5e-3, atol=5e-4
    )
    assert report.backend == "cvxpy-sequential"
    assert "compact subset" in (report.reason or "")
    assert report.n_rebuilds == 1
    assert report.n_warm_starts >= 1


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
        path_sharpes(observed), path_sharpes(reference), rtol=3e-3, atol=2e-4
    )
    assert report.backend == "cvxpy-sequential"


def test_fixed_window_cvar_reuses_one_problem():
    X = synthetic_returns(84, 5, seed=8)
    estimator = MeanRisk(risk_measure=RiskMeasure.CVAR, min_return=1e-6, l2_coef=1e-5)
    _, report = cross_val_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, return_report=True
    )
    assert report.backend == "cvxpy-sequential"
    assert report.n_rebuilds == 1
    assert report.n_warm_starts >= 1


def test_expanding_window_rebuilds_when_T_changes():
    X = synthetic_returns(80, 4, seed=6)
    cv = TimeSeriesSplit(n_splits=3)
    estimator = MeanRisk(risk_measure=RiskMeasure.CVAR, min_return=1e-6)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed), path_sharpes(reference), rtol=5e-3, atol=5e-4
    )
    assert report.backend == "cvxpy-sequential"
    assert report.n_rebuilds == cv.get_n_splits(X)


def test_ratio_and_costs_are_not_sequential():
    assert "MAXIMIZE_RATIO" in (
        sequential_blocked_reason(
            MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO)
        )
        or ""
    )
    assert sequential_blocked_reason(MeanRisk(transaction_costs=1e-4))


def test_custom_hooks_stay_on_fit_assemble():
    estimator = MeanRisk(add_constraints=lambda w: [w[0] >= 0])
    assert sequential_blocked_reason(estimator) == "add_constraints uses fit-assemble"
    X = synthetic_returns(60, 4, seed=7)
    _, report = cross_val_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, return_report=True
    )
    assert report.backend == "fit-assemble"


def test_mean_risk_subclasses_are_not_parameterized():
    from skfolio.optimization import MaximumDiversification

    reason = sequential_blocked_reason(MaximumDiversification())
    assert reason is not None
    assert "MaximumDiversification" in reason
