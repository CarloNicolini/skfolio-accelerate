"""COSMO.jl compact engines: skip when the optional runtime is missing.

Weight gates for variance / CVaR / SOC match the Clarabel compact tests.
MAD and EVaR use COSMO's default ADMM tolerances (see
:func:`skfolio_accelerate._cosmo._workspace_options`); those families are
opt-in and are allowed a looser ``atol``.
"""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.model_selection import KFold

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate._cosmo import cosmo_available
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.moments import empirical_from_window
from skfolio_accelerate.predict import blocked_reason
from tests.helpers import synthetic_returns

COSMO_RISKS = (
    RiskMeasure.VARIANCE,
    RiskMeasure.CVAR,
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.EVAR,
)


def test_cosmo_missing_runtime_blocks_compaction():
    if cosmo_available():
        pytest.skip("COSMO runtime is installed")
    reason = blocked_reason(MeanRisk(solver="COSMO"))
    assert reason == "COSMO runtime is not installed"


@pytest.mark.cosmo
@pytest.mark.skipif(not cosmo_available(), reason="COSMO.jl runtime is not installed")
def test_default_mean_risk_stays_osqp_when_cosmo_is_installed():
    X = synthetic_returns(90, 5, seed=13)
    cv = WalkForward(train_size=40, test_size=10)
    _, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    assert report.backend == "osqp"


@pytest.mark.cosmo
@pytest.mark.skipif(not cosmo_available(), reason="COSMO.jl runtime is not installed")
@pytest.mark.parametrize("risk_measure", COSMO_RISKS, ids=lambda risk: risk.name)
@pytest.mark.parametrize(
    "objective",
    [ObjectiveFunction.MINIMIZE_RISK, ObjectiveFunction.MAXIMIZE_UTILITY],
)
def test_cosmo_family_weights_and_feasibility(risk_measure, objective):
    X = synthetic_returns(80, 5, seed=92)
    estimator = MeanRisk(
        solver="COSMO",
        risk_measure=risk_measure,
        objective_function=objective,
        min_weights=0.05,
        max_weights=0.6,
        l2_coef=1e-5,
    )
    assert blocked_reason(estimator) is None
    reference = MeanRisk(
        risk_measure=risk_measure,
        objective_function=objective,
        min_weights=0.05,
        max_weights=0.6,
        l2_coef=1e-5,
    )
    reference_weights = reference.fit(X).weights_
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
        tolerance = 5e-2
    elif risk_measure is RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
        tolerance = 1e-2
    else:
        tolerance = 2e-5
    np.testing.assert_allclose(observed, reference_weights, rtol=0, atol=tolerance)
    assert observed.sum() == pytest.approx(1.0, abs=2e-7)
    assert np.min(observed) >= 0.05 - 2e-7
    assert np.max(observed) <= 0.6 + 2e-7


@pytest.mark.cosmo
@pytest.mark.skipif(not cosmo_available(), reason="COSMO.jl runtime is not installed")
def test_cosmo_kfold_warm_starts():
    X = synthetic_returns(60, 4, seed=14)
    estimator = MeanRisk(solver="COSMO")
    n_splits = 3
    pred, report = cross_val_predict(
        estimator,
        X,
        cv=KFold(n_splits=n_splits, shuffle=False),
        return_report=True,
    )
    reference = skfolio_cv_predict(
        MeanRisk(), X, cv=KFold(n_splits=n_splits, shuffle=False)
    )
    np.testing.assert_allclose(
        path_sharpes(pred),
        path_sharpes(reference),
        rtol=3e-3,
        atol=2e-4,
    )
    assert report.backend == "cosmo"
    assert report.n_warm_starts == n_splits - 1
    assert report.n_solves == n_splits


@pytest.mark.cosmo
@pytest.mark.skipif(not cosmo_available(), reason="COSMO.jl runtime is not installed")
@pytest.mark.parametrize(
    "risk_measure",
    [RiskMeasure.CVAR, RiskMeasure.MEAN_ABSOLUTE_DEVIATION],
    ids=lambda risk: risk.name,
)
def test_cosmo_walk_forward_matches_native(risk_measure):
    X = synthetic_returns(90, 5, seed=12)
    cv = WalkForward(train_size=40, test_size=10)
    estimator = MeanRisk(solver="COSMO", risk_measure=risk_measure, l2_coef=1e-5)
    reference = skfolio_cv_predict(
        MeanRisk(risk_measure=risk_measure, l2_coef=1e-5),
        X,
        cv=cv,
        n_jobs=1,
    )
    pred, report = cross_val_predict(
        estimator,
        X,
        cv=WalkForward(train_size=40, test_size=10),
        n_jobs=1,
        return_report=True,
    )
    # MAD ADMM weights are looser than Clarabel; CVaR matches tightly.
    rtol = 5e-2 if risk_measure is RiskMeasure.MEAN_ABSOLUTE_DEVIATION else 3e-3
    atol = 5e-3 if risk_measure is RiskMeasure.MEAN_ABSOLUTE_DEVIATION else 2e-4
    np.testing.assert_allclose(
        path_sharpes(pred),
        path_sharpes(reference),
        rtol=rtol,
        atol=atol,
    )
    assert report.backend == "cosmo"
    assert report.n_warm_starts >= 1
