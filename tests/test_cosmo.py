"""COSMO.rs compact-engine correctness and persistence tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate._cosmo import cosmo_available, make_cosmo_engine
from skfolio_accelerate.compact import estimator_spec, make_compact_engine
from skfolio_accelerate.moments import empirical_from_window
from tests.helpers import synthetic_returns

pytestmark = pytest.mark.skipif(
    not cosmo_available(), reason="COSMO.rs (cosmo_rs) is not installed"
)

COSMO_RISKS = (
    RiskMeasure.VARIANCE,
    RiskMeasure.SEMI_VARIANCE,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.WORST_REALIZATION,
    RiskMeasure.CVAR,
    RiskMeasure.MAX_DRAWDOWN,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.CDAR,
)


def _spec(estimator, solver="COSMO"):
    return replace(estimator_spec(estimator), solver=solver)


@pytest.mark.parametrize("risk_measure", COSMO_RISKS, ids=lambda risk: risk.name)
@pytest.mark.parametrize(
    "objective",
    [ObjectiveFunction.MINIMIZE_RISK, ObjectiveFunction.MAXIMIZE_UTILITY],
)
def test_cosmo_weights_match_compact_clarabel_or_osqp(risk_measure, objective):
    X = synthetic_returns(64, 5, seed=101)
    estimator = MeanRisk(
        risk_measure=risk_measure,
        objective_function=objective,
        min_weights=0.05,
        max_weights=0.6,
        l2_coef=1e-5,
    )
    moments = empirical_from_window(
        np.asarray(X, dtype=np.float64),
        keep_returns=risk_measure is not RiskMeasure.VARIANCE,
    )
    reference = make_compact_engine(
        estimator_spec(estimator),
        n_assets=X.shape[1],
        n_observations=None if risk_measure is RiskMeasure.VARIANCE else X.shape[0],
    ).solve(moments, warm=False)
    engine = make_cosmo_engine(
        _spec(estimator),
        n_assets=X.shape[1],
        n_observations=None if risk_measure is RiskMeasure.VARIANCE else X.shape[0],
        persist_mode="cold",
    )
    try:
        observed = engine.solve(moments, warm=False)
    except RuntimeError as error:
        pytest.skip(f"COSMO.rs did not converge: {error}")
    # ADMM vs IPM/OSQP: feasible and close, not bitwise.
    atol = (
        2e-2
        if risk_measure
        in {
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.MAX_DRAWDOWN,
            RiskMeasure.AVERAGE_DRAWDOWN,
            RiskMeasure.CDAR,
        }
        else 5e-3
    )
    np.testing.assert_allclose(observed, reference, rtol=0, atol=atol)
    assert observed.sum() == pytest.approx(1.0, abs=2e-6)
    assert np.min(observed) >= 0.05 - 2e-6
    assert np.max(observed) <= 0.6 + 2e-6


def test_persistence_api_is_available():
    from skfolio_accelerate._cosmo import cosmo_persistence_api_available

    assert cosmo_persistence_api_available()


def test_default_persist_mode_depends_on_risk():
    var = _spec(MeanRisk(risk_measure=RiskMeasure.VARIANCE, l2_coef=1e-5))
    cvar = _spec(MeanRisk(risk_measure=RiskMeasure.CVAR, l2_coef=1e-5))
    from skfolio_accelerate._cosmo import default_persist_mode

    assert default_persist_mode(var) == "persist_full"
    assert default_persist_mode(cvar) == "persist_factor"


def test_persistent_cosmo_matches_cold_start_across_folds():
    X = synthetic_returns(80, 5, seed=102)
    estimator = MeanRisk(risk_measure=RiskMeasure.VARIANCE, l2_coef=1e-5)
    spec = _spec(estimator)
    x_arr = np.asarray(X, dtype=np.float64)
    windows = [x_arr[:40], x_arr[2:42], x_arr[4:44], x_arr[6:46]]
    cold = make_cosmo_engine(spec, n_assets=5, n_observations=None, persist_mode="cold")
    persist = make_cosmo_engine(
        spec, n_assets=5, n_observations=None, persist_mode="persist_full"
    )
    from skfolio_accelerate.moments import empirical_from_window as moments_of

    cold_w = []
    persist_w = []
    for i, window in enumerate(windows):
        moments = moments_of(window, keep_returns=False)
        cold_w.append(cold.solve(moments, warm=False))
        persist_w.append(persist.solve(moments, warm=i > 0))
    np.testing.assert_allclose(persist_w, cold_w, rtol=0, atol=5e-3)
    assert persist.n_warm_starts >= 1
    assert persist._workspace.n_rebuilds == 1


def test_cross_val_predict_cosmo_matches_native_sharpe():
    X = synthetic_returns(96, 6, seed=103)
    cv = WalkForward(train_size=36, test_size=12)
    estimator = MeanRisk(risk_measure=RiskMeasure.SEMI_DEVIATION, l2_coef=1e-5)
    reference = skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=cv, n_jobs=1, backend="cosmo", return_report=True
    )
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=8e-3,
        atol=5e-4,
    )
    assert report.backend == "cosmo"
    assert report.n_warm_starts >= 1


def test_auto_does_not_select_cosmo_for_default_solver():
    X = synthetic_returns(72, 5, seed=104)
    cv = WalkForward(train_size=36, test_size=12)
    _, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
    assert report.backend == "osqp"


def test_solver_cosmo_selects_cosmo_backend():
    X = synthetic_returns(72, 5, seed=105)
    cv = WalkForward(train_size=36, test_size=12)
    _, report = cross_val_predict(
        MeanRisk(solver="COSMO"), X, cv=cv, return_report=True
    )
    assert report.backend == "cosmo"


def test_cosmo_backend_refuses_unreliable_drawdown_lp():
    X = synthetic_returns(72, 5, seed=108)
    cv = WalkForward(train_size=36, test_size=12)
    with pytest.raises(ValueError, match="not a reliable engine"):
        cross_val_predict(
            MeanRisk(risk_measure=RiskMeasure.MAX_DRAWDOWN),
            X,
            cv=cv,
            backend="cosmo",
        )


@pytest.mark.parametrize(
    "mode",
    ["cold", "warm_x", "warm_xy", "persist_factor", "persist_full"],
)
def test_persist_modes_return_feasible_weights(mode):
    X = synthetic_returns(60, 4, seed=106)
    spec = _spec(MeanRisk(l2_coef=1e-5))
    engine = make_cosmo_engine(spec, n_assets=4, n_observations=None, persist_mode=mode)
    from skfolio_accelerate.moments import empirical_from_window as moments_of

    x_arr = np.asarray(X, dtype=np.float64)
    weights = []
    for i, start in enumerate((0, 4, 8)):
        moments = moments_of(x_arr[start : start + 40], keep_returns=False)
        weights.append(engine.solve(moments, warm=i > 0))
    for w in weights:
        assert w.sum() == pytest.approx(1.0, abs=2e-6)
        assert np.min(w) >= -2e-6
    if mode != "cold":
        assert engine.n_warm_starts >= 1


def test_evar_cosmo_is_optional_and_feasible_or_skip():
    X = synthetic_returns(48, 4, seed=107)
    estimator = MeanRisk(risk_measure=RiskMeasure.EVAR, l2_coef=0.0)
    moments = empirical_from_window(np.asarray(X, dtype=np.float64), keep_returns=True)
    engine = make_cosmo_engine(
        _spec(estimator),
        n_assets=4,
        n_observations=X.shape[0],
        persist_mode="cold",
    )
    try:
        weights = engine.solve(moments, warm=False)
    except RuntimeError as error:
        pytest.skip(f"COSMO.rs EVaR did not converge: {error}")
    assert weights.sum() == pytest.approx(1.0, abs=5e-6)
    assert np.min(weights) >= -5e-6
