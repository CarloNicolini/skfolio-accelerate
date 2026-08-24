"""Numerical instantiate reuses fold-constant A and matches a full apply."""

from __future__ import annotations

import numpy as np

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compile import extract_problem_template, instantiate
from skfolio_accelerate.estimators.mean_risk_twin import (
    bind_twin_values,
    build_mean_risk_twin,
)
from skfolio_accelerate.moments import FoldMoments
from tests.helpers import synthetic_returns


def _moments(t: int, n: int, seed: int = 0) -> FoldMoments:
    X = synthetic_returns(t, n, seed=seed)
    cov = np.cov(X, rowvar=False) + 1e-8 * np.eye(n)
    return FoldMoments(
        mu=X.mean(axis=0),
        covariance=np.empty((0, 0)),
        cholesky=np.linalg.cholesky(cov),
        returns=X,
        n_observations=t,
    )


def test_l2_instantiate_reuses_A_and_matches_full_apply():
    n = 12
    moments = _moments(80, n)
    twin = build_mean_risk_twin(
        n,
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
    )
    bind_twin_values(twin, moments, l2_coef=1e-4)
    template = extract_problem_template(twin, "var")
    first = instantiate(template, data_token="fold0")
    bind_twin_values(twin, moments, l2_coef=2e-2)
    reused = instantiate(template, data_token="fold0")
    full = instantiate(template, data_token=None)
    assert first.A_data is reused.A_data
    assert first.b is reused.b
    assert reused.A_data is not full.A_data
    np.testing.assert_allclose(reused.P_data, full.P_data, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.A_data, full.A_data, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.q, full.q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.b, full.b, rtol=0, atol=1e-12)


def test_cvar_l2_instantiate_reuses_A():
    t, n = 40, 6
    moments = _moments(t, n, seed=3)
    twin = build_mean_risk_twin(
        n,
        risk_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        n_observations=t,
    )
    bind_twin_values(twin, moments, l2_coef=1e-4)
    template = extract_problem_template(twin, "cvar")
    instantiate(template, data_token="fold0")
    bind_twin_values(twin, moments, l2_coef=1e-2)
    reused = instantiate(template, data_token="fold0")
    full = instantiate(template, data_token=None)
    np.testing.assert_allclose(reused.P_data, full.P_data, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.A_data, full.A_data, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.q, full.q, rtol=0, atol=1e-12)
