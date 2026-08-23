"""CVaR twin vs MeanRisk.fit."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk

from skfolio_accelerate.backends.python_clarabel import PythonClarabelEngine
from skfolio_accelerate.compile import extract_problem_template, instantiate
from skfolio_accelerate.estimators.mean_risk_twin import (
    bind_from_estimator,
    build_twin_from_estimator,
)
from skfolio_accelerate.moments import fit_prior
from tests.helpers import synthetic_returns


def test_cvar_twin_matches_mean_risk_fit():
    X = synthetic_returns(40, 5, seed=9)
    params = {"risk_measure": RiskMeasure.CVAR, "l2_coef": 1e-3}
    estimator = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        l2_coef=1e-3,
        solver="CLARABEL",
    )
    fitted = clone(estimator).fit(X)
    moments = fit_prior(estimator, X)
    twin = build_twin_from_estimator(
        estimator, params, n_observations=X.shape[0], n_assets=X.shape[1]
    )
    bind_from_estimator(twin, moments, estimator, params)
    template = extract_problem_template(twin, "cvar")
    result = PythonClarabelEngine(template).solve(instantiate(template))
    np.testing.assert_allclose(result.weights, fitted.weights_, rtol=1e-4, atol=1e-5)
