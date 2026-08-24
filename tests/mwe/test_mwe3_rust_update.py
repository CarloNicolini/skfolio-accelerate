"""MWE 3: Rust Clarabel update_data vs the Python Clarabel engine."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.backends.python_clarabel import PythonClarabelEngine
from skfolio_accelerate.backends.rust_clarabel import rust_is_available
from skfolio_accelerate.compile import extract_problem_template, instantiate
from skfolio_accelerate.estimators.mean_risk_twin import (
    bind_twin_values,
    build_mean_risk_twin,
)
from skfolio_accelerate.moments import FoldMoments
from tests.helpers import synthetic_returns

pytestmark = pytest.mark.native


@pytest.mark.skipif(not rust_is_available(), reason="Rust extension is not installed")
def test_rust_update_matches_python_engine():
    from skfolio_accelerate.backends.rust_clarabel import RustClarabelEngine

    X = synthetic_returns(80, 8, seed=3)
    cov = np.cov(X, rowvar=False)
    moments = FoldMoments(
        mu=X.mean(axis=0),
        covariance=cov,
        cholesky=np.linalg.cholesky(cov),
        returns=X,
    )
    twin = build_mean_risk_twin(
        8,
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
    )
    template = extract_problem_template(twin, "var")
    py_engine = PythonClarabelEngine(template)
    rust_engine = RustClarabelEngine(template, n_jobs=1, solver_threads=1)

    for l2 in (1e-4, 1e-3, 1e-2):
        bind_twin_values(twin, moments, l2_coef=l2)
        instance = instantiate(template)
        py_w = py_engine.solve(instance).weights
        rs_w = rust_engine.solve(instance).weights
        np.testing.assert_allclose(rs_w, py_w, rtol=1e-5, atol=1e-7)
