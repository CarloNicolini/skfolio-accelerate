"""Optional Moreau smoke tests. Skip when the extra is not installed."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.optimization import MeanRisk, ObjectiveFunction

from skfolio_accelerate.compact import MeanRiskSpec
from skfolio_accelerate.moments import empirical_from_window
from tests.helpers import synthetic_returns

moreau = pytest.importorskip("moreau")
from experiments.moreau_batch import (  # noqa: E402
    solve_mean_variance_batch,
    solve_mean_variance_one,
)


def test_moreau_cpu_qp_smoke():
    import scipy.sparse as sp

    p = sp.diags([1.0, 1.0], format="csr")
    q = np.array([1.0, 1.0])
    a = sp.csr_array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([0.5, 0.5])
    cones = moreau.Cones(num_nonneg_cones=2)
    settings = moreau.Settings(device="cpu", verbose=False)
    solver = moreau.Solver(p, q, a, b, cones=cones, settings=settings)
    solution = solver.solve()
    assert solver.info.status == moreau.SolverStatus.Solved
    assert solution.x.shape == (2,)


def _variance_spec() -> MeanRiskSpec:
    return MeanRiskSpec(
        risk_measure=RiskMeasure.VARIANCE,
        objective=ObjectiveFunction.MINIMIZE_RISK,
        l2_coef=1e-5,
        risk_aversion=1.0,
        cvar_beta=0.95,
        evar_beta=0.95,
        cdar_beta=0.95,
        edar_beta=0.95,
        min_acceptable_return=None,
        min_weights=0.0,
        max_weights=1.0,
        budget=1.0,
    )


def test_batched_matches_sequential_mean_variance():
    x = synthetic_returns(60, 4, seed=3)
    spec = _variance_spec()
    windows = [x[:40], x[10:50], x[20:60]]
    moments = [empirical_from_window(window, keep_returns=False) for window in windows]
    sequential = np.vstack([solve_mean_variance_one(spec, item) for item in moments])
    batched = solve_mean_variance_batch(spec, moments)
    np.testing.assert_allclose(batched, sequential, rtol=1e-5, atol=1e-6)
    from skfolio_accelerate.compact import make_compact_engine

    engine = make_compact_engine(spec, n_assets=4, n_observations=40)
    osqp_w = np.vstack([engine.solve(item) for item in moments])
    np.testing.assert_allclose(batched, osqp_w, rtol=1e-4, atol=1e-5)


def test_cvxpy_moreau_meanrisk_matches_clarabel():
    import cvxpy as cp
    from sklearn.base import clone

    if not hasattr(cp, "MOREAU"):
        pytest.skip("cvxpy MOREAU solver is not registered")
    from experiments.moreau_mean_risk import cvxpy_moreau_available

    ok, reason = cvxpy_moreau_available()
    if not ok:
        pytest.skip(reason)
    x = synthetic_returns(48, 4, seed=8)
    clarabel = MeanRisk(l2_coef=1e-5, solver="CLARABEL")
    moreau_est = clone(clarabel)
    moreau_est.set_params(solver="MOREAU", solver_params={"device": "cpu", "verbose": False})
    clarabel.fit(x)
    moreau_est.fit(x)
    np.testing.assert_allclose(moreau_est.weights_, clarabel.weights_, rtol=1e-4, atol=1e-5)


def test_walk_forward_plan_batches():
    from skfolio_accelerate._arrays import as_float_2d
    from skfolio_accelerate.cv_plan import compile_cv_plan

    from experiments.moreau_batch import batched_weights_for_plan

    x = synthetic_returns(72, 4, seed=2)
    plan = compile_cv_plan(WalkForward(train_size=36, test_size=12), x)
    solved = batched_weights_for_plan(_variance_spec(), as_float_2d(x), plan)
    assert len(solved) == plan.n_splits
    for item in solved:
        assert item.weights.shape == (4,)
        assert abs(float(item.weights.sum()) - 1.0) < 1e-5
