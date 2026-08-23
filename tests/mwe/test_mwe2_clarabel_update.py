"""MWE 2: Clarabel Python update vs CVXPY solve after apply_parameters."""

from __future__ import annotations

import clarabel
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from skfolio_accelerate.compile import dims_to_clarabel_cones


def test_clarabel_update_matches_cvxpy_solve():
    rng = np.random.default_rng(1)
    n = 30
    x = cp.Variable(n)
    q = cp.Parameter(n)
    lam = cp.Parameter(nonneg=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x + lam * cp.norm1(x)),
        [x >= -1, x <= 1, cp.sum(x) == 1],
    )
    assert problem.is_dcp(dpp=True)

    q.value = rng.normal(size=n)
    lam.value = 1e-3
    data, _, _ = problem.get_problem_data(cp.CLARABEL, enforce_dpp=True)
    param_prob = data[cp.settings.PARAM_PROB]
    cones = dims_to_clarabel_cones(data["dims"])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.presolve_enable = False
    settings.chordal_decomposition_enable = False
    settings.input_sparse_dropzeros = False

    P, qv, _d, A, b = param_prob.apply_parameters(keep_zeros=True, quad_obj=True)
    P = sp.triu(P).tocsc()
    A = -A
    solver = clarabel.DefaultSolver(P, qv, A, b, cones, settings)
    sol = solver.solve()
    problem.solve(solver="CLARABEL", verbose=False)
    np.testing.assert_allclose(np.asarray(sol.x)[:n], x.value, rtol=1e-5, atol=1e-7)

    q.value = rng.normal(size=n)
    lam.value = 2e-2
    problem.solve(solver="CLARABEL", verbose=False)
    P2, q2, _d2, A2, b2 = param_prob.apply_parameters(keep_zeros=True, quad_obj=True)
    solver.update(P=sp.triu(P2).tocsc(), q=q2, A=-A2, b=b2)
    sol2 = solver.solve()
    np.testing.assert_allclose(np.asarray(sol2.x)[:n], x.value, rtol=1e-5, atol=1e-7)
    assert solver.is_data_update_allowed()
