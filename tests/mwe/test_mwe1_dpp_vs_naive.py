"""MWE 1: DPP parameter updates vs naive CVXPY rebuild."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np


def test_dpp_matches_naive_and_is_faster():
    rng = np.random.default_rng(0)
    n = 40
    repeats = 40
    x = cp.Variable(n)
    q = cp.Parameter(n)
    lam = cp.Parameter(nonneg=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x + lam * cp.norm1(x)),
        [x >= -1, x <= 1, cp.sum(x) == 1],
    )
    assert problem.is_dcp(dpp=True)
    q_values = rng.normal(size=(repeats, n))
    lam_values = np.logspace(-4, -1, repeats)

    xs_dpp = []
    t0 = time.perf_counter()
    for qv, lv in zip(q_values, lam_values, strict=True):
        q.value, lam.value = qv, lv
        problem.solve(solver="CLARABEL", warm_start=True)
        xs_dpp.append(x.value.copy())
    dpp_time = time.perf_counter() - t0

    xs_naive = []
    t0 = time.perf_counter()
    for qv, lv in zip(q_values, lam_values, strict=True):
        x2 = cp.Variable(n)
        p = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(x2) + qv @ x2 + lv * cp.norm1(x2)),
            [x2 >= -1, x2 <= 1, cp.sum(x2) == 1],
        )
        p.solve(solver="CLARABEL")
        xs_naive.append(x2.value.copy())
    naive_time = time.perf_counter() - t0

    for a, b in zip(xs_dpp, xs_naive, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    print("DPP", dpp_time, "naive", naive_time, "speedup", naive_time / dpp_time)
    assert dpp_time < naive_time
