"""DPP parameter updates vs naive CVXPY rebuild."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    n = 80
    repeats = 80
    x = cp.Variable(n)
    q = cp.Parameter(n)
    lam = cp.Parameter(nonneg=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x + lam * cp.norm1(x)),
        [x >= -1, x <= 1, cp.sum(x) == 1],
    )
    q_values = rng.normal(size=(repeats, n))
    lam_values = np.logspace(-4, -1, repeats)

    t0 = time.perf_counter()
    for qv, lv in zip(q_values, lam_values, strict=True):
        q.value, lam.value = qv, lv
        problem.solve(solver="CLARABEL", warm_start=True)
    dpp_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for qv, lv in zip(q_values, lam_values, strict=True):
        x2 = cp.Variable(n)
        p = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(x2) + qv @ x2 + lv * cp.norm1(x2)),
            [x2 >= -1, x2 <= 1, cp.sum(x2) == 1],
        )
        p.solve(solver="CLARABEL")
    naive_s = time.perf_counter() - t0
    print("DPP", dpp_s, "naive", naive_s, "speedup", naive_s / dpp_s)


if __name__ == "__main__":
    main()
