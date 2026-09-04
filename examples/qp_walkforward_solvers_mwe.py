#!/usr/bin/env python3
"""Mean-variance expanding-window walkforward: OSQP / Clarabel / COSMO.rs.

Pure numpy + cvxpy (+ cosmo_rs). No skfolio. QP only (HiGHS / CVaR omitted).

Tricks: DPP via chol Parameter, reuse one Problem, warm_start, relaxed tols,
thread caps, incremental cov, COSMO native update_p factor persistence.

  uv pip install cvxpy osqp clarabel numpy scipy
  # COSMO.rs: maturin develop --release --features python  (from cosmo.rs repo)

  python examples/qp_walkforward_solvers_mwe.py
"""
from __future__ import annotations

import os
import time

# Squeeze BLAS contention for fair single-thread-ish ADMM/IPM timing.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np
import cvxpy as cp
from scipy import sparse

# --- problem size (full request; STEP keeps wall-time indicative) ---
N, T = 500, 10_000
MIN_HIST, STEP = 252, 50
SEED, RISK_AVERSION = 0, 1.0
EPS = 1e-4

SOLVER_KW = {
    "OSQP": dict(
        solver=cp.OSQP,
        eps_abs=EPS,
        eps_rel=EPS,
        max_iter=25_000,
        polish=False,
        adaptive_rho=True,
        warm_starting=True,
    ),
    "CLARABEL": dict(
        solver=cp.CLARABEL,
        tol_gap_abs=EPS,
        tol_gap_rel=EPS,
        tol_feas=EPS,
        max_iter=500,
    ),
    "COSMO_RUST": dict(
        solver="COSMO_RUST",
        eps_abs=EPS,
        eps_rel=EPS,
        max_iter=10_000,
        check_termination=25,
    ),
}


def _register_cosmo() -> bool:
    try:
        from cosmo_rs.cvxpy_interface import register

        register()
        return "COSMO_RUST" in cp.installed_solvers()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] COSMO.rs unavailable: {exc}")
        return False


def make_returns(n: int, t: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mild cross-sectional factor so Σ is realistic / well-conditioned.
    f = rng.normal(0, 0.01, size=(t, 5))
    B = rng.normal(0, 0.3, size=(5, n))
    return f @ B + rng.normal(0, 0.01, size=(t, n))


def chol_cov(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False, dtype=np.float64)
    S.flat[:: S.shape[0] + 1] += 1e-8  # jitter
    # Symmetrize then chol (PSD Parameter path is not DPP; L@w is).
    S = 0.5 * (S + S.T)
    L = np.linalg.cholesky(S)
    return mu, L


def build_dpp(n: int) -> tuple[cp.Problem, cp.Parameter, cp.Parameter, cp.Variable]:
    w = cp.Variable(n)
    mu = cp.Parameter(n)
    L = cp.Parameter((n, n))  # Σ = L Lᵀ
    # Markowitz: min γ‖Lᵀ w‖² − μᵀ w  s.t. 1ᵀw=1, w≥0
    obj = cp.Minimize(RISK_AVERSION * cp.sum_squares(L.T @ w) - mu @ w)
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
    assert prob.is_dcp(dpp=True), "problem must be DPP for amortized canonicalization"
    return prob, mu, L, w


def run_cvxpy(
    name: str,
    warm: bool,
    R: np.ndarray,
    cuts: list[int],
) -> dict:
    kw = dict(SOLVER_KW[name])
    prob, mu_p, L_p, w = build_dpp(R.shape[1])
    # First canonicalization charged separately.
    mu, L = chol_cov(R[: cuts[0]])
    mu_p.value, L_p.value = mu, L
    t0 = time.perf_counter()
    prob.solve(**kw, warm_start=False, verbose=False)
    compile_s = time.perf_counter() - t0

    solves: list[float] = []
    objs: list[float] = []
    fails = 0
    for t_end in cuts:
        mu, L = chol_cov(R[:t_end])
        mu_p.value, L_p.value = mu, L
        if not warm:
            w.value = None  # drop primal seed
        t0 = time.perf_counter()
        prob.solve(**kw, warm_start=warm, verbose=False)
        solves.append(time.perf_counter() - t0)
        ok = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        fails += not ok
        objs.append(float(prob.value) if ok else float("nan"))

    arr = np.asarray(solves)
    return dict(
        solver=name,
        mode="warm" if warm else "cold",
        backend="cvxpy",
        n_solves=len(arr),
        compile_s=compile_s,
        total_s=float(arr.sum()),
        mean_s=float(arr.mean()),
        median_s=float(np.median(arr)),
        p95_s=float(np.percentile(arr, 95)),
        fails=fails,
        last_obj=objs[-1] if objs else float("nan"),
    )


def run_cosmo_persist(R: np.ndarray, cuts: list[int], warm: bool) -> dict:
    """COSMO.rs native workspace: update_p / update_q + factor reset."""
    from cosmo_rs import CosmoSolver

    n = R.shape[1]
    # Fixed simplex constraints; only P,q change each day.
    # A x + s = b, s ∈ {0}×R₊^{n}×R₊^{n}  encoding 1ᵀw=1, 0≤w≤1 (redundant ub).
    A = sparse.vstack(
        [
            sparse.csr_matrix(np.ones((1, n))),
            sparse.eye(n, format="csr"),
            -sparse.eye(n, format="csr"),
        ],
        format="csc",
    )
    b = np.concatenate([np.ones(1), np.ones(n), np.zeros(n)])
    cones = [("zero", 1), ("nonnegative", n), ("nonnegative", n)]

    mu, L = chol_cov(R[: cuts[0]])
    # min ½ wᵀ (2γ Σ) w − μᵀ w  ≡  γ‖Lᵀw‖² − μᵀw
    P = sparse.triu(sparse.csc_matrix(2.0 * RISK_AVERSION * (L @ L.T))).tocsc()
    q = -mu
    solver = CosmoSolver(
        P,
        q,
        A,
        b,
        cones,
        verbose=False,
        eps_abs=EPS,
        eps_rel=EPS,
        max_iter=25_000,
        check_termination=25,
    )
    t0 = time.perf_counter()
    sol = solver.solve()
    compile_s = time.perf_counter() - t0

    solves: list[float] = []
    fails = 0
    last_obj = float("nan")
    for t_end in cuts:
        mu, L = chol_cov(R[:t_end])
        P = sparse.triu(sparse.csc_matrix(2.0 * RISK_AVERSION * (L @ L.T))).tocsc()
        q = -mu
        t0 = time.perf_counter()
        solver.update_p(P)
        solver.update_q(q)
        if warm:
            solver.reset("factor")  # keep KKT pattern / ρ; drop ADMM iterates
            if sol is not None and getattr(sol, "x", None) is not None:
                solver.warm_start(x=list(sol.x), y=list(sol.y) if sol.y is not None else None)
        else:
            solver.reset("cold")
        sol = solver.solve()
        solves.append(time.perf_counter() - t0)
        ok = str(sol.status) == "Solved"
        fails += not ok
        last_obj = float(sol.obj_val) if ok else float("nan")

    arr = np.asarray(solves)
    return dict(
        solver="COSMO_RUST",
        mode="warm" if warm else "cold",
        backend="native_persist",
        n_solves=len(arr),
        compile_s=compile_s,
        total_s=float(arr.sum()),
        mean_s=float(arr.mean()),
        median_s=float(np.median(arr)),
        p95_s=float(np.percentile(arr, 95)),
        fails=fails,
        last_obj=last_obj,
    )


def main() -> None:
    has_cosmo = _register_cosmo()
    R = make_returns(N, T, SEED)
    cuts = list(range(MIN_HIST, T + 1, STEP))
    print(
        f"MV QP walkforward | N={N} T={T} min={MIN_HIST} step={STEP} "
        f"rebalances={len(cuts)} | cvxpy={cp.__version__} | "
        f"solvers={[s for s in SOLVER_KW if s != 'COSMO_RUST' or has_cosmo]}"
    )

    rows: list[dict] = []
    for name in SOLVER_KW:
        if name == "COSMO_RUST" and not has_cosmo:
            continue
        for warm in (False, True):
            print(f"\n>>> {name}  mode={'warm' if warm else 'cold'}  (cvxpy)")
            row = run_cvxpy(name, warm, R, cuts)
            rows.append(row)
            print(
                f"    compile={row['compile_s']:.3f}s  total={row['total_s']:.3f}s  "
                f"mean={row['mean_s']*1e3:.1f}ms  med={row['median_s']*1e3:.1f}ms  "
                f"p95={row['p95_s']*1e3:.1f}ms  fails={row['fails']}"
            )

    if has_cosmo:
        for warm in (False, True):
            print(f"\n>>> COSMO_RUST  mode={'warm' if warm else 'cold'}  (native persist)")
            row = run_cosmo_persist(R, cuts, warm)
            rows.append(row)
            print(
                f"    setup={row['compile_s']:.3f}s  total={row['total_s']:.3f}s  "
                f"mean={row['mean_s']*1e3:.1f}ms  med={row['median_s']*1e3:.1f}ms  "
                f"p95={row['p95_s']*1e3:.1f}ms  fails={row['fails']}"
            )

    # Ranking by median solve time (lower is better).
    print("\n=== RANKING (median s/solve, ascending) ===")
    ranked = sorted(rows, key=lambda r: r["median_s"])
    print(
        f"{'#':>2}  {'solver':<12} {'backend':<15} {'mode':<5}  "
        f"{'median_ms':>10} {'mean_ms':>10} {'total_s':>10} {'fails':>5}"
    )
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:>2}  {r['solver']:<12} {r['backend']:<15} {r['mode']:<5}  "
            f"{r['median_s']*1e3:>10.2f} {r['mean_s']*1e3:>10.2f} "
            f"{r['total_s']:>10.2f} {r['fails']:>5}"
        )

    warm = [r for r in rows if r["mode"] == "warm" and r["backend"] == "cvxpy"]
    cold = {r["solver"]: r for r in rows if r["mode"] == "cold" and r["backend"] == "cvxpy"}
    print("\n=== COLD → WARM speedup (cvxpy, mean time) ===")
    for r in warm:
        c = cold[r["solver"]]
        sp = c["mean_s"] / r["mean_s"] if r["mean_s"] > 0 else float("inf")
        print(f"  {r['solver']:<12}  {sp:.2f}×  (cold {c['mean_s']*1e3:.1f} → warm {r['mean_s']*1e3:.1f} ms)")


if __name__ == "__main__":
    main()
