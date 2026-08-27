"""Compare compact OSQP, Clarabel, COSMO, and native skfolio on one CV plan.

Moment construction is shared, so reported solve times exclude empirical-moment
work. Julia JIT warmup is reported separately and is not included in COSMO
fold timings.

Usage::

    PYTHONPATH=src python benchmarks/benchmark_cosmo.py
    PYTHONPATH=src python benchmarks/benchmark_cosmo.py --long
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import MeanRisk
from sklearn.model_selection import KFold

from skfolio_accelerate import path_sharpes
from skfolio_accelerate._cosmo import cosmo_available
from skfolio_accelerate.compact import EngineCache, MeanRiskSpec, estimator_spec
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.moments import path_moment_session
from skfolio_accelerate.scoring import path_sharpes_from_weights


@dataclass(frozen=True)
class SolverCase:
    name: str
    spec_factory: Callable[[MeanRiskSpec], MeanRiskSpec]
    enabled: bool = True


def _with_solver(solver: str) -> Callable[[MeanRiskSpec], MeanRiskSpec]:
    def _apply(spec: MeanRiskSpec) -> MeanRiskSpec:
        return MeanRiskSpec(
            risk_measure=spec.risk_measure,
            objective=spec.objective,
            l2_coef=spec.l2_coef,
            risk_aversion=spec.risk_aversion,
            cvar_beta=spec.cvar_beta,
            evar_beta=spec.evar_beta,
            cdar_beta=spec.cdar_beta,
            edar_beta=spec.edar_beta,
            min_acceptable_return=spec.min_acceptable_return,
            min_weights=spec.min_weights,
            max_weights=spec.max_weights,
            budget=spec.budget,
            solver=solver,
        )

    return _apply


def _solve_folds(spec: MeanRiskSpec, X, plan, *, warm: bool) -> dict:
    keep_returns = spec.needs_returns()
    session = path_moment_session(X, plan.folds, keep_returns=keep_returns)
    engines = EngineCache(spec=spec)
    weights: dict[int, np.ndarray] = {}
    fold_times: list[float] = []
    fold_iters: list[int] = []
    for fold_index, fold in enumerate(plan.folds):
        moments = session.get(fold)
        engine = engines.get(
            int(moments.mu.size),
            int(moments.n_observations) if spec.needs_returns() else None,
        )
        started = time.perf_counter()
        weights[fold.fold_id] = engine.solve(
            moments, warm=bool(warm and fold_index > 0)
        )
        fold_times.append(time.perf_counter() - started)
        fold_iters.append(int(getattr(engine, "last_iterations", 0)))
    n_warm = int(getattr(engines.engine, "n_warm_starts", 0))
    return {
        "weights": weights,
        "fold_times": fold_times,
        "fold_iters": fold_iters,
        "n_warm_starts": n_warm,
        "mean_ms": float(np.mean(fold_times) * 1000.0),
        "sum_s": float(np.sum(fold_times)),
    }


def _print_row(cells: list[str]) -> None:
    print("  ".join(f"{cell:<16}" for cell in cells))


def _scipy_boxed_qp(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float]:
    """SciPy SLSQP boxed min-variance; not an equivalent cone backend."""
    from scipy.optimize import minimize

    n = mu.size

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    started = time.perf_counter()
    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"SciPy SLSQP failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), elapsed


def _q_only_reuse(n_assets: int) -> None:
    """Frozen P and A, varying q: COSMO can skip the KKT refactor."""
    from skfolio_accelerate._cosmo import _cosmo_runtime

    runtime = _cosmo_runtime()
    runtime.warmup()
    workspace = runtime.make_workspace()
    p = sp.eye(n_assets, format="csc", dtype=np.float64)
    a = sp.vstack(
        [sp.csr_matrix(np.ones((1, n_assets))), -sp.eye(n_assets, format="csr")]
    ).tocsc()
    b = np.concatenate([[1.0], np.zeros(n_assets)])
    box_l = np.zeros(n_assets)
    box_u = np.ones(n_assets)
    q_grid = [
        np.zeros(n_assets),
        *[np.linspace(-scale, scale, n_assets) for scale in (0.1, 0.2, 0.3)],
    ]
    times: list[float] = []
    iters: list[int] = []
    for index, q in enumerate(q_grid):
        started = time.perf_counter()
        runtime.solve_qp(
            workspace,
            p,
            q,
            a,
            b,
            n_zero=1,
            n_nonneg=0,
            soc_dims=[],
            n_exp=0,
            box_l=box_l,
            box_u=box_u,
            warm=index > 0,
            update_p=False,
            update_a=False,
        )
        times.append(time.perf_counter() - started)
        iters.append(int(workspace.n_iter))
    print()
    print("== COSMO q-only grid (frozen P, A; factorization reuse) ==")
    print(
        f"first_ms={times[0] * 1000.0:.3f}  "
        f"later_mean_ms={float(np.mean(times[1:])) * 1000.0:.3f}  "
        f"iters={iters}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use a 5-year WalkForward instead of the smoke KFold.",
    )
    args = parser.parse_args()

    n_obs = 5 * 252 if args.long else 120
    n_assets = 12 if args.long else 6
    X = factor_returns(n_obs, n_assets, seed=42)
    cv = (
        WalkForward(train_size=2 * 252, test_size=21)
        if args.long
        else KFold(n_splits=5, shuffle=False)
    )
    plan = compile_cv_plan(cv, X)
    x_arr = np.asarray(X, dtype=np.float64)
    print(f"workload  n_obs={n_obs} n_assets={n_assets} n_folds={plan.n_splits}")
    print(f"cosmo_available={cosmo_available()}")

    risks = [
        RiskMeasure.VARIANCE,
        RiskMeasure.CVAR,
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.EVAR,
    ]
    for risk in risks:
        base = estimator_spec(MeanRisk(risk_measure=risk, l2_coef=1e-5))
        cases = [
            SolverCase(
                "osqp",
                _with_solver("OSQP"),
                enabled=risk is RiskMeasure.VARIANCE,
            ),
            SolverCase(
                "clarabel",
                _with_solver("CLARABEL"),
                enabled=risk is not RiskMeasure.VARIANCE,
            ),
            SolverCase("cosmo", _with_solver("COSMO"), enabled=cosmo_available()),
        ]
        print()
        print(f"== {risk.name} ==")
        native_estimator = MeanRisk(risk_measure=risk, l2_coef=1e-5)
        native_started = time.perf_counter()
        native_pred = skfolio_cross_val_predict(native_estimator, X, cv=cv, n_jobs=1)
        native_s = time.perf_counter() - native_started
        native_scores = path_sharpes(native_pred)
        _print_row(
            [
                "solver",
                "mean_ms/fold",
                "sum_s",
                "warm",
                "mean_iter",
                "max_|dw|",
                "sharpe_d",
                "warm/cold",
            ]
        )
        print(f"native skfolio wall {native_s:.4f}s")
        if risk is RiskMeasure.VARIANCE:
            moments_cov = np.cov(x_arr, rowvar=False, ddof=1)
            moments_mu = x_arr.mean(axis=0)
            try:
                _w_scipy, scipy_s = _scipy_boxed_qp(moments_mu, moments_cov)
                print(
                    f"scipy SLSQP one-shot {scipy_s * 1000.0:.3f}ms (not a CV backend)"
                )
            except Exception as error:
                print(f"scipy SLSQP skipped: {error}")
        for case in cases:
            if not case.enabled:
                continue
            spec = case.spec_factory(base)
            if case.name == "cosmo":
                from skfolio_accelerate._cosmo import _cosmo_runtime

                warm_t0 = time.perf_counter()
                _cosmo_runtime().warmup()
                # The 1-variable JIT warmup does not compile this topology.
                _solve_folds(spec, x_arr, plan, warm=False)
                warmup_s = time.perf_counter() - warm_t0
                print(f"cosmo warmup {warmup_s:.4f}s (excluded from fold times)")
            warm = _solve_folds(spec, x_arr, plan, warm=True)
            cold = _solve_folds(spec, x_arr, plan, warm=False)
            scores = path_sharpes_from_weights(X, plan, warm["weights"])
            max_dw = max(
                float(np.max(np.abs(warm["weights"][k] - cold["weights"][k])))
                for k in warm["weights"]
            )
            sharpe_delta = float(np.max(np.abs(scores - native_scores)))
            speedup = (
                cold["sum_s"] / warm["sum_s"] if warm["sum_s"] > 0 else float("nan")
            )
            _print_row(
                [
                    case.name,
                    f"{warm['mean_ms']:.3f}",
                    f"{warm['sum_s']:.4f}",
                    str(warm["n_warm_starts"]),
                    f"{float(np.mean(warm['fold_iters'])):.1f}",
                    f"{max_dw:.2e}",
                    f"{sharpe_delta:.2e}",
                    f"{speedup:.2f}x",
                ]
            )

    if cosmo_available():
        _q_only_reuse(n_assets)


if __name__ == "__main__":
    main()
