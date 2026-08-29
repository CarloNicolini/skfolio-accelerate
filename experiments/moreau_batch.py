"""Boxed mean-variance via Moreau Solver / CompiledSolver.

Matches the compact OSQP quadratic
``wᵀ (scale Σ + ℓ₂ I) w + qᵀ w`` with budget equality and box bounds.
Moreau uses the same ``½ xᵀ P x + qᵀ x`` convention as OSQP, so
``P = 2 scale Σ + 2 ℓ₂ I``.

Fold batches (WalkForward, MRC paths, CPCV) share that ``n × n`` P sparsity
and a constant constraint pattern, so one ``CompiledSolver`` call can replace
a loop of independent solves. Scenario-based MeanRisk is out of scope here:
those problems grow ``A`` with the training length.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from skfolio.optimization import ObjectiveFunction

from skfolio_accelerate.compact import MeanRiskSpec
from skfolio_accelerate.moments import FoldMoments

_CPU_SETTINGS = {"device": "cpu", "verbose": False}


def _scale(spec: MeanRiskSpec) -> float:
    if spec.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
        return float(spec.risk_aversion)
    return 1.0


def _as_bounds(value, n: int, default: float) -> NDArray[np.float64]:
    if value is None:
        return np.full(n, default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(n), dtype=np.float64)


def _bounds(spec: MeanRiskSpec, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _as_bounds(spec.min_weights, n, 0.0), _as_bounds(spec.max_weights, n, 1.0)


def _qp_matrices(
    spec: MeanRiskSpec, n: int
) -> tuple[sp.csr_array, NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Constant A (CSR) plus the dense-P CSR index arrays."""
    budget = float(spec.budget)
    min_w, max_w = _bounds(spec, n)
    a = sp.vstack(
        [
            sp.csr_array(np.ones((1, n))),
            -sp.eye(n, format="csr"),
            sp.eye(n, format="csr"),
        ],
        format="csr",
    )
    b = np.concatenate(
        [np.array([budget], dtype=np.float64), -min_w, max_w]
    )
    p_row_offsets = np.arange(0, n * n + 1, n, dtype=np.int32)
    p_col_indices = np.tile(np.arange(n, dtype=np.int32), n)
    return a, b, p_row_offsets, p_col_indices


def _p_and_q(spec: MeanRiskSpec, moments: FoldMoments) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = int(moments.covariance.shape[0])
    scale = _scale(spec)
    p = 2.0 * scale * np.asarray(moments.covariance, dtype=np.float64)
    diag = np.diag_indices(n)
    p[diag] += 2.0 * float(spec.l2_coef)
    if spec.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
        q = -np.ascontiguousarray(moments.mu, dtype=np.float64)
    else:
        q = np.zeros(n, dtype=np.float64)
    return np.ascontiguousarray(p), q


def _cpu_settings(batch_size: int = 1):
    import moreau

    kwargs = dict(_CPU_SETTINGS, batch_size=int(batch_size))
    try:
        return moreau.Settings(**kwargs)
    except TypeError:
        return moreau.Settings(device="cpu", verbose=False)


def _cones(n: int):
    import moreau

    return moreau.Cones(num_zero_cones=1, num_nonneg_cones=2 * n)


def _status_ok(status) -> bool:
    import moreau

    if status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved):
        return True
    text = getattr(status, "name", str(status)).lower()
    return text in {"solved", "almostsolved", "almost_solved"}


def _assert_solved(info) -> None:
    status = getattr(info, "status", info)
    statuses = status if isinstance(status, (list, tuple)) else [status]
    if not statuses or not all(_status_ok(item) for item in statuses):
        raise RuntimeError(f"Moreau failed: {status}")


def solve_mean_variance_one(spec: MeanRiskSpec, moments: FoldMoments) -> NDArray[np.float64]:
    """Single boxed mean-variance QP with ``moreau.Solver``."""
    import moreau

    n = int(moments.covariance.shape[0])
    p_dense, q = _p_and_q(spec, moments)
    a, b, _, _ = _qp_matrices(spec, n)
    p = sp.csr_array(
        (p_dense.ravel(), np.tile(np.arange(n), n), np.arange(0, n * n + 1, n)),
        shape=(n, n),
    )
    solver = moreau.Solver(p, q, a, b, cones=_cones(n), settings=_cpu_settings(1))
    solution = solver.solve()
    _assert_solved(solver.info)
    return np.asarray(solution.x, dtype=np.float64).reshape(n)


def solve_mean_variance_batch(
    spec: MeanRiskSpec, moments_list: Sequence[FoldMoments]
) -> NDArray[np.float64]:
    """Solve many same-``n`` mean-variance QPs in one ``CompiledSolver`` call.

    Parameters
    ----------
    spec : MeanRiskSpec
        Boxed variance configuration (objective, ℓ₂, bounds, budget).

    moments_list : sequence of FoldMoments
        One covariance (and μ when maximizing utility) per problem.

    Returns
    -------
    weights : ndarray of shape (batch, n_assets)
    """
    import moreau

    if not moments_list:
        return np.empty((0, 0), dtype=np.float64)
    n = int(moments_list[0].covariance.shape[0])
    for moments in moments_list:
        if int(moments.covariance.shape[0]) != n:
            raise ValueError("batched Moreau requires a constant n_assets")
    batch = len(moments_list)
    a, b, p_row_offsets, p_col_indices = _qp_matrices(spec, n)
    settings = _cpu_settings(batch)
    solver = moreau.CompiledSolver(
        n=n,
        m=int(a.shape[0]),
        P_row_offsets=p_row_offsets.tolist(),
        P_col_indices=p_col_indices.tolist(),
        A_row_offsets=a.indptr.tolist(),
        A_col_indices=a.indices.tolist(),
        cones=_cones(n),
        settings=settings,
    )
    p_values = np.empty((batch, n * n), dtype=np.float64)
    qs = np.empty((batch, n), dtype=np.float64)
    for i, moments in enumerate(moments_list):
        p_dense, q = _p_and_q(spec, moments)
        p_values[i] = p_dense.ravel()
        qs[i] = q
    a_values = np.asarray(a.data, dtype=np.float64)
    solver.setup(p_values, a_values)
    bs = np.tile(b, (batch, 1))
    solution = solver.solve(qs, bs)
    _assert_solved(getattr(solver, "info", solution))
    weights = np.asarray(solution.x, dtype=np.float64)
    return weights.reshape(batch, n)


@dataclass(frozen=True)
class FoldSolve:
    """Weights for one compiled CV fold."""

    fold_id: int
    path_id: int
    weights: NDArray[np.float64]


def moments_for_plan(X: NDArray[np.float64], plan, spec: MeanRiskSpec) -> list[tuple[object, FoldMoments]]:
    """Empirical moments per fold, grouped the same way compact CV does."""
    from skfolio_accelerate.moments import path_moment_session

    keep_returns = spec.needs_returns()
    pairs: list[tuple[object, FoldMoments]] = []
    for batch in plan.path_batches():
        session = path_moment_session(
            X,
            batch,
            keep_returns=keep_returns,
            fold_blocks=plan.fold_blocks,
        )
        for fold in batch:
            pairs.append((fold, session.get(fold)))
    return pairs


def batched_weights_for_plan(
    spec: MeanRiskSpec,
    X: NDArray[np.float64],
    plan,
) -> list[FoldSolve]:
    """Batch mean-variance folds that share ``n_assets``."""
    pairs = moments_for_plan(X, plan, spec)
    groups: dict[int, list[tuple[object, FoldMoments]]] = {}
    for fold, moments in pairs:
        n = int(moments.covariance.shape[0])
        groups.setdefault(n, []).append((fold, moments))
    solved: list[FoldSolve] = []
    for _n, items in groups.items():
        weights = solve_mean_variance_batch(spec, [m for _, m in items])
        for (fold, _), row in zip(items, weights, strict=True):
            solved.append(
                FoldSolve(
                    fold_id=int(fold.fold_id),
                    path_id=int(fold.path_id),
                    weights=row,
                )
            )
    solved.sort(key=lambda item: (item.path_id, item.fold_id))
    return solved
