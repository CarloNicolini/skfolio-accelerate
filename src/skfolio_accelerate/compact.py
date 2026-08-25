"""Compact QP/LP engines for MeanRisk, with warm start across adjacent windows.

VARIANCE is a dense n-variable QP solved by OSQP. CVaR is an LP in (w, alpha, u)
solved by Clarabel. Neither goes through CVXPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import clarabel
import numpy as np
import osqp
import scipy.sparse as sp
from numpy.typing import NDArray

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.moments import FoldMoments


def _as_bounds(value: Any, n: int, default: float) -> NDArray[np.float64]:
    if value is None:
        return np.full(n, default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(n), dtype=np.float64)


def _upper_csc(matrix: NDArray[np.float64]) -> sp.csc_matrix:
    """Upper-triangular CSC that keeps explicit zeros so OSQP updates stay valid."""
    n = int(matrix.shape[0])
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for col in range(n):
        column = matrix[: col + 1, col]
        data.extend(column.tolist())
        indices.extend(range(col + 1))
        indptr.append(len(data))
    return sp.csc_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n, n),
    )


def _upper_data(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    n = int(matrix.shape[0])
    return np.concatenate([matrix[: col + 1, col] for col in range(n)])


def _clarabel_settings() -> clarabel.DefaultSettings:
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    if hasattr(settings, "presolve_enable"):
        settings.presolve_enable = False
    if hasattr(settings, "chordal_decomposition_enable"):
        settings.chordal_decomposition_enable = False
    if hasattr(settings, "input_sparse_dropzeros"):
        settings.input_sparse_dropzeros = False
    if hasattr(settings, "max_threads"):
        settings.max_threads = 1
    if hasattr(settings, "tol_gap_abs"):
        settings.tol_gap_abs = 1e-9
    if hasattr(settings, "tol_gap_rel"):
        settings.tol_gap_rel = 1e-9
    return settings


def estimator_spec(estimator) -> dict[str, Any]:
    return {
        "risk_measure": getattr(estimator, "risk_measure", RiskMeasure.VARIANCE),
        "objective": getattr(
            estimator, "objective_function", ObjectiveFunction.MINIMIZE_RISK
        ),
        "l2_coef": float(getattr(estimator, "l2_coef", 0.0) or 0.0),
        "risk_aversion": float(getattr(estimator, "risk_aversion", 1.0) or 1.0),
        "cvar_beta": float(getattr(estimator, "cvar_beta", 0.95) or 0.95),
        "min_weights": getattr(estimator, "min_weights", 0.0),
        "max_weights": getattr(estimator, "max_weights", 1.0),
        "budget": float(getattr(estimator, "budget", 1.0) or 1.0),
    }


class MinVarianceOSQP:
    """Long-only (boxed) mean-variance QP with OSQP warm starts."""

    def __init__(self, spec: dict[str, Any], n_assets: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = _as_bounds(spec["min_weights"], n_assets, 0.0)
        self.max_w = _as_bounds(spec["max_weights"], n_assets, 1.0)
        self.budget = float(spec["budget"])
        self.l2 = float(spec["l2_coef"])
        self.objective = spec["objective"]
        self.risk_aversion = float(spec["risk_aversion"])
        self._prob: osqp.OSQP | None = None
        self._x: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None
        self.n_warm_starts = 0
        self._build()

    def _build(self) -> None:
        n = self.n_assets
        self._A = sp.vstack(
            [sp.csr_matrix(np.ones((1, n))), sp.eye(n, format="csr")]
        ).tocsc()
        self._l = np.concatenate([[self.budget], self.min_w])
        self._u = np.concatenate([[self.budget], self.max_w])
        self._q = np.zeros(n, dtype=np.float64)
        eye = np.eye(n)
        # Keep the full upper-triangular pattern so later dense Σ updates match nnz.
        p0 = _upper_csc(2.0 * (eye + self.l2 * eye + 1e-16 * np.ones((n, n))))
        self._prob = osqp.OSQP()
        self._prob.setup(
            P=p0,
            q=self._q,
            A=self._A,
            l=self._l,
            u=self._u,
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=4000,
        )

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        n = self.n_assets
        cov = np.asarray(moments.covariance, dtype=np.float64)
        if cov.shape != (n, n):
            raise ValueError(f"covariance shape {cov.shape} != {(n, n)}")
        scale = (
            float(self.risk_aversion)
            if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY
            else 1.0
        )
        p_dense = 2.0 * (scale * cov + self.l2 * np.eye(n))
        q = self._q
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q = -np.ascontiguousarray(moments.mu, dtype=np.float64)
        self._prob.update(Px=_upper_data(p_dense), q=q)
        if warm and self._x is not None:
            self._prob.warm_start(x=self._x, y=self._y)
            self.n_warm_starts += 1
        result = self._prob.solve(raise_error=False)
        status = str(result.info.status).lower()
        if "solved" not in status:
            p_dense = p_dense + 1e-10 * np.eye(n)
            self._prob.update(Px=_upper_data(p_dense))
            result = self._prob.solve(raise_error=False)
            status = str(result.info.status).lower()
            if "solved" not in status:
                raise RuntimeError(f"OSQP failed: {result.info.status}")
        self._x = np.asarray(result.x, dtype=np.float64)
        self._y = np.asarray(result.y, dtype=np.float64)
        return self._x.copy()


class CVaRClarabel:
    """Boxed CVaR LP: min alpha + c 1'u s.t. R w + alpha + u >= 0."""

    def __init__(self, spec: dict[str, Any], n_assets: int, n_observations: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_observations = int(n_observations)
        self.min_w = _as_bounds(spec["min_weights"], n_assets, 0.0)
        self.max_w = _as_bounds(spec["max_weights"], n_assets, 1.0)
        self.budget = float(spec["budget"])
        self.l2 = float(spec["l2_coef"])
        self.beta = float(spec["cvar_beta"])
        self.objective = spec["objective"]
        self.risk_aversion = float(spec["risk_aversion"])
        self.solver: clarabel.DefaultSolver | None = None
        self._A: sp.csc_matrix | None = None
        self._q: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None
        self._x: NDArray[np.float64] | None = None
        self.n_warm_starts = 0
        self._build_pattern()

    def _c(self) -> float:
        return 1.0 / (self.n_observations * (1.0 - self.beta))

    def _build_pattern(self) -> None:
        n = self.n_assets
        t = self.n_observations
        nv = n + 1 + t
        # Columns 0..n-1 (w), n (alpha), n+1..n+t (u)
        # Rows:
        #  0: 1'w = budget                         zero
        #  1..n: -w <= -min_w  (w >= min_w)        nonneg
        #  n+1..2n: w <= max_w                     nonneg
        #  2n+1..2n+t: -u <= 0                     nonneg
        #  2n+t+1..2n+2t: -R w - alpha - u <= 0    nonneg
        data: list[float] = []
        rows: list[int] = []
        cols: list[int] = []

        def put(row: int, col: int, value: float) -> None:
            rows.append(row)
            cols.append(col)
            data.append(value)

        for j in range(n):
            put(0, j, 1.0)
            put(1 + j, j, -1.0)
            put(1 + n + j, j, 1.0)
            for k in range(t):
                put(1 + 2 * n + t + k, j, 0.0)
        for k in range(t):
            put(1 + 2 * n + t + k, n, -1.0)
        for k in range(t):
            put(1 + 2 * n + k, n + 1 + k, -1.0)
            put(1 + 2 * n + t + k, n + 1 + k, -1.0)
        n_cons = 1 + 2 * n + 2 * t
        self._A = sp.csc_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(n_cons, nv),
        )
        self._A.sum_duplicates()
        self._A.sort_indices()
        c = self._c()
        q = np.zeros(nv, dtype=np.float64)
        lam = (
            float(self.risk_aversion)
            if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY
            else 1.0
        )
        q[n] = lam
        q[n + 1 :] = lam * c
        self._q = q
        b = np.zeros(n_cons, dtype=np.float64)
        b[0] = self.budget
        b[1 : 1 + n] = -self.min_w
        b[1 + n : 1 + 2 * n] = self.max_w
        self._b = b
        self._r_slices: list[slice] = []
        for j in range(n):
            start = int(self._A.indptr[j])
            stop = int(self._A.indptr[j + 1])
            self._r_slices.append(slice(stop - t, stop))
        p_data = np.zeros(n, dtype=np.float64)
        if self.l2 != 0.0:
            p_data[:] = 2.0 * self.l2
        p_idx = np.arange(n, dtype=np.int32)
        p_ptr = np.arange(n + 1, dtype=np.int32)
        p_ptr_full = np.concatenate([p_ptr, np.full(nv - n, n, dtype=np.int32)])
        if self.l2 == 0.0:
            self._P = sp.csc_matrix((nv, nv))
        else:
            self._P = sp.csc_matrix((p_data, p_idx, p_ptr_full), shape=(nv, nv))
        self._cones = [
            clarabel.ZeroConeT(1),
            clarabel.NonnegativeConeT(n_cons - 1),
        ]
        self.solver = None

    def _bind_R(self, returns: NDArray[np.float64]) -> None:
        r = np.ascontiguousarray(returns, dtype=np.float64)
        t, n = r.shape
        if t != self.n_observations or n != self.n_assets:
            raise ValueError("CVaR returns shape mismatch")
        assert self._A is not None
        for j in range(n):
            self._A.data[self._r_slices[j]] = -r[:, j]

    def _bind_q(self, moments: FoldMoments) -> None:
        assert self._q is not None
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            self._q[: self.n_assets] = -np.ascontiguousarray(
                moments.mu, dtype=np.float64
            )

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations or moments.returns.shape[0])
        if t != self.n_observations:
            self.n_observations = t
            self._build_pattern()
        self._bind_R(moments.returns)
        self._bind_q(moments)
        assert self._A is not None and self._q is not None and self._b is not None
        settings = _clarabel_settings()
        if self.solver is None:
            self.solver = clarabel.DefaultSolver(
                self._P, self._q, self._A, self._b, self._cones, settings
            )
        else:
            try:
                allowed = True
                if hasattr(self.solver, "is_data_update_allowed"):
                    allowed = bool(self.solver.is_data_update_allowed())
                if not allowed:
                    self.solver = clarabel.DefaultSolver(
                        self._P, self._q, self._A, self._b, self._cones, settings
                    )
                else:
                    if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
                        self.solver.update(q=self._q, A=self._A)
                    else:
                        self.solver.update(A=self._A)
                    if warm:
                        self.n_warm_starts += 1
            except Exception:
                self.solver = clarabel.DefaultSolver(
                    self._P, self._q, self._A, self._b, self._cones, settings
                )
        solution = self.solver.solve()
        status = str(solution.status).lower()
        if "solved" not in status:
            raise RuntimeError(f"Clarabel CVaR failed: {solution.status}")
        self._x = np.asarray(solution.x, dtype=np.float64)
        return self._x[: self.n_assets].copy()


def make_compact_engine(
    spec: dict[str, Any], *, n_assets: int, n_observations: int | None
):
    risk = spec["risk_measure"]
    if risk is RiskMeasure.VARIANCE:
        return MinVarianceOSQP(spec, n_assets)
    if risk is RiskMeasure.CVAR:
        if n_observations is None:
            raise ValueError("CVaR engine requires n_observations")
        return CVaRClarabel(spec, n_assets, n_observations)
    raise ValueError(f"Unsupported risk_measure {risk}")


@dataclass
class EngineCache:
    """Reuse one compact engine while (n_assets, T) stay constant."""

    spec: dict[str, Any]
    engine: Any = None
    n_assets: int = -1
    n_observations: int | None = None

    def get(self, n_assets: int, n_observations: int | None):
        risk = self.spec["risk_measure"]
        need_new = self.engine is None or n_assets != self.n_assets
        if risk is RiskMeasure.CVAR:
            need_new = need_new or n_observations != self.n_observations
        if need_new:
            self.engine = make_compact_engine(
                self.spec, n_assets=n_assets, n_observations=n_observations
            )
            self.n_assets = n_assets
            self.n_observations = n_observations
        return self.engine
