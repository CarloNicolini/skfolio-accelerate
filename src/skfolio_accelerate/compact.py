"""Compact QP/LP engines for MeanRisk, with warm start across adjacent windows.

VARIANCE is a dense n-variable QP solved by OSQP. CVaR is an LP in (w, alpha, u)
solved by Clarabel. Neither goes through CVXPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=32)
def _upper_indices(n: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """CSC-ordered indices for a dense upper triangle.

    Only immutable structural metadata is cached. Numeric matrices depend on
    the data window and must never live in a process-wide cache.
    """
    cols = np.repeat(np.arange(n, dtype=np.intp), np.arange(1, n + 1))
    rows = np.concatenate([np.arange(col + 1, dtype=np.intp) for col in range(n)])
    rows.flags.writeable = False
    cols.flags.writeable = False
    return rows, cols


@lru_cache(maxsize=32)
def _identity(n: int) -> NDArray[np.float64]:
    identity = np.eye(n, dtype=np.float64)
    identity.flags.writeable = False
    return identity


def _upper_data(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, cols = _upper_indices(int(matrix.shape[0]))
    return matrix[rows, cols]


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
        "evar_beta": float(getattr(estimator, "evar_beta", 0.95) or 0.95),
        "cdar_beta": float(getattr(estimator, "cdar_beta", 0.95) or 0.95),
        "edar_beta": float(getattr(estimator, "edar_beta", 0.95) or 0.95),
        "min_acceptable_return": getattr(estimator, "min_acceptable_return", None),
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
        self._p_dense = np.empty((n_assets, n_assets), dtype=np.float64)
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
        eye = _identity(n)
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
        np.multiply(cov, 2.0 * scale, out=self._p_dense)
        diagonal = np.diag_indices(n)
        self._p_dense[diagonal] += 2.0 * self.l2
        q = self._q
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q = -np.ascontiguousarray(moments.mu, dtype=np.float64)
        self._prob.update(Px=_upper_data(self._p_dense), q=q)
        if warm and self._x is not None:
            self._prob.warm_start(x=self._x, y=self._y)
            self.n_warm_starts += 1
        elif self._x is not None:
            # OSQP otherwise carries its previous iterate across update calls.
            self._prob.warm_start(
                x=np.zeros_like(self._x),
                y=np.zeros_like(self._y),
            )
        result = self._prob.solve(raise_error=False)
        status = str(result.info.status).lower()
        if "solved" not in status:
            self._p_dense[diagonal] += 1e-10
            self._prob.update(Px=_upper_data(self._p_dense))
            result = self._prob.solve(raise_error=False)
            status = str(result.info.status).lower()
            if "solved" not in status:
                raise RuntimeError(f"OSQP failed: {result.info.status}")
        self._x = np.asarray(result.x, dtype=np.float64)
        self._y = np.asarray(result.y, dtype=np.float64)
        return self._x.copy()


class CVaRClarabel:
    """Boxed CVaR LP: min alpha + c 1'u s.t. R w + alpha + u >= 0."""

    def __init__(
        self, spec: dict[str, Any], n_assets: int, n_observations: int
    ) -> None:
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


def _scenario_deviations(
    moments: FoldMoments, min_acceptable_return: Any
) -> NDArray[np.float64]:
    """Return skfolio's ``(returns - MAR)`` scenario matrix."""
    returns = np.asarray(moments.returns, dtype=np.float64)
    if min_acceptable_return is None:
        target: float | NDArray[np.float64] = moments.mu
    elif np.isscalar(min_acceptable_return):
        target = float(min_acceptable_return)
    else:
        target = np.asarray(min_acceptable_return, dtype=np.float64).reshape(
            moments.mu.size
        )
    return np.ascontiguousarray(returns - target, dtype=np.float64)


def _rows_to_csc(
    rows: list[list[tuple[int, float]]], n_variables: int
) -> sp.csc_matrix:
    data: list[float] = []
    row_indices: list[int] = []
    columns: list[int] = []
    for row, entries in enumerate(rows):
        for column, value in entries:
            row_indices.append(row)
            columns.append(column)
            data.append(value)
    matrix = sp.csc_matrix(
        (np.asarray(data, dtype=np.float64), (row_indices, columns)),
        shape=(len(rows), n_variables),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _diagonal_quadratic(
    n_variables: int, n_assets: int, l2_coef: float
) -> sp.csc_matrix:
    if l2_coef == 0:
        return sp.csc_matrix((n_variables, n_variables))
    indices = np.arange(n_assets, dtype=np.int32)
    indptr = np.concatenate(
        [
            np.arange(n_assets + 1, dtype=np.int32),
            np.full(n_variables - n_assets, n_assets, dtype=np.int32),
        ]
    )
    return sp.csc_matrix(
        (
            np.full(n_assets, 2.0 * l2_coef, dtype=np.float64),
            indices,
            indptr,
        ),
        shape=(n_variables, n_variables),
    )


class ScenarioClarabel:
    """Direct LP/SOCP/exponential-cone engines for scenario risk measures.

    The row and cone topology depends only on ``(risk, n_assets, n_observations)``.
    Numeric returns and expected returns are rebound for each training window.
    """

    def __init__(
        self, spec: dict[str, Any], n_assets: int, n_observations: int
    ) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_observations = int(n_observations)
        self.min_w = _as_bounds(spec["min_weights"], n_assets, 0.0)
        self.max_w = _as_bounds(spec["max_weights"], n_assets, 1.0)
        self.budget = float(spec["budget"])
        self.l2 = float(spec["l2_coef"])
        self.objective = spec["objective"]
        self.risk_aversion = float(spec["risk_aversion"])
        self.solver: clarabel.DefaultSolver | None = None
        self.n_warm_starts = 0

    def _risk_scale(self) -> float:
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            return self.risk_aversion
        return 1.0

    def _weight_objective(self, q: NDArray[np.float64], moments: FoldMoments) -> None:
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q[: self.n_assets] = -np.asarray(moments.mu, dtype=np.float64)

    def _weight_rows(
        self,
    ) -> tuple[
        list[list[tuple[int, float]]],
        list[float],
        list[list[tuple[int, float]]],
        list[float],
    ]:
        zero_rows = [[(j, 1.0) for j in range(self.n_assets)]]
        zero_rhs = [self.budget]
        nonnegative_rows: list[list[tuple[int, float]]] = []
        nonnegative_rhs: list[float] = []
        for j in range(self.n_assets):
            nonnegative_rows.append([(j, -1.0)])
            nonnegative_rhs.append(-float(self.min_w[j]))
        for j in range(self.n_assets):
            nonnegative_rows.append([(j, 1.0)])
            nonnegative_rhs.append(float(self.max_w[j]))
        return zero_rows, zero_rhs, nonnegative_rows, nonnegative_rhs

    def _linear_problem(
        self, moments: FoldMoments
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        list[Any],
    ]:
        risk = self.spec["risk_measure"]
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        lam = self._risk_scale()
        zero, zero_b, nonneg, nonneg_b = self._weight_rows()

        if risk in {
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        }:
            nv = n + t
            q = np.zeros(nv, dtype=np.float64)
            coefficient = 2.0 if risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION else 1.0
            q[n:] = lam * coefficient / t
            deviations = _scenario_deviations(
                moments, self.spec["min_acceptable_return"]
            )
            for k in range(t):
                nonneg.append([(n + k, -1.0)])
                nonneg_b.append(0.0)
            for k in range(t):
                nonneg.append(
                    [(j, -float(deviations[k, j])) for j in range(n)] + [(n + k, -1.0)]
                )
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.WORST_REALIZATION:
            nv = n + 1
            q = np.zeros(nv, dtype=np.float64)
            q[n] = lam
            for k in range(t):
                nonneg.append(
                    [(j, -float(returns[k, j])) for j in range(n)] + [(n, -1.0)]
                )
                nonneg_b.append(0.0)
        else:
            return self._drawdown_problem(moments)

        self._weight_objective(q, moments)
        rows = zero + nonneg
        b = np.asarray(zero_b + nonneg_b, dtype=np.float64)
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
        ]
        return (
            _diagonal_quadratic(nv, n, self.l2),
            q,
            _rows_to_csc(rows, nv),
            b,
            cones,
        )

    def _drawdown_problem(
        self, moments: FoldMoments
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        list[Any],
    ]:
        risk = self.spec["risk_measure"]
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        d0 = n
        lam = self._risk_scale()
        zero, zero_b, nonneg, nonneg_b = self._weight_rows()
        zero.append([(d0, 1.0)])
        zero_b.append(0.0)
        for k in range(t):
            nonneg.append([(d0 + 1 + k, -1.0)])
            nonneg_b.append(0.0)
            nonneg.append(
                [(j, -float(returns[k, j])) for j in range(n)]
                + [(d0 + k, 1.0), (d0 + 1 + k, -1.0)]
            )
            nonneg_b.append(0.0)

        if risk is RiskMeasure.MAX_DRAWDOWN:
            extra = d0 + t + 1
            nv = extra + 1
            q = np.zeros(nv, dtype=np.float64)
            q[extra] = lam
            for k in range(t):
                nonneg.append([(d0 + 1 + k, 1.0), (extra, -1.0)])
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.AVERAGE_DRAWDOWN:
            nv = d0 + t + 1
            q = np.zeros(nv, dtype=np.float64)
            q[d0 + 1 :] = lam / t
        elif risk is RiskMeasure.CDAR:
            alpha = d0 + t + 1
            z0 = alpha + 1
            nv = z0 + t
            q = np.zeros(nv, dtype=np.float64)
            q[alpha] = lam
            q[z0:] = lam / (t * (1.0 - float(self.spec["cdar_beta"])))
            for k in range(t):
                nonneg.append([(z0 + k, -1.0)])
                nonneg_b.append(0.0)
                nonneg.append(
                    [
                        (d0 + 1 + k, 1.0),
                        (alpha, -1.0),
                        (z0 + k, -1.0),
                    ]
                )
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.EDAR:
            return self._exponential_problem(moments, drawdown=True)
        else:
            raise ValueError(f"Unsupported drawdown risk {risk}")

        self._weight_objective(q, moments)
        rows = zero + nonneg
        b = np.asarray(zero_b + nonneg_b, dtype=np.float64)
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
        ]
        return (
            _diagonal_quadratic(nv, n, self.l2),
            q,
            _rows_to_csc(rows, nv),
            b,
            cones,
        )

    def _semi_deviation_problem(
        self, moments: FoldMoments
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        list[Any],
    ]:
        deviations = _scenario_deviations(moments, self.spec["min_acceptable_return"])
        t, n = deviations.shape
        u0 = n
        radius = n + t
        nv = radius + 1
        q = np.zeros(nv, dtype=np.float64)
        q[radius] = self._risk_scale() / np.sqrt(t - 1)
        self._weight_objective(q, moments)
        zero, zero_b, nonneg, nonneg_b = self._weight_rows()
        for k in range(t):
            nonneg.append([(u0 + k, -1.0)])
            nonneg_b.append(0.0)
            nonneg.append(
                [(j, -float(deviations[k, j])) for j in range(n)] + [(u0 + k, -1.0)]
            )
            nonneg_b.append(0.0)

        rows = zero + nonneg
        b_values = zero_b + nonneg_b
        soc_rows = [[(radius, -1.0)]] + [[(u0 + k, -1.0)] for k in range(t)]
        rows += soc_rows
        b_values += [0.0] * (t + 1)
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
            clarabel.SecondOrderConeT(t + 1),
        ]
        return (
            _diagonal_quadratic(nv, n, self.l2),
            q,
            _rows_to_csc(rows, nv),
            np.asarray(b_values, dtype=np.float64),
            cones,
        )

    def _semi_variance_problem(
        self, moments: FoldMoments
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        list[Any],
    ]:
        deviations = _scenario_deviations(moments, self.spec["min_acceptable_return"])
        t, n = deviations.shape
        nv = n + t
        q = np.zeros(nv, dtype=np.float64)
        self._weight_objective(q, moments)
        zero, zero_b, nonneg, nonneg_b = self._weight_rows()
        for k in range(t):
            nonneg.append([(n + k, -1.0)])
            nonneg_b.append(0.0)
            nonneg.append(
                [(j, -float(deviations[k, j])) for j in range(n)] + [(n + k, -1.0)]
            )
            nonneg_b.append(0.0)

        diagonal = np.concatenate(
            [
                np.full(n, 2.0 * self.l2),
                np.full(t, 2.0 * self._risk_scale() / (t - 1)),
            ]
        )
        rows = zero + nonneg
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
        ]
        return (
            sp.diags(diagonal, format="csc"),
            q,
            _rows_to_csc(rows, nv),
            np.asarray(zero_b + nonneg_b, dtype=np.float64),
            cones,
        )

    def _exponential_problem(
        self, moments: FoldMoments, *, drawdown: bool
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        list[Any],
    ]:
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        zero, zero_b, nonneg, nonneg_b = self._weight_rows()
        if drawdown:
            d0 = n
            zero.append([(d0, 1.0)])
            zero_b.append(0.0)
            for k in range(t):
                nonneg.append([(d0 + 1 + k, -1.0)])
                nonneg_b.append(0.0)
                nonneg.append(
                    [(j, -float(returns[k, j])) for j in range(n)]
                    + [(d0 + k, 1.0), (d0 + 1 + k, -1.0)]
                )
                nonneg_b.append(0.0)
            x = d0 + t + 1
            beta = float(self.spec["edar_beta"])
        else:
            d0 = -1
            x = n
            beta = float(self.spec["evar_beta"])
        y = x + 1
        z0 = y + 1
        nv = z0 + t
        q = np.zeros(nv, dtype=np.float64)
        q[x] = self._risk_scale()
        q[y] = self._risk_scale() * np.log(1.0 / (t * (1.0 - beta)))
        self._weight_objective(q, moments)
        nonneg.append([(z0 + k, 1.0) for k in range(t)] + [(y, -1.0)])
        nonneg_b.append(0.0)

        rows = zero + nonneg
        b_values = zero_b + nonneg_b
        exponential_rows: list[list[tuple[int, float]]] = []
        for k in range(t):
            if drawdown:
                # Slack is (drawdown - x, y, z).
                exponential_rows.append([(d0 + 1 + k, -1.0), (x, 1.0)])
            else:
                # Slack is (-return - x, y, z).
                exponential_rows.append(
                    [(j, float(returns[k, j])) for j in range(n)] + [(x, 1.0)]
                )
            exponential_rows.append([(y, -1.0)])
            exponential_rows.append([(z0 + k, -1.0)])
        rows += exponential_rows
        b_values += [0.0] * (3 * t)
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
            *[clarabel.ExponentialConeT() for _ in range(t)],
        ]
        return (
            _diagonal_quadratic(nv, n, self.l2),
            q,
            _rows_to_csc(rows, nv),
            np.asarray(b_values, dtype=np.float64),
            cones,
        )

    def _problem(self, moments: FoldMoments):
        risk = self.spec["risk_measure"]
        if risk is RiskMeasure.SEMI_VARIANCE:
            return self._semi_variance_problem(moments)
        if risk is RiskMeasure.SEMI_DEVIATION:
            return self._semi_deviation_problem(moments)
        if risk is RiskMeasure.EVAR:
            return self._exponential_problem(moments, drawdown=False)
        return self._linear_problem(moments)

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations)
        if t != self.n_observations:
            self.n_observations = t
            self.solver = None
        P, q, A, b, cones = self._problem(moments)
        settings = _clarabel_settings()
        if self.solver is None:
            self.solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        else:
            try:
                allowed = (
                    not hasattr(self.solver, "is_data_update_allowed")
                    or self.solver.is_data_update_allowed()
                )
                if not allowed:
                    raise RuntimeError("Clarabel data update is unavailable")
                self.solver.update(q=q, A=A, b=b)
                if warm:
                    self.n_warm_starts += 1
            except Exception:
                self.solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = self.solver.solve()
        status = str(solution.status).lower()
        if "solved" not in status:
            raise RuntimeError(
                f"Clarabel {self.spec['risk_measure'].name} failed: {solution.status}"
            )
        return np.asarray(solution.x[: self.n_assets], dtype=np.float64)


class SemiVarianceOSQP:
    """Exact downside-square QP with fixed scenario topology."""

    def __init__(
        self, spec: dict[str, Any], n_assets: int, n_observations: int
    ) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_observations = int(n_observations)
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

    def _matrices(
        self, moments: FoldMoments
    ) -> tuple[
        sp.csc_matrix,
        NDArray[np.float64],
        sp.csc_matrix,
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        deviations = _scenario_deviations(moments, self.spec["min_acceptable_return"])
        t, n = deviations.shape
        nv = n + t
        lam = (
            self.risk_aversion
            if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY
            else 1.0
        )
        p_diag = np.concatenate(
            [
                np.full(n, 2.0 * self.l2),
                np.full(t, 2.0 * lam / (t - 1)),
            ]
        )
        P = sp.diags(p_diag, format="csc")
        q = np.zeros(nv, dtype=np.float64)
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q[:n] = -np.asarray(moments.mu, dtype=np.float64)

        rows: list[list[tuple[int, float]]] = [[(j, 1.0) for j in range(n)]]
        lower = [self.budget]
        upper = [self.budget]
        for j in range(n):
            rows.append([(j, 1.0)])
            lower.append(float(self.min_w[j]))
            upper.append(float(self.max_w[j]))
        for k in range(t):
            rows.append([(n + k, 1.0)])
            lower.append(0.0)
            upper.append(np.inf)
        for k in range(t):
            rows.append(
                [(j, float(deviations[k, j])) for j in range(n)] + [(n + k, 1.0)]
            )
            lower.append(0.0)
            upper.append(np.inf)
        return (
            P,
            q,
            _rows_to_csc(rows, nv),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations)
        if t != self.n_observations:
            self.n_observations = t
            self._prob = None
            self._x = self._y = None
        P, q, A, lower, upper = self._matrices(moments)
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(
                P=P,
                q=q,
                A=A,
                l=lower,
                u=upper,
                verbose=False,
                warm_starting=True,
                polishing=False,
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=10000,
            )
        else:
            self._prob.update(q=q, Ax=A.data, l=lower, u=upper)
            if warm and self._x is not None:
                self._prob.warm_start(x=self._x, y=self._y)
                self.n_warm_starts += 1
            elif self._x is not None:
                self._prob.warm_start(
                    x=np.zeros_like(self._x), y=np.zeros_like(self._y)
                )
        result = self._prob.solve(raise_error=False)
        if "solved" not in str(result.info.status).lower():
            raise RuntimeError(f"OSQP SEMI_VARIANCE failed: {result.info.status}")
        self._x = np.asarray(result.x, dtype=np.float64)
        self._y = np.asarray(result.y, dtype=np.float64)
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
    if risk is RiskMeasure.SEMI_VARIANCE:
        if n_observations is None:
            raise ValueError("semi-variance engine requires n_observations")
        return ScenarioClarabel(spec, n_assets, n_observations)
    if risk in {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
        RiskMeasure.EVAR,
        RiskMeasure.EDAR,
    }:
        if n_observations is None:
            raise ValueError(f"{risk.name} engine requires n_observations")
        return ScenarioClarabel(spec, n_assets, n_observations)
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
        if risk is not RiskMeasure.VARIANCE:
            need_new = need_new or n_observations != self.n_observations
        if need_new:
            self.engine = make_compact_engine(
                self.spec, n_assets=n_assets, n_observations=n_observations
            )
            self.n_assets = n_assets
            self.n_observations = n_observations
        return self.engine
