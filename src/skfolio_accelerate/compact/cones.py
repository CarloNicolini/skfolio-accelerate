"""Clarabel SOCP / LP / exponential-cone engines with a shared boxed graph."""

from __future__ import annotations

from typing import Any

import clarabel
import numpy as np
import osqp
import scipy.sparse as sp
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compact._util import (
    as_bounds,
    clarabel_try_update,
    diagonal_quadratic,
    identity,
    rows_to_csc,
    MeanRiskSpec,
    upper_csc,
    upper_data,
)
from skfolio_accelerate.moments import FoldMoments


def scenario_deviations(moments: FoldMoments, min_acceptable_return) -> NDArray[np.float64]:
    returns = np.asarray(moments.returns, dtype=np.float64)
    if min_acceptable_return is None:
        target = moments.mu
    elif np.isscalar(min_acceptable_return):
        target = float(min_acceptable_return)
    else:
        target = np.asarray(min_acceptable_return, dtype=np.float64).reshape(moments.mu.size)
    return np.ascontiguousarray(returns - target, dtype=np.float64)


class ClarabelEngine:
    """Persistent Clarabel workspace: topology stays, fold data is rebound."""

    def __init__(self, spec: MeanRiskSpec, n_assets: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.l2 = float(spec.l2_coef)
        self.objective = spec.objective
        self.solver = None
        self._P = self._q = self._A = self._b = self._cones = None
        self.n_warm_starts = 0

    def weight_objective(self, q: NDArray[np.float64], moments: FoldMoments) -> None:
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q[: self.n_assets] = -np.asarray(moments.mu, dtype=np.float64)

    def weight_rows(self):
        zero = [[(j, 1.0) for j in range(self.n_assets)]]
        zero_b = [self.budget]
        nonneg, nonneg_b = [], []
        for j in range(self.n_assets):
            nonneg.append([(j, -1.0)])
            nonneg_b.append(-float(self.min_w[j]))
        for j in range(self.n_assets):
            nonneg.append([(j, 1.0)])
            nonneg_b.append(float(self.max_w[j]))
        return zero, zero_b, nonneg, nonneg_b

    def run(self, update: dict, *, warm: bool) -> NDArray[np.float64]:
        self.solver, updated = clarabel_try_update(
            self.solver, self._P, self._q, self._A, self._b, self._cones, update=update
        )
        if updated and warm:
            self.n_warm_starts += 1
        solution = self.solver.solve()
        status = str(solution.status).lower()
        if "solved" not in status:
            raise RuntimeError(
                f"Clarabel {self.spec.risk_measure.name} failed: {solution.status}"
            )
        return np.asarray(solution.x[: self.n_assets], dtype=np.float64)


class StandardDeviationClarabel(ClarabelEngine):
    def __init__(self, spec: MeanRiskSpec, n_assets: int) -> None:
        super().__init__(spec, n_assets)
        n = self.n_assets
        soc_start = 1 + 2 * n
        n_variables, n_constraints = n + 1, soc_start + n + 1
        data, rows, cols = [], [], []

        def put(row, col, value):
            rows.append(row)
            cols.append(col)
            data.append(value)

        for column in range(n):
            put(0, column, 1.0)
            put(1 + column, column, -1.0)
            put(1 + n + column, column, 1.0)
            for component in range(n):
                put(soc_start + 1 + component, column, 0.0)
        put(soc_start, n, -1.0)
        self._A = sp.csc_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(n_constraints, n_variables),
        )
        self._A.sum_duplicates()
        self._A.sort_indices()
        self._factor_slices = [
            slice(int(self._A.indptr[c + 1]) - n, int(self._A.indptr[c + 1]))
            for c in range(n)
        ]
        self._q = np.zeros(n_variables, dtype=np.float64)
        self._q[n] = spec.risk_scale()
        self._b = np.zeros(n_constraints, dtype=np.float64)
        self._b[0] = self.budget
        self._b[1 : 1 + n] = -self.min_w
        self._b[1 + n : 1 + 2 * n] = self.max_w
        self._P = sp.diags(
            np.concatenate([np.full(n, 2.0 * self.l2, dtype=np.float64), np.zeros(1)]),
            format="csc",
        )
        self._cones = [
            clarabel.ZeroConeT(1),
            clarabel.NonnegativeConeT(2 * n),
            clarabel.SecondOrderConeT(n + 1),
        ]

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        cov = np.asarray(moments.covariance, dtype=np.float64)
        expected = (self.n_assets, self.n_assets)
        if cov.shape != expected:
            raise ValueError(f"covariance shape {cov.shape} != {expected}")
        try:
            factor = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
            floor = np.finfo(np.float64).eps * max(1.0, float(values[-1]))
            factor = vectors * np.sqrt(np.maximum(values, floor))
        for column, target in enumerate(self._factor_slices):
            self._A.data[target] = -factor[column]
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            self._q[: self.n_assets] = -np.asarray(moments.mu, dtype=np.float64)
        return self.run({"q": self._q, "A": self._A}, warm=warm)


class ScenarioClarabel(ClarabelEngine):
    """LP / SOCP / exponential cones; scenario coefficients rebound per fold."""

    def __init__(self, spec: MeanRiskSpec, n_assets: int, n_observations: int) -> None:
        super().__init__(spec, n_assets)
        self.n_observations = int(n_observations)
        self._scenario_slots = None

    def _pack(self, nv, q, zero, zero_b, nonneg, nonneg_b, extra_rows=None, extra_b=None, extra_cones=None, P=None):
        self.weight_objective(q, self._moments)
        rows = zero + nonneg + (extra_rows or [])
        b = np.asarray(zero_b + nonneg_b + (extra_b or []), dtype=np.float64)
        cones: list[Any] = [
            clarabel.ZeroConeT(len(zero)),
            clarabel.NonnegativeConeT(len(nonneg)),
            *(extra_cones or []),
        ]
        return P if P is not None else diagonal_quadratic(nv, self.n_assets, self.l2), q, rows_to_csc(rows, nv), b, cones

    def _linear_problem(self, moments: FoldMoments):
        risk = self.spec.risk_measure
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        lam = self.spec.risk_scale()
        zero, zero_b, nonneg, nonneg_b = self.weight_rows()
        if risk in {RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT}:
            nv = n + t
            q = np.zeros(nv, dtype=np.float64)
            q[n:] = lam * (2.0 if risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION else 1.0) / t
            deviations = scenario_deviations(moments, self.spec.min_acceptable_return)
            for k in range(t):
                nonneg.append([(n + k, -1.0)])
                nonneg_b.append(0.0)
            for k in range(t):
                nonneg.append([(j, -float(deviations[k, j])) for j in range(n)] + [(n + k, -1.0)])
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.WORST_REALIZATION:
            nv = n + 1
            q = np.zeros(nv, dtype=np.float64)
            q[n] = lam
            for k in range(t):
                nonneg.append([(j, -float(returns[k, j])) for j in range(n)] + [(n, -1.0)])
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.CVAR:
            nv = n + 1 + t
            q = np.zeros(nv, dtype=np.float64)
            q[n] = lam
            q[n + 1 :] = lam / (t * (1.0 - float(self.spec.cvar_beta)))
            for k in range(t):
                nonneg.append([(n + 1 + k, -1.0)])
                nonneg_b.append(0.0)
            for k in range(t):
                nonneg.append(
                    [(j, -float(returns[k, j])) for j in range(n)]
                    + [(n, -1.0), (n + 1 + k, -1.0)]
                )
                nonneg_b.append(0.0)
        else:
            return self._drawdown_problem(moments)
        return self._pack(nv, q, zero, zero_b, nonneg, nonneg_b)

    def _drawdown_problem(self, moments: FoldMoments):
        risk = self.spec.risk_measure
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        d0, lam = n, self.spec.risk_scale()
        zero, zero_b, nonneg, nonneg_b = self.weight_rows()
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
            alpha, z0 = d0 + t + 1, d0 + t + 2
            nv = z0 + t
            q = np.zeros(nv, dtype=np.float64)
            q[alpha] = lam
            q[z0:] = lam / (t * (1.0 - float(self.spec.cdar_beta)))
            for k in range(t):
                nonneg.append([(z0 + k, -1.0)])
                nonneg_b.append(0.0)
                nonneg.append([(d0 + 1 + k, 1.0), (alpha, -1.0), (z0 + k, -1.0)])
                nonneg_b.append(0.0)
        elif risk is RiskMeasure.EDAR:
            return self._exponential_problem(moments, drawdown=True)
        else:
            raise ValueError(f"Unsupported drawdown risk {risk}")
        return self._pack(nv, q, zero, zero_b, nonneg, nonneg_b)

    def _semi_deviation_problem(self, moments: FoldMoments):
        deviations = scenario_deviations(moments, self.spec.min_acceptable_return)
        t, n = deviations.shape
        u0, radius, nv = n, n + t, n + t + 1
        q = np.zeros(nv, dtype=np.float64)
        q[radius] = self.spec.risk_scale() / np.sqrt(t - 1)
        zero, zero_b, nonneg, nonneg_b = self.weight_rows()
        for k in range(t):
            nonneg.append([(u0 + k, -1.0)])
            nonneg_b.append(0.0)
            nonneg.append([(j, -float(deviations[k, j])) for j in range(n)] + [(u0 + k, -1.0)])
            nonneg_b.append(0.0)
        soc = [[(radius, -1.0)]] + [[(u0 + k, -1.0)] for k in range(t)]
        return self._pack(
            nv, q, zero, zero_b, nonneg, nonneg_b,
            extra_rows=soc, extra_b=[0.0] * (t + 1),
            extra_cones=[clarabel.SecondOrderConeT(t + 1)],
        )

    def _semi_variance_problem(self, moments: FoldMoments):
        deviations = scenario_deviations(moments, self.spec.min_acceptable_return)
        t, n = deviations.shape
        nv = n + t
        q = np.zeros(nv, dtype=np.float64)
        zero, zero_b, nonneg, nonneg_b = self.weight_rows()
        for k in range(t):
            nonneg.append([(n + k, -1.0)])
            nonneg_b.append(0.0)
            nonneg.append([(j, -float(deviations[k, j])) for j in range(n)] + [(n + k, -1.0)])
            nonneg_b.append(0.0)
        diagonal = np.concatenate(
            [np.full(n, 2.0 * self.l2), np.full(t, 2.0 * self.spec.risk_scale() / (t - 1))]
        )
        return self._pack(
            nv, q, zero, zero_b, nonneg, nonneg_b, P=sp.diags(diagonal, format="csc")
        )

    def _exponential_problem(self, moments: FoldMoments, *, drawdown: bool):
        returns = np.asarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        zero, zero_b, nonneg, nonneg_b = self.weight_rows()
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
            x, beta = d0 + t + 1, float(self.spec.edar_beta)
        else:
            d0, x, beta = -1, n, float(self.spec.evar_beta)
        y, z0, nv = x + 1, x + 2, x + 2 + t
        q = np.zeros(nv, dtype=np.float64)
        q[x] = self.spec.risk_scale()
        q[y] = self.spec.risk_scale() * np.log(1.0 / (t * (1.0 - beta)))
        nonneg.append([(z0 + k, 1.0) for k in range(t)] + [(y, -1.0)])
        nonneg_b.append(0.0)
        expo = []
        for k in range(t):
            if drawdown:
                expo.append([(d0 + 1 + k, -1.0), (x, 1.0)])
            else:
                expo.append([(j, float(returns[k, j])) for j in range(n)] + [(x, 1.0)])
            expo.append([(y, -1.0)])
            expo.append([(z0 + k, -1.0)])
        return self._pack(
            nv, q, zero, zero_b, nonneg, nonneg_b,
            extra_rows=expo, extra_b=[0.0] * (3 * t),
            extra_cones=[clarabel.ExponentialConeT() for _ in range(t)],
        )

    def _problem(self, moments: FoldMoments):
        risk = self.spec.risk_measure
        if risk is RiskMeasure.SEMI_VARIANCE:
            return self._semi_variance_problem(moments)
        if risk is RiskMeasure.SEMI_DEVIATION:
            return self._semi_deviation_problem(moments)
        if risk is RiskMeasure.EVAR:
            return self._exponential_problem(moments, drawdown=False)
        return self._linear_problem(moments)

    def _scenario_binding(self, moments: FoldMoments):
        risk, n, t = self.spec.risk_measure, self.n_assets, self.n_observations
        if risk in {RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT}:
            return 1 + 2 * n + t + np.arange(t, dtype=np.intp), -scenario_deviations(
                moments, self.spec.min_acceptable_return
            )
        if risk is RiskMeasure.WORST_REALIZATION:
            return 1 + 2 * n + np.arange(t, dtype=np.intp), -np.asarray(moments.returns, dtype=np.float64)
        if risk is RiskMeasure.CVAR:
            return 1 + 2 * n + t + np.arange(t, dtype=np.intp), -np.asarray(
                moments.returns, dtype=np.float64
            )
        if risk in {RiskMeasure.SEMI_VARIANCE, RiskMeasure.SEMI_DEVIATION}:
            return 1 + 2 * n + 2 * np.arange(t, dtype=np.intp) + 1, -scenario_deviations(
                moments, self.spec.min_acceptable_return
            )
        if risk is RiskMeasure.EVAR:
            return 2 + 2 * n + 3 * np.arange(t, dtype=np.intp), np.asarray(
                moments.returns, dtype=np.float64
            )
        return 2 + 2 * n + 2 * np.arange(t, dtype=np.intp) + 1, -np.asarray(
            moments.returns, dtype=np.float64
        )

    def _compile(self, moments: FoldMoments) -> None:
        self._moments = moments
        self._P, self._q, self._A, self._b, self._cones = self._problem(moments)
        rows, _ = self._scenario_binding(moments)
        slots = np.empty((self.n_observations, self.n_assets), dtype=np.intp)
        for column in range(self.n_assets):
            start, stop = int(self._A.indptr[column]), int(self._A.indptr[column + 1])
            column_rows = self._A.indices[start:stop]
            positions = np.searchsorted(column_rows, rows)
            if np.any(positions == column_rows.size) or np.any(column_rows[positions] != rows):
                raise RuntimeError("ScenarioClarabel sparse pattern is incomplete")
            slots[:, column] = start + positions
        self._scenario_slots = slots

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations)
        if t != self.n_observations:
            self.n_observations = t
            self.solver = None
            self._P = self._q = self._A = self._b = self._cones = self._scenario_slots = None
        self._moments = moments
        if self._A is None:
            self._compile(moments)
        _, values = self._scenario_binding(moments)
        self._A.data[self._scenario_slots] = values
        self._q[: self.n_assets] = 0.0
        self.weight_objective(self._q, moments)
        return self.run({"q": self._q, "A": self._A}, warm=warm)


class MaxReturnBox:
    def __init__(self, spec: MeanRiskSpec, n_assets: int) -> None:
        self.n_assets = int(n_assets)
        self.min_w = as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.l2 = float(spec.l2_coef)
        self.n_warm_starts = 0

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        del warm
        mu = np.asarray(moments.mu, dtype=np.float64)
        if mu.shape != (self.n_assets,):
            raise ValueError(f"expected return shape {mu.shape} != {(self.n_assets,)}")
        min_budget, max_budget = float(self.min_w.sum()), float(self.max_w.sum())
        tol = 1e-12 * max(1.0, abs(self.budget))
        if self.budget < min_budget - tol or self.budget > max_budget + tol:
            raise ValueError("budget is infeasible for the weight bounds")
        weights = self.min_w.copy()
        if self.l2 == 0.0:
            remaining = self.budget - min_budget
            for index in np.argsort(-mu, kind="stable"):
                addition = min(remaining, self.max_w[index] - self.min_w[index])
                weights[index] += addition
                remaining -= addition
                if remaining <= tol:
                    break
            return weights
        lower = float(np.min(mu - 2.0 * self.l2 * self.max_w))
        upper = float(np.max(mu - 2.0 * self.l2 * self.min_w))
        for _ in range(64):
            multiplier = 0.5 * (lower + upper)
            weights = np.clip((mu - multiplier) / (2.0 * self.l2), self.min_w, self.max_w)
            if float(weights.sum()) > self.budget:
                lower = multiplier
            else:
                upper = multiplier
        residual = self.budget - float(weights.sum())
        if abs(residual) > tol:
            movable = np.flatnonzero(
                weights < self.max_w - tol if residual > 0 else weights > self.min_w + tol
            )
            if movable.size:
                weights[movable[0]] += residual
        return weights


class MinVarianceOSQP:
    def __init__(self, spec: MeanRiskSpec, n_assets: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.l2 = float(spec.l2_coef)
        self.objective = spec.objective
        self.risk_aversion = float(spec.risk_aversion)
        self._x = self._y = None
        self._p_dense = np.empty((n_assets, n_assets), dtype=np.float64)
        self.n_warm_starts = 0
        n = n_assets
        self._A = sp.vstack([sp.csr_matrix(np.ones((1, n))), sp.eye(n, format="csr")]).tocsc()
        self._l = np.concatenate([[self.budget], self.min_w])
        self._u = np.concatenate([[self.budget], self.max_w])
        self._q = np.zeros(n, dtype=np.float64)
        eye = identity(n)
        self._prob = osqp.OSQP()
        self._prob.setup(
            P=upper_csc(2.0 * (eye + self.l2 * eye + 1e-16 * np.ones((n, n)))),
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
        self._prob.update(Px=upper_data(self._p_dense), q=q)
        if warm and self._x is not None:
            self._prob.warm_start(x=self._x, y=self._y)
            self.n_warm_starts += 1
        elif self._x is not None:
            self._prob.warm_start(x=np.zeros_like(self._x), y=np.zeros_like(self._y))
        result = self._prob.solve(raise_error=False)
        status = str(result.info.status).lower()
        if "solved" not in status:
            self._p_dense[diagonal] += 1e-10
            self._prob.update(Px=upper_data(self._p_dense))
            result = self._prob.solve(raise_error=False)
            status = str(result.info.status).lower()
            if "solved" not in status:
                raise RuntimeError(f"OSQP failed: {result.info.status}")
        self._x = np.asarray(result.x, dtype=np.float64)
        self._y = np.asarray(result.y, dtype=np.float64)
        return self._x.copy()

