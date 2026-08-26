"""Compiled CVXPY MeanRisk problems with Parameter updates.

The compact OSQP engine is faster for boxed min-variance. This engine exists
for MeanRisk cases that still only need ``(μ, Σ)`` but add a linear return
constraint or an L1 term. CVXPY compiles the QP once (DPP); later windows
only write ``mu.value`` / ``cov.value`` and warm-start OSQP.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
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


def uses_cvxpy_params(spec: dict[str, Any]) -> bool:
    """True when variance MeanRisk needs the Parameter engine instead of OSQP."""
    if spec.get("risk_measure", RiskMeasure.VARIANCE) is not RiskMeasure.VARIANCE:
        return False
    if float(spec.get("l1_coef", 0.0) or 0.0) != 0.0:
        return True
    return spec.get("min_return") is not None


class CvxpyParamEngine:
    """Boxed mean-variance QP compiled once, then updated through Parameters."""

    def __init__(self, spec: dict[str, Any], n_assets: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = _as_bounds(spec["min_weights"], n_assets, 0.0)
        self.max_w = _as_bounds(spec["max_weights"], n_assets, 1.0)
        self.budget = float(spec["budget"])
        self.l2 = float(spec.get("l2_coef", 0.0) or 0.0)
        self.l1 = float(spec.get("l1_coef", 0.0) or 0.0)
        self.objective = spec["objective"]
        self.risk_aversion = float(spec.get("risk_aversion", 1.0) or 1.0)
        min_return = spec.get("min_return")
        self.min_return = None if min_return is None else float(min_return)
        self.n_warm_starts = 0
        self._solved_once = False
        self._build()

    def _build(self) -> None:
        n = self.n_assets
        self._mu = cp.Parameter(n)
        self._cov = cp.Parameter((n, n), PSD=True)
        self._w = cp.Variable(n)
        risk = cp.quad_form(self._w, self._cov)
        reg = self.l2 * cp.sum_squares(self._w)
        if self.l1 != 0.0:
            reg = reg + self.l1 * cp.norm1(self._w)
        constraints = [
            cp.sum(self._w) == self.budget,
            self._w >= self.min_w,
            self._w <= self.max_w,
        ]
        if self.min_return is not None:
            constraints.append(self._mu @ self._w >= self.min_return)
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            objective = cp.Minimize(
                self.risk_aversion * risk - self._mu @ self._w + reg
            )
        else:
            objective = cp.Minimize(risk + reg)
        self._problem = cp.Problem(objective, constraints)
        self._mu.value = np.zeros(n, dtype=np.float64)
        self._cov.value = np.eye(n, dtype=np.float64)

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        n = self.n_assets
        cov = np.asarray(moments.covariance, dtype=np.float64)
        if cov.shape != (n, n):
            raise ValueError(f"covariance shape {cov.shape} != {(n, n)}")
        mu = np.ascontiguousarray(moments.mu, dtype=np.float64)
        if mu.shape != (n,):
            raise ValueError(f"mu shape {mu.shape} != {(n,)}")
        self._mu.value = mu
        self._cov.value = np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
        weights = self._solve(warm=warm)
        if weights is None:
            self._cov.value = self._cov.value + 1e-10 * np.eye(n)
            weights = self._solve(warm=False)
        if weights is None:
            raise RuntimeError(f"CVXPY failed: {self._problem.status}")
        return weights

    def _solve(self, *, warm: bool) -> NDArray[np.float64] | None:
        self._problem.solve(
            solver=cp.OSQP,
            warm_start=bool(warm and self._solved_once),
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=4000,
            polish=False,
        )
        if warm and self._solved_once:
            self.n_warm_starts += 1
        self._solved_once = True
        status = str(self._problem.status).lower()
        if "optimal" not in status:
            return None
        return np.ascontiguousarray(self._w.value, dtype=np.float64)
