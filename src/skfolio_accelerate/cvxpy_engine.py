"""Compiled CVXPY MeanRisk problems with Parameter updates.

The compact OSQP engine is faster for boxed min-variance. This engine exists
for MeanRisk cases that still only need ``(μ, Σ)`` but add a linear return
constraint or an L1 term.

``cp.quad_form(w, Σ)`` with a PSD Parameter is DPP only on recent CVXPY.
``||Lᵀ w||²`` with ``L`` a Parameter is DPP on every CVXPY that skfolio 1.x
pulls in, so each window factorizes ``Σ_t = L Lᵀ`` and writes ``Lᵀ`` and
``μ`` into the compiled QP.
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


def _cholesky(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lower-triangular factor of a symmetric covariance, with light jitter."""
    n = int(cov.shape[0])
    symmetric = np.ascontiguousarray(0.5 * (cov + cov.T), dtype=np.float64)
    jitter = 0.0
    eye = np.eye(n, dtype=np.float64)
    for _ in range(6):
        try:
            return np.linalg.cholesky(symmetric + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    raise RuntimeError("covariance is not positive definite")


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
        # Lᵀ from Σ = L Lᵀ. sum_squares(Lᵀ w) is DPP for all supported CVXPY.
        self._chol_t = cp.Parameter((n, n))
        self._w = cp.Variable(n)
        risk = cp.sum_squares(self._chol_t @ self._w)
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
        self._chol_t.value = np.eye(n, dtype=np.float64)

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        n = self.n_assets
        cov = np.asarray(moments.covariance, dtype=np.float64)
        if cov.shape != (n, n):
            raise ValueError(f"covariance shape {cov.shape} != {(n, n)}")
        mu = np.ascontiguousarray(moments.mu, dtype=np.float64)
        if mu.shape != (n,):
            raise ValueError(f"mu shape {mu.shape} != {(n,)}")
        self._mu.value = mu
        self._chol_t.value = _cholesky(cov).T
        weights = self._solve(warm=warm)
        if weights is None:
            self._chol_t.value = _cholesky(cov + 1e-8 * np.eye(n)).T
            weights = self._solve(warm=False)
        if weights is None:
            raise RuntimeError(f"CVXPY failed: {self._problem.status}")
        return weights

    def _solve(self, *, warm: bool) -> NDArray[np.float64] | None:
        solve_kwargs: dict[str, Any] = {
            "solver": cp.OSQP,
            "warm_start": bool(warm and self._solved_once),
            "verbose": False,
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "max_iter": 4000,
        }
        # OSQP 1.x renamed polish → polishing; accept whichever this CVXPY maps.
        try:
            self._problem.solve(polishing=False, **solve_kwargs)
        except TypeError:
            self._problem.solve(polish=False, **solve_kwargs)
        if warm and self._solved_once:
            self.n_warm_starts += 1
        self._solved_once = True
        status = str(self._problem.status).lower()
        if "optimal" not in status:
            return None
        return np.ascontiguousarray(self._w.value, dtype=np.float64)
