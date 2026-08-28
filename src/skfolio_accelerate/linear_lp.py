"""Persistent HiGHS simplex engines for boxed MeanRisk LPs.

MAD, FLPM, CVaR, and worst realization are linear programs. Adjacent
WalkForward folds share most scenario rows; baking ``R − μ`` into every
coefficient hides that overlap, so a previous simplex basis is a *worse*
start than a cold crash. This engine keeps an auxiliary portfolio mean
(MAD/FLPM) or raw returns (CVaR / worst realization) so overlapping
observations keep the same constraint rows and auxiliary variables.

A rolling step of ``s`` then overwrites ``s`` scenario rows plus the mean
equality, restores the previous basis, and reoptimizes. Dual/primal simplex
typically needs far fewer pivots than fold 1. Non-overlapping windows still
reuse the compiled sparsity pattern and attempt a basis warm start.
"""

from __future__ import annotations

import numpy as np
from highspy import Highs, HighsModelStatus, MatrixFormat, ObjSense, kHighsInf
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compact import MeanRiskSpec, _as_bounds
from skfolio_accelerate.moments import FoldMoments

_LP_RISKS = frozenset(
    {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.CVAR,
        RiskMeasure.WORST_REALIZATION,
    }
)


def is_highs_lp_risk(spec: MeanRiskSpec) -> bool:
    """True when the compact problem is a pure LP that HiGHS should solve."""
    return spec.risk_measure in _LP_RISKS and float(spec.l2_coef) == 0.0


def rolling_shift(previous: NDArray[np.float64], current: NDArray[np.float64]) -> int | None:
    """Return ``s`` if ``current`` is ``previous`` advanced by ``s`` rows."""
    if previous.shape != current.shape or previous.size == 0:
        return None
    matches = np.flatnonzero(np.all(previous == current[0], axis=1))
    for start in matches:
        s = int(start)
        if s <= 0:
            continue
        if np.array_equal(previous[s:], current[: previous.shape[0] - s]):
            return s
    return None


class LinearHighs:
    """Persistent HiGHS LP with circular scenario slots and a warm basis."""

    def __init__(self, spec: MeanRiskSpec, n_assets: int, n_observations: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_observations = int(n_observations)
        self.min_w = _as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = _as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.solver = Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("presolve", "off")
        self.solver.setOptionValue("solver", "simplex")
        self.solver.setOptionValue("threads", 1)
        self.n_warm_starts = 0
        self._basis = None
        self._returns: NDArray[np.float64] | None = None
        self._slots = np.zeros((self.n_observations, self.n_assets), dtype=np.float64)
        self._window_to_slot = np.arange(self.n_observations, dtype=np.int32)
        self._built = False
        self._cost: NDArray[np.float64]
        self._col_lower: NDArray[np.float64]
        self._col_upper: NDArray[np.float64]
        self._row_lower: NDArray[np.float64]
        self._row_upper: NDArray[np.float64]
        self._a_start: NDArray[np.int32]
        self._a_index: NDArray[np.int32]
        self._a_value: NDArray[np.float64]
        self._mu_nz: NDArray[np.int32] | None = None
        self._r_nz: NDArray[np.int32]
        self._build_pattern()

    def _risk_scale(self) -> float:
        if self.spec.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            return float(self.spec.risk_aversion)
        return 1.0

    def _build_pattern(self) -> None:
        risk = self.spec.risk_measure
        n = self.n_assets
        t = self.n_observations
        lam = self._risk_scale()
        if risk in {
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        }:
            nv = n + 1 + t
            cost = np.zeros(nv, dtype=np.float64)
            coef = 2.0 if risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION else 1.0
            cost[n + 1 :] = lam * coef / t
            col_lower = np.zeros(nv, dtype=np.float64)
            col_upper = np.full(nv, kHighsInf, dtype=np.float64)
            col_lower[n] = -kHighsInf
            col_lower[:n] = self.min_w
            col_upper[:n] = self.max_w
            n_row = 2 + t
            starts = [0]
            indices: list[int] = []
            values: list[float] = []
            mu_nz = np.empty(n, dtype=np.int32)
            r_nz = np.empty((n, t), dtype=np.int32)
            for j in range(n):
                indices.extend([0, 1])
                values.extend([1.0, 0.0])
                mu_nz[j] = len(values) - 1
                for k in range(t):
                    indices.append(2 + k)
                    values.append(0.0)
                    r_nz[j, k] = len(values) - 1
                starts.append(len(indices))
            indices.append(1)
            values.append(1.0)
            for k in range(t):
                indices.append(2 + k)
                values.append(-1.0)
            starts.append(len(indices))
            for k in range(t):
                indices.append(2 + k)
                values.append(1.0)
                starts.append(len(indices))
            row_lower = np.zeros(n_row, dtype=np.float64)
            row_upper = np.full(n_row, kHighsInf, dtype=np.float64)
            row_lower[0] = self.budget
            row_upper[0] = self.budget
            row_upper[1] = 0.0
            self._mu_nz = mu_nz
            self._r_nz = r_nz
        elif risk is RiskMeasure.CVAR:
            nv = n + 1 + t
            cost = np.zeros(nv, dtype=np.float64)
            cost[n] = lam
            cost[n + 1 :] = lam / (t * (1.0 - float(self.spec.cvar_beta)))
            col_lower = np.full(nv, -kHighsInf, dtype=np.float64)
            col_upper = np.full(nv, kHighsInf, dtype=np.float64)
            col_lower[:n] = self.min_w
            col_upper[:n] = self.max_w
            col_lower[n + 1 :] = 0.0
            n_row = 1 + t
            starts = [0]
            indices = []
            values = []
            r_nz = np.empty((n, t), dtype=np.int32)
            for j in range(n):
                indices.append(0)
                values.append(1.0)
                for k in range(t):
                    indices.append(1 + k)
                    values.append(0.0)
                    r_nz[j, k] = len(values) - 1
                starts.append(len(indices))
            for k in range(t):
                indices.append(1 + k)
                values.append(1.0)
            starts.append(len(indices))
            for k in range(t):
                indices.append(1 + k)
                values.append(1.0)
                starts.append(len(indices))
            row_lower = np.zeros(n_row, dtype=np.float64)
            row_upper = np.full(n_row, kHighsInf, dtype=np.float64)
            row_lower[0] = self.budget
            row_upper[0] = self.budget
            self._mu_nz = None
            self._r_nz = r_nz
        elif risk is RiskMeasure.WORST_REALIZATION:
            nv = n + 1
            cost = np.zeros(nv, dtype=np.float64)
            cost[n] = lam
            col_lower = np.full(nv, -kHighsInf, dtype=np.float64)
            col_upper = np.full(nv, kHighsInf, dtype=np.float64)
            col_lower[:n] = self.min_w
            col_upper[:n] = self.max_w
            n_row = 1 + t
            starts = [0]
            indices = []
            values = []
            r_nz = np.empty((n, t), dtype=np.int32)
            for j in range(n):
                indices.append(0)
                values.append(1.0)
                for k in range(t):
                    indices.append(1 + k)
                    values.append(0.0)
                    r_nz[j, k] = len(values) - 1
                starts.append(len(indices))
            for k in range(t):
                indices.append(1 + k)
                values.append(1.0)
            starts.append(len(indices))
            row_lower = np.zeros(n_row, dtype=np.float64)
            row_upper = np.full(n_row, kHighsInf, dtype=np.float64)
            row_lower[0] = self.budget
            row_upper[0] = self.budget
            self._mu_nz = None
            self._r_nz = r_nz
        else:
            raise ValueError(f"unsupported LP risk {risk}")
        self._cost = cost
        self._col_lower = col_lower
        self._col_upper = col_upper
        self._row_lower = row_lower
        self._row_upper = row_upper
        self._a_start = np.asarray(starts, dtype=np.int32)
        self._a_index = np.asarray(indices, dtype=np.int32)
        self._a_value = np.asarray(values, dtype=np.float64)

    def _write_slots(self, slot_indices: NDArray[np.int32], rows: NDArray[np.float64]) -> None:
        n = self.n_assets
        for j in range(n):
            self._a_value[self._r_nz[j, slot_indices]] = rows[:, j]

    def _write_mu(self, mu: NDArray[np.float64]) -> None:
        if self._mu_nz is None:
            return
        self._a_value[self._mu_nz] = -np.asarray(mu, dtype=np.float64)

    def _bind_full(self, returns: NDArray[np.float64], mu: NDArray[np.float64]) -> None:
        t = self.n_observations
        self._slots = np.ascontiguousarray(returns, dtype=np.float64)
        self._window_to_slot = np.arange(t, dtype=np.int32)
        self._write_slots(self._window_to_slot, self._slots)
        self._write_mu(mu)

    def _bind_roll(
        self, returns: NDArray[np.float64], mu: NDArray[np.float64], shift: int
    ) -> None:
        dropped = self._window_to_slot[:shift]
        new_rows = np.ascontiguousarray(returns[-shift:], dtype=np.float64)
        self._slots[dropped] = new_rows
        self._write_slots(dropped, new_rows)
        self._window_to_slot = np.concatenate(
            [self._window_to_slot[shift:], dropped]
        ).astype(np.int32, copy=False)
        self._write_mu(mu)

    def _bind_objective(self, moments: FoldMoments) -> None:
        if self.spec.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            self._cost[: self.n_assets] = -np.asarray(moments.mu, dtype=np.float64)

    def _pass(self) -> None:
        integrality = np.zeros(self._cost.size, dtype=np.int32)
        status = self.solver.passModel(
            int(self._cost.size),
            int(self._row_lower.size),
            int(self._a_value.size),
            int(MatrixFormat.kColwise),
            int(ObjSense.kMinimize),
            0.0,
            self._cost,
            self._col_lower,
            self._col_upper,
            self._row_lower,
            self._row_upper,
            self._a_start,
            self._a_index,
            self._a_value,
            integrality,
        )
        if str(status) != "HighsStatus.kOk":
            raise RuntimeError(f"HiGHS passModel failed: {status}")

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        returns = np.ascontiguousarray(moments.returns, dtype=np.float64)
        t, n = returns.shape
        if t != self.n_observations or n != self.n_assets:
            self.n_observations = t
            self.n_assets = n
            self.min_w = _as_bounds(self.spec.min_weights, n, 0.0)
            self.max_w = _as_bounds(self.spec.max_weights, n, 1.0)
            self._basis = None
            self._returns = None
            self._built = False
            self._build_pattern()
        mu = np.asarray(moments.mu, dtype=np.float64)
        shift = (
            rolling_shift(self._returns, returns) if self._returns is not None else None
        )
        if shift is None:
            self._bind_full(returns, mu)
        else:
            self._bind_roll(returns, mu, shift)
        self._bind_objective(moments)
        self._pass()
        if warm and self._basis is not None:
            self.solver.setBasis(self._basis)
            self.n_warm_starts += 1
        run_status = self.solver.run()
        model_status = self.solver.getModelStatus()
        if model_status != HighsModelStatus.kOptimal:
            raise RuntimeError(
                f"HiGHS {self.spec.risk_measure.name} failed: {model_status} ({run_status})"
            )
        self._basis = self.solver.getBasis()
        self._returns = returns
        self._built = True
        x = np.asarray(self.solver.getSolution().col_value[: self.n_assets], dtype=np.float64)
        return x.copy()
