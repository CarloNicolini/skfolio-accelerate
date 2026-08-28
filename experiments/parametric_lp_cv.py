"""Expendable CV-fold LP continuation experiments for MAD / CVaR / FLPM.

Compares:
* compact Clarabel (current skfolio-accelerate engine, rebuild vs data-update)
* HiGHS simplex from scratch
* persistent HiGHS with passModel + previous basis
* persistent HiGHS with coefficient updates + previous basis
* projected subgradient on the simplex (MAD only)

Ignore ADMM. The question is whether sequential CV LPs reoptimize much
cheaper than fold 1 once a basis exists.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from highspy import (
    Highs,
    HighsModelStatus,
    MatrixFormat,
    ObjSense,
    kHighsInf,
)
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.optimization import ObjectiveFunction

from skfolio_accelerate.compact import MeanRiskSpec, make_compact_engine
from skfolio_accelerate.flagship import factor_returns
from skfolio_accelerate.moments import FoldMoments, empirical_from_window


def _windows(n_obs: int, train: int, test: int) -> list[slice]:
    starts = range(0, n_obs - train - test + 1, test)
    return [slice(s, s + train) for s in starts]


def _moments(returns: NDArray[np.float64]) -> FoldMoments:
    return empirical_from_window(returns, keep_returns=True)


def _spec(risk: RiskMeasure) -> MeanRiskSpec:
    return MeanRiskSpec(
        risk_measure=risk,
        objective=ObjectiveFunction.MINIMIZE_RISK,
        l2_coef=0.0,
        risk_aversion=1.0,
        cvar_beta=0.95,
        evar_beta=0.95,
        cdar_beta=0.95,
        edar_beta=0.95,
        min_acceptable_return=None,
        min_weights=0.0,
        max_weights=1.0,
        budget=1.0,
    )


def _deviations(moments: FoldMoments) -> NDArray[np.float64]:
    return np.ascontiguousarray(moments.returns - moments.mu, dtype=np.float64)


# ---------------------------------------------------------------------------
# MAD / FLPM / CVaR as standard-form LPs with variable bounds
# ---------------------------------------------------------------------------

@dataclass
class LpData:
    cost: NDArray[np.float64]
    col_lower: NDArray[np.float64]
    col_upper: NDArray[np.float64]
    row_lower: NDArray[np.float64]
    row_upper: NDArray[np.float64]
    a_start: NDArray[np.int32]
    a_index: NDArray[np.int32]
    a_value: NDArray[np.float64]
    n_assets: int

    @property
    def num_col(self) -> int:
        return int(self.cost.size)

    @property
    def num_row(self) -> int:
        return int(self.row_lower.size)

    @property
    def num_nz(self) -> int:
        return int(self.a_value.size)


def build_mad_lp(moments: FoldMoments, *, double_sided: bool = False) -> LpData:
    """Konno–Yamazaki MAD (one-sided, default) or two-sided epigraph LP."""
    r = np.asarray(moments.returns, dtype=np.float64)
    t, n = r.shape
    dev = _deviations(moments)
    if double_sided:
        # min (1/T) 1'u  s.t. u >= |dev w|, 1'w=1, 0<=w<=1, u>=0
        nv = n + t
        cost = np.zeros(nv, dtype=np.float64)
        cost[n:] = 1.0 / t
        col_lower = np.zeros(nv, dtype=np.float64)
        col_upper = np.full(nv, kHighsInf, dtype=np.float64)
        col_upper[:n] = 1.0
        # rows: budget; +dev w - u <= 0; -dev w - u <= 0
        n_row = 1 + 2 * t
        # CSC: each w_j appears in budget + 2T scenario rows; each u_k in two rows
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for j in range(n):
            indices.append(0)
            values.append(1.0)
            for k in range(t):
                indices.append(1 + k)
                values.append(float(dev[k, j]))
                indices.append(1 + t + k)
                values.append(float(-dev[k, j]))
            starts.append(len(indices))
        for k in range(t):
            indices.extend([1 + k, 1 + t + k])
            values.extend([-1.0, -1.0])
            starts.append(len(indices))
        row_lower = np.full(n_row, -kHighsInf, dtype=np.float64)
        row_upper = np.zeros(n_row, dtype=np.float64)
        row_lower[0] = 1.0
        row_upper[0] = 1.0
    else:
        # MAD = 2 E[(dev w)_-] when E[dev w]=0: min (2/T) 1'u, u >= -dev w, u>=0
        nv = n + t
        cost = np.zeros(nv, dtype=np.float64)
        cost[n:] = 2.0 / t
        col_lower = np.zeros(nv, dtype=np.float64)
        col_upper = np.full(nv, kHighsInf, dtype=np.float64)
        col_upper[:n] = 1.0
        n_row = 1 + t
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for j in range(n):
            indices.append(0)
            values.append(1.0)
            for k in range(t):
                indices.append(1 + k)
                values.append(float(-dev[k, j]))
            starts.append(len(indices))
        for k in range(t):
            indices.append(1 + k)
            values.append(-1.0)
            starts.append(len(indices))
        row_lower = np.full(n_row, -kHighsInf, dtype=np.float64)
        row_upper = np.zeros(n_row, dtype=np.float64)
        row_lower[0] = 1.0
        row_upper[0] = 1.0
    return LpData(
        cost=cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        a_start=np.asarray(starts, dtype=np.int32),
        a_index=np.asarray(indices, dtype=np.int32),
        a_value=np.asarray(values, dtype=np.float64),
        n_assets=n,
    )


def build_flpm_lp(moments: FoldMoments) -> LpData:
    lp = build_mad_lp(moments, double_sided=False)
    t = int(moments.n_observations or moments.returns.shape[0])
    lp.cost[lp.n_assets :] = 1.0 / t
    return lp


def build_cvar_lp(moments: FoldMoments, beta: float = 0.95) -> LpData:
    r = np.asarray(moments.returns, dtype=np.float64)
    t, n = r.shape
    nv = n + 1 + t  # w, alpha, u
    cost = np.zeros(nv, dtype=np.float64)
    cost[n] = 1.0
    cost[n + 1 :] = 1.0 / (t * (1.0 - beta))
    col_lower = np.full(nv, -kHighsInf, dtype=np.float64)
    col_upper = np.full(nv, kHighsInf, dtype=np.float64)
    col_lower[:n] = 0.0
    col_upper[:n] = 1.0
    col_lower[n + 1 :] = 0.0
    # rows: 1'w = 1;  -R w - alpha - u <= 0
    n_row = 1 + t
    starts = [0]
    indices: list[int] = []
    values: list[float] = []
    for j in range(n):
        indices.append(0)
        values.append(1.0)
        for k in range(t):
            indices.append(1 + k)
            values.append(float(-r[k, j]))
        starts.append(len(indices))
    # alpha
    for k in range(t):
        indices.append(1 + k)
        values.append(-1.0)
    starts.append(len(indices))
    for k in range(t):
        indices.append(1 + k)
        values.append(-1.0)
        starts.append(len(indices))
    row_lower = np.full(n_row, -kHighsInf, dtype=np.float64)
    row_upper = np.zeros(n_row, dtype=np.float64)
    row_lower[0] = 1.0
    row_upper[0] = 1.0
    return LpData(
        cost=cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        a_start=np.asarray(starts, dtype=np.int32),
        a_index=np.asarray(indices, dtype=np.int32),
        a_value=np.asarray(values, dtype=np.float64),
        n_assets=n,
    )


def build_mad_mup_lp(moments: FoldMoments) -> LpData:
    """MAD with explicit portfolio mean so scenario rows store raw r_t, not r_t-μ.

    Variables: w (n), μ_p (1), u (T)
    min (2/T) 1ᵀu
    1ᵀw = 1,  μ_p = μᵀw,  r_tᵀw − μ_p + u_t ≥ 0,  0≤w≤1, u≥0
    """
    r = np.asarray(moments.returns, dtype=np.float64)
    t, n = r.shape
    mu = np.asarray(moments.mu, dtype=np.float64)
    nv = n + 1 + t
    cost = np.zeros(nv, dtype=np.float64)
    cost[n + 1 :] = 2.0 / t
    col_lower = np.zeros(nv, dtype=np.float64)
    col_upper = np.full(nv, kHighsInf, dtype=np.float64)
    col_lower[n] = -kHighsInf
    col_upper[:n] = 1.0
    n_row = 2 + t
    starts = [0]
    indices: list[int] = []
    values: list[float] = []
    for j in range(n):
        indices.extend([0, 1])
        values.extend([1.0, -float(mu[j])])
        for k in range(t):
            indices.append(2 + k)
            values.append(float(r[k, j]))
        starts.append(len(indices))
    # μ_p column: +1 on mean equality, −1 on each scenario
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
    row_lower[0] = 1.0
    row_upper[0] = 1.0
    row_upper[1] = 0.0
    return LpData(
        cost=cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        a_start=np.asarray(starts, dtype=np.int32),
        a_index=np.asarray(indices, dtype=np.int32),
        a_value=np.asarray(values, dtype=np.float64),
        n_assets=n,
    )


def build_lp(risk: RiskMeasure, moments: FoldMoments, *, mad_two_sided: bool) -> LpData:
    if risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
        return build_mad_lp(moments, double_sided=mad_two_sided)
    if risk is RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
        return build_flpm_lp(moments)
    if risk is RiskMeasure.CVAR:
        return build_cvar_lp(moments)
    raise ValueError(risk)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _configure_highs(h: Highs, *, solver: str) -> None:
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", solver)
    h.setOptionValue("threads", 1)


def _pass_lp(h: Highs, lp: LpData):
    integrality = np.zeros(lp.num_col, dtype=np.int32)
    return h.passModel(
        lp.num_col,
        lp.num_row,
        lp.num_nz,
        int(MatrixFormat.kColwise),
        int(ObjSense.kMinimize),
        0.0,
        lp.cost,
        lp.col_lower,
        lp.col_upper,
        lp.row_lower,
        lp.row_upper,
        lp.a_start,
        lp.a_index,
        lp.a_value,
        integrality,
    )


def _highs_weights(h: Highs, n_assets: int) -> NDArray[np.float64]:
    status = h.getModelStatus()
    if status != HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS status {status}")
    return np.asarray(h.getSolution().col_value[:n_assets], dtype=np.float64)


def basis_overlap(prev, curr) -> float:
    if prev is None or curr is None:
        return float("nan")
    col = np.array([int(s) for s in prev.col_status]) == np.array(
        [int(s) for s in curr.col_status]
    )
    row = np.array([int(s) for s in prev.row_status]) == np.array(
        [int(s) for s in curr.row_status]
    )
    return float(np.mean(np.concatenate([col, row])))


class HighsCold:
    name = "highs-cold"

    def __init__(self, solver: str = "simplex") -> None:
        self.solver = solver
        self.last_iters = 0
        self.last_overlap = float("nan")

    def solve(self, lp: LpData) -> NDArray[np.float64]:
        h = Highs()
        _configure_highs(h, solver=self.solver)
        _pass_lp(h, lp)
        h.run()
        self.last_iters = int(h.getInfo().simplex_iteration_count)
        if self.solver == "ipm":
            self.last_iters = int(h.getInfo().ipm_iteration_count)
        self.last_overlap = float("nan")
        return _highs_weights(h, lp.n_assets)


class HighsPersistent:
    """Reuse one HiGHS instance: passModel each fold, optionally setBasis."""

    def __init__(self, solver: str = "simplex", *, use_basis: bool = True) -> None:
        self.solver = solver
        self.use_basis = use_basis
        self.h: Highs | None = None
        self.basis = None
        self.last_iters = 0
        self.last_overlap = float("nan")
        self.name = f"highs-{solver}-{'basis' if use_basis else 'nobasis'}"

    def solve(self, lp: LpData) -> NDArray[np.float64]:
        if self.h is None:
            self.h = Highs()
            _configure_highs(self.h, solver=self.solver)
        _pass_lp(self.h, lp)
        if self.use_basis and self.basis is not None:
            self.h.setBasis(self.basis)
        self.h.run()
        info = self.h.getInfo()
        self.last_iters = int(
            info.simplex_iteration_count if self.solver == "simplex" else info.ipm_iteration_count
        )
        new_basis = self.h.getBasis()
        self.last_overlap = basis_overlap(self.basis, new_basis)
        self.basis = new_basis
        return _highs_weights(self.h, lp.n_assets)


class HighsChangeCoeff:
    """Keep sparsity; rewrite scenario coefficients in-place, then reoptimize."""

    def __init__(self) -> None:
        self.h: Highs | None = None
        self.basis = None
        self.last_iters = 0
        self.last_overlap = float("nan")
        self.name = "highs-changecoeff-basis"
        self._prev_values: NDArray[np.float64] | None = None

    def solve(self, lp: LpData) -> NDArray[np.float64]:
        if self.h is None:
            self.h = Highs()
            _configure_highs(self.h, solver="simplex")
            _pass_lp(self.h, lp)
        else:
            prev = self._prev_values
            assert prev is not None
            # Only rewrite numerically changed entries (Python loop; measured).
            changed = np.flatnonzero(np.abs(lp.a_value - prev) > 0.0)
            for nz in changed:
                col = int(np.searchsorted(lp.a_start, nz, side="right") - 1)
                row = int(lp.a_index[nz])
                self.h.changeCoeff(row, col, float(lp.a_value[nz]))
        self._prev_values = lp.a_value.copy()
        if self.basis is not None:
            self.h.setBasis(self.basis)
        self.h.run()
        self.last_iters = int(self.h.getInfo().simplex_iteration_count)
        new_basis = self.h.getBasis()
        self.last_overlap = basis_overlap(self.basis, new_basis)
        self.basis = new_basis
        return _highs_weights(self.h, lp.n_assets)


class HighsMuPPersistent:
    """passModel of the μ_p formulation + previous basis (no circular slots)."""

    def __init__(self) -> None:
        self.h: Highs | None = None
        self.basis = None
        self.last_iters = 0
        self.last_overlap = float("nan")
        self.name = "highs-mup-basis"

    def solve_moments(self, moments: FoldMoments) -> NDArray[np.float64]:
        lp = build_mad_mup_lp(moments)
        if self.h is None:
            self.h = Highs()
            _configure_highs(self.h, solver="simplex")
        _pass_lp(self.h, lp)
        if self.basis is not None:
            self.h.setBasis(self.basis)
        self.h.run()
        self.last_iters = int(self.h.getInfo().simplex_iteration_count)
        new_basis = self.h.getBasis()
        self.last_overlap = basis_overlap(self.basis, new_basis)
        self.basis = new_basis
        return _highs_weights(self.h, lp.n_assets)


class HighsCircularMad:
    """Persistent MAD LP: scenario rows are circular in calendar time.

    Slot ``t % T`` stores observation ``t``. A walk-forward step of ``s``
    therefore overwrites ``s`` return rows plus the n mean-equality
    coefficients, instead of rewriting (R − μ).
    """

    def __init__(self, train: int, n_assets: int) -> None:
        self.train = int(train)
        self.n_assets = int(n_assets)
        self.h: Highs | None = None
        self.basis = None
        self.last_iters = 0
        self.last_overlap = float("nan")
        self.name = "highs-circular-mup"
        self._filled = False
        self.r_slots = np.zeros((train, n_assets), dtype=np.float64)
        self.mu = np.zeros(n_assets, dtype=np.float64)
        self.lp: LpData | None = None

    def _build_lp(self) -> LpData:
        dummy = FoldMoments(
            mu=self.mu,
            covariance=np.eye(self.n_assets),
            returns=self.r_slots,
            n_observations=self.train,
        )
        return build_mad_mup_lp(dummy)

    def solve_window(self, returns: NDArray[np.float64], start: int) -> NDArray[np.float64]:
        window = np.ascontiguousarray(returns[start : start + self.train], dtype=np.float64)
        mu = window.mean(axis=0)
        t0 = start
        if not self._filled:
            for i in range(self.train):
                self.r_slots[(t0 + i) % self.train] = window[i]
            self.mu = mu
            self.lp = self._build_lp()
            self.h = Highs()
            _configure_highs(self.h, solver="simplex")
            _pass_lp(self.h, self.lp)
            self._filled = True
        else:
            assert self.h is not None and self.lp is not None
            # Replace the s observations that left the window. Caller may jump
            # by more than 1; overwrite every slot from current mapping.
            new_slots = np.empty_like(self.r_slots)
            for i in range(self.train):
                new_slots[(t0 + i) % self.train] = window[i]
            changed_slots = np.where(np.any(new_slots != self.r_slots, axis=1))[0]
            self.r_slots = new_slots
            self.mu = mu
            lp = self.lp
            n = self.n_assets
            # Column j values: [1, -mu_j, r_0j, ..., r_{T-1}j]
            for j in range(n):
                start_nz = int(lp.a_start[j])
                lp.a_value[start_nz + 1] = -float(mu[j])
                self.h.changeCoeff(1, j, -float(mu[j]))
                for slot in changed_slots:
                    val = float(self.r_slots[slot, j])
                    lp.a_value[start_nz + 2 + int(slot)] = val
                    self.h.changeCoeff(2 + int(slot), j, val)
        if self.basis is not None:
            self.h.setBasis(self.basis)
        self.h.run()
        self.last_iters = int(self.h.getInfo().simplex_iteration_count)
        new_basis = self.h.getBasis()
        self.last_overlap = basis_overlap(self.basis, new_basis)
        self.basis = new_basis
        return _highs_weights(self.h, self.n_assets)


class ClarabelEngine:
    def __init__(self, spec: MeanRiskSpec, n_assets: int, n_obs: int, *, reuse: bool) -> None:
        self.spec = spec
        self.n_assets = n_assets
        self.n_obs = n_obs
        self.reuse = reuse
        self.engine = None
        self.last_iters = 0
        self.last_overlap = float("nan")
        self.name = "clarabel-update" if reuse else "clarabel-rebuild"

    def solve_moments(self, moments: FoldMoments) -> NDArray[np.float64]:
        if (not self.reuse) or self.engine is None:
            self.engine = make_compact_engine(
                self.spec, n_assets=self.n_assets, n_observations=self.n_obs
            )
        return self.engine.solve(moments, warm=self.reuse)


class SubgradientMAD:
    """Projected subgradient of MAD onto the probability simplex.

    Cheap per iteration (R @ w), but not an LP. Included to test whether a
    specialized first-order method beats simplex on CV folds with warm w.
    """

    name = "subgradient-mad"

    def __init__(self, max_iter: int = 400, step0: float = 0.5) -> None:
        self.max_iter = max_iter
        self.step0 = step0
        self.w: NDArray[np.float64] | None = None
        self.last_iters = 0
        self.last_overlap = float("nan")

    def solve_moments(self, moments: FoldMoments) -> NDArray[np.float64]:
        r = np.asarray(moments.returns, dtype=np.float64)
        t, n = r.shape
        dev = r - moments.mu
        w = (
            self.w.copy()
            if self.w is not None and self.w.size == n
            else np.full(n, 1.0 / n)
        )
        best = w.copy()
        best_f = float("inf")
        for k in range(1, self.max_iter + 1):
            z = dev @ w
            f = float(np.mean(np.abs(z)))
            if f < best_f:
                best_f = f
                best = w.copy()
            g = (dev.T @ np.sign(z)) / t
            eta = self.step0 / np.sqrt(k)
            w = w - eta * g
            w = np.clip(w, 0.0, 1.0)
            s = float(w.sum())
            w = w / s if s > 0 else np.full(n, 1.0 / n)
        self.w = best
        self.last_iters = self.max_iter
        return best


@dataclass
class FoldStat:
    method: str
    fold: int
    seconds: float
    iters: int
    overlap: float
    mad: float
    max_dw: float


def _portfolio_mad(returns: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    excess = (returns - returns.mean(axis=0)) @ w
    return float(np.mean(np.abs(excess)))


def run(
    *,
    risk: RiskMeasure,
    n_obs: int,
    n_assets: int,
    train: int,
    test: int,
    mad_two_sided: bool,
    methods: list[str],
) -> list[FoldStat]:
    X = factor_returns(n_obs, n_assets, seed=42).to_numpy(dtype=np.float64)
    windows = _windows(n_obs, train, test)
    spec = _spec(risk)
    ref_w: NDArray[np.float64] | None = None
    stats: list[FoldStat] = []

    solvers: dict[str, object] = {}
    if "clarabel-rebuild" in methods:
        solvers["clarabel-rebuild"] = ClarabelEngine(spec, n_assets, train, reuse=False)
    if "clarabel-update" in methods:
        solvers["clarabel-update"] = ClarabelEngine(spec, n_assets, train, reuse=True)
    if "highs-cold" in methods:
        solvers["highs-cold"] = HighsCold("simplex")
    if "highs-simplex-basis" in methods:
        solvers["highs-simplex-basis"] = HighsPersistent("simplex", use_basis=True)
    if "highs-simplex-nobasis" in methods:
        solvers["highs-simplex-nobasis"] = HighsPersistent("simplex", use_basis=False)
    if "highs-ipm-basis" in methods:
        solvers["highs-ipm-basis"] = HighsPersistent("ipm", use_basis=True)
    if "highs-changecoeff-basis" in methods:
        solvers["highs-changecoeff-basis"] = HighsChangeCoeff()
    if "highs-mup-basis" in methods and risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
        solvers["highs-mup-basis"] = HighsMuPPersistent()
    if "highs-circular-mup" in methods and risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
        solvers["highs-circular-mup"] = HighsCircularMad(train, n_assets)
    if "subgradient-mad" in methods and risk is RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
        solvers["subgradient-mad"] = SubgradientMAD()

    for fold, sl in enumerate(windows):
        moments = _moments(X[sl])
        need_lp = any(
            name.startswith("highs-")
            and name not in {"highs-mup-basis", "highs-circular-mup"}
            for name in solvers
        )
        lp = build_lp(risk, moments, mad_two_sided=mad_two_sided) if need_lp else None
        for name, solver in solvers.items():
            t0 = time.perf_counter()
            if hasattr(solver, "solve_window"):
                w = solver.solve_window(X, sl.start)
            elif hasattr(solver, "solve_moments"):
                w = solver.solve_moments(moments)
            else:
                w = solver.solve(lp)
            elapsed = time.perf_counter() - t0
            if ref_w is None and name == "clarabel-update":
                ref_w = w.copy()
            # reset reference per fold from first solver
            if name == next(iter(solvers)):
                fold_ref = w.copy()
            max_dw = float(np.max(np.abs(w - fold_ref)))
            stats.append(
                FoldStat(
                    method=name,
                    fold=fold,
                    seconds=elapsed,
                    iters=int(getattr(solver, "last_iters", 0)),
                    overlap=float(getattr(solver, "last_overlap", float("nan"))),
                    mad=_portfolio_mad(moments.returns, w),
                    max_dw=max_dw,
                )
            )
    return stats


def summarize(stats: list[FoldStat]) -> None:
    methods = list(dict.fromkeys(s.method for s in stats))
    print(
        f"{'method':<28} {'folds':>5} {'setup_s':>9} {'mean_s':>9} "
        f"{'fold0_s':>9} {'later_s':>9} {'later/0':>8} {'mean_it':>8} "
        f"{'later_it':>8} {'overlap':>8} {'max_dw':>9}"
    )
    for method in methods:
        rows = [s for s in stats if s.method == method]
        times = np.array([s.seconds for s in rows])
        iters = np.array([s.iters for s in rows], dtype=float)
        later_t = times[1:] if times.size > 1 else times
        later_i = iters[1:] if iters.size > 1 else iters
        overlaps = [s.overlap for s in rows[1:] if np.isfinite(s.overlap)]
        print(
            f"{method:<28} {len(rows):5d} {times[0]:9.4f} {times.mean():9.4f} "
            f"{times[0]:9.4f} {later_t.mean():9.4f} {later_t.mean() / times[0]:8.3f} "
            f"{np.nanmean(iters):8.1f} {np.nanmean(later_i):8.1f} "
            f"{(np.mean(overlaps) if overlaps else float('nan')):8.3f} "
            f"{max(s.max_dw for s in rows):9.2e}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--risk", default="MEAN_ABSOLUTE_DEVIATION")
    p.add_argument("--n-obs", type=int, default=800)
    p.add_argument("--n-assets", type=int, default=20)
    p.add_argument("--train", type=int, default=252)
    p.add_argument("--test", type=int, default=21)
    p.add_argument("--no-folds", action="store_true")
    p.add_argument("--two-sided", action="store_true")
    p.add_argument(
        "--methods",
        default=(
            "clarabel-rebuild,clarabel-update,highs-cold,"
            "highs-simplex-nobasis,highs-simplex-basis,"
            "highs-mup-basis,highs-circular-mup"
        ),
    )
    args = p.parse_args()
    risk = RiskMeasure[args.risk]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(
        f"risk={risk.name}  data={args.n_obs}x{args.n_assets}  "
        f"train={args.train} test={args.test}  two_sided={args.two_sided}"
    )
    stats = run(
        risk=risk,
        n_obs=args.n_obs,
        n_assets=args.n_assets,
        train=args.train,
        test=args.test,
        mad_two_sided=args.two_sided,
        methods=methods,
    )
    summarize(stats)
    folds = sorted({s.fold for s in stats})
    if len(folds) >= 2 and not args.no_folds:
        print("\nper-fold simplex iterations:")
        for method in ("highs-simplex-basis", "highs-mup-basis", "highs-circular-mup"):
            rows = [s for s in stats if s.method == method]
            if not rows:
                continue
            print(f"  {method}")
            for s in rows:
                print(
                    f"    fold {s.fold:3d}  {s.seconds:8.4f}s  iters={s.iters:5d}  "
                    f"overlap={s.overlap:6.3f}  mad={s.mad:.6e}  dw={s.max_dw:.2e}"
                )


if __name__ == "__main__":
    main()
