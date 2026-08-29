"""Persistent COSMO.rs compact engines for boxed MeanRisk.

COSMO is an optional backend. ``backend="auto"`` still selects OSQP, HiGHS,
or Clarabel. Pass ``backend="cosmo"`` or ``MeanRisk(solver="COSMO")`` to
dispatch the same compact cone problems to a native Rust COSMO workspace.

The workspace is local to one ``cross_val_predict`` call. Consecutive folds
update ``P`` / ``q`` / ``A`` / ``b`` in place when the persist mode allows it
and keep ADMM iterates when that is mathematically valid. See
:mod:`skfolio_accelerate.formulations` for the class A–F map.

COSMO.rs Python bindings expose ``update_q``, ``update_b`` (no refactor),
``update_p`` (numerical refactor, same sparsity), and ``update_a`` (KKT
rebuild). This module does not claim factorisation reuse for ``update_a``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import clarabel
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compact import (
    CVaRClarabel,
    MeanRiskSpec,
    ScenarioClarabel,
    _as_bounds,
    _rows_to_csc,
    _upper_csc,
)
from skfolio_accelerate.moments import FoldMoments

PersistMode = Literal["cold", "warm_x", "warm_xy", "persist_factor", "persist_full"]
RestartPolicy = Literal["never", "iter_threshold", "status"]

COSMO_SOLVER_NAMES = frozenset({"COSMO", "COSMO_RS", "COSMO_RUST"})
_LP_RISKS = frozenset(
    {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.CVAR,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
    }
)
# ADMM is a poor simplex substitute. These LPs need COSMO.jl-scale gaps.
_SLOW_ADMM_RISKS = frozenset(
    {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
    }
)


def cosmo_available() -> bool:
    """Return whether the optional ``cosmo_rs`` extension can be imported."""
    try:
        import cosmo_rs  # noqa: F401
    except ImportError:
        return False
    return True


def cosmo_persistence_api_available() -> bool:
    """True when COSMO.rs exposes ``update_p`` / ``update_a`` / ``reset``.

    GitHub ``main`` Python bindings currently expose only ``update_q``,
    ``update_b``, and ``warm_start``. Persistence requires the extra methods
    (Rust already has them). Without them, persist modes reconstruct the
    solver each fold.
    """
    if not cosmo_available():
        return False
    from cosmo_rs import CosmoSolver

    return all(
        hasattr(CosmoSolver, name) for name in ("update_p", "update_a", "reset")
    )


def default_persist_mode(spec: MeanRiskSpec) -> PersistMode:
    """Persist mode justified by the walk-forward ablation.

    Variance is class C (``update_p``, KKT pattern reusable). Scenario risks
    are class B (``update_a`` drops the KKT system); carrying stale ADMM
    iterates increased iteration count on the measured panel.
    """
    if spec.risk_measure is RiskMeasure.VARIANCE:
        return "persist_full"
    return "persist_factor"


def uses_cosmo_solver(estimator) -> bool:
    """True when the estimator asks for COSMO rather than Clarabel/OSQP."""
    return str(getattr(estimator, "solver", "") or "").upper() in COSMO_SOLVER_NAMES


def clarabel_cones_to_cosmo(cones: list[Any]) -> list[tuple[str, int] | tuple[str]]:
    """Map Clarabel cone objects to COSMO.rs Python cone tuples."""
    out: list[tuple[str, int] | tuple[str]] = []
    for cone in cones:
        name = type(cone).__name__
        dim = int(getattr(cone, "dim", getattr(cone, "n", 0)) or 0)
        if name == "ZeroConeT":
            out.append(("zero", dim))
        elif name in {"NonnegativeConeT", "NonNegativeConeT"}:
            out.append(("nonnegative", dim))
        elif name == "SecondOrderConeT":
            out.append(("soc", dim))
        elif name == "ExponentialConeT":
            out.append(("exp",))
        else:
            raise TypeError(f"Unsupported Clarabel cone {name} for COSMO.rs")
    return out


def _as_csc(matrix: sp.spmatrix) -> sp.csc_matrix:
    csc = matrix.tocsc()
    csc.sum_duplicates()
    csc.sort_indices()
    return csc


@dataclass(slots=True)
class CosmoFoldTrace:
    """Per-fold COSMO diagnostics used by the persistence experiment."""

    status: str
    iterations: int
    r_prim: float
    r_dual: float
    obj_val: float
    setup_time: float
    solve_time: float
    factor_time: float
    iter_time: float
    proj_time: float
    reused_solver: bool
    updated_p: bool
    updated_a: bool
    updated_q: bool
    updated_b: bool
    restarted: bool
    weight_step: float


def default_cosmo_settings(spec: MeanRiskSpec) -> dict[str, Any]:
    """Settings biased toward correctness, not COSMO.jl's loose defaults.

    Linear programs (and LPs with a tiny ℓ₂ diagonal) disable Anderson
    acceleration and use COSMO.jl-scale tolerances. Tight ``1e-8`` IPM-style
    gaps routinely exhaust ``max_iter`` on MAD / drawdown LPs; HiGHS remains
    the auto engine for those problems.
    """
    lp = spec.risk_measure in _LP_RISKS and float(spec.l2_coef) == 0.0
    slow = spec.risk_measure in _SLOW_ADMM_RISKS
    if slow:
        return {
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 25_000,
            "verbose": False,
            "scaling": 10,
            "adaptive_rho": True,
            "accelerate": False,
            "check_termination": 25,
        }
    return {
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "max_iter": 15_000,
        "verbose": False,
        "scaling": 10,
        "adaptive_rho": True,
        "accelerate": not lp,
        "check_termination": 25,
    }


class PersistentCosmo:
    """One COSMO.rs solver that can be updated across related problems."""

    def __init__(
        self,
        *,
        settings: dict[str, Any],
        persist_mode: PersistMode = "persist_full",
        restart_policy: RestartPolicy = "status",
        restart_iter_threshold: int = 8_000,
    ) -> None:
        if persist_mode not in {
            "cold",
            "warm_x",
            "warm_xy",
            "persist_factor",
            "persist_full",
        }:
            raise ValueError(f"Unknown persist_mode {persist_mode!r}")
        self.settings = dict(settings)
        self.persist_mode: PersistMode = persist_mode
        self.restart_policy: RestartPolicy = restart_policy
        self.restart_iter_threshold = int(restart_iter_threshold)
        self.solver = None
        self._x: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None
        self.n_warm_starts = 0
        self.n_restarts = 0
        self.n_rebuilds = 0
        self.last_trace: CosmoFoldTrace | None = None
        self.traces: list[CosmoFoldTrace] = []

    def _import_solver(self):
        try:
            from cosmo_rs import CosmoSolver
        except ImportError as error:
            raise ImportError(
                "COSMO.rs is not installed. Build it with "
                "`maturin develop --release --features python` from "
                "https://github.com/CarloNicolini/COSMO.rs and retry, or "
                "install the optional extra `skfolio-accelerate[cosmo]`."
            ) from error
        return CosmoSolver

    def _new_solver(self, P, q, A, b, cones):
        CosmoSolver = self._import_solver()
        self.n_rebuilds += 1
        return CosmoSolver(P, q, A, b, cones, **self.settings)

    def _should_restart(self, last_iter: int, status: str) -> bool:
        if self.restart_policy == "never":
            return False
        if "solved" not in status.lower():
            return True
        if self.restart_policy == "iter_threshold":
            return last_iter >= self.restart_iter_threshold
        return False

    def solve(
        self,
        P: sp.spmatrix,
        q: NDArray[np.float64],
        A: sp.spmatrix,
        b: NDArray[np.float64],
        cones: list[Any],
        *,
        warm: bool = True,
        p_changed: bool = False,
        a_changed: bool = False,
        q_changed: bool = False,
        b_changed: bool = False,
    ) -> NDArray[np.float64]:
        """Solve one canonical problem, optionally reusing the workspace."""
        P = _as_csc(P)
        A = _as_csc(A)
        q = np.ascontiguousarray(q, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        cosmo_cones = clarabel_cones_to_cosmo(cones)
        mode = self.persist_mode
        reused = False
        restarted = False
        updated_p = updated_a = updated_q = updated_b = False

        # cold / warm_x / warm_xy reconstruct so ablations do not inherit
        # rho, scaling, or a factorisation. persist_* keep the workspace.
        reconstruct = mode in {"cold", "warm_x", "warm_xy"}
        if not reconstruct and not cosmo_persistence_api_available():
            if self.solver is None:
                warnings.warn(
                    "This COSMO.rs build has no update_p/update_a/reset; "
                    "persist modes reconstruct the solver each fold.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            reconstruct = True
        if self.solver is None or reconstruct:
            self.solver = self._new_solver(P, q, A, b, cosmo_cones)
            if mode in {"warm_x", "warm_xy"} and warm and self._x is not None:
                y = self._y if mode == "warm_xy" else None
                self.solver.warm_start(
                    self._x.tolist(),
                    None if y is None else y.tolist(),
                )
                self.n_warm_starts += 1
                reused = True
        else:
            reused = True
            if p_changed:
                self.solver.update_p(P)
                updated_p = True
            if a_changed:
                self.solver.update_a(A)
                updated_a = True
            if q_changed:
                self.solver.update_q(q)
                updated_q = True
            if b_changed:
                self.solver.update_b(b)
                updated_b = True
            if not warm or mode == "persist_factor":
                self.solver.reset("factor" if mode == "persist_factor" else "cold")
            if warm:
                self.n_warm_starts += 1

        solution = self.solver.solve()
        status = str(solution.status)
        iterations = int(getattr(solution, "iter", 0) or 0)
        if self._should_restart(iterations, status) and not reconstruct:
            self.solver.reset("cold")
            self.n_restarts += 1
            restarted = True
            solution = self.solver.solve()
            status = str(solution.status)
            iterations = int(getattr(solution, "iter", 0) or 0)

        if "solved" not in status.lower():
            raise RuntimeError(f"COSMO.rs failed: {status} (iter={iterations})")

        x = np.asarray(solution.x, dtype=np.float64)
        y = np.asarray(solution.y, dtype=np.float64)
        step = (
            float("nan")
            if self._x is None
            else float(np.linalg.norm(x[: self._x.size] - self._x[: x.size]))
        )
        self._x = x.copy()
        self._y = y.copy()
        trace = CosmoFoldTrace(
            status=status,
            iterations=iterations,
            r_prim=float(getattr(solution, "r_prim", float("nan"))),
            r_dual=float(getattr(solution, "r_dual", float("nan"))),
            obj_val=float(getattr(solution, "obj_val", float("nan"))),
            setup_time=float(getattr(solution, "setup_time", 0.0)),
            solve_time=float(getattr(solution, "solve_time", 0.0)),
            factor_time=float(getattr(solution, "factor_time", 0.0)),
            iter_time=float(getattr(solution, "iter_time", 0.0)),
            proj_time=float(getattr(solution, "proj_time", 0.0)),
            reused_solver=reused,
            updated_p=updated_p,
            updated_a=updated_a,
            updated_q=updated_q,
            updated_b=updated_b,
            restarted=restarted,
            weight_step=step,
        )
        self.last_trace = trace
        self.traces.append(trace)
        return x


class MinVarianceCosmo:
    """Boxed mean-variance QP solved by persistent COSMO.rs."""

    def __init__(
        self,
        spec: MeanRiskSpec,
        n_assets: int,
        *,
        persist_mode: PersistMode | None = None,
        settings: dict[str, Any] | None = None,
        restart_policy: RestartPolicy = "status",
    ) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = _as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = _as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.l2 = float(spec.l2_coef)
        self.objective = spec.objective
        self.risk_aversion = float(spec.risk_aversion)
        self._p_dense = np.empty((n_assets, n_assets), dtype=np.float64)
        mode = persist_mode if persist_mode is not None else default_persist_mode(spec)
        self._workspace = PersistentCosmo(
            settings=settings or default_cosmo_settings(spec),
            persist_mode=mode,
            restart_policy=restart_policy,
        )
        self._build_pattern()

    @property
    def n_warm_starts(self) -> int:
        return self._workspace.n_warm_starts

    @property
    def last_iterations(self) -> int:
        trace = self._workspace.last_trace
        return 0 if trace is None else trace.iterations

    @property
    def last_trace(self) -> CosmoFoldTrace | None:
        return self._workspace.last_trace

    def _build_pattern(self) -> None:
        n = self.n_assets
        rows: list[list[tuple[int, float]]] = [[(j, 1.0) for j in range(n)]]
        rhs = [self.budget]
        nonneg: list[list[tuple[int, float]]] = []
        nonneg_b: list[float] = []
        for j in range(n):
            nonneg.append([(j, -1.0)])
            nonneg_b.append(-float(self.min_w[j]))
        for j in range(n):
            nonneg.append([(j, 1.0)])
            nonneg_b.append(float(self.max_w[j]))
        self._A = _rows_to_csc(rows + nonneg, n)
        self._b = np.asarray(rhs + nonneg_b, dtype=np.float64)
        self._q = np.zeros(n, dtype=np.float64)
        self._cones = [
            clarabel.ZeroConeT(1),
            clarabel.NonnegativeConeT(2 * n),
        ]

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
        P = _upper_csc(self._p_dense)
        q = self._q
        q_changed = False
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q = -np.ascontiguousarray(moments.mu, dtype=np.float64)
            q_changed = True
        x = self._workspace.solve(
            P,
            q,
            self._A,
            self._b,
            self._cones,
            warm=warm,
            p_changed=True,
            a_changed=False,
            q_changed=q_changed,
            b_changed=False,
        )
        return x.copy()


class ScenarioCosmo:
    """Boxed scenario LP / QP / SOCP / exp-cone solved by persistent COSMO.rs."""

    def __init__(
        self,
        spec: MeanRiskSpec,
        n_assets: int,
        n_observations: int,
        *,
        persist_mode: PersistMode | None = None,
        settings: dict[str, Any] | None = None,
        restart_policy: RestartPolicy = "status",
    ) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_observations = int(n_observations)
        mode = persist_mode if persist_mode is not None else default_persist_mode(spec)
        self._workspace = PersistentCosmo(
            settings=settings or default_cosmo_settings(spec),
            persist_mode=mode,
            restart_policy=restart_policy,
        )
        if spec.risk_measure is RiskMeasure.CVAR:
            self._pattern: CVaRClarabel | ScenarioClarabel = CVaRClarabel(
                spec, n_assets, n_observations
            )
            self._pattern.solver = None
        else:
            self._pattern = ScenarioClarabel(spec, n_assets, n_observations)
            self._pattern.solver = None

    @property
    def n_warm_starts(self) -> int:
        return self._workspace.n_warm_starts

    @property
    def last_iterations(self) -> int:
        trace = self._workspace.last_trace
        return 0 if trace is None else trace.iterations

    @property
    def last_trace(self) -> CosmoFoldTrace | None:
        return self._workspace.last_trace

    def _matrices(self, moments: FoldMoments):
        pattern = self._pattern
        if isinstance(pattern, CVaRClarabel):
            t = int(moments.n_observations or moments.returns.shape[0])
            if t != pattern.n_observations:
                pattern.n_observations = t
                pattern._build_pattern()
                self._workspace.solver = None
            pattern._bind_R(moments.returns)
            pattern._bind_q(moments)
            assert pattern._P is not None
            assert pattern._q is not None
            assert pattern._A is not None
            assert pattern._b is not None
            return (
                pattern._P,
                pattern._q,
                pattern._A,
                pattern._b,
                pattern._cones,
            )
        t = int(moments.n_observations)
        if t != pattern.n_observations:
            pattern.n_observations = t
            pattern.solver = None
            self._workspace.solver = None
        return pattern._problem(moments)

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        P, q, A, b, cones = self._matrices(moments)
        utility = self.spec.objective is ObjectiveFunction.MAXIMIZE_UTILITY
        x = self._workspace.solve(
            P,
            q,
            A,
            b,
            cones,
            warm=warm,
            p_changed=False,
            a_changed=True,
            q_changed=utility,
            b_changed=True,
        )
        return np.ascontiguousarray(x[: self.n_assets], dtype=np.float64)


def make_cosmo_engine(
    spec: MeanRiskSpec,
    *,
    n_assets: int,
    n_observations: int | None,
    persist_mode: PersistMode | None = None,
    settings: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = "status",
):
    """Construct a compact COSMO engine for ``spec``.

    ``persist_mode`` defaults to ``persist_full`` for variance and
    ``persist_factor`` for scenario risks (class B: ``update_a`` drops KKT).
    """
    if spec.risk_measure is RiskMeasure.VARIANCE:
        return MinVarianceCosmo(
            spec,
            n_assets,
            persist_mode=persist_mode,
            settings=settings,
            restart_policy=restart_policy,
        )
    if n_observations is None:
        raise ValueError(
            f"{spec.risk_measure.name} COSMO engine requires n_observations"
        )
    return ScenarioCosmo(
        spec,
        n_assets,
        n_observations,
        persist_mode=persist_mode,
        settings=settings,
        restart_policy=restart_policy,
    )
