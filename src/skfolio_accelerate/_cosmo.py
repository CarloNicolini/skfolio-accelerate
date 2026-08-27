"""Optional COSMO.jl compact engines behind a persistent juliacall runtime.

COSMO is not a default dependency. Engines are constructed only when
``MeanRisk(solver="COSMO")`` is compact-eligible and the Julia runtime plus
``COSMO.jl`` are installed. See :func:`cosmo_available`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from typing import Any

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
)
from skfolio_accelerate.moments import FoldMoments

_BRIDGE_MODULE = "CosmoBridge"


class CosmoRuntimeError(RuntimeError):
    """Raised when the optional COSMO.jl runtime cannot be started."""


def _full_csc(matrix: NDArray[np.float64]) -> sp.csc_matrix:
    """Dense CSC that keeps explicit zeros so COSMO nzval updates stay valid."""
    dense = np.ascontiguousarray(matrix, dtype=np.float64)
    n = int(dense.shape[0])
    data = np.asfortranarray(dense, dtype=np.float64).ravel(order="F")
    indices = np.tile(np.arange(n, dtype=np.int32), n)
    indptr = np.arange(0, n * n + 1, n, dtype=np.int32)
    return sp.csc_matrix((data, indices, indptr), shape=(n, n))


def _csc_parts(
    matrix: sp.spmatrix,
) -> tuple[int, int, NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return 1-based CSC buffers for ``SparseMatrixCSC`` construction in Julia."""
    csc = matrix.tocsc()
    csc.sort_indices()
    rows, cols = csc.shape
    colptr = np.asarray(csc.indptr, dtype=np.int64) + 1
    rowval = np.asarray(csc.indices, dtype=np.int64) + 1
    nzval = np.ascontiguousarray(csc.data, dtype=np.float64)
    return int(rows), int(cols), colptr, rowval, nzval


def clarabel_cones_to_cosmo(
    cones: list[Any],
) -> tuple[int, int, NDArray[np.int64], int]:
    """Map Clarabel cone objects to COSMO ``set!`` dimensions.

    Parameters
    ----------
    cones : list
        Clarabel cone list as produced by the compact scenario engines.

    Returns
    -------
    n_zero, n_nonneg, soc_dims, n_exp : tuple
        COSMO cone counts / sizes in the same row order as ``cones``.
    """
    n_zero = 0
    n_nonneg = 0
    soc: list[int] = []
    n_exp = 0
    for cone in cones:
        name = type(cone).__name__
        dim = int(getattr(cone, "dim", getattr(cone, "n", 0)))
        if name == "ZeroConeT":
            n_zero += dim
        elif name in {"NonnegativeConeT", "NonNegativeConeT"}:
            n_nonneg += dim
        elif name == "SecondOrderConeT":
            soc.append(dim)
        elif name == "ExponentialConeT":
            n_exp += 1
        else:
            raise TypeError(f"Unsupported Clarabel cone {name}")
    return n_zero, n_nonneg, np.asarray(soc, dtype=np.int64), n_exp


def _bang(module: Any, name: str) -> Any:
    """Resolve a Julia ``foo!`` function through juliacall's naming."""
    func = getattr(module, f"{name}!", None)
    if func is not None:
        return func
    func = getattr(module, name, None)
    if func is not None:
        return func
    raise AttributeError(f"{type(module).__name__} has no function {name}!")


class CosmoRuntime:
    """Process-local juliacall session with one included ``CosmoBridge`` module."""

    def __init__(self) -> None:
        os.environ.setdefault("JULIA_NUM_THREADS", "1")
        try:
            from juliacall import Main as jl
        except ImportError as error:
            raise CosmoRuntimeError(
                "juliacall is not installed; install skfolio-accelerate[cosmo]"
            ) from error
        self.jl = jl
        try:
            jl.seval("using COSMO")
        except Exception:
            try:
                jl.seval("using Pkg")
                jl.seval('Pkg.add("COSMO")')
                jl.seval("using COSMO")
            except Exception as error:
                raise CosmoRuntimeError(
                    "COSMO.jl is not available in this Julia environment. "
                    'Install Julia, then add COSMO.jl with Pkg.add("COSMO").'
                ) from error
        jl.seval("using SparseArrays")
        bridge = files("skfolio_accelerate") / "julia" / "cosmo_bridge.jl"
        jl.include(str(bridge))
        self.bridge = getattr(jl, _BRIDGE_MODULE)
        self._warmed = False

    def warmup(self) -> None:
        """JIT-compile a 1-variable QP once per process."""
        if self._warmed:
            return
        _bang(self.bridge, "warmup")()
        self._warmed = True

    def make_workspace(
        self,
        *,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-8,
        max_iter: int = 10_000,
        empty_accelerator: bool = False,
    ) -> Any:
        return self.bridge.make_workspace(
            eps_abs=float(eps_abs),
            eps_rel=float(eps_rel),
            max_iter=int(max_iter),
            empty_accelerator=bool(empty_accelerator),
        )

    def solve_qp(
        self,
        workspace: Any,
        P: sp.spmatrix,
        q: NDArray[np.float64],
        A: sp.spmatrix,
        b: NDArray[np.float64],
        *,
        n_zero: int,
        n_nonneg: int,
        soc_dims: NDArray[np.int64] | list[int],
        n_exp: int,
        box_l: NDArray[np.float64] | None,
        box_u: NDArray[np.float64] | None,
        warm: bool,
        update_p: bool,
        update_a: bool,
    ) -> NDArray[np.float64]:
        _p_rows, n_vars, p_colptr, p_rowval, p_nzval = _csc_parts(P)
        n_cons, _a_cols, a_colptr, a_rowval, a_nzval = _csc_parts(A)
        soc = np.asarray(soc_dims, dtype=np.int64)
        lower = (
            np.ascontiguousarray(box_l, dtype=np.float64)
            if box_l is not None and np.size(box_l) > 0
            else np.empty(0, dtype=np.float64)
        )
        upper = (
            np.ascontiguousarray(box_u, dtype=np.float64)
            if box_u is not None and np.size(box_u) > 0
            else np.empty(0, dtype=np.float64)
        )
        x = _bang(self.bridge, "solve_qp")(
            workspace,
            int(n_cons),
            int(n_vars),
            p_colptr,
            p_rowval,
            p_nzval,
            np.ascontiguousarray(q, dtype=np.float64),
            a_colptr,
            a_rowval,
            a_nzval,
            np.ascontiguousarray(b, dtype=np.float64),
            int(n_zero),
            int(n_nonneg),
            soc,
            int(n_exp),
            lower,
            upper,
            bool(warm),
            bool(update_p),
            bool(update_a),
        )
        return np.asarray(x, dtype=np.float64)


@lru_cache(maxsize=1)
def _runtime() -> CosmoRuntime:
    return CosmoRuntime()


def _cosmo_runtime() -> CosmoRuntime:
    """Return the process-local COSMO runtime, starting Julia on first use."""
    return _runtime()


@lru_cache(maxsize=1)
def cosmo_available() -> bool:
    """``True`` when juliacall and COSMO.jl can assemble a workspace.

    The result is cached. A failed first probe (missing juliacall, Julia, or
    COSMO.jl) is not retried inside the same process.
    """
    try:
        _cosmo_runtime().warmup()
    except Exception:
        return False
    return True


_COSMO_LP_RISKS = frozenset(
    {
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
    }
)
_COSMO_EXP_RISKS = frozenset({RiskMeasure.EVAR, RiskMeasure.EDAR})


def _workspace_options(spec: MeanRiskSpec) -> dict[str, Any]:
    """ADMM settings that actually terminate on each compact family.

    Variance / CVaR / SOC keep tight residuals. Scenario LPs disable Anderson
    acceleration (it cycles and blocks rho updates). Exponential cones use
    COSMO's default 1e-5 tolerances so ``optimize!`` reports ``Solved``.
    """
    risk = spec.risk_measure
    if risk in _COSMO_LP_RISKS:
        return {
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 25_000,
            "empty_accelerator": True,
        }
    if risk in _COSMO_EXP_RISKS:
        return {
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 15_000,
            "empty_accelerator": False,
        }
    return {
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "max_iter": 10_000,
        "empty_accelerator": False,
    }


def _workspace_assembled(workspace: Any) -> bool:
    return bool(workspace.assembled)


def _require_solved(workspace: Any, *, label: str) -> None:
    status = str(workspace.status).lower()
    if "solved" not in status:
        raise RuntimeError(f"COSMO {label} failed: {workspace.status}")


class MinVarianceCOSMO:
    """Boxed mean-variance QP on a persistent COSMO.jl ADMM workspace."""

    def __init__(self, spec: MeanRiskSpec, n_assets: int) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.min_w = _as_bounds(spec.min_weights, n_assets, 0.0)
        self.max_w = _as_bounds(spec.max_weights, n_assets, 1.0)
        self.budget = float(spec.budget)
        self.l2 = float(spec.l2_coef)
        self.objective = spec.objective
        self.risk_aversion = float(spec.risk_aversion)
        self._p_dense = np.empty((n_assets, n_assets), dtype=np.float64)
        self._q = np.zeros(n_assets, dtype=np.float64)
        # COSMO set! form: A x + s = b. Equality then Box(min_w, max_w) on w.
        self._A = sp.vstack(
            [
                sp.csr_matrix(np.ones((1, n_assets))),
                -sp.eye(n_assets, format="csr"),
            ]
        ).tocsc()
        self._b = np.concatenate([[self.budget], np.zeros(n_assets, dtype=np.float64)])
        runtime = _cosmo_runtime()
        runtime.warmup()
        self._ws = runtime.make_workspace(**_workspace_options(spec))
        self.n_warm_starts = 0
        self.last_iterations = 0

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        """Solve the mean-variance QP for one training window.

        Parameters
        ----------
        moments : FoldMoments
            Must provide ``covariance`` of shape ``(n_assets, n_assets)`` and,
            for maximize-utility, ``mu``.

        warm : bool, default=True
            Reuse the previous COSMO iterate when ``True``.

        Returns
        -------
        weights : ndarray of shape (n_assets,)
            Optimal portfolio weights.
        """
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
        self._p_dense[np.diag_indices(n)] += 2.0 * self.l2
        q = self._q
        if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY:
            q = -np.ascontiguousarray(moments.mu, dtype=np.float64)
        use_warm = bool(warm and _workspace_assembled(self._ws))
        x = _cosmo_runtime().solve_qp(
            self._ws,
            _full_csc(self._p_dense),
            q,
            self._A,
            self._b,
            n_zero=1,
            n_nonneg=0,
            soc_dims=[],
            n_exp=0,
            box_l=self.min_w,
            box_u=self.max_w,
            warm=use_warm,
            update_p=True,
            update_a=False,
        )
        if use_warm:
            self.n_warm_starts += 1
        _require_solved(self._ws, label="variance")
        self.last_iterations = int(self._ws.n_iter)
        return np.asarray(x[:n], dtype=np.float64).copy()


class CVaRCOSMO(CVaRClarabel):
    """Boxed CVaR LP on a persistent COSMO.jl workspace.

    Reuses :class:`CVaRClarabel` CSC pattern construction (including the ``-R``
    slices) and solves with COSMO instead of Clarabel.
    """

    def __init__(self, spec: MeanRiskSpec, n_assets: int, n_observations: int) -> None:
        super().__init__(spec, n_assets, n_observations)
        runtime = _cosmo_runtime()
        runtime.warmup()
        self._ws = runtime.make_workspace(**_workspace_options(spec))
        self.last_iterations = 0

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations or moments.returns.shape[0])
        if t != self.n_observations:
            self.n_observations = t
            self._build_pattern()
            self._ws = _cosmo_runtime().make_workspace(**_workspace_options(self.spec))
        self._bind_R(moments.returns)
        self._bind_q(moments)
        assert self._A is not None and self._q is not None and self._b is not None
        n_zero, n_nonneg, soc, n_exp = clarabel_cones_to_cosmo(self._cones)
        use_warm = bool(warm and _workspace_assembled(self._ws))
        x = _cosmo_runtime().solve_qp(
            self._ws,
            self._P,
            self._q,
            self._A,
            self._b,
            n_zero=n_zero,
            n_nonneg=n_nonneg,
            soc_dims=soc,
            n_exp=n_exp,
            box_l=None,
            box_u=None,
            warm=use_warm,
            update_p=False,
            update_a=True,
        )
        if use_warm:
            self.n_warm_starts += 1
        _require_solved(self._ws, label="CVaR")
        self.last_iterations = int(self._ws.n_iter)
        self._x = np.asarray(x, dtype=np.float64)
        return self._x[: self.n_assets].copy()


class ScenarioCOSMO(ScenarioClarabel):
    """Scenario LP / SOCP / exponential-cone engine on COSMO.jl."""

    def __init__(self, spec: MeanRiskSpec, n_assets: int, n_observations: int) -> None:
        super().__init__(spec, n_assets, n_observations)
        runtime = _cosmo_runtime()
        runtime.warmup()
        self._ws = runtime.make_workspace(**_workspace_options(spec))
        self.last_iterations = 0

    def solve(self, moments: FoldMoments, *, warm: bool = True) -> NDArray[np.float64]:
        t = int(moments.n_observations)
        if t != self.n_observations:
            self.n_observations = t
            self._ws = _cosmo_runtime().make_workspace(**_workspace_options(self.spec))
        P, q, A, b, cones = self._problem(moments)
        n_zero, n_nonneg, soc, n_exp = clarabel_cones_to_cosmo(cones)
        use_warm = bool(warm and _workspace_assembled(self._ws))
        x = _cosmo_runtime().solve_qp(
            self._ws,
            P,
            q,
            A,
            b,
            n_zero=n_zero,
            n_nonneg=n_nonneg,
            soc_dims=soc,
            n_exp=n_exp,
            box_l=None,
            box_u=None,
            warm=use_warm,
            update_p=True,
            update_a=True,
        )
        if use_warm:
            self.n_warm_starts += 1
        _require_solved(self._ws, label=self.spec.risk_measure.name)
        self.last_iterations = int(self._ws.n_iter)
        return np.asarray(x[: self.n_assets], dtype=np.float64)


def make_cosmo_engine(spec: MeanRiskSpec, *, n_assets: int, n_observations: int | None):
    """Construct the COSMO compact engine for ``spec``."""
    risk = spec.risk_measure
    if risk is RiskMeasure.VARIANCE:
        return MinVarianceCOSMO(spec, n_assets)
    if n_observations is None:
        raise ValueError(f"{risk.name} engine requires n_observations")
    if risk is RiskMeasure.CVAR:
        return CVaRCOSMO(spec, n_assets, n_observations)
    return ScenarioCOSMO(spec, n_assets, n_observations)
