"""Compile a CVXPY DPP problem into a Clarabel ProblemTemplate."""

from __future__ import annotations

import inspect
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import CONSTANT_ID

from skfolio_accelerate.estimators.mean_risk_twin import TwinProblem
from skfolio_accelerate.ir import NumericInstance, ProblemTemplate

# CVXPY Parameter names whose columns we treat as numerical (not fold data).
_NUMERICAL_PARAM_NAMES = {
    "l1_coef",
    "l2_coef",
    "risk_aversion",
    "min_return",
    "cvar_coef",
}


def _problem_data_kwargs() -> dict[str, Any]:
    signature = inspect.signature(cp.Problem.get_problem_data)
    if "enforce_dpp" in signature.parameters:
        return {"enforce_dpp": True}
    return {}


def _apply_parameters(param_prob) -> tuple:
    signature = inspect.signature(param_prob.apply_parameters)
    kwargs: dict[str, Any] = {}
    if "keep_zeros" in signature.parameters:
        kwargs["keep_zeros"] = True
    if "quad_obj" in signature.parameters:
        kwargs["quad_obj"] = True
    return param_prob.apply_parameters(**kwargs)


def dims_to_clarabel_cones(cone_dims) -> list:
    from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import (
        dims_to_solver_cones,
    )

    return dims_to_solver_cones(cone_dims)


def cones_to_spec(cones: list) -> list[tuple[str, int]]:
    spec: list[tuple[str, int]] = []
    for cone in cones:
        name = type(cone).__name__.lower()
        if "zero" in name:
            spec.append(("zero", int(cone.dim) if hasattr(cone, "dim") else int(cone)))
        elif "nonneg" in name:
            dim = getattr(cone, "dim", None)
            spec.append(("nonnegative", int(dim if dim is not None else cone)))
        elif "second" in name or "soc" in name:
            dim = getattr(cone, "dim", None)
            spec.append(("soc", int(dim if dim is not None else cone)))
        elif "exp" in name:
            spec.append(("exp", 3))
        elif "psd" in name:
            dim = getattr(cone, "dim", None)
            spec.append(("psd", int(dim if dim is not None else cone)))
        else:
            raise ValueError(f"Unsupported cone type {type(cone)!r}")
    return spec


def _clarabel_pqab(param_prob, n_vars: int):
    applied = _apply_parameters(param_prob)
    if len(applied) == 5:
        P, q, _offset, A, b = applied
    else:
        q, _offset, A, b = applied
        P = sp.csc_matrix((n_vars, n_vars))
    # get_problem_data stores A with Clarabel sign; apply_parameters does not.
    A = sp.csc_matrix(-A)
    P = sp.triu(P).tocsc()
    q = np.ascontiguousarray(q, dtype=np.float64).ravel()
    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    return P, q, A, b


def _set_pattern(template: ProblemTemplate, P, A) -> None:
    template.P_shape = P.shape
    template.A_shape = A.shape
    template.P_indices = np.asarray(P.indices, dtype=np.int32)
    template.P_indptr = np.asarray(P.indptr, dtype=np.int32)
    template.A_indices = np.asarray(A.indices, dtype=np.int32)
    template.A_indptr = np.asarray(A.indptr, dtype=np.int32)


def extract_problem_template(twin: TwinProblem, structure_key: str) -> ProblemTemplate:
    problem = twin.problem
    if not problem.is_dcp(dpp=True):
        raise ValueError("Problem is not DPP; cannot compile a reusable template")
    data, _chain, _inv = problem.get_problem_data(
        cp.CLARABEL,
        **_problem_data_kwargs(),
    )
    param_prob = data[cp.settings.PARAM_PROB]
    cones = dims_to_clarabel_cones(data["dims"])
    var_map = getattr(param_prob, "var_id_to_col", None)
    if var_map is None:
        var_map = param_prob.var_id_to_col
    offset = int(var_map[twin.weights.id])
    n_assets = int(twin.weights.size)
    n_vars = int(param_prob.x.size)
    P, _q, A, b = _clarabel_pqab(param_prob, n_vars)
    return ProblemTemplate(
        structure_key=structure_key,
        problem=problem,
        param_prob=param_prob,
        cones=cones,
        n_vars=n_vars,
        n_cons=int(b.shape[0]),
        weight_slice=slice(offset, offset + n_assets),
        parameters=twin.parameters,
        risk_measure=str(twin.risk_measure),
        n_observations=twin.n_observations,
        n_assets=n_assets,
        scale_objective=twin.scale_objective,
        scale_constraints=twin.scale_constraints,
        P_shape=P.shape,
        A_shape=A.shape,
        P_indices=np.asarray(P.indices, dtype=np.int32),
        P_indptr=np.asarray(P.indptr, dtype=np.int32),
        A_indices=np.asarray(A.indices, dtype=np.int32),
        A_indptr=np.asarray(A.indptr, dtype=np.int32),
    )


class Instantiator:
    """Reuse fold-constant A, b and apply numerical params as a saxpy on P (and q).

    ``apply_parameters`` rebuilds the full CSC ``A`` (O(n²) for VARIANCE, O(Tn)
    for CVaR) even when only ``l2_coef`` changes. Those maps do not depend on
    ``l2``. After one full apply per fold we only form ``P`` (diagonal, n nnz)
    from cached tensor columns.
    """

    def __init__(self, template: ProblemTemplate) -> None:
        self.template = template
        self._token: object | None = object()
        self._A_data: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._P_const: np.ndarray | None = None
        self._P_terms: list[tuple[int, np.ndarray]] = []
        self._q_const: np.ndarray | None = None
        self._q_terms: list[tuple[int, np.ndarray]] = []
        self._scalar_cols: list[tuple[Any, int]] = []
        self._A_numerical = True
        self._q_numerical = False
        self._use_p_map = False
        self._warmed = False

    def _col_nnz(self, mat, col: int) -> int:
        if mat is None:
            return 0
        csc = mat.tocsc()
        if col < 0 or col + 1 >= csc.indptr.size:
            return 0
        return int(csc.indptr[col + 1] - csc.indptr[col])

    def _warm(self) -> None:
        if self._warmed:
            return
        pp = self.template.param_prob
        if getattr(pp, "reduced_A", None) is not None:
            pp.reduced_A.cache(keep_zeros=True)
        if getattr(pp, "reduced_P", None) is not None:
            pp.reduced_P.cache(keep_zeros=True)
        n_col = int(pp.total_param_size) + 1
        const_col = int(pp.param_id_to_col.get(CONSTANT_ID, n_col - 1))
        RA = getattr(pp.reduced_A, "reduced_mat", None)
        RP = getattr(pp.reduced_P, "reduced_mat", None)
        Rq = pp.q
        num_cols: list[int] = []
        self._scalar_cols = []
        for param in pp.parameters:
            name = param.name()
            if name not in _NUMERICAL_PARAM_NAMES:
                continue
            col = int(pp.param_id_to_col[param.id])
            self._scalar_cols.append((param, col))
            num_cols.append(col)

        self._A_numerical = any(self._col_nnz(RA, col) > 0 for col in num_cols)
        self._q_numerical = any(self._col_nnz(Rq, col) > 0 for col in num_cols)
        p_numerical = any(self._col_nnz(RP, col) > 0 for col in num_cols)

        if RP is not None and p_numerical:
            self._P_const = self._p_from_unit(const_col)
            self._P_terms = [
                (col, self._p_from_unit(col))
                for col in num_cols
                if self._col_nnz(RP, col) > 0
            ]
            self._use_p_map = (
                self._P_const.size == int(self.template.P_indices.size)
                and all(term.size == self._P_const.size for _, term in self._P_terms)
            )
        if Rq is not None and self._q_numerical:
            self._q_const = self._q_from_unit(const_col)
            self._q_terms = [
                (col, self._q_from_unit(col))
                for col in num_cols
                if self._col_nnz(Rq, col) > 0
            ]
        self._warmed = True

    def _p_from_unit(self, col: int) -> np.ndarray:
        pp = self.template.param_prob
        e = np.zeros(int(pp.total_param_size) + 1, dtype=np.float64)
        e[col] = 1.0
        P, _ = pp.reduced_P.get_matrix_from_tensor(e, with_offset=False)
        P = sp.triu(P).tocsc()
        return np.ascontiguousarray(P.data, dtype=np.float64)

    def _q_from_unit(self, col: int) -> np.ndarray:
        from cvxpy.cvxcore.python import canonInterface

        pp = self.template.param_prob
        e = np.zeros(int(pp.total_param_size) + 1, dtype=np.float64)
        e[col] = 1.0
        q, _d = canonInterface.get_matrix_from_tensor(
            pp.q, e, pp.x.size, with_offset=True
        )
        return np.ascontiguousarray(q.toarray().flatten(), dtype=np.float64)

    def _scalars(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for param, col in self._scalar_cols:
            value = param.value
            out[col] = 0.0 if value is None else float(np.asarray(value).reshape(-1)[0])
        return out

    def _p_from_scalars(self, scalars: dict[int, float]) -> np.ndarray:
        p = np.array(self._P_const, dtype=np.float64, copy=True)
        for col, term in self._P_terms:
            p += scalars[col] * term
        return p

    def _q_from_scalars(self, scalars: dict[int, float]) -> np.ndarray:
        q = np.array(self._q_const, dtype=np.float64, copy=True)
        for col, term in self._q_terms:
            q += scalars[col] * term
        return q

    def _full(self) -> NumericInstance:
        template = self.template
        P, q, A, b = _clarabel_pqab(template.param_prob, template.n_vars)
        if (
            int(template.P_indices.size) != int(P.indices.size)
            or int(template.A_indices.size) != int(A.indices.size)
        ):
            _set_pattern(template, P, A)
        return NumericInstance(
            P_data=np.ascontiguousarray(P.data, dtype=np.float64),
            q=q,
            A_data=np.ascontiguousarray(A.data, dtype=np.float64),
            b=b,
        )

    def instantiate(self, data_token: object | None = None) -> NumericInstance:
        self._warm()
        scalars = self._scalars()
        reuse_data = (
            data_token is not None
            and data_token == self._token
            and self._A_data is not None
            and not self._A_numerical
        )
        if not reuse_data:
            instance = self._full()
            self._token = data_token
            self._A_data = instance.A_data
            self._b = instance.b
            self._q = instance.q
            if self._use_p_map:
                p_fast = self._p_from_scalars(scalars)
                if p_fast.shape != instance.P_data.shape or not np.allclose(
                    p_fast, instance.P_data, rtol=0.0, atol=1e-12
                ):
                    self._use_p_map = False
            if self._q_numerical and self._q_terms:
                q_fast = self._q_from_scalars(scalars)
                if q_fast.shape != instance.q.shape or not np.allclose(
                    q_fast, instance.q, rtol=0.0, atol=1e-12
                ):
                    self._q_numerical = False
            return instance

        p_data = (
            self._p_from_scalars(scalars)
            if self._use_p_map
            else self._full().P_data
        )
        q = (
            self._q_from_scalars(scalars)
            if self._q_numerical and self._q_terms
            else self._q
        )
        return NumericInstance(
            P_data=p_data,
            q=q,
            A_data=self._A_data,
            b=self._b,
        )


def instantiate(
    template: ProblemTemplate, *, data_token: object | None = None
) -> NumericInstance:
    cache = template.instantiator
    if cache is None:
        cache = Instantiator(template)
        template.instantiator = cache
    return cache.instantiate(data_token)


def instance_to_scipy(template: ProblemTemplate, instance: NumericInstance):
    P = sp.csc_matrix(
        (instance.P_data, template.P_indices, template.P_indptr),
        shape=template.P_shape,
    )
    A = sp.csc_matrix(
        (instance.A_data, template.A_indices, template.A_indptr),
        shape=template.A_shape,
    )
    return P, instance.q, A, instance.b
