"""Compile a CVXPY DPP problem into a Clarabel ProblemTemplate."""

from __future__ import annotations

import inspect
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from skfolio_accelerate.estimators.mean_risk_twin import TwinProblem
from skfolio_accelerate.ir import NumericInstance, ProblemTemplate


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
    n_cons = int(np.asarray(data["b"]).shape[0])
    return ProblemTemplate(
        structure_key=structure_key,
        problem=problem,
        param_prob=param_prob,
        cones=cones,
        n_vars=n_vars,
        n_cons=n_cons,
        weight_slice=slice(offset, offset + n_assets),
        parameters=twin.parameters,
        risk_measure=str(twin.risk_measure),
        n_observations=twin.n_observations,
        n_assets=n_assets,
        scale_objective=twin.scale_objective,
        scale_constraints=twin.scale_constraints,
    )


def instantiate(template: ProblemTemplate) -> NumericInstance:
    applied = _apply_parameters(template.param_prob)
    if len(applied) == 5:
        P, q, _offset, A, b = applied
    else:
        q, _offset, A, b = applied
        P = sp.csc_matrix((template.n_vars, template.n_vars))
    # get_problem_data stores A with Clarabel sign; apply_parameters does not.
    A = -A
    P = sp.triu(P).tocsc()
    A = sp.csc_matrix(A)
    q = np.asarray(q, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return NumericInstance(
        P_data=np.asarray(P.data, dtype=float),
        q=q,
        A_data=np.asarray(A.data, dtype=float),
        b=b,
        P_shape=P.shape,
        A_shape=A.shape,
        P_indices=np.asarray(P.indices, dtype=np.int32),
        P_indptr=np.asarray(P.indptr, dtype=np.int32),
        A_indices=np.asarray(A.indices, dtype=np.int32),
        A_indptr=np.asarray(A.indptr, dtype=np.int32),
    )


def instance_to_scipy(instance: NumericInstance):
    P = sp.csc_matrix(
        (instance.P_data, instance.P_indices, instance.P_indptr),
        shape=instance.P_shape,
    )
    A = sp.csc_matrix(
        (instance.A_data, instance.A_indices, instance.A_indptr),
        shape=instance.A_shape,
    )
    return P, instance.q, A, instance.b
