"""Linear MeanRisk constraints compiled once into OSQP ``A``, ``l``, ``u``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import input_to_array

from skfolio_accelerate.compact._util import MeanRiskSpec, as_bounds

INF = np.inf


def _groups_matrix(spec: MeanRiskSpec, n: int):
    names = spec.asset_names
    groups = spec.groups
    if groups is None:
        if names is None:
            raise ValueError(
                "If linear_constraints is provided you must provide either groups "
                "or X as a DataFrame with asset names in columns"
            )
        return np.asarray([list(names)])
    return input_to_array(
        items=groups,
        n_assets=n,
        fill_value="",
        dim=2,
        assets_names=None if names is None else np.asarray(names),
        name="groups",
    )


def compile_osqp_constraints(spec: MeanRiskSpec, n: int):
    """Budget, bounds, L1 slacks, and linear inequalities for an n-asset QP."""
    l1 = float(spec.l1_coef or 0.0)
    n_extra = n if l1 > 0.0 else 0
    min_w = as_bounds(spec.min_weights, n, 0.0, names=spec.asset_names)
    max_w = as_bounds(spec.max_weights, n, 1.0, names=spec.asset_names)
    parts: list[sp.csc_matrix] = []
    lower: list[NDArray[np.float64]] = []
    upper: list[NDArray[np.float64]] = []

    def add(block, lo, hi, *, weights_only: bool = True) -> None:
        matrix = sp.csc_matrix(np.asarray(block) if not sp.issparse(block) else block)
        if weights_only and n_extra:
            matrix = sp.hstack([matrix, sp.csc_matrix((matrix.shape[0], n_extra))]).tocsc()
        parts.append(matrix.tocsc())
        lower.append(np.atleast_1d(np.asarray(lo, dtype=np.float64)))
        upper.append(np.atleast_1d(np.asarray(hi, dtype=np.float64)))

    add(sp.eye(n, format="csc"), min_w, max_w)
    ones = np.ones((1, n))
    if spec.budget is not None:
        add(ones, spec.budget, spec.budget)
    else:
        lo = -INF if spec.min_budget is None else float(spec.min_budget)
        hi = INF if spec.max_budget is None else float(spec.max_budget)
        add(ones, lo, hi)

    if n_extra:
        eye = sp.eye(n, format="csc")
        zeros = sp.csc_matrix((n, n))
        add(sp.hstack([eye, -eye]), np.full(n, -INF), np.zeros(n), weights_only=False)
        add(sp.hstack([-eye, -eye]), np.full(n, -INF), np.zeros(n), weights_only=False)
        add(sp.hstack([zeros, eye]), np.zeros(n), np.full(n, INF), weights_only=False)

    left, right = spec.left_inequality, spec.right_inequality
    if left is not None or right is not None:
        if left is None or right is None:
            raise ValueError("left_inequality and right_inequality must be provided together")
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.ndim != 2 or left.shape[1] != n:
            raise ValueError(f"left_inequality must have shape (n_constraints, {n})")
        if right.ndim != 1 or right.shape[0] != left.shape[0]:
            raise ValueError("right_inequality must match left_inequality rows")
        add(left, np.full(left.shape[0], -INF), right)

    if spec.linear_constraints is not None:
        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=_groups_matrix(spec, n),
            equations=spec.linear_constraints,
            raise_if_group_missing=False,
        )
        if len(a_eq):
            add(np.asarray(a_eq, dtype=np.float64), b_eq, b_eq)
        if len(a_ineq):
            add(np.asarray(a_ineq, dtype=np.float64), np.full(len(b_ineq), -INF), b_ineq)

    mu_row = None
    if spec.min_return is not None:
        mu_row = sum(int(part.shape[0]) for part in parts)
        add(np.ones((1, n)), float(np.asarray(spec.min_return).reshape(())), INF)

    A = sp.vstack(parts, format="csc")
    A.sort_indices()
    return A, np.concatenate(lower), np.concatenate(upper), n_extra, mu_row, min_w, max_w
