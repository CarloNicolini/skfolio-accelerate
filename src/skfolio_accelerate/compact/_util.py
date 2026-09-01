"""Shared numeric helpers for compact OSQP / Clarabel engines."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

SCENARIO_RISKS = frozenset(
    getattr(RiskMeasure, name)
    for name in (
        "SEMI_VARIANCE SEMI_DEVIATION MEAN_ABSOLUTE_DEVIATION "
        "FIRST_LOWER_PARTIAL_MOMENT WORST_REALIZATION CVAR EVAR "
        "MAX_DRAWDOWN AVERAGE_DRAWDOWN CDAR EDAR"
    ).split()
)


@dataclass(frozen=True, slots=True)
class MeanRiskSpec:
    risk_measure: RiskMeasure
    objective: ObjectiveFunction
    l2_coef: float
    l1_coef: float
    risk_aversion: float
    cvar_beta: float
    evar_beta: float
    cdar_beta: float
    edar_beta: float
    min_acceptable_return: Any
    min_weights: Any
    max_weights: Any
    budget: float | None
    min_budget: float | None
    max_budget: float | None
    min_return: float | None
    left_inequality: Any
    right_inequality: Any
    linear_constraints: Any
    groups: Any
    asset_names: tuple[str, ...] | None
    management_fees: Any
    max_long: float | None
    max_short: float | None

    def needs_returns(self) -> bool:
        return self.objective is not ObjectiveFunction.MAXIMIZE_RETURN and self.risk_measure not in {
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
        }

    def risk_scale(self) -> float:
        return self.risk_aversion if self.objective is ObjectiveFunction.MAXIMIZE_UTILITY else 1.0


def estimator_spec(estimator, *, names: tuple[str, ...] | None = None) -> MeanRiskSpec:
    min_return = estimator.min_return
    if min_return is not None and np.ndim(min_return) == 0:
        min_return = float(min_return)
    return MeanRiskSpec(
        risk_measure=estimator.risk_measure,
        objective=estimator.objective_function,
        l2_coef=float(estimator.l2_coef or 0.0),
        l1_coef=float(estimator.l1_coef or 0.0),
        risk_aversion=float(estimator.risk_aversion or 1.0),
        cvar_beta=float(estimator.cvar_beta or 0.95),
        evar_beta=float(estimator.evar_beta or 0.95),
        cdar_beta=float(estimator.cdar_beta or 0.95),
        edar_beta=float(estimator.edar_beta or 0.95),
        min_acceptable_return=estimator.min_acceptable_return,
        min_weights=estimator.min_weights,
        max_weights=estimator.max_weights,
        budget=None if estimator.budget is None else float(estimator.budget),
        min_budget=None if estimator.min_budget is None else float(estimator.min_budget),
        max_budget=None if estimator.max_budget is None else float(estimator.max_budget),
        min_return=min_return,
        left_inequality=estimator.left_inequality,
        right_inequality=estimator.right_inequality,
        linear_constraints=estimator.linear_constraints,
        groups=estimator.groups,
        asset_names=names,
        management_fees=estimator.management_fees,
        max_long=None if estimator.max_long is None else float(estimator.max_long),
        max_short=None if estimator.max_short is None else float(estimator.max_short),
    )


def as_bounds(
    value: Any, n: int, default: float, *, names: tuple[str, ...] | None = None
) -> NDArray[np.float64]:
    if value is None:
        return np.full(n, default, dtype=np.float64)
    if type(value) is dict:
        if names is None:
            raise ValueError("dict weight bounds require asset names")
        out = np.full(n, default, dtype=np.float64)
        index = {name: i for i, name in enumerate(names)}
        for key, item in value.items():
            out[index[str(key)]] = float(item)
        return out
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(n), dtype=np.float64)


def fee_vector(
    value: Any, n: int, *, names: tuple[str, ...] | None = None
) -> NDArray[np.float64]:
    return as_bounds(value, n, 0.0, names=names)


@lru_cache(maxsize=32)
def upper_indices(n: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    cols = np.repeat(np.arange(n, dtype=np.intp), np.arange(1, n + 1))
    rows = np.concatenate([np.arange(col + 1, dtype=np.intp) for col in range(n)])
    rows.flags.writeable = False
    cols.flags.writeable = False
    return rows, cols


@lru_cache(maxsize=32)
def identity(n: int) -> NDArray[np.float64]:
    eye = np.eye(n, dtype=np.float64)
    eye.flags.writeable = False
    return eye


def upper_data(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, cols = upper_indices(int(matrix.shape[0]))
    return matrix[rows, cols]


def upper_csc(matrix: NDArray[np.float64]) -> sp.csc_matrix:
    n = int(matrix.shape[0])
    rows, _ = upper_indices(n)
    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(np.arange(1, n + 1, dtype=np.int32))
    return sp.csc_matrix(
        (upper_data(matrix), np.asarray(rows, dtype=np.int32), indptr),
        shape=(n, n),
    )


def rows_to_csc(rows: list[list[tuple[int, float]]], n_variables: int) -> sp.csc_matrix:
    data, row_indices, columns = [], [], []
    for row, entries in enumerate(rows):
        for column, value in entries:
            row_indices.append(row)
            columns.append(column)
            data.append(value)
    matrix = sp.csc_matrix(
        (np.asarray(data, dtype=np.float64), (row_indices, columns)),
        shape=(len(rows), n_variables),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def diagonal_quadratic(n_variables: int, n_assets: int, l2_coef: float) -> sp.csc_matrix:
    if l2_coef == 0:
        return sp.csc_matrix((n_variables, n_variables))
    indices = np.arange(n_assets, dtype=np.int32)
    indptr = np.concatenate(
        [
            np.arange(n_assets + 1, dtype=np.int32),
            np.full(n_variables - n_assets, n_assets, dtype=np.int32),
        ]
    )
    return sp.csc_matrix(
        (np.full(n_assets, 2.0 * l2_coef, dtype=np.float64), indices, indptr),
        shape=(n_variables, n_variables),
    )


def clarabel_settings() -> clarabel.DefaultSettings:
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    for name, value in (
        ("presolve_enable", False),
        ("chordal_decomposition_enable", False),
        ("input_sparse_dropzeros", False),
        ("max_threads", 1),
        ("tol_gap_abs", 1e-9),
        ("tol_gap_rel", 1e-9),
    ):
        if hasattr(settings, name):
            setattr(settings, name, value)
    return settings


def clarabel_try_update(solver, P, q, A, b, cones, *, update: dict) -> tuple:
    settings = clarabel_settings()
    if solver is None:
        return clarabel.DefaultSolver(P, q, A, b, cones, settings), False
    try:
        if hasattr(solver, "is_data_update_allowed") and not solver.is_data_update_allowed():
            raise RuntimeError("Clarabel data update is unavailable")
        solver.update(**update)
        return solver, True
    except Exception:
        return clarabel.DefaultSolver(P, q, A, b, cones, settings), False
