"""Python Clarabel backend: one DefaultSolver per template, update in a loop."""

from __future__ import annotations

import clarabel
import numpy as np

from skfolio_accelerate.compile import instance_to_scipy
from skfolio_accelerate.ir import NumericInstance, ProblemTemplate, SolveResult


def clarabel_settings(
    *, solver_threads: int = 1, verbose: bool = False
) -> clarabel.DefaultSettings:
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    if hasattr(settings, "presolve_enable"):
        settings.presolve_enable = False
    if hasattr(settings, "chordal_decomposition_enable"):
        settings.chordal_decomposition_enable = False
    if hasattr(settings, "input_sparse_dropzeros"):
        settings.input_sparse_dropzeros = False
    if hasattr(settings, "max_threads"):
        settings.max_threads = int(solver_threads)
    elif hasattr(settings, "max_threads"):
        settings.max_threads = int(solver_threads)
    if hasattr(settings, "tol_gap_abs"):
        settings.tol_gap_abs = 1e-9
    if hasattr(settings, "tol_gap_rel"):
        settings.tol_gap_rel = 1e-9
    return settings


class PythonClarabelEngine:
    def __init__(
        self,
        template: ProblemTemplate,
        *,
        solver_threads: int = 1,
        verbose: bool = False,
    ) -> None:
        self.template = template
        self.settings = clarabel_settings(
            solver_threads=solver_threads, verbose=verbose
        )
        self.solver: clarabel.DefaultSolver | None = None

    def _build_solver(self, instance: NumericInstance) -> clarabel.DefaultSolver:
        P, q, A, b = instance_to_scipy(instance)
        return clarabel.DefaultSolver(P, q, A, b, self.template.cones, self.settings)

    def solve(self, instance: NumericInstance) -> SolveResult:
        if self.solver is None:
            self.solver = self._build_solver(instance)
        else:
            P, q, A, b = instance_to_scipy(instance)
            try:
                allowed = True
                if hasattr(self.solver, "is_data_update_allowed"):
                    allowed = bool(self.solver.is_data_update_allowed())
                elif hasattr(self.solver, "is_data_update_allowed"):
                    allowed = bool(self.solver.is_data_update_allowed())
                if not allowed:
                    self.solver = self._build_solver(instance)
                else:
                    self.solver.update(P=P, q=q, A=A, b=b)
            except Exception:
                self.solver = self._build_solver(instance)
        solution = self.solver.solve()
        x = np.asarray(solution.x, dtype=float)
        weights = x[self.template.weight_slice].copy()
        return SolveResult(
            status=str(solution.status),
            weights=weights,
            x=x,
            objective=float(getattr(solution, "obj_val", np.nan)),
            iterations=int(getattr(solution, "iterations", 0)),
            solve_time=float(getattr(solution, "solve_time", 0.0)),
        )
