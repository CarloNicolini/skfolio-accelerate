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
        self._P = None
        self._q = None
        self._A = None
        self._b = None
        self._A_src = None
        self._q_src = None
        self._b_src = None

    def _load(self, instance: NumericInstance):
        if self._P is None:
            P, q, A, b = instance_to_scipy(self.template, instance)
            self._P = P.copy()
            self._A = A.copy()
            self._q = np.array(q, dtype=np.float64, copy=True)
            self._b = np.array(b, dtype=np.float64, copy=True)
            self._A_src = instance.A_data
            self._q_src = instance.q
            self._b_src = instance.b
            return self._P, self._q, self._A, self._b, True, True
        self._P.data[:] = instance.P_data
        q_changed = instance.q is not self._q_src
        a_changed = instance.A_data is not self._A_src
        b_changed = instance.b is not self._b_src
        if q_changed:
            np.copyto(self._q, instance.q)
            self._q_src = instance.q
        if a_changed:
            self._A.data[:] = instance.A_data
            self._A_src = instance.A_data
        if b_changed:
            np.copyto(self._b, instance.b)
            self._b_src = instance.b
        return self._P, self._q, self._A, self._b, a_changed or b_changed, q_changed

    def _build_solver(self, instance: NumericInstance) -> clarabel.DefaultSolver:
        P, q, A, b, _a_changed, _q_changed = self._load(instance)
        return clarabel.DefaultSolver(P, q, A, b, self.template.cones, self.settings)

    def solve(self, instance: NumericInstance) -> SolveResult:
        if self.solver is None:
            self.solver = self._build_solver(instance)
        else:
            P, q, A, b, a_changed, q_changed = self._load(instance)
            try:
                allowed = True
                if hasattr(self.solver, "is_data_update_allowed"):
                    allowed = bool(self.solver.is_data_update_allowed())
                if not allowed:
                    self.solver = self._build_solver(instance)
                elif a_changed:
                    self.solver.update(P=P, q=q, A=A, b=b)
                elif q_changed:
                    self.solver.update(P=P, q=q)
                else:
                    self.solver.update(P=P)
            except Exception:
                self.solver = self._build_solver(instance)
        solution = self.solver.solve()
        sl = self.template.weight_slice
        weights = np.array(solution.x[sl], dtype=np.float64, copy=True)
        return SolveResult(
            status=str(solution.status),
            weights=weights,
            objective=float(getattr(solution, "obj_val", np.nan)),
            iterations=int(getattr(solution, "iterations", 0)),
            solve_time=float(getattr(solution, "solve_time", 0.0)),
        )
