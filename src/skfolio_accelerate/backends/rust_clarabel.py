"""Optional Rust Clarabel execution engine."""

from __future__ import annotations

import numpy as np

from skfolio_accelerate.compile import cones_to_spec, instantiate
from skfolio_accelerate.ir import NumericInstance, ProblemTemplate, SolveResult


def rust_is_available() -> bool:
    try:
        from skfolio_accelerate._skfolio_accel import ExecutionEngine  # noqa: F401

        return True
    except Exception:
        return False


class RustClarabelEngine:
    def __init__(
        self,
        template: ProblemTemplate,
        *,
        n_jobs: int = 1,
        solver_threads: int = 1,
    ) -> None:
        from skfolio_accelerate._skfolio_accel import ExecutionEngine

        self.template = template
        dummy = instantiate(template)
        cone_kinds, cone_dims = zip(*cones_to_spec(template.cones), strict=True)
        self._engine = ExecutionEngine(
            n_vars=int(dummy.P_shape[0]),
            n_cons=int(dummy.A_shape[0]),
            p_indptr=np.asarray(dummy.P_indptr, dtype=np.int64),
            p_indices=np.asarray(dummy.P_indices, dtype=np.int64),
            a_indptr=np.asarray(dummy.A_indptr, dtype=np.int64),
            a_indices=np.asarray(dummy.A_indices, dtype=np.int64),
            cone_kinds=list(cone_kinds),
            cone_dims=np.asarray(cone_dims, dtype=np.int64),
            weight_start=int(template.weight_slice.start or 0),
            weight_len=int(
                (template.weight_slice.stop or 0) - (template.weight_slice.start or 0)
            ),
            n_jobs=int(n_jobs),
            solver_threads=int(solver_threads),
        )

    def solve(self, instance: NumericInstance) -> SolveResult:
        xs, statuses, objs, iters, times = self._engine.solve_many(
            np.asarray(instance.P_data, dtype=np.float64)[None, :],
            np.asarray(instance.q, dtype=np.float64)[None, :],
            np.asarray(instance.A_data, dtype=np.float64)[None, :],
            np.asarray(instance.b, dtype=np.float64)[None, :],
        )
        x = np.asarray(xs[0], dtype=float)
        return SolveResult(
            status=str(statuses[0]),
            weights=x[self.template.weight_slice].copy(),
            x=x,
            objective=float(objs[0]),
            iterations=int(iters[0]),
            solve_time=float(times[0]),
        )

    def solve_many(self, instances: list[NumericInstance]) -> list[SolveResult]:
        if not instances:
            return []
        p = np.vstack([inst.P_data for inst in instances])
        q = np.vstack([inst.q for inst in instances])
        a = np.vstack([inst.A_data for inst in instances])
        b = np.vstack([inst.b for inst in instances])
        xs, statuses, objs, iters, times = self._engine.solve_many(p, q, a, b)
        out: list[SolveResult] = []
        for i in range(len(instances)):
            x = np.asarray(xs[i], dtype=float)
            out.append(
                SolveResult(
                    status=str(statuses[i]),
                    weights=x[self.template.weight_slice].copy(),
                    x=x,
                    objective=float(objs[i]),
                    iterations=int(iters[i]),
                    solve_time=float(times[i]),
                )
            )
        return out
