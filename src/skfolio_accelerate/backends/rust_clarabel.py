"""Optional Rust Clarabel execution engine."""

from __future__ import annotations

import numpy as np

from skfolio_accelerate.compile import cones_to_spec
from skfolio_accelerate.ir import NumericInstance, ProblemTemplate, SolveResult


def rust_is_available() -> bool:
    try:
        from skfolio_accelerate._skfolio_accel import ExecutionEngine  # noqa: F401

        return True
    except Exception:
        return False


def _stack_or_share(instances: list[NumericInstance], attr: str) -> np.ndarray:
    first = np.asarray(getattr(instances[0], attr), dtype=np.float64)
    if all(getattr(inst, attr) is getattr(instances[0], attr) for inst in instances):
        return np.ascontiguousarray(first, dtype=np.float64).reshape(1, -1)
    n = len(instances)
    out = np.empty((n, first.size), dtype=np.float64)
    out[0] = first
    for i, inst in enumerate(instances[1:], start=1):
        np.copyto(out[i], np.asarray(getattr(inst, attr), dtype=np.float64))
    return out


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
        cone_kinds, cone_dims = zip(*cones_to_spec(template.cones), strict=True)
        self._engine = ExecutionEngine(
            n_vars=int(template.P_shape[0]),
            n_cons=int(template.A_shape[0]),
            p_indptr=np.asarray(template.P_indptr, dtype=np.int64),
            p_indices=np.asarray(template.P_indices, dtype=np.int64),
            a_indptr=np.asarray(template.A_indptr, dtype=np.int64),
            a_indices=np.asarray(template.A_indices, dtype=np.int64),
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
        return self.solve_many([instance])[0]

    def solve_many(self, instances: list[NumericInstance]) -> list[SolveResult]:
        if not instances:
            return []
        weights, statuses, objs, iters, times = self._engine.solve_many(
            _stack_or_share(instances, "P_data"),
            _stack_or_share(instances, "q"),
            _stack_or_share(instances, "A_data"),
            _stack_or_share(instances, "b"),
        )
        out: list[SolveResult] = []
        for i in range(len(instances)):
            out.append(
                SolveResult(
                    status=str(statuses[i]),
                    weights=np.asarray(weights[i], dtype=np.float64),
                    objective=float(objs[i]),
                    iterations=int(iters[i]),
                    solve_time=float(times[i]),
                )
            )
        return out
