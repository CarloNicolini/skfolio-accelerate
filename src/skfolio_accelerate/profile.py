"""Lightweight timing helpers and the public acceleration report."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from skfolio_accelerate.ir import AccelerationReport


class Timer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (perf_counter() - start)

    def report(self) -> str:
        total = sum(self.times.values()) or 1.0
        lines = ["Profiler"]
        for name, value in sorted(self.times.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:24s} {value:8.4f}s  ({100.0 * value / total:5.1f}%)")
        return "\n".join(lines)


def acceleration_report(
    *,
    backend: str,
    n_folds: int,
    n_params: int,
    n_solves: int,
    n_updates: int,
    n_prior_fits: int,
    compile_s: float,
    instantiate_s: float,
    solve_s: float,
    eval_s: float,
    wall_s: float,
    fallback_reason: str | None,
    n_templates: int | None = None,
    dpp: str = "compatible",
) -> AccelerationReport:
    del n_folds
    return AccelerationReport(
        backend=backend,
        dpp="n/a" if backend == "sklearn" else dpp,
        n_templates=int(n_templates if n_templates is not None else n_solves),
        n_evaluations=int(n_updates if n_updates else n_params),
        n_prior_fits=int(n_prior_fits),
        n_native_solves=int(n_updates),
        n_updates=int(n_updates),
        fallback_reason=fallback_reason,
        compile_s=float(compile_s),
        instantiate_s=float(instantiate_s),
        solve_s=float(solve_s),
        eval_s=float(eval_s),
        wall_s=float(wall_s),
    )
