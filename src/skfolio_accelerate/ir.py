"""Intermediate representation for compiled search plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ParameterClass(str, Enum):
    STRUCTURAL = "structural"
    NUMERICAL = "numerical"
    DATA = "data"
    NON_EXECUTABLE = "non_executable"


@dataclass
class FoldSpec:
    """One train/test split, with optional combinatorial test segments."""

    fold_id: int
    train_idx: NDArray[np.intp]
    test_idx: NDArray[np.intp]
    test_segments: list[NDArray[np.intp]] = field(default_factory=list)
    path_ids: list[int] = field(default_factory=list)
    asset_idx: NDArray[np.intp] | None = None
    train_block_ids: tuple[int, ...] = ()

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def path_id(self) -> int:
        return int(self.path_ids[0]) if self.path_ids else 0


@dataclass
class CVPlan:
    splitter_name: str
    folds: list[FoldSpec]
    n_paths: int = 1
    combinatorial: bool = False
    multi_path: bool = False
    kind: str = "kfold"

    @property
    def n_splits(self) -> int:
        return len(self.folds)


@dataclass
class ProblemTemplate:
    """Frozen Clarabel topology plus CVXPY parameter handles."""

    structure_key: str
    problem: Any
    param_prob: Any
    cones: list[Any]
    n_vars: int
    n_cons: int
    weight_slice: slice
    parameters: dict[str, Any]
    risk_measure: str
    n_observations: int | None
    n_assets: int
    scale_objective: float
    scale_constraints: float
    P_shape: tuple[int, int] = (0, 0)
    A_shape: tuple[int, int] = (0, 0)
    P_indices: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    P_indptr: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    A_indices: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    A_indptr: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    instantiator: Any = field(default=None, repr=False, compare=False)


@dataclass
class NumericInstance:
    """Numeric CSC values only; index arrays live on the template."""

    P_data: NDArray[np.float64]
    q: NDArray[np.float64]
    A_data: NDArray[np.float64]
    b: NDArray[np.float64]


@dataclass
class SolveResult:
    status: str
    weights: NDArray[np.float64]
    x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    objective: float = float("nan")
    iterations: int = 0
    solve_time: float = 0.0


@dataclass
class Evaluation:
    template_id: int
    fold_id: int
    param_id: int
    params: dict[str, Any]
    weights: NDArray[np.float64] | None = None
    score: float = float("nan")
    n_test: int = 0
    path_ids: list[int] = field(default_factory=list)
    status: str = ""


@dataclass
class SearchPlan:
    templates: list[ProblemTemplate] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    cv_plan: CVPlan | None = None
    param_grid_list: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "python"
    native_scoring: bool = True
    fallback_reason: str | None = None
    estimator_name: str = ""
    classification: dict[str, ParameterClass] | None = None
    scoring: Any = None
    n_jobs: int = 1


@dataclass
class AccelerationReport:
    backend: str
    dpp: str = "compatible"
    n_templates: int = 0
    n_evaluations: int = 0
    n_prior_fits: int = 0
    n_prior_updates: int = 0
    n_native_solves: int = 0
    n_updates: int = 0
    n_warm_starts: int = 0
    fallback_reason: str | None = None
    compile_s: float = 0.0
    instantiate_s: float = 0.0
    moments_s: float = 0.0
    solve_s: float = 0.0
    eval_s: float = 0.0
    wall_s: float = 0.0
    baseline_s: float = 0.0
    speedup: float = float("nan")

    def __str__(self) -> str:
        lines = [
            f"Backend: {self.backend}",
            f"DPP: {self.dpp}",
            f"Evaluations: {self.n_evaluations}",
            f"Moment fits: {self.n_prior_fits}",
            f"Moment updates: {self.n_prior_updates}",
            f"Native solves: {self.n_native_solves}",
            f"Warm starts: {self.n_warm_starts}",
            f"moments {self.moments_s:.4f}s  solve {self.solve_s:.4f}s  "
            f"eval {self.eval_s:.4f}s  wall {self.wall_s:.4f}s",
            f"Fallback: {self.fallback_reason or 'none'}",
        ]
        if self.baseline_s > 0:
            lines.append(
                f"Baseline {self.baseline_s:.4f}s  speedup {self.speedup:.2f}×"
            )
        return "\n".join(lines)
