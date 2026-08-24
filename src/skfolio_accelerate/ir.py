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

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)


@dataclass
class CVPlan:
    splitter_name: str
    folds: list[FoldSpec]
    n_paths: int = 1
    combinatorial: bool = False

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
    n_native_solves: int = 0
    n_updates: int = 0
    fallback_reason: str | None = None
    compile_s: float = 0.0
    instantiate_s: float = 0.0
    solve_s: float = 0.0
    eval_s: float = 0.0
    wall_s: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"Backend: {self.backend} / Clarabel",
            f"DPP: {self.dpp}",
            f"Templates: {self.n_templates}",
            f"Evaluations: {self.n_evaluations}",
            f"Moment fits: {self.n_prior_fits}",
            f"Native solves: {self.n_native_solves}",
            f"Fallback: {self.fallback_reason or 'none'}",
        ]
        return "\n".join(lines)
