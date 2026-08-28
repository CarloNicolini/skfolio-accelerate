"""Single explicit configuration for the canonical cross_val_predict benchmark.

Every numeric default, splitter, solver, timeout, and estimator-selection flag
lives in :data:`CONFIG`. CLI flags override these values; they are not scattered
through the runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Explicit configuration (override via CLI; do not edit call sites instead)
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    # Synthetic factor-model panel (see datasets.make_synthetic).
    "synthetic_n_observations": 504,
    "synthetic_n_assets": 12,
    "synthetic_n_factors": 8,
    "synthetic_seed": 42,
    # WalkForward / MRC window sizes. Sequential native benchmark uses
    # train=252, test=21 on the 20-year panel; defaults here are the same
    # ratio on a shorter panel so a full estimator sweep stays tractable.
    "cv_kind": "walk-forward",
    "train_size": 126,
    "test_size": 21,
    "mrc_n_subsamples": 3,
    "mrc_asset_subset_size": 8,
    "mrc_window_size": 252,
    "mrc_random_state": 43,
    "cpcv_n_folds": 4,
    "cpcv_n_test_folds": 2,
    "cpcv_purged_size": 1,
    "cpcv_embargo_size": 1,
    # Timing protocol.
    "repetitions": 3,
    "warmups": 1,
    "workers": 1,
    "thread_limit": 1,
    "n_jobs": 1,
    # Native MeanRisk default solver. Compact engines (OSQP / HiGHS / Clarabel)
    # are selected by backend="auto" and must not receive custom solver_params
    # or they become compact-ineligible (see predict._UNSUPPORTED_IF_SET).
    "solver": "CLARABEL",
    "solver_params": None,
    "l2_coef": 1e-5,
    # Per-call wall-clock timeout in seconds; None disables.
    "timeout_s": None,
    # Estimator universe (see estimators.mean_risk_specs).
    "include_gini": False,
    "include_annualized": False,
    "include_extras": True,
    "include_lp_l2_zero": True,
    "random_seed": 42,
    "sp500_tail_observations": None,
}

# 20-year sequential-benchmark panel (benchmarks/benchmark_sequential_mean_risk.py).
FULL_PRESET: dict[str, Any] = {
    "synthetic_n_observations": 20 * 252,
    "synthetic_n_assets": 20,
    "train_size": 252,
    "test_size": 21,
    "mrc_n_subsamples": 20,
    "mrc_asset_subset_size": 12,
    "mrc_window_size": 756,
    "sp500_tail_observations": None,
}

# Smoke sizes matching --quick in the existing sequential / LP scripts.
QUICK_PRESET: dict[str, Any] = {
    "synthetic_n_observations": 120,
    "synthetic_n_assets": 6,
    "train_size": 40,
    "test_size": 20,
    "mrc_n_subsamples": 3,
    "mrc_asset_subset_size": 4,
    "mrc_window_size": 100,
    "repetitions": 1,
    "warmups": 1,
    "sp500_tail_observations": 252,
}

DATASETS = ("synthetic", "sp500")
METHODS = ("native", "accelerated")
CV_KINDS = ("walk-forward", "multiple-randomized", "purged-cpcv")

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated snapshot of :data:`CONFIG` after CLI overrides."""

    synthetic_n_observations: int
    synthetic_n_assets: int
    synthetic_n_factors: int
    synthetic_seed: int
    cv_kind: str
    train_size: int
    test_size: int
    mrc_n_subsamples: int
    mrc_asset_subset_size: int
    mrc_window_size: int
    mrc_random_state: int
    cpcv_n_folds: int
    cpcv_n_test_folds: int
    cpcv_purged_size: int
    cpcv_embargo_size: int
    repetitions: int
    warmups: int
    workers: int
    thread_limit: int
    n_jobs: int
    solver: str
    solver_params: dict[str, Any] | None
    l2_coef: float
    timeout_s: float | None
    include_gini: bool
    include_annualized: bool
    include_extras: bool
    include_lp_l2_zero: bool
    random_seed: int
    sp500_tail_observations: int | None
    datasets: tuple[str, ...] = field(default_factory=lambda: DATASETS)
    methods: tuple[str, ...] = field(default_factory=lambda: METHODS)
    cv_kinds: tuple[str, ...] = field(default_factory=lambda: ("walk-forward",))

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthetic_n_observations": self.synthetic_n_observations,
            "synthetic_n_assets": self.synthetic_n_assets,
            "synthetic_n_factors": self.synthetic_n_factors,
            "synthetic_seed": self.synthetic_seed,
            "cv_kind": self.cv_kind,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "mrc_n_subsamples": self.mrc_n_subsamples,
            "mrc_asset_subset_size": self.mrc_asset_subset_size,
            "mrc_window_size": self.mrc_window_size,
            "mrc_random_state": self.mrc_random_state,
            "cpcv_n_folds": self.cpcv_n_folds,
            "cpcv_n_test_folds": self.cpcv_n_test_folds,
            "cpcv_purged_size": self.cpcv_purged_size,
            "cpcv_embargo_size": self.cpcv_embargo_size,
            "repetitions": self.repetitions,
            "warmups": self.warmups,
            "workers": self.workers,
            "thread_limit": self.thread_limit,
            "n_jobs": self.n_jobs,
            "solver": self.solver,
            "solver_params": self.solver_params,
            "l2_coef": self.l2_coef,
            "timeout_s": self.timeout_s,
            "include_gini": self.include_gini,
            "include_annualized": self.include_annualized,
            "include_extras": self.include_extras,
            "include_lp_l2_zero": self.include_lp_l2_zero,
            "random_seed": self.random_seed,
            "sp500_tail_observations": self.sp500_tail_observations,
            "datasets": list(self.datasets),
            "methods": list(self.methods),
            "cv_kinds": list(self.cv_kinds),
        }


def _positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def validate_raw(raw: dict[str, Any]) -> None:
    """Raise ``ValueError`` if a configuration mapping is inconsistent."""
    _positive_int("synthetic_n_observations", int(raw["synthetic_n_observations"]))
    _positive_int("synthetic_n_assets", int(raw["synthetic_n_assets"]), minimum=2)
    _positive_int("synthetic_n_factors", int(raw["synthetic_n_factors"]))
    if int(raw["synthetic_n_assets"]) < 2:
        raise ValueError("synthetic_n_assets must be >= 2")
    _positive_int("train_size", int(raw["train_size"]))
    _positive_int("test_size", int(raw["test_size"]))
    _positive_int("repetitions", int(raw["repetitions"]))
    if int(raw["warmups"]) < 0:
        raise ValueError("warmups must be >= 0")
    _positive_int("workers", int(raw["workers"]))
    _positive_int("thread_limit", int(raw["thread_limit"]))
    n_jobs = int(raw["n_jobs"])
    if n_jobs < 1 and n_jobs != -1:
        raise ValueError("n_jobs must be >= 1 or -1")
    timeout = raw.get("timeout_s")
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("timeout_s must be None or a positive number")
    tail = raw.get("sp500_tail_observations")
    if tail is not None and int(tail) < 2:
        raise ValueError("sp500_tail_observations must be None or >= 2")
    l2 = float(raw["l2_coef"])
    if l2 < 0:
        raise ValueError("l2_coef must be >= 0")


def build_config(
    overrides: dict[str, Any] | None = None,
    *,
    datasets: tuple[str, ...] | list[str] = DATASETS,
    methods: tuple[str, ...] | list[str] = METHODS,
    cv_kinds: tuple[str, ...] | list[str] | None = None,
    preset: str | None = None,
) -> BenchmarkConfig:
    """Merge :data:`CONFIG` with a named preset and CLI overrides, then validate."""
    raw = deepcopy(CONFIG)
    if preset == "full":
        raw.update(FULL_PRESET)
    elif preset == "quick":
        raw.update(QUICK_PRESET)
    elif preset is not None:
        raise ValueError(f"unknown preset {preset!r}")
    if overrides:
        raw.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    validate_raw(raw)
    dataset_tuple = tuple(datasets)
    method_tuple = tuple(methods)
    cv_tuple = tuple(cv_kinds) if cv_kinds is not None else (raw["cv_kind"],)
    unknown_ds = [name for name in dataset_tuple if name not in DATASETS]
    if unknown_ds:
        raise ValueError(f"unknown datasets: {unknown_ds}")
    unknown_m = [name for name in method_tuple if name not in METHODS]
    if unknown_m:
        raise ValueError(f"unknown methods: {unknown_m}")
    unknown_cv = [name for name in cv_tuple if name not in CV_KINDS]
    if unknown_cv:
        raise ValueError(f"unknown cv kinds: {unknown_cv}")
    if not dataset_tuple or not method_tuple or not cv_tuple:
        raise ValueError("datasets, methods, and cv_kinds must be non-empty")
    return BenchmarkConfig(
        synthetic_n_observations=int(raw["synthetic_n_observations"]),
        synthetic_n_assets=int(raw["synthetic_n_assets"]),
        synthetic_n_factors=int(raw["synthetic_n_factors"]),
        synthetic_seed=int(raw["synthetic_seed"]),
        cv_kind=str(raw["cv_kind"]),
        train_size=int(raw["train_size"]),
        test_size=int(raw["test_size"]),
        mrc_n_subsamples=int(raw["mrc_n_subsamples"]),
        mrc_asset_subset_size=int(raw["mrc_asset_subset_size"]),
        mrc_window_size=int(raw["mrc_window_size"]),
        mrc_random_state=int(raw["mrc_random_state"]),
        cpcv_n_folds=int(raw["cpcv_n_folds"]),
        cpcv_n_test_folds=int(raw["cpcv_n_test_folds"]),
        cpcv_purged_size=int(raw["cpcv_purged_size"]),
        cpcv_embargo_size=int(raw["cpcv_embargo_size"]),
        repetitions=int(raw["repetitions"]),
        warmups=int(raw["warmups"]),
        workers=int(raw["workers"]),
        thread_limit=int(raw["thread_limit"]),
        n_jobs=int(raw["n_jobs"]),
        solver=str(raw["solver"]),
        solver_params=raw["solver_params"],
        l2_coef=float(raw["l2_coef"]),
        timeout_s=None if raw["timeout_s"] is None else float(raw["timeout_s"]),
        include_gini=bool(raw["include_gini"]),
        include_annualized=bool(raw["include_annualized"]),
        include_extras=bool(raw["include_extras"]),
        include_lp_l2_zero=bool(raw["include_lp_l2_zero"]),
        random_seed=int(raw["random_seed"]),
        sp500_tail_observations=(
            None
            if raw["sp500_tail_observations"] is None
            else int(raw["sp500_tail_observations"])
        ),
        datasets=dataset_tuple,
        methods=method_tuple,
        cv_kinds=cv_tuple,
    )
