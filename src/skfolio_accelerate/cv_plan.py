"""Compile sklearn / skfolio splitters into a compact CV plan."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from skfolio.model_selection import BaseCombinatorialCV, MultipleRandomizedCV
from sklearn.model_selection import check_cv


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
    train_excluded_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

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


def cpcv_fold_blocks(n_samples: int, n_folds: int) -> list[np.ndarray]:
    """Observation indices belonging to each CPCV fold (same rule as skfolio)."""
    fold_index_num = np.arange(n_samples) // (n_samples // n_folds)
    fold_index_num[fold_index_num == n_folds] = n_folds - 1
    return [
        np.flatnonzero(fold_index_num == fold_id).astype(np.intp)
        for fold_id in range(n_folds)
    ]


def compile_cv_plan(cv, X, y=None) -> CVPlan:
    if isinstance(cv, MultipleRandomizedCV):
        return _compile_mrc(cv, X, y)
    if isinstance(cv, BaseCombinatorialCV):
        return _compile_cpcv(cv, X, y)

    splitter = check_cv(cv)
    if isinstance(splitter, BaseCombinatorialCV):
        return _compile_cpcv(splitter, X, y)
    if isinstance(splitter, MultipleRandomizedCV):
        return _compile_mrc(splitter, X, y)

    folds: list[FoldSpec] = []
    for fold_id, split in enumerate(splitter.split(X, y)):
        train, test = split[:2]
        test_idx = np.asarray(test, dtype=np.intp)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=test_idx,
                test_segments=[test_idx],
                path_ids=[0],
            )
        )
    name = type(splitter).__name__
    kind = "walk_forward" if name == "WalkForward" else "kfold"
    return CVPlan(
        splitter_name=name,
        folds=folds,
        n_paths=1,
        combinatorial=False,
        multi_path=False,
        kind=kind,
    )


def _compile_cpcv(splitter, X, y=None) -> CVPlan:
    path_ids = np.asarray(splitter.get_path_ids())
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    n_samples = int(np.asarray(X).shape[0])
    blocks = cpcv_fold_blocks(n_samples, int(splitter.n_folds))
    fold_of = np.empty(n_samples, dtype=np.intp)
    for block_id, rows in enumerate(blocks):
        fold_of[rows] = block_id
    folds: list[FoldSpec] = []
    for fold_id, (train, test_segments) in enumerate(splitter.split(X, y)):
        segments = [np.asarray(seg, dtype=np.intp) for seg in test_segments]
        concatenated = (
            np.concatenate(segments) if segments else np.array([], dtype=np.intp)
        )
        train_idx = np.asarray(train, dtype=np.intp)
        train_blocks = tuple(int(v) for v in np.unique(fold_of[train_idx]))
        full_train_rows = np.concatenate([blocks[i] for i in train_blocks])
        excluded = np.setdiff1d(
            full_train_rows,
            train_idx,
            assume_unique=True,
        ).astype(np.intp, copy=False)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=concatenated,
                test_segments=segments,
                path_ids=[int(v) for v in path_ids[fold_id]],
                train_block_ids=train_blocks,
                train_excluded_idx=excluded,
            )
        )
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=folds,
        n_paths=n_paths,
        combinatorial=True,
        multi_path=True,
        kind="cpcv",
    )


def _compile_mrc(splitter, X, y=None) -> CVPlan:
    splits = list(splitter.split(X, y))
    path_ids = np.asarray(splitter.get_path_ids())
    folds: list[FoldSpec] = []
    for fold_id, ((train, test, assets), path_id) in enumerate(
        zip(splits, path_ids, strict=True)
    ):
        test_idx = np.asarray(test, dtype=np.intp)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=test_idx,
                test_segments=[test_idx],
                path_ids=[int(path_id)],
                asset_idx=np.asarray(assets, dtype=np.intp),
            )
        )
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=folds,
        n_paths=n_paths,
        combinatorial=False,
        multi_path=True,
        kind="mrc",
    )
