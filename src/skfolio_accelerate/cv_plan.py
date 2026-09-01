"""Compile a splitter once so solvers never re-iterate ``split``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from skfolio.model_selection import BaseCombinatorialCV, MultipleRandomizedCV
from sklearn.model_selection import check_cv

PlanKind = Literal["kfold", "walk_forward", "cpcv", "mrc"]


@dataclass(frozen=True, slots=True)
class FoldSpec:
    fold_id: int
    train_idx: NDArray[np.intp]
    test_idx: NDArray[np.intp]
    test_segments: tuple[NDArray[np.intp], ...] = ()
    path_ids: tuple[int, ...] = ()
    asset_idx: NDArray[np.intp] | None = None
    train_block_ids: tuple[int, ...] = ()
    train_excluded_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

    @property
    def path_id(self) -> int:
        return int(self.path_ids[0]) if self.path_ids else 0


@dataclass(frozen=True, slots=True)
class CVPlan:
    splitter_name: str
    folds: tuple[FoldSpec, ...]
    n_paths: int = 1
    combinatorial: bool = False
    multi_path: bool = False
    kind: PlanKind = "kfold"
    fold_blocks: tuple[NDArray[np.intp], ...] | None = None

    @property
    def n_splits(self) -> int:
        return len(self.folds)

    def path_batches(self) -> tuple[tuple[FoldSpec, ...], ...]:
        if self.kind != "mrc":
            return (self.folds,)
        buckets: dict[int, list[FoldSpec]] = {}
        for fold in self.folds:
            buckets.setdefault(fold.path_id, []).append(fold)
        return tuple(tuple(buckets[key]) for key in sorted(buckets))


def cpcv_fold_blocks(n_samples: int, n_folds: int) -> list[NDArray[np.intp]]:
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
    folds = []
    for fold_id, split in enumerate(splitter.split(X, y)):
        train, test = split[:2]
        test_idx = np.asarray(test, dtype=np.intp)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=test_idx,
                test_segments=(test_idx,),
                path_ids=(0,),
            )
        )
    name = type(splitter).__name__
    return CVPlan(
        splitter_name=name,
        folds=tuple(folds),
        kind="walk_forward" if name == "WalkForward" else "kfold",
    )


def _compile_cpcv(splitter, X, y=None) -> CVPlan:
    path_ids = np.asarray(splitter.get_path_ids())
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    n_samples = int(np.asarray(X).shape[0])
    blocks = tuple(cpcv_fold_blocks(n_samples, int(splitter.n_folds)))
    fold_of = np.empty(n_samples, dtype=np.intp)
    for block_id, rows in enumerate(blocks):
        fold_of[rows] = block_id
    folds = []
    for fold_id, (train, test_segments) in enumerate(splitter.split(X, y)):
        segments = tuple(np.asarray(seg, dtype=np.intp) for seg in test_segments)
        concatenated = (
            np.concatenate(segments) if segments else np.array([], dtype=np.intp)
        )
        train_idx = np.asarray(train, dtype=np.intp)
        train_blocks = tuple(int(v) for v in np.unique(fold_of[train_idx]))
        excluded = np.setdiff1d(
            np.concatenate([blocks[i] for i in train_blocks]),
            train_idx,
            assume_unique=True,
        ).astype(np.intp, copy=False)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=concatenated,
                test_segments=segments,
                path_ids=tuple(int(v) for v in path_ids[fold_id]),
                train_block_ids=train_blocks,
                train_excluded_idx=excluded,
            )
        )
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=tuple(folds),
        n_paths=n_paths,
        combinatorial=True,
        multi_path=True,
        kind="cpcv",
        fold_blocks=blocks,
    )


def _compile_mrc(splitter, X, y=None) -> CVPlan:
    splits = list(splitter.split(X, y))
    path_ids = np.asarray(splitter.get_path_ids())
    folds = []
    for fold_id, ((train, test, assets), path_id) in enumerate(
        zip(splits, path_ids, strict=True)
    ):
        test_idx = np.asarray(test, dtype=np.intp)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=test_idx,
                test_segments=(test_idx,),
                path_ids=(int(path_id),),
                asset_idx=np.asarray(assets, dtype=np.intp),
            )
        )
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=tuple(folds),
        n_paths=int(path_ids.max()) + 1 if path_ids.size else 1,
        multi_path=True,
        kind="mrc",
    )
