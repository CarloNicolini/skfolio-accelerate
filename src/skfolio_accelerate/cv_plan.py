"""Compile sklearn / skfolio splitters into a compact CV plan."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import check_cv

from skfolio.model_selection import BaseCombinatorialCV, MultipleRandomizedCV

from skfolio_accelerate.ir import CVPlan, FoldSpec


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
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=np.asarray(test, dtype=np.intp),
                test_segments=[np.asarray(test, dtype=np.intp)],
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
    purged = int(getattr(splitter, "purged_size", 0) or 0)
    embargo = int(getattr(splitter, "embargo_size", 0) or 0)
    folds: list[FoldSpec] = []
    for fold_id, (train, test_segments) in enumerate(splitter.split(X, y)):
        segments = [np.asarray(seg, dtype=np.intp) for seg in test_segments]
        concatenated = (
            np.concatenate(segments) if segments else np.array([], dtype=np.intp)
        )
        train_idx = np.asarray(train, dtype=np.intp)
        if purged or embargo:
            train_blocks: tuple[int, ...] = ()
        else:
            train_blocks = tuple(
                int(v) for v in sorted(set(fold_of[train_idx].tolist()))
            )
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=concatenated,
                test_segments=segments,
                path_ids=[int(v) for v in path_ids[fold_id]],
                train_block_ids=train_blocks,
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
    folds: list[FoldSpec] = []
    for fold_id, (train, test, assets) in enumerate(splitter.split(X, y)):
        test_idx = np.asarray(test, dtype=np.intp)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_idx=np.asarray(train, dtype=np.intp),
                test_idx=test_idx,
                test_segments=[test_idx],
                path_ids=[int(splitter.get_path_ids()[fold_id])],
                asset_idx=np.asarray(assets, dtype=np.intp),
            )
        )
    path_ids = np.asarray(splitter.get_path_ids())
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=folds,
        n_paths=n_paths,
        combinatorial=False,
        multi_path=True,
        kind="mrc",
    )


def _contiguous_slice(idx: np.ndarray) -> slice | None:
    if idx.ndim != 1 or idx.size == 0:
        return None
    start = int(idx[0])
    stop = int(idx[-1]) + 1
    if stop - start != idx.size or start < 0:
        return None
    if idx.size == 1 or (int(idx[1]) == start + 1 and int(idx[-1]) == stop - 1):
        if idx.size > 2 and int(idx[idx.size // 2]) != start + idx.size // 2:
            return None
        return slice(start, stop)
    return None


def slice_rows(X: Any, idx: np.ndarray):
    sl = _contiguous_slice(np.asarray(idx))
    if sl is not None:
        if hasattr(X, "iloc"):
            return X.iloc[sl]
        return np.asarray(X)[sl]
    if hasattr(X, "iloc"):
        return X.iloc[np.asarray(idx)]
    return np.asarray(X)[np.asarray(idx)]


def slice_panel(X: Any, rows: np.ndarray, cols: np.ndarray | None = None):
    """Slice observations and optional asset columns, preserving DataFrame metadata."""
    sub = slice_rows(X, rows)
    if cols is None:
        return sub
    col_idx = np.asarray(cols)
    if hasattr(sub, "iloc"):
        return sub.iloc[:, col_idx]
    return np.asarray(sub)[:, col_idx]
