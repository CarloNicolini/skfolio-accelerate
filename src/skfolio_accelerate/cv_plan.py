"""Compile sklearn / skfolio splitters into a compact CV plan."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import check_cv

from skfolio.model_selection import BaseCombinatorialCV

from skfolio_accelerate.ir import CVPlan, FoldSpec


def compile_cv_plan(cv, X, y=None) -> CVPlan:
    if isinstance(cv, BaseCombinatorialCV):
        splitter = cv
        combinatorial = True
    else:
        splitter = check_cv(cv)
        combinatorial = isinstance(splitter, BaseCombinatorialCV)

    folds: list[FoldSpec] = []
    if combinatorial:
        path_ids = np.asarray(splitter.get_path_ids())
        n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
        for fold_id, (train, test_segments) in enumerate(splitter.split(X, y)):
            segments = [np.asarray(seg, dtype=np.intp) for seg in test_segments]
            concatenated = (
                np.concatenate(segments) if segments else np.array([], dtype=np.intp)
            )
            folds.append(
                FoldSpec(
                    fold_id=fold_id,
                    train_idx=np.asarray(train, dtype=np.intp),
                    test_idx=concatenated,
                    test_segments=segments,
                    path_ids=[int(v) for v in path_ids[fold_id]],
                )
            )
        return CVPlan(
            splitter_name=type(splitter).__name__,
            folds=folds,
            n_paths=n_paths,
            combinatorial=True,
        )

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
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=folds,
        n_paths=1,
        combinatorial=False,
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
