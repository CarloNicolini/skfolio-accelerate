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


def slice_rows(X: Any, idx: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[np.asarray(idx)]
    return np.asarray(X)[np.asarray(idx)]
