"""Compile sklearn / skfolio splitters into an immutable CV execution plan.

The public flow is:

    splitter definition → :func:`compile_cv_plan` → :class:`CVPlan` → execution

``compile_cv_plan`` is the only place that iterates ``splitter.split``. Execution
paths consume the compiled folds, optional CPCV fold blocks, and MRC path batches
without calling the splitter again. Mutable splitters (for example a
``RandomState`` advanced inside ``split``) are therefore consumed exactly once
per ``cross_val_predict`` call; a deepcopy of the original splitter is kept only
for native fallback.
"""

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
    """One train/test split, with optional combinatorial test segments.

    Numpy index arrays remain mutable buffers, but the fold identity (which
    arrays and metadata belong together) cannot be reassigned after compilation.

    Attributes
    ----------
    fold_id : int
        Stable identifier used as the key in weight maps.

    train_idx : ndarray of shape (n_train,)
        Training observation indices.

    test_idx : ndarray of shape (n_test,)
        Concatenated test indices (all combinatorial segments joined).

    test_segments : tuple of ndarray
        Per-path test segments for combinatorial CV. For single-path splitters
        this is ``(test_idx,)``.

    path_ids : tuple of int
        Path identifiers aligned with ``test_segments``.

    asset_idx : ndarray of shape (n_subset,) or None
        Asset subset for MultipleRandomizedCV. ``None`` means the full universe.

    train_block_ids : tuple of int
        CPCV fold-block identifiers that cover the training window (before purge
        / embargo exclusions).

    train_excluded_idx : ndarray of shape (n_excluded,)
        Observations belonging to training blocks but excluded by purge/embargo.
    """

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
        """Primary path id (first entry of ``path_ids``, or ``0``)."""
        return int(self.path_ids[0]) if self.path_ids else 0


@dataclass(frozen=True, slots=True)
class CVPlan:
    """Compiled, reusable description of a cross-validation split sequence.

    Attributes
    ----------
    splitter_name : str
        Class name of the original splitter (for diagnostics).

    folds : tuple of FoldSpec
        All compiled train/test splits in splitter order.

    n_paths : int, default=1
        Number of test paths (``1`` for WalkForward / KFold).

    combinatorial : bool, default=False
        ``True`` for CombinatorialPurgedCV-style multi-segment tests.

    multi_path : bool, default=False
        ``True`` when results should be assembled into a Population.

    kind : {"kfold", "walk_forward", "cpcv", "mrc"}, default="kfold"
        Coarse splitter family used by execution backends.

    fold_blocks : tuple of ndarray or None, default=None
        CPCV fold-block row indices for moment reuse. ``None`` unless ``kind``
        is ``"cpcv"``.
    """

    splitter_name: str
    folds: tuple[FoldSpec, ...]
    n_paths: int = 1
    combinatorial: bool = False
    multi_path: bool = False
    kind: PlanKind = "kfold"
    fold_blocks: tuple[NDArray[np.intp], ...] | None = None

    @property
    def n_splits(self) -> int:
        """Number of compiled folds."""
        return len(self.folds)

    def path_batches(self) -> tuple[tuple[FoldSpec, ...], ...]:
        """Folds grouped by MRC path; a single batch for every other splitter.

        MRC paths do not share assets, so moment caches and (for variance) OSQP
        workspaces are built per batch. CPCV is multi-path but shares one set of
        fold-block statistics, so it stays as a single batch.

        Returns
        -------
        batches : tuple of tuple of FoldSpec
            One batch per MRC path, or a single batch containing all folds.
        """
        if self.kind != "mrc":
            return (self.folds,)
        buckets: dict[int, list[FoldSpec]] = {}
        for fold in self.folds:
            buckets.setdefault(fold.path_id, []).append(fold)
        return tuple(tuple(buckets[key]) for key in sorted(buckets))


def order_cpcv_folds_by_similarity(
    folds: tuple[FoldSpec, ...],
) -> tuple[FoldSpec, ...]:
    """Greedily order CPCV solves so adjacent training block unions overlap."""
    if len(folds) < 3:
        return folds
    block_sets = [frozenset(fold.train_block_ids) for fold in folds]
    excluded_sets = [
        frozenset(int(row) for row in fold.train_excluded_idx) for fold in folds
    ]
    ordered = [0]
    remaining = set(range(1, len(folds)))
    while remaining:
        current = ordered[-1]
        next_index = min(
            remaining,
            key=lambda candidate: (
                len(block_sets[current] ^ block_sets[candidate]),
                len(excluded_sets[current] ^ excluded_sets[candidate]),
                candidate,
            ),
        )
        ordered.append(next_index)
        remaining.remove(next_index)
    return tuple(folds[index] for index in ordered)


def cpcv_fold_blocks(n_samples: int, n_folds: int) -> list[NDArray[np.intp]]:
    """Observation indices belonging to each CPCV fold (same rule as skfolio).

    Parameters
    ----------
    n_samples : int
        Number of observations.

    n_folds : int
        Number of combinatorial folds.

    Returns
    -------
    blocks : list of ndarray of shape (n_block_i,)
        Contiguous (or nearly contiguous) index arrays, one per fold.
    """
    fold_index_num = np.arange(n_samples) // (n_samples // n_folds)
    fold_index_num[fold_index_num == n_folds] = n_folds - 1
    return [
        np.flatnonzero(fold_index_num == fold_id).astype(np.intp)
        for fold_id in range(n_folds)
    ]


def compile_cv_plan(cv, X, y=None) -> CVPlan:
    """Materialize train/test indices and CPCV metadata for one predict call.

    This is the only function that iterates ``splitter.split``. Downstream
    backends must not call the splitter again.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable
        Splitter definition. Combinatorial and MultipleRandomizedCV instances
        are detected before falling back to
        :func:`~sklearn.model_selection.check_cv`.

    X : array-like of shape (n_observations, n_assets)
        Returns matrix (used only for shape and ``split``).

    y : array-like, optional
        Target forwarded to ``split``.

    Returns
    -------
    plan : CVPlan
        Immutable plan with folds, path metadata, and optional CPCV blocks.

    Notes
    -----
    Mutable splitters that advance RNG state inside ``split`` are consumed
    exactly once. Callers that need a native fallback should deepcopy ``cv``
    *before* compiling.

    Examples
    --------
    >>> from skfolio.model_selection import WalkForward
    >>> plan = compile_cv_plan(WalkForward(train_size=100, test_size=20), X)
    ... # doctest: +SKIP
    >>> plan.n_splits  # doctest: +SKIP
    >>> plan.kind  # doctest: +SKIP
    'walk_forward'
    """
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
                test_segments=(test_idx,),
                path_ids=(0,),
            )
        )
    name = type(splitter).__name__
    kind: PlanKind = "walk_forward" if name == "WalkForward" else "kfold"
    return CVPlan(
        splitter_name=name,
        folds=tuple(folds),
        n_paths=1,
        combinatorial=False,
        multi_path=False,
        kind=kind,
    )


def _compile_cpcv(splitter, X, y=None) -> CVPlan:
    path_ids = np.asarray(splitter.get_path_ids())
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    n_samples = int(np.asarray(X).shape[0])
    blocks = tuple(cpcv_fold_blocks(n_samples, int(splitter.n_folds)))
    fold_of = np.empty(n_samples, dtype=np.intp)
    for block_id, rows in enumerate(blocks):
        fold_of[rows] = block_id
    folds: list[FoldSpec] = []
    for fold_id, (train, test_segments) in enumerate(splitter.split(X, y)):
        segments = tuple(np.asarray(seg, dtype=np.intp) for seg in test_segments)
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
                test_segments=(test_idx,),
                path_ids=(int(path_id),),
                asset_idx=np.asarray(assets, dtype=np.intp),
            )
        )
    n_paths = int(path_ids.max()) + 1 if path_ids.size else 1
    return CVPlan(
        splitter_name=type(splitter).__name__,
        folds=tuple(folds),
        n_paths=n_paths,
        combinatorial=False,
        multi_path=True,
        kind="mrc",
    )
