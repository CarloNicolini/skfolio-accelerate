"""Small array helpers shared by planning, moments, and assembly.

These exist so contiguous-window detection and dtype coercion are defined once.
They must not copy when the input is already a C-contiguous float64 array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_float_array(data) -> NDArray[np.float64]:
    """Return a C-contiguous float64 ndarray, copying only when needed."""
    arr = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def as_float_2d(X) -> NDArray[np.float64]:
    """Return returns as a contiguous float64 ndarray (usually 2-D)."""
    return as_float_array(X)


def contiguous_row_slice(rows: NDArray[np.intp]) -> slice | None:
    """Return ``slice(start, stop)`` when ``rows`` is a contiguous increasing range.

    The check is O(1): length, endpoints, and one interior sample. CV indices from
    WalkForward, TimeSeriesSplit, and CPCV fold blocks satisfy this. Fancy-indexed
    KFold / MRC rows fall through to integer indexing.
    """
    if rows.ndim != 1 or rows.size == 0:
        return None
    start = int(rows[0])
    stop = int(rows[-1]) + 1
    if start < 0 or stop - start != rows.size:
        return None
    if rows.size > 1 and int(rows[1]) != start + 1:
        return None
    if rows.size > 2 and int(rows[rows.size // 2]) != start + rows.size // 2:
        return None
    return slice(start, stop)
