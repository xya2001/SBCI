"""Helpers for the ico4 computational grid and its condensed storage."""

from __future__ import annotations

import numpy as np

from . import spec


def condensed_size(n: int) -> int:
    """Number of strict-upper-triangular entries of an ``n x n`` matrix.

    Examples
    --------
    >>> condensed_size(5)
    10
    """
    return n * (n - 1) // 2


def n_from_condensed(size: int) -> int:
    """Recover the matrix order ``n`` from a condensed vector length.

    Examples
    --------
    >>> n_from_condensed(10)
    5
    """
    n = int(round((1 + np.sqrt(1 + 8 * size)) / 2))
    if condensed_size(n) != size:
        raise ValueError(f"{size} is not a valid strict-upper-triangular length")
    return n


def to_dense(condensed: np.ndarray, n: int | None = None) -> np.ndarray:
    """Expand a condensed vector to a symmetric matrix with a zero diagonal.

    Examples
    --------
    >>> to_dense(np.array([1.0, 2.0, 3.0]))
    array([[0., 1., 2.],
           [1., 0., 3.],
           [2., 3., 0.]])
    """
    condensed = np.asarray(condensed)
    if n is None:
        n = n_from_condensed(condensed.size)
    out = np.zeros((n, n), dtype=condensed.dtype)
    rows, cols = np.triu_indices(n, k=1)
    # Fill both triangles rather than adding the transpose, which would
    # allocate a second n x n array -- 105 MB on the ico4 grid.
    out[rows, cols] = condensed
    out[cols, rows] = condensed
    return out


def to_condensed(dense: np.ndarray) -> np.ndarray:
    """Take the strict upper triangle of a square matrix as a flat vector.

    Examples
    --------
    >>> to_condensed(np.array([[0., 1.], [1., 0.]]))
    array([1.])
    """
    dense = np.asarray(dense)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {dense.shape}")
    return dense[np.triu_indices(dense.shape[0], k=1)]


def hemisphere_labels(n_vertices: int = spec.N_VERTICES) -> np.ndarray:
    """Per-vertex hemisphere code, ``0`` for left and ``1`` for right.

    Left hemisphere occupies the first half of the grid, per ``spec.GRID``.
    """
    half = n_vertices // 2
    labels = np.ones(n_vertices, dtype=np.int8)
    labels[:half] = 0
    return labels
