"""Vertex-to-region aggregation.

Port of ``SBCI_Toolkit/analysis/parcellate_sc.m`` and ``parcellate_fc.m``.
The aggregation is area-weighted: a region's connectivity is the mass carried
by all vertex pairs crossing between the two regions, so that the result does
not change when the grid is refined.
"""

from __future__ import annotations

import numpy as np

from .atlas import Atlas, load_atlas

HOW = ("mass", "mean")


def region_weights(atlas: Atlas, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``n_vertices x n_regions`` area-weight matrix.

    Returns the weight matrix ``W`` with ``W[v, r] = area[v]`` when vertex ``v``
    belongs to region ``r`` and ``0`` otherwise, together with the region ids.
    """
    area = np.asarray(area, dtype=np.float64)
    labels = np.asarray(atlas.labels)
    if labels.shape != area.shape:
        raise ValueError(f"atlas has {labels.size} vertices but area weights have {area.size}")

    ids = atlas.region_ids
    weights = np.zeros((labels.size, ids.size), dtype=np.float64)
    for column, region in enumerate(ids):
        member = labels == region
        weights[member, column] = area[member]
    return weights, ids


def parcellate(
    dense: np.ndarray,
    atlas: Atlas | str,
    area: np.ndarray,
    how: str = "mass",
    fisher_z: bool = False,
) -> np.ndarray:
    """Aggregate a vertex-level connectivity matrix to region level.

    Parameters
    ----------
    dense
        Symmetric ``n_vertices x n_vertices`` connectivity with a zero diagonal.
    atlas
        Parcellation on the same grid, or the name of a bundled one.
    area
        Per-vertex area weight; zero for vertices that should not count.
    how
        ``"mass"`` sums the area-weighted connectivity crossing each region
        pair, so the total over all pairs is preserved. ``"mean"`` divides that
        mass by the product of the two regions' total areas, giving a density
        comparable across regions of different size. ``"mean"`` reproduces
        ``parcellate_sc.m``.
    fisher_z
        Aggregate through ``arctanh`` and map back with ``tanh``. Required for
        FC, where averaging correlations directly is biased. Only meaningful
        with ``how="mean"``: the ``tanh`` of a *sum* of z-values saturates.

    Notes
    -----
    The region diagonal is computed here (within-region connectivity, with the
    vertex self-diagonal excluded). The legacy MATLAB routine loops only over
    ``i < j`` and leaves the diagonal at zero -- see ``PORTING.md`` item 3;
    WP1 should confirm which behavior the released files carry.
    """
    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, got {how!r}")
    if fisher_z and how != "mean":
        raise ValueError("fisher_z aggregation is an average of correlations; use how='mean'")
    if isinstance(atlas, str):
        atlas = load_atlas(atlas)

    dense = np.asarray(dense, dtype=np.float64)
    if fisher_z:
        dense = np.arctanh(np.clip(dense, -0.999999, 0.999999))

    weights, _ = region_weights(atlas, area)
    # The weight matrix is one-hot: a sparse product is O(n^2) where the dense
    # one is O(n^2 K), which for a 1000-region atlas is the difference between
    # tens of milliseconds and a second.
    from scipy import sparse

    sparse_weights = sparse.csr_matrix(weights)
    mass = np.asarray((sparse_weights.T @ dense) @ sparse_weights)

    if how == "mean":
        totals = weights.sum(axis=0)
        denominator = np.outer(totals, totals)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denominator > 0, mass / denominator, 0.0)
    else:
        out = mass

    if fisher_z:
        out = np.tanh(out)
    return out
