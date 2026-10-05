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
    """One-hot area weights per region, as a sparse ``(n_vertices, K)`` matrix, and the ids.

    Column ``k`` holds each member vertex's area weight for region ``ids[k]``; a
    vertex belongs to one region at most, and label 0 (no region) to none.
    """
    area = np.asarray(area, dtype=np.float64)
    labels = np.asarray(atlas.labels)
    if labels.shape != area.shape:
        raise ValueError(f"atlas has {labels.size} vertices but area weights have {area.size}")

    from scipy import sparse

    ids = atlas.region_ids
    # One-hot columns, built in sparse form directly: the dense ``(n, K)`` array
    # is 41 MB for a thousand-region atlas and a hundred times slower to fill.
    # Labels run 1..K with no gaps (:attr:`Atlas.region_ids`), so label ``r``
    # is column ``r - 1``; label 0 is no region and gets no column.
    rows = np.flatnonzero(labels > 0)
    weights = sparse.csr_matrix(
        (area[rows], (rows, labels[rows] - 1)), shape=(labels.size, ids.size)
    )
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
        mass by the total weight of the vertex pairs it ran over -- the product
        of the two regions' areas, less the self-pairs within a region -- giving
        a density comparable across regions of different size. ``"mean"``
        reproduces ``parcellate_sc.m`` off the diagonal.
    fisher_z
        Aggregate through ``arctanh`` and map back with ``tanh``. Required for
        FC, where averaging correlations directly is biased. Only meaningful
        with ``how="mean"``: the ``tanh`` of a *sum* of z-values saturates.

    Notes
    -----
    The region diagonal is within-region connectivity, defined under
    ``"mean"`` as the area-weighted mean over the region's *distinct* vertex
    pairs. The vertex self-diagonal of ``dense`` is zero
    (:data:`sbci.spec.DIAGONAL_INCLUDED`), so the self-pairs' weight
    ``sum_i a_i^2`` is left out of the within-region denominators too:
    dividing by the full ``A_k^2`` would put a region whose pairs all carry
    ``r`` at ``r (1 - sum_i a_i^2 / A_k^2)``, about 4% low for a
    Schaefer200-sized region on ico4. A region with a single vertex (or none
    at this resolution) has no distinct pair, and its diagonal entry is
    ``NaN``. The legacy MATLAB routine loops only over ``i < j`` and leaves
    the region diagonal at zero -- see ``PORTING.md`` item 3.
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
    # The weight matrix is one-hot: the sparse product is O(n^2) where a dense
    # one is O(n^2 K), which for a 1000-region atlas is the difference between
    # tens of milliseconds and a second.
    mass = np.asarray((weights.T @ dense) @ weights)

    if how == "mean":
        totals = np.asarray(weights.sum(axis=0)).ravel()
        denominator = np.outer(totals, totals)
        # Within a region the mass runs over distinct vertex pairs only, so the
        # self-pairs come off the diagonal denominators. A region of one vertex
        # has no pair, and no within-region mean: NaN rather than zero.
        self_pairs = np.asarray(weights.multiply(weights).sum(axis=0)).ravel()
        diagonal = np.diag_indices_from(denominator)
        denominator[diagonal] -= self_pairs
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denominator > 0, mass / denominator, 0.0)
        out[diagonal] = np.where(denominator[diagonal] > 0, out[diagonal], np.nan)
    else:
        out = mass

    if fisher_z:
        out = np.tanh(out)
    return out
