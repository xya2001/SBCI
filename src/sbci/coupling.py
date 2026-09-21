"""Structure-function coupling (SFC).

Port of the three functions in ``SBCI_Toolkit/sfc/``: ``calculate_sfc_gbl.m``,
``calculate_sfc_loc.m`` and ``calculate_sfc_dct.m``.

Each gives one value per vertex (or per region, for the discrete form): how
closely that vertex's structural connectivity profile matches its functional
one. Negative FC values are kept throughout -- discarding them changes the sign
of the coupling map in association cortex.

Three details of the reference are easy to miss and are reproduced here.

*The global and local forms are not correlations.* They use the uncentred inner
product, ``dot(fc, sc) / (|fc| |sc|)`` -- a cosine similarity. The discrete form
alone uses MATLAB's ``corr2``, which centres both vectors first, making it a
Pearson correlation. The three are documented together as "cosine similarity"
but only two of them are.

*Constant profiles become NaN, not zero.* A vertex whose row never changes
carries no information about coupling, and dividing by its zero norm would give
a spurious value. The reference drops those vertices before computing and
writes NaN back into their slots, and the mask is the intersection of the
constant columns of SC and of FC.

*The diagonal is handled inconsistently.* ``calculate_sfc_gbl`` removes it only
when asked to symmetrize, while ``calculate_sfc_loc`` and ``calculate_sfc_dct``
always remove it. That asymmetry is preserved here because it changes the
numbers, and `PORTING.md` records it as a question for the group rather than
something to quietly tidy up.
"""

from __future__ import annotations

import numpy as np

SCOPES = ("global", "region", "discrete")

MIN_AREA = 10
"""Default vertex count below which a region's local coupling is NaN.

A region with only a handful of vertices gives an inflated similarity, because
two short vectors agree by chance.
"""


def _nonconstant(matrix: np.ndarray) -> np.ndarray:
    """Columns that are not constant down their length.

    Mirrors the reference's ``~all(~diff(x))``: an exact equality test, not a
    tolerance.
    """
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[1], dtype=bool)
    return ~np.all(np.diff(matrix, axis=0) == 0, axis=0)


def _usable(sc: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Vertices whose SC and FC profiles both carry information."""
    return _nonconstant(fc) & _nonconstant(sc)


def _prepare(sc, fc, triangular: bool, drop_diagonal: bool):
    """Symmetrize and clear the diagonal, following the reference's order."""
    sc = np.array(sc, dtype=np.float64, copy=True)
    fc = np.array(fc, dtype=np.float64, copy=True)
    if sc.shape != fc.shape:
        raise ValueError(f"SC is {sc.shape} but FC is {fc.shape}")
    if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
        raise ValueError(f"expected square matrices, got {sc.shape}")

    if triangular:
        sc = sc + sc.T
        fc = fc + fc.T
    if drop_diagonal or triangular:
        np.fill_diagonal(sc, 0.0)
        np.fill_diagonal(fc, 0.0)
    return sc, fc


def _cosine_rows(sc: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, ``dot(f, s) / (|f| |s|)``."""
    numerator = np.einsum("ij,ij->i", fc, sc)
    denominator = np.linalg.norm(fc, axis=1) * np.linalg.norm(sc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _pearson_rows(sc: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation, as MATLAB's ``corr2`` computes it."""
    return _cosine_rows(sc - sc.mean(axis=1, keepdims=True), fc - fc.mean(axis=1, keepdims=True))


def global_coupling(sc, fc, triangular: bool = False) -> np.ndarray:
    """Coupling of each vertex's whole-surface profile.

    Port of ``calculate_sfc_gbl.m``.

    Parameters
    ----------
    sc, fc
        Square vertex-by-vertex matrices on the same grid.
    triangular
        Symmetrize both matrices first. The reference clears the diagonal only
        in this case, and that behaviour is preserved.

    Returns
    -------
    One value per vertex, NaN where either profile is constant.

    Examples
    --------
    >>> import numpy as np
    >>> sc = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    >>> np.round(global_coupling(sc, sc), 6)
    array([1., 1., 1.])
    """
    sc, fc = _prepare(sc, fc, triangular, drop_diagonal=False)
    usable = _usable(sc, fc)

    out = np.full(sc.shape[0], np.nan)
    if usable.any():
        block = np.ix_(usable, usable)
        out[usable] = _cosine_rows(sc[block], fc[block])
    return out


def discrete_coupling(sc, fc, triangular: bool = False) -> np.ndarray:
    """Coupling of already-parcellated region profiles.

    Port of ``calculate_sfc_dct.m``. Unlike the other two this is a Pearson
    correlation, because the reference uses ``corr2``.

    Examples
    --------
    >>> import numpy as np
    >>> sc = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    >>> np.round(discrete_coupling(sc, sc), 6)
    array([1., 1., 1.])
    """
    sc, fc = _prepare(sc, fc, triangular, drop_diagonal=True)
    usable = _usable(sc, fc)

    out = np.full(sc.shape[0], np.nan)
    if usable.any():
        block = np.ix_(usable, usable)
        out[usable] = _pearson_rows(sc[block], fc[block])
    return out


def local_coupling(
    sc,
    fc,
    labels,
    min_area: int = MIN_AREA,
    triangular: bool = False,
) -> np.ndarray:
    """Coupling computed within each region rather than across the surface.

    Port of ``calculate_sfc_loc.m``. For every region, the SC and FC submatrices
    over that region's vertices are compared row by row, so the result describes
    how structure and function agree *inside* a region.

    Parameters
    ----------
    labels
        Per-vertex region label, as from :func:`sbci.load_atlas`.
    min_area
        Regions with fewer vertices than this give NaN: a similarity between
        two very short vectors is inflated.
    """
    sc, fc = _prepare(sc, fc, triangular, drop_diagonal=True)
    labels = np.asarray(labels).ravel()
    if labels.size != sc.shape[0]:
        raise ValueError(f"{labels.size} labels for {sc.shape[0]} vertices")

    usable = _usable(sc, fc)
    out = np.full(sc.shape[0], np.nan)
    if not usable.any():
        return out

    block = np.ix_(usable, usable)
    sc_u, fc_u = sc[block], fc[block]
    labels_u = labels[usable]

    result = np.zeros(sc_u.shape[0])
    for region in np.unique(labels_u):
        member = labels_u == region
        size = int(member.sum())
        if size < min_area:
            result[member] = np.nan
            continue
        sub = np.ix_(member, member)
        result[member] = _cosine_rows(sc_u[sub], fc_u[sub])

    out[usable] = result
    return out


def structure_function_coupling(sc, fc, scope: str = "global", **kwargs) -> np.ndarray:
    """Coupling map between a structural and a functional connectome.

    Parameters
    ----------
    sc, fc
        Two :class:`~sbci.ContinuousConnectome` objects on the same grid, for
        the same subject.
    scope
        ``"global"`` for one value per vertex over the whole surface,
        ``"region"`` for the within-region pattern, which needs ``labels=``,
        ``"discrete"`` for the atlas-level summary.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if sc.modality != "sc" or fc.modality != "fc":
        raise ValueError("coupling() takes a structural connectome and a functional one")
    if sc.n_vertices != fc.n_vertices:
        raise ValueError("the two connectomes are on different grids")

    sc_dense, fc_dense = sc.dense(), fc.dense()
    if scope == "global":
        return global_coupling(sc_dense, fc_dense, **kwargs)
    if scope == "discrete":
        return discrete_coupling(sc_dense, fc_dense, **kwargs)

    labels = kwargs.pop("labels", None)
    if labels is None:
        raise ValueError("scope='region' needs labels=, e.g. load_atlas('Desikan').labels")
    return local_coupling(sc_dense, fc_dense, labels, **kwargs)
