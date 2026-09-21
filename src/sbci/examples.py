"""A runnable connectome, so the package can be tried before the data lands.

The five-minute start in the README begins with ``sbci download``, which needs
the WP1 data release (SPEC_QUESTIONS.md item 6). Until that exists, anyone who
installs this package has nothing to run it on. :func:`example` closes that
gap: it builds a connectome with the right shape, the right grid and the real
medial wall, so every call in the README works today.

**The connectivity is synthetic.** It is generated from a handful of smooth
bumps on the sphere, not measured from anyone's brain, and the metadata says
so: ``pipeline_version`` reads ``synthetic-example`` and
``streamline_count`` is zero. Use it to learn the API, to write tests, and to
check that a plotting stack works. Never use it to make a claim about brains.

What *is* real: the ico4 grid, the vertex areas, the medial-wall mask (taken
from the bundled Desikan atlas), and the file format. A file written by
:func:`example` passes ``sbci validate``.
"""

from __future__ import annotations

import numpy as np

from . import grid, spec
from .atlas import load_atlas
from .connectome import ContinuousConnectome
from .metadata import template
from .surface import load_surface

#: Smooth bumps used to build the synthetic structure. Enough that a
#: parcellation into a few hundred regions still shows variation.
N_COMPONENTS = 48

#: Concentration of each bump. About 25 gives a spatial scale of roughly 20
#: degrees, the scale at which real cortical connectivity varies.
CONCENTRATION = 25.0

#: Inter-hemispheric connectivity is genuinely sparser than within a
#: hemisphere, so the synthetic one is damped rather than left symmetric.
CROSS_HEMISPHERE = 0.25


def _bumps(vertices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth, positive, localized fields on the sphere.

    Each column is a von Mises-Fisher bump around a random direction, which is
    the cheapest way to get a field that is smooth on a sphere without forming
    an ``n x n`` distance matrix.
    """
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    centres = rng.normal(size=(N_COMPONENTS, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return np.exp(CONCENTRATION * (unit @ centres.T - 1.0)).astype(np.float32)


def _metadata(modality: str, version: str):
    """Complete metadata that says plainly the file is synthetic."""
    common = dict(
        normalization="unit-mass" if modality == "sc" else "none",
        registration_reference="fsaverage",
        pipeline_version=f"synthetic-example/{version}",
        container_version="none (synthetic example, not pipeline output)",
    )
    if modality == "sc":
        common.update(
            streamline_count=0,
            streamline_weighting="none (synthetic example, no tractography)",
            kernel="none (synthetic example, not smoothed)",
            bandwidth=0.0,
        )
    else:
        common.update(fc_nuisance_model="none (synthetic example, no confounds to regress)")
    return template(modality, **common)


def example(modality: str = "sc", seed: int = 0) -> ContinuousConnectome:
    """Build a synthetic connectome on the real ico4 grid.

    Everything in the README runs against it, so the package can be tried
    without waiting for the data release. The connectivity is generated, not
    measured -- see this module's docstring.

    Parameters
    ----------
    modality
        ``"sc"`` for a nonnegative density normalized to unit mass, or
        ``"fc"`` for a signed correlation matrix built from synthetic
        timeseries.
    seed
        Seed for the generator, so the same call always gives the same file.

    Returns
    -------
    ContinuousConnectome
        Valid, and it round-trips through :meth:`ContinuousConnectome.save`.

    Examples
    --------
    >>> import sbci
    >>> cc = sbci.example()
    >>> cc.n_vertices
    5124
    >>> cc.modality
    'sc'
    >>> dense = cc.dense().astype("float64")   # a density integrates to one
    >>> round(float(cc.area @ dense @ cc.area), 6)
    1.0
    >>> float(dense[~cc.mask].sum())           # nothing on the medial wall
    0.0
    >>> bool(sbci.example("fc").data.min() < 0)   # FC is signed
    True
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")

    rng = np.random.default_rng(seed)
    sphere = load_surface("sphere")
    n = sphere.n_vertices
    fields = _bumps(sphere.vertices, rng)

    # The mask is real: the medial wall carries no Desikan region.
    mask = load_atlas("Desikan").labels != 0
    area = _vertex_areas(sphere)

    left = np.arange(n) < n // 2
    if modality == "sc":
        weights = rng.gamma(2.0, 1.0, size=N_COMPONENTS).astype(np.float32)
        dense = (fields * weights) @ fields.T
        # Inter-hemispheric structural connectivity is the sparser half.
        dense[np.ix_(left, ~left)] *= CROSS_HEMISPHERE
        dense[np.ix_(~left, left)] *= CROSS_HEMISPHERE
    else:
        # Correlate synthetic timeseries, which is how FC is actually made.
        signals = fields @ rng.normal(size=(N_COMPONENTS, 240)).astype(np.float32)
        signals += rng.normal(scale=0.35, size=signals.shape).astype(np.float32)
        signals -= signals.mean(axis=1, keepdims=True)
        signals /= np.linalg.norm(signals, axis=1, keepdims=True)
        dense = signals @ signals.T

    # The medial wall carries no connectivity, as the spec requires.
    dense[~mask, :] = 0.0
    dense[:, ~mask] = 0.0
    np.fill_diagonal(dense, 0.0)

    if modality == "sc":
        # A density integrates to one against the area weights, which is the
        # quantity `sbci validate` checks -- not the bare sum of the entries.
        dense = dense.astype(np.float64)
        dense /= area @ dense @ area

    condensed = grid.to_condensed(dense)

    from . import __version__

    return ContinuousConnectome(
        data=condensed.astype(np.float32),
        area=area,
        mask=mask,
        metadata=_metadata(modality, __version__),
        coords=sphere.vertices,
    )


def _vertex_areas(surface) -> np.ndarray:
    """Barycentric vertex areas of the mesh, normalized to sum to one.

    A third of each triangle's area goes to each of its corners, which is the
    standard lumped-mass rule and matches what the pipeline stores.
    """
    v, f = surface.vertices, surface.faces
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.zeros(surface.n_vertices, dtype=np.float64)
    np.add.at(areas, f.ravel(), np.repeat(face_area / 3.0, 3))
    return areas / areas.sum()
