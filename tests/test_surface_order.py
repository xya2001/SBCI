"""The shipped mesh must be in the same vertex order as the data.

This is the test that was missing when the toolkit's inflated and white ico4
meshes were first bundled: they are in a different vertex order from the
connectivity and the atlases, so every figure drawn on them was silently wrong
while every other test still passed.

The check is that a parcellation drawn on the mesh is spatially contiguous.
Almost every edge of the mesh should join two vertices of the same region,
because regions are patches of cortex rather than scattered vertices. A mesh in
the wrong order scores near the chance level instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci import list_atlases, load_atlas, load_surface
from sbci.surface import GEOMETRIES

#: Agreement corrected for chance, as a Cohen's-kappa-style statistic:
#:
#:     adjusted = (observed - chance) / (1 - chance)
#:
#: A raw ratio does not work across atlases. An atlas with three regions scores
#: 0.45 by chance alone, so even perfect contiguity is only twice chance, while
#: a 400-region atlas has a chance level near zero. The adjusted form is 1 for a
#: perfectly contiguous parcellation and 0 for a scattered one whatever the
#: number of regions. The correctly ordered sphere gives 0.84 for Desikan; the
#: wrongly ordered inflated mesh gave 0.27.
MIN_ADJUSTED_AGREEMENT = 0.4

#: Atlases finer than the grid can represent. ico4 has 5124 vertices of which
#: about 4685 are cortex, so a 1000-parcel atlas gets roughly five vertices per
#: parcel and most of its edges necessarily cross a boundary. That caps
#: agreement for reasons of resolution, not vertex order: Schaefer900 and
#: Schaefer1000 reach 0.47-0.49 on the correctly ordered mesh, against 0.84 for
#: Desikan. The converter already reports that ico4 loses one parcel outright
#: from each of them.
RESOLUTION_LIMITED = {
    "Schaefer2018_900Parcels_7Networks_order",
    "Schaefer2018_1000Parcels_7Networks_order",
}


def _edges(faces: np.ndarray) -> np.ndarray:
    return np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])


def _edge_agreement(labels: np.ndarray, edges: np.ndarray) -> float:
    """Fraction of mesh edges whose endpoints share a region."""
    valid = (labels[edges[:, 0]] > 0) & (labels[edges[:, 1]] > 0)
    kept = edges[valid]
    if not len(kept):
        return 0.0
    return float((labels[kept[:, 0]] == labels[kept[:, 1]]).mean())


def _shuffled(labels: np.ndarray, edges: np.ndarray, seed: int) -> float:
    """The same measure with region membership randomized, keeping sizes."""
    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    assigned = permuted > 0
    values = permuted[assigned]
    rng.shuffle(values)
    permuted[assigned] = values
    return _edge_agreement(permuted, edges)


def _adjusted_agreement(labels: np.ndarray, edges: np.ndarray, seed: int) -> float:
    """Edge agreement corrected for the level expected by chance."""
    observed = _edge_agreement(labels, edges)
    chance = _shuffled(labels, edges, seed)
    if chance >= 1.0:
        return 0.0
    return (observed - chance) / (1.0 - chance)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("atlas_name", ["Desikan", "Schaefer200", "Glasser"])
def test_parcellations_are_contiguous_on_the_shipped_mesh(geometry, atlas_name):
    """A mesh in the wrong vertex order scatters every atlas drawn on it."""
    surface = load_surface(geometry)
    atlas = load_atlas(atlas_name)
    assert atlas.labels.size == surface.n_vertices

    edges = _edges(surface.faces)
    adjusted = _adjusted_agreement(atlas.labels, edges, seed=0)

    assert adjusted > MIN_ADJUSTED_AGREEMENT, (
        f"{atlas_name} on the {geometry} mesh has chance-corrected agreement "
        f"{adjusted:.4f}; the mesh and the labels are probably in different "
        "vertex orders"
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_mesh_edges_are_short_and_even(geometry):
    """Neighbouring vertices must be close together.

    A permuted mesh joins vertices from opposite sides of the brain, so its
    longest edge dwarfs its mean.
    """
    surface = load_surface(geometry)
    edges = _edges(surface.faces)
    lengths = np.linalg.norm(surface.vertices[edges[:, 0]] - surface.vertices[edges[:, 1]], axis=1)
    assert lengths.max() < 5 * lengths.mean(), (
        f"{geometry}: longest edge {lengths.max():.1f} against a mean of "
        f"{lengths.mean():.1f}; the vertex order looks wrong"
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_every_atlas_is_contiguous(geometry):
    """Sweep all 44 atlases rather than a sample."""
    surface = load_surface(geometry)
    edges = _edges(surface.faces)

    failures = []
    for name in list_atlases():
        adjusted = _adjusted_agreement(load_atlas(name).labels, edges, seed=1)
        floor = 0.3 if name in RESOLUTION_LIMITED else MIN_ADJUSTED_AGREEMENT
        if adjusted <= floor:
            failures.append(f"{name} ({adjusted:.4f})")

    assert not failures, "atlases scattered on the mesh: " + ", ".join(failures)


def test_every_geometry_is_bundled():
    assert set(GEOMETRIES) == {"inflated", "white", "pial", "sphere"}
    for name in GEOMETRIES:
        assert load_surface(name).n_vertices == 5124


def test_an_unknown_geometry_lists_the_alternatives():
    with pytest.raises(ValueError, match="bundled geometries are"):
        load_surface("midthickness")
