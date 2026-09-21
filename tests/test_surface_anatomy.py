"""Named regions must sit where anatomy says they do.

This is the test that was missing when anatomical surfaces were first bundled.
A mesh whose vertices carry the wrong labels still passes every
self-consistency check -- short even edges, contiguous parcels, plausible
extents -- because a consistent relabelling preserves all of them. The first
attempt at these surfaces passed all three and was anatomically scrambled.

What catches it is a claim about the world rather than about the mesh: the
superior frontal gyrus is anterior to lateral occipital cortex, the temporal
pole is anterior and inferior, the cuneus is medial. Those relations hold for
any correct left hemisphere and fail immediately for a permuted one.

fsaverage is RAS: +x right, +y anterior, +z superior. The left hemisphere sits
at negative x, so "medial" means x nearer zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci import load_atlas, load_surface
from sbci.spec import N_VERTICES_PER_HEMI as N
from sbci.surface import ANATOMICAL, NATIVE_SPACE

atlas = load_atlas("Desikan")
LABEL_OF = {name: index + 1 for index, name in enumerate(atlas.names)}


def centroid(region: str, vertices: np.ndarray) -> np.ndarray:
    member = atlas.labels == LABEL_OF[region]
    assert member.sum() >= 5, f"{region} has too few vertices to locate"
    return vertices[member].mean(axis=0)


@pytest.fixture(params=sorted(ANATOMICAL))
def anatomical(request):
    """Each bundled surface that claims to be anatomical."""
    return request.param, load_surface(request.param).vertices


def test_superiorfrontal_is_anterior_to_lateraloccipital(anatomical):
    name, v = anatomical
    assert centroid("LH_superiorfrontal", v)[1] > centroid("LH_lateraloccipital", v)[1], name


def test_temporalpole_is_anterior_to_lateraloccipital(anatomical):
    name, v = anatomical
    assert centroid("LH_temporalpole", v)[1] > centroid("LH_lateraloccipital", v)[1], name


def test_superiorfrontal_is_superior_to_temporalpole(anatomical):
    name, v = anatomical
    assert centroid("LH_superiorfrontal", v)[2] > centroid("LH_temporalpole", v)[2], name


@pytest.mark.parametrize("surface", sorted(NATIVE_SPACE))
def test_cuneus_is_more_medial_than_lateraloccipital(surface):
    """The cuneus sits on the medial wall; lateral occipital does not.

    Meaningful only where coordinates are true anatomical positions: the
    inflated surface is a deformation, so its x says nothing about medial.
    """
    v = load_surface(surface).vertices
    cuneus = abs(centroid("LH_cuneus", v)[0])
    lateral = abs(centroid("LH_lateraloccipital", v)[0])
    assert cuneus < lateral, f"{surface}: cuneus |x| {cuneus:.1f} vs {lateral:.1f}"


@pytest.mark.parametrize("surface", sorted(NATIVE_SPACE))
def test_the_two_hemispheres_are_on_opposite_sides(surface):
    v = load_surface(surface).vertices
    assert v[:N, 0].mean() < 0 < v[N:, 0].mean(), f"{surface}: hemispheres not separated"


def test_left_regions_are_in_the_left_hemisphere(anatomical):
    """A region named LH_ must have all its vertices in the first half."""
    name, _ = anatomical
    for region in ("LH_superiorfrontal", "LH_cuneus", "LH_insula"):
        member = np.flatnonzero(atlas.labels == LABEL_OF[region])
        assert member.max() < N, f"{name}: {region} reaches into the right hemisphere"


@pytest.mark.parametrize("surface", sorted(ANATOMICAL))
def test_edges_are_short_and_even(surface):
    """A permuted mesh joins distant vertices, giving a huge max/mean ratio."""
    mesh = load_surface(surface)
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    assert lengths.max() < 5 * lengths.mean(), (
        f"{surface}: longest edge {lengths.max():.1f} against mean {lengths.mean():.1f}"
    )


def test_anatomical_surfaces_carry_real_dimensions():
    """A human hemisphere pair spans roughly 120-180 mm in each direction."""
    v = load_surface("white").vertices
    extent = v.max(axis=0) - v.min(axis=0)
    assert (extent > 90).all() and (extent < 260).all(), f"white extent {extent}"
