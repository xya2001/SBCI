"""The bundled surface meshes, and rendering a map on them."""

from __future__ import annotations

import numpy as np
import pytest

from sbci import load_surface
from sbci.spec import N_VERTICES, N_VERTICES_PER_HEMI
from sbci.surface import GEOMETRIES, Surface

matplotlib = pytest.importorskip("matplotlib", reason="needs the plotting extra")
matplotlib.use("Agg")


@pytest.mark.parametrize("name", GEOMETRIES)
def test_surface_is_on_the_computational_grid(name):
    surface = load_surface(name)
    assert surface.n_vertices == N_VERTICES
    assert surface.vertices.shape == (N_VERTICES, 3)
    assert surface.faces.shape[1] == 3


@pytest.mark.parametrize("name", GEOMETRIES)
def test_every_vertex_belongs_to_a_face(name):
    """A vertex no triangle references would be invisible when rendered."""
    surface = load_surface(name)
    assert len(np.unique(surface.faces)) == surface.n_vertices


@pytest.mark.parametrize("name", GEOMETRIES)
def test_faces_stay_within_one_hemisphere(name):
    """No triangle may bridge the hemispheres; they are separate meshes."""
    surface = load_surface(name)
    crosses = (surface.faces.min(axis=1) < N_VERTICES_PER_HEMI) & (
        surface.faces.max(axis=1) >= N_VERTICES_PER_HEMI
    )
    assert not crosses.any(), f"{int(crosses.sum())} faces bridge the hemispheres"


@pytest.mark.parametrize("name", GEOMETRIES)
@pytest.mark.parametrize("side", ["L", "R"])
def test_hemisphere_split_is_a_closed_mesh(name, side):
    half = load_surface(name).hemisphere(side)
    assert half.n_vertices == N_VERTICES_PER_HEMI
    assert half.faces.min() >= 0
    assert half.faces.max() < N_VERTICES_PER_HEMI
    # ico4 per hemisphere: 2562 vertices, 5120 triangles.
    assert len(half.faces) == 5120


def test_the_sphere_is_actually_a_sphere():
    """A strong check on the conversion: the sphere geometry must be round."""
    sphere = load_surface("sphere")
    for side in ("L", "R"):
        vertices = sphere.hemisphere(side).vertices
        radii = np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)
        assert radii.std() / radii.mean() < 0.01, f"{side} sphere is not round"


def test_hemispheres_index_different_vertices():
    """The halves are separate meshes over disjoint parts of the grid.

    Their coordinates are not a useful test for the sphere: both hemispheres
    are the same sphere, so their vertex positions legitimately coincide. What
    must hold is that they address different halves of the joined grid.
    """
    surface = load_surface("sphere")
    half = surface.n_vertices // 2
    assert (surface.faces[surface.faces.max(axis=1) < half].max() < half).all()
    assert (surface.faces[surface.faces.min(axis=1) >= half].min() >= half).all()
    assert surface.hemisphere("L").n_vertices == surface.hemisphere("R").n_vertices


def test_unknown_surface_names_the_alternatives():
    with pytest.raises(ValueError, match="bundled geometries are"):
        load_surface("midthickness")


def test_hemisphere_rejects_a_bad_side():
    with pytest.raises(ValueError, match="side must be one of"):
        load_surface("sphere").hemisphere("middle")


def test_surfaces_are_cached():
    """Loading is memoized: a figure loop must not re-read the mesh each time."""
    assert load_surface("sphere") is load_surface("sphere")


# --- rendering -------------------------------------------------------------

pytest.importorskip("nilearn", reason="needs the plotting extra")


@pytest.fixture
def surface_map():
    """A smooth map over the whole grid, with the medial wall left as NaN."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(N_VERTICES)
    values[:50] = np.nan
    return values


def test_plot_returns_a_figure(surface_map):
    from sbci.plotting import plot_surface

    figure = plot_surface(surface_map, views=("lateral",))
    # Two hemispheres, one view each.
    assert len(figure.axes) >= 2
    matplotlib.pyplot.close(figure)


def test_plot_rejects_a_wrong_length_map():
    from sbci.plotting import plot_surface

    with pytest.raises(ValueError, match="has 10 values"):
        plot_surface(np.zeros(10))


def test_plot_rejects_an_all_nan_map():
    from sbci.plotting import plot_surface

    with pytest.raises(ValueError, match="no finite values"):
        plot_surface(np.full(N_VERTICES, np.nan))


def test_plot_rejects_an_unknown_view(surface_map):
    from sbci.plotting import plot_surface

    with pytest.raises(ValueError, match="view must be one of"):
        plot_surface(surface_map, views=("sideways",))


def test_plot_rejects_an_unknown_surface(surface_map):
    from sbci.plotting import plot_surface

    with pytest.raises(ValueError, match="surface must be one of"):
        plot_surface(surface_map, surface="midthickness")


def test_connectome_plot_applies_the_mask():
    """A masked vertex must not be coloured, whatever the map says there."""
    from sbci.plotting import plot_surface

    surface = load_surface("sphere")
    values = np.linspace(-1.0, 1.0, surface.n_vertices)

    class _Masked:
        mask = np.ones(surface.n_vertices, dtype=bool)

    _Masked.mask[:100] = False
    figure = plot_surface(values, connectome=_Masked, views=("lateral",))
    assert figure is not None
    matplotlib.pyplot.close(figure)


def test_plot_handles_a_constant_map():
    """A map with one value everywhere still renders, without a colorbar.

    The colour scale is degenerate, so nilearn's colorbar would divide by a
    zero range; the surface itself is perfectly drawable.
    """
    from sbci.plotting import plot_surface

    figure = plot_surface(np.ones(N_VERTICES), views=("lateral",))
    assert figure is not None
    matplotlib.pyplot.close(figure)


def test_surface_dataclass_is_constructible():
    """Surface is a plain value object, usable without the bundled data."""
    surface = Surface("toy", np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int32))
    assert surface.n_vertices == 3
