"""Drawing a grid map on FreeSurfer's finer fsaverage meshes (display only).

fsaverage5 ships inside nilearn, so these need no network; they skip without
the plotting extra.
"""

import numpy as np
import pytest

nilearn = pytest.importorskip("nilearn")
matplotlib = pytest.importorskip("matplotlib")

import sbci  # noqa: E402
from sbci.plotting import DISPLAY_MESHES, display_mesh, plot_surface  # noqa: E402

HALF = 2562


def test_the_grid_vertices_are_fsaverage5_vertices_and_interpolation_is_exact_there():
    display = display_mesh("fsaverage5")
    assert display.n_vertices == (10242, 10242)
    inflated = sbci.load_surface("inflated")
    for side in (0, 1):
        grid = np.asarray(inflated.vertices[side * HALF : (side + 1) * HALF], dtype=np.float64)
        matched = display.geometries["inflated"][side][display.nearest[side]]
        assert np.linalg.norm(grid - matched, axis=1).max() < 1.0  # mm
        assert len(set(display.nearest[side].tolist())) == HALF
    # the mesh vertex matched to a grid vertex takes exactly that grid value
    values = np.arange(2 * HALF, dtype=np.float64)
    left, right = display.interpolate(values)
    np.testing.assert_allclose(left[display.nearest[0]], values[:HALF], atol=1e-6)
    np.testing.assert_allclose(right[display.nearest[1]], values[HALF:], atol=1e-6)


def test_interpolation_stays_inside_the_range_and_keeps_constants():
    display = display_mesh("fsaverage5")
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.0, 1.0, size=2 * HALF)
    for part in display.interpolate(values):
        assert part.min() >= values.min() - 1e-12 and part.max() <= values.max() + 1e-12
    for part in display.interpolate(np.full(2 * HALF, 3.0)):
        np.testing.assert_allclose(part, 3.0)


def test_a_masked_map_keeps_its_edge():
    display = display_mesh("fsaverage5")
    values = np.where(sbci.atlas.cortex_mask(), 1.0, np.nan)
    left, right = display.interpolate(values)
    finite = np.concatenate([np.isfinite(left), np.isfinite(right)])
    assert finite.any() and not finite.all()
    np.testing.assert_allclose(left[np.isfinite(left)], 1.0)


def test_plot_surface_draws_on_fsaverage5():
    cc = sbci.example()
    figure = plot_surface(
        cc.seed(vertex=1234), connectome=cc, mesh="fsaverage5", views=("lateral",)
    )
    assert len(figure.axes) >= 2
    matplotlib.pyplot.close(figure)


def test_unknown_meshes_are_refused():
    assert DISPLAY_MESHES[0] == "ico4"
    with pytest.raises(ValueError, match="mesh"):
        plot_surface(np.zeros(2 * HALF), mesh="fsaverage9")
    with pytest.raises(ValueError, match="mesh"):
        display_mesh("ico4")
