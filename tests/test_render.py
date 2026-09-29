"""The PyVista renderer: colours are right without a display, and a view renders with one."""

import numpy as np
import pytest

from sbci import render

matplotlib = pytest.importorskip("matplotlib")


def test_shading_is_dark_in_the_sulci_and_flat_on_the_wall():
    depth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    cortex = np.array([True, True, True, True, True, False])
    grey = render.shade(depth, cortex)
    assert grey[0] > grey[4]  # crowns lighter than sulci
    assert grey[5] == render.WALL_GREY
    assert (
        grey[cortex].max() <= 0.86 + 1e-9
        and grey[cortex].min() >= 0.86 - render.SHADING_DEPTH - 1e-9
    )


def test_colour_shows_the_map_only_above_threshold_and_darkens_it_in_sulci():
    grey = np.array([0.86, 0.36, 0.86, 0.86])
    values = np.array([1.0, 1.0, 0.01, np.nan])
    rgb = render.colour(values, grey, "Blues", 0.02, 1.0, threshold=0.02)
    assert not np.allclose(rgb[0], grey[0])  # shown
    assert np.allclose(rgb[2], grey[2]) and np.allclose(rgb[3], grey[3])  # left to the shading
    assert rgb[1].mean() < rgb[0].mean()  # the same colour, darker in a sulcus
    signed = render.colour(
        np.array([-1.0, 0.0, 1.0]), np.full(3, 0.86), "coolwarm", -1, 1, 0.5, True
    )
    assert np.allclose(signed[1], 0.86) and not np.allclose(signed[0], 0.86)


def test_the_headless_guard_names_the_cases_that_would_abort():
    assert render.would_abort("linux", None, "vtkXOpenGLRenderWindow")
    assert render.would_abort("linux2", "", "vtkXOpenGLRenderWindow")
    assert not render.would_abort("linux", ":0", "vtkXOpenGLRenderWindow")  # a display
    assert not render.would_abort("linux", None, "vtkOSOpenGLRenderWindow")  # software build
    assert not render.would_abort("linux", None, "vtkEGLRenderWindow")
    assert not render.would_abort("darwin", None, "vtkCocoaRenderWindow")
    assert not render.would_abort("win32", None, "vtkWin32OpenGLRenderWindow")


def test_a_view_renders_and_trims():
    pv = pytest.importorskip("pyvista")
    import os
    import sys

    import sbci

    window = render.window_class()
    assert window.endswith("RenderWindow")
    if render.would_abort(sys.platform, os.environ.get("DISPLAY"), window):
        pytest.skip(f"{window} needs a display and there is none")
    try:
        pv.Plotter(off_screen=True, window_size=[64, 48]).screenshot(return_img=True)
    except Exception as error:  # pragma: no cover - no OpenGL on this machine
        pytest.skip(f"off-screen rendering unavailable: {error}")
    surface = sbci.load_surface("inflated").hemisphere("L")
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    rgb = np.repeat(np.linspace(0.3, 0.9, vertices.shape[0])[:, None], 3, axis=1)
    image = render.render_view(vertices, surface.faces, rgb, "lateral", "L", window=(320, 240))
    assert image.shape == (240, 320, 4) and image.dtype == np.uint8
    assert image[..., 3].max() == 255  # something was drawn
    trimmed = render.trim(image)
    assert trimmed.shape[0] <= 240 and trimmed.shape[1] <= 320
    with pytest.raises(ValueError, match="view"):
        render.render_view(vertices, surface.faces, rgb, "sideways", "L")


def test_compose_lays_out_rows_and_a_colorbar():
    images = {
        ("L", "lateral"): np.zeros((10, 12, 4), np.uint8),
        ("L", "medial"): np.zeros((10, 12, 4), np.uint8),
    }
    figure = render.compose(images, ("lateral", "medial"), "Blues", 0.0, 1.0)
    assert len(figure.axes) == 3  # two panels and one colorbar
    matplotlib.pyplot.close(figure)
