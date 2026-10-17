"""Publication-grade rendering of surface maps: smooth normals, lighting, specular highlights.

matplotlib paints each triangle flat, with no light. This module renders the
same surfaces through PyVista (VTK) off-screen -- per-vertex normals, a
three-point light kit, Phong shading with a specular highlight, supersampled
anti-aliasing -- and hands the views back as images, so :func:`plot_surface`
can lay them out with the same titles and colorbar as before. It is the
optional ``render`` extra (``pip install sbci[render]``); on a machine without
a display VTK 9.4 or later renders in software by itself.

The map is coloured as in the matplotlib path: the surface is shaded by
sulcal depth, the map's colours are laid over it and darkened in the sulci,
values below the threshold are left to the shading, and the medial wall is
flat grey.
"""

from __future__ import annotations

import os
import sys

import numpy as np

#: Pixels rendered per view, before it is placed in the figure.
WINDOW = (1400, 1050)
#: Material: a matte cortex with a soft highlight.
MATERIAL = {"ambient": 0.18, "diffuse": 0.88, "specular": 0.3, "specular_power": 28.0}
#: Grey of the medial wall and of cortex outside the map.
WALL_GREY = 0.58
#: How dark the sulci get relative to the crowns (0 = no shading).
SHADING_DEPTH = 0.5
#: Camera directions per view, for a left hemisphere: (position, view-up), in
#: fsaverage's RAS frame. The bundled sphere is not in that frame;
#: :func:`sbci.plotting.display_coordinates` turns it before it gets here.
_CAMERAS = {
    "lateral": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "medial": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "dorsal": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "ventral": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "anterior": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "posterior": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}

_SOFTWARE_BUILD = (
    "on a machine without a display VTK 9.4 or later renders in software by itself; with an "
    "older VTK install its OSMesa build in place of vtk: pip uninstall -y vtk; "
    "pip install --extra-index-url https://wheels.vtk.org vtk-osmesa"
)
_MISSING = (
    f"rendering needs the render extra: pip install 'sbci[render]' (PyVista); {_SOFTWARE_BUILD}"
)
_NO_DISPLAY = (
    "this VTK draws through an X display and there is none (DISPLAY is unset), so it would "
    f"abort rather than render; {_SOFTWARE_BUILD}, or run under a virtual display (xvfb-run)"
)


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING) from exc
    pv.OFF_SCREEN = True
    if would_abort(sys.platform, os.environ.get("DISPLAY"), window_class()):
        raise RuntimeError(_NO_DISPLAY)
    return pv


def window_class() -> str:
    """The render window VTK would open on this machine, by class name.

    ``vtkXOpenGLRenderWindow`` needs an X display; ``vtkOSOpenGLRenderWindow``
    (the software build) and ``vtkEGLRenderWindow`` do not.
    """
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - registers the platform's window
    from vtkmodules.vtkRenderingCore import vtkRenderWindow

    return type(vtkRenderWindow()).__name__


def would_abort(platform: str, display: str | None, window: str) -> bool:
    """Whether rendering would abort: Linux, no display, and a window that needs one."""
    return platform.startswith("linux") and not display and window.startswith("vtkX")


def shade(depth: np.ndarray, cortex: np.ndarray) -> np.ndarray:
    """Grey level per vertex from sulcal depth: dark in the sulci, flat on the wall."""
    depth = np.asarray(depth, dtype=np.float64)
    cortex = np.asarray(cortex, dtype=bool)
    grey = np.full(depth.shape, WALL_GREY)
    if cortex.any():
        low, high = np.percentile(depth[cortex], [2, 98])
        span = high - low if high > low else 1.0
        normalized = np.clip((depth[cortex] - low) / span, 0.0, 1.0)  # 1 = deepest sulcus
        grey[cortex] = 0.86 - SHADING_DEPTH * normalized
    return grey


def colour(
    values: np.ndarray,
    grey: np.ndarray,
    cmap,
    vmin: float,
    vmax: float,
    threshold: float | None = None,
    symmetric: bool = False,
) -> np.ndarray:
    """RGB in ``[0, 1]`` per vertex: the map's colours over the shading, or the shading alone.

    A vertex shows the map where its value is finite and, when ``threshold`` is
    given, at least that far from zero in magnitude, as nilearn thresholds; the map's colour is
    darkened by the shading so the folds show through, as ``bg_on_data`` does
    in nilearn. Elsewhere the vertex is its grey.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    values = np.asarray(values, dtype=np.float64)
    grey = np.asarray(grey, dtype=np.float64)
    cmap = colormaps[cmap] if isinstance(cmap, str) else cmap
    rgb = np.repeat(grey[:, None], 3, axis=1)
    shown = np.isfinite(values)
    if threshold is not None:
        shown &= np.abs(values) >= threshold
    if shown.any():
        mapped = cmap(Normalize(vmin=vmin, vmax=vmax, clip=True)(values[shown]))[:, :3]
        light = 0.55 + 0.45 * (grey[shown] - (0.86 - SHADING_DEPTH)) / SHADING_DEPTH
        rgb[shown] = mapped * np.clip(light, 0.4, 1.0)[:, None]
    return rgb


def render_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    view: str = "lateral",
    hemisphere: str = "L",
    window: tuple[int, int] = WINDOW,
    points: list | None = None,
    zoom: float = 1.0,
) -> np.ndarray:
    """One hemisphere from one direction as an RGBA image (``height, width, 4`` uint8).

    ``points`` are markers to draw on the surface: ``(coordinates, colour,
    radius)`` triples, drawn as lit spheres of that radius in the mesh's units
    (millimetres on an anatomical surface), so they keep their size whatever
    the window. The camera looks along the axis of ``view`` with a parallel
    projection, taking the vertices to be in fsaverage's RAS frame (the bundled
    sphere is not: pass it through :func:`sbci.plotting.display_coordinates`
    first); a right hemisphere's lateral and medial views are mirrored from
    the left's.
    """
    pv = _import_pyvista()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if view not in _CAMERAS:
        raise ValueError(f"view must be one of {sorted(_CAMERAS)}, got {view!r}")
    cells = np.column_stack([np.full(faces.shape[0], 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(vertices, cells)
    mesh.point_data["rgb"] = np.clip(np.asarray(rgb, dtype=np.float64) * 255, 0, 255).astype(
        np.uint8
    )
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, split_vertices=False)

    plotter = pv.Plotter(off_screen=True, window_size=list(window), lighting="light kit")
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="rgb",
        rgb=True,
        smooth_shading=True,
        show_scalar_bar=False,
        interpolate_before_map=True,
        **MATERIAL,
    )
    for coordinates, marker_colour, radius in points or ():
        cloud = pv.PolyData(np.asarray(coordinates, dtype=np.float64).reshape(-1, 3))
        spheres = cloud.glyph(
            geom=pv.Sphere(radius=float(radius), theta_resolution=24, phi_resolution=24),
            scale=False,
            orient=False,
        )
        plotter.add_mesh(
            spheres,
            color=marker_colour,
            smooth_shading=True,
            specular=0.4,
            specular_power=20.0,
        )
    direction, up = _CAMERAS[view]
    if hemisphere == "R" and view in ("lateral", "medial"):
        direction = (-direction[0], direction[1], direction[2])
    centre = np.asarray(mesh.center)
    extent = float(
        np.linalg.norm(
            np.asarray(mesh.bounds).reshape(3, 2)[:, 1]
            - np.asarray(mesh.bounds).reshape(3, 2)[:, 0]
        )
    )
    plotter.camera_position = [
        tuple(centre + 3.0 * extent * np.asarray(direction)),
        tuple(centre),
        up,
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.parallel_scale = 0.36 * extent / zoom
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:  # pragma: no cover - depends on the OpenGL build
        pass
    image = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()
    return np.asarray(image)


def trim(image: np.ndarray, margin: int = 12) -> np.ndarray:
    """Crop the transparent border of a rendered view, keeping ``margin`` pixels."""
    alpha = image[..., 3] if image.shape[-1] == 4 else np.any(image < 250, axis=-1) * 255
    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return image
    top, bottom = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, image.shape[0])
    left, right = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, image.shape[1])
    return image[top:bottom, left:right]


def compose(
    images: dict,
    views: tuple,
    cmap,
    vmin: float,
    vmax: float,
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Lay rendered views out as ``plot_surface`` does: rows L and R, one column per view.

    ``images[(side, view)]`` are RGBA arrays. Returns the matplotlib figure,
    with a colorbar for the map's scale on the right of each row.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    sides = [side for side in ("L", "R") if any(key[0] == side for key in images)]
    figure, axes = plt.subplots(
        len(sides),
        len(views),
        figsize=figsize or (5 * len(views), 4 * len(sides)),
        layout="constrained",
        squeeze=False,
    )
    for row, side in enumerate(sides):
        for column, view in enumerate(views):
            axis = axes[row, column]
            axis.imshow(trim(images[(side, view)]), interpolation="lanczos")
            axis.set_axis_off()
            axis.set_title(f"{side} {view}", fontsize=12)
        if colorbar:
            cmap_object = colormaps[cmap] if isinstance(cmap, str) else cmap
            bar = figure.colorbar(
                ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_object),
                ax=axes[row, :].tolist(),
                shrink=0.6,
                pad=0.02,
            )
            bar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            bar.ax.tick_params(labelsize=11)
            bar.outline.set_visible(False)
    return figure
