"""Surface figures.

Plotting dependencies are optional on purpose: a cluster node running the
pipeline should not need a rendering stack, so ``nilearn`` and ``matplotlib``
are extras and the imports happen inside the call.

    pip install 'sbci[plotting]'

The standard neuroimaging view of a whole-brain map is four panels -- each
hemisphere seen laterally and medially -- and that is what
:func:`plot_surface` produces by default. The surface is shaded by sulcal
depth (:func:`sbci.surface.sulcal_depth`), so the folds show through the map
the way they do in FreeSurfer's and nilearn's own figures; without it a map
that is small over most of the cortex leaves the brain an unreadable white.
The medial wall, which carries no cortex, is shaded flat; a map with NaNs
there, as :func:`sbci.coupling.global_coupling` returns, renders those
vertices as that flat surface rather than at one end of the scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import spec
from .atlas import cortex_mask
from .surface import GEOMETRIES, load_surface, sulcal_depth

#: Surfaces a map can be drawn on: the grid itself, or FreeSurfer's finer
#: fsaverage meshes with the map interpolated onto them for display.
DISPLAY_MESHES = ("ico4", "fsaverage5", "fsaverage")
#: How far (mm, on the inflated surface) a grid vertex may sit from the nearest
#: vertex of the finer mesh before the two are judged not to be the same
#: anatomy. The grid's vertices are fsaverage vertices: 0.09 mm on median,
#: 0.47 mm at most.
COVERAGE_TOLERANCE = 2.0
VIEWS = ("lateral", "medial", "dorsal", "ventral", "anterior", "posterior")
#: The bundled sphere into anatomical orientation: x and y negated, a half-turn
#: about z. The pipeline's sphere is stored in its own frame, turned 180 degrees
#: about z from fsaverage's RAS, so its x and y run against the inflated
#: surface's; drawn as stored, the lateral camera (at -x for a left hemisphere)
#: shows the medial side. The stored coordinates stay as they are -- every grid
#: and warp is built on them -- and only what is handed to the renderer is
#: turned. A rotation, not a reflection, so face winding and normals are unchanged.
SPHERE_TO_ANATOMICAL = np.array([-1.0, -1.0, 1.0])

_MISSING = (
    "Plotting needs the optional dependencies. Install them with:\n    pip install 'sbci[plotting]'"
)


def _import_nilearn():
    try:
        from nilearn import plotting as nilearn_plotting
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING) from exc
    return nilearn_plotting


@dataclass(frozen=True)
class DisplayMesh:
    """A finer surface to draw the grid's maps on, with the interpolation from ico4.

    Everything is per hemisphere, left first: ``geometries[name][side]`` are the
    coordinates of FreeSurfer's ``name`` surface, ``faces[side]`` its triangles,
    ``sulc[side]`` its sulcal depth, ``weights[side]``/``indices[side]`` the
    barycentric interpolation of a grid map onto it, found by locating each of
    its sphere vertices in the grid's sphere mesh, and ``nearest[side]`` the
    mesh vertex closest to each grid vertex, for marking a grid vertex on it.
    """

    name: str
    geometries: dict
    faces: tuple
    sulc: tuple
    weights: tuple
    indices: tuple
    nearest: tuple

    @property
    def n_vertices(self) -> tuple[int, int]:
        """Vertices per hemisphere."""
        return (self.faces[0].max() + 1, self.faces[1].max() + 1)

    def interpolate(self, values) -> list[np.ndarray]:
        """A grid map (length 5124, NaN allowed) on this mesh, one array per hemisphere.

        Each vertex takes the barycentric mix of the three grid vertices around
        it. NaN grid vertices are left out of the mix, and a vertex whose
        neighbours are mostly NaN is NaN itself, so a masked map keeps a clean
        edge instead of bleeding into the medial wall.
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        half = spec.N_VERTICES_PER_HEMI
        if values.size != 2 * half:
            raise ValueError(f"map has {values.size} values but the grid has {2 * half} vertices")
        out = []
        for side, block in enumerate((values[:half], values[half:])):
            weights, indices = self.weights[side], self.indices[side]
            finite = np.isfinite(block)
            mixed = (weights * np.where(finite, block, 0.0)[indices]).sum(axis=1)
            support = (weights * finite[indices]).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append(np.where(support > 0.5, mixed / support, np.nan))
        return out


def _fsaverage_files(name: str) -> dict:
    """FreeSurfer's fsaverage surfaces through nilearn, as arrays per hemisphere."""
    try:
        from nilearn import datasets
        from nilearn import surface as nilearn_surface
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING) from exc

    keys = {"inflated": "infl", "white": "white", "pial": "pial", "sphere": "sphere"}
    arrays: dict = {"geometries": {}, "faces": {}, "sulc": {}}
    if hasattr(datasets, "fetch_surf_fsaverage"):
        bunch = datasets.fetch_surf_fsaverage(mesh=name)

        def mesh_arrays(key):
            loaded = nilearn_surface.load_surf_mesh(bunch[key])
            coordinates = getattr(loaded, "coordinates", None)
            faces = getattr(loaded, "faces", None)
            if coordinates is None:
                coordinates, faces = loaded[0], loaded[1]
            return np.asarray(coordinates, dtype=np.float64), np.asarray(faces, dtype=np.int64)

        for geometry, prefix in keys.items():
            left, right = mesh_arrays(f"{prefix}_left"), mesh_arrays(f"{prefix}_right")
            arrays["geometries"][geometry] = (left[0], right[0])
            arrays["faces"] = (left[1], right[1])
        arrays["sulc"] = tuple(
            np.asarray(nilearn_surface.load_surf_data(bunch[key]), dtype=np.float64)
            for key in ("sulc_left", "sulc_right")
        )
    else:  # pragma: no cover - newer nilearn without the fetcher
        meshes = datasets.load_fsaverage(mesh=name)
        depth = datasets.load_fsaverage_data(mesh=name, data_type="sulcal")
        for geometry in keys:
            parts = meshes[geometry].parts
            arrays["geometries"][geometry] = tuple(
                np.asarray(parts[side].coordinates, dtype=np.float64) for side in ("left", "right")
            )
            arrays["faces"] = tuple(
                np.asarray(parts[side].faces, dtype=np.int64) for side in ("left", "right")
            )
        arrays["sulc"] = tuple(
            np.asarray(depth.data.parts[side], dtype=np.float64).ravel()
            for side in ("left", "right")
        )
    return arrays


@lru_cache(maxsize=2)
def display_mesh(name: str) -> DisplayMesh:
    """Build (once per process) the finer mesh a map is drawn on.

    ``"fsaverage5"`` (10,242 vertices per hemisphere) ships with nilearn;
    ``"fsaverage"`` (163,842) is fetched by nilearn on first use into
    ``~/nilearn_data``. The grid's vertices are fsaverage vertices (each lies
    within half a millimetre of one on the inflated surface), so every grid
    vertex is first matched to its mesh vertex anatomically, the mesh's own
    sphere then gives the grid its coordinates in the mesh's frame, and each
    mesh vertex is located in the grid's spherical triangles there. The
    bundled ``sphere`` is a different parameterization and is not used.
    """
    from scipy.spatial import cKDTree

    from .alignment import MeshQuery, normalize_rows

    if name not in DISPLAY_MESHES or name == "ico4":
        raise ValueError(
            f"mesh must be one of {DISPLAY_MESHES[1:]} to interpolate onto, got {name!r}"
        )
    files = _fsaverage_files(name)
    inflated = load_surface("inflated")
    weights, indices, nearest = [], [], []
    for side, letter in enumerate("LR"):
        hemisphere = inflated.hemisphere(letter)
        gap, closest = cKDTree(files["geometries"]["inflated"][side]).query(
            np.asarray(hemisphere.vertices, dtype=np.float64)
        )
        if gap.max() > COVERAGE_TOLERANCE:
            raise ValueError(
                f"the {name} inflated surface does not contain the grid's vertices: one sits "
                f"{gap.max():.2f} mm from its nearest vertex, so the two are not the same "
                "anatomy and the map cannot be interpolated"
            )
        target = normalize_rows(files["geometries"]["sphere"][side])
        grid_points = target[closest]
        w, i = MeshQuery(grid_points, np.asarray(hemisphere.faces, dtype=np.int64)).query(target)
        weights.append(w)
        indices.append(i)
        nearest.append(np.asarray(closest, dtype=np.int64))
    return DisplayMesh(
        name=name,
        geometries=files["geometries"],
        faces=tuple(files["faces"]),
        sulc=tuple(files["sulc"]),
        weights=tuple(weights),
        indices=tuple(indices),
        nearest=tuple(nearest),
    )


def display_coordinates(part, surface: str) -> np.ndarray:
    """One hemisphere's vertices as the renderer should see them, in anatomical orientation.

    The anatomical surfaces are returned as stored; the bundled ``sphere`` is
    turned by :data:`SPHERE_TO_ANATOMICAL`, so that the lateral view shows the
    lateral side. Always a fresh array.
    """
    coordinates = np.array(part.vertices, dtype=np.float64)
    if surface == "sphere":
        coordinates *= SPHERE_TO_ANATOMICAL
    return coordinates


def _colour_range(values, vmin=None, vmax=None, symmetric=None):
    """``(vmin, vmax, symmetric)`` for finite ``values``: the bounds a figure draws with.

    ``symmetric`` defaults to whether the values take both signs. A symmetric
    scale is centred on zero: one bound given is mirrored -- ``vmin=-0.5``
    alone means ``(-0.5, 0.5)``, not ``-0.5`` up to the map's largest value --
    and none given spans the largest magnitude both ways; both given are kept
    as they are. A constant map collapses the range, and the renderer divides
    by ``vmax - vmin`` (its colorbar takes its own bounds from the data and so
    cannot be fixed by widening them alone), so a small window is opened.
    """
    values = np.asarray(values, dtype=np.float64)
    if symmetric is None:
        symmetric = bool(values.min() < 0 < values.max())
    if vmin is None or vmax is None:
        if symmetric:
            given = vmin if vmin is not None else vmax
            limit = abs(given) if given is not None else np.abs(values).max()
            auto_min, auto_max = -limit, limit
        else:
            auto_min, auto_max = values.min(), values.max()
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    if vmax <= vmin:
        pad = abs(vmin) * 1e-6 or 1e-6
        vmin, vmax = vmin - pad, vmax + pad
    return float(vmin), float(vmax), bool(symmetric)


def plot_surface(
    surface_map,
    surface: str = "inflated",
    connectome=None,
    views: tuple[str, ...] = ("lateral", "medial"),
    cmap: str = "coolwarm",
    threshold: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    symmetric: bool | None = None,
    shading: bool = True,
    mesh: str = "ico4",
    engine: str = "matplotlib",
    **kwargs,
):
    """Render a per-vertex map on a cortical surface.

    Parameters
    ----------
    surface_map
        One value per vertex on the computational grid, length 5124. NaN marks
        a vertex with no value, such as the medial wall.
    surface
        Geometry to draw on, one of :data:`sbci.surface.GEOMETRIES`. The
        anatomical surfaces are in fsaverage's RAS frame, which the views
        assume. The bundled ``"sphere"`` is stored in the pipeline's own frame,
        a half-turn about z from RAS; it is turned into RAS for display
        (:data:`SPHERE_TO_ANATOMICAL`), the stored coordinates untouched.
    connectome
        Optional :class:`~sbci.ContinuousConnectome`; its mask is applied so
        that masked vertices are not coloured.
    views
        Which views to draw for each hemisphere. The default gives the usual
        four-panel figure.
    cmap, threshold, vmin, vmax
        Passed to nilearn. ``vmin``/``vmax`` default to the map's own range;
        on a symmetric scale one of them alone is mirrored about zero.
    symmetric
        Centre the colour scale on zero. Defaults to true when the map has
        both signs, which is what a coupling map usually wants.
    shading
        Shade the surface by sulcal depth from the white surface, darker in
        the sulci, with the map's colours laid over it (``bg_on_data`` in
        nilearn's terms). Outside cortex -- the medial wall, from the
        connectome's mask or the bundled atlas -- the shading is flat, so the
        wall reads as the cut surface it is. ``False`` draws the bare mesh
        under the map. Pass your own ``bg_map`` through ``kwargs`` to shade by
        something else.
    mesh
        The resolution to draw at. ``"ico4"`` draws the grid itself (5124
        vertices, faceted). ``"fsaverage5"`` (10,242 vertices per hemisphere,
        shipped with nilearn) and ``"fsaverage"`` (163,842, fetched once by
        nilearn into ``~/nilearn_data``) interpolate the map onto FreeSurfer's
        surfaces through the shared spherical registration and shade them by
        FreeSurfer's own sulcal depth. Display only: the values stay on ico4.
        The finest mesh takes about a minute per view to render.
    engine
        ``"matplotlib"`` draws through nilearn, flat-shaded. ``"pyvista"``
        renders each view off-screen with smooth normals, a light kit and a
        specular highlight (the ``render`` extra), then lays the views out in
        the same figure: a few seconds a view on a laptop, about twenty in
        software on a cluster node without a display (VTK 9.4 or later; an
        older VTK there needs its OSMesa build, ``pip install --extra-index-url
        https://wheels.vtk.org vtk-osmesa``).

    Returns
    -------
    The matplotlib figure.

    Examples
    --------
    >>> figure = cc.plot(cc.seed(vertex=1234))       # doctest: +SKIP
    >>> figure.savefig("seed.png", dpi=150)           # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING) from exc

    nilearn_plotting = _import_nilearn()

    if surface not in GEOMETRIES:
        raise ValueError(f"surface must be one of {GEOMETRIES}, got {surface!r}")
    for view in views:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    if not views:
        raise ValueError("pass at least one view")

    if mesh not in DISPLAY_MESHES:
        raise ValueError(f"mesh must be one of {DISPLAY_MESHES}, got {mesh!r}")
    if engine not in ("matplotlib", "pyvista"):
        raise ValueError(f"engine must be 'matplotlib' or 'pyvista', got {engine!r}")
    values = np.asarray(surface_map, dtype=np.float64).ravel()
    grid_mesh = load_surface(surface)
    if values.size != grid_mesh.n_vertices:
        raise ValueError(
            f"map has {values.size} values but the {surface} surface has "
            f"{grid_mesh.n_vertices} vertices"
        )

    if connectome is not None:
        values = np.where(np.asarray(connectome.mask, dtype=bool), values, np.nan)

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("the map has no finite values to plot")

    vmin, vmax, symmetric = _colour_range(values[finite], vmin, vmax, symmetric)
    # A constant map collapses the colour range: drop the colorbar, which
    # conveys nothing for a single value (see _colour_range for the window).
    constant = float(values[finite].min()) == float(values[finite].max())

    half = grid_mesh.n_vertices // 2
    # Flat outside cortex: FreeSurfer fills the medial wall with a surface
    # that has curvature of its own, which would otherwise be shaded as if
    # it were folded cortex. The connectome's mask says where cortex is;
    # without one, the bundled medial wall does.
    cortex = np.asarray(connectome.mask, dtype=bool) if connectome is not None else cortex_mask()
    if mesh == "ico4":
        per_hemisphere = {"L": values[:half], "R": values[half:]}
        hemispheres = {side: grid_mesh.hemisphere(side) for side in ("L", "R")}
        meshes = {
            side: (display_coordinates(part, surface), np.array(part.faces))
            for side, part in hemispheres.items()
        }
        depth = np.where(cortex, sulcal_depth(), 0.0)
        depths = {"L": depth[:half], "R": depth[half:]}
    else:
        display = display_mesh(mesh)
        per_hemisphere = dict(zip(("L", "R"), display.interpolate(values), strict=True))
        cortex_fraction = display.interpolate(cortex.astype(np.float64))
        meshes = {
            side: (display.geometries[surface][k], display.faces[k]) for k, side in enumerate("LR")
        }
        depths = {
            side: np.where(np.nan_to_num(cortex_fraction[k]) > 0.5, display.sulc[k], 0.0)
            for k, side in enumerate("LR")
        }
    if engine == "pyvista":
        from . import render

        on_cortex = (
            {"L": cortex[:half], "R": cortex[half:]}
            if mesh == "ico4"
            else {side: np.nan_to_num(cortex_fraction[k]) > 0.5 for k, side in enumerate("LR")}
        )
        images = {}
        for side, hemisphere_values in per_hemisphere.items():
            grey = (
                render.shade(depths[side], on_cortex[side])
                if shading
                else np.full(hemisphere_values.shape, render.WALL_GREY + 0.2)
            )
            rgb = render.colour(
                hemisphere_values, grey, cmap, vmin, vmax, threshold=threshold, symmetric=symmetric
            )
            coordinates, faces = meshes[side]
            for view in views:
                images[(side, view)] = render.render_view(coordinates, faces, rgb, view, side)
        figure = render.compose(images, tuple(views), cmap, vmin, vmax, colorbar=not constant)
        if title:
            figure.suptitle(title, fontsize=15)
        return figure

    if shading and "bg_map" not in kwargs:
        backgrounds = depths
        kwargs.setdefault("bg_on_data", True)
        kwargs.setdefault("alpha", 1.0)
    else:
        backgrounds = {}

    figure, axes = plt.subplots(
        len(per_hemisphere),
        len(views),
        figsize=(5 * len(views), 4 * len(per_hemisphere)),
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    axes = np.atleast_2d(axes).reshape(len(per_hemisphere), len(views))

    for row, (side, hemisphere_values) in enumerate(per_hemisphere.items()):
        coordinates, faces = meshes[side]
        for column, view in enumerate(views):
            nilearn_plotting.plot_surf(
                # nilearn recentres the coordinates in place, so hand it a copy each time
                surf_mesh=(coordinates.copy(), faces.copy()),
                surf_map=hemisphere_values,
                hemi="left" if side == "L" else "right",
                view=view,
                cmap=cmap,
                threshold=threshold,
                vmin=vmin,
                vmax=vmax,
                colorbar=(column == len(views) - 1) and not constant,
                axes=axes[row, column],
                figure=figure,
                **({"bg_map": backgrounds[side]} if side in backgrounds else {}),
                **kwargs,
            )
            axes[row, column].set_title(f"{side} {view}", fontsize=12)

    # nilearn places colorbar ticks at the threshold and at quarter points of
    # what is left, which gives 0.27, 0.51, 0.76 and, on a symmetric scale, a
    # -0.05 printed over a 0.05. Round ticks at readable spacing instead.
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    grid = {id(axis) for axis in axes.ravel()}
    for extra in figure.axes:
        if id(extra) not in grid:  # a colorbar
            extra.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            extra.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            extra.tick_params(labelsize=11)

    if title:
        figure.suptitle(title, fontsize=15)
    return figure
