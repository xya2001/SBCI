"""Surface figures.

Plotting dependencies are optional on purpose: a cluster node running the
pipeline should not need a rendering stack, so ``nilearn`` and ``matplotlib``
are extras and the imports happen inside the call.

    pip install 'sbci[plotting]'

The standard neuroimaging view of a whole-brain map is four panels -- each
hemisphere seen laterally and medially -- and that is what
:func:`plot_surface` produces by default. A map with NaNs, as
:func:`sbci.coupling.global_coupling` returns for the medial wall, renders
those vertices in the background colour rather than at one end of the scale.
"""

from __future__ import annotations

import numpy as np

from .surface import GEOMETRIES, load_surface

SURFACES = GEOMETRIES
VIEWS = ("lateral", "medial", "dorsal", "ventral", "anterior", "posterior")

_MISSING = (
    "Plotting needs the optional dependencies. Install them with:\n    pip install 'sbci[plotting]'"
)


def _import_nilearn():
    try:
        from nilearn import plotting as nilearn_plotting
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING) from exc
    return nilearn_plotting


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
    **kwargs,
):
    """Render a per-vertex map on a cortical surface.

    Parameters
    ----------
    surface_map
        One value per vertex on the computational grid, length 5124. NaN marks
        a vertex with no value, such as the medial wall.
    surface
        Geometry to draw on, one of :data:`SURFACES`.
    connectome
        Optional :class:`~sbci.ContinuousConnectome`; its mask is applied so
        that masked vertices are not coloured.
    views
        Which views to draw for each hemisphere. The default gives the usual
        four-panel figure.
    cmap, threshold, vmin, vmax
        Passed to nilearn. ``vmin``/``vmax`` default to the map's own range.
    symmetric
        Centre the colour scale on zero. Defaults to true when the map has
        both signs, which is what a coupling map usually wants.

    Returns
    -------
    The matplotlib figure.

    Examples
    --------
    >>> figure = cc.plot(cc.seed(vertex=1234))       # doctest: +SKIP
    >>> figure.savefig("seed.png", dpi=150)           # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    nilearn_plotting = _import_nilearn()

    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    for view in views:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    if not views:
        raise ValueError("pass at least one view")

    values = np.asarray(surface_map, dtype=np.float64).ravel()
    mesh = load_surface(surface)
    if values.size != mesh.n_vertices:
        raise ValueError(
            f"map has {values.size} values but the {surface} surface has {mesh.n_vertices} vertices"
        )

    if connectome is not None:
        values = np.where(np.asarray(connectome.mask, dtype=bool), values, np.nan)

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("the map has no finite values to plot")

    if symmetric is None:
        symmetric = bool(values[finite].min() < 0 < values[finite].max())
    if vmin is None or vmax is None:
        limit = np.abs(values[finite]).max()
        if symmetric:
            auto_min, auto_max = -limit, limit
        else:
            auto_min, auto_max = values[finite].min(), values[finite].max()
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax

    # A constant map collapses the colour range. The renderer divides by
    # (vmax - vmin), and its colorbar derives its own bounds from the data and
    # so cannot be fixed by widening vmin/vmax alone. Open a small window for
    # the surface and drop the colorbar, which conveys nothing for a single
    # value anyway.
    constant = float(values[finite].min()) == float(values[finite].max())
    if vmax <= vmin:
        pad = abs(vmin) * 1e-6 or 1e-6
        vmin, vmax = vmin - pad, vmax + pad

    half = mesh.n_vertices // 2
    per_hemisphere = {"L": values[:half], "R": values[half:]}

    figure, axes = plt.subplots(
        len(per_hemisphere),
        len(views),
        figsize=(5 * len(views), 4 * len(per_hemisphere)),
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    axes = np.atleast_2d(axes).reshape(len(per_hemisphere), len(views))

    for row, (side, hemisphere_values) in enumerate(per_hemisphere.items()):
        hemisphere = mesh.hemisphere(side)
        for column, view in enumerate(views):
            nilearn_plotting.plot_surf(
                surf_mesh=(hemisphere.vertices, hemisphere.faces),
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
                **kwargs,
            )
            axes[row, column].set_title(f"{side} {view}", fontsize=10)

    if title:
        figure.suptitle(title)
    return figure
