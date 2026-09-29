"""Draw the example figures in docs/figures/ from the synthetic data.

Nothing here needs lab data: every figure comes from ``sbci.example()`` and
``sbci.example_cohort()``, so the set regenerates anywhere the package and its
plotting extra are installed::

    python scripts/make_figures.py docs/figures

The cohort figures fit a rank-4 FPCA on ten subjects and register two of them
with ConSEAL, which takes a few minutes; run it in a batch job on a cluster.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

import sbci  # noqa: E402
from sbci import render  # noqa: E402
from sbci.plotting import display_mesh  # noqa: E402

# One blue for magnitude, orange for a second series, muted ink for text and axes.
BLUE, ORANGE, INK, MUTED, RULE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "sbci_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# On the shaded surface the map is thresholded, so its ramp can start visible.
SURFACE_RAMP = LinearSegmentedColormap.from_list(
    "sbci_surface", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)
#: Streamlines for the single-subject figures: ten times the example's default,
#: so the profile and the region matrix are smooth rather than speckled.
N_STREAMLINES = 200_000
VIEWS = ("lateral", "medial")
#: Surface figures are drawn on FreeSurfer's fsaverage (163,842 vertices per
#: hemisphere) with the map interpolated onto it, not on the faceted grid.
DISPLAY_MESH = "fsaverage"
#: Surfaces are rendered through PyVista (smooth normals, lighting, specular highlight).
DISPLAY_ENGINE = "pyvista"
#: Marker radii (mm) for the endpoints and the seed on the rendered smoothing figure.
DOT_SIZE, SEED_SIZE = 1.1, 2.4
TITLE_SIZE = 17  # the surface figures are large; a 12-point title reads as a footnote on them

plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": RULE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(figure, out: Path, name: str, dpi: int = 200) -> None:
    path = out / name
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path} ({path.stat().st_size // 1024} KB)", flush=True)


SEED = 1234  # a left temporal vertex


def rendered_map(
    values,
    views=VIEWS,
    cmap=SURFACE_RAMP,
    vmin=0.0,
    vmax=1.0,
    threshold=None,
    symmetric=False,
    markers=(),
):
    """Grid ``values`` on the display mesh, drawn as ``plot(engine="pyvista")`` draws them.

    ``markers`` are ``(hemisphere, grid_vertices, colour, radius)`` tuples: lit
    spheres of ``radius`` mm at those grid vertices (global indices) on
    hemisphere 0 (left) or 1 (right), which ``plot`` has no option for.
    """
    from sbci.atlas import cortex_mask
    from sbci.surface import vertex_normals

    display = display_mesh(DISPLAY_MESH)
    cortex = cortex_mask()
    half = cortex.size // 2
    values = np.where(cortex, np.asarray(values, dtype=np.float64), np.nan)
    per_hemisphere = display.interpolate(values)
    on_cortex = [
        np.nan_to_num(fraction) > 0.5 for fraction in display.interpolate(cortex.astype(np.float64))
    ]
    images = {}
    for k, side in enumerate("LR"):
        vertices, faces = display.geometries["inflated"][k], display.faces[k]
        grey = render.shade(display.sulc[k], on_cortex[k])
        rgb = render.colour(per_hemisphere[k], grey, cmap, vmin, vmax, threshold, symmetric)
        lift = 1.2 * vertex_normals(vertices, faces)
        points = []
        for hemisphere, grid_vertices, colour, radius in markers:
            if hemisphere == k:
                local = np.unique(display.nearest[k][np.asarray(grid_vertices) - k * half])
                points.append((vertices[local] + lift[local], colour, radius))
        for view in views:
            images[(side, view)] = render.render_view(
                vertices, faces, rgb, view, side, points=points
            )
    return render.compose(images, tuple(views), cmap, vmin, vmax)


def seed_profile(out: Path, cc) -> None:
    profile = cc.seed(vertex=SEED)
    profile = profile / profile.max()  # unit-mass densities are 1e-10 per pair; show the shape
    figure = rendered_map(
        profile, threshold=0.02, vmin=0.02, vmax=1.0, markers=[(0, [SEED], ORANGE, SEED_SIZE)]
    )
    figure.suptitle(
        f"Where one vertex connects to: the density of streamlines between vertex {SEED} "
        "(orange dot, left temporal cortex)\nand every other vertex, relative to the strongest",
        fontsize=TITLE_SIZE,
        y=1.11,  # two lines at this size need clearance above the panel labels
    )
    save(figure, out, "seed_profile.png")


#: Streamlines per draw in the smoothing figure: the example's default, and a
#: tenth of the single-subject figures.
SMOOTHING_DRAWS = 20_000


def _partners(cc, vertex: int) -> np.ndarray:
    """Global vertex index of the far end of every streamline touching ``vertex``."""
    ends = cc.endpoints
    first = ends.vtx_in + ends.n_per_hemi * ends.surf_in.astype(np.int64)
    second = ends.vtx_out + ends.n_per_hemi * ends.surf_out.astype(np.int64)
    return np.r_[second[first == vertex], first[second == vertex]]


def smoothing_power(out: Path) -> None:
    """The endpoints touching one vertex against its smoothed density, in three draws."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from sbci.atlas import cortex_mask
    from sbci.surface import vertex_normals

    # The same bundles every time, only the streamlines redrawn: a test-retest pair
    # at the example's default count, and one draw with ten times as many.
    still = {"effect": 0.0, "variation": 0.0, "anatomy": 0.0}
    draws = sbci.example_cohort(n_subjects=3, seed=0, n_streamlines=SMOOTHING_DRAWS, **still)
    reference = sbci.example_cohort(n_subjects=1, seed=0, n_streamlines=N_STREAMLINES, **still)
    subjects = [draws.connectomes[1], draws.connectomes[2], reference.connectomes[0]]
    columns = [
        f"draw A, {SMOOTHING_DRAWS:,} streamlines",
        f"draw B, {SMOOTHING_DRAWS:,} streamlines",
        f"{N_STREAMLINES:,} streamlines",
    ]
    cortex = cortex_mask()
    partners = [_partners(cc, SEED) for cc in subjects]
    counts = [
        np.bincount(p, minlength=cc.n_vertices) for p, cc in zip(partners, subjects, strict=True)
    ]
    smooth = [cc.seed(vertex=SEED) for cc in subjects]
    smooth = [values / values.max() for values in smooth]

    def correlation(a, b):
        return round(float(np.corrcoef(a[cortex], b[cortex])[0, 1]), 2) + 0.0

    r_raw = correlation(counts[0], counts[1])
    r_smooth = correlation(smooth[0], smooth[1])
    r_reference = correlation(smooth[0], smooth[2])
    touching = [int(p.size) for p in partners]
    print(
        f"  vertex {SEED}: {touching} streamlines touch it; r between draws: raw {r_raw:.2f}, "
        f"smoothed {r_smooth:.2f}; smoothed draw A against the {N_STREAMLINES:,} map: "
        f"{r_reference:.2f}",
        flush=True,
    )

    display = display_mesh(DISPLAY_MESH)
    half = sbci.load_surface("inflated").n_vertices // 2
    vertices, faces = display.geometries["inflated"][0], display.faces[0]
    lift = 1.2 * vertex_normals(vertices, faces)  # markers sit just above the surface
    on_cortex = np.nan_to_num(display.interpolate(cortex.astype(float))[0]) > 0.5
    grey = render.shade(display.sulc[0], on_cortex)
    smooth_hi = [display.interpolate(np.where(cortex, values, np.nan))[0] for values in smooth]
    seed_vertex = display.nearest[0][SEED]

    figure, axes = plt.subplots(2, 3, figsize=(15, 7.6), layout="constrained")
    for row in range(2):
        for column in range(3):
            axis = axes[row, column]
            if row == 0:
                # The far ends of the streamlines that touch the vertex; the
                # renderer hides those on the far side of the hemisphere.
                local = partners[column][partners[column] < half]  # left hemisphere
                local = np.unique(display.nearest[0][local])
                rgb = np.repeat(grey[:, None], 3, axis=1)
                markers = [(vertices[local] + lift[local], BLUE, DOT_SIZE)]
                kind = f"raw: the {touching[column]:,} streamlines touching the vertex"
            else:
                rgb = render.colour(smooth_hi[column], grey, SURFACE_RAMP, 0.02, 1.0, 0.02)
                markers = []
                kind = "smoothed density"
            markers.append((vertices[[seed_vertex]] + lift[[seed_vertex]], ORANGE, SEED_SIZE))
            image = render.render_view(vertices, faces, rgb, "lateral", "L", points=markers)
            axis.imshow(render.trim(image), interpolation="lanczos")
            axis.set_axis_off()
            axis.set_title(f"{columns[column]}\n{kind}", fontsize=13)
    bar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.02, vmax=1.0), cmap=SURFACE_RAMP),
        ax=axes.ravel().tolist(),
        shrink=0.45,
        pad=0.02,
        format="%g",
    )
    bar.set_label("smoothed density, relative to the strongest vertex", color=MUTED)
    bar.outline.set_visible(False)
    figure.suptitle(
        "Smoothing: from a handful of endpoints to a map you can compare\n"
        f"vertex {SEED} (orange dot) in three draws of one synthetic subject, left hemisphere",
        fontsize=TITLE_SIZE,
    )
    figure.supxlabel(
        f"Draws A and B agree across the cortex at r = {r_raw:.2f} as raw counts and "
        f"r = {r_smooth:.2f} once smoothed;\nsmoothed draw A matches the "
        f"{N_STREAMLINES:,}-streamline map at r = {r_reference:.2f}.",
        fontsize=13,
    )
    save(figure, out, "smoothing_power.png")


def region_matrix(out: Path, cc) -> None:
    atlas = sbci.load_atlas("Desikan")
    matrix = cc.to_atlas(atlas, how="mass")
    # A log scale over the full range spans nine decades and turns the matrix
    # into speckle; the top four decades carry the structure, and anything
    # fainter is drawn as "no connection".
    top = float(matrix.max())
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    image = axis.imshow(
        np.where(matrix > 0, matrix, np.nan),
        cmap=BLUE_RAMP,
        norm=LogNorm(vmin=top * 1e-4, vmax=top),
        interpolation="nearest",
    )
    half = atlas.n_regions // 2
    for position in (half - 0.5,):
        axis.axhline(position, color="white", linewidth=2)
        axis.axvline(position, color="white", linewidth=2)
    axis.set_xticks([half / 2 - 0.5, half + half / 2 - 0.5])
    axis.set_xticklabels(["left hemisphere", "right hemisphere"])
    axis.set_yticks([half / 2 - 0.5, half + half / 2 - 0.5])
    axis.set_yticklabels(["left", "right"], rotation=90, va="center")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    bar.set_label("mass between the two regions (log scale, top four decades)", color=MUTED)
    bar.outline.set_visible(False)
    axis.set_title("The synthetic subject parcellated with the Desikan atlas (68 regions)")
    save(figure, out, "region_matrix.png")


def coupling(out: Path, sc, fc) -> None:
    values = sc.coupling(fc)
    # Coupling is high almost everywhere on the synthetic pair, so a symmetric
    # scale would paint the whole surface one shade: run the ramp from zero.
    figure = sc.plot(
        values,
        views=VIEWS,
        cmap=SURFACE_RAMP,
        symmetric=False,
        vmin=0.0,
        mesh=DISPLAY_MESH,
        engine=DISPLAY_ENGINE,
    )
    figure.suptitle(
        "Structure-function coupling: at each vertex, the cosine similarity of its SC and FC "
        "profiles",
        fontsize=TITLE_SIZE,
        y=1.03,
    )
    save(figure, out, "coupling.png")


def cohort_figures(out: Path) -> None:
    t = time.time()
    cohort = sbci.example_cohort(n_subjects=10, seed=0)
    print(f"  cohort built in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    reduction = sbci.reduce(cohort.connectomes, rank=4)
    print(f"  FPCA in {time.time() - t:.0f}s", flush=True)
    result = sbci.local_test(reduction.scores, cohort.age)
    found = result.significant()
    adjusted = result.adjusted.round(5).tolist()
    print(f"  significant components: {found.tolist()}, adjusted p {adjusted}")
    subject = cohort.connectomes[0]

    figure = subject.plot(
        cohort.truth,
        views=VIEWS,
        cmap=SURFACE_RAMP,
        threshold=0.02,
        vmin=0.02,
        vmax=1.0,
        mesh=DISPLAY_MESH,
        engine=DISPLAY_ENGINE,
    )
    figure.suptitle(
        "The planted bundle: the field whose weight scales with age (cohort.truth)",
        fontsize=TITLE_SIZE,
        y=1.03,
    )
    save(figure, out, "cohort_truth.png")

    effect = result.effect_map(reduction, alpha=0.05)
    effect = effect / np.abs(effect).max()
    figure = subject.plot(
        effect,
        views=VIEWS,
        cmap="coolwarm",
        symmetric=True,
        threshold=0.05,
        mesh=DISPLAY_MESH,
        engine=DISPLAY_ENGINE,
    )
    correlation = np.corrcoef(effect, cohort.truth)[0, 1]
    figure.suptitle(
        "Recovered: the effect map of the component that tracks age\n"
        f"(correlation with the planted field {correlation:.2f})",
        fontsize=TITLE_SIZE,
        y=1.11,  # two lines at this size need clearance above the panel labels
    )
    save(figure, out, "cohort_effect.png")

    k = int(found[0])
    scores = reduction.scores[:, k]
    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    axis.scatter(cohort.age, scores, s=64, color=BLUE, zorder=3, label="subjects")
    slope, intercept = np.polyfit(cohort.age, scores, 1)
    grid = np.linspace(cohort.age.min(), cohort.age.max(), 2)
    axis.plot(grid, slope * grid + intercept, color=ORANGE, linewidth=2, label="least-squares fit")
    r = np.corrcoef(cohort.age, scores)[0, 1]
    axis.set_xlabel("synthetic age (years)")
    axis.set_ylabel(f"score on component {k + 1}")
    axis.set_title(
        f"The component that tracks age: r = {r:.2f}, adjusted p = {result.adjusted[k]:.1e}",
        loc="left",
    )
    axis.legend(frameon=False, loc="lower right")
    save(figure, out, "cohort_scores.png")


def angles(p, q):
    return np.degrees(np.arccos(np.clip((p * q).sum(axis=1), -1.0, 1.0)))


def alignment_recovery(out: Path) -> None:
    """Deform a subject by a known smooth warp and measure how much of it each method undoes."""
    from sbci.alignment import Encore, MeshQuery, _hemisphere_grids, align
    from sbci.conseal import DEFAULT_WARP_ORDER, EndpointConnectome, StationaryWarp, default_grids
    from sbci.smoothing import endpoint_positions

    subject = sbci.example(seed=0)
    lh, rh = default_grids()
    rng = np.random.default_rng(7)

    def known_warp(grid):
        warp = StationaryWarp(grid)
        coefficients = rng.standard_normal(grid.basis.shape[1])
        displacement = (coefficients[None, :, None] * grid.basis).sum(axis=1)
        displacement *= 0.07 / np.linalg.norm(displacement, axis=1).max()  # four degrees at most
        assert warp.compose(displacement)
        return warp

    lh_true, rh_true = known_warp(lh), known_warp(rh)
    carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
    original = carrier.positions()
    carrier.warp(lh_true, rh_true)
    deformed = sbci.example(seed=0)
    deformed.endpoints = carrier.to_endpoints()
    deformed = deformed.smooth(kernel="shk", mask_medial_wall=True)
    moved = EndpointConnectome.from_endpoints(deformed.endpoints, lh, rh).positions()
    before = np.r_[angles(original[0], moved[0]), angles(original[1], moved[1])]

    t = time.time()
    grids, rotations = _hemisphere_grids(DEFAULT_WARP_ORDER, return_rotations=True)
    encore = align(
        [subject, deformed],
        template=Encore(*grids).root(subject.dense()),
        grids=grids,
        max_iterations=50,
    )
    warp = encore.warps[1]
    warped = (warp.lh_vertices @ rotations[0], warp.rh_vertices @ rotations[1])
    # ENCORE's warp is a pull-back, so undo the deformation through its inverse: locate each
    # moved endpoint on the warped mesh and carry those weights to the unwarped vertices.
    fixed = []
    ends = (carrier.hemisphere_in, carrier.hemisphere_out)
    for points, hemispheres in zip(moved, ends, strict=True):
        placed = np.empty_like(points)
        for side, grid in enumerate((lh, rh)):
            pick = np.asarray(hemispheres) == side
            weights, indices = MeshQuery(warped[side], grid.faces).query(points[pick])
            combined = np.einsum("nk,nkj->nj", weights, grid.vertices[indices])
            placed[pick] = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        fixed.append(placed)
    after_encore = np.r_[angles(original[0], fixed[0]), angles(original[1], fixed[1])]
    print(
        f"  ENCORE in {time.time() - t:.0f}s: {before.mean():.2f} -> {after_encore.mean():.2f} deg",
        flush=True,
    )

    t = time.time()
    conseal = sbci.endpoints_align(
        [subject, deformed],
        template=0,
        max_iterations=60,
        threshold=1e-7,
        delta=0.1,
        step_clamp=float("inf"),
        viscosity=0.0,
    )
    back = endpoint_positions(conseal.aligned_endpoints(1))
    after_conseal = np.r_[angles(original[0], back[0]), angles(original[1], back[1])]
    print(
        f"  ConSEAL in {time.time() - t:.0f}s: "
        f"{before.mean():.2f} -> {after_conseal.mean():.2f} deg",
        flush=True,
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    bins = np.linspace(0, 3.0, 61)
    for values, color, label in (
        (before, MUTED, "as deformed"),
        (after_encore, BLUE, "after ENCORE"),
        (after_conseal, ORANGE, "after ConSEAL"),
    ):
        left.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{label}: mean {values.mean():.2f} deg",
        )
    left.set_xlabel("distance of each endpoint from where it started (degrees)")
    left.set_ylabel("endpoints")
    left.set_title("Endpoints moved by a known warp, and put back", loc="left")
    left.legend(frameon=False)
    for trace, color, label in (
        (np.asarray(encore.traces[1]), BLUE, "ENCORE"),
        (np.asarray(conseal.costs[1]), ORANGE, "ConSEAL"),
    ):
        relative = trace / trace[0]
        right.plot(range(len(relative)), relative, color=color, linewidth=2, label=label)
        right.annotate(
            f"{label} {relative[-1]:.2f}",
            (len(relative) - 1, relative[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=color,
            fontsize=9,
        )
    right.set_xlabel("iteration")
    right.set_ylabel("cost, relative to the start")
    right.set_ylim(0, 1.05)
    right.set_title("Cost of the registration", loc="left")
    right.legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Alignment with a known answer: a synthetic subject registered onto an undeformed copy "
        "of itself",
        fontsize=14,
        y=1.03,
    )
    save(figure, out, "alignment_recovery.png")


def spherical_kernel(out: Path) -> None:
    from sbci.smoothing import kernel_cutoff, spherical_heat_kernel

    angles = np.radians(np.linspace(0, 25, 1001))
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    for sigma, color, height in ((0.005, BLUE, 0.16), (0.01, ORANGE, 0.06)):
        values = spherical_heat_kernel(np.cos(angles), sigma)
        values = values / values[0]
        cutoff = np.degrees(kernel_cutoff(sigma))
        axis.plot(np.degrees(angles), values, color=color, linewidth=2, label=f"sigma = {sigma}")
        axis.plot([cutoff, cutoff], [0.0, height - 0.02], color=color, linewidth=1)
        axis.annotate(
            f"cutoff {cutoff:.1f} deg",
            (cutoff, height),
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
        )
    axis.set_xlabel("angle from the endpoint (degrees)")
    axis.set_ylabel("kernel value, relative to its peak")
    axis.set_xlim(0, 25)
    axis.set_title(
        "The spherical kernel concon applies, at the released bandwidth and twice it", loc="left"
    )
    axis.legend(frameon=False)
    save(figure, out, "spherical_kernel.png")


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    print("single subject", flush=True)
    sc, fc = sbci.example("sc", n_streamlines=N_STREAMLINES), sbci.example("fc")
    seed_profile(out, sc)
    smoothing_power(out)
    region_matrix(out, sc)
    coupling(out, sc, fc)
    print("kernel", flush=True)
    spherical_kernel(out)
    print("cohort", flush=True)
    cohort_figures(out)
    print("alignment", flush=True)
    alignment_recovery(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
