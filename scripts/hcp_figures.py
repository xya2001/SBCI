"""Draw the documentation figures from an HCP example cohort.

    python scripts/hcp_figures.py hcp-ya docs/figures \
        [--skip-cohort | --only-cohort | --surfaces-only | --only-recovery | --only-migration]
        [--timeseries /path/to/fc_ts.npz]
        [--full-cohort DIR [--traits CSV] [--rank 20] [--candidates 1]]
        [--dataset NAME]

The cohort directory is what ``sbci download hcp-ya`` writes, or what
``tools/build_hcp_cohort.py`` builds: ``manifest.csv`` (``subject`` and, where
the cohort has them, ``sex`` and ``age_bin``) and one ``<subject>_sc.h5``,
with endpoints, per subject, plus ``<subject>_fc.h5`` where there is FC. The
coupling figure needs the FC and the warp-migration figure a resting-state
time series (``--timeseries``); without them both are skipped. Titles name the
dataset from the subject ids (``sub-HCA...`` is HCP-Aging, ``sub-`` and six
digits HCP Young Adult) unless ``--dataset`` names it. Every surface is drawn
on fsaverage.

The single-subject figures take a few minutes. The cohort figures register
every subject with ENCORE and ConSEAL, about an hour and a half for ten on
eight cores; run this in a batch job.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (  # noqa: E402
    BLUE,
    BLUE_RAMP,
    DISPLAY_ENGINE,
    DISPLAY_MESH,
    DOT_SIZE,
    INK,
    MUTED,
    ORANGE,
    SEED,
    SEED_SIZE,
    SURFACE_RAMP,
    TITLE_SIZE,
    VIEWS,
    angles,
    rendered_map,
    save,
)

import sbci  # noqa: E402
from sbci import render  # noqa: E402
from sbci.connectome import ContinuousConnectome  # noqa: E402
from sbci.plotting import display_mesh  # noqa: E402
from sbci.smoothing import Endpoints  # noqa: E402

#: The cohort's name in figure titles: :func:`dataset_of`, or ``--dataset``.
DATASET = "HCP Young Adult"


def dataset_of(directory: Path) -> str:
    """The cohort's name for the titles, read off its first subject id."""
    with open(directory / "manifest.csv") as handle:
        subject = next(csv.DictReader(handle))["subject"]
    if subject.startswith("sub-HCA"):
        return "HCP-Aging"
    if subject.startswith("sub-HCD"):
        return "HCP-Development"
    return "HCP Young Adult"


def load_cohort(directory: Path):
    """The subjects in manifest order, their FC paths (``None`` where absent) and ages.

    Ages are NaN when the manifest has no ``age_years`` column, as the young
    adult cohort's has not; FC is absent for it too.
    """
    rows = list(csv.DictReader(open(directory / "manifest.csv")))
    subjects = [sbci.load(directory / f"{row['subject']}_sc.h5") for row in rows]
    functional = [
        path if (path := directory / f"{row['subject']}_fc.h5").exists() else None for row in rows
    ]
    ages = np.array([float(row.get("age_years") or "nan") for row in rows])
    return subjects, functional, ages


def subset(endpoints: Endpoints, index: np.ndarray) -> Endpoints:
    """The streamlines at ``index``, positions included."""
    fields = {name: getattr(endpoints, name)[index] for name in ("surf_in", "surf_out")}
    fields.update({name: getattr(endpoints, name)[index] for name in ("vtx_in", "vtx_out")})
    for name in ("tri_in", "tri_out", "bary_in", "bary_out"):
        value = getattr(endpoints, name)
        fields[name] = None if value is None else value[index]
    return Endpoints(n_per_hemi=endpoints.n_per_hemi, **fields)


def resmoothed(cc: ContinuousConnectome, endpoints: Endpoints) -> ContinuousConnectome:
    """The density these endpoints give under the default kernel."""
    carrier = ContinuousConnectome(
        data=np.zeros_like(cc.data),
        area=cc.area,
        mask=cc.mask,
        metadata=cc.metadata,
        coords=cc.coords,
        endpoints=endpoints,
    )
    return carrier.smooth(kernel="shk", mask_medial_wall=True)


# --- one subject ---------------------------------------------------------------------


def seed_profile(out: Path, cc) -> None:
    profile = cc.seed(vertex=SEED)
    profile = profile / profile.max()
    figure = rendered_map(
        profile, threshold=0.02, vmin=0.02, vmax=1.0, markers=[(0, [SEED], ORANGE, SEED_SIZE)]
    )
    figure.suptitle(
        f"Where one vertex connects to: an {DATASET} subject's density of streamlines between "
        f"vertex {SEED}\n(orange dot, left temporal cortex) and every other vertex, relative to "
        "the strongest",
        fontsize=TITLE_SIZE,
        y=1.11,
    )
    save(figure, out, "seed_profile.png")


def region_matrix(out: Path, cc) -> None:
    atlas = sbci.load_atlas("Desikan")
    matrix = cc.to_atlas(atlas, how="mass")
    top = float(matrix.max())
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    image = axis.imshow(
        np.where(matrix > 0, matrix, np.nan),
        cmap=BLUE_RAMP,
        norm=LogNorm(vmin=top * 1e-4, vmax=top),
        interpolation="nearest",
    )
    half = atlas.n_regions // 2
    axis.axhline(half - 0.5, color="white", linewidth=2)
    axis.axvline(half - 0.5, color="white", linewidth=2)
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
    axis.set_title(f"An {DATASET} subject parcellated with the Desikan atlas (68 regions)")
    save(figure, out, "region_matrix.png")


def coupling(out: Path, sc, fc) -> None:
    values = sc.coupling(fc)
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
        f"Structure-function coupling in an {DATASET} subject: at each vertex, the cosine "
        "similarity\nof its SC and FC profiles",
        fontsize=TITLE_SIZE,
        y=1.11,
    )
    save(figure, out, "coupling.png")


def smoothing_power(out: Path, cc) -> None:
    """Two random halves of one subject's streamlines against all of them."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from sbci.atlas import cortex_mask
    from sbci.surface import vertex_normals

    ends = cc.endpoints
    rng = np.random.default_rng(0)
    order = rng.permutation(ends.n_streamlines)
    halves = (np.sort(order[: ends.n_streamlines // 2]), np.sort(order[ends.n_streamlines // 2 :]))
    draws = [subset(ends, index) for index in halves]
    t = time.time()
    smoothed = [resmoothed(cc, draw) for draw in draws]
    print(
        f"  two halves of {ends.n_streamlines:,} streamlines re-smoothed in {time.time() - t:.0f}s"
    )
    subjects = [smoothed[0], smoothed[1], cc]
    labels = [
        f"half A, {halves[0].size:,} streamlines",
        f"half B, {halves[1].size:,} streamlines",
        f"all {ends.n_streamlines:,} streamlines",
    ]
    cortex = cortex_mask()

    def partners(endpoints):
        first = endpoints.vtx_in + endpoints.n_per_hemi * endpoints.surf_in.astype(np.int64)
        second = endpoints.vtx_out + endpoints.n_per_hemi * endpoints.surf_out.astype(np.int64)
        return np.r_[second[first == SEED], first[second == SEED]]

    partner = [partners(draws[0]), partners(draws[1]), partners(ends)]
    counts = [np.bincount(p, minlength=cc.n_vertices) for p in partner]
    smooth = [s.seed(vertex=SEED) for s in subjects]
    smooth = [values / values.max() for values in smooth]

    def correlation(a, b):
        return round(float(np.corrcoef(a[cortex], b[cortex])[0, 1]), 2) + 0.0

    r_raw = correlation(counts[0], counts[1])
    r_smooth = correlation(smooth[0], smooth[1])
    r_all = correlation(smooth[0], smooth[2])
    touching = [int(p.size) for p in partner]
    print(
        f"  vertex {SEED}: {touching} streamlines touch it; r between halves: raw {r_raw:.2f}, "
        f"smoothed {r_smooth:.2f}; half A against all: {r_all:.2f}",
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
                local = partner[column][partner[column] < half]  # left hemisphere
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
            axis.set_title(f"{labels[column]}\n{kind}", fontsize=13)
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
        "Smoothing: from a scatter of endpoints to a map you can compare\n"
        f"vertex {SEED} (orange dot) in two random halves of one {DATASET} subject's "
        "streamlines, left hemisphere",
        fontsize=TITLE_SIZE,
    )
    figure.supxlabel(
        f"The two halves agree across the cortex at r = {r_raw:.2f} as raw counts and "
        f"r = {r_smooth:.2f} once smoothed;\nsmoothed half A matches the map from all "
        f"streamlines at r = {r_all:.2f}.",
        fontsize=13,
    )
    save(figure, out, "smoothing_power.png")


#: The known warp of the recovery figures: a random tangent field of harmonic
#: order WARP_ORDER, scaled so that its largest displacement is WARP_AMPLITUDE
#: radians; and the order of the tangent basis ENCORE searches over.
WARP_ORDER, WARP_AMPLITUDE, ENCORE_ORDER = 4, 0.07, 6


def known_warp_experiment(subject) -> dict:
    """Deform a real subject's endpoints by a known smooth warp and register back with ENCORE.

    The undeformed copy is the subject's own endpoints through the package's
    smoother, so that the deformation is the only difference between the two.
    Shared by the recovery figure and the warp-migration figure.
    """
    from sbci.alignment import Encore, MeshQuery, _hemisphere_grids, normalize_rows
    from sbci.conseal import EndpointConnectome, StationaryWarp, default_grids

    lh, rh = default_grids()
    rng = np.random.default_rng(7)

    def known_warp(grid):
        warp = StationaryWarp(grid)
        coefficients = rng.standard_normal(grid.basis.shape[1])
        displacement = (coefficients[None, :, None] * grid.basis).sum(axis=1)
        displacement *= WARP_AMPLITUDE / np.linalg.norm(displacement, axis=1).max()
        assert warp.compose(displacement)
        return warp

    lh_true, rh_true = (known_warp(grid) for grid in default_grids(WARP_ORDER))
    reference = resmoothed(subject, subject.endpoints)
    carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
    original = carrier.positions()
    carrier.warp(lh_true, rh_true)
    deformed = resmoothed(subject, carrier.to_endpoints())
    moved = EndpointConnectome.from_endpoints(deformed.endpoints, lh, rh).positions()
    before = np.r_[angles(original[0], moved[0]), angles(original[1], moved[1])]

    t = time.time()
    grids, rotations = _hemisphere_grids(ENCORE_ORDER, return_rotations=True)
    engine = Encore(*grids, max_iterations=100)
    fixed = engine.root(reference.dense())
    trace = [float(((fixed - engine.root(deformed.dense())) ** 2 * engine.area_product).sum())]
    _, lh_warp, rh_warp, _ = engine.register(
        fixed, deformed.dense(), target_is_root=True, callback=lambda _i, c: trace.append(float(c))
    )
    warped = (lh_warp.vertices @ rotations[0], rh_warp.vertices @ rotations[1])
    fixed_points = []
    ends = (carrier.hemisphere_in, carrier.hemisphere_out)
    for points, hemispheres in zip(moved, ends, strict=True):
        placed = np.empty_like(points)
        for side, grid in enumerate((lh, rh)):
            pick = np.asarray(hemispheres) == side
            weights, indices = MeshQuery(normalize_rows(warped[side]), grid.faces).query(
                points[pick]
            )
            placed[pick] = normalize_rows(np.einsum("nk,nkj->nj", weights, grid.vertices[indices]))
        fixed_points.append(placed)
    after = np.r_[angles(original[0], fixed_points[0]), angles(original[1], fixed_points[1])]
    print(
        f"  ENCORE in {time.time() - t:.0f}s: {before.mean():.2f} -> {after.mean():.2f} "
        f"deg, cost {trace[0]:.4f} -> {trace[-1]:.4f} in {len(trace) - 1} steps",
        flush=True,
    )
    return {
        "grids": (lh, rh),
        "true": (lh_true, rh_true),
        "deformed": deformed,
        "original": original,
        "before": before,
        "after_encore": after,
        "encore": (lh_warp, rh_warp),
        "rotations": rotations,
        "trace": np.asarray(trace),
    }


def alignment_recovery(out: Path, subject, experiment: dict | None = None) -> None:
    """How far the endpoints still are from where they started, after each method."""
    from sbci.atlas import cortex_mask
    from sbci.smoothing import endpoint_positions

    experiment = known_warp_experiment(subject) if experiment is None else experiment
    original, before = experiment["original"], experiment["before"]
    after_encore, trace = experiment["after_encore"], experiment["trace"]

    t = time.time()
    conseal = sbci.endpoints_align(
        [subject, experiment["deformed"]],
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

    # Where the endpoints still are, vertex by vertex: the mean over the
    # endpoints that started at each vertex.
    endpoints = subject.endpoints
    vertex = np.r_[
        endpoints.vtx_in + endpoints.n_per_hemi * endpoints.surf_in.astype(np.int64),
        endpoints.vtx_out + endpoints.n_per_hemi * endpoints.surf_out.astype(np.int64),
    ]
    n_vertices = 2 * endpoints.n_per_hemi
    counts = np.bincount(vertex, minlength=n_vertices)

    def per_vertex(values):
        sums = np.bincount(vertex, weights=values, minlength=n_vertices)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    stages = (
        ("as deformed", before, MUTED),
        ("after ENCORE", after_encore, BLUE),
        ("after ConSEAL", after_conseal, ORANGE),
    )
    display = display_mesh(DISPLAY_MESH)
    vertices, faces = display.geometries["inflated"][0], display.faces[0]
    cortex = cortex_mask()
    on_cortex = np.nan_to_num(display.interpolate(cortex.astype(float))[0]) > 0.5
    grey = render.shade(display.sulc[0], on_cortex)
    vmax = float(np.ceil(np.nanpercentile(per_vertex(before), 98) * 2) / 2)

    figure = plt.figure(figsize=(15, 9.6), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[1.2, 1.0])
    axes = top.subplots(1, 3)
    for axis, (label, values, _color) in zip(axes, stages, strict=True):
        surface = display.interpolate(np.where(cortex, per_vertex(values), np.nan))[0]
        rgb = render.colour(surface, grey, "Reds", 0.0, vmax)
        image = render.render_view(vertices, faces, rgb, "lateral", "L")
        axis.imshow(render.trim(image), interpolation="lanczos")
        axis.set_axis_off()
        axis.set_title(f"{label}: mean {values.mean():.2f} deg", fontsize=13)
    bar = top.colorbar(
        ScalarMappable(norm=Normalize(0.0, vmax), cmap="Reds"),
        ax=axes.tolist(),
        shrink=0.7,
        pad=0.02,
    )
    bar.set_label("degrees from where the endpoints started", fontsize=11)
    bar.outline.set_visible(False)

    left, right = bottom.subplots(1, 2)
    bins = np.linspace(0, max(3.0, vmax), 61)
    for label, values, color in stages:
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
    for values, color, label in (
        (trace, BLUE, "ENCORE"),
        (np.asarray(conseal.costs[1]), ORANGE, "ConSEAL"),
    ):
        relative = values / values[0]
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
        f"Alignment with a known answer: an {DATASET} subject's endpoints moved by a smooth "
        "warp and registered back onto the undeformed subject",
        fontsize=14,
    )
    save(figure, out, "alignment_recovery.png")


def migration_power(out: Path, subject, experiment: dict, timeseries: Path) -> None:
    """ENCORE's warp, found on the grid, carried to fsaverage and applied to a 163,842-vertex map.

    The map is the subject's own resting-state connectivity of the seed vertex,
    from the pipeline's time series on fsaverage. The known warp moves it; the
    migrated ENCORE warp puts it back.
    """
    from types import SimpleNamespace

    from sbci.alignment import MeshQuery, Warp, normalize_rows
    from sbci.atlas import cortex_mask
    from sbci.plotting import sulcal_depth
    from sbci.templates import migrate_warp, template_mesh

    lh_true, rh_true = experiment["true"]
    lh_warp, rh_warp = experiment["encore"]
    rotations = experiment["rotations"]
    display = display_mesh("fsaverage")
    meshes = template_mesh("fsaverage")
    queries = [MeshQuery(vertices, faces) for vertices, faces in meshes]

    def sample(fields, points):
        """A per-vertex field on the fsaverage sphere, read at other points of it."""
        out = []
        for field, at, query in zip(fields, points, queries, strict=True):
            weights, indices = query.query(normalize_rows(at))
            out.append((weights * field[indices]).sum(axis=1))
        return out

    # The subject's resting-state map of the seed, at fsaverage resolution.
    t = time.time()
    seed = int(display.nearest[0][SEED])
    with np.load(timeseries) as loaded:
        series = [
            np.asarray(loaded[key], dtype=np.float32)
            for key in ("lh_time_series", "rh_time_series")
        ]
    centred = [s - s.mean(axis=1, keepdims=True) for s in series]
    scale = [np.sqrt((c * c).sum(axis=1)) for c in centred]
    reference = centred[0][seed] / scale[0][seed]
    fc_map = [
        np.where(sd > 0, (c @ reference) / np.where(sd > 0, sd, 1.0), np.nan)
        for c, sd in zip(centred, scale, strict=True)
    ]
    del series, centred
    print(
        f"  seed map from {series_shape(timeseries)} time points in {time.time() - t:.0f}s; "
        f"r at the seed {fc_map[0][seed]:.2f}",
        flush=True,
    )

    # The warps on fsaverage: the known warp, its inverse, and ENCORE's.
    t = time.time()
    to_fs = lambda lh, rh, **kw: migrate_warp(  # noqa: E731
        SimpleNamespace(lh_vertices=lh, rh_vertices=rh), to="fsaverage", **kw
    )
    true_fs = to_fs(lh_true.vertices, rh_true.vertices)
    inverse_fs = to_fs(lh_true.copy().invert().vertices, rh_true.copy().invert().vertices)
    encore_fs = migrate_warp(
        Warp(lh_warp.vertices, lh_warp.jacobian, rh_warp.vertices, rh_warp.jacobian),
        to="fsaverage",
        grid_rotations=rotations,
    )
    moved_map = sample(fc_map, (inverse_fs.lh_vertices, inverse_fs.rh_vertices))
    recovered_map = sample(moved_map, (encore_fs.lh_vertices, encore_fs.rh_vertices))
    sphere = [vertices for vertices, _ in meshes]
    deformation = [angles(sphere[0], true_fs.lh_vertices), angles(sphere[1], true_fs.rh_vertices)]
    found = [angles(sphere[0], encore_fs.lh_vertices), angles(sphere[1], encore_fs.rh_vertices)]
    residual = [
        angles(true_fs.lh_vertices, encore_fs.lh_vertices),
        angles(true_fs.rh_vertices, encore_fs.rh_vertices),
    ]
    lh_grid, _ = experiment["grids"]
    on_grid = angles(lh_grid.vertices, normalize_rows(lh_warp.vertices @ rotations[0]))

    def correlation(a, b):
        keep = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[keep], b[keep])[0, 1])

    both = lambda maps: np.concatenate(maps)  # noqa: E731
    r_moved = correlation(both(moved_map), both(fc_map))
    r_back = correlation(both(recovered_map), both(fc_map))
    print(
        f"  migrated in {time.time() - t:.0f}s: deformation "
        f"{np.concatenate(deformation).mean():.2f} deg on fsaverage; ENCORE's warp "
        f"{np.concatenate(residual).mean():.2f} deg from the true one there; "
        f"map r {r_moved:.3f} moved, {r_back:.3f} put back",
        flush=True,
    )

    # Draw: the map's three states above, the warps below, left hemisphere.
    cortex = cortex_mask()
    on_cortex = np.nan_to_num(display.interpolate(cortex.astype(float))[0]) > 0.5
    vertices, faces = display.geometries["inflated"][0], display.faces[0]
    grey = render.shade(display.sulc[0], on_cortex)
    grid_surface = sbci.load_surface("inflated").hemisphere("L")
    grid_vertices = np.asarray(grid_surface.vertices, dtype=np.float64)
    half = cortex.size // 2
    grid_grey = render.shade(np.where(cortex[:half], sulcal_depth()[:half], 0.0), cortex[:half])
    vmax_map = 0.6
    vmax_warp = float(np.ceil(np.nanpercentile(deformation[0], 99) * 2) / 2)

    figure = plt.figure(figsize=(15, 9.8), layout="constrained")
    top, bottom = figure.subfigures(2, 1)
    axes = top.subplots(1, 3)
    panels = (
        (
            fc_map[0],
            "the subject's resting-state map of vertex 1234\non fsaverage, 163,842 vertices",
        ),
        (moved_map[0], f"moved by the known warp\n(r = {r_moved:.2f} with the original)"),
        (
            recovered_map[0],
            "put back by ENCORE's warp carried from the grid\n"
            f"(r = {r_back:.2f} with the original)",
        ),
    )
    for axis, (values, label) in zip(axes, panels, strict=True):
        shown = np.where(on_cortex, values, np.nan)
        rgb = render.colour(shown, grey, "coolwarm", -vmax_map, vmax_map, 0.1, True)
        image = render.render_view(vertices, faces, rgb, "lateral", "L")
        axis.imshow(render.trim(image), interpolation="lanczos")
        axis.set_axis_off()
        axis.set_title(label, fontsize=12)
    bar = top.colorbar(
        ScalarMappable(norm=Normalize(-vmax_map, vmax_map), cmap="coolwarm"),
        ax=axes.tolist(),
        shrink=0.7,
        pad=0.02,
    )
    bar.set_label("correlation with the seed's time series", fontsize=11)
    bar.outline.set_visible(False)

    axes = bottom.subplots(1, 3)
    warps = (
        (
            grid_vertices,
            grid_surface.faces,
            grid_grey,
            on_grid,
            f"ENCORE's warp on the grid, 5,124 vertices\n(mean {on_grid.mean():.2f} deg)",
        ),
        (
            vertices,
            faces,
            grey,
            found[0],
            "the same warp carried to fsaverage, 163,842 vertices\n"
            f"(mean {found[0].mean():.2f} deg)",
        ),
        (
            vertices,
            faces,
            grey,
            deformation[0],
            f"the known warp on fsaverage (mean {deformation[0].mean():.2f} deg);\n"
            f"ENCORE's is {residual[0].mean():.2f} deg from it",
        ),
    )
    for axis, (mesh_vertices, mesh_faces, shading, values, label) in zip(axes, warps, strict=True):
        rgb = render.colour(values, shading, "Reds", 0.0, vmax_warp)
        image = render.render_view(mesh_vertices, mesh_faces, rgb, "lateral", "L")
        axis.imshow(render.trim(image), interpolation="lanczos")
        axis.set_axis_off()
        axis.set_title(label, fontsize=12)
    bar = bottom.colorbar(
        ScalarMappable(norm=Normalize(0.0, vmax_warp), cmap="Reds"),
        ax=axes.tolist(),
        shrink=0.7,
        pad=0.02,
    )
    bar.set_label("displacement (degrees)", fontsize=11)
    bar.outline.set_visible(False)
    figure.suptitle(
        "Carrying a warp between templates: a warp found from connectivity on 5,124 vertices, "
        "restated on fsaverage, puts a 163,842-vertex map back",
        fontsize=14,
    )
    save(figure, out, "migration_power.png")


def series_shape(timeseries: Path) -> int:
    """How many time points the pipeline's fsaverage series hold."""
    with np.load(timeseries) as loaded:
        return int(loaded["lh_time_series"].shape[1])


# --- the cohort ----------------------------------------------------------------------


def cohort_figures(out: Path, subjects, conseal: bool = True) -> None:
    """The alignment of the example subjects.

    Ten or eleven subjects are too few to ask a question of, so the example cohort
    illustrates alignment only; the analysis is drawn from a full cohort
    (:func:`cohort_trait`, :func:`cohort_age`).
    """

    # Alignment: ENCORE on the densities, ConSEAL on the endpoints, and how alike
    # the subjects are before and after.
    def similarity(matrices):
        data = np.stack([np.asarray(m, dtype=np.float32).ravel() for m in matrices])
        r = np.corrcoef(data)
        return r[np.triu_indices(len(matrices), 1)]

    # One smoother for everything compared: the subjects' endpoints through the
    # package's kernel. The pipeline's stored densities differ from that by more
    # than a small warp changes them, which would make any comparison across the
    # two unfair (see alignment_recovery).
    t = time.time()
    smoothed = [resmoothed(subject, subject.endpoints) for subject in subjects]
    print(f"  re-smoothed {len(subjects)} subjects in {time.time() - t:.0f}s", flush=True)
    before = similarity([s.data for s in smoothed])
    t = time.time()
    encore = sbci.align(smoothed, max_iterations=10, backtracks=4)
    from sbci.grid import to_condensed

    aligned_encore = [to_condensed(np.asarray(d)) for d in encore.aligned]
    after_encore = similarity(aligned_encore)
    print(
        f"  ENCORE on {len(subjects)} subjects in {time.time() - t:.0f}s: "
        f"mean pairwise correlation {before.mean():.4f} -> {after_encore.mean():.4f}",
        flush=True,
    )
    after_conseal = None
    conseal_costs = None
    if conseal:
        # ConSEAL's Karcher median settles on one subject when the subjects sit
        # evenly around the mean, as it does on the ten young adults drawn at random (0.001
        # degrees from sub-212116; USAGE, ConSEAL caveats); register onto the mean
        # of the square-root densities instead so that every subject moves, as
        # with ENCORE's template.
        from sbci.conseal import (
            DEFAULT_KERNEL_DEGREE,
            DEFAULT_SIGMA,
            EndpointConnectome,
            HeatKernelBuilder,
            default_grids,
        )

        lh, rh = default_grids()
        kernel = HeatKernelBuilder(lh, rh, DEFAULT_KERNEL_DEGREE).compute(
            DEFAULT_SIGMA, derivative=True
        )[0]
        carriers = [EndpointConnectome.from_endpoints(s.endpoints, lh, rh) for s in subjects]
        mean_template = sum(c.q_transform(kernel) for c in carriers) / len(carriers)
        mean_template /= np.sqrt((mean_template**2).sum())
        t = time.time()
        registration = sbci.endpoints_align(
            carriers,
            template=mean_template,
            max_iterations=30,
            threshold=1e-7,
            delta=0.1,
            step_clamp=float("inf"),
            viscosity=0.0,
        )
        aligned = [
            resmoothed(subject, registration.aligned_endpoints(i)).data
            for i, subject in enumerate(subjects)
        ]
        after_conseal = similarity(aligned)
        conseal_costs = registration.costs
        print(
            f"  ConSEAL on {len(subjects)} subjects in {time.time() - t:.0f}s: "
            "mean pairwise correlation "
            f"{before.mean():.4f} -> {after_conseal.mean():.4f}",
            flush=True,
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for trace in encore.traces:
        relative = np.asarray(trace) / trace[0]
        left.plot(range(len(relative)), relative, color=BLUE, linewidth=1.2, alpha=0.8)
    if conseal_costs is not None:
        for trace in conseal_costs:
            trace = np.asarray(trace)
            if trace.size > 1 and trace[0] > 0:
                left.plot(
                    range(trace.size), trace / trace[0], color=ORANGE, linewidth=1.2, alpha=0.8
                )
    left.plot([], [], color=BLUE, linewidth=2, label="ENCORE, one line per subject")
    if conseal_costs is not None:
        left.plot([], [], color=ORANGE, linewidth=2, label="ConSEAL, one line per subject")
    left.set_xlabel("iteration")
    left.set_ylabel("cost, relative to the start")
    left.set_ylim(0, 1.05)
    left.set_title(f"Registering {len(subjects)} subjects onto a template", loc="left")
    left.legend(frameon=False)
    groups = [("before", before, MUTED), ("after ENCORE", after_encore, BLUE)]
    if after_conseal is not None:
        groups.append(("after ConSEAL", after_conseal, ORANGE))
    for position, (_label, values, color) in enumerate(groups):
        jitter = np.random.default_rng(1).uniform(-0.15, 0.15, size=values.size)
        right.scatter(position + jitter, values, s=18, color=color, alpha=0.7)
        right.plot(
            [position - 0.25, position + 0.25], [values.mean()] * 2, color=color, linewidth=3
        )
    right.set_xticks(range(len(groups)))
    right.set_xticklabels([label for label, _, _ in groups])
    right.set_ylabel("correlation between two subjects' connectomes")
    n = len(subjects)
    right.set_title(f"The {n * (n - 1) // 2} pairs of subjects, and the mean", loc="left")
    figure.suptitle(
        f"Aligning {n} {DATASET} subjects: the cost falls for every subject, and the "
        "subjects grow more alike",
        fontsize=14,
        y=1.03,
    )
    save(figure, out, "cohort_alignment.png")


class LazyCohort:
    """SC files loaded when indexed and not kept, so reduce() reads one subject at a time.

    Also records each subject's streamline count as it is read.
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self.count = np.zeros(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if not -len(self.paths) <= index < len(self.paths):
            raise IndexError(index)
        cc = sbci.load(self.paths[index])
        self.count[index] = float(cc.metadata.get("streamline_count") or 0)
        return cc


def cohort_age(out: Path, directory: Path, rank: int = 20, candidates: int = 1) -> None:
    """FPCA of the full cohort against age: the components' scores, and the effect on the surface.

    ``directory`` holds the full cohort's SC files and ``manifest.csv`` with
    ``subject, sex, age_bin``; age enters as the bin's midpoint and sex as a
    covariate. The dense matrices of 528 subjects are 111 GB and the fit works
    on them, so this is a large-memory batch job; the fitted basis and scores
    are saved beside the data (``fpca_rank<rank>_c<candidates>.npz``) and
    reused on the next run, so the figures can be redrawn without refitting.
    """
    from sbci.reduction import Reduction

    rows = list(csv.DictReader(open(directory / "manifest.csv")))
    age = np.array([np.mean([int(v) for v in row["age_bin"].split("-")]) for row in rows])
    female = np.array([row["sex"] == "F" for row in rows], dtype=float)
    saved = directory.parent / f"fpca_rank{rank}_c{candidates}.npz"
    first = sbci.load(directory / f"{rows[0]['subject']}_sc.h5")
    if saved.exists():
        with np.load(saved) as fit:
            reduction = Reduction(
                basis=fit["basis"],
                scores=fit["scores"],
                scales=fit["scales"],
                explained=fit["explained"],
                objective=np.zeros((rank, 1)),
            )
            count = fit["count"]
        print(f"  reusing {saved.name}: explained {reduction.explained[-1]:.3f}", flush=True)
    else:
        t = time.time()
        cohort = LazyCohort(directory / f"{row['subject']}_sc.h5" for row in rows)
        reduction = sbci.reduce(cohort, rank=rank, candidates=candidates)
        count = cohort.count
        print(
            f"  FPCA rank {rank} (candidates={candidates}) of {len(cohort)} subjects in "
            f"{time.time() - t:.0f}s, loading included: explained {reduction.explained[-1]:.3f}",
            flush=True,
        )
        np.savez(
            saved,
            basis=reduction.basis,
            scores=reduction.scores,
            scales=reduction.scales,
            explained=reduction.explained,
            count=count,
            age=age,
            female=female,
        )
    print(
        f"  {len(rows)} subjects: ages {age.min():.0f}-{age.max():.0f} (bin midpoints), "
        f"{int(female.sum())} F",
        flush=True,
    )

    def r_with(values):
        return [float(np.corrcoef(reduction.scores[:, k], values)[0, 1]) for k in range(rank)]

    # Age is tested with sex and the streamline count as nuisance covariates: the count
    # (tractography yield, 0.5 to 1.8 million here) is the largest source of variation
    # between subjects' connectomes and correlates only weakly with age.
    millions = count / 1e6
    alone = sbci.local_test(reduction.scores, age)
    with_sex = sbci.local_test(reduction.scores, np.column_stack([age, female]), terms=[1])
    result = sbci.local_test(reduction.scores, np.column_stack([age, female, millions]), terms=[1])
    found = result.significant()
    correlations = r_with(age)
    for label, test in (
        ("age alone", alone),
        ("age given sex", with_sex),
        ("age given sex and streamline count", result),
    ):
        print(
            f"  {label}: adjusted p {np.round(test.adjusted, 4).tolist()}, "
            f"significant {test.significant().tolist()}",
            flush=True,
        )
    for name, values in (("age", age), ("sex", female), ("streamline count", count)):
        print(f"  r(score, {name}): {np.round(r_with(values), 3).tolist()}", flush=True)

    # Scores against age for the components most associated with it.
    shown = np.argsort(result.adjusted)[: min(4, rank)]
    figure, axes = plt.subplots(
        1, len(shown), figsize=(4.2 * len(shown), 3.9), layout="constrained", squeeze=False
    )
    axes = axes[0]
    for axis, k in zip(axes, shown, strict=True):
        scores = reduction.scores[:, k]
        jitter = np.random.default_rng(int(k)).uniform(-1.5, 1.5, size=age.size)
        for mask, color, label in ((female == 1, ORANGE, "F"), (female == 0, BLUE, "M")):
            axis.scatter(
                age[mask] + jitter[mask],
                scores[mask],
                s=11,
                color=color,
                alpha=0.5,
                linewidths=0,
                label=label,
            )
        slope, intercept = np.polyfit(age, scores, 1)
        grid = np.linspace(age.min(), age.max(), 2)
        axis.plot(grid, slope * grid + intercept, color=INK, linewidth=2)
        axis.set_title(
            f"component {k + 1}: r = {correlations[k]:.2f}, adjusted p = {result.adjusted[k]:.2g}",
            fontsize=11,
        )
        axis.set_xlabel("age (bin midpoint, years)")
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False, markerscale=2)
    figure.suptitle(
        f"FPCA of {len(rows)} HCP-Aging subjects, rank {rank}: the {len(shown)} components most "
        "associated with age (sex and streamline count as covariates)",
        fontsize=14,
    )
    save(figure, out, "cohort_age_scores.png")

    if not found.size:
        print("  no component reaches significance; no effect map drawn", flush=True)
        return
    effect = result.effect_map(reduction, alpha=0.05)
    effect = effect / np.abs(effect).max()
    figure = first.plot(
        effect,
        views=VIEWS,
        cmap="coolwarm",
        symmetric=True,
        threshold=0.05,
        mesh=DISPLAY_MESH,
        engine=DISPLAY_ENGINE,
    )
    figure.suptitle(
        f"Where age shows in the structural connectome of {len(rows)} HCP-Aging subjects: the "
        f"effect over the {found.size} significant component(s) of {rank},\nrelative to its "
        "largest value (positive: connectivity rising with age)",
        fontsize=TITLE_SIZE,
        y=1.11,
    )
    save(figure, out, "cohort_age_effect.png")


#: Midpoints of the HCP Young Adult open-access age bands, for use as a covariate.
BAND_MIDPOINTS = {"22-25": 23.5, "26-30": 28.0, "31-35": 33.0, "36+": 37.0}


def open_access_table(path: Path, trait: str) -> dict:
    """Subject id to sex, age band and ``trait``, from an open-access table.

    Either the HCP's own table (ConnectomeDB's open-access csv, with
    ``Subject``, ``Gender``, ``Age`` and ``PMAT24_A_CR``), for which ``trait``
    is a column name such as ``PMAT24_A_CR``, or a table with ``subject``,
    ``sex``, ``age_bin`` and the ``trait`` column.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "Subject" in rows[0]:
        column = "PMAT24_A_CR" if trait == "fluid_intelligence_pmat24" else trait
        return {
            f"sub-{r['Subject']}": {"sex": r["Gender"], "age_bin": r["Age"], "trait": r[column]}
            for r in rows
        }
    return {
        r["subject"]: {"sex": r["sex"], "age_bin": r["age_bin"], "trait": r[trait]} for r in rows
    }


def cohort_trait(
    out: Path,
    directory: Path,
    traits: Path,
    trait: str = "fluid_intelligence_pmat24",
    label: str = "fluid intelligence (PMAT24 correct responses)",
    rank: int = 20,
    candidates: int = 1,
) -> None:
    """FPCA of a full young adult cohort against a trait, given sex, age band and streamline count.

    ``directory`` holds the SC files, ``<subject>_sc.h5``; ``traits`` is an
    open-access table (:func:`open_access_table`) giving each subject's sex,
    age band and ``trait``. Subjects without a value are left out. The fit is
    saved beside the data as ``fpca_rank<rank>_c<candidates>.npz`` and reused,
    as in :func:`cohort_age`.
    """
    from sbci.reduction import Reduction

    table = open_access_table(traits, trait)
    built = sorted(p.name.split("_")[0] for p in directory.glob("sub-*_sc.h5"))
    unknown = [s for s in built if s not in table]
    if unknown:
        raise ValueError(f"{len(unknown)} subjects are not in {traits.name}: {unknown[:3]}")
    rows = [dict(subject=s, **table[s]) for s in built]
    keep = [r for r in rows if r["trait"] not in ("", "NA", "nan")]
    if len(keep) != len(rows):
        print(f"  {len(rows) - len(keep)} subjects have no {trait}; left out", flush=True)
    rows = keep
    score = np.array([float(r["trait"]) for r in rows])
    female = np.array([r["sex"] == "F" for r in rows], dtype=float)
    band = np.array([BAND_MIDPOINTS[r["age_bin"]] for r in rows])
    saved = directory.parent / f"fpca_rank{rank}_c{candidates}.npz"
    first = sbci.load(directory / f"{rows[0]['subject']}_sc.h5")
    if saved.exists():
        with np.load(saved) as fit:
            if list(fit["subjects"]) != [r["subject"] for r in rows]:
                raise ValueError(f"{saved.name} was fitted to other subjects; remove it to refit")
            reduction = Reduction(
                basis=fit["basis"],
                scores=fit["scores"],
                scales=fit["scales"],
                explained=fit["explained"],
                objective=np.zeros((rank, 1)),
            )
            count = fit["count"]
        print(f"  reusing {saved.name}: explained {reduction.explained[-1]:.3f}", flush=True)
    else:
        t = time.time()
        cohort = LazyCohort(directory / f"{r['subject']}_sc.h5" for r in rows)
        reduction = sbci.reduce(cohort, rank=rank, candidates=candidates)
        count = cohort.count
        print(
            f"  FPCA rank {rank} (candidates={candidates}) of {len(cohort)} subjects in "
            f"{time.time() - t:.0f}s, loading included: explained {reduction.explained[-1]:.3f}",
            flush=True,
        )
        np.savez(
            saved,
            basis=reduction.basis,
            scores=reduction.scores,
            scales=reduction.scales,
            explained=reduction.explained,
            count=count,
            trait=score,
            female=female,
            band=band,
            subjects=np.array([r["subject"] for r in rows]),
        )
    print(
        f"  {len(rows)} subjects, {int(female.sum())} F; "
        f"{trait} {score.min():.0f}-{score.max():.0f}",
        flush=True,
    )
    millions = count / 1e6
    design = np.column_stack([score, female, band, millions])
    result = sbci.local_test(reduction.scores, design, terms=[1])
    alone = sbci.local_test(reduction.scores, score)
    found = result.significant()
    correlations = [float(np.corrcoef(reduction.scores[:, k], score)[0, 1]) for k in range(rank)]
    for name, test in (
        (f"{trait} alone", alone),
        (f"{trait} given sex, age band and streamline count", result),
    ):
        print(
            f"  {name}: adjusted p {np.round(test.adjusted, 4).tolist()}, "
            f"significant {test.significant().tolist()}",
            flush=True,
        )
    for name, v in (
        ("the trait", score),
        ("sex", female),
        ("age band", band),
        ("streamline count", count),
    ):
        r = [float(np.corrcoef(reduction.scores[:, k], v)[0, 1]) for k in range(rank)]
        print(f"  r(score, {name}): {np.round(r, 3).tolist()}", flush=True)

    shown = np.argsort(result.adjusted)[: min(4, rank)]
    figure, axes = plt.subplots(
        1, len(shown), figsize=(4.2 * len(shown), 3.9), layout="constrained", squeeze=False
    )
    axes = axes[0]
    for axis, k in zip(axes, shown, strict=True):
        scores = reduction.scores[:, k]
        for mask, color, sex in ((female == 1, ORANGE, "F"), (female == 0, BLUE, "M")):
            axis.scatter(
                score[mask], scores[mask], s=9, color=color, alpha=0.45, linewidths=0, label=sex
            )
        slope, intercept = np.polyfit(score, scores, 1)
        grid = np.linspace(score.min(), score.max(), 2)
        axis.plot(grid, slope * grid + intercept, color=INK, linewidth=2)
        axis.set_title(
            f"component {k + 1}: r = {correlations[k]:.2f}, adjusted p = {result.adjusted[k]:.2g}",
            fontsize=11,
        )
        axis.set_xlabel(label)
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False, markerscale=2)
    figure.suptitle(
        f"FPCA of {len(rows)} HCP Young Adult subjects, rank {rank}: the {len(shown)} components "
        "most associated with fluid intelligence\n"
        "(sex, age band and streamline count as covariates)",
        fontsize=14,
    )
    save(figure, out, "cohort_trait_scores.png")
    if not found.size:
        print("  no component reaches significance; no effect map drawn", flush=True)
        return
    effect = result.effect_map(reduction, alpha=0.05)
    effect = effect / np.abs(effect).max()
    figure = first.plot(
        effect,
        views=VIEWS,
        cmap="coolwarm",
        symmetric=True,
        threshold=0.05,
        mesh=DISPLAY_MESH,
        engine=DISPLAY_ENGINE,
    )
    figure.suptitle(
        f"Where fluid intelligence shows in the structural connectome of {len(rows)} HCP Young "
        f"Adult subjects: the effect over the {found.size} significant component(s) of {rank},\n"
        "relative to its largest value (positive: connectivity higher with higher scores)",
        fontsize=TITLE_SIZE,
        y=1.11,
    )
    save(figure, out, "cohort_trait_effect.png")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    cohort_dir, out = Path(argv[1]), Path(argv[2])
    skip_cohort = "--skip-cohort" in argv
    only_cohort = "--only-cohort" in argv
    no_conseal = "--no-conseal" in argv
    surfaces_only = "--surfaces-only" in argv  # the rendered surface figures, no registrations
    only_recovery = "--only-recovery" in argv  # the known-warp figures alone
    only_migration = "--only-migration" in argv  # the warp-migration figure alone
    # --full-cohort DIR: the full cohort's SC files and manifest; draws the FPCA-against-age
    # figures from them and nothing else.
    if "--full-cohort" in argv:
        directory = Path(argv[argv.index("--full-cohort") + 1])
        rank = int(argv[argv.index("--rank") + 1]) if "--rank" in argv else 20
        candidates = int(argv[argv.index("--candidates") + 1]) if "--candidates" in argv else 1
        out.mkdir(parents=True, exist_ok=True)
        print("full cohort", flush=True)
        if "--traits" in argv:  # a young adult cohort against an open-access trait
            traits = Path(argv[argv.index("--traits") + 1])
            cohort_trait(out, directory, traits, rank=rank, candidates=candidates)
        else:  # the HCP-Aging cohort against age
            cohort_age(out, directory, rank=rank, candidates=candidates)
        return 0
    # --timeseries PATH: the pipeline's fc_ts.npz of the first subject (its resting-state
    # time series on fsaverage), for the warp-migration figure; skipped without it.
    timeseries = None
    if "--timeseries" in argv:
        timeseries = Path(argv[argv.index("--timeseries") + 1])
        argv = [a for a in argv if a not in ("--timeseries", str(timeseries))]
    global DATASET
    DATASET = dataset_of(cohort_dir)
    if "--dataset" in argv:
        DATASET = argv[argv.index("--dataset") + 1]
        argv = [a for a in argv if a not in ("--dataset", DATASET)]
    out.mkdir(parents=True, exist_ok=True)
    subjects, functional, ages = load_cohort(cohort_dir)
    span = f", ages {np.nanmin(ages):.0f}-{np.nanmax(ages):.0f}" if np.isfinite(ages).any() else ""
    print(f"loaded {len(subjects)} {DATASET} subjects{span}", flush=True)
    first = subjects[0]
    fc_first = sbci.load(functional[0]) if functional[0] is not None else None
    if surfaces_only:
        print("surface figures", flush=True)
        seed_profile(out, first)
        if fc_first is not None:
            coupling(out, first, fc_first)
        smoothing_power(out, first)
        return 0
    if only_recovery or only_migration:
        experiment = known_warp_experiment(first)
        if only_recovery:
            print("alignment recovery", flush=True)
            alignment_recovery(out, first, experiment)
        if timeseries is not None:
            print("warp migration", flush=True)
            migration_power(out, first, experiment, timeseries)
        return 0
    if not only_cohort:
        print("single subject", flush=True)
        seed_profile(out, first)
        region_matrix(out, first)
        if fc_first is not None:
            coupling(out, first, fc_first)
        print("smoothing", flush=True)
        smoothing_power(out, first)
        print("alignment recovery", flush=True)
        experiment = known_warp_experiment(first)
        alignment_recovery(out, first, experiment)
        if timeseries is not None:
            print("warp migration", flush=True)
            migration_power(out, first, experiment, timeseries)
    if not skip_cohort:
        print("cohort", flush=True)
        cohort_figures(out, subjects, conseal=not no_conseal)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
