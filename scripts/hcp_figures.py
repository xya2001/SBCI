"""Draw the documentation figures from the ten-subject HCP-Aging example cohort.

    python scripts/hcp_figures.py /path/to/cohort docs/figures \
        [--skip-cohort | --only-cohort | --surfaces-only | --only-recovery]

The cohort directory holds ``manifest.csv`` (subject, age_years, sex) and one
``<subject>_sc.h5`` (with endpoints) and ``<subject>_fc.h5`` per subject, as
written by ``tools/build_hcp_cohort.py``. Every surface is drawn on fsaverage.

The single-subject figures take a few minutes. The cohort figures register
ten subjects with ENCORE and ConSEAL and take a couple of hours on four
cores; run this in a batch job.
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


def load_cohort(directory: Path):
    """The subjects in manifest order, their FC paths and their ages."""
    rows = list(csv.DictReader(open(directory / "manifest.csv")))
    subjects = [sbci.load(directory / f"{row['subject']}_sc.h5") for row in rows]
    functional = [directory / f"{row['subject']}_fc.h5" for row in rows]
    ages = np.array([float(row["age_years"]) for row in rows])
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
        f"Where one vertex connects to: an HCP-Aging subject's density of streamlines between "
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
    axis.set_title("An HCP-Aging subject parcellated with the Desikan atlas (68 regions)")
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
        "Structure-function coupling in an HCP-Aging subject: at each vertex, the cosine "
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
        f"vertex {SEED} (orange dot) in two random halves of one HCP-Aging subject's "
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


#: The known warp of the recovery figure: a random tangent field of harmonic
#: order WARP_ORDER, scaled so that its largest displacement is WARP_AMPLITUDE
#: radians; and the order of the tangent basis ENCORE searches over.
WARP_ORDER, WARP_AMPLITUDE, ENCORE_ORDER = 4, 0.07, 6


def alignment_recovery(out: Path, subject) -> None:
    """Deform a real subject by a known smooth warp and measure how much each method undoes.

    The undeformed copy is the subject's own endpoints through the package's
    smoother, so that the deformation is the only difference between the two.
    """
    from sbci.alignment import Encore, MeshQuery, _hemisphere_grids, normalize_rows
    from sbci.atlas import cortex_mask
    from sbci.conseal import EndpointConnectome, StationaryWarp, default_grids
    from sbci.smoothing import endpoint_positions

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
    after_encore = np.r_[angles(original[0], fixed_points[0]), angles(original[1], fixed_points[1])]
    print(
        f"  ENCORE in {time.time() - t:.0f}s: {before.mean():.2f} -> {after_encore.mean():.2f} "
        f"deg, cost {trace[0]:.4f} -> {trace[-1]:.4f} in {len(trace) - 1} steps",
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
        (np.asarray(trace), BLUE, "ENCORE"),
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
        "Alignment with a known answer: an HCP-Aging subject's endpoints moved by a smooth "
        "warp and registered back onto the undeformed subject",
        fontsize=14,
    )
    save(figure, out, "alignment_recovery.png")


# --- the cohort ----------------------------------------------------------------------


def cohort_figures(out: Path, subjects, conseal: bool = True) -> None:
    """The alignment of the ten subjects.

    A rank-4 FPCA of ten subjects against their ages finds nothing, as it
    should with ten subjects, so the real cohort illustrates alignment only;
    the synthetic cohort, with its planted effect, illustrates the analysis.
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
    print(f"  re-smoothed ten subjects in {time.time() - t:.0f}s", flush=True)
    before = similarity([s.data for s in smoothed])
    t = time.time()
    encore = sbci.align(smoothed, max_iterations=10, backtracks=4)
    from sbci.grid import to_condensed

    aligned_encore = [to_condensed(np.asarray(d)) for d in encore.aligned]
    after_encore = similarity(aligned_encore)
    print(
        f"  ENCORE on ten subjects in {time.time() - t:.0f}s: mean pairwise correlation "
        f"{before.mean():.4f} -> {after_encore.mean():.4f}",
        flush=True,
    )
    after_conseal = None
    conseal_costs = None
    if conseal:
        # ConSEAL's Karcher median settles on one subject when the subjects sit
        # evenly around the mean, which on this cohort it does (USAGE, ConSEAL
        # caveats); register onto the mean of the square-root densities instead
        # so that every subject moves, as with ENCORE's template.
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
            f"  ConSEAL on ten subjects in {time.time() - t:.0f}s: mean pairwise correlation "
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
    left.set_title("Registering ten subjects onto a template", loc="left")
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
    right.set_title("The 45 pairs of subjects, and the mean", loc="left")
    figure.suptitle(
        "Aligning the ten HCP-Aging subjects: the cost falls for every subject, and the "
        "subjects grow more alike",
        fontsize=14,
        y=1.03,
    )
    save(figure, out, "cohort_alignment.png")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    cohort_dir, out = Path(argv[1]), Path(argv[2])
    skip_cohort = "--skip-cohort" in argv
    only_cohort = "--only-cohort" in argv
    no_conseal = "--no-conseal" in argv
    surfaces_only = "--surfaces-only" in argv  # the rendered surface figures, no registrations
    only_recovery = "--only-recovery" in argv  # the known-warp figure alone
    out.mkdir(parents=True, exist_ok=True)
    subjects, functional, ages = load_cohort(cohort_dir)
    print(f"loaded {len(subjects)} subjects, ages {ages.min():.0f}-{ages.max():.0f}", flush=True)
    first = subjects[0]
    if surfaces_only:
        print("surface figures", flush=True)
        seed_profile(out, first)
        coupling(out, first, sbci.load(functional[0]))
        smoothing_power(out, first)
        return 0
    if only_recovery:
        print("alignment recovery", flush=True)
        alignment_recovery(out, first)
        return 0
    if not only_cohort:
        print("single subject", flush=True)
        seed_profile(out, first)
        region_matrix(out, first)
        coupling(out, first, sbci.load(functional[0]))
        print("smoothing", flush=True)
        smoothing_power(out, first)
        print("alignment recovery", flush=True)
        alignment_recovery(out, first)
    if not skip_cohort:
        print("cohort", flush=True)
        cohort_figures(out, subjects, conseal=not no_conseal)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
