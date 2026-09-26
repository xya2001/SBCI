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


def save(figure, out: Path, name: str, dpi: int = 125) -> None:
    path = out / name
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path} ({path.stat().st_size // 1024} KB)", flush=True)


SEED = 1234  # a left temporal vertex


def seed_profile(out: Path, cc) -> None:
    profile = cc.seed(vertex=SEED)
    profile = profile / profile.max()  # unit-mass densities are 1e-10 per pair; show the shape
    figure = cc.plot(profile, views=VIEWS, cmap=SURFACE_RAMP, threshold=0.02, vmin=0.02, vmax=1.0)
    # Mark the seed on the lateral view of its hemisphere. nilearn recentres
    # each hemisphere on its own mean before drawing, so do the same.
    left = np.asarray(sbci.load_surface("inflated").vertices[: cc.n_vertices // 2], dtype=float)
    x, y, z = left[SEED] - left.mean(axis=0)
    lateral = figure.axes[0]
    # A 3-D axes sorts artists by depth and would bury the dot under the mesh;
    # switch to drawing order so the marker sits on top.
    lateral.computed_zorder = False
    lateral.scatter(
        [x],
        [y],
        [z],
        s=90,
        color=ORANGE,
        edgecolor="white",
        linewidth=1.5,
        zorder=10,
        depthshade=False,
    )
    figure.suptitle(
        f"Where one vertex connects to: the density of streamlines between vertex {SEED} "
        "(orange dot, left temporal cortex)\nand every other vertex, relative to the strongest",
        fontsize=TITLE_SIZE,
        y=1.11,  # two lines at this size need clearance above the panel labels
    )
    save(figure, out, "seed_profile.png")


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
    figure = sc.plot(values, views=VIEWS, cmap=SURFACE_RAMP, symmetric=False, vmin=0.0)
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
        cohort.truth, views=VIEWS, cmap=SURFACE_RAMP, threshold=0.02, vmin=0.02, vmax=1.0
    )
    figure.suptitle(
        "The planted bundle: the field whose weight scales with age (cohort.truth)",
        fontsize=TITLE_SIZE,
        y=1.03,
    )
    save(figure, out, "cohort_truth.png")

    effect = result.effect_map(reduction, alpha=0.05)
    effect = effect / np.abs(effect).max()
    figure = subject.plot(effect, views=VIEWS, cmap="coolwarm", symmetric=True, threshold=0.05)
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

    t = time.time()
    aligned = sbci.endpoints_align(cohort.connectomes[:2], template=0, max_iterations=5)
    print(f"  ConSEAL in {time.time() - t:.0f}s", flush=True)
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    costs = np.asarray(aligned.costs[1], dtype=float)
    relative = costs / costs[0]
    axis.plot(range(len(relative)), relative, color=BLUE, linewidth=2, marker="o", markersize=6)
    axis.annotate(
        f"{relative[-1]:.3f}",
        (len(relative) - 1, relative[-1]),
        textcoords="offset points",
        xytext=(8, 0),
        va="center",
        color=INK,
    )
    axis.set_xlabel("iteration")
    axis.set_ylabel("cost, relative to the start")
    axis.set_xticks(range(len(relative)))
    axis.set_title("ConSEAL registering synthetic subject 2 onto subject 1", loc="left")
    save(figure, out, "conseal_cost.png")


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
    region_matrix(out, sc)
    coupling(out, sc, fc)
    print("kernel", flush=True)
    spherical_kernel(out)
    print("cohort", flush=True)
    cohort_figures(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
