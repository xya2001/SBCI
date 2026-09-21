"""Draw the exchange file itself: everything below is read from the .dconn.

Layout note: nilearn draws its colourbar inside the axes it is given, which
puts it on top of the cortex. Every panel is therefore drawn with
``colorbar=False`` and the two colourbars get their own gridspec column.
``tight_layout`` is not used -- it cannot handle 3D axes, and says so.
"""

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn import plotting as nlp

W = "/work/users/x/y/xya"
OUT = f"{W}/wb-out"
DCONN = f"{W}/sbci-derivatives/sub-example_space-fsLR_den-32k_desc-concon_sc.dconn.nii"
N_FSLR = 32492
SEED_FSLR = 9286

geometry = {}
for H in ("L", "R"):
    g = nib.load(f"{OUT}/fsaverage.{H}.inflated.32k_fs_LR.surf.gii")
    geometry[H] = (np.asarray(g.darrays[0].data), np.asarray(g.darrays[1].data))

img = nib.load(DCONN)
seed = np.asarray(img.dataobj[SEED_FSLR], dtype=np.float64)

degree = np.zeros(2 * N_FSLR)
for start in range(0, 2 * N_FSLR, 4096):
    stop = min(start + 4096, 2 * N_FSLR)
    degree[start:stop] = np.asarray(img.dataobj[start:stop], dtype=np.float64).sum(axis=1)
np.save(f"{OUT}/degree_from_dconn.npy", degree)
print(
    f"degree map from the file: min {degree.min():.3g}, max {degree.max():.3g}, "
    f"{int((degree == 0).sum()):,} vertices at exactly zero (the medial wall)"
)

seed_top = float(np.percentile(seed[seed > 0], 99.5))
inside = degree > 0
degree_low = float(np.percentile(degree[inside], 1))
degree_top = float(np.percentile(degree[inside], 99))

panels = [("L", "lateral"), ("L", "medial"), ("R", "lateral"), ("R", "medial")]
rows = [
    dict(
        values=seed,
        cmap="inferno",
        vmin=0.0,
        vmax=seed_top,
        threshold=seed_top / 200,
        label=f"seed profile at fsLR vertex {SEED_FSLR}",
    ),
    dict(
        values=np.where(inside, degree, 0.0),
        cmap="viridis",
        vmin=degree_low,
        vmax=degree_top,
        threshold=degree_low,
        label="total connectivity per vertex",
    ),
]

figure = plt.figure(figsize=(17, 7.0))
grid = figure.add_gridspec(
    2,
    5,
    width_ratios=[1, 1, 1, 1, 0.075],
    left=0.005,
    right=0.94,
    top=0.875,
    bottom=0.01,
    wspace=0.0,
    hspace=0.02,
)

for r, row in enumerate(rows):
    for c, (H, view) in enumerate(panels):
        axis = figure.add_subplot(grid[r, c], projection="3d")
        offset = 0 if H == "L" else N_FSLR
        coords, faces = geometry[H]
        nlp.plot_surf_stat_map(
            (coords, faces),
            row["values"][offset : offset + N_FSLR],
            hemi="left" if H == "L" else "right",
            view=view,
            colorbar=False,
            cmap=row["cmap"],
            vmax=row["vmax"],
            threshold=row["threshold"],
            symmetric_cbar=False,
            axes=axis,
            figure=figure,
        )
        # nilearn leaves a wide margin around the mesh; pull the camera in so
        # the brains fill their cells instead of floating in whitespace.
        axis.set_box_aspect(None, zoom=1.35)

    # A colourbar spanning the whole row reads as a wall; keep it to the
    # middle 55% of the cell.
    cell = figure.add_subplot(grid[r, 4])
    cell.set_axis_off()
    bar_axis = cell.inset_axes([0.0, 0.22, 0.34, 0.56])
    bar = figure.colorbar(
        ScalarMappable(norm=Normalize(row["vmin"], row["vmax"]), cmap=row["cmap"]),
        cax=bar_axis,
    )
    bar.ax.tick_params(labelsize=9)
    bar_axis.set_ylabel(row["label"], fontsize=10, labelpad=8)
    bar_axis.yaxis.set_label_position("right")

for c, (H, view) in enumerate(panels):
    figure.text(0.005 + (c + 0.5) * 0.935 / 4.075, 0.885, f"{H} {view}", ha="center", fontsize=12)

figure.text(
    0.47, 0.960, "The exchange file, drawn from the .dconn.nii alone", ha="center", fontsize=15
)
figure.text(
    0.47,
    0.922,
    "fsLR-32k, 64,984 vertices; grey is the medial wall, which the file carries as exact zeros",
    ha="center",
    fontsize=10.5,
    color="0.35",
)
figure.savefig(f"{OUT}/exchange_file.png", dpi=110)
print(f"wrote {OUT}/exchange_file.png")
