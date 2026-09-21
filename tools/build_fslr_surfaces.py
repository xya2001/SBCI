"""Build fsLR-32k anatomical surfaces from fsaverage.

fsLR-32k ships spheres but no anatomical geometry, so there is nothing to draw
an exchange file on. Each fsLR vertex takes the mean fsaverage position over
the full-resolution vertices nearest to it, the same construction
``tools/build_surfaces.py`` uses for the ico4 meshes, with faces taken from
HCP's deformed sphere. Inflated, white and pial are written as ``.surf.gii``,
so the ``.dconn`` can be opened against them in ``wb_view``.

The script also draws a seed profile from the exchange file beside the same
seed on the ico4 grid, as a check that the two carry the same data.
"""

import matplotlib
import nibabel as nib
import numpy as np
from nibabel import gifti
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting as nlp

W = "/work/users/x/y/xya"
FSLR = f"{W}/fslr"
FS = f"{W}/sbci-reference/SBCI_Pipeline/data/fsaverage"
DCONN = f"{W}/sbci-derivatives/sub-example_space-fsLR_den-32k_desc-concon_sc.dconn.nii"
OUT = f"{W}/wb-out"
N_FSAVG, N_FSLR, N_ICO4 = 163842, 32492, 2562
SEED = 1234

import os

os.makedirs(OUT, exist_ok=True)

print("=== build fsLR-32k anatomical surfaces from fsaverage ===")
geometry = {}
for H, h in (("L", "lh"), ("R", "rh")):
    sphere = nib.load(f"{FSLR}/fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii")
    fslr_coords = np.asarray(sphere.darrays[0].data, dtype=np.float64)
    fslr_faces = np.asarray(sphere.darrays[1].data, dtype=np.int32)
    fsavg_sphere = np.asarray(
        nib.load(f"{FSLR}/fsaverage_std_sphere.{H}.164k_fsavg_{H}.surf.gii").darrays[0].data,
        dtype=np.float64,
    )
    a = fsavg_sphere / np.linalg.norm(fsavg_sphere, axis=1, keepdims=True)
    b = fslr_coords / np.linalg.norm(fslr_coords, axis=1, keepdims=True)
    _, fsavg_to_fslr = cKDTree(b).query(a, k=1)

    for name in ("inflated", "white", "pial"):
        coords, _ = nib.freesurfer.read_geometry(f"{FS}/surf/{h}.{name}")
        summed = np.zeros((N_FSLR, 3))
        counts = np.zeros(N_FSLR)
        np.add.at(summed, fsavg_to_fslr, coords)
        np.add.at(counts, fsavg_to_fslr, 1.0)
        counts[counts == 0] = 1.0
        positions = summed / counts[:, None]
        geometry[(H, name)] = (positions.astype(np.float32), fslr_faces)
        if name == "inflated":
            extent = positions.max(axis=0) - positions.min(axis=0)
            print(
                f"  {H} {name}: {N_FSLR:,} vertices, extent "
                f"{extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} mm, "
                f"{int((counts == 1).sum())} vertices with a single source"
            )

    # Save for Workbench, so the dconn can be opened in wb_view against them.
    for name in ("inflated", "white", "pial"):
        positions, faces = geometry[(H, name)]
        image = gifti.GiftiImage(
            darrays=[
                gifti.GiftiDataArray(positions.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
                gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
            ]
        )
        image.to_filename(f"{OUT}/fsaverage.{H}.{name}.32k_fs_LR.surf.gii")
print(f"  wrote 6 .surf.gii files to {OUT}")

print("\n=== the seed, on both grids ===")
import sys

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/src")
from sbci import ContinuousConnectome
from sbci.io.cifti import transfer

sc = ContinuousConnectome.load(f"{W}/sbci-derivatives/sub-example_sc.h5")
row_ico4 = np.asarray(sc.seed(vertex=SEED), dtype=np.float64)

P = transfer()
column = P[:, SEED].toarray().ravel()
seed_fslr = int(np.argmax(column))
print(
    f"  ico4 vertex {SEED} -> fsLR vertex {seed_fslr} "
    f"(weight {column[seed_fslr]:.3f} of {int((column > 0).sum())} covering vertices)"
)

img = nib.load(DCONN)
row_fslr = np.asarray(img.dataobj[seed_fslr], dtype=np.float64)
print(f"  fsLR row: {row_fslr.shape}, max {row_fslr.max():.4g}; ico4 row max {row_ico4.max():.4g}")

expected = P @ row_ico4
print(
    f"  correlation of the file's row with the resampled ico4 row: "
    f"{np.corrcoef(row_fslr, expected)[0, 1]:.6f}"
)

print("\n=== render ===")
top = float(np.percentile(row_fslr[row_fslr > 0], 99.5))
panels = [("L", "lateral"), ("L", "medial"), ("R", "lateral"), ("R", "medial")]
figure, axes = plt.subplots(2, 4, figsize=(19, 8), subplot_kw={"projection": "3d"})

from sbci.surface import load_surface

ico4 = load_surface("inflated")
ico4_coords, ico4_faces = np.asarray(ico4.vertices), np.asarray(ico4.faces)

for column_index, (H, view) in enumerate(panels):
    hemi_index = 0 if H == "L" else 1
    # top row: ico4
    lo, hi = hemi_index * N_ICO4, (hemi_index + 1) * N_ICO4
    keep = (ico4_faces >= lo).all(axis=1) & (ico4_faces < hi).all(axis=1)
    nlp.plot_surf_stat_map(
        (ico4_coords[lo:hi], ico4_faces[keep] - lo),
        row_ico4[lo:hi],
        hemi="left" if H == "L" else "right",
        view=view,
        colorbar=False,
        cmap="inferno",
        vmax=top,
        threshold=top / 200,
        bg_on_data=True,
        axes=axes[0, column_index],
        figure=figure,
    )
    axes[0, column_index].set_title(f"ico4  {H} {view}", fontsize=11)

    positions, faces = geometry[(H, "inflated")]
    values = row_fslr[hemi_index * N_FSLR : (hemi_index + 1) * N_FSLR]
    nlp.plot_surf_stat_map(
        (positions, faces),
        values,
        hemi="left" if H == "L" else "right",
        view=view,
        colorbar=False,
        cmap="inferno",
        vmax=top,
        threshold=top / 200,
        bg_on_data=True,
        axes=axes[1, column_index],
        figure=figure,
    )
    axes[1, column_index].set_title(f"fsLR-32k from the .dconn  {H} {view}", fontsize=11)

figure.suptitle(
    f"Seed profile at ico4 vertex {SEED} / fsLR vertex {seed_fslr}: "
    f"computational file (top) against the 16.9 GB exchange file (bottom), "
    f"same colour scale",
    fontsize=13,
)
figure.tight_layout()
figure.savefig(f"{OUT}/seed_ico4_vs_fslr.png", dpi=110, bbox_inches="tight")
print(f"  wrote {OUT}/seed_ico4_vs_fslr.png")
