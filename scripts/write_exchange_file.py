"""Write the exchange file for the example subject, then verify it end to end.

Three claims are checked, each against something outside the resampling code:

1. The written file is a valid CIFTI-2 dense connectome that plain nibabel can
   open, with the structures and mesh sizes the header promises.
2. The area-weighted mass the specification validates -- `area @ D @ area` --
   survives the move to fsLR-32k. This is the invariant that a distributing
   (column-stochastic) transfer would destroy.
3. Parcellating the fsLR file with the *pipeline's own* fsLR Desikan
   annotation reproduces the region-by-region matrix obtained on the ico4 grid.
   The two calculations share no code and no label file, so agreement is
   evidence about the correspondence rather than about the implementation.
"""

import os
import time

import nibabel as nib
import numpy as np

from sbci import ContinuousConnectome, load_atlas
from sbci.io.cifti import N_FSLR_PER_HEMI, vertex_areas

DATA = "/work/users/x/y/xya/sbci-derivatives"
MSM = "/work/users/x/y/xya/sbci-reference/SBCI_Pipeline/data/MSMLabels"
TARGET = f"{DATA}/sub-example_space-fsLR_den-32k_desc-concon_sc.dconn.nii"
BLOCK = 4096

sc = ContinuousConnectome.load(f"{DATA}/sub-example_sc.h5")
dense = sc.dense().astype(np.float64)
area = np.asarray(sc.area, dtype=np.float64)
mass_ico4 = float(area @ dense @ area)
print(f"ico4: {dense.shape}, plain sum {dense.sum():.10g}")
print(f"      area-weighted mass a'Da = {mass_ico4:.12g}   <- the specification's unit mass")

atlas = load_atlas("Desikan")
region_ico4 = sc.to_atlas(atlas, how="mass")
print(f"      Desikan {region_ico4.shape}, total {region_ico4.sum():.10g}")

print("\n=== write ===")
t0 = time.time()
path = sc.to_cifti(TARGET)
print(f"  wrote in {time.time() - t0:.0f}s")
for f in (
    path,
    str(path).replace(".dconn.nii", ".json"),
    str(path).replace(".dconn.nii", "_vertexarea.dscalar.nii"),
):
    if os.path.exists(f):
        print(f"  {os.path.basename(f)}  {os.path.getsize(f) / 1e9:.3f} GB")
del dense

print("\n=== read back with plain nibabel, no sbci code ===")
img = nib.load(str(path))
print(f"  shape {img.shape}")
for i in range(2):
    ax = img.header.get_axis(i)
    print(f"  axis {i}: {type(ax).__name__}, {len(ax)} elements, structures {sorted(set(ax.name))}")
    print(f"    vertices per structure: {dict(ax.nvertices)}")

# Areas taken from the companion file the writer produced, not from the package.
areas_img = nib.load(str(path).replace(".dconn.nii", "_vertexarea.dscalar.nii"))
fslr_area = np.asarray(areas_img.get_fdata()).ravel().astype(np.float64)
bundled, _ = vertex_areas()
print(
    f"  companion areas: {fslr_area.shape}, sum {fslr_area.sum():,.0f}, "
    f"max |difference| from the bundled vector {np.abs(fslr_area - bundled).max():.3g}"
)

# fsLR Desikan labels from the pipeline's own annotation.
labels = np.zeros(2 * N_FSLR_PER_HEMI, dtype=int)
lookup = {name: i + 1 for i, name in enumerate(atlas.names)}
for hemi_index, (hemi, prefix) in enumerate((("lh", "LH"), ("rh", "RH"))):
    values, _, names = nib.freesurfer.read_annot(f"{MSM}/{hemi}.fs_LR.aparc.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    offset = hemi_index * N_FSLR_PER_HEMI
    for vertex, value in enumerate(values):
        if value >= 0:
            labels[offset + vertex] = lookup.get(f"{prefix}_{names[value]}", 0)
print(
    f"  fsLR labels from {MSM}: {int((labels > 0).sum()):,} of "
    f"{labels.size:,} vertices are in a Desikan region"
)

weights = np.zeros((labels.size, len(atlas.names)))
for column in range(len(atlas.names)):
    member = labels == column + 1
    weights[member, column] = fslr_area[member]

print("\n=== stream the file: mass and parcellation together ===")
t0 = time.time()
mass_fslr = 0.0
region_fslr = np.zeros((len(atlas.names),) * 2)
total = 0.0
for start in range(0, img.shape[0], BLOCK):
    stop = min(start + BLOCK, img.shape[0])
    rows = np.asarray(img.dataobj[start:stop], dtype=np.float64)
    mass_fslr += float(fslr_area[start:stop] @ (rows @ fslr_area))
    region_fslr += weights[start:stop].T @ (rows @ weights)
    total += float(rows.sum())
print(f"  streamed {img.shape[0]:,} rows in {time.time() - t0:.0f}s")

print("\n=== 1. the invariant the specification validates ===")
print(f"  ico4 area-weighted mass : {mass_ico4:.12g}")
print(f"  fsLR area-weighted mass : {mass_fslr:.12g}")
print(f"  relative change         : {abs(mass_fslr - mass_ico4) / mass_ico4:.3e}")

print("\n=== 2. the plain sum, which is NOT the invariant ===")
print(
    f"  plain sum on fsLR: {total:.10g}  "
    f"(ico4 was {float(np.asarray(sc.dense(), dtype=np.float64).sum()):.10g}; "
    f"a density need not preserve this)"
)

print("\n=== 3. Desikan on fsLR against Desikan on ico4 ===")
print(f"  ico4 total {region_ico4.sum():.10g}, fsLR total {region_fslr.sum():.10g}")
denominator = np.linalg.norm(region_ico4)
print(
    f"  relative Frobenius difference : "
    f"{np.linalg.norm(region_fslr - region_ico4) / denominator:.4f}"
)
print(
    f"  correlation of the 68x68 entries: "
    f"{np.corrcoef(region_ico4.ravel(), region_fslr.ravel())[0, 1]:.6f}"
)
upper = np.triu_indices(len(atlas.names), 1)
ratio = region_fslr[upper] / np.where(region_ico4[upper] == 0, np.nan, region_ico4[upper])
print(
    f"  off-diagonal entry ratio: median {np.nanmedian(ratio):.4f}, "
    f"5th {np.nanpercentile(ratio, 5):.4f}, 95th {np.nanpercentile(ratio, 95):.4f}"
)
worst = np.nanargmax(np.abs(ratio - 1))
print(
    f"  worst pair: {atlas.names[upper[0][worst]]} - {atlas.names[upper[1][worst]]}, "
    f"ico4 {region_ico4[upper][worst]:.4g} against fsLR {region_fslr[upper][worst]:.4g}"
)

print("\n=== 4. symmetry ===")
corner = np.asarray(img.dataobj[:400, :400])
print(f"  symmetric on a 400x400 corner: {np.allclose(corner, corner.T, atol=1e-12)}")
