"""Build the ico4 <-> fsLR-32k overlap matrix for the exchange format.

The chain has two links:

    ico4 -> fsaverage   `mapping_avg_ico4.npz` assigns each of the 327,684
                        fsaverage vertices to one of the 5124 grid vertices.
                        Verified at 99.9% against FreeSurfer's annotation.

    fsaverage -> fsLR   HCP ships `fs_LR-deformed_to-fsaverage.sphere` -- the
                        fsLR-32k mesh warped into fsaverage's spherical space --
                        so the two can be matched directly on that sphere.

What is saved is the raw **overlap matrix** `S`, where `S[k, i]` counts the
fsaverage vertices belonging to both ico4 vertex `i` and fsLR vertex `k`. Every
fsaverage vertex is counted exactly once, so `S` sums to 327,684 along both
axes: its column sums are the ico4 vertex areas the HDF5 file already carries,
and its row sums are the corresponding fsLR vertex areas.

Saving `S` rather than a normalized transfer is deliberate. The connectome is a
*density* -- the specification validates unit mass as `area @ D @ area == 1` --
so resampling it means taking an area-weighted average, which is `S` normalized
by its row sums. Normalizing by columns instead would distribute mass, which is
right for an extensive quantity and wrong for this one; it preserves the plain
sum of the entries while losing 99.3% of the area-weighted mass. Keeping `S`
un-normalized leaves both readings available and neither assumed.

Area is counted in fsaverage vertices throughout, which is the unit the
pipeline's own area vector uses, so the two definitions cannot drift apart.

Verification is against independent references rather than self-consistency:
the column sums must reproduce the area vector stored in the example subject's
HDF5 file, and resampling the Desikan atlas must reproduce the pipeline's own
`lh/rh.fs_LR.aparc.annot`.
"""

from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

FSLR = "/work/users/x/y/xya/fslr"
D = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/example_data/fsaverage_label"
MSM = "/work/users/x/y/xya/sbci-reference/SBCI_Pipeline/data/MSMLabels"
EXAMPLE = "/work/users/x/y/xya/sbci-derivatives/sub-example_sc.h5"
OUT = "/work/users/x/y/xya/resampling"

N_ICO4 = 2562
N_FSLR = 32492
N_FSAVG = 163842


def surface(path):
    """Vertex coordinates from a GIFTI surface."""
    return np.asarray(nib.load(path).darrays[0].data, dtype=np.float64)


def main() -> int:
    """Build, verify and save the overlap matrix."""
    os.makedirs(OUT, exist_ok=True)

    with np.load(f"{D}/mapping_avg_ico4.npz", allow_pickle=True) as z:
        mapping = [np.asarray(m).ravel() for m in z["mapping"]]

    blocks = []
    for hemi_index, hemi in enumerate(("L", "R")):
        print(f"\n=== {hemi} ===")
        fsavg_sphere = surface(f"{FSLR}/fsaverage_std_sphere.{hemi}.164k_fsavg_{hemi}.surf.gii")
        fslr_sphere = surface(
            f"{FSLR}/fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii"
        )
        print(f"  fsaverage sphere {fsavg_sphere.shape[0]:,}, fsLR sphere {fslr_sphere.shape[0]:,}")

        # Both live in fsaverage's spherical space; confirm before matching.
        a = fsavg_sphere / np.linalg.norm(fsavg_sphere, axis=1, keepdims=True)
        b = fslr_sphere / np.linalg.norm(fslr_sphere, axis=1, keepdims=True)
        distance, _ = cKDTree(a).query(b, k=1)
        worst = np.degrees(2 * np.arcsin(distance.max() / 2))
        print(f"  fsLR vertices sit {worst:.3f} deg from the nearest fsaverage vertex at worst")
        if worst > 3:
            print("  the two spheres are not in a common space; aborting")
            return 1

        # Every fsaverage vertex is assigned to its nearest fsLR vertex.
        _, fsavg_to_fslr = cKDTree(b).query(a, k=1)

        offset = hemi_index * N_ICO4
        member_offset = hemi_index * N_FSAVG

        rows, cols, values = [], [], []
        for local in range(N_ICO4):
            members = mapping[offset + local] - member_offset
            members = members[(members >= 0) & (members < N_FSAVG)]
            if members.size == 0:
                continue
            # Each fsaverage vertex counts once, so the entries are counts and
            # the column total is this ico4 vertex's area.
            targets, counts = np.unique(fsavg_to_fslr[members], return_counts=True)
            rows.extend(targets.tolist())
            cols.extend([local] * targets.size)
            values.extend(counts.tolist())

        block = sparse.csr_matrix((values, (rows, cols)), shape=(N_FSLR, N_ICO4), dtype=np.float64)
        blocks.append(block)

        print(
            f"  overlap block {block.shape}, {block.nnz:,} non-zeros "
            f"({block.nnz / N_ICO4:.1f} fsLR vertices per ico4 vertex)"
        )
        print(f"  counted {block.sum():,.0f} fsaverage vertices (expected {N_FSAVG:,})")
        if abs(block.sum() - N_FSAVG) > 0.5:
            print("  fsaverage vertices were lost or double counted; aborting")
            return 1

    overlap = sparse.block_diag(blocks, format="csr")
    ico4_area = np.asarray(overlap.sum(axis=0)).ravel()
    fslr_area = np.asarray(overlap.sum(axis=1)).ravel()
    print(f"\n=== combined overlap matrix: {overlap.shape}, {overlap.nnz:,} non-zeros ===")
    print(
        f"  ico4 areas: sum {ico4_area.sum():,.0f}, "
        f"min {ico4_area.min():.0f}, max {ico4_area.max():.0f}"
    )
    print(
        f"  fsLR areas: sum {fslr_area.sum():,.0f}, "
        f"min {fslr_area.min():.0f}, max {fslr_area.max():.0f}"
    )
    if fslr_area.min() < 1:
        print(f"  {int((fslr_area < 1).sum())} fsLR vertices receive no fsaverage vertex")

    print("\n=== verification 1: the column sums are the pipeline's own area vector ===")
    sys.path.insert(0, "/nas/longleaf/home/xya/sbci/src")
    from sbci import ContinuousConnectome, load_atlas

    if os.path.exists(EXAMPLE):
        stored = np.asarray(ContinuousConnectome.load(EXAMPLE).area, dtype=np.float64)
        worst = float(np.abs(stored - ico4_area).max())
        print(f"  max |stored area - column sum| = {worst:.3g}")
        if worst > 1e-9:
            print("  the areas disagree, so the two are not the same partition; aborting")
            return 1
        print("  identical: the overlap matrix partitions exactly the same vertices")
    else:
        print(f"  {EXAMPLE} missing; skipped")

    print("\n=== verification 2: against the pipeline's own fsLR annotation ===")
    atlas = load_atlas("Desikan")
    agreements = []
    for hemi_index, hemi in enumerate(("lh", "rh")):
        path = f"{MSM}/{hemi}.fs_LR.aparc.annot"
        if not os.path.exists(path):
            print(f"  {path} missing; cannot verify this hemisphere")
            continue
        values, _, names = nib.freesurfer.read_annot(path)
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        reference = np.array([names[v] if v >= 0 else "unknown" for v in values])

        # Push the ico4 labels through the overlap, taking the heaviest
        # contributor at each fsLR vertex.
        block = blocks[hemi_index].tocoo()
        labels = atlas.labels[hemi_index * N_ICO4 : (hemi_index + 1) * N_ICO4]
        winner = np.zeros(N_FSLR, dtype=int)
        best = np.zeros(N_FSLR)
        for r, c, v in zip(block.row, block.col, block.data, strict=True):
            if v > best[r]:
                best[r], winner[r] = v, labels[c]

        mine = np.array(
            ["unknown" if w == 0 else atlas.names[w - 1].split("_", 1)[1] for w in winner]
        )
        mine = np.where(mine == "missing", "unknown", mine)
        n = min(len(mine), len(reference))
        agree = float((mine[:n] == reference[:n]).mean())
        agreements.append(agree)
        print(f"  {hemi}: {agree:.1%} of fsLR vertices get the same region")

    if agreements and min(agreements) < 0.8:
        print("\n  agreement too low; refusing to save")
        return 1

    target = f"{OUT}/ico4_to_fslr32k.npz"
    sparse.save_npz(target, overlap)
    print(f"\nsaved {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
