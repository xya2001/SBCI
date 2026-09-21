"""sbci.align on real multi-subject HCP data.

The connectomes are the lab's SBCI-processed HCP test-retest tensors, on the
4121-vertex alignment grid that ConCon_Alignment itself uses. Read only.

The question is not whether the code runs -- that is already established -- but
whether alignment does what it is for: subjects should look more like each
other afterwards than before.
"""

import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/tools")
from convert_surfaces import read_vtk_polydata

import sbci
from sbci.alignment import Encore, SphericalGrid, rotate_off_poles

TENSOR = "/overflow/zzhanglab/HCP_TEST_RETEST/TEST/sbci_sc_tensor_1.mat"
GRID = "/work/users/x/y/xya/sbci-reference/ConCon_Alignment/grid"
N_SUBJECTS = 8
ORDER = 4

print("=== the grid ===")
grids = []
for hemi in ("lh", "rh"):
    vertices, faces = read_vtk_polydata(f"{GRID}/{hemi}_grid_avg_0.94.vtk")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    before = SphericalGrid(vertices, faces, ORDER)
    poles = before.pole_vertices().size
    grid = SphericalGrid(rotate_off_poles(vertices), faces, ORDER) if poles else before
    print(
        f"  {hemi}: {grid.n_vertices} vertices, {poles} on the coordinate axis"
        f"{' (rotated clear)' if poles else ''}, areas sum {grid.areas.sum():.4f}"
    )
    grids.append(grid)
lh_grid, rh_grid = grids
n = lh_grid.n_vertices + rh_grid.n_vertices

print(f"\n=== {N_SUBJECTS} HCP subjects from {TENSOR.split('/')[-1]} ===")
with h5py.File(TENSOR, "r") as handle:
    dataset = handle["sbci_sc_tensor"]
    print(f"  tensor {dataset.shape}, taking the first {N_SUBJECTS}")
    raw = np.asarray(dataset[:N_SUBJECTS], dtype=np.float64)

subjects = []
for i in range(N_SUBJECTS):
    matrix = raw[i]
    lower = np.abs(np.tril(matrix, -1)).max()
    matrix = matrix + matrix.T if lower == 0 else matrix
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    subjects.append(matrix)
    if i == 0:
        print(f"  stored as an upper triangle: {lower == 0}")
print(
    f"  each {subjects[0].shape}, total mass {subjects[0].sum():.6g}, "
    f"nonnegative: {all((s >= 0).all() for s in subjects)}"
)


def similarity(matrices):
    """Mean pairwise correlation between subjects, over the upper triangle."""
    upper = np.triu_indices(matrices[0].shape[0], 1)
    flat = np.stack([m[upper] for m in matrices])
    correlation = np.corrcoef(flat)
    return float(correlation[np.triu_indices(len(matrices), 1)].mean())


encore = Encore(lh_grid, rh_grid, step=0.05, max_iterations=6, delta=1e-5)
roots = [encore.root(s) for s in subjects]
before = similarity(roots)
print(f"\n  mean pairwise correlation before alignment: {before:.6f}")

print("\n=== align ===")
t0 = time.time()
result = sbci.align(
    subjects,
    grids=(lh_grid, rh_grid),
    order=ORDER,
    max_iterations=6,
    template_iterations=5,
    verbose=True,
)
print(f"\nfinished in {time.time() - t0:.0f}s: {result!r}")

aligned_roots = [encore.root(a) for a in result.aligned]
after = similarity(aligned_roots)
print("\n=== did alignment help? ===")
print(f"  mean pairwise correlation before : {before:.6f}")
print(f"  mean pairwise correlation after  : {after:.6f}")
print(f"  change                           : {after - before:+.6f}")

print("\n=== are the warps proper diffeomorphisms? ===")
for i, warp in enumerate(result.warps[:4]):
    moved = np.degrees(np.arccos(np.clip((warp.lh_vertices * lh_grid.vertices).sum(axis=1), -1, 1)))
    print(
        f"  subject {i + 1}: lh Jacobian [{warp.lh_jacobian.min():.4f}, "
        f"{warp.lh_jacobian.max():.4f}], moved {moved.mean():.2f} deg mean / "
        f"{moved.max():.2f} max"
    )
positive = all(w.lh_jacobian.min() > 0 and w.rh_jacobian.min() > 0 for w in result.warps)
print(f"  every Jacobian strictly positive: {positive}")
print(f"  costs: {[round(c, 6) for c in result.costs]}")
