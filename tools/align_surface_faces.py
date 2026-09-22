"""Put the bundled surfaces' faces in the order stored triangle indices mean.

The HDF5 format stores each streamline endpoint as a triangle index plus
barycentric weights, and those indices come from the pipeline's grid mesh
(``?h_grid_avg_ico4.m``, and the matching ``.vtk``, which carry identical face
order). The bundled surfaces held the same 10,240 triangles in a different
order, so every stored triangle index resolved to an unrelated triangle --
about 85 degrees away on average, which made ``smooth(kernel="shk")`` produce
a density uncorrelated with the reference.

Vertices were never affected: all four geometries already agree with the grid
to 6e-12. Only the face list is reordered here, and all four share one.

Run with the lab grid files reachable; the output is committed.
"""

import sys
from importlib import resources

import numpy as np

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/tools")
from convert_surfaces import read_vtk_polydata

AVE = "/overflow/zzhanglab/ADNI/ADNI-bids/SBCI_AVE"
N_PER_HEMI = 2562

lh_faces = np.asarray(read_vtk_polydata(f"{AVE}/lh_grid_avg_ico4.vtk")[1], dtype=np.int32)
rh_faces = np.asarray(read_vtk_polydata(f"{AVE}/rh_grid_avg_ico4.vtk")[1], dtype=np.int32)
canonical = np.vstack([lh_faces, rh_faces + N_PER_HEMI]).astype(np.int32)
print(f"canonical face list: {canonical.shape}, lh {lh_faces.shape[0]} + rh {rh_faces.shape[0]}")

directory = resources.files("sbci.data.surfaces")
for name in ("inflated", "white", "pial", "sphere"):
    path = directory / f"{name}_ico4.npz"
    with np.load(str(path), allow_pickle=False) as data:
        stored = {k: data[k] for k in data.files}
    old = np.asarray(stored["faces"])
    same_set = {tuple(sorted(t)) for t in old} == {tuple(sorted(t)) for t in canonical}
    already = np.array_equal(np.sort(old, axis=1), np.sort(canonical, axis=1))
    stored["faces"] = canonical
    np.savez_compressed(str(path), **stored)
    print(
        f"  {name:9s} faces {old.shape} -> {canonical.shape}  "
        f"same triangles: {same_set}  was already ordered: {already}"
    )
print("\nrewritten")
