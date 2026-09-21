"""Build the ico4 anatomical surfaces from FreeSurfer's fsaverage.

    python tools/build_surfaces.py

The correspondence is `mapping_avg_ico4.npz`: for each of the 5124 grid
vertices it lists the full-resolution fsaverage vertices that vertex
represents, and those lists partition all 327,684 of them. The majority
annotation label over each list agrees with the bundled atlases 99.9% of the
time, which is what establishes the correspondence.

    position[i] = mean of the fsaverage coordinates over mapping[i]

Faces come from `lh/rh_grid_avg_ico4.vtk`, already in grid order.

An earlier attempt instead matched the grid sphere against the toolkit's VTK
export of the full-resolution sphere. That export is not in fsaverage's vertex
order, so the result was anatomically scrambled while still passing every
self-consistency check -- short even edges, contiguous parcels, plausible
extents. Only the checks against FreeSurfer's own annotation caught it, which
is why they run here and in tests/test_surface_anatomy.py.

No coordinates are adjusted. The inflated hemispheres are each centred on their
own origin and therefore overlap; plotting draws them in separate panels, and
shifting them would destroy the axis the medial/lateral tests read.
"""

from __future__ import annotations

import collections
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_surfaces import read_vtk_polydata  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbci import load_atlas  # noqa: E402

FS = "/work/users/x/y/xya/sbci-reference/SBCI_Pipeline/data/fsaverage"
D = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/example_data/fsaverage_label"
OUT = "/work/users/x/y/xya/cortical-meshes-v2"
N = 2562

GEOMETRIES = ("white", "inflated", "pial")


def clean(name: str) -> str:
    """Strip the toolkit's LH_/RH_ prefix and normalize the background name."""
    name = name.split("_", 1)[1] if name.startswith(("LH_", "RH_")) else name
    return "unknown" if name == "missing" else name


def annotation_names(hemi: str) -> list[str]:
    """Per-vertex region names from fsaverage, for one hemisphere."""
    values, _, names = nib.freesurfer.read_annot(f"{FS}/label/{hemi}.aparc.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return [names[v] if v >= 0 else "unknown" for v in values]


def main() -> int:
    """Build every geometry and validate it before writing."""
    os.makedirs(OUT, exist_ok=True)

    with np.load(f"{D}/mapping_avg_ico4.npz", allow_pickle=True) as data:
        mapping = [np.asarray(m).ravel() for m in data["mapping"]]

    faces = np.vstack(
        [
            read_vtk_polydata(f"{D}/lh_grid_avg_ico4.vtk")[1],
            read_vtk_polydata(f"{D}/rh_grid_avg_ico4.vtk")[1] + N,
        ]
    ).astype(np.int32)

    surfaces = {}
    for name in GEOMETRIES:
        lh, _ = nib.freesurfer.read_geometry(f"{FS}/surf/lh.{name}")
        rh, _ = nib.freesurfer.read_geometry(f"{FS}/surf/rh.{name}")
        joined = np.vstack([lh, rh])
        surfaces[name] = np.array([joined[m].mean(axis=0) for m in mapping], dtype=np.float32)
        extent = surfaces[name].max(axis=0) - surfaces[name].min(axis=0)
        print(f"  {name:9s} extent {np.round(extent, 1)}")

    print("\nvalidation 1 -- edges short and even")
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    for name, v in surfaces.items():
        lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        print(
            f"  {name:9s} mean {lengths.mean():6.2f} max {lengths.max():7.2f} "
            f"ratio {lengths.max() / lengths.mean():5.2f}"
        )

    print("\nvalidation 2 -- named regions in anatomically right places")
    atlas = load_atlas("Desikan")
    label_of = {n: i + 1 for i, n in enumerate(atlas.names)}

    def centroid(region: str, v: np.ndarray) -> np.ndarray:
        """Mean position of one region's vertices."""
        return v[atlas.labels == label_of[region]].mean(axis=0)

    for name, v in surfaces.items():
        checks = {
            "superiorfrontal anterior of lateraloccipital": centroid("LH_superiorfrontal", v)[1]
            > centroid("LH_lateraloccipital", v)[1],
            "temporalpole anterior of lateraloccipital": centroid("LH_temporalpole", v)[1]
            > centroid("LH_lateraloccipital", v)[1],
            "superiorfrontal superior of temporalpole": centroid("LH_superiorfrontal", v)[2]
            > centroid("LH_temporalpole", v)[2],
        }
        print(f"  {name:9s} {sum(checks.values())}/{len(checks)}")
        for text, ok in checks.items():
            if not ok:
                print(f"    FAIL  {text}")

    print("\nvalidation 3 -- labels agree with fsaverage's annotation")
    joined_names = annotation_names("lh") + annotation_names("rh")
    mine = ["unknown" if lab == 0 else clean(atlas.names[lab - 1]) for lab in atlas.labels]
    agree = sum(
        collections.Counter(joined_names[m] for m in members).most_common(1)[0][0] == mine[i]
        for i, members in enumerate(mapping)
    )
    fraction = agree / len(mapping)
    print(f"  {fraction:.1%}")
    if fraction < 0.95:
        print("  REFUSING to write: the correspondence is wrong")
        return 1

    for name, v in surfaces.items():
        np.savez_compressed(f"{OUT}/{name}_ico4.npz", vertices=v, faces=faces)
    print(f"\nwrote {len(surfaces)} surfaces to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
