r"""Convert the toolkit's ico4 surface meshes into the package's format.

Run once, commit the output, so plotting needs neither a checkout of
SBCI_Toolkit nor any VTK library at runtime.

    python tools/convert_surfaces.py \
        --source <SBCI_Toolkit>/example_data/fsaverage_label \
        --dest src/sbci/data/surfaces

Input
-----
ASCII VTK POLYDATA, one file per hemisphere per geometry, 2562 vertices and
5120 triangles each -- the ico4 icosphere. The ``_lps`` variants are used
where they exist because that is the convention ``load_sbci_surface.m``
reads; the sphere is only shipped in one orientation.

Output
------
One ``<name>_ico4.npz`` per geometry holding the two hemispheres already
joined into the 5124-vertex computational grid, left first:

``vertices``
    ``float32 (5124, 3)``
``faces``
    ``int32 (10240, 3)``, right-hemisphere indices offset by 2562 so they
    address the joined grid.

A minimal VTK reader is included rather than a dependency: these files are
plain ASCII and the alternative would put a mesh library in the build path of
a one-off conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

N_PER_HEMI = 2562
N_FACES_PER_HEMI = 5120

# Only the sphere is converted. The toolkit's inflated and white ico4 meshes
# are in a DIFFERENT vertex order from the sphere/grid mesh, which is the order
# the connectivity and the atlases use, so painting a map on them scatters it.
# Evidence, on the left hemisphere with the Desikan parcellation:
#
#   sphere and grid have identical face arrays; inflated and white do not
#   Desikan edge agreement   sphere 0.8415   inflated/white 0.2827
#
# 0.84 is a contiguous parcellation; 0.28 is scattered. That was the toolkit's
# `_lps_` exports, resolved in SPEC_QUESTIONS.md item 11; the anatomical
# surfaces are built by tools/build_surfaces.py instead.
GEOMETRIES = {
    "sphere": ("lh_sphere_avg_ico4.vtk", "rh_sphere_avg_ico4.vtk"),
}

# Faces are NOT taken from the geometry file being converted. Stored endpoint
# triangle indices refer to the pipeline grid's face order, and the toolkit's
# `example_data/fsaverage_label/lh_grid_avg_ico4.vtk` and `?h_sphere_avg_ico4.vtk`
# exports hold the same triangles in a different order -- reading faces from
# them is exactly how the bundled surfaces once came out wrong (SPEC_QUESTIONS.md
# item 13). Every geometry therefore takes its faces from canonical_faces().
CANONICAL_GRID_DIR = os.environ.get("SBCI_AVE", "/overflow/zzhanglab/ADNI/ADNI-bids/SBCI_AVE")
FACE_DIGEST = "87cf55ceb53254a7"
"""SHA-256 prefix of the canonical face list; tests/test_surface.py pins the same value."""


def canonical_faces(grid_dir: str | None = None) -> np.ndarray:
    """The face list stored triangle indices refer to: pipeline grid order.

    Left hemisphere first, then right with 2562 added. Verified to resolve the
    stored triangle indices of six subjects from two pipeline sources to within
    one edge of their stored nearest vertex (median 1.6 degrees).
    """
    grid_dir = grid_dir or CANONICAL_GRID_DIR
    lh = np.asarray(read_vtk_polydata(f"{grid_dir}/lh_grid_avg_ico4.vtk")[1], dtype=np.int64)
    rh = np.asarray(read_vtk_polydata(f"{grid_dir}/rh_grid_avg_ico4.vtk")[1], dtype=np.int64)
    faces = np.vstack([lh, rh + N_PER_HEMI]).astype(np.int32)
    digest = hashlib.sha256(faces.tobytes()).hexdigest()[:16]
    if digest != FACE_DIGEST:
        raise RuntimeError(
            f"{grid_dir} gives face digest {digest}, not the canonical {FACE_DIGEST}: "
            "this is not the pipeline grid order the stored triangle indices use"
        )
    return faces


def read_vtk_polydata(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read points and triangles from an ASCII VTK POLYDATA file."""
    with open(path) as handle:
        tokens = handle.read().split()

    try:
        start = tokens.index("POINTS")
    except ValueError as exc:
        raise ValueError(f"{path}: no POINTS section") from exc

    n_points = int(tokens[start + 1])
    # tokens[start + 2] is the dtype keyword.
    values = tokens[start + 3 : start + 3 + 3 * n_points]
    vertices = np.array(values, dtype=np.float64).reshape(n_points, 3)

    try:
        start = tokens.index("POLYGONS")
    except ValueError as exc:
        raise ValueError(f"{path}: no POLYGONS section") from exc

    n_faces = int(tokens[start + 1])
    total = int(tokens[start + 2])
    if total != 4 * n_faces:
        raise ValueError(
            f"{path}: {total} polygon entries for {n_faces} faces; "
            "this reader only handles triangles"
        )
    flat = np.array(tokens[start + 3 : start + 3 + total], dtype=np.int64).reshape(n_faces, 4)
    if not np.all(flat[:, 0] == 3):
        raise ValueError(f"{path}: contains a non-triangular face")
    return vertices, flat[:, 1:]


def join_hemispheres(left: str, right: str) -> tuple[np.ndarray, np.ndarray]:
    """Read both hemispheres and index them on the joined 5124-vertex grid."""
    lh_vertices, lh_faces = read_vtk_polydata(left)
    rh_vertices, rh_faces = read_vtk_polydata(right)

    for name, vertices in ((left, lh_vertices), (right, rh_vertices)):
        if vertices.shape[0] != N_PER_HEMI:
            raise ValueError(f"{name}: {vertices.shape[0]} vertices, expected {N_PER_HEMI}")

    vertices = np.vstack([lh_vertices, rh_vertices])
    del lh_faces, rh_faces  # same triangles, but not in the canonical order
    return vertices.astype(np.float32), canonical_faces()


def main(argv: list[str] | None = None) -> int:
    """Convert every geometry under ``--source`` into ``--dest``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="fsaverage_label directory")
    parser.add_argument("--dest", required=True, help="src/sbci/data/surfaces")
    args = parser.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    print(f"{'geometry':10s} {'vertices':>10s} {'faces':>8s} {'extent (mm)':>28s}")
    print("-" * 62)

    written = 0
    for name, (lh, rh) in GEOMETRIES.items():
        left = os.path.join(args.source, lh)
        right = os.path.join(args.source, rh)
        if not (os.path.exists(left) and os.path.exists(right)):
            print(f"{name:10s} missing: {lh if not os.path.exists(left) else rh}", file=sys.stderr)
            continue

        vertices, faces = join_hemispheres(left, right)
        out = os.path.join(args.dest, f"{name}_ico4.npz")
        np.savez_compressed(out, vertices=vertices, faces=faces)

        extent = vertices.max(axis=0) - vertices.min(axis=0)
        print(
            f"{name:10s} {vertices.shape[0]:10,d} {faces.shape[0]:8,d} "
            f"{extent[0]:8.1f} {extent[1]:8.1f} {extent[2]:8.1f}"
        )
        written += 1

    print(f"\nwrote {written} surfaces to {args.dest}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
