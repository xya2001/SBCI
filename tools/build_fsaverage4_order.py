"""Where each grid vertex sits in FreeSurfer's fsaverage4: the same mesh, numbered differently.

    python tools/build_fsaverage4_order.py [SUBJECTS_DIR] [MAPPING_DIR]

The ico4 grid is FreeSurfer's fsaverage4 -- 2,562 vertices a hemisphere and the
same triangles -- with its vertices in another order. This finds the order, so
that :func:`sbci.save_map` can write GIFTI files FreeSurfer's own tools read
against their fsaverage4 subject.

Each grid vertex is matched to the nearest fsaverage4 vertex on the inflated
surface, and the match is accepted only if it is one to one, every vertex lies
within half a millimetre of its match, and it carries the grid's triangles onto
fsaverage4's exactly. A second derivation, from different files, must agree:
fsaverage4's vertices are fsaverage's first 2,562, in another order (on the
spheres they coincide to 0.01 mm, which gives that order), and the grid's
correspondence with fsaverage, ``mapping_avg_ico4.npz``, puts exactly one of
those 2,562 in each grid vertex's list.

``SUBJECTS_DIR`` defaults to ``$FREESURFER_HOME/subjects`` (FreeSurfer 7.4.1 on
Longleaf: ``module load freesurfer/7.4.1``), and ``MAPPING_DIR`` to the
toolkit's ``fsaverage_label`` folder that ``tools/build_surfaces.py`` reads.
Writes ``fsaverage4_order_ico4.npz`` into ``src/sbci/data/surfaces``: ``lh``
and ``rh``, the fsaverage4 index of each grid vertex of that hemisphere, and
``faces_sha256``, the digest of fsaverage4's triangles (both hemispheres, each
as sorted index triples, sorted), which the test suite checks the grid's faces
against without FreeSurfer installed.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sbci import load_surface  # noqa: E402
from sbci.spec import N_VERTICES_PER_HEMI  # noqa: E402

OUT = HERE.parent / "src" / "sbci" / "data" / "surfaces" / "fsaverage4_order_ico4.npz"
MAPPING = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/example_data/fsaverage_label"
TOLERANCE_MM = 0.5
N_FSAVERAGE_PER_HEMI = 163842


def faces_digest(faces_by_hemisphere) -> str:
    """SHA-256 of the triangles as sorted index triples, sorted, left hemisphere first."""
    digest = hashlib.sha256()
    for faces in faces_by_hemisphere:
        triples = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
        triples = triples[np.lexsort(triples.T[::-1])]
        digest.update(np.ascontiguousarray(triples, dtype="<i8").tobytes())
    return digest.hexdigest()


def main() -> int:
    """Derive the order both ways, refuse any disagreement, and write it."""
    subjects = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.environ["FREESURFER_HOME"], "subjects")
    )
    mapping_dir = Path(sys.argv[2] if len(sys.argv) > 2 else MAPPING)
    with np.load(mapping_dir / "mapping_avg_ico4.npz", allow_pickle=True) as data:
        mapping = [np.asarray(m).ravel().astype(np.int64) for m in data["mapping"]]

    inflated = load_surface("inflated")
    orders, fs_faces = {}, []
    for side, hemi in enumerate(("lh", "rh")):
        grid = inflated.hemisphere("LR"[side])
        coarse, faces4 = nib.freesurfer.read_geometry(
            str(subjects / "fsaverage4/surf" / f"{hemi}.inflated")
        )
        if coarse.shape[0] != N_VERTICES_PER_HEMI:
            raise SystemExit(f"{hemi}: fsaverage4 has {coarse.shape[0]} vertices")
        gap, nearest = cKDTree(coarse).query(np.asarray(grid.vertices, dtype=np.float64))
        if np.unique(nearest).size != N_VERTICES_PER_HEMI:
            raise SystemExit(f"{hemi}: the nearest-vertex match is not one to one")
        if gap.max() > TOLERANCE_MM:
            raise SystemExit(f"{hemi}: a grid vertex lies {gap.max():.3f} mm from its match")
        carried = {tuple(sorted(t)) for t in nearest[np.asarray(grid.faces)]}
        theirs = {tuple(sorted(t)) for t in faces4}
        if carried != theirs:
            raise SystemExit(f"{hemi}: {len(carried - theirs)} grid triangles are not fsaverage4's")

        # The second derivation. fsaverage4's vertices are fsaverage's first
        # 2,562 in another order: on the spheres each lies on one of them ...
        sphere4, _ = nib.freesurfer.read_geometry(
            str(subjects / "fsaverage4/surf" / f"{hemi}.sphere")
        )
        sphere, _ = nib.freesurfer.read_geometry(
            str(subjects / "fsaverage/surf" / f"{hemi}.sphere")
        )
        apart, index4 = cKDTree(sphere4).query(sphere[:N_VERTICES_PER_HEMI])
        # 0.05 mm on a sphere of radius 100, where neighbours lie about 7 mm apart.
        if apart.max() > 0.05 or np.unique(index4).size != N_VERTICES_PER_HEMI:
            raise SystemExit(
                f"{hemi}: fsaverage's first vertices are not fsaverage4's ({apart.max()})"
            )
        # ... and each grid vertex's list in the correspondence holds exactly one of them.
        offset = side * N_VERTICES_PER_HEMI
        second = np.empty(N_VERTICES_PER_HEMI, dtype=np.int64)
        for i in range(N_VERTICES_PER_HEMI):
            members = mapping[offset + i] % N_FSAVERAGE_PER_HEMI
            coarse_members = members[members < N_VERTICES_PER_HEMI]
            if coarse_members.size != 1:
                raise SystemExit(
                    f"{hemi} grid vertex {i}: {coarse_members.size} of fsaverage's first vertices"
                )
            second[i] = index4[coarse_members[0]]
        if not np.array_equal(second, nearest):
            raise SystemExit(
                f"{hemi}: the two derivations disagree at {(second != nearest).sum()} vertices"
            )
        orders[hemi] = nearest.astype(np.int16)
        fs_faces.append(faces4)
        print(
            f"{hemi}: one to one, within {gap.max():.3f} mm (median {np.median(gap):.4f}), "
            f"all {len(theirs)} triangles carried onto fsaverage4's; the correspondence agrees"
        )

    grid_faces = []
    for side, hemi in enumerate(("lh", "rh")):
        part = inflated.hemisphere("LR"[side])
        grid_faces.append(orders[hemi].astype(np.int64)[np.asarray(part.faces)])
    digest = faces_digest(fs_faces)
    if faces_digest(grid_faces) != digest:
        raise SystemExit("the digest of the carried triangles differs from fsaverage4's")
    np.savez_compressed(OUT, lh=orders["lh"], rh=orders["rh"], faces_sha256=np.array(digest))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); fsaverage4 triangles sha256 {digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
