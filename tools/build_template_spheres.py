r"""Build the sphere data that carries a warp between templates.

    python tools/build_template_spheres.py --fslr /work/users/x/y/xya/fslr

Writes two files into ``src/sbci/data``:

``surfaces/fsaverage_sphere_ico4.npz``
    FreeSurfer's standard-sphere coordinates of the grid's 5124 vertices
    (``vertices``, unit vectors) and the fsaverage vertex each grid vertex is
    (``fsaverage_index``). The grid's vertex ``i`` of a hemisphere is
    fsaverage's vertex ``i``: the pipeline built the grid from fsaverage's
    first 2,562 vertices, and its ``mapping_avg_ico4.npz`` lists them as ids
    ``0..2561`` on each side. Until October 2026 the match was made on the
    inflated surfaces instead, nearest vertex to nearest vertex, which picked a
    neighbouring fsaverage vertex for 27 grid vertices (14 left, 13 right),
    half a degree off on the standard sphere. Needs HCP's
    ``fsaverage_std_sphere.{L,R}.164k_fsavg_{L,R}.surf.gii``.

``templates/fslr32k_spheres.npz``
    Per hemisphere, ``{L,R}_standard`` (``sphere.32k_fs_LR``), ``{L,R}_deformed``
    (``fs_LR-deformed_to-fsaverage.sphere.32k_fs_LR``, the same vertices on the
    fsaverage sphere) as unit float32 vectors, and ``{L,R}_faces``. From HCP's
    standard_mesh_atlases (Washington-University/HCPpipelines,
    ``global/templates/standard_mesh_atlases``), whose licence permits this.

Digests of the files as shipped: fsaverage_sphere_ico4.npz cf7feb4cc89eaf26,
fslr32k_spheres.npz cee3e4ad50866fd9 (first 16 hex digits of SHA-256).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HALF = 2562


def unit(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def gifti(path):
    import nibabel as nib

    image = nib.load(path)
    return (
        np.asarray(image.darrays[0].data, dtype=np.float64),
        np.asarray(image.darrays[1].data, dtype=np.int64),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fslr", required=True, help="directory with HCP's standard mesh atlases")
    parser.add_argument("--out", default=os.path.join("src", "sbci", "data"))
    args = parser.parse_args(argv)

    vertices = np.zeros((2 * HALF, 3))
    index = np.zeros(2 * HALF, dtype=np.int32)
    for hemisphere, lo in (("L", 0), ("R", HALF)):
        standard, _ = gifti(
            f"{args.fslr}/fsaverage_std_sphere.{hemisphere}.164k_fsavg_{hemisphere}.surf.gii"
        )
        # grid vertex i is fsaverage vertex i, as the pipeline's mapping lists it
        vertices[lo : lo + HALF] = unit(standard[:HALF])
        index[lo : lo + HALF] = np.arange(HALF)
    path = os.path.join(args.out, "surfaces", "fsaverage_sphere_ico4.npz")
    np.savez_compressed(path, vertices=vertices, fsaverage_index=index)

    spheres = {}
    for hemisphere in ("L", "R"):
        standard, faces = gifti(f"{args.fslr}/{hemisphere}.sphere.32k_fs_LR.surf.gii")
        deformed, faces_too = gifti(
            f"{args.fslr}/fs_LR-deformed_to-fsaverage.{hemisphere}.sphere.32k_fs_LR.surf.gii"
        )
        if not np.array_equal(faces, faces_too):
            raise SystemExit(f"{hemisphere}: the two fs_LR spheres do not share a face list")
        spheres[f"{hemisphere}_standard"] = unit(standard).astype(np.float32)
        spheres[f"{hemisphere}_deformed"] = unit(deformed).astype(np.float32)
        spheres[f"{hemisphere}_faces"] = faces.astype(np.int32)
    other = os.path.join(args.out, "templates", "fslr32k_spheres.npz")
    np.savez_compressed(other, **spheres)
    for written in (path, other):
        digest = hashlib.sha256(open(written, "rb").read()).hexdigest()[:16]
        print(f"wrote {written} ({os.path.getsize(written) / 1e6:.2f} MB, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
