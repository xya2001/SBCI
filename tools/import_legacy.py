r"""Convert legacy SBCI pipeline output into the package's HDF5 format.

    python tools/import_legacy.py \\
        --sc  <...>/smoothed_sc_avg_0.005_ico4.mat \\
        --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \\
        --subject 100307 --out derivatives/

Every existing SBCI cohort is in the legacy layout, so this is how the released
data gets produced without rerunning the pipeline, and how anyone with old
output can try the package today.

What the legacy files hold
--------------------------
``smoothed_sc_*.mat`` carries one variable ``sc``: a ``5124 x 5124`` float64
array written as the **strict upper triangle**, with the lower triangle and the
diagonal exactly zero. ``fc_*.mat`` carries ``fc`` in the same layout but with
ones on the diagonal, alongside ``sub_surf_fc`` and ``sub_sub_fc``, which cover
19 subcortical regions and are not part of the surface connectome.

Area weights are not stored with the connectivity. ``parcellate_sc.m`` derives
them from the ico4 mapping as the number of high-resolution fsaverage vertices
that fall on each ico4 vertex, and this script does the same; they sum to
327,684, the full two-hemisphere fsaverage surface.

The medial wall is taken from the Desikan (``aparc``) parcellation, whose
unassigned vertices are FreeSurfer's own definition of it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

N_VERTICES = 5124


def area_weights(mapping_path: str) -> np.ndarray:
    """Vertex areas: how many high-resolution vertices map onto each ico4 vertex."""
    with np.load(mapping_path, allow_pickle=True) as z:
        mapping = z["mapping"]
    areas = np.array([np.asarray(m).size for m in mapping], dtype=np.float64)
    if areas.size != N_VERTICES:
        raise ValueError(f"{mapping_path}: {areas.size} vertices, expected {N_VERTICES}")
    if (areas == 0).any():
        raise ValueError(f"{mapping_path}: {(areas == 0).sum()} vertices have zero area")
    return areas


def read_legacy(path: str, variable: str) -> np.ndarray:
    """Read a legacy matrix and return its strict upper triangle, condensed."""
    import scipy.io

    from sbci.grid import to_condensed

    contents = scipy.io.loadmat(path)
    if variable not in contents:
        available = [k for k in contents if not k.startswith("__")]
        raise KeyError(f"{path} has no variable {variable!r}; it has {available}")

    matrix = np.asarray(contents[variable], dtype=np.float64)
    if matrix.shape != (N_VERTICES, N_VERTICES):
        raise ValueError(f"{path}: shape {matrix.shape}, expected ({N_VERTICES}, {N_VERTICES})")

    lower = matrix[np.tril_indices(N_VERTICES, k=-1)]
    if np.any(lower != 0):
        # A full symmetric matrix would also be fine, but silently accepting one
        # when only half the mass was written is not, so say which we got.
        raise ValueError(
            f"{path}: the lower triangle is not empty, so this is not the "
            "upper-triangular layout this importer expects"
        )
    return to_condensed(matrix)


def main(argv: list[str] | None = None) -> int:
    """Convert one subject's legacy SC and/or FC."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sc", help="legacy smoothed_sc_*.mat")
    parser.add_argument("--fc", help="legacy fc_*.mat")
    parser.add_argument("--mapping", required=True, help="mapping_avg_ico4.npz")
    parser.add_argument("--subject", required=True, help="subject id, e.g. 100307")
    parser.add_argument("--out", default=".", help="destination directory")
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.005,
        help="smoothing bandwidth the legacy SC was built with",
    )
    parser.add_argument(
        "--streamline-count",
        type=int,
        default=None,
        help="streamlines used; recorded in the metadata",
    )
    args = parser.parse_args(argv)

    if not args.sc and not args.fc:
        parser.error("pass --sc, --fc, or both")

    from sbci.atlas import load_atlas
    from sbci.grid import to_dense
    from sbci.io import write_hdf5
    from sbci.metadata import template

    areas = area_weights(args.mapping)
    mask = load_atlas("aparc").labels != 0
    os.makedirs(args.out, exist_ok=True)
    print(f"area weights: sum={areas.sum():.0f}, cortex vertices={int(mask.sum())}/{N_VERTICES}")

    written = []
    for modality, path, variable in (("sc", args.sc, "sc"), ("fc", args.fc, "fc")):
        if not path:
            continue
        print(f"\n--- {modality.upper()} from {os.path.basename(path)}")
        condensed = read_legacy(path, variable)

        # The medial wall carries no connectivity. The legacy files do put mass
        # there, so it is zeroed here and the amount reported: a large figure
        # means the mask and the data disagree about the grid.
        dense = to_dense(condensed, N_VERTICES)
        leaked = float(np.abs(dense[~mask]).sum())
        dense[~mask, :] = 0.0
        dense[:, ~mask] = 0.0
        print(f"    medial-wall mass removed: {leaked:.6g}")

        extra = {}
        if modality == "sc":
            # Structural connectivity is a density: normalize so that the
            # area-weighted total integrates to one, which is what the
            # validator's unit-mass check expects.
            total = float(areas @ dense @ areas)
            dense /= total
            print(f"    normalized to unit mass (was {total:.6g})")
            extra = {
                "normalization": "unit-mass",
                "kernel": "rdk",
                "bandwidth": args.bandwidth,
                "streamline_count": args.streamline_count or "unrecorded",
                "streamline_weighting": "unrecorded",
            }
        else:
            # FC is a correlation field, not a density; rescaling it would
            # destroy the units. Nothing is normalized.
            extra = {"normalization": "none", "fc_nuisance_model": "unrecorded"}

        from sbci.grid import to_condensed

        metadata = template(
            modality,
            registration_reference="fsaverage",
            pipeline_version="legacy-SBCI_Pipeline",
            container_version="none (imported from legacy output)",
            entry_point="precomputed",
            provenance=f"imported by tools/import_legacy.py from {os.path.basename(path)}",
            **extra,
        )

        out = os.path.join(args.out, f"sub-{args.subject}_{modality}.h5")
        write_hdf5(
            out,
            data=to_condensed(dense).astype(np.float32),
            area=areas,
            mask=mask,
            metadata=metadata,
        )
        size = os.path.getsize(out) / 1e6
        print(f"    wrote {out} ({size:.1f} MB)")
        written.append(out)

    print(f"\nwrote {len(written)} file(s). Check them with:")
    for path in written:
        print(f"    sbci validate {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
