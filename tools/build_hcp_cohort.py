r"""Build the ten-subject HCP-Aging example cohort in the package's format.

    python tools/build_hcp_cohort.py \
        --manifest manifest.csv \
        --pipeline /overflow/zzhanglab/HCP_Aging_Development_Data/HCP_preprocess_data/HCP_Aging \
        --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \
        --out /work/users/x/y/xya/hcp-example/data

``manifest.csv`` has the columns ``subject`` (``sub-HCA...``), ``age_years``
and ``sex``. For each subject the script reads the pipeline's smoothed SC,
FC and endpoints (``dwi_pipeline/sbci_connectome/`` under ``ses-V1_MR``),
converts them exactly as ``tools/import_legacy.py`` does -- medial wall
zeroed, SC normalized to unit mass, FC left as correlations -- attaches the
endpoints with their barycentric positions to the SC file, and validates
every file written. One subject's SC with endpoints is 40 to 56 MB; FC is
40 MB.

The Longleaf paths are the lab's; nothing here is redistributed by the
package. The subjects' ages are recorded in the manifest, not in the files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_legacy import N_VERTICES, area_weights, read_legacy  # noqa: E402

SESSION = "ses-V1_MR"
FILES = {
    "sc": ("smoothed_sc_avg_0.005_ico4.mat", "sc"),
    "fc": ("fc_avg_ico4.mat", "fc"),
}
ENDPOINTS = "mesh_intersections_ico4.mat"


def convert(subject: str, row: dict, source: str, areas, mask, out: str) -> list[str]:
    """One subject's SC (with endpoints) and FC; returns the paths written."""
    import sbci
    from sbci.grid import to_condensed, to_dense
    from sbci.io import write_hdf5
    from sbci.metadata import template
    from sbci.smoothing import Endpoints

    endpoints = Endpoints.from_matlab(os.path.join(source, ENDPOINTS))
    written = []
    for modality, (name, variable) in FILES.items():
        t = time.time()
        dense = to_dense(read_legacy(os.path.join(source, name), variable), N_VERTICES)
        leaked = float(np.abs(dense[~mask]).sum())
        dense[~mask, :] = 0.0
        dense[:, ~mask] = 0.0
        if modality == "sc":
            dense /= float(areas @ dense @ areas)
            extra = {
                "normalization": "unit-mass",
                "kernel": "shk",
                "bandwidth": 0.005,
                "streamline_count": int(endpoints.n_streamlines),
                "streamline_weighting": "unrecorded",
            }
        else:
            extra = {"normalization": "none", "fc_nuisance_model": "unrecorded"}
        metadata = template(
            modality,
            registration_reference="fsaverage",
            pipeline_version="legacy-SBCI_Pipeline",
            container_version="none (imported from legacy output)",
            entry_point="precomputed",
            provenance=f"HCP-Aging {subject}, {SESSION}; imported from {name} by "
            "tools/build_hcp_cohort.py",
            **extra,
        )
        path = os.path.join(out, f"{subject}_{modality}.h5")
        write_hdf5(
            path,
            data=to_condensed(dense).astype(np.float32),
            area=areas,
            mask=mask,
            metadata=metadata,
            endpoints=endpoints if modality == "sc" else None,
        )
        back = sbci.load(path)
        streamlines = back.endpoints.n_streamlines if back.has_endpoints else 0
        print(
            f"{subject} {modality}: medial-wall mass removed {leaked:.3g}; wrote "
            f"{os.path.getsize(path) / 1e6:.1f} MB; endpoints {streamlines:,}; "
            f"{time.time() - t:.0f}s",
            flush=True,
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="subject, age_years, sex")
    parser.add_argument("--pipeline", required=True, help="directory holding sub-*/ses-V1_MR")
    parser.add_argument("--mapping", required=True, help="mapping_avg_ico4.npz")
    parser.add_argument("--out", required=True, help="destination directory")
    args = parser.parse_args(argv)

    from sbci.atlas import load_atlas
    from sbci.validate import validate_file

    os.makedirs(args.out, exist_ok=True)
    areas = area_weights(args.mapping)
    mask = load_atlas("aparc").labels != 0
    with open(args.manifest) as handle:
        rows = list(csv.DictReader(handle))
    written = []
    for row in rows:
        subject = row["subject"]
        source = os.path.join(args.pipeline, subject, SESSION, "dwi_pipeline", "sbci_connectome")
        written += convert(subject, row, source, areas, mask, args.out)
    failures = 0
    for path in written:
        failures += not all(check.passed for check in validate_file(path))
    print(f"wrote {len(written)} files, {failures} failed validation", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
