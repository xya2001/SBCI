r"""Build an HCP example cohort in the package's format, from the lab's SBCI pipeline output.

    python tools/build_hcp_cohort.py --layout aging \
        --manifest manifest.csv \
        --pipeline /overflow/zzhanglab/HCP_Aging_Development_Data/HCP_preprocess_data/HCP_Aging \
        --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \
        --out /work/users/x/y/xya/hcp-example/data

    python tools/build_hcp_cohort.py --layout young-adult \
        --manifest manifest.csv \
        --pipeline /overflow/zzhanglab/encore_project/encore_paper_code/prediction_subs \
        --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \
        --out /work/users/x/y/xya/hcp-ya/data

``manifest.csv`` has a ``subject`` column (``sub-HCA...`` for HCP-Aging,
``sub-100206`` for the young adults); other columns are ignored, so no age
or sex reaches the files.

Both layouts take the streamline endpoints from the pipeline's snapped
streamlines (``snapped_fibers.npz``) placed on the subject's registered
sphere, which is the branch the pipeline's own smoothed connectome comes from
(:meth:`sbci.smoothing.Endpoints.from_snapped`). ``mesh_intersections_ico4.mat``,
which earlier versions of this tool used, is the pipeline's other branch, the
unsnapped intersections, and does not re-smooth to the stored connectome
(PORTING.md item 6).

``aging``
    ``<pipeline>/<subject>/ses-V1_MR/dwi_pipeline/sbci_connectome/``: the
    pipeline's ico4 SC and FC, and the registered spheres
    ``?h_sphere_reg_lps.vtk``, already in the grid's frame. The SC is the
    pipeline's, converted as ``tools/import_legacy.py`` does -- medial wall
    zeroed, unit mass -- and FC is left as correlations.
``young-adult``
    ``<pipeline>/<id>/``, the lab's copy of the HCP Young Adult subjects of the
    ENCORE paper: snapped streamlines and FreeSurfer-registered spheres
    ``?h_sphere_freesurfer_reg.vtk``, stored in RAS and negated in x and y to
    reach the grid's frame. The lab's young-adult connectomes are on the 0.94
    grid, so the SC is smoothed here with the package's kernel (``shk``,
    bandwidth 0.005, the medial wall masked), and there is no FC.

Every file written is validated. With ``--check`` each SC file's endpoints are
re-smoothed and compared with its stored connectome, which is how the
endpoints were shown to belong to it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_surfaces import read_vtk_polydata  # noqa: E402
from import_legacy import N_VERTICES, area_weights, read_legacy  # noqa: E402

LAYOUTS = ("aging", "young-adult")
SESSION = "ses-V1_MR"
SNAPPED = "snapped_fibers.npz"
BANDWIDTH = 0.005
RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])


def subject_directory(layout: str, pipeline: str, subject: str) -> str:
    """Where one subject's pipeline output lives."""
    if layout == "aging":
        return os.path.join(pipeline, subject, SESSION, "dwi_pipeline", "sbci_connectome")
    return os.path.join(pipeline, subject.removeprefix("sub-"))


def registered_spheres(layout: str, source: str) -> list[np.ndarray]:
    """The subject's registered spheres, left and right, in the grid's frame."""
    spheres = []
    for hemisphere in ("lh", "rh"):
        if layout == "aging":
            vertices, _ = read_vtk_polydata(
                os.path.join(source, f"{hemisphere}_sphere_reg_lps.vtk")
            )
        else:
            name = f"{hemisphere}_sphere_freesurfer_reg.vtk"
            vertices, _ = read_vtk_polydata(os.path.join(source, name))
            vertices = vertices * RAS_TO_LPS
        spheres.append(np.asarray(vertices, dtype=np.float64))
    return spheres


def endpoints_of(layout: str, source: str):
    """The streamline endpoints that belong with the pipeline's smoothed connectome."""
    from sbci.smoothing import Endpoints

    return Endpoints.from_snapped(
        os.path.join(source, SNAPPED), *registered_spheres(layout, source)
    )


def carrier(endpoints, areas, mask):
    """A connectome with no data yet, carrying ``endpoints`` on the ico4 grid."""
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import condensed_size
    from sbci.metadata import template

    return ContinuousConnectome(
        data=np.zeros(condensed_size(N_VERTICES), dtype=np.float32),
        area=areas,
        mask=mask,
        metadata=template(
            "sc",
            registration_reference="fsaverage",
            pipeline_version="sbci",
            container_version="none",
            entry_point="preprocessed",
            normalization="unit-mass",
            kernel="shk",
            bandwidth=BANDWIDTH,
            streamline_count=int(endpoints.n_streamlines),
            streamline_weighting="none",
        ),
        endpoints=endpoints,
    )


def convert(layout, subject, source, areas, mask, out, check=False) -> list[str]:
    """One subject's SC (with endpoints) and, for HCP-Aging, FC; returns the paths written."""
    import sbci
    from sbci.grid import to_condensed, to_dense
    from sbci.io import write_hdf5
    from sbci.metadata import template

    t = time.time()
    endpoints = endpoints_of(layout, source)
    written = []
    if layout == "aging":
        dataset = f"HCP-Aging {subject}, {SESSION}"
        sources = {"sc": ("smoothed_sc_avg_0.005_ico4.mat", "sc"), "fc": ("fc_avg_ico4.mat", "fc")}
    else:
        dataset = f"HCP Young Adult {subject}"
        sources = {"sc": None}
    for modality, pipeline_file in sources.items():
        if pipeline_file is not None:
            name, variable = pipeline_file
            dense = to_dense(read_legacy(os.path.join(source, name), variable), N_VERTICES)
            imported = f"imported from {name}"
            entry_point, pipeline_version = "precomputed", "legacy-SBCI_Pipeline"
        else:
            smoothed = carrier(endpoints, areas, mask).smooth(
                kernel="shk", bandwidth=BANDWIDTH, mask_medial_wall=True
            )
            dense = np.asarray(smoothed.dense(), dtype=np.float64)
            imported = f"smoothed by sbci (kernel shk, bandwidth {BANDWIDTH})"
            entry_point, pipeline_version = (
                "preprocessed",
                "SBCI_Pipeline endpoints; sbci smoothing",
            )
        leaked = float(np.abs(dense[~mask]).sum())
        dense[~mask, :] = 0.0
        dense[:, ~mask] = 0.0
        if modality == "sc":
            dense /= float(areas @ dense @ areas)
            extra = {
                "normalization": "unit-mass",
                "kernel": "shk",
                "bandwidth": BANDWIDTH,
                "streamline_count": int(endpoints.n_streamlines),
                "streamline_weighting": "unrecorded" if layout == "aging" else "none",
            }
        else:
            extra = {"normalization": "none", "fc_nuisance_model": "unrecorded"}
        metadata = template(
            modality,
            registration_reference="fsaverage",
            pipeline_version=pipeline_version,
            container_version="none (built from the lab's pipeline output)",
            entry_point=entry_point,
            provenance=f"{dataset}; {imported}; endpoints from {SNAPPED} on the registered "
            "sphere; built by tools/build_hcp_cohort.py",
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
        agreement = ""
        if check and modality == "sc":
            again = carrier(back.endpoints, areas, mask).smooth(
                kernel="shk", bandwidth=BANDWIDTH, mask_medial_wall=True
            )
            r = float(np.corrcoef(np.asarray(again.data), np.asarray(back.data))[0, 1])
            agreement = f"; endpoints re-smooth to the stored SC at r = {r:.6f}"
        print(
            f"{subject} {modality}: medial-wall mass removed {leaked:.3g}; wrote "
            f"{os.path.getsize(path) / 1e6:.1f} MB; endpoints {streamlines:,}; "
            f"{time.time() - t:.0f}s{agreement}",
            flush=True,
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", choices=LAYOUTS, default="aging")
    parser.add_argument("--manifest", required=True, help="a csv with a subject column")
    parser.add_argument("--pipeline", required=True, help="directory of the subjects' output")
    parser.add_argument("--mapping", required=True, help="mapping_avg_ico4.npz")
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument(
        "--check", action="store_true", help="re-smooth each SC file's endpoints and compare"
    )
    args = parser.parse_args(argv)

    from sbci.atlas import load_atlas
    from sbci.validate import validate_file

    os.makedirs(args.out, exist_ok=True)
    areas = area_weights(args.mapping)
    mask = load_atlas("aparc").labels != 0
    with open(args.manifest) as handle:
        subjects = [row["subject"] for row in csv.DictReader(handle)]
    written = []
    for subject in subjects:
        source = subject_directory(args.layout, args.pipeline, subject)
        written += convert(args.layout, subject, source, areas, mask, args.out, args.check)
    failures = 0
    for path in written:
        failures += not all(check.passed for check in validate_file(path))
    print(f"wrote {len(written)} files, {failures} failed validation", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
