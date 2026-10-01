r"""Build HCP Young Adult subjects in the package's format, from the lab's SBCI pipeline output.

    python tools/build_hcp_cohort.py --layout young-adult \
        --manifest manifest.csv \
        --pipeline /overflow/zzhanglab/encore_project/encore_paper_code/prediction_subs \
        --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \
        --out /work/users/x/y/xya/hcp-ya/data

``manifest.csv`` has a ``subject`` column (``sub-100307``); other columns are
ignored, so no age or sex reaches the files. ``<pipeline>/<id>/`` is the lab's
copy of the subjects of the ENCORE project: the snapped streamlines
(``snapped_fibers.npz``) and the FreeSurfer-registered spheres
(``?h_sphere_freesurfer_reg.vtk``, stored in RAS and negated in x and y to
reach the grid's frame).

Each subject's endpoints are the snapped streamlines placed on its registered
sphere (:meth:`sbci.smoothing.Endpoints.from_snapped`), the branch of the
pipeline its smoothed connectome comes from; ``mesh_intersections_ico4.mat``
is the other, unsnapped branch, and does not re-smooth to the stored
connectome (PORTING.md item 6). The lab's young adult connectomes are on the
retired 0.94 grid, so the SC is smoothed here with the package's kernel
(``shk``, bandwidth 0.005, the medial wall masked), converted as
``tools/import_legacy.py`` does -- medial wall zeroed, unit mass -- and stored
with those endpoints. There is no FC: the lab's copy has no resting-state data.

Every file written is validated. With ``--check`` each file's endpoints are
re-smoothed and compared with its stored connectome.
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
from import_legacy import N_VERTICES, area_weights  # noqa: E402

LAYOUTS = ("young-adult",)
SNAPPED = "snapped_fibers.npz"
BANDWIDTH = 0.005
RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])


def subject_directory(pipeline: str, subject: str) -> str:
    """Where one subject's pipeline output lives."""
    return os.path.join(pipeline, subject.removeprefix("sub-"))


def registered_spheres(source: str) -> list[np.ndarray]:
    """The subject's FreeSurfer-registered spheres, left and right, in the grid's frame."""
    spheres = []
    for hemisphere in ("lh", "rh"):
        vertices, _ = read_vtk_polydata(
            os.path.join(source, f"{hemisphere}_sphere_freesurfer_reg.vtk")
        )
        spheres.append(np.asarray(vertices, dtype=np.float64) * RAS_TO_LPS)
    return spheres


def endpoints_of(source: str):
    """The streamline endpoints the pipeline's smoothing starts from."""
    from sbci.smoothing import Endpoints

    return Endpoints.from_snapped(os.path.join(source, SNAPPED), *registered_spheres(source))


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


def convert(subject, source, areas, mask, out, check=False) -> str:
    """One subject's SC, with its endpoints; returns the path written."""
    import sbci
    from sbci.grid import to_condensed
    from sbci.io import write_hdf5
    from sbci.metadata import template

    t = time.time()
    endpoints = endpoints_of(source)
    smoothed = carrier(endpoints, areas, mask).smooth(
        kernel="shk", bandwidth=BANDWIDTH, mask_medial_wall=True
    )
    dense = np.asarray(smoothed.dense(), dtype=np.float64)
    leaked = float(np.abs(dense[~mask]).sum())
    dense[~mask, :] = 0.0
    dense[:, ~mask] = 0.0
    dense /= float(areas @ dense @ areas)
    metadata = template(
        "sc",
        registration_reference="fsaverage",
        pipeline_version="SBCI_Pipeline endpoints; sbci smoothing",
        container_version="none (built from the lab's pipeline output)",
        entry_point="preprocessed",
        provenance=f"HCP Young Adult {subject}; smoothed by sbci (kernel shk, bandwidth "
        f"{BANDWIDTH}); endpoints from {SNAPPED} on the registered sphere; built by "
        "tools/build_hcp_cohort.py",
        normalization="unit-mass",
        kernel="shk",
        bandwidth=BANDWIDTH,
        streamline_count=int(endpoints.n_streamlines),
        streamline_weighting="none",
    )
    path = os.path.join(out, f"{subject}_sc.h5")
    write_hdf5(
        path,
        data=to_condensed(dense).astype(np.float32),
        area=areas,
        mask=mask,
        metadata=metadata,
        endpoints=endpoints,
    )
    back = sbci.load(path)
    agreement = ""
    if check:
        again = carrier(back.endpoints, areas, mask).smooth(
            kernel="shk", bandwidth=BANDWIDTH, mask_medial_wall=True
        )
        r = float(np.corrcoef(np.asarray(again.data), np.asarray(back.data))[0, 1])
        agreement = f"; endpoints re-smooth to the stored SC at r = {r:.6f}"
    print(
        f"{subject} sc: medial-wall mass removed {leaked:.3g}; wrote "
        f"{os.path.getsize(path) / 1e6:.1f} MB; endpoints {back.endpoints.n_streamlines:,}; "
        f"{time.time() - t:.0f}s{agreement}",
        flush=True,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--layout", choices=LAYOUTS, default="young-adult", help="the pipeline's layout"
    )
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
        source = subject_directory(args.pipeline, subject)
        written.append(convert(subject, source, areas, mask, args.out, args.check))
    failures = 0
    for path in written:
        failures += not all(check.passed for check in validate_file(path))
    print(f"wrote {len(written)} files, {failures} failed validation", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
