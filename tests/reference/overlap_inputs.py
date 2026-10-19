"""Inputs for overlap_reference.m: two subjects' MdLF endpoints from the released files.

    python tests/reference/overlap_inputs.py

The first two subjects of the lab's ENCORE bundle evaluation
(``encore_dice_scores/get_dice_scores.m``: its list sorted, the second onwards) and their
middle longitudinal fasciculus, left and right, as RecoBundles labelled it there. A released SC
file keeps, in order, the streamlines of the pipeline's ``snapped_fibers.npz`` with both ends on
the white surfaces (``Endpoints.from_snapped``), which RecoBundles' indices count. Their
endpoints are written on the bundled grids, with the grids, for the MATLAB to read; the test
(``tests/test_matlab_reference.py``, the overlap tests) rebuilds them from the same file. Needs
the lab's data on Longleaf, named below.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

import sbci
from sbci.conseal import EndpointConnectome, default_grids

LAB = Path("/overflow/zzhanglab/encore_project/encore_dice_scores")
PIPELINE = Path("/overflow/zzhanglab/encore_project/encore_paper_code/prediction_subs")
SC = Path("/work/users/x/y/xya/hcp-ya/full/data")
OUT = Path("/work/users/x/y/xya/matlab-reference/overlap")
BUNDLE = ("results/MdLF_L", "results/MdLF_R")

lines = (LAB / "dice_subs").read_text().split("\n")[:-1]
subjects = sorted(set(lines))[1:3]
lh_grid, rh_grid = default_grids()
records = {}
for key, subject in zip(("X", "Y"), subjects, strict=True):
    connectome = EndpointConnectome.from_endpoints(
        sbci.load(SC / f"sub-{subject}_sc.h5").endpoints, lh_grid, rh_grid
    )
    with np.load(PIPELINE / subject / "snapped_fibers.npz") as snapped:
        ends = [snapped[name].astype(np.int64) for name in ("surf_ids0", "surf_ids1")]
    kept = np.flatnonzero(((ends[0] >= 0) & (ends[0] <= 1)) & ((ends[1] >= 0) & (ends[1] <= 1)))
    assert kept.size == connectome.n_streamlines
    with open(LAB / subject / "results" / "results.json") as handle:
        labels = json.load(handle)
    listed = np.concatenate(
        [np.asarray(labels[name]["indices"], dtype=np.int64) for name in BUNDLE]
    )
    pick = np.isin(kept, listed)
    points_in, points_out = connectome.positions()
    records[key] = {
        "st_points": points_in[pick],
        "en_points": points_out[pick],
        "st_hemi": connectome.hemisphere_in[pick].astype(float),
        "en_hemi": connectome.hemisphere_out[pick].astype(float),
    }
    print(f"{subject}: {int(pick.sum())} MdLF streamlines", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
sio.savemat(
    OUT / "overlap_inputs.mat",
    dict(
        records,
        lh_V=lh_grid.vertices,
        lh_T=lh_grid.faces + 1,
        rh_V=rh_grid.vertices,
        rh_T=rh_grid.faces + 1,
    ),
)
print("wrote", OUT / "overlap_inputs.mat")
