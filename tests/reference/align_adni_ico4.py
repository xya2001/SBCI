"""sbci.align on real multi-subject data, on ico4 -- the package's own grid.

The earlier multi-subject check (scripts/align_hcp_cohort.py) used the HCP
test-retest tensors, which exist only on the retired 4121-vertex 0.94 grid.
These five ADNI subjects are the pipeline's ico4 output, so this is the check
that counts for the package: alignment must make the subjects more alike, and
every Jacobian must stay positive. Reads lab data; run by hand:

    python tests/reference/align_adni_ico4.py
"""

import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/src")
sys.path.insert(0, "/nas/longleaf/home/xya/sbci/tools")
from import_legacy import area_weights

import sbci
from sbci.grid import to_condensed

ROOT = "/overflow/zzhanglab/ADNI/ADNI-bids"
SUBJECTS = ("sub-168S6561", "sub-068S0473", "sub-002S0413", "sub-041S4974", "sub-013S4268")

area = area_weights(f"{ROOT}/SBCI_AVE/mapping_avg_ico4.npz")
mask = np.asarray(sbci.load_atlas("Desikan").labels) != 0
meta = sbci.metadata.template(
    "sc",
    normalization="unit-mass",
    registration_reference="fsaverage",
    pipeline_version="legacy-SBCI_Pipeline",
    container_version="unrecorded",
    streamline_count=0,
    streamline_weighting="unrecorded",
    kernel="shk",
    bandwidth=0.005,
)

subjects = []
for s in SUBJECTS:
    sc = np.asarray(
        sio.loadmat(f"{ROOT}/{s}/dwi_pipeline/sbci_connectome/smoothed_sc_avg_0.005_ico4.mat")[
            "sc"
        ],
        dtype=np.float64,
    )
    dense = sc + sc.T
    np.fill_diagonal(dense, 0.0)
    dense /= area @ dense @ area
    subjects.append(
        sbci.ContinuousConnectome(to_condensed(dense).astype(np.float32), area, mask, meta)
    )
    print(
        f"  {s}: unit mass {float(area @ subjects[-1].dense().astype(np.float64) @ area):.6f}",
        flush=True,
    )


def mean_pairwise_r(mats):
    rs = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            rs.append(np.corrcoef(mats[i], mats[j])[0, 1])
    return float(np.mean(rs)), float(np.min(rs)), float(np.max(rs))


before = [c.data.astype(np.float64) for c in subjects]
print(
    f"\nbefore alignment: mean pairwise r {mean_pairwise_r(before)[0]:.6f} "
    f"(min {mean_pairwise_r(before)[1]:.6f}, max {mean_pairwise_r(before)[2]:.6f})",
    flush=True,
)

t0 = time.time()
result = sbci.align(subjects, verbose=False)
print(f"\nalign() finished in {time.time() - t0:.0f}s: {result!r}", flush=True)
after = [np.asarray(m, dtype=np.float64).ravel() for m in result.aligned]
r_after = mean_pairwise_r(after)
r_before = mean_pairwise_r(before)
print(
    f"after alignment : mean pairwise r {r_after[0]:.6f} "
    f"(min {r_after[1]:.6f}, max {r_after[2]:.6f})"
)
print(f"improvement     : {r_after[0] - r_before[0]:+.6f}")
print(f"costs           : {[round(float(c), 6) for c in result.costs]}")
print("\nJacobians (every one must be strictly positive):")
worst = np.inf
for name, warp in zip(SUBJECTS, result.warps, strict=True):
    lh, rh = np.asarray(warp.lh_jacobian, float), np.asarray(warp.rh_jacobian, float)
    worst = min(worst, lh.min(), rh.min())
    print(f"  {name}: lh [{lh.min():.4f}, {lh.max():.4f}]  rh [{rh.min():.4f}, {rh.max():.4f}]")
print(
    f"\n{'PASS' if worst > 0 and r_after[0] > r_before[0] else 'FAIL'}: "
    f"smallest Jacobian {worst:.4f}, correlation {r_before[0]:.4f} -> {r_after[0]:.4f}"
)
