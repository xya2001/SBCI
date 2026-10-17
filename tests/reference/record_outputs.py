"""Record the heavy methods on the released young adults.

    SBCI_HCP_DIR=hcp-ya python tests/reference/record_outputs.py OUT.npz

The arrays the recording of 2 October 2026 held, under the same keys (the package's defaults,
now the corrected mathematics). ``conseal_*`` runs at the settings that recording used -- the
public code's, which were the package's defaults until PORTING.md item 19 made the paper's the
defaults -- so that the same key holds the same measurement in every recording;
``consealpaper_*`` runs at the paper's settings as they were the defaults from 7 to 11 October
2026, and ``consealnarrow_*`` at the defaults since: the kernel narrowed to 0.0027 and the
floor under the threshold. Every ConSEAL key spells out its kernel and its floor, so that a
change of default never moves a recorded key. Also what the reviews changed elsewhere: the
projection of the training cohort, and a region seed profile over a region that overlaps the
medial wall.

Recordings made before 11 October 2026 also hold the reference's arithmetic: ``alignref_*``,
ENCORE with ``reference=True``, which reproduced the 2 October recording bit for bit on the
same CPU type; ``consealref_*``, ConSEAL at the public settings on grids built with
``reference=True``; and ``project_reference``, the reference's projection. PORTING.md item 21
removed those switches, and the keys with them.
"""

import os
import sys
import time

import numpy as np

import sbci
from sbci.atlas import load_atlas
from sbci.conseal import (
    PUBLIC_DELTA,
    PUBLIC_SIGMA,
    PUBLIC_STEP_CLAMP,
    PUBLIC_THRESHOLD,
    PUBLIC_VISCOSITY,
)
from sbci.reduction import project
from sbci.smoothing import endpoint_positions

D = os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data")
SUBJECTS = ["sub-100307", "sub-103010", "sub-108121", "sub-116221"]
out = {}
t = time.time()
sc = sbci.load(f"{D}/sub-100307_sc.h5")
fc = sbci.load(f"{D}/sub-100307_fc.h5")
desikan = load_atlas("Desikan")
out["desikan_mass"] = np.asarray(sc.to_atlas(desikan))
out["desikan_mean"] = np.asarray(sc.to_atlas(desikan, how="mean"))
out["schaefer_fc"] = np.asarray(fc.to_atlas("Schaefer200"))
out["coupling_global"] = sc.coupling(fc)
out["coupling_region"] = sc.coupling(fc, scope="region", labels=desikan.labels)
out["coupling_discrete"] = sc.coupling(fc, scope="discrete")
out["seed"] = sc.seed(vertex=1234)
limbic = load_atlas("PALS_B12_Lobes").region_mask("LH_LOBE.LIMBIC")
out["seed_region"] = sc.seed(region=limbic)
cortical = float(sc.area[limbic & sc.mask].sum())
masked = float(sc.area[limbic & ~sc.mask].sum())
out["seed_region_old_ratio"] = np.array([cortical / (cortical + masked)])
print(f"parcellation, coupling, seed: {time.time() - t:.0f}s", flush=True)
print(
    f"limbic region: {int((limbic & sc.mask).sum())} cortical and {int((limbic & ~sc.mask).sum())} "
    f"masked vertices; the old profile was {cortical / (cortical + masked):.6f} of the new",
    flush=True,
)

t = time.time()
out["smooth"] = sc.smooth(kernel="shk", mask_medial_wall=True).data
print(f"smooth: {time.time() - t:.0f}s", flush=True)

t = time.time()
paths = [f"{D}/{s}_sc.h5" for s in SUBJECTS]
reduction = sbci.reduce(paths, rank=4)
out["reduce_basis"] = reduction.basis
out["reduce_scores"] = reduction.scores
out["reduce_explained"] = reduction.explained
out["reduce_scales"] = np.asarray(reduction.scales)
matrices = np.stack([sbci.load(p).dense(np.float64) for p in paths])
out["project_default"] = project(reduction, matrices)
print(f"reduce and project: {time.time() - t:.0f}s", flush=True)
print(
    "projected / fitted:",
    np.round(out["project_default"] / reduction.scores, 8).tolist(),
    flush=True,
)
del matrices

two = [sbci.load(p) for p in paths[:2]]
t = time.time()
alignment = sbci.align(two)
out["align_costs"] = np.concatenate(
    [np.ravel(np.asarray(c, dtype=np.float64)) for c in alignment.costs]
)
out["align_aligned0"] = np.asarray(alignment.aligned[0])
out["align_template"] = np.asarray(alignment.template)
out["align_lh_vertices0"] = np.asarray(alignment.warps[0].lh_vertices)
out["align_traces0"] = np.asarray(alignment.traces[0], dtype=np.float64)
print(f"align: {time.time() - t:.0f}s, costs {out['align_costs'].tolist()}", flush=True)

# The 2 October recording's settings, pinned: the defaults have been the paper's since item 19.
public = {
    "delta": PUBLIC_DELTA,
    "threshold": PUBLIC_THRESHOLD,
    "step_clamp": PUBLIC_STEP_CLAMP,
    "viscosity": PUBLIC_VISCOSITY,
    "sigma": PUBLIC_SIGMA,
    "relative_threshold": 0.0,
}
paper = {
    "delta": 0.1,
    "threshold": 1e-6,
    "step_clamp": float("inf"),
    "viscosity": 0.0,
    "sigma": PUBLIC_SIGMA,
    "relative_threshold": 0.0,
}
for tag, kwargs in (
    ("conseal", public),
    ("consealpaper", paper),
    ("consealnarrow", {**paper, "sigma": 0.0027, "relative_threshold": 1e-5}),
):
    t = time.time()
    registration = sbci.endpoints_align(two, max_iterations=3, **kwargs)
    out[f"{tag}_costs"] = np.concatenate(
        [np.ravel(np.asarray(c, dtype=np.float64)) for c in registration.costs]
    )
    ends = registration.aligned_endpoints(0)
    positions = endpoint_positions(ends)
    for k, part in enumerate(positions if isinstance(positions, tuple) else (positions,)):
        out[f"{tag}_positions_{k}"] = np.asarray(part)
    print(f"{tag}: {time.time() - t:.0f}s, costs {out[f'{tag}_costs'].tolist()}", flush=True)

np.savez(sys.argv[1], **out)
print("saved", sys.argv[1], {k: v.shape for k, v in out.items()})
