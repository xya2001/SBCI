"""The eleventh list's measurements, after its fixes (PORTING.md item 18).

    SBCI_HCP_DIR=hcp-ya python tests/reference/eleventh_list_probe.py

Needs the eleven released subjects (``sbci download hcp-ya``) for the coupling, and a
checkout with history for the fsaverage file as it was (``git show 8611f0e:...``).
"""

import io
import os
import subprocess
from pathlib import Path

import numpy as np

import sbci
from sbci.alignment import pole_rotation
from sbci.atlas import cortex_mask, load_atlas
from sbci.conseal import default_grids
from sbci.coupling import discrete_coupling, global_coupling
from sbci.io import cifti
from sbci.smoothing import Endpoints
from sbci.stats import local_test

D = Path(os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data"))


def degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


print("== 1. permutation ties on discrete scores (the test's data; exact p 0.058)")
y = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1, 1, 2, 2, 0, 2, 1, 0, 2], dtype=float)
group = np.repeat([0.0, 1.0], 10)
for scale in (1.0, 7.3, 0.01, 1e5):
    p = local_test((y * scale)[:, None], group, permutations=499, seed=3).pvalue[0]
    print(f"  scores times {scale:g}: p = {p:.3f}")

print("== 2. FC exchange: fsLR vertices beside the medial wall")
mask = np.asarray(cortex_mask(), dtype=bool)
overlap = cifti.load_overlap()
total = np.asarray(overlap.sum(axis=1)).ravel()
wall = np.asarray(overlap[:, ~mask].sum(axis=1)).ravel()
share = np.divide(wall, total, out=np.zeros_like(wall), where=total > 0)
straddling = (share > 0) & (share < 1)
print(
    f"  {straddling.sum()} straddle the wall; the whole operator scaled their correlations by "
    f"{1 - share[straddling].max():.3f} to {1 - share[straddling].min():.3f}: halved at "
    f"{int(np.isclose(share[straddling], 0.5).sum())}, cut by more than half at "
    f"{int((share[straddling] > 0.5 + 1e-12).sum())}"
)

print("== 3. atlas-level coupling on the released subjects: SC as mass against its mean density")
desikan = load_atlas("Desikan")
for how in ("mass", "mean"):
    means = []
    for path in sorted(D.glob("sub-*_sc.h5")):
        sc, fc = sbci.load(path), sbci.load(str(path).replace("_sc.h5", "_fc.h5"))
        regions = discrete_coupling(sc.to_atlas(desikan, how=how), fc.to_atlas(desikan))
        means.append(np.nanmean(regions))
    print(
        f"  SC as {how}: per-subject means {min(means):.3f} to {max(means):.3f}, "
        f"overall {np.mean(means):.3f} ({len(means)} subjects)"
    )

print("== 4. ConSEAL area weighting: raw areas against the plain sum, on ico4")
lh, rh = default_grids()
areas = np.concatenate([lh.areas, rh.areas])
print(
    f"  mean vertex area {areas.mean():.6f}: raw weights scaled the cost by about "
    f"{areas.mean() ** 2:.2e} (1/{1 / areas.mean() ** 2:,.0f})"
)

print("== 6. the fsaverage sphere file: before (8611f0e) and now")
old_bytes = subprocess.run(
    ["git", "show", "8611f0e:src/sbci/data/surfaces/fsaverage_sphere_ico4.npz"],
    capture_output=True,
    check=True,
).stdout
old = np.load(io.BytesIO(old_bytes))
new = np.load(Path(sbci.__file__).parent / "data" / "surfaces" / "fsaverage_sphere_ico4.npz")
moved = np.degrees(np.arccos(np.clip((old["vertices"] * new["vertices"]).sum(1), -1, 1)))
changed = old["fsaverage_index"] != new["fsaverage_index"]
print(
    f"  {changed.sum()} vertices took another fsaverage vertex ({changed[:2562].sum()} left), "
    f"{moved[changed].min():.3f} to {moved[changed].max():.3f} degrees off; the rest within "
    f"{moved[~changed].max():.1e}"
)

print("== 7. an index past its hemisphere, through the whole-grid encoding")
flipped = Endpoints.from_global(np.array([2600]), np.array([5]))
print(
    f"  left vertex 2600 stored as 2600 reads back as hemisphere {int(flipped.surf_in[0])}, "
    f"vertex {int(flipped.vtx_in[0])}; Endpoints now refuses it"
)

print("== a. the turn align's old advice applied to the bundled grid and nothing recorded")
for name, grid in zip(("left", "right"), default_grids(), strict=True):
    print(f"  {name}: pole_rotation turns it {degrees(pole_rotation(grid.vertices)):.1f} degrees")

print("== c. Schaefer-1000's one-vertex regions and global coupling (sub-100307, how='mean')")
schaefer = load_atlas("Schaefer1000")
sizes = np.bincount(schaefer.labels[mask], minlength=schaefer.labels.max() + 1)[1:]
sc, fc = sbci.load(D / "sub-100307_sc.h5"), sbci.load(D / "sub-100307_fc.h5")
coupling = global_coupling(sc.to_atlas(schaefer, how="mean"), fc.to_atlas(schaefer, how="mean"))
single = np.flatnonzero(sizes == 1)
print(
    f"  {single.size} regions with one cortical vertex; NaN among them now: "
    f"{int(np.isnan(coupling[single]).sum())}"
)
