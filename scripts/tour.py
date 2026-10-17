"""A worked tour of everything the package can currently do, on a released subject.

Run on Longleaf with the environment active:

    module load python/3.12.4
    source /work/users/x/y/xya/sbci-venv/bin/activate
    python ~/sbci/scripts/tour.py

The subject is sub-100307 of the HCP Young Adult example cohort, as
``sbci download hcp-ya`` writes it (D below is the lab's copy; point it at
yours), with its FC from ``tools/build_hcp_fc.py``; without an FC file the
steps that need one say so and are skipped.
"""

import os

import matplotlib

matplotlib.use("Agg")  # no display on a cluster node
import numpy as np

import sbci

D = "/work/users/x/y/xya/hcp-ya/data"
SUBJECT = "sub-100307"


def heading(text):
    print(f"\n{'=' * 4} {text} {'=' * (60 - len(text))}")


heading("1. LOADING")
from sbci import ContinuousConnectome

sc = ContinuousConnectome.load(f"{D}/{SUBJECT}_sc.h5")
print(f"  sc                {sc!r}")
print(f"  stored form       {sc.data.shape} float32, the strict upper triangle")
print(f"  area weights      {sc.area.shape}, sum {sc.area.sum():,.0f}")
print(f"  cortex mask       {int(sc.mask.sum()):,} of {sc.mask.size:,} vertices")
print(f"  modality          {sc.modality!r}")
print("  load() validates the metadata and refuses a file missing any key.")

heading("2. METADATA - what every file must declare")
for key in ("grid", "kernel", "bandwidth", "normalization", "streamline_count"):
    print(f"  {key:18s} {sc.metadata.get(key)!r}")

heading("3. ATLASES")
from sbci import list_atlases, load_atlas

print(f"  {len(list_atlases())} bundled, no download needed")
for name in ("Desikan", "Schaefer200", "Glasser"):
    atlas = load_atlas(name)
    print(
        f"  load_atlas({name!r}) -> {atlas.n_regions} regions, "
        f"covers {atlas.coverage:.1%} of the surface"
    )
print("  Short names resolve to the stored names, ignoring case and hyphens.")

heading("4. to_atlas - vertex matrix down to a region matrix")
atlas = load_atlas("Desikan")
mass = sc.to_atlas(atlas, how="mass")
mean = sc.to_atlas(atlas, how="mean")
print(f"  how='mass'  {mass.shape}  total {mass.sum():.6f}  (preserves connectivity)")
print(f"  how='mean'  {mean.shape}  range [{mean.min():.3g}, {mean.max():.3g}]  (a density)")
i, j = np.unravel_index(np.argmax(np.triu(mass, 1)), mass.shape)
print(f"  strongest pair: {atlas.names[i]} <-> {atlas.names[j]}")
fc_path = f"{D}/{SUBJECT}_fc.h5"
fc = sbci.load(fc_path) if os.path.exists(fc_path) else None
if fc is not None:
    region_fc = fc.to_atlas(atlas)
    pairs = region_fc[np.triu_indices(68, 1)]
    print(f"  FC {region_fc.shape}, through Fisher-z: mean {pairs.mean():.3f}")
else:
    print(f"  FC is aggregated through Fisher-z automatically (no {SUBJECT}_fc.h5 here).")

heading("5. seed - one vertex's or one region's connectivity profile")
profile = sc.seed(vertex=1234)
print(f"  sc.seed(vertex=1234)  -> {profile.shape}, {int((profile > 0).sum()):,} non-zero targets")
region = atlas.labels == atlas.region_ids[10]
marginal = sc.seed(region=region)
print(
    f"  sc.seed(region=mask)  -> {marginal.shape}, region {atlas.names[10]!r} "
    f"({int(region.sum())} vertices)"
)
print("  region= takes a boolean mask and returns the area-weighted marginal.")

heading("6. coupling - structure against function")
if fc is not None:
    whole = sc.coupling(fc, scope="global")
    local = sc.coupling(fc, scope="region", labels=atlas.labels)
    print(f"  sc.coupling(fc, scope='global')  -> {whole.shape}, mean {np.nanmean(whole):.3f}")
    print(f"  sc.coupling(fc, scope='region')  -> mean {np.nanmean(local):.3f} within regions")
    print(f"  NaN on the {int(np.isnan(whole).sum())} medial-wall vertices.")
else:
    print("  sc.coupling(fc, scope='global'|'region') needs an FC file on the same grid;")
    print(f"  there is no {SUBJECT}_fc.h5 here, so this step is skipped.")

heading("7. plot - a surface figure")
from sbci import load_surface

surface = load_surface("sphere")
print(f"  load_surface('sphere') -> {surface.n_vertices} vertices, {len(surface.faces):,} faces")
figure = sc.plot(profile, title="seed profile of vertex 1234")
out = "/work/users/x/y/xya/sbci-figures/tour_seed.png"
figure.savefig(out, dpi=100)
matplotlib.pyplot.close(figure)
print(f"  sc.plot(map) -> matplotlib figure, saved to {out}")
print("  Inflated, white, pial and sphere are bundled, in the grid's vertex order (Q11).")

heading("8. save and validate")
import subprocess
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    path = f"{tmp}/sub-copy_sc.h5"
    sc.save(path)
    print(f"  sc.save(path) wrote {path}")
    result = subprocess.run(["sbci", "validate", path], capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")
    print(f"  exit status {result.returncode} (0 means every check passed)")

heading("9. WHAT RAISES, AND WHAT IT TELLS YOU")
# Everything in the API table is implemented, so these are genuine misuses.
# (to_cifti and reduce used to sit here as "not yet"; both now run, and both
# are batch jobs on the full grid, so they are deliberately not called.)
for label, call in (
    ("load_surface('nonesuch')", lambda: load_surface("nonesuch")),
    ("sc.seed()", lambda: sc.seed()),
    ("sc.coupling(sc)", lambda: sc.coupling(sc)),
    ("sc.smooth(kernel='nonesuch')", lambda: sc.smooth(kernel="nonesuch")),
):
    try:
        call()
    except (sbci.SbciError, ValueError) as exc:
        first = str(exc).split(".")[0]
        print(f"  {label:28s} {type(exc).__name__}")
        print(f"  {'':28s} {first[:78]}...")

if sc.has_endpoints:
    # The released file carries the endpoints it was smoothed from, so the
    # default kernel gives back the connectome we loaded.
    again = sc.smooth(kernel="shk", mask_medial_wall=True)
    gap = np.abs(again.data - sc.data).max()
    print("\n  sc.smooth(kernel='shk', mask_medial_wall=True) reproduces the file:")
    print(f"  {sc.endpoints.n_streamlines:,} endpoints, max |difference| {gap:.3g}")

heading("10. THE KERNEL MATHS, USABLE DIRECTLY")
from sbci.smoothing import kappa_candidates, load_eigenpairs

REF = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/concon_estimate"
lam, U = load_eigenpairs(f"{REF}/EV_LBO_ds_ico4_L.mat", "L")
print(f"  load_eigenpairs(...)  -> {lam.size} eigenvalues, U {U.shape}")
print(f"  kappa_candidates(lam) -> {np.array2string(kappa_candidates(lam)[:4], precision=3)} ...")
print("  diffusion_kernel / matern_kernel / smooth_endpoints work directly;")
print("  cc.smooth() runs on any file that carries endpoints (kernel='shk' by default).")
