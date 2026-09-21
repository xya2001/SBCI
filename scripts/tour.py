"""A worked tour of everything the package can currently do.

Run on Longleaf with the environment active:

    module load python/3.12.4
    source /work/users/x/y/xya/sbci-venv/bin/activate
    python ~/sbci/scripts/tour.py
"""

import matplotlib

matplotlib.use("Agg")  # no display on a cluster node
import numpy as np

D = "/work/users/x/y/xya/sbci-derivatives"


def heading(text):
    print(f"\n{'=' * 4} {text} {'=' * (60 - len(text))}")


heading("1. LOADING")
from sbci import ContinuousConnectome

sc = ContinuousConnectome.load(f"{D}/sub-example_sc.h5")
fc = ContinuousConnectome.load(f"{D}/sub-example_fc.h5")
print(f"  sc                {sc!r}")
print(f"  fc                {fc!r}")
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
print("  FC is aggregated through Fisher-z automatically:")
print(
    f"  fc.to_atlas(...) range [{fc.to_atlas(atlas, how='mean').min():.3f}, "
    f"{fc.to_atlas(atlas, how='mean').max():.3f}]"
)

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
glb = sc.coupling(fc, scope="global")
finite = np.isfinite(glb)
print(
    f"  scope='global'   {glb.shape}, {int(finite.sum()):,} finite "
    f"(NaN = medial wall or a constant profile)"
)
print(
    f"                   range [{glb[finite].min():.3f}, {glb[finite].max():.3f}], "
    f"mean {glb[finite].mean():.3f}"
)
loc = sc.coupling(fc, scope="region", labels=atlas.labels)
ok = np.isfinite(loc)
print(
    f"  scope='region'   needs labels=; mean {loc[ok].mean():.3f} (higher: neighbours share both)"
)
from sbci.coupling import discrete_coupling

dct = discrete_coupling(sc.to_atlas(atlas, "mean"), fc.to_atlas(atlas, "mean"))
print(f"  discrete_coupling on atlas matrices -> {dct.shape}, mean {np.nanmean(dct):.3f}")
print("  global and region are cosine similarity; discrete is Pearson.")

heading("7. plot - a surface figure")
from sbci import load_surface

surface = load_surface("sphere")
print(f"  load_surface('sphere') -> {surface.n_vertices} vertices, {len(surface.faces):,} faces")
figure = sc.plot(glb, title="SC-FC coupling", cmap="coolwarm")
out = "/work/users/x/y/xya/sbci-figures/tour_coupling.png"
figure.savefig(out, dpi=100)
matplotlib.pyplot.close(figure)
print(f"  sc.plot(map) -> matplotlib figure, saved to {out}")
print("  Only the sphere is bundled; inflated and white are withheld (Q11).")

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
for label, call in (
    ("sc.to_cifti(...)", lambda: sc.to_cifti("/tmp/x.dconn.nii")),
    ("sc.smooth()", lambda: sc.smooth(kernel="rdk")),
    ("sc.reduce(rank=5)", lambda: sc.reduce(rank=5)),
    ("load_surface('inflated')", lambda: load_surface("inflated")),
):
    try:
        call()
    except (NotImplementedError, ValueError) as exc:
        first = str(exc).split(".")[0]
        print(f"  {label:26s} {type(exc).__name__}")
        print(f"  {'':26s} {first[:78]}...")

heading("10. THE KERNEL MATHS, USABLE DIRECTLY")
from sbci.smoothing import kappa_candidates, load_eigenpairs

REF = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/concon_estimate"
lam, U = load_eigenpairs(f"{REF}/EV_LBO_ds_ico4_L.mat", "L")
print(f"  load_eigenpairs(...)  -> {lam.size} eigenvalues, U {U.shape}")
print(f"  kappa_candidates(lam) -> {np.array2string(kappa_candidates(lam)[:4], precision=3)} ...")
print("  diffusion_kernel / matern_kernel / smooth_endpoints work today;")
print("  only cc.smooth() is blocked, because the format stores no endpoints.")
