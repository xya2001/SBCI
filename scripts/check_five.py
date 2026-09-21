"""Exercise the five working entry points and print what each returns.

module load python/3.12.4
source /work/users/x/y/xya/sbci-venv/bin/activate
python ~/sbci/scripts/check_five.py
"""

import os
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")  # no display on a cluster node
import matplotlib.pyplot as plt
import numpy as np

from sbci import ContinuousConnectome, load_atlas
from sbci.metadata import MetadataError, template

DATA = "/work/users/x/y/xya/sbci-derivatives"
FIGURES = "/work/users/x/y/xya/sbci-figures"

sc = ContinuousConnectome.load(f"{DATA}/sub-example_sc.h5")
fc = ContinuousConnectome.load(f"{DATA}/sub-example_fc.h5")
atlas = load_atlas("Desikan")


def heading(text):
    print(f"\n{'-' * 66}\n{text}\n{'-' * 66}")


# ---------------------------------------------------------------- 1. seed
heading("1.  sc.seed(vertex=1234)")
profile = sc.seed(vertex=1234)
print(f"  type            {type(profile).__name__}, dtype {profile.dtype}")
print(f"  shape           {profile.shape}")
print(f"  non-zero        {int((profile > 0).sum()):,} of {profile.size:,} vertices")
print(f"  range           [{profile.min():.4g}, {profile.max():.4g}]")
print(f"  strongest target vertex {int(np.argmax(profile))}")
print(f"  self-connection profile[1234] = {profile[1234]:.4g}  (diagonal is excluded)")

print("\n  sc.seed(region=mask)  -- a whole region's marginal")
mask = atlas.labels == atlas.region_ids[10]
marginal = sc.seed(region=mask)
print(f"  region          {atlas.names[10]!r}, {int(mask.sum())} vertices")
print(f"  shape           {marginal.shape}")
print(f"  range           [{marginal.min():.4g}, {marginal.max():.4g}]")
print("  region= takes a BOOLEAN MASK over vertices, not a region id.")

# ------------------------------------------------------------ 2. coupling
heading("2.  sc.coupling(fc, scope='global')")
coupling = sc.coupling(fc, scope="global")
finite = np.isfinite(coupling)
print(f"  shape           {coupling.shape}")
print(f"  finite          {int(finite.sum()):,}   NaN {int((~finite).sum()):,}")
print(f"  range           [{coupling[finite].min():.4f}, {coupling[finite].max():.4f}]")
print(f"  mean            {coupling[finite].mean():.4f}")
print(f"  negative        {int((coupling[finite] < 0).sum()):,} vertices")
print(f"  NaN is the medial wall: mask says {int((~sc.mask).sum()):,} non-cortex vertices")

# ---------------------------------------------------------------- 3. plot
heading("3.  sc.plot(map)")
os.makedirs(FIGURES, exist_ok=True)
figure = sc.plot(coupling, title="SC-FC coupling (global)", cmap="coolwarm")
out = f"{FIGURES}/check_five_coupling.png"
figure.savefig(out, dpi=110)
print(f"  returns         {type(figure).__name__}")
print(f"  panels          {len(figure.axes)}  (2 hemispheres x 2 views, plus colourbars)")
print(f"  saved           {out}  ({os.path.getsize(out) / 1024:.0f} KB)")
plt.close(figure)
print("  matplotlib.use('Agg') must come BEFORE importing pyplot on a node.")

# ---------------------------------------------------------------- 4. save
heading("4.  sc.save(path)  -- and the claim that it validates first")
with tempfile.TemporaryDirectory() as tmp:
    good = f"{tmp}/sub-copy_sc.h5"
    returned = sc.save(good)
    print(f"  returns         {returned}")
    print(f"  wrote           {os.path.getsize(good) / 1e6:.1f} MB")

    print("\n  now the same write with deliberately incomplete metadata:")
    bad = f"{tmp}/sub-broken_sc.h5"
    from sbci.io import write_hdf5

    try:
        write_hdf5(
            bad, data=sc.data, area=sc.area, mask=sc.mask, metadata=template("sc")
        )  # acquisition keys left as None
        print("  UNEXPECTED: the write succeeded")
    except MetadataError as exc:
        print("  raised          MetadataError")
        print(f"                  {str(exc)[:70]}...")
    print(f"  file created?   {os.path.exists(bad)}   <- no partial file left behind")

    # ------------------------------------------------------- 5. validate
    heading("5.  sbci validate <file>")
    result = subprocess.run(["sbci", "validate", good], capture_output=True, text=True)
    for check in result.stdout.strip().split("\n"):
        print(f"  {check}")
    print(f"\n  exit status     {result.returncode}   (0 = every check passed)")

    print("\n  and on a file that is wrong -- connectivity doubled, so the")
    print("  density no longer integrates to one:")
    doubled = f"{tmp}/sub-doubled_sc.h5"
    write_hdf5(doubled, data=sc.data * 2, area=sc.area, mask=sc.mask, metadata=sc.metadata)
    result = subprocess.run(["sbci", "validate", doubled], capture_output=True, text=True)
    for check in result.stdout.strip().split("\n"):
        marker = "  <-- caught" if "FAIL" in check else ""
        print(f"  {check}{marker}")
    print(f"\n  exit status     {result.returncode}   (non-zero = something failed)")
