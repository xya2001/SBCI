"""Cortical surface plotting, five worked examples.

module load python/3.12.4
source /work/users/x/y/xya/sbci-venv/bin/activate
python ~/sbci/scripts/plot_examples.py
"""

# Agg must be selected BEFORE pyplot is imported: a compute node has no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbci import ContinuousConnectome, load_atlas

DATA = "/work/users/x/y/xya/sbci-derivatives"
OUT = "/work/users/x/y/xya/sbci-figures"

sc = ContinuousConnectome.load(f"{DATA}/sub-example_sc.h5")
fc = ContinuousConnectome.load(f"{DATA}/sub-example_fc.h5")


# --------------------------------------------------------------- 1. simplest
# A per-vertex map, one value for each of the 5124 vertices. Defaults to the
# inflated surface and a lateral + medial view of each hemisphere.
coupling = sc.coupling(fc, scope="global")

figure = sc.plot(coupling)
figure.savefig(f"{OUT}/ex1_default.png", dpi=110)
plt.close(figure)
print("1. sc.plot(map)                       -> ex1_default.png")


# ------------------------------------------------------- 2. choosing surfaces
for surface in ("inflated", "white", "sphere"):
    figure = sc.plot(coupling, surface=surface, title=f"coupling, {surface}")
    figure.savefig(f"{OUT}/ex2_{surface}.png", dpi=110)
    plt.close(figure)
print("2. surface='inflated'|'white'|'sphere' -> ex2_*.png")


# ------------------------------------------------------------- 3. more views
# Any of: lateral, medial, dorsal, ventral, anterior, posterior.
figure = sc.plot(
    coupling,
    views=("lateral", "medial", "dorsal"),
    cmap="RdBu_r",
    title="three views per hemisphere",
)
figure.savefig(f"{OUT}/ex3_views.png", dpi=110)
plt.close(figure)
print("3. views=('lateral','medial','dorsal') -> ex3_views.png")


# ------------------------------------------------ 4. a seed profile, log scale
# Raw density values are around 1e-10 because the file is normalized to unit
# mass over 13 million vertex pairs, so a linear scale shows almost nothing.
profile = sc.seed(vertex=1234)
# np.where would still evaluate log10 on the zeros and warn; the `where`
# argument skips them instead, leaving NaN.
log_profile = np.full_like(profile, np.nan)
np.log10(profile, out=log_profile, where=profile > 0)

figure = sc.plot(
    log_profile,
    cmap="inferno",
    title="seed profile at vertex 1234 (log10 density)",
)
figure.savefig(f"{OUT}/ex4_seed.png", dpi=110)
plt.close(figure)
print("4. log10 of a seed profile             -> ex4_seed.png")


# ---------------------------------------------------------- 5. an atlas, and
#                                                     thresholding a map
# Label 0 means "no region"; set it to NaN so the medial wall is not coloured.
atlas = load_atlas("Desikan")
labels = atlas.labels.astype(float)
labels[labels == 0] = np.nan

figure = sc.plot(labels, cmap="tab20", title=f"{atlas.name}, {atlas.n_regions} regions")
figure.savefig(f"{OUT}/ex5_atlas.png", dpi=110)
plt.close(figure)

# threshold hides values whose magnitude is below the cut, letting the
# underlying surface show through.
figure = sc.plot(
    coupling,
    threshold=0.3,
    cmap="coolwarm",
    title="coupling above 0.3 only",
)
figure.savefig(f"{OUT}/ex5_threshold.png", dpi=110)
plt.close(figure)
print("5. an atlas, and threshold=0.3         -> ex5_atlas.png, ex5_threshold.png")


# ------------------------------------------------------------ what you can set
print("\nArguments to sc.plot():")
print("  surface    'inflated' (default), 'white', 'sphere'")
print("  views      any of lateral, medial, dorsal, ventral, anterior, posterior")
print("  cmap       any matplotlib colormap")
print("  threshold  hide values with |value| below this")
print("  vmin/vmax  fix the colour range; default is the map's own range")
print("  symmetric  centre the scale on zero; defaults to True if the map has both signs")
print("  title      figure title")
print("\nNaN vertices are left uncoloured, which is how the medial wall is shown.")
print("plot() returns a matplotlib Figure, so savefig/close are yours to call.")
