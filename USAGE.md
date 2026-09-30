# Using the package

Everything below is live: `scripts/tour.py` runs all of it and prints the
output quoted here.

## Setup anywhere

```bash
pip install "sbci[plotting] @ git+https://github.com/xya2001/SBCI.git"
sbci download hcp-ya --subject 100307     # one real subject, 50 MB, into the current directory
```

Python 3.10 or newer; the plotting extra is only for figures. The examples
below that need a released subject run on the HCP Young Adult cohort:
`sbci download hcp-ya --out hcp-ya` fetches all eleven (about 560 MB) and
`--subject` one of them. They are structural only, so the examples with FC use the lab's files
on the Longleaf cluster, whose paths are quoted; substitute yours. The `rdk` and `matern` kernels need the
Laplace-Beltrami basis, two files from `SBCI_Toolkit/concon_estimate`; *Storing
endpoints, and re-smoothing from them* says where to put them.

## Setup on Longleaf

**Once:**

```bash
cd ~/sbci && ./scripts/setup_longleaf.sh
```

That loads the Python module, creates the virtualenv on `/work` and installs
the package in editable mode with its development extras.

**Every session:**

```bash
module load python/3.12.4
source /work/users/x/y/xya/sbci-venv/bin/activate
```

**Load the module before activating, every time.** The virtualenv's interpreter
is linked against the module's `libpython3.12.so.1.0`, which is not on the
default library path. Activating without loading gives

```
python: error while loading shared libraries: libpython3.12.so.1.0
```

That is a missing module, not a broken environment. `module load` fixes it
without recreating anything.

`/work` is scratch and is purged, so the environment is disposable by design —
`scripts/setup_longleaf.sh` rebuilds it from nothing. Only the repository is
permanent.

## Data to try it on

The released cohort is described under *Getting the example cohort* below.
On Longleaf, one real subject is already imported, converted
from legacy pipeline output:

```
/work/users/x/y/xya/sbci-derivatives/sub-example_sc.h5
/work/users/x/y/xya/sbci-derivatives/sub-example_fc.h5
```

Both pass every validator check. To convert more, use
`tools/import_legacy.py`.

> Every path below is literal and copy-pasteable. If you see a
> `FileNotFoundError` mentioning `...`, an ellipsis placeholder was pasted as a
> real path.

## Getting the example cohort

Eleven HCP Young Adult subjects, each as an SC file that carries its
streamline endpoints: sub-100307, the subject the package brief's acceptance
test names, and ten drawn at random, five women and five men, from the 946
with complete pipeline output in the lab's copy. They were rebuilt on the ico4 grid from
the SBCI pipeline's output by `tools/build_hcp_cohort.py --layout
young-adult` and are hosted on a public Google Drive ([the
folder](https://drive.google.com/drive/folders/1gG2ZmxxVm4w5dvlCQvMaEEOBypU7nDpx)).
The lab's copy of the cohort has no resting-state data, so there is no FC yet.
The package ships the manifest, not the data:

```bash
sbci download hcp-ya --out hcp-ya                    # the eleven SC files into ./hcp-ya, about 560 MB
sbci download hcp-ya --subject 100307                # one file, sub-100307_sc.h5, into the current directory
```

```python
from sbci.download import fetch_cohort, load_manifest
paths = fetch_cohort("hcp-ya")                       # all eleven into ./hcp-ya, from Python
load_manifest()["subjects"][0]                       # subject, sex, age_bin, files
```

The data are the WU-Minn Human Connectome Project's, redistributed under its
[Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
Downloading them means accepting those terms, and every download writes
`DATA_USE.txt` beside the files with the acknowledgment a publication must
carry. The manifest gives each subject's sex and the HCP's open-access age
band (22-25, 26-30, 31-35 or 36+), never an exact age.

Every file is checked against its SHA-256 from the manifest; a file that
fails is removed and the error says so. A re-run verifies and skips what is
present, so the command is safe to repeat. Asking only for FC, which this
cohort does not hold, says so instead of fetching nothing. Google Drive
answers with a page instead of a file when the file is not shared with anyone
who has the link; the error names the file id.

A cohort too large for one file per subject can travel as zip bundles on
Zenodo, which `fetch_cohort` reads as well, fetching only the bundles the
requested subjects need and resuming an interrupted one
(`tools/bundle_hcp_cohort.py` writes them). No cohort is released that way at
present.

## Loading

```python
from sbci import ContinuousConnectome

sc = ContinuousConnectome.load("/work/users/x/y/xya/sbci-derivatives/sub-example_sc.h5")
fc = ContinuousConnectome.load("/work/users/x/y/xya/sbci-derivatives/sub-example_fc.h5")
```

```
sc                ContinuousConnectome(modality='sc', n_vertices=5124)
stored form       (13125126,) float32, the strict upper triangle
area weights      (5124,), sum 327,684
cortex mask       4,685 of 5,124 vertices
```

Connectivity is held as the strict upper triangle in float32 — the form it
takes on disk — and expanded on demand with `.dense()`, which allocates about
105 MB. `load()` validates the metadata and **refuses a file missing any
required key**, rather than defaulting, because a silently defaulted bandwidth
makes two incomparable files look comparable.

Useful attributes: `.data`, `.area`, `.mask`, `.metadata`, `.modality`,
`.n_vertices`.

## Atlases

```python
from sbci import list_atlases, load_atlas

load_atlas("Desikan").n_regions        # 68
load_atlas("Schaefer200").n_regions    # 200
load_atlas("Glasser").n_regions        # 360
```

44 atlases ship inside the package (332 KB), so this needs no download, no
FreeSurfer and no MATLAB. Short names resolve to the stored names ignoring
case, spaces, hyphens and underscores. `list_atlases()` gives the full set,
which also includes Gordon, Yeo, the PALS-B12 family and CoCoNest at 22 scales.

Label `0` means "no region" — the medial wall, plus anything outside a
partial-coverage atlas. Regions are numbered `1..K` with no gaps, and
`atlas.names[i]` names label `i + 1`.

## `to_atlas` — vertex matrix to region matrix

```python
mass = sc.to_atlas(atlas, how="mass")   # (68, 68), sums to 1.000000
mean = sc.to_atlas(atlas, how="mean")   # (68, 68), a density
```

`"mass"` sums the area-weighted connectivity crossing each region pair, so the
total is preserved. `"mean"` divides that by the product of the two regions'
areas, giving a density comparable across regions of different size; this
reproduces `parcellate_sc.m`. FC is aggregated through Fisher-z automatically,
because averaging correlations directly is biased.

![The Desikan region matrix of an HCP Young Adult subject](docs/figures/region_matrix.png)

*`cc.to_atlas("Desikan")` on sub-103010: 68 regions, left hemisphere first,
mass on a log scale.*

## `seed` — one profile

```python
sc.seed(vertex=1234)      # (5124,) — that vertex's density slice
sc.seed(region=mask)      # (5124,) — a region's area-weighted marginal
```

`region=` takes a boolean mask over vertices, so
`atlas.labels == atlas.region_ids[10]` selects one region. The vertex form
reads its row without materializing the dense matrix.

## `coupling` — structure against function

```python
sc.coupling(fc, scope="global")                        # (5124,)
sc.coupling(fc, scope="region", labels=atlas.labels)   # (5124,)
```

```
global    4,683 finite, range [-0.106, 0.533], mean 0.248
region    mean 0.671
```

NaN marks the medial wall and any vertex whose profile is constant. Local
coupling runs higher than global because neighbouring vertices inside a region
share both structure and function.

Two things worth knowing: **negative FC values are kept** — discarding them
flips the sign of the map in association cortex — and **`discrete_coupling` is
a Pearson correlation, not a cosine**, matching MATLAB's `corr2`, while global
and region are uncentred cosine similarity.

![Structure-function coupling on the surface](docs/figures/coupling.png)

*`sc.coupling(fc)` for an HCP-Aging subject's SC and FC, since the young adult
files have no FC yet and the HCP-Aging files are not distributed: one cosine
similarity per vertex, drawn on the inflated surface.*

## `plot` — a surface figure

```python
import matplotlib
matplotlib.use("Agg")          # no display on a cluster node

figure = sc.plot(coupling_map, title="SC-FC coupling", cmap="coolwarm")
figure.savefig("coupling.png", dpi=150)
```

Four panels, each hemisphere seen laterally and medially, on a surface shaded
by sulcal depth so the folds show through the map, with the medial wall left
flat since it is a cut surface and not cortex (`shading=False` draws the bare
mesh; `threshold=` hides the small values so the shading shows there).
Needs the plotting extra, which `setup_longleaf.sh` installs. Four geometries
are bundled -- inflated (the default), white, pial and sphere -- all in the
grid's vertex order and sharing one face list (`SPEC_QUESTIONS.md` items 11
and 13).

The grid itself has 5124 vertices and draws faceted. For a figure worth
showing, pass `mesh="fsaverage"`: the map is interpolated onto FreeSurfer's
fsaverage surface (163,842 vertices per hemisphere), shaded by FreeSurfer's
own sulcal depth, with the medial wall still flat. The values stay on ico4;
only the drawing is finer. nilearn fetches fsaverage once into
`~/nilearn_data` (do that on a login node if compute nodes have no network);
`mesh="fsaverage5"` (10,242 per hemisphere) ships inside nilearn and needs no
download. The finest mesh takes about a minute per view to render.

```python
figure = sc.plot(coupling_map, mesh="fsaverage", cmap="coolwarm")
figure.savefig("coupling.png", dpi=200)
```

For a figure to publish, add `engine="pyvista"`: each view is rendered
off-screen through PyVista (VTK) with smooth per-vertex normals, a
three-point light kit and a specular highlight, then laid out in the same
figure with the same colorbar. It needs the `render` extra
(`pip install 'sbci[render]'`). Four fsaverage views take about ten seconds
on a laptop and about a minute on a cluster node without a display, where
VTK 9.4 or later renders in software by itself (an older VTK there needs its
OSMesa build: `pip install --extra-index-url https://wheels.vtk.org
vtk-osmesa`).

```python
figure = sc.plot(coupling_map, mesh="fsaverage", engine="pyvista", cmap="coolwarm")
```

## `save` and `sbci validate`

```python
sc.save("sub-copy_sc.h5")
```

```bash
sbci validate sub-copy_sc.h5
```

```
[PASS] readable    [PASS] metadata     [PASS] grid    [PASS] symmetry
[PASS] nonnegativity    [PASS] unit mass    [PASS] mask
```

Exit status 0 means every check passed. `save()` validates before writing, so a
run that forgot to record its bandwidth fails without leaving a partial file.

## The kernel maths, usable directly

`cc.smooth()` is the high-level route; the functions underneath are usable
directly:

```python
from sbci.smoothing import (
    Endpoints, diffusion_kernel, kappa_candidates, load_eigenpairs, smooth_endpoints,
)

lam, U = load_eigenpairs("/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/concon_estimate/EV_LBO_ds_ico4_L.mat", "L")
kappa  = kappa_candidates(lam)[3]
K_L    = diffusion_kernel(lam, U, kappa)
lam_R, U_R = load_eigenpairs("/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/concon_estimate/EV_LBO_ds_ico4_R.mat", "R")
K_R    = diffusion_kernel(lam_R, U_R, kappa)
density = smooth_endpoints(Endpoints.from_matlab(
    "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/example_data/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat"), K_L, K_R)
```

Verified against a MATLAB reference run to single-precision rounding, which is
all the reference itself carries. Takes 4.4 s for both hemispheres.

![The spherical kernel at two bandwidths](docs/figures/spherical_kernel.png)

*The kernel `smooth(kernel="shk")` applies, relative to its peak, at the
released bandwidth and at twice it, with the cutoff beyond which it is zero.*

## Storing endpoints, and re-smoothing from them

A structural file can carry the streamline endpoints it was built from, in an
optional `/endpoints` group. Without them a connectome is a finished product;
with them it can be re-smoothed at another bandwidth or with another kernel.
The released subjects carry theirs (803,741 for `sub-100307`), so
`sbci.load("sub-100307_sc.h5").smooth(kernel="shk", mask_medial_wall=True)`
re-smooths a real subject and gives back the stored connectome at
r = 1.0000000; the rest of this section is about the lab's files.

```python
from sbci import ContinuousConnectome
from sbci.smoothing import Endpoints

TOOLKIT = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit"

sc = ContinuousConnectome.load("/work/users/x/y/xya/sbci-derivatives/sub-example_sc.h5")
sc.endpoints = Endpoints.from_matlab(
    f"{TOOLKIT}/example_data/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat"
)
sc.save("/work/users/x/y/xya/sub-example_desc-withendpoints_sc.h5")
```

**Which endpoints.** The pipeline writes two sets, and only one belongs with
its smoothed connectome. `mesh_intersections_ico4.mat` holds the unsnapped
intersections mapped onto the grid; the connectome was smoothed from the
snapped streamlines placed on the subject's registered sphere. Build those
with `Endpoints.from_snapped`, which is what `tools/build_hcp_cohort.py`
does:

```python
from sbci.smoothing import Endpoints
# ?h_sphere_reg_lps.vtk: the subject's registered spheres, one vertex per native
# vertex, already in the grid's frame (a sphere stored in RAS needs x and y negated)
ends = Endpoints.from_snapped("snapped_fibers.npz", lh_sphere_vertices, rh_sphere_vertices)
```

On an HCP-Aging subject these re-smooth to the pipeline's stored connectome at
r = 0.9997; the endpoints from `mesh_intersections_ico4.mat` reach 0.968
(PORTING.md item 6). The young adult files were smoothed by the package from
the snapped endpoints themselves, which is why they re-smooth exactly. The
example above uses the other file only because the toolkit's demo subject has
nothing else.

On the example subject that is 383,760 streamlines and takes the file from
**20.0 MB to 31.8 MB**. The group is optional, so a file without it is still
valid -- it simply raises when you try to re-smooth.

```python
back = ContinuousConnectome.load("/work/users/x/y/xya/sub-example_desc-withendpoints_sc.h5")
back.has_endpoints                      # True
back.endpoints.n_streamlines            # 383760
back.endpoints.has_positions            # True: barycentric positions too

resmoothed = back.smooth(kernel="rdk", eigenpairs=f"{TOOLKIT}/concon_estimate")
resmoothed.metadata["kernel"]           # 'rdk'
resmoothed.metadata["bandwidth"]        # 1.948262, from the reference formula
```

Six seconds on a login node. The result is a new `ContinuousConnectome`
normalized to unit mass, carrying the same endpoints, so it can be re-smoothed
again at a different bandwidth.

**The Laplace-Beltrami basis** is `EV_LBO_ds_ico4_L.mat` and
`EV_LBO_ds_ico4_R.mat`. Pass the directory as `eigenpairs=`, or set it once and
forget it:

```bash
export SBCI_LBO_DIR=/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/concon_estimate
```

`.smooth()` also looks in `$SBCI_TOOLKIT/concon_estimate`, `~/.cache/sbci/lbo`
and `~/.sbci/lbo`, and names every one of them if it finds nothing. The files
are 50 MB per hemisphere and describe the grid rather than the subject, so they
are neither bundled in the wheel nor duplicated into every file. Recomputing
them from the bundled white surface was tried and is not good enough -- it
reproduces the reference kernel only to r = 0.94.

**Is it correct?** `smooth(kernel="rdk")` reproduces MATLAB's
`reference_density.mat` to **3.25 float32-eps**, correlation
0.999999999999921, and the bandwidth you get by passing nothing is the one the
reference run used to within 5e-08. Both are asserted in
`tests/test_matlab_reference.py`.

`kernel="shk"`, the default, reproduces `c3_main` at **r = 1.000000** across
five ADNI subjects, with the scale factor at 1.000000 and nothing fitted
(0.99858 with the closed-form series, `quantized=False`). The kernel is not
the heat kernel its name suggests: `concon` compounds a `(2l+1)` weight with a
normalized spherical harmonic, giving `(2l+1)^(3/2)`, and it has compact
support at about `2.9*sqrt(sigma)` radians. The 0.14% amplitude offset the
closed form carried was the binary's lookup-table quantization, and the tables
are now reproduced; what remains is 0.071% more non-zero pairs, where one ULP
in a dot product moves a vertex across the kernel cutoff (PORTING.md item 6). A full subject takes about a
minute. See PORTING.md item 6.

A re-smoothed density can carry mass on the medial wall, exactly as both
references do, so `sbci validate` may flag a file saved straight from it --
SPEC_QUESTIONS.md item 14. Pass `mask_medial_wall=True` to zero it first --
a file saved that way passes all nine `sbci validate` checks, verified end to
end on a real subject -- and `progress=lambda done, total: ...` to watch a run.

The kernel is evaluated the way `c3_main` evaluates it: through two
truncation-indexed lookup tables (10^5 harmonic samples, 10^6 kernel samples),
which is what removed the last 0.14% amplitude offset against the released
files. `spherical_heat_kernel(..., quantized=False)` gives the closed-form
series the tables approximate. Only the ~31 vertices inside the 12.2-degree
cutoff are visited per endpoint, found with a KD-tree.

![Smoothing, from endpoints to a comparable map](docs/figures/smoothing_power.png)

*Why the density and not the counts: the far ends of the streamlines touching
vertex 1234 (blue dots) in two random halves of sub-103010's 988,788
streamlines and in the whole set, and the smoothed density of the same vertex
from each. Only 41 and 28 streamlines touch the vertex in the two halves, so
their raw counts agree across the cortex at r = 0.46; smoothed they agree at
r = 1.00, and half A matches the map from all streamlines at r = 1.00.*

## The cohort, end to end

The example subjects show the alignments. The question of an association
goes to all 946 HCP Young Adult subjects with complete pipeline output, which
are not distributed; `tools/build_hcp_cohort.py --layout young-adult` builds
them from the pipeline's output for anyone with HCP access. Sex, the age band
and fluid intelligence come from the HCP's open-access table, and the age
band enters as its midpoint.

```python
import csv
from pathlib import Path
import numpy as np
import sbci
from sbci.download import fetch_cohort

paths = fetch_cohort(out="hcp-ya")                         # the eleven, 560 MB
subjects = [sbci.load(p) for p in paths]

# ENCORE on the densities: five minutes on eight cores for ten subjects
alignment = sbci.align(subjects, max_iterations=10)
alignment.aligned[0]                                       # subject 1's warped density, dense
# ConSEAL on the endpoints (an hour; see the ConSEAL notes on its template)
registration = sbci.endpoints_align(subjects, max_iterations=30)
for i, cc in enumerate(subjects):
    cc.endpoints = registration.aligned_endpoints(i)
aligned = [cc.smooth(kernel="shk", mask_medial_wall=True) for cc in subjects]

# the association, on the 946: a batch job with 230 GB of memory
table = {f"sub-{r['Subject']}": r for r in csv.DictReader(open("unrestricted.csv"))}  # the HCP's open-access table
cohort = [p.name.split("_")[0] for p in sorted(Path("hcp-ya-full").glob("sub-*_sc.h5"))]
cohort = [s for s in cohort if table[s]["PMAT24_A_CR"]]    # those with a score
files = [f"hcp-ya-full/{s}_sc.h5" for s in cohort]
reduction = sbci.reduce(files, rank=20)                    # FPCA, one file at a time
rows = [table[s] for s in cohort]
score = np.array([float(r["PMAT24_A_CR"]) for r in rows])  # fluid intelligence
female = np.array([r["Gender"] == "F" for r in rows], dtype=float)
band = np.array([{"22-25": 23.5, "26-30": 28, "31-35": 33, "36+": 37}[r["Age"]] for r in rows])
count = np.array([sbci.load(f).metadata["streamline_count"] for f in files]) / 1e6
design = np.column_stack([score, female, band, count])
result = sbci.local_test(reduction.scores, design, terms=[1])
result.significant()                                       # components tracking the score, given the rest
effect = result.effect_map(reduction, alpha=0.05)          # one value per vertex, significant components only
sbci.load(files[0]).plot(effect, mesh="fsaverage", engine="pyvista")
```

`terms=[1]` tests the score column only: column 0 is the intercept the test
adds, and columns 2 to 4, sex, the age band and the streamline count, stay in
the model as nuisance.
The streamline count matters: it is the largest source of difference between
subjects' connectomes, and on 528 HCP-Aging subjects a rank-4 fit, which never
reaches past it, missed the age effect a rank-20 fit found (PORTING.md item
5).

**Running it on Longleaf.** `reduce` holds the cohort once, as one dense
float64 array: 199 GB for 946 subjects, 111 GB for 528. Given paths, or a
sequence whose items are loaded when indexed, it reads one subject at a time,
so the files need not be held as well. The 528 peaked at 116 GiB; ask for
230 GB for the 946.
Ask for the cores as one task, `--ntasks=1 --cpus-per-task=8`: with eight
one-CPU tasks the cluster sets `OMP_NUM_THREADS=1`, and the linear algebra
then runs on one core. On the 528 HCP-Aging subjects the rank-20 fit took four
hours that way; with eight threads it reads the cohort about three times
faster.

The pipeline is checked against a planted answer on a synthetic cohort
(`sbci.example_cohort()`, one bundle scaled by a synthetic age) in PORTING.md
items 5 and 7: what more anatomical variation does, what aligning first does
to the analysis, and which solver start reaches the planted component.

## Reducing a cohort to a handful of numbers

A connectome is thirteen million numbers. FPCA finds a small set of surface
functions shared across the cohort, so each subject becomes a short score
vector suitable for regression or classification.

```python
import sbci
from sbci import ContinuousConnectome

subjects = [ContinuousConnectome.load(p) for p in paths]
result = sbci.reduce(subjects, rank=20)

result.basis            # (5124, 20)  the shared functions
result.scores           # (n_subjects, 20)  one row per subject
result.explained[-1]    # fraction of the cohort's norm captured
result.reconstruct(0)   # subject 1 rebuilt from its 20 numbers
```

Score new subjects against an existing basis with
`sbci.reduction.project(result, matrices)`, which is how a test-retest or
held-out set is handled. A single connectome is allowed and gives its own
rank-K separable approximation:

```python
sc.reduce(rank=10)
```

Three things to know:

- **The fit is a local optimum, and the start is fixed.** Each component
  begins from a random vector, as in the reference; `seed=0` by default, so a
  run repeats exactly, and `seed=None` draws afresh. In the synthetic checks
  of PORTING.md item 5 twelve seeds agree on the first three components and
  differ in the fourth,
  and about one start in fifteen misses a component the others find; with
  more anatomical variation it can miss the largest component altogether
  (PORTING.md item 5). `candidates=6` starts each component from the six
  leading eigenvectors of the mode-1 Gram matrix and keeps the largest, at
  about three times the cost. If a component matters, check that a second
  seed or `candidates=6` reproduces it, and read `explained` before trusting
  a rank.
- **`alpha` is a roughness penalty with a turning point.** Components are
  chosen by largest *magnitude* eigenvalue and the penalty enters negatively,
  so once it outweighs the data the fit returns the *roughest* direction
  instead of the smoothest and explains almost nothing. The default of `1e-10`
  is far below that; if you raise it, check `explained` has not collapsed.
- **This is a batch job on the full grid.** Each component involves several
  5124 x 5124 products; budget tens of GB and start with a small `rank` and
  `max_outer`.

## Testing scores against a covariate

```python
import sbci

result = sbci.reduce(subjects, rank=20)
test = sbci.stats.local_test(result.scores, age)

test.significant(0.05)            # which components carry an association
test.adjusted                     # Benjamini-Hochberg p-values
test.effect_map(result, alpha=0.05)   # where on the cortex the effect sits
```

Pass `method="bonferroni"` for family-wise control, or `permutations=5000` to
drop the Gaussian assumption. `effect_map` pushes the fitted coefficients back
through the basis and sums over one endpoint, giving one value per vertex --
it describes where the fitted effect lives; the p-values belong to the
components, not to individual vertices.

**Read this one more sceptically than the rest.** Every other function here is
checked against a MATLAB run of the original. There is no reference
implementation for local inference -- the FPCA repository contains none -- so
the choice of test is mine. It is verified against `scipy.stats` and against
the properties the procedures are defined by, but it has not been blessed by
anyone who wrote the method. See PORTING.md item 5.

## Aligning a cohort

ENCORE warps each subject's two spherical surfaces so that their connectivity
profiles line up, then reports the template and the per-subject warps.

```python
import sbci
from sbci import ContinuousConnectome

subjects = [ContinuousConnectome.load(p) for p in paths]
result = sbci.align(subjects, order=6, max_iterations=50)

result.template          # the Karcher median, as a square-root density
result.aligned[0]        # subject 1 warped onto it
result.costs             # final cost per subject
result.warps[0].save("sub-001_warp.npz")
```

**How it is used.** `align` registers every connectome in the list onto one
template and hands back the warps and the warped connectomes. With no
`template=`, which is the default, it estimates the template first: the
Karcher median of the subjects' square-root densities (ten Weiszfeld steps
from the subject nearest the mean), so that no subject is the reference and
every subject moves. That is the cohort study's setting. Pass `template=` a
square-root density of unit mass to register onto that instead: an earlier
run's `result.template`, to bring a new subject onto a cohort's template
without re-estimating it, or one subject's own
`sbci.alignment.Encore(*grids).root(cc.dense())`, to register one subject
onto another. A given template needs no second connectome in the list.

```python
new = sbci.align([late_subject], template=result.template)   # onto the cohort's template
pair = sbci.align([moving], template=sbci.alignment.Encore(*grids).root(fixed.dense()),
                  grids=grids)                                # one subject onto another
```

**This is a batch job.** On the ico4 grid each iteration multiplies 5124 x 5124
matrices several times over; budget tens of GB of memory and hours for a
cohort, and start with a small `max_iterations` to see the cost falling.

`result.traces` holds each subject's cost before registration and after every
accepted step.

Three things to know before trusting the numbers:

- **The bundled grid is rotated first.** The Jacobian is built in `(theta, phi)`
  coordinates and closes with a factor of `sin(theta)`, so a vertex on the
  coordinate axis gets a Jacobian of exactly zero and loses its whole row and
  column. The ico4 grid has four such vertices, so `align()` rotates the mesh
  clear of them -- a change of coordinates and nothing else. It refuses a grid
  you supply yourself that still has poles.
- **`delta` defaults to 1e-5, not the reference's 1e-10.** A central difference
  at 1e-10 loses six of sixteen digits; the reference's own derivative moves by
  2% of its range between adjacent step sizes. Pass `delta=1e-10` to reproduce
  the reference's conditioning exactly. PORTING.md item 4 has the measurements.
- **A step that fails is halved, not fatal.** The reference moves by a fixed
  length and stops the first time the cost does not fall, which on two similar
  subjects is the first step: it returns the identity having done nothing. The
  port halves the step up to `backtracks=4` times first, and on a known
  one-degree deformation undoes 59% of it where the reference undoes none
  (PORTING.md item 4); `backtracks=0` reproduces the reference.

## Aligning by endpoints (ConSEAL)
ENCORE moves a smoothed density. ConSEAL moves the streamline endpoints
themselves: each endpoint rides along with the warped triangle it fell in, the
density is re-smoothed from where the endpoints now sit, and nothing is ever
resampled. It needs files that store the endpoints with their barycentric
positions (`cc.endpoints.has_positions`), as the released subjects do;
*Storing endpoints, and re-smoothing from them* says how to build them from
the pipeline's output.
```python
import sbci
subjects = [sbci.load(p) for p in paths]
result = sbci.endpoints_align(subjects)        # the public code's defaults
result.template                               # Karcher median, a square-root density
result.costs[0]                               # the cost trace of subject 1
result.warps[0].save("sub-001_conseal_warp.npz")
aligned = result.aligned_endpoints(0)         # an Endpoints object: re-smooth it, count it
```
It is used the same way as `align`. With no `template=`, the default, it
estimates the Karcher median of the subjects' square-root densities first
(up to 100 Weiszfeld steps) and registers every subject onto it. `template=2`
registers every subject onto subject 3's own density, which stays where it
is; `template=` a square-root density array registers onto that, whether an
earlier run's `result.template` or the normalized mean of the subjects'
`q_transform(kernel)` arrays, which the caveat below on the median explains
when to prefer.

Two of the released subjects will do to see it run:
`sbci.endpoints_align([sbci.load(a), sbci.load(b)], max_iterations=5)` with
two of the released SC files takes a few minutes on four cores, and both costs
fall at every step.
**This is a batch job too**, though a lighter one: the heat kernel at the
published bandwidth has 89 nonzeros per row on ico4, so an iteration costs
seconds per 100,000 streamlines rather than minutes. Six things to know:
- **The defaults are the public code's, not the paper's.** Step 0.05, up to
  100 iterations, threshold 1e-4, a 0.2 clamp on the largest displacement and
  5% Laplacian smoothing of the velocity field every step. The paper's own
  experiments were run from a fork without the clamp and smoothing, at step 0.1
  and threshold 1e-6; `viscosity=0, step_clamp=float("inf"), delta=0.1,
  threshold=1e-6` gives that update rule. PORTING.md item 7 explains the
  lineage.
- **Four errors in the reference are corrected by default.** Its gradient adds
  a term in the wrong tangent frame and differentiates a differently
  normalized kernel; a refused warp step still enters its velocity field; and
  a rising cost is accepted as convergence. `strict_upstream=True` reproduces
  all four, and does so to the digits of the MATLAB reference run.
- **Rigid initialization is off by default**, as in the reference's own
  example; `init_rotation=True` runs the multi-shell rotation search first.
- **The stopping threshold is absolute.** The public default of 1e-4 is a
  quarter of the whole cost when two subjects are alike, and stops the
  registration after a few iterations; `threshold=1e-7` lets it converge. On
  a known deformation the public defaults undo a third of it, the paper's
  update rule four fifths (PORTING.md item 7).
- **The Karcher median can be one subject.** With `template=None` the median
  starts at the subject nearest the mean and takes Weiszfeld steps until one
  is shorter than 0.005, as `get_template` does. When the subjects sit evenly
  around the mean the first step is already that short and the template *is*
  that subject: its cost is 0, it takes no iterations, and everyone else is
  registered onto its bundles. In the synthetic checks of PORTING.md item 7
  this happens at three degrees of anatomical spread (every subject 10 to 11 Fisher-Rao degrees
  from the mean) and not at two, and it happens on the ten young adults of
  the example cohort drawn at random: the median lands 0.001 degrees from sub-212116 and 24 to 27 from the
  others, while every subject is 17 to 20 degrees from the mean. Pass `template=` a subject index or a
  precomputed square-root density -- the normalized mean of the subjects'
  `q_transform(kernel)` arrays, for one -- to choose.
- **The paper's unregularized update onto one subject can align away a real
  difference.** With `delta=0.1, step_clamp=inf, viscosity=0` and the template
  collapsed onto a subject, the planted bundle of the synthetic cohort in
  PORTING.md item 7 (`anatomy=0.05`) is gone from a rank-4 FPCA that finds it
  unaligned, after ENCORE, after the same update onto the mean template, and
  after the public update (clamp 0.2, viscosity 0.05) onto that same subject:
  an unclamped, unsmoothed warp can match one subject's bundles exactly.
  Either a chosen template or the public regularization keeps the difference
  (PORTING.md item 7).
![Alignment with a known answer](docs/figures/alignment_recovery.png)

*The endpoints of sub-103010's 988,788 streamlines moved by a known smooth
warp (degree 4, 1.7 degrees on average, 4 at most) and registered back onto
the undeformed subject, both through the package's smoother. Top: how far the
endpoints still are from where they started, vertex by vertex. Bottom: the
same as a histogram, and the cost per iteration. ENCORE, with its default
degree-6 basis, brings the endpoints back from 1.65 to 0.22 degrees in eight
steps; ConSEAL with the paper's update (`delta=0.1, step_clamp=inf,
viscosity=0`) and a stopping threshold of 1e-7 to 0.12 degrees in sixty. The
reference has to go through the same smoother as the deformed copy, and the
warp has to be one a smoothed density can see; PORTING.md item 4 shows what
happens otherwise.*

## Carrying a warp to another template

ENCORE and ConSEAL estimate a warp of the sphere the grid lives on. Data
registered to another template -- HCP's fs_LR, where MSMAll-aligned surfaces
live, or fsaverage at full resolution -- sit on a different sphere, related
to the grid's by a registration. `sbci.migrate_warp` carries the warp across:
each template vertex is taken back to the grid's sphere, moved by the warp,
and taken forward again, so the deformation is the same one seen from the
other template.

**When you would want this.** ENCORE has aligned ten subjects' structural
connectomes on the grid, and the same subjects have MSMAll-registered
resting-state or myelin maps on fs_LR 32k. The warp that made subject 4's
connectivity line up with the cohort is an anatomical correspondence, and it
applies to those maps too, but they live on a different sphere. `migrate_warp`
restates the warp on fs_LR and writes it as a deformed sphere; Workbench then
resamples any 32k map through it, and the map is aligned the way
`alignment.aligned[3]` is:

```python
alignment = sbci.align(subjects)                                 # ENCORE on the grid
warp = sbci.migrate_warp(alignment.warps[3], to="fs_LR_32k",
                         grid_rotations=alignment.grid_rotations)
warp.to_gifti("sub-004_encore")        # sub-004_encore.L.sphere.surf.gii and .R.
```

```bash
wb_command -metric-resample sub-004.L.myelin.32k_fs_LR.func.gii \
    L.sphere.32k_fs_LR.surf.gii sub-004_encore.L.sphere.surf.gii \
    BARYCENTRIC sub-004.L.myelin.aligned.func.gii
```

The order of the two spheres carries the direction, and the two methods
differ in it. ENCORE's warp is a pull-back: the aligned value at a vertex is
read from where the warp sends that vertex, so the standard sphere is
Workbench's current sphere and the deformed one its new sphere, as above.
ConSEAL's warp is a push-forward: an endpoint moves with the warp, so for a
ConSEAL warp the deformed sphere is the current one and the standard sphere
the new one. `warp.apply(points, "L")` gives the same correspondence for
points you handle yourself.

![Carrying a warp between templates](docs/figures/migration_power.png)

*Measured on an HCP-Aging subject, since the young adult files carry no resting-state data; the HCP-Aging files are not distributed. Top: the subject's resting-state connectivity of vertex 1234 on fsaverage (163,842 vertices per hemisphere, correlation with the seed's time series), the same map moved by the known warp of the recovery figure (r = 0.87 with the original), and put back by ENCORE's warp carried from the grid with `migrate_warp` (r = 0.99). Bottom: ENCORE's warp on the grid (5,124 vertices), the same warp restated on fsaverage, and the known warp on fsaverage; the carried warp is 0.35 degrees from the true one over both hemispheres and 0.33 over the left one shown, where the true warp moved vertices 1.58 degrees on average.*

```python
alignment = sbci.align(subjects)                              # ENCORE
moved = sbci.migrate_warp(alignment.warps[0], to="fs_LR_32k",
                          grid_rotations=alignment.grid_rotations)
registration = sbci.endpoints_align(subjects)                 # ConSEAL
moved = sbci.migrate_warp(registration.warps[0], to="fs_LR_32k")

moved.lh_vertices           # (32492, 3): where each fs_LR vertex lands
moved.lh_jacobian           # area ratio at each vertex
moved.apply(points, "L")    # any fs_LR-sphere points, moved
moved.to_gifti("sub-001")   # sub-001.L.sphere.surf.gii and R: deformed spheres
                            # that Connectome Workbench takes as a registration
moved.save("sub-001_fslr_warp.npz")
```

Three things to know:

- **Two frames, and they differ.** The bundled `sphere` is the pipeline's own
  parameterization, not FreeSurfer's standard sphere: the grid's vertices are
  fsaverage vertices, but their sphere coordinates sit 119 degrees from the
  standard sphere's on median. The package carries the standard-sphere
  coordinates of every grid vertex and maps between the two first.
- **fs_LR is reached through HCP's deformed sphere.** `fs_LR-deformed_to-fsaverage`
  places every fs_LR-32k vertex on the standard sphere; with the fs_LR sphere
  itself that is the registration (`tools/build_template_spheres.py`). MSMSulc
  and MSMAll differ per subject, not in the group sphere, so a group warp lands
  on fs_LR either way; a subject's own sphere pair adds one more step:
  `sbci.SphereMap.from_gifti(native_sphere, msmall_sphere)`.
- **The maps are piecewise linear.** Each step is a barycentric lookup, so a
  point carried to fs_LR and back returns within 0.3 degrees, and the identity
  warp migrates to the identity within the same. Pass an ENCORE warp with its
  `grid_rotations`: ENCORE works on a rotated copy of the grid.

## Writing the exchange file

```python
from sbci import ContinuousConnectome

sc = ContinuousConnectome.load("/work/users/x/y/xya/sbci-derivatives/sub-example_sc.h5")
sc.to_cifti("/work/users/x/y/xya/sub-example_sc.dconn.nii")
```

Three files are written: the dense connectome, a `.json` sidecar with the
metadata table, and a `_vertexarea.dscalar.nii` of fsLR vertex areas. The
connectome holds a density, so anything that integrates it -- parcellation,
totals, region means -- needs those areas; they are not recoverable from the
`.dconn` alone.

**This does not belong on a login node.** The output is 64,984 x 64,984 in
float32, **16.9 GB on disk**, and the writer needs about 25 GB of memory and
three minutes. Use a batch job:

```bash
sbatch --mem=60G --time=01:00:00 --wrap="module load python/3.12.4; source /work/users/x/y/xya/sbci-venv/bin/activate; python scripts/write_exchange_file.py"
```

To read it back, use plain `nibabel` -- the package deliberately does not
provide the inverse, because 5,124 values cannot be recovered from 64,984 and
a silent round trip would degrade the data:

```python
import nibabel as nib, numpy as np

img = nib.load("/work/users/x/y/xya/sub-example_sc.dconn.nii")
axis = img.header.get_axis(0)          # BrainModelAxis, 64,984 elements
row = np.asarray(img.dataobj[1234])    # one vertex's profile, memory-mapped
```

Slice `img.dataobj` rather than calling `get_fdata()`, which would pull all
16.9 GB into memory at once.

### Checking it with Connectome Workbench

Workbench is the reference reader for this format, and it is already on
Longleaf. Use **1.5.0** -- the 2.0.1 build on this cluster is missing
`libglapi.so.0` and will not start.

```bash
module load connectome/1.5.0
wb_command -file-information /work/users/x/y/xya/sbci-derivatives/sub-example_space-fsLR_den-32k_desc-concon_sc.dconn.nii
```

On the example subject this reports `Type: CIFTI - Dense`, `Structure:
CortexLeft CortexRight`, 64984 x 64984, and `32492 out of 32492 vertices` for
each hemisphere -- so Workbench's own reader agrees with the header the writer
produced. The phrase "out of" is worth noting: it is Workbench saying the file
covers every vertex of the mesh including the medial wall, rather than the
59,412 cortical grayordinates HCP files usually carry. That is the convention
still open for ratification in SPEC_QUESTIONS.md item 4.

`wb_view` is the graphical browser, but it needs a display; from Longleaf that
means an OnDemand desktop session rather than a plain `ssh`.

## What raises, and what it tells you

| Call | Raises | Because |
| --- | --- | --- |
| `sc.smooth()` on a file with no `/endpoints` | `MissingDataError` | nothing to re-smooth from; the group is optional |
| `sc.smooth()` on endpoints stored as vertices only | `MissingDataError` | `shk` needs where each streamline crossed, not the nearest vertex |
| `sbci download hcp-aging` | exit status 1: `cohort must be one of ('hcp-ya',)` | only the young adult cohort is released; HCP-Aging data are not distributed |
| `sbci download hcp-ya --fc-only` | exit status 1: `the hcp-ya cohort has no fc files` | the young adult files are structural only for now |

Every message names what has to happen and where it is tracked.
