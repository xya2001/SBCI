# MATLAB components to port

WP2 named four MATLAB components to port; seven are ported here. They are listed
with their source files, the Python module that will hold them, and the test
that will decide the port is correct. **A port is not done until its
correctness test passes against the MATLAB output on the tutorial subject** --
a function that runs and returns a plausible array is not a finished port.

Sources are the canonical repositories under
<https://github.com/sbci-brain>, not any local working copy. On Longleaf a
shallow clone of each is kept at `/work/users/x/y/xya/sbci-reference` for
reading; re-clone rather than edit, since `/work` is purged.

Subjects are named here by study-specific labels -- the tutorial subject, and B to E for
four others of the eleven released -- never by their HCP ids: several items report
analyses that treat families as clusters, and the HCP's Restricted Data Use Terms allow no
HCP subject id in a public report of an analysis that uses restricted data, family
structure among it. README.md and USAGE.md, which name the released subjects, report none.

## 1. Riemannian diffusion and Matern kernel smoothing -- DONE; `rdk` VERIFIED AGAINST MATLAB, `matern` AGAINST ITS CLOSED FORM

- **From:** [`SBCI_Toolkit`](https://github.com/sbci-brain/SBCI_Toolkit) --
  `concon_estimate/compute_diffusion_kernel_matrix.m`,
  `concon_estimate/compute_matern_kernel_matrix.m`,
  `rdk_smoothed_concon_compute.m`
- **To:** `src/sbci/smoothing.py` -- `diffusion_kernel`, `matern_kernel`,
  `kappa_candidates`, `Endpoints`, `smooth_endpoints`, `load_eigenpairs`.

### Verified against a MATLAB run

`tests/reference/reference_run.m` is the reference implementation, run under
MATLAB R2023b on Longleaf (`module load matlab/2023b`) on the example
subject's 383,760 streamline endpoints at the bandwidth the script itself
selects, `kappa = 1.94826245307922`. Comparing its output to this port:

| Quantity | Agreement |
| --- | --- |
| Adjacency `A11`, `A22`, `A12` | **bit-identical**, max difference exactly 0 |
| Kernel `KM_L` | 2.01 float32-eps relative, correlation 0.9999999999999 |
| Density `DM` | 5.15 float32-eps relative, correlation 0.9999999999999 |
| The `smooth()` method, its density normalized and its diagonal zeroed as stored | 3.14 float32-eps relative, correlation 0.99999999999994; diagonal exactly 0, unit mass 1.000000000014 |
| The bandwidth `smooth()` takes when given none | 1.948262356 against 1.948262453, 5.0e-08 relative |

*As `tests/reference/smooth_matlab_probe.py` printed them on 12 October 2026
on an Intel and an AMD node (jobs 4627321 and 4628494, item 24; the
correlations differ between the two in the fifteenth decimal), against the
reference files of 21 September. This
table first quoted 3.18 and 3.17 float32-eps for the kernel and the density,
a run no record names; Tier 4 holds every row within 10 float32-eps.*

**Why float32 and not float64.** `Lambda` is stored as `single` in
`EV_LBO_ds_ico4_*.mat`. In MATLAB `single .* double` yields `single`, so
`( U .* repmat(rho',[nnds,1]) ) * U'` runs entirely in single precision --
confirmed by the reference writing `KM_L` out as a float32 array. The
agreement above is therefore at the limit of the precision the reference
itself carries.

`sbci.smoothing` promotes the eigenvalues to float64 and keeps float64
throughout, so it is **more accurate than the reference, not different from
it**. If bit-compatibility with legacy output is ever needed, cast `Lambda` to
float32 before calling `diffusion_kernel`.

### Speed

The whole smoothing takes **4.4 s** for both hemispheres, against a WP3 target
of 60 s and the 2-8 hours the current pipeline needs.

### It does NOT reproduce `smoothed_sc_avg_0.005_ico4.mat`, and should not

That file is not made by this MATLAB script. `sbci_step5_structural.sh` builds
it with a different program entirely:

```
c3_main Compute_Kernel --subj subject --sigma ${BANDWIDTH} --epsilon 0.001 \
        --final_thold 0.000000001 --OPT_VAL_num_harm 33 ...
```

That is the ConCon C++ binary using a **spherical harmonic** kernel with 33
harmonics and a 1e-9 threshold -- which is also why the released file is
sparser (5.08M non-zero pairs) than any Laplace-Beltrami bandwidth produces.
`sigma = 0.005` parameterizes that kernel, not this one, and the two are not
the same quantity: at `kappa = 0.005` and `kappa = 0.05` the diffusion kernel
damps nothing at all (`rho` is 1.0 to six decimals), giving r = 0.215 against
the released file, the worst of any bandwidth tried. Agreement peaks at
r = 0.58 near `kappa = 8.9`, which is simply two different smoothers applied
to the same brain.

If a like-for-like comparison with the pipeline's output is wanted, the
spherical-harmonic kernel is a separate port, not a bandwidth of this one.

## 2. The three SFC functions -- DONE, VERIFIED

- **From:** [`SBCI_Toolkit`](https://github.com/sbci-brain/SBCI_Toolkit) --
  `sfc/calculate_sfc_gbl.m`, `sfc/calculate_sfc_loc.m`,
  `sfc/calculate_sfc_dct.m`
- **To:** `src/sbci/coupling.py` -- `global_coupling`, `local_coupling`,
  `discrete_coupling`, and `structure_function_coupling` behind
  `ContinuousConnectome.coupling()`.

### Verified against a MATLAB run

`tests/reference/sfc_reference.m`, run under MATLAB R2023b on the example
subject's SC and FC, compared to the port:

| Function | Agreement | Finite values |
| --- | --- | --- |
| `calculate_sfc_gbl` | 10.4 float64-eps | 4758 / 4758, identical NaN pattern |
| `calculate_sfc_dct` | 88.2 float64-eps | 4758 / 4758, identical NaN pattern |
| `calculate_sfc_loc` | 3.6 float64-eps | 4746 / 4746, identical NaN pattern |

Unlike the smoothing port this agrees to full float64, because the SFC code
path carries no single-precision inputs.

### Two things the reference does that are worth knowing

**The discrete form is not a cosine similarity.** `calculate_sfc_gbl` and
`calculate_sfc_loc` use the uncentred inner product,
`dot(fc, sc) / (|fc| |sc|)`. `calculate_sfc_dct` uses MATLAB's `corr2`, which
centres both vectors first and is therefore a Pearson correlation. The
manuscript and the brief describe all three as cosine similarity; only two of
them are. Worth checking which one the published figures used.

**The diagonal is handled inconsistently.** `calculate_sfc_gbl` clears the
diagonal only when asked to symmetrize a triangular input, while
`calculate_sfc_loc` and `calculate_sfc_dct` always clear it. So calling the
global form on an already-symmetric matrix leaves self-connectivity in the
profile while the other two drop it. The port preserves this because changing
it would change published numbers, but it looks like an oversight rather than
a decision -- one for the group.

### Measured on the HCP Young Adult cohort

The lab's copy of the young adults holds no pipeline FC, so
`tools/build_hcp_fc.py` builds it from the HCP's ICA-FIX-cleaned
resting-state runs, placing each 32k fs_LR vertex on the grid through the
subject's MSMSulc and FreeSurfer spheres so that the FC sits where the SC's
endpoints do (USAGE.md, *Functional connectivity from the HCP's resting
state*). There is no pipeline FC of these subjects to compare it with, so the
checks are what resting-state FC must show, on the tutorial subject:

| Check | Result |
| --- | --- |
| the 32k vertices a grid vertex averages | in the median cell all within 2.4 degrees of the grid vertex (95th percentile 3.3), against a grid spacing of about 4 |
| two 32k vertices of one cell, correlated in time | median 0.65; two at random, 0.009 |
| neighbouring grid vertices | FC 0.27 (0.43 band-passed to 0.01-0.1 Hz) |
| homotopic Desikan regions | 0.38 (0.54) |
| seed in the left isthmus cingulate | precuneus 0.46, inferior parietal 0.36, medial orbitofrontal 0.31: the default network |
| the white-surface route subject B needs, run on the tutorial subject | 91% of the 32k vertices in the same cell as the exact route, 0.24 degrees from it at the median |

The released files are full band, as the pipeline computes FC. Coupling of
each of the eleven subjects' SC with its own FC:

| Form | Mean per subject |
| --- | --- |
| global | 0.200 to 0.268 |
| within Desikan regions | 0.595 to 0.679 |
| discrete, on the Desikan matrices | 0.329 to 0.421 with SC's mean density, `how="mean"` (0.271 to 0.370 with its mass, as quoted until item 18) |

The NaN are the 439 medial-wall vertices and no others, and the global maps
of two subjects correlate at r = 0.53 to 0.71 (mean 0.62). Averaged over the
eleven, coupling is 0.299 over 18 primary and unimodal sensory and motor
regions and 0.190 over 26 association regions, highest in the pericalcarine,
cuneus and lateral occipital cortex (0.36 to 0.41) and lowest in the caudal
anterior and posterior cingulate and the entorhinal cortex (0.05 to 0.09):
the sensory-to-transmodal gradient of Vázquez-Rodríguez et al. (PNAS 2019)
and Baum et al. (PNAS 2020). A subject's own FC fits its SC only a little
better than another subject's: 0.233 against 0.225 on average, and better
than each other subject's in 60% of pairs.

**The cohort.** FC was built for all 946 young adults with SC: 916 have the
four runs, 903 of them complete (1,200 frames each; 13 have a truncated
run), 8 three, 21 two and 1 one. Every file validates; subject B alone took
the white-surface route, and 1.1 to 1.8% of each subject's cortical 32k
vertices fall on the grid's medial wall. Three subjects' MSMSulc spheres
have one or two 32k vertices inside a stretched triangle whose centroid is
not among the 128 nearest, which the search over every triangle now finds. The coupling
analysis takes the 903 with four complete runs (`scripts/hcp_figures.py
--full-fc`). Their mean coupling is 0.223 (0.170 to 0.278 per subject), 0.285
over unimodal against 0.178 over association regions, lowest in the caudal
anterior and posterior cingulate and the entorhinal cortex (0.05 to 0.08) and
highest in the pericalcarine, cuneus and lateral occipital cortex (0.34 to
0.40). The mean map correlates with the eleven's at r = 0.972, and two halves
of the cohort split by family reproduce it at r = 0.998.

Coupling vertex by vertex against fluid intelligence and against sex, each
given the other, the age band and the streamline count, families as clusters
(`local_test` on the 4,685 cortical vertices, FDR across them), for the 900
with a score:

| Model | Fluid intelligence: vertices; mean coupling p | Sex: vertices; mean coupling p |
| --- | --- | --- |
| as above | 7 (smallest adjusted p 0.00075); 0.04 | 1,021 (2e-14); 0.002 |
| and head motion | 6 (0.0038); 0.014 | 1,104 (1.5e-14); 0.00055 |
| and intracranial volume | 2 (0.028); 0.036, higher with the score | 325 (0.00014); 0.045, lower in women |

Head motion is the mean over the four runs of the HCP's
`Movement_RelativeRMS_mean.txt` (median 0.078 mm; r = -0.17 with the score,
0.09 with mean coupling); intracranial volume is FreeSurfer's eTIV from each
subject's `aseg.stats` (r = -0.63 with being a woman, 0.24 with the score,
0.19 with mean coupling). Head size takes most of the sex difference. In the
full model, 253 of the 325 vertices have coupling higher in men, mostly in
the left inferior parietal (22), right superior frontal (15), right pars
opercularis (14), both insulae (22), right pericalcarine (11), left superior
temporal (10) and right lingual (10) cortex, the fitted difference reaching
-0.038. The 72 higher in women (to +0.029) lie mostly in the left insula (13)
and along the isthmus and posterior cingulate (29), and 44% of them are within two
grid rings of the medial wall, against 6% of cortex (24% within one ring,
against 3%); the vertices higher in men are not (6%). Next to the wall SC
loses the endpoints the mask drops and the FC fills cells the HCP's surface
leaves empty, so that part of the map is the one to doubt. The same test on
the eleven released subjects finds hundreds of vertices: the cluster-robust
test needs many clusters (the stats module notes), and the eleven are for
checking the code, not for inference.

## 3. Parcellation -- DONE, VERIFIED

- **From:** [`SBCI_Toolkit`](https://github.com/sbci-brain/SBCI_Toolkit) --
  `analysis/parcellate_sc.m`, `analysis/parcellate_fc.m`,
  `analysis/upsample_data.m`, `analysis/downsample_data.m`
- **To:** `src/sbci/parcellation.py`, reached through
  `ContinuousConnectome.to_atlas()`.
- **Atlas label files are vendored.** `tools/convert_atlases.py` turns each
  `*_avg_roi_ico4.mat` into an `.npz` under `src/sbci/data/atlases/`, 44 of
  them, so no release step needs MATLAB.

### Agreement with the reference

`parcellate_sc.m` was run on the example subject at full scale and diffed
against `to_atlas(how="mean")`. Over the off-diagonal entries:

| Quantity | Result |
| --- | --- |
| max relative difference | 1.875e-16, **float64 rounding** |
| correlation | 1.000000000000000 |

`tests/test_matlab_reference.py::test_to_atlas_matches_parcellate_sc` asserts
it. Three conventions have to be reconciled first, and each is a real
difference rather than a rounding detail:

- **Background regions.** MATLAB returns 70 rows for Desikan, not 68:
  FreeSurfer-derived atlases carry a background entry per hemisphere
  (`LH_missing` at index 0, `RH_missing` at 35) and the toolkit keeps both.
  `tools/convert_atlases.py` folds all background to label 0, which is why
  Schaefer200 here is 200x200 rather than 201x201.
- **Triangle.** `parcellate_sc.m` loops over `i < j` and leaves the lower
  triangle at zero, so its output is not symmetric. This package returns a
  symmetric matrix.
- **Diagonal.** It also leaves the region diagonal at zero, discarding
  within-region connectivity. The port computes it as the area-weighted mean
  over the region's *distinct* vertex pairs: a vertex paired with itself has
  no connectivity and is left out of the denominator, and a one-vertex
  region, having no pair, gets NaN. Until the review of 5 October 2026 (item
  8) the denominator was the squared region area, which counted the
  self-pairs and ran the diagonal low by `1 - sum_i a_i^2 / A^2`: half with
  two vertices, about 4% for Schaefer-sized regions on ico4. Of the 11,822
  regions in the 44 bundled atlases, 68 have a single cortical vertex on
  ico4 (21 of them in Schaefer-1000, 15 in Schaefer-900) and 45 have none
  (six in Gordon, two in Glasser, one in PALS OrbitoFrontal, and one to four
  in each CoCoNest scale of 100 regions or more); those read NaN under
  `how="mean"`. Desikan's and Schaefer-200's smallest regions have five.

- **Open divergence:** the triangle convention above. The diagonal one was
  decided with the review. Confirm which behavior the released files should
  carry -- it changes every published region matrix, so it is a decision, not
  a detail.

## 4. ENCORE -- DONE, VERIFIED IN TWO TIERS

- **From:** `ConCon_Alignment` under <https://github.com/sbci-brain>. Four
  MATLAB classes plus three MEX kernels built on libigl: an AABB closest-point
  tree, a mixed-Voronoi mass matrix, and a tensor interpolation over the
  product mesh.
- **To:** `src/sbci/alignment.py`, reached through `sbci.align()`. The MEX
  kernels are reimplemented rather than wrapped -- a spatial index is an index,
  not an algorithm -- and libigl's Voronoi mass matrix is reproduced directly.
- **Reference run:** `encore_reference.m` on a level-2 icosphere, 162 vertices
  per hemisphere, with the right hemisphere rotated so the two are not
  interchangeable. It dumps every intermediate, so the port is checked stage by
  stage rather than only end to end. The figures below that no test of Tier 4
  asserts were measured on 20 September by scripts never committed;
  `tests/reference/encore_first_checks_probe.py` measures them again with the
  package (item 24), and they are its output.

### What matches exactly

| Stage | Agreement |
| --- | --- |
| Voronoi vertex areas (libigl massmatrix) | 1.6 float64-eps |
| Spherical harmonic tangent basis, with the reference's clamp at the poles put back (item 21) | 12.2 eps |
| Basis Laplacian, with the clamp and the reference's recurrence put back (items 8 and 21) | 5.3 eps |
| Tangent frames, exponential and logarithm maps | 0.5 to 4 eps |
| Barycentric query against the AABB tree | 2.0 eps, same triangle 162/162 |
| Identity warp Jacobian: 1 at every vertex, which the reference sets rather than computes and the port's calibration gives (its scheme reads 0.935 there on this grid, *Four more* below) | exact |
| Warp composition from the identity, vertex by vertex (both orders of item 21 are one map there) | 1.0 eps |
| **Karcher median template** | **44 eps** |

### What cannot match, and why

Everything downstream of the reference's finite difference agrees only to
about 1e-3, and that is a property of the reference rather than of the port.
It differentiates with a step of `1e-10`. A central difference with that step
on quantities known to float64 precision has a condition number near `1e6`:
six of sixteen digits are gone before anything else happens.

Measured on the reference itself, changing only the step size
(`tests/reference/encore_delta_sweep.m` for the derivative,
`encore_jacobian_sweep.m` for the Jacobian):

| Quantity | Change between adjacent step sizes |
| --- | --- |
| `get_derivative` | **0.037**, about 2% of its range |
| warp Jacobian | 2.3e-04 |

The port lands inside that envelope: Jacobian 4.9e-04, `evaluate` 7.1e-04,
and the transported square root, the reference's renormalized, 5.4e-04. The
registered connectome agreed to 1.2e-03, at a correlation of **0.99999979**,
until the port took the order of composition its gradient assumes (item 21);
the reference's registration is no longer compared. Given
*identical* input points the Jacobian arithmetic reproduces the reference to
3.8e-06 on an Intel node and 4.4e-06 on an AMD one, of the order of float64
epsilon divided by the step, 2.2e-06 -- so the arithmetic is identical and only
the conditioning separates them.

### A second bug in the reference: the returned cost

`register` stops when a step fails to improve, restores the previous warps and
returns -- but returns `cost` from the **rejected** step, not from the warps it
hands back. On the reference run:

| | |
| --- | --- |
| cost the reference returns | 0.092453322308 |
| cost of the warps it actually returns | 0.089877610062 |
| difference | 2.6e-03 |

The port returns the cost belonging to the warps it returns, which agrees with
the reference's own accepted-step cost to **8.8e-06**. An earlier draft of this
note reported a 2.8e-02 cost disagreement; that was the port's correct value
being compared against the reference's mis-reported one.

`sbci.alignment` therefore defaults `delta` to `1e-5`, near the optimum for a
central difference in double precision; agreement with the reference is the
same at every step size, so nothing is given up. **This is a deliberate
divergence and needs WP1 sign-off.**

### A third weakness: one step length, and the first failure ends it

The reference moves by a fixed length in coefficient space each iteration
(`step`, 0.05 by default, along the normalized gradient) and stops the first
time a step fails to lower the cost. That works when the subjects are far
apart. When they are close -- a subject and a copy of itself deformed by a
degree, which is the size of anatomical differences after a good initial
registration -- the first step already overshoots, the cost rises, and the
reference returns the identity having done nothing. Measured on a synthetic
subject whose endpoints were moved by 1.0 degrees on average: cost 0.00935
before, 0.00935 after, at 20 and at 50 iterations. The port now halves the
step length up to `backtracks` times (four by default) before giving up, and
doubles it back after an accepted step. `backtracks=0` stops at the first
failed step, as the reference does; the reference comparison above ran that
way until item 21 removed it.

### How close is it possible to get?

The port's finite difference sits **0.0367** from the reference at every step
size from 1e-10 to 1e-6 (0.0366 at 1e-4, 0.0359 at 1e-3). The reference sits
0.0352 to 0.0420 from *itself* across adjacent step sizes. So the port is
already as close as the reference is to its own neighbouring configuration,
and that is the floor.

The residual comes from one place. Face agreement with libigl's closest-point
query is **100%** for ordinary points and about **85%** (82.7% to 87.0%) for
queries that coincide with a mesh vertex, at every step size -- and the offset
queries are all of that second kind. At a vertex every incident face is
exactly tied, so libigl returns whichever its tree reaches first. Closing that
gap means reproducing an AABB traversal order, not any mathematics. Choosing
the face by the distance to its plane, which ignores whether the point falls
inside the triangle, agrees on none of either kind, among all the faces or
those at the nearest vertex, so the port's rule is the right one. (The 13.6%
first quoted for the plane came from a sum of absolute coordinate products,
not a distance.)

### Both derivative operators are checked against known answers

`sbci.alignment.gradient_operators` differentiates the piecewise-linear
interpolant exactly, and is available as `derivative="analytic"`. Both it and
the reference's central difference are tested against closed forms:

| Test | Result |
| --- | --- |
| `f(x) = x` on a flat mesh | gradient exactly 1, error 3.6e-15 |
| `f = 3x + 2y` | exactly (3, 2), error 1.4e-14 |
| `z` along `e1` on the level-2, 3 and 4 icospheres, analytic operator | converges, 2.3e-02 -> 8.2e-03 -> 2.1e-03 |
| the same, the reference's central difference | converges, 3.4e-02 -> 8.8e-03 -> 2.7e-03 |

They estimate the same quantity and both converge, the analytic one somewhat
faster. How far apart they are depends on the smoothness of the field:

| Field | max difference / scale | correlation |
| --- | --- | --- |
| smooth analytic field | 0.019 | 0.99996 |
| kernel-smoothed noise | 0.049 | 0.99946 |
| a real smoothed connectome on ico4 | 0.40 | 0.9925 |
| white noise | 0.57 | 0.892 |

A density varying at grid scale does not determine its own derivative from
vertex samples, and real connectomes sit closer to the noisy end than one
would like. The default therefore stays with the reference estimator, and the
analytic one is opt-in. **An earlier draft of this note claimed the two were
different estimators rather than two accuracies of one; that was measured on
`rand()`, which has no derivative to estimate, and was wrong.**

### A defect inherited from the reference

The Jacobian is assembled in `(theta, phi)` coordinates and closes with a
factor of `sin(theta)`, which vanishes on the coordinate axis. **Any vertex at
a pole therefore gets a Jacobian of exactly zero**, and a density pushed
through the warp loses that vertex's entire row and column.

The ico4 grid this package ships has **four such vertices** -- 0 and 11 on the
left, 2562 and 2573 on the right -- so a naive run would have silently dropped
them. The sphere has no distinguished axis, so `rotate_off_poles()` turns the
mesh before the frames are built, and since item 18 `align()` turns a
supplied grid with a vertex on the axis or within 1e-3 of it (in
`sin(theta)`) clear of it too, recording the turn in every warp
(`grid_rotations`); it refuses only a grid none of its turns clears (a
`RuntimeError` until item 24). The
finite differences are unreliable there: a vertex 1e-5 off the axis read 1.27
for a 0.5-degree rotation (item 10). Worth reporting upstream: the reference's own demo grid happens to avoid
the poles, which is why this has not bitten anyone.

### Four more, found by the reviews of 5 October 2026

The Legendre derivative recurrence that builds the tangent basis has a wrong
m = 0 term, the transported square-root density is normalized before its
diagonal is zeroed, a warp that folds the mesh is accepted whenever the cost
falls, and the finite-difference Jacobian of the identity warp is 0.9965
rather than 1 on ico4 (0.805 on a 42-vertex icosphere: a discretization bias
of the scheme, which shrinks under refinement), so the aligned density lost
0.7% of its mass, once, when the final warp was applied to it. All four are the reference's,
reproduced here to rounding until the reviews; items 8 and 9 have the sizes.
The port now corrects them -- the Jacobian is calibrated by the identity's own
value, so a zero step gives exactly 1 and the transported mass is conserved to
1e-4 (to about 2e-4 on real subjects, before the zeroed diagonal loses 0.3% to 0.5%:
item 38). Until 11 October 2026 `reference=True` restored the first, second and
fourth for the comparison with MATLAB; since item 21 there is no such switch,
and the comparison puts the reference's choices back into the port's output
instead.

### Still open

Nothing since item 21. The registration loop descends a gradient built from
the unstable derivative, so two implementations take different steps, and
their agreement was reported as a correlation, not a tolerance; item 21
changed the order in which the port composes a step, and that comparison went.

The grid is no constraint: `SphericalGrid(mesh, l)` takes the mesh as an
argument, only the reference's demo scripts hardcode its retired 0.94 grid,
and this package uses the ico4 sphere throughout (SPEC_QUESTIONS.md item 12,
withdrawn).

## 5. FPCA reduction -- DONE, VERIFIED; LOCAL INFERENCE IMPLEMENTED WITHOUT A REFERENCE

- **From:**
  [`SBCI_Modeling_FPCA`](https://github.com/sbci-brain/SBCI_Modeling_FPCA)
- **To:** `src/sbci/reduction.py`, reached through `sbci.reduce()` and
  `ContinuousConnectome.reduce()`.
- `stats.local_test()` is **implemented, but to a weaker standard than
  everything else here.** **All ten `sbci-brain` repositories were searched**
  -- `SBCI_Toolkit`, `SBCI_Pipeline`, `SBCI_Py3`, `SBCI_Modeling_FPCA`,
  `ConCon_Alignment`, `ConCon_Alignment_1D`, `SBCI_Datasets`,
  `SBCI_documentation`, `CoCoNest` and `BridgeBP` -- and none contains
  inference code. There is no reference to run and nothing to diff against. What is implemented is ordinary linear-model inference on the
  component scores -- an F test per component, then Benjamini-Hochberg or
  Bonferroni across components, with a permutation option -- and it is verified
  against `scipy.stats` as an independent implementation and against the
  properties that define the procedures: p-values uniform under the null
  (Kolmogorov-Smirnov), false discovery held below the nominal rate in
  simulation, power rising with effect size, Bonferroni never less
  conservative than the FDR correction. `groups=` makes the test hold for
  related subjects, families of twins and siblings, with cluster-robust
  standard errors (*Related subjects* below). **This should be reviewed by
  whoever specified the API**, because the choice of test is mine, not the
  reference's.

### The dependencies were not the blocker

This file previously recorded FPCA as blocked because its MATLAB dependencies
were unobtainable. That was wrong on both counts.

| Dependency | Used by | Status |
| --- | --- | --- |
| `tensor_toolbox` | the algorithm, 12 calls | **clones fine** from GitLab; each call is one line of NumPy anyway |
| `splinepak` | `SplineBasis` only | not needed -- see below |
| `getLebedevSphere` | `SplineBasis` only | not needed -- see below |

The last two exist to build a spherical **spline** basis and the quadrature
that gives its Gram matrix `J` and roughness matrix `R`. This package does not
use that basis: its functions live on the ico4 grid, where the L2 inner product
is the Voronoi vertex areas -- verified to 1.6 float64-eps against libigl -- and
the roughness penalty is the cotangent Dirichlet energy. So the basis is the
identity on the grid and both matrices come from geometry already in hand.

### The real blocker was a bug

`ConConBasis.Fit` **cannot run as published**. At line 260 it reads
`auto_sparse`, a variable never defined; the parsed option is
`params.auto_sparse`. Every call fails with an undefined-variable error before
the first component is recorded. The reference run for this port applies a
one-line fix binding the name, and changes nothing else. *Worth reporting
upstream.*

### Agreement

With the same initialization and the reference run to its fixed point
(`tests/reference/fpca_fixed_point.m`: 400 updates, never stopped early), the
port reproduces the reference on every component. At the reference's default
tolerance they agree on the first vector, and the scores differ by design
(item 17): the reference keeps each component's scores from the vector before
its last update, the port takes those of the vector it returns.

| Quantity | At the fixed point | At the default tolerance |
| --- | --- | --- |
| component basis vectors | **9.7e-16** at most, all four, after sign alignment | 2.3e-16 for the first; 1e-9 to 6e-5 for the other three, fitted to what the scores leave |
| component scales | 6.9e-16 relative | 1.6e-5 relative |
| subject scores | 5.8e-16 | the first vector's own scores to 1.5e-15; the reference records the previous vector's, 2.4e-5 from them |
| explained fraction | 1.0e-15 | 8.7e-7 |

Until 7 October 2026 the port kept the reference's scores too, and matched its
default run to rounding (2.1e-16 on the basis). `tests/test_matlab_reference.py`
asserts the fixed point in full and, at the default tolerance, the first
vector and its scores. The tests in `tests/test_reduction.py` check what holds
without MATLAB: exact recovery of genuinely low-rank data, ordering by weight,
monotone explained variance, and scores that reproduce their own subjects.

### Fitting and projection were two objectives

`ConConSmooth.smooth` scores a new subject by least squares over the lower
triangle *with the diagonal*; `ConConBasis.Fit` scores the training subjects
by contracting the whole matrix. For a zero-diagonal connectome and one
component the two differ by exactly `1/(1 + sum_i psi(i)^4)`: two thirds on
a two-vertex toy, 0.9996 for a smooth component on ico4. The port's
`project()` reproduced `smooth`, and its docstring called the result directly
comparable with `Reduction.scores`, which it was not. Since the review of
5 October 2026 (item 8) it scores a new subject exactly as the fit scored the
training cohort -- the sequential deflation `c_k = psi_k' Y psi_k - sum_{l<k}
c_l (psi_k . psi_l)^2`, solved through the overlaps in `O(n^2 K)` -- so
projecting that cohort returns its scores, to rounding. Since 7 October 2026
the fit records the scores of the vectors it returns (item 17); until then it
kept the reference's, from the vector before each component's last update,
and the two agreed only to the fit's tolerance (the reference's triangle form
was off by 5e-2 to 1e-1 on the same cohorts). `project(..., reference=True)`
kept the reference's triangle objective until item 21 removed it.

### What the areas weight

`reduce()` passes the vertex areas as the Gram matrix, and they define the
inner product in which the components are made orthogonal and in which each
new component is deflated against the earlier ones. They do not enter the
leading component of a step: that is the Euclidean eigenvector of the
contracted matrix, as in the reference (matched to 2.3e-16 above), so the
objective is the unweighted fit on the grid, not the area-weighted
approximation error. On a four-vertex toy with areas 100, 100, 1, 1 the
returned component has twice the area-weighted residual of an alternative.
Whether the method should minimize the weighted error is WP1's decision; the
docstrings now say what the areas do.

### A trap in the roughness penalty

Each component is the eigenvector of `P (G - alpha R) P'` of largest
*magnitude* eigenvalue, which is what `eigs(M, 1)` returns, and the penalty
enters with a minus sign. Once `alpha * R` outweighs the data term the
largest-magnitude eigenvalue is the most negative one, so the fit returns the
**roughest** direction. Measured on a test cohort
(`tests/reference/fpca_alpha_sweep.py`), roughness falls 1.96 to 1.71
as `alpha` rises to 1, then jumps to 3.98 -- the maximum the penalty admits --
at `alpha = 5`, with the explained fraction collapsing from 0.83 to 0.01. The
reference's default of `1e-10` is far below the turn.

### One deliberate divergence

Above 400 vertices the port finds each component with ARPACK rather than a full
diagonalization. A full `eigh` on the 5124-vertex grid would cost hours for a
single vector. ARPACK is the library MATLAB's `eigs` itself calls, so the large
case is if anything closer to the reference; the two paths agree to 5.3e-15.

### A second divergence, off by default: where each component starts

The reference starts each component from a random vector (`normrnd`) put
through thirty power iterations on the mode-1 Gram matrix `sum_s R_s R_s'`,
then stops at the first stationary point of the alternating updates. The port
does the same by default, now from `seed=0` so that a run repeats. Measured on
the synthetic cohort (ten subjects, 20,000 streamlines, rank 4):

- twelve seeds agree to the last digit on the first three components and
  differ in the fourth, and about one start in fifteen misses a component the
  others find -- two jobs on identical data disagreed about whether the
  planted bundle was among the four;
- with the anatomy jittered by three degrees (`anatomy=0.05`) no seed of
  twelve finds the planted bundle. Started at the bundle, the fit returns it
  as a component of scale 2.65e-8 against 2.60e-8 for the component the
  default finds first, with adjusted p 0.0003: it is the largest single
  component in the data. It is the *sixth* eigenvector of the Gram matrix,
  whose top six eigenvalues lie within 30% of one another (1.26 down to 0.91
  e-15), so the power iteration lands on the first. At one and a half degrees
  it is the third eigenvector (scale 2.67e-8 against 2.46e-8) and the fit
  finds it only as the second to fourth component.

The Gram matrix ranks patterns by their Frobenius norm across subjects, which
favours a coherent shift of many bundles over a change of weight in one; the
objective ranks them by their best separable approximation, and the two
disagree exactly when several patterns are of similar size. `candidates=k`
computes the top `k` eigenvectors of the Gram operator by Lanczos (a few
seconds on ico4), runs the alternating updates from each and keeps the largest
component; `candidates=6` finds the bundle in both cases -- at three degrees
as the second component, with the largest scale of the four and adjusted p
0.0003, the explained fraction rising from 0.181 to 0.193 (the 0.182 and
0.194 first quoted here were a run before later changes; the code gives
these before and after item 17, `tests/reference/final_scores_probe.py`) -- at about three
times the cost of the default (312 s against 94 s for ten subjects at rank
4). The default stays at 1 because the reference has no such step; making it
6 is a WP1 decision.

### Speed

The fit is bound by memory traffic over the cohort: every application of the
mode-1 Gram operator reads all `S` matrices twice, and each outer step reads
them twice more. Three things that cost passes for nothing were removed, each
checked to give the same numbers -- the basis, scores and explained fractions
of a ten-subject ico4 cohort agree with the previous code to 1e-15, the
difference of BLAS summation order:

- the mesh inner product is `diag(areas)` and was stored as a dense `n x n`
  matrix, so every projection onto the complement of the kept components paid
  a full matrix-vector product for an elementwise scaling -- about a sixth of
  the fit;
- the subject weights and the contracted matrix were `einsum` contractions,
  about twice as slow as the equivalent BLAS forms `(R v) . v` and
  `tensordot(score, R)`;
- the explained fraction was measured by differencing the cohort against an
  untouched copy after every component; it is now the closed form
  `sum_jl (psi_j . psi_l)^2 (C' C)_jl` over the components removed, and
  `reduce()` deflates its own stack in place, so the cohort is held once
  (210 MB per subject in float64) rather than twice.

Ten ico4 subjects at rank 4 take about 40 s on four cores of a compute node,
down from 66 s, and 2.1 GB rather than 4.2 GB.

### Related subjects

The F test assumes independent subjects. The HCP Young Adult study recruited
families, twins and their siblings, and members of a family resemble each
other in their connectomes and in their traits; counting them as independent
makes every p-value too small. `local_test(..., groups=)` takes one family
label per subject and judges the tested terms by a Wald test with
cluster-robust standard errors, `c * B @ M @ B` with `B = (X'X)^-1`, `M` the
sum over families of each family's `X_g' e_g` outer product, and the usual
small-sample factor `c = G / (G - 1) * (n - 1) / (n - p)`; the F statistic's
denominator degrees of freedom are `G - 1`, for `G` families. Only the labels
are needed, not who is a twin of whom.

How it is verified:

- Against **statsmodels**, an independent implementation: `OLS(...).fit(
  cov_type="cluster", cov_kwds={"groups": ...}, use_t=True).f_test` gives the
  same statistic and p-value to 1e-9, for one tested term and for two
  (`tests/test_stats.py`; statsmodels comes with the `test` extra, and the
  test is skipped where it is not installed).
- Against the **sandwich written out family by family**, in a loop, to 1e-10.
- **Under the null**, in simulated cohorts of families of one to four in which
  the covariate and the score are both shared within a family (intra-family
  correlation 0.69 in each) and unrelated to each other, 10,000 independent
  cohorts for each number of families (`tools/clustered_null.py`):

  | families | rejected at 0.05, as independent subjects | with families as clusters |
  | --- | --- | --- |
  | 20 | 0.166 | 0.097 |
  | 50 | 0.161 | 0.067 |
  | 100 | 0.156 | 0.060 |
  | 422 | 0.163 | 0.051 |

  Ignoring the families triples the false-positive rate; clustering holds it
  at the nominal rate with hundreds of families, as in the HCP analysis, and
  not with dozens, as is known of cluster-robust tests. The test suite checks
  a smaller version (150 families: 17.1% against 5.9%).
- **Permutation is refused with groups.** Freedman-Lane shuffles subjects
  freely, and shuffling between families of different make-up, twins against
  siblings, is not exchangeable. A permutation test for family data needs the
  structure inside each family, who is a monozygotic twin of whom and who
  shares which parents, as the multi-level block permutation of Winkler et
  al. (NeuroImage, 2015) does with the same restricted table; it is not
  implemented.

### Measured on the HCP Young Adult cohort

The 946 young adults with complete pipeline output (item 6, *The HCP Young
Adult cohort, rebuilt on ico4*) are the association example of docs/RESULTS.md: fluid
intelligence, the number of Penn Matrix Test items answered correctly
(`PMAT24_A_CR` in the HCP's open-access table, 4 to 24), which 943 of them
have. They come from 423 families, and 850 of them have a relative among the
946; the 943 come from 422. Every test below treats the families as clusters.
The first version of this analysis did not, and reported a component as
significant that is not; its numbers are kept beside the corrected ones,
because the difference is the point. Family membership is in the HCP's
restricted table: it was read on Longleaf from a private copy; only counts
drawn from it are in the repository, and no document that reports them names a
subject by its HCP id (item 33).

`tools/age_probe.py --table ... --column ... --groups ...` streams the cohort
once, in fifteen minutes on four cores (`--measures` repeats the tests on the
saved measures in seconds), and tests each measure against the trait alone. So
tested, the trait shows weakly, far more weakly than sex; with sex, the age band
and the streamline count in the model 26 vertices remain, and with head size as
well none (item 33):

| measured on each subject | fluid intelligence | counted as independent | sex |
| --- | --- | --- | --- |
| area-weighted strength of each cortical vertex | 113 of 4,685 significant at FDR 0.05, \|r\| up to 0.17 | 217 | 1,020, \|r\| up to 0.31 (1,177 counted as independent) |
| each Desikan region pair | none of 2,278, \|r\| up to 0.16 | 52 | 761, \|r\| up to 0.35 (843) |
| interhemispheric share of the connectivity | r = 0.04, p 0.30 | p 0.29 | r = 0.14 |
| long-range share, over 50 mm | r = -0.03, p 0.41 | p 0.37 | r = 0.28 |
| the leading eight principal directions, vertex level | \|r\| up to 0.16 (the fifth) | | up to 0.22 (the second) |

The streamline count, 0.60 to 1.07 million per subject, correlates with the
trait at r = -0.08 and with the fourth and fifth vertex-level directions at
-0.36; the trait correlates with sex at -0.14.

At rank 20 (`candidates=1`, 27% of the cohort's norm), with sex, the age band
and the streamline count as covariates, no component tracks the score after
the false-discovery-rate correction across the twenty. The closest are
component 13, r = 0.11, adjusted p 0.064, and component 7, r = 0.10, adjusted
p 0.082; without the covariates both are at 0.051. Counted as 943 independent
subjects, component 13 passes at 0.024 (and component 7 at 0.029 without the
covariates); the first version of this analysis counted them so and reported
component 13, with an effect map in right medial occipital cortex. A check
that needs no clustering agrees with the correction: keeping one subject per
family, the first met in a random order
(`numpy.random.default_rng(seed).permutation`, seeds 0 to 4), leaves 422
independent subjects, in whom component 13 correlates with the score at
r = 0.10 to 0.12 with adjusted p 0.22 to 0.43
(`tests/reference/inference_probe.py`, step 4). The effect's size holds; its
significance does not.

At rank 4, the first four of the same components (17% of the norm), nothing
comes close (smallest adjusted p 0.33), while sex shows in all four. At rank
20 sex, tested with the score among the covariates, shows in thirteen
components (1, 2, 3, 5, 6, 8, 9, 10, 12, 15, 16, 18 and 19; the same thirteen
counted as independent), the strongest at adjusted p 1e-7. Its effect map
over those thirteen (`docs/figures/cohort_sex_effect.png` until item 32) sits at the
occipital poles of both hemispheres, where connectivity is relatively higher
in men, and nowhere else above a twentieth of its largest value. Head size
differs between the sexes and is not in the model, so part of that may be
size. (Item 32 puts it in the model: sex then shows in three components, at
the right cuneus and pericalcarine cortex.) The streamline count matters less here than the rank does: it
correlates with the first component at -0.12 and with no other beyond 0.18.
The fit read the cohort one file at a time, held it as 199 GB of float64 and
peaked at 196 GiB, and took 5.4 hours on eight threads, loading included; the
refit of item 17 took 4.2 hours on another node.

These are the refit's numbers (7 October 2026), with each subject's scores
taken from the final map of its component (item 17). The fit before it, whose
scores lagged by a step, put sex in eleven components (1, 2, 3, 5, 6, 8, 9,
12, 15, 16 and 18), the strongest at 2e-8; component 7 at 0.083 for the trait,
component 13 at 0.023 counted as independent and at up to 0.48 in one subject
per family; and the smallest rank-4 adjusted p at 0.32.

### Estimates, intervals and contrasts (6 October 2026)

A request of 6 October 2026 asked for effect estimates with confidence
intervals, and for designs and contrasts written by name.
`sbci.stats.design` codes named covariates, text as factors against a
reference level, into a `Design`; `local_test` takes names in `terms=` and a
`contrast=` of the coefficients. Both are one linear hypothesis, ``C beta =
0``, whose reduced model is the design restricted to the null space of ``C``,
so the F test, its permutation version and the cluster-robust Wald test serve
both. The result carries the tested estimates with classical or cluster-robust
standard errors, t intervals, and the partial R-squared. The `terms=`
arithmetic is unchanged: index-based results are identical.

Checked against statsmodels (`tests/test_stats_design.py`): coefficients,
standard errors and intervals, cluster-robust ones included, to 1e-9; the F
tests of single and joint contrasts, classical and robust, to 1e-8. On the 943
young adults (`tests/reference/inference_probe.py`), the named design gives
the components RESULTS.md publishes for sex bit for bit (13 of 20 since the
refit of item 17, 11 before it, and 3 since item 32 put head size in the
model), and the published 1,020
and 113 vertices of the per-vertex tests of sex and fluid intelligence. A
contrast against a level as small as the 36+ age band, 9 subjects, leans on
cluster-robust errors from a handful of families; USAGE.md says so.

### Memory on a large cohort

`reduce` collected every subject's dense matrix in a list and then stacked
them, so a large cohort was held twice at the peak. It now fills one
preallocated array, and a sequence that loads each connectome when indexed is
read one subject at a time. Basis, scores, scales, explained fractions and
mean of a six-subject ico4 cohort are bitwise identical to the previous code,
as objects and as arrays, with both starts. On Longleaf a fit runs on one core
when eight one-CPU tasks leave `OMP_NUM_THREADS=1`; one task with eight CPUs
gives the linear algebra its threads.

## 6. Spherical kernel -- DONE, r = 1.000000 AT FULL SCALE, BUT FOR THE SOUTH POLE THE BINARY NEVER WRITES

- **From:** [`dcmoyer/concon`](https://github.com/dcmoyer/concon), C++, MIT.
  Invoked by `sbci_step5_structural.sh` as `c3_main Compute_Kernel --sigma
  ${BANDWIDTH} --epsilon 0.001 --final_thold 0.000000001 --OPT_VAL_num_harm 33
  --OPT_VAL_exp_num_kern_samps 6 --OPT_VAL_exp_num_harm_samps 5`.
- **To:** `src/sbci/smoothing.py`, reached through `cc.smooth(kernel="shk")`.
  Against `c3_main` itself the port reaches **r = 1.00000000** at full
  scale, but for the south pole of each hemisphere, which the binary never writes (*Full
  scale*, below); the
  0.65 this file reported for most of its life was measured against the wrong input.

### The verification that was missing, and now exists

Every earlier comparison used `mesh_intersections_ico4.mat` as the input and
`smoothed_sc_avg_0.005_ico4.mat` as the target. The pipeline script shows those
come from different branches, so they were never the same data.

What was needed is a pairing from one run: `subject_xing_sphere_avg_coords.tsv`,
exactly what `c3_main` was fed (both endpoints of every streamline as
continuous unit-sphere positions), beside what `c3_main` wrote from it and the
grid it was handed (`SBCI_AVE/lh_grid_avg_ico4.vtk`, which matches the bundled
ico4 sphere to 6.0e-12). The lab keeps such pairings for some of its ADNI
subjects, whose data are shared under ADNI's own agreement. The port was first
checked on those, and the reference scripts that use them are kept, but this
file no longer names the subjects or gives the numbers measured on them (item
39). The full-scale comparison it reports instead runs `c3_main` on a released
young adult's stored endpoints (*Full scale*, below).

Recomputing the density from such a pairing at sigma = 0.005 with 33 harmonics
left two differences, which the rest of this item traces.

**The global scale** was a normalization convention: that draft
normalized each endpoint's kernel column to sum one, while `convert_raw.py`
applies no normalization at all -- its normalizing block is commented out in
the source. The shipped port follows `convert_raw.py` and needs no scale.

**The residual after scaling is compact support.** An earlier draft of this
section said "the support difference is not the issue", on the grounds that
this port placed little of its mass outside the *released file's* support.
That was the wrong comparison. The released file's support is the union over a
million streamlines and says almost nothing about the support of the *kernel*,
which is far tighter -- and it is the kernel's support that differs.

### Asking concon directly

`c3_main` is an ordinary x86-64 Linux executable, it runs, and it is not
stripped. That makes the kernel measurable rather than inferable. Feed it a
single streamline from p to q and its output is

    D(i, j) = K(theta_i) * K(theta_j)

so the stored entries over-determine K: `log D(i, j) = f(theta_i) + f(theta_j)`
is a linear system for `f = log K` which many runs solve on a fine angular grid
with no per-run normalization to guess. `tests/reference/concon_probe.py` does
this; it needs the binary and the lab grid files, so it is run by hand.

**concon's kernel has compact support.** One streamline at sigma = 0.005 yields
2,046 stored entries -- a 64 x 64 outer product, not the 13 million a
whole-sphere kernel would give. Measured:

| sigma | support | vertices | support / sqrt(sigma) |
| --- | --- | --- | --- |
| 0.0025 | 7.93 deg | 16 | 2.769 |
| 0.0050 | 11.89 deg | 31 | 2.936 |
| 0.0100 | 16.53 deg | 61 | 2.885 |

So the kernel vanishes at about **2.9 sqrt(sigma) radians**, and the draft's
kernel, which spread over the whole sphere, was wrong everywhere beyond that.
It placed 10.7% of its mass there. That is exactly the deficit the distance
analysis kept finding at 12-60 degrees and could not attribute to bandwidth or
truncation.

**`--epsilon` is not what does it.** Swept over 0.0001, 0.001, 0.01 and 0.1 the
support is identical at 11.894 degrees, to the vertex. Whatever that argument
controls, it is not the cutoff -- which removes the leading suspect this file
carried for months.

**Inside the support there is a taper.** Against the recovered profile, the bare
truncated heat kernel has rms error 0.085; multiplying by a window that
vanishes at the cutoff drops that to **0.010-0.013** for any reasonable window
shape. The taper turned out to be the lookup tables themselves (*The kernel,
resolved*, below); the candidates at the time were
`--OPT_VAL_exp_num_kern_samps 6` and `--OPT_VAL_exp_num_harm_samps 5`: the
symbols `c3::Subject::calc_kern_lookup_table(double, int)` and
`calc_harm_lookup_table(int, int)` confirm the kernel is tabulated rather than
evaluated in closed form, and 2^5 = 32 matches `--OPT_VAL_num_harm 33`.

The binary also links Google's `spherical-harmonics` library --
`sh::EvalSH`, `sh::EvalSHSlow`, `sh::EvalLegendrePolynomial` -- which is where
the harmonic evaluation convention should be read from.

**The taper is not an artifact of the tabulation.** Raising
`--OPT_VAL_exp_num_kern_samps` from the pipeline's 6 to 7 and 8 leaves the
kernel bit-identical -- 1.0000, 0.7432, 0.4436, 0.2751, 0.1035 across the
rings -- and only dropping it to 3 changes anything. The support stays at
11.894 degrees to the vertex throughout. So concon's kernel is a
compactly-supported kernel by definition, not a coarsely-sampled heat kernel,
and the port needs the real functional form rather than a finer evaluation of
the one it has.

### Confirming it on the released data

The draft's kernel was multiplied by a window that vanishes at the measured
cutoff, `1 - u^3` or `1 - u^4` with `u = theta / (2.9 sqrt(sigma))`, and
compared again with a pairing of `c3_main`'s input and output (item 39). The
deficit at 12-60 degrees that no bandwidth and no threshold had removed went
away, and the ratio against the released matrix flattened across the distance
bands -- a flat ratio means the shape is right, which correlation alone does
not test. The windows were approximations chosen to test the diagnosis, not the
real taper, which the next section identifies.

### The kernel, resolved

There is no taper. `concon` is public MIT code and the binary's debug info
names its files, so the kernel could be read rather than fitted.
`sigma_opt.cpp` sums `exp(-sigma*l*(l+1)) * (2l+1) * harm_lookup[l][x]`, and
`subject.cpp` fills that lookup with `sh::EvalSH(l, 0, 0, acos(x))` -- the
*normalized* spherical harmonic `Y_l^0`, which already carries
`sqrt((2l+1)/4pi)`. The two factors compound, so the weight is

    (2l+1)^(3/2) / sqrt(4 pi)

and **not** the heat kernel's `(2l+1)/(4pi)`. It looks like a Legendre
polynomial was intended where a normalized harmonic was used. Intended or not,
it produced every released cohort, so it is what this package reproduces. The
extra `sqrt(2l+1)` upweights high degrees, which is the whole reason concon's
kernel is so much narrower than a heat kernel -- and the reason the port's
kernel could not match the binary's for months.

The term count is 33, degrees `l = 0..32`, fixed by measuring the binary's
peak at five bandwidths rather than by reading: at sigma 0.01 and above the
series has converged and the count cannot matter, and all candidates agree;
below, they separate, and 33 tracks the binary to 0.06-0.1% where 32 drifts to
4% and 34 overshoots. `--OPT_VAL_num_harm` is ignored either way.

The cutoff does come from `--epsilon`: `compute_kernel.cpp` scans
`x = cos(theta)` down from 1.0 in steps of 1e-4 and stops where the kernel
first falls below it. The descent there is steep enough that 1e-6 to 0.1 all
give the same angle, which is why sweeping the flag against the binary looked
inert. `compute_kernel.cpp` also divides by the streamline count, which is why
the normalizing block in `convert_raw.py` is commented out.

### Full scale, on a released young adult

`tests/reference/concon_fullscale.py` writes the stored endpoints of the
tutorial subject's released file, all 803,741 streamlines, as `c3_main`'s input,
runs the binary with the pipeline's flags, and compares what it writes with the
package's density from the same positions (job 5989183):

| | as computed | the south pole emptied |
| --- | --- | --- |
| correlation, all 13,125,126 pairs | 0.99910256 | **1.00000000** |
| best scale | 0.998245 | **1.000000** |
| relative error, nothing rescaled | 0.041928 | **0.000006** |
| non-zero pairs | 6,892,526, against the binary's 6,887,499 | **6,887,499**, the same |
| median ratio by distance band | 1.000001 throughout | 1.000001 throughout |

**`c3_main` never writes the grid's south pole.** The vertex at (0, 0, -1),
vertex 11 of each hemisphere, in the left fusiform and the right inferior
temporal cortex, gets nothing from the binary -- in this run, and in the
toolkit's example output alike -- while the package gives it what the kernel
puts there: 0.072% of the tutorial subject's mass, in the 5,027
pairs the binary lacks. The cause is in concon's source (commit `ce7e0d9`):
`compute_kernel.cpp` looks up the vertices near an endpoint in a quad tree over
the polar angle, clipping the range it asks for at pi, and `quad_tree.cpp`'s
range test is half-open, `x < max_x`, so a vertex at exactly pi is never
returned. The grid stores each pole as exactly (0, 0, -1), which the hole
depends on; and the binary reads `--LOAD_grid_file` for both hemispheres,
ignoring `--LOAD_rh_grid_file` (`main.cpp`), which is harmless because the
lab's two `.m` files are identical, as the bundled sphere's halves are. Every
connectome `c3_main` made has that hole. The package, which finds neighbours
with a KD-tree, has none, and keeps it that way: the hole is the reference's
error, put back only for the comparison, as the MATLAB references' are. With
the pole emptied, the two write the same pairs and agree to 6e-6, and to
machine precision once the binary's own two inaccuracies are put back as well
(*Shipping it*, below).

The port kept more pairs than `c3_main` until `apply_final_threshold`
reproduced `--final_thold 1e-9`, the per-streamline value below which the
reference writes nothing. Earlier versions of this item put the pairs that
remained on the kernel's cutoff boundary; on the tutorial subject every one of
them is the south pole's.

### Endpoints from the snapped streamlines

A connectome has to be stored with the endpoints it was smoothed from: the
pipeline's `mesh_intersections_ico4.mat` holds the other, unsnapped branch
(*Confirmed from the pipeline itself*, below), and endpoints read from it
re-smooth to a different connectome from the one beside them.
`Endpoints.from_snapped` builds the right set. Each end of a snapped streamline
is a vertex of the subject's native white surface, and the registered sphere
(`?h_sphere_reg_lps.vtk`, one vertex per native vertex) says where it lands;
streamlines with an end on any other surface are dropped, as
`intersections_to_sphere.py` drops them.

`tools/build_hcp_cohort.py` takes its endpoints this way for the HCP Young
Adult subjects of the ENCORE project, whose FreeSurfer-registered spheres are
stored in RAS and negated in x and y to reach the grid's frame (as stored they
sit 121 degrees from the subject's native LPS sphere on median, flipped 12 to
14). `--check` re-smooths each file's endpoints and compares them with its
stored connectome.

### The HCP Young Adult cohort, rebuilt on ico4

The lab's HCP Young Adult connectomes are on the retired 0.94 grid (4,121
vertices), so the young adults the package distributes were not converted
from them. `tools/build_hcp_cohort.py` rebuilds each
subject on ico4 from the ENCORE project's copy: the snapped streamlines,
placed on the subject's FreeSurfer-registered sphere and negated in x and y as
above, smoothed by the package's `shk` kernel at bandwidth 0.005 with the
medial wall masked, and stored with those endpoints. Nothing is resampled
from the 0.94 grid. The frame was checked before anything was released:

| check, on the ten released subjects drawn at random | measured |
| --- | --- |
| stored endpoints re-smoothed, against the stored connectome | r = 1.000000 for every subject |
| share of endpoints on the medial wall | 0.1% to 0.3% |
| correlation between two young adults' connectomes | 0.67 to 0.71 |
| distance from a seed vertex to the peak of its profile | a median of about 12 degrees |

All 946 young adults with complete pipeline output were built the same way,
47.2 GB (44 GiB), and an audit read every file back. All 946 pass `sbci validate` and
carry positioned endpoints, with 599,553 to 1,071,113 streamlines each (median
823,732), and every connectome correlates with the mean of five released
subjects at 0.73 or more (median 0.82).

The medial wall takes a median 0.21% of a subject's endpoints, but 67 subjects
put more than 1% there and 22 more than 4%, up to 7.9% in one subject. In the
three examined, nearly all of it is in one hemisphere: 12 to 13% of that
hemisphere's endpoints on the registered sphere, against 0.2 to 0.4% in the
other. The subject's own unregistered sphere puts 9 to 12% of them there as
well, so these streamlines end on the medial wall of the subject's surface, and
the registration is not the cause: it sits 11 to 20 degrees from the native
sphere, as in every other subject checked. The mask drops those endpoints, and
the subjects stay in the analysis. Their connectomes correlate with the
example mean at 0.77 to 0.83, against 0.79 to 0.84 for twelve others drawn at
random.

### The public API path, end to end

The checks above validated the kernel through a hand-written density loop. Driving
`ContinuousConnectome.smooth(kernel="shk")` instead -- building a connectome
from `mesh_intersections_ico4.mat`, calling the method, saving, validating --
is what found the three bugs *Shipping it* records, and the face-order one could
not have been found any other way.

Against `c3_main`'s output on a pairing from the lab's archive (item 39) it
fell short of the match above, and that was the input file, not the port. On
the same rows, so that sampling noise is identical and only the endpoints
differ, endpoints read from `subject_xing_sphere_avg_coords.tsv`, what
`c3_main` was fed, matched, and those read from `mesh_intersections_ico4.mat`
did not. The two endpoint sets mostly agree to about a degree but have a tail
of endpoints placed far apart, and moving every endpoint by the typical
difference costs almost nothing, so it is that tail of relocated endpoints that
accounts for the gap (*Endpoints from the snapped streamlines*, above).

### Shipping it

`smooth(kernel="shk")` no longer raises. Getting from "the kernel is right" to
"the method works" took three more bugs, none of which the 554-test suite
caught, because none of its tests had barycentric endpoints:

1. `endpoint_positions` read `endpoints.n_faces_per_hemi`, which `Endpoints`
   does not carry -- `from_global` takes it as an argument and drops it. The
   offset now comes from the mesh itself.
2. The in-place Legendre recurrence (below) refused a scalar argument, because
   a scalar is a 0-d array and numpy will not use one as an `out=` target.
3. **The bundled surfaces' faces were in the wrong order.** See
   SPEC_QUESTIONS.md item 13: stored triangle indices refer to the pipeline
   grid's face list, the package shipped a different order of the same
   triangles, and every index resolved about 85 degrees from where it should.
   This was latent in the format handling long before the kernel was touched.

**The closed form's amplitude offset is explained and removed.** `c3_main` never
evaluates the series at a point. `compute_kernel.cpp` reads
`kern_lookup_table[(int)(dot * M + M)]`, a table of `2M+1` values with
`M = 10^6` (`subject.cpp`: `num_kern_samps = pow(10, exp_num_kern_samps)`),
and that table was filled by reading `harm_lookup_table[l][(int)(x * L + L)]`
with `L = 10^5`. Both reads truncate toward the sample at or below the true
cosine -- a slightly larger angle, a slightly smaller kernel -- and the bias
grows with `l(l+1)`, which is exactly the bandwidth trend the exact series
showed (0.99877 at sigma 0.00125 rising to 0.99985 at 0.02). Reproducing the
tables reproduces the binary: on a single streamline its peak entry is
`K(1.0) * K(1 - 1e-12) = 270.2668 * 269.9403`, whose square root is the
measured 270.1035 to six digits, and likewise at all five bandwidths.
`sbci.smoothing.KernelTable` is that emulation and is the default;
`quantized=False` gives the closed-form series the tables approximate.

At full scale, on the tutorial subject with the south pole emptied as in *Full
scale* above, the two paths side by side (job 5989183):

| | tables (default) | closed form |
| --- | --- | --- |
| correlation, all pairs | **1.00000000** | 1.00000000 |
| best scale | **1.000000** | 0.998591 |
| relative error, nothing rescaled | **0.000006** | 0.001413 |
| pairs equal to 1e-6 relative | **3,092,070** | 1 |
| median ratio by distance band | 1.000001 throughout | 0.99834 to 0.99851 |

The residual of 6e-6 is not the binary's arithmetic, as this item first said
(a one-ULP shift of a cosine moves about 2e-10 of the run's lookups across a
bucket, a hundredth of one lookup in all), but two inaccuracies of the binary,
found by the review of 22 October (item 41). `subject.cpp` reads the endpoint
coordinates with `fscanf("%f")` into `float`, so the binary smooths
single-precision positions, up to 3e-8 off a coordinate, where the package was
handed the nine-decimal doubles. And its harmonics come from Google's
spherical-harmonics library, whose closed forms for degrees 0 to 4 carry
six-significant-digit constants (0.282095, 0.488603, 0.315392, 0.373176,
0.105786; relative errors up to 4e-6) where `concon_kernel_table` uses the
exact `sqrt((2l+1)/4pi)`: the kernel table's peak moves by 1.2e-7 and its
tail near the cutoff by 2e-3, which is why the peak was only ever matched to
six digits. Both are put back test-side in `concon_fullscale.py`, as the
MATLAB references' errors are, and compared with the recorded run's output
(job 6186083):

| the package's density, south pole emptied | relative error | pairs within 1e-6 | largest difference |
| --- | --- | --- | --- |
| as shipped: double positions, exact constants | 0.000006 | 3,092,070 of 6,887,499 | 0.60 |
| the positions cast to float32, as the binary reads them | 5.9e-7 | 3,475,156 of 6,887,499 | 2.1e-3 |
| float32 positions and the library's six-digit constants | 1.05e-15 | 6,887,499 of 6,887,499 (all 6,887,499 within 1e-12, 2,736,226 bit-identical) | 6.4e-13 |

So the table emulation is the binary's, and the package keeps the exact
constants and the double positions, as it keeps the pole.

The **excess of non-zero pairs** was mostly `--final_thold`: `compute_kernel.cpp`
writes a pair only `if(temp > final_thold)` after dividing by the streamline
count, and `smooth()` now applies the same rule on the same scale. The rest is
the south pole (*Full scale*, above): with it emptied, the package writes the
binary's 6,887,499 pairs and no others.

### Speed

`_legendre_series` accumulated with `total = total + weight[l] * current`,
allocating a fresh 168 MB array per term over a `(5124, 4096)` block, a few
hundred times per subject. Reusing three buffers made it **5.8x faster**
(28.98s to 5.02s per block) and changed nothing beyond float reordering, 2.3e-12.

`endpoint_density(method="sparse")`, now the default, finds each endpoint's
neighbourhood with a KD-tree over the grid (per hemisphere, so the hemisphere
rule is enforced by construction) and evaluates the kernel only there -- about
31 of 5,124 vertices per endpoint, since the kernel is zero past 12.2 degrees.
With the kernel itself a table lookup, the evaluation is a gather. The dense
path is kept as `method="dense"` and `test_sparse_density_matches_dense`
holds the two to 1e-9 of the peak. A `progress(done, total)` callback reports
each block. On a subject of about a million streamlines the density took
**53 s**, and `cc.smooth(kernel="shk")` end to end **54 s**, against 3,184 s for
the dense exact path and 6,114 s before the in-place recurrence -- about 110x.
The tutorial subject's 803,741 took 15 s (job 5989183), where the binary
took 2,975 s.

### What this cost, and what it bought

For most of this file's life the disagreement was recorded as a normalization
or sampling convention, and three separate hypotheses -- bandwidth, spectral
truncation, kernel thresholding -- were each tested and eliminated against the
released matrix. None of them could have found it, because the released matrix
is a million kernels summed together and is a weak instrument for inspecting
one. What found it was running `c3_main` on a *single* streamline, where the
output is the kernel itself; the weight was then confirmed from the source.
`tests/reference/concon_probe.py` keeps that measurement reproducible.

> The subsections from here to *What to ask -- answered* are the earlier
> analysis, kept as the record of how the answer was reached; *The kernel,
> resolved* and *Full scale* above supersede them where they disagree.

### What the port gets right

The closed form in Legendre polynomials is implemented and `--sigma` is
identified as **diffusion time**, `exp(-l(l+1) sigma)`: it beats the two
alternative readings on correlation and rank correlation, and its support lands
within 2.4% of the released file's 5,079,082 non-zero pairs.

### What has been ruled out

| Hypothesis | Test | Result |
| --- | --- | --- |
| different data | median released value against streamline count | **same data** -- rises 1.26, 3.89, 7.08, 11.47, 17.99 across count bins |
| support overlap is coincidence | shuffled-endpoint control | 96.8% observed against a 52.7% floor |
| wrong vertex ordering | angular separation of streamline ends | 16.9 deg median against 90 deg chance -- ordering correct |
| wrong bandwidth | scan 0.002 to 0.02 | flat 0.57-0.65, peak 0.6513 at 0.008; no peak near 1 |
| wrong truncation | 17 to 65 harmonics | identical to four decimals; converged |
| wrong bandwidth convention | scan the decay constant 0.005 down to 0.0005 | **0.005 is right** -- flatness of the ratio against distance worsens monotonically, 1.77 to 79.06, as the kernel narrows |
| thresholding the kernel | cutoffs 1e-4 to 0.1 of the peak | helps the far band, but the 12-60 deg deficit holds at ~0.65 at every cutoff |
| `--epsilon` sets the support | sweep 1e-4 to 0.1 against `c3_main` | **no effect** -- support identical to the vertex |
| missing per-vertex Jacobian | fit log(released/mine) = g_i + g_j | explains 13.5% of variance, and applying it makes r *worse* (0.29) |
| endpoints snapped to vertices | recompute at continuous barycentric positions | **no effect** -- r = 0.6438 against 0.6446 snapped |

The disagreement is also uniform rather than localized: pairs carrying their own
streamline agree at r = 0.56, pairs whose value is smoothing alone at r = 0.62,
and the two estimates put nearly the same share of total mass in each group
(21.5% against 24.2%).

### The structural difference, resolved -- and it was not the answer

ConCon evaluates the kernel at **continuous endpoint positions**; the first
implementation snapped each endpoint to its nearest grid vertex. An earlier
draft of this file recorded that as blocked, claiming the barycentric triangle
indices in `mesh_intersections_ico4.mat` "do not refer to the grid mesh".
**That was wrong.** The indices run 1..5120, which is exactly the grid's face
count per hemisphere; what was wrong was the face list they were checked
against.

| Face list | Median angle to `vtx_in` | Heaviest corner is `vtx_in` |
| --- | --- | --- |
| `lh/rh_grid_avg_ico4.vtk` (used by the earlier check) | 82.558 deg | 0.06% |
| `lh/rh_sphere_avg_ico4.vtk` | 82.558 deg | 0.06% |
| **`template_sphere_grid_ico4.mat`** | **1.618 deg** | **97.16%** |
| **`lh/rh_white_avg_ico4.vtk`** | **1.618 deg** | **97.16%** |

The grid and sphere VTK exports triangulate the same vertices in a different
face order. With the right face list the barycentric weights reconstruct each
endpoint's continuous position, 1.534 degrees from its snapped vertex on
average.

**Recomputing with those positions changes nothing.** At sigma = 0.005, over
the strict upper triangle against the released file:

| endpoints | r | non-zero pairs |
| --- | --- | --- |
| snapped to grid vertices | 0.6446 | 13,125,126 |
| continuous barycentric positions | 0.6438 | 13,125,126 |

A 1.5 degree displacement against a 5.7 degree kernel was never going to
matter, and it does not. The discretization is now ruled out alongside the
data, the ordering, the bandwidth, the truncation and the Jacobian.

The positions are kept anyway: the HDF5 format now stores `triangle_*` and
`barycentric_*` beside the endpoint vertices (SPEC_QUESTIONS.md item 8), so
any future estimator that wants off-grid positions has them.

### The old r = 0.65 analysis, and why it was answering the wrong question

ENCORE's own MATLAB reference for spherical smoothing,
`ConCon_Alignment/scripts/kde/spherical_kernel.m`, was implemented as well: a
Gaussian in geodesic degrees, row-normalized, truncated by distance -- which is
what `--epsilon` controls. It reaches **the same ceiling**, r = 0.6528, and the
way it gets there is the important part.

| FWHM | r | non-zero pairs |
| --- | --- | --- |
| 8 deg | 0.2788 | 5,705,657 |
| 20 deg | 0.6014 | 12,262,809 |
| 33 deg | **0.6528** | 13,124,419 |

The released file has 5,079,082 non-zero pairs. The bandwidth that *matches
that sparsity* gives r = 0.28; the bandwidth that maximizes r smears mass over
essentially every pair, 2.6 times the released support. Row normalization and
distance truncation change the answer by 0.0014, so neither is the missing
ingredient.

Two unrelated kernel families -- a truncated spherical-harmonic heat kernel and
a geodesic Gaussian -- both plateau at 0.65 by over-smoothing. That is what two
heavily blurred versions of the same endpoints correlate at regardless of
kernel.

That analysis was sound and its conclusion was right in the narrow sense: the
released file is not `K A K` of **those** endpoints under any simple kernel.
What it could not see is that the endpoints were the wrong ones. Given the
input `c3_main` was actually given, the same implementation reaches 0.978. The
lesson is that a long chain of careful eliminations can still be conditioned on
an assumption nobody checked -- here, that two files in the same example folder
came from the same pipeline stage.

### The likely explanation, and what to check

`sbci_step5_structural.sh` runs two steps between the raw intersections and the
smoother: SET filtering into `snapped_fibers.npz`, then
`intersections_to_sphere.py` into `subject_xing_sphere_avg_coords.tsv`, and
only that last file reaches `c3_main`.

`mesh_intersections_ico4.mat` is the toolkit's demo input for
`rdk_smoothed_concon_compute.m`. It is very likely a **different stage of the
pipeline** from whatever produced the released file -- same subject, hence the
streamline counts tracking, but not the same array of endpoints. That would
explain every observation at once: the data matches, the ordering matches, the
support overlaps, and no kernel reproduces the values.

Before any more kernel work, confirm with whoever ran the pipeline whether
`mesh_intersections_ico4.mat` and `smoothed_sc_avg_0.005_ico4.mat` come from
the same stage of the same run. If they do not, this file is simply the wrong
reference and a correct port cannot be checked against it.

**Confirmed from the pipeline itself.** `SBCI_Py3` was not checked out when
the above was written; its `scripts/ADNI_example/sbci_step5_structural.sh`
settles it. The two files come from **different branches**:

```
snapped_fibers.npz
  -> concon/intersections_to_sphere.py   (high-resolution registered sphere)
  -> c3_main Compute_Kernel
  -> convert_raw.py
  -> smoothed_sc_avg_0.005_ico4.mat      <-- the released file

set_filtered_intersections.npz           (before snapping)
  -> intersections_to_sphere.py          (a different script, different surface)
  -> get_fibers_barycentric.py           (barycentric against the ico4 grid)
  -> mesh_intersections_ico4.mat         <-- the toolkit demo input
```

The released file is built from **snapped** fibers on the high-resolution
sphere; the demo input is built from the **unsnapped** filtered intersections,
mapped barycentrically onto the grid. `snap_fibers.py` sits on one branch and
not the other, so the two describe different endpoint sets.

**No kernel applied to `mesh_intersections_ico4.mat` can reproduce
`smoothed_sc_avg_0.005_ico4.mat`, because they are not the same data.** The
r = 0.65 ceiling was never a porting failure; the comparison target was wrong.

**What actually verified the port:** a subject's
`subject_xing_sphere_avg_coords.tsv` together with its `smoothed_sc_avg_*.mat`
from the same run, found later in the lab space -- *The verification that was
missing, and now exists*, at the top of this item.

### What to ask -- answered

The three parameters that were unaccounted for are now read from the source:
`--OPT_VAL_exp_num_kern_samps 6` and `--OPT_VAL_exp_num_harm_samps 5` are the
sizes of the two lookup tables (10^6 and 10^5 samples over the cosine) that
`concon_kernel_table` reproduces, and `--epsilon 0.001` is the cutoff scan's
threshold, which the steep descent makes insensitive across four decades. The
estimator is the plain kernel density estimate assumed here.

## 7. ConSEAL -- DONE, VERIFIED AGAINST MATLAB; THE REFERENCE'S ERRORS CORRECTED (1-4, 6, 7, 9-17), WITH NO SWITCH BACK

- **From:** the public MATLAB in [`MartyCole/Encore`](https://github.com/MartyCole/Encore)
  at commit `27e4e7d` (`concons/SConcon.m`, `core/Concon.m`, `core/Encore.m`,
  `core/SphericalWarp.m`, `core/SphericalGrid.m`, `kernels/SphericalHeatKernel.m`,
  `mexfiles/aabb_mex.cpp`, `mexfiles/build_adjacency.cpp`,
  `utils/create_search_schedule.m`, `utils/icosahedral_*.m`,
  `analysis/get_dice_score.m`, `analysis/get_overlap_coefficient.m`), the
  implementation behind Xiang, Cole and Zhang, *ConSEAL* (arXiv:2605.16742).
- **To:** `src/sbci/conseal.py`, reached through `sbci.endpoints_align()`. The
  MEX kernels are reimplemented (a centroid k-d tree replaces the libigl AABB
  tree, the adjacency is a sparse scatter in float64), the kernel is kept
  sparse, and the geometry (`SphericalGrid`, exponential and logarithm maps,
  Voronoi areas, tangent basis) is shared with ENCORE in `alignment.py`.
- **Reference run:** `tests/reference/conseal_reference.m` runs the public
  code, CPU only, on a level-4 icosphere with three synthetic 40,000-streamline
  subjects and dumps every intermediate; `conseal_compare.py` diffs the port
  against it with the reference's errors put back (item 22; until then the
  port ran in `strict_upstream` mode).

### The author's copies on Longleaf were checked first

The request was to port the public code but to check `/users/x/y/xya/` for a
newer version. Five copies exist there and in `~/ondemand`; none changes the
public algorithm, but they do say which code the paper's numbers came from.

| Copy | What it is | Differs from `27e4e7d` by |
| --- | --- | --- |
| `/users/x/y/xya/Encore_new` (June 13, 2026) | clone of `27e4e7d` | `use_GPU` defaults to `false`, `canUseGPU()` guard, the GPU flag passed through `apply_warp`. No algorithmic change. |
| `/users/x/y/xya/Encore-main` (May 12) | earlier snapshot, CPU-only rewrite | `SConcon` stores triangle indices as `int32` (fixing item 6 below); `trace(Q Q_mu)` written as an elementwise sum; a plotting colormap; two data paths. Same mathematics. |
| `/users/x/y/xya/Encore` (October 2025 lineage, edited to June 19, 2026) | the research fork the experiments were run in (`experiments/`, `accuracy_comparison_under_large_deformation/`, `MMD_kernel/`, `trait_predict/`) | a different algorithm: the derivative of `Q` is ENCORE's central difference through `ConConInterpolator`, not the analytic kernel derivative; warps compose directly on vertex positions (`compose_warp`), not through a stationary velocity field; no step clamp and no Laplacian smoothing; step 0.1 and threshold 1e-6, stopping on `abs(cost change)`; `F = K A K'` with a precomputed kernel. These are the paper's stated settings. |
| `/users/x/y/xya/ConCon_Alignment`, `~/ondemand/data/encore` | the ENCORE reference of item 4 | not ConSEAL |

So the paper describes the fork, and the public repository is a later
refactor that introduced the stationary-velocity warp, the analytic
derivative, the regularization -- and the errors below. This item ports the
public code, as asked; `viscosity=0, step_clamp=inf, delta=0.1, threshold=1e-6`
gives the fork's settings but not its update (its finite-difference derivative
and direct composition), and since item 19 these settings are the defaults.

### What matches the MATLAB reference

`conseal_compare.py` on the reference dump, the port in float64 against
MATLAB's `single` kernels, with the reference's errors put back into the
port's output (`reference_geometry.py` and `reference_conseal.py`; item 22),
as its last run printed it (job 4613665, 11 October 2026; job 4603697 printed
the same before the trace was compared again):

| Stage (as `conseal_compare.py` labels it) | Agreement (max abs, relative to the largest entry) |
| --- | --- |
| Voronoi areas, `e1`, basis, divergence (the basis and divergence with the reference's clamp and recurrence put back; entry 14 below) | 3.0e-16, 8.0e-16, 5.3e-15, 2.0e-15 (float64 rounding) |
| heat kernel `K`, cutoff included | 4.1e-8 |
| the reference's kernel derivatives `dK.x/y/z` | 1.3e-6 |
| adjacency (float64 here, float32 there) | 2.4e-4 of the largest entry, 2.4e-9 absolute |
| `F`, `F_e1`, `F_e2` | 1.1e-5, 3.1e-5, 3.7e-5 |
| `Q1`, `Q2`, `Q2_e1`, `Q2_e2` | 2.2e-5, 2.3e-5, 6.5e-5, 1.0e-4 |
| gradient, left and right hemisphere; the step, likewise | 5.7e-7 and 1.2e-6; 5.7e-7 and 1.2e-6 |
| warp vertices after one composed step, left and right | 2.4e-10, 2.3e-9 |
| velocity field, Jacobian (left) | 1.6e-16, 1.2e-9 |
| endpoints carried along by the warp, start and end points | 3.2e-5, 3.4e-5 (single-precision coordinates in MATLAB) |
| `Q2` after one step | 2.3e-5 |
| cost after one step | 0.102993879 against 0.102993876 |
| cost trace, six registration iterations stepped in the reference's arithmetic | within 3.5e-7 relative at every iteration |
| Karcher-median template, five iterations | 2.0e-5 |

*First run on 23 September 2026, when the trace agreed to 7e-8 to 4e-7 at
every iteration; 3.3e-7 after the corrections of 5 October (item 8). Item 22
stopped comparing it, and item 23 compares it again.*

Every stage agrees to the precision MATLAB carries it in. The two closest-point
searches picked the same triangle for every one of 240,000 endpoints.

### What was found in the reference

The port corrects 1 to 4, 6, 7 and 9 to 17, takes the paper's settings for 8
(item 19), and keeps 5 as the choice it is; 7 only reaches the bandwidth
selection. Until item 22 removed it, `strict_upstream=True` reproduced the
reference's arithmetic in 1 to 4, 7, 12, 13 and 15 to 17, its settings (8) and
its rigid search without turns about z (item 18);
`tests/reference/reference_conseal.py` keeps what the comparisons and probes
need, 1, 2, 12, 13, 15 and 16 (item 22). Numbering matches the module
docstring.

1. **A gradient term in the wrong tangent frame.** `Concon.evaluate` forms
   `Dx = dK A K'` and symmetrizes it, `Dx + Dx.'`, before projecting each row
   onto that row's frame. The added term is the derivative of `F(a, c)` with
   respect to the *other* vertex `c` -- a vector tangent at `c` -- dotted with
   the frame at `a`. ENCORE's derivative is single-slot and its gradient
   formula already doubles the first-slot term for the second slot;
   `Encore.compute_gradient` kept the doubling, so the second slot is counted
   once correctly and once nonsensically.
2. **The derivative is of the wrong product.** `SConcon.evaluate` returns
   `K~' A K~` with `K~` row-normalized (each source spreads unit mass, so `F`
   sums to one), but `SphericalHeatKernel.build_derivative` differentiates
   `K~` with respect to its row vertex with the normalization propagated, and
   `Concon.evaluate` contracts it as `dK A K~'`: the derivative of `K~ A K~'`,
   whose normalization sits on the evaluation side. The two coincide only
   where the kernel's row sums are constant.
3. **A refused step still enters the velocity field.** `SphericalWarp.compose`
   adds the displacement and smooths it before testing for folded triangles;
   when the test fails it keeps the field and discards only the vertices, so
   field and vertices disagree from then on.
4. **A cost increase is treated as convergence.** `Encore.register` stops when
   `cost(iter) - cost(iter+1) < threshold`, which includes an increase; the
   step that caused it is already composed and is what is returned. `Final
   Cost` prints `cost(iter)`, the previous value. (The same family of defect
   as item 4's second bug.)
5. **The cost is a plain sum over vertex pairs.** The Voronoi integration
   matrix `A` is built and never used. The gradient is the consistent gradient
   of that plain sum, so this is a discretization choice and not a bug;
   `area_weighted=True` weights both.
6. **`int16` triangle indices** overflow once both hemispheres exceed 32,767
   vertices, i.e. ico6 and finer. The author's `Encore-main` copy already uses
   `int32`; the port uses `int64`.
7. **The leave-one-out bandwidth score** subtracts `K(t_in,t_in) K(t_out,t_out)' / (M-1)`
   as the streamline's own contribution, without its barycentric weights and
   with `1/(M-1)` where the symmetrized adjacency uses `1/(2N)`. Bandwidth
   selection only; the paper fixes `sigma = 0.005`.
8. **Undocumented regularization.** The paper says none; the public code
   smooths the velocity field by 5% of its cotangent Laplacian at every
   composition and clamps the largest step to 0.2 (`compute_step_size`). The
   fork has neither. Both were the defaults until item 19 made the paper's
   settings the defaults; the public code's are the `PUBLIC_*` constants.
9. `get_template` counts subjects with `size(Fs, 1)`, so a row cell array
   silently uses one subject; and `acos(trace(Q Q_mu))` is taken without a
   clip, which rounding above 1 turns complex in MATLAB.
10. The shipped scripts do not run: `README.md`, `simulation.m` and
    `real_data_registration.m` construct the abstract `Concon`;
    `random_diffeomorphism.m` calls a `compose_warp` that does not exist;
    `simulation.m` copies `lh_warp` where it means `rh_warp`.
11. Kernels, frames and the adjacency are `single`; `build_adjacency.cpp`
    accumulates a million outer products in float32.
12. **The cotangent Laplacian weights each edge with an adjacent angle.**
    `cot_matrix` puts the cotangent of the angle at `i2` on the edge
    `(i2, i3)`, where the cotangent formula takes the angle opposite the
    edge. The result is still a symmetric graph Laplacian that annihilates
    constants, so the 5% smoothing still smooths, with the weights around the
    twelve five-valent vertices swapped (0.33 for 0.73).
13. **The square root's chain rule divides by a floor where the density was
    clamped.** `Q_transform` scales the derivative by `1 / (2 max(Q, 1e-15))`;
    where `F` was negative and clamped to zero, the derivative of
    `sqrt(max(F, 0))` is zero, but the reference multiplies a nonzero `dF` by
    `5e14`.
14. **The tangent basis shares ENCORE's Legendre recurrence**, with its wrong
    m = 0 term (item 4, *Four more*; item 8). The basis fields are right,
    their divergences are not, and the gradient uses them. Corrected, in both
    modes from item 21 until item 22 left one; `conseal_compare.py` rebuilds
    the reference's geometry for the comparison above. Found by the review of
    5 October 2026, which also found a defect of the port's own: `register()`
    copied an `EndpointConnectome` but kept its older committed locations, so
    an object returned by one alignment was silently reset at the start of
    the next; it now commits the copy's current locations first.
15. **The Karcher median stops on its starting subject** when that subject's
    square-root density rounds to a squared norm below 1 (item 9).
16. **The velocity field is smoothed as two scalar components**, which mixes
    frames where they turn sharply around a coordinate pole (item 9).
17. **A rigid rotation is held as a velocity field**, realized 1.6 degrees
    off at 150 degrees and eroded by every later smoothing (item 9). The
    port holds it exactly, outside the field.

### The gradient, measured

`tests/reference/conseal_gradient_probe.py` differentiates the *actual* cost --
move every endpoint along a field, relocate, re-smooth, compare -- by central
differences and sets it against the analytic coefficients `dE/dbeta_k` of the
port and of the reference (`strict_upstream` until item 22, the same code in
`tests/reference/reference_conseal.py` since). The kernel derivatives
themselves are checked in `tests/test_conseal.py`: the port's matches a
finite difference of `K~(i, x)` in the evaluation point to 1e-5 (the strict
one matched a finite difference of the reference's own definition to 1e-5
until item 22), and the connectome derivative `D_e1` matches a finite
difference of `F(x, c)` in `x` to 2e-4 of its range.

Along each method's own descent direction, `dE/dt` by finite differences over
the port's first-order prediction:

| Grid, settings | port's direction | reference's direction | cosine between the two gradient fields (lh, rh) |
| --- | --- | --- | --- |
| ico2, sigma 0.05, degree 12, order 3, 3,000 streamlines | 1.054 | 1.064 to 1.066 | 0.966, 0.826 |
| ico3, same, 20,000 streamlines | 1.140 | 1.142 | 0.855, 0.922 |
| ico4, sigma 0.005, degree 30, order 15, 100,000 streamlines (the paper's) | 1.019 | 0.986 | 0.716, 0.691 |

(As the probe printed them on 12 October 2026, jobs 4628494, 4628699 and 4628490;
the reference's direction on ico2 reads 1.064 or 1.066 by node. The ico2 and ico3
rows had been printed before item 10's basis at the poles: 1.050, 1.058, 0.969 and
0.828, and 1.138, 1.139, 0.855 and 0.923. Item 24.)

Both directions descend; the port's prediction of its own descent rate is
within 2% on the paper's settings (13% before the review corrected the
divergence, item 8). Per coefficient the reference is unreliable: on ico4
its coefficients for the eight sampled basis fields are off by factors of
0.28 to 3.4 with three sign flips, against 0.25 to 2.1 with two for the port
on the same fields (the transported-density model behind both is itself only
first order on a mesh). The definitive measurement is the exact gradient of the
discrete cost, all 510 left-hemisphere coefficients by finite differences:

| Quantity | port | reference |
| --- | --- | --- |
| cosine with the exact gradient, all 510 coefficients | **0.981** (0.931 before the corrected divergence) | 0.678 |
| cosine on the 20 largest exact coefficients | 0.997 | 0.941 |
| norm, relative to the exact gradient | 0.947 | 0.753 |
| descent efficiency: first-order decrease per unit step, as a fraction of the exact gradient's | 0.981 | 0.678 |

(ico4, sigma 0.005, degree 30, order 15, 100,000 streamlines per subject;
1,020 cost evaluations, 59 minutes on 8 cores, rerun on 5 October 2026 with
the corrected divergence; the reference's column runs on the reference's own
basis and is unchanged from the earlier run. Run again on 12 October, it
printed every figure of the table to the digits quoted, in 68 minutes; job
4628490, item 24.) The port's gradient points 11
degrees from the exact gradient of the discrete cost, the remainder being the
transported-density approximation and the mesh; the reference's points 47
degrees away and buys two thirds of the decrease per step. Both descend,
which is why the reference works at all:
its large components are right (cosine 0.94 on the twenty largest) and its
errors live in the smaller ones.

### On ico4, real data

`tests/reference/conseal_adni_ico4.py` aligns three subjects of the lab's
archive from their `mesh_intersections_ico4.mat` at the public defaults, and
prints what a registration has to show: the cost falls and its trace is
monotone, every Jacobian stays positive, each subject moves closer to the
template, and the subjects' pairwise distances and endpoint overlaps improve.
It reads the subjects from `SBCI_ADNI_SUBJECTS`; they, and the numbers measured
on them, are not given here (item 39). The heat kernel at the published
bandwidth has 89 nonzeros per row (cutoff 21.5 degrees). ConSEAL on released
young adults -- known warps undone, bundle overlaps after alignment, and the
time an iteration takes -- is in items 19, 35 and 36.

### How to run it

```python
import sbci
subjects = [sbci.load(p) for p in paths]           # files that carry endpoints with positions
result = sbci.endpoints_align(subjects)             # the paper's settings since item 19
result.warps[0].save("sub-001_conseal_warp.npz")
aligned = result.aligned_endpoints(0)               # an Endpoints object, ready to re-smooth
```

`conseal_compare.py` reproduces the reference to the digits above with the
reference's errors put back (entry 14 above; items 21 and 22);
`init_rotation=True` adds the multi-shell rigid search (its schedule is
regenerated from `create_search_schedule.m`, 80/20/20/20/16 rotations, and the
60 icosahedral rotations are fitted to the grid rather than assumed, and the
densities are transported by interpolation, so it also runs on the bundled
FreeSurfer sphere, which is an icosphere to 1.7e-4 -- the precision of its
stored coordinates).
On ico4 an iteration costs a few seconds per 100,000 streamlines: the kernel
has 32 nonzeros per row at the default bandwidth (cutoff 12.4 degrees; 89 and
21.5 at the public code's 0.005), so
`K' A K` is a sparse sandwich (its sparse-times-dense products run on all the
cores the job was given, in row blocks that reproduce the single-core result
bit for bit), and relocating the endpoints is a k-d-tree
query.

### Measured on a known deformation

*The synthetic measurements in this subsection were made before the review
of 5 October 2026 corrected the divergence both gradients use (item 8,
finding 2); the figure's numbers on the tutorial subject were regenerated with the
corrected code and came out the same to the digits quoted, ENCORE in eleven
steps rather than ten (ten again since item 21).*

The reference's own tests compare registrations of unrelated subjects, which
says the cost falls but not how far the warp is from the right one. So: a
synthetic subject's endpoints were moved by a known smooth warp (a random
combination of the warp basis fields, moving the grid's vertices 1.0 degrees on
average and 3.6 at most and the endpoints 0.98 and 2.9, `scripts/make_figures.py`),
the deformed copy re-smoothed, and registered back
onto the original. What matters is where the endpoints end up:

| setting | iterations taken | endpoint residual (from 0.98 deg) | cost removed |
| --- | --- | --- | --- |
| public defaults (`threshold=1e-4`) | 10 | 0.64 deg, 34% undone | 72% |
| public defaults, `threshold=1e-7` | 60 | 0.47 deg, 52% | 89% |
| public update, `delta=0.2`, `threshold=1e-7` | 60 | 0.27 deg, 72% | 97% |
| the paper's settings, through the package's update (`delta=0.1`, no clamp, no viscosity), `threshold=1e-7` | 60 | 0.17 deg, 82% | 99% |

Two things follow. The public stopping rule is an *absolute* change in cost,
and 1e-4 is a quarter of the whole cost when two subjects are this alike, so
it stops after ten iterations; pass a smaller threshold for similar subjects.
And the recovered warp should be judged by the endpoints, not by the grid
vertices: the vertex field the registration finds differs from the one that
made the deformation wherever there are no endpoints to constrain it (the
vertex residual grows even as the endpoint residual falls), which is a
property of the problem, not a fault of the port. ENCORE on the same pair,
with its inverse applied to the endpoints, reaches 0.40 degrees, 59% undone
(the figure the ENCORE notes in USAGE.md cite).

On a real subject the same test can make a registration look as if it
recovered nothing, through two flaws of the experiment rather than of the
port (`tools/encore_probe.py` measures both):

- **The reference has to be smoothed from the same endpoints as the deformed
  copy.** A stored density whose endpoints are not the ones it was smoothed
  from (item 6, *Endpoints from the snapped streamlines*) sits at a cost from
  them that no warp removes, and a registration onto it fits the difference
  between the two endpoint sets.
- **A warp with structure at the grid scale is one the smoothed density cannot
  see.** Deforming the endpoints and then smoothing is not smoothing and then
  deforming once the local stretch is large, so even against a matching
  reference such a warp need not be where the cost is least, and no optimizer
  of that cost can be expected to find it.

The figure in docs/RESULTS.md uses a smooth warp, a random field of degree 4,
and the matching reference (`scripts/hcp_figures.py`: `WARP_ORDER`,
`WARP_AMPLITUDE`, `ENCORE_ORDER`), on the tutorial subject: ENCORE brings the endpoints from 1.63 to 0.20 degrees, 88%,
with the cost at 0.15 of its start; ConSEAL, at the defaults of 11 October
2026, to 0.02 degrees, 99%, in 80 iterations (at the paper's settings and the
public code's bandwidth, with a threshold of 1e-7, it had reached 0.11
degrees, 93%, in sixty).
Two rules follow for any comparison: build every density from the same
endpoints through the same smoother, and judge a registration by a warp it can
represent.

### Measured on the synthetic cohort

*Measured before the correction of the shared divergence (item 8, finding
2); the conclusions are about which template and which regularization
preserve a planted difference, and do not rest on the gradient's exact
direction.*

`sbci.example_cohort(n_subjects=10, n_streamlines=20000, anatomy=a)` plants
one bundle whose weight rises with age and jitters every subject's anatomy by
a smooth field of spread `a`. Aligning the cohort and then fitting a rank-4
FPCA with `candidates=6` (so that the fit reaches the largest component, item
5) measures what each method does to a real between-subject difference:

| cohort | unaligned | after ENCORE (10 it) | after ConSEAL, paper's settings, Karcher median (30 it) | after ConSEAL, public settings, Karcher median | after ConSEAL, paper's settings, mean template |
| --- | --- | --- | --- | --- | --- |
| `anatomy=0.03` | found, first component, r = 0.94 | found first, 0.94 | found first, 0.94 | found first, 0.94 | not run |
| `anatomy=0.04` | found first, 0.93 | found first, 0.95 | found first, 0.95 | not run | not run |
| `anatomy=0.05` | found second, 0.92 | found first, 0.94 | **not found** | found second, 0.94 | found first, 0.94 |

(`r` is the correlation of the component with the planted bundle; every
"found" has adjusted p below 0.001.) Three things follow. ENCORE never
removes the effect and brings it to the front. The Karcher median is one
subject at three degrees: subject 1 becomes the template (cost 0, no
iterations) and the others are registered onto its bundles. *The cause first
given here -- that with every subject 10 to 11 Fisher-Rao degrees from the
mean the first Weiszfeld step is already shorter than 0.005 -- was wrong.
The step is that short because subject 1's square-root density rounds to a
squared norm 1.4e-10 below 1, which the reference's coincidence snap misses,
so subject 1 weighs about 7e4 times the others (item 9); at two degrees the
starting subject's norm rounded 5.1e-10 above 1 and nothing happened. The
corrected median sits 0.3 degrees from the mean of the three-degree cohort;
the "not found" above is the collapsed template's.* And the paper's settings (no clamp, no viscosity)
onto that one subject reshape the bundles enough to remove a fivefold weight
difference, while the same settings onto the mean of the square-root densities
keeps it, as does the public code's clamp and viscosity onto the one subject.
With the reference start instead of `candidates=6`, alignment *lowers* the
chance of finding the bundle at rank 4 (six seeds, spreads 0.05, 0.04 and
0.03: unaligned 0, 4 and 6 of 6; after ENCORE 1, 0 and 4 of 6; after
ConSEAL at the paper's settings 0, 0 and 1 of 6), because the aligned cohort's Gram spectrum
is flatter still. Rank 8 with the reference start finds it unaligned in 5, 6
and 6 of 6 and after ENCORE in 4, 6 and 6 of 6; after ConSEAL at the paper's
settings in 1, 3 and 6 of 6.

### Still open

- The rigid search's cost surface is interpolated on the grid; on ico2 it
  lands a few degrees from a known rotation (test tolerance 0.12 rad). On ico4
  this is a fraction of a degree but has not been measured against a known
  rotation on real data.
- Bandwidth cross-validation is ported with the corrected self-term only, an
  exact leave-one-out (item 23 checks it against one); the reference's went
  with `strict_upstream` (item 22). It is untested against MATLAB, since the
  paper does not use it.

## 8. The review of 5 October 2026 -- ELEVEN FINDINGS, ALL CONFIRMED; FIVE CAME FROM THE REFERENCES

An independent implementation review (5 October 2026, of v0.0.1.dev0 at
commit 4b1850b) built the wheel, ran the suite (724 passed, 31 skipped) and
then checked the mathematics rather than the agreement: fitting against
projection, area weighting, shift invariance, regional aggregation, the
harmonic derivatives, warp orientation, transported-density normalization,
repeated registration, seed masks and file interoperability. Every one of its
eleven findings reproduced here with independent probes before anything was
changed. The lesson is the one VERIFICATION.md now states first: **a green
suite and MATLAB agreement cannot catch an error the reference shares**, and
three of the findings were exactly that. `docs/review-2026-10-05.md` is the
response written for the reviewer; this item is the record.

| # | Finding | Cause | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | `project()` scores differ from the fit's | `ConConSmooth.smooth` fits the lower triangle with the diagonal; the fit contracts the whole matrix (**reference**) | the fit's own scoring; `reference=True` kept the triangle (until item 21) | factor 1/(1 + sum psi^4): two thirds on two vertices, 0.04% on ico4 |
| 2 | wrong m = 0 term in the Legendre derivative recurrence | `legendre_2nd_derivative.m` applies the unnormalized relation to normalized functions (**reference**, shared by ENCORE and ConSEAL) | the normalized relation; `reference=True` / `strict_upstream=True` kept the reference's (until items 21 and 22) | m = 0 derivative times (1 + 1/(l(l+1)))/2; zonal divergences up to twice too large; the basis fields unchanged |
| 3 | `ConSEAL.register` resets an object that was already warped | the copy kept the older committed locations | `commit()` on the copy | endpoints moved 0.07 rad (4 degrees, on the test's ico2 grid) while the trace read 0 |
| 4 | ENCORE accepts a map that folds | acceptance by cost alone; the Jacobian is an absolute value (**reference**) | a folding trial is rejected, always | 22 of 80 faces at step 2; none in 40 trials at the default step |
| 5 | the transported square-root density is not of unit norm | normalized before its diagonal is zeroed (`Concon.m`, **reference**) | zero first; `reference=True` kept the order (until item 21) | 0.02% to 0.2% on ico4; 5% on a coarse, very local toy |
| 6 | the areas do not weight the first component | the Gram enters only the deflation (**reference**) | documented; the method decision stays with WP1 | twice the area-weighted residual on a four-vertex toy |
| 7 | `seed(region=)` divides by masked area | weights over every selected vertex | intersect with the mask; refuse an all-masked region | 0.8327 of the cortical-only profile on the tutorial subject's PALS left limbic lobe |
| 8 | an intercept-adjusted F test changes when the response is shifted | the untestable rule judged on the uncentred scale | the centred scale when the reduced model can reproduce a constant: an intercept column, and since 6 October dummy codes for every level (below) | F = 10.9 became NaN after adding 1e6 |
| 9 | the within-region mean counts each vertex paired with itself | denominator A_k^2 | mean over distinct pairs; a one-vertex region gets NaN | 0.8(1 - 1/n): 4% low for Schaefer-sized regions |
| 10 | GIFTI arrays read by position | `darrays[0]`, `darrays[1]` | by intent | a triangles-first file gave zero faces |
| 11 | the FC exchange sidecar says density | one string for both modalities | by modality | provenance only |

Items 1, 2 and 5 are reference arithmetic the port reproduced to rounding,
and 4 and 6 are the references' rule and algorithm. The first three are what
the agreement tables above measured; it now computes
the right thing by default and keeps the reference arithmetic behind
`reference=True` (ENCORE and `project`) and `strict_upstream=True` (ConSEAL,
whose default grids then carry the reference basis), so that the Tier 4
comparisons still run (until items 21 and 22, which took both switches out and
put the references' errors back in the tests instead). Item 4 is corrected
without a switch: a fold is never
a diffeomorphism, and at the default step the check never fires on the
recorded runs. The reviewer's diagnostic scripts were not adopted; each
invariant has its own test in the suite -- the harmonic derivatives against
finite differences and the divergence against `-l(l+1) Y`, unit norm after
transport, no accepted fold, a pre-warped object registered onto itself,
the training cohort projecting onto its own scores, a region mean that does
not depend on the grid, shift invariance, both GIFTI orders, the FC sidecar.

### What the corrections change on the released subjects

Recorded with `tests/reference/record_outputs.py` on two of the released
young adults, the tutorial subject and subject B (`reduce` and `project` on four), on
an AMD EPYC 9654 node, and set against the recording of 2 October 2026 made
on the same CPU type with `tests/reference/compare_outputs.py` (the
cluster's Intel and AMD nodes round differently, VERIFICATION.md Tier 3):

- **The reference arithmetic reproduces the earlier results bit for bit.**
  `align(reference=True)`, and `endpoints_align` on grids built with
  `reference=True`, return the identical costs, template, warped vertices,
  aligned density and endpoints (worst difference 0.0 over seven arrays);
  every method the review left alone -- parcellation off the diagonal, the
  three couplings, the vertex seed, smoothing, the FPCA basis, scores, scales
  and explained fractions -- is identical too.
- **ENCORE.** The corrected gradient takes the two subjects to costs lower by
  4e-4 and 1.1e-3 relative (0.012707 against 0.012712 for the tutorial subject), in 15
  accepted steps rather than 11, with more halvings near convergence (352 s
  against 48 s on 8 cores). The warped left-hemisphere vertices land 0.043
  degrees from the reference's on average and 0.25 at most; the aligned
  density differs by 1.5% of its largest entry (correlation 0.999991). The
  Karcher template is identical: it does not involve the gradient.
- **ConSEAL**, three iterations: the costs differ by up to 1.4e-3 relative at
  the last iteration (higher for one subject, lower for the other), and the
  aligned endpoints sit 0.023 degrees from the reference's on average, 0.11
  at most.
- **Projection.** On the four-subject rank-4 fit the default projection
  returns the fitted scores to 0.9% (median), which is the fit's own
  tolerance at the default `tol_outer`; the reference's triangle objective
  runs 2.2% lower still (median ratio 0.978).
- **Region seed.** The tutorial subject's PALS left limbic lobe has 170 cortical and 34
  medial-wall vertices; the old profile was 0.832668 of the corrected one,
  the number the review measured.
- **Within-region means.** The Desikan SC diagonal moves by 1.9e-10 in
  density units and the Schaefer-200 FC diagonal rises by up to 0.042 in
  correlation; neither atlas has a one-vertex region, so neither matrix
  gains a NaN.

### Finding 8, for an intercept the columns only imply (6 October 2026)

The fix recognized an intercept only as a constant column. Dummy codes for
every level of a factor, passed with `add_intercept=False`, sum to one and
absorb a shift just as well, but the untestable rule still judged them on the
uncentred scale: on three components of a three-level probe, a shift of 1e6
turned every component NaN where the same model coded against a reference
level gave F = 12.8, 11.5 and 15.9. The rule now asks whether the reduced
model's columns can reproduce a constant, by the numerical rank the degrees of
freedom are counted with. The dummy-coded design then agrees with the
reference-level coding to 2e-15 unshifted and 6e-11 at 1e6, 1e-9 with
families as clusters, and its permutation p-values are equal. With the
intercept only implied, the default `terms` -- every column but the intercept
-- has no column to leave out and would test the intercept with the rest, so
it is refused and the message names `terms=`. With the intercept prepended,
the default, nothing changes: the reduced model holds the intercept column
whenever the old rule applied, and 300 random designs (a third with a factor
coded at every level), each tested twelve ways -- four choices of terms,
plain, with families and with permutations -- give identical results, all
3,600 bit for bit.

## 9. The second list, 5 October 2026 -- NINE FINDINGS AND A DOZEN SMALL ONES, ALL BUT ONE CONFIRMED

A second reviewer's list arrived the same day, built on the first and
re-run against the corrected code. Each item was checked with an independent
probe or against the code before anything changed; one (the pole vertices,
item 2 below) holds for a different reason than the one given.

| # | Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | ConSEAL's Karcher median collapses onto its starting subject | **confirmed**: the square-root densities are not renormalized, float32 barycentric weights put a subject's squared norm 1e-10 off 1, and when the starting subject's is below 1 the `1e-14` snap misses it, that subject's Weiszfeld weight is about 7e4 and the first step is already shorter than 0.005 (**the reference's arithmetic**, item 7 docstring item 15) | unit-norm densities and a 1e-6 radian coincidence guard; `strict_upstream=True` kept the reference's (until item 22) | on the eleven released subjects the median started from subject C, whose squared norm rounds 1.7e-10 below 1, and stopped 0.001 degrees from it and 24-27 from the rest; it now sits 0.43 degrees from their mean and 16.6-20.2 from every subject. The synthetic cohort that collapsed at three degrees of spread had its starting subject 1.4e-10 below 1, the one that did not at two degrees 5.1e-10 above |
| 2 | ConSEAL's default grids have four vertices on the poles | **partly**: the vertex at the north pole moves 2.0 degrees of a 4-degree shift of the endpoints around it, but the endpoints, carried by their triangles, land within 0.09 degrees of the truth either way; the cause is the 5% velocity smoothing, which averages the two frame components as scalars across neighbours whose frames turn 72 degrees around a pole -- after a 20-degree `rotate()`, ten zero-size steps move the two most polar vertices 8.8 degrees and one 11.3, on the unrotated grid and on one rotated off the poles alike (7.5 and 9.9); the reference smooths the same way (docstring item 16) | the velocity field is smoothed as vectors: first as ambient 3-vectors projected back onto the tangent planes, and since the third list (item 10) with the connection Laplacian, which parallel-transports each neighbour's vector to the vertex; `strict_upstream=True` kept the component-wise smoothing (until item 22); the grids stay in the file's frame, since a grid rotated off the poles moves the same vertex 3.7 degrees and lands its endpoints worse (0.10 against 0.035) | the ten-step drift at the poles falls to 0.09 degrees with the ambient smoothing and 0.046 with the connection Laplacian (the 5% smoothing of a degree-1 field, as everywhere else); the pole vertex follows 3.45 of the 4 degrees and the endpoints around it land 0.035 degrees from the truth (0.085 before); within 10 degrees of the coordinate equator the connection Laplacian agrees with the reference's smoothing to 1e-4 of the field (1.7e-4 about the coordinate axis, 1.3e-3 to 1.5e-3 about others; item 11) |
| 3 | `local_test(terms=...)` tests a different column when a covariate is constant | **confirmed** in the code: the intercept was not prepended when any column was constant, so the indices shifted | the intercept is always column 0, so a constant covariate is collinear with it and testing it is refused as such (and `groups=`, which needs full column rank, now refuses such a design instead of running on the shifted one); boolean, duplicate and empty `terms` are refused. *Superseded*: since the third list a constant column is refused outright (item 10, row 2), and since the fourth a column of zeros is kept (item 11) | an all-female stratum's `terms=[1]` tested the intercept |
| 4 | the FC exchange file has zeros next to its diagonal | **confirmed** by construction: the zero ico4 FC diagonal spreads to every pair of fsLR vertices in one cell | the FC diagonal is set to the self-correlation 1 before resampling, on cortex only since the third list (the medial wall, for which FC has no data, keeps its 0); the sidecar says so | 12.7 fsLR vertices to an ico4 cell, so most pairs of neighbours share one and read 0 |
| 5 | sidecar files overwrite each other | **confirmed**: the name was cut at the first dot | only the `.dconn.nii` suffix is stripped | `sub-01.ses-1_sc` and `sub-01.ses-2_fc` both wrote `sub-01.json` |
| 6 | three PALS atlases keep the medial wall as two regions | **confirmed**: `MEDIAL.WALL`, with a dot, slipped past the background pattern | pattern widened in `tools/convert_atlases.py`, the three atlases rebuilt from the toolkit's files | Lobes 12 -> 10 regions (394 wall vertices now unassigned), Brodmann 82 -> 80, Visuotopic 25 -> 23; 11,825 bundled regions |
| 7 | `surface="sphere"` plots show the wrong side | **confirmed** from the coordinates: the bundled sphere is the pipeline's frame, its x and y anti-correlated with the inflated surface's (-0.97, -0.95; z +0.92), a half turn about z | the plotting negates x and y of the sphere it draws; the stored sphere, which every grid and warp is built on, is untouched | "lateral" rendered the medial view |
| 8 | ENCORE refuses every step for a raw array with a nonzero diagonal | **confirmed** in the code: the starting cost kept the diagonal, the trial costs did not | `Encore.root()` zeroes the diagonal | one subject stuck at 0.0306 instead of 0.0168; connectomes unaffected |
| 9 | `Warp.save` does not store the grid rotation | **confirmed**: a reloaded warp was migrated as if unrotated | the rotations are fields of the warp, saved and reloaded, and `migrate_warp` reads them | 13.5 degrees of error in the reviewer's run |

The small ones: ENCORE's identity-warp Jacobian came out 0.9965 (0.9958 to
0.9971) after any step, a discretization bias of the finite-difference scheme
inherited from the reference, so `aligned` densities lost 0.7% of their mass
-- it is now calibrated by the identity's own value (`reference=True` kept
the bias, until item 21); `parcellate` zeroes the vertex diagonal it assumes zero and treats
an empty region as NaN throughout under `"mean"`; `smooth("shk")` refuses a
bandwidth that is not positive; the kernels refuse eigenvalues that are not
ascending; `from_snapped` refuses negative vertex ids; `TemplateWarp.apply`
refuses a hemisphere that is not L or R, and every `save` returns the path
that exists; `save()` refuses a suffix `load()` would not accept; an
interrupted download keeps its `.part` file and resumes; `validate` fails
whatever `load` refuses, which it did not for a misnamed `.mat` file, a
coordinate array of the wrong shape, a mask or area of the wrong length or a
`/metadata` dataset that is not a string (two checks added, eleven in all);
`strict_upstream` said which items it reproduced (1 to 4 and 12 to 17; until item 22);
Schaefer-900 and -1000 have 899 and 999 regions on ico4 (BLUEPRINT.md, now
also USAGE.md).

Large `StationaryWarp.rotate` rotations were inexact, as the reviewer said:
0.035 degrees of error at 20 degrees, 0.15 at 45, 0.58 at 90, 1.03 at 120,
1.60 at 150 and 2.29 at a half turn. The reference holds a rotation as its
own velocity field, and the first scaled step of the scaling-and-squaring
flow follows a great circle where a rotation moves a point along a small
circle, an error six squarings double each time (predicted `alpha^2 sin(psi)
cos(psi) / 128`, 93-96% of the measured maxima). Measuring it found a second
effect, larger: every later step smooths the field by 5%, rotation and all,
so a stored 150-degree rotation erodes to 146.2 over the 100 iterations of a
registration (3.9 degrees at most from where it belongs; 142.6 and 7.5 under
the ambient smoothing the port had before item 10), which the gradient has to
keep restoring. More squarings would only have narrowed the first effect, and
would have moved every small-field result with it. The port instead holds a
rotation exactly, outside the velocity field, and the warp is the field's
flow after it (the module docstring's item 17): `rotate` is exact to rounding
at every angle, nothing erodes, a 5-degree rotation composed after a
150-degree one lands within 0.007 degrees of their product (0.003 of it the
5-degree flow's own error), and inverting such a warp returns every vertex
within 0.007 degrees. The rigid part changes nothing for a warp without one,
which is every registration without `init_rotation=True`.
`tests/reference/rotation_probe.py` measures all of this, and what it costs a
registration: a synthetic subject turned by 150 degrees and handed the exact
inverse rotation, as a perfect rigid search would find it, started with its
endpoints 0.98 degrees from the truth on average (1.60 at most) when the
rotation was a field, and thirty steps left them at 0.44 (1.85; 0.81 under
the ambient smoothing) -- the registration cannot remove an error it keeps
re-creating -- where the rotation held exactly leaves them at 0.001; at 60
degrees, 0.17 and 0.089 against 0.001. `strict_upstream=True` kept the reference's arithmetic
(until item 22).

### What the corrections change on the released subjects

Recorded with `tests/reference/record_outputs.py` on an AMD EPYC 9654 node
and set against the first review's recording on the same CPU type with
`tests/reference/compare_outputs.py`, key by key:

- **Everything but the two aligners' defaults is identical, bit for bit**:
  parcellation (the new diagonal and empty-region rules change nothing on
  Desikan and Schaefer-200), the three couplings, seeding, smoothing, the
  FPCA fit and both projections; and so is ENCORE under `reference=True`.
- **ENCORE**: the calibrated Jacobian keeps the aligned density's mass,
  0.70% more of it than before (but not what lands on the zeroed diagonal: item 38); the warped vertices move 0.001 degrees on
  average (0.003 at most) and the costs 3e-5 to 6e-5 relative. The
  known-warp recovery figure is unchanged to the digits it quotes.
- **ConSEAL**: the vector smoothing moves the aligned endpoints of the two
  subjects 0.001 degrees on average, ten times that within 10 degrees of
  the coordinate poles (0.009) and 0.085 at most. The two-subject template
  did not move: it sits a fifth of the way from the tutorial subject to subject B
  (5.6 and 22.4 degrees) under both arithmetics.
- **The template of the eleven** (`tests/reference/template_probe.py`, on
  all of them, default against `strict_upstream=True` until item 22; it also runs the two
  synthetic cohorts of item 7): the reference's arithmetic stops on
  subject C, 0.001 degrees from it and 16.9 from the mean; the corrected
  median sits 0.43 degrees from the mean and 16.6 to 20.2 from every
  subject. The cohort figure of docs/RESULTS.md, which registered ConSEAL
  onto the mean of the square-root densities because of the collapse, now
  registers it onto this median: the subjects' mean pairwise correlation
  rises from 0.716 to 0.784 (0.785 onto the mean, before the corrections),
  and ENCORE's half holds at 0.761.

## 10. The third list, 6 October 2026 -- A RE-CHECK OF THE FIRST TWO, AND OF THE FIXES

A third list re-ran the earlier findings against 64722e6 and audited what
that commit changed. Each item was checked with a probe or against the code
before anything changed; all hold, one in part.

| # | Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | the ConSEAL pole vertices only partly fixed | **confirmed**: the tangent basis divides the phi term by `max(sin(theta), 1e-5)` -- the reference's clamp (`Ylm_2nd_derivative.m`, whose own comment asks why) -- so at a vertex exactly on the axis the m = 1 sine fields and their curl partners are zero, 30 of the 510 fields, while their cosine partners are full size; the second list's smoothing hid it | the limit of `P_l^m / sin(theta)` near the axis, `(dP/dtheta) / cos(theta)` for m = 1 and 0 above; `reference=True` keeps the clamp. Away from the poles the fields change only by their normalization, by 0.05% to 0.5% | with `viscosity=0` and endpoints shifted 4 degrees around the pole, the pole vertex now moves 3.54 degrees among neighbours moving 2.66 to 3.86; the reference's basis leaves it at 2.07 among 2.22 to 3.75 (the reviewer's setup: 2.84 against 4.60) |
| 2 | `local_test(terms=)` tests another column when the design has its own intercept | **confirmed**: since 64722e6 the intercept is always prepended, so `[ones, age, sex]` with `terms=[2]` tested age, its p-value statsmodels' age p-value exactly | a constant column, or a column of zeros, is refused when an intercept is prepended, the message naming both remedies (`add_intercept=False`, or drop the covariate); the docstring and USAGE.md carry the migration note | -- |
| 3 | coupling wrong on atlases with empty regions | **confirmed**: the NaN row of an empty region made a constant profile (a region with no SC) look varying, so it stayed in every correlation | empty regions dropped before the constancy test | every value moved (-0.021 to -0.258 in my reproduction); now bit for bit as without the empty region, for all three forms; 260 of 262 captured outputs unchanged, the two that changed the bug itself |
| 4 | a ConSEAL subject with no streamlines makes the cohort NaN; ENCORE likewise for a density whose mass is all on its diagonal | **confirmed** | refused in words, by index | -- |
| 5 | ConSEAL's template held two copies of the cohort | **confirmed** | each density normalized in place | peak 2.0 cohorts' worth to 1.2 to 1.4 (about 12 GB for 50 ico4 subjects) |
| 6 | the FC export set the diagonal to 1 on the medial wall | **confirmed** | cortex only | 14,871 pairs of medial-wall fsLR vertices had read exactly 1 |
| 7 | the Jacobian calibration beside the axis | **confirmed, smaller than reported**: `align()` let through a vertex 1e-5 rad off the axis, where the calibrated Jacobian of a 0.5-degree rotation read 1.27 (the reviewer saw 920 closer still) | `align()` refuses vertices within the 1e-3 that `rotate_off_poles` clears, and the calibration ignores an identity value below 0.5 (*superseded* by item 11: it left coarse grids uncalibrated, and the axis alone now decides) | 1.0001 at 1e-3 |
| 8 | the vector smoothing shrinks rotations twice as fast as the reference | **partly**: twice as fast only about the coordinate axis -- 95.2% of a rotation field left after 100 smoothings against the reference's 98.9% -- while about a tilted axis the reference shrinks it faster (90.5%), its rate depending on the frame (item 16 of the ConSEAL docstring). The ambient smoothing's own rate came from a curvature term: projecting a neighbour's vector instead of transporting it adds the sphere's Ricci curvature, doubling the damping of rotation fields | the connection Laplacian, which parallel-transports each neighbour's vector | 97.6% left for every axis, exp(-5 * 4 pi / 2562); within 10 degrees of the coordinate equator it agrees with the reference's smoothing to 1e-4 of the field, 3 to 20 times closer than the ambient one (1.7e-4 about the coordinate axis, 1.3e-3 to 1.5e-3 about others; item 11) |
| 9 | stale documents | **confirmed** | `atlas.py` (PALS_B12_Lobes is ten lobes, 394 vertices unlabelled), and the 0.7% of item 9, which is lost once, when the final warp is applied, not at every step | -- |

And the small ones: the CIFTI export checks the file name and validates the
metadata before the 16.9 GB resample; `fetch_cohort(modalities="sc")` takes
the string as one modality; `local_test(groups=)` refuses missing family
labels, which `np.unique` had made one family (the text strings `"NA"` and
`"nan"` are labels like any other); the HDF5 reader splits the endpoints at
the file's own vertex count. Two checks went further than asked: `align()`
refuses a non-square input, and `pole_rotation` measures `sin(theta)` on
normalized rows, which a sphere of radius other than one had defeated.

### What the corrections change on the released subjects

Recorded with `tests/reference/record_outputs.py` and set against the second
list's recording with `tests/reference/compare_outputs.py`, key by key. The
job landed on an Intel Gold 6140 rather than the AMD EPYC 9654 of the
earlier recording; the two are in the same AVX-512 group (VERIFICATION.md,
Tier 3) and differ here by one unit in the last place in `reduce`'s explained
fractions and the projections, nowhere else.

- **ENCORE**, by default and under `reference=True`, and every method
  outside the two aligners -- parcellation, the three couplings, seeding,
  smoothing, the FPCA fit: identical, bit for bit. ENCORE's grids are
  rotated off the axis, so the pole basis never reaches them.
- **ConSEAL**: the two subjects' aligned endpoints move 0.00035 degrees on
  average and 0.06 at most, 0.005 within 10 degrees of the coordinate poles,
  where the corrected basis acts, and 0.0003 elsewhere. The run on the
  reference basis, which isolates the connection-Laplacian smoothing, moves
  them 0.00006 degrees on average and 0.002 at most.

## 11. The fourth list, 6 October 2026 -- A RE-CHECK OF THE THIRD, AND OF ITS FIXES

A fourth list re-ran the open items against 1af5e32 and audited the third
round's changes. Each item was checked first, with a probe or against the
code; `tests/reference/fourth_list_probe.py` records the measurements
(`fourth-probe.sbatch` on Longleaf). All hold.

Reported as still present:

| Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- |
| finding 8's shift invariance, with one-hot coding and `add_intercept=False` | **confirmed**, and fixed by 41635ad the same day (item 8, *Finding 8, for an intercept the columns only imply*) | -- | a four-level one-hot probe gives F = 3.12 and 12.08 before and after adding 1e6 |
| three PALS atlases keep an unassigned `RH_GYRUS` region | **confirmed**: the PALS files name the right hemisphere's unassigned cortex `RH_GYRUS` where the left's is `LH_???` -- in OrbitoFrontal 2,347 vertices against 2,359 | the background pattern of `tools/convert_atlases.py` takes `[LR]H_GYRUS`; the three atlases rebuilt from the toolkit's files, the other 41 unchanged; the coverage floor lowered to 5% so that OrbitoFrontal, at 8.2%, stays | Brodmann 80 -> 79 regions, OrbitoFrontal 48 -> 47, Visuotopic 23 -> 22; OrbitoFrontal labels 203 vertices on the left and 215 on the right, where it labelled the whole right hemisphere; 11,822 bundled regions |
| ConSEAL's docstrings say `viscosity=0` and `step_clamp=inf` "recover the fork's update rule" | **confirmed**: item 8 of the same docstring lists the fork's other differences, its central-difference gradient and its direct composition | reworded: the two settings remove the regularizations the fork does without, and nothing else | -- |
| findings 1, 3 and 6 | open, as the review note's *Still open* records (docs/review-2026-10-05.md): the scores the fit records one step early, the re-registration drift of 0.001 degrees, the area weighting | unchanged (the first fixed since, item 17) | -- |

New in the third round's commits, and from the audit:

| # | Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | a missing family label in a plain list is not refused | **confirmed**: NumPy turns `np.nan` in a list of strings into the string `'nan'`, one more family | the labels are judged as given, on an object view, before NumPy converts them | 21 groups instead of 20 in the review's case; now refused |
| 2 | the constant-column error numbers the caller's column, not `terms=`'s; refusing a column of zeros breaks stratified runs | **confirmed** | the message gives both numberings; a column of zeros is kept, fits as nothing and is set aside by the clustered test, and testing one is refused in words | -- |
| 3 | global coupling keeps an empty one-vertex region | **confirmed**: its NaN diagonal made a constant profile look varying | constancy judged on the entries present, NaN aside; bit for bit on matrices without NaN | on the tutorial subject in Schaefer-1000, the other regions moved by up to 0.005; now not at all |
| 4 | the 0.5 cutoff of the Jacobian calibration skips coarse grids | **confirmed**: the 12-vertex icosahedron reads 0.463 for the identity at every vertex | the calibration skips only vertices within `AXIS_CLEARANCE` of the axis, and refuses a vertex clear of it where the scheme reads nothing positive | the identity's Jacobian 0.463 -> 1 exactly; the bundled grids unchanged |
| 5 | HDF5: triangle indices split at 5,120 faces a hemisphere; an area of the wrong length reported as an endpoints error | **confirmed** | triangles split at `2 V - 4`, a closed spherical mesh's faces (5,120 at ico4's 2,562 vertices); endpoints split at the connectivity's own count, the area left to `load` and `sbci validate`, which name it | -- |
| 6 | `StationaryWarp.invert()` with a rotation is not exactly reversible; assigning `warp.rigid` changes nothing | **confirmed**: the inverse carried the field to the rotated frame by interpolation, and `rigid` was a plain attribute | the warp records which of its two parts comes first, so the inverse -- the flow of `-v`, then `R'` -- is computed, not interpolated; `rigid` is a property that checks the rotation and re-flows | a double inversion changed the field by 1.0e-3 on ico4 and 4.3e-3 on ico3 (2.8% in the review's case); now 0, bit for bit |
| 7 | documents: the ConSEAL docstring's item 16 and its "1e-4"; a passage on the superseded `local_test` rule; `write_cifti` refusing `.DCONN.NII` | **confirmed** | that item 16 restated by axis, with the cotangent layout's share; this file's item 9's row marked superseded; file endings compared without case by the exchange file, its companions and `save_map` | 1.7e-4 about the coordinate axis, 1.3e-3 to 1.5e-3 about others, nearly all the layout's |

One more, found while fixing: 41635ad's check of the default `terms` counted a
column of zeros as an intercept column, so a dummy-coded design beside one
would have tested its implied intercept. Zeros no longer count.

None of these touch the recorded paths on the released subjects: the
registration never inverts a warp and holds its rotation first, as before;
coupling at the vertex level has no NaN; every vertex of the bundled grids
is clear of the axis and read above one half for the identity; and the
designs of docs/RESULTS.md carry no column of zeros. Recorded on the released
subjects with `tests/reference/record_outputs.py` before and after, on the
same AMD EPYC 7702 nodes, all 32 arrays are identical, bit for bit
(`outputs-base4.npz` against `outputs-review4.npz`).

## 12. The fifth list, 6 October 2026 -- THE COHORT LOADER AND THE EXPORT

A fifth list re-checked the fourth's items against 6bf5ff0, before 0073e24
answered them, so its "still present" items are the ones 0073e24 fixes; and it
audited the two modules new that day, `sbci.cohort` (6bf5ff0) and
`sbci.export` (168bc87). Each finding was checked with a probe first; all
hold.

| # | Finding | Verdict | Fixed by |
| --- | --- | --- | --- |
| 1 | (high) SC and FC from different sessions are paired silently | **confirmed**: sessions were checked one modality at a time, so a subject with SC only from session 1 and FC only from session 2 was paired across visits | a subject's files have to share a session across the modalities; `session=` takes one label for all, or `{modality: label}` to pair two visits on purpose, and the report gives each file's session |
| 2 | files `sbci.load` reads are invisible to the loader | **confirmed** for all four: an id holding underscores, a value holding hyphens, `_SC` and `.H5`, a folder behind a symbolic link; with a table they read "no sc file" | names parsed by their parts, the id running to the first `key-value` part, values free to hold hyphens, case ignored in the modality and the ending; folders walked following links, each real folder once; a `.h5` file whose name says no subject is listed as not read |
| 3 | the check accepts files `load` refuses | **confirmed**: coordinates of the wrong shape; an endpoints group without `vertex_out` | the structure `load` checks is checked, without reading the arrays: the coordinates' shape, every dataset of the endpoint group present and of one streamline count; `validate=True` reads the values, endpoint indices among them |
| 4 | `mismatch="exclude"` takes a majority per setting | **confirmed**: majorities taken one setting at a time can name a combination no file has, keeping no one, and the outcome depended on the order of the modalities | the most common way of making the files -- every setting of every modality together -- is kept; a tie is refused |
| 5 | a table with two columns of one name uses the last | **confirmed** | refused |
| 6 | the small ones: `session=1` and `ses-01`; a bandwidth of 2 and 2.0; `none` read as missing; a required column named like a report field; frames recorded run by run; the command line | **confirmed** | labels compared as numbers where both are; settings compared as numbers; missing cells are pandas' defaults, case and all, and `missing=` sets others; a clashing column refused; frames summed; `--subject-column` and `--exclude`, and the report written when the cohort is refused (a `CohortError` carries it, marking everyone out and the odd ones' settings) |
| 7 | (export) an atlas of the wrong size is used, and a short one leaves a partial file | **confirmed** | refused before anything is written, and every file written under a temporary name, moved into place once complete |
| 8 | (export) the CIFTI map's area-weighted mean is not "unchanged" | **confirmed** with the wall masked: 2.5e-5 for a uniform random map, 4e-4 and 1.5% in the review's | the documents say it holds exactly for a map written whole |
| 9 | (export) a region seed's regional means equal `to_atlas(how="mean")` for SC only | **confirmed**: FC is averaged through Fisher z pair by pair (0.010 and 0.040 in the review's tests) | the documents say so |
| 10 | (export) the small ones: `to_table(names="ab")`; a BIDS `hemi-L` name; infinite values; masked arrays; the test of the bundled order | **confirmed** | a string is one name; a `hemi-` entity is set per hemisphere; infinite values refused; masked entries missing; the test now also places the icosahedron's twelve corners where FreeSurfer's fsaverage4 has them, which a symmetric relabeling, carrying triangles onto triangles too, cannot |

On the lab's data the loader still takes all eleven released subjects and 943
of the 946, the three without a fluid-intelligence score out, in 37 seconds
for the 946: the endpoint groups are checked by their structure, as `load`
checks it, since reading their values as well, which `validate=True` does,
took six minutes.

## 13. The sixth list, 6 October 2026 -- ERRORS IN A COVARIATE'S UNITS, AND A PARTIAL SESSION MAPPING

Two findings, both high; each was checked with a probe first, and both hold.

| # | Finding | Verdict | Fixed by |
| --- | --- | --- | --- |
| 1 | (high) `local_test`'s standard errors and intervals are wrong when a covariate is in large units | **confirmed**. The review's case, a raw streamline count near 1e7: a standard error of 9.2e-9 against statsmodels' 1.3e-7, and a 95% interval excluding zero at p = 0.24. In a probe with age and a count of 8.6e6 on average beside the intercept, the count's error was 6.0e-8 against 8.2e-8 and the intercept's 3.6e-4 against 1.10, and the count's interval excluded zero at p = 0.070; with a date in seconds the intercept's and age's errors were missing and the date's 2,400 times too small. The estimates and p-values were right. The covariance was `pinv(X'X)`, whose condition is the square of the design's, and its cutoff dropped the small direction; the estimability check built its projector from `pinv(X)`, accurate only to the design's conditioning, against a tolerance of 1e-8, and so refused contrasts; the cluster-robust bread inverted `X'X` as well | the rank, the fit, the covariance -- classical, and the sandwich's bread -- and estimability are computed from the design with its columns scaled to unit length, through its singular value decomposition, so a covariate's units change none of them; the covariance is kept as the decomposition's factor, and the cluster-robust sandwich formed from its orthonormal columns, because forming `B M B` from the inverse loses the square of the conditioning again (1.4e-6 between two p-values that should agree, which the first version of the fix left and CI caught on two platforms; 3e-12 now). Errors, intervals and a contrast's error agree with statsmodels' to 1e-7 with the count and with a month of dates in seconds, cluster-robust ones too, and dates in seconds within one day, worse conditioned than statsmodels can be checked against, test exactly as the same dates in days |
| 2 | (high) a partial `session=` mapping pairs two visits again | **confirmed**: `session={"sc": "1"}` chose SC's session and switched off the one-visit check, so sub-02's SC from session 1 was paired with its FC from session 2 | a mapping names every modality asked for, and one left out is refused with what to write instead; `None` in a mapping takes the files without a session label, which is how the HCP's unlabelled SC is paired with one day's FC, `{"sc": None, "fc": "REST1"}` |

On the 943 young adults the numbers USAGE quotes from
`tests/reference/inference_probe.py` -- the estimates, the cluster-robust
intervals and the counts -- come out as before at the precision quoted: at
that cohort's streamline counts the inverted `X'X` was still accurate enough.

## 14. The seventh list, 6 October 2026 -- CONTRASTS, DESIGNS, THE LOADER'S EDGES

Eight findings, six medium and two low; each was checked with a probe first,
and all hold.

| # | Finding | Verdict | Fixed by |
| --- | --- | --- | --- |
| 1 | (medium) a contrast on a rank-deficient design with a large-scale column gets the wrong degrees of freedom | **confirmed**, once the contrast weighs the large column: drawn from the design's rows, 2-row contrasts got 1 degree of freedom and single rows were refused as collinear, in 60 of 60 trials beside a column at 1e7 to 1e12. Contrasts on the dummies alone came out right, which is why a first probe missed it. In the column's own units the contrast's rows look parallel, so their null space, the reduced model and its rank were wrong | the null space is found on unit-length columns, where `c @ beta` is `(c / norms) @ (norms * beta)`; the F and its degrees of freedom equal those of the same model with the column in unit units, in every trial |
| 2 | (medium) `design(intercept=False)` drops one level of each factor | **confirmed**: names `site[B]`, `site[C]`, `age`, a model with neither an intercept nor the reference level | the first factor is coded with a column for every level, standing in for the intercept, as R's `~ 0 + site` does; age's p equals statsmodels' and the design with an intercept's |
| 3 | (medium) `load_cohort(missing="-999")` splits the string into characters | **confirmed**: the markers became `-` and `9`, so -999 stayed a score, and they replaced the defaults, so `NA` became a group level | a string is one marker, and markers are added to pandas' defaults, as pandas' `na_values` are; `keep_default_missing=False` reads only the markers named |
| 4 | (medium) the session refusal ignores `exclude=` and writes no report | **confirmed**: it was raised before the caller's exclusions were looked at, with an empty report | a subject the caller leaves out is not held to the one-visit rule, and a refusal carries the report: the subjects whose files cross sessions say which, everyone else is out with the cohort |
| 5 | (medium) `icc` returns 1.0 for a feature that never varies | **confirmed** for constants such as 0.1 and 0.3, where the mean squares are rounding, which over themselves read 1 | a feature agreeing with itself to ten significant digits does not vary, the rule `local_test` uses, and its ICC is missing |
| 6 | (medium) `design()` decides factor or number by whether the values parse as numbers | **confirmed**: codes `"01"`, `"02"`, `"03"` made one numeric column, and codes 2, 3, 10 taken as a factor had 10 as the reference, sorted as text | text is a factor whatever it reads as and numbers are numbers unless `categorical=` says otherwise; levels go in their natural order, as numbers when every one reads as a number; the loader keeps codes written with a leading zero as text, where it read them as numbers (revised in item 15: one padded value turned a column of ages into a factor) |
| 7 | (low) a backup copy becomes a phantom subject | **confirmed**: `sub-01_old_sc.h5` became subject `sub-01_old` | a file named like another subject's with a suffix is one more of that subject's files, a duplicate it reports and leaves the subject out for, unless a table lists it as a subject of its own (revised in item 15: that swallowed real subjects with underscores in their ids) |
| 8 | (low) a re-smoothed half keeps the full streamline count in its metadata | **confirmed**: 20,000 recorded for 10,000 streamlines | smoothing records the count of the streamlines it smoothed; on the released files the recorded count and the stored endpoints agree, 865,820 for the first subject of the cohort |

## 15. The eighth list, 6 October 2026 -- ITEM 14'S EDGES, AND SEVEN SMALL ONES STILL STANDING

Three findings on item 14's changes, and seven low items from earlier lists
that still stood; each was checked with a probe first.

| # | Finding | Verdict | Fixed by |
| --- | --- | --- | --- |
| 1 | (medium) one zero-padded value turns a whole numeric column into a factor | **confirmed**: an age column with one `07` among `30, 25, 41, 33` was read as text, and `design()` made it four columns, silently | item 14's rule is reverted: a column of numbers is read as numbers, as pandas reads it, and a warning names a column holding a number written with a leading zero, and how to make it a factor if it is codes |
| 2 | (low-medium) the backup-copy merge can swallow a real subject | **confirmed**: with no table, `sub-NDAR_INV1_2` was merged into `sub-NDAR_INV1`, and both were left out for two files | without a table to say which, a name extending another subject's id is set aside and reported -- neither read as a subject nor merged; a table that lists it reads it as one |
| 3 | (low-medium) some contrasts still lose a degree of freedom | **not reproduced** on Longleaf, in about 1,000 trials on numpy 2.2 and 2.5 (dates over a year, a day, minutes; contrast rows drawn from the design), against 4 to 15 of 40 on the review's machine. The cause it names is real: taken as the null space of the contrast alone, the reduced model held a direction the design cannot see, a column of rounding, which `_rank` re-normalized | the reduced model is built inside the design's row space, `U S` times the null space of `C D^-1 V'`, so no such direction can enter, and its rank is `rank(X) - q` exactly; a test against the same model with the dates in days guards it |
| 4 | (low) the left/right GIFTI pair is not written all-or-nothing | **confirmed** with the left name taken by a folder: the right hemisphere was written alone | both targets are checked before either is written |
| 5 | (low) temporary-name writes replace read-only or symbolically linked targets | **confirmed**: a link was replaced by a plain file, its target untouched, and a read-only file was overwritten | a write goes through a link to its target, and a read-only file or a folder is refused, as a plain write would refuse it |
| 6 | (low) ConSEAL's hidden rigid-last flag after `invert()` | **confirmed**: the order of a warp's rotation and flow was private, and the exported warp described the other order for an inverted warp | `StationaryWarp.rigid_after` says it, and `EndpointWarp` keeps `lh_rigid_after` / `rh_rigid_after` with the warp, saved and loaded |
| 7 | (low) `Endpoints.take` reads a 0/1 integer mask as indices | **confirmed**: streamlines 1, 0, 1, 0 | 0s and 1s, one per streamline, are refused, saying how to pass a mask or indices |
| 8 | (low) a file reached twice through links counts twice | **confirmed**: two folders, one a link to the other, made `2 sc files` of one | every real folder is searched once and every file taken once under its name; two names for one file stay two subjects |
| 9 | (low) the endpoint arrays' width is not checked | **confirmed**: positions two wide passed the header check, which `load` refused | the header check gives each dataset its shape, `(S, 3)` for the positions |
| 10 | (low) `to_table` does not record which contrast was tested | **confirmed** | a `tested` column holds the hypothesis in the design's names, `age_band[31-35] - age_band[26-30]` or the tested columns |

## 16. A recheck of everything, 7 October 2026 -- NON-FINITE SCORES, TOO FEW GROUPS, THE LOADER'S TABLES

After the tenth list, four reviewers read the code, the notebooks and the
documents again, independently, and each finding was reproduced with a probe
before anything changed. The code's findings are here; the notebooks' and the
documents' are in docs/review-2026-10-05.md, *The recheck after the tenth
list*. Every fix comes with a test that fails before it: 18 of the 19 added,
the nineteenth pinning `icc`'s rounding threshold, which did not change.

| # | Finding | Verdict | Fixed by |
| --- | --- | --- | --- |
| 1 | (medium) one infinite score makes every column of `local_test` NaN | **confirmed**: LAPACK's least squares scales the whole right-hand side by its largest entry, so one `inf` turned every column's statistic, coefficients and p-value NaN, in the classical, `groups=` and permutation tests alike | a column holding a non-finite score is left out of the solve and comes back NaN on its own; the others are as they were |
| 2 | (medium) a CSV row longer than its header is cut short | **confirmed**: an unquoted comma inside a value shifted the row, and `02,Site 1, 2,31` under `subject,site,age` read the age as 2 | a row whose cells beyond the header are not empty is refused, naming the line; trailing empty cells stay allowed |
| 3 | (medium) `missing=` markers match only text written exactly as given | **confirmed**: `-999.0` in a CSV and the number -999 in a dict table passed `missing="-999"`, and `missing=-999` raised a TypeError | a marker that reads as a number matches that number however a cell writes it, as pandas' markers do, and one number is one marker |
| 4 | (medium) `design()` applies the loader's default markers again | **confirmed**: a site coded `NA`, kept by `load_cohort(keep_default_missing=False)`, was refused by `design()` as missing, with advice that could not help | `design(missing=...)`, by default the loader's markers: `missing=()` takes `NA` as a level, and the refusal says so |
| 5 | (low-medium) `MISSING` lacks five of pandas' defaults | **confirmed**: `#N/A N/A`, `-1.#IND`, `-1.#QNAN`, `1.#IND` and `1.#QNAN` were read as text, and one of them made an age column a factor | added |
| 6 | (low-medium) a cluster-robust test of more combinations than there are groups less one | **confirmed**: with two groups and two tested columns the robust covariance is singular, and F came out near 1e17, or below zero | refused: G groups give a robust covariance of rank G - 1 at most |
| 7 | (low-medium) naming a factor coded in full, with no intercept, tests the intercept with it | **confirmed**: `terms=["site"]` under `design(intercept=False)` tested whether every level's mean is zero, and rejected for data with no site effect | refused when the tested columns are needed to make the design's constant, as the default terms already were |
| 8 | (low) `reference=` on a covariate of numbers is ignored | **confirmed**: the covariate entered as a number, its codes as a trend | refused, pointing to `categorical=` |
| 9 | (low) a non-finite value in a design fails as an SVD that did not converge | **confirmed** | `design()` and `local_test` refuse it, naming the column and the subject |
| 10 | (low) `design()` merges the names `1` and `"1"`, and ravels a 2-D covariate into more subjects | **confirmed** | both refused |
| 11 | (low) empty input raises an IndexError | **confirmed**: `design({"age": []})`, `local_test` on no subjects, `Endpoints.take([])` | each refused with a reason; `take([])` takes no streamlines, and indices that are not integers are refused |
| 12 | (low) `ses-1` and `ses-01` are two visits to the one-visit check | **confirmed**, though `session=1` took both | the check reads a numeric label as its number |
| 13 | (low) the roughness warning fires for a component the cohort leaves nothing for | **confirmed**: rank 4 from a rank-2 cohort said to lower alpha, at a scale of 4e-17 | such a component is named for what it is, noise, and the penalty is not blamed |
| 14 | (low) misleading messages | **confirmed**: a link to nothing named like a subject's file was listed among the names that say no subject; "no subject passed" asked about a table when none was given; a DataFrame passed as a table was taken for a list of paths; a table's repeated column was said to need the maps renamed; a write through a link checked the link's folder rather than the target's, and a read-only folder failed naming the temporary file; `identification` on one feature said the features do not vary | the link is reported under its subject; the table hint comes only with a table; a DataFrame is a table; the refusals name what is wrong |
| 15 | (low) docstrings that said more than the code | **confirmed**: `identification`'s read as if it dropped constant features, and `icc`'s NaN rule left out its rounding threshold and a zero denominator | both say what the code does |
| 16 | (low) an `exclude=` key that matches no subject is ignored | **confirmed**: `exclude={"sub-1": ...}` left `sub-01` in without a word | warned about |

The reviewers also checked, against statsmodels and direct computation, and
found right: ordinary least squares (coefficients, errors, F, degrees of
freedom, contrasts, partial R-squared); the cluster-robust errors, Wald F and
intervals, with covariates in large units too; rank-deficient designs against
a full-rank recoding; the size of the Freedman-Lane permutation test; and
`icc` against a two-way ANOVA. The suite's own comparisons pin these.

## 17. The scores of the returned vectors, 7 October 2026 -- THE FIT AND `project()` AGREE AT ANY TOLERANCE

`ConConBasis.Fit` alternates two updates for each component -- the subjects'
scores `s_i = psi' R_i psi`, normalized, for the current vector; then the
vector, the leading eigenvector of `sum_i s_i R_i` -- and stops after a vector
update. So it returns each vector with the scores of the vector before it, and
deflates the residual with those. The stopping rule watches the objective,
which near the optimum changes with the square of the vector's change, so at
the default `tol_outer` of 1e-3 the vector, and the scores with it, still move
by about the square root: on the prediction notebook's rank-15 fits of 240
young adults the recorded scores lay 1.6% to 2.0% of the largest score from the
returned vectors' own (up to 3.6% in a component, every component still
correlating at 0.9997 or more), and on the four released subjects at rank 4
projected over fitted ran from 0.91 to 1.13. The documents had said 1e-3, which
holds only for well-separated synthetic components; on a synthetic cohort a
hundredfold tighter tolerance shrank the gap only tenfold. The reference's own
sparse branch (`auto_sparse`) takes the scores again after it moves the
vector; the dense path does not.

Since 7 October 2026 the port scores the vector it returns -- one more pass
over the cohort per component -- and deflates with those scores. There is no
switch back to the reference's pairing: it pairs a vector with another
vector's scores, which nothing downstream should want. `project()` then
returns `Reduction.scores` on the training cohort to rounding at any tolerance
(4e-15 to 6e-15 of the largest score on the refitted folds, 1e-16 on synthetic
cohorts), and each subject's residual keeps nothing along a removed component.
Against the MATLAB reference (item 5's table): run to its fixed point
(`tests/reference/fpca_fixed_point.m`) the two agree on every component to
1e-15; at its default tolerance the first vector agrees to 2.3e-16, the port's
scores are that vector's own (1.5e-15), the reference's recorded ones 2.4e-5
from them, and the later components, fitted to what the scores leave, move by
1e-9 to 6e-5.

What it changed, each measured before and after on the same data:

| Where | Before | After |
| --- | --- | --- |
| synthetic cohorts of item 5, ten subjects at rank 4, three starts (`tests/reference/final_scores_probe.py`) | | the same significant components; adjusted p within 0.015 and r within 0.009 of before; the share explained up by at most 8e-6 |
| the four released subjects at rank 4, recorded before and after on one CPU type | projected over fitted 0.91 to 1.13 | 1, to rounding; the scores moved by 0.3% to 0.7% of the largest, the basis by 2.8e-6, the share explained not in five decimals; every output that is not the reduction's is bit for bit what it was |
| the reliability notebook, rank 20 on 30 subjects | 32.4% of the norm | 32.1%, a few components in another order; the quoted reliabilities unchanged |
| the prediction notebook, rank 15 on each fold's 240 | training scores 1.6% to 2.0% of the largest from the vector's own | 4e-15 to 6e-15; sex from the components alone AUC 0.658 to 0.662, fluid intelligence r 0.038 to 0.039, the separable form's cost 0.169 to 0.164; no conclusion moved |
| the 943 of RESULTS.md, rank 20 | sex in 11 components, the strongest at adjusted p 2e-8 | 13 (10 and 19 now among them), the strongest at 1e-7; fluid intelligence still in none (component 13 at 0.064 both times) |

## 18. The eleventh list, 7 October 2026 -- PERMUTATION TIES, THE FC EXCHANGE AT THE WALL, ConSEAL'S AREA WEIGHTS AND TURNS ABOUT z

A further review found seven bugs that change results and eight failures on
less common paths. All fifteen hold. Each fix comes with a test that fails
before it (sixteen tests), and `tests/reference/eleventh_list_probe.py`
records the sizes on the released subjects and the bundled data.

| # | Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | permutation p-values too small for discrete scores | **confirmed**: permuted F values equal to the observed one in exact arithmetic fell on either side of it in floating point, and only those above counted | a permuted F within `1e-9 (F + d/q)` of the observed one counts as a tie, as a permutation test counts it | the same discrete scores scaled by 1, 7.3, 0.01 and 1e5 gave p 0.046, 0.044, 0.050 and 0.022 against an exact 0.058 over the same draws; now 0.058 every time |
| 2 | the FC exchange file shrinks correlations beside the medial wall | **confirmed**: the operator averaged the wall's empty rows in as zeros | FC goes through the operator restricted to cortex, its rows renormalized; a density keeps the whole one | 407 fsLR vertices straddle the wall; their correlations were scaled by 0.125 to 0.889 -- halved at 69, cut by more than half at 157 |
| 3 | the documented atlas-level coupling takes SC's mass, against FC's mean correlation | **confirmed**: USAGE's recipe used `to_atlas`'s default for SC, which grows with the regions' size | the recipe takes SC's mean density, `how="mean"`; the quoted figures re-measured | on the eleven, 0.271 to 0.370 per subject (the tutorial subject 0.312) becomes 0.329 to 0.421 (0.364). The review's 0.566 and 0.668 were not reproduced |
| 4 | ConSEAL's `area_weighted=True` does nothing | **confirmed**: raw Voronoi areas scaled the cost by their mean squared, 2.4e-5 on ico4 (1/41,666), below the stopping threshold, so a registration stopped before its first step | the areas weigh cost and gradient in units of their mean | the weighted cost is now within 25% of the plain one, and a weighted registration takes its steps |
| 5 | the rigid search cannot undo a turn about z | **confirmed**: every rotation in the reference's caps takes z to a nearby direction, about an axis perpendicular to z | each shell's best candidates are turned about z as well, over the same span and spacing (`spin_shells`); `strict_upstream=True` keeps the reference's search (until item 22; `reference_conseal.py` since) | on ico3, 8 degrees about z was left 9.4 and 8.0 degrees off (left, right), about a tilted axis 3.3 and 3.0; now 0.37 and 0.11, and 0.33 and 0.20; about x 0.07 either way. The search takes half again as long (14 s against 9 there) |
| 6 | `fsaverage_sphere_ico4.npz` maps 27 grid vertices to the wrong fsaverage vertex | **confirmed**: the file was built by matching the inflated surfaces, nearest to nearest; the pipeline's `mapping_avg_ico4.npz` lists grid vertex `i` as fsaverage vertex `i` | rebuilt from that identity (`tools/build_template_spheres.py`) | 27 vertices (14 left, 13 right) 0.49 to 0.50 degrees off on the standard sphere, which `migrate_warp` carried into every migrated warp there; the other 5097 unchanged |
| 7 | an endpoint index past its hemisphere switches hemisphere through `save()` and `load()` | **confirmed**: left vertex 2600 is stored as whole-grid 2600 and read back as right vertex 38 | `Endpoints` refuses vertex and triangle indices outside the hemisphere, with the existing hint about one-based indices | -- |
| a | following `align()`'s advice for a supplied grid on the axis misplaces its warps in `migrate_warp` | **confirmed**: the advised `rotate_off_poles` turned the grid, and nothing recorded the turn | `align()` turns such a grid clear of the axis itself and records the turn in every warp; `grid_rotations=` records a turn made beforehand | the bundled sphere's own turn is 17.2 degrees per hemisphere |
| b | ENCORE accepts a density that is not symmetric, and one stored as a single triangle never registers | **confirmed** | refused, with the advice to form `D + D.T - diag(D)` | -- |
| c | global coupling is NaN for every one-vertex region | **confirmed**: the global form keeps the diagonal, and `to_atlas(how="mean")` leaves such a region's NaN there | a NaN on the diagonal is left out of that profile | Schaefer-1000's 21 one-vertex regions, all finite now |
| d | the ARPACK fallback fails on SciPy 1.10, which the package allows | **confirmed**: SciPy 1.10 multiplies an operator by a matrix in (n, 1) columns, which the deflation broadcast to (n, n) -- "cannot reshape array of size 202500 into shape (450,1)" | the operators flatten what they are given | the forced fallback now runs under SciPy 1.10.1 |
| e | the cohort loader accepts `.H5` files that `sbci.load` refuses | **confirmed** | `load` and `save` take the suffix in any case | -- |
| f | `write_cifti` writes its 17 GB onto the final name | **confirmed**: a failure part way left a truncated file there | the three files go under temporary names and are moved into place once all are written | -- |
| g | the colour scale is not centred on zero when one bound is given | **confirmed**: `vmin=-0.5` alone drew from -0.5 to the map's largest value | on a symmetric scale a single bound is mirrored | -- |
| h | the release tools would publish HCP-derived data under CC-BY-4.0 without the HCP's terms | **confirmed**: `zenodo_upload.py` set open access and `cc-by-4.0`, and the bundles carried no terms | every bundle, the bundle folder and the cohort manifest carry the HCP's terms (`DATA_USE.txt`); the Zenodo record is restricted, its access conditions the terms, with no licence, and bundles without the terms are refused | -- (the tools were removed with the Zenodo route on 11 October 2026, item 20) |

## 19. The paper's ConSEAL beside the package's, 7 to 11 October 2026 -- THE PAPER'S SETTINGS AND KERNEL WIDTH BY DEFAULT, THE VELOCITY FIELD KEPT

The package ports the public ConSEAL. The paper's experiments ran in the
author's research fork (item 7 lists the copies), which differs in its update
-- each step composed directly onto the warp so far, the derivative of `Q` by
ENCORE's central differences -- and in its settings: step 0.1, threshold
1e-6, up to 1000 iterations, no clamp and no smoothing. Before deciding which
the package should follow, the fork was run in MATLAB on the same cases as
the package, with the criteria fixed first: how far the endpoints end from
where a known warp moved them from; then how far they move where nothing
should; then the final mismatch and the time. A folded triangle counts
against a method.

**The cases** (`tests/reference/fork_cases.py`):

- three known warps of the tutorial subject's 803,741 streamlines: the recovery
  figure's smooth degree-4 warp (1.63 degrees on average), the same twice as
  large (3.26), and one built the fork's way, four smaller maps each composed
  onto the last (1.13);
- the split halves of five young adults (386,115 to 466,113 streamlines a
  half), one half registered onto the other, where the right answer is no
  motion;
- the fork's own logged pair of subjects, one fixed and the other moving.

**The fork as it ran** (`tests/reference/fork_case.m`). Its files are
unchanged, but two things had to be worked around. Its compiled interpolation
(`bary_interp_2D_mex`) crashes on Longleaf's CPU and V100 nodes at full size,
even rebuilt from its own source, so a MATLAB stand-in computing the same
`W2 * data * W1'` went first on the path
(`tests/reference/fork_standin/`); it agrees with the MEX to 1.8e-15
wherever the MEX runs. And the working copy of its `Concon.m` (19 June)
returns from `evaluate` the raw adjacency rather than the smoothed connectome
it computes, which made the first step's square root complex; a copy with
the committed line restored went first as well. So run, the pair reproduces
the fork's logged run of 6 May to the printed digit: 0.291389 to 0.199431 in
460 iterations.

**The package** ran at its defaults then, the public code's; at the fork's
settings; and at the fork's settings with the fork's direct composition in
place of the velocity field (`tests/reference/fork_package_case.py`). Every
result was scored by both methods' cost functions
(`tests/reference/fork_evaluate.py`).

| | the fork | package, public settings | package, the paper's settings | package, the paper's settings, composed directly |
| --- | --- | --- | --- | --- |
| known warps: mean error left, degrees | 0.057, 0.114, 0.042 | 0.631, 0.815, 0.540 | 0.091, 0.110, 0.076 | 0.095, 0.143, 0.075 |
| share of the displacement undone | 96% | 52% to 75% | 93% to 97% | 93% to 96% |
| split halves: mean motion, degrees | 0.18 to 0.21 | 0.015 to 0.019 | 0.19 to 0.22 | 0.19 to 0.22 |
| the pair: folded triangles | **111** | 0 | 0 (43 steps refused) | **166** |
| the pair: area ratios, smallest to largest | -4.9 to 32 | 0.23 to 3.1 | 0.040 to 7.4 | -7.2 to 24 |
| the pair: final cost, the package's and the fork's | 0.1391, 0.1994 | 0.1537, 0.2217 | 0.1391, 0.2025 | 0.1367, 0.1997 |
| iterations: known warps, the pair | 50 to 90, 460 | 13 to 39, 67 | 55 to 106, 267 | 56 to 106, 846 |
| time: known warps, the pair | 2.4 to 4.0 min, 18 min | 4 to 12 min, 20 min | 20 to 33 min, 83 min | 19 to 31 min, 4.8 h |

The fork ran on 12 cores and the package on 8, on different nodes, so the
times are rough. What the table shows:

- **The settings make most of the difference.** The public code's stop at a
  change of 1e-4 and undo 52% to 75% of a known warp; at the paper's the
  package undoes 93% to 97% and the fork 96%. On two of the three warps the
  fork ends 0.03 degrees closer than the package at its settings, on the
  third 0.004 further. Composing directly is not what makes the fork better:
  the package composing directly is no closer (0.104 degrees on average
  against 0.092). The rest is the fork's derivative or the scale of its cost,
  1.4 to 2.3 times the package's on the same endpoints, which lengthens the
  same step; neither was isolated.
- **Composing directly folds on real data.** On the pair both direct
  compositions fold triangles, 111 in the fork's warp and 166 in the
  package's, and stretch others 24 to 32 times. A map that folds is not one
  to one: endpoints in a folded triangle land on top of others. The velocity
  field refused the 43 steps that would have folded and stays a
  diffeomorphism; on the known warps nothing folded.
- **The paper's settings fit noise, slightly.** On split halves every method
  at the paper's settings moves the endpoints 0.18 to 0.22 degrees where
  nothing should move, against 0.015 to 0.019 for the public code's, which
  stop after one iteration: about a twentieth of the 4.3 degrees between
  ico4 vertices.
- **The fork is five to eight times faster** at the same settings. The
  package's iteration on ico4, with 803,741 streamlines, spends four fifths
  of its 12 seconds carrying every endpoint with the warp and locating it
  again on the grid, 9.8 s; the density and its derivatives take 1.6 to 1.7
  s, the gradient 0.25 s and the step 0.18 s (`tests/reference/conseal_profile.py`).
  The fork's equivalent takes about 0.6 s.

**The decision.** The package keeps its update -- a stationary velocity
field that refuses a step that would fold, with the analytic derivative --
and takes the paper's settings as its defaults: step 0.1, up to 1000
iterations, threshold 1e-6, no clamp and no smoothing. The fork's direct
composition is not adopted: it folds on real data and gained nothing in the
package. `strict_upstream=True` now takes the public code's settings for any
not given, so it still reproduces the public code's runs, and any setting can
be passed in either mode (until item 22 removed the switch; the `PUBLIC_*`
constants hold those settings since). Two tests fail before the change: the settings each
mode takes, and a known warp on ico2 that the defaults now undo 76% of where
the public code's settings undid 2%.

**What it changes.** Registrations run longer: 55 to 106 iterations for the
known warps and 267 for the pair, tens of minutes for two whole subjects on
eight cores, which `max_iterations` caps. The figures and numbers of
docs/RESULTS.md were already made at these settings, with a cap and a
threshold of 1e-7, and stand. Registering onto one subject needs care at
these settings: item 7's planted bundle was aligned away by the
unregularized update onto a single subject and kept by the public code's
regularization or a cohort template, and USAGE.md says when to pass the
public settings.

### Synthetic warps, 8 October 2026

Three known warps left open whether the fork's lead holds, and whether its
folding is confined to real data. So the fork, the fork with a guard that
refuses a hemisphere's step that would fold a triangle (a copy of its
`Encore.m` changed in those lines only: `tests/reference/fork_case.m` with
`guard=true`, and `fork_standin/fold_flip.m`), and the package at both
settings were run on 24 more known warps of the tutorial subject
(`tests/reference/fork_synthetic_cases.py`): smooth order-4 warps whose
largest displacement is 2, 4, 8, 12 and 16 degrees, and warps built the
fork's way from four composed maps, three seeds each, scored by
`tests/reference/fork_synthetic_evaluate.py`. The mean endpoint error left,
in degrees, over the three seeds:

| known warps, mean displacement | the fork | package, the paper's settings | package, public settings |
| --- | --- | --- | --- |
| smooth, 0.81 degrees | 0.041 | 0.072 | 0.494 |
| smooth, 1.63 | 0.052 | 0.086 | 0.642 |
| smooth, 3.25 | 0.103 | 0.104 | 0.789 |
| smooth, 4.87 | 0.185 | 0.122 | 0.885 |
| smooth, 6.48 | 0.290 | 0.142 | 0.977 |
| composed, 0.56 | 0.039 | 0.067 | 0.414 |
| composed, 1.11 | 0.046 | 0.080 | 0.569 |
| composed, 2.22 | 0.068 | 0.096 | 0.716 |

- **Nothing folded.** No method folded a triangle on any of the 24, and the
  guard never refused a step, so the guarded fork ran exactly as the fork.
- **The fork is closer on small warps, the velocity field on large ones.**
  Up to about three degrees the fork ends 30% to 45% closer than the package
  at the same settings, and it is closest in 17 of the 24. Beyond that the
  package is: at 4.9 and 6.5 degrees it leaves 0.12 and 0.14 degrees against
  the fork's 0.19 and 0.29, its 95th percentile 0.41 against 0.72 at the
  largest, and its warps stretch triangles less (area ratios 0.34 to 3.8
  against 0.16 to 6.9). Over all 24 the fork undid 95.8%, the package 94.3%
  and the public code's settings 60.4%.
- **The fork is faster:** 1.4 to 14 minutes a warp against 8 to 52.
- **On the real pair the guarded fork folds nothing but stops early,** at
  its first step refused on both hemispheres: 114 iterations and 6 refused
  steps, ending at the package's cost 0.1413 where the package and the
  unguarded fork reach 0.1391.

So the decision holds where the package is used most, registering one
subject onto another, which moved the pair's endpoints five to six degrees
on average: in that range the velocity field is ahead on synthetic warps too,
and it stays unfolded on real ones. For small corrections the fork was more
accurate. Its edge there is not the direct composition, since the package
composing directly was no closer (above); it is where the stopping threshold
falls and the fork's kernel (*Small warps*, below).

### Small warps, 11 October 2026

**The descent is stopped early, not wrong.** Run on past its threshold
(`tests/reference/conseal_convergence_probe.py`: 300 iterations at the
paper's settings, the step 0.1 or 0.2), the package's error on three small
known warps keeps falling, and ends far below the fork's:

| known warp | step | stopped at 1e-6 | after 300 iterations | the fork |
| --- | --- | --- | --- | --- |
| smooth, 0.89 degrees (svf035, seed 101) | 0.1 | 0.077 (48 iterations) | 0.0055 | 0.042 |
| smooth, 1.78 (svf070, seed 101) | 0.1 | 0.089 (74) | 0.0115 | 0.054 |
| composed, 1.04 (comp100, seed 201) | 0.2 | 0.060 (35) | 0.0055 | 0.045 |

The paper's threshold is an absolute change in cost, and a small warp's cost
is small -- 0.003 to 0.012 at the start here, against 0.20 for two subjects --
so the same 1e-6 stops it relatively early. A threshold relative to the cost a
registration starts from does not: 1e-5 of it stops the real pair at the same
267 iterations as 1e-6, and the three warps above at 0.019, 0.038 and 0.018
degrees.

**The fork's kernel.** The fork smooths with its own `kde/spherical_heat_kernel.m`,
whose MATLAB normalized Legendre functions give weights `(2l+1)^(3/2)/sqrt(2)`
with no constant term, where the package's heat kernel weighs `(2l+1)`: cut
by the same rule at 12.0 degrees rather than 21.5, 29 nonzeros a row rather
than 89, its rows spread 6.0 degrees rather than 8.1
(`ForkKernel` in `tests/reference/fork_package_case.py`). The package at the
paper's settings with the fork's kernel, scored as above:

| known warps, mean displacement | the fork | package, the paper's settings | package, the fork's kernel |
| --- | --- | --- | --- |
| smooth, 0.81 degrees | 0.041 | 0.072 | **0.038** |
| smooth, 1.63 | 0.052 | 0.086 | **0.043** |
| smooth, 3.25 | 0.103 | 0.104 | **0.050** |
| smooth, 4.87 | 0.185 | 0.122 | **0.058** |
| smooth, 6.48 | 0.290 | 0.142 | **0.072** |
| composed, 0.56 | 0.039 | 0.067 | **0.037** |
| composed, 1.11 | 0.046 | 0.080 | **0.041** |
| composed, 2.22 | 0.068 | 0.096 | **0.046** |

It is the closest of every method on all 24, folds nothing, and undoes 97.0%
on average (the fork 95.8%). On the real pair it ends at the fork's own cost
0.2017 (the fork 0.1994 with 111 folded triangles, the guarded fork 0.2023),
with no fold, in 191 iterations and 60 minutes. Part of its edge is the
threshold again: the narrower kernel raises a small warp's starting cost 1.7
to 1.8 times, so 1e-6 stops it relatively later.

### The default, 11 October 2026

Three more runs on the same 24 warps settled which becomes the default
(`tests/reference/fork_package_case.py` with `heat0027`, `relative` and
`heat0027min`, scored by `fork_synthetic_evaluate.py`): the package's own heat
kernel narrowed to the fork's width, `sigma=0.0027` (cut at 12.4 degrees
against the fork's 12.0, 32 nonzeros a row against 29, its rows spread 5.6
degrees against 6.0); the threshold made relative, 1e-5 of the starting cost;
and the two together, the floor taken only where it is the smaller of the two
thresholds. The mean error left, in degrees:

| known warps, mean displacement | the fork | the paper's settings | the fork's kernel | heat kernel at 0.0027 | 1e-5 of the start | 0.0027 and the smaller threshold |
| --- | --- | --- | --- | --- | --- | --- |
| smooth, 0.81 degrees | 0.041 | 0.072 | 0.038 | 0.032 | 0.018 | **0.009** |
| smooth, 1.63 | 0.052 | 0.086 | 0.043 | 0.035 | 0.037 | **0.018** |
| smooth, 3.25 | 0.103 | 0.104 | 0.050 | 0.041 | 0.075 | **0.038** |
| smooth, 4.87 | 0.185 | 0.122 | 0.058 | **0.049** | 0.115 | **0.049** |
| smooth, 6.48 | 0.290 | 0.142 | 0.072 | **0.062** | 0.162 | **0.062** |
| composed, 0.56 | 0.039 | 0.067 | 0.037 | 0.031 | 0.013 | **0.007** |
| composed, 1.11 | 0.046 | 0.080 | 0.041 | 0.034 | 0.026 | **0.013** |
| composed, 2.22 | 0.068 | 0.096 | 0.046 | 0.038 | 0.051 | **0.026** |
| share undone, all 24 | 95.8% | 94.3% | 97.0% | 97.5% | 97.7% | **98.9%** |

The width is what helped, not the fork's formula: the package's own kernel
at the fork's width beats the fork's kernel at every size. A relative
threshold alone stops a large warp earlier than the absolute one, its starting
cost being large; the smaller of the two keeps both. Together they are the
closest of every method on every warp, or tied with the narrowed kernel alone
on the six largest, where the floor never binds; nothing folds, and a whole
subject takes 76 to 155 iterations, 15 to 42 minutes on eight cores. On the
real pair, where no truth is known, the floor does not bind either (171
iterations); the narrower kernel matches a less smoothed picture of the two
subjects and moves the endpoints 5.7 degrees on average against 6.3, folding
nothing. The fork itself, run 300 iterations past its stopping rule
(`fork_case.m` with `max_iters=300, threshold=-1`; `fork_long_score.py`),
reaches 0.008, 0.029 and 0.012 degrees on the probe's three warps, where it
had stopped at 0.042, 0.054 and 0.045: its stop limited it as the package's
limited the package.

So since 11 October 2026 the defaults are the paper's settings with the
paper's kernel width, `sigma=0.0027` for this heat kernel, and a floor of
1e-5 of the starting cost under the threshold (`relative_threshold`);
`strict_upstream=True` restores the public code's `sigma=0.005` and no floor
(until item 22; `PUBLIC_SIGMA` and `PUBLIC_RELATIVE_THRESHOLD` since).
Every ConSEAL key of `tests/reference/record_outputs.py` spells out its kernel
and floor, and `consealnarrow_*` records the new defaults.

Verified with the change: the suite on the released subjects passes, 1,069
tests with 14 skipped, and 1,065 under Python 3.10; Tier 4 passes as before,
`strict_upstream=True` keeping the public code's kernel and stop; the API
audit passes 22 of 22, and the checks on a fresh download of the eleven pass.
Recorded before (e8b1aa7) and after, one after the other on one AMD EPYC 7702
node, all 35 arrays the two recordings share are identical, bit for bit,
every ConSEAL key among them (`outputs-e8b1aa7.npz` against
`outputs-default1011.npz`). Redrawn at the new defaults, the known warp of
the tutorial subject comes back to 0.02 degrees in 80 iterations (it was 0.11 in sixty),
and the eleven, each registered onto ConSEAL's own median, grow more alike:
their mean pairwise correlation rises from 0.716 to 0.790 in thirty
iterations, against 0.784 before, every subject's cost still falling when the
thirty stop it.

**Still open.** Locating the endpoints again from their last triangles,
rather than searching the grid afresh each iteration, would make the longer
runs much cheaper without changing a result.

## 20. A review of the whole repository, 11 October 2026 -- 104 FINDINGS, AND WHAT THEY LEFT TO DECIDE

Six reviewers read the repository at db18df5 against its own code: the
statistics and the cohort loader, the files, the command line and the
download, the alignments, the scripts and probes, the user documents, and the
records. Their 109 findings come to 104 once the ones two reviewers both
found are merged; three more turned up while fixing them. Each was reproduced
before anything changed, and every change to the code comes with a test that
fails before it (33 tests). What is left needs a decision rather than a fix,
and is listed at the end, after what was decided the same day.

**The code.** The ones that change results come first.

| Area | Finding | Fixed by |
| --- | --- | --- |
| cohort | a subject the table leaves out, with files from two visits, made the loader refuse the whole cohort | the visit check skips subjects the table leaves out, as the settings vote already did |
| cohort | codes that read as numbers merged: `12_345` and `123_45`, codes past 15 digits, `07` and `7` | a cell holding an underscore, or an integer float64 cannot hold exactly, keeps its column as text, and so do spellings like `07` and `7` that would read as one number |
| stats | a response its nuisance explains all but 1e-5 of read NaN, where least squares tests it soundly | the bar for nothing left to test is 1e-20 of the sum of squares, at rounding |
| reduce | dense ico4 arrays got the plain inner product, connectomes the grid's areas | dense ico4 arrays get the grid's areas too |
| parcellation | a NaN on the medial wall reached every region that labels wall vertices (84 of the 100 entries of the PALS lobes' matrix) | wall vertices are left out of the region weights |
| ConSEAL | the gradient test passed three broken gradients | the gradient is checked against the cost along every basis field, cosine above 0.95 |
| ConSEAL | a non-finite template ran every iteration on NaN, 1000 by default | refused; a NaN improvement stops the descent |
| ConSEAL | at threshold 0, a step refused on both hemispheres was taken again until `max_iterations` | such a step ends the registration, since the next would be the same |
| ConSEAL | a kernel derivative built in one arithmetic ran silently in the other | the derivative records `strict_upstream`, and a mismatch is refused (until item 22 took the switch out) |
| ConSEAL | the callback heard the cost of a step then rolled back, and the rollback forgot its refusals | only kept steps are reported; refusals count |
| ConSEAL | NaN or zero endpoint coordinates were placed on a triangle; endpoints of another grid's resolution were accepted | both refused |
| ConSEAL | strict `rotate()` kept an earlier rigid part; `compose()` on an inverted warp moved it in the wrong frame | reset; refused |
| align | `grid_rotations` took a reflection; every `Alignment` built without one shared a single array | `det > 0` required; one array each |
| download | a Drive download was written onto the final name, so an interruption left a truncated file and `--force` destroyed the verified one | downloaded beside it, verified, then moved into place; ids may be numbers |
| files | `.H5` validated as unreadable; malformed files escaped `SbciError` and crashed the command line; NaN metadata and endpoints on FC passed; `save` wrote arrays `load` refuses, and replaced a link with a file | one name test for load, save and validate; malformed files raise `FormatError` or `InvalidFileError` and `validate` reports them; both refused on reading; checked before writing; written through the link |
| export | an unusable CIFTI target was found only after the 25 GB resampling; a killed export left a partial file no rerun removes | checked first; a fixed temporary name the next write replaces |
| smoothing | hemisphere flags wrapped through int8, fractional vertices were truncated, and an FC file smoothed into a density labelled FC | checked before the cast; refused; refused |
| small | `seed(vertex=)` took floats and bools; `sbci example --seed -1` raised a traceback; `atlases --match` ignored the short names; an example's metadata was shared with the next; an unknown region name escaped `except SbciError`; `load_cohort(missing=None)` raised `TypeError`; the statsmodels checks stopped the whole test run where statsmodels fails to import | an integer required; refused by the parser; matched and listed; copied; `UnknownRegionError`, an `SbciError` and still a `ValueError`; the defaults; skipped with the reason |

**The scripts.** The recorder ran both ConSEAL keys at the defaults, which
became the paper's settings with item 19, so a key no longer held what it had
held: both are pinned to the settings they were first recorded at, and the
paper's are recorded under `consealpaper_*`. The convergence probe read its
stopping points off one iteration short; `fork_case.m` named runs that differ
only in threshold alike; the figure scripts reused cached fits without checking
how they were made, printed components from zero and called the norm's
fraction the variance explained; one colour bar mislabelled its colours; a
probe paired two files by row without checking their subjects; a tool
rewrote the bundled surfaces without checking their triangles; and
`verify_all.sh` counted 17 audit checks of 22 and left one MATLAB reference
out.

**The documents.** Twenty-seven findings in USAGE.md, README.md,
VERIFICATION.md and docs/RESULTS.md, and twenty-nine in the records. The ones
a reader would trip on: the exchange file's read-back used another name; three
cohort snippets passed SC and FC together to methods that take SC; the lint
and build commands failed as written (notebooks linted, `build` missing from
the dev extra); the Tier 4 block ran its test from the wrong folder and never
said its scripts carry the lab's paths; the audit could not reach its count
off Longleaf (the exchange row now follows `SBCI_EXCHANGE_DIR` and is
skipped without the file). The rest were counts and figures left behind by
later items -- the FPCA agreement, the sex counts of RESULTS.md (tested
alone), the reference errors (seven, not three), the atlas files' size --
and cross-references, labels and one garbled sentence.

**Decided the same day.**

- *ConSEAL on small warps.* The paper's absolute threshold stopped a small
  known warp early, at 0.06 to 0.09 degrees where the package, run on,
  reaches 0.005 to 0.012 and the fork 0.04 to 0.05. The paper's kernel width
  and a floor under the threshold are the defaults now (item 19, *The
  default, 11 October 2026*): 98.9% of the 24 known warps undone, the closest
  of every method on 18 of them and tied with the closest on the six largest
  (item 33).
- *The release.* The Zenodo route is gone, with `tools/bundle_hcp_cohort.py`,
  `tools/zenodo_upload.py` and the bundled download in `sbci.download`; the
  eleven on Google Drive stay the released cohort, and a later release will
  come from UNC's research data services. (Superseded by items 29 and 30: all
  946 are released on UNC Dataverse, the eleven among them.)

**Still open, as decisions.** (Item 21 took two of them the same day: ENCORE
now composes as its gradient assumes, and the exchange file takes
`save_map`'s half.)

- *ENCORE's composition* (alignment finding 11): a step is composed after the
  warp, `psi o phi`, while the gradient is taken for `phi o exp(D)`. The two
  agree at the identity (cosine 0.9998) and drift apart with the warp: after
  a 30-degree rotation the cost falls at 0.80 of the predicted rate, against
  0.98 composed the other way. The reference composes the same way.
  Decided by item 21, which composes as the gradient assumes (item 23
  re-measured the two rates as 0.79 and 0.99).
- *Area weighting in the template*: with `area_weighted=True` the cost is
  weighted but the Karcher median is still taken in the plain metric; doing
  it properly means normalizing the densities in the weighted measure
  throughout, not in the median alone.
- *The exchange file at the wall*: the FC `.dconn.nii` gives an fsLR vertex
  with as little as 12.5% of its area in cortex a full cortical row, where
  `save_map` marks a vertex below one half missing. Decided by item 21: the
  file takes `save_map`'s half.
- The WP1 sign-offs of items 3, 4 and 5, the unrestricted PCA of the review
  note, and the endpoints' relocation, most of a ConSEAL iteration.

**Verified.** The whole suite on the released subjects passes, 1,073 tests
with 14 skipped (the slow planted-answer test and the 13 doctest examples
marked `+SKIP`), and under Python 3.10 from conda-forge 1,069 with 18
skipped; Tier 4's 22 MATLAB-reference tests pass and the ConSEAL comparison
agrees as before (the template to 2.0e-5 relative); `scripts/audit_api.py`
passes 22 of 22, `scripts/tour.py` runs, and `scripts/check_hcp_ya.py`
passes on a fresh download of the eleven. Recorded with
`tests/reference/record_outputs.py` before (db18df5) and after, one after the
other on one Intel Xeon Gold 6140 node, all 35 arrays are identical, bit for
bit (`outputs-db18df5.npz` against `outputs-review1011.npz`): none of the
fixes touches a recorded path on the released subjects.

## 21. Two of item 20's decisions, 11 October 2026 -- ENCORE COMPOSES EACH STEP AS ITS GRADIENT ASSUMES, `reference=True` GOES, THE FC EXCHANGE FILE TAKES `save_map`'S HALF

Asked: "for encore's step order, do (a), and once u checked this one is better,
u can remove any reference=True for matlab reference", and "fix 3 by applying
save_map's 50% rule to the FC exchange file".

### ENCORE's order of composition

The registration gradient is taken for the warp `phi(exp_x(D(x)))`: the step
first, then the current warp. The reference -- and the port until now -- composes
the other way, `exp(D)(phi(x))`, moving each image by the step interpolated
where that image lies. The two agree at the identity and part as the warp
grows. `SphericalWarp.compose` now moves each vertex of the base mesh by its
step, finds the face of the base mesh the moved point lies in, and evaluates
the current warp there. The weights are those along the ray through the point,
the solution of `[a b c] w = y` normalized to sum one, not those of its
closest point on the flat face; with them the identity, and any rotation,
carry the point exactly. (Closest-point weights moved the identity by the
face's sag, which the Jacobian test beside the axis caught.)

Checked three ways, with the reference's order put back by
`tests/reference/encore_order_probe.py`:

- **The descent rate.** This is the cost's fall along a step, by central
  differences, over the fall the gradient predicts. With the new order it is
  0.98 at the identity and 0.99 after a 30-degree rotation (ico3, basis order
  4; `test_a_step_is_composed_as_the_gradient_assumes`); after the rotation
  the reference's order gives 0.79 (`encore_order_probe.py rate`; until item
  23 re-measured them this said 0.975 to 0.997, and 0.80).
- **Known warps of the tutorial subject**, registered back as the recovery figure does
  (job 4584482):

| Warp | Off before (degrees) | The gradient's order | The reference's order |
| --- | --- | --- | --- |
| smooth, 0.07 rad, seed 7 | 1.634 | 0.201 (cost to 0.155, 10 steps) | 0.200 (0.155, 11 steps) |
| smooth, 0.07 rad, seed 8 | 1.701 | 0.198 | 0.198 |
| smooth, 0.14 rad, seed 7 | 3.263 | 0.400 | 0.396 |
| smooth, 0.14 rad, seed 8 | 3.405 | 0.407 | 0.409 |
| smooth, 0.21 rad, seed 7 | 4.883 | 0.600 | 0.590 |
| smooth, 0.21 rad, seed 8 | 5.111 | 0.613 | 0.621 |
| rigid turn, 10 degrees | 6.809 | 0.257 | 0.258 |
| rigid turn, 20 degrees | 13.599 | **0.557** (cost to 0.024, 100 steps) | 1.298 (cost to 0.053, stopped after 58) |

  On the smooth warps and the small turn the two orders end within 0.01
  degrees of each other. On the 20-degree turn the reference's order stalls
  with twice the error and twice the cost.
- **The eleven released subjects**, aligned ten steps onto ENCORE's median as
  the cohort figure does (job 4587735). The mean pairwise correlation rises
  from 0.7163 to 0.7608 with the gradient's order and to 0.7606 with the
  reference's. Each subject's cost falls to 0.796 to 0.880 of its start (0.840
  on average), against 0.795 to 0.880 (0.841). Ten steps from the median are
  too few for the two orders to part.

### The switches that reproduced MATLAB's errors are gone

`reference=True` existed so that each correction could be checked against the
MATLAB run. It restored:

- the reference's Legendre recurrence and its clamp on the axis
  (`SphericalGrid`, `tangent_basis`);
- its normalization before the diagonal is zeroed (`Concon`);
- its uncalibrated Jacobian, and since this item its order of composition
  (`SphericalWarp`);
- its lower-triangle projection (`project`).

All of these reached `align` and `Encore` through the switch, and
`conseal.default_grids(reference=True)` gave ConSEAL the same grids. The new
order is better on every measure above, so all of it went.
`strict_upstream=True` stayed for ConSEAL's own errors (item 7's 1 to 4, 7,
12, 13 and 15 to 17, with the public code's settings and the reference's rigid
search), with item 7's entry 14, the recurrence, corrected in both modes,
until item 22 removed it too.

The comparisons with MATLAB now put the reference's errors back into the port's
output instead of switching the port's arithmetic
(`tests/reference/reference_geometry.py`, which reproduced what the switch
built to 7e-16 on a level-2 icosphere with a vertex on each pole; item 23
extended it to vertices beside the axis). Tier 4's
ENCORE rows:

| Stage | Agreement |
| --- | --- |
| areas; exp and log maps; barycentric weights | 1.6; 0.5 and 1.6; 2.0 float64-eps |
| tangent basis, the clamp put back | 12.2 eps (1.0 relative without it: MATLAB's grid has a vertex on each pole) |
| basis divergence, the clamp and the recurrence put back | 5.3 eps (0.46 without) |
| template; warp composition from the identity | 44 eps; 1.0 eps |
| Jacobian, uncalibrated | 4.9e-4 |
| `evaluate`; `evaluate_root` against the reference's result renormalized | 7.1e-4; 5.4e-4 |

The registration comparison (r = 0.99999979) is gone. It needed every one of
the reference's choices, its order of composition among them, so Tier 4 now
has 21 tests. `conseal_compare.py` runs strict ConSEAL on the reference's
geometry and agrees as before: basis 5.3e-15, divergence 2.0e-15, gradient
5.7e-7 and 1.2e-6, the six-iteration trace within 3.5e-7, the template to
2.0e-5. The recorder drops its `alignref_*`, `consealref_*` and
`project_reference` keys. `fourth_list_probe.py` uses the same grid for both
smoothings, since smoothing never reads the harmonic basis.

### The FC exchange file at the wall

The FC `.dconn.nii` used to give an fsLR vertex a full cortical row when as
little as 12.5% of its area lay in cortex. Now a vertex whose cortical share
is at most one half (`sbci.export.COVERAGE`) gets a row of zeros, just as
`save_map` marks it missing. Of the 407 fsLR vertices that straddle the wall,
226 are at or below the half
(`test_the_fc_exchange_and_save_map_leave_the_same_vertices_missing`).

### Verified

On Longleaf the whole suite passes on the released subjects: 1,068 tests, 14
skipped. The affected modules pass under Python 3.10 from conda-forge (234),
and Tier 4's 21 tests pass. `conseal_compare.py` agrees as above.
`scripts/audit_api.py` passes 22 of 22, `scripts/tour.py` runs, and
`scripts/check_hcp_ya.py` passes on a fresh download of the eleven.

The outputs were recorded before (b25f6f2, with its own recorder) and after,
one after the other on one Intel Xeon Gold 6140 node (`outputs-b25f6f2.npz`
against `outputs-phaseb.npz`). Only ENCORE's arrays move:

- the two registrations' costs fall from 0.0127058 and 0.187036 to 0.0127026
  and 0.186965;
- the first subject gets there in 12 steps instead of 15;
- its warped vertices sit 0.008 degrees from where they were (0.033 at most);
- the template, which no warp enters, is identical.

Every other array is identical, bit for bit: the ConSEAL keys, the FPCA fit
and projection, coupling, parcellation, seed and smoothing. The recovery text
of docs/RESULTS.md now says ten steps, as job 4584482 ran it. The three
figures that show ENCORE were redrawn with the new order (jobs 4588256 and
4588257). Every number the documents quote from them held: the recovery
1.63 to 0.20 degrees in ten steps with the cost to 0.15, the migration 0.33
degrees from the true warp (0.32 on the left hemisphere shown) with the map
put back to r = 0.996, and the cohort 0.7163 to 0.7608 with ENCORE and to
0.7900 with ConSEAL, onto a median 21.03 to 25.06 degrees from the subjects
(job 4588257, as job 4570438 printed them at b25f6f2).

## 22. ConSEAL's `strict_upstream` goes too, 11 October 2026 -- THE DEFAULTS UNDO EVERY KNOWN WARP BETTER

Asked: "if conseal's default is better than strict_upstream=True, remove it".

### The comparison

`strict_upstream=True` reproduced the reference's arithmetic, items 1 to 4, 7,
12, 13 and 15 to 17 of item 7, and its rigid search without turns about z
(item 18). It came with the public code's settings: sigma 0.005, step 0.05, up
to 100 iterations, threshold 1e-4 with no floor, a clamp of 0.2 and 5%
smoothing. It was run on the 24 known warps of item 19 and the real pair (job
4598330), by `fork_package_case.py` at 3806775 with a `strict` key,
`dict(strict_upstream=True)`, that was never committed: the job's log records
the commit, two uncommitted files and those settings, and that commit accepted
the mode only with the kernel derivative built in it too. With the mode gone
the run cannot be repeated; its results stay in `cases/`. The defaults beside
it are item 19's `heat0027min` runs, which three reruns on today's code
reproduced exactly: the same errors, iterations and area ranges (job 4598329).
Mean error left over the three seeds of each size:

| Warps | Off before (degrees) | The defaults | `strict_upstream=True` |
| --- | --- | --- | --- |
| svf035 | 0.81 | 0.009 | 0.468 |
| svf070 | 1.63 | 0.018 | 0.613 |
| svf140 | 3.25 | 0.038 | 0.784 |
| svf210 | 4.87 | 0.049 | 0.920 |
| svf280 | 6.48 | 0.062 | 1.286 |
| comp050 | 0.56 | 0.007 | 0.395 |
| comp100 | 1.11 | 0.013 | 0.538 |
| comp200 | 2.22 | 0.026 | 0.693 |

- **The 24 known warps.** The defaults undo 98.9% of each warp on average,
  `strict_upstream` 61.3% (24.6% to 84.8%). It is behind on every one of the
  24, after 4 to 85 iterations, and neither folds a triangle.
- **The real pair.** It reaches a cost of 0.16471 in the common measure (the
  cost at sigma 0.005, in the default arithmetic), against the defaults'
  0.14154. It moved the endpoints 2.64 degrees in 33 iterations, against 5.70
  degrees in 171.

### What went

- **The switch itself.** `strict_upstream` came out of `endpoints_align`,
  `ConSEAL`, `StationaryWarp` (and its `compose` and `_smooth`),
  `EndpointConnectome.evaluate` and `q_transform`, and
  `HeatKernelBuilder.compute` and `cross_validate`. `reference_layout` came
  out of `cotangent_laplacian`. `KernelDerivative` lost its `strict_upstream`
  field, and with it the refusal of a derivative built in the other
  arithmetic, which can no longer arise.
- **The public code's settings stay** as the `PUBLIC_*` constants, to pass
  explicitly.
- **The tests.** Four tests that exercised strict mode alone went. The others
  keep their assertions on the defaults, and the component-wise smoothing they
  measure against is two lines of the test. (Item 23 put back the halves that
  measured the reference, through `reference_conseal.py`.)

### The comparisons now

`tests/reference/reference_conseal.py` holds the removed code, unchanged: the
reference's kernel derivative, the densities' derivatives, the square root's
chain rule, the median, the cotangent layout and the smoothing. On a level-2
icosphere it reproduced the last commit's strict mode bit for bit, in all
twelve arrays compared. `conseal_compare.py` puts these back into the port's
output (job 4603697). Every stage agrees with exactly the numbers strict mode
gave earlier the same day:
- the geometry to 3e-16 to 5e-15;
- the kernel to 4.1e-8 and its derivatives to 1.3e-6;
- the densities and their derivatives to 1.1e-5 to 1.0e-4;
- the gradient and the step to 5.7e-7 on the left hemisphere and 1.2e-6 on
  the right;
- the flow of the step to 2.3e-9 and the endpoints it carries to 3.4e-5;
- the cost after one step, 0.102993879 against MATLAB's 0.102993876;
- the five-iteration template to 2.0e-5.

The six-iteration registration trace was not compared here: the code that
stepped it went with the mode. The reason given, that the trace also needs the
reference's descent (item 7's entries 3 and 4), was wrong, since in those six iterations
nothing folds and the cost never rises; item 23 steps the trace in the
reference's arithmetic, and it agrees as before.
`conseal_gradient_probe.py`, `template_probe.py` and `fourth_list_probe.py`
take the reference's side from the helper, and their outputs are what they
were:
- the gradient probe's table is identical to its run in strict mode earlier
  the same day (job 4587733);
- the template probe again finds the reference's median on subject C, 0.001
  degrees from it, and the default's 0.43 degrees from the mean and 16.6 to
  20.2 from every subject (item 9);
- the fourth list's smoothing differences are identical, line for line, to
  its run of 6 October (job 4019556).

That last run also showed the ConSEAL docstring misquoting one figure. After
100 smoothings the difference from the reference reaches 2.0e-2; the
docstring's 2.2e-2 was the cotangent layout's alone, and is corrected.

### Verified

On Longleaf (job 4603697) the whole suite passes on the released subjects,
1,072 tests with 14 skipped. The ConSEAL tests pass under Python 3.10 (74), and
Tier 4's 21 tests pass. The audit passes 22 of 22, the tour runs, and
`check_hcp_ya.py` passes on a fresh download of the eleven.

The outputs were recorded before (3806775, with its own recorder) and after,
one after the other on one Intel Xeon Gold 6140 node (`outputs-3806775.npz`
against `outputs-strict.npz`). Every one of the 29 arrays is identical, bit for
bit: no recorded run used the switch.

## 23. A recheck of everything, 11 October 2026 -- NO HIGH-SEVERITY DEFECT; CONSEAL'S STOPPING RULE AND ROLLBACK, TESTS THAT COULD NOT FAIL, AND THE RECORDS

Asked: "check again on everything, see any remaining issues are there and any u
can fix for me, be very careful, consistent and thoughtful".

Six reviewers read the repository at 3ef5d21, one area each: the package outside
the two alignments, ENCORE, ConSEAL, the deposit tool and its documents, the
documents a user reads, and these records. None found a high-severity defect.
ConSEAL's default path is bit for bit what it was at 3806775, and
`reference_conseal.py` and `reference_geometry.py` reproduce the switches items
21 and 22 removed. Every finding was checked before it was fixed. Each change
to the code comes with a test that fails without it: for ConSEAL each fix was
undone in a copy, and its test failed there and passed with the fix.

### ConSEAL

- **A zero starting cost removed the stopping rule.** The floor under the
  threshold is a share of the starting cost, so at zero cost the threshold
  was zero, and a step that changed nothing counted as progress.
  `endpoints_align(..., template=i)` gives subject `i` exactly that cost.
  With its endpoints on their triangles' corners, which the identity leaves
  where they are, a subject registered onto itself ran 40 of 40 iterations,
  each a full density and its derivatives. Now a moving density equal to the
  fixed one takes no step, and a step that leaves the cost where it was ends
  the descent, whatever the threshold.
- **Undoing the first step did not return to the start.** The rollback carried
  the endpoints through the previous warps, and before the first step those are
  the identity. The identity relocates each endpoint through its triangle's
  closest point, so it moves it. Forced on ico2, the registration returned the
  endpoints 0.048 degrees from where it found them on average and 0.075 at
  most, with 91 of 6,000 triangles changed. They are now reset to where they
  started, bit for bit.
- `tests/reference/conseal_recheck_probe.py` measures both cases, at 3ef5d21
  and now (job 4614533).
- **Refused now:**
  - barycentric weights that are not finite (`from_endpoints`): one NaN made
    the density NaN, and the template and the costs with it, silently;
  - a fixed density not on the moving connectome's grids: a vector broadcast
    against the matrix;
  - a starting cost that is not finite.
- The rigid search's branches for the removed mode went, as did the module's
  wording of two modes.

### ENCORE

- `align(template=...)` refuses an array with a nonzero diagonal, or whose
  squares do not carry unit mass. A density passed where its square root
  belongs had registered silently.
- `parallel_transport` moved to `encore_order_probe.py`. Since item 21 nothing
  in the package used it, and that probe is the one script that needs the
  reference's order.
- The docstrings now say:
  - what the MATLAB comparison still covers;
  - that `order` sets the bundled grids' basis only;
  - that a warp carrying a vertex to within 1e-4 radians of the axis reads its
    Jacobian too small: 0.9967 at 1e-4, 0.79 at 1e-5. The scheme works in
    `(theta, phi)` at the image too, which clearing the grid does not prevent.
    Rare at the sizes registration produces, it is left as documented.
- **New test:** two rotations composed in turn give their product to rounding,
  in the gradient's order.
- **The descent rates, re-measured.** `encore_order_probe.py rate` runs the
  configuration of `test_a_step_is_composed_as_the_gradient_assumes`, a
  29.7-degree turn (job 4614533):
  - with the gradient's order: 0.982 at the identity, 0.987 after the turn;
  - with the reference's order: 0.982 at the identity, 0.789 after the turn.

  Items 20 and 21 and VERIFICATION had said 0.80 and 0.98; item 21 also said
  0.975 to 0.997.

### The rest of the package

- **The cohort loader** admitted an FC file carrying streamline endpoints,
  which `sbci.load` refuses. It now leaves the file out with that reason.
- **`read_hdf5`** read the modality of a file whose metadata names none
  before validating it. The result was a `KeyError` in place of the
  metadata's own report of what is missing.
- **Metadata:**
  - a `hemisphere_order` of `"LR"`, which iterates to L then R, was accepted;
  - a number there raised a `TypeError` outside the package's errors, which
    crashed `sbci validate` and `sbci info`;
  - metadata that is not UTF-8 raised a `UnicodeDecodeError`, which crashed
    `sbci info`.

  All three are `MetadataError`s now.
- **`COVERAGE`'s docstring** now says what the code does: an fsLR vertex needs
  more than half its area on vertices with a value, so one at exactly half
  is missing too. The test of item 21's exchange-file rule now
  compares the file `save_map` writes vertex by vertex.
- **`smoothing.py`:** two docstrings still carried figures measured on
  subjects item 20 removed. They cite item 6's own now.
- **`project()`** cited item 21 for a figure that is in items 5 and 8.

### The deposit tool

`tools/dataverse_deposit.py` was rewritten against its reviewer's findings; its
docstring says what each step checks. In short:

- the released copies must match the size and SHA-256 recorded in the
  release's manifest;
- every file's metadata values are checked, not only its keys;
- the records are written whole, so a failed run leaves none;
- a document edited since `prepare` wrote it is kept unless
  `--rewrite-documents` is given;
- the upload compares whatever checksum the installation computes;
- when a request's outcome is uncertain, the upload checks the draft before
  sending it again;
- it refuses a redirect, plain http, and a token not in Dataverse's form;
- `verify` reports a file that looks like a second copy.

Its tests went from 8 to 36. The last of them came from preparing the deposit
again with the rewritten tool (job 4614544):

- **The bug.** The comparison of a released copy with the build directory's
  walked each file's datasets with `visititems`, which stops at the first
  value a visit returns, and the visit returned what it stored. So only a
  built file's first dataset, `area`, was compared.
- **What it hid.** The ten FC copies that differ from their rebuilds, in 1 to
  7 connectivity entries by at most 1.9e-9 (as b9fb510's prepare reported
  them), read as "the same arrays".
- **The fix.** The visit returns nothing now. The tests' stand-in files have
  `area` first, as built files do, and a new test compares a difference
  nested under `endpoints/`. With the old walk, both fail.

### The comparisons with MATLAB

- **`reference_geometry.py` beside the axis.**
  - **What it got wrong.** The reference divides the phi-component by
    `max(sin(theta), 1e-5)`, so within 1e-5 of the axis it scales it by
    `sin(theta) / 1e-5`. The helper zeroed it there instead. That matched on
    the axis, but beside it the helper was off by 1% to 96% of the largest
    entry.
  - **Who was affected.** No comparison was: MATLAB's grids put vertices
    exactly on the poles.
  - **The fix.** The helper now scales. It takes `sin(theta)` through
    `cos(theta)`, as the reference's Legendre functions carry it, which is
    exactly zero at both poles where `np.sin(pi)` is 1.2e-16.
  - **Measured** against b25f6f2's `reference=True` by
    `reference_geometry_probe.py` (job 4614533), as the basis's largest
    difference relative to its largest entry (the divergence agrees as
    closely or more). What is
    left beside the axis is the port's own limit, which it takes within 1e-5
    of the axis where the reference evaluates the field.

| Poles off the axis (rad) | Order 3 | Order 6 | Order 15 |
| --- | --- | --- | --- |
| 0 | 6.3e-16 | 6.1e-16 | 4.8e-16 |
| 1e-7 | 5.2e-16 | 1.1e-15 | 7.1e-15 |
| 1e-6 | 2.7e-13 | 1.1e-12 | 7.1e-12 |
| 5e-6 | 3.3e-11 | 1.3e-10 | 7.9e-10 |
| 9e-6 | 1.9e-10 | 6.9e-10 | 3.7e-9 |
| 2e-5 | 4.1e-16 | 3.8e-16 | 3.7e-16 |

- **ConSEAL's registration trace is compared again.** `conseal_compare.py`
  steps the six iterations of `conseal_reference.m`'s registration in the
  reference's arithmetic, each step smoothed into the field as the reference
  does it. Nothing folds and the cost falls at every step, so the reference's
  handling of a refused step and of a rise (item 7's 3 and 4) never acts; the
  loop would say if it did. Item 22 had given that as the reason the trace
  could not be compared.
  - Every iteration agrees within 3.5e-7 relative (jobs 4613665 and 4614533;
    the last iteration's 1.5e-9 and 1.3e-9 differ between the two runs by
    the order of the threads' sums).
  - Every other stage prints what item 22's run printed.
- **`test_conseal.py` measures the reference beside the port again,** through
  `reference_conseal.py`:
  - its kernel derivative is taken in the row vertex, and its densities'
    derivative differs from the true one (item 7's 1 and 2);
  - its cotangent layout weights the wrong angle (12);
  - its median stops on its starting subject (15);
  - its smoothing moves the polar vertices of a 20-degree rotation 8.4 and 8.5
    degrees, and their neighbours up to 11.5 (16). Item 9 had 8.8 and 11.3,
    measured with the port's own component-wise smoothing of the time;
  - it keeps 98.9% of a rotation about z and 90.5% of one about a tilted axis
    (16);
  - its rigid search, with no turns about z, leaves an 8-degree turn 9.6 and
    8.2 degrees off on the test's ico2 grid (9.4 and 8.0 on ico3, item 18).
- **Tests that could not fail now can:**
  - cross-validation is compared with a brute-force leave-one-out, and agrees
    to 0.0 on 60 streamlines;
  - the chain rule is tested where the density goes negative (bandwidth 0.005,
    degree 30: two entries). The fixture before had none;
  - the coincidence guard is tested on a cohort whose normalized starting
    subject sits 2.1e-8 radians from itself. Without the guard the median
    lands on it;
  - the rollback and the both-refused break are reached, and checked by what
    they leave behind.
- **Tier 4 holds `to_atlas` to 1e-12, as VERIFICATION documents.** It held it
  to 10 float32-eps; it measures 1.9e-16.

### The records

- **Counts and figures:**
  - the skips: 38, of which 21 are MATLAB comparisons;
  - ten ENCORE steps in the recovery figure, not eleven;
  - which 226 of the 407 vertices beside the wall read 0;
  - the seed rows agree to float32 rounding, not float64;
  - ENCORE's agreement with MATLAB, stage by stage;
  - ConSEAL's iteration counts, 76 to 155 and 171;
  - the bundled surfaces: 368 kB, and the sphere converted from the toolkit's
    export, not built from fsaverage;
  - the exchange file is 340 times an SC file;
  - the basis would add 100 MB to the wheel, not 52;
  - the coupling's reject line in VERIFICATION is 1e-12.
- **Item 7:**
  - its heading;
  - its comparison table, rebuilt from job 4613665's output with the labels
    `conseal_compare.py` prints;
  - the kernel's nonzeros at the default bandwidth: 32 a row, cut at 12.4
    degrees;
  - its *Still open*;
  - one list of what `strict_upstream` reproduced, the same in items 7, 21
    and 22.
- **Status:**
  - item 4's *Still open* is closed;
  - item 2's question for the group joins the Status;
  - item 20's decided bullets are marked;
  - BLUEPRINT names the two decisions item 20 leaves.
- **Traced:**
  - the 21 to 25 degrees quoted for ConSEAL's median come from the cohort
    figure's runs (jobs 4570438 and 4588257, 21.03 to 25.06), now cited;
  - job 4598330 ran `fork_package_case.py` with a `strict` key that was never
    committed, and item 22 now says so;
  - the probes behind item 4's step-size sweeps and item 5's roughness are
    named (not every figure of those items had one; item 24).
- **Cross-references corrected:** item 9 for the template probe, items 8, 9,
  17 and 21 for VERIFICATION's findings, entry 14 of item 7.
- **Wording:**
  - "corrected by default" is now "corrected";
  - "`backtracks=0` reproduces the reference" now says what it does: it stops
    at the first failed step;
  - VERIFICATION dates its ADNI block, recorded before items 8, 9 and 21, and
    gives today's figure on the young adults;
  - SPEC_QUESTIONS says the 935 are not distributed yet, as the release
    plans.

### Verified

On Longleaf (job 4614533):

- the whole suite passes on the released subjects, 1,112 tests with 14
  skipped;
- Tier 4 passes 21 of 21, `to_atlas` now held to 1e-12;
- `conseal_compare.py` agrees as above, its trace included;
- the gradient, template and fourth-list probes print what item 22's run
  printed;
- the audit passes 22 of 22, the tour runs, and `check_hcp_ya.py` passes on a
  fresh download of the eleven;
- under Python 3.10 the modules this item touched pass, 356 tests.

The deposit tool's last fix came after that run; its 36 tests pass here, and
the deposit is prepared again with it once this item is committed, so that its
README names the commit.

The outputs were recorded before (3ef5d21) and after, one after the other on
one AMD EPYC 7702 node (`outputs-3ef5d21.npz` against `outputs-recheck.npz`,
job 4614534). Every one of the 29 arrays is identical, bit for bit: no recorded
run starts from a zero cost or undoes its first step.

## 24. A third review, 12 October 2026 -- NO HIGH-SEVERITY DEFECT; MALFORMED FILES, UNCHECKED SETTINGS, UNTESTED BRANCHES, AND FIGURES NOTHING RECORDED

Asked, once item 23 was pushed: "run another check over the whole repo and make revisions
over the issues u find".

Six reviewers read beb58e8, one area each, as before: the package outside the two
alignments, ENCORE, ConSEAL, the deposit tool with the scripts and tools, the documents a
user reads, and these records. None found a high-severity defect. Every finding was checked
before it was fixed, and every fix was then undone in a copy of the tree, one at a time,
with the mutations each reviewer showed the suite could not catch: the tests fail on all
105 (the package 22, ConSEAL 30, ENCORE 20, the deposit tool 33), and pass on the tree.
No recorded output moves (*Verified*).

### The package

- **Malformed files still escaped the package's errors.** Arrays were compared by
  size, not shape, so a connectivity stored as a row, `(1, N_UPPER)`, passed all
  eleven checks of `sbci validate` while `seed()` raised an `IndexError`, and one
  stored as a column, or an area or mask stored 2-D, crashed `validate` with a
  traceback, and `info` with most of them. An endpoint dataset stored as a group,
  a scalar `/endpoints` and a damaged compressed chunk raised `AttributeError`,
  `TypeError` and `OSError`. Now the shapes are compared, and `validate` computes nothing
  further once they fail; the endpoint group's members are checked to be
  datasets; and an error reading the file is an `InvalidFileError`.
- **The cohort loader** still admitted what `sbci.load` refuses: an FC file with
  an empty endpoint group, and connectivity held as text. `read_header` now says
  whether the group is there and whether each dataset is flat and numeric, and
  the loader leaves out what `load` would refuse.
- **Both endpoint readers cast before they checked.** `Endpoints.from_global` read
  a vertex stored as 2965.6 as 2965, and `validate` passed the file;
  `from_matlab` cast a hemisphere of 1.9 to 1, 256 to 0 and a vertex of 2.5 to 1.
  Indices are now checked to be whole before the cast, and the hemisphere flags
  before theirs.
- **`sbci cohort --report X`** crashed with a traceback when the cohort was
  refused and X could not be written, and checked X only after the whole scan.
  It is checked first now, and the write after a refusal is guarded.
- **`local_coupling`** gave 0.0 where a vertex's label is NaN; it gives NaN.
- **`Atlas`** took fractional labels: with 1.5 and 1.2, `to_atlas` counted both
  vertices in region 1, and `region_mask(1)` neither of them. Labels must be whole
  and finite, and are stored as integers; text is refused.
- **`design(categorical=...)`** merged integer codes past 2**53, as the cohort
  loader did until item 20: they went through float.
- **Docstrings:** the bundled surfaces are 368 kB, the sphere converted by
  `convert_surfaces.py`; an exchange file is about 160 times the dense matrix and
  340 times the tutorial subject's SC file; the kernel's last term at 0.005 carries 0.28% of
  the 270.267 peak.
- **Tests that could not fail now can:** the Freedman-Lane permutation is pinned by
  replaying its seeded draws (the scores permuted whole passed); a download that
  fails its digest leaves no `.part` behind, and a present file with the wrong
  digest is fetched again; `region_means(mask=)`; `validate`'s thresholds (the
  unit mass, the area units, the sign, the spec version's minor number); NaN
  standard errors where a contrast is not estimable; identification's within and
  between; region names without regard to case, and the atlas spellings `_key`
  promises.

### ConSEAL

- **A density passed where its square root belongs registered silently.** On ico2
  its squares sum to 2.4e-4; sixty steps moved the cost from 0.9727 to 0.9723 and
  left the endpoints 4.15 degrees from the truth, against 1.15 given the square
  root. `register` now refuses a fixed array whose squares do not sum to 1 within
  1e-6 (a subject's sum to 1 within about 1e-10, a template's exactly), and, as
  `align` does since this item, one that is negative or held in one triangle.
  `endpoints_align(template=True)`, which registered everyone onto subject 1, is
  refused too.
- **Item 23's rollback was tested only at the first step, with no turn.** Six
  mutants passed the suite: a rollback that always resets, one that never
  re-carries, `relocated` never set, `_rigid` reporting a kept turn as refused, a
  refused turn not reset, and a turn kept unless ten times worse. New tests take the descent through a
  rollback after three kept steps, after a kept turn and after a refused one, and
  check that what `register` returns agrees with itself: the connectome is the
  moving one carried through the warps returned, the last cost is that
  connectome's, and the callback heard of every step kept and nothing else.
- **Other branches with no test:** a step that changes nothing ending the descent
  at threshold 0; the last step kept below the threshold reaching the callback;
  the smaller of the threshold and the floor; the steps refused on the right
  hemisphere counted; the clamp's value; a verbose run; `to_endpoints` naming the
  nearest vertex; the Jacobian's orientation; the sparse route of `evaluate`,
  which every subject up to about 15,000 streamlines takes, against the dense
  one; and the area-weighted gradient, on a grid pulled toward a pole so that its
  areas vary eightfold, since on ico2 they vary by a tenth and dropping either
  weighting still gave a cosine of 0.98 (0.92 and 0.94 there, 0.976 with both).
  The README example's test asserts a strict fall: a rollback guaranteed the old
  one whatever the step's sign.
- **Settings that cannot register are refused:** a step or clamp of 0 or less, a
  negative or NaN threshold, a negative or fractional iteration count, a negative
  viscosity, NaN anywhere; each returned identity warps or stopped after a step,
  as though it had converged. `cross_validate` refuses a connectome with no
  streamlines (it returned the first bandwidth with NaN scores), and
  `endpoints_align` a negative `template_iterations`.
- **Hemisphere flags are checked before the cast,** as `Endpoints` checks them:
  `from_points` took 256 and 0.5 for the left, 1.9 and 257 for the right.
- **Tests item 23 left stale:** registered onto itself a connectome takes no step
  now, so the test of a warped connectome registered onward compared two rounding
  floors; it now steps it onto another subject and checks the endpoints went on
  from where they were. A comment described a rollback that no longer happens, and
  a docstring blamed F's two negative entries on the kernel, which has none inside
  its cutoff: they come from the adjacency, whose barycentric weights reach
  -0.0075 for an endpoint just outside its triangle.
- **`tests/reference/rotation_probe.py`** crashed since item 23 changed `_rigid`'s
  return; it runs again and reproduces item 9's figures (job 4628494).

### ENCORE

- **Nothing tested the gradient `register` steps along.** The rate test rebuilt it
  in its own body, and four broken gradients passed all 96 tests. The descent is
  `Encore._descent` now, which the rate test calls, and a new test takes the step
  `register` itself takes and sets it against the cost's derivatives along every
  basis field: cosine 0.997, against 0.91, 0.23, 0.82 and 0.70 for the four.
- **A template array** that is negative or held in one triangle was accepted, as
  a connectome passed there was not; it is refused now, as densities are.
- **Nothing local tested the median.** Weiszfeld's weights counted, a damping of
  0.6, or the mean's step all passed; two tests pin it: from one of two subjects
  the template takes a fifth of the way to the other, where their pulls cancel,
  and with four subjects alike and one apart it ends a hundredth of the way from
  the four to the fifth, where the mean would sit about a fifth of the way.
- **A grid turned by hand that still needs a turn** maps back to its file's frame
  through the rotations recorded; with them composed the other way it lands 9
  degrees off, which no test saw.
- **A grid no turn clears of the axis** raised `pole_rotation`'s `RuntimeError`,
  where the documents promise a refusal; `align`'s own check after the turn could
  never fire. It is a `ValueError` naming the grid, and the dead check went.
- **Settings that cannot register are refused,** as for ConSEAL: a step or
  difference of 0, a negative or fractional count of iterations, backtracks or
  template iterations, a negative threshold (which accepted every step), NaN, and
  a basis of order 0, which holds no field at all.
- **The first checks of the port are measured again.** Item 4's figures that no
  test asserts were measured on 20 September by scripts never committed, some of
  them against a standalone draft. `tests/reference/encore_first_checks_probe.py`
  measures them with the package (jobs 4628016 and 4628494), and the two MATLAB
  scripts that wrote the dumps it needs are committed (`encore_face_agreement.m`,
  `encore_jacobian_inner.m`). Most reproduce to the digit. Three did not, and
  item 4 now says what the probe prints: the derivative sits 0.0367 from the
  reference at every step to 1e-6 and 0.0359 at 1e-3; the Jacobian arithmetic
  on the reference's own points agrees to 3.8e-6 and 4.4e-6 (an Intel and an
  AMD node), not 2.2e-6; and choosing a face by the distance to its plane
  agrees on none of the queries, the 13.6% first quoted having come from a sum
  of absolute coordinate products. The identity's Jacobian is 1 because the
  reference sets it so and the port's calibration gives it; its scheme reads
  0.935 there. The module docstring no longer says every layer is checked
  against MATLAB: the derivative is the probe's.
- `tests/reference/encore_axis_probe.py` measures the Jacobian beside the axis that
  the docstrings and tests quote. 1e-5 rad off it, the identity's reads 0.74 and
  0.76 on the level-2 icosphere and 0.78 on the bundled sphere, and a 0.5-degree
  turn's, divided by it, 1.27 and 1.25, and 1.27; 3e-6 rad off, 34 and 37, and 38.
  A turn carrying a vertex of the bundled sphere 1e-4, 1e-5 and 0 rad from the
  axis reads 0.9967, 0.786 and 0 there (item 23 quoted these with no probe).

### The deposit tool

The tool's reviewer found one serious gap and thirteen smaller ones; every one held.

- **A deposit with no record of what prepare wrote kept its old documents for good.**
  Without `.documents.json` -- a deposit prepared before item 23, as the one on
  Longleaf was -- every document counted as edited: a plain rerun kept b9fb510's
  README and called it "kept as edited". The deposit on Longleaf was not left so:
  item 23's rerun took `--rewrite-documents`. Now prepare refuses such a deposit
  until `--rewrite-documents` is given.
- **What upload takes is checked again.** It refuses a deposit whose prepare stopped
  before `CHECKS.txt`, a `subjects.csv` with a column or a value beyond the
  open-access three, and a `SHA256SUMS` that is not what `MANIFEST.tsv` says.
  Before, a column added to `subjects.csv` after prepare went up, and verify passed
  it.
- **prepare removes only what it wrote.** A `--subjects` table inside `--out`, and
  records in a folder no prepare wrote, are refused; before, both were deleted
  unread.
- **Restriction** is checked on resuming (a trial sent open and the rest restricted
  left a mixed draft) and by verify, which takes `--restrict` too.
- **Smaller ones:** a published dataset with no draft counts its published files as
  held (a 404 for the draft read as an empty dataset, so every file would have gone
  again); a retry that finds another file in the draft stops rather than sending it
  again; a second copy of a file without an extension is caught; a lock file keeps
  two uploads of one deposit from running at once; a server echoing the key gets it
  printed as `[token]`; `--limit` takes only a number of subjects; an FC file whose
  hemispheres took different routes is accepted and described.
- **Tests that could not fail.** Fourteen of sixteen one-line deletions of the tool's
  refusals passed its 36 tests. Its tests are 62 now, and each of 33 such deletions
  -- those fourteen, item 23's two, and one for every fix above -- fails them.

Elsewhere in tools/ and scripts/: `write_exchange_file.py` now fails when one of its
three claims does (it printed them and exited 0 whatever they were); `audit_api.py`
checks both hemispheres' Jacobians where its message said so of both and it checked
one; `import_legacy.py` gets `--endpoints`, which Tier 4's `smooth()` comparison needs
and which nothing in the repository wrote (the lab's copy was made by hand on 20
September), and VERIFICATION gives the command; two descriptions are corrected (the
importer's inputs, the figures' extra).

### The documents

- VERIFICATION still said ConSEAL's registration is not compared, beside a table
  quoting its trace; it says what item 23 made true, and names the cotangent layout
  among what goes back in.
- An outside verifier following Tier 4 got 20 passed and 1 skipped, not 21: the
  `smooth()` comparison needs an SC file carrying its endpoints, which nothing
  documented built. The import above builds it, and Tier 4 passes 21 of 21 on it
  (jobs 4627321 and 4628494).
- **The smoothing figures** the README, USAGE, VERIFICATION and SPEC_QUESTIONS quote
  (3.25 float32-eps, r = 0.999999999999921, 5e-08) traced to no record, and item 1's
  own (3.18 and 3.17) are not what today's run gives. `tests/reference/
  smooth_matlab_probe.py` prints them (jobs 4627321 and 4628494): the kernel 2.01
  float32-eps, the density 5.15, the `smooth()` method 3.14 with r =
  0.99999999999994, its diagonal zero and its mass 1.000000000014, and the
  bandwidth it takes unasked 5.0e-08 from the reference's. Every document quotes
  those now, and Tier 4 still holds each within 10 float32-eps.
- **The reliability notebook's section 5** was computed on 7 October, before item 21
  changed ENCORE's order of composition. Aligned again from the saved halves
  (`tests/reference/reliability_alignment_probe.py`, job 4620898), a subject's two
  halves sit as before, 0.109 degrees apart (0.087 to 0.124), and two subjects 3.083
  rather than 3.070 (2.434 to 3.949 rather than 2.440 to 3.851); one vertex's
  displacement changes by 4.2 degrees at most, where the warp between two subjects
  reaches 16 to 19. The new alignments replace the split folders' (the 7 October
  ones kept beside them), section 5 is run again on them (job 4627566), and the
  notebook and USAGE say 3.08.
- The recovery figure's warp moves the grid's vertices 1.59 degrees on average and
  3.98 at most (1.67 with seed 8, which the figure does not use): "1.6", where three
  documents said 1.7.
- The reference's rigid search left an 8-degree turn 9.4 and 8.0 degrees off on ico3
  (item 18), where USAGE said "8 to 9"; the projection's figures are items 5 and 8,
  not 21; ConSEAL's iteration time is item 19's 12 seconds, measured at the public
  code's bandwidth, where USAGE said 10 to 22.

### The records

- Item 20's bullet recording a dataset's removal still named it; the bullet went.
- **The review note:** finding 3's *Still open* no longer held (onto itself a
  connectome takes no step since item 23, and its test checks it to the bit), so it
  went; its opening, its table and its account of the paper's ConSEAL say which
  switches were removed when; and the registration time is item 19's, 76 to 155
  iterations and 15 to 42 minutes, where it said 15 to 45.
- **Switches named as if they were still there:** items 8, 18 (row 5) and 19 say
  "until item 22" where they describe `strict_upstream` or `reference=True`.
- **Cross-references:** item 21's "item 14" and item 22's "items 3 and 4" are item 7's
  entries; item 11's "item 16" is the ConSEAL docstring's.
- **The Status and BLUEPRINT** list item 5's request that whoever specified the API
  review `local_test`'s choice of test, and item 2's second question, which inner
  product the published figures used.
- **Figures:** VERIFICATION and BLUEPRINT gave ConSEAL's gradient agreement as 6e-7,
  where the last run gave 5.7e-7 on the left and 1.2e-6 on the right; item 7's
  gradient table carried ico2 and ico3 rows printed before item 10's basis at the
  poles, and carries the probe's output now (its ico4 row and its table over all
  510 coefficients reproduce to the digits quoted, job 4628490);
  item 7's known warp moves the grid's vertices 1.0 degrees on average and 3.6 at
  most, and the endpoints 0.98 and 2.9, which a reviewer re-measuring the endpoints
  had taken for an error.
- **Item 23:** the old geometry helper was off by up to 96% beside the axis, not 90%
  (order 15, 9e-6 off it); the rigid search's 9.6 and 8.2 degrees are on the test's
  ico2 grid and item 18's 9.4 and 8.0 on ico3; and "the probes behind items 4 and 5
  are named" said more than it did. The 91 of 6,000 triangles it quotes are
  Longleaf's (job 4614533); a Mac prints 89, ties between closest triangles rounding
  the other way.
- **Still traced to no committed probe:** item 4's 0.00935 for the reference's single
  step; item 5's 40 s against 66 s and 2.1 GB against 4.2 GB; and item 18's
  tilted-axis figures (3.3 and 3.0 left, 0.33 and 0.20 after) and its 14 s against 9.

### What was measured and left

- **ENCORE's template after its default ten steps** is short of the median. On the
  eleven released subjects (`tests/reference/encore_template_probe.py`, job 4628491)
  the sum of the geodesic distances from the template to them is 4.3948 after ten
  steps, 4.3380 after twenty and 4.3367 after fifty, where the 0.005 rule had stopped
  it: ten steps leave the sum 1.3% above that, and the template 0.067 rad from the
  median, a sixth of the 0.39 rad a subject lies from it on average (twenty steps,
  0.007). The reference takes ten too, and `align` keeps its default, which its
  docstring now documents with these figures; raising it would move every
  template and warp made at the defaults, so it is a decision (Status). Decided
  the same day: up to a hundred, item 25.
- Each hemisphere's step is normalized on its own, as the reference does, so the two
  together are not quite the steepest descent when their gradients differ in size;
  `register` says so where it does it.

### Verified

On Longleaf, from a frozen copy of the tree (job 4628494, an AMD EPYC 9654 node):

- the whole suite passes on the released subjects, 1,222 tests with 14 skipped;
- Tier 4 passes 21 of 21 on the lab's derivatives, and 21 of 21 on a fresh
  `import_legacy.py --endpoints` (job 4627321 had too);
- `conseal_compare.py` agrees as item 7's table (the trace within 3.5e-7, the
  template to 2.0e-5), and `encore_order_probe.py rate` prints item 23's rates,
  0.982 and 0.987 against 0.982 and 0.789, after the descent moved into
  `_descent`;
- the probes print what this item quotes: the smoothing, ENCORE's first checks and
  its axis, ConSEAL's gradient rows, the rotation probe's item 9 figures, the
  recheck probe's 0 iterations and 0 of 6,000 triangles, and the template,
  fourth-list and reliability probes as before;
- the audit passes 22 of 22, the tour runs, and `check_hcp_ya.py` passes on a
  fresh download of the eleven;
- `write_exchange_file.py` holds its three claims and exits 0 (the mass changes
  by 1.1e-10 on fsLR, and Desikan on fsLR against ico4 correlates at r =
  0.99964);
- under Python 3.10 the modules this item touched pass, 843 tests with 1
  skipped.

The ico4 gradient table over all 510 coefficients (job 4628490) and the template
probe (job 4628491) are above. The outputs were recorded before (beb58e8) and
after, one after the other on one Intel Xeon Gold 6140 node (`outputs-beb58e8.npz`
against `outputs-review3.npz`, job 4628489). Every one of the 29 arrays is
identical, bit for bit: what this item changed in the code refuses inputs that
no recorded run gives, or moves ENCORE's descent into a method without changing
its arithmetic.

The deposit tool's 62 tests pass in the suite above, and the deposit on Longleaf
passes the new checks of `upload --dry-run` (1,897 files, 85.06 GB). It is
prepared again once this item is committed, from that checkout, so that its README
names the commit.

The tree verified differs from the commit in comments and documents only: this
item, the figures it puts in other documents, and a few comments and docstrings
that quote what the jobs printed.

## 25. ENCORE's template takes up to a hundred steps, 12 October 2026 -- THE MEDIAN ITS RULE REACHES, NOT TEN STEPS TOWARD IT

Decided by the user once item 24 reported the template short of the median: "please raise
the ENCORE template steps default, but test it thoroughly before make the change in the
repo".

**The change.** `align(template_iterations=...)` and `Encore.template(iterations=...)` take
up to 100 Weiszfeld steps by default, where they took the reference's ten. The rule that
ends the steps is the reference's and is unchanged: a step whose mean direction to the
subjects is shorter than 0.005 is the last. ConSEAL's template has taken up to 100 under
the same rule all along. Nothing else in the arithmetic changed.

### Measured before the change

`tests/reference/encore_template_probe.py` takes the steps one at a time as
`Encore.template` takes them -- the same arithmetic, checked against it bit for bit at ten
steps and at the rule on the eleven -- and past the rule to the 80th, the median for every
purpose here.

**Where the steps go** (`convergence`; the eleven, job 4648690, and twenty other young
adults of the 946 drawn with a fixed seed, job 4648691):

| | the eleven | twenty others |
| --- | --- | --- |
| the subjects from the median, rad | 0.367 to 0.436 (0.394 on average) | 0.366 to 0.451 (0.398) |
| the mean direction after 10, 20, 30 steps | 0.069, 0.012, 0.002 | 0.062, 0.009, 0.001 |
| the rule stops after | 25 steps | 23 steps |
| the template after 10 steps, from the median | 0.071 rad | 0.058 rad |
| where the rule stops, from the median | 0.0047 rad | 0.0046 rad |
| the sum of the distances after 10 steps, where the rule stops, at the median | 4.3948, 4.3367, 4.3364 | 8.0335, 7.9585, 7.9580 |
| a step, on eight cores | 2.2 s (an AMD EPYC 9654 node) | 15.7 s (an EPYC 7763) |

Ten steps leave the template 15% to 18% of a subject's distance short of the median; the
rule stops it 1.2% short, on both cohorts, and well inside a hundred steps.

**Where they start** (`starts`, the eleven, job 4648690). From four subjects -- the one
nearest the mean, as the template starts, the fourth and eighth nearest, and the farthest
-- the rule stops after 25 or 26 steps. After ten steps the four templates lie up to 0.117
rad apart, and each is nearer the subject it started from (0.30 to 0.36 rad) than the
others (0.40 on median): the reference's template leans toward its first subject, where
the median is meant to make no subject the reference. Where the rule stops they lie within
0.0075 rad of one another; after sixty steps within 1.3e-5, one median.

### What it changes

**The eleven aligned as the cohort figure aligns them** (`cohort`, job 4648690: re-smoothed,
ten iterations, four backtracks). With the template where the rule stops rather than after
ten steps, the subjects start nearer it on the whole, 1.692 in all against 1.746, and end
nearer, 1.416 against 1.463, each at 0.793 to 0.870 of its start against 0.796 to 0.880. The
lowest starting cost had been the first subject's, 0.087, 0.296 rad off (above); now it is
0.130, and the costs span 0.130 to 0.187 rather than 0.087 to 0.195. The mean correlation
between two subjects' aligned connectomes is 0.7615 against 0.7608. The two templates lie
0.067 rad apart, and the alignment takes 349 s against 328.

**What else moves.** Every `align` that estimates its template from three subjects or more.
Two subjects meet after two steps whatever the cap, a fifth of the way from the first and
then pulled both ways equally, so the alignments of two in the audit, in `check_hcp_ya.py`
and in the recorded outputs do not move (*Verified*). The cohort figure registers the
eleven, and is drawn again (job 4649389): after ENCORE the mean correlation between two
subjects is 0.7615, where it was 0.7608, and the rest is as it was, since ConSEAL's
template already ran until its steps were short: 0.7900 after ConSEAL, onto a median 21.03
to 25.06 degrees from the subjects. The template takes about two and a half times as long,
25 steps where it took ten: under a minute on the eleven, whose registrations take five
times that.

### Tests

`test_the_default_template_runs_until_its_steps_are_short`: on six synthetic subjects whose
median the rule reaches after 36 steps, the default template, `align`'s and
`Encore.template`'s, is the one the rule stops, bit for bit, where ten steps leave it 0.16
rad away. With either default back at ten it fails, and the tests that pass the step count
explicitly are unchanged.

### Verified

On Longleaf, from a frozen copy of the tree with the change (job 4650784, an AMD EPYC 7742
node): the whole suite passes on the released subjects, 1,223 tests with 14 skipped; Tier 4
passes 21 of 21; the audit passes 22 of 22, the tour runs and `check_hcp_ya.py` passes on a
fresh download of the eleven; under Python 3.10 the alignment tests pass, 151 with 1
skipped. Recorded before (181a7b0) and after, one after the other on one AMD EPYC 9654 node
(`outputs-181a7b0.npz` against `outputs-template100.npz`, job 4650785), all 29 arrays are
identical, bit for bit: ENCORE's recorded alignment is of two subjects. The cohort figure
was drawn from the same tree on an AMD EPYC 7713 node (job 4649389), ENCORE in 809 s and
ConSEAL in 7,894 s, and looked at before it replaced the old one.

## 26. The alignments on a GPU, 12 October 2026 -- `device="gpu"`: EACH STEP THE CPU'S TO ROUNDING, A REGISTRATION WITHIN THE CPU'S OWN SPREAD, THE CPU UNCHANGED

Asked once the package was found to run on the CPU alone: "can u also build the gpu
support for this repo?", both alignments, ConSEAL's relocation of the endpoints included.

**The change.** `align(..., device="gpu")` and `endpoints_align(..., device="gpu")`, and
the same argument on `Encore`, `Concon` and `ConSEAL`, compute on a CUDA GPU through CuPy,
an optional dependency (`pip install 'sbci[gpu]'`, CuPy for CUDA 12; 14.2 tested). The CPU
stays the default. A new private module, `sbci._device`, holds what the two share: the
check that refuses any device but `"cpu"` and `"gpu"`, and refuses `"gpu"` before any work
when CuPy is missing (with the line that installs it) or sees no GPU; the array and sparse
modules a device uses; the moves between them; and the two GPU operations below.

| | on the GPU | on the CPU |
| --- | --- | --- |
| ENCORE | the densities: their transport through a warp (`Concon`, its fixed interpolation operators sent once), the cost, the template's steps, the descent's sums over pairs | the warps (composition, Jacobian, folds), and the descent's three vectors against the basis |
| ConSEAL | the endpoints, their density `F = K^T A K` with its derivatives (the kernel sent once a registration), their relocation (`MeshQuery` on the GPU), the cost, the template's steps, the gradient's sums over pairs | the warps, the gradient against the basis, and the rotation search of `init_rotation` (on the GPU since item 28) |

What comes back is what the CPU gives: `align` and `endpoints_align` return NumPy arrays,
floats and connectomes on the CPU, and so do `Encore`'s and `ConSEAL`'s methods, which
take NumPy or CuPy arrays; `Concon`, the transport the engines use, works in the GPU's own
arrays. Each algorithm is written once, in its device's array module, so on the CPU it is
NumPy doing the same operations in the same order as before: the two templates now add
their subjects into a running mean one at a time, as NumPy's `mean` and `sum` did, and take
their cosines and sines as Python floats, the same numbers as NumPy's scalars. Seventy-two
arrays recorded on synthetic subjects by the code before and after -- ENCORE's template at
ten steps and at the rule and ConSEAL's at the rule (this said both at both until item 28),
roots, `register`, `align` with an estimated and a given template,
ConSEAL's cost, plain, area-weighted and with the rotation search, and `endpoints_align`
with each kind of template -- are identical, bit for bit, and so is the recording on the
released subjects (*Verified*). The recorder was committed by item 27 as
`tests/reference/cpu_parity_probe.py`, with arrays on the bundled ico4 grids added; run
again there on one node, all 95 are identical across 03f4668, 81f6c99 and item 27's code
(job 4707454). ENCORE's template no longer stacks the subjects' roots into one
array from a list of them, which held them twice while it was made.

**The point search on the GPU.** The CPU narrows each point to the twelve faces whose
centroids a k-d tree finds nearest. CuPy has no k-d tree, and its `argpartition` sorts
every row in full, so the GPU narrows each point to the 36 faces nearest the centroid
nearest it, a table made once a mesh: on the bundled ico4 hemispheres and on icospheres of
1 to 3 subdivisions, the first 25 such faces held the tree's twelve for every one of
400,000 random points and every vertex (`tests/reference/gpu_candidates_probe.py`, job
4707454; this said 28 until item 27). The closest face among them and the lowest index on a
tie are the CPU's, and so, since item 27, is the rule that keeps it: no face outside the
candidates can be as near. The two bound the faces outside in their own ways -- the CPU by
its farthest candidate's distance, the GPU by its table's -- and send any other point on
in their own ways, the CPU to the faces near enough, the GPU to every face; either way the
face is the closest (until item 28 this said the two sent a point on by one rule). So
are the sums: NumPy adds three products as `(x0 y0 + x1 y1) + x2 y2`, and so does its
`einsum` on x86 for a point's three corners (on arm64 it fuses them, so since item 27 the
CPU adds the corners in that order itself); CuPy's reductions add in another order, so on
the GPU they are added in NumPy's. The GPU then finds the CPU's faces and weights bit for bit for the same points,
and relocates a subject's endpoints as the CPU does, bit for bit, through the same warps.

**Run twice, the same bits.** Two of CuPy's operations add in whatever order its threads
land: building a sparse matrix whose entries share a place, which it does with atomic
additions, and cuSPARSE's product of a sparse and a dense matrix as CuPy calls it, whose
partial sums meet in an order that changes from run to run: twice the same product of a
released subject's kernel, transposed, and a dense 5,124 x 5,124 matrix of numbers below 1
differed in their last bits, by 1.1e-16 where its entries reach 0.87 (`gpu_probe.py
repeat`, job 4674506). Together they made the same registration end differently twice --
two runs of ten ConSEAL iterations on a V100 ended 4.2e-9 and 4.0e-9 of their costs from
the CPU's (job 4666744) -- since the descent is as sensitive as the next paragraph says. So
the endpoints' adjacency `A + A^T` is built at once by sorting its entries stably by place
and adding each place's in the order given (`_device.summed_csr`), and every
sparse-times-dense product on the GPU goes through `_device.rows_times_dense`, which adds
each entry's products along its row one after another, the transposes it needs stored as
matrices of their own. It is also faster than cuSPARSE's on that product, 4.0 ms against
7.6 ms on a V100 and 18.1 ms against 48.3 ms on a GTX 1080, and agrees with it to 8.9e-16.
Run twice, a subject's density with its derivatives, three steps of ENCORE and three
iterations of ConSEAL on two released subjects give the same bits.

**How close a registration comes.** Every step agrees with the CPU to rounding: the
transport to 1e-11 of its largest entry and its central-difference derivative, which
divides that by its step of 2e-5, to 1e-9; the densities and their derivatives, the roots
and the templates to 1e-10 and closer. A whole registration cannot be held to that. Its
warps locate points on the mesh, and a point on an edge -- or a vertex warped by a short
step, which still sits at one -- has two or more faces at the same distance, rounding
choosing among them, and the face chosen changes where the point is taken to be by its
distance from the mesh times the angle between the faces; from there the descent goes its
own way. The CPU does the same with itself. Moved by one unit in the last place, one input
moves the result, its twin, on the ico2 grids of the tests by 2e-7 to 1.1e-4 of what is
compared after a few steps, by as much as 1.9% of the largest entry of an aligned density
after fifty steps of ENCORE, and 0.03% to 0.4% of ConSEAL's endpoints into a neighbouring
face (`tests/reference/gpu_probe.py spread`, ten twins of each, job 4685821). The GPU is
held to that spread: the tests allow it twenty times the farther of two twins, or, where
both happen to lie near, as one twin can lie eighty times nearer than another, a floor:
three ten-thousandths of the result until item 27, and since then ten times the farthest
of those ten twins, for each quantity. On two released subjects on ico4 both are far smaller
(`gpu_probe.py`, the CPU's results computed on the same node: on eight cores beside the
V100, on two beside the GTX 1080):

| | the CPU's twin | a V100 (jobs 4671538, 4707448) | a GTX 1080 (jobs 4671579, 4701939) |
| --- | --- | --- | --- |
| ENCORE, one registration of 19 steps: the cost trace | 3.7e-8 | 2.7e-8 | 9.1e-9 (its twin 7.7e-9) |
| the aligned density | 1.8e-7 | 2.2e-7 | 1.4e-7 (3.7e-7) |
| ConSEAL, ten iterations: the costs | 4.0e-9 | 2.0e-9 | 2.8e-9 (4.3e-9) |
| the endpoints, at most | 3.5e-5 rad | 3.9e-5 rad | 3.7e-5 rad (3.9e-5) |
| in the same face, both ends | 99.994% | 99.991% | 99.995% (99.988%) |

The template agrees to 1.4e-16 on the V100 and 2.2e-16 on the GTX 1080 (until item 28
this gave the V100's alone), a subject's density with its derivatives to 1.8e-16, and
the relocated endpoints bit for bit; run twice, every GPU result is the same, bit for bit.
(The twins differ between the two nodes since their CPUs round differently.) The two
endpoint rows are item 27's, by angle and at both ends of every streamline (jobs 4707448
and 4701939, every other row the same to the digits); the first runs took the largest
coordinate difference and the first end alone: 3.0e-5, 3.8e-5 and 3.2e-5 (3.3e-5) rad,
and 99.994%, 99.990% and 99.995% (99.986%).

**The eleven, compared subject by subject** (`tests/reference/gpu_cohort_probe.py`). Aligned
as the cohort figure aligns them -- ENCORE on the re-smoothed densities, ten iterations;
ConSEAL on the endpoints, thirty -- on eight cores of an AMD EPYC 7702 (job 4674259), on a
V100 (job 4674198) and on a GTX 1080 (job 4677274), and as the CPU's twins on another EPYC
7702 (job 4676794) and on an EPYC 9654 (job 4674260), all from the same inputs; compared in
jobs 4676795 and 4685707. The largest over the eleven subjects of each distance from the
CPU's result:

| | twin, EPYC 7702 | twin, EPYC 9654 | V100 | GTX 1080 |
| --- | --- | --- | --- | --- |
| ENCORE: the template | 8.6e-17 | 8.6e-17 | 1.0e-16 | 1.4e-16 |
| a subject's cost trace | 7.5e-8 | 6.1e-8 | 5.3e-8 | 4.8e-8 |
| its aligned density | 7.1e-6 | 3.2e-6 | 5.6e-6 | 4.9e-6 |
| a vertex of its warp | 3.1e-5 deg | 3.1e-5 deg | 3.1e-5 deg | 2.9e-5 deg |
| ConSEAL: the template | 6.7e-17 | 6.6e-17 | 1.6e-16 | 1.2e-16 |
| a subject's costs | 8.2e-9 | 7.7e-9 | 8.7e-9 | 8.6e-9 |
| its velocity field | 1.8e-7 | 3.1e-7 | 2.9e-7 | 2.9e-7 |
| an endpoint | 2.3e-3 deg | 2.3e-3 deg | 2.2e-3 deg | 2.3e-3 deg |
| endpoints in the CPU's face, at least | 99.988% | 99.988% | 99.987% | 99.985% |
| the connectome re-smoothed from them | 1.1e-6 | 1.3e-6 | 1.5e-6 | 1.1e-6 |

On all four every subject took the CPU's number of steps -- ten of ENCORE, seven for
subject D, and thirty of ConSEAL -- the mean pairwise correlations are the CPU's to six
decimals (0.716341 before, 0.761495 after ENCORE, 0.789990 after ConSEAL), and ConSEAL's
median lies 21.0347 to 25.0593 degrees from the subjects. An endpoint moves 2e-3 degrees
where it falls into another face, and 3e-7 to 8e-7 degrees on average.

**How fast, and in how much memory.** Against eight cores of the same node, an Intel
Xeon Gold 6148, on the two subjects (job 4671538):

| | 8 CPU cores | a V100 |
| --- | --- | --- |
| ENCORE: the template of three subjects | 72.5 s | 1.1 s |
| ENCORE: one registration, 19 steps | 84.9 s | 7.0 s |
| ConSEAL: a subject's density and its derivatives | 2.1 s | 0.06 s |
| ConSEAL: relocating 1.6 million endpoints | 17.2 s | 0.74 s |
| ConSEAL: ten iterations of a registration | 242.5 s | 14.3 s |

The GPU's first call takes a second or two more, compiling CuPy's kernels. The eleven of
the cohort figure, above, took 48 s for ENCORE and 393 s for ConSEAL on the V100 (65 s and
1,239 s on the GTX 1080), where eight cores took 570 to 1,240 s and 4,288 to 6,837 s (the
three CPU runs; 809 s and 7,894 s when the figure was drawn, job 4649389). One
registration holds 2.5 GB on the GPU with ENCORE and 3.9 GB with ConSEAL. The templates
keep the subjects' square-root densities on the GPU while they fill less than half its free
memory, and past that send them across at each step (`_device.Held`; since item 27, as many
as fit stay and only the rest are sent), so a cohort is not limited by the card; a test
forces the second way and gets the first's template.

### Tests

`tests/test_gpu.py`. Three run anywhere, CI included: a device other than `"cpu"` or
`"gpu"` is refused by all four entry points; without CuPy, `device="gpu"` is refused with
the line that installs it, before any work; with CuPy and no GPU, with what is missing.
Twelve need a GPU and skip without one: the point search against the CPU's, bit for bit,
on ico4 and an icosphere, at random points, at the vertices, where faces tie, just above the
edges and inside the sphere; endpoints carried through the same warps, bit for bit;
`Concon`'s transport and derivative; the endpoint density and its derivatives by both of
`evaluate`'s routes; `Encore`'s root, template and registration, and `align`; `ConSEAL`'s
template, cost and registration, plain, area-weighted and with the rotation search;
`endpoints_align` with each kind of template, a subject registered onto its own density
taking no step, and a CuPy template; a template too big for the GPU, sent across at each
step, is the one held there; and the GPU repeats itself, bit for bit, on both routes. The
registrations are compared with the CPU's twins, as above. The surface test now leaves the
package's private modules out of the submodules it requires.

### Verified

On Longleaf, from a frozen copy of the tree with the change, on the CPU in the environment
without CuPy (job 4671654, an Intel Xeon Gold 6140 node): the whole suite passes on the
released subjects, 1,226 tests with 26 skipped, the GPU's twelve among them, skipping with
the line that installs CuPy; Tier 4 passes 21 of 21; the audit passes 22 of 22, the tour
runs and `check_hcp_ya.py` passes on a fresh download of the eleven; under Python 3.10 the
alignment, ConSEAL, GPU, template, example and surface tests pass, 348 with 13 skipped.
Recorded before (03f4668) and after, one after the other on one node (`outputs-03f4668.npz`
against `outputs-gpu.npz`, job 4671655), all 29 arrays are identical, bit for bit, the two
alignments' among them. On a GPU, from the same tree: `tests/test_gpu.py` passes, 15 of 15,
on a GTX 1080 (job 4686029) and on a V100 (job 4674601, with the floor under the twins'
spread still at 1e-4, a tighter bound than the 3e-4 it has now); run twice, the product, a
subject's density, three steps of ENCORE and three iterations of ConSEAL give the same
bits on both (`gpu_probe.py repeat`, jobs 4674507 and 4674506). The probe ran on a
V100 and on a GTX 1080 (jobs 4671538 and 4671579, the tables above).
The eleven were aligned and compared in the jobs above.

## 27. A fourth review, 12 October 2026 -- NO HIGH-SEVERITY DEFECT; A DEPOSIT TABLE THAT COULD CARRY MORE, A POINT SEARCH NOTHING CHECKED ON AN IRREGULAR MESH, AND THE GPU'S EDGES

Asked, once the Dataverse ticket was submitted: "check again about the whole framework and
check any issue. try ur best to fix the issues".

Six reviewers read 81f6c99, one area each: the deposit tool with the scripts and tools, the
package outside the two alignments, ENCORE on the CPU, ConSEAL on the CPU, the GPU, and the
documents with these records. None found a high-severity defect. Every finding was checked
before it was fixed -- by a probe, a test that failed, or a mutation the suite let through
-- and every fix has a test that fails on 81f6c99, or kills a mutation that passed it.
On the CPU the alignments compute as before (*Verified*).

### The deposit tool

- **`subjects.csv` could carry more than its three columns.** upload read the table's
  header and its values, so a cell added to a row, under no column -- an exact age, say
  -- passed the dry run and verify and would have gone up. upload now takes the table
  only when it is, line for line, the one prepare writes from the subjects it lists.
- **The README could pin a branch.** Outside a git checkout, `code_version` said "main"
  without a word, and the README's install line and CHECKS.txt named `@main`, a branch
  that moves, where they promise "the commit below"; a copy of the tool inside a checkout
  of something else would have named that checkout's commit. Now prepare settles the
  commit before it removes anything, takes it only from a checkout whose top is the
  tool's own, and otherwise refuses unless `--ref` names a commit's 40 digits.
- **The eleven's build copies could go up.** Without `--released` prepare took every
  file from the build, so the eleven already public on Google Drive would have been
  deposited as other bytes (ten of their FC files differ from Drive's by up to 1.9e-9).
  A deposit holding any of them must now give `--released`.
- **A dry run checked nothing it would send.** A broken link and an `http://` server
  passed it; the upload then failed midway with a traceback. The dry run now refuses
  both, a file missing at upload is a refusal, and an SSL or socket error during a
  request (`SSLEOFError`, "no route to host") is a failure that may have reached the
  server, retried when the call can be repeated and looked for in the draft when not,
  where before it escaped as a traceback.
- **macOS's checksum line** lacked `--ignore-missing`, so checking a partial download
  failed there and not on Linux (`shasum` 6.02 takes it).
- **Two checks no test held:** removing prepare's `validate_file` call, or upload's
  check that `subjects.csv` lists the subjects the files are of, passed all 62 tests,
  every one of which skipped validation. Six one-line mutations of the new checks and
  the two old ones now each fail the tool's tests, 74 of them.

Elsewhere: tools/README.md no longer says every tool's output is committed under
`src/sbci/data/` (the HCP builders' and the deposit's are not), and
`scripts/show_build.sh` lints `scripts/` and checks the format, as CI does.

### The package

- **A quote left open in a subject table swallowed every row after it.** With
  `01,25,"A` in a table, the next three subjects were read as part of the first one's
  site, reported "not in the table", and `sbci cohort --require site` exited 0; a cell
  past `csv`'s size limit raised `csv.Error`, no `ValueError`, out of `load_cohort` and a
  traceback out of `sbci cohort`. Tables are read strictly now, and what `csv` cannot read
  is a `ValueError` naming the file and the line.
- **One NaN made `reduce` return a NaN fit without a word**: every score NaN, the
  explained fractions NaN, `local_test`'s p-values NaN and nothing significant. A subject
  that is not finite is refused as it is read, by its index and its file, and so by
  `project`.
- **`save()` replaced a read-only file**, through `os.replace`, and reset its mode, where
  every other writer of the package refuses one; it refuses one too (`sbci example --out`
  with it).
- **The cohort loader and `load` disagreed on what is a number.** Coordinates stored as
  text passed the loader and failed `load`; connectivity stored as numeric text, or as
  complex numbers (the imaginary part dropped with only a warning), loaded and was refused
  by the loader. Every array's kind is checked before its cast, by both. `read_header` had
  been tested only through a stand-in, so its two flags made always true passed every
  test; it is tested on real files.
- **An endpoint's contents were never checked.** A NaN barycentric weight, or an endpoint
  group of no streamlines, passed `load`, all eleven checks of `sbci validate` and the
  loader's `validate=True`, and then `smooth()` failed with a plain `ValueError`; weights
  of a compound type raised a `TypeError` out of `sbci.load` and crashed `sbci info`. The
  weights must be finite numbers, an empty group is refused and never written, and what
  the reader refuses is one of the package's errors.
- **What `sbci validate` never looked at.** The mask was cast to booleans before any
  check, so NaN, 2, -1 and 0.5 read as cortex; an FC file whose mask keeps nothing passed
  all eleven checks; and FC beyond [-1, 1] -- Fisher z values, say, which pass 1 wherever
  r passes 0.762 -- passed them too, and `to_atlas` clipped it at 0.999999 and averaged it
  as correlations. The mask is checked before its cast, a mask with no cortex fails, a
  twelfth check (`correlation`) fails FC past 1 in magnitude by more than rounding allows
  (`CORRELATION_ROUNDING`, 1e-4: a correlation computed in float32 passes 1 by up to
  4.8e-6), and `parcellate(fisher_z=True)` refuses such values rather than clipping them,
  as `region_means(fisher_z=True)` does too (a follow-up commit).
- **`sbci download`** died with an `IncompleteRead` traceback when Drive's chunked answer
  was cut off; it is an `OSError` now, which says to try again, and the partial file still
  goes.
- **Smaller ones:** `sbci cohort --report` refused a read-only report only after the whole
  scan, where item 24 said it was checked first; `load_cohort` refused a folder given as a
  path-like other than `pathlib.Path`, which `load`, `reduce` and `project` take; and the
  example's comment said its density covers "about six million vertex pairs", where
  1,720,328 of the 10,972,270 pairs of cortical vertices, 16%, carry any.
- **Tests that could not fail.** `test_stats_design.py` skipped all eighteen of its tests
  when statsmodels did not import, twelve of which use none of it: twelve one-line
  mutations of `design()` and `contrast()` passed everything that ran without it, as did
  one of the partial R-squared and one of `icc`. Only the six comparisons with statsmodels
  skip now; the partial R-squared is checked by least squares of its own, and `icc`'s NaN
  for a non-positive denominator has a test; all fourteen mutations fail.
  (statsmodels 0.15.0 imports under pandas 3.0.5 on Longleaf, as CI installs them, so CI
  runs the comparisons.) Of 25 one-line reverts of the fixes above, 24 fail the tests; the
  other, the reader catching `ValueError` alone, is equivalent now that the checks before
  it raise nothing else.

### ENCORE

- **The point search was exact on regular meshes only.** `MeshQuery` kept the best of
  the twelve faces whose centroids lie nearest a point unless that face lay more than
  four sagittas away, so on an irregular mesh a nearer face outside the twelve went
  unseen: on a sphere meshed finely except over a 35-degree cap, 130 of 4,000 points
  got a face 0.069 away where one lay 0.025 away, and transport on that grid differed
  from the exhaustive search's by 4.5 of 80 (a reviewer's probe). The best candidate is
  now kept only when no face outside the twelve can be as near -- every other centroid
  lies at least as far as the farthest candidate's, and no point of a face lies farther
  from its centroid than the mesh's largest centroid-to-corner distance -- and any other
  point is tested against every face whose centroid lies near enough to matter, which
  the k-d tree finds as well. On the bundled grids, the standard fs_LR spheres and the
  grid on fsaverage no point needs that; on the fs_LR spheres carried to fsaverage, which
  `migrate_warp` searches, 0.6% to 0.9% of random points do (0.2% to 0.4% of the
  vertices), and every one of them ends in the face the twelve gave
  (`tests/reference/mesh_query_probe.py`, job 4713317). The GPU's search keeps a point by
  the same rule, from its table's farthest centroid.
- **`Encore.template` lost `np.stack`'s checks** when item 26 rewrote it: a row or a
  column among the densities broadcast against the areas and gave a template, and an
  empty list an `IndexError`. The densities' shape is checked, and an empty list
  refused.
- **Settings checked and then used raw:** `template_iterations=10.0` passed the check
  and stopped `range()` with a `TypeError`, and `SphericalGrid(order=2.0)` passed its
  check and failed in the basis, so `align(order=6.0)` crashed; `Encore.template`
  took -1 steps and returned its starting subject; and an infinite threshold accepted no
  step, returning identity warps as though converged. Whole floats are taken as their
  numbers, the template's count is checked, and the threshold must be finite, in both
  alignments.
- **A template in single precision was refused as a density.** Normalized in float32, a
  5,124 x 5,124 square-root density's squares sum to 1 only within about 1e-6 (1.7e-6
  was seen), where the check that refuses a density passed in its place allows 1e-6. A
  float32 template within 1e-4 of unit mass is now taken in float64 and scaled to it, by
  `align`, `endpoints_align` and `ConSEAL.register`; in float64 the check is as before.
- **Lines nothing held:** where the template starts (from the first subject, or the
  farthest, or a mean that missed the last), its 0.005 rule (at 0.002 or 0.01), the
  doubling of a halved step after an accepted one (PORTING.md item 4), `align`'s
  tolerance on the template's mass, and `migrate_warp`'s Jacobian (inverted, or all
  ones) passed every test. Tests now pin each, and each mutation fails them.
- **Docstrings:** `MeshQuery`'s said ties went to the lowest index "as libigl did on every
  point of the ENCORE reference", where item 4 found libigl's choice agrees at about
  85% of the points where faces tie; and `pole_vertices`'s 1 to 3e-5 is the bundled
  grid's, now printed by `encore_axis_probe.py` (1 +- 2.7e-5 to 2.8e-5 at the
  clearance).

### ConSEAL

- **A connectome built on other grids was registered in the wrong frame.** Nothing
  compared an `EndpointConnectome`'s grids with the registration's: one with as many
  vertices in other places was read by vertex index, and its endpoints moved 79 degrees
  on average under a step that moved no vertex (95 after three iterations of
  `endpoints_align`); a coarser one met SciPy's "dimension mismatch". Subjects, a
  template connectome and `register`'s two are now compared by their vertices and faces
  (grids built anew from the same mesh are the same grids) and refused with what to do.
- **Item 26's template held a generator whole.** Listed first, to count the subjects for
  the GPU, eight lazily built subjects took 95.5 MB more than one, against 19.5 MB at
  03f4668; on ico4 a subject's endpoints are about 180 MB beside its 210 MB density.
  Both templates read their subjects one at a time again, on either device: the GPU now
  keeps as many as fit and sends the rest across, needing no count.
- **One subject with no template** returned identity warps and a cost of 0, where `align`
  refuses it; `endpoints_align` refuses it too.
- **The kernel's settings were never checked.** A negative bandwidth built the identity
  kernel, under which nothing moved; NaN or inf failed only after the kernel and the
  template were built, blamed on the template; a degree of 0 made every density uniform,
  2.5 ran as 2, and -3 raised an `IndexError`. Both are refused before anything is read
  or built, and `ConSEAL.template` checks its count of steps as `endpoints_align` does.
- **Lines nothing held:** the median's start (from the first subject or the farthest),
  its step (a third for a fifth), the turns about z one way only, and the heat kernel's
  cutoff one scan point further out passed all 113 tests; each fails one now, the
  cutoff against a Legendre series of NumPy's own at both bandwidths.
- `mesh_symmetries` said FreeSurfer's sphere is 1.7e-4 off an icosphere, "a fortieth of
  its edge length": it is about a four-hundredth of its shortest edge (0.0691).

### The GPU

- **The relocation's order of addition held on x86 only.** On the CPU `_points` used
  `einsum`, which adds a point's three corners as the GPU does on x86 but fuses them on
  arm64 (65.5% of points the same there), so "bit for bit" was x86's. The CPU now adds
  them in that order itself: on x86 nothing changes (*Verified*), and elsewhere the CPU
  finds the GPU's points.
- **The CPU converted a kernel derivative's parts to CSR,** since item 26, so a CSC
  derivative was summed in another order than before (2e-17). It takes them as given
  again; only the GPU converts.
- **Smaller ones:** the GPU's product took float32 or integer arrays as an error of
  CuPy's and checked no shapes; `align(device="gpu")` refused CuPy arrays; a stand-in
  with a `q_transform`, which the CPU takes, met an `AttributeError` on the GPU, now a
  `TypeError` that says what it takes; `scripts/hcp_figures.py --device` was checked
  after the single-subject figures, now before the data are read.
- **CuPy without CUDA's libraries passed the check.** On Longleaf without `module load
  cuda`, CuPy saw the GPU and the run failed at the first kernel, with
  `libnvJitLink.so` missing (job 4701939). The check now compiles a kernel, and refuses
  with what to load.
- **Tests that could not fail.** The GPU's search of every face ran only for points
  inside the sphere, where the candidates were right anyway: switched off, or breaking
  ties to the highest index, it passed. A test now turns every point's candidates to
  the faces farthest away, so every point is searched, and the faces come out the
  CPU's at the vertices too; both mutations fail it (on a NumPy stand-in for CuPy, as
  the GPU reviewer ran them). The registrations' floor of three ten-thousandths decided
  every comparison, above twenty times the twins: each quantity has its own floor now,
  ten times the farthest of the ten twins of job 4685821. A template held partly on the
  GPU repeats the held one bit for bit, ConSEAL's included, where the test had allowed
  rounding.
- **`gpu_probe.py`** measured how far the endpoints lie by their largest coordinate
  difference, not their angle, and the share in the same face by the first end alone;
  both are now taken over both ends, the distance as an angle. Item 26's table has the
  new runs.

### The documents and records

- **The skip count** was 38 in README and VERIFICATION since item 26 added the GPU's
  tests, which skip on any machine without a CUDA GPU, CuPy or not; the suite skips 55
  there now, and both name the seventeen.
- **USAGE's examples stopped a newcomer.** The covariate examples used `age`, `family`,
  `score`, `female`, `band` and `count` before defining them -- `age` cannot be built from
  the young adults' open-access table, which has age bands -- and called the fit
  `reduction` after naming it `result`; `fetch_cohort("hcp-ya")` passed the folder as
  `out` by position; *Setup* downloaded into the current directory where later examples
  read `hcp-ya/`; and ConSEAL's notes recommended the mean of `q_transform(kernel)`
  arrays, a method of an internal class with a kernel USAGE never built. Each name is now
  defined where it is used, from the cohort's table and the files' metadata, with the
  restricted family table and the open-access score said to be the reader's own; the mean
  is built in the lines that do it. Every changed block ran, on synthetic stand-ins for the
  eleven where HCP data would be read.
- **Components are counted from 1** in USAGE and docs/RESULTS.md and from 0 by the code;
  both say so where the numbers first appear.
- **ENCORE's 59%** of a known one-degree deformation undone was item 7's measurement from
  before the review of 5 October and item 21's order of composition, in the present
  tense. `tests/reference/known_warp_probe.py` measures it again: 59.1% with the basis of
  order 15 the deformation was drawn in, an eighth at the default order, 6, and none with
  `backtracks=0`, the reference's rule (job 4707445, as a Mac prints it); the last traces item 4's
  single step, which item 24 listed as untraced. No document had said that at the default
  order ENCORE undoes an eighth of a deformation drawn in the finer basis.
- **Timings quoted two ways:** USAGE gave the eleven's ENCORE alignment as six to fourteen
  minutes on eight cores in one place and 9.5 to 21 in another (items 25 and 26, other
  nodes), and set item 19's 12 s an iteration, on an AMD EPYC 75F3, beside item 26's 24 s
  on a V100 node's Xeon Gold 6148, as though the defaults had doubled the cost. One range,
  six to 21 minutes by the node, and each processor named.
- **The review note** still said the reference's search had left a turn about z "8 to 9
  degrees off"; it carries item 18's 9.4 and 8.0 as a dated correction.
- **The records:** items 8, 9 and 20 described `strict_upstream=True` and
  `reference=True` in the present tense in ten places; each says "until item 21" or
  "22" now. Item 26's untraced figures trace now (*The GPU*), its 28 candidate faces are
  25, its endpoint rows are measured again, and its floor and its store are described as
  item 27 left them.

### Verified

On Longleaf, from a frozen copy of the tree, on the CPU in the environment without CuPy:
the suite passes on the released subjects, 1,316 tests with 32 skipped, the GPU's seventeen
among them (job 4712067, an Intel Xeon Gold 6140 node). Job 4707445, on an AMD EPYC 9654,
had found one new test asking too much: there a float32 template's registration came within
9e-5 of the float64 one's costs, as the CPU's own twins do after a few steps, where the test
had asked 1e-6; it asks 1e-3. As a fresh clone has it, with no data, 1,313 pass and 35 skip in
six minutes on one core (55 skip off Longleaf, where the MATLAB references are not). In job
4707445: Tier 4 passes 21 of 21, and 21 of 21 on a fresh import; `encore_axis_probe.py`,
`encore_first_checks_probe.py`, `conseal_recheck_probe.py` and `template_probe.py` print what
the records quote, and `known_warp_probe.py` what the Mac printed; the audit passes 22 of 22,
the tour runs, `check_hcp_ya.py` passes on a fresh download of the eleven, and
`write_exchange_file.py`'s three claims hold; and `endpoints_align` on two released subjects,
five iterations, takes 245 s on four cores, both costs falling at every step, as USAGE says.
Under Python 3.10 the files the review touched pass, 834 tests with 19 skipped (job 4712862).

- **Every file the deposit holds** -- the 946 subjects' SC and FC and the eleven's released
  copies, 1,914 -- passes the twelve checks of `validate_file`; the largest FC entry is 1.0
  in magnitude (job 4707444).
- **The CPU computes as before.** Recorded before (81f6c99) and after, one after the other on
  one node, the 29 arrays of `record_outputs.py` on the released subjects are identical, bit
  for bit (job 4707445), and the 95 of `cpu_parity_probe.py` across 03f4668, 81f6c99 and this
  tree (job 4707454, an AMD EPYC 9654).
- **On a GPU:** `tests/test_gpu.py` passes 21 of 21 on a V100 (job 4707448); on a GTX 1080
  it passed 20 of 20 from the tree before the last changes, which touched no GPU arithmetic
  and added the twenty-first test, one that runs without a GPU (job 4701939). `gpu_probe.py`
  gives item 26's table (jobs 4707448 and 4701939); without the CUDA module the check
  refuses at once (job 4707448); and `scripts/hcp_figures.py --only-cohort --device gpu`
  draws the cohort figure on a V100 in 14 minutes, with the CPU's correlations, 0.7163,
  0.7615 after ENCORE and 0.7900 after ConSEAL, and the median 21.03 to 25.06 degrees from
  the subjects (job 4707449).
- **Mutations:** every one a reviewer showed the suite let through fails the tests now:
  ENCORE's and ConSEAL's sixteen, the GPU's two (on the NumPy stand-in), the deposit tool's
  two and the package's fourteen; so do the deposit tool's four new checks undone, and 24 of
  the package's 25 one-line reverts of its fixes (the other is equivalent, above). `ruff
  check` and `ruff format --check` pass on src, tests, scripts and tools.

## 28. The GPU's own review, 13 October 2026 -- NO WRONG NUMBER; TABLES LEFT ON THE GPU, CONNECTOMES FROM THE OTHER DEVICE, A ROTATION SEARCH ON THE CPU, AND TESTS THAT LET WEAKER RULES PASS

Asked, after item 27: "keep fixing until u fix all the bugs/issues in the gpu side".

Two reviewers read e6616a8's GPU side alone: one its code paths through every entry point
and option, on a NumPy stand-in for CuPy and on real GPUs; the other its tests, probes and
documents, with 47 one-line mutations of the GPU's branches. Meanwhile the lead fixed what
item 27's review had left in sight, the point search's memory and the check's message, and
ran the GPU at the released subjects' scale on the options the tests try only on small
grids. Neither reviewer found a wrong number: in 13 ENCORE and 20 ConSEAL scenarios and
four method calls -- float32, integer and CuPy densities, each kind of template, the
analytic derivative, zero thresholds and iterations, subjects with one hemisphere, the
rotation search, area weighting -- the GPU gave back NumPy, templates within 3e-15 of the
CPU's and registrations within the CPU's twins. What they found: tables the GPU kept as long
as the subjects that made them, connectomes and kernels on one device met by an engine on
the other, tests that let weaker rules pass, and documents that claimed more than the runs
had shown; the lead's run found the rotation search taking nearly all of a registration's
time on the GPU. One suspicion did not hold (*A connectome with no streamlines*). Every
finding was checked before it was fixed, and each fix has a test that fails on e6616a8 or
kills a mutation that passed it. On the CPU the alignments compute as before (*Verified*).

### The tables the GPU searches

Each `EndpointConnectome` builds two point searches, one a hemisphere, and every copy of it
shared them. The GPU's search makes its tables -- the faces' corners and centroids, each
face's 36 nearest and how far they reach -- the first time it is asked, and keeps them on
the search. The copy `register` sends to the GPU searched through its subject's own, so the
tables were made there and stayed on the GPU as long as the subject did, or the connectome
handed back: on a V100, `endpoints_align` of six synthetic subjects on ico4 left 24.8 MiB on
the GPU, 4.14 MiB a subject, and each registration after it 4.14 MiB more, held by its
result -- 46 MB for the eleven, and 3.8 GB for 946 held in a list, half a GTX 1080. A copy
on another device now has searches of its own, sharing the k-d tree and the corners
(`MeshQuery._without_tables`), and the tables go with it: the same run leaves nothing on the
GPU (`gpu_review_probe.py tables`, on e6616a8's code and on this, on a V100 and a GTX 1080,
jobs 4722437 and 4722468). A test aligns a cohort on the GPU and finds no tables on the
subjects or the connectomes handed back, and CuPy's pool holding what it held before.

### The point search's memory

The GPU's search for each point's nearest centroid took 4,096 points by every centroid a
pass, and its search of every face 256 points by every face, each pass holding some twenty
temporaries of that many triples; the CPU's search of every face did the same. On the ico4
grids that is a fraction of a gigabyte, but the buffers grew with the faces: a grid passed
through `grids=` of icosphere(6), 81,920 faces, would have taken some ten gigabytes a pass,
more than a GTX 1080 holds. Each pass now takes a fixed budget of pairs
(`_PAIRS_PER_PASS`, 2^20, and `_CENTROID_PAIRS_PER_PASS`, 2^24, for the centroids), a
point at least, and so does the CPU's search of the faces near an unsettled point (item
27). On icosphere(6) the GPU's search holds under 2 GB and finds the CPU's faces (a test);
on ico4 the passes are 204 and 3,276 points, and nothing else changes.

### Connectomes, kernels and derivatives from the other device

- **A CPU engine took a connectome on the GPU as it was.** Given two, it registered on the
  GPU and handed back a connectome there; given one, CuPy refused to mix it with the CPU's
  arrays. Each engine now copies an `EndpointConnectome` on the other device to its own,
  either way, and `register` hands back on the CPU whatever device its copy was on.
- **A derivative on the GPU** met a CPU engine's diagonals as CuPy matrices; it is fetched,
  as a kernel on the GPU already was.
- **A dense kernel on the GPU** had no `tocsr`, on either engine; it becomes a CSR matrix,
  as a dense NumPy kernel does.
- **A stand-in for the moving connectome** -- anything with a connectome's methods, which
  the CPU registers -- met an `AttributeError` on the GPU; it is refused with the
  `TypeError` item 27 gave a stand-in for the fixed one.

A test registers each on a CPU engine and gets, bit for bit, the registration of the same
inputs fetched by hand; a dense kernel on the GPU gives a GPU engine the registration of the
dense NumPy one, bit for bit; and a connectome given on the GPU is left as it was.

### The rotation search

Item 26 left `init_rotation`'s search on the CPU, as small; on the released subjects it is
nearly all of a registration. On two of them, three iterations took 204.5 s on a V100, the
search 199.9 s of it, where without the search they took 5.1 s, and on the node's eight
cores 297.9 s, the search 219.3 s (job 4720863); `endpoints_align` of the two with the
search took 439 s on the V100 against the CPU's 695 s (job 4716126). The search pulls each
hemisphere's block back through some 650 rotations, each a point search for the rotated
vertices and two sparse products, and it now does that where the engine computes: on the
GPU the vertices are located by its search, which finds the CPU's faces and weights bit for
bit, each row's three weights kept in the order SciPy's CSR keeps them, and the products
added by `_device.rows_times_dense`; the candidates, 3 x 3 rotations, are kept and composed
on the CPU either way. A block pulled back on the GPU is the CPU's to rounding (a test), and
the rotations the search picks are the CPU's, bit for bit, in `tests/test_gpu.py` on a V100,
a GTX 1080 and an L40S and on the two released subjects, where `endpoints_align` with the
search now takes 37 s on the V100 against the CPU's 530 s (`gpu_review_probe.py rotation`,
job 4722437). The three iterations take 21.1 s on the V100, the search 16.8 s of it, against
268.5 s on the node's eight cores; 45.3 s on a GTX 1080, the search 32.2 s (job 4722468);
and 17.7 s on an L40S, the search 13.4 s (job 4806991). On the CPU the search computes as
before, bit for bit (*Verified*).

### The check of a usable GPU

Item 27's check compiles and runs a kernel, and said, when that failed, that CUDA's
libraries were missing. A GPU that is busy, or in another process's exclusive use -- as one
of Longleaf's GTX 1080s now and then is -- fails the same way; the message now names both,
with the error itself (a test of each). The test's stand-in failed when its kernel was
made, but CuPy compiles a kernel when it is first called, so a check that made one and never
ran it passed; the stand-in fails at the call now, and a CuPy that counts no device is
tested too. With CuPy's kernel cache warm from earlier runs and without the CUDA module, the
check still refuses at once, on a V100 and on a GTX 1080 (CuPy finds no CUDA headers; jobs
4720863 and 4720865): a warm cache hides nothing.

### A connectome with no streamlines

The CPU refuses a template's subject, or a fixed connectome, with no streamlines on the
grid, and says so. A reviewer suspected that on the GPU its adjacency reached CuPy's
`bincount` with an empty array, which it could not size, before the package could refuse
it. CuPy 14, which the `gpu` extra requires, returns an empty count for one (its source), and
e6616a8 refuses such a connectome on the GPU as on the CPU (`gpu_review_probe.py empty`,
jobs 4722437 and 4722468), so nothing changed; a test holds the GPU to it.

### Tests that let weaker rules pass

The GPU's test on an irregular mesh used test_alignment's capped sphere, whose cap faces
reach 0.314 from their centroids, farther than the GPU's table spans at all but 15 of its
6,020 centroids, so 1,673 of its 1,702 points went to the search of every face and the rule
that keeps a best candidate was hardly asked: a bound without the point's own distance from
its nearest centroid passed it, and gives a farther face on 3 of 300 random meshes (the
reviewer's). On the CPU, 1,699 of the 1,702 went on to the faces near them. test_alignment's
`small_irregular_mesh` now draws such meshes -- the hull of 40 to 400 points drawn denser
toward an axis, with points 0.3 to 1.6 from the centre -- and on four of them the CPU's
faces and the GPU's are the closest, their candidates settling some points. Without the
point's distance, the GPU's bound gives a farther face on three of the four; the CPU's
without the reach, or with half of it, on all four, 2 to 78 points of 3,000. One weaker
bound passes still: the GPU deciding on the squared distance, which settles more points but
gave no farther face on the reviewer's 300 meshes nor on 150 more.

The bundled grids' test that no point needs the search of every face has a GPU twin, which
fails a table that takes its first candidate's reach, the farthest centroid, the worst
candidate, a third as many faces, or no bound. `Held` is tested on what it keeps -- NumPy
arrays for those that wait, CuPy ones when read -- and on room that counts the pool's free
memory; the GPU's product on a float32 matrix; the adjacency's one entry a place, which a
sum that kept every entry passed, the products adding them up anyway; `align` with a CuPy
template; `register` leaving a connectome given on the GPU as it was, which a copy that
shared its arrays passed; and the analytic derivative, run twice the same bits and the CPU's
to 1e-13.

Of the GPU reviewer's 47 one-line mutations, the tests let 28 through at e6616a8 (on the
NumPy stand-in for CuPy), 16 of them changing what the GPU computes and 12 only its speed or
a choice among equals; 12 pass now. Ten are of the second kind: a table of every face
sending points on, the order a place's values are added in (the same every run), room
exactly equal to an array, a scale where the density and its slope are both zero, a copy
too many, a kernel, template or grid sent at each use instead of once, and the last of tied
starting subjects. Two are of the first: the squared distance, above, and cuSPARSE's product
in the analytic derivative, which the stand-in cannot tell from `rows_times_dense`; on a
GPU the derivative repeats itself, bit for bit (a test, and `gpu_review_probe.py repeat`).

### The documents and records

- **The eleven's time on the CPU** was 9.5 to 21 minutes for ENCORE in USAGE's GPU section
  and six to 21 by the node in its cohort section, item 27's one range. Six is job
  4648690's 349 s on an AMD EPYC 9654, and the same processor took 1,240 s in job 4674260,
  so the range is the node's: both say six to 21 now, on Longleaf's AMD EPYC nodes.
- **"Every subject's results lay as near the CPU's as the CPU's own twins"** (USAGE). Over
  the eleven the GPU's largest distances are about the twins' largest -- an aligned density
  5.6e-6 from the CPU's against 7.1e-6, ConSEAL's costs 8.7e-9 against 8.2e-9 -- but subject
  by subject the V100 lay up to 6.8 times as far as that subject's twins on an aligned
  density (subject E, 5.6e-6 against 8.2e-7), and 25 times on the mean movement of a
  warp's vertices, where the twins happened to land within 3e-9 degrees (jobs 4676795 and
  4685707). USAGE says so now, and its table of times has the rotation search.
- **Item 26** gave the template's agreement as 1.4e-16, the V100's; the GTX 1080's is
  2.2e-16 (job 4671579). It said the GPU's point search sends a point on by the CPU's rule:
  the two keep a best candidate by one rule, each with its own bound on the faces outside,
  and send the rest on their own ways, the CPU to the faces near enough, the GPU to every
  face. It said its twin table had the CPU's times on the same node's eight cores: the table
  holds distances, and beside the GTX 1080 the CPU had two cores. It said the recorder
  takes both templates at ten steps and at the rule; ConSEAL's it takes at the rule only.
  And its table put the rotation search on the CPU. Each is corrected where it stands.
- **tests/test_gpu.py's repeat test** said the transposes were stored row by row to stop
  cuSPARSE's variation; the product does that, `_device.rows_times_dense` (item 26).
- **The skips:** `tests/test_gpu.py` has 39 tests now, 34 of them needing a GPU, so the
  suite skips 72 off Longleaf (README, VERIFICATION).

The reviewers' probes are kept as `tests/reference/gpu_review_probe.py`, one mode each, with
the rotation search's times and its rotations on the released subjects.

### A registration's time

Asked once this was committed: how long one registration of a subject takes on each device,
onto the template and onto the first subject. `tests/reference/registration_timing_probe.py`
aligns the eleven as the cohort figure does, twice -- onto their template, and onto the
first subject -- and times every call of the engines' `template` and `register`, the GPU
synchronized (8a5691a's code; jobs 4777745 and 4777746 on eight cores of an AMD EPYC 75F3
and of an EPYC 9654, 4777743 on a V100, 4777744 on a GTX 1080). Every registration ran its
full count, ten steps of ENCORE (subject D's onto the template seven) and thirty iterations
of ConSEAL, so no stopping rule decided anything. The mean, and the range, of a
registration, the first subject's onto itself left out:

| | 8 CPU cores | V100 | GTX 1080 |
| --- | --- | --- | --- |
| ENCORE, onto the template | 24.6 s (21.1 to 26.9) | 2.9 s (2.6 to 3.1) | 4.2 s (3.8 to 4.5) |
| ENCORE, onto the first subject | 23.1 s (22.3 to 24.0) | 2.5 s (2.3 to 2.7) | 3.7 s (3.5 to 4.0) |
| ConSEAL, onto the template | 347 s (304 to 417) | 35.5 s (32.9 to 39.5) | 109 s (98 to 126) |
| ConSEAL, onto the first subject | 361 s (331 to 424) | 35.5 s (32.8 to 39.5) | 110 s (98 to 127) |
| the templates, ENCORE's and ConSEAL's | 54 s and 72 s | 2.3 s and 2.5 s | 5.4 s and 5.6 s |
| the eleven onto the template, all of it | 333 s and 3,890 s | 43 s and 394 s | 63 s and 1,209 s |

A ConSEAL iteration costs four to eight and a half times an ENCORE step (1.18 s against
0.29 s on the V100, 3.6 s against 0.43 s on the GTX 1080): ENCORE carries a density through
the warp by barycentric interpolation, three weights a row, where ConSEAL carries a
subject's 1.6 million endpoints, finds each one's triangle on the warped mesh -- on the CPU
on one core, the k-d tree's query and NumPy's arithmetic being single-threaded (until item
36, which runs that search in threads, and corrected this: the triangles are found on the
grid) -- and
rebuilds the density through a heat kernel of 32 entries a row, with its derivatives. The
CPU's nodes here are among Longleaf's faster: in item 26's runs the eleven took 570 to
1,240 s with ENCORE and 4,288 to 7,894 s with ConSEAL.

### Verified

On Longleaf, from a frozen copy of the tree, on the CPU in the environment without CuPy: the
suite passes on the released subjects, 1,323 tests with 49 skipped (job 4721738, an AMD EPYC
9654 node), and as a fresh clone has it, with no data, 1,320 with 52 skipped, the GPU's
thirty-four among them; under Python 3.10 the files of the alignments, the GPU and the
package around them pass, 841 tests with 36 skipped (job 4721741). The tree committed
differs from that copy in its documents, this record and `gpu_review_probe.py`; the tests
that read the documents pass on it, 226 with 5 skipped (on a Mac).

- **The CPU computes as before.** The 95 arrays of `cpu_parity_probe.py`, ConSEAL's rotation
  search among them, are identical, bit for bit, across 03f4668, e6616a8 and this tree, one
  after the other on one AMD EPYC 7742 node (job 4721739); recorded at e6616a8 and after on
  one node, the 29 arrays of `record_outputs.py` on the released subjects are identical (job
  4721740).
- **On a GPU,** from the tree committed: `tests/test_gpu.py` passes 39 of 39 on a V100 and
  on a GTX 1080 (jobs 4722437 and 4722468), and with test_alignment's point-search tests 47
  of 47 from the frozen copy (jobs 4721728 and 4721730). In the same jobs
  `gpu_review_probe.py` finds what the sections above quote, and besides: a released
  subject's 1,607,482 endpoints relocated with none sent to the search of every face, in
  0.73 s on the V100 and 3.0 s on the GTX 1080; the analytic derivative, the sparse route on
  ico4 and the area-weighted cost the same bits twice, and the GPU's adjacency the CPU's
  182,934 entries; every registration the tests compare within 0.062 of its allowance on the
  V100 and 0.077 on the GTX 1080; and, without CUDA's module, the check refusing at once.
  An L40S, a later generation, passed them too, 39 of 39, and the probe's repeat,
  relocation, margins and rotation search agree there: the same bits twice, the endpoints
  relocated in 0.72 s with none sent to the search of every face, every registration within
  0.062 of its allowance (job 4806991). The A100's run, queued from the frozen copy, ran
  last, on 13 October: 47 of 47 on an A100-PCIE-40GB, in 232 s (job 4721729).
- **At the released subjects' scale,** on a V100 (job 4716126): three subjects' densities,
  float32 as the files store them, aligned by `align` on the GPU give the CPU's template to
  2.6e-16 and its costs to 2.8e-8; and a template of four held partly on the GPU repeats the
  one held whole, bit for bit.
- **Mutations:** besides the reviewer's, above, 21 the lead ran on the fixes themselves --
  each undone, or a bound weakened -- fail the tests, all but the squared distance. `ruff
  check` and `ruff format --check` pass on src, tests, scripts and tools.

## 29. The deposit in archives, 13 October 2026 -- 1,892 FILES IN 40 TAR ARCHIVES, AND DIRECT UPLOAD

UNC Dataverse's technical staff answered the deposit ticket's questions on the collection,
the upload and replacing files (its curation questions went to the repository's archivist):
a dataset is best kept to about a hundred files, the files grouped into archives that make
sense to the people downloading them; a large deposit goes up with DVUploader by direct
upload to the repository's object store, which UNC's Dataverse 6.8 offers; and a file
replaced in a draft stays on storage and every published version is kept, so only the final
copy should go up, after a rehearsal on demo-dataverse.rdmc.unc.edu or
dataverse-staging.rdmc.unc.edu. Asked which grouping is better, about 50 subjects an archive
or about 100: "may be 50?", and then "go ahead".

### The archives

`prepare` keeps every check it made, then puts the files in tar archives in
`DEPOSIT/archives`: the eleven of the tutorial, already released on Drive, in
`sc_tutorial.tar` and `fc_tutorial.tar`, so that they can be had without the rest, and the
other 935, in order, in as few archives as hold at most 50 each, as nearly equal as they
divide: `sc_01.tar` to `sc_19.tar` and `fc_01.tar` to `fc_19.tar`, of 49 or 50 subjects, the
SC and FC archives of one number holding the same subjects. With six documents beside them,
the dataset is 46 files. The HDF5 files are compressed inside already (SC 50 MB a subject,
FC 40 MB), so the archives are plain tar: compressing them again would gain nothing. A
member is the file's bytes under `sc/` or `fc/`, so archives extract beside each other into
the deposit's own folders, and every member's SHA-256 is checked against the manifest as it
is copied: a file changed after its check stops the run. Every member has one modification
time, owner and mode, so the same files always give the same archive, byte for byte, and a
later `prepare` keeps an archive whose files have not changed (`.archives.json` records its
members and digests), rewriting only those that would differ; it removes only the archives
that record names, and refuses a folder holding anything else.

`ARCHIVES.tsv`, a document of the dataset, gives each archive's subjects, size and four
checksums, so `grep sub-NNNNNN ARCHIVES.tsv` names the two archives a subject is in;
`SHA256SUMS` gives every archive's checksum and then every HDF5 file's, so one command
checks a download and then what it extracts (`sha256sum -c SHA256SUMS --ignore-missing`); and
`MANIFEST.tsv`, a local record, gives each file's archive. The README says all of this.

### Direct upload, from the tool

DVUploader takes the API token on its command line, and on Longleaf's login nodes the
processes of 190 other accounts were visible, with their arguments, so for the hours an
upload takes the token would be readable by anyone logged in. `upload` does the same direct
upload itself instead, through the API DVUploader uses. Each file's SHA-256 is checked
first; the server hands out signed addresses on its object store, the file goes there whole
or, past the size the server names, in parts, each part sent again when its transfer fails,
and an upload in parts that will not finish is aborted; the file is then registered in the
draft with the checksum of the type the installation uses (`/api/files/fixityAlgorithm`,
MD5 where it does not say). The token goes to the Dataverse server alone: the store's
addresses are signed, and an address the server hands back on another host gets no token.
The tag that marks an upload not yet registered is sent where the address was signed with
it. `--through-server` sends a file through the server in one request instead, for a server
without direct upload, up to 4 GB. `--limit 1` now sends the documents and the tutorial's
two archives, about a gigabyte, for the rehearsal. Resuming, never adding a file twice,
the restriction, the lock, the redirects and the token's handling are as they were.

### Tests

`tests/test_dataverse_deposit.py`, 88 pass. Its stand-in for Dataverse now also stands in
for the object store: signed addresses with and without the tag, uploads in parts, their
completion and abort, the checksum type, and failures of the store. New: the archives'
members, headers, grouping and records; the same files giving the same bytes and unchanged
archives kept; leftovers and archives no longer planned removed, a folder holding anything
else refused; a member changed after its check stopping the run; a large file in parts with
no token at the store; no tag where none is signed; the server's checksum type; a failed
transfer sent again and a refused one aborting its parts; the route through the server, and
its limit; an archive table that does not match the manifest; an address on another host
getting no token. Twelve one-line mutations of the new code -- the member check, the tag,
the token at the store, the checksum type, the abort, the reuse, the leftovers, the archive
folder's check, the check before sending, the table's check, the grouping and the members'
time -- each fail the tests.

### Verified

The deposit's tests pass, 88 with one skipped outside a checkout, and `ruff check` and
`ruff format --check` pass on src, tests, scripts and tools; CI passes at 53eacc0.

The deposit was prepared again on Longleaf from the checkout at 53eacc0, clean (job 4824609,
an AMD EPYC 9654 node, eight workers): every one of the 1,892 files passed `sbci.validate`
and the metadata checks again, and the 40 archives were written in about eleven minutes,
85.06 GB -- `sc_tutorial.tar` 0.56 GB and `fc_tutorial.tar` 0.44 GB, the others 2.39 to 2.51 GB
(SC) and 1.96 to 2.00 GB (FC), of 49 or 50 subjects. Read back by a script of the job's own,
as a downloader would, without extracting anything, every archive's SHA-256 and those of the
1,892 files in them match `SHA256SUMS`, and every archive holds the subjects `ARCHIVES.tsv`
lists: no problem. `PROCESSING.md`, `DATA_USE.txt` and `subjects.csv` are as they were; the
README describes the archives and names 53eacc0. The dry run plans 46 files, 85.06 GB, by
direct upload. Nothing has been uploaded: the rehearsal on the demo server and the upload
are the user's, with the user's own token.

### The rehearsal on the demo server, 14 October 2026

The user rehearsed the upload on demo-dataverse.rdmc.unc.edu with their own token, on a
scratch draft. `upload --limit 1` sent the six documents and the two tutorial archives,
eight files and 1.00 GB, by direct upload. `verify --limit 1` then found "all 8 files
present, every size, checksum and restriction as prepared". `fc_tutorial.tar` and
`SHA256SUMS` were then downloaded from the draft in a browser on macOS, and
`shasum -a 256 -c SHA256SUMS --ignore-missing` passed the archive and the eleven FC files
the browser unpacked from it. The user then deleted the draft and revoked the token.

The first attempt stopped at its first request. The DOI had been copied with the prefix
of an example, `doi:10.5072/FK2/...`, but this server's DOIs are `10.33563/DMO/...`.
Dataverse 6.10.1 answers any DOI it does not hold with 404 and "When accessing a dataset
based on Persistent ID, a persistentId query parameter must be present". A request with
the dataset's own DOI got 200 and an empty list of locks. Since then, `upload` and
`verify` first ask the server for the dataset, and stop on an unknown DOI, saying so
(`test_a_doi_the_server_does_not_hold_is_named_before_anything_is_sent`).

The servers' versions, from `/api/info/version` the same day:

- dataverse.unc.edu runs 6.8.
- The demo and staging servers run 6.10.1.

From 6.10, a file download through the API needs the guestbook response when the
dataset has a guestbook; without it the server answers 400 (IQSS #12001 and #12110,
which the RDMC archivist pointed to). The deposit's `upload` and `verify` read only
metadata, so neither is affected. The Dataverse fetcher `sbci download` will need once
the data are published must send the guestbook response.

The user chose a guestbook whose only required answer is "I agree" to the HCP terms;
name, email and institution are left optional. Browser downloads already show
Dataverse's own window for the terms.

## 30. The release on UNC Dataverse, 15 October 2026 -- ALL 946, ANY SUBJECT BY ITS OWN BYTES, THE TERMS ACCEPTED ONCE A MACHINE

Asked once the dataset was published: "check again the repo, find any thing u can add or
fix", with the dataset's address. Two choices were the user's. The terms: the first download
on a machine shows the HCP's terms and asks for "I agree", the acceptance is remembered,
scripts and CI pass `--accept-terms`, and the package answers a Dataverse 6.10 guestbook
"I agree". The source: everything from Dataverse -- the eleven of the tutorial from the
tutorial's archives, which hold the bytes released on Google Drive, `--subject` any of the
946, `--all` every one -- and Google Drive no longer read.

### The upload and the publication

- **The upload, 14 October, 21:20 to 21:47.** The user set their own token in a terminal
  session on Longleaf; `upload` sent the 46 files of item 29, 85.06 GB, by direct upload at
  about 54 MB/s, and `verify` found "all 46 files present, every size, checksum and
  restriction as prepared". The token was then cleared from the session.
- **The review.** The PI has the Admin role on the SBCI collection.
- **The publication, 15 October.** Version 1.0 of doi:10.15139/S3/NDAJYX was published at
  13:07:28 UTC, its collection first: 46 files, the HCP's terms as the dataset's custom
  terms, with citation requirements, and the guestbook of item 29. Dataverse cites it as
  Zhang, Zhengwu; Xiang, Yang, 2026, "Continuous structural and functional connectomes of
  946 HCP Young Adult subjects on the fsaverage ico4 cortical grid",
  https://doi.org/10.15139/S3/NDAJYX, UNC Dataverse, V1.

### The terms and the guestbook, on the demo server

A draft never shows Dataverse's download window (its `FileUtil.popupDueToStateOrTerms`: "if
version is draft then Popup not required"), so the terms and the guestbook were tried on a
published test dataset on demo-dataverse.rdmc.unc.edu, with the same terms, the same
guestbook and a dummy file, on 14 October:

- In a browser, signed out, the download window showed the HCP's terms and the required
  "I agree"; the guestbook recorded the answer.
- Through the API, on 6.10.1, a download without the answer is refused, 400, naming the
  guestbook. A POST of the answer returns a signed address, but one that starts `http://`;
  after the proxy's redirect to https the server refuses it, 401 "Bad signed URL". The
  route for several files, a POST to `/api/access/datafiles`, returns a zip, but zips are
  capped at 1 GB on production and staging (100 MB on demo), and 38 of the 40 archives are
  2.0 to 2.5 GB. Both went into the reply to the archivist.
- Production runs 6.8, whose API asks for no guestbook: an anonymous GET of a file redirects
  to the object store, which serves byte ranges (206, with `Content-Range`).

### The index

`tools/dataverse_deposit.py index DEPOSIT --dataset ... --version 1.0 --out
src/sbci/data/hcp_ya.json` writes the package's index of the published version from the
deposit itself, so nothing is downloaded to make it. It asks the repository, with no token,
for the version, refusing one that is not published, then for its files and their citation,
and refuses an archive the version does not hold with the deposit's size and checksum (of
the type the repository registered, MD5 here). Each file's offset comes from its archive's
own tar header, and must agree with the archive and size `MANIFEST.tsv` records; sex and age
band come from `subjects.csv`; a subject is the tutorial's when its archive is. The index
carries the dataset, the version, its publication time, the citation, the terms' address,
the published `DATA_USE.txt` as the terms, each archive's file id, size and SHA-256, and
each subject's two files, one line a subject, so a later version reads as a diff.
`prepare` compares only the tutorial's files with the released copies, as before.

### The download

`sbci.download` reads Dataverse instead of Drive, with the standard library alone:

- **One subject is its own bytes.** A file comes as a range request for its bytes inside
  its archive, through the repository's redirect to the store, which must answer 206 with
  the range asked for; one subject is 90 MB, whichever archive holds it. A transfer cut part
  way is taken up from the bytes that came, up to four tries, whether the cut raises
  `IncompleteRead`, resets the connection, drops TLS or just ends the response early; after
  the fourth the error says to run the download again, which takes up the `.part` left.
- **An archive wanted whole is read whole,** as one stream, its files split off, verified
  and moved into place as they pass: the tutorial's two archives by default, every archive
  with `--all`. If the stream breaks, or ends early, the files it gave are kept and the rest
  come by range in the same run, the one it cut short from the bytes of it that came.
- **The guestbook, when a server asks.** On a 400 naming a guestbook the package reads the
  guestbook, answers each required question "I agree", refuses one that asks for name,
  email, institution or position (download those in a browser), and uses the address the
  server signs; an address that has lapsed is signed anew once, and a signed address
  refused at once is said to be the server's refusal. Other refusals (404, 403) are said at
  once, not retried as broken connections.
- **The terms, once a machine.** `fetch_cohort(accept_terms=False)` raises
  `TermsNotAccepted`, naming the terms and the two ways to accept them, until they are
  accepted; `accept_terms()` records the SHA-256 of the index's terms text, the terms'
  address, the dataset and the time in `~/.config/sbci/accepted_terms.json`
  (`$XDG_CONFIG_HOME/sbci`, or `$SBCI_CONFIG_DIR`), and new terms are asked for again.
  `sbci download` checks what is asked for first, so a wrong id is not met with the terms,
  then in a terminal shows a summary of the terms with their address and asks for
  `I agree`; with no terminal it stops, saying to run it in one or to add `--accept-terms`.
- **What is asked for.** No subjects means the tutorial's eleven; ids may be bare or
  `sub-` ids, strings or numbers; `"all"` (`--all`) is every subject. `DATA_USE.txt` beside
  the files ends with the dataset's address, its version and its citation, and
  `manifest.csv` lists every subject with a file there.

The Drive transfer and its tests are gone. CI's five-minute start, `scripts/check_hcp_ya.py`
(given `--accept-terms`) and `scripts/audit_api.py` download from Dataverse with
`--accept-terms`. README, USAGE (*Getting the data*, and the terms in the table of
errors), SPEC_QUESTIONS item 6, BLUEPRINT, VERIFICATION, docs/RESULTS.md and the figures'
and tools' READMEs describe the release, and the README cites the dataset.

### Tests

`tests/test_download.py` is rewritten, 30 tests. Its transfers go to a repository on the
machine serving tar archives the test builds as the deposit builds them: byte ranges
through a redirect, whole archives, a server that ignores ranges, cuts part way (taken up
where they stopped, whether the response ends early or raises `IncompleteRead`; every try
cut, an error and a kept `.part`; an archive's stream cut inside its second file, or
cleanly after its first, the rest by range), the 6.10 guestbook (answered "I agree", a
signed address used, a lapsed one signed anew, a guestbook asking for an email refused, a
signed address refused), and a 404 said at once. Its terms tests keep the record in a folder of their own: refused
before anything is fetched, accepted with the flag, not asked again, asked again for new
terms, asked at a terminal ("no" refused, " I agree " taken), and refused without one. The
shipped index is checked: 946 subjects with SC and FC each, eleven of the tutorial, six
women and five men, bands and no ages, offsets on 512-byte blocks inside their archives,
40 archives. `tests/test_dataverse_deposit.py` gains three: `index` against the local
stand-in for Dataverse, every file read back from its archive at its offset and fetched by
`sbci.download` from that index; `index` refusing an unknown or unpublished version, an
archive with another checksum and one the version does not hold; and `released_manifest`
taking the tutorial's 22 files from the shipped index. Seven mutants of the transfer and
the terms each fail the tests: no fallback to ranges after a stream, no retry, no resuming,
no signing anew, no check of the 206, every guestbook question answered, and an acceptance
taken whatever the terms now say.

### Verified

On Longleaf, from a frozen copy of dde7ca1 with this change laid over it (checked by MD5),
`index` wrote the index on a login node in 18 seconds: "946 subjects (11 of the tutorial),
1892 files in 40 archives of version 1.0 of doi:10.15139/S3/NDAJYX". Against the version's
public file list and the `SHA256SUMS` it publishes, every archive's file id, size and
SHA-256 agree, and every file's SHA-256; every offset is on a 512-byte block inside its
archive; and the tutorial's 22 files are, by the digests the Drive release's index
recorded, the files released on Drive. A subject is 75.6 to 100.5 MB.

From the same copy: the suite on the released subjects passes, 1,354 tests with 49
skipped, and as a fresh clone has it, with no data, 1,351 with 52 skipped (job 5123755,
an Intel Xeon Gold 6140 node); under Python 3.10 the download's and the deposit's tests
pass, 122 with one skipped, the commit test outside a checkout. `scripts/check_hcp_ya.py
--accept-terms` on the tutorial's eleven, linked from the lab's copy and found present and
verified against the index, passes every check, from the download to `reduce` and
`local_test` (job 5123899). `ruff check` and `ruff format --check` pass on src, tests,
scripts and tools.

From the published dataset itself, on 15 October, from macOS outside Longleaf:

- Without a terminal and without `--accept-terms`, `sbci download hcp-ya --subject` with the
  tutorial subject's id stops with exit status 1 and the message above, before fetching anything.
- Two subjects with `--subject` and `--accept-terms`, the tutorial subject and the last subject
  of `sc_19.tar`, fetched four files, 178 MB, as byte ranges in 7.5 seconds; the second's SC
  is the last file of `sc_19.tar`, 2.45 GB into it.
  `shasum -a 256 -c` against the dataset's `SHA256SUMS`: four OK.
- The default, `sbci download hcp-ya`, read the tutorial's two archives whole: 22 files,
  1.0 GB in 21 seconds, every one OK against `SHA256SUMS` and the Drive release's digests,
  with no `.part` left.
- In a terminal, typing `I agree` at the prompt recorded the acceptance, and the next run
  did not ask; `DATA_USE.txt` ended with the dataset's address, version and citation.
- On the files downloaded, the five-minute test passes, 3 of 3, and the download's tests,
  the shipped index's among them, 30 of 30.

CI passes at 2bedcd9, eight jobs of eight, its five-minute start downloading the tutorial subject from
the dataset on a machine none of us configured, with `--accept-terms`.

## 31. For newcomers, 16 October 2026 -- THE BASIS FETCHED WHEN FIRST NEEDED, THE 946 LISTED, USAGE ON THE PUBLIC DATA

Asked "anything still needed to improve the repo", the lead found no defect -- no open issue,
no broken link between the documents, no marker of unfinished work, CI passing, the README's
quick start running as written on the published data -- and listed what could be improved
and what waits on decisions. The user chose four: USAGE's examples on the public data, the
Laplace-Beltrami basis fetched when first needed, a listing of the 946, and CI's result in
item 30.

### The basis

The `rdk` and `matern` kernels need `EV_LBO_ds_ico4_L.mat` and `EV_LBO_ds_ico4_R.mat`, 50 MB
each, and SPEC_QUESTIONS.md item 8 had left open where they should come from: a newcomer had
to find them in SBCI_Toolkit. They are public there, on GitHub. At a558fe8, the toolkit's
head and the commit of the Longleaf clone the comparisons with MATLAB were run with, their
SHA-256 equal that clone's files', and GitHub's blob ids equal the bytes'.

- `sbci.download.fetch_basis(out=None, force=False)` reads the two files at that commit,
  by byte range, takes up a transfer cut part way, checks each against its SHA-256, and
  keeps files already there and correct. They go into `$SBCI_LBO_DIR`, else
  `~/.cache/sbci/lbo`, both of them places `find_basis` looks.
- `smooth(kernel="rdk" | "matern")` that finds no basis fetches it, saying so on stderr; if
  the fetch fails, the `MissingDataError` says why, how to fetch it where the network can
  be reached (`sbci basis`, on a cluster on a login node) and every place it looked.
- `sbci basis [--out DIR] [--force]` fetches it ahead of time, and names `SBCI_LBO_DIR`
  when the folder given is not one `smooth()` looks in.
- The download's resuming transfer is now one function, which the Dataverse files and the
  basis share; a plain address must answer a range with a 206 for that range, or a whole
  file only from byte 0, and more bytes than recorded are refused.
- `tests/conftest.py` replaces the fetch in every test with a refusal, so a test that
  reached `smooth(kernel="rdk")` without a basis fails instead of downloading 100 MB in CI;
  the fetch's own tests serve their files from the machine.

### The listing

`sbci download hcp-ya --list` prints the 946, one a line -- id, sex, age band, size in MB,
the tutorial's eleven marked -- and the total, 85.1 GB, or 47.2 GB with `--sc-only`; it
fetches nothing and needs no terms.

### USAGE on the public data

The examples that read the lab's copies of files now public read `./hcp-ya` instead, where
USAGE's first command puts them: *Loading*, the re-smoothing, the exchange file and reading
it back. The kernel maths runs on the reference's own example subject from a clone of the
toolkit, and the basis paragraph describes the fetch. The cohort check names the public
route, `sbci cohort hcp-ya-full --table hcp-ya-full/manifest.csv --modalities sc,fc`
(item 33 corrects the folder and adds `--modalities`, which defaults to `sc`; without it
the command checks the SC alone), which on the tutorial's download reports "11 subjects
seen; 11 in the cohort (sc, fc)". Eleven lines of 22 keep the
lab's paths, all of them what is not public or the lab's own setup: the Longleaf environment
(three), the lab's copy of the download (two), the lab's tables for the cohort example, whose
output is quoted (three), the pipeline's output (one) and the HCP's scans (two).

### Tests

`tests/test_download.py` gains eight, 38 in all: the listing, of a stand-in index and of the
shipped one, and of an unreleased cohort; and the basis fetched at its commit, verified and
kept, a file with another digest removed, a cut transfer taken up and a server without
ranges refused when taking up, `sbci basis` and its failures, and the guard itself.
`tests/test_smoothing.py`'s two tests of a missing basis now fail the fetch on purpose and
check that the message says why and how, and two more check that a basis found nowhere is
fetched once, giving the result the basis passed by hand gives, into a place `find_basis`
looks. Four mutants fail them: no digest check, a whole file taken when taking up, `smooth()`
never fetching, and the basis fetched where it is not looked for.

### Verified

On Longleaf, from a frozen copy of 2bedcd9 with this change laid over it (checked by MD5),
on an AMD EPYC 7763 node (job 5181381): the suite on the released subjects passes, 1,364
tests with 49 skipped, and as a fresh clone has it, with no data, 1,361 with 52 skipped;
under Python 3.10 the download's, the deposit's and the smoothing's tests pass, 215 with one
skipped, the commit test outside a checkout. In the same job a compute node fetched the
basis from GitHub itself, `sbci basis --out`, both files with their SHA-256. `ruff check` and
`ruff format --check` pass on src, tests, scripts and tools.

From macOS, against GitHub: `sbci basis --out` fetched the two files, 100 MB, in 3.6
seconds, each matching its SHA-256, and a second run kept them. With no basis anywhere (an
empty home, no `SBCI_LBO_DIR` or `SBCI_TOOLKIT`), `smooth(kernel="rdk",
mask_medial_wall=True)` on the tutorial subject as downloaded fetched it into `~/.cache/sbci/lbo` and
gave, bit for bit, what the same call gives with the basis passed by hand; the next call
found it there and fetched nothing. `sbci download hcp-ya --list` on the shipped index
lists 946 subjects, eleven marked, 85.1 GB, and 47.2 GB with `--sc-only`.

## 32. Head size in the cohort's model, 16 October 2026 -- SEX IN 3 COMPONENTS, NOT 13; THE RIGHT CUNEUS, NOT BOTH OCCIPITAL POLES

Asked to verify a reading of the literature on the paper's Figure 1D -- that the occipital
poles tell men from women -- the lead checked each claim against its source, with two
agents reading the papers, and against the analysis. Head size, which was not in the
model, was the most likely explanation, and the user asked for panel D redrawn with it in
the model and the repository brought into line.

### The literature

- No tractography or structural-connectivity study found names the occipital poles as
  where men and women differ. Moyer et al. 2017 (Medical Image Analysis), with the same
  per-vertex measure on 731 HCP subjects and intracranial volume in the model, found the
  left prefrontal and anterior cingulate cortex. Where occipital differences in
  structural connectivity are reported, most are higher in women (Duarte-Carvajalino et
  al. 2012, right cuneus clustering; Yan et al. 2011; Gong et al. 2009).
- Size-adjusted grey-matter volume gives partial support to a primary visual cortex
  larger in men: Liu et al. 2020 (976 HCP young adults, relative to total grey matter)
  and Lotze et al. 2019 (2,838 adults, BA 17 in small clusters, with the lateral
  occipital BA 19 larger in women).
- Ingalhalikar et al. 2014's whole-brain pattern follows brain size rather than sex
  (Hänggi et al. 2014), and connectome differences and sex prediction largely follow it
  too (Eliot et al. 2021).

### The model

The documents' cohort tests now hold head size: FreeSurfer's estimated intracranial
volume, `EstimatedTotalIntraCranialVol` from each subject's `aseg.stats` in the HCP's
open-access release, in litres -- the volume the coupling analysis already used, now in a
table of all 946 on Longleaf (`hcp-ya/head_size.csv`). It averages 1.708 L in men and
1.473 L in women (d = 1.59), and correlates with fluid intelligence at r = 0.25.

- `scripts/hcp_figures.py --full-cohort ... --covariates CSV` adds the table's columns to
  every test, the trait's and sex's, as the coupling analysis does; the titles name the
  covariates, and the trait figure orders its components by the unadjusted p-value, since
  Benjamini-Hochberg ties the adjusted ones.
- `tests/reference/inference_probe.py --head-size CSV` adds it to every design.
- docs/RESULTS.md, USAGE.md and docs/figures/README.md give the results of the model with
  head size, and those without it beside them; RESULTS.md no longer says only the eleven
  are distributed.

| | Without head size | With it |
| --- | --- | --- |
| Sex, rank 20 (FDR 0.05) | 13 of 20, the strongest at adjusted p 1e-7 | 3 (components 3, 5 and 8), each at 0.035 |
| Sex, rank 4 | all four | one (component 3) |
| The effect map | both occipital poles; peak in the right cuneus; left lateral occipital vertices to 0.90 of it; 340 vertices past -0.1, none past +0.1 | the right cuneus (region mean -0.66) and pericalcarine (-0.33), higher in men; a patch near the right occipital pole higher in women (to +0.28); 113 vertices past -0.1 and 20 past +0.1; the left hemisphere below 0.046 |
| Sex in vertex strength, the others as nuisance | 855 vertices, partial R-squared to 0.108 | 433, to 0.068 |
| Fluid intelligence, rank 20 | none; closest component 13 at 0.064, and at 0.024 counted as independent | none; components 12, 13 and 7 at 0.087, component 13 at 0.061 counted as independent |
| Fluid intelligence, rank 4 | smallest adjusted p 0.33 | 0.28 |

The strongest sex component is now component 3: estimate -0.0086 (95% interval -0.0146 to
-0.0027, 421 degrees of freedom), partial R-squared 0.010, the three running from 0.010 to
0.011. The age band as a factor shows in one component, and so does 31-35 against 26-30
(component 5, -0.0089, interval -0.0136 to -0.0042).

### How firm it is

Twelve subjects' volumes lie beyond three standard deviations of their sex's mean, eleven
below it (0.83 to 2.14 L in all); the lowest, 0.83 L, is implausible for an adult, so the
estimate has probably failed for some of them. With the logarithm of the volume in the
model, five components pass (3, 5, 8, 12 and 16; smallest adjusted p 0.026). Leaving the
twelve out, none does: components 3, 5 and 8 come to 0.099, 0.07 and 0.07. Over all twenty
components the fitted difference keeps its shape across the three versions (r = 0.999
with the logarithm and 0.993 without the twelve) and its peak in the right cuneus, at 0.78,
0.79 and 0.68 of the peak without head size. So the components put the residual sex
difference in the right medial occipital cortex, higher in men, at the edge of the
threshold. (That the fitted difference keeps its shape over three fits to nearly the same
subjects is to be expected, and says where the components place it, not where women and men
differ: item 33 tests every vertex, and finds sex mostly higher in women, bilateral and the
most in the insula.) The coupling analysis uses the same volumes, the twelve among them, and was not
rerun without them.

### The paper's Figure 1

Panel D, outside the repository, is redrawn from the same test with head size
(`fig1_data.py`; the earlier version kept as `fig1_data_unadjusted.py`, the earlier outputs
as `out-unadjusted`): 3 of 20 components, the peak at the right cuneus, the Desikan means
-0.661 for the right cuneus and -0.327 for the right pericalcarine cortex; at vertex
resolution the right superior parietal cortex reaches -0.912 and the right lateral
occipital cortex runs from -0.720 to +0.282, while every region of the left hemisphere
averages within 0.013 of zero.

### Verified

The numbers are the logs of jobs 5329082 (the figures, the probe and the first
sensitivity check), 5330262 (the sensitivity in detail), 5330299 (the figures redrawn with
their titles wrapped), 5329021 (panel D) and 5254506 (the first comparison), and of
`i33-independent.out` (fluid intelligence counted as independent), all on Longleaf under
`/work/users/x/y/xya/edits` and `paper-fig1`. Jobs 5254506 and 5330262 fit the model
without head size beside it and get the published 13 components, so the change is head
size alone. From a frozen copy of b0ca491 with this change laid over it (checked by MD5),
the suite passes on the released subjects, 1,364 tests with 49 skipped, and as a fresh
clone has it, 1,361 with 52 skipped (job 5354538, an AMD EPYC 7763 node); `ruff check` and
`ruff format --check` pass on src, tests, scripts and tools.

## 33. A claims review of the whole repository, 17 October 2026 -- FLUID INTELLIGENCE IN NO VERTEX ONCE THE COVARIATES ARE IN; SEX, VERTEX BY VERTEX, MOSTLY HIGHER IN WOMEN AND NOT WHERE THE EFFECT MAP PUTS IT; NO HCP ID BESIDE A FAMILY-BASED RESULT

Asked to check the whole repository as panel D had been checked (item 32), every claim
against its source, the data and the recorded runs, five reviewers read a copy of cd2a343:
the literature, the statistics, the documents against one another, the documents against
the code, and every number against the Longleaf logs. The lead checked each finding before
reporting it, with a run of its own wherever a finding turned on a number, and the user
asked for all of it to be fixed.

### The HCP's restricted data

The HCP's Restricted Data Use Terms for the Young Adult study, read on the HCP's site,
allow no HCP subject id in a publication or public report of an analysis that uses
restricted data, and family structure is restricted. RESULTS.md, this file and
docs/figures/README.md reported tests with the families as clusters and named the tutorial
subject, and four others, by their HCP ids; and item 6 reported a test of the released
subjects, who are named, with their families as clusters.

- The two are now kept apart page by page. A page that reports a result the family table
  went into names no subject by HCP id: the tutorial subject, and B to E for four others
  here, as the paragraph at the head of this file says. USAGE.md and VERIFICATION.md, whose
  commands name released subjects, report none of those results and point to RESULTS.md,
  where the test-retest and prediction findings moved from USAGE.md. Item 6's sentence on
  that test is gone.
- Counts drawn from the table stay where no subject is named (the 946 in 423 families, the
  943 in 422); RESULTS.md now says only counts appear, not that nothing from the table is in
  the package. `src/sbci/stats.py` no longer gives the largest family's size, and
  `tools/clustered_null.py` simulates 400 families rather than the analysis's own number.
- The notebooks name no subject. The paper's Figure 1 notes, outside the repository, call
  panel B's subject "subject A", and say the paper should do the same.
- The commits before this one held that sentence and the ids beside family-based results;
  the history is collapsed since (*The history*, below).

### Fluid intelligence, vertex by vertex

RESULTS.md said fluid intelligence "is in the data", at 113 vertices: the test of
`tools/age_probe.py`, which holds the score alone. With the covariates of the component
test (job r5-vertex-tests-5361031, area-weighted strength of the 943, families as
clusters):

| Model | Vertices at FDR 0.05 | Largest partial R-squared |
| --- | --- | --- |
| the score alone | 113 | 0.027 |
| given sex, age band and streamline count | 26 | 0.022 |
| and intracranial volume | 0 | 0.019 |
| the same, without the twelve implausible volumes (931) | 0 | 0.018 |

The components' verdict stands but lies near the threshold: no component passes among the
943 (components 12, 13 and 7 at 0.087), and among the 913 with a resting-state measure of
head motion, on the same fit, one passes at 0.046, none once motion is in the model (0.062;
job r5-motion-5361157). RESULTS.md says all of it.

### Sex, vertex by vertex

| Model (strength of the 943, families as clusters) | Vertices at FDR 0.05 | Largest partial R-squared |
| --- | --- | --- |
| sex alone | 1,020 | 0.094 |
| given the score, age band and streamline count | 855 | 0.108 |
| and intracranial volume | 433 | 0.068 |
| with its logarithm instead | 440 | 0.072 |
| without the twelve implausible volumes (931) | 420 | 0.064 |
| the 913 with motion: head size | 411 | |
| the 913 with motion: head size and motion | 402 | |

The count is firm; where it lies is not where the effect map puts it. Of the 433, 214 are
on the left and 219 on the right, and 327 have connectivity higher in women, 106 in men.
The most are in the insula (57 vertices on the right, 33 on the left), then the left
superior frontal and fusiform cortex (21 each), the left superior parietal and right
lateral orbitofrontal (17), the left pars opercularis (16), both parahippocampal gyri, the
right fusiform and the right lateral occipital cortex (14 each) and the left supramarginal
(12). The effect map over the three components that pass (RESULTS.md, and the paper's
panel D) peaks in
the right cuneus, higher in men, with nothing on the left: it describes those components,
`(basis * weights) @ basis.sum(axis=0)` over them, not where women and men differ. Item
32's "a stable place" read too much into the map's keeping its shape across three fits to
nearly the same subjects, which it would do whatever the signal. RESULTS.md, the figure's
title and caption, docs/figures/README.md and item 32 now say what the map describes;
whether panel D becomes a map of the vertex test, or the coupling difference, is the
user's. Among the 913, the components test gives four with head size (3, 5, 8 and 12;
smallest adjusted p 0.047) and five with motion as well (and 16; 0.048).

Coupling's sex difference holds however the volume enters (job r5-coupling-icv-5360139;
900 subjects): 325 vertices as published, 253 higher in men; 349 (271) with the logarithm
of the volume; 279 (219) without the nine implausible volumes among the 900; 1,104 (911)
without the volume. Item 32 had left this check open.

### Against the literature

- **ICA-FIX** (USAGE.md, *What the FC is*). FIX had been said to remove the white-matter
  and CSF artefacts the pipeline regresses; it regresses 24 motion parameters and removes
  the components it classes as noise, and no white-matter or CSF signal (Smith et al.,
  NeuroImage 2013), leaving some motion and global artefact (Burgess et al., Brain
  Connectivity 2016), while the pipeline regresses six motion parameters, five components
  each of white matter and CSF and the global signal (Cole et al. 2021). USAGE now says the
  FC is near the pipeline's, not the same.
- **ENCORE's warp** was called an anatomical correspondence that applies to any map of
  the subject. It is a correspondence of connectivity that can be carried to other maps;
  the migration figure tests the carrying, with one warp planted in both the endpoints and
  the map, and whether it aligns myelin or function better than MSMAll is untested.
- **ConSEAL's defaults** were all "the paper's". The step and the threshold are; the
  iteration cap and the kernel width (0.0027, as narrow as the fork's kernel at the paper's
  0.005) are the authors' code's. USAGE.md and RESULTS.md say which is which, and that the
  default was the closest on 18 of the 24 known warps and tied with the closest on the six
  largest, not "on each".
- **Smoothing.** "Half A matches the map from all streamlines at r = 1.00" correlated a set
  with a whole that holds it, which two halves at 0.99 make about 0.9975 by construction.
  The figure and RESULTS.md now give the same vertex in the other ten subjects instead:
  r = 0.54 to 0.80 (median 0.70), against 0.99 between the halves.
- **The trait figure** paired the plotted, unadjusted r with the test's adjusted p. Each
  title now gives the partial r given the covariates beside it: -0.09 for component 12
  and 0.09 for component 13, where the plotted r are -0.06 and 0.11.
- **Mean coupling**'s p 0.036 against the score is one of six such tests, uncorrected;
  RESULTS.md says so.
- **The coupling gradient.** The studies cited measure coupling otherwise and in other
  samples, so only the ordering compares; and the entorhinal cortex, lowest here, is where
  a vertex's FC strength repeats least between days (ICC 0.34 and 0.35,
  `notebooks/reliability.ipynb`).
- **Citations and terms.** The Riemannian diffusion kernel is published (Wang, Li and
  Zhang, Imaging Neuroscience 3, 2025, doi:10.1162/imag.a.912); README.md, BLUEPRINT.md,
  SPEC_QUESTIONS.md and `smoothing.py` cite it. The grid is the icosahedron subdivided four
  times, fsaverage4, not fsaverage subdivided. Rank 20's 27% of the norm is about 7% of the
  sum of squares, which RESULTS.md now adds.

### Instructions that did not work as written

- The cohort examples read `hcp-ya-full/`, which no documented download wrote. The
  downloads of the 946 now write it (`--out hcp-ya-full`), in README.md, USAGE.md,
  RESULTS.md and docs/figures/README.md, which also says the coupling run needs the FC.
- The public cohort check lacked `--modalities sc,fc`, so it checked the SC alone, and item
  31 quoted it as "(sc, fc)". With the flag, on the tutorial's download, it reports "11
  subjects seen; 11 in the cohort (sc, fc)"; without it, "(sc)".
- VERIFICATION.md ran `check_hcp_ya.py` without `--accept-terms`, which stops on a machine
  that has not accepted the HCP's terms, and called `verify_all.sh` "tiers 1 and 2" where it
  runs 1, 2 and 4, as did the script's own header.
- USAGE.md still documented `template=2`, registering onto a subject by its index in the
  list, the form the user rejected; the documents give the fixed subject itself and an
  array, and the code's acceptance of an index is unchanged.
- USAGE.md's errors. `SbciError` covers connectome files and atlas and region names, not
  every input: a missing path raises `FileNotFoundError` and a malformed table or eigenpair
  file `ValueError`, so the example now catches `OSError` too. `UnknownAtlasError`,
  `TermsNotAccepted` (a `RuntimeError`) and `ConSEAL` are named by their modules, since
  `sbci` does not carry them, and "where it is tracked" is gone. The two smoothing errors
  that sent a user to `Endpoints.from_matlab`, whose endpoints do not re-smooth to the
  pipeline's connectome, name `Endpoints.from_snapped`.
- USAGE.md now names `$XDG_CONFIG_HOME` where the terms' acceptance is kept, as
  `download.config_dir` does; says that composing a subject's own MSMAll sphere pair with a
  carried warp is left to the user, with the calls to do it; and says the test-retest
  example needs each day's FC, which the release does not hold.

### Stale text

BLUEPRINT.md said eleven subjects are public, listed neither the modules added since nor
`sbci basis`, and put the suite "past six hundred". notebooks/README.md said only the
prediction notebook reads the family table; the reliability notebook does too.
SPEC_QUESTIONS.md item 6 described the Drive release in the present tense. Items 5, 6, 7,
20, 31 and 32 here, the review note of 5 October, tools/README.md, the docstrings of
`tools/dataverse_deposit.py`, `tools/encore_probe.py`, the `ConSEAL` class (the rotation
search scores on the GPU since item 28), `ContinuousConnectome.smooth` (the basis is
fetched) and `download.config_dir` (`$XDG_CONFIG_HOME`) now say what is so. Two numbers
now carry their records' precision: the exchange file's mass is conserved to 1.1e-10, which
"1e-10" understated, and the known warp's cost falls to 0.155 of its start, which "0.15"
rounded.

### Numbers rerun with their logs

The number trace found 72 of 88 statements traced to a log, one mismatch of wording
(USAGE.md's "the closest, component 13", a three-way tie, removed with the paragraph) and 15
with no saved run, or a run that bounds the value without printing it. Each is rerun here
with its log kept (`r6-numbers-5408060.out`, on an Intel Xeon Gold 6140 node; the figures'
numbers from `r6-figures-5408061.out`, and the coupling maps' last digits from
`r6-digits.out`, all under `/work/users/x/y/xya/edits`; the scripts beside them in `r6/`, and
`tests/reference/tier4_numbers.py`, new here, prints the agreements Tier 4 bounds):

| Statement | As documented | Rerun |
| --- | --- | --- |
| the family table's counts | the 946 in 423 families, 850 with a relative among them; the 943 in 422 | the same; 847 of the 943 with a relative |
| the effect map's left hemisphere (panel D) | nothing above a twentieth of the peak, below 0.046 | at most 0.046 |
| coupling's sex vertices by the medial wall | 253 higher in men, 72 in women, 44% of those within two grid rings of the wall, where 6% of cortex lies | the same (`hcp-ya/fc/sex_edge_check.py`) |
| the eleven's coupling by region | 0.36 to 0.41 in visual cortex, 0.05 to 0.09 in the cingulate and entorhinal cortex, 0.30 over unimodal regions against 0.19; coloured 0.05 to 0.42 | 0.363 to 0.412, 0.055 to 0.092, 0.299 against 0.190; 0.052 to 0.415 |
| the eleven's coupling, subject by subject | global 0.200 to 0.268, within regions 0.595 to 0.679 | the same |
| one subject's SC with another's FC | 0.225 against 0.233; its own FC fits better in 60% of pairs | the same, of 110 ordered pairs |
| the cohort's mean coupling map | r = 0.972 with the eleven's; coloured 0.05 to 0.40 | 0.9715; 0.051 to 0.405 |
| the warp carried to fsaverage | 0.32 degrees from the true one on the left, 0.33 over both | 0.324 and 0.33 (0.343 on the right) |
| the known warp on the grid | 1.59 degrees on average, 3.98 at most | 1.591 and 3.983 |
| vertex 1234's streamlines | more streamlines end at 85% of the cortical vertices | 84.8% |
| the tutorial subject's Desikan-level coupling | 0.364 | 0.364 |
| the FPCA against MATLAB | every component to 1e-15 at the fixed point; at the default tolerance the first vector to 2.3e-16 | basis 1.0e-15, scores 6.1e-16, scales 1.2e-15; 2.32e-16, the later vectors 9.8e-10 to 5.6e-5 |
| ENCORE behind the 1e-10 finite difference | 4.9e-4, 7.1e-4 and 5.4e-4 | 4.92e-4, 7.06e-4 and 5.35e-4 |
| the parcellation and the coupling against MATLAB | 1.875e-16; 3.6 to 88.2 eps | 1.875e-16; 3.6 (within regions), 9.9 (global) and 88.2 eps (discrete) |
| the clustered null | ignoring the families 16%; clustered 9.7%, 6.7%, 6.0% and 5.1% for 20, 50, 100 and 422 families | 16.6%, 16.1%, 15.6% and 16.8%; 9.7%, 6.7%, 6.0% and, for 400 families, 5.6% |
| negative FC | discarding it changes the sign of the coupling map in association cortex | not so: with it kept, 0.5% to 1.7% of a subject's cortical vertices are negative; with it set to 0, the map correlates with the kept one at r = 0.88 to 0.96 (0.885 to 0.957; item 40) and unimodal regions stay above association regions (0.38 against 0.30) in all eleven |
| the eleven's streamlines | 734,039 to 988,788 | the same |
| `import sbci` | about 300 ms | 341 to 350 ms with the interpreter's own start |
| the default ConSEAL against the other methods | the closest of every method on each of 24 known warps | the closest on 18; on the other six, the two largest warps' (4.9 and 6.5 degrees), the same kernel without the floor comes out as close (0.049 and 0.062 degrees on average, for both) |

USAGE.md, the coupling module's docstring and this item now say what was measured for
negative FC; every other value stands, and the clustered null's rates are those of 400
families. The ConSEAL record is `conseal-fork-compare/synth-evaluate-4550924.out`, read
again here.

### Verified

The findings rest on jobs r5-vertex-tests-5361031 (the vertex tests of the trait and of
sex), r5-coupling-icv-5360139 (coupling's sex difference by how the volume enters) and
r5-motion-5361157 (head motion), and the reruns on those named above, all under
`/work/users/x/y/xya/edits`. The three figures were redrawn from the frozen copy below (job
5408061) and looked at before they went in. From a frozen copy of cd2a343 with this change
laid over it (checked by MD5), the suite passes on the released subjects, 1,364 tests with
49 skipped, and as a fresh clone has it, 1,361 with 52 skipped, as before the change (job
5408062, an AMD EPYC 7702 node, nine to ten minutes each), and again on the final copy,
its two docstrings included (job 5408923, the same counts in ten to eleven minutes); `ruff
check` and `ruff format --check` pass on src, tests, scripts and tools.

### The history

At the user's request, on 17 October 2026, the repository's history was collapsed into one
commit, the first of the history since: the commits before it held material this item
removes from the documents. Those 105 commits are kept privately, as a git bundle on
Longleaf, so nothing of them is lost, and the commit ids these notes cite name commits of
that history. Version 1.0 of the dataset on UNC Dataverse installs the package at one of
them, 53eacc0; a later version's README would name the commit that holds the code now.

## 34. The paper's panel D from the test at every vertex, 17 October 2026 -- 433 VERTICES BESIDE 15 REGIONS; NOT HEAD SIZE LEFT OVER, IN WOMEN AND MEN MATCHED ON IT

The review of item 33 found that the effect map over three FPCA components, which the
paper's Figure 1D showed, describes those components and not where women and men differ.
Asked for a panel D drawn from the test at every vertex, with the regional test beside it
and the literature checked, the lead drew it outside the repository (the paper's figure)
and ran the checks the literature uses for head size. This item records the numbers.

### The test at every vertex and at every region

The vertex test is item 33's, as docs/RESULTS.md reports it: each cortical vertex's
connectivity strength among the 943 (`probe_pmat24.npz`, from `tools/age_probe.py`), sex
given fluid intelligence, age band, streamline count and intracranial volume, families as
clusters, FDR 0.05. It finds 433 of 4,685 vertices, 214 on the left and 219 on the right,
327 higher in women, with partial correlations from 0.089 to 0.261 in magnitude. The same
test on the 68 Desikan-Killiany regions, each region's strength the area-weighted mean of
its vertices', finds 15 regions, 9 higher in women and 6 in men, the strongest the right
insula (partial r +0.231), then the right parahippocampal gyrus (+0.140) and the left
insula (+0.133); 15 with the logarithm of the volume, 12 without the twelve implausible
volumes (440 and 420 vertices).

| What the regional test does with the vertices' differences | Regions |
| --- | --- |
| misses them, where 5 or more of a region's vertices differ | 16, among them the left fusiform (21 vertices) and the left superior parietal (17) |
| averages over vertices that differ in both directions | 9: both lingual gyri, both precentral gyri, both superior temporal gyri, the left postcentral, the right fusiform and the right pars opercularis |
| finds an effect few of the region's vertices show | the right inferior parietal (3 of 149 vertices), the left banks of the superior temporal sulcus (1 of 35), the right cuneus (6 of 25) and the right frontal pole (4 of 6) |

### Head size, the way the literature checks it

The studies that compared women and men of matched brain size are the literature's caution
on such a result, so the checks they use were run (`r7_sex_size.py`):

| Model | Vertices at FDR 0.05 | Higher in women | In the insula |
| --- | --- | --- | --- |
| intracranial volume linear, as drawn | 433 | 327 | 90 |
| intracranial volume cubic | 411 | 318 | 93 |
| 187 women and 187 men matched on it, 1.589 L on average in both | 186 | 152 | 66 |
| without streamline count, 5.5% lower in women (r = -0.32 with being a woman) | 485 | 297 | |
| without fluid intelligence | 427 | 324 | |
| without either | 489 | 302 | |

- In the matched sample every one of the 433 vertices keeps its sign, its partial r
  correlates with the whole sample's at 0.984 over them (0.933 over the cortex), and the
  effects grow: a median magnitude of 0.155 against 0.117. Twenty samples of the same size
  drawn without matching give 0.101 to 0.134, so the growth is not the smaller sample's.
  If the difference were head size left over, matching would shrink it.
- Within each sex, strength at the women-higher vertices does not fall with head size
  (median partial r +0.021 among women, +0.009 among men), so women's smaller heads do not
  explain why they are higher there. At the men-higher vertices it falls (-0.042 and
  -0.046), against the direction a size effect would give men.
- Without streamline count the map keeps its shape (r = 0.92 with the model drawn), and the
  women-higher share falls from 76% to 61%: the majority holds, the balance depends on that
  covariate. Fluid intelligence changes nothing (r = 1.000).

### The literature

Read for this item, by an agent and then by the user's own review of it, which corrected two
of the agent's readings:
- Moyer et al. 2017 (Medical Image Analysis) is the closest comparison: the same per-vertex
  measure on 731 HCP subjects with intracranial volume in the model, left prefrontal and
  anterior cingulate clusters higher in women. It draws on the same cohort.
- Szalkai et al. 2018 (Brain Imaging and Behavior) supports the matched result: 36 women with
  larger brains than any of 36 men keep, in its abstract's words, "the significant, deep
  connectivity advantages" in numerous parameters. The agent had listed it as a conflict.
- Hänggi et al. 2014 is about the balance of within- and between-hemisphere connectivity,
  which follows brain size; it does not test total strength. The agent had listed it as a
  conflict too.
- Eliot et al. 2021 find no consistent structural pattern across studies.
- Grey matter: Ruigrok et al. 2014 and Lotze et al. 2019 disagree on the insula, and Lotze
  et al. 2019, Liu et al. 2020 and Williams et al. 2021 find the fusiform and
  parahippocampal gyri larger in men, where the connectivity here is higher in women.
- Gong et al. 2009 and Duarte-Carvajalino et al. 2012 could not be read in full, and their
  abstracts name neither the insula nor how brain size was handled, so they are not cited.

### Verified

The numbers are the logs of jobs 5424182 (the vertex and regional tests, region by region,
drawn at d7fd23c) and 5426385 (the head-size and covariate checks), in
`/work/users/x/y/xya/paper-fig1/` with their scripts (`fig1_data_vertex.py`,
`fig1_render_vertex.py`, `fig1_compose_vertex.py`, `r7_sex_size.py`). The vertex test
reproduces item 33's 433 vertices, 327 higher in women. Only documents change here.

## 35. The overlap of bundles after alignment, 18 October 2026 -- CONNECTION-WISE, AS THE CONSEAL PAPER MEASURES IT; BOTH METHODS RAISE IT, CONSEAL THE MORE OUTSIDE THE ANTERIOR CINGULUM

Asked whether the overlap of fibre bundles had been measured after alignment, the lead
measured it with the package's own registrations. The first answer used the vertex-wise
overlap coefficient: `overlap_coefficient`, ported from the public `get_overlap_coefficient.m`
(item 7), whose header calls it connectivity-level though its code counts vertices. The user
pointed out that the ConSEAL paper's coefficient is connection-wise: `get_connectivity_dice.m`
in the author's evaluation folder, as `Encore/experiments/run.m` calls it, with threshold 0
and `use_symmetricDice` false. Both are now in the package and checked against the MATLAB.

### What was added

`connection_overlap(first, second, threshold=0.0)` and `connection_dice` in `sbci.conseal`.
- **The set.** Each streamline gives the vertex pairs of its two endpoints' triangles the
  products of their barycentric weights. The matrix is symmetrized and divided by the
  number of streamlines: `EndpointConnectome.adjacency`, the matrix ConSEAL smooths. The
  pairs above the threshold form the set, each pair counted in both orders, as in the
  MATLAB's full matrix.
- **The two measures.** The overlap coefficient is `|X & Y| / min(|X|, |Y|)`, and Dice is
  `2 |X & Y| / (|X| + |Y|)`.
- **Empty sets.** Two empty sets score 1 and one empty set 0, as the MATLAB has it.
- **Threshold.** A negative or non-finite threshold is refused.
- **The vertex-wise functions.** `overlap_coefficient` and `dice_score` now say they are
  vertex-wise.

Tests:
- **`tests/test_conseal.py`**, three new tests, and the identity test extended to both
  measures:
  - streamlines from vertex to vertex, whose pairs can be counted by hand;
  - an independent count from the dense adjacency at three thresholds;
  - the empty sets and the refused thresholds, in one test.
- **`tests/test_matlab_reference.py`**, two, on two subjects' MdLF from the released files
  (`tests/reference/overlap_inputs.py`, `overlap_reference.m`):
  - vertex-wise to 12 decimals at three thresholds, the endpoint masses to 2.4e-6 of the
    largest, on the same vertices;
  - connection-wise exactly: 11,290 and 14,416 pairs, 1,670 shared, overlap coefficient
    0.147918511957, Dice 0.129930755466.

### The analysis

**Setup:** the lab's ENCORE bundle evaluation
(`/overflow/zzhanglab/encore_project/encore_dice_scores/get_dice_scores.m`):
- **Bundles.** RecoBundles, left and right pooled: MdLF, ILF, SLF, FAT and the anterior
  cingulum.
- **Subjects.** 50 subjects, the 2nd to 51st of its sorted list. `run.m` takes the first
  50, so it shares 49 of them and adds one.
- **Streamlines.** A released SC file keeps, in order, the streamlines of
  `snapped_fibers.npz` with both ends on the white surfaces. These are exactly the streamlines
  the lab's intersection files list, so a bundle is the SC streamlines RecoBundles listed.
- **Registrations.** Both methods at their defaults, onto templates of the 50, on the CPU.
  ENCORE took 8–46 steps (cost down 14–34%). ConSEAL took 92–487 iterations (cost down
  20–44%), refusing folding steps in six subjects.
- **Endpoints after ENCORE.** They go through the inverse warp, since USAGE says ENCORE's warp
  is a pull-back. Re-smoothed, they reproduce ENCORE's own registered density at r = 0.993,
  against 0.95 unmoved and 0.87 through the warp itself.
- **Endpoints after ConSEAL.** Taken as ConSEAL moved them.

The connection-wise overlap coefficient, threshold 0, mean over the 1,225 pairs. The 95%
interval of each change comes from resampling subjects.

| Bundle | no alignment | ENCORE | ConSEAL | ENCORE - none | ConSEAL - none | ConSEAL - ENCORE |
| --- | --- | --- | --- | --- | --- | --- |
| MdLF | 0.236 | 0.239 | 0.252 | +0.003 (-0.004, 0.010) | +0.016 (0.007, 0.025) | +0.013 (0.007, 0.019) |
| ILF | 0.506 | 0.528 | 0.530 | +0.021 (0.015, 0.028) | +0.024 (0.014, 0.033) | +0.003 (-0.003, 0.008) |
| SLF | 0.310 | 0.329 | 0.354 | +0.019 (0.015, 0.025) | +0.043 (0.037, 0.050) | +0.024 (0.020, 0.028) |
| FAT | 0.486 | 0.529 | 0.535 | +0.043 (0.029, 0.058) | +0.050 (0.032, 0.068) | +0.007 (0.000, 0.013) |
| anterior cingulum | 0.307 | 0.315 | 0.314 | +0.008 (0.002, 0.015) | +0.007 (-0.001, 0.015) | -0.002 (-0.007, 0.003) |

`run.m`'s own 50 are the 49 shared and one more. That subject was registered onto the same two
templates (ENCORE in 14 steps, ConSEAL in 383 iterations; job 5588909). On those 50 the
coefficient moves by at most 0.004 (job 5602148):

| Bundle | no alignment | ENCORE | ConSEAL | ENCORE - none | ConSEAL - none | ConSEAL - ENCORE |
| --- | --- | --- | --- | --- | --- | --- |
| MdLF | 0.236 | 0.239 | 0.252 | +0.003 (-0.004, 0.010) | +0.016 (0.007, 0.026) | +0.013 (0.008, 0.019) |
| ILF | 0.504 | 0.525 | 0.529 | +0.021 (0.015, 0.028) | +0.025 (0.015, 0.034) | +0.003 (-0.002, 0.009) |
| SLF | 0.312 | 0.332 | 0.358 | +0.020 (0.015, 0.026) | +0.045 (0.038, 0.053) | +0.025 (0.021, 0.029) |
| FAT | 0.485 | 0.528 | 0.535 | +0.043 (0.030, 0.058) | +0.050 (0.033, 0.069) | +0.007 (0.000, 0.013) |
| anterior cingulum | 0.304 | 0.312 | 0.311 | +0.008 (0.002, 0.015) | +0.007 (-0.001, 0.015) | -0.001 (-0.006, 0.004) |

What the tables show:
- **Both methods raise the overlap.** ConSEAL raises it in 58% (anterior cingulum) to 95%
  (SLF) of pairs, and ENCORE in 56% to 85% (57% to 85% on `run.m`'s 50).
- **ConSEAL is the more for three bundles.** It leads ENCORE significantly for MdLF, SLF and
  FAT. ILF is level.
- **The anterior cingulum is the exception.** The two are level there and neither gains
  much, as the ConSEAL paper reports.
- **Dice and the vertex-wise measures.** Connection-wise Dice tells the same story. The
  vertex-wise measures rise less, by 0.1 to 1.4 points: Dice from 0.623 to 0.629 (ENCORE)
  and 0.632 (ConSEAL) for MdLF, from 0.747 to 0.753 and 0.759 for SLF. On them ENCORE leads
  ConSEAL for the anterior cingulum: vertex-wise Dice 0.741 against 0.734, a difference of
  -0.007 (-0.012, -0.003); on `run.m`'s 50, 0.734 against 0.728.

### Verified

**Jobs** (the scripts in `/work/users/x/y/xya/paper-open-items/dice/scripts/`):
- 5563502: the vertex-wise check;
- 5588961: the connection-wise check and the lab's 50;
- 5566179: the vertex-wise measures;
- 5588909: `run.m`'s extra subject registered;
- 5602148: `run.m`'s 50, both measures;
- 5600812: the references regenerated, the new tests, ruff.

**The suite with the change** (job 5608993, an AMD EPYC 7713 node):
- 1,370 tests pass with 48 skipped on the released subjects;
- 1,367 pass with 51 skipped as a fresh clone has it;
- the difference from item 34's counts is the five new tests, and one test that item 34's
  runs skipped and that runs today. 7bf26e8 skips exactly the same tests in today's
  environment (job 5619247);
- `ruff check` and `ruff format --check` pass.

Only the two functions, their tests and the documents change here.

## 36. ConSEAL's search of the grid in threads, 19 October 2026 -- THE SAME BITS, A REGISTRATION ABOUT TWICE AS FAST ON EIGHT CORES

Asked to check the times the documents gave for ConSEAL, which another session had found too
short on the CPU and on the GPU, and then whether the CPU could be made faster, checking
first, the lead found the times right for what they timed but short of a registration at the
defaults, and the CPU's eight cores nearly idle. The search of the grid now runs in threads.

### The times the documents gave

They held for what they timed, but read as a registration's time:
- **Thirty iterations.** USAGE.md's *A registration, subject by subject* timed the cohort
  figure's registrations, thirty iterations each: 5.8 minutes on eight cores, 35.5 s on a
  V100.
- **Known warps.** The ConSEAL notes' 76 to 155 iterations, 15 to 42 minutes on eight cores,
  undid known warps (item 19), which takes fewer iterations than registering real subjects.
- **The docstring.** `endpoints_align` said two whole subjects take tens of minutes on eight
  cores.

At the defaults a registration runs until its cost stops falling. 148 whole registrations on
eight cores, with the code before this item, took:

| | runs | iterations | time |
| --- | --- | --- | --- |
| onto a template (item 35's) | 50 | 92 to 487, median 253 | 25 min to 2.6 h, median 70 min |
| onto one subject (item 35's) | 49 | 62 to 396, median 117 | 12 min to 2.4 h, median 28 min |
| onto one subject (the conseal-ot project's, at 86e51f4) | 49 | 49 to 489, median 147 | 11 min to 3.5 h, median 49 min |

An iteration took 9 to 37 s, by the node: 12 s at the median on an AMD EPYC 9654, 23 s on
an EPYC 7702. A registration held 3.3 to 3.6 GiB. On a V100 the three registrations onto one
subject with the fewest, the median and the most iterations on the CPU took the CPU's
iterations, refused steps and costs to six decimals, at 1.15 to 1.21 s an iteration: 62
iterations in 75 s against 837 s on eight cores, 117 in 138 s against 1,650 s, and 396 in
457 s against 8,790 s (job 5678572).

### Where an iteration went

Ten iterations of the median subject onto the paper's first, each stage timed by wrapping the
package's functions from outside, on one Intel Xeon Gold 6140 node (job 5678969):

| stage | eight cores | one core |
| --- | --- | --- |
| locating the moved endpoints on the grid (`MeshQuery.query_faces`) | 16.5 s | 16.3 s |
| the density and its derivatives | 2.3 s | 4.7 s |
| the gradient | 0.5 s | 0.5 s |
| the warp's flow and its fold check | 0.3 s | 0.3 s |
| the endpoints carried through the warp | 0.3 s | 0.3 s |
| an iteration | 19.9 s | 22.1 s |

Eight cores bought 1.1 times one: only the density's sparse products ran in threads. The
search was 83% of an iteration, most of it each of 1.6 million endpoints tested against its
twelve candidate triangles, in blocks of 32,768 points that do not depend on one another.
Item 19 had found it four fifths of an iteration.

### The change

`MeshQuery.query_faces` shares its blocks among threads: the cores the process may use, at
most eight, as `sparse_times_dense` counts them, or its new `threads=`. Each block is searched
by the code that searched it before and writes its own rows, so the faces and weights do not
depend on the threads. The GPU's search is unchanged. A test in `tests/test_alignment.py`
searches ten blocks of points on the sphere, outside it and near its centre, so that both
fallbacks run, in one, two, three, eight and the default number of threads, against each
block searched alone.

### What it gives

- **The search.** A subject's 1,632,466 endpoints, located once on an Intel Xeon Gold 6140:
  16.9 s on one core, 5.1 s in eight threads (job 5679109, the blocks mapped in threads from
  outside, the same bits).
- **An iteration.** Ten iterations on the same kind of node: 22.9 s each before, 10.0 s
  after, the cost trace, warps, moved endpoints and refused steps the same bits (job
  5679940).
- **Whole registrations.** Onto the paper's first subject, before and after on one AMD EPYC
  7702 node, the costs, refused steps, both warps and the moved endpoints the same bits (job
  5683082):
  - 62 iterations: 972 s before, 445 s after, 2.2 times as fast;
  - 117 iterations: 1,733 s before, 926 s after, 1.9 times as fast. The first pair ran
    beside the suite's job on the node, so 1.9 is the fairer figure.
  - With the threads a registration held up to 4.6 GiB.
- **Thirty iterations, as the cohort figure registers** (`registration_timing_probe.py`,
  before and after on one AMD EPYC 9654 node, job 5683544): a registration onto the template
  took 5.4 minutes before and 2.4 after, onto the first subject 5.8 and 2.6, and the eleven
  onto their template, the template included, 61 minutes and 28.
- **On a V100 node's Xeon Gold 6148** (`gpu_probe.py conseal`, job 5683083): relocating a
  subject's endpoints takes 3.4 s, against 17.2 s at item 28, and ten iterations 65.9 s
  against 242.5 s; the V100's lead there falls from seventeen times to five.

### Verified

- **The suite.** 1,371 tests pass with 48 skipped on the released subjects, and 1,368 with 51
  skipped as a fresh clone has it: item 35's counts and the new test (job 5683080, an AMD
  EPYC 7702). `ruff check` and `ruff format --check` pass on CI's paths.
- **The CPU computes as before.** Before and after on one AMD EPYC 7702 node (job 5683081),
  the 95 arrays of `cpu_parity_probe.py` and the 29 of `record_outputs.py` on the released
  subjects are identical, bit for bit.
- **On a GPU.** On a V100 (job 5683083):
  - `tests/test_gpu.py` and `tests/test_alignment.py` pass, 162 of 162;
  - `gpu_review_probe.py` finds what item 28 found: no table left on the GPU, the refusals,
    1,607,482 endpoints relocated in 0.73 s with none sent to the search of every face, the
    same bits twice, the CPU's rotations bit for bit, and every registration the tests
    compare within 0.062 of its allowance;
  - `gpu_probe.py conseal` finds the GPU as near the CPU as before: the costs 2.0e-9 apart
    after ten iterations, the twin's 4.0e-9.

The probes and their batch files are in `/work/users/x/y/xya/paper-open-items/dice/scripts/`
and `/work/users/x/y/xya/edits/i36/`.

### The documents

- `endpoints_align`'s docstring gives the iterations a registration takes and an
  iteration's time.
- USAGE.md's ConSEAL notes give real registrations' times beside the known warps', the
  threads and the memory. *How fast* and *A registration, subject by subject* hold ConSEAL's
  new CPU times; the second no longer says the endpoints are found on the warped mesh, since
  they are found on the grid.
- docs/RESULTS.md, docs/figures/README.md, `scripts/hcp_figures.py` and USAGE.md's cohort
  example say their ConSEAL hours came before this item.

Only `MeshQuery.query_faces`, its test and the documents change here.

## 37. A session label no file carries, 20 October 2026 -- NAMED AS THAT, AND THE REPORT KEPT

Reported by the user (medium): `load_cohort`'s file loop dropped every file of another session
before the subjects were counted. A `session=` label that matched no file therefore left no
subject at all, the refusal said "no files of the asked modalities were found" over a folder
full of them, and `sbci cohort FOLDER --session 5 --report r.csv` wrote nothing, since the
refusal carried no rows. A subject whose files were of other sessions only, beside others of
the asked one, vanished from the report the same way.

- **The files left out are counted.** The loop records, for each subject and modality, the
  sessions of the files `session=` leaves out.
- **Their subjects are reported.** Such a subject stays in the report, its reason naming the
  session asked for and the ones it has: `no fc file of session '2', only of sessions 1 and
  3`, or, under `{"sc": None, ...}`, `no sc file without a session label, only of session 1`.
- **A label no file carries says so.** When no file is of the session asked for, the refusal
  adds `No sc file is of session '5': the files are of session 1.` and drops the hint about
  the table's ids, which are not what failed. The report goes with the refusal, so
  `--report` writes it.
- **No report, said.** When a refusal carries no rows at all (no file of the asked
  modalities, and no table), `sbci cohort --report` now says it wrote none.
- **Tests.** Three new ones in `tests/test_cohort.py`: the refusal and its report, a subject
  reported rather than dropped, and the command line writing the report and saying when
  there is none. Three assertions take the new reasons.
- **Verified.** 1,374 tests pass with 48 skipped on the released subjects, and 1,371 with 51
  skipped as a fresh clone has it: item 36's counts and the three new tests (job 5903044, an
  AMD EPYC 7713). `ruff check` and `ruff format --check` pass on CI's paths.
- **The documents.** `load_cohort`'s docstring and USAGE.md's paragraph on sessions say it.

Only `load_cohort`, the command line's report, the tests and the documents change here.

## 38. The review of 20 October 2026 -- ELEVEN MEDIUM FINDINGS AND THE LOW ONES FIXED; THE NUMBERS UNCHANGED BUT FOR TWO-SUBJECT TEMPLATES

An external review of 5095374, on macOS with Python 3.10 and 3.12, found nothing broken. It
reported eleven medium findings, about forty-five low ones and some drift in the documents.
Each was checked again on Longleaf, read-only, by reproducing it. All held but one, the
import time, which is 0.2 to 0.3 s on Longleaf, as USAGE.md says; several went further
than the review said. The second medium finding, a session label no file carries, is item
37. This item fixes the rest.

### The user's decisions

- **Every save has a temporary file of its own.** A killed write may leave one stray file
  with a recognizable name; that is accepted.
- **ENCORE's diagonal keeps the MATLAB's arithmetic.** The reference zeroes the diagonal
  after the interpolation and does not give its mass back (ConCon_Alignment
  `scripts/Concon.m` lines 71-73, the same in the fork's `ConConInterpolator.m`), so the
  numbers stay and the loss is documented. Items 4 and 9's claims are corrected (see
  *Corrections*).
- **A template of two subjects is their geodesic midpoint**, in both methods.
- **Warps record their grid.**
- **The Matérn smoothness is recorded.**
- **Unchanged:** the LICENSE holder (BLUEPRINT.md Q7, the user's), the Python
  classifiers and the import-time sentence.

### Saving, reading and fetching files

- **Temporary files.** `write_hdf5` and `export._atomic` each write to a file of their own
  in the target's folder: `.<name>.<random>.partial` for HDF5, and
  `.partial-<random>-<name>` for exports, since nibabel writes CIFTI and GIFTI only to a
  name with the target's ending. Two saves of one path no longer delete each other's file.
  With HDF5's file locking off they had also replaced a good file with an unreadable one,
  and two table writes had spliced their rows. The file is made with mode 0666 and the
  umask, as a file written in place is (`tempfile.mkstemp`'s 0600 would have carried over
  to every saved file).
- **The cohort loader:**
  - `summary()` counts each reason once: the joined text was split on the "; " that some
    reasons hold. `LoadedCohort.reasons` keeps them as lists.
  - Empty, unnamed table columns (Excel's trailing ones) are dropped, and an unnamed
    column with values is refused.
  - A table that is not UTF-8 is refused by name.
  - The `sub-` prefix is read in any case, as the documents said.
  - A part of a name is a key only if it starts with a lowercase key and a hyphen, as in
    BIDS. So `sub-NDAR_INV-1_sc.h5` is one subject, where it was read as `sub-NDAR`. A
    capitalized key such as `SES-1` is now part of the id.
  - Two files of one name are named by their folders.
- **Downloads:**
  - A terms record of another shape counts as not accepted.
  - DATA_USE.txt is written before the first file is fetched.
  - manifest.csv lists only files the index knows.
  - A subject id may be any whole number (a spreadsheet's `NNNNNN.0` is read as `sub-NNNNNN`).
  - An `out` that is a file is refused before the terms are asked.
- **The command line** ends quietly when its reader stops reading (`--list | head` printed
  a traceback every time).
- **`example` and `example_cohort`** refuse a seed or a count that is not a whole number,
  where `seed=1.7` became 1.

### Smoothing

- **The Matérn smoothness.**
  - `smooth()` takes `nu=`: 3, as before, by default, and refused with any other kernel.
  - A Matérn file records it, and `load_cohort` compares it, reading a Matérn file written
    before as 3.
  - Two files at different `nu` differ by up to 7%, and the cohort had taken them as one
    way of making.
- **The spherical kernel's memory.** Its sparse path shrinks a block of endpoints until its
  neighbour pairs fit a budget of 1 GB a side. It keeps 65,536 endpoints at the default
  bandwidth and up to 0.1, and takes 21,805 at 0.5, where a block had needed 13.5 GB.
  (The budget counted the pairs' 24 bytes alone, where the block's peak is about 64 a
  pair: item 41 made the budget mean the peak, so a block at 0.5 is 8,176 endpoints.)
  `bandwidth=True` and `bandwidth="0.005"` are refused.
- **What the kernel is given:**
  - sigma must be finite and positive, and a block a positive whole number;
  - a NaN cosine gives NaN;
  - `to_dense` refuses a vector of the wrong size;
  - a NumPy float bandwidth is the decimal it prints as, so `np.float32(0.005)` is 0.005.
- **The endpoints.** A row of barycentric weights must sum to 1 within 1e-4; slightly
  negative weights stay, since the search keeps unclamped ones. SPEC_QUESTIONS item 13's
  rule, a stored vertex within one edge of its triangle, is checked where positions are
  rebuilt and is `sbci validate`'s thirteenth check. The released subjects lie at most
  2.73 degrees from their vertex, against an edge of up to 4.74. The item-13 probe runs
  again: it had divided the bundled sphere's read-only array in place.

### The alignments

- **Warps know their grid.**
  - `Warp` and `EndpointWarp` record a digest of the grid they were estimated on; a turn of
    the bundled sphere gives the same digest.
  - `migrate_warp` refuses a warp whose hemispheres are not 2,562 vertices each, and one
    whose digest is not the bundled sphere's.
  - A warp of another size had "migrated" with 180-degree errors. One from a same-size
    icosphere had passed silently, with garbage.
  - A warp saved before has no digest and is checked on its size alone.
- **Two subjects.** `Encore.template` and `ConSEAL.template` return the normalized sum of
  the two square-root densities, a Karcher median that does not depend on their order. The
  median had ended a fifth of the way from the first-listed subject. Three or more subjects
  give the same bits as before.
- **GIFTI spheres** carry their structure, `GeometricType=Spherical` and a Talairach
  transform, as HCP's do.
- **ConSEAL:**
  - A connectome whose weights do not sum to one is refused, as the moving subject or as
    the fixed one.
  - The vertex-wise overlap and Dice check their threshold. They score two empty sets 1, as
    the connection-wise pair does; the MATLAB gives 0/0 there.
  - `template=` is checked before the kernel is built.
  - The spin shells are cached.
  - The threads the package starts are capped by `SBCI_THREADS`, else `OMP_NUM_THREADS`.
- **ENCORE:**
  - `step`, `threshold` and `delta` refuse a bool or a string. `threshold=True` had made a
    registration take no step, silently.
  - `compose` takes only a displacement of one 2-vector per vertex.
  - A string `template=` is refused with a pointer to `template=None`.
  - A basis order above 85 is refused, since it gave a NaN basis.
- **ENCORE's diagonal:**
  - `Concon.evaluate`'s docstring and USAGE.md say the aligned density loses the mass the
    interpolation puts on the diagonal, as the MATLAB's does.
  - On two released subjects after ten steps, the loss was 0.34% and 0.49%, almost all of
    it on the diagonal; the transport itself moved the mass by about 2e-4 (job 5958360).
  - A test pins it on an SC-like density.

### Seeds, plots and statistics

- **Seeds.** A boolean region mask must be one entry per vertex: a (5124, 1) mask had
  broadcast, and a single-vertex region had given a wrong profile with no error.
- **Plots:**
  - `bg_map` is a named parameter: one value per grid vertex, split by hemisphere and
    brought onto the fsaverage meshes as the map is. A half map had shaded both
    hemispheres.
  - Under `engine="pyvista"` a keyword the engine would drop is refused.
  - `render.py` no longer sets PyVista's `OFF_SCREEN` for the whole session.
  - An explicit `vmin >= vmax` is refused. A range nilearn's two-digit ticks cannot tell
    apart gets enough digits, and one that six digits cannot tell apart is drawn without a
    colorbar; both had crashed.
- **Statistics:**
  - `icc` refuses a bare 2-D array.
  - `benjamini_hochberg` refuses p outside [0, 1].
  - `identification` gives a t-way tie 1/t, so the subjects' order no longer matters.
  - `design()` takes a DataFrame.
  - Family labels that are one only as text (1 and "1") are refused.
- **Reduction.** `reduce()` checks `rank` before reading the cohort, and `start=` must be
  (n, rank), finite and without an empty direction. `explained`'s docstring says that its
  square is the share of the sum of squares exactly only under a plain inner product; under
  ico4's areas it differs by about 1e-9.

### Tests, CI, scripts and documents

- **Test collection.** A root `conftest.py` keeps pytest out of `tests/reference`,
  `scripts` and `tools` from any folder. `pytest .` from the root had imported them, and
  two do damage on import: `write_exchange_file.py`, which writes the 16.9 GB exchange
  file, and `tools/align_surface_faces.py`, which rewrites the bundled surfaces. Those two
  now run only as scripts. The suite collected item 37's 1,422 tests, in the same order,
  before this item's hundred were added.
- **`make_figures.py`** parses its arguments (`--help` had made a folder of that name),
  takes `--skip-cohort`, and refuses to write into `docs/figures`, whose real figures it
  would have overwritten.
- **CI** uses actions/checkout and actions/setup-python v7 (Node 24) and Ubuntu 24.04,
  where `ubuntu-latest` was moving to 26.
- **`verify_all.sh`** asks that python import sbci and pytest instead of a virtual
  environment, so it runs from conda too, and takes `--audit` for tier 2 outside Slurm.
  `show_build.sh` sizes the tree portably.
- **The documents:**
  - README.md's alignment and reduction rows unwrap the cohort as USAGE.md does.
  - The skip counts are 74 on a clean install, with 23 MATLAB-reference tests, and CI's 75.
  - Schaefer-900, Schaefer-1000 and CoCoNest-500 have 899, 999 and 499 regions at ico4. A
    label after a gap is renumbered, and CoCoNest's labels never equal its cluster ids, so
    regions are matched by name.
  - The sdist leaves out docs/figures' PNGs, with hatchling 1.22 or later.
  - A deliberate warning in tests/test_reduction.py is caught.

### Left as they are

- **Download parts.** A download's `.part` files keep their fixed names so that a re-run
  resumes them. Two downloads into one folder at once still share them.
- **Broken pipes in some commands.** `sbci download`, `info` and `cohort` still print
  `error: [Errno 32] Broken pipe` when their reader stops, since they catch OSError
  themselves.
- **Scripts without a guard.** `scripts/audit_api.py`, `plot_examples.py`,
  `plot_exchange_file.py` and `tour.py`, and `tools/age_probe.py` and
  `build_fslr_surfaces.py`, still run on import. They write only to their own folders, and
  pytest no longer collects them.
- **`sbci info`** does not print `nu`.

### Corrections

- **Item 4.** Item 4 said the calibrated Jacobian conserves the transported mass to 1e-4.
  The transport alone moves it by about 2e-4 on two released subjects. Zeroing the
  diagonal, as the MATLAB does, then loses 0.3% to 0.5% of the aligned density
  (`Concon.evaluate`); the square-root density the cost compares is normalized after its
  diagonal is zeroed, as item 4 says.
- **Item 9.** Item 9's "keeps the aligned density's mass" holds against the reference's
  0.7% loss only.

### Verified

- **The suite.** 1,474 tests pass with 48 skipped on the released subjects, and 1,471 with 51
  skipped as a fresh clone has it. That is item 37's counts and 100 new tests, with no
  warning left (job 5961302, an AMD EPYC 9654). `ruff check` and `ruff format --check`
  pass on CI's paths.
- **The CPU computes as before.** Before (5095374) and after, on one AMD EPYC 9654 node
  (job 5961303):
  - the 95 arrays of `cpu_parity_probe.py` are identical, bit for bit;
  - of the 29 of `record_outputs.py`, the 15 outside the alignments are identical. The 14
    of ENCORE and ConSEAL move, since it aligns two subjects, whose template is now their
    midpoint.
  - Templates and registrations of three to six subjects keep their bits (48 arrays, job
    5958273).
  - The default smoothing keeps its bits with all three kernels, on a synthetic subject
    and the tutorial subject (job 5958391).
- **On a GPU.** On a V100 (job 5961304):
  - tests/test_gpu.py, test_alignment.py and test_conseal.py pass, 332 of 332;
  - `gpu_review_probe.py` finds what items 28 and 36 found, every registration the tests
    compare within 0.062 of its allowance;
  - `gpu_probe.py conseal` finds the GPU as near the CPU as before: the costs 2.0e-9 apart
    after ten iterations, the twin's 4.0e-9.
- **The MATLAB tier.** It passes, 23 of 23.

Five parts were written in parallel, one per group of modules, each with its own tests. They
were then merged without a conflict and checked together. The checks, the probes and their
outputs are in `/work/users/x/y/xya/edits/i38/` and `/work/users/x/y/xya/edits/review-1020/`.

## 39. The lab's ADNI subjects out of the documents, 21 October 2026 -- THE CHECKS KEPT; THE KERNEL COMPARED ON A RELEASED YOUNG ADULT, WHERE THE BINARY NEVER WRITES THE SOUTH POLE

Three reference checks ran on the lab's ADNI subjects, the only subjects for which the
pipeline had kept both what `c3_main` was fed and what it wrote: the spherical kernel at
full scale against that output (item 6), ENCORE making five subjects more alike
(VERIFICATION.md, Tier 3), and ConSEAL on three (item 7). The face-order check of
SPEC_QUESTIONS.md item 13 read them too. ADNI's data are shared under their own agreement,
which governs what may be published from them, so on 21 October 2026 the user asked that
the checks stay and that the subjects and their results leave the public documents.

- **The documents.** Items 6 and 7, VERIFICATION.md, SPEC_QUESTIONS.md items 10 and 13,
  BLUEPRINT.md, USAGE.md and the docstrings of `sbci.smoothing` name none of those
  subjects and give no number measured on them. What their comparisons taught about the
  kernel stays, in words; item 6 keeps one timing, on "a subject of about a million
  streamlines".
- **The checks.** `tests/reference/align_adni_ico4.py` and `conseal_adni_ico4.py` read
  their subjects from `SBCI_ADNI_SUBJECTS`, comma-separated, and refuse to run with fewer
  than two; the lab keeps the list with its data. Nothing else in them changes, and
  neither runs in CI.
- **The kernel's full-scale comparison** runs on the tutorial subject instead.
  `tests/reference/concon_fullscale.py` writes the stored endpoints of its released file
  as `c3_main`'s input, runs the binary with the pipeline's flags, and compares what it
  writes with the package's density from the same positions. On all 803,741 streamlines
  the two agree at r = 1.00000000 over the 13,125,126 pairs, scale 1.000000 with
  nothing fitted, and write the same 6,887,499 pairs (job 5989183) -- once the grid's south
  pole is emptied. Like `concon_probe.py`, it needs the binary and the pipeline grid's
  `.m` files, neither of which is in this repository.
- **What the new comparison found: `c3_main` never writes the south pole**, the vertex
  at (0, 0, -1) in each hemisphere (left fusiform, right inferior temporal). A half-open
  range test in concon's vertex lookup skips it, so every connectome the binary made has
  nothing there; the toolkit's example output is empty there too. The package writes it,
  0.072% of the tutorial subject's mass, and keeps doing so: the hole is the
  reference's error, put back only in the comparison. As computed, the correlation over
  all pairs is 0.99910256. Item 6 used to put the extra pairs on the kernel's cutoff
  boundary; they are the pole's (item 6, *Full scale*).
- **The lab's fMRI copy** is now at `/hickory/proj/STOR/zz10c`.
  USAGE.md's note on lab paths, its FC example and the usage line of
  `tools/build_hcp_fc.py` name it.
- **Panel D** of the paper shows the test at every vertex (item 34), so the Status no
  longer lists it as open.

### Verified

- **The suite.** 1,475 tests pass with 48 skipped on the released subjects, and 1,472 with 51
  skipped as a fresh clone has it: item 38's counts and the new
  `test_the_south_pole_takes_its_mass_which_c3_main_never_writes` (job 5988556, an AMD EPYC
  9654). `ruff check` and `ruff format --check` pass on CI's paths. Three docstrings of
  `sbci.smoothing` were reworded after it ran; its doctests and `tests/test_smoothing.py`
  pass on the final text.
- **The full-scale comparison** is job 5989183, on an AMD EPYC 7763: the binary took 2,975 s and
  the package's density 15 s. A first run of the same comparison (job 5981456) found
  the pole, and the toolkit's example output is empty at both hemispheres' south poles too
  (job 5987601).
- **The ADNI checks still run.** Without `SBCI_ADNI_SUBJECTS` both refuse; given the lab's
  list, both load their subjects and start.
- **Nothing else changes.** `src/` differs from item 38 only in those docstrings.

## 40. The paper's three worked examples as notebooks, 21 October 2026 -- EVERY NUMBER THEY QUOTE REPRODUCED

The paper's Section 5 walks through three examples, each a notebook; until now only the
evidence sections' two analyses were notebooks. `notebooks/01_explore.ipynb`,
`02_coupling.ipynb` and `03_cohort.ipynb` follow each example's items -- question, input,
choices, execution, output, interpretation, sensitivity -- in the form of the two before
them: the lab's paths at the top, the package's figure style, prose whose every number is
the notebook's own output, saved with the outputs.

- **01_explore** opens with the paper's listing as printed and runs it: `fetch_cohort` finds
  the eleven files present and verified, and fetches nothing. Subject A's seed profile at
  vertex 1234, the split-half check (raw maps r = 0.41, smoothed 0.99, against 0.54 to 0.80
  between subjects), the Desikan and Schaefer-200 matrices as mass and as mean, what the
  banks-superior temporal entry gathers (3,885 vertex pairs whose densities span 1,865-fold),
  the export, and the matrices at sigma 0.01 (Schaefer-200 r = 0.993, Desikan 0.998) with the
  control at 0.005 (r = 1.000000).
- **02_coupling**: the eleven's coupling (highest in occipital cortex, 0.36 to 0.41; lowest
  in the caudal anterior cingulate, entorhinal and posterior cingulate cortex, 0.05 to 0.09;
  subjects' maps r = 0.53 to 0.71), the precentral gyrus (0.14 to 0.37 inside, rising
  dorsally), the cingulate against the medial wall (63% of its gap is nearness to the wall),
  negative FC set to zero (r = 0.885 to 0.957 per subject) and FC without global signal
  regression (r = 0.684 to 0.926, median 0.815), the last run only where that FC exists.
- **03_cohort**: the 943, the rank-20 fit (27.3% of the norm), the component tests (fluid
  intelligence in none; sex in components 3, 5 and 8, adjusted p 0.035), the test at every
  vertex (sex at 433), the results in Desikan terms, bctpy's graph measures and the
  network-based statistic with whole-family sign flips (343 and 60 connections), and the
  sensitivity checks. `scripts/cohort_data.py` reads each SC file once for it; its arrays
  are bit for bit the ones the earlier runs used.
- **The HCP's terms.** The cohort notebook reads the family table at run time and prints
  counts only, and names no subject. The first two use no family table.

Building them found four places where the paper's wording or this repository's notes ran
ahead of the numbers, which the user has: the zeroed-negative range rounds to 0.88, not
0.89; association regions vary less, not more, inside than unimodal ones, except relative
to their mean; without global signal regression the temporal pole is among the lowest; and
for fluid intelligence two Desikan regions tie for closest.

### Verified

- **The notebooks** run without an error, a traceback or a line on stderr, and the first
  holds the paper's listing as printed (jobs 6105246, 6107062 and 6106816 on Longleaf).
- **No subject id** appears in the cohort notebook; the other two show the eleven's only in
  `fetch_cohort`'s report.
- **`scripts/cohort_data.py`** passes `ruff check` and `ruff format --check`.

## 41. The review of 22 October 2026 -- FIVE REVIEWERS, NOTHING BROKEN; AN HCP ID OUT, THE BINARY'S RESIDUAL EXPLAINED, THE NOTEBOOKS' CHOICES WIRED THROUGH

Five read-only reviewers read 89a9798, one each for the three example notebooks, the
kernel commit (item 39), the code changed since the last outside review (5095374), the
documents, and the packaging, CI and tests. None found a method wrong or a number
unreproduced; the one high finding was an HCP subject id that item 38 had written into this
file, a page that reports family-based results. What they found, and what changed:

- **The HCP id** in item 38's *Downloads* ("a whole number ... is sub-...") is a placeholder
  now. It was the tutorial subject's, written as an example of a spreadsheet's `.0`; the rule
  of item 33 stands whatever the reason. No other id is on a family-result page: the review
  scanned every tracked text file for the 946 ids.
- **The binary's residual explained** (item 6, *Shipping it*): the 6e-6 left after the south
  pole is emptied is not the binary's arithmetic but its single-precision read of the
  positions and the six-digit constants of its harmonics library; with both put back, the
  package reproduces `c3_main` to 1.05e-15 relative, 2,736,226 of 6,887,499 pairs bit for bit
  (`tests/reference/concon_fullscale.py --reuse`, job 6186083). The script gained the two
  emulations, `--reuse`, a check of the binary's exit status and the names of the variables
  that point at the binary and the grid files. Item 6 also says now that the grid stores the
  pole as exactly (0, 0, -1), that the binary reads one grid file for both hemispheres, and
  that "the one vertex" was two, each hemisphere's south pole (README, BLUEPRINT, item 6).
- **Downloads.** Two downloads into one folder took up the same `.part` between them and
  blamed the server for the bytes ("sent more than the ... bytes", or a wrong SHA-256). A
  second download into a folder is now refused while the first writes a file, by an advisory
  lock on a sidecar beside the part, held while it is written, verified and moved onto its
  destination (`_owned`); the resumable names stay. A basis folder that cannot be made is named.
- **The command line.** `sbci info`, `cohort`, `download` and `basis` caught
  `BrokenPipeError`, an `OSError`, in their own handlers and printed "[Errno 32] Broken pipe"
  before `main()` could end quietly; each lets it through now. `sbci info` prints `nu` beside
  a Matern file's bandwidth, and `sbci cohort`'s summary leaves out a setting the files do not
  record (`nu None` beside every `shk` cohort). `python -m sbci` runs the command line.
- **Cohort names.** A BIDS key written in capitals (`sub-01_SES-1`) is read into the id with a
  warning now, once a spelling; a blank `session=` is refused; an empty part (`sub-01__sc.h5`)
  or a key given twice makes a name unreadable, as `ses-` without a value did.
- **The spherical kernel's memory budget** counted the pairs' 24 bytes alone, where the sparse
  path's peak is about 64 a pair (tracemalloc: 658 MB at 2,000 endpoints and sigma 0.5, of
  which 420 MB are the two dense arrays); `PAIR_BYTES` is 64, so the budget means the peak,
  and a block at sigma 0.5 is 8,176 endpoints. The default bandwidth keeps its block, and its
  bits.
- **Tests from the root with the package installed, not `-e`**, failed to collect: the doctest
  modules under `src/` were imported a second time beside the installed `sbci` and refused as
  a mismatch (26 errors), which CI never saw because every job installs `-e`.
  `--import-mode=importlib` in `addopts` fixes it (1,447 passed, 76 skipped on such an
  install). `conftest.py` says that naming a script still collects it; the six scripts and
  tools that did their work on import (`audit_api.py`, `plot_examples.py`,
  `plot_exchange_file.py`, `tour.py`, `age_probe.py`, `build_fslr_surfaces.py`) refuse to be
  imported instead.
- **Packaging and CI.** `hatchling>=1.22`, which the sdist exclusion needs; the notebooks with
  their outputs (10 MB of an 11 MB sdist), the CI configuration and `.gitignore` out of the
  sdist; a `notebooks` extra (`sbci[render]`, scikit-learn, bctpy, jupyter); `.gitignore` covers
  the save temporaries and what the notebooks write where they sit; the workflow runs on pushes
  to main and on pull requests, with `permissions: contents: read`, a 60-minute limit on the
  matrix, and ruff pinned. A test ties `__version__` to the installed metadata; the
  five-minute-start fixture records the HCP terms in a scratch config, not the user's; the
  MATLAB tier's basis fixture looks where the package keeps the basis too.
- **The notebooks.** The cohort notebook's exposed outcome and covariates flow through every section now: `require` and `values` derive from them, the head-size comparisons run only when the volume is among them, and a model under which nothing passes says so instead of raising. Its smoothing-control sentence quotes docs/RESULTS.md's record instead of reading a log outside the repository; its network-based statistic's p-values are (b + 1)/(m + 1), so the 343-pair network reads 0.0002 and the 60-pair one 0.0022, and it says that each direction is tested at 0.05 on its own. The coupling notebook's spreads within and between regions use one weighting (unweighted; the between-region SD is 0.079 over 68 regions and 0.080 over 66, as the area-weighted means gave). The second and third notebooks end with the timing cell the first has, which their opening cells quote; nilearn's home-folder line is gone from their outputs. Re-executed as jobs 6186207, 6186208 and 6186209 (2.6, 2.7 and 11.4 minutes), every other number stands, but one count of the null control (random labels shared within families, families ignored, the negative direction) moved from 26 to 24 of 100 with the reworked cell's draws.
- **Documents.** The GPU speed-up in README's table (ConSEAL's 17 times was before item 36; 5
  now); the zeroed-negative-FC range, 0.88 to 0.96, in USAGE, `coupling.py` and item 33's table;
  the review note's *Still open* bullet that item 36 answered; item 28's description of the
  one-core search, until item 36; item 38's "1,422 tests" (item 37's count, before its hundred)
  and its budget sentence; the skip counts (76 without the render extra, two PyVista tests, not
  75 and one); "two notebooks" in RESULTS and BLUEPRINT; the atlases' source files are `.npz`;
  `cohort_coupling.png` is embedded where RESULTS gives its numbers; the benchmark's smoothing
  control is recorded in RESULTS (job 5477801), so the cohort notebook need not read a log;
  USAGE says what a killed save leaves; `md2pdf.py`'s reportlab is named; the notebooks' README
  says to run a copy, with the VTK setting a headless machine needs.

### Corrections

- **Item 6** said the residual was the binary's arithmetic; it is the binary's float read and
  six-digit constants (above). It said "the one vertex"; it is each hemisphere's south pole.
- **Item 28** described the point location on one core and on the warped mesh; item 36 threads
  it and finds the triangles on the grid (marked in place).
- **Item 33's table and item 38's text** carry the corrected range and counts (above).
- **Item 40** quoted "0.89"; the notebook prints 0.885 to 0.957, and the documents say 0.88.

### Left to the user

- **The ADNI subjects in the history.** Commit 949ad82 took them out of the documents;
  `git show 067c39d:PORTING.md` still prints them. If ADNI's agreement reaches the repository,
  the history needs the treatment of item 33's purge (the ticket is open).
- **The paper's wording** that notebook 02 found wanting (item 40), and Example 3's *Choices*
  bullet, which names inclusion rules, a cross-validation design and atlas comparators the
  cohort notebook does not expose (the prediction notebook holds the cross-validation).

### Verified

- **The suite.** 1,483 tests pass with 48 skipped on the released subjects, and 1,480 with 51 skipped as a fresh clone has
  it (job 6187744, an AMD EPYC); the fresh clone's 51 skips are Longleaf's, where the MATLAB
  references are present; a machine without them skips 74 with every extra, 76 without
  render. `ruff check` and `ruff format --check` pass on CI's paths and `conftest.py`.
- **The kernel comparison** against the recorded run's output, job 6186083 (above).
- **The notebooks** ran again without an error, a traceback or a line on stderr (jobs
  6186207, 6186208 and 6186209); no subject id in the cohort notebook, no home path in any output.
- **CI** passed on the pushed commit, the first with `--import-mode=importlib`.

## Status

All seven ports are done and verified, and the reviews of 5 to 7 October
2026 have been answered in full: the eight of the code here (items 8 to 15),
the two of the notebooks in docs/review-2026-10-05.md, a recheck of all of it
(item 16), the scores of the returned vectors (item 17) and an eleventh list
(item 18); the paper's ConSEAL was run beside the package's, which now
takes the paper's settings (item 19); and a review of the whole repository
was answered on 11 October 2026 (item 20), two of its decisions the same day
(item 21: ENCORE's order of composition, with `reference=True` removed, and
the exchange file at the wall), and ConSEAL's `strict_upstream` went too once
the defaults undid every known warp better (item 22); a recheck of all of it
found no high-severity defect and fixed what it found (item 23), and a third
did the same (item 24); ENCORE's template takes up to a hundred steps since
(item 25), and both alignments run on a GPU when asked (item 26); and a fourth
review, once the deposit's ticket was in, found no high-severity defect either
and fixed what it found (item 27), and the GPU's own review after it found no
wrong number and fixed what it found, the rotation search moving to the GPU
(item 28); the deposit goes up in 40 archives, by direct upload, as UNC Dataverse
advised, and the upload was rehearsed on its demo server (item 29); and the 946 are
published on UNC Dataverse, where `sbci download` fetches any of them, the HCP's terms
accepted once a machine (item 30); the Laplace-Beltrami basis is fetched when first needed,
and USAGE runs on the public data (item 31); and head size is in the cohort's model, where
sex shows in 3 components rather than 13, in the right cuneus (item 32); and a review of
every claim found fluid intelligence in no vertex once the covariates are in, sex at 433
vertices, mostly higher in women, elsewhere than the components' effect map, and
family-based results beside HCP ids, all of it fixed (item 33); and the paper's panel D,
drawn from the test at every vertex, holds in women and men matched on head size (item 34);
and the overlap of bundles after alignment, connection-wise as the ConSEAL paper measures it,
is in the package, agrees with the MATLAB exactly, and rises under both methods (item 35);
and ConSEAL's search of the grid runs in threads, with the same bits and a registration about
twice as fast on eight cores (item 36); and a session label no file carries is refused as
that, with its report (item 37); and the review of 20 October is answered, its eleven medium
findings and the low ones fixed, the numbers unchanged but for two-subject templates (item
38); and the lab's ADNI subjects are out of the documents, the checks on them kept and the
kernel's full-scale comparison made on a released young adult instead, which found that
`c3_main` never writes the grid's south pole (item 39); and the paper's three worked
examples are notebooks, every number they quote reproduced (item 40); and five reviewers
read all of it once more, an HCP id came out of this file, the binary's residual is the
binary's, and the notebooks' choices flow through (item 41). What is left: a version of the
dataset whose README names
the commit that holds the code now, and GitHub's removal of the commits before it (item 33,
*The history*); the components' sex result, which sits at the edge of the threshold (item
32); the WP1 sign-offs of items 3 (the
triangle convention), 4 (`delta`) and 5 (where each component starts, and the
area weighting); item 5's request that whoever specified the API review the
choice of test in `local_test` (VERIFICATION.md's checklist has it too); item
2's two questions for the group, the diagonal of the global coupling and
which inner product the published figures used; the rest of the
decisions item 20 ends with; items 7 and 19's *Still open*; and the review
note's. The ports were done in the order 3, 1, 2, 5, 4, 6, 7: parcellation
unblocked the first notebook, kernel smoothing the WP3 speed target, and the
two alignments came last because nothing else depends on them.
