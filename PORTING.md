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
| Kernel `KM_L` | 3.18 float32-eps relative |
| Density `DM` | 3.17 float32-eps relative, correlation 0.9999999999999 |

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
checks are what resting-state FC must show, on sub-100307:

| Check | Result |
| --- | --- |
| the 32k vertices a grid vertex averages | in the median cell all within 2.4 degrees of the grid vertex (95th percentile 3.3), against a grid spacing of about 4 |
| two 32k vertices of one cell, correlated in time | median 0.65; two at random, 0.009 |
| neighbouring grid vertices | FC 0.27 (0.43 band-passed to 0.01-0.1 Hz) |
| homotopic Desikan regions | 0.38 (0.54) |
| seed in the left isthmus cingulate | precuneus 0.46, inferior parietal 0.36, medial orbitofrontal 0.31: the default network |
| the white-surface route sub-103010 needs, run on sub-100307 | 91% of the 32k vertices in the same cell as the exact route, 0.24 degrees from it at the median |

The released files are full band, as the pipeline computes FC. Coupling of
each of the eleven subjects' SC with its own FC:

| Form | Mean per subject |
| --- | --- |
| global | 0.200 to 0.268 |
| within Desikan regions | 0.595 to 0.679 |
| discrete, on the Desikan matrices | 0.271 to 0.370 |

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
run), 8 three, 21 two and 1 one. Every file validates; sub-103010 alone took
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
leaves empty, so that part of the map is the one to doubt. With ten families
as clusters, the same test on the released subjects finds hundreds of
vertices: the cluster-robust test needs many families (the stats module
notes), and the eleven are for checking the code, not for inference.

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
  (six in Gordon, one to four in each CoCoNest scale); those read NaN under
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
  stage rather than only end to end.

### What matches exactly

| Stage | Agreement |
| --- | --- |
| Voronoi vertex areas (libigl massmatrix) | 1.6 float64-eps |
| Spherical harmonic tangent basis | 13.6 eps |
| Basis Laplacian (`reference=True`; item 8 says why the default differs) | 5.9 eps |
| Tangent frames, exponential and logarithm maps | 0.5 to 4 eps |
| Barycentric query against the AABB tree | 2.0 eps, same triangle 162/162 |
| Identity warp Jacobian (`reference=True`; the default calibrates it, *Four more* below) | exact |
| Warp composition, vertex by vertex | 0.3 eps |
| **Karcher median template** | **44 eps** |

### What cannot match, and why

Everything downstream of the reference's finite difference agrees only to
about 1e-3, and that is a property of the reference rather than of the port.
It differentiates with a step of `1e-10`. A central difference with that step
on quantities known to float64 precision has a condition number near `1e6`:
six of sixteen digits are gone before anything else happens.

Measured on the reference itself, changing only the step size:

| Quantity | Change between adjacent step sizes |
| --- | --- |
| `get_derivative` | **0.037**, about 2% of its range |
| warp Jacobian | 2.3e-04 |

The port lands inside that envelope: Jacobian 4.9e-04, `evaluate` 7.1e-04,
registered connectome 1.2e-03 at a correlation of **0.99999979**. Given
*identical* input points the Jacobian arithmetic reproduces the reference to
2.2e-06, which is exactly float64 epsilon divided by the step -- so the
arithmetic is identical and only the conditioning separates them.

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
doubles it back after an accepted step; `backtracks=0` reproduces the
reference, and the reference comparison above is run that way.

### How close is it possible to get?

The port's finite difference sits **0.0367** from the reference at every step
size. The reference sits 0.0352 to 0.0420 from *itself* across adjacent step
sizes. So the port is already as close as the reference is to its own
neighbouring configuration, and that is the floor.

The residual comes from one place. Face agreement with libigl's closest-point
query is **100%** for ordinary points and about **85%** for queries that
coincide with a mesh vertex, at every step size -- and the offset queries are
all of that second kind. At a vertex every incident face is exactly tied, so
libigl returns whichever its tree reaches first. Closing that gap means
reproducing an AABB traversal order, not any mathematics. Selecting by plane
distance instead drops agreement to 13.6%, and restricting to faces at the
nearest vertex to 0%, so the port's rule is the right one.

### Both derivative operators are checked against known answers

`sbci.alignment.gradient_operators` differentiates the piecewise-linear
interpolant exactly, and is available as `derivative="analytic"`. Both it and
the reference's central difference are tested against closed forms:

| Test | Result |
| --- | --- |
| `f(x) = x` on a flat mesh | gradient exactly 1, error 3.6e-15 |
| `f = 3x + 2y` | exactly (3, 2), error 1.4e-14 |
| sphere coordinate functions, analytic operator | converges, 2.3e-02 -> 8.2e-03 -> 2.1e-03 |
| sphere coordinate functions, reference difference | converges, 3.4e-02 -> 8.8e-03 -> 2.7e-03 |

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
mesh before the frames are built; `align()` refuses a grid with a vertex on
the axis or within 1e-3 of it (in `sin(theta)`), where the finite differences
are unreliable too: a vertex 1e-5 off the axis read 1.27 for a 0.5-degree
rotation (item 10). Worth reporting upstream: the reference's own demo grid happens to avoid
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
The port now corrects them by default -- the Jacobian is calibrated by the
identity's own value, so a zero step gives exactly 1 and the transported mass
is conserved to 1e-4 -- and `reference=True` restores the first, second and
fourth so that the agreement above can still be demonstrated; the fold check
stays on in either mode.

### Still open

- The registration loop descends a gradient built from the unstable
  derivative, so two implementations take different steps. Agreement is
  therefore reported as a correlation, not a tolerance.

The grid is no constraint: `SphericalGrid(mesh, l)` takes the mesh as an
argument, only the reference's demo scripts hardcode its retired 0.94 grid,
and this package uses the ico4 sphere throughout (SPEC_QUESTIONS.md item 12,
withdrawn).

## 5. FPCA reduction -- DONE, VERIFIED. Local inference still open.

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

With the same initialization, the port reproduces the reference exactly:

| Quantity | Result |
| --- | --- |
| component basis vectors | max difference **2.1e-16** after sign alignment |
| component scales | float64 rounding, worst 6.2e-15 |
| subject scores | float64 rounding |
| explained fraction | identical to 6 decimals |

`tests/test_matlab_reference.py` asserts the basis, scales, explained fraction
and scores. The tests in `tests/test_reduction.py` check what holds
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
projecting that cohort returns its scores: to rounding at a converged fit
(2.5e-16 in the tests), and to the fit's tolerance otherwise, because the fit
records each score one alternating step before its final component, as the
reference does (1e-3 at the default `tol_outer`; the reference's triangle form
was off by 5e-2 to 1e-1 on the same cohorts). `project(..., reference=True)`
keeps the reference's triangle objective, bit for bit.

### What the areas weight

`reduce()` passes the vertex areas as the Gram matrix, and they define the
inner product in which the components are made orthogonal and in which each
new component is deflated against the earlier ones. They do not enter the
leading component of a step: that is the Euclidean eigenvector of the
contracted matrix, as in the reference (matched to 6e-15 above), so the
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
**roughest** direction. Measured on a test cohort, roughness falls 1.96 to 1.71
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
0.0003, the explained fraction rising from 0.182 to 0.194 -- at about three
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
restricted table: it was read on Longleaf from a private copy, and nothing
from it is in the repository or its figures.

`tools/age_probe.py --table ... --column ... --groups ...` streams the cohort
once, in fifteen minutes on four cores (`--measures` repeats the tests on the
saved measures in seconds), and shows the trait is in the data but weakly,
far more weakly than sex:

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
p 0.083; without the covariates both are at 0.051. Counted as 943 independent
subjects, component 13 passed at 0.023 (and component 7 at 0.029 without the
covariates), which is what the first version reported, with an effect map in
right medial occipital cortex. A check that needs no clustering agrees with
the correction: keeping one subject per family, the first met in a random
order (`numpy.random.default_rng(seed).permutation`, seeds 0 to 4), leaves
422 independent subjects, in whom component 13 correlates with the score at
r = 0.10 to 0.12 with adjusted p 0.22 to 0.48. The effect's size holds; its
significance does not.

At rank 4, the first four of the same components (17% of the norm), nothing
comes close (smallest adjusted p 0.32), while sex shows in all four. At rank
20 sex, tested with the score among the covariates, shows in eleven
components (1, 2, 3, 5, 6, 8, 9, 12, 15, 16 and 18; thirteen counted as
independent), the strongest at adjusted p 2e-8. Its effect map over those
eleven (`docs/figures/cohort_sex_effect.png`) sits at the occipital poles of
both hemispheres, where connectivity is relatively higher in men, and nowhere
else above a twentieth of its largest value. Head size differs between the
sexes and is not in the model, so part of that may be size. The streamline
count matters less here than the rank does: it correlates with the first
component at -0.12 and with no other beyond 0.18. The fit read the cohort one
file at a time, held it as 199 GB of float64 and peaked at 196 GiB, and took
5.4 hours on eight threads, loading included.

### Measured earlier on HCP-Aging subjects, whose files are not distributed

A rank-4 fit of all 528 HCP-Aging subjects with complete pipeline output
(`candidates=6`, 20% of the cohort's norm) finds no component whose scores
track age: the correlations are 0.06, 0.08, -0.11 and -0.07, adjusted p 0.05
to 0.17. With 528 subjects aged 36 to 100 that is not a biological answer,
and `tools/age_probe.py`, which streams the cohort once, shows age is plainly
in the data:

| measured on each subject | against age |
| --- | --- |
| interhemispheric share of the connectivity | r = -0.56, p 5e-44 |
| area-weighted strength of each cortical vertex | 2,256 of 4,683 vertices significant at FDR 0.05, \|r\| up to 0.62 |
| each Desikan region pair | 1,188 of 2,278 significant, \|r\| up to 0.60 |
| first principal direction across subjects, vertex level | r = -0.08; with the streamline count, r = 0.48 |
| second principal direction, vertex level | r = -0.60 |
| first principal direction, region level | r = 0.09; with the streamline count, r = -0.45 |
| second principal direction, region level | r = 0.54 |

The largest source of difference between these subjects' connectomes is how
many streamlines tractography produced, 0.54 to 1.84 million per subject,
which correlates with age at only r = -0.08; the unit-mass normalization
removes its scale but not its effect on the shape of the density. Age is the
next direction. At rank 20 (`candidates=1`, 31% of the norm) five components
track age at FDR 0.05 with sex and the count as covariates, components 14 and
11 most strongly (r = 0.32 and 0.28, adjusted p 4e-12 and 2e-9); without the
count nine do, but the first two components correlate with the count at
-0.24 and -0.30 and with age at 0.06. So on real data the rank has to reach
past the acquisition's own variation, and its measure belongs in the design.

The rank-4 fit also exposed a memory cost: `reduce` collected every subject's
dense matrix in a list and then stacked them, so the cohort was held twice at
the peak, 278 GB for 528 subjects. It now fills one preallocated array, and a
sequence that loads each connectome when indexed is read one subject at a
time: the rank-20 fit peaked at 121 GB. Basis, scores, scales, explained
fractions and mean of a six-subject ico4 cohort are bitwise identical to the
previous code, as objects and as arrays, with both starts. On Longleaf the
fit ran on one core because eight one-CPU tasks leave `OMP_NUM_THREADS=1`;
four hours at rank 20. One task with eight CPUs gives the linear algebra its
threads.

## 6. Spherical kernel -- DONE, r = 1.000000 AT FULL SCALE

- **From:** [`dcmoyer/concon`](https://github.com/dcmoyer/concon), C++, MIT.
  Invoked by `sbci_step5_structural.sh` as `c3_main Compute_Kernel --sigma
  ${BANDWIDTH} --epsilon 0.001 --final_thold 0.000000001 --OPT_VAL_num_harm 33
  --OPT_VAL_exp_num_kern_samps 6 --OPT_VAL_exp_num_harm_samps 5`.
- **To:** `src/sbci/smoothing.py`, reached through `cc.smooth(kernel="shk")`.
  Against the correct reference the port reaches **r = 1.00000000** at full
  scale (see below); the 0.65 this file reported for most of its life was
  measured against the wrong input.

### The verification that was missing, and now exists

Every earlier comparison used `mesh_intersections_ico4.mat` as the input and
`smoothed_sc_avg_0.005_ico4.mat` as the target. The pipeline script shows those
come from different branches, so they were never the same data.

Five ADNI subjects under `/overflow/zzhanglab/ADNI/ADNI-bids/` carry the real
pairing, from one run in one directory:

| File | What it is |
| --- | --- |
| `subject_xing_sphere_avg_coords.tsv` | exactly what `c3_main` was fed: 1,001,877 streamlines, both endpoints as continuous unit-sphere positions |
| `smoothed_sc_avg_0.005_ico4.mat` | exactly what `c3_main` wrote |
| `SBCI_AVE/lh_grid_avg_ico4.vtk` | the grid it was handed -- matches the bundled ico4 sphere to 6.0e-12 |

Recomputing the density from those endpoints at sigma = 0.005 with 33
harmonics:

| Measure | Value |
| --- | --- |
| correlation over the upper triangle | **0.977757** |
| rank correlation | 0.912600 |
| restricted to the support `c3_main` kept | 0.978489 |
| best global scale | 6.52 |
| relative error after scaling | 0.206 |

**The global scale** of 6.5 was a normalization convention: that draft
normalized each endpoint's kernel column to sum one, while `convert_raw.py`
applies no normalization at all -- its normalizing block is commented out in
the source. The shipped port follows `convert_raw.py` and needs no scale.

**The 20% residual after scaling is compact support.** An earlier draft of this
section said "the support difference is not the issue", on the grounds that
this port places only 0.43% of its mass outside the *released file's* support.
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

Applying a window that vanishes at the measured cutoff, against the ADNI
subject's own endpoints (40,000 sampled of 1,001,877) and its released matrix:

| kernel | r | drift | 2-6 deg | 6-12 | 12-25 | 25-60 | 60-180 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no window (the draft at the time) | 0.9747 | 1.86 | 0.97 | 0.88 | **0.65** | **0.52** | 0.56 |
| times `1 - u^3` | 0.9882 | 1.36 | 1.04 | 1.05 | 1.13 | 1.24 | 0.91 |
| times `1 - u^4` | **0.9891** | 1.37 | 1.02 | 1.03 | 1.07 | 1.12 | 0.82 |

where `u = theta / (2.9 sqrt(sigma))` and "drift" is how far the ratio against
the released file wanders across the distance bands -- a flat ratio means the
shape is right, which correlation alone does not test.

The 12-60 degree deficit that no bandwidth and no threshold could remove is
gone: 0.65 and 0.52 become 1.07 and 1.12. The remaining error halves, `1 - r`
falling from 0.0253 to 0.0109. The windows above are approximations chosen to
test the diagnosis, not the real taper, so the residual overshoot at 12-60
degrees and the shortfall beyond 60 are what identifying the true form should
remove.

(The support counts are not comparable in that table: it samples 40,000
streamlines where the released file used 1,001,877, so this port's density is
sparser for that reason alone.)

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
correlation sat at 0.978 for months.

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

### Full scale, all 1,001,877 endpoints

Against what `c3_main` wrote for `sub-168S6561`, with the heat kernel the
package used to implement as a control:

| | shipped kernel | heat kernel |
| --- | --- | --- |
| correlation | **1.000000** | 0.981784 |
| best scale | **0.99858** | 155.83 |
| relative error after scaling | **0.00007** | 0.18980 |
| relative error, no rescaling | **0.00142** | 0.99381 |
| drift across distance bands | **1.000** | 1.756 |
| ratio by band | **1.000 throughout** | 0.872 ... 0.497 |
| non-zero pairs | 7,671,633 | 13,125,126 |

against the released file's 7,664,821. The scale factor is 0.99858 with
nothing fitted, where the old kernel needed 6.5 on this subject and 155.8 once
the normalization by streamline count was applied correctly. Correlation is
1.000000 over all 13,125,126 pairs.

The port kept 0.09% more pairs than `c3_main` until `apply_final_threshold`
reproduced `--final_thold 1e-9`, the per-streamline value below which the
reference writes nothing; what remains (+0.071%) sits on the kernel's cutoff
boundary -- *Shipping it*, below.

### The released files carried the other branch's endpoints

The HCP-Aging example files, as first released, stored the pipeline's smoothed
connectome with endpoints read from `mesh_intersections_ico4.mat`: the
unsnapped branch described below (*Confirmed from the pipeline itself*), not
the snapped streamlines the connectome was smoothed from. On an HCP-Aging
subject the two sets are a median 1.6 degrees apart, only 38% of endpoints
fall in the same grid triangle, and the stored endpoints re-smoothed correlate
with the stored connectome at r = 0.968. A user re-smoothing a released file
got a different connectome from the one it stored.

`Endpoints.from_snapped` builds the right set. Each end of a snapped streamline
is a vertex of the subject's native white surface, and the registered sphere
(`?h_sphere_reg_lps.vtk`, one vertex per native vertex) says where it lands;
streamlines with an end on any other surface are dropped, as
`intersections_to_sphere.py` drops them. On `sub-HCA6924080` this reproduces
the pipeline's own `subject_xing_sphere_avg_coords.tsv`, the file `c3_main`
smoothed, for every one of 903,797 streamlines to within 0.001 degrees, and
the two give the same connectome through the package's smoother to eight
decimals. Against the pipeline's stored connectome that connectome correlates
at r = 0.99974, with either order of medial-wall masking and normalization;
the kernel verification above reached 1.000000 on ADNI, so the remaining
difference is in how that HCP-Aging run smoothed, and is not yet explained.

`tools/build_hcp_cohort.py` took its endpoints this way for both layouts it
then built, HCP-Aging (retired with that data on 30 September; the `--layout`
option went with it) and the HCP Young Adult subjects of the ENCORE project,
whose FreeSurfer-registered spheres are stored in RAS and negated in x and y to
reach the grid's frame (as stored they sit 121 degrees from the subject's
native LPS sphere on median, flipped 12 to 14, against 120 and 8 for the
HCP-Aging pair, which is how the orientation was read). `--check` re-smooths
each file's endpoints and compares them with its stored connectome.

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
| share of endpoints on the medial wall | 0.1% to 0.3%, as in HCP-Aging |
| correlation between two young adults' connectomes | 0.67 to 0.71 |
| correlation with HCP-Aging subjects' connectomes | 0.63 to 0.69, against 0.68 between HCP-Aging subjects |
| distance from a seed vertex to the peak of its profile | a median of about 12 degrees, as in HCP-Aging |

All 946 young adults with complete pipeline output were built the same way,
44 GB, and an audit read every file back. All 946 pass `sbci validate` and
carry positioned endpoints, with 599,553 to 1,071,113 streamlines each (median
823,732), and every connectome correlates with the mean of five released
subjects at 0.73 or more (median 0.82).

The medial wall takes a median 0.21% of a subject's endpoints, but 67 subjects
put more than 1% there and 22 more than 4%, up to 7.9% (sub-211316). In the
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

It scores r = 0.972 rather than 1.000000, and that is the input file, not the
port. Both endpoint sources on the **same 200,000 rows**, so sampling noise is
identical and only the endpoints differ:

| endpoint source | correlation |
| --- | --- |
| `subject_xing_sphere_avg_coords.tsv`, what `c3_main` was fed | **0.998193** |
| `mesh_intersections_ico4.mat` | 0.969720 |

The two disagree by a median 1.25 degrees, but with a tail: 3.14 at the 90th
percentile and 42 at the maximum. Displacing the correct endpoints uniformly by
1.25 degrees costs almost nothing (0.99263 to 0.99235), so it is that tail of
relocated endpoints that accounts for the gap, not the typical difference.

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

**The 0.142% amplitude offset is explained and removed.** `c3_main` never
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
:class:`sbci.smoothing.KernelTable` is that emulation and is the default;
`quantized=False` gives the closed-form series the tables approximate.

At full scale, all 1,001,877 endpoints of `sub-168S6561` against the matrix
`c3_main` wrote, the two paths side by side:

| | tables (default) | closed form |
| --- | --- | --- |
| correlation | **1.00000000** | 1.00000000 |
| best scale | **1.000000** | 0.998584 |
| relative error, nothing rescaled | **0.000021** | 0.001420 |
| pairs equal to 1e-6 relative | **2,100,505** | 32 |
| ratio by distance band | 1.00000 throughout | 0.9985 throughout |

The residual is 2e-5 relative, at the level of the binary's own arithmetic:
the dot products are formed in a different order, and one ULP moves a
truncation bucket.

The **excess of non-zero pairs** is mostly `--final_thold`: `compute_kernel.cpp`
writes a pair only `if(temp > final_thold)` after dividing by the streamline
count, and `smooth()` now applies the same rule on the same scale. At full
scale that takes the tables' 7,676,193 pairs to 7,670,250 against the
released 7,664,821 -- **+0.071%**. What remains sits on the cutoff boundary,
where one ULP in a dot product moves a vertex across `dot >= cutoff_distance`.

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
each block. Full scale on `sub-168S6561`: **53 s** for the density, and
`cc.smooth(kernel="shk")` end to end in **54 s**, against 3,184 s for the dense
exact path and 6,114 s before the in-place recurrence -- about 110x.

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

## 7. ConSEAL -- DONE, VERIFIED AGAINST MATLAB; FOUR ERRORS IN THE REFERENCE, CORRECTED BY DEFAULT

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
  against it in `strict_upstream` mode.

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
recovers the fork's update rule (not its finite-difference derivative).

### What matches the MATLAB reference

`conseal_compare.py` on the reference dump, port in `strict_upstream=True`
mode, float64 against MATLAB's `single` kernels:

| Stage | Agreement (max abs, relative to the largest entry) |
| --- | --- |
| Voronoi areas, frames, tangent basis, divergence (`strict_upstream=True`; item 14 below) | 3e-16 to 6e-15 (float64 rounding) |
| heat kernel `K`, cutoff included | 4.1e-8 |
| kernel derivatives `dK.x/y/z` | 1.3e-6 |
| adjacency (float64 here, float32 there) | 2.0e-4 of the largest entry, 2.0e-9 absolute |
| `F`, `F_e1`, `F_e2` | 9e-6, 2.9e-5, 3.1e-5 |
| `Q`, `Q_e1`, `Q_e2` | 2.9e-5, 8.4e-5, 9.7e-5 |
| gradient and step, both hemispheres | 6.1e-7, 6.6e-7 |
| warp vertices after one composed step | 6.6e-10 |
| velocity field, Jacobian | 1.6e-16, 3.7e-9 |
| endpoints carried along by the warp | 2.9e-5 (single-precision coordinates in MATLAB) |
| cost after one step | 0.102993884 against 0.102993876 |
| cost trace, six registration iterations | 7e-8 to 4e-7 relative at every iteration |
| Karcher-median template, five iterations | 2.1e-5 |

Every stage agrees to the precision MATLAB carries it in. The two closest-point
searches picked the same triangle for every one of 240,000 endpoints.

### What was found in the reference

The port reproduces all of these under `strict_upstream=True`; by default it
corrects 1 to 4 and 12 to 17 (5 and 8 are choices and stay, 6, 9 and 11 are
fixed in both modes, 7 only reaches the bandwidth selection). Numbering
matches the module docstring.

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
   fork has neither. Both are reproduced here because they are what the public
   code does.
9. `get_template` counts subjects with `size(Fs, 1)`, so a row cell array
   silently uses one subject; and `acos(trace(Q Q_mu))` is taken without a
   clip, which rounding above 1 turns complex in MATLAB.
10. The shipped scripts do not run: `README.md`, `simulation.m` and
    `real_data_registration.m` construct the abstract `Concon`;
    `random_diffeomorphism.m` calls a `compose_warp` that does not exist;
    `simulation.m` copies `lh_warp` where it means `rh_warp`.
11. Kernels, frames and the adjacency are `single`; `build_adjacency.cpp`
    accumulates a million outer products in float32.
14. **The tangent basis shares ENCORE's Legendre recurrence**, with its wrong
    m = 0 term (item 4, *Three more*; item 8; the module docstring's items 12
    and 13 are the cotangent weights and the square root's chain rule, below
    under *The gradient, measured*). The basis fields are right,
    their divergences are not, and the gradient uses them.
    `strict_upstream=True` reproduces it with the other four errors; the
    default grids carry the corrected basis. Found by the review of
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
    `strict_upstream=True` reproduces 15 to 17 with the rest.

### The gradient, measured

`tests/reference/conseal_gradient_probe.py` differentiates the *actual* cost --
move every endpoint along a field, relocate, re-smooth, compare -- by central
differences and sets it against the analytic coefficients `dE/dbeta_k` of the
port and of the reference (`strict_upstream`). The kernel derivatives
themselves are checked in `tests/test_conseal.py`: the port's matches a
finite difference of `K~(i, x)` in the evaluation point to 1e-5, the strict
one matches a finite difference of the reference's own definition to 1e-5,
and the connectome derivative `D_e1` matches a finite difference of `F(x, c)`
in `x` to 2e-4 of its range.

Along each method's own descent direction, `dE/dt` by finite differences over
the port's first-order prediction:

| Grid, settings | port's direction | reference's direction | cosine between the two gradient fields (lh, rh) |
| --- | --- | --- | --- |
| ico2, sigma 0.05, degree 12, order 3, 3,000 streamlines | 1.050 | 1.058 | 0.969, 0.828 |
| ico3, same, 20,000 streamlines | 1.138 | 1.139 | 0.855, 0.923 |
| ico4, sigma 0.005, degree 30, order 15, 100,000 streamlines (the paper's) | 1.019 | 0.986 | 0.716, 0.691 |

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
basis and is unchanged from the earlier run.) The port's gradient points 11
degrees from the exact gradient of the discrete cost, the remainder being the
transported-density approximation and the mesh; the reference's points 47
degrees away and buys two thirds of the decrease per step. Both descend,
which is why the reference works at all:
its large components are right (cosine 0.94 on the twenty largest) and its
errors live in the smaller ones.

### On ico4, real data

*Measured before the correction of the shared divergence (item 8, finding
2). On two released young adults the corrected gradient moves ConSEAL's
aligned endpoints by 0.02 degrees on average after three iterations (item
8, *What the corrections change*).*

`tests/reference/conseal_adni_ico4.py` aligns three ADNI subjects
(`sub-168S6561`, `sub-068S0473`, `sub-002S0413`: 1,001,877, 1,034,758 and
937,465 streamlines) from their `mesh_intersections_ico4.mat` at the public
defaults, on 8 cores:

| | sub-168S6561 | sub-068S0473 | sub-002S0413 |
| --- | --- | --- | --- |
| cost, initial to final | 0.07128 to 0.05616 (-21.2%) | 0.05210 to 0.04193 (-19.5%) | 0.07049 to 0.05548 (-21.3%) |
| iterations to the 1e-4 threshold | 29 | 20 | 27 |
| refused steps | 0 | 0 | 0 |
| cost trace monotone | yes | yes | yes |
| Jacobian range | 0.37 to 2.06 | 0.56 to 1.78 | 0.49 to 1.77 |
| distance to the template, before to after | 0.268 to 0.238 | 0.229 to 0.205 | 0.266 to 0.236 |

Pairwise distances between the subjects' square-root densities fall from
0.433, 0.453 and 0.432 to 0.380, 0.400 and 0.385; endpoint-overlap
coefficients at a density threshold of 1e-4 rise from 0.761, 0.775 and 0.798
to 0.777, 0.786 and 0.802. The heat kernel at the published bandwidth has 89
nonzeros per row (cutoff 21.5 degrees); loading and locating three million
endpoints took 16 s and the template plus three registrations 1,616 s
together, about 21 s per iteration for a million streamlines. Every Jacobian
is positive and no step was refused, so the fold check never fired. The warps
and the aligned endpoints are in `/work/users/x/y/xya/conseal-ref/`.

### How to run it

```python
import sbci
subjects = [sbci.load(p) for p in paths]           # files that carry endpoints with positions
result = sbci.endpoints_align(subjects)             # the public code's defaults
result.warps[0].save("sub-001_conseal_warp.npz")
aligned = result.aligned_endpoints(0)               # an Endpoints object, ready to re-smooth
```

`strict_upstream=True` reproduces the reference to the digits above;
`init_rotation=True` adds the multi-shell rigid search (its schedule is
regenerated from `create_search_schedule.m`, 80/20/20/20/16 rotations, and the
60 icosahedral rotations are fitted to the grid rather than assumed, and the
densities are transported by interpolation, so it also runs on the bundled
FreeSurfer sphere, which is an icosphere to 1.7e-4 -- the precision of its
stored coordinates).
On ico4 an iteration costs a few seconds per 100,000 streamlines: the kernel
has 89 nonzeros per row at the published bandwidth (cutoff 21.5 degrees), so
`K' A K` is a sparse sandwich (its sparse-times-dense products run on all the
cores the job was given, in row blocks that reproduce the single-core result
bit for bit), and relocating the endpoints is a k-d-tree
query.

### Measured on a known deformation

*The synthetic measurements in this subsection were made before the review
of 5 October 2026 corrected the divergence both gradients use (item 8,
finding 2); the figure's numbers on sub-100307 were regenerated with the
corrected code and came out the same to the digits quoted, ENCORE in eleven
steps rather than ten.*

The reference's own tests compare registrations of unrelated subjects, which
says the cost falls but not how far the warp is from the right one. So: a
synthetic subject's endpoints were moved by a known smooth warp (a random
combination of the warp basis fields, 1.0 degrees on average and 3.6 at most,
`scripts/make_figures.py`), the deformed copy re-smoothed, and registered back
onto the original. What matters is where the endpoints end up:

| setting | iterations taken | endpoint residual (from 0.98 deg) | cost removed |
| --- | --- | --- | --- |
| public defaults (`threshold=1e-4`) | 10 | 0.64 deg, 34% undone | 72% |
| public defaults, `threshold=1e-7` | 60 | 0.47 deg, 52% | 89% |
| public update, `delta=0.2`, `threshold=1e-7` | 60 | 0.27 deg, 72% | 97% |
| the fork's update (`delta=0.1`, no clamp, no viscosity), `threshold=1e-7` | 60 | 0.17 deg, 82% | 99% |

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

The same test on a real subject (an HCP-Aging subject, 903,797 streamlines; a
run from before that data were retired, kept as the record) at first
seemed to separate the methods. With the same degree-15 warp (0.95 degrees on
average, 4 at most), ConSEAL with the paper's update and threshold 1e-7
reached 0.13 degrees, 86% undone, while ENCORE halved its cost with a warp of
the right size but only half the right direction (cosine 0.50 with the true
field) and brought the endpoints from 0.95 to 0.92 degrees; step lengths from
0.05 to 1.0 gave the same result. Two flaws in that experiment, not in the
port, produced it (`tools/encore_probe.py` makes the measurements):

- **The reference was the pipeline's stored density, and the endpoints stored
  beside it are not the ones it was smoothed from.** The files took their
  endpoints from `mesh_intersections_ico4.mat`, the pipeline's unsnapped
  branch, while the density comes from the snapped streamlines (item 6, *The
  released files carried the other branch's endpoints*). The smoother was
  never the difference: the pipeline's own endpoint coordinates re-smooth to
  the stored density at r = 0.9997. The stored density sits at a cost of 0.028
  from the stored endpoints re-smoothed, out of a starting cost of 0.039; the exact inverse warp, Jacobian included, lowers the cost
  only to 0.036, and undoing the warp exactly at the endpoint level and
  re-smoothing still costs 0.028. Even ConSEAL's warp, which agrees with the
  true field at cosine 0.93, scores 0.035 in that cost. ENCORE's halving was
  mostly fitting the difference between the two endpoint sets. With the
  reference re-smoothed from the same endpoints as the deformed copy the start
  is 0.009, the exact endpoint-level undo costs 0.0000,
  and ENCORE undoes 58% of that warp (cosine 0.79).
- **A warp with structure at the grid scale is one the smoothed density cannot
  see.** Deforming the endpoints and then smoothing is not smoothing and then
  deforming once the local stretch is large, so even against a matching
  reference the true degree-15 warp costs 0.0095 from a start of 0.0090; no
  optimizer of that cost can be expected to find it.

On a smooth warp the picture is what it should be. A random field of degree
4 (1.66 degrees on average, 4 at most) takes the matching-reference cost from
0.022 to 0.005 at the true warp, and ENCORE with its default degree-6 basis
undoes 87% of it in eight steps: the endpoints come back from 1.66 to 0.21
degrees, the cost to 0.18 of its start, and the field it finds agrees with the
truth at cosine 0.96. A degree-15 basis reaches the same 86% in 43 steps; the
analytic derivative changes nothing; a warp of 2.8 degrees on average is
undone 87% and a degree-2 warp 94%; against the stored reference the same
smooth warp is undone 65%. ConSEAL on the smooth warp reaches 0.10 degrees, 94% undone, in 60 iterations. The
figure in docs/RESULTS.md uses the smooth warp and the matching reference
(`scripts/hcp_figures.py`: `WARP_ORDER`, `WARP_AMPLITUDE`, `ENCORE_ORDER`),
on sub-100307: ENCORE brings the endpoints from 1.63 to 0.20 degrees, 88%,
with the cost at 0.15 of its start; ConSEAL to 0.11 degrees, 93%.
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

| cohort | unaligned | after ENCORE (10 it) | after ConSEAL, paper's update, Karcher median (30 it) | after ConSEAL, public update, Karcher median | after ConSEAL, paper's update, mean template |
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
the "not found" above is the collapsed template's.* And the paper's update (no clamp, no viscosity)
onto that one subject reshapes the bundles enough to remove a fivefold weight
difference, while the same update onto the mean of the square-root densities
keeps it, as does the public code's clamp and viscosity onto the one subject.
With the reference start instead of `candidates=6`, alignment *lowers* the
chance of finding the bundle at rank 4 (six seeds, spreads 0.05, 0.04 and
0.03: unaligned 0, 4 and 6 of 6; after ENCORE 1, 0 and 4 of 6; after the
paper's ConSEAL 0, 0 and 1 of 6), because the aligned cohort's Gram spectrum
is flatter still. Rank 8 with the reference start finds it unaligned in 5, 6
and 6 of 6 and after ENCORE in 4, 6 and 6 of 6; after the paper's ConSEAL in
1, 3 and 6 of 6.

### Still open

- The port follows the public code. Reproducing the paper's experiments
  exactly would need the fork's finite-difference derivative and direct warp
  composition, which are ENCORE's machinery (`alignment.py`) driving endpoint
  transport; wiring that combination is a small job if it is wanted.
- The rigid search's cost surface is interpolated on the grid; on ico2 it
  lands a few degrees from a known rotation (test tolerance 0.12 rad). On ico4
  this is a fraction of a degree but has not been measured against a known
  rotation on real data.
- Bandwidth cross-validation is ported in both forms and untested against
  MATLAB, since the paper does not use it.

## 8. The review of 5 October 2026 -- ELEVEN FINDINGS, ALL CONFIRMED; THREE WERE THE REFERENCES' OWN

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
| 1 | `project()` scores differ from the fit's | `ConConSmooth.smooth` fits the lower triangle with the diagonal; the fit contracts the whole matrix (**reference**) | the fit's own scoring; `reference=True` keeps the triangle | factor 1/(1 + sum psi^4): two thirds on two vertices, 0.04% on ico4 |
| 2 | wrong m = 0 term in the Legendre derivative recurrence | `legendre_2nd_derivative.m` applies the unnormalized relation to normalized functions (**reference**, shared by ENCORE and ConSEAL) | the normalized relation; `reference=True` / `strict_upstream=True` keep the reference's | m = 0 derivative times (1 + 1/(l(l+1)))/2; zonal divergences up to twice too large; the basis fields unchanged |
| 3 | `ConSEAL.register` resets an object that was already warped | the copy kept the older committed locations | `commit()` on the copy | endpoints moved 0.05 on the unit sphere while the trace read 0 |
| 4 | ENCORE accepts a map that folds | acceptance by cost alone; the Jacobian is an absolute value (**reference**) | a folding trial is rejected, always | 22 of 80 faces at step 2; none in 40 trials at the default step |
| 5 | the transported square-root density is not of unit norm | normalized before its diagonal is zeroed (`Concon.m`, **reference**) | zero first; `reference=True` keeps the order | 0.02% to 0.2% on ico4; 5% on a coarse, very local toy |
| 6 | the areas do not weight the first component | the Gram enters only the deflation (**reference**) | documented; the method decision stays with WP1 | twice the area-weighted residual on a four-vertex toy |
| 7 | `seed(region=)` divides by masked area | weights over every selected vertex | intersect with the mask; refuse an all-masked region | 0.8327 of the cortical-only profile on sub-100307's PALS left limbic lobe |
| 8 | an intercept-adjusted F test changes when the response is shifted | the untestable rule judged on the uncentred scale | the centred scale when the reduced model can reproduce a constant: an intercept column, and since 6 October dummy codes for every level (below) | F = 10.9 became NaN after adding 1e6 |
| 9 | the within-region mean counts each vertex paired with itself | denominator A_k^2 | mean over distinct pairs; a one-vertex region gets NaN | 0.8(1 - 1/n): 4% low for Schaefer-sized regions |
| 10 | GIFTI arrays read by position | `darrays[0]`, `darrays[1]` | by intent | a triangles-first file gave zero faces |
| 11 | the FC exchange sidecar says density | one string for both modalities | by modality | provenance only |

Items 1, 2 and 5 are the references' own. The port reproduced them to
rounding, which is what the agreement tables above measured; it now computes
the right thing by default and keeps the reference arithmetic behind
`reference=True` (ENCORE and `project`) and `strict_upstream=True` (ConSEAL,
whose default grids then carry the reference basis), so that the Tier 4
comparisons still run. Item 4 is corrected without a switch: a fold is never
a diffeomorphism, and at the default step the check never fires on the
recorded runs. The reviewer's diagnostic scripts were not adopted; each
invariant has its own test in the suite -- the harmonic derivatives against
finite differences and the divergence against `-l(l+1) Y`, unit norm after
transport, no accepted fold, a pre-warped object registered onto itself,
the training cohort projecting onto its own scores, a region mean that does
not depend on the grid, shift invariance, both GIFTI orders, the FC sidecar.

### What the corrections change on the released subjects

Recorded with `tests/reference/record_outputs.py` on two of the released
young adults, sub-100307 and sub-103010 (`reduce` and `project` on four), on
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
  4e-4 and 1.1e-3 relative (0.012707 against 0.012712 for sub-100307), in 15
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
- **Region seed.** sub-100307's PALS left limbic lobe has 170 cortical and 34
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
| 1 | ConSEAL's Karcher median collapses onto its starting subject | **confirmed**: the square-root densities are not renormalized, float32 barycentric weights put a subject's squared norm 1e-10 off 1, and when the starting subject's is below 1 the `1e-14` snap misses it, that subject's Weiszfeld weight is about 7e4 and the first step is already shorter than 0.005 (**the reference's arithmetic**, item 7 docstring item 15) | unit-norm densities and a 1e-6 radian coincidence guard; `strict_upstream=True` keeps the reference's | on the eleven released subjects the median started from sub-212116, whose squared norm rounds 1.7e-10 below 1, and stopped 0.001 degrees from it and 24-27 from the rest; it now sits 0.43 degrees from their mean and 16.6-20.2 from every subject. The synthetic cohort that collapsed at three degrees of spread had its starting subject 1.4e-10 below 1, the one that did not at two degrees 5.1e-10 above |
| 2 | ConSEAL's default grids have four vertices on the poles | **partly**: the vertex at the north pole moves 2.0 degrees of a 4-degree shift of the endpoints around it, but the endpoints, carried by their triangles, land within 0.09 degrees of the truth either way; the cause is the 5% velocity smoothing, which averages the two frame components as scalars across neighbours whose frames turn 72 degrees around a pole -- after a 20-degree `rotate()`, ten zero-size steps move the two most polar vertices 8.8 degrees and one 11.3, on the unrotated grid and on one rotated off the poles alike (7.5 and 9.9); the reference smooths the same way (docstring item 16) | the velocity field is smoothed as vectors: first as ambient 3-vectors projected back onto the tangent planes, and since the third list (item 10) with the connection Laplacian, which parallel-transports each neighbour's vector to the vertex; `strict_upstream=True` keeps the component-wise smoothing; the grids stay in the file's frame, since a grid rotated off the poles moves the same vertex 3.7 degrees and lands its endpoints worse (0.10 against 0.035) | the ten-step drift at the poles falls to 0.09 degrees with the ambient smoothing and 0.046 with the connection Laplacian (the 5% smoothing of a degree-1 field, as everywhere else); the pole vertex follows 3.45 of the 4 degrees and the endpoints around it land 0.035 degrees from the truth (0.085 before); within 10 degrees of the coordinate equator the connection Laplacian agrees with the reference's smoothing to 1e-4 of the field |
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
-- it is now calibrated by the identity's own value (`reference=True` keeps
the bias); `parcellate` zeroes the vertex diagonal it assumes zero and treats
an empty region as NaN throughout under `"mean"`; `smooth("shk")` refuses a
bandwidth that is not positive; the kernels refuse eigenvalues that are not
ascending; `from_snapped` refuses negative vertex ids; `TemplateWarp.apply`
refuses a hemisphere that is not L or R, and every `save` returns the path
that exists; `save()` refuses a suffix `load()` would not accept; an
interrupted download keeps its `.part` file and resumes; `validate` fails
whatever `load` refuses, which it did not for a misnamed `.mat` file, a
coordinate array of the wrong shape, a mask or area of the wrong length or a
`/metadata` dataset that is not a string (two checks added, eleven in all);
`strict_upstream` says which items it reproduces (1 to 4 and 12 to 17);
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
degrees, 0.17 and 0.089 against 0.001. `strict_upstream=True` keeps the reference's arithmetic.

### What the corrections change on the released subjects

Recorded with `tests/reference/record_outputs.py` on an AMD EPYC 9654 node
and set against the first review's recording on the same CPU type with
`tests/reference/compare_outputs.py`, key by key:

- **Everything but the two aligners' defaults is identical, bit for bit**:
  parcellation (the new diagonal and empty-region rules change nothing on
  Desikan and Schaefer-200), the three couplings, seeding, smoothing, the
  FPCA fit and both projections; and so is ENCORE under `reference=True`.
- **ENCORE**: the calibrated Jacobian keeps the aligned density's mass,
  0.70% more of it than before; the warped vertices move 0.001 degrees on
  average (0.003 at most) and the costs 3e-5 to 6e-5 relative. The
  known-warp recovery figure is unchanged to the digits it quotes.
- **ConSEAL**: the vector smoothing moves the aligned endpoints of the two
  subjects 0.001 degrees on average, ten times that within 10 degrees of
  the coordinate poles (0.009) and 0.085 at most. The two-subject template
  did not move: it sits a fifth of the way from sub-100307 to sub-103010
  (5.6 and 22.4 degrees) under both arithmetics.
- **The template of the eleven** (`tests/reference/template_probe.py`, on
  all of them, default against `strict_upstream=True`; it also runs the two
  synthetic cohorts of item 7): the reference's arithmetic stops on
  sub-212116, 0.001 degrees from it and 16.9 from the mean; the corrected
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
| 8 | the vector smoothing shrinks rotations twice as fast as the reference | **partly**: twice as fast only about the coordinate axis -- 95.2% of a rotation field left after 100 smoothings against the reference's 98.9% -- while about a tilted axis the reference shrinks it faster (90.5%), its rate depending on the frame (item 16 of the ConSEAL docstring). The ambient smoothing's own rate came from a curvature term: projecting a neighbour's vector instead of transporting it adds the sphere's Ricci curvature, doubling the damping of rotation fields | the connection Laplacian, which parallel-transports each neighbour's vector | 97.6% left for every axis, exp(-5 * 4 pi / 2562); within 10 degrees of the coordinate equator it agrees with the reference's smoothing to 1e-4 of the field, 3 to 20 times closer than the ambient one |
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
| findings 1, 3 and 6 | open, as item 8's *Still open* records: the scores the fit records one step early, the re-registration drift of 0.001 degrees, the area weighting | unchanged | -- |

New in the third round's commits, and from the audit:

| # | Finding | Verdict | Fixed by | Size |
| --- | --- | --- | --- | --- |
| 1 | a missing family label in a plain list is not refused | **confirmed**: NumPy turns `np.nan` in a list of strings into the string `'nan'`, one more family | the labels are judged as given, on an object view, before NumPy converts them | 21 groups instead of 20 in the review's case; now refused |
| 2 | the constant-column error numbers the caller's column, not `terms=`'s; refusing a column of zeros breaks stratified runs | **confirmed** | the message gives both numberings; a column of zeros is kept, fits as nothing and is set aside by the clustered test, and testing one is refused in words | -- |
| 3 | global coupling keeps an empty one-vertex region | **confirmed**: its NaN diagonal made a constant profile look varying | constancy judged on the entries present, NaN aside; bit for bit on matrices without NaN | on sub-100307 in Schaefer-1000, the other regions moved by up to 0.005; now not at all |
| 4 | the 0.5 cutoff of the Jacobian calibration skips coarse grids | **confirmed**: the 12-vertex icosahedron reads 0.463 for the identity at every vertex | the calibration skips only vertices within `AXIS_CLEARANCE` of the axis, and refuses a vertex clear of it where the scheme reads nothing positive | the identity's Jacobian 0.463 -> 1 exactly; the bundled grids unchanged |
| 5 | HDF5: triangle indices split at 5,120 faces a hemisphere; an area of the wrong length reported as an endpoints error | **confirmed** | triangles split at `2 V - 4`, a closed spherical mesh's faces (5,120 at ico4's 2,562 vertices); endpoints split at the connectivity's own count, the area left to `load` and `sbci validate`, which name it | -- |
| 6 | `StationaryWarp.invert()` with a rotation is not exactly reversible; assigning `warp.rigid` changes nothing | **confirmed**: the inverse carried the field to the rotated frame by interpolation, and `rigid` was a plain attribute | the warp records which of its two parts comes first, so the inverse -- the flow of `-v`, then `R'` -- is computed, not interpolated; `rigid` is a property that checks the rotation and re-flows | a double inversion changed the field by 1.0e-3 on ico4 and 4.3e-3 on ico3 (2.8% in the review's case); now 0, bit for bit |
| 7 | documents: item 16's "1e-4"; a passage on the superseded `local_test` rule; `write_cifti` refusing `.DCONN.NII` | **confirmed** | item 16 restated by axis, with the cotangent layout's share; item 9's row marked superseded; file endings compared without case by the exchange file, its companions and `save_map` | 1.7e-4 about the coordinate axis, 1.3e-3 to 1.5e-3 about others, nearly all the layout's |

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

## Status

All seven ports are done and verified, and the four reviews of 5 and 6
October 2026 (items 8 to 11) have been answered in full; what is left is
under each item's *Still open*. They were done in the order 3, 1, 2, 5, 4, 6, 7: parcellation
unblocked the first notebook, kernel smoothing the WP3 speed target, and the
two alignments came last because nothing else depends on them.
