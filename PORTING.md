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

## 1. Riemannian diffusion and Matern kernel smoothing -- DONE, VERIFIED

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

### The public API path, end to end

Steps 3 and 4 validated the kernel through a hand-written density loop. Driving
`ContinuousConnectome.smooth(kernel="shk")` instead -- building a connectome
from `mesh_intersections_ico4.mat`, calling the method, saving, validating --
is what found all three bugs above, and the face-order one could not have been
found any other way.

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
  within-region connectivity, which the port computes.

- **Open divergence:** the triangle and diagonal conventions above. Confirm
  which behavior the released files should carry -- it changes every published
  region matrix, so it is a decision, not a detail.

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
| Basis Laplacian | 5.9 eps |
| Tangent frames, exponential and logarithm maps | 0.5 to 4 eps |
| Barycentric query against the AABB tree | 2.0 eps, same triangle 162/162 |
| Identity warp Jacobian | exact |
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
mesh before the frames are built; `align()` refuses a grid that still has
poles. Worth reporting upstream: the reference's own demo grid happens to avoid
the poles, which is why this has not bitten anyone.

### Still open

- The registration loop descends a gradient built from the unstable
  derivative, so two implementations take different steps. Agreement is
  therefore reported as a correlation, not a tolerance.
- **Grid:** no constraint. `SphericalGrid(mesh, l)` takes the mesh as an
  argument; only the reference's demo scripts hardcode its retired 0.94 grid,
  and this package uses the ico4 sphere throughout.

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
  conservative than the FDR correction. **This should be reviewed by whoever
  specified the API**, because the choice of test is mine, not the
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
and scores. Fifteen further tests in `tests/test_reduction.py` check what holds
without MATLAB: exact recovery of genuinely low-rank data, ordering by weight,
monotone explained variance, and scores that reproduce their own subjects.

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
| Voronoi areas, frames, tangent basis, divergence | 3e-16 to 6e-15 (float64 rounding) |
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
corrects 1 to 4. Numbering matches the module docstring.

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
| ico2, sigma 0.05, degree 12, order 3, 3,000 streamlines | 0.955 | 0.961 | 0.964, 0.851 |
| ico3, same, 20,000 streamlines | 1.083 | 1.077 | 0.861, 0.939 |
| ico4, sigma 0.005, degree 30, order 15, 100,000 streamlines (the paper's) | 0.873 | 0.794 | 0.833, 0.747 |

Both directions descend; the port's prediction of its own descent rate is
within 13% on the paper's settings. Per coefficient the reference is
unreliable: on ico4 its coefficients for the eight sampled basis fields are off
by factors of 0.28 to 2.7 with two sign flips, against 0.25 to 2.1 for the port
on the same fields (the transported-density model behind both is itself only
first order on a mesh). The definitive measurement is the exact gradient of the
discrete cost, all 510 left-hemisphere coefficients by finite differences:

| Quantity | port | reference |
| --- | --- | --- |
| cosine with the exact gradient, all 510 coefficients | **0.931** | 0.678 |
| cosine on the 20 largest exact coefficients | 0.996 | 0.941 |
| norm, relative to the exact gradient | 1.050 | 0.753 |
| descent efficiency: first-order decrease per unit step, as a fraction of the exact gradient's | 0.931 | 0.678 |

(ico4, sigma 0.005, degree 30, order 15, 100,000 streamlines per subject;
1,020 cost evaluations, 83 minutes on 8 cores.) The port's gradient is the
gradient of the cost up to the transported-density approximation and the
mesh; the reference's points 47 degrees away from it and buys two thirds of
the decrease per step. Both descend, which is why the reference works at all:
its large components are right (cosine 0.94 on the twenty largest) and its
errors live in the smaller ones.

### On ico4, real data

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

## Status

All seven items are ported and verified; what is left is under each item's
*Still open*. They were done in the order 3, 1, 2, 5, 4, 6, 7: parcellation
unblocked the first notebook, kernel smoothing the WP3 speed target, and the
two alignments came last because nothing else depends on them.
