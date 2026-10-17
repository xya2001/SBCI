# How to verify this package independently

Everything here is checkable by someone who did not write it. This document
says exactly how, what each check proves, and — as importantly — which claims
**cannot** be checked yet and should be treated with suspicion until they are.

Work through the tiers in order. Tier 1 needs nothing but the repository and
takes minutes. Tier 4 needs a MATLAB licence and a few hours. A reviewer who
stops after Tier 2 has still confirmed most of what matters.

---

## The short path: one command

If you only do one thing, do this:

```bash
bash scripts/verify_all.sh            # tiers 1, 2 and 4, against the MATLAB references present
bash scripts/verify_all.sh --matlab   # also regenerate the MATLAB references
```

It runs the automated tiers, 1, 2 and 4 (tier 3's checks against lab data are
run by hand), and prints PASS, FAIL or SKIP per stage. **A SKIP is not a pass**
-- the summary lists what each skipped stage needs. On a compute node with the
young adult download, the toolkit and the MATLAB output present, expect
`8 passed, 0 failed, 0 skipped`.

The tiers below explain what each stage proves, and are worth reading before
trusting the summary.

## Tier 1 — anyone, in five minutes

```bash
git clone https://github.com/xya2001/SBCI.git && cd SBCI
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q                 # expect: everything passes; 72 skip with every extra installed, more without (below)
ruff check src tests scripts tools && ruff format --check src tests scripts tools
python -m build --wheel
```

**What this proves.** The package installs from a clean environment, and every
behaviour that can be checked without external data holds: file round trips,
metadata refusals, atlas region counts, grid arithmetic, surface anatomy, the
derivative operators against closed forms, alignment and reduction properties,
and the statistics in `stats.py`.

**What it does not prove.** Nothing here compares against the original MATLAB.
A port can be self-consistent and still wrong; that is what Tier 4 is for.

The skips, and what turns each on: `tests/test_five_minute_start.py` (three
tests) downloads the released tutorial subject only with `SBCI_DOWNLOAD=1`, or
reads it from `SBCI_HCP_DIR`; the MATLAB comparisons in
`tests/test_matlab_reference.py` (21) need the Tier 4 reference dumps, named
by the `SBCI_*` paths at the top of that file; the planted-answer analysis on
the ico4 grid in `tests/test_example.py` (three minutes of FPCA) runs with
`SBCI_SLOW_TESTS=1`, which CI's `render` job sets; thirteen doctest items
are whole examples marked `+SKIP`; the GPU's thirty-four in `tests/test_gpu.py`
need CuPy and a CUDA GPU (Tier 3); and without the plotting and render extras the
figure tests in `tests/test_surface.py`, `test_plot_mesh.py` and
`test_render.py` skip as well. With a network connection, two more checks run
on the released young adults:

```bash
SBCI_DOWNLOAD=1 python -m pytest tests/test_five_minute_start.py   # the brief's acceptance lines on sub-100307, 90 MB
python scripts/check_hcp_ya.py --out hcp-ya --accept-terms          # all eleven, the main methods; under ten minutes on four cores
```

The first is the acceptance criterion, which CI's `five-minute-start` job
also runs from a blank environment on every push. The second downloads the
cohort as a user would, checks every file against its digest, and runs
parcellation, seeds, smoothing, a surface figure, ENCORE, ConSEAL,
`migrate_warp`, `reduce` and `local_test` on it, stopping at the first check
that fails. `--accept-terms` accepts the HCP's data use terms for the machine,
which is for someone who has read them; without it, a machine that has not
accepted them stops at the download.

### Checks worth reading rather than just running

Three tests encode claims about the world rather than about the code, and are
the ones most likely to catch a silent regression:

| Test | What it would catch |
| --- | --- |
| `tests/test_surface_anatomy.py` | a scrambled mesh: it asserts the superior frontal gyrus is anterior to lateral occipital cortex, the cuneus medial, the medial wall facing the midline. An earlier build passed every self-consistency check and was still anatomically wrong |
| `tests/test_alignment.py::test_the_gradient_of_a_linear_field_is_exact` | a derivative that is subtly not a derivative: `f(x) = x` must differentiate to exactly 1 |
| `tests/test_cifti.py::test_area_weighted_mass_is_conserved_on_the_real_operator` | the resampling bug that cost 99.3% of the connectivity mass while preserving a plausible-looking total |

---

## Tier 2 — with the toolkit checked out, about an hour

Clone the four reference repositories and point the tests at them:

```bash
git clone https://github.com/sbci-brain/SBCI_Toolkit
git clone https://github.com/sbci-brain/SBCI_Pipeline
git clone https://github.com/sbci-brain/ConCon_Alignment
git clone https://github.com/sbci-brain/SBCI_Modeling_FPCA

sbci download hcp-ya --out hcp-ya          # the tutorial's eleven subjects, about 1 GB; asks once for the HCP's terms
export SBCI_TOOLKIT=$PWD/SBCI_Toolkit SBCI_HCP_DIR=$PWD/hcp-ya
python scripts/audit_api.py
```

`scripts/audit_api.py` exercises the methods of the README's capabilities table on the released
young adults, in the order a user would: load, save, parcellate, seed,
couple structure with function, plot, export, validate, download, smooth,
reduce, test, align with ENCORE and ConSEAL, carry a warp to fs_LR, gather a
cohort, score a held-out subject, test a named design and a contrast, split
streamlines for test-retest, and write maps and region tables: 22 checks (the synthetic example,
`sbci info` and `sbci atlases` are covered by the unit tests instead). Its
paths default to the lab's on Longleaf and follow `SBCI_HCP_DIR` and
`SBCI_TOOLKIT`. It prints a value for each and exits non-zero if any fails.
Expect `22 passed, 0 failed, 0 skipped`. The exchange-file row reads the
16.9 GB file `scripts/write_exchange_file.py` writes into `SBCI_EXCHANGE_DIR`
(Tier 3), which the release does not hold: without it, expect `21 passed, 0
failed, 1 skipped`. It smooths, reduces and aligns on the full grid, so run
it inside a batch job.

**What this proves.** The documented API works end to end on a real subject,
not only on fixtures.

---

## Tier 3 — with real cohort data

Two checks need more than one subject.

**Alignment must make subjects more alike.** A registration can run, reduce its
own cost, and leave the cohort no more similar than before — which would make it
useless. `tests/reference/align_adni_ico4.py` measures mean pairwise
correlation between subjects before and after, on the five ADNI subjects the
pipeline produced on ico4 (an earlier check on the lab's HCP test-retest
tensors, which exist only on the retired 4121-vertex grid, gave +0.023; that
script was retired with the grid). On the five, as recorded on 22 September
2026, before items 8, 9 and 21 of PORTING.md changed ENCORE's gradient, the
scale of its Jacobian and its order of composition:

```
before 0.737617   after 0.781972   change +0.044354   (min pair 0.682 -> 0.720)
costs 0.142, 0.126, 0.042, 0.112, 0.185
Jacobians: every one strictly positive, smallest 0.6146;
           per subject lh in [0.61, 1.38], rh in [0.63, 1.46]
```

Both criteria held on the package's own grid: the cohort got more alike, and
no warp folded. With today's code the eleven released young adults, aligned
ten steps onto ENCORE's median, rise from a mean pairwise correlation of
0.7163 to 0.7615 (PORTING.md item 25).

Re-run it on a different cohort. **The correlation must go up and every
Jacobian must stay positive**; a negative Jacobian means the warp has folded and
the result is not a diffeomorphism.

**What reproduces bit for bit, and across which machines.** The heavy
methods' outputs on the released subjects (parcellation, coupling, seeding,
smoothing, `reduce` on four subjects, ENCORE and ConSEAL on two) were recorded
four times on the cluster, before and after the October review's changes. On
nodes whose BLAS runs the same instruction set, every array is identical run
to run and across the change: the AVX2 nodes (EPYC 7702 and 7763) agree with
each other on all of them, and the AVX-512 nodes (Intel Gold 6140 and EPYC
9654) agree with each other to one ULP on one explained-variance value.
Between the two groups the rounding differs and the methods differ as their
arithmetic allows: parcellation, coupling, seeding and smoothing still bit for
bit (FC parcellation to 5.6e-17); `reduce` to 1.4e-16 on the basis; ENCORE to
5e-16 on the aligned densities and 5e-8 on the warp's vertices after ten
iterations. ConSEAL on two subjects without `template=` started its Karcher
median from subject 0 in one group and from subject 1 in the other, since the
two are equally near their mean and rounding decides, and the median of two
points is any point between them (it now ends a fifth of the way from the
starting subject to the other; until the fix of PORTING.md item 9 it could
also stop on the starting subject outright); its cost traces and warps are
therefore comparable across machines only with `template=` given or a third
subject in the cohort. A verifier comparing
against the numbers in these documents should expect rounding-level
differences on other hardware, and exact agreement only on the same
instruction set. After the corrections of 5 October 2026 (PORTING.md item 8)
the recording was repeated with `tests/reference/record_outputs.py`, which
until 11 October 2026 also ran ENCORE and ConSEAL in the reference
arithmetic: on the same CPU type that reproduced the earlier recording bit
for bit, and `tests/reference/compare_outputs.py` reports, array by array,
what the corrections change. Since PORTING.md item 21 there is no reference
arithmetic to run, and the recorder holds the defaults only. Since PORTING.md item 17 the reduction
scores the vectors it returns, with no switch back, so its arrays differ from
earlier recordings too. Since PORTING.md item 19 ConSEAL's
defaults are the paper's settings, and since 11 October 2026 its kernel's
width with a floor under the threshold; the recorder pins its `conseal_*` run
to the settings the earlier recordings used, the public code's, so that a
key holds the same measurement in every recording, records the paper's
settings at the public code's bandwidth under `consealpaper_*` and the
present defaults under `consealnarrow_*`, every one with its kernel and floor
spelled out.

**The exchange file must survive a round trip.**
`scripts/write_exchange_file.py` writes the 16.9 GB `.dconn.nii`, reads it back
with plain `nibabel`, and checks that the area-weighted unit mass survives and
that parcellating the fsLR file with the pipeline's *own* fsLR annotation
reproduces the ico4 region matrix. Needs about 25 GB of memory.

```
unit mass 0.999999999985 -> 0.999999999876   (1.1e-10)
Desikan on fsLR against ico4: r = 0.999642
```

Open the result in Connectome Workbench as the final word:

```bash
module load connectome/1.5.0     # 2.0.1 on Longleaf is missing libglapi.so.0
wb_command -file-information <the .dconn.nii>
```

**A cohort check must account for every subject.** `sbci cohort` checked the
structure of the 946 young adults' 1,892 files in 37 seconds, and read the
eleven released subjects' files in full, with every check of `sbci validate`,
in 34 (`/work/users/x/y/xya/hcp-ya/cohort-check.sbatch`):

```
the eleven, --validate             11 subjects seen; 11 in the cohort (sc, fc)
the 946, with the open-access      946 subjects seen; 943 in the cohort (sc, fc)
  traits and the motion table        left out, 3: no value for fluid_intelligence_pmat24
```

The 943 are the subjects with a score that docs/RESULTS.md analyses, and
every SC and every FC file agrees on every setting of `sbci.cohort.SETTINGS`.

**Exported maps must open where they are meant to.**
`tests/reference/export_probe.py` writes sub-100307's maps with `save_map`
and reads them back in FreeSurfer 7.4.1 and Workbench 1.5.0, each check
beside a control, the same values in the grid's own order, that has to fail:

```
GIFTI Desikan labels against FreeSurfer's fsaverage4 aparc.annot   lh 0.969, rh 0.964 agree   (control 0.062, 0.060)
mri_surf2surf fsaverage4 -> fsaverage, against the grid's correspondence   r = 1.0000   (control r = 0.054)
wb_command -cifti-separate on the .dscalar.nii   the file's own values; 2,874 of 32,492 left vertices missing (the wall)
area-weighted mean, map written whole   grid 7.993534e-12, fsLR 7.993534e-12
region_means of each region's seed, against to_atlas(how="mean")   all 68: 1.8e-15 relative; 340 empty pairs read 0
```

With the medial wall left out, the default, the mean moves slightly instead:
the fsLR vertices along the wall's edge take the mean of their cortical part
(USAGE.md, *Exporting maps and tables*). The labels agree on 96% rather than
on every vertex because the bundled
atlas gives each grid vertex the majority label of the fsaverage vertices it
stands for, and FreeSurfer's fsaverage4 annotation gives the label at the
vertex itself; the two differ along region borders. The order is bundled by
`tools/build_fsaverage4_order.py`, which derives it two independent ways that
agree at every vertex, and the suite checks by digest that it carries the
grid's triangles onto fsaverage4's. Run the probe with Workbench called by its
path: `module load connectome/1.5.0` puts an old `libstdc++` ahead of NumPy's,
and Python then fails to import it (the probe's docstring has the commands).

**On a GPU, the alignments must give what the CPU gives.** With CuPy
(`pip install 'sbci[gpu]'`) on a node with a CUDA GPU:

```bash
python -m pytest tests/test_gpu.py
SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py encore
SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py conseal
```

The tests hold every step of both alignments to the CPU: the point search
and the relocated endpoints bit for bit, the transport, the densities and
their derivatives, the roots and the templates to rounding. A whole
registration they hold to the CPU's own spread -- no farther from the CPU's
result than twenty times the farther of two of the CPU's twins, its results
when one input moves by one unit in the last place, or, where both happen to
lie near, ten times the farthest of ten such twins of that quantity
(`gpu_probe.py spread`, job 4685821) -- since rounding decides which face
carries a point on an edge, and the descent goes its own way from there. Run
twice, the GPU must give the same bits, and the rotation search of
`init_rotation` must pick the CPU's rotations, bit for bit. Expect `39 passed`
(a V100, a GTX 1080 and an L40S, jobs 4722437, 4722468 and 4806991); without a
GPU, `5 passed,
34 skipped`, as in CI. Without the CUDA toolkit loaded, CuPy sees the GPU but cannot compile its
kernels, and `device="gpu"` is refused saying so. The probe sets the GPU
beside the CPU on two of the released subjects and prints both distances. On
a V100 (job 4707448), ENCORE's registration ends 2.7e-8 of its cost trace from
the CPU's and the CPU's twin 3.7e-8; ConSEAL's ten iterations 2.0e-9 of their
costs, the twin 4.0e-9, with the endpoints at most 3.9e-5 radians from the
CPU's, the twin 3.5e-5; the endpoints are relocated bit for bit through the
same warps; and every GPU result is the same the second time, bit for bit.

---

## Tier 4 — with MATLAB, the real verification

This is the tier that decides whether the ports are right. The `*_reference.m`
scripts in `tests/reference/` regenerate the MATLAB references and the tests
diff against them, skipping if they are absent; the other files there are
hand-run probes and the helpers they use, and the document that quotes each
probe's measurement names it: PORTING.md for most, this one, SPEC_QUESTIONS.md
and docs/review-2026-10-05.md for the rest.

```bash
module load matlab/2023b
cd tests/reference

matlab -batch "run('reference_run.m')"         # kernels, density, adjacency
matlab -batch "run('sfc_reference.m')"         # the three coupling functions
matlab -batch "run('parcellate_reference.m')"  # parcellate_sc.m at full scale
matlab -batch "run('seed_reference.m')"        # seed rows and a region marginal
matlab -batch "run('encore_reference.m')"      # ENCORE, every intermediate
python fpca_make_inputs.py                      # shared inputs for FPCA
matlab -batch "run('fpca_reference.m')"        # ConConBasis.Fit
matlab -batch "run('fpca_fixed_point.m')"      # the same, run to its fixed point
matlab -batch "run('conseal_reference.m')"     # ConSEAL on a synthetic ico4 example
python conseal_compare.py PATH/conseal_reference.mat  # the dump just written: the ConSEAL row below
cd ../..

export SBCI_TOOLKIT=... SBCI_MATLAB_REFERENCE=... SBCI_ENCORE_REFERENCE=... SBCI_FPCA_REFERENCE=...
EXAMPLE=$SBCI_TOOLKIT/example_data
python tools/import_legacy.py --subject example --out derivatives \
    --sc $EXAMPLE/SBCI_Individual_Subject_Outcome/smoothed_sc_avg_0.005_ico4.mat \
    --endpoints $EXAMPLE/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat \
    --mapping $EXAMPLE/fsaverage_label/mapping_avg_ico4.npz
export SBCI_EXAMPLE_SC=derivatives SBCI_DERIVATIVES=derivatives
python -m pytest tests/test_matlab_reference.py -v    # expect 21 passed
```

The scripts name the lab's directories on Longleaf at their tops -- the
toolkit, `SBCI_Modeling_FPCA`, the tensor toolbox, and the folders they write
to -- with no environment override, so set those to your own copies first.
Then point the `SBCI_*` variables at what the scripts wrote. Unset, the
variables default to the lab's copies, so on Longleaf the tests run as they
stand; elsewhere a test whose variable is unset skips.

Expected agreement, and what to reject:

| Port | Expected | Reject if |
| --- | --- | --- |
| `to_atlas` | 1.9e-16, float64 rounding | worse than 1e-12 |
| `seed` | float32 rounding | worse than 10 float32-eps |
| `coupling` (3 forms) | 3.6 to 88.2 float64-eps | worse than 1e-12 |
| `smooth` (rdk) | 3.14 float32-eps (`smooth_matlab_probe.py`) | worse than 10 float32-eps |
| `smooth` (shk) | r = 1.00000000, scale 1.000000 against the released matrices | see PORTING.md item 6 |
| `endpoints_align` (ConSEAL), the reference's errors put back | the single precision the MATLAB reference carries: kernel 4e-8, gradient 5.7e-7 (left) and 1.2e-6 (right), the six-iteration cost trace 3.5e-7, the template 2e-5 (`tests/reference/conseal_compare.py`) | see PORTING.md items 7 and 22 |
| `reduce` | 1e-15 on every component at the reference's fixed point; at its default tolerance the first vector to 2e-16, with that vector's scores (item 11 below) | worse than 1e-12 |
| `align` geometry, basis and divergence, template, warp composition | 0.5 to 44 float64-eps, the basis and divergence with the reference's two errors put back | worse than 1e-12 |
| `align` Jacobian, `evaluate`, `evaluate_root` | 4.9e-4, 7.1e-4, 5.4e-4: inside the reference's own instability at its 1e-10 step (item 4 below) | worse than 1e-2 |

The references' errors matter. They carry eight mathematical errors (items 5
to 12 below) that the port shared until October 2026, and the port now
computes the right thing, with no switch back since 11 October 2026
(PORTING.md items 21 and 22). The comparisons put the reference's errors back
into the port's output instead:
- the clamp at the poles and the Legendre recurrence go into the basis and its
  divergence (`tests/reference/reference_geometry.py`);
- for ENCORE, the uncalibrated Jacobian and the normalization order go into
  the warp quantities;
- for ConSEAL, the kernel derivative, the density's derivatives, the square
  root's chain rule, the cotangent layout, the smoothing and the median go in
  (`tests/reference/reference_conseal.py`, the code `strict_upstream=True`
  ran).
ENCORE's registration is not compared, since it composes each step in its own
order (item 12). ConSEAL's six-iteration cost trace is, stepped in the
reference's arithmetic (3.5e-7; PORTING.md item 23). `project` has no MATLAB
row. The `reduce` row has no switch (item 11): it is met at the reference's
fixed point.

### Twelve things in the references that a verifier will hit

The first four are not port defects. Anyone reproducing this will meet them,
and should not conclude the port is broken. The last eight are errors in the
references that the port reproduced until the reviews of 5 to 11 October 2026;
it now corrects them, with no switch back:

1. **`ConConBasis.Fit` cannot run as published.** Line 260 reads `auto_sparse`,
   which is never defined; the parsed option is `params.auto_sparse`. Every call
   fails. `tests/reference/fpca_reference.m` uses a copy with that one line
   fixed and nothing else changed.
2. **ENCORE's `register` returns the wrong cost** — from the step it rejected,
   not from the warps it returns (0.0925 against 0.0899).
   `encore_cost_check.m` demonstrates it.
3. **ENCORE zeroes any vertex on the coordinate axis**, because the Jacobian
   closes with `sin(theta)`. The ico4 grid has four such vertices, so
   `align()` rotates the mesh clear first; a grid supplied with a vertex on
   the axis or within 1e-3 of it is turned clear too, the turn recorded in
   every warp, and refused only if none of the turns tried clears it.
4. **ENCORE's derivative does not converge.** Run `encore_delta_sweep.m`: it
   moves by about 0.037 between every adjacent pair of step sizes without
   settling, because at a vertex the interpolant has a kink and the measured
   secant depends on which face libigl's tree returns. This port sits 0.0367
   from the reference — the same distance the reference sits from itself
   (`encore_first_checks_probe.py`). **No
   independent implementation can do better**, so what depends on it is
   compared only to within that instability.
5. **The Legendre derivative recurrence has a wrong m = 0 term.**
   `legendre_2nd_derivative.m` builds `P_l^{-1}` as `-P_l^1 / (l(l+1))`, the
   relation for *unnormalized* functions, from the *normalized* ones, for
   which `P_l^{-1} = -P_l^1`. The m = 0 derivative comes out a factor
   `(1 + 1/(l(l+1)))/2` too small: 0.75 at degree 1, 0.502 at degree 15. The
   normalized basis fields are unaffected; the divergence of every zonal
   field is too large by the inverse factor, and both registration gradients
   use it. `tests/test_alignment.py` checks every order against finite
   differences and the divergence against `-l(l+1) Y`.
6. **The transported square-root density is normalized before its diagonal is
   zeroed** (`Concon.m`), so it is not the unit vector the cost assumes: on
   ico4 the norm is short by 0.02% to 0.2% for realistic warps, 5% on a
   coarse grid with very local connectivity. The port zeroes first.
7. **`ConConSmooth.smooth` projects onto the lower triangle with the
   diagonal**, while the fit scores by contracting the whole matrix. The two
   differ by the factor `1/(1 + sum_i psi_k(i)^4)`: two thirds on a two-vertex
   toy, 0.04% on a smooth ico4 component. The port's `project()` now scores a
   new subject exactly as the fit scored the training cohort.
8. **ConSEAL's `get_template` collapses onto its starting subject** when that
   subject's square-root density has a squared norm just below 1 -- which
   float32 barycentric weights (and the reference's own single precision)
   make routine: the `1e-14` snap misses it, the subject's Weiszfeld weight
   is about 7e4, and the first step is already shorter than the stopping
   length. The port normalizes the densities and uses a 1e-6 radian
   coincidence guard (PORTING.md item 9).
9. **ENCORE's finite-difference Jacobian of the identity warp is 0.9965**, not
   1, a discretization bias that costs the transported density 0.7% of its
   mass; the port calibrates by the identity's own value.
10. **ConSEAL holds a rigid rotation as a velocity field** (`rotate`, used by
    `init_rotation=True`): its flow is 1.6 degrees off at 150 degrees, and
    every later smoothing erodes it, 3.8 of 150 degrees over 100 steps. The
    port holds the rotation exactly, outside the field (PORTING.md item 9).
11. **`ConConBasis.Fit` keeps each component's scores from the vector before
    its last update**, and deflates with them, so the scores it returns are
    not those of the basis it returns: 2.4e-5 apart on the reference run, and
    1.6% to 2.0% of the largest score on the prediction notebook's rank-15
    fits of 240 young adults. Its own sparse branch takes the scores again
    after it moves the vector. The port scores the vector it returns, with no
    switch back (PORTING.md item 17); run to its fixed point
    (`fpca_fixed_point.m`) the reference agrees with it on every component to
    1e-15.
12. **ENCORE composes each step after the warp**, moving each image by the
    step interpolated where it lies, while its gradient is taken for the step
    applied first. The two agree at the identity and part as the warp grows:
    after a 30-degree rotation the cost falls at 0.79 of the rate the gradient
    predicts, against 0.99 composed as the gradient assumes, and a known
    20-degree turn of a real subject is left 1.30 degrees from the truth
    against 0.56. The port composes as the gradient assumes (PORTING.md
    item 21).

---

## What cannot be verified, and should be challenged

A reviewer should push on these rather than accept them.

**Agreement with MATLAB is not correctness.** Tier 4 proves that the port
computes what the reference computes; it cannot catch an error the two share.
The reviews of 5 to 7 October 2026 found seven (items 5 to 11 of the Tier 4
list: the Legendre recurrence, the normalization order of the transported
square-root density, the projection objective, the template's collapse onto
its starting subject, ENCORE's Jacobian of the identity, ConSEAL's rotation
held as a velocity field, and the fit's scores from the vector before its
last update), each reproduced here to rounding and each wrong; on 11 October
an eighth, ENCORE's order of composition (item 12). All are corrected, with
no switch back (PORTING.md items 21 and 22), and the comparisons put the
reference's errors back into the port's output.
What now stands between the port and a further such error is the independent
invariants in the suite -- derivatives against finite differences and the
divergence against `-l(l+1) Y`, unit norm after transport, no accepted fold,
a pre-warped object registering onto itself without moving, the training
cohort projecting onto its own scores, a region mean that does not depend on
the grid, an F test that does not change when the response is shifted -- and
a reviewer should look for the invariant the suite does not yet assert rather
than re-run the agreement. PORTING.md items 8, 9, 17 and 21 list the findings
and their sizes.

**`sbci.stats.local_test` has no reference at all.** Neither
`SBCI_Modeling_FPCA` nor the toolkit contains inference code. The choice of
test — an F test per component, then Benjamini-Hochberg — is the porter's, not
the method authors'. It is verified against `scipy.stats` and against the
procedures' own guarantees (null uniformity by Kolmogorov-Smirnov, false
discovery held in simulation, power rising with effect size), which establishes
that the statistics are correctly *implemented*, not that they are the right
statistics. **Ask whoever specified the API to confirm the intent.** For
related subjects, `groups=` swaps in cluster-robust standard errors with the
families as clusters, which agree with statsmodels to 1e-9 and come near the
nominal false-positive rate in simulation with hundreds of families (5.6% at
0.05 with 400), not with dozens (9.7% with 20; PORTING.md item 5, *Related
subjects*, and item 33). docs/RESULTS.md shows
what it changes on the 943 young adults; this page, which names subjects in its
commands, quotes nothing a family table went into.

**The spherical kernel ships, and is verified against the binary itself.**
`kernel="shk"` is the default by the WP1 decision and it works. It reproduces
`c3_main` at **r = 1.000000** over all 13,125,126 pairs at full scale on
`sub-168S6561`, the ADNI subject under `/overflow/zzhanglab/ADNI/ADNI-bids/`
whose input to `c3_main` and output from it were both kept, with the scale
factor at 1.000000 and nothing fitted (0.99858 for the closed-form series,
`quantized=False`). The kernel was read from `concon`'s source -- public MIT code, named by
the binary's debug info -- and confirmed by running `c3_main` on a single
streamline so that its output is the kernel: `tests/reference/concon_probe.py`
regenerates that measurement, and `tests/test_smoothing.py` pins the kernel to
the binary's own numbers at rms 0.0005.

One residual remains: 0.071% more non-zero pairs, where one ULP in a dot
product moves a vertex across the kernel cutoff (PORTING.md item 6). The 0.14%
amplitude offset an earlier version carried was the binary's lookup-table quantization
and is reproduced. A reviewer can re-run the comparison from PORTING.md item 6
(*Full scale*); it needs the lab data and about a minute. The other four ADNI
subjects with pipeline output on ico4 were checked for unit mass and used for
the alignment check, not for a second full-scale kernel comparison.


**The format is a draft.** `SPEC_VERSION` is `0.1.0-draft` and
`SPEC_QUESTIONS.md` lists what is unratified; PORTING.md item 3 records the
one divergence that changes every published region matrix, the triangle
convention of `parcellate_sc.m` (its diagonal convention was decided in
October 2026: the mean over distinct vertex pairs). Files written before
these are settled may need rewriting.

**The young adults' FC is built here, not by the pipeline.** The lab's copy
of the cohort holds no pipeline FC, so the FC files of the eleven, and of the
young adults analysed, come from the HCP's ICA-FIX-cleaned resting-state runs
through `tools/build_hcp_fc.py`, which follows the pipeline's nuisance model
and FC definition as far as those data allow (USAGE.md, *Functional
connectivity from the HCP's resting state*). With no pipeline FC of these
subjects to compare against, the files are checked against what resting-state
FC must show, not against a reference: grid cells coherent in time,
homotopic and default-network correlation, and coupling that falls from
sensory to association cortex (PORTING.md item 2). The coupling code itself
matches the MATLAB reference (Tier 4).

---

## A reviewer's checklist

- [ ] Tier 1 passes from a clean clone: tests, lint, wheel
- [ ] `scripts/check_hcp_ya.py` passes on a machine with network access
- [ ] `scripts/audit_api.py` reports 22 passed, 0 failed, 0 skipped
- [ ] Tier 4 reproduces the agreements in the table above
- [ ] Alignment raises inter-subject correlation on a cohort of your choosing
- [ ] The exchange file opens in Connectome Workbench
- [ ] You are satisfied with the choice of test in `stats.local_test`, or have
      replaced it
- [ ] WP1 has ratified the open items in `SPEC_QUESTIONS.md`, and
      `SPEC_VERSION` has lost its `-draft` suffix
