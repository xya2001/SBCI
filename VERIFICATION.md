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
bash scripts/verify_all.sh            # tiers 1 and 2
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
git clone <this repository> && cd sbci
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q                 # expect: everything passes; 33 skip with every extra installed, more without (below)
ruff check . && ruff format --check src tests scripts tools
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
`tests/test_matlab_reference.py` (22) need the Tier 4 reference dumps, named
by the `SBCI_*` paths at the top of that file; the planted-answer analysis on
the ico4 grid in `tests/test_example.py` (three minutes of FPCA) runs with
`SBCI_SLOW_TESTS=1`, which CI's `render` job sets; seven doctest items are
whole examples marked `+SKIP`; and without the plotting and render extras the
figure tests in `tests/test_surface.py`, `test_plot_mesh.py` and
`test_render.py` skip as well. With a network connection, two more checks run
on the released young adults:

```bash
SBCI_DOWNLOAD=1 python -m pytest tests/test_five_minute_start.py   # the brief's acceptance lines on sub-100307, 90 MB
python scripts/check_hcp_ya.py --out hcp-ya                         # all eleven, every method; under ten minutes on four cores
```

The first is the acceptance criterion, which CI's `five-minute-start` job
also runs from a blank environment on every push. The second downloads the
cohort as a user would, checks every file against its digest, and runs
parcellation, seeds, smoothing, a surface figure, ENCORE, ConSEAL,
`migrate_warp`, `reduce` and `local_test` on it, stopping at the first check
that fails.

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

sbci download hcp-ya --out hcp-ya          # the eleven released subjects, about 1 GB
export SBCI_TOOLKIT=$PWD/SBCI_Toolkit SBCI_HCP_DIR=$PWD/hcp-ya
python scripts/audit_api.py
```

`scripts/audit_api.py` exercises the API table in README.md on the released
young adults, in the order a user would: load, save, parcellate, seed,
couple structure with function, plot, export, validate, download, smooth,
reduce, test, align with ENCORE and
ConSEAL, and carry a warp to fs_LR: 17 checks (the synthetic example,
`sbci info` and `sbci atlases` are covered by the unit tests instead). Its
paths default to the lab's on Longleaf and follow `SBCI_HCP_DIR` and
`SBCI_TOOLKIT`. It prints a value for each and exits non-zero if any fails.
Expect `17 passed, 0 failed, 0 skipped`. It smooths, reduces and aligns on
the full grid, so run it inside a batch job.

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
script was retired with the grid). On the five:

```
before 0.737617   after 0.781972   change +0.044354   (min pair 0.682 -> 0.720)
costs 0.142, 0.126, 0.042, 0.112, 0.185
Jacobians: every one strictly positive, smallest 0.6146;
           per subject lh in [0.61, 1.38], rh in [0.63, 1.46]
```

Both criteria hold on the package's own grid: the cohort gets more alike, and
no warp folds.

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
iterations. ConSEAL on two subjects without `template=` landed its Karcher
median on subject 0 in one group and on subject 1 in the other, since the
median of two points is any point between them and rounding decides; its cost
traces and warps are therefore comparable across machines only with
`template=` given or a third subject in the cohort. A verifier comparing
against the numbers in these documents should expect rounding-level
differences on other hardware, and exact agreement only on the same
instruction set.

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

---

## Tier 4 — with MATLAB, the real verification

This is the tier that decides whether the ports are right. The `*_reference.m`
scripts in `tests/reference/` regenerate the MATLAB references and the tests
diff against them, skipping if they are absent; the other files there are
hand-run probes against lab data, and PORTING.md says which measurement each
made.

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
matlab -batch "run('conseal_reference.m')"     # ConSEAL, six iterations on the author's example
python conseal_compare.py                       # diffs the port against it (the ConSEAL row below)

export SBCI_MATLAB_REFERENCE=... SBCI_ENCORE_REFERENCE=... SBCI_FPCA_REFERENCE=...
export SBCI_EXAMPLE_SC=... SBCI_DERIVATIVES=...
python -m pytest tests/test_matlab_reference.py -v    # expect 22 passed
```

Expected agreement, and what to reject:

| Port | Expected | Reject if |
| --- | --- | --- |
| `to_atlas` | 1.9e-16, float64 rounding | worse than 1e-12 |
| `seed` | float32 rounding | worse than 10 float32-eps |
| `coupling` (3 forms) | 3.6 to 88.2 float64-eps | worse than 10 float32-eps |
| `smooth` (rdk) | 3.25 float32-eps | worse than 10 float32-eps |
| `smooth` (shk) | r = 1.00000000, scale 1.000000 against the released matrices | see PORTING.md item 6 |
| `endpoints_align` (ConSEAL) | the single precision the MATLAB reference carries: kernel 4e-8, gradient 6e-7, six-iteration cost trace 1e-7 | see PORTING.md item 7 |
| `reduce` | 2.1e-16 on the basis | worse than 1e-10 |
| `align` geometry, basis, template | 1.6 to 44 float64-eps | worse than 1e-12 |
| `align` registration | r = 0.99999979 | see the note below |

### Four bugs in the reference that a verifier will hit

These are not port defects. Anyone reproducing this will meet them, and should
not conclude the port is broken:

1. **`ConConBasis.Fit` cannot run as published.** Line 260 reads `auto_sparse`,
   which is never defined; the parsed option is `params.auto_sparse`. Every call
   fails. `tests/reference/fpca_reference.m` uses a copy with that one line
   fixed and nothing else changed.
2. **ENCORE's `register` returns the wrong cost** — from the step it rejected,
   not from the warps it returns (0.0925 against 0.0899).
   `encore_cost_check.m` demonstrates it.
3. **ENCORE zeroes any vertex on the coordinate axis**, because the Jacobian
   closes with `sin(theta)`. The ico4 grid has four such vertices, so
   `align()` rotates the mesh clear first and refuses a grid that still has
   them.
4. **ENCORE's derivative does not converge.** Run `encore_delta_sweep.m`: it
   moves by about 0.037 between every adjacent pair of step sizes without
   settling, because at a vertex the interpolant has a kink and the measured
   secant depends on which face libigl's tree returns. This port sits 0.0367
   from the reference — the same distance the reference sits from itself. **No
   independent implementation can do better**, so do not treat the registration
   agreement as a defect.

---

## What cannot be verified, and should be challenged

A reviewer should push on these rather than accept them.

**`sbci.stats.local_test` has no reference at all.** Neither
`SBCI_Modeling_FPCA` nor the toolkit contains inference code. The choice of
test — an F test per component, then Benjamini-Hochberg — is the porter's, not
the method authors'. It is verified against `scipy.stats` and against the
procedures' own guarantees (null uniformity by Kolmogorov-Smirnov, false
discovery held in simulation, power rising with effect size), which establishes
that the statistics are correctly *implemented*, not that they are the right
statistics. **Ask whoever specified the API to confirm the intent.** For
related subjects, `groups=` swaps in cluster-robust standard errors with the
families as clusters, which agree with statsmodels to 1e-9 and hold the
nominal false-positive rate in simulation with hundreds of families, not with
dozens (PORTING.md item 5, *Related subjects*). The README's analysis shows
why it matters: counted as 943 independent subjects, one component tracked
fluid intelligence (adjusted p 0.023); with their 422 families as clusters,
none does (0.064).

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
one divergence that changes every published region matrix, the triangle and
diagonal conventions of `parcellate_sc.m`. Files written before these are
settled may need rewriting.

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
- [ ] `scripts/audit_api.py` reports 17 passed, 0 failed, 0 skipped
- [ ] Tier 4 reproduces the agreements in the table above
- [ ] Alignment raises inter-subject correlation on a cohort of your choosing
- [ ] The exchange file opens in Connectome Workbench
- [ ] You are satisfied with the choice of test in `stats.local_test`, or have
      replaced it
- [ ] WP1 has ratified the open items in `SPEC_QUESTIONS.md`, and
      `SPEC_VERSION` has lost its `-draft` suffix
