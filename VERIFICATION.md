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

It runs every tier available in your environment and prints PASS, FAIL or SKIP
per stage. **A SKIP is not a pass** -- the summary lists what each skipped stage
needs. On a machine with the toolkit, the example subject and MATLAB output
present, expect `8 passed, 0 failed, 0 skipped`.

The tiers below explain what each stage proves, and are worth reading before
trusting the summary.

## Tier 1 — anyone, in five minutes

```bash
git clone <this repository> && cd sbci
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python -m pytest tests/ -q          # expect: 435 passed, 1 skipped
ruff check . && ruff format --check src tests
python -m build --wheel
```

**What this proves.** The package installs from a clean environment, and every
behaviour that can be checked without external data holds: file round trips,
metadata refusals, atlas region counts, grid arithmetic, surface anatomy, the
derivative operators against closed forms, alignment and reduction properties,
and the statistics in `stats.py`.

**What it does not prove.** Nothing here compares against the original MATLAB.
A port can be self-consistent and still wrong; that is what Tier 4 is for.

The one skip is `tests/test_five_minute_start.py`, which needs the data release
that does not exist yet (SPEC_QUESTIONS.md item 6). It should stay skipped.

### Checks worth reading rather than just running

Three tests encode claims about the world rather than about the code, and are
the ones most likely to catch a silent regression:

| Test | What it would catch |
| --- | --- |
| `tests/test_surface_anatomy.py` | a scrambled mesh: it asserts the superior frontal gyrus is anterior to lateral occipital cortex, the cuneus medial, the medial wall facing the midline. An earlier build passed every self-consistency check and was still anatomically wrong |
| `tests/test_alignment.py::test_the_gradient_of_a_linear_field_is_exact` | a derivative that is subtly not a derivative: `f(x) = x` must differentiate to exactly 1 |
| `tests/test_cifti.py::test_every_ico4_vertex_distributes_exactly_its_own_mass` | the resampling bug that cost 99.3% of the connectivity mass while preserving a plausible-looking total |

---

## Tier 2 — with the toolkit checked out, about an hour

Clone the four reference repositories and point the tests at them:

```bash
git clone https://github.com/sbci-brain/SBCI_Toolkit
git clone https://github.com/sbci-brain/SBCI_Pipeline
git clone https://github.com/sbci-brain/ConCon_Alignment
git clone https://github.com/sbci-brain/SBCI_Modeling_FPCA

export SBCI_TOOLKIT=$PWD/SBCI_Toolkit
python scripts/audit_api.py
```

`scripts/audit_api.py` exercises **every row of the API table in README.md** on
real data, in the order a user would: load, save, parcellate, seed, couple,
plot, export, validate, smooth, reduce, test, align. It prints a value for each
and exits non-zero if any fails. Expect `15 passed, 0 failed`.

**What this proves.** The documented API works end to end on a real subject,
not only on fixtures.

---

## Tier 3 — with real cohort data

Two checks need more than one subject.

**Alignment must make subjects more alike.** A registration can run, reduce its
own cost, and leave the cohort no more similar than before — which would make it
useless. `scripts/align_hcp_cohort.py` measures mean pairwise correlation
between subjects before and after. **Note the grid:** the lab's HCP
test-retest tensors exist only on the retired 4121-vertex `0.94` grid, so that
script is not an ico4 result and should not be read as one. The ico4 check is
`tests/reference/align_adni_ico4.py`, on the five ADNI subjects the pipeline
produced on ico4; its result is recorded below the HCP one. On eight HCP
subjects from `sbci_sc_tensor_1.mat` (4121-vertex grid):

```
before 0.831207   after 0.854531   change +0.023323
warps: every Jacobian strictly positive, vertices moved 0.67-2.23 deg on average
```

On the five ADNI subjects on **ico4** (`tests/reference/align_adni_ico4.py`):

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

**The exchange file must survive a round trip.**
`scripts/write_exchange_file.py` writes the 16.9 GB `.dconn.nii`, reads it back
with plain `nibabel`, and checks that the area-weighted unit mass survives and
that parcellating the fsLR file with the pipeline's *own* fsLR annotation
reproduces the ico4 region matrix. Needs about 25 GB of memory.

```
unit mass 1.00000000002 -> 0.999999999908   (1.1e-10)
Desikan on fsLR against ico4: r = 0.999729
```

Open the result in Connectome Workbench as the final word:

```bash
module load connectome/1.5.0     # 2.0.1 on Longleaf is missing libglapi.so.0
wb_command -file-information <the .dconn.nii>
```

---

## Tier 4 — with MATLAB, the real verification

This is the tier that decides whether the ports are right. Everything in
`tests/reference/` regenerates a MATLAB reference; the tests then diff against
it and skip if it is absent.

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

export SBCI_MATLAB_REFERENCE=... SBCI_ENCORE_REFERENCE=... SBCI_FPCA_REFERENCE=...
python -m pytest tests/test_matlab_reference.py -v    # expect 23 passed
```

Expected agreement, and what to reject:

| Port | Expected | Reject if |
| --- | --- | --- |
| `to_atlas` | 1.9e-16, float64 rounding | worse than 1e-12 |
| `seed` | float32 rounding | worse than 10 float32-eps |
| `coupling` (3 forms) | 3.6 to 88.2 float64-eps | worse than 10 float32-eps |
| `smooth` (rdk, matern) | 3.25 float32-eps | worse than 10 float32-eps |
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
statistics. **Ask whoever specified the API to confirm the intent.**

**The spherical kernel ships, and is verified against the binary itself.**
`kernel="shk"` is the default by the WP1 decision and it works. It reproduces
`c3_main` at **r = 1.000000** across all five ADNI subjects under
`/overflow/zzhanglab/ADNI/ADNI-bids/` that carry both the input `c3_main` was
fed and the matrix it wrote, with the scale factor at 0.99858 and nothing
fitted. The kernel was read from `concon`'s source -- public MIT code, named by
the binary's debug info -- and confirmed by running `c3_main` on a single
streamline so that its output is the kernel: `tests/reference/concon_probe.py`
regenerates that measurement, and `tests/test_smoothing.py` pins the kernel to
the binary's own numbers at rms 0.0005.

Two residuals remain, both reproducible to five digits across subjects and so
conventions rather than noise: a 0.14% amplitude offset and 0.08% more non-zero
pairs. A reviewer can re-run the five-subject comparison from PORTING.md item
6; it needs the lab data and about an hour per subject.


**The format is a draft.** `SPEC_VERSION` is `0.1.0-draft` and
`SPEC_QUESTIONS.md` lists what is unratified. Two of those questions — the
triangle and diagonal conventions in `parcellate_sc.m` — change every published
region matrix. Files written before they are settled may need rewriting.

**`sbci download` does not exist**, because the data release does not
(SPEC_QUESTIONS.md item 6). This is the only unimplemented row in the API table.

---

## A reviewer's checklist

- [ ] Tier 1 passes from a clean clone: 435 tests, lint, wheel
- [ ] `scripts/audit_api.py` reports 15 passed, 0 failed
- [ ] Tier 4 reproduces the agreements in the table above
- [ ] Alignment raises inter-subject correlation on a cohort of your choosing
- [ ] The exchange file opens in Connectome Workbench
- [ ] You are satisfied with the choice of test in `stats.local_test`, or have
      replaced it
- [ ] WP1 has ratified the open items in `SPEC_QUESTIONS.md`, and
      `SPEC_VERSION` has lost its `-draft` suffix
