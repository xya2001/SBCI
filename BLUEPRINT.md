# SBCI Python package — implementation blueprint

**Work package 2 of the Usability Implementation Brief.** A plan for the `sbci`
Python package: what it is, what "finished" means, what exists today, what
order the rest should be built in, and which decisions are holding it up.

## What this document asks of you

Most of the remaining work needs no input. **Five decisions do**, and they are
listed in section 9 with an owner and an estimated lead time against each.
Annex A gives each one in full, with the evidence and a recommendation.

One of the five should start immediately, because its answer comes from outside
this work and the project's acceptance criterion waits on it:

- **Q6 — may we redistribute derived dense connectomes openly?** This is a
  question for HCP, not for us. It decides whether the project's own acceptance
  test can pass as written. Nobody has asked yet.

The other four are ratifications rather than investigations: minutes of
discussion each, but everything written to disk depends on them.

**Q10, which smoother the project standardizes on, has been decided:** both
kernels ship, the choice is recorded in each file's metadata, and the spherical
heat kernel is the default. Released cohorts therefore stay canonical and
nothing is reprocessed. Annex A records the reasoning; section 7 records the
work it creates.

## Contents

1. What this package is
2. Definition of done
3. Architecture
4. Where things stand
5. The verification standard
6. What blocks what
7. Plan
8. Risk register
9. Decisions needed, with owners
10. Annex A — the open questions in full

[[pagebreak]]

## 1. What this package is

One installable library, `sbci`, that replaces five MATLAB-era repositories and
lets a neuroimaging lab with no contact with this group analyse a continuous
connectome without MATLAB, without the pipeline, and without asking anyone.

**In scope (WP2):** the file format reader and writer, the small public API,
the atlases, the surface geometry, the validator, the command line, and the
ported numerical methods.

**Out of scope, and deliberately so:** the processing pipeline and fast path
(WP3), the container and documentation site (WP4), the notebooks and benchmark
(WP5), and the Wanda skills (WP6). WP2 is on the critical path for all four,
which is the main reason its interfaces must be frozen early and its numbers
must be right.

## 2. Definition of done

The brief gives one acceptance criterion. Everything below serves it.

```python
pip install sbci
sbci download hcp-ya --subject 100307

from sbci import ContinuousConnectome, load_atlas
cc = ContinuousConnectome.load("sub-100307_sc.h5")
M  = cc.to_atlas(load_atlas("Schaefer200"))
p  = cc.seed(vertex=1234)
cc.plot(p)
cc.to_cifti("sub-100307_sc.dconn.nii")
```

On a machine nobody here configured, from a blank environment, in under five
minutes. Encoded in `tests/test_five_minute_start.py`, which skips until the
data release exists.

| Line | Status | Waiting on |
| --- | --- | --- |
| install the package | wheel builds, installs clean, imports elsewhere | publishing to PyPI; the name is free |
| download a subject | not implemented | Q6 — may not be legally possible as written |
| load a connectome | works, validated on real data | — |
| aggregate to an atlas | works, exactly 200×200 on real data | — |
| take a seed profile | works | — |
| plot a surface map | works on inflated, white, pial and spherical surfaces | — |
| export to CIFTI | works; writes a 16.9 GB fsLR-32k dense connectome | — |

**Four of seven lines work today.** Of the three that do not: one is a legal
question, one is publication, and one is a known bug with a known fix.

## 3. Architecture

A user holds exactly one object. Everything else is a function on it.

```
ContinuousConnectome          the only object a user constructs
├── load / save               HDF5 computational file
├── to_cifti                  fsLR-32k exchange file
├── to_atlas                  region matrix, area-weighted
├── seed                      one vertex's or region's profile
├── coupling                  structure–function coupling
├── smooth                    re-smooth from endpoints
├── reduce                    reduced-rank scores             [pending]
└── plot                      surface figure
```

```
src/sbci/
  spec.py          every on-disk convention, in one place
  metadata.py      the key contract; a file missing keys does not load
  grid.py          condensed ↔ dense on the 5124-vertex ico4 grid
  io/              hdf5.py, cifti.py
  connectome.py    the object above
  parcellation.py  vertex → region aggregation            [ported]
  smoothing.py     the kernels                            [see section 7]
  coupling.py      the three SFC forms                    [ported, verified]
  reduction.py     FPCA scores                            [not ported]
  alignment.py     ENCORE                                 [not ported]
  stats.py         local inference                        [no reference]
  atlas.py         44 bundled parcellations
  surface.py       bundled meshes
  plotting.py      surface figures
  validate.py      the checks behind `sbci validate`
  cli.py           `sbci download` / `sbci validate`
```

**Design rules that have earned their place.** The specification lives in one
module, so changing a convention is a one-file change. The loader refuses an
underspecified file rather than defaulting — a silently defaulted bandwidth
makes two incomparable files look comparable. Unported methods raise with the
name of the MATLAB file to port and the test that will prove it, and a test
asserts those messages stay informative. Heavy dependencies are optional: a
compute node running the pipeline should not need a rendering stack.

[[pagebreak]]

## 4. Where things stand

297 tests passing, lint clean, 96 files, 1.3 MB. Repository at `~/sbci` on
Longleaf; environment at `/work/users/x/y/xya/sbci-venv` (scratch, rebuilt by
`scripts/setup_longleaf.sh`); reference checkouts and MATLAB reference output
under `/work/users/x/y/xya/`.

**Working, and verified against real data**

- HDF5 read and write; both example files pass all seven validator checks
- `to_atlas` — 200×200 for Schaefer200, retaining 99.98% of connectome mass;
  Desikan retains 100.0000%, consistent with its full cortical coverage
- 44 atlases bundled in 332 KB, region counts matching each published atlas
  (Desikan 68, Glasser 360, Gordon 333, Yeo-7 14)
- Riemannian kernel smoothing — 4.4 s per subject, against a WP3 target of
  60 s and the current pipeline's 2–8 hours
- Structure–function coupling, all three forms
- Coupling reproduces the known biology: highest in lateral occipital and
  cuneus, lowest in posterior cingulate and insula, with left and right
  agreeing to 0.002 although nothing in the code enforces it

**Decided but not yet built**

- The spherical heat kernel, which under the Q10 decision is the default and is
  what produced every released cohort. Until it is ported the package cannot
  regenerate its own released data.

**Not built**

- Subject download

### The exchange file, as built

`to_cifti()` writes a CIFTI-2 dense connectome on fsLR-32k: 64,984 x 64,984 in
float32, **16.9 GB**, which plain `nibabel` and Connectome Workbench open
without this package. Beside it go a JSON sidecar with the metadata table and a
`.dscalar.nii` of fsLR vertex areas, without which the values cannot be
integrated.

The resampling turns on one point. The connectome is a **density**, normalized
so that `area @ D @ area == 1`, so moving it to a finer mesh is an
area-weighted *average*, not a *distribution* of each vertex's value. The
distributing form looks right -- it preserves the plain sum of the entries --
and silently loses **99.3%** of the area-weighted mass. The package stores the
raw overlap counts and normalizes by row.

Verified end to end on the example subject, against references the resampling
code does not share:

| Check | Reference | Result |
| --- | --- | --- |
| the overlap partitions the pipeline's own vertices | `area` vector in the subject's HDF5 file | identical, maximum difference 0 |
| regions land where the pipeline puts them | `lh/rh.fs_LR.aparc.annot` | 94.3% / 94.4% of vertices |
| unit mass survives the move | `area @ D @ area` before and after | 1.00000000002 against 0.999999999908 |
| the region matrix is unchanged | Desikan on ico4 against Desikan on fsLR | r = 0.9997, 2.3% relative Frobenius difference |
| the file is readable elsewhere | plain `nibabel`, no `sbci` code | both axes `BrainModelAxis`, 32,492 vertices per hemisphere, symmetric |

The last check is the load-bearing one: the two region matrices are computed
from different meshes with different label files and share no code, so their
agreement is evidence about the correspondence rather than about the
implementation.

## 5. The verification standard

This is the practice that has made the work trustworthy, and it is proposed as
the rule for everything that follows.

**No port is finished until it has been diffed against a reference run.**
MATLAB is available on Longleaf (versions 2018b–2026a), so "this cannot be
checked" is almost never true. Reference scripts live in `tests/reference/` and
their output feeds `tests/test_matlab_reference.py`, which skips cleanly where
that output is absent.

Achieved so far:

| Component | Agreement with MATLAB |
| --- | --- |
| Endpoint adjacency | bit-identical |
| Riemannian diffusion kernel | 3.18 float32-eps |
| Smoothed density | 3.17 float32-eps |
| SFC global / discrete / local | 10.4 / 88.2 / 3.6 float64-eps, identical NaN patterns |

Smoothing agrees only to single precision because the reference itself is
single precision: the eigenvalues are stored as `single`, and in MATLAB
`single .* double` yields `single`, so the whole kernel collapses to float32.
The port keeps float64 and is therefore more accurate than the reference, not
different from it.

**A second rule, learned the hard way.** A numerical diff is not sufficient on
its own. The surface meshes passed every test while being in the wrong vertex
order, because nothing checked that the mesh and the data agreed with each
other. Any bundled data now needs a test that ties it to the rest of the
package — for the meshes, that every atlas drawn on them is spatially
contiguous.

[[pagebreak]]

## 6. What blocks what

```
Q6  HCP redistribution ─────► subject download ──► acceptance test
Q1, Q2, Q3/Q9 format freeze ► every file ever written
                              └─► WP3 fast path, WP5 notebooks
Q7  licence holder ─────────► first tagged release
```

**Q6 is the critical path.** It gates the second line of the acceptance
criterion, it is the only item whose answer comes from outside the group, and
nobody has asked yet. If derived dense connectomes cannot be redistributed
openly, the five-minute test cannot pass as written and the criterion itself
needs rewording. It should start first precisely because its lead time is not
ours to control.

**Q1, Q2, Q3/Q9 are ratifications, not investigations.** They need one meeting,
and everything downstream writes these files.

## 7. Plan

**Immediately, in parallel with everything else: ask HCP (Q6).** One email.
Longest lead time, widest blast radius.

**Freeze the format (Q1, Q2, Q3/Q9).** One meeting, with recommendations
already drafted in Annex A. Until this lands, every file written is provisional
and the format version keeps its `-draft` suffix.

**Then, in dependency order:**

1. **The spherical kernel**, which the Q10 decision makes the default -- now
   **done**. It reproduces `c3_main` at r = 1.000000 across five ADNI
   subjects. The residual recorded here for months as "a normalization or
   sampling convention" was neither: `concon`'s kernel is not the heat kernel,
   its weight is `(2l+1)^(3/2)/sqrt(4 pi)` and it has compact support, both
   read from its source once the binary was found to run. The naming the
   decision implied is applied: `shk` is the spherical kernel and the Matérn
   kernel is `matern`.
2. **FPCA reduction.** Needed by the reduced-rank API and the cohort notebook,
   and **blocked on verification rather than effort**: two of its three MATLAB
   dependencies are no longer downloadable, so the reference cannot be run and
   the standard in section 5 cannot be met. Resolve the dependencies first.

**Publishing.** The name `sbci` is free on PyPI. Claiming it early costs
nothing and prevents an awkward rename later.

## 8. Risk register

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Q6 answers "credentials required" | the acceptance test cannot pass as written | ask now; prepare a fallback criterion |
| the spherical kernel port does not reproduce the released files | the package cannot regenerate its own released data | verify directly against `smoothed_sc_avg_0.005_ico4.mat`, which already exists |
| Q2 flips to "include the diagonal" | every released file must be rewritten | freeze before the first release, not after |
| Q3/Q9 picks the plain sum | the legacy importer normalizes wrongly | decide before importing a cohort |
| a cohort mixes the two kernels | the files are silently incomparable | the validator refuses mixed kernels; metadata records which was used |
| FPCA dependencies stay unobtainable | one method cannot meet the verification standard | decide whether to ship it unverified, and say so plainly in the docs |
| ico4 cannot resolve the finest atlases | Schaefer900 and Schaefer1000 lose parcels outright | already measured and documented; do not advertise them |

## 9. Decisions needed, with owners

| # | Decision | Who | Lead time |
| --- | --- | --- | --- |
| 6 | May derived dense connectomes be redistributed openly? | HCP, via whoever holds the data agreement | weeks |
| 1 | HDF5 dataset names | WP1 owner | minutes |
| 2 | Diagonal in the stored triangle | WP1 owner | minutes |
| 3/9 | Which normalization | WP1 owner | minutes |
| 7 | Copyright holder for the licence | PI | minutes |

Four of these are minutes of discussion. One is not, and it should start today.
Everything not listed here needs no decision — it is engineering work, already
scoped.

[[pagebreak]]

## 10. Annex A — the open questions in full

Grouped by who has to act. Numbers are stable references to
`SPEC_QUESTIONS.md` in the repository and do not run in order here.

### Tier 1 — needs an answer from outside this work

**Q6. May derived dense connectomes be redistributed openly?**
The package's `download` command needs a stable URL and a checksum manifest for
the tutorial subject and the HCP-YA cohort. Separately, and more importantly,
the brief's own open question: do the HCP data use terms permit redistributing
*derived* dense connectomes openly, or must every user hold HCP credentials?
*Why it matters:* this is the difference between the five-minute acceptance
test passing and not. If credentials are required, the criterion has to be
reworded rather than met. *Nobody has asked yet.*

### Tier 2 — ratifications for the format owner, minutes each

**Q1. HDF5 dataset names and layout.**
The brief specifies "upper-triangular float32 array, plus datasets for area
weights, mask, vertex coordinates, and a JSON metadata string" but not the
names. The package currently writes `/connectivity`, `/area`, `/mask`,
`/coordinates`, `/metadata`. *Recommendation: ratify as written.* Pure
convention, no technical content — but after the first file is released it
cannot change without a format version bump.

**Q2. Does the stored triangle include the diagonal?**
Currently excluded. *Recommendation: keep it excluded.* The evidence is
one-sided: the legacy structural file has an exactly zero diagonal, the
functional file has exactly 1.0, the toolkit's own parcellation strips the
diagonal before aggregating, and two of the three coupling functions always
strip it. No downstream consumer reads it.
*The honest cost:* excluding it discards genuine within-vertex structural
connectivity — streamlines with both endpoints on the same roughly 5 mm patch.
Those are mostly short U-fibres, so discarding is defensible, but it is a
choice with a cost rather than a free one.
*If this flips later, every released file must be rewritten*, because the
stored vector length changes.

**Q3 / Q9. Which normalization?**
Three conventions are in play: the MATLAB smoothing script normalizes so the
**plain sum** of the density is 1; the package's legacy importer normalizes so
the **area-weighted** sum is 1; the released files are neither, summing to
5.49e6.
*Recommendation: area-weighted.* The deciding argument is internal
consistency — the toolkit's own parcellation aggregates by integrating values
against vertex areas, which only makes sense if they are densities per unit
area. If that is what they are, the normalization must be area-weighted too.
The plain sum is exact only on a mesh with uniform vertex areas; ico4 areas
run from 57 to 68, so pairwise that is up to an 18% discrepancy. As it stands
the reference pipeline is internally inconsistent between its smoothing and its
parcellation.

**Q7. Who holds the copyright?**
The licence file currently says "The SBCI developers". It needs the
institutional holder before the first tagged release.

### Tier 3 — no decision needed; engineering work, already scoped

**Q4. The ico4 to fsLR-32k resampling.** — **Done.**
Built, bundled and verified; see the note below and SPEC_QUESTIONS.md item 4.

**Q8. The format carries no endpoints.** -- **Decided and built.** Endpoints
now live in the computational file itself, as an optional `/endpoints` group,
rather than in a sibling: a connectome and the streamlines it was built from
then cannot be separated or versioned apart. On the example subject they add
11.8 MB to a 20.0 MB file, and the group is optional so most users carry none.
`.smooth()` works from them for `rdk` and `matern`; the Laplace-Beltrami basis
is 50 MB per hemisphere and is passed in rather than bundled.

**Q11. No anatomical surface could be drawn.** — **Resolved.** The surfaces
are rebuilt from fsaverage directly, as the mean position over each grid
vertex's members under `mapping_avg_ico4.npz`. Inflated, white and pial are
bundled, agreement with FreeSurfer's own annotation is **99.9%**, and `plot()`
defaults to inflated.

The first attempt produced an anatomically scrambled mesh that passed every
check applied to it, and the diagnosis recorded at the time — that the
toolkit's anatomical meshes are in the wrong vertex order — was itself wrong.
Only the `_lps_` variants are out of step; `white_avg_ico4.vtk` and
`inflated_avg_ico4.vtk` score the same 0.8405 Desikan edge contiguity as the
grid and spherical meshes, and the rebuilt white surface agrees with the
toolkit's own to **0.26 mm median** once the LPS-to-RAS sign flip is applied.
SPEC_QUESTIONS.md item 11 carries the full table.

### Decided during preparation of this document

**Q10. Which smoother does the project standardize on?** **Both ship, with the
spherical heat kernel as the default and the choice recorded in metadata.**

These are two methods from two papers, not an inconsistency. The **spherical
heat kernel** (Moyer et al., *A Continuous Model of Cortical Connectivity*,
MICCAI 2016) inflates the cortex to a sphere and smooths with a truncated
spherical-harmonic heat kernel; it produced every released cohort. The
**Riemannian diffusion kernel** (bioRxiv 2025.09.08.674789) uses
Laplace–Beltrami eigenfunctions on the cortical surface itself and argues that
the spherical projection distorts the estimate. They agree at r = 0.58, about
the size of the difference the newer paper is about.

That the two really are different geometries has been confirmed rather than
assumed. The shipped eigenpairs were checked against both candidate surfaces:
their spectrum implies a surface of 60,634 mm² by Weyl's law, matching the
measured white-matter mesh at 61,243 mm² to within 1% and nothing like the
spherical mesh at 125,513 mm². The lowest eigenvalue triplet, which on a sphere
would be a single repeated value, spreads by a factor of 2.2. The operator is
that of the folded cortex.

Consequences: released cohorts stay canonical and nothing is reprocessed; the
validator must refuse to mix kernels within a cohort; and the two take
different bandwidth parameters — `sigma` for the spherical kernel, `kappa` for
the Riemannian one — so `bandwidth` in the metadata is meaningful only
alongside `kernel`. The spherical kernel still has to be ported, and unusually
it can be checked against a released file rather than a reference run.

### Closed

**Q5. Bundled surface geometry.** Answered. The toolkit already ships inflated,
white and spherical meshes at ico4 resolution; they are converted and bundled,
290 KB in total, so plotting needs no download. One caveat for the record: a
pial surface is not available at ico4 resolution.
