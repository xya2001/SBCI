# sbci

Continuous brain connectivity: read, parcellate, smooth, couple, align and
reduce surface-based continuous connectomes on the ico4 grid (5124 vertices),
with every method a verified port of the SBCI group's MATLAB. SBCI,
Surface-Based Connectivity Integration (Cole et al., *Human Brain Mapping*,
2021), represents structural and functional connectivity as continuous
functions on the cortical surface instead of between atlas regions; the ico4
grid is FreeSurfer's fsaverage sphere subdivided four times, 2,562 vertices per
hemisphere, left then right.

> **Status: pre-alpha.** Every method in the API table below is implemented
> and verified against its reference, and the example cohort, eleven HCP Young
> Adult subjects, downloads with `sbci download hcp-ya`.

## Install

Python 3.10 or newer. The package is not on PyPI yet, so install from GitHub:

```bash
pip install "sbci @ git+https://github.com/xya2001/SBCI.git"             # core: numpy, scipy, h5py, nibabel
pip install "sbci[plotting] @ git+https://github.com/xya2001/SBCI.git"   # adds matplotlib and nilearn for figures
pip install "sbci[render] @ git+https://github.com/xya2001/SBCI.git"     # adds PyVista, for the smoothly lit figures below
```

On UNC's Longleaf cluster use `scripts/setup_longleaf.sh` instead; see
*Development on Longleaf* below.

## Try it in two minutes

One real subject is 90 MB away: the example cohort (eleven HCP Young Adult
subjects) is fetched by the package, and everything below runs on the first
of them, sub-100307, the tutorial subject.

```python
import sbci
from sbci.download import fetch_cohort

sc_path, fc_path = fetch_cohort(out="hcp-ya", subjects=["100307"])   # once; verified
sc = sbci.load(sc_path)                  # structural connectome, with the endpoints of its 803,741 streamlines
fc = sbci.load(fc_path)                  # functional connectome, from the HCP's resting state
sc.to_atlas("Schaefer200").shape        # (200, 200)
sc.plot(sc.seed(vertex=1234))            # a figure; needs the plotting extra
sc.coupling(fc)                          # structure against function, one value per vertex
sc.smooth(kernel="shk", mask_medial_wall=True)     # re-smooths from the stored endpoints
```

Or from the shell:

```bash
sbci download hcp-ya --subject 100307   # sub-100307_sc.h5 and _fc.h5, 90 MB, into the current directory
sbci info sub-100307_sc.h5              # what it holds
sbci validate sub-100307_sc.h5          # nine checks, all should pass
sbci atlases --match Yeo                # the bundled atlases
```

Each young adult has an SC file and an FC file. The FC is the HCP's own
cleaned resting-state data, four runs of 14 minutes for most subjects, put on
the grid through the subject's own registration so that it lines up with the
SC (`tools/build_hcp_fc.py`).

`sbci.example()` still builds a synthetic connectome on the real grid, for the
tests and for a machine without network access; nothing in this README is
drawn from it.

The figures in this README are drawn from the eleven young adults of the
example cohort, 22 to 35 years old by the HCP's open-access age bands, and the
single-subject figures from sub-100307, the subject of the commands above. All
were rebuilt on the ico4 grid from the lab's pipeline output with
`tools/build_hcp_cohort.py`, their FC from the HCP's resting state with
`tools/build_hcp_fc.py`, and drawn on FreeSurfer's fsaverage surface, rendered with smooth lighting
through PyVista (`plot(mesh="fsaverage", engine="pyvista")`);
`scripts/hcp_figures.py` draws them from a downloaded copy, and
[docs/figures](docs/figures/README.md) describes each.
The analysis of a whole cohort uses all 946 young adults with complete
pipeline output.

![The connectivity of one vertex over the cortex](docs/figures/seed_profile.png)

*`sc.plot(sc.seed(vertex=1234), mesh="fsaverage", engine="pyvista")`: where one vertex of one
subject connects to. The value at each vertex is the density of streamlines
between the seed (the orange dot, left temporal cortex) and that vertex,
relative to the strongest, on the inflated surface shaded by sulcal depth.*

**What smoothing does.** Tractography gives endpoints. sub-100307's 803,741
streamlines were split into two random halves; 32 of them touch vertex 1234
in one half and 26 in the other. That is few: more streamlines end at 85% of
this subject's cortical vertices. So the two halves' raw counts share little,
and they agree across the cortex at only r = 0.41. Smoothed, they agree at
r = 0.99, and half A's map matches the map from all streamlines at r = 1.00. The kernel pools the streamlines that end near the
vertex, and that is what makes two subjects, or two sessions, comparable
vertex by vertex.

![Smoothing, from endpoints to a comparable map](docs/figures/smoothing_power.png)

*Top: the far ends of the streamlines that touch the vertex (blue dots), in
the two halves and in the whole set. Bottom: the smoothed density of the same
vertex from each, relative to its strongest vertex.*

**Structure against function.** `sc.coupling(fc)` asks, vertex by vertex, how
closely the cortex a vertex is wired to matches the cortex its activity moves
with: the cosine similarity of its SC and FC profiles. Averaged over the
eleven, coupling is highest in visual cortex (0.36 to 0.41 in the
pericalcarine, cuneus and lateral occipital regions) and lowest in the
cingulate and entorhinal cortex (0.05 to 0.09). It averages 0.30 over primary
and unimodal sensory and motor regions and 0.19 over association regions, the
gradient from sensory to transmodal cortex that coupling is known for
(Vázquez-Rodríguez et al., PNAS 2019; Baum et al., PNAS 2020). The
eleven subjects' maps correlate at r = 0.53 to 0.71.

![Structure-function coupling](docs/figures/coupling.png)

*`sc.coupling(fc)` for each of the eleven young adults, averaged: at each
vertex, the cosine similarity of its SC and FC profiles, coloured from the
map's 2nd to 98th percentile. The medial wall, which has no cortex, is gray.*

## With real data

```python
import sbci

cc = sbci.load("sub-100307_sc.h5")     # the .h5 "computational file": what the package reads and writes
M  = cc.to_atlas("Schaefer200")        # 200 x 200 matrix
p  = cc.seed(vertex=1234)              # profile on the surface
cc.plot(p)                             # inflated-surface figure
cc.to_cifti("sub-100307_sc.dconn.nii") # the "exchange file" for Workbench: 16.9 GB, 25 GB of memory (USAGE.md)
```

The example cohort, the eleven young adults the figures are drawn from, is
hosted on a public Google Drive ([the
folder](https://drive.google.com/drive/folders/1gG2ZmxxVm4w5dvlCQvMaEEOBypU7nDpx),
for browsing) and fetched by the package:

```bash
sbci download hcp-ya --out hcp-ya             # all eleven subjects into ./hcp-ya, about 1 GB
sbci download hcp-ya --subject 100307         # its SC and FC, 90 MB, into the current directory
```

Each file is verified against the SHA-256 in the package's manifest
(`src/sbci/data/hcp_ya.json`, which also gives each subject's sex and HCP age
band), and a re-run skips what is already present. Ten of them, five women
and five men, were drawn at random from the 946 young adults with complete
pipeline output in the lab's copy, and sub-100307, the tutorial subject,
joined them. Each was rebuilt on the ico4 grid from the
pipeline's snapped streamline endpoints on the subject's FreeSurfer-registered
sphere (`tools/build_hcp_cohort.py`), and the SC files
carry those endpoints, so `smooth()` and `endpoints_align()` run on them. The
FC files hold the HCP's ICA-FIX-cleaned resting-state runs, REST1 and REST2 in
both phase-encoding directions, placed on the grid through each subject's
MSMSulc and FreeSurfer spheres (`tools/build_hcp_fc.py`); sub-116221 has three
runs, which is all the HCP has, and the others four.

The data are the WU-Minn Human Connectome Project's, redistributed under its
[Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
Downloading them means accepting those terms, and every download writes
`DATA_USE.txt` beside the files with the acknowledgment a publication must
carry. The other 935 young adults are not distributed.

`scripts/hcp_figures.py hcp-ya docs/figures` draws the young adult figures
from a downloaded copy, and `scripts/check_hcp_ya.py` downloads the cohort
and runs every method on it, stopping at the first check that fails. Existing
pipeline output converts with `tools/import_legacy.py` (see *Using legacy
pipeline output*). `tests/test_five_minute_start.py` runs the lines above on
sub-100307 from a blank environment when `SBCI_DOWNLOAD=1` or `SBCI_HCP_DIR`
is set, and CI's `five-minute-start` job runs it on every push
(VERIFICATION.md).

## The documents

| Read | For |
| --- | --- |
| [USAGE.md](USAGE.md) | every function with a worked example on the released subjects, and what the common errors tell you |
| [BLUEPRINT.md](BLUEPRINT.md) | what the package is, what "finished" means, where it stands, and the decisions it waits on |
| [PORTING.md](PORTING.md) | how each of the seven MATLAB methods was ported and verified, and the errors found in the references |
| [SPEC_QUESTIONS.md](SPEC_QUESTIONS.md) | the file-format decisions, answered and open |
| [VERIFICATION.md](VERIFICATION.md) | how to check every claim yourself, in tiers from five minutes to a MATLAB licence |
| [scripts/](scripts/README.md), [tools/](tools/README.md) | worked scripts (with the lab's Longleaf paths) and the builders of the bundled data |

## API

| Call | Status |
| --- | --- |
| `sbci.load(path)` | implemented; the `.h5` computational file, validated on read. The `.dconn.nii` exchange file is write-only: its resampling to fsLR-32k is many-to-one |
| `ContinuousConnectome.load(path)` | the same thing, if you prefer the class |
| `.save(path)` | implemented |
| `.to_atlas(atlas, how="mass"\|"mean")` | implemented; takes an `Atlas` or a name such as `"Schaefer200"`; matches `parcellate_sc.m` to float64 rounding |
| `.seed(vertex=...)` / `.seed(region=...)` | implemented; `region=` takes a vertex mask or an `(atlas, region)` pair |
| `.coupling(fc, scope=...)` | implemented; matches the MATLAB to float64 rounding; on the young adults' own FC its map follows the gradient from sensory to association cortex (PORTING.md item 2) |
| `.plot(values, surface=...)` | implemented; inflated, white, pial and sphere bundled |
| `.to_cifti(path)` | implemented; fsLR-32k dense connectome, 16.9 GB |
| `sbci validate <file>` | implemented |
| `.smooth(kernel=..., bandwidth=..., eigenpairs=...)` | implemented for all three kernels. `shk` is the default and reproduces `concon` at r = 1.000000 at full scale on a pipeline subject; `rdk` matches MATLAB to 3.25 float32-eps, `matern` is checked against its closed form (no MATLAB reference exists), and both find the Laplace-Beltrami basis (`EV_LBO_ds_ico4_{L,R}.mat` from [SBCI_Toolkit](https://github.com/sbci-brain/SBCI_Toolkit)'s `concon_estimate`) via `$SBCI_LBO_DIR` (PORTING.md items 1 and 6) |
| `sbci download hcp-ya` / `sbci.download.fetch_cohort` | implemented; fetches the eleven-subject HCP Young Adult example cohort from its public Google Drive, verifies every file against the manifest, and writes the HCP's data use terms beside the files |
| `sbci.migrate_warp(warp, to="fs_LR_32k")` | implemented; restates an ENCORE or ConSEAL warp on fs_LR (MSMAll's sphere) or full-resolution fsaverage and writes it as a deformed sphere, so that Workbench can resample a subject's MSMAll fMRI or myelin map through the warp found from its connectivity (USAGE.md has the example) |
| `.reduce(rank=K)` / `sbci.reduce(cc_list, rank=K)` | implemented; takes connectomes or the paths of their files, which it reads one at a time; matches the MATLAB reference to float64 rounding (PORTING.md item 5) |
| `sbci.align(cc_list, template=None)` | implemented; ENCORE. Registers every connectome onto one template: with no `template=` it first estimates one, the Karcher median of the cohort's square-root densities, so that no subject is the reference (the default for a cohort); with `template=` a square-root density (an earlier run's `result.template`, or one subject's) it registers onto that instead, which is how a new subject joins a cohort or one subject is registered onto another. Geometry and template match MATLAB to float64 rounding, the registration to r = 0.99999979 (PORTING.md item 4) |
| `sbci.endpoints_align(cc_list, template=None)` | implemented; ConSEAL, which warps the streamline endpoints themselves, used the same way: no `template=` estimates the Karcher median first and registers everyone onto it; `template=k` registers everyone onto subject `k`, which stays put; a square-root density registers onto that. Every stage matches the public MATLAB to the single precision it carries; four errors in that reference are corrected by default and reproducible with `strict_upstream=True` (PORTING.md item 7) |
| `sbci.stats.local_test(scores, design)` | implemented; **no reference exists**, so verified against `scipy.stats` and against the procedures' own guarantees. `groups=` treats related subjects, families of twins and siblings, as clusters, with cluster-robust standard errors that match statsmodels to 1e-9 (PORTING.md item 5) |
| `sbci.example(modality="sc"\|"fc")` | implemented; a synthetic connectome on the real grid with 20,000 synthetic endpoints, for the tests and for a machine without network access; the documentation's examples use the released subjects instead |
| `sbci.example_cohort(n_subjects, seed, effect)` | implemented; a synthetic cohort with one bundle scaled by a synthetic age, with which `reduce`, `local_test` and the alignments are checked against a planted answer (PORTING.md items 5 and 7) |
| `sbci info <file>` | implemented; what a file holds, without opening Python |
| `sbci example --out <file>` | implemented; writes a file to try the package on |
| `sbci atlases [--match ...]` | implemented; lists the bundled atlases and their region counts |

## An end-to-end analysis

The single-subject methods need one subject; the cohort methods need a
cohort. The example subjects are enough to align (below), but not to ask a
question of: an association among eleven people is out of reach. The analysis
here uses all 946 HCP Young Adult subjects with complete pipeline output in
the lab's copy, rebuilt on the ico4 grid as the example subjects were. Only
the eleven are distributed; with HCP access and the pipeline's output,
`tools/build_hcp_cohort.py` builds the rest. The
question is whether fluid intelligence, the number of Penn Matrix Test items
a subject answered correctly (`PMAT24_A_CR` in the HCP's open-access table),
shows in the structural connectome, with sex, age band and the streamline
count in the model. The young adults are not independent: the 946 come from
423 families of twins and siblings, and a test that counts them as
independent makes every p-value too small. `groups=` treats the families as
clusters. Family membership is in the HCP's restricted table, which has its
own data use agreement: it is read at analysis time, and nothing from it is in
this package or its figures.

```python
import csv
from pathlib import Path
import numpy as np
import sbci

table = {f"sub-{r['Subject']}": r for r in csv.DictReader(open("unrestricted.csv"))}  # the HCP's open-access table
family = {f"sub-{r['Subject']}": r["Family_ID"] for r in csv.DictReader(open("RESTRICTED.csv"))}  # restricted
subjects = [p.name.split("_")[0] for p in sorted(Path("hcp-ya-full").glob("sub-*_sc.h5"))]
subjects = [s for s in subjects if table[s]["PMAT24_A_CR"]]    # those with a score
paths = [f"hcp-ya-full/{s}_sc.h5" for s in subjects]
reduction = sbci.reduce(paths, rank=20)                   # FPCA, one file at a time: a batch job, 230 GB
rows = [table[s] for s in subjects]
score = np.array([float(r["PMAT24_A_CR"]) for r in rows])
female = np.array([r["Gender"] == "F" for r in rows], dtype=float)
band = np.array([{"22-25": 23.5, "26-30": 28, "31-35": 33, "36+": 37}[r["Age"]] for r in rows])
count = np.array([sbci.load(p).metadata["streamline_count"] for p in paths]) / 1e6
design = np.column_stack([score, female, band, count])   # column 0 is the intercept local_test adds
groups = [family[s] for s in subjects]
trait = sbci.local_test(reduction.scores, design, terms=[1], groups=groups)   # fluid intelligence
trait.significant()                                       # the components that survive: none
sex = sbci.local_test(reduction.scores, design, terms=[2], groups=groups)     # sex, the trait as nuisance
effect = sex.effect_map(reduction, alpha=0.05)            # one value per vertex, surviving components only
sbci.load(paths[0]).plot(effect, mesh="fsaverage", engine="pyvista")
```

Fluid intelligence is in the data, but too weakly for the components to
carry it once the families are clusters. Streamed once, before any fit
(`tools/age_probe.py`), the 943 subjects with a score show it in the
connectivity strength of 113 of 4,685 cortical vertices at FDR 0.05, never
beyond |r| = 0.17, and in no Desikan region pair. Fit at rank 20, which
captures 27% of the cohort's norm, no component tracks it after the
false-discovery-rate correction across the twenty: the closest, component 13,
correlates with the score at r = 0.11 (adjusted p 0.064), and component 7 at
0.10 (0.083). Counting the 943 as independent would have put component 13 at
0.023 and the vertices at 217; the families are the difference. A rank-4 fit
comes nowhere near (smallest adjusted p 0.32).

Sex, tested in the same model with fluid intelligence as a nuisance, does
show: in 11 of the 20 components, the strongest at adjusted p 2e-8, in all
four of a rank-4 fit, and in 1,020 vertices and 761 Desikan pairs. The effect
map places it at the occipital poles, where connectivity is relatively higher
in men; head size, which differs between the sexes, is not in the model.
PORTING.md item 5 has the measurements.

![The components most associated with fluid intelligence](docs/figures/cohort_trait_scores.png)

*The two components of a rank-20 FPCA of 943 HCP Young Adult subjects most
associated with fluid intelligence, given sex, age band and streamline count,
with families as clusters: each subject's score against the number of PMAT24
items answered correctly (women orange, men blue), the mean in each fifth of
that range with its 95% interval, the least-squares line, and the adjusted
p-value. Neither survives the correction.*

![Where sex shows in the structural connectome](docs/figures/cohort_sex_effect.png)

*The effect map over the 11 components that track sex, families as clusters,
relative to its largest value: the fitted difference in connectivity between
women and men (positive: higher in women), summed over the other endpoint.*

**Coupling across the cohort.** With FC built for the young adults
(`tools/build_hcp_fc.py`), each of the 903 with four complete resting-state
runs gives a coupling map, and `local_test` takes the 4,685 cortical vertices
as its columns: one test per vertex, families as clusters, with head motion
and intracranial volume added to the model because both shape these measures.

```python
motion = ...   # per subject: Movement_RelativeRMS_mean.txt averaged over the four runs (mm)
volume = ...   # per subject: EstimatedTotalIntraCranialVol from T1w/<id>/stats/aseg.stats (litres)
maps, kept = [], []
for i, s in enumerate(subjects):
    fc = sbci.load(f"hcp-ya-full/{s}_fc.h5")                # tools/build_hcp_fc.py
    if fc.metadata["fc_frames"] == 4800:                      # four complete runs
        maps.append(sbci.load(paths[i]).coupling(fc))
        kept.append(i)
maps = np.array(maps)
cortex = ~np.isnan(maps).any(axis=0)                          # NaN on the medial wall
covariates = np.column_stack([female, score, band, count, motion, volume])[kept]
sex = sbci.local_test(maps[:, cortex], covariates, terms=[1], groups=np.array(groups)[kept])
sex.significant()                                             # vertices, here
```

The cohort's mean map is the eleven's (r = 0.972), and two halves of the
cohort, split by family, reproduce each other's at r = 0.998: the pattern is
settled, and the question is how people depart from it. Fluid intelligence
barely does, at 2 vertices of 4,685 at FDR 0.05, with mean coupling rising
with the score at p 0.036. Sex shows at 325 vertices, 253 of them with
coupling higher in men, in the inferior parietal, superior frontal and
opercular cortex, the insula and early visual cortex. Head size carries most
of the difference: without intracranial volume in the model the count is
1,104, which is reason to read the structural sex effect above, fitted
without it, with the same care. The 72 vertices higher in women are the less
trustworthy part: 44% of them lie within two grid rings of the medial wall,
where 6% of cortex lies, along the isthmus and posterior cingulate, where the
mask drops SC endpoints and the FC fills grid cells the HCP's surface leaves
empty. PORTING.md item 2 has the measurements.

![Where coupling differs between women and men](docs/figures/cohort_coupling_sex.png)

*The fitted difference in structure-function coupling, women minus men, at
the 325 of 4,685 cortical vertices significant at FDR 0.05 among 900 HCP
Young Adult subjects with a fluid-intelligence score and four complete
resting-state runs, given fluid intelligence, age band, streamline count,
head motion and intracranial volume, families as clusters. Blue: higher in
men.*

Alignment fits in front of `reduce`: `sbci.align(subjects)` (ENCORE) returns
the warped densities as `aligned`, and `sbci.endpoints_align(subjects)`
(ConSEAL) warps every subject's endpoints onto a common template,
`aligned_endpoints(i)` hands them back, and `smooth()` turns them into aligned
connectomes (USAGE.md shows the three lines).

**Alignment, with a known answer.** The endpoints of sub-100307's 803,741
streamlines were moved by a known smooth warp, a random field of
spherical harmonic degree up to four that moves the grid's vertices 1.7
degrees on average and 4 at most,
and the deformed copy was registered back onto the undeformed subject, both
through the package's smoother. ENCORE, searching its default degree-6 basis,
undoes 88% of the displacement in ten steps: the endpoints come back from
1.63 to 0.20 degrees, and the cost falls to 0.16 of its start. ConSEAL, with
the paper's update rule and a stopping threshold of 1e-7, brings them to
within 0.11 degrees, 93%, in sixty iterations. The figure shows, vertex by
vertex, how far the endpoints still are from where they started. Two rules
this test taught, measured in PORTING.md item 7: build every density from the
same endpoints through the same smoother, and judge a registration by a warp
it can represent.

![Alignment with a known answer](docs/figures/alignment_recovery.png)

**Carrying the warp to another template.** The warp ENCORE found on the grid
is an anatomical correspondence, and it applies to anything of the same
subject on another sphere. The map carried here is sub-100307's own sulcal
depth from its FreeSurfer reconstruction, at fsaverage resolution (163,842
vertices per hemisphere): a measure of anatomy the tractography never sees.
The known warp of the recovery figure moved it to a correlation of 0.95 with
the original. ENCORE's warp, estimated from connectivity on 5,124 vertices and
restated on fsaverage by `sbci.migrate_warp`, put it back to 0.996, and on
fsaverage it sits 0.33 degrees from the true warp, which moved fsaverage's
vertices 1.58 degrees on average. The figure shows the map, and what each warp
changes in it.

![Carrying a warp between templates](docs/figures/migration_power.png)

**Alignment and the analysis.** What aligning first does to an association
is measured against a planted answer in PORTING.md items 5 and 7: with the
cohort's anatomy jittered by three degrees the planted effect slips past a
rank-4 FPCA's default start, `candidates=6` reaches it, ENCORE puts it first,
and ConSEAL keeps it with its default regularization or a mean template.

**Aligning the eleven subjects.** Registered onto a template estimated from
the cohort, with every density from the package's smoother, the subjects grow
more alike: the mean correlation between two subjects' connectomes rises from
0.716 to 0.761 after ten ENCORE iterations and to 0.785 after thirty ConSEAL
iterations, and the cost falls for every subject. ENCORE registers onto its
Karcher median. ConSEAL registers onto the mean of the square-root densities
instead, because its median settles on one subject: on these eleven it comes
within 0.001 degrees of sub-212116 and sits 24 to 27 degrees from the others,
which would register everyone onto that subject's connectome, while the mean
sits 17 to 20 degrees from every subject. The USAGE notes explain when the
median does this and how to pass a template. Eleven subjects with 734,039 to
988,788 streamlines each take ten minutes with ENCORE and two hours with
ConSEAL on eight cores.

![Aligning the eleven subjects](docs/figures/cohort_alignment.png)

*Left: each subject's registration cost per iteration, relative to its start,
for ENCORE and ConSEAL. Right: the correlation between the connectomes of
each of the 55 pairs of subjects before alignment and after each method, with
the mean.*

## Getting around the package

Everything public is reachable straight off `sbci`, and so is every submodule,
so `import sbci` is all the import line anyone needs:

```python
import sbci

sbci.load, sbci.smooth, sbci.align, sbci.reduce, sbci.local_test
sbci.stats.local_test          # submodules resolve too
```

The heavier submodules load on first use, so `import sbci` costs about 300 ms
and pulls in neither matplotlib nor scipy unless you touch something that
needs them. `tests/test_api_surface.py` pins the import set; the time is not asserted.

Anything wrong with a *file, its contents, or a name the package looks up*
(an atlas, a region) raises a subclass of `sbci.SbciError`, so one `except`
covers a pipeline's input problems, while a mistake in a call's own arguments
raises a plain `ValueError`:

```python
try:
    cc = sbci.load(path)
except sbci.SbciError as exc:
    print(f"{path} is unusable: {exc}")
```

`SbciError` subclasses also derive from the built-in exception you would reach
for first -- `FormatError` is a `KeyError`, `MetadataError` is a `ValueError`
-- so existing code keeps working.

Names that do not exist suggest ones that do, rather than printing the whole
catalogue:

```python
>>> sbci.load_atlas("Shaefer200")
UnknownAtlasError: unknown atlas 'Shaefer200'. Did you mean 'Schaefer200' or 'Schaefer900' or 'Schaefer800'?
>>> sbci.load_atlas("Desikan").region_mask("LH_banksts")
ValueError: aparc has no region named 'LH_banksts'. Did you mean 'LH_bankssts' or 'RH_bankssts' or 'LH_parsorbitalis'?
```

## Atlases

44 atlases ship inside the package (about 330 KB), so `to_atlas` needs no
network, no FreeSurfer and no MATLAB:

```python
from sbci import list_atlases, load_atlas

load_atlas("Schaefer200").n_regions   # 200
load_atlas("Desikan").n_regions       # 68
load_atlas("Glasser").n_regions       # 360
```

Short names (`Schaefer200`, `Desikan`, `DK`, `Destrieux`, `Glasser`,
`Brainnetome`, `Yeo7`, `Yeo17`) resolve to the stored names, ignoring case,
spaces, hyphens and underscores. `list_atlases()` gives the full set, which
also includes Gordon, the PALS-B12 family and CoCoNest at 22 scales.

Label `0` means "no region" -- the medial wall, plus everything outside a
partial-coverage atlas. Regions are numbered `1..K` with no gaps.

Regenerate them with `tools/convert_atlases.py` if the source ever changes;
the output is committed so no release step needs MATLAB.

## Using legacy pipeline output

`tools/import_legacy.py` converts existing SBCI `.mat` output into the HDF5
format, so a cohort processed with the old pipeline can be used today without
reprocessing:

```bash
python tools/import_legacy.py \
    --sc smoothed_sc_avg_0.005_ico4.mat \
    --fc fc_avg_ico4.mat \
    --mapping mapping_avg_ico4.npz \
    --subject 100307 --out derivatives/
sbci validate derivatives/sub-100307_sc.h5
```

Verified end to end on the SBCI_Toolkit example subject: both files pass every
validator check, `to_atlas(Schaefer200)` returns a 200x200 matrix retaining
99.98% of the connectome mass, and SC and FC correlate at r = 0.26 across the
19,900 region pairs.

## Development on Longleaf

Longleaf's default `python3` is 3.8, which this package does not support.

```bash
./scripts/setup_longleaf.sh                                   # once
module load python/3.12.4                                     # every session
source /work/users/${USER:0:1}/${USER:1:1}/$USER/sbci-venv/bin/activate   # the two letters are your onyen's first two
pytest
```

**Load the module before activating, every time.** The virtualenv's interpreter
is dynamically linked against the module's `libpython3.12.so.1.0`, which is not
on the default library path. Activating without loading the module gives:

```
python: error while loading shared libraries: libpython3.12.so.1.0:
cannot open shared object file: No such file or directory
```

That is a missing module, not a broken environment — `module load` fixes it
without recreating anything.

The virtualenv lives on `/work` rather than in `$HOME` because home quotas are
50 GB and a scientific-Python environment is several hundred megabytes. `/work`
is scratch and **is purged**, so the environment is disposable by design:
`scripts/setup_longleaf.sh` rebuilds it from nothing. Only this repository is
persistent, so anything worth keeping belongs in git.

`scripts/test.sbatch` runs the suite as a SLURM job:

```bash
sbatch scripts/test.sbatch
```

The suite takes about two minutes on four cores, which the login node
tolerates. The batch script exists for the tests that need the released
subject, which re-smooth a million streamlines and do not belong on a login
node: `SBCI_HCP_DIR=hcp-ya sbatch --export=ALL scripts/test.sbatch` runs them
against a downloaded copy.

## License

MIT -- see [LICENSE](LICENSE).
