# Example figures

The figures here are discussed in [RESULTS.md](../RESULTS.md); they are drawn from the eleven HCP Young Adult subjects of the
example cohort (`sbci download hcp-ya`), 22 to 35 years old by the HCP's
open-access age bands. They were rebuilt on the ico4 grid from the lab's SBCI
pipeline output by `tools/build_hcp_cohort.py`: each subject's SC is smoothed
by the package from the pipeline's snapped streamline endpoints on the
subject's FreeSurfer-registered sphere, and those endpoints are stored beside
it. Their FC was built from the HCP's own cleaned resting-state runs by
`tools/build_hcp_fc.py`. The single-subject figures use the first subject,
the tutorial subject. Regenerate the set with

```bash
sbci download hcp-ya --out hcp-ya
python scripts/hcp_figures.py hcp-ya docs/figures --anatomy PIPELINE/SUBJECT
```

which needs the plotting extra, and the render extra for PyVista.
`--anatomy` names the tutorial subject's pipeline directory, for the FreeSurfer sulcal
depth and registered spheres the warp-migration figure reads; without it that
figure is skipped. The set took about three hours on eight cores, most of
it ConSEAL registering eleven subjects with 734,039 to 988,788 streamlines
each, until PORTING.md item 36 about halved ConSEAL's time; on a cluster, run
it in a batch job. `--device gpu` registers the
cohort on a CUDA GPU instead (USAGE.md, *Running the alignments on a GPU*):
on a V100 ENCORE takes 48 s and ConSEAL 6.6 minutes, to the same figure. `scripts/make_figures.py`
draws the same set from the synthetic example, as a smoke test of the figure
code on a machine without the data; no document uses those.

The analysis of the 946 young adults with complete pipeline output, published
on UNC Dataverse (`sbci download hcp-ya --all --out hcp-ya-full`; `--sc-only` fetches the SC alone, which the coupling run below cannot use), has its own flags: `scripts/hcp_figures.py hcp-ya
docs/figures --full-cohort DIR --traits CSV --groups CSV --covariates CSV` draws
`cohort_trait_scores.png` and `cohort_sex_effect.png`, and
`cohort_trait_effect.png` when a component tracks the trait, from the built
files, a table of open-access fields, a table of each subject's family and a
table of further covariates added to every test: the documents pass head size,
FreeSurfer's estimated intracranial volume in litres.
The young adults are twins and siblings, and the tests treat the families as
clusters; family membership is restricted HCP data, so that table stays out
of the repository. Adding `--full-fc FCDIR`, the directory of the cohort's FC
files (`tools/build_hcp_fc.py`), runs the coupling analysis instead and draws
`cohort_coupling.png` and `cohort_coupling_sex.png`; `--covariates CSV` adds a
table with a `subject` column and one column per further covariate. The
analysis in the documents passes head motion, the mean over the four runs of
each run's `Movement_RelativeRMS_mean.txt`, and intracranial volume, the
`EstimatedTotalIntraCranialVol` of each subject's `T1w/<id>/stats/aseg.stats`,
both from the HCP's open-access release.

Every surface is drawn on FreeSurfer's fsaverage (163,842 vertices per
hemisphere) with the map interpolated onto it and rendered through PyVista
with smooth normals, a light kit and a specular highlight
(`plot(mesh="fsaverage", engine="pyvista")`, the `render` extra), shaded by
FreeSurfer's sulcal depth (dark in the sulci); the medial wall,
which is the cut surface between the hemispheres and carries no cortex, is
left flat gray. Maps are thresholded at a few percent of their peak so that
the shading shows where there is nothing to see.

| Figure | What it shows |
| --- | --- |
| `seed_profile.png` | `cc.seed(vertex=1234)` on the inflated surface: at each vertex, the density of streamlines between the seed (orange dot, left temporal cortex) and that vertex, relative to the strongest, for the tutorial subject |
| `smoothing_power.png` | the far ends of the streamlines touching vertex 1234 (blue dots) against its smoothed density, in two random halves of the tutorial subject's 803,741 streamlines and in the whole set: 32 and 26 streamlines touch the vertex in the two halves, whose raw counts correlate at r = 0.41; the smoothed densities correlate at 0.99, and the same vertex in the other ten subjects agrees with the tutorial subject's at r = 0.54 to 0.80 |
| `coupling.png` | `sc.coupling(fc)` for each of the eleven young adults, averaged: at each vertex, the cosine similarity of its SC and FC profiles, coloured from the map's 2nd to 98th percentile (0.05 to 0.42). Highest in visual cortex (0.36 to 0.41 in the pericalcarine, cuneus and lateral occipital regions), lowest in the cingulate and entorhinal cortex (0.05 to 0.09) and low in the insula |
| `region_matrix.png` | `cc.to_atlas("Desikan")`: the same subject collapsed to 68 regions, on a log scale |
| `spherical_kernel.png` | the kernel `smooth(kernel="shk")` applies, at the released bandwidth and twice it, with its cutoffs |
| `alignment_recovery.png` | the tutorial subject's streamline endpoints moved by a known smooth warp (degree 4, 1.6 degrees on average) and registered back onto the undeformed subject by ENCORE and by ConSEAL, both through the package's smoother: how far each endpoint still is from where it started, vertex by vertex on the surface and as a histogram (1.63 degrees as deformed; 0.20 after ENCORE, 0.02 after ConSEAL at its defaults), and the cost per iteration |
| `migration_power.png` | the same known warp applied to the tutorial subject's own FreeSurfer sulcal depth at fsaverage resolution (163,842 vertices), shown with what it changes in the map (r = 0.95 with the original once moved), and what is left once ENCORE's warp, carried from the grid with `migrate_warp`, puts it back (r = 0.996); below, ENCORE's warp on the grid, restated on fsaverage, and the known warp there, 0.32 degrees apart on average on the left hemisphere shown (0.33 over both) |
| `cohort_alignment.png` | registering the eleven subjects with ENCORE and with ConSEAL, each onto its own Karcher median, every density from the package's smoother: each subject's cost per iteration, and the correlation between each of the 55 pairs of subjects' connectomes before and after (mean 0.7163, then 0.7615 after ENCORE and 0.7900 after ConSEAL at its defaults) |
| `cohort_trait_scores.png` | a rank-20 FPCA of the 943 young adults with a fluid-intelligence score: the two components most associated with it given sex, age band, streamline count and intracranial volume, families as clusters, each subject's score against the number of PMAT24 items answered correctly, with the mean in each fifth of that range and its 95% interval (components 12 and 13: r = -0.06 and 0.11 as drawn, unadjusted, and partial r -0.09 and 0.09 given the covariates, both at adjusted p 0.087: neither survives the correction) |
| `cohort_sex_effect.png` | the effect map over the 3 components of the same fit that track sex given the rest, head size among it, families as clusters: the fitted difference in connectivity between women and men, relative to its largest value, in the right cuneus and pericalcarine cortex, where it is higher in men, with a smaller patch near the right occipital pole higher in women; without head size, 13 components and both occipital poles. It describes those components, not where sex differs: tested at each vertex, sex shows at 433, most of them higher in women and the most in the insula (RESULTS.md) |
| `cohort_coupling.png` | the mean of the coupling maps of the 903 young adults with four complete resting-state runs, coloured from its 2nd to 98th percentile (0.05 to 0.40): the eleven's map (r = 0.972), and two halves of the cohort split by family reproduce it at r = 0.998 |
| `cohort_coupling_sex.png` | where coupling differs between women and men among the 900 of them with a fluid-intelligence score, given fluid intelligence, age band, streamline count, head motion and intracranial volume, families as clusters: the fitted difference at the 325 of 4,685 vertices significant at FDR 0.05, 253 of them higher in men (inferior parietal, superior frontal and opercular cortex, the insula, early visual cortex); 44% of the 72 higher in women lie within two grid rings of the medial wall, where 6% of cortex lies |
