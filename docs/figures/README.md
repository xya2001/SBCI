# Example figures

The figures here are drawn from the eleven HCP Young Adult subjects of the
example cohort (`sbci download hcp-ya`), 22 to 35 years old by the HCP's
open-access age bands. They were rebuilt on the ico4 grid from the lab's SBCI
pipeline output by `tools/build_hcp_cohort.py`: each subject's SC is smoothed
by the package from the pipeline's snapped streamline endpoints on the
subject's FreeSurfer-registered sphere, and those endpoints are stored beside
it. The single-subject figures use the first subject, sub-100307. Regenerate
the set with

```bash
sbci download hcp-ya --out hcp-ya
python scripts/hcp_figures.py hcp-ya docs/figures --anatomy PIPELINE/100307
```

which needs the plotting extra, and the render extra for PyVista.
`--anatomy` names sub-100307's pipeline directory, for the FreeSurfer sulcal
depth and registered spheres the warp-migration figure reads; without it that
figure is skipped. The set takes about two and a half hours on eight cores,
most of it ConSEAL registering eleven subjects with 734,039 to 988,788
streamlines each; on a cluster, run it in a batch job. `scripts/make_figures.py`
draws the same set from the synthetic example, as a smoke test of the figure
code on a machine without the data; no document uses those.

The analysis of the 946 young adults with complete pipeline output, which are
not distributed, has its own flags: `scripts/hcp_figures.py hcp-ya
docs/figures --full-cohort DIR --traits CSV` draws `cohort_trait_*.png` from
the built files and a table of open-access fields. There is no coupling
figure: coupling needs FC, which the young adult files do not have yet.

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
| `seed_profile.png` | `cc.seed(vertex=1234)` on the inflated surface: at each vertex, the density of streamlines between the seed (orange dot, left temporal cortex) and that vertex, relative to the strongest, for sub-100307 |
| `smoothing_power.png` | the far ends of the streamlines touching vertex 1234 (blue dots) against its smoothed density, in two random halves of sub-100307's 803,741 streamlines and in the whole set: 32 and 26 streamlines touch the vertex in the two halves, whose raw counts correlate at r = 0.41; the smoothed densities correlate at 0.99 |
| `region_matrix.png` | `cc.to_atlas("Desikan")`: the same subject collapsed to 68 regions, on a log scale |
| `spherical_kernel.png` | the kernel `smooth(kernel="shk")` applies, at the released bandwidth and twice it, with its cutoffs |
| `alignment_recovery.png` | sub-100307's streamline endpoints moved by a known smooth warp (degree 4, 1.7 degrees on average) and registered back onto the undeformed subject by ENCORE and by ConSEAL, both through the package's smoother: how far each endpoint still is from where it started, vertex by vertex on the surface and as a histogram (1.63 degrees as deformed; 0.20 after ENCORE, 0.11 after ConSEAL), and the cost per iteration |
| `migration_power.png` | the same known warp applied to sub-100307's own FreeSurfer sulcal depth at fsaverage resolution (163,842 vertices), shown with what it changes in the map (r = 0.95 with the original once moved), and what is left once ENCORE's warp, carried from the grid with `migrate_warp`, puts it back (r = 0.996); below, ENCORE's warp on the grid, restated on fsaverage, and the known warp there, 0.33 degrees apart on average |
| `cohort_alignment.png` | registering the eleven subjects with ENCORE (onto its Karcher median) and with ConSEAL (onto the mean square-root density, since its median settles on one subject), every density from the package's smoother: each subject's cost per iteration, and the correlation between each of the 55 pairs of subjects' connectomes before and after (mean 0.716, then 0.761 after ENCORE and 0.785 after ConSEAL) |
| `cohort_trait_scores.png` | a rank-20 FPCA of the 943 young adults with a fluid-intelligence score: the two components most associated with it given sex, age band and streamline count, each subject's score against the number of PMAT24 items answered correctly, with the mean in each fifth of that range and its 95% interval (component 13, r = 0.11, adjusted p 0.023; component 7, r = 0.10, 0.052) |
| `cohort_trait_effect.png` | the effect map of component 13, the one significant at FDR 0.05: connectivity higher with higher scores in right medial occipital cortex, relative to its largest value |
