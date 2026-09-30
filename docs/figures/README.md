# Example figures

The figures here are drawn from the ten HCP Young Adult subjects of the
example cohort (`sbci download hcp-ya`) that were drawn at random, five women
and five men, 22 to 35 years old by the HCP's open-access age bands; the
cohort's eleventh subject, sub-100307, joined it after they were drawn. They were rebuilt on the ico4
grid from the lab's SBCI pipeline output by `tools/build_hcp_cohort.py
--layout young-adult`: each subject's SC is smoothed by the package from the
pipeline's snapped streamline endpoints on the subject's FreeSurfer-registered
sphere, and those endpoints are stored beside it. The single-subject figures
use sub-103010. Draw the set from a downloaded copy with

```bash
sbci download hcp-ya --out hcp-ya
python scripts/hcp_figures.py hcp-ya docs/figures
```

which now draws it from all eleven, with sub-100307 first.

which needs the plotting extra, and the render extra for PyVista. It takes
about an hour and a half on eight cores, most of it ConSEAL registering ten
subjects with 734,039 to 988,788 streamlines each; on a cluster, run it in a
batch job. `scripts/make_figures.py` draws the same set from the synthetic
example, as a smoke test of the figure code on a machine without the data; no
document uses those.

The analysis of the 946 young adults with complete pipeline output, which
are not distributed, has its own flags: `scripts/hcp_figures.py hcp-ya
docs/figures --full-cohort DIR --traits CSV` draws `cohort_trait_*.png` from
the built files and a table of open-access fields. Four
figures come from HCP-Aging subjects, whose files are not distributed either.
`coupling.png` and `migration_power.png` need resting-state data, which the
young adult files do not carry yet, and `cohort_age_*.png` are the analysis of
age on 528 HCP-Aging subjects; until young adult resting-state data are in
hand, these stay as they are.

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
| `seed_profile.png` | `cc.seed(vertex=1234)` on the inflated surface: at each vertex, the density of streamlines between the seed (orange dot, left temporal cortex) and that vertex, relative to the strongest, for sub-103010 |
| `smoothing_power.png` | the far ends of the streamlines touching vertex 1234 (blue dots) against its smoothed density, in two random halves of sub-103010's 988,788 streamlines and in the whole set: 41 and 28 streamlines touch the vertex in the two halves, whose raw counts correlate at r = 0.46; the smoothed densities correlate at 1.00 |
| `region_matrix.png` | `cc.to_atlas("Desikan")`: the same subject collapsed to 68 regions, on a log scale |
| `coupling.png` | HCP-Aging: `sc.coupling(fc)`, structure-function coupling of one HCP-Aging subject's SC and FC, one value per vertex |
| `spherical_kernel.png` | the kernel `smooth(kernel="shk")` applies, at the released bandwidth and twice it, with its cutoffs |
| `alignment_recovery.png` | sub-103010's streamline endpoints moved by a known smooth warp (degree 4, 1.7 degrees on average) and registered back onto the undeformed subject by ENCORE and by ConSEAL, both through the package's smoother: how far each endpoint still is from where it started, vertex by vertex on the surface and as a histogram (1.65 degrees as deformed; 0.22 after ENCORE, 0.12 after ConSEAL), and the cost per iteration |
| `migration_power.png` | HCP-Aging: the same known warp applied to one HCP-Aging subject's endpoints and to its resting-state map of vertex 1234 at fsaverage resolution (163,842 vertices, r = 0.87 with the original when moved), and the map put back by ENCORE's warp carried from the grid with `migrate_warp` (r = 0.99); below, ENCORE's warp on the grid, restated on fsaverage, and the known warp there, 0.33 degrees apart on average over the left hemisphere shown and 0.35 over both |
| `cohort_alignment.png` | registering the ten subjects with ENCORE (onto its Karcher median) and with ConSEAL (onto the mean square-root density), every density from the package's smoother: each subject's cost per iteration, and the correlation between each pair of subjects' connectomes before and after (mean 0.715, then 0.761 after ENCORE and 0.785 after ConSEAL) |
| `cohort_age_scores.png` | HCP-Aging: a rank-20 FPCA of 528 HCP-Aging subjects, the four components most associated with age given sex and streamline count, each subject's score against age (components 14, 11, 7 and 3; r = 0.32, 0.28, -0.17, -0.11) |
| `cohort_age_effect.png` | HCP-Aging: the effect map over the five components significant at FDR 0.05, connectivity falling with age at the frontal poles and in rostral middle frontal cortex and rising in inferior temporal cortex and at the temporal poles, relative to its largest value |
