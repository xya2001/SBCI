# Example figures

The figures here are drawn from ten HCP-Aging subjects, 38 to 83 years old,
five women and five men, chosen across the age range of the lab's processed
cohort and converted from the pipeline's output (smoothed SC, FC, and the
streamline endpoints with their positions) by `tools/build_hcp_cohort.py`.
The single-subject figures use the youngest subject. Regenerate the set with

```bash
python scripts/hcp_figures.py /path/to/cohort docs/figures
```

which needs the plotting extra and the cohort directory (`manifest.csv` and
one `_sc.h5` and `_fc.h5` per subject), and takes a couple of hours on four
cores because ConSEAL registers ten subjects with about 800,000 streamlines
each; on a cluster, run it in a batch job. `scripts/make_figures.py` draws the
same set from the synthetic example, as a smoke test of the figure code on a
machine without the data; no document uses those.

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
| `seed_profile.png` | `cc.seed(vertex=1234)` on the inflated surface: at each vertex, the density of streamlines between the seed (orange dot, left temporal cortex) and that vertex, relative to the strongest, for one subject |
| `smoothing_power.png` | the far ends of the streamlines touching vertex 1234 (blue dots) against its smoothed density, in two random halves of one subject's 903,797 streamlines and in the whole set: the halves' raw counts correlate at r = 0.96, the smoothed densities at 1.00 |
| `region_matrix.png` | `cc.to_atlas("Desikan")`: the same subject collapsed to 68 regions, on a log scale |
| `coupling.png` | `sc.coupling(fc)`: structure-function coupling of the subject's SC and FC, one value per vertex |
| `spherical_kernel.png` | the kernel `smooth(kernel="shk")` applies, at the released bandwidth and twice it, with its cutoffs |
| `migration_power.png` | the known warp of the recovery figure applied to the subject's own resting-state map of vertex 1234 at fsaverage resolution (163,842 vertices, r = 0.87 with the original when moved), and the map put back by ENCORE's warp carried from the grid with `migrate_warp` (r = 0.99); below, ENCORE's warp on the grid, restated on fsaverage, and the known warp there (0.35 degrees apart on average) |
| `cohort_alignment.png` | registering the ten subjects with ENCORE (onto its Karcher median) and with ConSEAL (onto the mean square-root density, since its median settles on one subject), every density from the package's smoother: each subject's cost per iteration, and the correlation between each pair of subjects' connectomes before and after (mean 0.704, then 0.758 after ENCORE and 0.780 after ConSEAL) |
| `alignment_recovery.png` | one subject's 903,797 endpoints moved by a known smooth warp (degree 4, 1.7 degrees on average) and registered back onto the undeformed subject by ENCORE and by ConSEAL, both through the package's smoother: how far each endpoint still is from where it started, vertex by vertex on the surface and as a histogram (1.66 degrees as deformed; 0.21 after ENCORE, 0.10 after ConSEAL), and the cost per iteration |
