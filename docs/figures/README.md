# Example figures

Every figure here is drawn from the synthetic data the package builds itself,
`sbci.example()` and `sbci.example_cohort()`, so none of it is a measurement of
anyone's brain. Regenerate the set with

```bash
python scripts/make_figures.py docs/figures
```

which needs the plotting extra and takes a few minutes (a rank-4 FPCA on ten
subjects and one ConSEAL registration); on a cluster, run it in a batch job.
On the surface views, the gray mesh is the medial wall, which carries no
cortex and no connectivity.

| Figure | What it shows |
| --- | --- |
| `seed_profile.png` | `cc.seed(vertex=1234)` on the inflated surface: the density of connections from one left-hemisphere vertex to every other |
| `region_matrix.png` | `cc.to_atlas("Desikan")`: the same subject collapsed to 68 regions, on a log scale |
| `coupling.png` | `sc.coupling(fc)`: structure-function coupling of the matching SC and FC examples, one value per vertex |
| `spherical_kernel.png` | the kernel `smooth(kernel="shk")` applies, at the released bandwidth and twice it, with its cutoffs |
| `cohort_truth.png` | the planted bundle of `sbci.example_cohort()`, the field whose weight scales with age |
| `cohort_effect.png` | what `reduce` and `local_test` recover: the effect map of the significant component |
| `cohort_scores.png` | that component's scores against the synthetic age, with the fitted line and the adjusted p-value |
| `conseal_cost.png` | ConSEAL registering one synthetic subject onto another: the cost at each iteration, relative to the start |
