# scripts

Worked programs, meant to be read top to bottom as much as run. They were
written for the lab's Longleaf cluster and carry its paths
(`/work/users/x/y/xya/...`); change the constants at the top of each to run
them elsewhere. Run them with the environment active (USAGE.md, *Setup on
Longleaf*). Those that smooth, reduce or align on the full grid belong in a
batch job, not on a login node.

| Script | What it does |
| --- | --- |
| `setup_longleaf.sh` | creates (or recreates) the development environment on `/work` |
| `test.sbatch` | runs the test suite as a SLURM job |
| `tour.py` | a worked tour of everything the package does, on sub-100307, printing the output USAGE.md quotes |
| `plot_examples.py` | five worked surface figures |
| `write_exchange_file.py` | writes the exchange `.dconn.nii` for sub-100307 and verifies it end to end |
| `plot_exchange_file.py` | draws the exchange file itself, reading everything back from the `.dconn` |
| `audit_api.py` | runs every method in the README's capabilities table on the released young adults, in order, 22 checks (a batch job) |
| `check_hcp_ya.py` | downloads the released young adult cohort as a user would and runs the main methods on it, from `to_atlas` to `reduce` and `local_test`, stopping at the first check that fails -- the cohort loader, the test-retest functions and the export are `audit_api.py`'s (under ten minutes on four cores; a batch job) |
| `hcp_figures.py` | draws the documentation figures from a downloaded or built cohort, and with `--full-cohort` the analysis of a whole cohort against an open-access trait, with `--groups` treating families as clusters, and with `--full-fc` that cohort's structure-function coupling, `--covariates` adding head motion and size (a batch job; docs/figures/README.md) |
| `make_figures.py` | draws synthetic counterparts of the documentation figures from `sbci.example()` and `example_cohort()` into a directory you name: a smoke test of the figure code on a machine without the data, and the module `hcp_figures.py` imports its colours, rendering and data-free figures from; no document uses its output |
| `verify_all.sh` | runs every verification tier available in the environment and summarises (VERIFICATION.md) |
| `show_build.sh` | walks the whole build, printing what each stage produces |
| `reliability_data.py` | the structural half of `notebooks/reliability.ipynb`: splits thirty subjects' streamlines in two, re-smooths each half at three bandwidths, and aligns each subject's second half onto its first and the next subject's first half onto it (an array job, one subject a task), then fits a rank-20 basis to thirty other subjects of the sample and scores both halves of each split subject on it (a batch job) |
| `prediction_data.py` | the data behind `notebooks/prediction.ipynb`: three hundred young adults with a fluid-intelligence score, drawn family by family and dealt to five folds so that no family is split; for each fold a rank-15 reduction fitted to the other four, every subject scored on it (an array job, one fold a task); each subject's Schaefer-200 SC, with each region's mass with itself and the regions' areas, which turn the masses into densities as the functional PCA weighs them; and the head size and streamline count the baselines use. The family table is restricted HCP data, read from outside the repository, and the script prints counts only |
