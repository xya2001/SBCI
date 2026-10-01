# scripts

Worked programs, meant to be read top to bottom as much as run. They were
written for the lab's Longleaf cluster and carry its paths
(`/work/users/x/y/xya/...`); change the constants at the top of each to run
them elsewhere. Run them with the environment active (README.md, *Development
on Longleaf*). Those that smooth, reduce or align on the full grid belong in a
batch job, not on a login node.

| Script | What it does |
| --- | --- |
| `setup_longleaf.sh` | creates (or recreates) the development environment on `/work` |
| `test.sbatch` | runs the test suite as a SLURM job |
| `tour.py` | a worked tour of everything the package does, on sub-100307, printing the output USAGE.md quotes |
| `check_five.py` | exercises the five original entry points and prints what each returns |
| `plot_examples.py` | five worked surface figures |
| `write_exchange_file.py` | writes the exchange `.dconn.nii` for sub-100307 and verifies it end to end |
| `plot_exchange_file.py` | draws the exchange file itself, reading everything back from the `.dconn` |
| `audit_api.py` | runs every row of the README's API table on the released young adults, in order; coupling is skipped for want of FC (a batch job) |
| `align_hcp_cohort.py` | ENCORE on the lab's HCP test-retest tensors, which are on the retired 4121-vertex 0.94 grid, not ico4: a check of the algorithm on a foreign grid; the ico4 alignment of the eleven young adults is `hcp_figures.py`'s (a batch job) |
| `check_hcp_ya.py` | downloads the released young adult cohort as a user would and runs every method the README and USAGE show on it, from `to_atlas` to `reduce` and `local_test`, stopping at the first check that fails (under ten minutes on four cores; a batch job) |
| `hcp_figures.py` | draws the documentation figures from a downloaded or built cohort, and with `--full-cohort` the analysis of a whole cohort against an open-access trait, with `--groups` treating families as clusters (a batch job; docs/figures/README.md) |
| `make_figures.py` | draws the same figures from the synthetic example, a smoke test of the figure code on a machine without the data |
| `verify_all.sh` | runs every verification tier available in the environment and summarises (VERIFICATION.md) |
| `show_build.sh` | walks the whole build, printing what each stage produces |
