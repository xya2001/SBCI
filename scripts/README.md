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
| `tour.py` | a worked tour of everything the package does, printing the output USAGE.md quotes |
| `check_five.py` | exercises the five original entry points and prints what each returns |
| `plot_examples.py` | five worked surface figures |
| `write_exchange_file.py` | writes the exchange `.dconn.nii` for the example subject and verifies it end to end |
| `plot_exchange_file.py` | draws the exchange file itself, reading everything back from the `.dconn` |
| `audit_api.py` | runs every row of the README's API table on real data, in order (a batch job) |
| `align_hcp_cohort.py` | ENCORE on real multi-subject HCP data (a batch job) |
| `verify_all.sh` | runs every verification tier available in the environment and summarises (VERIFICATION.md) |
| `show_build.sh` | walks the whole build, printing what each stage produces |
