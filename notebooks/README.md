# notebooks

The paper's three worked examples and two analyses of HCP Young Adult subjects,
saved with their outputs so they can be read without running them. Each was run
on the lab's Longleaf copies of the data and names the files it reads at the
top; point those at your own to run it elsewhere. The first two examples read
only the eleven example subjects, which their first cell fetches, and run in a
few minutes. They need the package's `notebooks` extra: its `render` extra,
scikit-learn (`prediction.ipynb` and `03_cohort.ipynb`; joblib comes with it),
bctpy for the cohort notebook's graph measures, whose cells it skips without,
and jupyter:

```bash
pip install "sbci[notebooks] @ git+https://github.com/xya2001/SBCI.git"
mkdir -p ~/sbci-run && cp notebooks/01_explore.ipynb ~/sbci-run/ && cd ~/sbci-run
jupyter nbconvert --to notebook --execute --inplace 01_explore.ipynb
```

Execute a copy in a folder of its own, as above: a notebook runs where it sits, and
the first two write there, the eleven subjects' files (1 GB, into `hcp-ya/`) and the
exports. On a machine without a display, set
`VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow` before executing, so that VTK draws
off screen without first trying an X display and printing a warning into the
notebook.

Each notebook's last cell prints its run: the wall time, the cores, and the kernel's
peak memory (its worker processes, where it has them, not counted); the table's *Run*
column quotes those cells.

| Notebook | What it asks | What it reads | Run |
| --- | --- | --- | --- |
| [01_explore.ipynb](01_explore.ipynb) | The paper's Example 1: what one subject's continuous SC holds; a seed profile, and how smoothing makes it repeat across two halves of the streamlines; the Desikan and Schaefer-200 matrices as mass and as mean, and what one regional entry gathers; a map exported for Workbench; and the matrices at a second bandwidth | the eleven example subjects, through `fetch_cohort` as the paper's listing calls it | 2.6 minutes on 8 cores, 2.6 GB |
| [02_coupling.ipynb](02_coupling.ipynb) | The paper's Example 2: one vertex's SC and FC profiles; coupling in its three forms; the eleven's map, where it is highest and lowest, and how alike their maps are; the pattern inside the precentral gyrus; how much of the cingulate's low coupling is nearness to the medial wall; and coupling with negative FC set to zero and with FC rebuilt without global signal regression | the eleven, through `fetch_cohort`, and for its last check their FC rebuilt without global signal regression (`tools/build_hcp_fc.py`), which it skips without | 2.7 minutes on 4 cores, 5.1 GB |
| [03_cohort.ipynb](03_cohort.ipynb) | The paper's Example 3: the 943 young adults with a fluid-intelligence score; a rank-20 functional PCA, and sex and fluid intelligence on its components and at every vertex, families as clusters; the results in Desikan terms; graph measures and a network-based statistic with whole-family sign flips; and the sensitivity checks | the cohort's SC; the rank-20 fit `scripts/hcp_figures.py --full-cohort` saves; `scripts/cohort_data.py`'s strengths and region matrices; the open-access traits, head size and head motion; a table of families, restricted HCP data kept outside the repository | 11.4 minutes on 8 cores, 1.5 GB, with the saved fit (a fresh fit takes four to five and a half hours) |
| [reliability.ipynb](reliability.ipynb) | How far a continuous connectome repeats: FC across two days (100 young adults), coupling across days, SC across two halves of its streamlines at three bandwidths (30), component scores, and alignment's own noise, with Schaefer-200 regions beside them for identification and single connections | each day's FC (`tools/build_hcp_fc.py --session`), the cohort's SC, and `scripts/reliability_data.py`'s halves, components and alignments; a table of families, restricted HCP data kept outside the repository | about 17 minutes on 8 cores, 37 GB |
| [prediction.ipynb](prediction.ipynb) | Sex and fluid intelligence from component scores and from Schaefer-200 matrices, every fit inside the training folds, families kept whole and each fold scored on its own: what head size explains, why the components trail the regions, what splitting families does, and how far the intervals can be trusted | `scripts/prediction_data.py`'s sample, folds and features; the open-access traits; a table of families, restricted HCP data kept outside the repository | about 16 minutes on 16 cores, 5 GB |

The data they read are made by the scripts beside the package:
`tools/build_hcp_fc.py --runs REST1_LR REST1_RL --session REST1` builds one
day's FC, and `scripts/reliability_data.py`, `scripts/prediction_data.py` and
`scripts/cohort_data.py` the rest (scripts/README.md). The young adults are
twins and siblings, and three notebooks read which subjects are related from a
table of the HCP's restricted data: the reliability notebook resamples whole
families for its intervals, the prediction notebook keeps every family within
one fold, and the cohort notebook takes families as clusters and flips the
signs of whole families in its network-based statistic. The table stays out of
the repository and out of the notebooks' output, where only counts appear, and
none of the three names a subject. The first two examples use no family table;
their only subject ids are the file names `fetch_cohort` reports.
