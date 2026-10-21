# notebooks

The paper's three worked examples and two analyses of HCP Young Adult subjects,
saved with their outputs so they can be read without running them. Each was run
on the lab's Longleaf copies of the data and names the files it reads at the
top; point those at your own to run it elsewhere. The first two examples read
only the eleven example subjects, which their first cell fetches, and run in a
few minutes. They need the package with its `render` extra; `prediction.ipynb`
needs scikit-learn as well, `03_cohort.ipynb` joblib, which scikit-learn brings,
and bctpy for its graph measures, whose cells it skips without:

```bash
pip install "sbci[render] @ git+https://github.com/xya2001/SBCI.git" scikit-learn bctpy jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore.ipynb
```

| Notebook | What it asks | What it reads | Run |
| --- | --- | --- | --- |
| [01_explore.ipynb](01_explore.ipynb) | The paper's Example 1: what one subject's continuous SC holds; a seed profile, and how smoothing makes it repeat across two halves of the streamlines; the Desikan and Schaefer-200 matrices as mass and as mean, and what one regional entry gathers; a map exported for Workbench; and the matrices at a second bandwidth | the eleven example subjects, through `fetch_cohort` as the paper's listing calls it | about 3 minutes on 8 cores, 4 GB |
| [02_coupling.ipynb](02_coupling.ipynb) | The paper's Example 2: one vertex's SC and FC profiles; coupling in its three forms; the eleven's map, where it is highest and lowest, and how alike their maps are; the pattern inside the precentral gyrus; how much of the cingulate's low coupling is nearness to the medial wall; and coupling with negative FC set to zero and with FC rebuilt without global signal regression | the eleven, through `fetch_cohort`, and for its last check their FC rebuilt without global signal regression (`tools/build_hcp_fc.py`), which it skips without | about 3.5 minutes on 4 cores, 6 GB |
| [03_cohort.ipynb](03_cohort.ipynb) | The paper's Example 3: the 943 young adults with a fluid-intelligence score; a rank-20 functional PCA, and sex and fluid intelligence on its components and at every vertex, families as clusters; the results in Desikan terms; graph measures and a network-based statistic with whole-family sign flips; and the sensitivity checks | the cohort's SC; the rank-20 fit `scripts/hcp_figures.py --full-cohort` saves; `scripts/cohort_data.py`'s strengths and region matrices; the open-access traits, head size and head motion; a table of families, restricted HCP data kept outside the repository | 7 to 11 minutes on 8 cores, 2 GB, with the saved fit (a fresh fit takes four to five hours) |
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
