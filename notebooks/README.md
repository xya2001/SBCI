# notebooks

Two analyses of HCP Young Adult subjects, saved with their outputs so they can
be read without running them. Each was run on the lab's Longleaf copies of the
data and names the files it reads at the top; point those at your own to run
it elsewhere. They need the package with its `render` extra, and
`prediction.ipynb` needs scikit-learn as well:

```bash
pip install "sbci[render] @ git+https://github.com/xya2001/SBCI.git" scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/reliability.ipynb
```

| Notebook | What it asks | What it reads | Run |
| --- | --- | --- | --- |
| [reliability.ipynb](reliability.ipynb) | How far a continuous connectome repeats: FC across two days (100 young adults), coupling across days, SC across two halves of its streamlines at three bandwidths (30), component scores, and alignment's own noise, with Schaefer-200 regions beside them where a result has a region counterpart | each day's FC (`tools/build_hcp_fc.py --session`), the cohort's SC, and `scripts/reliability_data.py`'s halves, components and alignments | about 20 minutes on 8 cores, 30 GB |
| [prediction.ipynb](prediction.ipynb) | Sex and fluid intelligence from component scores and from Schaefer-200 matrices, every fit inside the training folds, families kept whole and each fold scored on its own: what head size explains, why the components trail the regions, and what splitting families does | `scripts/prediction_data.py`'s sample, folds and features; the open-access traits; a table of families, restricted HCP data kept outside the repository | about 23 minutes on 8 cores, 2 GB |

The data each reads are made by the scripts beside the package:
`tools/build_hcp_fc.py --runs REST1_LR REST1_RL --session REST1` builds one
day's FC, and `scripts/reliability_data.py` and `scripts/prediction_data.py`
the rest (scripts/README.md). The young adults are twins and siblings; the
prediction notebook keeps every family within one fold, and reads which
subjects are related from a table of the HCP's restricted data, which stays
out of the repository and out of the notebook's output, where only counts
appear.
