"""The tenth list's measurement of a bootstrap that refits every model (docs/review-2026-10-05.md).

    module load python/3.12.4
    python tests/reference/tenth_list_probe.py       # about 15 minutes on 16 cores

The prediction notebook's intervals resample the held-out families with each
fold's fitted model held fixed, and its section 8 shows them too narrow where
a model is weak. The obvious repair is to refit: resample whole families within
each fold, then rerun the whole procedure -- the inner choice of components and
penalty, the scaling, the fit -- on the resampled training folds, and score the
resampled held-out ones. This measures what that gives, beside the fixed-model
bootstrap, on the notebook's own data (``scripts/prediction_data.py``) and
procedure, two hundred draws each. Where the null's spread is known -- the
weak fluid-intelligence models, section 8 of the notebook -- the refits
overshoot it, because a resampled training fold repeats families and so holds
fewer distinct ones; the notebook tests the weak results by permutation
instead. The family table is restricted HCP data, read from outside the
repository; only statistics are printed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/work/users/x/y/xya/hcp-ya")
DATA = ROOT / "prediction"
FOLDS, RANK, DRAWS = 5, 15, 200
COMPONENTS = [1, 2, 3, 5, 8, 10, 15]
PENALTY = {
    "sex": {"logisticregression__C": np.logspace(-3, 2, 6)},
    "pmat": {"ridge__alpha": np.logspace(-1, 5, 7)},
}

rows = list(csv.DictReader(open(DATA / "sample.csv", newline="")))
subjects = np.array([row["subject"] for row in rows])
fold = np.array([int(row["fold"]) for row in rows])
traits = {row["subject"]: row for row in csv.DictReader(open(ROOT / "open_access_traits.csv"))}
family_of = {
    row["subject"]: row["group"]
    for row in csv.DictReader(open(ROOT / "restricted" / "families.csv"))
}
family = np.unique([family_of[s] for s in subjects], return_inverse=True)[1]
del family_of
men = np.array([traits[s]["sex"] == "M" for s in subjects], dtype=int)
pmat = np.array([float(traits[s]["fluid_intelligence_pmat24"]) for s in subjects])
band = {"22-25": 23.5, "26-30": 28.0, "31-35": 33.0, "36+": 37.0}
age = np.array([band[traits[s]["age_bin"]] for s in subjects])
covariates = np.load(DATA / "covariates.npz")
head = np.column_stack([covariates["brain_volume"], covariates["streamlines"]])
demographic = np.column_stack([men, age, head])
none = np.empty((subjects.size, 0))

position = {s: i for i, s in enumerate(subjects)}
continuous = []
for k in range(FOLDS):
    data = np.load(DATA / f"fold{k}.npz")
    scores = np.empty((subjects.size, RANK))
    scores[[position[s] for s in data["train"]]] = data["train_scores"]
    scores[[position[s] for s in data["test"]]] = data["test_scores"]
    continuous.append(scores)
logged = np.log10(np.load(DATA / "atlas.npz")["schaefer200"] + 1e-9)
regions = [
    make_pipeline(StandardScaler(), PCA(n_components=RANK, random_state=0))
    .fit(logged[fold != k])
    .transform(logged)
    for k in range(FOLDS)
]


class First(BaseEstimator, TransformerMixin):
    """The first k of the leading n columns, and every column after them (the notebook's)."""

    def __init__(self, k=RANK, n=RANK):
        self.k, self.n = k, n

    def fit(self, x, y=None):
        """Nothing to learn; mark the step fitted, as scikit-learn checks."""
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        """Keep the first k components and every column after the n."""
        return np.hstack([x[:, : self.k], x[:, self.n :]])


def pearson(y, predicted):
    return float(np.corrcoef(y, predicted)[0, 1])


def run(scores, target, extra, folds, n_jobs=1):
    """The notebook's procedure on five index arrays, one a fold: each fold's predictions.

    Inside each training fold a four-fold split by family chooses the number of
    components and the penalty, as in the notebook; a family drawn twice stays in
    one inner fold.
    """
    y = men if target == "sex" else pmat
    out = []
    for k in range(FOLDS):
        train = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        model = LogisticRegression(max_iter=10_000) if target == "sex" else Ridge()
        if scores is None:
            pipeline, grid, x = make_pipeline(StandardScaler(), model), PENALTY[target], extra
        else:
            pipeline = make_pipeline(First(), StandardScaler(), model)
            grid, x = {"first__k": COMPONENTS, **PENALTY[target]}, np.hstack([scores[k], extra])
        search = GridSearchCV(
            pipeline,
            grid,
            cv=GroupKFold(n_splits=4),
            n_jobs=n_jobs,
            scoring="roc_auc" if target == "sex" else "r2",
        )
        search.fit(x[train], y[train], groups=family[train])
        test = x[folds[k]]
        out.append(search.predict_proba(test)[:, 1] if target == "sex" else search.predict(test))
    return out


def mean_over_folds(target, folds, predictions):
    y = men if target == "sex" else pmat
    metric = roc_auc_score if target == "sex" else pearson
    return float(
        np.mean([metric(y[index], p) for index, p in zip(folds, predictions, strict=True)])
    )


def refitted(scores, target, extra, draw):
    return mean_over_folds(target, draw, run(scores, target, extra, draw))


groups = [[np.flatnonzero(family == f) for f in np.unique(family[fold == k])] for k in range(FOLDS)]
rng = np.random.default_rng(1)
draws = [
    [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in groups]
    for _ in range(DRAWS)
]

MODELS = {
    "fluid intelligence, covariates": (None, "pmat", demographic),
    "fluid intelligence, continuous": (continuous, "pmat", none),
    "fluid intelligence, Schaefer-200": (regions, "pmat", none),
    "sex, head size and streamlines": (None, "sex", head),
    "sex, Schaefer-200 with head size": (regions, "sex", head),
}
sample = [np.flatnonzero(fold == k) for k in range(FOLDS)]
refits, fixes = {}, {}
for name, (scores, target, extra) in MODELS.items():
    # The sample's own models, their predictions rescored on each draw's held-out families.
    predicted = np.empty(subjects.size)
    for index, p in zip(sample, run(scores, target, extra, sample, n_jobs=-1), strict=True):
        predicted[index] = p
    observed = mean_over_folds(target, sample, [predicted[index] for index in sample])
    fixes[name] = np.array(
        [mean_over_folds(target, d, [predicted[index] for index in d]) for d in draws]
    )
    refits[name] = np.array(
        Parallel(n_jobs=-1)(delayed(refitted)(scores, target, extra, d) for d in draws)
    )
    print(
        f"{name:33s} {observed:.3f}: spread over {DRAWS} draws {fixes[name].std():.3f} with the "
        f"models held fixed, {refits[name].std():.3f} refitted (whose mean is "
        f"{refits[name].mean():.3f})"
    )
gap = "sex, Schaefer-200 with head size", "sex, head size and streamlines"
print(
    f"the regions added to head size: spread {(fixes[gap[0]] - fixes[gap[1]]).std():.4f} with the "
    f"models held fixed, {(refits[gap[0]] - refits[gap[1]]).std():.4f} refitted"
)
