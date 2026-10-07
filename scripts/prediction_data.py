"""The data behind notebooks/prediction.ipynb: a family-aware sample, one reduction per fold.

    python scripts/prediction_data.py sample      # the sample and its folds
    python scripts/prediction_data.py fold K      # K = 0..4, a batch job each
    python scripts/prediction_data.py atlas       # Schaefer-200 SC for the sample
    python scripts/prediction_data.py covariates  # head size and streamline count

The lab's paths on Longleaf. ``sample`` draws whole families of the young
adults with a fluid-intelligence score until there are 300 subjects, and
deals the families to five folds of near-equal size, so that no family is
split between training and test. The families are restricted HCP data: the
sample and fold file stays under ``/work`` and the scripts print counts only.

``fold K`` fits the reduction -- the cohort mean and a rank-15 basis -- to
the four other folds and scores every subject on it with ``project``, the
held-out fold ``K`` and the training subjects alike, so the two are scored
the same way and the held-out subjects never touch the basis. ``atlas`` computes each
subject's Schaefer-200 SC matrix, for the atlas-based comparison, whose PCA
the notebook fits inside each training fold the same way. ``covariates``
records what the baselines use: each subject's brain-mask volume, from the
pipeline's skull-stripped T1 (``t1_brain.nii.gz``, 1.25 mm), as a measure of
head size, and the number of streamlines in its SC file.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

import sbci
from sbci.io import read_header

ROOT = Path("/work/users/x/y/xya/hcp-ya")
SC = ROOT / "full" / "data"
OUT = ROOT / "prediction"
PIPELINE = Path("/overflow/zzhanglab/encore_project/encore_paper_code/prediction_subs")
SIZE = 300
FOLDS = 5
RANK = 15
SEED = 2026


def read_sample() -> list:
    with open(OUT / "sample.csv", newline="") as handle:
        return [(row["subject"], int(row["fold"])) for row in csv.DictReader(handle)]


def sample() -> None:
    with open(ROOT / "open_access_traits.csv", newline="") as handle:
        scored = [
            row["subject"]
            for row in csv.DictReader(handle)
            if row["fluid_intelligence_pmat24"].strip() not in ("", "NA")
        ]
    with open(ROOT / "restricted" / "families.csv", newline="") as handle:
        family = {row["subject"]: row["group"] for row in csv.DictReader(handle)}
    members = defaultdict(list)
    for subject in scored:
        members[family[subject]].append(subject)
    rng = np.random.default_rng(SEED)
    chosen, families = [], []
    for name in rng.permutation(sorted(members)):
        if len(chosen) >= SIZE:
            break
        chosen += members[name]
        families.append(name)
    load = [0] * FOLDS
    fold_of = {}
    for name in sorted(families, key=lambda f: (-len(members[f]), f)):
        k = int(np.argmin(load))
        fold_of[name] = k
        load[k] += len(members[name])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sample.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "fold"])
        for subject in sorted(chosen):
            writer.writerow([subject, fold_of[family[subject]]])
    print(f"{len(chosen)} subjects in {len(families)} families; folds of {sorted(load)}")


def fold(k: int) -> None:
    rows = read_sample()
    train = [s for s, f in rows if f != k]
    test = [s for s, f in rows if f == k]
    start = time.time()
    reduction = sbci.reduce([SC / f"{s}_sc.h5" for s in train], rank=RANK)
    fitted = time.time() - start

    def scores(subjects):
        dense = (sbci.load(SC / f"{s}_sc.h5").dense(np.float64) for s in subjects)
        return np.vstack([sbci.project(reduction, matrix) for matrix in dense])

    train_scores, test_scores = scores(train), scores(test)
    np.savez_compressed(
        OUT / f"fold{k}.npz",
        train=np.array(train),
        test=np.array(test),
        train_scores=train_scores,
        test_scores=test_scores,
        fit_scores=reduction.scores,
        explained=reduction.explained,
        scales=reduction.scales,
        basis=reduction.basis,
    )
    drift = np.abs(train_scores - reduction.scores).max() / np.abs(reduction.scores).max()
    print(
        f"fold {k}: rank {RANK} on {len(train)} training subjects in {fitted:.0f}s, "
        f"{reduction.explained[-1]:.1%} explained; all {len(train) + len(test)} scored in "
        f"{time.time() - start - fitted:.0f}s; the training scores differ from the fit's own "
        f"by {drift:.1e} of the largest"
    )


def atlas() -> None:
    subjects = [s for s, _ in read_sample()]
    schaefer = sbci.load_atlas("Schaefer200")
    upper = np.triu_indices(schaefer.n_regions, 1)
    matrices = np.vstack([sbci.load(SC / f"{s}_sc.h5").to_atlas(schaefer)[upper] for s in subjects])
    np.savez_compressed(OUT / "atlas.npz", subjects=np.array(subjects), schaefer200=matrices)
    print(f"Schaefer-200 SC for {len(subjects)} subjects: {matrices.shape[1]:,} region pairs each")


def covariates() -> None:
    subjects = [s for s, _ in read_sample()]
    volume, streamlines = [], []
    for subject in subjects:
        image = nib.load(PIPELINE / subject.removeprefix("sub-") / "t1_brain.nii.gz")
        voxel = float(np.prod(image.header.get_zooms()[:3]))
        volume.append(np.count_nonzero(np.asarray(image.dataobj)) * voxel / 1e6)
        streamlines.append(read_header(SC / f"{subject}_sc.h5")["n_endpoints"])
    np.savez_compressed(
        OUT / "covariates.npz",
        subjects=np.array(subjects),
        brain_volume=np.array(volume),
        streamlines=np.array(streamlines),
    )
    print(
        f"{len(subjects)} subjects: brain-mask volume {min(volume):.2f} to {max(volume):.2f} L, "
        f"{min(streamlines):,} to {max(streamlines):,} streamlines"
    )


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "sample":
        sample()
    elif command == "fold":
        fold(int(sys.argv[2]))
    elif command == "atlas":
        atlas()
    elif command == "covariates":
        covariates()
    else:
        raise SystemExit("sample, fold K, atlas or covariates")
