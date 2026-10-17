"""The structural half of notebooks/reliability.ipynb: split streamlines, re-smoothed and aligned.

    python scripts/reliability_data.py split SUBJECT_INDEX   # one subject a task; see below
    python scripts/reliability_data.py components             # after every split task
    python scripts/reliability_data.py components --own-basis # the same, fitted to the split

The lab's paths on Longleaf; the sample is ``reliability/manifest.csv``, a
hundred young adults with all four resting-state runs, and the first thirty
are split here. For each, ``split``:

1. divides the subject's streamlines into two random halves (seeded by the
   subject's number, so a rerun draws the same halves) and smooths each with
   the package's kernel at three bandwidths, 0.0025, 0.005 (the cohort's) and
   0.01, saving per half and bandwidth the vertex strength, the Schaefer-200
   matrix and the cortical pairs of the whole connectome;
2. at 0.005 keeps both halves as computational files, for ``components``;
3. aligns, with ENCORE, the subject's second half onto its first -- two
   samples of one brain, whose warp is alignment's noise -- and the next
   subject's first half onto this subject's, the warp between two brains.

``components`` fits a rank-20 reduction to thirty other subjects of the
sample, none of them split, and scores both halves of each split subject on
it with ``project``: a basis fitted to the split subjects themselves would
have seen both halves of each, and flattered their agreement. ``--own-basis``
fits exactly that basis instead, to the thirty split subjects' whole
connectomes, for the comparison the notebook prints beside the first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci.connectome import ContinuousConnectome

ROOT = Path("/work/users/x/y/xya/hcp-ya")
SC = ROOT / "full" / "data"
OUT = ROOT / "reliability" / "split"
BANDWIDTHS = (0.0025, 0.005, 0.01)
SPLIT = 30


def sample() -> list:
    with open(ROOT / "reliability" / "manifest.csv") as handle:
        return [line.strip() for line in handle.readlines()[1:] if line.strip()][:SPLIT]


def others() -> list:
    """The next thirty of the sample, none of them split: the subjects the basis is fitted to."""
    with open(ROOT / "reliability" / "manifest.csv") as handle:
        return [line.strip() for line in handle.readlines()[1:] if line.strip()][SPLIT : 2 * SPLIT]


def halves(subject: str):
    """The subject's connectome and its two halves of streamlines, the same on every run."""
    connectome = sbci.load(SC / f"{subject}_sc.h5")
    endpoints = connectome.endpoints
    rng = np.random.default_rng(int(subject.removeprefix("sub-")))
    first = rng.permutation(endpoints.n_streamlines) < endpoints.n_streamlines // 2
    return connectome, endpoints.take(first), endpoints.take(~first)


def smoothed(connectome, endpoints, bandwidth):
    """A connectome smoothed from these endpoints alone, as the cohort's were made."""
    holder = ContinuousConnectome(
        connectome.data, connectome.area, connectome.mask, connectome.metadata, endpoints=endpoints
    )
    return holder.smooth(kernel="shk", bandwidth=bandwidth, mask_medial_wall=True)


def cortical_pairs(mask) -> np.ndarray:
    """Positions in the condensed upper triangle of the pairs of two cortical vertices."""
    n = mask.size
    rows, cols = np.triu_indices(n, 1)
    return np.flatnonzero(mask[rows] & mask[cols])


def displacement(warp) -> np.ndarray:
    """How far a warp moves each vertex, in degrees, on the bundled sphere."""
    sphere = np.array(
        sbci.load_surface("sphere").vertices, dtype=np.float64
    )  # a copy: the bundled one is frozen
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    moved = np.vstack([warp.lh_vertices @ warp.lh_rotation, warp.rh_vertices @ warp.rh_rotation])
    moved /= np.linalg.norm(moved, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((moved * sphere).sum(axis=1), -1.0, 1.0)))


def split(index: int) -> None:
    subjects = sample()
    subject, following = subjects[index], subjects[(index + 1) % len(subjects)]
    out = OUT / subject
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    connectome, first, second = halves(subject)
    pairs = cortical_pairs(connectome.mask)
    print(
        f"{subject}: {connectome.endpoints.n_streamlines:,} streamlines, halves of "
        f"{first.n_streamlines:,} and {second.n_streamlines:,}",
        flush=True,
    )
    kept = {}
    for bandwidth in BANDWIDTHS:
        for name, part in (("A", first), ("B", second)):
            half = smoothed(connectome, part, bandwidth)
            np.savez_compressed(
                out / f"half{name}_bw{bandwidth}.npz",
                strength=half.dense(np.float64) @ half.area,
                schaefer200=half.to_atlas("Schaefer200"),
                pairs=np.asarray(half.data[pairs], dtype=np.float32),
            )
            if bandwidth == 0.005:
                kept[name] = half
                half.save(out / f"{subject}_split-{name}_sc.h5")
        print(
            f"  bandwidth {bandwidth}: both halves smoothed ({time.time() - start:.0f}s)",
            flush=True,
        )
    within = sbci.align([kept["B"]], template=kept["A"])
    neighbour, other, _ = halves(following)
    between = sbci.align([smoothed(neighbour, other, 0.005)], template=kept["A"])
    np.savez_compressed(
        out / "alignment.npz",
        within=displacement(within.warps[0]),
        between=displacement(between.warps[0]),
    )
    print(f"  aligned within and between ({time.time() - start:.0f}s)", flush=True)


def components(own_basis: bool = False) -> None:
    subjects = sample()
    fitted_to = subjects if own_basis else others()
    assert own_basis or not set(subjects) & set(fitted_to)
    start = time.time()
    reduction = sbci.reduce([SC / f"{s}_sc.h5" for s in fitted_to], rank=20)
    whose = "the split subjects' own" if own_basis else f"{len(fitted_to)} other subjects'"
    print(
        f"rank 20 on {whose} whole connectomes: {reduction.explained[-1]:.1%} of their norm "
        f"({time.time() - start:.0f}s)",
        flush=True,
    )
    scores = {}
    for name in "AB":
        paths = [OUT / s / f"{s}_split-{name}_sc.h5" for s in subjects]
        scores[name] = np.vstack(
            [sbci.project(reduction, sbci.load(path).dense(np.float64)) for path in paths]
        )
    np.savez_compressed(
        OUT / ("components_own-basis.npz" if own_basis else "components.npz"),
        first=scores["A"],
        second=scores["B"],
        explained=reduction.explained,
    )
    print(f"both halves scored ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(int(sys.argv[2]))
    elif sys.argv[1] == "components":
        components(own_basis="--own-basis" in sys.argv[2:])
    else:
        raise SystemExit("split SUBJECT_INDEX, or components [--own-basis]")
