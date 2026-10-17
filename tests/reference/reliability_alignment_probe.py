"""Section 5 of notebooks/reliability.ipynb again, with ENCORE as it composes now.

    python tests/reference/reliability_alignment_probe.py INDEX   # 0..29, one task of an array
    python tests/reference/reliability_alignment_probe.py summary

``scripts/reliability_data.py split`` aligned each split subject's second half of streamlines
onto its first, and the next subject's first half onto it, on 7 October 2026, before PORTING.md
item 21 changed the order in which ENCORE composes a step. This aligns the same saved halves
again and keeps the displacements under ``reliability/recheck``; the summary prints what the
notebook prints from both, and how far one vertex moved between them (item 24 quotes it).

On 12 October the recheck's ``alignment.npz`` replaced each split folder's, which the notebook
reads; the 7 October ones are kept beside them as ``alignment-2026-10-07.npz``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import reliability_data as data  # noqa: E402

import sbci  # noqa: E402

RECHECK = data.ROOT / "reliability" / "recheck"
BEFORE = "alignment-2026-10-07.npz"  # the 7 October alignments, kept in each split folder


def half(subject, name):
    return sbci.load(data.OUT / subject / f"{subject}_split-{name}_sc.h5")


def align(index):
    subjects = data.sample()
    subject, following = subjects[index], subjects[(index + 1) % len(subjects)]
    first = half(subject, "A")
    within = sbci.align([half(subject, "B")], template=first)
    between = sbci.align([half(following, "A")], template=first)
    out = RECHECK / subject
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "alignment.npz",
        within=data.displacement(within.warps[0]),
        between=data.displacement(between.warps[0]),
    )
    print(f"{subject}: aligned within and between", flush=True)


def summary():
    subjects = data.sample()
    cortex = half(subjects[0], "A").mask
    for label, root, name in (
        ("7 October", data.OUT, BEFORE),
        ("now", RECHECK, "alignment.npz"),
    ):
        moved = [np.load(root / s / name) for s in subjects]
        within = np.array([np.median(item["within"][cortex]) for item in moved])
        between = np.array([np.median(item["between"][cortex]) for item in moved])
        print(
            f"{label}: within {np.median(within):.3f} degrees ({within.min():.3f} to "
            f"{within.max():.3f}), between {np.median(between):.3f} ({between.min():.3f} to "
            f"{between.max():.3f})"
        )
    gaps = [
        np.abs(np.load(data.OUT / s / BEFORE)[key] - np.load(RECHECK / s / "alignment.npz")[key])[
            cortex
        ].max()
        for s in subjects
        for key in ("within", "between")
    ]
    print(f"largest change of one vertex's displacement: {max(gaps):.3f} degrees")


if __name__ == "__main__":
    summary() if sys.argv[1] == "summary" else align(int(sys.argv[1]))
