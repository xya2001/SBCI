"""The data behind notebooks/03_cohort.ipynb: each young adult's vertex strength and region matrix.

    python scripts/cohort_data.py measures   # every SC file read once (a batch job)

The lab's paths on Longleaf. ``measures`` reads the SC file of each of the 946
young adults once, ``SLURM_CPUS_PER_TASK`` files at a time in processes of one
thread each, and records per subject what the notebook's tests at every vertex
and its hand-off to graph tools need:

- the connectivity strength of every vertex, its density summed over the other
  endpoint weighted by the vertex areas (``dense() @ area``), as
  ``tools/age_probe.py`` records it;
- the Desikan-Killiany matrix of connection mass (``to_atlas("Desikan",
  how="mass")``): each region pair's share of the connectome's unit mass, with
  each region's own mass on the diagonal;
- the streamline count the file records.

Every file has to carry the same vertex areas and cortex, which are kept once.
It takes a few minutes on eight cores. The files are named by subject, so
``cohort/measures.npz`` holds HCP ids: it stays under ``/work`` with the lab's
other copies, out of the repository, and the notebook prints no subject.
Nothing here reads the HCP's restricted tables.
"""

from __future__ import annotations

import os

# Processes of one thread each: a thread count set in the environment would be inherited by
# every process, and their threads would crowd the cores. Set before numpy is imported.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sbci

ROOT = Path("/work/users/x/y/xya/hcp-ya")
SC = ROOT / "full" / "data"
OUT = ROOT / "cohort"
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def one(path: Path) -> tuple:
    """One subject's vertex strength, Desikan-Killiany masses and streamline count.

    With the file's vertex areas and cortex, which every file has to share.
    """
    connectome = sbci.load(path)
    area = np.asarray(connectome.area, dtype=np.float64)
    strength = connectome.dense() @ area
    mass = connectome.to_atlas("Desikan", how="mass")
    count = float(connectome.metadata.get("streamline_count"))
    return strength, mass, count, area, np.asarray(connectome.mask, dtype=bool)


def measures() -> None:
    paths = sorted(SC.glob("sub-*_sc.h5"))
    start = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, paths))
    elapsed = time.time() - start
    strength, mass, count, areas, masks = (list(part) for part in zip(*results, strict=True))
    for name, values in (("vertex areas", areas), ("cortex", masks)):
        differing = sum(not np.array_equal(v, values[0]) for v in values)
        if differing:
            raise ValueError(f"{differing} files' {name} differ from the first file's")
    mass = np.stack(mass)
    total = mass.sum(axis=(1, 2))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "measures.npz",
        subjects=np.array([path.name.split("_")[0] for path in paths]),
        strength=np.vstack(strength),
        desikan=mass,
        regions=np.array(sbci.load_atlas("Desikan").names),
        streamlines=np.array(count),
        area=areas[0],
        cortex=masks[0],
    )
    print(
        f"{len(paths)} subjects in {elapsed:.0f}s on {WORKERS} processes: the strength of "
        f"{areas[0].size:,} vertices and a {mass.shape[1]} x {mass.shape[2]} Desikan-Killiany "
        f"matrix each, summing to {total.min():.6f} to {total.max():.6f}; every file has the same "
        f"vertex areas and cortex ({int(masks[0].sum()):,} vertices); {min(count):,.0f} to "
        f"{max(count):,.0f} streamlines"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["measures"]:
        raise SystemExit("usage: python scripts/cohort_data.py measures")
    measures()
