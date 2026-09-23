"""The checks behind ``sbci validate <file>``.

Runs in CI on the tutorial subject, so every released file is known to satisfy
them. Each check returns a human-readable line; the command exits non-zero if
any check fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import grid, spec
from .errors import SbciError
from .io import read_hdf5
from .metadata import MetadataError


@dataclass
class Check:
    """One validation result."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f": {self.detail}" if self.detail else "")


def validate_file(path: str | Path, tolerance: float = 1e-5) -> list[Check]:
    """Check one computational file against the specification.

    Checks that the file is readable, its metadata complete and canonical, the
    grid the ico4 grid, the stored form the strict upper triangle in float32,
    every value finite, SC nonnegative, the area weights positive and in the
    pipeline's units, the area-weighted mass one, and the medial wall empty.
    Tolerances are relative to the file's own scale where a scale exists.
    """
    path = Path(path)
    checks: list[Check] = []

    try:
        parts = read_hdf5(path)
        import h5py

        with h5py.File(path, "r") as handle:
            stored_dtype = handle["connectivity"].dtype
    except (OSError, KeyError, SbciError, ValueError) as exc:
        return [Check("readable", False, str(exc))]
    checks.append(Check("readable", True))

    metadata = parts["metadata"]
    try:
        metadata.validate()
        checks.append(Check("metadata", True))
    except MetadataError as exc:
        checks.append(Check("metadata", False, str(exc)))

    data = parts["data"]
    area = parts["area"]
    mask = parts["mask"]

    # Grid: the condensed length has to correspond to the canonical grid.
    try:
        n = grid.n_from_condensed(data.size)
        ok = n == spec.N_VERTICES
        checks.append(Check("grid", ok, "" if ok else f"{n} vertices, expected {spec.N_VERTICES}"))
    except ValueError as exc:
        n = None
        checks.append(Check("grid", False, str(exc)))

    # Symmetry is structural: storing only the upper triangle guarantees it,
    # so the check is that the stored form is the one the spec names.
    convention = metadata.get("storage_convention", "")
    expected = f"{spec.TRIANGLE}-triangular-{spec.DTYPE}"
    checks.append(
        Check(
            "symmetry",
            convention == expected and stored_dtype == np.float32,
            f"storage_convention={convention!r}, stored dtype={stored_dtype}",
        )
    )

    # Every value has to be a number: NaN passes every inequality below.
    non_finite = int(np.count_nonzero(~np.isfinite(data)))
    checks.append(Check("finite", non_finite == 0, f"{non_finite} non-finite entries"))

    # Nonnegativity applies to SC only; FC is signed by construction.
    if metadata.get("included_connections") == "sc":
        negative = int(np.count_nonzero(data < 0))
        checks.append(Check("nonnegativity", negative == 0, f"{negative} negative entries"))
    else:
        checks.append(Check("nonnegativity", True, "skipped: FC is signed"))

    # Area weights: positive, finite, and -- on the canonical grid -- in the
    # pipeline's units, fsaverage vertices per ico4 vertex (SPEC_QUESTIONS item 3).
    if area.size and np.isfinite(area).all() and (area > 0).all():
        if n == spec.N_VERTICES:
            ok = abs(area.sum() / spec.AREA_TOTAL - 1.0) <= 0.01
            checks.append(
                Check("area", ok, f"sum {area.sum():.6g}, expected about {spec.AREA_TOTAL}")
            )
        else:
            checks.append(Check("area", True, "positive and finite"))
    else:
        checks.append(Check("area", False, "area weights must be positive and finite"))

    dense = None
    if n is not None and area.size == n and mask.size == n:
        dense = grid.to_dense(data.astype(np.float64), n)

    # Unit mass: the area-weighted total density integrates to one. This is a
    # property of a density, so it applies to SC only -- FC is a field of
    # correlations whose area-weighted total is an arbitrary number near zero,
    # and rescaling it to one would destroy its units.
    if metadata.get("included_connections") != "sc":
        checks.append(Check("unit mass", True, "skipped: FC is not a density"))
    elif dense is not None:
        total = float(area @ dense @ area)
        ok = abs(total - 1.0) <= tolerance
        checks.append(Check("unit mass", ok, f"total mass {total:.6g}"))
    else:
        checks.append(Check("unit mass", False, "area weights do not match the grid"))

    # Mask: the medial wall must carry no connectivity. The leak is judged
    # against the file's own total, since a unit-mass density on the ico4 grid
    # has entries of order 1e-11 and an absolute tolerance would pass anything.
    if dense is not None:
        if mask.all():
            checks.append(Check("mask", False, "no vertices are masked out"))
        else:
            leaked = float(np.abs(dense[~mask]).sum())
            total_abs = float(np.abs(dense).sum())
            ok = leaked <= tolerance * total_abs
            share = leaked / total_abs if total_abs > 0 else 0.0
            checks.append(Check("mask", ok, f"medial wall carries {share:.3g} of the total"))
    else:
        checks.append(Check("mask", False, "mask does not match the grid"))

    return checks
