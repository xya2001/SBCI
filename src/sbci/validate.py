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

    Checks symmetry, nonnegativity, unit mass, mask, grid, and metadata.
    """
    path = Path(path)
    checks: list[Check] = []

    try:
        parts = read_hdf5(path)
    except (OSError, KeyError) as exc:
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
            convention == expected and data.dtype == np.float32,
            f"storage_convention={convention!r}, dtype={data.dtype}",
        )
    )

    # Nonnegativity applies to SC only; FC is signed by construction.
    if metadata.get("included_connections") == "sc":
        negative = int(np.count_nonzero(data < 0))
        checks.append(Check("nonnegativity", negative == 0, f"{negative} negative entries"))
    else:
        checks.append(Check("nonnegativity", True, "skipped: FC is signed"))

    # Unit mass: the area-weighted total density integrates to one. This is a
    # property of a density, so it applies to SC only -- FC is a field of
    # correlations whose area-weighted total is an arbitrary number near zero,
    # and rescaling it to one would destroy its units.
    if metadata.get("included_connections") != "sc":
        checks.append(Check("unit mass", True, "skipped: FC is not a density"))
    elif n is not None and area.size == n:
        dense = grid.to_dense(data.astype(np.float64), n)
        total = float(area @ dense @ area)
        ok = abs(total - 1.0) <= tolerance
        checks.append(Check("unit mass", ok, f"total mass {total:.6g}"))
    else:
        checks.append(Check("unit mass", False, "area weights do not match the grid"))

    # Mask: the medial wall must carry no connectivity.
    if n is not None and mask.size == n:
        if mask.all():
            checks.append(Check("mask", False, "no vertices are masked out"))
        else:
            dense = grid.to_dense(data.astype(np.float64), n)
            leaked = float(np.abs(dense[~mask]).sum())
            ok = leaked <= tolerance
            checks.append(Check("mask", ok, f"medial-wall mass {leaked:.6g}"))
    else:
        checks.append(Check("mask", False, "mask does not match the grid"))

    return checks
