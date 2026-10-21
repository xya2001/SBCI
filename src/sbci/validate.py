"""The checks behind ``sbci validate <file>``.

Every one of the 946 young adults passed them when the cohort was built
(PORTING.md item 6), and every file of theirs passes the twelve there were from
12 October 2026 (item 27); the thirteenth, the endpoints' positions against
their vertices, holds on the eleven released subjects (item 38). CI runs them
on synthetic files. Each check returns a human-readable line; the command
exits non-zero if any check fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import grid, spec
from .connectome import COMPUTATIONAL_SUFFIXES, is_computational_name
from .errors import SbciError
from .io import read_hdf5
from .metadata import MetadataError
from .parcellation import CORRELATION_ROUNDING


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

    Checks that the file is readable, named as :meth:`sbci.ContinuousConnectome.load`
    reads it, its metadata complete and canonical, the grid the ico4 grid, the
    area, mask and coordinates the size of that grid, the stored form the
    strict upper triangle in float32, every value finite, SC nonnegative, FC
    within ``[-1, 1]`` to rounding (:data:`sbci.parcellation.CORRELATION_ROUNDING`),
    the area weights positive and in the pipeline's units, the area-weighted
    mass one, the medial wall empty with some cortex beside it, and each
    stored endpoint's position, rebuilt from its triangle, within one edge of
    its stored vertex. Tolerances are relative to the file's own scale where a
    scale exists. Everything ``load`` refuses fails some check here; the
    converse does not hold, since ``load`` checks few of the values -- the
    mask's, the endpoints' -- and none of the connectivity's.
    """
    path = Path(path)
    checks: list[Check] = []
    if not Path(path).exists():
        return [Check("readable", False, f"no such file: {path}")]

    try:
        parts = read_hdf5(path)
        import h5py

        with h5py.File(path, "r") as handle:
            stored_dtype = handle["connectivity"].dtype
    except (OSError, KeyError, SbciError, ValueError, TypeError) as exc:
        return [Check("readable", False, str(exc))]
    checks.append(Check("readable", True))

    # load() decides what kind of file it has from the name, before reading.
    named = is_computational_name(path)
    detail = "" if named else f"{path.name!r} would not load; expected {COMPUTATIONAL_SUFFIXES}"
    checks.append(Check("name", named, detail))

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

    # The arrays have to agree on the vertex count, as load() requires of them;
    # the coordinates are optional but, when present, one triple per vertex.
    coords = parts["coords"]
    shapes_agree = n is not None and data.ndim == 1 and area.shape == (n,) and mask.shape == (n,)
    if n is None:
        checks.append(Check("shapes", False, "no vertex count to compare the arrays against"))
    elif area.size != n or mask.size != n:
        checks.append(
            Check(
                "shapes",
                False,
                f"connectivity implies {n} vertices but area has {area.size} and mask has "
                f"{mask.size}",
            )
        )
    elif not shapes_agree:
        checks.append(
            Check(
                "shapes",
                False,
                f"connectivity is stored {data.shape}, area {area.shape} and mask "
                f"{mask.shape}, where each is flat",
            )
        )
    elif coords is not None and np.shape(coords) != (n, 3):
        checks.append(
            Check("shapes", False, f"coordinates are {np.shape(coords)}, expected {(n, 3)}")
        )
    else:
        checks.append(Check("shapes", True))

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

    # FC holds correlations, which rounding may carry a little past 1 and no further:
    # a file of Fisher z values, or of correlations rescaled, passed every check, and
    # to_atlas's Fisher z average clipped it without a word.
    if metadata.get("included_connections") == "fc":
        outside = int(np.count_nonzero(np.abs(data) > 1.0 + CORRELATION_ROUNDING))
        detail = f"{outside} entries outside [-1, 1]"
        if outside:
            detail += f", up to {float(np.nanmax(np.abs(data))):.7g} in magnitude"
        checks.append(Check("correlation", outside == 0, detail))
    else:
        checks.append(Check("correlation", True, "skipped: only FC holds correlations"))

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
    if shapes_agree:
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
        checks.append(Check("unit mass", False, "the arrays' shapes do not match the grid"))

    # Mask: the medial wall must carry no connectivity. The leak is judged
    # against the file's own total, since a unit-mass density on the ico4 grid
    # has entries of order 1e-11 and an absolute tolerance would pass anything.
    # A mask that keeps nothing leaves nothing to leak: an FC file of zeros with no
    # cortex passed every check.
    if dense is not None:
        if mask.all():
            checks.append(Check("mask", False, "no vertices are masked out"))
        elif not mask.any():
            checks.append(Check("mask", False, "no vertex is cortex: the mask keeps nothing"))
        else:
            leaked = float(np.abs(dense[~mask]).sum())
            total_abs = float(np.abs(dense).sum())
            ok = leaked <= tolerance * total_abs
            share = leaked / total_abs if total_abs > 0 else 0.0
            checks.append(Check("mask", ok, f"medial wall carries {share:.3g} of the total"))
    else:
        checks.append(Check("mask", False, "the arrays' shapes do not match the grid"))

    # Endpoints: the position rebuilt from an endpoint's triangle lies within one edge of
    # its stored vertex when the triangles index the grid's own face list; against a
    # face list in another order it lands about 85 degrees away (SPEC_QUESTIONS.md item 13).
    endpoints = parts["endpoints"]
    if endpoints is None or not endpoints.has_positions:
        checks.append(Check("endpoints", True, "skipped: no endpoint positions"))
    elif n != spec.N_VERTICES:
        checks.append(
            Check("endpoints", True, "skipped: not on the ico4 grid, whose faces they index")
        )
    else:
        from .smoothing import _vertex_offsets

        try:
            _, _, offsets, edge = _vertex_offsets(endpoints)
        except ValueError as exc:
            checks.append(Check("endpoints", False, str(exc)))
        else:
            far = int(np.count_nonzero(offsets > edge))
            detail = (
                f"{far:,} of {offsets.size:,} endpoints rebuild farther than one edge "
                f"({edge:.2f} degrees) from their stored vertex, up to {offsets.max():.1f}"
                if far
                else f"every endpoint rebuilds within one edge ({edge:.2f} degrees) of its "
                f"stored vertex, at most {offsets.max():.2f}"
            )
            checks.append(Check("endpoints", far == 0, detail))

    return checks
