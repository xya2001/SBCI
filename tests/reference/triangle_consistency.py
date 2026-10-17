"""Do stored triangle indices resolve to where the streamline actually crossed?

Needs no reference matrix: a legacy `mesh_intersections_*.mat` stores both the
nearest vertex of each crossing and the triangle plus barycentric weights. The
position rebuilt from the triangle -- through the package's bundled face list --
must lie within one edge length of that vertex, about 4 degrees on ico4. If the
face list is in the wrong order it lands about 85 degrees away.

This is the check that established the bundled face order is the pipeline's
canonical one (SPEC_QUESTIONS.md item 13): six subjects from two sources all
resolve at a median of about 1.6 degrees. Run by hand against lab data.

    python tests/reference/triangle_consistency.py <mesh_intersections_ico4.mat> ...
"""

from __future__ import annotations

import sys

import numpy as np

from sbci.smoothing import Endpoints, endpoint_positions
from sbci.surface import load_surface


def check(path: str) -> tuple[float, float, int]:
    """Median and 99th-percentile angle between rebuilt position and stored vertex."""
    sphere = load_surface("sphere")
    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    endpoints = Endpoints.from_matlab(path)
    if not endpoints.has_positions:
        raise SystemExit(f"{path} stores vertices only; nothing to check")
    rebuilt, _ = endpoint_positions(endpoints, surface=sphere)
    nearest = vertices[endpoints.global_vertex_in]
    angle = np.degrees(np.arccos(np.clip((rebuilt * nearest).sum(axis=1), -1.0, 1.0)))
    return float(np.median(angle)), float(np.percentile(angle, 99)), endpoints.n_streamlines


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    worst = 0.0
    for path in paths:
        median, p99, count = check(path)
        worst = max(worst, median)
        verdict = "OK   " if median < 4.0 else "WRONG"
        print(f"{verdict} median {median:6.2f}  99th {p99:6.2f} deg  n={count:>9,}  {path}")
    return 0 if worst < 4.0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
