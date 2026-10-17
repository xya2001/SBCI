"""Two of ConSEAL's ways of ending a descent, as a checkout's ``register`` takes them.

    PYTHONPATH=src python tests/reference/conseal_recheck_probe.py

Run it from a checkout of 3ef5d21 as well (``git worktree add``, with ``PYTHONPATH`` at that
checkout's ``src``) to see what PORTING.md item 23 changed. On the level-2 icosphere of
``tests/test_conseal.py``:

- a subject registered onto itself, its endpoints on their triangles' corners, so that the
  starting cost is exactly zero: how many iterations the descent takes, at the default floor
  under the threshold and with none;
- a first step forced to raise the cost, and so undone: how far the endpoints the
  registration returns sit from where it found them.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbci.alignment import SphericalGrid  # noqa: E402
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, icosphere  # noqa: E402
from sbci.smoothing import Endpoints  # noqa: E402
from test_conseal import synthetic  # noqa: E402

warnings.simplefilter("ignore")
vertices, faces = icosphere(2)
grid = SphericalGrid(vertices, faces, order=3)
kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)

rng = np.random.default_rng(0)
m = 2000
corner = np.tile([1.0, 0.0, 0.0], (m, 1))
ends = Endpoints(
    surf_in=rng.integers(0, 2, m).astype(np.int8),
    surf_out=rng.integers(0, 2, m).astype(np.int8),
    vtx_in=np.zeros(m, int),
    vtx_out=np.zeros(m, int),
    n_per_hemi=grid.n_vertices,
    tri_in=rng.integers(0, faces.shape[0], m),
    tri_out=rng.integers(0, faces.shape[0], m),
    bary_in=corner,
    bary_out=corner.copy(),
)
itself = EndpointConnectome.from_endpoints(ends, grid, grid)
for floor in (1e-5, 0.0):
    engine = ConSEAL(grid, grid, max_iterations=40, relative_threshold=floor)
    _, _, costs, _ = engine.register(itself, itself, kernel, derivative)
    print(
        f"onto itself, relative_threshold={floor:g}: {len(costs) - 1} iterations of 40, "
        f"every cost {np.unique(costs).tolist()}"
    )

fixed = EndpointConnectome.from_points(grid, grid, *synthetic(np.random.default_rng(7)))
moving = EndpointConnectome.from_points(
    grid, grid, *synthetic(np.random.default_rng(8), jitter=0.15)
)
cost, calls = ConSEAL.cost, [0]


def raised_by_the_first_step(self, difference):
    calls[0] += 1  # call 1 is the starting cost, call 2 the first step's
    return cost(self, difference) + (1.0 if calls[0] == 2 else 0.0)


ConSEAL.cost = raised_by_the_first_step
_, _, costs, warped = ConSEAL(grid, grid, max_iterations=5).register(
    fixed, moving, kernel, derivative
)
ConSEAL.cost = cost
moved = np.concatenate(
    [
        np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))
        for a, b in zip(moving.positions(), warped.positions(), strict=True)
    ]
)
changed = int((warped.face_in != moving.face_in).sum() + (warped.face_out != moving.face_out).sum())
print(
    f"first step undone ({len(costs) - 1} kept): the endpoints {moved.mean():.3f} degrees from "
    f"where they started on average, {moved.max():.3f} at most; {changed} of "
    f"{2 * moving.n_streamlines} triangles changed"
)
