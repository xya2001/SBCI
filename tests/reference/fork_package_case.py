"""The package's ConSEAL on one comparison case of PORTING.md item 19.

    module load python/3.12.4
    python tests/reference/fork_package_case.py CASE defaults|fork|compose

``defaults`` are the package's defaults when the comparison ran, the public code's: step
0.05, up to 100 iterations, threshold 1e-4, a 0.2 clamp and 5% smoothing; since 7 October
2026 they are written out here, because the defaults are now ``fork``'s. ``fork`` takes the
fork's settings, the paper's: step 0.1, no clamp, no smoothing, up to 1000 iterations,
threshold 1e-6. ``compose`` adds the fork's update to them: each step applied after the warp
so far at the warped vertices (``SphericalWarp.compose_warp``), with no velocity field and no
fold check; the gradient, kernel, grid and stopping rule stay the package's. The cases are
``fork_cases.py``'s; ``fork_evaluate.py`` scores the results.
"""

import copy
import sys
import time

import numpy as np

import sbci
from sbci.alignment import MeshQuery, normalize_rows, sphere_exp_map
from sbci.conseal import (
    DEFAULT_KERNEL_DEGREE,
    DEFAULT_SIGMA,
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    _transport,
    default_grids,
)

OUT = "/work/users/x/y/xya/conseal-fork-compare/cases"
SETTINGS = {
    "defaults": dict(
        delta=0.05, max_iterations=100, threshold=1e-4, step_clamp=0.2, viscosity=0.05
    ),
    "fork": dict(delta=0.1, max_iterations=1000, threshold=1e-6, step_clamp=np.inf, viscosity=0.0),
}
SETTINGS["compose"] = SETTINGS["fork"]


class ComposedWarp:
    """The fork's ``SphericalWarp.compose_warp``: each step applied after the warp so far.

    The step, given at the grid's vertices, is evaluated where each vertex now sits --
    barycentric on the unwarped grid, the corners' vectors transported to the point -- and the
    vertex moves along it on the sphere. Nothing is refused, as in the fork.
    """

    def __init__(self, grid):
        self.e1, self.e2 = grid.e1, grid.e2
        self.base = grid.vertices.copy()
        self.vertices = grid.vertices.copy()
        self.rejected = 0
        self._query = MeshQuery(self.base, grid.faces)

    def copy(self):
        """An independent copy, as the registration's rollback needs."""
        other = copy.copy(self)
        other.vertices = self.vertices.copy()
        return other

    def compose(self, displacement, strict_upstream=False):
        """Apply a step after the warp so far; never refused."""
        tangent = self.e1 * displacement[:, :1] + self.e2 * displacement[:, 1:2]
        weights, indices, _ = self._query.query_faces(self.vertices)
        n = self.vertices.shape[0]
        moved = _transport(
            tangent[indices].reshape(-1, 3),
            self.base[indices].reshape(-1, 3),
            np.repeat(self.vertices, 3, axis=0),
        ).reshape(n, 3, 3)
        self.vertices = normalize_rows(
            sphere_exp_map(self.vertices, (weights[:, :, None] * moved).sum(axis=1))
        )
        return True


case, name = sys.argv[1], sys.argv[2]
print(
    f"sbci from {sbci.__file__}; case {case}, settings {name}: {SETTINGS[name] or 'the defaults'}",
    flush=True,
)
x = np.load(f"{OUT}/{case}.npz")
lh, rh = default_grids()


def connectome(prefix):
    flags = [x[f"{prefix}_h{end}"].ravel().astype(np.int64) for end in ("in", "out")]
    return EndpointConnectome.from_points(lh, rh, x[f"{prefix}_in"], x[f"{prefix}_out"], *flags)


fixed, moving = connectome("fixed"), connectome("moving")
kernel, derivative = HeatKernelBuilder(lh, rh, DEFAULT_KERNEL_DEGREE).compute(DEFAULT_SIGMA)
engine = ConSEAL(lh, rh, **SETTINGS[name])
if name == "compose":
    engine.new_warps = lambda: (ComposedWarp(lh), ComposedWarp(rh))
start = time.time()
lh_warp, rh_warp, costs, warped = engine.register(fixed, moving, kernel, derivative, verbose=True)
elapsed = time.time() - start
print(
    f"{len(costs) - 1} iterations in {elapsed:.0f}s: cost {costs[0]:.6f} -> {costs[-1]:.6f}",
    flush=True,
)
moved_in, moved_out = warped.positions()
np.savez(
    f"{OUT}/{case}_package-{name}.npz",
    costs=costs,
    elapsed=elapsed,
    moved_in=moved_in,
    moved_out=moved_out,
    lh_vertices=lh_warp.vertices,
    rh_vertices=rh_warp.vertices,
    grid_faces=lh.faces,
    rejected=lh_warp.rejected + rh_warp.rejected,
)
print("PACKAGE CASE SAVED", flush=True)
