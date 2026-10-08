"""How ConSEAL holds a rigid rotation: as its own velocity field, as it did, or exactly.

    python tests/reference/rotation_probe.py

On the bundled ico4 left grid, about one tilted axis: how far ``rotate`` lands from the exact
rotation at angles up to a half turn, and how much of a stored rotation the 5% smoothing erodes
over a registration's steps, held both ways; then a synthetic subject turned by 150 and 60
degrees, handed the exact inverse rotation as a perfect rigid search would find it, and
registered for 30 steps, again both ways. "As a field" is the port before October 2026: the
reference's representation (``strict_upstream=True`` still gives it) with every other
correction in place. PORTING.md item 9 quotes the output.
"""

import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from sbci.conseal import (
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    StationaryWarp,
    default_grids,
)

warnings.simplefilter("ignore", RuntimeWarning)
lh, rh = default_grids(15)
axis = np.array([0.2, 0.9, 0.4]) / np.linalg.norm([0.2, 0.9, 0.4])


def turn(degrees):
    return Rotation.from_rotvec(np.radians(degrees) * axis).as_matrix()


def degrees_off(points, expected):
    return np.degrees(np.arccos(np.clip((points * expected).sum(1), -1, 1)))


def place(warp, rotation, as_field):
    """Set ``warp`` to ``rotation``: as the rotation's own velocity field, or held exactly."""
    if not as_field:
        return warp.rotate(rotation)
    omega = Rotation.from_matrix(rotation).as_rotvec()
    field = np.cross(np.broadcast_to(omega, warp.base.shape), warp.base)
    warp.velocity = np.stack([(field * warp.e1).sum(1), (field * warp.e2).sum(1)], axis=1)
    warp.vertices = warp.exponential()
    return warp


print("rotate(): degrees from the exact rotation, max (mean)")
for angle in (20, 45, 90, 120, 150, 180):
    expected = lh.vertices @ turn(angle).T
    row = [f"{angle:3d} deg"]
    for as_field, label in ((True, "as a field"), (False, "held exactly")):
        off = degrees_off(place(StationaryWarp(lh), turn(angle), as_field).vertices, expected)
        row.append(f"{label} {off.max():.3f} ({off.mean():.3f})")
    print("   " + "; ".join(row), flush=True)

print("a stored rotation after zero-size steps: best-fit rotation, and max error")
for angle in (20, 150):
    expected = lh.vertices @ turn(angle).T
    for as_field, label in ((True, "as a field"), (False, "held exactly")):
        warp = place(StationaryWarp(lh), turn(angle), as_field)
        for _ in range(100):
            warp.compose(np.zeros((lh.n_vertices, 2)))
        u, _, vt = np.linalg.svd(lh.vertices.T @ warp.vertices)
        fitted = np.degrees(Rotation.from_matrix((u @ vt).T).magnitude())
        off = degrees_off(warp.vertices, expected)
        print(
            f"   {angle:3d} deg {label}, 100 steps: {fitted:.2f} deg, max error {off.max():.2f}",
            flush=True,
        )

# --- a registration started from the exact rotation ---------------------------------

rng = np.random.default_rng(11)


def cloud(center, kappa, n):
    center = np.asarray(center, float) / np.linalg.norm(center)
    points = rng.normal(size=(n, 3)) + kappa * center
    return points / np.linalg.norm(points, axis=1, keepdims=True)


# six bundles inside each hemisphere and four between them, of different sizes and spreads
bundles = [(side, side) for side in (0, 1) for _ in range(6)] + [(0, 1), (1, 0), (0, 1), (1, 0)]
p_in, p_out, h_in, h_out = [], [], [], []
for a, b in bundles:
    n = int(rng.integers(1500, 4000))
    kappa = float(rng.uniform(6, 14))
    p_in.append(cloud(rng.normal(size=3), kappa, n))
    p_out.append(cloud(rng.normal(size=3), kappa, n))
    h_in.append(np.full(n, a))
    h_out.append(np.full(n, b))
p_in, p_out, h_in, h_out = map(np.concatenate, (p_in, p_out, h_in, h_out))
fixed = EndpointConnectome.from_points(lh, rh, p_in, p_out, h_in, h_out)
kernel, derivative = HeatKernelBuilder(lh, rh, 30).compute(0.005)


def summary(connectome):
    q_in, q_out = connectome.positions()
    error = np.r_[degrees_off(q_in, p_in), degrees_off(q_out, p_out)]
    return f"mean {error.mean():.3f}, max {error.max():.2f}"


print(f"registration from the exact rotation, {len(p_in):,} streamlines in {len(bundles)} bundles")
for angle in (150, 60):
    rotation = turn(angle)
    moving = EndpointConnectome.from_points(
        lh, rh, p_in @ rotation.T, p_out @ rotation.T, h_in, h_out
    )
    for as_field, label in ((True, "as a field"), (False, "held exactly")):
        start = {}

        def exact_rigid(
            self,
            q1,
            moving,
            kernel,
            verbose=False,
            as_field=as_field,
            rotation=rotation,
            start=start,
        ):
            """The rigid step a perfect search would take, held one way or the other."""
            warps = self.new_warps()
            for warp in warps:
                place(warp, rotation.T, as_field)
            moving.warp(*warps)
            start["moving"] = moving.copy()
            return warps

        # the public code's step, clamp and smoothing, the defaults when this was measured: the
        # erosion it measures is the smoothing's (PORTING.md item 19 changed the defaults)
        engine = ConSEAL(
            lh, rh, delta=0.05, max_iterations=30, threshold=1e-9, step_clamp=0.2, viscosity=0.05
        )
        engine._rigid = exact_rigid.__get__(engine)
        t = time.time()
        _, _, costs, warped = engine.register(fixed, moving, kernel, derivative, init_rotation=True)
        print(
            f"   {angle:3d} deg {label}: cost {costs[0]:.6f} -> {costs[-1]:.6f} in "
            f"{len(costs) - 1} steps ({time.time() - t:.0f}s); endpoints from the truth "
            f"{summary(start['moving'])} after the rigid step, {summary(warped)} after",
            flush=True,
        )
