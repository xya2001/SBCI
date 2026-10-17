"""ENCORE's ``(theta, phi)`` Jacobian beside the coordinate axis, as numbers.

    python tests/reference/encore_axis_probe.py

The docstrings of ``SphericalGrid.pole_vertices`` and ``SphericalWarp`` quote these, and
PORTING.md items 10, 23, 24 and 27. Three cases:

- a grid with vertices beside the axis -- the level-2 icosphere of ``tests/test_alignment.py``,
  and the bundled sphere's left hemisphere unturned, whose vertices 0 and 11 are on it -- turned
  1e-5 and 3e-6 rad off it: the identity's Jacobian there as the scheme forms it, and a
  0.5-degree rotation's, divided by that and as the package returns it;
- the bundled left hemisphere turned clear of the axis, as ``align`` uses it, and rotated so
  that the vertex nearest the axis lands 1e-4, 1e-5 or 0 rad from it: the rotation's Jacobian
  at that vertex's image, which is 1 wherever the scheme can tell;
- the bundled left hemisphere turned so that its poles sit just clear of ``AXIS_CLEARANCE``,
  and farther: rotations' Jacobians at the vertices nearest the axis, which ``pole_vertices``
  quotes at the clearance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbci.alignment import (  # noqa: E402
    AXIS_CLEARANCE,
    SphericalGrid,
    SphericalWarp,
    _hemisphere_grids,
    normalize_rows,
    sphere_log_map,
)
from sbci.spec import N_VERTICES_PER_HEMI  # noqa: E402
from sbci.surface import load_surface  # noqa: E402
from test_alignment import _rotation_about_y, icosphere  # noqa: E402


def as_a_step(grid, rotation):
    """The displacement that carries each vertex where ``rotation`` does, in the grid's frame."""
    gamma = sphere_log_map(grid.vertices, grid.vertices @ rotation.T)
    return np.stack([(gamma * grid.e1).sum(axis=1), (gamma * grid.e2).sum(axis=1)], axis=1)


def beside(vertices, faces, label):
    for offset in (1e-5, 3e-6):
        grid = SphericalGrid(vertices @ _rotation_about_y(offset).T, faces, order=2)
        near = np.flatnonzero(np.sin(grid.theta) < 2 * offset)
        warp = SphericalWarp(grid, delta=1e-5)
        identity = warp._raw_jacobian()[near]
        rotated = warp.compose(as_a_step(grid, _rotation_about_y(np.radians(0.5))))
        divided = rotated._raw_jacobian()[near] / identity
        print(
            f"  {label}, {offset:g} rad off: the identity's {np.round(identity, 3).tolist()}, a "
            f"0.5-degree rotation's divided by it {np.round(divided, 2).tolist()}, as returned "
            f"{np.round(rotated.jacobian[near], 3).tolist()}"
        )


sphere = load_surface("sphere")
n = N_VERTICES_PER_HEMI
left = normalize_rows(np.asarray(sphere.vertices, dtype=np.float64)[:n])
left_faces = np.asarray(sphere.faces)
left_faces = left_faces[(left_faces < n).all(axis=1)]

print("a grid beside the axis:")
beside(*icosphere(2), "level-2 icosphere")
beside(left, left_faces, "the bundled left hemisphere")

print("an image carried to the axis, on the bundled left hemisphere turned clear of it:")
grid = _hemisphere_grids(2)[0]
warp = SphericalWarp(grid)
k = int(np.argmin(np.sin(grid.theta)))
for distance in (1e-4, 1e-5, 0.0):
    # the rotation about the axis through the origin that carries vertex k to the goal
    goal = np.array([np.sin(distance), 0.0, np.cos(distance)])
    axis = np.cross(grid.vertices[k], goal)
    sine, cosine = np.linalg.norm(axis), grid.vertices[k] @ goal
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    cross /= sine
    rotation = np.eye(3) + sine * cross + (1 - cosine) * cross @ cross
    jacobian = warp.compose(as_a_step(grid, rotation)).jacobian
    print(
        f"  vertex {k} carried {distance:g} rad from the axis: the rotation's Jacobian there "
        f"{jacobian[k]:.4f}; elsewhere {np.delete(jacobian, k).min():.4f} to "
        f"{np.delete(jacobian, k).max():.4f}"
    )

print("rotations near the axis, on the bundled left hemisphere turned to the clearance and beyond:")


def about(axis, angle):
    k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


for offset in (1.0001 * AXIS_CLEARANCE, 2e-3, 5e-3, 1e-2):
    grid = SphericalGrid(left @ _rotation_about_y(offset).T, left_faces, order=2)
    near = np.flatnonzero(np.sin(grid.theta) < 1.5 * offset)
    warp = SphericalWarp(grid)
    worst = [
        float(
            np.abs(
                warp.compose(as_a_step(grid, about(axis, np.radians(angle)))).jacobian[near] - 1
            ).max()
        )
        for axis in ([0, 1, 0], [1, 0, 0], [1, 1, 1])
        for angle in (0.5, 2.0, 10.0)
    ]
    print(
        f"  poles {offset:g} rad off the axis ({grid.pole_vertices().size} counted as on it): "
        f"rotations' Jacobians there 1 +- {min(worst):.1e} to {max(worst):.1e}"
    )
