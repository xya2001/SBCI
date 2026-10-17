"""The reference's tangent basis and divergence, for what is compared with a MATLAB run.

The package builds both correctly, and since 11 October 2026 has no switch that builds them as
the reference does (PORTING.md item 21). What has to reproduce a MATLAB run puts the
reference's two errors back:

- its clamp divides the phi-component of every gradient field by ``max(sin(theta), 1e-5)``,
  which beside the coordinate axis scales it by ``sin(theta) / 1e-5`` and on the axis zeroes it,
  and so too the theta-component of the field's rotated partner, before the fields are
  normalized (PORTING.md item 10, finding 1);
- its Legendre recurrence makes the divergence of each zonal field ``2 / (1 + 1/(l(l+1)))``
  times too large (item 8, finding 2).

On the level-2 icosphere of ``tests/test_alignment.py``, which has a vertex on each pole, this
reproduces what the removed ``reference=True`` built to 9e-16, basis and divergence alike, at
orders 3, 6 and 15. With the poles turned off the axis by up to 9e-6 it agrees to 4e-9: within
1e-5 of the axis the port takes the field's limit where the reference evaluates the field
(``reference_geometry_probe.py``, PORTING.md item 23).
"""

from __future__ import annotations

import copy

import numpy as np


def as_the_reference_forms_them(grid):
    """``(basis, laplacian)`` of a :class:`~sbci.alignment.SphericalGrid` as the reference's."""
    basis = grid.basis.copy()
    fields = basis.shape[1] // 2
    # sin(theta) as the reference's Legendre functions carry it, through cos(theta): zero on
    # the axis, where np.sin(pi) is 1.2e-16
    cos = np.cos(grid.theta)
    scale = np.minimum(np.sqrt((1 - cos) * (1 + cos)) / 1e-5, 1.0)[:, None]
    basis[:, :fields, 1] *= scale
    basis[:, fields:, 0] *= scale
    norms = np.sqrt((basis[:, :fields] ** 2).sum(axis=2).T @ grid.areas)
    norms = np.concatenate([norms, norms])
    basis /= norms[None, :, None]
    laplacian = grid.laplacian / norms
    for degree in range(1, grid.order + 1):
        laplacian[:, degree**2 - 1] *= 2 / (1 + 1 / (degree * (degree + 1)))
    return basis, laplacian


def with_the_reference_geometry(grid):
    """A copy of ``grid`` that carries the reference's basis and divergence."""
    clone = copy.copy(grid)
    clone.basis, clone.laplacian = as_the_reference_forms_them(grid)
    return clone
