"""Endpoints from the pipeline's snapped streamlines and a registered sphere."""

import numpy as np
import pytest

from sbci.conseal import default_grids
from sbci.errors import FormatError
from sbci.smoothing import Endpoints, endpoint_positions


@pytest.fixture(scope="module")
def grids():
    return default_grids()


def test_snapped_vertices_land_where_the_registered_sphere_puts_them(grids):
    lh, rh = grids
    rng = np.random.default_rng(0)
    # a "native" sphere per hemisphere: the grid's own vertices at radius 100, shuffled,
    # so every snapped vertex should land exactly on a known grid vertex
    order = [rng.permutation(g.n_vertices) for g in (lh, rh)]
    spheres = [100 * g.vertices[o] for g, o in zip((lh, rh), order, strict=True)]
    n = 500
    surf0, surf1 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    v0 = np.array([rng.integers(0, spheres[s].shape[0]) for s in surf0])
    v1 = np.array([rng.integers(0, spheres[s].shape[0]) for s in surf1])
    # streamlines touching another surface are dropped
    surf0[:7] = 4
    snapped = {
        "surf_ids0": surf0.astype(float),
        "surf_ids1": surf1.astype(float),
        "v_ids0": v0.astype(float),
        "v_ids1": v1.astype(float),
    }
    ends = Endpoints.from_snapped(snapped, *spheres, grids=grids)
    assert ends.n_streamlines == n - 7 and ends.has_positions
    np.testing.assert_array_equal(ends.surf_in, surf0[7:])
    positions_in, positions_out = endpoint_positions(ends)
    expected_in = np.stack([spheres[s][v] / 100 for s, v in zip(surf0[7:], v0[7:], strict=True)])
    angle = np.degrees(np.arccos(np.clip((positions_in * expected_in).sum(1), -1, 1)))
    assert angle.max() < 1e-5  # arccos near zero amplifies float rounding to about 1e-6 degrees
    grid_vertex = np.array(
        [o[v] for o, v in ((order[s], v) for s, v in zip(surf0[7:], v0[7:], strict=True))]
    )
    np.testing.assert_array_equal(ends.vtx_in, grid_vertex)
    assert positions_out.shape == (n - 7, 3)


def test_from_snapped_reads_a_file_and_refuses_what_it_cannot_place(grids, tmp_path):
    lh, rh = grids
    spheres = [g.vertices for g in (lh, rh)]
    path = tmp_path / "snapped_fibers.npz"
    np.savez(
        path,
        surf_ids0=np.array([0.0, 1.0]),
        surf_ids1=np.array([1.0, 0.0]),
        v_ids0=np.array([3.0, 5.0]),
        v_ids1=np.array([8.0, 13.0]),
    )
    assert Endpoints.from_snapped(path, *spheres, grids=grids).n_streamlines == 2
    np.savez(
        path,
        surf_ids0=np.array([0.0]),
        surf_ids1=np.array([0.0]),
        v_ids0=np.array([lh.n_vertices + 1.0]),
        v_ids1=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="outside the left sphere"):
        Endpoints.from_snapped(path, *spheres, grids=grids)
    # a negative id would wrap round to the far end of the sphere
    np.savez(
        path,
        surf_ids0=np.array([0.0, 1.0]),
        surf_ids1=np.array([1.0, 1.0]),
        v_ids0=np.array([3.0, -1.0]),
        v_ids1=np.array([8.0, 2.0]),
    )
    with pytest.raises(ValueError, match="vertex -1 is outside the right sphere"):
        Endpoints.from_snapped(path, *spheres, grids=grids)
    np.savez(path, surf_ids0=np.array([0.0]))
    with pytest.raises(FormatError, match="have no surf_ids1, v_ids0"):
        Endpoints.from_snapped(path, *spheres, grids=grids)
