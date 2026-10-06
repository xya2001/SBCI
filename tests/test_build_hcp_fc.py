"""tools/build_hcp_fc.py: the pieces that place the HCP's time series on the grid.

The tool itself runs on HCP data (PORTING.md item 2 records what it gave); these check
the geometry and the arithmetic it rests on, on the bundled sphere and on made-up series.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
build = pytest.importorskip("build_hcp_fc")

from sbci.surface import load_surface  # noqa: E402

N = 5124
HALF = N // 2


@pytest.fixture(scope="module")
def left_sphere():
    """The bundled ico4 sphere's left hemisphere, on the unit sphere."""
    sphere = load_surface("sphere")
    vertices = np.asarray(sphere.vertices, dtype=np.float64)[:HALF]
    faces = np.asarray(sphere.faces)
    faces = faces[np.all(faces < HALF, axis=1)]
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True), faces


def test_every_point_is_found_in_its_triangle(left_sphere):
    vertices, faces = left_sphere
    rng = np.random.default_rng(0)
    points = rng.standard_normal((5000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    tree = cKDTree(vertices[faces].mean(axis=1))
    tri, bary = build.locate(points, vertices, faces, tree)
    assert (bary >= -1e-9).all()
    np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-12)
    back = np.einsum("ij,ijk->ik", bary, vertices[faces[tri]])
    back /= np.linalg.norm(back, axis=1, keepdims=True)
    np.testing.assert_allclose(back, points, atol=1e-10)


def test_a_point_its_nearest_triangles_miss_is_found_among_all_of_them(left_sphere):
    # a tree over the wrong centroids, as a stretched triangle's distant centroid is to
    # the points it holds: the candidates miss, and the search over every triangle finds it
    vertices, faces = left_sphere
    rng = np.random.default_rng(3)
    points = rng.standard_normal((200, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    wrong = cKDTree(-vertices[faces].mean(axis=1))
    tri, bary = build.locate(points, vertices, faces, wrong)
    assert (bary >= -1e-9).all()
    back = np.einsum("ij,ijk->ik", bary, vertices[faces[tri]])
    back /= np.linalg.norm(back, axis=1, keepdims=True)
    np.testing.assert_allclose(back, points, atol=1e-10)
    right, _ = build.locate(points, vertices, faces, cKDTree(vertices[faces].mean(axis=1)))
    np.testing.assert_array_equal(tri, right)


def test_a_vertex_of_the_mesh_is_found_with_all_its_weight(left_sphere):
    vertices, faces = left_sphere
    tree = cKDTree(vertices[faces].mean(axis=1))
    tri, bary = build.locate(vertices[:50], vertices, faces, tree)
    for k in range(50):
        corner = np.flatnonzero(faces[tri[k]] == k)
        assert corner.size == 1
        assert bary[k, corner[0]] == pytest.approx(1.0, abs=1e-9)


def test_cells_average_their_members_and_empty_cells_take_the_nearest():
    # five 32k vertices on the left, two on the right; cells 0 and 1 on the left and 2562
    # on the right have members, cell 2 (cortex) has none, and the rest is medial wall
    cells_lh = np.array([0, 0, 1, 1, 1])
    cells_rh = np.array([HALF, HALF])
    rng = np.random.default_rng(1)
    grid = rng.standard_normal((N, 3))
    grid /= np.linalg.norm(grid, axis=1, keepdims=True)
    where_lh = grid[[0, 0, 1, 1, 1]] + 1e-3 * rng.standard_normal((5, 3))
    where_lh[4] = grid[2] + 1e-6  # the vertex nearest to the empty cell's grid vertex
    where_rh = grid[[HALF, HALF]]
    cells = {"lh": (cells_lh, where_lh), "rh": (cells_rh, where_rh)}
    ids = {"lh": np.arange(5), "rh": np.arange(2)}
    mask = np.zeros(N, dtype=bool)
    mask[[0, 1, 2, HALF]] = True
    averaging, on_wall, empty = build.averaging_matrix(cells, ids, mask, grid)
    dense = averaging.toarray()
    np.testing.assert_allclose(dense[0, :5], [0.5, 0.5, 0, 0, 0])
    np.testing.assert_allclose(dense[1, :5], [0, 0, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(dense[2, :5], [0, 0, 0, 0, 1.0])  # filled from vertex 4
    np.testing.assert_allclose(dense[HALF, 5:], [0.5, 0.5])
    assert empty == 1
    assert on_wall == 0.0


def test_regression_removes_the_global_signal_only_when_asked():
    rng = np.random.default_rng(2)
    t = 600
    shared = rng.standard_normal(t)
    data = shared[:, None] * rng.uniform(0.5, 1.5, 40) + np.linspace(0, 5, t)[:, None]
    data += 0.3 * rng.standard_normal((t, 40))
    kept = build.regress(data, gsr=False)
    removed = build.regress(data, gsr=True)
    assert np.abs(kept.mean(axis=0)).max() < 1e-10  # the constant goes either way
    assert abs(np.corrcoef(kept[:, 0], np.arange(t))[0, 1]) < 1e-8  # and the trend
    assert abs(np.corrcoef(kept[:, 0], shared)[0, 1]) > 0.8
    # with the global signal regressed out, every column is orthogonal to it, so the
    # columns no longer share it: their mean is zero
    signal = data.mean(axis=1)
    assert max(abs(np.corrcoef(removed[:, j], signal)[0, 1]) for j in range(40)) < 1e-8
    assert np.abs(removed.mean(axis=1)).max() < 1e-10


def test_runs_can_be_chosen_for_one_day(tmp_path):
    """--runs REST1_LR REST1_RL builds the first day's FC alone."""
    folder = tmp_path / "HCP_fMRI100307"
    folder.mkdir()
    for run in build.RUNS:
        (folder / f"rfMRI_{run}_Atlas_hp2000_clean.dtseries.nii").write_bytes(b"x")
    assert sorted(build.run_paths("sub-100307", [str(tmp_path)])) == sorted(build.RUNS)
    first = build.run_paths("sub-100307", [str(tmp_path)], ("REST1_LR", "REST1_RL"))
    assert sorted(first) == ["REST1_LR", "REST1_RL"]
