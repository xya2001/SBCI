"""Carrying a warp to another template's sphere: the maps invert, and a warp survives the trip."""

import numpy as np
import pytest

import sbci
from sbci.alignment import Warp, normalize_rows
from sbci.templates import (
    SphereMap,
    TemplateWarp,
    chain_to,
    fsaverage_to_fslr,
    grid_to_fsaverage,
    migrate_warp,
    template_mesh,
)

HALF = 2562


def _random_points(n: int, seed: int) -> np.ndarray:
    return normalize_rows(np.random.default_rng(seed).standard_normal((n, 3)))


def _degrees(p, q) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((p * q).sum(axis=1), -1.0, 1.0)))


def test_grid_to_fsaverage_inverts_and_is_exact_at_vertices():
    left, _ = grid_to_fsaverage()
    assert left.source.shape == (HALF, 3)
    np.testing.assert_allclose(_degrees(left.forward(left.source), left.target), 0.0, atol=1e-6)
    points = _random_points(2000, 0)
    back = left.inverse(left.forward(points))
    assert _degrees(points, back).max() < 0.2


def test_fsaverage_to_fslr_inverts():
    left, right = fsaverage_to_fslr()
    assert left.source.shape == (32492, 3) and right.faces.shape == (64980, 3)
    points = _random_points(2000, 1)
    for step in (left, right):
        assert _degrees(points, step.inverse(step.forward(points))).max() < 0.1


def test_the_whole_chain_to_fslr_inverts():
    left, _ = chain_to("fs_LR_32k")
    points = _random_points(2000, 2)
    assert _degrees(points, left.inverse(left.forward(points))).max() < 0.3


def test_an_identity_warp_migrates_to_the_identity():
    sphere = sbci.load_surface("sphere")
    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    warp = Warp(
        lh_vertices=normalize_rows(vertices[:HALF]),
        lh_jacobian=np.ones(HALF),
        rh_vertices=normalize_rows(vertices[HALF:]),
        rh_jacobian=np.ones(HALF),
    )
    moved = migrate_warp(warp, to="fs_LR_32k")
    (left_vertices, _), (right_vertices, _) = template_mesh("fs_LR_32k")
    assert _degrees(moved.lh_vertices, left_vertices).max() < 0.3
    assert _degrees(moved.rh_vertices, right_vertices).max() < 0.3
    assert abs(moved.lh_jacobian - 1).max() < 0.1 and abs(moved.rh_jacobian - 1).max() < 0.1


def test_a_rotation_migrates_to_its_conjugate():
    """A rotation of the grid's sphere must, on fs_LR, agree with rotating through the maps."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(np.radians(3.0) * np.array([0.0, 0.0, 1.0])).as_matrix()
    sphere = sbci.load_surface("sphere")
    vertices = normalize_rows(np.asarray(sphere.vertices, dtype=np.float64))
    warp = Warp(
        lh_vertices=vertices[:HALF] @ rotation.T,
        lh_jacobian=np.ones(HALF),
        rh_vertices=vertices[HALF:] @ rotation.T,
        rh_jacobian=np.ones(HALF),
    )
    moved = migrate_warp(warp, to="fs_LR_32k")
    chain, _ = chain_to("fs_LR_32k")
    (target, _), _ = template_mesh("fs_LR_32k")
    expected = chain.forward(chain.inverse(target) @ rotation.T)
    assert _degrees(moved.lh_vertices, expected).max() < 0.3
    assert np.median(_degrees(moved.lh_vertices, target)) > 1.0  # it did move


def test_migrated_warp_round_trips_and_saves(tmp_path):
    sphere = sbci.load_surface("sphere")
    vertices = normalize_rows(np.asarray(sphere.vertices, dtype=np.float64))
    warp = Warp(
        lh_vertices=vertices[:HALF],
        lh_jacobian=np.ones(HALF),
        rh_vertices=vertices[HALF:],
        rh_jacobian=np.ones(HALF),
    )
    moved = migrate_warp(warp, to="fs_LR_32k")
    again = TemplateWarp.load(moved.save(tmp_path / "warp.npz"))
    np.testing.assert_allclose(again.lh_vertices, moved.lh_vertices)
    assert again.template == "fs_LR_32k"
    points = _random_points(50, 3)
    assert moved.apply(points, "L").shape == (50, 3)
    nib = pytest.importorskip("nibabel")
    left, right = moved.to_gifti(tmp_path / "sub-example")
    image = nib.load(str(left))
    assert image.darrays[0].data.shape == (32492, 3)
    assert abs(np.linalg.norm(image.darrays[0].data, axis=1) - 100).max() < 1e-2


def test_maps_need_one_vertex_set():
    with pytest.raises(ValueError, match="same"):
        SphereMap(np.eye(3), np.eye(4)[:, :3][:2], np.array([[0, 1, 2]]))
