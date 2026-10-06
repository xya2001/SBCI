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


def test_a_reloaded_warp_migrates_with_its_own_rotations(tmp_path):
    """A warp saved and loaded carries ENCORE's grid rotations, so migrating it needs no argument.

    Before the second review's finding C the rotations lived only in
    ``Alignment.grid_rotations``: a warp read back from its file was taken as
    unrotated, and the identity warp of the rotated grid landed 15 degrees
    from where it belongs on fs_LR.
    """
    from sbci.alignment import pole_rotation

    sphere = sbci.load_surface("sphere")
    vertices = normalize_rows(np.asarray(sphere.vertices, dtype=np.float64))
    lh_rotation, rh_rotation = pole_rotation(vertices[:HALF]), pole_rotation(vertices[HALF:])
    assert not np.allclose(lh_rotation, np.eye(3))
    # the identity warp of the rotated grids, as align() returns it
    warp = Warp(
        lh_vertices=vertices[:HALF] @ lh_rotation.T,
        lh_jacobian=np.ones(HALF),
        rh_vertices=vertices[HALF:] @ rh_rotation.T,
        rh_jacobian=np.ones(HALF),
        lh_rotation=lh_rotation,
        rh_rotation=rh_rotation,
    )
    reloaded = Warp.load(warp.save(tmp_path / "sub-rotated_warp.npz"))
    np.testing.assert_array_equal(reloaded.lh_rotation, lh_rotation)

    own = migrate_warp(reloaded, to="fs_LR_32k")
    explicit = migrate_warp(reloaded, to="fs_LR_32k", grid_rotations=(lh_rotation, rh_rotation))
    np.testing.assert_array_equal(own.lh_vertices, explicit.lh_vertices)
    np.testing.assert_array_equal(own.rh_vertices, explicit.rh_vertices)
    (left_vertices, _), _ = template_mesh("fs_LR_32k")
    assert _degrees(own.lh_vertices, left_vertices).max() < 0.3  # an identity lands on the template
    # an explicit argument still overrides: read as unrotated, the same file is a 15-degree rotation
    unrotated = migrate_warp(reloaded, to="fs_LR_32k", grid_rotations=(np.eye(3), np.eye(3)))
    assert np.median(_degrees(unrotated.lh_vertices, left_vertices)) > 5.0


def test_apply_refuses_a_hemisphere_other_than_l_or_r():
    points = np.eye(3)
    warp = TemplateWarp("fs_LR_32k", points, np.ones(3), points, np.ones(3), faces=(None, None))
    for hemisphere in ("left", "l", "X", ""):
        with pytest.raises(ValueError, match="'L' or 'R'"):
            warp.apply(points, hemisphere)


def test_save_returns_the_file_numpy_wrote(tmp_path):
    """``np.savez_compressed`` adds ``.npz`` to a name without it; the returned path must exist."""
    (left, left_faces), (right, right_faces) = template_mesh("fs_LR_32k")
    warp = TemplateWarp(
        "fs_LR_32k",
        left,
        np.ones(left.shape[0]),
        right,
        np.ones(right.shape[0]),
        faces=(left_faces, right_faces),
    )
    path = warp.save(tmp_path / "sub-example_warp")
    assert path == tmp_path / "sub-example_warp.npz" and path.exists()
    assert TemplateWarp.load(path).template == "fs_LR_32k"
    assert warp.save(tmp_path / "sub-example_warp.npz") == tmp_path / "sub-example_warp.npz"


def test_maps_need_one_vertex_set():
    with pytest.raises(ValueError, match="same"):
        SphereMap(np.eye(3), np.eye(4)[:, :3][:2], np.array([[0, 1, 2]]))


# --- reading spheres from GIFTI ----------------------------------------------


def _tetrahedron():
    """The smallest closed triangle mesh: four unit vertices, four faces."""
    vertices = normalize_rows(
        np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return vertices, faces


def _rotated(vertices, degrees):
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(np.radians(degrees) * np.array([0.0, 0.0, 1.0])).as_matrix()
    return vertices @ rotation.T


def _write_sphere(path, vertices, faces, triangles_first=False, intents=True):
    """A GIFTI sphere at radius 100, its two arrays in either order, with or without intents."""
    nib = pytest.importorskip("nibabel")
    from nibabel import gifti

    coordinates = gifti.GiftiDataArray(
        (vertices * 100.0).astype(np.float32),
        intent="NIFTI_INTENT_POINTSET" if intents else 0,
        datatype="NIFTI_TYPE_FLOAT32",
    )
    triangles = gifti.GiftiDataArray(
        faces.astype(np.int32),
        intent="NIFTI_INTENT_TRIANGLE" if intents else 0,
        datatype="NIFTI_TYPE_INT32",
    )
    arrays = [triangles, coordinates] if triangles_first else [coordinates, triangles]
    nib.save(gifti.GiftiImage(darrays=arrays), str(path))
    return path


@pytest.mark.parametrize("triangles_first", [False, True])
def test_from_gifti_finds_the_arrays_by_intent_not_position(tmp_path, triangles_first):
    """A GIFTI may store its triangles first; a source vertex still lands on its rotated image."""
    vertices, faces = _tetrahedron()
    rotated = _rotated(vertices, 25.0)
    source = _write_sphere(tmp_path / "source.surf.gii", vertices, faces, triangles_first)
    # the target in the other order: the two files need not even agree
    target = _write_sphere(tmp_path / "target.surf.gii", rotated, faces, not triangles_first)

    sphere_map = SphereMap.from_gifti(source, target)
    np.testing.assert_array_equal(sphere_map.faces, faces)
    np.testing.assert_allclose(sphere_map.forward(vertices), rotated, atol=1e-6)
    np.testing.assert_allclose(sphere_map.inverse(rotated), vertices, atol=1e-6)


def test_from_gifti_assumes_the_conventional_order_only_without_intents(tmp_path):
    vertices, faces = _tetrahedron()
    rotated = _rotated(vertices, 40.0)
    source = _write_sphere(tmp_path / "source.surf.gii", vertices, faces, intents=False)
    target = _write_sphere(tmp_path / "target.surf.gii", rotated, faces, intents=False)
    sphere_map = SphereMap.from_gifti(source, target)
    np.testing.assert_allclose(sphere_map.forward(vertices), rotated, atol=1e-6)
    # triangles first and no intents: the dtypes give it away, and it is refused rather than guessed
    reversed_ = _write_sphere(tmp_path / "rev.surf.gii", vertices, faces, True, intents=False)
    with pytest.raises(ValueError, match="one coordinate array and one triangle array"):
        SphereMap.from_gifti(reversed_, target)


def test_from_gifti_refuses_a_file_without_one_of_each_array(tmp_path):
    nib = pytest.importorskip("nibabel")
    from nibabel import gifti

    vertices, faces = _tetrahedron()
    coordinates = gifti.GiftiDataArray(
        (vertices * 100.0).astype(np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(gifti.GiftiImage(darrays=[coordinates, coordinates]), str(tmp_path / "twice.surf.gii"))
    good = _write_sphere(tmp_path / "good.surf.gii", vertices, faces)
    with pytest.raises(ValueError, match="found 2 and 0 by intent"):
        SphereMap.from_gifti(tmp_path / "twice.surf.gii", good)


def test_from_gifti_refuses_two_different_meshes(tmp_path):
    vertices, faces = _tetrahedron()
    source = _write_sphere(tmp_path / "source.surf.gii", vertices, faces)
    target = _write_sphere(
        tmp_path / "target.surf.gii", vertices, faces[::-1], triangles_first=True
    )
    with pytest.raises(ValueError, match="do not share a face list"):
        SphereMap.from_gifti(source, target)
