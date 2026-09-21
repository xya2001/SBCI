"""ENCORE alignment: the properties that hold without a MATLAB reference.

The numerical comparison against a real MATLAB run lives in
``test_matlab_reference.py`` and skips without it. What is asserted here is
what the mathematics guarantees on any mesh: areas that sum to the surface, a
basis orthonormal in the area inner product, an identity warp with unit
Jacobian, a template that is a unit vector, and a registration that does not
increase its own cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.alignment import (
    Alignment,
    Concon,
    Encore,
    MeshQuery,
    SphericalGrid,
    SphericalWarp,
    Warp,
    align,
    cart_to_sphere,
    gradient_operators,
    rotate_off_poles,
    sphere_exp_map,
    sphere_log_map,
    tangent_basis,
    voronoi_areas,
)


def icosphere(subdivisions=1):
    """A subdivided icosahedron, as the reference's ``icosphere``."""
    t = (1 + np.sqrt(5)) / 2
    vertices = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    vertices = [np.array(v, dtype=float) for v in vertices]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    for _ in range(subdivisions):
        midpoints = {}
        refined = []
        for corner_a, corner_b, corner_c in faces:
            middles = []
            for i, j in ((corner_a, corner_b), (corner_b, corner_c), (corner_c, corner_a)):
                key = (min(i, j), max(i, j))
                if key not in midpoints:
                    midpoints[key] = len(vertices)
                    vertices.append((vertices[i] + vertices[j]) / 2)
                middles.append(midpoints[key])
            ab, bc, ca = middles
            refined += [[corner_a, ab, ca], [corner_b, bc, ab], [corner_c, ca, bc], [ab, bc, ca]]
        faces = refined

    points = np.array(vertices)
    return points / np.linalg.norm(points, axis=1, keepdims=True), np.array(faces)


@pytest.fixture(scope="module")
def sphere():
    """A coarse sphere, rotated so no vertex sits on the coordinate axis."""
    vertices, faces = icosphere(1)
    return rotate_off_poles(vertices), faces


@pytest.fixture(scope="module")
def grid(sphere):
    return SphericalGrid(sphere[0], sphere[1], order=2)


def test_voronoi_areas_cover_the_surface(sphere):
    """Every triangle's area is distributed, so the total is the mesh area."""
    vertices, faces = sphere
    corners = vertices[faces]
    mesh_area = (
        0.5
        * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        ).sum()
    )
    assert voronoi_areas(vertices, faces).sum() == pytest.approx(mesh_area, rel=1e-12)


def test_voronoi_areas_are_positive(sphere):
    assert voronoi_areas(*sphere).min() > 0


def test_exp_and_log_are_inverse(grid):
    gamma = 0.3 * (grid.e1 * np.sin(grid.theta)[:, None] + grid.e2 * np.cos(grid.phi)[:, None])
    moved = sphere_exp_map(grid.vertices, gamma)
    np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(sphere_log_map(grid.vertices, moved), gamma, atol=1e-10)


def test_the_basis_is_orthonormal_in_the_area_inner_product(grid):
    """Each field is normalized against the Voronoi areas, as the reference does."""
    norms = ((grid.basis[:, :, 0] ** 2 + grid.basis[:, :, 1] ** 2) * grid.areas[:, None]).sum(
        axis=0
    )
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_the_basis_has_the_size_the_order_implies(sphere):
    for order in (2, 3, 4):
        basis, laplacian = tangent_basis(order, *cart_to_sphere(sphere[0]), voronoi_areas(*sphere))
        assert basis.shape == (sphere[0].shape[0], 2 * (order + 1) ** 2 - 2, 2)
        assert laplacian.shape == basis.shape[:2]
        # The rotated half is divergence free, so it carries no Laplacian.
        assert np.abs(laplacian[:, basis.shape[1] // 2 :]).max() == 0.0


def test_a_query_at_a_vertex_returns_that_vertex(grid):
    weights, indices = MeshQuery(grid.vertices, grid.faces).query(grid.vertices)
    recovered = (weights[:, :, None] * grid.vertices[indices]).sum(axis=1)
    np.testing.assert_allclose(recovered, grid.vertices, atol=1e-12)


def test_barycentric_weights_sum_to_one(grid):
    rng = np.random.default_rng(0)
    points = grid.vertices + 0.02 * rng.standard_normal(grid.vertices.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    weights, _ = MeshQuery(grid.vertices, grid.faces).query(points)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)


def test_the_identity_warp_has_unit_jacobian(grid):
    warp = SphericalWarp(grid, delta=1e-5)
    np.testing.assert_array_equal(warp.vertices, grid.vertices)
    np.testing.assert_allclose(warp.jacobian, 1.0, atol=1e-12)


def test_composing_a_zero_displacement_moves_no_vertex(grid):
    warp = SphericalWarp(grid, delta=1e-5)
    composed = warp.compose(np.zeros((grid.n_vertices, 2)))
    np.testing.assert_allclose(composed.vertices, grid.vertices, atol=1e-12)


def test_the_jacobian_of_the_identity_converges_to_one():
    """It is not 1 on a coarse mesh, and it should not be.

    The Jacobian is a finite difference of a function interpolated over flat
    triangles, so it carries the mesh's chord-versus-arc error. What has to be
    true is that refining the mesh drives it to 1. The reference behaves the
    same way: on its own demo mesh it reports a range of 0.824 to 1.070 for a
    small warp.
    """
    errors = []
    for level in (1, 2, 3):
        vertices, faces = icosphere(level)
        mesh = SphericalGrid(vertices, faces, order=2)
        warp = SphericalWarp(mesh, delta=1e-5)
        jacobian = warp.compose(np.zeros((mesh.n_vertices, 2))).jacobian
        away = np.setdiff1d(np.arange(mesh.n_vertices), mesh.pole_vertices())
        errors.append(np.abs(jacobian[away] - 1).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.05


def test_vertices_on_the_coordinate_axis_get_a_zero_jacobian():
    """A property of the reference's formulation, not of this port.

    The Jacobian closes with a factor of sin(theta), so a vertex on the axis
    collapses to zero and would lose its whole row and column. The grid the
    package ships puts four vertices exactly there, which is why
    :func:`rotate_off_poles` exists.
    """
    vertices, faces = icosphere(2)
    mesh = SphericalGrid(vertices, faces, order=2)
    poles = mesh.pole_vertices()
    assert poles.size == 2, "this icosphere should have both poles as vertices"

    warp = SphericalWarp(mesh, delta=1e-5)
    jacobian = warp.compose(np.zeros((mesh.n_vertices, 2))).jacobian
    np.testing.assert_allclose(jacobian[poles], 0.0, atol=1e-15)

    rotated = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    assert rotated.pole_vertices().size == 0
    rotated_jacobian = (
        SphericalWarp(rotated, delta=1e-5).compose(np.zeros((rotated.n_vertices, 2))).jacobian
    )
    assert rotated_jacobian.min() > 0.5


def test_the_bundled_grid_is_rotated_clear_of_the_poles():
    """`align` must not silently drop four vertices of the ico4 grid."""
    from sbci.alignment import _hemisphere_grids

    for grid in _hemisphere_grids(order=2):
        assert grid.pole_vertices().size == 0


def test_align_refuses_a_grid_with_poles():
    vertices, faces = icosphere(2)
    mesh = SphericalGrid(vertices, faces, order=2)
    n = 2 * mesh.n_vertices
    rng = np.random.default_rng(2)
    matrix = rng.random((n, n))
    matrix = matrix + matrix.T
    with pytest.raises(ValueError, match="coordinate axis"):
        align([matrix, matrix], grids=(mesh, mesh), order=2)


def test_a_composed_warp_stays_on_the_sphere(grid):
    displacement = 0.02 * np.stack([np.sin(3 * grid.theta), np.cos(2 * grid.phi)], axis=1)
    composed = SphericalWarp(grid, delta=1e-5).compose(displacement)
    np.testing.assert_allclose(np.linalg.norm(composed.vertices, axis=1), 1.0, atol=1e-12)
    assert composed.jacobian.min() > 0


@pytest.fixture(scope="module")
def pair(grid):
    """Two hemispheres and a few synthetic connectomes on them."""
    rng = np.random.default_rng(1)
    n = 2 * grid.n_vertices
    densities = []
    for _ in range(3):
        matrix = rng.random((n, n))
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0.0)
        densities.append(matrix)
    return grid, densities


def test_the_template_is_a_unit_square_root_density(pair):
    grid, densities = pair
    encore = Encore(grid, grid, max_iterations=3, delta=1e-5)
    template = encore.template(densities, iterations=4)
    mass = (template**2 * encore.area_product).sum()
    assert mass == pytest.approx(1.0, rel=1e-9)
    assert template.min() >= 0


def test_registering_a_connectome_to_itself_costs_nothing(pair):
    grid, densities = pair
    encore = Encore(grid, grid, max_iterations=3, delta=1e-5)
    _, _, _, cost = encore.register(densities[0], densities[0])
    assert cost == pytest.approx(0.0, abs=1e-12)


def test_registration_does_not_increase_its_own_cost(pair):
    grid, densities = pair
    encore = Encore(grid, grid, step=0.05, max_iterations=5, delta=1e-5)
    before = (
        (encore.root(densities[0]) - encore.root(densities[1])) ** 2 * encore.area_product
    ).sum()
    _, _, _, after = encore.register(densities[0], densities[1])
    assert after <= before + 1e-12


def test_evaluate_through_the_identity_warp_returns_the_input(pair):
    grid, densities = pair
    concon = Concon(grid, grid, delta=1e-5)
    identity = SphericalWarp(grid, delta=1e-5)
    expected = densities[0].copy()
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(
        concon.evaluate(densities[0], identity, identity), expected, atol=1e-10
    )


def test_align_returns_one_warp_per_subject(pair):
    grid, densities = pair
    result = align(densities, grids=(grid, grid), order=2, max_iterations=3, delta=1e-5)
    assert isinstance(result, Alignment)
    assert len(result.warps) == len(densities)
    assert len(result.aligned) == len(densities)
    assert all(a.shape == densities[0].shape for a in result.aligned)
    assert all(np.isfinite(c) for c in result.costs)


def test_align_validates_its_arguments(pair):
    grid, densities = pair
    with pytest.raises(ValueError, match="method must be one of"):
        align(densities, method="affine")
    with pytest.raises(ValueError, match="at least two"):
        align(densities[:1])
    with pytest.raises(ValueError, match="different grids"):
        align([densities[0], densities[1][:-2, :-2]])
    with pytest.raises(ValueError, match="but the grid has"):
        align([densities[0][:6, :6], densities[1][:6, :6]], grids=(grid, grid))


def test_warps_round_trip_through_save_and_load(tmp_path, grid):
    """PORTING.md item 4 names this as part of being correct."""
    displacement = 0.01 * np.stack([np.sin(grid.theta), np.cos(grid.phi)], axis=1)
    composed = SphericalWarp(grid, delta=1e-5).compose(displacement)
    warp = Warp(composed.vertices, composed.jacobian, composed.vertices, composed.jacobian)

    path = warp.save(tmp_path / "sub-toy_warp.npz")
    back = Warp.load(path)
    np.testing.assert_array_equal(back.lh_vertices, warp.lh_vertices)
    np.testing.assert_array_equal(back.lh_jacobian, warp.lh_jacobian)
    np.testing.assert_array_equal(back.rh_vertices, warp.rh_vertices)
    np.testing.assert_array_equal(back.rh_jacobian, warp.rh_jacobian)


# --- the derivative operators, against derivatives we know ----------------


@pytest.fixture(scope="module")
def flat_patch():
    """A triangulated unit square in the z = 0 plane, with a constant frame."""
    n = 15
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    vertices = np.stack([u.ravel(), v.ravel(), np.zeros(u.size)], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, d], [a, d, c]]
    e1 = np.tile([1.0, 0.0, 0.0], (vertices.shape[0], 1))
    e2 = np.tile([0.0, 1.0, 0.0], (vertices.shape[0], 1))
    return vertices, np.array(faces), e1, e2


@pytest.mark.parametrize(
    ("label", "coefficients"),
    [
        ("f = x", (1.0, 0.0)),
        ("f = y", (0.0, 1.0)),
        ("f = 3x + 2y", (3.0, 2.0)),
        ("f = 0", (0.0, 0.0)),
    ],
)
def test_the_gradient_of_a_linear_field_is_exact(flat_patch, label, coefficients):
    """A linear field is represented exactly, so its gradient must be exact.

    ``f(x) = x`` has to come back as 1 everywhere, not 1 to a few digits.
    """
    vertices, faces, e1, e2 = flat_patch
    first, second = gradient_operators(vertices, faces, e1, e2)
    a, b = coefficients
    field = a * vertices[:, 0] + b * vertices[:, 1] + 5.0

    np.testing.assert_allclose(first @ field, a, atol=1e-12)
    np.testing.assert_allclose(second @ field, b, atol=1e-12)


def test_a_constant_field_has_no_gradient(flat_patch):
    vertices, faces, e1, e2 = flat_patch
    first, second = gradient_operators(vertices, faces, e1, e2)
    constant = np.full(vertices.shape[0], -3.25)
    np.testing.assert_allclose(first @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(second @ constant, 0.0, atol=1e-12)


def _sphere_fields(grid):
    """Coordinate functions with their closed-form surface gradients."""
    theta, phi = grid.theta, grid.phi
    return {
        "z": (grid.vertices[:, 2], -np.sin(theta), np.zeros_like(theta)),
        "x": (grid.vertices[:, 0], np.cos(theta) * np.cos(phi), -np.sin(phi)),
        "y": (grid.vertices[:, 1], np.cos(theta) * np.sin(phi), np.cos(phi)),
    }


@pytest.mark.parametrize("name", ["z", "x", "y"])
def test_the_analytic_gradient_converges_on_the_sphere(name):
    """Refining the mesh must drive the error to zero against the closed form."""
    errors = []
    for level in (2, 3, 4):
        vertices, faces = icosphere(level)
        grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
        first, second = grid.gradients
        field, truth_1, truth_2 = _sphere_fields(grid)[name]
        errors.append(
            max(np.abs(first @ field - truth_1).max(), np.abs(second @ field - truth_2).max())
        )
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] < 0.01


@pytest.mark.parametrize("name", ["z", "x", "y"])
def test_the_reference_difference_converges_on_the_sphere(name):
    """The reference's estimator has to converge too, or the port is wrong."""
    from sbci.alignment import Concon

    errors = []
    for level in (2, 3):
        vertices, faces = icosphere(level)
        grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
        field, truth_1, _ = _sphere_fields(grid)[name]
        n = grid.n_vertices

        concon = Concon(grid, grid, delta=1e-5, derivative="difference")
        block = np.tile(np.concatenate([field, field])[:, None], (1, 2 * n))
        errors.append(np.abs(concon.derivative(block)[0][:n, 0] - truth_1).max())
    assert errors[1] < errors[0]
    assert errors[-1] < 0.05


def test_the_two_derivative_paths_agree_on_a_smooth_field():
    """They estimate the same quantity; on a smooth field they must show it."""
    from sbci.alignment import Concon

    vertices, faces = icosphere(2)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    theta = np.concatenate([grid.theta, grid.theta])
    smooth = np.outer(1 + np.cos(theta), 1 + np.cos(theta))

    analytic = Concon(grid, grid, derivative="analytic").derivative(smooth)[0]
    difference = Concon(grid, grid, delta=1e-5, derivative="difference").derivative(smooth)[0]
    assert np.corrcoef(analytic.ravel(), difference.ravel())[0, 1] > 0.999


def test_an_unknown_derivative_method_is_refused():
    from sbci.alignment import Concon

    vertices, faces = icosphere(1)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    with pytest.raises(ValueError, match="analytic.*difference"):
        Concon(grid, grid, derivative="spectral")
