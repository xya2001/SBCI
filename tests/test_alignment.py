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
    AXIS_CLEARANCE,
    Alignment,
    Concon,
    Encore,
    MeshQuery,
    SphericalGrid,
    SphericalWarp,
    Warp,
    _harmonic_derivatives,
    _legendre_derivatives,
    align,
    cart_to_sphere,
    gradient_operators,
    pole_rotation,
    rotate_off_poles,
    sphere_exp_map,
    sphere_log_map,
    tangent_basis,
    triangles_fold,
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


@pytest.fixture(scope="module")
def ico4_grids():
    """The bundled sphere's two hemispheres, rotated off the poles, at a low basis order."""
    from sbci.alignment import _hemisphere_grids

    return _hemisphere_grids(order=2)


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


def test_a_zero_step_keeps_the_identity_jacobian_at_one(grid, ico4_grids):
    """The scheme's bias on the identity is divided out; ``reference=True`` keeps it.

    Uncalibrated, the central difference through points relocated on the
    planar mesh returns 0.805 for the identity on this 42-vertex sphere and
    0.9965 on the bundled ico4 grid (second review, finding A). On the toy
    grid the calibrated value comes back bit for bit; on ico4 the
    renormalization of the unmoved vertices in :meth:`SphericalWarp.compose`
    changes last bits, which the ``2e-5`` step magnifies to 4e-11.
    """
    zero = np.zeros((grid.n_vertices, 2))
    calibrated = SphericalWarp(grid, delta=1e-5).compose(zero).jacobian
    np.testing.assert_allclose(calibrated, 1.0, atol=1e-12)
    uncalibrated = SphericalWarp(grid, delta=1e-5, reference=True).compose(zero).jacobian
    assert 0.79 < uncalibrated.min() and uncalibrated.max() < 0.82

    left = ico4_grids[0]
    zero = np.zeros((left.n_vertices, 2))
    calibrated = SphericalWarp(left, delta=1e-5).compose(zero).jacobian
    np.testing.assert_allclose(calibrated, 1.0, atol=1e-10)
    uncalibrated = SphericalWarp(left, delta=1e-5, reference=True).compose(zero).jacobian
    assert uncalibrated.mean() == pytest.approx(0.9965, abs=5e-4)
    assert 0.995 < uncalibrated.min() and uncalibrated.max() < 0.998


def test_evaluate_conserves_the_mass_of_a_density_on_the_bundled_grid(ico4_grids):
    """A density pushed through small warps keeps its mass; uncalibrated, it loses 0.7%.

    :meth:`Concon.evaluate` transports with the product of the two
    hemispheres' Jacobians, so the identity's bias of 0.9965 on each costs
    ``1 - 0.9965^2``, about 0.7% of the mass (second review, finding A). That
    is a loss of the aligned density's, once, when the final warps are
    applied; the registration's own transport (:meth:`Concon.evaluate_root`)
    renormalizes every trial. Interpolation alone moves the mass of this
    smooth density by 2e-5, so the rest is the Jacobian's.
    """
    lh_grid, rh_grid = ico4_grids
    concon = Concon(lh_grid, rh_grid, delta=1e-5)
    smooth = 1.0 + 0.5 * np.concatenate([lh_grid.vertices[:, 2], rh_grid.vertices[:, 2]])
    density = np.outer(smooth, smooth)
    np.fill_diagonal(density, 0.0)
    mass = (density * concon.area_product).sum()

    calibrated, uncalibrated = [], []
    for hemisphere in (lh_grid, rh_grid):
        displacement = 0.01 * np.stack(
            [np.sin(3 * hemisphere.theta), np.cos(2 * hemisphere.phi) * np.sin(hemisphere.theta)],
            axis=1,
        )
        calibrated.append(SphericalWarp(hemisphere, delta=1e-5).compose(displacement))
        uncalibrated.append(
            SphericalWarp(hemisphere, delta=1e-5, reference=True).compose(displacement)
        )
    assert not any(warp.folds() for warp in calibrated + uncalibrated)

    kept = (concon.evaluate(density, *calibrated) * concon.area_product).sum() / mass
    assert kept == pytest.approx(1.0, abs=1e-3)
    lost = 1 - (concon.evaluate(density, *uncalibrated) * concon.area_product).sum() / mass
    assert 0.005 < lost < 0.01


def test_the_bias_the_calibration_removes_shrinks_under_refinement():
    """Uncalibrated, the identity's Jacobian is not 1 on a coarse mesh, and it should not be.

    The Jacobian is a finite difference of a function interpolated over flat
    triangles, so it carries the mesh's chord-versus-arc error. What has to be
    true is that refining the mesh drives it to 1 -- the reference behaves the
    same way: on its own demo mesh it reports a range of 0.824 to 1.070 for a
    small warp -- and that the calibration removes it at every level.
    """
    errors = []
    for level in (1, 2, 3):
        vertices, faces = icosphere(level)
        mesh = SphericalGrid(vertices, faces, order=2)
        zero = np.zeros((mesh.n_vertices, 2))
        away = np.setdiff1d(np.arange(mesh.n_vertices), mesh.pole_vertices())
        uncalibrated = SphericalWarp(mesh, delta=1e-5, reference=True).compose(zero).jacobian
        errors.append(np.abs(uncalibrated[away] - 1).max())
        calibrated = SphericalWarp(mesh, delta=1e-5).compose(zero).jacobian
        np.testing.assert_allclose(calibrated[away], 1.0, atol=1e-10)
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


def test_align_turns_a_supplied_grid_off_the_poles_and_records_the_turn():
    """As the bundled grid is turned: the warps then carry the turn, which migrate_warp needs.

    align used to refuse such a grid and advise rotate_off_poles; a grid turned
    that way gave warps in a frame nothing recorded, which migrate_warp placed
    by the turn off -- up to 17 degrees for the bundled sphere's. A grid the caller
    turned is recorded when ``grid_rotations`` says so.
    """
    from sbci.alignment import pole_rotation

    vertices, faces = icosphere(2)
    mesh = SphericalGrid(vertices, faces, order=2)
    n = 2 * mesh.n_vertices
    rng = np.random.default_rng(2)
    matrix = rng.random((n, n))
    matrix = matrix + matrix.T
    turn = pole_rotation(vertices)
    options = dict(order=2, max_iterations=1, template_iterations=1)
    result = align([matrix, matrix], grids=(mesh, mesh), **options)
    for rotation in (
        *result.grid_rotations,
        result.warps[0].lh_rotation,
        result.warps[1].rh_rotation,
    ):
        np.testing.assert_allclose(rotation, turn)
    turned = SphericalGrid(vertices @ turn.T, faces, order=2)
    told = align([matrix, matrix], grids=(turned, turned), grid_rotations=(turn, turn), **options)
    np.testing.assert_allclose(told.warps[0].lh_rotation, turn)
    with pytest.raises(ValueError, match="pass grids="):
        align([matrix, matrix], grid_rotations=(turn, turn), **options)


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


def test_a_step_too_long_is_halved_rather_than_ending_the_registration(pair):
    """The reference stops at the first non-improving step; the port halves it first."""
    grid, densities = pair
    before = (
        (Encore(grid, grid).root(densities[0]) - Encore(grid, grid).root(densities[1])) ** 2
        * Encore(grid, grid).area_product
    ).sum()
    reference = Encore(grid, grid, step=5.0, max_iterations=3, delta=1e-5, backtracks=0)
    _, _, _, stuck = reference.register(densities[0], densities[1])
    assert stuck == pytest.approx(before)  # a step this long overshoots, and the reference gives up
    halving = Encore(grid, grid, step=5.0, max_iterations=3, delta=1e-5, backtracks=8)
    _, _, _, moved = halving.register(densities[0], densities[1])
    assert moved < before - 1e-9


def test_a_default_step_registration_lowers_the_cost_with_the_calibrated_jacobian(pair):
    """The calibration rescales the Jacobian field; the descent must still make progress.

    On this 42-vertex sphere the uncalibrated identity is 0.805, so before the
    second review's finding A the aligned density came back with 65% of its
    mass; now its Jacobians sit around 1 and the mass is kept.
    """
    grid, densities = pair
    encore = Encore(grid, grid, step=0.05, max_iterations=5, delta=1e-5)
    before = (
        (encore.root(densities[0]) - encore.root(densities[1])) ** 2 * encore.area_product
    ).sum()
    aligned, lh_warp, rh_warp, after = encore.register(densities[0], densities[1])
    assert after < before - 1e-6
    assert not lh_warp.reference and not rh_warp.reference
    for warp in (lh_warp, rh_warp):
        assert 0.9 < warp.jacobian.min() and warp.jacobian.max() < 1.1
    mass = (densities[1] * encore.area_product).sum()
    assert (aligned * encore.area_product).sum() == pytest.approx(mass, rel=0.01)


def test_a_density_with_a_nonzero_diagonal_registers_like_one_without(pair):
    """The diagonal is not part of the density, so it must not change a registration.

    Before the second review's finding B, ``root`` kept it: the starting cost
    then carried the fixed density's whole diagonal as a residual that no
    trial image -- all of which come from :meth:`Concon.evaluate_root`, which
    zeroes it -- could remove, and every step was refused.
    """
    grid, densities = pair
    encore = Encore(grid, grid, step=0.05, max_iterations=5, delta=1e-5)
    with_diagonal = [density + 3.0 * np.eye(density.shape[0]) for density in densities[:2]]
    untouched = [density.copy() for density in with_diagonal]

    clean = encore.register(densities[0], densities[1])
    dirty = encore.register(*with_diagonal)
    before = (
        (encore.root(with_diagonal[0]) - encore.root(with_diagonal[1])) ** 2 * encore.area_product
    ).sum()
    assert dirty[3] < before - 1e-6
    np.testing.assert_array_equal(dirty[0], clean[0])
    for mine, theirs in zip(dirty[1:3], clean[1:3], strict=True):
        np.testing.assert_array_equal(mine.vertices, theirs.vertices)
        np.testing.assert_array_equal(mine.jacobian, theirs.jacobian)
    assert dirty[3] == clean[3]
    for given, kept in zip(with_diagonal, untouched, strict=True):
        np.testing.assert_array_equal(given, kept)  # root() and register() work on copies
    # the template and align() go through root() as well
    np.testing.assert_array_equal(encore.root(with_diagonal[0]), encore.root(densities[0]))
    np.testing.assert_array_equal(
        encore.template(with_diagonal, iterations=2), encore.template(densities[:2], iterations=2)
    )


def test_align_records_a_cost_trace_per_subject(pair):
    from sbci.alignment import align

    grid, densities = pair
    result = align(densities[:2], grids=(grid, grid), max_iterations=3, template_iterations=1)
    assert len(result.traces) == 2
    for trace, final in zip(result.traces, result.costs, strict=True):
        assert trace.ndim == 1 and trace.size >= 1 and trace[-1] == pytest.approx(final)
        assert np.all(np.diff(trace) <= 1e-12)  # every accepted step lowers the cost


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
    with pytest.raises(ValueError, match="at least one"):
        align([])
    # one connectome is enough once there is a template to register it onto
    template = Encore(grid, grid).root(densities[0])
    single = align(densities[:1], template=template, grids=(grid, grid), max_iterations=1)
    assert len(single.warps) == 1 and len(single.aligned) == 1
    with pytest.raises(ValueError, match="different grids"):
        align([densities[0], densities[1][:-2, :-2]])
    with pytest.raises(ValueError, match="but the grid has"):
        align([densities[0][:6, :6], densities[1][:6, :6]], grids=(grid, grid))


def test_warps_round_trip_through_save_and_load(tmp_path, grid):
    """PORTING.md item 4 names this as part of being correct."""
    displacement = 0.01 * np.stack([np.sin(grid.theta), np.cos(grid.phi)], axis=1)
    composed = SphericalWarp(grid, delta=1e-5).compose(displacement)
    warp = Warp(composed.vertices, composed.jacobian, composed.vertices, composed.jacobian)
    np.testing.assert_array_equal(warp.lh_rotation, np.eye(3))  # the grid's own frame by default

    path = warp.save(tmp_path / "sub-toy_warp.npz")
    back = Warp.load(path)
    np.testing.assert_array_equal(back.lh_vertices, warp.lh_vertices)
    np.testing.assert_array_equal(back.lh_jacobian, warp.lh_jacobian)
    np.testing.assert_array_equal(back.rh_vertices, warp.rh_vertices)
    np.testing.assert_array_equal(back.rh_jacobian, warp.rh_jacobian)
    np.testing.assert_array_equal(back.lh_rotation, np.eye(3))
    np.testing.assert_array_equal(back.rh_rotation, np.eye(3))

    # the grid rotations travel with the warp (second review, finding C)
    lh_rotation, rh_rotation = _rotation_about_y(0.3), _rotation_about_y(0.7)
    rotated = Warp(
        composed.vertices,
        composed.jacobian,
        composed.vertices,
        composed.jacobian,
        lh_rotation=lh_rotation,
        rh_rotation=rh_rotation,
    )
    back = Warp.load(rotated.save(tmp_path / "sub-rotated_warp.npz"))
    np.testing.assert_array_equal(back.lh_rotation, lh_rotation)
    np.testing.assert_array_equal(back.rh_rotation, rh_rotation)


def _rotation_about_y(angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def test_a_warp_saved_before_the_rotations_were_kept_still_loads(tmp_path, grid):
    """A file holding only the four original arrays reads back in the grid's own frame."""
    path = tmp_path / "sub-old_warp.npz"
    ones = np.ones(grid.n_vertices)
    np.savez_compressed(
        path,
        lh_vertices=grid.vertices,
        lh_jacobian=ones,
        rh_vertices=grid.vertices,
        rh_jacobian=ones,
    )
    old = Warp.load(path)
    np.testing.assert_array_equal(old.lh_vertices, grid.vertices)
    np.testing.assert_array_equal(old.lh_rotation, np.eye(3))
    np.testing.assert_array_equal(old.rh_rotation, np.eye(3))


def test_save_returns_the_file_numpy_wrote(tmp_path, grid):
    """``np.savez_compressed`` adds ``.npz`` to a name without it; the returned path must exist."""
    ones = np.ones(grid.n_vertices)
    warp = Warp(grid.vertices, ones, grid.vertices, ones)
    path = warp.save(tmp_path / "sub-toy_warp")
    assert path == tmp_path / "sub-toy_warp.npz" and path.exists()
    np.testing.assert_array_equal(Warp.load(path).lh_vertices, grid.vertices)
    assert warp.save(tmp_path / "sub-toy_warp.npz") == tmp_path / "sub-toy_warp.npz"


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


def test_the_tree_search_finds_the_same_face_as_the_exhaustive_one(grid):
    """The k-d-tree narrowing must never pick a different face than testing every face."""
    rng = np.random.default_rng(13)
    points = rng.normal(size=(2000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    query = MeshQuery(grid.vertices, grid.faces)
    _, _, faces = query.query_faces(points)
    assert np.array_equal(faces, query._exact_faces(points))


def test_pole_rotation_is_the_rotation_rotate_off_poles_applies():
    vertices, _ = icosphere(1)
    rotation = pole_rotation(vertices)
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    np.testing.assert_allclose(rotate_off_poles(vertices), vertices @ rotation.T)
    np.testing.assert_allclose(rotate_off_poles(vertices) @ rotation, vertices, atol=1e-12)


def test_one_subject_registers_onto_another(pair):
    """template=<connectome> moves the listed subject onto that one, which stays where it is."""
    from sbci.alignment import Encore

    grid, densities = pair
    fixed, moving = densities[0], densities[1]

    class _Fixed:  # anything with .dense(), as a ContinuousConnectome has
        def dense(self):
            return fixed

    result = align([moving], template=_Fixed(), grids=(grid, grid), max_iterations=2)
    np.testing.assert_allclose(result.template, Encore(grid, grid).root(fixed))
    assert len(result.warps) == 1 and len(result.aligned) == 1
    assert result.traces[0][0] > 0 and result.costs[0] <= result.traces[0][0]
    # the fixed subject is already on the template, so registering it there costs nothing
    same = align([fixed], template=_Fixed(), grids=(grid, grid), max_iterations=1)
    assert same.traces[0][0] == pytest.approx(0.0, abs=1e-12)


def test_align_refuses_a_misshapen_template_and_a_negative_density(pair):
    grid, densities = pair
    n = 2 * grid.n_vertices
    with pytest.raises(ValueError, match="template"):
        align(densities, template=np.ones(n), grids=(grid, grid), max_iterations=1)
    with pytest.raises(ValueError, match="nonnegative"):
        align([densities[0], -densities[1]], grids=(grid, grid), max_iterations=1)


def test_a_custom_grid_records_the_identity_rotation(pair):
    """Warps live in the grid's frame; a grid built off the poles records no rotation."""
    grid, densities = pair
    result = align(densities[:2], grids=(grid, grid), max_iterations=1)
    lh, rh = result.grid_rotations
    assert np.allclose(lh, np.eye(3)) and np.allclose(rh, np.eye(3))
    for warp in result.warps:
        np.testing.assert_array_equal(warp.lh_rotation, np.eye(3))
        np.testing.assert_array_equal(warp.rh_rotation, np.eye(3))


def test_align_records_the_grid_rotations_in_every_warp(pair, monkeypatch):
    """Each warp holds what ``Alignment.grid_rotations`` holds, so a saved warp keeps its frame.

    The bundled grids are stood in for by the toy grid, with two rotations of
    the kind :func:`rotate_off_poles` applies (second review, finding C).
    """
    from sbci import alignment

    grid, densities = pair
    lh_rotation, rh_rotation = _rotation_about_y(0.3), _rotation_about_y(0.7)

    def toy_hemispheres(order, rotate=True, return_rotations=False, reference=False):
        return ((grid, grid), (lh_rotation, rh_rotation)) if return_rotations else (grid, grid)

    monkeypatch.setattr(alignment, "_hemisphere_grids", toy_hemispheres)
    result = align(densities[:2], max_iterations=1, template_iterations=1)
    np.testing.assert_array_equal(result.grid_rotations[0], lh_rotation)
    np.testing.assert_array_equal(result.grid_rotations[1], rh_rotation)
    assert len(result.warps) == 2
    for warp in result.warps:
        np.testing.assert_array_equal(warp.lh_rotation, lh_rotation)
        np.testing.assert_array_equal(warp.rh_rotation, rh_rotation)


def test_the_threaded_sparse_product_is_bit_for_bit_the_single_call():
    """Row blocks in threads accumulate every output row exactly as one call does."""
    from scipy import sparse

    from sbci.alignment import PARALLEL_ELEMENTS, sparse_times_dense

    rng = np.random.default_rng(23)
    n = 700
    matrix = sparse.random(n, n, density=0.02, random_state=24, format="csr")
    dense = rng.standard_normal((n, n))
    assert dense.size < PARALLEL_ELEMENTS
    single = np.asarray(matrix @ dense)
    np.testing.assert_array_equal(sparse_times_dense(matrix, dense), single)  # small: one call
    forced = sparse_times_dense(matrix, dense, threads=4)
    np.testing.assert_array_equal(forced, single)  # still one call, below the threshold
    big = rng.standard_normal((n, (PARALLEL_ELEMENTS // n) + 1))
    np.testing.assert_array_equal(
        sparse_times_dense(matrix, big, threads=3), np.asarray(matrix @ big)
    )
    np.testing.assert_array_equal(
        sparse_times_dense(matrix.T.tocsc(), big, threads=2), np.asarray(matrix.T @ big)
    )
    fortran = np.asfortranarray(big)
    np.testing.assert_array_equal(
        sparse_times_dense(matrix, fortran, threads=2), np.asarray(matrix @ big)
    )


def test_an_empty_cohort_is_refused_however_it_is_passed():
    with pytest.raises(ValueError, match="at least one connectome"):
        align([])
    with pytest.raises(ValueError, match="at least one connectome"):
        align(np.zeros((0, 4, 4)))


# --- the Legendre recurrence, the root normalization and folding steps ------
#
# Three things the reference does that the port corrects by default and keeps
# behind ``reference=True`` for the MATLAB comparison, and one it does not.


def _normalized_legendre(degree, theta):
    """``P~_l^m(cos theta)`` for ``m = 0..l`` with the harmonic normalization."""
    from scipy.special import factorial, lpmv

    order = np.arange(degree + 1)[:, None]
    normalization = np.sqrt(
        ((2 * degree + 1) / (4 * np.pi)) * (factorial(degree - order) / factorial(degree + order))
    )
    return normalization * np.stack([lpmv(m, degree, np.cos(theta)) for m in range(degree + 1)])


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 15])
def test_the_legendre_derivatives_match_finite_differences(degree):
    """Both derivatives in theta, at every order, against central differences of the values.

    The reference's recurrence writes the ``m = -1`` member as if the functions
    were unnormalized, so its ``m = 0`` first derivative is scaled by
    ``(1 + 1/(l(l+1))) / 2`` and its second derivative is wrong below the top
    order; ``reference=True`` keeps that.
    """
    theta = np.linspace(0.3, np.pi - 0.3, 41)  # away from the poles
    h1, h2 = 1e-5, 1e-4
    values = _normalized_legendre(degree, theta)
    plus, minus = _normalized_legendre(degree, theta + h1), _normalized_legendre(degree, theta - h1)
    first_fd = (plus - minus) / (2 * h1)
    plus, minus = _normalized_legendre(degree, theta + h2), _normalized_legendre(degree, theta - h2)
    second_fd = (plus - 2 * values + minus) / h2**2

    second, first, got = _legendre_derivatives(degree, theta)
    np.testing.assert_allclose(got, values, rtol=1e-12)
    np.testing.assert_allclose(first, first_fd, rtol=1e-6, atol=1e-6 * np.abs(first_fd).max())
    np.testing.assert_allclose(second, second_fd, rtol=1e-5, atol=1e-5 * np.abs(second_fd).max())

    factor = 0.5 * (1 + 1 / (degree * (degree + 1)))
    reference_second, reference_first, _ = _legendre_derivatives(degree, theta, reference=True)
    np.testing.assert_allclose(reference_first[0], factor * first[0], rtol=1e-12)
    np.testing.assert_array_equal(reference_first[1:], first[1:])  # the same arithmetic there
    assert np.abs(reference_second - second_fd).max() > 0.1 * np.abs(second_fd).max()


def test_the_corrected_recurrence_changes_the_divergence_and_not_the_basis():
    """A pure scale at m = 0: normalizing removed it from the fields, not from their divergence.

    The zonal gradient field ``grad Y_l0 / ||grad Y_l0||`` has divergence
    ``-l(l+1) Y_l0 / ||grad Y_l0||`` in the area-weighted norm the basis uses:
    at degree 1 that is ``-2 cos(theta) / sqrt(sum sin(theta)^2 areas)``. The
    reference's comes out 4/3 of it at degree 1 and 24/13 at degree 3.
    """
    vertices, faces = icosphere(2)
    vertices = rotate_off_poles(vertices)
    corrected = SphericalGrid(vertices, faces, order=3)
    reference = SphericalGrid(vertices, faces, order=3, reference=True)
    assert corrected.reference is False and reference.reference is True
    np.testing.assert_allclose(corrected.basis, reference.basis, atol=1e-14)

    cos, sin, areas = np.cos(corrected.theta), np.sin(corrected.theta), corrected.areas
    degree_1 = -2 * cos / np.sqrt((sin**2 * areas).sum())
    legendre_3 = (5 * cos**3 - 3 * cos) / 2
    d_legendre_3 = -sin * (15 * cos**2 - 3) / 2
    degree_3 = -12 * legendre_3 / np.sqrt((d_legendre_3**2 * areas).sum())
    # degree 1, m = 0 is column 0; degree 3, m = 0 follows the 3 + 5 fields of degrees 1 and 2
    np.testing.assert_allclose(corrected.laplacian[:, 0], degree_1, atol=1e-13)
    np.testing.assert_allclose(corrected.laplacian[:, 8], degree_3, atol=1e-13)
    np.testing.assert_allclose(reference.laplacian[:, 0], (4 / 3) * degree_1, atol=1e-13)
    np.testing.assert_allclose(reference.laplacian[:, 8], (24 / 13) * degree_3, atol=1e-13)


def test_evaluate_root_keeps_unit_norm_once_the_diagonal_is_zeroed():
    """Interpolation puts mass on the diagonal; normalizing before zeroing it loses that mass.

    A density concentrated near the diagonal -- each vertex connected to its
    neighbours -- shows it most: transported by the reference's order
    (``reference=True``) it comes back short of unit norm, by the port's not.
    """
    vertices, faces = icosphere(2)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    points = np.vstack([grid.vertices, grid.vertices])
    distance = np.arccos(np.clip(points @ points.T, -1.0, 1.0))
    local = np.exp(-(distance**2) / (2 * 0.25**2))
    np.fill_diagonal(local, 0.0)
    displacement = 0.05 * np.stack([np.sin(3 * grid.theta), np.cos(2 * grid.phi)], axis=1)
    warp = SphericalWarp(grid, delta=1e-5).compose(displacement)
    assert not warp.folds()

    corrected = Concon(grid, grid, delta=1e-5)
    reference = Concon(grid, grid, delta=1e-5, reference=True)
    root = np.sqrt(local / (local * corrected.area_product).sum())

    def norm(q):
        return (q**2 * corrected.area_product).sum()

    transported = corrected.evaluate_root(root, warp, warp)
    assert norm(transported) == pytest.approx(1.0, abs=1e-12)
    assert np.abs(np.diag(transported)).max() == 0.0
    assert norm(reference.evaluate_root(root, warp, warp)) < 0.999
    # nothing left off the diagonal: zeros, not a division by zero
    nothing = corrected.evaluate_root(np.zeros_like(root), warp, warp)
    np.testing.assert_array_equal(nothing, 0.0)


def test_a_step_that_folds_a_face_is_refused(monkeypatch):
    """A flipped face has a positive Jacobian (an absolute value), so the reference accepts it.

    At the reviewer's setting -- a step of 2 with two halvings, order 6 on a
    coarse mesh -- the reference's rule returns warps with folded faces and a
    positive Jacobian everywhere; the port halves the step instead, and the
    cost still falls.
    """
    vertices, faces = icosphere(1)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=6)
    rng = np.random.default_rng(5)
    n = 2 * grid.n_vertices
    densities = []
    for _ in range(2):
        matrix = rng.random((n, n))
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0.0)
        densities.append(matrix)
    encore = Encore(grid, grid, step=2.0, backtracks=2, max_iterations=5, delta=1e-5)
    before = (
        (encore.root(densities[0]) - encore.root(densities[1])) ** 2 * encore.area_product
    ).sum()

    _, lh_warp, rh_warp, after = encore.register(densities[0], densities[1])
    assert not triangles_fold(lh_warp.vertices, grid.faces)
    assert not triangles_fold(rh_warp.vertices, grid.faces)
    assert not lh_warp.folds() and not rh_warp.folds()
    assert after <= before + 1e-12

    monkeypatch.setattr(SphericalWarp, "folds", lambda self: False)  # the reference's rule
    _, lh_accepting, rh_accepting, _ = encore.register(densities[0], densities[1])
    assert triangles_fold(lh_accepting.vertices, grid.faces) or triangles_fold(
        rh_accepting.vertices, grid.faces
    )
    assert min(lh_accepting.jacobian.min(), rh_accepting.jacobian.min()) > 0


def test_the_fold_check_leaves_a_default_step_registration_untouched(pair, monkeypatch):
    """At the default step no trial folds, so the check may not change a single bit."""
    grid, densities = pair
    encore = Encore(grid, grid, step=0.05, max_iterations=5, delta=1e-5)
    checked = encore.register(densities[0], densities[1])
    monkeypatch.setattr(SphericalWarp, "folds", lambda self: False)
    unchecked = encore.register(densities[0], densities[1])

    np.testing.assert_array_equal(checked[0], unchecked[0])
    for mine, theirs in zip(checked[1:3], unchecked[1:3], strict=True):
        np.testing.assert_array_equal(mine.vertices, theirs.vertices)
        np.testing.assert_array_equal(mine.jacobian, theirs.jacobian)
    assert checked[3] == unchecked[3]


def test_a_mesh_wound_inward_is_tested_for_folds_with_its_faces_reversed(grid):
    """The fold test assumes outward faces; an inward mesh is a valid grid everywhere else."""
    inward = SphericalGrid(grid.vertices, grid.faces[:, ::-1], order=2)
    assert triangles_fold(inward.vertices, inward.faces)
    assert not SphericalWarp(inward, delta=1e-5).folds()
    smooth = 0.02 * np.stack([np.sin(3 * grid.theta), np.cos(2 * grid.phi)], axis=1)
    rough = 0.3 * np.random.default_rng(8).normal(size=(grid.n_vertices, 2))
    for displacement, expected in ((smooth, False), (rough, True)):
        outward = SphericalWarp(grid, delta=1e-5).compose(displacement)
        reversed_faces = SphericalWarp(inward, delta=1e-5).compose(displacement)
        assert outward.folds() is expected
        assert reversed_faces.folds() is expected


def test_reference_mode_reaches_the_grids_and_the_estimator(pair):
    """``align(reference=True)`` builds reference grids and hands the switch to :class:`Encore`."""
    from sbci.alignment import _hemisphere_grids

    assert all(g.reference for g in _hemisphere_grids(order=1, reference=True))
    assert not any(g.reference for g in _hemisphere_grids(order=1))

    grid, densities = pair
    reference = SphericalGrid(grid.vertices, grid.faces, order=2, reference=True)
    encore = Encore(reference, reference, max_iterations=2, delta=1e-5, reference=True)
    assert encore.concon.reference and not Encore(grid, grid).concon.reference
    template = encore.root(densities[0])
    _, lh_warp, rh_warp, cost = encore.register(template, densities[1], target_is_root=True)
    assert lh_warp.reference and rh_warp.reference  # the uncalibrated Jacobian, as the reference's
    result = align(
        densities[1:2],
        template=template,
        grids=(reference, reference),
        reference=True,
        max_iterations=2,
        delta=1e-5,
    )
    assert result.costs[0] == cost


# --- the third review: the basis on the axis, the clearance from it, empty densities ------


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 15])
def test_the_m1_fields_at_a_pole_are_the_gradient_of_a_smooth_function(degree):
    """``P~_l^1(cos theta) cos(phi)`` is ``-N x P_l'(z)``, so its gradient at ``z = +-1`` is fixed.

    It is ``-N P_l'(+-1)`` times the x axis, and the sine field's the same
    times the y axis, whatever phi the pole was given: equal in size and
    orthogonal. The reference's clamp on ``sin(theta)`` drops their
    phi-components there instead; at ``phi = pi``, which atan2 gives the
    bundled sphere's poles, the sine field vanishes and the cosine field
    keeps its size (third review).
    """
    theta = np.array([0.0, 0.0, 0.0, 0.0, np.pi, np.pi, np.pi])
    phi = np.array([np.pi, 0.0, np.pi / 2, 4.0, np.pi, 0.0, 2.0])
    e1 = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], axis=1
    )
    e2 = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    z = np.cos(theta)  # exactly +1 or -1, where P_l'(z) = z^(l+1) l(l+1)/2
    normalization = np.sqrt((2 * degree + 1) / (4 * np.pi * degree * (degree + 1)))
    slope = normalization * z ** (degree + 1) * degree * (degree + 1) / 2

    def vectors(gradient):
        return [gradient[0, k][:, None] * e1 + gradient[1, k][:, None] * e2 for k in (1, 2)]

    gradient, _ = _harmonic_derivatives(degree, theta, phi)
    cosine, sine = vectors(gradient)
    tolerance = 1e-12 * np.abs(slope).max()
    np.testing.assert_allclose(cosine, -slope[:, None] * np.eye(3)[0], rtol=0, atol=tolerance)
    np.testing.assert_allclose(sine, -slope[:, None] * np.eye(3)[1], rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        np.linalg.norm(sine, axis=1), np.linalg.norm(cosine, axis=1), rtol=1e-9
    )
    assert np.abs((cosine * sine).sum(axis=1)).max() <= tolerance * np.abs(slope).max()
    # every other field of the degree has no gradient on the axis
    assert np.abs(gradient[:, [0, *range(3, 2 * degree + 1)]]).max() <= tolerance

    reference, _ = _harmonic_derivatives(degree, theta, phi, reference=True)
    reference_cosine, reference_sine = vectors(reference)
    at_pi = [0, 4]  # both poles with phi = pi
    np.testing.assert_allclose(reference_cosine[at_pi], cosine[at_pi], rtol=0, atol=tolerance)
    assert np.abs(reference_sine[at_pi]).max() <= tolerance


@pytest.mark.parametrize("degree", [1, 6, 15])
def test_the_phi_component_is_right_at_any_distance_from_the_axis(degree):
    """Near the axis ``P_l^1 / sin(theta)`` loses digits to ``cos(theta)`` rounding towards 1.

    Checked against ``P~_l^1 / sin = -N P_l'(cos)`` and ``P~_l^2 / sin =
    N sin P_l''(cos)``, which have no 0/0: the quotient alone is 5% off for
    ``m = 1`` 2e-8 from the axis, and the reference's clamp 90% off 1e-6
    from it, so within 1e-5 the ``m = 1`` term is the limit instead. Just
    outside, the quotient is good to about 1e-6, ``eps / sin(theta)^2``.
    """
    from numpy.polynomial.legendre import Legendre
    from scipy.special import factorial

    distance = np.array([0.0, 1e-9, 2e-8, 1e-7, 1e-6, 9e-6, 1.1e-5, 1e-4, 1e-2, 0.5])
    theta = np.concatenate([distance, np.pi - distance])
    gradient, _ = _harmonic_derivatives(degree, theta, np.zeros_like(theta))
    # at phi = 0 the sine slot 2m carries m P~_l^m / sin(theta) as its phi-component
    for m, tolerance in ((1, 1e-5), (2, 1e-6)):
        if m > degree:
            continue
        normalization = np.sqrt(
            (2 * degree + 1) / (4 * np.pi) * factorial(degree - m) / factorial(degree + m)
        )
        derivative = Legendre.basis(degree).deriv(m)(np.cos(theta))
        exact = m * normalization * derivative * (-1.0 if m == 1 else np.sin(theta))
        np.testing.assert_allclose(
            gradient[1, 2 * m], exact, rtol=tolerance, atol=tolerance * np.abs(exact).max()
        )


def test_the_basis_at_a_pole_does_not_depend_on_the_phi_it_was_given():
    """The bundled sphere unrotated, as ConSEAL's default grids are, puts a vertex on each pole.

    atan2 gives those vertices, ``(-0, -0, +-1)``, ``phi = pi``; ``(0, 0,
    +-1)`` would get 0, and a y of 1e-300 pi/2 -- the same points. With the
    reference's clamp the 15 ``m = 1`` sine fields and their 15 rotated
    partners, 30 of the 510, vanished at the poles while their cosine
    partners kept their size, and which ones vanished depended on phi (third
    review). Now every field is one vector there whatever phi the pole has,
    and the two ``m = 1`` fields of a degree are orthogonal and of one size --
    up to the grid's own quadrature, which tells their norms apart by 1e-5:
    the file's ico4 sphere is five-fold symmetric about the axis only to
    1e-4.
    """
    from sbci.alignment import _hemisphere_grids

    bundled = _hemisphere_grids(1, rotate=False)[0]
    poles = np.flatnonzero(np.sin(bundled.theta) < 1e-8)
    assert poles.tolist() == [0, 11] and np.all(bundled.phi[poles] == np.pi)
    cosine = np.array([degree**2 for degree in range(1, 16)])  # (degree, m = 1) cosine fields
    sine = cosine + 1
    rotated = 255  # the rotated partner of field k is field k + 255

    def at_poles(x, y, reference):
        vertices = bundled.vertices.copy()
        vertices[poles, 0], vertices[poles, 1] = x, y
        grid = SphericalGrid(vertices, bundled.faces, order=15, reference=reference)
        basis, e1, e2 = grid.basis[poles], grid.e1[poles][:, None], grid.e2[poles][:, None]
        return grid.phi[poles], basis[..., :1] * e1 + basis[..., 1:] * e2

    _, vectors = at_poles(-0.0, -0.0, reference=False)
    for x, y, expected in ((0.0, 0.0, 0.0), (0.0, 1e-300, np.pi / 2)):
        other_phi, other = at_poles(x, y, reference=False)
        np.testing.assert_array_equal(other_phi, expected)
        np.testing.assert_allclose(other, vectors, rtol=0, atol=1e-12)

    for first, second in ((cosine, sine), (cosine + rotated, sine + rotated)):
        size = np.linalg.norm(vectors[:, first], axis=-1)
        assert size.min() > 0.3
        np.testing.assert_allclose(np.linalg.norm(vectors[:, second], axis=-1), size, rtol=2e-5)
        assert np.abs((vectors[:, first] * vectors[:, second]).sum(axis=-1)).max() < 1e-12

    _, clamped = at_poles(-0.0, -0.0, reference=True)
    sizes = np.linalg.norm(clamped, axis=-1)
    assert sizes[:, np.concatenate([sine, sine + rotated])].max() < 1e-12
    np.testing.assert_allclose(sizes[:, cosine], np.linalg.norm(vectors[:, cosine], axis=-1))
    _, turned = at_poles(0.0, 1e-300, reference=True)
    assert np.abs(turned - clamped).max() > 0.3  # with the clamp, the basis there depends on phi


def test_the_pole_limit_changes_nothing_on_a_grid_clear_of_the_axis(ico4_grids):
    """ENCORE's grids are rotated off the axis, so their basis keeps every bit it had.

    Off the axis the phi-component is still the reference's expression,
    ``max(sin(theta), 1e-5)`` and all -- the clamp never binds 0.023 from the
    axis, where the bundled grids' nearest vertex sits -- so the default and
    reference bases differ only where the recurrence does, at ``m = 0``.
    """
    for grid in ico4_grids:
        theta, phi = grid.theta, grid.phi
        assert np.sin(theta).min() > AXIS_CLEARANCE
        clamped = np.maximum(np.sin(theta), 1e-5)
        for degree in (1, 2, 6, 15):
            gradient, _ = _harmonic_derivatives(degree, theta, phi)
            _, _, values = _legendre_derivatives(degree, theta)
            order = np.arange(degree + 1)[:, None]
            cosine_slots = np.concatenate([[0], np.arange(1, 2 * degree, 2)])
            sine_slots = np.arange(2, 2 * degree + 1, 2)
            sin_phi, cos_phi = np.sin(order * phi), np.cos(order * phi)
            np.testing.assert_array_equal(
                gradient[1, cosine_slots], values * -(order * sin_phi) / clamped
            )
            np.testing.assert_array_equal(
                gradient[1, sine_slots], values[1:] * (order[1:] * cos_phi[1:]) / clamped
            )

        default = SphericalGrid(grid.vertices, grid.faces, order=6)
        reference = SphericalGrid(grid.vertices, grid.faces, order=6, reference=True)
        fields = default.basis.shape[1] // 2
        zonal = [degree**2 - 1 for degree in range(1, 7)]
        others = np.setdiff1d(np.arange(2 * fields), zonal + [k + fields for k in zonal])
        np.testing.assert_array_equal(default.basis[:, others], reference.basis[:, others])


def test_a_grid_with_a_vertex_beside_the_axis_is_turned_clear_of_it():
    """Near the axis the ``(theta, phi)`` Jacobian is unreliable, not only on it.

    1e-5 rad from the axis the identity's comes out 0.74 on this mesh and a
    rigid rotation's, calibrated against it, 1.27. The check used to look
    only for vertices on the axis, ``sin(theta) < 1e-8``, and let such a grid
    through (third review); it now asks for the clearance that
    :func:`rotate_off_poles` guarantees, and the rotated grid passes.
    """
    vertices, faces = icosphere(2)
    n = 2 * vertices.shape[0]
    matrix = np.random.default_rng(3).random((n, n))
    matrix = matrix + matrix.T
    from sbci.alignment import pole_rotation

    for offset in (1e-5, 5e-4):
        near = SphericalGrid(vertices @ _rotation_about_y(offset).T, faces, order=2)
        assert near.pole_vertices(tolerance=1e-8).size == 0  # none on the axis itself
        assert near.pole_vertices().size == 2
        turned = align(
            [matrix, matrix], grids=(near, near), order=2, max_iterations=1, template_iterations=1
        )  # turned clear of the axis, and the turn recorded
        np.testing.assert_allclose(turned.grid_rotations[0], pole_rotation(near.vertices))

    clear = SphericalGrid(rotate_off_poles(vertices @ _rotation_about_y(1e-5).T), faces, order=2)
    assert clear.pole_vertices().size == 0
    result = align(
        [matrix, matrix], grids=(clear, clear), order=2, max_iterations=1, template_iterations=1
    )
    assert len(result.warps) == 2


def test_rotate_off_poles_clears_the_axis_as_the_grid_measures_it():
    """The clearance is ``sin(theta)`` on the unit sphere on both sides, whatever the radius.

    Measured on the raw coordinates, a sphere of radius 100 (FreeSurfer's)
    could not be rotated at all, and one of radius 0.5 was passed unrotated
    with its poles on the axis.
    """
    vertices, faces = icosphere(2)
    for radius in (1.0, 100.0, 0.5):
        grid = SphericalGrid(rotate_off_poles(radius * vertices), faces, order=1)
        assert grid.pole_vertices().size == 0
        assert np.sin(grid.theta).min() > AXIS_CLEARANCE
        np.testing.assert_array_equal(pole_rotation(radius * vertices), pole_rotation(vertices))


def test_the_jacobian_is_not_calibrated_against_a_vertex_beside_the_axis():
    """3e-6 rad from the axis the scheme's identity value is 0.03, which says nothing of the mesh.

    Divided out, it turned a rigid rotation's Jacobian there into 34 and 37
    (third review). Left alone, the two vertices read the scheme's own
    value, 0.945 on this coarse mesh, as the vertices on the axis always
    have; everywhere else the calibration is what it was.
    """
    vertices, faces = icosphere(2)
    grid = SphericalGrid(vertices @ _rotation_about_y(3e-6).T, faces, order=2)
    beside = np.flatnonzero(np.sin(grid.theta) < 1e-5)
    away = np.setdiff1d(np.arange(grid.n_vertices), beside)
    warp = SphericalWarp(grid, delta=1e-5)
    identity = warp._raw_jacobian()
    assert beside.size == 2 and identity[beside].max() < 0.5
    np.testing.assert_array_equal(warp._calibration[beside], 1.0)
    np.testing.assert_array_equal(warp._calibration[away], identity[away])

    turned = grid.vertices @ _rotation_about_y(np.radians(0.5)).T
    gamma = sphere_log_map(grid.vertices, turned)
    displacement = np.stack([(gamma * grid.e1).sum(axis=1), (gamma * grid.e2).sum(axis=1)], axis=1)
    jacobian = warp.compose(displacement).jacobian
    assert 0.9 < jacobian[beside].min() and jacobian[beside].max() < 1.1
    np.testing.assert_allclose(jacobian[away], 1.0, atol=1e-7)


def test_a_density_with_nothing_off_its_diagonal_is_refused(pair):
    """The diagonal is no part of a density, so a density with nothing else is empty.

    :meth:`Encore.root` zeroes the diagonal before it normalizes (second
    review), so such a density divided 0 by 0 and the template and costs
    came back NaN without a word, while :func:`align` checked the whole
    matrix for mass (third review).
    """
    grid, densities = pair
    n = 2 * grid.n_vertices
    diagonal = np.diag(np.linspace(1.0, 2.0, n))

    class _Connectome:  # anything with .dense(), as a ContinuousConnectome has
        def __init__(self, matrix):
            self.matrix = matrix

        def dense(self):
            return self.matrix

    encore = Encore(grid, grid)
    for empty in (diagonal, np.zeros((n, n))):
        with pytest.raises(ValueError, match="no mass off its diagonal"):
            encore.root(empty)
    with pytest.raises(ValueError, match="not finite"):
        encore.root(np.full((n, n), np.nan))
    for given in (diagonal, _Connectome(diagonal)):
        with pytest.raises(ValueError, match="connectome 1 has no mass off its diagonal"):
            align([densities[0], given], grids=(grid, grid), max_iterations=1)
    with pytest.raises(ValueError, match="template connectome has no mass off its diagonal"):
        align([densities[0]], template=_Connectome(diagonal), grids=(grid, grid), max_iterations=1)
    with pytest.raises(ValueError, match="not square"):
        align([np.ones(n), np.ones(n)], grids=(grid, grid))

    one_pair = diagonal.copy()
    one_pair[0, 1] = one_pair[1, 0] = 1.0  # a single connection off the diagonal is mass enough
    root = encore.root(one_pair)
    assert np.isfinite(root).all()
    assert (root**2 * encore.area_product).sum() == pytest.approx(1.0)


def test_a_coarse_grid_is_calibrated_however_far_its_identity_is_from_one():
    """On the 12-vertex icosahedron the scheme reads 0.46 for the identity: the mesh's chord error.

    Clear of the axis it is divided out like any other value, so the identity
    reads exactly 1. Until 6 October 2026 a value below one half was left
    alone (fourth review), and this grid's identity Jacobian read 0.46.
    """
    vertices, faces = icosphere(0)
    grid = SphericalGrid(vertices, faces, order=1)
    assert grid.n_vertices == 12 and np.sin(grid.theta).min() > AXIS_CLEARANCE
    warp = SphericalWarp(grid)
    assert warp._raw_jacobian().min() < 0.5
    np.testing.assert_allclose(warp._compute_jacobian(), 1.0, rtol=1e-12)


def test_a_density_stored_as_one_triangle_is_refused(pair):
    """The gradient reads each vertex's row; one triangle gives half the rows nothing.

    Accepted before, such a matrix silently never registered.
    """
    grid, densities = pair
    upper = np.triu(densities[1])
    with pytest.raises(ValueError, match="connectome 1 is not symmetric.*D \\+ D.T - diag"):
        align([densities[0], upper], grids=(grid, grid), max_iterations=1)
    symmetric = upper + upper.T - np.diag(np.diagonal(upper))
    align([densities[0], symmetric], grids=(grid, grid), max_iterations=1, template_iterations=1)
