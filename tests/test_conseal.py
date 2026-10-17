"""ConSEAL: what the mathematics guarantees on any icosphere.

The comparison against the reference MATLAB lives in ``tests/reference``
(``conseal_reference.m`` and ``conseal_compare.py``) and is run by hand on
Longleaf. What is asserted here holds without it: point location agrees with
the exact search, the adjacency and the kernel conserve mass, every analytic
derivative matches a finite difference of the quantity it claims to
differentiate, warps stay diffeomorphic, and registration recovers a rotation
and reduces the cost of a warp. Where a test names an error of the reference,
``tests/reference/reference_conseal.py`` puts that error back beside the
port's correction, so the test shows the two part.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from sbci.alignment import MeshQuery, SphericalGrid, sphere_exp_map
from sbci.conseal import (
    ConSEAL,
    EndpointConnectome,
    EndpointWarp,
    HeatKernelBuilder,
    StationaryWarp,
    _legendre_with_derivative,
    cotangent_laplacian,
    default_grids,
    dice_score,
    endpoint_mass,
    endpoints_align,
    icosphere,
    mesh_symmetries,
    overlap_coefficient,
    rotation_shells,
    triangles_fold,
)
from sbci.errors import MissingDataError
from sbci.smoothing import Endpoints

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
import reference_conseal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:skipping a warp update")

# where an EndpointConnectome holds its endpoints
LOCATIONS = ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out")


def von_mises(rng, centre, concentration, n):
    """Points concentrated about ``centre`` on the unit sphere."""
    centre = np.asarray(centre, dtype=np.float64) / np.linalg.norm(centre)
    points = rng.normal(size=(n, 3)) + concentration * centre
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def synthetic(rng, n=3000, concentration=4.0, jitter=0.0):
    """Endpoints in bundles: within the left, within the right, and across.

    ``jitter`` moves the bundle centres, so two calls give two subjects that
    differ smoothly rather than only by sampling noise.
    """
    thirds = n // 3
    j = jitter * rng.normal(size=(6, 3))
    p_in = np.vstack(
        [
            von_mises(rng, np.add([-1, 0.3, 0.2], j[0]), concentration, thirds),
            von_mises(rng, np.add([1, -0.2, 0.4], j[1]), concentration, thirds),
            von_mises(rng, np.add([-0.8, -0.5, 0.1], j[2]), concentration, n - 2 * thirds),
        ]
    )
    p_out = np.vstack(
        [
            von_mises(rng, np.add([-0.5, -0.6, 0.6], j[3]), concentration, thirds),
            von_mises(rng, np.add([0.7, 0.6, -0.3], j[4]), concentration, thirds),
            von_mises(rng, np.add([0.9, 0.3, -0.3], j[5]), concentration, n - 2 * thirds),
        ]
    )
    h_in = np.r_[np.zeros(thirds), np.ones(thirds), np.zeros(n - 2 * thirds)].astype(int)
    h_out = np.r_[np.zeros(thirds), np.ones(thirds), np.ones(n - 2 * thirds)].astype(int)
    return p_in, p_out, h_in, h_out


@pytest.fixture(scope="module")
def grid():
    vertices, faces = icosphere(2)
    return SphericalGrid(vertices, faces, order=3)


@pytest.fixture(scope="module")
def connectome(grid):
    rng = np.random.default_rng(7)
    return EndpointConnectome.from_points(grid, grid, *synthetic(rng))


@pytest.fixture(scope="module")
def kernel(grid):
    return HeatKernelBuilder(grid, grid, degree=12).compute(0.05)


# --- point location ------------------------------------------------------------


def test_query_faces_agrees_with_the_exact_search(grid):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(500, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    query = MeshQuery(grid.vertices, grid.faces)
    fast_w, fast_i, fast_f = query.query_faces(points)
    exact_w, exact_i = MeshQuery(grid.vertices, grid.faces).query(points)
    assert np.array_equal(fast_i, exact_i)
    assert np.allclose(fast_w, exact_w, atol=1e-12)
    assert np.array_equal(grid.faces[fast_f], fast_i)
    assert np.array_equal(fast_f, query._exact_faces(points))


def test_a_query_at_a_vertex_puts_all_weight_on_it(grid):
    w, i, _ = MeshQuery(grid.vertices, grid.faces).query_faces(grid.vertices[:20])
    assert np.allclose(w.max(axis=1), 1.0, atol=1e-9)
    assert np.array_equal(i[np.arange(20), np.argmax(w, axis=1)], np.arange(20))


# --- the endpoint connectome ---------------------------------------------------


def test_the_adjacency_is_symmetric_and_has_unit_mass(connectome):
    a = connectome.adjacency()
    assert np.allclose(a, a.T)
    assert np.isclose(a.sum(), 1.0)
    # unclamped barycentric weights make a few entries slightly negative, as in the
    # reference; the clamp happens on the smoothed connectome
    assert a.min() > -1e-3


def test_the_adjacency_matches_a_direct_loop(grid):
    rng = np.random.default_rng(11)
    c = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=60))
    n = c.n_vertices
    expected = np.zeros((n, n))
    for k in range(c.n_streamlines):
        for i in range(3):
            for j in range(3):
                expected[c.index_in[k, i], c.index_out[k, j]] += (
                    c.weights_in[k, i] * c.weights_out[k, j]
                )
    expected = (expected + expected.T) / (2 * c.n_streamlines)
    assert np.allclose(c.adjacency(), expected)


def test_the_smooth_connectome_conserves_mass_and_is_symmetric(connectome, kernel):
    k, _ = kernel
    f = connectome.evaluate(k)
    assert np.isclose(f.sum(), 1.0, atol=1e-9)
    assert np.allclose(f, f.T)
    assert (f >= 0).all()


def test_positions_round_trip_through_location(grid):
    rng = np.random.default_rng(5)
    p_in, p_out, h_in, h_out = synthetic(rng, n=300)
    c = EndpointConnectome.from_points(grid, grid, p_in, p_out, h_in, h_out)
    q_in, q_out = c.positions()
    # a point off the flat triangle is re-projected radially, so agreement is to the
    # sphere-to-chord gap of an ico2 triangle, not to rounding
    assert np.abs(np.arccos(np.clip((q_in * p_in).sum(1), -1, 1))).max() < 0.02
    assert np.abs(np.arccos(np.clip((q_out * p_out).sum(1), -1, 1))).max() < 0.02


def test_from_endpoints_uses_the_stored_barycentric_data(grid, connectome):
    endpoints = connectome.to_endpoints()
    rebuilt = EndpointConnectome.from_endpoints(endpoints, grid, grid)
    assert np.array_equal(rebuilt.index_in, connectome.index_in)
    assert np.allclose(rebuilt.weights_out, connectome.weights_out)
    assert np.array_equal(rebuilt.hemisphere_in, connectome.hemisphere_in)


def test_from_endpoints_refuses_vertex_only_data(grid):
    endpoints = Endpoints(
        surf_in=np.zeros(4, dtype=np.int8),
        surf_out=np.ones(4, dtype=np.int8),
        vtx_in=np.arange(4),
        vtx_out=np.arange(4),
        n_per_hemi=grid.n_vertices,
    )
    with pytest.raises(MissingDataError):
        EndpointConnectome.from_endpoints(endpoints, grid, grid)


def test_copy_and_reset_are_independent(connectome, grid):
    clone = connectome.copy()
    warp = StationaryWarp(grid)
    warp.compose(0.05 * np.ones((grid.n_vertices, 2)))
    clone.warp(warp, StationaryWarp(grid))
    assert not np.allclose(clone.weights_in, connectome.weights_in)
    clone.reset()
    assert np.allclose(clone.weights_in, connectome.weights_in)
    assert np.array_equal(clone.index_in, connectome.index_in)


# --- the kernel -------------------------------------------------------------------


def test_the_kernel_rows_sum_to_one_and_are_block_diagonal(grid, kernel):
    k, _ = kernel
    dense = k.toarray()
    assert np.allclose(dense.sum(axis=1), 1.0)
    n = grid.n_vertices
    assert not dense[:n, n:].any() and not dense[n:, :n].any()
    assert (dense >= 0).all()


def test_the_cutoff_zeroes_only_beyond_the_scan(grid):
    builder = HeatKernelBuilder(grid, grid, degree=12)
    cut = builder.cutoff(0.05)
    k = builder.compute(0.05, derivative=False).toarray()[: grid.n_vertices, : grid.n_vertices]
    cosines = grid.vertices @ grid.vertices.T
    assert not k[cosines < cut - 1e-12].any()
    assert (k[cosines > cut + 1e-12] > 0).all()


def raw_kernel_column(builder, grid, sigma, x):
    """The unnormalized kernel between every vertex and the point ``x`` (not clipped)."""
    w = builder.weights(sigma)
    cut = builder.cutoff(sigma)
    cos = grid.vertices @ x
    series = np.zeros(grid.n_vertices)
    for order, p, _ in _legendre_with_derivative(cos, builder.degree):
        series += w[order] * p
    series[cos < cut] = 0.0
    return series


def test_the_evaluation_derivative_matches_a_finite_difference(grid):
    """Move one evaluation vertex; the source normalization stays fixed."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigma, eps, a = 0.05, 1e-6, 17
    _, dk = builder.compute(sigma)
    n = grid.n_vertices
    raw = np.column_stack([raw_kernel_column(builder, grid, sigma, v) for v in grid.vertices])
    s = raw.sum(axis=1)
    inside = grid.vertices @ grid.vertices[a] > builder.cutoff(sigma) + 1e-4
    for axis, matrix in enumerate((dk.x, dk.y, dk.z)):
        step = np.zeros(3)
        step[axis] = eps
        plus = raw_kernel_column(builder, grid, sigma, grid.vertices[a] + step)
        minus = raw_kernel_column(builder, grid, sigma, grid.vertices[a] - step)
        numeric = (plus - minus) / (2 * eps) / s
        analytic = matrix.toarray()[:n, a]
        assert np.allclose(numeric[inside], analytic[inside], atol=1e-5 * np.abs(analytic).max())


def test_the_reference_derivative_is_taken_in_the_row_vertex(grid):
    """The reference differentiates K~ in its row vertex, normalization included (item 2)."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigma, eps, i = 0.05, 1e-6, 23
    dk = reference_conseal.kernel_derivative(builder, sigma)
    n = grid.n_vertices

    def normalized_row(x):
        row = raw_kernel_column(builder, grid, sigma, x)
        return row / row.sum()

    inside = grid.vertices @ grid.vertices[i] > builder.cutoff(sigma) + 1e-4
    for axis, matrix in enumerate((dk.x, dk.y, dk.z)):
        step = np.zeros(3)
        step[axis] = eps
        plus = normalized_row(grid.vertices[i] + step)
        minus = normalized_row(grid.vertices[i] - step)
        numeric = (plus - minus) / (2 * eps)
        analytic = matrix.toarray()[i, :n]
        assert np.allclose(numeric[inside], analytic[inside], atol=1e-5 * np.abs(analytic).max())


def test_bandwidth_cross_validation_is_an_exact_leave_one_out(grid):
    """Each streamline's own term is removed exactly, as leaving it out of the density does.

    The reference subtracts ``K(t_in, t_in) K(t_out, t_out)' / (M - 1)`` instead, without
    the barycentric weights (module docstring, item 7).
    """
    connectome = EndpointConnectome.from_points(
        grid, grid, *synthetic(np.random.default_rng(11), n=60)
    )
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigmas = [0.02, 0.05, 0.1]
    best, scores = builder.cross_validate(connectome, sigmas)
    m, n = connectome.n_streamlines, connectome.n_vertices
    adjacency = connectome.adjacency()
    expected = []
    for sigma in sigmas:
        kernel = builder.compute(sigma, derivative=False).toarray()
        values = []
        for t in range(m):
            a, b = np.zeros(n), np.zeros(n)
            np.add.at(a, connectome.index_in[t], connectome.weights_in[t])
            np.add.at(b, connectome.index_out[t], connectome.weights_out[t])
            without = adjacency - (np.outer(a, b) + np.outer(b, a)) / (2 * m)
            values.append(a @ kernel.T @ without @ kernel @ b)
        expected.append(np.mean(np.log(np.maximum(values, np.finfo(float).eps))))
    np.testing.assert_allclose(scores, expected, rtol=1e-12)
    assert best == sigmas[int(np.argmax(expected))]


# --- the connectome derivative -------------------------------------------------------


def test_the_connectome_derivative_matches_a_finite_difference(grid, connectome):
    """D_e1(a, c) is d/dt F(exp(x_a, t e1(a)), c) at t = 0, with F evaluated off-grid."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigma = 0.05
    k, dk = builder.compute(sigma)
    _, d_e1, _ = connectome.evaluate(k, dk)
    ak = connectome.adjacency() @ k.toarray()
    n = grid.n_vertices
    raw = np.column_stack([raw_kernel_column(builder, grid, sigma, v) for v in grid.vertices])
    s = raw.sum(axis=1)  # source normalization of one hemisphere; both grids are the same

    def f_row(x, side):
        column = np.zeros(2 * n)
        column[side * n : (side + 1) * n] = raw_kernel_column(builder, grid, sigma, x) / s
        return column @ ak  # F(x, c) for every c

    eps = 1e-5
    for a in (5, 40, n + 12):
        side, x, e1 = a // n, grid.vertices[a % n], connectome.e1[a]
        plus = f_row(sphere_exp_map(x[None], (eps * e1)[None])[0], side)
        minus = f_row(sphere_exp_map(x[None], (-eps * e1)[None])[0], side)
        numeric = (plus - minus) / (2 * eps)
        assert np.allclose(numeric, d_e1[a], atol=2e-4 * np.abs(d_e1[a]).max())


def test_the_reference_derivative_differs_from_the_true_one(grid, connectome):
    """The reference adds the other slot's derivative in the wrong frame (items 1 and 2)."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    k, dk = builder.compute(0.05)
    _, true_e1, _ = connectome.evaluate(k, dk)
    reference_dk = reference_conseal.kernel_derivative(builder, 0.05)
    _, reference_e1, _ = reference_conseal.evaluate(connectome, k, reference_dk)
    assert np.corrcoef(true_e1.ravel(), reference_e1.ravel())[0, 1] < 0.99


# --- the warp -----------------------------------------------------------------------


def test_the_identity_warp_moves_nothing(grid):
    warp = StationaryWarp(grid)
    assert np.allclose(warp.vertices, grid.vertices)
    assert np.allclose(warp.jacobian, 1.0)
    assert warp.compose(np.zeros((grid.n_vertices, 2)))
    assert np.allclose(warp.vertices, grid.vertices, atol=1e-12)


def test_the_flow_of_a_rotational_field_is_that_rotation(grid):
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    warp = StationaryWarp(grid)
    warp.velocity = rotational_velocity(warp, rotation)
    warp.vertices = warp.exponential()
    expected = grid.vertices @ rotation.T
    # the flow interpolates the field linearly over ico2 triangles, so this is approximate
    assert np.abs(np.arccos(np.clip((warp.vertices * expected).sum(1), -1, 1))).max() < 1e-2
    assert np.allclose(warp.jacobian, 1.0, atol=0.1)


def test_a_composed_warp_stays_on_the_sphere_with_positive_jacobian(grid):
    rng = np.random.default_rng(2)
    warp = StationaryWarp(grid)
    for _ in range(5):
        assert warp.compose(0.03 * rng.normal(size=(grid.n_vertices, 2)))
    assert np.allclose(np.linalg.norm(warp.vertices, axis=1), 1.0)
    assert (warp.jacobian > 0).all()
    assert not triangles_fold(warp.vertices, grid.faces)


def test_the_fold_test_sees_an_inverted_mesh(grid):
    assert not triangles_fold(grid.vertices, grid.faces)
    assert triangles_fold(grid.vertices * np.array([-1.0, 1.0, 1.0]), grid.faces)


def test_a_refused_update_leaves_the_field_alone(grid, monkeypatch):
    """The flow of a velocity field cannot fold, so force the check to fire.

    The reference keeps the refused displacement in the field (module docstring, item 3).
    """
    import sbci.conseal as conseal

    warp = StationaryWarp(grid)  # before the fold test is faked
    monkeypatch.setattr(conseal, "triangles_fold", lambda vertices, faces: True)
    displacement = 0.01 * np.ones((grid.n_vertices, 2))
    before = warp.velocity.copy()
    assert not warp.compose(displacement)
    assert warp.rejected == 1
    assert np.array_equal(warp.velocity, before)
    assert np.allclose(warp.vertices, grid.vertices)


def test_inverting_undoes_the_warp_approximately(grid):
    rng = np.random.default_rng(4)
    warp = StationaryWarp(grid)
    warp.compose(0.05 * rng.normal(size=(grid.n_vertices, 2)))
    forward = warp.apply(grid.vertices)
    back = warp.copy().invert().apply(forward)
    moved = np.arccos(np.clip((forward * grid.vertices).sum(1), -1, 1))
    residual = np.arccos(np.clip((back * grid.vertices).sum(1), -1, 1))
    # the inverse flows the negated field and is interpolated on ico2 triangles: it
    # undoes most of the warp, not all of it
    assert residual.mean() < 0.3 * moved.mean()
    assert residual.max() < 0.1


def test_the_cotangent_laplacian_annihilates_constants(grid):
    laplacian = cotangent_laplacian(grid.vertices, grid.faces)
    assert np.allclose(laplacian @ np.ones(grid.n_vertices), 0.0, atol=1e-12)
    assert np.allclose((laplacian - laplacian.T).toarray(), 0.0, atol=1e-12)


# --- symmetries and the rotation schedule -------------------------------------------


def test_an_icosphere_has_sixty_symmetries_in_any_orientation():
    vertices, faces = icosphere(2)
    rotations, permutations = mesh_symmetries(vertices, faces)
    assert rotations.shape == (60, 3, 3) and permutations.shape == (60, vertices.shape[0])
    for rotation, perm in zip(rotations[:5], permutations[:5], strict=True):
        assert np.allclose(vertices[perm] @ rotation.T, vertices, atol=1e-8)
    tilt = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]])
    assert np.allclose(tilt @ tilt.T, np.eye(3))
    rotated, _ = mesh_symmetries(vertices @ tilt.T, faces)
    assert rotated.shape[0] == 60


def test_the_rotation_schedule_matches_the_reference_counts():
    shells = rotation_shells()
    assert [r.shape[0] for r, _ in shells] == [80, 20, 20, 20, 16]
    assert [k for _, k in shells] == [5, 3, 1, 1, 1]
    for rotations, _ in shells:
        assert np.allclose(np.einsum("nij,nkj->nik", rotations, rotations), np.eye(3), atol=1e-9)


# --- registration --------------------------------------------------------------------


def test_the_gradient_matches_a_finite_difference_of_the_cost():
    """Perturb the identity warp along the gradient; the cost must rise at the predicted rate."""
    # a finer mesh, many streamlines and a smooth difference between the subjects: the
    # gradient is derived in the continuum, for a transported density
    vertices, faces = icosphere(3)
    grid = SphericalGrid(vertices, faces, order=3)
    builder = HeatKernelBuilder(grid, grid, degree=12)
    k, dk = builder.compute(0.05)
    rng = np.random.default_rng(9)
    fixed = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=20000))
    moving = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=20000, jitter=0.15))
    engine = ConSEAL(grid, grid)
    q1 = fixed.q_transform(k)
    q2, q2_e1, q2_e2 = moving.q_transform(k, dk)
    difference = q1 - q2
    n = grid.n_vertices
    gradient_lh = engine._gradient(difference, q2, q2_e1, q2_e2, np.arange(n), grid)
    gradient_rh = engine._gradient(difference, q2, q2_e1, q2_e2, np.arange(n, 2 * n), grid)

    def tangent(field):
        return grid.e1 * field[:, :1] + grid.e2 * field[:, 1:]

    def cost_after(t):
        moved = moving.copy()
        lh = StationaryWarp(grid, viscosity=0.0)
        rh = StationaryWarp(grid, viscosity=0.0)
        lh.vertices = sphere_exp_map(grid.vertices, t * tangent(gradient_lh))
        rh.vertices = sphere_exp_map(grid.vertices, t * tangent(gradient_rh))
        moved.warp(lh, rh)
        return engine.cost(q1 - moved.q_transform(k))

    largest = max(
        np.linalg.norm(gradient_lh, axis=1).max(), np.linalg.norm(gradient_rh, axis=1).max()
    )
    eps = 1e-3 / largest
    numeric = (cost_after(eps) - cost_after(-eps)) / (2 * eps)
    # the gradient field is sum_k c_k b_k with c_k = dE/dbeta_k, and the basis is
    # orthonormal in the area inner product, so along that field the cost changes at
    # the rate sum_k c_k^2 = the area-weighted squared norm of the field. Moving
    # endpoints on a mesh is only approximately the transported-density model the
    # gradient is derived from, hence the loose factor.
    predicted = (grid.areas[:, None] * gradient_lh**2).sum()
    predicted += (grid.areas[:, None] * gradient_rh**2).sum()
    assert numeric > 0
    assert 0.5 < numeric / predicted < 2.0


def test_registering_a_connectome_to_itself_does_nothing(grid, connectome, kernel):
    k, dk = kernel
    engine = ConSEAL(grid, grid, max_iterations=3)
    lh, rh, costs, warped = engine.register(connectome, connectome, k, dk)
    # the cost is exactly zero, so no step is taken and the endpoints are not relocated:
    # carried through the identity, an endpoint on an ico2 triangle would move
    assert costs.tolist() == [0.0]
    np.testing.assert_array_equal(lh.vertices, grid.vertices)
    for name in LOCATIONS:
        np.testing.assert_array_equal(getattr(warped, name), getattr(connectome, name))


def test_registration_reduces_the_cost_of_a_warped_copy(grid, connectome, kernel):
    k, dk = kernel
    rng = np.random.default_rng(12)
    lh_true = StationaryWarp(grid)
    rh_true = StationaryWarp(grid)
    for _ in range(3):
        lh_true.compose(0.04 * rng.normal(size=(grid.n_vertices, 2)))
        rh_true.compose(0.04 * rng.normal(size=(grid.n_vertices, 2)))
    moved = connectome.copy()
    moved.warp(lh_true, rh_true)
    moved.commit()
    engine = ConSEAL(grid, grid, delta=0.05, max_iterations=15, threshold=1e-9)
    lh, rh, costs, warped = engine.register(connectome, moved, k, dk)
    assert costs[-1] < 0.7 * costs[0]
    assert np.all(np.diff(costs) <= 1e-12)
    assert (lh.jacobian > 0).all() and (rh.jacobian > 0).all()


def test_the_defaults_are_the_papers_settings_and_the_public_codes_are_constants(grid):
    """Since 7 October 2026 (PORTING.md item 19): no smoothing, no clamp, step 0.1, 1e-6.

    The public code's settings are the ``PUBLIC_*`` constants, given explicitly, and a
    setting given is kept.
    """
    import sbci.conseal as conseal

    def settings(engine):
        return (
            engine.delta,
            engine.max_iterations,
            engine.threshold,
            engine.step_clamp,
            engine.viscosity,
        )

    paper = ConSEAL(grid, grid)
    assert settings(paper) == (0.1, 1000, 1e-6, np.inf, 0.0)
    assert all(warp.viscosity == 0.0 for warp in paper.new_warps())
    public = ConSEAL(
        grid,
        grid,
        delta=conseal.PUBLIC_DELTA,
        max_iterations=conseal.PUBLIC_MAX_ITERATIONS,
        threshold=conseal.PUBLIC_THRESHOLD,
        step_clamp=conseal.PUBLIC_STEP_CLAMP,
        viscosity=conseal.PUBLIC_VISCOSITY,
        relative_threshold=conseal.PUBLIC_RELATIVE_THRESHOLD,
    )
    assert settings(public) == (0.05, 100, 1e-4, 0.2, 0.05)
    assert public.relative_threshold == 0.0
    assert all(warp.viscosity == 0.05 for warp in public.new_warps())
    assert settings(ConSEAL(grid, grid, threshold=1e-3)) == (0.1, 1000, 1e-3, np.inf, 0.0)


def test_the_default_settings_undo_more_of_a_known_warp_than_the_public_codes(
    grid, connectome, kernel
):
    """A smooth warp the basis can represent, moved back: the measure the defaults were chosen by.

    On ico4 the paper's settings undid 93% to 97% of three known warps and the
    public code's 52% to 75% (PORTING.md item 19); on this ico2 grid the defaults,
    capped at 100 steps, 79% against 2%, because 1e-4 stops the public descent after
    its first step. The public code has no floor under its threshold.
    """
    k, dk = kernel
    rng = np.random.default_rng(0)
    truth = []
    for _ in range(2):
        warp = StationaryWarp(grid, viscosity=0.0)
        field = (rng.normal(size=grid.basis.shape[1])[None, :, None] * grid.basis).sum(axis=1)
        warp.compose(field * (0.08 / np.linalg.norm(field, axis=1).max()))
        truth.append(warp)
    moved = connectome.copy()
    moved.warp(*truth)
    moved.commit()

    def degrees_off(warped):
        ends = zip(connectome.positions(), warped.positions(), strict=True)
        return np.mean([np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))) for a, b in ends])

    before = degrees_off(moved)
    public = dict(delta=0.05, max_iterations=100, threshold=1e-4, step_clamp=0.2, viscosity=0.05)
    public["relative_threshold"] = 0.0
    errors = {
        name: degrees_off(ConSEAL(grid, grid, **settings).register(connectome, moved, k, dk)[3])
        for name, settings in (("paper", {"max_iterations": 100}), ("public", public))
    }
    assert errors["paper"] < 0.35 * before
    assert errors["paper"] < 0.5 * errors["public"]


def test_the_rigid_search_recovers_a_rotation(grid, connectome, kernel):
    k, dk = kernel
    angle = np.radians(14.0)
    axis = np.array([0.2, 0.9, 0.4]) / np.linalg.norm([0.2, 0.9, 0.4])
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)
    p_in, p_out = connectome.positions()
    rotated = EndpointConnectome.from_points(
        grid,
        grid,
        p_in @ rotation.T,
        p_out @ rotation.T,
        connectome.hemisphere_in,
        connectome.hemisphere_out,
    )
    engine = ConSEAL(grid, grid, max_iterations=0)
    q1 = connectome.q_transform(k)
    before = engine.cost(q1 - rotated.q_transform(k))
    lh, rh, costs, warped = engine.register(connectome, rotated, k, dk, init_rotation=True)
    assert costs[0] < 0.25 * before
    # the search scores rotations on Q2 interpolated over ico2 triangles, so the
    # optimum it finds sits a few degrees from the true rotation
    recovered = np.arccos(np.clip((lh.vertices * (grid.vertices @ rotation)).sum(1), -1, 1))
    assert np.abs(recovered).max() < 0.12


def test_the_template_is_a_unit_vector_close_to_the_subjects(grid, kernel):
    k, _ = kernel
    rng = np.random.default_rng(21)
    subjects = [
        EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=1500)) for _ in range(3)
    ]
    template = ConSEAL(grid, grid).template(subjects, k, iterations=20)
    assert np.isclose((template**2).sum(), 1.0)
    distances = [np.arccos(np.clip((template * s.q_transform(k)).sum(), -1, 1)) for s in subjects]
    pairwise = np.arccos(
        np.clip((subjects[0].q_transform(k) * subjects[1].q_transform(k)).sum(), -1, 1)
    )
    assert max(distances) < pairwise


def test_endpoints_align_returns_one_warp_per_subject(grid, tmp_path):
    rng = np.random.default_rng(31)
    subjects = [
        EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=1200)) for _ in range(2)
    ]
    result = endpoints_align(
        subjects,
        sigma=0.05,
        kernel_degree=12,
        max_iterations=2,
        template_iterations=3,
        grids=(grid, grid),
    )
    assert len(result.warps) == 2 and len(result.costs) == 2 and len(result.connectomes) == 2
    assert all(len(c) >= 1 for c in result.costs)
    endpoints = result.aligned_endpoints(0)
    assert endpoints.has_positions and endpoints.n_streamlines == 1200
    path = result.warps[0].save(tmp_path / "warp.npz")
    loaded = EndpointWarp.load(path)
    assert np.array_equal(loaded.lh_vertices, result.warps[0].lh_vertices)
    assert np.array_equal(loaded.rh_velocity, result.warps[0].rh_velocity)


def test_endpoints_align_accepts_endpoint_objects_and_a_template_index(grid):
    rng = np.random.default_rng(41)
    a = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=900))
    b = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=900))
    result = endpoints_align(
        [a.to_endpoints(), b],
        template=0,
        sigma=0.05,
        kernel_degree=12,
        max_iterations=1,
        grids=(grid, grid),
    )
    assert np.allclose(
        result.template,
        a.q_transform(HeatKernelBuilder(grid, grid, 12).compute(0.05, derivative=False)),
    )
    assert result.costs[0][0] == 0.0


def test_endpoints_align_registers_one_subject_onto_another(grid):
    """template=<the fixed subject> is its Q-transform on the same grids; it need not be listed."""
    rng = np.random.default_rng(43)
    fixed = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=900))
    moving = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=900))
    settings = {"sigma": 0.05, "kernel_degree": 12, "max_iterations": 1, "grids": (grid, grid)}
    result = endpoints_align([moving], template=fixed, **settings)
    kernel = HeatKernelBuilder(grid, grid, 12).compute(0.05, derivative=False)
    np.testing.assert_allclose(result.template, fixed.q_transform(kernel))
    assert len(result.warps) == 1 and result.costs[0][0] > 0
    assert endpoints_align([fixed], template=fixed, **settings).costs[0][0] == 0.0
    as_endpoints = endpoints_align([moving], template=fixed.to_endpoints(), **settings)
    np.testing.assert_allclose(as_endpoints.template, result.template)

    class Bare:
        endpoints = None

    with pytest.raises(MissingDataError, match="template"):
        endpoints_align([moving], template=Bare(), **settings)
    with pytest.raises(ValueError, match="names no subject"):
        endpoints_align([moving], template=3, **settings)


def test_endpoints_align_refuses_connectomes_without_endpoints(grid):
    class Bare:
        """A connectome-like object that never stored its endpoints."""

        endpoints = None

    with pytest.raises(MissingDataError):
        endpoints_align([Bare()], grids=(grid, grid))


# --- metrics -------------------------------------------------------------------------


def test_the_overlap_metrics_are_one_for_identical_endpoints(connectome):
    assert overlap_coefficient(connectome, connectome, 1e-4) == 1.0
    assert dice_score(connectome, connectome, 1e-4) == 1.0
    assert np.isclose(endpoint_mass(connectome).sum(), 1.0)


def test_cotangent_weights_use_the_angle_opposite_the_edge():
    """A right triangle: the edge opposite the right angle gets no weight.

    The reference's layout gave it -0.5, the weight of an adjacent angle (item 12).
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    standard = cotangent_laplacian(vertices, faces).toarray()
    assert standard[1, 2] == pytest.approx(0.0)
    assert standard[0, 1] == pytest.approx(-0.5) and standard[0, 2] == pytest.approx(-0.5)
    reference = reference_conseal.cotangent_laplacian(vertices, faces).toarray()
    assert reference[1, 2] == pytest.approx(-0.5)
    for matrix in (standard, reference):
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(matrix @ np.ones(3), 0.0)


def test_a_half_turn_rotation_is_not_mistaken_for_the_identity(grid):
    """A half turn, whose skew part vanishes, is held as exactly as any other rotation.

    The reference read the axis off that skew part to hold the rotation as a velocity field
    (module docstring, item 17), so a half turn moved almost nothing.
    """
    half_turn = np.diag([-1.0, -1.0, 1.0])
    expected = grid.vertices @ half_turn.T
    np.testing.assert_allclose(
        StationaryWarp(grid).rotate(half_turn).vertices, expected, atol=1e-12
    )


def test_the_chain_rule_is_zero_where_the_density_was_clamped(grid, connectome):
    """Where F went negative and was clamped, sqrt(max(F, 0)) has derivative zero.

    The reference divides the nonzero dF there by 2e-15 (module docstring, item 13). The
    kernel has no negative entry inside its cutoff; an endpoint just outside its triangle
    has a slightly negative barycentric weight (to -0.0075 here), so 16 entries of the
    adjacency are negative (to -2.4e-7), and at bandwidth 0.005 and degree 30 F goes
    negative at two entries (-1.8e-12).
    """
    k, dk = HeatKernelBuilder(grid, grid, degree=30).compute(0.005)
    unclamped = k.T @ (k.T @ connectome.adjacency()).T  # K^T A K, A symmetric
    _, raw_e1, raw_e2 = connectome.evaluate(k, dk)
    _, d_e1, d_e2 = connectome.q_transform(k, dk)
    negative = unclamped < 0
    assert negative.any()
    assert (raw_e1[negative] != 0).any() or (raw_e2[negative] != 0).any()
    assert np.isfinite(d_e1).all() and np.isfinite(d_e2).all()
    assert not d_e1[negative].any() and not d_e2[negative].any()


def test_an_inward_mesh_is_refused_and_a_non_icosphere_has_no_symmetry_group(grid):
    mirrored = SphericalGrid(grid.vertices * np.array([-1.0, 1.0, 1.0]), grid.faces, order=2)
    with pytest.raises(ValueError, match="oriented"):
        StationaryWarp(mirrored)
    rng = np.random.default_rng(3)
    jittered = grid.vertices + 0.01 * rng.normal(size=grid.vertices.shape)
    jittered /= np.linalg.norm(jittered, axis=1, keepdims=True)
    with pytest.raises(ValueError, match="symmetries"):
        mesh_symmetries(jittered, grid.faces)


def test_the_bundled_grid_is_oriented_outward_and_icosahedral_to_stored_precision():
    """FreeSurfer's ico4 sphere is an icosphere to 1.7e-4: its coordinates carry so many digits.

    The 60 rotations are found, they permute the vertices at the default
    tolerance, and an exact-arithmetic tolerance refuses the mesh rather than
    pretend.
    """
    from sbci.conseal import default_grids, icosahedral_rotations

    lh, rh = default_grids(order=1)
    for hemisphere in (lh, rh):
        assert not triangles_fold(hemisphere.vertices, hemisphere.faces)
        rotations, deviation = icosahedral_rotations(hemisphere.vertices, hemisphere.faces)
        assert rotations.shape == (60, 3, 3)
        assert 1e-5 < deviation < 5e-4
        _, permutations = mesh_symmetries(hemisphere.vertices, hemisphere.faces)
        assert permutations.shape == (60, hemisphere.n_vertices)
        with pytest.raises(ValueError, match="symmetries"):
            mesh_symmetries(hemisphere.vertices, hemisphere.faces, tolerance=1e-6)


def test_the_rigid_search_transport_is_the_permutation_on_an_exact_icosphere(grid):
    """Pulling back by a symmetry through barycentric interpolation is the reference's gather."""
    from sbci.conseal import ConSEAL

    rotations, permutations = mesh_symmetries(grid.vertices, grid.faces)
    rng = np.random.default_rng(5)
    values = rng.random((grid.n_vertices, grid.n_vertices))
    query = MeshQuery(grid.vertices, grid.faces)
    for rotation, perm in zip(rotations[:7], permutations[:7], strict=True):
        image = np.argsort(perm)  # image[i] = where vertex i lands
        pulled = ConSEAL._pull_back(values, grid, query, rotation)
        np.testing.assert_allclose(pulled, values[np.ix_(image, image)], atol=1e-9)


def test_endpoints_align_refuses_a_misshapen_template(grid, connectome):
    with pytest.raises(ValueError, match="template"):
        endpoints_align(
            [connectome], template=np.ones(3), sigma=0.05, kernel_degree=12, grids=(grid, grid)
        )


# --- registering from where the endpoints are, and the shared basis switch --------


def _furthest_move(before, after):
    """Largest angle any endpoint of ``before`` travelled to its place in ``after``."""
    return max(
        np.arccos(np.clip((p * q).sum(1), -1, 1)).max()
        for p, q in zip(before.positions(), after.positions(), strict=True)
    )


def test_registering_an_already_warped_connectome_starts_where_it_is(grid, connectome, kernel):
    """A connectome warped before -- one an alignment returned, say -- is not snapped back.

    The fresh warps act on the committed locations. Before the fix the copy
    ``register`` works on kept the locations the object was built with, so
    registering a warped connectome onto itself reported a cost of zero while
    moving every endpoint back by the whole warp. Registered onto itself it now
    takes no step at all, so the step that shows it is taken onto another.
    """
    k, dk = kernel
    engine = ConSEAL(grid, grid, max_iterations=1)
    lh_warp, rh_warp = StationaryWarp(grid, viscosity=0.0), StationaryWarp(grid, viscosity=0.0)
    field = 0.2 * grid.basis[:, 0, :]
    assert lh_warp.compose(field) and rh_warp.compose(field)
    warped = connectome.copy()
    warped.warp(lh_warp, rh_warp)
    carried = np.arccos(np.clip((lh_warp.vertices * grid.vertices).sum(1), -1, 1)).max()
    assert carried > 0.05  # snapped back, the endpoints would move this far

    _, _, costs, registered = engine.register(warped, warped, k, dk)
    assert costs.tolist() == [0.0]
    for name in LOCATIONS:
        np.testing.assert_array_equal(getattr(registered, name), getattr(warped, name))

    # one step onto another subject: from where the endpoints are, not from where they were built
    lh, rh, costs, registered = engine.register(connectome, warped, k, dk)
    assert len(costs) == 2
    onward = warped.copy()
    onward.commit()
    onward.warp(lh, rh)
    snapped = warped.copy()
    snapped.warp(lh, rh)  # from the locations the object was built with
    for name in LOCATIONS:
        np.testing.assert_array_equal(getattr(registered, name), getattr(onward, name))
    assert _furthest_move(registered, snapped) > 0.5 * carried


def test_registering_a_fresh_connectome_is_unchanged_by_the_commit(
    grid, connectome, kernel, monkeypatch
):
    """For a connectome that was never warped the commit is a no-op, bit for bit."""
    k, dk = kernel
    rng = np.random.default_rng(12)
    moving = EndpointConnectome.from_points(grid, grid, *synthetic(rng, jitter=0.15))
    engine = ConSEAL(grid, grid, max_iterations=2)
    lh, rh, costs, warped = engine.register(connectome, moving, k, dk)
    assert len(costs) > 1
    monkeypatch.setattr(EndpointConnectome, "commit", lambda self: None)  # the code before the fix
    lh_before, rh_before, costs_before, warped_before = engine.register(connectome, moving, k, dk)
    assert costs.tolist() == costs_before.tolist()
    np.testing.assert_array_equal(lh.vertices, lh_before.vertices)
    np.testing.assert_array_equal(rh.velocity, rh_before.velocity)
    for name in ("weights_in", "index_in", "weights_out", "index_out"):
        np.testing.assert_array_equal(getattr(warped, name), getattr(warped_before, name))


def test_the_default_grids_are_built_only_when_none_are_given(grid, monkeypatch):
    """Grids passed in are used as they are; without them the default grids are built once."""
    import sbci.conseal as conseal

    asked = []

    def small_grids(order):
        asked.append(order)
        return grid, grid

    monkeypatch.setattr(conseal, "default_grids", small_grids)
    rng = np.random.default_rng(41)
    subject = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=600))
    settings = {"template": 0, "sigma": 0.05, "kernel_degree": 12, "max_iterations": 0}
    endpoints_align([subject], **settings)
    endpoints_align([subject], grids=(grid, grid), **settings)
    assert asked == [conseal.DEFAULT_WARP_ORDER]


# --- the second review: the median's arithmetic, smoothing as vectors, save's path -----


class _Density:
    """Stands in for a connectome whose square-root density is given."""

    def __init__(self, q):
        self.q = q

    def q_transform(self, kernel):
        return self.q.copy()


def regular_simplex(rng, side_degrees, count=5, n=40):
    """``count`` unit matrices, every pair ``side_degrees`` apart.

    Their Karcher median is their normalized mean, equidistant from all of
    them, so a template that stops short shows as a spread of distances.
    """
    base = rng.random((n, n)) + 0.5
    base /= np.sqrt((base**2).sum())
    directions = []
    for _ in range(count):
        u = rng.normal(size=(n, n))
        u -= (u * base).sum() * base
        for d in directions:
            u -= (u * d).sum() * d
        directions.append(u / np.sqrt((u**2).sum()))
    half = np.arccos(np.sqrt(np.cos(np.radians(side_degrees))))
    return [np.cos(half) * base + np.sin(half) * u for u in directions]


def degrees_between(template, densities):
    """Fisher-Rao distance from ``template`` to each density in degrees, norms ignored."""
    return np.degrees(
        [np.arccos(np.clip((q * template).sum() / np.sqrt((q**2).sum()), -1, 1)) for q in densities]
    )


def test_the_template_does_not_stop_on_a_subject_whose_norm_rounds_low(grid):
    """Item 15: a squared norm 5e-11 short of one once pinned the median to that subject."""
    qs = regular_simplex(np.random.default_rng(3), side_degrees=5.3)
    assert np.degrees(np.arccos((qs[0] * qs[1]).sum())) == pytest.approx(5.3, abs=1e-6)
    for factor in (1 - 5e-11, 1.0, 1 + 5e-11):
        subjects = [_Density(np.sqrt(factor) * q) for q in qs]
        template = ConSEAL(grid, grid).template(subjects, None)
        distances = degrees_between(template, qs)
        assert np.isclose((template**2).sum(), 1.0)
        assert distances.min() > 2.5  # between the subjects, not on one
        assert distances.max() - distances.min() < 0.5
    # the reference's arithmetic: the starting subject's distance to itself is
    # acos(1 - 5e-11), a weight near 1e5, and the median stays put
    short = [np.sqrt(1 - 5e-11) * q for q in qs]
    mean = sum(short) / len(short)
    start = int(np.argmin([((q - mean) ** 2).sum() for q in short]))
    distances = degrees_between(reference_conseal.template(short), qs)
    assert distances[start] < 0.01
    assert np.sort(distances)[1] > 5.0


def test_the_template_counts_a_subject_on_the_estimate_as_coincident(grid):
    """Normalized, this cohort's starting subject has a squared norm 2.2e-16 short of one.

    That reads as 2e-8 radians from itself, and without the 1e-6 guard the median lands on
    it; with the guard it lies between the subjects (item 15).
    """
    qs = regular_simplex(np.random.default_rng(8), side_degrees=5.3)
    template = ConSEAL(grid, grid).template([_Density(q) for q in qs], None)
    distances = degrees_between(template, qs)
    assert distances.min() > 2.5
    assert distances.max() - distances.min() < 0.5


def test_endpoints_align_estimates_a_unit_norm_template_between_the_subjects(grid):
    rng = np.random.default_rng(23)
    subjects = [
        EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=900)) for _ in range(3)
    ]
    result = endpoints_align(
        subjects,
        sigma=0.05,
        kernel_degree=12,
        max_iterations=0,
        template_iterations=10,
        grids=(grid, grid),
    )
    assert np.isclose((result.template**2).sum(), 1.0)
    kernel = HeatKernelBuilder(grid, grid, 12).compute(0.05, derivative=False)
    distances = degrees_between(result.template, [s.q_transform(kernel) for s in subjects])
    assert distances.min() > 0.5


@pytest.fixture(scope="module")
def bundled_left():
    """The bundled ico4 left grid, which has a vertex at each coordinate pole."""
    from sbci.conseal import default_grids

    return default_grids(order=1)[0]


def rotational_velocity(warp, rotation):
    """The ``(e1, e2)`` components of the field whose unit-time flow is ``rotation``."""
    from scipy.spatial.transform import Rotation

    omega = Rotation.from_matrix(rotation).as_rotvec()
    field = np.cross(np.broadcast_to(omega, warp.base.shape), warp.base)
    return np.stack([(field * warp.e1).sum(1), (field * warp.e2).sum(1)], axis=1)


def rotation_about_x(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )


def componentwise_smoothing(warp, velocity):
    """The ``(e1, e2)`` components smoothed as two scalars, as the reference does (item 16).

    With the port's cotangent weights, so that only the frame differs from ``warp._smooth``;
    ``reference_smoothing`` has the reference's weights too (item 12).
    """
    return velocity - warp.viscosity * (warp._laplacian @ velocity)


def reference_smoothing(warp):
    """The reference's smoothing of ``warp``'s field, its cotangent layout and all."""
    layout = reference_conseal.cotangent_laplacian(warp.base, warp.faces)
    return lambda velocity: reference_conseal.smooth(layout, velocity, warp.viscosity)


def projected_smoothing(warp, velocity):
    """The smoothing of the second review: 3-vectors per Cartesian component, projected back."""
    lifted = warp.e1 * velocity[:, :1] + warp.e2 * velocity[:, 1:]
    lifted = lifted - warp.viscosity * (warp._laplacian @ lifted)
    return np.stack([(lifted * warp.e1).sum(1), (lifted * warp.e2).sum(1)], axis=1)


def test_smoothing_leaves_a_rotation_alone_at_the_coordinate_poles(bundled_left):
    """Item 16: ten of the reference's smoothing passes move the polar vertices 8.4 and 8.5 degrees.

    And their neighbours up to 11.5, where vertices more than 20 degrees from a
    pole move 0.14 on average. What remains with the connection Laplacian is the
    smoothing itself: it maps a rotation's field to itself scaled by the vertex
    area, so ten passes of 5% shrink the rotation by a quarter of a percent
    everywhere alike (half a percent with the projected 3-vectors of the second
    review).
    """
    grid = bundled_left
    poles = np.flatnonzero(np.abs(grid.vertices[:, 2]) > 1 - 1e-9)
    assert poles.size == 2
    drift = {}
    for name in ("port", "reference"):
        # the rotation's own field, as the reference holds a rotation; rotate() itself
        # keeps the rotation out of the field (item 17)
        warp = StationaryWarp(grid)
        warp.velocity = rotational_velocity(warp, rotation_about_x(20.0))
        warp.vertices = warp.exponential()
        start = warp.vertices.copy()
        smooth = reference_smoothing(warp) if name == "reference" else None
        for _ in range(10):
            if smooth is None:
                assert warp.compose(np.zeros((grid.n_vertices, 2)))
            else:
                warp.velocity = smooth(warp.velocity)
                warp.vertices = warp.exponential()
        drift[name] = np.degrees(np.arccos(np.clip((warp.vertices * start).sum(1), -1, 1)))
    assert drift["port"][poles].max() < 0.06  # 0.046; the projected 3-vectors gave 0.092
    assert drift["port"][poles].max() < 1.5 * drift["port"].mean()
    assert drift["reference"][poles].min() > 8.0
    assert drift["reference"].max() > 11.0
    far = np.abs(grid.vertices[:, 2]) < np.cos(np.radians(20.0))
    assert drift["reference"][far].mean() < 0.15


def test_both_smoothings_agree_where_the_frame_turns_slowly(bundled_left):
    """Within 10 degrees of the coordinate equator the frame is nearly parallel: 1e-4 apart.

    The projected 3-vectors of the second review differed there by their
    curvature term, 5% of the field weighted by the vertex area (2.7e-4).
    """
    grid = bundled_left
    warp = StationaryWarp(grid)
    velocity = rotational_velocity(warp, rotation_about_x(20.0))
    smoothed = warp._smooth(velocity)
    componentwise = componentwise_smoothing(warp, velocity)
    scale = np.linalg.norm(velocity, axis=1).max()
    difference = np.linalg.norm(smoothed - componentwise, axis=1) / scale
    projected = np.linalg.norm(projected_smoothing(warp, velocity) - componentwise, axis=1) / scale
    z = np.abs(grid.vertices[:, 2])
    equator = z < np.sin(np.radians(10.0))
    assert difference[equator].max() < 1e-4  # 8.3e-5
    assert projected[equator].max() > 2e-4
    # further out the reference's own frame error grows to the size of that term
    assert difference[z < 0.5].max() < 4e-4  # 2.7e-4
    # 5% of the Laplacian barely touches a degree-1 field; the poles are where they part
    assert np.linalg.norm(smoothed - velocity, axis=1).max() < 1e-3 * scale
    assert difference[z > 0.99].max() > 0.05


def test_smoothing_damps_a_rotation_at_the_same_rate_about_any_axis(bundled_left):
    """The connection Laplacian maps a rotation's field to itself scaled by the vertex area.

    So 100 passes of 5% keep exp(-5 * 4 pi / 2562) = 97.6% of a rotation
    about any axis. The projected 3-vectors of the second review add the
    field once more, the curvature term the third review found, and damp it
    twice as fast (95.2%); the reference's component-wise smoothing keeps
    98.9% about the coordinate z axis but 90.5% about a tilted one (item 16).
    """
    from scipy.spatial.transform import Rotation

    grid = bundled_left
    warp = StationaryWarp(grid)
    smoothings = {
        "connection": warp._smooth,
        "projected": lambda velocity: projected_smoothing(warp, velocity),
        "reference": reference_smoothing(warp),
    }
    kept = {}
    for axis, direction in (("z", [0.0, 0.0, 1.0]), ("tilted", [0.2, 0.9, 0.4])):
        omega = np.radians(20.0) * np.asarray(direction) / np.linalg.norm(direction)
        velocity = rotational_velocity(warp, Rotation.from_rotvec(omega).as_matrix())
        for name, smooth in smoothings.items():
            smoothed = velocity
            for _ in range(100):
                smoothed = smooth(smoothed)
            kept[axis, name] = (smoothed * velocity).sum() / (velocity**2).sum()
    assert kept["z", "connection"] == pytest.approx(kept["tilted", "connection"], rel=5e-3)
    assert kept["z", "connection"] == pytest.approx(0.976, abs=1e-3)
    assert kept["z", "projected"] == pytest.approx(kept["z", "connection"] ** 2, abs=1e-3)
    assert kept["z", "reference"] == pytest.approx(0.989, abs=1e-3)
    assert kept["tilted", "reference"] == pytest.approx(0.905, abs=1e-3)


def test_save_returns_the_path_it_wrote(grid, tmp_path):
    warp = StationaryWarp(grid)
    exported = EndpointWarp(
        lh_vertices=warp.vertices,
        lh_velocity=warp.velocity,
        lh_jacobian=warp.jacobian,
        rh_vertices=warp.vertices,
        rh_velocity=warp.velocity,
        rh_jacobian=warp.jacobian,
    )
    written = exported.save(tmp_path / "warp")  # np.savez_compressed appends the suffix
    assert written == tmp_path / "warp.npz" and written.exists()
    assert not (tmp_path / "warp").exists()
    assert exported.save(tmp_path / "named.npz") == tmp_path / "named.npz"
    loaded = EndpointWarp.load(written)
    np.testing.assert_array_equal(loaded.lh_jacobian, exported.lh_jacobian)


# --- rigid rotations, held outside the velocity field (module docstring, item 17) ---


def degrees_off(points, expected):
    """Angle between corresponding unit vectors, in degrees."""
    return np.degrees(np.arccos(np.clip((points * expected).sum(1), -1, 1)))


@pytest.mark.parametrize("degrees", [20.0, 150.0, 180.0])
def test_rotate_holds_the_rotation_exactly_and_steps_do_not_erode_it(bundled_left, degrees):
    """The reference's flow of a rotation was 1.6 degrees off at 150, and each step shrank it.

    ``tests/reference/rotation_probe.py`` measures the reference's; this checks the port's.
    """
    grid = bundled_left
    rotation = rotation_about_x(degrees)
    expected = grid.vertices @ rotation.T
    warp = StationaryWarp(grid).rotate(rotation)
    np.testing.assert_allclose(warp.vertices, expected, atol=1e-12)
    np.testing.assert_allclose(warp.rigid, rotation, atol=1e-12)
    assert not warp.velocity.any()
    np.testing.assert_allclose(warp.jacobian, 1.0, atol=1e-9)
    for _ in range(10):
        assert warp.compose(np.zeros((grid.n_vertices, 2)))
    np.testing.assert_allclose(warp.vertices, expected, atol=1e-12)


def test_a_deformation_after_a_rotation_is_its_flow_after_the_rotation(bundled_left):
    """A 5-degree rotation composed after a 150-degree one lands on their product."""
    from scipy.spatial.transform import Rotation

    grid = bundled_left
    large = rotation_about_x(150.0)
    small = Rotation.from_rotvec(np.radians(5.0) * np.array([0.0, 0.6, 0.8])).as_matrix()
    warp = StationaryWarp(grid, viscosity=0.0).rotate(large)
    assert warp.compose(rotational_velocity(warp, small))
    np.testing.assert_allclose(warp.rigid, large, atol=1e-12)  # the step went into the field
    # 0.007 degrees at most, of which 0.003 is the flow of the small field itself
    assert degrees_off(warp.vertices, grid.vertices @ (small @ large).T).max() < 0.02


def test_inverting_a_rotated_warp_inverts_its_rigid_part(bundled_left):
    from scipy.spatial.transform import Rotation

    grid = bundled_left
    large = rotation_about_x(150.0)
    small = Rotation.from_rotvec(np.radians(5.0) * np.array([0.0, 0.6, 0.8])).as_matrix()
    warp = StationaryWarp(grid, viscosity=0.0).rotate(large)
    assert warp.compose(rotational_velocity(warp, small))
    inverse = warp.copy().invert()
    np.testing.assert_allclose(inverse.rigid, large.T, atol=1e-12)
    np.testing.assert_allclose(warp.rigid, large, atol=1e-12)  # the copy was inverted
    assert degrees_off(inverse.apply(warp.vertices), grid.vertices).max() < 0.02


def test_inverting_twice_gives_back_the_warp_bit_for_bit(bundled_left):
    """The inverse holds the same parts in the other order, so nothing is interpolated.

    Until 6 October 2026 the field was carried to the rotated frame by
    interpolation, which changed it by 0.1% over a double inversion on ico4
    and 0.4% on ico3 (2.8% in the fourth review's case).
    """
    from scipy.spatial.transform import Rotation

    grid = bundled_left
    warp = StationaryWarp(grid, viscosity=0.0).rotate(rotation_about_x(150.0))
    small = Rotation.from_rotvec(np.radians(5.0) * np.array([0.0, 0.6, 0.8])).as_matrix()
    assert warp.compose(rotational_velocity(warp, small))
    twice = warp.copy().invert().invert()
    np.testing.assert_array_equal(twice.velocity, warp.velocity)
    np.testing.assert_array_equal(twice.rigid, warp.rigid)
    np.testing.assert_array_equal(twice.vertices, warp.vertices)
    once = warp.copy().invert()  # the flow of -v, then R': computed, not interpolated
    expected = warp.exponential(-warp.velocity) @ warp.rigid
    np.testing.assert_allclose(once.vertices, expected / np.linalg.norm(expected, axis=1)[:, None])


def test_assigning_a_rotation_to_rigid_moves_the_warp(bundled_left):
    """``warp.rigid = R`` keeps the field and re-flows; it used to change nothing."""
    from scipy.spatial.transform import Rotation

    grid = bundled_left
    turn = rotation_about_x(30.0)
    assigned = StationaryWarp(grid, viscosity=0.0)
    assigned.rigid = turn
    np.testing.assert_allclose(
        assigned.vertices, StationaryWarp(grid, viscosity=0.0).rotate(turn).vertices, atol=1e-12
    )
    small = Rotation.from_rotvec(np.radians(3.0) * np.array([0.0, 1.0, 0.0])).as_matrix()
    deformed = StationaryWarp(grid, viscosity=0.0)
    assert deformed.compose(rotational_velocity(deformed, small))
    field = deformed.velocity.copy()
    deformed.rigid = turn
    np.testing.assert_array_equal(deformed.velocity, field)  # kept, unlike rotate()
    assert degrees_off(deformed.vertices, grid.vertices @ (small @ turn).T).max() < 0.02
    with pytest.raises(ValueError, match="3 x 3 rotation"):
        deformed.rigid = 2 * np.eye(3)
    with pytest.raises(ValueError, match="3 x 3 rotation"):
        deformed.rigid = np.diag([1.0, 1.0, -1.0])  # a reflection
    with pytest.raises(ValueError):
        deformed.rigid[0, 0] = 2.0  # read-only: an edit in place would change nothing


def test_the_rigid_search_leaves_its_rotation_out_of_the_field(grid, connectome, kernel):
    k, dk = kernel
    rotation = rotation_about_x(14.0)
    p_in, p_out = connectome.positions()
    rotated = EndpointConnectome.from_points(
        grid,
        grid,
        p_in @ rotation.T,
        p_out @ rotation.T,
        connectome.hemisphere_in,
        connectome.hemisphere_out,
    )
    engine = ConSEAL(grid, grid, max_iterations=0)
    lh, rh, _, _ = engine.register(connectome, rotated, k, dk, init_rotation=True)
    for warp in (lh, rh):
        assert not warp.velocity.any()
        assert not np.allclose(warp.rigid, np.eye(3))
        np.testing.assert_allclose(warp.vertices, grid.vertices @ warp.rigid.T, atol=1e-12)


def test_an_endpoint_warp_keeps_its_rigid_part_through_save_and_load(grid, tmp_path):
    lh = StationaryWarp(grid).rotate(rotation_about_x(30.0))
    exported = EndpointWarp(
        lh.vertices, lh.velocity, lh.jacobian, lh.vertices, lh.velocity, lh.jacobian, lh.rigid
    )
    loaded = EndpointWarp.load(exported.save(tmp_path / "rigid"))
    np.testing.assert_array_equal(loaded.lh_rigid, lh.rigid)
    np.testing.assert_array_equal(loaded.rh_rigid, np.eye(3))
    # a file written before the rigid part was kept loads with identities
    older = {k: v for k, v in exported.__dict__.items() if not k.endswith("_rigid")}
    np.savez_compressed(tmp_path / "older.npz", **older)
    np.testing.assert_array_equal(EndpointWarp.load(tmp_path / "older.npz").lh_rigid, np.eye(3))


# --- the third review: an empty subject, and the template's memory -----------------


def test_a_subject_without_streamlines_is_refused_by_its_index(
    grid, connectome, kernel, monkeypatch
):
    """Its density is zero everywhere: normalizing it was 0/0, and the cohort came back NaN."""
    k, dk = kernel
    nothing = np.zeros((0, 3))
    empty = EndpointConnectome.from_points(grid, grid, nothing, nothing, [], [])
    assert not empty.q_transform(k).any()
    with pytest.raises(ValueError, match="subject 1 has no streamlines"):
        ConSEAL(grid, grid).template([connectome, empty], k)
    engine = ConSEAL(grid, grid, max_iterations=1)
    with pytest.raises(ValueError, match="moving connectome has no streamlines"):
        engine.register(connectome, empty, k, dk)
    with pytest.raises(ValueError, match="fixed connectome has no streamlines"):
        engine.register(empty, connectome, k, dk)
    with pytest.raises(ValueError, match="fixed connectome has no streamlines"):
        engine.register(np.zeros((2 * grid.n_vertices,) * 2), connectome, k, dk)

    def no_kernel(*args, **kwargs):
        raise AssertionError("the kernel was built before the empty subject was refused")

    monkeypatch.setattr(HeatKernelBuilder, "compute", no_kernel)
    settings = {"sigma": 0.05, "kernel_degree": 12, "grids": (grid, grid)}
    with pytest.raises(ValueError, match="subject 1 has no streamlines"):
        endpoints_align([connectome, empty], **settings)
    with pytest.raises(ValueError, match="template connectome has no streamlines"):
        endpoints_align([connectome], template=empty, **settings)


def test_the_template_holds_the_cohort_once(grid):
    """Normalizing into a second list held every density twice: 21 GB for 50 ico4 subjects.

    Each fresh density is now divided in place, which gives the same values,
    so the median is that of the normalized densities. The peak is one list
    and a few working arrays (1.22 lists here; 2.00 before).
    """
    import tracemalloc

    rng = np.random.default_rng(8)
    densities = [rng.random((200, 200)) + 0.5 for _ in range(32)]
    one_list = sum(q.nbytes for q in densities)
    engine = ConSEAL(grid, grid)
    tracemalloc.start()
    try:
        template = engine.template([_Density(q) for q in densities], None, iterations=5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert one_list < peak < 1.5 * one_list
    normalized = [_Density(q / np.sqrt((q**2).sum())) for q in densities]
    np.testing.assert_allclose(template, engine.template(normalized, None, iterations=5))


# --- the eighth review ------------------------------------------------------------


def test_an_inverted_warp_says_its_rotation_comes_after_the_flow(bundled_left, tmp_path):
    """The order of a warp's two parts was a private flag that nothing exported or saved."""
    from sbci.conseal import EndpointWarp

    warp = StationaryWarp(bundled_left, viscosity=0.0).rotate(rotation_about_x(40.0))
    assert not warp.rigid_after
    inverse = warp.copy().invert()
    assert inverse.rigid_after and not inverse.copy().invert().rigid_after
    vertices = np.asarray(inverse.vertices)
    saved = EndpointWarp(
        lh_vertices=vertices,
        lh_velocity=inverse.velocity,
        lh_jacobian=np.ones(len(vertices)),
        rh_vertices=vertices,
        rh_velocity=inverse.velocity,
        rh_jacobian=np.ones(len(vertices)),
        lh_rigid=np.asarray(inverse.rigid),
        lh_rigid_after=inverse.rigid_after,
    ).save(tmp_path / "warp.npz")
    loaded = EndpointWarp.load(saved)
    assert loaded.lh_rigid_after is True and loaded.rh_rigid_after is False


def test_area_weighting_keeps_the_cost_on_the_plain_sums_scale(grid, connectome, kernel):
    """The areas weigh in units of their mean, so the default threshold still means something.

    Raw Voronoi areas made the weighted cost about 40,000 times smaller on ico4
    (170 times on this ico2 grid), below the stopping threshold: a weighted
    registration stopped before its first step and moved nothing.
    """
    k, dk = kernel
    rng = np.random.default_rng(12)
    lh_true, rh_true = StationaryWarp(grid), StationaryWarp(grid)
    for _ in range(3):
        lh_true.compose(0.04 * rng.normal(size=(grid.n_vertices, 2)))
        rh_true.compose(0.04 * rng.normal(size=(grid.n_vertices, 2)))
    moved = connectome.copy()
    moved.warp(lh_true, rh_true)
    moved.commit()
    plain = ConSEAL(grid, grid, delta=0.05, max_iterations=15)
    weighted = ConSEAL(grid, grid, delta=0.05, max_iterations=15, area_weighted=True)
    difference = connectome.q_transform(k) - moved.q_transform(k)
    assert 0.8 < weighted.cost(difference) / plain.cost(difference) < 1.25
    _, _, costs, _ = weighted.register(connectome, moved, k, dk)
    assert len(costs) > 3 and costs[-1] < 0.8 * costs[0]


def test_the_rigid_search_undoes_a_turn_about_z(grid, connectome, kernel, monkeypatch):
    """The reference's caps turn about axes perpendicular to z, never about z itself.

    So an 8-degree turn about z is left 9.6 and 8.2 degrees off on this grid; each
    shell's best are now turned about z as well.
    """
    import sbci.conseal as conseal

    angle = np.radians(8.0)
    truth = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    p_in, p_out = connectome.positions()
    rotated = EndpointConnectome.from_points(
        grid,
        grid,
        p_in @ truth.T,
        p_out @ truth.T,
        connectome.hemisphere_in,
        connectome.hemisphere_out,
    )

    def degrees_left(rotation):
        return np.degrees(np.arccos(np.clip((np.trace(rotation @ truth) - 1) / 2, -1, 1)))

    k, dk = kernel
    lh, rh, _, _ = ConSEAL(grid, grid, max_iterations=0).register(
        connectome, rotated, k, dk, init_rotation=True
    )
    assert degrees_left(lh.rigid) < 1.5 and degrees_left(rh.rigid) < 1.5
    # the reference's search: no turns about z after any shell
    shells = len(conseal.rotation_shells())
    monkeypatch.setattr(conseal, "spin_shells", lambda: [np.eye(3)[None]] * shells)
    lh, rh, _, _ = ConSEAL(grid, grid, max_iterations=0).register(
        connectome, rotated, k, dk, init_rotation=True
    )
    assert degrees_left(lh.rigid) > 5.0 and degrees_left(rh.rigid) > 5.0


# --- the review of 11 October 2026 -------------------------------------------------


def test_the_gradient_matches_the_cost_along_every_basis_field():
    """Each coefficient of the gradient against a finite difference of the cost along its field.

    The rate along the gradient (the test above) is checked only to a factor of two,
    and it passed with the divergence term dropped or doubled, a slot's factor of two
    missing, or e1 and e2 swapped: on this grid their cosines with these differences
    are 0.81, 0.92, 0.92 and 0.50, where the gradient's own is 0.980.
    """
    vertices, faces = icosphere(2)
    grid = SphericalGrid(vertices, faces, order=2)
    k, dk = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    rng = np.random.default_rng(9)
    fixed = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=8000))
    moving = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=8000, jitter=0.15))
    engine = ConSEAL(grid, grid)
    q1 = fixed.q_transform(k)
    q2, q2_e1, q2_e2 = moving.q_transform(k, dk)
    n = grid.n_vertices
    fields = [
        engine._gradient(q1 - q2, q2, q2_e1, q2_e2, rows, grid)
        for rows in (np.arange(n), np.arange(n, 2 * n))
    ]
    # the basis is orthonormal in the area inner product: a field's coefficients
    analytic = np.concatenate(
        [np.einsum("i,ikc,ic->k", grid.areas, grid.basis, field) for field in fields]
    )

    def cost_after(field_lh, field_rh):
        moved = moving.copy()
        lh, rh = StationaryWarp(grid, viscosity=0.0), StationaryWarp(grid, viscosity=0.0)
        lh.vertices = sphere_exp_map(
            grid.vertices, grid.e1 * field_lh[:, :1] + grid.e2 * field_lh[:, 1:]
        )
        rh.vertices = sphere_exp_map(
            grid.vertices, grid.e1 * field_rh[:, :1] + grid.e2 * field_rh[:, 1:]
        )
        moved.warp(lh, rh)
        return engine.cost(q1 - moved.q_transform(k))

    zero, h = np.zeros((n, 2)), 2e-3
    numeric = []
    for side in (0, 1):
        for index in range(grid.basis.shape[1]):
            step = grid.basis[:, index, :] * h
            pair = [(s * step, zero) if side == 0 else (zero, s * step) for s in (1, -1)]
            numeric.append((cost_after(*pair[0]) - cost_after(*pair[1])) / (2 * h))
    numeric = np.array(numeric)
    cosine = analytic @ numeric / np.linalg.norm(analytic) / np.linalg.norm(numeric)
    assert cosine > 0.95


def test_a_template_that_is_not_finite_is_refused(grid, connectome, kernel):
    """One NaN in the template ran every iteration on NaN, and exported a NaN velocity."""
    k, dk = kernel
    template = connectome.q_transform(k)
    template[3, 5] = template[5, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ConSEAL(grid, grid, max_iterations=8).register(template, connectome, k, dk)
    with pytest.raises(ValueError, match="non-finite"):
        endpoints_align(
            [connectome], template=template, sigma=0.05, kernel_degree=12, grids=(grid, grid)
        )


def test_points_that_are_not_on_the_sphere_are_refused(grid):
    """A NaN endpoint, or one at the origin, was put on a triangle as if it were a point."""
    rng = np.random.default_rng(7)
    p_in, p_out, h_in, h_out = synthetic(rng, n=30)
    for bad in (np.nan, 0.0):
        broken = p_in.copy()
        broken[4] = [bad, 0.0, 0.0] if np.isnan(bad) else [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="not finite or are at the origin"):
            EndpointConnectome.from_points(grid, grid, broken, p_out, h_in, h_out)


def test_endpoints_from_another_grid_are_refused():
    """An ico3 file's triangle indices all fit the ico4 grid, and named other triangles there."""
    lh, rh = default_grids(order=1)
    rng = np.random.default_rng(0)
    m, per_hemi = 50, 642
    ends = Endpoints(
        surf_in=np.zeros(m, np.int8),
        surf_out=np.ones(m, np.int8),
        vtx_in=rng.integers(0, per_hemi, m),
        vtx_out=rng.integers(0, per_hemi, m),
        n_per_hemi=per_hemi,
        tri_in=rng.integers(0, 1280, m),
        tri_out=rng.integers(0, 1280, m),
        bary_in=rng.dirichlet([1, 1, 1], m),
        bary_out=rng.dirichlet([1, 1, 1], m),
    )
    with pytest.raises(ValueError, match="642 vertices per hemisphere"):
        EndpointConnectome.from_endpoints(ends, lh, rh)


def test_a_rolled_back_step_is_not_reported_and_its_refusals_still_count(
    grid, connectome, kernel, monkeypatch
):
    """The callback was told the cost of a step then undone, and its refusal was forgotten.

    And undoing the first step carried the endpoints through the identity, which moves them,
    rather than back to where they started.
    """
    import sbci.conseal as conseal

    k, dk = kernel
    seen = []
    moved = known_deformation(grid, connectome)
    real_fold, real_cost = conseal.triangles_fold, ConSEAL.cost
    calls = {"fold": 0, "cost": 0}

    def refuse_the_first_left_step(vertices, faces):
        calls["fold"] += 1  # calls 1 and 2 are the constructors' orientation checks
        return True if calls["fold"] == 3 else real_fold(vertices, faces)

    def raised_by_the_first_step(self, difference):
        calls["cost"] += 1  # call 1 is the starting cost, call 2 the first step's
        return real_cost(self, difference) + (1.0 if calls["cost"] == 2 else 0.0)

    monkeypatch.setattr(conseal, "triangles_fold", refuse_the_first_left_step)
    monkeypatch.setattr(ConSEAL, "cost", raised_by_the_first_step)
    lh, rh, costs, warped = ConSEAL(grid, grid, max_iterations=5).register(
        connectome, moved, k, dk, callback=lambda i, c: seen.append((i, c))
    )
    assert len(costs) == 1 and seen == []  # the step was taken, raised the cost, and went
    assert (lh.rejected, rh.rejected) == (1, 0)  # the left half was refused all the same
    np.testing.assert_array_equal(rh.vertices, grid.vertices)
    assert not rh.velocity.any()
    # the endpoints are where they started, exactly; relocated through the identity they
    # would have moved
    for name in LOCATIONS:
        np.testing.assert_array_equal(getattr(warped, name), getattr(moved, name))
    through_identity = moved.copy()
    through_identity.warp(StationaryWarp(grid), StationaryWarp(grid))
    assert not np.array_equal(through_identity.weights_in, moved.weights_in)


def known_deformation(grid, connectome):
    """``connectome`` carried by a smooth warp of 0.2 radians, committed there."""
    rng = np.random.default_rng(12)
    truth = StationaryWarp(grid, viscosity=0.0), StationaryWarp(grid, viscosity=0.0)
    for warp in truth:
        field = (rng.normal(size=grid.basis.shape[1])[None, :, None] * grid.basis).sum(axis=1)
        warp.compose(field * (0.2 / np.linalg.norm(field, axis=1).max()))
    moved = connectome.copy()
    moved.warp(*truth)
    moved.commit()
    return moved


def test_a_step_refused_on_both_sides_ends_the_registration(grid, connectome, kernel, monkeypatch):
    """At threshold 0 the same refused pair of steps was taken again until max_iterations.

    It ends at once, before the endpoints are carried anywhere.
    """
    import sbci.conseal as conseal

    k, dk = kernel
    real = conseal.triangles_fold
    calls = {"fold": 0, "warp": 0}

    def refuse_every_step(vertices, faces):
        calls["fold"] += 1  # calls 1 and 2 are the constructors' orientation checks
        return True if calls["fold"] > 2 else real(vertices, faces)

    carry = EndpointConnectome.warp

    def counted(self, *warps):
        calls["warp"] += 1
        return carry(self, *warps)

    moved = known_deformation(grid, connectome)
    monkeypatch.setattr(conseal, "triangles_fold", refuse_every_step)
    monkeypatch.setattr(EndpointConnectome, "warp", counted)
    lh, rh, costs, _ = ConSEAL(grid, grid, threshold=0.0, max_iterations=50).register(
        connectome, moved, k, dk
    )
    assert len(costs) == 1 and (lh.rejected, rh.rejected) == (1, 1)
    assert calls["warp"] == 0


def test_a_subject_registered_onto_itself_takes_no_step(grid):
    """A zero starting cost leaves nothing to descend.

    It made the relative threshold zero, and a step that changed nothing then counted as
    progress: endpoints on their triangles' corners, which the identity leaves where they
    are, ran every iteration up to max_iterations.
    """
    rng = np.random.default_rng(0)
    m, faces = 2000, grid.faces.shape[0]
    corner = np.tile([1.0, 0.0, 0.0], (m, 1))
    ends = Endpoints(
        surf_in=rng.integers(0, 2, m).astype(np.int8),
        surf_out=rng.integers(0, 2, m).astype(np.int8),
        vtx_in=np.zeros(m, int),
        vtx_out=np.zeros(m, int),
        n_per_hemi=grid.n_vertices,
        tri_in=rng.integers(0, faces, m),
        tri_out=rng.integers(0, faces, m),
        bary_in=corner,
        bary_out=corner.copy(),
    )
    subject = EndpointConnectome.from_endpoints(ends, grid, grid)
    k, dk = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    for floor in (0.0, 1e-5):
        seen = []
        lh, rh, costs, warped = ConSEAL(
            grid, grid, max_iterations=40, relative_threshold=floor
        ).register(subject, subject, k, dk, callback=lambda i, c, s=seen: s.append(i))
        assert costs.tolist() == [0.0] and seen == []
        np.testing.assert_array_equal(warped.weights_in, subject.weights_in)
    result = endpoints_align(
        [subject, subject.copy()],
        template=0,
        sigma=0.05,
        kernel_degree=12,
        max_iterations=40,
        grids=(grid, grid),
    )
    assert [len(trace) for trace in result.costs] == [1, 1]


def test_what_cannot_be_registered_is_refused(grid, connectome, kernel):
    """A misshapen fixed density broadcast silently, and a NaN weight gave a NaN cost."""
    k, dk = kernel
    engine = ConSEAL(grid, grid, max_iterations=1)
    q = connectome.q_transform(k)
    with pytest.raises(ValueError, match="where the moving connectome's grids need"):
        engine.register(q[:, 0], connectome, k, dk)
    with pytest.raises(ValueError, match="where the moving connectome's grids need"):
        engine.register(q[:-1, :-1], connectome, k, dk)
    ends = connectome.to_endpoints()
    ends.bary_in[5] = np.nan
    with pytest.raises(ValueError, match="1 endpoint"):
        EndpointConnectome.from_endpoints(ends, grid, grid)
    broken = k.copy()
    broken.data[0] = np.nan
    with pytest.raises(ValueError, match="starting cost is not finite"):
        engine.register(q, connectome, broken, dk)


def test_composing_onto_an_inverted_warp_is_refused(grid):
    """The displacement went into the flow, which acts before the rotation, in the wrong frame."""
    from scipy.spatial.transform import Rotation

    warp = StationaryWarp(grid, viscosity=0.0)
    warp.rotate(Rotation.from_euler("y", 120, degrees=True).as_matrix())
    warp.invert()
    with pytest.raises(ValueError, match="compose before inverting"):
        warp.compose(np.zeros((grid.n_vertices, 2)))


# --- the defaults of 11 October 2026: the paper's kernel width, a floor under the threshold ---


def test_the_kernel_is_narrowed_to_the_papers_width_and_the_threshold_has_a_floor(
    grid, connectome, monkeypatch
):
    """The fork's kernel, at its 0.005, is as narrow as this one at 0.0027 (PORTING.md item 19)."""
    import sbci.conseal as conseal

    assert (conseal.DEFAULT_SIGMA, conseal.PUBLIC_SIGMA) == (0.0027, 0.005)
    builder = HeatKernelBuilder(grid, grid, conseal.DEFAULT_KERNEL_DEGREE)
    cut = [np.degrees(np.arccos(builder.cutoff(s))) for s in (0.0027, 0.005)]
    assert 12.0 < cut[0] < 12.6 and 21.0 < cut[1] < 22.0  # the fork's own is cut at 12.0
    assert ConSEAL(grid, grid).relative_threshold == 1e-5
    assert ConSEAL(grid, grid, relative_threshold=0.0).relative_threshold == 0.0
    with pytest.raises(ValueError, match="relative_threshold"):
        ConSEAL(grid, grid, relative_threshold=-1.0)

    seen = []

    class Built(Exception):
        pass

    def compute(self, sigma, derivative=True):
        seen.append(sigma)
        raise Built

    monkeypatch.setattr(HeatKernelBuilder, "compute", compute)
    with pytest.raises(Built):
        endpoints_align([connectome, connectome.copy()], grids=(grid, grid))
    assert seen == [0.0027]


def test_the_floor_runs_a_small_warp_on_past_the_absolute_threshold(grid, connectome, kernel):
    """A small warp's whole cost is small, so 1e-6 alone stopped it after a few steps."""
    k, dk = kernel
    rng = np.random.default_rng(1)
    truth = []
    for _ in range(2):
        warp = StationaryWarp(grid, viscosity=0.0)
        field = (rng.normal(size=grid.basis.shape[1])[None, :, None] * grid.basis).sum(axis=1)
        warp.compose(field * (0.02 / np.linalg.norm(field, axis=1).max()))
        truth.append(warp)
    moved = connectome.copy()
    moved.warp(*truth)
    moved.commit()

    def degrees_off(warped):
        ends = zip(connectome.positions(), warped.positions(), strict=True)
        return np.mean([np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))) for a, b in ends])

    runs = {
        floor: ConSEAL(grid, grid, max_iterations=100, relative_threshold=floor).register(
            connectome, moved, k, dk
        )
        for floor in (0.0, 1e-5)
    }
    plain, floored = runs[0.0], runs[1e-5]
    assert 1e-5 * plain[2][0] < 1e-6  # the starting cost is small enough for the floor to bind
    assert len(plain[2]) < 30 < len(floored[2])
    assert degrees_off(floored[3]) < 0.8 * degrees_off(plain[3])


# --- what register returns, however it ends, and what it refuses (PORTING.md item 24) ---


def registered(engine, fixed, moving, kernel, monkeypatch, raised=(), **options):
    """``engine.register`` with the listed calls of the cost raised by 1, and its callbacks.

    Call 1 is the starting cost and call 2 the first step's; with ``init_rotation``, calls 1
    and 2 are the rigid search's, before and after its turn, and the descent's follow.
    """
    k, dk = kernel
    real, calls, seen = ConSEAL.cost, [0], []

    def cost(self, difference):
        calls[0] += 1
        return real(self, difference) + (1.0 if calls[0] in raised else 0.0)

    monkeypatch.setattr(ConSEAL, "cost", cost)
    result = engine.register(
        fixed, moving, k, dk, callback=lambda i, c: seen.append((i, c)), **options
    )
    monkeypatch.setattr(ConSEAL, "cost", real)
    return result, seen


def assert_consistent(engine, fixed, moving, kernel, result, seen):
    """What ``register`` returned agrees with itself.

    The connectome is the moving one carried through the warps returned, or the moving one
    itself when those are identities; the last cost is that connectome's; and the callback
    was told of every step kept, once, and of nothing else.
    """
    lh, rh, costs, warped = result
    expected = moving.copy()
    if any(w.velocity.any() or not np.array_equal(w.rigid, np.eye(3)) for w in (lh, rh)):
        expected.commit()
        expected.warp(lh, rh)
    for name in LOCATIONS:
        np.testing.assert_array_equal(getattr(warped, name), getattr(expected, name))
    k, _ = kernel
    assert costs[-1] == pytest.approx(engine.cost(fixed.q_transform(k) - warped.q_transform(k)))
    assert seen == list(zip(range(1, len(costs)), costs[1:].tolist(), strict=True))


def turned(connectome, degrees=14.0):
    """``connectome`` turned rigidly about an oblique axis."""
    from scipy.spatial.transform import Rotation

    axis = np.array([0.2, 0.9, 0.4])
    rotation = Rotation.from_rotvec(np.radians(degrees) * axis / np.linalg.norm(axis))
    p_in, p_out = (rotation.apply(points) for points in connectome.positions())
    return EndpointConnectome.from_points(
        connectome.lh_grid,
        connectome.rh_grid,
        p_in,
        p_out,
        connectome.hemisphere_in,
        connectome.hemisphere_out,
    )


def test_a_later_step_undone_leaves_the_endpoints_where_the_kept_steps_put_them(
    grid, connectome, kernel, monkeypatch
):
    """Undone after steps were kept, a step is undone by carrying the endpoints again.

    Resetting them, or keeping them where the undone step put them, returned endpoints
    the warps did not describe.
    """
    moved = known_deformation(grid, connectome)
    engine = ConSEAL(grid, grid, max_iterations=10)
    result, seen = registered(engine, connectome, moved, kernel, monkeypatch, raised=(5,))
    assert len(result[2]) == 4  # three steps kept; the fourth raised the cost and was undone
    assert_consistent(engine, connectome, moved, kernel, result, seen)


def test_a_turn_kept_survives_an_undone_first_step(grid, connectome, kernel, monkeypatch):
    """The rigid search moves the endpoints, so undoing the first step after it re-carries them."""
    moving = turned(connectome)
    engine = ConSEAL(grid, grid, max_iterations=5)
    result, seen = registered(
        engine, connectome, moving, kernel, monkeypatch, raised=(4,), init_rotation=True
    )
    lh, rh, costs, _ = result
    assert not np.array_equal(lh.rigid, np.eye(3)) and len(costs) == 1
    assert_consistent(engine, connectome, moving, kernel, result, seen)


def test_a_turn_that_raises_the_cost_is_refused_and_leaves_the_endpoints_alone(
    grid, connectome, kernel, monkeypatch
):
    moving = turned(connectome)
    engine = ConSEAL(grid, grid, max_iterations=0)
    result, seen = registered(
        engine, connectome, moving, kernel, monkeypatch, raised=(2,), init_rotation=True
    )
    lh, rh, costs, _ = result
    assert np.array_equal(lh.rigid, np.eye(3)) and np.array_equal(rh.rigid, np.eye(3))
    assert_consistent(engine, connectome, moving, kernel, result, seen)


def test_a_step_that_changes_nothing_ends_the_descent_at_threshold_zero(
    grid, connectome, kernel, monkeypatch
):
    """Without a floor and at threshold 0, a fall of 0 met the threshold and ran on."""
    real, steps = ConSEAL._step, [0]

    def nothing_after_the_first(self, gradient):
        steps[0] += 1  # one call a hemisphere an iteration
        step = real(self, gradient)
        return step if steps[0] <= 2 else np.zeros_like(step)

    monkeypatch.setattr(ConSEAL, "_step", nothing_after_the_first)
    moved = known_deformation(grid, connectome)
    engine = ConSEAL(grid, grid, max_iterations=10, threshold=0.0, relative_threshold=0.0)
    result, seen = registered(engine, connectome, moved, kernel, monkeypatch)
    costs = result[2]
    assert len(costs) == 3 and costs[2] == costs[1]
    assert_consistent(engine, connectome, moved, kernel, result, seen)


def test_the_last_step_kept_is_reported_when_it_falls_below_the_threshold(
    grid, connectome, kernel, monkeypatch
):
    moved = known_deformation(grid, connectome)
    engine = ConSEAL(grid, grid, max_iterations=200, threshold=1e-3, relative_threshold=0.0)
    result, seen = registered(engine, connectome, moved, kernel, monkeypatch)
    costs = result[2]
    assert 2 < len(costs) < 50 and 0 < costs[-2] - costs[-1] < 1e-3
    assert_consistent(engine, connectome, moved, kernel, result, seen)


def test_the_descent_stops_below_the_smaller_of_the_threshold_and_the_floor(
    grid, connectome, kernel
):
    """A floor of half the starting cost stops nothing while the threshold is 0."""
    k, dk = kernel
    moved = known_deformation(grid, connectome)
    for threshold, floor, iterations in ((0.0, 0.5, 5), (0.5, 0.0, 1)):
        engine = ConSEAL(
            grid, grid, max_iterations=5, threshold=threshold, relative_threshold=floor
        )
        assert len(engine.register(connectome, moved, k, dk)[2]) - 1 == iterations


def test_the_alignment_counts_the_steps_refused_on_both_hemispheres(grid, connectome, monkeypatch):
    import sbci.conseal as conseal

    moved = known_deformation(grid, connectome)
    real, calls = conseal.triangles_fold, [0]

    def refuse_the_right(vertices, faces):
        calls[0] += 1  # 1 and 2 are the orientation checks; then left, right, left, ...
        return calls[0] > 2 and calls[0] % 2 == 0 or real(vertices, faces)

    monkeypatch.setattr(conseal, "triangles_fold", refuse_the_right)
    result = endpoints_align(
        [moved],
        template=connectome,
        sigma=0.05,
        kernel_degree=12,
        max_iterations=3,
        grids=(grid, grid),
    )
    assert result.rejected_steps == [3]


def test_the_clamp_caps_the_largest_step_at_its_value(grid):
    engine = ConSEAL(grid, grid, delta=0.5, step_clamp=0.2)
    gradient = np.random.default_rng(0).normal(size=(grid.n_vertices, 2))
    largest = np.linalg.norm(gradient, axis=1).max()
    np.testing.assert_allclose(engine._step(gradient), -0.5 * 0.2 * gradient / largest)
    small = 0.1 * gradient / largest  # under the clamp, only scaled by -delta
    np.testing.assert_allclose(engine._step(small), -0.5 * small)


def test_a_verbose_registration_reports_as_it_goes(grid, connectome, kernel, capsys):
    k, dk = kernel
    moved = known_deformation(grid, connectome)
    engine = ConSEAL(grid, grid, max_iterations=10, threshold=0.0, relative_threshold=0.0)
    costs = engine.register(connectome, moved, k, dk, verbose=True)[2]
    out = capsys.readouterr().out
    assert len(costs) == 11
    assert "initial cost" in out and "iteration 10: cost" in out
    assert "final cost" in out and "after 10 iterations" in out


def test_to_endpoints_names_the_vertex_nearest_each_endpoint(grid):
    rng = np.random.default_rng(3)
    picked = rng.choice(grid.n_vertices, 60, replace=False)
    nudged = sphere_exp_map(grid.vertices[picked], 0.02 * rng.normal(size=(60, 3)))
    hemisphere = np.zeros(60, dtype=int)
    connectome = EndpointConnectome.from_points(grid, grid, nudged, nudged, hemisphere, hemisphere)
    ends = connectome.to_endpoints()
    np.testing.assert_array_equal(ends.vtx_in, picked)
    np.testing.assert_array_equal(ends.vtx_out, picked)


def test_the_jacobian_is_the_warped_area_over_the_unwarped(grid):
    """A flow toward +z gathers the vertices there and spreads them about -z."""
    warp = StationaryWarp(grid, viscosity=0.0)
    toward_north = np.array([0.0, 0.0, 1.0]) - grid.vertices[:, 2:3] * grid.vertices
    assert warp.compose(np.stack([(toward_north * e).sum(1) for e in (grid.e1, grid.e2)], 1) * 0.3)
    z = grid.vertices[:, 2]
    assert (warp.jacobian[z > 0.9] < 0.7).all() and (warp.jacobian[z < -0.9] > 1.4).all()


def test_both_routes_of_evaluate_give_the_same_density_and_derivatives(
    grid, connectome, kernel, monkeypatch
):
    """Few streamlines take sparse products, many dense ones: a README subject takes the first."""
    import sbci.conseal as conseal

    k, dk = kernel
    routes = []
    for fraction in (np.inf, 0.0):  # every connectome counts as few, then none does
        monkeypatch.setattr(conseal, "SPARSE_PRODUCT_FRACTION", fraction)
        routes.append(connectome.evaluate(k, dk))
    for sparse_route, dense_route in zip(*routes, strict=True):
        np.testing.assert_allclose(
            sparse_route, dense_route, rtol=0, atol=1e-12 * np.abs(dense_route).max()
        )


def test_the_area_weighted_gradient_matches_the_cost_on_a_grid_of_uneven_areas():
    """On ico2 the areas vary by a tenth, too little to tell the weights' places apart.

    Pulled toward one pole, the grid's areas vary eightfold, and dropping the weights of
    the gradient's columns or of its rows takes the cosine below to 0.92 or 0.94.
    """
    from sbci.alignment import normalize_rows

    vertices, faces = icosphere(2)
    grid = SphericalGrid(normalize_rows(vertices + [0.0, 0.0, 0.5]), faces, order=2)
    assert grid.areas.max() > 8 * grid.areas.min()
    k, dk = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    rng = np.random.default_rng(9)
    fixed = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=8000))
    moving = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=8000, jitter=0.15))
    engine = ConSEAL(grid, grid, area_weighted=True)
    q1 = fixed.q_transform(k)
    q2, q2_e1, q2_e2 = moving.q_transform(k, dk)
    n = grid.n_vertices
    analytic = np.concatenate(
        [
            np.einsum(
                "i,ikc,ic->k",
                grid.areas,
                grid.basis,
                engine._gradient(q1 - q2, q2, q2_e1, q2_e2, rows, grid),
            )
            for rows in (np.arange(n), np.arange(n, 2 * n))
        ]
    )

    def cost_after(side, field):
        moved = moving.copy()
        warps = [StationaryWarp(grid, viscosity=0.0) for _ in range(2)]
        warps[side].vertices = sphere_exp_map(
            grid.vertices, grid.e1 * field[:, :1] + grid.e2 * field[:, 1:]
        )
        moved.warp(*warps)
        return engine.cost(q1 - moved.q_transform(k))

    h = 2e-3
    numeric = np.array(
        [
            (cost_after(side, h * grid.basis[:, i]) - cost_after(side, -h * grid.basis[:, i]))
            / (2 * h)
            for side in (0, 1)
            for i in range(grid.basis.shape[1])
        ]
    )
    cosine = analytic @ numeric / np.linalg.norm(analytic) / np.linalg.norm(numeric)
    assert cosine > 0.96  # 0.976 here


def test_a_fixed_array_must_be_a_unit_square_root_density(grid, connectome, kernel):
    """A density passed where its square root belongs registered silently, and badly."""
    k, dk = kernel
    q = connectome.q_transform(k)
    moving = known_deformation(grid, connectome)
    engine = ConSEAL(grid, grid, max_iterations=1)
    engine.register(q, moving, k, dk)  # its squares sum to 1, to rounding
    upper = np.triu(q)
    for bad, match in (
        (q**2, "sum to"),
        (-q, "nonnegative"),
        (upper / np.sqrt((upper**2).sum()), "not symmetric"),
    ):
        with pytest.raises(ValueError, match=match):
            engine.register(bad, moving, k, dk)
    with pytest.raises(ValueError, match="sum to"):
        endpoints_align([moving], template=q**2, sigma=0.05, kernel_degree=12, grids=(grid, grid))
    # with two subjects True would have been subject 1, registered onto silently
    with pytest.raises(ValueError, match="give a subject's index"):
        endpoints_align(
            [moving, connectome],
            template=True,
            sigma=0.05,
            kernel_degree=12,
            grids=(grid, grid),
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "setting",
    [
        {"delta": 0.0},
        {"delta": -0.1},
        {"delta": np.nan},
        {"max_iterations": -3},
        {"max_iterations": 2.5},
        {"threshold": -1.0},
        {"threshold": np.nan},
        {"step_clamp": 0.0},
        {"step_clamp": np.nan},
        {"viscosity": -0.05},
        {"viscosity": np.nan},
        {"relative_threshold": np.nan},
    ],
)
def test_settings_that_cannot_register_are_refused(grid, setting):
    """Each returned identity warps, or stopped after one step, as though it had converged."""
    (name,) = setting
    with pytest.raises(ValueError, match=name):
        ConSEAL(grid, grid, **setting)


def test_settings_at_their_limits_are_taken(grid):
    engine = ConSEAL(grid, grid, max_iterations=0, threshold=0.0, step_clamp=np.inf, viscosity=0)
    assert engine.max_iterations == 0 and engine.step_clamp == np.inf


@pytest.mark.parametrize("flag", [256, 257, 0.5, 1.9, -1])
def test_hemisphere_flags_are_checked_before_the_cast(grid, flag):
    """int8 wrapped 256 to 0 and truncated 1.9 to 1, so both were taken for a hemisphere."""
    p_in, p_out, h_in, h_out = synthetic(np.random.default_rng(1), n=30)
    h_in = h_in.astype(float)
    h_in[4] = flag
    with pytest.raises(ValueError, match="hemisphere_in"):
        EndpointConnectome.from_points(grid, grid, p_in, p_out, h_in, h_out)


def test_a_bandwidth_is_not_chosen_for_a_connectome_without_streamlines(grid):
    empty = EndpointConnectome.from_points(grid, grid, np.zeros((0, 3)), np.zeros((0, 3)), [], [])
    with pytest.raises(ValueError, match="no streamlines"):
        HeatKernelBuilder(grid, grid, degree=12).cross_validate(empty, [0.02, 0.05])


def test_a_negative_number_of_template_iterations_is_refused(grid, connectome):
    """It returned the subject nearest the mean, as though no step were needed."""
    with pytest.raises(ValueError, match="template_iterations"):
        endpoints_align(
            [connectome, connectome.copy()],
            template_iterations=-1,
            sigma=0.05,
            kernel_degree=12,
            grids=(grid, grid),
        )


# --- the fourth review of 12 October 2026 (PORTING.md item 27) --------------------------

SMALL = {"sigma": 0.05, "kernel_degree": 12, "max_iterations": 3}


def made(grid, seed, n=600):
    return EndpointConnectome.from_points(grid, grid, *synthetic(np.random.default_rng(seed), n=n))


def test_a_connectome_built_on_other_grids_is_refused(grid, kernel):
    """As many vertices, elsewhere: read by index, its endpoints moved 79 degrees.

    That under a step that moved no vertex. The same grids built again are taken.
    """
    vertices, faces = icosphere(2)
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = SphericalGrid(vertices @ turn.T, faces, order=3)
    elsewhere, here = made(turned, 4), made(grid, 5)
    with pytest.raises(ValueError, match="connectome 0 was built on other grids"):
        endpoints_align([elsewhere, here], grids=(grid, grid), **SMALL)
    with pytest.raises(ValueError, match="template connectome was built on other grids"):
        endpoints_align([here], template=elsewhere, grids=(grid, grid), **SMALL)
    k, dk = kernel
    engine = ConSEAL(grid, grid, max_iterations=0)
    for fixed, moving in ((here, elsewhere), (elsewhere, here)):
        with pytest.raises(ValueError, match="built on other grids"):
            engine.register(fixed, moving, k, dk)
    with pytest.raises(ValueError, match="subject 1 was built on other grids"):
        engine.template([here, elsewhere], k)
    again = SphericalGrid(vertices, faces, order=1)  # new objects, the same mesh
    engine.register(here, made(again, 6), k, dk)
    endpoints_align([made(again, 6), here], grids=(grid, grid), **SMALL)


def test_one_subject_does_not_estimate_a_template(grid):
    """align() refused this; endpoints_align returned identity warps and a cost of 0."""
    with pytest.raises(ValueError, match="at least two connectomes"):
        endpoints_align([made(grid, 1)], grids=(grid, grid), **SMALL)


@pytest.mark.parametrize("sigma", [-0.05, 0.0, np.nan, np.inf])
def test_a_bandwidth_that_makes_no_heat_kernel_is_refused_first(grid, sigma, monkeypatch):
    """-0.05 kept each vertex alone, the identity kernel, so nothing moved.

    NaN failed only after the kernel and the template were built, blamed on the template.
    """
    import sbci.conseal as conseal

    with pytest.raises(ValueError, match="sigma"):
        HeatKernelBuilder(grid, grid, degree=12).compute(sigma)

    def built(*args, **kwargs):
        raise AssertionError("the kernel was built first")

    monkeypatch.setattr(conseal, "HeatKernelBuilder", built)
    with pytest.raises(ValueError, match="sigma"):
        endpoints_align(
            [made(grid, 1), made(grid, 2)], grids=(grid, grid), **{**SMALL, "sigma": sigma}
        )


@pytest.mark.parametrize("degree", [0, 2.5, -3, np.nan])
def test_a_kernel_degree_that_is_no_degree_is_refused(grid, degree):
    """0 made a constant kernel and every density uniform; 2.5 ran as 2; -3 an IndexError."""
    with pytest.raises(ValueError, match="degree"):
        HeatKernelBuilder(grid, grid, degree=degree)
    with pytest.raises(ValueError, match="kernel_degree"):
        endpoints_align(
            [made(grid, 1), made(grid, 2)], grids=(grid, grid), **{**SMALL, "kernel_degree": degree}
        )


@pytest.mark.parametrize("iterations", [-1, 2.5, np.nan])
def test_the_template_refuses_a_count_of_steps_that_is_not_one(grid, kernel, iterations):
    """endpoints_align refused these since item 24; ConSEAL.template took them."""
    k, _ = kernel
    with pytest.raises(ValueError, match="iterations"):
        ConSEAL(grid, grid).template([made(grid, s) for s in (1, 2, 3)], k, iterations)


def test_the_template_counts_steps_given_as_whole_floats(grid, kernel):
    k, _ = kernel
    subjects = [made(grid, s) for s in (1, 2, 3)]
    engine = ConSEAL(grid, grid)
    np.testing.assert_array_equal(
        engine.template(subjects, k, 2.0), engine.template(subjects, k, 2)
    )


def test_a_generator_of_connectomes_is_read_one_at_a_time(grid):
    """At 03f4668 a generator was read lazily; listed first, it held every subject at once."""
    import tracemalloc

    kernel = HeatKernelBuilder(grid, grid, degree=12).compute(0.05, derivative=False)
    points = [synthetic(np.random.default_rng(i), n=30000, jitter=0.15) for i in range(8)]
    one = EndpointConnectome.from_points(grid, grid, *points[0])
    endpoints = sum(
        getattr(one, name).nbytes
        for name in ("weights_in", "weights_out", "index_in", "index_out", "face_in", "face_out")
    )

    def peak(count):
        tracemalloc.start()
        lazily = (EndpointConnectome.from_points(grid, grid, *p) for p in points[:count])
        ConSEAL(grid, grid).template(lazily, kernel, iterations=1)
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    # seven more densities, and one subject's endpoints in hand while the next is read;
    # listed, seven subjects' endpoints more, with their committed copies
    assert peak(8) - peak(1) < 7 * one.n_vertices**2 * 8 + 4 * endpoints


def test_the_median_starts_from_the_subject_nearest_the_mean():
    """The second of these four; starting from the first or the farthest passed every test."""
    vertices, faces = icosphere(1)
    sphere = SphericalGrid(vertices, faces, order=1)
    engine = ConSEAL(sphere, sphere)
    qs = four_densities()
    mean = sum(qs) / len(qs)
    nearest = int(np.argmin([((q - mean) ** 2).sum() for q in qs]))
    assert nearest not in (0, int(np.argmax([((q - mean) ** 2).sum() for q in qs])))
    start = engine.template([_Density(q) for q in qs], None, iterations=0)
    np.testing.assert_array_equal(start, qs[nearest])


def four_densities():
    rng = np.random.default_rng(2)
    base = rng.random((30, 30)) + 0.5
    qs = []
    for spread in (0.6, 0.1, 0.35, 0.8):  # the second sits nearest the mean
        q = base + spread * rng.random((30, 30))
        qs.append(q / np.sqrt((q**2).sum()))
    return qs


def test_a_weiszfeld_step_is_a_fifth_of_the_mean_direction():
    """One step against Weiszfeld's formula; a step of a third passed every test."""
    vertices, faces = icosphere(1)
    sphere = SphericalGrid(vertices, faces, order=1)
    engine = ConSEAL(sphere, sphere)
    qs = four_densities()
    start = engine.template([_Density(q) for q in qs], None, iterations=0)
    one = engine.template([_Density(q) for q in qs], None, iterations=1)
    directions, weights = np.zeros_like(start), 0.0
    for q in qs:
        distance = np.arccos(np.clip((q * start).sum(), -1, 1))
        if distance > 1e-6:
            directions += (q - np.cos(distance) * start) / np.sin(distance)
            weights += 1 / distance
    mean = directions / weights
    norm = np.sqrt((mean**2).sum())
    expected = np.cos(0.2 * norm) * start + np.sin(0.2 * norm) * mean / norm
    np.testing.assert_allclose(one, expected / np.sqrt((expected**2).sum()), atol=1e-14)


def test_the_rigid_search_undoes_a_turn_about_z_the_other_way(grid, connectome, kernel):
    """At -8 degrees too: a search that spun one way only passed the test at +8."""
    angle = np.radians(-8.0)
    truth = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    p_in, p_out = connectome.positions()
    rotated = EndpointConnectome.from_points(
        grid,
        grid,
        p_in @ truth.T,
        p_out @ truth.T,
        connectome.hemisphere_in,
        connectome.hemisphere_out,
    )
    k, dk = kernel
    lh, rh, _, _ = ConSEAL(grid, grid, max_iterations=0).register(
        connectome, rotated, k, dk, init_rotation=True
    )
    for rigid in (lh.rigid, rh.rigid):
        assert np.degrees(np.arccos(np.clip((np.trace(rigid @ truth) - 1) / 2, -1, 1))) < 1.5


def test_the_cutoff_is_the_reference_scan(grid):
    """The last positive scan point up to the first where K(x) K(1) falls below epsilon.

    Here against Legendre series of NumPy's own, at both bandwidths; one scan point
    further out passed every test without MATLAB.
    """
    import sbci.conseal as conseal

    builder = HeatKernelBuilder(grid, grid, conseal.DEFAULT_KERNEL_DEGREE)
    scan = np.linspace(1.0, -1.0, 5000)
    for sigma, degrees in ((conseal.DEFAULT_SIGMA, 12.37), (conseal.PUBLIC_SIGMA, 21.50)):
        order = np.arange(conseal.DEFAULT_KERNEL_DEGREE + 1)
        weights = (2 * order + 1) * np.exp(-order * (order + 1) * sigma)
        product = np.polynomial.legendre.legval(scan, weights) * weights.sum()
        first = int(np.flatnonzero(product < conseal.KERNEL_EPSILON)[0])
        last = int(np.flatnonzero(product[: first + 1] > 0)[-1])
        assert builder.cutoff(sigma) == scan[last]
        assert np.degrees(np.arccos(scan[last])) == pytest.approx(degrees, abs=0.01)


def test_endpoints_are_carried_in_one_order_of_addition(grid, connectome):
    """(w0 v0 + w1 v1) + w2 v2, normalized by squares added alike, on any processor.

    NumPy's einsum fuses the products on some processors, where the CPU's points were
    then not the GPU's.
    """
    from sbci.conseal import _points

    w, i = connectome.weights_in, connectome.index_in
    vertices = 1.01 * np.vstack([grid.vertices, grid.vertices])  # off the sphere, to normalize
    corners = vertices[i]
    points = w[:, 0, None] * corners[:, 0] + w[:, 1, None] * corners[:, 1]
    points += w[:, 2, None] * corners[:, 2]
    squares = points * points
    norms = np.sqrt((squares[:, 0] + squares[:, 1]) + squares[:, 2])[:, None]
    np.testing.assert_array_equal(_points(w, i, vertices), points / norms)


def test_the_cpu_takes_the_kernels_derivative_as_it_is_given(connectome, kernel, monkeypatch):
    """A CSC derivative was made CSR first, and summed in another order than before the GPU."""
    from scipy import sparse

    import sbci.conseal as conseal

    k, dk = kernel
    csc = conseal.KernelDerivative(*(sparse.csc_matrix(part) for part in (dk.x, dk.y, dk.z)))
    converted = []
    original = conseal._as_sparse

    def watched(matrix, device="cpu"):
        converted.append(id(matrix))
        return original(matrix, device)

    monkeypatch.setattr(conseal, "_as_sparse", watched)
    connectome.evaluate(k, csc)
    assert not {id(csc.x), id(csc.y), id(csc.z)} & set(converted)


def test_endpoints_align_refuses_the_gpu_before_it_builds_anything(monkeypatch):
    """Without CuPy, device="gpu" is refused at once, not after the grids are built."""
    import sbci.conseal as conseal

    def missing():
        raise ImportError("no CuPy here")

    def built(*args, **kwargs):
        raise AssertionError("the grids were built first")

    monkeypatch.setattr(conseal._device, "cupy", missing)
    monkeypatch.setattr(conseal, "default_grids", built)
    with pytest.raises(ImportError, match="no CuPy here"):
        endpoints_align([object(), object()], device="gpu")


def test_a_template_in_single_precision_is_taken_at_its_precision(grid, connectome, kernel):
    """Normalized in float32, a template's squares sum to 1 only within about 1e-6.

    So one 3e-6 off was refused as a density passed in place of its square root. It is
    taken in float64, scaled to unit mass; in float64 the same is still refused.
    """
    k, dk = kernel
    subjects = [connectome, made(grid, 9)]
    template = ConSEAL(grid, grid).template(subjects, k)
    single = (template * np.sqrt(1 + 3e-6)).astype(np.float32)
    settings = {**SMALL, "grids": (grid, grid)}
    taken = endpoints_align(subjects, template=single, **settings)
    exact = endpoints_align(subjects, template=template, **settings)
    np.testing.assert_allclose(taken.costs[0], exact.costs[0], rtol=1e-3)  # float32's rounding
    engine = ConSEAL(grid, grid, max_iterations=2)
    engine.register(single, subjects[1], k, dk)
    with pytest.raises(ValueError, match="not 1"):
        engine.register(template * np.sqrt(1 + 3e-6), subjects[1], k, dk)
