"""ConSEAL: what the mathematics guarantees on any icosphere.

The comparison against the reference MATLAB lives in ``tests/reference``
(``conseal_reference.m`` and ``conseal_compare.py``) and is run by hand on
Longleaf. What is asserted here holds without it: point location agrees with
the exact search, the adjacency and the kernel conserve mass, every analytic
derivative matches a finite difference of the quantity it claims to
differentiate, the reference's gradient does not, warps stay diffeomorphic,
and registration recovers a rotation and reduces the cost of a warp.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.alignment import MeshQuery, SphericalGrid, sphere_exp_map
from sbci.conseal import (
    ConSEAL,
    EndpointConnectome,
    EndpointWarp,
    FastMeshQuery,
    HeatKernelBuilder,
    StationaryWarp,
    _legendre_with_derivative,
    cotangent_laplacian,
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

pytestmark = pytest.mark.filterwarnings("ignore:skipping a warp update")


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


def test_fast_query_agrees_with_the_exact_search(grid):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(500, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    query = FastMeshQuery(grid.vertices, grid.faces)
    fast_w, fast_i, fast_f = query.query(points)
    exact_w, exact_i = MeshQuery(grid.vertices, grid.faces).query(points)
    assert np.array_equal(fast_i, exact_i)
    assert np.allclose(fast_w, exact_w, atol=1e-12)
    assert np.array_equal(grid.faces[fast_f], fast_i)
    assert np.array_equal(fast_f, query._exact_faces(points))


def test_a_query_at_a_vertex_puts_all_weight_on_it(grid):
    w, i, _ = FastMeshQuery(grid.vertices, grid.faces).query(grid.vertices[:20])
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


def test_the_strict_derivative_matches_the_reference_formula(grid):
    """Move one source vertex, normalization included: the reference's dK."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigma, eps, i = 0.05, 1e-6, 23
    _, dk = builder.compute(sigma, strict_upstream=True)
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


def test_bandwidth_cross_validation_scores_every_candidate(grid, connectome):
    """Both self-term conventions give finite scores and pick a bandwidth from the list."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    sigmas = [0.02, 0.05, 0.1]
    best, scores = builder.cross_validate(connectome, sigmas)
    strict_best, strict_scores = builder.cross_validate(connectome, sigmas, strict_upstream=True)
    assert best in sigmas and strict_best in sigmas
    assert np.isfinite(scores).all() and np.isfinite(strict_scores).all()
    assert scores.shape == (3,) and not np.allclose(scores, strict_scores)


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
    """The strict path adds the other slot's derivative in the wrong frame (item 1)."""
    builder = HeatKernelBuilder(grid, grid, degree=12)
    k, dk = builder.compute(0.05)
    _, true_e1, _ = connectome.evaluate(k, dk)
    k_up, dk_up = builder.compute(0.05, strict_upstream=True)
    _, strict_e1, _ = connectome.evaluate(k_up, dk_up, strict_upstream=True)
    assert np.allclose(k.toarray(), k_up.toarray())
    correlation = np.corrcoef(true_e1.ravel(), strict_e1.ravel())[0, 1]
    assert correlation < 0.99


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
    warp = StationaryWarp(grid).rotate(rotation)
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


def test_a_refused_update_leaves_the_field_alone_unless_strict(grid, monkeypatch):
    """The flow of a velocity field cannot fold, so force the reference's check to fire."""
    import sbci.conseal as conseal

    warp, strict = StationaryWarp(grid), StationaryWarp(grid)  # before the fold test is faked
    monkeypatch.setattr(conseal, "triangles_fold", lambda vertices, faces: True)
    displacement = 0.01 * np.ones((grid.n_vertices, 2))
    before = warp.velocity.copy()
    assert not warp.compose(displacement)
    assert warp.rejected == 1
    assert np.array_equal(warp.velocity, before)
    assert np.allclose(warp.vertices, grid.vertices)
    assert not strict.compose(displacement, strict_upstream=True)
    assert not np.array_equal(strict.velocity, before)  # the reference keeps it (item 3)
    assert np.allclose(strict.vertices, grid.vertices)


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
    # the cost is exactly zero, so the gradient is zero, the step is zero and a rising
    # cost (relocating an endpoint on an ico2 triangle is not exactly idempotent) is
    # rolled back: the trace stays at its initial value
    assert costs.tolist() == [0.0]
    assert np.allclose(lh.vertices, grid.vertices, atol=1e-9)
    p_before, _ = connectome.positions()
    p_after, _ = warped.positions()
    assert np.abs(np.arccos(np.clip((p_before * p_after).sum(1), -1, 1))).max() < 0.02


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
    """A right triangle: the edge opposite the right angle gets no weight."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    standard = cotangent_laplacian(vertices, faces).toarray()
    assert standard[1, 2] == pytest.approx(0.0)
    assert standard[0, 1] == pytest.approx(-0.5) and standard[0, 2] == pytest.approx(-0.5)
    reference = cotangent_laplacian(vertices, faces, reference_layout=True).toarray()
    assert reference[1, 2] == pytest.approx(-0.5)  # the reference's misplaced weight (item 12)
    for matrix in (standard, reference):
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(matrix @ np.ones(3), 0.0)


def test_a_half_turn_rotation_is_not_mistaken_for_the_identity(grid):
    half_turn = np.diag([-1.0, -1.0, 1.0])
    warp = StationaryWarp(grid).rotate(half_turn)
    expected = grid.vertices @ half_turn.T
    turned = np.arccos(np.clip((warp.vertices * expected).sum(1), -1, 1))
    unmoved = np.arccos(np.clip((warp.vertices * grid.vertices).sum(1), -1, 1))
    assert np.median(turned) < 0.2
    assert np.median(unmoved) > 2.0


def test_the_chain_rule_is_zero_where_the_density_was_clamped(grid, connectome, kernel):
    k, dk = kernel
    f = connectome.evaluate(k)
    _, d_e1, d_e2 = connectome.q_transform(k, dk)
    assert np.isfinite(d_e1).all() and np.isfinite(d_e2).all()
    assert not d_e1[f == 0].any() and not d_e2[f == 0].any()


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
    from sbci.conseal import ConSEAL, FastMeshQuery

    rotations, permutations = mesh_symmetries(grid.vertices, grid.faces)
    rng = np.random.default_rng(5)
    values = rng.random((grid.n_vertices, grid.n_vertices))
    query = FastMeshQuery(grid.vertices, grid.faces)
    for rotation, perm in zip(rotations[:7], permutations[:7], strict=True):
        image = np.argsort(perm)  # image[i] = where vertex i lands
        pulled = ConSEAL._pull_back(values, grid, query, rotation)
        np.testing.assert_allclose(pulled, values[np.ix_(image, image)], atol=1e-9)


def test_endpoints_align_refuses_a_misshapen_template(grid, connectome):
    with pytest.raises(ValueError, match="template"):
        endpoints_align(
            [connectome], template=np.ones(3), sigma=0.05, kernel_degree=12, grids=(grid, grid)
        )
