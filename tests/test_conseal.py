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


def test_the_defaults_are_the_papers_settings_and_strict_upstream_the_public_codes(grid):
    """Since 7 October 2026 (PORTING.md item 19): no smoothing, no clamp, step 0.1, 1e-6.

    ``strict_upstream=True`` takes the public code's settings for any not given, and a
    setting given is kept in either mode.
    """

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
    public = ConSEAL(grid, grid, strict_upstream=True)
    assert settings(public) == (0.05, 100, 1e-4, 0.2, 0.05)
    assert all(warp.viscosity == 0.05 for warp in public.new_warps())
    given = ConSEAL(grid, grid, delta=0.2, viscosity=0.01, strict_upstream=True)
    assert settings(given) == (0.2, 100, 1e-4, 0.2, 0.01)
    assert settings(ConSEAL(grid, grid, threshold=1e-3)) == (0.1, 1000, 1e-3, np.inf, 0.0)


def test_the_default_settings_undo_more_of_a_known_warp_than_the_public_codes(
    grid, connectome, kernel
):
    """A smooth warp the basis can represent, moved back: the measure the defaults were chosen by.

    On ico4 the paper's settings undid 93% to 97% of three known warps and the
    public code's 52% to 75% (PORTING.md item 19); on this ico2 grid 76% against 2%,
    because 1e-4 stops the public descent after its first step.
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
    errors = {
        name: degrees_off(ConSEAL(grid, grid, **settings).register(connectome, moved, k, dk)[3])
        for name, settings in (("paper", {}), ("public", public))
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
    """The reference reads the axis off the skew part of R, which vanishes for a half turn."""
    half_turn = np.diag([-1.0, -1.0, 1.0])
    expected = grid.vertices @ half_turn.T
    np.testing.assert_allclose(
        StationaryWarp(grid).rotate(half_turn).vertices, expected, atol=1e-12
    )
    warp = StationaryWarp(grid, strict_upstream=True).rotate(half_turn)
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
    moving every endpoint back by the whole warp.
    """
    k, dk = kernel
    engine = ConSEAL(grid, grid, max_iterations=1)
    # relocating through the identity is not exactly idempotent on ico2 triangles
    _, _, _, same = engine.register(connectome, connectome, k, dk)
    baseline = _furthest_move(connectome, same)

    lh_warp, rh_warp = StationaryWarp(grid, viscosity=0.0), StationaryWarp(grid, viscosity=0.0)
    field = 0.2 * grid.basis[:, 0, :]
    assert lh_warp.compose(field) and rh_warp.compose(field)
    warped = connectome.copy()
    warped.warp(lh_warp, rh_warp)
    carried = np.arccos(np.clip((lh_warp.vertices * grid.vertices).sum(1), -1, 1)).max()
    assert carried > 0.05  # the warp moves the endpoints by far more than the baseline

    _, _, costs, registered = engine.register(warped, warped, k, dk)
    assert costs.tolist() == [0.0]
    assert _furthest_move(warped, registered) <= 2 * baseline
    assert _furthest_move(warped, registered) < 0.1 * carried


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


def test_strict_upstream_builds_the_default_grids_with_the_reference_basis(grid, monkeypatch):
    """The recurrence error is one more thing ``strict_upstream`` reproduces; given grids stay."""
    import sbci.conseal as conseal

    lh, rh = conseal.default_grids(order=1, reference=True)
    assert lh.reference and rh.reference
    assert not any(g.reference for g in conseal.default_grids(order=1))

    asked = []

    def small_grids(order, reference=False):
        asked.append(reference)
        return grid, grid

    monkeypatch.setattr(conseal, "default_grids", small_grids)
    rng = np.random.default_rng(41)
    subject = EndpointConnectome.from_points(grid, grid, *synthetic(rng, n=600))
    settings = {"template": 0, "sigma": 0.05, "kernel_degree": 12, "max_iterations": 0}
    endpoints_align([subject], strict_upstream=True, **settings)
    endpoints_align([subject], **settings)
    endpoints_align([subject], grids=(grid, grid), strict_upstream=True, **settings)
    assert asked == [True, False]


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
    # the reference's arithmetic, kept under strict_upstream: the starting subject's
    # distance to itself is acos(1 - 5e-11), a weight near 1e5, and the median stays put
    subjects = [_Density(np.sqrt(1 - 5e-11) * q) for q in qs]
    mean = sum(s.q for s in subjects) / len(subjects)
    start = int(np.argmin([((s.q - mean) ** 2).sum() for s in subjects]))
    strict = ConSEAL(grid, grid, strict_upstream=True).template(subjects, None)
    distances = degrees_between(strict, qs)
    assert distances[start] < 0.01
    assert np.sort(distances)[1] > 5.0


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


def projected_smoothing(warp, velocity):
    """The smoothing of the second review: 3-vectors per Cartesian component, projected back."""
    lifted = warp.e1 * velocity[:, :1] + warp.e2 * velocity[:, 1:]
    lifted = lifted - warp.viscosity * (warp._laplacian @ lifted)
    return np.stack([(lifted * warp.e1).sum(1), (lifted * warp.e2).sum(1)], axis=1)


def test_smoothing_leaves_a_rotation_alone_at_the_coordinate_poles(bundled_left):
    """Item 16: ten smoothing passes of a 20-degree rotation moved the polar vertices 9 degrees.

    What remains is the smoothing itself: the connection Laplacian maps a
    rotation's field to itself scaled by the vertex area, so ten passes of 5%
    shrink the rotation by a quarter of a percent everywhere alike (half a
    percent with the projected 3-vectors of the second review).
    """
    grid = bundled_left
    poles = np.flatnonzero(np.abs(grid.vertices[:, 2]) > 1 - 1e-9)
    assert poles.size == 2
    drift = {}
    for strict in (False, True):
        # the rotation's own field, as the reference holds a rotation; rotate()
        # itself now keeps the rotation out of the field (item 17)
        warp = StationaryWarp(grid, strict_upstream=strict)
        warp.velocity = rotational_velocity(warp, rotation_about_x(20.0))
        warp.vertices = warp.exponential()
        start = warp.vertices.copy()
        for _ in range(10):
            assert warp.compose(np.zeros((grid.n_vertices, 2)), strict_upstream=strict)
        drift[strict] = np.degrees(np.arccos(np.clip((warp.vertices * start).sum(1), -1, 1)))
    assert drift[True][poles].min() > 5.0
    assert drift[False][poles].max() < 0.06  # 0.046; the projected 3-vectors gave 0.092
    assert drift[False][poles].max() < 1.5 * drift[False].mean()


def test_both_smoothings_agree_where_the_frame_turns_slowly(bundled_left):
    """Within 10 degrees of the coordinate equator the frame is nearly parallel: 1e-4 apart.

    The projected 3-vectors of the second review differed there by their
    curvature term, 5% of the field weighted by the vertex area (2.7e-4).
    """
    grid = bundled_left
    warp = StationaryWarp(grid)
    velocity = rotational_velocity(warp, rotation_about_x(20.0))
    smoothed = warp._smooth(velocity)
    componentwise = warp._smooth(velocity, strict_upstream=True)
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
    warp, reference = StationaryWarp(grid), StationaryWarp(grid, strict_upstream=True)
    smoothings = {
        "connection": warp._smooth,
        "projected": lambda velocity: projected_smoothing(warp, velocity),
        "reference": lambda velocity: reference._smooth(velocity, strict_upstream=True),
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
    assert kept["z", "reference"] > 0.985 and kept["tilted", "reference"] < 0.91


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
    """The reference's flow of a rotation was 1.6 degrees off at 150, and each step shrank it."""
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
    if degrees == 150.0:
        flowed = StationaryWarp(grid, strict_upstream=True).rotate(rotation)
        assert degrees_off(flowed.vertices, expected).max() > 1.0  # the reference's
        for _ in range(10):
            flowed.compose(np.zeros((grid.n_vertices, 2)), strict_upstream=True)
        assert degrees_off(flowed.vertices, expected).max() > 1.5  # and eroding


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
    for strict in (False, True):
        engine = ConSEAL(grid, grid, strict_upstream=strict)
        with pytest.raises(ValueError, match="subject 1 has no streamlines"):
            engine.template([connectome, empty], k)
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


def test_the_rigid_search_undoes_a_turn_about_z(grid, connectome, kernel):
    """The reference's caps turn about axes perpendicular to z, never about z itself.

    So an 8-degree turn about z was left 9.6 and 8.2 degrees off on this grid; each
    shell's best are now turned about z as well. ``strict_upstream=True`` keeps the
    reference's search, and the turn.
    """
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
    k, dk = HeatKernelBuilder(grid, grid, degree=12).compute(0.05, strict_upstream=True)
    lh, _, _, _ = ConSEAL(grid, grid, max_iterations=0, strict_upstream=True).register(
        connectome, rotated, k, dk, init_rotation=True
    )
    assert degrees_left(lh.rigid) > 5.0
