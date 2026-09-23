"""The kernel smoothing port.

The kernels are checked against closed forms on an orthonormal basis, where
``U diag(rho) U'`` reduces to something writable by hand, and against the
limiting behaviour the maths dictates: as the bandwidth grows, every mode but
the constant one is suppressed and the kernel must collapse to rank one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from sbci.metadata import template as metadata_template
from sbci.smoothing import (
    TRUNCATION_FLOOR,
    Endpoints,
    diffusion_kernel,
    kappa_candidates,
    matern_kernel,
    smooth_endpoints,
)


@pytest.fixture
def basis():
    """An orthonormal basis and a spectrum, so the kernels have a closed form."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    eigenvalues = np.linspace(0.0, 1.0, 12)
    return eigenvalues, q


# --- the kernels ----------------------------------------------------------


def test_diffusion_kernel_matches_its_closed_form(basis):
    """With an orthonormal U, K = U diag(rho) U' and rho is exp(-k^2 L / 2)."""
    eigenvalues, u = basis
    kappa = 1.7
    expected = u @ np.diag(np.exp(-(kappa**2) / 2 * eigenvalues)) @ u.T
    np.testing.assert_allclose(diffusion_kernel(eigenvalues, u, kappa), expected)


def test_matern_kernel_matches_its_closed_form(basis):
    eigenvalues, u = basis
    kappa, nu = 1.3, 2.0
    rho = (2 * nu / kappa**2 + eigenvalues) ** (-nu - 1)
    np.testing.assert_allclose(matern_kernel(eigenvalues, u, kappa, nu), u @ np.diag(rho) @ u.T)


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_are_symmetric(basis, build):
    eigenvalues, u = basis
    kernel = build(eigenvalues, u, 1.1)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_are_positive_semidefinite(basis, build):
    """A positive rho means U diag(rho) U' has no negative eigenvalue."""
    eigenvalues, u = basis
    assert np.linalg.eigvalsh(build(eigenvalues, u, 1.1)).min() > -1e-12


def test_diffusion_kernel_collapses_to_rank_one(basis):
    """Large bandwidth suppresses every mode but the one at lambda = 0."""
    eigenvalues, u = basis
    kernel = diffusion_kernel(eigenvalues, u, kappa=400.0)
    spectrum = np.linalg.eigvalsh(kernel)
    significant = spectrum > 1e-6 * spectrum.max()
    assert significant.sum() == 1, f"rank {significant.sum()}, expected 1"
    np.testing.assert_allclose(kernel, np.outer(u[:, 0], u[:, 0]), atol=1e-9)


def test_smaller_bandwidth_keeps_more_modes(basis):
    """Monotone in the only sense the maths guarantees: modes retained."""
    eigenvalues, u = basis
    counts = []
    for kappa in (1.0, 10.0, 100.0, 400.0):
        rho = np.exp(-(kappa**2) / 2 * eigenvalues)
        counts.append(int((rho > TRUNCATION_FLOOR).sum()))
    assert counts == sorted(counts, reverse=True), counts


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_reject_a_nonpositive_bandwidth(basis, build):
    eigenvalues, u = basis
    with pytest.raises(ValueError, match="kappa must be positive"):
        build(eigenvalues, u, 0.0)


def test_matern_rejects_a_nonpositive_nu(basis):
    eigenvalues, u = basis
    with pytest.raises(ValueError, match="nu must be positive"):
        matern_kernel(eigenvalues, u, 1.0, nu=0.0)


def test_kernel_rejects_a_mismatched_spectrum(basis):
    _, u = basis
    with pytest.raises(ValueError, match="eigenvalues for"):
        diffusion_kernel(np.zeros(5), u, 1.0)


# --- bandwidth selection ---------------------------------------------------


def test_kappa_candidates_are_increasing_and_the_right_count():
    eigenvalues = np.linspace(0.0, 0.54, 2562)
    candidates = kappa_candidates(eigenvalues, n=10)
    assert candidates.size == 10
    assert np.all(np.diff(candidates) > 0)


def test_kappa_candidates_match_the_reference_formula():
    """The bounds are the ones rdk_smoothed_concon_compute.m computes."""
    eigenvalues = np.linspace(0.0, 0.54, 2562)
    candidates = kappa_candidates(eigenvalues, n=10, index=200)
    assert candidates[0] == pytest.approx(np.sqrt(-2 * np.log(0.9) / eigenvalues[-1]))
    assert candidates[-1] == pytest.approx(np.sqrt(-2 * np.log(0.001) / eigenvalues[199]))


def test_kappa_candidates_need_enough_spectrum():
    with pytest.raises(ValueError, match="at least 200 eigenvalues"):
        kappa_candidates(np.linspace(0.0, 1.0, 50))


# --- endpoints -------------------------------------------------------------


@pytest.fixture
def endpoints():
    """Four streamlines over a two-vertex-per-hemisphere toy surface.

    left-left   : vertex 0 <-> vertex 1
    right-right : vertex 0 <-> vertex 0 (a self connection)
    left-right  : left 1 -> right 0
    right-left  : right 1 -> left 0
    """
    return Endpoints(
        surf_in=np.array([0, 1, 0, 1], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 0], dtype=np.int8),
        vtx_in=np.array([0, 0, 1, 1]),
        vtx_out=np.array([1, 0, 0, 0]),
        n_per_hemi=2,
    )


def test_adjacency_is_built_by_hand(endpoints):
    a11, a22, a12 = endpoints.adjacency()

    # One left-left streamline 0->1, symmetrized to both off-diagonal entries.
    np.testing.assert_array_equal(a11, [[0, 1], [1, 0]])

    # One right-right self connection, doubled by the symmetrization -- which
    # is what the reference does, so the released data carries it too.
    np.testing.assert_array_equal(a22, [[2, 0], [0, 0]])

    # left 1 -> right 0, plus right 1 -> left 0 transposed into the same block.
    np.testing.assert_array_equal(a12, [[0, 1], [1, 0]])


def test_adjacency_totals_count_each_within_hemisphere_streamline_twice(endpoints):
    a11, a22, _ = endpoints.adjacency()
    left_left = int(((endpoints.surf_in == 0) & (endpoints.surf_out == 0)).sum())
    assert a11.sum() == 2 * left_left


def test_one_based_indices_are_caught():
    """MATLAB indices are one-based; using them unconverted must not pass."""
    bad = Endpoints(
        surf_in=np.array([0], dtype=np.int8),
        surf_out=np.array([0], dtype=np.int8),
        vtx_in=np.array([2]),
        vtx_out=np.array([1]),
        n_per_hemi=2,
    )
    with pytest.raises(ValueError, match="One-based indices"):
        bad.adjacency()


def test_n_streamlines(endpoints):
    assert endpoints.n_streamlines == 4


# --- the smoothing itself --------------------------------------------------


def test_smoothing_produces_a_density(endpoints):
    """Symmetric, nonnegative, and summing to one over the whole matrix."""
    kernel = np.eye(2)
    density = smooth_endpoints(endpoints, kernel, kernel)

    assert density.shape == (4, 4)
    np.testing.assert_allclose(density, density.T)
    assert (density >= 0).all()
    assert density.sum() == pytest.approx(1.0)


def test_smoothing_with_an_identity_kernel_just_normalizes(endpoints):
    """With K = I the congruence is a no-op, so the result is A scaled to one."""
    a11, a22, a12 = endpoints.adjacency()
    total = a11.sum() + a22.sum() + 2 * a12.sum()
    density = smooth_endpoints(endpoints, np.eye(2), np.eye(2))

    np.testing.assert_allclose(density[:2, :2], a11 / total)
    np.testing.assert_allclose(density[2:, 2:], a22 / total)
    np.testing.assert_allclose(density[:2, 2:], a12 / total)


def test_smoothing_clips_negatives_from_spectral_truncation():
    """Truncation can make K A K negative; those entries become zero."""
    endpoints = Endpoints(
        surf_in=np.array([0], dtype=np.int8),
        surf_out=np.array([0], dtype=np.int8),
        vtx_in=np.array([0]),
        vtx_out=np.array([1]),
        n_per_hemi=2,
    )
    # A kernel with a negative off-diagonal drives one entry of K A K negative.
    kernel = np.array([[1.0, -0.6], [-0.6, 1.0]])
    density = smooth_endpoints(endpoints, kernel, np.eye(2))
    assert (density >= 0).all()
    assert density.sum() == pytest.approx(1.0)


def test_smoothing_rejects_a_mismatched_kernel(endpoints):
    with pytest.raises(ValueError, match="kernel_left has shape"):
        smooth_endpoints(endpoints, np.eye(3), np.eye(2))


def test_smoothing_rejects_empty_endpoints():
    empty = Endpoints(
        surf_in=np.array([], dtype=np.int8),
        surf_out=np.array([], dtype=np.int8),
        vtx_in=np.array([], dtype=np.int64),
        vtx_out=np.array([], dtype=np.int64),
        n_per_hemi=2,
    )
    with pytest.raises(ValueError, match="no positive mass"):
        smooth_endpoints(empty, np.eye(2), np.eye(2))


@pytest.fixture
def toy_endpoints():
    """Six streamlines over a toy grid of three vertices per hemisphere."""
    from sbci.smoothing import Endpoints

    return Endpoints(
        surf_in=np.array([0, 0, 0, 1, 1, 0], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 1, 0, 0], dtype=np.int8),
        vtx_in=np.array([0, 1, 2, 0, 1, 2]),
        vtx_out=np.array([1, 0, 2, 2, 0, 1]),
        n_per_hemi=3,
    )


@pytest.fixture
def toy_basis():
    """An orthonormal eigenbasis for that toy hemisphere."""
    u = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
    eigenvalues = np.array([0.0, 0.5, 1.0])
    return ((eigenvalues, u), (eigenvalues, u))


@pytest.fixture
def smoothable(sc_metadata, toy_endpoints):
    """A six-vertex SC connectome carrying those endpoints."""
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed

    rng = np.random.default_rng(1)
    matrix = rng.random((6, 6))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0.0)
    area = np.full(6, 2.0)
    matrix /= area @ matrix @ area
    return ContinuousConnectome(
        data=to_condensed(matrix).astype(np.float32),
        area=area,
        mask=np.ones(6, dtype=bool),
        metadata=sc_metadata,
        endpoints=toy_endpoints,
    )


def test_smooth_says_so_when_the_file_carries_no_endpoints(connectome):
    """A file without the optional group cannot be re-smoothed."""
    assert not connectome.has_endpoints
    with pytest.raises(ValueError, match="no streamline endpoints"):
        connectome.smooth(kernel="rdk")


def test_smooth_says_so_when_the_basis_is_missing(smoothable, monkeypatch):
    """With endpoints present, the next thing needed is the eigenbasis."""
    from sbci import smoothing

    assert smoothable.has_endpoints
    monkeypatch.setattr(smoothing, "find_basis", lambda: None)
    with pytest.raises(ValueError, match="Laplace-Beltrami basis"):
        smoothable.smooth(kernel="rdk")


def test_the_failure_names_every_place_it_looked(smoothable, monkeypatch):
    """A path error should say where to put the files, not just that they are absent."""
    from sbci import smoothing

    monkeypatch.setattr(smoothing, "find_basis", lambda: None)
    with pytest.raises(ValueError) as caught:
        smoothable.smooth(kernel="rdk")
    message = str(caught.value)
    assert smoothing.EIGENPAIR_ENV in message
    assert ".cache" in message
    for name in smoothing.EIGENPAIR_FILES:
        assert name in message


def test_the_basis_is_found_from_the_environment(tmp_path, monkeypatch):
    """SBCI_LBO_DIR saves passing eigenpairs= on every call."""
    from sbci import smoothing

    for name in smoothing.EIGENPAIR_FILES:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(tmp_path))
    assert smoothing.find_basis() == tmp_path


def test_a_directory_missing_one_hemisphere_is_not_accepted(tmp_path, monkeypatch):
    """Half a basis is not a basis."""
    from sbci import smoothing

    (tmp_path / smoothing.EIGENPAIR_FILES[0]).write_bytes(b"")
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    assert smoothing.find_basis() is None


def test_smooth_rejects_a_directory_without_the_basis(smoothable, tmp_path):
    with pytest.raises(FileNotFoundError, match="EV_LBO_ds_ico4_L.mat"):
        smoothable.smooth(kernel="rdk", eigenpairs=tmp_path)


@pytest.mark.parametrize("kernel", ["rdk", "matern"])
def test_smooth_returns_a_connectome_that_records_how_it_was_made(smoothable, toy_basis, kernel):
    """The whole path: endpoints in, a renormalized connectome out."""
    out = smoothable.smooth(kernel=kernel, bandwidth=1.0, eigenpairs=toy_basis)

    assert type(out) is type(smoothable)
    assert out.data.shape == smoothable.data.shape
    assert out.metadata["kernel"] == kernel
    assert out.metadata["bandwidth"] == 1.0
    assert out.endpoints is smoothable.endpoints
    # Unit mass is the package's convention, and re-smoothing must preserve it.
    area = np.asarray(out.area)
    assert area @ out.dense() @ area == pytest.approx(1.0, rel=1e-6)


def test_smooth_leaves_the_original_alone(smoothable, toy_basis):
    before = smoothable.data.copy()
    smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=toy_basis)
    np.testing.assert_array_equal(smoothable.data, before)


def test_smooth_method_still_validates_its_arguments(connectome):
    with pytest.raises(ValueError, match="kernel must be one of"):
        connectome.smooth(kernel="gaussian")


# --- the spherical heat kernel -------------------------------------------


def _fine_sphere(subdivisions=3):
    """A sphere fine enough to integrate on, with its vertex areas."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from sbci.alignment import voronoi_areas
    from test_alignment import icosphere

    vertices, faces = icosphere(subdivisions)
    return vertices, voronoi_areas(vertices, faces)


def test_the_kernel_does_not_integrate_to_one():
    """The kernel concon applies is not a probability density, and that is fine.

    A heat kernel integrates to 1 because its weight is (2l+1)/4pi and only the
    l=0 term survives integration. concon's weight is (2l+1)^(3/2)/sqrt(4pi),
    so the surviving term leaves sqrt(4pi) instead. The pipeline divides by the
    streamline count afterwards rather than normalizing the kernel, so this
    factor is part of what the released files carry.
    """
    from sbci.smoothing import spherical_heat_kernel

    pole = np.array([0.0, 0.0, 1.0])

    def integral(subdivisions):
        vertices, areas = _fine_sphere(subdivisions)
        return float(spherical_heat_kernel(vertices @ pole, 0.005, epsilon=None) @ areas)

    # The kernel spans about 12 degrees, so a coarse grid under-resolves it;
    # refining converges on sqrt(4 pi), which is what pins the constant.
    coarse, fine = integral(4), integral(5)
    target = np.sqrt(4 * np.pi)
    assert abs(fine - target) < abs(coarse - target), "should converge as the grid refines"
    assert fine == pytest.approx(target, rel=5e-3)
    assert fine > 3.0, "and so emphatically not 1"


def test_the_kernel_falls_from_its_centre_then_stops_dead():
    """Inside the cutoff it decays; beyond it, it is exactly zero.

    The truncated series would ring past the cutoff, but concon never evaluates
    it there: compute_kernel.cpp only visits vertices within the cutoff angle,
    so the kernel has compact support and no ringing survives into the output.
    """
    from sbci.smoothing import kernel_cutoff, spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 400)
    values = spherical_heat_kernel(np.cos(angles), sigma=0.005)
    cutoff = kernel_cutoff(0.005)

    assert values[0] == values.max()
    inside = angles < cutoff
    assert np.all(np.diff(values[inside]) < 0), "should fall away from its centre"
    assert np.all(values[~inside] == 0.0), "and be exactly zero beyond the cutoff"
    assert values.min() >= 0.0, "so nothing rings negative"


def test_the_kernel_matches_the_binary_it_ports():
    """Pinned against c3_main itself, run on a single streamline.

    One streamline gives D(i,j) = K(theta_i) K(theta_j), so the output is the
    kernel. These are the values it produced at sigma 0.005, at the ico4 ring
    distances from a grid vertex; tests/reference/concon_probe.py regenerates
    them. The heat kernel this package used to implement misses them by 0.09.
    """
    from sbci.smoothing import spherical_heat_kernel

    angle = np.array([0.0, 3.962, 6.430, 7.930, 9.960, 11.890])
    measured = np.array([1.0000, 0.7441, 0.4442, 0.2751, 0.1035, 0.01016])

    values = spherical_heat_kernel(np.cos(np.radians(angle)), sigma=0.005)
    ratio = values / values[0]
    assert np.sqrt(np.mean((ratio - measured) ** 2)) < 0.0005

    # The absolute scale is right too, not just the shape: concon's own peak
    # was 270.10, read off a run whose endpoints sat exactly on vertices.
    assert float(values[0]) == pytest.approx(270.10, rel=1e-3)


def test_the_cutoff_reproduces_the_binarys_support():
    """The predicted cutoff has to keep exactly the vertices concon keeps.

    Measured from c3_main: 16, 31 and 61 vertices at sigma 0.0025, 0.005 and
    0.01, whose outermost rings sit at 7.932, 11.894 and 16.528 degrees.
    """
    from sbci.smoothing import kernel_cutoff

    for sigma, outermost, next_ring in [
        (0.0025, 7.932, 8.9),
        (0.005, 11.894, 12.61),
        (0.01, 16.528, 17.5),
    ]:
        cutoff = np.degrees(kernel_cutoff(sigma))
        assert outermost < cutoff < next_ring, (
            f"sigma {sigma}: cutoff {cutoff:.3f} must keep the ring at "
            f"{outermost} and drop the next"
        )


def test_a_wider_kernel_is_flatter():
    from sbci.smoothing import spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 60)
    narrow = spherical_heat_kernel(np.cos(angles), sigma=0.002)
    wide = spherical_heat_kernel(np.cos(angles), sigma=0.05)
    assert narrow.max() > wide.max()
    assert narrow.std() > wide.std()


def test_a_very_wide_kernel_goes_flat():
    """At a large bandwidth every term but l=0 dies, leaving a constant.

    That constant is 1/sqrt(4pi), not the heat kernel's 1/4pi, for the same
    reason the kernel does not integrate to one.
    """
    from sbci.smoothing import spherical_heat_kernel

    values = spherical_heat_kernel(np.cos(np.linspace(0, np.pi, 50)), sigma=5.0)
    np.testing.assert_allclose(values, 1 / np.sqrt(4 * np.pi), rtol=1e-3)


def test_the_truncation_is_thirty_two_and_is_part_of_the_definition():
    """The binary hardcodes 33 terms, and 33 is not converged.

    The pipeline passes --OPT_VAL_num_harm 33, but c3_main ignores it: it
    prints "num_harmonics now global constant (for speed)" and subject.cpp
    prints so at startup. Feeding the binary 9, 17, 25, 33, 49 and 65 gives a
    bit-identical kernel. The count was pinned at 33 by matching the binary's
    peak across bandwidths 0.00125 to 0.02, where 32 drifts to 4% and 34
    overshoots. Since the sum is not converged, a port that "improved" on it by
    summing further would stop reproducing the released cohorts.
    """
    from sbci.smoothing import NUM_HARMONICS, spherical_heat_kernel

    assert NUM_HARMONICS == 33, "33 terms, l = 0..32, whatever the flag says"

    cosine = np.cos(np.linspace(0, np.radians(12.0), 200))
    truncated = spherical_heat_kernel(cosine, sigma=0.005, epsilon=None)
    further = spherical_heat_kernel(cosine, sigma=0.005, harmonics=65, epsilon=None)
    assert np.abs(truncated - further).max() / truncated.max() > 1e-3, (
        "32 terms is not converged, so the truncation has to be matched"
    )


def test_endpoint_density_is_symmetric_and_hemisphere_respecting():
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(2)
    n = vertices.shape[0]
    vertex_hemisphere = np.zeros(n, dtype=int)
    vertex_hemisphere[n // 2 :] = 1

    rng = np.random.default_rng(0)
    points = vertices[rng.integers(0, n, 40)]
    hemisphere = vertex_hemisphere[rng.integers(0, n, 40)]

    density = endpoint_density(
        vertices,
        points,
        points,
        hemisphere,
        hemisphere,
        vertex_hemisphere,
        sigma=0.01,
    )
    assert density.shape == (n, n)
    np.testing.assert_allclose(density, density.T, atol=1e-12)
    assert density.min() >= 0


def test_endpoint_density_puts_mass_where_the_endpoints_are():
    """A single streamline should light up its own two endpoints."""
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(3)
    n = vertices.shape[0]
    vertex_hemisphere = np.zeros(n, dtype=int)

    first, second = 10, 400
    density = endpoint_density(
        vertices,
        vertices[[first]],
        vertices[[second]],
        np.array([0]),
        np.array([0]),
        vertex_hemisphere,
        sigma=0.005,
    )
    peak = np.unravel_index(np.argmax(density), density.shape)
    for index in peak:
        angle = np.degrees(np.arccos(np.clip(vertices[index] @ vertices[first], -1, 1)))
        other = np.degrees(np.arccos(np.clip(vertices[index] @ vertices[second], -1, 1)))
        assert min(angle, other) < 10.0, "the peak is nowhere near an endpoint"


def test_endpoint_density_validates_its_arguments():
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(1)
    n = vertices.shape[0]
    zero = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="endpoints disagree"):
        endpoint_density(
            vertices,
            vertices[:3],
            vertices[:2],
            np.zeros(3, int),
            np.zeros(2, int),
            zero,
            sigma=0.01,
        )
    with pytest.raises(ValueError, match="normalize must be"):
        endpoint_density(
            vertices,
            vertices[:2],
            vertices[:2],
            np.zeros(2, int),
            np.zeros(2, int),
            zero,
            sigma=0.01,
            normalize="area",
        )


def test_reading_the_concon_endpoint_file(tmp_path):
    """The .tsv c3_main takes: a count header, then ten columns per streamline."""
    from sbci.smoothing import read_concon_endpoints

    path = tmp_path / "subject_xing_sphere_avg_coords.tsv"
    path.write_text(
        "#2\n"
        "0\t 1\t 1.0\t 0.0\t 0.0\t 0\t 0\t 0.0\t 1.0\t 0.0\n"
        "0\t 0\t 0.0\t 0.0\t 1.0\t 0\t 1\t 0.0\t 0.0\t -1.0\n"
    )
    points_in, points_out, hemisphere_in, hemisphere_out = read_concon_endpoints(path)

    assert points_in.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(points_in, axis=1), 1.0)
    # The file stores 1 - surface, so a 1 in the column means the left hemisphere.
    np.testing.assert_array_equal(hemisphere_in, [0, 1])
    np.testing.assert_array_equal(hemisphere_out, [1, 0])


def test_endpoint_positions_rebuilds_points_on_the_sphere():
    """The shk path needs continuous positions, and nothing else exercised it.

    Wiring smooth(kernel="shk") up the first time crashed here on a real file,
    because the triangle offset was read from an attribute Endpoints does not
    carry. Every unit test passed regardless: none of them had barycentric
    endpoints. This one does.
    """
    from sbci.smoothing import Endpoints, endpoint_positions
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    faces_per_hemi = sphere.faces.shape[0] // 2
    rng = np.random.default_rng(0)
    count = 64

    tri_in = rng.integers(0, faces_per_hemi, count)
    tri_out = rng.integers(0, faces_per_hemi, count)
    surf_in = rng.integers(0, 2, count).astype(np.int8)
    surf_out = rng.integers(0, 2, count).astype(np.int8)
    bary = rng.random((count, 3))
    bary /= bary.sum(axis=1, keepdims=True)

    endpoints = Endpoints(
        surf_in=surf_in,
        surf_out=surf_out,
        vtx_in=np.zeros(count, int),
        vtx_out=np.zeros(count, int),
        tri_in=tri_in,
        tri_out=tri_out,
        bary_in=bary,
        bary_out=bary.copy(),
    )
    points_in, points_out = endpoint_positions(endpoints, surface=sphere)

    assert points_in.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(points_in, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(points_out, axis=1), 1.0, atol=1e-12)

    # A point must lie in its own triangle, so the nearest vertex of that
    # triangle is nearer than the triangle's own circumradius is wide.
    corners = sphere.vertices[
        sphere.faces[tri_in + Endpoints.global_hemisphere_offset(surf_in, faces_per_hemi)]
    ]
    corners = corners / np.linalg.norm(corners, axis=2, keepdims=True)
    inside = np.einsum("sj,skj->sk", points_in, corners).max(axis=1)
    assert np.all(inside > np.cos(np.radians(5.0))), "should sit inside its own triangle"


def test_endpoint_positions_refuses_a_file_without_them():
    from sbci.errors import MissingDataError
    from sbci.smoothing import Endpoints, endpoint_positions

    bare = Endpoints(
        surf_in=np.zeros(3, np.int8),
        surf_out=np.zeros(3, np.int8),
        vtx_in=np.zeros(3, int),
        vtx_out=np.zeros(3, int),
    )
    with pytest.raises(MissingDataError, match="barycentric"):
        endpoint_positions(bare)


@pytest.mark.parametrize(
    "shape", [(), (1,), (5,), (3, 4), (2, 3, 4)], ids=["scalar", "one", "vector", "2d", "3d"]
)
def test_the_kernel_preserves_the_shape_it_is_given(shape):
    """Including a scalar, which is a 0-d array and cannot be an `out=` target.

    The recurrence writes through `out=` buffers to avoid allocating 168 MB per
    term when smoothing a subject. numpy refuses a 0-d array there, so
    `spherical_heat_kernel(1.0, ...)` raised TypeError until the series learned
    to work on a flat view and reshape back.
    """
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.full(shape, 0.999) if shape else 0.999
    assert np.shape(spherical_heat_kernel(cosine, 0.005)) == shape


def test_the_kernel_is_elementwise():
    """Evaluating together or one at a time has to give the same answer."""
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.cos(np.radians(np.array([0.0, 4.0, 8.0, 12.0, 30.0])))
    together = spherical_heat_kernel(cosine, 0.005)
    apart = np.array([float(spherical_heat_kernel(c, 0.005)) for c in cosine])
    np.testing.assert_allclose(together, apart, rtol=0, atol=1e-12)


def test_resmoothing_leaves_the_medial_wall_as_the_reference_does():
    """smooth() must not silently mask, however tempting it is.

    The format requires a stored file to carry nothing on the medial wall, but
    neither reference produces one: c3_main's released output puts 0.87% of its
    mass there, and MATLAB's rdk density does too. Zeroing it inside smooth()
    would make the method diverge from both and break
    test_smooth_method_matches_matlab, so the conflict is recorded as
    SPEC_QUESTIONS.md item 14 rather than papered over. This pins the choice.
    """
    from sbci.smoothing import _finish

    n = 8
    rng = np.random.default_rng(0)
    dense = rng.random((n, n))
    dense = dense + dense.T
    mask = np.ones(n, dtype=bool)
    mask[[2, 5]] = False

    class Fake:
        def __init__(self, data, area, mask, metadata, coords, endpoints):
            self.data, self.area, self.mask = data, area, mask
            self.metadata, self.coords, self.endpoints = metadata, coords, endpoints

    connectome = Fake(None, np.full(n, 1.0 / n), mask, metadata_template("sc"), None, None)
    out = _finish(connectome, dense.copy(), "shk", 0.005)

    from sbci.grid import to_dense

    result = to_dense(out.data, n)
    assert float(np.abs(result[~mask]).sum()) > 0.0, "the wall keeps what the reference gives it"
    assert float(connectome.area @ result.astype(np.float64) @ connectome.area) == pytest.approx(
        1.0, rel=1e-6
    )


def test_the_quantized_kernel_reproduces_the_binarys_peak_entry_exactly():
    """c3_main reads the kernel through two truncation-indexed lookup tables.

    On a single streamline its peak entry is K(dot_1) * K(dot_2) with one dot
    rounding into the exact x = 1.0 bucket and the other one truncation step
    below. The measured peaks were 270.1035 at sigma 0.005 and 1005.4661 at
    0.00125; the geometric mean of the two buckets has to give them back. The
    exact series cannot: it is 0.06% high at the peak, which is the 0.14%
    amplitude offset the densities used to carry.
    """
    from sbci.smoothing import concon_kernel_table

    for sigma, measured in ((0.005, 270.1035), (0.00125, 1005.4661), (0.02, 48.3245)):
        table = concon_kernel_table(sigma)
        peak = float(np.sqrt(table(1.0) * table(1.0 - 1e-12)))
        assert peak == pytest.approx(measured, rel=2e-5), (sigma, peak, measured)


def test_quantized_and_exact_kernels_differ_only_by_the_table_bias():
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.cos(np.radians(np.linspace(0.0, 11.5, 50)))
    quantized = spherical_heat_kernel(cosine, 0.005)
    exact = spherical_heat_kernel(cosine, 0.005, quantized=False)
    ratio = quantized / exact
    assert np.all(ratio <= 1.0 + 1e-12), "truncation only ever rounds the kernel down"
    assert ratio.min() > 0.995, "and by well under one percent"


def test_quantized_and_exact_share_the_cutoff():
    from sbci.smoothing import concon_kernel_table, kernel_cutoff

    for sigma in (0.0025, 0.005, 0.01):
        assert concon_kernel_table(sigma).cutoff == pytest.approx(kernel_cutoff(sigma), abs=2e-4)


def test_sparse_density_matches_dense():
    """The KD-tree path has to give the dense evaluation's answer, not an approximation."""
    from sbci.grid import hemisphere_labels
    from sbci.smoothing import endpoint_density
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = np.asarray(sphere.vertices, float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    hemi = hemisphere_labels(vertices.shape[0])
    rng = np.random.default_rng(0)
    count = 600
    p_in = rng.normal(size=(count, 3))
    p_in /= np.linalg.norm(p_in, axis=1, keepdims=True)
    p_out = rng.normal(size=(count, 3))
    p_out /= np.linalg.norm(p_out, axis=1, keepdims=True)
    h_in = rng.integers(0, 2, count)
    h_out = rng.integers(0, 2, count)

    dense = endpoint_density(
        vertices, p_in, p_out, h_in, h_out, hemi, 0.005, method="dense", block=256
    )
    sparse = endpoint_density(
        vertices, p_in, p_out, h_in, h_out, hemi, 0.005, method="sparse", block=256
    )
    assert dense.max() > 0
    np.testing.assert_allclose(sparse, dense, rtol=0, atol=1e-9 * dense.max())


def test_progress_callback_reports_every_block():
    from sbci.grid import hemisphere_labels
    from sbci.smoothing import endpoint_density
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = np.asarray(sphere.vertices, float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    hemi = hemisphere_labels(vertices.shape[0])
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(100, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    seen = []
    endpoint_density(
        vertices,
        pts,
        pts[::-1],
        np.zeros(100, int),
        np.zeros(100, int),
        hemi,
        0.005,
        block=30,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(30, 100), (60, 100), (90, 100), (100, 100)]


def test_mask_medial_wall_option_zeroes_the_wall_and_keeps_unit_mass():
    """The caller can choose the format's side of SPEC_QUESTIONS item 14."""
    from sbci.grid import to_dense
    from sbci.smoothing import _finish

    n = 8
    rng = np.random.default_rng(0)
    dense = rng.random((n, n))
    dense = dense + dense.T
    mask = np.ones(n, dtype=bool)
    mask[[2, 5]] = False

    class Fake:
        def __init__(self, data, area, mask, metadata, coords, endpoints):
            self.data, self.area, self.mask = data, area, mask
            self.metadata, self.coords, self.endpoints = metadata, coords, endpoints

    connectome = Fake(None, np.full(n, 1.0 / n), mask, metadata_template("sc"), None, None)
    out = to_dense(_finish(connectome, dense.copy(), "shk", 0.005, mask_medial_wall=True).data, n)
    assert float(np.abs(out[~mask]).sum()) == 0.0 and float(np.abs(out[:, ~mask]).sum()) == 0.0
    assert float(connectome.area @ out.astype(np.float64) @ connectome.area) == pytest.approx(
        1.0, rel=1e-6
    )


def test_final_threshold_matches_compute_kernel_cpp():
    """`if(temp > final_thold)` on the per-streamline scale, nothing else."""
    from sbci.smoothing import FINAL_THRESHOLD, apply_final_threshold

    assert FINAL_THRESHOLD == 1e-9, "--final_thold 0.000000001"
    n = 1_000_000
    density = np.array(
        [[0.0, 2e-9 * n, 1e-9 * n], [2e-9 * n, 0.0, 0.5e-9 * n], [1e-9 * n, 0.5e-9 * n, 0.0]]
    )
    out = apply_final_threshold(density.copy(), n)
    assert out[0, 1] == 2e-9 * n, "above the threshold survives"
    assert out[0, 2] == 0.0, "exactly at the threshold is dropped: the test is strictly greater"
    assert out[1, 2] == 0.0, "below is dropped"
