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


def test_the_spherical_kernel_integrates_to_one():
    """A heat kernel is a probability density: only the l=0 term survives."""
    from sbci.smoothing import spherical_heat_kernel

    vertices, areas = _fine_sphere(4)
    pole = np.array([0.0, 0.0, 1.0])
    values = spherical_heat_kernel(vertices @ pole, sigma=0.005)
    assert float(values @ areas) == pytest.approx(1.0, rel=2e-3)


def test_the_spherical_kernel_peaks_at_its_centre_and_falls_away():
    """It falls cleanly near the centre, then rings -- the series is truncated.

    Monotonicity would be the wrong thing to assert: at sigma 0.005 the kernel
    decays for the first 22 degrees and then oscillates at the 0.05% level,
    which is what truncating a Legendre series does.
    """
    from sbci.smoothing import spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 200)
    values = spherical_heat_kernel(np.cos(angles), sigma=0.005)

    assert values[0] == values.max()
    near = angles < np.radians(20)
    assert np.all(np.diff(values[near]) < 0), "should fall away from its centre"
    assert values.min() > -0.001 * values.max(), "ringing should stay tiny"


def test_a_wider_kernel_is_flatter():
    from sbci.smoothing import spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 60)
    narrow = spherical_heat_kernel(np.cos(angles), sigma=0.002)
    wide = spherical_heat_kernel(np.cos(angles), sigma=0.05)
    assert narrow.max() > wide.max()
    assert narrow.std() > wide.std()


def test_a_very_wide_kernel_approaches_the_uniform_density():
    """As diffusion runs, the kernel forgets where it started."""
    from sbci.smoothing import spherical_heat_kernel

    values = spherical_heat_kernel(np.cos(np.linspace(0, np.pi, 50)), sigma=5.0)
    np.testing.assert_allclose(values, 1 / (4 * np.pi), rtol=1e-3)


def test_the_pipelines_truncation_is_part_of_the_definition():
    """33 harmonics is not converged at the released bandwidth, and that matters.

    A porting attempt that "improves" on the pipeline by summing more terms
    would stop reproducing it. At sigma 0.005 the 33-term kernel sits 0.31%
    from a converged one and the 17-term kernel 20% away, so the truncation has
    to be matched, not exceeded.
    """
    from sbci.smoothing import SPHERICAL_HARMONICS, spherical_heat_kernel

    assert SPHERICAL_HARMONICS == 33, "the pipeline passes --OPT_VAL_num_harm 33"

    cosine = np.cos(np.linspace(0, np.pi, 200))
    converged = spherical_heat_kernel(cosine, sigma=0.005, harmonics=129)
    peak = converged.max()

    errors = {
        n: np.abs(spherical_heat_kernel(cosine, sigma=0.005, harmonics=n) - converged).max()
        for n in (17, 25, 33, 49, 65)
    }
    assert errors[17] / peak > 0.1, "17 terms is nowhere near converged"
    assert 0.001 < errors[33] / peak < 0.01, "33 terms leaves a small signature"
    assert errors[65] / peak < 1e-6, "65 terms has converged"
    assert errors[17] > errors[25] > errors[33] > errors[49] > errors[65]


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
