"""Compare the smoothing port against a real MATLAB run of the reference.

This is the acceptance test for PORTING.md item 1. It needs output from
``tests/reference/reference_run.m``, which is not committed because it is
260 MB; regenerate it with::

    module load matlab/2023b
    matlab -batch "run('tests/reference/reference_run.m')"

and point ``SBCI_MATLAB_REFERENCE`` at the directory it wrote, or leave it at
the default below. The tests skip when it is absent, so a checkout without
MATLAB still runs green.

Tolerances are in float32 epsilon on purpose. MATLAB computes the kernel in
single precision -- ``Lambda`` is stored as ``single`` and ``single .* double``
yields ``single`` -- so single-precision rounding is the most the reference can
agree to. The port itself works in float64 throughout.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from sbci.smoothing import Endpoints, diffusion_kernel, load_eigenpairs, smooth_endpoints

TOOLKIT = pathlib.Path(
    os.environ.get("SBCI_TOOLKIT", "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit")
)
REFERENCE = pathlib.Path(
    os.environ.get("SBCI_MATLAB_REFERENCE", "/work/users/x/y/xya/matlab-reference")
)

FLOAT32_EPS = float(np.finfo(np.float32).eps)
TOLERANCE = 10 * FLOAT32_EPS

pytestmark = pytest.mark.needs_data


def _require(*paths: pathlib.Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"MATLAB reference output not present: {missing}")


def _read(path: pathlib.Path, name: str) -> np.ndarray:
    """MATLAB -v7.3 is HDF5 written column-major, so it reads in transposed."""
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]).T.astype(np.float64)


@pytest.fixture(scope="module")
def eigenpairs():
    left = TOOLKIT / "concon_estimate/EV_LBO_ds_ico4_L.mat"
    right = TOOLKIT / "concon_estimate/EV_LBO_ds_ico4_R.mat"
    _require(left, right)
    return load_eigenpairs(left, "L"), load_eigenpairs(right, "R")


@pytest.fixture(scope="module")
def endpoints():
    path = TOOLKIT / "example_data/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat"
    _require(path)
    return Endpoints.from_matlab(path)


@pytest.fixture(scope="module")
def reference_kappa():
    path = REFERENCE / "reference_kernel_L.mat"
    _require(path)
    import h5py

    with h5py.File(path, "r") as handle:
        return float(np.asarray(handle["kappa"]).ravel()[0])


def _relative(mine: np.ndarray, theirs: np.ndarray) -> float:
    scale = np.abs(theirs).max()
    return float(np.abs(mine - theirs).max() / scale)


def test_adjacency_is_bit_identical(endpoints):
    """Endpoint counting involves no arithmetic, so it must match exactly."""
    path = REFERENCE / "reference_adjacency.mat"
    _require(path)
    a11, a22, a12 = endpoints.adjacency()

    for name, mine in (("A11", a11), ("A22", a22), ("A12", a12)):
        np.testing.assert_array_equal(mine, _read(path, name), err_msg=f"{name} differs")


def test_kernel_matches_matlab(eigenpairs, reference_kappa):
    (lam_l, u_l), _ = eigenpairs
    path = REFERENCE / "reference_kernel_L.mat"
    _require(path)

    relative = _relative(diffusion_kernel(lam_l, u_l, reference_kappa), _read(path, "KM_L"))
    assert relative < TOLERANCE, (
        f"kernel differs by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps), tolerance {TOLERANCE:.3g}"
    )


def test_density_matches_matlab(eigenpairs, endpoints, reference_kappa):
    """The whole pipeline: endpoints in, normalized density out."""
    (lam_l, u_l), (lam_r, u_r) = eigenpairs
    path = REFERENCE / "reference_density.mat"
    _require(path)

    density = smooth_endpoints(
        endpoints,
        diffusion_kernel(lam_l, u_l, reference_kappa),
        diffusion_kernel(lam_r, u_r, reference_kappa),
    )
    theirs = _read(path, "DM")

    relative = _relative(density, theirs)
    assert relative < TOLERANCE, (
        f"density differs by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps), tolerance {TOLERANCE:.3g}"
    )
    assert np.corrcoef(density.ravel(), theirs.ravel())[0, 1] > 1 - 1e-12


def test_smooth_method_matches_matlab(eigenpairs, reference_kappa):
    """The method, not just the function underneath it.

    ``smooth_endpoints`` is checked above. ``ContinuousConnectome.smooth()``
    adds four steps on top of it -- building both kernels from the basis,
    selecting the bandwidth, zeroing the diagonal the storage format excludes,
    and renormalizing to area-weighted unit mass -- and those need their own
    comparison. MATLAB's density is renormalized the same way here, so the two
    are the same quantity.
    """
    subject = (
        pathlib.Path(os.environ.get("SBCI_EXAMPLE_SC", "/work/users/x/y/xya/sbci-derivatives"))
        / "sub-example_desc-withendpoints_sc.h5"
    )
    path = REFERENCE / "reference_density.mat"
    _require(path)
    if not subject.is_file():
        pytest.skip(f"{subject} not built; run tools/import_legacy.py with endpoints")

    from sbci import ContinuousConnectome

    connectome = ContinuousConnectome.load(subject)
    area = np.asarray(connectome.area, dtype=np.float64)

    expected = np.asarray(_read(path, "DM"), dtype=np.float64)
    np.fill_diagonal(expected, 0.0)
    expected /= area @ expected @ area

    mine = (
        connectome.smooth(
            kernel="rdk", bandwidth=reference_kappa, eigenpairs=TOOLKIT / "concon_estimate"
        )
        .dense()
        .astype(np.float64)
    )

    relative = _relative(mine, expected)
    assert relative < TOLERANCE, (
        f"the smooth() method differs by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps), tolerance {TOLERANCE:.3g}"
    )
    assert np.abs(np.diag(mine)).max() == 0.0
    assert area @ mine @ area == pytest.approx(1.0, rel=1e-9)


def test_the_default_bandwidth_is_the_one_matlab_used(eigenpairs, reference_kappa):
    """A user who passes no bandwidth gets the reference run's bandwidth."""
    from sbci.smoothing import kappa_candidates

    (lam_l, _), _ = eigenpairs
    assert abs(float(kappa_candidates(lam_l)[3]) - reference_kappa) / reference_kappa < TOLERANCE


def test_reference_kappa_matches_the_selection_formula(eigenpairs, reference_kappa):
    """kappa_candidates reproduces the bandwidth MATLAB chose.

    MATLAB derives it from the float32 eigenvalues, so the two agree to single
    precision rather than double.
    """
    from sbci.smoothing import kappa_candidates

    (lam_l, _), _ = eigenpairs
    mine = kappa_candidates(lam_l)[3]
    assert abs(mine - reference_kappa) / reference_kappa < TOLERANCE


# --- SFC (PORTING.md item 2) ----------------------------------------------
#
# These agree to float64 rounding rather than float32: unlike the smoothing
# reference, the SFC functions carry no single-precision inputs.

SFC_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def legacy_matrices():
    """The legacy SC and FC, symmetrized exactly as the MATLAB driver does."""
    import scipy.io

    sub = TOOLKIT / "example_data/SBCI_Individual_Subject_Outcome"
    sc_path = sub / "smoothed_sc_avg_0.005_ico4.mat"
    fc_path = sub / "fc_avg_ico4.mat"
    _require(sc_path, fc_path)

    sc = scipy.io.loadmat(sc_path)["sc"]
    fc = scipy.io.loadmat(fc_path)["fc"]
    sc = sc + sc.T
    fc = fc + fc.T
    np.fill_diagonal(sc, 0.0)
    np.fill_diagonal(fc, 0.0)
    return sc, fc


@pytest.fixture(scope="module")
def reference_sfc():
    path = REFERENCE / "reference_sfc.mat"
    _require(path)
    import h5py

    with h5py.File(path, "r") as handle:
        return {k: np.asarray(handle[k]).ravel() for k in handle}


def _compare_sfc(mine, theirs):
    """Agreement over the finite entries, plus an identical NaN pattern."""
    mine = np.asarray(mine, dtype=np.float64).ravel()
    theirs = np.asarray(theirs, dtype=np.float64).ravel()

    np.testing.assert_array_equal(
        np.isfinite(mine), np.isfinite(theirs), err_msg="NaN patterns differ"
    )
    both = np.isfinite(mine) & np.isfinite(theirs)
    scale = np.abs(theirs[both]).max()
    return float(np.abs(mine[both] - theirs[both]).max() / scale)


def test_global_sfc_matches_matlab(legacy_matrices, reference_sfc):
    from sbci.coupling import global_coupling

    sc, fc = legacy_matrices
    relative = _compare_sfc(global_coupling(sc, fc), reference_sfc["sfc_gbl"])
    assert relative < SFC_TOLERANCE, f"global SFC differs by {relative:.3g} relative"


def test_discrete_sfc_matches_matlab(legacy_matrices, reference_sfc):
    from sbci.coupling import discrete_coupling

    sc, fc = legacy_matrices
    relative = _compare_sfc(discrete_coupling(sc, fc), reference_sfc["sfc_dct"])
    assert relative < SFC_TOLERANCE, f"discrete SFC differs by {relative:.3g} relative"


def test_local_sfc_matches_matlab(legacy_matrices, reference_sfc):
    from sbci.coupling import local_coupling

    sc, fc = legacy_matrices
    mine = local_coupling(sc, fc, reference_sfc["labels_used"])
    relative = _compare_sfc(mine, reference_sfc["sfc_loc"])
    assert relative < SFC_TOLERANCE, f"local SFC differs by {relative:.3g} relative"


# --- to_atlas and seed, against MATLAB rather than hand arithmetic ---------


@pytest.fixture(scope="module")
def legacy_sc():
    """The released smoothed SC, symmetrized with a zero diagonal."""
    import scipy.io

    path = TOOLKIT / "example_data/SBCI_Individual_Subject_Outcome/smoothed_sc_avg_0.005_ico4.mat"
    _require(path)
    sc = np.asarray(scipy.io.loadmat(path)["sc"], dtype=np.float64)
    sc = sc + sc.T
    np.fill_diagonal(sc, 0.0)
    return sc


@pytest.fixture(scope="module")
def grid_areas():
    """Vertex areas as the pipeline counts them, from the example subject."""
    from sbci import ContinuousConnectome

    path = (
        pathlib.Path(os.environ.get("SBCI_DERIVATIVES", "/work/users/x/y/xya/sbci-derivatives"))
        / "sub-example_sc.h5"
    )
    if not path.is_file():
        pytest.skip(f"{path} not built; run tools/import_legacy.py")
    return np.asarray(ContinuousConnectome.load(path).area, dtype=np.float64)


def test_to_atlas_matches_parcellate_sc(legacy_sc, grid_areas):
    """The full-scale check that the hand-computed toy case cannot give.

    Three conventions have to be lined up before the two are the same
    quantity, and each one is a real difference worth stating:

    - **Background regions.** MATLAB returns 70 rows. FreeSurfer-derived
      atlases carry a background entry per hemisphere, ``LH_missing`` at index
      0 and ``RH_missing`` at index 35, and the toolkit keeps both as regions.
      ``tools/convert_atlases.py`` folds all background to label 0 instead,
      which is why a Schaefer200 parcellation here is 200x200 and not 201x201.
    - **Triangle.** ``parcellate_sc.m`` loops over ``i < j`` and leaves the
      lower triangle at zero, so its output is not symmetric.
    - **Diagonal.** It also leaves the region diagonal at zero, where this
      package computes within-region connectivity. PORTING.md item 3 has the
      open question of which the released files should carry.

    With those three reconciled, ``how="mean"`` is the same calculation.
    """
    from sbci import load_atlas
    from sbci.parcellation import parcellate

    path = REFERENCE / "reference_parcellation.mat"
    _require(path)
    raw = _read(path, "result")
    assert raw.shape == (70, 70)

    keep = [i for i in range(70) if i not in (0, 35)]
    raw = raw[np.ix_(keep, keep)]
    assert np.abs(np.tril(raw, -1)).max() == 0.0, "expected an upper triangle"
    assert np.abs(np.diag(raw)).max() == 0.0, "expected a zero region diagonal"
    theirs = raw + raw.T

    atlas = load_atlas("Desikan")
    assert atlas.names[0] == "LH_bankssts" and atlas.names[-1] == "RH_insula"

    mine = parcellate(legacy_sc, atlas, grid_areas, how="mean")
    assert mine.shape == theirs.shape, f"{mine.shape} against {theirs.shape}"

    off = ~np.eye(len(keep), dtype=bool)
    relative = float(np.abs(mine[off] - theirs[off]).max() / np.abs(theirs[off]).max())
    assert relative < TOLERANCE, (
        f"to_atlas differs from parcellate_sc.m by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps), tolerance {TOLERANCE:.3g}"
    )
    assert np.corrcoef(mine[off], theirs[off])[0, 1] > 1 - 1e-14


def test_seed_rows_match_matlab(legacy_sc):
    """Seed profiles pulled out by MATLAB, at both ends of both hemispheres."""
    import h5py

    path = REFERENCE / "reference_seed.mat"
    _require(path)
    with h5py.File(path, "r") as handle:
        vertices = np.asarray(handle["vertices"]).ravel().astype(int)
        rows = np.asarray(handle["rows"]).T.astype(np.float64)
    if rows.shape[0] != vertices.size:
        rows = rows.T

    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed
    from sbci.metadata import template

    connectome = ContinuousConnectome(
        data=to_condensed(legacy_sc).astype(np.float32),
        area=np.ones(legacy_sc.shape[0]),
        mask=np.ones(legacy_sc.shape[0], dtype=bool),
        metadata=template("sc"),
    )

    for index, vertex in enumerate(vertices):
        mine = connectome.seed(vertex=int(vertex) - 1)  # MATLAB is one-based
        theirs = rows[index]
        scale = np.abs(theirs).max()
        relative = float(np.abs(mine - theirs).max() / scale)
        assert relative < TOLERANCE, (
            f"seed(vertex={vertex - 1}) differs by {relative:.3g} relative "
            f"({relative / FLOAT32_EPS:.1f} float32-eps)"
        )


def test_seed_region_marginal_matches_matlab(legacy_sc):
    """``seed(region=...)`` is the area-weighted mean over the region."""
    import h5py

    path = REFERENCE / "reference_seed.mat"
    _require(path)
    with h5py.File(path, "r") as handle:
        marginal = np.asarray(handle["marginal"]).ravel().astype(np.float64)
        mask = np.asarray(handle["mask"]).ravel().astype(bool)
        areas = np.asarray(handle["areas"]).ravel().astype(np.float64)

    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed
    from sbci.metadata import template

    connectome = ContinuousConnectome(
        data=to_condensed(legacy_sc).astype(np.float32),
        area=areas,
        mask=np.ones(legacy_sc.shape[0], dtype=bool),
        metadata=template("sc"),
    )
    mine = connectome.seed(region=mask)

    scale = np.abs(marginal).max()
    relative = float(np.abs(mine - marginal).max() / scale)
    assert relative < TOLERANCE, (
        f"seed(region=...) differs by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps), tolerance {TOLERANCE:.3g}"
    )


# --- ENCORE alignment ------------------------------------------------------

ENCORE_REFERENCE = (
    pathlib.Path(os.environ.get("SBCI_ENCORE_REFERENCE", "/work/users/x/y/xya/encore-ref"))
    / "encore_reference.mat"
)


@pytest.fixture(scope="module")
def encore_reference():
    """Intermediates from a MATLAB run of ConCon_Alignment on a small sphere."""
    import h5py

    _require(ENCORE_REFERENCE)
    with h5py.File(ENCORE_REFERENCE, "r") as handle:
        data = {key: np.asarray(handle[key]).T.astype(np.float64) for key in handle}
    return data


@pytest.fixture(scope="module")
def encore_grids(encore_reference):
    from sbci.alignment import SphericalGrid

    order = int(encore_reference["L"].ravel()[0])
    faces = encore_reference["grid_lh_T"].astype(int) - 1
    return (
        SphericalGrid(encore_reference["grid_lh_V"], faces, order),
        SphericalGrid(encore_reference["grid_rh_V"], faces, order),
    )


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("voronoi areas", "grid_lh_A"),
        ("tangent basis", "grid_lh_basis"),
        ("basis laplacian", "grid_lh_lap"),
    ],
)
def test_alignment_geometry_matches_matlab(encore_reference, encore_grids, name, key):
    """The grid, its areas and its harmonic basis, to float64 rounding."""
    grid = encore_grids[0]
    mine = {"grid_lh_A": grid.areas, "grid_lh_basis": grid.basis, "grid_lh_lap": grid.laplacian}[
        key
    ]
    theirs = encore_reference[key]
    if theirs.shape != mine.shape:
        theirs = theirs.reshape(mine.shape)
    relative = _relative(mine, np.asarray(theirs))
    assert relative < 1e-12, f"{name} differs by {relative:.3g}"


def test_alignment_exp_and_log_maps_match_matlab(encore_reference, encore_grids):
    from sbci.alignment import sphere_exp_map, sphere_log_map

    vertices = encore_grids[0].vertices
    gamma = encore_reference["gamma_test"]
    assert _relative(sphere_exp_map(vertices, gamma), encore_reference["exp_test"]) < 1e-12
    assert (
        _relative(
            sphere_log_map(vertices, encore_reference["exp_test"]), encore_reference["log_test"]
        )
        < 1e-12
    )


def test_alignment_barycentric_query_matches_libigl(encore_reference, encore_grids):
    """Closest-point queries, standing in for the reference's AABB tree."""
    from sbci.alignment import MeshQuery

    grid = encore_grids[0]
    weights, indices = MeshQuery(grid.vertices, grid.faces).query(encore_reference["query_pts"])
    reference_faces = encore_reference["bary_T_false"].astype(int) - 1
    same = (indices == reference_faces).all(axis=1)
    assert same.all(), f"different triangle for {(~same).sum()} points"
    assert _relative(weights, encore_reference["bary_V_false"]) < 1e-12


def test_alignment_template_matches_matlab(encore_reference, encore_grids):
    """The Karcher median, which involves no finite differences and is exact."""
    from sbci.alignment import Encore

    encore = Encore(*encore_grids, max_iterations=15, delta=1e-10)
    mine = encore.template(
        [encore_reference["F1"], encore_reference["F2"], encore_reference["F3"]],
        iterations=5,
    )
    assert _relative(mine, encore_reference["template"]) < 1e-12


def test_alignment_warp_composition_matches_matlab(encore_reference, encore_grids):
    """Where a warp sends each vertex, exactly; the Jacobian is checked apart."""
    from sbci.alignment import SphericalWarp

    warp = SphericalWarp(encore_grids[0], delta=1e-10)
    composed = warp.compose(encore_reference["disp_test"])
    assert _relative(composed.vertices, encore_reference["warp_V_composed"]) < 1e-12


def test_alignment_quantities_behind_the_finite_difference(encore_reference, encore_grids):
    """Everything downstream of the 1e-10 step, held to a looser bound.

    The reference's own derivative moves by 0.037 between adjacent step sizes
    and its Jacobian by 2.3e-04, so these cannot be reproduced to rounding by
    any independent implementation. What is asserted is that the port lands
    within that instability. See PORTING.md item 4.
    """
    from sbci.alignment import Concon, SphericalWarp

    lh_grid, rh_grid = encore_grids
    lh_warp = SphericalWarp(lh_grid, delta=1e-10).compose(encore_reference["disp_test"])
    rh_displacement = 0.02 * np.stack([np.cos(2 * rh_grid.theta), np.sin(3 * rh_grid.phi)], axis=1)
    rh_warp = SphericalWarp(rh_grid, delta=1e-10).compose(rh_displacement)

    assert _relative(lh_warp.jacobian, encore_reference["warp_J_composed"].ravel()) < 1e-2

    concon = Concon(lh_grid, rh_grid, delta=1e-10)
    assert (
        _relative(
            concon.evaluate(encore_reference["F_test"], lh_warp, rh_warp),
            encore_reference["evalF_test"],
        )
        < 1e-2
    )
    assert (
        _relative(
            concon.evaluate_root(encore_reference["Q_test"], lh_warp, rh_warp),
            encore_reference["evalQ_test"],
        )
        < 1e-2
    )


def test_alignment_registration_agrees_with_matlab(encore_reference, encore_grids):
    """The registered connectome, which inherits the derivative's instability."""
    from sbci.alignment import Encore

    encore = Encore(*encore_grids, step=0.05, max_iterations=15, threshold=1e-8, delta=1e-10)
    result, _, _, _ = encore.register(encore_reference["F1"], encore_reference["F2"])
    theirs = encore_reference["reg_result"]
    assert _relative(result, theirs) < 1e-2
    assert np.corrcoef(result.ravel(), theirs.ravel())[0, 1] > 0.9999


# --- FPCA reduction --------------------------------------------------------

FPCA_REFERENCE = pathlib.Path(os.environ.get("SBCI_FPCA_REFERENCE", "/work/users/x/y/xya/fpca-ref"))


@pytest.fixture(scope="module")
def fpca_reference():
    """A run of the published ConConBasis.Fit on a mesh basis.

    The reference cannot run as published: ``Fit`` reads ``auto_sparse``, which
    is never defined. The reference run applies a one-line fix binding it from
    the parsed options, and changes nothing else. See PORTING.md item 5.
    """
    import h5py

    inputs = FPCA_REFERENCE / "fpca_inputs.npz"
    output = FPCA_REFERENCE / "fpca_reference.mat"
    _require(inputs, output)

    with np.load(inputs) as data:
        Y, J, R = data["Y"], data["J_block"], data["R_block"]
    with h5py.File(output, "r") as handle:
        got = {
            key: np.asarray(handle[key]).T.astype(np.float64)
            for key in ("C0", "C1", "Smat", "scales", "residual_norms", "V0_RECORD")
        }
        got["K"] = int(np.asarray(handle["K"]).ravel()[0])

    zero = np.zeros_like(J)
    got["matrices"] = np.moveaxis(Y, 2, 0)
    got["gram"] = np.block([[J, zero], [zero, J]])
    got["roughness"] = np.block([[R, np.zeros_like(R)], [np.zeros_like(R), R]])
    got["basis"] = np.vstack([got["C0"], got["C1"]])
    return got


def test_reduction_matches_matlab(fpca_reference):
    """Basis, scales and explained variance, from the same initialization."""
    from sbci.reduction import fit_basis

    reference = fpca_reference
    result = fit_basis(
        reference["matrices"],
        reference["gram"],
        reference["roughness"],
        rank=reference["K"],
        alpha=1e-10,
        start=reference["V0_RECORD"],
    )

    relative = _relative(result.scales, reference["scales"].ravel())
    assert relative < TOLERANCE, (
        f"component scales differ by {relative:.3g} relative "
        f"({relative / FLOAT32_EPS:.1f} float32-eps)"
    )
    np.testing.assert_allclose(result.explained, reference["residual_norms"].ravel(), atol=1e-10)

    for k in range(reference["K"]):
        mine, theirs = result.basis[:, k], reference["basis"][:, k]
        sign = np.sign(mine @ theirs) or 1.0
        assert np.abs(sign * mine - theirs).max() < 1e-10, f"component {k + 1} differs"


def test_reduction_scores_match_matlab(fpca_reference):
    """The subject weights each component carries."""
    from sbci.reduction import fit_basis

    reference = fpca_reference
    result = fit_basis(
        reference["matrices"],
        reference["gram"],
        reference["roughness"],
        rank=reference["K"],
        alpha=1e-10,
        start=reference["V0_RECORD"],
    )
    for k in range(reference["K"]):
        mine, theirs = result.scores[:, k], reference["Smat"][:, k]
        sign = np.sign(mine @ theirs) or 1.0
        assert np.abs(sign * mine - theirs).max() < 1e-10, f"scores {k + 1} differ"
