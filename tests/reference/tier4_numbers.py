"""Print the agreement of each Tier 4 comparison, which the tests only bound.

    python tests/reference/tier4_numbers.py

``tests/test_matlab_reference.py`` asserts each comparison against a bound
(1e-12 for float64 rounding); README.md and PORTING.md quote what is measured
inside it. This prints those measurements, from the same inputs and the same
helpers: the parcellation and the three coupling forms against the MATLAB
functions, ENCORE's quantities behind the reference's 1e-10 finite
difference, and the FPCA at the reference's fixed point and at its default
tolerance. It needs what Tier 4 needs (VERIFICATION.md), and a minute.
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import test_matlab_reference as tier4  # noqa: E402

EPS = np.finfo(np.float64).eps


def coupling_and_parcellation() -> None:
    import h5py
    import scipy.io

    from sbci import ContinuousConnectome, load_atlas
    from sbci.coupling import discrete_coupling, global_coupling, local_coupling
    from sbci.parcellation import parcellate

    sub = tier4.TOOLKIT / "example_data/SBCI_Individual_Subject_Outcome"
    sc = scipy.io.loadmat(sub / "smoothed_sc_avg_0.005_ico4.mat")["sc"]
    fc = scipy.io.loadmat(sub / "fc_avg_ico4.mat")["fc"]
    sc, fc = sc + sc.T, fc + fc.T
    np.fill_diagonal(sc, 0.0)
    np.fill_diagonal(fc, 0.0)
    with h5py.File(tier4.REFERENCE / "reference_sfc.mat", "r") as handle:
        reference = {k: np.asarray(handle[k]).ravel() for k in handle}
    for name, mine, key in (
        ("global", global_coupling(sc, fc), "sfc_gbl"),
        ("discrete", discrete_coupling(sc, fc), "sfc_dct"),
        ("region", local_coupling(sc, fc, reference["labels_used"]), "sfc_loc"),
    ):
        relative = tier4._compare_sfc(mine, reference[key])
        print(f"coupling, {name}: {relative:.3g} of the largest value ({relative / EPS:.1f} eps)")

    raw = tier4._read(tier4.REFERENCE / "reference_parcellation.mat", "result")
    keep = [i for i in range(70) if i not in (0, 35)]
    raw = raw[np.ix_(keep, keep)]
    theirs = raw + raw.T
    derivatives = pathlib.Path(
        os.environ.get("SBCI_DERIVATIVES", "/work/users/x/y/xya/sbci-derivatives")
    )
    area = np.asarray(ContinuousConnectome.load(derivatives / "sub-example_sc.h5").area, np.float64)
    legacy = np.asarray(sc, dtype=np.float64)
    mine = parcellate(legacy, load_atlas("Desikan"), area, how="mean")
    off = ~np.eye(len(keep), dtype=bool)
    relative = float(np.abs(mine[off] - theirs[off]).max() / np.abs(theirs[off]).max())
    print(f"to_atlas against parcellate_sc.m: {relative:.4g} of the largest entry")


def encore() -> None:
    import h5py

    from sbci.alignment import Concon, SphericalGrid, SphericalWarp

    with h5py.File(tier4.ENCORE_REFERENCE, "r") as handle:
        reference = {key: np.asarray(handle[key]).T.astype(np.float64) for key in handle}
    order = int(reference["L"].ravel()[0])
    faces = reference["grid_lh_T"].astype(int) - 1
    lh_grid = SphericalGrid(reference["grid_lh_V"], faces, order)
    rh_grid = SphericalGrid(reference["grid_rh_V"], faces, order)
    lh_warp = tier4._uncalibrated(
        SphericalWarp(lh_grid, delta=1e-10).compose(reference["disp_test"])
    )
    shift = 0.02 * np.stack([np.cos(2 * rh_grid.theta), np.sin(3 * rh_grid.phi)], axis=1)
    rh_warp = tier4._uncalibrated(SphericalWarp(rh_grid, delta=1e-10).compose(shift))
    concon = Concon(lh_grid, rh_grid, delta=1e-10)
    theirs = reference["evalQ_test"]
    theirs = theirs / np.sqrt((theirs**2 * concon.area_product).sum())
    jacobian = tier4._relative(lh_warp.jacobian, reference["warp_J_composed"].ravel())
    density = concon.evaluate(reference["F_test"], lh_warp, rh_warp)
    root = concon.evaluate_root(reference["Q_test"], lh_warp, rh_warp)
    for name, value in (
        ("the composed warp's Jacobian", jacobian),
        ("the transported density", tier4._relative(density, reference["evalF_test"])),
        ("the transported square root", tier4._relative(root, theirs)),
    ):
        print(f"ENCORE behind the 1e-10 finite difference, {name}: {value:.3g}")


def fpca() -> None:
    fixed = tier4._fpca_run("fpca_reference_fixed_point.mat")
    result = tier4._fit_like(fixed, max_outer=400, tol_outer=-1.0)
    rank = fixed["K"]
    basis = max(tier4._differs(result.basis[:, k], fixed["basis"][:, k]) for k in range(rank))
    scores = max(tier4._differs(result.scores[:, k], fixed["Smat"][:, k]) for k in range(rank))
    scales = tier4._relative(result.scales, fixed["scales"].ravel())
    explained = float(np.abs(result.explained - fixed["residual_norms"].ravel()).max())
    print(
        f"FPCA at the fixed point, all {rank} components: basis {basis:.3g}, scores "
        f"{scores:.3g}, scales {scales:.3g} relative, explained {explained:.3g}"
    )
    default = tier4._fpca_run("fpca_reference.mat")
    result = tier4._fit_like(default)
    first = tier4._differs(result.basis[:, 0], default["basis"][:, 0])
    later = [tier4._differs(result.basis[:, k], default["basis"][:, k]) for k in range(1, rank)]
    print(
        f"FPCA at the reference's default tolerance: first vector {first:.3g}; later vectors "
        f"{min(later):.2g} to {max(later):.2g}"
    )


if __name__ == "__main__":
    coupling_and_parcellation()
    encore()
    fpca()
