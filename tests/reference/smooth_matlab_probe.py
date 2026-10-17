"""How closely smoothing reproduces the MATLAB reference, in numbers rather than a pass.

    python tests/reference/smooth_matlab_probe.py

Tier 4 asserts these against tolerances and prints nothing when they hold; this prints what it
compares, on the toolkit's example subject (PORTING.md item 24 quotes it): the Riemannian
kernel and the density ``smooth_endpoints`` builds from the endpoints, the ``smooth()`` method
on a file carrying them (with its diagonal and unit mass), and the bandwidth ``smooth()``
takes when given none. The paths are those of ``tests/test_matlab_reference.py``.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import test_matlab_reference as tier4  # noqa: E402
from sbci import ContinuousConnectome  # noqa: E402
from sbci.smoothing import (  # noqa: E402
    Endpoints,
    diffusion_kernel,
    kappa_candidates,
    load_eigenpairs,
    smooth_endpoints,
)

EPS = tier4.FLOAT32_EPS
basis = tier4.TOOLKIT / "concon_estimate"
(lam_l, u_l), (lam_r, u_r) = (
    load_eigenpairs(basis / f"EV_LBO_ds_ico4_{side}.mat", side) for side in "LR"
)
kappa = float(
    np.asarray(tier4._read(tier4.REFERENCE / "reference_kernel_L.mat", "kappa")).ravel()[0]
)


def compared(name, mine, theirs):
    relative = tier4._relative(mine, theirs)
    r = np.corrcoef(np.ravel(mine), np.ravel(theirs))[0, 1]
    print(f"{name}: {relative:.3g} relative ({relative / EPS:.2f} float32-eps), r = {r:.15f}")


compared(
    "kernel KM_L",
    diffusion_kernel(lam_l, u_l, kappa),
    tier4._read(tier4.REFERENCE / "reference_kernel_L.mat", "KM_L"),
)
density_mat = tier4._read(tier4.REFERENCE / "reference_density.mat", "DM")
ends = Endpoints.from_matlab(
    tier4.TOOLKIT / "example_data/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat"
)
compared(
    "density DM, smooth_endpoints",
    smooth_endpoints(
        ends, diffusion_kernel(lam_l, u_l, kappa), diffusion_kernel(lam_r, u_r, kappa)
    ),
    density_mat,
)

folder = pathlib.Path(
    tier4.os.environ.get("SBCI_EXAMPLE_SC", "/work/users/x/y/xya/sbci-derivatives")
)
for path in (folder / "sub-example_desc-withendpoints_sc.h5", folder / "sub-example_sc.h5"):
    if path.is_file() and ContinuousConnectome.load(path).endpoints is not None:
        connectome = ContinuousConnectome.load(path)
        break
area = np.asarray(connectome.area, dtype=np.float64)
expected = np.asarray(density_mat, dtype=np.float64)
np.fill_diagonal(expected, 0.0)
expected /= area @ expected @ area
smoothed = connectome.smooth(kernel="rdk", bandwidth=kappa, eigenpairs=basis).dense()
mine = np.asarray(smoothed, dtype=np.float64)
compared(f"smooth() on {path.name}", mine, expected)
print(
    f"  its diagonal at most {np.abs(np.diag(mine)).max():.3g}; unit mass {area @ mine @ area:.15f}"
)
chosen = kappa_candidates(lam_l)[3]
print(
    f"the bandwidth smooth() takes when given none: {chosen:.10g} against the reference's "
    f"{kappa:.10g}, {abs(chosen - kappa) / kappa:.3g} relative"
)
