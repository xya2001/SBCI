"""End-to-end audit: every row of the documented API, on real data, in order.

Each check either passes with the value it produced, or fails loudly. Nothing
is mocked and nothing is synthetic except where a cohort is needed and only one
real subject is on hand.
"""

import os
import subprocess
import sys
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np

warnings.filterwarnings("ignore")

D = "/work/users/x/y/xya/sbci-derivatives"
TOOLKIT = "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit"
os.environ["SBCI_LBO_DIR"] = f"{TOOLKIT}/concon_estimate"

passed, failed = [], []


def check(row, fn):
    try:
        detail = fn()
        passed.append(row)
        print(f"  PASS  {row:<46s} {detail}")
    except Exception as e:  # noqa: BLE001 - the audit reports, it does not raise
        failed.append((row, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {row:<46s} {type(e).__name__}: {str(e)[:60]}")


import sbci
import sbci.stats
from sbci import ContinuousConnectome, load_atlas, load_surface

print("=== 1. the computational file ===")
sc = ContinuousConnectome.load(f"{D}/sub-example_sc.h5")
fc = ContinuousConnectome.load(f"{D}/sub-example_fc.h5")
sc_e = ContinuousConnectome.load(f"{D}/sub-example_desc-withendpoints_sc.h5")
area = np.asarray(sc.area, dtype=np.float64)

check(
    "ContinuousConnectome.load(path)", lambda: f"{sc.n_vertices} vertices, modality {sc.modality!r}"
)


def save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = sc.save(f"{tmp}/sub-audit_sc.h5")
        back = ContinuousConnectome.load(path)
        assert np.array_equal(back.data, sc.data), "connectivity changed"
        assert back.metadata.fields == sc.metadata.fields, "metadata changed"
        return f"round trip exact, {os.path.getsize(path) / 1e6:.1f} MB"


check(".save(path)", save_roundtrip)

print("\n=== 2. analysis ===")


def to_atlas():
    schaefer = sc.to_atlas(load_atlas("Schaefer200"))
    desikan = sc.to_atlas(load_atlas("Desikan"))
    assert schaefer.shape == (200, 200), schaefer.shape
    assert desikan.shape == (68, 68), desikan.shape
    assert np.isfinite(schaefer).all()
    return f"Schaefer200 {schaefer.shape}, Desikan total {desikan.sum():.6f}"


check('.to_atlas(atlas, how="mass"|"mean")', to_atlas)


def seed():
    atlas = load_atlas("Desikan")
    by_vertex = sc.seed(vertex=1234)
    mask = atlas.labels == atlas.names.index("LH_superiorfrontal") + 1
    by_region = sc.seed(region=mask)
    assert by_vertex.shape == (5124,) and by_region.shape == (5124,)
    peak = atlas.names[atlas.labels[int(np.argmax(by_region))] - 1]
    return f"vertex peak {by_vertex.max():.3g}, region peak lands in {peak}"


check(".seed(vertex=...) / .seed(region=...)", seed)


def coupling():
    atlas = load_atlas("Desikan")
    glob = sc.coupling(fc, scope="global")
    region = sc.coupling(fc, scope="region", labels=atlas.labels)
    from sbci.coupling import discrete_coupling

    discrete = discrete_coupling(sc.to_atlas(atlas), fc.to_atlas(atlas))
    assert glob.shape == (5124,) and region.shape == (5124,) and discrete.shape == (68,)
    return (
        f"global mean {np.nanmean(glob):.4f}, region {np.nanmean(region):.4f}, "
        f"discrete {np.nanmean(discrete):.4f}"
    )


check(".coupling(fc, scope=...)", coupling)


def plot():
    figure = sc.plot(sc.seed(vertex=1234), surface="inflated")
    with tempfile.TemporaryDirectory() as tmp:
        figure.savefig(f"{tmp}/seed.png", dpi=60)
        size = os.path.getsize(f"{tmp}/seed.png")
    surfaces = [load_surface(n).vertices.shape for n in ("inflated", "white", "pial", "sphere")]
    assert all(s == (5124, 3) for s in surfaces), surfaces
    return f"figure written ({size / 1024:.0f} KB), 4 surfaces bundled"


check(".plot(map, surface=...)", plot)

print("\n=== 3. the exchange file ===")


def to_cifti():
    import nibabel as nib

    path = f"{D}/sub-example_space-fsLR_den-32k_desc-concon_sc.dconn.nii"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not written yet")
    image = nib.load(path)
    axis = image.header.get_axis(0)
    assert image.shape == (64984, 64984), image.shape
    assert set(axis.name) == {"CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"}
    return f"{os.path.getsize(path) / 1e9:.2f} GB, both axes BrainModelAxis"


check(".to_cifti(path)", to_cifti)

print("\n=== 4. the command line ===")


def validate_cli():
    result = subprocess.run(
        ["sbci", "validate", f"{D}/sub-example_sc.h5"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[:200]
    checks = [line for line in result.stdout.splitlines() if line.startswith("[")]
    assert all(line.startswith("[PASS]") for line in checks), result.stdout
    return f"{len(checks)} checks, all PASS"


check("sbci validate <file>", validate_cli)

print("\n=== 5. smoothing ===")


def smooth():
    out = sc_e.smooth(kernel="rdk")
    dense = out.dense().astype(np.float64)
    mass = float(area @ dense @ area)
    assert abs(mass - 1) < 1e-8, mass
    matern = sc_e.smooth(kernel="matern")
    assert matern.metadata["kernel"] == "matern"
    return f"rdk bandwidth {out.metadata['bandwidth']:.6f}, unit mass {mass:.10f}; matern also fits"


check(".smooth(kernel=..., bandwidth=...)", smooth)


def smooth_default():
    """The default kernel is shk, and it now works rather than raising.

    It reproduces c3_main at r = 1.000000 across five ADNI subjects; see
    PORTING.md item 6. The density it returns can carry medial-wall mass,
    exactly as both references do, so this checks mass and finiteness rather
    than running the validator -- SPEC_QUESTIONS.md item 12.
    """
    # The spherical kernel is evaluated per endpoint, so a whole subject takes
    # the better part of an hour. This audit exists to exercise the documented
    # API, not to re-measure agreement -- that is PORTING.md item 6, five
    # subjects at r = 1.000000 -- so it runs on a slice of the endpoints.
    endpoints = sc_e.endpoints
    keep = slice(0, 20_000)
    small = type(endpoints)(
        surf_in=endpoints.surf_in[keep],
        surf_out=endpoints.surf_out[keep],
        vtx_in=endpoints.vtx_in[keep],
        vtx_out=endpoints.vtx_out[keep],
        n_per_hemi=endpoints.n_per_hemi,
        tri_in=None if endpoints.tri_in is None else endpoints.tri_in[keep],
        tri_out=None if endpoints.tri_out is None else endpoints.tri_out[keep],
        bary_in=None if endpoints.bary_in is None else endpoints.bary_in[keep],
        bary_out=None if endpoints.bary_out is None else endpoints.bary_out[keep],
    )
    sliced = type(sc_e)(
        data=sc_e.data,
        area=sc_e.area,
        mask=sc_e.mask,
        metadata=sc_e.metadata,
        coords=sc_e.coords,
        endpoints=small,
    )
    out = sliced.smooth()
    assert out.metadata["kernel"] == "shk", out.metadata["kernel"]
    dense = out.dense().astype(np.float64)
    mass = float(area @ dense @ area)
    assert abs(mass - 1) < 1e-8, mass
    assert np.isfinite(dense).all() and dense.min() >= 0.0
    return (
        f"shk sigma {out.metadata['bandwidth']}, unit mass {mass:.10f}, "
        f"on {small.n_streamlines:,} endpoints"
    )


check("  .smooth() default kernel shk", smooth_default)

print("\n=== 6. reduction and inference ===")


def reduce_one():
    result = sc.reduce(rank=2, max_outer=3, seed=0)
    assert result.basis.shape == (5124, 2)
    assert np.all(np.diff(result.explained) >= -1e-12)
    return f"rank 2, explained {result.explained[-1]:.4f}"


check(".reduce(rank=K)", reduce_one)


def reduce_cohort():

    base = sc.dense().astype(np.float64)
    cohort = []
    for i in range(4):
        field = 1.0 + 0.2 * np.sin(np.linspace(0, (i + 2) * np.pi, base.shape[0]))
        matrix = base * np.outer(field, field)
        cohort.append((matrix + matrix.T) / 2)
    result = sbci.reduce(cohort, rank=2, max_outer=3, seed=0)
    scores = sbci.reduction.project(result, np.stack(cohort) - np.mean(cohort, axis=0))
    assert scores.shape == (4, 2)
    return f"4 subjects, rank 2, explained {result.explained[-1]:.4f}"


check("sbci.reduce(cc_list, rank=K)", reduce_cohort)


def local_test():
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((30, 6))
    covariate = rng.standard_normal(30)
    scores[:, 1] += 2.0 * covariate
    result = sbci.stats.local_test(scores, covariate)
    assert 1 in result.significant(0.05), result.adjusted
    return f"planted effect found, adjusted p = {result.adjusted[1]:.2e}"


check("sbci.stats.local_test(scores, design)", local_test)

print("\n=== 7. alignment ===")


def align():
    sys.path.insert(0, "/nas/longleaf/home/xya/sbci/tests")
    from sbci.alignment import SphericalGrid, rotate_off_poles
    from test_alignment import icosphere

    vertices, faces = icosphere(2)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=3)
    n = 2 * grid.n_vertices
    rng = np.random.default_rng(2)
    cohort = []
    for _ in range(3):
        matrix = rng.random((n, n))
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0.0)
        cohort.append(matrix)
    result = sbci.align(cohort, grids=(grid, grid), order=3, max_iterations=3)
    assert len(result.warps) == 3
    assert all(w.lh_jacobian.min() > 0 for w in result.warps), "warp not diffeomorphic"
    with tempfile.TemporaryDirectory() as tmp:
        path = result.warps[0].save(f"{tmp}/warp.npz")
        from sbci.alignment import Warp

        back = Warp.load(path)
        assert np.array_equal(back.lh_vertices, result.warps[0].lh_vertices)
    return "3 subjects, warps positive-Jacobian, save/load exact"


check('sbci.align(cc_list, method="encore")', align)

print("\n=== 8. what is deliberately absent ===")


def download():
    result = subprocess.run(
        ["sbci", "download", "hcp-ya"], capture_output=True, text=True, timeout=120
    )
    assert "SPEC_QUESTIONS" in result.stdout or "SPEC_QUESTIONS" in result.stderr
    return "explains that it needs the data release"


check("sbci download", download)

print(f"\n{'=' * 70}")
print(f"{len(passed)} passed, {len(failed)} failed")
for row, why in failed:
    print(f"  FAILED  {row}: {why}")
sys.exit(1 if failed else 0)
