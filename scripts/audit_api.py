"""End-to-end audit: every row of the documented API, on the released subjects, in order.

    python scripts/audit_api.py            # a batch job: eight cores, 32 GB, about seven minutes

Each check either passes with the value it produced, or fails loudly; a check
that needs data the release does not have is reported as skipped, with the
reason. The subjects are the HCP Young Adult example cohort as
``sbci download hcp-ya`` writes it (D below is the lab's copy), with the FC
``tools/build_hcp_fc.py`` builds from the HCP's resting state. Nothing is
mocked; the one synthetic input is the planted effect that checks
``local_test`` against a known answer.
"""

import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

warnings.filterwarnings("ignore")

D = os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data")  # sbci download hcp-ya
EXCHANGE = "/work/users/x/y/xya/hcp-ya/exchange"
TOOLKIT = os.environ.get("SBCI_TOOLKIT", "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit")
os.environ["SBCI_LBO_DIR"] = f"{TOOLKIT}/concon_estimate"
FIRST = ["sub-100307", "sub-103010", "sub-108121", "sub-116221"]

passed, failed, skipped = [], [], []


def check(row, fn):
    try:
        detail = fn()
        passed.append(row)
        print(f"  PASS  {row:<46s} {detail}")
    except Exception as e:  # noqa: BLE001 - the audit reports, it does not raise
        failed.append((row, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {row:<46s} {type(e).__name__}: {str(e)[:60]}")


def skip(row, why):
    skipped.append((row, why))
    print(f"  SKIP  {row:<46s} {why}")


import sbci
import sbci.stats
from sbci import ContinuousConnectome, load_atlas, load_surface

print("=== 1. the computational file ===")
sc = ContinuousConnectome.load(f"{D}/sub-100307_sc.h5")
area = np.asarray(sc.area, dtype=np.float64)

check(
    "ContinuousConnectome.load(path)",
    lambda: (
        f"{sc.n_vertices} vertices, modality {sc.modality!r}, "
        f"{sc.endpoints.n_streamlines:,} streamlines"
    ),
)


def save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = sc.save(f"{tmp}/sub-audit_sc.h5")
        back = ContinuousConnectome.load(path)
        assert np.array_equal(back.data, sc.data), "connectivity changed"
        assert back.metadata.fields == sc.metadata.fields, "metadata changed"
        assert back.endpoints.n_streamlines == sc.endpoints.n_streamlines, "endpoints changed"
        return f"round trip exact, {os.path.getsize(path) / 1e6:.1f} MB with endpoints"


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
    fc = ContinuousConnectome.load(f"{D}/sub-100307_fc.h5")
    desikan = load_atlas("Desikan")
    whole = sc.coupling(fc, scope="global")
    local = sc.coupling(fc, scope="region", labels=desikan.labels)
    wall = int((~np.asarray(sc.mask, dtype=bool)).sum())
    assert whole.shape == (5124,), whole.shape
    assert int(np.isnan(whole).sum()) == wall, "NaN beyond the medial wall"
    assert np.nanmax(np.abs(whole)) <= 1 and np.nanmax(np.abs(local)) <= 1 + 1e-12
    assert np.nanmean(local) > np.nanmean(whole), "local coupling should run higher than global"
    return (
        f"global mean {np.nanmean(whole):.3f}, within Desikan regions {np.nanmean(local):.3f}; "
        f"NaN on the {wall} medial-wall vertices only"
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

    path = f"{EXCHANGE}/sub-100307_space-fsLR_den-32k_desc-concon_sc.dconn.nii"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not written yet: run scripts/write_exchange_file.py")
    image = nib.load(path)
    axis = image.header.get_axis(0)
    assert image.shape == (64984, 64984), image.shape
    assert set(axis.name) == {"CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"}
    return f"{os.path.getsize(path) / 1e9:.2f} GB, both axes BrainModelAxis"


check(".to_cifti(path)", to_cifti)

print("\n=== 4. the command line ===")


def validate_cli():
    result = subprocess.run(
        ["sbci", "validate", f"{D}/sub-100307_sc.h5"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[:200]
    checks = [line for line in result.stdout.splitlines() if line.startswith("[")]
    assert all(line.startswith("[PASS]") for line in checks), result.stdout
    return f"{len(checks)} checks, all PASS"


check("sbci validate <file>", validate_cli)


def download():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["sbci", "download", "hcp-ya", "--subject", "100307"],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=tmp,
        )
        assert result.returncode == 0, result.stderr[:200]
        here = Path(tmp)
        got = ContinuousConnectome.load(here / "sub-100307_sc.h5")
        assert np.array_equal(got.data, sc.data), "the download differs from the lab's copy"
        assert (here / "DATA_USE.txt").exists() and (here / "manifest.csv").exists()
    return "sub-100307 fetched into the current directory, verified, terms beside it"


check("sbci download hcp-ya", download)

print("\n=== 5. smoothing ===")


def smooth():
    out = sc.smooth(kernel="rdk")
    dense = out.dense().astype(np.float64)
    mass = float(area @ dense @ area)
    assert abs(mass - 1) < 1e-8, mass
    matern = sc.smooth(kernel="matern")
    assert matern.metadata["kernel"] == "matern"
    return f"rdk bandwidth {out.metadata['bandwidth']:.6f}, unit mass {mass:.10f}; matern also fits"


check(".smooth(kernel=..., bandwidth=...)", smooth)


def smooth_default():
    """The default kernel, shk, gives back the stored connectome from the stored endpoints."""
    out = sc.smooth(mask_medial_wall=True)
    assert out.metadata["kernel"] == "shk", out.metadata["kernel"]
    r = float(np.corrcoef(np.asarray(out.data), np.asarray(sc.data))[0, 1])
    assert r > 0.999999, r
    return f"shk sigma {out.metadata['bandwidth']}, r = {r:.7f} with the stored connectome"


check("  .smooth() default kernel shk", smooth_default)

print("\n=== 6. reduction and inference ===")


def reduce_one():
    result = sc.reduce(rank=2, max_outer=3, seed=0)
    assert result.basis.shape == (5124, 2)
    assert np.all(np.diff(result.explained) >= -1e-12)
    return f"rank 2, explained {result.explained[-1]:.4f}"


check(".reduce(rank=K)", reduce_one)


def reduce_cohort():
    paths = [f"{D}/{s}_sc.h5" for s in FIRST]
    result = sbci.reduce(paths, rank=2, max_outer=3, seed=0)  # read one file at a time
    assert result.scores.shape == (len(paths), 2)
    return f"{len(paths)} subjects from their files, rank 2, explained {result.explained[-1]:.4f}"


check("sbci.reduce(cc_list, rank=K)", reduce_cohort)


def local_test():
    # A planted effect, so the answer is known: the one synthetic input here.
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((30, 6))
    covariate = rng.standard_normal(30)
    scores[:, 1] += 2.0 * covariate
    result = sbci.stats.local_test(scores, covariate)
    assert 1 in result.significant(0.05), result.adjusted
    return f"planted effect found, adjusted p = {result.adjusted[1]:.2e}"


check("sbci.stats.local_test(scores, design)", local_test)

print("\n=== 7. alignment ===")
pair = [ContinuousConnectome.load(f"{D}/{s}_sc.h5") for s in FIRST[:2]]
alignment = None


def align():
    global alignment
    alignment = sbci.align(pair, max_iterations=3)
    assert len(alignment.warps) == 2
    assert all(w.lh_jacobian.min() > 0 for w in alignment.warps), "warp not diffeomorphic"
    assert all(trace[-1] < trace[0] for trace in alignment.traces), "a cost did not fall"
    with tempfile.TemporaryDirectory() as tmp:
        path = alignment.warps[1].save(f"{tmp}/warp.npz")
        from sbci.alignment import Warp

        back = Warp.load(path)
        assert np.array_equal(back.lh_vertices, alignment.warps[1].lh_vertices)
    return "2 subjects on ico4, costs fall, warps positive-Jacobian, save/load exact"


check("sbci.align(cc_list, template=None)", align)


def endpoints_align():
    # Moving onto fixed: the fixed subject itself is the template, and stays where it is.
    result = sbci.endpoints_align([pair[1]], template=pair[0], max_iterations=3, threshold=1e-7)
    costs = result.costs[0]
    assert costs[-1] < costs[0], costs
    back = result.aligned_endpoints(0)
    assert back.n_streamlines == pair[1].endpoints.n_streamlines
    return f"cost {costs[0]:.5f} -> {costs[-1]:.5f}, endpoints handed back"


check("sbci.endpoints_align([moving], template=fixed)", endpoints_align)


def migrate():
    if alignment is None:
        raise RuntimeError("the ENCORE row failed, so there is no warp to carry")
    carried = sbci.migrate_warp(
        alignment.warps[1], to="fs_LR_32k", grid_rotations=alignment.grid_rotations
    )
    assert carried.lh_vertices.shape == (32492, 3)
    assert np.allclose(np.linalg.norm(carried.lh_vertices, axis=1), 1.0, atol=1e-6)
    return "ENCORE warp restated on fs_LR 32k, 32,492 vertices per hemisphere"


check('sbci.migrate_warp(warp, to="fs_LR_32k")', migrate)

print("\n=== 8. a cohort, held-out scores, designs, and test-retest ===")


def cohort():
    loaded = sbci.load_cohort(D, modalities=("sc",))
    assert len(loaded.subjects) >= len(FIRST), loaded.summary()
    return loaded.summary().splitlines()[0]


check("sbci.load_cohort(folder)", cohort)


def project():
    fit = sbci.reduce([f"{D}/{s}_sc.h5" for s in FIRST[:3]], rank=2, max_outer=3, seed=0)
    held_out = sbci.project(fit, [f"{D}/{FIRST[3]}_sc.h5"])
    assert held_out.shape == (1, 2) and np.isfinite(held_out).all(), held_out
    return f"{FIRST[3]} scored on three others' basis: {np.round(held_out[0], 4).tolist()}"


check("sbci.project(fit, held_out)", project)


def design_and_contrast():
    # A planted site effect, so the answer is known.
    rng = np.random.default_rng(2)
    site = np.repeat(["a", "b", "c", "d"], 10)
    named = sbci.stats.design({"age": rng.uniform(22, 36, 40), "site": site})
    scores = rng.standard_normal((40, 3))
    scores[:, 0] += 1.5 * (site == "c")
    by_name = sbci.stats.local_test(scores, named, terms=["site"])
    contrast = sbci.stats.local_test(scores, named, contrast={"site[c]": 1, "site[b]": -1})
    assert by_name.pvalue[0] < 0.01 and contrast.pvalue[0] < 0.01, (by_name.pvalue, contrast.pvalue)
    low, high = contrast.interval()
    return (
        f"site by name p = {by_name.pvalue[0]:.1e}; c less b {contrast.estimate[0, 0]:.2f} "
        f"({low[0, 0]:.2f} to {high[0, 0]:.2f})"
    )


check("sbci.stats.design(...), terms= by name, contrast=", design_and_contrast)


def test_retest():
    from sbci.stats import icc, identification

    halves = ([], [])
    for subject in FIRST[:3]:
        connectome = ContinuousConnectome.load(f"{D}/{subject}_sc.h5")
        endpoints = connectome.endpoints
        first = (
            np.random.default_rng(0).permutation(endpoints.n_streamlines)
            < endpoints.n_streamlines // 2
        )
        for half, chosen in zip(halves, (first, ~first), strict=True):
            holder = ContinuousConnectome(
                connectome.data,
                connectome.area,
                connectome.mask,
                connectome.metadata,
                endpoints=endpoints.take(chosen),
            )
            half.append(holder.smooth(kernel="shk", bandwidth=0.005, mask_medial_wall=True).data)
    found = identification(np.vstack(halves[0]), np.vstack(halves[1]))
    sample = np.random.default_rng(1).choice(halves[0][0].size, 20_000, replace=False)
    reliability = icc([np.vstack(h)[:, sample] for h in halves])
    assert found.accuracy == (1.0, 1.0), found.accuracy
    return (
        f"3 subjects' halves identified {found.accuracy}; median ICC of 20,000 pairs "
        f"{np.nanmedian(reliability):.2f}"
    )


check("endpoints.take, sbci.stats.identification, icc", test_retest)


def export():
    with tempfile.TemporaryDirectory() as tmp:
        strength = sc.dense().sum(axis=1)
        written = sbci.save_map({"strength": strength}, f"{tmp}/strength.dscalar.nii")
        table = sbci.save_regions(
            sbci.region_means(strength, "Desikan"), "Desikan", f"{tmp}/regions.csv"
        )
        assert all(os.path.getsize(p) > 0 for p in (*written, table))
    return "a vertex map as CIFTI, its Desikan means as a table"


check("sbci.save_map, region_means, save_regions", export)

print(f"\n{'=' * 70}")
print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
for row, why in failed:
    print(f"  FAILED   {row}: {why}")
for row, why in skipped:
    print(f"  SKIPPED  {row}: {why}")
sys.exit(1 if failed else 0)
