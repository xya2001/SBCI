"""Check the released HCP Young Adult example cohort end to end, the way a user meets it.

    python scripts/check_hcp_ya.py [--out DIR]

Downloads the eleven subjects with ``sbci download hcp-ya`` into DIR (default
``./hcp-ya``; files already there and correct are kept), then runs the methods
the README and USAGE show on them and prints one line per check with what it
measured. The first check that does not hold stops the script with exit code 1.

It needs the plotting extra; with the render extra the surface figure is drawn
through PyVista as well. Under ten minutes on four cores (three to eight measured).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import sbci
from sbci.download import fetch_cohort, load_manifest

SEED = 1234
FAILURES = []


def check(name: str, passed: bool, detail: str) -> None:
    """Report one check; stop at the first failure."""
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}", flush=True)
    if not passed:
        FAILURES.append(name)
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("hcp-ya"))
    args = parser.parse_args(argv)

    # 1. the download, as a user runs it
    t = time.time()
    manifest = load_manifest("hcp-ya")
    paths = fetch_cohort(out=args.out, cohort="hcp-ya", report=lambda _line: None)
    rows = (args.out / "manifest.csv").read_text().splitlines()
    terms = args.out / "DATA_USE.txt"
    check(
        "download",
        len(paths) == len(manifest["subjects"]) == 11
        and rows[0] == "subject,sex,age_bin"
        and terms.exists()
        and "1U54MH091657" in terms.read_text(encoding="utf-8"),
        f"{len(paths)} files verified against their SHA-256 in {time.time() - t:.0f}s; "
        f"manifest.csv has {len(rows) - 1} subjects with sex and age band; "
        "DATA_USE.txt beside them with the HCP's terms and acknowledgment",
    )

    # 2. every file loads, validates, and carries positioned endpoints
    from sbci.validate import validate_file

    subjects = [sbci.load(p) for p in paths]
    valid = all(all(c.passed for c in validate_file(p)) for p in paths)
    counts = [s.endpoints.n_streamlines for s in subjects]
    recorded = [int(s.metadata.get("streamline_count")) for s in subjects]
    check(
        "load and validate",
        valid and all(s.endpoints.has_positions for s in subjects) and counts == recorded,
        f"all {len(paths)} pass sbci validate; {min(counts):,} to {max(counts):,} "
        "streamlines each, every endpoint with its triangle and barycentric position",
    )
    first = subjects[0]

    # 3. parcellation
    schaefer = np.asarray(first.to_atlas("Schaefer200"))
    desikan = np.asarray(first.to_atlas("Desikan"))
    check(
        "to_atlas",
        schaefer.shape == (200, 200)
        and desikan.shape == (68, 68)
        and np.isfinite(schaefer).all()
        and (schaefer >= 0).all(),
        f"Schaefer200 {schaefer.shape}, Desikan {desikan.shape}, finite and nonnegative",
    )

    # 4. a seed profile is the connectome's row, and two subjects agree as the cohorts do
    profile = first.seed(vertex=SEED)
    row = np.asarray(first.dense()[SEED], dtype=np.float64)
    between = float(np.corrcoef(np.asarray(first.data), np.asarray(subjects[1].data))[0, 1])
    check(
        "seed",
        profile.shape == (5124,) and (profile >= 0).all() and np.allclose(profile, row),
        f"profile of vertex {SEED} equals its row of the connectome; two subjects correlate at "
        f"r = {between:.2f}, as pairs of young adults do (0.67 to 0.73)",
    )
    check("frame", between > 0.5, f"subjects 1 and 2 correlate at r = {between:.2f}")

    # 5. re-smoothing from the stored endpoints gives the stored connectome back
    t = time.time()
    again = first.smooth(kernel="shk", mask_medial_wall=True)
    r = float(np.corrcoef(np.asarray(again.data), np.asarray(first.data))[0, 1])
    check("smooth", r > 0.999999, f"re-smoothed in {time.time() - t:.0f}s, r = {r:.7f}")

    # 6. a surface figure on fsaverage
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import pyvista  # noqa: F401

        engine = "pyvista"
    except ImportError:
        engine = "matplotlib"
    with tempfile.TemporaryDirectory() as scratch:
        figure = first.plot(profile / profile.max(), mesh="fsaverage", engine=engine)
        target = Path(scratch) / "seed.png"
        figure.savefig(target, dpi=80)
        plt.close(figure)
        check("plot", target.stat().st_size > 10_000, f"fsaverage figure through {engine}")

    # 7. ENCORE on two subjects: the template is estimated, both costs fall
    t = time.time()
    alignment = sbci.align(subjects[:2], max_iterations=4)
    falls = [trace[-1] < trace[0] for trace in alignment.traces]
    check(
        "align (ENCORE)",
        all(falls),
        f"Karcher-median template, costs "
        f"{[f'{tr[0]:.4f}->{tr[-1]:.4f}' for tr in alignment.traces]} in {time.time() - t:.0f}s",
    )

    # 8. ConSEAL on two subjects, onto the first
    t = time.time()
    registration = sbci.endpoints_align(subjects[:2], template=0, max_iterations=3, threshold=1e-7)
    moved = registration.costs[1]
    back = registration.aligned_endpoints(1)
    check(
        "endpoints_align (ConSEAL)",
        moved[-1] < moved[0] and back.n_streamlines == subjects[1].endpoints.n_streamlines,
        f"cost {moved[0]:.5f} -> {moved[-1]:.5f} in {len(moved) - 1} iterations, "
        f"{time.time() - t:.0f}s; endpoints handed back",
    )

    # 9. the ENCORE warp carried to fs_LR and to fsaverage
    with tempfile.TemporaryDirectory() as scratch:
        carried = sbci.migrate_warp(
            alignment.warps[1], to="fs_LR_32k", grid_rotations=alignment.grid_rotations
        )
        on_sphere = np.allclose(np.linalg.norm(carried.lh_vertices, axis=1), 1.0, atol=1e-6)
        written = carried.to_gifti(Path(scratch) / "warp")
        check(
            "migrate_warp",
            carried.lh_vertices.shape == (32492, 3)
            and on_sphere
            and all(p.exists() for p in written),
            "ENCORE warp restated on fs_LR 32k, 32,492 vertices per hemisphere, written as GIFTI",
        )

    # 10. FPCA of the cohort and a local test against sex
    t = time.time()
    reduction = sbci.reduce(subjects, rank=4)
    female = np.array([line.split(",")[1] == "F" for line in rows[1:]], dtype=float)
    result = sbci.local_test(reduction.scores, female)
    check(
        "reduce and local_test",
        reduction.scores.shape == (len(subjects), 4)
        and 0 < reduction.explained[-1] < 1
        and result.adjusted.shape == (4,),
        f"rank 4 explains {reduction.explained[-1]:.3f} in {time.time() - t:.0f}s; "
        f"sex tested per component, adjusted p {np.round(result.adjusted, 3).tolist()}",
    )
    print("all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
