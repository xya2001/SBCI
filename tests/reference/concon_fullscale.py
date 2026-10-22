"""Run concon on a whole released subject and compare it with the package.

`concon_probe.py` measures the kernel on single streamlines. This runs
`c3_main` at full scale: it writes the stored endpoints of one released SC
file as the input the binary takes, `subject_xing_sphere_avg_coords.tsv`, runs
`Compute_Kernel` with the pipeline's flags (`sbci_step5_structural.sh`), and
compares what the binary writes with the package's density from the same
positions, both through the lookup tables the package reproduces by default and
through the closed-form series they approximate. It also correlates the
binary's output with the released file, which the package smoothed with the
medial wall masked.

The binary never writes the grid's south pole, the vertex at (0, 0, -1) in each
hemisphere: `compute_kernel.cpp` looks vertices up in a quad tree over the
polar angle, clipping the range at pi, and `quad_tree.cpp`'s range test is
half-open, `x < max_x`, so a vertex at exactly pi is never found. The package
has no such hole, so each comparison is printed twice, as computed and with
the pole emptied as the binary leaves it.

What is left then is the binary's own inaccuracy, put back test-side as the
MATLAB references' errors are: `subject.cpp` reads the positions with
`fscanf("%f")` into `float`, and its harmonics come from Google's
spherical-harmonics library, whose closed forms for l <= 4 carry six-digit
constants where the package uses exact ones. With the positions cast to
float32, and then the constants as well, the package reproduces the binary to
machine precision (PORTING.md item 41).

Run by hand: it needs the binary (`CONCON_C3_MAIN`) and the pipeline grid's
`.m` files (`SBCI_AVE`), as `concon_probe.py` names them, neither of which is
in this repository (PORTING.md items 6 and 39). The positions are written with
nine decimals where the pipeline's `intersections_to_sphere.py` writes six, so
that both sides smooth the same points; `--reuse` skips the binary when its
input and output are already in the work folder, which an earlier run left.

    python tests/reference/concon_fullscale.py hcp-ya/sub-100307_sc.h5 /tmp/concon-full
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concon_probe import AVE, C3, RECORD, read_m_grid  # noqa: E402

import sbci  # noqa: E402
from sbci import smoothing  # noqa: E402
from sbci.grid import hemisphere_labels  # noqa: E402
from sbci.smoothing import (  # noqa: E402
    KernelTable,
    _grid_vertices,
    apply_final_threshold,
    endpoint_density,
    endpoint_positions,
    read_concon_endpoints,
)

SIGMA = 0.005
BANDS = ((0, 2), (2, 6), (6, 12), (12, 25), (25, 60), (60, 180))

# The constants of sh::EvalSH's closed forms for l <= 4, m = 0, in Google's
# spherical-harmonics library (sh/spherical_harmonics.cc), six significant digits where
# sqrt((2l+1)/4pi) times the Legendre polynomial's leading factor has more: 0.2820948,
# 0.4886025, 0.3153916, 0.3731763, 0.1057855.
SIX_DIGIT_CONSTANTS = {0: 0.282095, 1: 0.488603, 2: 0.315392, 3: 0.373176, 4: 0.105786}
LEADING_FACTOR = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 0.125}  # of P_l, folded into the constant


@lru_cache(maxsize=8)
def six_digit_table(
    sigma: float,
    harmonics: int = smoothing.NUM_HARMONICS,
    epsilon: float = smoothing.KERNEL_EPSILON,
    harmonic_exponent: int = smoothing.HARMONIC_SAMPLES_EXPONENT,
    kernel_exponent: int = smoothing.KERNEL_SAMPLES_EXPONENT,
) -> KernelTable:
    """`concon_kernel_table` with the library's six-digit constants for l <= 4 put back."""
    degree = np.arange(harmonics)
    big_l, big_m = 10**harmonic_exponent, 10**kernel_exponent
    x_h = np.arange(-big_l, big_l + 1, dtype=np.float64) / big_l
    harmonic = (
        smoothing._legendre_polynomials(x_h, harmonics)
        * np.sqrt((2 * degree + 1) / (4 * np.pi))[:, None]
    )
    for order, constant in SIX_DIGIT_CONSTANTS.items():
        exact = np.sqrt((2 * order + 1) / (4 * np.pi)) * LEADING_FACTOR[order]
        harmonic[order] *= constant / exact
    x_k = np.arange(-big_m, big_m + 1, dtype=np.float64) / big_m
    index = np.clip((x_k * big_l + big_l).astype(np.int64), 0, 2 * big_l)
    weight = np.exp(-sigma * degree * (degree + 1)) * (2.0 * degree + 1.0)
    summed = np.zeros(x_h.size, dtype=np.float64)
    for order in range(harmonics):
        summed += weight[order] * harmonic[order]
    values = np.maximum(summed[index], 1e-20)
    scan = 1.0 - 1e-4 * np.arange(0, int(2.0 / 1e-4) + 1)
    scan_index = np.clip((scan * big_m + big_m).astype(np.int64), 0, 2 * big_m)
    below = np.nonzero(values[scan_index] * values[2 * big_m] < epsilon)[0]
    cutoff_cosine = float(scan[below[0]]) if below.size else -1.0
    return KernelTable(values=values, samples=big_m, cutoff_cosine=cutoff_cosine)


@contextlib.contextmanager
def the_binarys_constants():
    """Have the package's density use the six-digit table, as the binary does; test-side only."""
    exact = smoothing.concon_kernel_table
    smoothing.concon_kernel_table = six_digit_table  # spherical_heat_kernel looks it up by name
    try:
        yield
    finally:
        smoothing.concon_kernel_table = exact


def write_input(path, points_in, points_out, hemisphere_in, hemisphere_out):
    """Write endpoints as `intersections_to_sphere.py` does, ``1 - surface`` and all."""
    n = points_in.shape[0]
    rows = np.column_stack(
        [np.zeros(n), 1 - hemisphere_in, points_in, np.zeros(n), 1 - hemisphere_out, points_out]
    )
    with open(path, "w") as handle:
        handle.write(f"#{n}\n")
        np.savetxt(handle, rows, fmt="%i\t %i\t %.9f\t %.9f\t %.9f\t %i\t %i\t %.9f\t %.9f\t %.9f")


def pipeline_command(work):
    """`Compute_Kernel` with the pipeline's flags, on the paths `concon_probe.py` uses."""
    flags = [
        ("--subj", "subject"),
        ("--sigma", str(SIGMA)),
        ("--epsilon", "0.001"),
        ("--final_thold", "0.000000001"),
        ("--OPT_VAL_exp_num_kern_samps", "6"),
        ("--OPT_VAL_exp_num_harm_samps", "5"),
        ("--OPT_VAL_num_harm", "33"),
        ("--LOAD_xing_path", f"{work}/"),
        ("--LOAD_xing_postfix", "_xing_sphere_avg_coords.tsv"),
        ("--LOAD_kernel_path", ""),
        ("--LOAD_kernel_postfix", ""),
        ("--LOAD_mask_file", "MASK"),
        ("--SAVE_Compute_Kernel_prefix", f"{work}/"),
        ("--SAVE_Compute_Kernel_postfix", "_out.raw"),
        ("--LOAD_grid_file", f"{AVE}/lh_grid_avg_ico4.m"),
        ("--LOAD_rh_grid_file", f"{AVE}/rh_grid_avg_ico4.m"),
    ]
    return [C3, "Compute_Kernel"] + [part for flag in flags for part in flag]


def read_output(work):
    """The matrix `c3_main` wrote into `work`, in the package's hemisphere order."""
    raw = f"{work}/subject_out.raw"
    with open(raw, "rb") as handle:
        size = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        records = np.fromfile(handle, dtype=RECORD)
    matrix = np.zeros((size, size))
    matrix[records["x"], records["y"]] = records["lambda"]
    above, below = np.count_nonzero(np.triu(matrix, 1)), np.count_nonzero(np.tril(matrix, -1))
    print(
        f"c3_main wrote {records.size:,} entries: {above:,} above the diagonal, {below:,} below, "
        f"{np.count_nonzero(np.diag(matrix)):,} on it",
        flush=True,
    )
    if not above or not below:
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # concon puts the right hemisphere first; convert_raw.py swaps the two back. (The binary
    # reads --LOAD_grid_file for both hemispheres and ignores --LOAD_rh_grid_file; the lab's
    # two .m files are identical, as the bundled sphere's halves are.)
    half = size // 2
    order = np.r_[half:size, 0:half]
    return matrix[np.ix_(order, order)]


def run_c3_main(work):
    """Run the binary on `work/subject_xing_sphere_avg_coords.tsv`; the matrix it writes."""
    raw = f"{work}/subject_out.raw"
    if os.path.exists(raw):
        os.remove(raw)
    done = subprocess.run(pipeline_command(work), capture_output=True, text=True, timeout=12 * 3600)
    if done.returncode != 0 or not os.path.exists(raw):
        raise RuntimeError(
            f"c3_main exited {done.returncode}"
            + (" and wrote nothing" if not os.path.exists(raw) else "")
            + f": {done.stdout[-800:]}{done.stderr[-800:]}"
        )
    return read_output(work)


def compare(name, port, binary, angle):
    """The table PORTING.md item 6 reports, for one path of the package."""
    both = (port > 0) & (binary > 0)
    print(f"\n=== {name} against c3_main, the same positions ===")
    print(f"  correlation, all {port.size:,} pairs   {np.corrcoef(port, binary)[0, 1]:.8f}")
    print(f"  correlation, pairs both write  {np.corrcoef(port[both], binary[both])[0, 1]:.8f}")
    print(f"  best scale                     {np.dot(port, binary) / np.dot(port, port):.6f}")
    relative = np.linalg.norm(port - binary) / np.linalg.norm(binary)
    print(f"  relative error, not rescaled   {relative:.2e}")
    equal = int(np.isclose(port[both], binary[both], rtol=1e-6, atol=0).sum())
    print(f"  pairs equal to 1e-6 relative   {equal:,} of {int(both.sum()):,}")
    tight = int(np.isclose(port[both], binary[both], rtol=1e-12, atol=0).sum())
    same = int((port[both] == binary[both]).sum())
    print(f"  pairs within 1e-12, identical  {tight:,}, {same:,}")
    largest = float(np.max(np.abs(port[both] - binary[both]) / binary[both])) if both.any() else 0.0
    print(f"  largest relative difference    {largest:.2e}")
    ours, theirs = int((port > 0).sum()), int((binary > 0).sum())
    print(f"  non-zero pairs  {ours:,} against {theirs:,} ({ours / theirs - 1:+.3%})")
    for low, high in BANDS:
        band = both & (angle >= low) & (angle < high)
        if band.any():
            print(
                f"  median ratio c3_main / port, {low:>3}-{high:<3} deg  "
                f"{np.median(binary[band] / port[band]):.6f}  ({int(band.sum()):,} pairs)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sc", help="a released SC file, with endpoints")
    parser.add_argument("work", help="a folder for c3_main's input and output")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="compare with the binary's input and output already in the work folder",
    )
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)

    sc = sbci.load(args.sc)
    points_in, points_out = endpoint_positions(sc.endpoints)
    hemisphere_in = np.asarray(sc.endpoints.surf_in, dtype=np.int64)
    hemisphere_out = np.asarray(sc.endpoints.surf_out, dtype=np.int64)
    n = points_in.shape[0]
    print(f"{os.path.basename(args.sc)}: {n:,} streamlines", flush=True)

    tsv = f"{args.work}/subject_xing_sphere_avg_coords.tsv"
    if not args.reuse:
        write_input(tsv, points_in, points_out, hemisphere_in, hemisphere_out)
    # Both sides smooth the positions as written, to nine decimals.
    q_in, q_out, h_in, h_out = read_concon_endpoints(tsv)
    if not (np.array_equal(h_in, hemisphere_in) and np.array_equal(h_out, hemisphere_out)):
        raise RuntimeError("the hemispheres did not survive the round trip through the .tsv")
    moved = max(np.abs(q_in - points_in).max(), np.abs(q_out - points_out).max())
    print(f"positions as written are within {moved:.1e} of the stored ones", flush=True)

    if args.reuse:
        print(f"reusing c3_main's output in {args.work}", flush=True)
        binary = read_output(args.work)
    else:
        start = time.time()
        binary = run_c3_main(args.work)
        print(f"c3_main took {time.time() - start:.0f} s", flush=True)

    vertices = _grid_vertices(sc)
    labels = hemisphere_labels(sc.n_vertices)
    upper = np.triu_indices(sc.n_vertices, 1)
    cosine = np.clip((vertices[upper[0]] * vertices[upper[1]]).sum(axis=1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    target = binary[upper]
    poles = np.flatnonzero(vertices[:, 2] < -0.9999)  # (0, 0, -1), once per hemisphere
    print(
        f"\nthe south pole, vertices {poles.tolist()}: c3_main's rows there sum to "
        f"{binary[poles].sum(axis=1).tolist()}",
        flush=True,
    )
    for quantized, name in ((True, "tables (the default)"), (False, "closed form")):
        start = time.time()
        density = endpoint_density(
            vertices, q_in, q_out, h_in, h_out, labels, sigma=SIGMA, quantized=quantized
        )
        apply_final_threshold(density, n)
        print(f"\nthe package's density, {name}: {time.time() - start:.0f} s", flush=True)
        port = density / n
        emptied = port.copy()
        emptied[poles, :] = 0.0
        emptied[:, poles] = 0.0
        share = 1 - emptied[upper].sum() / port[upper].sum()
        print(f"  its mass at the south pole: {share:.4%} of the pairs' total")
        compare(f"{name}, as computed", port[upper], target, angle)
        compare(
            f"{name}, the south pole emptied as c3_main leaves it", emptied[upper], target, angle
        )

    # The binary's own inaccuracies, put back: fscanf("%f") reads the written positions into
    # float, as they are and not renormalised, and the harmonics library's constants have six
    # digits. The grid is the .m file the binary was handed, the bundled sphere to 6e-12.
    raw = np.loadtxt(tsv, skiprows=1)
    f_in = raw[:, 2:5].astype(np.float32).astype(np.float64)
    f_out = raw[:, 7:10].astype(np.float32).astype(np.float64)
    m_grid = read_m_grid(f"{AVE}/lh_grid_avg_ico4.m")
    m_grid = np.vstack([m_grid, m_grid])  # one file for both hemispheres, as the binary reads it
    for name, context in (
        ("tables, the positions in float32 as the binary reads them", contextlib.nullcontext()),
        ("tables, float32 positions and the binary's six-digit constants", the_binarys_constants()),
    ):
        start = time.time()
        with context:
            density = endpoint_density(m_grid, f_in, f_out, h_in, h_out, labels, sigma=SIGMA)
        apply_final_threshold(density, n)
        print(f"\nthe package's density, {name}: {time.time() - start:.0f} s", flush=True)
        emptied = density / n
        emptied[poles, :] = 0.0
        emptied[:, poles] = 0.0
        compare(f"{name}, the south pole emptied", emptied[upper], target, angle)

    mask = np.asarray(sc.mask, dtype=bool)
    cortex = mask[upper[0]] & mask[upper[1]]
    released = sc.dense()[upper][cortex]
    print("\n=== c3_main against the released file, which the package smoothed ===")
    r = np.corrcoef(target[cortex], released)[0, 1]
    print(f"  correlation over the cortical pairs  {r:.8f}")
    print(f"  c3_main's mass off the cortex        {1 - target[cortex].sum() / target.sum():.4%}")


if __name__ == "__main__":
    main()
