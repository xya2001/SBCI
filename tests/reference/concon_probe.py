"""Measure concon's spherical kernel by running concon on one streamline.

Every other statement about the spherical heat kernel in PORTING.md was
inferred from a released matrix built out of a million streamlines, where the
kernel is convolved with itself and with the endpoint distribution. That is a
weak instrument. The `c3_main` binary shipped with SBCI_Pipeline is an ordinary
x86-64 Linux executable and runs, so it can be asked directly.

Feed it a single streamline from p to q and its output is

    D(i, j) = K(theta_i) * K(theta_j),   theta_i = angle(vertex_i, p)

so one column of D is the kernel as a function of angle, up to a constant, and
the full set of stored entries over-determines it: taking logs,

    log D(i, j) = f(theta_i) + f(theta_j),  f = log K

is a linear system for f which many runs solve on a fine angular grid, with no
per-run normalization to guess.

This is what established that concon's kernel has **compact support** -- it
vanishes at about 2.9*sqrt(sigma) radians, 11.9 degrees at sigma = 0.005 --
while the port's kernel spreads over the whole sphere. Run by hand; it needs
the binary and the lab grid files, neither of which is in this repository.

    python tests/reference/concon_probe.py --what kernel
    python tests/reference/concon_probe.py --what support
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

C3 = os.environ.get(
    "CONCON_C3_MAIN",
    "/work/users/x/y/xya/sbci-reference/SBCI_Pipeline/concon/c3_main",
)
AVE = os.environ.get("SBCI_AVE", "/overflow/zzhanglab/ADNI/ADNI-bids/SBCI_AVE")
WORK = os.environ.get("CONCON_PROBE_DIR", "/work/users/x/y/xya/shkprobe")

RECORD = np.dtype([("x", np.int32), ("y", np.int32), ("lambda", np.double)])


def read_m_grid(path: str) -> np.ndarray:
    """Unit-normalized vertices of a MeshLib ``.m`` grid, in file order."""
    rows = [
        [float(value) for value in line.split()[2:5]]
        for line in open(path)
        if line.startswith("Vertex")
    ]
    vertices = np.asarray(rows, dtype=np.float64)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def run_one(p, q, sigma, epsilon=0.001, tag="probe"):
    """Run Compute_Kernel on a single streamline and return its stored entries."""
    os.makedirs(WORK, exist_ok=True)
    tsv = f"{WORK}/{tag}_xing_sphere_avg_coords.tsv"
    with open(tsv, "w") as handle:
        handle.write(
            "#1\n0\t 0\t "
            + "\t ".join(f"{v:.9f}" for v in p)
            + "\t 0\t 0\t "
            + "\t ".join(f"{v:.9f}" for v in q)
            + "\n"
        )
    raw = f"{WORK}/{tag}_out.raw"
    if os.path.exists(raw):
        os.remove(raw)
    done = subprocess.run(
        [
            C3,
            "Compute_Kernel",
            "--subj",
            tag,
            "--sigma",
            str(sigma),
            "--epsilon",
            str(epsilon),
            "--final_thold",
            "0.000000001",
            "--OPT_VAL_exp_num_kern_samps",
            "6",
            "--OPT_VAL_exp_num_harm_samps",
            "5",
            "--OPT_VAL_num_harm",
            "33",
            "--LOAD_xing_path",
            f"{WORK}/",
            "--LOAD_xing_postfix",
            "_xing_sphere_avg_coords.tsv",
            "--LOAD_kernel_path",
            "",
            "--LOAD_kernel_postfix",
            "",
            "--LOAD_mask_file",
            "MASK",
            "--SAVE_Compute_Kernel_prefix",
            f"{WORK}/",
            "--SAVE_Compute_Kernel_postfix",
            "_out.raw",
            "--LOAD_grid_file",
            f"{AVE}/lh_grid_avg_ico4.m",
            "--LOAD_rh_grid_file",
            f"{AVE}/rh_grid_avg_ico4.m",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if not os.path.exists(raw):
        raise RuntimeError(f"c3_main wrote nothing: {done.stdout[-500:]}{done.stderr[-500:]}")
    with open(raw, "rb") as handle:
        np.fromfile(handle, dtype=np.int32, count=1)
        return np.fromfile(handle, dtype=RECORD)


def kernel_profile(sigma, runs, seed=1):
    """Solve log D = f(theta_i) + f(theta_j) for the kernel on a fine grid."""
    grid = read_m_grid(f"{AVE}/lh_grid_avg_ico4.m")
    half = grid.shape[0]
    rng = np.random.default_rng(seed)
    theta_i, theta_j, observed = [], [], []
    for index in range(runs):
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        q = rng.normal(size=3)
        q /= np.linalg.norm(q)
        record = run_one(p, q, sigma, tag=f"k{index}")
        theta_i.append(np.degrees(np.arccos(np.clip(grid[record["x"] % half] @ p, -1, 1))))
        theta_j.append(np.degrees(np.arccos(np.clip(grid[record["y"] % half] @ q, -1, 1))))
        observed.append(np.log(record["lambda"]))
    theta_i = np.concatenate(theta_i)
    theta_j = np.concatenate(theta_j)
    observed = np.concatenate(observed)

    limit = np.degrees(3.4 * np.sqrt(sigma))
    edges = np.linspace(0.0, limit, 34)
    centre = 0.5 * (edges[:-1] + edges[1:])
    bi = np.clip(np.digitize(theta_i, edges) - 1, 0, centre.size - 1)
    bj = np.clip(np.digitize(theta_j, edges) - 1, 0, centre.size - 1)
    design = np.zeros((observed.size, centre.size))
    np.add.at(design, (np.arange(observed.size), bi), 1.0)
    np.add.at(design, (np.arange(observed.size), bj), 1.0)
    present = design.sum(axis=0) > 0
    fitted, *_ = np.linalg.lstsq(design[:, present], observed, rcond=None)
    logk = np.full(centre.size, np.nan)
    logk[present] = fitted
    return centre, np.exp(logk - np.nanmax(logk))


def support_radius(sigma, epsilon=0.001):
    """Outermost vertex concon keeps around one endpoint, in degrees.

    Taken from a single column: concon stores both (i, j) and (j, i), so the
    row indices alone span both endpoints' supports and would overstate it.
    """
    grid = read_m_grid(f"{AVE}/lh_grid_avg_ico4.m")
    half = grid.shape[0]
    p = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 0.0, 0.0])
    record = run_one(p, q, sigma, epsilon=epsilon, tag="support")
    peak = np.argmax(record["lambda"])
    keep = record["y"] == record["y"][peak]
    centre = grid[record["x"][keep][np.argmax(record["lambda"][keep])] % half]
    angle = np.degrees(np.arccos(np.clip(grid[record["x"][keep] % half] @ centre, -1, 1)))
    return angle.max(), int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=("kernel", "support"), default="kernel")
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--runs", type=int, default=40)
    args = parser.parse_args()

    if args.what == "kernel":
        angle, kernel = kernel_profile(args.sigma, args.runs)
        print(f"{'angle(deg)':>11} {'K / K(0)':>10}")
        for a, k in zip(angle, kernel, strict=True):
            if not np.isnan(k):
                print(f"{a:11.3f} {k:10.6f}")
        return

    print(f"{'sigma':>9} {'epsilon':>9} {'support(deg)':>13} {'vertices':>9} {'/sqrt(sigma)':>13}")
    for sigma in (0.0025, 0.005, 0.01):
        radius, count = support_radius(sigma)
        print(
            f"{sigma:9.4f} {0.001:9.4g} {radius:13.3f} {count:9d} "
            f"{np.radians(radius) / np.sqrt(sigma):13.3f}"
        )
    for epsilon in (0.0001, 0.01, 0.1):
        radius, count = support_radius(0.005, epsilon=epsilon)
        print(
            f"{0.005:9.4f} {epsilon:9.4g} {radius:13.3f} {count:9d} "
            f"{np.radians(radius) / np.sqrt(0.005):13.3f}"
        )


if __name__ == "__main__":
    main()
