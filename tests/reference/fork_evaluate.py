"""Score the fork and the package on every comparison case (PORTING.md item 19).

    module load python/3.12.4
    python tests/reference/fork_evaluate.py      # a batch job: 8 cores, 64 GB, about 40 minutes

For each case (cases.txt, and the pair): each method's own run (iterations, time, its own
cost), both methods' costs of where it left the moving endpoints -- the package's (its kernel
on its ico4 grid) and the fork's (its own kernel on its own icosphere, computed here as
Concon.evaluate does, and checked against the fork's first cost) -- how far the endpoints end
from where they truly began (the known warps) or how far they were moved at all (the split
halves, where nothing should move), and the warp's folded triangles and area ratios.
"""

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy import sparse

from sbci.alignment import MeshQuery
from sbci.conseal import (
    DEFAULT_KERNEL_DEGREE,
    DEFAULT_SIGMA,
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    default_grids,
)

ROOT = Path("/work/users/x/y/xya/conseal-fork-compare")
CASES = ROOT / "cases"
names = [line.strip() for line in open(ROOT / "cases.txt") if line.strip()] + ["pair"]

lh, rh = default_grids()
kernel, _ = HeatKernelBuilder(lh, rh, DEFAULT_KERNEL_DEGREE).compute(DEFAULT_SIGMA)
engine = ConSEAL(lh, rh)
fork_kernel = sparse.csr_matrix(
    sio.loadmat("/users/x/y/xya/encore_dice_scores_2/heat_kernel_0005_30.mat")["kernel"]
)


def degrees(p, q):
    return np.degrees(np.arccos(np.clip((p * q).sum(1), -1.0, 1.0)))


def load_fork(path):
    with h5py.File(path, "r") as h:
        out = {
            k: np.asarray(h[k]).T
            for k in ("cost", "moved_in", "moved_out", "grid_V", "grid_T", "lh_V", "rh_V")
        }
        out["elapsed"] = float(np.asarray(h["elapsed"]).ravel()[0])
    out["cost"] = out["cost"].ravel()
    out["faces"] = out.pop("grid_T").astype(np.int64) - 1
    return out


def results(name):
    if name == "pair":
        paths = {
            "fork": ROOT / "fork_pair.mat",
            "fork (CPU)": CASES / "pair_fork-cpu.mat",
            "package, defaults": ROOT / "package_defaults.npz",
            "package, fork's settings": ROOT / "package_fork.npz",
            "package, composed": CASES / "pair_package-compose.npz",
        }
    else:
        paths = {
            "fork": CASES / f"{name}_fork.mat",
            "fork (CPU)": CASES / f"{name}_fork-cpu.mat",
            "package, defaults": CASES / f"{name}_package-defaults.npz",
            "package, fork's settings": CASES / f"{name}_package-fork.npz",
            "package, composed": CASES / f"{name}_package-compose.npz",
        }
    found = {}
    for method, path in paths.items():
        if not path.exists():
            continue
        if method.startswith("fork"):
            run = load_fork(path)
            found[method] = dict(
                own=run["cost"],
                elapsed=run["elapsed"],
                moved=(run["moved_in"], run["moved_out"]),
                warps=(run["lh_V"], run["rh_V"]),
                base=(run["grid_V"], run["grid_V"]),
                faces=(run["faces"], run["faces"]),
            )
        else:
            run = dict(np.load(path))
            found[method] = dict(
                own=run["costs"],
                elapsed=float(run["elapsed"]),
                moved=(run["moved_in"], run["moved_out"]),
                warps=(run["lh_vertices"], run["rh_vertices"]),
                base=(lh.vertices, rh.vertices),
                faces=(lh.faces, rh.faces),
                rejected=int(run["rejected"]),
            )
    return found


def signed_areas(vertices, faces):
    p1, p2, p3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normal = np.cross(p2 - p1, p3 - p1)
    return np.linalg.norm(normal, axis=1) * np.sign((normal * (p1 + p2 + p3)).sum(1)) / 2


def fork_q(query, points_in, points_out, flags):
    size = query_vertices.shape[0]
    ends = []
    for points, side in zip((points_in, points_out), flags, strict=True):
        w, idx, _ = query.query_faces(points)
        ends.append((w, idx + size * side[:, None]))
    (w_a, i_a), (w_b, i_b) = ends
    data = (w_a[:, :, None] * w_b[:, None, :]).ravel()
    adjacency = sparse.coo_matrix(
        (data, (np.repeat(i_a, 3, axis=1).ravel(), np.tile(i_b, (1, 3)).ravel())),
        shape=(2 * size, 2 * size),
    ).tocsr()
    adjacency = (adjacency + adjacency.T) / (2 * points_in.shape[0])
    return np.sqrt(np.maximum((fork_kernel @ adjacency @ fork_kernel.T).toarray(), 0.0))


def package_cost(q_fixed, points, flags):
    """The package's cost of moving endpoints at ``points`` against the fixed density."""
    moving = EndpointConnectome.from_points(lh, rh, *points, *flags)
    return engine.cost(q_fixed - moving.q_transform(kernel))


def fork_cost(fork_fixed, query, points, flags):
    """The fork's plain-sum cost of the same, on its own grid and kernel."""
    return float(((fork_fixed - fork_q(query, *points, flags)) ** 2).sum())


query_vertices = None
fork_query = None
summary = []
for name in names:
    x = np.load(CASES / f"{name}.npz")
    flags = {
        p: [x[f"{p}_h{e}"].ravel().astype(np.int64) for e in ("in", "out")]
        for p in ("fixed", "moving")
    }
    start = (x["moving_in"], x["moving_out"])
    found = results(name)
    if not found:
        print(f"== {name}: no results yet")
        continue
    fork_runs = [m for m in found if m.startswith("fork")]
    if fork_query is None and fork_runs:
        query_vertices = found[fork_runs[0]]["base"][0]
        fork_query = MeshQuery(query_vertices, found[fork_runs[0]]["faces"][0])
    q_fixed = EndpointConnectome.from_points(
        lh, rh, x["fixed_in"], x["fixed_out"], *flags["fixed"]
    ).q_transform(kernel)
    fork_fixed = (
        None
        if fork_query is None
        else fork_q(fork_query, x["fixed_in"], x["fixed_out"], flags["fixed"])
    )
    truth = (x["truth_in"], x["truth_out"]) if "truth_in" in x.files else None
    print(f"== {name}: {start[0].shape[0]:,} moving streamlines")
    line = f"  start: package cost {package_cost(q_fixed, start, flags['moving']):.5f}"
    if fork_fixed is not None:
        line += f", fork cost {fork_cost(fork_fixed, fork_query, start, flags['moving']):.5f}"
        if fork_runs:
            line += f" (the fork's own first cost {found[fork_runs[0]]['own'][0]:.5f})"
    if truth is not None:
        before = np.r_[degrees(truth[0], start[0]), degrees(truth[1], start[1])]
        line += (
            f"; endpoints from where they began: mean {before.mean():.3f} deg, "
            f"95th {np.percentile(before, 95):.3f}"
        )
    print(line)
    for method, run in found.items():
        moved = run["moved"]
        travel = np.r_[degrees(start[0], moved[0]), degrees(start[1], moved[1])]
        own = run["own"]
        text = (
            f"  {method:24s} {own.size - 1:4d} it, {run['elapsed']:6.0f}s; "
            f"own cost {own[0]:.5f} -> {own[-1]:.5f}; "
            f"package cost {package_cost(q_fixed, moved, flags['moving']):.5f}"
        )
        if fork_fixed is not None:
            text += f", fork cost {fork_cost(fork_fixed, fork_query, moved, flags['moving']):.5f}"
        record = dict(case=name, method=method, travel=travel.mean())
        if truth is not None:
            error = np.r_[degrees(truth[0], moved[0]), degrees(truth[1], moved[1])]
            text += (
                f"; from the truth: mean {error.mean():.3f} deg, "
                f"median {np.median(error):.3f}, 95th {np.percentile(error, 95):.3f} "
                f"({1 - error.mean() / before.mean():.1%} undone)"
            )
            record["error"] = error.mean()
        else:
            text += (
                f"; moved: mean {travel.mean():.3f} deg, median {np.median(travel):.3f}, "
                f"95th {np.percentile(travel, 95):.3f}"
            )
        ratios = np.concatenate(
            [
                signed_areas(w, f) / signed_areas(b, f)
                for w, b, f in zip(run["warps"], run["base"], run["faces"], strict=True)
            ]
        )
        folds = int((ratios <= 0).sum())
        text += f"; folded {folds}, area ratio {ratios.min():.3f} to {ratios.max():.3f}"
        if "rejected" in run:
            text += f", {run['rejected']} steps refused"
        record["folds"] = folds
        summary.append(record)
        print(text, flush=True)

print("== summary")
for kind, key in (
    ("known warps: mean error from the truth (deg)", "error"),
    ("split halves: mean distance moved (deg)", "travel"),
):
    print(f"  {kind}")
    for method in (
        "fork",
        "fork (CPU)",
        "package, defaults",
        "package, fork's settings",
        "package, composed",
    ):
        rows = [
            r
            for r in summary
            if r["method"] == method
            and (key in r if key == "error" else r["case"].startswith("split"))
        ]
        if rows:
            values = ", ".join(f"{r['case']} {r[key]:.3f}" for r in rows)
            print(f"    {method:24s} {values}; folds {sum(r['folds'] for r in rows)}")
