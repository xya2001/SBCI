"""Score the synthetic warps of ``fork_synthetic_cases.py``: which method does best, unfolded.

    module load python/3.12.4
    python tests/reference/fork_synthetic_evaluate.py      # a few minutes

For each case and method -- the fork as it is (``_fork-cpu``), the fork with its folding steps
refused (``_fork-guard``), the package at the public code's settings (``_package-defaults``)
and at the paper's (``_package-fork``) -- how far the endpoints end from where the known warp
moved them from, how many triangles the estimated warp folds, its area ratios, iterations,
refused steps and time. Then, case by case, the best method among those that fold nothing,
and the guarded fork on the real pair against the package there.
"""

import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

ROOT = Path("/work/users/x/y/xya/conseal-fork-compare")
CASES = ROOT / "cases"
METHODS = {
    "fork": "_fork-cpu.mat",
    "fork, guarded": "_fork-guard.mat",
    "package, public settings": "_package-defaults.npz",
    "package, paper's settings": "_package-fork.npz",
}
names = [line.strip() for line in open(ROOT / "cases_synthetic.txt") if line.strip()]


def degrees(p, q):
    """Angle between unit vectors, row by row."""
    return np.degrees(np.arccos(np.clip((p * q).sum(1), -1.0, 1.0)))


def signed_areas(vertices, faces):
    """Twice-halved signed triangle areas, positive for the outward orientation."""
    p1, p2, p3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normal = np.cross(p2 - p1, p3 - p1)
    return np.linalg.norm(normal, axis=1) * np.sign((normal * (p1 + p2 + p3)).sum(1)) / 2


def refusals():
    """The guarded fork's refused steps, case by case, from its logs."""
    found = {}
    for log in sorted(CASES.glob("fork-synth-*.out")):
        text = log.read_text(errors="replace")
        case = re.search(r"case (\S+): fixed", text)
        count = re.search(r"Refused steps \(guarded copy\): (\d+)", text)
        if case and count:
            found[case.group(1)] = int(count.group(1))
    return found


def load(path):
    """One method's result: costs, time, moved endpoints, and the warp on its own grid."""
    if path.suffix == ".mat":
        with h5py.File(path, "r") as h:
            run = {k: np.asarray(h[k]).T for k in ("cost", "moved_in", "moved_out", "lh_V", "rh_V")}
            faces = np.asarray(h["grid_T"]).T.astype(np.int64) - 1
            base = np.asarray(h["grid_V"]).T
            elapsed = float(np.asarray(h["elapsed"]).ravel()[0])
        return dict(
            costs=run["cost"].ravel(),
            elapsed=elapsed,
            moved=(run["moved_in"], run["moved_out"]),
            warps=(run["lh_V"], run["rh_V"]),
            base=(base, base),
            faces=(faces, faces),
            rejected=None,
        )
    run = dict(np.load(path))
    lh, rh = default_grids()
    return dict(
        costs=run["costs"],
        elapsed=float(run["elapsed"]),
        moved=(run["moved_in"], run["moved_out"]),
        warps=(run["lh_vertices"], run["rh_vertices"]),
        base=(lh.vertices, rh.vertices),
        faces=(lh.faces, rh.faces),
        rejected=int(run["rejected"]),
    )


guarded_refusals = refusals()
rows = []
for name in names:
    x = np.load(CASES / f"{name}.npz")
    truth, start = (x["truth_in"], x["truth_out"]), (x["moving_in"], x["moving_out"])
    before = np.r_[degrees(truth[0], start[0]), degrees(truth[1], start[1])].mean()
    for method, suffix in METHODS.items():
        path = CASES / f"{name}{suffix}"
        if not path.exists():
            continue
        run = load(path)
        error = np.r_[degrees(truth[0], run["moved"][0]), degrees(truth[1], run["moved"][1])]
        ratios = np.concatenate(
            [
                signed_areas(w, f) / signed_areas(b, f)
                for w, b, f in zip(run["warps"], run["base"], run["faces"], strict=True)
            ]
        )
        refused = run["rejected"]
        if refused is None and method == "fork, guarded":
            refused = guarded_refusals.get(name)
        rows.append(
            dict(
                case=name,
                family=re.match(r"[a-z]+", name).group(0),
                size=int(re.search(r"(\d+)_s", name).group(1)),
                method=method,
                before=before,
                error=error.mean(),
                p95=np.percentile(error, 95),
                folds=int((ratios <= 0).sum()),
                low=ratios.min(),
                high=ratios.max(),
                iterations=run["costs"].size - 1,
                minutes=run["elapsed"] / 60,
                refused=refused,
            )
        )

print("== every run")
for r in rows:
    refused = "" if r["refused"] is None else f", {r['refused']} refused"
    print(
        f"  {r['case']:14s} {r['method']:26s} from {r['before']:.3f}: error {r['error']:.3f} deg "
        f"({1 - r['error'] / r['before']:6.1%} undone), 95th {r['p95']:.3f}; "
        f"folded {r['folds']:3d}, area {r['low']:.3f} to {r['high']:.3f}; "
        f"{r['iterations']:4d} it, {r['minutes']:5.1f} min{refused}"
    )

print("== by size: mean error left over the three seeds (deg), folds summed")
groups = defaultdict(list)
for r in rows:
    groups[(r["family"], r["size"], r["method"])].append(r)
for family in ("svf", "comp"):
    sizes = sorted({r["size"] for r in rows if r["family"] == family})
    for size in sizes:
        cells = []
        for method in METHODS:
            got = groups.get((family, size, method), [])
            if got:
                cells.append(
                    f"{method} {np.mean([g['error'] for g in got]):.3f}"
                    f" ({len(got)} runs, {sum(g['folds'] for g in got)} folds)"
                )
        before = np.mean([g["before"] for g in rows if g["family"] == family and g["size"] == size])
        print(f"  {family}{size:03d} (from {before:.2f} deg): " + "; ".join(cells))

print("== the best method, case by case, among the runs that fold nothing")
wins = defaultdict(int)
complete = 0
for name in names:
    runs = [r for r in rows if r["case"] == name]
    if len(runs) < len(METHODS):
        continue
    complete += 1
    clean = [r for r in runs if r["folds"] == 0]
    best = min(clean, key=lambda r: r["error"])
    wins[best["method"]] += 1
    folded = ", ".join(f"{r['method']} ({r['folds']})" for r in runs if r["folds"])
    print(
        f"  {name:14s} best {best['method']:26s} {best['error']:.3f} deg"
        + (f"; folded: {folded}" if folded else "")
    )
print(f"  {complete} cases complete; wins: " + ", ".join(f"{m} {n}" for m, n in wins.items()))
for method in METHODS:
    got = [r for r in rows if r["method"] == method]
    if got:
        clean = [r for r in got if r["folds"] == 0]
        print(
            f"  {method:26s} {len(got)} runs, {len(got) - len(clean)} folded; mean share undone "
            f"{np.mean([1 - r['error'] / r['before'] for r in got]):.1%}"
        )

pair = CASES / "pair_fork-guard.mat"
if pair.exists():
    print("== the real pair, the guarded fork beside the others")
    x = np.load(CASES / "pair.npz")
    lh, rh = default_grids()
    kernel, _ = HeatKernelBuilder(lh, rh, 30).compute(0.005)
    engine = ConSEAL(lh, rh)
    flags = {
        p: [x[f"{p}_h{e}"].ravel().astype(np.int64) for e in ("in", "out")]
        for p in ("fixed", "moving")
    }
    q_fixed = EndpointConnectome.from_points(
        lh, rh, x["fixed_in"], x["fixed_out"], *flags["fixed"]
    ).q_transform(kernel)
    start = (x["moving_in"], x["moving_out"])
    for label, path in (
        ("fork", CASES / "pair_fork-cpu.mat"),
        ("fork, guarded", pair),
        ("package, paper's settings", ROOT / "package_fork.npz"),
        ("package, public settings", ROOT / "package_defaults.npz"),
    ):
        if not path.exists():
            continue
        run = load(path)
        moving = EndpointConnectome.from_points(lh, rh, *run["moved"], *flags["moving"])
        cost = engine.cost(q_fixed - moving.q_transform(kernel))
        ratios = np.concatenate(
            [
                signed_areas(w, f) / signed_areas(b, f)
                for w, b, f in zip(run["warps"], run["base"], run["faces"], strict=True)
            ]
        )
        travel = np.r_[degrees(start[0], run["moved"][0]), degrees(start[1], run["moved"][1])]
        refused = run["rejected"]
        if refused is None and label == "fork, guarded":
            refused = guarded_refusals.get("pair")
        print(
            f"  {label:26s} package cost {cost:.5f}, own {run['costs'][0]:.5f} -> "
            f"{run['costs'][-1]:.5f}; moved {travel.mean():.2f} deg; "
            f"folded {int((ratios <= 0).sum())}, area {ratios.min():.3f} to {ratios.max():.3f}; "
            f"{run['costs'].size - 1} it, "
            f"{run['elapsed'] / 60:.1f} min" + ("" if refused is None else f", {refused} refused")
        )
