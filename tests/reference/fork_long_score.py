"""The fork run 300 iterations past its stopping rule: the error left from the known warp.

    module load python/3.12.4
    python tests/reference/fork_long_score.py

For the three cases ``conseal_convergence_probe.py`` runs the package on, the fork as it
stops (``_fork-cpu``) and run 300 iterations with no stopping rule (``fork_case.m`` with
``max_iters=300, threshold=-1``, which writes ``_fork-cpu_it300_th-1``): how far its endpoints
end from where the known warp moved them from, its folded triangles and its cost (PORTING.md
item 19, *The default, 11 October 2026*).
"""

from pathlib import Path

import h5py
import numpy as np

CASES = Path("/work/users/x/y/xya/conseal-fork-compare/cases")


def degrees(p, q):
    return np.degrees(np.arccos(np.clip((p * q).sum(1), -1.0, 1.0)))


def signed_areas(vertices, faces):
    p1, p2, p3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normal = np.cross(p2 - p1, p3 - p1)
    return np.linalg.norm(normal, axis=1) * np.sign((normal * (p1 + p2 + p3)).sum(1)) / 2


for name in ("svf035_s101", "svf070_s101", "comp100_s201"):
    x = np.load(CASES / f"{name}.npz")
    truth = (x["truth_in"], x["truth_out"])
    for suffix in ("_fork-cpu.mat", "_fork-cpu_it300_th-1.mat"):
        with h5py.File(CASES / f"{name}{suffix}", "r") as h:
            moved = (np.asarray(h["moved_in"]).T, np.asarray(h["moved_out"]).T)
            cost = np.asarray(h["cost"]).ravel()
            faces = np.asarray(h["grid_T"]).T.astype(np.int64) - 1
            base = np.asarray(h["grid_V"]).T
            warps = (np.asarray(h["lh_V"]).T, np.asarray(h["rh_V"]).T)
        error = np.r_[degrees(truth[0], moved[0]), degrees(truth[1], moved[1])].mean()
        folds = sum(
            int((signed_areas(w, faces) / signed_areas(base, faces) <= 0).sum()) for w in warps
        )
        print(
            f"{name} {suffix:26s} {cost.size - 1:4d} it: error {error:.4f} deg, {folds} folded, "
            f"cost {cost[0]:.6f} -> {cost[-1]:.6f}"
        )
