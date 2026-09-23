"""Compare the ConSEAL port, in strict mode, with the MATLAB reference dump.

Reads the ``.mat`` written by ``conseal_reference.m``, rebuilds the same grid
from the mesh it saved, feeds the port the same endpoints, and reports the
agreement of every intermediate quantity. Run by hand on Longleaf::

    python tests/reference/conseal_compare.py /work/users/x/y/xya/conseal-ref/conseal_reference.mat

Point location, the barycentric weights, the kernel and the densities are
expected to agree to float32 rounding (the reference casts kernels, frames
and the adjacency to ``single``); everything downstream of the gradient to a
few parts in a million; the registration trace to the same order after each
step (both sides start from the same state each iteration only if the
previous steps agreed, so disagreement compounds).
"""

from __future__ import annotations

import sys

import h5py
import numpy as np

from sbci.alignment import SphericalGrid
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, StationaryWarp


def load(path):
    out = {}
    with h5py.File(path, "r") as f:

        def read(name):
            value = f[name][()]
            return np.asarray(value).T if value.ndim > 1 else np.asarray(value)

        for key in f:
            if key == "subjects":
                subjects = []
                for ref in f[key][()].ravel():
                    group = f[ref]
                    subjects.append({k: np.asarray(group[k][()]).T for k in group})
                out[key] = subjects
            elif key == "ico_mesh":
                out[key] = {k: np.asarray(f[key][k][()]).T for k in f[key]}
            else:
                try:
                    out[key] = read(key)
                except Exception:  # noqa: BLE001 - opaque MATLAB objects
                    pass
    return out


def report(name, ours, theirs, scale=None):
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    if ours.shape != theirs.shape:
        print(f"{name:28s} SHAPE {ours.shape} vs {theirs.shape}")
        return
    diff = np.abs(ours - theirs)
    scale = float(np.abs(theirs).max()) if scale is None else scale
    corr = np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] if ours.size > 1 else 1.0
    relative = diff.max() / max(scale, 1e-300)
    print(f"{name:28s} max|diff| {diff.max():.3e}  relative {relative:.3e}  r {corr:.9f}")


def main(path):
    ref = load(path)
    vertices = ref["ico_mesh"]["V"]
    faces = ref["ico_mesh"]["T"].astype(np.int64) - 1
    order = int(np.asarray(ref["L_WARP"]).ravel()[0])
    degree = int(np.asarray(ref["H_KERNEL"]).ravel()[0])
    sigma = float(np.asarray(ref["SIGMA"]).ravel()[0])
    grid = SphericalGrid(vertices, faces, order=order)

    print("--- geometry")
    report("voronoi areas", grid.areas, ref["grid_A"].ravel())
    report("e1", grid.e1, ref["grid_e1"])
    report("basis", grid.basis, ref["grid_basis"])
    report("divergence", grid.laplacian, ref["grid_div"])

    subjects = ref["subjects"]

    def build(s):
        return EndpointConnectome.from_points(
            grid, grid, s["sp"], s["ep"], s["h1"].ravel(), s["h2"].ravel()
        )

    f1, f2, f3 = (build(s) for s in subjects[:3])

    print("--- kernel (strict)")
    builder = HeatKernelBuilder(grid, grid, degree)
    kernel, derivative = builder.compute(sigma, strict_upstream=True)
    report("K", kernel.toarray(), ref["K"])
    report("dK.x", derivative.x.toarray(), ref["dKx"])
    report("dK.y", derivative.y.toarray(), ref["dKy"])
    report("dK.z", derivative.z.toarray(), ref["dKz"])

    print("--- densities")
    report("adjacency", f2.adjacency(), ref["A2"])
    fm, fe1, fe2 = f2.evaluate(kernel, derivative, strict_upstream=True)
    report("F", fm, ref["Fm"])
    report("F_e1", fe1, ref["Fe1"])
    report("F_e2", fe2, ref["Fe2"])
    q2, q2e1, q2e2 = f2.q_transform(kernel, derivative, strict_upstream=True)
    q1 = f1.q_transform(kernel)
    report("Q1", q1, ref["Q1"])
    report("Q2", q2, ref["Q2"])
    report("Q2_e1", q2e1, ref["Q2e1"])
    report("Q2_e2", q2e2, ref["Q2e2"])

    print("--- one gradient step (strict)")
    engine = ConSEAL(grid, grid, delta=0.05, strict_upstream=True)
    n = grid.n_vertices
    difference = q1 - q2
    lh_dh = engine._gradient(difference, q2, q2e1, q2e2, np.arange(n), grid)
    rh_dh = engine._gradient(difference, q2, q2e1, q2e2, np.arange(n, 2 * n), grid)
    report("lh gradient", lh_dh, ref["lh_dH"])
    report("rh gradient", rh_dh, ref["rh_dH"])
    lh_step, rh_step = engine._step(lh_dh), engine._step(rh_dh)
    report("lh step", lh_step, ref["lh_step"])
    report("rh step", rh_step, ref["rh_step"])
    lh_warp = StationaryWarp(grid, strict_upstream=True)
    rh_warp = StationaryWarp(grid, strict_upstream=True)
    lh_warp.compose(ref["lh_step"], strict_upstream=True)
    rh_warp.compose(ref["rh_step"], strict_upstream=True)
    report("lh warp vertices", lh_warp.vertices, ref["lh_V"])
    report("rh warp vertices", rh_warp.vertices, ref["rh_V"])
    report("lh velocity", lh_warp.velocity, ref["lh_v"])
    report("lh jacobian", lh_warp.jacobian, ref["lh_J"].ravel())
    moved = f2.copy()
    p_in, p_out = moved.warp(lh_warp, rh_warp)
    report("warped start points", p_in, ref["w_sp"])
    report("warped end points", p_out, ref["w_ep"])
    q2_after = moved.q_transform(kernel)
    report("Q2 after one step", q2_after, ref["Q2_after"])
    theirs_cost = float(np.asarray(ref["cost_after_one_step"]).ravel()[0])
    ours_cost = engine.cost(q1 - q2_after)
    print(f"{'cost after one step':28s} ours {ours_cost:.9f}  theirs {theirs_cost:.9f}")

    print("--- short registration (strict, 6 iterations)")
    engine = ConSEAL(
        grid, grid, delta=0.05, max_iterations=6, threshold=1e-12, strict_upstream=True
    )
    _, _, costs, _ = engine.register(f1, f2, kernel, derivative)
    theirs = np.asarray(ref["cost"]).ravel()
    for i, (a, b) in enumerate(zip(costs, theirs, strict=False)):
        print(f"  iteration {i}: ours {a:.9f}  theirs {b:.9f}  rel {abs(a - b) / b:.2e}")

    print("--- template (5 iterations)")
    template = ConSEAL(grid, grid).template([f1, f2, f3], kernel, iterations=5)
    report("template", template, ref["template"])


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/work/users/x/y/xya/conseal-ref/conseal_reference.mat"
    )
