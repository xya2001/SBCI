"""Compare the ConSEAL port with the MATLAB reference dump, stage by stage.

Reads the ``.mat`` written by ``conseal_reference.m``, rebuilds the same grid
from the mesh it saved, feeds the port the same endpoints, and reports the
agreement of every intermediate quantity. Run by hand on Longleaf::

    python tests/reference/conseal_compare.py /work/users/x/y/xya/conseal-ref/conseal_reference.mat

The reference's errors are put back into the port's output for the comparison:
its geometry by ``reference_geometry.py``, its kernel derivative, densities,
square-root chain rule, smoothing and median by ``reference_conseal.py``. That
is the code of ``strict_upstream=True``, which PORTING.md item 22 took out of
the package. Point location, the barycentric weights, the kernel and the
densities are expected to agree to float32 rounding (the reference casts
kernels, frames and the adjacency to ``single``), and everything downstream of
the gradient to a few parts in a million. The six-iteration registration is
stepped here with the same arithmetic, each step smoothed into the field as the
reference does; nothing folds and the cost falls at every step, so the
reference's handling of a refused step and of a rise (items 3 and 4 of the
ConSEAL docstring) never acts, and the loop says so if it would.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import reference_conseal
from reference_geometry import with_the_reference_geometry

from sbci.alignment import SphericalGrid
from sbci.conseal import (
    PUBLIC_STEP_CLAMP,
    PUBLIC_VISCOSITY,
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    StationaryWarp,
    triangles_fold,
)


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
    # the reference's basis and divergence, as its run had them (reference_geometry.py)
    grid = with_the_reference_geometry(SphericalGrid(vertices, faces, order=order))

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

    print("--- kernel, the reference's derivative")
    builder = HeatKernelBuilder(grid, grid, degree)
    kernel = builder.compute(sigma, derivative=False)
    derivative = reference_conseal.kernel_derivative(builder, sigma)
    report("K", kernel.toarray(), ref["K"])
    report("dK.x", derivative.x.toarray(), ref["dKx"])
    report("dK.y", derivative.y.toarray(), ref["dKy"])
    report("dK.z", derivative.z.toarray(), ref["dKz"])

    print("--- densities, the reference's derivatives")
    report("adjacency", f2.adjacency(), ref["A2"])
    fm, fe1, fe2 = reference_conseal.evaluate(f2, kernel, derivative)
    report("F", fm, ref["Fm"])
    report("F_e1", fe1, ref["Fe1"])
    report("F_e2", fe2, ref["Fe2"])
    q2, q2e1, q2e2 = reference_conseal.q_transform(f2, kernel, derivative)
    q1 = f1.q_transform(kernel)
    report("Q1", q1, ref["Q1"])
    report("Q2", q2, ref["Q2"])
    report("Q2_e1", q2e1, ref["Q2e1"])
    report("Q2_e2", q2e2, ref["Q2e2"])

    print("--- one gradient step, smoothed into the field as the reference does")
    engine = ConSEAL(grid, grid, delta=0.05, step_clamp=PUBLIC_STEP_CLAMP)
    n = grid.n_vertices
    difference = q1 - q2
    lh_dh = engine._gradient(difference, q2, q2e1, q2e2, np.arange(n), grid)
    rh_dh = engine._gradient(difference, q2, q2e1, q2e2, np.arange(n, 2 * n), grid)
    report("lh gradient", lh_dh, ref["lh_dH"])
    report("rh gradient", rh_dh, ref["rh_dH"])
    lh_step, rh_step = engine._step(lh_dh), engine._step(rh_dh)
    report("lh step", lh_step, ref["lh_step"])
    report("rh step", rh_step, ref["rh_step"])
    laplacian = reference_conseal.cotangent_laplacian(grid.vertices, grid.faces)
    lh_warp, rh_warp = StationaryWarp(grid), StationaryWarp(grid)
    for warp, step in ((lh_warp, ref["lh_step"]), (rh_warp, ref["rh_step"])):
        step = np.asarray(step, dtype=np.float64)
        warp.velocity = reference_conseal.smooth(laplacian, step, PUBLIC_VISCOSITY)
        warp.vertices = warp.exponential()
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

    print("--- a short registration, the reference's descent (6 iterations)")
    lh_warp, rh_warp = StationaryWarp(grid), StationaryWarp(grid)
    moved = f2.copy()
    q2, q2e1, q2e2 = reference_conseal.q_transform(moved, kernel, derivative)
    costs = [engine.cost(q1 - q2)]
    for _ in range(6):
        difference = q1 - q2
        for warp, rows in ((lh_warp, np.arange(n)), (rh_warp, np.arange(n, 2 * n))):
            step = engine._step(engine._gradient(difference, q2, q2e1, q2e2, rows, grid))
            warp.velocity = reference_conseal.smooth(
                laplacian, warp.velocity + step, PUBLIC_VISCOSITY
            )
            warp.vertices = warp.exponential()
            if triangles_fold(warp.vertices, grid.faces):
                print("  a step folds the mesh: the reference keeps it in the field (item 3)")
        moved.warp(lh_warp, rh_warp)
        q2, q2e1, q2e2 = reference_conseal.q_transform(moved, kernel, derivative)
        costs.append(engine.cost(q1 - q2))
        if costs[-1] > costs[-2]:
            print("  the cost rose: the reference would stop and keep the step (item 4)")
    theirs = np.asarray(ref["cost"]).ravel()
    for i, (a, b) in enumerate(zip(costs, theirs, strict=False)):
        print(f"  iteration {i}: ours {a:.9f}  theirs {b:.9f}  rel {abs(a - b) / b:.2e}")

    print("--- template, the reference's median (5 iterations)")
    densities = [f.q_transform(kernel) for f in (f1, f2, f3)]
    report("template", reference_conseal.template(densities, iterations=5), ref["template"])


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/work/users/x/y/xya/conseal-ref/conseal_reference.mat"
    )
