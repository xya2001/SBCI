"""ENCORE with each step composed as its gradient assumes, against the reference's order.

    SBCI_HCP_DIR=hcp-ya python tests/reference/encore_order_probe.py known CASE   # CASE 0..7
    SBCI_HCP_DIR=hcp-ya python tests/reference/encore_order_probe.py cohort
    python tests/reference/encore_order_probe.py rate

The gradient is taken for the warp ``phi(exp_x(D(x)))``; the reference composes the step after
the warp instead, ``exp(D)(phi(x))``. ``known`` deforms sub-100307's endpoints by a known warp
-- a smooth field of degree 4 whose largest displacement is 0.07, 0.14 or 0.21 radians, seeds 7
and 8, or a rigid turn of 10 or 20 degrees -- and registers it back, as the recovery figure of
docs/RESULTS.md does, once with each order; ``cohort`` aligns the eleven released subjects ten
steps onto ENCORE's median, as the cohort figure does, once with each order. ``rate`` measures
the descent rate of ``test_a_step_is_composed_as_the_gradient_assumes`` -- the cost's fall along
a step, by central differences, over the fall the gradient predicts -- at the identity and after
the test's 30-degree turn, once with each order. Everything else is the package's default
arithmetic; only ``SphericalWarp.compose`` changes.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_figures import angles, known_warp, resmoothed  # noqa: E402

import sbci  # noqa: E402
from sbci.alignment import (  # noqa: E402
    Encore,
    MeshQuery,
    SphericalGrid,
    SphericalWarp,
    _hemisphere_grids,
    normalize_rows,
    rotate_off_poles,
    sphere_exp_map,
)
from sbci.conseal import EndpointConnectome, StationaryWarp, default_grids  # noqa: E402

D = os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data")
CASES = [("smooth", 0.07, 7), ("smooth", 0.07, 8), ("smooth", 0.14, 7), ("smooth", 0.14, 8)]
CASES += [("smooth", 0.21, 7), ("smooth", 0.21, 8), ("turn", 10.0, 7), ("turn", 20.0, 7)]
gradient_order = SphericalWarp.compose


def parallel_transport(tangent, origin, destination):
    """Move tangent vectors along the geodesic from ``origin`` to ``destination``.

    The reference ENCORE's closed form, which ``sbci.alignment`` carried until its
    composition changed (PORTING.md item 21); it is kept here for the reference's
    order alone. :func:`sbci.conseal._transport` is the same transport written
    differently, and the two agree to 1e-13.
    """
    tangent = np.asarray(tangent, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    eps = np.finfo(float).eps

    axis = np.cross(origin, destination)
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + eps)
    before = np.cross(axis, origin)
    before = before / (np.linalg.norm(before, axis=1, keepdims=True) + eps)
    after = np.cross(axis, destination)
    after = after / (np.linalg.norm(after, axis=1, keepdims=True) + eps)

    moved = (tangent * before).sum(axis=1, keepdims=True) * after + (tangent * axis).sum(
        axis=1, keepdims=True
    ) * axis

    unchanged = (np.linalg.norm(origin - destination, axis=1) < 1e-4) | (
        np.linalg.norm(origin + destination, axis=1) < 1e-4
    )
    moved[unchanged] = tangent[unchanged]
    return moved


def reference_order(self, displacement):
    """The reference's composition: each image moved by the step interpolated where it lies."""
    displacement = np.asarray(displacement, dtype=np.float64)
    tangent = self._e1 * displacement[:, :1] + self._e2 * displacement[:, 1:2]
    weights, indices = self._query.query(self.vertices)
    n = self.vertices.shape[0]
    transported = parallel_transport(
        tangent[indices].reshape(-1, 3),
        self.base[indices].reshape(-1, 3),
        np.repeat(self.vertices, 3, axis=0),
    ).reshape(n, 3, 3)
    tangent = (weights[:, :, None] * transported).sum(axis=1)
    clone = object.__new__(SphericalWarp)
    clone.__dict__.update(self.__dict__)
    clone.vertices = normalize_rows(sphere_exp_map(self.vertices, tangent))
    clone.jacobian = clone._compute_jacobian()
    return clone


ORDERS = (("the gradient's order", gradient_order), ("the reference's order", reference_order))


def known(case):
    kind, size, seed = CASES[case]
    subject = sbci.load(f"{D}/sub-100307_sc.h5")
    lh, rh = default_grids()
    rng = np.random.default_rng(seed)
    if kind == "smooth":
        truth = [known_warp(grid, rng, size) for grid in default_grids(4)]
    else:
        axis = rng.standard_normal(3)
        axis *= np.radians(size) / np.linalg.norm(axis)
        cosine, sine = np.cos(np.radians(size)), np.sin(np.radians(size))
        k = axis / np.linalg.norm(axis)
        cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        turn = np.eye(3) + sine * cross + (1 - cosine) * cross @ cross
        truth = []
        for grid in default_grids(4):
            warp = StationaryWarp(grid, viscosity=0.0)
            warp.rotate(turn)
            truth.append(warp)
    reference = resmoothed(subject, subject.endpoints)
    carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
    original = carrier.positions()
    carrier.warp(*truth)
    deformed = resmoothed(subject, carrier.to_endpoints())
    moved = EndpointConnectome.from_endpoints(deformed.endpoints, lh, rh).positions()
    before = np.r_[angles(original[0], moved[0]), angles(original[1], moved[1])].mean()
    print(f"case {case}: {kind} {size:g}, seed {seed}: endpoints {before:.3f} deg off", flush=True)
    grids, rotations = _hemisphere_grids(6, return_rotations=True)
    ends = (carrier.hemisphere_in, carrier.hemisphere_out)
    for label, compose in ORDERS:
        SphericalWarp.compose = compose
        engine = Encore(*grids, max_iterations=100)
        fixed = engine.root(reference.dense())
        start = float(((fixed - engine.root(deformed.dense())) ** 2 * engine.area_product).sum())
        steps = []
        t = time.time()
        _, lh_warp, rh_warp, cost = engine.register(
            fixed,
            deformed.dense(),
            target_is_root=True,
            callback=lambda i, _c, taken=steps: taken.append(i),
        )
        elapsed = time.time() - t
        warped = (lh_warp.vertices @ rotations[0], rh_warp.vertices @ rotations[1])
        placed_all = []
        for points, hemispheres in zip(moved, ends, strict=True):
            placed = np.empty_like(points)
            for side, grid in enumerate((lh, rh)):
                pick = np.asarray(hemispheres) == side
                weights, indices = MeshQuery(normalize_rows(warped[side]), grid.faces).query(
                    points[pick]
                )
                placed[pick] = normalize_rows(
                    np.einsum("nk,nkj->nj", weights, grid.vertices[indices])
                )
            placed_all.append(placed)
        after = np.r_[angles(original[0], placed_all[0]), angles(original[1], placed_all[1])]
        print(
            f"  {label:22s}: {before:.3f} -> {after.mean():.3f} deg "
            f"({1 - after.mean() / before:.1%} undone), cost {start:.5f} -> {cost:.5f} "
            f"({cost / start:.3f}), {len(steps)} steps, {elapsed:.0f}s",
            flush=True,
        )
    SphericalWarp.compose = gradient_order


def cohort():
    paths = sorted(Path(D).glob("sub-*_sc.h5"))
    subjects = [sbci.load(p) for p in paths]
    smoothed = [resmoothed(s, s.endpoints) for s in subjects]

    def similarity(matrices):
        data = np.stack([np.asarray(m, dtype=np.float32).ravel() for m in matrices])
        return np.corrcoef(data)[np.triu_indices(len(matrices), 1)]

    from sbci.grid import to_condensed

    before = similarity([s.data for s in smoothed])
    print(f"{len(subjects)} subjects: mean pairwise correlation {before.mean():.4f}", flush=True)
    for label, compose in ORDERS:
        SphericalWarp.compose = compose
        t = time.time()
        result = sbci.align(smoothed, max_iterations=10, backtracks=4)
        after = similarity([to_condensed(np.asarray(d)) for d in result.aligned])
        falls = [trace[-1] / trace[0] for trace in result.traces]
        steps = [len(trace) - 1 for trace in result.traces]
        print(
            f"  {label:22s}: mean pairwise correlation {after.mean():.4f}; cost to "
            f"{min(falls):.3f}-{max(falls):.3f} of its start ({np.mean(falls):.3f} on average), "
            f"{min(steps)}-{max(steps)} steps; {time.time() - t:.0f}s",
            flush=True,
        )
    SphericalWarp.compose = gradient_order


def rate():
    import ast

    from scipy.spatial.transform import Rotation

    tests = Path(__file__).resolve().parents[1] / "test_alignment.py"
    node = next(
        n
        for n in ast.parse(tests.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "icosphere"
    )
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), "icosphere", "exec"), namespace)
    vertices, faces = namespace["icosphere"](3)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=4)
    n = grid.n_vertices
    both = np.vstack([grid.vertices, grid.vertices])

    def bundles(ends):
        density = np.zeros((2 * n, 2 * n))
        for a, b in ends:
            ka = np.exp(6.0 * (both @ (np.asarray(a) / np.linalg.norm(a)) - 1))
            kb = np.exp(6.0 * (both @ (np.asarray(b) / np.linalg.norm(b)) - 1))
            density += np.outer(ka, kb) + np.outer(kb, ka)
        np.fill_diagonal(density, 0.0)
        return density

    fixed = bundles([([1, 0.2, 0.3], [0.1, 1, -0.5]), ([-0.3, 1, 0.2], [0.7, -0.6, 0.2])])
    moving = bundles([([1, 0.35, 0.2], [0.2, 1, -0.4]), ([-0.4, 1, 0.35], [0.75, -0.5, 0.3])])
    encore = Encore(grid, grid)
    target, source = encore.root(fixed), encore.root(moving)

    def cost(lh, rh):
        image = encore.concon.evaluate_root(source, lh, rh)
        return ((target - image) ** 2 * encore.area_product).sum(), image

    def measured(lh, rh):
        _, image = cost(lh, rh)
        d_e1, d_e2 = encore.concon.derivative(image)
        weighted = (target - image) * encore.area_product
        a = (weighted * 2 * d_e1).sum(axis=1)
        b = (weighted * 2 * d_e2).sum(axis=1)
        c = (weighted * image).sum(axis=1)
        steps, predicted = [], 0.0
        for lo in (0, n):
            g = 2 * (
                a[lo : lo + n] @ grid.basis[:, :, 0]
                + b[lo : lo + n] @ grid.basis[:, :, 1]
                + c[lo : lo + n] @ grid.laplacian
            )
            predicted -= g @ g
            steps.append((g[None, :, None] * grid.basis).sum(axis=1))
        eps = 2e-3 / max(np.linalg.norm(d, axis=1).max() for d in steps)
        ahead = cost(lh.compose(eps * steps[0]), rh.compose(eps * steps[1]))[0]
        behind = cost(lh.compose(-eps * steps[0]), rh.compose(-eps * steps[1]))[0]
        return (ahead - behind) / (2 * eps) / predicted

    turn = np.array([0.16, 0.26, 0.42])
    print(f"ico3, basis order 4, a turn of {np.degrees(np.linalg.norm(turn)):.1f} degrees")
    for label, compose in ORDERS:
        SphericalWarp.compose = compose
        turned = SphericalWarp(grid)
        turned.vertices = grid.vertices @ Rotation.from_rotvec(turn).as_matrix().T
        turned.jacobian = turned._compute_jacobian()
        identity = measured(SphericalWarp(grid), SphericalWarp(grid))
        print(f"  {label:22s}: identity {identity:.3f}, turned {measured(turned, turned):.3f}")
    SphericalWarp.compose = gradient_order


if __name__ == "__main__":
    if sys.argv[1] == "known":
        known(int(sys.argv[2]))
    elif sys.argv[1] == "rate":
        rate()
    else:
        cohort()
    print("ENCORE ORDER PROBE DONE")
