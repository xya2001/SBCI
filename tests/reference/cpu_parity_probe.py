"""Whether two versions of the code compute the same thing on the CPU, bit for bit.

    PYTHONPATH=<one tree>/src python tests/reference/cpu_parity_probe.py record one.npz
    PYTHONPATH=<other tree>/src python tests/reference/cpu_parity_probe.py record other.npz
    python tests/reference/cpu_parity_probe.py compare one.npz other.npz

``record`` runs ENCORE and ConSEAL on synthetic subjects and saves what they return: on a
level-2 icosphere, ENCORE's root, its template to the rule and after ten steps, a
registration, and ``align`` with an estimated and a given template; ConSEAL's template,
cost, a plain, an area-weighted and a rotation-searching registration, and
``endpoints_align`` with each kind of template; and on the bundled ico4 grids, ENCORE's
root with a registration step and ConSEAL's density with its derivatives by both of
``evaluate``'s routes, a three-subject template and a registration step (an array of the
whole ico4 grid is kept as the SHA-256 of its bytes and every fiftieth row, to keep the
record small). ``compare`` says which arrays are identical and, for any that are not, how
far apart they lie. Run both
records on one node: processors of another kind may round differently (PORTING.md item
26 used it to show that the GPU left the CPU unchanged, items 26 and 27 record its runs).
"""

import hashlib
import sys

import numpy as np

BIG = 1_000_000  # entries past which an array is kept as its digest and every fiftieth row


def kept(out, name, array):
    """Keep ``array`` under ``name``; a big one as its digest and a sample of its rows."""
    array = np.asarray(array)
    if array.size <= BIG:
        out[name] = array
        return
    out[name + "_sha256"] = np.array(
        hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    )
    out[name + "_rows"] = array[::50]


def bundles(vertices, ends, width=6.0):
    """A density of smooth bundles between the given directions, over both hemispheres."""
    both = np.vstack([vertices, vertices])
    density = np.zeros((both.shape[0], both.shape[0]))
    for a, b in ends:
        ka = np.exp(width * (both @ (np.asarray(a) / np.linalg.norm(a)) - 1))
        kb = np.exp(width * (both @ (np.asarray(b) / np.linalg.norm(b)) - 1))
        density += np.outer(ka, kb) + np.outer(kb, ka)
    np.fill_diagonal(density, 0.0)
    return density


def von_mises(rng, centre, concentration, n):
    centre = np.asarray(centre, dtype=np.float64) / np.linalg.norm(centre)
    points = rng.normal(size=(n, 3)) + concentration * centre
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def synthetic(rng, n=3000, concentration=4.0, jitter=0.0):
    """Endpoints in bundles within the left, within the right and across, as the tests make."""
    thirds = n // 3
    j = jitter * rng.normal(size=(6, 3))
    p_in = np.vstack(
        [
            von_mises(rng, np.add([-1, 0.3, 0.2], j[0]), concentration, thirds),
            von_mises(rng, np.add([1, -0.2, 0.4], j[1]), concentration, thirds),
            von_mises(rng, np.add([-0.8, -0.5, 0.1], j[2]), concentration, n - 2 * thirds),
        ]
    )
    p_out = np.vstack(
        [
            von_mises(rng, np.add([-0.5, -0.6, 0.6], j[3]), concentration, thirds),
            von_mises(rng, np.add([0.7, 0.6, -0.3], j[4]), concentration, thirds),
            von_mises(rng, np.add([0.9, 0.3, -0.3], j[5]), concentration, n - 2 * thirds),
        ]
    )
    h_in = np.r_[np.zeros(thirds), np.ones(thirds), np.zeros(n - 2 * thirds)].astype(int)
    h_out = np.r_[np.zeros(thirds), np.ones(thirds), np.ones(n - 2 * thirds)].astype(int)
    return p_in, p_out, h_in, h_out


def record(path):
    from sbci.alignment import Encore, SphericalGrid, _hemisphere_grids, align, rotate_off_poles
    from sbci.conseal import (
        ConSEAL,
        EndpointConnectome,
        HeatKernelBuilder,
        default_grids,
        endpoints_align,
        icosphere,
    )

    out = {}

    def keep(prefix, warp):
        out[prefix + "_vertices"] = warp.vertices
        out[prefix + "_jacobian"] = warp.jacobian

    # ENCORE on a level-2 icosphere
    vertices, faces = icosphere(2)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    rng = np.random.default_rng(5)
    densities = [
        bundles(grid.vertices, [(rng.normal(size=3), rng.normal(size=3)) for _ in range(3)], 4.0)
        for _ in range(6)
    ]
    encore = Encore(grid, grid)
    out["encore_template_rule"] = encore.template(densities, iterations=1000)
    out["encore_template_10"] = encore.template(densities, iterations=10)
    out["encore_root"] = encore.root(densities[2])
    aligned, lh, rh, cost = encore.register(
        out["encore_template_rule"], densities[1], target_is_root=True
    )
    out["encore_aligned"], out["encore_cost"] = aligned, np.array(cost)
    keep("encore_lh", lh)
    keep("encore_rh", rh)
    result = align(densities[:4], grids=(grid, grid), max_iterations=6)
    out["align_template"] = result.template
    for i, (a, t, c) in enumerate(zip(result.aligned, result.traces, result.costs, strict=True)):
        out[f"align_aligned{i}"], out[f"align_trace{i}"], out[f"align_cost{i}"] = a, t, np.array(c)
        out[f"align_warp{i}"] = result.warps[i].lh_vertices
    given = align(
        densities[:3], grids=(grid, grid), max_iterations=4, template=encore.root(densities[3])
    )
    out["align_given"] = given.template
    out["align_given_trace0"] = given.traces[0]

    # ConSEAL on a level-2 icosphere
    vertices, faces = icosphere(2)
    cgrid = SphericalGrid(vertices, faces, order=3)
    rng = np.random.default_rng(11)
    subjects = [
        EndpointConnectome.from_points(cgrid, cgrid, *synthetic(rng, jitter=0.15)) for _ in range(4)
    ]
    kernel, derivative = HeatKernelBuilder(cgrid, cgrid, degree=12).compute(0.05)
    engine = ConSEAL(cgrid, cgrid, max_iterations=8)
    out["conseal_template"] = engine.template(subjects, kernel)
    out["conseal_cost"] = np.array(
        engine.cost(out["conseal_template"] - subjects[0].q_transform(kernel))
    )
    lh, rh, costs, warped = engine.register(
        out["conseal_template"], subjects[1], kernel, derivative
    )
    out["conseal_costs"] = costs
    out["conseal_lh_velocity"] = lh.velocity
    for name in ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out"):
        out["conseal_" + name] = getattr(warped, name)
    weighted = ConSEAL(cgrid, cgrid, max_iterations=5, area_weighted=True)
    lh, rh, costs, warped = weighted.register(subjects[0], subjects[2], kernel, derivative)
    out["conseal_weighted_costs"] = costs
    out["conseal_weighted_weights"] = warped.weights_in
    lh, rh, costs, warped = ConSEAL(cgrid, cgrid, max_iterations=3).register(
        subjects[0], subjects[3], kernel, derivative, init_rotation=True
    )
    out["conseal_rigid_costs"] = costs
    out["conseal_rigid_rotation"] = lh.rigid
    for label, template in (("estimated", None), ("index", 2), ("fixed", subjects[3])):
        result = endpoints_align(
            subjects[:3],
            template=template,
            sigma=0.05,
            kernel_degree=12,
            order=3,
            max_iterations=4,
            grids=(cgrid, cgrid),
        )
        out[f"endpoints_{label}_template"] = result.template
        for i, costs in enumerate(result.costs):
            out[f"endpoints_{label}_costs{i}"] = costs
            out[f"endpoints_{label}_weights{i}"] = result.connectomes[i].weights_out
            out[f"endpoints_{label}_velocity{i}"] = result.warps[i].rh_velocity

    # the bundled ico4 grids: ENCORE's turned off the axis, ConSEAL's as stored
    lh_grid, rh_grid = _hemisphere_grids(2)
    rng = np.random.default_rng(3)
    ends = [[(rng.normal(size=3), rng.normal(size=3)) for _ in range(3)] for _ in range(2)]
    pair = [bundles(lh_grid.vertices, e, 4.0) for e in ends]
    big = Encore(lh_grid, rh_grid, max_iterations=1)
    kept(out, "ico4_encore_root", big.root(pair[0]))
    aligned, lh, _, cost = big.register(pair[0], pair[1])
    kept(out, "ico4_encore_aligned", aligned)
    out["ico4_encore_cost"] = np.array(cost)
    out["ico4_encore_vertices"] = lh.vertices
    del pair, aligned

    import sbci.conseal as conseal

    lh_grid, rh_grid = default_grids(order=3)
    rng = np.random.default_rng(13)
    subjects = [
        EndpointConnectome.from_points(lh_grid, rh_grid, *synthetic(rng, n=20000, jitter=0.15))
        for _ in range(3)
    ]
    kernel, derivative = HeatKernelBuilder(lh_grid, rh_grid).compute()
    for route, fraction in (("sparse", np.inf), ("dense", 0.0)):
        saved, conseal.SPARSE_PRODUCT_FRACTION = conseal.SPARSE_PRODUCT_FRACTION, fraction
        try:
            q, d_e1, d_e2 = subjects[0].q_transform(kernel, derivative)
        finally:
            conseal.SPARSE_PRODUCT_FRACTION = saved
        kept(out, f"ico4_conseal_{route}_q", q)
        kept(out, f"ico4_conseal_{route}_e1", d_e1)
        kept(out, f"ico4_conseal_{route}_e2", d_e2)
        del q, d_e1, d_e2
    engine = ConSEAL(lh_grid, rh_grid, max_iterations=1)
    template = engine.template(subjects, kernel, iterations=3)
    kept(out, "ico4_conseal_template", template)
    lh, _, costs, warped = engine.register(template, subjects[1], kernel, derivative)
    out["ico4_conseal_costs"] = costs
    out["ico4_conseal_velocity"] = lh.velocity
    out["ico4_conseal_weights"] = warped.weights_in

    np.savez(path, **out)
    print(f"{len(out)} arrays recorded in {path}")


def compare(first, second):
    one, other = np.load(first), np.load(second)
    names = sorted(set(one.files) | set(other.files))
    missing = [name for name in names if name not in one.files or name not in other.files]
    same, apart = [], []
    for name in names:
        if name in missing:
            continue
        a, b = one[name], other[name]
        if a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes():
            same.append(name)
        elif a.shape == b.shape and np.array_equal(a, b):
            apart.append(f"{name}: equal values, other bits (the sign of a zero)")
        elif a.shape != b.shape:
            apart.append(f"{name}: shapes {a.shape} and {b.shape}")
        else:
            scale = max(float(np.abs(b).max()), 1e-300)
            largest = float(np.abs(a.astype(float) - b.astype(float)).max())
            share = largest / scale
            apart.append(f"{name}: largest difference {largest:.3g}, {share:.3g} of the largest")
    print(f"{len(same)} of {len(names)} arrays identical, bit for bit")
    for line in apart:
        print(f"  {line}")
    for name in missing:
        print(f"  {name}: in one record only")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "record":
        record(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
