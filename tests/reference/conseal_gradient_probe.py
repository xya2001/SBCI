"""How good is the ConSEAL registration gradient, and how wrong is the reference's?

For a synthetic pair of endpoint connectomes on an icosphere, differentiate
the actual cost -- move the endpoints along a basis field, relocate them,
re-smooth, compare -- by central differences, and set that against the
analytic coefficient ``dE/dbeta_k`` from (a) this port's formula and (b) the
reference's (``strict_upstream``). Also reports the agreement of the whole
gradient field with the finite-difference gradient through random
directions. Run by hand on Longleaf; PORTING.md item 7 quotes the numbers::

    python tests/reference/conseal_gradient_probe.py --subdivisions 4 --sigma 0.005 \
        --degree 30 --order 15 --streamlines 100000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sbci.alignment import SphericalGrid, sphere_exp_map
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, StationaryWarp, icosphere


def von_mises(rng, centre, concentration, n):
    centre = np.asarray(centre, dtype=np.float64) / np.linalg.norm(centre)
    points = rng.normal(size=(n, 3)) + concentration * centre
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def synthetic(rng, n, concentration, jitter=0.0):
    """Endpoint bundles; ``jitter`` moves the bundle centres so two subjects differ."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--streamlines", type=int, default=3000)
    parser.add_argument("--fields", type=int, default=8)
    parser.add_argument("--directions", type=int, default=6)
    parser.add_argument(
        "--full", action="store_true", help="also differentiate every left-hemisphere coefficient"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    vertices, faces = icosphere(args.subdivisions)
    grid = SphericalGrid(vertices, faces, order=args.order)
    n = grid.n_vertices
    print(
        f"ico{args.subdivisions}: {n} vertices per hemisphere, basis {grid.basis.shape[1]} fields"
    )

    t0 = time.time()
    fixed = EndpointConnectome.from_points(grid, grid, *synthetic(rng, args.streamlines, 4.0))
    moving = EndpointConnectome.from_points(
        grid, grid, *synthetic(rng, args.streamlines, 4.0, 0.15)
    )
    builder = HeatKernelBuilder(grid, grid, args.degree)
    kernel, derivative = builder.compute(args.sigma)
    kernel_up, derivative_up = builder.compute(args.sigma, strict_upstream=True)
    cut = builder.cutoff(args.sigma)
    print(f"kernel: {kernel.nnz / kernel.shape[0]:.1f} nonzeros per row, cutoff {cut:.4f}")
    print(f"  ({np.degrees(np.arccos(cut)):.1f} degrees); setup {time.time() - t0:.1f} s")

    q1 = fixed.q_transform(kernel)
    engine = ConSEAL(grid, grid)
    strict = ConSEAL(grid, grid, strict_upstream=True)

    def tangent(field):
        return grid.e1 * field[:, :1] + grid.e2 * field[:, 1:]

    def cost_of(field_lh, field_rh, t):
        """The actual cost after moving every endpoint along t * field."""
        moved = moving.copy()
        lh, rh = StationaryWarp(grid, viscosity=0.0), StationaryWarp(grid, viscosity=0.0)
        lh.vertices = sphere_exp_map(grid.vertices, t * tangent(field_lh))
        rh.vertices = sphere_exp_map(grid.vertices, t * tangent(field_rh))
        moved.warp(lh, rh)
        return engine.cost(q1 - moved.q_transform(kernel))

    def analytic(eng, kern, deriv):
        q2, q2_e1, q2_e2 = moving.q_transform(kern, deriv, eng.strict_upstream)
        difference = q1 - q2
        g_lh = eng._gradient(difference, q2, q2_e1, q2_e2, np.arange(n), grid)
        g_rh = eng._gradient(difference, q2, q2_e1, q2_e2, np.arange(n, 2 * n), grid)
        return g_lh, g_rh

    g_lh, g_rh = analytic(engine, kernel, derivative)
    u_lh, u_rh = analytic(strict, kernel_up, derivative_up)
    print(f"initial cost {engine.cost(q1 - moving.q_transform(kernel)):.6e}")
    cos_lh = (g_lh * u_lh).sum() / np.sqrt((g_lh**2).sum() * (u_lh**2).sum())
    cos_rh = (g_rh * u_rh).sum() / np.sqrt((g_rh**2).sum() * (u_rh**2).sum())
    print(f"cosine(port gradient, reference gradient): lh {cos_lh:.4f} rh {cos_rh:.4f}")
    ratio_lh = np.sqrt((u_lh**2).sum() / (g_lh**2).sum())
    ratio_rh = np.sqrt((u_rh**2).sum() / (g_rh**2).sum())
    print(f"norm ratio reference/port: lh {ratio_lh:.3f} rh {ratio_rh:.3f}")

    # --- per-field derivatives: dE/dbeta_k by finite differences vs the coefficients
    basis = grid.basis  # (P, B, 2)
    b_count = basis.shape[1]
    picks = np.linspace(0, b_count - 1, args.fields).astype(int)
    eps = 2e-3
    print("\nleft-hemisphere basis fields: finite difference vs analytic coefficient")
    header = f"{'k':>4} {'kind':>8} {'numeric':>13} {'port':>13} {'ratio':>7}"
    print(header + f" {'reference':>13} {'ratio':>7}")
    zero = np.zeros((n, 2))
    coeff_port = 2.0 * _coefficients(engine, moving, q1, kernel, derivative, n, grid)
    coeff_ref = 2.0 * _coefficients(strict, moving, q1, kernel_up, derivative_up, n, grid)
    for k in picks:
        field = basis[:, k, :]
        numeric = (cost_of(field, zero, eps) - cost_of(field, zero, -eps)) / (2 * eps)
        kind = "gradient" if k < b_count // 2 else "curl"
        r1 = numeric / coeff_port[k] if coeff_port[k] else np.nan
        r2 = numeric / coeff_ref[k] if coeff_ref[k] else np.nan
        row = f"{k:>4} {kind:>8} {numeric:>13.4e} {coeff_port[k]:>13.4e} {r1:>7.3f}"
        print(row + f" {coeff_ref[k]:>13.4e} {r2:>7.3f}")

    # --- random directions in coefficient space: agreement of the whole gradient
    print("\nrandom directions (left hemisphere): finite difference vs c . dE/dbeta")
    agree_port, agree_ref = [], []
    for _ in range(args.directions):
        c = rng.normal(size=b_count)
        c /= np.linalg.norm(c)
        field = (c[None, :, None] * basis).sum(axis=1)
        numeric = (cost_of(field, zero, eps) - cost_of(field, zero, -eps)) / (2 * eps)
        agree_port.append(numeric / (c @ coeff_port))
        agree_ref.append(numeric / (c @ coeff_ref))
    print(f"port:      ratios {np.round(agree_port, 3).tolist()}")
    print(f"reference: ratios {np.round(agree_ref, 3).tolist()}")

    # --- along each method's own descent direction
    for label, (a_lh, a_rh) in (("port", (g_lh, g_rh)), ("reference", (u_lh, u_rh))):
        scale = 1e-3 / max(np.linalg.norm(a_lh, axis=1).max(), np.linalg.norm(a_rh, axis=1).max())
        numeric = (cost_of(a_lh, a_rh, scale) - cost_of(a_lh, a_rh, -scale)) / (2 * scale)
        along = (grid.areas[:, None] * (g_lh * a_lh)).sum()
        along += (grid.areas[:, None] * (g_rh * a_rh)).sum()
        size = np.sqrt(
            (grid.areas[:, None] * a_lh**2).sum() + (grid.areas[:, None] * a_rh**2).sum()
        )
        print(f"\nalong the {label}'s descent direction: dE/dt numeric {numeric:.4e},")
        print(f"  port's prediction <g, a>_area {along:.4e}, ratio {numeric / along:.3f}")
        print(f"  decrease rate per unit area-norm of the step: {numeric / size:.4e}")

    if args.full:
        # the exact gradient of the discrete cost over every left-hemisphere coefficient
        print(f"\nfull finite-difference gradient over {b_count} left-hemisphere coefficients")
        t1 = time.time()
        exact = np.zeros(b_count)
        for k in range(b_count):
            field = basis[:, k, :]
            exact[k] = (cost_of(field, zero, eps) - cost_of(field, zero, -eps)) / (2 * eps)
        print(f"  took {time.time() - t1:.0f} s")

        def cosine(a, b):
            return float(a @ b / np.sqrt((a @ a) * (b @ b)))

        print(f"  cosine(exact, port)      {cosine(exact, coeff_port):.4f}")
        print(f"  cosine(exact, reference) {cosine(exact, coeff_ref):.4f}")
        print(
            f"  norm ratio port/exact      {np.linalg.norm(coeff_port) / np.linalg.norm(exact):.3f}"
        )
        print(
            f"  norm ratio reference/exact {np.linalg.norm(coeff_ref) / np.linalg.norm(exact):.3f}"
        )
        top = np.argsort(-np.abs(exact))[:20]
        c_port = cosine(exact[top], coeff_port[top])
        c_ref = cosine(exact[top], coeff_ref[top])
        print(f"  cosine on the 20 largest coefficients: port {c_port:.4f}, reference {c_ref:.4f}")
        # first-order decrease per unit coefficient norm along each direction: exact
        # gradient is the optimum, so these are fractions of what is achievable
        best = np.linalg.norm(exact)
        for label, c in (("port", coeff_port), ("reference", coeff_ref)):
            efficiency = (exact @ c) / (np.linalg.norm(c) * best)
            print(f"  descent efficiency of the {label}'s direction: {efficiency:.4f}")
    print(f"\ntotal {time.time() - t0:.1f} s")


def _coefficients(eng, moving, q1, kernel, derivative, n, grid):
    """The left-hemisphere coefficients dE/dbeta_k the engine's gradient is built from."""
    q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative, eng.strict_upstream)
    difference = q1 - q2
    rows = np.arange(n)
    block = difference[rows]
    if eng.area_weighted:
        block = block * eng._areas[None, :]
    s1 = (block * q2_e1[rows]).sum(axis=1)
    s2 = (block * q2_e2[rows]).sum(axis=1)
    s3 = (block * q2[rows]).sum(axis=1)
    integrand = (
        2 * s1[:, None] * grid.basis[:, :, 0]
        + 2 * s2[:, None] * grid.basis[:, :, 1]
        + s3[:, None] * grid.laplacian
    )
    if eng.area_weighted:
        integrand = integrand * grid.areas[:, None]
    return integrand.sum(axis=0)


if __name__ == "__main__":
    main()
