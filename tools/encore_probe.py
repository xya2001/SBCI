"""ENCORE on a known warp of a real subject: where its cost's minimum is, and when it recovers it.

    python tools/encore_probe.py oracle SUBJECT_sc.h5
    python tools/encore_probe.py variants SUBJECT_sc.h5

``oracle`` walks the cost along the true warp (Jacobian included), measures
the floor a reference through a different smoother puts under it, undoes the
warp exactly at the endpoint level, and evaluates ConSEAL's answer in ENCORE's
cost. ``variants`` repeats the recovery over the warp's smoothness and
amplitude, ENCORE's basis order, the reference and the derivative estimator.
Both register one subject's streamlines many times over, ENCORE and ConSEAL
alike, and print their measurements: a batch job of hours on eight cores.

The subject file must carry endpoints with positions (``tools/build_hcp_cohort.py``
writes them; ``sbci download hcp-ya`` fetches some). PORTING.md item 4,
"Measured on a known deformation", records what these measured: a reference
through a different smoother than the deformed copy, and a warp with structure
at the grid scale, are what made ENCORE look as if it recovered nothing.
"""

import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci.alignment import (
    Encore,
    MeshQuery,
    SphericalWarp,
    _hemisphere_grids,
    normalize_rows,
    sphere_exp_map,
    sphere_log_map,
)
from sbci.connectome import ContinuousConnectome
from sbci.conseal import EndpointConnectome, StationaryWarp, default_grids
from sbci.smoothing import endpoint_positions


def log(message):
    print(message, flush=True)


def angles(p, q):
    return np.degrees(np.arccos(np.clip((p * q).sum(axis=1), -1.0, 1.0)))


def tangent(base, moved):
    d = moved - base
    return d - (d * base).sum(1, keepdims=True) * base


def cosine(u, v):
    return float((u * v).sum() / np.sqrt((u**2).sum() * (v**2).sum()))


def resmoothed(cc, endpoints):
    carrier = ContinuousConnectome(
        data=np.zeros_like(cc.data),
        area=cc.area,
        mask=cc.mask,
        metadata=cc.metadata,
        coords=cc.coords,
        endpoints=endpoints,
    )
    return carrier.smooth(kernel="shk", mask_medial_wall=True)


def known_warp(grid, rng, amplitude):
    warp = StationaryWarp(grid)
    coefficients = rng.standard_normal(grid.basis.shape[1])
    displacement = (coefficients[None, :, None] * grid.basis).sum(axis=1)
    displacement *= amplitude / np.linalg.norm(displacement, axis=1).max()
    assert warp.compose(displacement)
    return warp


class Probe:
    """One subject, one known warp of its endpoints, and ENCORE's cost around it."""

    def __init__(self, subject, warp_order, amplitude, encore_order, reference="stored", seed=7):
        self.subject = subject
        lh_w, rh_w = default_grids(warp_order)  # the known warp is a random field of this order
        self.lh, self.rh = default_grids(15)
        rng = np.random.default_rng(seed)
        self.lh_true = known_warp(lh_w, rng, amplitude)
        self.rh_true = known_warp(rh_w, rng, amplitude)
        self.vertex_true = [normalize_rows(w.vertices) for w in (self.lh_true, self.rh_true)]
        carrier = EndpointConnectome.from_endpoints(subject.endpoints, self.lh, self.rh)
        self.original = carrier.positions()
        carrier.warp(self.lh_true, self.rh_true)
        self.carrier = carrier
        self.deformed = resmoothed(subject, carrier.to_endpoints())
        moved = EndpointConnectome.from_endpoints(self.deformed.endpoints, self.lh, self.rh)
        self.moved = moved.positions()
        self.before = np.r_[
            angles(self.original[0], self.moved[0]), angles(self.original[1], self.moved[1])
        ]
        self.grids, self.rotations = _hemisphere_grids(encore_order, return_rotations=True)
        for side, grid in enumerate((self.lh, self.rh)):
            rotated = grid.vertices @ self.rotations[side].T
            assert np.allclose(self.grids[side].vertices, rotated, atol=1e-6)
        self.engine = Encore(*self.grids)
        self.reference = (
            subject if reference == "stored" else resmoothed(subject, subject.endpoints)
        )
        self.fixed = self.engine.root(self.reference.dense())
        self.source = self.engine.root(self.deformed.dense())
        self.start = self.cost_of_root(self.source)

    def cost_of_root(self, root):
        """ENCORE's cost of a square-root density against the reference."""
        return float(((self.fixed - root) ** 2 * self.engine.area_product).sum())

    def pullback(self, vertices_pair):
        """ENCORE warps, Jacobians included, sending grid vertex v to vertices_pair[side][v]."""
        warps = []
        for side in range(2):
            w = SphericalWarp(self.grids[side])
            w.vertices = normalize_rows(np.asarray(vertices_pair[side]) @ self.rotations[side].T)
            w.jacobian = w._compute_jacobian()
            warps.append(w)
        return warps

    def cost_of_pullback(self, vertices_pair):
        """ENCORE's cost of the deformed subject pulled back through a warp (grid frame)."""
        warps = self.pullback(vertices_pair)
        return self.cost_of_root(self.engine.concon.evaluate_root(self.source, *warps))

    def endpoints_after(self, vertices_pair):
        """How far each moved endpoint is from its start once put back through a warp's inverse."""
        out = []
        ends = (self.carrier.hemisphere_in, self.carrier.hemisphere_out)
        for points, hemispheres in zip(self.moved, ends, strict=True):
            placed = np.empty_like(points)
            for side, grid in enumerate((self.lh, self.rh)):
                pick = np.asarray(hemispheres) == side
                query = MeshQuery(normalize_rows(vertices_pair[side]), grid.faces)
                weights, indices = query.query(points[pick])
                combined = np.einsum("nk,nkj->nj", weights, grid.vertices[indices])
                placed[pick] = normalize_rows(combined)
            out.append(placed)
        return np.r_[angles(self.original[0], out[0]), angles(self.original[1], out[1])]

    def field_cosine(self, vertices_pair):
        """Cosine between a warp's displacement field and the true one, per hemisphere."""
        return [
            cosine(
                tangent(grid.vertices, self.vertex_true[side]),
                tangent(grid.vertices, normalize_rows(vertices_pair[side])),
            )
            for side, grid in enumerate((self.lh, self.rh))
        ]

    def run_encore(self, max_iterations=50, step=0.05, derivative="difference", backtracks=4):
        """Register the deformed copy onto the reference; returns the warp, trace and seconds."""
        t = time.time()
        engine = Encore(
            *self.grids,
            step=step,
            max_iterations=max_iterations,
            derivative=derivative,
            backtracks=backtracks,
        )
        trace = [self.start]
        _, lh_warp, rh_warp, _ = engine.register(
            self.fixed,
            self.deformed.dense(),
            target_is_root=True,
            callback=lambda _i, c: trace.append(float(c)),
        )
        found = (lh_warp.vertices @ self.rotations[0], rh_warp.vertices @ self.rotations[1])
        return found, np.asarray(trace), time.time() - t

    def report(self, label, found, trace, seconds):
        """Print what a registration achieved; returns the endpoint residuals."""
        after = self.endpoints_after(found)
        moved = np.r_[
            angles(self.lh.vertices, normalize_rows(found[0])),
            angles(self.rh.vertices, normalize_rows(found[1])),
        ]
        cos = self.field_cosine(found)
        undone = 1 - after.mean() / self.before.mean()
        log(
            f"  {label}: {len(trace) - 1} steps in {seconds:.0f}s; "
            f"cost {trace[0]:.5f} -> {trace[-1]:.5f} ({trace[-1] / trace[0]:.2f}); "
            f"endpoints {self.before.mean():.2f} -> {after.mean():.2f} deg ({undone:.0%} undone); "
            f"vertices moved {moved.mean():.2f} deg mean, {moved.max():.2f} max; "
            f"field cosine with the truth L {cos[0]:+.2f} R {cos[1]:+.2f}"
        )
        return after


def oracle(subject):
    log("== oracle: a degree-15 warp of 0.07 rad at most, ENCORE at order 15 ==")
    for reference in ("stored", "resmoothed"):
        t = time.time()
        probe = Probe(subject, warp_order=15, amplitude=0.07, encore_order=15, reference=reference)
        log(
            f"[{reference} reference] built in {time.time() - t:.0f}s: deformation "
            f"{probe.before.mean():.2f} deg mean, {probe.before.max():.2f} max; "
            f"start cost {probe.start:.5f}"
        )
        if reference == "stored":
            same = probe.engine.root(resmoothed(subject, subject.endpoints).dense())
            log(
                "  floor: the stored density against its own endpoints through sbci's smoother: "
                f"cost {probe.cost_of_root(same):.5f}"
            )
        base = [probe.lh.vertices, probe.rh.vertices]
        logs = [sphere_log_map(base[s], probe.vertex_true[s]) for s in range(2)]
        check = [normalize_rows(sphere_exp_map(base[s], logs[s])) for s in range(2)]
        worst = max(angles(check[s], probe.vertex_true[s]).max() for s in range(2))
        log(f"  check: exp(log(true warp)) is {worst:.2e} deg from the true warp")
        for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            vertices = [normalize_rows(sphere_exp_map(base[s], t * logs[s])) for s in range(2)]
            log(
                f"  along the true warp, t = {t:.2f}: cost {probe.cost_of_pullback(vertices):.5f}; "
                f"endpoints {probe.endpoints_after(vertices).mean():.3f} deg from the start"
            )
        undo = EndpointConnectome.from_endpoints(probe.deformed.endpoints, probe.lh, probe.rh)
        undo.warp(probe.lh_true.copy().invert(), probe.rh_true.copy().invert())
        back = undo.positions()
        residual = np.r_[angles(probe.original[0], back[0]), angles(probe.original[1], back[1])]
        undone = probe.engine.root(resmoothed(subject, undo.to_endpoints()).dense())
        log(
            f"  endpoint-level exact undo: endpoints {residual.mean():.3f} deg from the start; "
            f"re-smoothed, cost {probe.cost_of_root(undone):.5f}"
        )
        found, trace, seconds = probe.run_encore()
        probe.report("ENCORE", found, trace, seconds)
        log(
            "  check: ENCORE's warp re-evaluated through the pull-back path costs "
            f"{probe.cost_of_pullback(found):.5f} (its trace ends at {trace[-1]:.5f})"
        )
        if reference != "stored":
            continue
        t = time.time()
        conseal = sbci.endpoints_align(
            [subject, probe.deformed],
            template=0,
            max_iterations=60,
            threshold=1e-7,
            delta=0.1,
            step_clamp=float("inf"),
            viscosity=0.0,
        )
        back = endpoint_positions(conseal.aligned_endpoints(1))
        after = np.r_[angles(probe.original[0], back[0]), angles(probe.original[1], back[1])]
        log(
            f"  ConSEAL in {time.time() - t:.0f}s: endpoints {probe.before.mean():.2f} -> "
            f"{after.mean():.2f} deg; its own cost {conseal.costs[1][0]:.5f} -> "
            f"{conseal.costs[1][-1]:.5f}"
        )
        aligned = probe.engine.root(resmoothed(subject, conseal.aligned_endpoints(1)).dense())
        log(
            "  ConSEAL's aligned endpoints re-smoothed, in ENCORE's cost: "
            f"{probe.cost_of_root(aligned):.5f}"
        )
        warp = conseal.warps[1]
        for name, sign in (
            ("the inverse of ConSEAL's warp as pull-back", -1.0),
            ("ConSEAL's warp as pull-back", 1.0),
        ):
            vertices = []
            for grid, velocity in ((probe.lh, warp.lh_velocity), (probe.rh, warp.rh_velocity)):
                w = StationaryWarp(grid)
                w.velocity = sign * np.asarray(velocity)
                w.vertices = w.exponential()
                vertices.append(w.vertices)
            log(
                f"  {name}: ENCORE cost {probe.cost_of_pullback(vertices):.5f}; "
                f"field cosine with the truth {probe.field_cosine(vertices)[0]:+.2f}"
            )


def variants(subject):
    configs = [
        # warp order, amplitude (rad, max), ENCORE order, reference, derivative
        (15, 0.07, 15, "resmoothed", "difference"),
        (15, 0.07, 6, "resmoothed", "difference"),
        (4, 0.07, 6, "resmoothed", "difference"),
        (4, 0.07, 15, "resmoothed", "difference"),
        (4, 0.12, 6, "resmoothed", "difference"),
        (2, 0.10, 6, "resmoothed", "difference"),
        (4, 0.07, 6, "resmoothed", "analytic"),
        (4, 0.07, 6, "stored", "difference"),
    ]
    for warp_order, amplitude, encore_order, reference, derivative in configs:
        t = time.time()
        probe = Probe(subject, warp_order, amplitude, encore_order, reference)
        log(
            f"== warp order {warp_order}, max {np.degrees(amplitude):.1f} deg "
            f"(mean {probe.before.mean():.2f}); ENCORE order {encore_order}; {reference} "
            f"reference; {derivative} derivative (built in {time.time() - t:.0f}s) =="
        )
        log(
            f"  start cost {probe.start:.5f}; "
            f"at the true warp {probe.cost_of_pullback(probe.vertex_true):.5f}"
        )
        found, trace, seconds = probe.run_encore(max_iterations=100, derivative=derivative)
        probe.report("ENCORE", found, trace, seconds)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("oracle", "variants"):
        sys.exit(__doc__)
    mode, path = sys.argv[1], sys.argv[2]
    subject = sbci.load(path)
    log(f"loaded {Path(path).name}: {subject.endpoints.vtx_in.size:,} streamlines")
    {"oracle": oracle, "variants": variants}[mode](subject)
    log("done")
