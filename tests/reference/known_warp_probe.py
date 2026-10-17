"""ENCORE on a known one-degree deformation of the synthetic subject: how much of it comes back.

    python tests/reference/known_warp_probe.py

USAGE.md's ENCORE notes say that halving a step that fails, rather than stopping there, lets
the port undo a small known deformation that the reference, which stops at the first failed
step, leaves as it is. PORTING.md item 7 (*Measured on a known deformation*) measured it at
59% before the review of 5 October 2026 corrected the divergence both gradients use (item 8)
and before item 21 composed each step as the gradient assumes; this measures it with the
package as it is. As there, the endpoints of ``sbci.example()`` are moved by a known smooth
warp, ``scripts/make_figures.py``'s ``known_warp`` (seed 7, largest displacement 0.07
radians, in ConSEAL's warp basis of order 15), and the original and the deformed copy are both
smoothed from their endpoints through the package's kernel (``make_figures.resmoothed``).
``sbci.align([deformed], template=original)`` registers the copy back, once with the basis of
order 15 the warp was drawn in, as item 7 did, and once with ENCORE's default order, 6; and
each moved endpoint is carried back through the inverse of ENCORE's warp, which is a
pull-back: it is located on the warped mesh and its barycentric weights are taken to the
unwarped vertices. Printed: how far the warp moves the grid's vertices and the endpoints, and
for each registration the mean distance of the endpoints from where they started, before and
after, the share of it undone, the cost and the steps taken -- with the default
``backtracks=4``, and with ``backtracks=0``, the reference's rule of stopping at the first
failed step. A registration that takes no step leaves the endpoints where the identity
carries them, within 0.002 degrees of where they were, which prints as -0.0% undone. Nothing
is read from disk; four minutes on the eight cores of an Apple M3.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_figures import angles, known_warp, resmoothed  # noqa: E402

import sbci  # noqa: E402
from sbci.alignment import DEFAULT_ORDER, MeshQuery, normalize_rows  # noqa: E402
from sbci.conseal import DEFAULT_WARP_ORDER, EndpointConnectome, default_grids  # noqa: E402


def carried_back(warp, moved, ends, grids):
    """Each moved endpoint through the inverse of ENCORE's warp, in the file's frame."""
    warped = (
        normalize_rows(warp.lh_vertices @ warp.lh_rotation),
        normalize_rows(warp.rh_vertices @ warp.rh_rotation),
    )
    back = []
    for points, hemispheres in zip(moved, ends, strict=True):
        placed = np.empty_like(points)
        for side, grid in enumerate(grids):
            pick = np.asarray(hemispheres) == side
            weights, indices = MeshQuery(warped[side], grid.faces).query(points[pick])
            placed[pick] = normalize_rows(np.einsum("nk,nkj->nj", weights, grid.vertices[indices]))
        back.append(placed)
    return back


def main() -> int:
    subject = sbci.example()
    lh, rh = default_grids()  # ConSEAL's grids, order 15, in the file's frame
    rng = np.random.default_rng(7)
    truth = [known_warp(grid, rng, 0.07) for grid in (lh, rh)]
    shift = np.r_[angles(lh.vertices, truth[0].vertices), angles(rh.vertices, truth[1].vertices)]
    print(
        f"the known warp moves the grid's vertices {shift.mean():.2f} degrees on average, "
        f"{shift.max():.2f} at most",
        flush=True,
    )
    original = resmoothed(subject, subject.endpoints)
    carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
    start = carrier.positions()
    carrier.warp(*truth)
    deformed = resmoothed(subject, carrier.to_endpoints())
    moved = EndpointConnectome.from_endpoints(deformed.endpoints, lh, rh).positions()
    before = np.r_[angles(start[0], moved[0]), angles(start[1], moved[1])]
    print(
        f"and the {subject.endpoints.n_streamlines} streamlines' endpoints {before.mean():.2f} "
        f"on average, {before.max():.2f} at most",
        flush=True,
    )
    ends = (carrier.hemisphere_in, carrier.hemisphere_out)
    for order in (DEFAULT_WARP_ORDER, DEFAULT_ORDER):
        for backtracks in (4, 0):
            t = time.time()
            result = sbci.align([deformed], template=original, order=order, backtracks=backtracks)
            back = carried_back(result.warps[0], moved, ends, (lh, rh))
            after = np.r_[angles(start[0], back[0]), angles(start[1], back[1])]
            trace = result.traces[0]
            print(
                f"order {order:2d}, backtracks={backtracks}: endpoints {before.mean():.3f} -> "
                f"{after.mean():.3f} degrees, {1 - after.mean() / before.mean():.1%} undone; "
                f"cost {trace[0]:.4g} -> {trace[-1]:.4g} in {len(trace) - 1} steps; "
                f"{time.time() - t:.0f} s",
                flush=True,
            )
    print("KNOWN WARP PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
