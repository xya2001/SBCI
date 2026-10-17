"""How the package's ConSEAL error falls, iteration by iteration, on a small known warp.

    module load python/3.12.4
    python tests/reference/conseal_convergence_probe.py CASE DELTA [ITERATIONS]

Registers a case of ``fork_synthetic_cases.py`` back onto the undeformed subject at the
paper's settings with step ``DELTA`` and a threshold of the smallest positive float, so the
descent runs until the cost stops falling or ``ITERATIONS`` (300) are taken, and records after
every step the cost and the mean distance of the moved endpoints from where the known warp
moved them from. Where a threshold would have stopped it can then be read off for every
threshold at once: ``register`` keeps the step whose fall first drops below the threshold and
stops after it, and rolls back only a step that raises the cost.
"""

import sys
import time

import numpy as np

import sbci.conseal as conseal
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

CASES = "/work/users/x/y/xya/conseal-fork-compare/cases"
case, delta = sys.argv[1], float(sys.argv[2])
iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 300
x = np.load(f"{CASES}/{case}.npz")
lh, rh = default_grids()


def carrier(prefix):
    flags = [x[f"{prefix}_h{end}"].ravel().astype(np.int64) for end in ("in", "out")]
    return EndpointConnectome.from_points(lh, rh, x[f"{prefix}_in"], x[f"{prefix}_out"], *flags)


truth = (x["truth_in"], x["truth_out"])


def degrees_off(points):
    angles = [
        np.arccos(np.clip((p * q).sum(1), -1.0, 1.0)) for p, q in zip(points, truth, strict=True)
    ]
    return float(np.degrees(np.concatenate(angles)).mean())


errors = []
warp = EndpointConnectome.warp


def recording(self, lh_warp, rh_warp):
    """The package's warp, recording how far the endpoints then are from the truth."""
    points = warp(self, lh_warp, rh_warp)
    errors.append(degrees_off(points))
    return points


conseal.EndpointConnectome.warp = recording
fixed, moving = carrier("fixed"), carrier("moving")
kernel, derivative = HeatKernelBuilder(lh, rh, 30).compute(0.005)
# Not 0: a step refused in both hemispheres leaves the cost exactly where it was, and at 0 the
# descent would repeat that refused step until ITERATIONS.
engine = ConSEAL(
    lh,
    rh,
    delta=delta,
    max_iterations=iterations,
    threshold=np.finfo(float).tiny,
    step_clamp=np.inf,
    viscosity=0.0,
)
start = time.time()
_, _, costs, warped = engine.register(fixed, moving, kernel, derivative)
elapsed = time.time() - start
before = degrees_off(moving.positions())
final = degrees_off(warped.positions())
print(
    f"{case}, delta {delta}: {costs.size - 1} iterations in {elapsed / 60:.1f} min; endpoints "
    f"{before:.4f} deg from the truth at the start, {final:.4f} at the end"
)
steps = errors[: costs.size - 1]
for k in range(len(steps)):
    if k < 10 or (k + 1) % 10 == 0 or k == len(steps) - 1:
        print(
            f"  iteration {k + 1:4d}: cost {costs[k + 1]:.3e} "
            f"(fell {costs[k] - costs[k + 1]:.2e}), error {steps[k]:.4f} deg"
        )
falls = costs[:-1] - costs[1:]
for threshold in (1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9):
    below = np.flatnonzero(falls < threshold)
    # the step whose fall first drops below the threshold is kept, and the descent ends with it
    taken = int(below[0]) + 1 if below.size else len(falls)
    print(
        f"  threshold {threshold:.0e} would stop after {taken} iterations at error "
        f"{(steps[taken - 1] if taken else before):.4f} deg"
    )
np.savez(
    f"{CASES}/{case}_convergence_delta{delta:g}.npz",
    costs=costs,
    errors=np.array(steps),
    before=before,
    elapsed=elapsed,
)
print("CONVERGENCE PROBE DONE")
