"""ConSEAL on real ico4 endpoints: three ADNI subjects at the published settings.

Loads each subject's ``mesh_intersections_ico4.mat`` (the barycentric endpoint
crossings the SBCI pipeline wrote), aligns the three with
:func:`sbci.conseal.endpoints_align` at the public code's defaults (sigma
0.005, degree 30, warp order 15, step 0.05, up to 100 iterations, threshold
1e-4), and reports what a reviewer would ask for: cost traces, iterations,
refused steps, Jacobian ranges, pairwise square-root-density distances and
endpoint-overlap coefficients before and after, and wall time. Run by hand on
Longleaf through SLURM; it takes hours. Lab data is read only.

    python tests/reference/conseal_adni_ico4.py /work/users/x/y/xya/conseal-ref
"""

from __future__ import annotations

import sys
import time

import numpy as np

from sbci.conseal import (
    EndpointConnectome,
    HeatKernelBuilder,
    default_grids,
    endpoints_align,
    overlap_coefficient,
)
from sbci.smoothing import Endpoints

ROOT = "/overflow/zzhanglab/ADNI/ADNI-bids"
SUBJECTS = ("sub-168S6561", "sub-068S0473", "sub-002S0413")
THRESHOLD = 1e-4  # endpoint density per vertex; the mean over 5,124 vertices is 1.95e-4


def distance(a, b):
    return float(np.arccos(np.clip((a * b).sum(), -1.0, 1.0)))


def matrix(function, items):
    return [[round(function(a, b), 4) for b in items] for a in items]


def main(out):
    t0 = time.time()
    lh, rh = default_grids(15)
    subjects = []
    for name in SUBJECTS:
        path = f"{ROOT}/{name}/dwi_pipeline/sbci_connectome/mesh_intersections_ico4.mat"
        endpoints = Endpoints.from_matlab(path)
        subjects.append(EndpointConnectome.from_endpoints(endpoints, lh, rh))
        print(f"{name}: {subjects[-1].n_streamlines} streamlines", flush=True)
    builder = HeatKernelBuilder(lh, rh, 30)
    kernel = builder.compute(0.005, derivative=False)
    cut = builder.cutoff(0.005)
    degrees = np.degrees(np.arccos(cut))
    print(f"kernel: {kernel.nnz / kernel.shape[0]:.1f} nonzeros per row, cutoff {degrees:.2f} deg")
    qs = [c.q_transform(kernel) for c in subjects]
    print("pairwise Q distance before:", matrix(distance, qs))
    print(
        "overlap coefficient before:",
        matrix(lambda a, b: overlap_coefficient(a, b, THRESHOLD), subjects),
    )
    print(f"setup {time.time() - t0:.0f} s", flush=True)

    t1 = time.time()
    result = endpoints_align(subjects, grids=(lh, rh), verbose=True)
    elapsed = time.time() - t1
    print(f"alignment {elapsed:.0f} s")

    rows = zip(SUBJECTS, result.costs, result.warps, result.rejected_steps, strict=True)
    for name, costs, warp, rejected in rows:
        jac = np.concatenate([warp.lh_jacobian, warp.rh_jacobian])
        lower = 100 * (1 - costs[-1] / costs[0])
        monotone = bool(np.all(np.diff(costs) <= 0))
        print(f"{name}: cost {costs[0]:.6f} -> {costs[-1]:.6f} ({lower:.1f}% lower)")
        print(f"  {len(costs) - 1} iterations, {rejected} refused steps, monotone {monotone}")
        print(f"  Jacobian {jac.min():.3f} to {jac.max():.3f}")
        warp.save(f"{out}/{name}_conseal_warp.npz")
        aligned = result.aligned_endpoints(SUBJECTS.index(name))
        np.savez_compressed(
            f"{out}/{name}_conseal_endpoints.npz",
            surf_in=aligned.surf_in,
            surf_out=aligned.surf_out,
            vtx_in=aligned.vtx_in,
            vtx_out=aligned.vtx_out,
            tri_in=aligned.tri_in,
            tri_out=aligned.tri_out,
            bary_in=aligned.bary_in,
            bary_out=aligned.bary_out,
        )
    after = [c.q_transform(kernel) for c in result.connectomes]
    print("distance to template before:", [round(distance(q, result.template), 4) for q in qs])
    print("distance to template after: ", [round(distance(q, result.template), 4) for q in after])
    print("pairwise Q distance after:", matrix(distance, after))
    print(
        "overlap coefficient after:",
        matrix(lambda a, b: overlap_coefficient(a, b, THRESHOLD), result.connectomes),
    )
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/work/users/x/y/xya/conseal-ref")
