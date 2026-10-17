"""Where one ConSEAL iteration of the package spends its time, on a whole ico4 subject.

    module load python/3.12.4
    python tests/reference/conseal_profile.py    # 8 cores; reads fork_cases.py's known_007

On an AMD EPYC 75F3 (PORTING.md item 19): the density alone 1.0 to 1.5 s, with its
derivatives 1.6 to 1.7 s, the gradient 0.25 s, the step 0.18 s, and carrying the 803,741
streamlines' endpoints with the warp and locating them again on the grid 9.8 s.
"""

import time

import numpy as np

from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

x = np.load("/work/users/x/y/xya/conseal-fork-compare/cases/known_007.npz")
lh, rh = default_grids()


def connectome(prefix):
    flags = [x[f"{prefix}_h{end}"].ravel().astype(np.int64) for end in ("in", "out")]
    return EndpointConnectome.from_points(lh, rh, x[f"{prefix}_in"], x[f"{prefix}_out"], *flags)


fixed, moving = connectome("fixed"), connectome("moving")
t = time.time()
kernel, derivative = HeatKernelBuilder(lh, rh, 30).compute(0.005)
print(f"kernel: {time.time() - t:.1f}s, {kernel.nnz / kernel.shape[0] ** 2:.1%} filled")
engine = ConSEAL(lh, rh, delta=0.1, threshold=1e-6, step_clamp=np.inf, viscosity=0.0)
q1 = fixed.q_transform(kernel)
lh_warp, rh_warp = engine.new_warps()
moving = moving.copy()
moving.commit()
rows_lh, rows_rh = slice(0, moving.n_left), slice(moving.n_left, moving.n_vertices)
for iteration in range(3):
    t0 = time.time()
    connectome_only = moving.evaluate(kernel)
    t1 = time.time()
    q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative)
    t2 = time.time()
    difference = q1 - q2
    g_lh = engine._gradient(difference, q2, q2_e1, q2_e2, rows_lh, lh)
    g_rh = engine._gradient(difference, q2, q2_e1, q2_e2, rows_rh, rh)
    t3 = time.time()
    lh_warp.compose(engine._step(g_lh))
    rh_warp.compose(engine._step(g_rh))
    t4 = time.time()
    moving.warp(lh_warp, rh_warp)
    t5 = time.time()
    print(
        f"iteration {iteration}: F alone {t1 - t0:.2f}s; F with derivatives {t2 - t1:.2f}s; "
        f"gradient {t3 - t2:.2f}s; compose {t4 - t3:.2f}s; endpoints {t5 - t4:.2f}s; "
        f"cost {engine.cost(difference):.6f}",
        flush=True,
    )
