"""Inputs for the comparison of the paper's ConSEAL with the package's (PORTING.md item 19).

    module load python/3.12.4
    python tests/reference/fork_cases.py

Each case is a fixed and a moving set of streamline endpoints (unit-sphere coordinates in the
bundled sphere's frame, hemisphere flags 0/1), saved as CASES/<case>.mat for the fork and
CASES/<case>.npz for the package:

- known_007, known_014: sub-100307 deformed by a known stationary-velocity warp of harmonic
  order 4 whose largest displacement is 0.07 (the recovery figure's: the same seed) or 0.14
  radians, registered back onto itself undeformed; truth_* is where each moving endpoint began.
- composed_010: the same subject deformed the fork's way -- four random order-4 maps, each
  applied after the last at the warped vertices (barycentric, transported, exponential map),
  0.025 radians at most apiece.
- split_<subject>: one half of a subject's streamlines registered onto the other half
  (reliability_data.py's seeded halves) for the first five split subjects; nothing should move.
- pair: the fork's own pair, 100206 fixed and 106824 moving.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/scripts")
from make_figures import known_warp  # noqa: E402
from reliability_data import halves, sample  # noqa: E402

import sbci  # noqa: E402
from sbci.alignment import MeshQuery, normalize_rows, sphere_exp_map  # noqa: E402
from sbci.conseal import EndpointConnectome, _transport, default_grids  # noqa: E402

OUT = Path("/work/users/x/y/xya/conseal-fork-compare/cases")
OUT.mkdir(parents=True, exist_ok=True)
lh, rh = default_grids()


def save(name, fixed, moving, truth=None):
    (f_in, f_out), (fh_in, fh_out) = fixed
    (m_in, m_out), (mh_in, mh_out) = moving
    arrays = dict(
        fixed_in=f_in,
        fixed_out=f_out,
        fixed_hin=fh_in[None].astype(float),
        fixed_hout=fh_out[None].astype(float),
        moving_in=m_in,
        moving_out=m_out,
        moving_hin=mh_in[None].astype(float),
        moving_hout=mh_out[None].astype(float),
    )
    if truth is not None:
        arrays.update(truth_in=truth[0], truth_out=truth[1])
    sio.savemat(OUT / f"{name}.mat", arrays)
    np.savez(OUT / f"{name}.npz", **arrays)
    print(f"{name}: fixed {f_in.shape[0]:,} streamlines, moving {m_in.shape[0]:,}", flush=True)


def endpoints(connectome):
    return connectome.positions(), (connectome.hemisphere_in, connectome.hemisphere_out)


def composed(grid, rng, amplitude, steps=4):
    """Several random smooth maps, each applied after the last at the warped vertices."""
    vertices = grid.vertices.copy()
    query = MeshQuery(grid.vertices, grid.faces)
    n = vertices.shape[0]
    for _ in range(steps):
        coefficients = rng.standard_normal(grid.basis.shape[1])
        field = (coefficients[None, :, None] * grid.basis).sum(axis=1)
        field *= (amplitude / steps) / np.linalg.norm(field, axis=1).max()
        tangent = grid.e1 * field[:, :1] + grid.e2 * field[:, 1:2]
        weights, indices, _ = query.query_faces(vertices)
        moved = _transport(
            tangent[indices].reshape(-1, 3),
            grid.vertices[indices].reshape(-1, 3),
            np.repeat(vertices, 3, axis=0),
        ).reshape(n, 3, 3)
        vertices = normalize_rows(
            sphere_exp_map(vertices, (weights[:, :, None] * moved).sum(axis=1))
        )
    return SimpleNamespace(vertices=vertices)


subject = sbci.load("/work/users/x/y/xya/hcp-ya/data/sub-100307_sc.h5")
carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
original, flags = endpoints(carrier)
for name, amplitude in (("known_007", 0.07), ("known_014", 0.14)):
    rng = np.random.default_rng(7)  # the recovery figure's seed: known_007 is its warp
    lh_true, rh_true = (known_warp(grid, rng, amplitude) for grid in default_grids(4))
    deformed = carrier.copy()
    deformed.warp(lh_true, rh_true)
    save(name, (original, flags), endpoints(deformed), truth=original)
rng = np.random.default_rng(11)
lh_true, rh_true = (composed(grid, rng, 0.10) for grid in default_grids(4))
deformed = carrier.copy()
deformed.warp(lh_true, rh_true)
save("composed_010", (original, flags), endpoints(deformed), truth=original)

for name in sample()[:5]:
    _, first, second = halves(name)
    save(
        f"split_{name}",
        endpoints(EndpointConnectome.from_endpoints(first, lh, rh)),
        endpoints(EndpointConnectome.from_endpoints(second, lh, rh)),
    )

D = "/users/x/y/xya/encore_dice_scores_2"
pair = []
for number in ("100206", "106824"):
    x = sio.loadmat(f"{D}/{number}/fsaverage_sphere_intersections.npz.mat")
    unit = tuple(x[k] / np.linalg.norm(x[k], axis=1, keepdims=True) for k in ("vtx_in", "vtx_out"))
    pair.append(
        (unit, tuple(np.asarray(x[k]).ravel().astype(np.int8) for k in ("surf_in", "surf_out")))
    )
save("pair", pair[0], pair[1])
print("CASES WRITTEN")
