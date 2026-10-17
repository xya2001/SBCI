"""Synthetic warps for the fork and the package: which recovers a known warp best, unfolded.

    module load python/3.12.4
    python tests/reference/fork_synthetic_cases.py      # writes CASES/<case>.mat and .npz

PORTING.md item 19 compared the author's research fork of ConSEAL with the package on three
known warps, five split halves and one real pair, where the fork's warp folded. This adds
24 known warps of sub-100307's 803,741 streamlines, each registered back onto the
undeformed subject, so that every method can be judged against the truth at sizes where
folding may or may not set in:

- ``svf<a>_s<seed>``: a random smooth warp of harmonic order 4, the flow of a stationary
  velocity field whose largest displacement is ``a`` thousandths of a radian (35, 70, 140,
  210 and 280), as the recovery figure's (``scripts/make_figures.py``'s ``known_warp``);
- ``comp<a>_s<seed>``: four random order-4 maps, each applied after the last at the warped
  vertices, the fork's own way of building a warp, ``a`` thousandths of a radian in all (50,
  100 and 200);

three seeds each. ``truth_in`` and ``truth_out`` are where each moving endpoint began. The
names go to ``cases_synthetic.txt``; ``fork_case.m`` and ``fork_package_case.py`` run them and
``fork_synthetic_evaluate.py`` scores them.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

sys.path.insert(0, "/nas/longleaf/home/xya/sbci/scripts")
from make_figures import known_warp

import sbci
from sbci.alignment import MeshQuery, normalize_rows, sphere_exp_map
from sbci.conseal import EndpointConnectome, _transport, default_grids

ROOT = Path("/work/users/x/y/xya/conseal-fork-compare")
OUT = ROOT / "cases"
OUT.mkdir(parents=True, exist_ok=True)
lh, rh = default_grids()


def save(name, fixed, moving, truth):
    """One case, as the fork reads it (.mat) and as the package does (.npz)."""
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
        truth_in=truth[0],
        truth_out=truth[1],
    )
    sio.savemat(OUT / f"{name}.mat", arrays)
    np.savez(OUT / f"{name}.npz", **arrays)
    moved = np.r_[degrees(truth[0], m_in), degrees(truth[1], m_out)]
    print(f"{name}: endpoints moved {moved.mean():.3f} deg on average, {moved.max():.2f} at most")


def degrees(p, q):
    """Angle between unit vectors, row by row."""
    return np.degrees(np.arccos(np.clip((p * q).sum(1), -1.0, 1.0)))


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
        vertices = normalize_rows(sphere_exp_map(vertices, (weights[:, :, None] * moved).sum(1)))
    return SimpleNamespace(vertices=vertices)


subject = sbci.load("/work/users/x/y/xya/hcp-ya/data/sub-100307_sc.h5")
carrier = EndpointConnectome.from_endpoints(subject.endpoints, lh, rh)
original = carrier.positions()
flags = (carrier.hemisphere_in, carrier.hemisphere_out)
names = []
for family, amplitudes, seeds, make in (
    ("svf", (0.035, 0.07, 0.14, 0.21, 0.28), (101, 102, 103), known_warp),
    ("comp", (0.05, 0.10, 0.20), (201, 202, 203), composed),
):
    for amplitude in amplitudes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            lh_true, rh_true = (make(grid, rng, amplitude) for grid in default_grids(4))
            deformed = carrier.copy()
            deformed.warp(lh_true, rh_true)
            name = f"{family}{round(amplitude * 1000):03d}_s{seed}"
            save(name, (original, flags), (deformed.positions(), flags), truth=original)
            names.append(name)
(ROOT / "cases_synthetic.txt").write_text("\n".join(names) + "\n")
print(f"{len(names)} CASES WRITTEN")
