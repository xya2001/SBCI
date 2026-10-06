"""The fourth list's measurements, on the corrected code (6 October 2026; PORTING.md item 11).

    module load python/3.12.4
    SBCI_HCP_DIR=/work/users/x/y/xya/hcp-ya/data python tests/reference/fourth_list_probe.py

1. ConSEAL's smoothing against the reference's (``strict_upstream=True``),
   within 10 degrees of the coordinate equator, for rotations about three
   axes, after one and 100 smoothings, with the reference's cotangent layout
   (item 12 of the ConSEAL docstring) separated from the frames (item 16).
2. ``StationaryWarp.invert`` with a rigid part: how far a double inversion
   moves the velocity field, by the interpolated inverse this replaced
   (reimplemented here) and by the exact one, on ico4 and on ico3.
3. The Jacobian calibration on the 12-vertex icosahedron.
4. Global coupling on sub-100307 in Schaefer-1000 with a one-vertex region
   emptied of SC: the largest change it makes to the other regions.
5. The PALS-B12 atlases: labelled vertices in each hemisphere.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

import sbci
from sbci.alignment import SphericalGrid, SphericalWarp
from sbci.atlas import cortex_mask
from sbci.conseal import StationaryWarp, _transport, default_grids, icosphere
from sbci.coupling import global_coupling

warnings.simplefilter("ignore")
DATA = Path(os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data"))


def components(warp, field):
    return np.stack([(field * warp.e1).sum(1), (field * warp.e2).sum(1)], axis=1)


def ambient(warp, velocity):
    return warp.e1 * velocity[:, :1] + warp.e2 * velocity[:, 1:]


def smoothing():
    print("1. One and 100 smoothings of a rotation, within 10 degrees of the coordinate equator,")
    print("   as a share of the field's largest value (largest difference; median)")
    lh, _ = default_grids(15)
    lh_reference, _ = default_grids(15, reference=True)
    ours = StationaryWarp(lh)
    theirs = StationaryWarp(lh_reference, strict_upstream=True)
    x = ours.base
    band = np.abs(np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))) < 10
    axes = (("the coordinate z axis", (0, 0, 1)), ("x", (1, 0, 0)), ("(1, 1, 1)", (1, 1, 1)))
    for name, axis in axes:
        omega = 0.1 * np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        field = np.cross(omega, x)
        scale = np.linalg.norm(field, axis=1).max()
        for passes in (1, 100):
            a = components(ours, field)  # the connection Laplacian, corrected weights
            b = components(theirs, field)  # the reference: frame components, its layout
            c = components(ours, field)  # frame components, corrected weights
            for _ in range(passes):
                a = ours._smooth(a)
                b = theirs._smooth(b, strict_upstream=True)
                c = ours._smooth(c, strict_upstream=True)
            a, b, c = ambient(ours, a), ambient(theirs, b), ambient(ours, c)

            def share(p, q, scale=scale):
                d = np.linalg.norm(p - q, axis=1)[band] / scale
                return f"{d.max():.1e} ({np.median(d):.1e})"

            print(
                f"   about {name:22s} x{passes:<3d}: against the reference {share(a, b)}; "
                f"the frames alone {share(a, c)}; the layout alone {share(b, c)}"
            )


def old_invert(warp):
    """The inverse this replaced: the field carried to the rotated frame by interpolation."""
    weights, indices, rotated = warp._rigid_query
    field = ambient(warp, warp.velocity)
    moved = _transport(
        field[indices].reshape(-1, 3),
        warp.base[indices].reshape(-1, 3),
        np.repeat(rotated, 3, axis=0),
    ).reshape(warp.n_vertices, 3, 3)
    pulled = -((weights[:, :, None] * moved).sum(axis=1) @ warp._rigid)
    warp._set_rigid(warp._rigid.T)
    warp.velocity = components(warp, pulled)
    warp.vertices = warp._images()
    return warp


def inversion():
    from scipy.spatial.transform import Rotation

    print("2. A double inversion with a rigid part: the velocity field's change, as a share of it")
    vertices, faces = icosphere(3)
    grids = {
        "ico4 (bundled)": default_grids(15)[0],
        "ico3": SphericalGrid(vertices, faces, order=6),
    }
    for name, grid in grids.items():
        changes = []
        for method in ("interpolated", "exact"):
            warp = StationaryWarp(grid, viscosity=0.0).rotate(
                Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
            )
            x = warp.base
            field = np.cross(np.array([0.01, 0.02, -0.015]), x)
            field += 0.01 * np.stack([x[:, 1] * x[:, 2], -x[:, 0] * x[:, 2], 0 * x[:, 0]], axis=1)
            field -= (field * x).sum(1, keepdims=True) * x
            warp.velocity = components(warp, field)
            warp.vertices = warp._images()
            start = warp.velocity.copy()
            if method == "interpolated":
                old_invert(old_invert(warp))
            else:
                warp.invert().invert()
            change = np.linalg.norm(warp.velocity - start) / np.linalg.norm(start)
            changes.append(f"{method} {change:.2e}")
        print(f"   {name:15s}: {'; '.join(changes)}")


def calibration():
    vertices, faces = icosphere(0)
    warp = SphericalWarp(SphericalGrid(vertices, faces, order=1))
    raw, calibrated = warp._raw_jacobian(), warp._compute_jacobian()
    print(
        "3. The 12-vertex icosahedron: the scheme reads "
        f"{raw.min():.3f} to {raw.max():.3f} for the identity; calibrated, "
        f"{calibrated.min():.15f} to {calibrated.max():.15f}"
    )


def coupling():
    sc, fc = sbci.load(DATA / "sub-100307_sc.h5"), sbci.load(DATA / "sub-100307_fc.h5")
    atlas = sbci.load_atlas("Schaefer1000")
    s, f = sc.to_atlas(atlas, how="mean"), fc.to_atlas(atlas)
    counts = np.bincount(np.asarray(atlas.labels)[cortex_mask()], minlength=atlas.n_regions + 1)[1:]
    k = int(np.flatnonzero(counts == 1)[0])
    s[k, :] = s[:, k] = 0.0
    s[k, k] = np.nan
    keep = np.arange(atlas.n_regions) != k
    block = np.ix_(keep, keep)
    change = np.nanmax(np.abs(global_coupling(s, f)[keep] - global_coupling(s[block], f[block])))
    print(
        f"4. sub-100307, Schaefer-1000, {atlas.names[k]} (one vertex) emptied of SC: global "
        f"coupling of the other regions changes by at most {change:.1e}"
    )


def pals():
    half = sbci.spec.N_VERTICES_PER_HEMI
    parts = []
    for name in ("PALS_B12_Brodmann", "PALS_B12_OrbitoFrontal", "PALS_B12_Visuotopic"):
        labels = np.asarray(sbci.load_atlas(name).labels)
        left, right = int((labels[:half] != 0).sum()), int((labels[half:] != 0).sum())
        parts.append(f"{name[9:]} {left} and {right}")
    print("5. PALS-B12, labelled vertices left and right: " + "; ".join(parts))


if __name__ == "__main__":
    smoothing()
    inversion()
    calibration()
    coupling()
    pals()
    print("FOURTH LIST PROBE DONE")
