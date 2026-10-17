"""The first checks of the ENCORE port, as numbers, measured again with the package.

    python tests/reference/encore_first_checks_probe.py

PORTING.md item 4 quotes these. They were first measured on 20 September 2026 by scripts that
were never committed, some of them against a standalone draft of the port; this measures them
with ``sbci.alignment`` as it is (item 24). The reference's dumps are in
``$SBCI_ENCORE_REFERENCE``, as for ``tests/test_matlab_reference.py``: ``encore_reference.m``
writes ``encore_reference.mat``, and the hand-run ``encore_delta_sweep.m``,
``encore_face_agreement.m`` and ``encore_jacobian_inner.m`` beside it write the other three (the
reference's derivative at five step sizes, the faces its tree picks for offset queries, and its
Jacobian's intermediates). A part whose files are missing is skipped; the closed forms need
none, and one line of the last part needs the example subject's SC file (``$SBCI_EXAMPLE_SC``,
as for Tier 4).
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sbci.alignment import (  # noqa: E402
    Concon,
    MeshQuery,
    SphericalGrid,
    SphericalWarp,
    gradient_operators,
    sphere_exp_map,
    sphere_log_map,
)
from test_alignment import icosphere  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
FOLDER = Path(os.environ.get("SBCI_ENCORE_REFERENCE", "/work/users/x/y/xya/encore-ref"))


def read(path, *names):
    """MATLAB -v7.3 is HDF5 written column-major, so it reads in transposed."""
    import h5py

    with h5py.File(path, "r") as handle:
        return [np.asarray(handle[name]).T.astype(np.float64) for name in names]


def relative(mine, theirs):
    return float(np.abs(mine - theirs).max() / np.abs(theirs).max())


def reference_grids():
    vertices, faces, rh_vertices, order = read(
        FOLDER / "encore_reference.mat", "grid_lh_V", "grid_lh_T", "grid_rh_V", "L"
    )
    faces = faces.astype(int) - 1
    order = int(order.ravel()[0])
    return SphericalGrid(vertices, faces, order), SphericalGrid(rh_vertices, faces, order)


def frames_and_maps():
    """Item 4's first table: the tangent frames, the maps and the identity's Jacobian."""
    lh, _ = reference_grids()
    e1, e2, gamma, exp_test, log_test, identity, delta = read(
        FOLDER / "encore_reference.mat",
        "grid_lh_e1",
        "grid_lh_e2",
        "gamma_test",
        "exp_test",
        "log_test",
        "warp_J_identity",
        "DELTA",
    )
    for name, mine, theirs in (
        ("tangent frame e1", lh.e1, e1),
        ("tangent frame e2", lh.e2, e2),
        ("exponential map", sphere_exp_map(lh.vertices, gamma), exp_test),
        ("logarithm map", sphere_log_map(lh.vertices, exp_test), log_test),
    ):
        print(f"  {name:22s} {relative(mine, theirs) / EPS:5.1f} float64-eps")
    warp = SphericalWarp(lh, float(delta.ravel()[0]))
    gap = float(np.abs(warp.jacobian - identity.ravel()).max())
    print(
        f"  identity's Jacobian: the reference's is {np.unique(identity).tolist()} (set, not "
        f"computed); the port's differs by {gap:.3g}, its scheme giving "
        f"{warp._raw_jacobian().mean():.4f} there before the calibration divides it out"
    )


def derivative_against_the_reference():
    """Item 4's floor: the port's central difference against the reference's, step by step."""
    lh, rh = reference_grids()
    (f_test,) = read(FOLDER / "encore_reference.mat", "F_test")
    deltas, sweep = read(FOLDER / "delta_sweep.mat", "deltas", "dQe1_sweep")
    deltas = deltas.ravel()
    for k, delta in enumerate(deltas):
        mine = Concon(lh, rh, delta=float(delta)).derivative(f_test)[0]
        gap = np.abs(mine - sweep[:, :, k]).max()
        print(f"  step {delta:g}: the port {gap:.4f} from the reference")
    for k in range(deltas.size - 1):
        gap = np.abs(sweep[:, :, k] - sweep[:, :, k + 1]).max()
        print(f"  the reference at {deltas[k]:g} against itself at {deltas[k + 1]:g}: {gap:.4f}")
    print(f"  (the derivative's largest value {np.abs(sweep).max():.3f})")


def face_agreement():
    """Item 4's face agreement: which closest-face rule reproduces libigl's choice."""
    lh, _ = reference_grids()
    (query, reference_faces) = read(FOLDER / "encore_reference.mat", "query_pts", "bary_T_false")
    deltas, picked = read(FOLDER / "face_agree.mat", "deltas", "pTx")
    a, b, c = (lh.vertices[lh.faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    touching = [np.flatnonzero((lh.faces == v).any(axis=1)) for v in range(lh.n_vertices)]

    tree = MeshQuery(lh.vertices, lh.faces)

    def closest(points):  # the port's rule, MeshQuery's: the face whose closest point is nearest
        return tree.query_faces(points)[2]

    def plane(points):  # distance to each triangle's plane
        return np.argmin(np.abs(((points[:, None, :] - a[None]) * normal[None]).sum(-1)), axis=1)

    def plane_at_vertex(points):  # the same, among the faces at the nearest vertex
        nearest = np.argmin(((points[:, None, :] - lh.vertices[None]) ** 2).sum(-1), axis=1)
        chosen = np.empty(points.shape[0], dtype=int)
        for i, point in enumerate(points):
            faces = touching[nearest[i]]
            chosen[i] = faces[np.argmin(np.abs(((point - a[faces]) * normal[faces]).sum(-1)))]
        return chosen

    def agreement(chosen, corners):
        mine = np.sort(lh.faces[chosen], axis=1)
        return float((mine == np.sort(corners.astype(int) - 1, axis=1)).all(axis=1).mean())

    rules = (
        ("closest point", closest),
        ("plane", plane),
        ("plane, nearest vertex", plane_at_vertex),
    )
    print(
        "  ordinary points (encore_reference.m's queries): "
        + ", ".join(f"{name} {agreement(rule(query), reference_faces):.1%}" for name, rule in rules)
    )
    for k, delta in enumerate(deltas.ravel()):
        points = sphere_exp_map(lh.vertices, lh.e1 * float(delta))
        print(
            f"  offset {delta:g} along e1: "
            + ", ".join(
                f"{name} {agreement(rule(points), picked[:, :, k]):.1%}" for name, rule in rules
            )
        )


def jacobian_on_the_references_points():
    """Item 4's 2.2e-06: the port's Jacobian arithmetic fed the reference's own points."""
    lh, _ = reference_grids()
    names = ("Vw", "p_Vx", "p_Tx", "m_Vx", "m_Tx", "p_Vy", "p_Ty", "m_Vy", "m_Ty", "Jfinal")
    values = dict(zip(names, read(FOLDER / "jac_inner.mat", *names), strict=True))
    (delta,) = read(FOLDER / "encore_reference.mat", "DELTA")
    warp = SphericalWarp(lh, float(delta.ravel()[0]))
    warp.vertices = values["Vw"]
    warp._offsets = {
        name: (values[f"{sign}_V{axis}"], values[f"{sign}_T{axis}"].astype(int) - 1)
        for name, sign, axis in (
            ("px", "p", "x"),
            ("mx", "m", "x"),
            ("py", "p", "y"),
            ("my", "m", "y"),
        )
    }
    gap = relative(warp._raw_jacobian(), values["Jfinal"].ravel())
    print(f"  {gap:.2g} relative, against eps / step = {EPS / float(delta.ravel()[0]):.2g}")


def closed_forms():
    """Item 4's known answers: a linear function on a flat mesh, coordinates on spheres."""
    n = 21
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    flat = np.stack([u.ravel(), v.ravel(), np.zeros(u.size)], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, d], [a, d, c]]
    first, second = gradient_operators(
        flat, np.array(faces), np.tile([1.0, 0, 0], (n * n, 1)), np.tile([0, 1.0, 0], (n * n, 1))
    )
    for name, field, slope in (
        ("x", flat[:, 0], (1, 0)),
        ("3x + 2y", 3 * flat[:, 0] + 2 * flat[:, 1], (3, 2)),
    ):
        error = max(np.abs(first @ field - slope[0]).max(), np.abs(second @ field - slope[1]).max())
        print(f"  f = {name} on a flat mesh: largest error {error:.2g}")
    # z on an icosphere not turned off the axis, along e1, where d z / d e1 = -sin(theta)
    rows = {"analytic": [], "difference": []}
    for level in (2, 3, 4):
        vertices, faces = icosphere(level)
        grid = SphericalGrid(vertices, faces, order=2)
        z, truth = grid.vertices[:, 2], -np.sin(grid.theta)
        rows["analytic"].append(np.abs(grid.gradients[0] @ z - truth).max())
        block = np.tile(np.concatenate([z, z])[:, None], (1, 2 * grid.n_vertices))
        found = Concon(grid, grid, delta=1e-5).derivative(block)[0][: grid.n_vertices, 0]
        rows["difference"].append(np.abs(found - truth).max())
    for name, errors in rows.items():
        print(
            f"  z along e1 on ico2, ico3, ico4, {name}: " + " -> ".join(f"{e:.2g}" for e in errors)
        )


def two_derivatives():
    """Item 4's last table: the analytic derivative against the difference, by smoothness."""
    vertices, faces = icosphere(2)
    grid = SphericalGrid(vertices, faces, order=3)
    n = grid.n_vertices
    theta, phi = np.tile(grid.theta, 2), np.tile(grid.phi, 2)
    smooth = np.outer(1 + np.cos(theta), 1 + np.cos(theta)) + 0.3 * np.outer(
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.cos(phi)
    )
    noise = np.random.default_rng(0).random((2 * n, 2 * n))
    noise = (noise + noise.T) / 2
    weights = np.exp(5.0 * (grid.vertices @ grid.vertices.T))
    weights = np.block([[weights, weights], [weights, weights]])
    weights /= weights.sum(axis=1, keepdims=True)
    fields = [
        ("a smooth field", smooth, grid, grid),
        ("kernel-smoothed noise", weights @ noise @ weights.T, grid, grid),
    ]
    example = Path(os.environ.get("SBCI_EXAMPLE_SC", "/work/users/x/y/xya/sbci-derivatives"))
    example = example / "sub-example_sc.h5"
    if example.is_file():
        from sbci import ContinuousConnectome
        from sbci.alignment import _hemisphere_grids

        lh, rh = _hemisphere_grids(2)
        fields.append(
            (
                "the example subject's SC, ico4",
                ContinuousConnectome.load(example).dense(np.float64),
                lh,
                rh,
            )
        )
    fields.append(("white noise", noise, grid, grid))
    for name, field, lh, rh in fields:
        analytic = Concon(lh, rh, derivative="analytic").derivative(field)[0]
        difference = Concon(lh, rh, delta=1e-5).derivative(field)[0]
        scale = max(np.abs(analytic).max(), np.abs(difference).max())
        r = np.corrcoef(analytic.ravel(), difference.ravel())[0, 1]
        print(
            f"  {name:32s} largest difference {np.abs(analytic - difference).max() / scale:.3f} "
            f"of the scale, r = {r:.5f}"
        )


PARTS = (
    ("frames, maps and the identity's Jacobian", frames_and_maps, ["encore_reference.mat"]),
    (
        "the derivative against the reference's",
        derivative_against_the_reference,
        ["encore_reference.mat", "delta_sweep.mat"],
    ),
    ("face agreement with libigl", face_agreement, ["encore_reference.mat", "face_agree.mat"]),
    (
        "the Jacobian on the reference's own points",
        jacobian_on_the_references_points,
        ["encore_reference.mat", "jac_inner.mat"],
    ),
    ("derivatives against closed forms", closed_forms, []),
    ("the analytic derivative against the difference", two_derivatives, []),
)

if __name__ == "__main__":
    for title, part, needs in PARTS:
        missing = [name for name in needs if not (FOLDER / name).is_file()]
        print(f"{title}:")
        if missing:
            print(f"  skipped: {', '.join(missing)} not in {FOLDER}")
            continue
        part()
