"""Build the inputs for the FPCA reference run, identically for both sides.

The reference represents functions in a spherical spline basis whose Gram
matrix J and roughness matrix R come from splinepak and Lebedev quadrature.
This package already has both on its own grid: the Voronoi vertex areas are
the L2 inner product, and the cotangent stiffness matrix is the Dirichlet
energy. So the basis is the identity on the grid, and the two dependencies the
reference needs for its basis are not needed at all.
"""

import sys

import numpy as np
from scipy import sparse
from scipy.io import savemat


def main():
    sys.path.insert(0, "/nas/longleaf/home/xya/sbci/tests")
    from sbci.alignment import SphericalGrid, rotate_off_poles
    from test_alignment import icosphere

    OUT = "/work/users/x/y/xya/fpca-ref"
    rng = np.random.default_rng(20260921)

    vertices, faces = icosphere(1)
    grid = SphericalGrid(rotate_off_poles(vertices), faces, order=2)
    nv = grid.n_vertices
    print(f"grid: {nv} vertices per hemisphere, {2 * nv} combined")

    def stiffness(points, triangles):
        """Cotangent stiffness: the Dirichlet energy quadratic form."""
        corners = points[triangles]
        n = points.shape[0]
        edges = [
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
            corners[:, 1] - corners[:, 0],
        ]
        double_area = np.linalg.norm(np.cross(edges[0], -edges[1]), axis=1)
        rows, cols, values = [], [], []
        for i in range(3):
            cot = -(edges[(i + 1) % 3] * edges[(i + 2) % 3]).sum(axis=1) / double_area
            a, b = triangles[:, (i + 1) % 3], triangles[:, (i + 2) % 3]
            rows += [a, b, a, b]
            cols += [b, a, a, b]
            values += [-0.5 * cot, -0.5 * cot, 0.5 * cot, 0.5 * cot]
        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).toarray()
        return 0.5 * (matrix + matrix.T)

    J_block = np.diag(grid.areas)
    R_block = stiffness(grid.vertices, grid.faces)
    print(f"J: areas sum to {grid.areas.sum():.6f}")
    print(
        f"R: symmetric {np.allclose(R_block, R_block.T)}, "
        f"smallest eigenvalue {np.linalg.eigvalsh(R_block).min():.3e}"
    )

    # Synthetic subjects with a known separable structure plus noise, so the
    # decomposition has something real to find.
    N, K_true = 12, 3
    truth = rng.standard_normal((2 * nv, K_true))
    truth /= np.linalg.norm(truth, axis=0, keepdims=True)
    weights = rng.standard_normal((N, K_true)) * np.array([3.0, 2.0, 1.0])

    Y = np.zeros((2 * nv, 2 * nv, N))
    for i in range(N):
        matrix = sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(K_true))
        noise = 0.1 * rng.standard_normal((2 * nv, 2 * nv))
        matrix = matrix + (noise + noise.T) / 2
        Y[:, :, i] = matrix
    Y -= Y.mean(axis=2, keepdims=True)
    print(f"data: {Y.shape}, {N} subjects, true rank {K_true}")

    savemat(
        f"{OUT}/fpca_inputs.mat",
        {
            "Y": Y,
            "J_block": J_block,
            "R_block": R_block,
            "nv": nv,
            "N": N,
            "truth": truth,
            "weights": weights,
        },
        do_compression=True,
    )
    np.savez_compressed(
        f"{OUT}/fpca_inputs.npz",
        Y=Y,
        J_block=J_block,
        R_block=R_block,
        nv=nv,
        truth=truth,
        weights=weights,
    )
    print(f"wrote {OUT}/fpca_inputs.mat and .npz")


if __name__ == "__main__":
    main()
