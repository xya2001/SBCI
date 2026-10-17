"""The reference's arithmetic in ConSEAL, for comparing with its MATLAB run or measuring beside it.

The package corrects these errors, items of the ConSEAL module docstring. Until 11 October 2026
``strict_upstream=True`` reproduced them; PORTING.md item 22 removed it once the defaults were
shown to undo known warps better. Each function here is that switch's code, moved out of the
package unchanged, for ``conseal_compare.py`` and the probes that measured the reference
beside the port:

- :func:`kernel_derivative`: the kernel differentiated with respect to its row vertex,
  normalization included (item 2);
- :func:`evaluate` and :func:`q_transform`: ``F``'s derivatives as ``dK A K^T`` symmetrized
  and projected on each row's frame (item 1), and the square root's chain rule over a floor
  (item 13);
- :func:`template`: the Karcher median of unnormalized densities, with no coincidence guard
  (item 15);
- :func:`cotangent_laplacian`: each edge weighted with an adjacent angle (item 12);
- :func:`smooth`: the velocity field smoothed as two scalar components (item 16).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from sbci.alignment import sparse_times_dense
from sbci.conseal import KernelDerivative
from sbci.smoothing import _legendre_with_derivative


def kernel_derivative(builder, sigma):
    """The reference's derivative of ``builder``'s kernel, with respect to the row vertex."""
    w = builder.weights(sigma)
    cut = builder.cutoff(sigma)
    blocks = [[], [], []]
    for grid, cosines in zip((builder.lh_grid, builder.rh_grid), builder._cosines, strict=True):
        rows, cols = np.nonzero(cosines >= cut)
        x = cosines[rows, cols]
        k = np.zeros_like(x)
        dk = np.zeros_like(x)
        for order, p, dp in _legendre_with_derivative(x, builder.degree):
            k += w[order] * p
            dk += w[order] * dp
        n = cosines.shape[0]
        raw = sparse.csr_matrix((k, (rows, cols)), shape=(n, n))
        s = np.asarray(raw.sum(axis=1)).ravel()
        k_norm = sparse.diags(1.0 / s) @ raw
        d_raw = sparse.csr_matrix((dk, (rows, cols)), shape=(n, n))
        for axis in range(3):
            # d/d(row vertex i) of K(i, a) = P'(x_i . x_a) x_a; then the quotient rule for the
            # row normalization: (D - K~ sum_b D(i, b)) / s_i.
            d = d_raw @ sparse.diags(grid.vertices[:, axis])
            t = np.asarray(d.sum(axis=1)).ravel()
            blocks[axis].append(sparse.diags(1.0 / s) @ d - sparse.diags(t / s) @ k_norm)
    return KernelDerivative(*(sparse.block_diag(b, format="csr") for b in blocks))


def evaluate(connectome, kernel, derivative):
    """``F = K^T A K`` and the reference's ``(D_e1, D_e2)``: ``dK A K^T`` symmetrized, framed."""
    kernel = sparse.csr_matrix(kernel)
    dense = connectome._sparse_adjacency().toarray()
    ak = np.ascontiguousarray(sparse_times_dense(kernel.T, dense).T)
    density = sparse_times_dense(kernel.T, ak)
    np.maximum(density, 0.0, out=density)
    akt = np.ascontiguousarray(sparse_times_dense(kernel, dense).T)  # A K^T
    parts = []
    for d in (derivative.x, derivative.y, derivative.z):
        part = sparse_times_dense(d, akt)
        parts.append(part + part.T)
    d_e1 = sum(p * e[:, None] for p, e in zip(parts, connectome.e1.T, strict=True))
    d_e2 = sum(p * e[:, None] for p, e in zip(parts, connectome.e2.T, strict=True))
    return density, d_e1, d_e2


def q_transform(connectome, kernel, derivative):
    """``Q = sqrt(F)`` and its derivatives as the reference forms them, over ``2 max(Q, 1e-15)``."""
    density, d_e1, d_e2 = evaluate(connectome, kernel, derivative)
    q = np.sqrt(density)
    scale = 1.0 / (2.0 * np.maximum(q, 1e-15))
    return q, d_e1 * scale, d_e2 * scale


def template(densities, iterations=100):
    """``get_template`` on the square-root densities as given, unnormalized, with no guard."""
    qs = [np.asarray(q, dtype=np.float64) for q in densities]
    q_bar = sum(qs) / len(qs)
    q_mu = qs[int(np.argmin([((q - q_bar) ** 2).sum() for q in qs]))].copy()
    for _ in range(int(iterations)):
        directions = np.zeros_like(q_mu)
        inverse_distance = 0.0
        for q in qs:
            inner = float((q * q_mu).sum())
            if 1 - abs(inner) < 1e-14:
                inner = float(np.sign(inner))
            distance = float(np.arccos(np.clip(inner, -1.0, 1.0)))
            if distance > 0.0:
                directions += (q - np.cos(distance) * q_mu) / np.sin(distance)
                inverse_distance += 1.0 / distance
        if inverse_distance == 0:
            break
        v_bar = directions / inverse_distance
        norm = float(np.sqrt((v_bar**2).sum()))
        if norm == 0:
            break
        q_mu = np.cos(0.2 * norm) * q_mu + np.sin(0.2 * norm) * (v_bar / norm)
        q_mu /= np.sqrt((q_mu**2).sum())
        if norm < 0.005:
            break
    return q_mu


def cotangent_laplacian(vertices, faces):
    """The reference's ``cot_matrix``: each edge weighted with an adjacent angle."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    i1, i2, i3 = faces[:, 0], faces[:, 1], faces[:, 2]
    v1 = vertices[i2] - vertices[i3]
    v2 = vertices[i3] - vertices[i1]
    v3 = vertices[i1] - vertices[i2]
    double_area = np.linalg.norm(np.cross(v1, -v3), axis=1)
    cot_at_2 = (v1 * -v3).sum(1) / double_area
    cot_at_3 = (v2 * -v1).sum(1) / double_area
    cot_at_1 = (v3 * -v2).sum(1) / double_area
    rows = np.concatenate([i2, i3, i3, i1, i1, i2])
    cols = np.concatenate([i3, i2, i1, i3, i2, i1])
    vals = 0.5 * np.concatenate([cot_at_2, cot_at_2, cot_at_3, cot_at_3, cot_at_1, cot_at_1])
    n = vertices.shape[0]
    weights = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(weights.sum(axis=1)).ravel()
    return (sparse.diags(degree) - weights).tocsr()


def smooth(laplacian, velocity, viscosity):
    """The reference's smoothing: ``viscosity`` of the Laplacian of each frame component."""
    return velocity - viscosity * (laplacian @ velocity)
