"""ENCORE: diffeomorphic alignment of continuous connectomes.

Port of ``ConCon_Alignment`` (Cole et al.). A connectome is treated as a
density on the product of two spheres; alignment finds, for each subject, a
diffeomorphism of each hemisphere's sphere that carries that subject's density
onto a common template. Registration is gradient descent on the L2 distance
between square-root densities, which is the Fisher-Rao metric, and the template
is their Karcher median.

Layers, each checked against a MATLAB run of the reference:

``SphericalGrid``
    A spherical mesh with libigl's mixed Voronoi vertex areas and the tangent
    vector basis built from spherical harmonics up to a given order.
``MeshQuery``
    Closest-point queries, replacing the reference's libigl AABB tree. The tree
    is an index, not an algorithm, so the port computes the same closest point
    directly.
``SphericalWarp``
    A diffeomorphism, stored as where it sends each vertex, with the
    determinant of its differential by central differences, calibrated so
    that the identity's is exactly 1.
``Concon``
    The density on the product mesh: push it through a pair of warps, and
    differentiate it along the two frame fields.
``Encore``
    Template estimation and registration.

Conditioning
------------
The reference differentiates with a step of ``1e-10``. A central difference
with that step on quantities known to float64 precision has a condition number
near ``1e6``: six of sixteen digits are lost before anything else happens. The
reference's own derivative moves by 0.037 -- about 2% of its range -- between
adjacent step sizes, and its Jacobian by 2.3e-04, so neither is determined to
better than about three digits by the mathematics. This port therefore defaults
``delta`` to ``1e-5``, near the optimum for a central difference in double
precision; agreement with the reference is the same at every step size, so
nothing is given up by the change.

There is a floor on how closely any independent implementation can match. The
reference's derivative does not converge as the step shrinks -- it moves by
about 0.037 between every adjacent pair of step sizes, without settling --
because at a vertex the interpolant has a kink and the measured secant depends
on which face libigl's tree happens to return. This port's finite difference
sits 0.0367 from the reference at every step size, which is the same distance
the reference sits from itself. That is the floor, and it is reached.
See PORTING.md item 4.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

METHODS = ("encore",)

DEFAULT_ORDER = 6
"""Spherical harmonic order of the tangent basis: ``2(l+1)^2 - 2`` fields."""

DEFAULT_DELTA = 1e-5
"""Central-difference step. See the conditioning note in the module docstring."""


# --- sphere utilities ------------------------------------------------------


def cart_to_sphere(vertices):
    """``(theta, phi)`` from unit Cartesian coordinates, phi in ``[0, 2*pi)``."""
    vertices = np.asarray(vertices, dtype=np.float64)
    theta = np.arccos(np.clip(vertices[:, 2], -1.0, 1.0))
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    return theta, np.where(phi < 0, phi + 2 * np.pi, phi)


def normalize_rows(vertices):
    """Rows scaled to unit length; zero rows are left alone."""
    vertices = np.asarray(vertices, dtype=np.float64)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return np.divide(vertices, norms, out=np.zeros_like(vertices), where=norms > 0)


def sphere_exp_map(vertices, gamma):
    """Exponential map on the unit sphere, in Cartesian coordinates."""
    vertices = np.asarray(vertices, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    norm = np.sqrt((gamma**2).sum(axis=1))
    moved = norm > np.finfo(float).eps
    out = vertices.copy()
    scale = norm[moved][:, None]
    out[moved] = np.cos(scale) * vertices[moved] + np.sin(scale) * (gamma[moved] / scale)
    return out


def sphere_log_map(base, target):
    """Inverse of :func:`sphere_exp_map`."""
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    inner = np.clip((target * base).sum(axis=1), -1.0, 1.0)
    angle = np.arccos(inner)
    moved = angle > np.finfo(float).eps
    gamma = np.zeros_like(base)
    scale = (angle[moved] / np.sin(angle[moved]))[:, None]
    gamma[moved] = scale * (target[moved] - inner[moved][:, None] * base[moved])
    return gamma


def voronoi_areas(vertices, faces):
    """Libigl's ``MASSMATRIX_TYPE_VORONOI`` diagonal.

    Mixed Voronoi area (Meyer et al.): the circumcentric dual cell, except in
    an obtuse triangle, where the obtuse corner takes half the triangle and the
    other two a quarter each.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = vertices[faces]

    length = np.stack(
        [
            np.linalg.norm(corners[:, 1] - corners[:, 2], axis=1),
            np.linalg.norm(corners[:, 2] - corners[:, 0], axis=1),
            np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1),
        ],
        axis=1,
    )
    double_area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    l0, l1, l2 = length.T
    cosines = np.stack(
        [
            (l2**2 + l1**2 - l0**2) / (2 * l1 * l2),
            (l0**2 + l2**2 - l1**2) / (2 * l2 * l0),
            (l1**2 + l0**2 - l2**2) / (2 * l0 * l1),
        ],
        axis=1,
    )
    barycentric = cosines * length
    barycentric = barycentric / barycentric.sum(axis=1, keepdims=True)
    partial = barycentric * (double_area * 0.5)[:, None]
    quads = np.stack(
        [
            (partial[:, 1] + partial[:, 2]) * 0.5,
            (partial[:, 2] + partial[:, 0]) * 0.5,
            (partial[:, 0] + partial[:, 1]) * 0.5,
        ],
        axis=1,
    )
    for corner in range(3):
        obtuse = cosines[:, corner] < 0
        for other in range(3):
            quads[obtuse, other] = (0.25 if other == corner else 0.125) * double_area[obtuse]

    areas = np.zeros(vertices.shape[0])
    for corner in range(3):
        np.add.at(areas, faces[:, corner], quads[:, corner])
    return areas


# --- spherical harmonic tangent basis --------------------------------------


def _legendre_derivatives(degree, colatitude, reference: bool = False):
    """``(ddP, dP, P)`` for one degree, carrying the harmonic normalization.

    The derivative in theta comes from the recurrence on the **normalized**
    functions, ``2 dP~_l^m = sqrt((l-m)(l+m+1)) P~_l^{m+1} - sqrt((l+m)(l-m+1))
    P~_l^{m-1}``, whose ``m = -1`` member is ``P~_l^{-1} = -P~_l^{1}``. The
    reference writes that member as ``-P_l^1 / (l(l+1))``, the relation for
    the *unnormalized* functions, so its zonal (``m = 0``) derivative is scaled
    by ``(1 + 1/(l(l+1))) / 2``: 0.75 at degree 1, 0.502 at degree 15. Its
    second derivative also carries the ``m + 1`` term with the wrong sign.
    ``reference=True`` keeps that arithmetic bit for bit, for the MATLAB
    comparison.
    """
    from scipy.special import factorial, lpmv

    colatitude = np.asarray(colatitude, dtype=np.float64).ravel()
    order = np.arange(degree + 1)[:, None]
    normalization = np.sqrt(
        ((2 * degree + 1) / (4 * np.pi)) * (factorial(degree - order) / factorial(degree + order))
    )
    lower = np.sqrt((degree + order) * (degree - order + 1))
    upper = np.sqrt((degree - order) * (degree + order + 1))

    values = normalization * np.stack(
        [lpmv(m, degree, np.cos(colatitude)) for m in range(degree + 1)]
    )
    if degree == 0:
        zero = np.zeros_like(values)
        return zero, zero, values

    padding = np.zeros((1, colatitude.size))
    above = np.vstack([values[1:], padding])
    if reference:
        below = np.vstack([-values[1:2] / (degree * (degree + 1)), values[:-1]])
        first = -0.5 * ((below * lower) - (above * upper))
        d_below = np.vstack([first[1:2] / (degree * (degree + 1)), -first[:-1]])
        d_above = np.vstack([first[1:], padding])
        second = 0.5 * ((d_below * lower) - (d_above * upper))
        return second, first, values

    below = np.vstack([-values[1:2], values[:-1]])
    first = -0.5 * ((below * lower) - (above * upper))
    # the same recurrence on the first derivatives, whose m = -1 member is -dP~_l^1
    d_below = np.vstack([-first[1:2], first[:-1]])
    d_above = np.vstack([first[1:], padding])
    second = -0.5 * ((d_below * lower) - (d_above * upper))
    return second, first, values


def _harmonic_derivatives(degree, theta, phi, reference: bool = False):
    """Gradient and Laplacian of the real harmonics of one degree.

    Ordered ``Re(Y_0), Re(Y_1), Im(Y_1), ...`` as the reference orders them.
    ``reference`` selects the reference's Legendre recurrence
    (:func:`_legendre_derivatives`).
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    _, first, values = _legendre_derivatives(degree, theta, reference)

    order = np.arange(degree + 1)[:, None]
    sin_phi, cos_phi = np.sin(order * phi), np.cos(order * phi)
    sin_theta = np.maximum(np.sin(theta), 1e-5)

    width = 2 * (degree + 1) - 1
    gradient = np.zeros((2, width, theta.size))
    laplacian = np.zeros((width, theta.size))

    cosine_slots = np.concatenate([[0], np.arange(1, width, 2)])
    sine_slots = np.arange(2, width, 2)

    gradient[0, cosine_slots] = first * cos_phi
    gradient[0, sine_slots] = first[1:] * sin_phi[1:]
    gradient[1, cosine_slots] = values * -(order * sin_phi) / sin_theta
    gradient[1, sine_slots] = values[1:] * (order[1:] * cos_phi[1:]) / sin_theta

    scale = -degree * (degree + 1)
    laplacian[cosine_slots] = scale * values * cos_phi
    laplacian[sine_slots] = scale * values[1:] * sin_phi[1:]
    return gradient, laplacian


def tangent_basis(order, theta, phi, areas, reference: bool = False):
    """Gradient and rotated-gradient fields of the harmonics up to ``order``.

    Returns ``basis`` of shape ``(P, 2(order+1)^2 - 2, 2)`` in the ``(e1, e2)``
    frame, and the Laplacian of each field.

    With ``reference=True`` the fields are differentiated by the reference's
    Legendre recurrence (:func:`_legendre_derivatives`). That scales every
    zonal gradient field by a constant, which the area-weighted normalization
    below removes again, so the basis itself is the same to rounding; the
    Laplacian of those fields, normalized by the same constant, comes out
    ``2 / (1 + 1/(l(l+1)))`` times too large -- 4/3 at degree 1.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    areas = np.asarray(areas, dtype=np.float64).ravel()
    n_points = theta.size
    n_fields = (order + 1) ** 2 - 1

    fields = np.zeros((n_points, n_fields, 2))
    field_laplacian = np.zeros((n_points, n_fields))
    index = 0
    for degree in range(1, order + 1):
        width = 2 * (degree + 1) - 1
        gradient, laplacian = _harmonic_derivatives(degree, theta, phi, reference)
        fields[:, index : index + width, 0] = gradient[0].T
        fields[:, index : index + width, 1] = gradient[1].T
        field_laplacian[:, index : index + width] = laplacian.T
        index += width

    norms = np.sqrt(((fields[:, :, 0] ** 2 + fields[:, :, 1] ** 2) * areas[:, None]).sum(axis=0))
    positive = norms > 0
    fields[:, positive] /= norms[positive][None, :, None]
    field_laplacian[:, positive] /= norms[positive][None, :]

    basis = np.zeros((n_points, 2 * n_fields, 2))
    basis_laplacian = np.zeros((n_points, 2 * n_fields))
    basis[:, :n_fields] = fields
    basis[:, n_fields:, 0] = fields[:, :, 1]
    basis[:, n_fields:, 1] = -fields[:, :, 0]
    basis_laplacian[:, :n_fields] = field_laplacian
    return basis, basis_laplacian


# --- closest-point queries and product interpolation -----------------------


def barycentric_coordinates(point, a, b, c):
    """Libigl's ``barycentric_coordinates``, for points already in the plane."""
    v0, v1, v2 = b - a, c - a, point - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(denom == 0, 1.0, denom)
    second = (d11 * d20 - d01 * d21) / denom
    third = (d00 * d21 - d01 * d20) / denom
    return np.stack([1.0 - second - third, second, third], axis=-1)


def _closest_on_triangles(p, a, b, c):
    """Closest point on triangle ``(a, b, c)`` to ``p``; the arrays broadcast to ``(..., 3)``.

    Ericson's algorithm. The broadcasting serves both uses: each point against
    its own candidate triangles, and a chunk of points against every face.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    total = va + vb + vc
    scale = 1.0 / np.where(total == 0, 1.0, total)
    closest = a + (vb * scale)[..., None] * ab + (vc * scale)[..., None] * ac
    closest = np.array(np.broadcast_to(closest, np.broadcast(p, a).shape))

    def put(mask, value):
        np.copyto(closest, np.broadcast_to(value, closest.shape), where=mask[..., None])

    denom_ab = np.where(d1 - d3 == 0, 1.0, d1 - d3)
    denom_ac = np.where(d2 - d6 == 0, 1.0, d2 - d6)
    denom_bc = np.where((d4 - d3) + (d5 - d6) == 0, 1.0, (d4 - d3) + (d5 - d6))
    put(
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
        b + ((d4 - d3) / denom_bc)[..., None] * (c - b),
    )
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / denom_ac)[..., None] * ac)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / denom_ab)[..., None] * ab)
    put((d6 >= 0) & (d5 <= d6), c)
    put((d3 >= 0) & (d4 <= d3), b)
    put((d1 <= 0) & (d2 <= 0), a)
    return closest


class MeshQuery:
    """Closest-point queries against a triangle mesh.

    Replaces the reference's libigl AABB tree. The face is the one carrying the
    closest point; the weights then come from the **unclamped** projection of
    the query onto that face's plane, which is what the reference's libigl
    build returns -- a point just outside a triangle keeps a slightly negative
    weight rather than being snapped to the edge.

    The search is narrowed to the ``candidates`` faces whose centroids lie
    nearest each point, found with a k-d tree, so a query costs time in
    proportion to the number of points rather than points times faces. On an
    icosphere the closest face is always among the first few candidates
    (checked against the exhaustive search on the bundled grid). A point that
    ends up farther from its best candidate than a point on the unit sphere can
    be from its own face is re-queried against every face, so an irregular
    mesh stays exact, only slower. Ties -- a closest point at a shared vertex,
    which every face around it carries bit for bit -- go to the lowest face
    index, as the exhaustive search does (and as libigl did on every point of
    the ENCORE reference).
    """

    def __init__(self, vertices, faces, candidates: int = 12, block: int = 32768):
        from scipy.spatial import cKDTree

        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.candidates = int(min(candidates, self.faces.shape[0]))
        self.block = int(block)
        self._a = self.vertices[self.faces[:, 0]]
        self._b = self.vertices[self.faces[:, 1]]
        self._c = self.vertices[self.faces[:, 2]]
        centroids = (self._a + self._b + self._c) / 3.0
        self._tree = cKDTree(centroids)
        # A point of the unit sphere lies at most one sagitta above its own flat
        # face; anything farther than a few sagittas means the candidates missed.
        sagitta = float(np.abs(1.0 - np.linalg.norm(centroids, axis=1)).max())
        self._tolerance = 4.0 * sagitta + 1e-9

    def query(self, points):
        """``(weights, vertex_indices)``, both ``(n, 3)``."""
        weights, indices, _ = self.query_faces(points)
        return weights, indices

    def query_faces(self, points):
        """``(weights, vertex_indices, face_indices)`` for each point."""
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        weights = np.empty((n, 3))
        indices = np.empty((n, 3), dtype=np.int64)
        face_of = np.empty(n, dtype=np.int64)
        for start in range(0, n, self.block):
            chunk = points[start : start + self.block]
            m = chunk.shape[0]
            _, near = self._tree.query(chunk, k=self.candidates)
            near = near.reshape(m, self.candidates)
            near.sort(axis=1)  # argmin's first minimum is then the lowest face index
            closest = _closest_on_triangles(
                chunk[:, None, :], self._a[near], self._b[near], self._c[near]
            )
            distance = ((closest - chunk[:, None, :]) ** 2).sum(-1)
            best = np.argmin(distance, axis=1)
            rows = np.arange(m)
            face = near[rows, best]
            far = np.sqrt(distance[rows, best]) > self._tolerance
            if far.any():
                face[far] = self._exact_faces(chunk[far])

            a, b, c = self._a[face], self._b[face], self._c[face]
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            projected = chunk - ((chunk - a) * normal).sum(axis=1, keepdims=True) * normal

            face_of[start : start + self.block] = face
            indices[start : start + self.block] = self.faces[face]
            weights[start : start + self.block] = barycentric_coordinates(projected, a, b, c)
        return weights, indices, face_of

    def _exact_faces(self, points):
        """The closest face by testing every face, for the points the tree may have missed."""
        out = np.empty(points.shape[0], dtype=np.int64)
        for start in range(0, points.shape[0], 256):
            chunk = points[start : start + 256]
            closest = _closest_on_triangles(
                chunk[:, None, :], self._a[None], self._b[None], self._c[None]
            )
            out[start : start + 256] = np.argmin(
                ((closest - chunk[:, None, :]) ** 2).sum(-1), axis=1
            )
        return out


def interpolation_operator(weights, indices, n_vertices):
    """The sparse matrix that applies one set of barycentric weights."""
    from scipy import sparse

    rows = np.repeat(np.arange(weights.shape[0]), 3)
    return sparse.csr_matrix(
        (weights.ravel(), (rows, indices.ravel())), shape=(weights.shape[0], n_vertices)
    )


PARALLEL_ELEMENTS = 1 << 21
"""Dense operands smaller than this go through a single SciPy call in
:func:`sparse_times_dense`; the threads are not worth starting below it."""


def _thread_count() -> int:
    """Cores this process may use, capped at eight."""
    try:
        cores = len(os.sched_getaffinity(0))  # respects a SLURM allocation
    except AttributeError:  # pragma: no cover - macOS has no affinity API
        cores = os.cpu_count() or 1
    return max(1, min(8, cores))


def sparse_times_dense(matrix, dense, threads: int | None = None) -> np.ndarray:
    """``matrix @ dense`` for a sparse matrix and a dense ``(n, m)`` array, on several cores.

    SciPy's product runs on one core and is bound by traffic on the dense
    operand. Splitting the sparse matrix into row blocks and multiplying them
    in threads computes every output row exactly as the single call would --
    the same terms in the same order, so the same bits -- and on four cores
    runs about twice as fast. Below :data:`PARALLEL_ELEMENTS` elements a single
    call is made. This is the product behind the density evaluations of both
    alignment methods.
    """
    from scipy import sparse

    dense = np.ascontiguousarray(dense, dtype=np.float64)
    threads = _thread_count() if threads is None else int(threads)
    rows = matrix.shape[0]
    if threads <= 1 or dense.size < PARALLEL_ELEMENTS or rows < 2 * threads:
        return np.asarray(matrix @ dense)
    matrix = sparse.csr_matrix(matrix)
    edges = np.linspace(0, rows, threads + 1).astype(int)
    out = np.empty((rows, dense.shape[1]), dtype=np.float64)

    def block(i):
        out[edges[i] : edges[i + 1]] = matrix[edges[i] : edges[i + 1]] @ dense

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(block, range(threads)))
    return out


def interpolate_product(left, right, values):
    """Interpolate a function of two points, as ``bary_interp_2D_mex`` does.

    ``left`` and ``right`` are ``(weights, indices)`` pairs. The MEX writes
    ``out[j, i]``, which makes the whole thing ``W_right @ values @ W_left'``,
    computed as two sparse-times-dense products (the second on the transpose).
    """
    n = values.shape[0]
    halfway = sparse_times_dense(interpolation_operator(*right, n), values)
    return sparse_times_dense(interpolation_operator(*left, n), halfway.T).T


# --- geometry objects ------------------------------------------------------


def gradient_operators(vertices, faces, e1, e2):
    """Exact directional-derivative operators for a piecewise-linear field.

    A function given by values at the vertices is linear on each triangle, so
    its gradient there is exact::

        grad f = sum_i f_i * (n x edge_opposite_i) / (2 * area)

    At a vertex the gradient is defined only per face, so the faces meeting
    there are averaged with weights proportional to their area -- the standard
    choice, and the one that converges to the smooth gradient under
    refinement. Projecting onto the frame gives two sparse ``(P, P)`` matrices
    with ``(G1 @ f)[k] = df/de1`` at vertex ``k``.

    This replaces the reference's central difference, which divides a
    difference of order ``1e-10`` by a step of ``1e-10`` and so loses six of
    sixteen digits. See the conditioning note in the module docstring.
    """
    from scipy import sparse

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = vertices[faces]

    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normal = cross / double_area[:, None]
    face_area = 0.5 * double_area

    coefficients = np.empty_like(corners)
    for corner in range(3):
        opposite = corners[:, (corner + 2) % 3] - corners[:, (corner + 1) % 3]
        coefficients[:, corner] = np.cross(normal, opposite) / double_area[:, None]

    n = vertices.shape[0]
    weight_total = np.zeros(n)
    np.add.at(weight_total, faces.ravel(), np.repeat(face_area, 3))
    weight_total[weight_total == 0] = 1.0

    rows, cols, first, second = [], [], [], []
    for corner in range(3):
        at = faces[:, corner]
        share = (face_area / weight_total[at])[:, None]
        for other in range(3):
            rows.append(at)
            cols.append(faces[:, other])
            contribution = coefficients[:, other] * share
            first.append((contribution * e1[at]).sum(axis=1))
            second.append((contribution * e2[at]).sum(axis=1))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    shape = (n, n)
    return (
        sparse.csr_matrix((np.concatenate(first), (rows, cols)), shape=shape),
        sparse.csr_matrix((np.concatenate(second), (rows, cols)), shape=shape),
    )


class SphericalGrid:
    """A spherical mesh with its Voronoi areas and tangent basis.

    ``order`` is the spherical harmonic order of the tangent basis.
    ``reference=True`` builds the basis with the reference's Legendre
    recurrence, whose zonal derivatives are wrong by ``(1 + 1/(l(l+1))) / 2``
    (see :func:`tangent_basis`); the basis fields are unchanged to rounding
    but their divergence, ``laplacian``, is not, and it enters every
    registration gradient. The setting is kept as ``reference`` so the
    estimators built on the grid can see what they were given.
    """

    def __init__(self, vertices, faces, order: int = DEFAULT_ORDER, reference: bool = False):
        self.vertices = normalize_rows(vertices)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.reference = bool(reference)
        self.theta, self.phi = cart_to_sphere(self.vertices)
        self.areas = voronoi_areas(self.vertices, self.faces)
        self.basis, self.laplacian = tangent_basis(
            order, self.theta, self.phi, self.areas, self.reference
        )
        self.e1 = np.stack(
            [
                np.cos(self.theta) * np.cos(self.phi),
                np.cos(self.theta) * np.sin(self.phi),
                -np.sin(self.theta),
            ],
            axis=1,
        )
        self.e2 = np.stack([-np.sin(self.phi), np.cos(self.phi), np.zeros_like(self.phi)], axis=1)
        self._gradients = None

    @property
    def gradients(self):
        """Exact directional-derivative operators, built once on first use."""
        if self._gradients is None:
            self._gradients = gradient_operators(self.vertices, self.faces, self.e1, self.e2)
        return self._gradients

    @property
    def n_vertices(self) -> int:
        """Vertices in the mesh."""
        return int(self.vertices.shape[0])

    def pole_vertices(self, tolerance: float = 1e-8) -> np.ndarray:
        """Vertices sitting on the axis of the spherical coordinate system.

        The Jacobian is formed in ``(theta, phi)`` coordinates and closes with
        a factor of ``sin(theta)``, which vanishes at the poles. A vertex there
        therefore gets a Jacobian of exactly zero, and a density pushed through
        the warp loses that vertex's whole row and column. The sphere has no
        distinguished axis, so this is an artifact of the parametrization
        rather than of the data -- see :func:`rotate_off_poles`.
        """
        return np.flatnonzero(np.sin(self.theta) < tolerance)


def parallel_transport(tangent, origin, destination):
    """Move tangent vectors along the geodesic from ``origin`` to ``destination``.

    ENCORE's closed form; :func:`sbci.conseal._transport` is ConSEAL's, the same
    transport written differently and with its own degenerate threshold. The two
    agree to 1e-13 and are kept apart so that each port stays bit-for-bit its
    reference's.
    """
    tangent = np.asarray(tangent, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    eps = np.finfo(float).eps

    axis = np.cross(origin, destination)
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + eps)
    before = np.cross(axis, origin)
    before = before / (np.linalg.norm(before, axis=1, keepdims=True) + eps)
    after = np.cross(axis, destination)
    after = after / (np.linalg.norm(after, axis=1, keepdims=True) + eps)

    moved = (tangent * before).sum(axis=1, keepdims=True) * after + (tangent * axis).sum(
        axis=1, keepdims=True
    ) * axis

    unchanged = (np.linalg.norm(origin - destination, axis=1) < 1e-4) | (
        np.linalg.norm(origin + destination, axis=1) < 1e-4
    )
    moved[unchanged] = tangent[unchanged]
    return moved


def triangles_fold(vertices, faces) -> bool:
    """Whether any triangle has turned inside out, by the reference's normal test.

    A face is folded when its normal points into the sphere, ``((p2 - p1) x
    (p3 - p1)) . (p1 + p2 + p3) <= 0``, so the faces must be oriented outward
    (counter-clockwise seen from outside) for the test to mean anything; an
    inward mesh reads as folded everywhere. Shared by ENCORE's
    :class:`SphericalWarp` and ConSEAL's :class:`sbci.conseal.StationaryWarp`.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    p1, p2, p3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = (np.cross(p2 - p1, p3 - p1) * (p1 + p2 + p3)).sum(axis=1)
    return bool(np.any(normals <= 0))


class SphericalWarp:
    """A diffeomorphism of the sphere, as the image of each vertex.

    The Jacobian is the reference's central difference through four offset
    points relocated on the planar mesh (:meth:`_raw_jacobian`). That scheme
    carries the mesh's chord-versus-arc error: on the identity it returns not
    1 but 0.9965 on the ico4 grid (0.805 on a 42-vertex icosphere), and since
    :meth:`Concon.evaluate` transports a density with the product of two
    Jacobians, every accepted step would take about 0.7% of its mass on ico4.
    So the scheme's value on the identity is computed once, at construction,
    and every Jacobian is divided by it: the identity gives exactly 1 and a
    small warp 1 plus its divergence. ``reference=True`` leaves the Jacobian
    uncalibrated, as the reference computes it, for the MATLAB comparison. A
    vertex on the coordinate axis has an identity Jacobian of zero
    (:meth:`SphericalGrid.pole_vertices`) and is left as it is: there is
    nothing there to calibrate against.
    """

    def __init__(self, grid: SphericalGrid, delta: float = DEFAULT_DELTA, reference: bool = False):
        self.grid = grid
        self.faces = grid.faces
        self.base = grid.vertices
        self.vertices = grid.vertices.copy()
        self.jacobian = np.ones(grid.n_vertices)
        self.reference = bool(reference)
        # The fold test wants outward faces. A mesh wound the other way is a
        # valid grid everywhere else (areas, queries and the Jacobian do not
        # care), so test it with its faces reversed rather than refuse it.
        if not triangles_fold(self.base, self.faces):
            self._fold_faces = self.faces
        elif not triangles_fold(self.base, self.faces[:, ::-1]):
            self._fold_faces = self.faces[:, ::-1]
        else:
            raise ValueError("the mesh faces are not consistently oriented")
        self._e1, self._e2 = grid.e1, grid.e2
        self._step = 2 * delta
        self._query = MeshQuery(grid.vertices, grid.faces)
        self._offsets = {
            name: self._query.query(sphere_exp_map(grid.vertices, frame * sign * delta))
            for name, frame, sign in (
                ("px", grid.e1, +1),
                ("mx", grid.e1, -1),
                ("py", grid.e2, +1),
                ("my", grid.e2, -1),
            )
        }
        # The scheme's own value on the identity, divided out of every Jacobian
        # (class docstring); a zero, at a vertex on the axis, is left alone.
        identity = self._raw_jacobian()
        self._calibration = np.where(identity > 0, identity, 1.0)
        self._calibration[grid.pole_vertices()] = 1.0

    def compose(self, displacement) -> SphericalWarp:
        """Compose with a further displacement given in the tangent basis."""
        displacement = np.asarray(displacement, dtype=np.float64)
        tangent = self._e1 * displacement[:, :1] + self._e2 * displacement[:, 1:2]

        weights, indices = self._query.query(self.vertices)
        n = self.vertices.shape[0]
        transported = parallel_transport(
            tangent[indices].reshape(-1, 3),
            self.base[indices].reshape(-1, 3),
            np.repeat(self.vertices, 3, axis=0),
        ).reshape(n, 3, 3)
        tangent = (weights[:, :, None] * transported).sum(axis=1)

        clone = object.__new__(SphericalWarp)
        clone.__dict__.update(self.__dict__)
        clone.vertices = normalize_rows(sphere_exp_map(self.vertices, tangent))
        clone.jacobian = clone._compute_jacobian()
        return clone

    def folds(self) -> bool:
        """Whether the warp has turned a face inside out, so it is no longer a diffeomorphism.

        The Jacobian cannot tell: it is the absolute value of the determinant,
        the right factor for transporting a density but blind to a flip.
        """
        return triangles_fold(self.vertices, self._fold_faces)

    def _raw_jacobian(self):
        """Determinant of the differential by central differences, as the reference forms it."""
        angles = {}
        for name, (weights, indices) in self._offsets.items():
            point = (weights[:, :, None] * self.vertices[indices]).sum(axis=1)
            angles[name] = cart_to_sphere(normalize_rows(point))

        wrap = np.arange(-2, 3) * np.pi

        def difference(plus, minus):
            """Smallest change modulo pi, so theta and phi may wrap."""
            candidates = (plus - minus)[:, None] + wrap[None, :]
            chosen = np.argmin(np.abs(candidates), axis=1)
            return candidates[np.arange(candidates.shape[0]), chosen] / self._step

        d_tt = difference(angles["px"][0], angles["mx"][0])
        d_pt = difference(angles["px"][1], angles["mx"][1])
        d_tp = difference(angles["py"][0], angles["my"][0])
        d_pp = difference(angles["py"][1], angles["my"][1])

        theta, _ = cart_to_sphere(self.vertices)
        return np.abs(((d_tt * d_pp) - (d_tp * d_pt)) * np.sin(theta))

    def _compute_jacobian(self):
        """The Jacobian, divided by the scheme's value on the identity unless ``reference``."""
        jacobian = self._raw_jacobian()
        return jacobian if self.reference else jacobian / self._calibration


class Concon:
    """A connectivity density over the product of two spherical meshes.

    ``reference=True`` makes :meth:`evaluate_root` normalize before it zeroes
    the diagonal, as ``Concon.m`` does, so the transported square-root density
    comes back short of unit norm by the mass interpolation put on the
    diagonal: 0.02 to 0.2% on ico4 for realistic warps. The default zeroes
    first and then normalizes.
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        delta: float = DEFAULT_DELTA,
        derivative: str = "difference",
        reference: bool = False,
    ):
        if derivative not in ("analytic", "difference"):
            raise ValueError(f"derivative must be 'analytic' or 'difference', got {derivative!r}")
        self.lh_grid = lh_grid
        self.rh_grid = rh_grid
        self.derivative_method = derivative
        self.reference = bool(reference)
        self._operators = None
        self.step = 2 * delta
        self.n_per_hemi = lh_grid.n_vertices
        self.areas = np.concatenate([lh_grid.areas, rh_grid.areas])
        self.area_product = np.outer(self.areas, self.areas)

        self._lh_query = MeshQuery(lh_grid.vertices, lh_grid.faces)
        self._rh_query = MeshQuery(rh_grid.vertices, rh_grid.faces)
        self._grid = self._coordinates(lh_grid.vertices, rh_grid.vertices)
        self._offsets = {
            name: self._coordinates(
                sphere_exp_map(lh_grid.vertices, lh_frame * sign * delta),
                sphere_exp_map(rh_grid.vertices, rh_frame * sign * delta),
            )
            for name, lh_frame, rh_frame, sign in (
                ("px", lh_grid.e1, rh_grid.e1, +1),
                ("mx", lh_grid.e1, rh_grid.e1, -1),
                ("py", lh_grid.e2, rh_grid.e2, +1),
                ("my", lh_grid.e2, rh_grid.e2, -1),
            )
        }

    def _coordinates(self, lh_points, rh_points):
        """Barycentric data for both hemispheres, right-hemisphere indices offset."""
        lh_weights, lh_indices = self._lh_query.query(lh_points)
        rh_weights, rh_indices = self._rh_query.query(rh_points)
        return (
            np.vstack([lh_weights, rh_weights]),
            np.vstack([lh_indices, rh_indices + self.n_per_hemi]),
        )

    def derivative(self, values):
        """Directional derivatives along the two frame fields.

        ``difference`` is the reference's central difference and the default.
        ``analytic`` differentiates the piecewise-linear interpolant exactly,
        averaging the per-face gradients at each vertex.

        Both estimate the same thing and both converge to the closed-form
        surface gradient under refinement, the analytic one somewhat faster.
        How far apart they are depends entirely on how smooth the field is at
        the scale of the mesh:

        ===================================  =========  ============
        field                                max diff   correlation
        ===================================  =========  ============
        smooth analytic field                0.019      0.99996
        kernel-smoothed noise                0.049      0.99946
        a real smoothed connectome on ico4   0.40       0.9925
        white noise                          0.57       0.892
        ===================================  =========  ============

        A density that varies at grid scale does not determine its own
        derivative from vertex samples, so the two legitimately part company
        there. Real connectomes sit closer to the noisy end than one might
        hope, which is why the default stays with the reference estimator.
        """
        if self.derivative_method == "analytic":
            from scipy import sparse

            if self._operators is None:
                lh_first, lh_second = self.lh_grid.gradients
                rh_first, rh_second = self.rh_grid.gradients
                self._operators = (
                    sparse.block_diag((lh_first, rh_first), format="csr"),
                    sparse.block_diag((lh_second, rh_second), format="csr"),
                )
            first, second = self._operators
            return np.asarray(first @ values), np.asarray(second @ values)

        def at(key):
            return interpolate_product(self._grid, self._offsets[key], values)

        return (at("px") - at("mx")) / self.step, (at("py") - at("my")) / self.step

    def evaluate(self, values, lh_warp: SphericalWarp, rh_warp: SphericalWarp):
        """Push a density through a pair of warps."""
        jacobian = np.concatenate([lh_warp.jacobian, rh_warp.jacobian])
        coords = self._coordinates(lh_warp.vertices, rh_warp.vertices)
        warped = interpolate_product(coords, coords, values) * np.outer(jacobian, jacobian)
        warped = np.maximum((warped + warped.T) / 2, 0)
        np.fill_diagonal(warped, 0.0)
        return warped

    def evaluate_root(self, root, lh_warp: SphericalWarp, rh_warp: SphericalWarp):
        """Push a square-root density through, keeping it a unit vector.

        Interpolation leaks mass onto the diagonal, which the density does not
        carry, so the diagonal is zeroed before the result is normalized. The
        reference (``Concon.m`` lines 62-63) normalizes first, and its result
        is short of unit norm by that leaked mass; ``reference=True`` on the
        constructor reproduces it. A density whose every entry lands on the
        diagonal comes back as zeros rather than as a division by zero.
        """
        jacobian = np.sqrt(np.concatenate([lh_warp.jacobian, rh_warp.jacobian]))
        coords = self._coordinates(lh_warp.vertices, rh_warp.vertices)
        warped = interpolate_product(coords, coords, root) * np.outer(jacobian, jacobian)
        warped = np.maximum((warped + warped.T) / 2, 0)
        if self.reference:
            warped = warped / np.sqrt((warped**2 * self.area_product).sum())
            np.fill_diagonal(warped, 0.0)
            return warped
        np.fill_diagonal(warped, 0.0)
        norm = np.sqrt((warped**2 * self.area_product).sum())
        return warped / norm if norm > 0 else warped


# --- the estimator ---------------------------------------------------------


def _npz_path(path) -> Path:
    """The file ``np.savez_compressed`` writes: it appends ``.npz`` when the name lacks it."""
    path = Path(path)
    return path if str(path).endswith(".npz") else Path(f"{path}.npz")


@dataclass
class Warp:
    """One subject's diffeomorphism of both hemispheres.

    The vertices are in the frame of the grid the warp was estimated on, which
    for the bundled sphere is rotated off the coordinate poles
    (:func:`rotate_off_poles`). ``lh_rotation`` and ``rh_rotation`` are those
    rotations, identities for a grid that needed none; ``lh_vertices @
    lh_rotation`` is back in the file's frame. They are saved with the warp,
    and :func:`sbci.migrate_warp` reads them, so a warp reloaded from its file
    still knows its frame.
    """

    lh_vertices: np.ndarray
    lh_jacobian: np.ndarray
    rh_vertices: np.ndarray
    rh_jacobian: np.ndarray
    lh_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    rh_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def save(self, path) -> Path:
        """Write the warp to ``.npz``; the path returned is the file written, suffix included."""
        path = _npz_path(path)
        np.savez_compressed(
            path,
            lh_vertices=self.lh_vertices,
            lh_jacobian=self.lh_jacobian,
            rh_vertices=self.rh_vertices,
            rh_jacobian=self.rh_jacobian,
            lh_rotation=self.lh_rotation,
            rh_rotation=self.rh_rotation,
        )
        return path

    @classmethod
    def load(cls, path) -> Warp:
        """Read a warp written by :meth:`save`; a file without the rotations gets identities."""
        with np.load(Path(path)) as data:
            return cls(
                lh_vertices=data["lh_vertices"],
                lh_jacobian=data["lh_jacobian"],
                rh_vertices=data["rh_vertices"],
                rh_jacobian=data["rh_jacobian"],
                lh_rotation=data["lh_rotation"] if "lh_rotation" in data.files else np.eye(3),
                rh_rotation=data["rh_rotation"] if "rh_rotation" in data.files else np.eye(3),
            )


@dataclass
class Alignment:
    """What :func:`align` returns.

    ``grid_rotations`` are the rotations applied to the left and right
    hemispheres of the bundled sphere before alignment (see
    :func:`rotate_off_poles`); the vertices in each :class:`Warp` live in that
    rotated frame, and ``warp.lh_vertices @ grid_rotations[0]`` maps them back
    to the file's sphere. Each warp carries the same two rotations as its
    ``lh_rotation`` and ``rh_rotation``, so a warp saved on its own keeps its
    frame. Both are identities when ``grids`` were supplied.
    """

    template: np.ndarray
    warps: list = field(default_factory=list)
    aligned: list = field(default_factory=list)
    costs: list = field(default_factory=list)
    grid_rotations: tuple = (np.eye(3), np.eye(3))
    traces: list = field(default_factory=list)
    """Per subject, the cost before registration and after every accepted step."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Alignment of {len(self.warps)} subjects on {self.template.shape[0]} vertices>"


def _without_diagonal(density) -> np.ndarray:
    """A float64 copy of ``density`` with its diagonal zeroed; the input is left alone."""
    density = np.array(density, dtype=np.float64)
    np.fill_diagonal(density, 0.0)
    return density


class Encore:
    """Template estimation and registration on a pair of spherical meshes.

    ``reference=True`` is handed to the :class:`Concon` underneath, whose
    :meth:`Concon.evaluate_root` then loses unit norm as the reference's does,
    and to the :class:`SphericalWarp` pair of every registration, whose
    Jacobian is then left uncalibrated (0.9965 on the identity instead of 1);
    the grids carry their own ``reference`` setting for the tangent basis. A
    step that folds a face is refused in both modes (:meth:`register`).
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        step: float = 0.05,
        max_iterations: int = 50,
        threshold: float = 1e-8,
        delta: float = DEFAULT_DELTA,
        derivative: str = "difference",
        backtracks: int = 4,
        reference: bool = False,
    ):
        self.lh_grid = lh_grid
        self.rh_grid = rh_grid
        self.reference = bool(reference)
        self.concon = Concon(
            lh_grid, rh_grid, delta, derivative=derivative, reference=self.reference
        )
        self.area_product = self.concon.area_product
        self.step = step
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.delta = delta
        self.backtracks = int(backtracks)

    def root(self, density):
        """The square-root density, normalized to unit mass.

        The diagonal of a connectivity density is not part of it: a
        :class:`~sbci.ContinuousConnectome` carries none, and
        :meth:`Concon.evaluate_root` zeroes it on every transported image. It
        is zeroed here too, before the normalization, so that a raw array
        with a nonzero diagonal starts :meth:`register` at the cost its trial
        steps are measured by; left in, it sat in every residual and no step
        could ever be accepted. The input is not modified.
        """
        density = _without_diagonal(density)
        return np.sqrt(density / (density * self.area_product).sum())

    def template(self, densities, iterations: int = 10):
        """Karcher median of the square-root densities."""
        roots = np.stack([self.root(d) for d in densities])
        mean = roots.mean(axis=0)
        distances = [((root - mean) ** 2 * self.area_product).sum() for root in roots]
        current = roots[int(np.argmin(distances))].copy()

        for _ in range(iterations):
            # Summed as they are computed: a second cohort-sized array would cost
            # another 210 MB per subject on the ico4 grid.
            directions = np.zeros_like(current)
            distance = np.zeros(roots.shape[0])
            for i, root in enumerate(roots):
                inner = (root * current * self.area_product).sum()
                if 1 - abs(inner) < 1e-14:
                    inner = np.sign(inner)
                distance[i] = np.arccos(np.clip(inner, -1.0, 1.0))
                if distance[i] > 0:
                    directions += (root - np.cos(distance[i]) * current) / np.sin(distance[i])

            moving = distance > 0
            if not moving.any():
                break
            mean_direction = directions / (1.0 / distance[moving]).sum()
            size = np.sqrt((mean_direction**2 * self.area_product).sum())
            if size == 0:
                break
            current = np.cos(0.2 * size) * current + np.sin(0.2 * size) * (mean_direction / size)
            current = current / np.sqrt((current**2 * self.area_product).sum())
            if size < 0.005:
                break
        return current

    def register(
        self, target, moving, target_is_root: bool = False, verbose: bool = False, callback=None
    ):
        """Warp ``moving`` onto ``target``; returns result, warps and cost.

        A step is accepted when it lowers the cost by at least ``threshold``
        and folds no face on either hemisphere; otherwise it is halved and
        retried up to ``backtracks`` times, and the registration stops when
        every attempt fails. ``callback(iteration, cost)`` is called after
        every accepted step.
        """
        lh_warp = SphericalWarp(self.lh_grid, self.delta, reference=self.reference)
        rh_warp = SphericalWarp(self.rh_grid, self.delta, reference=self.reference)

        fixed = np.asarray(target, dtype=np.float64) if target_is_root else self.root(target)
        # Not part of the density (see root); the final evaluate must not see
        # it either, or it would interpolate that mass off the diagonal.
        moving = _without_diagonal(moving)
        source = self.root(moving)

        image = source
        residual = fixed - image
        last_cost = (residual**2 * self.area_product).sum()
        cost = last_cost
        last = (lh_warp, rh_warp)
        n = self.lh_grid.n_vertices

        scale = 1.0  # of self.step: halved when a step fails, doubled back after a success
        for iteration in range(1, self.max_iterations + 1):
            d_e1, d_e2 = self.concon.derivative(image)
            weighted = residual * self.area_product
            a = (weighted * (2 * d_e1)).sum(axis=1)
            b = (weighted * (2 * d_e2)).sum(axis=1)
            c = (weighted * image).sum(axis=1)

            moves = []
            for grid, lo, hi in (
                (self.lh_grid, 0, n),
                (self.rh_grid, n, n + self.rh_grid.n_vertices),
            ):
                gradient = 2 * (
                    a[lo:hi] @ grid.basis[:, :, 0]
                    + b[lo:hi] @ grid.basis[:, :, 1]
                    + c[lo:hi] @ grid.laplacian
                )
                direction = (gradient[None, :, None] * grid.basis).sum(axis=1)
                moves.append(direction * (self.step / (np.linalg.norm(gradient) + 1e-15)))

            # The reference takes one step of fixed length and stops the moment
            # it fails to lower the cost -- on a small deformation, the very
            # first step. Halve the length a few times before giving up;
            # backtracks=0 reproduces the reference (PORTING.md item 4).
            accepted = False
            for _attempt in range(self.backtracks + 1):
                trial = (last[0].compose(scale * moves[0]), last[1].compose(scale * moves[1]))
                # A warp that folds a face is not a diffeomorphism, whatever
                # it does to the cost, and is refused like a cost increase.
                # The reference accepts it: its Jacobian is an absolute value,
                # so a flipped face still reads as positive. This holds in
                # both modes -- the reference only ever folds at step lengths
                # far above its default, so the comparison is unaffected.
                if trial[0].folds() or trial[1].folds():
                    scale *= 0.5
                    continue
                trial_image = self.concon.evaluate_root(source, *trial)
                trial_residual = fixed - trial_image
                trial_cost = (trial_residual**2 * self.area_product).sum()
                if (last_cost - trial_cost) >= self.threshold:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                lh_warp, rh_warp = last
                cost = last_cost
                if verbose:
                    print(f"converged at iteration {iteration}, cost {cost:.6f}")
                break

            lh_warp, rh_warp = trial
            image, residual, cost = trial_image, trial_residual, trial_cost
            last = (lh_warp, rh_warp)
            last_cost = cost
            scale = min(1.0, 2.0 * scale)
            if callback is not None:
                callback(iteration, cost)

        result = self.concon.evaluate(moving, lh_warp, rh_warp)
        return result, lh_warp, rh_warp, cost


def pole_rotation(vertices, tolerance: float = 1e-3) -> np.ndarray:
    """The rotation :func:`rotate_off_poles` applies, as a ``3 x 3`` matrix.

    ``rotate_off_poles(v) == v @ pole_rotation(v).T``, so ``rotated @ R`` maps
    points back into the mesh's original frame.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    for angle in (0.0, 0.3, 0.7, 1.1, 1.7):
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        candidate = vertices @ rotation.T
        if np.sin(np.arccos(np.clip(candidate[:, 2], -1.0, 1.0))).min() > tolerance:
            return rotation
    raise RuntimeError("could not rotate the mesh clear of the coordinate poles")


def rotate_off_poles(vertices, tolerance: float = 1e-3):
    """Rotate a spherical mesh so that no vertex lies on the coordinate axis.

    The ico4 grid puts four vertices exactly at the poles, where the
    ``(theta, phi)`` parametrization is singular and the Jacobian collapses to
    zero. A rotation is a change of coordinates and nothing else: areas,
    distances and the connectivity are untouched, and the vertex order is
    unchanged. The angles below are irrational multiples of the icosahedral
    symmetry, so none of them can bring a vertex back onto the axis.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices @ pole_rotation(vertices, tolerance).T


def _hemisphere_grids(
    order, rotate: bool = True, return_rotations: bool = False, reference: bool = False
):
    """The bundled ico4 sphere, split into two hemispheres.

    With ``return_rotations`` the pair of ``3 x 3`` rotations applied to the two
    hemispheres comes back as well (identities when ``rotate`` is false).
    ``reference`` is passed on to :class:`SphericalGrid`.
    """
    from .spec import N_VERTICES_PER_HEMI
    from .surface import load_surface

    sphere = load_surface("sphere")
    vertices = normalize_rows(np.asarray(sphere.vertices, dtype=np.float64))
    faces = np.asarray(sphere.faces, dtype=np.int64)
    n = N_VERTICES_PER_HEMI

    left = faces[(faces < n).all(axis=1)]
    right = faces[(faces >= n).all(axis=1)] - n
    lh_vertices, rh_vertices = vertices[:n], vertices[n:]
    lh_rotation, rh_rotation = np.eye(3), np.eye(3)
    if rotate:
        lh_rotation, rh_rotation = pole_rotation(lh_vertices), pole_rotation(rh_vertices)
        lh_vertices = lh_vertices @ lh_rotation.T
        rh_vertices = rh_vertices @ rh_rotation.T
    grids = (
        SphericalGrid(lh_vertices, left, order, reference=reference),
        SphericalGrid(rh_vertices, right, order, reference=reference),
    )
    if return_rotations:
        return grids, (lh_rotation, rh_rotation)
    return grids


def align(
    cc_list,
    method: str = "encore",
    order: int = DEFAULT_ORDER,
    step: float = 0.05,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    delta: float = DEFAULT_DELTA,
    template_iterations: int = 10,
    template=None,
    grids=None,
    derivative: str = "difference",
    backtracks: int = 4,
    reference: bool = False,
    verbose: bool = False,
) -> Alignment:
    """Estimate a template and register every connectome onto it.

    Parameters
    ----------
    cc_list
        Connectomes to align, all on the same grid. Each may be a
        :class:`~sbci.ContinuousConnectome` or a dense square array.
    method
        Only ``"encore"`` for now.
    order
        Spherical harmonic order of the tangent basis.
    step, max_iterations, threshold
        Gradient descent controls, as in the reference.
    delta
        Central-difference step; see the conditioning note in this module.
    template
        What every connectome is registered onto. ``None``, the default,
        estimates it first: the Karcher median of the connectomes' square-root
        densities, so that no subject is the reference and every subject
        moves, which is what a cohort wants. A :class:`~sbci.ContinuousConnectome`
        registers the connectomes in the list onto that subject, which stays
        where it is and need not be in the list: this is how one subject is
        registered onto another. A **square-root** density array of unit
        mass in the area inner product, as :attr:`Alignment.template` is,
        registers onto that: an earlier run's template, to bring a new subject
        onto it without re-estimating. With a template given, one connectome
        is enough.
    grids
        ``(lh_grid, rh_grid)`` to align on, as :class:`SphericalGrid`. Defaults
        to the bundled ico4 sphere, split at the hemisphere boundary.
    derivative
        ``"difference"``, the default, is the reference's central difference at
        ``delta``. ``"analytic"`` differentiates the piecewise-linear
        interpolant exactly instead. Both converge to the same surface
        gradient; they part company where the density varies at grid scale.
        See :meth:`Concon.derivative`.
    backtracks
        How many times a step that fails to lower the cost is halved before
        the registration stops. The reference stops at the first such step,
        which on a small deformation is the first step of all; ``0``
        reproduces that.
    reference
        Reproduce the reference's arithmetic in the three places the port
        departs from it, for comparison with its MATLAB output; ``False``,
        the default, corrects all three. The reference's Legendre recurrence
        treats the normalized functions as unnormalized at ``m = 0``, so
        every zonal field's derivative in theta is scaled by
        ``(1 + 1/(l(l+1))) / 2`` -- 0.75 at degree 1, 0.502 at degree 15. The
        normalized basis fields come out the same, but their divergence
        (:attr:`SphericalGrid.laplacian`) is 4/3 of the true one at degree 1,
        and it enters every registration gradient.
        :meth:`Concon.evaluate_root` normalizes the transported density
        before zeroing its diagonal, so it falls short of unit norm by the
        mass interpolation put there: 0.02 to 0.2% on ico4. And the
        finite-difference Jacobian is left as the scheme returns it, 0.9965
        on the identity on ico4 rather than 1, so every accepted step takes
        about 0.7% of a density's mass; the default divides that bias out
        (:class:`SphericalWarp`). Grids passed in through ``grids`` keep the
        setting they were built with. A step that folds a face is refused in
        both modes; the reference's Jacobian is an absolute value and cannot
        see one, but it only ever folds at step lengths far above the
        default.

    Returns
    -------
    :class:`Alignment`, with the template, one :class:`Warp` per subject, the
    registered connectomes and the final costs.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if len(cc_list) == 0:
        raise ValueError("alignment needs at least one connectome")
    if template is None and len(cc_list) < 2:
        raise ValueError(
            "estimating a template needs at least two connectomes; pass template= to "
            "register one connectome onto an existing template"
        )

    densities = [
        np.asarray(cc.dense(), dtype=np.float64)
        if hasattr(cc, "dense")
        else np.asarray(cc, dtype=np.float64)
        for cc in cc_list
    ]
    shapes = {d.shape for d in densities}
    if len(shapes) != 1:
        raise ValueError(f"connectomes are on different grids: {sorted(shapes)}")
    for index, density in enumerate(densities):
        if not np.isfinite(density).all() or density.min() < 0 or density.sum() <= 0:
            raise ValueError(f"connectome {index} is not a nonnegative density with positive mass")

    if grids is None:
        (lh_grid, rh_grid), rotations = _hemisphere_grids(
            order, return_rotations=True, reference=reference
        )
    else:
        lh_grid, rh_grid = grids
        rotations = (np.eye(3), np.eye(3))
    for name, grid in (("left", lh_grid), ("right", rh_grid)):
        poles = grid.pole_vertices()
        if poles.size:
            raise ValueError(
                f"the {name} grid has {poles.size} vertices on the coordinate "
                f"axis ({poles[:4].tolist()}...), where the Jacobian is "
                "identically zero and those vertices would be dropped. Pass "
                "vertices through sbci.alignment.rotate_off_poles first."
            )
    expected = lh_grid.n_vertices + rh_grid.n_vertices
    if densities[0].shape != (expected, expected):
        raise ValueError(
            f"connectomes are {densities[0].shape} but the grid has {expected} vertices"
        )
    encore = Encore(
        lh_grid,
        rh_grid,
        step=step,
        max_iterations=max_iterations,
        threshold=threshold,
        delta=delta,
        derivative=derivative,
        backtracks=backtracks,
        reference=reference,
    )

    if template is None:
        template = encore.template(densities, iterations=template_iterations)
    elif hasattr(template, "dense"):
        # the fixed subject: its root density on the grids the others are registered on
        fixed = np.asarray(template.dense(), dtype=np.float64)
        if fixed.shape != (expected, expected):
            raise ValueError(
                f"the template connectome is {fixed.shape}, not {expected} x {expected}"
            )
        if not np.isfinite(fixed).all() or fixed.min() < 0 or fixed.sum() <= 0:
            raise ValueError(
                "the template connectome is not a nonnegative density with positive mass"
            )
        template = encore.root(fixed)
    template = np.asarray(template, dtype=np.float64)
    if template.shape != (expected, expected):
        raise ValueError(
            f"template is {template.shape}, expected a {expected} x {expected} square-root density"
        )

    result = Alignment(template=template, grid_rotations=rotations)
    for index, density in enumerate(densities):
        if verbose:
            print(f"registering subject {index + 1} of {len(densities)}")
        start = ((template - encore.root(density)) ** 2 * encore.area_product).sum()
        trace = [float(start)]

        def record(_iteration, value, trace=trace):
            trace.append(float(value))

        aligned, lh_warp, rh_warp, cost = encore.register(
            template, density, target_is_root=True, verbose=verbose, callback=record
        )
        result.warps.append(
            Warp(
                lh_warp.vertices,
                lh_warp.jacobian,
                rh_warp.vertices,
                rh_warp.jacobian,
                lh_rotation=rotations[0],
                rh_rotation=rotations[1],
            )
        )
        result.aligned.append(aligned)
        result.costs.append(float(cost))
        result.traces.append(np.asarray(trace))
    return result
