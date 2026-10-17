"""ENCORE: diffeomorphic alignment of continuous connectomes.

Port of ``ConCon_Alignment`` (Cole et al.). A connectome is treated as a
density on the product of two spheres; alignment finds, for each subject, a
diffeomorphism of each hemisphere's sphere that carries that subject's density
onto a common template. Registration is gradient descent on the L2 distance
between square-root densities, which is the Fisher-Rao metric, and the template
is their Karcher median.

Layers, checked against a MATLAB run of the reference with the reference's
own errors put back into the port's output (``tests/test_matlab_reference.py``),
all but the derivative of ``Concon``, which no implementation can reproduce
past the reference's conditioning (below) and a probe measures instead:

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
    Template estimation and registration. The template is compared with the
    reference's; the registration no longer is, since the port composes each
    step in the order its gradient assumes (PORTING.md item 21).

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
sits 0.0367 from the reference at every step size to 1e-6, which is the same
distance the reference sits from itself. That is the floor, and it is reached
(``tests/reference/encore_first_checks_probe.py``; PORTING.md item 4).
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _device

METHODS = ("encore",)

DEFAULT_ORDER = 6
"""Spherical harmonic order of the tangent basis: ``2(l+1)^2 - 2`` fields."""

DEFAULT_DELTA = 1e-5
"""Central-difference step. See the conditioning note in the module docstring."""

AXIS_CLEARANCE = 1e-3
"""How far every vertex must sit from the coordinate axis, as ``sin(theta)``, for the
``(theta, phi)`` Jacobian: what :func:`rotate_off_poles` guarantees and :func:`align`
checks (:meth:`SphericalGrid.pole_vertices`)."""


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


def _legendre_derivatives(degree, colatitude):
    """``(ddP, dP, P)`` for one degree, carrying the harmonic normalization.

    The derivative in theta comes from the recurrence on the **normalized**
    functions, ``2 dP~_l^m = sqrt((l-m)(l+m+1)) P~_l^{m+1} - sqrt((l+m)(l-m+1))
    P~_l^{m-1}``, whose ``m = -1`` member is ``P~_l^{-1} = -P~_l^{1}``. The
    reference writes that member as ``-P_l^1 / (l(l+1))``, the relation for
    the *unnormalized* functions, so its zonal (``m = 0``) derivative is scaled
    by ``(1 + 1/(l(l+1))) / 2``: 0.75 at degree 1, 0.502 at degree 15. Its
    second derivative also carries the ``m + 1`` term with the wrong sign.
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
    below = np.vstack([-values[1:2], values[:-1]])
    first = -0.5 * ((below * lower) - (above * upper))
    # the same recurrence on the first derivatives, whose m = -1 member is -dP~_l^1
    d_below = np.vstack([-first[1:2], first[:-1]])
    d_above = np.vstack([first[1:], padding])
    second = -0.5 * ((d_below * lower) - (d_above * upper))
    return second, first, values


def _harmonic_derivatives(degree, theta, phi):
    """Gradient and Laplacian of the real harmonics of one degree.

    Ordered ``Re(Y_0), Re(Y_1), Im(Y_1), ...`` as the reference orders them.

    The phi-component of the gradient is ``m P_l^m / sin(theta)`` times a sine
    or cosine of ``m phi``, which is 0/0 at a point on the coordinate axis. Its
    limit there is ``dP_l^m/dtheta / cos(theta)`` at ``m = 1`` and zero above,
    so the ``m = 1`` cosine and sine fields at the north pole are
    ``dP_l^1/dtheta`` times the x and the y axis, whatever phi the point was
    given, as the gradient of a smooth function must be. Within ``1e-5`` of
    the axis the ``m = 1`` term is taken as that limit, which is good to 1e-8
    there (degree 15); the quotient itself loses digits as ``cos(theta)``
    rounds towards 1 -- about ``eps / sin(theta)^2``, 5% at 2e-8 from the
    axis. Above ``m = 1`` the quotient stays accurate, and is zero on the
    axis. The reference divides by ``max(sin(theta), 1e-5)`` instead, which
    zeroes the phi-component on the axis, so at a pole with ``phi = pi`` (or
    0) every ``m = 1`` sine field vanishes while its cosine partner keeps its
    full size. Further than ``1e-5`` from the axis the two are the same
    arithmetic.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    _, first, values = _legendre_derivatives(degree, theta)

    order = np.arange(degree + 1)[:, None]
    sin_phi, cos_phi = np.sin(order * phi), np.cos(order * phi)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-5
    sin_theta = np.where(sin_theta > 0, sin_theta, 1.0)  # P_l^m, m > 0, is zero there too

    width = 2 * (degree + 1) - 1
    gradient = np.zeros((2, width, theta.size))
    laplacian = np.zeros((width, theta.size))

    cosine_slots = np.concatenate([[0], np.arange(1, width, 2)])
    sine_slots = np.arange(2, width, 2)

    gradient[0, cosine_slots] = first * cos_phi
    gradient[0, sine_slots] = first[1:] * sin_phi[1:]
    gradient[1, cosine_slots] = values * -(order * sin_phi) / sin_theta
    gradient[1, sine_slots] = values[1:] * (order[1:] * cos_phi[1:]) / sin_theta
    if degree > 0 and near.any():
        # l'Hopital, with cos(theta) = +1 or -1 on the axis; slots 1 and 2 are m = 1
        limit = first[1, near] / np.cos(theta[near])
        gradient[1, 1, near] = -limit * sin_phi[1, near]
        gradient[1, 2, near] = limit * cos_phi[1, near]

    scale = -degree * (degree + 1)
    laplacian[cosine_slots] = scale * values * cos_phi
    laplacian[sine_slots] = scale * values[1:] * sin_phi[1:]
    return gradient, laplacian


def tangent_basis(order, theta, phi, areas):
    """Gradient and rotated-gradient fields of the harmonics up to ``order``.

    Returns ``basis`` of shape ``(P, 2(order+1)^2 - 2, 2)`` in the ``(e1, e2)``
    frame, and the Laplacian of each field.

    The reference's Legendre recurrence (:func:`_legendre_derivatives`) scales
    every zonal gradient field by a constant, which the area-weighted
    normalization below removes again, so its basis is the same to rounding;
    the Laplacian of those fields, normalized by the same constant, comes out
    ``2 / (1 + 1/(l(l+1)))`` times too large in it -- 4/3 at degree 1. This
    one is the true divergence.
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
        gradient, laplacian = _harmonic_derivatives(degree, theta, phi)
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


def _dot(x, y, xp=np):
    """Dot products over a last axis of three, rounded alike on both devices.

    NumPy's sum adds the three products as ``(x0 y0 + x1 y1) + x2 y2``; CuPy's
    reduction adds them in another order, so on the GPU they are added in
    NumPy's, and the closest faces and weights come out as the CPU's do.
    """
    product = x * y
    if xp is np:
        return product.sum(-1)
    return (product[..., 0] + product[..., 1]) + product[..., 2]


def barycentric_coordinates(point, a, b, c, xp=np):
    """Libigl's ``barycentric_coordinates``, for points already in the plane.

    ``xp`` is NumPy or CuPy, as the arrays are.
    """
    v0, v1, v2 = b - a, c - a, point - a
    d00 = _dot(v0, v0, xp)
    d01 = _dot(v0, v1, xp)
    d11 = _dot(v1, v1, xp)
    d20 = _dot(v2, v0, xp)
    d21 = _dot(v2, v1, xp)
    denom = d00 * d11 - d01 * d01
    denom = xp.where(denom == 0, 1.0, denom)
    second = (d11 * d20 - d01 * d21) / denom
    third = (d00 * d21 - d01 * d20) / denom
    return xp.stack([1.0 - second - third, second, third], axis=-1)


def _closest_on_triangles(p, a, b, c, xp=np):
    """Closest point on triangle ``(a, b, c)`` to ``p``; the arrays broadcast to ``(..., 3)``.

    Ericson's algorithm. The broadcasting serves both uses: each point against
    its own candidate triangles, and a chunk of points against every face.
    ``xp`` is NumPy or CuPy, as the arrays are.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = _dot(ab, ap, xp)
    d2 = _dot(ac, ap, xp)
    bp = p - b
    d3 = _dot(ab, bp, xp)
    d4 = _dot(ac, bp, xp)
    cp = p - c
    d5 = _dot(ab, cp, xp)
    d6 = _dot(ac, cp, xp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    total = va + vb + vc
    scale = 1.0 / xp.where(total == 0, 1.0, total)
    closest = a + (vb * scale)[..., None] * ab + (vc * scale)[..., None] * ac
    closest = xp.array(xp.broadcast_to(closest, xp.broadcast(p, a).shape))

    def put(mask, value):
        xp.copyto(closest, xp.broadcast_to(value, closest.shape), where=mask[..., None])

    denom_ab = xp.where(d1 - d3 == 0, 1.0, d1 - d3)
    denom_ac = xp.where(d2 - d6 == 0, 1.0, d2 - d6)
    denom_bc = xp.where((d4 - d3) + (d5 - d6) == 0, 1.0, (d4 - d3) + (d5 - d6))
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


#: The most point-face pairs one pass of a search of every face takes: each pass holds some
#: twenty temporaries of this many triples, so the search keeps to about half a gigabyte
#: whatever the mesh's size, where 256 points a pass held ten on a mesh of 81,920 faces.
_PAIRS_PER_PASS = 1 << 20

#: The most point-centroid pairs one pass of the GPU's search for each point's nearest
#: centroid takes (two temporaries of this many doubles, a quarter of a gigabyte).
_CENTROID_PAIRS_PER_PASS = 1 << 24


def _per_pass(budget, n_faces, most):
    """How many points one pass over every face takes, within ``budget`` pairs and ``most``."""
    return max(1, min(most, budget // max(int(n_faces), 1)))


class MeshQuery:
    """Closest-point queries against a triangle mesh.

    Replaces the reference's libigl AABB tree. The face is the one carrying the
    closest point; the weights then come from the **unclamped** projection of
    the query onto that face's plane, which is what the reference's libigl
    build returns -- a point just outside a triangle keeps a slightly negative
    weight rather than being snapped to the edge.

    The search is narrowed to the ``candidates`` faces whose centroids lie
    nearest each point, found with a k-d tree, so a query costs time in
    proportion to the number of points rather than points times faces. The
    best candidate is kept only when no other face can be as near: every
    other face's centroid lies at least as far from the point as the farthest
    candidate's, and no point of a face lies farther from its centroid than
    the mesh's largest centroid-to-corner distance, so a best candidate
    nearer than the first less the second is the closest face. Any other
    point is tested against every face whose centroid lies near enough for
    the face to be as near as its best candidate, which the tree finds too,
    so the search is exact on any mesh; on the bundled grids no point needs
    it. Ties -- a closest point at a shared vertex, which every face around
    it carries bit for bit -- go to the lowest face index, as the exhaustive
    search does.
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
        # every point of a face lies within this of the face's centroid
        self._reach = float(
            max(
                np.linalg.norm(corner - centroids, axis=1).max()
                for corner in (self._a, self._b, self._c)
            )
        )
        self._on_device = None  # the tables the GPU searches, made the first time it is asked

    def _settled(self, gap, beyond):
        """Whether each best candidate, ``gap`` from its point, is surely the closest face.

        ``beyond`` is how far the nearest centroid outside the candidates can be at
        least; a face there is at least ``beyond - reach`` away. The margin keeps a
        rounding of either from settling a tie with a face outside.
        """
        return gap < beyond - self._reach - 1e-9

    def _without_tables(self):
        """A copy sharing the k-d tree and the corners, without the tables the GPU searches.

        An endpoint connectome sent to the GPU takes one, so the tables it makes live as long
        as it does: kept on the caller's search, they stayed on the GPU, some 4 MB a subject on
        ico4, as long as any connectome sharing it did.
        """
        clone = copy.copy(self)
        clone._on_device = None
        return clone

    def query(self, points):
        """``(weights, vertex_indices)``, both ``(n, 3)``."""
        weights, indices, _ = self.query_faces(points)
        return weights, indices

    def query_faces(self, points):
        """``(weights, vertex_indices, face_indices)`` for each point.

        CuPy points are located on the GPU (:meth:`_query_faces_device`), and the
        results come back there.
        """
        if _device.on_device(points):
            return self._query_faces_device(points)
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        weights = np.empty((n, 3))
        indices = np.empty((n, 3), dtype=np.int64)
        face_of = np.empty(n, dtype=np.int64)
        for start in range(0, n, self.block):
            chunk = points[start : start + self.block]
            m = chunk.shape[0]
            reached, near = self._tree.query(chunk, k=self.candidates)
            near = near.reshape(m, self.candidates)
            # every centroid outside the candidates is at least as far as the farthest in
            beyond = np.reshape(reached, (m, self.candidates))[:, -1]
            if self.candidates == self.faces.shape[0]:
                beyond = np.full(m, np.inf)  # the candidates are every face
            near.sort(axis=1)  # argmin's first minimum is then the lowest face index
            closest = _closest_on_triangles(
                chunk[:, None, :], self._a[near], self._b[near], self._c[near]
            )
            distance = ((closest - chunk[:, None, :]) ** 2).sum(-1)
            best = np.argmin(distance, axis=1)
            rows = np.arange(m)
            face = near[rows, best]
            gap = np.sqrt(distance[rows, best])
            far = ~self._settled(gap, beyond)
            if far.any():
                face[far] = self._nearest_faces(chunk[far], gap[far])

            a, b, c = self._a[face], self._b[face], self._c[face]
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            projected = chunk - ((chunk - a) * normal).sum(axis=1, keepdims=True) * normal

            face_of[start : start + self.block] = face
            indices[start : start + self.block] = self.faces[face]
            weights[start : start + self.block] = barycentric_coordinates(projected, a, b, c)
        return weights, indices, face_of

    def _query_faces_device(self, points):
        """:meth:`query_faces` on the GPU, for CuPy points: the CPU's faces and weights.

        A point's candidates are the faces whose centroids lie nearest the centroid
        nearest it, three times as many as the k-d tree's ``candidates``: on the
        bundled grids and on icospheres they hold every one of the tree's (the
        first 25 do, for 400,000 random points and the vertices:
        tests/reference/gpu_candidates_probe.py). Sorted by index, the one carrying
        the closest point wins, the lowest index on a tie. It is kept when no face
        outside can be as near, as on the CPU: a centroid outside lies at least the
        farthest candidate centroid's distance from the nearest centroid, less the
        point's own distance from that, away. Any other point is tested against
        every face. Either way the face is the closest, and the arithmetic is the
        CPU's, its sums added in NumPy's order (:func:`_dot`), so the faces and
        weights match the CPU's bit for bit.
        """
        cp = _device.cupy()
        if self._on_device is None:
            centroids = (self._a + self._b + self._c) / 3.0
            count = min(3 * self.candidates, self.faces.shape[0])
            reached, near = self._tree.query(centroids, k=count)
            near = np.sort(near.reshape(centroids.shape[0], count), axis=1)
            # how far from each centroid every centroid outside its table lies, at least
            reached = np.reshape(reached, (centroids.shape[0], count))[:, -1]
            if count == self.faces.shape[0]:
                reached = np.full(centroids.shape[0], np.inf)  # the table is every face
            half_squared = 0.5 * (centroids**2).sum(axis=1)
            self._on_device = tuple(
                cp.asarray(x)
                for x in (
                    self._a,
                    self._b,
                    self._c,
                    centroids,
                    half_squared,
                    self.faces,
                    near,
                    reached,
                )
            )
        a, b, c, centroids, half_squared, faces, near, reached = self._on_device
        points = cp.asarray(points, dtype=cp.float64)
        n = points.shape[0]
        weights = cp.empty((n, 3))
        indices = cp.empty((n, 3), dtype=cp.int64)
        face_of = cp.empty(n, dtype=cp.int64)
        for start in range(0, n, self.block):
            chunk = points[start : start + self.block]
            m = chunk.shape[0]
            nearest = cp.empty(m, dtype=cp.int64)
            step = _per_pass(_CENTROID_PAIRS_PER_PASS, centroids.shape[0], 4096)
            for first in range(0, m, step):  # |p - c|^2 / 2 less the point's own |p|^2 / 2
                part = chunk[first : first + step]
                nearest[first : first + step] = cp.argmin(half_squared - part @ centroids.T, axis=1)
            candidates = near[nearest]
            closest = _closest_on_triangles(
                chunk[:, None, :], a[candidates], b[candidates], c[candidates], xp=cp
            )
            offset = closest - chunk[:, None, :]
            gap = _dot(offset, offset, cp)
            best = cp.argmin(gap, axis=1)  # the first minimum: the lowest face index
            rows = cp.arange(m)
            face = candidates[rows, best]
            off = chunk - centroids[nearest]
            beyond = reached[nearest] - cp.sqrt(_dot(off, off, cp))
            far = ~self._settled(cp.sqrt(gap[rows, best]), beyond)
            if bool(far.any()):
                face[far] = self._exact_faces_device(chunk[far], a, b, c)

            fa, fb, fc = a[face], b[face], c[face]
            u, v = fb - fa, fc - fa
            normal = cp.stack(  # np.cross, product by product
                [
                    u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                    u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                    u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
                ],
                axis=1,
            )
            normal /= cp.sqrt(_dot(normal, normal, cp))[:, None]
            projected = chunk - _dot(chunk - fa, normal, cp)[:, None] * normal

            face_of[start : start + m] = face
            indices[start : start + m] = faces[face]
            weights[start : start + m] = barycentric_coordinates(projected, fa, fb, fc, xp=cp)
        return weights, indices, face_of

    @staticmethod
    def _exact_faces_device(points, a, b, c):
        """:meth:`_exact_faces` on the GPU, in its arithmetic."""
        cp = _device.cupy()
        out = cp.empty(points.shape[0], dtype=cp.int64)
        step = _per_pass(_PAIRS_PER_PASS, a.shape[0], 256)
        for start in range(0, points.shape[0], step):
            chunk = points[start : start + step]
            closest = _closest_on_triangles(chunk[:, None, :], a[None], b[None], c[None], xp=cp)
            offset = closest - chunk[:, None, :]
            out[start : start + step] = cp.argmin(_dot(offset, offset, cp), axis=1)
        return out

    def _nearest_faces(self, points, gap):
        """The closest face to each point whose best candidate, ``gap`` from it, is unsettled.

        A face as near as ``gap`` has its centroid within ``gap`` and the reach of the
        point, so the faces the tree finds there are all that can win; each is tested, a
        block of points at a time, and the lowest index wins a tie, as in
        :meth:`_exact_faces`. A point far off the mesh, where nearly every face is that
        near, is tested against every face instead.
        """
        out = np.empty(points.shape[0], dtype=np.int64)
        wide = gap > 8.0 * self._reach
        if wide.any():
            out[wide] = self._exact_faces(points[wide])
        for start in range(0, points.shape[0], 1024):
            todo = start + np.flatnonzero(~wide[start : start + 1024])
            if not todo.size:
                continue
            nearby = self._tree.query_ball_point(points[todo], gap[todo] + self._reach + 1e-9)
            counts = np.fromiter((len(found) for found in nearby), dtype=np.int64, count=todo.size)
            totals = np.cumsum(counts)
            begin = 0
            while begin < todo.size:  # passes of at most _PAIRS_PER_PASS pairs, a point at least
                before = int(totals[begin - 1]) if begin else 0
                stop = int(np.searchsorted(totals, before + _PAIRS_PER_PASS, side="right"))
                stop = max(stop, begin + 1)
                owner = np.repeat(np.arange(stop - begin), counts[begin:stop])
                faces = np.concatenate(
                    [np.asarray(nearby[i], dtype=np.int64) for i in range(begin, stop)]
                )
                at = points[todo[begin:stop]][owner]
                closest = _closest_on_triangles(at, self._a[faces], self._b[faces], self._c[faces])
                offset = closest - at
                order = np.lexsort((faces, _dot(offset, offset), owner))  # point, distance, index
                first = np.ones(order.size, dtype=bool)
                first[1:] = owner[order][1:] != owner[order][:-1]
                out[todo[begin:stop]] = faces[order][first]
                begin = stop
        return out

    def _exact_faces(self, points):
        """The closest face by testing every face: the search the narrowed one must agree with."""
        out = np.empty(points.shape[0], dtype=np.int64)
        step = _per_pass(_PAIRS_PER_PASS, self.faces.shape[0], 256)
        for start in range(0, points.shape[0], step):
            chunk = points[start : start + step]
            closest = _closest_on_triangles(
                chunk[:, None, :], self._a[None], self._b[None], self._c[None]
            )
            out[start : start + step] = np.argmin(
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
    CuPy values are interpolated on the GPU, the operators built here and sent
    there, unless they come already sent (``Concon`` keeps its fixed ones).
    """
    n = values.shape[0]
    if _device.on_device(values):

        def sent(coordinates):
            if _device.on_device(coordinates):
                return coordinates
            return _device.sparse_to_device(interpolation_operator(*coordinates, n), "gpu")

        halfway = _device.rows_times_dense(sent(right), values)
        return _device.rows_times_dense(sent(left), halfway.T).T
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

    ``order`` is the spherical harmonic order of the tangent basis, built with
    the corrected Legendre recurrence (:func:`tangent_basis`). On a grid clear
    of the coordinate axis its fields are the reference's to rounding and their
    divergence, ``laplacian``, is not. At a vertex on the axis the reference's
    clamp drops the phi-component of the ``m = 1`` fields, which this basis
    keeps (:func:`_harmonic_derivatives`), so there the fields differ too.
    """

    def __init__(self, vertices, faces, order: int = DEFAULT_ORDER):
        # order 0 has no tangent field at all, so a registration on it never moves
        if not (1 <= float(order) < np.inf and float(order) == round(float(order))):
            raise ValueError(
                f"order is a spherical harmonic order, a whole number of 1 or more; got {order}"
            )
        self.vertices = normalize_rows(vertices)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.order = int(order)
        self.theta, self.phi = cart_to_sphere(self.vertices)
        self.areas = voronoi_areas(self.vertices, self.faces)
        self.basis, self.laplacian = tangent_basis(self.order, self.theta, self.phi, self.areas)
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

    def pole_vertices(self, tolerance: float = AXIS_CLEARANCE) -> np.ndarray:
        """Vertices on the axis of the spherical coordinate system, or too near it.

        The Jacobian is formed in ``(theta, phi)`` coordinates and closes with
        a factor of ``sin(theta)``, which vanishes at the poles. A vertex there
        therefore gets a Jacobian of exactly zero, and a density pushed through
        the warp loses that vertex's whole row and column. Near the axis the
        central difference straddles it and is no better: 1e-5 rad from the
        axis the identity's Jacobian comes out 0.78, and a rigid rotation's,
        calibrated against it (:class:`SphericalWarp`), 1.27 instead of 1. So
        a vertex counts here when ``sin(theta)`` is at most ``tolerance``, by
        default the clearance :func:`rotate_off_poles` guarantees, at which
        a rotation's Jacobian on the bundled grid is 1 to 3e-5 at the default
        step (tests/reference/encore_axis_probe.py). The sphere
        has no distinguished axis, so all of this is an artifact of the
        parametrization rather than of the data.
        """
        return np.flatnonzero(np.sin(self.theta) <= tolerance)


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
    1 but 0.9965 on the ico4 grid (0.805 on a 42-vertex icosphere). The
    registration transports only square-root densities, renormalized at
    every trial (:meth:`Concon.evaluate_root`), but :meth:`Concon.evaluate`
    applies the final warps' Jacobians, a product of two, once at the end to
    give the aligned density, which would come out about 0.7% short of its
    mass on ico4 (35% on the 42-vertex sphere). So the scheme's value on the
    identity is computed once, at construction, and every Jacobian is
    divided by it: the identity gives exactly 1 and a small warp 1 plus its
    divergence; the reference leaves it uncalibrated. A vertex within
    :data:`AXIS_CLEARANCE` of the coordinate axis is left as it is: on the
    axis the value is zero, and closer to the axis than about a step the
    central difference straddles it, so the value measures the
    parametrization rather than the mesh -- divided out, it would turn a rigid
    rotation's Jacobian of 1 into 38 at 3e-6 rad from the axis. That is why
    :func:`align` turns such a grid clear of the axis first, and refuses one
    it cannot (:meth:`SphericalGrid.pole_vertices`).
    Everywhere else the identity's value is divided out however far from one
    it lies, since it is the mesh's chord error: 0.46 on the 12-vertex
    icosahedron. (Until 6 October 2026 a value below one half was left alone,
    which left so coarse a grid uncalibrated.) A vertex clear of the axis
    where the scheme reads nothing positive for the identity is a degenerate
    mesh, and is refused. The scheme works in ``(theta, phi)`` at each
    *image* too, so a vertex the warp carries to within about 1e-4 rad of the
    axis reads a Jacobian too small: 0.9967 for a rotation at 1e-4, 0.79 at
    1e-5, 0 on the axis (``tests/reference/encore_axis_probe.py``). Clearing
    the grid cannot prevent that, but a warp of the sizes registration
    produces puts an image that close only rarely.
    """

    def __init__(self, grid: SphericalGrid, delta: float = DEFAULT_DELTA):
        self.grid = grid
        self.faces = grid.faces
        self.base = grid.vertices
        self.vertices = grid.vertices.copy()
        self.jacobian = np.ones(grid.n_vertices)
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
        # (class docstring) except within AXIS_CLEARANCE of the coordinate axis.
        identity = self._raw_jacobian()
        clear = np.sin(grid.theta) >= AXIS_CLEARANCE
        unmeasured = clear & ~(np.isfinite(identity) & (identity > 0))
        if unmeasured.any():
            j = int(np.flatnonzero(unmeasured)[0])
            raise ValueError(
                f"the Jacobian's central difference reads {identity[j]:.3g} for the identity "
                f"at vertex {j}, so it cannot measure the mesh there: the grid is degenerate"
            )
        self._calibration = np.where(clear, identity, 1.0)

    def compose(self, displacement) -> SphericalWarp:
        """Compose with a further displacement given in the tangent basis.

        Each grid vertex is moved by its displacement first and then carried
        through the warp so far, ``phi(exp_x(D(x)))``, the warp interpolated
        between the images of the triangle the moved point falls in, with the
        weights of the ray through it, which carry the identity and a rotation
        exactly: the composition the registration's gradient is taken for, since it
        differentiates the image ``root o phi`` at the grid vertices. The
        reference moves each image instead, ``phi(x)`` by the displacement
        interpolated where it lies, which applies the step after the warp. The
        two agree at the identity and part as the warp grows: after a turn of
        30 degrees the cost fell at 0.79 of the rate the gradient predicts
        with the reference's order, and at 0.99 with this one
        (``test_a_step_is_composed_as_the_gradient_assumes``); on a known
        20-degree turn of a real subject ENCORE ended 0.56 degrees from the
        truth with this order and 1.30 with the reference's (PORTING.md item
        21).
        """
        displacement = np.asarray(displacement, dtype=np.float64)
        tangent = self._e1 * displacement[:, :1] + self._e2 * displacement[:, 1:2]
        moved = normalize_rows(sphere_exp_map(self.base, tangent))
        _, indices = self._query.query(moved)
        # Weights along the ray through each moved point, not those of its closest point
        # on the flat face: then the identity, and any rotation, carry it exactly.
        corners = np.swapaxes(self.base[indices], 1, 2)  # one corner a column
        along = np.linalg.solve(corners, moved[:, :, None])[:, :, 0]
        weights = along / along.sum(axis=1, keepdims=True)
        vertices = (weights[:, :, None] * self.vertices[indices]).sum(axis=1)

        clone = object.__new__(SphericalWarp)
        clone.__dict__.update(self.__dict__)
        clone.vertices = normalize_rows(vertices)
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
        """The Jacobian, divided by the scheme's value on the identity."""
        return self._raw_jacobian() / self._calibration


class Concon:
    """A connectivity density over the product of two spherical meshes.

    :meth:`evaluate_root` zeroes the diagonal of a transported square-root
    density before it normalizes; ``Concon.m`` normalizes first, so its result
    comes back short of unit norm by the mass interpolation put on the
    diagonal: 0.02 to 0.2% on ico4 for realistic warps.

    With ``device="gpu"`` it works on the GPU: it takes NumPy or CuPy densities
    and returns CuPy ones, and ``area_product`` is there too. The grids and the
    warps stay on the CPU. :class:`Encore` is what hands results back.
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        delta: float = DEFAULT_DELTA,
        derivative: str = "difference",
        device: str = "cpu",
    ):
        if derivative not in ("analytic", "difference"):
            raise ValueError(f"derivative must be 'analytic' or 'difference', got {derivative!r}")
        if not 0 < float(delta) < np.inf:
            raise ValueError(f"delta is a central-difference step, finite and above 0; got {delta}")
        self.device = _device.check(device)
        self._xp = _device.array_module(self.device)
        self.lh_grid = lh_grid
        self.rh_grid = rh_grid
        self.derivative_method = derivative
        self._operators = None
        self.step = 2 * delta
        self.n_per_hemi = lh_grid.n_vertices
        self.areas = np.concatenate([lh_grid.areas, rh_grid.areas])
        self.area_product = _device.to_device(np.outer(self.areas, self.areas), self.device)

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
        if self.device == "gpu":  # the fixed operators, sent to the GPU once
            n = self.areas.size
            self._grid = _device.sparse_to_device(interpolation_operator(*self._grid, n), "gpu")
            self._offsets = {
                name: _device.sparse_to_device(interpolation_operator(*coordinates, n), "gpu")
                for name, coordinates in self._offsets.items()
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
        values = _device.to_device(values, self.device)
        if self.derivative_method == "analytic":
            from scipy import sparse

            if self._operators is None:
                lh_first, lh_second = self.lh_grid.gradients
                rh_first, rh_second = self.rh_grid.gradients
                self._operators = tuple(
                    _device.sparse_to_device(operator, self.device)
                    for operator in (
                        sparse.block_diag((lh_first, rh_first), format="csr"),
                        sparse.block_diag((lh_second, rh_second), format="csr"),
                    )
                )
            first, second = self._operators
            if self.device == "gpu":
                product = _device.rows_times_dense
                return product(first, values), product(second, values)
            return np.asarray(first @ values), np.asarray(second @ values)

        def at(key):
            return interpolate_product(self._grid, self._offsets[key], values)

        return (at("px") - at("mx")) / self.step, (at("py") - at("my")) / self.step

    def evaluate(self, values, lh_warp: SphericalWarp, rh_warp: SphericalWarp):
        """Push a density through a pair of warps."""
        xp = self._xp
        values = _device.to_device(values, self.device)
        jacobian = xp.asarray(np.concatenate([lh_warp.jacobian, rh_warp.jacobian]))
        coords = self._coordinates(lh_warp.vertices, rh_warp.vertices)
        warped = interpolate_product(coords, coords, values) * xp.outer(jacobian, jacobian)
        warped = xp.maximum((warped + warped.T) / 2, 0)
        xp.fill_diagonal(warped, 0.0)
        return warped

    def evaluate_root(self, root, lh_warp: SphericalWarp, rh_warp: SphericalWarp):
        """Push a square-root density through, keeping it a unit vector.

        Interpolation leaks mass onto the diagonal, which the density does not
        carry, so the diagonal is zeroed before the result is normalized. The
        reference (``Concon.m`` lines 62-63) normalizes first, and its result
        is short of unit norm by that leaked mass. A density whose every entry lands on the
        diagonal comes back as zeros rather than as a division by zero.
        """
        xp = self._xp
        root = _device.to_device(root, self.device)
        jacobian = xp.asarray(np.sqrt(np.concatenate([lh_warp.jacobian, rh_warp.jacobian])))
        coords = self._coordinates(lh_warp.vertices, rh_warp.vertices)
        warped = interpolate_product(coords, coords, root) * xp.outer(jacobian, jacobian)
        warped = xp.maximum((warped + warped.T) / 2, 0)
        xp.fill_diagonal(warped, 0.0)
        norm = xp.sqrt((warped**2 * self.area_product).sum())
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
    frame. With ``grids`` supplied they are what :func:`align` was told
    (``grid_rotations=``) and what it applied itself to a grid with vertices
    beside the axis -- identities for a grid clear of it and given as is.
    """

    template: np.ndarray
    warps: list = field(default_factory=list)
    aligned: list = field(default_factory=list)
    costs: list = field(default_factory=list)
    grid_rotations: tuple = field(default_factory=lambda: (np.eye(3), np.eye(3)))
    traces: list = field(default_factory=list)
    """Per subject, the cost before registration and after every accepted step."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Alignment of {len(self.warps)} subjects on {self.template.shape[0]} vertices>"


def _scalar(value):
    """A sum as a number on the CPU: NumPy's as it is, CuPy's fetched."""
    return float(value) if _device.on_device(value) else value


def _without_diagonal(density, xp=np):
    """A float64 copy of ``density`` with its diagonal zeroed, in ``xp``; the input is unchanged."""
    density = xp.array(density, dtype=xp.float64)
    xp.fill_diagonal(density, 0.0)
    return density


def _whole(value) -> bool:
    """Whether ``value`` is a whole number, 0 or more: a count of iterations or retries."""
    return 0 <= float(value) < np.inf and float(value) == round(float(value))


#: How far a template given in single precision may sit from unit mass and still be taken,
#: scaled to it in float64: float32 holds the unit mass of a 5,124 x 5,124 square-root density
#: only to about 1e-6 (1.7e-6 was seen after normalizing in float32), where the check that
#: refuses a density passed in its place allows 1e-6. A density so passed sits far further off.
SINGLE_PRECISION_MASS = 1e-4


def _single_precision(array) -> bool:
    """Whether ``array`` holds floats less precise than float64, as a float32 copy does."""
    dtype = getattr(array, "dtype", None)
    return dtype is not None and dtype.kind == "f" and dtype.itemsize < 8


def _density_problem(density) -> str | None:
    """Why a square array is not a connectivity density :func:`align` can use, or ``None``.

    Its diagonal is no part of it (:meth:`Encore.root`), so the mass that
    counts is the mass off the diagonal. A CuPy array is checked on the GPU.
    """
    xp = _device.module_of(density)
    if not xp.isfinite(density).all() or density.min() < 0:
        return "is not a nonnegative density"
    # Both triangles: the gradient reads each vertex's row, and a matrix stored as one
    # triangle gives half the rows nothing, so it never registers -- silently.
    if xp.abs(density - density.T).max() > 1e-8 * xp.abs(density).max():
        return (
            "is not symmetric; a connectivity density holds every pair in both triangles "
            "(one stored as a single triangle D needs D + D.T - diag(D) first)"
        )
    # the entries are nonnegative, so mass off the diagonal is a nonzero entry there
    if xp.count_nonzero(density) == xp.count_nonzero(density.diagonal()):
        return "has no mass off its diagonal"
    return None


class Encore:
    """Template estimation and registration on a pair of spherical meshes.

    A step that folds a face is refused (:meth:`register`), and so is a setting
    that cannot register: a step or difference of 0, a negative threshold or
    count, NaN.

    With ``device="gpu"`` the densities are held and transported on the GPU
    (:class:`Concon`) and the warps stay on the CPU; :meth:`root`,
    :meth:`template` and :meth:`register` take NumPy or CuPy arrays and return
    NumPy arrays and floats, as on the CPU: the root and the template to
    rounding, a registration as closely as the CPU's own does when an input
    moves by one unit in the last place (PORTING.md item 26).
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
        device: str = "cpu",
    ):
        self.lh_grid = lh_grid
        self.rh_grid = rh_grid
        self.concon = Concon(lh_grid, rh_grid, delta, derivative=derivative, device=device)
        self.device = self.concon.device
        self._xp = self.concon._xp
        # the areas' products beside the densities, and on the CPU for whoever reads them
        self._area_product = self.concon.area_product
        self.area_product = _device.to_host(self._area_product)
        # NaN fails every one of these; each setting refused returned unmoved warps, or
        # took every step whatever it did to the cost, without a word
        for name, value, ok, needs in (
            ("step", step, 0 < float(step) < np.inf, "finite and above 0"),
            ("max_iterations", max_iterations, _whole(max_iterations), "a whole number, 0 or more"),
            ("threshold", threshold, 0 <= float(threshold) < np.inf, "a finite fall, 0 or more"),
            ("backtracks", backtracks, _whole(backtracks), "a whole number, 0 or more"),
        ):
            if not ok:
                raise ValueError(f"{name} must be {needs}; got {value}")
        self.step = step
        self.max_iterations = int(max_iterations)
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
        could ever be accepted. A density with nothing off its diagonal then
        has no mass to normalize and is refused. The input is not modified.
        """
        return _device.to_host(self._root(density))

    def _root(self, density):
        """:meth:`root`, left on this engine's device."""
        density = _without_diagonal(density, self._xp)
        n = self._area_product.shape[0]
        if density.shape != (n, n):  # a row or column would broadcast against the areas
            raise ValueError(f"a density on these grids is {n} x {n}; got {density.shape}")
        mass = float((density * self._area_product).sum())
        if not np.isfinite(mass):
            raise ValueError("the density is not finite")
        if mass <= 0:
            raise ValueError("the density has no mass off its diagonal")
        return self._xp.sqrt(density / mass)

    def template(self, densities, iterations: int = 100):
        """Karcher median of the square-root densities.

        Weiszfeld steps from the subject nearest their mean, each a fifth of the
        mean direction to the subjects, until that direction is shorter than
        0.005 or ``iterations`` steps are taken. The reference stops after
        ten, short of the median; with up to a hundred the rule ends them
        first (PORTING.md item 25). The densities are read one at a time, so a
        generator never holds them all at once. On the GPU as many square-root
        densities stay there as fill less than half its free memory; the rest
        wait on the CPU and are sent across at each step.
        """
        if not _whole(iterations):
            raise ValueError(f"iterations must be a whole number, 0 or more; got {iterations}")
        xp = self._xp
        roots = _device.Held(self.device)
        for density in densities:
            roots.append(self._root(density))
        count = len(roots)
        if not count:
            raise ValueError("a template needs at least one density")
        # their mean, added in the order NumPy's mean adds them
        mean = roots[0].copy()
        for i in range(1, count):
            mean += roots[i]
        mean /= count
        distances = [
            float(((roots[i] - mean) ** 2 * self._area_product).sum()) for i in range(count)
        ]
        current = roots[int(np.argmin(distances))].copy()
        del mean

        for _ in range(int(iterations)):
            # Summed as they are computed: a second cohort-sized array would cost
            # another 210 MB per subject on the ico4 grid.
            directions = xp.zeros_like(current)
            distance = np.zeros(count)
            for i in range(count):
                root = roots[i]
                inner = float((root * current * self._area_product).sum())
                if 1 - abs(inner) < 1e-14:
                    inner = np.sign(inner)
                distance[i] = np.arccos(np.clip(inner, -1.0, 1.0))
                if distance[i] > 0:
                    cosine, sine = float(np.cos(distance[i])), float(np.sin(distance[i]))
                    directions += (root - cosine * current) / sine

            moving = distance > 0
            if not moving.any():
                break
            mean_direction = directions / float((1.0 / distance[moving]).sum())
            size = float(xp.sqrt((mean_direction**2 * self._area_product).sum()))
            if size == 0:
                break
            cosine, sine = float(np.cos(0.2 * size)), float(np.sin(0.2 * size))
            current = cosine * current + sine * (mean_direction / size)
            current = current / xp.sqrt((current**2 * self._area_product).sum())
            if size < 0.005:
                break
        return _device.to_host(current)

    def _descent(self, image, residual):
        """The cost's steepest descent at ``image``, in basis coefficients: ``(left, right)``.

        The cost is ``sum (fixed - image)^2`` weighted by the area product, with
        ``residual = fixed - image``, both square-root densities. A hemisphere's
        coefficients are minus the cost's derivative along each of its basis fields, for
        a step composed as ``phi(exp_x(D(x)))``, the warp taken first (PORTING.md item 21).
        """
        d_e1, d_e2 = self.concon.derivative(image)
        weighted = residual * self._area_product
        a = (weighted * (2 * d_e1)).sum(axis=1)
        b = (weighted * (2 * d_e2)).sum(axis=1)
        c = (weighted * image).sum(axis=1)
        # vectors now, brought to the basis on the CPU
        a, b, c = (_device.to_host(v) for v in (a, b, c))
        n = self.lh_grid.n_vertices
        return tuple(
            2
            * (
                a[rows] @ grid.basis[:, :, 0]
                + b[rows] @ grid.basis[:, :, 1]
                + c[rows] @ grid.laplacian
            )
            for grid, rows in (
                (self.lh_grid, slice(0, n)),
                (self.rh_grid, slice(n, n + self.rh_grid.n_vertices)),
            )
        )

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
        lh_warp = SphericalWarp(self.lh_grid, self.delta)
        rh_warp = SphericalWarp(self.rh_grid, self.delta)

        xp = self._xp
        if target_is_root:
            fixed = _device.to_device(target, self.device, np.float64)
        else:
            fixed = self._root(target)
        # Not part of the density (see root); the final evaluate must not see
        # it either, or it would interpolate that mass off the diagonal.
        moving = _without_diagonal(moving, xp)
        source = self._root(moving)

        image = source
        residual = fixed - image
        last_cost = _scalar((residual**2 * self._area_product).sum())
        cost = last_cost
        last = (lh_warp, rh_warp)

        scale = 1.0  # of self.step: halved when a step fails, doubled back after a success
        for iteration in range(1, self.max_iterations + 1):
            # each hemisphere's step of length step in coefficient space, as the reference
            # takes it: the two together are not quite the steepest descent when their
            # gradients differ in size
            moves = []
            for grid, descent in zip(
                (self.lh_grid, self.rh_grid), self._descent(image, residual), strict=True
            ):
                direction = (descent[None, :, None] * grid.basis).sum(axis=1)
                moves.append(direction * (self.step / (np.linalg.norm(descent) + 1e-15)))

            # The reference takes one step of fixed length and stops the moment
            # it fails to lower the cost -- on a small deformation, the very
            # first step. Halve the length a few times before giving up;
            # backtracks=0 stops at the first failed step, as the reference
            # does (PORTING.md item 4).
            accepted = False
            for _attempt in range(self.backtracks + 1):
                trial = (last[0].compose(scale * moves[0]), last[1].compose(scale * moves[1]))
                # A warp that folds a face is not a diffeomorphism, whatever
                # it does to the cost, and is refused like a cost increase.
                # The reference accepts it: its Jacobian is an absolute value,
                # so a flipped face still reads as positive. It only ever folds
                # at step lengths far above its default.
                if trial[0].folds() or trial[1].folds():
                    scale *= 0.5
                    continue
                trial_image = self.concon.evaluate_root(source, *trial)
                trial_residual = fixed - trial_image
                trial_cost = _scalar((trial_residual**2 * self._area_product).sum())
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
        return _device.to_host(result), lh_warp, rh_warp, cost


def pole_rotation(vertices, tolerance: float = AXIS_CLEARANCE) -> np.ndarray:
    """The rotation :func:`rotate_off_poles` applies, as a ``3 x 3`` matrix.

    ``rotate_off_poles(v) == v @ pole_rotation(v).T``, so ``rotated @ R`` maps
    points back into the mesh's original frame.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    for angle in (0.0, 0.3, 0.7, 1.1, 1.7):
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        # measured as SphericalGrid.pole_vertices measures it, on the unit sphere
        theta, _ = cart_to_sphere(normalize_rows(vertices @ rotation.T))
        if np.sin(theta).min() > tolerance:
            return rotation
    raise RuntimeError("could not rotate the mesh clear of the coordinate poles")


def rotate_off_poles(vertices, tolerance: float = AXIS_CLEARANCE):
    """Rotate a spherical mesh so that every vertex clears the coordinate axis.

    The ico4 grid puts four vertices exactly at the poles, where the
    ``(theta, phi)`` parametrization is singular and the Jacobian collapses to
    zero, and within about a finite-difference step of them it is unreliable
    (:meth:`SphericalGrid.pole_vertices`). After the rotation every vertex has
    ``sin(theta)`` above ``tolerance``, by default :data:`AXIS_CLEARANCE`, the
    clearance :func:`align` checks. A rotation is a change of coordinates and
    nothing else: areas, distances and the connectivity are untouched, and
    the vertex order is unchanged. The angles below are irrational multiples
    of the icosahedral symmetry, so none of them can bring a vertex back onto
    the axis.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices @ pole_rotation(vertices, tolerance).T


def _hemisphere_grids(order, rotate: bool = True, return_rotations: bool = False):
    """The bundled ico4 sphere, split into two hemispheres.

    With ``return_rotations`` the pair of ``3 x 3`` rotations applied to the two
    hemispheres comes back as well (identities when ``rotate`` is false).
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
        SphericalGrid(lh_vertices, left, order),
        SphericalGrid(rh_vertices, right, order),
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
    template_iterations: int = 100,
    template=None,
    grids=None,
    derivative: str = "difference",
    backtracks: int = 4,
    verbose: bool = False,
    grid_rotations=None,
    device: str = "cpu",
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
        Spherical harmonic order of the tangent basis of the bundled grids. Grids
        given through ``grids`` keep the order they were built with.
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
        mass in the area inner product, with a zero diagonal, as
        :attr:`Alignment.template` is, registers onto that: an earlier run's
        template, to bring a new subject onto it without re-estimating. An
        array that is not (a density passed where its root belongs, say) is
        refused. With a template given, one connectome is enough.
    template_iterations
        Weiszfeld steps toward the median when the template is estimated, at
        most; they stop once a step's mean direction falls below 0.005, as
        ConSEAL's do. The reference takes ten, which on the eleven released
        subjects leave the template 0.071 rad from the median, a sixth of a
        subject's distance from it; the rule stops them after 25 steps,
        0.005 rad from it (PORTING.md item 25).
    grids
        ``(lh_grid, rh_grid)`` to align on, as :class:`SphericalGrid`. Defaults
        to the bundled ico4 sphere, split at the hemisphere boundary, rotated
        clear of the coordinate axis. A supplied grid with vertices on or
        beside the axis is rotated clear of it here too
        (:func:`pole_rotation`), and the rotation is recorded in every warp.
    grid_rotations
        With ``grids``: the rotations ``R`` already applied to their vertices
        (``vertices @ R.T``), one per hemisphere, so that the warps record the
        frame they are in -- :func:`sbci.templates.migrate_warp` places a warp
        by it, and a rotation it does not know (the bundled sphere turned off
        its poles by hand, say) puts the warp that far from where it belongs.
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
    device
        ``"cpu"``, the default, or ``"gpu"``: hold and transport the densities
        on a CUDA GPU through CuPy, ``pip install 'sbci[gpu]'`` (:class:`Encore`).
        The result is the same object on the CPU. Its template agrees with the
        CPU's to rounding, and each registration as closely as the CPU's own
        does when an input moves by one unit in the last place (PORTING.md item
        26).

    Returns
    -------
    :class:`Alignment`, with the template, one :class:`Warp` per subject, the
    registered connectomes and the final costs.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    _device.check(device)
    if not _whole(template_iterations):
        raise ValueError(
            f"template_iterations must be a whole number, 0 or more; got {template_iterations}"
        )
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
        else np.asarray(_device.to_host(cc), dtype=np.float64)  # NumPy's, or CuPy's fetched
        for cc in cc_list
    ]
    shapes = {d.shape for d in densities}
    if len(shapes) != 1:
        raise ValueError(f"connectomes are on different grids: {sorted(shapes)}")
    shape = densities[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"connectomes are {shape}, not square matrices")
    for index, density in enumerate(densities):
        problem = _density_problem(density)
        if problem:
            raise ValueError(f"connectome {index} {problem}")

    if grids is None:
        if grid_rotations is not None:
            raise ValueError("grid_rotations describe supplied grids; pass grids= with them")
        (lh_grid, rh_grid), rotations = _hemisphere_grids(order, return_rotations=True)
    else:
        given = (None, None) if grid_rotations is None else tuple(grid_rotations)
        if len(given) != 2:
            raise ValueError("grid_rotations takes one rotation per hemisphere, (left, right)")
        placed = []
        for name, grid, rotation in zip(("left", "right"), grids, given, strict=True):
            rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
            if (
                rotation.shape != (3, 3)
                or not np.allclose(rotation @ rotation.T, np.eye(3))
                or np.linalg.det(rotation) <= 0
            ):
                raise ValueError(
                    "each of grid_rotations must be a 3 x 3 rotation: orthonormal, with "
                    "determinant +1 (a reflection turns the mesh inside out)"
                )
            poles = grid.pole_vertices()
            if poles.size:
                # Turned clear of the axis here, as the bundled grid is, and recorded:
                # turned by hand, its warps came back in a frame nothing recorded.
                try:
                    turn = pole_rotation(grid.vertices)
                except RuntimeError as exc:
                    raise ValueError(
                        f"the {name} grid has {poles.size} vertices on or within "
                        f"{AXIS_CLEARANCE:g} of the coordinate axis ({poles[:4].tolist()}...), "
                        "where the Jacobian is zero or unreliable, and no turn pole_rotation "
                        "tries clears them all"
                    ) from exc
                grid = SphericalGrid(grid.vertices @ turn.T, grid.faces, grid.order)
                rotation = turn @ rotation
            placed.append((grid, rotation))
        (lh_grid, lh_rotation), (rh_grid, rh_rotation) = placed
        rotations = (lh_rotation, rh_rotation)
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
        device=device,
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
        problem = _density_problem(fixed)
        if problem:
            raise ValueError(f"the template connectome {problem}")
        template = encore.root(fixed)
    single = _single_precision(template)
    template = np.asarray(_device.to_host(template), dtype=np.float64)
    if template.shape != (expected, expected):
        raise ValueError(
            f"template is {template.shape}, expected a {expected} x {expected} square-root density"
        )
    if not np.isfinite(template).all():
        raise ValueError("the template holds non-finite values; every cost would be NaN")
    if np.any(np.diagonal(template)):
        raise ValueError(
            "the template has a nonzero diagonal, where a square-root density has none "
            "(Encore.root zeroes it); template= takes a square-root density"
        )
    problem = _density_problem(template)
    if problem:
        raise ValueError(f"the template {problem}; template= takes a square-root density")
    mass = float((template**2 * encore.area_product).sum())
    if single and abs(mass - 1.0) <= SINGLE_PRECISION_MASS:  # its precision, not another object
        template = template / np.sqrt(mass)
        mass = float((template**2 * encore.area_product).sum())
    if abs(mass - 1.0) > 1e-6:
        raise ValueError(
            f"the template's squares carry mass {mass:.6g}, not 1: template= takes a "
            "square-root density of unit mass, as Alignment.template and Encore.root give, "
            "not a density"
        )

    result = Alignment(template=template, grid_rotations=rotations)
    target = _device.to_device(template, device)  # sent once; on the CPU, the template itself
    for index, density in enumerate(densities):
        if verbose:
            print(f"registering subject {index + 1} of {len(densities)}")
        start = ((target - encore._root(density)) ** 2 * encore._area_product).sum()
        trace = [float(start)]

        def record(_iteration, value, trace=trace):
            trace.append(float(value))

        aligned, lh_warp, rh_warp, cost = encore.register(
            target, density, target_is_root=True, verbose=verbose, callback=record
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
