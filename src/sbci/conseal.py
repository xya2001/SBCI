"""ConSEAL: connectivity-informed streamline endpoint alignment.

Xiang, Cole and Zhang, *ConSEAL: Connectivity-Informed Streamline Endpoint
Alignment for Diffeomorphic Cortical Registration* (arXiv:2605.16742). Ported
from the public MATLAB in ``github.com/MartyCole/Encore`` at commit
``27e4e7d`` -- ``concons/SConcon.m``, ``core/Concon.m``, ``core/Encore.m``,
``core/SphericalWarp.m``, ``core/SphericalGrid.m``,
``kernels/SphericalHeatKernel.m``, ``mexfiles/*.cpp`` and ``utils/`` -- after
checking the author's own copies on Longleaf (PORTING.md item 7 lists them).
The public entry point is :func:`endpoints_align`.

How it differs from :mod:`sbci.alignment` (ENCORE)
--------------------------------------------------
ENCORE warps a *discretized density*: every iteration interpolates the moving
connectome onto the warped grid, and interpolation error accumulates. ConSEAL
warps the *endpoints*. Each streamline's two endpoints are spread onto the
three vertices of the triangles they fall in (an adjacency matrix ``A``),
smoothed with a row-normalized heat kernel as ``F = K^T A K``, square-rooted
to ``Q``, and compared with the fixed subject's ``Q``. The gradient of that
mismatch gives a tangent displacement, a stationary velocity field absorbs it,
the endpoints ride along with the warped vertices, and ``F`` is recomputed
from where they now sit. Nothing is ever resampled.

What was found in the reference while porting
---------------------------------------------
The reference is research code, and this port reproduces its behaviour --
including the shortcuts below -- when ``strict_upstream=True``. By default the
first four are corrected. PORTING.md item 7 has the measurements.

1. **The gradient contains a term evaluated in the wrong tangent frame.**
   ``Concon.evaluate`` forms ``Dx = dK A K^T`` and then symmetrizes it,
   ``Dx + Dx.'``, before projecting every row onto that row's frame
   ``(e1, e2)``. The transposed term is the derivative of ``F(a, c)`` with
   respect to the *other* vertex ``c``, a vector tangent at ``c``, dotted with
   a frame at ``a``. ``Encore.compute_gradient`` already accounts for the
   second slot through its factor of two, exactly as ENCORE does with its
   single-slot derivative, so the extra term is spurious.
2. **The kernel derivative is of the wrong product.** ``SConcon.evaluate``
   returns ``K~^T A K~`` with ``K~`` the row-normalized kernel (each *source*
   vertex spreads unit mass), but ``SphericalHeatKernel`` differentiates
   ``K~`` with respect to its *row* vertex, normalization included, and
   ``Concon.evaluate`` contracts it as ``dK A K~^T`` -- the derivative of
   ``K~ A K~^T``, whose normalization sits on the evaluation side. The two
   agree only where the kernel's row sums are constant.
3. **A rejected warp step still pollutes the velocity field.**
   ``SphericalWarp.compose`` refuses an update that would fold a triangle, but
   has already added the displacement (and smoothed it) into the stationary
   velocity field, so the field and the vertex positions disagree from then
   on.
4. **A cost increase is accepted as convergence.** ``Encore.register`` stops
   when the cost falls by less than the threshold, which includes rising; the
   step that made it rise has already been composed and is what is returned,
   and ``Final Cost`` prints the previous iteration's value.
5. **The cost is a plain sum over vertex pairs, not the paper's integral.**
   ``Encore`` builds the Voronoi integration matrix ``A`` and never uses it.
   The gradient it computes is the consistent gradient of that plain sum, so
   this is a discretization choice rather than a bug -- on an icosphere the
   two differ by the spread of vertex areas. ``area_weighted=True`` weights
   both the cost and its gradient with the areas.
6. **Triangle indices are ``int16``**, which overflows once both hemispheres
   exceed 32,767 vertices -- ico6 and finer. The author's ``Encore-main`` copy
   already changes this to ``int32``; this port uses ``int64``.
7. **The leave-one-out bandwidth score removes an approximate self-term**
   (no barycentric weights, ``1/(M-1)`` rather than the ``1/(2N)`` of the
   symmetrized adjacency). Only :meth:`HeatKernelBuilder.cross_validate` is
   affected, and the published pipeline fixes ``sigma = 0.005`` anyway.
8. **The paper says no explicit regularization**; the public code smooths the
   velocity field by 5% of its cotangent Laplacian at every step and clamps
   the largest displacement to 0.2. The author's research fork (``Encore/``
   on Longleaf, the lineage of the paper's experiments) has neither, uses a
   step of 0.1 and a threshold of 1e-6 as the paper states, differentiates
   ``Q`` by ENCORE's central differences instead of the analytic kernel
   derivative, and composes warps directly rather than through a stationary
   velocity field. This module ports the public code; ``viscosity=0`` and
   ``step_clamp=inf`` recover the fork's update rule.
9. ``get_template`` sizes the cohort with ``size(Fs, 1)``, so a row cell
   array silently registers one subject, and it takes ``acos`` of an inner
   product that rounding can push past one, which MATLAB returns complex.
10. The shipped scripts do not run: ``README.md``, ``simulation.m`` and
    ``real_data_registration.m`` construct the abstract ``Concon``,
    ``random_diffeomorphism.m`` calls a ``compose_warp`` that does not exist,
    and ``simulation.m`` copies ``lh_warp`` where it means ``rh_warp``.
11. Kernels, frames and the adjacency accumulate in single precision. This
    port is float64 throughout.
12. **The cotangent Laplacian weights each edge with an adjacent angle.**
    ``cot_matrix`` puts the cotangent of the angle at ``i2`` on the edge
    ``(i2, i3)``; the cotangent formula weights an edge by the angle *opposite*
    it. The result is still a symmetric graph Laplacian that annihilates
    constants, so the 5% smoothing still smooths -- with the weights around the
    twelve five-valent vertices swapped (0.33 for 0.73).
13. **The square root's chain rule divides by a floor where the density was
    clamped.** ``Q_transform`` scales the derivative by ``1 / (2 max(Q, 1e-15))``;
    where ``F`` was negative -- the unclamped barycentric weights allow that --
    and clamped to zero, the derivative of ``sqrt(max(F, 0))`` is zero but the
    reference multiplies a nonzero ``dF`` by ``5e14``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .alignment import (
    MeshQuery,
    SphericalGrid,
    normalize_rows,
    sparse_times_dense,
    sphere_exp_map,
    sphere_log_map,
    voronoi_areas,
)
from .errors import MissingDataError

DEFAULT_SIGMA = 0.005
"""Kernel bandwidth the published pipeline uses for every subject."""

DEFAULT_KERNEL_DEGREE = 30
"""Harmonic degree the reference truncates the heat kernel at."""

DEFAULT_WARP_ORDER = 15
"""Harmonic order of the velocity field's tangent basis (``Encore(..., 15, ...)``)."""

DEFAULT_DELTA = 0.05
"""Gradient-descent step size (``Encore(..., 0.05, ...)``)."""

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_THRESHOLD = 1e-4
"""Stop when the cost falls by less than this between iterations."""

STEP_CLAMP = 0.2
"""Largest per-vertex displacement allowed in one step (``compute_step_size``)."""

VISCOSITY = 0.05
"""Fraction of the cotangent Laplacian removed from the velocity field each step."""

SQUARINGS = 6
"""Scaling-and-squaring steps in the exponential map."""

KERNEL_EPSILON = 1e-3
"""Where the heat kernel is cut off, on the product ``K(x) K(1)``."""


# --- closest-triangle queries ---------------------------------------------------


class FastMeshQuery(MeshQuery):
    """:class:`sbci.alignment.MeshQuery` whose ``query`` also returns the face index.

    The k-d-tree search itself now lives in :class:`~sbci.alignment.MeshQuery`,
    shared with ENCORE; this subclass only keeps the three-value ``query`` that
    the endpoint code and the tests were written against.
    """

    def query(self, points):
        """``(weights, vertex_indices, face_indices)`` for each point."""
        return self.query_faces(points)


# --- the endpoint connectome (SConcon) --------------------------------------


class EndpointConnectome:
    """A structural connectome held as streamline endpoints on two spheres.

    Port of ``SConcon``. Each endpoint is kept as three barycentric weights
    and the three (whole-grid) vertex indices of the triangle it lies in. The
    locations the object was built with are kept beside the current ones, so
    that every warp is applied from the original positions rather than
    accumulating relocation error (``orig_st_points`` in the reference).

    Build one with :meth:`from_endpoints` (from the barycentric data an SBCI
    file stores) or :meth:`from_points` (from unit-sphere coordinates in the
    grids' frame).
    """

    def __init__(
        self, lh_grid: SphericalGrid, rh_grid: SphericalGrid, hemisphere_in, hemisphere_out
    ):
        """Set up an empty connectome; use :meth:`from_points` or :meth:`from_endpoints`."""
        self.lh_grid = lh_grid
        self.rh_grid = rh_grid
        self.n_left = lh_grid.n_vertices
        self.n_vertices = lh_grid.n_vertices + rh_grid.n_vertices
        self.hemisphere_in = np.asarray(hemisphere_in).astype(np.int8).ravel()
        self.hemisphere_out = np.asarray(hemisphere_out).astype(np.int8).ravel()
        if self.hemisphere_in.shape != self.hemisphere_out.shape:
            raise ValueError("hemisphere flags of the two endpoints differ in length")
        for name, flags in (
            ("hemisphere_in", self.hemisphere_in),
            ("hemisphere_out", self.hemisphere_out),
        ):
            if flags.size and not np.isin(flags, (0, 1)).all():
                raise ValueError(f"{name} must be 0 (left) or 1 (right)")
        self.n_streamlines = int(self.hemisphere_in.size)
        self.e1 = np.vstack([lh_grid.e1, rh_grid.e1])
        self.e2 = np.vstack([lh_grid.e2, rh_grid.e2])
        self._query = {
            0: FastMeshQuery(lh_grid.vertices, lh_grid.faces),
            1: FastMeshQuery(rh_grid.vertices, rh_grid.faces),
        }
        n = self.n_streamlines
        self.weights_in = np.zeros((n, 3))
        self.weights_out = np.zeros((n, 3))
        self.index_in = np.zeros((n, 3), dtype=np.int64)
        self.index_out = np.zeros((n, 3), dtype=np.int64)
        self.face_in = np.zeros(n, dtype=np.int64)
        self.face_out = np.zeros(n, dtype=np.int64)
        self.commit()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_points(cls, lh_grid, rh_grid, points_in, points_out, hemisphere_in, hemisphere_out):
        """Locate unit-sphere endpoint coordinates on the grids (``SConcon``'s constructor).

        The coordinates must be in the grids' own frame; the bundled ico4
        sphere is not rotated for this module, so positions rebuilt by
        :func:`sbci.smoothing.endpoint_positions` qualify.
        """
        self = cls(lh_grid, rh_grid, hemisphere_in, hemisphere_out)
        points_in = normalize_rows(np.asarray(points_in, dtype=np.float64))
        points_out = normalize_rows(np.asarray(points_out, dtype=np.float64))
        expected = (self.n_streamlines, 3)
        if points_in.shape != expected or points_out.shape != expected:
            raise ValueError("endpoint coordinates must be (n_streamlines, 3)")
        self._locate(points_in, points_out)
        self.commit()
        return self

    @classmethod
    def from_endpoints(cls, endpoints, lh_grid, rh_grid):
        """Build from :class:`sbci.smoothing.Endpoints` carrying barycentric data.

        No point location is needed: the file already says which triangle each
        endpoint crossed and where inside it. The grids' faces must be in the
        same order as the file's triangle indices, which holds for the bundled
        ico4 surfaces (the face digest test guarantees it).
        """
        if not endpoints.has_positions:
            raise MissingDataError(
                "ConSEAL needs the triangle and barycentric position of every endpoint "
                "(tri_in/bary_in), not only the nearest vertex"
            )
        self = cls(lh_grid, rh_grid, endpoints.surf_in, endpoints.surf_out)
        faces = {0: lh_grid.faces, 1: rh_grid.faces}
        sides = (
            (
                self.hemisphere_in,
                endpoints.tri_in,
                endpoints.bary_in,
                self.weights_in,
                self.index_in,
                self.face_in,
            ),
            (
                self.hemisphere_out,
                endpoints.tri_out,
                endpoints.bary_out,
                self.weights_out,
                self.index_out,
                self.face_out,
            ),
        )
        for flags, tri, bary, weights, index, face in sides:
            tri = np.asarray(tri, dtype=np.int64).ravel()
            bary = np.asarray(bary, dtype=np.float64)
            if tri.shape != (self.n_streamlines,) or bary.shape != (self.n_streamlines, 3):
                raise ValueError(
                    "triangle indices and barycentric weights do not match the endpoints"
                )
            for side in (0, 1):
                mask = flags == side
                if not mask.any():
                    continue
                if tri[mask].min() < 0 or tri[mask].max() >= faces[side].shape[0]:
                    raise ValueError(
                        f"triangle index out of range for a {faces[side].shape[0]}-face hemisphere"
                    )
                weights[mask] = bary[mask]
                face[mask] = tri[mask]
                index[mask] = faces[side][tri[mask]] + (0 if side == 0 else self.n_left)
        self.commit()
        return self

    def to_endpoints(self):
        """The current locations as :class:`sbci.smoothing.Endpoints`, ready to re-smooth."""
        from .smoothing import Endpoints

        if self.rh_grid.n_vertices != self.n_left:
            raise ValueError("Endpoints assumes two hemispheres with the same vertex count")
        picks = np.arange(self.n_streamlines)
        vertex_in = self.index_in[picks, np.argmax(self.weights_in, axis=1)]
        vertex_out = self.index_out[picks, np.argmax(self.weights_out, axis=1)]
        faces_per_hemi = self.lh_grid.faces.shape[0]
        return Endpoints.from_global(
            vertex_in,
            vertex_out,
            n_per_hemi=self.n_left,
            n_faces_per_hemi=faces_per_hemi,
            triangle_in=self.face_in + self.hemisphere_in.astype(np.int64) * faces_per_hemi,
            triangle_out=self.face_out + self.hemisphere_out.astype(np.int64) * faces_per_hemi,
            barycentric_in=self.weights_in.copy(),
            barycentric_out=self.weights_out.copy(),
        )

    # -- location bookkeeping --------------------------------------------------

    def _locate(self, points_in, points_out) -> None:
        sides = (
            (self.hemisphere_in, points_in, self.weights_in, self.index_in, self.face_in),
            (self.hemisphere_out, points_out, self.weights_out, self.index_out, self.face_out),
        )
        for side in (0, 1):
            offset = 0 if side == 0 else self.n_left
            for flags, points, weights, index, face in sides:
                mask = flags == side
                if mask.any():
                    w, idx, f = self._query[side].query(points[mask])
                    weights[mask] = w
                    index[mask] = idx + offset
                    face[mask] = f

    def commit(self) -> None:
        """Make the current locations the ones later warps start from (``apply_warp``)."""
        self._original = tuple(
            a.copy()
            for a in (
                self.weights_in,
                self.index_in,
                self.face_in,
                self.weights_out,
                self.index_out,
                self.face_out,
            )
        )

    def reset(self) -> None:
        """Return to the committed locations (``SConcon.reset``)."""
        (
            self.weights_in,
            self.index_in,
            self.face_in,
            self.weights_out,
            self.index_out,
            self.face_out,
        ) = (a.copy() for a in self._original)

    def copy(self) -> EndpointConnectome:
        """An independent copy that shares the (immutable) grids and search trees."""
        clone = object.__new__(EndpointConnectome)
        clone.__dict__.update(self.__dict__)
        for name in ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out"):
            setattr(clone, name, getattr(self, name).copy())
        clone._original = tuple(a.copy() for a in self._original)
        return clone

    def positions(self, current: bool = True):
        """Unit-sphere endpoint coordinates, ``(points_in, points_out)``."""
        if current:
            w_in, i_in, w_out, i_out = (
                self.weights_in,
                self.index_in,
                self.weights_out,
                self.index_out,
            )
        else:
            w_in, i_in, _, w_out, i_out, _ = self._original
        vertices = np.vstack([self.lh_grid.vertices, self.rh_grid.vertices])
        return (
            normalize_rows(np.einsum("nk,nkj->nj", w_in, vertices[i_in])),
            normalize_rows(np.einsum("nk,nkj->nj", w_out, vertices[i_out])),
        )

    # -- the density -------------------------------------------------------------

    def adjacency(self) -> np.ndarray:
        """Endpoint mass spread onto vertex pairs: symmetric and summing to one.

        ``build_adjacency.cpp`` followed by ``(A + A') / (2N)``, accumulated in
        float64 rather than the reference's float32.
        """
        return self._sparse_adjacency().toarray()

    def _sparse_adjacency(self):
        """:meth:`adjacency` as a CSR matrix: 18 entries per streamline at most."""
        from scipy import sparse

        n = self.n_vertices
        rows = np.repeat(self.index_in, 3, axis=1).ravel()
        cols = np.tile(self.index_out, (1, 3)).ravel()
        vals = (self.weights_in[:, :, None] * self.weights_out[:, None, :]).reshape(-1)
        adjacency = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        return (adjacency + adjacency.T) / (2.0 * max(self.n_streamlines, 1))

    def evaluate(self, kernel, derivative=None, strict_upstream: bool = False):
        """The smooth connectome ``F = K^T A K``, clamped non-negative.

        With ``derivative`` (a :class:`KernelDerivative` from
        :meth:`HeatKernelBuilder.compute`) also returns the derivatives of
        ``F(a, c)`` with respect to moving the evaluation point ``a`` along its
        frame, ``(D_e1, D_e2)``. ``strict_upstream=True`` reproduces the
        reference's ``dK A K^T + (dK A K^T)^T`` instead (module docstring,
        items 1 and 2); the ``derivative`` must then come from
        ``compute(..., strict_upstream=True)`` as well.
        """
        kernel = _as_sparse(kernel)
        adjacency = self._sparse_adjacency()
        n = self.n_vertices
        # A has at most 18 entries per streamline, so with few streamlines A K
        # stays sparse and two sparse products beat four passes of the kernel
        # over a dense n x n matrix; with a subject's million streamlines A is
        # dense and the dense route is the cheaper one.
        few = not strict_upstream and adjacency.nnz * kernel.nnz < SPARSE_PRODUCT_FRACTION * n**3
        if few:
            ak = (adjacency @ kernel).tocsc()  # A K, since A is symmetric
            connectome = (kernel.T @ ak).toarray()
        else:
            dense = adjacency.toarray()
            # (K^T A)^T is A K but Fortran-ordered; make it contiguous once
            # rather than letting each sparse product copy it.
            ak = np.ascontiguousarray(sparse_times_dense(kernel.T, dense).T)
            connectome = sparse_times_dense(kernel.T, ak)
        np.maximum(connectome, 0.0, out=connectome)
        if derivative is None:
            return connectome

        if strict_upstream:
            akt = np.ascontiguousarray(sparse_times_dense(kernel, dense).T)  # A K^T
            parts = []
            for d in (derivative.x, derivative.y, derivative.z):
                part = sparse_times_dense(d, akt)
                parts.append(part + part.T)
            d_e1 = sum(p * e[:, None] for p, e in zip(parts, self.e1.T, strict=True))
            d_e2 = sum(p * e[:, None] for p, e in zip(parts, self.e2.T, strict=True))
            return connectome, d_e1, d_e2

        # sum_axis e1_axis(a) (dK_axis^T A K)(a, c) = (M1^T A K)(a, c) with the frame
        # folded into the sparse derivative: two sparse products instead of three,
        # and no dense per-axis parts.
        from scipy import sparse

        axes = (derivative.x, derivative.y, derivative.z)
        m1 = sum(d @ sparse.diags(e) for d, e in zip(axes, self.e1.T, strict=True))
        m2 = sum(d @ sparse.diags(e) for d, e in zip(axes, self.e2.T, strict=True))
        if few:
            return connectome, (m1.T @ ak).toarray(), (m2.T @ ak).toarray()
        return connectome, sparse_times_dense(m1.T, ak), sparse_times_dense(m2.T, ak)

    def q_transform(self, kernel, derivative=None, strict_upstream: bool = False):
        """The square-root density ``Q = sqrt(F)``, with derivatives by the chain rule.

        ``F`` is clamped at zero, so its square root has derivative zero
        wherever it was clamped. The reference divides by ``2 max(Q, 1e-15)``
        there instead (module docstring, item 13); ``strict_upstream=True``
        reproduces that.
        """
        if derivative is None:
            return np.sqrt(self.evaluate(kernel))
        connectome, d_e1, d_e2 = self.evaluate(kernel, derivative, strict_upstream)
        q = np.sqrt(connectome)
        if strict_upstream:
            scale = 1.0 / (2.0 * np.maximum(q, 1e-15))
        else:
            scale = np.divide(0.5, q, out=np.zeros_like(q), where=q > 0)
        # d_e1 and d_e2 are fresh arrays from evaluate(): scale them in place
        # rather than allocating two more n x n matrices.
        d_e1 *= scale
        d_e2 *= scale
        return q, d_e1, d_e2

    # -- warping -------------------------------------------------------------------

    def warp(self, lh_warp: StationaryWarp, rh_warp: StationaryWarp):
        """Carry every endpoint along with the warped grid, from the committed locations.

        An endpoint is the barycentric combination of its triangle's corners;
        the same weights applied to the *warped* corner positions move it with
        the diffeomorphism (``warp_connectome``). The result is re-projected
        onto the sphere and re-located on the unwarped grid, where the density
        is evaluated. Returns the new ``(points_in, points_out)``.
        """
        vertices = np.vstack([lh_warp.vertices, rh_warp.vertices])
        w_in, i_in, _, w_out, i_out, _ = self._original
        points_in = normalize_rows(np.einsum("nk,nkj->nj", w_in, vertices[i_in]))
        points_out = normalize_rows(np.einsum("nk,nkj->nj", w_out, vertices[i_out]))
        self._locate(points_in, points_out)
        return points_in, points_out


def _as_sparse(kernel):
    from scipy import sparse

    if sparse.issparse(kernel):
        return sparse.csr_matrix(kernel)
    return sparse.csr_matrix(np.asarray(kernel, dtype=np.float64))


# --- the kernel ---------------------------------------------------------------


@dataclass(frozen=True)
class KernelDerivative:
    """Cartesian derivatives of a kernel matrix, one sparse ``(P, P)`` matrix per axis.

    Rows are source vertices and columns evaluation vertices, as in the kernel
    itself. Which vertex the derivative is with respect to depends on how the
    matrix was built: see :meth:`HeatKernelBuilder.compute`.
    """

    x: object
    y: object
    z: object


def _legendre_with_derivative(x, degree: int):
    """Yield ``(l, P_l(x), dP_l/dx)`` for ``l = 0..degree`` by the reference's recurrences."""
    x = np.asarray(x, dtype=np.float64)
    p_prev, p_cur = np.ones_like(x), x.copy()
    d_prev, d_cur = np.zeros_like(x), np.ones_like(x)
    yield 0, p_prev, d_prev
    if degree >= 1:
        yield 1, p_cur, d_cur
    for h in range(1, degree):
        p_next = ((2 * h + 1) * x * p_cur - h * p_prev) / (h + 1)
        d_next = ((2 * h + 1) * (p_cur + x * d_cur) - h * d_prev) / (h + 1)
        yield h + 1, p_next, d_next
        p_prev, p_cur, d_prev, d_cur = p_cur, p_next, d_cur, d_next


class HeatKernelBuilder:
    """Row-normalized spherical heat kernel on two grids, with its derivative.

    Port of ``SphericalHeatKernel``. On each hemisphere
    ``K(x, y) = sum_{l<=H} (2l+1) exp(-l(l+1) sigma) P_l(x . y)``, zero where
    the cosine falls below :meth:`cutoff`, assembled block-diagonally and
    **row-normalized** so that every source vertex spreads unit mass. This is
    the heat kernel proper, weight ``(2l+1)``; concon's smoother in
    :mod:`sbci.smoothing` uses ``(2l+1)^(3/2)`` and no normalization, and the
    two are different objects. Kernels are returned sparse: with the published
    bandwidth only a few percent of the entries survive the cutoff, and the
    registration's cost is dominated by products with them.
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        degree: int = DEFAULT_KERNEL_DEGREE,
        epsilon: float = KERNEL_EPSILON,
    ):
        """Cache the vertex cosines of both grids for repeated bandwidths."""
        self.lh_grid, self.rh_grid = lh_grid, rh_grid
        self.degree = int(degree)
        self.epsilon = float(epsilon)
        self._cosines = [np.clip(g.vertices @ g.vertices.T, -1.0, 1.0) for g in (lh_grid, rh_grid)]
        self._scan = np.linspace(1.0, -1.0, 5000)

    def weights(self, sigma: float) -> np.ndarray:
        """``(2l+1) exp(-l(l+1) sigma)`` for ``l = 0..degree``."""
        degree = np.arange(self.degree + 1)
        return (2 * degree + 1) * np.exp(-degree * (degree + 1) * float(sigma))

    def cutoff(self, sigma: float) -> float:
        """Smallest cosine kept: the reference's scan of ``K(x) K(1)`` against ``epsilon``.

        Reproduces ``compute_cutoff`` step for step: on 5,000 cosines from 1
        down to -1, find the first at which ``K(x) K(1) < epsilon``, then the
        last positive value up to and including it.
        """
        w = self.weights(sigma)
        series = np.zeros_like(self._scan)
        for order, p, _ in _legendre_with_derivative(self._scan, self.degree):
            series += w[order] * p
        product = series * w.sum()  # K(1) = sum of the weights, since P_l(1) = 1
        below = np.flatnonzero(product < self.epsilon)
        if below.size == 0:
            return -1.0
        positive = np.flatnonzero(product[: below[0] + 1] > 0)
        return float(self._scan[positive[-1]]) if positive.size else -1.0

    def compute(
        self, sigma: float = DEFAULT_SIGMA, derivative: bool = True, strict_upstream: bool = False
    ):
        """The kernel ``K`` and, unless ``derivative=False``, its :class:`KernelDerivative`.

        By default the derivative of ``K~(i, a) = K(i, a) / s_i`` is taken with
        respect to the *evaluation* vertex ``a`` with the source normalization
        ``s_i`` held fixed, which is what differentiating ``F = K~^T A K~`` in
        its first argument needs. ``strict_upstream=True`` returns the
        reference's matrix instead: the derivative with respect to the *row*
        vertex ``i``, normalization included (module docstring, item 2).
        """
        from scipy import sparse

        w = self.weights(sigma)
        cut = self.cutoff(sigma)
        kernels, raw_derivatives, row_sums = [], [], []
        for cosines in self._cosines:
            rows, cols = np.nonzero(cosines >= cut)
            x = cosines[rows, cols]
            k = np.zeros_like(x)
            dk = np.zeros_like(x)
            for order, p, dp in _legendre_with_derivative(x, self.degree):
                k += w[order] * p
                if derivative:
                    dk += w[order] * dp
            n = cosines.shape[0]
            raw = sparse.csr_matrix((k, (rows, cols)), shape=(n, n))
            s = np.asarray(raw.sum(axis=1)).ravel()
            kernels.append(sparse.diags(1.0 / s) @ raw)
            raw_derivatives.append(sparse.csr_matrix((dk, (rows, cols)), shape=(n, n)))
            row_sums.append(s)

        kernel = sparse.block_diag(kernels, format="csr")
        if not derivative:
            return kernel

        parts = []
        for axis in range(3):
            blocks = []
            hemispheres = zip(
                (self.lh_grid, self.rh_grid), kernels, raw_derivatives, row_sums, strict=True
            )
            for grid, k_norm, d_raw, s in hemispheres:
                coordinate = grid.vertices[:, axis]
                if strict_upstream:
                    # d/d(row vertex i) of K(i, a) = P'(x_i . x_a) x_a; then the quotient
                    # rule for the row normalization: (D - K~ sum_b D(i, b)) / s_i.
                    d = d_raw @ sparse.diags(coordinate)
                    t = np.asarray(d.sum(axis=1)).ravel()
                    block = sparse.diags(1.0 / s) @ d - sparse.diags(t / s) @ k_norm
                else:
                    # d/d(evaluation vertex a) of K(i, a) / s_i = P'(x_i . x_a) x_i / s_i.
                    block = sparse.diags(coordinate / s) @ d_raw
                blocks.append(block)
            parts.append(sparse.block_diag(blocks, format="csr"))
        return kernel, KernelDerivative(*parts)

    def cross_validate(self, connectome: EndpointConnectome, sigmas, strict_upstream: bool = False):
        """Leave-one-out log-likelihood over candidate bandwidths; returns ``(best, scores)``.

        Port of ``Kernel.cross_validate_sigma``. The reference subtracts an
        approximate self-term, ``K(t_in, t_in) K(t_out, t_out)' / (M - 1)``
        (module docstring, item 7); by default each streamline's actual
        contribution to the symmetrized density is removed.
        """
        sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
        adjacency = connectome.adjacency()
        m = connectome.n_streamlines
        w_in, i_in = connectome.weights_in, connectome.index_in
        w_out, i_out = connectome.weights_out, connectome.index_out
        scores = np.zeros(sigmas.size)
        for s, sigma in enumerate(sigmas):
            sparse_kernel = self.compute(sigma, derivative=False)
            full = np.maximum(np.asarray(sparse_kernel.T @ (sparse_kernel.T @ adjacency).T), 0.0)
            kernel = sparse_kernel.toarray()  # dense only for the (m, 3, 3) gathers below
            block = full[i_in[:, :, None], i_out[:, None, :]]  # (m, 3, 3)
            k_in = kernel[i_in[:, :, None], i_in[:, None, :]]  # K(t_in, t_in)
            k_out = kernel[i_out[:, :, None], i_out[:, None, :]]
            if strict_upstream:
                own = np.einsum("nij,nkj->nik", k_in, k_out) / max(m - 1, 1)
            else:
                # the streamline's own term of K~^T A K~ on its two triangles, both
                # orientations of the symmetrized adjacency included
                left = np.einsum("nij,ni->nj", k_in, w_in)
                right = np.einsum("nij,ni->nj", k_out, w_out)
                k_oi = kernel[i_out[:, :, None], i_in[:, None, :]]
                k_io = kernel[i_in[:, :, None], i_out[:, None, :]]
                cross_in = np.einsum("nij,ni->nj", k_oi, w_out)
                cross_out = np.einsum("nij,ni->nj", k_io, w_in)
                own = (
                    left[:, :, None] * right[:, None, :]
                    + cross_in[:, :, None] * cross_out[:, None, :]
                ) / (2.0 * m)
            values = np.einsum("ni,nij,nj->n", w_in, block - own, w_out)
            scores[s] = float(np.mean(np.log(np.maximum(values, np.finfo(float).eps))))
        return float(sigmas[int(np.argmax(scores))]), scores


# --- the warp -------------------------------------------------------------------


def cotangent_laplacian(vertices, faces, reference_layout: bool = False):
    """The cotangent Laplacian ``L = D - W`` of a mesh, sparse.

    Each edge is weighted by half the cotangent of the angle opposite it. The
    reference's ``cot_matrix`` weights it with an *adjacent* angle instead
    (module docstring, item 12); ``reference_layout=True`` reproduces that.
    """
    from scipy import sparse

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    i1, i2, i3 = faces[:, 0], faces[:, 1], faces[:, 2]
    v1 = vertices[i2] - vertices[i3]
    v2 = vertices[i3] - vertices[i1]
    v3 = vertices[i1] - vertices[i2]
    double_area = np.linalg.norm(np.cross(v1, -v3), axis=1)
    cot_at_2 = (v1 * -v3).sum(1) / double_area  # angle at i2
    cot_at_3 = (v2 * -v1).sum(1) / double_area  # angle at i3
    cot_at_1 = (v3 * -v2).sum(1) / double_area  # angle at i1
    if reference_layout:
        rows = np.concatenate([i2, i3, i3, i1, i1, i2])
        cols = np.concatenate([i3, i2, i1, i3, i2, i1])
    else:
        # the angle at i1 is opposite edge (i2, i3), and so on around the triangle
        rows = np.concatenate([i3, i1, i1, i2, i2, i3])
        cols = np.concatenate([i1, i3, i2, i1, i3, i2])
    vals = 0.5 * np.concatenate([cot_at_2, cot_at_2, cot_at_3, cot_at_3, cot_at_1, cot_at_1])
    n = vertices.shape[0]
    weights = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(weights.sum(axis=1)).ravel()
    return (sparse.diags(degree) - weights).tocsr()


def _transport(tangent, origin, destination):
    """Parallel transport on the sphere, the reference's closed form."""
    pq = (origin * destination).sum(axis=1, keepdims=True)
    vq = (tangent * destination).sum(axis=1, keepdims=True)
    denominator = 1.0 + pq
    safe = np.where(denominator < 1e-8, 1.0, denominator)
    moved = tangent - (vq / safe) * (origin + destination)
    bad = denominator[:, 0] < 1e-8
    moved[bad] = tangent[bad]
    return moved


def triangles_fold(vertices, faces) -> bool:
    """Whether any triangle has turned inside out, by the reference's normal test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    p1, p2, p3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = (np.cross(p2 - p1, p3 - p1) * (p1 + p2 + p3)).sum(axis=1)
    return bool(np.any(normals <= 0))


class StationaryWarp:
    """A diffeomorphism of one hemisphere's sphere, as a stationary velocity field.

    Port of the reference's ``SphericalWarp``. The warp is the unit-time flow
    of a tangent field ``velocity`` (``(P, 2)`` in the ``(e1, e2)`` frame),
    computed by scaling and squaring with barycentric interpolation and
    parallel transport. Composing is addition of velocity fields, followed by
    the reference's 5% Laplacian smoothing (module docstring, item 8).
    """

    def __init__(
        self,
        grid: SphericalGrid,
        viscosity: float = VISCOSITY,
        squarings: int = SQUARINGS,
        strict_upstream: bool = False,
    ):
        """The identity warp of ``grid``.

        ``strict_upstream`` selects the reference's cotangent weights (module
        docstring, item 12). The faces must be oriented outward, since that is
        what the fold test assumes; an inward mesh would refuse every step.
        """
        self.grid = grid
        self.e1, self.e2 = grid.e1, grid.e2
        self.faces = grid.faces
        self.base = grid.vertices.copy()
        if triangles_fold(self.base, self.faces):
            raise ValueError("the mesh faces must be oriented outward (counter-clockwise)")
        self.vertices = grid.vertices.copy()
        self.velocity = np.zeros((grid.n_vertices, 2))
        self.viscosity = float(viscosity)
        self.squarings = int(squarings)
        self.rejected = 0
        self._base_areas = voronoi_areas(self.base, self.faces)
        self._query = FastMeshQuery(self.base, self.faces)
        self._laplacian = cotangent_laplacian(
            self.base, self.faces, reference_layout=strict_upstream
        )

    @property
    def n_vertices(self) -> int:
        """Vertices in the hemisphere."""
        return int(self.base.shape[0])

    @property
    def jacobian(self) -> np.ndarray:
        """Warped over unwarped Voronoi area, per vertex (``SphericalWarp.J``)."""
        return voronoi_areas(self.vertices, self.faces) / self._base_areas

    def copy(self) -> StationaryWarp:
        """An independent copy sharing the grid and its search tree."""
        clone = object.__new__(StationaryWarp)
        clone.__dict__.update(self.__dict__)
        clone.velocity = self.velocity.copy()
        clone.vertices = self.vertices.copy()
        return clone

    def exponential(self, velocity=None) -> np.ndarray:
        """Flow the base vertices along ``velocity`` for unit time (``exp_svf``)."""
        velocity = self.velocity if velocity is None else np.asarray(velocity, dtype=np.float64)
        scaled = velocity / (2**self.squarings)
        tangent = self.e1 * scaled[:, :1] + self.e2 * scaled[:, 1:2]
        current = normalize_rows(sphere_exp_map(self.base, tangent))
        n = self.n_vertices
        for _ in range(self.squarings):
            weights, indices, _ = self._query.query(current)
            displacement = sphere_log_map(self.base, current)
            moved = _transport(
                displacement[indices].reshape(-1, 3),
                self.base[indices].reshape(-1, 3),
                np.repeat(current, 3, axis=0),
            ).reshape(n, 3, 3)
            step = (weights[:, :, None] * moved).sum(axis=1)
            current = normalize_rows(sphere_exp_map(current, step))
        return current

    def compose(self, displacement, strict_upstream: bool = False) -> bool:
        """Add a tangent displacement to the velocity field and re-flow; returns whether applied.

        An update that would fold a triangle is refused. The reference keeps
        the refused displacement in the velocity field regardless (module
        docstring, item 3), which ``strict_upstream=True`` reproduces.
        """
        displacement = np.asarray(displacement, dtype=np.float64)
        velocity = self.velocity + displacement
        if self.viscosity:
            velocity = velocity - self.viscosity * (self._laplacian @ velocity)
        candidate = self.exponential(velocity)
        if triangles_fold(candidate, self.faces):
            self.rejected += 1
            warnings.warn(
                "skipping a warp update that would fold triangles", RuntimeWarning, stacklevel=2
            )
            if strict_upstream:
                self.velocity = velocity
            return False
        self.velocity = velocity
        self.vertices = candidate
        return True

    def invert(self) -> StationaryWarp:
        """Negate the velocity field and re-flow, in place (``invert``)."""
        self.velocity = -self.velocity
        self.vertices = self.exponential()
        return self

    def rotate(self, rotation) -> StationaryWarp:
        """Set the warp to a rigid rotation, as the flow of its rotational field (``rotate``)."""
        from scipy.spatial.transform import Rotation

        # The reference reads the axis off the skew part of R, which vanishes for
        # a half turn and silently makes it the identity; the rotation vector
        # from a proper decomposition does not.
        omega = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()
        field = np.cross(np.broadcast_to(omega, self.base.shape), self.base)
        self.velocity = np.stack([(field * self.e1).sum(1), (field * self.e2).sum(1)], axis=1)
        self.vertices = self.exponential()
        return self

    def apply(self, points) -> np.ndarray:
        """Where the warp sends arbitrary points of this hemisphere's sphere."""
        points = normalize_rows(np.asarray(points, dtype=np.float64))
        weights, indices, _ = self._query.query(points)
        return normalize_rows(np.einsum("nk,nkj->nj", weights, self.vertices[indices]))


# --- rigid initialization -----------------------------------------------------


def icosphere(subdivisions: int):
    """The reference's ``icosphere``: vertices as cyclic permutations of ``(0, +-1, +-phi)``."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    vertices = normalize_rows(vertices)
    for _ in range(int(subdivisions)):
        cache: dict[tuple[int, int], int] = {}
        points = list(vertices)
        new_faces = []
        for a, b, c in faces:
            midpoints = []
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key not in cache:
                    cache[key] = len(points)
                    middle = (vertices[u] + vertices[v]) / 2.0
                    points.append(middle / np.linalg.norm(middle))
                midpoints.append(cache[key])
            ab, bc, ca = midpoints
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        vertices = np.asarray(points)
        faces = np.asarray(new_faces, dtype=np.int64)
    return vertices, faces


def icosahedral_rotations(vertices, faces):
    """The 60 rotations of the icosahedral group fitted to a mesh, and how well they fit.

    Returns ``(rotations, deviation)``: ``rotations`` is ``(60, 3, 3)`` and
    ``deviation`` the largest distance, as a chord on the unit sphere (about
    the angle in radians), from a rotated vertex to the nearest mesh vertex
    over all 60 rotations. It is rounding error on an icosphere and 1.7e-4 (a
    hundredth of a degree) on FreeSurfer's ico4 sphere, whose stored
    coordinates carry only so many digits.

    The twelve five-valent vertices are the icosahedron's corners; a group
    element is fixed by where it sends one corner and one of that corner's
    neighbours, which gives exactly 60 candidates, and each is snapped to the
    nearest proper rotation (polar decomposition), since on an inexact mesh the
    two corner frames are not related by a rotation.
    """
    from scipy.spatial import cKDTree

    vertices = normalize_rows(np.asarray(vertices, dtype=np.float64))
    faces = np.asarray(faces, dtype=np.int64)
    n = vertices.shape[0]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    valence = np.bincount(edges.ravel(), minlength=n)
    corners = np.flatnonzero(valence == 5)
    if corners.size != 12:
        raise ValueError(f"an icosphere has 12 five-valent vertices, this mesh has {corners.size}")
    corner_points = vertices[corners]
    dots = corner_points @ corner_points.T
    np.fill_diagonal(dots, -2.0)
    neighbours = np.argsort(-dots, axis=1)[:, :5]
    first, second = corner_points[0], corner_points[neighbours[0, 0]]
    frame0 = np.stack([first, second, np.cross(first, second)])
    tree = cKDTree(vertices)
    rotations, deviations, seen = [], [], set()
    for i in range(12):
        for j in neighbours[i]:
            a, b = corner_points[i], corner_points[j]
            candidate = np.linalg.solve(frame0, np.stack([a, b, np.cross(a, b)])).T
            u, _, vt = np.linalg.svd(candidate)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                continue
            key = tuple(np.round(rotation, 2).ravel())  # distinct elements differ by >= 72 degrees
            if key in seen:
                continue
            seen.add(key)
            distance, _ = tree.query(vertices @ rotation.T)
            rotations.append(rotation)
            deviations.append(float(distance.max()))
    if len(rotations) != 60:
        raise ValueError(
            f"an icosphere has 60 rotational symmetries; this mesh's corners give {len(rotations)}"
        )
    return np.stack(rotations), float(max(deviations))


def mesh_symmetries(vertices, faces, tolerance: float = 1e-3):
    """The 60 rotations that map an icosphere onto itself, ``(rotations, permutations)``.

    Replaces ``icosahedral_permutations``, which hard-codes the reference's
    icosahedron orientation and errors on any other. The rotations come from
    :func:`icosahedral_rotations`; here each must permute the vertex set to
    within ``tolerance``, a chord on the unit sphere. The default admits
    FreeSurfer's ico4 sphere (1.7e-4 off an exact icosphere, a fortieth of its
    edge length) and refuses a mesh jittered by more. ``permutations[g]``
    follows the reference's convention: the function transported by rotation
    ``g`` is ``Q[perm][:, perm]``.
    """
    from scipy.spatial import cKDTree

    vertices = normalize_rows(np.asarray(vertices, dtype=np.float64))
    rotations, deviation = icosahedral_rotations(vertices, faces)
    if deviation > tolerance:
        raise ValueError(
            f"found 0 exact symmetries instead of 60: the mesh is not an icosphere to "
            f"within {tolerance:g} (its icosahedral rotations move vertices by up to "
            f"{deviation:.2g})"
        )
    tree = cKDTree(vertices)
    permutations = []
    for rotation in rotations:
        _, image = tree.query(vertices @ rotation.T)
        if np.unique(image).size != vertices.shape[0]:
            raise ValueError("found fewer than 60 symmetries: a rotation is not a permutation")
        permutations.append(np.argsort(image))  # perm[j] = the vertex that lands on j
    return rotations, np.stack(permutations)


#: :meth:`EndpointConnectome.evaluate` keeps the adjacency sparse while
#: ``nnz(A) * nnz(K) / n``, an upper bound on the entries of ``A K``, is below
#: this fraction of ``n^2``. On ico4 at the published bandwidth that is about
#: 5,000 streamlines; a subject's million make ``A K`` dense, and the dense
#: route is then the faster one.
SPARSE_PRODUCT_FRACTION = 0.3

_SHELLS: list | None = None


def rotation_shells():
    """The reference's rotation search schedule, regenerated from ``create_search_schedule``.

    Caps of 20, 10, 5, 2.5 and 1 degrees about ``+z``, sampled by icospheres
    of subdivision 4, 4, 5, 6 and 7; each direction gives the rotation taking
    ``z`` to it, plus an appended identity; 5, 3, 1, 1 and 1 candidates are
    kept after each shell. Returns ``[(rotations (n, 3, 3), keep), ...]``.
    """
    global _SHELLS
    if _SHELLS is not None:
        return [(r.copy(), k) for r, k in _SHELLS]
    schedule = [(20.0, 4, 5), (10.0, 4, 3), (5.0, 5, 1), (2.5, 6, 1), (1.0, 7, 1)]
    z = np.array([0.0, 0.0, 1.0])
    shells = []
    for radius, subdivision, keep in schedule:
        directions, _ = icosphere(subdivision)
        inside = np.arccos(np.clip(directions[:, 2], -1.0, 1.0)) <= np.radians(radius)
        rotations = []
        for target in directions[inside]:
            angle = float(np.arccos(np.clip(z @ target, -1.0, 1.0)))
            k = np.cross(z, target)
            norm = np.linalg.norm(k)
            if angle < 1e-12 or norm < 1e-12:
                rotations.append(np.eye(3))
                continue
            k = k / norm
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            rotations.append(np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx))
        rotations.append(np.eye(3))
        shells.append((np.stack(rotations), keep))
    _SHELLS = shells
    return [(r.copy(), k) for r, k in shells]


# --- registration ---------------------------------------------------------------


@dataclass
class EndpointWarp:
    """One subject's warp of both hemispheres, exported from :class:`StationaryWarp`."""

    lh_vertices: np.ndarray
    lh_velocity: np.ndarray
    lh_jacobian: np.ndarray
    rh_vertices: np.ndarray
    rh_velocity: np.ndarray
    rh_jacobian: np.ndarray

    def save(self, path) -> Path:
        """Write the warp to ``.npz``."""
        path = Path(path)
        np.savez_compressed(path, **dict(self.__dict__))
        return path

    @classmethod
    def load(cls, path) -> EndpointWarp:
        """Read a warp written by :meth:`save`."""
        with np.load(Path(path)) as data:
            return cls(**{k: data[k] for k in data.files})


@dataclass
class EndpointAlignment:
    """What :func:`endpoints_align` returns."""

    template: np.ndarray
    warps: list = field(default_factory=list)
    connectomes: list = field(default_factory=list)
    costs: list = field(default_factory=list)
    rejected_steps: list = field(default_factory=list)

    def aligned_endpoints(self, index: int):
        """Subject ``index``'s warped endpoints as :class:`sbci.smoothing.Endpoints`."""
        return self.connectomes[index].to_endpoints()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<EndpointAlignment of {len(self.warps)} subjects "
            f"on {self.template.shape[0]} vertices>"
        )


class ConSEAL:
    """Diffeomorphic registration of endpoint connectomes (the reference's ``Encore`` class).

    Parameters
    ----------
    lh_grid, rh_grid
        :class:`sbci.alignment.SphericalGrid` built with the warp basis order.
    delta, max_iterations, threshold, step_clamp, viscosity
        Gradient step, iteration cap, stopping change in cost, largest
        per-vertex displacement per step, and Laplacian smoothing of the
        velocity field. The defaults are the public code's; ``step_clamp=inf``
        and ``viscosity=0`` give the author's fork's update.
    area_weighted
        Weight the cost and its gradient with the Voronoi areas, as the
        paper's integral does. Off by default to match the reference (item 5).
    strict_upstream
        Reproduce items 1 to 4 of the module docstring.
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        delta: float = DEFAULT_DELTA,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        threshold: float = DEFAULT_THRESHOLD,
        step_clamp: float = STEP_CLAMP,
        viscosity: float = VISCOSITY,
        area_weighted: bool = False,
        strict_upstream: bool = False,
    ):
        """Fix the grids and the descent settings."""
        self.lh_grid, self.rh_grid = lh_grid, rh_grid
        self.delta = float(delta)
        self.max_iterations = int(max_iterations)
        self.threshold = float(threshold)
        self.step_clamp = float(step_clamp)
        self.viscosity = float(viscosity)
        self.area_weighted = bool(area_weighted)
        self.strict_upstream = bool(strict_upstream)
        self._areas = np.concatenate([lh_grid.areas, rh_grid.areas])

    def new_warps(self):
        """A pair of identity warps with this engine's viscosity."""
        return (
            StationaryWarp(self.lh_grid, self.viscosity, strict_upstream=self.strict_upstream),
            StationaryWarp(self.rh_grid, self.viscosity, strict_upstream=self.strict_upstream),
        )

    # -- cost --------------------------------------------------------------------

    def cost(self, difference: np.ndarray) -> float:
        """Squared mismatch of two square-root densities, ``sum (Q1 - Q2)^2``."""
        if self.area_weighted:
            return float(self._areas @ (difference**2) @ self._areas)
        return float((difference**2).sum())

    # -- template ------------------------------------------------------------------

    def template(self, connectomes, kernel, iterations: int = 100, verbose: bool = False):
        """Karcher median of the subjects' square-root densities (``get_template``).

        Starts from the subject nearest the Euclidean mean and takes Weiszfeld
        steps of length 0.2 until one is shorter than 0.005.
        """
        qs = [c.q_transform(kernel) for c in connectomes]
        if not qs:
            raise ValueError("a template needs at least one connectome")
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
                if distance > 0:
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
            if verbose:
                print(f"template step {norm:.4f}")
            if norm < 0.005:
                break
        return q_mu

    # -- gradient --------------------------------------------------------------------

    def _gradient(self, difference, q, q_e1, q_e2, rows, grid: SphericalGrid):
        """``compute_gradient``: the descent direction on one hemisphere, ``(P, 2)``."""
        block = difference[rows]
        if self.area_weighted:
            block = block * self._areas[None, :]
        s1 = (block * q_e1[rows]).sum(axis=1)
        s2 = (block * q_e2[rows]).sum(axis=1)
        s3 = (block * q[rows]).sum(axis=1)
        integrand = (
            2 * s1[:, None] * grid.basis[:, :, 0]
            + 2 * s2[:, None] * grid.basis[:, :, 1]
            + s3[:, None] * grid.laplacian
        )
        if self.area_weighted:
            integrand = integrand * grid.areas[:, None]
        coefficients = 2.0 * integrand.sum(axis=0)
        return (coefficients[None, :, None] * grid.basis).sum(axis=1)

    def _step(self, gradient) -> np.ndarray:
        """``compute_step_size``: clamp the largest displacement, then scale by ``-delta``."""
        largest = float(np.linalg.norm(gradient, axis=1).max())
        if largest > self.step_clamp:
            gradient = gradient * (self.step_clamp / largest)
        return -self.delta * gradient

    # -- rigid initialization ---------------------------------------------------------

    @staticmethod
    def _pull_back(values, grid: SphericalGrid, query: FastMeshQuery, rotation):
        """``rotate_matrix_barycentric`` on one hemisphere: ``g(x) = values(R x)``."""
        from scipy import sparse

        weights, indices, _ = query.query(grid.vertices @ rotation.T)
        n = grid.n_vertices
        operator = sparse.csr_matrix(
            (weights.ravel(), (np.repeat(np.arange(n), 3), indices.ravel())), shape=(n, n)
        )
        halfway = sparse_times_dense(operator, values)
        return sparse_times_dense(operator, halfway.T).T

    def _rigid(self, q1, moving: EndpointConnectome, kernel, verbose: bool = False):
        """``initial_registration``: a multi-shell rotation search, per hemisphere."""
        q2 = moving.q_transform(kernel)
        initial = self.cost(q1 - q2)
        n_left = moving.n_left
        shells = rotation_shells()
        best_pull_back = []
        hemispheres = (
            (moving.lh_grid, slice(0, n_left), moving._query[0]),
            (moving.rh_grid, slice(n_left, moving.n_vertices), moving._query[1]),
        )
        for grid, rows, query in hemispheres:
            target = q1[rows, rows]
            block = q2[rows, rows]
            group, _ = icosahedral_rotations(grid.vertices, grid.faces)

            # shell 1: every shell rotation combined with the 60 icosahedral
            # rotations. Both sides are transported by barycentric pull-back: on
            # an exact icosphere that is the reference's vertex permutation
            # (weights 1, 0, 0), and on the bundled FreeSurfer sphere, which is
            # 1.7e-4 off one, it is the rotated function rather than a gather.
            # ||T.g - B.r||^2 expands into two norms and one inner product, so
            # the 60 x 80 costs are a few matrix products; the rotations are
            # chunked so that at most ~1 GB of transported blocks is held.
            rotations, keep = shells[0]
            costs = np.zeros((group.shape[0], rotations.shape[0]))
            chunk = max(1, min(rotations.shape[0], int(1.2e8) // max(target.size, 1)))
            for first in range(0, rotations.shape[0], chunk):
                rotated = np.stack(
                    [
                        self._pull_back(block, grid, query, rotation).ravel()
                        for rotation in rotations[first : first + chunk]
                    ]
                )
                rotated_norm = (rotated**2).sum(axis=1)
                for g, symmetry in enumerate(group):
                    pulled = self._pull_back(target, grid, query, symmetry).ravel()
                    cross = rotated @ pulled
                    costs[g, first : first + chunk] = (pulled @ pulled) + rotated_norm - 2.0 * cross
            order = np.argsort(costs.ravel())[:keep]
            g_idx, s_idx = np.unravel_index(order, costs.shape)
            candidates = [rotations[s] @ group[g].T for g, s in zip(g_idx, s_idx, strict=True)]

            # later shells: refine each candidate with a finer cap
            for rotations, keep in shells[1:]:
                costs = np.zeros((len(candidates), rotations.shape[0]))
                for k, candidate in enumerate(candidates):
                    for s, rotation in enumerate(rotations):
                        rotated = self._pull_back(block, grid, query, candidate @ rotation)
                        costs[k, s] = ((target - rotated) ** 2).sum()
                order = np.argsort(costs.ravel())[:keep]
                k_idx, s_idx = np.unravel_index(order, costs.shape)
                candidates = [
                    candidates[k] @ rotations[s] for k, s in zip(k_idx, s_idx, strict=True)
                ]
            best_pull_back.append(candidates[0])

        lh_warp, rh_warp = self.new_warps()
        lh_warp.rotate(best_pull_back[0].T)
        rh_warp.rotate(best_pull_back[1].T)
        moving.warp(lh_warp, rh_warp)
        final = self.cost(q1 - moving.q_transform(kernel))
        if verbose:
            print(f"rigid initialization: cost {initial:.6f} -> {final:.6f}")
        if final > initial:
            moving.reset()
            return self.new_warps()
        return lh_warp, rh_warp

    # -- registration ------------------------------------------------------------------

    def register(
        self,
        fixed,
        moving: EndpointConnectome,
        kernel,
        derivative: KernelDerivative,
        init_rotation: bool = False,
        verbose: bool = False,
        callback=None,
    ):
        """Register ``moving`` onto ``fixed``; returns ``(lh_warp, rh_warp, costs, warped)``.

        ``fixed`` is an :class:`EndpointConnectome` or a square-root density
        such as a template. ``moving`` is copied; ``warped`` is that copy with
        its endpoints carried along, and ``costs`` the trace from the initial
        cost onward. ``callback(iteration, cost)`` is called after every step.
        """
        if isinstance(fixed, EndpointConnectome):
            q1 = fixed.q_transform(kernel)
        else:
            q1 = np.asarray(fixed, dtype=np.float64)
        moving = moving.copy()
        if init_rotation:
            lh_warp, rh_warp = self._rigid(q1, moving, kernel, verbose)
        else:
            lh_warp, rh_warp = self.new_warps()

        rows_lh = slice(0, moving.n_left)
        rows_rh = slice(moving.n_left, moving.n_vertices)
        strict = self.strict_upstream
        q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative, strict)
        difference = q1 - q2
        costs = [self.cost(difference)]
        if verbose:
            print(f"initial cost {costs[0]:.6f}")

        for iteration in range(1, self.max_iterations + 1):
            previous = (lh_warp.copy(), rh_warp.copy())
            gradient_lh = self._gradient(difference, q2, q2_e1, q2_e2, rows_lh, self.lh_grid)
            gradient_rh = self._gradient(difference, q2, q2_e1, q2_e2, rows_rh, self.rh_grid)
            lh_warp.compose(self._step(gradient_lh), strict)
            rh_warp.compose(self._step(gradient_rh), strict)
            moving.warp(lh_warp, rh_warp)
            q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative, strict)
            difference = q1 - q2
            costs.append(self.cost(difference))
            if callback is not None:
                callback(iteration, costs[-1])
            if verbose and iteration % 10 == 0:
                print(f"iteration {iteration}: cost {costs[-1]:.6f}")
            improvement = costs[-2] - costs[-1]
            if improvement < self.threshold:
                if improvement < 0 and not strict:
                    # roll back rather than return the warp that made things worse (item 4)
                    lh_warp, rh_warp = previous
                    moving.warp(lh_warp, rh_warp)
                    costs.pop()
                break
        if verbose:
            print(f"final cost {costs[-1]:.6f} after {len(costs) - 1} iterations")
        return lh_warp, rh_warp, np.asarray(costs), moving


# --- metrics --------------------------------------------------------------------------


def endpoint_mass(connectome: EndpointConnectome) -> np.ndarray:
    """Per-vertex endpoint density, both ends together, summing to one."""
    density = np.zeros(connectome.n_vertices)
    for weights, index in (
        (connectome.weights_in, connectome.index_in),
        (connectome.weights_out, connectome.index_out),
    ):
        np.add.at(density, index.ravel(), weights.ravel())
    return density / (2.0 * max(connectome.n_streamlines, 1))


def overlap_coefficient(first: EndpointConnectome, second: EndpointConnectome, threshold: float):
    """``|X & Y| / min(|X|, |Y|)`` over vertices whose endpoint density exceeds ``threshold``."""
    x = endpoint_mass(first) > threshold
    y = endpoint_mass(second) > threshold
    return float((x & y).sum() / max(min(x.sum(), y.sum()), 1))


def dice_score(first: EndpointConnectome, second: EndpointConnectome, threshold: float) -> float:
    """``2 |X & Y| / (|X| + |Y|)`` over vertices whose endpoint density exceeds ``threshold``."""
    x = endpoint_mass(first) > threshold
    y = endpoint_mass(second) > threshold
    return float(2 * (x & y).sum() / max(x.sum() + y.sum(), 1))


# --- the public entry point ---------------------------------------------------------


def default_grids(order: int = DEFAULT_WARP_ORDER):
    """Both hemispheres of the bundled ico4 sphere as :class:`SphericalGrid`, unrotated.

    ENCORE rotates the sphere clear of the coordinate poles because its
    Jacobian is formed in ``(theta, phi)``; ConSEAL's Jacobian is an area
    ratio, so the grids stay in the file's frame and endpoint coordinates need
    no rotation.
    """
    from .alignment import _hemisphere_grids

    return _hemisphere_grids(order, rotate=False)


def endpoints_align(
    connectomes,
    template=None,
    sigma: float = DEFAULT_SIGMA,
    kernel_degree: int = DEFAULT_KERNEL_DEGREE,
    order: int = DEFAULT_WARP_ORDER,
    delta: float = DEFAULT_DELTA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    threshold: float = DEFAULT_THRESHOLD,
    step_clamp: float = STEP_CLAMP,
    viscosity: float = VISCOSITY,
    template_iterations: int = 100,
    init_rotation: bool = False,
    area_weighted: bool = False,
    strict_upstream: bool = False,
    grids=None,
    verbose: bool = False,
) -> EndpointAlignment:
    """Align a cohort by their streamline endpoints (ConSEAL).

    Parameters
    ----------
    connectomes
        :class:`~sbci.ContinuousConnectome` objects carrying endpoints with
        barycentric positions, :class:`~sbci.smoothing.Endpoints`, or
        :class:`EndpointConnectome` built on ``grids``.
    template
        A subject index to register everyone to, a precomputed square-root
        density, or ``None`` for the Karcher median of the cohort.
    sigma, kernel_degree
        Heat-kernel bandwidth and truncation degree (0.005 and 30 in the paper).
    order, delta, max_iterations, threshold, step_clamp, viscosity
        Velocity-basis order and gradient-descent settings; the defaults are
        the public code's (15, 0.05, 100, 1e-4, 0.2, 0.05).
    init_rotation
        Search rotations per hemisphere before the diffeomorphic step. The 60
        icosahedral rotations are fitted to the grid rather than assumed, and
        the densities are transported by interpolation, so the search runs on
        the bundled FreeSurfer sphere, which is an icosphere to 1.7e-4.
    area_weighted, strict_upstream
        See :class:`ConSEAL`.
    grids
        ``(lh_grid, rh_grid)`` to work on; defaults to :func:`default_grids`.

    Returns
    -------
    EndpointAlignment
        The template, one :class:`EndpointWarp` per subject, the warped
        connectomes (``aligned_endpoints(i)`` gives them back as
        :class:`~sbci.smoothing.Endpoints`), and each cost trace.
    """
    from .smoothing import Endpoints

    lh_grid, rh_grid = grids if grids is not None else default_grids(order)
    subjects = []
    for item in connectomes:
        if isinstance(item, EndpointConnectome):
            subjects.append(item)
        elif isinstance(item, Endpoints):
            subjects.append(EndpointConnectome.from_endpoints(item, lh_grid, rh_grid))
        else:
            endpoints = getattr(item, "endpoints", None)
            if endpoints is None:
                raise MissingDataError(
                    "endpoints_align needs connectomes that carry their endpoints"
                )
            subjects.append(EndpointConnectome.from_endpoints(endpoints, lh_grid, rh_grid))
    if not subjects:
        raise ValueError("endpoints_align needs at least one connectome")

    builder = HeatKernelBuilder(lh_grid, rh_grid, kernel_degree)
    kernel, derivative = builder.compute(sigma, derivative=True, strict_upstream=strict_upstream)
    engine = ConSEAL(
        lh_grid,
        rh_grid,
        delta,
        max_iterations,
        threshold,
        step_clamp,
        viscosity,
        area_weighted=area_weighted,
        strict_upstream=strict_upstream,
    )

    if template is None:
        target = engine.template(subjects, kernel, template_iterations, verbose)
    elif isinstance(template, (int, np.integer)):
        target = subjects[int(template)].q_transform(kernel)
    else:
        target = np.asarray(template, dtype=np.float64)
    expected = lh_grid.n_vertices + rh_grid.n_vertices
    if target.shape != (expected, expected):
        raise ValueError(
            f"template is {target.shape}, expected a {expected} x {expected} square-root density"
        )

    result = EndpointAlignment(template=target)
    for i, subject in enumerate(subjects):
        if verbose:
            print(f"registering subject {i + 1} of {len(subjects)}")
        lh_warp, rh_warp, costs, warped = engine.register(
            target, subject, kernel, derivative, init_rotation=init_rotation, verbose=verbose
        )
        result.warps.append(
            EndpointWarp(
                lh_vertices=lh_warp.vertices,
                lh_velocity=lh_warp.velocity,
                lh_jacobian=lh_warp.jacobian,
                rh_vertices=rh_warp.vertices,
                rh_velocity=rh_warp.velocity,
                rh_jacobian=rh_warp.jacobian,
            )
        )
        result.connectomes.append(warped)
        result.costs.append(costs)
        result.rejected_steps.append(lh_warp.rejected + rh_warp.rejected)
    return result
