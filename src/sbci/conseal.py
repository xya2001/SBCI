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
The reference is research code. The port corrects items 1 to 4, 6, 7 and 9 to
17 below, takes the paper's settings for item 8, and keeps 5 as the choice it
is. Until 11 October 2026 ``strict_upstream=True`` reproduced the reference's
arithmetic, settings and all. It went once ConSEAL's defaults were shown to
undo known warps better (PORTING.md item 22). PORTING.md items 7 and 19 have
the other measurements.

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
   both the cost and its gradient with the areas, divided by their mean so
   that the cost stays on the plain sum's scale: the raw areas made it about
   40,000 times smaller on ico4, below the stopping threshold, and a
   registration stopped before its first step (until October 2026).
6. **Triangle indices are ``int16``**, which overflows once both hemispheres
   exceed 32,767 vertices -- ico6 and finer. The author's ``Encore-main`` copy
   already changes this to ``int32``; this port uses ``int64``.
7. **The leave-one-out bandwidth score removes an approximate self-term**
   (no barycentric weights, ``1/(M-1)`` rather than the ``1/(2N)`` of the
   symmetrized adjacency). Only :meth:`HeatKernelBuilder.cross_validate` is
   affected, and the published pipeline fixes ``sigma = 0.005`` anyway.
8. **The paper says no explicit regularization**; the public code smooths the
   velocity field by 5% of its cotangent Laplacian at every step, caps the
   gradient's largest per-vertex norm at 0.2 -- so a step moves a vertex at
   most 0.01 radians -- and steps by 0.05 until the cost falls by
   less than 1e-4. The author's research fork (``Encore/`` on Longleaf, the
   lineage of the paper's experiments) has neither regularization, steps by
   0.1 until the cost changes by less than 1e-6, as the paper states,
   differentiates ``Q`` by ENCORE's central differences instead of the
   analytic kernel derivative, and composes each step directly onto the
   warp so far rather than through a stationary velocity field. Run in
   MATLAB on the same subjects as this module (PORTING.md item 19), the fork
   undid 96% of each of three known warps, where the public code's settings
   stopped at 52% to 75%; but its warp of two real subjects folded 111
   triangles, and this module's update with the fork's direct composition
   folded 166. The stationary velocity field, which refuses a step that would
   fold, folded none, and at the paper's settings undid 93% to 97% of the
   known warps. On 24 more known warps nothing folded: the fork ended closer
   on warps up to about three degrees on average, the velocity field on
   larger ones, up to the six degrees the real pair's endpoints moved. So
   since 7 October 2026 the defaults are the paper's
   settings -- no smoothing, no clamp, a step of 0.1 and a threshold of 1e-6,
   up to 1000 iterations -- with this module's update: the analytic
   derivative and the velocity field. On small warps the fork's edge was
   where its descent stopped and how narrow its kernel is: its own
   ``spherical_heat_kernel.m`` weighs ``(2l+1)^(3/2)``, which at its 0.005 is
   as narrow as this heat kernel at 0.0027, and an absolute threshold stops a
   small warp, whose cost is small, early. So since 11 October 2026 the
   kernel is narrowed to the paper's width, ``sigma=0.0027``, and the
   threshold has a floor of 1e-5 of the starting cost: on the 24 known warps
   this module then undid 98.9% on average, the closest of every method on
   each of them, against 94.3% before and the fork's 95.8%, with nothing
   folded (PORTING.md item 19). The public code's settings stay available as
   the ``PUBLIC_*`` constants of this module.
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
14. **The tangent basis shares ENCORE's Legendre recurrence**, whose ``m = 0``
    term applies the unnormalized relation to normalized functions
    (:func:`sbci.alignment._legendre_derivatives`). The basis fields come out
    right; the divergence of every zonal field, which enters the gradient
    through ``grid.laplacian``, is too large by ``2 / (1 + 1/(l(l+1)))``.
    Found by the review of 5 October 2026; since 11 October 2026 nothing in
    the package reproduces it (PORTING.md item 21).
15. **The Karcher median can stop on its starting subject.** ``get_template``
    starts from the subject nearest the mean and weights every subject by the
    inverse of its distance to the estimate. For the starting subject that
    distance is ``acos`` of its own squared norm, which is one only to the
    rounding of the ``single`` barycentric weights: a norm 1e-10 short of one
    reads as 1.4e-5 radians, a weight near 1e5 that makes the first Weiszfeld
    step shorter than the 0.005 stopping length, and the template is that
    subject (a norm past one clips to zero and does no harm). The port
    normalizes the square-root densities to unit norm first, and a subject
    within 1e-6 radians of the estimate counts as coincident, contributing
    neither a direction nor a weight. Found by the second review of 5 October
    2026.
16. **The velocity field is smoothed component by component.** The 5%
    Laplacian smoothing of item 8 is applied to the ``(e1, e2)`` components as
    two scalars. The frame turns with longitude -- by 72 degrees between
    neighbours of a coordinate pole -- so the average over neighbours there
    mixes vectors expressed in incompatible frames: ten smoothing passes move
    the polar vertices of a 20-degree rotation by 8.4 and 8.5 degrees and
    their neighbours by up to 11.5, where vertices more than 20 degrees from a
    pole move 0.14 on average; the vertex at the pole follows only half of a
    4-degree shift of the
    endpoints around it (3.5 degrees once corrected), and 100 passes keep
    98.9% of a rotation about the coordinate z axis but 90.5% of one about a
    tilted axis. The port smooths the field with the connection
    Laplacian, which carries each neighbour's vector to the vertex by parallel
    transport before the cotangent weights combine them: it does not depend
    on the frame, keeps 97.6% of a rotation about any axis, and within 10
    degrees of the coordinate equator, where the frame is nearly parallel,
    differs from the reference's smoothing of a rotation, after one
    smoothing, by at most 1.7e-4 of the field for a rotation about the
    coordinate axis and 1.3e-3 to 1.5e-3 about others (2.8e-3 to 2.0e-2
    after 100). Nearly all of that is the reference's cotangent layout (item
    12): smoothing the frame components with the corrected weights differs
    from the connection Laplacian there by 1.3e-5 to 8.3e-5
    (``tests/reference/fourth_list_probe.py``; until 6 October 2026 this
    said 1e-4, which holds about the coordinate axis only). Smoothing the
    3-vectors per Cartesian component, the port's first correction, projects
    the neighbours' vectors onto the tangent plane instead, which drops a
    curvature term: on the unit sphere it adds the field itself to the
    connection Laplacian, so a rotation, which that maps to itself, was
    damped twice as fast (95.2%). Found by the second review of 5 October
    2026; the third review found the curvature term.
17. **A rigid rotation is held as a velocity field.** ``rotate``, which the
    rigid initialization uses, sets the field to the rotation's own rotational
    field and realizes it by scaling and squaring, whose first scaled step
    follows a great circle where the rotation moves a point along a small
    circle; six squarings double that error each time, to 0.035 degrees at a
    20-degree rotation, 1.6 at 150 and 2.3 at a half turn. And every later
    step smooths the field by 5%, rotation and all, so over the 100 iterations
    of a registration a 150-degree rotation erodes to 146.2 degrees (item 16's
    smoothing; 142.6 with the projected 3-vectors it replaced), which the
    gradient has to keep restoring. The port holds the rotation exactly,
    outside the field, and the warp is the field's flow after it; the field
    carries only the deformation. Its inverse is the same two parts in the
    other order -- the flow of the negated field, then the inverse rotation --
    so ``invert`` is exact: inverting twice gives back the warp, bit for bit.
    Found by the second review of 5 October 2026, which measured the first
    effect.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _device
from .alignment import (
    SINGLE_PRECISION_MASS,
    MeshQuery,
    SphericalGrid,
    _density_problem,
    _dot,
    _single_precision,
    _whole,
    normalize_rows,
    sparse_times_dense,
    sphere_exp_map,
    sphere_log_map,
    triangles_fold,
    voronoi_areas,
)
from .errors import MissingDataError
from .smoothing import _legendre_with_derivative

DEFAULT_SIGMA = 0.0027
"""Kernel bandwidth: at 0.0027 this heat kernel is as narrow as the kernel the paper's
experiments ran, the fork's ``kde/spherical_heat_kernel.m`` at its 0.005, whose normalized
Legendre functions weigh ``(2l+1)^(3/2)``: cut at 12.4 degrees against its 12.0, 32 nonzeros a
row against its 29, its rows spread 5.6 degrees against 6.0 (module docstring, item 8)."""

PUBLIC_SIGMA = 0.005
"""The public code's bandwidth for this kernel: cut at 21.5 degrees, 89 nonzeros a row, its
rows spread 8.1 degrees. The ``PUBLIC_*`` values are the public code's settings, kept for
anyone who wants them; pass them to :func:`endpoints_align` or :class:`ConSEAL`."""

DEFAULT_KERNEL_DEGREE = 30
"""Harmonic degree the reference truncates the heat kernel at."""

DEFAULT_WARP_ORDER = 15
"""Harmonic order of the velocity field's tangent basis (``Encore(..., 15, ...)``)."""

DEFAULT_DELTA = 0.1
"""Gradient-descent step size: the paper's (``Encore(grid, grid, 15, 0.1, 1000, 1e-6)``)."""

DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_THRESHOLD = 1e-6
"""Stop when the cost falls by less than this between iterations: the paper's."""

DEFAULT_RELATIVE_THRESHOLD = 1e-5
"""A floor under the threshold, as a share of the cost a registration starts from: the
descent stops when a step's fall is below both. An absolute threshold alone stops a small
warp early, its cost being small (module docstring, item 8); the public code has none."""

STEP_CLAMP = float("inf")
"""Cap on the gradient's largest per-vertex norm, before ``delta``: none, as in the paper."""

VISCOSITY = 0.0
"""Share of the cotangent Laplacian taken from the velocity field each step: none (the paper)."""

PUBLIC_DELTA = 0.05
"""The public code's step (``Encore(..., 15, 0.05, 100, 1e-4)``)."""

PUBLIC_MAX_ITERATIONS = 100
PUBLIC_THRESHOLD = 1e-4
"""The public code's stopping change in cost."""

PUBLIC_STEP_CLAMP = 0.2
"""The public code's cap on the gradient's norm (``compute_step_size``): a step of ``0.01``."""

PUBLIC_VISCOSITY = 0.05
"""The public code's smoothing of the velocity field (``SphericalWarp.compose``), and
:class:`StationaryWarp`'s own default, as the port of that class."""

PUBLIC_RELATIVE_THRESHOLD = 0.0
"""The public code's floor under the threshold: none."""

SQUARINGS = 6
"""Scaling-and-squaring steps in the exponential map."""

KERNEL_EPSILON = 1e-3
"""Where the heat kernel is cut off, on the product ``K(x) K(1)``."""


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
        for name, flags in (("hemisphere_in", hemisphere_in), ("hemisphere_out", hemisphere_out)):
            raw = np.asarray(flags).ravel()
            # checked before the cast: int8 wraps 256 to 0 and truncates 1.9 to 1
            if raw.size and not np.isin(raw, (0, 1)).all():
                raise ValueError(f"{name} must be 0 (left) or 1 (right)")
            setattr(self, name, raw.astype(np.int8))
        if self.hemisphere_in.shape != self.hemisphere_out.shape:
            raise ValueError("hemisphere flags of the two endpoints differ in length")
        self.n_streamlines = int(self.hemisphere_in.size)
        self.e1 = np.vstack([lh_grid.e1, rh_grid.e1])
        self.e2 = np.vstack([lh_grid.e2, rh_grid.e2])
        self._query = {
            0: MeshQuery(lh_grid.vertices, lh_grid.faces),
            1: MeshQuery(rh_grid.vertices, rh_grid.faces),
        }
        n = self.n_streamlines
        self.weights_in = np.zeros((n, 3))
        self.weights_out = np.zeros((n, 3))
        self.index_in = np.zeros((n, 3), dtype=np.int64)
        self.index_out = np.zeros((n, 3), dtype=np.int64)
        self.face_in = np.zeros(n, dtype=np.int64)
        self.face_out = np.zeros(n, dtype=np.int64)
        self.device = "cpu"
        self.commit()

    _ARRAYS = (
        "hemisphere_in",
        "hemisphere_out",
        "weights_in",
        "index_in",
        "face_in",
        "weights_out",
        "index_out",
        "face_out",
        "e1",
        "e2",
    )

    def _to_device(self, device) -> EndpointConnectome:
        """A copy whose endpoints are on ``device``: CuPy arrays for ``"gpu"``, NumPy for ``"cpu"``.

        :class:`ConSEAL` makes one on the GPU to register with ``device="gpu"``, and
        hands back one on the CPU; the grids and their search trees stay shared.
        Its methods then compute where its endpoints are.
        """
        _device.check(device)
        clone = object.__new__(EndpointConnectome)
        clone.__dict__.update(self.__dict__)
        clone.device = device
        for name in self._ARRAYS:
            setattr(clone, name, _device.copy_to(getattr(self, name), device))
        clone._original = tuple(_device.copy_to(a, device) for a in self._original)
        # its own searches: the tables the GPU makes for them go when the clone does
        clone._query = {side: query._without_tables() for side, query in self._query.items()}
        return clone

    # -- construction --------------------------------------------------------

    @classmethod
    def from_points(cls, lh_grid, rh_grid, points_in, points_out, hemisphere_in, hemisphere_out):
        """Locate unit-sphere endpoint coordinates on the grids (``SConcon``'s constructor).

        The coordinates must be in the grids' own frame; the bundled ico4
        sphere is not rotated for this module, so positions rebuilt by
        :func:`sbci.smoothing.endpoint_positions` qualify.
        """
        self = cls(lh_grid, rh_grid, hemisphere_in, hemisphere_out)
        points_in = np.asarray(points_in, dtype=np.float64)
        points_out = np.asarray(points_out, dtype=np.float64)
        expected = (self.n_streamlines, 3)
        if points_in.shape != expected or points_out.shape != expected:
            raise ValueError("endpoint coordinates must be (n_streamlines, 3)")
        for name, points in (("points_in", points_in), ("points_out", points_out)):
            norms = np.linalg.norm(points, axis=1)
            if not (np.isfinite(norms).all() and (norms > 0).all()):
                bad = int((~np.isfinite(norms) | (norms <= 0)).sum())
                raise ValueError(
                    f"{name} has {bad} endpoint(s) that are not finite or are at the origin, "
                    "which a triangle would take in anyway"
                )
        points_in, points_out = normalize_rows(points_in), normalize_rows(points_out)
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
        if not endpoints.n_per_hemi == lh_grid.n_vertices == rh_grid.n_vertices:
            raise ValueError(
                f"the endpoints are on a grid of {endpoints.n_per_hemi} vertices per hemisphere, "
                f"these grids have {lh_grid.n_vertices} and {rh_grid.n_vertices}: their triangle "
                "indices name other triangles here"
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
            if not np.isfinite(bary).all():
                bad = int((~np.isfinite(bary)).any(axis=1).sum())
                raise ValueError(
                    f"{bad} endpoint(s) have barycentric weights that are not finite, which "
                    "would make the density NaN"
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
                    w, idx, f = self._query[side].query_faces(points[mask])
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
        xp = _device.array_module(self.device)
        vertices = xp.asarray(np.vstack([self.lh_grid.vertices, self.rh_grid.vertices]))
        return _points(w_in, i_in, vertices), _points(w_out, i_out, vertices)

    # -- the density -------------------------------------------------------------

    def adjacency(self) -> np.ndarray:
        """Endpoint mass spread onto vertex pairs: symmetric and summing to one.

        ``build_adjacency.cpp`` followed by ``(A + A') / (2N)``, accumulated in
        float64 rather than the reference's float32.
        """
        return self._sparse_adjacency().toarray()

    def _sparse_adjacency(self):
        """:meth:`adjacency` as a CSR matrix: 18 entries per streamline at most."""
        xp = _device.array_module(self.device)
        sparse = _device.sparse_module(self.device)

        n = self.n_vertices
        rows = xp.repeat(self.index_in, 3, axis=1).ravel()
        cols = xp.tile(self.index_out, (1, 3)).ravel()
        vals = (self.weights_in[:, :, None] * self.weights_out[:, None, :]).reshape(-1)
        if self.device == "gpu":  # A + A^T at once, its sums in an order every run repeats
            both = _device.summed_csr(
                xp.concatenate([rows, cols]),
                xp.concatenate([cols, rows]),
                xp.concatenate([vals, vals]),
                (n, n),
            )
            return both / (2.0 * max(self.n_streamlines, 1))
        adjacency = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        return (adjacency + adjacency.T) / (2.0 * max(self.n_streamlines, 1))

    def evaluate(self, kernel, derivative=None):
        """The smooth connectome ``F = K^T A K``, clamped non-negative.

        With ``derivative`` (a :class:`KernelDerivative` from
        :meth:`HeatKernelBuilder.compute`) also returns the derivatives of
        ``F(a, c)`` with respect to moving the evaluation point ``a`` along its
        frame, ``(D_e1, D_e2)``. The reference forms ``dK A K^T + (dK A K^T)^T``
        instead (module docstring, items 1 and 2).
        """
        xp = _device.array_module(self.device)
        kernel = _as_sparse(kernel, self.device)
        kernel_t = _transposed(kernel)
        adjacency = self._sparse_adjacency()
        n = self.n_vertices
        # A has at most 18 entries per streamline, so with few streamlines A K
        # stays sparse and two sparse products beat four passes of the kernel
        # over a dense n x n matrix; with a subject's million streamlines A is
        # dense and the dense route is the cheaper one.
        few = adjacency.nnz * kernel.nnz < SPARSE_PRODUCT_FRACTION * n**3
        if few:
            ak = (adjacency @ kernel).tocsc()  # A K, since A is symmetric
            connectome = (kernel_t @ ak).toarray()
        else:
            dense = adjacency.toarray()
            # (K^T A)^T is A K but Fortran-ordered; make it contiguous once
            # rather than letting each sparse product copy it.
            ak = xp.ascontiguousarray(_times(kernel_t, dense).T)
            connectome = _times(kernel_t, ak)
        xp.maximum(connectome, 0.0, out=connectome)
        if derivative is None:
            return connectome

        # sum_axis e1_axis(a) (dK_axis^T A K)(a, c) = (M1^T A K)(a, c) with the frame
        # folded into the sparse derivative: two sparse products instead of three,
        # and no dense per-axis parts.
        sparse = _device.sparse_module(self.device)

        axes = (derivative.x, derivative.y, derivative.z)
        if self.device == "gpu":
            axes = tuple(_as_sparse(part, self.device) for part in axes)
        else:  # as given, whatever their format, but on the CPU
            axes = tuple(part.get() if _device.on_device(part) else part for part in axes)
        m1 = sum(d @ sparse.diags(e) for d, e in zip(axes, self.e1.T, strict=True))
        m2 = sum(d @ sparse.diags(e) for d, e in zip(axes, self.e2.T, strict=True))
        m1, m2 = _transposed(m1), _transposed(m2)
        if few:
            return connectome, (m1 @ ak).toarray(), (m2 @ ak).toarray()
        return connectome, _times(m1, ak), _times(m2, ak)

    def q_transform(self, kernel, derivative=None):
        """The square-root density ``Q = sqrt(F)``, with derivatives by the chain rule.

        ``F`` is clamped at zero, so its square root has derivative zero
        wherever it was clamped. The reference divides by ``2 max(Q, 1e-15)``
        there instead (module docstring, item 13).
        """
        xp = _device.array_module(self.device)
        if derivative is None:
            return xp.sqrt(self.evaluate(kernel))
        connectome, d_e1, d_e2 = self.evaluate(kernel, derivative)
        q = xp.sqrt(connectome)
        if self.device == "gpu":  # CuPy's divide takes no where=
            scale = xp.where(q > 0, 0.5 / xp.where(q > 0, q, 1.0), 0.0)
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
        xp = _device.array_module(self.device)
        vertices = xp.asarray(np.vstack([lh_warp.vertices, rh_warp.vertices]))
        w_in, i_in, _, w_out, i_out, _ = self._original
        points_in = _points(w_in, i_in, vertices)
        points_out = _points(w_out, i_out, vertices)
        self._locate(points_in, points_out)
        return points_in, points_out


def _points(weights, indices, vertices):
    """Unit-sphere points from barycentric weights over rows of ``vertices``.

    The corners are added as ``(w0 v0 + w1 v1) + w2 v2`` and the rows normalized
    by squares added in the same order, on either device: NumPy's einsum adds
    them so on x86 but fuses them on other processors, so the GPU finds the
    CPU's points whatever the CPU.
    """
    xp = _device.module_of(weights)
    corners = vertices[indices]
    points = weights[:, 0, None] * corners[:, 0] + weights[:, 1, None] * corners[:, 1]
    points += weights[:, 2, None] * corners[:, 2]
    norms = xp.sqrt(_dot(points, points, xp))[:, None]
    return xp.where(norms > 0, points / xp.where(norms > 0, norms, 1.0), 0.0)


def _transposed(matrix):
    """``matrix.T``, and on the GPU a CSR matrix of its own, for the GPU's row-wise product."""
    return matrix.T.tocsr() if _device.on_device(matrix) else matrix.T


def _times(matrix, dense):
    """A sparse matrix times a dense one: in threads on the CPU, row by row on the GPU."""
    if _device.on_device(dense):
        return _device.rows_times_dense(matrix, dense)
    return sparse_times_dense(matrix, dense)


def _sent(kernel, derivative, device):
    """The kernel and its derivative as CSR matrices on ``device``, made once for many products."""
    if device == "cpu":
        return kernel, derivative
    if derivative is not None:
        derivative = KernelDerivative(
            *(_as_sparse(part, device) for part in (derivative.x, derivative.y, derivative.z))
        )
    return _as_sparse(kernel, device), derivative


def _as_sparse(kernel, device="cpu"):
    """``kernel`` as a CSR matrix on ``device``."""
    if _device.on_device(kernel):
        if not hasattr(kernel, "tocsr"):  # a dense CuPy array
            kernel = _device.sparse_module("gpu").csr_matrix(kernel.astype(np.float64))
        return kernel.tocsr() if device == "gpu" else kernel.get().tocsr()
    from scipy import sparse

    if sparse.issparse(kernel):
        kernel = sparse.csr_matrix(kernel)
    else:
        kernel = sparse.csr_matrix(np.asarray(kernel, dtype=np.float64))
    return _device.sparse_to_device(kernel, device)


# --- the kernel ---------------------------------------------------------------


@dataclass(frozen=True)
class KernelDerivative:
    """Cartesian derivatives of a kernel matrix, one sparse ``(P, P)`` matrix per axis.

    Rows are source vertices and columns evaluation vertices, as in the kernel
    itself; the derivative is with respect to the evaluation vertex
    (:meth:`HeatKernelBuilder.compute`).
    """

    x: object
    y: object
    z: object


def _bandwidth(sigma) -> float:
    """``sigma`` as a heat-kernel bandwidth, once it is one: finite and above 0.

    At 0 or below the series does not decay, and the cutoff keeps a vertex alone (the
    identity kernel, under which no registration moves); NaN or inf built a kernel
    that failed only later, blamed on the template.
    """
    value = float(sigma)
    if not 0 < value < np.inf:
        raise ValueError(f"sigma must be a finite bandwidth above 0; got {sigma}")
    return value


def _kernel_degree(degree) -> int:
    """``degree`` as the kernel's truncation degree, once it is one: a whole number of 1 or more.

    At 0 the kernel is constant and every density uniform; a fraction was truncated
    without a word.
    """
    if not (_whole(degree) and float(degree) >= 1):
        raise ValueError(f"kernel_degree must be a whole number of 1 or more; got {degree}")
    return int(degree)


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
        self.degree = _kernel_degree(degree)
        self.epsilon = float(epsilon)
        self._cosines = [np.clip(g.vertices @ g.vertices.T, -1.0, 1.0) for g in (lh_grid, rh_grid)]
        self._scan = np.linspace(1.0, -1.0, 5000)

    def weights(self, sigma: float) -> np.ndarray:
        """``(2l+1) exp(-l(l+1) sigma)`` for ``l = 0..degree``."""
        degree = np.arange(self.degree + 1)
        return (2 * degree + 1) * np.exp(-degree * (degree + 1) * _bandwidth(sigma))

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

    def compute(self, sigma: float = DEFAULT_SIGMA, derivative: bool = True):
        """The kernel ``K`` and, unless ``derivative=False``, its :class:`KernelDerivative`.

        The derivative of ``K~(i, a) = K(i, a) / s_i`` is taken with respect to
        the *evaluation* vertex ``a`` with the source normalization ``s_i`` held
        fixed, which is what differentiating ``F = K~^T A K~`` in its first
        argument needs. The reference differentiates with respect to the *row*
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
            hemispheres = zip((self.lh_grid, self.rh_grid), raw_derivatives, row_sums, strict=True)
            for grid, d_raw, s in hemispheres:
                coordinate = grid.vertices[:, axis]
                # d/d(evaluation vertex a) of K(i, a) / s_i = P'(x_i . x_a) x_i / s_i.
                blocks.append(sparse.diags(coordinate / s) @ d_raw)
            parts.append(sparse.block_diag(blocks, format="csr"))
        return kernel, KernelDerivative(*parts)

    def cross_validate(self, connectome: EndpointConnectome, sigmas):
        """Leave-one-out log-likelihood over candidate bandwidths; returns ``(best, scores)``.

        Port of ``Kernel.cross_validate_sigma``. The reference subtracts an
        approximate self-term, ``K(t_in, t_in) K(t_out, t_out)' / (M - 1)``
        (module docstring, item 7); the port removes each streamline's actual
        contribution to the symmetrized density, so the score is an exact
        leave-one-out.
        """
        sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
        m = connectome.n_streamlines
        if not m:
            raise ValueError(
                "the connectome has no streamlines on this grid, so no bandwidth has a "
                "likelihood to score"
            )
        adjacency = connectome.adjacency()
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
            # the streamline's own term of K~^T A K~ on its two triangles, both
            # orientations of the symmetrized adjacency included
            left = np.einsum("nij,ni->nj", k_in, w_in)
            right = np.einsum("nij,ni->nj", k_out, w_out)
            k_oi = kernel[i_out[:, :, None], i_in[:, None, :]]
            k_io = kernel[i_in[:, :, None], i_out[:, None, :]]
            cross_in = np.einsum("nij,ni->nj", k_oi, w_out)
            cross_out = np.einsum("nij,ni->nj", k_io, w_in)
            own = (
                left[:, :, None] * right[:, None, :] + cross_in[:, :, None] * cross_out[:, None, :]
            ) / (2.0 * m)
            values = np.einsum("ni,nij,nj->n", w_in, block - own, w_out)
            scores[s] = float(np.mean(np.log(np.maximum(values, np.finfo(float).eps))))
        return float(sigmas[int(np.argmax(scores))]), scores


# --- the warp -------------------------------------------------------------------


def cotangent_laplacian(vertices, faces):
    """The cotangent Laplacian ``L = D - W`` of a mesh, sparse.

    Each edge is weighted by half the cotangent of the angle opposite it. The
    reference's ``cot_matrix`` weights it with an *adjacent* angle instead
    (module docstring, item 12).
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
    # the angle at i1 is opposite edge (i2, i3), and so on around the triangle
    rows = np.concatenate([i3, i1, i1, i2, i2, i3])
    cols = np.concatenate([i1, i3, i2, i1, i3, i2])
    vals = 0.5 * np.concatenate([cot_at_2, cot_at_2, cot_at_3, cot_at_3, cot_at_1, cot_at_1])
    n = vertices.shape[0]
    weights = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(weights.sum(axis=1)).ravel()
    return (sparse.diags(degree) - weights).tocsr()


def _transport(tangent, origin, destination):
    """Parallel transport on the sphere, the reference's closed form.

    The reference ENCORE writes the same transport differently; the two agree to
    1e-13, and ``tests/reference/encore_order_probe.py`` keeps ENCORE's for its
    order of composition.
    """
    pq = (origin * destination).sum(axis=1, keepdims=True)
    vq = (tangent * destination).sum(axis=1, keepdims=True)
    denominator = 1.0 + pq
    safe = np.where(denominator < 1e-8, 1.0, denominator)
    moved = tangent - (vq / safe) * (origin + destination)
    bad = denominator[:, 0] < 1e-8
    moved[bad] = tangent[bad]
    return moved


class StationaryWarp:
    """A diffeomorphism of one hemisphere's sphere, as a stationary velocity field.

    Port of the reference's ``SphericalWarp``. The warp is the unit-time flow
    of a tangent field ``velocity`` (``(P, 2)`` in the ``(e1, e2)`` frame),
    computed by scaling and squaring with barycentric interpolation and
    parallel transport. Composing is addition of velocity fields, followed by
    the reference's 5% Laplacian smoothing (module docstring, item 8), with
    the connection Laplacian, which compares neighbouring vectors by parallel
    transport, rather than on the frame components (item 16).

    A rigid rotation set by :meth:`rotate` is held exactly, as ``rigid``, and
    the warp is the field's flow after it: the field then carries only the
    deformation, which the flow computes accurately and the smoothing is meant
    for (item 17). ``rigid`` is the identity until :meth:`rotate` is called,
    and the warp is then the flow alone, exactly as before.
    """

    def __init__(
        self,
        grid: SphericalGrid,
        viscosity: float = PUBLIC_VISCOSITY,
        squarings: int = SQUARINGS,
    ):
        """The identity warp of ``grid``.

        The faces must be oriented outward, since that is what the fold test
        assumes; an inward mesh would refuse every step.
        """
        self._rigid = np.eye(3)
        self._rigid_last = False  # the rotation acts before the flow, or after it once inverted
        self._rigid_query = None  # weights, indices and the rotated vertices, once rigid is set
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
        self._base_areas = grid.areas  # the same Voronoi areas the grid already holds
        self._query = MeshQuery(self.base, self.faces)
        self._laplacian = cotangent_laplacian(self.base, self.faces)
        # the connection Laplacian's off-diagonal entries, along which the smoothing
        # carries each neighbour's vector to the vertex (module docstring, item 16)
        entries = self._laplacian.tocoo()
        off_diagonal = entries.row != entries.col
        self._edges = (
            entries.row[off_diagonal],
            entries.col[off_diagonal],
            entries.data[off_diagonal],
        )
        self._diagonal = self._laplacian.diagonal()

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
        clone._rigid = self._rigid.copy()
        return clone

    def exponential(self, velocity=None) -> np.ndarray:
        """Flow the base vertices along ``velocity`` for unit time (``exp_svf``)."""
        velocity = self.velocity if velocity is None else np.asarray(velocity, dtype=np.float64)
        scaled = velocity / (2**self.squarings)
        tangent = self.e1 * scaled[:, :1] + self.e2 * scaled[:, 1:2]
        current = normalize_rows(sphere_exp_map(self.base, tangent))
        n = self.n_vertices
        for _ in range(self.squarings):
            weights, indices, _ = self._query.query_faces(current)
            displacement = sphere_log_map(self.base, current)
            moved = _transport(
                displacement[indices].reshape(-1, 3),
                self.base[indices].reshape(-1, 3),
                np.repeat(current, 3, axis=0),
            ).reshape(n, 3, 3)
            step = (weights[:, :, None] * moved).sum(axis=1)
            current = normalize_rows(sphere_exp_map(current, step))
        return current

    def _images(self, velocity=None) -> np.ndarray:
        """Where the warp sends the base vertices: the flow of ``velocity`` after ``rigid``.

        Without a rigid part that is the flow itself. With one, the flow is
        carried to the rotated vertices as each squaring carries it, by
        interpolating its displacements transported to the point, so a zero
        field leaves the rotation exact.
        """
        if self._rigid_query is None:
            return self.exponential(velocity)
        if self._rigid_last:  # an inverted warp: the flow first, then the rotation, exactly
            return normalize_rows(self.exponential(velocity) @ self._rigid.T)
        weights, indices, rotated = self._rigid_query
        velocity = self.velocity if velocity is None else np.asarray(velocity, dtype=np.float64)
        if not np.any(velocity):
            return rotated.copy()
        displacement = sphere_log_map(self.base, self.exponential(velocity))
        moved = _transport(
            displacement[indices].reshape(-1, 3),
            self.base[indices].reshape(-1, 3),
            np.repeat(rotated, 3, axis=0),
        ).reshape(self.n_vertices, 3, 3)
        step = (weights[:, :, None] * moved).sum(axis=1)
        return normalize_rows(sphere_exp_map(rotated, step))

    @property
    def rigid(self) -> np.ndarray:
        """The warp's rigid rotation, the identity when it has none.

        Read-only as an array. Assigning a rotation to it keeps the velocity
        field and moves the vertices to match, where :meth:`rotate` starts the
        warp afresh at the rotation. (Until 6 October 2026 it was a plain
        attribute, and assigning to it changed nothing.)
        """
        view = self._rigid.view()
        view.setflags(write=False)
        return view

    @property
    def rigid_after(self) -> bool:
        """Whether the rigid rotation acts after the flow, as it does once the warp is inverted.

        A warp is the flow of its velocity field after its rigid rotation; its
        inverse holds the same parts in the other order. :meth:`invert` turns
        this over, and :meth:`rotate` starts the warp afresh, the rotation first.
        """
        return self._rigid_last

    @rigid.setter
    def rigid(self, rotation) -> None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
            or np.linalg.det(rotation) <= 0
        ):
            raise ValueError("rigid takes a 3 x 3 rotation: orthonormal, with determinant +1")
        self._set_rigid(rotation)
        self.vertices = self._images()

    def _set_rigid(self, rotation) -> None:
        """Hold ``rotation`` as the rigid part, and where it sends the base vertices."""
        self._rigid = np.asarray(rotation, dtype=np.float64).copy()
        if np.array_equal(self._rigid, np.eye(3)):
            self._rigid_query = None
            return
        rotated = normalize_rows(self.base @ self._rigid.T)
        weights, indices, _ = self._query.query_faces(rotated)
        self._rigid_query = (weights, indices, rotated)

    def compose(self, displacement) -> bool:
        """Add a tangent displacement to the velocity field and re-flow; returns whether applied.

        An update that would fold a triangle is refused. The reference keeps
        the refused displacement in the velocity field regardless (module
        docstring, item 3) and smooths the frame components as scalars (item
        16).
        """
        if self._rigid_last and self._rigid_query is not None:
            raise ValueError(
                "this warp was inverted, so its rotation acts after the flow, and a "
                "displacement added to the flow would act in the frame before it; compose "
                "before inverting"
            )
        displacement = np.asarray(displacement, dtype=np.float64)
        velocity = self.velocity + displacement
        if self.viscosity:
            velocity = self._smooth(velocity)
        candidate = self._images(velocity)
        if triangles_fold(candidate, self.faces):
            self.rejected += 1
            warnings.warn(
                "skipping a warp update that would fold triangles", RuntimeWarning, stacklevel=2
            )
            return False
        self.velocity = velocity
        self.vertices = candidate
        return True

    def _smooth(self, velocity) -> np.ndarray:
        """Remove ``viscosity`` of the field's connection Laplacian from it.

        The cotangent weights combine each neighbour's vector carried to the
        vertex by parallel transport, so the result does not depend on the
        frame and is tangent there. Smoothing the 3-vectors per Cartesian
        component would project the neighbours' vectors onto the tangent plane
        instead, which drops a curvature term and damps a rotation twice as
        fast. The reference smooths the ``(e1, e2)`` components as two scalars,
        which mixes frames where they turn sharply (module docstring, item 16).
        """
        ambient = self.e1 * velocity[:, :1] + self.e2 * velocity[:, 1:]
        rows, cols, weights = self._edges
        carried = _transport(ambient[cols], self.base[cols], self.base[rows])
        laplacian = self._diagonal[:, None] * ambient
        np.add.at(laplacian, rows, weights[:, None] * carried)
        ambient = ambient - self.viscosity * laplacian
        return np.stack([(ambient * self.e1).sum(1), (ambient * self.e2).sum(1)], axis=1)

    def invert(self) -> StationaryWarp:
        """Invert the warp in place, exactly: negate the velocity field and re-flow (``invert``).

        A rigid part ``R`` inverts too. The warp is the flow of ``v`` after
        ``R``; its inverse is ``R'`` after the flow of ``-v``, the same parts
        in the other order, which the warp records, so nothing is
        interpolated and inverting twice gives back the warp, bit for bit.
        (Until 6 October 2026 the field was carried to the rotated frame by
        interpolation instead, which changed it over a double inversion by
        0.1% of its size on ico4 and 0.4% on ico3 in
        ``tests/reference/fourth_list_probe.py``, 2.8% in the review's case.)
        """
        self.velocity = -self.velocity
        if self._rigid_query is not None:
            self._rigid_last = not self._rigid_last
            self._set_rigid(self._rigid.T)
        self.vertices = self._images()
        return self

    def rotate(self, rotation) -> StationaryWarp:
        """Set the warp to a rigid rotation, held exactly (``rotate``).

        The rotation becomes the warp's rigid part and the velocity field is
        cleared, so the vertices are the rotated vertices to rounding and later
        steps deform after the rotation without eroding it (module docstring,
        item 17). The reference sets the field to the rotation's own rotational
        field instead, whose flow is 1.6 degrees off at 150 degrees and which
        every later smoothing shrinks.
        """
        from scipy.spatial.transform import Rotation

        # The nearest proper rotation, whatever rounding the caller's matrix carries.
        rotation = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64))
        self._rigid_last = False
        self._set_rigid(rotation.as_matrix())
        self.velocity = np.zeros_like(self.velocity)
        self.vertices = self._images()
        return self

    def apply(self, points) -> np.ndarray:
        """Where the warp sends arbitrary points of this hemisphere's sphere."""
        points = normalize_rows(np.asarray(points, dtype=np.float64))
        weights, indices, _ = self._query.query_faces(points)
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
    FreeSurfer's ico4 sphere (1.7e-4 off an exact icosphere, about a
    four-hundredth of its shortest edge) and refuses a mesh jittered by more.
    ``permutations[g]`` follows the reference's convention: the function
    transported by rotation ``g`` is ``Q[perm][:, perm]``.
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


def spin_shells():
    """Turns about ``+z`` to go with :func:`rotation_shells`: what its caps cannot reach.

    Every rotation in a cap takes ``z`` to a nearby direction, about an axis
    perpendicular to ``z``, so no shell turns anything about ``z`` itself: on
    ico3 an 8-degree rotation about ``z`` was left 9.4 and 8.0 degrees off,
    one about ``x`` 0.07, one about a tilted axis 3.3 and 3.0 (PORTING.md item
    18; until October 2026). Shell for
    shell, these are the turns about ``z`` within the cap's radius, spaced as
    the cap's directions are (the mean edge of its icosphere), plus the
    identity; :meth:`ConSEAL._rigid` tries them after each shell's directions.
    Returns ``[rotations (n, 3, 3), ...]``.
    """
    shells = []
    for radius, subdivision, _keep in [
        (20.0, 4, 5),
        (10.0, 4, 3),
        (5.0, 5, 1),
        (2.5, 6, 1),
        (1.0, 7, 1),
    ]:
        vertices, faces = icosphere(subdivision)
        edges = vertices[faces[:, 0]] * vertices[faces[:, 1]]
        spacing = float(np.degrees(np.arccos(np.clip(edges.sum(axis=1), -1.0, 1.0)).mean()))
        steps = int(radius // spacing)
        rotations = [np.eye(3)]
        for angle in np.radians(spacing * np.arange(-steps, steps + 1)):
            if angle == 0.0:
                continue
            cos, sin = np.cos(angle), np.sin(angle)
            rotations.append(np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]))
        shells.append(np.stack(rotations))
    return shells


# --- registration ---------------------------------------------------------------


@dataclass
class EndpointWarp:
    """One subject's warp of both hemispheres, exported from :class:`StationaryWarp`.

    The vertices are the whole map. Each hemisphere's warp is the flow of its
    velocity field after the rigid rotation ``lh_rigid`` / ``rh_rigid`` -- or
    the rotation after the flow where ``lh_rigid_after`` / ``rh_rigid_after``
    is set, as for an inverted warp (:attr:`StationaryWarp.rigid_after`). The
    rotation is the identity unless the registration began with
    ``init_rotation=True`` (module docstring, item 17); a file written before
    the rigid part, or its order, was kept loads with identities, rotation first.
    """

    lh_vertices: np.ndarray
    lh_velocity: np.ndarray
    lh_jacobian: np.ndarray
    rh_vertices: np.ndarray
    rh_velocity: np.ndarray
    rh_jacobian: np.ndarray
    lh_rigid: np.ndarray = field(default_factory=lambda: np.eye(3))
    rh_rigid: np.ndarray = field(default_factory=lambda: np.eye(3))
    lh_rigid_after: bool = False
    rh_rigid_after: bool = False

    def save(self, path) -> Path:
        """Write the warp to ``.npz``; returns the path written, suffix included."""
        path = Path(path)
        if not path.name.endswith(".npz"):
            path = path.with_name(path.name + ".npz")  # what np.savez_compressed appends anyway
        np.savez_compressed(path, **dict(self.__dict__))
        return path

    @classmethod
    def load(cls, path) -> EndpointWarp:
        """Read a warp written by :meth:`save`."""
        with np.load(Path(path)) as data:
            return cls(
                **{k: bool(data[k]) if k.endswith("_after") else data[k] for k in data.files}
            )


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
        Gradient step, iteration cap, stopping change in cost, the cap on the
        gradient's largest per-vertex norm (so a step moves a vertex at most
        ``delta * step_clamp`` radians), and Laplacian smoothing of the
        velocity field. Left out, they are the paper's -- 0.1, 1000, 1e-6, no
        clamp and no smoothing; the public code's, 0.05, 100, 1e-4, 0.2 and
        5%, are the ``PUBLIC_*`` constants (module docstring, item 8).
    area_weighted
        Weight the cost and its gradient with the Voronoi areas, as the
        paper's integral does, in units of their mean, so that the threshold
        means what it does for the plain sum. Off by default to match the
        reference (item 5).
    relative_threshold
        A floor under ``threshold``, as a share of the cost a registration
        starts from: the descent stops when a step's fall is below both, or
        when the cost does not fall at all. Left out, 1e-5; 0 turns it off,
        as the public code has it.
    device
        ``"cpu"``, or ``"gpu"`` to hold the densities and the endpoints on a
        CUDA GPU through CuPy (:mod:`sbci._device`), where the rigid search
        scores its rotations too. The warps, their gradients and the 3 x 3
        rotations stay on the CPU, and :meth:`template` and
        :meth:`register` hand back what they do on the CPU: the template to
        rounding, a registration as closely as the CPU's own does when an input
        moves by one unit in the last place (PORTING.md item 26).

    A setting that cannot register -- a step or clamp of 0, a negative
    threshold, NaN -- is refused rather than run as a registration that
    never moves.
    """

    def __init__(
        self,
        lh_grid: SphericalGrid,
        rh_grid: SphericalGrid,
        delta: float | None = None,
        max_iterations: int | None = None,
        threshold: float | None = None,
        step_clamp: float | None = None,
        viscosity: float | None = None,
        area_weighted: bool = False,
        relative_threshold: float | None = None,
        device: str = "cpu",
    ):
        """Fix the grids and the descent settings."""
        self.lh_grid, self.rh_grid = lh_grid, rh_grid
        self.device = _device.check(device)
        self._xp = _device.array_module(self.device)

        def chosen(value, paper):
            return paper if value is None else value

        self.delta = float(chosen(delta, DEFAULT_DELTA))
        iterations = float(chosen(max_iterations, DEFAULT_MAX_ITERATIONS))
        self.threshold = float(chosen(threshold, DEFAULT_THRESHOLD))
        self.step_clamp = float(chosen(step_clamp, STEP_CLAMP))
        self.viscosity = float(chosen(viscosity, VISCOSITY))
        self.relative_threshold = float(chosen(relative_threshold, DEFAULT_RELATIVE_THRESHOLD))
        # NaN fails every one of these; each setting refused left the warps at the identity,
        # or stopped the descent after a step, as though the registration had converged
        for name, value, ok, needs in (
            ("delta", self.delta, 0 < self.delta < np.inf, "a finite step above 0"),
            (
                "max_iterations",
                iterations,
                _whole(iterations),
                "a whole number of iterations, 0 or more",
            ),
            ("threshold", self.threshold, 0 <= self.threshold < np.inf, "a finite fall, 0 or more"),
            ("step_clamp", self.step_clamp, self.step_clamp > 0, "above 0 (inf for no clamp)"),
            ("viscosity", self.viscosity, 0 <= self.viscosity < np.inf, "finite, 0 or more"),
            (
                "relative_threshold",
                self.relative_threshold,
                self.relative_threshold >= 0,
                "a share of the starting cost, 0 or more",
            ),
        ):
            if not ok:
                raise ValueError(f"{name} must be {needs}; got {value}")
        self.max_iterations = int(iterations)
        self.area_weighted = bool(area_weighted)
        # In units of the mean area: the weighted cost then sits on the plain sum's scale,
        # where the stopping threshold is meant; raw areas shrank it ~40,000-fold on ico4.
        areas = np.concatenate([lh_grid.areas, rh_grid.areas])
        self._area_unit = float(areas.mean())
        self._areas = _device.to_device(areas / self._area_unit, self.device)

    def new_warps(self):
        """A pair of identity warps with this engine's viscosity."""
        return (
            StationaryWarp(self.lh_grid, self.viscosity),
            StationaryWarp(self.rh_grid, self.viscosity),
        )

    # -- cost --------------------------------------------------------------------

    def _here(self, connectome: EndpointConnectome) -> EndpointConnectome:
        """``connectome`` where this engine computes: itself, or a copy sent there.

        An :class:`EndpointConnectome` on the other device is copied to this one, whichever
        way. On the CPU anything else with a ``q_transform`` is taken as it is; the GPU
        refuses it.
        """
        if getattr(connectome, "device", None) == self.device:
            return connectome
        if isinstance(connectome, EndpointConnectome):
            return connectome._to_device(self.device)
        if self.device == "cpu":
            return connectome
        raise TypeError(
            "device='gpu' registers EndpointConnectome objects, whose endpoints it copies "
            f"to the GPU; got {type(connectome).__name__}"
        )

    def _check_grids(self, connectome, what: str) -> None:
        """Refuse an :class:`EndpointConnectome` built on other grids than this engine's."""
        if isinstance(connectome, EndpointConnectome) and not _on_grids(
            connectome, self.lh_grid, self.rh_grid
        ):
            raise ValueError(f"{what} {_OTHER_GRIDS}")

    def cost(self, difference: np.ndarray) -> float:
        """Squared mismatch of two square-root densities, ``sum (Q1 - Q2)^2``."""
        difference = _device.to_device(difference, self.device)
        if self.area_weighted:
            return float(self._areas @ (difference**2) @ self._areas)
        return float((difference**2).sum())

    # -- template ------------------------------------------------------------------

    def template(self, connectomes, kernel, iterations: int = 100, verbose: bool = False):
        """Karcher median of the subjects' square-root densities (``get_template``).

        Starts from the subject nearest the Euclidean mean and takes Weiszfeld
        steps a fifth as long as the mean direction to the subjects, until that
        direction is shorter than 0.005 (a step of 0.001). The densities are
        normalized to unit norm first, and a subject within 1e-6 radians of
        the estimate is coincident with it: no direction, no weight. The
        reference's arithmetic lets the rounding of the starting subject's
        norm pin the median to that subject (module docstring, item 15). A
        subject with no streamlines has no density to normalize and is
        refused, and so is one built on other grids than this engine's. The
        connectomes are read one at a time, so a generator never holds them all.
        """
        if not _whole(iterations):
            raise ValueError(f"iterations must be a whole number, 0 or more; got {iterations}")
        xp = self._xp
        kernel, _ = _sent(kernel, None, self.device)
        qs = _device.Held(self.device)
        for index, connectome in enumerate(connectomes):
            self._check_grids(connectome, f"subject {index}")
            q = self._here(connectome).q_transform(kernel)
            if not q.any():
                raise ValueError(
                    f"subject {index} has no streamlines on this grid, so its density is zero "
                    "everywhere and cannot enter the template; leave it out"
                )
            # q_transform returns a fresh array: normalized in place, the cohort's
            # densities are held once rather than twice
            q /= xp.sqrt((q**2).sum())
            qs.append(q)
        if not qs:
            raise ValueError("a template needs at least one connectome")
        count = len(qs)
        coincident = 1e-6  # radians; subjects sit 0.2 to 0.5 apart (item 15)
        # their mean, added in the order sum() adds them
        q_bar = qs[0].copy()
        for i in range(1, count):
            q_bar += qs[i]
        q_bar /= count
        q_mu = qs[int(np.argmin([float(((qs[i] - q_bar) ** 2).sum()) for i in range(count)]))]
        q_mu = q_mu.copy()
        del q_bar
        for _ in range(int(iterations)):
            directions = xp.zeros_like(q_mu)
            inverse_distance = 0.0
            for i in range(count):
                q = qs[i]
                inner = float((q * q_mu).sum())
                distance = float(np.arccos(np.clip(inner, -1.0, 1.0)))
                if distance > coincident:
                    cosine, sine = float(np.cos(distance)), float(np.sin(distance))
                    directions += (q - cosine * q_mu) / sine
                    inverse_distance += 1.0 / distance
            if inverse_distance == 0:
                break
            v_bar = directions / inverse_distance
            norm = float(xp.sqrt((v_bar**2).sum()))
            if norm == 0:
                break
            cosine, sine = float(np.cos(0.2 * norm)), float(np.sin(0.2 * norm))
            q_mu = cosine * q_mu + sine * (v_bar / norm)
            q_mu /= xp.sqrt((q_mu**2).sum())
            if verbose:
                print(f"template step {norm:.4f}")
            if norm < 0.005:
                break
        return _device.to_host(q_mu)

    # -- gradient --------------------------------------------------------------------

    def _gradient(self, difference, q, q_e1, q_e2, rows, grid: SphericalGrid):
        """``compute_gradient``: the descent direction on one hemisphere, ``(P, 2)``."""
        block = difference[rows]
        if self.area_weighted:
            block = block * self._areas[None, :]
        s1 = (block * q_e1[rows]).sum(axis=1)
        s2 = (block * q_e2[rows]).sum(axis=1)
        s3 = (block * q[rows]).sum(axis=1)
        # vectors now, brought to the basis on the CPU
        s1, s2, s3 = (_device.to_host(s) for s in (s1, s2, s3))
        integrand = (
            2 * s1[:, None] * grid.basis[:, :, 0]
            + 2 * s2[:, None] * grid.basis[:, :, 1]
            + s3[:, None] * grid.laplacian
        )
        if self.area_weighted:
            integrand = integrand * (grid.areas / self._area_unit)[:, None]
        coefficients = 2.0 * integrand.sum(axis=0)
        return (coefficients[None, :, None] * grid.basis).sum(axis=1)

    def _step(self, gradient) -> np.ndarray:
        """``compute_step_size``: cap the gradient's largest norm, then scale by ``-delta``."""
        largest = float(np.linalg.norm(gradient, axis=1).max())
        if largest > self.step_clamp:
            gradient = gradient * (self.step_clamp / largest)
        return -self.delta * gradient

    # -- rigid initialization ---------------------------------------------------------

    @staticmethod
    def _pull_back(values, grid: SphericalGrid, query: MeshQuery, rotation):
        """``rotate_matrix_barycentric`` on one hemisphere: ``g(x) = values(R x)``.

        CuPy ``values`` are pulled back on the GPU: the rotated vertices located there,
        which finds the CPU's faces and weights bit for bit, each row's three entries in
        the order SciPy keeps them, and the products added along each row in turn.
        """
        n = grid.n_vertices
        if _device.on_device(values):
            cp = _device.cupy()
            weights, indices, _ = query.query_faces(cp.asarray(grid.vertices @ rotation.T))
            order = cp.argsort(indices, axis=1)  # SciPy's CSR sorts each row by column
            operator = _device.sparse_module("gpu").csr_matrix(
                (
                    cp.take_along_axis(weights, order, axis=1).ravel(),
                    cp.take_along_axis(indices, order, axis=1).ravel().astype(cp.int32),
                    cp.arange(0, 3 * n + 1, 3, dtype=cp.int32),
                ),
                shape=(n, n),
            )
            halfway = _device.rows_times_dense(operator, values)
            return _device.rows_times_dense(operator, halfway.T).T
        from scipy import sparse

        weights, indices, _ = query.query_faces(grid.vertices @ rotation.T)
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
        # The reference's caps turn nothing about z (spin_shells); each shell's best are
        # turned about z as well.
        spins = spin_shells()

        def refine(candidates, rotations, keep, target, block, grid, query):
            """The ``keep`` best of every candidate composed with every rotation."""
            costs = np.zeros((len(candidates), rotations.shape[0]))
            for k, candidate in enumerate(candidates):
                for s, rotation in enumerate(rotations):
                    rotated = self._pull_back(block, grid, query, candidate @ rotation)
                    costs[k, s] = float(((target - rotated) ** 2).sum())
            order = np.argsort(costs.ravel())[:keep]
            k_idx, s_idx = np.unravel_index(order, costs.shape)
            return [candidates[k] @ rotations[s] for k, s in zip(k_idx, s_idx, strict=True)]

        best_pull_back = []
        hemispheres = (
            (moving.lh_grid, slice(0, n_left), moving._query[0]),
            (moving.rh_grid, slice(n_left, moving.n_vertices), moving._query[1]),
        )
        xp = self._xp
        for grid, rows, query in hemispheres:
            # a hemisphere's blocks, searched where the engine computes; the candidates,
            # 3 x 3 rotations, are kept and composed on the CPU either way
            target, block = q1[rows, rows], q2[rows, rows]
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
                rotated = xp.stack(
                    [
                        self._pull_back(block, grid, query, rotation).ravel()
                        for rotation in rotations[first : first + chunk]
                    ]
                )
                rotated_norm = (rotated**2).sum(axis=1)
                for g, symmetry in enumerate(group):
                    pulled = self._pull_back(target, grid, query, symmetry).ravel()
                    cross = rotated @ pulled
                    costs[g, first : first + chunk] = _device.to_host(
                        (pulled @ pulled) + rotated_norm - 2.0 * cross
                    )
            order = np.argsort(costs.ravel())[:keep]
            g_idx, s_idx = np.unravel_index(order, costs.shape)
            candidates = [rotations[s] @ group[g].T for g, s in zip(g_idx, s_idx, strict=True)]

            here = (target, block, grid, query)
            candidates = refine(candidates, spins[0], keep, *here)
            # later shells: refine each candidate with a finer cap, then turn it about z
            for (rotations, keep), turns in zip(shells[1:], spins[1:], strict=True):
                candidates = refine(candidates, rotations, keep, *here)
                candidates = refine(candidates, turns, keep, *here)
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
            return (*self.new_warps(), False)
        return lh_warp, rh_warp, True

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
        such as a template. ``moving`` is copied and registered from where its
        endpoints currently sit, so a connectome that has already been warped
        -- one from an earlier alignment's ``connectomes``, say -- is
        registered onward rather than reset to the locations it was built
        with; ``warped`` is that copy with its endpoints carried along, and
        ``costs`` the trace from the initial cost onward.
        ``callback(iteration, cost)`` is called after every step kept. A step
        refused on both hemispheres ends the registration, since the next
        would be the same, and so does a step that leaves the cost where it
        was; a moving connectome whose density already is the fixed one, as
        ``endpoints_align(..., template=i)`` makes subject ``i``'s, takes no
        step. A moving connectome with no streamlines is refused, and so is a
        fixed density that is zero everywhere, not finite or not on the moving
        connectome's grids, and a connectome built on other grids than this
        engine's. A fixed array must be a square-root density: one
        that is negative, held in one triangle, or whose squares do not sum
        to 1 -- a density passed in place of its square root, say -- is
        refused.
        """
        if not moving.n_streamlines:
            raise ValueError(
                "the moving connectome has no streamlines on this grid, so there is nothing "
                "to register"
            )
        self._check_grids(moving, "the moving connectome")
        self._check_grids(fixed, "the fixed connectome")
        xp = self._xp
        kernel, derivative = _sent(kernel, derivative, self.device)
        if isinstance(fixed, EndpointConnectome):
            q1 = self._here(fixed).q_transform(kernel)
        else:
            single = _single_precision(fixed)
            q1 = _device.to_device(fixed, self.device, np.float64)
        if q1.shape != (moving.n_vertices, moving.n_vertices):
            raise ValueError(
                f"the fixed density is {q1.shape}, where the moving connectome's grids need "
                f"({moving.n_vertices}, {moving.n_vertices})"
            )
        if not q1.any():
            raise ValueError(
                "the fixed connectome has no streamlines on this grid: its density is zero "
                "everywhere, so there is nothing to register onto"
            )
        if not xp.isfinite(q1).all():
            raise ValueError(
                "the fixed density holds non-finite values; every cost would be NaN and the "
                "registration would run to max_iterations on them"
            )
        if not isinstance(fixed, EndpointConnectome):
            problem = _density_problem(q1)
            if problem:
                raise ValueError(f"the fixed square-root density {problem}")
            # a subject's squares sum to 1 within about 1e-10, a template's exactly
            mass = float(xp.vdot(q1, q1))
            if single and abs(mass - 1.0) <= SINGLE_PRECISION_MASS:  # its precision, no error
                q1 = q1 / np.sqrt(mass)
                mass = float(xp.vdot(q1, q1))
            if abs(mass - 1.0) > 1e-6:
                raise ValueError(
                    f"the fixed density's squares sum to {mass:.6g}, not 1: register onto a "
                    "square-root density, as q_transform and ConSEAL.template give, not a "
                    "density"
                )
        if not isinstance(moving, EndpointConnectome):
            moving = self._here(moving).copy()  # the CPU takes it as it is, the GPU refuses it
        elif moving.device == self.device == "cpu":
            moving = moving.copy()
        else:  # a copy on this engine's device
            moving = moving._to_device(self.device)
        # The fresh identity warps act on the committed locations, so commit
        # the current ones: a no-op for a connectome that was never warped,
        # and for one that was, the difference between registering it onward
        # and silently snapping it back to where it was built. The rollback
        # below then also returns to the state the registration started from.
        moving.commit()
        # whether moving's endpoints have been carried off the committed locations: carried
        # through the identity they would still move, by its triangles' closest points (0.05
        # degrees on average on ico2, conseal_recheck_probe.py), so a rollback to the start
        # resets them instead
        relocated = False
        if init_rotation:
            lh_warp, rh_warp, relocated = self._rigid(q1, moving, kernel, verbose)
        else:
            lh_warp, rh_warp = self.new_warps()

        rows_lh = slice(0, moving.n_left)
        rows_rh = slice(moving.n_left, moving.n_vertices)
        q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative)
        difference = q1 - q2
        costs = [self.cost(difference)]
        if not np.isfinite(costs[0]):
            raise ValueError(
                "the starting cost is not finite, so neither is the moving connectome's "
                "density: is the kernel finite?"
            )
        # the fall that ends the descent: the threshold, or the share of the starting cost
        # when that is smaller, since a small warp's whole cost is small (item 8)
        threshold = self.threshold
        if self.relative_threshold:
            threshold = min(threshold, self.relative_threshold * costs[0])
        if verbose:
            print(f"initial cost {costs[0]:.6f}, stopping below a fall of {threshold:.3g}")

        # a zero starting cost leaves nothing to descend, and its share is no threshold
        iterations = self.max_iterations if costs[0] > 0 else 0
        for iteration in range(1, iterations + 1):
            previous = (lh_warp.copy(), rh_warp.copy())
            gradient_lh = self._gradient(difference, q2, q2_e1, q2_e2, rows_lh, self.lh_grid)
            gradient_rh = self._gradient(difference, q2, q2_e1, q2_e2, rows_rh, self.rh_grid)
            applied = [
                lh_warp.compose(self._step(gradient_lh)),
                rh_warp.compose(self._step(gradient_rh)),
            ]
            if not any(applied):
                # both steps refused leaves everything as it was, so the next iteration would
                # take the same two steps and be refused again: stop rather than repeat them
                break
            moving.warp(lh_warp, rh_warp)
            q2, q2_e1, q2_e2 = moving.q_transform(kernel, derivative)
            difference = q1 - q2
            costs.append(self.cost(difference))
            improvement = costs[-2] - costs[-1]
            # NaN compares false either way: it stops the descent rather than run it out; and
            # a step that leaves the cost where it was stops it whatever the threshold
            if not (improvement >= threshold and improvement > 0):
                if not improvement >= 0:
                    # roll back rather than return the warp that made things worse (item 4);
                    # the refusals the step met still count
                    refused = (lh_warp.rejected, rh_warp.rejected)
                    lh_warp, rh_warp = previous
                    lh_warp.rejected, rh_warp.rejected = refused
                    if relocated:
                        moving.warp(lh_warp, rh_warp)
                    else:
                        moving.reset()
                    costs.pop()
                elif callback is not None:
                    callback(iteration, costs[-1])
                break
            relocated = True
            if callback is not None:
                callback(iteration, costs[-1])
            if verbose and iteration % 10 == 0:
                print(f"iteration {iteration}: cost {costs[-1]:.6f}")
        if verbose:
            print(f"final cost {costs[-1]:.6f} after {len(costs) - 1} iterations")
        if isinstance(moving, EndpointConnectome) and moving.device != "cpu":
            moving = moving._to_device("cpu")  # handed back on the CPU
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
    """``|X & Y| / min(|X|, |Y|)`` over vertices whose endpoint density exceeds ``threshold``.

    Vertex-wise: which vertices the endpoints touch, not which pairs of vertices the
    streamlines join; :func:`connection_overlap` is the connection-wise coefficient.
    """
    x = endpoint_mass(first) > threshold
    y = endpoint_mass(second) > threshold
    return float((x & y).sum() / max(min(x.sum(), y.sum()), 1))


def dice_score(first: EndpointConnectome, second: EndpointConnectome, threshold: float) -> float:
    """``2 |X & Y| / (|X| + |Y|)`` over vertices whose endpoint density exceeds ``threshold``.

    Vertex-wise, as :func:`overlap_coefficient`; :func:`connection_dice` is connection-wise.
    """
    x = endpoint_mass(first) > threshold
    y = endpoint_mass(second) > threshold
    return float(2 * (x & y).sum() / max(x.sum() + y.sum(), 1))


def _connection_counts(first: EndpointConnectome, second: EndpointConnectome, threshold: float):
    """How many vertex pairs each connectome joins above ``threshold``, and how many both join.

    A pair counts in both orders, and a vertex with itself once, as in the full symmetric
    matrix. The adjacency is :meth:`EndpointConnectome.adjacency`, taken on the CPU.
    """
    threshold = float(threshold)
    if not (np.isfinite(threshold) and threshold >= 0):
        raise ValueError(f"threshold must be finite and 0 or more, got {threshold}")
    sets = []
    for connectome in (first, second):
        if connectome.device != "cpu":
            connectome = connectome._to_device("cpu")
        sets.append(connectome._sparse_adjacency() > threshold)
    x, y = sets
    return int(x.count_nonzero()), int(y.count_nonzero()), int(x.multiply(y).count_nonzero())


def connection_overlap(
    first: EndpointConnectome, second: EndpointConnectome, threshold: float = 0.0
) -> float:
    """``|X & Y| / min(|X|, |Y|)`` over the vertex pairs the streamlines join above ``threshold``.

    Connection-wise, where :func:`overlap_coefficient` is vertex-wise: each streamline gives
    the vertex pairs of its two endpoints' triangles the products of their barycentric
    weights, the matrix is symmetrized and divided by the number of streamlines
    (:meth:`EndpointConnectome.adjacency`, what ConSEAL smooths), and a pair belongs to a
    connectome's set where that exceeds ``threshold``. Two sets that are both empty overlap
    fully (1), and one empty set not at all (0). With ``threshold=0``, this is the overlap of
    bundles the ConSEAL paper reports.
    """
    nx, ny, shared = _connection_counts(first, second, threshold)
    if nx == 0 and ny == 0:
        return 1.0
    if nx == 0 or ny == 0:
        return 0.0
    return shared / min(nx, ny)


def connection_dice(
    first: EndpointConnectome, second: EndpointConnectome, threshold: float = 0.0
) -> float:
    """``2 |X & Y| / (|X| + |Y|)`` over the vertex pairs of :func:`connection_overlap`."""
    nx, ny, shared = _connection_counts(first, second, threshold)
    if nx == 0 and ny == 0:
        return 1.0
    if nx == 0 or ny == 0:
        return 0.0
    return 2 * shared / (nx + ny)


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


_OTHER_GRIDS = (
    "was built on other grids than this alignment's: its endpoints would be read by vertex "
    "index in the wrong frame. Build it on these grids (EndpointConnectome.from_points or "
    "from_endpoints with this alignment's lh_grid and rh_grid)"
)


def _on_grids(connectome, lh_grid, rh_grid) -> bool:
    """Whether ``connectome`` was built on these grids: the same vertices and faces.

    Compared by value, since :func:`default_grids` makes new grids at every call; the
    basis order is no part of where the endpoints lie.
    """
    for theirs, mine in ((connectome.lh_grid, lh_grid), (connectome.rh_grid, rh_grid)):
        if theirs is not mine and not (
            np.array_equal(theirs.vertices, mine.vertices)
            and np.array_equal(theirs.faces, mine.faces)
        ):
            return False
    return True


def _as_subject(
    item,
    lh_grid,
    rh_grid,
    needs: str = "connectomes that carry their endpoints",
    name: str = "a connectome",
):
    """The :class:`EndpointConnectome` on the grids for what :func:`endpoints_align` was given."""
    from .smoothing import Endpoints

    if isinstance(item, EndpointConnectome):
        if not _on_grids(item, lh_grid, rh_grid):
            raise ValueError(f"{name} {_OTHER_GRIDS}")
        return item
    if isinstance(item, Endpoints):
        return EndpointConnectome.from_endpoints(item, lh_grid, rh_grid)
    endpoints = getattr(item, "endpoints", None)
    if endpoints is None:
        raise MissingDataError(f"endpoints_align needs {needs}")
    return EndpointConnectome.from_endpoints(endpoints, lh_grid, rh_grid)


def endpoints_align(
    connectomes,
    template=None,
    sigma: float | None = None,
    kernel_degree: int = DEFAULT_KERNEL_DEGREE,
    order: int = DEFAULT_WARP_ORDER,
    delta: float | None = None,
    max_iterations: int | None = None,
    threshold: float | None = None,
    step_clamp: float | None = None,
    viscosity: float | None = None,
    template_iterations: int = 100,
    init_rotation: bool = False,
    area_weighted: bool = False,
    grids=None,
    verbose: bool = False,
    relative_threshold: float | None = None,
    device: str = "cpu",
) -> EndpointAlignment:
    """Align a cohort by their streamline endpoints (ConSEAL).

    Parameters
    ----------
    connectomes
        :class:`~sbci.ContinuousConnectome` objects carrying endpoints with
        barycentric positions, :class:`~sbci.smoothing.Endpoints`, or
        :class:`EndpointConnectome` built on ``grids``, each with at least one
        streamline.
    template
        What every subject is registered onto. ``None``, the default,
        estimates it first: the Karcher median of the subjects' square-root
        densities (up to ``template_iterations`` Weiszfeld steps), so that no
        subject is the reference. A connectome with endpoints (anything the
        list takes) registers the listed subjects onto that one, which stays
        where it is and need not be in the list: this is how one subject is
        registered onto another. An integer registers everyone onto that
        subject's own density. A square-root density array registers onto
        that: an earlier run's :attr:`EndpointAlignment.template`, say, or the
        normalized mean of the subjects' ``q_transform(kernel)`` arrays. Its
        squares must sum to 1, so a density passed in its place is refused.
    sigma, kernel_degree
        Heat-kernel bandwidth and truncation degree: 0.0027, at which this
        kernel is as narrow as the paper's (the fork's 0.005 with its own
        weights, module docstring item 8), and 30. The public code's is
        ``PUBLIC_SIGMA``, 0.005.
    order, delta, max_iterations, threshold, step_clamp, viscosity
        Velocity-basis order and gradient-descent settings. The order is 15;
        the others, left out, are the paper's -- step 0.1, up to 1000
        iterations, threshold 1e-6, no clamp and no smoothing; the public
        code's (0.05, 100, 1e-4, 0.2, 0.05) are the ``PUBLIC_*`` constants.
        Before 7 October 2026 the public code's were the defaults; the
        paper's undo 93% to 97% of a known warp where they stopped at 52% to
        75% (module docstring, item 8). A whole subject takes 50 to 490
        iterations onto a template or another subject, about 1.2 s each on a
        V100 and 5 to 8 s on eight cores of Longleaf's AMD EPYC nodes
        (PORTING.md item 36); ``max_iterations`` caps it.
    relative_threshold
        A floor under ``threshold`` as a share of the starting cost, 1e-5 by
        default; 0, the public code's, turns it off (:class:`ConSEAL`). With
        it and the narrowed kernel, a known warp of 0.8 degrees is undone to
        0.009 degrees, where the paper's absolute threshold alone stopped at
        0.07.
    init_rotation
        Search rotations per hemisphere before the diffeomorphic step. The 60
        icosahedral rotations are fitted to the grid rather than assumed, and
        the densities are transported by interpolation, so the search runs on
        the bundled FreeSurfer sphere, which is an icosphere to 1.7e-4.
    area_weighted
        See :class:`ConSEAL`.
    grids
        ``(lh_grid, rh_grid)`` to work on; defaults to :func:`default_grids`.
    device
        ``"cpu"``, the default, or ``"gpu"``: hold the densities and the
        endpoints on a CUDA GPU through CuPy, ``pip install 'sbci[gpu]'``
        (:class:`ConSEAL`). The result is the same object on the CPU, and agrees
        with the CPU's as :class:`ConSEAL` says.

    Returns
    -------
    EndpointAlignment
        The template, one :class:`EndpointWarp` per subject, the warped
        connectomes (``aligned_endpoints(i)`` gives them back as
        :class:`~sbci.smoothing.Endpoints`), and each cost trace.
    """
    from .smoothing import Endpoints

    _device.check(device)
    if not _whole(template_iterations):
        raise ValueError(
            f"template_iterations must be a whole number, 0 or more; got {template_iterations}"
        )
    # the kernel's settings, before any subject is read or anything built
    sigma = _bandwidth(DEFAULT_SIGMA if sigma is None else sigma)
    _kernel_degree(kernel_degree)
    if grids is None:
        grids = default_grids(order)
    lh_grid, rh_grid = grids
    subjects = [
        _as_subject(item, lh_grid, rh_grid, name=f"connectome {index}")
        for index, item in enumerate(connectomes)
    ]
    if not subjects:
        raise ValueError("endpoints_align needs at least one connectome")
    if template is None and len(subjects) < 2:
        raise ValueError(
            "estimating a template needs at least two connectomes; pass template= to "
            "register one connectome onto another, or onto an existing template"
        )
    fixed = None
    if isinstance(template, (EndpointConnectome, Endpoints)) or hasattr(template, "endpoints"):
        # the fixed subject, built on the same grids as the moving ones; it need not be listed
        fixed = _as_subject(
            template,
            lh_grid,
            rh_grid,
            "a template connectome that carries its endpoints",
            name="the template connectome",
        )
    # an empty subject's density is zero everywhere: there is nothing to register, and
    # normalizing it for the template made the whole cohort NaN. Refuse it before the
    # kernel is built.
    for index, subject in enumerate(subjects):
        if not subject.n_streamlines:
            raise ValueError(
                f"subject {index} has no streamlines on this grid, so there is nothing to "
                "register; leave it out"
            )
    if fixed is not None and not fixed.n_streamlines:
        raise ValueError(
            "the template connectome has no streamlines on this grid, so there is nothing "
            "to register onto"
        )

    builder = HeatKernelBuilder(lh_grid, rh_grid, kernel_degree)
    kernel, derivative = builder.compute(sigma, derivative=True)
    engine = ConSEAL(
        lh_grid,
        rh_grid,
        delta,
        max_iterations,
        threshold,
        step_clamp,
        viscosity,
        area_weighted=area_weighted,
        relative_threshold=relative_threshold,
        device=device,
    )
    # sent to the GPU once for every subject; on the CPU, as they are
    kernel, derivative = _sent(kernel, derivative, device)

    def root_density(connectome):
        """Its square-root density, computed where the engine computes, on the CPU."""
        return _device.to_host(engine._here(connectome).q_transform(kernel))

    if template is None:
        target = engine.template(subjects, kernel, template_iterations, verbose)
    elif isinstance(template, (bool, np.bool_)):
        raise ValueError(
            f"template={template} names no subject: give a subject's index, a connectome or "
            "a square-root density"
        )
    elif isinstance(template, (int, np.integer)):
        if not 0 <= int(template) < len(subjects):
            raise ValueError(f"template={template} names no subject; the list has {len(subjects)}")
        target = root_density(subjects[int(template)])
    elif fixed is not None:
        target = root_density(fixed)
    else:
        single = _single_precision(template)
        target = np.asarray(_device.to_host(template), dtype=np.float64)
        if single:  # its unit mass held only to its precision: scaled to it in float64
            mass = float(np.vdot(target, target))
            if abs(mass - 1.0) <= SINGLE_PRECISION_MASS:
                target = target / np.sqrt(mass)
    expected = lh_grid.n_vertices + rh_grid.n_vertices
    if target.shape != (expected, expected):
        raise ValueError(
            f"template is {target.shape}, expected a {expected} x {expected} square-root density"
        )

    result = EndpointAlignment(template=target)
    target = _device.to_device(target, device)  # sent once; on the CPU, the template itself
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
                lh_rigid=lh_warp.rigid.copy(),
                rh_rigid=rh_warp.rigid.copy(),
                lh_rigid_after=lh_warp.rigid_after,
                rh_rigid_after=rh_warp.rigid_after,
            )
        )
        result.connectomes.append(warped)
        result.costs.append(costs)
        result.rejected_steps.append(lh_warp.rejected + rh_warp.rejected)
    return result
