"""Reduced-rank representation of a cohort of connectomes.

Port of ``SBCI_Modeling_FPCA`` (Continuous and Atlas-free Analysis of Brain
Structural Connectivity, AOAS 2024). A connectome is a function on the product
of two cortical surfaces; this finds a small set of surface functions
``psi_1 ... psi_K`` such that every subject is well approximated by a weighted
sum of the separable products ``psi_k psi_k'``::

    Y_i  ~  sum_k  s_ik * gamma_k * psi_k psi_k'

so a subject is summarised by ``K`` numbers instead of thirteen million. The
basis is shared across the cohort and estimated from it, one component at a
time, each fitted to the residual left by the previous ones.

What the dependencies were, and why none are needed
---------------------------------------------------
The reference represents ``psi`` in a spherical spline basis, which is where
its three external dependencies live: ``splinepak`` for the splines,
``getLebedevSphere`` for the quadrature that builds the basis Gram matrix
``J`` and roughness matrix ``R``, and ``tensor_toolbox`` for the tensor
algebra. Only the last touches the algorithm, and its handful of operations
are one line each in NumPy.

This package does not need the spline layer at all. Its functions already live
on the ico4 grid, where the L2 inner product is the Voronoi vertex areas --
verified to 1.6 float64-eps against libigl -- and the roughness penalty is the
cotangent Dirichlet energy. So the basis is the identity on the grid, and
``J`` and ``R`` come from geometry the package already has.

A trap in the roughness penalty
-------------------------------
Each component is the eigenvector of ``P (G - alpha R) P'`` with the largest
*magnitude* eigenvalue, which is what ``eigs(M, 1)`` returns. The penalty
enters with a minus sign, so once ``alpha * R`` outweighs the data term the
largest-magnitude eigenvalue is the most negative one and the fit returns the
**roughest** direction rather than the smoothest. On a test cohort the turn
happens between ``alpha`` 1 and 5: roughness falls 1.96 -> 1.71 as alpha rises
to 1, then jumps to 3.98 -- the maximum the penalty admits -- at 5, with the
explained fraction collapsing from 0.83 to 0.01. The reference's default of
``1e-10`` is far below the turn, so this only bites someone who raises it.

A bug in the reference
----------------------
``ConConBasis.Fit`` as published cannot run: at line 260 it reads
``auto_sparse``, a variable that is never defined -- the parsed option is
``params.auto_sparse``. Every call fails with an undefined-variable error
before the first component is recorded. Verifying this port needed a one-line
fix to the reference, which is recorded in PORTING.md item 5.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import spec

DEFAULT_ALPHA = 1e-10
"""Roughness penalty. The reference's default, effectively off."""


def _as_operator(matrix):
    """A function applying ``matrix`` to a vector.

    ``matrix`` may be a square array, which is symmetrized as the reference does,
    or something that is already an operator: a callable, or a SciPy
    ``LinearOperator``. The iterations below only ever multiply, so they never
    need the matrix itself -- and on the ico4 grid forming it is what costs.
    """
    if callable(matrix):
        return matrix
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    return lambda vector: matrix @ vector


def power_iteration(matrix, start, max_iter: int = 30, tol: float = 1e-3):
    """Leading eigenvector by power iteration, as ``power_iterations.m``.

    ``matrix`` may be an array or an operator; see :func:`_as_operator`.
    """
    apply = _as_operator(matrix)
    current = np.asarray(start, dtype=np.float64).ravel()
    for _ in range(max_iter - 1):
        nxt = apply(current)
        norm = np.linalg.norm(nxt)
        if norm == 0:
            break
        nxt = nxt / norm
        change = np.linalg.norm(nxt - current)
        current = nxt
        if change <= tol:
            break
    return current


def sparse_power_iteration(
    matrix, gram, start, support: int, max_iter: int = 30, tol: float = 1e-3
):
    """Leading eigenvector with a hard support constraint.

    Port of ``generalized_power.m``: keep the ``support`` largest components,
    zero the rest, and normalize in the ``gram`` inner product. ``matrix`` may
    be an array or an operator; see :func:`_as_operator`.
    """
    apply = _as_operator(matrix)
    gram = np.asarray(gram, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64).ravel()

    for _ in range(max_iter - 1):
        nxt = apply(current)
        if support < nxt.size:
            keep = np.argsort(np.abs(nxt))[::-1][:support]
            mask = np.zeros(nxt.size, dtype=bool)
            mask[keep] = True
            nxt = np.where(mask, nxt, 0.0)
        scale = np.sqrt(nxt @ gram @ nxt)
        if scale == 0:
            break
        nxt = nxt / scale
        change = np.linalg.norm(nxt - current)
        current = nxt
        if change <= tol:
            break
    return current


ARPACK_THRESHOLD = 400
"""Above this size, one eigenvector is found iteratively rather than all of them."""


def _leading_eigenvector(matrix, start=None):
    """The eigenvector of largest absolute eigenvalue, as ``eigs(M, 1)``.

    A full diagonalization costs ``O(n^3)`` and is wasteful when one vector is
    wanted; on the ico4 grid it would make a fit take hours. Above
    :data:`ARPACK_THRESHOLD` this uses ARPACK, which is the library MATLAB's
    ``eigs`` calls, so the large case is if anything closer to the reference
    than the small one. ``matrix`` is a dense array below the threshold and a
    SciPy ``LinearOperator`` above it (see :meth:`_Deflation.sandwich`), so the
    ``n x n`` operator is never formed on a large grid.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    if not isinstance(matrix, LinearOperator):
        matrix = (matrix + matrix.T) / 2
        if matrix.shape[0] <= ARPACK_THRESHOLD:
            values, vectors = np.linalg.eigh(matrix)
            return vectors[:, int(np.argmax(np.abs(values)))]

    try:
        _, vectors = eigsh(matrix, k=1, which="LM", v0=start, tol=0)
        return vectors[:, 0]
    except ArpackNoConvergence:  # pragma: no cover - rare, and recoverable
        dense = matrix @ np.eye(matrix.shape[0]) if isinstance(matrix, LinearOperator) else matrix
        values, vectors = np.linalg.eigh((dense + dense.T) / 2)
        return vectors[:, int(np.argmax(np.abs(values)))]


class _Deflation:
    """The projector onto the complement of the components already taken.

    ``P = I - K (K' G K)^-1 K' G`` for the kept components ``K`` and the inner
    product ``G``. The reference forms ``P`` and then ``P M P'`` explicitly, two
    ``n^3`` products per outer iteration -- about half a minute each on the
    ico4 grid. ``P`` is a rank-``k`` update of the identity, so applying it is
    ``O(n k)`` plus one product with ``G``; this class applies it and never
    forms it, except for the small case where a dense diagonalization is used.

    ``gram`` is the ``(n, n)`` inner product, or its diagonal as a 1-D array:
    on a mesh it is ``diag(areas)``, and applying that as a dense matrix would
    cost a full matrix-vector pass per projection for an elementwise scaling.
    """

    def __init__(self, kept, gram):
        gram = np.asarray(gram, dtype=np.float64)
        self.kept = kept
        self.diagonal = gram if gram.ndim == 1 else None
        self.matrix = gram if gram.ndim == 2 else None
        if kept is None or kept.shape[1] == 0:
            self.inverse = None
        else:
            self.inverse = np.linalg.inv(kept.T @ self._gram_times(kept))

    def _gram_times(self, x):
        """``G x`` for a vector or a matrix of columns."""
        if self.diagonal is not None:
            return self.diagonal[:, None] * x if x.ndim == 2 else self.diagonal * x
        return self.matrix @ x

    def _gram_transposed_times(self, x):
        """``G' x``."""
        if self.diagonal is not None:
            return self.diagonal * x
        return self.matrix.T @ x

    def apply(self, vector):
        """``P v``."""
        if self.inverse is None:
            return vector
        return vector - self.kept @ (self.inverse @ (self.kept.T @ self._gram_times(vector)))

    def apply_transposed(self, vector):
        """``P' v``."""
        if self.inverse is None:
            return vector
        return vector - self._gram_transposed_times(
            self.kept @ (self.inverse.T @ (self.kept.T @ vector))
        )

    def quadratic(self, matrix, vector):
        """``v' P M P' v``."""
        projected = self.apply_transposed(vector)
        return float(projected @ matrix @ projected)

    def sandwich(self, matrix):
        """``P M P'``: dense below :data:`ARPACK_THRESHOLD`, otherwise an operator."""
        n = matrix.shape[0]
        if n <= ARPACK_THRESHOLD:
            if self.inverse is None:
                return matrix
            gram = self.matrix if self.matrix is not None else np.diag(self.diagonal)
            projector = np.eye(n) - self.kept @ self.inverse @ self.kept.T @ gram
            return projector @ matrix @ projector.T

        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            (n, n),
            matvec=lambda v: self.apply(matrix @ self.apply_transposed(v)),
            dtype=np.float64,
        )


def _mode1_gram(residual):
    """The mode-1 Gram matrix ``sum_s R_s R_s'`` of the cohort, as an operator.

    The reference forms it from the mode-1 unfolding: an ``(n, n S)`` copy of
    the whole cohort followed by an ``n^2 S`` product, which on the ico4 grid
    is eight gigabytes and five trillion flops per component, all to seed a
    power iteration that only ever applies the matrix. Applying it directly
    costs ``2 n^2 S`` per vector and no copy.
    """

    def apply(vector):
        halfway = vector @ residual  # row s is R_s' v
        return (residual @ halfway[:, :, None])[:, :, 0].sum(axis=0)

    return apply


@dataclass
class Reduction:
    """The fitted basis, the scores, and how much each component explains."""

    basis: np.ndarray
    """``(n_vertices, rank)``; column ``k`` is ``psi_k`` on the grid."""
    scores: np.ndarray
    """``(n_subjects, rank)``; row ``i`` summarises subject ``i``."""
    scales: np.ndarray
    """``(rank,)``; the weight of each component in the fit.

    The first is the largest, but the rest are **not** guaranteed to decrease.
    Each component maximizes the objective in the space orthogonal to those
    already taken, which does not order the scales. Use :attr:`explained`,
    which only grows, to judge how many components are worth keeping.
    """
    explained: np.ndarray
    """``(rank,)``; fraction of the cohort's norm captured up to component k."""
    objective: np.ndarray
    """``(rank, iterations)``; the objective at each outer iteration."""
    mean: np.ndarray | None = None
    """``(n, n)`` cohort mean the fit was centred on, or ``None`` if uncentred.

    :func:`reduce` centres a cohort of two or more subjects and records the
    mean here, so :func:`project` can centre new subjects the same way and
    :meth:`reconstruct` can put it back.
    """

    @property
    def rank(self) -> int:
        """Components kept."""
        return int(self.basis.shape[1])

    def reconstruct(self, index: int) -> np.ndarray:
        """Rebuild one subject's connectome from its scores, mean included."""
        weights = self.scores[index] * self.scales
        rebuilt = (self.basis * weights) @ self.basis.T
        return rebuilt if self.mean is None else rebuilt + self.mean

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<Reduction rank {self.rank} over {self.scores.shape[0]} subjects, "
            f"{self.explained[-1]:.1%} explained>"
        )


def fit_basis(
    matrices,
    gram,
    roughness=None,
    rank: int = 10,
    alpha: float = DEFAULT_ALPHA,
    support=None,
    max_outer: int = 30,
    max_inner: int = 30,
    tol_outer: float = 1e-3,
    tol_inner: float = 1e-3,
    seed=None,
    start=None,
    copy: bool = True,
) -> Reduction:
    """Estimate the shared basis, one component at a time.

    Parameters
    ----------
    matrices
        ``(n_subjects, n, n)`` symmetric, already centred on the cohort mean.
    gram
        ``(n, n)`` inner-product matrix, or its diagonal as a 1-D array; on a
        mesh this is ``diag(areas)``. A dense diagonal is recognized and
        applied elementwise.
    roughness
        ``(n, n)`` penalty matrix, or ``None`` for no penalty.
    rank
        Components to extract.
    alpha
        Weight on the roughness penalty.
    support
        Keep only this many non-zero entries per component, using the sparse
        power iteration. ``None`` leaves components dense.
    start
        ``(n, rank)`` initial vectors, one per component. Supplying them makes
        the fit deterministic; otherwise they are drawn from ``seed``.
    copy
        Work on a copy of ``matrices`` (the default). ``False`` deflates the
        given array in place, which :func:`reduce` uses so that the cohort is
        held once rather than twice -- 2 GB rather than 4 for ten ico4 subjects.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"expected (n_subjects, n, n), got {matrices.shape}")
    n_subjects, n, _ = matrices.shape
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")

    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim == 1:
        if gram.shape != (n,):
            raise ValueError(f"gram is {gram.shape}, expected {(n, n)} or a diagonal of length {n}")
    elif gram.shape != (n, n):
        raise ValueError(f"gram is {gram.shape}, expected {(n, n)}")
    elif np.count_nonzero(gram) == np.count_nonzero(np.diagonal(gram)):
        # A diagonal inner product -- vertex areas, or the identity -- is
        # applied elementwise from here on: as a dense matrix it costs a full
        # matrix-vector pass per projection, a sixth of the fit on ico4.
        gram = np.diagonal(gram).copy()
    penalty = None if roughness is None else np.asarray(roughness, dtype=np.float64)
    if penalty is not None and penalty.shape != (n, n):
        raise ValueError(f"roughness is {penalty.shape}, expected {(n, n)}")

    rng = np.random.default_rng(seed)
    starts = None if start is None else np.asarray(start, dtype=np.float64)

    # The tensor is (n, n, n_subjects) in the reference; keep subjects first
    # here and contract explicitly, which is clearer and avoids a transpose.
    residual = matrices.copy() if copy else matrices
    total_norm = np.linalg.norm(residual)

    components = np.zeros((n, rank))
    score_matrix = np.zeros((n_subjects, rank))
    scales = np.zeros(rank)
    explained = np.zeros(rank)
    objective = np.zeros((rank, max_outer))
    inverted_warning_given = False

    for k in range(rank):
        deflation = _Deflation(components[:, :k] if k else None, gram)

        # Initialise from the leading eigenvector of the mode-1 Gram matrix.
        if starts is not None:
            guess = starts[:, k]
        else:
            guess = rng.standard_normal(n)
            guess /= np.linalg.norm(guess)
        vector = power_iteration(_mode1_gram(residual), guess, max_inner, tol_inner)
        vector = vector / np.linalg.norm(vector)

        weights = (residual @ vector) @ vector  # v' R_s v for every subject, by BLAS
        norm = np.linalg.norm(weights)
        score = weights / norm if norm else weights

        contracted = np.tensordot(score, residual, axes=(0, 0))
        regularized = contracted if penalty is None else contracted - alpha * penalty
        regularized = (regularized + regularized.T) / 2
        objective[k, 0] = deflation.quadratic(regularized, vector)

        change = np.inf
        step = 0
        while step < max_outer - 1 and change > tol_outer:
            weights = (residual @ vector) @ vector  # v' R_s v for every subject, by BLAS
            norm = np.linalg.norm(weights)
            score = weights / norm if norm else weights

            contracted = np.tensordot(score, residual, axes=(0, 0))
            regularized = contracted if penalty is None else contracted - alpha * penalty
            regularized = (regularized + regularized.T) / 2
            operator = deflation.sandwich(regularized)
            if support is None:
                vector = _leading_eigenvector(operator, start=vector)
            else:
                vector = sparse_power_iteration(
                    operator, gram, vector, support, max_inner, tol_inner
                )
                vector = vector / np.linalg.norm(vector)

            objective[k, step + 1] = deflation.quadratic(regularized, vector)
            if objective[k, step + 1] < 0 and not inverted_warning_given:
                # The trap described in the module docstring: the penalty (or a
                # dominant negative mode) has taken over and the component is
                # the roughest direction, not the smoothest.
                warnings.warn(
                    f"component {k}: the selected eigenvalue is negative, so the fit "
                    "is returning the roughest direction; lower alpha",
                    RuntimeWarning,
                    stacklevel=2,
                )
                inverted_warning_given = True
            if objective[k, 0] != 0:
                change = abs((objective[k, step + 1] - objective[k, step]) / objective[k, 0])
            step += 1

        scale = float(vector @ contracted @ vector)
        # Deflate subject by subject: one n x n temporary instead of a whole
        # cohort-sized one, which on the ico4 grid is the difference between
        # 210 MB and 8 GB per component.
        outer = np.outer(vector, vector)
        for index in range(n_subjects):
            residual[index] -= (scale * score[index]) * outer

        components[:, k] = vector
        score_matrix[:, k] = score
        scales[k] = scale
        # ||M - R||^2 in closed form: what has been removed so far is
        # sum_j scale_j score_sj psi_j psi_j', so the norm needs only the
        # (k+1) x (k+1) overlaps of the components, not another pass over the
        # cohort (and no untouched copy of it, which `copy=False` gave up).
        kept_basis = components[:, : k + 1]
        overlap = kept_basis.T @ kept_basis
        coefficients = scales[: k + 1] * score_matrix[:, : k + 1]
        captured = float(np.sum(overlap**2 * (coefficients.T @ coefficients)))
        explained[k] = np.sqrt(captured) / total_norm

    return Reduction(
        basis=components,
        scores=score_matrix,
        scales=scales,
        explained=explained,
        objective=objective,
    )


def project(reduction: Reduction, matrices) -> np.ndarray:
    """Score new connectomes against an existing basis.

    Port of ``ConConSmooth.smooth``: least squares of each subject's matrix
    against the separable products ``psi_k psi_k'`` over the lower triangle,
    diagonal included. Subjects are first centred on :attr:`Reduction.mean`
    when the basis was fitted to a centred cohort, and the result is in the
    units of :attr:`Reduction.scores` -- the fitted coefficient divided by the
    component's scale -- so it is directly comparable with them.

    For symmetric input the normal equations have a closed form,
    ``(X'X)_kl = ((psi_k . psi_l)^2 + sum_i psi_k(i)^2 psi_l(i)^2) / 2`` and
    ``(X'y)_k = (psi_k' Y psi_k + sum_i psi_k(i)^2 Y_ii) / 2``, which is what is
    solved here: ``O(n^2 K)`` per subject, and no thirteen-million-row design.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None]
    basis = np.asarray(reduction.basis, dtype=np.float64)
    n = basis.shape[0]
    if matrices.shape[1:] != (n, n):
        raise ValueError(f"matrices are {matrices.shape[1:]}, the basis is on {n} vertices")

    squares = basis**2
    normal = 0.5 * ((basis.T @ basis) ** 2 + squares.T @ squares)
    out = np.empty((matrices.shape[0], reduction.rank))
    for i, matrix in enumerate(matrices):
        centred = matrix if reduction.mean is None else matrix - reduction.mean
        quadratic = (basis * (centred @ basis)).sum(axis=0)
        right = 0.5 * (quadratic + squares.T @ np.diagonal(centred))
        out[i], *_ = np.linalg.lstsq(normal, right, rcond=None)

    scales = np.asarray(reduction.scales, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scales != 0, out / scales, 0.0)


def _grid_gram_and_roughness(area, coordinates=None):
    """The mesh inner product (as its diagonal) and roughness penalty for the bundled grid."""
    from .surface import load_surface

    gram = np.asarray(area, dtype=np.float64)  # diag(areas), kept as the diagonal
    surface = load_surface("sphere")
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)

    corners = vertices[faces]
    edges = [
        corners[:, 2] - corners[:, 1],
        corners[:, 0] - corners[:, 2],
        corners[:, 1] - corners[:, 0],
    ]
    double_area = np.linalg.norm(np.cross(edges[0], -edges[1]), axis=1)
    n = vertices.shape[0]
    roughness = np.zeros((n, n))
    for i in range(3):
        cot = -(edges[(i + 1) % 3] * edges[(i + 2) % 3]).sum(axis=1) / double_area
        a, b = faces[:, (i + 1) % 3], faces[:, (i + 2) % 3]
        np.add.at(roughness, (a, b), -0.5 * cot)
        np.add.at(roughness, (b, a), -0.5 * cot)
        np.add.at(roughness, (a, a), 0.5 * cot)
        np.add.at(roughness, (b, b), 0.5 * cot)
    return gram, 0.5 * (roughness + roughness.T)


def reduce(cc_list, rank: int = 10, **kwargs) -> Reduction:
    """Fit a reduced-rank basis to a cohort of connectomes.

    Parameters
    ----------
    cc_list
        Connectomes on the same grid, as :class:`~sbci.ContinuousConnectome`
        or dense arrays. A single connectome is allowed and gives its own
        rank-``K`` separable approximation.
    rank
        Components to keep.
    **kwargs
        Passed to :func:`fit_basis`.

    Examples
    --------
    >>> result = sbci.reduce(subjects, rank=20)       # doctest: +SKIP
    >>> result.scores.shape                            # doctest: +SKIP
    (40, 20)
    """
    single = hasattr(cc_list, "dense") or (isinstance(cc_list, np.ndarray) and cc_list.ndim == 2)
    items = [cc_list] if single else list(cc_list)

    densities = []
    area = None
    for item in items:
        if hasattr(item, "dense"):
            densities.append(np.asarray(item.dense(), dtype=np.float64))
            if area is None:
                area = np.asarray(item.area, dtype=np.float64)
        else:
            densities.append(np.asarray(item, dtype=np.float64))
    shapes = {d.shape for d in densities}
    if len(shapes) != 1:
        raise ValueError(f"connectomes are on different grids: {sorted(shapes)}")

    matrices = np.stack(densities)
    densities.clear()  # the stack owns the only copy now
    mean = None
    if matrices.shape[0] > 1:
        mean = matrices.mean(axis=0)
        matrices -= mean[None]

    n = matrices.shape[1]
    if area is not None and area.size == n == spec.N_VERTICES:
        # The mesh inner product and roughness are those of the bundled ico4
        # grid, so they only apply on it; any other grid gets the plain ones.
        gram, roughness = _grid_gram_and_roughness(area)
    else:
        gram, roughness = np.ones(n), None

    kwargs.setdefault("copy", False)  # `matrices` is ours: deflate it in place
    result = fit_basis(matrices, gram, roughness, rank=rank, **kwargs)
    result.mean = mean
    return result
