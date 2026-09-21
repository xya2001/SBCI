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

from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHA = 1e-10
"""Roughness penalty. The reference's default, effectively off."""


def power_iteration(matrix, start, max_iter: int = 30, tol: float = 1e-3):
    """Leading eigenvector by power iteration, as ``power_iterations.m``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    current = np.asarray(start, dtype=np.float64).ravel()
    for _ in range(max_iter - 1):
        nxt = matrix @ current
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
    zero the rest, and normalize in the ``gram`` inner product.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    gram = np.asarray(gram, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64).ravel()

    for _ in range(max_iter - 1):
        nxt = matrix @ current
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
    than the small one.
    """
    matrix = (matrix + matrix.T) / 2
    if matrix.shape[0] <= ARPACK_THRESHOLD:
        values, vectors = np.linalg.eigh(matrix)
        return vectors[:, int(np.argmax(np.abs(values)))]

    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    try:
        _, vectors = eigsh(matrix, k=1, which="LM", v0=start, tol=0)
        return vectors[:, 0]
    except ArpackNoConvergence:  # pragma: no cover - rare, and recoverable
        values, vectors = np.linalg.eigh(matrix)
        return vectors[:, int(np.argmax(np.abs(values)))]


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

    @property
    def rank(self) -> int:
        """Components kept."""
        return int(self.basis.shape[1])

    def reconstruct(self, index: int) -> np.ndarray:
        """Rebuild one subject's connectome from its scores."""
        weights = self.scores[index] * self.scales
        return (self.basis * weights) @ self.basis.T

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
) -> Reduction:
    """Estimate the shared basis, one component at a time.

    Parameters
    ----------
    matrices
        ``(n_subjects, n, n)`` symmetric, already centred on the cohort mean.
    gram
        ``(n, n)`` inner-product matrix; on a mesh this is ``diag(areas)``.
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
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"expected (n_subjects, n, n), got {matrices.shape}")
    n_subjects, n, _ = matrices.shape
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")

    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (n, n):
        raise ValueError(f"gram is {gram.shape}, expected {(n, n)}")
    penalty = np.zeros((n, n)) if roughness is None else np.asarray(roughness, dtype=np.float64)

    rng = np.random.default_rng(seed)
    starts = None if start is None else np.asarray(start, dtype=np.float64)

    # The tensor is (n, n, n_subjects) in the reference; keep subjects first
    # here and contract explicitly, which is clearer and avoids a transpose.
    residual = matrices.copy()
    total_norm = np.linalg.norm(residual)

    components = np.zeros((n, rank))
    score_matrix = np.zeros((n_subjects, rank))
    scales = np.zeros(rank)
    explained = np.zeros(rank)
    objective = np.zeros((rank, max_outer))

    for k in range(rank):
        if k == 0:
            projector = np.eye(n)
        else:
            kept = components[:, :k]
            projector = np.eye(n) - kept @ np.linalg.inv(kept.T @ gram @ kept) @ kept.T @ gram

        # Initialise from the leading eigenvector of the mode-1 Gram matrix.
        unfolded = np.moveaxis(residual, 0, -1).reshape(n, n * n_subjects, order="F")
        if starts is not None:
            guess = starts[:, k]
        else:
            guess = rng.standard_normal(n)
            guess /= np.linalg.norm(guess)
        vector = power_iteration(unfolded @ unfolded.T, guess, max_inner, tol_inner)
        vector = vector / np.linalg.norm(vector)

        weights = np.einsum("nij,i,j->n", residual, vector, vector)
        norm = np.linalg.norm(weights)
        score = weights / norm if norm else weights

        contracted = np.einsum("nij,n->ij", residual, score)
        objective[k, 0] = vector @ projector @ (contracted - alpha * penalty) @ projector.T @ vector

        change = np.inf
        step = 0
        while step < max_outer - 1 and change > tol_outer:
            weights = np.einsum("nij,i,j->n", residual, vector, vector)
            norm = np.linalg.norm(weights)
            score = weights / norm if norm else weights

            contracted = np.einsum("nij,n->ij", residual, score)
            operator = projector @ (contracted - alpha * penalty) @ projector.T
            if support is None:
                vector = _leading_eigenvector(operator, start=vector)
            else:
                vector = sparse_power_iteration(
                    operator, gram, vector, support, max_inner, tol_inner
                )
                vector = vector / np.linalg.norm(vector)

            objective[k, step + 1] = (
                vector @ projector @ (contracted - alpha * penalty) @ projector.T @ vector
            )
            if objective[k, 0] != 0:
                change = abs((objective[k, step + 1] - objective[k, step]) / objective[k, 0])
            step += 1

        scale = float(vector @ contracted @ vector)
        residual = residual - scale * np.einsum("i,j,n->nij", vector, vector, score)

        components[:, k] = vector
        score_matrix[:, k] = score
        scales[k] = scale
        explained[k] = np.linalg.norm(matrices - residual) / total_norm

    return Reduction(
        basis=components,
        scores=score_matrix,
        scales=scales,
        explained=explained,
        objective=objective,
    )


def project(reduction: Reduction, matrices, gram=None) -> np.ndarray:
    """Score new connectomes against an existing basis.

    Port of ``ConConSmooth.smooth``: least squares of each subject's matrix
    against the separable products ``psi_k psi_k'``, over the lower triangle.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None]
    n = reduction.basis.shape[0]
    lower = np.tril_indices(n)

    design = np.empty((lower[0].size, reduction.rank))
    for k in range(reduction.rank):
        column = np.outer(reduction.basis[:, k], reduction.basis[:, k])
        design[:, k] = column[lower]

    out = np.empty((matrices.shape[0], reduction.rank))
    for i, matrix in enumerate(matrices):
        out[i], *_ = np.linalg.lstsq(design, matrix[lower], rcond=None)
    return out


def _grid_gram_and_roughness(area, coordinates=None):
    """The mesh inner product and roughness penalty for the bundled grid."""
    from .surface import load_surface

    gram = np.diag(np.asarray(area, dtype=np.float64))
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
    if matrices.shape[0] > 1:
        matrices = matrices - matrices.mean(axis=0, keepdims=True)

    n = matrices.shape[1]
    if area is not None and area.size == n:
        gram, roughness = _grid_gram_and_roughness(area)
    else:
        gram, roughness = np.eye(n), None

    return fit_basis(matrices, gram, roughness, rank=rank, **kwargs)
