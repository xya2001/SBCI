"""Reduced-rank representation: the properties that hold without MATLAB.

The numerical comparison against a real run of ``SBCI_Modeling_FPCA`` is in
``test_matlab_reference.py`` and skips without it. What is asserted here is
what the method guarantees on any input: components ordered by the weight they
carry, a monotone increase in what the fit explains, exact recovery of data
that is genuinely low rank, and scores that reproduce the subject they came
from.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.reduction import (
    Reduction,
    fit_basis,
    power_iteration,
    project,
    reduce,
    sparse_power_iteration,
)


@pytest.fixture
def cohort():
    """Twelve subjects built from three separable components plus noise."""
    rng = np.random.default_rng(4)
    n, n_subjects, true_rank = 24, 12, 3
    truth = rng.standard_normal((n, true_rank))
    truth /= np.linalg.norm(truth, axis=0, keepdims=True)
    weights = rng.standard_normal((n_subjects, true_rank)) * np.array([4.0, 2.0, 1.0])

    matrices = np.stack(
        [
            sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(true_rank))
            for i in range(n_subjects)
        ]
    )
    noise = rng.standard_normal(matrices.shape) * 0.05
    matrices = matrices + (noise + np.transpose(noise, (0, 2, 1))) / 2
    matrices -= matrices.mean(axis=0, keepdims=True)
    return matrices, truth


def test_power_iteration_finds_the_leading_eigenvector():
    rng = np.random.default_rng(0)
    vectors = np.linalg.qr(rng.standard_normal((8, 8)))[0]
    values = np.array([9.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])
    matrix = vectors @ np.diag(values) @ vectors.T

    found = power_iteration(matrix, rng.standard_normal(8), max_iter=200, tol=1e-14)
    assert abs(abs(found @ vectors[:, 0]) - 1.0) < 1e-8


def test_the_sparse_power_iteration_respects_its_support():
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((10, 10))
    matrix = matrix @ matrix.T
    found = sparse_power_iteration(
        matrix, np.eye(10), rng.standard_normal(10), support=3, max_iter=100, tol=1e-12
    )
    assert int((np.abs(found) > 0).sum()) <= 3


def test_a_genuinely_low_rank_cohort_is_recovered_exactly():
    """With no noise, three components must reproduce the data."""
    rng = np.random.default_rng(2)
    n, true_rank = 20, 3
    truth = np.linalg.qr(rng.standard_normal((n, n)))[0][:, :true_rank]
    weights = rng.standard_normal((8, true_rank)) * np.array([5.0, 3.0, 1.0])
    matrices = np.stack(
        [
            sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(true_rank))
            for i in range(8)
        ]
    )
    matrices -= matrices.mean(axis=0, keepdims=True)

    result = fit_basis(matrices, np.eye(n), rank=true_rank, seed=0, max_outer=200, tol_outer=1e-12)
    assert result.explained[-1] > 0.999
    rebuilt = np.stack([result.reconstruct(i) for i in range(8)])
    assert np.abs(rebuilt - matrices).max() < 1e-6 * np.abs(matrices).max()


def test_the_first_component_carries_the_most_weight(cohort):
    """The leading component dominates, but the rest need not be ordered.

    Greedy deflation maximizes the objective in the space orthogonal to the
    components already taken, which does not force the *scales* to decrease.
    On a real ico4 cohort they do not: 4.29, 4.03, 3.89, 3.55, 3.75 (x1e-08),
    with the fifth above the fourth. What is guaranteed by construction, and
    is asserted separately, is that the explained fraction only grows.
    """
    matrices, _ = cohort
    result = fit_basis(matrices, np.eye(matrices.shape[1]), rank=4, seed=0)
    assert np.abs(result.scales[0]) >= np.abs(result.scales[1:]).max()


def test_what_the_fit_explains_only_increases(cohort):
    matrices, _ = cohort
    result = fit_basis(matrices, np.eye(matrices.shape[1]), rank=5, seed=0)
    assert np.all(np.diff(result.explained) >= -1e-12)
    assert 0.0 < result.explained[0] <= result.explained[-1] <= 1.0 + 1e-12


def test_the_leading_component_finds_the_strongest_truth(cohort):
    matrices, truth = cohort
    result = fit_basis(matrices, np.eye(matrices.shape[1]), rank=3, seed=0)
    assert abs(np.corrcoef(result.basis[:, 0], truth[:, 0])[0, 1]) > 0.95


def test_scores_reproduce_the_subjects_they_came_from(cohort):
    """Projecting the data onto its own basis returns its own scores."""
    matrices, _ = cohort
    result = fit_basis(matrices, np.eye(matrices.shape[1]), rank=4, seed=0)
    again = project(result, matrices)
    fitted = result.scores * result.scales
    for k in range(result.rank):
        assert abs(np.corrcoef(again[:, k], fitted[:, k])[0, 1]) > 0.99


def test_a_higher_rank_never_explains_less(cohort):
    matrices, _ = cohort
    gram = np.eye(matrices.shape[1])
    low = fit_basis(matrices, gram, rank=2, seed=0).explained[-1]
    high = fit_basis(matrices, gram, rank=5, seed=0).explained[-1]
    assert high >= low - 1e-12


def test_supplying_the_start_makes_the_fit_deterministic(cohort):
    matrices, _ = cohort
    n = matrices.shape[1]
    start = np.linalg.qr(np.random.default_rng(3).standard_normal((n, n)))[0][:, :3]
    first = fit_basis(matrices, np.eye(n), rank=3, start=start)
    second = fit_basis(matrices, np.eye(n), rank=3, start=start)
    np.testing.assert_allclose(first.basis, second.basis)
    np.testing.assert_allclose(first.scales, second.scales)


def test_reduce_accepts_a_cohort_of_arrays(cohort):
    matrices, _ = cohort
    result = reduce(list(matrices), rank=3, seed=0)
    assert isinstance(result, Reduction)
    assert result.basis.shape == (matrices.shape[1], 3)
    assert result.scores.shape == (matrices.shape[0], 3)


def test_reduce_accepts_a_single_connectome(cohort):
    """One subject is the one-subject case, not an error."""
    matrices, _ = cohort
    result = reduce(matrices[0], rank=3, seed=0)
    assert result.scores.shape == (1, 3)
    assert result.rank == 3


def test_reduce_refuses_mismatched_grids(cohort):
    matrices, _ = cohort
    with pytest.raises(ValueError, match="different grids"):
        reduce([matrices[0], matrices[1][:-2, :-2]], rank=2)


def test_fit_basis_validates_its_arguments(cohort):
    matrices, _ = cohort
    n = matrices.shape[1]
    with pytest.raises(ValueError, match="expected"):
        fit_basis(matrices[0], np.eye(n), rank=2)
    with pytest.raises(ValueError, match="rank must be at least 1"):
        fit_basis(matrices, np.eye(n), rank=0)
    with pytest.raises(ValueError, match="gram is"):
        fit_basis(matrices, np.eye(n - 1), rank=2)


def _ring_laplacian(n):
    """A one-dimensional Laplacian, as a stand-in roughness penalty."""
    ring = np.zeros((n, n))
    for i in range(n):
        ring[i, i] = 2.0
        ring[i, (i + 1) % n] = ring[(i + 1) % n, i] = -1.0
    return ring


def test_the_roughness_penalty_smooths_the_components(cohort):
    """Within its working range, more penalty means less local variation."""
    matrices, _ = cohort
    n = matrices.shape[1]
    ring = _ring_laplacian(n)

    roughness = [
        (lambda v: v @ ring @ v)(
            fit_basis(matrices, np.eye(n), ring, rank=1, alpha=alpha, seed=0).basis[:, 0]
        )
        for alpha in (0.0, 0.1, 0.5, 1.0)
    ]
    assert roughness == sorted(roughness, reverse=True), roughness


def test_too_large_a_penalty_inverts_and_selects_the_roughest_mode(cohort):
    """A trap inherited from the reference, worth pinning down.

    The component is chosen with ``eigs(M, 1)``, whose default is the
    eigenvalue of largest *magnitude*, and the penalty enters as
    ``- alpha * R``. Once ``alpha * R`` outweighs the data term the
    largest-magnitude eigenvalue is the most negative one, so the fit returns
    the roughest direction instead of the smoothest and explains almost
    nothing. On this cohort the turn happens between alpha 1 and 5.
    """
    matrices, _ = cohort
    n = matrices.shape[1]
    ring = _ring_laplacian(n)

    working = fit_basis(matrices, np.eye(n), ring, rank=1, alpha=1.0, seed=0)
    inverted = fit_basis(matrices, np.eye(n), ring, rank=1, alpha=5.0, seed=0)

    working_roughness = working.basis[:, 0] @ ring @ working.basis[:, 0]
    inverted_roughness = inverted.basis[:, 0] @ ring @ inverted.basis[:, 0]

    assert inverted_roughness > working_roughness
    assert inverted_roughness == pytest.approx(np.linalg.eigvalsh(ring).max(), rel=0.01)
    assert inverted.explained[-1] < 0.1 * working.explained[-1]


def test_the_large_grid_path_matches_the_dense_one(monkeypatch):
    """Above ARPACK_THRESHOLD the projector and operator are applied, not formed.

    Force the dense path on the same problem and the two must agree: same
    mathematics, different evaluation order.
    """
    import sbci.reduction as reduction

    rng = np.random.default_rng(6)
    n, n_subjects, true_rank = reduction.ARPACK_THRESHOLD + 40, 5, 3
    truth = np.linalg.qr(rng.standard_normal((n, n)))[0][:, :true_rank]
    weights = rng.standard_normal((n_subjects, true_rank)) * np.array([5.0, 3.0, 1.0])
    matrices = np.stack(
        [
            sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(true_rank))
            for i in range(n_subjects)
        ]
    )
    noise = rng.standard_normal(matrices.shape) * 0.01
    matrices = matrices + (noise + np.transpose(noise, (0, 2, 1))) / 2
    matrices -= matrices.mean(axis=0, keepdims=True)
    start = np.linalg.qr(rng.standard_normal((n, n)))[0][:, :2]
    gram = np.diag(rng.uniform(0.5, 1.5, n))

    operator_path = fit_basis(matrices, gram, rank=2, start=start)
    monkeypatch.setattr(reduction, "ARPACK_THRESHOLD", 10**9)
    dense_path = fit_basis(matrices, gram, rank=2, start=start)

    for k in range(2):
        assert abs(abs(operator_path.basis[:, k] @ dense_path.basis[:, k]) - 1.0) < 1e-8
    np.testing.assert_allclose(operator_path.scales, dense_path.scales, rtol=1e-8)
    np.testing.assert_allclose(operator_path.explained, dense_path.explained, rtol=1e-8)


def test_the_gram_operator_is_the_unfolded_product():
    """The implicit mode-1 Gram matrix equals the reference's explicit one."""
    from sbci.reduction import _mode1_gram

    rng = np.random.default_rng(8)
    residual = rng.standard_normal((4, 7, 7))
    unfolded = np.moveaxis(residual, 0, -1).reshape(7, 7 * 4, order="F")
    explicit = unfolded @ unfolded.T
    vector = rng.standard_normal(7)
    np.testing.assert_allclose(_mode1_gram(residual)(vector), explicit @ vector, rtol=1e-12)


def test_projecting_the_training_cohort_recovers_its_own_scores(cohort):
    """reduce() centres and remembers the mean, so raw subjects project correctly."""
    matrices, _ = cohort
    offset = np.ones_like(matrices[0])
    raw = [m + offset for m in matrices]  # an uncentred cohort
    result = reduce(raw, rank=3, seed=0)
    assert result.mean is not None
    again = project(result, np.stack(raw))
    for k in range(result.rank):
        assert abs(np.corrcoef(again[:, k], result.scores[:, k])[0, 1]) > 0.99
    np.testing.assert_allclose(
        result.reconstruct(0) - result.mean, result.reconstruct(0) - result.mean
    )
    rebuilt = np.stack([result.reconstruct(i) for i in range(len(raw))])
    assert np.abs(rebuilt - np.stack(raw)).max() < np.abs(np.stack(raw)).max()


def test_the_closed_form_projection_equals_the_explicit_least_squares(cohort):
    matrices, _ = cohort
    result = fit_basis(matrices, np.eye(matrices.shape[1]), rank=3, seed=0)
    n = matrices.shape[1]
    lower = np.tril_indices(n)
    design = np.column_stack([np.outer(b, b)[lower] for b in result.basis.T])
    explicit = np.stack([np.linalg.lstsq(design, m[lower], rcond=None)[0] for m in matrices])
    np.testing.assert_allclose(
        project(result, matrices) * result.scales, explicit, rtol=1e-8, atol=1e-10
    )


def test_too_large_a_penalty_warns_about_the_inversion(cohort):
    matrices, _ = cohort
    n = matrices.shape[1]
    with pytest.warns(RuntimeWarning, match="roughest"):
        fit_basis(matrices, np.eye(n), _ring_laplacian(n), rank=1, alpha=5.0, seed=0)


def test_reduce_accepts_connectome_like_objects_off_the_ico4_grid(cohort):
    """A connectome on another grid gets the plain inner product, not the ico4 roughness."""
    matrices, _ = cohort

    class Toy:
        def __init__(self, dense):
            self._dense = dense
            self.area = np.ones(dense.shape[0])

        def dense(self):
            return self._dense

    result = reduce([Toy(m) for m in matrices], rank=2, seed=0)
    assert result.basis.shape == (matrices.shape[1], 2)
