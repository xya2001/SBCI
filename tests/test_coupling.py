"""Structure-function coupling.

Checked against closed forms where the answer is writable by hand, and against
the behaviours the reference depends on: constant profiles become NaN rather
than zero, small regions are suppressed, and the discrete form centres its
vectors while the other two do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.coupling import (
    MIN_AREA,
    discrete_coupling,
    global_coupling,
    local_coupling,
    structure_function_coupling,
)


@pytest.fixture
def symmetric():
    """A small symmetric matrix with a zero diagonal."""
    return np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )


# --- global ----------------------------------------------------------------


def test_identical_matrices_couple_perfectly(symmetric):
    """Cosine similarity of a vector with itself is exactly one."""
    np.testing.assert_allclose(global_coupling(symmetric, symmetric), 1.0)


def test_scaling_does_not_change_cosine_similarity(symmetric):
    """A cosine compares direction, so FC scaled by any positive factor agrees."""
    np.testing.assert_allclose(global_coupling(symmetric, 7.5 * symmetric), 1.0)


def test_negated_fc_gives_minus_one(symmetric):
    np.testing.assert_allclose(global_coupling(symmetric, -symmetric), -1.0)


def test_global_matches_a_hand_computed_row():
    """One row worked out by hand, to pin the formula rather than a property."""
    sc = np.array([[0.0, 3.0, 4.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    fc = np.array([[0.0, 4.0, 3.0], [4.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    # row 0: dot = 3*4 + 4*3 = 24; |sc| = 5, |fc| = 5; 24 / 25 = 0.96
    assert global_coupling(sc, fc)[0] == pytest.approx(0.96)


def test_global_keeps_negative_fc():
    """Negative FC must contribute, not be clipped: the brief is explicit."""
    sc = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    fc = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]])
    # row 0: dot = 1*1 + 1*(-1) = 0, so the negative entry cancels the positive.
    assert global_coupling(sc, fc)[0] == pytest.approx(0.0)


def test_constant_profiles_become_nan():
    """A vertex whose row never varies carries no coupling information."""
    sc = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    fc = np.ones((3, 3))  # every column constant
    assert np.isnan(global_coupling(sc, fc)).all()


def test_triangular_input_is_symmetrized():
    """Passing the upper triangle with triangular=True matches the full matrix."""
    full = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    upper = np.triu(full)
    np.testing.assert_allclose(
        global_coupling(upper, upper, triangular=True), global_coupling(full, full)
    )


# --- discrete --------------------------------------------------------------


def test_discrete_and_global_differ_as_pearson_and_cosine():
    """Both worked out by hand on the same row, which pins each formula.

    sc row 0 = [0, 1, 5], fc row 0 = [0, 2, 3].

    Cosine (global): dot = 0*0 + 1*2 + 5*3 = 17, |sc| = sqrt(26),
    |fc| = sqrt(13), so 17 / sqrt(338) = 0.9246781.

    Pearson (discrete): centring gives sc = [-2, -1, 3] and
    fc = [-5/3, 1/3, 4/3]; dot = 10/3 - 1/3 + 4 = 7, |sc| = sqrt(14),
    |fc| = sqrt(42)/3, so 7 / (sqrt(14) sqrt(42) / 3) = 0.8660254.
    """
    sc = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 2.0], [5.0, 2.0, 0.0]])
    fc = np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 9.0], [3.0, 9.0, 0.0]])

    assert global_coupling(sc, fc)[0] == pytest.approx(17 / np.sqrt(338))
    assert discrete_coupling(sc, fc)[0] == pytest.approx(7 / (np.sqrt(14) * np.sqrt(42) / 3))

    # And they are genuinely different numbers, not the same one twice.
    assert global_coupling(sc, fc)[0] != pytest.approx(discrete_coupling(sc, fc)[0])


def test_centring_is_what_separates_the_two_forms():
    """Shift a row uniformly and only the cosine moves.

    The diagonal is cleared after any shift, so a shifted matrix is not a
    uniformly shifted *row*. Comparing the helpers directly avoids that.
    """
    from sbci.coupling import _cosine_rows, _pearson_rows

    sc = np.array([[1.0, 2.0, 6.0]])
    fc = np.array([[2.0, 3.0, 4.0]])

    np.testing.assert_allclose(_pearson_rows(sc, fc), _pearson_rows(sc, fc + 10.0))
    assert not np.allclose(_cosine_rows(sc, fc), _cosine_rows(sc, fc + 10.0))


def test_discrete_of_identical_matrices_is_one(symmetric):
    np.testing.assert_allclose(discrete_coupling(symmetric, symmetric), 1.0)


# --- local -----------------------------------------------------------------


def test_local_suppresses_small_regions(symmetric):
    """Below min_area a similarity is inflated, so the reference returns NaN."""
    labels = np.array([1, 1, 2, 2])
    assert np.isnan(local_coupling(symmetric, symmetric, labels)).all()


def test_local_computes_where_a_region_is_large_enough():
    n = 2 * MIN_AREA
    rng = np.random.default_rng(0)
    matrix = rng.random((n, n))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0.0)

    labels = np.repeat([1, 2], MIN_AREA)
    result = local_coupling(matrix, matrix, labels)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 1.0)


def test_local_mixes_large_and_small_regions():
    """Only the undersized region is NaN; the rest still computes."""
    n = MIN_AREA + 3
    rng = np.random.default_rng(1)
    matrix = rng.random((n, n))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0.0)

    labels = np.array([1] * MIN_AREA + [2, 2, 2])
    result = local_coupling(matrix, matrix, labels)
    assert np.isfinite(result[:MIN_AREA]).all()
    assert np.isnan(result[MIN_AREA:]).all()


def test_local_rejects_mismatched_labels(symmetric):
    with pytest.raises(ValueError, match="labels for"):
        local_coupling(symmetric, symmetric, np.array([1, 2]))


# --- shape and argument checks ---------------------------------------------


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="SC is"):
        global_coupling(np.zeros((3, 3)), np.zeros((4, 4)))


def test_rejects_a_nonsquare_matrix():
    with pytest.raises(ValueError, match="square matrices"):
        global_coupling(np.zeros((3, 4)), np.zeros((3, 4)))


# --- the connectome entry point --------------------------------------------


def test_coupling_rejects_two_structural_connectomes(connectome):
    with pytest.raises(ValueError, match="structural connectome"):
        structure_function_coupling(connectome, connectome)


def test_coupling_rejects_an_unknown_scope(connectome):
    with pytest.raises(ValueError, match="scope must be one of"):
        structure_function_coupling(connectome, connectome, scope="elsewhere")


def test_region_scope_needs_labels(connectome, sc_metadata):
    """scope='region' is meaningless without a parcellation, and says so."""
    from sbci.metadata import template

    fc = type(connectome)(
        data=connectome.data,
        area=connectome.area,
        mask=connectome.mask,
        metadata=template(
            "fc",
            normalization="none",
            registration_reference="fsaverage",
            pipeline_version="test",
            container_version="test",
            fc_nuisance_model="none",
        ),
    )
    with pytest.raises(ValueError, match="needs labels="):
        structure_function_coupling(connectome, fc, scope="region")
