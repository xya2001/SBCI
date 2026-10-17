"""Condensed / dense round trips."""

from __future__ import annotations

import numpy as np
import pytest

from sbci import spec
from sbci.grid import condensed_size, hemisphere_labels, n_from_condensed, to_condensed, to_dense


def test_round_trip(dense):
    np.testing.assert_allclose(to_dense(to_condensed(dense)), dense)


def test_dense_is_symmetric_with_zero_diagonal():
    result = to_dense(np.arange(6, dtype=float))
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0.0)


def test_condensed_size_matches_spec():
    assert condensed_size(spec.N_VERTICES) == spec.N_UPPER
    assert n_from_condensed(spec.N_UPPER) == spec.N_VERTICES


def test_invalid_condensed_length():
    with pytest.raises(ValueError, match="not a valid"):
        n_from_condensed(11)


def test_to_condensed_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        to_condensed(np.zeros((3, 4)))


def test_hemisphere_split_is_left_then_right():
    labels = hemisphere_labels()
    assert labels.size == spec.N_VERTICES
    assert (labels[: spec.N_VERTICES_PER_HEMI] == 0).all()
    assert (labels[spec.N_VERTICES_PER_HEMI :] == 1).all()
