"""``to_atlas`` against a hand-computed three-region case.

The arithmetic below is worked out by hand from the fixtures in ``conftest``:
areas ``[1, 2, 3, 1, 2]``, regions ``A = {0, 1}``, ``B = {2, 3}``, ``C = {4}``.
Each region-pair entry is the sum of ``area_a * D[a, b] * area_b`` over vertex
pairs crossing the two regions.

    mass(A, A) = 1*1*2 + 2*1*1                     = 4
    mass(A, B) = 1*2*3 + 1*3*1 + 2*5*3 + 2*6*1     = 51
    mass(A, C) = 1*4*2 + 2*7*2                     = 36
    mass(B, B) = 3*8*1 + 1*8*3                     = 48
    mass(B, C) = 3*9*2 + 1*1*2                     = 56
    mass(C, C) =                                     0

Region areas are ``A = 3``, ``B = 4``, ``C = 2``, so the ``"mean"`` form
divides each entry by the product of the two region areas.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.parcellation import parcellate, region_weights

EXPECTED_MASS = np.array(
    [
        [4.0, 51.0, 36.0],
        [51.0, 48.0, 56.0],
        [36.0, 56.0, 0.0],
    ]
)

EXPECTED_MEAN = np.array(
    [
        [4 / 9, 51 / 12, 36 / 6],
        [51 / 12, 48 / 16, 56 / 8],
        [36 / 6, 56 / 8, 0.0],
    ]
)


def test_mass_matches_hand_computation(dense, atlas, area):
    result = parcellate(dense, atlas, area, how="mass")
    np.testing.assert_allclose(result, EXPECTED_MASS)


def test_mean_matches_hand_computation(dense, atlas, area):
    result = parcellate(dense, atlas, area, how="mean")
    np.testing.assert_allclose(result, EXPECTED_MEAN)


def test_result_is_symmetric(dense, atlas, area):
    result = parcellate(dense, atlas, area, how="mass")
    np.testing.assert_allclose(result, result.T)


def test_mass_preserves_total(dense, atlas, area):
    """Total region-level mass equals total area-weighted vertex-level mass."""
    result = parcellate(dense, atlas, area, how="mass")
    assert result.sum() == pytest.approx(area @ dense @ area)


def test_mean_is_invariant_to_area_scaling(dense, atlas, area):
    """Doubling every area weight leaves the density unchanged."""
    baseline = parcellate(dense, atlas, area, how="mean")
    scaled = parcellate(dense, atlas, 2 * area, how="mean")
    np.testing.assert_allclose(baseline, scaled)


def test_fisher_z_differs_from_plain_mean(atlas, area):
    """FC aggregation goes through arctanh, so it is not the plain average."""
    correlations = np.full((5, 5), 0.8)
    np.fill_diagonal(correlations, 0.0)
    plain = parcellate(correlations, atlas, area, how="mean", fisher_z=False)
    fisher = parcellate(correlations, atlas, area, how="mean", fisher_z=True)
    assert not np.allclose(plain, fisher)


def test_region_weights_layout(atlas, area):
    weights, ids = region_weights(atlas, area)
    assert weights.shape == (5, 3)
    np.testing.assert_array_equal(ids, [1, 2, 3])
    np.testing.assert_allclose(weights.sum(axis=0), [3.0, 4.0, 2.0])


def test_rejects_unknown_how(dense, atlas, area):
    with pytest.raises(ValueError, match="how must be one of"):
        parcellate(dense, atlas, area, how="median")


def test_rejects_mismatched_grid(dense, atlas):
    with pytest.raises(ValueError, match="vertices"):
        parcellate(dense, atlas, np.ones(4))
