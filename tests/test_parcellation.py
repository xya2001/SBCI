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
divides each off-diagonal entry by the product of the two region areas. A
within-region entry is a mean over the region's distinct vertex pairs, so its
denominator leaves out the self-pairs ``sum a_i^2`` (``A: 5``, ``B: 10``) that
the zero vertex diagonal never carried: ``A`` divides by ``9 - 5 = 4`` and
``B`` by ``16 - 10 = 6``, and ``C``, a single vertex with no pair, is ``NaN``.
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
        [4 / 4, 51 / 12, 36 / 6],
        [51 / 12, 48 / 6, 56 / 8],
        [36 / 6, 56 / 8, np.nan],
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
    """FC aggregation goes through arctanh, so a mixed block is not the plain average.

    Between ``A`` and ``B`` the pair ``(0, 2)`` carries 0.9 with weight 3 and
    the other three pairs 0.3 with weights 1, 6 and 2.
    """
    correlations = np.array(
        [
            [0.0, 0.5, 0.9, 0.3, 0.0],
            [0.5, 0.0, 0.3, 0.3, 0.0],
            [0.9, 0.3, 0.0, 0.5, 0.0],
            [0.3, 0.3, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    plain = parcellate(correlations, atlas, area, how="mean", fisher_z=False)
    fisher = parcellate(correlations, atlas, area, how="mean", fisher_z=True)
    assert plain[0, 1] == pytest.approx((3 * 0.9 + 9 * 0.3) / 12)
    assert fisher[0, 1] == pytest.approx(np.tanh((3 * np.arctanh(0.9) + 9 * np.arctanh(0.3)) / 12))
    assert not np.isclose(plain[0, 1], fisher[0, 1])
    # a region whose pairs all carry one value gives that value either way
    assert plain[0, 0] == pytest.approx(0.5) and fisher[0, 0] == pytest.approx(0.5)


def _all_pairs(n_vertices, value):
    """``n_vertices`` vertices whose every distinct pair carries ``value``."""
    matrix = np.full((n_vertices, n_vertices), value)
    np.fill_diagonal(matrix, 0.0)
    return matrix


@pytest.mark.parametrize("fisher_z", [False, True])
@pytest.mark.parametrize("n_vertices", [2, 4, 20])
def test_within_region_mean_runs_over_distinct_pairs(n_vertices, fisher_z):
    """All pairs at 0.8 give a within-region mean of 0.8, not 0.8 (1 - sum a_i^2 / A^2)."""
    from sbci.atlas import Atlas

    atlas = Atlas(name="one", labels=np.ones(n_vertices, dtype=int), names=("A",))
    area = np.random.default_rng(n_vertices).uniform(0.5, 2.0, n_vertices)  # unequal, on purpose
    result = parcellate(_all_pairs(n_vertices, 0.8), atlas, area, how="mean", fisher_z=fisher_z)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(0.8, rel=1e-12)


def test_the_self_pair_correction_touches_the_diagonal_only(dense, atlas, area):
    """Off the diagonal the mean is still mass over the product of the two region areas."""
    mass = parcellate(dense, atlas, area, how="mass")
    mean = parcellate(dense, atlas, area, how="mean")
    totals = np.array([3.0, 4.0, 2.0])
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(mean[off], (mass / np.outer(totals, totals))[off])


def test_a_region_without_a_distinct_pair_has_nan_on_the_diagonal(dense, atlas, area):
    """One vertex (C), or none at this resolution: no pair to average, so NaN rather than zero."""
    from sbci.atlas import Atlas

    mean = parcellate(dense, atlas, area, how="mean")
    assert np.isnan(mean[2, 2])
    assert np.isfinite(mean[:2, :]).all() and np.isfinite(mean[2, :2]).all()
    gappy = Atlas(name="gappy", labels=np.array([1, 1, 2, 2, 0]), names=("A", "B", "C"))
    assert np.isnan(parcellate(dense, gappy, area, how="mean")[2, 2])
    # ``mass`` has nothing to divide by, and stays at zero there
    assert parcellate(dense, atlas, area, how="mass")[2, 2] == 0.0


def test_ones_on_the_vertex_diagonal_change_nothing(dense, atlas, area):
    """``np.corrcoef`` puts ones on the diagonal; the self-pairs are excluded by contract.

    Left in, they sat in the within-region mass but not in its denominator:
    ``A`` would have read ``(4 + 1*1*1 + 2*1*2) / 4 = 2.25`` instead of 1.
    """
    with_ones = dense + np.eye(5)
    for how in ("mass", "mean"):
        np.testing.assert_allclose(
            parcellate(with_ones, atlas, area, how=how), parcellate(dense, atlas, area, how=how)
        )
    assert parcellate(with_ones, atlas, area, how="mean")[0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(with_ones, dense + np.eye(5))  # the caller's matrix is left alone
    correlations = np.corrcoef(np.random.default_rng(0).standard_normal((5, 30)))
    np.testing.assert_allclose(np.diag(correlations), 1.0)
    zeroed = correlations.copy()
    np.fill_diagonal(zeroed, 0.0)
    np.testing.assert_allclose(
        parcellate(correlations, atlas, area, how="mean", fisher_z=True),
        parcellate(zeroed, atlas, area, how="mean", fisher_z=True),
    )


def test_an_empty_region_is_nan_throughout_under_mean_and_zero_under_mass(dense, atlas, area):
    """No vertex at this resolution means no pair with anyone, not a mean of zero with everyone."""
    from sbci.atlas import Atlas

    gappy = Atlas(name="gappy", labels=np.array([1, 1, 2, 2, 0]), names=("A", "B", "C"))
    mean = parcellate(dense, gappy, area, how="mean")
    assert np.isnan(mean[2, :]).all() and np.isnan(mean[:, 2]).all()
    np.testing.assert_allclose(mean[:2, :2], EXPECTED_MEAN[:2, :2])
    mass = parcellate(dense, gappy, area, how="mass")
    assert (mass[2, :] == 0.0).all() and (mass[:, 2] == 0.0).all()
    np.testing.assert_allclose(mass[:2, :2], EXPECTED_MASS[:2, :2])
    # a region whose only vertex carries no area -- the medial wall -- is empty too
    walled = parcellate(dense, atlas, np.array([1.0, 2.0, 3.0, 1.0, 0.0]), how="mean")
    assert np.isnan(walled[2, :]).all() and np.isnan(walled[:, 2]).all()
    assert np.isfinite(walled[:2, :2]).all()
    # one vertex with area is a region with pairs to others: NaN on its diagonal only
    single = parcellate(dense, atlas, area, how="mean")
    assert np.isnan(single[2, 2]) and np.isfinite(single[2, :2]).all()


def test_region_weights_layout(atlas, area):
    weights, ids = region_weights(atlas, area)
    assert weights.shape == (5, 3)
    np.testing.assert_array_equal(ids, [1, 2, 3])
    np.testing.assert_allclose(np.asarray(weights.sum(axis=0)).ravel(), [3.0, 4.0, 2.0])
    np.testing.assert_array_equal(
        weights.toarray(), [[1, 0, 0], [2, 0, 0], [0, 3, 0], [0, 1, 0], [0, 0, 2]]
    )


def test_fisher_z_goes_with_mean_only(dense, atlas, area):
    with pytest.raises(ValueError, match="use how='mean'"):
        parcellate(dense, atlas, area, how="mass", fisher_z=True)


def test_rejects_unknown_how(dense, atlas, area):
    with pytest.raises(ValueError, match="how must be one of"):
        parcellate(dense, atlas, area, how="median")


def test_rejects_mismatched_grid(dense, atlas):
    with pytest.raises(ValueError, match="vertices"):
        parcellate(dense, atlas, np.ones(4))
