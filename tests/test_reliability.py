"""Test-retest: the intraclass correlation, connectome identification, and split streamlines.

The ICC is checked against Shrout and Fleiss's published example (in the
docstring) and against what its two kinds are defined to do with a shift
between sessions; identification against cohorts whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci import stats
from sbci.smoothing import Endpoints
from sbci.stats import icc, identification


def test_agreement_counts_a_shift_between_sessions_and_consistency_forgives_it():
    rng = np.random.default_rng(0)
    trait = rng.standard_normal((40, 6))
    first = trait + 0.2 * rng.standard_normal((40, 6))
    second = trait + 0.2 * rng.standard_normal((40, 6))
    shifted = second + 3.0  # the second session reads higher by a constant
    same = icc([first, second])
    np.testing.assert_allclose(
        icc([first, shifted], "consistency"), icc([first, second], "consistency")
    )
    assert (icc([first, shifted]) < same - 0.3).all()
    assert (same > 0.85).all() and (same < 1).all()
    np.testing.assert_allclose(icc([trait, trait]), 1.0)


def test_a_feature_with_a_gap_or_without_variance_reads_nan():
    rng = np.random.default_rng(1)
    first, second = rng.standard_normal((2, 20, 4))
    first[3, 1] = np.nan
    first[:, 2] = second[:, 2] = 5.0
    values = icc([first, second])
    assert np.isnan(values[[1, 2]]).all() and np.isfinite(values[[0, 3]]).all()
    with pytest.raises(ValueError, match="two sessions and two subjects"):
        icc([first])
    with pytest.raises(ValueError, match="kind must be"):
        icc([first, second], kind="absolute")


def test_identification_finds_each_subject_and_says_how_sure():
    rng = np.random.default_rng(2)
    trait = rng.standard_normal((30, 500))
    first = trait + 0.5 * rng.standard_normal((30, 500))
    second = trait + 0.5 * rng.standard_normal((30, 500))
    result = identification(first, second)
    assert result.accuracy == (1.0, 1.0) and result.within > result.between + 0.5
    assert result.similarity.shape == (30, 30) and result.features == 500
    shuffled = identification(first, second[rng.permutation(30)])
    assert max(shuffled.accuracy) < 0.3
    # Float32 input agrees; a feature missing anywhere is left out.
    single = identification(first.astype(np.float32), second.astype(np.float32))
    np.testing.assert_allclose(single.similarity, result.similarity, atol=1e-5)
    first[0, :10] = np.nan
    assert identification(first, second).features == 490


def test_identification_keeps_float64_precision_where_float32_sums_drift():
    # Like smoothed SC: mostly near zero, a few large values. Once each row's mean is
    # removed, a float32 sum adds a great many nearly equal terms and drifts upward
    # (to r = 1.016 on two halves of one subject's streamlines); float64 does not.
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1e-3, size=(4, 500_000))
    hot = rng.random(base.shape) < 0.01
    base[hot] = rng.lognormal(0.0, 1.5, size=hot.sum())
    first = (base * rng.lognormal(0.0, 0.05, size=base.shape)).astype(np.float32)
    second = (base * rng.lognormal(0.0, 0.05, size=base.shape)).astype(np.float32)
    exact = np.corrcoef(first.astype(np.float64), second.astype(np.float64))[:4, 4:]
    result = identification(first, second)
    np.testing.assert_allclose(result.similarity, exact, rtol=0, atol=1e-9)
    assert (np.diag(result.similarity) < 1).all()


def test_identification_by_blocks_matches_one_pass(monkeypatch):
    rng = np.random.default_rng(3)
    first, second = rng.standard_normal((2, 6, 1000))
    first[2, 17] = np.nan
    whole = identification(first, second)
    monkeypatch.setattr(stats, "_BLOCK_ELEMENTS", 6 * 64)  # 64 features a block
    blocked = identification(first, second)
    np.testing.assert_allclose(blocked.similarity, whole.similarity, rtol=0, atol=1e-12)
    assert blocked.features == whole.features == 999
    full = np.delete(first, 17, axis=1), np.delete(second, 17, axis=1)
    np.testing.assert_allclose(identification(*full).similarity, whole.similarity, atol=1e-12)


def test_take_keeps_the_streamlines_asked_for_and_their_positions():
    rng = np.random.default_rng(3)
    count = 50
    endpoints = Endpoints(
        surf_in=rng.integers(0, 2, count),
        surf_out=rng.integers(0, 2, count),
        vtx_in=rng.integers(0, 10, count),
        vtx_out=rng.integers(0, 10, count),
        n_per_hemi=10,
        tri_in=rng.integers(0, 16, count),
        tri_out=rng.integers(0, 16, count),
        bary_in=rng.dirichlet(np.ones(3), count),
        bary_out=rng.dirichlet(np.ones(3), count),
    )
    half = rng.permutation(count) < count // 2
    first, second = endpoints.take(half), endpoints.take(~half)
    assert first.n_streamlines + second.n_streamlines == count
    np.testing.assert_array_equal(first.vtx_out, endpoints.vtx_out[half])
    np.testing.assert_array_equal(second.bary_in, endpoints.bary_in[~half])
    with pytest.raises(IndexError):
        endpoints.take([count])
    with pytest.raises(ValueError, match="a mask over 50 streamlines"):
        endpoints.take(np.ones(3, dtype=bool))


# --- the seventh review -----------------------------------------------------------


def test_a_feature_that_never_varies_is_nan_whatever_its_value():
    """A constant 0.1 left rounding in the mean squares, which over themselves read 1."""
    for value in (0.1, 0.3, 1e6 / 3, -7.25):
        constant = np.full((6, 1), value)
        assert np.isnan(icc([constant, constant])).all()
        assert np.isnan(icc([constant, constant], kind="consistency")).all()


def test_a_resmoothed_half_records_its_own_streamline_count():
    """A half of 20,000 streamlines said 20,000 in its metadata."""
    import sbci
    from sbci.connectome import ContinuousConnectome

    sc = sbci.example(modality="sc")
    half = sc.endpoints.take(np.arange(sc.endpoints.n_streamlines) % 2 == 0)
    holder = ContinuousConnectome(sc.data, sc.area, sc.mask, sc.metadata, endpoints=half)
    smoothed = holder.smooth(kernel="shk", bandwidth=0.005, mask_medial_wall=True)
    assert smoothed.metadata.fields["streamline_count"] == half.n_streamlines
    assert half.n_streamlines == sc.endpoints.n_streamlines // 2


# --- the eighth review ------------------------------------------------------------


def test_take_refuses_a_mask_written_as_integers():
    """0s and 1s were read as indices: streamlines 0 and 1, over and over."""
    ends = Endpoints([0, 0, 1, 1], [1, 1, 0, 0], [3, 5, 7, 9], [2, 4, 6, 8], n_per_hemi=10)
    with pytest.raises(ValueError, match="a mask written as integers"):
        ends.take(np.array([1, 0, 1, 0]))
    assert ends.take(np.array([1, 0, 1, 0], dtype=bool)).vtx_in.tolist() == [3, 7]
    assert ends.take([1, 0]).vtx_in.tolist() == [5, 3]  # indices of another length stay indices


def test_taking_no_streamlines_or_indices_that_are_not_integers():
    """take([]) raised an IndexError, NumPy reading [] as floats."""
    both = Endpoints([0, 1, 1], [1, 0, 1], [3, 5, 7], [2, 4, 6], n_per_hemi=10)
    assert both.take([]).n_streamlines == 0
    with pytest.raises(TypeError, match="integer indices or a boolean mask"):
        both.take([0.0, 2.0])


def test_identification_needs_two_features_and_says_which_way_is_which():
    with pytest.raises(ValueError, match="two at least"):
        identification(np.arange(3.0)[:, None], np.arange(3.0)[:, None] + 1)
    rng = np.random.default_rng(6)
    first = rng.standard_normal((3, 400))
    second = first + 0.1 * rng.standard_normal((3, 400))
    second[2] = 0.7 * first[1] + 0.3 * first[2]  # subject 2's second session looks like subject 1
    result = identification(first, second)
    assert result.accuracy == (1.0, 2 / 3)  # first to second, then second to first


def test_a_feature_constant_to_rounding_has_no_icc_and_a_small_spread_has_one():
    rng = np.random.default_rng(7)
    constant = np.full((2, 6, 1), 0.1)
    assert np.isnan(icc(constant)).all()
    subject = rng.standard_normal(6)
    small = 1e3 + 1e-3 * (subject[None, :, None] + 0.1 * rng.standard_normal((2, 6, 1)))
    assert np.isfinite(icc(small)).all() and icc(small)[0] > 0.9
