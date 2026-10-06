"""Test-retest: the intraclass correlation, connectome identification, and split streamlines.

The ICC is checked against Shrout and Fleiss's published example (in the
docstring) and against what its two kinds are defined to do with a shift
between sessions; identification against cohorts whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    # Float32 stays float32 and agrees; a feature missing anywhere is left out.
    single = identification(first.astype(np.float32), second.astype(np.float32))
    np.testing.assert_allclose(single.similarity, result.similarity, atol=1e-5)
    first[0, :10] = np.nan
    assert identification(first, second).features == 490


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
