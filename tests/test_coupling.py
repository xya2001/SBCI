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


def test_a_region_that_is_nan_throughout_is_skipped_not_spread():
    """``to_atlas(how="mean")`` gives an empty region a NaN row; it must not make every region NaN.

    The empty region is dropped like a constant profile and comes out NaN
    itself; the others are correlated over the regions that exist, exactly as
    if the empty one had never been in the atlas.
    """
    rng = np.random.default_rng(0)
    n = 6
    sc = rng.random((n, n))
    sc = sc + sc.T
    fc = rng.random((n, n)) - 0.5
    fc = fc + fc.T
    np.fill_diagonal(sc, 0.0)
    np.fill_diagonal(fc, 0.0)
    expected = discrete_coupling(sc, fc)

    def with_empty(matrix):
        return np.insert(np.insert(matrix, 2, np.nan, axis=0), 2, np.nan, axis=1)

    result = discrete_coupling(with_empty(sc), with_empty(fc))
    assert np.isnan(result[2]) and np.isfinite(np.delete(result, 2)).all()
    np.testing.assert_allclose(np.delete(result, 2), expected)
    # SC aggregated by mass has zeros there instead of NaN: the same answer
    sc_mass = with_empty(sc)
    sc_mass[2, :] = 0.0
    sc_mass[:, 2] = 0.0
    np.testing.assert_allclose(np.delete(discrete_coupling(sc_mass, with_empty(fc)), 2), expected)
    for compute in (global_coupling, lambda a, b: local_coupling(a, b, np.ones(n + 1), min_area=1)):
        out = compute(with_empty(sc), with_empty(fc))
        assert np.isnan(out[2]) and np.isfinite(np.delete(out, 2)).all()


def test_to_atlas_matrices_with_an_empty_region_couple_region_by_region(connectome):
    """End to end: an atlas with a region that has no cortical vertex, SC by mass and FC by mean."""
    from sbci.atlas import Atlas

    fc, _ = _toy_pair(connectome)
    gappy = Atlas(name="gappy", labels=np.array([1, 1, 2, 2, 0]), names=("A", "B", "C"))
    full = Atlas(name="full", labels=np.array([1, 1, 2, 2, 0]), names=("A", "B"))
    sc_m, fc_m = connectome.to_atlas(gappy), fc.to_atlas(gappy)
    assert np.isnan(fc_m[2]).all() and (sc_m[2] == 0.0).all()
    result = discrete_coupling(sc_m, fc_m)
    assert np.isnan(result[2])
    np.testing.assert_allclose(
        result[:2], discrete_coupling(connectome.to_atlas(full), fc.to_atlas(full))
    )
    # both by mean, so both carry the NaN row
    both = discrete_coupling(connectome.to_atlas(gappy, how="mean"), fc_m)
    assert np.isnan(both[2]) and np.isfinite(both[:2]).all()


def test_an_empty_region_does_not_bring_back_a_region_without_sc():
    """A zero SC profile with the empty region's NaN in it is still constant, and still dropped.

    A NaN equals nothing, so the NaN made that profile pass for varying: it was
    kept, entered every other region's comparison, and adding an empty region
    to six changed all five values. Every form now gives the regions present
    exactly what it gives without the empty one.
    """
    rng = np.random.default_rng(0)
    n = 6
    sc = rng.random((n, n))
    sc = sc + sc.T
    fc = rng.random((n, n)) - 0.5
    fc = fc + fc.T
    np.fill_diagonal(sc, 0.0)
    np.fill_diagonal(fc, 0.0)
    sc[2, :] = 0.0  # region 2 has no SC
    sc[:, 2] = 0.0

    def with_empty(matrix):
        return np.pad(matrix, (0, 1), constant_values=np.nan)

    def local(a, b, **kwargs):
        return local_coupling(a, b, np.ones(a.shape[0], dtype=int), min_area=1, **kwargs)

    for compute in (global_coupling, discrete_coupling, local):
        expected = compute(sc, fc)
        assert np.isnan(expected[2]) and np.isfinite(np.delete(expected, 2)).all()
        result = compute(with_empty(sc), with_empty(fc))
        assert np.isnan(result[n])
        np.testing.assert_array_equal(result[:n], expected)
        upper = compute(np.triu(with_empty(sc)), np.triu(with_empty(fc)), triangular=True)
        np.testing.assert_array_equal(upper[:n], compute(np.triu(sc), np.triu(fc), triangular=True))


def test_an_atlas_with_an_empty_region_and_one_without_sc_couples_as_without_the_empty_one(
    sc_metadata,
):
    """End to end, through ``to_atlas``: region D has no streamline, region E no cortical vertex.

    E covers only the medial wall, so FC by mean reads NaN along its row and
    column, and so does SC by mean. With SC by mass or by mean, D comes out NaN
    and A to C exactly as on the same atlas without E.
    """
    from sbci.atlas import Atlas
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed
    from sbci.metadata import template

    rng = np.random.default_rng(3)
    n = 7
    area = rng.uniform(1.0, 3.0, n)
    mask = np.arange(n) < 6  # vertex 6 is the medial wall
    structural = rng.random((n, n))
    structural = structural + structural.T
    structural[[5, 6], :] = 0.0  # no streamline ends at vertex 5, nor on the medial wall
    structural[:, [5, 6]] = 0.0
    np.fill_diagonal(structural, 0.0)
    structural /= area @ structural @ area
    functional = np.tanh(rng.standard_normal((n, n)))
    functional = (functional + functional.T) / 2
    functional[6, :] = 0.0
    functional[:, 6] = 0.0
    fc_metadata = template(
        "fc",
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="test",
        container_version="test",
        fc_nuisance_model="none",
    )
    sc, fc = (
        ContinuousConnectome(
            data=to_condensed(matrix).astype(np.float32), area=area, mask=mask, metadata=metadata
        )
        for matrix, metadata in ((structural, sc_metadata), (functional, fc_metadata))
    )
    gappy = Atlas(name="gappy", labels=np.array([1, 1, 2, 3, 3, 4, 5]), names=tuple("ABCDE"))
    full = Atlas(name="full", labels=np.array([1, 1, 2, 3, 3, 4, 0]), names=tuple("ABCD"))
    fc_gappy, fc_full = fc.to_atlas(gappy), fc.to_atlas(full)
    assert np.isnan(fc_gappy[4]).all()
    for how in ("mass", "mean"):
        sc_gappy, sc_full = sc.to_atlas(gappy, how=how), sc.to_atlas(full, how=how)
        assert np.all(sc_gappy[3, :3] == 0.0)  # D has no SC
        expected = discrete_coupling(sc_full, fc_full)
        assert np.isnan(expected[3]) and np.isfinite(expected[:3]).all()
        result = discrete_coupling(sc_gappy, fc_gappy)
        assert np.isnan(result[4])
        np.testing.assert_array_equal(result[:4], expected)


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


def _toy_pair(connectome):
    """The toy SC with a toy FC of signed correlations on the same five vertices."""
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed
    from sbci.metadata import template

    correlations = np.array(
        [
            [0.0, 0.8, 0.3, -0.2, 0.0],
            [0.8, 0.0, 0.5, 0.1, 0.0],
            [0.3, 0.5, 0.0, 0.6, 0.0],
            [-0.2, 0.1, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    fc = ContinuousConnectome(
        data=to_condensed(correlations).astype(np.float32),
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
    return fc, correlations


def test_the_connectome_entry_point_matches_the_array_functions(connectome):
    """The method is the array functions on the stored (float32) values, expanded in float64."""
    fc, _ = _toy_pair(connectome)
    sc_dense, fc_dense = connectome.dense(np.float64), fc.dense(np.float64)
    labels = np.array([1, 1, 2, 2, 0])
    np.testing.assert_array_equal(connectome.coupling(fc), global_coupling(sc_dense, fc_dense))
    np.testing.assert_array_equal(
        connectome.coupling(fc, scope="discrete"), discrete_coupling(sc_dense, fc_dense)
    )
    np.testing.assert_array_equal(
        connectome.coupling(fc, scope="region", labels=labels, min_area=1),
        local_coupling(sc_dense, fc_dense, labels, min_area=1),
    )


def test_labels_without_the_region_scope_are_refused_plainly(connectome):
    fc, _ = _toy_pair(connectome)
    with pytest.raises(ValueError, match="labels= goes with scope='region'"):
        connectome.coupling(fc, labels=np.ones(5, dtype=int))


def test_connectomes_on_different_grids_are_refused(connectome):
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed

    fc, _ = _toy_pair(connectome)
    smaller = ContinuousConnectome(
        data=to_condensed(np.zeros((4, 4))).astype(np.float32),
        area=fc.area[:4],
        mask=fc.mask[:4],
        metadata=fc.metadata,
    )
    with pytest.raises(ValueError, match="different grids"):
        connectome.coupling(smaller)
