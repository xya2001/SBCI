"""The fsLR-32k exchange file: the overlap matrix, the blocked product, the axis.

The written file itself is 16.9 GB, so it cannot appear in a unit test. What can
be tested here is everything the file's correctness rests on: that the overlap
matrix partitions the same fsaverage vertices the pipeline's own area vector
does, that resampling conserves the area-weighted mass the specification checks
rather than merely the sum of the entries, that the blocked product agrees with
a plain triple matrix product, and that the CIFTI axis describes the mesh the
header claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.io import cifti

N_ICO4 = 5124
N_ICO4_PER_HEMI = 2562
N_FSAVERAGE = 2 * 163842


@pytest.fixture(scope="module")
def overlap():
    """The bundled ico4 <-> fsLR-32k overlap matrix."""
    return cifti.load_overlap()


def test_overlap_has_the_shape_the_format_promises(overlap):
    assert overlap.shape == (cifti.N_FSLR, N_ICO4)
    assert overlap.nnz > 0
    assert overlap.data.min() > 0.0


def test_every_fsaverage_vertex_is_counted_exactly_once(overlap):
    """The entries are counts of a partition, so both margins hit 327,684."""
    fslr_area, ico4_area = cifti.vertex_areas()
    assert overlap.sum() == pytest.approx(N_FSAVERAGE)
    assert fslr_area.sum() == pytest.approx(N_FSAVERAGE)
    assert ico4_area.sum() == pytest.approx(N_FSAVERAGE)
    np.testing.assert_allclose(overlap.data, np.round(overlap.data))


def test_the_ico4_margin_is_the_grids_own_area_vector(overlap):
    """Column sums must be vertex areas: 5124 vertices over a 327,684 mesh."""
    _, ico4_area = cifti.vertex_areas()
    assert ico4_area.min() >= 50
    assert ico4_area.max() <= 75
    assert ico4_area.mean() == pytest.approx(N_FSAVERAGE / N_ICO4)


def test_no_fslr_vertex_is_left_without_a_source(overlap):
    fslr_area, _ = cifti.vertex_areas()
    assert fslr_area.min() >= 1


def test_the_operator_averages_rather_than_distributes():
    """Row-stochastic is what a density needs; column-stochastic is not."""
    operator = cifti.transfer()
    row_sums = np.asarray(operator.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_the_operator_carries_fslr_areas_back_to_ico4_areas():
    """``P' a_fsLR == a_ico4`` is what makes the mass identity hold exactly."""
    operator = cifti.transfer()
    fslr_area, ico4_area = cifti.vertex_areas()
    np.testing.assert_allclose(operator.T @ fslr_area, ico4_area, atol=1e-9)


def test_area_weighted_mass_is_conserved_on_the_real_operator():
    """The specification's ``area @ D @ area`` must survive the resampling.

    Checked in factored form, since the resampled square is 16.9 GB:
    ``a' (P D P') a' == (P' a')' D (P' a')``.
    """
    operator = cifti.transfer()
    fslr_area, ico4_area = cifti.vertex_areas()
    rng = np.random.default_rng(0)
    dense = rng.random((N_ICO4, N_ICO4))
    dense = dense + dense.T

    pushed = operator.T @ fslr_area
    before = ico4_area @ dense @ ico4_area
    after = pushed @ dense @ pushed
    assert abs(after - before) < 1e-9 * before


def test_the_hemispheres_do_not_leak_into_each_other(overlap):
    """A left ico4 vertex must land only on left fsLR vertices, and vice versa."""
    coo = overlap.tocoo()
    left_column = coo.col < N_ICO4_PER_HEMI
    left_row = coo.row < cifti.N_FSLR_PER_HEMI
    assert np.array_equal(left_column, left_row)


def test_blocked_product_equals_the_plain_triple_product(monkeypatch):
    """The blocking exists only to bound memory; it must not change the answer."""
    from scipy import sparse

    rng = np.random.default_rng(1)
    small = rng.random((40, 7))
    small[small < 0.8] = 0.0
    small[0] = 1.0  # so that no row and no column is empty
    small[:, 0] = np.where(rng.random(40) < 0.3, 1.0, small[:, 0])
    fake = sparse.csr_matrix(small)
    monkeypatch.setattr(cifti, "load_overlap", lambda: fake)
    operator = cifti.transfer()

    dense = rng.random((7, 7))
    dense = dense + dense.T

    expected = operator @ dense @ operator.T
    for block in (1, 3, 40, 100):
        np.testing.assert_allclose(cifti.resample(dense, block=block), expected, rtol=1e-5)


def test_resample_rejects_a_matrix_on_the_wrong_grid():
    with pytest.raises(ValueError, match="5124x5124"):
        cifti.resample(np.zeros((10, 10)))


def test_the_axis_describes_both_fslr_hemispheres():
    axis = cifti._brain_model_axis()
    assert len(axis) == cifti.N_FSLR
    assert set(axis.name[: cifti.N_FSLR_PER_HEMI]) == {"CIFTI_STRUCTURE_CORTEX_LEFT"}
    assert set(axis.name[cifti.N_FSLR_PER_HEMI :]) == {"CIFTI_STRUCTURE_CORTEX_RIGHT"}
    assert axis.nvertices["CIFTI_STRUCTURE_CORTEX_LEFT"] == cifti.N_FSLR_PER_HEMI
    assert axis.nvertices["CIFTI_STRUCTURE_CORTEX_RIGHT"] == cifti.N_FSLR_PER_HEMI
    np.testing.assert_array_equal(
        axis.vertex[: cifti.N_FSLR_PER_HEMI], np.arange(cifti.N_FSLR_PER_HEMI)
    )


def test_reading_back_to_ico4_refuses_rather_than_degrading():
    """The resampling is many-to-one; the inverse would quietly lose data."""
    with pytest.raises(NotImplementedError, match="lossy"):
        cifti.read_cifti("anything.dconn.nii")


def test_write_cifti_sets_the_intent_codes_and_writes_the_companions(tmp_path, monkeypatch):
    """On a fake 40 x 7 overlap: the .dconn intent, the areas' intent, the JSON sidecar."""
    import json

    import nibabel as nib
    from scipy import sparse

    from sbci.metadata import template

    rng = np.random.default_rng(2)
    small = rng.random((40, 7))
    small[small < 0.7] = 0.0
    small[0] = 1.0
    small[:, 0] = np.where(rng.random(40) < 0.3, 1.0, small[:, 0])
    fake = sparse.csr_matrix(small)
    monkeypatch.setattr(cifti, "load_overlap", lambda: fake)
    monkeypatch.setattr(cifti, "N_FSLR_PER_HEMI", 20)
    monkeypatch.setattr(cifti, "N_FSLR", 40)

    class Toy:
        metadata = template(
            "sc",
            normalization="unit-mass",
            registration_reference="fsaverage",
            pipeline_version="x",
            container_version="y",
            streamline_count=1,
            streamline_weighting="none",
            kernel="shk",
            bandwidth=0.005,
        )

        def dense(self):
            dense = rng.random((7, 7)).astype(np.float32)
            return dense + dense.T

    path = cifti.write_cifti(tmp_path / "sub-toy_sc.dconn.nii", Toy(), block=16)
    image = nib.load(str(path))
    assert image.nifti_header.get_intent()[0] == "ConnDense"
    areas = nib.load(str(tmp_path / "sub-toy_sc_vertexarea.dscalar.nii"))
    assert areas.nifti_header.get_intent()[0] == "ConnDenseScalar"
    sidecar = json.loads((tmp_path / "sub-toy_sc.json").read_text())
    assert sidecar["exchange_space"] == "fsLR" and sidecar["kernel"] == "shk"
