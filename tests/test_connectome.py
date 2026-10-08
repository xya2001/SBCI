"""The public object: loading, seeding, aggregating, and what it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from sbci import ContinuousConnectome
from sbci.io import write_hdf5
from sbci.metadata import MetadataError, template
from sbci.smoothing import KERNELS


def _write(tmp_path, connectome, metadata=None, name="sub-toy_sc.h5"):
    return write_hdf5(
        tmp_path / name,
        data=connectome.data,
        area=connectome.area,
        mask=connectome.mask,
        metadata=metadata or connectome.metadata,
        compression=None,
    )


def test_load_round_trips(tmp_path, connectome):
    loaded = ContinuousConnectome.load(_write(tmp_path, connectome))
    np.testing.assert_allclose(loaded.data, connectome.data)
    assert loaded.modality == "sc"
    assert loaded.n_vertices == 5


def test_load_refuses_missing_metadata_keys(tmp_path, connectome):
    """The headline requirement: an underspecified file does not load."""
    import h5py

    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("connectivity", data=connectome.data)
        handle.create_dataset("area", data=connectome.area)
        handle.create_dataset("mask", data=connectome.mask)
        handle.create_dataset("metadata", data=template("sc").to_json())

    with pytest.raises(MetadataError, match="missing required metadata keys"):
        ContinuousConnectome.load(path)


def test_load_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "sub-toy_sc.mat"
    path.touch()
    with pytest.raises(ValueError, match="unrecognized connectome file"):
        ContinuousConnectome.load(path)


def test_dense_is_symmetric_with_zero_diagonal(connectome):
    matrix = connectome.dense()
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)


def test_seed_vertex_matches_the_dense_row(connectome):
    matrix = connectome.dense()
    for vertex in range(connectome.n_vertices):
        np.testing.assert_allclose(connectome.seed(vertex=vertex), matrix[vertex], rtol=1e-6)


def test_seed_region_is_the_area_weighted_mean_profile(connectome):
    member = np.array([True, True, False, False, False])
    expected = connectome.dense() @ np.where(member, connectome.area, 0.0)
    np.testing.assert_allclose(connectome.seed(region=member), expected / 3.0, rtol=1e-6)


def test_seed_region_leaves_medial_wall_vertices_out(connectome):
    """Vertex 4 is medial wall: adding it to a region must not scale the profile down."""
    cortical = np.array([True, True, False, False, False])
    with_wall = cortical.copy()
    with_wall[4] = True
    np.testing.assert_allclose(connectome.seed(region=with_wall), connectome.seed(region=cortical))
    weights = np.where(cortical, connectome.area, 0.0)
    expected = (connectome.dense().astype(np.float64) @ weights) / weights.sum()
    np.testing.assert_allclose(connectome.seed(region=with_wall), expected, rtol=1e-6)


def test_seed_region_inside_the_medial_wall_is_refused(connectome):
    with pytest.raises(ValueError, match="no cortical vertices"):
        connectome.seed(region=np.array([False, False, False, False, True]))


def test_seed_region_counts_cortex_only_on_the_real_medial_wall():
    """PALS_B12_Lobes labels the wall too; a lobe's profile is over its cortex, as in to_atlas."""
    import sbci

    cc = sbci.example()
    atlas = sbci.load_atlas("PALS_B12_Lobes")
    # the lobe with the most medial-wall vertices
    on_wall = np.bincount(np.asarray(atlas.labels)[~cc.mask], minlength=atlas.n_regions + 1)[1:]
    lobe = atlas.region_mask(int(np.argmax(on_wall)) + 1)
    assert (lobe & ~cc.mask).sum() > 10 and (lobe & cc.mask).sum() > 10
    np.testing.assert_allclose(cc.seed(region=lobe), cc.seed(region=lobe & cc.mask))


def test_seed_requires_exactly_one_argument(connectome):
    with pytest.raises(ValueError, match="exactly one"):
        connectome.seed()
    with pytest.raises(ValueError, match="exactly one"):
        connectome.seed(vertex=0, region=np.ones(5, dtype=bool))


def test_seed_rejects_an_out_of_range_vertex(connectome):
    with pytest.raises(IndexError):
        connectome.seed(vertex=99)


def test_seed_rejects_an_empty_region(connectome):
    with pytest.raises(ValueError, match="no vertices"):
        connectome.seed(region=np.zeros(5, dtype=bool))


def test_to_atlas_shape(connectome, atlas):
    assert connectome.to_atlas(atlas).shape == (3, 3)


def test_shape_mismatch_is_caught_on_load(tmp_path, connectome):
    import h5py

    path = connectome.save(tmp_path / "sub-short_sc.h5")
    with h5py.File(path, "a") as handle:
        del handle["area"]
        handle.create_dataset("area", data=np.asarray(connectome.area[:4], dtype=np.float64))
    with pytest.raises(ValueError, match="implies 5 vertices"):
        ContinuousConnectome.load(path)


def test_a_region_mask_of_the_wrong_length_is_refused(connectome):
    with pytest.raises(ValueError, match="region mask has 3 entries"):
        connectome.seed(region=np.array([True, False, True]))


@pytest.mark.parametrize("kernel", KERNELS)
def test_no_kernel_is_a_placeholder(connectome, kernel):
    """Every name in KERNELS has to be implemented, not a stub.

    ``shk`` raised NotImplementedError for as long as the kernel concon
    actually applies was unidentified. Now that all three are ported, the only
    thing smoothing this connectome can complain about is that it carries no
    endpoints -- so a NotImplementedError here means a regression, or a fourth
    kernel added to the vocabulary before it was written.
    """
    from sbci.errors import MissingDataError

    with pytest.raises(MissingDataError, match="endpoints"):
        connectome.smooth(kernel=kernel)


def _functional_toy(connectome):
    """The toy connectome's layout with correlations in it."""
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
    metadata = template(
        "fc",
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="0.0.1.dev0",
        container_version="sbci.sif@sha256:0",
        fc_nuisance_model="36p",
    )
    return ContinuousConnectome(
        data=to_condensed(correlations).astype(np.float32),
        area=connectome.area,
        mask=connectome.mask,
        metadata=metadata,
    ), correlations


def test_functional_connectomes_are_aggregated_as_fisher_z_means(connectome, atlas):
    """The default for FC is a Fisher-z average; a mass of correlations is refused."""
    from sbci.parcellation import parcellate

    fc, correlations = _functional_toy(connectome)
    matrix = fc.to_atlas(atlas)
    area = np.where(fc.mask, fc.area, 0.0)
    expected = parcellate(correlations, atlas, area, how="mean", fisher_z=True)
    np.testing.assert_allclose(matrix, expected)
    assert np.nanmax(np.abs(matrix)) < 1.0  # not saturated
    # C is the medial-wall vertex alone: no cortical pair with anyone, so NaN throughout
    assert np.isnan(matrix[2, :]).all() and np.isnan(matrix[:, 2]).all()
    assert np.isfinite(matrix[:2, :2]).all()
    with pytest.raises(ValueError, match="no mass"):
        fc.to_atlas(atlas, how="mass")


def test_to_atlas_gives_an_empty_region_nan_under_mean_and_zero_under_mass(connectome):
    """A region with no vertex at this resolution keeps its row; what it holds depends on how."""
    from sbci.atlas import Atlas

    gappy = Atlas(name="gappy", labels=np.array([1, 1, 2, 2, 0]), names=("A", "B", "C"))
    mass = connectome.to_atlas(gappy)
    mean = connectome.to_atlas(gappy, how="mean")
    assert (mass[2, :] == 0.0).all() and (mass[:, 2] == 0.0).all()
    assert np.isnan(mean[2, :]).all() and np.isnan(mean[:, 2]).all()
    assert np.isfinite(mean[:2, :2]).all() and np.isfinite(mass).all()


def test_to_atlas_leaves_masked_vertices_out_of_the_region_areas(connectome):
    """Vertex 4 is medial wall; put it in a cortical region and it must not dilute that mean."""
    from sbci.atlas import Atlas
    from sbci.parcellation import parcellate

    atlas = Atlas(name="toy2", labels=np.array([1, 1, 2, 2, 2]), names=("A", "B"))
    with_mask = connectome.to_atlas(atlas, how="mean")
    dense = connectome.dense()
    without = parcellate(dense, atlas, connectome.area, how="mean")
    masked_area = np.where(connectome.mask, connectome.area, 0.0)
    expected = parcellate(dense, atlas, masked_area, how="mean")
    np.testing.assert_allclose(with_mask, expected)
    assert not np.allclose(with_mask, without)


def test_seed_region_matches_the_dense_computation(connectome, atlas):
    member = atlas.region_mask("A")
    weights = np.where(member, connectome.area, 0.0)
    expected = (connectome.dense().astype(np.float64) @ weights) / weights.sum()
    np.testing.assert_allclose(connectome.seed(region=member), expected)


def test_exchange_files_are_refused_by_load_with_an_explanation(tmp_path):
    from sbci.connectome import ContinuousConnectome
    from sbci.errors import InvalidFileError

    path = tmp_path / "sub-x.dconn.nii"
    path.write_bytes(b"")
    with pytest.raises(InvalidFileError, match="write"):
        ContinuousConnectome.load(path)


def test_load_names_a_missing_file_plainly(tmp_path):
    """Not h5py's 'unable to synchronously open file' with its flags."""
    with pytest.raises(FileNotFoundError, match="no such file") as caught:
        ContinuousConnectome.load(tmp_path / "nope.h5")
    assert "synchronously" not in str(caught.value)


def test_save_names_a_missing_directory_plainly(tmp_path, connectome):
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        connectome.save(tmp_path / "nowhere" / "sub-x_sc.h5")


def test_save_refuses_a_name_load_would_not_read(tmp_path, connectome):
    """HDF5 under any suffix wrote fine and then would not load; refuse before writing."""
    target = tmp_path / "sub-x_sc.mat"
    with pytest.raises(ValueError, match=r"load\(\) would read back"):
        connectome.save(target)
    assert not list(tmp_path.iterdir())
    for name in ("sub-x_sc.h5", "sub-x_sc.hdf5", "sub-01.ses-1_sc.h5"):
        assert ContinuousConnectome.load(connectome.save(tmp_path / name)).n_vertices == 5


def test_a_computational_file_loads_whatever_the_case_of_its_suffix(tmp_path):
    """load_cohort finds sub-01_sc.H5 as a computational file; load and save take it too."""
    import sbci

    for name in ("sub-01_sc.H5", "sub-02_sc.HDF5"):
        written = sbci.example().save(tmp_path / name)
        assert sbci.load(written).n_vertices == 5124
    found = sbci.load_cohort(tmp_path)
    assert sorted(found.subjects) == ["sub-01", "sub-02"]
