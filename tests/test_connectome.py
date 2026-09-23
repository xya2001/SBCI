"""The public object: loading, seeding, aggregating, and the pending ports."""

from __future__ import annotations

import numpy as np
import pytest

from sbci import ContinuousConnectome
from sbci.io import write_hdf5
from sbci.metadata import MetadataError, template


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


def test_seed_region_is_the_area_weighted_marginal(connectome):
    member = np.array([True, True, False, False, False])
    expected = connectome.dense() @ np.where(member, connectome.area, 0.0)
    np.testing.assert_allclose(connectome.seed(region=member), expected / 3.0, rtol=1e-6)


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
    truncated = ContinuousConnectome(
        data=connectome.data,
        area=connectome.area[:4],
        mask=connectome.mask,
        metadata=connectome.metadata,
    )
    with pytest.raises(ValueError, match="implies 5 vertices"):
        truncated._check_shapes()


@pytest.mark.parametrize("kernel", ["shk", "rdk", "matern"])
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


def test_coupling_validates_before_failing(connectome):
    """Argument checks run first, so a misuse is reported as a misuse."""
    with pytest.raises(ValueError, match="structural connectome"):
        connectome.coupling(connectome)


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
    assert np.abs(matrix).max() < 1.0  # not saturated
    with pytest.raises(ValueError, match="no mass"):
        fc.to_atlas(atlas, how="mass")


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
