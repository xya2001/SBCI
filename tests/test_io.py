"""HDF5 round trips and the writer's refusal to emit an invalid file."""

from __future__ import annotations

import numpy as np
import pytest

from sbci.io import read_hdf5, write_hdf5
from sbci.metadata import MetadataError, template


def test_round_trip(tmp_path, connectome):
    path = write_hdf5(
        tmp_path / "sub-toy_sc.h5",
        data=connectome.data,
        area=connectome.area,
        mask=connectome.mask,
        metadata=connectome.metadata,
    )
    parts = read_hdf5(path)

    np.testing.assert_allclose(parts["data"], connectome.data)
    np.testing.assert_allclose(parts["area"], connectome.area)
    np.testing.assert_array_equal(parts["mask"], connectome.mask)
    assert parts["metadata"].fields == connectome.metadata.fields
    assert parts["coords"] is None


def test_coordinates_are_optional_but_preserved(tmp_path, connectome):
    coords = np.arange(15, dtype=np.float32).reshape(5, 3)
    path = write_hdf5(
        tmp_path / "sub-toy_sc.h5",
        data=connectome.data,
        area=connectome.area,
        mask=connectome.mask,
        metadata=connectome.metadata,
        coords=coords,
    )
    np.testing.assert_allclose(read_hdf5(path)["coords"], coords)


def test_writer_refuses_incomplete_metadata(tmp_path, connectome):
    target = tmp_path / "sub-toy_sc.h5"
    with pytest.raises(MetadataError):
        write_hdf5(
            target,
            data=connectome.data,
            area=connectome.area,
            mask=connectome.mask,
            metadata=template("sc"),
        )
    assert not target.exists(), "a rejected write must not leave a partial file"


def test_reader_reports_a_missing_dataset(tmp_path, connectome):
    import h5py

    path = tmp_path / "truncated.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("connectivity", data=connectome.data)

    with pytest.raises(KeyError, match="/area"):
        read_hdf5(path)


# --- the optional endpoint group -----------------------------------------


@pytest.fixture
def endpoints():
    """Four streamlines with continuous positions.

    Unlike the connectome fixtures these use the real grid's 2562 vertices per
    hemisphere, because the file format writes global indices and the reader
    splits them at the grid boundary -- a toy hemisphere size would not survive
    the round trip, and should not.
    """
    from sbci import spec
    from sbci.smoothing import Endpoints

    return Endpoints(
        surf_in=np.array([0, 0, 1, 1], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 0], dtype=np.int8),
        vtx_in=np.array([0, 2, 1, 2]),
        vtx_out=np.array([1, 0, 2, 1]),
        n_per_hemi=spec.N_VERTICES_PER_HEMI,
        tri_in=np.array([0, 5, 9, 2]),
        tri_out=np.array([1, 3, 7, 4]),
        bary_in=np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]]),
        bary_out=np.array([[0.0, 1.0, 0.0], [0.5, 0.25, 0.25], [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]]),
    )


def _write(tmp_path, connectome, **extra):
    return write_hdf5(
        tmp_path / "sub-toy_sc.h5",
        data=connectome.data,
        area=connectome.area,
        mask=connectome.mask,
        metadata=connectome.metadata,
        **extra,
    )


def test_a_file_without_endpoints_reads_back_as_none(tmp_path, connectome):
    """The group is optional: its absence is not an error."""
    assert read_hdf5(_write(tmp_path, connectome))["endpoints"] is None


def test_endpoints_round_trip(tmp_path, connectome, endpoints):
    back = read_hdf5(_write(tmp_path, connectome, endpoints=endpoints))["endpoints"]

    np.testing.assert_array_equal(back.global_vertex_in, endpoints.global_vertex_in)
    np.testing.assert_array_equal(back.global_vertex_out, endpoints.global_vertex_out)
    np.testing.assert_array_equal(back.surf_in, endpoints.surf_in)
    np.testing.assert_array_equal(back.surf_out, endpoints.surf_out)
    assert back.n_streamlines == endpoints.n_streamlines


def test_continuous_positions_round_trip(tmp_path, connectome, endpoints):
    """Triangles and barycentric weights survive, so off-grid kernels stay possible."""
    back = read_hdf5(_write(tmp_path, connectome, endpoints=endpoints))["endpoints"]

    assert back.has_positions
    np.testing.assert_allclose(back.bary_in, endpoints.bary_in, rtol=1e-6)
    np.testing.assert_allclose(back.bary_out, endpoints.bary_out, rtol=1e-6)


def test_vertices_alone_are_enough(tmp_path, connectome, endpoints):
    """Positions are optional within the group; vertices are not."""
    from sbci.smoothing import Endpoints

    bare = Endpoints(
        surf_in=endpoints.surf_in,
        surf_out=endpoints.surf_out,
        vtx_in=endpoints.vtx_in,
        vtx_out=endpoints.vtx_out,
        n_per_hemi=endpoints.n_per_hemi,
    )
    back = read_hdf5(_write(tmp_path, connectome, endpoints=bare))["endpoints"]
    assert not back.has_positions
    np.testing.assert_array_equal(back.global_vertex_in, bare.global_vertex_in)


def test_half_a_position_is_refused(tmp_path, connectome, endpoints):
    """Barycentric weights without their triangles cannot be interpreted."""
    import h5py

    path = _write(tmp_path, connectome, endpoints=endpoints)
    with h5py.File(path, "a") as handle:
        del handle["endpoints"]["triangle_in"]
    with pytest.raises(ValueError, match="positions need all of"):
        read_hdf5(path)


def test_a_truncated_endpoint_dataset_is_refused(tmp_path, connectome, endpoints):
    import h5py

    path = _write(tmp_path, connectome, endpoints=endpoints)
    with h5py.File(path, "a") as handle:
        group = handle["endpoints"]
        kept = group["vertex_out"][:2]
        del group["vertex_out"]
        group.create_dataset("vertex_out", data=kept)
    with pytest.raises(ValueError, match="disagree on the streamline count"):
        read_hdf5(path)


def test_functional_connectomes_have_no_streamlines(tmp_path, connectome, endpoints):
    """FC carries correlations, so endpoints on one would be meaningless."""
    fc_metadata = template(
        "fc",
        normalization="unit-mass",
        registration_reference="fsaverage",
        pipeline_version="0.0.1.dev0",
        container_version="sbci.sif@sha256:0",
        fc_nuisance_model="36p",
    )
    with pytest.raises(ValueError, match="structural connectome"):
        write_hdf5(
            tmp_path / "sub-toy_fc.h5",
            data=connectome.data,
            area=connectome.area,
            mask=connectome.mask,
            metadata=fc_metadata,
            endpoints=endpoints,
        )


def test_the_hemisphere_comes_from_the_index(tmp_path, connectome, endpoints):
    """Stored indices are global, so hemisphere and vertex cannot disagree."""
    import h5py

    path = _write(tmp_path, connectome, endpoints=endpoints)
    with h5py.File(path, "r") as handle:
        stored = np.asarray(handle["endpoints"]["vertex_in"][()])
    np.testing.assert_array_equal(stored, endpoints.global_vertex_in)
    assert stored.max() >= endpoints.n_per_hemi  # the right hemisphere is offset
