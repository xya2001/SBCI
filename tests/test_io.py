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
def paired(sc_metadata):
    """A twelve-vertex SC connectome, six vertices per hemisphere, for the endpoints below.

    The file stores whole-grid indices and the reader splits them at half the
    file's own grid, so endpoints need a connectome with twice their vertices
    per hemisphere; the five-vertex fixture cannot be halved.
    """
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed

    rng = np.random.default_rng(1)
    matrix = rng.random((12, 12))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0.0)
    area = np.full(12, 2.0)
    matrix /= area @ matrix @ area
    return ContinuousConnectome(
        data=to_condensed(matrix).astype(np.float32),
        area=area,
        mask=np.ones(12, dtype=bool),
        metadata=sc_metadata,
    )


@pytest.fixture
def endpoints():
    """Four streamlines with continuous positions, on an octahedron a hemisphere.

    Six vertices and eight triangles a hemisphere: the file splits triangle
    indices at ``2 V - 4``, the faces of a closed spherical mesh, as it splits
    vertex indices at half the file's grid.
    """
    from sbci.smoothing import Endpoints

    return Endpoints(
        surf_in=np.array([0, 0, 1, 1], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 0], dtype=np.int8),
        vtx_in=np.array([0, 2, 1, 2]),
        vtx_out=np.array([1, 0, 2, 1]),
        n_per_hemi=6,
        tri_in=np.array([0, 5, 7, 2]),
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


def test_endpoints_round_trip(tmp_path, paired, endpoints):
    back = read_hdf5(_write(tmp_path, paired, endpoints=endpoints))["endpoints"]

    np.testing.assert_array_equal(back.global_vertex_in, endpoints.global_vertex_in)
    np.testing.assert_array_equal(back.global_vertex_out, endpoints.global_vertex_out)
    np.testing.assert_array_equal(back.surf_in, endpoints.surf_in)
    np.testing.assert_array_equal(back.surf_out, endpoints.surf_out)
    assert back.n_streamlines == endpoints.n_streamlines


def test_continuous_positions_round_trip(tmp_path, paired, endpoints):
    """Triangles and barycentric weights survive, so off-grid kernels stay possible."""
    back = read_hdf5(_write(tmp_path, paired, endpoints=endpoints))["endpoints"]

    assert back.has_positions
    np.testing.assert_allclose(back.bary_in, endpoints.bary_in, rtol=1e-6)
    np.testing.assert_allclose(back.bary_out, endpoints.bary_out, rtol=1e-6)


def test_vertices_alone_are_enough(tmp_path, paired, endpoints):
    """Positions are optional within the group; vertices are not."""
    from sbci.smoothing import Endpoints

    bare = Endpoints(
        surf_in=endpoints.surf_in,
        surf_out=endpoints.surf_out,
        vtx_in=endpoints.vtx_in,
        vtx_out=endpoints.vtx_out,
        n_per_hemi=endpoints.n_per_hemi,
    )
    back = read_hdf5(_write(tmp_path, paired, endpoints=bare))["endpoints"]
    assert not back.has_positions
    np.testing.assert_array_equal(back.global_vertex_in, bare.global_vertex_in)


def test_half_a_position_is_refused(tmp_path, paired, endpoints):
    """Barycentric weights without their triangles cannot be interpreted."""
    import h5py

    path = _write(tmp_path, paired, endpoints=endpoints)
    with h5py.File(path, "a") as handle:
        del handle["endpoints"]["triangle_in"]
    with pytest.raises(ValueError, match="positions need all of"):
        read_hdf5(path)


def test_a_truncated_endpoint_dataset_is_refused(tmp_path, paired, endpoints):
    import h5py

    path = _write(tmp_path, paired, endpoints=endpoints)
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


def test_the_hemisphere_comes_from_the_index(tmp_path, paired, endpoints):
    """Stored indices are global, so hemisphere and vertex cannot disagree."""
    import h5py

    path = _write(tmp_path, paired, endpoints=endpoints)
    with h5py.File(path, "r") as handle:
        stored = np.asarray(handle["endpoints"]["vertex_in"][()])
    np.testing.assert_array_equal(stored, endpoints.global_vertex_in)
    assert stored.max() >= endpoints.n_per_hemi  # the right hemisphere is offset


def test_endpoints_are_split_at_half_the_files_own_grid(tmp_path, paired, endpoints):
    """Not at the ico4 grid's 2562 and 5120: six vertices a hemisphere come back as six.

    Split at 2562, every endpoint of the twelve-vertex file read as the left
    hemisphere's; split at 5120 faces, every triangle did, and a right-hemisphere
    triangle was out of range on reading.
    """
    import h5py

    path = _write(tmp_path, paired, endpoints=endpoints)
    back = read_hdf5(path)["endpoints"]
    assert back.n_per_hemi == 6
    np.testing.assert_array_equal(back.surf_in, endpoints.surf_in)
    np.testing.assert_array_equal(back.vtx_out, endpoints.vtx_out)
    np.testing.assert_array_equal(back.tri_in, endpoints.tri_in)
    np.testing.assert_array_equal(back.tri_out, endpoints.tri_out)
    with h5py.File(path, "r") as handle:
        stored = np.asarray(handle["endpoints"]["triangle_in"][()])
    np.testing.assert_array_equal(stored, endpoints.tri_in + 8 * endpoints.surf_in)


def test_endpoints_on_a_grid_that_cannot_be_halved_are_refused(tmp_path, connectome, paired):
    """An odd vertex count is refused plainly; an area of the wrong length is named as such.

    The endpoints are split at the connectivity's own count, so an area that
    disagrees is reported as the area's fault by ``load`` and ``sbci validate``,
    not as one of the endpoints.
    """
    import h5py

    from sbci.connectome import ContinuousConnectome
    from sbci.errors import InvalidFileError
    from sbci.validate import validate_file

    odd = connectome.save(tmp_path / "sub-odd_sc.h5")  # five vertices
    with h5py.File(odd, "a") as handle:
        group = handle.create_group("endpoints")
        group.create_dataset("vertex_in", data=np.array([0, 1], dtype=np.int32))
        group.create_dataset("vertex_out", data=np.array([2, 3], dtype=np.int32))
    with pytest.raises(InvalidFileError, match="the grid has 5 vertices, an odd number"):
        read_hdf5(odd)

    disagreeing = paired.save(tmp_path / "sub-pair_sc.h5")
    with h5py.File(disagreeing, "a") as handle:
        group = handle.create_group("endpoints")
        group.create_dataset("vertex_in", data=np.array([0, 7], dtype=np.int32))
        group.create_dataset("vertex_out", data=np.array([1, 11], dtype=np.int32))
        del handle["area"]
        handle.create_dataset("area", data=np.full(8, 2.0))
    assert read_hdf5(disagreeing)["endpoints"].n_per_hemi == 6
    failed = [str(check) for check in validate_file(disagreeing) if not check.passed]
    assert any("area has 8" in line for line in failed)
    assert not any("endpoints" in line for line in failed)
    with pytest.raises(InvalidFileError, match="area has 8"):
        ContinuousConnectome.load(disagreeing)


def test_endpoints_beyond_the_files_grid_are_refused(tmp_path, paired, endpoints):
    import h5py

    from sbci.errors import InvalidFileError

    path = _write(tmp_path, paired, endpoints=endpoints)
    with h5py.File(path, "a") as handle:
        handle["endpoints"]["vertex_in"][0] = 12
    with pytest.raises(InvalidFileError, match="out of range for a 12-vertex grid"):
        read_hdf5(path)


def test_endpoints_on_another_grid_are_not_written(tmp_path, connectome, paired, endpoints):
    """The writer refuses what the reader would split elsewhere, or refuse."""
    from sbci.smoothing import Endpoints

    with pytest.raises(ValueError, match="grid of 12 vertices but the connectome on one of 5"):
        _write(tmp_path, connectome, endpoints=endpoints)
    ico4 = Endpoints(
        surf_in=endpoints.surf_in,
        surf_out=endpoints.surf_out,
        vtx_in=endpoints.vtx_in,
        vtx_out=endpoints.vtx_out,
    )
    with pytest.raises(ValueError, match="grid of 5124 vertices but the connectome on one of 12"):
        _write(tmp_path, paired, endpoints=ico4)
    assert not list(tmp_path.iterdir())


def test_a_failed_write_leaves_the_previous_file_intact_and_no_partial(tmp_path, connectome):
    """The file is written beside the destination and moved into place only when complete."""
    path = tmp_path / "sub-x_sc.h5"
    connectome.save(path)
    before = path.read_bytes()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        write_hdf5(
            path,
            connectome.data,
            connectome.area,
            connectome.mask,
            connectome.metadata,
            endpoints="not endpoints",
        )
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_a_file_that_is_not_hdf5_is_named_as_such(tmp_path):
    from sbci.errors import InvalidFileError

    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not an hdf5 file")
    with pytest.raises(InvalidFileError, match="cannot be opened as an HDF5 file"):
        read_hdf5(junk)


def test_a_metadata_dataset_that_is_not_a_string_is_refused_plainly(tmp_path, connectome):
    """Not json's 'the JSON object must be str, bytes or bytearray, not ndarray'."""
    import h5py

    from sbci.errors import InvalidFileError

    path = connectome.save(tmp_path / "sub-z_sc.h5")
    with h5py.File(path, "a") as handle:
        del handle["metadata"]
        handle.create_dataset("metadata", data=np.arange(3))
    with pytest.raises(InvalidFileError, match="/metadata is not a JSON string"):
        read_hdf5(path)


def test_an_incomplete_endpoint_group_names_what_is_missing(tmp_path, connectome):
    import h5py

    from sbci.errors import FormatError

    path = connectome.save(tmp_path / "sub-y_sc.h5")
    with h5py.File(path, "a") as handle:
        group = handle.create_group("endpoints")
        group.create_dataset("vertex_in", data=np.zeros(3, dtype=np.int32))
    with pytest.raises(FormatError, match="has no vertex_out"):
        read_hdf5(path)


# --- the eighth review ------------------------------------------------------------


def test_the_header_check_refuses_endpoint_positions_of_the_wrong_width(tmp_path):
    """Two coordinates a position passed the header check, which load then refused."""
    import h5py

    import sbci
    from sbci.errors import InvalidFileError
    from sbci.io import read_header

    path = tmp_path / "sub-01_sc.h5"
    sbci.example(modality="sc").save(path)
    with h5py.File(path, "a") as handle:
        group = handle["endpoints"]
        narrow = group["barycentric_in"][()][:, :2]
        del group["barycentric_in"]
        group["barycentric_in"] = narrow
    with pytest.raises(
        InvalidFileError, match="barycentric_in is \\(20000, 2\\), expected \\(20000, 3\\)"
    ):
        read_header(path)


# --- the review of 11 October 2026 -------------------------------------------------


def test_endpoints_on_a_functional_file_are_refused(tmp_path):
    """write_hdf5 refuses them, but read_hdf5 took them, and smooth() made an 'fc' density."""
    import h5py

    import sbci
    from sbci.errors import InvalidFileError
    from sbci.io.hdf5 import ENDPOINTS

    sc = sbci.example().save(tmp_path / "sub-01_sc.h5")
    fc = sbci.example("fc").save(tmp_path / "sub-01_fc.h5")
    with h5py.File(sc, "r") as source, h5py.File(fc, "r+") as target:
        source.copy(ENDPOINTS, target)
    with pytest.raises(InvalidFileError, match="belongs to a structural connectome"):
        sbci.load(fc)


def test_nan_in_the_metadata_reads_as_missing(tmp_path):
    """A NaN bandwidth loaded and validated, and the package could not write the file back."""
    import json

    import h5py

    import sbci

    path = sbci.example().save(tmp_path / "sub-01_sc.h5")
    with h5py.File(path, "r+") as handle:
        fields = json.loads(handle["metadata"][()])
        fields["bandwidth"] = float("nan")
        del handle["metadata"]
        handle.create_dataset("metadata", data=json.dumps(fields))  # NaN, as json writes it
    with pytest.raises(MetadataError, match="bandwidth"):
        sbci.load(path)


def test_saving_through_a_link_writes_its_target(tmp_path):
    """The new file replaced the symbolic link, and the file it pointed to stayed stale."""
    import sbci

    target = sbci.example().save(tmp_path / "stored_sc.h5")
    link = tmp_path / "sub-01_sc.h5"
    link.symlink_to(target)
    other = sbci.example(seed=1)
    other.save(link)
    assert link.is_symlink()
    np.testing.assert_array_equal(sbci.load(target).data, other.data)


# --- the fourth review of 12 October 2026 ------------------------------------------------


def test_save_refuses_a_read_only_file_as_every_writer_does(tmp_path, connectome, capsys):
    """os.replace replaced it, and reset its mode to 0644; export's writers refused it."""
    import os
    import stat

    from sbci.cli import main

    path = connectome.save(tmp_path / "sub-x_sc.h5")
    before = path.read_bytes()
    path.chmod(stat.S_IRUSR)
    with pytest.raises(PermissionError, match="is read-only"):
        connectome.save(path)
    assert path.read_bytes() == before and stat.S_IMODE(os.stat(path).st_mode) == stat.S_IRUSR
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sub-x_sc.h5"]
    assert main(["example", "--modality", "fc", "--out", str(path)]) == 1
    assert "is read-only" in capsys.readouterr().err and path.read_bytes() == before


@pytest.fixture(scope="module")
def example_sc(tmp_path_factory):
    """The structural example on disk, endpoints and coordinates included."""
    import sbci

    return sbci.example("sc").save(tmp_path_factory.mktemp("example") / "sub-01_sc.h5")


def _restored(example_sc, tmp_path, name, values):
    """A copy of the example with the dataset ``name`` stored as ``values``."""
    import shutil

    import h5py

    path = tmp_path / "sub-02_sc.h5"
    shutil.copy(example_sc, path)
    with h5py.File(path, "r+") as handle:
        del handle[name]
        handle.create_dataset(name, data=values, compression="gzip")
    return path


def test_the_header_says_what_load_would_refuse_read_from_real_files(example_sc, tmp_path):
    """read_header was tested through a stub alone: numeric or flat always True passed the suite.

    Every array counts, the coordinates and the endpoint datasets with the
    connectivity, area and mask; and what the header calls no numbers, load refuses.
    """
    import h5py

    from sbci.errors import InvalidFileError
    from sbci.io import read_header

    header = read_header(example_sc)
    assert header["numeric"] and header["flat"] and header["n_endpoints"] == 20_000
    with h5py.File(example_sc, "r") as handle:
        n_upper, bary = handle["connectivity"].shape[0], handle["endpoints/barycentric_in"][()]
    not_numbers = {
        "connectivity": np.full(n_upper, b"0"),  # numeric text: NumPy reads it as 0.0
        "coordinates": np.full((5124, 3), b"abc"),
        "mask": np.full(5124, b"1"),
        "endpoints/barycentric_in": bary.astype(np.complex64),
        "endpoints/vertex_out": np.zeros(20_000, dtype=[("v", "i4"), ("w", "i4")]),
    }
    for name, values in not_numbers.items():
        path = _restored(example_sc, tmp_path, name, values)
        header = read_header(path)
        assert not header["numeric"] and header["flat"], name
        with pytest.raises(InvalidFileError, match="not numbers|not barycentric|not indices"):
            read_hdf5(path)
    for name, shape in (("connectivity", (n_upper, 1)), ("area", (2, 2562)), ("mask", (1, 5124))):
        with h5py.File(example_sc, "r") as handle:
            values = handle[name][()].reshape(shape)
        header = read_header(_restored(example_sc, tmp_path, name, values))
        assert header["numeric"] and not header["flat"], name


def test_numbers_of_the_wrong_kind_are_refused_before_the_cast(example_sc, tmp_path):
    """Complex connectivity loaded with a ComplexWarning, its imaginary part dropped."""
    import warnings

    import h5py

    import sbci
    from sbci.errors import InvalidFileError

    with h5py.File(example_sc, "r") as handle:
        data = handle["connectivity"][()]
    path = _restored(example_sc, tmp_path, "connectivity", data.astype(np.complex64) + 1j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InvalidFileError, match="/connectivity holds complex64 values"):
            sbci.load(path)


@pytest.mark.parametrize("weight", [np.nan, np.inf])
def test_endpoint_weights_that_are_not_finite_are_refused_on_reading(
    example_sc, tmp_path, weight, capsys
):
    """A NaN weight loaded and validated, and smooth() failed with a plain ValueError."""
    import h5py

    import sbci
    from sbci.cli import main
    from sbci.errors import InvalidFileError
    from sbci.validate import validate_file

    with h5py.File(example_sc, "r") as handle:
        bary = handle["endpoints/barycentric_out"][()]
    bary[7, 1] = weight
    path = _restored(example_sc, tmp_path, "endpoints/barycentric_out", bary)
    with pytest.raises(InvalidFileError, match=r"barycentric_out holds 1 weight\(s\) that are not"):
        sbci.load(path)
    assert not validate_file(path)[0].passed
    assert main(["info", str(path)]) == 1 and "not finite" in capsys.readouterr().err


def test_endpoints_built_in_memory_are_held_to_finite_weights():
    from sbci.smoothing import Endpoints

    flags, vertices = np.array([0, 1]), np.array([0, 1])
    good = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    nan = np.array([[1.0, 0.0, 0.0], [0.2, np.nan, 0.8]])
    with pytest.raises(ValueError, match=r"bary_out holds 1 weight\(s\) that are not finite"):
        Endpoints(flags, flags, vertices, vertices, 6, vertices, vertices, good, nan)
    with pytest.raises(ValueError, match="bary_in holds <U3 values, not barycentric weights"):
        Endpoints(flags, flags, vertices, vertices, 6, vertices, vertices, [["0.5"] * 3] * 2, good)
    # no streamlines is still a set of endpoints in memory: take([]) gives one
    assert Endpoints(flags, flags, vertices, vertices, 6).take([]).n_streamlines == 0


def test_an_empty_endpoint_group_is_refused_and_never_written(tmp_path, paired, endpoints):
    """Nothing to re-smooth: the reader took the group, and smooth() raised a bare ValueError."""
    import h5py

    from sbci.errors import MissingDataError
    from sbci.validate import validate_file

    path = _write(tmp_path, paired, endpoints=endpoints)
    with h5py.File(path, "a") as handle:
        group = handle["endpoints"]
        for name in list(group):
            shape, dtype = group[name].shape, group[name].dtype
            del group[name]
            group.create_dataset(name, data=np.zeros((0,) + shape[1:], dtype=dtype))
    with pytest.raises(MissingDataError, match="/endpoints is present but holds no streamlines"):
        read_hdf5(path)
    checks = validate_file(path)
    assert not checks[0].passed and "no streamlines" in checks[0].detail
    before = path.read_bytes()
    with pytest.raises(ValueError, match="the endpoints hold no streamlines"):
        _write(tmp_path, paired, endpoints=endpoints.take([]))
    assert path.read_bytes() == before and not list(tmp_path.glob("*.partial"))
