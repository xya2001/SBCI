"""The validator and the `sbci validate` command."""

from __future__ import annotations

import numpy as np
import pytest

from sbci import ContinuousConnectome
from sbci.cli import main
from sbci.errors import SbciError
from sbci.io import write_hdf5
from sbci.validate import validate_file


def _write(tmp_path, connectome, **overrides):
    parts = {
        "data": connectome.data,
        "area": connectome.area,
        "mask": connectome.mask,
        "metadata": connectome.metadata,
    }
    parts.update(overrides)
    return write_hdf5(tmp_path / "sub-toy_sc.h5", compression=None, **parts)


def _results(checks):
    return {check.name: check.passed for check in checks}


def test_a_well_formed_file_passes_every_grid_independent_check(tmp_path, connectome):
    """The toy grid is not ico4, so `grid` is expected to fail and nothing else."""
    results = _results(validate_file(_write(tmp_path, connectome)))
    assert results["readable"]
    assert results["name"]
    assert results["metadata"]
    assert results["shapes"]
    assert results["symmetry"]
    assert results["nonnegativity"]
    assert results["unit mass"]
    assert results["mask"]
    assert not results["grid"], "toy grid must not be mistaken for ico4"


def _rewrite(path, name, data):
    import h5py

    with h5py.File(path, "a") as handle:
        del handle[name]
        handle.create_dataset(name, data=data)
    return path


def _misnamed(tmp_path, connectome):
    return _write(tmp_path, connectome).rename(tmp_path / "sub-toy_sc.mat")


def _flat_coordinates(tmp_path, connectome):
    return _write(tmp_path, connectome, coords=np.zeros((5, 2), dtype=np.float32))


def _short_mask(tmp_path, connectome):
    return _rewrite(_write(tmp_path, connectome), "mask", connectome.mask[:4])


def _metadata_not_a_string(tmp_path, connectome):
    return _rewrite(_write(tmp_path, connectome), "metadata", np.arange(3))


def _old_spec_version(tmp_path, connectome):
    metadata = type(connectome.metadata)(dict(connectome.metadata.fields, spec_version="9.0.0"))
    return _rewrite(_write(tmp_path, connectome), "metadata", metadata.to_json())


@pytest.mark.parametrize(
    ("make", "failing"),
    [
        (_misnamed, "name"),
        (_flat_coordinates, "shapes"),
        (_short_mask, "shapes"),
        (_metadata_not_a_string, "readable"),
        (_old_spec_version, "metadata"),
    ],
)
def test_what_load_refuses_fails_a_named_check(tmp_path, connectome, make, failing):
    """`sbci validate` must not pass a file that `ContinuousConnectome.load` then rejects.

    Each file here is refused by ``load`` and, apart from the toy grid's
    expected ``grid`` failure, fails exactly the check named; the well-formed
    toy file fails ``grid`` alone.
    """
    path = make(tmp_path, connectome)
    with pytest.raises(SbciError):
        ContinuousConnectome.load(path)
    failed = {check.name for check in validate_file(path) if not check.passed} - {"grid"}
    assert failing in failed, failed
    if failing != "readable":
        assert "readable" not in failed


def test_unit_mass_violation_is_caught(tmp_path, connectome):
    results = _results(validate_file(_write(tmp_path, connectome, data=connectome.data * 2)))
    assert not results["unit mass"]


def test_negative_structural_connectivity_is_caught(tmp_path, connectome):
    data = connectome.data.copy()
    data[0] = -1.0
    assert not _results(validate_file(_write(tmp_path, connectome, data=data)))["nonnegativity"]


def test_connectivity_on_the_medial_wall_is_caught(tmp_path, connectome):
    """A file that puts mass on masked vertices fails, whatever its metadata says."""
    mask = np.array([True, True, True, False, False])
    assert not _results(validate_file(_write(tmp_path, connectome, mask=mask)))["mask"]


def test_an_all_true_mask_is_caught(tmp_path, connectome):
    mask = np.ones(5, dtype=bool)
    assert not _results(validate_file(_write(tmp_path, connectome, mask=mask)))["mask"]


def test_an_unreadable_file_reports_one_failure(tmp_path):
    path = tmp_path / "not-hdf5.h5"
    path.write_text("this is not an HDF5 file")
    checks = validate_file(path)
    assert len(checks) == 1
    assert not checks[0].passed


def test_cli_exits_non_zero_on_failure(tmp_path, connectome, capsys):
    """The toy file fails the grid check, so the command must report failure."""
    assert main(["validate", str(_write(tmp_path, connectome))]) == 1
    assert "[FAIL] grid" in capsys.readouterr().out


def test_a_missing_file_is_reported_in_plain_words(tmp_path, capsys):
    checks = validate_file(tmp_path / "nope.h5")
    assert [c.passed for c in checks] == [False]
    assert "no such file" in str(checks[0])
    assert main(["validate", str(tmp_path / "nope.h5")]) == 1
    assert "no such file" in capsys.readouterr().out


def test_non_finite_values_fail(tmp_path, connectome):
    import h5py

    path = _write(tmp_path, connectome)
    with h5py.File(path, "a") as handle:
        data = handle["connectivity"][()]
        data[0] = np.nan
        del handle["connectivity"]
        handle.create_dataset("connectivity", data=data)
    results = _results(validate_file(path))
    assert not results["finite"]


def test_the_stored_dtype_is_checked_on_disk(tmp_path, connectome):
    """The loader casts to float32, so the check has to look at the file."""
    import h5py

    path = _write(tmp_path, connectome)
    with h5py.File(path, "a") as handle:
        data = handle["connectivity"][()].astype(np.float64)
        del handle["connectivity"]
        handle.create_dataset("connectivity", data=data)
    assert not _results(validate_file(path))["symmetry"]


def test_a_small_relative_leak_onto_the_wall_fails(tmp_path, connectome):
    """The tolerance is relative to the file's own scale, not an absolute 1e-5."""
    import h5py

    path = _write(tmp_path, connectome)
    with h5py.File(path, "a") as handle:
        data = handle["connectivity"][()]
        data[3] = 1e-4 * np.abs(data).sum()  # pair (0, 4): vertex 4 is the wall
        del handle["connectivity"]
        handle.create_dataset("connectivity", data=data)
    assert not _results(validate_file(path))["mask"]


def test_area_weights_must_be_positive(tmp_path, connectome):
    import h5py

    path = _write(tmp_path, connectome)
    with h5py.File(path, "a") as handle:
        area = handle["area"][()]
        area[1] = -1.0
        del handle["area"]
        handle.create_dataset("area", data=area)
    assert not _results(validate_file(path))["area"]


def test_corrupt_metadata_is_a_failed_check_not_a_crash(tmp_path, connectome):
    import h5py

    path = _write(tmp_path, connectome)
    with h5py.File(path, "a") as handle:
        del handle["metadata"]
        handle.create_dataset("metadata", data="{not json")
    checks = validate_file(path)
    assert checks[0].name == "readable" and not checks[0].passed


# --- the review of 11 October 2026 -------------------------------------------------


def test_a_file_named_in_upper_case_validates(tmp_path):
    """load() and save() take sub-01_sc.H5; validate failed its name, and sbci cohort dropped it."""
    import sbci

    path = sbci.example().save(tmp_path / "sub-01_sc.H5")
    failed = [str(check) for check in validate_file(path) if not check.passed]
    assert failed == []
    assert main(["validate", str(path)]) == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hemisphere_order": 0}, "hemisphere_order"),
        ({"hemisphere_order": "LR"}, "hemisphere_order"),
        ({"hemisphere_order": {"L": 1, "R": 2}}, "hemisphere_order"),
        ({"included_connections": None}, "included_connections"),
        (b"\xff\xfe not text", "not UTF-8"),
    ],
)
def test_malformed_metadata_is_refused_in_the_package_own_errors(tmp_path, change, message):
    """Each of these escaped SbciError, crashed `sbci validate` and `sbci info`, or passed.

    A number for hemisphere_order, a string or mapping that iterates to L then R, a missing
    modality beside endpoints, and metadata bytes that are not text.
    """
    import json

    import h5py

    import sbci

    path = sbci.example().save(tmp_path / "sub-01_sc.h5")
    with h5py.File(path, "r+") as handle:
        if isinstance(change, bytes):
            raw = change
        else:
            metadata = json.loads(handle["metadata"][()])
            for key, value in change.items():
                if value is None:
                    metadata.pop(key)
                else:
                    metadata[key] = value
            raw = json.dumps(metadata)
        del handle["metadata"]
        handle["metadata"] = raw
    with pytest.raises(SbciError, match=message):
        sbci.load(path)
    checks = validate_file(path)
    assert not all(check.passed for check in checks)
    assert main(["info", str(path)]) == 1


def test_a_malformed_file_fails_its_checks_rather_than_crashing(tmp_path):
    """A group where /area should be, or text for the connectivity, raised a TypeError."""
    import shutil

    import h5py

    import sbci

    good = sbci.example().save(tmp_path / "sub-01_sc.h5")
    for name, how in (("area", "group"), ("coordinates", "group"), ("connectivity", "text")):
        broken = tmp_path / f"sub-{name}{how}_sc.h5"
        shutil.copy(good, broken)
        with h5py.File(broken, "r+") as handle:
            if name in handle:
                del handle[name]
            if how == "group":
                handle.create_group(name)
            else:
                handle.create_dataset(name, data=np.array([b"a", b"b", b"c"]))
        with pytest.raises(SbciError):
            sbci.load(broken)
        assert not validate_file(broken)[0].passed
        assert main(["info", str(broken)]) == 1


# --- the third review of 12 October 2026 (PORTING.md item 24) --------------------------


@pytest.fixture(scope="module")
def example_files(tmp_path_factory):
    """An example SC with endpoints and an example FC, written once for these tests."""
    import sbci

    folder = tmp_path_factory.mktemp("examples")
    return {
        "sc": sbci.example("sc", seed=0).save(folder / "sub-01_sc.h5"),
        "fc": sbci.example("fc", seed=0).save(folder / "sub-01_fc.h5"),
    }


def _changed(example_files, modality, tmp_path, change):
    import shutil

    import h5py

    path = tmp_path / f"sub-02_{modality}.h5"
    shutil.copy(example_files[modality], path)
    with h5py.File(path, "r+") as handle:
        change(handle)
    return path


@pytest.mark.parametrize(
    ("name", "shape"),
    [("connectivity", (-1, 1)), ("connectivity", (1, -1)), ("area", (-1, 1)), ("mask", (1, -1))],
)
def test_an_array_stored_on_two_axes_is_refused_and_does_not_crash(
    example_files, tmp_path, name, shape
):
    """Sizes were compared, not shapes: validate crashed on some, and passed (1, n) data."""
    import sbci

    def reshape(handle):
        values = handle[name][()]
        del handle[name]
        handle.create_dataset(name, data=values.reshape(shape))

    path = _changed(example_files, "sc", tmp_path, reshape)
    with pytest.raises(SbciError):
        sbci.load(path)
    assert not _results(validate_file(path))["shapes"]
    assert main(["validate", str(path)]) == 1
    assert main(["info", str(path)]) == 1


@pytest.mark.parametrize(
    "form", ["a group in place of a dataset", "a dataset in place of the group"]
)
def test_endpoints_of_the_wrong_kind_are_refused_and_do_not_crash(example_files, tmp_path, form):
    import sbci

    def damage(handle):
        if form == "a group in place of a dataset":
            del handle["endpoints/vertex_in"]
            handle["endpoints"].create_group("vertex_in")
        else:
            del handle["endpoints"]
            handle.create_dataset("endpoints", data=3)

    path = _changed(example_files, "sc", tmp_path, damage)
    with pytest.raises(SbciError):
        sbci.load(path)
    assert main(["validate", str(path)]) == 1
    assert main(["info", str(path)]) == 1


def test_a_damaged_chunk_is_an_sbci_error(example_files, tmp_path):
    """h5py reads a corrupt gzip chunk as an OSError, which escaped the package's errors."""
    import shutil

    import h5py

    import sbci

    path = tmp_path / "sub-02_fc.h5"
    shutil.copy(example_files["fc"], path)
    with h5py.File(path, "r") as handle:
        chunk = handle["connectivity"].id.get_chunk_info(0)
    raw = bytearray(path.read_bytes())
    for k in range(chunk.byte_offset + 100, chunk.byte_offset + 400):
        raw[k] ^= 0x5A
    path.write_bytes(bytes(raw))
    with pytest.raises(SbciError, match="cannot be read"):
        sbci.load(path)
    assert main(["info", str(path)]) == 1


def test_fractional_endpoint_indices_in_a_file_are_refused(example_files, tmp_path):
    """Cast to integers before they were checked, 2965.6 read as vertex 2965."""
    import sbci

    def fractional(handle):
        vertex = handle["endpoints/vertex_in"][()].astype(np.float64)
        vertex[:3] += 0.6
        del handle["endpoints/vertex_in"]
        handle["endpoints"].create_dataset("vertex_in", data=vertex)

    path = _changed(example_files, "sc", tmp_path, fractional)
    with pytest.raises(SbciError, match="whole numbers"):
        sbci.load(path)


@pytest.mark.parametrize(
    ("change", "check", "passes"),
    [
        ("mass 1 + 2e-5", "unit mass", False),
        ("mass 1 + 5e-6", "unit mass", True),
        ("area 2% high", "area", False),
        ("area 0.5% high", "area", True),
        ("one entry -1e-6", "nonnegativity", False),
    ],
)
def test_the_validators_thresholds_are_where_it_says(
    example_files, tmp_path, change, check, passes
):
    """Each could be moved -- the mass to 1e-2, the area to 50%, the sign to -1e-3 -- unnoticed."""

    def edit(handle):
        if change.startswith("mass"):
            total = float(change.split("+ ")[1])
            values = handle["connectivity"][()].astype(np.float64)
            area = handle["area"][()]
            from sbci.grid import to_dense

            mass = area @ to_dense(values, area.size) @ area
            handle["connectivity"][...] = (values * (1 + total) / mass).astype(np.float32)
        elif change.startswith("area"):
            handle["area"][...] = handle["area"][()] * (1.02 if "2%" in change else 1.005)
        else:
            values = handle["connectivity"][()]
            values[np.flatnonzero(values == 0)[0]] = -1e-6
            handle["connectivity"][...] = values

    path = _changed(example_files, "sc", tmp_path, edit)
    assert bool(_results(validate_file(path))[check]) is passes


@pytest.mark.parametrize(("version", "accepted"), [("0.1.5", True), ("0.2.0", False)])
def test_a_spec_version_must_share_the_major_and_minor_number(version, accepted):
    """Comparing the major number alone passed every test."""
    from sbci.metadata import MetadataError, template

    metadata = template(
        "fc",
        registration_reference="fsaverage",
        pipeline_version="p",
        container_version="c",
        normalization="none",
        fc_nuisance_model="gsr",
        fc_frames=4800,
        fc_runs=["a"],
    )
    metadata.fields["spec_version"] = version
    if accepted:
        metadata.validate()
    else:
        with pytest.raises(MetadataError, match="not compatible"):
            metadata.validate()


# --- the fourth review of 12 October 2026 ------------------------------------------------


def _mask_stored_as(values):
    def change(handle):
        mask = handle["mask"][()]
        del handle["mask"]
        handle.create_dataset("mask", data=np.where(mask, values[0], values[1]))

    return change


@pytest.mark.parametrize("value", [np.nan, 2.0, -1.0, 0.5])
def test_a_mask_of_values_other_than_0_and_1_is_refused_before_the_cast(
    example_files, tmp_path, value
):
    """Cast to booleans first, NaN, 2, -1 and 0.5 all read as cortex, and passed every check."""
    import sbci
    from sbci.errors import InvalidFileError

    path = _changed(example_files, "fc", tmp_path, _mask_stored_as((value, 0.0)))
    with pytest.raises(InvalidFileError, match="/mask holds 4685 values other than 0 and 1"):
        sbci.load(path)
    checks = validate_file(path)
    assert [c.name for c in checks] == ["readable"] and not checks[0].passed


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_a_mask_of_0s_and_1s_reads_alike_whatever_its_numbers(example_files, tmp_path, dtype):
    import sbci

    path = _changed(example_files, "fc", tmp_path, _mask_stored_as((dtype(1), dtype(0))))
    np.testing.assert_array_equal(sbci.load(path).mask, sbci.example("fc").mask)
    assert all(check.passed for check in validate_file(path))


def test_an_fc_file_with_no_cortex_fails_the_mask_check(example_files, tmp_path):
    """Nothing kept, nothing leaked: zeros under a mask of no cortex passed all eleven checks."""

    def no_cortex(handle):
        handle["mask"][...] = False
        handle["connectivity"][...] = 0.0

    failed = {
        check.name: check.detail
        for check in validate_file(_changed(example_files, "fc", tmp_path, no_cortex))
        if not check.passed
    }
    assert failed == {"mask": "no vertex is cortex: the mask keeps nothing"}


def _fc_with(tmp_path, change):
    """The FC example with its values changed, written uncompressed to keep the test quick."""
    import sbci

    fc = sbci.example("fc")
    values = change(fc.data.astype(np.float64)).astype(np.float32)
    return write_hdf5(
        tmp_path / "sub-02_fc.h5", values, fc.area, fc.mask, fc.metadata, compression=None
    )


def _at_extremes(high, low):
    def change(r):
        return np.where(r == r.max(), high, np.where(r == r.min(), low, r))

    return change


@pytest.mark.parametrize(
    ("change", "passes"),
    [
        (lambda r: np.arctanh(np.clip(r, -0.999, 0.999)), False),  # stored as Fisher z
        (_at_extremes(1.0, -1.001), False),  # one entry past -1 by 1e-3
        (_at_extremes(1.0, -float(np.float32(1 + 5e-5))), True),  # 1, and rounding past -1
    ],
    ids=["fisher z", "one entry past -1", "rounding"],
)
def test_fc_beyond_minus_one_to_one_fails_the_correlation_check(tmp_path, change, passes):
    """FC stored as Fisher z passed every check, and to_atlas clipped it to 0.999999 silently."""
    import sbci

    path = _fc_with(tmp_path, change)
    failed = [check.name for check in validate_file(path) if not check.passed]
    assert failed == ([] if passes else ["correlation"])
    if not passes:
        with pytest.raises(ValueError, match="needs correlations, which lie in"):
            sbci.load(path).to_atlas("Desikan")


def test_the_correlation_check_is_twelfth_and_skipped_for_sc(example_files):
    checks = validate_file(example_files["sc"])
    assert [c.name for c in checks] == [
        "readable",
        "name",
        "metadata",
        "grid",
        "shapes",
        "symmetry",
        "finite",
        "nonnegativity",
        "correlation",
        "area",
        "unit mass",
        "mask",
        "endpoints",
    ]
    assert all(c.passed for c in checks)
    assert str(checks[8]) == "[PASS] correlation: skipped: only FC holds correlations"
    fc = {c.name: c for c in validate_file(example_files["fc"])}
    assert str(fc["correlation"]) == "[PASS] correlation: 0 entries outside [-1, 1]"


def _bary(change):
    def edit(handle):
        values = handle["endpoints/barycentric_in"][()]
        del handle["endpoints/barycentric_in"]
        handle["endpoints"].create_dataset("barycentric_in", data=change(values))

    return edit


def _no_streamlines(handle):
    group = handle["endpoints"]
    for name in list(group):
        shape, dtype = group[name].shape, group[name].dtype
        del group[name]
        group.create_dataset(name, data=np.zeros((0,) + shape[1:], dtype=dtype))


@pytest.mark.parametrize(
    "change",
    [
        _bary(lambda w: np.where(np.arange(len(w))[:, None] == 3, np.nan, w)),
        _bary(lambda w: np.zeros(w.shape, dtype=[("w", "f4"), ("v", "f4")])),
        _no_streamlines,
        _bary(lambda w: np.where(np.arange(len(w))[:, None] == 3, 0.0, w)),
    ],
    ids=["a NaN weight", "weights of a compound type", "no streamlines", "a row of zeros"],
)
def test_endpoints_load_refuses_fail_validate_and_info_without_a_traceback(
    example_files, tmp_path, change
):
    """Each loaded and validated, or raised a TypeError out of load and sbci info."""
    import sbci

    path = _changed(example_files, "sc", tmp_path, change)
    with pytest.raises(SbciError):
        sbci.load(path)
    checks = validate_file(path)
    assert [c.name for c in checks] == ["readable"] and not checks[0].passed
    assert main(["info", str(path)]) == 1


# --- the review of 20 October 2026 -------------------------------------------------


def test_triangles_stored_in_another_face_order_fail_the_endpoints_check(example_files, tmp_path):
    """Every stored triangle once named another, a median 85 degrees off, and nothing failed."""

    def reorder(handle):  # the same triangles of each hemisphere, numbered backwards
        triangle = handle["endpoints/triangle_in"][()]
        handle["endpoints/triangle_in"][...] = triangle // 5120 * 5120 + 5119 - triangle % 5120

    checks = {c.name: c for c in validate_file(_changed(example_files, "sc", tmp_path, reorder))}
    assert not checks["endpoints"].passed and "farther than one edge" in checks["endpoints"].detail
    assert all(c.passed for name, c in checks.items() if name != "endpoints")
    good = {c.name: c for c in validate_file(example_files["sc"])}["endpoints"]
    assert good.passed and "every endpoint rebuilds within one edge (4.74 degrees)" in good.detail
    fc = {c.name: c for c in validate_file(example_files["fc"])}["endpoints"]
    assert str(fc) == "[PASS] endpoints: skipped: no endpoint positions"
