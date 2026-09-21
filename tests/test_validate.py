"""The validator and the `sbci validate` command."""

from __future__ import annotations

import numpy as np
import pytest

from sbci.cli import main
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
    assert results["metadata"]
    assert results["symmetry"]
    assert results["nonnegativity"]
    assert results["unit mass"]
    assert results["mask"]
    assert not results["grid"], "toy grid must not be mistaken for ico4"


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


def test_cli_download_explains_what_is_missing():
    with pytest.raises(SystemExit, match="WP1 data release"):
        main(["download", "hcp-ya", "--subject", "100307"])
