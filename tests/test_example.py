"""The example connectome, which is what a new user runs first.

`sbci download` needs the data release, so until that exists `sbci.example()`
is the only thing anyone can run. It therefore has to be exactly as valid as a
real file, and it has to say clearly that it is not one.
"""

from __future__ import annotations

import numpy as np
import pytest

import sbci
from sbci.cli import main
from sbci.validate import validate_file


@pytest.fixture(scope="module")
def sc():
    return sbci.example("sc")


@pytest.fixture(scope="module")
def fc():
    return sbci.example("fc")


def test_it_is_on_the_real_grid(sc):
    assert sc.n_vertices == sbci.spec.N_VERTICES


def test_it_is_reproducible():
    np.testing.assert_array_equal(sbci.example(seed=3).data, sbci.example(seed=3).data)


def test_different_seeds_differ():
    assert not np.array_equal(sbci.example(seed=1).data, sbci.example(seed=2).data)


def test_sc_is_a_density_of_unit_mass(sc):
    dense = sc.dense().astype(np.float64)
    assert float(sc.area @ dense @ sc.area) == pytest.approx(1.0, abs=1e-5)


def test_sc_is_nonnegative(sc):
    assert sc.data.min() >= 0.0


def test_fc_is_signed(fc):
    assert fc.data.min() < 0 < fc.data.max()


def test_fc_stays_within_correlation_bounds(fc):
    assert fc.data.min() >= -1.0 and fc.data.max() <= 1.0


@pytest.mark.parametrize("modality", ["sc", "fc"])
def test_the_medial_wall_carries_nothing(modality):
    cc = sbci.example(modality)
    assert float(np.abs(cc.dense()[~cc.mask]).sum()) == 0.0


def test_the_mask_is_the_real_medial_wall(sc):
    """Not invented: it is where the bundled Desikan atlas assigns no region."""
    np.testing.assert_array_equal(sc.mask, sbci.load_atlas("Desikan").labels != 0)


@pytest.mark.parametrize("modality", ["sc", "fc"])
def test_it_passes_the_validator(modality, tmp_path):
    path = tmp_path / f"sub-example_{modality}.h5"
    sbci.example(modality).save(path)
    failed = [c for c in validate_file(path) if not c.passed]
    assert not failed, [str(c) for c in failed]


def test_the_metadata_says_it_is_synthetic(sc):
    """Someone must never mistake this for measured data."""
    assert "synthetic" in sc.metadata.fields["pipeline_version"]
    assert sc.metadata.fields["streamline_count"] == 0


def test_an_unknown_modality_is_refused():
    with pytest.raises(ValueError, match="modality must be one of"):
        sbci.example("dwi")


def test_the_readme_walkthrough_runs(sc):
    """Every line of the five-minute start, minus the download and the 16.9 GB."""
    assert sc.to_atlas("Schaefer200").shape == (200, 200)
    assert sc.seed(vertex=1234).shape == (sbci.spec.N_VERTICES,)
    assert sc.seed(region=("Desikan", "LH_bankssts")).shape == (sbci.spec.N_VERTICES,)


def test_cli_writes_a_file_that_validates(tmp_path, capsys):
    path = tmp_path / "example.h5"
    assert main(["example", "--out", str(path)]) == 0
    assert "not measured data" in capsys.readouterr().out
    assert not [c for c in validate_file(path) if not c.passed]


def test_cli_info_reports_the_modality(tmp_path, capsys):
    path = tmp_path / "example.h5"
    main(["example", "--out", str(path)])
    capsys.readouterr()
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "modality        sc" in out
    assert "synthetic-example" in out


def test_cli_atlases_lists_and_filters(capsys):
    assert main(["atlases", "--match", "Yeo"]) == 0
    out = capsys.readouterr().out
    assert "Yeo2011_7Networks_N1000" in out and "Schaefer" not in out


def test_cli_atlases_reports_no_match(capsys):
    assert main(["atlases", "--match", "nonesuch"]) == 1
    assert "no bundled atlas matches" in capsys.readouterr().out
