"""load_cohort: which subjects come in, which stay out, and the reason given for each.

The matching rules are tested on placeholder files whose headers are stubbed,
so each case can say exactly what a file holds; one test runs the whole path
on real files, written by the synthetic example, with ``validate=True``.
"""

from __future__ import annotations

import csv
import os

import pytest

import sbci
from sbci import cohort as cohort_module
from sbci import grid, io, spec
from sbci.cohort import load_cohort, parse_name, subject_id
from sbci.errors import InvalidFileError
from sbci.metadata import template

N = spec.N_VERTICES


def _header(modality="sc", **changes):
    """A header as io.read_header returns it, for a file that passes unless changed."""
    common = dict(
        normalization="unit-mass" if modality == "sc" else "none",
        registration_reference="fsaverage",
        pipeline_version="pipeline 1",
        container_version="container 1",
    )
    if modality == "sc":
        common.update(
            streamline_count=800_000, streamline_weighting="none", kernel="shk", bandwidth=0.005
        )
    else:
        common.update(fc_nuisance_model="gsr", fc_frames=4800, fc_runs=["a", "b", "c", "d"])
    sizes = changes.pop("sizes", {})
    metadata = template(modality, **{**common, **changes})
    return {
        "metadata": metadata,
        "n_connectivity": sizes.get("n_connectivity", grid.condensed_size(N)),
        "n_vertices": sizes.get("n_vertices", N),
        "n_mask": N,
        "n_endpoints": 1000 if modality == "sc" else 0,
    }


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Placeholder files in ``tmp_path``; ``headers[name]`` says what each holds."""
    headers: dict = {}

    def add(name, header=None, where=""):
        directory = tmp_path / where
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"")
        modality = parse_name(name)[2]
        headers[name] = header if header is not None else _header(modality)

    def read_header(path):
        header = headers[os.path.basename(path)]
        if isinstance(header, Exception):
            raise header
        return header

    monkeypatch.setattr(io, "read_header", read_header)
    return tmp_path, add


def _table(path, rows, delimiter=","):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerows(rows)
    return path


def test_names_and_ids():
    assert parse_name("sub-01_ses-pre_desc-x_sc.hdf5") == ("sub-01", "pre", "sc")
    assert parse_name("sub-01_sc.dconn.nii") is None and parse_name("participants.tsv") is None
    assert subject_id(100307.0) == "sub-100307" and subject_id("sub-A1") == "sub-A1"


def test_subjects_pair_their_files_and_meet_the_table(folder):
    root, add = folder
    for s in ("01", "02", "03", "04", "05"):
        add(f"sub-{s}_sc.h5")
    for s in ("01", "02", "03", "04"):
        add(f"sub-{s}_fc.h5")
    rows = [["subject", "age", "sex"], ["01", "25", "F"], ["02", "31", "M"], ["03", "NA", "F"]]
    rows += [["sub-04", "40", "M"], ["05", "22", "F"], ["06", "29", "M"]]
    table = _table(root / "participants.csv", rows)
    cohort = load_cohort(root, table, modalities=("sc", "fc"), require=["age"])
    assert cohort.subjects == ["sub-01", "sub-02", "sub-04"]
    assert [p.name for p in cohort.paths("fc")] == ["sub-01_fc.h5", "sub-02_fc.h5", "sub-04_fc.h5"]
    assert cohort.column("age").tolist() == [25.0, 31.0, 40.0]
    assert cohort.column("sex").tolist() == ["F", "M", "M"]
    assert cohort.excluded == {
        "sub-03": "no value for age",
        "sub-05": "no fc file",
        "sub-06": "no sc file; no fc file",
    }
    row = {r["subject"]: r for r in cohort.report}["sub-01"]
    assert row["included"] and row["sc_streamlines"] == 800_000 and row["fc_runs"] == 4
    assert (
        cohort.settings["sc"]["kernel"] == "shk"
        and cohort.settings["fc"]["fc_nuisance_model"] == "gsr"
    )
    summary = cohort.summary()
    assert "6 subjects seen; 3 in the cohort" in summary and "no fc file" in summary


def test_folders_and_tables_combine(folder):
    """SC and FC in separate folders, covariates in two tables joined on the subject."""
    root, add = folder
    for s in ("01", "02", "03"):
        add(f"sub-{s}_sc.h5", where="sc")
        add(f"sub-{s}_fc.h5", where="fc")
    traits = {"subject": ["01", "02", "03"], "score": ["10", "12", "NA"]}
    motion = {"participant_id": ["sub-01", "sub-03"], "motion": [0.1, 0.3]}
    cohort = load_cohort(
        [root / "sc", root / "fc", root / "sc" / "sub-01_sc.h5"],
        [traits, motion],
        modalities=("sc", "fc"),
    )
    assert cohort.subjects == ["sub-01", "sub-02", "sub-03"]
    assert (
        cohort.column("motion")[0] == 0.1
        and cohort.column("motion")[1] != cohort.column("motion")[1]
    )
    needed = load_cohort(
        [root / "sc", root / "fc"], [traits, motion], ("sc", "fc"), require=["score", "motion"]
    )
    assert needed.excluded == {"sub-02": "no value for motion", "sub-03": "no value for score"}
    with pytest.raises(ValueError, match="two tables have a column 'score'"):
        load_cohort(root / "sc", [traits, traits])


def test_bids_participant_ids_and_a_tsv(folder):
    root, add = folder
    add("sub-01_sc.h5", where="sub-01/anat")
    add("sub-02_sc.h5", where="sub-02/anat")
    table = _table(
        root / "participants.tsv",
        [["participant_id", "group"], ["sub-01", "a"], ["sub-02", "b"]],
        "\t",
    )
    cohort = load_cohort(root, table)
    assert cohort.subjects == ["sub-01", "sub-02"] and cohort.column("group").tolist() == ["a", "b"]


def test_files_that_disagree_on_a_setting_are_refused_or_left_out(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5", _header(bandwidth=0.01))
    add("sub-03_sc.h5")
    with pytest.raises(ValueError, match="differ in bandwidth.*sub-02"):
        load_cohort(root)
    kept = load_cohort(root, mismatch="exclude")
    assert kept.subjects == ["sub-01", "sub-03"]
    assert kept.excluded == {"sub-02": "sc bandwidth is 0.01, not the cohort's 0.005"}
    everyone = load_cohort(root, mismatch="report")
    assert everyone.n_subjects == 3 and everyone.settings["sc"]["bandwidth"] == [0.005, 0.01]


def test_a_tie_has_no_majority_to_keep(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5", _header(kernel="rdk"))
    with pytest.raises(ValueError, match="no setting has a majority"):
        load_cohort(root, mismatch="exclude")


def test_a_file_that_would_not_load_is_left_out_with_its_reason(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5", InvalidFileError("sub-02_sc.h5 cannot be opened as an HDF5 file"))
    add("sub-03_sc.h5", _header(kernel=None))
    add("sub-04_sc.h5", _header("fc"))
    add("sub-05_sc.h5", _header(sizes={"n_vertices": 4121}))
    cohort = load_cohort(root)
    assert cohort.subjects == ["sub-01"]
    reasons = cohort.excluded
    assert "cannot be opened" in reasons["sub-02"]
    assert "missing required metadata keys: kernel" in reasons["sub-03"]
    assert "holds fc, though its name says sc" in reasons["sub-04"]
    assert "4121 vertices" in reasons["sub-05"]


def test_two_files_for_one_subject_are_named(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5")
    add("sub-02_desc-old_sc.h5", where="old")
    cohort = load_cohort(root)
    assert cohort.excluded == {"sub-02": "2 sc files: sub-02_desc-old_sc.h5, sub-02_sc.h5"}


def test_sessions_must_be_chosen(folder):
    root, add = folder
    for s in ("01", "02"):
        add(f"sub-{s}_ses-1_fc.h5")
        add(f"sub-{s}_ses-2_fc.h5")
    with pytest.raises(ValueError, match=r"sessions \['1', '2'\].*session="):
        load_cohort(root, modalities="fc")
    second = load_cohort(root, modalities="fc", session=2)
    assert [p.name for p in second.paths("fc")] == ["sub-01_ses-2_fc.h5", "sub-02_ses-2_fc.h5"]
    assert second.session == "2"


def test_the_callers_exclusions_stand_in_the_report(folder, tmp_path):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5")
    cohort = load_cohort(root, exclude={"02": "motion above 0.2 mm"})
    assert cohort.subjects == ["sub-01"]
    assert cohort.excluded == {"sub-02": "excluded by the caller: motion above 0.2 mm"}
    path = cohort.save_report(tmp_path / "qc.tsv")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows[0][:3] == ["subject", "included", "reason"]
    assert rows[1][:2] == ["sub-01", "1"] and rows[2][:3] == [
        "sub-02",
        "0",
        cohort.excluded["sub-02"],
    ]


def test_nobody_passing_says_why(folder):
    root, add = folder
    add("sub-01_sc.h5")
    table = {"subject": ["0001"], "age": [30]}
    with pytest.raises(ValueError, match="no subject passed: 1 no sc file; 1 not in the table"):
        load_cohort(root, table)


def test_what_is_refused_before_anything_is_read(folder, tmp_path):
    root, add = folder
    add("sub-01_sc.h5")
    with pytest.raises(ValueError, match="modalities must be drawn from"):
        load_cohort(root, modalities=("dwi",))
    with pytest.raises(ValueError, match="mismatch must be one of"):
        load_cohort(root, mismatch="ignore")
    with pytest.raises(ValueError, match="no table was given"):
        load_cohort(root, require=["age"])
    with pytest.raises(ValueError, match="no subject column"):
        load_cohort(root, {"id": ["01"]})
    with pytest.raises(ValueError, match="more than once"):
        load_cohort(root, {"subject": ["01", "sub-01"]})
    (tmp_path / "subject01.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="is not named sub-"):
        load_cohort([tmp_path / "subject01.h5"])
    with pytest.raises(FileNotFoundError):
        load_cohort(tmp_path / "nowhere")


# --- real files ---------------------------------------------------------------


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """One synthetic SC file on disk, linked under two subjects' names."""
    root = tmp_path_factory.mktemp("cohort")
    source = sbci.example("sc").save(root / "example_sc.h5")
    for subject in ("sub-a", "sub-b"):
        os.symlink(source, root / f"{subject}_sc.h5")
    return root


def test_real_files_pass_their_headers_and_every_check(written):
    header = io.read_header(written / "sub-a_sc.h5")
    assert header["n_vertices"] == N and header["n_endpoints"] > 0
    cohort = load_cohort(written, {"subject": ["a", "b"], "score": [1.5, None]}, validate=True)
    assert cohort.subjects == ["sub-a", "sub-b"]
    assert (
        cohort.column("score")[0] == 1.5 and cohort.column("score")[1] != cohort.column("score")[1]
    )
    row = cohort.report[0]
    assert (
        row["sc_endpoints"] == 1 and row["sc_streamlines"] == header["metadata"]["streamline_count"]
    )
    assert cohort.settings["sc"]["grid"] == spec.GRID


def test_the_command_line_writes_the_report(written, tmp_path, capsys):
    from sbci.cli import main

    report = tmp_path / "qc.csv"
    assert main(["cohort", str(written), "--report", str(report)]) == 0
    assert "2 subjects seen; 2 in the cohort" in capsys.readouterr().out
    assert report.read_text().splitlines()[0].startswith("subject,included,reason")
    assert cohort_module.SETTINGS["sc"][0] == "spec_version"
