"""load_cohort: which subjects come in, which stay out, and the reason given for each.

The matching rules are tested on placeholder files whose headers are stubbed,
so each case can say exactly what a file holds; one test runs the whole path
on real files, written by the synthetic example, with ``validate=True``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
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
        "coordinates": sizes.get("coordinates"),
        "n_endpoints": 1000 if modality == "sc" else 0,
        "has_endpoints": modality == "sc",
        "flat": True,
        "numeric": True,
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
    with pytest.raises(ValueError, match="sc bandwidth: 0.005 in 2, 0.01 in 1.*sub-02"):
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
    with pytest.raises(ValueError, match="no way of making them is the most common"):
        load_cohort(root, mismatch="exclude")


def test_a_file_that_would_not_load_is_left_out_with_its_reason(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5", InvalidFileError("sub-02_sc.h5 cannot be opened as an HDF5 file"))
    add("sub-03_sc.h5", _header(kernel=None))
    add("sub-04_sc.h5", _header("fc"))
    add("sub-05_sc.h5", _header(sizes={"n_vertices": 4121}))
    add("sub-06_sc.h5", _header(sizes={"coordinates": (N, 2)}))
    cohort = load_cohort(root)
    assert cohort.subjects == ["sub-01"]
    assert f"coordinates are ({N}, 2)" in cohort.excluded["sub-06"]
    reasons = cohort.excluded
    assert "cannot be opened" in reasons["sub-02"]
    assert "missing required metadata keys: kernel" in reasons["sub-03"]
    assert "holds fc, though its name says sc" in reasons["sub-04"]
    assert "area has 4121 entries" in reasons["sub-05"]


def test_an_fc_file_carrying_endpoints_is_left_out_as_load_would_refuse_it(folder):
    """Even an empty group: read_hdf5 refuses any, so load and every method after would."""
    root, add = folder
    add("sub-01_fc.h5")
    with_endpoints = _header("fc")
    with_endpoints["has_endpoints"] = True  # with no streamlines in it
    add("sub-02_fc.h5", with_endpoints)
    not_flat = _header("fc")
    not_flat["flat"] = False
    add("sub-03_fc.h5", not_flat)
    text = _header("fc")
    text["numeric"] = False
    add("sub-04_fc.h5", text)
    cohort = load_cohort(root, modalities="fc")
    assert cohort.subjects == ["sub-01"]
    assert "carries streamline endpoints" in cohort.excluded["sub-02"]
    assert "not stored flat" in cohort.excluded["sub-03"]
    assert "does not hold numbers" in cohort.excluded["sub-04"]


def test_two_files_for_one_subject_are_named(folder):
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5")
    add("sub-02_desc-old_sc.h5", where="old")
    cohort = load_cohort(root)
    assert cohort.excluded == {"sub-02": "2 sc files: old/sub-02_desc-old_sc.h5, sub-02_sc.h5"}


def test_sessions_must_be_chosen(folder):
    root, add = folder
    for s in ("01", "02"):
        add(f"sub-{s}_ses-1_fc.h5")
        add(f"sub-{s}_ses-2_fc.h5")
    with pytest.raises(ValueError, match="more than one session.*fc from 1, 2.*session="):
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


# --- the fifth review -------------------------------------------------------------


def test_sc_and_fc_from_different_visits_are_not_paired_silently(folder):
    """sub-B's SC from session 1 beside its FC from session 2 would compare two visits."""
    from sbci.cohort import CohortError

    root, add = folder
    for name in (
        "sub-A_ses-1_sc.h5",
        "sub-A_ses-1_fc.h5",
        "sub-B_ses-1_sc.h5",
        "sub-B_ses-2_fc.h5",
    ):
        add(name)
    with pytest.raises(CohortError, match="sub-B's files come from more than one session"):
        load_cohort(root, modalities=("sc", "fc"))
    same = load_cohort(root, modalities=("sc", "fc"), session=1)
    assert same.subjects == ["sub-A"]
    assert same.excluded == {"sub-B": "no fc file of session '1', only of session 2"}
    crossed = load_cohort(root, modalities=("sc", "fc"), session={"sc": "1", "fc": "2"})
    assert crossed.subjects == ["sub-B"]
    assert crossed.report[1]["sc_session"] == "1" and crossed.report[1]["fc_session"] == "2"


def test_a_session_mapping_names_every_modality(folder):
    """{"sc": "1"} alone left FC free to come from any session, and sub-B's visits paired."""
    root, add = folder
    for name in (
        "sub-A_ses-1_sc.h5",
        "sub-A_ses-1_fc.h5",
        "sub-B_ses-1_sc.h5",
        "sub-B_ses-2_fc.h5",
    ):
        add(name)
    with pytest.raises(ValueError, match="session= names sc but not fc: a modality left out"):
        load_cohort(root, modalities=("sc", "fc"), session={"sc": "1"})
    both = load_cohort(root, modalities=("sc", "fc"), session={"sc": "1", "fc": "1"})
    assert both.subjects == ["sub-A"]
    assert both.excluded == {"sub-B": "no fc file of session '1', only of session 2"}


def test_none_in_a_session_mapping_takes_the_files_without_a_label(folder):
    """The HCP's SC has no session and its FC one a day: {"sc": None, "fc": "REST1"} pairs them."""
    root, add = folder
    for name in (
        "sub-A_sc.h5",
        "sub-A_ses-REST1_fc.h5",
        "sub-A_ses-REST2_fc.h5",
        "sub-B_ses-1_sc.h5",
        "sub-B_ses-REST1_fc.h5",
    ):
        add(name)
    cohort = load_cohort(root, modalities=("sc", "fc"), session={"sc": None, "fc": "REST1"})
    assert cohort.subjects == ["sub-A"]
    assert cohort.excluded == {"sub-B": "no sc file without a session label, only of session 1"}
    assert cohort.report[0]["sc_session"] is None and cohort.report[0]["fc_session"] == "REST1"
    with pytest.raises(ValueError, match="session= names fc but not sc"):
        load_cohort(root, modalities=("sc", "fc"), session={"fc": "REST1"})


def test_a_session_label_matches_with_or_without_leading_zeros(folder):
    root, add = folder
    add("sub-01_ses-01_sc.h5")
    assert load_cohort(root, session=1).subjects == ["sub-01"]
    assert load_cohort(root, session="01").subjects == ["sub-01"]


def test_a_session_label_no_file_carries_is_named_and_reported(folder):
    """session=5 over ses-1 files said no files of the asked modalities were found."""
    from sbci.cohort import CohortError

    root, add = folder
    add("sub-01_ses-1_sc.h5")
    add("sub-02_ses-1_sc.h5")
    with pytest.raises(
        CohortError, match=r"No sc file is of session '5': the files are of session 1\."
    ) as refused:
        load_cohort(root, session=5)
    assert {row["subject"]: row["reason"] for row in refused.value.report} == {
        "sub-01": "no sc file of session '5', only of session 1",
        "sub-02": "no sc file of session '5', only of session 1",
    }


def test_a_subject_without_the_asked_session_is_reported_not_dropped(folder):
    root, add = folder
    add("sub-01_ses-1_fc.h5")
    add("sub-01_ses-2_fc.h5")
    add("sub-02_ses-1_fc.h5")
    add("sub-02_ses-3_fc.h5")
    add("sub-03_fc.h5")
    cohort = load_cohort(root, modalities="fc", session=2)
    assert cohort.subjects == ["sub-01"]
    reasons = {row["subject"]: row["reason"] for row in cohort.report}
    assert reasons["sub-02"] == "no fc file of session '2', only of sessions 1 and 3"
    assert reasons["sub-03"] == "no fc file of session '2', only without a session label"


def test_a_session_label_no_file_carries_still_writes_the_report(folder, tmp_path, capsys):
    from sbci.cli import main

    root, add = folder
    add("sub-01_ses-1_sc.h5")
    report = tmp_path / "r.csv"
    assert main(["cohort", str(root), "--session", "5", "--report", str(report)]) == 1
    assert "No sc file is of session '5'" in capsys.readouterr().err
    assert "no sc file of session '5', only of session 1" in report.read_text()
    other = tmp_path / "other"
    other.mkdir()
    add("sub-01_fc.h5", where="other")
    assert (
        main(["cohort", str(other), "--modalities", "sc", "--report", str(tmp_path / "e.csv")]) == 1
    )
    said = capsys.readouterr().err
    assert "no files of the asked modalities were found" in said and "no report written" in said
    assert not (tmp_path / "e.csv").exists()


def test_names_sbci_load_reads_are_found(folder, tmp_path):
    """Ids with underscores, values with hyphens, any case, folders behind a symbolic link."""
    import os

    root, add = folder
    add("sub-NDAR_INV1_sc.h5")
    add("sub-02_acq-multi-shell_sc.h5")
    add("sub-03_SC.H5")
    add("sub-04_sc.h5", where="elsewhere")
    os.symlink(root / "elsewhere", root / "linked")
    (root / "notes_sc.h5").write_bytes(b"")  # says no subject: listed, not read
    cohort = load_cohort(root, {"subject": ["NDAR_INV1", "02", "03", "04"]})
    assert cohort.subjects == ["sub-02", "sub-03", "sub-04", "sub-NDAR_INV1"]
    assert [p.name for p in cohort.unread] == ["notes_sc.h5"]
    assert "not read, 1 .h5 file" in cohort.summary()


def test_the_most_common_way_of_making_the_files_is_kept_whole(folder):
    """Kept together, every setting of every modality: per-setting majorities could keep no one.

    Three files share shk at 0.005 and four disagree; by kernel alone the
    majority is shk and by bandwidth alone 0.01, which no file has.
    """
    root, add = folder
    ways = [("shk", 0.005)] * 3 + [("rdk", 0.01)] * 2 + [("matern", 0.01)] * 2
    for i, (kernel, bandwidth) in enumerate(ways):
        add(f"sub-{i:02d}_sc.h5", _header(kernel=kernel, bandwidth=bandwidth))
    kept = load_cohort(root, mismatch="exclude")
    assert kept.subjects == ["sub-00", "sub-01", "sub-02"]
    assert kept.settings["sc"]["kernel"] == "shk" and kept.settings["sc"]["bandwidth"] == 0.005
    assert "sc kernel is 'rdk', not the cohort's 'shk'" in kept.excluded["sub-03"]


def test_the_way_kept_does_not_depend_on_the_order_of_the_modalities(folder):
    root, add = folder
    for i in range(5):
        add(f"sub-{i}_sc.h5", _header(bandwidth=0.005 if i < 4 else 0.01))
        add(f"sub-{i}_fc.h5", _header("fc", fc_nuisance_model="gsr" if i != 1 else "none"))
    one = load_cohort(root, modalities=("sc", "fc"), mismatch="exclude")
    other = load_cohort(root, modalities=("fc", "sc"), mismatch="exclude")
    assert one.subjects == other.subjects == ["sub-0", "sub-2", "sub-3"]


def test_numbers_compare_as_numbers(folder):
    root, add = folder
    add("sub-01_sc.h5", _header(bandwidth=2))
    add("sub-02_sc.h5", _header(bandwidth=2.0))
    assert load_cohort(root).n_subjects == 2


def test_matern_files_differ_by_nu_and_one_made_before_it_was_recorded_has_three(folder):
    """Matern files at nu 1 and 3, up to 7% apart, came in together: nu was no setting."""
    root, add = folder
    add("sub-01_sc.h5", _header(kernel="matern", bandwidth=0.2, nu=3.0))
    add("sub-02_sc.h5", _header(kernel="matern", bandwidth=0.2))  # written before nu was
    add("sub-03_sc.h5", _header(kernel="matern", bandwidth=0.2, nu=1))
    with pytest.raises(
        ValueError, match=r"sc nu: 3\.0 in 2, 1\.0 in 1 \(first of the others: sub-03"
    ):
        load_cohort(root)
    kept = load_cohort(root, mismatch="exclude")
    assert kept.subjects == ["sub-01", "sub-02"] and kept.settings["sc"]["nu"] == 3.0


def test_a_table_with_two_columns_of_one_name_is_refused(folder, tmp_path):
    root, add = folder
    add("sub-01_sc.h5")
    table = _table(tmp_path / "t.csv", [["subject", "age", "age"], ["01", "30", "99"]])
    with pytest.raises(ValueError, match="two columns named 'age'"):
        load_cohort(root, table)


def test_what_counts_as_missing(folder):
    """Pandas' defaults, case and all: None is missing, none is a value; missing= adds markers."""
    root, add = folder
    add("sub-01_sc.h5")
    add("sub-02_sc.h5")
    table = {"subject": ["01", "02"], "medication": ["none", "None"], "site": ["NA", "B"]}
    cohort = load_cohort(root, table)
    assert cohort.column("medication").tolist() == ["none", None]
    assert cohort.column("site").tolist() == [None, "B"]
    literal = load_cohort(root, table, missing={""}, keep_default_missing=False)
    assert literal.column("site").tolist() == ["NA", "B"]


def test_a_required_column_named_like_a_report_field_is_refused(folder):
    root, add = folder
    add("sub-01_sc.h5")
    with pytest.raises(ValueError, match="'reason' would overwrite the report's own field"):
        load_cohort(root, {"subject": ["01"], "reason": ["x"]}, require=["reason"])


def test_frames_given_run_by_run_are_counted_in_the_report(folder, tmp_path):
    root, add = folder
    add("sub-01_fc.h5", _header("fc", fc_frames=[1200, 1200]))
    cohort = load_cohort(root, modalities="fc")
    assert cohort.report[0]["fc_frames"] == 2400
    assert cohort.save_report(tmp_path / "r.csv").exists()


def test_a_refused_cohort_still_writes_its_report_from_the_command_line(folder, tmp_path, capsys):
    from sbci.cli import main

    root, add = folder
    add("sub-01_sc.h5", _header(bandwidth=0.005))
    add("sub-02_sc.h5", _header(bandwidth=0.01))
    report = tmp_path / "qc.csv"
    assert main(["cohort", str(root), "--report", str(report)]) == 1
    assert "not all made the same way" in capsys.readouterr().err
    rows = report.read_text().splitlines()
    assert len(rows) == 3 and all(",0,refused with the cohort" in row for row in rows[1:])
    assert "sc bandwidth is 0.01" in rows[2]
    assert (
        main(["cohort", str(root), "--exclude", "sub-02=bandwidth", "--report", str(report)]) == 0
    )
    assert "excluded by the caller: bandwidth" in report.read_text()


def test_a_real_file_that_load_refuses_is_refused_here_too(written, tmp_path):
    """An endpoints group without vertex_out: load raises, so the loader leaves it out."""
    import shutil

    import h5py

    copy = tmp_path / "sub-z_sc.h5"
    shutil.copy(os.path.realpath(written / "sub-a_sc.h5"), copy)
    with h5py.File(copy, "a") as handle:
        del handle["endpoints"]["vertex_out"]
    with pytest.raises(sbci.errors.SbciError):
        sbci.load(copy)
    cohort = load_cohort([written / "sub-a_sc.h5", copy])
    assert cohort.subjects == ["sub-a"] and "vertex_out" in cohort.excluded["sub-z"]


# --- the seventh review -----------------------------------------------------------


def test_missing_markers_add_to_the_defaults(folder):
    """missing="-999" was split into "-", "9", and replaced the defaults: NA became a level."""
    root, add = folder
    for subject in ("01", "02", "03"):
        add(f"sub-{subject}_sc.h5")
    table = {
        "subject": ["01", "02", "03"],
        "score": ["12", "-999", "15"],
        "group": ["a", "NA", "b"],
    }
    cohort = load_cohort(root, table, missing="-999")
    assert np.isnan(cohort.column("score")[1]) and cohort.column("group").tolist() == [
        "a",
        None,
        "b",
    ]
    listed = load_cohort(root, table, missing=["-999", "a"])
    assert listed.column("group").tolist() == [None, None, "b"]


def test_the_session_check_heeds_exclude_and_reports_a_refusal(folder):
    from sbci.cohort import CohortError

    root, add = folder
    for name in (
        "sub-A_ses-1_sc.h5",
        "sub-A_ses-2_fc.h5",
        "sub-B_ses-1_sc.h5",
        "sub-B_ses-1_fc.h5",
    ):
        add(name)
    kept = load_cohort(root, modalities=("sc", "fc"), exclude={"sub-A": "two visits"})
    assert kept.subjects == ["sub-B"] and kept.excluded == {
        "sub-A": "excluded by the caller: two visits"
    }
    with pytest.raises(
        CohortError, match="sub-A's files come from more than one session"
    ) as refused:
        load_cohort(root, modalities=("sc", "fc"))
    reasons = {row["subject"]: row["reason"] for row in refused.value.report}
    assert reasons["sub-A"] == "files from more than one session (sc from 1; fc from 2)"
    assert reasons["sub-B"].startswith("refused with the cohort")


def test_a_name_extending_another_subjects_is_set_aside_without_a_table(folder):
    """Without a table, a name extending another subject's id is reported, not read.

    sub-01_old was a subject of its own; merged into sub-01 instead, it took
    sub-NDAR_INV1_2, a real subject, with it, and both NDAR subjects dropped out.
    """
    root, add = folder
    for name in (
        "sub-01_sc.h5",
        "sub-01_old_sc.h5",
        "sub-02_sc.h5",
        "sub-NDAR_INV1_sc.h5",
        "sub-NDAR_INV1_2_sc.h5",
    ):
        add(name)
    cohort = load_cohort(root)
    assert cohort.subjects == ["sub-01", "sub-02", "sub-NDAR_INV1"]
    assert set(cohort.excluded) == {"sub-01_old", "sub-NDAR_INV1_2"}
    assert "its name extends sub-NDAR_INV1's" in cohort.excluded["sub-NDAR_INV1_2"]
    listed = load_cohort(root, {"subject": ["01", "02", "NDAR_INV1", "NDAR_INV1_2"]})
    assert listed.subjects == ["sub-01", "sub-02", "sub-NDAR_INV1", "sub-NDAR_INV1_2"]


def test_numbers_written_with_a_leading_zero_stay_numbers_and_warn(folder):
    """One 07 among the ages turned the column to text, and design() made it a 4-level factor."""
    from sbci.stats import design

    root, add = folder
    for subject in ("01", "02", "03", "04", "05"):
        add(f"sub-{subject}_sc.h5")
    table = {"subject": ["01", "02", "03", "04", "05"], "age": ["07", "30", "25", "41", "33"]}
    with pytest.warns(UserWarning, match="'age' is read as numbers.*categorical=\\['age'\\]"):
        cohort = load_cohort(root, table)
    assert cohort.column("age").tolist() == [7.0, 30.0, 25.0, 41.0, 33.0]
    assert design({"age": cohort.column("age")}).names == ("intercept", "age")


# --- the eighth review ------------------------------------------------------------


def test_a_file_reached_through_two_links_counts_once(folder):
    root, add = folder
    add("sub-01_sc.h5", where="a")
    (root / "b").symlink_to(root / "a")
    cohort = load_cohort([root / "a", root / "b"])
    assert cohort.subjects == ["sub-01"]


def test_a_row_longer_than_the_header_is_refused(folder, tmp_path):
    """An unquoted comma inside a value shifted the row: an age of 31 was read as 2."""
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    shifted = _table(
        tmp_path / "subjects.csv",
        [["subject", "site", "age"], ["01", "UNC", "25"], ["02", "Site 1", " 2", "31"]],
    )
    with pytest.raises(ValueError, match="subjects.csv, line 3: 4 cells under 3 column names"):
        load_cohort(root, shifted)
    trailing = _table(
        tmp_path / "trailing.csv", [["subject", "age"], ["01", "25", ""], ["02", "31"]]
    )
    assert load_cohort(root, trailing).column("age").tolist() == [25.0, 31.0]


def test_missing_markers_match_numbers_however_written(folder, tmp_path):
    """missing="-999" missed -999.0 and a number -999, and missing=-999 raised a TypeError."""
    root, add = folder
    for subject in ("01", "02", "03"):
        add(f"sub-{subject}_sc.h5")
    table = _table(
        tmp_path / "scores.csv", [["subject", "age"], ["01", "25"], ["02", "-999.0"], ["03", "30"]]
    )
    for marker in ("-999", -999, -999.0):
        assert np.isnan(load_cohort(root, table, missing=marker).column("age")[1])
    as_numbers = {"subject": ["01", "02", "03"], "age": [25, -999, 30]}
    assert np.isnan(load_cohort(root, as_numbers, missing="-999").column("age")[1])


def test_every_pandas_default_marker_reads_as_missing(folder):
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    for marker in ("#N/A N/A", "-1.#IND", "-1.#QNAN", "1.#IND", "1.#QNAN"):
        cohort = load_cohort(root, {"subject": ["01", "02"], "age": ["25", marker]})
        assert np.isnan(cohort.column("age")[1]), marker


def test_a_dataframe_is_a_table(folder):
    """A DataFrame was taken for a list of tables, its column names for paths."""
    pandas = pytest.importorskip("pandas")
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    frame = pandas.DataFrame({"subject": ["01", "02"], "age": [25.0, 31.0]})
    assert load_cohort(root, frame).column("age").tolist() == [25.0, 31.0]


def test_one_visit_written_two_ways_is_one_visit(folder):
    """ses-1 and ses-01 were refused as files from more than one session."""
    root, add = folder
    add("sub-A_ses-1_sc.h5")
    add("sub-A_ses-01_fc.h5")
    assert load_cohort(root, modalities=("sc", "fc")).subjects == ["sub-A"]


def test_an_exclude_key_that_matches_no_one_warns(folder):
    root, add = folder
    add("sub-01_sc.h5")
    with pytest.warns(
        UserWarning, match=r"exclude= names 1 subject\(s\) in no file or table: sub-1"
    ):
        cohort = load_cohort(root, exclude={"sub-1": "a typo for sub-01"})
    assert cohort.subjects == ["sub-01"]


def test_a_link_to_nothing_is_reported_under_its_subject(folder):
    """A broken link named as a subject's file was listed as a name that says no subject."""
    root, add = folder
    add("sub-01_sc.h5")
    (root / "sub-02_sc.h5").symlink_to(root / "nowhere.h5")
    cohort = load_cohort(root)
    assert cohort.subjects == ["sub-01"]
    assert "a link to nothing" in cohort.excluded["sub-02"]


def test_without_a_table_the_refusal_does_not_ask_about_one(folder):
    from sbci.cohort import CohortError

    root, add = folder
    add("sub-01_sc.h5", header=ValueError("unreadable"))
    with pytest.raises(CohortError, match="unreadable") as refused:
        load_cohort(root)
    assert "table" not in str(refused.value)


# --- the review of 11 October 2026 -------------------------------------------------


def test_a_subject_the_table_leaves_out_is_not_asked_about_its_visits(folder):
    """sub-99, in no table, had sc from one visit and fc from another, and refused everyone."""
    root, add = folder
    for subject in ("01", "02", "03"):
        add(f"sub-{subject}_ses-1_sc.h5")
        add(f"sub-{subject}_ses-1_fc.h5")
    add("sub-99_ses-1_sc.h5")
    add("sub-99_ses-2_fc.h5")
    table = {"subject": ["01", "02", "03"], "age": [25.0, 30.0, 35.0]}
    cohort = load_cohort(root, table, modalities=("sc", "fc"))
    assert cohort.subjects == ["sub-01", "sub-02", "sub-03"]
    assert cohort.excluded["sub-99"].startswith("not in the table")


def test_codes_that_read_as_numbers_are_not_merged(folder):
    """float() read 12_345 and 123_45 as one number, 18 digits lost their last, 07 met 7."""
    root, add = folder
    subjects = ["01", "02", "03", "04", "05", "06"]
    for subject in subjects:
        add(f"sub-{subject}_sc.h5")
    table = {
        "subject": subjects,
        "family": ["12_345", "12_345", "123_45", "123_45", "9_9", "99"],
        "long": [f"10000000000000000{k}" for k in range(1, 7)],
        "site": ["07", "7", "07", "7", "08", "08"],
        "age": ["07", "30", "25", "41", "33", "28"],
    }
    with pytest.warns(UserWarning, match="'age' is read as numbers"):
        cohort = load_cohort(root, table)
    assert sorted(set(cohort.column("family"))) == ["123_45", "12_345", "99", "9_9"]
    assert len(set(cohort.column("long"))) == 6
    assert cohort.column("site").tolist() == ["07", "7", "07", "7", "08", "08"]
    assert cohort.column("age").tolist() == [7.0, 30.0, 25.0, 41.0, 33.0, 28.0]
    numbers = load_cohort(root, {"subject": subjects, "count": [2**53, 1, 2, 3, 4, 5]})
    assert numbers.column("count").dtype == np.float64


def test_missing_none_names_no_markers(folder):
    """load_cohort(missing=None) raised a TypeError; design(missing=None) took it as none."""
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    table = {"subject": ["01", "02"], "age": ["NA", "30"]}
    cohort = load_cohort(root, table, missing=None)
    assert np.isnan(cohort.column("age")[0]) and cohort.column("age")[1] == 30.0


def test_a_real_fc_file_with_an_empty_endpoint_group_is_left_out(written, tmp_path):
    """Its streamline count is zero, so the loader took it, though load refuses it."""
    import shutil

    import h5py

    import sbci

    folder = tmp_path / "fc"
    folder.mkdir()
    path = folder / "sub-01_fc.h5"
    sbci.example("fc", seed=0).save(path)
    shutil.copy(path, folder / "sub-02_fc.h5")
    with h5py.File(folder / "sub-02_fc.h5", "r+") as handle:
        group = handle.create_group("endpoints")
        group.create_dataset("vertex_in", data=np.zeros(0, dtype=np.int32))
        group.create_dataset("vertex_out", data=np.zeros(0, dtype=np.int32))
    with pytest.raises(sbci.SbciError):
        sbci.load(folder / "sub-02_fc.h5")
    cohort = load_cohort(folder, modalities="fc")
    assert cohort.subjects == ["sub-01"]
    assert "carries streamline endpoints" in cohort.excluded["sub-02"]


@pytest.mark.parametrize("report", ["qc.txt", "a folder", "no such folder/qc.csv"])
def test_a_report_that_cannot_be_written_is_refused_before_the_scan(
    folder, tmp_path, capsys, report
):
    """A refused cohort with such a report crashed with a traceback after the whole scan."""
    from sbci.cli import main

    root, add = folder
    add("sub-01_sc.h5", _header(bandwidth=0.005))
    add("sub-02_sc.h5", _header(bandwidth=0.01))  # refused: not all made the same way
    target = tmp_path / report
    if report == "a folder":
        target.mkdir()
    assert main(["cohort", str(root), "--report", str(target)]) == 1
    printed = capsys.readouterr().err
    assert printed.startswith("error: --report") and "not all made the same way" not in printed


# --- the fourth review of 12 October 2026 ------------------------------------------------


def test_a_quote_left_open_in_a_table_is_refused_not_swallowing_the_rows_after_it(folder, tmp_path):
    """sub-02 and sub-03 were read into sub-01's site, and left out as not in the table."""
    root, add = folder
    for subject in ("01", "02", "03"):
        add(f"sub-{subject}_sc.h5")
    table = tmp_path / "table.csv"
    table.write_text('subject,site\n01,"A\n02,B\n03,A\n')
    with pytest.raises(ValueError, match="table.csv, line 2: .*a quote left open"):
        load_cohort(root, table)
    closed = tmp_path / "closed.csv"
    closed.write_text('subject,site\n01,"A"\n02,"B, east"\n03,"say ""C"""\n')
    assert load_cohort(root, closed).column("site").tolist() == ["A", "B, east", 'say "C"']


def test_a_cell_past_the_csv_limit_is_a_value_error_and_the_command_says_so(
    folder, tmp_path, capsys
):
    """csv.Error is no ValueError: it escaped load_cohort, and sbci cohort as a traceback."""
    from sbci.cli import main

    root, add = folder
    add("sub-01_sc.h5")
    table = tmp_path / "notes.csv"
    table.write_text("subject,note\n01," + "x" * 200_000 + "\n")
    with pytest.raises(ValueError, match="notes.csv, line 2: field larger than field limit"):
        load_cohort(root, table)
    assert main(["cohort", str(root), "--table", str(table)]) == 1
    assert capsys.readouterr().err.startswith("error: notes.csv, line 2:")


def test_a_folder_or_a_table_may_be_any_path_like(folder, tmp_path):
    """A path-like other than a pathlib.Path was taken for a list: "object is not iterable"."""
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    table = _table(tmp_path / "ages.csv", [["subject", "age"], ["01", "25"], ["02", "31"]])

    class Named:
        def __init__(self, path):
            self.path = str(path)

        def __fspath__(self):
            return self.path

    cohort = load_cohort(Named(root), Named(table))
    assert cohort.subjects == ["sub-01", "sub-02"] and cohort.column("age").tolist() == [25, 31]
    assert load_cohort(str(root)).subjects == ["sub-01", "sub-02"]


def test_a_read_only_report_is_refused_before_the_scan(folder, tmp_path, capsys, monkeypatch):
    """Only the report's folder was checked first: a read-only file failed after the whole scan."""
    import stat

    from sbci.cli import main

    root, add = folder
    add("sub-01_sc.h5")
    report = tmp_path / "qc.csv"
    report.write_text("kept\n")
    report.chmod(stat.S_IRUSR)

    def scanned(*args, **kwargs):
        raise AssertionError("the files were scanned before the report was checked")

    monkeypatch.setattr(cohort_module, "load_cohort", scanned)
    assert main(["cohort", str(root), "--report", str(report)]) == 1
    assert capsys.readouterr().err.startswith(f"error: --report: {report} is read-only")
    assert report.read_text() == "kept\n"


def _altered(source, folder, name, change):
    """A copy of ``source`` named ``name`` in ``folder``, with ``change(handle)`` applied."""
    import shutil

    import h5py

    folder.mkdir(exist_ok=True)
    path = folder / name
    shutil.copy(source, path)
    with h5py.File(path, "r+") as handle:
        change(handle)
    return path


def _stored_as(name, values):
    def change(handle):
        del handle[name]
        handle.create_dataset(name, data=values, compression="gzip")

    return change


def test_arrays_that_do_not_hold_numbers_are_refused_by_load_and_the_loader_alike(
    written, tmp_path
):
    """Text coordinates passed the loader, which load refused; the other way round, too.

    load took connectivity stored as numeric text, or as complex values with
    their imaginary part dropped and a warning, which the loader refused; and a
    compound endpoint dataset raised a TypeError out of load.
    """
    n_upper = grid.condensed_size(N)
    fc = sbci.example("fc").save(tmp_path / "example_fc.h5")
    files = tmp_path / "files"
    _altered(fc, files, "sub-01_fc.h5", lambda handle: None)
    altered = [
        _altered(fc, files, "sub-02_fc.h5", _stored_as("coordinates", np.full((N, 3), b"abc"))),
        _altered(fc, files, "sub-03_fc.h5", _stored_as("connectivity", np.full(n_upper, b"0"))),
        _altered(
            fc, files, "sub-04_fc.h5", _stored_as("connectivity", np.zeros(n_upper, np.complex64))
        ),
    ]
    for path in altered:
        with pytest.raises(sbci.SbciError, match="not numbers"):
            sbci.load(path)
    cohort = load_cohort(files, modalities="fc")
    assert cohort.subjects == ["sub-01"]
    assert all("does not hold numbers" in cohort.excluded[f"sub-0{k}"] for k in (2, 3, 4))

    def compound(handle):
        shape = handle["endpoints/barycentric_in"].shape
        del handle["endpoints/barycentric_in"]
        handle["endpoints"].create_dataset(
            "barycentric_in", data=np.zeros(shape, dtype=[("w", "f4"), ("v", "f4")])
        )

    sc = _altered(
        os.path.realpath(written / "sub-a_sc.h5"), tmp_path / "sc", "sub-z_sc.h5", compound
    )
    with pytest.raises(sbci.SbciError, match="barycentric_in holds"):
        sbci.load(sc)
    cohort = load_cohort([written / "sub-a_sc.h5", sc])
    assert cohort.subjects == ["sub-a"] and "does not hold numbers" in cohort.excluded["sub-z"]


def test_an_empty_endpoint_group_is_left_out_as_load_refuses_it(written, tmp_path):
    """No streamlines in the group: load, validate and the loader took it; smooth() then failed."""

    def emptied(handle):
        group = handle["endpoints"]
        for name in list(group):
            shape, dtype = group[name].shape, group[name].dtype
            del group[name]
            group.create_dataset(name, data=np.zeros((0,) + shape[1:], dtype=dtype))

    path = _altered(
        os.path.realpath(written / "sub-a_sc.h5"), tmp_path / "sc", "sub-z_sc.h5", emptied
    )
    with pytest.raises(sbci.errors.MissingDataError, match="holds no streamlines"):
        sbci.load(path)
    cohort = load_cohort([written / "sub-a_sc.h5", path])
    assert cohort.subjects == ["sub-a"] and "holds no streamlines" in cohort.excluded["sub-z"]


# --- PORTING.md item 38 -------------------------------------------------------------------


def test_the_summary_counts_a_reason_holding_a_semicolon_once(folder):
    """Split on "; ", one reason of two subjects was counted as two reasons of two each."""
    root, add = folder
    for subject in ("01", "02", "03", "04"):
        add(f"sub-{subject}_sc.h5")
    for subject in ("05", "06"):
        add(f"sub-{subject}_sc.h5", _header(kernel="rdk", bandwidth=0.01))
    cohort = load_cohort(root, mismatch="exclude", exclude={"01": "motion"})
    left = [line for line in cohort.summary().splitlines() if line.startswith("  left out")]
    assert left == [
        "  left out, 2: sc kernel is 'rdk', not the cohort's 'shk'; sc bandwidth is 0.01, not "
        "the cohort's 0.005",
        "  left out, 1: excluded by the caller: motion",
    ]
    assert cohort.reasons["sub-05"] == [cohort.excluded["sub-05"]]


def test_empty_columns_without_a_name_are_dropped(folder, tmp_path):
    """subject,age,, was refused as two columns named '', and subject,age, read a covariate ''."""
    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    for blank in (1, 2):
        rows = [["subject", "age", *[""] * blank], ["01", "25", *[""] * blank], ["02", "31"]]
        cohort = load_cohort(root, _table(tmp_path / f"blank{blank}.csv", rows))
        assert list(cohort.covariates) == ["age"] and cohort.column("age").tolist() == [25.0, 31.0]
    unnamed = _table(tmp_path / "unnamed.csv", [["subject", "age", ""], ["01", "25", "x"]])
    with pytest.raises(ValueError, match="unnamed.csv: column 3 holds values but has no name"):
        load_cohort(root, unnamed)


def test_a_table_that_is_not_utf8_is_named_and_a_bom_is_read(folder, tmp_path, capsys):
    """A table saved as cp1252 raised a bare UnicodeDecodeError that named no file."""
    from sbci.cli import main

    root, add = folder
    for subject in ("01", "02"):
        add(f"sub-{subject}_sc.h5")
    text = "subject,site\n01,Malmö\n02,Zürich\n"
    windows = tmp_path / "sites.csv"
    windows.write_bytes(text.encode("cp1252"))
    with pytest.raises(ValueError, match="sites.csv, line 2: not UTF-8 .*Save it as UTF-8") as no:
        load_cohort(root, windows)
    assert type(no.value) is ValueError
    assert main(["cohort", str(root), "--table", str(windows)]) == 1
    assert capsys.readouterr().err.startswith("error: sites.csv, line 2: not UTF-8 text")
    marked = tmp_path / "marked.csv"
    marked.write_bytes(text.encode("utf-8-sig"))
    assert load_cohort(root, marked).column("site").tolist() == ["Malmö", "Zürich"]


def test_the_sub_prefix_is_read_in_any_case(folder):
    """SUB-01_sc.h5 was not read, and subject_id("SUB-01") was 'sub-SUB-01'."""
    root, add = folder
    add("SUB-01_sc.h5")
    add("Sub-02_SC.H5")
    assert subject_id("SUB-01") == "sub-01" and parse_name("SUB-01_sc.h5")[0] == "sub-01"
    cohort = load_cohort(root, {"subject": ["SUB-01", "sub-02"]})
    assert cohort.subjects == ["sub-01", "sub-02"]


def test_a_hyphen_in_an_id_is_not_taken_for_a_key(folder):
    """sub-NDAR_INV-1_sc.h5 was read as subject sub-NDAR, with INV-1 a key and its value."""
    root, add = folder
    add("sub-NDAR_INV-1_sc.h5")
    add("sub-NDAR_INV-2_ses-1_sc.h5")
    assert parse_name("sub-NDAR_INV-1_sc.h5") == ("sub-NDAR_INV-1", None, "sc")
    assert parse_name("sub-NDAR_INV-2_ses-1_sc.h5") == ("sub-NDAR_INV-2", "1", "sc")
    assert load_cohort(root).subjects == ["sub-NDAR_INV-1", "sub-NDAR_INV-2"]


def test_two_files_of_one_name_are_told_apart_by_their_folders(folder):
    """Listed by name alone, they read 2 sc files: sub-01_sc.h5, sub-01_sc.h5."""
    root, add = folder
    add("sub-01_sc.h5", where="visit-a")
    add("sub-01_sc.h5", where="visit-b")
    add("sub-02_sc.h5")
    assert load_cohort(root).excluded == {
        "sub-01": "2 sc files: visit-a/sub-01_sc.h5, visit-b/sub-01_sc.h5"
    }
