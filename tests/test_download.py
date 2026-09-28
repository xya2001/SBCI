"""Fetching the example cohort: the manifest drives it, digests are checked, re-runs are cheap."""

import hashlib
import json
from pathlib import Path

import pytest

from sbci import download
from sbci.cli import main


def _fake_manifest(tmp_path, monkeypatch, payloads):
    """A manifest whose files are ``payloads[(subject, modality)]`` bytes, from a fake fetcher."""
    subjects = []
    for (subject, modality), data in payloads.items():
        entry = next((s for s in subjects if s["subject"] == subject), None)
        if entry is None:
            entry = {"subject": subject, "sex": "F", "age_bin": "36-40", "files": {}}
            subjects.append(entry)
        entry["files"][modality] = {
            "drive_id": f"id-{subject}-{modality}",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = {"cohort": "hcp-aging", "host": "google-drive", "subjects": subjects}
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-aging": manifest)
    served = {f"id-{s}-{m}": data for (s, m), data in payloads.items()}
    calls = []

    def fetcher(file_id, destination, report):
        calls.append(file_id)
        Path(destination).write_bytes(served[file_id])

    return fetcher, calls


def test_fetches_verifies_and_skips_on_rerun(tmp_path, monkeypatch):
    payloads = {("sub-A", "sc"): b"sc-A" * 100, ("sub-A", "fc"): b"fc-A", ("sub-B", "sc"): b"sc-B"}
    fetcher, calls = _fake_manifest(tmp_path, monkeypatch, payloads)
    written = download.fetch_cohort(tmp_path, fetcher=fetcher, report=lambda _: None)
    assert [p.name for p in written] == ["sub-A_sc.h5", "sub-A_fc.h5", "sub-B_sc.h5"]
    assert (tmp_path / "sub-A_sc.h5").read_bytes() == b"sc-A" * 100
    assert (tmp_path / "manifest.csv").read_text().splitlines()[0] == "subject,sex,age_bin"
    assert len(calls) == 3
    download.fetch_cohort(tmp_path, fetcher=fetcher, report=lambda _: None)
    assert len(calls) == 3  # nothing re-fetched


def test_a_wrong_digest_is_removed_and_reported(tmp_path, monkeypatch):
    payloads = {("sub-A", "sc"): b"good"}
    fetcher, _ = _fake_manifest(tmp_path, monkeypatch, payloads)

    def corrupting(file_id, destination, report):
        Path(destination).write_bytes(b"bad")

    with pytest.raises(OSError, match="SHA-256"):
        download.fetch_cohort(
            tmp_path, modalities=("sc",), fetcher=corrupting, report=lambda _: None
        )
    assert not (tmp_path / "sub-A_sc.h5").exists()


def test_subjects_and_modalities_are_checked(tmp_path, monkeypatch):
    fetcher, _ = _fake_manifest(tmp_path, monkeypatch, {("sub-A", "sc"): b"x"})
    with pytest.raises(ValueError, match="unknown subjects"):
        download.fetch_cohort(tmp_path, subjects=["sub-Z"], fetcher=fetcher, report=lambda _: None)
    with pytest.raises(ValueError, match="modalities"):
        download.fetch_cohort(tmp_path, modalities=("dwi",), fetcher=fetcher, report=lambda _: None)


def test_only_the_known_cohort_has_a_manifest():
    with pytest.raises(ValueError, match="cohort"):
        download.load_manifest("hcp-ya")


def test_the_shipped_manifest_lists_ten_subjects_with_bins_not_ages():
    manifest = download.load_manifest()
    assert len(manifest["subjects"]) == 10
    for entry in manifest["subjects"]:
        assert set(entry["files"]) == {"sc", "fc"}
        assert entry["sex"] in ("F", "M")
        assert "-" in entry["age_bin"] and "age" not in entry


def test_an_unreleased_manifest_says_so(tmp_path, monkeypatch):
    manifest = json.loads(json.dumps(download.load_manifest()))
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-aging": manifest)
    if any(f["drive_id"] for s in manifest["subjects"] for f in s["files"].values()):
        pytest.skip("the cohort is released; nothing unreleased to test")
    with pytest.raises(RuntimeError, match="not been released"):
        download.fetch_cohort(tmp_path, report=lambda _: None)


def test_drive_url_carries_the_confirm_and_uuid():
    assert download.drive_url("abc") == (
        "https://drive.usercontent.google.com/download?id=abc&export=download&confirm=t"
    )
    assert download.drive_url("abc", "u1").endswith("&uuid=u1")


def test_cli_download_runs_the_fetch(tmp_path, monkeypatch):
    payloads = {("sub-A", "sc"): b"sc"}
    fetcher, calls = _fake_manifest(tmp_path, monkeypatch, payloads)
    monkeypatch.setattr(download, "_fetch_drive", fetcher)
    assert main(["download", "hcp-aging", "--out", str(tmp_path), "--sc-only"]) == 0
    assert calls == ["id-sub-A-sc"]
    assert main(["download", "hcp-aging", "--subject", "sub-Q", "--out", str(tmp_path)]) == 1
