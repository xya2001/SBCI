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


def test_only_the_known_cohorts_have_manifests():
    with pytest.raises(ValueError, match="cohort"):
        download.load_manifest("hcp-ya")
    assert download.load_manifest("hcp-aging-full")["host"] == "zenodo"


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


def _bundled_manifest(tmp_path, monkeypatch, payloads, per_bundle=2):
    """A Zenodo-style manifest: zip bundles per modality, served by a fake fetcher."""
    import zipfile

    subjects = {}
    for subject, _modality in payloads:
        subjects.setdefault(
            subject, {"subject": subject, "sex": "M", "age_bin": "41-45", "files": {}}
        )
    ordered = sorted(subjects)
    bundles, served = [], {}
    for modality in ("sc", "fc"):
        for index, start in enumerate(range(0, len(ordered), per_bundle), start=1):
            name = f"hcp-aging-full_{modality}_{index:02d}.zip"
            archive = tmp_path / "served" / name
            archive.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive, "w") as z:
                for subject in ordered[start : start + per_bundle]:
                    data = payloads.get((subject, modality))
                    if data is None:
                        continue
                    z.writestr(f"{subject}_{modality}.h5", data)
                    subjects[subject]["files"][modality] = {
                        "bundle": name,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
            url = f"https://zenodo.example/{name}"
            bundles.append(
                {
                    "name": name,
                    "url": url,
                    "bytes": archive.stat().st_size,
                    "sha256": download.sha256_of(archive),
                }
            )
            served[url] = archive.read_bytes()
    manifest = {
        "cohort": "hcp-aging-full",
        "host": "zenodo",
        "record": "1",
        "bundles": bundles,
        "subjects": [subjects[s] for s in ordered],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-aging": manifest)
    calls = []

    def fetcher(url, destination, report):
        calls.append(url)
        Path(destination).write_bytes(served[url])

    return fetcher, calls


def test_the_full_cohort_comes_in_bundles_that_are_fetched_once(tmp_path, monkeypatch):
    payloads = {
        ("sub-A", "sc"): b"A-sc" * 50,
        ("sub-A", "fc"): b"A-fc",
        ("sub-B", "sc"): b"B-sc",
        ("sub-B", "fc"): b"B-fc",
        ("sub-C", "sc"): b"C-sc",
        ("sub-C", "fc"): b"C-fc",
    }
    fetcher, calls = _bundled_manifest(tmp_path, monkeypatch, payloads)
    out = tmp_path / "out"
    written = download.fetch_cohort(
        out,
        cohort="hcp-aging-full",
        subjects=["sub-A", "sub-B"],
        modalities=("sc",),
        fetcher=fetcher,
        report=lambda _: None,
    )
    assert [p.name for p in written] == ["sub-A_sc.h5", "sub-B_sc.h5"]
    assert (out / "sub-A_sc.h5").read_bytes() == b"A-sc" * 50
    assert len(calls) == 1 and calls[0].endswith("_sc_01.zip")  # one bundle held both
    assert not list(out.glob("*.zip"))  # removed after extraction
    assert (out / "manifest.csv").read_text().splitlines()[1] == "sub-A,M,41-45"
    # a re-run finds the files and fetches nothing; sub-C needs the second bundle
    download.fetch_cohort(
        out,
        cohort="hcp-aging-full",
        modalities=("sc",),
        fetcher=fetcher,
        report=lambda _: None,
        keep_bundles=True,
    )
    assert len(calls) == 2 and calls[1].endswith("_sc_02.zip")
    assert (out / "hcp-aging-full_sc_02.zip").exists()


def test_a_corrupt_bundle_is_removed_and_reported(tmp_path, monkeypatch):
    fetcher, _ = _bundled_manifest(tmp_path, monkeypatch, {("sub-A", "sc"): b"x" * 10})

    def corrupting(url, destination, report):
        fetcher(url, destination, report)
        Path(destination).write_bytes(b"not a zip")

    with pytest.raises(OSError, match="does not match"):
        download.fetch_cohort(
            tmp_path / "o", cohort="hcp-aging-full", fetcher=corrupting, report=lambda _: None
        )
    assert not list((tmp_path / "o").glob("*.zip"))


def test_an_unreleased_full_cohort_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="not been released"):
        download.fetch_cohort(tmp_path, cohort="hcp-aging-full", report=lambda _: None)


def test_https_fetch_resumes_a_partial_download(tmp_path):
    """A local server that honours Range, a partial file, and the fetch completing it."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = bytes(range(256)) * 40  # 10,240 bytes

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = 0
            header = self.headers.get("Range")
            if header:
                start = int(header.split("=")[1].rstrip("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/bundle.zip"
        destination = tmp_path / "bundle.zip"
        partial = tmp_path / "bundle.zip.part"
        partial.write_bytes(body[:3000])
        lines = []
        download._fetch_https(url, destination, lines.append)
        assert destination.read_bytes() == body and not partial.exists()
        # and from nothing
        destination.unlink()
        download._fetch_https(url, destination, lines.append)
        assert destination.read_bytes() == body
    finally:
        server.shutdown()
