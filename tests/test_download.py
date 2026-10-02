"""Fetching the example cohort: the manifest drives it, digests are checked, re-runs are cheap."""

import hashlib
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
    manifest = {"cohort": "hcp-ya", "host": "google-drive", "subjects": subjects}
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
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


def test_a_bare_hcp_id_names_the_same_subject(tmp_path, monkeypatch):
    fetcher, calls = _fake_manifest(tmp_path, monkeypatch, {("sub-100307", "sc"): b"x"})
    for subjects in (["100307"], ["sub-100307"], "100307"):
        written = download.fetch_cohort(
            tmp_path, subjects=subjects, fetcher=fetcher, report=lambda _: None, force=True
        )
        assert [p.name for p in written] == ["sub-100307_sc.h5"]


def test_subjects_and_modalities_are_checked(tmp_path, monkeypatch):
    fetcher, _ = _fake_manifest(tmp_path, monkeypatch, {("sub-A", "sc"): b"x"})
    with pytest.raises(ValueError, match="unknown subjects"):
        download.fetch_cohort(tmp_path, subjects=["sub-Z"], fetcher=fetcher, report=lambda _: None)
    with pytest.raises(ValueError, match="modalities"):
        download.fetch_cohort(tmp_path, modalities=("dwi",), fetcher=fetcher, report=lambda _: None)


def test_asking_only_for_what_the_cohort_lacks_says_so(tmp_path, monkeypatch, capsys):
    # A cohort that holds only SC: asking for FC alone says so rather than fetch nothing.
    data = b"sc"
    record = {"drive_id": "x", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "cohort": "hcp-ya",
        "host": "google-drive",
        "subjects": [{"subject": "sub-A", "files": {"sc": record}}],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
    out = tmp_path / "hcp-ya"
    with pytest.raises(ValueError, match="hcp-ya cohort has no fc files; it holds sc"):
        download.fetch_cohort(out, cohort="hcp-ya", modalities=("fc",), report=lambda _: None)
    assert main(["download", "hcp-ya", "--fc-only", "--out", str(out)]) == 1
    assert "has no fc files" in capsys.readouterr().err
    assert not out.exists()  # nothing fetched, not even the directory made


def test_the_terms_of_use_travel_with_the_files(tmp_path, monkeypatch):
    shipped = download.load_manifest("hcp-ya")["data_use"]
    assert "wu-minn-hcp-consortium-open-access-data-use-terms" in shipped
    assert "1U54MH091657" in shipped  # the acknowledgment a publication must carry
    data = b"sc"
    record = {"drive_id": "x", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "cohort": "hcp-ya",
        "host": "google-drive",
        "data_use": shipped,
        "subjects": [{"subject": "sub-A", "files": {"sc": record}}],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
    lines = []
    download.fetch_cohort(
        tmp_path, fetcher=lambda _i, d, _r: Path(d).write_bytes(data), report=lines.append
    )
    assert (tmp_path / "DATA_USE.txt").read_text(encoding="utf-8") == shipped
    assert any("DATA_USE.txt" in line for line in lines)


def test_only_the_known_cohorts_have_manifests():
    for unknown in ("hcp-development", "hcp-aging", "hcp-aging-full"):
        with pytest.raises(ValueError, match="cohort"):
            download.load_manifest(unknown)
    assert download.load_manifest("hcp-ya")["cohort"] == "hcp-ya"


def test_the_shipped_manifest_lists_eleven_subjects_with_bands_not_ages():
    manifest = download.load_manifest()
    assert manifest["cohort"] == "hcp-ya" and len(manifest["subjects"]) == 11
    assert manifest["subjects"][0]["subject"] == "sub-100307"  # the brief's tutorial subject
    assert sorted(e["sex"] for e in manifest["subjects"]) == ["F"] * 6 + ["M"] * 5
    for entry in manifest["subjects"]:
        assert set(entry["files"]) == {"sc", "fc"}  # every subject has both
        assert entry["age_bin"] in ("22-25", "26-30", "31-35", "36+") and "age" not in entry
        for record in entry["files"].values():
            assert record["drive_id"] and record["bytes"] > 0 and len(record["sha256"]) == 64


def test_cli_download_explains_an_unreleased_manifest(tmp_path, capsys, monkeypatch):
    """A manifest without Drive ids is a cohort not yet released, and the command says so."""
    manifest = {
        "cohort": "hcp-ya",
        "subjects": [
            {
                "subject": "sub-A",
                "sex": "F",
                "age_bin": "22-25",
                "files": {"sc": {"drive_id": "", "bytes": 0, "sha256": ""}},
            }
        ],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
    assert main(["download", "hcp-ya", "--out", str(tmp_path)]) == 1
    assert "not been released" in capsys.readouterr().err


def test_drive_url_carries_the_confirm_and_uuid():
    assert download.drive_url("abc") == (
        "https://drive.usercontent.google.com/download?id=abc&export=download&confirm=t"
    )
    assert download.drive_url("abc", "u1").endswith("&uuid=u1")


def test_cli_download_runs_the_fetch(tmp_path, monkeypatch):
    payloads = {("sub-A", "sc"): b"sc"}
    fetcher, calls = _fake_manifest(tmp_path, monkeypatch, payloads)
    monkeypatch.setattr(download, "_fetch_drive", fetcher)
    assert main(["download", "hcp-ya", "--out", str(tmp_path), "--sc-only"]) == 0
    assert calls == ["id-sub-A-sc"]
    assert main(["download", "hcp-ya", "--subject", "sub-Q", "--out", str(tmp_path)]) == 1


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
            name = f"hcp-ya-full_{modality}_{index:02d}.zip"
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
        "cohort": "hcp-ya-full",
        "host": "zenodo",
        "record": "1",
        "bundles": bundles,
        "subjects": [subjects[s] for s in ordered],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
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
        cohort="hcp-ya-full",
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
        cohort="hcp-ya-full",
        modalities=("sc",),
        fetcher=fetcher,
        report=lambda _: None,
        keep_bundles=True,
    )
    assert len(calls) == 2 and calls[1].endswith("_sc_02.zip")
    assert (out / "hcp-ya-full_sc_02.zip").exists()


def test_a_corrupt_bundle_is_removed_and_reported(tmp_path, monkeypatch):
    fetcher, _ = _bundled_manifest(tmp_path, monkeypatch, {("sub-A", "sc"): b"x" * 10})

    def corrupting(url, destination, report):
        fetcher(url, destination, report)
        Path(destination).write_bytes(b"not a zip")

    with pytest.raises(OSError, match="does not match"):
        download.fetch_cohort(
            tmp_path / "o", cohort="hcp-ya-full", fetcher=corrupting, report=lambda _: None
        )
    assert not list((tmp_path / "o").glob("*.zip"))


def test_an_unreleased_cohort_says_so(tmp_path, monkeypatch):
    for manifest in (
        {"cohort": "hcp-ya", "host": "google-drive", "subjects": []},
        {
            "cohort": "hcp-ya",
            "host": "zenodo",
            "record": "",
            "bundles": [],
            "subjects": [{"subject": "sub-A", "files": {}}],
        },
    ):
        monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya", m=manifest: m)
        with pytest.raises(RuntimeError, match="not been released"):
            download.fetch_cohort(tmp_path, report=lambda _: None)


def test_a_cohort_without_demographics_gets_a_subject_only_manifest(tmp_path, monkeypatch):
    data = b"young adult"
    manifest = {
        "cohort": "hcp-ya",
        "host": "google-drive",
        "subjects": [
            {
                "subject": "sub-100206",
                "files": {
                    "sc": {
                        "drive_id": "id-1",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                },
            }
        ],
    }
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)

    def fetcher(file_id, destination, report):
        Path(destination).write_bytes(data)

    written = download.fetch_cohort(
        tmp_path, cohort="hcp-ya", fetcher=fetcher, report=lambda _: None
    )
    assert [p.name for p in written] == ["sub-100206_sc.h5"]
    assert (tmp_path / "manifest.csv").read_text().splitlines() == ["subject", "sub-100206"]


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
