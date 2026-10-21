"""Fetching the HCP Young Adult connectomes from UNC Dataverse.

The index drives it, digests are checked, re-runs are cheap, and the terms are accepted once.
The transfer is exercised against a repository on this machine, serving tar archives built
here as the deposit builds them: byte ranges through a redirect, whole archives, the
guestbook of Dataverse 6.10, and connections cut part way. Nothing leaves the machine.
"""

import hashlib
import io
import json
import tarfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sbci import download
from sbci.cli import main

#: The real fetch of the basis, which conftest.py replaces in every test with a refusal.
FETCH_BASIS = download.fetch_basis

TERMS = "These files come under the WU-Minn HCP Consortium Open Access Data Use Terms.\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    """Every test keeps what the package remembers in its own folder, never in ~/.config."""
    folder = tmp_path / "config"
    monkeypatch.setenv("SBCI_CONFIG_DIR", str(folder))
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("no_proxy", "*")  # straight to the local server, past any proxy set
    return folder


def release(files, tutorial=("sub-A",), server="http://127.0.0.1:1"):
    """An index and its archives for ``files[(subject, modality)] = bytes``.

    The tutorial's subjects go in ``sc_tutorial.tar`` and ``fc_tutorial.tar``, the rest in
    ``sc_01.tar`` and ``fc_01.tar``, written as the deposit writes its archives.
    """
    groups = {}
    for (subject, modality), data in sorted(files.items()):
        label = "tutorial" if subject in tutorial else "01"
        member = f"{modality}/{subject}_{modality}.h5"
        groups.setdefault(f"{modality}_{label}.tar", []).append((member, data))
    blobs, archives, entries = {}, {}, {}
    for number, (name, members) in enumerate(sorted(groups.items())):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for member, data in members:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        blob = buffer.getvalue()
        blobs[name] = blob
        archives[name] = {"id": 100 + number, "bytes": len(blob), "sha256": sha(blob)}
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for info in tar:
                modality, base = info.name.split("/")
                subject = base.split("_")[0]
                entry = entries.setdefault(
                    subject,
                    {
                        "subject": subject,
                        "sex": "F",
                        "age_bin": "26-30",
                        "tutorial": subject in tutorial,
                        "files": {},
                    },
                )
                entry["files"][modality] = {
                    "archive": name,
                    "offset": info.offset_data,
                    "bytes": info.size,
                    "sha256": sha(files[(subject, modality)]),
                }
    manifest = {
        "cohort": "hcp-ya",
        "data_use": TERMS,
        "host": "dataverse",
        "release": {
            "server": server,
            "dataset": "doi:10.5072/FK2/TEST",
            "version": "1.0",
            "citation": "Zhang; Xiang, 2026, Test, https://doi.org/10.5072/FK2/TEST, V1",
        },
        "archives": archives,
        "subjects": [entries[s] for s in sorted(entries)],
    }
    return manifest, blobs


def use(monkeypatch, manifest, accepted=True):
    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": manifest)
    if accepted:
        download.accept_terms("hcp-ya")


def local_fetcher(blobs):
    """A fetcher reading the asked bytes straight out of the archives, and the calls made."""
    calls = []

    def fetcher(source, destination, report):
        calls.append((source["subject"], source["modality"]))
        start = int(source["offset"])
        Path(destination).write_bytes(blobs[source["archive"]][start : start + source["bytes"]])

    return fetcher, calls


FILES = {
    ("sub-A", "sc"): b"sc-A" * 300,
    ("sub-A", "fc"): b"fc-A" * 200,
    ("sub-B", "sc"): b"sc-B" * 150,
    ("sub-B", "fc"): b"fc-B" * 100,
}
QUIET = {"report": lambda _: None}
#: With a third subject beside sub-B in its archives, asking for sub-B alone comes as ranges.
MORE = {**FILES, ("sub-C", "sc"): b"sc-C" * 120, ("sub-C", "fc"): b"fc-C" * 80}


# --- choosing, verifying, re-running ------------------------------------------------


def test_fetches_the_tutorial_by_default_verifies_and_skips_on_rerun(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, calls = local_fetcher(blobs)
    out = tmp_path / "out"
    written = download.fetch_cohort(out, fetcher=fetcher, **QUIET)
    assert [p.name for p in written] == ["sub-A_sc.h5", "sub-A_fc.h5"]  # the tutorial's
    assert (out / "sub-A_sc.h5").read_bytes() == FILES[("sub-A", "sc")]
    assert (out / "manifest.csv").read_text().splitlines() == [
        "subject,sex,age_bin",
        "sub-A,F,26-30",
    ]
    download.fetch_cohort(out, fetcher=fetcher, **QUIET)
    assert len(calls) == 2  # nothing fetched again


def test_any_subject_by_id_and_all_of_them(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, calls = local_fetcher(blobs)
    out = tmp_path / "out"
    written = download.fetch_cohort(out, subjects="B", fetcher=fetcher, **QUIET)
    assert [p.name for p in written] == ["sub-B_sc.h5", "sub-B_fc.h5"]
    written = download.fetch_cohort(out, subjects="all", fetcher=fetcher, **QUIET)
    assert [p.name for p in written] == ["sub-A_sc.h5", "sub-A_fc.h5", "sub-B_sc.h5", "sub-B_fc.h5"]
    assert calls[2:] == [("sub-A", "sc"), ("sub-A", "fc")]  # sub-B's were here already
    assert (out / "manifest.csv").read_text().splitlines()[1:] == ["sub-A,F,26-30", "sub-B,F,26-30"]


def test_a_wrong_digest_is_removed_and_reported(tmp_path, monkeypatch):
    manifest, _ = release(FILES)
    use(monkeypatch, manifest)

    def corrupting(source, destination, report):
        Path(destination).write_bytes(b"bad")

    with pytest.raises(OSError, match="SHA-256"):
        download.fetch_cohort(tmp_path, modalities=("sc",), fetcher=corrupting, **QUIET)
    assert not (tmp_path / "sub-A_sc.h5").exists()
    assert not (tmp_path / "sub-A_sc.h5.part").exists()  # where the bytes actually went


def test_a_present_file_with_a_wrong_digest_is_fetched_again(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, calls = local_fetcher(blobs)
    (tmp_path / "sub-A_sc.h5").write_bytes(b"stale")
    download.fetch_cohort(tmp_path, modalities=("sc",), fetcher=fetcher, **QUIET)
    assert calls == [("sub-A", "sc")]
    assert (tmp_path / "sub-A_sc.h5").read_bytes() == FILES[("sub-A", "sc")]


def test_bare_hcp_ids_name_the_same_subject_as_strings_or_numbers(tmp_path, monkeypatch):
    manifest, blobs = release({("sub-100307", "sc"): b"x" * 10}, tutorial=("sub-100307",))
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)
    for subjects in (["100307"], ["sub-100307"], "100307", 100307, [100307], ["100307 "]):
        written = download.fetch_cohort(
            tmp_path, subjects=subjects, fetcher=fetcher, force=True, **QUIET
        )
        assert [p.name for p in written] == ["sub-100307_sc.h5"]


def test_subjects_and_modalities_are_checked(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)
    with pytest.raises(ValueError, match="unknown subjects"):
        download.fetch_cohort(tmp_path, subjects=["sub-Z"], fetcher=fetcher, **QUIET)
    with pytest.raises(ValueError, match="got 'dwi'"):
        download.fetch_cohort(tmp_path, modalities="dwi", fetcher=fetcher, **QUIET)


def test_a_single_modality_may_be_a_plain_string(tmp_path, monkeypatch):
    """``modalities="sc"`` is the one modality; read letter by letter it was ``("s", "c")``."""
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, calls = local_fetcher(blobs)
    assert [
        p.name for p in download.fetch_cohort(tmp_path, modalities="sc", fetcher=fetcher, **QUIET)
    ] == ["sub-A_sc.h5"]
    assert [
        p.name for p in download.fetch_cohort(tmp_path, modalities="fc", fetcher=fetcher, **QUIET)
    ] == ["sub-A_fc.h5"]
    assert calls == [("sub-A", "sc"), ("sub-A", "fc")]


def test_asking_only_for_what_the_cohort_lacks_says_so(tmp_path, monkeypatch, capsys):
    manifest, _ = release({("sub-A", "sc"): b"sc"})
    use(monkeypatch, manifest)
    out = tmp_path / "hcp-ya"
    with pytest.raises(ValueError, match="hcp-ya cohort has no fc files; it holds sc"):
        download.fetch_cohort(out, modalities=("fc",), **QUIET)
    assert main(["download", "hcp-ya", "--fc-only", "--out", str(out)]) == 1
    assert "has no fc files" in capsys.readouterr().err
    assert not out.exists()  # nothing fetched, not even the directory made


def test_an_index_without_places_is_a_cohort_not_released(tmp_path, monkeypatch, capsys):
    for manifest in (
        {"cohort": "hcp-ya", "subjects": []},
        {"cohort": "hcp-ya", "subjects": [{"subject": "sub-A", "files": {"sc": {"sha256": ""}}}]},
    ):
        use(monkeypatch, manifest, accepted=False)
        with pytest.raises(RuntimeError, match="not been released"):
            download.fetch_cohort(tmp_path, **QUIET)
        assert main(["download", "hcp-ya", "--out", str(tmp_path), "--accept-terms"]) == 1
        assert "not been released" in capsys.readouterr().err


def test_manifest_csv_lists_the_subjects_here_and_leaves_a_table_of_ones_own(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    for entry in manifest["subjects"]:
        del entry["sex"], entry["age_bin"]  # a cohort without demographics: subjects alone
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)
    download.fetch_cohort(tmp_path, fetcher=fetcher, **QUIET)
    assert (tmp_path / "manifest.csv").read_text().splitlines() == ["subject", "sub-A"]
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "manifest.csv").write_text("subject,score\nsub-B,3\n")
    lines = []
    download.fetch_cohort(mine, subjects="B", fetcher=fetcher, report=lines.append)
    assert (mine / "manifest.csv").read_text() == "subject,score\nsub-B,3\n"
    assert any("did not write" in line for line in lines)


def test_an_interrupted_fetch_leaves_nothing_under_the_name_and_force_keeps_a_good_copy(
    tmp_path, monkeypatch
):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)

    def cut_off(source, destination, report):
        Path(destination).write_bytes(b"sc")
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(ConnectionResetError):
        download.fetch_cohort(tmp_path, modalities=("sc",), fetcher=cut_off, **QUIET)
    assert not (tmp_path / "sub-A_sc.h5").exists()
    download.fetch_cohort(tmp_path, modalities=("sc",), fetcher=fetcher, **QUIET)
    with pytest.raises(ConnectionResetError):
        download.fetch_cohort(tmp_path, modalities=("sc",), force=True, fetcher=cut_off, **QUIET)
    assert (tmp_path / "sub-A_sc.h5").read_bytes() == FILES[("sub-A", "sc")]


# --- the terms ---------------------------------------------------------------------------


def test_downloading_needs_the_terms_accepted_once(tmp_path, monkeypatch, config):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest, accepted=False)
    fetcher, calls = local_fetcher(blobs)
    with pytest.raises(download.TermsNotAccepted, match="--accept-terms"):
        download.fetch_cohort(tmp_path / "out", fetcher=fetcher, **QUIET)
    assert not calls and not (tmp_path / "out").exists()
    download.fetch_cohort(tmp_path / "out", fetcher=fetcher, accept_terms=True, **QUIET)
    record = json.loads((config / download.TERMS_RECORD).read_text())
    assert record["hcp-ya"]["terms_sha256"] == sha(TERMS.encode())
    assert download.terms_accepted("hcp-ya")
    download.fetch_cohort(tmp_path / "again", fetcher=fetcher, **QUIET)  # not asked again
    notice = (tmp_path / "again" / "DATA_USE.txt").read_text()
    assert (
        notice.startswith(TERMS) and "version 1.0 of the dataset https://doi.org/10.5072" in notice
    )
    assert "Cite it as: Zhang; Xiang, 2026, Test" in notice


def test_new_terms_are_asked_for_again(tmp_path, monkeypatch):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    assert download.terms_accepted("hcp-ya")
    manifest["data_use"] = TERMS + "A new clause.\n"
    assert not download.terms_accepted("hcp-ya")


def test_the_command_asks_for_the_terms_in_a_terminal(tmp_path, monkeypatch, capsys):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest, accepted=False)
    fetcher, calls = local_fetcher(blobs)
    real = download.fetch_cohort
    monkeypatch.setattr(download, "fetch_cohort", lambda *a, **k: real(*a, **k, fetcher=fetcher))

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert main(["download", "hcp-ya", "--out", str(tmp_path)]) == 1
    assert "not accepted" in capsys.readouterr().err and not calls
    monkeypatch.setattr("builtins.input", lambda prompt="": " I agree ")
    assert main(["download", "hcp-ya", "--out", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert download.TERMS_URL in shown and "accepted, and recorded" in shown
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked twice"))
    assert main(["download", "hcp-ya", "--out", str(tmp_path / "again")]) == 0


def test_the_command_outside_a_terminal_needs_the_flag(tmp_path, monkeypatch, capsys):
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest, accepted=False)
    fetcher, calls = local_fetcher(blobs)
    real = download.fetch_cohort
    monkeypatch.setattr(download, "fetch_cohort", lambda *a, **k: real(*a, **k, fetcher=fetcher))

    class Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", Pipe())
    assert main(["download", "hcp-ya", "--out", str(tmp_path)]) == 1
    assert "--accept-terms" in capsys.readouterr().err and not calls
    # a wrong id is said first, before any word of the terms
    assert main(["download", "hcp-ya", "--subject", "sub-Q", "--out", str(tmp_path)]) == 1
    assert "unknown subjects" in capsys.readouterr().err
    assert main(["download", "hcp-ya", "--accept-terms", "--out", str(tmp_path)]) == 0
    assert calls == [("sub-A", "sc"), ("sub-A", "fc")]
    assert main(["download", "hcp-ya", "--all", "--sc-only", "--out", str(tmp_path)]) == 0
    assert calls[2:] == [("sub-B", "sc")]


# --- the shipped index -------------------------------------------------------------------


def test_only_the_known_cohorts_have_indexes():
    for unknown in ("hcp-development", "hcp-ya-full", "nonesuch"):
        with pytest.raises(ValueError, match="cohort"):
            download.load_manifest(unknown)
    assert download.load_manifest("hcp-ya")["cohort"] == "hcp-ya"


def test_the_shipped_index_holds_the_946_with_bands_not_ages():
    manifest = download.load_manifest()
    subjects = manifest["subjects"]
    assert manifest["cohort"] == "hcp-ya" and len(subjects) == 946
    assert manifest["release"]["dataset"] == "doi:10.15139/S3/NDAJYX"
    assert manifest["release"]["server"] == "https://dataverse.unc.edu"
    tutorial = [e for e in subjects if e["tutorial"]]
    assert len(tutorial) == 11 and "sub-100307" in {e["subject"] for e in tutorial}
    assert sorted(e["sex"] for e in tutorial) == ["F"] * 6 + ["M"] * 5
    archives = manifest["archives"]
    assert len(archives) == 40 and all(
        a["id"] > 0 and len(a["sha256"]) == 64 for a in archives.values()
    )
    for entry in subjects:
        assert set(entry["files"]) == {"sc", "fc"}  # every subject has both
        assert entry["age_bin"] in ("22-25", "26-30", "31-35", "36+") and "age" not in entry
        for record in entry["files"].values():
            archive = archives[record["archive"]]
            assert record["offset"] % 512 == 0 and record["bytes"] > 0
            assert record["offset"] + record["bytes"] <= archive["bytes"]
            assert len(record["sha256"]) == 64


def test_the_terms_travel_with_the_files():
    shipped = download.load_manifest("hcp-ya")["data_use"]
    assert "wu-minn-hcp-consortium-open-access-data-use-terms" in shipped
    assert "1U54MH091657" in shipped  # the acknowledgment a publication must carry
    assert download.TERMS_URL in download.TERMS_NOTICE


# --- the transfer, against a repository on this machine ----------------------------------


class Repository(BaseHTTPRequestHandler):
    """Dataverse's file access as UNC's answers it: a redirect to the store, which serves ranges.

    With ``guestbook`` set it answers as 6.10 does: 400 until the guestbook's response comes
    in a POST, which returns a signed address.
    """

    blobs: dict = {}
    requests: list = []
    answers: list = []
    redirect = True
    ignore_range = False
    guestbook = None
    refuse_signed = False
    single_use = False  # a signed address serves one request, as an expired one would not
    chunked = False  # bodies in chunks, so a cut inside one raises IncompleteRead
    signatures: list = []
    cut: list = []  # bytes to send before closing, one entry a store request

    def log_message(self, *args):
        """Keep the test output quiet."""

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _blob(self, file_id):
        return next((b for name, b in self.blobs.items() if self.ids[name] == file_id), None)

    def _serve(self, data):
        wanted = self.headers.get("Range")
        if wanted and not self.ignore_range:
            first, last = (int(v) for v in wanted.removeprefix("bytes=").split("-"))
            body, status = data[first : last + 1], 206
        else:
            body, status = data, 200
        self.send_response(status)
        self.send_header("Content-Type", "application/x-tar")
        if status == 206:
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        if self.chunked:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sent = body[: self.cut.pop(0)] if self.cut else body
            self.wfile.write(f"{len(body):x}\r\n".encode() + sent)
            if len(sent) == len(body):
                self.wfile.write(b"\r\n0\r\n\r\n")
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.cut:
            self.wfile.write(body[: self.cut.pop(0)])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def do_GET(self):
        """A file, a range of it, a redirect, a refusal, or a guestbook."""
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        Repository.requests.append((parsed.path, self.headers.get("Range")))
        if parsed.path.startswith("/api/access/datafile/"):
            file_id = int(parsed.path.rsplit("/", 1)[1])
            signature = query.get("signature", [None])[0]
            if self._blob(file_id) is None:
                self._json({"status": "ERROR", "message": "File not found for given id."}, 404)
            elif self.guestbook and signature is None:
                message = (
                    "You may not download this file without the required Guestbook response "
                    f"for guestbookID {self.guestbook['id']}."
                )
                self._json({"status": "ERROR", "message": message}, 400)
            elif self.guestbook and (self.refuse_signed or signature not in self.signatures):
                self._json({"status": "ERROR", "message": "Bad signed URL"}, 401)
            elif self.redirect:
                if self.single_use and signature:
                    self.signatures.remove(signature)
                self.send_response(302)
                self.send_header("Location", f"/store/{file_id}")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._serve(self._blob(file_id))
        elif parsed.path.startswith("/store/"):
            self._serve(self._blob(int(parsed.path.rsplit("/", 1)[1])))
        elif self.guestbook and parsed.path == f"/api/guestbooks/{self.guestbook['id']}":
            self._json({"status": "OK", "data": self.guestbook})
        else:
            self._json({"status": "ERROR"}, 404)

    def do_POST(self):
        """A guestbook answered: a signed address for the file."""
        parsed = urllib.parse.urlsplit(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Repository.answers.append(body)
        file_id = int(parsed.path.rsplit("/", 1)[1])
        port = self.server.server_address[1]
        Repository.signatures.append(f"s{len(Repository.answers)}")
        signature = Repository.signatures[-1]
        signed = f"http://127.0.0.1:{port}/api/access/datafile/{file_id}?signature={signature}"
        self._json({"status": "OK", "data": {"signedUrl": signed}})


@pytest.fixture
def repository(monkeypatch):
    Repository.requests, Repository.answers, Repository.cut = [], [], []
    Repository.redirect, Repository.ignore_range = True, False
    Repository.guestbook, Repository.refuse_signed = None, False
    Repository.single_use, Repository.signatures, Repository.chunked = False, [], False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Repository)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    manifest, blobs = release(MORE, server=url)
    Repository.blobs = blobs
    Repository.ids = {name: a["id"] for name, a in manifest["archives"].items()}
    use(monkeypatch, manifest)
    yield manifest
    httpd.shutdown()


def test_a_subject_comes_as_byte_ranges_through_the_redirect(tmp_path, repository):
    download.fetch_cohort(tmp_path, subjects="B", **QUIET)
    for modality in ("sc", "fc"):
        assert (tmp_path / f"sub-B_{modality}.h5").read_bytes() == FILES[("sub-B", modality)]
    stored = [(path, wanted) for path, wanted in Repository.requests if path.startswith("/store/")]
    record = repository["subjects"][1]["files"]["sc"]
    first = record["offset"]
    assert (
        f"/store/{Repository.ids['sc_01.tar']}",
        f"bytes={first}-{first + record['bytes'] - 1}",
    ) in stored


def test_the_tutorial_comes_as_whole_archives_split_as_they_pass(tmp_path, repository):
    written = download.fetch_cohort(tmp_path, **QUIET)
    assert [p.read_bytes() for p in written] == [FILES[("sub-A", "sc")], FILES[("sub-A", "fc")]]
    access = [wanted for path, wanted in Repository.requests if path.startswith("/api/access")]
    assert access == [None, None]  # one request an archive, no ranges
    assert not list(tmp_path.glob("*.part"))


def test_a_server_that_sends_all_of_an_archive_is_refused(tmp_path, repository):
    Repository.ignore_range = True
    with pytest.raises(OSError, match="does not serve parts of files"):
        download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    assert not (tmp_path / "sub-B_sc.h5").exists()


def test_a_cut_transfer_is_taken_up_where_it_stopped(tmp_path, repository):
    Repository.cut = [100, 200]  # two cuts, then the rest
    download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    assert (tmp_path / "sub-B_sc.h5").read_bytes() == FILES[("sub-B", "sc")]
    ranges = [w for p, w in Repository.requests if p.startswith("/store/")]
    starts = [int(w.removeprefix("bytes=").split("-")[0]) for w in ranges]
    assert starts[1] - starts[0] == 100 and starts[2] - starts[1] == 200


def test_a_cut_that_raises_is_tried_again(tmp_path, repository, capsys):
    """Cut inside a chunk, a response raises IncompleteRead, which is no OSError."""
    Repository.chunked, Repository.cut = True, [100]
    download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    assert (tmp_path / "sub-B_sc.h5").read_bytes() == FILES[("sub-B", "sc")]
    assert len([p for p, _ in Repository.requests if p.startswith("/store/")]) == 2
    Repository.cut = [100] * download.ATTEMPTS
    assert main(["download", "hcp-ya", "--subject", "B", "--fc-only", "--out", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert "broke 4 times" in error and "IncompleteRead" in error and "Traceback" not in error


def test_a_transfer_cut_every_time_is_an_error_and_keeps_its_part(tmp_path, repository, capsys):
    """A cut inside a response is http.client.IncompleteRead, no OSError: never a traceback."""
    Repository.cut = [10] * download.ATTEMPTS
    assert main(["download", "hcp-ya", "--subject", "B", "--sc-only", "--out", str(tmp_path)]) == 1
    assert "run the download again to take up where it stopped" in capsys.readouterr().err
    assert (tmp_path / "sub-B_sc.h5.part").stat().st_size == 10 * download.ATTEMPTS
    assert not (tmp_path / "sub-B_sc.h5").exists()
    download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)  # and later, the rest
    assert (tmp_path / "sub-B_sc.h5").read_bytes() == FILES[("sub-B", "sc")]


def test_a_whole_archive_cut_part_way_gives_the_rest_by_range(tmp_path, repository, monkeypatch):
    big = {("sub-A", "sc"): b"a" * 5000, ("sub-C", "sc"): b"c" * 5000}
    server = repository["release"]["server"]
    manifest, blobs = release(big, tutorial=("sub-A", "sub-C"), server=server)
    Repository.blobs = blobs
    Repository.ids = {name: a["id"] for name, a in manifest["archives"].items()}
    use(monkeypatch, manifest)
    Repository.cut = [512 + 5120 + 600]  # the first member and a little of the second
    lines = []
    written = download.fetch_cohort(tmp_path, report=lines.append)
    assert [p.read_bytes() for p in written] == [big[("sub-A", "sc")], big[("sub-C", "sc")]]
    assert any("after 1 of its 2 files; the rest come one by one" in line for line in lines)
    stored = [wanted for path, wanted in Repository.requests if path.startswith("/store/")]
    record = manifest["subjects"][1]["files"]["sc"]
    last = record["offset"] + record["bytes"] - 1
    assert stored[0] is None and stored[-1] == f"bytes={record['offset']}-{last}"
    assert not list(tmp_path.glob("*.part"))


def test_an_archive_that_ends_early_gives_the_rest_by_range(tmp_path, repository, monkeypatch):
    """A response cut cleanly between two files ends the stream without an error."""
    big = {("sub-A", "sc"): b"a" * 5000, ("sub-C", "sc"): b"c" * 5000}
    server = repository["release"]["server"]
    manifest, blobs = release(big, tutorial=("sub-A", "sub-C"), server=server)
    Repository.blobs = blobs
    Repository.ids = {name: a["id"] for name, a in manifest["archives"].items()}
    use(monkeypatch, manifest)
    Repository.cut = [512 + 5120]  # exactly the first member, its header and its blocks
    written = download.fetch_cohort(tmp_path, **QUIET)
    assert [p.read_bytes() for p in written] == [big[("sub-A", "sc")], big[("sub-C", "sc")]]


def test_a_guestbook_is_answered_i_agree_and_the_signed_address_used(tmp_path, repository):
    Repository.guestbook = {"id": 19, "customQuestions": [{"id": 20, "required": True}, {"id": 21}]}
    download.fetch_cohort(tmp_path, subjects="B", **QUIET)
    assert (tmp_path / "sub-B_fc.h5").read_bytes() == FILES[("sub-B", "fc")]
    assert Repository.answers[0] == {
        "guestbookResponse": {"answers": [{"id": 20, "value": "I agree"}]}
    }


def test_a_guestbook_asking_more_or_a_refused_signature_says_so(tmp_path, repository):
    Repository.guestbook = {"id": 19, "emailRequired": True, "customQuestions": []}
    with pytest.raises(OSError, match="asks each downloader for email"):
        download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    Repository.guestbook = {"id": 19, "customQuestions": [{"id": 20, "required": True}]}
    Repository.refuse_signed = True
    with pytest.raises(OSError, match="refused the download address it signed"):
        download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)


def test_a_signed_address_that_lapsed_is_signed_anew(tmp_path, repository):
    """The second try of a cut transfer finds its address spent, and asks for another."""
    Repository.guestbook = {"id": 19, "customQuestions": [{"id": 20, "required": True}]}
    Repository.single_use, Repository.cut = True, [100]
    download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    assert (tmp_path / "sub-B_sc.h5").read_bytes() == FILES[("sub-B", "sc")]
    assert len(Repository.answers) == 2


def test_a_refusal_by_the_server_is_not_taken_for_a_broken_connection(tmp_path, repository):
    Repository.ids = {name: number + 1000 for name, number in Repository.ids.items()}
    with pytest.raises(OSError, match="answered 404 for sc_01.tar"):
        download.fetch_cohort(tmp_path, subjects="B", modalities="sc", **QUIET)
    assert len(Repository.requests) == 1  # said at once, not tried again


# --- the listing -------------------------------------------------------------------------


def test_the_subjects_are_listed_with_their_sizes_and_nothing_fetched(
    tmp_path, monkeypatch, capsys
):
    manifest, _ = release(MORE)
    use(monkeypatch, manifest, accepted=False)  # a listing needs no terms: nothing comes

    class Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", Pipe())
    monkeypatch.chdir(tmp_path)
    assert main(["download", "hcp-ya", "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["subject", "sex", "age", "MB"]
    assert lines[1].split() == ["sub-A", "F", "26-30", "0.0", "tutorial"]
    assert lines[2].split() == ["sub-B", "F", "26-30", "0.0"]
    assert lines[-1].startswith("3 subjects, 0.0 GB of sc and fc;") and "--all" in lines[-1]
    assert not list(tmp_path.iterdir())
    assert main(["download", "hcp-ya", "--list", "--fc-only"]) == 0
    assert "GB of fc" in capsys.readouterr().out.splitlines()[-1]
    with pytest.raises(SystemExit):  # one or the other: a listing, or what to fetch
        main(["download", "hcp-ya", "--list", "--all"])


def test_the_listing_of_an_unreleased_cohort_says_so(monkeypatch, capsys):
    use(monkeypatch, {"cohort": "hcp-ya", "subjects": []}, accepted=False)
    assert main(["download", "hcp-ya", "--list"]) == 1
    assert "not been released" in capsys.readouterr().out


def test_the_shipped_listing_has_the_946():
    lines = []
    import contextlib
    import io as text_io

    with contextlib.redirect_stdout(text_io.StringIO()) as printed:
        assert main(["download", "hcp-ya", "--list"]) == 0
    lines = printed.getvalue().splitlines()
    assert len(lines) == 948 and sum(line.endswith("tutorial") for line in lines) == 11
    assert lines[-1].startswith("946 subjects, 85.1 GB of sc and fc")


# --- the Laplace-Beltrami basis ----------------------------------------------------------


class RawFiles(BaseHTTPRequestHandler):
    """Files by name, whole or by byte range, as raw.githubusercontent.com serves them."""

    files: dict = {}
    requests: list = []
    cut: list = []
    ignore_range = False

    def log_message(self, *args):
        """Keep the test output quiet."""

    def do_GET(self):
        """A file, or a range of it; 404 for a file it does not have."""
        name = self.path.rsplit("/", 1)[1]
        wanted = self.headers.get("Range")
        RawFiles.requests.append((self.path, wanted))
        data = self.files.get(name)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if wanted and not self.ignore_range:
            first, last = (int(v) for v in wanted.removeprefix("bytes=").split("-"))
            body, status = data[first : last + 1], 206
        else:
            body, status = data, 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        self.end_headers()
        if self.cut:
            self.wfile.write(body[: self.cut.pop(0)])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


BASIS = {"EV_LBO_ds_ico4_L.mat": b"left" * 2000, "EV_LBO_ds_ico4_R.mat": b"right" * 1500}


@pytest.fixture
def raw(monkeypatch):
    """A stand-in for GitHub's raw files, serving a basis of two small files."""
    RawFiles.files, RawFiles.requests, RawFiles.cut = dict(BASIS), [], []
    RawFiles.ignore_range = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RawFiles)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}/sbci-brain/SBCI_Toolkit/{{commit}}/concon_estimate/{{name}}"
    monkeypatch.setattr(download, "BASIS_URL", url)
    monkeypatch.setattr(
        download, "BASIS_FILES", {name: (len(data), sha(data)) for name, data in BASIS.items()}
    )
    monkeypatch.setattr(download, "fetch_basis", FETCH_BASIS)
    yield
    httpd.shutdown()


def test_the_basis_is_fetched_at_its_commit_verified_and_kept(tmp_path, raw):
    lines = []
    where = download.fetch_basis(tmp_path / "lbo", report=lines.append)
    assert where == tmp_path / "lbo"
    for name, data in BASIS.items():
        assert (where / name).read_bytes() == data
    assert [path.split("/")[3] for path, _ in RawFiles.requests] == [download.BASIS_COMMIT] * 2
    assert [wanted for _, wanted in RawFiles.requests] == [
        f"bytes=0-{len(data) - 1}" for data in BASIS.values()
    ]
    assert len(lines) == 2 and "commit a558fe8" in lines[0]
    download.fetch_basis(tmp_path / "lbo", report=lines.append)
    assert len(RawFiles.requests) == 2 and len(lines) == 2  # present and correct: kept
    download.fetch_basis(tmp_path / "lbo", force=True, report=lines.append)
    assert len(RawFiles.requests) == 4


def test_a_basis_file_with_another_digest_is_removed(tmp_path, raw, monkeypatch):
    name = "EV_LBO_ds_ico4_R.mat"
    files = dict(download.BASIS_FILES, **{name: (len(BASIS[name]), "0" * 64)})
    monkeypatch.setattr(download, "BASIS_FILES", files)
    with pytest.raises(OSError, match="EV_LBO_ds_ico4_R.mat: SHA-256"):
        download.fetch_basis(tmp_path, report=lambda _: None)
    assert not (tmp_path / name).exists() and not (tmp_path / f"{name}.part").exists()
    assert (tmp_path / "EV_LBO_ds_ico4_L.mat").read_bytes() == BASIS["EV_LBO_ds_ico4_L.mat"]


def test_a_cut_basis_transfer_is_taken_up_and_a_server_without_ranges_refused(tmp_path, raw):
    RawFiles.cut = [1000]
    download.fetch_basis(tmp_path / "a", report=lambda _: None)
    assert RawFiles.requests[1][1] == f"bytes=1000-{len(BASIS['EV_LBO_ds_ico4_L.mat']) - 1}"
    for name, data in BASIS.items():
        assert (tmp_path / "a" / name).read_bytes() == data
    RawFiles.ignore_range, RawFiles.cut = True, [1000]  # the whole file again, when taking up
    with pytest.raises(OSError, match="does not serve parts of files"):
        download.fetch_basis(tmp_path / "b", report=lambda _: None)
    RawFiles.cut = []
    download.fetch_basis(tmp_path / "c", report=lambda _: None)  # a whole file from 0 is fine
    assert (tmp_path / "c" / "EV_LBO_ds_ico4_L.mat").read_bytes() == BASIS["EV_LBO_ds_ico4_L.mat"]


def test_sbci_basis_fetches_and_says_where_smooth_looks(tmp_path, raw, monkeypatch, capsys):
    monkeypatch.setenv("SBCI_LBO_DIR", str(tmp_path / "lbo"))
    assert main(["basis"]) == 0
    out = capsys.readouterr().out
    assert f"the Laplace-Beltrami basis is in {tmp_path / 'lbo'}" in out
    assert "SBCI_LBO_DIR at" not in out  # where smooth() looks: nothing more to do
    assert main(["basis", "--out", str(tmp_path / "elsewhere")]) == 0
    assert "point SBCI_LBO_DIR at" in capsys.readouterr().out
    del RawFiles.files["EV_LBO_ds_ico4_R.mat"]
    assert main(["basis", "--out", str(tmp_path / "third"), "--force"]) == 1
    assert "answered 404 for EV_LBO_ds_ico4_R.mat" in capsys.readouterr().err


def test_the_suite_never_fetches_the_real_basis():
    """conftest.py's guard: outside the tests above, the fetch is refused."""
    with pytest.raises(OSError, match="do not fetch"):
        download.fetch_basis()


# --- PORTING.md item 38 -------------------------------------------------------------------


@pytest.mark.parametrize("text", ["[]", '"accepted"', '{"hcp-ya": "yes"}', "null"])
def test_a_terms_record_of_another_shape_is_terms_not_accepted(monkeypatch, config, text):
    """JSON, but not an object of objects: it escaped as an AttributeError or a TypeError."""
    manifest, _ = release(FILES)
    use(monkeypatch, manifest, accepted=False)
    config.mkdir()
    (config / download.TERMS_RECORD).write_text(text)
    assert not download.terms_accepted("hcp-ya")
    download.accept_terms("hcp-ya")
    assert download.terms_accepted("hcp-ya")


def test_a_download_that_fails_part_way_leaves_the_terms_beside_its_files(tmp_path, monkeypatch):
    """DATA_USE.txt was written once every file had come, so a failure left files without it."""
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest)
    fetcher, calls = local_fetcher(blobs)

    def second_fails(source, destination, report):
        if calls:
            raise ConnectionResetError("connection reset by peer")
        fetcher(source, destination, report)

    with pytest.raises(ConnectionResetError):
        download.fetch_cohort(tmp_path, fetcher=second_fails, **QUIET)
    assert (tmp_path / "sub-A_sc.h5").exists() and not (tmp_path / "sub-A_fc.h5").exists()
    assert (tmp_path / "DATA_USE.txt").read_text().startswith(TERMS)


def test_manifest_csv_lists_a_subject_only_for_a_file_the_index_knows(tmp_path, monkeypatch):
    """A file of a subject's name was enough: an empty one, or a stale one being replaced."""
    manifest, blobs = release(MORE)
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)
    download.fetch_cohort(tmp_path, modalities="sc", fetcher=fetcher, **QUIET)  # sub-A
    (tmp_path / "sub-B_sc.h5").write_bytes(b"")
    stale = tmp_path / "sub-C_sc.h5"
    stale.write_bytes(bytes(len(MORE[("sub-C", "sc")])))  # the size the index gives, not its bytes

    def cut_off(source, destination, report):
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(ConnectionResetError):
        download.fetch_cohort(tmp_path, subjects="C", modalities="sc", fetcher=cut_off, **QUIET)
    assert stale.exists()
    assert (tmp_path / "manifest.csv").read_text().splitlines()[1:] == ["sub-A,F,26-30"]


def test_one_id_may_be_any_whole_number_and_a_fraction_is_refused(tmp_path, monkeypatch):
    """np.int64(100307) and 100307.0 raised TypeErrors; [100307.0] named sub-100307.0 unknown."""
    import numpy as np

    manifest, blobs = release({("sub-100307", "sc"): b"x" * 10}, tutorial=("sub-100307",))
    use(monkeypatch, manifest)
    fetcher, _ = local_fetcher(blobs)
    for subjects in (np.int64(100307), 100307.0, [100307.0], [np.float32(100307)], "SUB-100307"):
        written = download.fetch_cohort(tmp_path, subjects=subjects, fetcher=fetcher, **QUIET)
        assert [p.name for p in written] == ["sub-100307_sc.h5"]
    for subjects in (100307.5, [np.float64(100307.5)], float("nan")):
        with pytest.raises(ValueError, match="a subject id is a name or a whole number"):
            download.fetch_cohort(tmp_path, subjects=subjects, fetcher=fetcher, **QUIET)


def test_a_listing_whose_reader_stops_reading_ends_quietly():
    """Piped into head -3, sbci download hcp-ya --list ended in a BrokenPipeError traceback.

    The pipe's reading end is closed before the command starts, as head's is
    once it has its three lines.
    """
    import os
    import subprocess
    import sys

    reader, writer = os.pipe()
    os.close(reader)
    try:
        done = subprocess.run(
            [sys.executable, "-c", "import sys; from sbci.cli import main; sys.exit(main())"]
            + ["download", "hcp-ya", "--list"],
            stdout=writer,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    finally:
        os.close(writer)
    assert done.returncode == 1
    assert b"Traceback" not in done.stderr and b"BrokenPipeError" not in done.stderr


def test_an_out_that_is_a_file_is_refused_before_the_terms(tmp_path, monkeypatch, config, capsys):
    """--out naming a file recorded the terms, then failed with a bare [Errno 17]."""
    manifest, blobs = release(FILES)
    use(monkeypatch, manifest, accepted=False)
    fetcher, calls = local_fetcher(blobs)
    taken = tmp_path / "hcp-ya"
    taken.write_text("a file")

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("the terms were asked for"))
    for out in (taken, taken / "inside"):
        with pytest.raises(NotADirectoryError, match="hcp-ya is a file"):
            download.fetch_cohort(out, fetcher=fetcher, accept_terms=True, **QUIET)
        for flags in ([], ["--accept-terms"]):
            assert main(["download", "hcp-ya", "--out", str(out), *flags]) == 1
            assert "hcp-ya is a file" in capsys.readouterr().err
    assert not calls and not (config / download.TERMS_RECORD).exists()
    assert taken.read_text() == "a file"
