"""tools/dataverse_deposit.py: the deposit's checks, its archives, records and calls to Dataverse.

The files here are small stand-ins with the metadata the build tools write; the real
files are checked on Longleaf by ``sbci.validate`` as part of ``prepare``. Dataverse is
stood in for by a local server that keeps what it is sent, through itself or by direct
upload to a stand-in object store, and can be told to report another checksum type, to
have no draft yet (with or without a published version), to be locked for a while, to fail
after keeping a file, to echo the key in an error, to redirect, to offer no direct upload,
to sign no tag, or to fail a transfer to the store; so the upload's resumption, its parts,
the files' restriction and the token's handling are checked without a network.
"""

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import ssl
import subprocess
import sys
import tarfile
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
deposit = pytest.importorskip("dataverse_deposit")

SUBJECTS = ["sub-100307", "sub-103010", "sub-108121"]
TOKEN = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
COMMIT_ID = "0123456789abcdef0123456789abcdef01234567"


def write_file(path, part, subject, values=None, streamlines=7, **extra):
    """A stand-in for one built file: the metadata the build tools write, a few values."""
    metadata = {key: "x" for key in deposit.COMMON_KEYS}
    metadata.update(deposit.METADATA_VALUES[part])
    metadata.update(
        included_connections=part,
        provenance=f"HCP Young Adult {subject}; built",
        spec_version="0.1.0",
    )
    if part == "sc":
        metadata.update(streamline_count=streamlines)
    else:
        metadata.update(
            fc_runs=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
            fc_frames=4800,
            fc_nuisance_model=deposit.FC_NUISANCE,
            provenance=f"HCP Young Adult {subject}; on the grid (barycentric route); built",
        )
    metadata.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle["area"] = np.ones(5)  # first, as in a built file
        handle["connectivity"] = np.arange(10, dtype=np.float32) if values is None else values
        handle["metadata"] = json.dumps(metadata)
        if part == "sc":
            handle["endpoints/vertex_in"] = np.zeros(streamlines, dtype=np.int32)


@pytest.fixture
def built(tmp_path, monkeypatch):
    """Built files for three subjects; one released, its FC one value off the build's."""
    for subject in SUBJECTS:
        for part in deposit.PARTS:
            write_file(tmp_path / part / f"{subject}_{part}.h5", part, subject)
    released = tmp_path / "released"
    released.mkdir()
    shutil.copyfile(tmp_path / "sc" / "sub-100307_sc.h5", released / "sub-100307_sc.h5")
    off = np.arange(10, dtype=np.float32)
    off[3] += 1e-6
    write_file(released / "sub-100307_fc.h5", "fc", "sub-100307", values=off)
    public = {}
    for path in released.iterdir():
        data = path.read_bytes()
        public[path.name] = (len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(deposit, "released_manifest", lambda: dict(public))
    monkeypatch.setattr(deposit, "code_version", lambda: (COMMIT_ID, False))
    table = tmp_path / "manifest.csv"
    with open(table, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "sex", "age_bin", "fluid_intelligence_pmat24"])
        for subject in SUBJECTS:
            writer.writerow([subject, "F", "26-30", 17])
    return tmp_path


def prepare(root, *extra, out=None, validate=False):
    out = out or root / "deposit"
    deposit.main(
        [
            "prepare",
            "--sc",
            str(root / "sc"),
            "--fc",
            str(root / "fc"),
            "--released",
            str(root / "released"),
            "--subjects",
            str(root / "manifest.csv"),
            "--out",
            str(out),
            "--workers",
            "1",
            *([] if validate else ["--no-validate"]),
            *extra,
        ]
    )
    return out


def test_prepare_links_the_released_copies_and_records_every_file(built):
    out = prepare(built)
    link = out / "fc" / "sub-100307_fc.h5"
    assert link.is_symlink() and link.resolve() == (built / "released" / link.name).resolve()
    assert (out / "fc" / "sub-103010_fc.h5").resolve() == (built / "fc/sub-103010_fc.h5").resolve()

    with open(out / "MANIFEST.tsv") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2 * len(SUBJECTS)
    for row in rows:
        data = (out / row["file"]).read_bytes()
        assert int(row["bytes"]) == len(data)
        for name in deposit.DIGESTS:
            assert row[name] == hashlib.new(name, data).hexdigest()
    with open(out / "ARCHIVES.tsv") as handle:
        archives = list(csv.DictReader(handle, delimiter="\t"))
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert sums == [f"{row['sha256']}  {row['archive']}" for row in archives] + [
        f"{row['sha256']}  {row['file']}" for row in rows
    ]

    with open(out / "subjects.csv") as handle:
        table = list(csv.reader(handle))
    assert table[0] == ["subject", "sex", "age_bin"]  # the trait column is left behind
    assert [row[0] for row in table[1:]] == SUBJECTS

    readme = (out / "README.md").read_text()
    assert "$" not in readme and "3 HCP Young Adult subjects" in readme
    assert "version 0.1.0 of the format" in readme
    processing = (out / "PROCESSING.md").read_text()
    assert "7 to 7 (median 7)" in processing and "$" not in processing
    assert "checked again with `sbci.validate`" not in processing  # --no-validate
    checks = (out / "CHECKS.txt").read_text()
    assert "sc/sub-100307_sc.h5: the released bytes; byte-identical" in checks
    assert "fc/sub-100307_fc.h5: the released bytes; connectivity: 1 entries differ" in checks


def test_copies_are_compared_dataset_by_dataset_however_they_nest(tmp_path):
    """The walk stopped at a built file's first dataset, ``area``: the rest went unseen."""
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    write_file(first, "sc", "sub-100307")
    write_file(second, "sc", "sub-100307")
    with h5py.File(second, "r+") as handle:
        handle["endpoints/vertex_in"][2] = 5
    assert deposit.compare_copies(first, second) == (
        "endpoints/vertex_in: 1 entries differ, by at most 5.0e+00"
    )


def test_prepare_never_touches_a_file_that_is_not_its_link(built):
    """An --out whose folder is a source folder, or holds real files, is refused before anything."""
    before = sorted(p.name for p in (built / "fc").iterdir())
    with pytest.raises(SystemExit, match="is, holds or lies inside the source folder"):
        prepare(built, out=built)  # out/fc would be the FC build folder itself
    assert sorted(p.name for p in (built / "fc").iterdir()) == before
    elsewhere = built / "elsewhere"
    (elsewhere / "sc").mkdir(parents=True)
    (elsewhere / "sc" / "notes.h5").write_text("mine")
    with pytest.raises(SystemExit, match="holds files that are not links"):
        prepare(built, out=elsewhere)
    assert (elsewhere / "sc" / "notes.h5").read_text() == "mine"


def test_prepare_refuses_a_released_copy_that_is_not_the_released_file(built):
    write_file(built / "released" / "sub-100307_fc.h5", "fc", "sub-100307")  # rebuilt, not released
    with pytest.raises(SystemExit, match="is not the released file"):
        prepare(built)


def test_prepare_refuses_a_subject_without_both_files(built):
    (built / "fc" / "sub-108121_fc.h5").unlink()
    with pytest.raises(SystemExit, match="sub-108121 has no FC file"):
        prepare(built)


@pytest.mark.parametrize(
    ("column", "value"), [("sex", "Female"), ("age_bin", "27"), ("age_bin", "")]
)
def test_prepare_refuses_values_beyond_the_open_access_ones(built, column, value):
    with open(built / "manifest.csv") as handle:
        rows = list(csv.DictReader(handle))
    rows[1][column] = value
    with open(built / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match=f"{column} holds"):
        prepare(built)


@pytest.mark.parametrize(
    ("part", "change", "message"),
    [
        ("sc", {"Age_in_Yrs": 27}, "metadata keys the build tools do not write: ['Age_in_Yrs']"),
        ("sc", {"provenance": "HCP Young Adult sub-999999; built"}, "provenance does not name"),
        ("sc", {"included_connections": "fc"}, "included_connections is 'fc'"),
        ("sc", {"bandwidth": 0.0027}, "bandwidth is 0.0027, not 0.005"),
        ("sc", {"streamline_count": 9}, "7 endpoints stored for 9 streamlines"),
        ("fc", {"fc_nuisance_model": deposit.FC_NUISANCE + "; band-passed"}, "fc_nuisance_model"),
        ("fc", {"fc_runs": ["REST3_LR"]}, "fc_runs is"),
        ("fc", {"fc_frames": 4801}, "fc_frames is 4801 for 4 runs"),
        ("fc", {"fc_frames": 0}, "fc_frames is 0 for 4 runs"),
        ("fc", {"provenance": "HCP Young Adult sub-103010; built"}, "names no route"),
        ("sc", {"spec_version": "0.2.0"}, "more than one spec_version"),
    ],
)
def test_prepare_refuses_metadata_that_does_not_belong(built, capsys, part, change, message):
    write_file(built / part / f"sub-103010_{part}.h5", part, "sub-103010", **change)
    with pytest.raises(SystemExit, match="problems; no records written"):
        prepare(built)
    assert message in capsys.readouterr().err
    assert not (built / "deposit" / "MANIFEST.tsv").exists()


def test_a_failed_rerun_leaves_no_records_beside_the_new_links(built):
    out = prepare(built)
    write_file(built / "sc" / "sub-103010_sc.h5", "sc", "sub-103010", kernel="rdk")
    with pytest.raises(SystemExit):
        prepare(built)
    assert not any((out / record).exists() for record in deposit.RECORDS)
    with pytest.raises(SystemExit, match="run prepare first"):
        deposit.main(["upload", str(out), "--dataset", "doi:10.5072/FK2/TEST", "--dry-run"])


def test_an_edited_document_is_kept_unless_asked(built, capsys):
    out = prepare(built)
    (out / "README.md").write_text("# edited by hand\n")
    prepare(built)
    assert (out / "README.md").read_text() == "# edited by hand\n"
    assert "kept README.md as edited" in capsys.readouterr().out
    prepare(built, "--rewrite-documents")
    assert (out / "README.md").read_text().startswith("# Continuous structural")


class FakeDataverse(BaseHTTPRequestHandler):
    """Keeps the files it is sent, as a draft dataset does, and answers what the tool asks.

    A file comes through the server in one request, or by direct upload: to a signed
    address on a stand-in object store (``/store/...``), whole or in parts, then registered.
    """

    held: dict = {}
    published: dict = {}
    version_state = None  # of version 1.0, once one is published: RELEASED
    listing: list = []  # version 1.0's files, as its listing gives them
    sent: list = []
    keys: list = []
    checksum = "MD5"
    no_draft = False
    unknown = False
    locked = 0
    added_while_locked = 0
    fail_after_keeping = 0
    corrupt_kept = False
    echo_key = False
    redirect_to = None
    contents: dict = {}
    store: dict = {}
    store_parts: dict = {}
    store_keys: list = []
    store_tags: list = []
    aborted: list = []
    part_size = 1 << 30
    tagging = True
    fixity = "MD5"
    direct = True
    store_fail = 0
    store_refuse = False

    def log_message(self, *args):
        """Keep the test output quiet."""

    def _reply(self, payload, status=200, headers=()):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _base(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def do_GET(self):
        """The draft's file list, its locks (none), the checksum type, the upload addresses."""
        self.keys.append(self.headers.get("X-Dataverse-key"))
        if FakeDataverse.redirect_to:
            self.send_response(302)
            self.send_header("Location", FakeDataverse.redirect_to + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        parsed = urllib.parse.urlsplit(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        if path == "/api/files/fixityAlgorithm":
            if FakeDataverse.fixity is None:
                self._reply({"status": "ERROR"}, 404)
            else:
                self._reply({"status": "OK", "data": {"message": FakeDataverse.fixity}})
        elif path.endswith("/uploadurls"):
            if not FakeDataverse.direct:
                self._reply({"status": "ERROR", "message": "Direct upload not supported"}, 400)
                return
            size, name = int(query["size"][0]), uuid.uuid4().hex
            storage = f"s3://bucket:{name}"
            if size <= FakeDataverse.part_size:
                signed = "host;x-amz-tagging" if FakeDataverse.tagging else "host"
                url = (
                    f"{self._base()}/store/{name}?X-Amz-SignedHeaders={urllib.parse.quote(signed)}"
                )
                data = {
                    "url": url,
                    "partSize": FakeDataverse.part_size,
                    "storageIdentifier": storage,
                }
            else:
                count = -(-size // FakeDataverse.part_size)
                urls = {
                    str(number): f"{self._base()}/store/{name}?partNumber={number}&uploadId=u"
                    f"&X-Amz-SignedHeaders=host"
                    for number in range(1, count + 1)
                }
                finish = f"/api/datasets/mpupload?uploadid=u&storageidentifier={storage}"
                data = {
                    "urls": urls,
                    "abort": finish,
                    "complete": finish,
                    "partSize": FakeDataverse.part_size,
                    "storageIdentifier": storage,
                }
            self._reply({"status": "OK", "data": data})
        elif path.endswith("/locks"):
            locks = [{"lockType": "Ingest"}] if FakeDataverse.locked else []
            FakeDataverse.locked = max(FakeDataverse.locked - 1, 0)
            self._reply({"status": "OK", "data": locks})
        elif path.endswith("/versions/:draft/files"):
            if FakeDataverse.no_draft:
                self._reply({"status": "ERROR", "message": "no draft"}, 404)
            else:
                self._reply({"status": "OK", "data": list(self.held.values())})
        elif path.endswith("/versions/:latest-published/files") and self.published:
            self._reply({"status": "OK", "data": list(self.published.values())})
        elif path.startswith("/api/datasets/:persistentId/versions/1.0"):
            if FakeDataverse.version_state is None:
                self._reply({"status": "ERROR", "message": "no such version"}, 404)
            elif path.endswith("/files"):
                self._reply({"status": "OK", "data": FakeDataverse.listing})
            elif path.endswith("/citation"):
                link = '<a href="https://doi.org/10.5072/FK2/TEST">https://doi.org/10.5072/FK2/TEST</a>'
                message = f'Zhang, Z.; Xiang, Y., 2026, "Connectomes &amp; more", {link}, V1'
                self._reply({"status": "OK", "data": {"message": message}})
            else:
                state = {"versionState": FakeDataverse.version_state}
                self._reply(
                    {"status": "OK", "data": dict(state, releaseTime="2026-10-15T13:07:28Z")}
                )
        elif path == "/api/datasets/:persistentId" and query.get("persistentId"):
            if FakeDataverse.unknown:  # as Dataverse 6.10.1 answers a DOI it does not hold
                message = (
                    "When accessing a dataset based on Persistent ID, a persistentId query "
                    "parameter must be present."
                )
                self._reply({"status": "ERROR", "message": message}, 404)
            else:
                self._reply({"status": "OK", "data": {"id": 1}})
        else:
            self._reply({"status": "ERROR"}, 404)

    def do_PUT(self):
        """A file or a part to the store, or the parts joined when the upload completes."""
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if parsed.path.startswith("/store/"):
            self.store_keys.append(self.headers.get("X-Dataverse-key"))
            name = parsed.path.split("/")[-1]
            if FakeDataverse.store_refuse:
                self._reply({"Error": "AccessDenied"}, 403)
                return
            if FakeDataverse.store_fail:
                FakeDataverse.store_fail -= 1
                self._reply({"Error": "InternalError"}, 500)
                return
            if "partNumber" in query:
                number = query["partNumber"][0]
                self.store_parts.setdefault(name, {})[number] = body
                self._reply({}, headers=[("ETag", f'"etag-{name[:6]}-{number}"')])
            else:
                self.store[name] = body
                self.store_tags.append(self.headers.get("x-amz-tagging"))
                self._reply({}, headers=[("ETag", '"etag"')])
            return
        self.keys.append(self.headers.get("X-Dataverse-key"))
        name = query["storageidentifier"][0].split(":")[-1]
        tags = json.loads(body)
        parts = self.store_parts.pop(name, {})
        expected = {number: f'"etag-{name[:6]}-{number}"' for number in parts}
        if tags != expected:
            self._reply({"status": "ERROR", "message": "the parts' ETags do not match"}, 400)
            return
        self.store[name] = b"".join(parts[str(n)] for n in range(1, len(parts) + 1))
        self._reply({"status": "OK"})

    def do_DELETE(self):
        """Abort an upload in parts."""
        self.keys.append(self.headers.get("X-Dataverse-key"))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        name = query["storageidentifier"][0].split(":")[-1]
        self.aborted.append(name)
        self.store_parts.pop(name, None)
        self._reply({"status": "OK"})

    def do_POST(self):
        """Add a file: from the request itself, or registered from the store."""
        self.keys.append(self.headers.get("X-Dataverse-key"))
        if FakeDataverse.echo_key:
            key = self.headers.get("X-Dataverse-key")
            self._reply({"status": "ERROR", "message": f"bad request with key {key}"}, 400)
            return
        if FakeDataverse.no_draft:  # the first file starts a draft holding the published ones
            FakeDataverse.no_draft = False
            self.held.update(self.published)
        FakeDataverse.added_while_locked += bool(FakeDataverse.locked)
        boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
        body = self.rfile.read(int(self.headers["Content-Length"]))
        fields = {}
        for part in body.split(b"--" + boundary)[1:-1]:
            head, _, content = part.partition(b"\r\n\r\n")
            name = head.split(b'name="')[1].split(b'"')[0].decode()
            fields[name] = (head, content[:-2])
        settings = json.loads(fields["jsonData"][1])
        folder = settings.get("directoryLabel")
        if "storageIdentifier" in settings:  # registered from the store, as sent there
            data = self.store.get(settings["storageIdentifier"].split(":")[-1])
            if data is None:
                self._reply({"status": "ERROR", "message": "no such object in the store"}, 400)
                return
            label = settings["fileName"]
            kind, value = settings["checksum"]["@type"], settings["checksum"]["@value"]
        else:
            label = fields["file"][0].split(b'filename="')[1].split(b'"')[0].decode()
            data = fields["file"][1]
            kind = FakeDataverse.checksum
            value = hashlib.new({"MD5": "md5", "SHA-1": "sha1"}[kind], data).hexdigest()
        key = f"{folder}/{label}" if folder else label
        if key in self.held:  # Dataverse renames a second file of one name
            stem, dot, ext = label.rpartition(".")
            label = f"{stem}-1{dot}{ext}" if dot else f"{label}-1"
            key = f"{folder}/{label}" if folder else label
        entry = {
            "label": label,
            "directoryLabel": folder,
            "restricted": settings.get("restrict", False),
            "dataFile": {"filesize": len(data), "checksum": {"type": kind, "value": value}},
        }
        self.held[key] = entry
        self.contents[key] = data
        self.sent.append(settings)
        if FakeDataverse.fail_after_keeping:
            FakeDataverse.fail_after_keeping -= 1
            if FakeDataverse.corrupt_kept:
                entry["dataFile"]["checksum"]["value"] = "0" * 32
            self._reply({"status": "ERROR", "message": "gateway timeout"}, 504)
        else:
            self._reply({"status": "OK", "data": {"files": [entry]}})


def start():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeDataverse)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def server(monkeypatch):
    FakeDataverse.held, FakeDataverse.sent, FakeDataverse.keys = {}, [], []
    FakeDataverse.published, FakeDataverse.version_state, FakeDataverse.listing = {}, None, []
    FakeDataverse.checksum, FakeDataverse.no_draft = "MD5", False
    FakeDataverse.unknown = False
    FakeDataverse.locked = FakeDataverse.added_while_locked = 0
    FakeDataverse.fail_after_keeping, FakeDataverse.redirect_to = 0, None
    FakeDataverse.corrupt_kept = FakeDataverse.echo_key = False
    FakeDataverse.contents, FakeDataverse.store, FakeDataverse.store_parts = {}, {}, {}
    FakeDataverse.store_keys, FakeDataverse.store_tags, FakeDataverse.aborted = [], [], []
    FakeDataverse.part_size, FakeDataverse.tagging = 1 << 30, True
    FakeDataverse.fixity, FakeDataverse.direct = "MD5", True
    FakeDataverse.store_fail, FakeDataverse.store_refuse = 0, False
    monkeypatch.setattr(deposit.time, "sleep", lambda seconds: None)
    httpd, url = start()
    yield url
    httpd.shutdown()


def run(command, out, url, *extra, dataset="doi:10.5072/FK2/TEST"):
    deposit.main([command, str(out), "--dataset", dataset, "--server", url, *extra])


def local(out, name):
    """A file the deposit sends, by its name in the dataset."""
    return out / deposit.ARCHIVES / name if name.endswith(".tar") else out / name


def test_upload_resumes_sends_the_archives_and_never_shows_the_token(
    built, server, monkeypatch, capsys
):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN + "\n")  # a pasted line break is stripped
    run("upload", out, server, "--limit", "1")  # the trial: the documents and the tutorial's
    assert sorted(FakeDataverse.held) == sorted(
        [*deposit.DOCUMENTS, "sc_tutorial.tar", "fc_tutorial.tar"]
    )
    run("upload", out, server)  # the rest, the trial's files skipped
    assert len(FakeDataverse.sent) == len(deposit.DOCUMENTS) + 4
    by_name = {s["description"]: s for s in FakeDataverse.sent}
    assert by_name["subjects.csv"]["tabIngest"] is False
    assert by_name["subjects.csv"]["mimeType"] == "text/csv"
    assert all(s["restrict"] is False and "directoryLabel" not in s for s in FakeDataverse.sent)
    for name in FakeDataverse.held:  # what the store holds is what the deposit has
        assert FakeDataverse.contents[name] == local(out, name).read_bytes()
    assert set(FakeDataverse.keys) == {TOKEN}
    assert set(FakeDataverse.store_keys) == {None}  # the store's addresses are signed
    assert set(FakeDataverse.store_tags) == {"dv-state=temp"}
    run("verify", out, server, dataset="https://doi.org/10.5072/FK2/TEST")
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err
    assert "all 10 files present" in printed.out


def test_a_server_that_uses_sha1_is_resumed_and_verified_alike(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.fixity = FakeDataverse.checksum = "SHA-1"
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    run("upload", out, server)
    run("verify", out, server)
    assert len(FakeDataverse.sent) == len(deposit.DOCUMENTS) + 4
    assert {s["checksum"]["@type"] for s in FakeDataverse.sent} == {"SHA-1"}


def test_a_dataset_with_no_draft_yet_takes_its_first_files(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.no_draft = True
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    assert "README.md" in FakeDataverse.held


def test_a_file_kept_before_a_failure_is_not_sent_again(built, server, monkeypatch):
    """A gateway may time out after the server kept the file: the draft is checked first."""
    out = prepare(built)
    FakeDataverse.fail_after_keeping = 1
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    assert len(FakeDataverse.sent) == len(deposit.DOCUMENTS) + 2  # nothing twice
    assert not any("-1." in path for path in FakeDataverse.held)
    run("verify", out, server, "--limit", "1")


def test_verify_finds_a_changed_file_and_a_second_copy(built, server, monkeypatch, capsys):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server)
    FakeDataverse.held["fc_01.tar"]["dataFile"]["checksum"]["value"] = "0" * 32
    copy = dict(FakeDataverse.held["sc_tutorial.tar"], label="sc_tutorial-1.tar")
    FakeDataverse.held["sc_tutorial-1.tar"] = copy
    capsys.readouterr()
    with pytest.raises(SystemExit, match="2 problems"):
        run("verify", out, server)
    printed = capsys.readouterr().out
    assert "differs: fc_01.tar" in printed
    assert "a second copy: sc_tutorial-1.tar (of sc_tutorial.tar)" in printed
    with pytest.raises(SystemExit, match="1 problems"):
        run("verify", out, server, "--limit", "1", "--part", "sc")  # the copy, even in a trial
    with pytest.raises(SystemExit, match="delete it there and run again"):
        run("upload", out, server)


def test_upload_refuses_an_archive_that_changed_since_prepare(built, server, monkeypatch):
    out = prepare(built)
    archive = out / deposit.ARCHIVES / "sc_01.tar"
    data = bytearray(archive.read_bytes())
    data[600] ^= 1  # a byte of its first member, the size the same
    archive.write_bytes(bytes(data))
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit, match="changed since prepare"):
        run("upload", out, server)
    assert "sc_01.tar" not in FakeDataverse.held and "sc_tutorial.tar" in FakeDataverse.held


@pytest.mark.parametrize("token", ["", "not-a-token", TOKEN + "\rx", TOKEN.upper()])
def test_a_missing_or_malformed_token_is_refused_without_being_shown(
    built, server, monkeypatch, capsys, token
):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, token)
    with pytest.raises(SystemExit) as raised:
        run("upload", out, server)
    assert deposit.TOKEN in str(raised.value)
    if token:
        printed = capsys.readouterr()
        assert token.strip() not in str(raised.value) + printed.out + printed.err
    assert not FakeDataverse.keys  # nothing was sent


def test_a_redirect_is_refused_and_the_token_goes_nowhere_else(built, server, monkeypatch):
    out = prepare(built)
    other, other_url = start()
    try:
        FakeDataverse.redirect_to = other_url
        monkeypatch.setenv(deposit.TOKEN, TOKEN)
        with pytest.raises(SystemExit, match="redirect"):
            run("upload", out, server)
        assert len(FakeDataverse.keys) == 1  # the first server's request alone
    finally:
        other.shutdown()


def test_the_server_must_be_https_unless_it_is_this_machine(built, monkeypatch):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit, match="must be https"):
        run("upload", out, "http://dataverse.example.org")


@pytest.mark.parametrize(
    "given",
    ["doi:10.15139/S3/ABC123", "10.15139/S3/ABC123", "https://doi.org/10.15139/S3/ABC123"],
)
def test_a_doi_is_taken_in_the_forms_it_is_copied_in(given):
    assert deposit.normalize_dataset(given) == "doi:10.15139/S3/ABC123"


def test_a_doi_pasted_twice_is_refused():
    with pytest.raises(SystemExit, match="is not a DOI"):
        deposit.normalize_dataset("doi:doi:10.15139/S3/ABC123")


def test_upload_asks_for_the_token_and_a_dry_run_does_not(built, server, monkeypatch, capsys):
    out = prepare(built)
    capsys.readouterr()
    monkeypatch.delenv(deposit.TOKEN, raising=False)
    with pytest.raises(SystemExit, match=deposit.TOKEN):
        run("upload", out, server)
    run("upload", out, server, "--dry-run", "--part", "sc")
    printed = capsys.readouterr().out
    assert "would send sc_01.tar" in printed and "fc_" not in printed
    assert "by direct upload" in printed
    assert not FakeDataverse.held


def test_the_expected_nuisance_model_is_the_fc_builders_default():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    build = pytest.importorskip("build_hcp_fc")
    assert " ".join(build.NUISANCE.format("the global signal,").split()) == deposit.FC_NUISANCE


# --- the third review of 12 October 2026 (PORTING.md item 24) --------------------------


def upload_dry_run(out):
    deposit.main(["upload", str(out), "--dataset", "doi:10.5072/FK2/TEST", "--dry-run"])


def test_documents_with_no_record_of_what_prepare_wrote_are_refused_until_rewritten(built):
    """A deposit from before the record kept its old documents for good, as if edited."""
    out = prepare(built)
    (out / deposit.GENERATED_HASHES).unlink()
    with pytest.raises(SystemExit, match="no record of what prepare wrote"):
        prepare(built)
    write_file(built / "sc" / "sub-108121_sc.h5", "sc", "sub-108121", streamlines=9)
    prepare(built, "--rewrite-documents")
    assert "7 to 9 (median 7)" in (out / "PROCESSING.md").read_text()
    assert (out / deposit.GENERATED_HASHES).exists()


def test_a_document_nobody_edited_is_written_afresh(built):
    out = prepare(built)
    write_file(built / "sc" / "sub-108121_sc.h5", "sc", "sub-108121", streamlines=9)
    prepare(built)
    assert "7 to 9 (median 7)" in (out / "PROCESSING.md").read_text()
    assert "kept as edited" not in (out / "CHECKS.txt").read_text()


def test_prepare_removes_nothing_it_did_not_write(built):
    """A --subjects table inside --out, or records no prepare wrote, were removed unread."""
    out = built / "release"
    out.mkdir()
    mine = out / "subjects.csv"
    shutil.copyfile(built / "manifest.csv", mine)
    arguments = ["--sc", str(built / "sc"), "--fc", str(built / "fc"), "--no-validate"]
    with pytest.raises(SystemExit, match="lies inside --out"):
        deposit.main(["prepare", *arguments, "--subjects", str(mine), "--out", str(out)])
    assert mine.exists()
    (out / "SHA256SUMS").write_text("the Drive folder's own sums\n")
    with pytest.raises(SystemExit, match="which no prepare wrote there"):
        prepare(built, out=out)
    assert (out / "SHA256SUMS").read_text() == "the Drive folder's own sums\n"


def test_prepare_refuses_an_out_inside_a_source_folder(built):
    with pytest.raises(SystemExit, match="is, holds or lies inside the source folder"):
        prepare(built, out=built / "sc" / "deposit")
    assert not (built / "sc" / "deposit").exists()


def test_prepare_refuses_a_released_folder_that_is_not_the_release(built):
    shutil.copyfile(built / "sc" / "sub-103010_sc.h5", built / "released" / "sub-103010_sc.h5")
    with pytest.raises(SystemExit, match="holds files the Drive release never had"):
        prepare(built)
    (built / "released" / "sub-103010_sc.h5").unlink()
    with open(built / "manifest.csv") as handle:
        rows = [row for row in csv.DictReader(handle) if row["subject"] != "sub-100307"]
    with open(built / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="released files of subjects not in the deposit"):
        prepare(built)


@pytest.mark.parametrize(
    ("subject", "message"),
    [("sub-100307", "lists a subject twice"), ("sub-12345", "not of the form sub-NNNNNN")],
)
def test_prepare_refuses_a_subject_list_that_is_not_one_line_a_subject(built, subject, message):
    with open(built / "manifest.csv", "a", newline="") as handle:
        csv.writer(handle).writerow([subject, "F", "26-30", 17])
    with pytest.raises(SystemExit, match=message):
        prepare(built)


def test_the_processing_note_names_the_subjects_off_the_usual_route_and_the_short_runs(built):
    route = "HCP Young Adult {}; on the grid ({} route); built"
    write_file(
        built / "fc" / "sub-103010_fc.h5",
        "fc",
        "sub-103010",
        provenance=route.format("sub-103010", "white surface"),
        fc_frames=4700,
    )
    write_file(
        built / "fc" / "sub-108121_fc.h5",
        "fc",
        "sub-108121",
        provenance=route.format("sub-108121", "barycentric and white surface"),
    )
    out = prepare(built)
    processing = (out / "PROCESSING.md").read_text()
    assert "One subject's HCP native mesh is not FreeSurfer's (sub-103010)" in processing
    assert "For 1 subject (sub-108121) one hemisphere's is not" in processing
    assert "Of the 3 with 4, 1 have a run shorter than its 1,200 frames" in processing


@pytest.mark.parametrize("change", ["column", "value", "sums"])
def test_upload_takes_only_the_records_prepare_wrote(built, change):
    """A column added to subjects.csv after prepare was sent, and verify passed it."""
    out = prepare(built)
    table = out / "subjects.csv"
    rows = list(csv.reader(table.open()))
    if change == "column":
        rows = [[*row, "Age_in_Yrs" if i == 0 else "27"] for i, row in enumerate(rows)]
        message = "may be deposited"
    elif change == "value":
        rows[1][2] = "27"
        message = "age_bin holds"
    if change == "sums":
        sums = out / "SHA256SUMS"
        sums.write_text(sums.read_text().replace("sc/", "sc/x", 1))
        message = "is not what MANIFEST.tsv and ARCHIVES.tsv say"
    else:
        with table.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    with pytest.raises(SystemExit, match=message):
        upload_dry_run(out)


def test_a_prepare_that_stopped_while_writing_its_records_is_not_uploaded(built, monkeypatch):
    out = prepare(built)
    write_file(built / "sc" / "sub-108121_sc.h5", "sc", "sub-108121", streamlines=9)
    fill = deposit.fill

    def failing(template, values):
        if template == "PROCESSING.md":
            raise OSError(122, "Disk quota exceeded")
        return fill(template, values)

    monkeypatch.setattr(deposit, "fill", failing)
    with pytest.raises(OSError):
        prepare(built)
    assert (out / "MANIFEST.tsv").exists() and not (out / "CHECKS.txt").exists()
    with pytest.raises(SystemExit, match="CHECKS.txt is missing"):
        upload_dry_run(out)


@pytest.mark.parametrize("limit", ["-1", "0"])
def test_a_limit_must_be_a_count(built, capsys, limit):
    out = prepare(built)
    with pytest.raises(SystemExit):
        deposit.main(
            ["upload", str(out), "--dataset", "doi:10.5072/FK2/TEST", "--dry-run", "--limit", limit]
        )
    assert "not a count" in capsys.readouterr().err


def test_a_held_file_matches_only_with_its_size_and_checksum():
    digest = {"bytes": 10, "md5": "a" * 32, "sha1": "b" * 40, "sha256": "c" * 64, "sha512": "d"}
    assert deposit.matches((10, "MD5", "A" * 32, None), digest)
    assert not deposit.matches((11, "MD5", "a" * 32, None), digest)
    assert not deposit.matches((10, "MD5", "e" * 32, None), digest)
    with pytest.raises(SystemExit, match="cannot compare"):
        deposit.matches((10, "CRC32", "a" * 32, None), digest)


def test_the_restriction_must_be_the_same_on_resuming_and_in_verify(
    built, server, monkeypatch, capsys
):
    """A trial sent open and the rest restricted left a mixed draft that verify passed."""
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    with pytest.raises(SystemExit, match="in the draft open, where this run sends restricted"):
        run("upload", out, server, "--restrict")
    assert len(FakeDataverse.sent) == len(deposit.DOCUMENTS) + 2
    run("upload", out, server)
    run("verify", out, server)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="10 problems"):
        run("verify", out, server, "--restrict")
    assert "open, where --restrict says otherwise: README.md" in capsys.readouterr().out


def test_a_doi_the_server_does_not_hold_is_named_before_anything_is_sent(
    built, server, monkeypatch
):
    """Dataverse 6.10.1 answers an unknown DOI with a message about a missing parameter.

    On UNC's demo server, whose DOIs are 10.33563/DMO/..., a DOI copied with another
    prefix got that message from the first request and nothing said what was wrong.
    """
    out = prepare(built)
    FakeDataverse.unknown = True
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    for command in ("upload", "verify"):
        FakeDataverse.keys = []
        with pytest.raises(SystemExit, match="holds no dataset doi:10.5072/FK2/TEST") as raised:
            run(command, out, server, "--limit", "1")
        assert "persistentId" not in str(raised.value).replace("persistentId=", "")
        assert len(FakeDataverse.keys) == 1  # the question, and nothing after it
    assert not FakeDataverse.sent and not FakeDataverse.store


def test_upload_waits_until_the_dataset_is_unlocked(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.locked = 3
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    assert FakeDataverse.added_while_locked == 0
    assert len(FakeDataverse.sent) == len(deposit.DOCUMENTS) + 2


def test_a_published_dataset_counts_its_published_files_as_held(built, server, monkeypatch):
    """A 404 for the draft read as an empty dataset, so every published file went again."""
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    FakeDataverse.published, FakeDataverse.held = dict(FakeDataverse.held), {}
    FakeDataverse.no_draft, FakeDataverse.sent = True, []  # published, no draft since
    run("upload", out, server)
    assert len(FakeDataverse.sent) == 2  # sc_01.tar and fc_01.tar
    assert len(FakeDataverse.held) == len(deposit.DOCUMENTS) + 4  # no copies
    run("verify", out, server)


def test_verify_finds_a_second_copy_of_a_file_without_an_extension(
    built, server, monkeypatch, capsys
):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server)
    FakeDataverse.held["SHA256SUMS-1"] = dict(
        FakeDataverse.held["SHA256SUMS"], label="SHA256SUMS-1"
    )
    capsys.readouterr()
    with pytest.raises(SystemExit, match="1 problems"):
        run("verify", out, server, "--limit", "1")  # a trial, which skips "not in the deposit"
    assert "a second copy: SHA256SUMS-1 (of SHA256SUMS)" in capsys.readouterr().out


def test_verify_reports_a_file_the_deposit_does_not_hold(built, server, monkeypatch, capsys):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server)
    FakeDataverse.held["notes.txt"] = dict(FakeDataverse.held["README.md"], label="notes.txt")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="1 problems"):
        run("verify", out, server)
    assert "not in the deposit: notes.txt" in capsys.readouterr().out


def test_a_retry_that_finds_another_file_in_the_draft_stops(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.fail_after_keeping, FakeDataverse.corrupt_kept = 1, True
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit, match="is in the draft as another file"):
        run("upload", out, server, "--limit", "1")
    assert len(FakeDataverse.sent) == 1  # not sent a second time


def test_two_uploads_of_one_deposit_cannot_run_at_once(built, server, monkeypatch):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    (out / ".upload.lock").write_text("12345 earlier\n")
    with pytest.raises(SystemExit, match="another upload of this deposit is running"):
        run("upload", out, server)
    assert not FakeDataverse.sent
    (out / ".upload.lock").unlink()
    run("upload", out, server, "--limit", "1")
    assert not (out / ".upload.lock").exists()


def test_a_server_that_echoes_the_key_does_not_get_it_printed(built, server, monkeypatch, capsys):
    out = prepare(built)
    FakeDataverse.echo_key = True
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit) as raised:
        run("upload", out, server, "--limit", "1")
    printed = capsys.readouterr()
    assert TOKEN not in str(raised.value) + printed.out + printed.err
    assert "[token]" in str(raised.value)


# --- the fourth review of 12 October 2026 (PORTING.md item 27) --------------------------


def test_a_cell_added_to_a_row_of_subjects_csv_is_not_sent(built):
    """A row's extra cell sat under no column, so the header's check let it through."""
    out = prepare(built)
    table = out / "subjects.csv"
    rows = list(csv.reader(table.open()))
    rows[1].append("27")  # an exact age, say
    with table.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(SystemExit, match="not the table prepare wrote"):
        upload_dry_run(out)


def test_subjects_csv_must_list_the_subjects_whose_files_go_up(built):
    out = prepare(built)
    table = out / "subjects.csv"
    table.write_text("".join(table.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match="does not list the subjects MANIFEST.tsv holds"):
        upload_dry_run(out)


def test_the_readme_installs_the_commit_prepare_ran_from(built):
    out = prepare(built)
    assert f"SBCI.git@{COMMIT_ID}" in (out / "README.md").read_text()
    assert f"the code at {COMMIT_ID}" in (out / "CHECKS.txt").read_text()


def test_without_a_commit_prepare_refuses_before_it_changes_anything(built, monkeypatch):
    """Outside a checkout the README pinned "@main", a branch that moves, without a word."""
    out = prepare(built)
    before = {name: (out / name).read_bytes() for name in deposit.RECORDS}
    monkeypatch.setattr(deposit, "code_version", lambda: (None, False))
    with pytest.raises(SystemExit, match="not in a git checkout of sbci"):
        prepare(built)
    assert {name: (out / name).read_bytes() for name in deposit.RECORDS} == before
    with pytest.raises(SystemExit, match="is not a commit"):
        prepare(built, "--ref", "main")
    prepare(built, "--ref", "f" * 40)
    assert f"SBCI.git@{'f' * 40}" in (out / "README.md").read_text()


@pytest.mark.parametrize("where", ["a plain folder", "another repository"])
def test_a_copy_of_the_tool_outside_its_checkout_names_no_commit(tmp_path, where):
    copy = tmp_path / "project" / "vendor" / "tools" / "dataverse_deposit.py"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(deposit.__file__, copy)
    if where == "another repository":  # whose HEAD is no commit of sbci
        if shutil.which("git") is None:
            pytest.skip("git is not installed")
        for command in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=a", "-c", "user.email=a@example.org", "-c", "commit.gpgsign=false"]
            + ["commit", "-q", "-m", "vendored"],
        ):
            subprocess.run(["git", "-C", str(tmp_path / "project"), *command], check=True)
    spec = importlib.util.spec_from_file_location("deposit_copy", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.code_version() == (None, False)


def test_in_a_checkout_the_commit_is_the_checkouts_head():
    if not (Path(__file__).resolve().parents[1] / ".git").exists():
        pytest.skip("this tree is not a git checkout")
    commit, _ = deposit.code_version()
    assert commit is not None and deposit.COMMIT.match(commit)


def test_the_released_subjects_go_up_only_as_their_released_copies(built):
    """Without --released the build's copies of the eleven went up, other bytes than Drive's."""
    arguments = ["--sc", str(built / "sc"), "--fc", str(built / "fc"), "--workers", "1"]
    arguments += ["--no-validate", "--out", str(built / "deposit")]
    with pytest.raises(SystemExit, match="give their released copies with --released"):
        deposit.main(["prepare", *arguments, "--subjects", str(built / "manifest.csv")])
    assert not (built / "deposit").exists()
    others = built / "others.csv"
    others.write_text("subject,sex,age_bin\nsub-103010,F,26-30\nsub-108121,M,22-25\n")
    deposit.main(["prepare", *arguments, "--subjects", str(others)])
    assert (built / "deposit" / "CHECKS.txt").exists()


def test_prepare_validates_every_file_unless_told_not_to(built, monkeypatch, capsys):
    """Every other test passes --no-validate, so dropping the validation went unnoticed."""
    with pytest.raises(SystemExit, match="problems; no records written"):
        prepare(built, validate=True)  # the stand-ins are no ico4 files
    assert "sc/sub-100307_sc.h5: " in capsys.readouterr().err
    monkeypatch.setattr("sbci.validate.validate_file", lambda path: [])
    out = prepare(built, validate=True)
    assert "validated by sbci.validate.validate_file" in (out / "CHECKS.txt").read_text()
    assert "checked again with `sbci.validate`" in (out / "PROCESSING.md").read_text()


def test_a_dry_run_checks_the_server_and_that_every_file_is_there(built):
    """A dry run passed a broken link and an http server; the upload then failed midway."""
    out = prepare(built)
    command = ["upload", str(out), "--dataset", "doi:10.5072/FK2/TEST", "--dry-run"]
    with pytest.raises(SystemExit, match="must be https"):
        deposit.main([*command, "--server", "http://dataverse.example.org"])
    (out / deposit.ARCHIVES / "fc_01.tar").unlink()
    with pytest.raises(SystemExit, match="missing or not the size prepare recorded"):
        upload_dry_run(out)


@pytest.mark.parametrize(
    "failure",
    [ssl.SSLEOFError(8, "EOF occurred in violation of protocol"), OSError(113, "No route")],
)
def test_a_connection_lost_mid_request_is_retried_or_checked(monkeypatch, failure):
    """An SSL or socket error escaped as a traceback: neither retried nor looked for."""
    dataverse = deposit.Dataverse(deposit.SERVER, "doi:10.5072/FK2/TEST", TOKEN, pause=0)

    def lost(*args):
        raise failure

    monkeypatch.setattr(dataverse, "_once", lost)
    with pytest.raises(deposit.Uncertain):
        dataverse._call("POST", dataverse._url("add"), repeatable=False)
    with pytest.raises(deposit.Refused, match="after|GET"):
        dataverse._call("GET", dataverse._url("locks"))


# --- the deposit in archives, 13 October 2026 (PORTING.md item 29) ------------------------


def read_tar(path):
    """``{member name: (header, bytes)}`` of an archive."""
    with tarfile.open(path) as archive:
        return {m.name: (m, archive.extractfile(m).read()) for m in archive.getmembers()}


def test_the_archives_hold_the_files_the_tutorials_apart_and_the_rest_in_groups(built):
    """``--per-archive 1``: the released subject alone in the tutorial's, one in each other."""
    out = prepare(built, "--per-archive", "1")
    names = sorted(p.name for p in (out / deposit.ARCHIVES).iterdir())
    assert names == sorted(
        f"{part}_{label}.tar" for part in deposit.PARTS for label in ("tutorial", "01", "02")
    )
    with open(out / "ARCHIVES.tsv") as handle:
        rows = {row["archive"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert list(rows) == [
        f"{part}_{label}.tar" for part in deposit.PARTS for label in ("tutorial", "01", "02")
    ]
    assert [row["holds"] for row in rows.values()] == SUBJECTS * 2
    for name, row in rows.items():
        part = row["part"]
        members = read_tar(out / deposit.ARCHIVES / name)
        assert sorted(members) == [f"{part}/{s}_{part}.h5" for s in row["holds"].split()]
        for member, (info, data) in members.items():
            assert data == (out / member).read_bytes()
            assert (info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname) == (
                deposit.ARCHIVE_TIME,
                0o644,
                0,
                0,
                "",
                "",
            )
        data = (out / deposit.ARCHIVES / name).read_bytes()
        assert int(row["bytes"]) == len(data)
        for digest in deposit.DIGESTS:
            assert row[digest] == hashlib.new(digest, data).hexdigest()
    with open(out / "MANIFEST.tsv") as handle:
        manifest = {row["file"]: row["archive"] for row in csv.DictReader(handle, delimiter="\t")}
    assert manifest["fc/sub-108121_fc.h5"] == "fc_02.tar"
    assert manifest["sc/sub-100307_sc.h5"] == "sc_tutorial.tar"
    readme = (out / "README.md").read_text()
    assert "`sc_tutorial.tar` and `sc_01.tar` to `sc_02.tar` for SC" in readme
    assert "hold the 1 subjects of the sbci package's tutorial" in readme
    assert "grep sub-100307 ARCHIVES.tsv" in readme and "$" not in readme
    assert "6 archives of at most 1 subjects" in (out / "CHECKS.txt").read_text()


def test_the_rest_go_in_as_few_archives_as_hold_them_as_equal_as_they_divide():
    subjects = [f"sub-{100000 + i}" for i in range(946)]
    labelled = deposit.groups(subjects, set(subjects[:11]), 50)
    assert labelled[0] == ("tutorial", subjects[:11])
    sizes = [len(chunk) for _, chunk in labelled[1:]]
    assert len(sizes) == 19 and set(sizes) == {49, 50} and sum(sizes) == 935
    assert [label for label, _ in labelled[1:]] == [f"{i:02d}" for i in range(1, 20)]
    assert [s for _, chunk in labelled[1:] for s in chunk] == subjects[11:]
    assert deposit.groups(subjects[:3], set(), 50) == [("01", subjects[:3])]
    assert [label for label, _ in deposit.groups(subjects, set(), 1)][-1] == "946"


def test_the_same_files_give_the_same_archives_and_unchanged_ones_are_kept(built, capsys):
    out = prepare(built)
    first = {p.name: p.read_bytes() for p in (out / deposit.ARCHIVES).iterdir()}
    shutil.rmtree(out / deposit.ARCHIVES)
    (out / deposit.ARCHIVE_RECORD).unlink()
    prepare(built)  # written afresh: the same bytes
    assert {p.name: p.read_bytes() for p in (out / deposit.ARCHIVES).iterdir()} == first
    capsys.readouterr()
    write_file(built / "sc" / "sub-108121_sc.h5", "sc", "sub-108121", streamlines=9)
    prepare(built)
    printed = capsys.readouterr().out
    assert "wrote sc_01.tar" in printed and "wrote fc_01.tar" not in printed
    assert "1 written, 3 kept from the last prepare" in (out / "CHECKS.txt").read_text()
    assert (out / deposit.ARCHIVES / "fc_01.tar").read_bytes() == first["fc_01.tar"]
    assert (out / deposit.ARCHIVES / "sc_01.tar").read_bytes() != first["sc_01.tar"]


def test_a_rerun_removes_its_leftovers_and_the_archives_no_longer_planned(built):
    out = prepare(built, "--per-archive", "1")
    for half in ("sc_01.tar.part", "fc_tutorial.tar.part"):  # a stopped run's
        (out / deposit.ARCHIVES / half).write_bytes(b"half")
    prepare(built)  # the rest in one archive a part now; the tutorial's kept
    assert sorted(p.name for p in (out / deposit.ARCHIVES).iterdir()) == [
        "fc_01.tar",
        "fc_tutorial.tar",
        "sc_01.tar",
        "sc_tutorial.tar",
    ]


def test_prepare_refuses_an_archives_folder_holding_what_it_did_not_write(built):
    out = prepare(built)
    (out / deposit.ARCHIVES / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit, match="holds files no prepare wrote there"):
        prepare(built)
    assert (out / deposit.ARCHIVES / "notes.txt").read_text() == "mine"
    assert (out / "CHECKS.txt").exists()  # refused before anything changed


def test_a_member_that_changed_after_its_check_stops_the_archive(built, monkeypatch):
    """A file rewritten between its check and its archiving would go up unchecked."""
    real = deposit.build_archive

    def rewritten(task):
        path, members = task
        _, source, _ = members[0]
        write_file(Path(source), "sc", "sub-103010", values=np.ones(10, "f4"))
        return real(task)

    monkeypatch.setattr(deposit, "build_archive", rewritten)
    with pytest.raises(SystemExit, match="changed since it was checked"):
        prepare(built)
    assert not (built / "deposit" / "CHECKS.txt").exists()


def test_a_large_file_goes_up_in_parts_and_the_token_never_reaches_the_store(
    built, server, monkeypatch
):
    out = prepare(built)
    FakeDataverse.part_size = 4096  # every archive, and the README, in parts
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server)
    for name in ("README.md", "sc_tutorial.tar", "sc_01.tar", "fc_01.tar"):
        assert FakeDataverse.contents[name] == local(out, name).read_bytes()
    assert (out / deposit.ARCHIVES / "sc_01.tar").stat().st_size > 2 * 4096
    assert set(FakeDataverse.store_keys) == {None} and not FakeDataverse.aborted
    assert not FakeDataverse.store_parts  # every upload in parts was completed
    run("verify", out, server)


def test_a_store_that_signs_no_tag_is_sent_none(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.tagging = False
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    assert FakeDataverse.store_tags and set(FakeDataverse.store_tags) == {None}


@pytest.mark.parametrize(
    ("fixity", "kind", "name"), [("SHA-256", "SHA-256", "sha256"), (None, "MD5", "md5")]
)
def test_a_file_is_registered_with_the_checksum_the_server_uses(
    built, server, monkeypatch, fixity, kind, name
):
    """``None``: a server without ``fixityAlgorithm``, taken to use MD5."""
    out = prepare(built)
    FakeDataverse.fixity = fixity
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    run("upload", out, server, "--limit", "1")
    for settings in FakeDataverse.sent:
        data = FakeDataverse.contents[settings["fileName"]]
        assert settings["checksum"] == {
            "@type": kind,
            "@value": hashlib.new(name, data).hexdigest(),
        }
        assert settings["mimeType"] == deposit.media_type(settings["fileName"])
    run("verify", out, server, "--limit", "1")


def test_a_failed_transfer_is_sent_again_and_a_refused_one_aborts_its_parts(
    built, server, monkeypatch
):
    out = prepare(built)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    FakeDataverse.store_fail = 2
    run("upload", out, server, "--limit", "1")  # two failures, then through
    run("verify", out, server, "--limit", "1")
    FakeDataverse.part_size, FakeDataverse.store_refuse = 4096, True
    with pytest.raises(SystemExit, match="PUT to the store: 403"):
        run("upload", out, server)
    assert FakeDataverse.aborted and "sc_01.tar" not in FakeDataverse.held


def test_without_direct_upload_the_files_go_through_the_server(built, server, monkeypatch):
    out = prepare(built)
    FakeDataverse.direct = False
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit, match="--through-server"):
        run("upload", out, server, "--limit", "1")
    run("upload", out, server, "--through-server")
    assert not FakeDataverse.store and not any("storageIdentifier" in s for s in FakeDataverse.sent)
    for name in FakeDataverse.held:
        assert FakeDataverse.contents[name] == local(out, name).read_bytes()
    run("verify", out, server)


def test_through_the_server_a_file_too_large_for_one_request_is_refused(built, server, monkeypatch):
    out = prepare(built)
    monkeypatch.setattr(deposit, "THROUGH_SERVER_LIMIT", 100)
    monkeypatch.setenv(deposit.TOKEN, TOKEN)
    with pytest.raises(SystemExit, match="more than goes through the server"):
        run("upload", out, server, "--through-server")


def test_upload_takes_only_an_archive_table_that_matches_the_manifest(built):
    out = prepare(built)
    table = out / "ARCHIVES.tsv"
    swapped = table.read_text().replace("sub-103010 sub-108121", "sub-108121 sub-103010", 1)
    assert swapped != table.read_text()
    table.write_text(swapped)
    with pytest.raises(SystemExit, match="does not list the subjects MANIFEST.tsv puts in"):
        upload_dry_run(out)


def test_an_address_the_server_hands_back_on_another_host_gets_no_token():
    dataverse = deposit.Dataverse(deposit.SERVER, "doi:10.5072/FK2/TEST", TOKEN, pause=0)
    assert (
        dataverse._server_url("/api/datasets/mpupload?x=1")
        == f"{deposit.SERVER}/api/datasets/mpupload?x=1"
    )
    with pytest.raises(deposit.Refused, match="another host"):
        dataverse._server_url("https://evil.example.org/api/datasets/mpupload?x=1")
    with pytest.raises(deposit.Refused, match="not https"):
        dataverse._store("http://store.example.org/x", io.BytesIO(b"ab"), 0, 2, True)


def publish(out, change=None):
    """Version 1.0 of the deposit published: its files listed as Dataverse lists them."""
    FakeDataverse.version_state, FakeDataverse.listing = "RELEASED", []
    rows = deposit.read_archives(out, "all", 0)
    for number, row in enumerate(rows):
        entry = {
            "label": row["archive"],
            "directoryLabel": None,
            "restricted": False,
            "dataFile": {
                "id": 7000 + number,
                "filesize": int(row["bytes"]),
                "checksum": {"type": "MD5", "value": row["md5"]},
            },
        }
        FakeDataverse.listing.append(change(entry) if change else entry)
    return rows


def index(out, server, path):
    deposit.main(
        ["index", str(out), "--dataset", "doi:10.5072/FK2/TEST", "--version", "1.0"]
        + ["--server", server, "--out", str(path)]
    )
    return json.loads(path.read_text())


def test_the_index_places_every_file_where_its_bytes_lie_and_sbci_download_reads_it(
    built, server, tmp_path, monkeypatch
):
    out = prepare(built, "--per-archive", "1")
    rows = publish(out)
    made = index(out, server, tmp_path / "hcp_ya.json")
    assert all(key is None for key in FakeDataverse.keys)  # asked with no token
    assert made["release"] == {
        "server": server,
        "dataset": "doi:10.5072/FK2/TEST",
        "version": "1.0",
        "published": "2026-10-15T13:07:28Z",
        "citation": 'Zhang, Z.; Xiang, Y., 2026, "Connectomes & more", '
        "https://doi.org/10.5072/FK2/TEST, V1",
        "terms": deposit.TERMS_URL,
    }
    assert made["data_use"] == (out / "DATA_USE.txt").read_text()
    assert [a["id"] for a in made["archives"].values()] == [7000 + n for n in range(len(rows))]
    assert [e["subject"] for e in made["subjects"]] == SUBJECTS
    assert [e["tutorial"] for e in made["subjects"]] == [True, False, False]  # the released one
    for entry in made["subjects"]:
        assert (entry["sex"], entry["age_bin"]) == ("F", "26-30")
        for part, record in entry["files"].items():
            with open(out / deposit.ARCHIVES / record["archive"], "rb") as handle:
                handle.seek(record["offset"])
                data = handle.read(record["bytes"])
            assert data == (out / part / f"{entry['subject']}_{part}.h5").read_bytes()
            assert hashlib.sha256(data).hexdigest() == record["sha256"]

    # and the package's downloader, given this index, fetches those bytes and checks them
    from sbci import download

    monkeypatch.setattr(download, "load_manifest", lambda cohort="hcp-ya": made)
    monkeypatch.setenv("SBCI_CONFIG_DIR", str(tmp_path / "config"))

    def fetcher(source, destination, report):
        with open(out / deposit.ARCHIVES / source["archive"], "rb") as handle:
            handle.seek(source["offset"])
            destination.write_bytes(handle.read(source["bytes"]))

    got = tmp_path / "got"
    written = download.fetch_cohort(
        got, subjects="all", fetcher=fetcher, accept_terms=True, report=lambda _: None
    )
    assert len(written) == 6
    notice = (got / "DATA_USE.txt").read_text()
    assert notice.startswith(made["data_use"]) and "Cite it as: Zhang, Z.; Xiang, Y." in notice


def test_the_index_is_of_a_published_version_holding_the_deposits_archives(built, server, tmp_path):
    out = prepare(built)
    path = tmp_path / "hcp_ya.json"
    with pytest.raises(SystemExit, match="no such version"):
        index(out, server, path)
    publish(out)
    FakeDataverse.version_state = "DRAFT"
    with pytest.raises(SystemExit, match="version 1.0 of doi:10.5072/FK2/TEST is not published"):
        index(out, server, path)

    def other_bytes(entry):
        if entry["label"] == "fc_01.tar":
            entry["dataFile"]["checksum"]["value"] = "0" * 32
        return entry

    publish(out, other_bytes)
    with pytest.raises(SystemExit, match="fc_01.tar in version 1.0 .* is not the deposit's"):
        index(out, server, path)
    publish(
        out,
        lambda entry: dict(entry, label="renamed.tar") if entry["label"] == "sc_01.tar" else entry,
    )
    with pytest.raises(SystemExit, match="sc_01.tar is not in version 1.0"):
        index(out, server, path)
    assert not path.exists()


def test_prepare_compares_only_the_tutorials_files_with_the_shipped_index(monkeypatch):
    """The shipped index lists all 946; the files released before the deposit are the 11."""
    public = deposit.released_manifest()
    assert len(public) == 22 and "sub-100307_sc.h5" in public
