"""Fetching the HCP Young Adult connectomes from their release on UNC Dataverse.

The SC and FC files of 946 HCP Young Adult subjects are published as one dataset on UNC
Dataverse, in tar archives of at most 50 subjects. The repository carries an index instead
of the files: ``data/hcp_ya.json`` gives, for every subject's SC and FC file, its archive,
where its bytes start there, its size and SHA-256, beside the subject's sex and open-access
age band, the dataset's DOI and version, each archive's file id in that version, and the
terms the files come under. :func:`fetch_cohort` downloads what is asked for -- by default
the eleven subjects of the tutorial, any of the 946 by id, or all of them -- verifies every
file against the index, and skips files already present and correct, so it can be re-run;
``sbci download hcp-ya`` is the command form.

A file comes as a range request for its own bytes inside its archive, so one subject is
about 90 MB whichever archive holds it, and a transfer cut part way is taken up where it
stopped. An archive whose files are all wanted, and none of them here yet, is read whole as
one stream instead, its files split off as they pass; if that stream breaks, the files it
had not yet given come by range.

The files are redistributed under the WU-Minn HCP Consortium Open Access Data Use Terms.
Downloading needs those terms accepted once on the machine -- ``sbci download`` asks, or
``accept_terms=True`` / ``--accept-terms`` -- and the acceptance is kept in the user's
configuration directory (``~/.config/sbci``, or ``$SBCI_CONFIG_DIR``). Every download also
writes them beside the files, as ``DATA_USE.txt``. A Dataverse that asks a downloader for
the dataset's guestbook (6.10 and later) is sent the answer "I agree", which that acceptance
stands behind.

:func:`fetch_basis` fetches the other data the package does not bundle: the Laplace-Beltrami
basis of the grid that the ``rdk`` and ``matern`` kernels need, two files of 50 MB from the
SBCI group's SBCI_Toolkit, read at one commit and verified. ``smooth()`` calls it the first
time it needs them, and ``sbci basis`` does it ahead of time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

COHORTS = ("hcp-ya",)
MANIFESTS = {"hcp-ya": "hcp_ya.json"}
CHUNK = 1 << 20
#: Tries of one transfer before giving up; each takes up where the last stopped.
ATTEMPTS = 4
#: The answer a Dataverse guestbook is sent, once the terms are accepted.
ANSWER = "I agree"
TERMS_RECORD = "accepted_terms.json"
TERMS_URL = (
    "https://www.humanconnectome.org/study/hcp-young-adult/document/"
    "wu-minn-hcp-consortium-open-access-data-use-terms"
)
#: What ``sbci download`` shows before asking for the terms to be accepted.
TERMS_NOTICE = f"""\
The HCP Young Adult connectomes are derived from data of the WU-Minn Human Connectome
Project and redistributed under the WU-Minn HCP Consortium Open Access Data Use Terms:
  {TERMS_URL}
Among them: you will not attempt to identify or contact any subject; you will follow your
institution's rules for research on human subjects; you will redistribute these data, or
data derived from them, only under the same terms; and you will acknowledge the HCP in any
public presentation of results, in the words of DATA_USE.txt, written beside the files."""

#: The Laplace-Beltrami basis of the ico4 grid, which the ``rdk`` and ``matern`` kernels need:
#: two files of the SBCI group's SBCI_Toolkit, read at one commit, each with its size and its
#: SHA-256 -- the files the comparisons with MATLAB were run with.
BASIS_URL = (
    "https://raw.githubusercontent.com/sbci-brain/SBCI_Toolkit/{commit}/concon_estimate/{name}"
)
BASIS_COMMIT = "a558fe82c04ccb477841026314012560407a7f72"
BASIS_FILES = {
    "EV_LBO_ds_ico4_L.mat": (
        50_446_455,
        "ad8b489b211ca54c084702d889068af2b9e63ebc3f836b0a972efc504c76a81f",
    ),
    "EV_LBO_ds_ico4_R.mat": (
        50_446_322,
        "ffd64b273b5c786e7c33aca43de3a33c35de8b0c9781c38140f071357e68a49a",
    ),
}


class TermsNotAccepted(RuntimeError):
    """The cohort's terms of use have not been accepted on this machine."""


def load_manifest(cohort: str = "hcp-ya") -> dict:
    """The cohort's index as shipped with the package."""
    if cohort not in MANIFESTS:
        raise ValueError(f"cohort must be one of {COHORTS}, got {cohort!r}")
    path = resources.files("sbci.data") / MANIFESTS[cohort]
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def sha256_of(path: Path) -> str:
    """Hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# --- the terms ---------------------------------------------------------------------------


def config_dir() -> Path:
    """Where the package keeps what it remembers.

    ``$SBCI_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/sbci``, else ``~/.config/sbci``.
    """
    given = os.environ.get("SBCI_CONFIG_DIR")
    if given:
        return Path(given)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sbci"


def _terms_digest(manifest: dict) -> str:
    """What an acceptance is of: these terms, as the index words them."""
    return hashlib.sha256(manifest.get("data_use", "").encode("utf-8")).hexdigest()


def _read_record() -> dict:
    try:
        return json.loads((config_dir() / TERMS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def terms_accepted(cohort: str = "hcp-ya") -> bool:
    """Whether the cohort's terms, as they now stand, were accepted on this machine."""
    manifest = load_manifest(cohort)
    if not manifest.get("data_use"):
        return True
    entry = _read_record().get(cohort) or {}
    return entry.get("terms_sha256") == _terms_digest(manifest)


def accept_terms(cohort: str = "hcp-ya") -> Path:
    """Record that the cohort's terms of use are accepted on this machine; the record's path.

    Call it only having read the terms: for ``hcp-ya``, the WU-Minn HCP Consortium Open
    Access Data Use Terms (:data:`TERMS_URL`), which ``DATA_USE.txt`` summarizes.
    """
    manifest = load_manifest(cohort)
    record = _read_record()
    record[cohort] = {
        "terms_sha256": _terms_digest(manifest),
        "terms": manifest.get("release", {}).get("terms", TERMS_URL),
        "dataset": manifest.get("release", {}).get("dataset"),
        "accepted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    folder = config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / TERMS_RECORD
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path


def _require_terms(manifest: dict, cohort: str, accept: bool) -> None:
    if not manifest.get("data_use") or terms_accepted(cohort):
        return
    if accept:
        accept_terms(cohort)
        return
    raise TermsNotAccepted(
        f"the {cohort} files come under the WU-Minn HCP Consortium Open Access Data Use Terms "
        f"({manifest.get('release', {}).get('terms', TERMS_URL)}), which must be accepted "
        "once on this machine before downloading: run `sbci download` in a terminal and "
        "type I agree when asked, or, having read them, pass --accept-terms (accept_terms=True "
        "in Python)"
    )


# --- what is asked for -------------------------------------------------------------------


def _check_request(manifest: dict, subjects, modalities) -> tuple[list, tuple]:
    """The index's entries asked for, in its order, and the modalities as a tuple."""
    entries = manifest["subjects"]
    if subjects is None:
        chosen = [e for e in entries if e.get("tutorial", True)]
    elif isinstance(subjects, str) and subjects.strip() == "all":
        chosen = list(entries)
    else:
        if isinstance(subjects, (str, int)):
            subjects = [subjects]
        # HCP ids are often written bare (100307), and as numbers; the files are sub-100307.
        names = [str(s).strip() for s in subjects]
        wanted = {s if s.startswith("sub-") else f"sub-{s}" for s in names}
        known = {entry["subject"] for entry in entries}
        if not wanted <= known:
            raise ValueError(
                f"unknown subjects {sorted(wanted - known)}; the cohort has {len(known)} subjects"
            )
        chosen = [e for e in entries if e["subject"] in wanted]
    # One modality may come as a plain string, as one subject may; tuple("sc")
    # would read it letter by letter.
    modalities = (modalities,) if isinstance(modalities, str) else tuple(modalities)
    for modality in modalities:
        if modality not in ("sc", "fc"):
            raise ValueError(f"modalities are 'sc' and 'fc', got {modality!r}")
    held = sorted({modality for entry in entries for modality in entry["files"]})
    if held and not set(modalities) & set(held):
        raise ValueError(
            f"the {manifest.get('cohort', 'requested')} cohort has no {' or '.join(modalities)} "
            f"files; it holds {' and '.join(held)}"
        )
    return chosen, modalities


def _write_manifest_copy(manifest: dict, out: Path, report) -> None:
    """``manifest.csv`` beside the files: every subject with a file here, and its demographics.

    Rewritten as subjects arrive; a table of another making under that name is left alone.
    """
    entries = manifest["subjects"]
    columns = ["subject"] + [c for c in ("sex", "age_bin") if any(c in e for e in entries)]
    present = [
        e for e in entries if any((out / f"{e['subject']}_{m}.h5").exists() for m in e["files"])
    ]
    header = ",".join(columns) + "\n"
    text = header + "".join(",".join(str(e.get(c, "")) for c in columns) + "\n" for e in present)
    path = out / "manifest.csv"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == text:
            return
        if not old.startswith(header):
            report(f"  {path} is a table sbci did not write; left as it is")
            return
    path.write_text(text, encoding="utf-8")


def _write_data_use(manifest: dict, out: Path, report) -> None:
    """``DATA_USE.txt`` beside the files: the terms, and the dataset they came from."""
    terms = manifest.get("data_use")
    if not terms:
        return
    release = manifest.get("release") or {}
    if release.get("dataset"):
        doi = release["dataset"].removeprefix("doi:")
        terms += (
            f"\nThese files are from version {release.get('version')} of the dataset "
            f"https://doi.org/{doi}, whose README.md and PROCESSING.md describe them.\n"
        )
        if release.get("citation"):
            terms += f"Cite it as: {release['citation']}\n"
    notice = out / "DATA_USE.txt"
    if not notice.exists() or notice.read_text(encoding="utf-8") != terms:
        notice.write_text(terms, encoding="utf-8")
    report(f"  these data come with terms of use: read {notice}")


# --- the transfer --------------------------------------------------------------------------


def _transfer(opener, destination: Path, size: int, name: str, where: str) -> None:
    """Bytes 0 to ``size - 1`` of a file into ``destination``, taken up where a try stopped.

    ``opener(first, last)`` returns a response holding those bytes of the file, inclusive,
    checked to be those. A connection cut part way may raise ``IncompleteRead``, an
    HTTPException and no OSError, or end the response early without a word; either way the
    next try takes up from the bytes that came, up to :data:`ATTEMPTS` tries.
    """
    from http.client import HTTPException
    from ssl import SSLError
    from urllib.error import URLError

    last = None
    for attempt in range(1, ATTEMPTS + 1):
        have = destination.stat().st_size if destination.exists() else 0
        if have > size:
            destination.unlink()
            have = 0
        if have == size:
            return
        try:
            with opener(have, size - 1) as response:
                with open(destination, "ab") as handle:
                    while block := response.read(CHUNK):
                        handle.write(block)
        except (HTTPException, URLError, TimeoutError, ConnectionError, SSLError) as error:
            last = error
            if attempt < ATTEMPTS:
                time.sleep(2 * attempt)
            continue
        got = destination.stat().st_size
        if got == size:
            return
        if got > size:
            destination.unlink()
            raise OSError(f"{where} sent more than the {size} bytes {name} should have")
    reason = f" ({type(last).__name__}: {last})" if last else ""
    raise OSError(
        f"the connection broke {ATTEMPTS} times while {where} was sending "
        f"{name}{reason}; run the download again to take up where it stopped"
    ) from last


def _ranged(url: str, first: int, last: int, name: str):
    """Bytes ``first`` to ``last`` of the file at ``url``, checked to be those, or refused."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from . import __version__

    host = url.split("/")[2]
    headers = {"User-Agent": f"sbci/{__version__}", "Range": f"bytes={first}-{last}"}
    try:
        response = urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as error:
        raise OSError(f"{host} answered {error.code} for {name}") from None
    if response.status == 206:
        found = re.match(r"bytes (\d+)-(\d+)/", response.headers.get("Content-Range", ""))
        if found and (int(found.group(1)), int(found.group(2))) == (first, last):
            return response
    elif response.status == 200 and first == 0:
        return response  # the whole file, which is what was asked for
    response.close()
    raise OSError(
        f"{host} sent other bytes of {name} than {first} to {last}; it does not serve parts "
        "of files"
    )


class _Dataverse:
    """Reads files of one published dataset version, whole or by byte range, without a token."""

    def __init__(self, release: dict, archives: dict, report: Callable[[str], None]):
        self.server = release["server"].rstrip("/")
        self.archives = archives
        self.report = report
        self.signed: dict = {}

    def _get(self, url, start=None, end=None, data=None, json_body=False):
        from urllib.request import Request, urlopen

        from . import __version__

        headers = {"User-Agent": f"sbci/{__version__}"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
        if json_body:
            headers["Content-Type"] = "application/json"
        return urlopen(Request(url, data=data, headers=headers), timeout=120)

    def _call(self, url: str, body: dict | None = None) -> dict:
        """The ``data`` of the API's JSON answer to a GET, or to a POST of ``body``."""
        from urllib.error import HTTPError

        data = None if body is None else json.dumps(body).encode("utf-8")
        try:
            with self._get(url, data=data, json_body=body is not None) as response:
                return json.loads(response.read().decode("utf-8")).get("data") or {}
        except HTTPError as error:
            text = error.read().decode("utf-8", "replace")[:300]
            raise OSError(f"{self.server} answered {error.code} to {url}: {text}") from None

    def _guestbook_answers(self, guestbook: int) -> list:
        """Each required question of the guestbook answered "I agree"; refused if it asks more."""
        book = self._call(f"{self.server}/api/guestbooks/{guestbook}")
        asked = [
            name
            for name in ("name", "email", "institution", "position")
            if book.get(f"{name}Required")
        ]
        if asked:
            raise OSError(
                f"the dataset's guestbook asks each downloader for {', '.join(asked)}, which "
                "sbci does not send; download in a browser instead"
            )
        return [
            {"id": question["id"], "value": ANSWER}
            for question in book.get("customQuestions", [])
            if question.get("required")
        ]

    def _signed_url(self, file_id: int, guestbook: int) -> str:
        """A download address the server signs once the guestbook is answered (Dataverse 6.10)."""
        answers = self._guestbook_answers(guestbook)
        url = f"{self.server}/api/access/datafile/{file_id}"
        data = self._call(url, {"guestbookResponse": {"answers": answers}})
        signed = data.get("signedUrl") or data.get("url")
        if not signed:
            raise OSError(f"{self.server} answered the guestbook but gave no download address")
        return signed

    def open(self, archive: str, start: int | None = None, end: int | None = None):
        """The archive's bytes from ``start`` to ``end`` (inclusive), or the whole archive."""
        from urllib.error import HTTPError

        file_id = self.archives[archive]["id"]
        plain = f"{self.server}/api/access/datafile/{file_id}"
        signed_now = False
        while True:
            url = self.signed.get(file_id) or plain
            try:
                response = self._get(url, start, end)
                break
            except HTTPError as error:
                text = error.read().decode("utf-8", "replace")[:300]
                if url != plain:
                    del self.signed[file_id]
                    if signed_now:
                        raise OSError(
                            f"{self.server} took the guestbook's answer for {archive} but "
                            f"refused the download address it signed ({error.code} {text})"
                        ) from None
                    continue  # signed for an earlier request, and expired since: sign anew
                found = re.search(r"Guestbook response for guestbookID (\d+)", text)
                if error.code != 400 or not found or signed_now:
                    raise OSError(
                        f"{self.server} answered {error.code} for {archive}: {text}"
                    ) from None
                self.signed[file_id] = self._signed_url(file_id, int(found.group(1)))
                signed_now = True
        if start is not None:
            if response.status != 206:
                response.close()
                raise OSError(
                    f"{self.server} sent all of {archive} where only bytes {start}-{end} were "
                    "asked for; it does not serve parts of files"
                )
            found = re.match(r"bytes (\d+)-(\d+)/", response.headers.get("Content-Range", ""))
            if not found or (int(found.group(1)), int(found.group(2))) != (start, end):
                response.close()
                raise OSError(
                    f"{self.server} sent bytes "
                    f"{response.headers.get('Content-Range', '(unsaid)')} of {archive}, not "
                    f"{start}-{end}"
                )
        return response

    def fetch(self, source: dict, destination: Path, report: Callable[[str], None]) -> None:
        """One file's bytes into ``destination``, taken up where an earlier try stopped."""
        offset = int(source["offset"])
        _transfer(
            lambda first, last: self.open(source["archive"], offset + first, offset + last),
            destination,
            int(source["bytes"]),
            destination.name.removesuffix(".part"),
            self.server,
        )

    def stream(self, archive: str, members: dict, report: Callable[[str], None]) -> list:
        """Read ``archive`` whole, writing each member named in ``members`` as it passes.

        ``members[name] = (destination, sha256)``; each is written beside its destination,
        verified, and moved onto it. Returns the destinations written. A stream that breaks
        is reported and ends there: the files that came are kept, and the caller fetches the
        rest by range, the one cut short from the bytes of it that came.
        """
        from http.client import HTTPException
        from ssl import SSLError
        from urllib.error import URLError

        written = []
        try:
            with self.open(archive) as response:
                with tarfile.open(fileobj=response, mode="r|") as stream:
                    for info in stream:
                        if info.name not in members:
                            continue
                        destination, digest = members[info.name]
                        partial = destination.with_name(destination.name + ".part")
                        source = stream.extractfile(info)
                        with open(partial, "wb") as handle:
                            while block := source.read(CHUNK):
                                handle.write(block)
                        _verify(partial, digest, destination.name)
                        partial.replace(destination)
                        written.append(destination)
        except (
            HTTPException,
            URLError,
            TimeoutError,
            ConnectionError,
            SSLError,
            tarfile.TarError,
        ) as error:
            report(
                f"  the connection broke while {self.server} was sending {archive} whole "
                f"({type(error).__name__}), after {len(written)} of its {len(members)} files; "
                "the rest come one by one"
            )
        return written


def _verify(destination: Path, digest: str, name: str | None = None) -> None:
    """Remove ``destination`` and raise when its SHA-256 is not ``digest``; ``name`` is shown."""
    found = sha256_of(destination)
    if found != digest:
        destination.unlink(missing_ok=True)
        raise OSError(
            f"{name or destination.name}: SHA-256 {found[:16]}... does not match the "
            f"{digest[:16]}... recorded for it; the download was removed. Try again, and if it "
            "repeats the published file has changed."
        )


def fetch_cohort(
    out,
    cohort: str = "hcp-ya",
    subjects: Iterable[str] | str | int | None = None,
    modalities: Iterable[str] = ("sc", "fc"),
    force: bool = False,
    fetcher: Callable[[dict, Path, Callable[[str], None]], None] | None = None,
    report: Callable[[str], None] = print,
    accept_terms: bool = False,
) -> list[Path]:
    """Download HCP Young Adult connectomes into ``out`` and verify every file.

    Parameters
    ----------
    out
        Destination directory; created if needed.
    cohort
        ``"hcp-ya"``, the 946 HCP Young Adult subjects on UNC Dataverse.
    subjects
        ``None`` for the eleven subjects of the tutorial; subject ids to fetch
        (``sub-100307``, or the bare HCP id ``100307``), any of the 946; or ``"all"``
        for every subject, 85 GB.
    modalities
        ``"sc"``, ``"fc"``, or both. Asking only for a modality the cohort does
        not hold raises ``ValueError``.
    force
        Re-download files that are already present and correct.
    fetcher
        ``fetcher(source, destination, report)`` fetching one file, where ``source``
        names its archive, offset and size. Tests pass their own; with one given, every
        file is fetched by itself, no archive read whole.
    report
        Where progress lines go.
    accept_terms
        Accept the cohort's terms of use (the WU-Minn HCP Consortium Open Access Data
        Use Terms) on this machine before downloading, as :func:`accept_terms` does. They
        need accepting once; without that, :class:`TermsNotAccepted` is raised.

    Returns
    -------
    The paths written or found, in the index's order, SC before FC. A file whose
    digest does not match the index is removed and reported as an error. Beside
    them go ``manifest.csv`` (the subjects with a file here, with sex and age band)
    and ``DATA_USE.txt``, the terms.
    """
    manifest = load_manifest(cohort)
    if not manifest["subjects"]:
        raise RuntimeError(f"the {cohort} cohort has not been released yet")
    chosen, modalities = _check_request(manifest, subjects, modalities)
    for entry in chosen:
        for record in entry["files"].values():
            if not record.get("archive") or record.get("offset") is None:
                raise RuntimeError(
                    f"the index has no place for {entry['subject']}'s files yet; the cohort "
                    "has not been released"
                )
    _require_terms(manifest, cohort, accept_terms)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    release = manifest.get("release", {})
    transfer = None if fetcher else _Dataverse(release, manifest.get("archives", {}), report)
    try:
        written = _fetch_files(manifest, chosen, out, modalities, force, fetcher, transfer, report)
    finally:
        _write_manifest_copy(manifest, out, report)
    _write_data_use(manifest, out, report)
    return written


def _fetch_files(manifest, chosen, out, modalities, force, fetcher, transfer, report):
    """Every file asked for, by archive: read whole when all of it is wanted and none is here."""
    wanted = []  # (entry, modality, record, destination), in the index's order
    for entry in chosen:
        for modality in modalities:
            record = entry["files"].get(modality)
            if record is not None:
                wanted.append((entry, modality, record, out / f"{entry['subject']}_{modality}.h5"))

    def present(record, destination):
        return destination.exists() and not force and sha256_of(destination) == record["sha256"]

    done = {}
    by_archive = {}
    for entry, modality, record, destination in wanted:
        if present(record, destination):
            report(f"  {destination.name}: present and verified")
            done[destination] = True
        else:
            by_archive.setdefault(record["archive"], []).append(
                (entry, modality, record, destination)
            )
    if transfer is not None:
        held = {}
        for entry in manifest["subjects"]:
            for modality, record in entry["files"].items():
                held.setdefault(record["archive"], []).append((entry["subject"], modality))
        for archive, items in list(by_archive.items()):
            if len(items) < len(held.get(archive, ())):
                continue  # only some of it is wanted, or some is here: its files one by one
            size = manifest.get("archives", {}).get(archive, {}).get("bytes", 0)
            report(f"  reading {archive} whole ({int(size) / 1e9:.2f} GB): its {len(items)} files")
            members = {
                f"{modality}/{entry['subject']}_{modality}.h5": (destination, record["sha256"])
                for entry, modality, record, destination in items
            }
            for destination in transfer.stream(archive, members, report):
                done[destination] = True
            # what the stream did not give, if it broke or ended early, comes by range below
    fetch = fetcher or transfer.fetch
    for items in by_archive.values():
        for entry, modality, record, destination in items:
            if destination in done:
                continue
            report(
                f"  fetching {destination.name} ({int(record['bytes']) / 1e6:.1f} MB, in "
                f"{record['archive']})"
            )
            # Fetched beside the destination and moved onto it once verified: an
            # interrupted download leaves its part, which the next try takes up, and
            # force=True does not truncate a good copy before a byte has arrived.
            partial = destination.with_name(destination.name + ".part")
            if force:
                partial.unlink(missing_ok=True)
            source = dict(record, subject=entry["subject"], modality=modality)
            fetch(source, partial, report)
            _verify(partial, record["sha256"], destination.name)  # removed if it is wrong
            partial.replace(destination)
            done[destination] = True
    missing = [d.name for _e, _m, _r, d in wanted if d not in done]
    if missing:
        raise OSError(f"{len(missing)} file(s) did not come: {missing[:3]}; run the download again")
    return [destination for _e, _m, _r, destination in wanted]


# --- the Laplace-Beltrami basis --------------------------------------------------------------


def basis_dir() -> Path:
    """Where :func:`fetch_basis` puts the basis: ``$SBCI_LBO_DIR``, else ``~/.cache/sbci/lbo``.

    :func:`sbci.smoothing.find_basis` looks in both, so a basis fetched there is found again.
    """
    given = os.environ.get("SBCI_LBO_DIR")
    return Path(given) if given else Path.home() / ".cache" / "sbci" / "lbo"


def fetch_basis(out=None, force: bool = False, report: Callable[[str], None] = print) -> Path:
    """Fetch the Laplace-Beltrami basis the ``rdk`` and ``matern`` kernels need; its folder.

    The two files, ``EV_LBO_ds_ico4_L.mat`` and ``EV_LBO_ds_ico4_R.mat``, 50 MB each, come
    from the SBCI group's SBCI_Toolkit on GitHub at commit :data:`BASIS_COMMIT`, and each is
    checked against its SHA-256 (:data:`BASIS_FILES`). Files already there and correct are
    kept, so it can be re-run; a transfer cut part way is taken up where it stopped.

    Parameters
    ----------
    out
        The folder to put them in; by default :func:`basis_dir`, where ``smooth()`` looks.
    force
        Fetch them again even if present and correct.
    report
        Where progress lines go.
    """
    out = basis_dir() if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in BASIS_FILES.items():
        destination = out / name
        if (
            not force
            and destination.is_file()
            and destination.stat().st_size == size
            and sha256_of(destination) == digest
        ):
            continue
        url = BASIS_URL.format(commit=BASIS_COMMIT, name=name)
        report(
            f"  fetching {name} ({size / 1e6:.0f} MB) from SBCI_Toolkit on GitHub, commit "
            f"{BASIS_COMMIT[:7]}, into {out}"
        )
        partial = destination.with_name(name + ".part")
        if force:
            partial.unlink(missing_ok=True)
        _transfer(
            lambda first, last, url=url, name=name: _ranged(url, first, last, name),
            partial,
            size,
            name,
            url.split("/")[2],
        )
        _verify(partial, digest, name)
        partial.replace(destination)
    return out
