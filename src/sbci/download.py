"""Fetching the released example cohort: eleven HCP Young Adult subjects.

The files are too large for the repository (40 to 56 MB each), so the
repository carries a manifest instead: ``data/hcp_ya.json`` lists every
subject's SC and FC files with their Google Drive ids, sizes and SHA-256s, and each
subject's sex and open-access age band. :func:`fetch_cohort` downloads what is
asked for, verifies every file against the manifest, and skips files already
present and correct, so it can be re-run; ``sbci download hcp-ya`` is the
command form. The files are redistributed under the WU-Minn HCP Open Access
Data Use Terms: the manifest carries the note that points to them, and every
download writes it beside the files as ``DATA_USE.txt``.

Google Drive serves a public file at ``drive.usercontent.google.com`` with
``confirm=t``; for files it will not scan for viruses it answers with an HTML
page carrying a ``uuid`` token, which is read and sent back once.

A cohort too large for one file per subject can be released as zip bundles on
Zenodo, whose records take at most a hundred files: a manifest with
``host: zenodo`` lists each bundle's URL, size and SHA-256 and, for every file,
its bundle and its own digest, and :func:`fetch_cohort` then fetches only the
bundles the requested subjects need, resuming an interrupted one with a range
request (``tools/bundle_hcp_cohort.py`` writes such bundles and manifests).
No cohort is released that way at present.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from collections.abc import Callable, Iterable
from importlib import resources
from pathlib import Path

COHORTS = ("hcp-ya",)
MANIFESTS = {"hcp-ya": "hcp_ya.json"}
DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
CHUNK = 1 << 20


def load_manifest(cohort: str = "hcp-ya") -> dict:
    """The cohort's manifest as shipped with the package."""
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


def drive_url(file_id: str, uuid: str | None = None) -> str:
    """The direct-download URL of a public Google Drive file."""
    url = DRIVE_URL.format(id=file_id)
    return f"{url}&uuid={uuid}" if uuid else url


def _fetch_drive(file_id: str, destination: Path, report: Callable[[str], None]) -> None:
    """Download one Drive file to ``destination``, following the virus-scan page once."""
    from urllib.request import Request, urlopen

    uuid = None
    for _attempt in range(2):
        request = Request(drive_url(file_id, uuid), headers={"User-Agent": "sbci"})
        with urlopen(request, timeout=120) as response:
            kind = response.headers.get("Content-Type", "")
            if kind.startswith("text/html"):
                page = response.read().decode("utf-8", "replace")
                found = re.search(r'name="uuid"\s+value="([^"]+)"', page)
                if found is None or uuid is not None:
                    raise OSError(
                        f"Google Drive did not serve file {file_id}: it answered with a page "
                        "instead of the file. Is the file shared with anyone who has the link?"
                    )
                uuid = found.group(1)
                continue
            total = response.headers.get("Content-Length")
            with open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle, CHUNK)
            report(
                f"    {destination.name}: {destination.stat().st_size / 1e6:.1f} MB"
                + (f" of {int(total) / 1e6:.1f}" if total else "")
            )
            return
    raise OSError(f"Google Drive kept answering with a page for file {file_id}")


def _fetch_https(url: str, destination: Path, report: Callable[[str], None]) -> None:
    """Download ``url`` to ``destination``, resuming a partial file with a range request.

    A connection that closes before ``Content-Length`` is reached leaves the
    ``.part`` file where it is and raises, so the next call resumes it; only a
    complete download takes the final name. Renamed short, the file would be
    found wrong by the digest check, removed, and the download started over.
    """
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "sbci"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        response = urlopen(Request(url, headers=headers), timeout=120)
    except HTTPError as error:
        if error.code == 416 and have:  # the partial file is already complete
            partial.replace(destination)
            return
        raise
    with response:
        resumed = response.status == 206
        if have and not resumed:
            have = 0
        total = response.headers.get("Content-Length")
        expected = have + int(total) if total else None
        with open(partial, "ab" if resumed else "wb") as handle:
            done = have
            last = -1
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
                done += len(block)
                if expected and done * 20 // expected != last:
                    last = done * 20 // expected
                    report(f"    {destination.name}: {done / 1e6:.0f} of {expected / 1e6:.0f} MB")
    if expected is not None and done < expected:
        raise OSError(
            f"{destination.name}: the connection closed after {done / 1e6:.1f} of "
            f"{expected / 1e6:.1f} MB; the partial file is kept and the next call resumes it"
        )
    partial.replace(destination)


def _verify(destination: Path, digest: str) -> None:
    """Remove ``destination`` and raise when its SHA-256 is not ``digest``."""
    found = sha256_of(destination)
    if found != digest:
        destination.unlink(missing_ok=True)
        raise OSError(
            f"{destination.name}: SHA-256 {found[:16]}... does not match the manifest's "
            f"{digest[:16]}...; the download was removed. Try again, and if it repeats the "
            "hosted file has changed."
        )


def _check_request(manifest: dict, subjects, modalities) -> tuple[set | None, tuple]:
    if isinstance(subjects, str):
        subjects = [subjects]
    # HCP ids are often written bare (100307); the files are named sub-100307.
    wanted = (
        None if subjects is None else {s if s.startswith("sub-") else f"sub-{s}" for s in subjects}
    )
    known = {entry["subject"] for entry in manifest["subjects"]}
    if wanted is not None and not wanted <= known:
        raise ValueError(
            f"unknown subjects {sorted(wanted - known)}; the cohort has {len(known)} subjects"
        )
    # One modality may come as a plain string, as one subject may; tuple("sc")
    # would read it letter by letter.
    modalities = (modalities,) if isinstance(modalities, str) else tuple(modalities)
    for modality in modalities:
        if modality not in ("sc", "fc"):
            raise ValueError(f"modalities are 'sc' and 'fc', got {modality!r}")
    held = sorted({modality for entry in manifest["subjects"] for modality in entry["files"]})
    if held and not set(modalities) & set(held):  # an unreleased cohort lists no files
        raise ValueError(
            f"the {manifest.get('cohort', 'requested')} cohort has no {' or '.join(modalities)} "
            f"files; it holds {' and '.join(held)}"
        )
    return wanted, modalities


def _write_manifest_copy(manifest: dict, out: Path) -> None:
    """``manifest.csv`` beside the files: the subjects and whatever demographics the cohort has."""
    manifest_copy = out / "manifest.csv"
    if not manifest_copy.exists():
        subjects = manifest["subjects"]
        columns = ["subject"] + [c for c in ("sex", "age_bin") if any(c in e for e in subjects)]
        with open(manifest_copy, "w", encoding="utf-8") as handle:
            handle.write(",".join(columns) + "\n")
            for entry in subjects:
                handle.write(",".join(str(entry.get(c, "")) for c in columns) + "\n")


def fetch_cohort(
    out,
    cohort: str = "hcp-ya",
    subjects: Iterable[str] | None = None,
    modalities: Iterable[str] = ("sc", "fc"),
    force: bool = False,
    fetcher: Callable[[str, Path, Callable[[str], None]], None] | None = None,
    report: Callable[[str], None] = print,
    keep_bundles: bool = False,
) -> list[Path]:
    """Download a cohort (or some of its subjects) into ``out`` and verify every file.

    Parameters
    ----------
    out
        Destination directory; created if needed.
    cohort
        ``"hcp-ya"``, the eleven HCP Young Adult subjects on Google Drive.
    subjects
        Subject ids to fetch (``sub-100307``, or the bare HCP id ``100307``),
        or ``None`` for the whole cohort.
    modalities
        ``"sc"``, ``"fc"``, or both; a subject without one of them gets the
        other. Asking only for a modality the cohort does not hold raises
        ``ValueError``.
    force
        Re-download files that are already present and correct.
    fetcher
        ``fetcher(source, destination, report)`` doing the transfer, where
        ``source`` is a Drive file id, or a bundle URL for a cohort released in
        bundles. Tests pass their own.
    report
        Where progress lines go.
    keep_bundles
        For a cohort released in bundles, keep the zips next to the files
        instead of removing them once their files are out.

    Returns
    -------
    The paths written or found, in manifest order. A file whose digest does
    not match the manifest is removed and reported as an error. Beside them go
    ``manifest.csv`` (the subjects, with sex and age band where the cohort has
    them) and, for a cohort that comes with terms of use, ``DATA_USE.txt``.
    """
    manifest = load_manifest(cohort)
    if not manifest["subjects"]:
        raise RuntimeError(f"the {cohort} cohort has not been released yet")
    wanted, modalities = _check_request(manifest, subjects, modalities)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if manifest.get("host") == "zenodo":
        written = _fetch_bundled(
            manifest, out, wanted, modalities, force, fetcher or _fetch_https, report, keep_bundles
        )
    else:
        written = _fetch_files(
            manifest, out, wanted, modalities, force, fetcher or _fetch_drive, report
        )
    _write_manifest_copy(manifest, out)
    _write_data_use(manifest, out, report)
    return written


def _write_data_use(manifest: dict, out: Path, report) -> None:
    """``DATA_USE.txt`` beside the files: the terms the data are redistributed under."""
    terms = manifest.get("data_use")
    if not terms:
        return
    notice = out / "DATA_USE.txt"
    if not notice.exists() or notice.read_text(encoding="utf-8") != terms:
        notice.write_text(terms, encoding="utf-8")
    report(f"  these data come with terms of use: read {notice}")


def _fetch_files(manifest, out, wanted, modalities, force, fetcher, report) -> list[Path]:
    """The example cohort: one hosted file per subject and modality."""
    written = []
    for entry in manifest["subjects"]:
        subject = entry["subject"]
        if wanted is not None and subject not in wanted:
            continue
        for modality in modalities:
            record = entry["files"].get(modality)
            if record is None:
                continue
            if not record.get("drive_id"):
                raise RuntimeError(
                    f"the manifest has no Google Drive id for {subject} {modality} yet; "
                    "the cohort has not been released"
                )
            destination = out / f"{subject}_{modality}.h5"
            if destination.exists() and not force and sha256_of(destination) == record["sha256"]:
                report(f"  {destination.name}: present and verified")
                written.append(destination)
                continue
            report(f"  fetching {destination.name}")
            fetcher(record["drive_id"], destination, report)
            _verify(destination, record["sha256"])
            written.append(destination)
    return written


def _fetch_bundled(manifest, out, wanted, modalities, force, fetcher, report, keep) -> list[Path]:
    """The full cohort: zip bundles, each holding one modality of a run of subjects."""
    if not manifest.get("record"):
        raise RuntimeError(
            "the manifest names no Zenodo record yet; the cohort has not been released"
        )
    bundles = {bundle["name"]: bundle for bundle in manifest["bundles"]}
    # Which files are asked for, in manifest order, and which bundles they need.
    asked: list[tuple[str, str, dict]] = []
    for entry in manifest["subjects"]:
        if wanted is not None and entry["subject"] not in wanted:
            continue
        for modality in modalities:
            record = entry["files"].get(modality)
            if record is not None:
                asked.append((entry["subject"], modality, record))
    written: dict[tuple[str, str], Path] = {}
    pending: dict[str, list[tuple[str, str, dict]]] = {}
    for subject, modality, record in asked:
        destination = out / f"{subject}_{modality}.h5"
        if destination.exists() and not force and sha256_of(destination) == record["sha256"]:
            report(f"  {destination.name}: present and verified")
            written[(subject, modality)] = destination
        else:
            pending.setdefault(record["bundle"], []).append((subject, modality, record))
    for name in sorted(pending):
        bundle = bundles[name]
        archive = out / name
        if archive.exists() and sha256_of(archive) == bundle["sha256"]:
            report(f"  {name}: present and verified")
        else:
            report(f"  fetching {name} ({bundle['bytes'] / 1e9:.2f} GB)")
            fetcher(bundle["url"], archive, report)
            _verify(archive, bundle["sha256"])
        with zipfile.ZipFile(archive) as opened:
            for subject, modality, record in pending[name]:
                member = f"{subject}_{modality}.h5"
                destination = out / member
                with opened.open(member) as source, open(destination, "wb") as handle:
                    shutil.copyfileobj(source, handle, CHUNK)
                _verify(destination, record["sha256"])
                report(f"    {member}: extracted and verified")
                written[(subject, modality)] = destination
        if not keep:
            archive.unlink()
    return [written[(subject, modality)] for subject, modality, _ in asked]
