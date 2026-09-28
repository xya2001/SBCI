"""Fetching the example cohort: ten HCP-Aging subjects hosted on a public Google Drive.

The files are too large for the repository (40 to 56 MB each, about a
gigabyte for the cohort), so the repository carries a manifest instead:
``data/hcp_aging.json`` lists every subject's SC and FC file with its Google
Drive id, size and SHA-256. :func:`fetch_cohort` downloads what is asked for,
verifies each file against the manifest, and skips files already present and
correct, so it can be re-run. ``sbci download hcp-aging`` is the command form.

Google Drive serves a public file at ``drive.usercontent.google.com`` with
``confirm=t``; for files it will not scan for viruses it answers with an HTML
page carrying a ``uuid`` token, which is read and sent back once.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable
from importlib import resources
from pathlib import Path

COHORTS = ("hcp-aging",)
MANIFESTS = {"hcp-aging": "hcp_aging.json"}
DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
CHUNK = 1 << 20


def load_manifest(cohort: str = "hcp-aging") -> dict:
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


def fetch_cohort(
    out,
    cohort: str = "hcp-aging",
    subjects: Iterable[str] | None = None,
    modalities: Iterable[str] = ("sc", "fc"),
    force: bool = False,
    fetcher: Callable[[str, Path, Callable[[str], None]], None] | None = None,
    report: Callable[[str], None] = print,
) -> list[Path]:
    """Download the cohort (or some subjects) into ``out`` and verify every file.

    Parameters
    ----------
    out
        Destination directory; created if needed.
    cohort
        ``"hcp-aging"``.
    subjects
        Subject ids to fetch (``sub-HCA...``), or ``None`` for all ten.
    modalities
        ``"sc"``, ``"fc"``, or both.
    force
        Re-download files that are already present and correct.
    fetcher
        ``fetcher(file_id, destination, report)`` doing the transfer; the
        default fetches from Google Drive. Tests pass their own.
    report
        Where progress lines go.

    Returns
    -------
    The paths written or found, in manifest order. A file whose digest does
    not match the manifest is removed and reported as an error.
    """
    manifest = load_manifest(cohort)
    fetcher = _fetch_drive if fetcher is None else fetcher
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = None if subjects is None else set(subjects)
    known = {entry["subject"] for entry in manifest["subjects"]}
    if wanted is not None and not wanted <= known:
        raise ValueError(
            f"unknown subjects {sorted(wanted - known)}; the cohort has {sorted(known)}"
        )
    modalities = tuple(modalities)
    for modality in modalities:
        if modality not in ("sc", "fc"):
            raise ValueError(f"modalities are 'sc' and 'fc', got {modality!r}")

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
            digest = sha256_of(destination)
            if digest != record["sha256"]:
                destination.unlink(missing_ok=True)
                raise OSError(
                    f"{destination.name}: SHA-256 {digest[:16]}... does not match the manifest's "
                    f"{record['sha256'][:16]}...; the download was removed. Try again, and if it "
                    "repeats the hosted file has changed."
                )
            written.append(destination)
    manifest_copy = out / "manifest.csv"
    if not manifest_copy.exists():
        with open(manifest_copy, "w", encoding="utf-8") as handle:
            handle.write("subject,sex,age_bin\n")
            for entry in manifest["subjects"]:
                handle.write(f"{entry['subject']},{entry['sex']},{entry['age_bin']}\n")
    return written
