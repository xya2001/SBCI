r"""Prepare, upload and verify the deposit of the HCP Young Adult connectomes in UNC Dataverse.

Three steps, run where the files are::

    python tools/dataverse_deposit.py prepare --sc SC_DIR --fc FC_DIR --released DRIVE_DIR \
        --subjects manifest.csv --out DEPOSIT [--per-archive 50] [--workers 8] \
        [--rewrite-documents] [--ref COMMIT]
    python tools/dataverse_deposit.py upload DEPOSIT --dataset doi:10.15139/S3/XXXXXX \
        [--part sc|fc|all] [--limit 1] [--restrict] [--through-server] [--dry-run]
    python tools/dataverse_deposit.py verify DEPOSIT --dataset doi:10.15139/S3/XXXXXX \
        [--part sc|fc|all] [--limit 1]
    python tools/dataverse_deposit.py index DEPOSIT --dataset doi:10.15139/S3/XXXXXX \
        --version 1.0 --out src/sbci/data/hcp_ya.json

``prepare`` links every subject's SC and FC file into ``DEPOSIT/sc`` and ``DEPOSIT/fc``. A
file already released on Google Drive is taken from its released copy (``--released``, which a
deposit holding any of those subjects must give), which must match the size and SHA-256 the
package's index records for the tutorial's eleven (:func:`released_manifest`), so its bytes are
the public ones; how it compares with the build directory's copy is reported. ``DEPOSIT``'s
folders may hold nothing but those links: an output folder that is, holds or lies inside a
source folder is refused, as are a ``--subjects`` table inside ``DEPOSIT`` and a ``DEPOSIT``
holding records that no prepare wrote, and only the links and prepare's own records and
archives are ever removed. Every file is checked by
:func:`sbci.validate.validate_file` and against what the build tools write: the metadata keys,
the values that say how the file was made (the kernel and bandwidth, the normalization, the
nuisance model and the runs), ``included_connections`` naming its folder, the provenance
naming its subject, and for SC as many stored endpoints as streamlines.

The files then go into tar archives in ``DEPOSIT/archives``, for a Dataverse dataset is best
kept to about a hundred files: the subjects already released, the eleven of the sbci tutorial,
in ``sc_tutorial.tar`` and ``fc_tutorial.tar``, and the rest, in order, in archives of at most
``--per-archive`` subjects each, as nearly equal as they divide, ``sc_01.tar`` holding the same
subjects as ``fc_01.tar``. The HDF5 files are compressed inside already, so the archives are
not. A member is the file's bytes under ``sc/`` or ``fc/``, its SHA-256 checked as it is
copied, with one time, owner and mode for every member, so the same files give the same
archive, byte for byte; an archive whose files have not changed since the last prepare is kept
as it is. Only the archives prepare lists in ``.archives.json`` are ever removed, and a folder
holding anything else is refused.

It writes ``subjects.csv`` (the open-access fields only, their values checked: sex and the HCP
age band), ``ARCHIVES.tsv`` (each archive's bytes, its MD5, SHA-1, SHA-256 and SHA-512, and its
subjects), ``SHA256SUMS`` (every archive's, then every HDF5 file's), ``MANIFEST.tsv`` (each HDF5
file's bytes, digests and archive), ``CHECKS.txt``, and the documents in ``tools/deposit/``
with the counts filled in; the README names the commit of the checkout prepare runs from,
which must be pushed before the dataset is published (outside a checkout, ``--ref`` names it,
and without either prepare refuses before it changes anything). The records are removed first
and written whole at the end, ``CHECKS.txt`` last, so a failed run leaves no deposit that upload
takes. A document edited since ``prepare`` wrote it is kept unless ``--rewrite-documents`` is
given; one with no record of what prepare wrote (a deposit from before that record) is
refused until ``--rewrite-documents`` replaces it.

``upload`` adds the documents and the archives to a draft dataset, all at its top level. It
takes only a deposit whose prepare finished, with ``subjects.csv`` still the table prepare wrote
(the open-access fields alone, nothing more in any row), ``SHA256SUMS`` still what
``MANIFEST.tsv`` and ``ARCHIVES.tsv`` say, and every archive still there. Each file is hashed
first and refused if it no longer matches what ``prepare`` recorded. By default it goes by
Dataverse's direct upload, as the repository advises for a large deposit: the server hands out
addresses on its object store, the file goes there whole or, when large, in parts, and is then
registered in the draft with the checksum the server's algorithm gives (``fixityAlgorithm``).
The token goes to the Dataverse server alone, never to the store, whose addresses are signed;
a part is sent again when its transfer fails, and an upload in parts that does not finish is
aborted. ``--through-server`` sends a file through the Dataverse server instead, in one request,
for a server without direct upload and for files of a few gigabytes at most. A file the draft
already holds with the same checksum, of whatever type the installation uses, is skipped, so a
stopped run resumes where it stopped; one held with another checksum, or restricted where this
run's are not (or the other way round), stops the run. A registration that may have reached the
server before it failed is checked against the draft before the file is sent again, so a retry
adds nothing twice; a lock file in ``DEPOSIT`` keeps two uploads from running at once. A
dataset already published, with no draft, counts its published files as held, since the draft
a first file starts holds them. ``--limit 1`` sends the documents and the first archive of each
part, the tutorial's, for a trial on a scratch draft; ``--dry-run`` lists what would be sent.
The API token is read from ``DATAVERSE_API_TOKEN``, never from the command line, where others
on a shared machine could read it; it is checked to be the form Dataverse issues, sent only to
the server over https and never printed or written; a redirect is refused rather than followed.
``upload`` and ``verify`` first ask the server for the dataset, and stop on a DOI it holds no
dataset under, saying so: each server issues DOIs under a prefix of its own (UNC's demo server
under 10.33563/DMO), and Dataverse answers an unknown one with a message about a missing
``persistentId`` parameter.

``verify`` lists the draft's files and checks every name, size and checksum against
``ARCHIVES.tsv`` and the documents, and each file's restriction against ``--restrict``, and
reports any file that looks like a second copy. The dataset's terms are not checked here: set
them to the HCP's, not CC0, before publishing (``DATA_USE.txt``).

Nothing here publishes: the dataset stays a draft until its owner publishes it in the web
interface, which registers the DOI and cannot be undone.

``index``, once a version is published, writes the package's index of it, the
``data/hcp_ya.json`` that ``sbci download`` reads: for every subject's SC and FC file, its
archive, where its bytes start there (read from the archive's own tar headers), its size and
SHA-256, beside the subject's sex and age band and whether it is one of the tutorial's; and
each archive's file id in that version. It asks the repository for the version's files with
no token, as anyone may once it is published, and refuses an archive the version does not
hold with the deposit's size and checksum, so the offsets are into the bytes the repository
serves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import http.client
import json
import os
import re
import string
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PARTS = ("sc", "fc")
SERVER = "https://dataverse.unc.edu"
TOKEN = "DATAVERSE_API_TOKEN"
TOKEN_FORM = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
TEMPLATES = Path(__file__).resolve().parent / "deposit"
GENERATED = ("README.md", "DATA_USE.txt", "PROCESSING.md")
RECORDS = ("MANIFEST.tsv", "SHA256SUMS", "subjects.csv", "ARCHIVES.tsv", "CHECKS.txt")
DOCUMENTS = (
    "README.md",
    "DATA_USE.txt",
    "PROCESSING.md",
    "subjects.csv",
    "SHA256SUMS",
    "ARCHIVES.tsv",
)
GENERATED_HASHES = ".documents.json"
ARCHIVES = "archives"
ARCHIVE_RECORD = ".archives.json"
ARCHIVE_FORMAT = 1
"""Raised whenever the archives' bytes would change for the same files, so none is kept."""
ARCHIVE_TIME = 1791849600
"""2026-10-13 00:00 UTC: every member's modification time, so an archive's bytes are its files'."""
PER_ARCHIVE = 50
TUTORIAL = "tutorial"
MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".tar": "application/x-tar",
}
THROUGH_SERVER_LIMIT = 4 * 10**9
"""The largest file --through-server sends: it goes in one request, held in memory."""
OPEN_FIELDS = ("subject", "sex", "age_bin")
TERMS_URL = (
    "https://www.humanconnectome.org/study/hcp-young-adult/document/"
    "wu-minn-hcp-consortium-open-access-data-use-terms"
)
SEXES = {"F", "M"}
AGE_BANDS = {"22-25", "26-30", "31-35", "36+"}
SUBJECT = re.compile(r"^sub-\d{6}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DIGESTS = ("md5", "sha1", "sha256", "sha512")
SERVER_DIGESTS = {"MD5": "md5", "SHA-1": "sha1", "SHA-256": "sha256", "SHA-512": "sha512"}
COMMON_KEYS = {
    "area_weights",
    "container_version",
    "entry_point",
    "grid",
    "hemisphere_order",
    "included_connections",
    "mask",
    "normalization",
    "pipeline_version",
    "provenance",
    "registration_reference",
    "spec_version",
    "storage_convention",
}
METADATA_KEYS = {
    "sc": COMMON_KEYS | {"bandwidth", "kernel", "streamline_count", "streamline_weighting"},
    "fc": COMMON_KEYS | {"fc_frames", "fc_nuisance_model", "fc_runs"},
}
COMMON_VALUES = {
    "grid": "fsaverage-ico4",
    "mask": "medial-wall",
    "storage_convention": "upper-triangular-float32",
    "area_weights": "vertex-area",
}
METADATA_VALUES = {
    "sc": {
        **COMMON_VALUES,
        "kernel": "shk",
        "bandwidth": 0.005,
        "normalization": "unit-mass",
        "streamline_weighting": "none",
    },
    "fc": {**COMMON_VALUES, "normalization": "none"},
}
FC_NUISANCE = (
    "HCP minimal preprocessing and ICA-FIX (rfMRI *_Atlas_hp2000_clean); per run, the global "
    "signal, a constant and a linear trend regressed out; grid-vertex mean time series z-scored "
    "and the runs concatenated"
)
"""What ``tools/build_hcp_fc.py`` records by default: no ``--no-gsr``, no ``--bandpass``."""
FC_RUNS = {"REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"}
FRAMES_PER_RUN = 1200
ROUTES = ("barycentric", "white surface", "barycentric and white surface")
CHUNK = 1 << 22


# --- prepare ------------------------------------------------------------------------------


def read_subjects(path):
    """The subjects of a manifest: only the open-access fields, their values checked."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} lists no subjects")
    missing = [field for field in OPEN_FIELDS if field not in rows[0]]
    if missing:
        raise SystemExit(f"{path} has no {', '.join(missing)} column")
    subjects = [{field: (row[field] or "").strip() for field in OPEN_FIELDS} for row in rows]
    names = [row["subject"] for row in subjects]
    bad = [name for name in names if not SUBJECT.match(name)]
    if bad:
        raise SystemExit(f"subject ids not of the form sub-NNNNNN: {bad[:5]}")
    if len(set(names)) != len(names):
        raise SystemExit(f"{path} lists a subject twice")
    for field, allowed in (("sex", SEXES), ("age_bin", AGE_BANDS)):
        odd = sorted({row[field] for row in subjects} - allowed)
        if odd:
            raise SystemExit(
                f"{path}: {field} holds {odd[:5]}, not only the open-access values "
                f"{sorted(allowed)}; nothing restricted may be deposited"
            )
    return sorted(subjects, key=lambda row: row["subject"])


def subjects_table(subjects):
    """``subjects.csv`` as prepare writes it: the open-access fields alone, a line a subject."""
    table = ",".join(OPEN_FIELDS) + "\n"
    return table + "".join(",".join(row[field] for field in OPEN_FIELDS) + "\n" for row in subjects)


def released_manifest():
    """``{file name: (bytes, sha256)}`` of the files released before the deposit, on Drive.

    Those are the tutorial's eleven: an index of the published deposit marks them
    ``tutorial``, and lists the other 935 beside them.
    """
    try:
        from importlib.resources import files

        text = (files("sbci") / "data" / "hcp_ya.json").read_text()
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        text = (Path(__file__).resolve().parents[1] / "src/sbci/data/hcp_ya.json").read_text()
    public = {}
    for subject in json.loads(text)["subjects"]:
        if not subject.get("tutorial", True):
            continue
        for part, entry in subject["files"].items():
            public[f"{subject['subject']}_{part}.h5"] = (int(entry["bytes"]), entry["sha256"])
    return public


def sources(subjects, roots, released):
    """For each part, ``{file name: (source, built)}``; a released copy wins over the build's.

    With ``released``, every file the package's manifest of the Drive release lists must be
    there with the recorded size and SHA-256, and nothing else may be; without it, the deposit
    may hold none of those files.
    """
    public = released_manifest()
    names = {f"{row['subject']}_{part}.h5" for row in subjects for part in PARTS}
    if not released:
        shared = sorted(names & set(public))
        if shared:
            raise SystemExit(
                f"{len(shared)} of the deposit's files were released on Google Drive "
                f"({', '.join(shared[:2])}, ...): give their released copies with --released, "
                "so the deposit holds the bytes already public"
            )
        public = {}
    else:
        here = {p.name for p in Path(released).glob("sub-*.h5")}
        unknown = sorted(here - set(public))
        if unknown:
            raise SystemExit(f"{released} holds files the Drive release never had: {unknown[:5]}")
        absent = sorted(set(public) - names)
        if absent:
            raise SystemExit(f"released files of subjects not in the deposit: {absent[:5]}")
        for name, (size, sha256) in sorted(public.items()):
            copy = Path(released) / name
            if not copy.is_file():
                raise SystemExit(f"the released copy {copy} is missing")
            got = digests(copy)
            if (got["bytes"], got["sha256"]) != (size, sha256):
                raise SystemExit(
                    f"{copy} is not the released file: {got['bytes']} bytes, SHA-256 "
                    f"{got['sha256'][:16]}..., where the release recorded {size} bytes, "
                    f"{sha256[:16]}..."
                )
    found = {}
    for part in PARTS:
        found[part] = {}
        for row in subjects:
            name = f"{row['subject']}_{part}.h5"
            built = Path(roots[part]) / name
            if not built.is_file():
                raise SystemExit(f"{row['subject']} has no {part.upper()} file: {built}")
            found[part][name] = (Path(released) / name if name in public else built, built)
    return found


def check_folders(out, sources_dirs):
    """Refuse an output that could hold or touch the data: only links may live in its folders."""
    for part in (*PARTS, ARCHIVES):
        folder = (Path(out) / part).resolve()
        for source in sources_dirs:
            if source is None:
                continue
            source = Path(source).resolve()
            if folder == source or source in folder.parents or folder in source.parents:
                raise SystemExit(
                    f"{folder} is, holds or lies inside the source folder {source}; choose "
                    "another --out, since only links may live in the deposit's folders"
                )
        if part == ARCHIVES:
            continue  # check_archives: only the archives prepare wrote
        if folder.exists():
            others = [p.name for p in folder.iterdir() if not p.is_symlink()]
            if others:
                raise SystemExit(
                    f"{folder} holds files that are not links ({others[:3]}); choose another "
                    "--out, since prepare removes and rewrites the links there"
                )


def digests(path):
    """Size and the MD5, SHA-1, SHA-256 and SHA-512 of a file, in one pass."""
    hashes = {name: hashlib.new(name) for name in DIGESTS}
    size = 0
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            for value in hashes.values():
                value.update(block)
            size += len(block)
    return {"bytes": size, **{name: value.hexdigest() for name, value in hashes.items()}}


def read_metadata(path):
    """The JSON metadata an sbci HDF5 file carries."""
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["metadata"][()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def route_of(metadata):
    """The way an FC file's vertices reached the grid, read off its provenance."""
    found = re.search(r"\(([a-z ]+) route\)", str(metadata.get("provenance", "")))
    return found.group(1) if found else None


def metadata_problems(metadata, part, subject):
    """What is wrong with a file's metadata for the deposit, in words; empty when nothing is."""
    problems = []
    extra = sorted(set(metadata) - METADATA_KEYS[part])
    if extra:
        problems.append(f"metadata keys the build tools do not write: {extra}")
    if metadata.get("included_connections") != part:
        problems.append(f"included_connections is {metadata.get('included_connections')!r}")
    if f"HCP Young Adult {subject};" not in str(metadata.get("provenance", "")):
        problems.append(f"provenance does not name {subject}")
    for key, expected in METADATA_VALUES[part].items():
        if metadata.get(key) != expected:
            problems.append(f"{key} is {metadata.get(key)!r}, not {expected!r}")
    if part == "fc":
        if metadata.get("fc_nuisance_model") != FC_NUISANCE:
            problems.append(f"fc_nuisance_model is {metadata.get('fc_nuisance_model')!r}")
        runs = metadata.get("fc_runs") or []
        if not runs or not set(runs) <= FC_RUNS:
            problems.append(f"fc_runs is {runs!r}")
        frames = metadata.get("fc_frames")
        if not isinstance(frames, int) or not 0 < frames <= FRAMES_PER_RUN * len(runs or [1]):
            problems.append(f"fc_frames is {frames!r} for {len(runs)} runs")
        if route_of(metadata) not in ROUTES:
            problems.append("the provenance names no route the FC builder takes")
    return problems


def stored_streamlines(path):
    """How many streamlines an SC file's endpoints hold, or ``None`` without endpoints."""
    import h5py

    with h5py.File(path, "r") as handle:
        if "endpoints" not in handle or "vertex_in" not in handle["endpoints"]:
            return None
        return int(handle["endpoints"]["vertex_in"].shape[0])


def check_file(task):
    """Validate one file, digest it and note what PROCESSING.md reports; run in a worker."""
    part, name, path, validate = task
    subject = name.split("_")[0]
    metadata = read_metadata(path)
    problems = metadata_problems(metadata, part, subject)
    if part == "sc":
        stored = stored_streamlines(path)
        if stored is None:
            problems.append("no endpoints stored")
        elif stored != metadata.get("streamline_count"):
            problems.append(
                f"{stored} endpoints stored for {metadata.get('streamline_count')} streamlines"
            )
    if validate:
        from sbci.validate import validate_file

        problems += [str(check) for check in validate_file(path) if not check.passed]
    noted = {"spec_version": metadata.get("spec_version")}
    if part == "sc":
        noted["streamlines"] = metadata.get("streamline_count")
    else:
        noted.update(
            runs=len(metadata.get("fc_runs") or []),
            frames=metadata.get("fc_frames"),
            route=route_of(metadata),
            subject=subject,
        )
    return part, name, digests(path), problems, noted


def describe(noted):
    """What PROCESSING.md says of the streamlines, the runs and the routes, from the files."""
    from collections import Counter
    from statistics import median

    counts = sorted(int(n["streamlines"]) for n in noted["sc"])
    streamlines = f"{counts[0]:,} to {counts[-1]:,} (median {int(median(counts)):,})"
    runs = Counter(n["runs"] for n in noted["fc"])
    by_runs = "; ".join(
        f"{n} for {k:,} subject{'s' if k != 1 else ''}"
        for n, k in sorted(runs.items(), reverse=True)
    )
    full = max(runs)
    short = sum(1 for n in noted["fc"] if n["runs"] == full and n["frames"] < FRAMES_PER_RUN * full)
    runs_text = f"the number of runs is {by_runs}"
    if short:
        runs_text += (
            f". Of the {runs[full]:,} with {full}, {short:,} have a run shorter than its "
            f"{FRAMES_PER_RUN:,} frames, and each file records its frames in ``fc_frames``"
        )
    whole = sorted(n["subject"] for n in noted["fc"] if n["route"] == "white surface")
    half = sorted(n["subject"] for n in noted["fc"] if n["route"] == ROUTES[2])
    sentences = []
    if len(whole) == 1:
        sentences.append(
            f"One subject's HCP native mesh is not FreeSurfer's ({whole[0]}). It goes through "
            "the white surfaces instead."
        )
    elif whole:
        sentences.append(
            f"{len(whole)} subjects' HCP native meshes are not FreeSurfer's "
            f"({', '.join(whole)}). They go through the white surfaces instead."
        )
    if half:
        sentences.append(
            f"For {len(half)} subject{'s' if len(half) != 1 else ''} ({', '.join(half)}) "
            "one hemisphere's is not, and that hemisphere goes through the white surface."
        )
    return streamlines, runs_text, " ".join(sentences)


def compare_copies(released, built):
    """How a released file differs from the build directory's copy of it, in words."""
    import h5py
    import numpy as np

    if digests(released)["sha256"] == digests(built)["sha256"]:
        return "byte-identical to the build directory's"

    def datasets(handle):
        found = {}

        def keep(name, item):
            # nothing returned: visititems stops at the first value a visit returns
            if isinstance(item, h5py.Dataset):
                found[name] = item

        handle.visititems(keep)
        return found

    with h5py.File(released, "r") as a, h5py.File(built, "r") as b:
        da, db = datasets(a), datasets(b)
        if sorted(da) != sorted(db):
            return f"different datasets: {sorted(da)} against {sorted(db)}"
        words = []
        for key in sorted(da):
            x, y = np.asarray(da[key][()]), np.asarray(db[key][()])
            if np.array_equal(x, y):
                continue
            if x.dtype.kind in "fiu" and x.shape == y.shape:
                gap = np.abs(x.astype(np.float64) - y.astype(np.float64))
                words.append(
                    f"{key}: {np.count_nonzero(gap)} entries differ, by at most {gap.max():.1e}"
                )
            else:
                words.append(f"{key} differs")
    return "; ".join(words) or "the same arrays, different bytes"


def groups(subjects, tutorial, per_archive):
    """Each archive's label and subjects: the tutorial's first, then the rest in order.

    The rest go in as few archives as hold at most ``per_archive`` subjects each, as nearly
    equal as they divide; a part's archive of one label holds the same subjects as the other's.
    """
    rest = [subject for subject in subjects if subject not in tutorial]
    count = -(-len(rest) // per_archive)
    width = max(2, len(str(count)))
    labelled = [(TUTORIAL, sorted(tutorial))] if tutorial else []
    for index in range(count):
        chunk = rest[index * len(rest) // count : (index + 1) * len(rest) // count]
        labelled.append((f"{index + 1:0{width}d}", chunk))
    return labelled


class _Hashing:
    """An archive being written, hashed as it is: its digests in the one pass."""

    def __init__(self, handle):
        self.handle = handle
        self.size = 0
        self.hashes = {name: hashlib.new(name) for name in DIGESTS}

    def write(self, data):
        self.handle.write(data)
        for value in self.hashes.values():
            value.update(data)
        self.size += len(data)
        return len(data)

    def tell(self):
        return self.size

    def flush(self):
        self.handle.flush()

    def digest(self):
        return {
            "bytes": self.size,
            **{name: value.hexdigest() for name, value in self.hashes.items()},
        }


class _Checked:
    """A member's bytes as the archive takes them, hashed to compare with the manifest's."""

    def __init__(self, handle):
        self.handle = handle
        self.sha256 = hashlib.sha256()

    def read(self, size=-1):
        data = self.handle.read(size)
        self.sha256.update(data)
        return data


def build_archive(task):
    """Write one archive: its members in order, every header the same whoever owns the files.

    Each member's SHA-256 is checked as it is copied; the archive is written under a
    temporary name and moved into place whole. Run in a worker; ``(name, digest)``.
    """
    path, members = task
    temporary = Path(f"{path}.part")
    with open(temporary, "wb") as handle:
        writer = _Hashing(handle)
        with tarfile.open(fileobj=writer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for member, source, sha256 in members:
                info = tarfile.TarInfo(member)
                info.size = os.stat(source).st_size
                info.mtime = ARCHIVE_TIME
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(source, "rb") as data:
                    reading = _Checked(data)
                    archive.addfile(info, reading)
                if reading.sha256.hexdigest() != sha256:
                    raise RuntimeError(f"{source} changed since it was checked; run prepare again")
        digest = writer.digest()
    os.replace(temporary, path)
    return Path(path).name, digest


def read_archive_record(out):
    """``{archive: entry}`` of what prepare last wrote into ``out/archives``; empty with none.

    An entry holds the archive's members with their SHA-256 and its digest; one written by
    another version of the archives keeps its name, so it is still prepare's to remove, but
    not its entry, so it is not kept.
    """
    try:
        record = json.loads((Path(out) / ARCHIVE_RECORD).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    archives = record.get("archives", {})
    if record.get("format") != ARCHIVE_FORMAT:
        return {name: {} for name in archives}
    return archives


def check_archives(out):
    """Refuse an archives folder holding anything prepare did not write there."""
    folder = Path(out) / ARCHIVES
    if not folder.exists():
        return
    ours = set(read_archive_record(out))
    others = sorted(p.name for p in folder.iterdir() if p.name.removesuffix(".part") not in ours)
    if others:
        raise SystemExit(
            f"{folder} holds files no prepare wrote there ({others[:3]}); choose another --out, "
            "since prepare removes and rewrites the archives there"
        )


def make_archives(out, planned, workers):
    """Write the planned archives, keeping each already written from the same files.

    ``planned`` is ``{name: members}``, a member ``(path in the archive, source, sha256)``.
    The record names every archive the folder may hold before a byte is written, so a run
    stopped midway leaves nothing a later one cannot remove. ``(digests, built)``.
    """
    folder = Path(out) / ARCHIVES
    folder.mkdir(exist_ok=True)
    record = read_archive_record(out)
    kept, tasks = {}, []
    for name, members in planned.items():
        entry = record.get(name) or {}
        listed = [[member, sha256] for member, _, sha256 in members]
        path = folder / name
        size = (entry.get("digest") or {}).get("bytes")
        if entry.get("members") == listed and path.is_file() and path.stat().st_size == size:
            kept[name] = entry
        else:
            tasks.append((str(path), members))
    write_whole(
        Path(out) / ARCHIVE_RECORD,
        json.dumps(
            {"format": ARCHIVE_FORMAT, "archives": {**record, **{n: {} for n in planned}, **kept}}
        )
        + "\n",
    )
    for stale in folder.iterdir():  # check_archives saw only names of the record's
        if stale.name.endswith(".part") or stale.name not in kept:
            stale.unlink()
    done = dict(kept)
    pool = (
        ProcessPoolExecutor(max_workers=min(workers, len(tasks))) if workers > 1 and tasks else None
    )
    try:
        results = pool.map(build_archive, tasks) if pool else map(build_archive, tasks)
        for count, (name, digest) in enumerate(results, 1):
            listed = [[member, sha256] for member, _, sha256 in planned[name]]
            done[name] = {"members": listed, "digest": digest}
            print(
                f"  wrote {name}, {digest['bytes'] / 1e9:.2f} GB ({count} of {len(tasks)})",
                flush=True,
            )
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    finally:
        if pool:
            pool.shutdown()
    write_whole(
        Path(out) / ARCHIVE_RECORD,
        json.dumps({"format": ARCHIVE_FORMAT, "archives": {n: done[n] for n in planned}}, indent=1)
        + "\n",
    )
    return {name: done[name]["digest"] for name in planned}, len(tasks)


def archives_text(names, tutorial, per_archive):
    """The README's account of the archives, from their names."""
    words = []
    for part in PARTS:
        mine = [name for name in names if name.startswith(f"{part}_")]
        numbered = [name for name in mine if not name.endswith(f"_{TUTORIAL}.tar")]
        listed = [f"`{part}_{TUTORIAL}.tar`"] if len(numbered) < len(mine) else []
        if len(numbered) == 1:
            listed.append(f"`{numbered[0]}`")
        elif numbered:
            listed.append(f"`{numbered[0]}` to `{numbered[-1]}`")
        words.append(" and ".join(listed))
    text = f"{words[0]} for SC, {words[1]} for FC, each of at most {per_archive} subjects"
    if tutorial:
        text += (
            f". `sc_{TUTORIAL}.tar` and `fc_{TUTORIAL}.tar` hold the {len(tutorial)} subjects of "
            "the sbci package's tutorial, released earlier on their own, so they can be had "
            "without the rest"
        )
    return text


def code_version():
    """The commit of the sbci checkout this tool runs from, for the README's install line.

    ``(commit, local changes)``; ``(None, False)`` when this file is not in a git checkout of
    the package (no git, no checkout, or a checkout of something else).
    """
    here = Path(__file__).resolve().parent

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(here), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        top = git("rev-parse", "--show-toplevel")
        commit = git("rev-parse", "--verify", "HEAD")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None, False
    if Path(top).resolve() != here.parent or not COMMIT.match(commit):
        return None, False
    return commit, bool(dirty)


def fill(template, values):
    """A document from ``tools/deposit/``, with ``$name`` placeholders filled."""
    return string.Template((TEMPLATES / template).read_text()).substitute(values)


def write_whole(path, text):
    """Write a file under a temporary name and move it into place, so it is whole or absent."""
    temporary = Path(f"{path}.part")
    temporary.write_text(text)
    os.replace(temporary, path)


def sha256_of_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_generated(out):
    """``{document: sha256}`` of what prepare last wrote into ``out``; empty with no record."""
    try:
        return json.loads((Path(out) / GENERATED_HASHES).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def check_records(out, subjects_path, rewrite):
    """Refuse to remove or overwrite what prepare did not write itself."""
    out = Path(out).resolve()
    if out in Path(subjects_path).resolve().parents:
        raise SystemExit(
            f"--subjects {subjects_path} lies inside --out, where prepare removes and rewrites "
            "subjects.csv; keep the table elsewhere"
        )
    if not out.exists():
        return
    records = [record for record in RECORDS if (out / record).exists()]
    ours = any(
        (out / name).exists() for name in ("CHECKS.txt", GENERATED_HASHES, ARCHIVE_RECORD, *PARTS)
    )
    if records and not ours:
        raise SystemExit(
            f"{out} holds {records}, which no prepare wrote there; choose another --out, since "
            "prepare removes and rewrites its records"
        )
    generated = read_generated(out)
    unrecorded = [name for name in GENERATED if (out / name).exists() and name not in generated]
    if unrecorded and not rewrite:
        raise SystemExit(
            f"{out} holds {unrecorded} but no record of what prepare wrote there "
            f"({GENERATED_HASHES}), so an edit cannot be told from an older prepare's text: copy "
            "out any edit you want to keep, then run again with --rewrite-documents"
        )


def prepare(args):
    subjects = read_subjects(args.subjects)
    out = Path(args.out)
    check_folders(out, (args.sc, args.fc, args.released))
    check_records(out, args.subjects, args.rewrite_documents)
    check_archives(out)
    # the README's install line names this commit: settled before anything is removed
    if args.ref is not None:
        if not COMMIT.match(args.ref):
            raise SystemExit(f"--ref {args.ref!r} is not a commit's 40 hexadecimal digits")
        commit, dirty = args.ref, False
    else:
        commit, dirty = code_version()
        if commit is None:
            raise SystemExit(
                "this copy of tools/ is not in a git checkout of sbci, so the README could name "
                "no commit to install: run prepare from the checkout, or name the pushed commit "
                "with --ref"
            )
    found = sources(subjects, {"sc": args.sc, "fc": args.fc}, args.released)
    out.mkdir(parents=True, exist_ok=True)
    for record in RECORDS:  # a failed run must not leave old records beside new links
        (out / record).unlink(missing_ok=True)
    for part in PARTS:
        folder = out / part
        folder.mkdir(exist_ok=True)
        for stale in folder.iterdir():
            if stale.is_symlink():  # check_folders made sure nothing else is there
                stale.unlink()
        for name, (source, _) in found[part].items():
            (folder / name).symlink_to(Path(source).resolve())
    print(f"linked {len(subjects)} subjects' SC and FC into {out}", flush=True)

    released_notes = []
    for part in PARTS:
        for name, (source, built) in found[part].items():
            if Path(source) != Path(built):
                note = f"{part}/{name}: the released bytes; {compare_copies(source, built)}"
                released_notes.append(note)
                print(f"  released copy {note}", flush=True)

    tasks = [
        (part, name, str(out / part / name), not args.no_validate)
        for part in PARTS
        for name in found[part]
    ]
    rows, failed, noted = [], [], {part: [] for part in PARTS}
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = pool.map(check_file, tasks, chunksize=4) if pool else map(check_file, tasks)
    for done, (part, name, digest, problems, note) in enumerate(results, 1):
        rows.append((f"{part}/{name}", part, digest))
        noted[part].append(note)
        failed += [f"{part}/{name}: {problem}" for problem in problems]
        if done % 200 == 0 or done == len(tasks):
            print(f"  checked {done} of {len(tasks)} files", flush=True)
    if pool:
        pool.shutdown()
    versions = sorted({str(n["spec_version"]) for part in PARTS for n in noted[part]})
    if len(versions) != 1:
        failed.append(f"the files are of more than one spec_version: {versions}")
    if failed:
        print("\n".join(failed[:50]), file=sys.stderr)
        raise SystemExit(f"{len(failed)} problems; no records written, the links stay")

    rows.sort(key=lambda row: row[0])
    tutorial = sorted(
        {
            name.split("_")[0]
            for part in PARTS
            for name, (source, built) in found[part].items()
            if Path(source) != Path(built)
        }
    )
    labelled = groups([row["subject"] for row in subjects], set(tutorial), args.per_archive)
    sha256_of = {path: d["sha256"] for path, _, d in rows}
    planned, archive_of, holds = {}, {}, {}
    for part in PARTS:
        for label, chunk in labelled:
            name = f"{part}_{label}.tar"
            members = [f"{part}/{subject}_{part}.h5" for subject in chunk]
            planned[name] = [
                (member, str((out / member).resolve()), sha256_of[member]) for member in members
            ]
            archive_of.update({member: name for member in members})
            holds[name] = chunk
    print(f"writing {len(planned)} archives into {out / ARCHIVES}", flush=True)
    archived, built = make_archives(out, planned, args.workers)

    manifest = "file\tpart\tarchive\tbytes\t" + "\t".join(DIGESTS) + "\n"
    manifest += "".join(
        "\t".join([path, part, archive_of[path], str(d["bytes"]), *(d[n] for n in DIGESTS)]) + "\n"
        for path, part, d in rows
    )
    write_whole(out / "MANIFEST.tsv", manifest)
    table = "archive\tpart\tsubjects\tbytes\t" + "\t".join(DIGESTS) + "\tholds\n"
    table += "".join(
        "\t".join(
            [
                name,
                name.split("_")[0],
                str(len(holds[name])),
                str(archived[name]["bytes"]),
                *(archived[name][n] for n in DIGESTS),
                " ".join(holds[name]),
            ]
        )
        + "\n"
        for name in planned
    )
    write_whole(out / "ARCHIVES.tsv", table)
    write_whole(
        out / "SHA256SUMS",
        "".join(f"{archived[name]['sha256']}  {name}\n" for name in planned)
        + "".join(f"{d['sha256']}  {path}\n" for path, _, d in rows),
    )
    write_whole(out / "subjects.csv", subjects_table(subjects))

    sizes = {part: sum(d["bytes"] for _, p, d in rows if p == part) for part in PARTS}
    streamlines, runs, routes = describe(noted)
    values = {
        "streamlines": streamlines,
        "runs": runs,
        "routes": routes,
        "subjects": f"{len(subjects):,}",
        "files": f"{len(rows):,}",
        "sc_gb": f"{sizes['sc'] / 1e9:.1f}",
        "fc_gb": f"{sizes['fc'] / 1e9:.1f}",
        "total_gb": f"{(sizes['sc'] + sizes['fc']) / 1e9:.1f}",
        "archives": f"{len(planned)}",
        "archive_list": archives_text(list(planned), tutorial, args.per_archive),
        "example": next(iter(holds.values()))[0],
        "spec_version": versions[0],
        "sbci_ref": commit,
        "validated": (
            "Before deposit, every file was checked again with `sbci.validate`."
            if not args.no_validate
            else ""
        ),
    }
    hashes_path = out / GENERATED_HASHES
    generated = read_generated(out)
    kept = []
    for document in GENERATED:
        text = fill(document, values)
        target = out / document
        if target.exists() and not args.rewrite_documents:
            current = sha256_of_text(target.read_text())
            if current != generated.get(document) and current != sha256_of_text(text):
                kept.append(document)  # edited since prepare wrote it: the edit stays
                continue
        write_whole(target, text)
        generated[document] = sha256_of_text(text)
    write_whole(hashes_path, json.dumps(generated, indent=1) + "\n")

    lines = [
        f"{len(rows)} files of {len(subjects)} subjects; every one "
        + ("validated by sbci.validate.validate_file; " if not args.no_validate else "")
        + "its metadata keys and values, included_connections and provenance checked, and "
        "each SC file's endpoints counted",
        f"spec_version {versions[0]}; the code at {commit}"
        + (" with local changes" if dirty else ""),
        *(f"released copy {note}" for note in released_notes),
        *(f"{part}: {sizes[part]:,} bytes" for part in PARTS),
        f"{len(planned)} archives of at most {args.per_archive} subjects, "
        f"{sum(d['bytes'] for d in archived.values()):,} bytes: {built} written, "
        f"{len(planned) - built} kept from the last prepare, their files unchanged"
        + (
            f"; the tutorial's {len(tutorial)} in {PARTS[0]}_{TUTORIAL}.tar and "
            f"{PARTS[1]}_{TUTORIAL}.tar"
            if tutorial
            else ""
        ),
        f"streamlines per subject: {streamlines}",
        f"FC: {runs}",
        f"FC routes: {routes or 'barycentric for every subject'}",
        *(f"kept as edited: {document} (--rewrite-documents replaces it)" for document in kept),
    ]
    write_whole(out / "CHECKS.txt", "\n".join(lines) + "\n")
    if dirty:
        print(f"note: the checkout at {commit} has local changes; README names that commit")
    for document in kept:
        print(f"kept {document} as edited; --rewrite-documents would replace it", flush=True)
    print(
        f"wrote MANIFEST.tsv, ARCHIVES.tsv, SHA256SUMS, subjects.csv, the documents and "
        f"CHECKS.txt: {len(rows)} files in {len(planned)} archives, {values['total_gb']} GB",
        flush=True,
    )


# --- the Dataverse native API -------------------------------------------------------------


class Refused(Exception):
    """A request the server did not accept, or that could not be made; never retried."""


class Uncertain(Exception):
    """A request that failed after it may have reached the server."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the token goes to no address but the one given."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise Refused(
            f"the server answered {code}, a redirect, which is refused so that the token is "
            "sent nowhere else; check --server"
        )


def normalize_dataset(dataset):
    """``doi:10...`` from the forms a DOI is copied in: ``doi:10...``, ``10...``, a doi.org link."""
    text = dataset.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi.org/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    if text.lower().startswith("doi:"):
        text = text[4:]
    if not re.match(r"^10\.\d{4,9}/\S+$", text):
        raise SystemExit(f"--dataset {dataset!r} is not a DOI such as doi:10.15139/S3/XXXXXX")
    return f"doi:{text}"


def check_server(server):
    """The server's base URL, https unless it is this machine."""
    parsed = urllib.parse.urlsplit(server)
    if parsed.scheme != "https" and parsed.hostname not in LOCAL_HOSTS:
        raise SystemExit(f"--server must be https, so the token is never sent in clear: {server}")
    return server.rstrip("/")


class Dataverse:
    """The few native-API calls a deposit needs, against one dataset."""

    def __init__(self, server, dataset, token, retries=3, pause=10):
        self.server = server.rstrip("/")
        self.dataset = dataset
        self.token = token
        self.retries = retries
        self.pause = pause
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _url(self, path, **query):
        query = {"persistentId": self.dataset, **query}
        return f"{self.server}/api/datasets/:persistentId/{path}?{urllib.parse.urlencode(query)}"

    def _once(self, method, url, body, content_type, timeout):
        headers = {"X-Dataverse-key": self.token}
        if content_type:
            headers["Content-Type"] = content_type
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            with self._opener.open(request, timeout=timeout) as response:
                text = response.read().decode(errors="replace")
        except ValueError:  # a header the library refuses; its message would carry the token
            raise Refused(
                f"the request could not be formed; check {TOKEN}, which must be the token "
                "alone, with no spaces or line breaks"
            ) from None
        try:
            return json.loads(text or "{}")
        except ValueError:
            raise Refused(f"{method} {url.split('?')[0]}: not JSON: {text[:200]!r}") from None

    def _call(self, method, url, body=None, content_type=None, timeout=900, repeatable=True):
        """One request, retried while it may safely be repeated; ``Uncertain`` when it may not."""
        where = f"{method} {url.split('?')[0]}"
        attempt = 0
        while True:
            try:
                return self._once(method, url, body, content_type, timeout)
            except urllib.error.HTTPError as error:
                # a server that echoed the request would print the token: never let it through
                detail = error.read().decode(errors="replace")[:500].replace(self.token, "[token]")
                if error.code == 429 and attempt < self.retries + 3:
                    wait = error.headers.get("Retry-After", "") if error.headers else ""
                    time.sleep(int(wait) if wait.isdigit() else 3 * self.pause)
                    attempt += 1
                    continue
                if error.code < 500:
                    raise Refused(f"{where}: {error.code} {detail}") from None
                failure = f"{where}: {error.code} {detail}"
            except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
                # OSError: a reset, a timeout or an SSL failure, while sending or reading
                failure = f"{where}: {type(error).__name__}: {error}"
            if not repeatable:
                raise Uncertain(failure)
            if attempt >= self.retries:
                raise Refused(failure)
            attempt += 1
            time.sleep(self.pause * attempt)

    def files(self):
        """``{path: (bytes, checksum type, value, restricted)}`` of the draft.

        With no draft, those of the latest published version, since the draft the first
        file added starts holds them all; empty for a dataset with neither. ``restricted``
        is ``None`` where the listing does not say.
        """
        listing = None
        for version in (":draft", ":latest-published"):
            try:
                listing = self._call("GET", self._url(f"versions/{version}/files"))
                break
            except Refused as error:
                if " 404 " not in f"{error} ":
                    raise
        held = {}
        for entry in (listing or {}).get("data", []):
            data_file = entry.get("dataFile", {})
            checksum = data_file.get("checksum") or {}
            kind, value = checksum.get("type"), checksum.get("value")
            if not value and data_file.get("md5"):
                kind, value = "MD5", data_file["md5"]
            folder = entry.get("directoryLabel")
            name = entry.get("label") or data_file.get("filename")
            restricted = entry.get("restricted", data_file.get("restricted"))
            held[f"{folder}/{name}" if folder else name] = (
                data_file.get("filesize"),
                kind,
                value,
                None if restricted is None else bool(restricted),
            )
        return held

    def exists(self):
        """Stop, saying so, when the server holds no dataset under the DOI given.

        Dataverse answers a request about a DOI it does not know with 404 and "a persistentId
        query parameter must be present" (6.10.1, on UNC's demo server, 14 October 2026),
        which hides the cause.
        """
        query = urllib.parse.urlencode({"persistentId": self.dataset})
        try:
            self._call("GET", f"{self.server}/api/datasets/:persistentId?{query}")
        except Refused as error:
            if " 404 " not in f"{error} ":
                raise
            raise Refused(
                f"{self.server} holds no dataset {self.dataset}: copy the DOI from the "
                "dataset's own page there (its citation, or persistentId= in the address bar); "
                "each server issues DOIs under a prefix of its own"
            ) from None

    def wait_for_locks(self, patience=600):
        """Wait until the dataset carries no lock; name the locks if they outlast ``patience``."""
        waited = 0
        while True:
            locks = self._call("GET", self._url("locks")).get("data") or []
            if not locks:
                return
            if waited >= patience:
                kinds = sorted({str(lock.get("lockType", "?")) for lock in locks})
                raise Refused(f"the dataset is still locked after {patience} s: {kinds}")
            time.sleep(2)
            waited += 2

    def fixity(self):
        """The checksum type the installation registers files with: MD5 where it does not say."""
        try:
            data = self._call("GET", f"{self.server}/api/files/fixityAlgorithm").get("data")
        except Refused as error:
            if " 404 " in f"{error} ":
                return "MD5"
            raise
        algorithm = str((data or {}).get("message") or "MD5")
        if algorithm not in SERVER_DIGESTS:
            raise Refused(
                f"the server registers files with {algorithm!r} checksums, which this tool "
                "cannot compute; ask the repository"
            )
        return algorithm

    def _server_url(self, path):
        """An address the server handed back, on the server itself: the token goes nowhere else."""
        if path.startswith("/"):
            return f"{self.server}{path}"
        given, ours = urllib.parse.urlsplit(path), urllib.parse.urlsplit(self.server)
        if (given.scheme, given.netloc) != (ours.scheme, ours.netloc):
            raise Refused(
                f"the server handed back an address on another host ({given.netloc}), where "
                "the token is not sent"
            )
        return path

    def _store(self, url, handle, start, length, tag):
        """Send ``length`` bytes from ``start`` to a signed address on the object store.

        No token goes with it: the address is signed. The tag that marks an upload not yet
        registered is sent when the address was signed with it, as a single part's is when
        the installation tags; a failed transfer is sent again. The store's ETag.
        """
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" and parsed.hostname not in LOCAL_HOSTS:
            raise Refused(
                f"the server handed out a store address that is not https: {parsed.netloc}"
            )
        query = urllib.parse.parse_qs(parsed.query)
        signed = query.get("X-Amz-SignedHeaders", [""])[0].lower().split(";")
        kind = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        attempt = 0
        while True:
            connection = kind(parsed.hostname, parsed.port, timeout=900)
            try:
                connection.putrequest("PUT", target, skip_accept_encoding=True)
                connection.putheader("Content-Length", str(length))
                if tag and "x-amz-tagging" in signed:
                    connection.putheader("x-amz-tagging", "dv-state=temp")
                connection.endheaders()
                handle.seek(start)
                left = length
                while left:
                    block = handle.read(min(CHUNK, left))
                    if not block:
                        raise Refused(
                            f"{handle.name} is shorter than prepare recorded; run prepare again"
                        )
                    connection.send(block)
                    left -= len(block)
                response = connection.getresponse()
                body = response.read(2000).decode(errors="replace")
                if response.status in (200, 201):
                    return response.getheader("ETag", "")
                failure = f"PUT to the store: {response.status} {body[:200]}"
                if response.status < 500:
                    raise Refused(failure)
            except (http.client.HTTPException, OSError) as error:
                failure = f"PUT to the store: {type(error).__name__}: {error}"
            finally:
                connection.close()
            if attempt >= self.retries:
                raise Refused(failure)
            attempt += 1
            time.sleep(self.pause * attempt)

    def _abort(self, path):
        """Abandon an upload in parts that will not be finished; whatever happens, carry on."""
        if not path:
            return
        try:
            self._call("DELETE", self._server_url(path))
        except (Refused, Uncertain):
            pass

    def add_direct(self, path, name, description, category, restrict, digest, algorithm, media):
        """Send one file to the object store, whole or in parts, and register it in the draft.

        ``digest`` is what prepare recorded, the file having been checked against it just before;
        the draft records the checksum of the type the installation uses.
        """
        size = digest["bytes"]
        try:
            offer = self._call("GET", self._url("uploadurls", size=str(size))).get("data") or {}
        except Refused as error:
            raise Refused(
                f"{error}; the server may offer no direct upload: run again with "
                "--through-server, or ask the repository to enable it"
            ) from None
        storage = offer.get("storageIdentifier")
        if not storage or not (offer.get("url") or offer.get("urls")):
            raise Refused(
                "the server offered no direct upload for this dataset; run again with "
                "--through-server, or ask the repository to enable direct upload"
            )
        with open(path, "rb") as handle:
            if offer.get("url"):
                self._store(offer["url"], handle, 0, size, tag=True)
            else:
                part = int(offer["partSize"])
                tags = {}
                try:
                    for number, url in sorted(offer["urls"].items(), key=lambda item: int(item[0])):
                        start = (int(number) - 1) * part
                        tags[str(number)] = self._store(
                            url, handle, start, min(part, size - start), False
                        )
                    complete = self._server_url(offer["complete"])
                except BaseException:
                    self._abort(offer.get("abort"))
                    raise
                try:
                    self._call(
                        "PUT",
                        complete,
                        json.dumps(tags).encode(),
                        "application/json",
                        repeatable=False,
                    )
                except Refused:
                    self._abort(offer.get("abort"))
                    raise
                except Uncertain:
                    pass  # it may have finished: registering tells, and a retry starts afresh
        settings = {
            "description": description,
            "categories": [category],
            "restrict": restrict,
            "tabIngest": False,  # keep subjects.csv a CSV, not an ingested table
            "storageIdentifier": storage,
            "fileName": name,
            "mimeType": media,
            "checksum": {"@type": algorithm, "@value": digest[SERVER_DIGESTS[algorithm]]},
        }
        boundary = uuid.uuid4().hex
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="jsonData"\r\n\r\n'
            f"{json.dumps(settings)}\r\n--{boundary}--\r\n"
        ).encode()
        self._call(
            "POST",
            self._url("add"),
            body,
            f"multipart/form-data; boundary={boundary}",
            repeatable=False,
        )

    def add(self, path, folder, description, category, restrict, sha256):
        """Add one file to the draft through the server, refusing it if it is not as prepared."""
        if Path(path).stat().st_size > THROUGH_SERVER_LIMIT:
            raise Refused(
                f"{Path(path).name} is over {THROUGH_SERVER_LIMIT / 1e9:.0f} GB, more than goes "
                "through the server in one request; send it without --through-server"
            )
        try:
            payload = Path(path).read_bytes()
        except OSError as error:
            raise Refused(f"{path} could not be read: {error}; run prepare again") from None
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise Refused(f"{path} changed since prepare recorded it; run prepare again")
        settings = {
            "description": description,
            "categories": [category],
            "restrict": restrict,
            "tabIngest": False,  # keep subjects.csv a CSV, not an ingested table
        }
        if folder:
            settings["directoryLabel"] = folder
        boundary = uuid.uuid4().hex
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="jsonData"\r\n\r\n'
            f"{json.dumps(settings)}\r\n"
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{Path(path).name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode()
        body = head + payload + f"\r\n--{boundary}--\r\n".encode()
        self._call(
            "POST",
            self._url("add"),
            body,
            f"multipart/form-data; boundary={boundary}",
            repeatable=False,
        )


# --- upload and verify --------------------------------------------------------------------


def read_table(path, columns):
    """The rows of one of prepare's tables, refused when an older prepare wrote it."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run prepare first, to the end")
    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or any(name not in rows[0] for name in columns):
        raise SystemExit(f"{path} was written by an older prepare; run prepare again")
    return rows


def read_manifest(deposit):
    """Every HDF5 file of the deposit, with its digests and its archive."""
    return read_table(
        Path(deposit) / "MANIFEST.tsv", ("file", "part", "archive", "bytes", *DIGESTS)
    )


def read_archives(deposit, part, limit):
    """The archives of ``part``; with ``limit``, the first ``limit`` of each part."""
    columns = ("archive", "part", "subjects", "bytes", *DIGESTS, "holds")
    rows = read_table(Path(deposit) / "ARCHIVES.tsv", columns)
    rows = [row for row in rows if part == "all" or row["part"] == part]
    if limit:  # the tutorial's archive comes first in each part
        seen, kept = {}, []
        for row in rows:
            seen[row["part"]] = seen.get(row["part"], 0) + 1
            if seen[row["part"]] <= limit:
                kept.append(row)
        rows = kept
    return rows


def check_deposit(deposit):
    """Refuse a deposit whose prepare did not finish, or whose records changed since."""
    deposit = Path(deposit)
    if not (deposit / "CHECKS.txt").exists():
        raise SystemExit(f"{deposit / 'CHECKS.txt'} is missing: run prepare first, to the end")
    rows = read_manifest(deposit)
    archives = read_archives(deposit, "all", 0)
    sums = "".join(f"{row['sha256']}  {row['archive']}\n" for row in archives)
    sums += "".join(f"{row['sha256']}  {row['file']}\n" for row in rows)
    if (deposit / "SHA256SUMS").read_text() != sums:
        raise SystemExit(
            f"{deposit / 'SHA256SUMS'} is not what MANIFEST.tsv and ARCHIVES.tsv say; run "
            "prepare again"
        )
    inside = {}
    for row in rows:
        inside.setdefault(row["archive"], []).append(Path(row["file"]).name.split("_")[0])
    for row in archives:
        held = inside.pop(row["archive"], [])
        if row["holds"].split() != held or int(row["subjects"]) != len(held):
            raise SystemExit(
                f"ARCHIVES.tsv does not list the subjects MANIFEST.tsv puts in {row['archive']}; "
                "run prepare again"
            )
    if inside:
        raise SystemExit(
            f"MANIFEST.tsv puts files in archives ARCHIVES.tsv does not list "
            f"({sorted(inside)[:3]}); run prepare again"
        )
    table = deposit / "subjects.csv"
    with open(table, newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(OPEN_FIELDS):
        raise SystemExit(
            f"{table} has the columns {header}, where only {list(OPEN_FIELDS)} may be "
            "deposited; run prepare again"
        )
    subjects = read_subjects(table)
    if table.read_text() != subjects_table(subjects):
        raise SystemExit(
            f"{table} is not the table prepare wrote, the open-access fields alone and nothing "
            "more in any row; run prepare again"
        )
    listed = {row["subject"] for row in subjects}
    filed = {Path(row["file"]).name.split("_")[0] for row in rows}
    if listed != filed:
        raise SystemExit(
            f"{table} does not list the subjects MANIFEST.tsv holds; run prepare again"
        )


def media_type(name):
    """The media type the draft records for a file, by its name."""
    return MEDIA_TYPES.get(Path(name).suffix, "text/plain")


def plan(deposit, part, limit):
    """Each file to send: its name in the dataset, local file, digests and labels."""
    deposit = Path(deposit)
    check_deposit(deposit)
    kinds = {"sc": "Structural", "fc": "Functional"}
    items = []
    for document in DOCUMENTS:
        if not (deposit / document).exists():
            raise SystemExit(f"{deposit / document} is missing: run prepare first, to the end")
        items.append(
            dict(
                path=document,
                local=deposit / document,
                digest=digests(deposit / document),
                folder=None,
                description=document,
                category="Documentation",
                media=media_type(document),
            )
        )
    rows = read_archives(deposit, part, limit)
    absent = [
        row["archive"]
        for row in rows
        if not (deposit / ARCHIVES / row["archive"]).is_file()
        or (deposit / ARCHIVES / row["archive"]).stat().st_size != int(row["bytes"])
    ]
    if absent:
        raise SystemExit(
            f"{len(absent)} archives of the deposit are missing or not the size prepare recorded "
            f"({', '.join(absent[:3])}); run prepare again"
        )
    for row in rows:
        held = row["holds"].split()
        whose = (
            f"the {len(held)} subjects of the sbci tutorial"
            if row["archive"].endswith(f"_{TUTORIAL}.tar")
            else f"{len(held)} subjects, {held[0]} to {held[-1]}"
        )
        items.append(
            dict(
                path=row["archive"],
                local=deposit / ARCHIVES / row["archive"],
                digest={"bytes": int(row["bytes"]), **{name: row[name] for name in DIGESTS}},
                folder=None,
                description=(
                    f"{kinds[row['part']]} continuous connectomes of {whose} of the HCP Young "
                    f"Adult study, on the fsaverage ico4 grid (sbci HDF5, in {row['part']}/); "
                    "ARCHIVES.tsv lists them"
                ),
                category="Data",
                media=media_type(row["archive"]),
            )
        )
    return items


def unchanged(path, digest):
    """Whether a file still has the size and SHA-256 prepare recorded, read through once."""
    value, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            value.update(block)
            size += len(block)
    return size == digest["bytes"] and value.hexdigest() == digest["sha256"]


def matches(held, digest):
    """Whether a file the draft holds is the one prepared, by the checksum the server computed."""
    size, kind, value, _ = held
    name = SERVER_DIGESTS.get(str(kind))
    if name is None:
        raise SystemExit(
            f"the server reports {kind!r} checksums, which this tool cannot compare; ask the "
            "repository which algorithm it uses"
        )
    return size == digest["bytes"] and str(value).lower() == digest[name]


def connect(args):
    token = os.environ.get(TOKEN, "").strip()
    if not token:
        raise SystemExit(
            f"set {TOKEN} in your own shell first (Dataverse: your name > API Token); "
            "it is read from the environment and never printed"
        )
    if not TOKEN_FORM.match(token):
        raise SystemExit(
            f"{TOKEN} is not of the form Dataverse issues (8-4-4-4-12 hexadecimal digits); "
            "copy the token again, whole, and nothing else"
        )
    dataverse = Dataverse(check_server(args.server), normalize_dataset(args.dataset), token)
    dataverse.exists()
    return dataverse


def upload(args):
    items = plan(args.deposit, args.part, args.limit)
    total = sum(item["digest"]["bytes"] for item in items)
    if args.dry_run:
        for item in items:
            print(f"would send {item['path']} ({item['digest']['bytes'] / 1e9:.2f} GB)")
        dataset, server = normalize_dataset(args.dataset), check_server(args.server)
        how = "through the server" if args.through_server else "by direct upload"
        print(
            f"{len(items)} files, {total / 1e9:.2f} GB, to {dataset} on {server} {how}; the "
            "server is not asked, so a file the draft already holds is listed too (a run skips it)"
        )
        return
    dataverse = connect(args)
    lock = Path(args.deposit) / ".upload.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"{lock} says another upload of this deposit is running; if none is (one was "
            "stopped), remove it and run again"
        ) from None
    os.write(handle, f"{os.getpid()} {time.ctime()}\n".encode())
    os.close(handle)
    try:
        _upload(args, items, dataverse)
    finally:
        lock.unlink(missing_ok=True)


def held_as_another(path, held):
    """The refusal of a file the draft holds with other bytes than those prepared."""
    return SystemExit(
        f"{path} is in the draft as another file ({held[1]} {str(held[2])[:16]}...); delete it "
        "there and run again"
    )


def _upload(args, items, dataverse):
    held = dataverse.files()
    algorithm = None if args.through_server else dataverse.fixity()
    sent = skipped = sent_bytes = 0
    started = time.time()
    for index, item in enumerate(items, 1):
        path = item["path"]
        if path in held:
            if not matches(held[path], item["digest"]):
                raise held_as_another(path, held[path])
            if held[path][3] is not None and held[path][3] != args.restrict:
                state = "restricted" if held[path][3] else "open"
                raise SystemExit(
                    f"{path} is in the draft {state}, where this run sends "
                    f"{'restricted' if args.restrict else 'open'} files: run with the --restrict "
                    "the first run had, or change the file in the draft"
                )
            skipped += 1
            continue
        if not unchanged(item["local"], item["digest"]):
            raise SystemExit(
                f"{item['local']} changed since prepare recorded it; run prepare again"
            )
        for attempt in range(dataverse.retries + 1):
            dataverse.wait_for_locks()
            try:
                if args.through_server:
                    dataverse.add(
                        item["local"],
                        item["folder"],
                        item["description"],
                        item["category"],
                        args.restrict,
                        item["digest"]["sha256"],
                    )
                else:
                    dataverse.add_direct(
                        item["local"],
                        item["path"],
                        item["description"],
                        item["category"],
                        args.restrict,
                        item["digest"],
                        algorithm,
                        item["media"],
                    )
                break
            except Uncertain as error:
                # the file may have arrived before the failure: look before sending it again
                time.sleep(dataverse.pause * (attempt + 1))
                dataverse.wait_for_locks()
                now = dataverse.files()
                if path in now:
                    if matches(now[path], item["digest"]):
                        break
                    raise held_as_another(path, now[path]) from None
                if attempt == dataverse.retries:
                    raise SystemExit(f"{path} could not be added: {error}") from None
        sent += 1
        sent_bytes += item["digest"]["bytes"]
        rate = sent_bytes / max(time.time() - started, 1e-9) / 1e6
        print(
            f"  {index} of {len(items)}: {path} sent; {sent} sent, {skipped} already there, "
            f"{rate:.0f} MB/s",
            flush=True,
        )
    dataverse.wait_for_locks()
    print(f"sent {sent}, skipped {skipped} already in the draft; now run verify", flush=True)


def verify(args):
    items = plan(args.deposit, args.part, args.limit)
    held = connect(args).files()
    problems = []
    for item in items:
        path = item["path"]
        if path not in held:
            problems.append(f"missing: {path}")
        elif not matches(held[path], item["digest"]):
            problems.append(f"differs: {path}: the draft has {held[path][:3]}")
        elif held[path][3] is not None and held[path][3] != args.restrict:
            state = "restricted" if held[path][3] else "open"
            problems.append(f"{state}, where --restrict says otherwise: {path}")
    expected = {item["path"] for item in items}
    for path in sorted(set(held) - expected):
        # Dataverse names a second file of the same name name-1.ext, or name-1 without one
        first = re.sub(r"-\d+(\.[^./]+)?$", r"\1", path)
        if first != path and first in held:
            problems.append(f"a second copy: {path} (of {first})")
        elif args.part == "all" and not args.limit:
            problems.append(f"not in the deposit: {path}")
    for problem in problems[:50]:
        print(problem)
    if problems:
        raise SystemExit(f"{len(problems)} problems in {normalize_dataset(args.dataset)}")
    print(
        f"all {len(items)} files present, every size, checksum and restriction as prepared; "
        "the dataset's terms are not checked here: they must be the HCP's, not CC0"
    )


def _public(url):
    """A JSON answer of the repository's API, asked with no token."""
    request = urllib.request.Request(url, headers={"User-Agent": "sbci-deposit"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200]
        raise Refused(f"GET {url.split('?')[0]}: {error.code} {detail}") from None


def published(server, dataset, version):
    """The published version's description and ``{name: (id, bytes, kind, checksum)}``.

    The description gains ``citation``, the dataset's citation as the repository words it.
    """
    query = urllib.parse.urlencode({"persistentId": dataset})
    base = f"{server}/api/datasets/:persistentId/versions/{version}"
    about = _public(f"{base}?{query}").get("data") or {}
    if about.get("versionState") != "RELEASED":
        raise Refused(
            f"version {version} of {dataset} is not published "
            f"({about.get('versionState') or 'no such version'}); index a published one"
        )
    message = (_public(f"{base}/citation?{query}").get("data") or {}).get("message") or ""
    about["citation"] = html.unescape(re.sub(r"<[^>]*>", "", message)).strip() or None
    files = {}
    for entry in _public(f"{base}/files?{query}").get("data") or []:
        data_file = entry.get("dataFile", {})
        checksum = data_file.get("checksum") or {}
        files[entry.get("label")] = (
            data_file.get("id"),
            data_file.get("filesize"),
            checksum.get("type"),
            checksum.get("value"),
        )
    return about, files


def index_text(header, archives, subjects):
    """The index as JSON, one archive and one subject a line, so a change reads as a diff."""
    lines = ["{"]
    lines += [f" {json.dumps(key)}: {json.dumps(value)}," for key, value in header.items()]
    lines.append(' "archives": {')
    lines.append(",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in archives.items()))
    lines.append(" },")
    lines.append(' "subjects": [')
    lines.append(",\n".join(f"  {json.dumps(entry)}" for entry in subjects))
    lines += [" ]", "}"]
    return "\n".join(lines) + "\n"


def index(args):
    """Write the package's index of a published version: each file's place in its archive."""
    deposit = Path(args.deposit)
    check_deposit(deposit)
    dataset, server = normalize_dataset(args.dataset), check_server(args.server)
    about, held = published(server, dataset, args.version)
    rows = read_archives(deposit, "all", 0)
    for row in rows:
        name = row["archive"]
        if name not in held:
            raise Refused(f"{name} is not in version {args.version} of {dataset}")
        _, size, kind, value = held[name]
        digest = SERVER_DIGESTS.get(str(kind))
        if int(size or -1) != int(row["bytes"]) or not digest or str(value).lower() != row[digest]:
            raise Refused(
                f"{name} in version {args.version} of {dataset} is not the deposit's archive: "
                f"{size} bytes and {kind} {value}, where the deposit has {row['bytes']} bytes"
            )
    places = {}
    for row in rows:
        with tarfile.open(deposit / ARCHIVES / row["archive"]) as archive:
            for member in archive:
                places[member.name] = (row["archive"], member.offset_data, member.size)
    demographics = {row["subject"]: row for row in read_subjects(deposit / "subjects.csv")}
    subjects = {}
    for row in read_manifest(deposit):
        place = places.get(row["file"])
        if place is None or place[0] != row["archive"] or place[2] != int(row["bytes"]):
            raise Refused(f"{row['file']} is not in {row['archive']} as MANIFEST.tsv records it")
        name = Path(row["file"]).name.split("_")[0]
        entry = subjects.setdefault(
            name,
            {
                "subject": name,
                "sex": demographics[name]["sex"],
                "age_bin": demographics[name]["age_bin"],
                "tutorial": row["archive"].endswith(f"_{TUTORIAL}.tar"),
                "files": {},
            },
        )
        entry["files"][row["part"]] = {
            "archive": row["archive"],
            "offset": place[1],
            "bytes": place[2],
            "sha256": row["sha256"],
        }
    tutorial = sorted(n for n, e in subjects.items() if e["tutorial"])
    header = {
        "cohort": "hcp-ya",
        "description": (
            f"{len(subjects)} HCP Young Adult subjects, released on UNC Dataverse ({dataset}, "
            f"version {args.version}): per subject, one SC file with its streamline endpoints "
            "and one FC file, on the fsaverage ico4 grid, in tar archives of at most "
            f"{max(int(r['subjects']) for r in rows)} subjects. Each file is listed with its "
            "archive, where its bytes start there, its size and SHA-256. The "
            f"{len(tutorial)} marked tutorial ({', '.join(tutorial[:3])}, ...) are those "
            "`sbci download hcp-ya` fetches by default."
        ),
        "data_use": (deposit / "DATA_USE.txt").read_text(),
        "host": "dataverse",
        "release": {
            "server": server,
            "dataset": dataset,
            "version": args.version,
            "published": about.get("releaseTime"),
            "citation": about.get("citation"),
            "terms": TERMS_URL,
        },
    }
    archives = {
        row["archive"]: {
            "id": held[row["archive"]][0],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
        for row in rows
    }
    out = Path(args.out)
    out.write_text(index_text(header, archives, [subjects[n] for n in sorted(subjects)]))
    files = sum(len(e["files"]) for e in subjects.values())
    print(
        f"{out}: {len(subjects)} subjects ({len(tutorial)} of the tutorial), {files} files in "
        f"{len(rows)} archives of version {args.version} of {dataset}"
    )


def positive(text):
    """An argument that must be a whole number of at least one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a count, 1 or more")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("prepare", help="link, check, archive and describe the files")
    step.add_argument("--sc", required=True, help="the SC files, sub-NNNNNN_sc.h5")
    step.add_argument("--fc", required=True, help="the FC files, sub-NNNNNN_fc.h5")
    step.add_argument(
        "--released",
        help="copies already public, used in place of the build's; needed when any is deposited",
    )
    step.add_argument("--subjects", required=True, help="CSV with subject, sex and age_bin")
    step.add_argument("--out", required=True)
    allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    step.add_argument("--workers", type=int, default=allowed or 1)
    step.add_argument(
        "--per-archive",
        type=positive,
        default=PER_ARCHIVE,
        help=f"the most subjects in one archive (default {PER_ARCHIVE})",
    )
    step.add_argument("--rewrite-documents", action="store_true")
    step.add_argument(
        "--ref", help="the pushed commit the README names, when prepare runs from no checkout"
    )
    step.add_argument("--no-validate", action="store_true", help=argparse.SUPPRESS)
    for name, text in (("upload", "send the files to a draft"), ("verify", "check the draft")):
        step = commands.add_parser(name, help=text)
        step.add_argument("deposit", help="the directory prepare wrote")
        step.add_argument("--dataset", required=True, help="the draft's DOI, doi:10.xxxx/...")
        step.add_argument("--part", choices=("sc", "fc", "all"), default="all")
        step.add_argument(
            "--limit", type=positive, default=0, help="only the first N archives of each part"
        )
        step.add_argument("--server", default=SERVER)
        step.add_argument("--restrict", action="store_true", help="the files are restricted")
        if name == "upload":
            step.add_argument("--dry-run", action="store_true")
            step.add_argument(
                "--through-server",
                action="store_true",
                help="send each file through the server in one request, not by direct upload",
            )
    step = commands.add_parser("index", help="write sbci download's index of a published version")
    step.add_argument("deposit", help="the directory prepare wrote and upload sent")
    step.add_argument("--dataset", required=True, help="the dataset's DOI, doi:10.xxxx/...")
    step.add_argument("--version", required=True, help="the published version, such as 1.0")
    step.add_argument("--server", default=SERVER)
    step.add_argument("--out", required=True, help="where to write it: src/sbci/data/hcp_ya.json")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        # one thread a worker: each holds a whole matrix, and the BLAS would otherwise
        # start a thread per core in every one of them
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
    try:
        steps = {"prepare": prepare, "upload": upload, "verify": verify, "index": index}
        steps[args.command](args)
    except Refused as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
