"""A cohort from files on disk: subjects matched to a table, SC paired with FC, and a report.

:func:`load_cohort` is the way into the package for a study's data. It takes

* the computational files, named ``sub-<id>[_<key>-<value>...]_<sc|fc>.h5``:
  the names :func:`sbci.download.fetch_cohort` and the lab's builders write,
  the prefix, the modality and the extension in any case, with ``key-value``
  parts such as ``ses-1`` or ``acq-multi-shell`` allowed, their keys in
  lowercase as BIDS writes them, and an id that may hold underscores and
  hyphens (``sub-NDAR_INV1``, ``sub-NDAR_INV-1``). A folder is searched with
  its subfolders, symbolic links followed, so a BIDS-style tree works as well
  as a flat folder; a list of folders and files works too. A ``.h5`` file
  whose name says neither a subject nor a modality is listed in the summary
  as not read.
* optionally one or more tables of subjects, ``.csv`` or ``.tsv`` (or dicts of
  columns), with a ``subject`` or BIDS ``participant_id`` column. Ids match
  with or without the ``sub-`` prefix, so the HCP's ``Subject`` column of bare
  numbers matches ``sub-100307``.

It returns a :class:`LoadedCohort`: the subjects that have a usable file for
every modality asked for and a value for every covariate required, whose files
agree on how they were made; their files and covariates in one order; the
settings they share; and a report with one row for every subject seen in the
files or the tables, saying whether it is in and, if not, why.

What a file has to pass
-----------------------
Each file's structure is checked as :meth:`~sbci.ContinuousConnectome.load`
checks it, without reading its arrays: complete metadata on the ico4 grid, the
modality its name says, every array stored as numbers, the area, mask and
coordinates the grid's size, and the endpoint group, if stored, with every
dataset present and of one streamline count, not zero. ``validate=True`` reads
every file in full and runs every check of ``sbci validate``, the values among
them: finite connectivity, unit mass, an empty medial wall, a mask of 0s and
1s, endpoint indices on the grid and finite weights.

Then the files have to agree on how they were made: the keys in
:data:`SETTINGS`, the kernel and bandwidth of SC among them, numbers compared
as numbers. A cohort that mixes two ways is refused by default, naming who
differs; ``mismatch="exclude"`` keeps the subjects whose files share the most
common way, every setting of every modality at once, and leaves the others out,
and ``mismatch="report"`` keeps everyone and lists the differences.

A subject's files have to come from one visit: SC from one session and FC
from another would compare different visits, and is refused unless
``session=`` says which, one label for every modality or a label for each. A
mapping names every modality asked for, ``None`` taking the files without a
session label: one left out could take its files from any session.

The report also carries what is worth a look before an analysis: each SC
file's streamline count and whether it stores its endpoints (needed to
re-smooth and for :func:`sbci.endpoints_align`), and each FC file's frames and
runs. Nothing is excluded on those numbers; ``exclude=`` takes the subjects to
leave out and the reasons, which then stand in the report.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import numbers
import os
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import grid, spec
from .errors import SbciError

SETTINGS: dict[str, tuple[str, ...]] = {
    "sc": (
        "spec_version",
        "grid",
        "mask",
        "area_weights",
        "normalization",
        "kernel",
        "bandwidth",
        "nu",
        "streamline_weighting",
        "registration_reference",
        "pipeline_version",
        "container_version",
    ),
    "fc": (
        "spec_version",
        "grid",
        "mask",
        "area_weights",
        "normalization",
        "fc_nuisance_model",
        "registration_reference",
        "pipeline_version",
        "container_version",
    ),
}
"""Metadata keys every file of a modality must agree on: how the files were made."""

SUBJECT_COLUMNS = ("subject", "participant_id", "Subject", "subject_id", "src_subject_id")
"""Column names taken for the subject id, the first present, unless one is named."""

MISSING = frozenset(
    {
        "",
        "NA",
        "N/A",
        "n/a",
        "NaN",
        "nan",
        "-NaN",
        "-nan",
        "null",
        "NULL",
        "None",
        "<NA>",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "1.#IND",
        "1.#QNAN",
    }
)
"""Table cells read as missing, after stripping surrounding space: pandas' defaults.

Case matters, as it does to pandas: ``none`` is a value, ``None`` is missing.
``load_cohort(missing=...)`` adds to them, and ``keep_default_missing=False``
reads only the markers it names. A marker that reads as a number matches that
number however it is written, as pandas' do: ``-999`` matches ``-999.0``."""

MISMATCH = ("refuse", "exclude", "report")
"""What :func:`load_cohort` does when files disagree on a setting."""

_EXTENSION = re.compile(r"\.(h5|hdf5)$", re.IGNORECASE)


class CohortError(SbciError, ValueError):
    """A cohort refused as a whole; :attr:`report` holds the rows built before it was."""

    def __init__(self, message: str, report: list | None = None) -> None:
        super().__init__(message)
        self.report = list(report or [])


def subject_id(value) -> str:
    """The id as files carry it: ``sub-`` and the label, the prefix read in any case.

    Examples
    --------
    >>> subject_id(100307), subject_id("sub-100307"), subject_id(" 100307 ")
    ('sub-100307', 'sub-100307', 'sub-100307')
    >>> subject_id("SUB-01")
    'sub-01'
    """
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():  # a number read as a float
        text = text[:-2]
    return "sub-" + (text[4:] if text[:4].lower() == "sub-" else text)


_ENTITY = re.compile(r"[a-z]+-")
"""How a ``key-value`` part of a name begins: a lowercase key, as BIDS writes its keys."""

_BIDS_KEYS = re.compile(r"(?i)(ses|acq|run|task|desc|space|dir|echo|rec|ce|hemi|den|res)-.+")
"""A BIDS key in any case, to notice one written in capitals and so read into the id."""


def _uppercase_keys(subject: str) -> list[str]:
    """The parts of an id that read as a BIDS key in capitals: ``SES-1`` in ``sub-01_SES-1``."""
    return [
        part
        for part in subject[4:].split("_")[1:]
        if _BIDS_KEYS.fullmatch(part) and not _ENTITY.match(part)
    ]


def parse_name(name: str) -> tuple[str, str | None, str] | None:
    """``(subject, session, modality)`` from a computational file's name, or ``None``.

    The id runs from ``sub-`` to the first ``key-value`` part or the modality,
    so it may hold underscores and hyphens: a ``key-value`` part begins with a
    lowercase key, as BIDS writes its keys (``ses-1``, ``acq-multi-shell``), and
    a part such as ``INV-1`` belongs to the id. A value may hold hyphens; case
    does not matter in the ``sub-`` prefix, the modality or the extension. An
    empty part (``sub-01__sc.h5``) or a key given twice makes a name unreadable.

    Examples
    --------
    >>> parse_name("sub-100307_sc.h5")
    ('sub-100307', None, 'sc')
    >>> parse_name("sub-01_ses-2_desc-concon_fc.h5")
    ('sub-01', '2', 'fc')
    >>> parse_name("sub-NDAR_INV1_acq-multi-shell_SC.H5")
    ('sub-NDAR_INV1', None, 'sc')
    >>> parse_name("SUB-NDAR_INV-1_sc.h5")
    ('sub-NDAR_INV-1', None, 'sc')
    >>> parse_name("sub-01_vertexarea.dscalar.nii") is None
    True
    >>> parse_name("sub-01__sc.h5") is None and parse_name("sub-01_ses-1_ses-2_sc.h5") is None
    True
    """
    match = _EXTENSION.search(name)
    if match is None:
        return None
    parts = name[: match.start()].split("_")
    if len(parts) < 2 or parts[0][:4].lower() != "sub-" or len(parts[0]) == 4:
        return None
    modality = parts[-1].lower()
    if modality not in spec.MODALITIES or "" in parts:
        return None
    label = [parts[0][4:]]
    index = 1
    while index < len(parts) - 1 and not _ENTITY.match(parts[index]):
        label.append(parts[index])
        index += 1
    entities = {}
    for part in parts[index:-1]:
        if not _ENTITY.match(part):
            return None
        key, value = part.split("-", 1)
        if not value or key in entities:
            return None
        entities[key] = value
    return "sub-" + "_".join(label), entities.get("ses"), modality


def _same_label(found: str | None, wanted) -> bool:
    """Whether a file's ``ses-`` label is the one asked for: ``1`` matches ``01``."""
    if found is None or wanted is None:
        return found is None and wanted is None
    wanted = str(wanted).strip()
    if found == wanted:
        return True
    return found.isdigit() and wanted.isdigit() and int(found) == int(wanted)


def _markers(missing) -> frozenset:
    """``missing=`` as marker strings: one marker (``"-999"``, or ``-999``), several, or none."""
    if missing is None:
        return frozenset()
    named = (missing,) if isinstance(missing, (str, bytes, numbers.Number)) else tuple(missing)
    return frozenset(
        (marker.decode() if isinstance(marker, bytes) else str(marker)).strip() for marker in named
    )


@functools.lru_cache(maxsize=64)
def _numeric_markers(missing: frozenset) -> frozenset:
    """The markers that read as finite numbers, as numbers."""
    found = set()
    for marker in missing:
        try:
            number = float(marker)
        except ValueError:
            continue
        if math.isfinite(number):
            found.add(number)
    return frozenset(found)


def _missing(value, missing=MISSING) -> bool:
    """A missing cell: ``None``, a NaN of any kind, or what a marker in ``missing`` says.

    A string is missing if a marker names it, or if it reads as the number a
    marker reads as; a number, if a marker reads as it: ``-999`` takes ``-999``,
    ``"-999.0"`` and ``-999.0`` alike.
    """
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        text = (value.decode() if isinstance(value, bytes) else value).strip()
        if text in missing:
            return True
        targets = _numeric_markers(missing)
        if not targets:
            return False
        try:
            return float(text) in targets
        except ValueError:
            return False
    try:
        if value != value:  # NaN, and NaT, are unequal to themselves
            return True
    except TypeError:  # pandas' NA will not say whether it equals itself
        return True
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    return float(value) in _numeric_markers(missing)


def _visit(label: str | None) -> str:
    """A ``ses-`` label as visits are told apart: ``1`` and ``01`` are one; no label, ``none``."""
    if label is None:
        return "none"
    return str(int(label)) if label.isdigit() else label


def _asked_session(label) -> str:
    """The session ``session=`` asked for, as a reason names it: ``of session '5'``."""
    return "without a session label" if label is None else f"of session {str(label).strip()!r}"


def _other_sessions(labels) -> str:
    """The sessions files were of, as :func:`_visit` gives them, in words."""
    named = sorted((v for v in labels if v != "none"), key=lambda v: (not v.isdigit(), v.zfill(9)))
    parts = []
    if named:
        listed = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
        parts.append(("of session " if len(named) == 1 else "of sessions ") + listed)
    if "none" in labels:
        parts.append("without a session label")
    return " and ".join(parts)


def _is_frame(table) -> bool:
    """A pandas DataFrame, or anything holding named columns the way one does."""
    return hasattr(table, "columns") and hasattr(table, "iloc")


def _next_row(reader, path: Path) -> list | None:
    """The table's next row, or ``None`` at its end; what ``csv`` cannot read, a ``ValueError``.

    ``csv.Error`` is no ``ValueError``: a cell past the module's size limit
    escaped ``sbci cohort`` as a traceback. The line given is where the row starts.
    A table that is not UTF-8 (a BOM is read) raised a ``UnicodeDecodeError``
    that named no file.
    """
    start = reader.line_num + 1
    try:
        return next(reader, None)
    except csv.Error as error:
        raise ValueError(
            f"{path.name}, line {start}: {error}, as Python's csv module reads it. A cell that "
            "opens with a quote runs on to the next quote, across lines, so a quote left open, "
            "or not doubled inside a quoted value, takes in the rows after it"
        ) from None
    except UnicodeDecodeError:
        # The text is decoded ahead of the rows, a block at a time: the line is found anew.
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                start = number
                break
        raise ValueError(
            f"{path.name}, line {start}: not UTF-8 text, which a table is read as. Save it as "
            "UTF-8 (in Excel, as CSV UTF-8) and load it again"
        ) from None


def _read_table(table, subject_column, missing=MISSING):
    """The subject column's name, the ids in the table's order, and every column as a list."""
    if _is_frame(table):
        table = {str(name): list(table[name]) for name in table.columns}
    if isinstance(table, Mapping):
        columns = {str(key): list(values) for key, values in table.items()}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"the table's columns differ in length: {sorted(lengths)}")
    else:
        path = Path(table)
        if not path.is_file():
            raise FileNotFoundError(f"no such table: {path}")
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, newline="", encoding="utf-8-sig") as handle:
            # strict: a quote left open is an error, not a cell that runs on through
            # every row after it, as one did, reading three subjects as one
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            header = _next_row(reader, path)
            if header is None:
                raise ValueError(f"{path.name} is empty")
            header = [name.strip() for name in header]
            repeated = sorted({name for name in header if name and header.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"{path.name} has two columns named {repeated[0]!r}; which to use is not "
                    "for the loader to guess"
                )
            rows = []
            while (row := _next_row(reader, path)) is not None:
                if not any(cell.strip() for cell in row):
                    continue  # a blank line
                if any(cell.strip() for cell in row[len(header) :]):
                    raise ValueError(
                        f"{path.name}, line {reader.line_num}: {len(row)} cells under "
                        f"{len(header)} column names. A "
                        f"{'tab' if delimiter == chr(9) else 'comma'} inside a value has to be "
                        "quoted; read unquoted, it shifts every later cell of the row"
                    )
                rows.append(row[: len(header)] + [""] * (len(header) - len(row)))
            # Columns without a name are what a spreadsheet's trailing commas leave
            # (subject,age,,): dropped when they hold no value, where two were refused as
            # two columns named '' and one was read as a covariate named ''. One that
            # holds values has no name to go by.
            for index, name in enumerate(header):
                if not name and any(row[index].strip() for row in rows):
                    raise ValueError(
                        f"{path.name}: column {index + 1} holds values but has no name; name "
                        "it in the first row"
                    )
            columns = {
                name: [row[index] for row in rows] for index, name in enumerate(header) if name
            }
    if subject_column is None:
        found = [name for name in SUBJECT_COLUMNS if name in columns]
        if not found:
            raise ValueError(
                f"the table has no subject column ({', '.join(SUBJECT_COLUMNS)}); name it "
                f"with subject_column=. Its columns: {', '.join(list(columns)[:12])}"
            )
        subject_column = found[0]
    elif subject_column not in columns:
        raise ValueError(f"the table has no column {subject_column!r}")
    ids = []
    for index, value in enumerate(columns[subject_column]):
        if _missing(value, missing):
            raise ValueError(f"row {index + 1} of the table has no {subject_column}")
        ids.append(subject_id(value))
    repeated = sorted(name for name, count in Counter(ids).items() if count > 1)
    if repeated:
        shown = ", ".join(repeated[:5]) + (" ..." if len(repeated) > 5 else "")
        raise ValueError(f"the table lists {len(repeated)} subjects more than once: {shown}")
    return subject_column, ids, columns


_PADDED = re.compile(r"[+-]?0\d")
"""A number written with a leading zero, as codes often are: ``01``, ``007``."""

_EXACT = 2**53
"""Integers beyond this lose digits as float64, so codes that long stay text."""


def _number(value):
    """``value`` as a float if it is written as a number that float64 holds; else ``None``.

    Python's ``float`` also reads ``12_345`` as 12345, which pandas does not; such a
    cell is a code. So is an integer of more digits than float64 keeps exactly: two
    18-digit family codes could read as one number.
    """
    if isinstance(value, numbers.Integral):  # bool included: True reads as 1, as before
        return float(value) if abs(int(value)) <= _EXACT else None
    if isinstance(value, numbers.Real):
        return float(value)
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip()
    if "_" in text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    digits = text.lstrip("+-").lstrip("0")
    if digits.isdigit() and int(digits) > _EXACT:
        return None
    return number


def _as_column(values, missing=MISSING) -> np.ndarray:
    """Numbers as float64 with NaN for missing; anything else as strings with None for missing.

    A column of numbers stays text when two of its cells are written differently
    but read as the same number and one of them has a leading zero: ``07`` and
    ``7`` are two codes, which as numbers would be one.
    """
    cells = [None if _missing(v, missing) else v for v in values]
    present = [v for v in cells if v is not None]
    numbers = [_number(v) for v in present]
    if any(n is None for n in numbers):
        numbers = None
    else:
        spellings: dict = {}
        for value, number in zip(present, numbers, strict=True):
            spellings.setdefault(number, set()).add(str(value).strip())
        if any(
            len(written) > 1 and any(_PADDED.match(w) for w in written)
            for written in spellings.values()
        ):
            numbers = None
    if numbers is not None:
        out = np.full(len(cells), np.nan)
        out[[i for i, v in enumerate(cells) if v is not None]] = numbers
        return out
    return np.array([None if v is None else str(v).strip() for v in cells], dtype=object)


def _find_files(files) -> tuple[dict[Path, str], list[Path]]:
    """The computational files ``files`` names, and the ``.h5`` files whose names say nothing.

    Folders are searched with their subfolders, symbolic links followed, and
    every real folder is searched once and every file taken once under its name,
    however many ways lead to it; two names for one file stay two subjects. Each
    file comes with where it is shown: its path from the folder it was found in,
    or as it was given, since a subject's two files may share a name.
    """
    # Any path-like, as load, reduce and project take one: another was read as a list.
    items = [files] if isinstance(files, (str, os.PathLike)) else list(files)
    paths: dict[tuple, tuple[Path, str]] = {}
    unnamed: dict[tuple, Path] = {}
    seen: set = set()
    for item in items:
        path = Path(item)
        if path.is_dir():
            for root, folders, names in os.walk(path, followlinks=True):
                real = os.path.realpath(root)
                if real in seen:
                    folders[:] = []
                    continue
                seen.add(real)
                folders.sort()
                for name in sorted(names):
                    found = Path(root) / name
                    key = (os.path.realpath(found), name)
                    if parse_name(name) is not None:
                        # a link to nothing fails its check, by name
                        paths.setdefault(key, (found, os.path.relpath(found, path)))
                    elif _EXTENSION.search(name):
                        unnamed.setdefault(key, found)
        elif not path.exists():
            raise FileNotFoundError(f"no such folder or file: {path}")
        elif parse_name(path.name) is None:
            raise ValueError(
                f"{path.name!r} is not named sub-<id>[_<key>-<value>...]_<sc|fc>.h5, so its "
                "subject and modality are unknown"
            )
        else:
            paths.setdefault((os.path.realpath(path), path.name), (path, str(path)))
    return dict(paths.values()), list(unnamed.values())


def _inspect(path: Path, modality: str, validate: bool) -> tuple[dict, dict]:
    """A file's metadata and the numbers the report carries; raises with the reason it fails.

    Checks the structure :meth:`~sbci.ContinuousConnectome.load` checks,
    without reading the arrays; ``validate`` reads them and checks the values.
    """
    from .io import read_header
    from .metadata import MetadataError

    if not path.is_file():
        raise ValueError("not a file that can be read: a link to nothing, or no file at all")
    header = read_header(path)
    metadata = header["metadata"]
    try:
        metadata.validate()
    except MetadataError as error:
        raise ValueError(str(error)) from None
    if metadata.modality != modality:
        raise ValueError(f"holds {metadata.modality}, though its name says {modality}")
    # what read_hdf5 and load refuse, refused here too: every method after would fail
    if header["has_endpoints"] and modality != "sc":
        raise ValueError("carries streamline endpoints, which belong to a structural connectome")
    if header["has_endpoints"] and header["n_endpoints"] == 0:
        raise ValueError(
            "its endpoint group holds no streamlines; a file without them leaves it out"
        )
    if not header["numeric"]:
        raise ValueError(
            "one of its connectivity, area, mask, coordinates and endpoint datasets does not "
            "hold numbers"
        )
    if not header["flat"]:
        raise ValueError("its connectivity, area or mask is not stored flat, on one axis")
    try:
        n = grid.n_from_condensed(header["n_connectivity"])
    except ValueError as error:
        raise ValueError(f"its connectivity is not a triangle: {error}") from None
    if n != spec.N_VERTICES:
        raise ValueError(f"its connectivity is for {n} vertices, not the grid's {spec.N_VERTICES}")
    if header["n_vertices"] != n or header["n_mask"] != n:
        raise ValueError(
            f"its area has {header['n_vertices']} entries and its mask {header['n_mask']}, "
            f"where the connectivity is for {n} vertices"
        )
    if header["coordinates"] is not None and tuple(header["coordinates"]) != (n, 3):
        raise ValueError(f"its coordinates are {tuple(header['coordinates'])}, not {(n, 3)}")
    if validate:
        from .validate import validate_file

        failed = [check for check in validate_file(path) if not check.passed]
        if failed:
            raise ValueError("fails " + "; ".join(str(check) for check in failed))
    fields = metadata.fields
    if modality == "sc":
        numbers = {
            "sc_streamlines": fields.get("streamline_count"),
            "sc_endpoints": int(header["n_endpoints"] > 0),
        }
    else:
        frames, runs = fields.get("fc_frames"), fields.get("fc_runs")
        if isinstance(frames, (list, tuple)):  # frames given run by run
            frames = (
                sum(frames) if all(isinstance(f, (int, float)) for f in frames) else len(frames)
            )
        numbers = {
            "fc_frames": frames,
            "fc_runs": len(runs) if isinstance(runs, (list, tuple)) else runs,
        }
    return fields, numbers


def _normalize(value):
    """A setting as it compares: numbers as floats, so a bandwidth of 2 is one of 2.0."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    return value


def _key(value) -> str:
    return json.dumps(_normalize(value), sort_keys=True, default=str)


@dataclass
class LoadedCohort:
    """What :func:`load_cohort` returns: the subjects that passed, their files and covariates."""

    subjects: list
    """Ids of the subjects in the cohort, ``sub-<id>``, sorted."""
    files: dict
    """``{modality: [path, ...]}``, one path per subject, in :attr:`subjects`' order."""
    covariates: dict
    """``{column: array}``, every column of the tables, in :attr:`subjects`' order: float64
    with ``NaN`` for missing where the column holds numbers, else strings with ``None``."""
    settings: dict
    """``{modality: {key: value}}``, the settings of :data:`SETTINGS` the files share (under
    ``mismatch="report"``, a differing key holds the list of its values)."""
    report: list
    """One dict per subject seen in the files or the tables: ``subject``, ``included``,
    ``reason``, each modality's file and session, the numbers worth a look, the
    required covariates."""
    modalities: tuple = ("sc",)
    session: object = None
    required: tuple = field(default_factory=tuple)
    unread: list = field(default_factory=list)
    """``.h5`` files found whose names say neither a subject nor a modality."""
    reasons: dict = field(default_factory=dict)
    """``{subject: [reason, ...]}`` for every subject left out, each reason whole: the report
    joins them with ``"; "``, which a reason may hold too."""

    @property
    def n_subjects(self) -> int:
        """Subjects in the cohort."""
        return len(self.subjects)

    @property
    def excluded(self) -> dict:
        """``{subject: reason}`` for every subject left out."""
        return {row["subject"]: row["reason"] for row in self.report if not row["included"]}

    def paths(self, modality: str = "sc") -> list:
        """The files of one modality, in :attr:`subjects`' order: what ``sbci.reduce`` takes."""
        if modality not in self.files:
            raise ValueError(f"this cohort has {', '.join(self.files)} files, not {modality}")
        return list(self.files[modality])

    def column(self, name: str) -> np.ndarray:
        """One covariate, in :attr:`subjects`' order."""
        if name not in self.covariates:
            raise KeyError(f"no column {name!r}; the tables have {', '.join(self.covariates)}")
        return self.covariates[name]

    def save_report(self, path) -> Path:
        """Write :attr:`report` as a ``.csv`` or ``.tsv`` table."""
        return _write_report(self.report, path)

    def summary(self) -> str:
        """A few lines: how many came in, who was left out and why, what the files share."""
        seen = len(self.report)
        lines = [
            f"{seen} subjects seen; {self.n_subjects} in the cohort ({', '.join(self.modalities)})."
        ]
        # each subject's reasons as they were given: split on "; ", a reason that holds one
        # counted twice, and three subjects left out were counted four times
        reasons = Counter(
            reason
            for row in self.report
            if not row["included"]
            for reason in self.reasons.get(row["subject"], [row["reason"]])
        )
        for reason, count in reasons.most_common():
            lines.append(f"  left out, {count}: {reason}")
        if self.unread:
            shown = ", ".join(p.name for p in self.unread[:3]) + (
                " ..." if len(self.unread) > 3 else ""
            )
            lines.append(
                f"  not read, {len(self.unread)} .h5 file(s) whose names say no subject and "
                f"modality: {shown}"
            )
        for modality, shared in self.settings.items():
            # a setting the files do not record, nu outside the Matern kernel, is left out
            listed = ", ".join(
                f"{key} {value!r}" for key, value in shared.items() if value is not None
            )
            lines.append(f"  {modality} files share: {listed}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<LoadedCohort of {self.n_subjects} subjects ({', '.join(self.modalities)}), "
            f"{len(self.excluded)} left out>"
        )


def _write_report(report, path) -> Path:
    from .export import _write_table

    header = list(report[0]) if report else ["subject", "included", "reason"]
    rows = ([row.get(key) for key in header] for row in report)
    return _write_table(path, header, (["" if v is None else v for v in r] for r in rows))


_NUMBERS = {"sc": ("sc_streamlines", "sc_endpoints"), "fc": ("fc_frames", "fc_runs")}


def load_cohort(
    files,
    table=None,
    modalities=("sc",),
    session=None,
    require: Iterable[str] = (),
    subject_column: str | None = None,
    exclude: Mapping | None = None,
    validate: bool = False,
    mismatch: str = "refuse",
    missing: str | float | Iterable[str] = (),
    keep_default_missing: bool = True,
) -> LoadedCohort:
    """Gather a cohort's files and covariates, check them, and report who is left out and why.

    Parameters
    ----------
    files
        A folder holding ``sub-<id>[_<key>-<value>...]_<sc|fc>.h5`` files,
        searched with its subfolders, or a list of such folders and files.
    table
        A ``.csv`` or ``.tsv`` file, or a dict of columns, with one row per
        subject; or a list of them, joined on the subject (a subject missing
        from one has no value in its columns). Optional: without one the
        cohort is the files alone. A file is read as UTF-8, with or without a
        BOM. A column with neither a name nor a value, as a spreadsheet's
        trailing commas leave, is dropped; a file with two columns of one
        name, or a column with values and no name, is refused.
    modalities
        ``"sc"``, ``"fc"``, or both: what every subject needs a file of.
    session
        Use the files of this ``ses-`` label only (``1`` matches ``ses-01``),
        or ``{modality: label}`` to pair, say, SC from one visit with FC from
        another on purpose. A mapping names every modality asked for, and
        ``None`` there takes the files without a ``ses-`` label: the HCP's SC,
        which has none, beside its FC from the first day is ``{"sc": None,
        "fc": "REST1"}``. Without ``session=`` a subject's files have to come
        from one session, whichever it is: several for one modality, or SC
        from one and FC from another, are refused. A subject whose files are of
        other sessions only stays in the report, its reason naming them, and a
        label no file carries is refused as that.
    require
        Table columns every subject needs a value in; a subject with a missing
        cell is left out.
    subject_column
        The table's id column, when it is not one of :data:`SUBJECT_COLUMNS`.
    exclude
        ``{subject: reason}``, subjects to leave out on the caller's grounds,
        such as a quality check upstream; the reasons stand in the report.
    validate
        Run every check of ``sbci validate`` on every file, reading each in
        full -- the values, endpoint indices among them -- rather than
        checking each file's structure alone.
    mismatch
        When files disagree on how they were made: ``"refuse"`` (the default)
        raises, naming who differs; ``"exclude"`` keeps the subjects whose
        files share the most common way -- every setting of every modality
        together -- and leaves the rest out; ``"report"`` keeps everyone and
        records the differences.
    missing
        Further table cells to read as missing, beside pandas' defaults
        (:data:`MISSING`): ``"-999"`` (or ``-999``) is one marker, a list
        several. A marker that reads as a number matches that number however a
        cell writes it, ``-999.0`` as well.
    keep_default_missing
        ``False`` reads only the markers ``missing=`` names, so that, say, a
        site coded ``NA`` stays a site.

    A refusal is a :class:`CohortError`, which carries the report as far as it
    was built, so that ``sbci cohort --report`` can still write it.

    Examples
    --------
    >>> cohort = sbci.load_cohort("hcp-ya", table="hcp-ya/manifest.csv",
    ...                           modalities=("sc", "fc"))              # doctest: +SKIP
    >>> print(cohort.summary())                                       # doctest: +SKIP
    >>> reduction = sbci.reduce(cohort.paths("sc"), rank=10)          # doctest: +SKIP
    """
    modalities = (modalities,) if isinstance(modalities, str) else tuple(modalities)
    if not modalities or any(m not in spec.MODALITIES for m in modalities):
        raise ValueError(f"modalities must be drawn from {spec.MODALITIES}, got {modalities}")
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"modalities {modalities} name one twice")
    if mismatch not in MISMATCH:
        raise ValueError(f"mismatch must be one of {MISMATCH}, got {mismatch!r}")
    if isinstance(session, Mapping):
        given = {str(key): label for key, label in session.items()}
        unknown = sorted(set(given) - set(modalities))
        if unknown:
            raise ValueError(f"session= names modalities not asked for: {unknown}")
        left_out = [m for m in modalities if m not in given]
        if left_out:
            named = ", ".join(m for m in modalities if m in given)
            raise ValueError(
                f"session= names {named} but not {', '.join(left_out)}: a modality left out "
                "would take its files from any session, and pair two visits unasked. Name "
                "every modality -- None takes the files without a session label -- or give "
                "one label for all"
            )
        wanted = {m: given[m] for m in modalities}
        chosen = True
    else:
        wanted = {m: session for m in modalities}
        chosen = session is not None
    for modality, label in wanted.items():
        if label is not None and not str(label).strip():
            raise ValueError(
                f"session= is blank for {modality}; None takes the files without a session label"
            )
    missing = _markers(missing)  # "-999", or -999, is one marker
    if keep_default_missing:
        missing |= MISSING
    required = (require,) if isinstance(require, str) else tuple(require)
    reserved = {"subject", "included", "reason"}
    for m in modalities:
        reserved |= {f"{m}_file", f"{m}_session", *_NUMBERS[m]}
    clashing = sorted(set(required) & reserved)
    if clashing:
        raise ValueError(
            f"the column {clashing[0]!r} would overwrite the report's own field of that name; "
            "rename it in the table"
        )
    excluded_by_caller = {subject_id(k): str(v) for k, v in (exclude or {}).items()}

    # The files, by subject and modality, for the sessions asked for.
    paths_found, unread = _find_files(files)
    found: dict[str, dict[str, list[tuple[Path, str | None]]]] = {}
    # the sessions of the files session= leaves out, so that the subjects they belong to are
    # reported, and a label no file carries is not taken for a folder without files
    elsewhere: dict[str, dict[str, set[str]]] = {}
    warned: set[str] = set()
    for path in paths_found:
        subject, ses, modality = parse_name(path.name)
        for part in _uppercase_keys(subject):
            if part not in warned:  # once a spelling, not once a file
                warned.add(part)
                warnings.warn(
                    f"{path.name}: {part!r} is read as part of the subject id; BIDS keys are "
                    f"lowercase ({part.lower()})",
                    stacklevel=2,
                )
        if modality not in modalities:
            continue
        if chosen and not _same_label(ses, wanted[modality]):
            elsewhere.setdefault(subject, {}).setdefault(modality, set()).add(_visit(ses))
            continue
        found.setdefault(subject, {}).setdefault(modality, []).append((path, ses))

    # The tables, joined on the subject: {column: {subject: cell}}.
    covariate_columns: dict[str, dict] = {}
    table_ids: list = []
    if table is not None:
        tables = (
            [table]
            if isinstance(table, (str, os.PathLike, Mapping)) or _is_frame(table)
            else list(table)
        )
        for one in tables:
            id_column, ids, columns = _read_table(one, subject_column, missing)
            for name, values in columns.items():
                if name == id_column:
                    continue
                if name in covariate_columns:
                    raise ValueError(f"two tables have a column {name!r}; rename one")
                covariate_columns[name] = dict(zip(ids, values, strict=True))
            seen_ids = set(table_ids)
            table_ids += [s for s in ids if s not in seen_ids]
        absent = [name for name in required if name not in covariate_columns]
        if absent:
            raise ValueError(f"no table has a column {absent[0]!r}")
    elif required:
        raise ValueError("require= names table columns, but no table was given")
    in_table = set(table_ids)

    # A file whose id is another subject's with a suffix may be a copy of that
    # subject's file (sub-01_old beside sub-01) or a subject of its own (sub-NDAR_INV1_2
    # beside sub-NDAR_INV1). Without a table to say which, it is set aside and
    # reported, neither read as a subject nor merged into the other; a table that
    # lists it reads it as one.
    extending: dict[str, str] = {}
    for subject in found:
        if subject in in_table:
            continue
        base = next(
            (other for other in sorted(found, key=len) if subject.startswith(other + "_")), None
        )
        if base is not None:
            extending[subject] = base

    # A subject's files have to come from one visit unless session= says otherwise; a
    # subject the caller or the table leaves out is not asked.
    crossed: dict[str, str] = {}
    if not chosen:
        for subject, by_modality in sorted(found.items()):
            if subject in excluded_by_caller or subject in extending:
                continue
            if table is not None and subject not in in_table:
                continue
            labels = {
                m: sorted({_visit(s) for _, s in entries}) for m, entries in by_modality.items()
            }
            if len({label for each in labels.values() for label in each}) > 1:
                crossed[subject] = "; ".join(
                    f"{m} from {', '.join(labels[m])}" for m in modalities if m in labels
                )

    # Every subject seen, and why each is out.
    everyone = sorted(set(found) | in_table | set(elsewhere))
    unknown = sorted(set(excluded_by_caller) - set(everyone))
    if unknown:
        warnings.warn(
            f"exclude= names {len(unknown)} subject(s) in no file or table: "
            + ", ".join(unknown[:5])
            + (" ..." if len(unknown) > 5 else ""),
            stacklevel=2,
        )
    reasons: dict[str, list[str]] = {subject: [] for subject in everyone}
    report: dict[str, dict] = {}
    settings_of: dict[str, dict] = {}
    for subject in everyone:
        row: dict = {"subject": subject, "included": True, "reason": ""}
        if subject in excluded_by_caller:
            reasons[subject].append(f"excluded by the caller: {excluded_by_caller[subject]}")
        if table is not None and subject not in in_table:
            reasons[subject].append("not in the table")
        if subject in extending:
            reasons[subject].append(
                f"its name extends {extending[subject]}'s: a copy of that subject's file, or a "
                "subject of its own? A table that lists it reads it as one"
            )
        made: dict = {}
        for modality in modalities:
            entries = found.get(subject, {}).get(modality, [])
            row[f"{modality}_file"] = str(entries[0][0]) if len(entries) == 1 else ""
            row[f"{modality}_session"] = entries[0][1] if len(entries) == 1 else None
            for key in _NUMBERS[modality]:
                row[key] = None
            if not entries:
                other = elsewhere.get(subject, {}).get(modality)
                if other:
                    reasons[subject].append(
                        f"no {modality} file {_asked_session(wanted[modality])}, "
                        f"only {_other_sessions(other)}"
                    )
                else:
                    reasons[subject].append(f"no {modality} file")
                continue
            if len(entries) > 1:
                # where each is, from the folder searched: by name alone, two read as one
                names = ", ".join(sorted(paths_found[p] for p, _ in entries))
                reasons[subject].append(f"{len(entries)} {modality} files: {names}")
                continue
            try:
                fields, numbers_found = _inspect(entries[0][0], modality, validate)
            except Exception as error:  # noqa: BLE001 - every reason a file fails is reported
                reasons[subject].append(f"{modality} file {entries[0][0].name}: {error}")
                continue
            row.update(numbers_found)
            made[modality] = {key: fields.get(key) for key in SETTINGS[modality]}
            if modality == "sc" and made["sc"]["kernel"] == "matern" and made["sc"]["nu"] is None:
                made["sc"]["nu"] = 3.0  # made before nu was recorded, when 3 was the only one
        if len(made) == len(modalities):
            settings_of[subject] = made
        for name in required:
            value = covariate_columns[name].get(subject)
            row[name] = None if _missing(value, missing) else value
        if subject in in_table:
            empty = [
                name for name in required if _missing(covariate_columns[name].get(subject), missing)
            ]
            if empty:
                reasons[subject].append("no value for " + ", ".join(empty))
        report[subject] = row

    def finished() -> list:
        for subject in everyone:
            if reasons[subject]:
                report[subject]["included"] = False
                report[subject]["reason"] = "; ".join(reasons[subject])
        return [report[s] for s in everyone]

    if crossed:
        # In the report, everyone is out with the cohort, and the ones whose files
        # cross sessions say which.
        for subject in everyone:
            if subject in crossed:
                reasons[subject].append(f"files from more than one session ({crossed[subject]})")
            elif not reasons[subject]:
                reasons[subject].append(
                    "refused with the cohort: some subjects' files come from more than one session"
                )
        first = next(iter(crossed))
        others = f", as do {len(crossed) - 1} other subjects'" if len(crossed) > 1 else ""
        raise CohortError(
            f"{first}'s files come from more than one session ({crossed[first]}){others}: pairing "
            "them would compare different visits. Choose with session=, one label for every "
            "modality or {modality: label} for each, or leave the subject out with exclude=",
            report=finished(),
        )

    # How the files were made, every setting of every modality at once, over
    # the subjects not already out.
    candidates = [s for s in everyone if not reasons[s] and s in settings_of]
    way = {
        s: tuple(_key(settings_of[s][m][k]) for m in modalities for k in SETTINGS[m])
        for s in candidates
    }
    keys = [(m, k) for m in modalities for k in SETTINGS[m]]

    def value_of(subject, i):
        modality, key = keys[i]
        return settings_of[subject][modality][key]

    ways = Counter(way.values()).most_common()
    shared: dict[str, dict] = {m: {} for m in modalities}
    if len(ways) > 1:
        differing = [i for i, _ in enumerate(keys) if len({w[i] for w, _ in ways}) > 1]
        described = "; ".join(
            f"{m} {k}: "
            + ", ".join(
                f"{json.loads(value)!r} in {count}"
                for value, count in Counter(way[s][i] for s in candidates).most_common()
            )
            for i, (m, k) in ((i, keys[i]) for i in differing)
        )
        tie = ways[0][1] == ways[1][1]
        odd = [s for s in candidates if way[s] != ways[0][0]]
        described += f" (first of the others: {', '.join(odd[:3])})"
        if mismatch == "refuse" or (mismatch == "exclude" and tie):
            remedy = (
                "no way of making them is the most common; choose the files with exclude="
                if mismatch == "exclude"
                else "pass mismatch='exclude' to keep the subjects made the most common way, "
                "or 'report' to keep everyone"
            )
            # In the report, everyone is out with the cohort, and the odd ones say how.
            for s in candidates:
                reasons[s].append("refused with the cohort: its files were not all made one way")
                if way[s] != ways[0][0]:
                    reasons[s].append(
                        "; ".join(
                            f"{keys[i][0]} {keys[i][1]} is {value_of(s, i)!r}"
                            for i in differing
                            if way[s][i] != ways[0][0][i]
                        )
                    )
            raise CohortError(
                f"the files were not all made the same way ({described}). Mixing them makes "
                f"the subjects incomparable; {remedy}",
                report=finished(),
            )
        if mismatch == "exclude":
            kept = ways[0][0]
            for s in candidates:
                if way[s] != kept:
                    parts = [
                        f"{keys[i][0]} {keys[i][1]} is {value_of(s, i)!r}, "
                        f"not the cohort's {json.loads(kept[i])!r}"
                        for i in differing
                        if way[s][i] != kept[i]
                    ]
                    reasons[s].append("; ".join(parts))
            for i, (m, k) in enumerate(keys):
                shared[m][k] = json.loads(kept[i])
        else:
            for i, (m, k) in enumerate(keys):
                values = [
                    json.loads(v) for v, _ in Counter(w[i] for w in way.values()).most_common()
                ]
                shared[m][k] = values[0] if len(values) == 1 else values
    elif candidates:
        first = candidates[0]
        shared = {m: dict(settings_of[first][m]) for m in modalities}

    rows = finished()
    included = [s for s in everyone if not reasons[s]]
    if not included:
        tally = Counter(part for s in everyone for part in reasons[s])
        common = "; ".join(f"{count} {reason}" for reason, count in tally.most_common(3))
        if chosen and not found and elsewhere:
            asked = (
                "of the sessions asked for"
                if isinstance(session, Mapping)
                else _asked_session(session)
            )
            seen = {v for by in elsewhere.values() for labels in by.values() for v in labels}
            hint = (
                f" No {' or '.join(modalities)} file is {asked}: the files are "
                f"{_other_sessions(seen)}."
            )
        elif table is not None:
            hint = (
                " Do the table's ids match the files' (100307 and sub-100307 match; 0100307 "
                "does not)?"
            )
        else:
            hint = ""
        raise CohortError(
            f"no subject passed: {common}.{hint}"
            if tally
            else "no files of the asked modalities were found",
            report=rows,
        )
    files_out = {m: [found[s][m][0][0] for s in included] for m in modalities}
    covariates = {}
    for name, cells in covariate_columns.items():
        values = [cells.get(s) for s in included]
        covariates[name] = _as_column(values, missing)
        padded = sorted(
            {v.strip() for v in values if isinstance(v, str) and _PADDED.match(v.strip())}
        )
        if covariates[name].dtype != object and padded:
            warnings.warn(
                f"column {name!r} is read as numbers, though some are written with a leading "
                f"zero ({', '.join(padded[:3])}), as codes often are; if it holds codes, pass "
                f"categorical=[{name!r}] to sbci.stats.design",
                UserWarning,
                stacklevel=2,
            )
    return LoadedCohort(
        subjects=included,
        files=files_out,
        covariates=covariates,
        settings=shared,
        report=rows,
        modalities=modalities,
        session=session if isinstance(session, Mapping) or session is None else str(session),
        required=required,
        unread=unread,
        reasons={s: list(reasons[s]) for s in everyone if reasons[s]},
    )
