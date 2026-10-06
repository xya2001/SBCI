"""A cohort from files on disk: subjects matched to a table, SC paired with FC, and a report.

:func:`load_cohort` is the way into the package for a study's data. It takes

* the computational files, named ``sub-<id>[_<key>-<value>...]_<sc|fc>.h5``:
  the names :func:`sbci.download.fetch_cohort` and the lab's builders write,
  in any case, with ``key-value`` parts such as ``ses-1`` or
  ``acq-multi-shell`` allowed and an id that may hold underscores
  (``sub-NDAR_INV1``). A folder is searched with its subfolders, symbolic
  links followed, so a BIDS-style tree works as well as a flat folder; a list
  of folders and files works too. A ``.h5`` file whose name says neither a
  subject nor a modality is listed in the summary as not read.
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
modality its name says, the area, mask and coordinates the grid's size, and
the endpoint group, if stored, with every dataset present and of one
streamline count. ``validate=True`` reads every file in full and runs every
check of ``sbci validate``, the values among them: finite connectivity, unit
mass, an empty medial wall, endpoint indices on the grid.

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
import json
import os
import re
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
        "#NA",
    }
)
"""Table cells read as missing, after stripping surrounding space: pandas' defaults.

Case matters, as it does to pandas: ``none`` is a value, ``None`` is missing.
``load_cohort(missing=...)`` adds to them, and ``keep_default_missing=False``
reads only the markers it names."""

MISMATCH = ("refuse", "exclude", "report")
"""What :func:`load_cohort` does when files disagree on a setting."""

_EXTENSION = re.compile(r"\.(h5|hdf5)$", re.IGNORECASE)


class CohortError(SbciError, ValueError):
    """A cohort refused as a whole; :attr:`report` holds the rows built before it was."""

    def __init__(self, message: str, report: list | None = None) -> None:
        super().__init__(message)
        self.report = list(report or [])


def subject_id(value) -> str:
    """The id as files carry it: ``sub-`` and the label.

    Examples
    --------
    >>> subject_id(100307), subject_id("sub-100307"), subject_id(" 100307 ")
    ('sub-100307', 'sub-100307', 'sub-100307')
    """
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():  # a number read as a float
        text = text[:-2]
    return text if text.startswith("sub-") else f"sub-{text}"


def parse_name(name: str) -> tuple[str, str | None, str] | None:
    """``(subject, session, modality)`` from a computational file's name, or ``None``.

    The id runs from ``sub-`` to the first ``key-value`` part or the modality,
    so it may hold underscores; a value may hold hyphens; case does not matter
    in the modality or the extension.

    Examples
    --------
    >>> parse_name("sub-100307_sc.h5")
    ('sub-100307', None, 'sc')
    >>> parse_name("sub-01_ses-2_desc-concon_fc.h5")
    ('sub-01', '2', 'fc')
    >>> parse_name("sub-NDAR_INV1_acq-multi-shell_SC.H5")
    ('sub-NDAR_INV1', None, 'sc')
    >>> parse_name("sub-01_vertexarea.dscalar.nii") is None
    True
    """
    match = _EXTENSION.search(name)
    if match is None:
        return None
    parts = name[: match.start()].split("_")
    if len(parts) < 2 or not parts[0].startswith("sub-") or len(parts[0]) == 4:
        return None
    modality = parts[-1].lower()
    if modality not in spec.MODALITIES:
        return None
    label = [parts[0][4:]]
    index = 1
    while index < len(parts) - 1 and "-" not in parts[index]:
        label.append(parts[index])
        index += 1
    entities = {}
    for part in parts[index:-1]:
        if "-" not in part:
            return None
        key, value = part.split("-", 1)
        if not key or not value:
            return None
        entities[key.lower()] = value
    return "sub-" + "_".join(label), entities.get("ses"), modality


def _same_label(found: str | None, wanted) -> bool:
    """Whether a file's ``ses-`` label is the one asked for: ``1`` matches ``01``."""
    if found is None or wanted is None:
        return found is None and wanted is None
    wanted = str(wanted).strip()
    if found == wanted:
        return True
    return found.isdigit() and wanted.isdigit() and int(found) == int(wanted)


def _missing(value, missing=MISSING) -> bool:
    """A missing cell: ``None``, a NaN of any kind, or a string ``missing`` names."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        return text.strip() in missing
    try:
        return bool(value != value)  # NaN, and NaT, are unequal to themselves
    except TypeError:  # pandas' NA will not say whether it equals itself
        return True


def _read_table(table, subject_column, missing=MISSING):
    """The subject column's name, the ids in the table's order, and every column as a list."""
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
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"{path.name} is empty")
            header = [name.strip() for name in header]
            repeated = sorted({name for name in header if header.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"{path.name} has two columns named {repeated[0]!r}; which to use is not "
                    "for the loader to guess"
                )
            columns = {name: [] for name in header}
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue  # a blank line
                for name, cell in zip(header, row + [""] * (len(header) - len(row)), strict=False):
                    columns[name].append(cell)
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


_CODE = re.compile(r"[+-]?0\d")
"""Text that reads as a number but is a code: a leading zero, as in ``01`` or ``007``."""


def _as_column(values, missing=MISSING) -> np.ndarray:
    """Numbers as float64 with NaN for missing; anything else as strings with None for missing.

    Text that reads as a number is taken as one, except a code written with a
    leading zero (``01``, ``007``), which no quantity is: such a column stays
    text, which :func:`sbci.stats.design` makes a factor.
    """
    cells = [None if _missing(v, missing) else v for v in values]
    present = [v for v in cells if v is not None]
    coded = any(isinstance(v, str) and _CODE.match(v.strip()) for v in present)
    try:
        numbers = None if coded else [float(v) for v in present]
    except (TypeError, ValueError):
        numbers = None
    if numbers is not None:
        out = np.full(len(cells), np.nan)
        out[[i for i, v in enumerate(cells) if v is not None]] = numbers
        return out
    return np.array([None if v is None else str(v).strip() for v in cells], dtype=object)


def _find_files(files) -> tuple[list[Path], list[Path]]:
    """The computational files ``files`` names, and the ``.h5`` files whose names say nothing.

    Folders are searched with their subfolders, symbolic links followed (each
    real folder once).
    """
    items = [files] if isinstance(files, (str, Path)) else list(files)
    paths: dict[str, Path] = {}
    unnamed: dict[str, Path] = {}
    for item in items:
        path = Path(item)
        if path.is_dir():
            seen: set = set()
            for root, folders, names in os.walk(path, followlinks=True):
                real = os.path.realpath(root)
                if real in seen:
                    folders[:] = []
                    continue
                seen.add(real)
                folders.sort()
                for name in sorted(names):
                    found = Path(root) / name
                    if parse_name(name) is not None and found.is_file():
                        paths.setdefault(os.path.abspath(found), found)
                    elif _EXTENSION.search(name):
                        unnamed.setdefault(os.path.abspath(found), found)
        elif not path.exists():
            raise FileNotFoundError(f"no such folder or file: {path}")
        elif parse_name(path.name) is None:
            raise ValueError(
                f"{path.name!r} is not named sub-<id>[_<key>-<value>...]_<sc|fc>.h5, so its "
                "subject and modality are unknown"
            )
        else:
            paths.setdefault(os.path.abspath(path), path)
    return list(paths.values()), list(unnamed.values())


def _inspect(path: Path, modality: str, validate: bool) -> tuple[dict, dict]:
    """A file's metadata and the numbers the report carries; raises with the reason it fails.

    Checks the structure :meth:`~sbci.ContinuousConnectome.load` checks,
    without reading the arrays; ``validate`` reads them and checks the values.
    """
    from .io import read_header
    from .metadata import MetadataError

    header = read_header(path)
    metadata = header["metadata"]
    try:
        metadata.validate()
    except MetadataError as error:
        raise ValueError(str(error)) from None
    if metadata.modality != modality:
        raise ValueError(f"holds {metadata.modality}, though its name says {modality}")
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
        reasons = Counter(
            part for row in self.report if not row["included"] for part in row["reason"].split("; ")
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
            listed = ", ".join(f"{key} {value!r}" for key, value in shared.items())
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
    missing: str | Iterable[str] = (),
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
        cohort is the files alone. A file with two columns of one name is
        refused.
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
        from one and FC from another, are refused.
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
        (:data:`MISSING`): ``"-999"`` is one marker, a list several.
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
    named = (missing,) if isinstance(missing, str) else tuple(missing)  # "-999" is one marker
    missing = frozenset(str(marker).strip() for marker in named)
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
    for path in paths_found:
        subject, ses, modality = parse_name(path.name)
        if modality not in modalities:
            continue
        if chosen and not _same_label(ses, wanted[modality]):
            continue
        found.setdefault(subject, {}).setdefault(modality, []).append((path, ses))

    # The tables, joined on the subject: {column: {subject: cell}}.
    covariate_columns: dict[str, dict] = {}
    table_ids: list = []
    if table is not None:
        tables = [table] if isinstance(table, (str, Path, Mapping)) else list(table)
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

    # A file whose id is another subject's with a suffix -- sub-01_old_sc.h5 beside
    # sub-01_sc.h5 -- is one more of that subject's files, a duplicate to choose
    # between, rather than a subject of its own; unless a table lists it as one.
    for subject in sorted(found, key=len, reverse=True):
        if subject in in_table:
            continue
        base = next(
            (other for other in sorted(found, key=len) if subject.startswith(other + "_")), None
        )
        if base is not None:
            for modality, entries in found.pop(subject).items():
                found[base].setdefault(modality, []).extend(entries)

    # A subject's files have to come from one visit unless session= says otherwise; a
    # subject the caller leaves out is not asked.
    crossed: dict[str, str] = {}
    if not chosen:
        for subject, by_modality in sorted(found.items()):
            if subject in excluded_by_caller:
                continue
            labels = {
                m: sorted({"none" if s is None else s for _, s in entries})
                for m, entries in by_modality.items()
            }
            if len({label for each in labels.values() for label in each}) > 1:
                crossed[subject] = "; ".join(
                    f"{m} from {', '.join(labels[m])}" for m in modalities if m in labels
                )

    # Every subject seen, and why each is out.
    everyone = sorted(set(found) | in_table)
    reasons: dict[str, list[str]] = {subject: [] for subject in everyone}
    report: dict[str, dict] = {}
    settings_of: dict[str, dict] = {}
    for subject in everyone:
        row: dict = {"subject": subject, "included": True, "reason": ""}
        if subject in excluded_by_caller:
            reasons[subject].append(f"excluded by the caller: {excluded_by_caller[subject]}")
        if table is not None and subject not in in_table:
            reasons[subject].append("not in the table")
        made: dict = {}
        for modality in modalities:
            entries = found.get(subject, {}).get(modality, [])
            row[f"{modality}_file"] = str(entries[0][0]) if len(entries) == 1 else ""
            row[f"{modality}_session"] = entries[0][1] if len(entries) == 1 else None
            for key in _NUMBERS[modality]:
                row[key] = None
            if not entries:
                reasons[subject].append(f"no {modality} file")
                continue
            if len(entries) > 1:
                names = ", ".join(sorted(p.name for p, _ in entries))
                reasons[subject].append(f"{len(entries)} {modality} files: {names}")
                continue
            try:
                fields, numbers_found = _inspect(entries[0][0], modality, validate)
            except Exception as error:  # noqa: BLE001 - every reason a file fails is reported
                reasons[subject].append(f"{modality} file {entries[0][0].name}: {error}")
                continue
            row.update(numbers_found)
            made[modality] = {key: fields.get(key) for key in SETTINGS[modality]}
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
        raise CohortError(
            f"no subject passed: {common}. Do the table's ids match the files' "
            "(100307 and sub-100307 match; 0100307 does not)?"
            if tally
            else "no files of the asked modalities were found",
            report=rows,
        )
    files_out = {m: [found[s][m][0][0] for s in included] for m in modalities}
    covariates = {
        name: _as_column([cells.get(s) for s in included], missing)
        for name, cells in covariate_columns.items()
    }
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
    )
