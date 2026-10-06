"""A cohort from files on disk: subjects matched to a table, SC paired with FC, and a report.

:func:`load_cohort` is the way into the package for a study's data. It takes

* the computational files, named ``sub-<id>[_ses-<session>]_<sc|fc>.h5``: the
  names :func:`sbci.download.fetch_cohort` and the lab's builders write, other
  ``key-value`` parts allowed. A folder is searched with its subfolders, so a
  BIDS-style tree works as well as a flat folder; a list of paths works too.
* optionally a table of subjects, ``.csv`` or ``.tsv`` (or a dict of columns),
  with a ``subject`` or BIDS ``participant_id`` column. Ids match with or
  without the ``sub-`` prefix, so the HCP's ``Subject`` column of bare numbers
  matches ``sub-100307``.

It returns a :class:`LoadedCohort`: the subjects that have a usable file for
every modality asked for and a value for every covariate required, whose files
agree on how they were made; their files and covariates in one order; the
settings they share; and a report with one row for every subject seen in the
files or the table, saying whether it is in and, if not, why.

What a file has to pass
-----------------------
Each file's header is read -- its metadata and the sizes of its arrays, not
the connectivity itself -- and must load as :meth:`~sbci.ContinuousConnectome.load`
would accept it: complete metadata on the ico4 grid, the modality its name
says, arrays the grid's size. ``validate=True`` also runs every check of
``sbci validate`` (finite values, unit mass, an empty medial wall), which reads
every file in full.

Then the files of each modality have to agree on how they were made: the keys
in :data:`SETTINGS`, the kernel and bandwidth of SC among them. A cohort that
mixes two is refused by default, naming who differs; ``mismatch="exclude"``
keeps the settings most files share and leaves the others out, and
``mismatch="report"`` keeps everyone and lists the differences.

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

MISSING = frozenset({"", "na", "n/a", "nan", "null", "none", "<na>", "#n/a", "-nan"})
"""Table cells read as missing, compared without case or surrounding space."""

MISMATCH = ("refuse", "exclude", "report")
"""What :func:`load_cohort` does when files disagree on a setting."""

_NAME = re.compile(r"^(sub-[A-Za-z0-9]+)((?:_[A-Za-z]+-[A-Za-z0-9]+)*)_(sc|fc)\.(?:h5|hdf5)$")


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

    Examples
    --------
    >>> parse_name("sub-100307_sc.h5")
    ('sub-100307', None, 'sc')
    >>> parse_name("sub-01_ses-2_desc-concon_fc.h5")
    ('sub-01', '2', 'fc')
    >>> parse_name("sub-01_vertexarea.dscalar.nii") is None
    True
    """
    match = _NAME.match(name)
    if match is None:
        return None
    entities = dict(part.split("-", 1) for part in match.group(2).split("_")[1:])
    return match.group(1), entities.get("ses"), match.group(3)


def _missing(value) -> bool:
    """A missing cell: ``None``, a NaN of any kind, or a string :data:`MISSING` names."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        return text.strip().lower() in MISSING
    try:
        return bool(value != value)  # NaN, and NaT, are unequal to themselves
    except TypeError:  # pandas' NA will not say whether it equals itself
        return True


def _read_table(table, subject_column):
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
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} is empty")
            columns = {name: [] for name in reader.fieldnames}
            for row in reader:
                for name in columns:
                    columns[name].append(row.get(name))
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
        if _missing(value):
            raise ValueError(f"row {index + 1} of the table has no {subject_column}")
        ids.append(subject_id(value))
    repeated = sorted(name for name, count in Counter(ids).items() if count > 1)
    if repeated:
        shown = ", ".join(repeated[:5]) + (" ..." if len(repeated) > 5 else "")
        raise ValueError(f"the table lists {len(repeated)} subjects more than once: {shown}")
    return subject_column, ids, columns


def _as_column(values) -> np.ndarray:
    """Numbers as float64 with NaN for missing; anything else as strings with None for missing."""
    cells = [None if _missing(v) else v for v in values]
    present = [v for v in cells if v is not None]
    try:
        numbers = [float(v) for v in present]
    except (TypeError, ValueError):
        numbers = None
    if numbers is not None:
        out = np.full(len(cells), np.nan)
        out[[i for i, v in enumerate(cells) if v is not None]] = numbers
        return out
    return np.array([None if v is None else str(v).strip() for v in cells], dtype=object)


def _find_files(files) -> list[Path]:
    """The computational files ``files`` names: folders searched with their subfolders."""
    items = [files] if isinstance(files, (str, Path)) else list(files)
    paths: dict[str, Path] = {}
    for item in items:
        path = Path(item)
        if path.is_dir():
            found = sorted(p for p in path.rglob("sub-*") if p.is_file() and parse_name(p.name))
        elif not path.exists():
            raise FileNotFoundError(f"no such folder or file: {path}")
        elif parse_name(path.name) is None:
            raise ValueError(
                f"{path.name!r} is not named sub-<id>[_ses-<session>]_<sc|fc>.h5, so its "
                "subject and modality are unknown"
            )
        else:
            found = [path]
        for p in found:  # a file reached twice, through a folder and by name, counts once
            paths.setdefault(os.path.abspath(p), p)
    return list(paths.values())


def _inspect(path: Path, modality: str, validate: bool) -> tuple[dict, dict]:
    """A file's metadata and the numbers the report carries; raises with the reason it fails."""
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
    n = spec.N_VERTICES
    if header["n_vertices"] != n or header["n_mask"] != n:
        raise ValueError(f"has {header['n_vertices']} vertices, not the grid's {n}")
    if header["n_connectivity"] != grid.condensed_size(n):
        raise ValueError(f"stores {header['n_connectivity']} values, not {grid.condensed_size(n)}")
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
        runs = fields.get("fc_runs")
        numbers = {
            "fc_frames": fields.get("fc_frames"),
            "fc_runs": len(runs) if isinstance(runs, (list, tuple)) else runs,
        }
    return fields, numbers


def _key(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@dataclass
class LoadedCohort:
    """What :func:`load_cohort` returns: the subjects that passed, their files and covariates."""

    subjects: list
    """Ids of the subjects in the cohort, ``sub-<id>``, sorted."""
    files: dict
    """``{modality: [path, ...]}``, one path per subject, in :attr:`subjects`' order."""
    covariates: dict
    """``{column: array}``, every column of the table, in :attr:`subjects`' order: float64
    with ``NaN`` for missing where the column holds numbers, else strings with ``None``."""
    settings: dict
    """``{modality: {key: value}}``, the settings of :data:`SETTINGS` the files share (under
    ``mismatch="report"``, a differing key holds the list of its values)."""
    report: list
    """One dict per subject seen in the files or the table: ``subject``, ``included``,
    ``reason``, each modality's file, the numbers worth a look, the required covariates."""
    modalities: tuple = ("sc",)
    session: str | None = None
    required: tuple = field(default_factory=tuple)

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
            raise KeyError(f"no column {name!r}; the table has {', '.join(self.covariates)}")
        return self.covariates[name]

    def save_report(self, path) -> Path:
        """Write :attr:`report` as a ``.csv`` or ``.tsv`` table."""
        from .export import _write_table

        header = list(self.report[0]) if self.report else ["subject", "included", "reason"]
        rows = ([row.get(key) for key in header] for row in self.report)
        return _write_table(path, header, (["" if v is None else v for v in r] for r in rows))

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
        for modality, shared in self.settings.items():
            listed = ", ".join(f"{key} {value!r}" for key, value in shared.items())
            lines.append(f"  {modality} files share: {listed}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<LoadedCohort of {self.n_subjects} subjects ({', '.join(self.modalities)}), "
            f"{len(self.excluded)} left out>"
        )


def load_cohort(
    files,
    table=None,
    modalities=("sc",),
    session: str | None = None,
    require: Iterable[str] = (),
    subject_column: str | None = None,
    exclude: Mapping | None = None,
    validate: bool = False,
    mismatch: str = "refuse",
) -> LoadedCohort:
    """Gather a cohort's files and covariates, check them, and report who is left out and why.

    Parameters
    ----------
    files
        A folder holding ``sub-<id>[_ses-<session>]_<sc|fc>.h5`` files, searched
        with its subfolders, or a list of such folders and files.
    table
        A ``.csv`` or ``.tsv`` file, or a dict of columns, with one row per
        subject; or a list of them, joined on the subject (a subject missing
        from one has no value in its columns). Optional: without one the
        cohort is the files alone.
    modalities
        ``"sc"``, ``"fc"``, or both: what every subject needs a file of.
    session
        Use the files of this ``ses-`` label only. Required when a subject has
        files from several sessions, which are otherwise refused.
    require
        Table columns every subject needs a value in; a subject with a missing
        cell (empty, ``NA``, ``NaN`` and the like) is left out.
    subject_column
        The table's id column, when it is not one of :data:`SUBJECT_COLUMNS`.
    exclude
        ``{subject: reason}``, subjects to leave out on the caller's grounds,
        such as a quality check upstream; the reasons stand in the report.
    validate
        Run every check of ``sbci validate`` on every file, reading each in
        full, rather than its header only.
    mismatch
        When files of a modality disagree on a setting: ``"refuse"`` (the
        default) raises, naming who differs; ``"exclude"`` keeps the settings
        most files share and leaves the rest out; ``"report"`` keeps everyone
        and records the differences.

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
    required = (require,) if isinstance(require, str) else tuple(require)
    excluded_by_caller = {subject_id(k): str(v) for k, v in (exclude or {}).items()}

    # The files, by subject and modality, for the session asked for.
    found: dict[str, dict[str, list[Path]]] = {}
    sessions: dict[tuple[str, str], set] = {}
    for path in _find_files(files):
        subject, ses, modality = parse_name(path.name)
        if modality not in modalities:
            continue
        sessions.setdefault((subject, modality), set()).add(ses)
        if session is not None and ses != str(session):
            continue
        found.setdefault(subject, {}).setdefault(modality, []).append(path)
    if session is None:
        several = {key: s for key, s in sessions.items() if len(s) > 1}
        if several:
            (subject, modality), labels = next(iter(sorted(several.items())))
            shown = sorted("none" if s is None else s for s in labels)
            raise ValueError(
                f"{subject} has {modality} files from sessions {shown} ({len(several)} subject "
                "and modality pairs have several): choose one with session="
            )

    # The tables, joined on the subject: {column: {subject: cell}}.
    covariate_columns: dict[str, dict] = {}
    table_ids: list = []
    if table is not None:
        tables = [table] if isinstance(table, (str, Path, Mapping)) else list(table)
        for one in tables:
            id_column, ids, columns = _read_table(one, subject_column)
            for name, values in columns.items():
                if name == id_column:
                    continue
                if name in covariate_columns:
                    raise ValueError(f"two tables have a column {name!r}; rename one")
                covariate_columns[name] = dict(zip(ids, values, strict=True))
            table_ids += [s for s in ids if s not in set(table_ids)]
        missing_columns = [name for name in required if name not in covariate_columns]
        if missing_columns:
            raise ValueError(f"no table has a column {missing_columns[0]!r}")
    elif required:
        raise ValueError("require= names table columns, but no table was given")
    in_table = set(table_ids)

    # Every subject seen, and why each is out.
    everyone = sorted(set(found) | set(table_ids))
    reasons: dict[str, list[str]] = {subject: [] for subject in everyone}
    report: dict[str, dict] = {}
    settings_of: dict[str, dict[str, dict]] = {m: {} for m in modalities}
    numbers = {"sc": ("sc_streamlines", "sc_endpoints"), "fc": ("fc_frames", "fc_runs")}
    for subject in everyone:
        row: dict = {"subject": subject, "included": True, "reason": ""}
        if subject in excluded_by_caller:
            reasons[subject].append(f"excluded by the caller: {excluded_by_caller[subject]}")
        if table is not None and subject not in in_table:
            reasons[subject].append("not in the table")
        for modality in modalities:
            paths = found.get(subject, {}).get(modality, [])
            row[f"{modality}_file"] = str(paths[0]) if len(paths) == 1 else ""
            for key in numbers[modality]:
                row[key] = None
            if not paths:
                reasons[subject].append(f"no {modality} file")
                continue
            if len(paths) > 1:
                names = ", ".join(sorted(p.name for p in paths))
                reasons[subject].append(f"{len(paths)} {modality} files: {names}")
                continue
            try:
                fields, numbers_found = _inspect(paths[0], modality, validate)
            except Exception as error:  # noqa: BLE001 - every reason a file fails is reported
                reasons[subject].append(f"{modality} file {paths[0].name}: {error}")
                continue
            row.update(numbers_found)
            settings_of[modality][subject] = {key: fields.get(key) for key in SETTINGS[modality]}
        for name in required:
            value = covariate_columns[name].get(subject)
            row[name] = None if _missing(value) else value
        if subject in in_table:
            empty = [name for name in required if _missing(covariate_columns[name].get(subject))]
            if empty:
                reasons[subject].append("no value for " + ", ".join(empty))
        report[subject] = row

    # The settings, over the subjects not already out.
    shared: dict[str, dict] = {}
    for modality in modalities:
        candidates = [s for s in everyone if not reasons[s] and s in settings_of[modality]]
        shared[modality] = {}
        for key in SETTINGS[modality]:
            counts = Counter(_key(settings_of[modality][s][key]) for s in candidates)
            if len(counts) <= 1:
                if counts:
                    shared[modality][key] = settings_of[modality][candidates[0]][key]
                continue
            ranked = counts.most_common()
            described = "; ".join(
                f"{json.loads(value)!r} in {count} file{'s' if count > 1 else ''}"
                for value, count in ranked
            )
            if mismatch == "refuse" or (mismatch == "exclude" and ranked[0][1] == ranked[1][1]):
                odd = [s for s in candidates if _key(settings_of[modality][s][key]) != ranked[0][0]]
                remedy = (
                    "no setting has a majority to keep; choose the files with exclude="
                    if mismatch == "exclude"
                    else "pass mismatch='exclude' to keep the majority, or 'report' to keep "
                    "everyone"
                )
                raise ValueError(
                    f"the {modality} files differ in {key}: {described} (first of the others: "
                    f"{', '.join(odd[:3])}). Mixing them makes the subjects incomparable; {remedy}"
                )
            if mismatch == "exclude":
                majority = ranked[0][0]
                for s in candidates:
                    value = settings_of[modality][s][key]
                    if _key(value) != majority:
                        reasons[s].append(
                            f"{modality} {key} is {value!r}, not the cohort's "
                            f"{json.loads(majority)!r}"
                        )
                shared[modality][key] = json.loads(majority)
            else:
                shared[modality][key] = [json.loads(value) for value, _ in ranked]

    for subject in everyone:
        if reasons[subject]:
            report[subject]["included"] = False
            report[subject]["reason"] = "; ".join(reasons[subject])
    included = [s for s in everyone if not reasons[s]]
    if not included:
        tally = Counter(part for s in everyone for part in reasons[s])
        common = "; ".join(f"{count} {reason}" for reason, count in tally.most_common(3))
        raise ValueError(
            f"no subject passed: {common}. Do the table's ids match the files' "
            "(100307 and sub-100307 match; 0100307 does not)?"
            if tally
            else "no files of the asked modalities were found"
        )
    files_out = {m: [found[s][m][0] for s in included] for m in modalities}
    covariates = {
        name: _as_column([cells.get(s) for s in included])
        for name, cells in covariate_columns.items()
    }
    return LoadedCohort(
        subjects=included,
        files=files_out,
        covariates=covariates,
        settings=shared,
        report=[report[s] for s in everyone],
        modalities=modalities,
        session=None if session is None else str(session),
        required=required,
    )
