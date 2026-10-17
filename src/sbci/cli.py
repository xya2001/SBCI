"""The ``sbci`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .errors import SbciError
from .validate import validate_file


def _seed(text: str) -> int:
    """A generator seed, as numpy takes it: a whole number, 0 or more."""
    import argparse

    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a seed is a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"a seed is 0 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sbci", description="Continuous brain connectivity toolkit"
    )
    parser.add_argument("--version", action="version", version=f"sbci {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download",
        help="fetch HCP Young Adult connectomes from UNC Dataverse: the tutorial's eleven "
        "subjects, about 1 GB, any of the 946 by id, or all of them, 85 GB",
    )
    download.add_argument("cohort", help="cohort name: hcp-ya")
    which = download.add_mutually_exclusive_group()
    which.add_argument(
        "--subject",
        action="append",
        help="one subject id, e.g. 100307 or sub-100307, any of the 946; repeatable",
    )
    which.add_argument("--all", action="store_true", help="every subject, 946, about 85 GB")
    which.add_argument(
        "--list",
        action="store_true",
        help="list the subjects, with sex, age band and size, and download nothing",
    )
    download.add_argument(
        "--out", default=".", help="destination directory (default: the current directory)"
    )
    modality = download.add_mutually_exclusive_group()
    modality.add_argument("--sc-only", action="store_true", help="structural files only")
    modality.add_argument("--fc-only", action="store_true", help="functional files only")
    download.add_argument("--force", action="store_true", help="re-download files already present")
    download.add_argument(
        "--accept-terms",
        action="store_true",
        help="accept the WU-Minn HCP Open Access Data Use Terms, having read them; without "
        "it the first download on a machine asks",
    )

    basis = subparsers.add_parser(
        "basis",
        help="fetch the Laplace-Beltrami basis the rdk and matern kernels need, 100 MB, once; "
        "smooth() fetches it too, where the network can be reached",
    )
    basis.add_argument(
        "--out",
        help="where to put it (default: $SBCI_LBO_DIR, else ~/.cache/sbci/lbo, where smooth() "
        "looks)",
    )
    basis.add_argument("--force", action="store_true", help="fetch it again even if present")

    validate = subparsers.add_parser("validate", help="check a file against the spec")
    validate.add_argument("path", help="path to a .h5 computational file")

    info = subparsers.add_parser("info", help="summarize a connectome file")
    info.add_argument("path", help="path to a .h5 computational file")

    example = subparsers.add_parser(
        "example", help="write a synthetic connectome, to try the package without data"
    )
    example.add_argument(
        "--modality", default="sc", choices=("sc", "fc"), help="which kind to build"
    )
    example.add_argument("--seed", type=_seed, default=0, help="generator seed, 0 or more")
    example.add_argument(
        "--out", default=None, help="destination file (default: sub-example_<modality>.h5)"
    )

    cohort = subparsers.add_parser(
        "cohort",
        help="check a folder of sub-<id>_<sc|fc>.h5 files against a subject table: who is in, "
        "who is left out, and why",
    )
    cohort.add_argument(
        "folder", nargs="+", help="folders of computational files, searched with subfolders"
    )
    cohort.add_argument(
        "--table",
        action="append",
        help="a .csv or .tsv with one row per subject, and a subject or participant_id column; "
        "repeatable, the tables joined on the subject",
    )
    cohort.add_argument(
        "--modalities", default="sc", help="what every subject needs: sc, fc or sc,fc (default sc)"
    )
    cohort.add_argument("--session", help="use only the files of this ses- label")
    cohort.add_argument(
        "--require",
        action="append",
        default=[],
        help="a table column every subject needs a value in; repeatable",
    )
    cohort.add_argument(
        "--validate",
        action="store_true",
        help="run every check of sbci validate on every file, reading each in full",
    )
    cohort.add_argument(
        "--mismatch",
        default="refuse",
        choices=("refuse", "exclude", "report"),
        help="when files disagree on how they were made: refuse (default), exclude the minority, "
        "or report and keep everyone",
    )
    cohort.add_argument(
        "--subject-column", help="the tables' id column, if not subject or the like"
    )
    cohort.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBJECT=REASON",
        help="leave a subject out on grounds decided upstream; repeatable",
    )
    cohort.add_argument(
        "--report",
        help="write the per-subject report here (.csv or .tsv), the cohort refused or not",
    )

    atlases = subparsers.add_parser("atlases", help="list the bundled atlases")
    atlases.add_argument("--match", help="only names containing this text")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        checks = validate_file(args.path)
        for check in checks:
            print(check)
        return 0 if all(c.passed for c in checks) else 1

    if args.command == "info":
        try:
            return _info(args.path)
        except (SbciError, OSError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    if args.command == "example":
        from .examples import example as _example

        connectome = _example(args.modality, seed=args.seed)
        try:
            path = connectome.save(args.out or f"sub-example_{args.modality}.h5")
        except (ValueError, OSError) as error:  # a name load() would refuse, or no such directory
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"wrote {path}")
        print("  synthetic connectivity on the real ico4 grid -- not measured data")
        if connectome.has_endpoints:
            print(
                f"  carries {connectome.endpoints.n_streamlines:,} synthetic streamline "
                "endpoints, so smooth() and endpoints_align() run on it"
            )
        return 0

    if args.command == "cohort":
        from pathlib import Path

        from .cohort import CohortError, _write_report, load_cohort
        from .export import TABLE_SUFFIXES, _form, _target

        if args.report:
            # before the scan, which takes a while on hundreds of files, not after it:
            # the folder and the file itself, a read-only one refused as the write would be
            try:
                _form(Path(args.report), TABLE_SUFFIXES)
                if Path(args.report).is_dir():
                    raise IsADirectoryError(f"{args.report} is a folder; name a file in it")
                _target(Path(args.report))
            except (ValueError, OSError) as error:
                print(f"error: --report: {error}", file=sys.stderr)
                return 1
        modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
        exclude = {}
        for item in args.exclude:
            subject, sep, reason = item.partition("=")
            if not sep or not subject.strip():
                print(f"error: --exclude takes SUBJECT=REASON, got {item!r}", file=sys.stderr)
                return 1
            exclude[subject.strip()] = reason.strip() or "excluded"
        try:
            loaded = load_cohort(
                args.folder,
                table=args.table,
                modalities=modalities,
                session=args.session,
                require=args.require,
                subject_column=args.subject_column,
                exclude=exclude,
                validate=args.validate,
                mismatch=args.mismatch,
            )
            print(loaded.summary())
            if args.report:
                print(f"report written to {loaded.save_report(args.report)}")
        except CohortError as error:
            print(f"error: {error}", file=sys.stderr)
            if args.report and error.report:
                try:
                    print(f"report written to {_write_report(error.report, args.report)}")
                except (ValueError, OSError) as failure:
                    print(f"error: the report could not be written: {failure}", file=sys.stderr)
            return 1
        except (ValueError, OSError, SbciError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0

    if args.command == "atlases":
        from .atlas import ALIASES, DISPLAY, _key, list_atlases, load_atlas

        names = list_atlases()
        # The short names load_atlas takes, Schaefer200 and the like, are found and shown too.
        known_as: dict[str, list[str]] = {}
        for alias, target in ALIASES.items():
            known_as.setdefault(target, []).append(DISPLAY.get(alias, alias.title()))
        if args.match:
            wanted = _key(args.match)
            names = tuple(
                n
                for n in names
                if wanted in _key(n) or any(wanted in _key(a) for a in known_as.get(n, []))
            )
        if not names:
            print(f"no bundled atlas matches {args.match!r}")
            return 1
        for name in names:
            label = name + (f" ({', '.join(known_as[name])})" if name in known_as else "")
            print(f"  {label:48s} {load_atlas(name).n_regions:5d} regions")
        return 0

    if args.command == "download":
        from . import download

        modalities = ("sc",) if args.sc_only else ("fc",) if args.fc_only else ("sc", "fc")
        subjects = "all" if args.all else args.subject
        if args.list:
            try:
                return _list_subjects(download.load_manifest(args.cohort), modalities)
            except ValueError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
        try:
            # what is asked for is checked first, so a wrong id is not met with the terms
            download._check_request(download.load_manifest(args.cohort), subjects, modalities)
            if not args.accept_terms and not download.terms_accepted(args.cohort):
                if not _ask_terms(args.cohort):
                    return 1
            written = download.fetch_cohort(
                args.out,
                cohort=args.cohort,
                subjects=subjects,
                modalities=modalities,
                force=args.force,
                accept_terms=args.accept_terms,
            )
        except (ValueError, OSError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{len(written)} file(s) in {args.out}")
        return 0

    if args.command == "basis":
        from pathlib import Path

        from . import download

        try:
            where = download.fetch_basis(args.out, force=args.force)
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"the Laplace-Beltrami basis is in {where}")
        if args.out is not None and Path(args.out).resolve() != download.basis_dir().resolve():
            print(f"  smooth() looks elsewhere: point SBCI_LBO_DIR at {where}, or pass eigenpairs=")
        return 0

    return 0  # pragma: no cover - unreachable, subcommand is required


def _list_subjects(manifest: dict, modalities: tuple) -> int:
    """Print the cohort's subjects, one a line: sex, age band, size, the tutorial's marked."""
    entries = manifest["subjects"]
    if not entries:
        print(f"the {manifest.get('cohort', 'requested')} cohort has not been released yet")
        return 1
    print(f"{'subject':<12}{'sex':<5}{'age':<7}{'MB':>7}")
    total = 0
    for entry in entries:
        size = sum(int(entry["files"][m]["bytes"]) for m in modalities if m in entry["files"])
        total += size
        mark = "  tutorial" if entry.get("tutorial") else ""
        print(
            f"{entry['subject']:<12}{entry.get('sex', ''):<5}{entry.get('age_bin', ''):<7}"
            f"{size / 1e6:>7.1f}{mark}"
        )
    tutorial = sum(1 for e in entries if e.get("tutorial"))
    print(
        f"{len(entries)} subjects, {total / 1e9:.1f} GB of {' and '.join(modalities)}; "
        f"`sbci download hcp-ya` fetches the {tutorial} marked tutorial, --subject ID any one "
        "(repeatable), --all every one"
    )
    return 0


def _ask_terms(cohort: str) -> bool:
    """Show the cohort's terms and record them accepted on "I agree"; asked once a machine."""
    from .download import TERMS_NOTICE, accept_terms

    if not sys.stdin.isatty():
        print(
            "error: these files come under the WU-Minn HCP Consortium Open Access Data Use "
            "Terms, which must be accepted once on this machine: run this command in a "
            "terminal and type I agree when asked, or, having read them, add --accept-terms",
            file=sys.stderr,
        )
        return False
    print(TERMS_NOTICE)
    try:
        answer = input('Type "I agree" to accept them (asked once on this machine): ')
    except EOFError:
        answer = ""
    if answer.strip().lower() != "i agree":
        print("error: the terms were not accepted; nothing was downloaded", file=sys.stderr)
        return False
    print(f"accepted, and recorded in {accept_terms(cohort)}")
    return True


def _info(path: str) -> int:
    """Print what a file holds, so a user need not open Python to look."""
    from .connectome import ContinuousConnectome

    connectome = ContinuousConnectome.load(path)
    fields = connectome.metadata.fields
    print(f"{path}")
    print(f"  modality        {connectome.modality}")
    print(f"  vertices        {connectome.n_vertices} ({int(connectome.mask.sum())} cortex)")
    print(f"  grid            {fields.get('grid')}")
    print(f"  spec version    {fields.get('spec_version')}")
    print(f"  normalization   {fields.get('normalization')}")
    if connectome.modality == "sc":
        print(f"  kernel          {fields.get('kernel')} (bandwidth {fields.get('bandwidth')})")
        print(f"  streamlines     {fields.get('streamline_count')}")
        print(f"  endpoints       {'yes' if connectome.has_endpoints else 'no'}")
    else:
        print(f"  nuisance model  {fields.get('fc_nuisance_model')}")
    print(f"  pipeline        {fields.get('pipeline_version')}")
    if connectome.modality == "sc":
        # A density's mass is area-weighted; the bare sum of entries is not it.
        dense = connectome.dense(np.float64)
        print(f"  total mass      {float(connectome.area @ dense @ connectome.area):.6g}")
    print(f"  value range     [{connectome.data.min():.6g}, {connectome.data.max():.6g}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
