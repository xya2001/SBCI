"""The ``sbci`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .errors import SbciError
from .validate import validate_file


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sbci", description="Continuous brain connectivity toolkit"
    )
    parser.add_argument("--version", action="version", version=f"sbci {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download",
        help="fetch the example cohort: hcp-ya, eleven HCP Young Adult subjects, SC and FC, "
        "about 1 GB",
    )
    download.add_argument("cohort", help="cohort name: hcp-ya")
    download.add_argument(
        "--subject", action="append", help="one subject id, e.g. 100307 or sub-100307; repeatable"
    )
    download.add_argument(
        "--out", default=".", help="destination directory (default: the current directory)"
    )
    modality = download.add_mutually_exclusive_group()
    modality.add_argument("--sc-only", action="store_true", help="structural files only")
    modality.add_argument("--fc-only", action="store_true", help="functional files only")
    download.add_argument("--force", action="store_true", help="re-download files already present")
    download.add_argument(
        "--keep-bundles",
        action="store_true",
        help="for a cohort released as zip bundles (none at present): keep the bundles after "
        "extracting their files",
    )

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
    example.add_argument("--seed", type=int, default=0, help="generator seed")
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
    cohort.add_argument("--report", help="write the per-subject report here (.csv or .tsv)")

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
        from .cohort import load_cohort

        modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
        try:
            loaded = load_cohort(
                args.folder,
                table=args.table,
                modalities=modalities,
                session=args.session,
                require=args.require,
                validate=args.validate,
                mismatch=args.mismatch,
            )
            print(loaded.summary())
            if args.report:
                print(f"report written to {loaded.save_report(args.report)}")
        except (ValueError, OSError, SbciError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0

    if args.command == "atlases":
        from .atlas import list_atlases, load_atlas

        names = list_atlases()
        if args.match:
            names = tuple(n for n in names if args.match.lower() in n.lower())
        if not names:
            print(f"no bundled atlas matches {args.match!r}")
            return 1
        for name in names:
            print(f"  {name:48s} {load_atlas(name).n_regions:5d} regions")
        return 0

    if args.command == "download":
        from .download import fetch_cohort

        modalities = ("sc",) if args.sc_only else ("fc",) if args.fc_only else ("sc", "fc")
        try:
            written = fetch_cohort(
                args.out,
                cohort=args.cohort,
                subjects=args.subject,
                modalities=modalities,
                force=args.force,
                keep_bundles=args.keep_bundles,
            )
        except (ValueError, OSError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{len(written)} file(s) in {args.out}")
        return 0

    return 0  # pragma: no cover - unreachable, subcommand is required


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
