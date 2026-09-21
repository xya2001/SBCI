"""The ``sbci`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .validate import validate_file


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sbci", description="Continuous brain connectivity toolkit"
    )
    parser.add_argument("--version", action="version", version=f"sbci {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch a released cohort or tutorial subject")
    download.add_argument("cohort", help="cohort name, e.g. hcp-ya")
    download.add_argument("--subject", help="single subject id, e.g. 100307")
    download.add_argument("--out", default=".", help="destination directory")

    validate = subparsers.add_parser("validate", help="check a file against the spec")
    validate.add_argument("path", help="path to a .h5 computational file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        checks = validate_file(args.path)
        for check in checks:
            print(check)
        return 0 if all(c.passed for c in checks) else 1

    if args.command == "download":
        raise SystemExit(
            "`sbci download` needs the WP1 data release: a stable URL and a "
            "checksum manifest for the tutorial subject and the HCP-YA cohort. "
            "Neither exists yet -- see SPEC_QUESTIONS.md item 6."
        )

    return 0  # pragma: no cover - unreachable, subcommand is required


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
