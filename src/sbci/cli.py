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

    download = subparsers.add_parser(
        "download", help="fetch a released cohort or tutorial subject (pending the data release)"
    )
    download.add_argument("cohort", help="cohort name, e.g. hcp-ya")
    download.add_argument("--subject", help="single subject id, e.g. 100307")
    download.add_argument("--out", default=".", help="destination directory")

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
    example.add_argument("--out", default="sub-example_sc.h5", help="destination file")

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
        return _info(args.path)

    if args.command == "example":
        from .examples import example as _example

        connectome = _example(args.modality, seed=args.seed)
        path = connectome.save(args.out or f"sub-example_{args.modality}.h5")
        print(f"wrote {path}")
        print("  synthetic connectivity on the real ico4 grid -- not measured data")
        if connectome.has_endpoints:
            print(
                f"  carries {connectome.endpoints.n_streamlines:,} synthetic streamline "
                "endpoints, so smooth() and endpoints_align() run on it"
            )
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
        raise SystemExit(
            "`sbci download` needs the WP1 data release: a stable URL and a "
            "checksum manifest for the tutorial subject and the HCP-YA cohort. "
            "Neither exists yet -- see SPEC_QUESTIONS.md item 6."
        )

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
        dense = connectome.dense().astype("float64")
        print(f"  total mass      {float(connectome.area @ dense @ connectome.area):.6g}")
    print(f"  value range     [{connectome.data.min():.6g}, {connectome.data.max():.6g}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
