"""Bundle a large cohort for Zenodo, and write the package's manifest for it.

    python tools/bundle_hcp_cohort.py bundle --data DIR --manifest manifest.csv --out BUNDLES
    python tools/bundle_hcp_cohort.py manifest --bundles BUNDLES --record 1234567 \
        [--doi 10.5281/...] [--release YYYY-MM-DD]

``bundle`` takes the per-subject files ``tools/build_hcp_cohort.py`` wrote
(``sub-*_sc.h5``, ``sub-*_fc.h5``) and the manifest they were built from
(``subject, age_years, sex``), and writes zip archives of ``--per-bundle``
subjects (24 by default) per modality, stored without compression since the
HDF5 files are compressed already, plus ``bundles.json`` recording every
bundle's and every member's size and SHA-256 and each subject's sex and
five-year age bin. Zenodo accepts at most a hundred files per record; 528
subjects give 44 bundles.

``manifest`` turns ``bundles.json`` and the Zenodo record id into
``src/sbci/data/<cohort>.json``, the manifest ``sbci download <cohort>``
reads once the cohort is added to ``sbci.download.MANIFESTS``. Run it after the
record is published, since the file URLs carry the record id. It was written
for the 528-subject HCP-Aging cohort, which is not to be released; nothing is
released this way at present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

COHORT = "hcp-ya-full"  # the default cohort name; --cohort sets it
FILE_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def age_bin(years: float) -> str:
    """Five-year bins, ``36-40``, ``41-45``, ... as the example cohort's manifest uses."""
    low = 36 + 5 * int((years - 36) // 5) if years >= 36 else 31
    return f"{low}-{low + 4}"


def bundle(args) -> int:
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subjects = []
    for row in rows:
        subject = row["subject"]
        files = {m: data / f"{subject}_{m}.h5" for m in ("sc", "fc")}
        missing = [m for m, p in files.items() if not p.exists()]
        if missing:
            print(f"{subject}: missing {missing}, skipped", file=sys.stderr)
            continue
        subjects.append((subject, row["sex"], age_bin(float(row["age_years"])), files))
    subjects.sort()
    bundles = []
    records = {}
    per = args.per_bundle
    for modality in ("sc", "fc"):
        for index, start in enumerate(range(0, len(subjects), per), start=1):
            name = f"{args.cohort}_{modality}_{index:02d}.zip"
            path = out / name
            members = []
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                for subject, _sex, _bin, files in subjects[start : start + per]:
                    source = files[modality]
                    z.write(source, arcname=source.name)
                    members.append(source.name)
                    records[(subject, modality)] = {
                        "bundle": name,
                        "bytes": os.path.getsize(source),
                        "sha256": sha256_of(source),
                    }
            bundles.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_of(path),
                    "members": members,
                }
            )
            print(
                f"wrote {name}: {len(members)} files, {path.stat().st_size / 1e9:.2f} GB",
                flush=True,
            )
    listing = {
        "cohort": args.cohort,
        "subjects": [
            {
                "subject": s,
                "sex": sex,
                "age_bin": b,
                "files": {m: records[(s, m)] for m in ("sc", "fc")},
            }
            for s, sex, b, _ in subjects
        ],
        "bundles": bundles,
    }
    (out / "bundles.json").write_text(json.dumps(listing, indent=1) + "\n")
    print(f"{len(subjects)} subjects in {len(bundles)} bundles; bundles.json written", flush=True)
    return 0


def manifest(args) -> int:
    listing = json.loads(Path(args.bundles, "bundles.json").read_text())
    for entry in listing["bundles"]:
        entry["url"] = FILE_URL.format(record=args.record, name=entry["name"])
    document = {
        "cohort": listing["cohort"],
        "description": (
            "Continuous connectomes built by tools/build_hcp_cohort.py on the ico4 grid, one SC "
            "file with its streamline endpoints (and, where there is FC, one FC file) per "
            "subject, in zip bundles of twenty-four subjects per modality."
        ),
        "host": "zenodo",
        "record": str(args.record),
        "doi": args.doi or "",
        "release": args.release,
        "bundles": listing["bundles"],
        "subjects": listing["subjects"],
    }
    target = Path(args.target)
    target.write_text(json.dumps(document, indent=1) + "\n")
    print(
        f"wrote {target}: {len(document['subjects'])} subjects, {len(document['bundles'])} bundles"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("bundle", help="write the zip bundles and bundles.json")
    b.add_argument("--data", required=True, help="directory of sub-*_sc.h5 and sub-*_fc.h5")
    b.add_argument("--manifest", required=True, help="subject, age_years, sex")
    b.add_argument("--out", required=True, help="where the bundles go")
    b.add_argument("--per-bundle", type=int, default=24, help="subjects per bundle (24)")
    b.add_argument("--cohort", default=COHORT, help=f"cohort name, the bundles' prefix ({COHORT})")
    b.set_defaults(run=bundle)
    m = commands.add_parser("manifest", help="write the package manifest from bundles.json")
    m.add_argument("--bundles", required=True, help="directory holding bundles.json")
    m.add_argument("--record", required=True, help="the Zenodo record id")
    m.add_argument("--doi", default="", help="the record's DOI")
    m.add_argument("--release", default="", help="release date, YYYY-MM-DD")
    m.add_argument(
        "--target", required=True, help="the manifest to write, src/sbci/data/<cohort>.json"
    )
    m.set_defaults(run=manifest)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
