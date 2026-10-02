"""Upload the cohort bundles to a Zenodo deposition, ready for the owner to review and publish.

    export ZENODO_TOKEN=...                       # a personal access token with deposit:write
    python tools/zenodo_upload.py --bundles BUNDLES --title TITLE [--sandbox] [--deposition ID]

Creates a new deposition (or continues ``--deposition``), uploads every zip in
``BUNDLES`` plus ``bundles.json`` through the deposition's file bucket,
skipping files already there with the right size, and sets the record's
metadata from ``--title``, ``--creators`` and ``--description``. It stops
short of publishing: open the deposition's page, check it, and press
Publish; then run ``tools/bundle_hcp_cohort.py manifest --bundles BUNDLES --record ID
--target src/sbci/data/<cohort>.json``.

The token is read from the environment and never written anywhere. With
``--sandbox`` everything goes to sandbox.zenodo.org, which takes its own
tokens and is the place to try this first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HOSTS = {False: "https://zenodo.org", True: "https://sandbox.zenodo.org"}
CHUNK = 1 << 22


def api(base: str, token: str, method: str, path: str, payload=None, raw: bytes | None = None):
    """One JSON call to the deposition API; ``raw`` uploads bytes instead."""
    url = path if path.startswith("http") else f"{base}/api{path}"
    headers = {"Authorization": f"Bearer {token}"}
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=600) as response:
            body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"{method} {url} failed: HTTP {error.code} {detail}") from error
    return json.loads(body) if body else {}


class Streamed:
    """A file object that reports progress while urllib streams it."""

    def __init__(self, path: Path, report):
        self.handle = open(path, "rb")
        self.size = path.stat().st_size
        self.done = 0
        self.report = report
        self.last = -1

    def read(self, size=-1):
        """Read a block and report the running total."""
        block = self.handle.read(size)
        self.done += len(block)
        step = self.done * 20 // max(self.size, 1)
        if step != self.last:
            self.last = step
            self.report(f"    {self.done / 1e9:.2f} of {self.size / 1e9:.2f} GB")
        return block

    def __len__(self):
        return self.size


def upload(base, token, bucket, path: Path, present: dict, report) -> None:
    if present.get(path.name) == path.stat().st_size:
        report(f"  {path.name}: already uploaded")
        return
    report(f"  uploading {path.name}")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
    request = Request(
        f"{bucket}/{path.name}", data=Streamed(path, report), headers=headers, method="PUT"
    )
    request.add_header("Content-Length", str(path.stat().st_size))
    try:
        with urlopen(request, timeout=3600) as response:
            response.read()
    except HTTPError as error:
        raise SystemExit(f"upload of {path.name} failed: HTTP {error.code}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bundles", required=True, help="directory of the zip bundles and bundles.json"
    )
    parser.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    parser.add_argument("--deposition", help="continue an existing deposition id")
    parser.add_argument("--title", required=True, help="the record's title")
    parser.add_argument(
        "--creators",
        default="Zhang, Zhengwu",
        help="semicolon-separated 'Family, Given[|Affiliation]' entries",
    )
    parser.add_argument(
        "--description",
        default=(
            "Continuous connectomes on the fsaverage ico4 grid (5,124 vertices), with their "
            "streamline endpoints, in the SBCI package's HDF5 format; bundles.json lists every "
            "file with its SHA-256 and each subject's demographics where released. Read them with "
            "the sbci Python package."
        ),
    )
    args = parser.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("set ZENODO_TOKEN in the environment first", file=sys.stderr)
        return 1
    base = HOSTS[args.sandbox]
    report = lambda line: print(line, flush=True)  # noqa: E731

    if args.deposition:
        deposition = api(base, token, "GET", f"/deposit/depositions/{args.deposition}")
    else:
        deposition = api(base, token, "POST", "/deposit/depositions", payload={})
        report(f"created deposition {deposition['id']}")
    bucket = deposition["links"]["bucket"]
    present = {f["filename"]: f["filesize"] for f in deposition.get("files", [])}

    creators = []
    for item in args.creators.split(";"):
        name, _, affiliation = item.strip().partition("|")
        creator = {"name": name.strip()}
        if affiliation.strip():
            creator["affiliation"] = affiliation.strip()
        creators.append(creator)
    metadata = {
        "title": args.title,
        "upload_type": "dataset",
        "description": args.description,
        "creators": creators,
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": [
            "connectome",
            "Human Connectome Project",
            "SBCI",
            "diffusion MRI",
            "fMRI",
            "cortical surface",
        ],
    }
    api(
        base,
        token,
        "PUT",
        f"/deposit/depositions/{deposition['id']}",
        payload={"metadata": metadata},
    )
    report("metadata set")

    folder = Path(args.bundles)
    for path in sorted(folder.glob("*.zip")) + [folder / "bundles.json"]:
        upload(base, token, bucket, path, present, report)
    report(
        f"done: review and publish at {base}/deposit/{deposition['id']} ; then "
        f"python tools/bundle_hcp_cohort.py manifest --bundles {folder} --record <published id> "
        "--target src/sbci/data/<cohort>.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
