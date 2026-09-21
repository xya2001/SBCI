"""Convert SBCI_Toolkit atlas files into the package's canonical form.

Run once, commit the output. The release must not depend on a MATLAB step or on
a checkout of SBCI_Toolkit being present.

    python tools/convert_atlases.py --source <SBCI_Toolkit>/example_data/fsaverage_label

Input
-----
The toolkit ships two different layouts, both on the 5124-vertex ico4 grid:

``('labels', 'names', 'sorted_idx')``
    The CoCoNest family. ``labels`` is ``(1, 5124)`` and its values are
    arbitrary identifiers running into the thousands, not positions.
    ``names[i]`` names the ``i``-th *sorted* label.

``('colors', 'fs_labels', 'names', 'sbci_labels', 'sorted_idx')``
    The FreeSurfer-derived atlases. ``sbci_labels`` is ``(5124,)`` with
    ``names[i]`` naming label ``i``. ``fs_labels`` is the per-hemisphere
    labelling and is *not* what we want: for Schaefer it carries 100 parcels
    where ``sbci_labels`` carries the 200 bilateral ones.

Output
------
One ``<name>_ico4.npz`` per atlas holding

``labels``
    ``int32 (5124,)``, ``0`` for unassigned, regions numbered ``1..K``
    contiguously.
``names``
    ``(K,)`` unicode, ``names[i]`` naming label ``i + 1``.

Why the renumbering matters
---------------------------
Most FreeSurfer-derived atlases carry **two** background entries, one per
hemisphere, and the right-hemisphere one sits in the middle of the list with a
non-zero label -- Schaefer200 has ``LH_Background...`` at label 0 and
``RH_Background...`` at label 101. Keeping that as a region yields a 201x201
matrix with one row and column of pure medial wall. Every background entry is
mapped to 0 here, and the remainder renumbered, so a 200-parcel atlas gives
exactly 200 regions.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

N_VERTICES = 5124

# Names meaning "this vertex belongs to no region". Matched case-insensitively
# against the region name; every match is folded into label 0.
BACKGROUND = re.compile(r"(missing|unknown|background|medial[_ ]?wall|\?\?\?)", re.IGNORECASE)


def _decode(names) -> list[str]:
    """Names arrive as bytes or numpy unicode depending on the source layout."""
    return [(n.decode() if isinstance(n, bytes) else str(n)).strip() for n in names]


def normalize(path: str) -> tuple[np.ndarray, list[str], dict]:
    """Read one source atlas and return canonical ``(labels, names, report)``."""
    with np.load(path, allow_pickle=True) as z:
        names = _decode(z["names"])
        key = "sbci_labels" if "sbci_labels" in z.files else "labels"
        labels = np.asarray(z[key]).ravel()

    if labels.size != N_VERTICES:
        raise ValueError(f"{path}: {labels.size} vertices, expected {N_VERTICES}")

    unique = np.unique(labels)
    if len(names) != unique.size:
        raise ValueError(f"{path}: {len(names)} names for {unique.size} labels")

    # How a name's position maps to a label value differs between the two
    # layouts; everything downstream works in label values.
    if unique[0] == 0:
        if not np.array_equal(unique, np.arange(unique.size)):
            raise ValueError(f"{path}: labels start at 0 but are not contiguous")
        index_to_label = {i: i for i in range(len(names))}
    else:
        index_to_label = {i: int(value) for i, value in enumerate(unique)}

    background = {index_to_label[i] for i, name in enumerate(names) if BACKGROUND.search(name)}

    kept = [
        (index_to_label[i], name)
        for i, name in enumerate(names)
        if index_to_label[i] not in background
    ]
    kept.sort()

    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for new_id, (old_id, _) in enumerate(kept, start=1):
        remap[old_id] = new_id

    out_labels = remap[labels].astype(np.int32)
    out_names = [name for _, name in kept]

    report = {
        "source_layout": "coconest" if key == "labels" else "freesurfer",
        "n_regions": len(out_names),
        "n_background_entries": len(background),
        "unassigned_vertices": int((out_labels == 0).sum()),
        "coverage": 1.0 - float((out_labels == 0).sum()) / N_VERTICES,
    }
    return out_labels, out_names, report


def main(argv: list[str] | None = None) -> int:
    """Convert every atlas under ``--source`` into ``--dest``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="fsaverage_label directory of SBCI_Toolkit")
    parser.add_argument("--dest", required=True, help="src/sbci/data/atlases")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.10,
        help="skip atlases covering less of the surface than this",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    sources = sorted(glob.glob(os.path.join(args.source, "*_avg_roi_ico4.npz")))
    if not sources:
        print(f"no atlases found under {args.source}", file=sys.stderr)
        return 1

    written, skipped = 0, []
    print(f"{'atlas':44s} {'layout':11s} {'regions':>8s} {'coverage':>9s}  {'bg':>3s}")
    print("-" * 82)

    for path in sources:
        atlas = os.path.basename(path).replace("_avg_roi_ico4.npz", "")
        try:
            labels, names, report = normalize(path)
        except ValueError as exc:
            skipped.append((atlas, str(exc)))
            continue

        if report["coverage"] < args.min_coverage:
            skipped.append((atlas, f"covers only {report['coverage']:.1%} of the surface"))
            continue

        np.savez_compressed(
            os.path.join(args.dest, f"{atlas}_ico4.npz"),
            labels=labels,
            names=np.array(names, dtype=np.str_),
        )
        written += 1
        print(
            f"{atlas:44s} {report['source_layout']:11s} {report['n_regions']:8d} "
            f"{report['coverage']:8.1%}  {report['n_background_entries']:3d}"
        )

    print(f"\nwrote {written} atlases to {args.dest}")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for atlas, why in skipped:
            print(f"  {atlas:44s} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
