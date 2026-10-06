"""Compare two recordings of the heavy methods on the released young adults, array by array.

    python tests/reference/compare_outputs.py OLD.npz NEW.npz

NEW is made by record_outputs.py. When OLD predates it -- the recording of 2 October 2026,
before the first review -- three comparisons: the reference-arithmetic runs in NEW against
the old defaults (identical on the same CPU type for ENCORE), the unchanged methods under the
same key (identical, the within-region diagonal of the mean parcellations reported
separately, since its definition changed), and the new defaults against the old ones, the
size of the change the corrections make. When OLD was made by record_outputs.py too, every
key the two share is compared as it stands, and the keys only one holds are listed.
"""

import sys

import numpy as np

old = np.load(sys.argv[1])
new = np.load(sys.argv[2])


def report(label, x, y, off_diagonal=False):
    if x.shape != y.shape:
        print(f"{label:34s} shapes differ: {x.shape} vs {y.shape}")
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if off_diagonal and x.ndim == 2 and x.shape[0] == x.shape[1]:
        mask = ~np.eye(x.shape[0], dtype=bool)
        diag = float(np.nanmax(np.abs(np.diagonal(x) - np.diagonal(y))))
        x, y = x[mask], y[mask]
        extra = f"; diagonal max |difference| {diag:.3e}"
    else:
        extra = ""
    if np.array_equal(x, y, equal_nan=True):
        print(f"{label:34s} identical{extra}")
        return 0.0
    diff = float(np.nanmax(np.abs(x - y)))
    scale = float(np.nanmax(np.abs(x))) or 1.0
    print(
        f"{label:34s} max |difference| {diff:.3e} ({diff / scale:.2e} of the largest entry){extra}"
    )
    return diff


if "alignref_costs" in old.files:
    print("== two recordings by record_outputs.py, key by key")
    for key in sorted(set(old.files) & set(new.files)):
        report(key, old[key], new[key])
    for key in sorted(set(old.files) ^ set(new.files)):
        print(f"{key:34s} only in {'OLD' if key in old.files else 'NEW'}")
    sys.exit(0)

print("== reference arithmetic in the new code against the old defaults (must be identical)")
worst = 0.0
for key in old.files:
    if key.startswith("align_"):
        worst = max(worst, report(f"alignref vs old {key}", old[key], new["alignref_" + key[6:]]))
    elif key.startswith("conseal_"):
        worst = max(
            worst, report(f"consealref vs old {key}", old[key], new["consealref_" + key[8:]])
        )
print("WORST", worst)

print("\n== unchanged methods (must be identical; mean parcellations off the diagonal)")
worst = 0.0
for key in old.files:
    if key.startswith(("align_", "conseal_")):
        continue
    off = key in ("desikan_mean", "schaefer_fc")
    worst = max(worst, report(key, old[key], new[key], off_diagonal=off))
print("WORST", worst)

print("\n== the corrections' effect: new defaults against the old defaults")
for key in old.files:
    if key.startswith(("align_", "conseal_")):
        report(key, old[key], new[key])
for key in ("project_default", "project_reference", "seed_region", "seed_region_old_ratio"):
    if key in new.files:
        print(f"{key:34s} recorded, shape {new[key].shape}")
