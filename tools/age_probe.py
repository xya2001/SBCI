"""Whether age, or another subject measure, is visible in a cohort's structural connectomes.

    python tools/age_probe.py COHORT_DIR [--out probe.npz]
    python tools/age_probe.py COHORT_DIR --table unrestricted.csv --column PMAT24_A_CR \
        [--groups families.csv] [--measures probe.npz]

COHORT_DIR holds ``sub-*_sc.h5`` and ``manifest.csv`` with ``subject``, ``sex``
and either ``age_years`` or ``age_bin`` (a bin enters as its midpoint, ``36+``
as 37), as ``sbci download`` writes it. With ``--table`` and ``--column`` the
measure tested is that column of a table keyed by subject, either the HCP's
open-access table (``Subject`` holds the bare id) or one with a ``subject``
column; subjects without a value are left out. ``--groups`` takes a table
with ``subject`` and ``group`` columns, the family of each subject: the HCP
young adults are twins and siblings, and the tests then treat families as
clusters (``sbci.local_test(..., groups=...)``). Family membership is
restricted HCP data, so keep that table out of the repository.
``--measures`` reads the per-subject measures an earlier ``--out`` saved
instead of streaming the files again. PORTING.md item 5, "Measured
on the HCP Young Adult cohort", records what it found against fluid
intelligence on the 946 young adults.

Streams the SC files once and records, per subject: the area-weighted
strength of every vertex, the Desikan region matrix, the interhemispheric and
long-range (over 50 mm on the white surface) fractions of the connectivity,
and the streamline count. Then correlates each with age, corrects across
vertices or edges, and asks whether age is a leading direction of variation at
the region and vertex levels (PCA), next to sex and the streamline count.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import sbci

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("cohort", type=Path, help="directory of sub-*_sc.h5 and manifest.csv")
parser.add_argument("--out", type=Path, default=None, help="save the per-subject measures here")
parser.add_argument("--table", type=Path, default=None, help="a table of subject measures")
parser.add_argument("--column", default=None, help="the measure to test, a column of --table")
parser.add_argument(
    "--groups", type=Path, default=None, help="subject, group: families as clusters"
)
parser.add_argument(
    "--measures", type=Path, default=None, help="measures saved by an earlier --out"
)
args = parser.parse_args()
rows = list(csv.DictReader(open(args.cohort / "manifest.csv")))


def years(row):
    if row.get("age_years"):
        return float(row["age_years"])
    if row["age_bin"].endswith("+"):
        return float(row["age_bin"][:-1]) + 1
    low, high = (int(v) for v in row["age_bin"].split("-"))
    return (low + high) / 2


label = "age"
if args.table is not None:
    if args.column is None:
        parser.error("--table needs --column")
    label = args.column
    with open(args.table, newline="") as handle:
        table = list(csv.DictReader(handle))
    key = "Subject" if table and "Subject" in table[0] else "subject"
    values = {
        (r[key] if r[key].startswith("sub-") else f"sub-{r[key]}"): r[args.column] for r in table
    }
    kept = [r for r in rows if values.get(r["subject"], "") not in ("", "NA", "nan")]
    if len(kept) != len(rows):
        print(f"{len(rows) - len(kept)} subjects have no {label}; left out", flush=True)
    rows = kept
    age = np.array([float(values[r["subject"]]) for r in rows])
else:
    age = np.array([years(r) for r in rows])
female = np.array([r["sex"] == "F" for r in rows], dtype=float)
n = len(rows)
family = None
if args.groups is not None:
    with open(args.groups, newline="") as handle:
        group_of = {r["subject"]: r["group"] for r in csv.DictReader(handle)}
    missing = [r["subject"] for r in rows if r["subject"] not in group_of]
    if missing:
        parser.error(f"{len(missing)} subjects have no group in {args.groups}: {missing[:3]}")
    family = np.array([group_of[r["subject"]] for r in rows])

if args.measures is not None:
    with np.load(args.measures) as saved:
        strength, edges = saved["strength"], saved["edges"]
        inter, longrange, count = saved["inter"], saved["longrange"], saved["count"]
        order = [str(x) for x in saved["subjects"]] if "subjects" in saved else None
    if order is not None and order != [r["subject"] for r in rows]:
        parser.error(f"{args.measures} holds other subjects, or in another order")
    if strength.shape[0] != n:
        parser.error(f"{args.measures} holds {strength.shape[0]} subjects, not {n}")
else:
    white = sbci.load_surface("white")
    coords = np.asarray(white.vertices, dtype=np.float64)
    half = coords.shape[0] // 2
    far = []
    for lo, hi in ((0, half), (half, 2 * half)):
        c = coords[lo:hi]
        distance = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        far.append(distance > 50.0)

    strength = np.zeros((n, 2 * half))
    iu = np.triu_indices(68, 1)
    edges = np.zeros((n, iu[0].size))
    inter = np.zeros(n)
    longrange = np.zeros(n)
    count = np.zeros(n)
    t = time.time()
    for i, r in enumerate(rows):
        cc = sbci.load(args.cohort / f"{r['subject']}_sc.h5")
        dense = cc.dense()
        area = np.asarray(cc.area, dtype=np.float64)
        strength[i] = dense @ area
        edges[i] = np.asarray(cc.to_atlas("Desikan"))[iu]
        total = area @ dense @ area
        inter[i] = 2 * (area[:half] @ dense[:half, half:] @ area[half:]) / total
        within = 0.0
        farmass = 0.0
        for k, (lo, hi) in enumerate(((0, half), (half, 2 * half))):
            block = dense[lo:hi, lo:hi]
            a = area[lo:hi]
            within += a @ block @ a
            farmass += a @ (block * far[k]) @ a
        longrange[i] = farmass / within
        count[i] = float(cc.metadata.get("streamline_count") or 0)
        if i % 50 == 0:
            print(f"  {i} subjects in {time.time() - t:.0f}s", flush=True)
    if args.out is not None:
        np.savez_compressed(
            args.out,
            strength=strength,
            edges=edges,
            inter=inter,
            longrange=longrange,
            count=count,
            age=age,
            female=female,
            subjects=np.array([r["subject"] for r in rows]),
        )


def corr(x, y):
    x = x - x.mean(0)
    y = y - y.mean()
    return (
        (x * y[:, None]).sum(0) / np.sqrt((x**2).sum(0) * (y**2).sum())
        if x.ndim > 1
        else float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum()))
    )


def tested(data, covariate):
    """Each column's association with ``covariate``: p-values and FDR-adjusted ones.

    The regression t test, which is the test of the correlation, or with
    ``--groups`` the same with families as clusters.
    """
    result = sbci.local_test(data, covariate, groups=family)
    return result.pvalue, result.adjusted


if family is not None:
    print(f"{np.unique(family).size} families: the tests treat them as clusters", flush=True)
print(
    f"{n} subjects; {label} {age.min():.1f}-{age.max():.1f}, {int(female.sum())} F; streamlines "
    f"{count.min():,.0f}-{count.max():,.0f} (r with {label} {corr(count, age):+.3f}; "
    f"{label} with sex {corr(age, female):+.3f})",
    flush=True,
)
for name, values in (
    ("interhemispheric fraction", inter),
    ("long-range (>50 mm) fraction", longrange),
):
    r = corr(values, age)
    pvalue, _ = tested(values, age)
    print(
        f"{name}: mean {values.mean():.4f}; r with {label} {r:+.3f} (p {pvalue[0]:.2g}); "
        f"with sex {corr(values, female):+.3f}; with streamline count {corr(values, count):+.3f}"
    )
for name, data in (("vertex strength", strength), ("Desikan edge", edges)):
    keep = np.isfinite(data).all(0) & (data.std(0) > 0)
    x = data[:, keep]
    r = corr(x, age)
    _, q = tested(x, age)
    rs = corr(x, female)
    _, qs = tested(x, female)
    print(
        f"{name}: {keep.sum()} tested; {label}: {int((q < 0.05).sum())} significant at FDR 0.05, "
        f"|r| max {np.abs(r).max():.3f}, 95th pct {np.percentile(np.abs(r), 95):.3f}; "
        f"sex: {int((qs < 0.05).sum())} significant, |r| max {np.abs(rs).max():.3f}"
    )
    z = (x - x.mean(0)) / x.std(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    share = s**2 / (s**2).sum()
    for k in range(min(8, s.size)):
        score = u[:, k] * s[k]
        print(
            f"   PC{k + 1} ({share[k]:.1%}): r {label} {corr(score, age):+.3f}, sex "
            f"{corr(score, female):+.3f}, streamlines {corr(score, count):+.3f}"
        )
print("done", flush=True)
