"""Whether age, or another subject measure, is visible in a cohort's structural connectomes.

    python tools/age_probe.py COHORT_DIR [--out probe.npz]
    python tools/age_probe.py COHORT_DIR --table unrestricted.csv --column PMAT24_A_CR

COHORT_DIR holds ``sub-*_sc.h5`` and ``manifest.csv`` with ``subject``, ``sex``
and either ``age_years`` or ``age_bin`` (a bin enters as its midpoint, ``36+``
as 37), as ``sbci download`` writes it. With ``--table`` and ``--column`` the
measure tested is that column of a table keyed by subject, either the HCP's
open-access table (``Subject`` holds the bare id) or one with a ``subject``
column; subjects without a value are left out. PORTING.md item 5, "Measured on
the full HCP-Aging cohort", records what this found against age on the 528
HCP-Aging subjects, and "Measured on the HCP Young Adult cohort" what it found
against fluid intelligence on the 946 young adults.

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
    )


def corr(x, y):
    x = x - x.mean(0)
    y = y - y.mean()
    return (
        (x * y[:, None]).sum(0) / np.sqrt((x**2).sum(0) * (y**2).sum())
        if x.ndim > 1
        else float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum()))
    )


def p_of_r(r):
    from scipy import stats

    tstat = r * np.sqrt((n - 2) / np.maximum(1 - r**2, 1e-12))
    return 2 * stats.t.sf(np.abs(tstat), n - 2)


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * p.size / (np.arange(p.size) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


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
    print(
        f"{name}: mean {values.mean():.4f}; r with {label} {r:+.3f} (p {p_of_r(r):.2g}); "
        f"with sex {corr(values, female):+.3f}; with streamline count {corr(values, count):+.3f}"
    )
for name, data in (("vertex strength", strength), ("Desikan edge", edges)):
    keep = np.isfinite(data).all(0) & (data.std(0) > 0)
    x = data[:, keep]
    r = corr(x, age)
    q = bh(p_of_r(r))
    rs = corr(x, female)
    qs = bh(p_of_r(rs))
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
