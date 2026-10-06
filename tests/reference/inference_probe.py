"""Named designs, intervals and contrasts on the 943 young adults of docs/RESULTS.md.

    module load python/3.12.4
    python tests/reference/inference_probe.py FULL_DIR FAMILIES_CSV [OUT_DIR]

``FULL_DIR`` holds ``fpca_rank20_c1.npz`` (the rank-20 reduction of the 943
with a fluid-intelligence score, with their covariates) and
``probe_pmat24.npz`` (each subject's per-vertex strength, from
``tools/age_probe.py``); ``FAMILIES_CSV`` maps subjects to families, which is
restricted HCP data and stays out of the repository; only counts and
summaries are printed. On Longleaf: ``/work/users/x/y/xya/hcp-ya/full`` and
``/work/users/x/y/xya/hcp-ya/restricted/families.csv``.

1. The component test of RESULTS.md, rebuilt with a named design: sex, with
   fluid intelligence, the age band and the streamline count as nuisance and
   families as clusters, must give the published 11 of 20 components and the
   same statistics as the index-based call, bit for bit; then the intervals
   and effect sizes the result now carries.
2. A contrast: the age band as a factor, its two largest bands against each other.
3. The per-vertex test of the same question on each subject's vertex
   strength: the published 1,020 vertices for sex and 113 for fluid
   intelligence, and the export of the estimate as a map and of the test as
   a table.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import sbci
from sbci.stats import design, local_test

BANDS = {23.5: "22-25", 28.0: "26-30", 33.0: "31-35", 37.0: "36+"}


def main() -> int:
    full, families_csv = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("inference-probe")
    out.mkdir(parents=True, exist_ok=True)
    fit = np.load(full / "fpca_rank20_c1.npz", allow_pickle=False)
    probe = np.load(full / "probe_pmat24.npz", allow_pickle=False)
    subjects = [str(s) for s in fit["subjects"]]
    with open(families_csv, newline="") as handle:
        family = {row["subject"]: row["group"] for row in csv.DictReader(handle)}
    groups = [family[s] for s in subjects]
    same_order = np.array_equal(probe["count"], fit["count"]) and np.array_equal(
        probe["female"], fit["female"]
    )
    print(
        f"{len(subjects)} subjects, {len(set(groups))} families; "
        f"the two files in one order: {same_order}"
    )

    covariates = {
        "fluid_intelligence": fit["trait"],
        "female": fit["female"],
        "age_band": fit["band"],
        "streamlines": fit["count"],
    }
    named = design(covariates)
    print("design:", ", ".join(named.names))

    print("1. Sex, with the rest as nuisance and families as clusters, on the 20 components")
    sex = local_test(fit["scores"], named, terms=["female"], groups=groups)
    old = local_test(
        fit["scores"],
        np.column_stack([fit["trait"], fit["female"], fit["band"], fit["count"]]),
        terms=[2],
        groups=groups,
    )
    identical = np.array_equal(sex.statistic, old.statistic) and np.array_equal(
        sex.adjusted, old.adjusted
    )
    hits = sex.significant(0.05)
    print(
        f"   {hits.size} of 20 components at FDR 0.05; "
        f"identical to the index-based call: {identical}"
    )
    best = int(np.nanargmin(sex.adjusted))
    low, high = sex.interval(0.95)
    print(
        f"   strongest, component {best}: adjusted p {sex.adjusted[best]:.1e}, estimate "
        f"{sex.estimate[best, 0]:+.3g} (95% CI {low[best, 0]:+.3g} to {high[best, 0]:+.3g}, "
        f"{sex.interval_dof} degrees of freedom), partial R2 {sex.partial_r2[best]:.3f}"
    )
    shown = sex.partial_r2[hits]
    print(f"   partial R2 over the {hits.size}: {shown.min():.3f} to {shown.max():.3f}")
    sex.to_table(out / "sex_components.csv")

    print("2. The age band as a factor: 31-35 against 26-30")
    banded = dict(covariates, age_band=[BANDS[float(b)] for b in fit["band"]])
    sizes = {band: banded["age_band"].count(band) for band in BANDS.values()}
    print("   subjects per band:", ", ".join(f"{band} {count}" for band, count in sizes.items()))
    factor = design(banded, reference={"age_band": "22-25"})
    print("   design:", ", ".join(factor.names))
    whole = local_test(fit["scores"], factor, terms=["age_band"], groups=groups)
    print(
        f"   the whole factor, {whole.numerator_dof} degrees of freedom: "
        f"{whole.significant().size} of 20 components"
    )
    oldest = local_test(
        fit["scores"], factor, contrast={"age_band[31-35]": 1, "age_band[26-30]": -1}, groups=groups
    )
    k = int(np.nanargmin(oldest.adjusted))
    low, high = oldest.interval(0.95)
    print(
        f"   31-35 against 26-30: {oldest.significant().size} of 20; the closest, component {k}, "
        f"estimate {oldest.estimate[k, 0]:+.3g} (95% CI {low[k, 0]:+.3g} to {high[k, 0]:+.3g}), "
        f"adjusted p {oldest.adjusted[k]:.2g}"
    )

    print("3. Vertex by vertex, on each subject's vertex strength")
    strength = probe["strength"]
    for term, published in (("female", 1020), ("fluid_intelligence", 113)):
        alone = local_test(strength, design({term: covariates[term]}), terms=[term], groups=groups)
        tested = int(np.isfinite(alone.pvalue).sum())
        print(
            f"   {term} alone, as tools/age_probe.py tests it: {alone.significant().size} of "
            f"{tested} cortical vertices at FDR 0.05 (published: {published})"
        )
    adjusted = local_test(strength, named, terms=["female"], groups=groups)
    print(
        f"   female with the other three as nuisance: {adjusted.significant().size} vertices; "
        f"partial R2 up to {np.nanmax(adjusted.partial_r2):.3f}"
    )
    sbci.save_map(
        {"estimate": adjusted.estimate[:, 0], "adjusted": adjusted.adjusted},
        out / "sex_strength.dscalar.nii",
    )
    adjusted.to_table(out / "sex_strength.csv", index="vertex")
    print(f"   wrote the map and the table to {out}")
    print("INFERENCE PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
