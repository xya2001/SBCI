"""How often ``local_test`` rejects a true null among families, with and without ``groups=``.

    python tools/clustered_null.py [--datasets 10000]

Simulates cohorts of families of one to four members in which both the
covariate and the score are shared within a family (an intra-family
correlation of 0.69 in each), with no association between them, and tests
each cohort twice: as independent subjects, and with the families as
clusters. Prints the fraction of tests rejecting at 0.05 for 20, 50, 100 and
422 families, 422 being the number in the HCP Young Adult analysis. Every
dataset is independent, so the rates carry a binomial error of about 0.004.
Fifteen seconds on one core; PORTING.md item 5 records the output.
"""

import argparse

import numpy as np

from sbci.stats import local_test

FAMILIES = (20, 50, 100, 422)
SHARED = 1.5


def cohort(rng, n_families):
    """Family labels and a covariate shared within each family."""
    sizes = rng.integers(1, 5, n_families)
    groups = np.repeat(np.arange(n_families), sizes)
    covariate = SHARED * rng.standard_normal(n_families)[groups] + rng.standard_normal(groups.size)
    return groups, covariate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", type=int, default=10000)
    args = parser.parse_args(argv)
    print(f"{'families':>8}  {'naive':>6}  {'clustered':>9}   (rejections at 0.05 of a true null)")
    for n_families in FAMILIES:
        rng = np.random.default_rng(99 + n_families)
        naive = np.empty(args.datasets)
        clustered = np.empty(args.datasets)
        for i in range(args.datasets):
            groups, covariate = cohort(rng, n_families)
            score = SHARED * rng.standard_normal((n_families, 1))[groups]
            score += rng.standard_normal((groups.size, 1))
            naive[i] = local_test(score, covariate, method="none").pvalue[0]
            clustered[i] = local_test(score, covariate, method="none", groups=groups).pvalue[0]
        print(
            f"{n_families:>8}  {np.mean(naive < 0.05):>6.3f}  {np.mean(clustered < 0.05):>9.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
