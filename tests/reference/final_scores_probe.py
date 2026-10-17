"""What recording the scores of the returned vectors changes (PORTING.md item 17).

    PYTHONPATH=OLD_SRC python tests/reference/final_scores_probe.py OLD.json   # the code before
    python tests/reference/final_scores_probe.py NEW.json
    python tests/reference/final_scores_probe.py OLD.json NEW.json            # the comparison

Fits the synthetic cohorts PORTING.md item 5 quotes -- ten subjects at rank 4: the example's
default anatomy, and three degrees of it (``anatomy=0.05``) with the reference's start and with
``candidates=6`` -- and records what each fit reports: the explained fraction, the scales, the
components that track the planted age, their adjusted p-values and correlations, and the time.
Run both on one node type, since the timings are compared.
"""

import json
import sys
import time

import numpy as np

import sbci


def fit(cohort, **options):
    start = time.time()
    reduction = sbci.reduce(cohort.connectomes, rank=4, **options)
    seconds = time.time() - start
    test = sbci.local_test(reduction.scores, cohort.age)
    return dict(
        seconds=seconds,
        explained=np.round(reduction.explained, 6).tolist(),
        scales=[float(f"{s:.4g}") for s in reduction.scales],
        significant=test.significant().tolist(),
        adjusted=[float(f"{p:.3g}") for p in test.adjusted],
        r=[
            round(float(np.corrcoef(reduction.scores[:, k], cohort.age)[0, 1]), 4) for k in range(4)
        ],
    )


if len(sys.argv) == 2:
    results = {"sbci": sbci.__file__}
    results["default anatomy"] = fit(sbci.example_cohort(n_subjects=10, seed=0))
    jittered = sbci.example_cohort(n_subjects=10, seed=0, anatomy=0.05)
    results["anatomy 0.05, reference start"] = fit(jittered)
    results["anatomy 0.05, candidates=6"] = fit(jittered, candidates=6)
    with open(sys.argv[1], "w") as handle:
        json.dump(results, handle, indent=1)
    print(json.dumps(results, indent=1))
else:
    with open(sys.argv[1]) as a, open(sys.argv[2]) as b:
        old, new = json.load(a), json.load(b)
    for key in old:
        if key == "sbci":
            continue
        print(key)
        for field in ("explained", "scales", "significant", "adjusted", "r", "seconds"):
            print(f"  {field:12s} before {old[key][field]}\n  {'':12s} after  {new[key][field]}")
