"""Where ConSEAL's Karcher median lands, on the released young adults and two synthetic cohorts.

    SBCI_HCP_DIR=hcp-ya python tests/reference/template_probe.py

Default arithmetic (unit-norm square-root densities, robust coincidence guard) against the
reference's (strict_upstream=True): the template's Fisher-Rao distance to every subject and to
the mean, in degrees, and whether it sits on one subject. Each subject's square-root density
is computed once and handed to both estimates. PORTING.md item 9 quotes its output.
"""

import glob
import os
import time

import numpy as np

import sbci
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

D = os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data")


class Precomputed:
    """A subject whose square-root density is already known."""

    def __init__(self, q):
        self.q = q

    def q_transform(self, kernel, *args, **kwargs):
        """The density computed beforehand, whatever the kernel."""
        return self.q


def angles(template, qs):
    return [float(np.degrees(np.arccos(np.clip((q * template).sum(), -1, 1)))) for q in qs]


def report(label, raw, lh, rh, names):
    qs = [q / np.sqrt((q**2).sum()) for q in raw]
    mean = sum(qs) / len(qs)
    mean /= np.sqrt((mean**2).sum())
    print(f"\n== {label}: {len(qs)} subjects", flush=True)
    print("    squared norm - 1 of each raw square-root density:", flush=True)
    print("   ", [f"{float((q**2).sum()) - 1:+.2e}" for q in raw], flush=True)
    start = int(np.argmin([((q - mean) ** 2).sum() for q in qs]))
    print(f"    the median starts from {names[start]}, the subject nearest the mean", flush=True)
    print(
        "    each subject's distance from the mean (deg):",
        np.round(angles(mean, qs), 1).tolist(),
        flush=True,
    )
    for mode, strict in (("default", False), ("reference (strict_upstream)", True)):
        t = time.time()
        template = ConSEAL(lh, rh, strict_upstream=strict).template(
            [Precomputed(q) for q in raw], None
        )
        d = angles(template, qs)
        nearest = int(np.argmin(d))
        to_mean = np.degrees(np.arccos(np.clip((template * mean).sum(), -1, 1)))
        verdict = "ON ONE SUBJECT" if min(d) < 0.01 else "between the subjects"
        print(
            f"    {mode}: {time.time() - t:.0f}s; template {to_mean:.2f} deg from the mean; "
            f"from the subjects {min(d):.3f} ({names[nearest]}) to {max(d):.2f} deg; {verdict}",
            flush=True,
        )


lh, rh = default_grids(15)
kernel, _ = HeatKernelBuilder(lh, rh, 30).compute(0.005)

paths = sorted(glob.glob(f"{D}/sub-*_sc.h5"))
names = [p.split("/")[-1].split("_")[0] for p in paths]
t = time.time()
raw = [
    EndpointConnectome.from_endpoints(sbci.load(p).endpoints, lh, rh).q_transform(kernel)
    for p in paths
]
print(
    f"square-root densities of {len(raw)} released subjects in {time.time() - t:.0f}s", flush=True
)
report("released young adults", raw, lh, rh, names)
del raw

for anatomy in (0.03, 0.05):
    cohort = sbci.example_cohort(n_subjects=10, n_streamlines=20000, anatomy=anatomy)
    raw = [
        EndpointConnectome.from_endpoints(c.endpoints, lh, rh).q_transform(kernel)
        for c in cohort.connectomes
    ]
    report(f"synthetic cohort, anatomy={anatomy}", raw, lh, rh, [f"s{i}" for i in range(10)])
