"""How far ENCORE's template gets toward the median, and what its number of steps changes.

    SBCI_HCP_DIR=hcp-ya python tests/reference/encore_template_probe.py convergence
    python tests/reference/encore_template_probe.py convergence FOLDER COUNT
    SBCI_HCP_DIR=hcp-ya python tests/reference/encore_template_probe.py starts
    SBCI_HCP_DIR=hcp-ya python tests/reference/encore_template_probe.py cohort

``Encore.template`` takes Weiszfeld steps toward the median of the subjects' square-root
densities, from the subject nearest their mean, each a fifth of the mean direction to them,
and stops when that direction is shorter than 0.005 or after ``iterations`` steps. Here the
steps are taken one at a time as it takes them, on the bundled grids at the default basis
order, the subjects being the eleven released young adults (``SBCI_HCP_DIR``) or, given a
folder and a count, that many others drawn from it with a fixed seed. PORTING.md items 24 and
25 quote it.

``convergence``
    The steps past the stopping rule to the 80th: each one's size, where the rule stops,
    and, for the template after 10 and 20 steps, where the rule stops, and after 80, the sum
    of its distances to the subjects and its distance from the 80th. On the eleven the steps
    are first checked against ``Encore.template``, bit for bit, at 10 steps and at the rule.
``starts``
    The same from three other subjects of the eleven, to 60 steps: how far apart the
    templates are after 10 steps, where the rule stops and after 60, and how much nearer each
    lies to the subject it started from than to the others.
``cohort``
    The eleven aligned as the cohort figure aligns them (re-smoothed, ten iterations, four
    backtracks), once with the template after ten steps and once after up to a hundred: the
    mean correlation between two subjects' aligned connectomes, each subject's cost, the time,
    and how far apart the two templates are.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci.alignment import DEFAULT_ORDER, Encore, _hemisphere_grids

STOP = 0.005  # the size of the mean direction at which Encore.template stops


def released():
    folder = Path(os.environ.get("SBCI_HCP_DIR", ""))
    paths = sorted(folder.glob("sub-*_sc.h5"))
    if not paths:
        sys.exit(f"no sub-*_sc.h5 in {folder!s}; set SBCI_HCP_DIR to the released subjects")
    return paths


def drawn(folder, count):
    """``count`` subjects of ``folder``, none of them released, drawn with a fixed seed."""
    exclude = {path.name for path in released()}
    paths = [p for p in sorted(Path(folder).glob("sub-*_sc.h5")) if p.name not in exclude]
    picked = np.random.default_rng(2026).choice(len(paths), size=count, replace=False)
    return [paths[i] for i in sorted(picked)]


def roots_of(encore, paths, keep_densities=False):
    """Each subject's square-root density, as ``Encore.template`` forms it, and the densities."""
    densities, roots = [], None
    for k, path in enumerate(paths):
        density = np.asarray(sbci.load(path).dense(), dtype=np.float64)
        if roots is None:
            roots = np.empty((len(paths), *density.shape))
        roots[k] = encore.root(density)
        if keep_densities:
            densities.append(density)
    return roots, densities


def geodesic(encore, first, second):
    inner = (first * second * encore.area_product).sum()
    return float(np.arccos(np.clip(inner, -1.0, 1.0)))


def nearest_the_mean(encore, roots):
    mean = roots.mean(axis=0)
    return int(np.argmin([((root - mean) ** 2 * encore.area_product).sum() for root in roots]))


def weiszfeld(encore, roots, start, count):
    """``Encore.template``'s steps, without its stopping rule: yields each size and template."""
    current = roots[start].copy()
    for _ in range(count):
        directions = np.zeros_like(current)
        distance = np.zeros(roots.shape[0])
        for i, root in enumerate(roots):
            inner = (root * current * encore.area_product).sum()
            if 1 - abs(inner) < 1e-14:
                inner = np.sign(inner)
            distance[i] = np.arccos(np.clip(inner, -1.0, 1.0))
            if distance[i] > 0:
                directions += (root - np.cos(distance[i]) * current) / np.sin(distance[i])
        moving = distance > 0
        if not moving.any():
            return
        mean_direction = directions / (1.0 / distance[moving]).sum()
        size = np.sqrt((mean_direction**2 * encore.area_product).sum())
        if size == 0:
            return
        current = np.cos(0.2 * size) * current + np.sin(0.2 * size) * (mean_direction / size)
        current = current / np.sqrt((current**2 * encore.area_product).sum())
        yield float(size), current


def walk(encore, roots, start, count, keep=(10, 20)):
    """Steps to ``count``: the sizes, where the rule stops, and the templates kept on the way."""
    sizes, kept, stop = [], {}, None
    begun = time.time()
    for step, (size, current) in enumerate(weiszfeld(encore, roots, start, count), start=1):
        sizes.append(size)
        if step in keep or step == count:
            kept[step] = current
        if stop is None and size < STOP:
            stop = step
            kept["rule"] = current
    return sizes, kept, stop, (time.time() - begun) / max(len(sizes), 1)


def total(encore, roots, template):
    return sum(geodesic(encore, template, root) for root in roots)


def convergence(paths, check):
    encore = Encore(*_hemisphere_grids(DEFAULT_ORDER))
    roots, densities = roots_of(encore, paths, keep_densities=check)
    start = nearest_the_mean(encore, roots)
    name = paths[start].name.split("_")[0]
    print(
        f"{len(paths)} subjects, basis order {DEFAULT_ORDER}; the steps start from {name}, "
        "the subject nearest the mean",
        flush=True,
    )
    sizes, kept, stop, per_step = walk(encore, roots, start, 80)
    if stop is None:
        print(f"  the rule never stops within 80 steps; the last step's size {sizes[-1]:.4f}")
    if check and stop is not None and stop > 10:
        for label, iterations, mine in (
            ("10 steps", 10, kept[10]),
            ("the rule", 1000, kept["rule"]),
        ):
            package = encore.template(densities, iterations=iterations)
            print(
                f"  against Encore.template, {label}: largest difference "
                f"{np.abs(package - mine).max():.1e}",
                flush=True,
            )
    del densities
    shown = ", ".join(
        f"{k}: {sizes[k - 1]:.4f}" for k in (1, 2, 5, 10, 15, 20, 30, 40, 60, 80) if k <= len(sizes)
    )
    print(f"  the size of each step's mean direction -- {shown}")
    print(
        f"  the rule (a direction below {STOP}) stops after {stop} steps; {per_step:.1f} s a step"
    )
    rows = [("after 10 steps", 10), ("after 20", 20)]
    if stop is not None:
        rows.append((f"where the rule stops, {stop}", "rule"))
    rows.append(("after 80", 80))
    rule = kept.get("rule")
    print(
        f"  {'template':>24} {'sum of distances':>17} {'from the rule (rad)':>20} "
        f"{'from the 80th (rad)':>20}"
    )
    for label, key in rows:
        template = kept[key]
        from_rule = geodesic(encore, template, rule) if rule is not None else float("nan")
        print(
            f"  {label:>24} {total(encore, roots, template):17.6f} {from_rule:20.2e} "
            f"{geodesic(encore, template, kept[80]):20.2e}"
        )
    distances = np.array([geodesic(encore, kept[80], root) for root in roots])
    print(
        f"  the subjects lie {distances.min():.3f} to {distances.max():.3f} rad from the 80th "
        f"({distances.mean():.3f} on average)"
    )


def starts(paths):
    encore = Encore(*_hemisphere_grids(DEFAULT_ORDER))
    roots, _ = roots_of(encore, paths)
    mean = roots.mean(axis=0)
    order = np.argsort([((root - mean) ** 2 * encore.area_product).sum() for root in roots])
    chosen = [int(order[k]) for k in (0, 3, 7, len(order) - 1)]  # nearest the mean, ..., farthest
    templates = {}
    for start in chosen:
        sizes, kept, stop, _ = walk(encore, roots, start, 60, keep=(10,))
        templates[start] = kept
        others = [
            geodesic(encore, kept["rule"], root) for k, root in enumerate(roots) if k != start
        ]
        own = geodesic(encore, kept["rule"], roots[start])
        ten_own = geodesic(encore, kept[10], roots[start])
        ten_others = [
            geodesic(encore, kept[10], root) for k, root in enumerate(roots) if k != start
        ]
        print(
            f"from {paths[start].name.split('_')[0]} (rank {list(order).index(start) + 1} of "
            f"{len(order)} by distance from the mean): the rule stops after {stop} steps; the "
            f"start subject lies {ten_own:.3f} rad from the template after 10 steps against "
            f"{np.median(ten_others):.3f} for the others (median), and {own:.3f} against "
            f"{np.median(others):.3f} where the rule stops",
            flush=True,
        )
    for label, key in (("after 10 steps", 10), ("where the rule stops", "rule"), ("after 60", 60)):
        spread = max(
            geodesic(encore, templates[a][key], templates[b][key])
            for a in chosen
            for b in chosen
            if a < b
        )
        print(f"  the four templates {label}: at most {spread:.2e} rad apart")


def cohort(paths):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_figures import resmoothed

    from sbci.grid import to_condensed

    subjects = [sbci.load(p) for p in paths]
    smoothed = [resmoothed(s, s.endpoints) for s in subjects]

    def similarity(matrices):
        data = np.stack([np.asarray(m, dtype=np.float32).ravel() for m in matrices])
        return np.corrcoef(data)[np.triu_indices(len(matrices), 1)]

    before = similarity([s.data for s in smoothed])
    print(f"{len(subjects)} subjects: mean pairwise correlation {before.mean():.4f}", flush=True)
    results = {}
    for steps in (10, 100):
        begun = time.time()
        result = sbci.align(smoothed, max_iterations=10, backtracks=4, template_iterations=steps)
        after = similarity([to_condensed(np.asarray(d)) for d in result.aligned])
        falls = np.array([trace[-1] / trace[0] for trace in result.traces])
        starts_ = np.array([trace[0] for trace in result.traces])
        ends = np.array([trace[-1] for trace in result.traces])
        print(
            f"  template after up to {steps:3d} steps: mean pairwise correlation "
            f"{after.mean():.4f}; each subject's cost {starts_.min():.4f}-{starts_.max():.4f} at "
            f"the start, {ends.min():.4f}-{ends.max():.4f} at the end, in all {starts_.sum():.4f} "
            f"-> {ends.sum():.4f}; to {falls.min():.3f}-{falls.max():.3f} of its start; "
            f"{time.time() - begun:.0f}s",
            flush=True,
        )
        results[steps] = result
    encore = Encore(*_hemisphere_grids(DEFAULT_ORDER))
    apart = geodesic(encore, results[10].template, results[100].template)
    print(f"  the two templates are {apart:.3f} rad apart")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "convergence" and len(sys.argv) == 4:
        convergence(drawn(sys.argv[2], int(sys.argv[3])), check=False)
    elif mode == "convergence":
        convergence(released(), check=True)
    elif mode == "starts":
        starts(released())
    elif mode == "cohort":
        cohort(released())
    else:
        sys.exit(__doc__)
