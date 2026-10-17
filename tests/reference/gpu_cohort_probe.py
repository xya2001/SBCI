"""The eleven aligned on the CPU and on the GPU as the cohort figure aligns them, compared.

    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_cohort_probe.py run cpu OUT
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_cohort_probe.py run twin OUT
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_cohort_probe.py run gpu OUT
    python tests/reference/gpu_cohort_probe.py compare CPU TWIN GPU

``run`` aligns the eleven released young adults (``SBCI_HCP_DIR``) as
``scripts/hcp_figures.py`` does for the cohort figure: ENCORE on their densities re-smoothed
through the package's kernel, ten iterations and four backtracks, and ConSEAL on their
endpoints, thirty iterations at its defaults. It does so on the CPU, on the GPU, or on the
CPU with every input moved by one unit in the last place -- the twin, whose distance from
the CPU's result is how reproducible the alignment is to begin with: ENCORE's inputs are
the re-smoothed densities, ConSEAL's the endpoints' barycentric weights. Into the folder OUT
it writes the templates, every subject's cost trace, warp and aligned density, ConSEAL's
aligned endpoints and the connectomes re-smoothed from them, the correlations the figure
draws, and the times (PORTING.md item 26).

``compare`` sets the twin's results and the GPU's beside the CPU's, subject by subject:
distances relative to the CPU's in Frobenius norms, warps and endpoints by the largest and
the mean angle in degrees between where they put a point.
"""

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci.grid import to_condensed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_figures import resmoothed  # noqa: E402

EPS = np.finfo(np.float64).eps


def nudged(array, seed):
    """``array`` with every entry moved by one unit in its last place, up or down."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=np.shape(array))
    return np.asarray(array) * (1 + EPS * signs)


def digest(arrays):
    hashed = hashlib.sha256()
    for array in arrays:
        hashed.update(np.ascontiguousarray(array).tobytes())
    return hashed.hexdigest()[:16]


def similarity(matrices):
    """The figure's: the correlation between every two subjects' connectomes, in float32."""
    data = np.stack([np.asarray(m, dtype=np.float32).ravel() for m in matrices])
    return np.corrcoef(data)[np.triu_indices(len(matrices), 1)]


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run(kind, out):
    from sbci.conseal import EndpointConnectome, default_grids

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = "gpu" if kind == "gpu" else "cpu"
    paths = sorted(Path(os.environ["SBCI_HCP_DIR"]).glob("sub-*_sc.h5"))
    subjects = [sbci.load(p) for p in paths]
    meta = {"kind": kind, "host": platform.node(), "cpu": cpu_name()}
    meta["subjects"] = [p.name.split("_")[0] for p in paths]
    if device == "gpu":
        from sbci import _device

        cupy = _device.cupy()
        meta["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()

    begun = time.time()
    smoothed = [resmoothed(s, s.endpoints) for s in subjects]
    meta["resmooth_s"] = time.time() - begun
    meta["encore_inputs"] = digest([s.data for s in smoothed])
    given = smoothed
    if kind == "twin":
        given = [nudged(s.dense(), seed) for seed, s in enumerate(smoothed)]
    begun = time.time()
    encore = sbci.align(given, max_iterations=10, backtracks=4, device=device)
    meta["encore_s"] = time.time() - begun
    del given
    np.save(out / "encore_template.npy", to_condensed(encore.template))
    arrays = {}
    for i, (aligned, warp) in enumerate(zip(encore.aligned, encore.warps, strict=True)):
        np.save(out / f"encore_aligned_{i}.npy", to_condensed(np.asarray(aligned)))
        arrays[f"trace_{i}"] = np.asarray(encore.traces[i])
        for side in ("lh", "rh"):
            arrays[f"{side}_vertices_{i}"] = getattr(warp, f"{side}_vertices")
            arrays[f"{side}_jacobian_{i}"] = getattr(warp, f"{side}_jacobian")
    np.savez(out / "encore.npz", **arrays)
    meta["correlation_before"] = float(similarity([s.data for s in smoothed]).mean())
    aligned = [to_condensed(np.asarray(a)) for a in encore.aligned]
    meta["correlation_encore"] = float(similarity(aligned).mean())
    del encore, aligned, smoothed
    print(f"{kind}: ENCORE in {meta['encore_s']:.0f} s", flush=True)

    lh, rh = default_grids()
    carriers = [EndpointConnectome.from_endpoints(s.endpoints, lh, rh) for s in subjects]
    meta["conseal_inputs"] = digest(
        [c.weights_in for c in carriers] + [c.weights_out for c in carriers]
    )
    if kind == "twin":
        for i, carrier in enumerate(carriers):
            carrier.weights_in = nudged(carrier.weights_in, 100 + 2 * i)
            carrier.weights_out = nudged(carrier.weights_out, 101 + 2 * i)
            carrier.commit()
    begun = time.time()
    registration = sbci.endpoints_align(carriers, max_iterations=30, device=device)
    meta["conseal_s"] = time.time() - begun
    np.save(out / "conseal_template.npy", to_condensed(registration.template))
    arrays, connectomes = {}, []
    for i, warped in enumerate(registration.connectomes):
        warp = registration.warps[i]
        arrays[f"costs_{i}"] = np.asarray(registration.costs[i])
        arrays[f"lh_velocity_{i}"] = warp.lh_velocity
        arrays[f"rh_velocity_{i}"] = warp.rh_velocity
        arrays[f"points_{i}"] = np.concatenate(warped.positions())
        arrays[f"faces_{i}"] = np.concatenate([warped.face_in, warped.face_out])
        smoothed = resmoothed(subjects[i], registration.aligned_endpoints(i)).data
        np.save(out / f"conseal_smoothed_{i}.npy", smoothed)
        connectomes.append(smoothed)
    np.savez(out / "conseal.npz", **arrays)
    meta["correlation_conseal"] = float(similarity(connectomes).mean())
    start = np.array([trace[0] for trace in registration.costs])
    angles = np.degrees(np.arccos(np.clip(1 - start / 2, -1.0, 1.0)))
    meta["conseal_median_degrees"] = [float(angles.min()), float(angles.max())]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


def angles(first, second):
    """Degrees between corresponding rows of two arrays of unit vectors, by the chord between them.

    The arc cosine of their inner product reads 1e-6 degrees between a point and itself.
    """
    chord = np.linalg.norm(first - second, axis=1)
    return np.degrees(2 * np.arcsin(np.clip(chord / 2, 0.0, 1.0)))


def gap(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    if np.array_equal(a, b):
        return 0.0
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(cpu, twin, gpu):
    cpu, twin, gpu = Path(cpu), Path(twin), Path(gpu)
    metas = {
        name: json.loads((d / "meta.json").read_text())
        for name, d in (("CPU", cpu), ("twin", twin), ("GPU", gpu))
    }
    for name, meta in metas.items():
        where = meta.get("gpu", meta["cpu"])
        print(
            f"{name}: {meta['host']}, {where}; re-smoothing {meta['resmooth_s']:.0f} s, ENCORE "
            f"{meta['encore_s']:.0f} s, ConSEAL {meta['conseal_s']:.0f} s"
        )
    for key in ("encore_inputs", "conseal_inputs"):
        same = len({meta[key] for meta in metas.values()}) == 1
        print(
            f"  the {key.replace('_', ' ')} before any nudge: "
            f"{'the same' if same else 'DIFFERENT'} in all three"
        )
    print("  mean pairwise correlation    before     after ENCORE   after ConSEAL")
    for name, meta in metas.items():
        print(
            f"    {name:5}                      {meta['correlation_before']:.6f}   "
            f"{meta['correlation_encore']:.6f}       {meta['correlation_conseal']:.6f}"
        )
    for name, meta in metas.items():
        low, high = meta["conseal_median_degrees"]
        print(f"  {name}: ConSEAL's median lies {low:.4f} to {high:.4f} degrees from the subjects")

    names = metas["CPU"]["subjects"]
    loaded = {
        name: np.load(d / "encore.npz") for name, d in (("CPU", cpu), ("twin", twin), ("GPU", gpu))
    }
    base = np.load(cpu / "encore_template.npy")
    print("\nENCORE, against the CPU          the twin                     the GPU")
    print(
        "  the template: "
        f"{gap(np.load(twin / 'encore_template.npy'), base):.1e}{'':20}"
        f"{gap(np.load(gpu / 'encore_template.npy'), base):.1e}"
    )
    print(
        "  subject      steps   trace    aligned  warp max/mean (deg)  |  steps   trace    "
        "aligned  warp max/mean (deg)"
    )
    worst = {"twin": [], "GPU": []}
    for i, name in enumerate(names):
        aligned = np.load(cpu / f"encore_aligned_{i}.npy")
        cells = []
        for other, folder in (("twin", twin), ("GPU", gpu)):
            trace, reference = loaded[other][f"trace_{i}"], loaded["CPU"][f"trace_{i}"]
            steps = f"{len(trace) - 1}/{len(reference) - 1}"
            moved = np.concatenate(
                [
                    angles(
                        loaded[other][f"{side}_vertices_{i}"], loaded["CPU"][f"{side}_vertices_{i}"]
                    )
                    for side in ("lh", "rh")
                ]
            )
            aligned_gap = gap(np.load(folder / f"encore_aligned_{i}.npy"), aligned)
            worst[other].append((gap(trace, reference), aligned_gap, moved.max()))
            cells.append(
                f"{steps:>7} {gap(trace, reference):8.1e} {aligned_gap:8.1e}  "
                f"{moved.max():8.1e}/{moved.mean():7.1e}"
            )
        print(f"  {name:12}{cells[0]}  | {cells[1]}")
    for other in ("twin", "GPU"):
        trace, aligned, moved = (max(column) for column in zip(*worst[other], strict=True))
        print(
            f"  the {other}, at most: trace {trace:.1e}, aligned density {aligned:.1e}, a vertex "
            f"{moved:.1e} degrees"
        )

    loaded = {
        name: np.load(d / "conseal.npz") for name, d in (("CPU", cpu), ("twin", twin), ("GPU", gpu))
    }
    base = np.load(cpu / "conseal_template.npy")
    print("\nConSEAL, against the CPU        the twin                                the GPU")
    print(
        "  the template: "
        f"{gap(np.load(twin / 'conseal_template.npy'), base):.1e}{'':31}"
        f"{gap(np.load(gpu / 'conseal_template.npy'), base):.1e}"
    )
    print(
        "  subject      iter   costs    velocity endpoints max/mean (deg)  same face  "
        "connectome  |  iter   costs    velocity endpoints max/mean (deg)  same face  connectome"
    )
    worst = {"twin": [], "GPU": []}
    for i, name in enumerate(names):
        smoothed = np.load(cpu / f"conseal_smoothed_{i}.npy")
        cells = []
        for other, folder in (("twin", twin), ("GPU", gpu)):
            data, reference = loaded[other], loaded["CPU"]
            costs = gap(data[f"costs_{i}"], reference[f"costs_{i}"])
            iterations = f"{len(data[f'costs_{i}']) - 1}/{len(reference[f'costs_{i}']) - 1}"
            velocity = gap(
                np.concatenate([data[f"lh_velocity_{i}"], data[f"rh_velocity_{i}"]]),
                np.concatenate([reference[f"lh_velocity_{i}"], reference[f"rh_velocity_{i}"]]),
            )
            moved = angles(data[f"points_{i}"], reference[f"points_{i}"])
            faces = float(np.mean(data[f"faces_{i}"] == reference[f"faces_{i}"]))
            connectome = gap(np.load(folder / f"conseal_smoothed_{i}.npy"), smoothed)
            worst[other].append((costs, velocity, moved.max(), connectome, faces))
            cells.append(
                f"{iterations:>6} {costs:8.1e} {velocity:8.1e}  "
                f"{moved.max():8.1e}/{moved.mean():7.1e}  {100 * faces:8.4f}%  {connectome:9.1e}"
            )
        print(f"  {name:12}{cells[0]}  | {cells[1]}")
    for other in ("twin", "GPU"):
        columns = list(zip(*worst[other], strict=True))
        print(
            f"  the {other}, at most: costs {max(columns[0]):.1e}, velocity {max(columns[1]):.1e}, "
            f"an endpoint {max(columns[2]):.1e} degrees, the connectome {max(columns[3]):.1e}; "
            f"at least {100 * min(columns[4]):.4f}% of endpoints in the CPU's face"
        )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "run" and sys.argv[2] in ("cpu", "twin", "gpu"):
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    else:
        sys.exit(__doc__)
