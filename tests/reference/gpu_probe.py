"""The GPU against the CPU on the released subjects: how close, how fast, how much memory.

    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py encore
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py conseal
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py align [COUNT]
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_probe.py repeat
    python tests/reference/gpu_probe.py spread

Run on a GPU node with CuPy (PORTING.md item 26), but for ``spread``, which needs
neither. Each part computes the same thing with ``device="cpu"`` and
``device="gpu"`` on the bundled ico4 grids, the subjects being the
eleven released young adults (``SBCI_HCP_DIR``), and prints the time each took, how far
apart the results are, and the most GPU memory CuPy held. Each GPU computation runs
twice: the first time includes compiling CuPy's kernels (cached on disk for later runs),
and the second shows whether the GPU repeats itself. A registration is also run on the
CPU with one input moved by one unit in the last place, its twin: how far the twin lies
from the CPU's own result is how reproducible the registration is to begin with
(``tests/test_gpu.py``). Distances are relative, in Frobenius norms, or for endpoints the
largest angle in radians, over both ends of every streamline, as is the share of
endpoints in the same face.

``encore``
    Three subjects' template, then one subject registered onto it with the defaults, as
    ``align`` does: ``Encore.template`` and ``Encore.register``.
``conseal``
    A subject's endpoint density with its derivatives, its endpoints relocated through a
    small warp, and one subject registered onto another for ten iterations:
    ``EndpointConnectome.q_transform``, ``warp`` and ``ConSEAL.register``.
``align``
    ``sbci.align`` on the first COUNT subjects (all eleven by default) as the cohort
    figure aligns them: re-smoothed, ten iterations, four backtracks.
``repeat``
    The GPU run twice: the product of a released subject's kernel, transposed, and a
    dense matrix by ``_device.rows_times_dense`` and by cuSPARSE, with their times; a
    subject's density with its derivatives; three steps of ENCORE and three iterations
    of ConSEAL; and whether each gives the same bits the second time.
``spread``
    The CPU alone: ten twins of each registration ``tests/test_gpu.py`` compares the GPU
    on, with how far they lie from the CPU's own result, the farthest against the nearest
    (the test allows the GPU twenty times the farther of two); ConSEAL's endpoints moved
    into another face; and ENCORE's aligned density after its default fifty steps, by its
    largest difference against its largest entry.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci import _device
from sbci.alignment import DEFAULT_ORDER, Encore, _hemisphere_grids
from sbci.conseal import (
    DEFAULT_KERNEL_DEGREE,
    DEFAULT_SIGMA,
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    _sent,
    default_grids,
)

EPS = np.finfo(np.float64).eps


def released():
    folder = Path(os.environ.get("SBCI_HCP_DIR", ""))
    paths = sorted(folder.glob("sub-*_sc.h5"))
    if not paths:
        sys.exit(f"no sub-*_sc.h5 in {folder!s}; set SBCI_HCP_DIR to the released subjects")
    return paths


def timed(device, compute):
    """``compute()``, and the seconds it took, the GPU's queued work included."""
    _device.synchronize(device)
    begun = time.perf_counter()
    result = compute()
    _device.synchronize(device)
    return result, time.perf_counter() - begun


def nudged(array, seed=0):
    """``array`` with every entry moved by one unit in its last place, up or down."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=np.shape(array))
    return np.asarray(array) * (1 + EPS * signs)


def gap(a, b):
    """``a``'s distance from ``b``, relative to ``b``, in Frobenius norms; 0 when identical."""
    a, b = _device.to_host(a), _device.to_host(b)
    if np.array_equal(a, b):
        return 0.0
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def endpoints_apart(first, second):
    """The largest angle between where two copies of a connectome put an endpoint (rad).

    Both ends of every streamline; the angle from the chord, which stays exact when it is
    small.
    """
    return max(
        float((2 * np.arcsin(np.minimum(np.linalg.norm(a - b, axis=1) / 2, 1.0))).max())
        for a, b in zip(first.positions(), second.positions(), strict=True)
    )


def faces_of(connectome):
    """The face of every endpoint, both ends."""
    return np.concatenate([connectome.face_in, connectome.face_out])


def memory():
    pool = _device.cupy().get_default_memory_pool()
    return f"{pool.total_bytes() / 1e9:.1f} GB held by CuPy"


def encore(paths):
    grids = _hemisphere_grids(DEFAULT_ORDER)
    densities = [np.asarray(sbci.load(p).dense(), dtype=np.float64) for p in paths[:3]]
    cpu, gpu = Encore(*grids), Encore(*grids, device="gpu")

    template, cpu_s = timed("cpu", lambda: cpu.template(densities))
    first, first_s = timed("gpu", lambda: gpu.template(densities))
    again, again_s = timed("gpu", lambda: gpu.template(densities))
    print(
        f"  the template of three: CPU {cpu_s:.1f} s, GPU {first_s:.2f} s the first time and "
        f"{again_s:.2f} s the second; the GPU's {gap(first, template):.1e} from the CPU's, "
        f"its second {gap(again, first):.1e} from its first",
        flush=True,
    )

    def registered(engine, moving):
        trace = []
        aligned, *_ = engine.register(
            template, moving, target_is_root=True, callback=lambda _, c: trace.append(c)
        )
        return np.asarray(trace), aligned

    runs = {}
    for name, device, engine, moving in (
        ("CPU", "cpu", cpu, densities[1]),
        ("twin", "cpu", cpu, nudged(densities[1])),
        ("GPU", "gpu", gpu, densities[1]),
        ("GPU again", "gpu", gpu, densities[1]),
    ):
        runs[name] = timed(device, lambda e=engine, m=moving: registered(e, m))
    (trace, aligned), cpu_s = runs["CPU"]
    print(
        f"  one registration onto it: CPU {cpu_s:.1f} s, GPU {runs['GPU'][1]:.2f} s the first "
        f"time and {runs['GPU again'][1]:.2f} s the second ({cpu_s / runs['GPU again'][1]:.0f}x)"
    )
    for name in ("twin", "GPU", "GPU again"):
        (other_trace, other_aligned), _ = runs[name]
        steps = f"{len(other_trace)} steps against {len(trace)}"
        if len(other_trace) == len(trace):
            steps += f", the cost trace {gap(other_trace, trace):.1e} from the CPU's"
        print(f"    {name}: {steps}, the aligned density {gap(other_aligned, aligned):.1e}")
    print(f"  final costs: CPU {trace[-1]:.10f}, GPU {runs['GPU'][0][0][-1]:.10f}; {memory()}")


def conseal(paths):
    lh, rh = default_grids()
    begun = time.perf_counter()
    kernel, derivative = HeatKernelBuilder(lh, rh, DEFAULT_KERNEL_DEGREE).compute(DEFAULT_SIGMA)
    print(f"  the kernel, on the CPU: {time.perf_counter() - begun:.1f} s, {kernel.nnz} entries")
    subjects = [
        EndpointConnectome.from_endpoints(sbci.load(p).endpoints, lh, rh) for p in paths[:2]
    ]
    print(f"  {subjects[0].n_streamlines} and {subjects[1].n_streamlines} streamlines")

    sent_kernel, sent_derivative = _sent(kernel, derivative, "gpu")
    on_gpu = subjects[0]._to_device("gpu")
    q, cpu_s = timed("cpu", lambda: subjects[0].q_transform(kernel, derivative))
    q_first, first_s = timed("gpu", lambda c=on_gpu: c.q_transform(sent_kernel, sent_derivative))
    q_again, again_s = timed("gpu", lambda c=on_gpu: c.q_transform(sent_kernel, sent_derivative))
    print(
        f"  Q with its derivatives: CPU {cpu_s:.1f} s, GPU {first_s:.2f} s the first time and "
        f"{again_s:.2f} s the second; the GPU's "
        + ", ".join(f"{gap(g, c):.1e}" for g, c in zip(q_first, q, strict=True))
        + " from the CPU's, its second "
        + ", ".join(f"{gap(g, f):.1e}" for g, f in zip(q_again, q_first, strict=True))
        + " from its first",
        flush=True,
    )
    del q, q_first, q_again

    engine = ConSEAL(lh, rh)
    lh_warp, rh_warp = engine.new_warps()
    rng = np.random.default_rng(0)
    lh_warp.compose(0.002 * rng.normal(size=(lh.n_vertices, 2)))
    rh_warp.compose(0.002 * rng.normal(size=(rh.n_vertices, 2)))
    moved = subjects[0].copy()
    _, cpu_s = timed("cpu", lambda c=moved: c.warp(lh_warp, rh_warp))
    _, first_s = timed("gpu", lambda c=on_gpu: c.warp(lh_warp, rh_warp))
    _, again_s = timed("gpu", lambda c=on_gpu: c.warp(lh_warp, rh_warp))
    back = on_gpu._to_device("cpu")
    same = all(
        np.array_equal(getattr(back, name), getattr(moved, name))
        for name in ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out")
    )
    print(
        f"  relocating the endpoints: CPU {cpu_s:.1f} s, GPU {first_s:.2f} s the first time and "
        f"{again_s:.2f} s the second; "
        + ("the CPU's faces and weights, bit for bit" if same else "NOT the CPU's"),
        flush=True,
    )
    del on_gpu, back, moved

    fixed = subjects[0].q_transform(kernel)
    runs = {}
    for name, device, target in (
        ("CPU", "cpu", fixed),
        ("twin", "cpu", nudged(fixed)),
        ("GPU", "gpu", fixed),
        ("GPU again", "gpu", fixed),
    ):
        engine = ConSEAL(lh, rh, max_iterations=10, device=device)
        runs[name] = timed(
            device,
            lambda e=engine, t=target: e.register(t, subjects[1], kernel, derivative),
        )
    (_, _, costs, warped), cpu_s = runs["CPU"]
    print(
        f"  ten iterations of a registration: CPU {cpu_s:.1f} s, GPU {runs['GPU'][1]:.2f} s the "
        f"first time and {runs['GPU again'][1]:.2f} s the second "
        f"({cpu_s / runs['GPU again'][1]:.0f}x)"
    )
    for name in ("twin", "GPU", "GPU again"):
        (_, _, other_costs, other_warped), _ = runs[name]
        steps = f"{len(other_costs) - 1} iterations against {len(costs) - 1}"
        if len(other_costs) == len(costs):
            steps += f", the costs {gap(other_costs, costs):.1e} from the CPU's"
        faces = np.mean(faces_of(other_warped) == faces_of(warped))
        print(
            f"    {name}: {steps}; the endpoints at most "
            f"{endpoints_apart(other_warped, warped):.1e} rad from the CPU's, "
            f"{100 * faces:.3f}% in the same face"
        )
    print(f"  {memory()}")


def align(paths, count):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_figures import resmoothed

    from sbci.grid import to_condensed

    subjects = [sbci.load(p) for p in paths[:count]]
    smoothed = [resmoothed(s, s.endpoints) for s in subjects]
    results = {}
    for device in ("cpu", "gpu"):
        results[device], seconds = timed(
            device,
            lambda d=device: sbci.align(smoothed, max_iterations=10, backtracks=4, device=d),
        )
        aligned = np.stack(
            [to_condensed(np.asarray(a)).astype(np.float32) for a in results[device].aligned]
        )
        correlation = np.corrcoef(aligned)[np.triu_indices(len(aligned), 1)].mean()
        print(
            f"  {device}: {seconds:.0f} s, mean pairwise correlation {correlation:.4f}", flush=True
        )
    cpu, gpu = results["cpu"], results["gpu"]
    print(f"  the templates: the GPU's {gap(gpu.template, cpu.template):.1e} from the CPU's")
    same = [len(g) == len(c) for g, c in zip(gpu.traces, cpu.traces, strict=True)]
    worst = max(gap(g, c) for g, c, s in zip(gpu.traces, cpu.traces, same, strict=True) if s)
    print(f"  {sum(same)} of {count} took as many steps; their traces at most {worst:.1e} apart")
    print(f"  {memory()}")


def repeat(paths):
    cupy = _device.cupy()

    def same(first, again):
        first, again = _device.to_host(first), _device.to_host(again)
        if np.array_equal(first, again):
            return "the same bits"
        return f"DIFFERENT, by {np.abs(first - again).max():.1e} of {np.abs(first).max():.1e}"

    def clocked(compute, times=3):
        compute()
        _device.synchronize("gpu")
        begun = time.perf_counter()
        for _ in range(times):
            result = compute()
        _device.synchronize("gpu")
        return result, (time.perf_counter() - begun) / times

    lh, rh = default_grids()
    kernel, derivative = HeatKernelBuilder(lh, rh, DEFAULT_KERNEL_DEGREE).compute(DEFAULT_SIGMA)
    sent_kernel, sent_derivative = _sent(kernel, derivative, "gpu")
    transposed = sent_kernel.T.tocsr()
    dense = cupy.asarray(np.random.default_rng(0).random(kernel.shape))
    mine, mine_s = clocked(lambda d=dense: _device.rows_times_dense(transposed, d))
    theirs, theirs_s = clocked(lambda d=dense: transposed @ d)
    print(
        f"  K^T times a dense {kernel.shape[0]}^2: rows_times_dense {1e3 * mine_s:.1f} ms, "
        f"cuSPARSE {1e3 * theirs_s:.1f} ms; they agree to "
        f"{float(cupy.abs(mine - theirs).max() / cupy.abs(theirs).max()):.1e}; run twice, "
        f"rows_times_dense gives {same(mine, _device.rows_times_dense(transposed, dense))} "
        f"and cuSPARSE {same(theirs, transposed @ dense)}",
        flush=True,
    )
    del dense, mine, theirs
    connectome = EndpointConnectome.from_endpoints(sbci.load(paths[0]).endpoints, lh, rh)
    on_gpu = connectome._to_device("gpu")
    first = on_gpu.q_transform(sent_kernel, sent_derivative)
    again = on_gpu.q_transform(sent_kernel, sent_derivative)
    print(
        "  a subject's density and its derivatives, twice: "
        + ", ".join(same(f, a) for f, a in zip(first, again, strict=True))
    )
    del first, again, on_gpu

    grids = _hemisphere_grids(DEFAULT_ORDER)
    densities = [np.asarray(sbci.load(p).dense(), dtype=np.float64) for p in paths[:2]]
    encore = Encore(*grids, max_iterations=3, device="gpu")
    root = encore.root(densities[0])
    runs = [encore.register(root, densities[1], target_is_root=True) for _ in range(2)]
    print(
        f"  three steps of ENCORE, twice: the aligned density {same(runs[0][0], runs[1][0])}, "
        f"the cost {'the same' if runs[0][3] == runs[1][3] else 'DIFFERENT'}"
    )
    subjects = [
        EndpointConnectome.from_endpoints(sbci.load(p).endpoints, lh, rh) for p in paths[:2]
    ]
    engine = ConSEAL(lh, rh, max_iterations=3, device="gpu")
    runs = [engine.register(subjects[0], subjects[1], kernel, derivative) for _ in range(2)]
    print(
        f"  three iterations of ConSEAL, twice: the costs {same(runs[0][2], runs[1][2])}, the "
        f"endpoints {same(runs[0][3].weights_in, runs[1][3].weights_in)}"
    )


def spread(draws=10):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import test_gpu as cases
    from sbci.conseal import endpoints_align

    def report(name, base, runs):
        for key in base:
            gaps = np.array([cases.gap(run[key], base[key]) for run in runs])
            print(
                f"  {name}, {key}: {gaps.min():.1e} to {gaps.max():.1e}, the farthest "
                f"{gaps.max() / gaps.min():.1f} times the nearest",
                flush=True,
            )

    grid = cases.encore_grid()
    cohort = cases.densities(grid)
    for steps in (10, 50):
        encore = Encore(grid, grid, max_iterations=steps)
        template = encore.template(cohort)

        def registered(moving, engine=encore, template=template):
            trace = []
            aligned, lh, rh, _ = engine.register(
                template, moving, target_is_root=True, callback=lambda _, c: trace.append(c)
            )
            return {
                "trace": trace,
                "aligned": aligned,
                "warp": lh.vertices,
                "Jacobian": rh.jacobian,
            }

        base = registered(cohort[1])
        runs = [registered(cases.nudged(cohort[1], seed)) for seed in range(draws)]
        if steps == 10:
            report("Encore.register, ten steps", base, runs)
        else:
            largest = [
                float(
                    np.abs(run["aligned"] - base["aligned"]).max() / np.abs(base["aligned"]).max()
                )
                for run in runs
            ]
            print(
                f"  Encore.register, fifty steps: the aligned density's largest difference "
                f"{min(largest):.1e} to {max(largest):.1e} of its largest entry"
            )

    def aligned(densities):
        result = sbci.align(densities, grids=(grid, grid), max_iterations=6)
        return {
            "traces": np.concatenate(result.traces),
            "aligned": np.stack(result.aligned),
            "warps": np.stack([warp.lh_vertices for warp in result.warps]),
        }

    cohort4 = cohort[:4]
    base = aligned(cohort4)
    runs = [
        aligned([cases.nudged(d, 10 * seed + i) for i, d in enumerate(cohort4)])
        for seed in range(draws)
    ]
    report("align, six steps", base, runs)

    grid = cases.conseal_grid()
    subjects = cases.subjects(grid)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    for settings in ({}, {"area_weighted": True}):
        engine = ConSEAL(grid, grid, max_iterations=8, **settings)
        template = engine.template(subjects, kernel)
        for rotate in (False, True):

            def registered(fixed, engine=engine, rotate=rotate):
                lh, _, costs, warped = engine.register(
                    fixed, subjects[1], kernel, derivative, init_rotation=rotate
                )
                return {
                    "costs": costs,
                    "velocity": lh.velocity,
                    "endpoints": np.concatenate(warped.positions()),
                }, faces_of(warped)

            base, faces = registered(template)
            runs = [registered(cases.nudged(template, seed)) for seed in range(draws)]
            name = f"ConSEAL.register, {settings or 'plain'}, init_rotation={rotate}"
            report(name, base, [run for run, _ in runs])
            moved = [float(np.mean(other != faces)) for _, other in runs]
            print(f"  {name}: {100 * min(moved):.2f}% to {100 * max(moved):.2f}% in another face")

    settings = dict(sigma=0.05, kernel_degree=12, order=3, max_iterations=4, grids=(grid, grid))
    for template, still in ((None, None), (1, 1), (subjects[2], 2)):
        moved = [i for i in range(len(subjects)) if i != still]

        def compared(result, moved=moved):
            return {
                "costs": np.concatenate([result.costs[i] for i in moved]),
                "endpoints": np.concatenate(
                    [np.concatenate(result.connectomes[i].positions()) for i in moved]
                ),
            }

        first = endpoints_align(subjects, template=template, **settings)
        runs = [
            compared(
                endpoints_align(subjects, template=cases.nudged(first.template, s), **settings)
            )
            for s in range(draws)
        ]
        report(f"endpoints_align, template={type(template).__name__}", compared(first), runs)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "spread":
        print("the CPU's twins", flush=True)
        spread()
        sys.exit(0)
    if mode not in ("encore", "conseal", "align", "repeat"):
        sys.exit(__doc__)
    paths = released()
    cupy = _device.cupy()
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"{mode} on {props['name'].decode()}, {props['totalGlobalMem'] / 1e9:.0f} GB", flush=True)
    if mode == "encore":
        encore(paths)
    elif mode == "conseal":
        conseal(paths)
    elif mode == "repeat":
        repeat(paths)
    else:
        align(paths, int(sys.argv[2]) if len(sys.argv) > 2 else len(paths))
