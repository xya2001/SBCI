"""The GPU's own review (PORTING.md item 28): what tests/test_gpu.py cannot show without a GPU.

Run from the root of the tree, on a GPU node with CuPy and CUDA's module loaded:

    python tests/reference/gpu_review_probe.py margins     # a few minutes
    python tests/reference/gpu_review_probe.py repeat      # a minute
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_review_probe.py relocate
    python tests/reference/gpu_review_probe.py tables [SUBJECTS]
    python tests/reference/gpu_review_probe.py empty
    SBCI_HCP_DIR=hcp-ya python tests/reference/gpu_review_probe.py rotation [gpu] [cpu]
    python tests/reference/gpu_review_probe.py compile     # WITHOUT CUDA's module

``margins``  every registration tests/test_gpu.py compares, with the GPU's distance from the
             CPU, the allowance (twenty times the farther twin, or the floor) and their
             ratio: how near this GPU and this CPU come to failing the test.
``repeat``   the routes test_the_gpu_repeats_itself leaves out, run twice: Concon's analytic
             derivative, the area-weighted cost, the sparse route on ico4; and the GPU's
             adjacency's entries against the CPU's.
``relocate`` a released subject's endpoints moved through a small warp: how many points
             the GPU's table cannot settle (sent to its search of every face), and the time.
``tables``   SUBJECTS synthetic subjects (default 6) on the bundled ico4 grids, 20,000
             random streamlines each, aligned with one iteration by
             ``endpoints_align(device="gpu")`` and then registered one at a time: the bytes
             CuPy's pool holds after each, what the subjects' point-search tables weigh,
             and the pool once those are dropped. Until item 28 the tables a registration
             made, 4.14 MiB a subject on ico4, stayed on the GPU as long as the subject did.
``empty``    a connectome with no streamlines, as a template's subject and as the fixed
             connectome, on the CPU and on the GPU; and one streamline's density on both.
``rotation`` two released subjects: a registration of three iterations with and without
             the rotation search (``init_rotation``), its time and the search's, on each
             device named (both by default); then ``endpoints_align`` of the two with the
             search, and whether the GPU picks the CPU's rotations, bit for bit.
``compile``  the device check with the default kernel cache, warm from earlier runs, and
             without CUDA's module: "refused" means a warm cache hides nothing.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "tests"))


def margins():
    import test_gpu as t
    from sbci.alignment import Encore, align
    from sbci.conseal import ConSEAL, HeatKernelBuilder, endpoints_align

    def report(name, gpu, cpu, twins, floor):
        for key in cpu:
            spread = max(t.gap(twin[key], cpu[key]) for twin in twins)
            distance = t.gap(gpu[key], cpu[key])
            allowed = max(20 * spread, floor[key])
            print(
                f"  {name}, {key}: GPU {distance:.1e}, twins at most {spread:.1e}, floor "
                f"{floor[key]:.1e}; {distance / allowed:.3f} of the allowance",
                flush=True,
            )

    grid = t.encore_grid()
    cohort = t.densities(grid)
    cpu = Encore(grid, grid, max_iterations=10)
    gpu = Encore(grid, grid, max_iterations=10, device="gpu")
    template = cpu.template(cohort)

    def registered(engine, moving):
        trace = []
        aligned, lh, rh, _ = engine.register(
            template, moving, target_is_root=True, callback=lambda _, c: trace.append(c)
        )
        return {"trace": trace, "aligned": aligned, "warp": lh.vertices, "Jacobian": rh.jacobian}

    report(
        "Encore.register",
        registered(gpu, cohort[1]),
        registered(cpu, cohort[1]),
        [registered(cpu, t.nudged(cohort[1], s)) for s in (1, 2)],
        t.FLOORS["Encore.register"],
    )

    four = t.densities(grid, count=4)

    def aligned(densities, device="cpu"):
        result = align(densities, grids=(grid, grid), max_iterations=6, device=device)
        return {
            "traces": np.concatenate(result.traces),
            "aligned": np.stack(result.aligned),
            "warps": np.stack([w.lh_vertices for w in result.warps]),
        }

    report(
        "align",
        aligned(four, "gpu"),
        aligned(four),
        [aligned([t.nudged(d, 10 * s + i) for i, d in enumerate(four)]) for s in (1, 2)],
        t.FLOORS["align"],
    )

    egrid = t.conseal_grid()
    subjects = t.subjects(egrid)
    kernel, derivative = HeatKernelBuilder(egrid, egrid, degree=12).compute(0.05)
    for settings in ({}, {"area_weighted": True}):
        engines = {
            d: ConSEAL(egrid, egrid, max_iterations=8, device=d, **settings) for d in ("cpu", "gpu")
        }
        fixed = engines["cpu"].template(subjects, kernel)
        for rotate in (False, True):

            def run(device, target, rotate=rotate, engines=engines):
                lh, _, costs, warped = engines[device].register(
                    target, subjects[1], kernel, derivative, init_rotation=rotate
                )
                return {
                    "costs": costs,
                    "velocity": lh.velocity,
                    "endpoints": np.concatenate(warped.positions()),
                }

            report(
                f"ConSEAL.register {settings or 'plain'} rotate={rotate}",
                run("gpu", fixed),
                run("cpu", fixed),
                [run("cpu", t.nudged(fixed, s)) for s in (1, 2)],
                t.FLOORS["ConSEAL.register"],
            )

    settings = dict(sigma=0.05, kernel_degree=12, order=3, max_iterations=4, grids=(egrid, egrid))
    for template, still in ((None, None), (1, 1), (subjects[2], 2)):
        moved = [i for i in range(len(subjects)) if i != still]

        def compared(result, moved=moved):
            return {
                "costs": np.concatenate([result.costs[i] for i in moved]),
                "endpoints": np.concatenate(
                    [np.concatenate(result.connectomes[i].positions()) for i in moved]
                ),
            }

        cpu_result = endpoints_align(subjects, template=template, **settings)
        gpu_result = endpoints_align(subjects, template=template, device="gpu", **settings)
        twins = [
            compared(
                endpoints_align(subjects, template=t.nudged(cpu_result.template, s), **settings)
            )
            for s in (1, 2)
        ]
        report(
            f"endpoints_align template={type(template).__name__}",
            compared(gpu_result),
            compared(cpu_result),
            twins,
            t.FLOORS["endpoints_align"],
        )


def repeat_ico4():
    import sbci.conseal as conseal
    from sbci import _device
    from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, _sent, default_grids

    sys.path.insert(0, str(Path.cwd() / "tests" / "reference"))
    from cpu_parity_probe import synthetic

    import test_gpu as t
    from sbci.alignment import Concon

    def same(a, b):
        a, b = _device.to_host(a), _device.to_host(b)
        return "same" if np.array_equal(a, b) else f"DIFFERENT by {np.abs(a - b).max():.1e}"

    grid = t.encore_grid()
    root = np.sqrt(t.densities(grid, count=1)[0])
    analytic = Concon(grid, grid, derivative="analytic", device="gpu")
    first, again = analytic.derivative(root), analytic.derivative(root)
    print(
        "  Concon's analytic derivative twice:",
        [same(a, b) for a, b in zip(first, again, strict=True)],
        flush=True,
    )

    lh, rh = default_grids(order=3)
    rng = np.random.default_rng(13)
    subject = EndpointConnectome.from_points(lh, rh, *synthetic(rng, n=20000, jitter=0.15))
    kernel, derivative = HeatKernelBuilder(lh, rh).compute()
    on_gpu = subject._to_device("gpu")
    sent = _sent(kernel, derivative, "gpu")
    saved, conseal.SPARSE_PRODUCT_FRACTION = conseal.SPARSE_PRODUCT_FRACTION, np.inf
    try:
        first, again = on_gpu.q_transform(*sent), on_gpu.q_transform(*sent)
    finally:
        conseal.SPARSE_PRODUCT_FRACTION = saved
    print(
        "  ico4, the sparse route, twice:",
        [same(a, b) for a, b in zip(first, again, strict=True)],
        flush=True,
    )
    gpu_adjacency, cpu_adjacency = on_gpu._sparse_adjacency(), subject._sparse_adjacency()
    print(
        f"  the adjacency's entries: GPU {gpu_adjacency.nnz}, CPU {cpu_adjacency.nnz}", flush=True
    )
    engine = ConSEAL(lh, rh, area_weighted=True, device="gpu")
    difference = first[0] - first[0].T
    print(
        "  the area-weighted cost twice:",
        "same" if engine.cost(difference) == engine.cost(difference) else "DIFFERENT",
    )


def relocate():
    import sbci
    from sbci import _device
    from sbci.alignment import MeshQuery
    from sbci.conseal import ConSEAL, EndpointConnectome, default_grids

    path = sorted(Path(os.environ["SBCI_HCP_DIR"]).glob("sub-*_sc.h5"))[0]
    lh, rh = default_grids()
    subject = EndpointConnectome.from_endpoints(sbci.load(path).endpoints, lh, rh)
    lh_warp, rh_warp = ConSEAL(lh, rh).new_warps()
    rng = np.random.default_rng(0)
    lh_warp.compose(0.002 * rng.normal(size=(lh.n_vertices, 2)))
    rh_warp.compose(0.002 * rng.normal(size=(rh.n_vertices, 2)))
    on_gpu = subject._to_device("gpu")
    searched = []
    exact = MeshQuery._exact_faces_device
    MeshQuery._exact_faces_device = staticmethod(
        lambda p, a, b, c: searched.append(int(p.shape[0])) or exact(p, a, b, c)
    )
    on_gpu.warp(lh_warp, rh_warp)
    _device.synchronize("gpu")
    searched.clear()
    begun = time.perf_counter()
    on_gpu.warp(lh_warp, rh_warp)
    _device.synchronize("gpu")
    print(
        f"  relocating {2 * subject.n_streamlines} endpoints: {time.perf_counter() - begun:.2f}"
        f" s; sent to the search of every face: {sum(searched)}"
    )


def tables(count):
    import cupy

    from sbci.conseal import (
        ConSEAL,
        EndpointConnectome,
        HeatKernelBuilder,
        default_grids,
        endpoints_align,
    )

    def subject(lh, rh, rng, n=20_000):
        p_in, p_out = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        return EndpointConnectome.from_points(
            lh, rh, p_in, p_out, rng.integers(0, 2, n), rng.integers(0, 2, n)
        )

    def weight(connectomes):
        total = 0
        for c in connectomes:
            for query in c._query.values():
                if query._on_device is not None:
                    total += sum(int(t.nbytes) for t in query._on_device)
        return total

    lh, rh = default_grids()
    rng = np.random.default_rng(0)
    pool = cupy.get_default_memory_pool()
    mib = 2.0**20

    cohort = [subject(lh, rh, rng) for _ in range(count)]
    gc.collect()
    start = pool.used_bytes()
    result = endpoints_align(cohort, max_iterations=1, device="gpu")
    gc.collect()
    print(
        f"endpoints_align of {count}: pool {start / mib:.1f} -> {pool.used_bytes() / mib:.1f} MiB; "
        f"the subjects' tables {weight(cohort) / mib:.2f} MiB; "
        f"returned connectomes on {sorted({c.device for c in result.connectomes})}, "
        f"sharing them: {weight(result.connectomes) / mib:.2f} MiB"
    )
    del result
    gc.collect()
    print(f"result dropped, subjects kept: pool {pool.used_bytes() / mib:.1f} MiB")
    for c in cohort:
        for query in c._query.values():
            query._on_device = None
    gc.collect()
    print(f"tables cleared: pool {pool.used_bytes() / mib:.1f} MiB")

    # one registration at a time, as a loop over a cohort registers onto a template
    kernel, derivative = HeatKernelBuilder(lh, rh).compute()
    engine = ConSEAL(lh, rh, max_iterations=1, device="gpu")
    fixed = subject(lh, rh, rng)
    kept = []
    gc.collect()
    base = pool.used_bytes()
    for i in range(count):
        moving = subject(lh, rh, rng)
        kept.append(engine.register(fixed, moving, kernel, derivative)[3])
        del moving
        gc.collect()
        print(
            f"after registration {i + 1}: pool {(pool.used_bytes() - base) / mib:+.2f} MiB "
            f"over the start; tables reachable from the kept results {weight(kept) / mib:.2f} MiB"
        )


def empty():
    import cupy

    from sbci import _device
    from sbci.alignment import SphericalGrid
    from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, _sent, icosphere

    def attempt(what, run):
        try:
            out = run()
            print(f"{what}: returned {out!r}"[:300])
        except Exception as exc:  # noqa: BLE001
            print(f"{what}: {type(exc).__name__}: {str(exc)[:200]}")
            if not isinstance(exc, ValueError) or "streamlines" not in str(exc):
                traceback.print_exc(limit=6)

    attempt(
        "cupy.bincount(empty, minlength=5)",
        lambda: cupy.bincount(cupy.empty(0, dtype=cupy.int64), minlength=5).get(),
    )
    empty_int = cupy.empty(0, dtype=cupy.int64)
    attempt(
        "_device.summed_csr with no entries",
        lambda: _device.summed_csr(empty_int, empty_int, cupy.empty(0), (10, 10)).nnz,
    )

    grid = SphericalGrid(*icosphere(2), order=3)
    rng = np.random.default_rng(0)
    n = 300
    full = EndpointConnectome.from_points(
        grid,
        grid,
        rng.normal(size=(n, 3)),
        rng.normal(size=(n, 3)),
        rng.integers(0, 2, n),
        rng.integers(0, 2, n),
    )
    none = EndpointConnectome.from_points(grid, grid, np.empty((0, 3)), np.empty((0, 3)), [], [])
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    for device in ("cpu", "gpu"):
        engine = ConSEAL(grid, grid, max_iterations=1, device=device)
        attempt(
            f"{device.upper()}: ConSEAL.template with a subject of no streamlines",
            lambda engine=engine: engine.template([full, none], kernel).shape,
        )
        attempt(
            f"{device.upper()}: ConSEAL.register onto a connectome of no streamlines",
            lambda engine=engine: engine.register(none, full, kernel, derivative)[2],
        )

    one = EndpointConnectome.from_points(
        grid, grid, rng.normal(size=(1, 3)), rng.normal(size=(1, 3)), [0], [1]
    )
    q, d1, d2 = one.q_transform(kernel, derivative)
    q_gpu, d1_gpu, d2_gpu = one._to_device("gpu").q_transform(*_sent(kernel, derivative, "gpu"))
    for name, cpu, gpu in (("Q", q, q_gpu), ("dQ e1", d1, d1_gpu), ("dQ e2", d2, d2_gpu)):
        scale = max(float(np.abs(cpu).max()), 1e-300)
        print(
            f"one streamline, {name}: largest difference "
            f"{float(np.abs(gpu.get() - cpu).max()) / scale:.2e} of the largest entry"
        )


def rotation(devices):
    import sbci
    from sbci import _device
    from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

    def gap(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    def degrees(first, second):
        out = []
        for a, b in zip(first.positions(), second.positions(), strict=True):
            chord = np.linalg.norm(a - b, axis=1)
            out.append(np.degrees(2 * np.arcsin(np.minimum(chord / 2, 1.0))))
        return float(np.concatenate(out).max())

    paths = sorted(Path(os.environ["SBCI_HCP_DIR"]).glob("sub-*_sc.h5"))[:2]
    lh, rh = default_grids()
    fixed, moving = (
        EndpointConnectome.from_endpoints(sbci.load(p).endpoints, lh, rh) for p in paths
    )
    kernel, derivative = HeatKernelBuilder(lh, rh).compute()
    spent = {}
    original = ConSEAL._rigid

    def timed(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            _device.synchronize(self.device)
            spent[self.device] = spent.get(self.device, 0.0) + time.perf_counter() - start

    ConSEAL._rigid = timed
    try:
        for device in devices:
            for rotate in (False, True):
                engine = ConSEAL(lh, rh, max_iterations=3, device=device)
                spent.clear()
                start = time.perf_counter()
                costs = engine.register(fixed, moving, kernel, derivative, init_rotation=rotate)[2]
                _device.synchronize(device)
                total = time.perf_counter() - start
                print(
                    f"{device}, init_rotation={rotate}: {total:.1f} s for {len(costs) - 1} "
                    f"iterations, the rotation search {spent.get(device, 0.0):.1f} s",
                    flush=True,
                )
    finally:
        ConSEAL._rigid = original

    subjects = [sbci.load(p) for p in paths]
    results = {}
    for device in devices:
        start = time.perf_counter()
        results[device] = sbci.endpoints_align(
            subjects, max_iterations=3, init_rotation=True, device=device
        )
        print(
            f"{device}: endpoints_align, init_rotation=True, 3 iterations: "
            f"{time.perf_counter() - start:.0f} s",
            flush=True,
        )
    if len(results) == 2:
        cpu, gpu = results["cpu"], results["gpu"]
        print(f"template {gap(gpu.template, cpu.template):.1e} from the CPU's")
        for i in range(2):
            same = all(
                np.array_equal(getattr(gpu.warps[i], side), getattr(cpu.warps[i], side))
                for side in ("lh_rigid", "rh_rigid")
            )
            print(
                f"subject {i}: the CPU's rotations bit for bit: {same}; costs "
                f"{gap(gpu.costs[i], cpu.costs[i]):.1e} apart; endpoints at most "
                f"{degrees(gpu.connectomes[i], cpu.connectomes[i]):.1e} deg apart"
            )


def compile_check():
    from sbci import _device

    print("  CUPY_CACHE_DIR:", os.environ.get("CUPY_CACHE_DIR", "(default)"))
    try:
        cp = _device.cupy()
        print("  the check: passed")
    except Exception as exc:  # noqa: BLE001
        print(f"  the check: refused ({type(exc).__name__}: {str(exc)[:200]})")
        return
    try:
        fresh = cp.ElementwiseKernel("T x", "T y", f"y = x + {os.getpid()}", "review_fresh")
        x = cp.arange(2.0)
        fresh(x, cp.empty_like(x))
        print("  a kernel never compiled before: compiled and ran")
    except Exception as exc:  # noqa: BLE001
        print(f"  a kernel never compiled before: FAILED ({type(exc).__name__}: {str(exc)[:200]})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    modes = ("margins", "repeat", "relocate", "tables", "empty", "rotation", "compile")
    if mode not in modes:
        sys.exit(__doc__)
    if mode != "compile":
        from sbci import _device

        props = _device.cupy().cuda.runtime.getDeviceProperties(0)
        print(f"{mode} on {props['name'].decode()}", flush=True)
    if mode == "tables":
        tables(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif mode == "rotation":
        rotation(sys.argv[2:] or ("gpu", "cpu"))
    else:
        {
            "margins": margins,
            "repeat": repeat_ico4,
            "relocate": relocate,
            "empty": empty,
            "compile": compile_check,
        }[mode]()
