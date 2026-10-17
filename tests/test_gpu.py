"""``device="gpu"``: refused plainly without a GPU, and with one, what the CPU gives.

The first tests run anywhere. The rest need CuPy and a CUDA GPU and skip without them; CI
has neither, so they are run by hand on a GPU node (VERIFICATION.md, PORTING.md item 26).
On the CPU the same code runs with NumPy, and the rest of the suite checks it there.

Each step of the arithmetic agrees with the CPU's to rounding, or bit for bit where the
GPU adds in NumPy's order. A whole registration cannot be held to that: its warps locate
points on the mesh, and where a point sits on an edge rounding decides which face carries
it, after which the descent goes its own way. The CPU does the same with itself, so a
registration on the GPU is held to the CPU's own spread, its twins: the CPU's results
when an input moves by one unit in the last place.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sbci import _device
from sbci.alignment import (
    Concon,
    Encore,
    MeshQuery,
    SphericalGrid,
    SphericalWarp,
    align,
    rotate_off_poles,
)
from sbci.conseal import (
    ConSEAL,
    EndpointConnectome,
    HeatKernelBuilder,
    StationaryWarp,
    _sent,
    endpoints_align,
    icosphere,
)

# --- without a GPU -----------------------------------------------------------------------


def encore_grid():
    vertices, faces = icosphere(2)
    return SphericalGrid(rotate_off_poles(vertices), faces, order=2)


def conseal_grid():
    vertices, faces = icosphere(2)
    return SphericalGrid(vertices, faces, order=3)


def test_a_device_other_than_cpu_or_gpu_is_refused():
    grid = encore_grid()
    for device in ("tpu", "GPU", "cuda", 0, None):
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            Encore(grid, grid, device=device)
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            ConSEAL(grid, grid, device=device)
    density = np.ones((2 * grid.n_vertices,) * 2)
    with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
        align([density, density], grids=(grid, grid), device="cuda")
    with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
        endpoints_align([], device="cuda")


def test_the_gpu_without_cupy_is_refused_with_how_to_install_it(monkeypatch):
    """Before any work: the error names the extra that installs CuPy."""
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy then fails, as when missing
    grid = encore_grid()
    density = np.ones((2 * grid.n_vertices,) * 2)
    for attempt in (
        lambda: Encore(grid, grid, device="gpu"),
        lambda: ConSEAL(grid, grid, device="gpu"),
        lambda: align([density, density], grids=(grid, grid), device="gpu"),
        lambda: endpoints_align([], device="gpu"),
    ):
        with pytest.raises(ImportError, match=r"pip install 'sbci\[gpu\]'"):
            attempt()


def test_cupy_without_a_gpu_is_refused(monkeypatch):
    def no_device():
        raise RuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected")

    grid = encore_grid()
    for count in (no_device, lambda: 0):  # CuPy raises, or counts none
        runtime = types.SimpleNamespace(getDeviceCount=count)
        monkeypatch.setitem(
            sys.modules, "cupy", types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=runtime))
        )
        with pytest.raises(RuntimeError, match="sees no CUDA GPU"):
            Encore(grid, grid, device="gpu")


def test_cupy_that_cannot_compile_a_kernel_is_refused_at_once(monkeypatch):
    """Without CUDA's libraries the device was found, and the first kernel failed mid-run.

    CuPy compiles a kernel when it is first called, not when it is made, and so does the
    stand-in: one that failed when made passed a check that never ran its kernel.
    """

    class Kernel:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            raise ImportError('Failure finding "libnvJitLink.so.13 or libnvJitLink.so.12"')

    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1)
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(runtime=runtime),
        arange=np.arange,
        empty_like=np.empty_like,
        ElementwiseKernel=Kernel,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(_device, "_USABLE", None)  # whatever passed it is put back after
    grid = encore_grid()
    with pytest.raises(RuntimeError, match="cannot compile or run a kernel.*libnvJitLink.*module"):
        Encore(grid, grid, device="gpu")


def test_a_gpu_that_is_busy_is_refused_at_once_saying_so(monkeypatch):
    """A GPU in another process's exclusive use was found, and the run failed at a kernel."""

    def busy(*args):
        raise RuntimeError("cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy")

    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1)
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(runtime=runtime),
        arange=busy,
        empty_like=np.empty_like,
        ElementwiseKernel=lambda *args: None,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    grid = encore_grid()
    with pytest.raises(RuntimeError, match="busy or in another process's exclusive use"):
        Encore(grid, grid, device="gpu")


# --- with one ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cp():
    try:
        return _device.cupy()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"no usable GPU here: {exc}")


EPS = np.finfo(np.float64).eps


def nudged(array, seed):
    """``array`` with every entry moved by one unit in its last place, up or down."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=np.shape(array))
    return np.asarray(array) * (1 + EPS * signs)


def gap(a, b):
    """How far ``a`` lies from ``b``, relative to ``b`` (Frobenius norms)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


#: Ten times the farthest of ten twins from the CPU's result, for each quantity the tests
#: below compare (``tests/reference/gpu_probe.py spread``, job 4685821, PORTING.md item 27).
FLOORS = {
    "Encore.register": {"trace": 2.0e-5, "aligned": 4.1e-5, "warp": 1.7e-5, "Jacobian": 1.9e-5},
    "align": {"traces": 2.4e-5, "aligned": 1.1e-3, "warps": 9.9e-6},
    "ConSEAL.register": {"costs": 1.7e-4, "velocity": 3.4e-4, "endpoints": 9.8e-4},
    "endpoints_align": {"costs": 4.8e-5, "endpoints": 8.0e-4},
}


def assert_within_the_cpus_spread(gpu, cpu, twins, what, floor, factor=20.0):
    """``gpu`` within ``factor`` times the farther of the CPU's twins, or ``floor``, of ``cpu``.

    On the cases here ten twins lie 2e-7 to 1.1e-4 from the CPU's result, and one can lie
    eighty times nearer than another (``tests/reference/gpu_probe.py spread``), so the two
    a test makes can both happen to lie near. Each quantity has a floor besides, ten times
    the farthest of the ten (``FLOORS``); the steps themselves are held to rounding by the
    tests above.
    """
    assert not _device.on_device(gpu), f"{what} came back as a CuPy array"
    spread = max(gap(twin, cpu) for twin in twins)
    distance = gap(gpu, cpu)
    assert distance <= max(factor * spread, floor), (
        f"{what}: the GPU lies {distance:.2e} from the CPU, its twins at most {spread:.2e}"
    )


def assert_close(gpu, cpu, rtol=1e-11, what=""):
    """``gpu`` is a NumPy array (or number) within ``rtol`` of ``cpu``'s largest entry."""
    assert not _device.on_device(gpu), f"{what} came back as a CuPy array"
    gpu, cpu = np.asarray(gpu), np.asarray(cpu)
    assert gpu.shape == cpu.shape, what
    scale = max(float(np.abs(cpu).max()), 1e-300)
    assert float(np.abs(gpu - cpu).max()) <= rtol * scale, (
        f"{what}: largest difference {np.abs(gpu - cpu).max():.3g} against {scale:.3g}"
    )


def bundles(grid, ends, width=6.0):
    both = np.vstack([grid.vertices, grid.vertices])
    density = np.zeros((both.shape[0], both.shape[0]))
    for a, b in ends:
        ka = np.exp(width * (both @ (np.asarray(a) / np.linalg.norm(a)) - 1))
        kb = np.exp(width * (both @ (np.asarray(b) / np.linalg.norm(b)) - 1))
        density += np.outer(ka, kb) + np.outer(kb, ka)
    np.fill_diagonal(density, 0.0)
    return density


def densities(grid, count=5, seed=5):
    rng = np.random.default_rng(seed)
    return [
        bundles(grid, [(rng.normal(size=3), rng.normal(size=3)) for _ in range(3)], width=4.0)
        for _ in range(count)
    ]


def von_mises(rng, centre, n, concentration=4.0):
    centre = np.asarray(centre, dtype=np.float64) / np.linalg.norm(centre)
    points = rng.normal(size=(n, 3)) + concentration * centre
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def subjects(grid, count=3, n=3000, seed=11):
    """Synthetic connectomes, streamlines in bundles within and across the hemispheres."""
    rng = np.random.default_rng(seed)
    made = []
    for _ in range(count):
        jitter = 0.15 * rng.normal(size=(6, 3))
        third = n // 3
        p_in = np.vstack(
            [
                von_mises(rng, np.add(c, j), third)
                for c, j in zip(
                    ([-1, 0.3, 0.2], [1, -0.2, 0.4], [-0.8, -0.5, 0.1]), jitter[:3], strict=True
                )
            ]
        )
        p_out = np.vstack(
            [
                von_mises(rng, np.add(c, j), third)
                for c, j in zip(
                    ([-0.5, -0.6, 0.6], [0.7, 0.6, -0.3], [0.9, 0.3, -0.3]), jitter[3:], strict=True
                )
            ]
        )
        h_in = np.r_[np.zeros(third), np.ones(third), np.zeros(third)].astype(int)
        h_out = np.r_[np.zeros(third), np.ones(third), np.ones(third)].astype(int)
        made.append(EndpointConnectome.from_points(grid, grid, p_in, p_out, h_in, h_out))
    return made


def test_mesh_query_on_the_gpu_finds_the_cpus_faces(cp):
    """The CPU's faces and weights, bit for bit, on the bundled ico4 sphere and a small one.

    Random points; the vertices themselves, where the faces around each tie and the
    lowest index wins; points just above the edges, where two faces nearly tie; and
    points well inside the sphere, which the candidates miss and every face is tried for.
    """
    from sbci.spec import N_VERTICES_PER_HEMI
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = np.array(sphere.vertices, dtype=np.float64)[:N_VERTICES_PER_HEMI]
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    faces = faces[(faces < N_VERTICES_PER_HEMI).all(axis=1)]
    rng = np.random.default_rng(2)
    for mesh, (v, f) in (("ico4", (vertices, faces)), ("icosphere(1)", icosphere(1))):
        query = MeshQuery(v, f)
        points = rng.normal(size=(50_000, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        edges = v[f[:, 0]] + v[f[:, 1]]
        edges /= np.linalg.norm(edges, axis=1, keepdims=True)
        for where, located in (
            ("random points", points),
            ("the vertices", v),
            ("above the edges", edges),
            ("inside the sphere", 0.5 * points[:500]),
        ):
            weights, indices, face = query.query_faces(located)
            on_gpu = query.query_faces(cp.asarray(located))
            assert all(_device.on_device(part) for part in on_gpu)
            weights_gpu, indices_gpu, face_gpu = (part.get() for part in on_gpu)
            what = f"{mesh}, {where}"
            np.testing.assert_array_equal(face_gpu, face, err_msg=what)
            np.testing.assert_array_equal(indices_gpu, indices, err_msg=what)
            np.testing.assert_array_equal(weights_gpu, weights, err_msg=what)


def test_concon_on_the_gpu_transports_as_on_the_cpu(cp):
    """To rounding; the derivative's central difference divides that by its step, 2e-5."""
    grid = encore_grid()
    density = densities(grid, count=1)[0]
    rng = np.random.default_rng(0)
    warps = []
    for _ in range(2):
        warp = SphericalWarp(grid)
        warps.append(warp.compose(0.02 * rng.normal(size=(grid.n_vertices, 2))))
    cpu = Concon(grid, grid)
    gpu = Concon(grid, grid, device="gpu")
    for name, compute, rtol in (
        ("evaluate", lambda c: c.evaluate(density, *warps), 1e-11),
        ("evaluate_root", lambda c: c.evaluate_root(np.sqrt(density), *warps), 1e-11),
        ("derivative", lambda c: c.derivative(np.sqrt(density))[0], 1e-9),
    ):
        on_gpu = compute(gpu)
        assert _device.on_device(on_gpu), f"Concon.{name} works on the GPU's arrays"
        assert_close(on_gpu.get(), compute(cpu), rtol=rtol, what=name)
    cpu = Concon(grid, grid, derivative="analytic")
    gpu = Concon(grid, grid, derivative="analytic", device="gpu")
    for part, (on_gpu, on_cpu) in enumerate(
        zip(gpu.derivative(np.sqrt(density)), cpu.derivative(np.sqrt(density)), strict=True)
    ):
        assert_close(on_gpu.get(), on_cpu, what=f"the analytic derivative, part {part}")


def test_encore_on_the_gpu_agrees_with_the_cpu(cp):
    grid = encore_grid()
    cohort = densities(grid)
    cpu = Encore(grid, grid, max_iterations=10)
    gpu = Encore(grid, grid, max_iterations=10, device="gpu")
    assert type(gpu.area_product) is np.ndarray
    assert_close(gpu.root(cohort[0]), cpu.root(cohort[0]), what="root")
    template = cpu.template(cohort)
    assert_close(gpu.template(cohort), template, rtol=1e-10, what="template")

    def registered(engine, moving):
        trace = []
        aligned, lh, rh, cost = engine.register(
            template, moving, target_is_root=True, callback=lambda _, c: trace.append(c)
        )
        result = {"trace": trace, "aligned": aligned, "warp": lh.vertices, "Jacobian": rh.jacobian}
        return result, cost

    on_cpu, _ = registered(cpu, cohort[1])
    on_gpu, cost = registered(gpu, cohort[1])
    twins = [registered(cpu, nudged(cohort[1], seed))[0] for seed in (1, 2)]
    assert isinstance(cost, float) and all(isinstance(c, float) for c in on_gpu["trace"])
    assert len(on_gpu["trace"]) == len(on_cpu["trace"]) > 3
    for key, value in on_cpu.items():
        twin_values = [twin[key] for twin in twins]
        floor = FLOORS["Encore.register"][key]
        assert_within_the_cpus_spread(on_gpu[key], value, twin_values, key, floor)


def test_align_on_the_gpu_agrees_with_the_cpu(cp):
    grid = encore_grid()
    cohort = densities(grid, count=4)

    def aligned(densities, device="cpu"):
        result = align(densities, grids=(grid, grid), max_iterations=6, device=device)
        return result, {
            "traces": np.concatenate(result.traces),
            "aligned": np.stack(result.aligned),
            "warps": np.stack([warp.lh_vertices for warp in result.warps]),
        }

    cpu, on_cpu = aligned(cohort)
    gpu, on_gpu = aligned(cohort, "gpu")
    twins = [
        aligned([nudged(d, 10 * seed + i) for i, d in enumerate(cohort)])[1] for seed in (1, 2)
    ]
    assert_close(gpu.template, cpu.template, rtol=1e-10, what="the template")
    assert [len(trace) for trace in gpu.traces] == [len(trace) for trace in cpu.traces]
    for key, value in on_cpu.items():
        twin_values = [twin[key] for twin in twins]
        assert_within_the_cpus_spread(on_gpu[key], value, twin_values, key, FLOORS["align"][key])
    assert all(isinstance(cost, float) for cost in gpu.costs)


def test_a_template_too_big_for_the_gpu_waits_on_the_cpu(cp, monkeypatch):
    """Sent across at each step instead of held, all of them or all but two: the same bits.

    The densities are held while those kept fill less than half the free memory, so with
    room for two the first two stay and the rest wait on the CPU.
    """
    grid = encore_grid()
    cohort = densities(grid)
    engine = Encore(grid, grid, device="gpu")
    held = engine.template(cohort)
    endpoints_grid = conseal_grid()
    connectomes = subjects(endpoints_grid)
    kernel, _ = HeatKernelBuilder(endpoints_grid, endpoints_grid, degree=12).compute(0.05)
    conseal = ConSEAL(endpoints_grid, endpoints_grid, device="gpu")
    held_conseal = conseal.template(connectomes, kernel)

    resident = []
    original = _device.Held.append

    def append(self, array):
        original(self, array)
        resident.append(self._resident[-1])

    monkeypatch.setattr(_device.Held, "append", append)
    monkeypatch.setattr(
        cp, "get_default_memory_pool", lambda: types.SimpleNamespace(free_bytes=lambda: 0)
    )
    for run, again, n in (
        (lambda: engine.template(cohort), held, 2 * grid.n_vertices),
        (
            lambda: conseal.template(connectomes, kernel),
            held_conseal,
            2 * endpoints_grid.n_vertices,
        ),
    ):
        size = n * n * 8
        for free, kept in ((1, 0), (4 * size + 2, 2)):  # room for none, room for two
            monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda free=free: (free, 1 << 34))
            resident.clear()
            np.testing.assert_array_equal(run(), again)
            assert len(resident) > kept and resident == [True] * kept + [False] * (
                len(resident) - kept
            )


@pytest.mark.parametrize("route", ["sparse", "dense"])
def test_endpoint_densities_on_the_gpu_agree_with_the_cpu(cp, monkeypatch, route):
    """Both of evaluate's routes: few streamlines through sparse products, many through dense."""
    import sbci.conseal

    if route == "dense":
        monkeypatch.setattr(sbci.conseal, "SPARSE_PRODUCT_FRACTION", 0.0)
    else:
        monkeypatch.setattr(sbci.conseal, "SPARSE_PRODUCT_FRACTION", np.inf)
    grid = conseal_grid()
    connectome = subjects(grid, count=1)[0]
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    on_gpu = connectome._to_device("gpu")
    sent_kernel, sent_derivative = _sent(kernel, derivative, "gpu")
    q, d_e1, d_e2 = connectome.q_transform(kernel, derivative)
    q_gpu, d_e1_gpu, d_e2_gpu = on_gpu.q_transform(sent_kernel, sent_derivative)
    assert_close(q_gpu.get(), q, what="Q")
    assert_close(d_e1_gpu.get(), d_e1, what="dQ along e1")
    assert_close(d_e2_gpu.get(), d_e2, what="dQ along e2")


@pytest.mark.parametrize("route", ["sparse", "dense"])
def test_the_gpu_repeats_itself(cp, monkeypatch, route):
    """Run twice, the GPU gives the same bits, as the CPU does.

    Two of CuPy's ways add in whatever order its threads land: building a sparse matrix
    whose values share a place, with atomic additions, and cuSPARSE's product of a sparse
    and a dense matrix. The endpoints' adjacency is built in an order every run repeats
    instead (``_device.summed_csr``), and every such product goes through
    ``_device.rows_times_dense``, which adds along each row in turn, the transposes it
    takes stored as matrices of their own (``conseal._transposed``).
    """
    import sbci.conseal

    fraction = np.inf if route == "sparse" else 0.0
    monkeypatch.setattr(sbci.conseal, "SPARSE_PRODUCT_FRACTION", fraction)
    grid = conseal_grid()
    cohort = subjects(grid, count=2)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    on_gpu = cohort[0]._to_device("gpu")
    sent = _sent(kernel, derivative, "gpu")
    for first, again in zip(on_gpu.q_transform(*sent), on_gpu.q_transform(*sent), strict=True):
        np.testing.assert_array_equal(first.get(), again.get())
    engine = ConSEAL(grid, grid, max_iterations=5, device="gpu")
    first, again = (engine.register(cohort[0], cohort[1], kernel, derivative) for _ in range(2))
    np.testing.assert_array_equal(first[2], again[2])
    np.testing.assert_array_equal(first[0].velocity, again[0].velocity)
    np.testing.assert_array_equal(first[3].weights_out, again[3].weights_out)

    sphere = encore_grid()
    cohort = densities(sphere, count=3)
    encore = Encore(sphere, sphere, max_iterations=5, device="gpu")
    template = encore.template(cohort)
    first, again = (encore.register(template, cohort[1], target_is_root=True) for _ in range(2))
    np.testing.assert_array_equal(first[0], again[0])
    assert first[3] == again[3]


def test_relocated_endpoints_on_the_gpu_land_where_the_cpus_do(cp):
    """Through the same warps the endpoints land as on the CPU, bit for bit.

    The points are added in one stated order on both devices (``_points``), and located
    in the CPU's arithmetic (``MeshQuery._query_faces_device``).
    """
    grid = conseal_grid()
    connectome = subjects(grid, count=1)[0]
    lh, rh = StationaryWarp(grid), StationaryWarp(grid)
    rng = np.random.default_rng(4)
    lh.compose(0.05 * rng.normal(size=(grid.n_vertices, 2)))
    rh.rotate(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    on_gpu = connectome._to_device("gpu")
    points = connectome.warp(lh, rh)
    points_gpu = on_gpu.warp(lh, rh)
    for cpu, gpu in zip(points, points_gpu, strict=True):
        np.testing.assert_array_equal(gpu.get(), cpu)
    back = on_gpu._to_device("cpu")
    for name in ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out"):
        np.testing.assert_array_equal(getattr(back, name), getattr(connectome, name), err_msg=name)


def test_conseal_on_the_gpu_agrees_with_the_cpu(cp):
    grid = conseal_grid()
    cohort = subjects(grid)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    for settings in ({}, {"area_weighted": True}):
        cpu = ConSEAL(grid, grid, max_iterations=8, **settings)
        gpu = ConSEAL(grid, grid, max_iterations=8, device="gpu", **settings)
        template = cpu.template(cohort, kernel)
        assert_close(gpu.template(cohort, kernel), template, rtol=1e-10, what="template")
        assert gpu.cost(template - cohort[0].q_transform(kernel)) == pytest.approx(
            cpu.cost(template - cohort[0].q_transform(kernel)), rel=1e-12
        )
        for rotate in (False, True):

            def registered(engine, fixed, rotate=rotate):
                lh, rh, costs, warped = engine.register(
                    fixed, cohort[1], kernel, derivative, init_rotation=rotate
                )
                assert warped.device == "cpu"
                return rh.rigid, {
                    "costs": costs,
                    "velocity": lh.velocity,
                    "endpoints": np.concatenate(warped.positions()),
                }

            rigid, on_cpu = registered(cpu, template)
            rigid_gpu, on_gpu = registered(gpu, template)
            twins = [registered(cpu, nudged(template, seed))[1] for seed in (1, 2)]
            what = f"{settings or 'plain'}, init_rotation={rotate}"
            assert len(on_gpu["costs"]) == len(on_cpu["costs"]) > 1, what
            np.testing.assert_array_equal(rigid_gpu, rigid, err_msg=what)
            for key, value in on_cpu.items():
                twin_values = [twin[key] for twin in twins]
                floor = FLOORS["ConSEAL.register"][key]
                assert_within_the_cpus_spread(
                    on_gpu[key], value, twin_values, f"{what}: {key}", floor
                )


def test_endpoints_align_on_the_gpu_agrees_with_the_cpu(cp):
    """Each kind of template: the template to rounding, the registrations to the CPU's spread.

    A subject registered onto its own density -- subject 1 with ``template=1``, subject 2
    with its own connectome given -- takes no step on either device.
    """
    grid = conseal_grid()
    cohort = subjects(grid)
    settings = dict(sigma=0.05, kernel_degree=12, order=3, max_iterations=4, grids=(grid, grid))
    for template, still in ((None, None), (1, 1), (cohort[2], 2)):
        moved = [i for i in range(len(cohort)) if i != still]

        def compared(result, moved=moved):
            return {
                "costs": np.concatenate([result.costs[i] for i in moved]),
                "endpoints": np.concatenate(
                    [np.concatenate(result.connectomes[i].positions()) for i in moved]
                ),
            }

        cpu = endpoints_align(cohort, template=template, **settings)
        gpu = endpoints_align(cohort, template=template, device="gpu", **settings)
        twins = [
            compared(endpoints_align(cohort, template=nudged(cpu.template, seed), **settings))
            for seed in (1, 2)
        ]
        what = f"template={type(template).__name__}"
        assert_close(gpu.template, cpu.template, rtol=1e-10, what=what)
        assert [len(c) for c in gpu.costs] == [len(c) for c in cpu.costs], what
        on_gpu, on_cpu = compared(gpu), compared(cpu)
        for key, value in on_cpu.items():
            twin_values = [twin[key] for twin in twins]
            floor = FLOORS["endpoints_align"][key]
            assert_within_the_cpus_spread(on_gpu[key], value, twin_values, f"{what}: {key}", floor)
        if still is not None:
            assert list(gpu.costs[still]) == list(cpu.costs[still]) == [0.0], what
        assert all(connectome.device == "cpu" for connectome in gpu.connectomes)
        assert len(gpu.aligned_endpoints(0).surf_in) == cohort[0].n_streamlines
    given = endpoints_align(cohort[:2], template=cp.asarray(cpu.template), device="gpu", **settings)
    np.testing.assert_array_equal(given.template, cpu.template)


# --- the fourth review of 12 October 2026 (PORTING.md item 27) --------------------------


def bundled_left():
    from sbci.spec import N_VERTICES_PER_HEMI
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = np.array(sphere.vertices, dtype=np.float64)[:N_VERTICES_PER_HEMI]
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    return vertices, faces[(faces < N_VERTICES_PER_HEMI).all(axis=1)]


def test_a_point_the_candidates_cannot_settle_is_searched_against_every_face(cp, monkeypatch):
    """With each table row turned to the faces farthest away, every point falls back.

    Only points inside the sphere reached the GPU's search of every face before, where
    the candidates were right anyway: a search that did nothing, or broke ties the other
    way, passed. Here it must give the CPU's faces, at the vertices too, where the faces
    around each tie and the lowest index wins.
    """
    searched = []
    exact = MeshQuery._exact_faces_device

    def counted(points, a, b, c):
        searched.append(int(points.shape[0]))
        return exact(points, a, b, c)

    monkeypatch.setattr(MeshQuery, "_exact_faces_device", staticmethod(counted))
    rng = np.random.default_rng(3)
    for v, f in (bundled_left(), icosphere(1)):
        query = MeshQuery(v, f)
        points = rng.normal(size=(5000, 3))
        located = np.vstack([points / np.linalg.norm(points, axis=1, keepdims=True), v])
        weights, indices, face = query.query_faces(located)
        query.query_faces(cp.asarray(located[:1]))  # the tables, made on first use
        a, b, c, centroids, half_squared, faces, near, reached = query._on_device
        count = near.shape[1]
        _, far = query._tree.query(-centroids.get(), k=count)  # nearest the antipode
        far = np.sort(far.reshape(-1, count), axis=1)
        query._on_device = (a, b, c, centroids, half_squared, faces, cp.asarray(far), reached)
        searched.clear()
        weights_gpu, indices_gpu, face_gpu = (
            part.get() for part in query.query_faces(cp.asarray(located))
        )
        assert sum(searched) == located.shape[0]
        np.testing.assert_array_equal(face_gpu, face)
        np.testing.assert_array_equal(indices_gpu, indices)
        np.testing.assert_array_equal(weights_gpu, weights)


def test_the_point_search_on_the_gpu_is_exact_on_an_irregular_mesh(cp):
    """test_alignment's sphere with a sparse cap: the closest face for every point."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_alignment import capped_sphere, closest_faces_near_the_cap

    vertices, faces, points = capped_sphere()
    _, _, face_gpu = MeshQuery(vertices, faces).query_faces(cp.asarray(points))
    np.testing.assert_array_equal(
        face_gpu.get(), closest_faces_near_the_cap(vertices, faces, points)
    )


def test_the_gpu_takes_endpoint_connectomes_and_says_so(cp):
    """A stand-in with a q_transform, which the CPU takes, met an AttributeError on the GPU."""
    grid = conseal_grid()
    kernel, _ = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)

    class Precomputed:
        def q_transform(self, kernel):
            return np.ones((2 * grid.n_vertices,) * 2)

    with pytest.raises(TypeError, match="EndpointConnectome"):
        ConSEAL(grid, grid, device="gpu").template([Precomputed(), Precomputed()], kernel)
    with pytest.raises(ValueError, match="at least one density"):
        Encore(encore_grid(), encore_grid(), device="gpu").template([])


def test_the_gpus_product_takes_any_real_type_and_refuses_a_misfit(cp):
    """A float32 array met the float64 kernel's one type; a misfit read past its rows."""
    from scipy import sparse

    rng = np.random.default_rng(0)
    matrix = sparse.random(40, 30, density=0.2, random_state=1, format="csr")
    on_gpu = _device.sparse_to_device(matrix, "gpu")
    for dense in (rng.random((30, 5)).astype(np.float32), rng.integers(0, 5, (30, 5))):
        product = _device.rows_times_dense(on_gpu, cp.asarray(dense))
        assert product.dtype == np.float64
        assert_close(product.get(), matrix @ dense, rtol=1e-12, what=str(dense.dtype))
    with pytest.raises(ValueError, match="cannot multiply"):
        _device.rows_times_dense(on_gpu, cp.ones((29, 5)))
    grid = encore_grid()
    density = densities(grid, count=1)[0].astype(np.float32)
    identity = (SphericalWarp(grid), SphericalWarp(grid))
    assert_close(
        Concon(grid, grid, device="gpu").evaluate(density, *identity).get(),
        Concon(grid, grid).evaluate(density, *identity),
        what="Concon on float32",
    )


def test_align_on_the_gpu_takes_cupy_arrays(cp):
    """np.asarray refused them before anything ran."""
    grid = encore_grid()
    cohort = densities(grid, count=3)
    given = align(cohort, grids=(grid, grid), max_iterations=2, device="gpu")
    sent = align(
        [cp.asarray(d) for d in cohort], grids=(grid, grid), max_iterations=2, device="gpu"
    )
    np.testing.assert_array_equal(sent.template, given.template)
    np.testing.assert_array_equal(np.stack(sent.aligned), np.stack(given.aligned))


# --- the GPU's review of 13 October 2026 (PORTING.md item 28) ---------------------------


def test_the_point_search_on_the_gpu_keeps_its_memory_on_a_fine_mesh(cp):
    """On icosphere(6), 81,920 faces, its buffers grew with the faces to some 10 GB.

    The search for each point's nearest centroid took 4,096 points by every centroid, and
    the search of every face 256 points by every face; both now take a fixed budget of
    pairs a pass, and the faces are the CPU's.
    """
    vertices, faces = icosphere(6)
    query = MeshQuery(vertices, faces)
    points = np.random.default_rng(1).normal(size=(20_000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    located = np.vstack([points, 0.5 * points[:300]])  # inside: every face is searched
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    on_gpu = query.query_faces(cp.asarray(located))[2].get()
    held = pool.total_bytes()
    np.testing.assert_array_equal(on_gpu, query.query_faces(located)[2])
    assert held < 2e9, f"{held / 1e9:.1f} GB held"


def test_no_point_of_the_bundled_grids_needs_the_gpus_search_of_every_face(cp, monkeypatch):
    """As test_alignment's for the CPU: the GPU's candidates settle every point there.

    Nothing held the GPU to it, so a search that took the farthest of its candidates, or
    no bound at all, passed while it sent points to every face.
    """
    from sbci.alignment import _hemisphere_grids, normalize_rows

    asked = []
    exact = MeshQuery._exact_faces_device
    monkeypatch.setattr(
        MeshQuery,
        "_exact_faces_device",
        staticmethod(lambda p, a, b, c: asked.append(int(p.shape[0])) or exact(p, a, b, c)),
    )
    points = normalize_rows(np.random.default_rng(0).normal(size=(20000, 3)))
    for rotate in (True, False):  # ENCORE's grids and ConSEAL's
        for grid in _hemisphere_grids(1, rotate=rotate):
            located = cp.asarray(np.vstack([points, grid.vertices]))
            MeshQuery(grid.vertices, grid.faces).query_faces(located)
    assert not asked, asked


@pytest.mark.parametrize("seed", [0, 3, 26, 44])
def test_the_gpu_keeps_a_best_candidate_only_where_no_other_face_can_beat_it(cp, monkeypatch, seed):
    """test_alignment's irregular meshes: every face the closest, its candidates settling some.

    On the capped sphere nearly every point went on to every face; a bound without the
    point's own distance from its nearest centroid passed there and gives farther faces
    here.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_alignment import small_irregular_mesh

    searched = []
    exact = MeshQuery._exact_faces_device
    monkeypatch.setattr(
        MeshQuery,
        "_exact_faces_device",
        staticmethod(lambda p, a, b, c: searched.append(int(p.shape[0])) or exact(p, a, b, c)),
    )
    vertices, faces, points = small_irregular_mesh(seed)
    query = MeshQuery(vertices, faces)
    _, _, face = query.query_faces(cp.asarray(points))
    assert sum(searched) < len(points)  # the candidates settled some
    np.testing.assert_array_equal(face.get(), query._exact_faces(points))


def test_what_waits_on_the_cpu_leaves_the_gpu(cp, monkeypatch):
    """With room for two, the rest are kept as NumPy arrays and come back as CuPy ones.

    The template's test counted where each was put, not what was kept, so one kept on
    the GPU while marked as waiting passed; so did room that left out the pool's free
    memory, which CuPy keeps for its next arrays.
    """
    size = 100 * 100 * 8
    for free, pooled in ((4 * size + 2, 0), (2, 4 * size)):  # the room in either
        monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda free=free: (free, 1 << 34))
        monkeypatch.setattr(
            cp,
            "get_default_memory_pool",
            lambda pooled=pooled: types.SimpleNamespace(free_bytes=lambda: pooled),
        )
        held = _device.Held("gpu")
        for _ in range(4):
            held.append(cp.ones((100, 100)))
        assert [_device.on_device(array) for array in held._arrays] == [True, True, False, False]
        assert all(_device.on_device(held[i]) for i in range(4))


def test_the_gpus_product_takes_a_float32_matrix(cp):
    """A float32 matrix, as a kernel kept in single precision, by float64 numbers: float64."""
    from scipy import sparse

    rng = np.random.default_rng(0)
    matrix = sparse.random(40, 30, density=0.2, random_state=1, format="csr", dtype=np.float32)
    dense = rng.random((30, 5))
    product = _device.rows_times_dense(_device.sparse_to_device(matrix, "gpu"), cp.asarray(dense))
    assert product.dtype == np.float64
    assert_close(product.get(), matrix @ dense, rtol=1e-12, what="float32 by float64")


def test_the_gpus_adjacency_has_one_entry_a_place(cp):
    """Each place's values added into one entry, as SciPy's: the same matrix, in order.

    A sum that marked every entry the first of its place kept them all, and the densities
    came out the same, the extra entries adding up in the products.
    """
    connectome = subjects(conseal_grid(), count=1)[0]
    on_gpu = connectome._to_device("gpu")._sparse_adjacency().get()
    on_cpu = connectome._sparse_adjacency()
    assert on_gpu.has_canonical_format  # sorted along each row, no place twice
    assert_close(on_gpu.toarray(), on_cpu.toarray(), what="A + A^T")


def test_align_on_the_gpu_takes_a_cupy_template(cp):
    """A template given as a CuPy array registers as the same one given as NumPy, bit for bit."""
    grid = encore_grid()
    cohort = densities(grid, count=3)
    template = Encore(grid, grid).template(cohort)
    settings = dict(grids=(grid, grid), max_iterations=2, device="gpu")
    given = align(cohort[:2], template=template, **settings)
    sent = align(cohort[:2], template=cp.asarray(template), **settings)
    np.testing.assert_array_equal(np.stack(sent.aligned), np.stack(given.aligned))
    np.testing.assert_array_equal(sent.template, given.template)


def gpu_tables(connectome):
    """The CuPy arrays a connectome's point searches hold."""
    return [table for query in connectome._query.values() for table in query._on_device or ()]


def test_a_gpu_alignment_leaves_nothing_on_the_gpu(cp):
    """Each subject's point searches kept the tables the GPU made for them, 4 MB a subject.

    Every copy of a connectome shared its searches, so the copy registered on the GPU
    made them on the subject's own, and they lived as long as the subject or the
    connectome handed back: on ico4, 46 MB for the eleven and 3.8 GB for 946. Each copy
    on the GPU now has searches of its own.
    """
    import gc

    grid = conseal_grid()
    cohort = subjects(grid)
    pool = cp.get_default_memory_pool()
    gc.collect()
    before = pool.used_bytes()
    settings = dict(sigma=0.05, kernel_degree=12, order=3, max_iterations=2, grids=(grid, grid))
    result = endpoints_align(cohort, device="gpu", **settings)
    for connectome in (*cohort, *result.connectomes):
        assert connectome.device == "cpu" and not gpu_tables(connectome)
    gc.collect()
    assert pool.used_bytes() == before, f"{pool.used_bytes() - before} bytes left on the GPU"


def test_a_registration_on_the_gpu_leaves_a_gpu_connectome_as_it_was(cp):
    """``register`` copies its moving connectome, on whichever device it was given."""
    grid = conseal_grid()
    fixed, moving = subjects(grid, count=2)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    on_gpu = moving._to_device("gpu")
    names = ("weights_in", "index_in", "face_in", "weights_out", "index_out", "face_out")
    before = [getattr(on_gpu, name).get() for name in names]
    ConSEAL(grid, grid, max_iterations=3, device="gpu").register(fixed, on_gpu, kernel, derivative)
    for name, was in zip(names, before, strict=True):
        np.testing.assert_array_equal(getattr(on_gpu, name).get(), was, err_msg=name)


def test_a_cpu_engine_takes_a_connectome_kernel_and_derivative_on_the_gpu(cp):
    """Each is fetched to the CPU, and the registration is the CPU's on them, bit for bit.

    A connectome on the GPU was registered there and handed back there, or met one on the
    CPU and CuPy refused the mix; a derivative on the GPU met the CPU's diagonals; and a
    dense kernel on the GPU, on either engine, had no ``tocsr``.
    """
    from sbci.conseal import KernelDerivative

    grid = conseal_grid()
    fixed, moving = subjects(grid, count=2)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    sent_kernel, sent_derivative = _sent(kernel, derivative, "gpu")
    fetched = KernelDerivative(
        *(part.get() for part in (sent_derivative.x, sent_derivative.y, sent_derivative.z))
    )
    dense = kernel.toarray()
    engine = ConSEAL(grid, grid, max_iterations=2)
    for given, expected in (
        (
            (fixed._to_device("gpu"), moving._to_device("gpu"), kernel, derivative),
            (fixed, moving, kernel, derivative),
        ),
        (
            (fixed, moving._to_device("gpu"), kernel, derivative),
            (fixed, moving, kernel, derivative),
        ),
        (
            (fixed, moving, sent_kernel, sent_derivative),
            (fixed, moving, sent_kernel.get(), fetched),
        ),
        ((fixed, moving, cp.asarray(dense), derivative), (fixed, moving, dense, derivative)),
    ):
        _, _, costs, warped = engine.register(*given)
        _, _, costs_cpu, warped_cpu = engine.register(*expected)
        assert warped.device == "cpu"
        np.testing.assert_array_equal(costs, costs_cpu)
        np.testing.assert_array_equal(warped.weights_out, warped_cpu.weights_out)
    gpu = ConSEAL(grid, grid, max_iterations=2, device="gpu")
    on_gpu = gpu.register(fixed, moving, cp.asarray(dense), derivative)[2]
    np.testing.assert_array_equal(on_gpu, gpu.register(fixed, moving, dense, derivative)[2])


def test_a_stand_in_moving_connectome_is_refused_plainly_on_the_gpu(cp):
    """The CPU registers anything with a connectome's methods; the GPU met an AttributeError."""
    grid = conseal_grid()
    fixed, moving = subjects(grid, count=2)
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)

    class StandIn:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "_to_device":
                raise AttributeError(name)
            return getattr(self._inner, name)

        def copy(self):
            return StandIn(self._inner.copy())

    ConSEAL(grid, grid, max_iterations=1).register(fixed, StandIn(moving), kernel, derivative)
    with pytest.raises(TypeError, match="EndpointConnectome"):
        ConSEAL(grid, grid, max_iterations=1, device="gpu").register(
            fixed, StandIn(moving), kernel, derivative
        )


def test_a_connectome_without_streamlines_is_refused_on_the_gpu_as_on_the_cpu(cp):
    """Its density is zero on the GPU too, and refused as on the CPU, with nothing to add up."""
    grid = conseal_grid()
    full = subjects(grid, count=1)[0]
    empty = EndpointConnectome.from_points(grid, grid, np.empty((0, 3)), np.empty((0, 3)), [], [])
    kernel, derivative = HeatKernelBuilder(grid, grid, degree=12).compute(0.05)
    density = empty._to_device("gpu").q_transform(_sent(kernel, None, "gpu")[0])
    assert density.shape == (2 * grid.n_vertices,) * 2 and not density.any()
    for device in ("cpu", "gpu"):
        engine = ConSEAL(grid, grid, max_iterations=1, device=device)
        with pytest.raises(ValueError, match="subject 1 has no streamlines"):
            engine.template([full, empty], kernel)
        with pytest.raises(ValueError, match="the fixed connectome has no streamlines"):
            engine.register(empty, full, kernel, derivative)


def test_the_analytic_derivative_on_the_gpu_repeats_itself(cp):
    """Run twice, the same bits, as the central difference gives; and the CPU's to rounding."""
    sphere = encore_grid()
    cohort = densities(sphere, count=2)
    engine = Encore(sphere, sphere, max_iterations=5, derivative="analytic", device="gpu")
    first, again = (engine.register(cohort[0], cohort[1]) for _ in range(2))
    np.testing.assert_array_equal(first[0], again[0])
    assert first[3] == again[3]
    values = cp.asarray(cohort[0])
    twice = zip(engine.concon.derivative(values), engine.concon.derivative(values), strict=True)
    for one, two in twice:
        np.testing.assert_array_equal(one.get(), two.get())
    on_cpu = Encore(sphere, sphere, derivative="analytic").concon.derivative(cohort[0])
    for one, cpu in zip(engine.concon.derivative(values), on_cpu, strict=True):
        assert_close(one.get(), cpu, rtol=1e-13, what="the analytic derivative")


def test_the_rotation_search_on_the_gpu_pulls_back_as_the_cpu_does(cp):
    """The search ran on the CPU: on two released subjects, 200 s of a registration's 204.

    On the GPU each block it pulls back through a rotation is the CPU's to rounding, its
    points located bit for bit; the rotations it picks are the CPU's, bit for bit
    (``test_conseal_on_the_gpu_agrees_with_the_cpu``).
    """
    from scipy.spatial.transform import Rotation

    grid = conseal_grid()
    query = MeshQuery(grid.vertices, grid.faces)
    values = np.random.default_rng(3).random((grid.n_vertices, grid.n_vertices))
    for rotation in Rotation.random(5, random_state=2).as_matrix():
        assert_close(
            ConSEAL._pull_back(cp.asarray(values), grid, query, rotation).get(),
            ConSEAL._pull_back(values, grid, query, rotation),
            rtol=1e-14,
            what="a block pulled back",
        )
