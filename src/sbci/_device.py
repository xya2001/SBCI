"""Where the alignments compute: on the CPU through NumPy and SciPy, or on a CUDA GPU through CuPy.

``align(..., device="gpu")`` and ``endpoints_align(..., device="gpu")`` keep their large arrays
-- the densities over pairs of vertices, and ConSEAL's streamline endpoints -- on the GPU, and
the per-vertex geometry of the warps on the CPU, where it is small. CuPy is optional (``pip
install 'sbci[gpu]'``); without it, or with no GPU visible, ``device="gpu"`` is refused with what
is missing. Each step agrees with the CPU's to rounding, since the GPU adds in another order,
and the endpoints' relocation bit for bit (both add a point's corners in one stated order); a
whole registration agrees as closely as the CPU's own does when an input moves by one unit in
the last place, and two runs on the GPU give the same bits (PORTING.md items 26 and 27). The
CPU computes as it did before the GPU came, bit for bit on x86.
"""

from __future__ import annotations

import numpy as np

DEVICES = ("cpu", "gpu")

_MISSING = (
    "device='gpu' needs CuPy built for your CUDA version, and a CUDA GPU. Install it with:\n"
    "    pip install 'sbci[gpu]'          # CuPy for CUDA 12\n"
    "or pick the CuPy package for your CUDA from https://docs.cupy.dev/en/stable/install.html"
)


def check(device) -> str:
    """``device`` itself, once it is known and usable here."""
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if device == "gpu":
        cupy()
    return device


_USABLE = None  # the CuPy module found able to compute, so each call after the first is quick


def cupy():
    """The CuPy module, once it imports, sees a GPU and compiles a kernel for it."""
    global _USABLE
    try:
        import cupy as module
    except ImportError as exc:
        raise ImportError(_MISSING) from exc
    if module is _USABLE:
        return module
    try:
        count = module.cuda.runtime.getDeviceCount()
    except Exception as exc:  # CUDARuntimeError, with no driver or no device
        raise RuntimeError(
            f"CuPy is installed but sees no CUDA GPU ({exc}); run on a GPU node, or use "
            "device='cpu'"
        ) from exc
    if count < 1:
        raise RuntimeError("CuPy is installed but sees no CUDA GPU; use device='cpu'")
    # CuPy compiles a kernel when it is first used, which needs CUDA's own libraries, and runs
    # it on a GPU that may be busy: either way the device was found and the run failed later,
    # at its first kernel
    try:
        x = module.arange(2.0)
        compiled = module.ElementwiseKernel("T x", "T y", "y = x + 1", "sbci_compiles")
        compiled(x, module.empty_like(x))
    except Exception as exc:
        first = (str(exc).splitlines() or [""])[0][:200]
        raise RuntimeError(
            f"CuPy sees a GPU but cannot compile or run a kernel on it ({type(exc).__name__}: "
            f"{first}). Without CUDA's libraries, load the CUDA toolkit (on Longleaf, module "
            "load cuda/12.6) or install the libraries CuPy's guide names, "
            "https://docs.cupy.dev/en/stable/install.html; with a GPU that is busy or in another "
            "process's exclusive use, run on another or use device='cpu'"
        ) from exc
    _USABLE = module
    return module


def on_device(array) -> bool:
    """Whether ``array`` is a CuPy array (or sparse matrix), without importing CuPy."""
    return type(array).__module__.split(".")[0] in ("cupy", "cupyx")


def module_of(array):
    """NumPy or CuPy, whichever ``array`` belongs to."""
    return cupy() if on_device(array) else np


def array_module(device):
    """NumPy for the CPU, CuPy for the GPU."""
    return np if device == "cpu" else cupy()


def sparse_module(device):
    """``scipy.sparse`` for the CPU, ``cupyx.scipy.sparse`` for the GPU."""
    if device == "cpu":
        from scipy import sparse as module
    else:
        cupy()
        import cupyx.scipy.sparse as module
    return module


def to_device(array, device, dtype=None):
    """``array`` on ``device``: a NumPy array for the CPU, a CuPy array for the GPU."""
    if device == "cpu":
        return to_host(array) if dtype is None else to_host(array).astype(dtype, copy=False)
    return cupy().asarray(array, dtype=dtype)


def to_host(array):
    """A NumPy array, from a CuPy array or anything NumPy reads."""
    if on_device(array):
        return array.get()
    return np.asarray(array)


def copy_to(array, device):
    """An independent copy of ``array`` on ``device``."""
    if device == "cpu":
        return np.array(to_host(array))
    return cupy().array(array)


def sparse_to_device(matrix, device):
    """A SciPy sparse matrix as CSR on ``device``; on the CPU, unchanged."""
    if device == "cpu":
        return matrix
    if on_device(matrix):
        return matrix
    return sparse_module("gpu").csr_matrix(matrix)


_SEGMENT_SUM = None
_ROWS_TIMES_DENSE = None


def rows_times_dense(matrix, dense):
    """A CSR matrix on the GPU times a dense one, each entry's products added in row order.

    cuSPARSE's product, as CuPy calls it, shares a row's work among threads whose
    partial sums meet in an order that changes from run to run, and with it the last
    bits; here each entry adds its row's products one after another, so every run
    gives the same bits. The result is C-ordered.
    """
    global _ROWS_TIMES_DENSE
    module = cupy()
    if _ROWS_TIMES_DENSE is None:
        _ROWS_TIMES_DENSE = module.ElementwiseKernel(
            "raw T data, raw I indices, raw I indptr, raw T dense, int64 width",
            "T out",
            "const long long row = i / width, col = i % width; T sum = 0;"
            " for (I p = indptr[row]; p < indptr[row + 1]; p++)"
            " sum += data[p] * dense[indices[p] * width + col];"
            " out = sum;",
            "sbci_rows_times_dense",
        )
    if dense.ndim != 2 or matrix.shape[1] != dense.shape[0]:
        raise ValueError(f"a {matrix.shape} matrix cannot multiply a {dense.shape} array")
    # one type for both, as SciPy's product gives: a float32 or integer array meets the
    # float64 kernel as float64, where the kernel's one type would refuse it
    dtype = module.result_type(matrix.dtype, dense.dtype)
    if matrix.dtype != dtype:
        matrix = matrix.astype(dtype)
    dense = module.ascontiguousarray(dense, dtype=dtype)
    out = module.empty((matrix.shape[0], dense.shape[1]), dtype=dtype)
    _ROWS_TIMES_DENSE(matrix.data, matrix.indices, matrix.indptr, dense, dense.shape[1], out)
    return out


def summed_csr(rows, cols, values, shape):
    """A CSR matrix on the GPU of ``values`` at ``(rows, cols)``, a place's values added in order.

    CuPy's ``coo_matrix(...).tocsr()`` adds the values that share a place with atomic
    operations, in whatever order its threads land, so the same entries gave a matrix that
    differed in its last bits from one run to the next. Here the entries are sorted by
    place, stably, and each place's values added one after another in the order given:
    every run gives the same bits.
    """
    global _SEGMENT_SUM
    module = cupy()
    from cupyx.scipy import sparse

    if _SEGMENT_SUM is None:
        _SEGMENT_SUM = module.ElementwiseKernel(
            "raw T values, raw int64 bounds",
            "T total",
            "T sum = 0; for (long long j = bounds[i]; j < bounds[i + 1]; j++) sum += values[j];"
            " total = sum;",
            "sbci_segment_sum",
        )
    n_rows, n_cols = shape
    places = rows.astype(module.int64) * n_cols + cols.astype(module.int64)
    order = module.argsort(places)  # stable, so a place keeps its values' order
    places, values = places[order], values[order]
    first = module.empty(places.size, dtype=bool)
    first[:1] = True
    first[1:] = places[1:] != places[:-1]
    starts = module.flatnonzero(first)
    bounds = module.concatenate([starts, module.asarray([places.size])]).astype(module.int64)
    total = module.empty(starts.size, dtype=values.dtype)
    _SEGMENT_SUM(values, bounds, total)
    places = places[starts]
    indptr = module.zeros(n_rows + 1, dtype=module.int64)
    module.cumsum(module.bincount(places // n_cols, minlength=n_rows), out=indptr[1:])
    index = module.int32 if max(n_cols, places.size) < 2**31 else module.int64
    return sparse.csr_matrix(
        (total, (places % n_cols).astype(index), indptr.astype(index)), shape=shape
    )


class Held:
    """Arrays kept for use again and again: on the GPU while they fit, the rest on the CPU.

    On the GPU each stays there while those kept so far fill less than half the memory
    free when the first came; the rest wait on the CPU and are sent across whenever one
    is read, a transfer each time rather than a limit on how many subjects a template
    takes. Nothing is counted beforehand, so a generator of subjects is read one at a
    time. On the CPU they are simply kept.
    """

    def __init__(self, device):
        self.device = device
        self._arrays = []
        self._resident = []
        self._room = None  # the bytes the GPU may still take, set when the first comes

    def append(self, array) -> None:
        resident = True
        if self.device == "gpu":
            if self._room is None:
                module = cupy()
                free = module.cuda.runtime.memGetInfo()[0]
                free += module.get_default_memory_pool().free_bytes()
                self._room = 0.5 * free
            resident = array.nbytes < self._room
            if resident:
                self._room -= array.nbytes
        self._arrays.append(array if resident else to_host(array))
        self._resident.append(resident)

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, index):
        array = self._arrays[index]
        return array if self._resident[index] else to_device(array, self.device)


def synchronize(device) -> None:
    """Wait for the GPU's queued work, so that a timing means what it says."""
    if device == "gpu":
        cupy().cuda.Device().synchronize()
