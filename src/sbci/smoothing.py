"""Kernel smoothing on the cortical surface.

Port of ``SBCI_Toolkit/concon_estimate/compute_diffusion_kernel_matrix.m``,
``compute_matern_kernel_matrix.m`` and the driver
``rdk_smoothed_concon_compute.m``.

The method
----------
Streamline endpoints give a discrete count matrix ``A`` on the surface mesh.
Smoothing it into a continuous density is a congruence, ``K A K``, where ``K``
is a kernel built from the Laplace-Beltrami eigenpairs of the hemisphere::

    K = U diag(rho(Lambda)) U'

``rho`` is the only thing that differs between the two kernels: the Riemannian
diffusion kernel uses ``exp(-kappa^2 Lambda / 2)`` and the Matern kernel uses
``(2 nu / kappa^2 + Lambda)^(-nu-1)``. ``K`` is symmetric by construction, so
``K A K`` and ``K A K'`` are the same matrix.

Written this way the whole smoothing step is a handful of dense matrix
products, which is where the WP3 speed target comes from -- under a minute per
subject against the 2-8 hours the original pipeline takes.

Two properties of the reference implementation are easy to get wrong and are
reproduced deliberately:

*Eigenvectors are not Euclidean-orthonormal.* ``U' U`` differs from the
identity by as much as 0.12 on the shipped ico4 eigenpairs, because
Laplace-Beltrami eigenfunctions are orthonormal against the surface mass
matrix, not against the dot product. ``K`` therefore does not tend to the
identity as the bandwidth shrinks, and no step here assumes it does.

*Spectral truncation makes small negative entries.* A truncated eigenbasis
cannot represent a nonnegative function exactly, so ``K A K`` carries small
negative values. The reference sums only the positive entries to form the
normalizing constant and then clips the negatives to zero, and so does this.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .errors import FormatError, MissingDataError

KERNELS = ("shk", "rdk", "matern")
"""The kernel vocabulary, following the decision recorded as SPEC_QUESTIONS.md
item 10.

``shk``
    The spherical kernel ``concon`` applies (Moyer et al., MICCAI 2016), the
    default, and the one that produced every released ``smoothed_sc_avg_*.mat``.
    Its bandwidth is ``sigma``. Reproduces ``c3_main`` at r = 1.000000 across
    five ADNI subjects, with a 0.14% amplitude offset that is a convention
    rather than noise -- see PORTING.md item 6. Not the heat kernel, despite
    the name: :func:`spherical_heat_kernel` explains the weight.
``rdk``
    Riemannian diffusion kernel (bioRxiv 2025.09.08.674789), built from
    Laplace-Beltrami eigenfunctions on the cortical surface itself. Bandwidth
    ``kappa``. Ported and verified -- :func:`diffusion_kernel`.
``matern``
    Matern kernel on the same eigenbasis, from
    ``compute_matern_kernel_matrix.m``. Bandwidth ``kappa`` plus a smoothness
    ``nu``. Ported and verified -- :func:`matern_kernel`.

``bandwidth`` therefore means different things for different kernels, and has
to be read together with ``kernel`` in the metadata.
"""

TRUNCATION_FLOOR = 1e-30
"""Kernel eigenvalues below this are set to zero, as in the MATLAB."""


def diffusion_kernel(eigenvalues: np.ndarray, eigenvectors: np.ndarray, kappa: float) -> np.ndarray:
    """Riemannian diffusion kernel ``U diag(exp(-kappa^2 Lambda / 2)) U'``.

    Parameters
    ----------
    eigenvalues
        Laplace-Beltrami eigenvalues, ascending, length ``maxk``.
    eigenvectors
        ``(n_vertices, maxk)`` eigenfunctions, one per column.
    kappa
        Bandwidth. Larger values smooth more.

    Examples
    --------
    >>> import numpy as np
    >>> U = np.eye(3)
    >>> K = diffusion_kernel(np.array([0.0, 1.0, 2.0]), U, kappa=1.0)
    >>> np.allclose(K, np.diag(np.exp(-0.5 * np.array([0.0, 1.0, 2.0]))))
    True
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    rho = np.exp(-(kappa**2) / 2.0 * np.asarray(eigenvalues, dtype=np.float64).ravel())
    return _assemble(rho, eigenvectors)


def matern_kernel(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, kappa: float, nu: float = 3.0
) -> np.ndarray:
    """Matern kernel ``U diag((2 nu / kappa^2 + Lambda)^(-nu-1)) U'``.

    Parameters
    ----------
    nu
        Differentiability. The reference script uses 1, 2 or 3.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    lam = np.asarray(eigenvalues, dtype=np.float64).ravel()
    rho = (2.0 * nu / kappa**2 + lam) ** (-nu - 1.0)
    return _assemble(rho, eigenvectors)


def _assemble(rho: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """``U diag(rho) U'``, with the reference's truncation floor applied."""
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    if eigenvectors.ndim != 2:
        raise ValueError(f"eigenvectors must be 2-D, got shape {eigenvectors.shape}")
    if rho.size != eigenvectors.shape[1]:
        raise ValueError(f"{rho.size} eigenvalues for {eigenvectors.shape[1]} eigenvectors")
    rho = np.where(rho < TRUNCATION_FLOOR, 0.0, rho)
    return (eigenvectors * rho[np.newaxis, :]) @ eigenvectors.T


NUM_HARMONICS = 33
"""Terms ``concon`` sums, so degrees ``l = 0 .. 32``.

The count is hardcoded and ``--OPT_VAL_num_harm`` is **ignored**: ``c3_main``
prints "num_harmonics now global constant (for speed)" at startup, and passing
it 9, 17, 25, 33, 49 or 65 gives a bit-identical kernel. The pipeline's
``--OPT_VAL_num_harm 33`` therefore changes nothing, though it happens to name
the value actually used.

Fixed at 33 by measurement rather than by reading: across bandwidths 0.00125
to 0.02, 33 terms reproduces the binary's peak to 0.06-0.1% and its shape to
rms 0.0002, where 32 terms drifts to 4% at the narrowest bandwidth and 34
terms overshoots. The truncation matters -- at sigma 0.005 the last term still
carries 0.37% of the peak -- so a port that summed further would stop
reproducing the released cohorts.
"""

DEFAULT_SIGMA = 0.005
"""Bandwidth the released cohorts use, from ``--sigma 0.005``."""

FINAL_THRESHOLD = 1e-9
"""``--final_thold 0.000000001``: after dividing by the streamline count,
``compute_kernel.cpp`` writes a pair only ``if(temp > final_thold)``. Applied
by :func:`smooth` on the same scale, before the unit-mass normalization."""

KERNEL_EPSILON = 0.001
"""Where ``concon`` cuts the kernel off, from ``--epsilon``.

``compute_kernel.cpp`` scans ``x = cos(theta)`` down from 1.0 in steps of
1e-4 and stops at the first ``x`` where the kernel falls below this; vertices
beyond that angle get nothing. The kernel's descent there is steep enough that
anything from 1e-6 to 0.1 gives the same cutoff to three decimals, which is why
sweeping the flag against the binary appeared to do nothing.
"""


@lru_cache(maxsize=32)
def kernel_cutoff(
    sigma: float, epsilon: float = KERNEL_EPSILON, harmonics: int = NUM_HARMONICS
) -> float:
    """Angle in radians beyond which the kernel is zero.

    Reproduces ``compute_kernel.cpp``'s scan exactly, including its 1e-4 step
    in the cosine, so the cutoff lands on the same grid vertices.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.degrees(kernel_cutoff(0.005))), 3)
    12.231
    >>> round(float(np.degrees(kernel_cutoff(0.01))), 3)
    17.369
    """
    cosines = np.arange(1.0, -1.0, -0.0001)
    values = _legendre_series(cosines, sigma, harmonics)
    below = np.nonzero(values < epsilon)[0]
    return float(np.arccos(np.clip(cosines[below[0]], -1.0, 1.0))) if below.size else np.pi


def _legendre_series(cosine: np.ndarray, sigma: float, harmonics: int) -> np.ndarray:
    """The untruncated sum, shared by the kernel and its cutoff scan.

    The recurrence runs in place. Smoothing one subject evaluates this over a
    ``(5124, 4096)`` block a few hundred times, and allocating a fresh array
    per term costs 168 MB each time; reusing three buffers instead made
    ``smooth(kernel="shk")`` about three times faster end to end.
    """
    degree = np.arange(harmonics)
    weight = (2 * degree + 1) ** 1.5 / np.sqrt(4 * np.pi) * np.exp(-degree * (degree + 1) * sigma)

    # Work on a flat view: a scalar argument becomes a 0-d array, which numpy
    # refuses as an `out=` target, and the recurrence writes through `out`.
    shape = np.shape(cosine)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)

    previous = np.ones_like(cosine)
    total = weight[0] * previous
    if harmonics == 1:
        return total.reshape(shape)

    current = cosine.copy()
    term = np.empty_like(cosine)
    np.multiply(current, weight[1], out=term)
    total += term

    scratch = np.empty_like(cosine)
    for order in range(1, harmonics - 1):
        # scratch <- ((2k+1) cos * current - k * previous) / (k + 1)
        np.multiply(cosine, current, out=scratch)
        scratch *= 2 * order + 1
        np.multiply(previous, float(order), out=term)
        scratch -= term
        scratch /= order + 1
        previous, current, scratch = current, scratch, previous
        np.multiply(current, weight[order + 1], out=term)
        total += term
    return total.reshape(shape)


HARMONIC_SAMPLES_EXPONENT = 5
"""``--OPT_VAL_exp_num_harm_samps 5``: ``concon`` tabulates each spherical
harmonic at ``10**5`` cosine samples per unit (``subject.cpp``:
``num_harm_samps = pow(10, exp_num_harm_samps)``)."""

KERNEL_SAMPLES_EXPONENT = 6
"""``--OPT_VAL_exp_num_kern_samps 6``: the kernel itself is tabulated at
``10**6`` cosine samples per unit and read back by integer truncation."""


@dataclass(frozen=True)
class KernelTable:
    """``concon``'s kernel exactly as ``c3_main`` evaluates it: a lookup table.

    ``compute_kernel.cpp`` never evaluates the series at a point. It reads
    ``kern_lookup_table[(int)(dot * M + M)]``, a table of ``2M + 1`` values over
    the cosine in ``[-1, 1]``, and that table was itself filled from a harmonic
    table read the same way, ``harm_lookup_table[l][(int)(x * L + L)]``. Both
    reads truncate, so every lookup lands on the sample at or below the true
    cosine -- a slightly larger angle, a slightly smaller kernel -- and the
    bias grows with ``l(l+1)``. That is the 0.14% amplitude offset the exact
    series left against the released files, and reproducing the tables removes
    it: on a single streamline the binary's peak entry is
    ``K(1.0) * K(1 - 1e-12) = 270.2668 * 269.9403``, whose square root is the
    measured 270.1035, at all five bandwidths tested.

    Attributes
    ----------
    values
        The ``2M + 1`` tabulated kernel values.
    samples
        ``M``.
    cutoff_cosine
        Vertices with ``dot < cutoff_cosine`` receive nothing, found by the
        binary's own scan: ``x`` from 1.0 down in steps of 1e-4 until
        ``K(x) * K(1) < epsilon``.
    """

    values: np.ndarray
    samples: int
    cutoff_cosine: float

    @property
    def cutoff(self) -> float:
        """Cutoff angle in radians."""
        return float(np.arccos(np.clip(self.cutoff_cosine, -1.0, 1.0)))

    def __call__(self, cosine) -> np.ndarray:
        """Kernel values by truncation lookup, zero beyond the cutoff."""
        # No `out=` anywhere here: a scalar argument is a 0-d array, which
        # numpy refuses as an output buffer.
        cosine = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
        index = np.clip(
            (cosine * self.samples + self.samples).astype(np.int64), 0, 2 * self.samples
        )
        return np.where(cosine < self.cutoff_cosine, 0.0, self.values[index])


@lru_cache(maxsize=8)
def concon_kernel_table(
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    epsilon: float = KERNEL_EPSILON,
    harmonic_exponent: int = HARMONIC_SAMPLES_EXPONENT,
    kernel_exponent: int = KERNEL_SAMPLES_EXPONENT,
) -> KernelTable:
    """Build the two lookup tables ``c3_main`` builds, and the cutoff it scans.

    Examples
    --------
    >>> table = concon_kernel_table(0.005)
    >>> round(float(np.sqrt(table(1.0) * table(1.0 - 1e-12))), 4)   # the binary's peak entry
    270.1035
    >>> round(float(np.degrees(table.cutoff)), 3)
    12.231
    """
    degree = np.arange(harmonics)
    big_l = 10**harmonic_exponent
    big_m = 10**kernel_exponent

    # harm_lookup_table[l][j + L] = EvalSH(l, 0, 0, acos(j / L)): the normalized
    # harmonic Y_l^0, which carries sqrt((2l+1)/4pi).
    x_h = np.arange(-big_l, big_l + 1, dtype=np.float64) / big_l
    harmonic = (
        _legendre_polynomials(x_h, harmonics) * np.sqrt((2 * degree + 1) / (4 * np.pi))[:, None]
    )

    # kern_lookup_table[i + M] = sum_l exp(-sigma l(l+1)) (2l+1) harm[l][(int)(x_i L + L)]
    x_k = np.arange(-big_m, big_m + 1, dtype=np.float64) / big_m
    index = (x_k * big_l + big_l).astype(np.int64)
    np.clip(index, 0, 2 * big_l, out=index)
    weight = np.exp(-sigma * degree * (degree + 1)) * (2.0 * degree + 1.0)
    values = np.zeros(x_k.size, dtype=np.float64)
    for order in range(harmonics):
        values += weight[order] * harmonic[order][index]
    np.maximum(values, 1e-20, out=values)  # fmax(ret_val, 1e-20)

    # The cutoff scan in compute_kernel.cpp, on the table.
    scan = 1.0 - 1e-4 * np.arange(0, int(2.0 / 1e-4) + 1)
    scan_index = np.clip((scan * big_m + big_m).astype(np.int64), 0, 2 * big_m)
    product = values[scan_index] * values[2 * big_m]
    below = np.nonzero(product < epsilon)[0]
    cutoff_cosine = float(scan[below[0]]) if below.size else -1.0
    return KernelTable(values=values, samples=big_m, cutoff_cosine=cutoff_cosine)


def _legendre_polynomials(x: np.ndarray, terms: int) -> np.ndarray:
    """``P_l(x)`` for ``l = 0 .. terms-1`` as rows, by the standard recurrence."""
    out = np.empty((terms, x.size), dtype=np.float64)
    out[0] = 1.0
    if terms > 1:
        out[1] = x
    for k in range(1, terms - 1):
        out[k + 1] = ((2 * k + 1) * x * out[k] - k * out[k - 1]) / (k + 1)
    return out


def spherical_heat_kernel(
    cosine,
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    epsilon: float | None = KERNEL_EPSILON,
    quantized: bool = True,
):
    r"""The kernel ``concon`` applies, on the unit sphere.

    .. math::
        K_\sigma(\cos\gamma) = \sum_{l=0}^{32}
            \frac{(2l+1)^{3/2}}{\sqrt{4\pi}}\, e^{-l(l+1)\sigma}\, P_l(\cos\gamma)

    zero beyond the cutoff, and -- by default -- read through the same two
    truncation-indexed lookup tables ``c3_main`` reads it through
    (:class:`KernelTable`), which is what the released cohorts carry.

    **This is not the heat kernel**, whose weight is :math:`(2l+1)/4\pi`.
    ``sigma_opt.cpp`` sums ``exp(-sigma*l*(l+1)) * (2l+1) * harm_lookup[l]``,
    and ``subject.cpp`` fills that lookup with ``sh::EvalSH(l, 0, 0, acos(x))``
    -- the *normalized* spherical harmonic :math:`Y_l^0`, which already carries
    :math:`\sqrt{(2l+1)/4\pi}`. The two factors compound into
    :math:`(2l+1)^{3/2}`, which looks like a Legendre polynomial was intended
    where a normalized harmonic was used. Intended or not, it is what produced
    every released cohort, so it is what this package reproduces. The extra
    :math:`\sqrt{2l+1}` upweights high degrees and makes the kernel markedly
    narrower than a true heat kernel.

    Parameters
    ----------
    cosine
        Cosines of the angle between points, any shape. Clipped to ``[-1, 1]``.
    sigma
        Bandwidth. The released cohorts use 0.005.
    harmonics
        Number of terms. ``concon`` hardcodes 33 -- see :data:`NUM_HARMONICS`.
    epsilon
        Cutoff threshold. ``None`` leaves the series untruncated in angle,
        which is useful for studying it but is not what the pipeline did.
    quantized
        ``True`` reproduces the binary's lookup tables exactly and is the
        default. ``False`` evaluates the series in closed form: the kernel the
        tables approximate, about 0.06% higher at the peak.

    Notes
    -----
    Verified against ``c3_main`` itself, run on a single streamline so its
    output is the kernel: the quantized form reproduces the binary's peak entry
    to six digits at five bandwidths, and its ring values at sigma 0.005 to rms
    0.0005; against the released matrices of five ADNI subjects the density
    correlates at 1.000000. ``tests/reference/concon_probe.py`` regenerates the
    measurement.

    Examples
    --------
    >>> import numpy as np
    >>> from sbci.smoothing import spherical_heat_kernel
    >>> centre = float(spherical_heat_kernel(1.0, 0.005))
    >>> round(centre, 3)
    270.267
    >>> float(spherical_heat_kernel(np.cos(np.radians(20.0)), 0.005))  # past the cutoff
    0.0
    """
    if quantized and epsilon is not None:
        return concon_kernel_table(sigma, harmonics, epsilon)(cosine)
    cosine = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    total = _legendre_series(cosine, sigma, harmonics)
    if epsilon is not None:
        total = np.where(cosine < np.cos(kernel_cutoff(sigma, epsilon, harmonics)), 0.0, total)
    return total


def endpoint_density(
    vertices,
    points_in,
    points_out,
    hemisphere_in,
    hemisphere_out,
    vertex_hemisphere,
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    normalize: str | None = None,
    block: int = 4096,
    method: str = "sparse",
    progress=None,
    quantized: bool = True,
):
    """Smooth streamline endpoints into a density with the spherical kernel.

    Each endpoint contributes a kernel centred on its **continuous** position,
    not on the nearest grid vertex, which is what ``c3_main`` does. Endpoints
    only contribute to vertices of their own hemisphere.

    Parameters
    ----------
    vertices
        ``(n, 3)`` grid vertices on the unit sphere.
    points_in, points_out
        ``(s, 3)`` continuous endpoint positions on the unit sphere.
    hemisphere_in, hemisphere_out, vertex_hemisphere
        Which hemisphere each endpoint and each vertex belongs to.
    sigma, harmonics, quantized
        Passed to :func:`spherical_heat_kernel`.
    normalize
        ``None``, the default, leaves each endpoint's kernel unscaled, which is
        what ``c3_main`` does -- it divides the finished density by the
        streamline count instead. ``"sum"`` scales each kernel to sum to one
        over the grid; it was a workaround for a scale mismatch that turned out
        to be the kernel's weight, and is kept only for comparison.
    block
        Endpoints processed at a time.
    method
        ``"sparse"``, the default, finds each endpoint's neighbourhood with a
        KD-tree and evaluates the kernel only there: the kernel is zero past
        about 12 degrees, so only ~31 of the 5124 vertices matter per endpoint
        and the dense evaluation was 99.4% zeros. ``"dense"`` evaluates every
        vertex and is kept as the check the sparse path is tested against.
    progress
        Optional callable ``progress(done, total)`` invoked after each block.

    Returns
    -------
    The symmetric ``(n, n)`` density.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    points_in = np.asarray(points_in, dtype=np.float64)
    points_out = np.asarray(points_out, dtype=np.float64)
    if points_in.shape != points_out.shape:
        raise ValueError(f"endpoints disagree: {points_in.shape} in, {points_out.shape} out")
    if normalize not in ("sum", None):
        raise ValueError(f"normalize must be 'sum' or None, got {normalize!r}")
    if method not in ("sparse", "dense"):
        raise ValueError(f"method must be 'sparse' or 'dense', got {method!r}")

    vertex_hemisphere = np.asarray(vertex_hemisphere)
    hemisphere_in = np.asarray(hemisphere_in)
    hemisphere_out = np.asarray(hemisphere_out)
    n = vertices.shape[0]
    total = points_in.shape[0]

    def kernel(cosine):
        return spherical_heat_kernel(cosine, sigma, harmonics, quantized=quantized)

    if method == "dense":

        def columns(points, hemisphere):
            values = np.asarray(kernel(vertices @ points.T), dtype=np.float64)
            values[vertex_hemisphere[:, None] != hemisphere[None, :]] = 0.0
            np.maximum(values, 0.0, out=values)
            if normalize == "sum":
                colsum = values.sum(axis=0)
                colsum[colsum == 0] = 1.0
                values = values / colsum
            return values

        density = np.zeros((n, n), dtype=np.float64)
        for start in range(0, total, block):
            stop = min(start + block, total)
            density += (
                columns(points_in[start:stop], hemisphere_in[start:stop])
                @ columns(points_out[start:stop], hemisphere_out[start:stop]).T
            )
            if progress is not None:
                progress(stop, total)
        return density + density.T

    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    cutoff = (
        concon_kernel_table(sigma, harmonics).cutoff
        if quantized
        else kernel_cutoff(sigma, KERNEL_EPSILON, harmonics)
    )
    # Chord length of the cutoff angle, with slack: the kernel itself decides
    # membership (it is zero past the cutoff), the tree only has to over-cover.
    radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.cos(cutoff)))) * (1.0 + 1e-6) + 1e-9
    sides = np.unique(vertex_hemisphere)
    trees = {int(h): cKDTree(vertices[vertex_hemisphere == h]) for h in sides}
    members = {int(h): np.nonzero(vertex_hemisphere == h)[0] for h in sides}

    def sparse_columns(points, hemisphere):
        rows, cols, vals = [], [], []
        for h in sides:
            chosen = np.nonzero(hemisphere == h)[0]
            if chosen.size == 0:
                continue
            neighbours = trees[int(h)].query_ball_point(points[chosen], radius)
            counts = np.fromiter(
                (len(a) for a in neighbours), dtype=np.int64, count=len(neighbours)
            )
            if counts.sum() == 0:
                continue
            local = np.concatenate([np.asarray(a, dtype=np.int64) for a in neighbours])
            col = np.repeat(chosen, counts)
            row = members[int(h)][local]
            cosine = np.einsum("ij,ij->i", vertices[row], points[col])
            value = np.asarray(kernel(cosine), dtype=np.float64)
            keep = value > 0.0
            rows.append(row[keep])
            cols.append(col[keep])
            vals.append(value[keep])
        if not rows:
            return csr_matrix((n, points.shape[0]), dtype=np.float64)
        matrix = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, points.shape[0]),
        )
        if normalize == "sum":
            colsum = np.asarray(matrix.sum(axis=0)).ravel()
            colsum[colsum == 0] = 1.0
            matrix = matrix @ csr_matrix(np.diag(1.0 / colsum))
        return matrix

    density = np.zeros((n, n), dtype=np.float64)
    for start in range(0, total, block):
        stop = min(start + block, total)
        product = (
            sparse_columns(points_in[start:stop], hemisphere_in[start:stop])
            @ sparse_columns(points_out[start:stop], hemisphere_out[start:stop]).T
        )
        density += product.toarray()
        if progress is not None:
            progress(stop, total)
    return density + density.T


def endpoint_positions(endpoints, surface=None):
    """Continuous unit-sphere positions of each endpoint, as ``concon`` sees them.

    ``concon`` centres a kernel on where the streamline actually crossed the
    surface, not on the nearest vertex, so the file stores each crossing as a
    triangle plus barycentric weights. This rebuilds the position from them.

    Parameters
    ----------
    endpoints
        :class:`Endpoints` carrying barycentric coordinates.
    surface
        The spherical grid mesh; loaded from the bundled surfaces if omitted.

    Returns
    -------
    tuple of ndarray
        ``(points_in, points_out)``, each ``(s, 3)`` on the unit sphere.
    """
    if not endpoints.has_positions:
        raise MissingDataError(_NO_POSITIONS)
    if surface is None:
        from .surface import load_surface

        surface = load_surface("sphere")
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    # Triangle indices are stored per hemisphere, so they need the same offset
    # the vertex indices get. Take it from the mesh rather than a constant, so
    # it stays right if the grid ever changes.
    faces_per_hemi = faces.shape[0] // 2

    def rebuild(triangle, bary, surf):
        index = np.asarray(triangle, dtype=np.int64) + Endpoints.global_hemisphere_offset(
            surf, faces_per_hemi
        )
        corners = vertices[faces[index]]  # (s, 3, 3)
        point = np.einsum("sk,skj->sj", np.asarray(bary, dtype=np.float64), corners)
        return point / np.linalg.norm(point, axis=1, keepdims=True)

    return (
        rebuild(endpoints.tri_in, endpoints.bary_in, endpoints.surf_in),
        rebuild(endpoints.tri_out, endpoints.bary_out, endpoints.surf_out),
    )


def read_concon_endpoints(path):
    """Read ``subject_xing_sphere_avg_coords.tsv``, the input ``c3_main`` takes.

    Returns ``(points_in, points_out, hemisphere_in, hemisphere_out)``, with
    hemispheres as 0 for left and 1 for right. The file stores ``1 - surface``
    in its hemisphere columns, which is undone here.
    """
    raw = np.loadtxt(path, skiprows=1)
    if raw.ndim == 1:
        raw = raw[None]
    if raw.shape[1] != 10:
        raise ValueError(f"{path}: expected 10 columns, got {raw.shape[1]}")

    def unit(block):
        return block / np.linalg.norm(block, axis=1, keepdims=True)

    return (
        unit(raw[:, 2:5]),
        unit(raw[:, 7:10]),
        (1 - raw[:, 1]).astype(np.int64),
        (1 - raw[:, 6]).astype(np.int64),
    )


def kappa_candidates(eigenvalues: np.ndarray, n: int = 10, index: int = 200) -> np.ndarray:
    """Log-spaced bandwidths spanning the usable range of the spectrum.

    The range is the one the reference script derives: the low end is the
    bandwidth at which the *highest* represented frequency is still damped only
    to 0.9, and the high end the bandwidth at which the ``index``-th eigenvalue
    is damped to 0.001. Below the low end the kernel barely smooths; above the
    high end it has thrown away everything but the coarsest modes.

    Examples
    --------
    >>> lam = np.linspace(0.0, 0.5, 300)
    >>> candidates = kappa_candidates(lam)
    >>> candidates.size, bool(np.all(np.diff(candidates) > 0))
    (10, True)
    """
    lam = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if lam.size < index:
        raise ValueError(f"need at least {index} eigenvalues, got {lam.size}")
    kappa_min = np.sqrt(-2.0 * np.log(0.9) / lam[-1])
    kappa_max = np.sqrt(-2.0 * np.log(0.001) / lam[index - 1])
    return np.exp(np.linspace(np.log(kappa_min), np.log(kappa_max), n))


@dataclass
class Endpoints:
    """Streamline endpoints on the two hemispheres.

    Attributes
    ----------
    surf_in, surf_out
        Hemisphere of each endpoint, ``0`` for left and ``1`` for right.
    vtx_in, vtx_out
        Zero-based vertex index within that hemisphere.
    tri_in, tri_out
        Optional zero-based triangle index within that hemisphere, and
        ``bary_in``/``bary_out`` the barycentric weights inside it. Together
        they give the endpoint's continuous position on the surface, which
        matters for a kernel evaluated off-grid: snapping to the nearest vertex
        moves an endpoint 1.5 degrees on average, against kernels a few degrees
        wide. Without them an endpoint is known only to the nearest vertex.
    n_per_hemi
        Vertices per hemisphere, 2562 on ico4.
    """

    surf_in: np.ndarray
    surf_out: np.ndarray
    vtx_in: np.ndarray
    vtx_out: np.ndarray
    n_per_hemi: int = 2562
    tri_in: np.ndarray | None = None
    tri_out: np.ndarray | None = None
    bary_in: np.ndarray | None = None
    bary_out: np.ndarray | None = None

    @property
    def n_streamlines(self) -> int:
        """How many streamlines these endpoints describe."""
        return int(np.asarray(self.vtx_in).size)

    @classmethod
    def from_matlab(cls, path, n_per_hemi: int = 2562) -> Endpoints:
        """Read ``mesh_intersections_*.mat`` from the legacy pipeline.

        MATLAB vertex indices are one-based and are converted here.
        """
        import scipy.io

        data = scipy.io.loadmat(path)
        missing = [k for k in ("surf_in", "surf_out", "vtx_in", "vtx_out") if k not in data]
        if missing:
            raise FormatError(f"{path} has no {missing}")
        optional = {}
        if all(k in data for k in ("tri_in", "tri_out", "pt_in", "pt_out")):
            optional = {
                "tri_in": np.asarray(data["tri_in"]).ravel().astype(np.int64) - 1,
                "tri_out": np.asarray(data["tri_out"]).ravel().astype(np.int64) - 1,
                "bary_in": np.asarray(data["pt_in"], dtype=np.float64),
                "bary_out": np.asarray(data["pt_out"], dtype=np.float64),
            }
        return cls(
            surf_in=np.asarray(data["surf_in"]).ravel().astype(np.int8),
            surf_out=np.asarray(data["surf_out"]).ravel().astype(np.int8),
            vtx_in=np.asarray(data["vtx_in"]).ravel().astype(np.int64) - 1,
            vtx_out=np.asarray(data["vtx_out"]).ravel().astype(np.int64) - 1,
            n_per_hemi=n_per_hemi,
            **optional,
        )

    @property
    def has_positions(self) -> bool:
        """Whether continuous positions are available, not only vertices."""
        return self.bary_in is not None and self.bary_out is not None

    @property
    def global_vertex_in(self) -> np.ndarray:
        """Endpoint vertices as indices into the whole 5124-vertex grid.

        The hemisphere flags are int8, so they are widened before being scaled
        by the vertices per hemisphere -- otherwise the offset overflows.
        """
        return np.asarray(self.vtx_in, dtype=np.int64) + self.global_hemisphere_offset(
            self.surf_in, self.n_per_hemi
        )

    @property
    def global_vertex_out(self) -> np.ndarray:
        """Endpoint vertices as indices into the whole 5124-vertex grid."""
        return np.asarray(self.vtx_out, dtype=np.int64) + self.global_hemisphere_offset(
            self.surf_out, self.n_per_hemi
        )

    @staticmethod
    def global_hemisphere_offset(surf: np.ndarray, per_hemi: int) -> np.ndarray:
        """``surf * per_hemi`` in a width that cannot overflow."""
        return np.asarray(surf, dtype=np.int64) * int(per_hemi)

    @classmethod
    def from_global(
        cls,
        vertex_in: np.ndarray,
        vertex_out: np.ndarray,
        n_per_hemi: int = 2562,
        n_faces_per_hemi: int = 5120,
        triangle_in: np.ndarray | None = None,
        triangle_out: np.ndarray | None = None,
        barycentric_in: np.ndarray | None = None,
        barycentric_out: np.ndarray | None = None,
    ) -> Endpoints:
        """Build from whole-grid indices, as the HDF5 file stores them.

        The hemisphere is read off the index rather than carried separately, so
        the two cannot disagree.
        """
        vertex_in = np.asarray(vertex_in, dtype=np.int64)
        vertex_out = np.asarray(vertex_out, dtype=np.int64)
        for name, vertex in (("vertex_in", vertex_in), ("vertex_out", vertex_out)):
            if vertex.size and (vertex.min() < 0 or vertex.max() >= 2 * n_per_hemi):
                raise ValueError(
                    f"{name} out of range for a {2 * n_per_hemi}-vertex grid: "
                    f"[{vertex.min()}, {vertex.max()}]"
                )
        optional: dict[str, np.ndarray] = {}
        if triangle_in is not None:
            triangle_in = np.asarray(triangle_in, dtype=np.int64)
            triangle_out = np.asarray(triangle_out, dtype=np.int64)
            optional = {
                "tri_in": triangle_in % n_faces_per_hemi,
                "tri_out": triangle_out % n_faces_per_hemi,
                "bary_in": np.asarray(barycentric_in, dtype=np.float64),
                "bary_out": np.asarray(barycentric_out, dtype=np.float64),
            }
        return cls(
            surf_in=(vertex_in // n_per_hemi).astype(np.int8),
            surf_out=(vertex_out // n_per_hemi).astype(np.int8),
            vtx_in=vertex_in % n_per_hemi,
            vtx_out=vertex_out % n_per_hemi,
            n_per_hemi=n_per_hemi,
            **optional,
        )

    def adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Count matrices ``(A11, A22, A12)`` for LL, RR and LR connections.

        ``A11`` and ``A22`` are symmetrized by adding their transpose, which
        doubles any self-connection on the diagonal; that is what the reference
        does and what the released data was built with.
        """
        n = self.n_per_hemi
        for name, vtx in (("vtx_in", self.vtx_in), ("vtx_out", self.vtx_out)):
            if vtx.size and (vtx.min() < 0 or vtx.max() >= n):
                raise ValueError(
                    f"{name} out of range for {n} vertices per hemisphere: "
                    f"[{vtx.min()}, {vtx.max()}]. One-based indices need converting."
                )

        def counts(rows, cols, shape):
            out = np.zeros(shape, dtype=np.float64)
            np.add.at(out, (rows, cols), 1.0)
            return out

        left_left = (self.surf_in == 0) & (self.surf_out == 0)
        a11 = counts(self.vtx_in[left_left], self.vtx_out[left_left], (n, n))
        a11 = a11 + a11.T

        right_right = (self.surf_in == 1) & (self.surf_out == 1)
        a22 = counts(self.vtx_in[right_right], self.vtx_out[right_right], (n, n))
        a22 = a22 + a22.T

        left_right = (self.surf_in == 0) & (self.surf_out == 1)
        a12 = counts(self.vtx_in[left_right], self.vtx_out[left_right], (n, n))
        right_left = (self.surf_in == 1) & (self.surf_out == 0)
        a12 = a12 + counts(self.vtx_in[right_left], self.vtx_out[right_left], (n, n)).T

        return a11, a22, a12


def smooth_endpoints(
    endpoints: Endpoints,
    kernel_left: np.ndarray,
    kernel_right: np.ndarray,
) -> np.ndarray:
    """Smooth endpoint counts into a density, as ``K A K`` per block.

    Returns the full ``(2 * n_per_hemi, 2 * n_per_hemi)`` symmetric density,
    left hemisphere first. It sums to one over the whole matrix.

    Negative entries produced by spectral truncation are clipped to zero, and
    the normalizing constant is formed from the positive entries only, exactly
    as in ``rdk_smoothed_concon_compute.m``.
    """
    a11, a22, a12 = endpoints.adjacency()
    n = endpoints.n_per_hemi

    for name, kernel in (("kernel_left", kernel_left), ("kernel_right", kernel_right)):
        if kernel.shape != (n, n):
            raise ValueError(f"{name} has shape {kernel.shape}, expected {(n, n)}")

    im11 = kernel_left @ a11 @ kernel_left
    im22 = kernel_right @ a22 @ kernel_right
    im12 = kernel_left @ a12 @ kernel_right

    total = im11[im11 > 0].sum() + im22[im22 > 0].sum() + 2.0 * im12[im12 > 0].sum()
    if total <= 0:
        raise ValueError("smoothed intensity has no positive mass; check the endpoints")

    density = np.zeros((2 * n, 2 * n), dtype=np.float64)
    density[:n, :n] = np.where(im11 > 0, im11 / total, 0.0)
    density[n:, n:] = np.where(im22 > 0, im22 / total, 0.0)
    block12 = np.where(im12 > 0, im12 / total, 0.0)
    density[:n, n:] = block12
    density[n:, :n] = block12.T
    return density


def load_eigenpairs(path, hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``EV_LBO_ds_ico4_{L,R}.mat`` and return ``(eigenvalues, eigenvectors)``."""
    import scipy.io

    hemisphere = hemisphere.upper()
    if hemisphere not in ("L", "R"):
        raise ValueError(f"hemisphere must be 'L' or 'R', got {hemisphere!r}")

    data = scipy.io.loadmat(path)
    lam_key, vec_key = f"Lambda_{hemisphere}", f"U_{hemisphere}"
    if lam_key not in data or vec_key not in data:
        available = [k for k in data if not k.startswith("__")]
        raise FormatError(f"{path} has no {lam_key}/{vec_key}; it has {available}")

    eigenvalues = np.asarray(data[lam_key], dtype=np.float64).ravel()
    eigenvectors = np.asarray(data[vec_key], dtype=np.float64)
    if not np.all(np.diff(eigenvalues) >= 0):
        raise ValueError(f"{path}: eigenvalues are not ascending")
    return eigenvalues, eigenvectors


_NO_ENDPOINTS = (
    "This file carries no streamline endpoints, so there is nothing to "
    "re-smooth from. The format has an optional /endpoints group: build one "
    "with Endpoints.from_matlab('mesh_intersections_ico4.mat') and pass it to "
    "ContinuousConnectome.save(), or construct the connectome with "
    "endpoints=... before saving."
)

_NO_POSITIONS = (
    "The spherical kernel smooths from where each streamline actually crossed "
    "the surface, so it needs the barycentric coordinates in the /endpoints "
    "group, not only the nearest vertex. This file carries vertices alone. "
    "Rebuild it with Endpoints.from_matlab('mesh_intersections_ico4.mat'), "
    "which keeps them, or pass kernel='rdk'."
)

_NO_EIGENPAIRS = (
    "The {kernel} kernel needs the Laplace-Beltrami basis for the grid. Pass "
    "eigenpairs= as the directory holding EV_LBO_ds_ico4_L.mat and "
    "EV_LBO_ds_ico4_R.mat, or as a pair of (eigenvalues, eigenvectors) tuples "
    "for left and right, or point SBCI_LBO_DIR at that directory. The files "
    "are 50 MB per hemisphere and are a property of the grid rather than of "
    "the subject, so they are not bundled -- and recomputing them from the "
    "bundled white surface reproduces the reference kernel only to r = 0.94, "
    "which is not close enough to substitute."
)

EIGENPAIR_FILES = ("EV_LBO_ds_ico4_L.mat", "EV_LBO_ds_ico4_R.mat")
"""What :func:`smooth` looks for when ``eigenpairs`` names a directory."""

EIGENPAIR_ENV = "SBCI_LBO_DIR"
"""Environment variable naming the directory that holds the basis."""


def _candidate_basis_directories():
    """Where :func:`find_basis` looks, in order."""
    import os
    from pathlib import Path

    candidates = []
    if os.environ.get(EIGENPAIR_ENV):
        candidates.append(Path(os.environ[EIGENPAIR_ENV]))
    if os.environ.get("SBCI_TOOLKIT"):
        candidates.append(Path(os.environ["SBCI_TOOLKIT"]) / "concon_estimate")
    candidates.append(Path.home() / ".cache" / "sbci" / "lbo")
    candidates.append(Path.home() / ".sbci" / "lbo")
    return candidates


def find_basis():
    """The Laplace-Beltrami basis directory, or ``None`` if nowhere obvious.

    The basis is 50 MB per hemisphere and describes the grid rather than any
    subject, so it is neither bundled in the wheel nor copied into every file.
    Rather than force every caller to pass a path, :func:`smooth` looks in a
    few conventional places first; set ``SBCI_LBO_DIR`` to make it explicit.
    """
    for directory in _candidate_basis_directories():
        if all((directory / name).is_file() for name in EIGENPAIR_FILES):
            return directory
    return None


def _resolve_eigenpairs(eigenpairs):
    """Accept a directory of MATLAB files or a pair of arrays; return both."""
    from pathlib import Path

    if isinstance(eigenpairs, (str, Path)):
        base = Path(eigenpairs)
        if not base.is_dir():
            raise NotADirectoryError(
                f"{base} is not a directory; eigenpairs= should name the folder "
                f"holding {EIGENPAIR_FILES[0]} and {EIGENPAIR_FILES[1]}"
            )
        paths = [base / name for name in EIGENPAIR_FILES]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing eigenpair files: {missing}")
        return load_eigenpairs(paths[0], "L"), load_eigenpairs(paths[1], "R")

    try:
        (lam_left, vec_left), (lam_right, vec_right) = eigenpairs
    except (TypeError, ValueError) as error:
        raise TypeError(
            "eigenpairs= should be a directory, or a pair of "
            "(eigenvalues, eigenvectors) tuples for left and right"
        ) from error
    return (
        (np.asarray(lam_left, dtype=np.float64).ravel(), np.asarray(vec_left, dtype=np.float64)),
        (np.asarray(lam_right, dtype=np.float64).ravel(), np.asarray(vec_right, dtype=np.float64)),
    )


def smooth(
    connectome,
    kernel: str = "shk",
    bandwidth: float | None = None,
    eigenpairs=None,
    mask_medial_wall: bool = False,
    progress=None,
):
    """Re-smooth a connectome from its stored endpoints.

    Returns a new connectome of the same type, carrying the same endpoints,
    mask, areas and coordinates, with ``kernel`` and ``bandwidth`` in its
    metadata set to what was actually used.

    :func:`smooth_endpoints` normalizes its density to sum to one over the
    whole matrix, diagonal included. This renormalizes to the package's
    convention instead -- ``area @ D @ area == 1`` with the diagonal dropped --
    so that the result satisfies the same unit-mass check as a file read from
    disk. The two differ by the self-connectivity the storage format excludes.

    Parameters
    ----------
    connectome
        A :class:`~sbci.ContinuousConnectome` carrying endpoints.
    kernel
        One of :data:`KERNELS`.
    bandwidth
        Kernel bandwidth ``kappa``. Defaults to the value the reference
        selection formula picks, ``kappa_candidates(eigenvalues)[3]``.
    eigenpairs
        Directory holding the Laplace-Beltrami basis, or a pair of
        ``(eigenvalues, eigenvectors)`` tuples.
    mask_medial_wall
        ``False``, the default, leaves whatever mass the kernel put on the
        medial wall, as both references do. ``True`` zeroes it before the
        unit-mass normalization so the result satisfies the format's mask rule
        and ``sbci validate`` -- SPEC_QUESTIONS.md item 14.
    progress
        Optional ``progress(done, total)`` callable, called after each block of
        endpoints for the ``shk`` kernel.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if getattr(connectome, "endpoints", None) is None:
        raise MissingDataError(_NO_ENDPOINTS)

    if kernel == "shk":
        sigma = DEFAULT_SIGMA if bandwidth is None else float(bandwidth)
        points_in, points_out = endpoint_positions(connectome.endpoints)
        from .grid import hemisphere_labels

        density = endpoint_density(
            _grid_vertices(connectome),
            points_in,
            points_out,
            connectome.endpoints.surf_in,
            connectome.endpoints.surf_out,
            hemisphere_labels(connectome.n_vertices),
            sigma=sigma,
            progress=progress,
        )
        apply_final_threshold(density, points_in.shape[0])
        return _finish(connectome, density, kernel, sigma, mask_medial_wall)

    if eigenpairs is None:
        eigenpairs = find_basis()
    if eigenpairs is None:
        searched = "\n  ".join(str(d) for d in _candidate_basis_directories())
        raise MissingDataError(
            _NO_EIGENPAIRS.format(kernel=kernel)
            + f"\n\nLooked for {EIGENPAIR_FILES[0]} and {EIGENPAIR_FILES[1]} in:"
            f"\n  {searched}"
        )

    (lam_left, vec_left), (lam_right, vec_right) = _resolve_eigenpairs(eigenpairs)
    if bandwidth is None:
        bandwidth = float(kappa_candidates(lam_left)[3])

    build = diffusion_kernel if kernel == "rdk" else matern_kernel
    density = smooth_endpoints(
        connectome.endpoints,
        build(lam_left, vec_left, bandwidth),
        build(lam_right, vec_right, bandwidth),
    )

    return _finish(connectome, density, kernel, bandwidth, mask_medial_wall)


def apply_final_threshold(
    density: np.ndarray, n_streamlines: int, threshold: float = FINAL_THRESHOLD
) -> np.ndarray:
    """Drop pairs ``c3_main`` would not have written, in place.

    Its output is the accumulated kernel product divided by the streamline
    count, written only where that exceeds ``--final_thold``. The density here
    is the undivided sum, so the rule is applied on the divided scale.

    Examples
    --------
    >>> d = np.array([[0.0, 3e-9], [3e-9, 0.0]]) * 1000
    >>> apply_final_threshold(d, 1000)            # 3e-9 per streamline survives
    array([[0.e+00, 3.e-06],
           [3.e-06, 0.e+00]])
    >>> apply_final_threshold(d, 10**7)           # 3e-13 per streamline does not
    array([[0., 0.],
           [0., 0.]])
    """
    density[density / n_streamlines <= threshold] = 0.0
    return density


def _grid_vertices(connectome) -> np.ndarray:
    """Unit-sphere grid vertices: the file's own if it carries them."""
    if getattr(connectome, "coords", None) is not None:
        vertices = np.asarray(connectome.coords, dtype=np.float64)
    else:
        from .surface import load_surface

        vertices = np.asarray(load_surface("sphere").vertices, dtype=np.float64)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def _finish(
    connectome, density: np.ndarray, kernel: str, bandwidth: float, mask_medial_wall: bool = False
):
    """Normalize a re-smoothed density and wrap it back into a connectome."""
    from .grid import to_condensed
    from .metadata import Metadata

    # The medial wall is NOT zeroed by default. Neither reference masks: c3_main's
    # own released output puts 0.87% of its mass on the wall, and the MATLAB rdk
    # density does too, so masking by default would make this method diverge from
    # both and `test_smooth_method_matches_matlab` would stop holding. It does
    # mean a re-smoothed density can fail the validator's mask check -- a genuine
    # conflict between the format and the reference, SPEC_QUESTIONS.md item 14.
    # `mask_medial_wall=True` is the caller's way to choose the format's side.
    if mask_medial_wall:
        wall = ~np.asarray(connectome.mask, dtype=bool)
        density[wall, :] = 0.0
        density[:, wall] = 0.0

    # The storage convention is the strict upper triangle, so self-connectivity
    # is dropped (SPEC_QUESTIONS.md item 2). Normalize after dropping it, or
    # the file that comes back would not satisfy the unit-mass check that
    # validates it.
    np.fill_diagonal(density, 0.0)
    area = np.asarray(connectome.area, dtype=np.float64)
    mass = float(area @ density @ area)
    if mass <= 0:
        raise ValueError("the re-smoothed density has no positive area-weighted mass")
    density /= mass

    fields = dict(connectome.metadata.fields)
    fields.update(kernel=kernel, bandwidth=float(bandwidth), normalization="unit-mass")
    return type(connectome)(
        data=to_condensed(density).astype(np.float32),
        area=connectome.area,
        mask=connectome.mask,
        metadata=Metadata(fields),
        coords=connectome.coords,
        endpoints=connectome.endpoints,
    )
