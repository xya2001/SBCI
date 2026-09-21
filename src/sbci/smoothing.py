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

import numpy as np

KERNELS = ("shk", "rdk", "matern")
"""The kernel vocabulary, following the decision recorded as SPEC_QUESTIONS.md
item 10.

``shk``
    Spherical heat kernel (Moyer et al., MICCAI 2016), the default, and the one
    that produced every released ``smoothed_sc_avg_*.mat``. Its bandwidth is
    ``sigma``. It is **not shipped yet**: against the input the pipeline
    actually feeds its smoother the port reaches r = 0.978, but a 20%
    amplitude error remains after the best global scale, traced to an
    undocumented normalization or sampling convention. Withheld until that
    closes -- see PORTING.md item 6.
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


SPHERICAL_HARMONICS = 33
"""Harmonics the pipeline truncates the spherical heat kernel at.

From ``sbci_step5_structural.sh``: ``--OPT_VAL_num_harm 33``.
"""


def spherical_heat_kernel(cosine, sigma: float, harmonics: int = SPHERICAL_HARMONICS):
    r"""Heat kernel on the unit sphere, as a truncated Legendre series.

    .. math::
        h_\sigma(\cos\gamma) = \sum_{l=0}^{L}
            \frac{2l+1}{4\pi} e^{-l(l+1)\sigma} P_l(\cos\gamma)

    This is the kernel ``concon`` applies, and ``--sigma`` is diffusion time:
    of the three readings of that flag this is the one that reproduces the
    pipeline's own output, and its support lands within 2.4% of the released
    file's.

    Parameters
    ----------
    cosine
        Cosines of the angle between points, any shape. Values are clipped to
        ``[-1, 1]`` so unit vectors that are a rounding step too long are safe.
    sigma
        Diffusion time. The released cohorts use 0.005.
    harmonics
        Where to truncate. The pipeline uses 33; the result is converged to
        four decimals by about 17.

    Notes
    -----
    **The truncation is part of the definition, not an approximation to it.**
    At the released bandwidth of 0.005 the ``l = 33`` term still carries weight
    0.0195, and the 33-term sum differs from a converged one by 0.31% of its
    peak; at 17 terms the error is 20%. Anything that hopes to reproduce the
    pipeline has to truncate where the pipeline truncates.

    A truncated series also rings. At sigma 0.005 the kernel falls cleanly for
    the first 22 degrees and then oscillates, dipping to -0.008 against a peak
    of 15.9 -- about -0.05%. :func:`endpoint_density` clips those negatives, as
    the reference does.

    Examples
    --------
    >>> import numpy as np
    >>> from sbci.smoothing import spherical_heat_kernel
    >>> float(spherical_heat_kernel(1.0, 0.005)) > float(spherical_heat_kernel(0.0, 0.005))
    True
    """
    cosine = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    degree = np.arange(harmonics + 1)
    weight = ((2 * degree + 1) / (4 * np.pi)) * np.exp(-degree * (degree + 1) * sigma)

    previous = np.ones_like(cosine)
    current = cosine.copy()
    total = weight[0] * previous + weight[1] * current
    for order in range(1, harmonics):
        previous, current = (
            current,
            ((2 * order + 1) * cosine * current - order * previous) / (order + 1),
        )
        total = total + weight[order + 1] * current
    return total


def endpoint_density(
    vertices,
    points_in,
    points_out,
    hemisphere_in,
    hemisphere_out,
    vertex_hemisphere,
    sigma: float,
    harmonics: int = SPHERICAL_HARMONICS,
    normalize: str = "sum",
    block: int = 4096,
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
    sigma, harmonics
        Passed to :func:`spherical_heat_kernel`.
    normalize
        ``"sum"`` scales each endpoint's kernel so its values over the grid sum
        to one; ``None`` leaves it unscaled. **This is the open question**: the
        port reaches r = 0.978 against the pipeline's own output with either,
        but a 20% amplitude error remains that has been traced to the
        normalization or to ``c3_main``'s sampling flags rather than to the
        kernel. See PORTING.md item 6.
    block
        Endpoints processed at a time; the intermediate is ``(n, block)``.

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

    vertex_hemisphere = np.asarray(vertex_hemisphere)
    hemisphere_in = np.asarray(hemisphere_in)
    hemisphere_out = np.asarray(hemisphere_out)
    n = vertices.shape[0]

    def columns(points, hemisphere):
        values = spherical_heat_kernel(vertices @ points.T, sigma, harmonics)
        values[vertex_hemisphere[:, None] != hemisphere[None, :]] = 0.0
        np.maximum(values, 0.0, out=values)
        if normalize == "sum":
            total = values.sum(axis=0)
            total[total == 0] = 1.0
            values = values / total
        return values.astype(np.float32)

    density = np.zeros((n, n), dtype=np.float64)
    for start in range(0, points_in.shape[0], block):
        stop = min(start + block, points_in.shape[0])
        density += (
            columns(points_in[start:stop], hemisphere_in[start:stop])
            @ columns(points_out[start:stop], hemisphere_out[start:stop]).T
        )
    return density + density.T


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
            raise KeyError(f"{path} has no {missing}")
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
        raise KeyError(f"{path} has no {lam_key}/{vec_key}; it has {available}")

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


def smooth(connectome, kernel: str = "shk", bandwidth: float | None = None, eigenpairs=None):
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
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if kernel == "shk":
        raise NotImplementedError(
            "The spherical heat kernel is the default by decision "
            "(SPEC_QUESTIONS.md item 10), because it is what produced the "
            "released cohorts. The port of it reaches r = 0.978 against the "
            "input the pipeline actually feeds its smoother, but a 20% "
            "amplitude error remains, so it is not shipped yet "
            "(PORTING.md item 6). The Riemannian diffusion kernel and the "
            "Matern kernel are ported and verified: pass kernel='rdk' or "
            "kernel='matern'."
        )
    if getattr(connectome, "endpoints", None) is None:
        raise ValueError(_NO_ENDPOINTS)
    if eigenpairs is None:
        eigenpairs = find_basis()
    if eigenpairs is None:
        searched = "\n  ".join(str(d) for d in _candidate_basis_directories())
        raise ValueError(
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

    from .grid import to_condensed

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

    from .metadata import Metadata

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
