"""The one object a user of this package holds: :class:`ContinuousConnectome`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import grid, io, parcellation
from .atlas import Atlas, load_atlas
from .errors import InvalidFileError
from .metadata import Metadata


class ContinuousConnectome:
    """A continuous connectome on the ico4 surface grid.

    Connectivity is held as the strict upper triangle in float32, the form it
    takes on disk, and expanded on demand. A structural connectome is a
    nonnegative density; a functional one carries signed values in the same
    layout.

    Construct instances with :meth:`load` rather than directly -- the
    constructor does not validate.

    Attributes
    ----------
    data
        Condensed connectivity, length ``n * (n - 1) / 2``.
    area
        Per-vertex area weight, length ``n``.
    mask
        ``True`` where the vertex is cortex; the medial wall is ``False``.
    metadata
        Validated :class:`~sbci.metadata.Metadata`.
    coords
        Vertex coordinates, or ``None`` if the file omitted them.
    endpoints
        :class:`sbci.smoothing.Endpoints` if the file carried them, else
        ``None``. Only a structural connectome has them, and only then can it
        be re-smoothed.
    """

    def __init__(
        self,
        data: np.ndarray,
        area: np.ndarray,
        mask: np.ndarray,
        metadata: Metadata,
        coords: np.ndarray | None = None,
        endpoints: Any = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.area = np.asarray(area, dtype=np.float64)
        self.mask = np.asarray(mask, dtype=bool)
        self.metadata = metadata
        self.coords = coords
        self.endpoints = endpoints

    # --- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> ContinuousConnectome:
        """Read a computational ``.h5`` file and validate it.

        The loader refuses a file whose metadata is missing a required key. The
        ``.dconn.nii`` exchange file is write-only: its resampling to fsLR-32k
        is many-to-one, so it cannot be read back onto the ico4 grid.

        Examples
        --------
        >>> from sbci import ContinuousConnectome
        >>> cc = ContinuousConnectome.load("sub-100307_sc.h5")   # doctest: +SKIP
        """
        path = Path(path)
        suffixes = "".join(path.suffixes)
        if suffixes.endswith((".h5", ".hdf5")):
            # Checked here, after the kind of file is known, so that an
            # unrecognized name is reported as such whether or not it exists.
            if not path.exists():
                raise FileNotFoundError(f"no such file: {path}")
            parts = io.read_hdf5(path)
        elif suffixes.endswith((".dconn.nii", ".dconn.nii.gz")):
            raise InvalidFileError(
                f"{path.name!r} is an exchange file, which this package writes but does "
                "not read: the resampling to fsLR-32k is many-to-one, so it cannot be "
                "mapped back onto the ico4 grid. Load the .h5 computational file instead."
            )
        else:
            raise InvalidFileError(
                f"unrecognized connectome file {path.name!r}; "
                "expected a .h5 computational file or a .dconn.nii exchange file"
            )

        parts["metadata"].validate()
        connectome = cls(**parts)
        connectome._check_shapes()
        return connectome

    def _check_shapes(self) -> None:
        """Confirm the arrays agree on the number of vertices."""
        n = grid.n_from_condensed(self.data.size)
        if self.area.size != n or self.mask.size != n:
            raise InvalidFileError(
                f"connectivity implies {n} vertices but area has {self.area.size} "
                f"and mask has {self.mask.size}"
            )
        if self.coords is not None and np.shape(self.coords) != (n, 3):
            raise InvalidFileError(f"coordinates are {np.shape(self.coords)}, expected {(n, 3)}")

    # --- properties --------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Vertices on the computational grid."""
        return grid.n_from_condensed(self.data.size)

    @property
    def has_endpoints(self) -> bool:
        """Whether this file carries the streamlines it was smoothed from."""
        return self.endpoints is not None

    @property
    def modality(self) -> str:
        """``'sc'`` or ``'fc'``."""
        return self.metadata.modality

    def dense(self) -> np.ndarray:
        """Expand to a symmetric ``n x n`` matrix with a zero diagonal.

        Allocates ``n**2`` float32, about 105 MB on the ico4 grid.
        """
        return grid.to_dense(self.data, self.n_vertices)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ContinuousConnectome {self.modality} on {self.n_vertices} vertices>"

    # --- implemented API ---------------------------------------------------

    def to_atlas(self, atlas: Atlas | str, how: str | None = None) -> np.ndarray:
        """Aggregate to a region-by-region matrix for a bundled atlas.

        Parameters
        ----------
        atlas
            An :class:`~sbci.Atlas`, or the name of a bundled one -- anything
            :func:`sbci.load_atlas` accepts, such as ``"Schaefer200"`` or
            ``"Desikan"``.
        how
            ``"mass"`` preserves total connectivity; ``"mean"`` divides by the
            region areas to give a density. The default is ``"mass"`` for a
            structural connectome and ``"mean"`` for a functional one: FC is
            aggregated through Fisher-z, which is an average of correlations
            and has no mass -- asking for ``"mass"`` on FC is refused.

        Vertices outside the cortical mask contribute neither connectivity nor
        area, so an atlas that labels the medial wall (``PALS_B12_Lobes``) is
        aggregated over cortex only.

        Examples
        --------
        >>> M = cc.to_atlas("Schaefer200")   # doctest: +SKIP
        >>> M.shape                           # doctest: +SKIP
        (200, 200)
        """
        if isinstance(atlas, str):
            atlas = load_atlas(atlas)
        functional = self.modality == "fc"
        if how is None:
            how = "mean" if functional else "mass"
        if functional and how == "mass":
            raise ValueError(
                "a functional connectome holds correlations, which have no mass; "
                "use how='mean' (the default for FC)"
            )
        return parcellation.parcellate(
            self.dense(),
            atlas,
            np.where(self.mask, self.area, 0.0),
            how=how,
            fisher_z=functional,
        )

    def seed(self, vertex: int | None = None, region: Any = None) -> np.ndarray:
        """Connectivity profile of one seed, as a map over the surface.

        Give either a ``vertex`` index, returning that vertex's density slice,
        or a ``region``, returning the area-weighted marginal over it. A region
        is a boolean mask over vertices, or an ``(atlas, region)`` pair naming
        one -- ``cc.seed(region=("Desikan", "LH_bankssts"))``.

        Examples
        --------
        >>> p = cc.seed(vertex=1234)    # doctest: +SKIP
        >>> p.shape                      # doctest: +SKIP
        (5124,)
        """
        if (vertex is None) == (region is None):
            raise ValueError("pass exactly one of vertex= or region=")

        n = self.n_vertices
        if vertex is not None:
            if not 0 <= vertex < n:
                raise IndexError(f"vertex {vertex} out of range for {n} vertices")
            return self._row(int(vertex))

        if isinstance(region, tuple) and len(region) == 2:
            atlas, which = region
            if isinstance(atlas, str):
                atlas = load_atlas(atlas)
            region = atlas.region_mask(which)

        member = np.asarray(region)
        if member.dtype != bool:
            raise TypeError(
                "region= must be a boolean mask over vertices, or an "
                '(atlas, region) pair such as ("Desikan", "LH_bankssts"); '
                f"got an array of dtype {member.dtype}"
            )
        if member.size != n:
            raise ValueError(f"region mask has {member.size} entries, expected {n}")
        weights = np.where(member, self.area, 0.0)
        total = weights.sum()
        if total == 0:
            raise ValueError("region mask selects no vertices")
        # Sum the members' rows straight from the condensed vector: no n x n
        # matrix for a region of a few dozen vertices.
        profile = np.zeros(n, dtype=np.float64)
        for vertex in np.flatnonzero(member):
            profile += weights[vertex] * self._row(int(vertex))
        return profile / total

    def _row(self, vertex: int) -> np.ndarray:
        """One row of the dense matrix, read from the condensed vector."""
        n = self.n_vertices
        profile = np.zeros(n, dtype=np.float64)
        # Entries (i, vertex) for i < vertex, then (vertex, j) for j > vertex.
        if vertex > 0:
            rows = np.arange(vertex)
            profile[:vertex] = self.data[_condensed_index(rows, vertex, n)]
        if vertex + 1 < n:
            start = _condensed_index(vertex, vertex + 1, n)
            profile[vertex + 1 :] = self.data[start : start + (n - vertex - 1)]
        return profile

    def save(self, path: str | Path) -> Path:
        """Write the computational HDF5 file."""
        return io.write_hdf5(
            path,
            data=self.data,
            area=self.area,
            mask=self.mask,
            metadata=self.metadata,
            coords=self.coords,
            endpoints=self.endpoints,
        )

    def to_cifti(self, path: str | Path) -> Path:
        """Write the exchange ``.dconn.nii`` on fsLR-32k."""
        return io.write_cifti(path, self)

    # --- analysis methods (each a verified port, see PORTING.md) -----------

    def coupling(self, fc: ContinuousConnectome, scope: str = "global", **kwargs) -> np.ndarray:
        """Structure-function coupling, as defined in the 2021 paper.

        Cosine similarity between the SC and FC profiles, with negative FC
        values kept. ``scope="global"`` gives one map over the surface,
        ``scope="region"`` the within-region pattern (which needs ``labels=``),
        and ``scope="discrete"`` an atlas-level summary -- that last one is a
        Pearson correlation rather than a cosine, following the reference.
        """
        from .coupling import structure_function_coupling

        return structure_function_coupling(self, fc, scope=scope, **kwargs)

    def smooth(
        self,
        kernel: str = "shk",
        bandwidth: float | None = None,
        eigenpairs: Any = None,
        **kwargs: Any,
    ):
        """Re-smooth from the stored endpoints, in congruence form ``K A K^T``.

        Extra keyword arguments -- ``mask_medial_wall``, ``progress`` -- pass
        through to :func:`sbci.smoothing.smooth`.

        Returns a new :class:`ContinuousConnectome` carrying the same endpoints
        and metadata, with ``kernel`` and ``bandwidth`` updated to what was
        used, so a re-smoothed file still says how it was made.

        Parameters
        ----------
        kernel
            One of :data:`sbci.smoothing.KERNELS`. The default is the spherical
            heat kernel, which is what the released cohorts use; see
            SPEC_QUESTIONS.md item 10 for the decision and what each name means.
        bandwidth
            ``sigma`` for ``shk`` (default 0.005, the released cohorts' value);
            ``kappa`` for ``rdk`` and ``matern``, defaulting to the value the
            reference selection formula picks from the spectrum.
        eigenpairs
            Where to get the Laplace-Beltrami basis: a directory holding
            ``EV_LBO_ds_ico4_L.mat`` and ``EV_LBO_ds_ico4_R.mat``, or a pair of
            ``(eigenvalues, eigenvectors)`` tuples for left and right. Needed
            by ``rdk`` and ``matern``; the files are 50 MB per hemisphere and
            belong to the grid rather than the subject, so they are not
            bundled.

        Examples
        --------
        >>> resmoothed = sc.smooth(kernel="rdk", eigenpairs="/path/to/lbo")  # doctest: +SKIP
        """
        from .smoothing import smooth as _smooth

        return _smooth(self, kernel=kernel, bandwidth=bandwidth, eigenpairs=eigenpairs, **kwargs)

    def reduce(self, rank: int, **kwargs):
        """Rank-``rank`` separable approximation of this one connectome.

        Returns a :class:`sbci.reduction.Reduction`. For a cohort use
        :func:`sbci.reduce`, which estimates one basis shared across subjects;
        a single connectome is the one-subject case of the same fit.

        Examples
        --------
        >>> result = cc.reduce(rank=10)             # doctest: +SKIP
        >>> result.explained[-1]                     # doctest: +SKIP
        0.87
        """
        from .reduction import reduce as _reduce

        return _reduce(self, rank=rank, **kwargs)

    def plot(self, values: np.ndarray, surface: str = "inflated", **kwargs):
        """Render a per-vertex map on a cortical surface.

        Returns the matplotlib figure. Needs the plotting extra:
        ``pip install 'sbci[plotting]'``.

        Parameters
        ----------
        values
            One number per vertex, such as the output of :meth:`seed` or
            :meth:`coupling`.
        surface
            Which bundled geometry to draw on: ``"inflated"``, ``"white"``,
            ``"pial"`` or ``"sphere"``.
        """
        from .plotting import plot_surface

        return plot_surface(values, surface=surface, connectome=self, **kwargs)


def _condensed_index(i, j, n: int):
    """Index into the strict upper triangle for ``i < j``; ``i`` may be an array.

    Examples
    --------
    >>> _condensed_index(0, 1, 3), _condensed_index(0, 2, 3), _condensed_index(1, 2, 3)
    (0, 1, 2)
    """
    return n * i - i * (i + 1) // 2 + j - i - 1
