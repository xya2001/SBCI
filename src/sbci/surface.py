"""Cortical surface meshes on the ico4 computational grid.

The three geometries the toolkit ships at ico4 resolution are bundled with the
package (about 290 KB), so plotting needs no VTK library, no FreeSurfer, and no
network. They are generated once by ``tools/convert_surfaces.py``.

Each mesh covers both hemispheres on the 5124-vertex grid, left first, in the
same vertex order as a :class:`~sbci.ContinuousConnectome`. A surface map can
therefore be handed straight to :func:`sbci.plotting.plot_surface` without any
reindexing.

The anatomical surfaces are in fsaverage's RAS space, so a figure drawn on
them can be read anatomically. ``tests/test_surface_anatomy.py`` checks that
named regions land where they belong, which is the only test that catches a
mesh carrying the wrong labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

from . import spec

SUFFIX = "_ico4.npz"

GEOMETRIES = ("inflated", "white", "pial", "sphere")
"""Bundled geometries. ``inflated`` is the default, since a folded surface
hides most of the cortex inside sulci."""

ANATOMICAL = ("inflated", "white", "pial")
"""Those carrying real anatomy, in fsaverage's RAS space.

Built by ``tools/build_surfaces.py`` from FreeSurfer's fsaverage through the
correspondence in ``mapping_avg_ico4.npz``, which assigns each of the 327,684
full-resolution vertices to one of the 5124 grid vertices. The majority
annotation label over each group agrees with the bundled atlases 99.9% of the
time, which is what establishes that the correspondence is right.

``sphere`` is not anatomical: it is the registered spherical parameterization,
useful for seeing the whole cortex at once but carrying no anatomy.
"""

NATIVE_SPACE = ("white", "pial")
"""Those whose coordinates are true anatomical positions.

``inflated`` carries anatomy in the sense that each vertex is the right
cortical point, but its coordinates are a deformation: the hemispheres are each
centred on their own origin and overlap, so absolute position no longer means
medial or lateral. Left-right geometry is tested on this tuple; anterior and
superior ordering on all of :data:`ANATOMICAL`.
"""


@dataclass(frozen=True)
class Surface:
    """A triangulated mesh on the computational grid.

    Attributes
    ----------
    name
        Geometry name, one of :data:`GEOMETRIES`.
    vertices
        ``(5124, 3)`` coordinates, left hemisphere first.
    faces
        ``(10240, 3)`` triangles indexing :attr:`vertices`.
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def n_vertices(self) -> int:
        """Vertices in the mesh, matching the connectome grid."""
        return int(self.vertices.shape[0])

    def hemisphere(self, side: str) -> Surface:
        """One hemisphere as a mesh in its own right.

        Plotting libraries draw a single hemisphere at a time, so the faces are
        re-indexed from zero here.

        Parameters
        ----------
        side
            ``"L"`` or ``"R"``.
        """
        side = side.upper()
        if side not in spec.HEMISPHERE_ORDER:
            raise ValueError(f"side must be one of {spec.HEMISPHERE_ORDER}, got {side!r}")

        half = self.n_vertices // 2
        if side == "L":
            keep = self.faces.max(axis=1) < half
            return Surface(self.name, self.vertices[:half], self.faces[keep])

        keep = self.faces.min(axis=1) >= half
        return Surface(self.name, self.vertices[half:], self.faces[keep] - half)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Surface {self.name} with {self.n_vertices} vertices, {len(self.faces)} faces>"


@lru_cache(maxsize=len(GEOMETRIES))
def load_surface(name: str = "inflated") -> Surface:
    """Load a bundled surface mesh.

    Parameters
    ----------
    name
        One of :data:`GEOMETRIES`.

    Examples
    --------
    >>> from sbci import load_surface
    >>> surface = load_surface("inflated")
    >>> surface.n_vertices
    5124
    >>> surface.faces.shape
    (10240, 3)
    >>> surface.hemisphere("L").n_vertices
    2562
    """
    if name not in GEOMETRIES:
        raise ValueError(f"unknown surface {name!r}; bundled geometries are {GEOMETRIES}.")

    path = resources.files("sbci.data.surfaces") / f"{name}{SUFFIX}"
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)

    if vertices.shape[0] != spec.N_VERTICES:
        raise ValueError(f"{name}: {vertices.shape[0]} vertices, expected {spec.N_VERTICES}")
    return Surface(name=name, vertices=vertices, faces=faces)
