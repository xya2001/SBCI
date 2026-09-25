"""Cortical surface meshes on the ico4 computational grid.

Four geometries at ico4 resolution are bundled with the package (about 376 KB),
so plotting needs no VTK library, no FreeSurfer, and no network. They are
generated once by ``tools/build_surfaces.py`` from FreeSurfer's fsaverage.

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
    if faces.shape != (spec.N_FACES, 3) or faces.min() < 0 or faces.max() >= spec.N_VERTICES:
        raise ValueError(f"{name}: faces are {faces.shape}, expected {(spec.N_FACES, 3)} in range")
    # The cached arrays are shared by every caller: freeze them.
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return Surface(name=name, vertices=vertices, faces=faces)


SHADING_PASSES = 4
"""Neighbourhood-averaging passes behind :func:`sulcal_depth`: about a
centimetre of smoothing at ico4 resolution, the scale of a sulcus."""


def vertex_normals(vertices, faces) -> np.ndarray:
    """Unit outward normals at the vertices: area-weighted means of the face normals."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)


@lru_cache(maxsize=1)
def sulcal_depth() -> np.ndarray:
    """A sulcal-depth map over the grid, for shading figures.

    Positive in sulci and negative on gyral crowns, like FreeSurfer's ``sulc``,
    so that drawn in grayscale under a surface map it darkens the fundi and
    lets the folds show through. It is the signed distance, along the outward
    normal, from each white-surface vertex to the mean of its neighbourhood
    after :data:`SHADING_PASSES` passes of neighbour averaging -- cheap, needs
    no FreeSurfer, and at ico4 resolution follows the gyral pattern well
    enough to shade a figure. In millimetres, roughly -3 to 3; the returned
    array is read-only and shared.

    Examples
    --------
    >>> from sbci.surface import sulcal_depth
    >>> depth = sulcal_depth()
    >>> depth.shape
    (5124,)
    >>> bool(0.3 < (depth > 0).mean() < 0.7)   # sulci and gyri in comparable measure
    True
    """
    from scipy import sparse

    white = load_surface("white")
    vertices = np.asarray(white.vertices, dtype=np.float64)
    faces = np.asarray(white.faces, dtype=np.int64)
    n = vertices.shape[0]
    rows = faces[:, [0, 1, 2, 0, 1, 2]].ravel()
    cols = faces[:, [1, 2, 0, 2, 0, 1]].ravel()
    adjacency = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0  # an edge shared by two faces counts once
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    averaging = sparse.diags(1.0 / np.maximum(degree, 1)) @ adjacency
    smoothed = vertices
    for _ in range(SHADING_PASSES):
        smoothed = averaging @ smoothed
    depth = ((smoothed - vertices) * vertex_normals(vertices, faces)).sum(axis=1)
    depth.setflags(write=False)
    return depth
