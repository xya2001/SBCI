"""Carrying a warp from the grid's sphere to another template's sphere.

ENCORE and ConSEAL estimate a warp of the sphere the grid lives on: the image
of every grid vertex, with a Jacobian. Data registered to another template --
HCP's fs_LR for MSMAll-aligned surfaces, or fsaverage at full resolution --
live on a different sphere, related to the grid's by a registration. The same
deformation on that sphere is the conjugate ``g o phi o g^-1``: carry a
template vertex back to the grid's sphere, apply the warp, carry the result
forward. Every step here is a barycentric lookup between two spherical meshes
that share a vertex set, which is what :class:`SphereMap` is.

Two frames are involved, and they are not the same:

- The grid's bundled ``sphere`` is the pipeline's own parameterization. Its
  vertices are fsaverage vertices -- vertex ``i`` of a hemisphere is
  fsaverage's vertex ``i``, as the pipeline's ``mapping_avg_ico4.npz`` lists
  them -- but their sphere coordinates are not those of FreeSurfer's standard
  sphere: the two differ by 119 degrees on median. ``fsaverage_sphere_ico4.npz``
  holds each grid vertex's standard-sphere coordinates, and
  :func:`grid_to_fsaverage` is the map between the two parameterizations.
  (Until October 2026 the file matched the vertices on the inflated surface,
  which picked a neighbouring fsaverage vertex for 27 of them, half a degree
  off.)
- HCP's ``fs_LR-deformed_to-fsaverage`` sphere places every fs_LR-32k vertex
  on FreeSurfer's standard sphere, and its standard ``sphere.32k_fs_LR`` gives
  the same vertices' fs_LR positions; together they are
  :func:`fsaverage_to_fslr`. MSMSulc and MSMAll differ per subject, not in the
  group sphere, so a group warp lands on fs_LR either way; a subject's own
  MSMAll sphere pair can be appended as one more :class:`SphereMap`.

The migrated warp is returned on the target template's vertices and can be
written as a deformed sphere in GIFTI, the form in which Connectome Workbench
takes a registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import spec
from .alignment import MeshQuery, _npz_path, normalize_rows, voronoi_areas
from .surface import load_surface

#: Templates a warp can be carried to.
TEMPLATES = ("fsaverage", "fs_LR_32k")
#: Radius HCP and FreeSurfer spheres are written with.
SPHERE_RADIUS = 100.0
#: GIFTI intent codes of a surface's two arrays, ``NIFTI_INTENT_POINTSET`` and
#: ``NIFTI_INTENT_TRIANGLE``. The arrays are identified by these, not by their
#: order in the file, which a valid file may reverse.
INTENT_POINTSET = 1008
INTENT_TRIANGLE = 1009


def _surface_arrays(image, path) -> tuple[np.ndarray, np.ndarray]:
    """A GIFTI surface's coordinates and triangles, found by intent.

    Only when no array declares either intent is the conventional order --
    coordinates, then triangles -- assumed, and then the dtypes have to agree
    with it.
    """
    points = [array for array in image.darrays if array.intent == INTENT_POINTSET]
    triangles = [array for array in image.darrays if array.intent == INTENT_TRIANGLE]
    if len(points) == 1 and len(triangles) == 1:
        return points[0].data, triangles[0].data
    if not points and not triangles and len(image.darrays) == 2:
        first, second = (np.asarray(array.data) for array in image.darrays)
        if np.issubdtype(first.dtype, np.floating) and np.issubdtype(second.dtype, np.integer):
            return first, second
    raise ValueError(
        f"{path} does not hold one coordinate array and one triangle array: found "
        f"{len(points)} and {len(triangles)} by intent among {len(image.darrays)} arrays; "
        "a GIFTI surface needs one of each"
    )


def _interpolate(points, base, images, faces, query: MeshQuery | None = None):
    """Barycentric transport: where ``points`` on the ``base`` mesh land under ``images``."""
    query = MeshQuery(base, faces) if query is None else query
    weights, indices = query.query(normalize_rows(np.asarray(points, dtype=np.float64)))
    return normalize_rows(np.einsum("nk,nkj->nj", weights, images[indices]))


class SphereMap:
    """A piecewise-linear bijection between two spherical meshes sharing a vertex set.

    ``source`` and ``target`` are the same vertices' unit positions on the two
    spheres, ``faces`` the shared triangles. :meth:`forward` locates a point in
    the source mesh and carries its barycentric weights to the target;
    :meth:`inverse` does the reverse. Maps compose with ``+``.
    """

    def __init__(self, source, target, faces):
        self.source = normalize_rows(np.asarray(source, dtype=np.float64))
        self.target = normalize_rows(np.asarray(target, dtype=np.float64))
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.source.shape != self.target.shape or self.source.shape[1] != 3:
            raise ValueError(
                f"source and target must be the same (n, 3) vertex set, got "
                f"{self.source.shape} and {self.target.shape}"
            )
        self._forward: MeshQuery | None = None
        self._inverse: MeshQuery | None = None

    @classmethod
    def from_gifti(cls, source_path, target_path) -> SphereMap:
        """Two sphere surfaces of one mesh as GIFTI, e.g. a subject's native and MSMAll spheres."""
        import nibabel as nib

        source, faces = _surface_arrays(nib.load(str(source_path)), source_path)
        target, target_faces = _surface_arrays(nib.load(str(target_path)), target_path)
        faces = np.asarray(faces, dtype=np.int64)
        if not np.array_equal(faces, np.asarray(target_faces, dtype=np.int64)):
            raise ValueError("the two spheres do not share a face list, so they are not one mesh")
        return cls(source, target, faces)

    def forward(self, points) -> np.ndarray:
        """Source-sphere points carried to the target sphere."""
        if self._forward is None:
            self._forward = MeshQuery(self.source, self.faces)
        return _interpolate(points, self.source, self.target, self.faces, self._forward)

    def inverse(self, points) -> np.ndarray:
        """Target-sphere points carried back to the source sphere."""
        if self._inverse is None:
            self._inverse = MeshQuery(self.target, self.faces)
        return _interpolate(points, self.target, self.source, self.faces, self._inverse)

    def __add__(self, other) -> SphereChain:
        return SphereChain([self]) + other


class SphereChain:
    """Several :class:`SphereMap` applied in order."""

    def __init__(self, maps):
        self.maps = list(maps)

    def __add__(self, other) -> SphereChain:
        if isinstance(other, SphereChain):
            return SphereChain(self.maps + other.maps)
        return SphereChain(self.maps + [other])

    def forward(self, points) -> np.ndarray:
        """Points carried through every map in order."""
        for step in self.maps:
            points = step.forward(points)
        return np.asarray(points)

    def inverse(self, points) -> np.ndarray:
        """Points carried back through every map in reverse order."""
        for step in reversed(self.maps):
            points = step.inverse(points)
        return np.asarray(points)


@lru_cache(maxsize=1)
def _fsaverage_at_grid() -> np.ndarray:
    """FreeSurfer's standard-sphere coordinates of the grid's vertices, ``(5124, 3)``."""
    path = resources.files("sbci.data.surfaces") / "fsaverage_sphere_ico4.npz"
    with np.load(str(path)) as data:
        return np.asarray(data["vertices"], dtype=np.float64)


@lru_cache(maxsize=1)
def _fslr_spheres() -> dict:
    """HCP's fs_LR-32k spheres: standard, deformed to fsaverage, and the faces, per hemisphere."""
    path = resources.files("sbci.data.templates") / "fslr32k_spheres.npz"
    with np.load(str(path)) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def grid_to_fsaverage() -> tuple[SphereMap, SphereMap]:
    """From the grid's bundled sphere to FreeSurfer's standard fsaverage sphere, per hemisphere."""
    sphere = load_surface("sphere")
    standard = _fsaverage_at_grid()
    half = spec.N_VERTICES_PER_HEMI
    maps = []
    for side, letter in enumerate("LR"):
        hemisphere = sphere.hemisphere(letter)
        block = slice(side * half, (side + 1) * half)
        maps.append(SphereMap(hemisphere.vertices, standard[block], hemisphere.faces))
    return maps[0], maps[1]


def fsaverage_to_fslr() -> tuple[SphereMap, SphereMap]:
    """From FreeSurfer's standard sphere to HCP's fs_LR-32k sphere, per hemisphere."""
    spheres = _fslr_spheres()
    return tuple(
        SphereMap(spheres[f"{h}_deformed"], spheres[f"{h}_standard"], spheres[f"{h}_faces"])
        for h in "LR"
    )


def template_mesh(template: str) -> tuple:
    """The target template's sphere vertices and faces, per hemisphere."""
    if template == "fs_LR_32k":
        spheres = _fslr_spheres()
        return tuple(
            (np.asarray(spheres[f"{h}_standard"], dtype=np.float64), spheres[f"{h}_faces"])
            for h in "LR"
        )
    if template == "fsaverage":
        from .plotting import _fsaverage_files

        files = _fsaverage_files("fsaverage")
        return tuple(
            (normalize_rows(files["geometries"]["sphere"][side]), files["faces"][side])
            for side in (0, 1)
        )
    raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")


def chain_to(template: str) -> tuple[SphereChain, SphereChain]:
    """The maps from the grid's sphere to ``template``, per hemisphere."""
    first = grid_to_fsaverage()
    if template == "fsaverage":
        return SphereChain([first[0]]), SphereChain([first[1]])
    if template == "fs_LR_32k":
        second = fsaverage_to_fslr()
        return first[0] + second[0], first[1] + second[1]
    raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")


@dataclass
class TemplateWarp:
    """A warp carried to another template: the image of each of its sphere vertices.

    ``lh_vertices``/``rh_vertices`` are unit positions on the template's sphere,
    ``lh_jacobian``/``rh_jacobian`` the ratio of each vertex's Voronoi area
    after and before, and ``faces`` the template's triangles.
    """

    template: str
    lh_vertices: np.ndarray
    lh_jacobian: np.ndarray
    rh_vertices: np.ndarray
    rh_jacobian: np.ndarray
    faces: tuple

    def apply(self, points, hemisphere: str) -> np.ndarray:
        """Where template-sphere ``points`` on hemisphere ``"L"`` or ``"R"`` land under the warp."""
        if hemisphere not in ("L", "R"):
            raise ValueError(f"hemisphere must be 'L' or 'R', got {hemisphere!r}")
        side = 0 if hemisphere == "L" else 1
        base, faces = template_mesh(self.template)[side]
        images = (self.lh_vertices, self.rh_vertices)[side]
        return _interpolate(points, base, images, faces)

    def save(self, path) -> Path:
        """Write the warp to ``.npz``; the path returned is the file written, suffix included."""
        path = _npz_path(path)
        np.savez_compressed(
            path,
            template=self.template,
            lh_vertices=self.lh_vertices,
            lh_jacobian=self.lh_jacobian,
            rh_vertices=self.rh_vertices,
            rh_jacobian=self.rh_jacobian,
            lh_faces=self.faces[0],
            rh_faces=self.faces[1],
        )
        return path

    @classmethod
    def load(cls, path) -> TemplateWarp:
        """Read a warp written by :meth:`save`."""
        with np.load(Path(path)) as data:
            return cls(
                template=str(data["template"]),
                lh_vertices=data["lh_vertices"],
                lh_jacobian=data["lh_jacobian"],
                rh_vertices=data["rh_vertices"],
                rh_jacobian=data["rh_jacobian"],
                faces=(data["lh_faces"], data["rh_faces"]),
            )

    def to_gifti(self, prefix) -> tuple[Path, Path]:
        """Write the warp as two deformed sphere surfaces, ``<prefix>.L.sphere.surf.gii`` and R.

        A deformed sphere is how Connectome Workbench takes a registration: it is
        the template's mesh with every vertex moved to where the warp sends it,
        at radius 100 like HCP's own spheres.
        """
        import nibabel as nib
        from nibabel import gifti

        written = []
        for letter, vertices, faces in (
            ("L", self.lh_vertices, self.faces[0]),
            ("R", self.rh_vertices, self.faces[1]),
        ):
            coordinates = gifti.GiftiDataArray(
                np.asarray(vertices * SPHERE_RADIUS, dtype=np.float32),
                intent="NIFTI_INTENT_POINTSET",
                datatype="NIFTI_TYPE_FLOAT32",
            )
            triangles = gifti.GiftiDataArray(
                np.asarray(faces, dtype=np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
                datatype="NIFTI_TYPE_INT32",
            )
            image = gifti.GiftiImage(darrays=[coordinates, triangles])
            path = Path(f"{prefix}.{letter}.sphere.surf.gii")
            nib.save(image, str(path))
            written.append(path)
        return written[0], written[1]


def migrate_warp(warp, to: str = "fs_LR_32k", grid_rotations=None) -> TemplateWarp:
    """Carry an ENCORE or ConSEAL warp of the grid to another template's sphere.

    Parameters
    ----------
    warp
        :class:`~sbci.alignment.Warp` from :func:`sbci.align` or
        :class:`~sbci.conseal.EndpointWarp` from :func:`sbci.endpoints_align`:
        anything with ``lh_vertices`` and ``rh_vertices``, the images of the
        grid's vertices.
    to
        ``"fs_LR_32k"`` (HCP's sphere, where MSMAll-aligned data live) or
        ``"fsaverage"`` (FreeSurfer's standard sphere at full resolution,
        fetched by nilearn on first use).
    grid_rotations
        The rotations of the grid the warp was estimated on, as
        ``Alignment.grid_rotations``: ENCORE works on a copy of the grid
        rotated off the coordinate poles, and its warps are in that frame.
        ``None``, the default, takes them from the warp itself -- an ENCORE
        :class:`~sbci.alignment.Warp` carries them as ``lh_rotation`` and
        ``rh_rotation``, through :meth:`~sbci.alignment.Warp.save` and back
        -- and a warp without them, such as ConSEAL's, which works in the
        grid's own frame, gets identities. Passing them overrides the warp's.

    Returns
    -------
    :class:`TemplateWarp` on the template's vertices, with Jacobians.
    """
    sphere = load_surface("sphere")
    chains = chain_to(to)
    meshes = template_mesh(to)
    if grid_rotations is None:
        grid_rotations = tuple(getattr(warp, f"{letter}h_rotation", None) for letter in "lr")
    rotations = tuple(np.eye(3) if r is None else r for r in grid_rotations)
    out = {}
    for side, letter in enumerate("LR"):
        hemisphere = sphere.hemisphere(letter)
        base = normalize_rows(np.asarray(hemisphere.vertices, dtype=np.float64))
        images = np.asarray(getattr(warp, f"{letter.lower()}h_vertices"), dtype=np.float64)
        images = normalize_rows(images @ np.asarray(rotations[side], dtype=np.float64))
        target_vertices, target_faces = meshes[side]
        back = chains[side].inverse(target_vertices)
        moved = _interpolate(back, base, images, hemisphere.faces)
        landed = chains[side].forward(moved)
        before = voronoi_areas(target_vertices, target_faces)
        after = voronoi_areas(landed, target_faces)
        out[letter] = (landed, after / before)
    return TemplateWarp(
        template=to,
        lh_vertices=out["L"][0],
        lh_jacobian=out["L"][1],
        rh_vertices=out["R"][0],
        rh_jacobian=out["R"][1],
        faces=(meshes[0][1], meshes[1][1]),
    )
