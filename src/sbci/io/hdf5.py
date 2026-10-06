"""The computational file: ``sub-XXXX_{sc,fc}.h5``.

Layout (WP1 draft)::

    /connectivity                float32 (N_UPPER,)  strict upper triangle
    /area                        float64 (N,)        per-vertex area weight
    /mask                        bool    (N,)        True where the vertex is cortex
    /coordinates                 float32 (N, 3)      vertex coordinates [optional]
    /metadata                    str     scalar      JSON string, keys per spec.py
    /endpoints/vertex_in         int32   (S,)        streamline endpoints [optional]
    /endpoints/vertex_out        int32   (S,)
    /endpoints/triangle_in       int32   (S,)        continuous position [optional]
    /endpoints/triangle_out      int32   (S,)
    /endpoints/barycentric_in    float32 (S, 3)
    /endpoints/barycentric_out   float32 (S, 3)

One file per subject per modality. FC uses the identical layout with signed
values, and carries no endpoints.

Endpoints live here rather than in a sibling file so that a connectome cannot
be separated from the streamlines it was built from; see ``SPEC_QUESTIONS.md``
item 8. They roughly double the size of an SC file -- 383,760 streamlines add
about 12 MB compressed -- so the group is optional and a file without it is
valid, it simply cannot be re-smoothed.

Vertex and triangle indices are global and zero-based over the whole grid, so
the hemisphere is implied by the index rather than stored beside it: a vertex
index is split at half the file's own grid, as its connectivity and area count
it, and a triangle index at the grid's 5120 faces per hemisphere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import grid, spec
from ..errors import FormatError, InvalidFileError
from ..metadata import Metadata

CONNECTIVITY = "connectivity"
AREA = "area"
MASK = "mask"
COORDINATES = "coordinates"
METADATA = "metadata"
ENDPOINTS = spec.ENDPOINTS_GROUP


def _vertices_per_hemisphere(handle) -> int:
    """Half the file's grid, where its whole-grid vertex indices change hemisphere.

    Counted from the file's own arrays rather than taken from the ico4 grid,
    so the connectivity and the area have to agree on the count, and it has
    to be even.
    """
    n_area = int(handle[AREA].size)
    try:
        n = grid.n_from_condensed(int(handle[CONNECTIVITY].size))
    except ValueError as exc:
        raise InvalidFileError(f"/{CONNECTIVITY}: {exc}") from exc
    if n != n_area:
        raise InvalidFileError(
            f"/{ENDPOINTS} cannot be split into hemispheres: the connectivity is for {n} "
            f"vertices but the area for {n_area}"
        )
    if n % 2:
        raise InvalidFileError(
            f"/{ENDPOINTS} cannot be split into hemispheres: the grid has {n} vertices, "
            "an odd number"
        )
    return n // 2


def _read_endpoints(handle) -> Any:
    """Rebuild :class:`sbci.smoothing.Endpoints` from the optional group."""
    if ENDPOINTS not in handle:
        return None
    from ..smoothing import Endpoints

    group = handle[ENDPOINTS]
    missing = [name for name in spec.ENDPOINT_DATASETS if name not in group]
    if missing:
        raise FormatError(f"/{ENDPOINTS} is present but has no {', '.join(missing)}")

    sizes = {name: group[name].shape[0] for name in spec.ENDPOINT_DATASETS}
    if len(set(sizes.values())) != 1:
        raise InvalidFileError(f"/{ENDPOINTS} datasets disagree on the streamline count: {sizes}")

    optional: dict[str, np.ndarray] = {}
    present = [name for name in spec.ENDPOINT_OPTIONAL_DATASETS if name in group]
    if present and len(present) != len(spec.ENDPOINT_OPTIONAL_DATASETS):
        raise InvalidFileError(
            f"/{ENDPOINTS} carries {present} but positions need all of "
            f"{list(spec.ENDPOINT_OPTIONAL_DATASETS)}"
        )
    if present:
        optional = {name: np.asarray(group[name][()]) for name in present}
        for name in spec.ENDPOINT_OPTIONAL_DATASETS:
            if optional[name].shape[0] != sizes["vertex_in"]:
                raise InvalidFileError(f"/{ENDPOINTS}/{name} disagrees on the streamline count")

    n_per_hemi = _vertices_per_hemisphere(handle)
    try:
        return Endpoints.from_global(
            vertex_in=np.asarray(group["vertex_in"][()]),
            vertex_out=np.asarray(group["vertex_out"][()]),
            n_per_hemi=n_per_hemi,
            n_faces_per_hemi=spec.N_FACES_PER_HEMI,
            **optional,
        )
    except ValueError as exc:
        raise InvalidFileError(f"/{ENDPOINTS}: {exc}") from exc


def _write_endpoints(handle, endpoints, opts, n_vertices: int) -> None:
    """Write the optional endpoint group, in global zero-based indices."""
    # The reader splits the indices at half the file's grid: endpoints on
    # another grid would come back in the wrong hemisphere, or not at all.
    if 2 * endpoints.n_per_hemi != n_vertices:
        raise ValueError(
            f"the endpoints are on a grid of {2 * endpoints.n_per_hemi} vertices but the "
            f"connectome on one of {n_vertices}"
        )
    group = handle.create_group(ENDPOINTS)
    group.create_dataset(
        "vertex_in", data=np.asarray(endpoints.global_vertex_in, dtype=np.int32), **opts
    )
    group.create_dataset(
        "vertex_out", data=np.asarray(endpoints.global_vertex_out, dtype=np.int32), **opts
    )
    if not endpoints.has_positions:
        return

    faces = spec.N_FACES_PER_HEMI
    offset = type(endpoints).global_hemisphere_offset
    group.create_dataset(
        "triangle_in",
        data=(
            np.asarray(endpoints.tri_in, dtype=np.int64) + offset(endpoints.surf_in, faces)
        ).astype(np.int32),
        **opts,
    )
    group.create_dataset(
        "triangle_out",
        data=(
            np.asarray(endpoints.tri_out, dtype=np.int64) + offset(endpoints.surf_out, faces)
        ).astype(np.int32),
        **opts,
    )
    group.create_dataset(
        "barycentric_in", data=np.asarray(endpoints.bary_in, dtype=np.float32), **opts
    )
    group.create_dataset(
        "barycentric_out", data=np.asarray(endpoints.bary_out, dtype=np.float32), **opts
    )


def read_hdf5(path: str | Path) -> dict[str, Any]:
    """Read a computational file into its raw parts, without validating.

    :meth:`sbci.ContinuousConnectome.load` is the public entry point; this
    function exists so the validator can inspect a file that fails to load.
    """
    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:  # h5py's wording names C-level details; say what matters
        raise InvalidFileError(f"{path.name} cannot be opened as an HDF5 file") from exc
    with handle:
        for required in (CONNECTIVITY, AREA, MASK, METADATA):
            if required not in handle:
                raise FormatError(f"{path.name} has no /{required} dataset")

        raw = handle[METADATA][()]
        if not isinstance(raw, (str, bytes)):
            raise InvalidFileError(f"{path.name}: /{METADATA} is not a JSON string")
        parts: dict[str, Any] = {
            "data": np.asarray(handle[CONNECTIVITY][()], dtype=np.float32),
            "area": np.asarray(handle[AREA][()], dtype=np.float64),
            "mask": np.asarray(handle[MASK][()], dtype=bool),
            "metadata": Metadata.from_json(raw),
        }
        parts["coords"] = (
            np.asarray(handle[COORDINATES][()], dtype=np.float32) if COORDINATES in handle else None
        )
        parts["endpoints"] = _read_endpoints(handle)
    return parts


def read_header(path: str | Path) -> dict[str, Any]:
    """A computational file's metadata and the sizes of its arrays, without reading the arrays.

    What :func:`sbci.load_cohort` reads of every file: enough to say whether it
    would load and how it was made, at a small fraction of a full read. Returns
    ``metadata`` (unvalidated), ``n_connectivity``, ``n_vertices`` (the area's
    length), ``n_mask`` and ``n_endpoints`` (0 without the endpoints group).
    """
    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:  # h5py's wording names C-level details; say what matters
        raise InvalidFileError(f"{path.name} cannot be opened as an HDF5 file") from exc
    with handle:
        for required in (CONNECTIVITY, AREA, MASK, METADATA):
            if required not in handle:
                raise FormatError(f"{path.name} has no /{required} dataset")
        raw = handle[METADATA][()]
        if not isinstance(raw, (str, bytes)):
            raise InvalidFileError(f"{path.name}: /{METADATA} is not a JSON string")
        first = spec.ENDPOINT_DATASETS[0]
        group = handle[ENDPOINTS] if ENDPOINTS in handle else None
        return {
            "metadata": Metadata.from_json(raw),
            "n_connectivity": int(handle[CONNECTIVITY].size),
            "n_vertices": int(handle[AREA].size),
            "n_mask": int(handle[MASK].size),
            "n_endpoints": int(group[first].size) if group is not None and first in group else 0,
        }


def write_hdf5(
    path: str | Path,
    data: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray,
    metadata: Metadata,
    coords: np.ndarray | None = None,
    endpoints: Any = None,
    compression: str | None = "gzip",
) -> Path:
    """Write a computational file.

    The metadata is validated before anything is written, so a run that forgot
    to record its bandwidth fails here rather than producing a file that the
    loader will later refuse.

    ``endpoints`` is a :class:`sbci.smoothing.Endpoints`; pass it to make the
    file re-smoothable. It is refused on a functional connectome, which has no
    streamlines, and when its vertices per hemisphere are not half the
    connectome's grid, where the reader will split it.
    """
    metadata.validate()
    if endpoints is not None and metadata.modality != "sc":
        raise ValueError(f"endpoints belong to a structural connectome, not {metadata.modality!r}")
    # Serialize before the file is opened: a value the encoder refuses must not
    # leave a truncated file behind.
    text = metadata.to_json()
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"directory does not exist: {path.parent}")
    # The byte-shuffle filter costs nothing to read and improves gzip on floats.
    opts = {"compression": compression, "shuffle": True} if compression else {}

    # Written beside the destination and moved into place once complete, so a
    # failure part-way (an endpoint array of the wrong kind, a full disk) leaves
    # neither a truncated file nor a destroyed previous one.
    partial = path.with_name(path.name + ".partial")
    try:
        with h5py.File(partial, "w") as handle:
            handle.create_dataset(CONNECTIVITY, data=np.asarray(data, dtype=np.float32), **opts)
            handle.create_dataset(AREA, data=np.asarray(area, dtype=np.float64), **opts)
            handle.create_dataset(MASK, data=np.asarray(mask, dtype=bool), **opts)
            if coords is not None:
                handle.create_dataset(
                    COORDINATES, data=np.asarray(coords, dtype=np.float32), **opts
                )
            if endpoints is not None:
                _write_endpoints(handle, endpoints, opts, np.size(area))
            handle.create_dataset(METADATA, data=text)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path
