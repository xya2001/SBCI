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
the hemisphere is implied by the index rather than stored beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import spec
from ..errors import FormatError
from ..metadata import Metadata

CONNECTIVITY = "connectivity"
AREA = "area"
MASK = "mask"
COORDINATES = "coordinates"
METADATA = "metadata"
ENDPOINTS = spec.ENDPOINTS_GROUP


def _read_endpoints(handle) -> Any:
    """Rebuild :class:`sbci.smoothing.Endpoints` from the optional group."""
    if ENDPOINTS not in handle:
        return None
    from ..smoothing import Endpoints

    group = handle[ENDPOINTS]
    missing = [name for name in spec.ENDPOINT_DATASETS if name not in group]
    if missing:
        raise FormatError(f"/{ENDPOINTS} is present but has no {missing}")

    sizes = {name: group[name].shape[0] for name in spec.ENDPOINT_DATASETS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"/{ENDPOINTS} datasets disagree on the streamline count: {sizes}")

    optional: dict[str, np.ndarray] = {}
    present = [name for name in spec.ENDPOINT_OPTIONAL_DATASETS if name in group]
    if present and len(present) != len(spec.ENDPOINT_OPTIONAL_DATASETS):
        raise ValueError(
            f"/{ENDPOINTS} carries {present} but positions need all of "
            f"{list(spec.ENDPOINT_OPTIONAL_DATASETS)}"
        )
    if present:
        optional = {name: np.asarray(group[name][()]) for name in present}
        for name in ("triangle_in", "triangle_out"):
            if optional[name].shape[0] != sizes["vertex_in"]:
                raise ValueError(f"/{ENDPOINTS}/{name} disagrees on the streamline count")

    return Endpoints.from_global(
        vertex_in=np.asarray(group["vertex_in"][()]),
        vertex_out=np.asarray(group["vertex_out"][()]),
        n_per_hemi=spec.N_VERTICES_PER_HEMI,
        n_faces_per_hemi=spec.N_FACES_PER_HEMI,
        **optional,
    )


def _write_endpoints(handle, endpoints, opts) -> None:
    """Write the optional endpoint group, in global zero-based indices."""
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
    with h5py.File(path, "r") as handle:
        for required in (CONNECTIVITY, AREA, MASK, METADATA):
            if required not in handle:
                raise FormatError(f"{path.name} has no /{required} dataset")

        raw = handle[METADATA][()]
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
    streamlines.
    """
    metadata.validate()
    if endpoints is not None and metadata.modality != "sc":
        raise ValueError(f"endpoints belong to a structural connectome, not {metadata.modality!r}")
    path = Path(path)
    opts = {"compression": compression} if compression else {}

    with h5py.File(path, "w") as handle:
        handle.create_dataset(CONNECTIVITY, data=np.asarray(data, dtype=np.float32), **opts)
        handle.create_dataset(AREA, data=np.asarray(area, dtype=np.float64), **opts)
        handle.create_dataset(MASK, data=np.asarray(mask, dtype=bool), **opts)
        if coords is not None:
            handle.create_dataset(COORDINATES, data=np.asarray(coords, dtype=np.float32), **opts)
        if endpoints is not None:
            _write_endpoints(handle, endpoints, opts)
        handle.create_dataset(METADATA, data=metadata.to_json())
    return path
