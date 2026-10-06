"""The exchange file: a CIFTI-2 dense connectome on fsLR-32k.

The point of this form is that a collaborator opens it in Connectome Workbench
or reads it with plain ``nibabel``, without installing this package.

Resampling
----------
The connectome is a **density**: the specification validates unit mass as
``area @ D @ area == 1``, with area counted in fsaverage vertices. Moving a
density to another mesh means taking an area-weighted average over the source
vertices covering each target vertex, not distributing each source vertex's
value among them. The distinction is not cosmetic -- distributing preserves the
plain sum of the entries while losing 99.3% of the area-weighted mass, because
fsLR-32k has 12.7 times more vertices per hemisphere than the ico4 grid.

The package therefore bundles the raw **overlap matrix** ``S``, built by
``tools/build_resampling.py``: ``S[k, i]`` counts the fsaverage vertices
belonging to both ico4 vertex ``i`` and fsLR vertex ``k``. Every fsaverage
vertex is counted once, so the column sums are the ico4 vertex areas -- they
reproduce the area vector in the HDF5 file exactly -- and the row sums are the
matching fsLR areas. Resampling uses the row-normalized form ``P``, and
``area' @ (P D P') @ area'`` then equals ``area @ D @ area`` identically.

The correspondence composes ``mapping_avg_ico4.npz`` from ico4 to fsaverage,
verified at 99.9% against FreeSurfer's annotation, with HCP's
``fs_LR-deformed_to-fsaverage`` spheres from fsaverage to fsLR. Pushing the
Desikan atlas through it reproduces the pipeline's own ``fs_LR.aparc.annot``
for 94.3% of vertices, the residual being boundary vertices at a 12.7-fold jump
in resolution.

Verified end to end on the example subject: the overlap's column sums are
identical to the area vector in the HDF5 file, unit mass survives the move at a
relative change of 1.1e-10, and the 68-region Desikan matrix computed on the
written fsLR file with the pipeline's own fsLR annotation matches the one
computed on ico4 at r = 0.9997. The plain sum of the entries is *not*
preserved, and should not be: it rises by a factor of about 161, which is what
averaging a density onto a finer mesh does.

Size
----
A dense connectome over both hemispheres at 32k density is 64,984 x 64,984,
which is **16.9 GB in float32** -- roughly 160 times the computational file.
That is inherent to the format at this resolution and was accepted
deliberately; see ``SPEC_QUESTIONS.md`` item 4. Writing one needs a machine
with around 25 GB of memory, so it belongs in a batch job rather than on a
login node.

Because the values are a density, integrating them needs the fsLR vertex
areas. :func:`write_cifti` writes those beside the connectome as a companion
``.dscalar.nii``, so the exchange file can be parcellated without this package.

A functional connectome goes through the same operator. Its entries are
Pearson correlations, and because ``P`` is row-stochastic, ``P D P'`` gives
each fsLR vertex pair the area-weighted mean of the ico4 correlations covering
it -- taken directly, not through Fisher z. The stored ico4 diagonal is zero
by the file format, and ``P D P'`` would spread that zero to every pair of
fsLR vertices falling in the same ico4 cell -- with 12.7 fsLR vertices to a
cell, most share theirs with another -- so those pairs would read a
correlation of exactly 0 beside neighbours at 0.6-0.8. The diagonal of cortex
is therefore set to the self-correlation, 1, before an FC resampling, and a
same-cell pair then carries 1. The medial wall has no FC, so its diagonal stays
zero with the rest of its rows; set to 1 there too, its fsLR vertices would
read a correlation of 1 with each other. A density's diagonal stays zero, as
the format has it. The sidecar's ``exchange_values`` says which of the two a
file holds (:data:`EXCHANGE_VALUES`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .. import spec
from ..metadata import Metadata, MetadataError

N_FSLR_PER_HEMI = 32492
N_FSLR = 2 * N_FSLR_PER_HEMI

INTENT_DENSE = 3001
"""NIfTI intent code of a ``.dconn.nii`` (``NIFTI_INTENT_CONNECTIVITY_DENSE``)."""
INTENT_DENSE_SCALARS = 3006
"""NIfTI intent code of a ``.dscalar.nii`` (``NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS``)."""

EXCHANGE_VALUES = {
    "sc": "density per fsaverage vertex squared",
    "fc": (
        "Pearson correlation, unitless; each fsLR vertex pair carries the area-weighted "
        "mean of the ico4 correlations covering it, averaged directly (no Fisher z), with "
        "the ico4 self-correlation taken as 1 on cortex, so two fsLR vertices in one cortical "
        "ico4 cell read 1; the medial wall has no FC and reads 0"
    ),
}
"""What the ``.dconn.nii`` entries are, by modality: the sidecar's ``exchange_values``."""

_SUFFIXES = (".dconn.nii.gz", ".dconn.nii", ".nii.gz", ".nii")
"""Exchange-file suffixes, longest first; what :func:`write_cifti` strips to name the companions."""

_RESAMPLING_UNAVAILABLE = (
    "The ico4 to fsLR-32k overlap matrix is not bundled. Regenerate it with "
    "tools/build_resampling.py, which needs HCP's standard_mesh_atlases."
)


@lru_cache(maxsize=1)
def load_overlap():
    """The sparse ico4 <-> fsLR-32k overlap matrix, shape ``(64984, 5124)``.

    Entries are counts of fsaverage vertices, so the matrix is symmetric in
    meaning: column sums are ico4 vertex areas, row sums are fsLR vertex areas,
    and both total 327,684.
    """
    from scipy import sparse

    path = resources.files("sbci.data.resampling") / "ico4_to_fslr32k.npz"
    if not path.is_file():
        raise NotImplementedError(_RESAMPLING_UNAVAILABLE)
    return sparse.load_npz(str(path))


def vertex_areas() -> tuple[np.ndarray, np.ndarray]:
    """Vertex areas on both meshes, in fsaverage vertices: ``(fslr, ico4)``."""
    overlap = load_overlap()
    return (
        np.asarray(overlap.sum(axis=1)).ravel(),
        np.asarray(overlap.sum(axis=0)).ravel(),
    )


def transfer():
    """The row-normalized resampling operator ``P``, shape ``(64984, 5124)``.

    Each row sums to one, so ``P D P'`` is an area-weighted average of the
    density and conserves ``area @ D @ area``.
    """
    from scipy import sparse

    overlap = load_overlap()
    fslr_area = np.asarray(overlap.sum(axis=1)).ravel()
    # A target vertex with no source is possible in principle; leave its row
    # at zero rather than dividing by zero.
    scale = np.divide(1.0, fslr_area, out=np.zeros_like(fslr_area), where=fslr_area > 0)
    return (sparse.diags(scale) @ overlap).tocsr()


def _brain_model_axis():
    """A CIFTI brain model covering both fsLR-32k cortical surfaces."""
    from nibabel import cifti2

    left = cifti2.BrainModelAxis.from_surface(
        np.arange(N_FSLR_PER_HEMI), N_FSLR_PER_HEMI, name="cortex_left"
    )
    right = cifti2.BrainModelAxis.from_surface(
        np.arange(N_FSLR_PER_HEMI), N_FSLR_PER_HEMI, name="cortex_right"
    )
    return left + right


def resample(dense: np.ndarray, block: int = 8192) -> np.ndarray:
    """Resample an ico4 density onto fsLR-32k, conserving area-weighted mass.

    Computed in row blocks straight into the output array, because the
    intermediate of a naive ``P D P'`` would double the peak memory.

    Parameters
    ----------
    dense
        Symmetric ``(5124, 5124)`` connectivity density.
    block
        Rows of the output computed at a time. Lower it if memory is tight.
    """
    operator = transfer()
    if dense.shape != (operator.shape[1],) * 2:
        raise ValueError(
            f"expected a {operator.shape[1]}x{operator.shape[1]} matrix, got {dense.shape}"
        )

    # P D is (64984, 5124): 2.7 GB in float64, which is affordable. The square
    # that follows is not, so it is filled a block of rows at a time.
    half = operator @ np.asarray(dense, dtype=np.float64)
    n = operator.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        # This is half[start:stop] @ P.T, written the other way round because
        # scipy multiplies sparse-by-dense efficiently and dense-by-sparse not.
        out[start:stop] = (operator @ half[start:stop].T).T
    return out


def companion_stem(name: str) -> str:
    """The exchange file's name without its CIFTI suffix, which names the sidecar and the areas.

    Only the suffix comes off, so ``sub-01.ses-1_sc.dconn.nii`` and
    ``sub-01.ses-2_fc.dconn.nii`` keep their own sidecars; cutting at the
    first dot gave both ``sub-01.json`` and let the second overwrite the first.

    Examples
    --------
    >>> companion_stem("sub-01.ses-1_sc.dconn.nii")
    'sub-01.ses-1_sc'
    >>> companion_stem("sub-01_fc.dconn.nii.gz")
    'sub-01_fc'
    """
    lowered = name.lower()  # as nibabel writes them, whatever the case
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def write_cifti(path: str | Path, connectome: Any, block: int = 8192) -> Path:
    """Write a ``.dconn.nii`` on fsLR-32k, with metadata and areas beside it.

    Three files are written: the dense connectome, a JSON sidecar carrying the
    metadata table from the manuscript -- CIFTI has nowhere natural to put
    kernel, bandwidth, streamline count, weighting and provenance -- and a
    ``.dscalar.nii`` of fsLR vertex areas, without which the density cannot be
    integrated. The two companions take the file's name with its
    ``.dconn.nii`` removed (:func:`companion_stem`).

    A functional connectome's diagonal is set to 1, the self-correlation, on
    its cortical vertices before resampling; see the module notes.

    The name, the directory and the metadata are checked before anything is
    computed: on the real grid the resampling takes the time and the memory,
    and nibabel would refuse a name only once it was done. The name has to
    end in ``.dconn.nii``, uncompressed, as nibabel writes CIFTI; the metadata
    is validated as :meth:`~sbci.ContinuousConnectome.load` validates it.
    """
    from nibabel import cifti2

    path = Path(path)
    if not path.name.lower().endswith(".dconn.nii"):
        raise ValueError(
            f"{path.name!r} does not end in .dconn.nii, the name of a CIFTI dense connectome; "
            "CIFTI is written uncompressed, so not .dconn.nii.gz either"
        )
    if not os.access(path.parent, os.W_OK):
        raise OSError(f"cannot write to {path.parent}")
    connectome.metadata.validate()
    stem = companion_stem(path.name)
    # Built first: metadata the sidecar cannot describe should fail here, not
    # after the dense file has gone to disk.
    metadata = sidecar(connectome.metadata.fields, stem)
    dense = connectome.dense()
    if connectome.metadata.fields["included_connections"] == "fc":
        # Zero is the format's placeholder, not a correlation: left in, every
        # fsLR pair inside one ico4 cell would read 0 beside neighbours at 0.7.
        # Cortex only: the medial wall has no FC, and keeps its zeros.
        cortex = np.flatnonzero(connectome.mask)
        dense[cortex, cortex] = 1.0
    resampled = resample(dense, block=block)

    axis = _brain_model_axis()
    image = cifti2.Cifti2Image(resampled, (axis, axis))
    # nibabel leaves the intent at "unknown CIFTI" unless told; Workbench keys
    # the file type on it.
    image.nifti_header.set_intent(INTENT_DENSE)
    image.to_filename(str(path))

    (path.parent / f"{stem}.json").write_text(metadata)

    fslr_area, _ = vertex_areas()
    from nibabel.cifti2 import ScalarAxis

    areas = cifti2.Cifti2Image(
        fslr_area[None, :].astype(np.float32), (ScalarAxis(["vertex area"]), axis)
    )
    areas.nifti_header.set_intent(INTENT_DENSE_SCALARS)
    areas.to_filename(str(path.parent / f"{stem}_vertexarea.dscalar.nii"))
    return path


def read_cifti(path: str | Path) -> dict[str, Any]:
    """Read a dense connectome and resample it back to the ico4 grid.

    Not implemented. The inverse is lossy by construction -- 64,984 values
    cannot be recovered from 5,124 -- so a round trip would silently degrade
    the data. If reading exchange files is wanted, it needs a decision about
    what "back to ico4" should mean.
    """
    raise NotImplementedError(
        "Reading CIFTI back to ico4 is not implemented: the resampling is "
        "many-to-one, so the inverse is lossy and needs a defined convention. "
        "Use the HDF5 computational file for round trips."
    )


def sidecar(fields: dict, stem: str) -> str:
    """The JSON written beside the exchange file: the connectome's metadata plus the exchange keys.

    ``exchange_values`` says what the entries are, which depends on the modality
    (``included_connections``): a density for SC, correlations for FC. Metadata that
    does not name the modality is refused rather than labelled a density by default.

    Serialized through :meth:`Metadata.to_json`, which writes NumPy scalars as the numbers
    they are; ``json.dumps`` alone would refuse a ``streamline_count`` held as ``np.int64``
    after the 17 GB ``.dconn.nii`` had already been written.
    """
    fields = dict(fields)
    modality = fields.get("included_connections")
    if modality not in EXCHANGE_VALUES:
        raise MetadataError(
            f"included_connections is {modality!r}, so the sidecar cannot say what the "
            f"values are; expected one of {tuple(EXCHANGE_VALUES)}"
        )
    fields["exchange_space"] = spec.EXCHANGE_SPACE
    fields["exchange_density"] = spec.EXCHANGE_DENSITY
    fields["exchange_values"] = EXCHANGE_VALUES[modality]
    fields["exchange_vertex_areas"] = f"{stem}_vertexarea.dscalar.nii"
    return Metadata(dict(sorted(fields.items()))).to_json(indent=2)
