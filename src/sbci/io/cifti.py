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
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

N_FSLR_PER_HEMI = 32492
N_FSLR = 2 * N_FSLR_PER_HEMI

_RESAMPLING_UNAVAILABLE = (
    "The ico4 to fsLR-32k overlap matrix is not bundled. Regenerate it with "
    "tools/build_resampling.py, which needs HCP's standard_mesh_atlases."
)


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
    half = operator @ dense.astype(np.float64)
    n = operator.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        # This is half[start:stop] @ P.T, written the other way round because
        # scipy multiplies sparse-by-dense efficiently and dense-by-sparse not.
        out[start:stop] = (operator @ half[start:stop].T).T
    return out


def write_cifti(path: str | Path, connectome: Any, block: int = 8192) -> Path:
    """Write a ``.dconn.nii`` on fsLR-32k, with metadata and areas beside it.

    Three files are written: the dense connectome, a JSON sidecar carrying the
    metadata table from the manuscript -- CIFTI has nowhere natural to put
    kernel, bandwidth, streamline count, weighting and provenance -- and a
    ``.dscalar.nii`` of fsLR vertex areas, without which the density cannot be
    integrated.
    """
    import json

    from nibabel import cifti2

    path = Path(path)
    resampled = resample(connectome.dense().astype(np.float64), block=block)

    axis = _brain_model_axis()
    cifti2.Cifti2Image(resampled, (axis, axis)).to_filename(str(path))

    stem = path.name.split(".")[0]
    fields = dict(connectome.metadata.fields)
    fields["exchange_space"] = "fsLR"
    fields["exchange_density"] = "32k"
    fields["exchange_values"] = "density per fsaverage vertex squared"
    fields["exchange_vertex_areas"] = f"{stem}_vertexarea.dscalar.nii"
    (path.parent / f"{stem}.json").write_text(json.dumps(fields, indent=2, sort_keys=True))

    fslr_area, _ = vertex_areas()
    from nibabel.cifti2 import ScalarAxis

    cifti2.Cifti2Image(
        fslr_area[None, :].astype(np.float32), (ScalarAxis(["vertex area"]), axis)
    ).to_filename(str(path.parent / f"{stem}_vertexarea.dscalar.nii"))
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
