"""The WP1 file specification, as constants.

Everything in the package reads its conventions from here so that a change to
the specification is a change to one file.

.. warning::
   ``SPEC_VERSION`` is a *draft*. WP1 has not frozen the specification yet, and
   the open questions are listed in ``SPEC_QUESTIONS.md`` at the repository
   root. Nothing here should be treated as final until that document is empty
   and ``SPEC_VERSION`` loses its ``-draft`` suffix.
"""

from __future__ import annotations

SPEC_VERSION = "0.1.0-draft"
"""Version of the on-disk format this build of the package reads and writes."""

# --- Computational grid (WP1) ---------------------------------------------

GRID = "fsaverage-ico4"
"""Canonical computational grid. The ``_0.99`` / ``_0.94`` reduction-fraction
grids and the 4121-vertex alignment grid are retired and are not read."""

N_VERTICES_PER_HEMI = 2562
N_VERTICES = 5124
"""Total vertices, left hemisphere first, then right."""

N_FACES_PER_HEMI = 5120
N_FACES = 10240

AREA_TOTAL = 327_684
"""Sum of the per-vertex area weights on the canonical grid.

Areas are counted in fsaverage vertices (163,842 per hemisphere), so they are
integers per ico4 vertex and total the whole fsaverage sphere; the ico4-to-fsLR
overlap matrix has the same column sums (SPEC_QUESTIONS.md item 3).
"""
"""Triangles of the grid mesh, left hemisphere first, then right."""

HEMISPHERE_ORDER = ("L", "R")

# --- Storage convention ----------------------------------------------------

TRIANGLE = "upper"
DIAGONAL_INCLUDED = False
"""Connectivity is stored as the strict upper triangle (k=1), float32.

The self-connectivity diagonal is excluded: the legacy ``parcellate_sc.m``
removes it before aggregating, so carrying it would change every downstream
number. See ``SPEC_QUESTIONS.md`` item 2 -- WP1 must confirm.
"""

DTYPE = "float32"

N_UPPER = N_VERTICES * (N_VERTICES - 1) // 2
"""Length of the condensed connectivity vector: 13,122,006 float32 ~= 50 MB."""

# --- Streamline endpoints --------------------------------------------------

ENDPOINTS_GROUP = "endpoints"
"""Optional group in the computational file holding the streamline endpoints
that ``ContinuousConnectome.smooth()`` re-smooths from.

Endpoints live in the same file as the connectivity rather than in a sibling,
so that a connectome and the endpoints it was built from cannot be separated,
mismatched or versioned apart. The group is optional: a file without it is
valid and simply cannot be re-smoothed.

Vertex and triangle indices are **global and zero-based** -- 0 to 5123 over the
two hemispheres for vertices, 0 to 10239 for triangles -- so the hemisphere is
implied by the index and cannot disagree with a separate field.

The barycentric datasets are optional within the group. They give each
endpoint's continuous position inside its triangle, which a kernel evaluated
off-grid needs; without them an endpoint is only known to the nearest vertex.
"""

ENDPOINT_DATASETS = ("vertex_in", "vertex_out")
"""Required inside :data:`ENDPOINTS_GROUP`."""

ENDPOINT_OPTIONAL_DATASETS = (
    "triangle_in",
    "triangle_out",
    "barycentric_in",
    "barycentric_out",
)
"""Present together or not at all."""

# --- Exchange format (WP1) -------------------------------------------------

EXCHANGE_SPACE = "fsLR"
EXCHANGE_DENSITY = "32k"
EXCHANGE_SUFFIX = ".dconn.nii"

BIDS_PATTERN = (
    "sub-{sub}/ses-{ses}/connectivity/sub-{sub}_space-fsLR_den-32k_desc-concon_{modality}.dconn.nii"
)

# --- Metadata contract -----------------------------------------------------

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "grid",
    "hemisphere_order",
    "mask",
    "area_weights",
    "storage_convention",
    "included_connections",
    "streamline_count",
    "streamline_weighting",
    "normalization",
    "kernel",
    "bandwidth",
    "fc_nuisance_model",
    "registration_reference",
    "pipeline_version",
    "container_version",
)
"""Keys every file must carry. ``ContinuousConnectome.load`` refuses a file
that is missing any of them -- WP2 requirement, not a warning."""

SC_ONLY_KEYS = ("streamline_count", "streamline_weighting", "kernel", "bandwidth")
FC_ONLY_KEYS = ("fc_nuisance_model",)

MODALITIES = ("sc", "fc")

ENTRY_POINTS = ("raw", "preprocessed", "fastpath", "precomputed")
