# Open questions for WP1

WP2 cannot freeze its file reader until these are answered. Each one is a place
where `src/sbci/spec.py` currently carries a **provisional** value, marked
`SPEC_VERSION = "0.1.0-draft"`. When this file is empty, the suffix comes off.

1. **HDF5 dataset names and layout.** The brief specifies "upper-triangular
   float32 array, plus datasets for area weights, mask, vertex coordinates, and
   a JSON metadata string" but not the names. This package currently writes
   `/connectivity`, `/area`, `/mask`, `/coordinates`, `/metadata`. Confirm, or
   name them otherwise -- after the first file is released this cannot change
   without a format version bump.

2. ~~**Does the stored triangle include the diagonal?**~~ **Answered by the
   pipeline.** `SBCI_Py3/scripts/concon/convert_raw.py` writes the released
   file as `np.triu(kernel)` -- `k=0`, so the diagonal slot is kept. Checked
   against `smoothed_sc_avg_0.005_ico4.mat`: the lower triangle is all zero and
   **every one of the 5124 diagonal entries is exactly zero**. The slot exists
   and is empty.

   So nothing is lost either way for SC, and this package's strict upper
   triangle (`k=1`, 13,122,006 entries) is safe. **What remains is a
   convention choice**, not a data question: adopt `k=1` and state that the
   diagonal is never stored, or adopt `k=0` to mirror the pipeline byte for
   byte. FC is the case that would decide it, since `import_legacy.py` found
   ones on the FC diagonal.

3. **Units and normalization.** "Unit mass" is validated as
   `area @ D @ area == 1`, and is now checked for SC only: FC is a field of
   correlations, not a density, and its area-weighted total is an arbitrary
   number near zero. Confirm that, and confirm the weighting -- is SC mass
   integrated against vertex areas, or already area-normalized at write time?
   Confirmed against the legacy example subject: both SC and FC are written as
   the strict upper triangle with an empty lower triangle, SC with a zero
   diagonal and FC with ones on it. Vertex areas are the count of
   high-resolution fsaverage vertices per ico4 vertex, summing to 327,684.

4. ~~**The ico4 <-> fsLR-32k resampling.**~~ **Built and vendored.**

   The correspondence is `src/sbci/data/resampling/ico4_to_fslr32k.npz`, 211
   KB, built by `tools/build_resampling.py` and bundled in the wheel, so
   `to_cifti()` needs no download. It composes `mapping_avg_ico4.npz` from ico4
   to fsaverage, which item 11 verified at 99.9% against FreeSurfer's
   annotation, with HCP's `fs_LR-deformed_to-fsaverage` spheres from fsaverage
   to fsLR.

   What is stored is the raw **overlap matrix** `S`, where `S[k, i]` counts the
   fsaverage vertices belonging to both ico4 vertex `i` and fsLR vertex `k`.
   Every fsaverage vertex is counted once, so `S` totals 327,684 along both
   axes: its column sums are ico4 vertex areas and its row sums are fsLR vertex
   areas, both in the same unit the pipeline uses.

   **The connectome is a density, and that decides the normalization.** Item 3
   fixes unit mass as `area @ D @ area == 1`, so resampling means an
   area-weighted *average* -- `S` normalized by its row sums -- not a
   *distribution* of each vertex's value, which is right for an extensive
   quantity and wrong for this one. The difference is not cosmetic: the
   distributing form preserves the plain sum of the entries while losing
   **99.3%** of the area-weighted mass, because fsLR-32k carries 12.7 times
   more vertices per hemisphere. Storing `S` un-normalized leaves both readings
   available and assumes neither.

   Two independent checks, neither of them self-consistency:

   - The column sums reproduce the area vector stored in the example subject's
     HDF5 file **exactly** -- maximum difference 0 -- so the overlap matrix
     partitions the same vertices the pipeline's own areas do.
   - Pushing the Desikan atlas through it reproduces the pipeline's own
     `lh/rh.fs_LR.aparc.annot` for **94.3%** (left) and **94.4%** (right) of
     vertices, the residual being boundary vertices at a 12.7-fold jump in
     resolution.

   Confirmed end to end on the example subject's 16.9 GB file: unit mass goes
   from 1.00000000002 to 0.999999999908, a relative change of 1.1e-10, and the
   68x68 Desikan matrix computed on the fsLR file with the pipeline's own fsLR
   annotation matches the one computed on ico4 at **r = 0.9997**, with a 2.3%
   relative Frobenius difference. The two calculations share no code and no
   label file. Plain `nibabel` reads the result back with both axes as
   `BrainModelAxis`, 32,492 vertices per hemisphere, and the matrix symmetric.

   **One thing still to ratify: the size, and the grayordinate convention.**
   The exchange file covers the full 32k mesh, 32,492 vertices per hemisphere,
   so a dense connectome is 64,984 x 64,984 = **16.9 GB in float32** -- about
   160 times the computational file, and two orders of magnitude above the
   "one subject, SC + FC, ~100 MB" in the brief's five-minute test. That size
   was accepted deliberately rather than worked around: the alternatives were
   a coarser exchange density, storing only the upper triangle (which
   Workbench does not expect for a `.dconn`), or shipping the exchange form
   only for the tutorial subject. The remaining question is whether the file
   should instead cover the **59,412 cortical grayordinates** with the medial
   wall excluded, which is HCP's own convention and 14.1 GB. The writer
   currently emits the full mesh, with medial-wall rows and columns zero.

5. ~~**Bundled surface geometry.**~~ **Answered.** The toolkit already ships
   `lh/rh_inflated_avg_lps_ico4.vtk`, `_white_`, and `_sphere_` under
   `example_data/fsaverage_label`. All three are now converted by
   `tools/convert_surfaces.py` and bundled in the wheel, 290 KB in total, so
   `plot()` needs no download. One caveat for the record: **pial is not
   available at ico4 resolution**, only inflated, white and sphere. If a pial
   view is wanted it has to be downsampled from the full-resolution fsaverage
   surface and added to the converter.

6. **The data release.** ~~Do the HCP terms permit redistributing derived
   connectomes openly?~~ **Answered, and the answer is no.** The group's own
   `sbci-brain/SBCI_Datasets` repository says it plainly:

   > Due to the size and DUAs, we can't publish the SBCI connectomes here.
   > Please contact Dr. Zhang (zhengwu_zhang@unc.edu) if you want to use
   > connectomes produced by SBCI.

   So `sbci download hcp-ya` **cannot work without authentication**, and the
   brief's five-minute test cannot pass as written -- its second line assumes
   an open download. The criterion itself needs rewording.

   **What is still needed**, and is now a decision rather than a question: a
   tutorial subject that *can* be released. It does not have to be HCP. One
   subject's smoothed connectome is 20 MB, carries no identifiable data, and
   would make the five-minute test real. Either designate one, or change the
   acceptance criterion to start from a file the user already holds.

7. **License holder.** `LICENSE` says "The SBCI developers". Replace with the
   institutional holder before the first tagged release.

8. ~~**The format carries no endpoints.**~~ **DECIDED: in the same file, as
   an optional group.** Implemented.

   The brief's API re-smooths "from stored endpoints", and the choice was
   between a sibling `sub-XXXX_endpoints.h5` and a group inside the
   computational file. Same file wins: a connectome and the streamlines it was
   built from then cannot be separated, mismatched or versioned apart, and
   `.smooth()` needs no second path argument.

   ```
   /endpoints/vertex_in         int32   (S,)   global, zero-based, 0..5123
   /endpoints/vertex_out        int32   (S,)
   /endpoints/triangle_in       int32   (S,)   optional, 0..10239
   /endpoints/triangle_out      int32   (S,)
   /endpoints/barycentric_in    float32 (S, 3)
   /endpoints/barycentric_out   float32 (S, 3)
   ```

   Three properties worth stating, because they are conventions and not
   accidents:

   - **The group is optional.** A file without it is valid and simply cannot be
     re-smoothed. On the example subject the endpoints add **11.8 MB** to a
     20.0 MB file, so most users should not carry them.
   - **Indices are global**, over the whole 5124-vertex grid and the whole
     10,240-triangle mesh, so the hemisphere is read off the index and cannot
     contradict a separate field.
   - **The barycentric datasets are optional within the group**, and present
     together or not at all. They give each endpoint's continuous position
     inside its triangle; without them an endpoint is known only to the nearest
     vertex, which moves it 1.5 degrees on average.

   FC files are refused endpoints: a field of correlations has no streamlines.

   **`smooth()` still needs one thing the file does not carry.** The
   Laplace-Beltrami basis, `EV_LBO_ds_ico4_{L,R}.mat`, is 50 MB per hemisphere.
   It is a property of the grid rather than of the subject, so duplicating it
   into every file would be wrong and bundling it in the wheel is too large;
   it is passed as `eigenpairs=`, or found automatically in `$SBCI_LBO_DIR`,
   `$SBCI_TOOLKIT/concon_estimate`, `~/.cache/sbci/lbo` or `~/.sbci/lbo`.
   Recomputing it from the bundled white surface was tried and rejected -- the
   cotangent Laplacian there reproduces the reference kernel only to r = 0.94,
   with a maximum absolute difference of 0.89 against a kernel maximum of 0.94.

   **Open: where the basis should be distributed from.** Bundling it would add
   about 52 MB to a wheel whose other data is 700 KB, so it needs the same
   answer item 6 needs about hosting. Until then `.smooth()` works wherever the
   toolkit is checked out, and says exactly where it looked when it is not.

   **Verified.** `ContinuousConnectome.smooth(kernel="rdk")` reproduces
   MATLAB's `reference_density.mat` to **3.25 float32-eps**, correlation
   0.999999999999921, with the diagonal exactly zero and unit mass
   1.000000000014. The bandwidth a user gets by passing nothing is the one the
   reference run used, to 5e-08. Both are asserted in
   `tests/test_matlab_reference.py`.

9. ~~**Which normalization is the released one?**~~ **Answered: none.**
   `SBCI_Py3/scripts/concon/convert_raw.py` reads the raw `c3_main` output,
   swaps the hemispheres back into left-then-right order, and saves. It applies
   no scaling of any kind. That is why `smoothed_sc_avg_0.005_ico4.mat` sums to
   5.49e6 and matches neither of the two normalizations in play -- **the
   released cohorts are unnormalized kernel output**, and normalization is
   whatever each downstream analysis applies.

   **The decision that remains** is what *this package* should write.
   `tools/import_legacy.py` normalizes to area-weighted unit mass, which is
   what `sbci validate` checks and what makes `to_atlas` and `to_cifti`
   comparable across subjects. That is the right choice, but it means a file
   written here is not byte-comparable with a released `.mat`, and the
   metadata must say so -- `normalization: "unit-mass"` already does.

10. ~~**Which smoother do the released cohorts use?**~~ **DECIDED: ship both,
    with the spherical heat kernel as the default.**

    The two are not a mismatch; they are two methods from two papers. The
    **spherical heat kernel** (Moyer et al., *A Continuous Model of Cortical
    Connectivity*, MICCAI 2016; code at `dcmoyer/concon`) inflates the cortical
    surface to a sphere and smooths with a truncated spherical-harmonic heat
    kernel. That is what `sbci_step5_structural.sh` invokes, and it produced
    every released `smoothed_sc_avg_*.mat`. The **Riemannian diffusion kernel**
    (bioRxiv 2025.09.08.674789) uses Laplace-Beltrami eigenfunctions on the
    cortical surface itself and explicitly critiques the spherical approach for
    the distortion the projection introduces.

    **How much they differ -- a correction.** An earlier draft of this document
    said the two agree at r = 0.58. That figure is the *best* agreement
    reachable by tuning the Riemannian bandwidth, at kappa = 8.87. At the
    bandwidth the reference selection formula actually picks, kappa = 1.948,
    they agree at **r = 0.29**. Measured on the example subject against the
    released `smoothed_sc_avg_0.005_ico4.mat`, over the strict upper triangle
    the file stores:

    | kappa | r |
    | --- | --- |
    | 0.625 | 0.220 |
    | 1.334 | 0.245 |
    | **1.948** (reference) | **0.286** |
    | 4.157 | 0.464 |
    | 8.868 | 0.581 |
    | 12.953 | 0.573 |
    | 18.920 | 0.517 |

    The rise and fall is the same blur ceiling PORTING.md item 6 describes:
    agreement peaks well past the working bandwidth, where both matrices have
    been smoothed into near-uniformity. So the two kernels disagree
    substantially at the bandwidths anyone would use, which makes recording
    which one produced a file more important, not less.

    **The decision:** both ship, the kernel used is recorded in the metadata,
    and the spherical heat kernel is the default. Consequences:

    - Released cohorts stay canonical; nothing is reprocessed.
    - `smooth()` defaults to `kernel="shk"`. `shk` means the **spherical heat
      kernel**, not the Matern kernel -- see the naming note below.
    - The validator must refuse to mix kernels within a cohort, since the two
      are not comparable.
    - The two kernels take different bandwidth parameters: the spherical kernel
      takes `sigma` (0.005 in the released files) and the Riemannian kernel
      takes `kappa` (about 1.95). `bandwidth` alone is therefore ambiguous in
      the metadata and must be read together with `kernel`.

    **Naming correction -- applied.** The brief's API specifies
    `smooth(kernel="rdk" or "shk")`. `shk` had been implemented as the *Matern*
    kernel, because `compute_matern_kernel_matrix.m` is what the toolkit ships.
    Under this decision `sbci.smoothing.KERNELS` is now
    `("shk", "rdk", "matern")`: `shk` is the spherical heat kernel and is the
    default, and the Matern kernel is `matern`. The spherical kernel is ported
    but not shipped yet. Against the correct reference it reaches **r =
    0.978**; a 20% amplitude error remains, traced to an undocumented
    normalization or sampling convention -- PORTING.md item 6. Until that
    closes, `smooth()` names the gap rather than silently substituting a
    different kernel.

11. ~~**No anatomical surface can be drawn.**~~ **RESOLVED.**

    The ico4 grid corresponds to fsaverage through `mapping_avg_ico4.npz`,
    which assigns each of the 327,684 full-resolution fsaverage vertices to one
    of the 5124 grid vertices. FreeSurfer's icosahedra are nested, so the same
    correspondence appears as fsaverage's first 2562 vertices per hemisphere.
    Both were checked against FreeSurfer's own annotation: the majority label
    over each grid vertex's members agrees with the bundled atlases **99.9%**
    of the time, and the nested reading agrees 99.3%.

    Inflated, white and pial are now bundled, built by
    `tools/build_surfaces.py` as the mean fsaverage position over each grid
    vertex's members, with faces from the grid mesh. They are in fsaverage's
    RAS space, so figures can be read anatomically.

    **Why the earlier attempt failed -- and a correction to the diagnosis.**
    The first rebuild was anatomically scrambled and still passed every check
    applied to it: mean edge 5.6, Desikan contiguity 0.8385, plausible extents.
    Those measure whether a mesh is self-consistent, and a consistent
    relabelling preserves all of them.

    The cause was not, as first recorded, that the toolkit's anatomical meshes
    are in the wrong vertex order. Re-measured on every ico4 mesh the toolkit
    ships, drawing the Desikan parcellation with grid-order labels on each
    mesh's own faces:

    | Mesh | Desikan edge contiguity |
    | --- | --- |
    | `lh/rh_grid_avg_ico4.vtk` | 0.8405 |
    | `lh/rh_sphere_avg_ico4.vtk` | 0.8405 |
    | `lh/rh_white_avg_ico4.vtk` | 0.8405 |
    | `lh/rh_inflated_avg_ico4.vtk` | 0.8405 |
    | `template_{sphere,inflated,white}_grid_ico4.mat` | 0.8405 |
    | `lh/rh_white_avg_lps_ico4.vtk` | **0.2794** |
    | `lh/rh_inflated_avg_lps_ico4.vtk` | **0.2794** |

    Only the `_lps_` variants are out of step, and the first rebuild read those
    -- the suffix reads like the pipeline's coordinate convention. They are not
    a re-orientation of the plain files and not even a permutation of them: as
    unordered point clouds the two differ by up to 65 mm for white and 37 mm
    for inflated, so they are a different discretization altogether.

    **The rebuild is independently confirmed.** The toolkit's own
    `white_avg_ico4.vtk`, once its x and y are negated for the LPS-to-RAS
    convention, agrees with the rebuilt white surface to **0.26 mm median** --
    two separate constructions of the same mesh landing on top of each other.

    What catches it is a claim about the world: the superior frontal gyrus is
    anterior to lateral occipital cortex, the cuneus is medial, the medial wall
    faces the midline. `tests/test_surface_anatomy.py` now asserts these on
    every bundled anatomical surface, and `tools/build_surfaces.py` refuses to
    write if agreement with fsaverage falls below 95%.

12. ~~**Which grid should ENCORE use?**~~ **Not an issue -- withdrawn.**
    This was a misreading. `SphericalGrid(mesh, l)` takes the mesh as a
    constructor argument, and the library classes -- `Encore.m`, `Concon.m`,
    `SphericalGrid.m`, `SphericalWarp.m`, the KDE and the tangent basis --
    reference no grid file at all. Only four demo scripts (`pca_test.m`,
    `pop_template.m`, `register_test.m`, `upsample_example.m`) hardcode
    `lh_grid_avg_0.94.vtk`, and that is just the data those demos ship with.
    ENCORE works on whatever spherical mesh it is handed, ico4 included, so
    there is nothing to decide before porting it.


## 11. Which face list do stored triangle indices refer to?  **ANSWERED, and it was wrong**

The optional `/endpoints` group stores `triangle_in` and `triangle_out`, a
zero-based triangle index per hemisphere. Nothing in the spec said *which*
face list those index, and the package shipped surfaces whose faces held the
same 10,240 triangles in a different order from the pipeline grid's. Every
stored index therefore resolved to an unrelated triangle -- a median 85 degrees
away -- and nothing caught it, because no test ever resolved a stored index
against the bundled mesh.

It surfaced only when `smooth(kernel="shk")` was wired up and produced a
density uncorrelated with the reference (r = 0.236 where the same kernel
scores 1.000000 from continuous positions). Reordering the bundled faces to
the grid's own order brought the rebuilt endpoints from 83.1 to 1.25 degrees
of the positions `c3_main` was fed.

**The convention, now fixed:** triangle indices refer to the face list of
`?h_grid_avg_ico4.m` and its matching `.vtk`, which carry identical order --
left hemisphere first, then right with 5,120 added. That list is what the
bundled surfaces carry, all four sharing it, and
`tests/test_surface.py::test_the_face_order_is_the_one_triangle_indices_mean`
pins it by digest.

**What WP1 owes:** this belongs in the written format spec, not only in a
test. A file written by another implementation against a different face order
would be silently wrong in exactly the way this package was.


## 12. Must a re-smoothed file carry nothing on the medial wall?  **CONFLICT**

The format says the medial wall carries no connectivity, and `sbci validate`
enforces it. Neither reference produces such a file:

| source | mass on the medial wall |
| --- | --- |
| `c3_main`'s released `smoothed_sc_avg_0.005_ico4.mat` | 0.868% of total |
| MATLAB's `reference_density.mat` (rdk) | present |

Streamlines really do terminate near the wall, and neither smoother discards
them, so a density re-smoothed from stored endpoints inherits that mass and
fails the validator -- even though it reproduces the reference exactly.

`smooth()` therefore does **not** mask. Zeroing the wall inside it would break
`test_smooth_method_matches_matlab`, which pins the rdk port to MATLAB at 3.25
float32-eps, and would make the shk port diverge from `c3_main` too. Fidelity
to the reference was judged worth more than the convenience of an output that
validates unaided.

**What WP1 owes:** a decision on where masking belongs. The options are to
apply it on write (which makes a saved file diverge from what was computed),
to apply it at import from the legacy `.mat` (where `tools/import_legacy.py`
already could), or to relax the validator for re-smoothed files. Until then, a
caller who needs a file that validates has to mask it themselves.
