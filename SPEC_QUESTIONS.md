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
   triangle (`k=1`, 13,125,126 entries) is safe. **What remains is a
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

   Confirmed end to end on sub-100307's 16.9 GB file: unit mass goes from
   0.999999999985 to 0.999999999876, a relative change of 1.1e-10, and the
   68x68 Desikan matrix computed on the fsLR file with the pipeline's own fsLR
   annotation matches the one computed on ico4 at **r = 0.9996**, with a 2.7%
   relative Frobenius difference (on the legacy toolkit example subject first:
   r = 0.9997 and 2.3%). The two calculations share no code and no
   label file. Plain `nibabel` reads the result back with both axes as
   `BrainModelAxis`, 32,492 vertices per hemisphere, and the matrix symmetric.

   **One thing still to ratify: the size, and the grayordinate convention.**
   The exchange file covers the full 32k mesh, 32,492 vertices per hemisphere,
   so a dense connectome is 64,984 x 64,984 = **16.9 GB in float32** -- about
   340 times sub-100307's SC file (50 MB with its endpoints), and two orders of
   magnitude above the
   "one subject, SC + FC, ~100 MB" in the brief's five-minute test. That size
   was accepted deliberately rather than worked around: the alternatives were
   a coarser exchange density, storing only the upper triangle (which
   Workbench does not expect for a `.dconn`), or shipping the exchange form
   only for the tutorial subject. The remaining question is whether the file
   should instead cover the **59,412 cortical grayordinates** with the medial
   wall excluded, which is HCP's own convention and 14.1 GB. The writer
   currently emits the full mesh, with medial-wall rows and columns zero.

5. ~~**Bundled surface geometry.**~~ **Answered.** Four geometries are bundled
   in the wheel at ico4 resolution, 368 kB in total, so `plot()` needs no
   download: the inflated, white and pial surfaces, built by
   `tools/build_surfaces.py` from FreeSurfer's fsaverage, and the sphere,
   converted by `tools/convert_surfaces.py` from the toolkit's
   `?h_sphere_avg_ico4.vtk`, the pipeline's own frame (item 11 records how the
   pial surface, which the toolkit does not ship at ico4, was downsampled).

6. **The data release.** ~~Do the HCP terms permit redistributing derived
   connectomes openly?~~ **Answered: it depends on the dataset.** The group's
   own `sbci-brain/SBCI_Datasets` repository declines for its cohorts:

   > Due to the size and DUAs, we can't publish the SBCI connectomes here.
   > Please contact Dr. Zhang (zhengwu_zhang@unc.edu) if you want to use
   > connectomes produced by SBCI.

   The WU-Minn HCP Young Adult open-access data's Open Access Data Use Terms
   allow derived data to be redistributed, provided they travel under the
   same terms.

   **Decided (30 September): the example cohort is eleven HCP Young Adult
   subjects, hosted by the lab on a public Google Drive.** Ten were drawn at
   random (seed 0), five women and five men, from the 946 young adults with
   complete SBCI pipeline output in the lab's copy; the eleventh, sub-100307,
   was added the same day because the brief's acceptance test names it, so
   `sbci download hcp-ya --subject 100307` now works as the brief writes it.
   All were rebuilt on the ico4
   grid from the pipeline's snapped streamline endpoints on each subject's
   FreeSurfer-registered sphere (`tools/build_hcp_cohort.py`). Each is one SC file with its endpoints and, since 2
   October, one FC file; the lab's copy holds no pipeline FC of the young
   adults, so the FC is built from the HCP's own cleaned resting-state runs
   (`tools/build_hcp_fc.py`; USAGE.md, *Functional connectivity from the HCP's
   resting state*). `sbci download hcp-ya` fetched them then from the Drive ids in `src/sbci/data/hcp_ya.json`, verified each
   file's SHA-256 and wrote `DATA_USE.txt` beside them: the terms, and the
   acknowledgment a publication must carry. The manifest carries only
   open-access fields, sex and the HCP's age band, and the files' metadata
   carry no age. The folder,
   https://drive.google.com/drive/folders/1gG2ZmxxVm4w5dvlCQvMaEEOBypU7nDpx,
   held the twenty-two files and the same `DATA_USE.txt`, shared with anyone who
   had the link; since 15 October the package fetches them from UNC Dataverse
   (below). `sbci download hcp-ya` fetched and verified the files from a
   machine with no Google account, and Drive's own SHA-256 of sub-100307
   matched the digest of the file built on Longleaf, and `scripts/check_hcp_ya.py` runs the main
   methods on the download, from `to_atlas` to `reduce` and `local_test` (the loader, test-retest
   and export are `scripts/audit_api.py`'s). The five-minute test runs against the first subject when
   asked (`SBCI_DOWNLOAD=1`), and CI's `five-minute-start` job does so from a
   blank environment on every push.

   **Decided (15 October 2026): all 946 are published on UNC Dataverse, and
   the package fetches every file from there.** The dataset,
   <https://doi.org/10.15139/S3/NDAJYX>, version 1.0, holds each subject's SC
   and FC in 40 tar archives of at most 50 subjects, the eleven above in
   `sc_tutorial.tar` and `fc_tutorial.tar` as the bytes released on Drive,
   with the HCP's terms as the dataset's terms and a guestbook asking for
   "I agree". The package ships an index of the version
   (`src/sbci/data/hcp_ya.json`, written by `tools/dataverse_deposit.py index`
   from the published version's own listing): for every file its archive, the
   offset of its bytes there, its size and SHA-256. `sbci download hcp-ya`
   fetches the eleven by default, any of the 946 by id as byte ranges, or all
   of them with `--all`, and asks once a machine for the terms to be accepted
   (`--accept-terms` in a script); Google Drive is no longer read. PORTING.md
   item 30 records the release and its checks.

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
     re-smoothed. On sub-100307 the endpoints add **22.2 MB** to a 28.0 MB
     file, so most users should not carry them.
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
   `$SBCI_TOOLKIT/concon_estimate`, `~/.cache/sbci/lbo` or `~/.sbci/lbo`, or
   fetched when it is in none of them (below).
   Recomputing it from the bundled white surface was tried and rejected -- the
   cotangent Laplacian there reproduces the reference kernel only to r = 0.94,
   with a maximum absolute difference of 0.89 against a kernel maximum of 0.94.

   **Decided (16 October 2026): fetched from SBCI_Toolkit when first needed.**
   Bundling it would add about 100 MB to a wheel whose other data is about
   2.8 MB. The two files are public in the SBCI group's SBCI_Toolkit on
   GitHub, so `sbci.download.fetch_basis` reads them there at one commit,
   a558fe8, the one the comparisons with MATLAB were run with, and checks each
   against its SHA-256. The first `rdk` or `matern` smoothing on a machine
   fetches them into `$SBCI_LBO_DIR`, else `~/.cache/sbci/lbo`, where
   `.smooth()` finds them again, and `sbci basis` fetches them ahead of time
   (on a cluster, on a login node). With no network, `.smooth()` says why the
   fetch failed and where it looked (PORTING.md item 31).

   **Verified.** `ContinuousConnectome.smooth(kernel="rdk")` reproduces
   MATLAB's `reference_density.mat` to **3.14 float32-eps**, correlation
   0.99999999999994, with the diagonal exactly zero and unit mass
   1.000000000014. The bandwidth a user gets by passing nothing is the one the
   reference run used, to 5.0e-08. `tests/test_matlab_reference.py` holds both
   within 10 float32-eps, and `tests/reference/smooth_matlab_probe.py` prints
   them (PORTING.md item 24).

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
    surface to a sphere and smooths with a truncated spherical-harmonic kernel
    that is not the heat kernel, despite the name: weight
    `(2l+1)^(3/2)/sqrt(4 pi)`, compact support (PORTING.md item 6). That is what `sbci_step5_structural.sh` invokes, and it produced
    every released `smoothed_sc_avg_*.mat`. The **Riemannian diffusion kernel**
    (bioRxiv 2025.09.08.674789, since published: Wang, Li and Zhang, *Imaging
    Neuroscience* 3, 2025, doi:10.1162/imag.a.912) uses Laplace-Beltrami eigenfunctions on the
    cortical surface itself and explicitly critiques the spherical approach for
    the distortion the projection introduces.

    **How much they differ -- corrected twice.** An earlier draft of this
    document said the two agree at r = 0.58, and a later one at r = 0.29. Both
    figures, tabulated below, compared the Riemannian estimate of the legacy
    example subject with its released `smoothed_sc_avg_0.005_ico4.mat`. That
    file was made by the reference pipeline's own route, so the comparison
    mixed the kernels' difference with everything else that differs between
    the two routes.

    From the same streamline endpoints the kernels agree better, though still
    far from fully. Those endpoints were the HCP-YA tutorial subject's 803,741
    streamlines, as its file stores them, with the medial wall masked:
    - the spherical kernel at sigma = 0.005 gives back the released file
      exactly (r = 1.000000);
    - the Riemannian kernel was run at each of its ten candidate bandwidths;
    - r is over the 10,972,270 cortical vertex pairs, and over the Schaefer-200
      masses.

    These numbers were measured on 17 October 2026 (job 5477553, `r8_quick.py`
    in `/work/users/x/y/xya/paper-open-items/`):

    | kappa | vertex pairs | Schaefer-200 masses |
    | --- | --- | --- |
    | 0.625 | 0.544 | 0.905 |
    | 0.913 | 0.561 | 0.908 |
    | 1.334 | 0.598 | 0.913 |
    | **1.948** (reference) | **0.676** | **0.922** |
    | 2.846 | 0.796 | 0.932 |
    | 4.157 | 0.900 | 0.942 |
    | 6.071 | 0.930 | 0.950 |
    | 8.868 | 0.877 | 0.944 |
    | 12.953 | 0.771 | 0.906 |
    | 18.920 | 0.642 | 0.823 |

    At the reference bandwidths the two estimates share under half their
    variance at vertex resolution (r = 0.68). The regional masses agree far
    better (0.92). Agreement peaks at kappa = 6.07, three times the reference
    value, and falls on either side. So the two kernels still differ
    substantially at vertex resolution at the bandwidths anyone would use,
    which makes recording which one produced a file more important, not less.

    For the record, the earlier comparison against the released file, over
    the strict upper triangle it stores:

    | kappa | r |
    | --- | --- |
    | 0.625 | 0.220 |
    | 1.334 | 0.245 |
    | **1.948** (reference) | **0.286** |
    | 4.157 | 0.464 |
    | 8.868 | 0.581 |
    | 12.953 | 0.573 |
    | 18.920 | 0.517 |

    Its rise and fall is the blur ceiling PORTING.md item 6 describes:
    agreement peaks well past the working bandwidth, where both matrices have
    been smoothed into near-uniformity.

    **The decision:** both ship, the kernel used is recorded in the metadata,
    and the spherical heat kernel is the default. Consequences:

    - Released cohorts stay canonical; nothing is reprocessed.
    - `smooth()` defaults to `kernel="shk"`. `shk` means the **spherical heat
      kernel**, not the Matern kernel -- see the naming note below.
    - The two are not comparable, so a cohort must not mix kernels. `sbci
      validate` checks one file and records the kernel, and `sbci.load_cohort`
      (`sbci cohort`) refuses a cohort whose files mix kernels or bandwidths.
    - The two kernels take different bandwidth parameters: the spherical kernel
      takes `sigma` (0.005 in the released files) and the Riemannian kernel
      takes `kappa` (about 1.95). `bandwidth` alone is therefore ambiguous in
      the metadata and must be read together with `kernel`.

    **Naming correction -- applied.** The brief's API specifies
    `smooth(kernel="rdk" or "shk")`. `shk` had been implemented as the *Matern*
    kernel, because `compute_matern_kernel_matrix.m` is what the toolkit ships.
    Under this decision `sbci.smoothing.KERNELS` is now
    `("shk", "rdk", "matern")`: `shk` is the spherical heat kernel and is the
    default, and the Matern kernel is `matern`. The spherical kernel is shipped
    and is the default: it reproduces `c3_main` at r = 1.00000000 and scale
    1.000000 on all 803,741 streamlines of the tutorial subject, once the grid's
    south pole, which the binary never writes, is emptied -- PORTING.md item 6.

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

13. **Which face list do stored triangle indices refer to?** **Answered, and it was wrong.**

    The optional `/endpoints` group stores `triangle_in` and `triangle_out`, a
    zero-based triangle index over both hemispheres (the right hemisphere's
    offset by 5,120). Nothing in the spec said *which*
    face list those index, and the package shipped surfaces whose faces held the
    same 10,240 triangles in a different order from the pipeline grid's. Every
    stored index therefore resolved to an unrelated triangle -- a median 85 degrees
    away -- and nothing caught it, because no test ever resolved a stored index
    against the bundled mesh.

    It surfaced only when `smooth(kernel="shk")` was wired up and produced a
    density uncorrelated with the reference, where the same kernel matched it
    from the continuous positions `c3_main` was fed. Reordering the bundled
    faces to the grid's own order brought the rebuilt endpoints to within about
    a degree of those positions (on one of the lab's ADNI subjects, whose
    numbers PORTING.md item 39 says are not given).

    **The convention, now fixed:** triangle indices refer to the face list of
    `?h_grid_avg_ico4.m` and its matching `.vtk`, which carry identical order --
    left hemisphere first, then right with 5,120 added. That list is what the
    bundled surfaces carry, all four sharing it, and
    `tests/test_surface.py::test_the_face_order_is_the_one_triangle_indices_mean`
    pins it by digest.

    **Is that order canonical, or one run's accident?** Checked without any
    `.tsv`: a file stores both the nearest vertex and the crossing triangle, so
    the position rebuilt from the triangle must sit within one edge (about 4
    degrees on ico4) of that vertex. Against the bundled face list, the
    toolkit's own example subject resolves at a median 1.62 degrees, and five
    of the lab's ADNI subjects resolve within the edge as well (PORTING.md
    item 39 says why their numbers are not given). Six subjects from two
    pipeline sources, one face order -- it is canonical.

    What is *not* in that order is the toolkit's
    `example_data/fsaverage_label/lh_grid_avg_ico4.vtk` export, nor its
    `?h_sphere_avg_ico4.vtk`: the same triangles, differently ordered. Both
    surface tools read faces from those exports, which is exactly how the
    bundled surfaces came out wrong. They now take faces from one
    digest-checked source, `tools/convert_surfaces.canonical_faces()`, and
    `tests/reference/triangle_consistency.py` re-runs the six-subject check.

    **What WP1 owes:** this belongs in the written format spec, not only in a
    test. A file written by another implementation against a different face order
    would be silently wrong in exactly the way this package was.


14. **Must a re-smoothed file carry nothing on the medial wall?** **Conflict.**

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
    `test_smooth_method_matches_matlab`, which holds the rdk port within 10
    float32-eps of MATLAB (3.14 measured), and would make the shk port diverge from `c3_main` too. Fidelity
    to the reference was judged worth more than the convenience of an output that
    validates unaided.

    **What WP1 owes:** a decision on where masking belongs. The options are to
    apply it on write (which makes a saved file diverge from what was computed),
    to apply it at import from the legacy `.mat` (where `tools/import_legacy.py`
    already could), or to relax the validator for re-smoothed files. Until then,
    `smooth(..., mask_medial_wall=True)` is the caller's explicit way to take the
    format's side; the default stays faithful to the references.
