# How the files were made

The subjects are the $subjects young adults of the HCP 1200 Subjects release with the Zhang
lab's SBCI pipeline output. Every file was built by a tool of the
[sbci](https://github.com/xya2001/SBCI) repository, which also validates the files it
writes. $validated Each file's `metadata` records how it was made, and its `provenance`
names its subject.

## Structural connectivity (`sc/`)

1. **Streamlines.** These come from the Zhang lab's run of the SBCI pipeline (Cole et al.,
   2021) on the HCP's diffusion MRI, the run the ENCORE project used (Cole et al., 2026).
   Each streamline's ends are snapped to the white-matter surface.
2. **Endpoints.** Each end is placed on the subject's FreeSurfer-registered sphere, and so
   on the fsaverage ico4 grid. The endpoints are stored in `endpoints/`: each end's grid
   vertex, triangle and barycentric coordinates. A subject has $streamlines streamlines.
3. **Density.** The endpoint pairs are smoothed on the sphere with the spherical heat
   kernel (`shk`) at bandwidth 0.005. The medial wall is masked and set to zero, and the
   density is normalized to unit mass in the grid's area-weighted inner product. Because
   the endpoints are stored, the density can be smoothed again with another kernel.

Built by `tools/build_hcp_cohort.py`.

## Functional connectivity (`fc/`)

1. **Time series.** These are the HCP's minimally preprocessed, ICA-FIX-cleaned
   resting-state runs on the 32k fs_LR mesh (`rfMRI_REST{1,2}_{LR,RL}_Atlas_hp2000_clean`,
   MSMSulc). Every run available was used, and each file lists its runs in `fc_runs`:
   $runs.
2. **Placement.** Each 32k vertex is located on the subject's native MSMSulc sphere. Its
   barycentric coordinates carry it to the same subject's FreeSurfer sphere. It is then
   assigned to the grid vertex whose fsaverage cell contains the nearest fsaverage vertex.
   This is the registration the SC endpoints were placed through, so SC and FC are
   compared at the same place. $routes
3. **Nuisance.** For each run, the global signal (the mean over all grayordinates), a
   constant and a linear trend are regressed out. No band-pass filter is applied.
4. **Connectivity.** The series of the 32k vertices in a grid vertex's cell are averaged.
   Each run's averages are z-scored and the runs concatenated. The FC is the Pearson
   correlation between grid vertices. A cortical grid vertex whose cell holds no 32k
   vertex takes the nearest one, and the medial wall is zero.

Built by `tools/build_hcp_fc.py`.

## References

- Cole, M., Murray, K., St-Onge, E., Risk, B., Zhong, J., Schifitto, G., Descoteaux, M., &
  Zhang, Z. (2021). Surface-Based Connectivity Integration: An atlas-free approach to
  jointly study functional and structural connectivity. *Human Brain Mapping*, 42(11),
  3481-3499. <https://doi.org/10.1002/hbm.25447>
- Cole, M. R., Xiang, Y., Consagra, W., Srivastava, A., Qiu, X., & Zhang, Z. (2026). ENCORE:
  Fast geometric framework for aligning brain structural connectivity on cortical
  manifolds. *Medical Image Analysis*, 114, 104242.
  <https://doi.org/10.1016/j.media.2026.104242>
- Glasser, M. F., et al. (2013). The minimal preprocessing pipelines for the Human
  Connectome Project. *NeuroImage*, 80, 105-124.
  <https://doi.org/10.1016/j.neuroimage.2013.04.127>
