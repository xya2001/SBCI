# tools

Builders of the data the package bundles, and converters for the pipeline's
files. Users never need them: every artifact they produce is committed under
`src/sbci/data/`, so no release step depends on MATLAB, FreeSurfer or the
toolkit checkout. Run one only to regenerate its output after its source
changes (PORTING.md says which item each belongs to).

| Tool | Produces |
| --- | --- |
| `convert_atlases.py` | the 44 atlas label files, `src/sbci/data/atlases/*_ico4.npz`, from the toolkit's `*_avg_roi_ico4.mat` |
| `build_surfaces.py` | the inflated, white, pial and sphere meshes at ico4, `src/sbci/data/surfaces/`, from FreeSurfer's fsaverage |
| `convert_surfaces.py` | the toolkit's own ico4 VTK meshes in the package's format (superseded by `build_surfaces.py` for the anatomical surfaces) |
| `align_surface_faces.py` | puts the bundled surfaces' faces in the order the stored triangle indices refer to (SPEC_QUESTIONS.md item 13) |
| `build_resampling.py` | the ico4 to fsLR-32k overlap matrix behind the exchange format |
| `build_fslr_surfaces.py` | fsLR-32k anatomical surfaces from fsaverage, for viewing the exchange file |
| `import_legacy.py` | converts legacy pipeline `.mat` output (SC, FC, endpoints) into the package's HDF5 files |
| `build_hcp_cohort.py` | builds an HCP cohort in the package's format from the lab's pipeline output, validating every file: `--layout young-adult` smooths each subject's SC with the package's kernel from the pipeline's snapped streamline endpoints on the FreeSurfer-registered sphere (the released example cohort and the 946 of the analysis); `--layout aging` converts the pipeline's own SC and FC for the lab's internal HCP-Aging copies. Takes a manifest of subject ids and reads nothing else from it; `--check` re-smooths each file's endpoints against its stored SC |
| `bundle_hcp_cohort.py` | a cohort too large for one file per subject, released on Zenodo: `bundle` zips the files into bundles of twenty-four subjects per modality (a record holds at most a hundred files) with `bundles.json`; `manifest` turns that and the published record id into a package manifest. No cohort is released this way at present, and the 528-subject HCP-Aging cohort is not to be |
| `zenodo_upload.py` | uploads such bundles to a Zenodo deposition through the API with a token from the environment, sets the record's metadata from `--title` and the defaults, and stops short of publishing so the owner can review; `--sandbox` for a dry run on sandbox.zenodo.org |
| `age_probe.py` | nothing to bundle: streams a cohort's SC files once and asks whether and where age shows, at the vertex, region and global levels and in the leading directions of variation next to sex and the streamline count; PORTING.md item 5 records what it found on the 528 HCP-Aging subjects |
| `encore_probe.py` | nothing to bundle: measures ENCORE's cost landscape around a known warp of a real subject (the oracle, the smoother floor, ConSEAL's answer in ENCORE's cost) and its recovery over the warp's smoothness, the basis order and the reference; PORTING.md item 4 records the results |
| `md2pdf.py` | renders a project markdown document as a typeset PDF |
