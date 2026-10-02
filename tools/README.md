# tools

Builders of the data the package bundles, and converters for the pipeline's
files. Users never need them: every artifact they produce is committed under
`src/sbci/data/`, so no release step depends on MATLAB, FreeSurfer or the
toolkit checkout. Run one only to regenerate its output after its source
changes (PORTING.md says which item each belongs to).

| Tool | Produces |
| --- | --- |
| `convert_atlases.py` | the 44 atlas label files, `src/sbci/data/atlases/*_ico4.npz`, from the toolkit's `*_avg_roi_ico4.mat` |
| `build_surfaces.py` | the inflated, white and pial meshes at ico4, `src/sbci/data/surfaces/`, from FreeSurfer's fsaverage |
| `convert_surfaces.py` | the sphere, `src/sbci/data/surfaces/sphere_ico4.npz`, from the toolkit's ico4 VTK export, and the canonical face order the stored triangle indices use, digest-checked |
| `build_template_spheres.py` | the fs_LR 32k registration spheres `migrate_warp` carries a warp to, `src/sbci/data/templates/fslr32k_spheres.npz` |
| `align_surface_faces.py` | puts the bundled surfaces' faces in the order the stored triangle indices refer to (SPEC_QUESTIONS.md item 13) |
| `build_resampling.py` | the ico4 to fsLR-32k overlap matrix behind the exchange format |
| `build_fslr_surfaces.py` | fsLR-32k anatomical surfaces from fsaverage, for viewing the exchange file |
| `import_legacy.py` | converts legacy pipeline `.mat` output (SC, FC, endpoints) into the package's HDF5 files |
| `build_hcp_cohort.py` | builds HCP Young Adult subjects in the package's format from the lab's pipeline output, validating every file: each subject's SC smoothed with the package's kernel from the pipeline's snapped streamline endpoints on its FreeSurfer-registered sphere, stored with those endpoints (the released example cohort and the 946 of the analysis). Takes a manifest of subject ids and reads nothing else from it; `--check` re-smooths each file's endpoints against its stored SC. The FC comes from `build_hcp_fc.py` |
| `build_hcp_fc.py` | HCP Young Adult FC on the grid from the HCP's ICA-FIX-cleaned resting-state runs (32k fs_LR): each 32k vertex is carried through the subject's MSMSulc and FreeSurfer spheres, so the FC sits where the SC's endpoints do, and the FC follows the pipeline's nuisance model and definition; `--bandpass` and `--no-gsr` vary them, and `--crossmesh` routes a subject whose FreeSurfer mesh is not the HCP's through the white surfaces (USAGE.md, *Functional connectivity from the HCP's resting state*) |
| `bundle_hcp_cohort.py` | a cohort too large for one file per subject, released on Zenodo: `bundle` zips the files into bundles of twenty-four subjects per modality (a record holds at most a hundred files) with `bundles.json`; `manifest` turns that and the published record id into a package manifest. No cohort is released this way at present |
| `zenodo_upload.py` | uploads such bundles to a Zenodo deposition through the API with a token from the environment, sets the record's metadata from `--title` and the defaults, and stops short of publishing so the owner can review; `--sandbox` for a dry run on sandbox.zenodo.org |
| `age_probe.py` | nothing to bundle: streams a cohort's SC files once and asks whether and where a subject measure shows, age by default or any column of an open-access table (`--table`, `--column`), at the vertex, region and global levels and in the leading directions of variation next to sex and the streamline count; `--groups` treats families as clusters, and `--measures` repeats the tests on measures an earlier `--out` saved; PORTING.md item 5 records what it found against fluid intelligence on the 946 young adults |
| `clustered_null.py` | nothing to bundle: simulates cohorts of families with no association and reports how often `local_test` rejects at 0.05, counting subjects as independent and with `groups=`, for 20 to 422 families; PORTING.md item 5, *Related subjects*, records the output |
| `encore_probe.py` | nothing to bundle: measures ENCORE's cost landscape around a known warp of a real subject (the oracle, the smoother floor, ConSEAL's answer in ENCORE's cost) and its recovery over the warp's smoothness, the basis order and the reference; PORTING.md item 4 records the results |
| `md2pdf.py` | renders a project markdown document as a typeset PDF |
