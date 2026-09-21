# Bundled atlases

Each atlas ships as `<Name>_ico4.npz` and `<Name>_fsLR32k.npz`, containing:

- `labels` -- `int32`, one entry per vertex, `0` for the medial wall
- `names` -- region names, index `i` naming label `i + 1`

These are not vendored yet. Convert them from the SBCI_Toolkit
`*_avg_roi_ico4.mat` files (see `PORTING.md`, item 3); the conversion script
belongs in `tools/` and its output is committed, since the files are small and
the release must not depend on a MATLAB step.
