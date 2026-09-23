# Bundled atlases

Each of the 44 atlases ships as `<Name>_ico4.npz`, on the ico4 grid (5124
vertices, left hemisphere first), containing:

- `labels` -- `int32`, one entry per vertex, `0` for the medial wall
- `names` -- region names, index `i` naming label `i + 1`

They are converted from the SBCI_Toolkit `*_avg_roi_ico4.mat` files by
`tools/convert_atlases.py` (PORTING.md, item 3) and committed, so no release
step depends on MATLAB. `sbci.list_atlases()` lists them.
