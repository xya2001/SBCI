# Continuous structural and functional connectomes of $subjects HCP Young Adult subjects

Each of $subjects subjects of the WU-Minn Human Connectome Project's Young Adult study
(HCP 1200 release) has two files here, both on the fsaverage ico4 grid. That grid has
5,124 vertices, 2,562 per hemisphere, left first. 4,685 of them are cortex and 439 the
medial wall, where every value is zero.

| File | What it holds | Size |
| --- | --- | --- |
| `sc/sub-NNNNNN_sc.h5` | the **structural** connectome, the density of streamlines between every pair of grid vertices, smoothed on the sphere, with the endpoints of every streamline | $sc_gb GB |
| `fc/sub-NNNNNN_fc.h5` | the **functional** connectome, the correlation of the HCP's cleaned resting-state time series between every pair of grid vertices | $fc_gb GB |

In all: $files files, $total_gb GB. They come in $archives tar archives, so that the dataset is
a few dozen files: $archive_list. An SC archive and the FC archive of the same number hold the
same subjects, and `ARCHIVES.tsv` lists each archive's subjects, so

```bash
grep $example ARCHIVES.tsv
```

names the two archives that hold one subject. Extracting an archive puts its files in `sc/`
or `fc/`:

```bash
tar -xf sc_01.tar
```

The HDF5 files are compressed inside, so the archives are not. Next to them:

- `subjects.csv` lists each subject's sex and HCP age band (22-25, 26-30, 31-35 or 36+). Both are HCP open-access fields; no exact age and no other restricted field appears anywhere.
- `ARCHIVES.tsv` gives each archive's subjects, size and checksums.
- `SHA256SUMS` gives the checksum of every archive and of every HDF5 file in them.
- `PROCESSING.md` says how the files were made.
- `DATA_USE.txt` gives the terms they are shared under.

## Terms of use

The files are derived from HCP data and are redistributed under the
[WU-Minn HCP Consortium Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
Downloading them means accepting those terms. Among them:

- you may share the files, or anything derived from them, only under the same terms;
- a publication must carry the HCP's acknowledgment, which `DATA_USE.txt` gives word for word;
- a publication should cite the papers describing how the HCP acquired and processed the data.

## Reading the files

The files are in version $spec_version of the format of the
[sbci](https://github.com/xya2001/SBCI) Python package, whose code at the commit below
reads them, once extracted:

```python
# pip install "sbci @ git+https://github.com/xya2001/SBCI.git@$sbci_ref"
import sbci

sc = sbci.load("sc/sub-100307_sc.h5")  # with the endpoints of its streamlines
fc = sbci.load("fc/sub-100307_fc.h5")
sc.dense()  # the 5,124 x 5,124 matrix
```

Without it, each file is plain HDF5:

| Dataset | Type | Holds |
| --- | --- | --- |
| `connectivity` | float32, 13,125,126 values | the strict upper triangle of the symmetric 5,124 x 5,124 matrix, row by row. The diagonal is not stored (for FC it is one on cortex). |
| `area` | float64, 5,124 | each vertex's area, in fsaverage vertices (the 5,124 sum to 327,684). These are the weights of the grid's inner product, in which an SC density has unit mass. |
| `mask` | bool, 5,124 | true on cortex (4,685 vertices), false on the medial wall (439) |
| `metadata` | JSON string | the grid, the processing, and the file's provenance |
| `endpoints/` (SC only) | one row per streamline | for each end: its grid vertex (0 to 5,123, left hemisphere first); its triangle on the ico4 sphere that ships with sbci (5,120 faces per hemisphere, left first); and its barycentric coordinates in that triangle |

```python
import h5py
from scipy.spatial.distance import squareform

with h5py.File("fc/sub-100307_fc.h5") as f:
    matrix = squareform(f["connectivity"][()])  # 5,124 x 5,124; the diagonal left zero
```

## Checking a download

The checksums cover the archives as they download and the files once extracted, whichever
you have. On Linux:

```bash
sha256sum -c SHA256SUMS --ignore-missing
```

On macOS:

```bash
shasum -a 256 -c SHA256SUMS --ignore-missing
```

## Citing

Please cite this dataset by its DOI, and the method the files were made with:

- Cole, M., Murray, K., St-Onge, E., Risk, B., Zhong, J., Schifitto, G., Descoteaux, M., &
  Zhang, Z. (2021). Surface-Based Connectivity Integration: An atlas-free approach to jointly
  study functional and structural connectivity. *Human Brain Mapping*, 42(11), 3481-3499.
  <https://doi.org/10.1002/hbm.25447>

The streamlines are the lab's run of that pipeline for the ENCORE project:

- Cole, M. R., Xiang, Y., Consagra, W., Srivastava, A., Qiu, X., & Zhang, Z. (2026). ENCORE:
  Fast geometric framework for aligning brain structural connectivity on cortical manifolds.
  *Medical Image Analysis*, 114, 104242. <https://doi.org/10.1016/j.media.2026.104242>

Include the HCP's acknowledgment from `DATA_USE.txt`, and cite the HCP's own papers,
among them:

- Van Essen, D. C., Smith, S. M., Barch, D. M., Behrens, T. E. J., Yacoub, E., Ugurbil, K., &
  WU-Minn HCP Consortium (2013). The WU-Minn Human Connectome Project: An overview.
  *NeuroImage*, 80, 62-79. <https://doi.org/10.1016/j.neuroimage.2013.05.041>
- Glasser, M. F., et al. (2013). The minimal preprocessing pipelines for the Human Connectome
  Project. *NeuroImage*, 80, 105-124. <https://doi.org/10.1016/j.neuroimage.2013.04.127>
