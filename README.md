# sbci

[![tests](https://github.com/xya2001/SBCI/actions/workflows/tests.yml/badge.svg)](https://github.com/xya2001/SBCI/actions/workflows/tests.yml)

Continuous brain connectivity in Python: read, parcellate, smooth, couple,
align and reduce surface-based connectomes on the ico4 grid, with every method
a verified port of the SBCI group's MATLAB.

SBCI, Surface-Based Connectivity Integration (Cole et al., *Human Brain
Mapping*, 2021), represents structural and functional connectivity as
continuous functions on the cortical surface instead of between atlas regions.
The ico4 grid is FreeSurfer's fsaverage sphere subdivided four times: 2,562
vertices per hemisphere, 5,124 in all, left then right. A connectome is a
5,124 × 5,124 matrix over that grid, stored in one HDF5 file per subject.

**Status: pre-alpha.** Every method below is implemented and verified against
its reference, and an example cohort of eleven HCP Young Adult subjects
downloads with one command.

## Install

Python 3.10 or newer. The package is not on PyPI yet, so install from GitHub:

```bash
pip install "sbci @ git+https://github.com/xya2001/SBCI.git"             # core: numpy, scipy, h5py, nibabel
pip install "sbci[plotting] @ git+https://github.com/xya2001/SBCI.git"   # + matplotlib and nilearn, for figures
pip install "sbci[render] @ git+https://github.com/xya2001/SBCI.git"     # + PyVista, for the smoothly lit figures
```

On UNC's Longleaf cluster, `scripts/setup_longleaf.sh` builds the environment
(USAGE.md, *Setup on Longleaf*).

## Quick start

One real subject is 90 MB away. The package fetches it, verifies it, and
writes the data use terms beside it:

```python
import sbci
from sbci.download import fetch_cohort

sc_path, fc_path = fetch_cohort(out="hcp-ya", subjects=["100307"])   # once; verified
sc = sbci.load(sc_path)                  # structural connectome, with the endpoints of its 803,741 streamlines
fc = sbci.load(fc_path)                  # functional connectome, from the HCP's resting state
sc.to_atlas("Schaefer200").shape        # (200, 200)
sc.plot(sc.seed(vertex=1234))            # a figure; needs the plotting extra
sc.coupling(fc)                          # structure against function, one value per vertex
sc.smooth(kernel="shk", mask_medial_wall=True)     # re-smooths from the stored endpoints
```

Or from the shell:

```bash
sbci download hcp-ya --subject 100307   # sub-100307_sc.h5 and _fc.h5, 90 MB, into the current directory
sbci info sub-100307_sc.h5              # what it holds
sbci validate sub-100307_sc.h5          # nine checks, all should pass
sbci atlases --match Yeo                # the bundled atlases
```

Without network access, `sbci example` writes a synthetic connectome on the
real grid to try the calls on; nothing in the documents is drawn from it.

![The connectivity of one vertex over the cortex](docs/figures/seed_profile.png)

*`sc.plot(sc.seed(vertex=1234), mesh="fsaverage", engine="pyvista")`: where
one vertex of sub-100307 connects to. The value at each vertex is the density
of streamlines between the seed (orange dot, left temporal cortex) and that
vertex, relative to the strongest.*

## What it does

| | Call | Verified against |
| --- | --- | --- |
| Load and save | `sbci.load(path)`, `cc.save(path)` | the format's own checks (`sbci validate`) |
| Parcellate | `cc.to_atlas("Schaefer200")`, 44 atlases bundled | `parcellate_sc.m`, to float64 rounding |
| Seed profile | `cc.seed(vertex=...)`, `cc.seed(region=...)` | the MATLAB seed rows, to float64 rounding |
| Structure-function coupling | `sc.coupling(fc, scope=...)`, three forms | the three MATLAB functions, to float64 rounding |
| Smoothing | `cc.smooth(kernel="shk" \| "rdk" \| "matern")` from the stored endpoints | `concon` at r = 1.000000 at full scale; MATLAB to 3.25 float32-eps; the Matern closed form |
| Alignment | `sbci.align(cohort)` (ENCORE, on densities), `sbci.endpoints_align(cohort)` (ConSEAL, on endpoints) | the MATLAB references to float64 rounding and r = 0.99999979; ConSEAL to the single precision its reference carries |
| Carrying a warp | `sbci.migrate_warp(warp, to="fs_LR_32k")` | an identity and a known rotation |
| Reduction | `sbci.reduce(cohort, rank=K)` (functional PCA, one file at a time) | the MATLAB reference to 2.1e-16 |
| Inference | `sbci.local_test(scores, design, groups=...)` | `scipy.stats` and statsmodels (no MATLAB reference exists) |
| Figures | `cc.plot(values, mesh="fsaverage", engine="pyvista")` | drawn on FreeSurfer's fsaverage, shaded by sulcal depth |
| Exchange | `cc.to_cifti(path)`: a 16.9 GB fsLR-32k dense connectome for Connectome Workbench | mass conserved to 1e-10; Workbench reads it |
| Data | `sbci download hcp-ya` | SHA-256 of every file |

[PORTING.md](PORTING.md) records how each port was verified and what was found
in the references; [docs/RESULTS.md](docs/RESULTS.md) shows what the methods
do on the released subjects.

## Example data

The example cohort is eleven HCP Young Adult subjects, each with a structural
connectome (SC, with its streamline endpoints) and a functional one (FC, from
the HCP's own cleaned resting-state runs placed on the grid through the
subject's registration), hosted on a public Google Drive
([the folder](https://drive.google.com/drive/folders/1gG2ZmxxVm4w5dvlCQvMaEEOBypU7nDpx),
for browsing) and fetched by the package:

```bash
sbci download hcp-ya --out hcp-ya        # all eleven, about 1 GB
sbci download hcp-ya --subject 100307    # one subject, 90 MB
```

The data are the WU-Minn Human Connectome Project's, redistributed under its
[Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms);
downloading them means accepting those terms, and every download writes
`DATA_USE.txt` beside the files with the acknowledgment a publication must
carry. The manifest gives each subject's sex and HCP age band, never an exact
age. The other 935 young adults with pipeline output are analysed in
[docs/RESULTS.md](docs/RESULTS.md) but not distributed; `tools/build_hcp_cohort.py`
and `tools/build_hcp_fc.py` build them for anyone with HCP access, and
`tools/import_legacy.py` converts existing pipeline output (USAGE.md,
*Getting the example cohort* and *Using legacy pipeline output*).

## Documentation

| Read | For |
| --- | --- |
| [USAGE.md](USAGE.md) | every function with a worked example on the released subjects, and what the common errors tell you |
| [docs/RESULTS.md](docs/RESULTS.md) | what the methods do on real data: the figures, and the analysis of the 946 young adults |
| [PORTING.md](PORTING.md) | how each of the seven MATLAB methods was ported and verified, and the errors found in the references |
| [VERIFICATION.md](VERIFICATION.md) | how to check every claim yourself, in tiers from five minutes to a MATLAB licence |
| [SPEC_QUESTIONS.md](SPEC_QUESTIONS.md) | the file-format decisions, answered and open |
| [BLUEPRINT.md](BLUEPRINT.md) | what the package is, what "finished" means, and the decisions it waits on |
| [scripts/](scripts/README.md), [tools/](tools/README.md), [docs/figures/](docs/figures/README.md) | worked scripts (with the lab's Longleaf paths), the builders of the bundled data, and the figures |

## Development

```bash
git clone https://github.com/xya2001/SBCI.git && cd SBCI
pip install -e ".[dev]"
pytest                                   # about a minute; 33 tests skip without data outside the repository
ruff check . && ruff format --check src tests scripts tools
```

CI runs the suite on Linux and macOS, Python 3.10 and 3.13, builds and
installs the wheel from a blank environment, runs the quick start above
against the public Drive, and renders the figures. On Longleaf,
`sbatch scripts/test.sbatch` runs the suite as a batch job, and
`SBCI_HCP_DIR=hcp-ya sbatch --export=ALL scripts/test.sbatch` adds the tests
on the released subject. [VERIFICATION.md](VERIFICATION.md) lists what each
check proves.

## References

The methods implemented here, and the paper each one ports:

- **SBCI** — Cole, M., Murray, K., St-Onge, E., Risk, B., Zhong, J., Schifitto, G.,
  Descoteaux, M., & Zhang, Z. (2021). Surface-Based Connectivity Integration: An
  atlas-free approach to jointly study functional and structural connectivity.
  *Human Brain Mapping*, 42(11), 3481–3499. <https://doi.org/10.1002/hbm.25447>
- **ENCORE** — Cole, M. R., Xiang, Y., Consagra, W., Srivastava, A., Qiu, X., & Zhang, Z.
  (2026). ENCORE: Fast geometric framework for aligning brain structural connectivity
  on cortical manifolds. *Medical Image Analysis*, 114, 104242.
  <https://doi.org/10.1016/j.media.2026.104242>
- **ConSEAL** — Xiang, Y., Cole, M., & Zhang, Z. (2026). ConSEAL: Connectivity-informed
  streamline endpoint alignment for diffeomorphic cortical registration. *arXiv*,
  2605.16742. <https://arxiv.org/abs/2605.16742>
- **FPCA** — Consagra, W., Cole, M., Qiu, X., & Zhang, Z. (2024). Continuous and
  atlas-free analysis of brain structural connectivity. *Annals of Applied
  Statistics*, 18(3), 1815–1839. <https://doi.org/10.1214/23-AOAS1858>
- **Riemannian diffusion kernel** — Wang, L., Li, D., & Zhang, Z. (2025). Riemannian
  diffusion kernel-smoothed continuous structural connectivity on cortical surface.
  *bioRxiv*, 2025.09.08.674789. <https://doi.org/10.1101/2025.09.08.674789>

## License

MIT -- see [LICENSE](LICENSE).
