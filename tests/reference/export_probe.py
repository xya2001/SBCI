"""The exported maps, checked in the software they are for: FreeSurfer and Connectome Workbench.

    module load python/3.12.4 freesurfer/7.4.1
    export WB_COMMAND=/nas/longleaf/apps/connectome/1.5.0/workbench/bin_rh_linux64/wb_command
    SBCI_HCP_DIR=/work/users/x/y/xya/hcp-ya/data python tests/reference/export_probe.py OUTDIR

Workbench is called by its path rather than through ``module load
connectome/1.5.0``, whose library path carries an old ``libstdc++`` that stops
NumPy from importing; its own launcher sets up what it needs.

On sub-100307, the tutorial subject:

1. **The GIFTI order is FreeSurfer's.** The Desikan atlas, written as a map of
   label ids, is read at each fsaverage4 vertex against FreeSurfer's own
   ``fsaverage4/label/?h.aparc.annot``. Control: the same values in the
   grid's own order.
2. **FreeSurfer reads the files.** ``mri_surf2surf`` moves the left precuneus
   seed profile from fsaverage4 to fsaverage, and the result is set against
   the profile placed on fsaverage through the grid's correspondence
   (``mapping_avg_ico4.npz``: each fsaverage vertex takes its grid vertex's
   value). Control: the same file in the grid's order.
3. **Workbench reads the dense scalars.** ``wb_command -cifti-separate`` takes
   the left cortex out as a metric, which must hold the file's own values;
   and the move keeps the area-weighted mean.
4. **A region seed's regional means are the atlas means**: on the real
   connectome, ``region_means(seed(region=k))`` against row ``k`` of
   ``to_atlas(how="mean")``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

import sbci
from sbci.export import region_means, save_map
from sbci.io.cifti import N_FSLR_PER_HEMI, vertex_areas

HALF = 2562
WB = os.environ.get("WB_COMMAND", "wb_command")
DATA = Path(os.environ.get("SBCI_HCP_DIR", "/work/users/x/y/xya/hcp-ya/data"))
MAPPING = Path(
    os.environ.get(
        "SBCI_FSAVERAGE_MAPPING",
        "/work/users/x/y/xya/sbci-reference/SBCI_Toolkit/example_data/fsaverage_label/mapping_avg_ico4.npz",
    )
)


def grid_order_gifti(values, path):
    """A left-hemisphere GIFTI in the grid's own order: the control a wrong order would be."""
    array = nib.gifti.GiftiDataArray(
        np.asarray(values[:HALF], dtype=np.float32), intent="NIFTI_INTENT_NONE"
    )
    image = nib.gifti.GiftiImage(
        darrays=[array], meta=nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary": "CortexLeft"})
    )
    nib.save(image, str(path))
    return path


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"{command[0]} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> int:
    for tool in (WB, "mri_surf2surf"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not on the PATH; see the module notes")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "export-probe")
    out.mkdir(parents=True, exist_ok=True)
    subjects = Path(os.environ["FREESURFER_HOME"]) / "subjects"
    os.environ["SUBJECTS_DIR"] = str(subjects)
    sc = sbci.load(DATA / "sub-100307_sc.h5")
    desikan = sbci.load_atlas("Desikan")

    print("1. The GIFTI order against FreeSurfer's fsaverage4 Desikan annotation")
    for side, hemi in enumerate(("lh", "rh")):
        written = save_map(desikan.labels.astype(float), out / "desikan.func.gii", mask=None)
        ours = np.rint(nib.load(written[side]).darrays[0].data).astype(int)
        codes, _, names = nib.freesurfer.read_annot(
            str(subjects / f"fsaverage4/label/{hemi}.aparc.annot")
        )
        theirs = np.array([names[c].decode() if c >= 0 else "unknown" for c in codes])
        block = desikan.labels[side * HALF : (side + 1) * HALF]

        def name(label):
            return desikan.names[label - 1].split("_", 1)[1] if label else "unknown"

        for title, labels in (
            ("in fsaverage4's order", ours),
            ("control, the grid's order", block),
        ):
            named = np.array([name(label) for label in labels])
            both = (named != "unknown") & ~np.isin(theirs, ("unknown", "corpuscallosum"))
            agree = (named[both] == theirs[both]).mean()
            print(f"   {hemi} {title:28s}: {agree:.4f} of {both.sum()} labelled vertices agree")

    print("2. mri_surf2surf, fsaverage4 to fsaverage, against the grid's correspondence")
    profile = sc.seed(region=("Desikan", "LH_precuneus"))
    with np.load(MAPPING, allow_pickle=True) as data:
        mapping = [np.asarray(m).ravel() for m in data["mapping"]]
    placed = np.full(163842, np.nan)
    for i in range(HALF):
        placed[mapping[i] % 163842] = profile[i]
    left, _ = save_map(profile, out / "precuneus.func.gii", mask=None)
    control = grid_order_gifti(profile, out / "precuneus_gridorder.L.func.gii")
    for title, source in (("in fsaverage4's order", left), ("control, the grid's order", control)):
        target = out / f"{source.name}.fsaverage.mgz"
        run(
            [
                "mri_surf2surf",
                "--srcsubject",
                "fsaverage4",
                "--sval",
                str(source),
                "--trgsubject",
                "fsaverage",
                "--tval",
                str(target),
                "--hemi",
                "lh",
            ]
        )
        moved = np.asarray(nib.load(str(target)).get_fdata()).ravel()
        r = np.corrcoef(moved, placed)[0, 1]
        print(f"   {title:28s}: r = {r:.4f} over fsaverage's 163,842 left vertices")

    print("3. Workbench reads the dense scalars")
    (dense,) = save_map({"precuneus": profile}, out / "precuneus.dscalar.nii")
    info = run([WB, "-file-information", str(dense)])
    for line in info.splitlines():
        if any(
            key in line
            for key in (
                "Type:",
                "Number of Rows",
                "Number of Columns",
                "CortexLeft",
                "CortexRight",
                "precuneus",
            )
        ):
            print("   wb:", line.strip())
    metric = out / "precuneus_left_from_wb.func.gii"
    run([WB, "-cifti-separate", str(dense), "COLUMN", "-metric", "CORTEX_LEFT", str(metric)])
    from_wb = np.asarray(nib.load(str(metric)).darrays[0].data, dtype=np.float64)
    ours = np.asarray(nib.load(str(dense)).get_fdata(), dtype=np.float64)[0, :N_FSLR_PER_HEMI]
    same = np.array_equal(np.isnan(from_wb), np.isnan(ours)) and np.array_equal(
        from_wb[~np.isnan(ours)], ours[~np.isnan(ours)]
    )
    print(
        f"   cifti-separate's left metric equals the file's values: {same}; "
        f"{int(np.isnan(ours).sum())} of {N_FSLR_PER_HEMI} left vertices missing (the wall)"
    )
    (whole,) = save_map(profile, out / "precuneus_all.dscalar.nii", mask=None)
    moved = np.asarray(nib.load(str(whole)).get_fdata(), dtype=np.float64).ravel()
    fslr_area, ico4_area = vertex_areas()
    print(
        f"   area-weighted mean: grid {profile @ ico4_area / ico4_area.sum():.6e}, "
        f"fsLR {moved @ fslr_area / fslr_area.sum():.6e}"
    )

    print("4. A region seed's regional means against to_atlas(how='mean')")
    matrix = sc.to_atlas(desikan, how="mean")
    relative, absolute, zeros = 0.0, 0.0, 0
    for k in range(desikan.n_regions):
        row = region_means(
            sc.seed(region=desikan.region_mask(k + 1)), desikan, area=sc.area, mask=sc.mask
        )
        others = np.arange(desikan.n_regions) != k
        theirs, ours = matrix[k, others], row[others]
        empty = theirs == 0  # no connectivity at all between the two: compared exactly
        zeros += int(empty.sum())
        relative = max(relative, float(np.max(np.abs(ours[~empty] / theirs[~empty] - 1))))
        absolute = max(absolute, float(np.max(np.abs(ours[empty]), initial=0.0)))
    print(
        f"   all 68 Desikan regions: largest relative difference {relative:.2e}; "
        f"{zeros} region pairs with no connectivity, where the means read at most {absolute:.1e}"
    )
    print("EXPORT PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
