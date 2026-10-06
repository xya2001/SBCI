r"""Build HCP Young Adult functional connectomes on the ico4 grid from the HCP's resting state.

    python tools/build_hcp_fc.py --manifest manifest.csv \
        --fmri /proj/STOR/zz10c/HCP_fMRI --fmri SUPPLEMENT \
        --spheres /overflow/zzhanglab/encore_project/encore_paper_code/prediction_subs \
        --fslr <...>/fslr --mapping <...>/fsaverage_label/mapping_avg_ico4.npz \
        --out /work/users/x/y/xya/hcp-ya/fc/data \
        [--fsaverage SURF_DIR] [--min-runs N] [--no-gsr] [--bandpass LOW HIGH] \
        [--crossmesh DIR [--crossmesh-all]]

``manifest.csv`` has a ``subject`` column (``sub-100307``). The time series are the
HCP's ICA-FIX-cleaned resting-state runs on the 32k fs_LR mesh,
``rfMRI_REST{1,2}_{LR,RL}_Atlas_hp2000_clean.dtseries.nii`` (MSMSulc-aligned), looked
for under each ``--fmri`` directory in turn, as ``HCP_fMRI<id>/`` or ``<id>/``; every run
found is used and the file records which.

The SC files place each streamline endpoint on the subject's FreeSurfer-registered sphere
(``?h.sphere.reg``). The FC is put on the grid through the same registration, so that
coupling compares the two at the same place: each 32k vertex is located on the
subject's native MSMSulc sphere (``?h_sphere_msmsulc_reg.vtk``), which the HCP resampled
the native data through, carried by its barycentric coordinates to the same subject's
FreeSurfer sphere (``?h_sphere_freesurfer_reg.vtk``), and assigned to the grid vertex
whose fsaverage cell (``mapping_avg_ico4.npz``) contains the nearest fsaverage vertex.
Where the HCP's native mesh is not FreeSurfer's (sub-103010 of the 972 checked: its
FreeSurfer surfaces come from another run), the vertex goes through the white surfaces
instead: its position on the native mesh's white surface in T1w space
(``--crossmesh <id>/<id>.?.white.native.surf.gii``), the nearest vertex of FreeSurfer's
white surface (``<id>/?h.white``), and that vertex's place on the FreeSurfer sphere.

The nuisance model follows the pipeline's ``calculate_residual_timeseries.py`` as far
as these data allow: ICA-FIX has already removed motion, white-matter and CSF artefacts,
and each run then has its global signal (the mean over all grayordinates), a constant
and a linear trend regressed out (``--no-gsr`` keeps the global signal). The pipeline
does not band-pass; ``--bandpass 0.01 0.1`` filters each grid vertex's series first,
which raises the fine-scale signal of these 0.72 s full-band data (PORTING.md). The
connectivity follows the pipeline's ``calculate_fc.py``: the time series of the 32k
vertices in a grid vertex's cell are averaged, and the FC is the Pearson correlation
between those averages, here with each run's averages z-scored and the runs
concatenated. A cortical grid vertex whose cell holds no 32k vertex (a few, where the
HCP's medial wall and the grid's differ) takes the nearest one. The diagonal is one and
the medial wall zero, as in the legacy FC. Every file written is validated.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.signal import butter, sosfiltfilt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_surfaces import read_vtk_polydata  # noqa: E402
from import_legacy import N_VERTICES, area_weights  # noqa: E402

RUNS = ("REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL")
CORTEX = {"lh": "CIFTI_STRUCTURE_CORTEX_LEFT", "rh": "CIFTI_STRUCTURE_CORTEX_RIGHT"}
N_FSAVERAGE = 163842
NUISANCE = (
    "HCP minimal preprocessing and ICA-FIX (rfMRI *_Atlas_hp2000_clean); per run, {} a "
    "constant and a linear trend regressed out; grid-vertex mean time series z-scored and "
    "the runs concatenated"
)


def run_paths(subject: str, roots: list[str], runs=RUNS) -> dict[str, str]:
    """The resting-state runs available for a subject, of ``runs``, first found under the roots."""
    bare = subject.removeprefix("sub-")
    found = {}
    for run in runs:
        name = f"rfMRI_{run}_Atlas_hp2000_clean.dtseries.nii"
        for root in roots:
            for folder in (f"HCP_fMRI{bare}", bare):
                path = os.path.join(root, folder, name)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    found.setdefault(run, path)
    return found


def locate(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, tree: cKDTree):
    """The triangle of a spherical mesh containing each point, and its barycentric weights.

    Solves ``p = a A + b B + c C`` up to scale for candidate triangles, the ones whose
    centroids are nearest, and keeps the first with all weights non-negative. A
    stretched triangle's centroid can rank far down that list (on sub-178647's right
    MSMSulc sphere, 172nd for the one 32k vertex it holds), so a point none of its 128
    nearest triangles contains is tested against every triangle.
    """
    n = points.shape[0]
    tri = np.full(n, -1, dtype=np.int64)
    bary = np.zeros((n, 3))
    todo = np.arange(n)
    for k in (8, 32, 128):
        if todo.size == 0:
            break
        _, cand = tree.query(points[todo], k=k)
        for column in range(k):
            if todo.size == 0:
                break
            t = cand[:, column] if cand.ndim > 1 else cand
            corners = vertices[faces[t]]  # (m, 3 corners, 3 coordinates)
            raw = np.linalg.solve(np.transpose(corners, (0, 2, 1)), points[todo][..., None])[..., 0]
            total = raw.sum(axis=1)
            weights = raw / total[:, None]
            inside = (total > 0) & np.all(weights >= -1e-9, axis=1)
            tri[todo[inside]] = t[inside]
            bary[todo[inside]] = weights[inside]
            keep = ~inside
            todo, cand = todo[keep], cand[keep]
    if todo.size:
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        centroids = (a + b + c) / 3.0
        numerators = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1)
        determinant = np.einsum("ij,ij->i", a, numerators[:, 0])
        with np.errstate(divide="ignore", invalid="ignore"):  # degenerate triangles fail below
            for p in todo:
                raw = (numerators @ points[p]) / determinant[:, None]  # Cramer's rule
                total = raw.sum(axis=1)
                weights = raw / total[:, None]
                inside = np.flatnonzero((total > 0) & np.all(weights >= -1e-9, axis=1))
                if inside.size == 0:
                    raise RuntimeError(f"point {p} falls in no triangle of the mesh")
                best = inside[np.argmin(np.linalg.norm(centroids[inside] - points[p], axis=1))]
                tri[p], bary[p] = best, weights[best]
    return tri, bary


def through_white(subject: str, h: str, args, faces, tri, bary, reg) -> np.ndarray:
    """FreeSurfer sphere positions through the white surfaces, when the two meshes differ."""
    if not args.crossmesh:
        raise ValueError(
            f"{subject} {h}: the MSMSulc and FreeSurfer spheres are not one mesh; pass --crossmesh"
        )
    bare = subject.removeprefix("sub-")
    folder = os.path.join(args.crossmesh, bare)
    hemi = "L" if h == "lh" else "R"
    native = nib.load(os.path.join(folder, f"{bare}.{hemi}.white.native.surf.gii"))
    native = np.asarray(native.darrays[0].data, dtype=np.float64)
    white, _, meta = nib.freesurfer.read_geometry(
        os.path.join(folder, f"{h}.white"), read_metadata=True
    )
    white = white + meta["cras"]
    if native.shape[0] != faces.max() + 1 or white.shape[0] != reg.shape[0]:
        raise ValueError(f"{subject} {h}: the white surfaces do not match the spheres' meshes")
    position = np.einsum("ij,ijk->ik", bary, native[faces[tri]])
    _, nearest = cKDTree(white).query(position)
    return reg[nearest]


def grid_cells(subject: str, args, standard: dict, fsaverage: dict, cell_of: np.ndarray):
    """For each hemisphere, the grid vertex of every 32k fs_LR vertex, through the subject.

    Returns each hemisphere's (grid vertex, FreeSurfer-sphere position) of every 32k
    vertex, and the routes taken.
    """
    folder = os.path.join(args.spheres, subject.removeprefix("sub-"))
    cells, routes = {}, set()
    for k, h in enumerate(("lh", "rh")):
        msm, faces = read_vtk_polydata(os.path.join(folder, f"{h}_sphere_msmsulc_reg.vtk"))
        reg, reg_faces = read_vtk_polydata(os.path.join(folder, f"{h}_sphere_freesurfer_reg.vtk"))
        faces = np.asarray(faces)
        one_mesh = msm.shape == reg.shape and np.array_equal(faces, np.asarray(reg_faces))
        unit = msm / np.linalg.norm(msm, axis=1, keepdims=True)
        tree = cKDTree(unit[faces].mean(axis=1))
        points = standard[h] / np.linalg.norm(standard[h], axis=1, keepdims=True)
        tri, bary = locate(points, unit, faces, tree)
        if one_mesh and not getattr(args, "crossmesh_all", False):
            on_reg = np.einsum("ij,ijk->ik", bary, reg[faces[tri]])
            routes.add("barycentric")
        else:
            on_reg = through_white(subject, h, args, faces, tri, bary, reg)
            routes.add("white surface")
        on_reg = on_reg / np.linalg.norm(on_reg, axis=1, keepdims=True)
        _, nearest = fsaverage[h].query(on_reg)
        cells[h] = (cell_of[nearest + k * N_FSAVERAGE], on_reg)
    return cells, routes


def regress(data: np.ndarray, gsr: bool) -> np.ndarray:
    """Residuals of every column of ``(time, grayordinates)`` data on the nuisance terms.

    The terms are a constant and a linear trend, and with ``gsr`` the global signal, the
    mean over all columns.
    """
    n = data.shape[0]
    terms = [np.ones(n), np.linspace(-1.0, 1.0, n)] + ([data.mean(axis=1)] if gsr else [])
    design = np.column_stack(terms)
    return data - design @ np.linalg.lstsq(design, data, rcond=None)[0]


def cortical_series(path: str, gsr: bool) -> tuple[np.ndarray, dict, float]:
    """A run's cortical time series, (time, vertices), each hemisphere's 32k ids, and its TR.

    The nuisance terms (:func:`regress`) are removed from every grayordinate first.
    """
    image = nib.load(path)
    axis = image.header.get_axis(1)
    data = regress(np.asarray(image.dataobj, dtype=np.float64), gsr)
    parts, ids = [], {}
    for h in ("lh", "rh"):
        for name, sl, bm in axis.iter_structures():
            if name == CORTEX[h]:
                parts.append(data[:, sl])
                ids[h] = np.asarray(bm.vertex)
    return np.concatenate(parts, axis=1), ids, float(image.header.get_axis(0).step)


def averaging_matrix(cells: dict, ids: dict, mask: np.ndarray, grid: np.ndarray):
    """Rows average the 32k vertices of each grid vertex's cell; empty cells take the nearest."""
    target = np.concatenate([cells["lh"][0][ids["lh"]], cells["rh"][0][ids["rh"]]])
    where = np.concatenate([cells["lh"][1][ids["lh"]], cells["rh"][1][ids["rh"]]])
    columns = np.arange(target.size)
    counts = np.bincount(target, minlength=N_VERTICES)
    empty = np.flatnonzero((counts == 0) & mask)
    if empty.size:
        half = N_VERTICES // 2
        for side, hemisphere in (
            (empty < half, slice(0, len(ids["lh"]))),
            (empty >= half, slice(len(ids["lh"]), target.size)),
        ):
            if side.any():
                tree = cKDTree(where[hemisphere])
                _, nearest = tree.query(grid[empty[side]])
                target = np.concatenate([target, empty[side]])
                columns = np.concatenate([columns, np.arange(target.size)[hemisphere][nearest]])
    counts = np.bincount(target, minlength=N_VERTICES).astype(np.float64)
    averaging = sparse.csr_matrix(
        (1.0 / counts[target], (target, columns)),
        shape=(N_VERTICES, len(ids["lh"]) + len(ids["rh"])),
    )
    on_wall = float(np.mean(~mask[target[: len(ids["lh"]) + len(ids["rh"])]]))
    return averaging, on_wall, int(empty.size)


def connectivity(
    runs: dict,
    cells: dict,
    mask: np.ndarray,
    grid: np.ndarray,
    gsr: bool,
    band: tuple[float, float] | None = None,
):
    """The FC matrix, the share of 32k vertices on the grid's medial wall, cells filled, frames."""
    blocks = []
    averaging = None
    for run in RUNS:
        if run not in runs:
            continue
        series, ids, tr = cortical_series(runs[run], gsr)
        if averaging is None:
            averaging, on_wall, empty = averaging_matrix(cells, ids, mask, grid)
        means = averaging @ series.T  # (grid, time)
        if band is not None:
            sos = butter(4, list(band), btype="bandpass", fs=1.0 / tr, output="sos")
            means = sosfiltfilt(sos, means, axis=1)
        means -= means.mean(axis=1, keepdims=True)
        scale = means.std(axis=1, keepdims=True)
        scale[scale == 0] = np.inf
        blocks.append(means / scale)
    z = np.concatenate(blocks, axis=1)
    fc = (z @ z.T) / z.shape[1]
    live = mask & (np.abs(z).sum(axis=1) > 0)
    fc[~live, :] = 0.0
    fc[:, ~live] = 0.0
    np.fill_diagonal(fc, np.where(live, 1.0, 0.0))
    return fc, on_wall, empty, z.shape[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="a csv with a subject column")
    parser.add_argument(
        "--fmri", required=True, action="append", help="time series roots, in order"
    )
    parser.add_argument("--spheres", required=True, help="directory of <id>/ registered spheres")
    parser.add_argument("--fslr", required=True, help="directory with ?.sphere.32k_fs_LR.surf.gii")
    parser.add_argument("--mapping", required=True, help="mapping_avg_ico4.npz")
    parser.add_argument("--fsaverage", default=None, help="fsaverage surf/ (default: FreeSurfer's)")
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument("--min-runs", type=int, default=1, help="skip subjects with fewer runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=RUNS,
        default=list(RUNS),
        help="the runs to use, as REST1_LR REST1_RL for the first day's alone (default: all four)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="name the files sub-<id>_ses-<SESSION>_fc.h5, as REST1 for the first day's runs",
    )
    parser.add_argument("--no-gsr", action="store_true", help="keep the global signal")
    parser.add_argument(
        "--bandpass",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="band-pass each grid vertex's series, in Hz",
    )
    parser.add_argument(
        "--crossmesh", default=None, help="<id>/ white surfaces, where the two meshes differ"
    )
    parser.add_argument(
        "--crossmesh-all",
        action="store_true",
        help="take the white-surface route for every subject (a check of it)",
    )
    args = parser.parse_args(argv)

    from sbci.atlas import load_atlas
    from sbci.grid import to_condensed
    from sbci.io import write_hdf5
    from sbci.metadata import template
    from sbci.validate import validate_file

    fs_surf = args.fsaverage or os.path.join(
        os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "surf"
    )
    os.makedirs(args.out, exist_ok=True)
    areas = area_weights(args.mapping)
    mask = load_atlas("aparc").labels != 0
    with np.load(args.mapping, allow_pickle=True) as z:
        cell_of = np.empty(2 * N_FSAVERAGE, dtype=np.int64)
        for g, members in enumerate(z["mapping"]):
            cell_of[np.asarray(members, dtype=np.int64)] = g
    standard, fsaverage, grid = {}, {}, []
    for h, H in (("lh", "L"), ("rh", "R")):
        sphere = nib.load(os.path.join(args.fslr, f"{H}.sphere.32k_fs_LR.surf.gii"))
        standard[h] = np.asarray(sphere.darrays[0].data, dtype=np.float64)
        v, _ = nib.freesurfer.read_geometry(os.path.join(fs_surf, f"{h}.sphere.reg"))
        unit = v / np.linalg.norm(v, axis=1, keepdims=True)
        fsaverage[h] = cKDTree(unit)
        grid.append(unit[: N_VERTICES // 2])  # the ico4 vertices come first on fsaverage
    grid = np.concatenate(grid)
    with open(args.manifest) as handle:
        subjects = [row["subject"] for row in csv.DictReader(handle)]

    written, failed, invalid = [], [], 0
    for subject in subjects:
        t = time.time()
        runs = run_paths(subject, args.fmri, args.runs)
        if len(runs) < args.min_runs:
            print(f"{subject}: {len(runs)} resting-state runs, fewer than {args.min_runs}; skipped")
            continue
        band = tuple(args.bandpass) if args.bandpass else None
        try:
            cells, routes = grid_cells(subject, args, standard, fsaverage, cell_of)
            fc, on_wall, empty, frames = connectivity(
                runs, cells, mask, grid, not args.no_gsr, band
            )
        except (OSError, ValueError, RuntimeError) as error:  # report it and go on
            print(f"{subject}: not built, {error}", flush=True)
            failed.append(subject)
            continue
        nuisance = NUISANCE.format("the global signal," if not args.no_gsr else "")
        if band:
            nuisance += f"; each grid vertex's series band-passed {band[0]:g}-{band[1]:g} Hz"
        metadata = template(
            "fc",
            registration_reference="fsaverage",
            pipeline_version="HCP minimal preprocessing and ICA-FIX; sbci grid assignment",
            container_version="none (built from the HCP's preprocessed data)",
            entry_point="preprocessed",
            normalization="none",
            fc_nuisance_model=" ".join(nuisance.split()),
            fc_runs=sorted(runs),
            fc_frames=int(frames),
            provenance=f"HCP Young Adult {subject}; resting-state runs {', '.join(sorted(runs))} "
            "(rfMRI *_Atlas_hp2000_clean, 32k fs_LR, MSMSulc) placed on the grid through the "
            f"subject's MSMSulc and FreeSurfer spheres ({' and '.join(sorted(routes))} route); "
            "built by tools/build_hcp_fc.py",
        )
        session = f"_ses-{args.session}" if args.session else ""
        path = os.path.join(args.out, f"{subject}{session}_fc.h5")
        write_hdf5(
            path, data=to_condensed(fc).astype(np.float32), area=areas, mask=mask, metadata=metadata
        )
        ok = all(check.passed for check in validate_file(path))
        invalid += not ok
        written.append(path)
        upper = fc[np.triu_indices(N_VERTICES, 1)]
        upper = upper[upper != 0]
        print(
            f"{subject} fc ({'/'.join(sorted(routes))}): {len(runs)} runs, {frames} frames; "
            f"{100 * on_wall:.2f}% of cortical "
            f"32k vertices on the grid's medial wall, {empty} empty cells filled; FC mean "
            f"{upper.mean():.3f}, sd {upper.std():.3f}; {os.path.getsize(path) / 1e6:.1f} MB; "
            f"{'valid' if ok else 'INVALID'}; {time.time() - t:.0f}s",
            flush=True,
        )
    print(
        f"wrote {len(written)} files, {invalid} failed validation; {len(failed)} not built"
        + (f" ({', '.join(failed)})" if failed else ""),
        flush=True,
    )
    return 1 if invalid or failed else 0


if __name__ == "__main__":
    sys.exit(main())
