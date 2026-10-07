"""Maps and tables out of the package: what each file holds, and where each value lands.

The surface forms are checked by reading them back with nibabel and undoing
what was done to them: the GIFTI files' fsaverage4 order is inverted and must
give the grid's values exactly, and the fsLR dense scalars must keep the
area-weighted mean that the resampling operator conserves. That the order is
FreeSurfer's own, and that Workbench reads the files, was checked against
FreeSurfer 7.4.1 and Workbench 1.5.0 on Longleaf (VERIFICATION.md); here the
bundled order is checked to carry the grid's triangles onto fsaverage4's, by
digest, without FreeSurfer installed.
"""

from __future__ import annotations

import csv
import hashlib

import nibabel as nib
import numpy as np
import pytest

import sbci
from sbci import spec
from sbci.atlas import cortex_mask, load_atlas
from sbci.export import fsaverage4_order, region_means, save_map, save_regions
from sbci.io.cifti import N_FSLR, load_overlap, vertex_areas

N = spec.N_VERTICES
HALF = spec.N_VERTICES_PER_HEMI


def _digest(faces_by_hemisphere) -> str:
    """As tools/build_fsaverage4_order.py computes it: sorted triples, sorted, left first."""
    digest = hashlib.sha256()
    for faces in faces_by_hemisphere:
        triples = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
        triples = triples[np.lexsort(triples.T[::-1])]
        digest.update(np.ascontiguousarray(triples, dtype="<i8").tobytes())
    return digest.hexdigest()


def _read_table(path):
    delimiter = "\t" if str(path).endswith(".tsv") else ","
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    return rows[0], rows[1:]


# --- the fsaverage4 order ----------------------------------------------------


def test_the_order_is_a_renumbering_that_carries_the_triangles_onto_fsaverage4():
    """A permutation per hemisphere, and the grid's faces renumbered by it are FreeSurfer's."""
    from importlib import resources

    left, right = fsaverage4_order()
    for order in (left, right):
        assert sorted(order.tolist()) == list(range(HALF))
    inflated = sbci.load_surface("inflated")
    carried = [
        order[np.asarray(inflated.hemisphere(side).faces)]
        for side, order in zip("LR", (left, right), strict=True)
    ]
    path = resources.files("sbci.data.surfaces") / "fsaverage4_order_ico4.npz"
    with np.load(path, allow_pickle=False) as data:
        expected = str(data["faces_sha256"])
    assert _digest(carried) == expected
    # And the identity numbering is not: the check can fail.
    assert _digest([np.asarray(inflated.hemisphere(s).faces) for s in "LR"]) != expected


# --- GIFTI ---------------------------------------------------------------


def test_gifti_files_hold_the_grid_values_in_fsaverage4_order(tmp_path):
    values = np.arange(N, dtype=np.float64)
    written = save_map(values, tmp_path / "ramp.func.gii", mask=None)
    assert [p.name for p in written] == ["ramp.L.func.gii", "ramp.R.func.gii"]
    for side, (path, order) in enumerate(zip(written, fsaverage4_order(), strict=True)):
        image = nib.load(path)
        assert image.meta["AnatomicalStructurePrimary"] == ("CortexLeft", "CortexRight")[side]
        (array,) = image.darrays
        assert array.meta["Name"] == "value"
        assert array.data.dtype == np.float32 and array.data.shape == (HALF,)
        # fsaverage4 vertex order[i] holds grid vertex i.
        np.testing.assert_array_equal(array.data[order], values[side * HALF : (side + 1) * HALF])


def test_several_maps_become_named_arrays_and_the_wall_is_missing(tmp_path):
    rng = np.random.default_rng(0)
    maps = {"strength": rng.random(N), "coupling": rng.random(N)}
    left, right = save_map(maps, tmp_path / "maps.func.gii")
    wall = ~cortex_mask()
    for side, path in enumerate((left, right)):
        arrays = nib.load(path).darrays
        assert [a.meta["Name"] for a in arrays] == ["strength", "coupling"]
        order = fsaverage4_order()[side]
        block = slice(side * HALF, (side + 1) * HALF)
        for array, values in zip(arrays, maps.values(), strict=True):
            back = np.asarray(array.data, dtype=np.float64)[order]
            np.testing.assert_array_equal(np.isnan(back), wall[block])
            np.testing.assert_allclose(back[~wall[block]], values[block][~wall[block]], rtol=1e-7)


def test_a_name_that_already_carries_a_hemisphere_is_refused(tmp_path):
    with pytest.raises(ValueError, match="already names a hemisphere"):
        save_map(np.zeros(N), tmp_path / "x.L.func.gii")


# --- CIFTI dense scalars on fsLR-32k -----------------------------------------


def test_dense_scalars_keep_the_area_weighted_mean(tmp_path):
    rng = np.random.default_rng(1)
    maps = rng.standard_normal((3, N))
    (path,) = save_map(maps, tmp_path / "three.dscalar.nii", names=["a", "b", "c"], mask=None)
    image = nib.load(path)
    assert int(image.nifti_header["intent_code"]) == 3006  # dense scalars
    scalars, models = image.header.get_axis(0), image.header.get_axis(1)
    assert list(scalars.name) == ["a", "b", "c"]
    assert len(models) == N_FSLR
    data = np.asarray(image.get_fdata(), dtype=np.float64)
    assert data.shape == (3, N_FSLR) and np.isfinite(data).all()
    fslr_area, ico4_area = vertex_areas()
    np.testing.assert_allclose(data @ fslr_area, maps @ ico4_area, rtol=1e-5)


def test_a_constant_map_stays_constant_and_the_wall_stays_missing(tmp_path):
    (path,) = save_map(np.full(N, 3.25), tmp_path / "flat.dscalar.nii")
    data = np.asarray(nib.load(path).get_fdata()).ravel()
    overlap = load_overlap().tocsr()
    cortex = cortex_mask().astype(np.float64)
    share = (overlap @ cortex) / np.asarray(overlap.sum(axis=1)).ravel()
    # All of an fsLR vertex's cover on cortex: the value; all on the wall: missing.
    np.testing.assert_allclose(data[share == 1.0], 3.25)
    assert np.isnan(data[share == 0.0]).all()
    # Between, by the share on cortex (clear of the 0.5 cut, where rounding decides).
    assert np.isnan(data[share < 0.49]).all() and np.isfinite(data[share > 0.51]).all()


# --- tables ------------------------------------------------------------------


def test_a_vertex_table_has_one_row_per_vertex_and_says_where_it_is(tmp_path):
    rng = np.random.default_rng(2)
    values = rng.random(N)
    (path,) = save_map(values, tmp_path / "seed.csv", names="profile", atlas=["Desikan", "Yeo7"])
    header, rows = _read_table(path)
    desikan, yeo = load_atlas("Desikan"), load_atlas("Yeo7")
    assert header == [
        "vertex",
        "hemisphere",
        "fsaverage4_vertex",
        "cortex",
        desikan.name,
        yeo.name,
        "profile",
    ]
    assert len(rows) == N
    order = np.concatenate(fsaverage4_order())
    cortex = cortex_mask()
    for vertex in (0, 1234, HALF, N - 1):
        row = rows[vertex]
        assert int(row[0]) == vertex and row[1] == ("L" if vertex < HALF else "R")
        assert int(row[2]) == order[vertex] and int(row[3]) == int(cortex[vertex])
        label = desikan.labels[vertex]
        assert row[4] == (desikan.names[label - 1] if label else "")
        if cortex[vertex]:
            assert float(row[6]) == pytest.approx(values[vertex], rel=1e-9)
        else:
            assert row[6] == ""  # the medial wall: a missing value, not a zero


def test_a_tsv_is_tab_separated(tmp_path):
    (path,) = save_map(np.ones(N), tmp_path / "ones.tsv", mask=None)
    header, rows = _read_table(path)
    assert header[-1] == "value" and {row[-1] for row in rows} == {"1"}


# --- region means and region tables -------------------------------------------


def test_region_means_are_area_weighted_and_skip_missing_values():
    atlas = load_atlas("Desikan")
    area = vertex_areas()[1]
    rng = np.random.default_rng(3)
    values = rng.random(N)
    values[:10] = np.nan
    means = region_means(values, atlas)
    keep = cortex_mask() & np.isfinite(values)
    for region in (1, 17, atlas.n_regions):
        member = (np.asarray(atlas.labels) == region) & keep
        expected = (area[member] * values[member]).sum() / area[member].sum()
        assert means[region - 1] == pytest.approx(expected, rel=1e-12)
    several = region_means(np.vstack([values, 2 * values]), atlas)
    np.testing.assert_allclose(several[1], 2 * means, rtol=1e-12)
    assert np.isnan(region_means(np.full(N, np.nan), atlas)).all()


def test_a_region_seed_averaged_over_another_region_is_the_atlas_mean():
    """region_means(seed(region=k)) reads the off-diagonal row k of to_atlas(how='mean')."""
    cc = sbci.example()
    atlas = load_atlas("Desikan")
    matrix = cc.to_atlas(atlas, how="mean")
    for k in (0, 40):
        profile = cc.seed(region=atlas.region_mask(k + 1))
        row = region_means(profile, atlas, area=cc.area, mask=cc.mask)
        others = np.arange(atlas.n_regions) != k
        np.testing.assert_allclose(row[others], matrix[k, others], rtol=1e-6)


def test_fisher_z_averages_correlations_as_to_atlas_does():
    atlas = load_atlas("Desikan")
    z = np.random.default_rng(4).standard_normal(N) * 0.5
    np.testing.assert_allclose(
        region_means(np.tanh(z), atlas, fisher_z=True), np.tanh(region_means(z, atlas)), rtol=1e-6
    )


def test_region_tables_label_vectors_and_matrices(tmp_path):
    atlas = load_atlas("Desikan")
    vector = np.linspace(0, 1, atlas.n_regions)
    vector[3] = np.nan
    path = save_regions(vector, "Desikan", tmp_path / "v.csv", names="coupling")
    header, rows = _read_table(path)
    assert header == ["label", "region", "coupling"] and len(rows) == atlas.n_regions
    assert rows[0][:2] == ["1", atlas.names[0]] and rows[3][2] == ""

    path = save_regions({"a": vector, "b": 2 * vector}, atlas, tmp_path / "two.tsv")
    header, rows = _read_table(path)
    assert header == ["label", "region", "a", "b"]
    assert float(rows[5][3]) == pytest.approx(2 * vector[5])

    matrix = np.arange(atlas.n_regions**2, dtype=float).reshape(atlas.n_regions, -1)
    path = save_regions(matrix, atlas, tmp_path / "m.csv")
    header, rows = _read_table(path)
    assert header == ["region", *atlas.names]
    assert rows[2][0] == atlas.names[2] and float(rows[2][5]) == matrix[2, 4]


# --- the results of a test ----------------------------------------------------


def test_a_local_test_writes_one_row_per_component(tmp_path):
    rng = np.random.default_rng(5)
    n = 40
    covariate = rng.standard_normal(n)
    scores = 0.5 * covariate[:, None] + rng.standard_normal((n, 3))
    scores[:, 2] = 1.0  # constant: untestable
    result = sbci.local_test(scores, covariate)
    path = result.to_table(tmp_path / "test.csv", names=["intercept", "age"])
    header, rows = _read_table(path)
    assert header[:6] == ["component", "statistic", "pvalue", "adjusted", "partial_r2", "tested"]
    assert header[6:] == ["estimate", "se", "ci_low", "ci_high", "coef_intercept", "coef_age"]
    assert {row[5] for row in rows} == {"age"}
    assert [row[0] for row in rows] == ["0", "1", "2"]
    assert float(rows[0][2]) == pytest.approx(result.pvalue[0], rel=1e-9)
    assert rows[2][1:4] == ["", "", ""]
    with pytest.raises(ValueError, match="3 names for the design's 2 columns"):
        result.to_table(tmp_path / "x.csv", names=["a", "b", "c"])


# --- refusals, before anything is computed --------------------------------------


@pytest.mark.parametrize(
    "name, message",
    [
        ("x.nii", "does not say what to write"),
        ("x.dscalar.nii.gz", "uncompressed"),
        ("x.txt", "does not say what to write"),
    ],
)
def test_a_name_that_says_no_form_is_refused(tmp_path, name, message):
    with pytest.raises(ValueError, match=message):
        save_map(np.zeros(N), tmp_path / name)


def test_shapes_and_names_are_checked(tmp_path):
    with pytest.raises(ValueError, match="one value per grid vertex"):
        save_map(np.zeros(100), tmp_path / "x.csv")
    with pytest.raises(ValueError, match="is a connectome, not maps"):
        save_map(np.zeros((N, N), dtype=np.float32), tmp_path / "x.csv")
    with pytest.raises(ValueError, match="2 names for 3 maps"):
        save_map(np.zeros((3, N)), tmp_path / "x.csv", names=["a", "b"])
    with pytest.raises(ValueError, match="must differ"):
        save_map(np.zeros((2, N)), tmp_path / "x.csv", names=["a", "a"])
    with pytest.raises(ValueError, match="two columns named"):
        save_map(np.zeros(N), tmp_path / "x.csv", names="vertex")
    with pytest.raises(ValueError, match="atlas= adds region columns to a table"):
        save_map(np.zeros(N), tmp_path / "x.dscalar.nii", atlas="Desikan")
    with pytest.raises(ValueError, match="boolean array"):
        save_map(np.zeros(N), tmp_path / "x.csv", mask=np.ones(N))
    with pytest.raises(FileNotFoundError, match="no such directory"):
        save_map(np.zeros(N), tmp_path / "missing" / "x.csv")
    with pytest.raises(ValueError, match="region_means first"):
        save_regions(np.zeros(N), "Desikan", tmp_path / "x.csv")
    with pytest.raises(ValueError, match="Desikan|aparc"):
        save_regions(np.zeros(7), "Desikan", tmp_path / "x.csv")
    # One map a column, as Reduction.basis holds its components.
    (path,) = save_map(np.zeros((N, 2)), tmp_path / "basis.csv", mask=None)
    assert _read_table(path)[0][-2:] == ["value_1", "value_2"]


def test_an_uppercase_ending_names_the_form_too(tmp_path):
    (path,) = save_map(np.ones(N), tmp_path / "ONES.CSV", mask=None)
    header, rows = _read_table(path)
    assert header[-1] == "value" and len(rows) == N


# --- the fifth review -------------------------------------------------------------

# FreeSurfer's fsaverage4 sphere, its first twelve vertices: the icosahedron's
# corners, in FreeSurfer's order, at radius 100 (from FreeSurfer 7.4.1).
FSAVERAGE4_CORNERS = np.array(
    [
        [0.0, 0.0, 100.0],
        [89.44, 0.0, 44.72],
        [27.64, 85.07, 44.72],
        [-72.36, 52.57, 44.72],
        [-72.36, -52.57, 44.72],
        [27.64, -85.07, 44.72],
        [72.36, -52.57, -44.72],
        [72.36, 52.57, -44.72],
        [-27.64, 85.07, -44.72],
        [-89.44, 0.0, -44.72],
        [-27.64, -85.07, -44.72],
        [0.0, 0.0, -100.0],
    ]
)


def test_the_order_puts_the_icosahedrons_corners_where_freesurfer_has_them():
    """The face digest fixes the order up to a symmetry of the mesh; the corners fix the symmetry.

    The bundled standard-sphere position of each grid vertex, placed in the
    fsaverage4 order, has to put fsaverage4's first twelve vertices at the
    icosahedron's corners as FreeSurfer numbers them: a rotated relabeling,
    which carries triangles onto triangles as well, would not.
    """
    from importlib import resources

    with np.load(resources.files("sbci.data.surfaces") / "fsaverage_sphere_ico4.npz") as data:
        positions = np.asarray(data["vertices"], dtype=np.float64)
    for side, order in enumerate(fsaverage4_order()):
        block = positions[side * HALF : (side + 1) * HALF]
        placed = np.empty_like(block)
        placed[order] = 100.0 * block / np.linalg.norm(block, axis=1, keepdims=True)
        assert np.abs(placed[:12] - FSAVERAGE4_CORNERS).max() < 2.0  # neighbours lie ~7 apart
        rolled = np.roll(order, 1)  # any other numbering puts other vertices there
        wrong = np.empty_like(block)
        wrong[rolled] = block
        assert np.abs(100.0 * wrong[:12] - FSAVERAGE4_CORNERS).max() > 2.0


def test_an_atlas_of_another_size_is_refused_and_nothing_is_left(tmp_path):
    from sbci.atlas import Atlas

    desikan = load_atlas("Desikan")
    for labels in (desikan.labels[:5000], np.concatenate([desikan.labels, desikan.labels[:10]])):
        atlas = Atlas("odd", labels, desikan.names)
        with pytest.raises(ValueError, match=f"labels {labels.size} vertices, but the grid has"):
            save_map(np.zeros(N), tmp_path / "odd.csv", atlas=atlas)
    assert not list(tmp_path.iterdir())


def test_a_write_that_fails_part_way_leaves_nothing(tmp_path, monkeypatch):
    from sbci import export

    calls = {"n": 0}
    original = export._cell

    def failing(value):
        calls["n"] += 1
        if calls["n"] > 500:
            raise RuntimeError("disk full")
        return original(value)

    monkeypatch.setattr(export, "_cell", failing)
    with pytest.raises(RuntimeError, match="disk full"):
        save_map(np.zeros(N), tmp_path / "half.csv")
    assert not list(tmp_path.iterdir())


def test_a_bids_hemisphere_entity_is_set_per_hemisphere(tmp_path):
    left, right = save_map(np.zeros(N), tmp_path / "sub-01_hemi-L_space-fsaverage4_map.func.gii")
    assert left.name == "sub-01_hemi-L_space-fsaverage4_map.func.gii"
    assert right.name == "sub-01_hemi-R_space-fsaverage4_map.func.gii"
    assert nib.load(right).meta["AnatomicalStructurePrimary"] == "CortexRight"


def test_infinities_are_refused_and_masked_entries_are_missing(tmp_path):
    values = np.ones(N)
    values[7] = np.inf
    with pytest.raises(ValueError, match="1 infinite value"):
        save_map(values, tmp_path / "inf.csv")
    with pytest.raises(ValueError, match="infinite"):
        region_means(values, "Desikan")
    masked = np.ma.masked_array(np.ones(N), mask=np.arange(N) < 3)
    (path,) = save_map(masked, tmp_path / "masked.csv", mask=None)
    _, rows = _read_table(path)
    assert [row[-1] for row in rows[:4]] == ["", "", "", "1"]


def test_a_string_of_names_is_one_name_not_its_letters(tmp_path):
    rng = np.random.default_rng(6)
    covariate = rng.standard_normal(30)
    result = sbci.local_test(rng.standard_normal((30, 2)), covariate)
    with pytest.raises(ValueError, match="1 names for the design's 2 columns"):
        result.to_table(tmp_path / "t.csv", names="ab")


# --- the eighth review ------------------------------------------------------------


def test_a_pair_is_not_left_half_written(tmp_path):
    """With the left name taken, the right hemisphere used to be written alone."""
    values = np.random.default_rng(0).standard_normal(5124)
    (tmp_path / "map.L.func.gii").mkdir()
    with pytest.raises(IsADirectoryError):
        sbci.save_map(values, tmp_path / "map.func.gii")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["map.L.func.gii"]


def test_a_write_goes_through_a_link_and_not_onto_a_read_only_file(tmp_path):
    values = np.random.default_rng(0).standard_normal(5124)
    (tmp_path / "real").mkdir()
    target = tmp_path / "real" / "map.csv"
    target.write_text("old")
    (tmp_path / "link.csv").symlink_to(target)
    sbci.save_map(values, tmp_path / "link.csv")
    assert (tmp_path / "link.csv").is_symlink() and target.read_text() != "old"
    locked = tmp_path / "locked.csv"
    locked.write_text("keep")
    locked.chmod(0o400)
    with pytest.raises(PermissionError, match="read-only"):
        sbci.save_map(values, locked)
    assert locked.read_text() == "keep"


def test_a_write_through_a_link_checks_the_folder_it_lands_in(tmp_path):
    """The link's folder was checked; the write happens beside the target, so that one counts."""
    values = np.random.default_rng(0).standard_normal(5124)
    (tmp_path / "dangling.csv").symlink_to(tmp_path / "gone" / "map.csv")
    with pytest.raises(FileNotFoundError, match="no such directory: .*gone, where the link"):
        sbci.save_map(values, tmp_path / "dangling.csv")
    shut = tmp_path / "shut"
    shut.mkdir()
    (shut / "map.csv").write_text("old")
    shut.chmod(0o500)
    try:
        with pytest.raises(PermissionError, match="moved into place"):
            sbci.save_map(values, shut / "map.csv")
    finally:
        shut.chmod(0o700)
    assert (shut / "map.csv").read_text() == "old"
