"""The bundled atlases, and the background-folding that makes their counts right."""

from __future__ import annotations

import re

import numpy as np
import pytest

from sbci import load_atlas
from sbci.atlas import ALIASES, Atlas, list_atlases, resolve
from sbci.parcellation import region_weights

# Published region counts, from each atlas's own paper. These are the check
# that the two hemispheres' background entries were both folded into label 0:
# miss one and every count here is one too high.
CANONICAL_COUNTS = {
    "Desikan": 68,
    "Destrieux": 148,
    "Glasser": 360,
    "Gordon": 333,
    "Yeo7": 14,
    "Yeo17": 34,
    "Schaefer100": 100,
    "Schaefer200": 200,
    "Schaefer400": 400,
}

BACKGROUND = re.compile(r"(missing|unknown|background|medial[_ .]?wall|\?\?\?)", re.IGNORECASE)


def test_atlases_are_bundled():
    available = list_atlases()
    assert len(available) >= 40, f"only {len(available)} atlases bundled"
    assert "aparc" in available
    assert "Schaefer2018_200Parcels_7Networks_order" in available


@pytest.mark.parametrize(("name", "expected"), sorted(CANONICAL_COUNTS.items()))
def test_region_count_matches_the_published_atlas(name, expected):
    assert load_atlas(name).n_regions == expected


@pytest.mark.parametrize("name", list_atlases())
def test_every_atlas_is_well_formed(name):
    """Labels are 0..K with no gaps, and names line up with the non-zero ones."""
    atlas = load_atlas(name)

    assert atlas.labels.shape == (5124,), "atlas is not on the ico4 grid"
    assert atlas.labels.dtype == np.int32
    assert atlas.labels.min() >= 0

    ids = atlas.region_ids
    np.testing.assert_array_equal(
        ids, np.arange(1, ids.size + 1), err_msg="region ids are not contiguous from 1"
    )
    assert len(atlas.names) == atlas.n_regions
    assert all(isinstance(n, str) and n for n in atlas.names)


@pytest.mark.parametrize("name", list_atlases())
def test_no_background_region_survives(name):
    """The regression test for the bug this conversion exists to avoid.

    Most FreeSurfer-derived atlases carry two background entries, one per
    hemisphere, and the right-hemisphere one sits mid-list with a non-zero
    label. If it is kept, every parcellated matrix gains a row and column of
    pure medial wall.
    """
    leaked = [n for n in load_atlas(name).names if BACKGROUND.search(n)]
    assert not leaked, f"{name} still lists background regions: {leaked}"


@pytest.mark.parametrize("name", list_atlases())
def test_coverage_is_plausible(name):
    """Every atlas covers some of the surface and none covers all of it.

    PALS_B12_Lobes used to: its two ``MEDIAL.WALL`` entries, spelt with a dot,
    slipped past the background pattern and were kept as regions until the
    review of October 2026.
    """
    atlas = load_atlas(name)
    assert 0.0 < atlas.coverage < 1.0
    assert (atlas.labels == 0).any(), "the medial wall should be unassigned"


def test_pals_lobes_has_ten_lobes_and_a_medial_wall_of_its_own():
    """The module notes' numbers: five lobes a hemisphere, and the medial wall as PALS draws it.

    The notes called it a twelve-lobe atlas with no label 0, as it was while
    its ``MEDIAL.WALL`` entries were kept as regions.
    """
    lobes = load_atlas("PALS_B12_Lobes")
    wall = load_atlas("Desikan").labels == 0  # the pipeline's medial wall
    assert lobes.n_regions == 10 and sum(n.startswith("LH_") for n in lobes.names) == 5
    assert int((lobes.labels == 0).sum()) == 394
    assert int(wall.sum()) == 439 and int((wall & (lobes.labels != 0)).sum()) == 89


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Schaefer200", "Schaefer2018_200Parcels_7Networks_order"),
        ("schaefer200", "Schaefer2018_200Parcels_7Networks_order"),
        ("Desikan", "aparc"),
        ("desikan-killiany", "aparc"),
        ("DK", "aparc"),
        ("Glasser", "HCPMMP1"),
        ("aparc", "aparc"),
        ("APARC", "aparc"),
    ],
)
def test_alias_resolution(given, expected):
    assert resolve(given) == expected


def test_every_alias_points_at_a_bundled_atlas():
    available = set(list_atlases())
    missing = {k: v for k, v in ALIASES.items() if v not in available}
    assert not missing, f"aliases point at atlases that are not bundled: {missing}"


def test_unknown_atlas_lists_the_alternatives():
    with pytest.raises(ValueError, match="unknown atlas"):
        load_atlas("Brodmann42")


def test_region_weights_on_a_real_atlas():
    """One column per region, and the columns partition the assigned vertices."""
    atlas = load_atlas("Schaefer200")
    area = np.ones(5124)
    weights, ids = region_weights(atlas, area)

    assert weights.shape == (5124, 200)
    assert ids.size == 200
    # Each vertex belongs to at most one region.
    assert (weights > 0).sum(axis=1).max() == 1
    assert weights.sum() == pytest.approx((atlas.labels != 0).sum())
    # The sparse one-hot form equals the dense one written out by hand.
    dense = np.zeros((5124, 200))
    for column, region in enumerate(ids):
        dense[atlas.labels == region, column] = area[atlas.labels == region]
    np.testing.assert_array_equal(weights.toarray(), dense)


def test_to_atlas_on_the_full_grid(sc_metadata):
    """The brief's headline call: a real atlas on a real-sized connectome."""
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed

    rng = np.random.default_rng(0)
    n = 5124
    dense = rng.random((n, n), dtype=np.float32)
    dense = np.triu(dense, k=1)
    dense = dense + dense.T

    cc = ContinuousConnectome(
        data=to_condensed(dense),
        area=np.ones(n),
        mask=load_atlas("Schaefer200").labels != 0,
        metadata=sc_metadata,
    )
    assert cc.to_atlas(load_atlas("Schaefer200")).shape == (200, 200)


def test_region_ids_follow_the_names_even_when_a_region_is_empty():
    from sbci.atlas import Atlas

    atlas = Atlas(name="gappy", labels=np.array([1, 1, 3, 3, 0]), names=("A", "B", "C"))
    np.testing.assert_array_equal(atlas.region_ids, [1, 2, 3])
    assert atlas.n_regions == 3
    np.testing.assert_array_equal(atlas.region_mask("C"), atlas.labels == 3)
    with pytest.raises(ValueError, match="names"):
        Atlas(name="bad", labels=np.array([1, 4]), names=("A", "B"))


def test_bundled_labels_are_read_only():
    from sbci.atlas import load_atlas

    atlas = load_atlas("Desikan")
    with pytest.raises(ValueError):
        atlas.labels[0] = 5


def test_an_atlas_does_not_freeze_the_callers_label_array():
    from sbci.atlas import Atlas

    labels = np.zeros(5124, dtype=np.int32)
    Atlas(name="mine", labels=labels, names=())
    labels[0] = 0  # still writable
    assert labels.flags.writeable


def test_no_hemisphere_keeps_its_unassigned_cortex_as_a_region():
    """The PALS files call the right hemisphere's unassigned cortex RH_GYRUS, the left's LH_???.

    Kept as a region, it labelled the whole right hemisphere in OrbitoFrontal
    against 8% of the left (fourth review); it is background now, as LH_??? is.
    """
    import re

    import numpy as np

    from sbci.atlas import list_atlases, load_atlas

    for name in list_atlases():
        assert not any(
            re.fullmatch(r"[LR]H_GYRUS", n, re.IGNORECASE) for n in load_atlas(name).names
        )
    labels = np.asarray(load_atlas("PALS_B12_OrbitoFrontal").labels)
    left, right = int((labels[:2562] != 0).sum()), int((labels[2562:] != 0).sum())
    assert 0.5 < left / right < 2 and left + right < 0.1 * labels.size


# --- the review of 11 October 2026 -------------------------------------------------


def test_the_atlas_list_finds_the_short_names(capsys):
    """``sbci atlases --match Schaefer200`` matched nothing, though load_atlas takes the name."""
    from sbci.cli import main

    assert main(["atlases", "--match", "Schaefer200"]) == 0
    assert "Schaefer2018_200Parcels_7Networks_order (Schaefer200)" in capsys.readouterr().out
    assert main(["atlases", "--match", "desikan"]) == 0
    assert "aparc (Desikan" in capsys.readouterr().out


def test_an_unknown_region_is_an_sbci_error():
    """An unknown region raised a plain ValueError, which ``except sbci.SbciError`` missed."""
    import sbci
    from sbci.atlas import UnknownRegionError, load_atlas

    atlas = load_atlas("Desikan")
    for region in ("LH_banksts", 999):
        with pytest.raises(sbci.SbciError, match="has no region"):
            atlas.region_mask(region)
    with pytest.raises(UnknownRegionError, match="Did you mean 'LH_bankssts'"):
        atlas.region_mask("LH_banksts")
    with pytest.raises(ValueError):  # still the built-in it always was
        atlas.region_mask("LH_banksts")


def test_an_atlas_takes_whole_region_numbers_only():
    """1.5 was counted in region 1 by to_atlas and in no region by region_mask."""
    with pytest.raises(ValueError, match="whole numbers"):
        Atlas("floats", np.array([1.0, 1.5, 2.0, 0.0]), ("A", "B"))
    with pytest.raises(ValueError, match="whole numbers"):
        Atlas("nan", np.array([1.0, np.nan, 2.0, 0.0]), ("A", "B"))
    whole = Atlas("whole floats", np.array([1.0, 2.0, 2.0, 0.0]), ("A", "B"))
    assert whole.labels.dtype.kind == "i" and whole.labels.tolist() == [1, 2, 2, 0]


def test_region_names_and_atlas_names_are_found_however_they_are_written():
    """Neither the case of a region's name nor the spellings _key promises was pinned."""
    atlas = load_atlas("Desikan Killiany")
    for spelling in ("desikan-killiany", "DESIKAN_KILLIANY", "desikan"):
        assert load_atlas(spelling).name == atlas.name
    assert (atlas.region_mask("lh_BANKSSTS") == atlas.region_mask("LH_bankssts")).all()
