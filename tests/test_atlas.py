"""The bundled atlases, and the background-folding that makes their counts right."""

from __future__ import annotations

import re

import numpy as np
import pytest

from sbci import load_atlas
from sbci.atlas import ALIASES, list_atlases, resolve
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

BACKGROUND = re.compile(r"(missing|unknown|background|medial[_ ]?wall|\?\?\?)", re.IGNORECASE)


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


#: The one bundled atlas that assigns every vertex, medial wall included.
FULL_COVERAGE = {"PALS_B12_Lobes"}


@pytest.mark.parametrize("name", list_atlases())
def test_coverage_is_plausible(name):
    """Every atlas covers some of the surface, and only one covers all of it."""
    atlas = load_atlas(name)
    assert 0.0 < atlas.coverage <= 1.0

    if name in FULL_COVERAGE:
        assert atlas.coverage == 1.0, f"{name} was expected to assign every vertex"
    else:
        assert (atlas.labels == 0).any(), "the medial wall should be unassigned"
        assert atlas.coverage < 1.0


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
