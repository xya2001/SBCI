"""The public API as a newcomer meets it.

These tests exist because the README's promises and the package's namespace
drifted apart once already: `sbci.stats.local_test` was documented while a
plain ``import sbci`` raised :class:`AttributeError` for it. Each test below
pins one thing a first-time user does in their first ten minutes.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import sbci
from sbci import grid

N = 5124


@pytest.fixture
def cc(sc_metadata):
    """A connectome on the real grid, with complete metadata so it round-trips.

    The values are arbitrary -- nothing here tests numerics -- but the grid has
    to be the real one because the bundled atlases are defined on it.
    """
    rng = np.random.default_rng(0)
    return sbci.ContinuousConnectome(
        rng.random(grid.condensed_size(N)).astype(np.float32),
        np.full(N, 1.0 / N),
        np.ones(N, bool),
        sc_metadata,
    )


# --- the namespace ---------------------------------------------------------


@pytest.mark.parametrize("name", sbci.__all__)
def test_everything_in_all_actually_resolves(name):
    """``__all__`` is a promise; the lazy layer has to keep it."""
    assert getattr(sbci, name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "stats",
        "smoothing",
        "plotting",
        "coupling",
        "alignment",
        "reduction",
        "parcellation",
        "validate",
        "grid",
        "io",
        "errors",
    ],
)
def test_submodules_reachable_after_plain_import(name):
    """``sbci.stats.local_test`` has to work after ``import sbci`` alone.

    The README documents it that way, and a submodule that needs its own
    import statement is a documented API that does not exist.
    """
    assert getattr(sbci, name).__name__ == f"sbci.{name}"


def test_star_import_matches_all():
    namespace: dict = {}
    exec("from sbci import *", namespace)  # noqa: S102 - that is the thing under test
    assert set(sbci.__all__) - {"__version__"} <= set(namespace)


def test_dir_includes_the_lazy_names():
    """Tab completion in a notebook has to find them."""
    listed = set(dir(sbci))
    assert {"smooth", "local_test", "plot_surface", "stats"} <= listed


def test_importing_sbci_stays_light():
    """The lazy layer exists so a plain import does not drag in the world.

    matplotlib alone costs several hundred milliseconds, and a user who only
    wants to parcellate a file should never pay it.
    """
    heavy = ("matplotlib", "scipy", "nilearn", "igl")
    code = f"import sys, sbci; print([m for m in {heavy} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_unknown_attribute_suggests_a_real_one():
    typo = "lod_atlas"
    with pytest.raises(AttributeError, match="Did you mean 'load_atlas'"):
        getattr(sbci, typo)


# --- names where an object is wanted ---------------------------------------


def test_to_atlas_takes_a_name(cc):
    """``cc.to_atlas("Desikan")`` should not require importing load_atlas."""
    assert cc.to_atlas("Desikan").shape == (68, 68)


def test_to_atlas_name_and_object_agree(cc):
    by_name = cc.to_atlas("Desikan")
    by_object = cc.to_atlas(sbci.load_atlas("Desikan"))
    np.testing.assert_array_equal(by_name, by_object)


def test_seed_takes_an_atlas_region_pair(cc):
    profile = cc.seed(region=("Desikan", "LH_bankssts"))
    assert profile.shape == (N,)


def test_seed_pair_matches_an_explicit_mask(cc):
    mask = sbci.load_atlas("Desikan").region_mask("LH_bankssts")
    np.testing.assert_allclose(cc.seed(region=("Desikan", "LH_bankssts")), cc.seed(region=mask))


def test_region_mask_by_name_and_id_agree():
    atlas = sbci.load_atlas("Desikan")
    by_name = atlas.region_mask("LH_bankssts")
    by_id = atlas.region_mask(int(atlas.region_ids[0]))
    np.testing.assert_array_equal(by_name, by_id)


def test_region_mask_is_case_insensitive():
    atlas = sbci.load_atlas("Desikan")
    np.testing.assert_array_equal(
        atlas.region_mask("lh_bankssts"), atlas.region_mask("LH_bankssts")
    )


# --- failure modes a newcomer will hit -------------------------------------


def test_atlas_typo_suggests_the_documented_spelling():
    with pytest.raises(ValueError, match="Did you mean 'Schaefer200'"):
        sbci.load_atlas("Shaefer200")


def test_atlas_error_no_longer_dumps_every_name():
    """The old message printed all 44 atlases, which buries the answer."""
    with pytest.raises(ValueError) as caught:
        sbci.load_atlas("Shaefer200")
    assert len(str(caught.value)) < 200


def test_region_typo_suggests_a_real_region():
    with pytest.raises(ValueError, match="Did you mean 'LH_bankssts'"):
        sbci.load_atlas("Desikan").region_mask("LH_banksts")


def test_seed_with_a_float_array_says_what_it_wanted(cc):
    with pytest.raises(TypeError, match="boolean mask"):
        cc.seed(region=np.zeros(N, dtype=float))


@pytest.mark.parametrize(
    "call",
    [
        lambda: sbci.load("/etc/hostname"),
        lambda: sbci.load_atlas("no-such-atlas"),
    ],
    ids=["unrecognized file", "unknown atlas"],
)
def test_data_errors_share_one_base_class(call):
    """One ``except sbci.SbciError`` has to cover a pipeline's input problems."""
    with pytest.raises(sbci.SbciError):
        call()


def test_smoothing_without_endpoints_is_an_sbci_error(cc):
    with pytest.raises(sbci.MissingDataError):
        cc.smooth(kernel="rdk")


def test_format_error_is_still_a_key_error():
    """Code written against the built-in exception keeps working."""
    assert issubclass(sbci.FormatError, KeyError)
    assert issubclass(sbci.MetadataError, ValueError)


def test_format_error_prints_without_key_error_quotes():
    """``KeyError`` reprs its message, which reads badly in a traceback."""
    assert str(sbci.FormatError("no /connectivity dataset")) == "no /connectivity dataset"


# --- the entry point -------------------------------------------------------


def test_load_is_the_classmethod(tmp_path, cc):
    path = tmp_path / "sub-01_sc.h5"
    cc.save(path)
    np.testing.assert_array_equal(sbci.load(path).data, sbci.ContinuousConnectome.load(path).data)
