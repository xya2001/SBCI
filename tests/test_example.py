"""The example connectome, which is what a new user runs first.

`sbci download` needs the data release, so until that exists `sbci.example()`
is the only thing anyone can run. It therefore has to be exactly as valid as a
real file, and it has to say clearly that it is not one.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import sbci
from sbci.cli import main
from sbci.validate import validate_file


@pytest.fixture(scope="module")
def sc():
    return sbci.example("sc")


@pytest.fixture(scope="module")
def fc():
    return sbci.example("fc")


def test_it_is_on_the_real_grid(sc):
    assert sc.n_vertices == sbci.spec.N_VERTICES


def test_it_is_reproducible():
    np.testing.assert_array_equal(sbci.example(seed=3).data, sbci.example(seed=3).data)


def test_different_seeds_differ():
    assert not np.array_equal(sbci.example(seed=1).data, sbci.example(seed=2).data)


def test_sc_is_a_density_of_unit_mass(sc):
    dense = sc.dense().astype(np.float64)
    assert float(sc.area @ dense @ sc.area) == pytest.approx(1.0, abs=1e-5)


def test_sc_is_nonnegative(sc):
    assert sc.data.min() >= 0.0


def test_fc_is_signed(fc):
    assert fc.data.min() < 0 < fc.data.max()


def test_fc_stays_within_correlation_bounds(fc):
    assert fc.data.min() >= -1.0 and fc.data.max() <= 1.0


@pytest.mark.parametrize("modality", ["sc", "fc"])
def test_the_medial_wall_carries_nothing(modality):
    cc = sbci.example(modality)
    assert float(np.abs(cc.dense()[~cc.mask]).sum()) == 0.0


def test_the_mask_is_the_real_medial_wall(sc):
    """Not invented: it is where the bundled Desikan atlas assigns no region."""
    np.testing.assert_array_equal(sc.mask, sbci.load_atlas("Desikan").labels != 0)


@pytest.mark.parametrize("modality", ["sc", "fc"])
def test_it_passes_the_validator(modality, tmp_path):
    path = tmp_path / f"sub-example_{modality}.h5"
    sbci.example(modality).save(path)
    failed = [c for c in validate_file(path) if not c.passed]
    assert not failed, [str(c) for c in failed]


def test_the_metadata_says_it_is_synthetic(sc):
    """Someone must never mistake this for measured data."""
    assert "synthetic" in sc.metadata.fields["pipeline_version"]
    assert sc.metadata.fields["streamline_count"] == sbci.examples.N_STREAMLINES
    assert "drawn at random" in sc.metadata.fields["streamline_weighting"]
    assert sc.metadata.fields["kernel"] == "shk" and sc.metadata.fields["bandwidth"] == 0.005


def test_sc_carries_synthetic_endpoints_that_end_in_cortex(sc):
    """Every endpoint has a continuous position, none of them on the medial wall."""
    endpoints = sc.endpoints
    assert sc.has_endpoints and endpoints.has_positions
    assert endpoints.n_streamlines == sbci.examples.N_STREAMLINES
    for vertex in (endpoints.global_vertex_in, endpoints.global_vertex_out):
        assert sc.mask[vertex].all()
    half = sc.n_vertices // 2
    np.testing.assert_array_equal(endpoints.surf_in, endpoints.global_vertex_in >= half)
    np.testing.assert_array_equal(endpoints.surf_out, endpoints.global_vertex_out >= half)
    crossing = np.mean(endpoints.surf_in != endpoints.surf_out)
    assert 0.15 < crossing < 0.25  # about a fifth of the streamlines cross hemispheres


def test_re_smoothing_the_endpoints_gives_the_file_back(sc):
    """The density *is* the default kernel applied to the endpoints, so smooth() reproduces it."""
    again = sc.smooth(kernel="shk", mask_medial_wall=True)
    np.testing.assert_array_equal(again.data, sc.data)
    assert again.metadata.fields["bandwidth"] == sc.metadata.fields["bandwidth"]


def test_the_endpoints_survive_the_file(sc, tmp_path):
    """The file stores barycentric weights as float32; the example is built from float32 weights."""
    path = tmp_path / "sub-example_sc.h5"
    sc.save(path)
    back = sbci.load(path)
    assert back.has_endpoints and back.endpoints.n_streamlines == sc.endpoints.n_streamlines
    np.testing.assert_array_equal(back.endpoints.bary_in, sc.endpoints.bary_in)
    np.testing.assert_array_equal(back.endpoints.global_vertex_out, sc.endpoints.global_vertex_out)
    np.testing.assert_array_equal(back.smooth(kernel="shk", mask_medial_wall=True).data, sc.data)


def test_fc_carries_no_endpoints(fc):
    assert not fc.has_endpoints


def test_the_streamline_count_is_a_parameter():
    cc = sbci.example(n_streamlines=500)
    assert cc.endpoints.n_streamlines == 500 and cc.metadata.fields["streamline_count"] == 500
    with pytest.raises(ValueError, match="n_streamlines"):
        sbci.example(n_streamlines=0)


def test_repeated_calls_hand_out_independent_copies():
    first, second = sbci.example(seed=5, n_streamlines=200), sbci.example(seed=5, n_streamlines=200)
    np.testing.assert_array_equal(first.data, second.data)
    first.data[0] = 1.0
    first.endpoints.vtx_in[0] = -1
    assert second.data[0] != 1.0 and second.endpoints.vtx_in[0] != -1


def test_two_examples_can_be_aligned_by_their_endpoints():
    """The ConSEAL line of the README runs on the synthetic subjects."""
    subjects = [sbci.example(seed=s, n_streamlines=2000) for s in (0, 1)]
    result = sbci.endpoints_align(subjects, template=0, max_iterations=1)
    assert len(result.warps) == 2 and result.template.shape == (sc_vertices(), sc_vertices())
    assert result.costs[1][-1] <= result.costs[1][0]


def sc_vertices():
    return sbci.spec.N_VERTICES


def test_an_unknown_modality_is_refused():
    with pytest.raises(ValueError, match="modality must be one of"):
        sbci.example("dwi")


def test_the_readme_walkthrough_runs(sc):
    """Every line of the five-minute start, minus the download and the 16.9 GB."""
    assert sc.to_atlas("Schaefer200").shape == (200, 200)
    assert sc.seed(vertex=1234).shape == (sbci.spec.N_VERTICES,)
    assert sc.seed(region=("Desikan", "LH_bankssts")).shape == (sbci.spec.N_VERTICES,)


def test_cli_writes_a_file_that_validates(tmp_path, capsys):
    path = tmp_path / "example.h5"
    assert main(["example", "--out", str(path)]) == 0
    assert "not measured data" in capsys.readouterr().out
    assert not [c for c in validate_file(path) if not c.passed]


def test_cli_info_reports_the_modality(tmp_path, capsys):
    path = tmp_path / "example.h5"
    main(["example", "--out", str(path)])
    capsys.readouterr()
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "modality        sc" in out
    assert "synthetic-example" in out


def test_cli_atlases_lists_and_filters(capsys):
    assert main(["atlases", "--match", "Yeo"]) == 0
    out = capsys.readouterr().out
    assert "Yeo2011_7Networks_N1000" in out and "Schaefer" not in out


def test_cli_atlases_reports_no_match(capsys):
    assert main(["atlases", "--match", "nonesuch"]) == 1
    assert "no bundled atlas matches" in capsys.readouterr().out


# From the compute-node calibration of the fixture below: the bundle's own mass
# tracks age at 0.98, its home region's self-connectivity at 0.76 (diluted by
# everything else the region carries), and two independent examples correlate
# at 0.63 over the Desikan matrix where cohort subjects sit above 0.95.
BUNDLE_MASS_CORRELATION = 0.9
REGION_CORRELATION = 0.6


@pytest.fixture(scope="module")
def cohort():
    """Six small subjects with a strong planted effect, so the checks below are cheap and sharp."""
    return sbci.example_cohort(n_subjects=6, seed=0, effect=0.9, n_streamlines=2000)


def test_a_cohort_is_reproducible_and_says_it_is_synthetic(cohort):
    again = sbci.example_cohort(n_subjects=6, seed=0, effect=0.9, n_streamlines=2000)
    np.testing.assert_array_equal(again.age, cohort.age)
    np.testing.assert_array_equal(again.connectomes[3].data, cohort.connectomes[3].data)
    assert cohort.n_subjects == 6 and cohort.age.min() >= 20 and cohort.age.max() <= 80
    assert all(
        "synthetic-cohort" in cc.metadata.fields["pipeline_version"] for cc in cohort.connectomes
    )
    assert all(cc.endpoints.n_streamlines == 2000 for cc in cohort.connectomes)


def test_cohort_subjects_share_anatomy_unlike_independent_examples(cohort):
    """Subjects of one cohort resemble each other far more than two examples do.

    Two independent examples correlate at about 0.63 over the Desikan matrix;
    subjects drawn around the same bundles sit well above that.
    """
    atlas = sbci.load_atlas("Desikan")
    iu = np.triu_indices(atlas.n_regions)
    within = np.stack([cc.to_atlas(atlas)[iu] for cc in cohort.connectomes])
    assert np.corrcoef(within)[np.triu_indices(6, 1)].min() > 0.85
    assert not np.array_equal(cohort.connectomes[0].data, cohort.connectomes[1].data)


def test_the_planted_bundle_rises_with_age(cohort):
    """The truth is recoverable by a plain regression, both on the bundle and on its region."""
    assert cohort.truth.shape == (sbci.spec.N_VERTICES,) and cohort.truth.max() == pytest.approx(
        1.0
    )
    # the bundle's own mass: the density integrated against its field at both ends
    mass = np.array(
        [cohort.truth @ cc.dense().astype(np.float64) @ cohort.truth for cc in cohort.connectomes]
    )
    assert np.corrcoef(mass, cohort.age)[0, 1] > BUNDLE_MASS_CORRELATION
    # and, diluted by everything else the region carries, its home region's self-connectivity
    atlas = sbci.load_atlas("Desikan")
    half = sbci.spec.N_VERTICES_PER_HEMI
    region = int(atlas.labels[np.argmax(cohort.truth[:half])]) - 1
    within = np.array([cc.to_atlas(atlas)[region, region] for cc in cohort.connectomes])
    assert np.corrcoef(within, cohort.age)[0, 1] > REGION_CORRELATION


def test_an_fc_cohort_matches_its_sc_cohort(cohort):
    fc = sbci.example_cohort(n_subjects=6, seed=0, effect=0.9, modality="fc")
    np.testing.assert_array_equal(fc.age, cohort.age)
    assert fc.effect_bundle == cohort.effect_bundle
    assert all(not cc.has_endpoints and cc.data.min() < 0 for cc in fc.connectomes)


def test_cohort_arguments_are_checked():
    with pytest.raises(ValueError, match="effect"):
        sbci.example_cohort(n_subjects=2, effect=1.0)
    with pytest.raises(ValueError, match="n_subjects"):
        sbci.example_cohort(n_subjects=0)


@pytest.mark.skipif(
    not os.environ.get("SBCI_SLOW_TESTS"),
    reason="three minutes of FPCA on ico4; set SBCI_SLOW_TESTS=1",
)
def test_reduce_and_local_test_recover_the_planted_bundle():
    """The README's end-to-end analysis, at its defaults, with its known answer."""
    cohort = sbci.example_cohort(n_subjects=10, seed=0)
    reduction = sbci.reduce(cohort.connectomes, rank=4)
    result = sbci.local_test(reduction.scores, cohort.age)
    found = result.significant(0.05)
    assert found.size >= 1
    assert abs(np.corrcoef(reduction.scores[:, found[0]], cohort.age)[0, 1]) > 0.8
    effect = result.effect_map(reduction, alpha=0.05)
    assert np.corrcoef(effect, cohort.truth)[0, 1] > 0.5
