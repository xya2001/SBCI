"""Named designs, contrasts, standard errors and intervals, against statsmodels.

statsmodels is the independent implementation: its OLS standard errors and
intervals, its cluster-robust covariance with the same small-sample
correction, and its F tests of linear hypotheses. A named design must also
give the index-based test bit for bit, so that the published numbers stand.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.stats import Design, design, local_test

sm = pytest.importorskip("statsmodels.api")


def _cohort(n=120, seed=0):
    rng = np.random.default_rng(seed)
    site = rng.choice(["A", "B", "C"], size=n)
    age = rng.uniform(22, 37, n)
    motion = rng.gamma(2.0, 0.05, n)
    effect = {"A": 0.0, "B": 0.4, "C": -0.3}
    signal = np.array([effect[s] for s in site]) + 0.05 * (age - 30)
    scores = signal[:, None] * np.array([1.0, 0.5, 0.0]) + rng.standard_normal((n, 3))
    return scores, {"site": site, "age": age, "motion": motion}


# --- the design --------------------------------------------------------------------


def test_numbers_enter_as_they_are_and_text_as_dummies_against_a_reference():
    d = design({"age": [22.0, 30.0, 35.0, 28.0, 25.0], "site": ["B", "A", "C", "A", "B"]})
    assert d.names == ("intercept", "age", "site[B]", "site[C]")
    np.testing.assert_array_equal(d.matrix[:, 2], [1, 0, 0, 0, 1])
    assert d.columns("site") == [2, 3] and d.columns("site[C]") == [3] and d.columns(1) == [1]
    other = design({"site": ["B", "A", "C", "A", "B"]}, reference={"site": "C"})
    assert other.names == ("intercept", "site[A]", "site[B]")
    coded = design({"scanner": [1, 2, 2, 1, 3]}, categorical=["scanner"])
    assert coded.names == ("intercept", "scanner[2]", "scanner[3]")
    assert design({"female": [True, False, True]}).names == ("intercept", "female")
    with pytest.raises(ValueError, match="did you mean 'site'"):
        d.columns("sites")


def test_what_a_design_refuses():
    with pytest.raises(ValueError, match="age has 1 missing value"):
        design({"age": [22.0, np.nan, 30.0]})
    with pytest.raises(ValueError, match="site has 1 missing value"):
        design({"site": ["A", "NA", "B"]})
    with pytest.raises(ValueError, match="does not vary"):
        design({"age": [30.0, 30.0, 30.0]})
    with pytest.raises(ValueError, match="one level"):
        design({"site": ["A", "A", "A"]})
    with pytest.raises(ValueError, match="no level 'Z'"):
        design({"site": ["A", "B"]}, reference={"site": "Z"})
    with pytest.raises(ValueError, match="differ in length"):
        design({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not given"):
        design({"a": [1.0, 2.0]}, categorical=["b"])


def test_a_named_design_tests_exactly_what_the_indices_do():
    scores, table = _cohort()
    d = design(table)
    by_name = local_test(scores, d, terms=["site"])
    by_index = local_test(scores, d.matrix, terms=[1, 2], add_intercept=False)
    np.testing.assert_array_equal(by_name.statistic, by_index.statistic)
    np.testing.assert_array_equal(by_name.pvalue, by_index.pvalue)
    assert by_name.terms == (1, 2) and by_name.numerator_dof == 2
    assert by_name.names == d.names and isinstance(d, Design)
    with pytest.raises(ValueError, match="need a design built"):
        local_test(scores, d.matrix, terms=["site"], add_intercept=False)


# --- standard errors and intervals ------------------------------------------------


def test_standard_errors_and_intervals_match_statsmodels():
    scores, table = _cohort()
    d = design(table)
    result = local_test(scores, d, terms=["age"], method="none")
    low, high = result.interval(0.9, coefficients=True)
    for k in range(scores.shape[1]):
        theirs = sm.OLS(scores[:, k], d.matrix).fit()
        np.testing.assert_allclose(result.coefficients[k], theirs.params, rtol=1e-9)
        np.testing.assert_allclose(result.standard_errors[k], theirs.bse, rtol=1e-9)
        bounds = theirs.conf_int(alpha=0.1)
        np.testing.assert_allclose(low[k], bounds[:, 0], rtol=1e-9)
        np.testing.assert_allclose(high[k], bounds[:, 1], rtol=1e-9)
    # the tested estimate is the age coefficient, its interval the age row's
    age = d.columns("age")[0]
    np.testing.assert_allclose(result.estimate[:, 0], result.coefficients[:, age])
    np.testing.assert_allclose(result.interval(0.9)[0][:, 0], low[:, age])


def test_cluster_robust_errors_match_statsmodels():
    scores, table = _cohort(n=150, seed=1)
    families = np.arange(150) // 3
    d = design(table)
    result = local_test(scores, d, terms=["age"], groups=families)
    assert result.interval_dof == 49
    for k in range(scores.shape[1]):
        theirs = sm.OLS(scores[:, k], d.matrix).fit(
            cov_type="cluster", cov_kwds={"groups": families}
        )
        np.testing.assert_allclose(result.standard_errors[k], theirs.bse, rtol=1e-9)


def test_partial_r2_is_the_share_of_the_reduced_models_residual():
    scores, table = _cohort(seed=2)
    d = design(table)
    result = local_test(scores, d, terms=["site"])
    for k in range(scores.shape[1]):
        full = sm.OLS(scores[:, k], d.matrix).fit()
        reduced = sm.OLS(scores[:, k], np.delete(d.matrix, [1, 2], axis=1)).fit()
        expected = (reduced.ssr - full.ssr) / reduced.ssr
        assert result.partial_r2[k] == pytest.approx(expected, rel=1e-9)


# --- contrasts ----------------------------------------------------------------------


def test_a_contrast_between_two_levels_matches_statsmodels():
    scores, table = _cohort(seed=3)
    d = design(table)  # site[B] and site[C] against A
    result = local_test(scores, d, contrast={"site[B]": 1, "site[C]": -1}, method="none")
    weights = np.zeros(len(d.names))
    weights[[d.names.index("site[B]"), d.names.index("site[C]")]] = [1, -1]
    for k in range(scores.shape[1]):
        theirs = sm.OLS(scores[:, k], d.matrix).fit()
        test = theirs.t_test(weights)
        assert result.estimate[k, 0] == pytest.approx(float(test.effect[0]), rel=1e-9)
        assert result.estimate_errors[k, 0] == pytest.approx(float(test.sd[0, 0]), rel=1e-9)
        assert result.statistic[k] == pytest.approx(float(test.tvalue[0, 0]) ** 2, rel=1e-8)
        assert result.pvalue[k] == pytest.approx(float(test.pvalue), rel=1e-8)
    assert result.numerator_dof == 1 and result.terms == (1, 2)


def test_rows_of_the_identity_are_the_terms_and_two_rows_test_jointly():
    scores, table = _cohort(seed=4)
    d = design(table)
    terms = local_test(scores, d, terms=["site"])
    rows = np.eye(len(d.names))[[1, 2]]
    contrast = local_test(scores, d, contrast=rows)
    np.testing.assert_allclose(contrast.statistic, terms.statistic, rtol=1e-10)
    joint = np.array([[0, 1, -1, 0, 0], [0, 0, 0, 1, 0]], dtype=float)
    result = local_test(scores, d, contrast=joint, method="none")
    for k in range(scores.shape[1]):
        test = sm.OLS(scores[:, k], d.matrix).fit().f_test(joint)
        assert result.statistic[k] == pytest.approx(float(test.fvalue), rel=1e-8)
        assert result.pvalue[k] == pytest.approx(float(test.pvalue), rel=1e-8)
    assert result.numerator_dof == 2 and result.estimate.shape == (3, 2)


def test_a_contrast_with_families_matches_statsmodels_robust_wald():
    scores, table = _cohort(n=150, seed=5)
    families = np.arange(150) // 3
    d = design(table)
    weights = np.zeros(len(d.names))
    weights[[1, 2]] = [1, -1]
    result = local_test(scores, d, contrast=weights, groups=families, method="none")
    for k in range(scores.shape[1]):
        theirs = sm.OLS(scores[:, k], d.matrix).fit(
            cov_type="cluster", cov_kwds={"groups": families}
        )
        test = theirs.t_test(weights)
        assert result.estimate_errors[k, 0] == pytest.approx(float(test.sd[0, 0]), rel=1e-9)
        assert result.statistic[k] == pytest.approx(float(test.tvalue[0, 0]) ** 2, rel=1e-8)


def test_what_a_contrast_refuses():
    scores, table = _cohort(seed=6)
    d = design(table)
    with pytest.raises(ValueError, match="not both"):
        local_test(scores, d, terms=["age"], contrast={"age": 1})
    with pytest.raises(ValueError, match="stands for 2 columns"):
        local_test(scores, d, contrast={"site": 1})
    with pytest.raises(ValueError, match="row of zeros"):
        local_test(scores, d, contrast=np.zeros(len(d.names)))
    with pytest.raises(ValueError, match="not independent"):
        local_test(scores, d, contrast=[{"age": 1}, {"age": 2}])
    with pytest.raises(ValueError, match="weighs the design's 5 columns"):
        local_test(scores, d, contrast=[1.0, 2.0])
    full = np.column_stack(
        [d.matrix, (table["site"] == "A").astype(float)]
    )  # every level: collinear
    with pytest.raises(ValueError, match="not estimable"):
        local_test(scores, full, contrast=np.eye(6)[5], add_intercept=False)
    permuted = local_test(scores, d, contrast={"site[B]": 1}, permutations=49, seed=0)
    assert np.isfinite(permuted.pvalue).all()


# --- what the result says ------------------------------------------------------------


def test_a_single_contrasts_effect_map_weighs_its_estimate():
    from types import SimpleNamespace

    scores, table = _cohort(seed=7)
    d = design(table)
    result = local_test(scores, d, contrast={"site[B]": 1, "site[C]": -1})
    rng = np.random.default_rng(8)
    reduction = SimpleNamespace(
        basis=rng.standard_normal((40, 3)), scales=np.array([3.0, 2.0, 1.0])
    )
    weights = result.estimate[:, 0] * reduction.scales
    expected = (reduction.basis * weights) @ reduction.basis.sum(axis=0)
    np.testing.assert_allclose(result.effect_map(reduction), expected)
    by_name = result.effect_map(reduction, term="age")
    np.testing.assert_allclose(
        by_name,
        (reduction.basis * (result.coefficients[:, 3] * reduction.scales)) @ reduction.basis.sum(0),
    )


def test_the_table_carries_estimates_intervals_and_effect_sizes(tmp_path):
    import csv

    scores, table = _cohort(seed=9)
    d = design(table)
    result = local_test(scores, d, terms=["site"])
    path = result.to_table(tmp_path / "site.csv")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert header[:6] == ["component", "statistic", "pvalue", "adjusted", "partial_r2", "tested"]
    assert header[6:10] == ["estimate_site[B]", "se_site[B]", "ci_low_site[B]", "ci_high_site[B]"]
    assert header[-5:] == [f"coef_{name}" for name in d.names]
    low, _ = result.interval(0.95)
    assert float(rows[1][8]) == pytest.approx(low[0, 0], rel=1e-9)


def test_features_are_tested_column_by_column():
    """Per-vertex features: the empty columns are untestable and left out of the correction."""
    rng = np.random.default_rng(10)
    n, columns = 80, 50
    age = rng.uniform(22, 37, n)
    features = rng.standard_normal((n, columns))
    features[:, :5] += 0.3 * (age - 30)[:, None]
    features[:, 45:] = 0.0  # the medial wall: nothing there
    result = local_test(features, design({"age": age}), terms=["age"])
    assert np.isnan(result.pvalue[45:]).all() and np.isfinite(result.pvalue[:45]).all()
    assert set(range(5)) <= set(result.significant(0.05).tolist())
    np.testing.assert_allclose(result.estimate[:, 0], result.coefficients[:, 1])


# --- the seventh review -----------------------------------------------------------


def test_without_an_intercept_the_first_factor_is_coded_in_full():
    """Dropping a level as well as the intercept left the model without either: p 3e-38."""
    rng = np.random.default_rng(3)
    site = np.array(["A", "B", "C"])[rng.integers(0, 3, 90)]
    age = rng.normal(28.0, 3.5, 90)
    scores = rng.standard_normal((90, 2)) + 5.0  # far from zero, as scores without centring are
    bare = design({"site": site, "age": age}, intercept=False)
    assert bare.names == ("site[A]", "site[B]", "site[C]", "age")
    usual = design({"site": site, "age": age})
    np.testing.assert_allclose(
        local_test(scores, bare, terms=["age"]).pvalue,
        local_test(scores, usual, terms=["age"]).pvalue,
        rtol=1e-9,
    )


def test_text_is_a_factor_and_levels_go_in_their_natural_order():
    """Codes "01", "02", "03" were one numeric column; codes 2, 3, 10 took 10 as reference."""
    age = [21.0, 25.0, 30.0, 33.0, 27.0, 35.0]
    codes = design({"site": ["01", "02", "03", "01", "02", "03"], "age": age})
    assert codes.names == ("intercept", "site[02]", "site[03]", "age")
    for site in ([2, 3, 10, 2, 3, 10], ["2", "3", "10", "2", "3", "10"]):
        made = design({"site": site, "age": age}, categorical=["site"])
        assert made.names == ("intercept", "site[3]", "site[10]", "age")
    chosen = design(
        {"site": [2.0, 3.0, 10.0, 2.0, 3.0, 10.0], "age": age},
        categorical=["site"],
        reference={"site": 10},
    )
    assert chosen.names == ("intercept", "site[2]", "site[3]", "age")


def test_naming_a_factor_coded_in_full_without_an_intercept_is_refused():
    """terms=["site"] tested whether every level's mean is zero: the intercept came in with it."""
    covariates = {"site": ["a", "b", "c"] * 6, "age": np.linspace(20.0, 40.0, 18)}
    full = design(covariates, intercept=False)
    scores = np.random.default_rng(5).standard_normal((18, 2)) + 5.0
    with pytest.raises(ValueError, match="take in the intercept"):
        local_test(scores, full, terms=["site"])
    assert local_test(scores, full, terms=["age"]).terms == tuple(full.columns("age"))
    assert np.isfinite(local_test(scores, design(covariates), terms=["site"]).statistic).all()


def test_design_refuses_what_it_would_have_ignored_or_mangled():
    """A numeric reference was ignored; inf, colliding names, 2-D and empty input got through."""
    with pytest.raises(ValueError, match="categorical="):
        design({"site": [1, 2, 3, 1]}, reference={"site": 2})
    coded = design({"site": [1, 2, 3, 1]}, reference={"site": 2}, categorical=["site"])
    assert coded.names == ("intercept", "site[1]", "site[3]")
    with pytest.raises(ValueError, match="infinite in row 1"):
        design({"age": [20.0, np.inf, 30.0]})
    with pytest.raises(ValueError, match="one name once written as text"):
        design({1: [1.0, 2.0, 3.0], "1": [3.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="2-D"):
        design({"age": [[20.0, 21.0], [30.0, 31.0], [25.0, 26.0]]})
    with pytest.raises(ValueError, match="hold no subjects"):
        design({"age": []})


def test_a_value_the_loader_kept_is_a_level_when_design_is_told_so():
    """keep_default_missing=False kept a site coded NA, which design() then refused as missing."""
    covariates = {"site": ["NA", "EU", "NA", "EU"], "age": [20.0, 30.0, 25.0, 35.0]}
    with pytest.raises(ValueError, match=r"pass missing=\(\)"):
        design(covariates)
    assert design(covariates, missing=()).names == ("intercept", "site[NA]", "age")
