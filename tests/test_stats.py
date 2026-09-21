"""Local inference: against an independent implementation, and against theory.

There is no reference implementation to port, so the statistics are checked two
ways. The F statistic and its p-value are compared with ``scipy.stats``
computed independently, and the procedures are checked against the properties
that define them: p-values uniform under the null, Benjamini-Hochberg holding
the false discovery rate, and power rising with effect size.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.stats import LocalTest, benjamini_hochberg, local_test


def test_the_f_statistic_matches_an_independent_calculation():
    """Against scipy's own regression, term by term."""
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 40
    covariate = rng.standard_normal(n)
    scores = (2.0 * covariate + rng.standard_normal(n))[:, None]

    result = local_test(scores, covariate, method="none")

    regression = stats.linregress(covariate, scores[:, 0])
    # For a single term the F statistic is the square of the t statistic.
    expected_t = regression.slope / regression.stderr
    assert result.statistic[0] == pytest.approx(expected_t**2, rel=1e-9)
    assert result.pvalue[0] == pytest.approx(regression.pvalue, rel=1e-9)
    assert result.coefficients[0, 1] == pytest.approx(regression.slope, rel=1e-9)


def test_a_multi_term_test_matches_the_closed_form_f():
    """Two terms jointly, against the sums-of-squares definition."""
    from scipy.stats import f as f_distribution

    rng = np.random.default_rng(1)
    n = 50
    design = rng.standard_normal((n, 2))
    response = design @ np.array([1.5, -0.8]) + rng.standard_normal(n)

    result = local_test(response[:, None], design, method="none")

    full = np.column_stack([np.ones(n), design])
    beta, *_ = np.linalg.lstsq(full, response, rcond=None)
    full_ss = float(((response - full @ beta) ** 2).sum())
    reduced_ss = float(((response - response.mean()) ** 2).sum())
    expected = ((reduced_ss - full_ss) / 2) / (full_ss / (n - 3))

    assert result.statistic[0] == pytest.approx(expected, rel=1e-9)
    assert result.pvalue[0] == pytest.approx(f_distribution.sf(expected, 2, n - 3), rel=1e-9)
    assert result.numerator_dof == 2
    assert result.residual_dof == n - 3


def test_pvalues_are_uniform_under_the_null():
    """No association means p-values spread evenly over [0, 1]."""
    from scipy.stats import kstest

    rng = np.random.default_rng(2)
    n, trials = 30, 400
    pvalues = []
    for _ in range(trials):
        covariate = rng.standard_normal(n)
        scores = rng.standard_normal((n, 1))
        pvalues.append(local_test(scores, covariate, method="none").pvalue[0])

    assert kstest(pvalues, "uniform").pvalue > 0.01


def test_an_effect_is_found_and_power_rises_with_it():
    rng = np.random.default_rng(3)
    n = 60
    covariate = rng.standard_normal(n)
    found = []
    for size in (0.0, 0.5, 1.5):
        detections = 0
        for _ in range(60):
            scores = (size * covariate + rng.standard_normal(n))[:, None]
            detections += local_test(scores, covariate, method="none").pvalue[0] < 0.05
        found.append(detections / 60)
    assert found[0] < 0.2
    assert found[0] < found[1] < found[2]
    assert found[2] > 0.9


def test_benjamini_hochberg_matches_its_definition():
    pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    adjusted = benjamini_hochberg(pvalues)

    n = pvalues.size
    expected = np.minimum.accumulate((pvalues * n / np.arange(1, n + 1))[::-1])[::-1]
    np.testing.assert_allclose(adjusted, np.minimum(expected, 1.0))
    assert np.all(np.diff(adjusted) >= -1e-15), "adjusted p-values must be monotone"
    assert adjusted.max() <= 1.0


def test_benjamini_hochberg_controls_the_false_discovery_rate():
    """With everything null, discoveries should be rare at the stated rate."""
    rng = np.random.default_rng(4)
    n, n_components, trials = 30, 20, 200
    any_discovery = 0
    for _ in range(trials):
        covariate = rng.standard_normal(n)
        scores = rng.standard_normal((n, n_components))
        result = local_test(scores, covariate, method="fdr")
        any_discovery += result.significant(0.05).size > 0
    # Under a complete null the FDR equals the family-wise error rate.
    assert any_discovery / trials < 0.10


def test_bonferroni_is_never_less_conservative_than_fdr():
    rng = np.random.default_rng(5)
    n, n_components = 40, 12
    covariate = rng.standard_normal(n)
    scores = rng.standard_normal((n, n_components))
    scores[:, 0] += 1.2 * covariate

    fdr = local_test(scores, covariate, method="fdr")
    bonferroni = local_test(scores, covariate, method="bonferroni")
    assert np.all(bonferroni.adjusted >= fdr.adjusted - 1e-12)


def test_permutation_pvalues_track_the_parametric_ones():
    """For Gaussian data the two should broadly agree."""
    rng = np.random.default_rng(6)
    n, n_components = 40, 6
    covariate = rng.standard_normal(n)
    scores = rng.standard_normal((n, n_components))
    scores[:, 0] += 1.5 * covariate

    parametric = local_test(scores, covariate, method="none")
    permuted = local_test(scores, covariate, method="none", permutations=500, seed=0)

    assert permuted.permutations == 500
    assert np.all(permuted.pvalue >= 1 / 501)
    assert np.corrcoef(parametric.pvalue, permuted.pvalue)[0, 1] > 0.9
    assert permuted.pvalue[0] < 0.05


def test_the_effect_map_lives_on_the_surface():
    """A fitted effect pushed back through the basis gives one value per vertex."""
    from sbci.reduction import fit_basis

    rng = np.random.default_rng(7)
    n_vertices, n_subjects = 20, 24
    truth = np.linalg.qr(rng.standard_normal((n_vertices, n_vertices)))[0][:, :3]
    covariate = rng.standard_normal(n_subjects)
    weights = rng.standard_normal((n_subjects, 3))
    weights[:, 0] += 3.0 * covariate

    matrices = np.stack(
        [
            sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(3))
            for i in range(n_subjects)
        ]
    )
    matrices -= matrices.mean(axis=0, keepdims=True)
    reduction = fit_basis(matrices, np.eye(n_vertices), rank=3, seed=0)

    result = local_test(reduction.scores, covariate)
    effect = result.effect_map(reduction)
    assert effect.shape == (n_vertices,)
    assert np.isfinite(effect).all()

    restricted = result.effect_map(reduction, alpha=0.05)
    assert restricted.shape == (n_vertices,)
    if result.significant(0.05).size == 0:
        np.testing.assert_allclose(restricted, 0.0, atol=1e-12)


def test_local_test_validates_its_arguments():
    rng = np.random.default_rng(8)
    scores = rng.standard_normal((20, 4))
    covariate = rng.standard_normal(20)

    with pytest.raises(ValueError, match="method must be one of"):
        local_test(scores, covariate, method="holm")
    with pytest.raises(ValueError, match="subjects in scores"):
        local_test(scores, rng.standard_normal(19))
    with pytest.raises(ValueError, match="cannot fit"):
        local_test(rng.standard_normal((3, 2)), rng.standard_normal((3, 5)))
    with pytest.raises(ValueError, match="intercept alone"):
        local_test(scores, np.ones((20, 1)), add_intercept=False)


def test_the_result_reports_what_it_found():
    rng = np.random.default_rng(9)
    n = 50
    covariate = rng.standard_normal(n)
    scores = rng.standard_normal((n, 5))
    scores[:, 2] += 2.0 * covariate

    result = local_test(scores, covariate)
    assert isinstance(result, LocalTest)
    assert 2 in result.significant(0.05)
    assert result.method == "fdr"
    assert result.coefficients.shape == (5, 2)
