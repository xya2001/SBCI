"""Inference on connectome scores.

After :func:`sbci.reduce` turns each subject into a short score vector, this
asks which components carry an association with a covariate, and where on the
cortex those components live.

**There is no reference implementation for this.** Every other port in this
package is checked against a MATLAB run; ``SBCI_Modeling_FPCA`` contains no
inference code, and neither does the toolkit. What is implemented here is
ordinary linear-model inference, and it is verified two ways: against
``scipy.stats`` as an independent implementation of the same statistics, and
against the properties the procedures are defined by -- uniform p-values under
the null, and false-discovery control at the stated rate. That is a weaker
standard than the rest of the package and is recorded as such in PORTING.md
item 5.

What "local" means here
-----------------------
Two different things, and both are provided:

* **Component-wise.** Each component is tested for association with the
  design, and the resulting p-values are corrected for testing ``K`` of them.
* **On the surface.** A component is a function on the cortex, so a fitted
  effect can be pushed back onto it: the effect on the connectivity between
  vertices ``i`` and ``j`` is ``sum_k beta_k psi_k(i) psi_k(j)``. Summing over
  ``j`` gives a per-vertex map of where the association sits, which is what
  :meth:`LocalTest.effect_map` returns.

The surface map is a description of the fitted effect, not a test at each
vertex: the p-values belong to the components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("fdr", "bonferroni", "none")
"""Multiplicity corrections. ``fdr`` is Benjamini-Hochberg."""


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values, monotone and clipped to one."""
    pvalues = np.asarray(pvalues, dtype=np.float64).ravel()
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _design_matrix(design, add_intercept):
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if add_intercept and not np.allclose(design[:, 0], 1.0):
        design = np.column_stack([np.ones(design.shape[0]), design])
    return design


@dataclass
class LocalTest:
    """What :func:`local_test` returns."""

    statistic: np.ndarray
    """``(n_components,)`` F statistic for the tested terms."""
    pvalue: np.ndarray
    """``(n_components,)`` unadjusted p-values."""
    adjusted: np.ndarray
    """``(n_components,)`` p-values after the multiplicity correction."""
    coefficients: np.ndarray
    """``(n_components, n_terms)`` fitted regression coefficients."""
    residual_dof: int
    """Residual degrees of freedom of each component's model."""
    numerator_dof: int
    """Degrees of freedom of the tested terms."""
    method: str
    """The correction that produced :attr:`adjusted`."""
    permutations: int = 0
    """How many permutations produced the p-values, or 0 if parametric."""

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        """Indices of components whose adjusted p-value is at or below ``alpha``."""
        return np.flatnonzero(self.adjusted <= alpha)

    def effect_map(self, reduction, term: int = 0, alpha: float | None = None):
        """Where on the surface the fitted effect sits.

        Sums the fitted change in connectivity over one endpoint, giving one
        value per vertex. With ``alpha`` set, only components surviving the
        correction contribute.
        """
        basis = np.asarray(reduction.basis, dtype=np.float64)
        weights = self.coefficients[:, term] * np.asarray(reduction.scales)
        if alpha is not None:
            keep = np.zeros(weights.size, dtype=bool)
            keep[self.significant(alpha)] = True
            weights = np.where(keep, weights, 0.0)
        return (basis * weights) @ (basis.sum(axis=0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<LocalTest {self.statistic.size} components, "
            f"{self.significant().size} significant at 0.05 ({self.method})>"
        )


def local_test(
    scores,
    design,
    terms=None,
    method: str = "fdr",
    add_intercept: bool = True,
    permutations: int = 0,
    seed=None,
) -> LocalTest:
    """Test each component's scores for association with the design.

    Fits ``scores[:, k] ~ design`` by least squares for every component and
    tests the ``terms`` columns jointly with an F test, then corrects across
    components.

    Parameters
    ----------
    scores
        ``(n_subjects, n_components)``, from :attr:`sbci.reduction.Reduction.scores`.
    design
        ``(n_subjects, n_covariates)``. An intercept column is prepended unless
        one is already there or ``add_intercept`` is false.
    terms
        Which design columns to test, as indices into the final design matrix.
        Defaults to every column except the intercept.
    method
        Multiplicity correction, one of :data:`METHODS`.
    permutations
        If positive, p-values come from permuting the rows of the tested terms
        this many times rather than from the F distribution. Use this when the
        scores are not plausibly Gaussian; the correction is still applied to
        the permutation p-values.
    seed
        Seed for the permutations.

    Examples
    --------
    >>> result = sbci.stats.local_test(reduction.scores, age)   # doctest: +SKIP
    >>> result.significant(0.05)                                 # doctest: +SKIP
    array([0, 3])
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    matrix = _design_matrix(design, add_intercept)
    n_subjects, n_terms = matrix.shape
    if scores.shape[0] != n_subjects:
        raise ValueError(f"{scores.shape[0]} subjects in scores but {n_subjects} in the design")

    if terms is None:
        intercept = [i for i in range(n_terms) if np.allclose(matrix[:, i], matrix[0, i])]
        tested = [i for i in range(n_terms) if i not in intercept]
    else:
        tested = list(np.atleast_1d(terms))
    if not tested:
        raise ValueError("no terms to test; the design is an intercept alone")

    residual_dof = n_subjects - n_terms
    if residual_dof <= 0:
        raise ValueError(
            f"{n_subjects} subjects cannot fit {n_terms} terms; reduce the design or add subjects"
        )

    def fit(response, full):
        """Coefficients and residual sum of squares for one design."""
        coefficients, *_ = np.linalg.lstsq(full, response, rcond=None)
        residual = response - full @ coefficients
        return coefficients, float(residual @ residual)

    reduced_columns = [i for i in range(n_terms) if i not in tested]
    reduced = matrix[:, reduced_columns] if reduced_columns else np.zeros((n_subjects, 0))

    n_components = scores.shape[1]
    statistic = np.empty(n_components)
    coefficients = np.empty((n_components, n_terms))
    for k in range(n_components):
        full_beta, full_ss = fit(scores[:, k], matrix)
        coefficients[k] = full_beta
        if reduced.shape[1]:
            _, reduced_ss = fit(scores[:, k], reduced)
        else:
            reduced_ss = float(scores[:, k] @ scores[:, k])
        numerator = (reduced_ss - full_ss) / len(tested)
        denominator = full_ss / residual_dof
        statistic[k] = numerator / denominator if denominator > 0 else np.inf

    if permutations > 0:
        rng = np.random.default_rng(seed)
        exceed = np.zeros(n_components)
        for _ in range(permutations):
            shuffled = matrix.copy()
            shuffled[:, tested] = shuffled[rng.permutation(n_subjects)][:, tested]
            for k in range(n_components):
                _, full_ss = fit(scores[:, k], shuffled)
                if reduced.shape[1]:
                    _, reduced_ss = fit(scores[:, k], reduced)
                else:
                    reduced_ss = float(scores[:, k] @ scores[:, k])
                numerator = (reduced_ss - full_ss) / len(tested)
                denominator = full_ss / residual_dof
                value = numerator / denominator if denominator > 0 else np.inf
                exceed[k] += value >= statistic[k]
        pvalue = (exceed + 1) / (permutations + 1)
    else:
        from scipy.stats import f as f_distribution

        pvalue = f_distribution.sf(statistic, len(tested), residual_dof)

    if method == "fdr":
        adjusted = benjamini_hochberg(pvalue)
    elif method == "bonferroni":
        adjusted = np.minimum(pvalue * n_components, 1.0)
    else:
        adjusted = pvalue.copy()

    return LocalTest(
        statistic=statistic,
        pvalue=np.asarray(pvalue, dtype=np.float64),
        adjusted=adjusted,
        coefficients=coefficients,
        residual_dof=residual_dof,
        numerator_dof=len(tested),
        method=method,
        permutations=permutations,
    )
