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

Related subjects
----------------
The tests assume independent subjects. A cohort of families is not: the HCP
Young Adult subjects come in families of up to six, twins and siblings, and
treating them as independent makes every p-value too small. ``groups=`` (the
family of each subject) replaces the classical standard errors by
cluster-robust ones, which allow any dependence inside a family, and takes the
degrees of freedom from the number of families, not of subjects. Like any
cluster-robust test it needs many families: in simulated cohorts with no
association it rejects at 0.05 9.7% of the time with 20 families, 6.0% with
100 and 5.1% with 422, where the test that ignores the families rejects 16%
(``tools/clustered_null.py``). Permutation is not offered with groups:
shuffling members between families of different make-up (twins against
siblings) is not exchangeable, and doing it properly needs each family's
structure, not just its label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("fdr", "bonferroni", "none")
"""Multiplicity corrections. ``fdr`` is Benjamini-Hochberg."""


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values, monotone and clipped to one.

    A ``NaN`` (a component that could not be tested) stays ``NaN`` and does
    not count towards the number of tests.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64).ravel()
    adjusted = np.full(pvalues.size, np.nan)
    finite = np.isfinite(pvalues)
    values = pvalues[finite]
    n = values.size
    if n == 0:
        return adjusted
    order = np.argsort(values)
    ranked = values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.minimum(ranked, 1.0)
    adjusted[finite] = corrected
    return adjusted


def _constant_columns(matrix) -> np.ndarray:
    """Which columns are a nonzero constant, and so serve as an intercept.

    Constancy is exact equality: a tolerance would take a covariate of tiny
    scale for a constant, and a column of zeros is no intercept at all.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=bool)
    return np.all(matrix == matrix[:1], axis=0) & (matrix[0] != 0)


def _spans_constant(matrix, rank: int | None = None) -> bool:
    """Whether the columns can reproduce a constant, and so absorb a shift of the response.

    A constant column can; so can columns that only combine into one, as dummy
    codes for every level of a factor sum to one. Judged by the numerical rank
    the degrees of freedom are counted with: a constant adds nothing to it.
    ``rank`` is the matrix's own rank, when already known.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        return False
    if rank is None:
        rank = int(np.linalg.matrix_rank(matrix))
    with_constant = np.column_stack([matrix, np.ones(matrix.shape[0])])
    return int(np.linalg.matrix_rank(with_constant)) == rank


def _missing_labels(labels: np.ndarray) -> np.ndarray:
    """Which group labels are missing: ``NaN``, ``None``, or an empty or blank string.

    Coded like any other label, the missing ones would make one family of
    every subject without one: ``np.unique`` takes all NaNs for one value.
    """
    if labels.dtype.kind in "fc":
        return np.isnan(labels)
    if labels.dtype.kind in "US":
        return np.char.str_len(np.char.strip(labels)) == 0
    if labels.dtype.kind == "O":
        return np.array([_is_missing(label) for label in labels], dtype=bool)
    return np.zeros(labels.size, dtype=bool)


def _is_missing(label) -> bool:
    if label is None:
        return True
    if isinstance(label, (str, bytes)):
        return not label.strip()
    try:
        return bool(label != label)  # NaN, and NaT, are unequal to themselves
    except TypeError:  # pandas' NA will not say whether it equals itself
        return True


def _design_matrix(design, add_intercept):
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if not add_intercept:
        return design
    # Always column 0, so that ``terms`` keep meaning the same columns whatever
    # the covariates hold. A constant column beside it is either the caller's
    # own intercept or a covariate that does not vary in these subjects (one
    # sex), and taking it silently for either misnumbers the columns of whoever
    # meant the other: it is refused, and with it a column of zeros, the other
    # coding of one sex.
    if design.shape[0]:
        zeros = ~design.any(axis=0)
        constant = np.flatnonzero(_constant_columns(design) | zeros)
        if constant.size:
            j = int(constant[0])
            if zeros[j]:
                raise ValueError(
                    f"design column {j} is all zeros, which no test can use: if it is a "
                    "covariate that does not vary in these subjects (one sex, say), drop it"
                )
            raise ValueError(
                f"design column {j} is constant, so it duplicates the intercept prepended as "
                "column 0: if it is your own intercept, pass add_intercept=False; if it is a "
                "covariate that does not vary in these subjects (one sex, say), drop it"
            )
    return np.column_stack([np.ones(design.shape[0]), design])


@dataclass
class LocalTest:
    """What :func:`local_test` returns."""

    statistic: np.ndarray
    """``(n_components,)`` F statistic for the tested terms; ``NaN`` where a
    component has no residual variance or non-finite scores and so cannot be
    tested."""
    pvalue: np.ndarray
    """``(n_components,)`` unadjusted p-values, ``NaN`` where the statistic is."""
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
    terms: tuple = ()
    """Columns of the design matrix that were tested."""
    groups: int = 0
    """How many groups (families) the subjects fall in when ``groups=`` was
    given, and the p-values come from cluster-robust standard errors with
    ``groups - 1`` denominator degrees of freedom; 0 for independent subjects."""

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        """Indices of components whose adjusted p-value is at or below ``alpha``."""
        return np.flatnonzero(self.adjusted <= alpha)

    def effect_map(self, reduction, term: int | None = None, alpha: float | None = None):
        """Where on the surface the fitted effect sits.

        Sums the fitted change in connectivity over one endpoint, giving one
        value per vertex. ``term`` is a column of the design matrix and
        defaults to the first *tested* one -- not column 0, which is the
        intercept when one was added. With ``alpha`` set, only components
        surviving the correction contribute.
        """
        if term is None:
            if not self.terms:
                raise ValueError("no tested terms recorded; pass term= explicitly")
            term = int(self.terms[0])
        basis = np.asarray(reduction.basis, dtype=np.float64)
        weights = self.coefficients[:, term] * np.asarray(reduction.scales)
        if alpha is not None:
            keep = np.zeros(weights.size, dtype=bool)
            keep[self.significant(alpha)] = True
            weights = np.where(keep, weights, 0.0)
        return (basis * weights) @ (basis.sum(axis=0))

    def to_table(self, path, names=None):
        """Write the results to a ``.csv`` or ``.tsv`` table, one row per component.

        The columns are ``component`` (counted from 0, as :meth:`significant`
        and the reduction's basis count them), ``statistic``, ``pvalue`` and
        ``adjusted``, then the fitted coefficient of every design column,
        ``coef_<name>``. ``names`` names the columns of the design matrix as
        tested, the intercept included when one was added; without it they
        are numbered. An untestable component's statistic and p-values are
        empty cells.

        Examples
        --------
        >>> columns = ["intercept", "score", "female", "age", "count"]
        >>> sex.to_table("sex.csv", names=columns)  # doctest: +SKIP
        """
        from .export import _write_table

        coefficients = np.asarray(self.coefficients)
        if names is None:
            labels = [str(j) for j in range(coefficients.shape[1])]
        else:
            labels = [str(name) for name in names]
            if len(labels) != coefficients.shape[1]:
                raise ValueError(
                    f"{len(labels)} names for the design's {coefficients.shape[1]} columns "
                    "(column 0 is the intercept when one was added)"
                )
        header = ["component", "statistic", "pvalue", "adjusted"]
        header += [f"coef_{label}" for label in labels]
        rows = (
            [k, self.statistic[k], self.pvalue[k], self.adjusted[k], *coefficients[k]]
            for k in range(self.statistic.size)
        )
        return _write_table(path, header, rows)

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
    groups=None,
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
        ``(n_subjects, n_covariates)``. Unless ``add_intercept`` is false an
        intercept is prepended as column 0, always, so covariate ``j`` is
        column ``j + 1`` whatever the covariates hold, and a constant column
        is refused: beside the prepended intercept it is either an intercept
        of your own, as statsmodels' ``add_constant`` makes one, or a
        covariate that does not vary in these subjects (sex in a single-sex
        subset), and taking it for either would misnumber the columns of
        whoever meant the other. A column of zeros, which no test can use, is
        refused with it.
    terms
        Which design columns to test, as integer indices into the final design
        matrix, each at most once. Defaults to every non-constant column except
        the intercept. A design with no constant column whose columns still
        combine into one -- dummy codes for every level of a factor, with
        ``add_intercept=False`` -- has no intercept column to leave out, and
        needs ``terms`` named. A boolean array is refused: it would be read as
        the indices 0 and 1.
    method
        Multiplicity correction, one of :data:`METHODS`.
    add_intercept
        Prepend the intercept as column 0. Pass ``False`` to supply it
        yourself, as a constant column of ``design``; the columns are then
        numbered as given. A design carrying its own constant column needs
        ``add_intercept=False``: until October 2026 such a column was taken
        as the intercept. Dummy codes for every level of a factor carry an
        intercept too, implicitly, and are tested exactly as the same factor
        coded against a reference level with the intercept prepended.
    permutations
        If positive, p-values come from a permutation test with this many
        draws rather than from the F distribution. The scheme is
        Freedman-Lane: the residuals of the reduced (nuisance-only) model are
        permuted and its fit added back, which keeps the null exact when the
        tested covariate is correlated with the nuisance ones. Use this when
        the scores are not plausibly Gaussian; the correction is still applied
        to the permutation p-values.
    seed
        Seed for the permutations.
    groups
        ``(n_subjects,)`` labels, one per subject, for subjects that are not
        independent: the family of each, in a cohort of twins and siblings.
        The tested terms are then judged by a Wald test with cluster-robust
        standard errors (the usual small-sample correction,
        ``G / (G - 1) * (n - 1) / (n - p)`` for ``G`` groups and ``p``
        columns), and the F statistic's denominator has ``G - 1`` degrees of
        freedom instead of ``n - p``. It needs a design of full column rank
        and hundreds of groups rather than dozens (the module notes), and
        cannot be combined with ``permutations``. Every subject needs a
        label: a missing one -- ``NaN``, ``None``, or an empty or blank
        string -- is refused, since coded like any other label it would put
        every subject without a family into one family together. A subject
        without a family takes a label of its own.

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
        intercept = [i for i in range(n_terms) if np.all(matrix[:, i] == matrix[0, i])]
        tested = [i for i in range(n_terms) if i not in intercept]
        if not tested:
            raise ValueError("no terms to test; the design is an intercept alone")
        if not intercept and _spans_constant(matrix):
            raise ValueError(
                "the design has no constant column, but its columns combine into one (dummy "
                "codes for every level of a factor do), so the default of testing every column "
                "would test the intercept with them: name the columns to test with terms=, or "
                "code the factor against a reference level and let the intercept be prepended"
            )
    else:
        requested = np.atleast_1d(np.asarray(terms))
        if requested.size == 0:
            raise ValueError("terms= is empty; name at least one design column")
        if requested.dtype.kind not in "iu":
            raise ValueError(
                f"terms must be integer column indices, got {requested.dtype}; a boolean "
                "array would be read as the indices 0 and 1"
            )
        tested = [int(t) for t in requested]
        if len(set(tested)) != len(tested):
            raise ValueError(f"terms {tested} name a column more than once")
        outside = [t for t in tested if not 0 <= t < n_terms]
        if outside:
            raise ValueError(
                f"terms {outside} are outside the design's {n_terms} columns (column 0 is "
                "the intercept when one is added)"
            )

    codes = None
    if groups is not None:
        labels = np.asarray(groups).ravel()
        if labels.size != n_subjects:
            raise ValueError(f"{labels.size} group labels for {n_subjects} subjects")
        missing = int(_missing_labels(labels).sum())
        if missing:
            raise ValueError(
                f"groups= has {missing} missing label{'s' if missing > 1 else ''}; give every "
                "subject a family, or a label of its own for a subject without one"
            )
        _, codes = np.unique(labels, return_inverse=True)
        if codes.max() + 1 < 2:
            raise ValueError("groups= needs at least two groups")
        if permutations > 0:
            raise ValueError(
                "permutations= does not respect groups=: shuffling subjects between families "
                "of different make-up is not exchangeable; use the parametric test with groups="
            )

    reduced_columns = [i for i in range(n_terms) if i not in tested]
    reduced = matrix[:, reduced_columns] if reduced_columns else np.zeros((n_subjects, 0))

    # Degrees of freedom follow the rank, not the column count: a collinear
    # design (dummy codes for every level plus an intercept) still fits by
    # least squares, but has fewer independent terms than columns.
    rank_full = int(np.linalg.matrix_rank(matrix))
    rank_reduced = int(np.linalg.matrix_rank(reduced)) if reduced.shape[1] else 0
    residual_dof = n_subjects - rank_full
    numerator_dof = rank_full - rank_reduced
    if numerator_dof == 0:
        raise ValueError("the tested terms are collinear with the rest of the design")
    if residual_dof <= 0:
        raise ValueError(
            f"{n_subjects} subjects cannot fit {rank_full} independent terms; "
            "reduce the design or add subjects"
        )

    def fit(responses, full):
        """Coefficients and residual sums of squares, every component at once."""
        coefficients, *_ = np.linalg.lstsq(full, responses, rcond=None)
        residual = responses - full @ coefficients
        return coefficients, (residual * residual).sum(axis=0)

    # The scale a residual is judged against is the variation the reduced
    # model has to explain. When its columns can reproduce a constant -- an
    # intercept column, or dummy codes for every level of a factor, which sum
    # to one -- the mean is absorbed, so that is the centred sum of squares:
    # scores sitting at 1e6 with unit spread are as testable as scores at
    # zero. Otherwise it is the plain sum of squares. (Until 6 October 2026
    # only an intercept column counted, and a shift of 1e6 turned every
    # component of a dummy-coded design NaN.)
    magnitude = (scores * scores).sum(axis=0)
    if reduced.shape[1] and _spans_constant(reduced, rank_reduced):
        centred = scores - scores.mean(axis=0)
        scale = (centred * centred).sum(axis=0)
    else:
        scale = magnitude
    # Scores agreeing to ten significant digits have no spread at all: what
    # centring leaves is the rounding of the mean, not variance.
    no_spread = scale <= 1e-20 * magnitude

    def f_statistic(reduced_ss, full_ss):
        with np.errstate(divide="ignore", invalid="ignore"):
            value = ((reduced_ss - full_ss) / numerator_dof) / (full_ss / residual_dof)
        # Nothing to test where the reduced model already leaves no residual
        # variance -- constant or non-finite scores -- judged against the
        # scores' own scale, since least squares leaves rounding rather than an
        # exact zero. A perfect fit of a varying response is real, and stays.
        untestable = (
            ~np.isfinite(reduced_ss)
            | ~np.isfinite(full_ss)
            | no_spread
            | (reduced_ss <= 1e-10 * scale)
        )
        return np.where(untestable, np.nan, value)

    n_components = scores.shape[1]
    full_beta, full_ss = fit(scores, matrix)
    coefficients = full_beta.T
    if reduced.shape[1]:
        reduced_beta, reduced_ss = fit(scores, reduced)
        fitted = reduced @ reduced_beta
    else:
        reduced_ss = (scores * scores).sum(axis=0)
        fitted = np.zeros_like(scores)
    statistic = f_statistic(reduced_ss, full_ss)
    n_groups = 0
    denominator_dof = residual_dof
    if codes is not None:
        n_groups = int(codes.max()) + 1
        if rank_full < n_terms:
            raise ValueError(
                "groups= needs a design of full column rank; a constant covariate duplicates "
                "the intercept, and dummy codes for every level of a factor do too"
            )
        statistic = _clustered_wald(
            matrix, scores, full_beta, codes, n_groups, tested, np.isfinite(statistic)
        )
        denominator_dof = n_groups - 1

    if permutations > 0:
        # Freedman-Lane: permute the residuals of the reduced model, add its fit
        # back, and refit both models to the surrogate scores -- for every
        # component in one solve per permutation.
        rng = np.random.default_rng(seed)
        residuals = scores - fitted
        exceed = np.zeros(n_components)
        for _ in range(permutations):
            surrogate = fitted + residuals[rng.permutation(n_subjects)]
            _, permuted_full = fit(surrogate, matrix)
            if reduced.shape[1]:
                _, permuted_reduced = fit(surrogate, reduced)
            else:
                permuted_reduced = (surrogate * surrogate).sum(axis=0)
            permuted = f_statistic(permuted_reduced, permuted_full)
            with np.errstate(invalid="ignore"):
                exceed += permuted >= statistic
        pvalue = (exceed + 1) / (permutations + 1)
        pvalue = np.where(np.isfinite(statistic), pvalue, np.nan)
    else:
        from scipy.stats import f as f_distribution

        pvalue = f_distribution.sf(statistic, numerator_dof, denominator_dof)

    if method == "fdr":
        adjusted = benjamini_hochberg(pvalue)
    elif method == "bonferroni":
        # Over the components that could be tested, as benjamini_hochberg counts them.
        adjusted = np.minimum(pvalue * int(np.isfinite(pvalue).sum()), 1.0)
    else:
        adjusted = pvalue.copy()

    return LocalTest(
        statistic=statistic,
        pvalue=np.asarray(pvalue, dtype=np.float64),
        adjusted=adjusted,
        coefficients=coefficients,
        residual_dof=residual_dof,
        numerator_dof=numerator_dof,
        method=method,
        permutations=permutations,
        terms=tuple(int(t) for t in tested),
        groups=n_groups,
    )


def _clustered_wald(matrix, scores, beta, codes, n_groups, tested, testable):
    """Wald F statistics for the ``tested`` columns, with cluster-robust standard errors.

    For each component, the coefficients' covariance is the sandwich
    ``c * B @ M @ B`` with ``B = (X'X)^-1`` and ``M`` the sum over groups of
    ``(X_g' e_g)(X_g' e_g)'``, where ``e_g`` is the group's residuals and
    ``c = G / (G - 1) * (n - 1) / (n - p)``; the statistic is
    ``b' V^-1 b / q`` over the ``q`` tested coefficients ``b``.
    """
    n_subjects, n_terms = matrix.shape
    residual = scores - matrix @ beta
    # Each group's score X_g' e_g, for every component: (groups, terms, components).
    contributions = np.zeros((n_groups, n_terms, scores.shape[1]))
    np.add.at(contributions, codes, matrix[:, :, None] * residual[:, None, :])
    meat = np.einsum("gik,gjk->kij", contributions, contributions)
    bread = np.linalg.inv(matrix.T @ matrix)
    correction = n_groups / (n_groups - 1) * (n_subjects - 1) / (n_subjects - n_terms)
    covariance = correction * np.einsum("ij,kjl,lm->kim", bread, meat, bread)
    index = np.asarray(tested)
    tested_beta = beta[index].T  # (components, q)
    block = covariance[:, index][:, :, index]  # (components, q, q)
    statistic = np.full(scores.shape[1], np.nan)
    for k in np.flatnonzero(testable):
        try:
            solved = np.linalg.solve(block[k], tested_beta[k])
        except np.linalg.LinAlgError:
            continue
        statistic[k] = float(tested_beta[k] @ solved) / index.size
    return statistic
