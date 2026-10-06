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
vertex: the p-values belong to the components. Tested feature by feature
instead -- ``scores`` holding any per-subject features, each vertex's
strength or a coupling map, rather than components -- the p-values do belong
to each vertex, and the coefficients are the effect at each: that is the
per-vertex question, where the components ask whether the effect lies in the
connectome's leading directions.

Hypotheses, estimates and intervals
-----------------------------------
A test is of a linear hypothesis about the coefficients: that some columns
of the design are zero together (``terms``), or that a contrast of them is
(``contrast``). Either way the result carries the tested estimates with their
standard errors and confidence intervals -- classical, or cluster-robust with
``groups=`` -- and the partial R-squared, the share of the reduced model's
residual that the tested part explains. :func:`design` builds a design with
named columns from a table, coding factors against a reference level, so
that hypotheses can be written by name; ``terms=["site"]`` then tests a
factor of any number of levels at once.

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

import difflib
from collections.abc import Mapping
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
    # meant the other: it is refused. A column of zeros can be neither, and is
    # kept: it fits as nothing and counts for nothing in the degrees of
    # freedom, which is what a dummy for a level absent from a stratum should do.
    if design.shape[0]:
        constant = np.flatnonzero(_constant_columns(design))
        if constant.size:
            j = int(constant[0])
            raise ValueError(
                f"column {j} of the design (column {j + 1} of the design matrix, as terms= "
                "counts with the intercept as column 0) is constant, so it duplicates the "
                "intercept: if it is your own intercept, pass add_intercept=False; if it is a "
                "covariate that does not vary in these subjects (one sex, say), drop it"
            )
    return np.column_stack([np.ones(design.shape[0]), design])


@dataclass(frozen=True)
class Design:
    """A design matrix whose columns have names: what :func:`design` builds.

    :func:`local_test` takes one in place of an array, and then takes names
    where it takes column indices: ``terms=["sex"]`` tests every column a
    covariate brought in -- all the dummies of a factor at once -- and a
    contrast can be written ``{"site[B]": 1, "site[C]": -1}``.
    """

    matrix: np.ndarray
    """``(n_subjects, n_columns)``, the intercept as column 0 when there is one."""
    names: tuple
    """A name for every column: ``intercept``, a covariate's name, or
    ``factor[level]`` for the dummy of a factor's level."""
    covariates: dict
    """``{covariate: (column, ...)}``, the columns each covariate brought in."""

    def columns(self, term) -> list:
        """The columns a name stands for: a covariate's, or one column's; an index stays."""
        if isinstance(term, (int, np.integer)) and not isinstance(term, (bool, np.bool_)):
            return [int(term)]
        name = str(term)
        if name in self.covariates:
            return list(self.covariates[name])
        if name in self.names:
            return [self.names.index(name)]
        close = difflib.get_close_matches(name, [*self.covariates, *self.names], n=3)
        hint = f"; did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"the design has no covariate or column {name!r}{hint}")


def design(covariates, intercept: bool = True, reference=None, categorical=()) -> Design:
    """A design matrix with named columns, from named covariates.

    Parameters
    ----------
    covariates
        ``{name: values}``, one value per subject in each: a table's columns,
        or :attr:`sbci.LoadedCohort.covariates` as it is.
    intercept
        Put a column of ones first, named ``intercept``.
    reference
        ``{factor: level}``, the level a factor's dummies are coded against;
        by default its first in sorted order.
    categorical
        Covariates to take as factors although their values are numbers, as a
        site coded 1, 2, 3. Text is a factor anyway.

    A number enters as it is. A factor of ``L`` levels enters as ``L - 1``
    columns of zeros and ones, one per level but the reference, named
    ``factor[level]``; beside the intercept each one's coefficient is that
    level's difference from the reference. A missing value is refused --
    leave those subjects out first, as :func:`sbci.load_cohort` does with
    ``require=`` -- and so is a covariate that does not vary.

    Examples
    --------
    >>> d = design({"age": [22.0, 30.0, 35.0, 28.0], "sex": ["F", "M", "M", "F"]})
    >>> d.names
    ('intercept', 'age', 'sex[M]')
    >>> d.matrix[:, 2]
    array([0., 1., 1., 0.])
    >>> d.columns("sex")
    [2]
    """
    from .cohort import _missing

    if not isinstance(covariates, Mapping) or not covariates:
        raise ValueError("design() takes a dict of named covariates, one value per subject")
    reference = {str(k): v for k, v in (reference or {}).items()}
    categorical = {categorical} if isinstance(categorical, str) else {str(c) for c in categorical}
    unknown = sorted((set(reference) | categorical) - {str(k) for k in covariates})
    if unknown:
        raise ValueError(f"reference= and categorical= name covariates not given: {unknown}")
    cells_of = {str(k): list(np.asarray(v, dtype=object).ravel()) for k, v in covariates.items()}
    lengths = {name: len(cells) for name, cells in cells_of.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"the covariates differ in length: {lengths}")
    n = next(iter(lengths.values()))
    columns: list = []
    names: list = []
    groups: dict = {}
    if intercept:
        columns.append(np.ones(n))
        names.append("intercept")
        groups["intercept"] = (0,)
    for name, cells in cells_of.items():
        missing = [i for i, value in enumerate(cells) if _missing(value)]
        if missing:
            count = len(missing)
            raise ValueError(
                f"{name} has {count} missing value{'s' if count > 1 else ''} (the first in row "
                f"{missing[0]}): leave those subjects out first, as load_cohort(require=...) does"
            )
        numbers = None
        if name not in categorical:
            try:
                numbers = np.array([float(value) for value in cells])
            except (TypeError, ValueError):
                numbers = None
        if numbers is not None:
            if np.all(numbers == numbers[0]):
                raise ValueError(f"{name} does not vary in these subjects; drop it")
            groups[name] = (len(columns),)
            columns.append(numbers)
            names.append(name)
            continue
        labels = np.array([str(value).strip() for value in cells])
        levels = sorted(set(labels.tolist()))
        if len(levels) < 2:
            raise ValueError(f"{name} has one level in these subjects, {levels[0]!r}; drop it")
        base = str(reference.get(name, levels[0]))
        if base not in levels:
            raise ValueError(f"{name} has no level {base!r}; its levels are {levels}")
        indices = []
        for level in levels:
            if level == base:
                continue
            indices.append(len(columns))
            columns.append((labels == level).astype(np.float64))
            names.append(f"{name}[{level}]")
        groups[name] = tuple(indices)
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"two columns would be named {repeated}; rename a covariate")
    return Design(matrix=np.column_stack(columns), names=tuple(names), covariates=groups)


def _estimable(matrix, rows) -> np.ndarray:
    """Which rows ``c`` make ``c @ beta`` estimable: ``c`` lies in the design's row space."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    projector = np.linalg.pinv(matrix) @ matrix
    gap = np.linalg.norm(rows @ projector - rows, axis=1)
    return gap <= 1e-8 * np.maximum(np.linalg.norm(rows, axis=1), 1e-300)


def _contrast_matrix(contrast, named, n_terms) -> np.ndarray:
    """``(q, n_columns)`` weights from an array, a dict of column names, or a list of dicts."""

    def row_from(weights) -> np.ndarray:
        row = np.zeros(n_terms)
        for key, weight in weights.items():
            if named is None and not isinstance(key, (int, np.integer)):
                raise ValueError("a contrast by column name needs a design built with design()")
            columns = named.columns(key) if named is not None else [int(key)]
            if len(columns) != 1:
                listed = ", ".join(named.names[j] for j in columns)
                raise ValueError(
                    f"{key!r} stands for {len(columns)} columns ({listed}); give each its weight"
                )
            if not 0 <= columns[0] < n_terms:
                raise ValueError(f"contrast column {columns[0]} is outside the design")
            row[columns[0]] += float(weight)
        return row

    if isinstance(contrast, Mapping):
        rows = np.atleast_2d(row_from(contrast))
    elif isinstance(contrast, (list, tuple)) and contrast and isinstance(contrast[0], Mapping):
        rows = np.vstack([row_from(one) for one in contrast])
    else:
        rows = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
        if rows.ndim != 2 or rows.shape[1] != n_terms:
            raise ValueError(
                f"a contrast weighs the design's {n_terms} columns; got shape {rows.shape}"
            )
    if not np.isfinite(rows).all():
        raise ValueError("a contrast's weights must be finite")
    if not np.any(rows, axis=1).all():
        raise ValueError("a contrast row of zeros tests nothing")
    return rows


@dataclass
class LocalTest:
    """What :func:`local_test` returns: one test per column of the scores.

    The columns are the components of a reduction, or any per-subject
    features -- vertex strengths, coupling maps, region pairs -- each tested
    on its own and corrected across all of them.
    """

    statistic: np.ndarray
    """``(n_columns,)`` F statistic for the tested hypothesis; ``NaN`` where a
    column has no residual variance or non-finite scores and so cannot be
    tested."""
    pvalue: np.ndarray
    """``(n_columns,)`` unadjusted p-values, ``NaN`` where the statistic is."""
    adjusted: np.ndarray
    """``(n_columns,)`` p-values after the multiplicity correction."""
    coefficients: np.ndarray
    """``(n_columns, n_terms)`` fitted regression coefficients."""
    residual_dof: int
    """Residual degrees of freedom of each column's model."""
    numerator_dof: int
    """Degrees of freedom of the tested hypothesis."""
    method: str
    """The correction that produced :attr:`adjusted`."""
    permutations: int = 0
    """How many permutations produced the p-values, or 0 if parametric."""
    terms: tuple = ()
    """Columns of the design matrix that were tested, or that the contrast weighs."""
    groups: int = 0
    """How many groups (families) the subjects fall in when ``groups=`` was
    given, and the p-values come from cluster-robust standard errors with
    ``groups - 1`` denominator degrees of freedom; 0 for independent subjects."""
    names: tuple = ()
    """The design's column names when it had them (:func:`design`), else empty."""
    contrast: np.ndarray | None = None
    """``(q, n_terms)`` weights of the contrast tested, ``None`` when ``terms`` were."""
    estimate: np.ndarray | None = None
    """``(n_columns, q)``: what was tested, estimated -- each tested column's
    coefficient, or each contrast row applied to the coefficients."""
    estimate_errors: np.ndarray | None = None
    """``(n_columns, q)`` standard errors of :attr:`estimate`; ``NaN`` where a
    row is not estimable on its own (a factor coded at every level beside the
    intercept)."""
    standard_errors: np.ndarray | None = None
    """``(n_columns, n_terms)`` standard errors of every coefficient, likewise."""
    interval_dof: int = 0
    """Degrees of freedom of the t intervals: :attr:`residual_dof`, or
    ``groups - 1`` with families as clusters."""
    partial_r2: np.ndarray | None = None
    """``(n_columns,)`` share of what the hypothesis's reduced model leaves
    unexplained that the tested part explains: an effect size, from the fits."""

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        """Indices of columns whose adjusted p-value is at or below ``alpha``."""
        return np.flatnonzero(self.adjusted <= alpha)

    def interval(self, level: float = 0.95, coefficients: bool = False):
        """Confidence intervals, ``(lower, upper)``, from the t distribution.

        Of :attr:`estimate` by default, ``(n_columns, q)``; of every coefficient
        with ``coefficients=True``. The standard errors are the classical ones,
        or cluster-robust with ``groups=``, on :attr:`interval_dof` degrees of
        freedom; a permutation test changes the p-values, not the intervals.
        """
        from scipy.stats import t

        if not 0 < level < 1:
            raise ValueError(f"level must lie between 0 and 1, got {level}")
        if coefficients:
            centre, error = self.coefficients, self.standard_errors
        else:
            centre, error = self.estimate, self.estimate_errors
        if centre is None or error is None:
            raise ValueError("this result carries no standard errors")
        half = t.ppf(0.5 + level / 2, self.interval_dof) * np.asarray(error)
        return np.asarray(centre) - half, np.asarray(centre) + half

    def _column(self, term) -> int:
        if isinstance(term, str):
            if term not in self.names:
                raise ValueError(f"no column named {term!r}; the design has {self.names}")
            return self.names.index(term)
        return int(term)

    def effect_map(self, reduction, term=None, alpha: float | None = None):
        """Where on the surface the fitted effect sits.

        Sums the fitted change in connectivity over one endpoint, giving one
        value per vertex. ``term`` is a column of the design matrix, by index
        or by name, and defaults to what was tested: a single contrast's
        estimate, or the first tested column -- not column 0, which is the
        intercept when one was added. With ``alpha`` set, only components
        surviving the correction contribute.
        """
        if term is None and self.contrast is not None:
            if self.contrast.shape[0] != 1:
                raise ValueError("a contrast of several rows has no single effect; pass term=")
            weights = np.asarray(self.estimate)[:, 0]
        else:
            if term is None:
                if not self.terms:
                    raise ValueError("no tested terms recorded; pass term= explicitly")
                term = int(self.terms[0])
            weights = self.coefficients[:, self._column(term)]
        basis = np.asarray(reduction.basis, dtype=np.float64)
        weights = weights * np.asarray(reduction.scales)
        if alpha is not None:
            keep = np.zeros(weights.size, dtype=bool)
            keep[self.significant(alpha)] = True
            weights = np.where(keep, weights, 0.0)
        return (basis * weights) @ (basis.sum(axis=0))

    def to_table(self, path, names=None, level: float = 0.95, index: str = "component"):
        """Write the results to a ``.csv`` or ``.tsv`` table, one row per column tested.

        The columns are ``index`` -- ``component`` by default, counted from 0
        as :meth:`significant` and the reduction's basis count them; ``vertex``
        for a test of per-vertex features -- ``statistic``, ``pvalue``,
        ``adjusted`` and ``partial_r2``; then what was tested -- ``estimate``,
        ``se`` and its ``level`` interval, ``ci_low`` and ``ci_high``, with
        the tested column's name appended when several were tested -- and the
        fitted coefficient of every design column, ``coef_<name>``. ``names``
        names the design matrix's columns, the intercept included when one was
        added; it defaults to the design's own (:func:`design`), else numbers.
        An untestable column's statistic and p-values are empty cells.

        Examples
        --------
        >>> sex.to_table("sex.csv")  # doctest: +SKIP
        """
        from .export import _write_table

        coefficients = np.asarray(self.coefficients)
        if names is None:
            labels = list(self.names) or [str(j) for j in range(coefficients.shape[1])]
        else:
            labels = [names] if isinstance(names, str) else [str(name) for name in names]
        if len(labels) != coefficients.shape[1]:
            raise ValueError(
                f"{len(labels)} names for the design's {coefficients.shape[1]} columns "
                "(column 0 is the intercept when one was added)"
            )
        header = [index, "statistic", "pvalue", "adjusted", "partial_r2"]
        blocks = []
        if self.estimate is not None and self.estimate_errors is not None:
            estimate = np.asarray(self.estimate)
            low, high = self.interval(level)
            if self.contrast is None:
                tags = [labels[t] for t in self.terms]
            else:
                tags = [f"contrast_{r + 1}" for r in range(estimate.shape[1])]
            several = estimate.shape[1] > 1
            for r, tag in enumerate(tags):
                suffix = f"_{tag}" if several else ""
                header += [
                    f"estimate{suffix}",
                    f"se{suffix}",
                    f"ci_low{suffix}",
                    f"ci_high{suffix}",
                ]
                blocks.append((estimate[:, r], self.estimate_errors[:, r], low[:, r], high[:, r]))
        header += [f"coef_{label}" for label in labels]
        effect = (
            self.partial_r2
            if self.partial_r2 is not None
            else np.full(coefficients.shape[0], np.nan)
        )
        rows = (
            [
                k,
                self.statistic[k],
                self.pvalue[k],
                self.adjusted[k],
                effect[k],
                *(column[k] for block in blocks for column in block),
                *coefficients[k],
            ]
            for k in range(self.statistic.size)
        )
        return _write_table(path, header, rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<LocalTest {self.statistic.size} columns, "
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
    contrast=None,
) -> LocalTest:
    """Test each column of the scores for association with the design.

    Fits ``scores[:, k] ~ design`` by least squares for every column and
    tests a hypothesis about its coefficients with an F test -- that the
    ``terms`` columns are zero together, or that a ``contrast`` of them is --
    then corrects across columns. The result carries each tested estimate
    with its standard error and confidence interval
    (:meth:`LocalTest.interval`) and an effect size
    (:attr:`LocalTest.partial_r2`).

    Parameters
    ----------
    scores
        ``(n_subjects, n_columns)``: a reduction's
        :attr:`~sbci.reduction.Reduction.scores`, whose columns are
        components, or any per-subject features -- a vertex's strength, a
        coupling map, the pairs of a region matrix -- whose columns are tested
        one by one, the correction running across all of them.
    design
        A :class:`Design` from :func:`design`, whose columns have names and
        which brings its own intercept; or an ``(n_subjects, n_covariates)``
        array. Unless ``add_intercept`` is false an intercept is prepended to
        an array as column 0, always, so covariate ``j`` is column ``j + 1``
        whatever the covariates hold, and a constant column is refused: beside
        the prepended intercept it is either an intercept of your own, as
        statsmodels' ``add_constant`` makes one, or a covariate that does not
        vary in these subjects (sex in a single-sex subset), and taking it for
        either would misnumber the columns of whoever meant the other. A column
        of zeros is kept -- it fits as nothing, as a dummy for a level absent
        from a stratum should -- but cannot be tested.
    terms
        Which design columns to test, jointly: integer indices into the final
        design matrix, or with a :class:`Design` names too -- a covariate's
        name stands for all its columns, so ``terms=["site"]`` tests a factor
        of any number of levels at once. Each column at most once. Defaults to
        every non-constant column except the intercept. A design with no
        constant column whose columns still combine into one -- dummy codes for
        every level of a factor, with ``add_intercept=False`` -- has no
        intercept column to leave out, and needs ``terms`` named. A boolean
        array is refused: it would be read as the indices 0 and 1.
    method
        Multiplicity correction, one of :data:`METHODS`.
    add_intercept
        Prepend the intercept to an array as column 0. Pass ``False`` to supply
        it yourself, as a constant column of ``design``; the columns are then
        numbered as given. A design carrying its own constant column needs
        ``add_intercept=False``: until October 2026 such a column was taken
        as the intercept. Dummy codes for every level of a factor carry an
        intercept too, implicitly, and are tested exactly as the same factor
        coded against a reference level with the intercept prepended. A
        :class:`Design` keeps its own intercept, or its lack of one.
    permutations
        If positive, p-values come from a permutation test with this many
        draws rather than from the F distribution. The scheme is
        Freedman-Lane: the residuals of the reduced (nuisance-only) model are
        permuted and its fit added back, which keeps the null exact when the
        tested covariate is correlated with the nuisance ones. Use this when
        the scores are not plausibly Gaussian; the correction is still applied
        to the permutation p-values, and the intervals stay the model's.
    seed
        Seed for the permutations.
    groups
        ``(n_subjects,)`` labels, one per subject, for subjects that are not
        independent: the family of each, in a cohort of twins and siblings.
        The hypothesis is then judged by a Wald test with cluster-robust
        standard errors (the usual small-sample correction,
        ``G / (G - 1) * (n - 1) / (n - p)`` for ``G`` groups and ``p``
        columns), and the F statistic's denominator has ``G - 1`` degrees of
        freedom instead of ``n - p``, as do the intervals. It needs a design of
        full column rank and hundreds of groups rather than dozens (the module
        notes), and cannot be combined with ``permutations``. Every subject
        needs a label: a missing one -- ``NaN``, ``None``, or an empty or blank
        string -- is refused, since coded like any other label it would put
        every subject without a family into one family together. A subject
        without a family takes a label of its own.
    contrast
        Instead of ``terms``, a contrast of the coefficients to test for zero:
        a weight per design column -- an array of them, or with a
        :class:`Design` a dict by name, ``{"site[B]": 1, "site[C]": -1}`` --
        or several such rows, tested jointly. It has to be estimable from the
        design: a combination the data can tell apart.

    Examples
    --------
    >>> d = sbci.stats.design({"group": group, "age": age, "motion": motion})  # doctest: +SKIP
    >>> result = sbci.local_test(reduction.scores, d, terms=["group"])          # doctest: +SKIP
    >>> result.significant(0.05)                                                # doctest: +SKIP
    array([0, 3])
    >>> low, high = result.interval(0.95)                                       # doctest: +SKIP
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    named = design if isinstance(design, Design) else None
    if named is not None:
        matrix = np.asarray(named.matrix, dtype=np.float64)
    else:
        matrix = _design_matrix(design, add_intercept)
    n_subjects, n_terms = matrix.shape
    if scores.shape[0] != n_subjects:
        raise ValueError(f"{scores.shape[0]} subjects in scores but {n_subjects} in the design")
    if contrast is not None and terms is not None:
        raise ValueError("give terms= or contrast=, not both")

    hypothesis = None
    if contrast is not None:
        hypothesis = _contrast_matrix(contrast, named, n_terms)
        tested = [int(j) for j in np.flatnonzero(np.any(hypothesis != 0, axis=0))]
    elif terms is None:
        constant = [i for i in range(n_terms) if np.all(matrix[:, i] == matrix[0, i])]
        intercept = [i for i in constant if matrix[0, i] != 0]  # zeros are no intercept
        tested = [i for i in range(n_terms) if i not in constant]
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
        items = (
            [terms]
            if isinstance(terms, str)
            else list(np.atleast_1d(np.asarray(terms, dtype=object)))
        )
        if not items:
            raise ValueError("terms= is empty; name at least one design column")
        if any(isinstance(item, str) for item in items):
            if named is None:
                raise ValueError("terms by name need a design built with sbci.stats.design()")
            tested = [column for item in items for column in named.columns(item)]
        else:
            requested = np.atleast_1d(np.asarray(terms))
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
        empty = [t for t in tested if not matrix[:, t].any()]
        if empty:
            raise ValueError(
                f"column {empty[0]} of the design matrix is all zeros in these subjects, so "
                "there is nothing in it to test"
            )

    codes = None
    if groups is not None:
        # Judged on the labels as given: NumPy would turn a NaN in a list of
        # strings into the string "nan", one more family, before it could be seen.
        given = np.asarray(groups, dtype=object).ravel()
        labels = np.asarray(groups).ravel()
        if labels.size != n_subjects:
            raise ValueError(f"{labels.size} group labels for {n_subjects} subjects")
        missing = int(_missing_labels(given).sum())
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

    if hypothesis is None:
        rows = np.eye(n_terms)[tested]
        reduced_columns = [i for i in range(n_terms) if i not in tested]
        reduced = matrix[:, reduced_columns] if reduced_columns else np.zeros((n_subjects, 0))
    else:
        # The model the hypothesis leaves: the design's columns in every
        # combination the contrast does not constrain, its null space.
        rows = hypothesis
        _, singular, right = np.linalg.svd(hypothesis)
        rank = int((singular > singular.max() * max(hypothesis.shape) * np.finfo(float).eps).sum())
        if rank < hypothesis.shape[0]:
            raise ValueError("the contrast's rows are not independent; drop the redundant ones")
        if not _estimable(matrix, hypothesis).all():
            raise ValueError(
                "the contrast is not estimable from this design: it asks about a combination "
                "of coefficients the data cannot tell apart, such as a column of zeros, or one "
                "dummy of a factor coded at every level beside the intercept"
            )
        free = right[rank:].T
        reduced = matrix @ free if free.shape[1] else np.zeros((n_subjects, 0))

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
        """Coefficients and residual sums of squares, every column at once."""
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

    n_columns = scores.shape[1]
    full_beta, full_ss = fit(scores, matrix)
    coefficients = full_beta.T
    if reduced.shape[1]:
        reduced_beta, reduced_ss = fit(scores, reduced)
        fitted = reduced @ reduced_beta
    else:
        reduced_ss = (scores * scores).sum(axis=0)
        fitted = np.zeros_like(scores)
    statistic = f_statistic(reduced_ss, full_ss)

    # The coefficients' covariance: classical, sigma^2 (X'X)^+ ...
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma2 = full_ss / residual_dof
    covariance = sigma2[:, None, None] * np.linalg.pinv(matrix.T @ matrix)[None, :, :]
    n_groups = 0
    denominator_dof = residual_dof
    if codes is not None:
        # ... or the cluster-robust sandwich. A column of zeros fits as
        # nothing, and the sandwich is formed without it.
        n_groups = int(codes.max()) + 1
        present = matrix.any(axis=0)
        if rank_full < int(present.sum()):
            raise ValueError(
                "groups= needs a design of full column rank; a constant covariate duplicates "
                "the intercept, and dummy codes for every level of a factor do too"
            )
        position = np.cumsum(present) - 1
        statistic, robust = _clustered_wald(
            matrix[:, present],
            scores,
            full_beta[present],
            codes,
            n_groups,
            [int(position[t]) for t in tested] if hypothesis is None else rows[:, present],
            np.isfinite(statistic),
        )
        covariance = np.full((n_columns, n_terms, n_terms), np.nan)
        kept = np.flatnonzero(present)
        covariance[:, kept[:, None], kept[None, :]] = robust
        denominator_dof = n_groups - 1

    estimable = _estimable(matrix, np.eye(n_terms))
    standard_errors = np.sqrt(np.einsum("kjj->kj", covariance))
    standard_errors[:, ~estimable] = np.nan
    estimate = coefficients @ rows.T
    spread = np.einsum("qi,kij,rj->kqr", rows, np.nan_to_num(covariance), rows)
    estimate_errors = np.sqrt(np.einsum("kqq->kq", spread))
    estimate_errors[:, ~_estimable(matrix, rows)] = np.nan
    if codes is not None:
        # Rows touching a column of zeros are not estimable; the rest read the sandwich.
        estimate_errors[:, np.any(rows[:, ~present] != 0, axis=1)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        partial_r2 = np.where(np.isfinite(statistic), (reduced_ss - full_ss) / reduced_ss, np.nan)

    if permutations > 0:
        # Freedman-Lane: permute the residuals of the reduced model, add its fit
        # back, and refit both models to the surrogate scores -- for every
        # column in one solve per permutation.
        rng = np.random.default_rng(seed)
        residuals = scores - fitted
        exceed = np.zeros(n_columns)
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
        # Over the columns that could be tested, as benjamini_hochberg counts them.
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
        names=tuple(named.names) if named is not None else (),
        contrast=hypothesis,
        estimate=estimate,
        estimate_errors=estimate_errors,
        standard_errors=standard_errors,
        interval_dof=denominator_dof,
        partial_r2=partial_r2,
    )


def _clustered_wald(matrix, scores, beta, codes, n_groups, tested, testable):
    """Wald F statistics with cluster-robust standard errors, and the covariance they come from.

    For each column, the coefficients' covariance is the sandwich
    ``c * B @ M @ B`` with ``B = (X'X)^-1`` and ``M`` the sum over groups of
    ``(X_g' e_g)(X_g' e_g)'``, where ``e_g`` is the group's residuals and
    ``c = G / (G - 1) * (n - 1) / (n - p)``; the statistic is
    ``b' V^-1 b / q`` over the ``q`` tested estimates ``b``. ``tested`` is
    the tested columns' indices, or a ``(q, p)`` contrast.
    """
    n_subjects, n_terms = matrix.shape
    residual = scores - matrix @ beta
    # Each group's score X_g' e_g, for every column: (groups, terms, columns).
    contributions = np.zeros((n_groups, n_terms, scores.shape[1]))
    np.add.at(contributions, codes, matrix[:, :, None] * residual[:, None, :])
    meat = np.einsum("gik,gjk->kij", contributions, contributions)
    bread = np.linalg.inv(matrix.T @ matrix)
    correction = n_groups / (n_groups - 1) * (n_subjects - 1) / (n_subjects - n_terms)
    covariance = correction * np.einsum("ij,kjl,lm->kim", bread, meat, bread)
    if isinstance(tested, np.ndarray) and tested.ndim == 2:
        tested_beta = beta.T @ tested.T  # (columns, q)
        block = np.einsum("qi,kij,rj->kqr", tested, covariance, tested)
        q = tested.shape[0]
    else:
        index = np.asarray(tested)
        tested_beta = beta[index].T  # (columns, q)
        block = covariance[:, index][:, :, index]  # (columns, q, q)
        q = index.size
    statistic = np.full(scores.shape[1], np.nan)
    for k in np.flatnonzero(testable):
        try:
            solved = np.linalg.solve(block[k], tested_beta[k])
        except np.linalg.LinAlgError:
            continue
        statistic[k] = float(tested_beta[k] @ solved) / q
    return statistic, covariance


# --- test-retest -------------------------------------------------------------------

ICC_KINDS = ("agreement", "consistency")
"""What :func:`icc` measures: absolute agreement, ICC(2,1), or consistency, ICC(3,1)."""


def icc(sessions, kind: str = "agreement") -> np.ndarray:
    """Test-retest reliability of every feature: the intraclass correlation across sessions.

    Parameters
    ----------
    sessions
        ``(k, n_subjects, n_features)``, or a list of ``k`` arrays of
        ``(n_subjects, n_features)``: the same features measured ``k >= 2``
        times on the same subjects, in one subject order -- two days' FC, or
        two halves of the same streamlines.
    kind
        ``"agreement"``, Shrout and Fleiss's ICC(2,1) (McGraw and Wong's
        ICC(A,1)): two-way random effects, absolute agreement, so a shift
        between sessions counts against it; or ``"consistency"``, ICC(3,1)
        (ICC(C,1)): a shift between sessions forgiven.

    Returns
    -------
    ``(n_features,)``, from each feature's two-way ANOVA mean squares between
    subjects (``MSR``), between sessions (``MSC``) and residual (``MSE``)::

        agreement    (MSR - MSE) / (MSR + (k - 1) MSE + k (MSC - MSE) / n)
        consistency  (MSR - MSE) / (MSR + (k - 1) MSE)

    ``NaN`` where a feature has a missing value or no variance.

    Examples
    --------
    >>> ratings = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
    ...                     [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
    >>> sessions = ratings.T[:, :, None]          # Shrout and Fleiss (1979), Table 2
    >>> round(float(icc(sessions)[0]), 2), round(float(icc(sessions, "consistency")[0]), 2)
    (0.29, 0.71)
    """
    if kind not in ICC_KINDS:
        raise ValueError(f"kind must be one of {ICC_KINDS}, got {kind!r}")
    data = np.asarray(sessions, dtype=np.float64)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(
            f"sessions are (k, n_subjects, n_features), or a list of (n_subjects, n_features) "
            f"arrays; got shape {data.shape}"
        )
    k, n, _ = data.shape
    if k < 2 or n < 2:
        raise ValueError(f"an ICC needs two sessions and two subjects at least; got {k} and {n}")
    grand = data.mean(axis=(0, 1))
    between_subjects = k * ((data.mean(axis=0) - grand) ** 2).sum(axis=0)
    between_sessions = n * ((data.mean(axis=1) - grand) ** 2).sum(axis=0)
    residual = ((data - grand) ** 2).sum(axis=(0, 1)) - between_subjects - between_sessions
    msr = between_subjects / (n - 1)
    msc = between_sessions / (k - 1)
    mse = residual / ((n - 1) * (k - 1))
    if kind == "agreement":
        denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    else:
        denominator = msr + (k - 1) * mse
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (msr - mse) / denominator
    usable = np.isfinite(data).all(axis=(0, 1)) & (denominator > 0)
    return np.where(usable, value, np.nan)


@dataclass
class Identification:
    """What :func:`identification` returns: how well each subject's two sessions pick each other."""

    similarity: np.ndarray
    """``(n, n)``: Pearson's r of subject ``i``'s first session with subject ``j``'s second."""
    accuracy: tuple
    """``(first to second, second to first)``: the share of subjects whose own other session
    is the most similar of all subjects'."""
    within: float
    """Mean similarity of a subject's two sessions: the diagonal."""
    between: float
    """Mean similarity of two subjects' sessions: off the diagonal. ``within - between`` is
    Amico and Goni's differential identifiability."""
    features: int
    """How many features entered: those present in every subject's two sessions."""


#: Values, subjects times features, that :func:`identification` holds in float64 at a time.
_BLOCK_ELEMENTS = 1 << 24


def identification(first, second) -> Identification:
    """Whether each subject's second session is most like its own first: connectome fingerprinting.

    Finn et al. (Nature Neuroscience, 2015) identified subjects from their
    functional connectomes across days. ``first`` and ``second`` are
    ``(n_subjects, n_features)`` in one subject order -- whole connectomes'
    upper triangles, region matrices, maps. A feature missing in any subject
    is left out of every comparison. The similarity is Pearson's r across
    features, summed in float64 a block of features at a time: float32 input
    is not copied whole, so a hundred whole ico4 connectomes fit, and the sums
    keep float64's precision, which they need. Smoothed SC is mostly near zero
    with a few large values, and float32 sums over its millions of pairs drift
    by more than a percent, to correlations above one.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> trait = rng.standard_normal((5, 40))
    >>> result = identification(trait + 0.3 * rng.standard_normal((5, 40)),
    ...                         trait + 0.3 * rng.standard_normal((5, 40)))
    >>> result.accuracy
    (1.0, 1.0)
    """
    one, two = np.asarray(first), np.asarray(second)
    if one.ndim != 2 or one.shape != two.shape:
        raise ValueError(
            f"first and second are (n_subjects, n_features) alike; got {one.shape} and {two.shape}"
        )
    n = one.shape[0]
    if n < 2:
        raise ValueError("identification needs two subjects at least")
    present = np.isfinite(one).all(axis=0) & np.isfinite(two).all(axis=0)
    if not present.any():
        raise ValueError("no feature is present in every subject")
    columns = np.flatnonzero(present)
    step = max(1, _BLOCK_ELEMENTS // n)
    if columns.size == present.size:
        blocks = [slice(start, start + step) for start in range(0, columns.size, step)]
    else:
        blocks = [columns[start : start + step] for start in range(0, columns.size, step)]

    means = [
        sum(rows[:, block].sum(axis=1, dtype=np.float64) for block in blocks) / columns.size
        for rows in (one, two)
    ]
    squares = np.zeros((2, n))
    cross = np.zeros((n, n))
    for block in blocks:
        a = np.asarray(one[:, block], dtype=np.float64) - means[0][:, None]
        b = np.asarray(two[:, block], dtype=np.float64) - means[1][:, None]
        squares[0] += np.einsum("ij,ij->i", a, a)
        squares[1] += np.einsum("ij,ij->i", b, b)
        cross += a @ b.T
    norms = np.sqrt(squares)
    if not (norms > 0).all():
        raise ValueError("a subject's features do not vary, so it has no correlation with any")
    similarity = cross / np.outer(norms[0], norms[1])
    own = np.arange(n)
    forward = float((similarity.argmax(axis=1) == own).mean())
    backward = float((similarity.argmax(axis=0) == own).mean())
    off = ~np.eye(n, dtype=bool)
    return Identification(
        similarity=similarity,
        accuracy=(forward, backward),
        within=float(np.diag(similarity).mean()),
        between=float(similarity[off].mean()),
        features=int(present.sum()),
    )
