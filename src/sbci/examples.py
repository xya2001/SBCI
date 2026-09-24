"""A runnable connectome, so the package can be tried before the data lands.

The five-minute start in the README begins with ``sbci download``, which needs
the WP1 data release (SPEC_QUESTIONS.md item 6). Until that exists, anyone who
installs this package has nothing to run it on. :func:`example` closes that
gap: it builds a connectome with the right shape, the right grid and the real
medial wall, so every call in the README works today -- including
:meth:`~sbci.ContinuousConnectome.smooth` and :func:`sbci.endpoints_align`,
because the structural example carries the synthetic streamline endpoints its
density was smoothed from.

**The connectivity is synthetic.** The structural example draws
:data:`N_STREAMLINES` streamline endpoints at random around a handful of
bundles on the sphere, keeps only the draws that land in cortex, and smooths
them with the default kernel exactly as ``smooth()`` would, so re-smoothing
gives the file back. The functional example correlates synthetic timeseries.
Nothing was measured from anyone's brain, and the metadata says so:
``pipeline_version`` reads ``synthetic-example`` and ``streamline_weighting``
says the endpoints were drawn, not tracked. Use it to learn the API, to write
tests, and to check that a plotting stack works. Never use it to make a claim
about brains.

What *is* real: the ico4 grid, the vertex areas, the medial-wall mask (taken
from the bundled Desikan atlas), and the file format. A file written by
:func:`example` passes ``sbci validate``.

:func:`example_cohort` goes one step further and builds a cohort with a known
answer: every subject draws its streamlines around the *same* bundles, jittered
in weight and position, and one bundle's weight scales with a synthetic age.
Alignment, reduction and inference then have something real to recover, and
the :class:`Cohort` carries the truth to compare against.

Building a structural subject takes a second or two, most of it the kernel.
The last two examples and the last cohort are kept, so repeated calls with the
same arguments are free and return independent copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import grid, spec
from .atlas import load_atlas
from .connectome import ContinuousConnectome
from .metadata import Metadata, template
from .surface import load_surface

#: Bundles the synthetic structure is built from: each is a smooth bump on the
#: sphere, with its mirror image in the other hemisphere. Enough that a
#: parcellation into a few hundred regions still shows variation.
N_COMPONENTS = 48

#: Concentration of each bump. About 25 gives a spatial scale of roughly 20
#: degrees, the scale at which real cortical connectivity varies.
CONCENTRATION = 25.0

#: Streamlines in the structural example: enough that the smoothed density
#: covers the cortex (about six million vertex pairs), few enough to build in
#: about a second.
N_STREAMLINES = 20_000

#: Fraction of synthetic streamlines with both ends in one hemisphere.
#: Inter-hemispheric connectivity is genuinely the sparser part.
SAME_HEMISPHERE = 0.8

#: Individual variation in a cohort: the bundle weights are multiplied by a
#: log-normal factor with this standard deviation ...
COHORT_WEIGHT_SPREAD = 0.10

#: ... and the bundle centres are displaced by a smooth random tangent field of
#: about this many radians (a degree), so subjects differ in anatomy while a
#: bundle stays a bundle. Calibrated so that a rank-4 FPCA of ten subjects still
#: finds the planted effect; at 0.05 (three degrees) it no longer does, which
#: is what the alignment methods are for.
COHORT_ANATOMY_SPREAD = 0.02


def _unit(vertices: np.ndarray) -> np.ndarray:
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def _centres(sphere, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bundle centres: the directions of random cortical vertices of the left hemisphere.

    The right hemisphere uses their mirror images (``x -> -x``), so the
    synthetic brain is roughly homotopic, as real ones are; the mirror of a
    cortical direction can land on the other medial wall, which the endpoint
    draw rejects.
    """
    half = sphere.n_vertices // 2
    cortical = np.flatnonzero(mask[:half])
    return _unit(sphere.vertices)[rng.choice(cortical, size=N_COMPONENTS, replace=False)]


def _mirror(centres: np.ndarray, hemisphere: int) -> np.ndarray:
    return centres if hemisphere == 0 else centres * np.array([-1.0, 1.0, 1.0])


def _bumps(sphere, centres: np.ndarray) -> np.ndarray:
    """Smooth, positive, localized fields on the sphere, one column per bundle.

    Each is a von Mises-Fisher bump around its centre, which is the cheapest way
    to get a field that is smooth on a sphere without forming an ``n x n``
    distance matrix.
    """
    half = sphere.n_vertices // 2
    unit = _unit(sphere.vertices)
    fields = np.empty((sphere.n_vertices, N_COMPONENTS), dtype=np.float32)
    for hemisphere, rows in ((0, slice(0, half)), (1, slice(half, None))):
        fields[rows] = np.exp(CONCENTRATION * (unit[rows] @ _mirror(centres, hemisphere).T - 1.0))
    return fields


def _streamline_endpoints(sphere, mask, centres, weights, rng: np.random.Generator, count: int):
    """Draw ``count`` synthetic streamlines and locate their endpoints on the grid.

    A streamline picks a bundle with probability proportional to its weight and
    a hemisphere for each end, the same one with probability
    :data:`SAME_HEMISPHERE`. Each end is the bundle's centre in that hemisphere
    plus normal jitter of ``1 / sqrt(CONCENTRATION)``, projected back to the
    sphere: the bump's own spread. Ends that land on the medial wall are drawn
    again, since tractography ends in cortex. Every endpoint is then located on
    its hemisphere's mesh the way the pipeline stores it -- a triangle and
    barycentric weights, rounded to the float32 the file keeps -- so the example
    and a file written from it describe exactly the same endpoints.
    """
    from .alignment import MeshQuery
    from .smoothing import Endpoints

    half, faces_per_hemi = sphere.n_vertices // 2, len(sphere.faces) // 2
    queries = []
    for name in ("L", "R"):
        part = sphere.hemisphere(name)
        queries.append(MeshQuery(_unit(part.vertices), part.faces))

    bundle = rng.choice(N_COMPONENTS, size=count, p=weights / weights.sum())
    first = rng.integers(0, 2, size=count)
    same = rng.random(count) < SAME_HEMISPHERE
    side = np.stack([first, np.where(same, first, 1 - first)], axis=1)
    spread = 1.0 / np.sqrt(CONCENTRATION)

    vertex = np.empty((count, 2), dtype=np.int64)
    triangle = np.empty((count, 2), dtype=np.int64)
    bary = np.empty((count, 2, 3))
    pending = np.ones((count, 2), dtype=bool)
    while pending.any():
        rows, ends = np.nonzero(pending)
        hemispheres = side[rows, ends]
        points = np.empty((rows.size, 3))
        for hemisphere in (0, 1):
            pick = hemispheres == hemisphere
            points[pick] = _mirror(centres, hemisphere)[bundle[rows[pick]]]
        points += spread * rng.normal(size=points.shape)
        points = _unit(points)
        for hemisphere in (0, 1):
            pick = hemispheres == hemisphere
            if not pick.any():
                continue
            w, indices, faces = queries[hemisphere].query_faces(points[pick])
            w = w.astype(np.float32).astype(np.float64)  # what the file stores
            nearest = indices[np.arange(len(indices)), np.argmax(w, axis=1)]
            vertex[rows[pick], ends[pick]] = nearest + hemisphere * half
            triangle[rows[pick], ends[pick]] = faces + hemisphere * faces_per_hemi
            bary[rows[pick], ends[pick]] = w
        pending[rows, ends] = ~mask[vertex[rows, ends]]  # medial wall: draw again

    return Endpoints.from_global(
        vertex[:, 0],
        vertex[:, 1],
        triangle_in=triangle[:, 0],
        triangle_out=triangle[:, 1],
        barycentric_in=bary[:, 0],
        barycentric_out=bary[:, 1],
    )


def _deform(centres: np.ndarray, rng: np.random.Generator, spread: float) -> np.ndarray:
    """One subject's anatomy: the bundle centres moved by a smooth random tangent field.

    The field is three very broad bumps, each carrying a random vector, so
    neighbouring bundles move together and the deformation stays smooth.
    """
    if spread <= 0:
        return centres
    poles = _unit(rng.normal(size=(3, 3)))
    vectors = spread * rng.normal(size=(3, 3))
    field = np.exp(3.0 * (centres @ poles.T - 1.0)) @ vectors
    field -= (field * centres).sum(axis=1, keepdims=True) * centres  # tangential
    return _unit(centres + field)


def _structural(sphere, mask, area, centres, weights, rng, n_streamlines: int, metadata):
    """A structural connectome: the default kernel applied to freshly drawn endpoints."""
    from .smoothing import smooth

    n = sphere.n_vertices
    endpoints = _streamline_endpoints(sphere, mask, centres, weights, rng, n_streamlines)
    # smooth() defines the density from the endpoints exactly as it would for
    # a real file, so re-smoothing the result reproduces it.
    seedling = ContinuousConnectome(
        data=np.zeros(n * (n - 1) // 2, dtype=np.float32),
        area=area,
        mask=mask,
        metadata=metadata,
        coords=sphere.vertices,
        endpoints=endpoints,
    )
    return smooth(seedling, kernel="shk", mask_medial_wall=True)


def _functional(sphere, mask, area, centres, amplitudes, rng, metadata):
    """A functional connectome: correlated synthetic timeseries, which is how FC is made."""
    fields = _bumps(sphere, centres) * np.sqrt(amplitudes).astype(np.float32)
    signals = fields @ rng.normal(size=(N_COMPONENTS, 240)).astype(np.float32)
    signals += rng.normal(scale=0.35, size=signals.shape).astype(np.float32)
    signals -= signals.mean(axis=1, keepdims=True)
    signals /= np.linalg.norm(signals, axis=1, keepdims=True)
    dense = signals @ signals.T
    # The medial wall carries no connectivity, as the spec requires.
    dense[~mask, :] = 0.0
    dense[:, ~mask] = 0.0
    np.fill_diagonal(dense, 0.0)
    return ContinuousConnectome(
        data=grid.to_condensed(dense).astype(np.float32),
        area=area,
        mask=mask,
        metadata=metadata,
        coords=sphere.vertices,
    )


def _metadata(
    modality: str,
    version: str,
    n_streamlines: int = 0,
    bandwidth: float = 0.0,
    kind: str = "example",
):
    """Complete metadata that says plainly the file is synthetic."""
    common = dict(
        normalization="unit-mass" if modality == "sc" else "none",
        registration_reference="fsaverage",
        pipeline_version=f"synthetic-{kind}/{version}",
        container_version=f"none (synthetic {kind}, not pipeline output)",
    )
    if modality == "sc":
        common.update(
            streamline_count=int(n_streamlines),
            streamline_weighting="none (synthetic example: endpoints drawn at random, not tracked)",
            kernel="shk",
            bandwidth=float(bandwidth),
        )
    else:
        common.update(fc_nuisance_model="none (synthetic example, no confounds to regress)")
    return template(modality, **common)


def example(
    modality: str = "sc", seed: int = 0, n_streamlines: int = N_STREAMLINES
) -> ContinuousConnectome:
    """Build a synthetic connectome on the real ico4 grid.

    Everything in the README runs against it, so the package can be tried
    without waiting for the data release. The connectivity is generated, not
    measured -- see this module's docstring.

    Parameters
    ----------
    modality
        ``"sc"`` for a nonnegative density of unit mass, smoothed with the
        default kernel from synthetic streamline endpoints that the connectome
        carries; ``"fc"`` for a signed correlation matrix built from synthetic
        timeseries, which carries no endpoints because correlations have none.
    seed
        Seed for the generator, so the same call always gives the same file.
        Different seeds give different subjects, which gives the alignment
        methods something to do.
    n_streamlines
        Streamlines to draw for ``"sc"``, :data:`N_STREAMLINES` by default.
        Ignored for ``"fc"``.

    Returns
    -------
    ContinuousConnectome
        Valid, and it round-trips through :meth:`ContinuousConnectome.save`.

    Examples
    --------
    >>> import sbci
    >>> cc = sbci.example()
    >>> cc.n_vertices
    5124
    >>> cc.modality
    'sc'
    >>> dense = cc.dense().astype("float64")   # a density integrates to one
    >>> round(float(cc.area @ dense @ cc.area), 6)
    1.0
    >>> float(dense[~cc.mask].sum())           # nothing on the medial wall
    0.0
    >>> cc.endpoints.n_streamlines             # the streamlines it was smoothed from
    20000
    >>> bool(sbci.example("fc").data.min() < 0)   # FC is signed
    True

    Re-smoothing the endpoints gives the same file back, and two seeds can be
    aligned by their endpoints::

        again = cc.smooth(kernel="shk", mask_medial_wall=True)   # again.data == cc.data
        result = sbci.endpoints_align([cc, sbci.example(seed=1)], max_iterations=5)
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")
    if modality == "sc" and int(n_streamlines) < 1:
        raise ValueError(f"n_streamlines must be at least 1, got {n_streamlines}")
    return _copy(_build(modality, int(seed), int(n_streamlines) if modality == "sc" else 0))


@lru_cache(maxsize=2)
def _build(modality: str, seed: int, n_streamlines: int) -> ContinuousConnectome:
    """The prototype for one set of arguments; :func:`example` hands out copies of it."""
    from . import __version__

    rng = np.random.default_rng(seed)
    sphere = load_surface("sphere")
    # The mask is real: the medial wall carries no Desikan region.
    mask = load_atlas("Desikan").labels != 0
    area = _vertex_areas(sphere)
    centres = _centres(sphere, mask, rng)

    if modality == "sc":
        from .smoothing import DEFAULT_SIGMA

        weights = rng.gamma(2.0, 1.0, size=N_COMPONENTS)
        metadata = _metadata("sc", __version__, n_streamlines, DEFAULT_SIGMA)
        return _structural(sphere, mask, area, centres, weights, rng, n_streamlines, metadata)
    amplitudes = np.ones(N_COMPONENTS)
    return _functional(sphere, mask, area, centres, amplitudes, rng, _metadata("fc", __version__))


def _copy(connectome: ContinuousConnectome) -> ContinuousConnectome:
    """An independent copy, so callers can edit what they get without touching the cache."""
    return ContinuousConnectome(
        data=connectome.data.copy(),
        area=connectome.area.copy(),
        mask=connectome.mask.copy(),
        metadata=Metadata(dict(connectome.metadata.fields)),
        coords=connectome.coords,  # the bundled sphere, which is read-only
        endpoints=None if connectome.endpoints is None else connectome.endpoints.copy(),
    )


@dataclass
class Cohort:
    """What :func:`example_cohort` returns: synthetic subjects, and the truth about them."""

    connectomes: list
    """One :class:`~sbci.ContinuousConnectome` per subject, on the ico4 grid."""
    age: np.ndarray
    """``(n_subjects,)`` synthetic ages in years, uniform on 20 to 80; the covariate."""
    effect_bundle: int
    """Which of the :data:`N_COMPONENTS` bundles was scaled with age."""
    truth: np.ndarray
    """``(n_vertices,)`` the planted bundle's field over the surface, both hemispheres:
    what an effect map recovered from the cohort should resemble."""
    seed: int
    """The seed everything was drawn from."""

    @property
    def n_subjects(self) -> int:
        """Subjects in the cohort."""
        return len(self.connectomes)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Cohort of {self.n_subjects} synthetic subjects, seed {self.seed}>"


def example_cohort(
    n_subjects: int = 10,
    seed: int = 0,
    effect: float = 0.7,
    modality: str = "sc",
    n_streamlines: int = N_STREAMLINES,
    variation: float = COHORT_WEIGHT_SPREAD,
    anatomy: float = COHORT_ANATOMY_SPREAD,
) -> Cohort:
    """Build a synthetic cohort with shared anatomy, individual variation and one planted effect.

    Every subject is an :func:`example` drawn around the same bundles: the
    centres and base weights come from ``seed``, each subject multiplies the
    weights by a log-normal factor (:data:`COHORT_WEIGHT_SPREAD`), moves the
    centres by a smooth random field (:data:`COHORT_ANATOMY_SPREAD`) and draws
    its own streamlines. One bundle of median weight is scaled by
    ``1 + effect * z``, where ``z`` is the subject's age standardized to
    ``[-1, 1]``, so its connectivity rises with age; ``effect=0.7`` means the
    oldest subject carries about six times the youngest's weight on it.

    This is what lets the whole pipeline be run with a known answer: at the
    defaults, :func:`sbci.reduce` at rank 4 on ten subjects puts the planted
    bundle in one component, :func:`sbci.local_test` finds that component
    (adjusted p about 1e-4), and its :meth:`~sbci.stats.LocalTest.effect_map`
    correlates above 0.8 with :attr:`Cohort.truth`. The effect is one source of
    variance among the individual variation, not the largest -- at ``effect=0.5``
    a rank-4 FPCA misses it, and with ``anatomy=0.05`` (three degrees) so does
    ``effect=0.7``, which is the case the alignment methods exist for. Nothing
    here was measured from anyone; the metadata says so.

    Parameters
    ----------
    n_subjects
        How many subjects. Each structural subject takes a second or two to
        build and 52 MB in memory.
    seed
        Seed for the shared structure and, through it, every subject.
    effect
        Strength of the planted effect, in ``[0, 1)``.
    modality
        ``"sc"`` for structural subjects with endpoints, ``"fc"`` for functional
        ones whose timeseries share the same bundles and weights (the same
        ``seed`` gives matching cohorts, for coupling analyses).
    n_streamlines
        Streamlines per structural subject.
    variation, anatomy
        Individual variation: the log-normal spread of the bundle weights and
        the size in radians of the random displacement of the bundle centres.
        Anatomy is what the alignment methods remove; turn it up to give them
        more to do, or down to make the planted effect easier to find.

    Examples
    --------
    >>> import sbci
    >>> cohort = sbci.example_cohort(n_subjects=2, n_streamlines=1000)
    >>> cohort.n_subjects, cohort.age.shape, cohort.truth.shape
    (2, (2,), (5124,))
    >>> bool(cohort.age.min() >= 20 and cohort.age.max() <= 80)
    True

    The analysis with a known answer::

        cohort = sbci.example_cohort(n_subjects=10)
        reduction = sbci.reduce(cohort.connectomes, rank=4)
        result = sbci.local_test(reduction.scores, cohort.age)
        result.significant()                      # the component carrying the planted bundle
        result.effect_map(reduction)              # compare with cohort.truth
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")
    if int(n_subjects) < 1:
        raise ValueError(f"n_subjects must be at least 1, got {n_subjects}")
    if not 0.0 <= float(effect) < 1.0:
        raise ValueError(f"effect must be in [0, 1), got {effect}")
    if modality == "sc" and int(n_streamlines) < 1:
        raise ValueError(f"n_streamlines must be at least 1, got {n_streamlines}")
    if float(variation) < 0 or float(anatomy) < 0:
        raise ValueError("variation and anatomy are spreads and cannot be negative")
    prototype = _build_cohort(
        int(n_subjects),
        int(seed),
        float(effect),
        modality,
        int(n_streamlines) if modality == "sc" else 0,
        float(variation),
        float(anatomy),
    )
    return Cohort(
        connectomes=[_copy(cc) for cc in prototype.connectomes],
        age=prototype.age.copy(),
        effect_bundle=prototype.effect_bundle,
        truth=prototype.truth.copy(),
        seed=prototype.seed,
    )


@lru_cache(maxsize=1)
def _build_cohort(
    n_subjects: int,
    seed: int,
    effect: float,
    modality: str,
    n_streamlines: int,
    variation: float,
    anatomy: float,
):
    """The prototype cohort for one set of arguments; :func:`example_cohort` hands out copies."""
    from . import __version__

    shared = np.random.default_rng([seed, 0])
    sphere = load_surface("sphere")
    mask = load_atlas("Desikan").labels != 0
    area = _vertex_areas(sphere)
    centres = _centres(sphere, mask, shared)
    weights = shared.gamma(2.0, 1.0, size=N_COMPONENTS)
    age = shared.uniform(20.0, 80.0, size=n_subjects)
    z = (age - 50.0) / 30.0
    effect_bundle = int(np.argsort(weights)[N_COMPONENTS // 2])  # a typical bundle, not an outlier
    truth = _bumps(sphere, centres)[:, effect_bundle].astype(np.float64)

    connectomes = []
    for subject in range(n_subjects):
        rng = np.random.default_rng([seed, subject + 1])
        own = weights * np.exp(variation * rng.normal(size=N_COMPONENTS))
        own[effect_bundle] *= 1.0 + effect * z[subject]
        centres_own = _deform(centres, rng, anatomy)
        if modality == "sc":
            from .smoothing import DEFAULT_SIGMA

            metadata = _metadata("sc", __version__, n_streamlines, DEFAULT_SIGMA, kind="cohort")
            connectomes.append(
                _structural(sphere, mask, area, centres_own, own, rng, n_streamlines, metadata)
            )
        else:
            metadata = _metadata("fc", __version__, kind="cohort")
            connectomes.append(_functional(sphere, mask, area, centres_own, own, rng, metadata))
    return Cohort(connectomes, age, effect_bundle, truth, seed)


def _vertex_areas(surface) -> np.ndarray:
    """Per-vertex area weights in the pipeline's units: fsaverage vertices.

    The pipeline counts, for each ico4 vertex, the fsaverage vertices nearest to
    it, so the weights are integers summing to 327,684 (SPEC_QUESTIONS.md item
    3). The bundled ico4-to-fsLR overlap matrix carries exactly those counts as
    its column sums, so the example uses the real weights when the overlap is
    bundled and a barycentric estimate scaled to the same total otherwise.
    Either way a unit-mass density has entries of order 1e-11, like a real file.
    """
    try:
        from .io.cifti import vertex_areas

        _, areas = vertex_areas()
        if areas.size == surface.n_vertices:
            return np.asarray(areas, dtype=np.float64)
    except (NotImplementedError, OSError):
        pass
    v, f = surface.vertices, surface.faces
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.zeros(surface.n_vertices, dtype=np.float64)
    np.add.at(areas, f.ravel(), np.repeat(face_area / 3.0, 3))
    return areas * (spec.AREA_TOTAL / areas.sum())
