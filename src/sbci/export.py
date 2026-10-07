"""Results out of the package, in files other software reads.

A per-vertex map -- a seed profile, a coupling map, an effect map, a test's
statistics -- goes out with :func:`save_map`, in the form its file name asks
for:

* ``.dscalar.nii``: CIFTI-2 dense scalars on fsLR-32k, for Connectome
  Workbench, the HCP's tools and plain ``nibabel``. The map moves by the same
  area-weighted operator as the exchange connectome (:mod:`sbci.io.cifti`):
  each fsLR vertex takes the area-weighted mean of the ico4 values covering
  it. A map written whole (``mask=None``, nothing missing) keeps its
  area-weighted mean exactly; with the medial wall left out, an fsLR vertex
  along its edge takes the mean of its cortical part, which moves the mean of
  a map concentrated beside the wall slightly (2.5e-5 for a uniform random
  map; 1.5% in one review's map against the wall).
* ``.func.gii``: GIFTI, one file a hemisphere, ``<stem>.L.func.gii`` and
  ``<stem>.R.func.gii`` -- or, for a BIDS name holding ``hemi-L`` or
  ``hemi-R``, the name with that entity set per hemisphere -- in
  **FreeSurfer's fsaverage4 vertex order**. The ico4
  grid is fsaverage4 -- the same 2,562 vertices a hemisphere and the same
  triangles -- numbered differently (:func:`fsaverage4_order`), so these
  files open on FreeSurfer's own ``fsaverage4`` surfaces and resample to
  ``fsaverage`` or ``fsaverage5`` with its ``mri_surf2surf``. Nothing is
  interpolated: each value is the grid's own.
* ``.csv`` or ``.tsv``: a table with one row per grid vertex, giving its
  hemisphere, its fsaverage4 index, whether it is cortex and, with
  ``atlas=``, its region in any bundled atlas.

Values on the medial wall, which carries no connectivity, are written as
missing by default: ``NaN`` in the surface files, an empty cell in a table. A
masked array's masked entries are missing too; an infinite value is refused
rather than written in one form and dropped in another. Every file is written
under a temporary name and moved into place once complete, so a refusal or
failure part way leaves nothing behind.

Region-level results -- a :meth:`~sbci.ContinuousConnectome.to_atlas` matrix,
discrete coupling, or a map's regional means from :func:`region_means` -- go
out with :func:`save_regions`, labelled with the atlas's region names.
:meth:`sbci.stats.LocalTest.to_table` writes a test's results the same way.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import spec
from .atlas import Atlas, cortex_mask, load_atlas

MAP_SUFFIXES = (".dscalar.nii", ".func.gii", ".csv", ".tsv")
"""The forms :func:`save_map` writes, chosen by the end of the file name."""

TABLE_SUFFIXES = (".csv", ".tsv")
"""The forms a table is written in: comma- or tab-separated."""

COVERAGE = 0.5
"""Share of an fsLR vertex's area that has to lie on ico4 vertices with a
value for it to get one. Below it, mostly on the medial wall, it is missing:
the edge of a masked map stays clean instead of fading into the wall."""


@lru_cache(maxsize=1)
def fsaverage4_order() -> tuple[np.ndarray, np.ndarray]:
    """FreeSurfer's fsaverage4 index of every grid vertex, per hemisphere: ``(left, right)``.

    The ico4 grid is FreeSurfer's fsaverage4 with its vertices numbered
    differently: grid vertex ``i`` of the left hemisphere is fsaverage4 vertex
    ``left[i]`` of ``lh``, and grid vertex ``2562 + i`` is ``right[i]`` of
    ``rh``. ``tools/build_fsaverage4_order.py`` found the numbering two
    independent ways, which agree at every vertex: by matching the inflated
    surfaces (one to one, within 0.21 mm, every triangle carried onto one of
    fsaverage4's) and through the grid's correspondence with full-resolution
    fsaverage, whose first 2,562 vertices are fsaverage4's.

    Examples
    --------
    >>> from sbci.export import fsaverage4_order
    >>> left, right = fsaverage4_order()
    >>> sorted(left.tolist()) == list(range(2562))
    True
    """
    path = resources.files("sbci.data.surfaces") / "fsaverage4_order_ico4.npz"
    with np.load(path, allow_pickle=False) as data:
        orders = tuple(np.asarray(data[key], dtype=np.int64) for key in ("lh", "rh"))
    for order in orders:
        order.setflags(write=False)
    return orders


def _floats(values, what: str = "the maps") -> np.ndarray:
    """Values as float64, a masked entry as NaN; an infinite one refused."""
    array = np.asarray(np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan))
    infinite = int(np.isinf(array).sum())
    if infinite:
        raise ValueError(
            f"{what} hold {infinite} infinite value{'s' if infinite > 1 else ''}; write a missing "
            "value as NaN"
        )
    return array


def _target(path: Path) -> Path:
    """Where writing ``path`` lands: through a symbolic link, the file it points to.

    A folder there is refused, and so is a read-only file, which a plain write
    would refuse too. The file is written beside its target and moved onto it,
    so the target's folder has to exist and take a new file -- a stricter need
    than a plain write's, which can overwrite a file in a read-only folder.
    """
    target = Path(os.path.realpath(path))
    if target.is_dir():
        raise IsADirectoryError(f"{path} is a folder")
    if target.exists() and not os.access(target, os.W_OK):
        raise PermissionError(f"{path} is read-only")
    _writable(path)
    return target


@contextmanager
def _atomic(path: Path):
    """A temporary name beside where ``path`` lands, with its ending; moved onto it once written.

    Entered for both files of a pair before either is written, it checks both
    targets first (:func:`_target`), so a pair is not left half written.
    """
    target = _target(path)
    temporary = target.parent / f".partial-{os.getpid()}-{target.name}"
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _as_maps(values, names=None) -> tuple[list[str], np.ndarray]:
    """Map names and a ``(k, 5124)`` float64 array from whatever form the maps came in."""
    n = spec.N_VERTICES
    if isinstance(values, Mapping):
        if names is not None:
            raise ValueError("names= is for an array of maps; a dict names its maps by its keys")
        labels = [str(key) for key in values]
        rows = []
        for label, value in zip(labels, values.values(), strict=True):
            row = _floats(value, f"map {label!r}")
            if row.shape != (n,):
                raise ValueError(
                    f"map {label!r} has shape {row.shape}; a map is one value per grid vertex, "
                    f"({n},)"
                )
            rows.append(row)
        maps = np.vstack(rows) if rows else np.empty((0, n))
    else:
        array = _floats(values)
        if array.ndim == 1 and array.size == n:
            maps = array[None, :]
        elif array.ndim == 2 and array.shape == (n, n):
            raise ValueError(
                f"a {n} x {n} array is a connectome, not maps; write it with "
                "ContinuousConnectome.to_cifti or save"
            )
        elif array.ndim == 2 and array.shape[1] == n:
            maps = array
        elif array.ndim == 2 and array.shape[0] == n:
            maps = array.T  # one map a column, as Reduction.basis holds its components
        else:
            raise ValueError(
                f"maps have shape {array.shape}; a map is one value per grid vertex ({n}), "
                f"and several are a 2-D array with one axis of {n}, or a dict of maps"
            )
        if names is None:
            count = maps.shape[0]
            labels = ["value"] if count == 1 else [f"value_{k + 1}" for k in range(count)]
        else:
            labels = [names] if isinstance(names, str) else [str(name) for name in names]
            if len(labels) != maps.shape[0]:
                raise ValueError(f"{len(labels)} names for {maps.shape[0]} maps")
    if maps.shape[0] == 0:
        raise ValueError("no maps to save")
    if len(set(labels)) != len(labels):
        raise ValueError(f"map names must differ; got {labels}")
    return labels, maps


def _keep(mask) -> np.ndarray | None:
    """The vertices whose values are written, or ``None`` for all of them."""
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask != "cortex":
            raise ValueError(f"mask must be 'cortex', None or a boolean array, got {mask!r}")
        return cortex_mask()
    keep = np.asarray(mask)
    if keep.dtype != bool or keep.shape != (spec.N_VERTICES,):
        raise ValueError(
            f"mask must be a boolean array over the {spec.N_VERTICES} grid vertices "
            f"(True to write), 'cortex' or None; got {keep.dtype} of shape {keep.shape}"
        )
    return keep


def _form(path: Path, suffixes) -> str:
    """Which of ``suffixes`` the name ends in, in any case; refused before anything is computed."""
    name = path.name.lower()
    for suffix in suffixes:
        if name.endswith(suffix):
            return suffix
    if name.endswith((".dscalar.nii.gz", ".func.gii.gz")):
        raise ValueError(f"{path.name!r}: CIFTI and GIFTI are written uncompressed; drop the .gz")
    raise ValueError(f"{path.name!r} does not say what to write; end the name in one of {suffixes}")


def _writable(path: Path) -> None:
    """Refuse, before anything is computed, a write whose folder cannot take it.

    The folder is the one where ``path`` lands -- through a symbolic link, the
    folder of the file it points to -- since the file is written there under a
    temporary name and moved onto ``path``.
    """
    path = Path(path)
    target = Path(os.path.realpath(path))
    directory = target.parent
    linked = f", where the link {path} points" if path.is_symlink() else ""
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}{linked}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(
            f"cannot write to {directory}{linked}: the file is written there under a temporary "
            "name and moved into place, which needs the folder writable"
        )


def save_map(values, path, names=None, mask="cortex", atlas=None) -> tuple[Path, ...]:
    """Write one or more per-vertex maps in a form other software reads.

    Parameters
    ----------
    values
        One map, an array of the grid's 5,124 values -- what
        :meth:`~sbci.ContinuousConnectome.seed`,
        :meth:`~sbci.ContinuousConnectome.coupling` and
        :meth:`~sbci.stats.LocalTest.effect_map` return -- or several: a 2-D
        array with one axis of 5,124 (``Reduction.basis`` as it is), or a dict
        of named maps.
    path
        The file to write; its ending picks the form (module notes):
        ``.dscalar.nii`` (fsLR-32k CIFTI), ``.func.gii`` (GIFTI in
        fsaverage4's order, written as ``<stem>.L.func.gii`` and
        ``<stem>.R.func.gii``), ``.csv`` or ``.tsv`` (one row per vertex).
    names
        Names for the maps of an array, one each: map names in the CIFTI and
        GIFTI files, column headings in a table. Defaults to ``value``, or
        ``value_1``, ``value_2`` and so on.
    mask
        Vertices whose values are written; the rest are missing. ``"cortex"``,
        the default, leaves out the medial wall; ``None`` writes every vertex;
        or a boolean array over the grid, ``True`` to write.
    atlas
        For a table only: a bundled atlas name or :class:`~sbci.Atlas`, or a
        list of them, each adding a column with every vertex's region (empty
        outside the atlas).

    Returns
    -------
    The paths written: two for GIFTI, one otherwise.

    The fsLR move averages, which suits continuous values; a map of labels
    keeps its values exactly in the GIFTI and table forms.

    Examples
    --------
    >>> profile = cc.seed(region=("Desikan", "LH_precuneus"))          # doctest: +SKIP
    >>> sbci.save_map(profile, "precuneus.dscalar.nii")                # doctest: +SKIP
    >>> sbci.save_map(profile, "precuneus.func.gii")                   # doctest: +SKIP
    (PosixPath('precuneus.L.func.gii'), PosixPath('precuneus.R.func.gii'))
    >>> sbci.save_map(profile, "precuneus.csv", atlas="Desikan")       # doctest: +SKIP
    """
    path = Path(path)
    form = _form(path, MAP_SUFFIXES)
    if atlas is not None and form not in TABLE_SUFFIXES:
        raise ValueError(f"atlas= adds region columns to a table; a {form} file has no columns")
    _writable(path)
    labels, maps = _as_maps(values, names)
    keep = _keep(mask)
    if keep is not None:
        maps = np.where(keep, maps, np.nan)
    if form == ".dscalar.nii":
        return (_write_dscalar(path, labels, maps),)
    if form == ".func.gii":
        return _write_gifti(path, labels, maps)
    atlases = [] if atlas is None else list(atlas) if isinstance(atlas, (list, tuple)) else [atlas]
    return (_write_vertex_table(path, labels, maps, atlases),)


def _write_dscalar(path: Path, labels, maps) -> Path:
    """fsLR-32k dense scalars, moved by the bundled area-weighted operator."""
    from nibabel import cifti2

    from .io import cifti

    operator = cifti.transfer()
    finite = np.isfinite(maps)
    mixed = np.asarray(operator @ np.where(finite, maps, 0.0).T).T
    support = np.asarray(operator @ finite.T.astype(np.float64)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        moved = np.where(support > COVERAGE, mixed / support, np.nan)
    image = cifti2.Cifti2Image(
        moved.astype(np.float32), (cifti2.ScalarAxis(labels), cifti._brain_model_axis())
    )
    image.nifti_header.set_intent(cifti.INTENT_DENSE_SCALARS)
    with _atomic(path) as temporary:
        image.to_filename(str(temporary))
    return path


def _write_gifti(path: Path, labels, maps) -> tuple[Path, Path]:
    """One GIFTI functional file a hemisphere, values in fsaverage4's vertex order."""
    import nibabel as nib
    from nibabel import gifti

    stem = path.name[: -len(".func.gii")]
    if stem.endswith((".L", ".R")):
        raise ValueError(
            f"{path.name!r} already names a hemisphere; give the name without it, as "
            f"{stem[:-2]}.func.gii, and both hemispheres are written"
        )
    entity = re.search(r"(?:^|[_.])hemi-([LR])(?=$|[_.])", stem)

    def named(letter: str) -> str:
        if entity is None:
            return f"{stem}.{letter}.func.gii"
        return f"{stem[: entity.start(1)]}{letter}{stem[entity.end(1) :]}.func.gii"

    half = spec.N_VERTICES_PER_HEMI
    images = []
    for side, (letter, structure) in enumerate((("L", "CortexLeft"), ("R", "CortexRight"))):
        ordered = np.empty((maps.shape[0], half))
        ordered[:, fsaverage4_order()[side]] = maps[:, side * half : (side + 1) * half]
        arrays = [
            gifti.GiftiDataArray(
                np.asarray(row, dtype=np.float32),
                intent="NIFTI_INTENT_NONE",
                datatype="NIFTI_TYPE_FLOAT32",
                meta=gifti.GiftiMetaData({"Name": label}),
            )
            for label, row in zip(labels, ordered, strict=True)
        ]
        image = gifti.GiftiImage(
            darrays=arrays, meta=gifti.GiftiMetaData({"AnatomicalStructurePrimary": structure})
        )
        images.append((path.parent / named(letter), image))
    # Both written under temporary names first: a failure leaves neither hemisphere.
    with _atomic(images[0][0]) as left, _atomic(images[1][0]) as right:
        nib.save(images[0][1], str(left))
        nib.save(images[1][1], str(right))
    return tuple(target for target, _ in images)


def _write_vertex_table(path: Path, labels, maps, atlases) -> Path:
    """One row per grid vertex: where it is, then the maps."""
    loaded = [load_atlas(a) if isinstance(a, str) else a for a in atlases]
    for atlas in loaded:
        if not isinstance(atlas, Atlas):
            raise TypeError(f"atlas= takes bundled atlas names or Atlas objects, got {atlas!r}")
        size = np.asarray(atlas.labels).size
        if size != spec.N_VERTICES:
            raise ValueError(
                f"the atlas {atlas.name!r} labels {size} vertices, but the grid has "
                f"{spec.N_VERTICES}: a label per grid vertex, left hemisphere first"
            )
    half = spec.N_VERTICES_PER_HEMI
    hemisphere = np.repeat(["L", "R"], half)
    fsaverage4 = np.concatenate(fsaverage4_order())
    cortex = cortex_mask()
    regions = [
        np.array(("", *atlas.names), dtype=object)[np.asarray(atlas.labels)] for atlas in loaded
    ]
    header = [
        "vertex",
        "hemisphere",
        "fsaverage4_vertex",
        "cortex",
        *(a.name for a in loaded),
        *labels,
    ]
    rows = (
        [
            vertex,
            hemisphere[vertex],
            int(fsaverage4[vertex]),
            int(cortex[vertex]),
            *(column[vertex] for column in regions),
            *maps[:, vertex],
        ]
        for vertex in range(spec.N_VERTICES)
    )
    return _write_table(path, header, rows)


def _cell(value):
    """A table cell: numbers to ten significant digits, a missing value as an empty cell."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, str)):
        return value
    if isinstance(value, (list, tuple)):
        return ";".join(str(_cell(item)) for item in value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if np.isnan(number) else format(number, ".10g")


def _write_table(path, header, rows) -> Path:
    """Write ``rows`` under ``header``, comma- or tab-separated by the file's ending."""
    path = Path(path)
    form = _form(path, TABLE_SUFFIXES)
    _writable(path)
    if len(set(header)) != len(header):
        repeated = sorted({h for h in header if list(header).count(h) > 1})
        raise ValueError(f"the table would have two columns named {repeated}; name them apart")
    with _atomic(path) as temporary, open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t" if form == ".tsv" else ",", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])
    return path


def region_means(values, atlas, area=None, mask="cortex", fisher_z: bool = False) -> np.ndarray:
    """The area-weighted mean of a per-vertex map over each region of an atlas.

    Parameters
    ----------
    values
        One map over the grid's 5,124 vertices, or a 2-D array of several with
        one axis of 5,124.
    atlas
        A bundled atlas name or an :class:`~sbci.Atlas`.
    area
        Per-vertex area weights. The default is the grid's own vertex areas,
        the area vector every connectome file carries.
    mask
        Vertices that count: ``"cortex"``, the default, leaves out the medial
        wall, as :meth:`~sbci.ContinuousConnectome.to_atlas` does; ``None``
        counts every vertex the atlas labels; or a boolean array.
    fisher_z
        Average the maps' ``arctanh`` and map back with ``tanh``, for maps of
        correlations, as ``to_atlas`` aggregates FC.

    Returns
    -------
    ``(K,)`` for one map, ``(k, K)`` for several, region ``j`` labelled
    ``atlas.names[j]``. A vertex whose value is ``NaN`` is left out of its
    region's mean, and a region with no vertex left is ``NaN``.

    For a structural connectome, a region profile from
    :meth:`~sbci.ContinuousConnectome.seed` averaged over another region gives
    the ``to_atlas(how="mean")`` entry for the two. Not for a functional one:
    ``to_atlas`` averages FC through Fisher z pair by pair, which no mean of a
    seed profile reproduces (they differed by up to 0.04 in a review's test).

    Examples
    --------
    >>> means = sbci.region_means(sc.coupling(fc), "Schaefer200")      # doctest: +SKIP
    >>> sbci.save_regions(means, "Schaefer200", "coupling.csv")         # doctest: +SKIP
    """
    from .parcellation import region_weights

    if isinstance(atlas, str):
        atlas = load_atlas(atlas)
    if isinstance(values, Mapping):
        raise TypeError("region_means takes an array of maps; average a dict's maps one at a time")
    _, maps = _as_maps(values)
    single = np.asarray(values).ndim == 1
    if area is None:
        from .io.cifti import vertex_areas

        area = vertex_areas()[1]
    area = np.asarray(area, dtype=np.float64)
    if area.shape != (spec.N_VERTICES,):
        raise ValueError(f"area has shape {area.shape}; expected ({spec.N_VERTICES},)")
    keep = _keep(mask)
    if keep is not None:
        area = np.where(keep, area, 0.0)
    weights, _ = region_weights(atlas, area)
    finite = np.isfinite(maps)
    filled = np.where(finite, maps, 0.0)
    if fisher_z:
        filled = np.arctanh(np.clip(filled, -0.999999, 0.999999))
    total = np.asarray(weights.T @ filled.T).T
    weight = np.asarray(weights.T @ finite.T.astype(np.float64)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(weight > 0, total / weight, np.nan)
    if fisher_z:
        means = np.tanh(means)
    return means[0] if single else means


def save_regions(values, atlas, path, names=None) -> Path:
    """Write region-level results as a table labelled with the atlas's region names.

    Parameters
    ----------
    values
        One value per region (``K`` of them, in the atlas's order), several --
        a dict of such vectors, or a 2-D array with one axis of ``K`` that is
        not square -- or a region-by-region ``K x K`` matrix such as
        :meth:`~sbci.ContinuousConnectome.to_atlas` returns.
    atlas
        The atlas the values are over: a bundled name or an :class:`~sbci.Atlas`.
    path
        A ``.csv`` or ``.tsv`` file.
    names
        Column headings for an array of vectors; a dict brings its own, and a
        matrix's columns are the regions.

    A vector table has the columns ``label`` (the atlas's label id), ``region``
    and the values; a matrix table has the regions down its first column and
    across its header. Missing values are empty cells.

    Examples
    --------
    >>> sbci.save_regions(sc.to_atlas("Desikan"), "Desikan", "sc_desikan.csv")   # doctest: +SKIP
    """
    if isinstance(atlas, str):
        atlas = load_atlas(atlas)
    count = atlas.n_regions
    if isinstance(values, Mapping):
        if names is not None:
            raise ValueError("names= is for an array; a dict names its columns by its keys")
        labels = [str(key) for key in values]
        if not labels:
            raise ValueError("no columns to save")
        columns = [
            _floats(value, f"{label!r}")
            for label, value in zip(labels, values.values(), strict=True)
        ]
        for label, column in zip(labels, columns, strict=True):
            if column.shape != (count,):
                raise ValueError(
                    f"{label!r} has shape {column.shape}; {atlas.name} has {count} regions"
                )
        return _region_vectors(path, atlas, labels, np.vstack(columns))
    array = _floats(values, "the values")
    if array.ndim == 2 and array.shape == (count, count):
        if names is not None:
            raise ValueError("a region-by-region matrix is labelled by the regions; drop names=")
        header = ["region", *atlas.names]
        rows = ([name, *array[i]] for i, name in enumerate(atlas.names))
        return _write_table(path, header, rows)
    if array.ndim == 1 and array.size == count:
        vectors = array[None, :]
    elif array.ndim == 2 and array.shape[1] == count:
        vectors = array
    elif array.ndim == 2 and array.shape[0] == count:
        vectors = array.T
    elif array.ndim == 1 and array.size == spec.N_VERTICES:
        raise ValueError(
            f"these are {array.size} values, one per grid vertex: average them over "
            f"{atlas.name}'s regions with region_means first, or write them per vertex "
            "with save_map"
        )
    else:
        raise ValueError(
            f"values have shape {array.shape}, but {atlas.name} has {count} regions: give "
            f"one value per region, several as a 2-D array with an axis of {count}, or a "
            f"{count} x {count} matrix"
        )
    if names is None:
        labels = (
            ["value"] if vectors.shape[0] == 1 else [f"value_{k + 1}" for k in range(len(vectors))]
        )
    else:
        labels = [names] if isinstance(names, str) else [str(name) for name in names]
        if len(labels) != vectors.shape[0]:
            raise ValueError(f"{len(labels)} names for {vectors.shape[0]} columns")
    return _region_vectors(path, atlas, labels, vectors)


def _region_vectors(path, atlas: Atlas, labels, vectors) -> Path:
    header = ["label", "region", *labels]
    rows = (
        [int(label), name, *vectors[:, j]]
        for j, (label, name) in enumerate(zip(atlas.region_ids, atlas.names, strict=True))
    )
    return _write_table(path, header, rows)
