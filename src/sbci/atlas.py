"""Bundled atlases, as per-vertex label vectors on the ico4 grid.

Atlases ship inside the package (about 1 MB all told) so that
:meth:`sbci.ContinuousConnectome.to_atlas` never needs a network call, a
FreeSurfer installation, or MATLAB. They are generated once by
``tools/convert_atlases.py`` and committed.

In every bundled atlas, label ``0`` means "no region": the medial wall, and for
partial-coverage atlases such as ``PALS_B12_Visuotopic`` everything outside the
atlas. Regions are numbered ``1..K`` with no gaps, and ``names[i]`` names label
``i + 1``.

``PALS_B12_Lobes`` is the one exception: it is a coarse twelve-lobe
parcellation that assigns *every* vertex, medial wall included, and so has no
label ``0``. A connectome carries its own medial-wall mask, so this changes
nothing downstream, but a lobe matrix built from it does include medial-wall
vertices in its totals.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

from .errors import SbciError

SUFFIX = "_ico4.npz"


class UnknownAtlasError(SbciError, ValueError):
    """No bundled atlas goes by that name."""


#: Short names accepted in place of the file name an atlas is stored under.
#: The brief's five-minute start calls ``load_atlas("Schaefer200")``, so the
#: short forms are part of the public API, not a convenience.
ALIASES: dict[str, str] = {
    "desikan": "aparc",
    "desikan-killiany": "aparc",
    "dk": "aparc",
    "destrieux": "aparc.a2009s",
    "destrieux2005": "aparc.a2005s",
    "glasser": "HCPMMP1",
    "brainnetome": "BN_Atlas",
    "yeo7": "Yeo2011_7Networks_N1000",
    "yeo17": "Yeo2011_17Networks_N1000",
    **{
        f"schaefer{n}": f"Schaefer2018_{n}Parcels_7Networks_order"
        for n in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    },
}


#: How to spell each short name back to a user. The alias table is keyed on
#: lowercase so lookup can be case-insensitive, but a suggestion should read
#: the way the documentation writes it.
DISPLAY: dict[str, str] = {
    "dk": "DK",
    "desikan-killiany": "Desikan-Killiany",
    "destrieux2005": "Destrieux2005",
    "yeo7": "Yeo7",
    "yeo17": "Yeo17",
    **{f"schaefer{n}": f"Schaefer{n}" for n in range(100, 1100, 100)},
}


@dataclass(frozen=True)
class Atlas:
    """A parcellation defined on the computational grid.

    Attributes
    ----------
    name
        Atlas name as passed to :func:`load_atlas`.
    labels
        Integer label per vertex, length ``n_vertices``. Label ``0`` is
        unassigned and is dropped by
        :meth:`sbci.ContinuousConnectome.to_atlas`.
    names
        Region names, ``names[i]`` naming ``region_ids[i]``.
    """

    name: str
    labels: np.ndarray
    names: tuple[str, ...]

    @property
    def region_ids(self) -> np.ndarray:
        """Sorted non-zero label ids present in :attr:`labels`."""
        ids = np.unique(self.labels)
        return ids[ids != 0]

    @property
    def n_regions(self) -> int:
        """Number of regions, excluding the unassigned label."""
        return int(self.region_ids.size)

    @property
    def coverage(self) -> float:
        """Fraction of vertices assigned to some region.

        Well under 1.0 even for whole-cortex atlases, because the medial wall
        carries no region. Around 0.55 for a partial atlas such as
        ``PALS_B12_OrbitoFrontal``.
        """
        return float((np.asarray(self.labels) != 0).mean())

    def region_mask(self, region):
        """Boolean mask over vertices for one region, by name or label id.

        This is what :meth:`sbci.ContinuousConnectome.seed` wants for its
        ``region=`` argument, so a region profile reads as one line.

        Parameters
        ----------
        region
            A region name as it appears in :attr:`names`, matched ignoring
            case, or an integer label from :attr:`region_ids`.

        Examples
        --------
        >>> from sbci import load_atlas
        >>> atlas = load_atlas("Desikan")
        >>> mask = atlas.region_mask("LH_bankssts")
        >>> bool(mask.any()), mask.dtype == bool
        (True, True)
        >>> bool((atlas.region_mask(1) == mask).all())
        True
        """
        labels = np.asarray(self.labels)
        if isinstance(region, (int, np.integer)) and not isinstance(region, bool):
            if int(region) not in set(self.region_ids.tolist()):
                raise ValueError(
                    f"{self.name} has no region {int(region)}; labels run 1..{self.n_regions}"
                )
            return labels == int(region)

        wanted = str(region).strip().lower()
        for index, candidate in enumerate(self.names):
            if candidate.strip().lower() == wanted:
                return labels == self.region_ids[index]

        close = difflib.get_close_matches(str(region), list(self.names), n=3, cutoff=0.5)
        if close:
            joined = " or ".join(repr(c) for c in close)
            hint = f" Did you mean {joined}?"
        else:
            hint = ""
        raise ValueError(f"{self.name} has no region named {region!r}.{hint}")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Atlas {self.name} with {self.n_regions} regions>"


def _directory():
    return resources.files("sbci.data.atlases")


@lru_cache(maxsize=1)
def list_atlases() -> tuple[str, ...]:
    """Names of every bundled atlas, sorted.

    Examples
    --------
    >>> from sbci import list_atlases
    >>> "aparc" in list_atlases()
    True
    """
    return tuple(
        sorted(
            entry.name[: -len(SUFFIX)]
            for entry in _directory().iterdir()
            if entry.name.endswith(SUFFIX)
        )
    )


def _key(name: str) -> str:
    """Normalize a name for lookup: case, spaces, hyphens and underscores.

    Applied to both sides of the alias table, so ``"desikan-killiany"``,
    ``"Desikan Killiany"`` and ``"DESIKAN_KILLIANY"`` all resolve alike.
    """
    return re.sub(r"[\s\-_]+", "", name).lower()


def resolve(name: str) -> str:
    """Map a short name such as ``"Schaefer200"`` to its bundled file name.

    A name that is already a bundled atlas is returned unchanged, so exact
    names keep working and are matched case-insensitively.

    Examples
    --------
    >>> resolve("Schaefer200")
    'Schaefer2018_200Parcels_7Networks_order'
    >>> resolve("Desikan")
    'aparc'
    >>> resolve("aparc")
    'aparc'
    """
    available = list_atlases()
    if name in available:
        return name

    key = _key(name)
    normalized = {_key(alias): target for alias, target in ALIASES.items()}
    if key in normalized:
        return normalized[key]

    for candidate in available:
        if _key(candidate) == key:
            return candidate

    lookup = {_key(c): c for c in available}
    lookup.update({_key(a): DISPLAY.get(a, a.title()) for a in ALIASES})
    close = difflib.get_close_matches(_key(name), list(lookup), n=3, cutoff=0.6)
    suggestions = [lookup[c] for c in close]
    if suggestions:
        joined = " or ".join(repr(s) for s in suggestions)
        hint = f" Did you mean {joined}?"
    else:
        hint = " Call sbci.list_atlases() for the full set."
    raise UnknownAtlasError(f"unknown atlas {name!r}.{hint}")


@lru_cache(maxsize=8)
def load_atlas(name: str) -> Atlas:
    """Load a bundled atlas on the ico4 grid.

    Parameters
    ----------
    name
        A bundled atlas name, or one of the short forms in :data:`ALIASES`
        such as ``"Schaefer200"``, ``"Desikan"`` or ``"Glasser"``.

    Examples
    --------
    >>> from sbci import load_atlas
    >>> load_atlas("Schaefer200").n_regions
    200
    >>> load_atlas("Desikan").n_regions
    68
    >>> load_atlas("Desikan").names[0]
    'LH_bankssts'
    """
    resolved = resolve(name)
    path = _directory() / f"{resolved}{SUFFIX}"
    with np.load(path, allow_pickle=False) as data:
        return Atlas(
            name=resolved,
            labels=np.asarray(data["labels"], dtype=np.int32),
            names=tuple(str(n) for n in data["names"]),
        )
