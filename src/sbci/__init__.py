"""sbci -- continuous brain connectivity.

Try it with nothing but the package installed::

    pip install "sbci[plotting] @ git+https://github.com/xya2001/SBCI.git"

    import sbci

    cc = sbci.example()                  # synthetic connectome on the real ico4 grid
    M  = cc.to_atlas("Schaefer200")      # 200 x 200 matrix
    p  = cc.seed(vertex=1234)            # profile over the surface
    cc.plot(p)                           # inflated-surface figure
    cc.save("sub-example_sc.h5")         # a file that passes `sbci validate`

With a real file, ``cc = sbci.load("sub-100307_sc.h5")`` and the same calls;
``cc.to_cifti(...)`` writes the exchange file for Connectome Workbench.

Everything in :data:`__all__` is reachable straight off ``sbci``, and so is
every submodule -- ``sbci.stats.local_test`` works after a plain ``import
sbci``. The heavier ones load on first use, so importing this package does not
pull in matplotlib or scipy unless you touch something that needs them.
"""

from __future__ import annotations

import difflib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from .atlas import Atlas, list_atlases, load_atlas
from .connectome import ContinuousConnectome
from .errors import FormatError, InvalidFileError, MissingDataError, SbciError
from .metadata import Metadata, MetadataError
from .spec import SPEC_VERSION
from .surface import Surface, load_surface

__version__ = "0.0.1.dev0"


def load(path: str | Path) -> ContinuousConnectome:
    """Read a connectome from disk.

    The one entry point: give it a ``.h5`` computational file or a
    ``.dconn.nii`` exchange file and it returns a
    :class:`ContinuousConnectome`, validated. Equivalent to
    :meth:`ContinuousConnectome.load`, and named to match :func:`load_atlas`
    and :func:`load_surface`.

    Examples
    --------
    >>> import sbci
    >>> cc = sbci.load("sub-100307_sc.h5")   # doctest: +SKIP
    """
    return ContinuousConnectome.load(path)


#: Names that live in a submodule and are pulled in on first use.
_LAZY: dict[str, str] = {
    "example": "examples",
    "example_cohort": "examples",
    "Cohort": "examples",
    "align": "alignment",
    "Alignment": "alignment",
    "Encore": "alignment",
    "Warp": "alignment",
    "endpoints_align": "conseal",
    "EndpointAlignment": "conseal",
    "EndpointConnectome": "conseal",
    "EndpointWarp": "conseal",
    "reduce": "reduction",
    "Reduction": "reduction",
    "fit_basis": "reduction",
    "project": "reduction",
    "smooth": "smoothing",
    "Endpoints": "smoothing",
    "KERNELS": "smoothing",
    "local_test": "stats",
    "LocalTest": "stats",
    "benjamini_hochberg": "stats",
    "parcellate": "parcellation",
    "validate_file": "validate",
    "plot_surface": "plotting",
    "structure_function_coupling": "coupling",
}

#: Submodules reachable as ``sbci.<name>`` after a plain ``import sbci``.
_SUBMODULES: tuple[str, ...] = (
    "alignment",
    "atlas",
    "connectome",
    "conseal",
    "coupling",
    "errors",
    "examples",
    "grid",
    "io",
    "metadata",
    "parcellation",
    "plotting",
    "reduction",
    "smoothing",
    "spec",
    "stats",
    "surface",
    "validate",
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .alignment import Alignment, Encore, Warp, align
    from .conseal import EndpointAlignment, EndpointConnectome, EndpointWarp, endpoints_align
    from .coupling import structure_function_coupling
    from .examples import Cohort, example, example_cohort
    from .parcellation import parcellate
    from .plotting import plot_surface
    from .reduction import Reduction, fit_basis, project, reduce
    from .smoothing import KERNELS, Endpoints, smooth
    from .stats import LocalTest, benjamini_hochberg, local_test
    from .validate import validate_file


def __getattr__(name: str):
    """Resolve the lazy names, and say something useful when a name is wrong.

    PEP 562 module ``__getattr__``: Python calls this only after ordinary
    lookup fails, so it costs nothing on the common path and still supports
    ``from sbci import smooth``.
    """
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    known = sorted(set(__all__) | set(_SUBMODULES))
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    raise AttributeError(f"module 'sbci' has no attribute {name!r}.{hint}")


def __dir__() -> list[str]:
    """Everything public, so tab completion finds the lazy names too."""
    return sorted(set(__all__) | set(_SUBMODULES) | set(globals()))


__all__ = [
    # the object you hold
    "ContinuousConnectome",
    "load",
    "example",
    "example_cohort",
    "Cohort",
    # parcellations and surfaces
    "Atlas",
    "load_atlas",
    "list_atlases",
    "Surface",
    "load_surface",
    "parcellate",
    # analysis
    "align",
    "Alignment",
    "Encore",
    "Warp",
    "endpoints_align",
    "EndpointAlignment",
    "EndpointConnectome",
    "EndpointWarp",
    "reduce",
    "Reduction",
    "fit_basis",
    "project",
    "smooth",
    "Endpoints",
    "KERNELS",
    "local_test",
    "LocalTest",
    "benjamini_hochberg",
    "structure_function_coupling",
    # figures and files
    "plot_surface",
    "validate_file",
    "Metadata",
    # errors
    "SbciError",
    "FormatError",
    "InvalidFileError",
    "MissingDataError",
    "MetadataError",
    # constants
    "SPEC_VERSION",
    "__version__",
]
