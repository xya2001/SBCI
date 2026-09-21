"""sbci -- continuous brain connectivity.

The five-minute start::

    pip install sbci
    sbci download hcp-ya --subject 100307

    from sbci import ContinuousConnectome, load_atlas
    cc = ContinuousConnectome.load("sub-100307_sc.h5")
    M  = cc.to_atlas(load_atlas("Schaefer200"))
    p  = cc.seed(vertex=1234)
    cc.plot(p)
    cc.to_cifti("sub-100307_sc.dconn.nii")
"""

from __future__ import annotations

from .alignment import align
from .atlas import Atlas, list_atlases, load_atlas
from .connectome import ContinuousConnectome
from .metadata import Metadata, MetadataError
from .reduction import Reduction, fit_basis, reduce
from .spec import SPEC_VERSION
from .surface import Surface, load_surface

__version__ = "0.0.1.dev0"

__all__ = [
    "ContinuousConnectome",
    "Atlas",
    "load_atlas",
    "list_atlases",
    "Surface",
    "load_surface",
    "align",
    "reduce",
    "Reduction",
    "fit_basis",
    "Metadata",
    "MetadataError",
    "SPEC_VERSION",
    "__version__",
]
