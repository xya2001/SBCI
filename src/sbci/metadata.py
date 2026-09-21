"""Metadata contract: the keys every SBCI file must carry, and their checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import spec
from .errors import SbciError


class MetadataError(SbciError, ValueError):
    """Raised when a file's metadata is missing keys or is self-inconsistent."""


@dataclass
class Metadata:
    """Validated metadata for one connectome file.

    The loader refuses a file whose metadata is missing a required key, rather
    than filling in a default: a silently defaulted bandwidth or normalization
    makes two files look comparable when they are not.
    """

    fields: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a metadata key, returning ``default`` if it is absent."""
        return self.fields.get(key, default)

    @property
    def modality(self) -> str:
        """``'sc'`` or ``'fc'``."""
        return self.fields["included_connections"]

    @classmethod
    def from_json(cls, text: str | bytes) -> Metadata:
        """Parse the JSON metadata string stored alongside the connectivity."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise MetadataError("metadata must be a JSON object")
        return cls(loaded)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize back to the stored JSON string."""
        return json.dumps(self.fields, indent=indent, sort_keys=True)

    def missing_keys(self) -> list[str]:
        """Required keys absent from this metadata, in specification order.

        A key present with a ``None`` value counts as missing. Writers build
        their metadata from :func:`template`, which seeds the acquisition-
        dependent keys as ``None``; treating a null as present would let an
        unfilled template through, which is the exact failure this contract
        exists to prevent.

        Modality-specific keys are only required for their own modality: an FC
        file is not expected to declare a streamline count.
        """
        modality = self.fields.get("included_connections")
        skip: set[str] = set()
        if modality == "fc":
            skip.update(spec.SC_ONLY_KEYS)
        elif modality == "sc":
            skip.update(spec.FC_ONLY_KEYS)
        return [
            key
            for key in spec.REQUIRED_METADATA_KEYS
            if key not in skip and self.fields.get(key) is None
        ]

    def validate(self) -> None:
        """Raise :class:`MetadataError` if the metadata is unusable.

        Examples
        --------
        >>> Metadata({"grid": "fsaverage-ico4"}).validate()
        Traceback (most recent call last):
        ...
        sbci.metadata.MetadataError: missing required metadata keys: ...
        """
        # The modality is checked first: it decides which keys are required, so
        # an unknown value must be reported as such rather than as a cascade of
        # keys the caller was never supposed to supply.
        modality = self.fields.get("included_connections")
        if modality is not None and modality not in spec.MODALITIES:
            raise MetadataError(
                f"included_connections must be one of {spec.MODALITIES}, got {modality!r}"
            )

        missing = self.missing_keys()
        if missing:
            raise MetadataError("missing required metadata keys: " + ", ".join(missing))

        grid = self.fields["grid"]
        if grid != spec.GRID:
            raise MetadataError(
                f"grid {grid!r} is not the canonical grid {spec.GRID!r}; "
                "the reduction-fraction and 4121-vertex grids are retired"
            )

        order = tuple(self.fields["hemisphere_order"])
        if order != spec.HEMISPHERE_ORDER:
            raise MetadataError(f"hemisphere_order must be {spec.HEMISPHERE_ORDER}, got {order}")


def template(modality: str, **overrides: Any) -> Metadata:
    """A metadata block with the structural keys filled and the rest empty.

    Useful for writers and for tests. Acquisition-dependent values are left as
    ``None`` on purpose so that a caller who forgets to set them gets a file
    that fails :meth:`Metadata.validate` rather than a plausible-looking lie.
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")

    fields: dict[str, Any] = {
        "spec_version": spec.SPEC_VERSION,
        "grid": spec.GRID,
        "hemisphere_order": list(spec.HEMISPHERE_ORDER),
        "mask": "medial-wall",
        "area_weights": "vertex-area",
        "storage_convention": f"{spec.TRIANGLE}-triangular-{spec.DTYPE}",
        "included_connections": modality,
        "normalization": None,
        "registration_reference": None,
        "pipeline_version": None,
        "container_version": None,
    }
    if modality == "sc":
        fields.update(streamline_count=None, streamline_weighting=None, kernel=None, bandwidth=None)
    else:
        fields.update(fc_nuisance_model=None)

    fields.update(overrides)
    return Metadata(fields)
