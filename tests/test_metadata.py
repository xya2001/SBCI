"""The metadata contract: a file with missing keys must not load."""

from __future__ import annotations

import numpy as np
import pytest

from sbci.metadata import Metadata, MetadataError, template


def test_template_sc_is_incomplete_until_filled():
    """A template left unfilled fails, rather than defaulting to plausible values."""
    with pytest.raises(MetadataError, match="missing required metadata keys"):
        template("sc").validate()


def test_complete_metadata_validates(sc_metadata):
    sc_metadata.validate()


def test_fc_does_not_require_streamline_keys():
    metadata = template(
        "fc",
        normalization="unit-mass",
        registration_reference="fsaverage",
        pipeline_version="0.0.1.dev0",
        container_version="sbci.sif@sha256:0",
        fc_nuisance_model="24P+aCompCor",
    )
    metadata.validate()
    assert "streamline_count" not in metadata.missing_keys()


def test_sc_does_not_require_the_fc_nuisance_model(sc_metadata):
    assert "fc_nuisance_model" not in sc_metadata.missing_keys()


def test_retired_grid_is_rejected(sc_metadata):
    sc_metadata.fields["grid"] = "fsaverage-ico4_0.94"
    with pytest.raises(MetadataError, match="retired"):
        sc_metadata.validate()


def test_hemisphere_order_is_enforced(sc_metadata):
    sc_metadata.fields["hemisphere_order"] = ["R", "L"]
    with pytest.raises(MetadataError, match="hemisphere_order"):
        sc_metadata.validate()


def test_unknown_modality_is_rejected(sc_metadata):
    sc_metadata.fields["included_connections"] = "ec"
    with pytest.raises(MetadataError, match="included_connections"):
        sc_metadata.validate()


def test_json_round_trip(sc_metadata):
    restored = Metadata.from_json(sc_metadata.to_json())
    restored.validate()
    assert restored.fields == sc_metadata.fields


def test_from_json_rejects_a_bare_array():
    with pytest.raises(MetadataError, match="JSON object"):
        Metadata.from_json("[1, 2, 3]")


def test_template_rejects_unknown_modality():
    with pytest.raises(ValueError, match="modality must be one of"):
        template("dti")


def test_the_storage_convention_and_spec_version_are_checked():
    from sbci.metadata import MetadataError, template

    good = template(
        "fc",
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="x",
        container_version="y",
        fc_nuisance_model="36p",
    )
    good.validate()
    bad = template("fc", **{**good.fields, "storage_convention": "lower-triangular-float64"})
    with pytest.raises(MetadataError, match="storage_convention"):
        bad.validate()
    old = template("fc", **{**good.fields, "spec_version": "9.0.0"})
    with pytest.raises(MetadataError, match="spec_version"):
        old.validate()


def test_json_round_trip_handles_numpy_scalars_and_refuses_nan():
    from sbci.metadata import Metadata, MetadataError

    text = Metadata({"bandwidth": np.float32(0.005), "count": np.int64(3)}).to_json()
    back = Metadata.from_json(text)
    assert back["count"] == 3 and abs(back["bandwidth"] - 0.005) < 1e-6
    with pytest.raises(MetadataError, match="serialized"):
        Metadata({"bandwidth": float("nan")}).to_json()
    with pytest.raises(MetadataError, match="JSON"):
        Metadata.from_json("{not json")
