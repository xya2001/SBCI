"""The metadata contract: a file with missing keys must not load."""

from __future__ import annotations

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
