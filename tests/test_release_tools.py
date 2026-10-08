"""tools/bundle_hcp_cohort.py and tools/zenodo_upload.py: the HCP's terms go with every release."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bundle_hcp_cohort  # noqa: E402
import zenodo_upload  # noqa: E402

HCP_TERMS = "wu-minn-hcp-consortium-open-access-data-use-terms"


def test_bundles_and_the_manifest_carry_the_hcp_terms(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for subject in ("sub-01", "sub-02"):
        for modality in ("sc", "fc"):
            (data / f"{subject}_{modality}.h5").write_bytes(b"not a real connectome")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject,sex,age_bin\nsub-01,F,22-25\nsub-02,M,26-30\n")
    out = tmp_path / "bundles"
    args = SimpleNamespace(data=data, manifest=manifest, out=out, per_bundle=24, cohort="toy")
    assert bundle_hcp_cohort.bundle(args) == 0

    listing = json.loads((out / "bundles.json").read_text())
    assert HCP_TERMS in listing["data_use"]
    assert (out / "DATA_USE.txt").read_text() == listing["data_use"]
    for entry in listing["bundles"]:
        with zipfile.ZipFile(out / entry["name"]) as archive:
            assert archive.read("DATA_USE.txt").decode() == listing["data_use"]

    target = tmp_path / "toy.json"
    args = SimpleNamespace(bundles=out, record=1, target=target, doi="", release="2026-10-07")
    assert bundle_hcp_cohort.manifest(args) == 0
    assert json.loads(target.read_text())["data_use"] == listing["data_use"]


def test_the_zenodo_record_is_restricted_under_the_hcp_terms_and_has_no_licence():
    """An open licence such as CC-BY would not carry the HCP's terms, which require them."""
    notice = bundle_hcp_cohort.terms("toy")
    metadata = zenodo_upload.record_metadata("A title", "A description.", [{"name": "A"}], notice)
    assert metadata["access_right"] == "restricted"
    assert "license" not in metadata
    assert HCP_TERMS in metadata["access_conditions"] and HCP_TERMS in metadata["description"]
    with pytest.raises(SystemExit, match="no data-use terms"):
        zenodo_upload.record_metadata("A title", "A description.", [{"name": "A"}], None)
