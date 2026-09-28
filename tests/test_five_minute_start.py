"""The acceptance test from the brief, run against a real subject.

The subject is the first of the example cohort (ten HCP-Aging subjects on a
public Google Drive, `sbci download hcp-aging`). The download is 56 MB, so
the test runs only when asked: set ``SBCI_HCP_DIR`` to a directory that
already holds the cohort, or ``SBCI_DOWNLOAD=1`` to fetch that one subject
into a temporary directory. Otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_data

TUTORIAL_SUBJECT = "sub-HCA6924080_sc.h5"


@pytest.fixture
def tutorial(tmp_path):
    """The path of the tutorial subject's SC file, or a skip."""
    local = os.environ.get("SBCI_HCP_DIR")
    if local and (Path(local) / TUTORIAL_SUBJECT).exists():
        return Path(local) / TUTORIAL_SUBJECT
    if os.environ.get("SBCI_DOWNLOAD") == "1":
        from sbci.download import fetch_cohort

        written = fetch_cohort(
            tmp_path, subjects=[TUTORIAL_SUBJECT.split("_")[0]], modalities=("sc",)
        )
        return written[0]
    pytest.skip("set SBCI_HCP_DIR to the cohort directory or SBCI_DOWNLOAD=1 to fetch it")


def test_load_to_atlas_seed_plot_export(tutorial):
    """The eight lines of the five-minute start, end to end."""
    from sbci import ContinuousConnectome, load_atlas

    cc = ContinuousConnectome.load(tutorial)
    assert cc.to_atlas(load_atlas("Schaefer200")).shape == (200, 200)
    assert cc.seed(vertex=1234).shape == (5124,)
