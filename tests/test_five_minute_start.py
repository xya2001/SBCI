"""The acceptance test from the brief, run against a real subject.

The brief's lines, as written:

    pip install sbci
    sbci download hcp-ya --subject 100307

    from sbci import ContinuousConnectome, load_atlas
    cc = ContinuousConnectome.load("sub-100307_sc.h5")
    M  = cc.to_atlas(load_atlas("Schaefer200"))
    p  = cc.seed(vertex=1234)
    cc.plot(p)
    cc.to_cifti("sub-100307_sc.dconn.nii")

The subject is the first of the example cohort, eleven HCP Young Adult subjects
on a public Google Drive. The download is 90 MB, the subject's SC and FC, so the
test runs only when asked: set ``SBCI_HCP_DIR`` to a directory that already holds
the file, or ``SBCI_DOWNLOAD=1`` to run the download line itself in a temporary
directory.
Otherwise it skips. The export line is left out: it writes 16.9 GB, and
``scripts/write_exchange_file.py`` checks it end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_data

SUBJECT = "100307"
TUTORIAL = f"sub-{SUBJECT}_sc.h5"


@pytest.fixture(scope="module")
def tutorial_dir(tmp_path_factory):
    """A directory holding the tutorial subject's file, downloaded once; or a skip."""
    local = os.environ.get("SBCI_HCP_DIR")
    if local and (Path(local) / TUTORIAL).exists():
        return Path(local)
    if os.environ.get("SBCI_DOWNLOAD") == "1":
        from sbci.cli import main

        where = tmp_path_factory.mktemp("brief")
        previous = os.getcwd()
        os.chdir(where)
        try:
            assert main(["download", "hcp-ya", "--subject", SUBJECT]) == 0  # the brief's line
        finally:
            os.chdir(previous)
        assert (where / TUTORIAL).exists() and (where / "DATA_USE.txt").exists()
        return where
    pytest.skip("set SBCI_HCP_DIR to a directory holding sub-100307_sc.h5, or SBCI_DOWNLOAD=1")


@pytest.fixture
def workdir(tutorial_dir, monkeypatch):
    """Each test runs where the download line left the file, as the brief's lines do."""
    monkeypatch.chdir(tutorial_dir)
    return tutorial_dir


def test_the_briefs_lines(workdir):
    """Load, parcellate and take a seed, from the file the download line left here."""
    from sbci import ContinuousConnectome, load_atlas

    cc = ContinuousConnectome.load(TUTORIAL)
    M = cc.to_atlas(load_atlas("Schaefer200"))
    p = cc.seed(vertex=1234)
    assert M.shape == (200, 200)
    assert p.shape == (5124,) and (p >= 0).all()
    assert cc.endpoints.n_streamlines == 803_741  # the count the README quotes


def test_the_briefs_plot(workdir):
    """``cc.plot(p)``, which needs the plotting extra."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sbci import ContinuousConnectome

    cc = ContinuousConnectome.load(TUTORIAL)
    figure = cc.plot(cc.seed(vertex=1234))
    assert figure.axes
    plt.close(figure)


def test_the_readme_smoothing(workdir):
    """The README's last quick-start line: the endpoints re-smooth to the stored connectome."""
    import numpy as np

    import sbci

    sc = sbci.load(TUTORIAL)
    again = sc.smooth(kernel="shk", mask_medial_wall=True)
    assert np.corrcoef(np.asarray(again.data), np.asarray(sc.data))[0, 1] > 0.999999
