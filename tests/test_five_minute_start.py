"""The acceptance test from the brief, run against the tutorial subject.

These are skipped until the WP1 data release exists. They are written now, and
kept passing-or-skipped, so that the day `sbci download` works the acceptance
criterion is already encoded rather than argued about.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

TUTORIAL_SUBJECT = "sub-100307_sc.h5"


@pytest.fixture
def tutorial(tmp_path):
    pytest.skip("tutorial subject not released yet (SPEC_QUESTIONS.md item 6)")


def test_load_to_atlas_seed_plot_export(tutorial):
    """The eight lines of the five-minute start, end to end."""
    from sbci import ContinuousConnectome, load_atlas

    cc = ContinuousConnectome.load(TUTORIAL_SUBJECT)
    assert cc.to_atlas(load_atlas("Schaefer200")).shape == (200, 200)
    assert cc.seed(vertex=1234).shape == (5124,)
