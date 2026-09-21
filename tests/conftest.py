"""Shared fixtures.

The fixtures use a five-vertex toy grid rather than the real 5124-vertex ico4
grid: every behavior tested here is grid-independent, and a real-grid file is
50 MB, which does not belong in a unit test. The full-grid checks run against
the downloaded tutorial subject and are marked ``needs_data``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbci.atlas import Atlas
from sbci.connectome import ContinuousConnectome
from sbci.grid import to_condensed
from sbci.metadata import template


@pytest.fixture
def area() -> np.ndarray:
    """Per-vertex area weights for the toy grid."""
    return np.array([1.0, 2.0, 3.0, 1.0, 2.0])


@pytest.fixture
def dense() -> np.ndarray:
    """A symmetric toy connectivity matrix with a zero diagonal."""
    return np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 5.0, 6.0, 7.0],
            [2.0, 5.0, 0.0, 8.0, 9.0],
            [3.0, 6.0, 8.0, 0.0, 1.0],
            [4.0, 7.0, 9.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def atlas() -> Atlas:
    """Three regions: A = {0, 1}, B = {2, 3}, C = {4}."""
    return Atlas(name="toy3", labels=np.array([1, 1, 2, 2, 3]), names=("A", "B", "C"))


@pytest.fixture
def sc_metadata():
    """Complete, valid SC metadata."""
    return template(
        "sc",
        normalization="unit-mass",
        registration_reference="fsaverage",
        pipeline_version="0.0.1.dev0",
        container_version="sbci.sif@sha256:0",
        streamline_count=100_000,
        streamline_weighting="sift2",
        kernel="rdk",
        bandwidth=0.005,
    )


@pytest.fixture
def connectome(dense, area, sc_metadata) -> ContinuousConnectome:
    """A valid toy SC connectome: unit mass, with vertex 4 as medial wall."""
    matrix = dense.copy()
    matrix[4, :] = 0.0
    matrix[:, 4] = 0.0
    matrix /= area @ matrix @ area

    mask = np.array([True, True, True, True, False])
    return ContinuousConnectome(
        data=to_condensed(matrix).astype(np.float32),
        area=area,
        mask=mask,
        metadata=sc_metadata,
    )
