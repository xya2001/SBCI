"""Readers and writers for the two WP1 file forms."""

from .cifti import read_cifti, write_cifti
from .hdf5 import read_hdf5, write_hdf5

__all__ = ["read_hdf5", "write_hdf5", "read_cifti", "write_cifti"]
