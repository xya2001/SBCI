"""Exceptions this package raises.

Every error about a *file or its contents* derives from :class:`SbciError`, so
a caller can wrap a whole pipeline in one ``except`` and still tell a bad input
from a bug in their own script -- a mistake in the arguments you pass still
raises a plain :class:`ValueError` or :class:`TypeError`::

    try:
        cc = sbci.load(path)
    except sbci.SbciError as exc:
        print(f"{path} is not usable: {exc}")

Each also derives from the built-in exception a reader would reach for first --
:class:`FormatError` is a :class:`KeyError`, :class:`InvalidFileError` and
:class:`MetadataError` are :class:`ValueError` -- so code written against the
built-ins keeps working.
The built-ins are the compatibility promise; :class:`SbciError` is the one to
catch in new code.
"""

from __future__ import annotations


class SbciError(Exception):
    """Base class for every error this package raises deliberately."""


class FormatError(SbciError, KeyError):
    """A file does not have the structure the spec requires.

    Derives from :class:`KeyError` because the usual cause is a missing
    dataset, but prints its message plainly rather than with the repr quotes
    :class:`KeyError` would add.

    Examples
    --------
    >>> raise FormatError("sub-01.h5 has no /connectivity dataset")
    Traceback (most recent call last):
        ...
    sbci.errors.FormatError: sub-01.h5 has no /connectivity dataset
    """

    def __str__(self) -> str:
        return " ".join(str(arg) for arg in self.args)


class InvalidFileError(SbciError, ValueError):
    """A file is not a connectome this package can read.

    The wrong kind of file, or one whose arrays disagree with each other.
    Distinct from :class:`FormatError`, which means a file of the right kind
    is missing a piece; this one means the file is not of the right kind at
    all. Derives from :class:`ValueError`, which is what these sites raised
    before the hierarchy existed.
    """


class MissingDataError(SbciError, ValueError):
    """An operation needs data this connectome does not carry.

    Raised where the file is valid but incomplete for what was asked -- for
    instance re-smoothing a connectome saved without its streamline endpoints.
    """
