"""Dummy module for backwards compatibility.

The `tenpy.algorithms.truncation` module was moved to :mod:`tenpy.linalg.truncation`.
Here we just import all the names again for backwards compatibility,
to support loading pickle and HDF5 data files with `TruncationError` instances in them.
"""
# Copyright (C) TeNPy Developers, Apache license

# just provide namespace from ..linalg.truncation

from ..linalg import truncation
from ..linalg.truncation import (  # noqa: F401
    TruncationError,
    decompose_theta_qr_based,
    svd_theta,
    truncate,
)

__all__ = truncation.__all__
