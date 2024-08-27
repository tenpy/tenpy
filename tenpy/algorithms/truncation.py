"""Dummy module for backwards compatibility.

The `tenpy.algorithms.truncation` module was moved to :mod:`tenpy.linalg.truncation`.
Here we just import all the names again for backwards compatibility,
to support loading pickle and HDF5 data files with `TruncationError` instances in them.
"""

from ..linalg.truncation import *
