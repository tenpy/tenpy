"""The contents of this module have been moved to :mod:`tenpy.algorithms.mps_common`.

This module is just around for backwards compatibility.

.. deprecated :: 0.7.0
    This module is just around for backwards compatibility.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import warnings

warnings.warn(
    "The module `tenpy.algorithms.mps_sweeps` is deprecated;\n"
    "all content is in `tenpy.algoriths.mps_common`", FutureWarning)

from .mps_common import *
