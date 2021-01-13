"""The contents of this module have been moved.

`PurificationTEBD` and `PurificationTEBD2` are now in :mod:`tenpy.algorithms.purification`,
and the various disentanglers are in :mod:`tenpy.algorithms.disentangler`

.. deprecated :: 0.7.1
    This module is just around for backwards compatibility.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from .purification import PurificationTEBD, PurificationTEBD2
from .disentangler import *

import warnings

msg = ("The module `tenpy.algorithms.purification_tebd` has been replaced by "
       "`tenpy.algorithms.purification`; simply import from there to avoid this warning.")
warnings.warn(msg, FutureWarning)
