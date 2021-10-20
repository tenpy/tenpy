"""The contents of this module have been moved to :mod:`tenpy.models.hubbard`.

.. deprecated :: 0.4.1
    This module is just around for backwards compatibility.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from .hubbard import BoseHubbardModel, BoseHubbardChain

import warnings

__all__ = ['BoseHubbardModel', 'BoseHubbardChain']

msg = """RESTRUCTURING
***********
* WARNING:
* The signs of hopping and chemical potential parameters were changed to the usual conventions!
* Moreover, "bose_hubbard.py" and "fermions_hubbard.py" models have now been consolidated into "hubbard.py".
***********
To avoid this warning, simply import the model class from `tenpy.models.hubbard` instead of `tenpy.models.bose_hubbard`."""
warnings.warn(msg)
warnings.warn("The module `tenpy.models.bose_hubbard` is deprecated now.", FutureWarning)
