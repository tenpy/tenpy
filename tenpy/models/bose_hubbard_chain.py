"""The contents of this module have been moved to :mod:`tenpy.models.hubbard`.

This module is just around for backwards compatibility.

.. deprecated :: 0.4.1
    This module is just around for backwards compatibility.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from .hubbard import BoseHubbardModel, BoseHubbardChain

import warnings

__all__ = ['BoseHubbardModel', 'BoseHubbardChain']

msg = """BUGFIX
***********
* WARNING for a bugfix:
* The Hamiltonian of the `BoseHubbardModel` was previously implemented with $ U n_i^2 $ as interaction term,
* but documented as $ U n_i (n_i-1)$. Now it is implemented as the latter as well.
***********
To avoid this warning, simply import the model class from `tenpy.models.hubbard` instead of `tenpy.models.bose_hubbard_chain`."""
warnings.warn(msg)
warnings.warn("The module `tenpy.models.bose_hubbard_chain` is deprecated now.", FutureWarning)
