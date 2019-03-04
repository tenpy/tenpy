"""The contents of this module have been moved to :mod:`tenpy.models.bose_hubbard`.

This module is just around for backwards compatibility."""
# Copyright 2018 TeNPy Developers

from .bose_hubbard import BoseHubbardModel, BoseHubbardChain

import warnings

msg = """BUGFIX
***********
* WARNING for a bugfix:
* The Hamiltonian of the `BoseHubbardModel` was previously implemented with $ U n_i^2 $ as interaction term,
* but documented as $ U n_i (n_i-1)$. Now it is implemented as the latter as well.
***********
To avoid this warning, simply import the model class from `tenpy.models.bose_hubbard` instead of `tenpy.models.bose_hubbard_chain`."""
warnings.warn(msg)
warnings.warn("The module `tenpy.models.bose_hubbard_chain` is deprecated now.", FutureWarning)
