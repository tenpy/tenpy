"""The contents of this module have been moved to :mod:`tenpy.models.fermions_spinless`.

This module is just around for backwards compatibility."""
# Copyright 2018 TeNPy Developers

from .fermions_spinless import FermionModel, FermionChain

import warnings

msg = """The module `tenpy.models.fermion_chain` is deprecated now.
Import the model classes from `tenpy.models.fermions_spinless`."""
warnings.warn(msg, FutureWarning)
