"""The contents of this module have been moved to :mod:`tenpy.models.fermions_spinless`.

.. deprecated :: 0.4.1
    This module is just around for backwards compatibility.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from .fermions_spinless import FermionModel, FermionChain

import warnings

__all__ = ['FermionModel', 'FermionChain']

msg = """The module `tenpy.models.fermion_chain` is deprecated now.
Import the model classes from `tenpy.models.fermions_spinless`."""
warnings.warn(msg, FutureWarning)
