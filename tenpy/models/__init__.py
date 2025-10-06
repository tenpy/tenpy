"""Definition of the various models.

For an introduction to models see :doc:`/intro/model`.

The module :mod:`tenpy.models.model` contains base classes for models.
The module :mod:`tenpy.models.lattice` contains base classes and implementations of lattices.
All other modules in this folder contain model classes derived from these base classes.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    lattice
    model

.. rubric:: Specific models

.. autosummary::
    :toctree: .

    tf_ising
    xxz_chain
    spins
    spins_nnn
    fermions_spinless
    hubbard
    tj_model
    aklt
    hofstadter
    haldane
    molecular
    toric_code
    mixed_xk
    clock
    pxp
"""
# Copyright (C) TeNPy Developers, Apache license

from . import lattice, model
from . import tf_ising, xxz_chain, spins, spins_nnn
from . import fermions_spinless, hubbard, tj_model, hofstadter, haldane, molecular
from . import toric_code, aklt, mixed_xk, clock
from .lattice import *
from .model import *
from .tf_ising import *
from .xxz_chain import *
from .spins import *
from .spins_nnn import *
from .fermions_spinless import *
from .hubbard import *
from .tj_model import *
from .hofstadter import *
from .haldane import *
from .molecular import *
from .toric_code import *
from .aklt import *
from .mixed_xk import *
from .clock import *
from .pxp import *

__all__ = [
    *lattice.__all__,
    *model.__all__,
    *tf_ising.__all__,
    *xxz_chain.__all__,
    *spins.__all__,
    *spins_nnn.__all__,
    *fermions_spinless.__all__,
    *hubbard.__all__,
    *tj_model.__all__,
    *hofstadter.__all__,
    *haldane.__all__,
    *molecular.__all__,
    *toric_code.__all__,
    *aklt.__all__,
    *mixed_xk.__all__,
    *clock.__all__,
    *pxp.__all__,
]
