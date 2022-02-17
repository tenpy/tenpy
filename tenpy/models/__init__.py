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
    aklt
    hofstadter
    haldane
    toric_code
    mixed_xk
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from . import lattice, model
from . import tf_ising, xxz_chain, spins, spins_nnn
from . import fermions_spinless, hubbard, hofstadter, haldane
from . import toric_code, aklt, mixed_xk

__all__ = [
    'lattice', 'model', 'tf_ising', 'xxz_chain', 'spins', 'spins_nnn', 'fermions_spinless',
    'hubbard', 'hofstadter', 'haldane', 'toric_code', 'aklt', 'mixed_xk'
]
