"""Definition of the various models.

For an introduction to models see :doc:`/intro_model`.

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
    bose_hubbard
    fermions_hubbard
    hofstadter
    toric_code
"""
# Copyright 2018 TeNPy Developers

from . import lattice, model

__all__ = ['lattice', 'model']
