"""Definition of the various models.

For an introduction to models see :doc:`/intro_model`.

The module :mod:`tenpy.models.model` contains base classes for models.
The module :mod:`tenpy.models.lattice` contains base classes and implementations of lattices.
All other modules in this folder contain model classed derived from these base classes.
"""
# Copyright 2018 TeNPy Developers

from . import lattice, model

__all__ = ['lattice', 'model']
