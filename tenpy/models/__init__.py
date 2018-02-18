"""Definition of the various models.

A general `model` represents an Hamiltonian and the Hilbert space it lives on.
As such, it consits of

1) the underlying lattice defining the local Hilbert space and onsite-operators
2) couplings between two (or more) sites.

MPS/MPO based algorithms like DMRG require to map this general model to a 1D-model with (possibly)
long range order.
"""
# Copyright 2018 TeNPy Developers

from . import lattice, model

__all__ = ['lattice', 'model']
