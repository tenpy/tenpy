"""Adjustments to employ MPI parallelization splitting up the MPO virtual legs.

This module implements parallelization of DMRG with the MPI framework [MPI]_.
It's based on the python interface of `mpi4py <https://mpi4py.readthedocs.io/>`_,
which needs to be installed when you want to use classes in this module.

.. note ::
    This module is not imported by default, since just importing mpi4py already initializes MPI.
    Hence, if you want to use it, you need to explicitly call
    ``import tenpy.mpi_parallel`` in your python script.

.. warning ::
    MPI parallelization is still under active development.


.. rubric:: Submodules

.. autosummary::
    :toctree: .

"""
# Copyright 2021-2023 TeNPy Developers, GNU GPLv3

from . import helpers, actions, distributed, simulation

__all__ = ["helpers", "actions", "distributed", "simulation"]
