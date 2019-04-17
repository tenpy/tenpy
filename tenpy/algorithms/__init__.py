"""A collection of algorithms such as TEBD and DMRG.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    truncation
    dmrg
    tebd
    tdvp
    purification_tebd
    network_contractor
    exact_diag

"""
# Copyright 2018 TeNPy Developers

from . import truncation, dmrg, tebd, tdvp, exact_diag, purification_tebd, network_contractor

__all__ = [
    "truncation", "dmrg", "tebd", "tdvp", "exact_diag", "purification_tebd", "network_contractor"
]
