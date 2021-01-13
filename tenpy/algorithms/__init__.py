"""A collection of algorithms such as TEBD and DMRG.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    algorithm
    truncation
    tebd
    mps_common
    dmrg
    tdvp
    purification
    mpo_evolution
    network_contractor
    exact_diag
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from . import algorithm, truncation, dmrg, disentangler, mps_common, tebd, tdvp, exact_diag, \
    purification, network_contractor, mpo_evolution

__all__ = [
    "algorithm",
    "truncation",
    "dmrg",
    "mps_common",
    "tebd",
    "tdvp",
    "exact_diag",
    "purification",
    "network_contractor",
    "mpo_evolution",
    "disentangler",
]
