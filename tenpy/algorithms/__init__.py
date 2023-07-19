"""A collection of algorithms such as TEBD and DMRG.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    algorithm
    truncation
    tebd
    mps_common
    dmrg
    dmrg_parallel
    tdvp
    purification
    mpo_evolution
    network_contractor
    exact_diag
    disentangler
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

from . import algorithm, truncation, dmrg, dmrg_parallel, disentangler, mps_common, tebd, tdvp, \
    exact_diag, purification, network_contractor, mpo_evolution
from .algorithm import *
from .disentangler import *
from .dmrg_parallel import *
from .dmrg import *
from .exact_diag import *
from .mpo_evolution import *
from .mps_common import *
from .network_contractor import *
from .purification import *
from .tdvp import *
from .tebd import *
from .truncation import *

__all__ = [
    "algorithm",
    "truncation",
    "dmrg",
    "dmrg_parallel",
    "mps_common",
    "tebd",
    "tdvp",
    "exact_diag",
    "purification",
    "network_contractor",
    "mpo_evolution",
    "disentangler",
    *algorithm.__all__,
    *truncation.__all__,
    *dmrg.__all__,
    *dmrg_parallel.__all__,
    *disentangler.__all__,
    *mps_common.__all__,
    *[n for n in tebd.__all__ if n != 'Engine'],
    *[n for n in tdvp.__all__ if n != 'Engine'],
    *exact_diag.__all__,
    *purification.__all__,
    *network_contractor.__all__,
    *mpo_evolution.__all__,
]
