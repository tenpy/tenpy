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
    vumps
    plane_wave_excitation
    network_contractor
    exact_diag
    disentangler
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from . import algorithm, truncation, dmrg, dmrg_parallel, disentangler, mps_common, tebd, tdvp, \
    exact_diag, purification, network_contractor, mpo_evolution, vumps, plane_wave_excitation
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
from .vumps import *
from .plane_wave_excitation import *


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
    "vumps",
    "plane_wave_excitation",
    *algorithm.__all__,
    *truncation.__all__,
    *[n for n in dmrg.__all__ if n not in ['Mixer', 'SubspaceExpansion', 'DensityMatrixMixer']],
    *dmrg_parallel.__all__,
    *disentangler.__all__,
    *mps_common.__all__,
    *[n for n in tebd.__all__ if n != 'Engine'],
    *[n for n in tdvp.__all__ if n != 'Engine'],
    *exact_diag.__all__,
    *purification.__all__,
    *network_contractor.__all__,
    *mpo_evolution.__all__,
    *vumps.__all__,
    *plane_wave_excitation.__all__,
]
