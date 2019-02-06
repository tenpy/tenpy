r"""Linear-algebra tools for tensor networks.

Most notably is the module :mod:`~tenpy.linalg.np_conserved`,
which contains everything needed to make use of
charge conservervation in the context of tensor networks.

Relevant contents of :mod:`~tenpy.linalg.charges`
are imported to :mod:`~tenpy.linalg.np_conserved`,
so you propably won't need to import `charges` directly.
"""
# Copyright 2018 TeNPy Developers

from . import charges, np_conserved, lanczos, random_matrix, sparse, svd_robust

__all__ = ['charges', 'np_conserved', 'lanczos', 'random_matrix', 'sparse', 'svd_robust']

try:
    # optimization: "monkey patch" with the optimized versions
    from . import npc_helper
    charges.ChargeInfo = npc_helper.ChargeInfo
    charges.LegCharge = npc_helper.LegCharge
    charges.LegPipe = npc_helper.LegPipe
    charges.QTYPE = npc_helper.QTYPE
    np_conserved.ChargeInfo = npc_helper.ChargeInfo
    np_conserved.LegCharge = npc_helper.LegCharge
    np_conserved.LegPipe = npc_helper.LegPipe
    np_conserved.QTYPE = npc_helper.QTYPE
    npc_helper._charges = charges
    npc_helper._np_conserved = np_conserved
except ImportError:
    np_conserved.ChargeInfo = charges.ChargeInfo
    np_conserved.LegCharge = charges.LegCharge
    np_conserved.LegPipe = charges.LegPipe
