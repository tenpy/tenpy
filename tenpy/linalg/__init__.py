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

from ..tools import optimization

if optimization.have_cython_functions:
    # "monkey patch" some objects to avoid cyclic import structure
    from . import _npc_helper
    _npc_helper._charges = charges
    _npc_helper._np_conserved = np_conserved
