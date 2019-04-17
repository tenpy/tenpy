r"""Linear-algebra tools for tensor networks.

Most notably is the module :mod:`~tenpy.linalg.np_conserved`,
which contains everything needed to make use of
charge conservervation in the context of tensor networks.

Relevant contents of :mod:`~tenpy.linalg.charges`
are imported to :mod:`~tenpy.linalg.np_conserved`,
so you propably won't need to import `charges` directly.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    np_conserved
    charges
    svd_robust
    random_matrix
    sparse
    lanczos

"""
# Copyright 2018 TeNPy Developers

from . import charges, np_conserved, lanczos, random_matrix, sparse, svd_robust

__all__ = ['charges', 'np_conserved', 'lanczos', 'random_matrix', 'sparse', 'svd_robust']

from ..tools import optimization


def _patch_cython():
    # "monkey patch" some objects to avoid cyclic import structure
    from . import _npc_helper
    _npc_helper._charges = charges
    _npc_helper._np_conserved = np_conserved
    assert _npc_helper.QTYPE == charges.QTYPE
    # check types
    warn = False
    import numpy as np
    check_types = [
        (np.float64, np.complex128),
        (np.ones([1]).dtype, (1.j * np.ones([1])).dtype),
        (np.array(1.).dtype, np.array(1.j).dtype),
        (np.array(1., dtype=np.float).dtype, np.array(1., dtype=np.complex).dtype),
    ]
    types_ok = [
        _npc_helper._float_complex_are_64_bit(dt_float, dt_real)
        for dt_float, dt_real in check_types
    ]
    if not np.all(types_ok):
        import warnings
        warnings.warn("(Some of) the default dtypes are not 64-bit. "
                      "Using the compiled cython code (as you do) might make it slower.")
    # done


assert optimization.have_cython_functions is not None  # check that we had a `@use_cython` somewhere

if optimization.have_cython_functions:
    _patch_cython()
