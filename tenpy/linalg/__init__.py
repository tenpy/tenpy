r"""Linear-algebra tools for tensor networks.

Most notably is the module :mod:`~tenpy.linalg.tensors`,
which provides the :class:`~tenpy.linalg.tensors.Tensor` class used by the rest of the library.


.. rubric:: Submodules

.. autosummary::
    :toctree: .

    tensors
    symmetries
    random_matrix
    sparse
    krylov_based
    old
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

from . import old
from . import symmetries, backends, tensors, random_matrix, sparse, charges, matrix_operations
from ..tools import optimization
from .symmetries import *
from .backends import *
from .tensors import *
from .random_matrix import *
from .sparse import *
from .matrix_operations import *

__all__ = ['old', 'symmetries', 'backends', 'tensors', 'random_matrix', 'sparse', 'charges',
           'matrix_operations',
           *symmetries.__all__,
           *backends.__all__,
           *tensors.__all__,
           *random_matrix.__all__,
           *sparse.__all__,
           *matrix_operations.__all__,
           ]


def _patch_cython():
    # "monkey patch" some objects to avoid cyclic import structure
    from .old import _npc_helper, charges, np_conserved
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
        (np.array(1., dtype=np.float64).dtype, np.array(1., dtype=np.complex128).dtype),
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


#  assert optimization.have_cython_functions is not None  # check that we had a `@use_cython` somewhere

if optimization.have_cython_functions:
    _patch_cython()
