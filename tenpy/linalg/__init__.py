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

from . import symmetries, backends, tensors, random_matrix, sparse, matrix_operations
from .symmetries import *
from .backends import *
from .tensors import *
from .random_matrix import *
from .sparse import *
from .matrix_operations import *

__all__ = ['symmetries', 'backends', 'tensors', 'random_matrix', 'sparse', 'matrix_operations',
           *symmetries.__all__,
           *backends.__all__,
           *tensors.__all__,
           *random_matrix.__all__,
           *sparse.__all__,
           *matrix_operations.__all__,
           ]
