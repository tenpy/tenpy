r"""Linear-algebra tools for tensor networks.

Most notably is the module :mod:`~tenpy.linalg.tensors`,
which provides the :class:`~tenpy.linalg.tensors.Tensor` class used by the rest of the library.


.. rubric:: Submodules

.. autosummary::
    :toctree: .

    groups
    spaces
    backends
    tensors
    random_matrix
    sparse
    matrix_operations
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

from . import (groups, spaces, backends, tensors, random_matrix, sparse,
               matrix_operations, krylov_based, trees)
from .groups import *
from .trees import *
from .spaces import *
from .backends import *
from .tensors import *
from .random_matrix import *
from .sparse import *
from .matrix_operations import *
from .krylov_based import *

__all__ = ['groups', 'spaces', 'trees', 'backends', 'tensors', 'random_matrix', 'sparse',
           'matrix_operations', 'krylov_based',
           *groups.__all__,
           *trees.__all__,
           *spaces.__all__,
           *backends.__all__,
           *tensors.__all__,
           *random_matrix.__all__,
           *sparse.__all__,
           *matrix_operations.__all__,
           *krylov_based.__all__,
           ]
