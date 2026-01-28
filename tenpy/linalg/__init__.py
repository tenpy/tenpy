r"""Linear-algebra tools for tensor networks.

Most notably is the module :mod:`~tenpy.linalg.np_conserved`,
which contains everything needed to make use of
charge conservation in the context of tensor networks.

Relevant contents of :mod:`~tenpy.linalg.charges`
are imported to :mod:`~tenpy.linalg.np_conserved`,
so you probably won't need to import `charges` directly.

.. rubric:: Submodules

.. autosummary::
    :toctree: .

    np_conserved
    charges
    svd_robust
    random_matrix
    sparse
    krylov_based
    truncation

"""
# Copyright (C) TeNPy Developers, Apache license

from . import truncation
from .truncation import TruncationError
