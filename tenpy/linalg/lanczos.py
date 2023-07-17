"""Lanczos algorithm for np_conserved arrays."""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3
import warnings
from .krylov_based import *

__all__ = [
    'KrylovBased', 'Arnoldi', 'LanczosGroundState', 'LanczosEvolution', 'lanczos',
    'lanczos_arpack', 'gram_schmidt', 'plot_stats'
]

warnings.warn('The tenpy.linalg.lanczos module has been renamed to tenpy.linalg.krylov_based and ' \
              'will be removed in v1.0')


def lanczos(H, psi, options={}, orthogonal_to=[]):
    """Simple wrapper calling ``LanczosGroundState(H, psi, options, orthogonal_to).run()``

    .. deprecated :: 0.6.0
        Going to remove the `orthogonal_to` argument.
        Instead, replace H with `OrthogonalNpcLinearOperator(H, orthogonal_to)`
        using the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`.

    .. deprecated :: 1.0.0
        Going to remove the `lanczos` function.
        Use :class:`~tenpy.linalg.krylov_based.LanczosGroundState` instead.

    Parameters
    ----------
    H, psi, options, orthogonal_to :
        See :class:`LanczosGroundState`.

    Returns
    -------
    E0, psi0, N :
        See :meth:`LanczosGroundState.run`.
    """
    warnings.warn('The lanczos function will be deprecated in favor of the LanczosGroundState ' \
                  'class in v1.0')
    return LanczosGroundState(H, psi, options, orthogonal_to).run()
