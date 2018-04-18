r"""Provide some random matrix ensembles for numpy.

The implemented ensembles are:

=========== ======================== ======================= ================== ===========
ensemble    matrix class drawn from  measure                 invariant under    beta
=========== ======================== ======================= ================== ===========
GOE         real, symmetric          ``~ exp(-n/4 tr(H^2))`` orthogonal O       1
----------- ------------------------ ----------------------- ------------------ -----------
GUE         hermitian                ``~ exp(-n/2 tr(H^2))`` unitary U          2
----------- ------------------------ ----------------------- ------------------ -----------
CRE         O(n)                     Haar                    orthogonal O       /
----------- ------------------------ ----------------------- ------------------ -----------
COE         U in U(n) with U = U^T   Haar                    orthogonal O       1
----------- ------------------------ ----------------------- ------------------ -----------
CUE         U(n)                     Haar                    unitary U          2
----------- ------------------------ ----------------------- ------------------ -----------
O_close_1   O(n)                     ?                       /                  /
----------- ------------------------ ----------------------- ------------------ -----------
U_close_1   U(n)                     ?                       /                  /
=========== ======================== ======================= ================== ===========


All functions in this module take a tuple ``(n, n)`` as first argument, such that
we can use the function :meth:`~tenpy.linalg.np_conserved.Array.from_func`
to generate a block diagonal :class:`~tenpy.linalg.np_conserved.Array` with the block from the
corresponding ensemble, for example::

    npc.Array.from_func_square(GOE, [leg, leg.conj()])

"""
# Copyright 2018 TeNPy Developers

import numpy as np

__all__ = [
    'box', 'standard_normal_complex', 'GOE', 'GUE', 'CRE', 'COE', 'CUE', 'O_close_1', 'U_close_1'
]


def box(size, W=1.):
    """return random number uniform in (-W, W]."""
    return (0.5 - np.random.random(size)) * (2. * W)


def standard_normal_complex(size):
    """return ``(R + 1.j*I)`` for independent `R` and `I` from np.random.standard_normal."""
    return np.random.standard_normal(size) + 1.j * np.random.standard_normal(size)


def GOE(size):
    r"""Gaussian orthogonal ensemble (GOE).

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.

    Returns
    -------
    H : ndarray
        Real, symmetric numpy matrix drawn from the GOE, i.e.
        :math:`p(H) = 1/Z exp(-n/4 tr(H^2))`
    """
    A = np.random.standard_normal(size)
    return (A + A.T) * 0.5


def GUE(size):
    r"""Gaussian unitary ensemble (GUE).

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.

    Returns
    -------
    H : ndarray
        Hermitian (complex) numpy matrix drawn from the GUE, i.e.
        :math:`p(H) = 1/Z exp(-n/4 tr(H^2))`.
    """
    A = standard_normal_complex(size)
    return A + A.T.conj()


def CRE(size):
    r"""Circular real ensemble (CRE).

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.

    Returns
    -------
    U : ndarray
        Orthogonal matrix drawn from the CRE (=Haar measure on O(n)).
    """
    # almost same code as for COE
    n, m = size
    assert n == m  # ensure that `mode` in qr doesn't matter.
    A = np.random.standard_normal(size)
    Q, R = np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = np.diagonal(R)
    Q *= L / abs(L)  # no need to construct the diagonal matrix explicitly, just scale last axis.
    return Q


def COE(size):
    r"""Circular orthogonal ensemble (COE).

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.

    Returns
    -------
    U : ndarray
        Unitary, symmetric (complex) matrix drawn from the COE (=Haar measure on this space).
    """
    U = CUE(size)
    return np.dot(U.T, U)


def CUE(size):
    r"""Circular unitary ensemble (CUE).

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.

    Returns
    -------
    U : ndarray
        Unitary matrix drawn from the CUE (=Haar measure on U(n)).
    """
    # almost same code as for CRE
    n, m = size
    assert n == m  # ensure that `mode` in qr doesn't matter.
    A = standard_normal_complex(size)
    Q, R = np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = np.diagonal(R)
    Q *= L / abs(L)  # no need to construct the diagonal matrix explicitly, just scale last axis.
    return Q


def O_close_1(size, a=0.01):
    r"""return an random orthogonal matrix 'close' to the Identity.

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.
    a : float
        Parameter determining how close the result is on `O`;
        :math:`\lim_{a \rightarrow 0} <|O-E|>_a = 0`` (where `E` is the identity).

    Returns
    -------
    O : ndarray
        Orthogonal matrix close to the identiy (for small `a`).
    """
    n, m = size
    assert n == m
    A = GOE(size) / (2. * n)**0.5  # scale such that eigenvalues are in [-1, 1]
    E = np.eye(size[0])
    Q, R = np.linalg.qr(E + a * A)
    L = np.diagonal(R)  # make QR decomposition unique & ensure Q is close to one for small `a`
    Q *= L / abs(L)
    return Q


def U_close_1(size, a=0.01):
    r"""return an random orthogonal matrix 'close' to the identity.

    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.
    a : float
        Parameter determining how close the result is to the identity.
        :math:`\lim_{a \rightarrow 0} <|O-E|>_a = 0`` (where `E` is the identity).

    Returns
    -------
    U : ndarray
        Unitary matrix close to the identiy (for small `a`).
        Eigenvalues are chosen i.i.d. as ``exp(1.j*a*x)`` with `x` uniform in [-1, 1].
    """
    n, m = size
    assert n == m
    U = CUE(size)  # random unitary
    E = np.exp(1.j * a * (np.random.rand(n) * 2. - 1.))
    return np.dot(U * E, U.T.conj())
