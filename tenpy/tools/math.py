"""Different math functions needed at some point in the library."""
# Copyright 2018 TeNPy Developers

import numpy as np
import warnings
from . import misc

from scipy.sparse.linalg import eigs as sparse_eigen

int_I3 = np.eye(3, dtype=int)
LeviCivita3 = np.array([[np.cross(b, a) for a in int_I3] for b in int_I3])


def matvec_to_array(H):
    """transform an linear operator with a `matvec` method into a dense numpy array.

    Parameters
    ----------
    H : linear operator
        should have `dim`, `dtype` attributes and a `matvec` method.

    Returns
    -------
    H_dense : ndarray, shape ``(H.dim, H.dim)``
        a dense array version of `H`.
    """
    dim, dim2 = H.shape
    assert (dim == dim2)
    X = np.zeros((dim, dim), H.dtype)
    v = np.zeros((dim), H.dtype)
    for i in range(dim):
        v[i] = 1
        X[i] = H.matvec(v)
        v[i] = 0
    return X


##########################################################################
##########################################################################
# Actual Math functions


def entropy(p, n=1):
    r"""Calculate the entropy of a distribution.

    Assumes that p is a normalized distribution (``np.sum(p)==1.``).

    Parameters
    ----------
    p : 1D array
        A normalized distribution.
    n : 1 | float | np.inf
        Selects the entropy, see below.

    Returns
    -------
    entropy : float
        Shannon-entropy :math:`-\sum_i p_i \log(p_i)` (n=1) or
        Renyi-entropy :math:`\frac{1}{1-n} \log(\sum_i p_i^n)` (n != 1)
        of the distribution `p`.
    """
    p = p[p > 1.e-30]  # just for stability reasons / to avoid NaN in log
    if n == 1:
        return -np.inner(np.log(p), p)
    elif n == np.inf:
        return -np.log(np.max(p))
    else:  # general n != 1, inf
        return np.log(np.sum(p**n)) / (1. - n)


def gcd(a, b):
    """Computes the greatest common divisor (GCD) of two numbers.
    Return 0 if both a, b are zero, otherwise always return a non-negative number."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def gcd_array(a):
    """Return the greatest common divisor of all of entries in `a`"""
    a = np.array(a).reshape(-1)
    if len(a) <= 0:
        raise ValueError
    t = a[0]
    for x in a[1:]:
        if t == 1:
            break
        t = gcd(t, x)
    return t


def lcm(a, b):
    """Returns the least common multiple (LCM) of two positive numbers."""
    a0, b0 = a, b
    while b:
        a, b = b, a % b
    return a0 * (b0 // a)


def speigs(A, k, *args, **kwargs):
    """Wrapper around :func:`scipy.sparse.linalg.eigs`, lifting the restriction ``k < rank(A)-1``.

    Parameters
    ----------
    A : MxM ndarray or like :class:`scipy.sparse.linalg.LinearOperator`
        the (square) linear operator for which the eigenvalues should be computed.
    k : int
        the number of eigenvalues to be computed.
    *args, **kwargs :
        further arguments are directly given to :func:`scipy.sparse.linalg.eigs`

    Returns
    -------
    w : ndarray
        array of min(`k`, A.shape[0]) eigenvalues
    v : ndarray
        array of min(`k`, A.shape[0]) eigenvectors, ``v[:, i]`` is the `i`-th eigenvector.
        Only returned if ``kwargs['return_eigenvectors'] == True``.
    """
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError("A.shape not a square matrix: " + str(A.shape))
    if k < d - 1:
        return sparse_eigen(A, k, *args, **kwargs)
    else:
        if k > d:
            warnings.warn("trimming speigs k to smaller matrix dimension d", stacklevel=2)
            k = d
        if isinstance(A, np.ndarray):
            Amat = A
        else:
            Amat = matvec_to_array(A)  # Constructs the matrix
        ret_eigv = kwargs.get('return_eigenvectors', args[7] if len(args) > 7 else True)
        which = kwargs.get('which', args[2] if len(args) > 2 else 'LM')
        if ret_eigv:
            W, V = np.linalg.eig(Amat)
            keep = misc.argsort(W, which)[:k]
            return W[keep], V[:, keep]
        else:
            W = np.linalg.eigvals(Amat)
            keep = misc.argsort(W, which)[:k]
            return W


def perm_sign(p):
    """ Given a permutation `p` of numbers, returns its sign. (+1 or -1)

    Assumes that all the elements are distinct, if not, you get crap.

    Examples
    --------
    >>> for p in itertools.permutations(range(3))]):
    ...      print('{p!s}: {sign!s}'.format(p=p, sign=perm_sign(p)))
    (0, 1, 2): 1
    (0, 2, 1): -1
    (1, 0, 2): -1
    (1, 2, 0): 1
    (2, 0, 1): 1
    (2, 1, 0): -1
    """
    rp = np.argsort(p)
    p = np.argsort(rp)
    s = 1
    for i, v in enumerate(p):
        if i == v:
            continue
        # by the way we loop, i < v, so we find where i is.
        # p[i] = p[rp[i]] # we don't have to do that becasue we never need p[i] again
        p[rp[i]] = v
        rp[v] = rp[i]
        s = -s
    return s
