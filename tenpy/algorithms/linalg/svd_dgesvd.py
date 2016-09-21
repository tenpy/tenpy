# Code based on jgarcke
# http://projects.scipy.org/numpy/ticket/990
# - MZ
import numpy as np
import ctypes
from ctypes import CDLL, POINTER, c_int, byref, c_char, c_double

from numpy.core import array, asarray, zeros, empty, transpose, \
    intc, single, double, csingle, cdouble, inexact, complexfloating, \
    newaxis, ravel, all, Inf, dot, add, multiply, identity, sqrt, \
    maximum, flatnonzero, diagonal, arange, fastCopyAndTranspose, sum, \
    isfinite, size
libs = ["libLAPACK.dylib", "libmkl_rt.so", "libmkl_intel_lp64.so", "liblapack.so",
        "libopenblas.dll"]

lib = None
for l in libs:
    try:
        lib = CDLL(l)
        print "Loaded " + l + " for dgesvd"
        break
    except OSError:
        pass

if lib == None:
    raise OSError, "Couldn't find lapack library for GESVD patch"


def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap


def isComplexType(t):
    return issubclass(t, complexfloating)


_real_types_map = {single: single, double: double, csingle: single, cdouble: double}

_complex_types_map = {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}


def _realType(t, default=double):
    return _real_types_map.get(t, default)


def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)


def _linalgRealType(t):
    """Cast the type t to either double or cdouble."""
    return double


_complex_types_map = {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}


def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" % (a.dtype.name, ))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type

# _fastCopyAndTranpose assumes the input is 2D (as all the calls in here are).

_fastCT = fastCopyAndTranspose


def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is type:
            cast_arrays = cast_arrays + (_fastCT(a), )
        else:
            cast_arrays = cast_arrays + (_fastCT(a.astype(type)), )
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays


def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError, '%d-dimensional array given. Array must be \
            two-dimensional' % len(a.shape)


def _assertSquareness(*arrays):
    for a in arrays:
        if max(a.shape) != min(a.shape):
            raise LinAlgError, 'Array must be square'


def _assertFinite(*arrays):
    for a in arrays:
        if not (isfinite(a).all()):
            raise LinAlgError, "Array must not contain infs or NaNs"


def _assertNonEmpty(*arrays):
    for a in arrays:
        if size(a) == 0:
            raise LinAlgError("Arrays cannot be empty")

# Shorthand data type for type checking
dbl_arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
dbl_2_arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2)

lib.dgesvd_.argtypes = [POINTER(c_char), POINTER(c_char), POINTER(c_int), POINTER(c_int),
                        dbl_2_arr, POINTER(c_int), dbl_arr, dbl_2_arr, POINTER(c_int), dbl_2_arr,
                        POINTER(c_int), dbl_arr, POINTER(c_int), POINTER(c_int)]


def svd_dgesvd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and ``Vh``,
    and a 1-dimensional array of singular values, ``s`` (real, non-negative),
    such that ``a == U S Vh``, where ``S`` is the diagonal
    matrix ``np.diag(s)``.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to decompose
    full_matrices : boolean, optional
        If True (default), ``U`` and ``Vh`` are shaped
        ``(M,M)`` and ``(N,N)``.  Otherwise, the shapes are
        ``(M,K)`` and ``(K,N)``, where ``K = min(M,N)``.
    compute_uv : boolean
        Whether to compute ``U`` and ``Vh`` in addition to ``s``.
        True by default.

    Returns
    -------
    U : ndarray, shape (M, M) or (M, K) depending on `full_matrices`
        Unitary matrix.
    s :  ndarray, shape (K,) where ``K = min(M, N)``
        The singular values, sorted so that ``s[i] >= s[i+1]``.
    Vh : ndarray, shape (N,N) or (K,N) depending on `full_matrices`
        Unitary matrix.

    Raises
    ------
    LinAlgError
        If SVD computation fails. 
        For details see dgesvd.f and dbdsqr.f of LAPACK
    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertNonEmpty(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    s = zeros((min(n, m), ), real_t)

    if compute_uv:
        if full_matrices:
            nu = m
            nvt = n
            option = 'A'
        else:
            nu = min(n, m)
            nvt = min(n, m)
            option = 'S'
        u = zeros((nu, m), t)
        vt = zeros((n, nvt), t)
    else:
        option = 'N'
        nu = 1
        nvt = 1
        u = empty((1, 1), t)
        vt = empty((1, 1), t)

    lapack_routine = lib.dgesvd_

    lwork = 1
    work = zeros((lwork, ), t)
    INFO = c_int(0)
    m = c_int(m)
    n = c_int(n)
    nvt = c_int(nvt)
    lwork = c_int(-1)
    print a.shape, a.dtype
    lapack_routine(option, option, m, n, a, m, s, u, m, vt, nvt, work, lwork, INFO)
    if INFO.value < 0:
        raise Exception('%d-th argument had an illegal value' % INFO.value)

    lwork = int(work[0])
    work = zeros((lwork, ), t)
    lwork = c_int(lwork)
    lapack_routine(option, option, m, n, a, m, s, u, m, vt, nvt, work, lwork, INFO)
    if INFO.value > 0:
        raise Exception('Error during factorization: %d' % INFO.value)
#        raise LinAlgError, 'SVD did not converge'
    s = s.astype(_realType(result_t))
    if compute_uv:
        u = u.transpose().astype(result_t)
        vt = vt.transpose().astype(result_t)
        return wrap(u), s, wrap(vt)
    else:
        return s
