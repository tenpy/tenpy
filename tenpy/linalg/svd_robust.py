r"""(More) robust version of singular value decomposition.

We often need to perform an SVD.
In general, an SVD is a matrix factorization that is always well defined and should also work
for ill-conditioned matrices.
But sadly, both :func:`numpy.linalg.svd` and :func:`scipy.linalg.svd` fail from time to time,
raising ``LinalgError("SVD did not converge")``.
The reason is that both of them call the LAPACK function `#gesdd`
(where `#` depends on the data type), which takes an iterative approach that can fail.
However, it is usually much faster than the alternative (and robust) `#gesvd`.

Our workaround is as follows: we provide a function :func:`svd` with call signature as scipy's svd.
This function is basically just a wrapper around scipy's svd, i.e., we keep calling the faster
`dgesdd`. But if that fails, we can still use `dgesvd` as a backup.

Sadly, `dgesvd` and `zgesvd` were not included into scipy until version '0.18.0' (nor in numpy),
which is as the time of this writing the latest stable scipy version.
For scipy version newer than '0.18.0', we make use of the new keyword 'lapack_driver' for svd,
otherwise we (try to) load `dgesvd` and `zgesvd` from shared LAPACK libraries.

The tribute for the dgesvd wrapper code goes to 'jgarcke', originally posted at
http://projects.scipy.org/numpy/ticket/990, which is now hosted
at https://github.com/numpy/numpy/issues/1588
He explains a bit more in detail what fails.

The include of `dgesvd` to scipy was done in https://github.com/scipy/scipy/pull/5994.


Examples
--------
The idea is that you just import the `svd` from this module and use it as replacement for
``np.linalg.svd`` or ``scipy.linalg.svd``:

>>> from svd_robust import svd
>>> U, S, VT = svd([[1., 1.], [0., [1.]])
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import scipy
import scipy.linalg
import warnings

from ctypes import CDLL, POINTER, c_int, c_char
from ctypes.util import find_library
from numpy.core import single, double, csingle, cdouble  # for those, 'c' = complex

from numpy.linalg.linalg import LinAlgError

try:
    from numpy.linalg.linalg import _makearray, _fastCopyAndTranspose, \
        isComplexType, _realType, _commonType, _assertRank2, _assertFinite, _assertNoEmpty2d
except:
    warnings.warn("Import problems: the work-around `svd_gesvd` will fail.")
    # If you get this warning, you might still be lucky and not need the workaround,
    # for examply, if you have a recent version of scipy....

    # If you can't upgrade your scipy, you might try to copy&paste the corresponding functions
    # from the numpy code (version 1.11.0 works for me) on github.

# check the scipy version wheter it includes LAPACK's 'gesvd'
try:
    # simply check wether scipy.linalg.svd has the keyword it should have
    scipy.linalg.svd([[1.]], lapack_driver='gesvd')
    _old_scipy = False
except TypeError:
    _old_scipy = True
    warnings.warn("Old scipy <= 0.18.0: support will be dropped in TeNPy version 1.0.0",
                  FutureWarning)

# (NumpyVersion parses the argument for version comparsion, not ``numpy.__version__``)

#: will be the the CLAPACK library loaded with _load_lapack()
_lapack_lib = None


def svd(a,
        full_matrices=True,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver='gesdd',
        warn=True):
    """Wrapper around :func:`scipy.linalg.svd` with `gesvd` backup plan.

    Tries to avoid raising an LinAlgError by using using the lapack_driver `gesvd`,
    if `gesdd` failed.

    Parameters
    ----------
    overwrite_a : bool
        Ignored (i.e. set to ``False``) if ``lapack_driver='gesdd'``.
        Otherwise described in :func:`scipy.linalg.svd`.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach (``'gesdd'``)
        or general rectangular approach (``'gesvd'``) to compute the SVD.
        MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.
        If ``'gesdd'`` fails, ``'gesvd'`` is used as backup.
    warn : bool
        whether to create a warning when the SVD failed.

    Other parameters as described in doc-string of :func:`scipy.linalg.svd`

    Returns
    -------
    U, S, Vh : ndarray
        As described in doc-string of :func:`scipy.linalg.svd`
    """
    if lapack_driver == 'gesdd':
        try:
            return scipy.linalg.svd(a, full_matrices, compute_uv, False, check_finite)
        except np.linalg.LinAlgError:
            # 'gesdd' failed to converge, so we continue with the backup plan
            if warn:
                warnings.warn("SVD with lapack_driver 'gesdd' failed. Use backup 'gesvd'",
                              stacklevel=2)
            pass
    if lapack_driver not in ['gesdd', 'gesvd']:
        raise ValueError("invalid `lapack_driver`: " + str(lapack_driver))
    # 'gesvd' lapack driver
    if not _old_scipy:
        # use LAPACK wrapper included in scipy version >= 0.18.0
        return scipy.linalg.svd(a, full_matrices, compute_uv, overwrite_a, check_finite, 'gesvd')
    else:  # for backwards compatibility
        _load_lapack(warn=warn)
        return svd_gesvd(a, full_matrices, compute_uv, check_finite)


def svd_gesvd(a, full_matrices=True, compute_uv=True, check_finite=True):
    """svd with LAPACK's '#gesvd' (with # = d/z for float/complex).

    Similar as :func:`numpy.linalg.svd`, but use LAPACK 'gesvd' driver.
    Works only with 2D arrays.
    Outer part is based on the code of `numpy.linalg.svd`.

    Parameters
    ----------
    a, full_matrices, compute_uv :
        See :func:`numpy.linalg.svd` for details.
    check_finite :
        check whether input arrays contain 'NaN' or 'inf'.

    Returns
    -------
    U, S, Vh : ndarray
        See :func:`numpy.linalg.svd` for details.
    """
    a, wrap = _makearray(a)  # uses order='C'
    _assertNoEmpty2d(a)
    _assertRank2(a)
    if check_finite:
        _assertFinite(a)
    M, N = a.shape
    # determine types
    t, result_t = _commonType(a)
    # t = type for calculation, (for my numpy version) actually always one of {double, cdouble}
    # result_t = one of {single, double, csingle, cdouble}
    is_complex = isComplexType(t)
    real_t = _realType(t)  # real version of t with same precision
    # copy: the array is destroyed
    a = _fastCopyAndTranspose(t, a)  # casts a to t, copy and transpose (=change to order='F')

    lapack_routine = _get_gesvd(t)
    # allocate output space & options
    if compute_uv:
        if full_matrices:
            nu = M
            lvt = N
            option = b'A'
        else:
            nu = min(N, M)
            lvt = min(N, M)
            option = b'S'
        u = np.zeros((M, nu), t, order='F')
        vt = np.zeros((lvt, N), t, order='F')
    else:
        option = b'N'
        nu = 1
        u = np.empty((1, 1), t, order='F')
        vt = np.empty((1, 1), t, order='F')
    s = np.zeros((min(N, M), ), real_t, order='F')
    INFO = c_int(0)
    m = c_int(M)
    n = c_int(N)
    lu = c_int(u.shape[0])
    lvt = c_int(vt.shape[0])
    work = np.zeros((1, ), t)
    lwork = c_int(-1)  # first call with lwork=-1
    args = [option, option, m, n, a, m, s, u, lu, vt, lvt, work, lwork, INFO]
    if is_complex:
        # differnt call signature: additional array 'rwork' of fixed size
        rwork = np.zeros((5 * min(N, M), ), real_t)
        args.insert(-1, rwork)
    lapack_routine(*args)  # first call: just calculate the required `work` size
    if INFO.value < 0:
        raise Exception('%d-th argument had an illegal value' % INFO.value)
    if is_complex:
        lwork = int(work[0].real)
    else:
        lwork = int(work[0])
    work = np.zeros((lwork, ), t, order='F')
    args[11] = work
    args[12] = c_int(lwork)

    lapack_routine(*args)  # second call: the actual calculation

    if INFO.value < 0:
        raise Exception('%d-th argument had an illegal value' % INFO.value)
    if INFO.value > 0:
        raise LinAlgError("SVD did not converge with 'gesvd'")
    s = s.astype(_realType(result_t))
    if compute_uv:
        u = u.astype(result_t)  # no repeated transpose: used fortran order
        vt = vt.astype(result_t)
        return wrap(u), s, wrap(vt)
    else:
        return s


def _load_lapack(libs=[
        "libLAPACK.dylib", "libmkl_rt.so", "libmkl_intel_lp64.so", "liblapack.so",
        "libopenblas.dll",
        find_library('lapack')
],
                 warn=True):
    """load & return a CLAPACK library."""
    global _lapack_lib
    if _lapack_lib is None:
        for l in libs:
            if l is None:
                continue
            try:
                _lapack_lib = CDLL(l)
                _set_CLAPACK_callsignatures(_lapack_lib)
                if warn:
                    warnings.warn("[Loaded " + l + " for gesvd]")
                break
            except OSError:
                pass
    if _lapack_lib is None:
        msg = "Couldn't find LAPACK library for 'gesvd' workaround.\nTried: " + str(libs)
        raise EnvironmentError(msg)
    return _lapack_lib


def _set_CLAPACK_callsignatures(lapack_lib):
    """define the call signature of the CLAPACK functions which we need.
    See http://www.netlib.org/lapack/explore-html/d8/d70/group__lapack.html
    for the (fortran) signature.
    In the C version, all arguments must be given as pointers of the corresponding C types."""
    # Shorthand data type for the arrays.
    # s/d/c/z = fortran single/double/complex_single/complex_double
    s_arr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1)
    d_arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    c_arr = np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1)
    z_arr = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1)
    s_2arr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2)
    d_2arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2)
    c_2arr = np.ctypeslib.ndpointer(dtype=np.complex64, ndim=2)
    z_2arr = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2)

    lapack_lib.sgesvd_.argtypes = \
        [POINTER(c_char), POINTER(c_char), POINTER(c_int), POINTER(c_int),
            s_2arr, POINTER(c_int), s_arr, s_2arr, POINTER(c_int), s_2arr,
            POINTER(c_int), s_arr, POINTER(c_int), POINTER(c_int)]
    lapack_lib.dgesvd_.argtypes = \
        [POINTER(c_char), POINTER(c_char), POINTER(c_int), POINTER(c_int),
            d_2arr, POINTER(c_int), d_arr, d_2arr, POINTER(c_int), d_2arr,
            POINTER(c_int), d_arr, POINTER(c_int), POINTER(c_int)]
    lapack_lib.cgesvd_.argtypes = \
        [POINTER(c_char), POINTER(c_char), POINTER(c_int), POINTER(c_int),
            c_2arr, POINTER(c_int), s_arr, c_2arr, POINTER(c_int),
            c_2arr, POINTER(c_int), c_arr, POINTER(c_int), s_arr,
            POINTER(c_int)]
    lapack_lib.zgesvd_.argtypes = \
        [POINTER(c_char), POINTER(c_char), POINTER(c_int), POINTER(c_int),
            z_2arr, POINTER(c_int), d_arr, z_2arr, POINTER(c_int),
            z_2arr, POINTER(c_int), z_arr, POINTER(c_int), d_arr,
            POINTER(c_int)]


def _get_gesvd(t):
    """return _lapack_lib.#gesvd_ where # = d/z is chosen depending on type `t`"""
    lib = _load_lapack()
    type2gesvd = {
        single: lib.sgesvd_,
        double: lib.dgesvd_,
        csingle: lib.cgesvd_,
        cdouble: lib.zgesvd_
    }
    return type2gesvd[t]
