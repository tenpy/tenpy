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

Examples
--------
The idea is that you just import the `svd` from this module and use it as replacement for
``np.linalg.svd`` or ``scipy.linalg.svd``:

>>> from tenpy.linalg.svd_robust import svd
>>> U, S, VT = svd([[1., 1.], [0., 1.]])
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import scipy
import scipy.linalg
import warnings

__all__ = ['svd']


def svd(a,
        full_matrices=True,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver='gesdd',
        warn=True):
    """Wrapper around :func:`scipy.linalg.svd` with `gesvd` backup plan.

    Tries to avoid raising an LinAlgError by using the lapack_driver `gesvd`,
    if `gesdd` failed.

    Parameters not described below are as in :func:`scipy.linalg.svd`

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
        Whether to create a warning when the SVD failed.


    Returns
    -------
    U, S, Vh : ndarray
        As described in doc-string of :func:`scipy.linalg.svd`.
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
    return scipy.linalg.svd(a, full_matrices, compute_uv, overwrite_a, check_finite, 'gesvd')
