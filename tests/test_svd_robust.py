"""A collection of tests for tenpy.linalg.svd_robust."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
from scipy.linalg import diagsvd

from tenpy.linalg import svd_robust

import random_test  # fix the random seed.  # noqa F401
from tenpy.linalg.random_matrix import standard_normal_complex


def test_svd():
    """check whether svd_function behaves as np.linalg.svd."""
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        print("dtype = ", dtype)
        for m, n in [(1, 1), (1, 10), (10, 1), (10, 10), (10, 20)]:
            print("m, n = ", m, n)
            tol_NULP = 200 * max(max(m, n)**3,
                                 100)  # quite large tolerance, but seems to be required...
            if np.dtype(dtype).kind == 'c':  # complex?
                A = standard_normal_complex((m, n))
            else:
                A = np.random.standard_normal(size=(m, n))
            A = np.asarray(A, dtype)
            Sonly = svd_robust.svd(A, compute_uv=False)

            U_full, S_full, VTfull = svd_robust.svd(A, full_matrices=True, compute_uv=True)
            npt.assert_array_almost_equal_nulp(Sonly, S_full, tol_NULP)
            recalc = U_full.dot(diagsvd(S_full, m, n)).dot(VTfull)
            npt.assert_array_almost_equal_nulp(recalc, A, tol_NULP)

            U, S, VT = svd_robust.svd(A, full_matrices=False, compute_uv=True)
            npt.assert_array_almost_equal_nulp(Sonly, S, tol_NULP)
            recalc = U.dot(np.diag(S)).dot(VT)
            npt.assert_array_almost_equal_nulp(recalc, A, tol_NULP)
        print("types of U, S, VT = ", U.dtype, S.dtype, VT.dtype)
        assert U.dtype == A.dtype
