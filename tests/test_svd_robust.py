"""A collection of tests for tenpy.linalg.svd_robust."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
from scipy.linalg import diagsvd

from tenpy.linalg import svd_robust

import random_test  # fix the random seed.
from tenpy.linalg.random_matrix import standard_normal_complex

import pytest


def test_CLAPACK_import():
    """just try to import the lapack library on the local system."""
    if not svd_robust._old_scipy:
        return  # scip test: no need to import that.
    try:
        svd_robust._load_lapack(warn=False)
    except EnvironmentError as e:
        print(str(e))
        if str(e).startswith("Couldn't find LAPACK"):
            print("(Not an issue if you have scipy >= 0.18.0)")
        assert (False)


def check_svd_function(svd_function):
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
            Sonly = svd_function(A, compute_uv=False)

            Ufull, Sfull, VTfull = svd_function(A, full_matrices=True, compute_uv=True)
            npt.assert_array_almost_equal_nulp(Sonly, Sfull, tol_NULP)
            recalc = Ufull.dot(diagsvd(Sfull, m, n)).dot(VTfull)
            npt.assert_array_almost_equal_nulp(recalc, A, tol_NULP)

            U, S, VT = svd_function(A, full_matrices=False, compute_uv=True)
            npt.assert_array_almost_equal_nulp(Sonly, S, tol_NULP)
            recalc = U.dot(np.diag(S)).dot(VT)
            npt.assert_array_almost_equal_nulp(recalc, A, tol_NULP)
        print("types of U, S, VT = ", U.dtype, S.dtype, VT.dtype)
        assert U.dtype == A.dtype


def test_svd():
    check_svd_function(svd_robust.svd)


def test_svd_gesvd():
    if not svd_robust._old_scipy:
        pytest.skip("have scipy >0.18.0, no need to load LAPACK directly for robust svd")
    check_svd_function(svd_robust.svd_gesvd)
