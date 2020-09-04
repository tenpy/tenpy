"""A collection of tests for tenpy.tools submodules."""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import itertools as it
import tenpy.tools as tools
import warnings
import pytest


def test_inverse_permutation(N=10):
    x = np.random.random(N)
    p = np.arange(N)
    np.random.shuffle(p)
    xnew = x[p]
    pinv = tools.misc.inverse_permutation(p)
    npt.assert_equal(x, xnew[pinv])
    npt.assert_equal(pinv[p], np.arange(N))
    npt.assert_equal(p[pinv], np.arange(N))
    pinv2 = tools.misc.inverse_permutation(tuple(p))
    npt.assert_equal(pinv, pinv2)


def test_argsort():
    x = [1., -1., 1.5, -1.5, 2.j, -2.j]
    npt.assert_equal(tools.misc.argsort(x, 'LM'), [4, 5, 2, 3, 0, 1])
    npt.assert_equal(tools.misc.argsort(x, 'SM'), [0, 1, 2, 3, 4, 5])
    npt.assert_equal(tools.misc.argsort(x, 'LR'), [2, 0, 4, 5, 1, 3])


def test_speigs():
    x = np.array([1., -1.2, 1.5, -1.8, 2.j, -2.2j])
    tol_NULP = len(x)**3
    x_LM = x[tools.misc.argsort(x, 'm>')]
    x_SM = x[tools.misc.argsort(x, 'SM')]
    A = np.diag(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        for k in range(4, 9):
            print(k)
            W, V = tools.math.speigs(A, k, which='LM')
            W = W[tools.misc.argsort(W, 'LM')]
            print(W, x_LM[:k])
            npt.assert_array_almost_equal_nulp(W, x_LM[:k], tol_NULP)
            W, V = tools.math.speigs(A, k, which='SM')
            W = W[tools.misc.argsort(W, 'SM')]
            print(W, x_SM[:k])
            npt.assert_array_almost_equal_nulp(W, x_SM[:k], tol_NULP)


def test_matvec_to_array():
    A_orig = np.random.random([5, 5]) + 1.j * np.random.random([5, 5])

    class A_matvec:
        def __init__(self, A):
            self.A = A
            self.shape = A.shape
            self.dtype = A.dtype

        def matvec(self, v):
            return np.dot(self.A, v)

    A_reg = tools.math.matvec_to_array(A_matvec(A_orig))
    npt.assert_array_almost_equal(A_orig, A_reg, 14)


def test_perm_sign():
    res = [tools.math.perm_sign(u) for u in it.permutations(range(4))]
    check = [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
    npt.assert_equal(res, check)


def test_qr_li():
    cutoff = 1.e-10
    for shape in [(5, 4), (4, 5)]:
        print('shape =', shape)
        A = np.arange(20).reshape(shape)  # linearly dependent: only two rows/columns independent
        A[3, :] = np.random.random() * (cutoff / 100)  # nearly linear dependent
        q, r = tools.math.qr_li(A)
        assert np.linalg.norm(r - np.triu(r)) == 0.
        qdq = q.T.conj().dot(q)
        assert np.linalg.norm(qdq - np.eye(len(qdq))) < 1.e-13
        assert np.linalg.norm(q.dot(r) - A) < cutoff * 20
        r, q = tools.math.rq_li(A)
        assert np.linalg.norm(r - np.triu(r, r.shape[1] - r.shape[0])) == 0.
        qqd = q.dot(q.T.conj())
        assert np.linalg.norm(qqd - np.eye(len(qqd))) < 1.e-13
        assert np.linalg.norm(r.dot(q) - A) < cutoff * 20


def test_memory_usage():
    tools.process.memory_usage()


def test_omp(n=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        if tools.process.omp_set_nthreads(n):
            nthreads = tools.process.omp_get_nthreads()
            print(nthreads)
            assert (nthreads == n)
        else:
            print("test_omp failed to import the OpenMP libaray.")


def test_mkl(n=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        if tools.process.mkl_set_nthreads(n):
            nthreads = tools.process.mkl_get_nthreads()
            print(nthreads)
            assert (nthreads == n)
        else:
            print("test_mkl failed to import the shared MKL libaray.")


def test_optimization():
    level_now = tools.optimization.get_level()
    level_change = "none" if level_now == 1 else "default"
    level_change = tools.optimization.OptimizationFlag[level_change]
    assert tools.optimization.get_level() == level_now
    assert tools.optimization.get_level() != level_change
    with tools.optimization.temporary_level(level_change):
        assert tools.optimization.get_level() == level_change
    assert tools.optimization.get_level() == level_now


def three_exp(x):
    lam = np.array([0.9, 0.4, 0.2])
    pref = np.array([0.01, 0.4, 20])
    return tools.fit.sum_of_exp(lam, pref, x)


def screened_coulomb(x):
    return np.exp(-0.1 * x) / x**2


def test_approximate_sum_of_exp(N=100):
    x = np.arange(1, N + 1)
    for n, f, max_err in [(3, three_exp, 1.e-13), (5, three_exp, 1.e-13), (2, three_exp, 0.04),
                          (1, three_exp, 0.1), (4, screened_coulomb, 7.e-4)]:
        lam, pref = tools.fit.fit_with_sum_of_exp(f, n=n, N=N)
        err = np.sum(np.abs(f(x) - tools.fit.sum_of_exp(lam, pref, x)))
        print(n, f.__name__, err)
        assert err < max_err
