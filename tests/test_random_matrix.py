"""A collection of tests for :module:`tenpy.linalg.random_matrix`."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import tenpy.linalg.random_matrix as rmat
import tenpy.linalg.np_conserved as npc

chinfo = npc.ChargeInfo([1], ['testcharge'])
_, leg = npc.LegCharge.from_qflat(chinfo, np.array([0, 1, 1, 2, 2, 2, 3, 4, 4])).bunch()

np.random.seed(123)

EPS = 1.e-14


def test_random_matrix_standard_normal_complex():
    for size in [1, 3, 4]:
        a = rmat.standard_normal_complex((size, size))
        assert (a.dtype == np.complex)
        print(a)
    b = npc.Array.from_func_square(rmat.standard_normal_complex, leg)
    b.test_sanity()
    print("b =", b)
    assert (b.dtype == np.complex)


def test_random_matrix_GOE():
    for size in [1, 3, 4]:
        a = rmat.GOE((size, size))
        assert (a.dtype == np.float)
        print(a)
        npt.assert_array_equal(a, a.T)
    b = npc.Array.from_func_square(rmat.GOE, leg)
    b.test_sanity()
    assert (b.dtype == np.float)
    print("b =", b)
    assert (npc.norm(b - b.conj().itranspose()) == 0)


def test_random_matrix_GUE():
    for size in [1, 3, 4]:
        a = rmat.GUE((size, size))
        assert (a.dtype == np.complex)
        print(a)
        npt.assert_array_equal(a, a.T.conj())
    b = npc.Array.from_func_square(rmat.GUE, leg)
    b.test_sanity()
    assert (b.dtype == np.complex)
    print("b =", b)
    assert (npc.norm(b - b.conj().itranspose()) == 0)


def test_random_matrix_CRE():
    for size in [1, 3, 4]:
        a = rmat.CRE((size, size))
        assert (a.dtype == np.float)
        print(a)
        npt.assert_allclose(np.dot(a, a.T), np.eye(size), EPS, EPS)
    b = npc.Array.from_func_square(rmat.CRE, leg)
    b.test_sanity()
    assert (b.dtype == np.float)
    print("b =", b)
    id = npc.eye_like(b)
    assert (npc.norm(npc.tensordot(b, b.conj().itranspose(), axes=[1, 0]) - id) < EPS)
    assert (npc.norm(npc.tensordot(b.conj().itranspose(), b, axes=[1, 0]) - id) < EPS)


def test_random_matrix_COE():
    for size in [1, 3, 4]:
        a = rmat.COE((size, size))
        assert (a.dtype == np.complex)
        print(a)
        npt.assert_array_equal(a, a.T)
        npt.assert_allclose(np.dot(a, a.conj().T), np.eye(size), EPS, EPS)
    b = npc.Array.from_func_square(rmat.COE, leg)
    b.test_sanity()
    assert (b.dtype == np.complex)
    print("b =", b)
    assert (npc.norm(b - b.conj(complex_conj=False).itranspose()) == 0)
    id = npc.eye_like(b)
    assert (npc.norm(npc.tensordot(b, b.complex_conj(), axes=[1, 0]) - id) < EPS)
    assert (npc.norm(npc.tensordot(b.complex_conj(), b, axes=[1, 0]) - id) < EPS)


def test_random_matrix_CUE():
    for size in [1, 3, 4]:
        a = rmat.CUE((size, size))
        print(a)
        assert (a.dtype == np.complex)
        npt.assert_allclose(np.dot(a, a.T.conj()), np.eye(size), EPS, EPS)
    b = npc.Array.from_func_square(rmat.CUE, leg)
    b.test_sanity()
    assert (b.dtype == np.complex)
    print("b =", b)
    id = npc.eye_like(b)
    assert (npc.norm(npc.tensordot(b, b.conj().itranspose(), axes=[1, 0]) - id) < EPS)
    assert (npc.norm(npc.tensordot(b.conj().itranspose(), b, axes=[1, 0]) - id) < EPS)


def test_random_matrix_O_close_1():
    for x in [0., 0.001]:
        for size in [1, 3, 4]:
            a = rmat.O_close_1((size, size), x)
            assert (a.dtype == np.float)
            print(a)
            npt.assert_allclose(np.dot(a, a.T), np.eye(size), EPS, EPS)
            npt.assert_allclose(a, np.eye(size), 10 * x, 10 * x)  # exact for x=0!
        b = npc.Array.from_func_square(rmat.O_close_1, leg, func_args=[x])
        b.test_sanity()
        assert (b.dtype == np.float)
        print("b =", b)
        id = npc.eye_like(b)
        assert (npc.norm(npc.tensordot(b, b.conj().itranspose(), axes=[1, 0]) - id) < EPS)
        assert (npc.norm(npc.tensordot(b.conj().itranspose(), b, axes=[1, 0]) - id) < EPS)
        assert (npc.norm(b - id) <= 10 * x)


def test_random_matrix_U_close_1():
    for x in [0., 0.001]:
        for size in [1, 3, 4]:
            a = rmat.U_close_1((size, size), x)
            print(a)
            assert (a.dtype == np.complex)
            npt.assert_allclose(np.dot(a, a.T.conj()), np.eye(size), EPS, EPS)
            npt.assert_allclose(a, np.eye(size), 10 * x, 10 * x + EPS)  # not exact for x=0!
        b = npc.Array.from_func_square(rmat.U_close_1, leg, func_args=[x])
        b.test_sanity()
        assert (b.dtype == np.complex)
        print("b =", b)
        id = npc.eye_like(b)
        assert (npc.norm(npc.tensordot(b, b.conj().itranspose(), axes=[1, 0]) - id) < EPS)
        assert (npc.norm(npc.tensordot(b.conj().itranspose(), b, axes=[1, 0]) - id) < EPS)
        assert (npc.norm(b - id) <= 10 * x + EPS)  # not exact for x=0!
