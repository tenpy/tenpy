"""A collection of tests for tenpy.linalg.backends.fusion_tree_backend"""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import Callable
import pytest
import numpy as np

from tenpy.linalg import backends
from tenpy.linalg.backends import fusion_tree_backend, get_backend
from tenpy.linalg.spaces import ElementarySpace, ProductSpace
from tenpy.linalg.tensors import SymmetricTensor
from tenpy.linalg.symmetries import ProductSymmetry, fibonacci_anyon_category, SU2Symmetry, SU3_3AnyonCategory
from tenpy.linalg.dtypes import Dtype


@pytest.mark.parametrize('num_spaces', [3, 4, 5])
def test_block_sizes(any_symmetry, make_any_space, make_any_sectors, block_backend, num_spaces):
    backend = get_backend('fusion_tree', block_backend)
    spaces = [make_any_space() for _ in range(num_spaces)]
    domain = ProductSpace(spaces, symmetry=any_symmetry, backend=backend)

    for coupled in make_any_sectors(10):
        expect = sum(fusion_tree_backend.forest_block_size(domain, uncoupled, coupled)
                    for uncoupled in domain.iter_uncoupled())
        res = fusion_tree_backend.block_size(domain, coupled)
        assert res == expect


def assert_tensors_almost_equal(a: SymmetricTensor, expect: SymmetricTensor, eps: float):
    assert a.codomain == expect.codomain
    assert a.domain == expect.domain
    assert a.backend.almost_equal(a, expect, rtol=eps, atol=eps)


def assert_repeated_braids_trivial(a: SymmetricTensor, funcs: list[Callable], levels: list[int], repeat: int, eps: float):
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            for _ in range(repeat):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

            assert_tensors_almost_equal(new_a, a, eps)


def assert_clockwise_counterclockwise_trivial(a: SymmetricTensor, funcs: list[Callable], levels: list[int], eps: float):
    # we still allow for input levels to test clockwise -> counterclockwise and counterclockwise -> clockwise
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            new_levels = levels[:]
            for _ in range(2):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=new_levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)
                new_levels[leg:leg+2] = new_levels[leg:leg+2][::-1]

            assert_tensors_almost_equal(new_a, a, eps)


def assert_braiding_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable], levels: list[int], eps: float):
    # TODO
    raise NotImplementedError


def test_c_symbol_fibonacci_anyons(block_backend: str):
    # TODO rescaling axes commutes with braiding

    funcs = [fusion_tree_backend._apply_single_c_symbol_inefficient,
             fusion_tree_backend._apply_single_c_symbol_more_efficient]
    backend = get_backend('fusion_tree', block_backend)
    eps = 1.e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    codomain = ProductSpace([s2, s1, s2, s2])
    domain = ProductSpace([s2, s1, s2])

    block_inds = np.array([[0,0], [1,1]])
    blocks = [backend.block_backend.block_random_uniform((8, 3), Dtype.complex128),
              backend.block_backend.block_random_uniform((13, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)

    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    r1 = np.exp(-4j*np.pi/5)  # R symbols
    rtau = np.exp(3j*np.pi/5)

    expect = [np.zeros((8, 3), dtype=complex), np.zeros((13, 5), dtype=complex)]

    expect[0][0, :] = blocks[0][0, :]
    expect[0][1, :] = blocks[0][1, :]
    expect[0][2, :] = blocks[0][2, :]
    expect[0][3, :] = blocks[0][3, :] * r1
    expect[0][4, :] = blocks[0][4, :] * rtau
    expect[0][5, :] = blocks[0][5, :] * rtau
    expect[0][6, :] = blocks[0][6, :] * r1
    expect[0][7, :] = blocks[0][7, :] * rtau

    expect[1][0, :] = blocks[1][0, :]
    expect[1][1, :] = blocks[1][1, :]
    expect[1][2, :] = blocks[1][2, :]
    expect[1][3, :] = blocks[1][3, :]
    expect[1][4, :] = blocks[1][4, :]
    expect[1][5, :] = blocks[1][5, :] * rtau
    expect[1][6, :] = blocks[1][6, :] * r1
    expect[1][7, :] = blocks[1][7, :] * rtau
    expect[1][8, :] = blocks[1][8, :] * r1
    expect[1][9, :] = blocks[1][9, :] * rtau
    expect[1][10, :] = blocks[1][10, :] * r1
    expect[1][11, :] = blocks[1][11, :] * rtau
    expect[1][12, :] = blocks[1][12, :] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1, s2, s2, s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, domain, backend=backend)

    # do this without permute_legs for the different implementations
    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchanges legs 5 and 6 (in domain)
    expect = [np.zeros((8, 3), dtype=complex), np.zeros((13, 5), dtype=complex)]

    expect[0][:, 0] = blocks[0][:, 0]
    expect[0][:, 1] = blocks[0][:, 1] * r1
    expect[0][:, 2] = blocks[0][:, 2] * rtau

    expect[1][:, 0] = blocks[1][:, 0]
    expect[1][:, 1] = blocks[1][:, 1]
    expect[1][:, 2] = blocks[1][:, 2] * rtau
    expect[1][:, 3] = blocks[1][:, 3] * r1
    expect[1][:, 4] = blocks[1][:, 4] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_domain = ProductSpace([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=5, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchanges legs 2 and 3 (in codomain)
    phi = (1 + 5**0.5) / 2
    ctttt11 = phi**-1 * r1.conj()  # C symbols
    cttttt1 = phi**-0.5 * rtau * r1.conj()
    ctttt1t = phi**-0.5 * rtau.conj()
    ctttttt = -1*phi**-1

    expect = [np.zeros((8, 3), dtype=complex), np.zeros((13, 5), dtype=complex)]

    expect[0][0, :] = blocks[0][1, :]
    expect[0][1, :] = blocks[0][0, :]
    expect[0][2, :] = blocks[0][2, :] * rtau
    expect[0][3, :] = blocks[0][3, :]
    expect[0][4, :] = blocks[0][5, :]
    expect[0][5, :] = blocks[0][4, :]
    expect[0][6, :] = blocks[0][6, :] * r1
    expect[0][7, :] = blocks[0][7, :] * rtau

    expect[1][0, :] = blocks[1][0, :]
    expect[1][1, :] = blocks[1][2, :]
    expect[1][2, :] = blocks[1][1, :]
    expect[1][3, :] = blocks[1][3, :] * ctttt11 + blocks[1][4, :] * cttttt1
    expect[1][4, :] = blocks[1][3, :] * ctttt1t + blocks[1][4, :] * ctttttt
    expect[1][5, :] = blocks[1][5, :]
    expect[1][6, :] = blocks[1][8, :]
    expect[1][7, :] = blocks[1][9, :]
    expect[1][8, :] = blocks[1][6, :]
    expect[1][9, :] = blocks[1][7, :]
    expect[1][10, :] = blocks[1][10, :] * rtau
    expect[1][11, :] = blocks[1][11, :] * ctttt11 + blocks[1][12, :] * cttttt1
    expect[1][12, :] = blocks[1][11, :] * ctttt1t + blocks[1][12, :] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=2, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)



def test_c_symbol_product_sym(block_backend: str):
    # TODO rescaling axes commutes with braiding
    # test c symbols
    
    funcs = [fusion_tree_backend._apply_single_c_symbol_inefficient,
             fusion_tree_backend._apply_single_c_symbol_more_efficient]
    backend = get_backend('fusion_tree', block_backend)
    eps = 1.e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [2])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    codomain = ProductSpace([s2, s2, s2])
    domain = ProductSpace([s2, s1, s2])

    # block charges: 0: [0, 0], 1: [1, 0], 2: [0, 1], 3: [1, 1]
    #                4: [0, 2], 5: [1, 2], 6: [0, 3], 7: [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(13, 8), (12, 8), (16, 16), (38, 34), (12, 8), (12, 8), (8, 8), (16, 16)]
    blocks = [backend.block_backend.block_random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)

    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    r1 = np.exp(-4j*np.pi/5)  # Fib R symbols
    rtau = np.exp(3j*np.pi/5)
    exc = [0, 2, 1, 3]
    exc2 = [4, 5, 6, 7, 0, 1, 2, 3]
    exc3 = [0, 1, 4, 5, 2, 3, 6, 7]

    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    expect[0][:9, :] = blocks[0][[0] + [1 + i for i in exc2], :]
    expect[0][9:, :] = blocks[0][[9 + i for i in exc], :] * r1 * -1

    expect[1][:8, :] = blocks[1][exc2, :]
    expect[1][8:, :] = blocks[1][[8 + i for i in exc], :] * rtau * -1

    expect[2][:8, :] = blocks[2][exc3, :] * rtau * -1
    expect[2][8:, :] = blocks[2][[8 + i for i in exc3], :] * rtau

    expect[3][:6, :] = blocks[3][[0, 1, 4, 5, 2, 3], :]
    expect[3][6:14, :] = blocks[3][[6 + i for i in exc3], :] * r1 * -1
    expect[3][14:22, :] = blocks[3][[14 + i for i in exc3], :] * r1
    expect[3][22:30, :] = blocks[3][[22 + i for i in exc3], :] * rtau * -1
    expect[3][30:, :] = blocks[3][[30 + i for i in exc3], :] * rtau

    expect[4][:8, :] = blocks[4][exc2, :]
    expect[4][8:, :] = blocks[4][[8 + i for i in exc], :] * r1

    expect[5][:8, :] = blocks[5][exc2, :]
    expect[5][8:, :] = blocks[5][[8 + i for i in exc], :] * rtau

    expect[6][:, :] = blocks[6][exc3, :] * rtau

    expect[7][:8, :] = blocks[7][exc3, :] * r1
    expect[7][8:, :] = blocks[7][[8 + i for i in exc3], :] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 4 and 5 (in domain)
    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, :4]
    expect[0][:, 4:] = blocks[0][:, [4 + i for i in exc]] * r1 * -1

    expect[1][:, :4] = blocks[1][:, :4]
    expect[1][:, 4:] = blocks[1][:, [4 + i for i in exc]] * rtau * -1

    expect[2][:, :8] = blocks[2][:, exc3] * rtau * -1
    expect[2][:, 8:] = blocks[2][:, [8 + i for i in exc3]] * rtau

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = blocks[3][:, [2 + i for i in exc3]] * r1 * -1
    expect[3][:, 10:18] = blocks[3][:, [10 + i for i in exc3]] * r1
    expect[3][:, 18:26] = blocks[3][:, [18 + i for i in exc3]] * rtau * -1
    expect[3][:, 26:34] = blocks[3][:, [26 + i for i in exc3]] * rtau

    expect[4][:, :4] = blocks[4][:, :4]
    expect[4][:, 4:] = blocks[4][:, [4 + i for i in exc]] * r1

    expect[5][:, :4] = blocks[5][:, :4]
    expect[5][:, 4:] = blocks[5][:, [4 + i for i in exc]] * rtau

    expect[6][:, :] = blocks[6][:, exc3] * rtau

    expect[7][:, :8] = blocks[7][:, exc3] * r1
    expect[7][:, 8:] = blocks[7][:, [8 + i for i in exc3]] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_domain = ProductSpace([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)


def test_c_symbol_su3_3(block_backend: str):
    # TODO SU(3)_3
    # braid in domain

    funcs = [fusion_tree_backend._apply_single_c_symbol_inefficient,
             fusion_tree_backend._apply_single_c_symbol_more_efficient]
    backend = get_backend('fusion_tree', block_backend)
    eps = 1.e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    [c0, c1, c2, c3] = [np.array([i]) for i in range(4)]  # charges
    codomain = ProductSpace([s1, s1, s1])
    domain = ProductSpace([s1, s2, s2])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(6, 12), (16, 36), (5, 12), (5, 12)]
    blocks = [backend.block_backend.block_random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)

    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    # SU(3)_3 R symbols
    # exchanging two 8s gives -1 except if they fuse to 8, then
    r8 = [-1j, 1j]  # for the two multiplicities
    # all other R symbols are trivial

    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    for i in [0, 2, 3]:
        expect[i][0, :] = blocks[i][0, :] * r8[0]
        expect[i][1, :] = blocks[i][1, :] * r8[1]
        expect[i][2, :] = blocks[i][2, :] * -1
        expect[i][[3, 4], :] = blocks[i][[4, 3], :]
    expect[0][5, :] = blocks[0][5, :]

    expect[1][[0, 5, 6], :] = blocks[1][[0, 5, 6], :] * -1
    expect[1][[1, 3, 7], :] = blocks[1][[1, 3, 7], :] * r8[0]
    expect[1][[2, 4, 8], :] = blocks[1][[2, 4, 8], :] * r8[1]
    expect[1][9:, :] = blocks[1][[12, 13, 14, 9, 10, 11, 15], :]

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 4 and 5 (in domain)
    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    for i in [0, 2, 3]:
        expect[i][:, :4] = blocks[i][:, :4] * r8[0]
        expect[i][:, 4:8] = blocks[i][:, 4:8] * r8[1]
        expect[i][:, 8:] = blocks[i][:, 8:]

    expect[1][:, :4] = blocks[1][:, :4] * -1
    expect[1][:, 4:8] = blocks[1][:, 4:8] * r8[0]
    expect[1][:, 8:12] = blocks[1][:, 8:12] * r8[1]
    expect[1][:, 12:16] = blocks[1][:, 12:16] * r8[0]
    expect[1][:, 16:20] = blocks[1][:, 16:20] * r8[1]
    expect[1][:, 20:28] = blocks[1][:, 20:28] * -1
    expect[1][:, 28:] = blocks[1][:, 28:]

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_domain = ProductSpace([s2, s1, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 1 and 2 (in codomain)
    # we usually use the convention that in the codomain, the two final indices are f, e
    # the F symbols f2 and f1 are chosen such that we use the indices e, f
    # e.g. f2[e, f] = _f_symbol(10, 8, 8, 8, f, e)
    #      f1[e, f] = _f_symbol(8, 10, 8, 8, f, e)
    f2 = np.array([[-.5, -3**.5/2], [3**.5/2, -.5]])
    f1 = f2.T
    csym = sym._c_symbol
    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    expect[0][0, :] = blocks[0][0, :] * r8[0]
    expect[0][1, :] = blocks[0][1, :] * r8[1]
    expect[0][[2, 3], :] = blocks[0][[3, 2], :]
    expect[0][4, :] = blocks[0][4, :] * -1
    expect[0][5, :] = blocks[0][5, :]

    v = [blocks[1][i, :] for i in range(7)]
    charges = [c0] + [c1]*4 + [c2, c3]
    mul1 = [0] * 7
    mul2 = [0] * 7
    mul1[2], mul1[4] = 1, 1
    mul2[3], mul2[4] = 1, 1

    for i in range(7):
        w = [csym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        expect[1][i, :] = np.sum([v[j] * w[j] for j in range(7)], axis=0)

    expect[1][7, :] = (blocks[1][9, :] * (f2[0,0]*f2[0,0] + f2[0,1]*f2[1,0])
                       + blocks[1][10, :] * (f2[1,0]*f2[0,0] + f2[1,1]*f2[1,0]))
    expect[1][8, :] = (blocks[1][9, :] * (f2[0,0]*f2[0,1] + f2[0,1]*f2[1,1])
                       + blocks[1][10, :] * (f2[1,0]*f2[0,1] + f2[1,1]*f2[1,1]))
    expect[1][9, :] = (blocks[1][7, :] * (f1[0,0]*f1[0,0] + f1[0,1]*f1[1,0])
                       + blocks[1][8, :] * (f1[1,0]*f1[0,0] + f1[1,1]*f1[1,0]))
    expect[1][10, :] = (blocks[1][7, :] * (f1[0,0]*f1[0,1] + f1[0,1]*f1[1,1])
                        + blocks[1][8, :] * (f1[1,0]*f1[0,1] + f1[1,1]*f1[1,1]))
    expect[1][11, :] = blocks[1][11, :]
    expect[1][12, :] = (blocks[1][12, :] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                        + blocks[1][13, :] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[1][13, :] = (blocks[1][12, :] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[1][13, :] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[1][[14, 15], :] = blocks[1][[15, 14], :] * -1

    expect[2][0, :] = (blocks[2][0, :] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                       + blocks[2][1, :] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[2][1, :] = (blocks[2][0, :] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                       + blocks[2][1, :] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[2][[2, 3], :] = blocks[2][[3, 2], :] * -1
    expect[2][4, :] = blocks[2][4, :] * -1

    expect[3][0, :] = (blocks[3][0, :] * (f2[0,0]*r8[0]*f1[0,0] + f2[0,1]*r8[1]*f1[1,0])
                       + blocks[3][1, :] * (f2[1,0]*r8[0]*f1[0,0] + f2[1,1]*r8[1]*f1[1,0]))
    expect[3][1, :] = (blocks[3][0, :] * (f2[0,0]*r8[0]*f1[0,1] + f2[0,1]*r8[1]*f1[1,1])
                       + blocks[3][1, :] * (f2[1,0]*r8[0]*f1[0,1] + f2[1,1]*r8[1]*f1[1,1]))
    expect[3][[2, 3], :] = blocks[3][[3, 2], :] * -1
    expect[3][4, :] = blocks[3][4, :] * -1

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=1, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 4 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=4, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)