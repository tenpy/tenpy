"""A collection of tests for tenpy.linalg.backends.fusion_tree_backend"""
# Copyright (C) TeNPy Developers, GNU GPLv3
import pytest
import numpy as np

from tenpy.linalg import backends
from tenpy.linalg.backends import fusion_tree_backend, get_backend
from tenpy.linalg.spaces import ElementarySpace, ProductSpace
from tenpy.linalg.tensors import SymmetricTensor
from tenpy.linalg.symmetries import fibonacci_anyon_category
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


def test_c_symbol_fibonacci_anyons(block_backend: backends.BlockBackend):
    # TODO rescaling axes commutes with braiding

    funcs = [fusion_tree_backend._apply_single_c_symbol_inefficient,
             fusion_tree_backend._apply_single_c_symbol_more_efficient]
    backend = backends.FusionTreeBackend(block_backend=block_backend)
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

        assert new_codomain == expect_codomain
        assert new_domain == domain
        assert backend.almost_equal(new_tens, expect_tens, rtol=eps, atol=eps)


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

        assert new_codomain == codomain
        assert new_domain == expect_domain
        assert backend.almost_equal(new_tens, expect_tens, rtol=eps, atol=eps)


    # exchanges legs 2 and 3 (in codomain)  e,c,d   a,c,f conj()
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

        assert new_codomain == codomain
        assert new_domain == domain
        assert backend.almost_equal(new_tens, expect_tens, rtol=eps, atol=eps)


    # braid 10 times == trivial
    for func in funcs:
        for leg in range(tens.num_legs-1):
            if leg == tens.num_codomain_legs - 1:
                continue
            new_tens = tens.copy()
            for _ in range(10):
                new_data, new_codomain, new_domain = func(new_tens, leg=leg, levels=levels)
                new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

            assert new_codomain == codomain
            assert new_domain == domain
            assert backend.almost_equal(new_tens, tens, rtol=eps, atol=eps)


    # braid clockwise and then counter-clockwise == trivial
    for func in funcs:
        for leg in range(tens.num_legs-1):
            if leg == tens.num_codomain_legs - 1:
                continue
            new_tens = tens.copy()
            new_levels = levels[:]
            for _ in range(2):
                new_data, new_codomain, new_domain = func(new_tens, leg=leg, levels=new_levels)
                new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
                new_levels[leg:leg+2] = new_levels[leg:leg+2][::-1]

            assert new_codomain == codomain
            assert new_domain == domain
            assert backend.almost_equal(new_tens, tens, rtol=eps, atol=eps)