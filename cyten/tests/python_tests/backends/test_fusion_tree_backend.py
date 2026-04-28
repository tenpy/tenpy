"""A collection of tests for cyten.backends.fusion_tree_backend"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Callable
from math import prod

import numpy as np
import pytest

from cyten import backends
from cyten.backends import fusion_tree_backend, get_backend
from cyten.block_backends.dtypes import Dtype
from cyten.symmetries import (
    ElementarySpace,
    FusionTree,
    ProductSymmetry,
    SU2_kAnyonCategory,
    SU2Symmetry,
    SU3_3AnyonCategory,
    Symmetry,
    TensorProduct,
    fibonacci_anyon_category,
    ising_anyon_category,
    u1_symmetry,
    z5_symmetry,
)
from cyten.tensors import DiagonalTensor, SymmetricTensor, permute_legs, transpose
from cyten.testing import assert_tensors_almost_equal, random_ElementarySpace, random_tensor


def test_c_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks, cross_check_single_c_symbol_tree_cols, apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    codomain = TensorProduct([s2, s1, s2, s2])
    domain = TensorProduct([s2, s1, s2])

    block_inds = np.array([[0, 0], [1, 1]])
    blocks = [
        backend.block_backend.random_uniform((8, 3), Dtype.complex128),
        backend.block_backend.random_uniform((13, 5), Dtype.complex128),
    ]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    R_1 = np.exp(-4j * np.pi / 5)  # R symbols
    R_tau = np.exp(3j * np.pi / 5)
    assert sym.r_symbol(sym.tau, sym.tau, sym.vacuum) == R_1
    assert sym.r_symbol(sym.tau, sym.tau, sym.tau) == R_tau
    phi = (1 + 5**0.5) / 2
    C_tttt11 = phi**-1 * R_1.conj()  # C symbols
    C_ttttt1 = phi**-0.5 * R_tau * R_1.conj()
    C_tttt1t = phi**-0.5 * R_tau.conj()
    C_tttttt = -1 * phi**-1
    assert np.allclose(sym.c_symbol(sym.tau, sym.tau, sym.tau, sym.tau, sym.vacuum, sym.vacuum), C_tttt11)
    assert np.allclose(sym.c_symbol(sym.tau, sym.tau, sym.tau, sym.tau, sym.tau, sym.vacuum), C_ttttt1)
    assert np.allclose(sym.c_symbol(sym.tau, sym.tau, sym.tau, sym.tau, sym.vacuum, sym.tau), C_tttt1t)
    assert np.allclose(sym.c_symbol(sym.tau, sym.tau, sym.tau, sym.tau, sym.tau, sym.tau), C_tttttt)
    # R_1=-0.8090-0.5878j  R_tau=-0.3090+0.9511j
    # C_tttt11=-0.5000+0.3633j   C_tttt1t=-0.2429-0.7477j   C_ttttt1=-0.2429-0.7477j   C_tttttt=-0.6180

    # Exchange legs 0 and 1 (in codomain)
    # =================================
    # build expected tensors from explicit blocks.
    expect_block_0 = backend.block_backend.copy_block(blocks[0])
    expect_block_0[[3, 6], :] *= R_1
    expect_block_0[[4, 5, 7], :] *= R_tau
    expect_block_1 = backend.block_backend.copy_block(blocks[1])
    expect_block_1[[6, 8, 10], :] *= R_1
    expect_block_1[[5, 7, 9, 11, 12], :] *= R_tau
    expect_data = backends.FusionTreeData(
        block_inds, [expect_block_0, expect_block_1], Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1, s2, s2, s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, domain, backend=backend)

    # do this without permute_legs for the different implementations
    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [1, 0, 2, 3], [6, 5, 4], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # Exchange legs 5 and 6 (in domain)
    # =================================
    expect_block_0 = backend.block_backend.copy_block(blocks[0])
    expect_block_0[:, 1] *= R_1
    expect_block_0[:, 2] *= R_tau
    expect_block_1 = backend.block_backend.copy_block(blocks[1])
    expect_block_1[:, 3] *= R_1
    expect_block_1[:, [2, 4]] *= R_tau
    expect_data = backends.FusionTreeData(
        block_inds, [expect_block_0, expect_block_1], Dtype.complex128, device=backend.block_backend.default_device
    )

    expect_domain = TensorProduct([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=5, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2, 3], [5, 6, 4], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # Exchange legs 2 and 3 (in codomain)
    # =================================
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][0, :] = blocks[0][1, :]
    expect[0][1, :] = blocks[0][0, :]
    expect[0][2, :] = blocks[0][2, :] * R_tau
    expect[0][3, :] = blocks[0][3, :]
    expect[0][4, :] = blocks[0][5, :]
    expect[0][5, :] = blocks[0][4, :]
    expect[0][6, :] = blocks[0][6, :] * R_1
    expect[0][7, :] = blocks[0][7, :] * R_tau

    expect[1][0, :] = blocks[1][0, :]
    expect[1][1, :] = blocks[1][2, :]
    expect[1][2, :] = blocks[1][1, :]
    expect[1][3, :] = blocks[1][3, :] * C_tttt11 + blocks[1][4, :] * C_ttttt1
    expect[1][4, :] = blocks[1][3, :] * C_tttt1t + blocks[1][4, :] * C_tttttt
    expect[1][5, :] = blocks[1][5, :]
    expect[1][6, :] = blocks[1][8, :]
    expect[1][7, :] = blocks[1][9, :]
    expect[1][8, :] = blocks[1][6, :]
    expect[1][9, :] = blocks[1][7, :]
    expect[1][10, :] = blocks[1][10, :] * R_tau
    expect[1][11, :] = blocks[1][11, :] * C_tttt11 + blocks[1][12, :] * C_ttttt1
    expect[1][12, :] = blocks[1][11, :] * C_tttt1t + blocks[1][12, :] * C_tttttt

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=2, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 3, 2], [6, 5, 4], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # Exchange legs 4 and 5 (in domain)
    # =================================
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][:, 0] = blocks[0][:, 0] * R_1
    expect[0][:, 1] = blocks[0][:, 1]
    expect[0][:, 2] = blocks[0][:, 2] * R_tau

    expect[1][:, 0] = blocks[1][:, 0]
    expect[1][:, 1] = blocks[1][:, 1] * R_tau
    expect[1][:, 2] = blocks[1][:, 2]
    expect[1][:, 3] = blocks[1][:, 3] * C_tttt11 + blocks[1][:, 4] * C_ttttt1
    expect[1][:, 4] = blocks[1][:, 3] * C_tttt1t + blocks[1][:, 4] * C_tttttt

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_domain = TensorProduct([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2, 3], [6, 4, 5], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # more checks
    # =================================
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_c_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks, cross_check_single_c_symbol_tree_cols, apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [2])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    codomain = TensorProduct([s2, s2, s2])
    domain = TensorProduct([s2, s1, s2])

    # block charges: 0: [0, 0], 1: [1, 0], 2: [0, 1], 3: [1, 1]
    #                4: [0, 2], 5: [1, 2], 6: [0, 3], 7: [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(13, 8), (12, 8), (16, 16), (38, 34), (12, 8), (12, 8), (8, 8), (16, 16)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    R_1 = np.exp(-4j * np.pi / 5)  # Fib R symbols
    R_tau = np.exp(3j * np.pi / 5)
    exc = [0, 2, 1, 3]
    exc2 = [4, 5, 6, 7, 0, 1, 2, 3]
    exc3 = [0, 1, 4, 5, 2, 3, 6, 7]

    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:9, :] = blocks[0][[0] + [1 + i for i in exc2], :]
    expect[0][9:, :] = blocks[0][[9 + i for i in exc], :] * R_1 * -1

    expect[1][:8, :] = blocks[1][exc2, :]
    expect[1][8:, :] = blocks[1][[8 + i for i in exc], :] * R_tau * -1

    expect[2][:8, :] = blocks[2][exc3, :] * R_tau * -1
    expect[2][8:, :] = blocks[2][[8 + i for i in exc3], :] * R_tau

    expect[3][:6, :] = blocks[3][[0, 1, 4, 5, 2, 3], :]
    expect[3][6:14, :] = blocks[3][[6 + i for i in exc3], :] * R_1 * -1
    expect[3][14:22, :] = blocks[3][[14 + i for i in exc3], :] * R_1
    expect[3][22:30, :] = blocks[3][[22 + i for i in exc3], :] * R_tau * -1
    expect[3][30:, :] = blocks[3][[30 + i for i in exc3], :] * R_tau

    expect[4][:8, :] = blocks[4][exc2, :]
    expect[4][8:, :] = blocks[4][[8 + i for i in exc], :] * R_1

    expect[5][:8, :] = blocks[5][exc2, :]
    expect[5][8:, :] = blocks[5][[8 + i for i in exc], :] * R_tau

    expect[6][:, :] = blocks[6][exc3, :] * R_tau

    expect[7][:8, :] = blocks[7][exc3, :] * R_1
    expect[7][8:, :] = blocks[7][[8 + i for i in exc3], :] * R_tau

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [1, 0, 2], [5, 4, 3], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 4 and 5 (in domain)
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, :4]
    expect[0][:, 4:] = blocks[0][:, [4 + i for i in exc]] * R_1 * -1

    expect[1][:, :4] = blocks[1][:, :4]
    expect[1][:, 4:] = blocks[1][:, [4 + i for i in exc]] * R_tau * -1

    expect[2][:, :8] = blocks[2][:, exc3] * R_tau * -1
    expect[2][:, 8:] = blocks[2][:, [8 + i for i in exc3]] * R_tau

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = blocks[3][:, [2 + i for i in exc3]] * R_1 * -1
    expect[3][:, 10:18] = blocks[3][:, [10 + i for i in exc3]] * R_1
    expect[3][:, 18:26] = blocks[3][:, [18 + i for i in exc3]] * R_tau * -1
    expect[3][:, 26:34] = blocks[3][:, [26 + i for i in exc3]] * R_tau

    expect[4][:, :4] = blocks[4][:, :4]
    expect[4][:, 4:] = blocks[4][:, [4 + i for i in exc]] * R_1

    expect[5][:, :4] = blocks[5][:, :4]
    expect[5][:, 4:] = blocks[5][:, [4 + i for i in exc]] * R_tau

    expect[6][:, :] = blocks[6][:, exc3] * R_tau

    expect[7][:, :8] = blocks[7][:, exc3] * R_1
    expect[7][:, 8:] = blocks[7][:, [8 + i for i in exc3]] * R_tau

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_domain = TensorProduct([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [4, 5, 3], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 3 and 4 (in domain)
    phi = (1 + 5**0.5) / 2
    C_tttt11 = phi**-1 * R_1.conj()  # C symbols
    C_ttttt1 = phi**-0.5 * R_tau * R_1.conj()
    C_tttt1t = phi**-0.5 * R_tau.conj()
    C_tttttt = -1 * phi**-1
    exc4 = [0, 2, 1, 3, 4, 6, 5, 7]
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * R_1 * -1
    expect[0][:, 4:] = blocks[0][:, 4:]

    expect[1][:, :4] = blocks[1][:, exc] * R_tau * -1
    expect[1][:, 4:] = blocks[1][:, 4:]

    # f-symbols for su(2) [e -> f]: 0 -> 0: -1/2, 2 -> 2: 1/2, 0 -> 2 and 2 -> 0: 3**.5/2
    expect[2][:, :8] = blocks[2][:, exc4] * R_tau * (-1 / 4 + 3 / 4) + blocks[2][:, [8 + i for i in exc4]] * R_tau * (
        3**0.5 / 4 + 3**0.5 / 4
    )
    expect[2][:, 8:] = blocks[2][:, exc4] * R_tau * (3**0.5 / 4 + 3**0.5 / 4) + blocks[2][
        :, [8 + i for i in exc4]
    ] * R_tau * (1 / 4 - 3 / 4)

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = (
        blocks[3][:, [2 + i for i in exc4]] * C_tttt11 * (-1 / 4 + 3 / 4)
        + blocks[3][:, [10 + i for i in exc4]] * C_tttt11 * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [18 + i for i in exc4]] * C_ttttt1 * (-1 / 4 + 3 / 4)
        + blocks[3][:, [26 + i for i in exc4]] * C_ttttt1 * (3**0.5 / 4 + 3**0.5 / 4)
    )
    expect[3][:, 10:18] = (
        blocks[3][:, [2 + i for i in exc4]] * C_tttt11 * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [10 + i for i in exc4]] * C_tttt11 * (1 / 4 - 3 / 4)
        + blocks[3][:, [18 + i for i in exc4]] * C_ttttt1 * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [26 + i for i in exc4]] * C_ttttt1 * (1 / 4 - 3 / 4)
    )
    expect[3][:, 18:26] = (
        blocks[3][:, [2 + i for i in exc4]] * C_tttt1t * (-1 / 4 + 3 / 4)
        + blocks[3][:, [10 + i for i in exc4]] * C_tttt1t * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [18 + i for i in exc4]] * C_tttttt * (-1 / 4 + 3 / 4)
        + blocks[3][:, [26 + i for i in exc4]] * C_tttttt * (3**0.5 / 4 + 3**0.5 / 4)
    )
    expect[3][:, 26:34] = (
        blocks[3][:, [2 + i for i in exc4]] * C_tttt1t * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [10 + i for i in exc4]] * C_tttt1t * (1 / 4 - 3 / 4)
        + blocks[3][:, [18 + i for i in exc4]] * C_tttttt * (3**0.5 / 4 + 3**0.5 / 4)
        + blocks[3][:, [26 + i for i in exc4]] * C_tttttt * (1 / 4 - 3 / 4)
    )

    expect[4][:, :4] = blocks[4][:, exc] * R_1
    expect[4][:, 4:] = blocks[4][:, 4:]

    expect[5][:, :4] = blocks[5][:, exc] * R_tau
    expect[5][:, 4:] = blocks[5][:, 4:]

    expect[6][:, :] = blocks[6][:, exc4] * R_tau

    expect[7][:, :8] = blocks[7][:, exc4] * C_tttt11 + blocks[7][:, [8 + i for i in exc4]] * C_ttttt1
    expect[7][:, 8:] = blocks[7][:, exc4] * C_tttt1t + blocks[7][:, [8 + i for i in exc4]] * C_tttttt

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_domain = TensorProduct([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [5, 3, 4], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_c_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks, cross_check_single_c_symbol_tree_cols, apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    [c0, c1, c2, c3] = [np.array([i]) for i in range(4)]  # charges
    codomain = TensorProduct([s1, s1, s1])
    domain = TensorProduct([s1, s2, s2])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(6, 12), (16, 36), (5, 12), (5, 12)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    # SU(3)_3 R symbols
    # exchanging two 8s gives -1 except if they fuse to 8, then
    r8 = [-1j, 1j]  # for the two multiplicities
    # all other R symbols are trivial
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

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

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [1, 0, 2], [5, 4, 3], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 4 and 5 (in domain)
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

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

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_domain = TensorProduct([s2, s1, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [4, 5, 3], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 1 and 2 (in codomain)
    # we usually use the convention that in the codomain, the two final indices are f, e
    # the F symbols f2 and f1 are chosen such that we use the indices e, f
    # e.g. f2[e, f] = _f_symbol(10, 8, 8, 8, f, e)
    #      f1[e, f] = _f_symbol(8, 10, 8, 8, f, e)
    f2 = np.array([[-0.5, -(3**0.5) / 2], [3**0.5 / 2, -0.5]])
    f1 = f2.T
    C_sym = sym._c_symbol
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][0, :] = blocks[0][0, :] * r8[0]
    expect[0][1, :] = blocks[0][1, :] * r8[1]
    expect[0][[2, 3], :] = blocks[0][[3, 2], :]
    expect[0][4, :] = blocks[0][4, :] * -1
    expect[0][5, :] = blocks[0][5, :]

    v = [blocks[1][i, :] for i in range(7)]
    charges = [c0] + [c1] * 4 + [c2, c3]
    mul1 = [0] * 7
    mul2 = [0] * 7
    mul1[2], mul1[4] = 1, 1
    mul2[3], mul2[4] = 1, 1

    for i in range(7):
        w = [C_sym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        amplitudes = zero_block([7, backend.block_backend.get_shape(expect[1])[1]], Dtype.complex128)
        for j in range(7):
            amplitudes[j, :] = v[j] * w[j]
        expect[1][i, :] = backend.block_backend.sum(amplitudes, ax=0)

    expect[1][7, :] = blocks[1][9, :] * (f2[0, 0] * f2[0, 0] + f2[0, 1] * f2[1, 0]) + blocks[1][10, :] * (
        f2[1, 0] * f2[0, 0] + f2[1, 1] * f2[1, 0]
    )
    expect[1][8, :] = blocks[1][9, :] * (f2[0, 0] * f2[0, 1] + f2[0, 1] * f2[1, 1]) + blocks[1][10, :] * (
        f2[1, 0] * f2[0, 1] + f2[1, 1] * f2[1, 1]
    )
    expect[1][9, :] = blocks[1][7, :] * (f1[0, 0] * f1[0, 0] + f1[0, 1] * f1[1, 0]) + blocks[1][8, :] * (
        f1[1, 0] * f1[0, 0] + f1[1, 1] * f1[1, 0]
    )
    expect[1][10, :] = blocks[1][7, :] * (f1[0, 0] * f1[0, 1] + f1[0, 1] * f1[1, 1]) + blocks[1][8, :] * (
        f1[1, 0] * f1[0, 1] + f1[1, 1] * f1[1, 1]
    )
    expect[1][11, :] = blocks[1][11, :]
    expect[1][12, :] = blocks[1][12, :] * (f1[0, 0] * r8[0] * f2[0, 0] + f1[0, 1] * r8[1] * f2[1, 0]) + blocks[1][
        13, :
    ] * (f1[1, 0] * r8[0] * f2[0, 0] + f1[1, 1] * r8[1] * f2[1, 0])
    expect[1][13, :] = blocks[1][12, :] * (f1[0, 0] * r8[0] * f2[0, 1] + f1[0, 1] * r8[1] * f2[1, 1]) + blocks[1][
        13, :
    ] * (f1[1, 0] * r8[0] * f2[0, 1] + f1[1, 1] * r8[1] * f2[1, 1])
    expect[1][[14, 15], :] = blocks[1][[15, 14], :] * -1

    expect[2][0, :] = blocks[2][0, :] * (f1[0, 0] * r8[0] * f2[0, 0] + f1[0, 1] * r8[1] * f2[1, 0]) + blocks[2][
        1, :
    ] * (f1[1, 0] * r8[0] * f2[0, 0] + f1[1, 1] * r8[1] * f2[1, 0])
    expect[2][1, :] = blocks[2][0, :] * (f1[0, 0] * r8[0] * f2[0, 1] + f1[0, 1] * r8[1] * f2[1, 1]) + blocks[2][
        1, :
    ] * (f1[1, 0] * r8[0] * f2[0, 1] + f1[1, 1] * r8[1] * f2[1, 1])
    expect[2][[2, 3], :] = blocks[2][[3, 2], :] * -1
    expect[2][4, :] = blocks[2][4, :] * -1

    expect[3][0, :] = blocks[3][0, :] * (f2[0, 0] * r8[0] * f1[0, 0] + f2[0, 1] * r8[1] * f1[1, 0]) + blocks[3][
        1, :
    ] * (f2[1, 0] * r8[0] * f1[0, 0] + f2[1, 1] * r8[1] * f1[1, 0])
    expect[3][1, :] = blocks[3][0, :] * (f2[0, 0] * r8[0] * f1[0, 1] + f2[0, 1] * r8[1] * f1[1, 1]) + blocks[3][
        1, :
    ] * (f2[1, 0] * r8[0] * f1[0, 1] + f2[1, 1] * r8[1] * f1[1, 1])
    expect[3][[2, 3], :] = blocks[3][[3, 2], :] * -1
    expect[3][4, :] = blocks[3][4, :] * -1

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=1, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 2, 1], [5, 4, 3], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 3 and 4 (in domain)
    exc = [0, 2, 1, 3]
    exc4, exc8 = [4 + i for i in exc], [8 + i for i in exc]
    exc28, exc32 = [28 + i for i in exc], [32 + i for i in exc]
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * r8[0]
    expect[0][:, 4:8] = blocks[0][:, exc4] * r8[1]
    expect[0][:, 8:] = blocks[0][:, exc8] * -1

    v = [blocks[1][:, [4 * i + j for j in exc]] for i in range(7)]
    for i in range(7):
        w = [C_sym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        amplitudes = zero_block([backend.block_backend.get_shape(expect[1])[0], 4], Dtype.complex128)
        for j in range(7):
            amplitudes += v[j] * w[j]
        expect[1][:, 4 * i : 4 * (i + 1)] = amplitudes

    expect[1][:, 28:32] = blocks[1][:, exc28] * (f1[0, 0] * r8[0] * f2[0, 0] + f1[0, 1] * r8[1] * f2[1, 0]) + blocks[1][
        :, exc32
    ] * (f1[1, 0] * r8[0] * f2[0, 0] + f1[1, 1] * r8[1] * f2[1, 0])
    expect[1][:, 32:] = blocks[1][:, exc28] * (f1[0, 0] * r8[0] * f2[0, 1] + f1[0, 1] * r8[1] * f2[1, 1]) + blocks[1][
        :, exc32
    ] * (f1[1, 0] * r8[0] * f2[0, 1] + f1[1, 1] * r8[1] * f2[1, 1])

    expect[2][:, :4] = blocks[2][:, exc] * (f1[0, 0] * r8[0] * f2[0, 0] + f1[0, 1] * r8[1] * f2[1, 0]) + blocks[2][
        :, exc4
    ] * (f1[1, 0] * r8[0] * f2[0, 0] + f1[1, 1] * r8[1] * f2[1, 0])
    expect[2][:, 4:8] = blocks[2][:, exc] * (f1[0, 0] * r8[0] * f2[0, 1] + f1[0, 1] * r8[1] * f2[1, 1]) + blocks[2][
        :, exc4
    ] * (f1[1, 0] * r8[0] * f2[0, 1] + f1[1, 1] * r8[1] * f2[1, 1])
    expect[2][:, 8:] = blocks[2][:, exc8] * -1

    expect[3][:, :4] = blocks[3][:, exc] * (f2[0, 0] * r8[0] * f1[0, 0] + f2[0, 1] * r8[1] * f1[1, 0]) + blocks[3][
        :, exc4
    ] * (f2[1, 0] * r8[0] * f1[0, 0] + f2[1, 1] * r8[1] * f1[1, 0])
    expect[3][:, 4:8] = blocks[3][:, exc] * (f2[0, 0] * r8[0] * f1[0, 1] + f2[0, 1] * r8[1] * f1[1, 1]) + blocks[3][
        :, exc4
    ] * (f2[1, 0] * r8[0] * f1[0, 1] + f2[1, 1] * r8[1] * f1[1, 1])
    expect[3][:, 8:] = blocks[3][:, exc8] * -1

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [5, 3, 4], levels, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # braid 4 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=4, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    s3 = ElementarySpace(sym, [[0], [1]], [2, 3])  # 1 and tau

    # tensor with single leg in codomain; bend down
    codomain = TensorProduct([s2])
    domain = TensorProduct([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((1, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_codomain = TensorProduct([], symmetry=sym)
    expect_domain = TensorProduct([s2.dual])
    expect_tens = SymmetricTensor(data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [], [0], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with single leg in domain; bend up
    codomain = TensorProduct([], symmetry=sym)
    domain = TensorProduct([s3])

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((1, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect = [backend.block_backend.reshape(blocks[0], (2, 1))]
    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s3.dual])
    expect_domain = TensorProduct([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0], [], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # more complicated tensor
    codomain = TensorProduct([s2, s1, s1])
    domain = TensorProduct([s2, s1, s2])

    block_inds = np.array([[0, 0], [1, 1]])
    blocks = [
        backend.block_backend.random_uniform((2, 3), Dtype.complex128),
        backend.block_backend.random_uniform((3, 5), Dtype.complex128),
    ]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    # bend up
    phi = (1 + 5**0.5) / 2
    expect = [zero_block([5, 1], Dtype.complex128), zero_block([8, 2], Dtype.complex128)]

    expect[0][0, 0] = blocks[0][0, 1]  # (0, 0, 0) = (a, b, c) as in _b_symbol(a, b, c)
    expect[0][1, 0] = blocks[1][0, 3] * phi**0.5  # (0, 1, 1)
    expect[0][2, 0] = blocks[0][1, 1]  # (0, 0, 0)
    expect[0][3, 0] = blocks[1][1, 3] * phi**0.5  # (0, 1, 1)
    expect[0][4, 0] = blocks[1][2, 3] * phi**0.5  # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0]  # (1, 0, 1)
    expect[1][1, 0] = blocks[0][0, 0] * phi**-0.5  # (1, 1, 0)
    expect[1][2, 0] = blocks[1][0, 1]  # (1, 1, 1)
    expect[1][3, 0] = blocks[1][1, 0]  # (1, 0, 1)
    expect[1][4, 0] = blocks[1][2, 0]  # (1, 0, 1)
    expect[1][5, 0] = blocks[1][1, 1]  # (1, 1, 1)
    expect[1][6, 0] = blocks[0][1, 0] * phi**-0.5  # (1, 1, 0)
    expect[1][7, 0] = blocks[1][2, 1]  # (1, 1, 1)

    expect[1][0, 1] = blocks[1][0, 2]  # (1, 0, 1)
    expect[1][1, 1] = blocks[0][0, 2] * phi**-0.5  # (1, 1, 0)
    expect[1][2, 1] = blocks[1][0, 4]  # (1, 1, 1)
    expect[1][3, 1] = blocks[1][1, 2]  # (1, 0, 1)
    expect[1][4, 1] = blocks[1][2, 2]  # (1, 0, 1)
    expect[1][5, 1] = blocks[1][1, 4]  # (1, 1, 1)
    expect[1][6, 1] = blocks[0][1, 2] * phi**-0.5  # (1, 1, 0)
    expect[1][7, 1] = blocks[1][2, 4]  # (1, 1, 1)

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s2, s1, s1, s2.dual])
    expect_domain = TensorProduct([s2, s1])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2, 3], [5, 4], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # bend down
    expect = [zero_block([1, 5], Dtype.complex128), zero_block([2, 8], Dtype.complex128)]

    expect[0][0, 0] = blocks[1][1, 0] * phi**0.5  # (0, 1, 1)
    expect[0][0, 1] = blocks[1][1, 1] * phi**0.5  # (0, 1, 1)
    expect[0][0, 2] = blocks[1][1, 2] * phi**0.5  # (0, 1, 1)
    expect[0][0, 3] = blocks[1][1, 3] * phi**0.5  # (0, 1, 1)
    expect[0][0, 4] = blocks[1][1, 4] * phi**0.5  # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0]  # (1, 1, 1)
    expect[1][0, 1] = blocks[0][0, 0] * phi**-0.5  # (1, 1, 0)
    expect[1][0, 2] = blocks[1][0, 1]  # (1, 1, 1)
    expect[1][0, 3] = blocks[0][0, 1] * phi**-0.5  # (1, 1, 0)
    expect[1][0, 4] = blocks[1][0, 2]  # (1, 1, 1)
    expect[1][0, 5] = blocks[1][0, 3]  # (1, 1, 1)
    expect[1][0, 6] = blocks[0][0, 2] * phi**-0.5  # (1, 1, 0)
    expect[1][0, 7] = blocks[1][0, 4]  # (1, 1, 1)

    expect[1][1, 0] = blocks[1][2, 0]  # (1, 1, 1)
    expect[1][1, 1] = blocks[0][1, 0] * phi**-0.5  # (1, 1, 0)
    expect[1][1, 2] = blocks[1][2, 1]  # (1, 1, 1)
    expect[1][1, 3] = blocks[0][1, 1] * phi**-0.5  # (1, 1, 0)
    expect[1][1, 4] = blocks[1][2, 2]  # (1, 1, 1)
    expect[1][1, 5] = blocks[1][2, 3]  # (1, 1, 1)
    expect[1][1, 6] = blocks[0][1, 2] * phi**-0.5  # (1, 1, 0)
    expect[1][1, 7] = blocks[1][2, 4]  # (1, 1, 1)

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s2, s1])
    expect_domain = TensorProduct([s2, s1, s2, s1.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1], [5, 4, 3, 2], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [
        TensorProduct([], symmetry=sym),
        TensorProduct([s2]),
        TensorProduct([s3]),
        TensorProduct([s1, s3]),
        TensorProduct([s2, s3]),
        TensorProduct([s3, s1, s3, s2]),
    ]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    perm_axes = backend.block_backend.permute_axes
    reshape = backend.block_backend.reshape
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [1])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    s3 = ElementarySpace(sym, [[0, 0], [1, 1], [1, 2]], [1, 2, 2])  # (1, spin-0), (tau, spin-1/2) and (tau, spin-1)

    # tensor with two legs in domain; bend up
    codomain = TensorProduct([], symmetry=sym)
    domain = TensorProduct([s2, s3])

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((1, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [1, 1]])
    expect = [zero_block([1, 1], Dtype.complex128), zero_block([2, 2], Dtype.complex128)]

    expect[0][0, 0] = blocks[0][0, 0]  # ([0, 0], [0, 0], [0, 0]) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][:, :] = perm_axes(reshape(blocks[0][0, 1:], (2, 2)), [1, 0])
    expect[1][:, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * -1  # ([1, 1], [1, 1], [0, 0])

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s3.dual])
    expect_domain = TensorProduct([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0], [1], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with two legs in codomain, two leg in domain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2, s3])

    # charges [0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(2, 5), (2, 4), (2, 4), (3, 8), (2, 4), (2, 6), (2, 4), (2, 4)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 2]])
    expect = [zero_block([1, 86], Dtype.complex128)]

    expect[0][0, :2] = blocks[0][:, 0]
    expect[0][0, 18:26] = reshape(perm_axes(blocks[0][:, 1:], [1, 0]), (1, 8))
    expect[0][0, np.r_[:2, 18:26]] *= sym.inv_sqrt_qdim(np.array([1, 1])) * -1
    # ([1, 1], [1, 1], [0, 0])

    expect[0][0, 34:42] = reshape(perm_axes(blocks[1][:, :], [1, 0]), (1, 8))
    expect[0][0, 34:42] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 0])) * -1
    # ([1, 1], [1, 1], [1, 0])

    expect[0][0, 54:62] = reshape(perm_axes(blocks[2][:, :], [1, 0]), (1, 8))
    expect[0][0, 54:62] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 1])) * -1
    # ([1, 1], [1, 2], [0, 1])

    expect[0][0, 2:4] = blocks[3][0, :2]
    expect[0][0, 12:14] = blocks[3][0, 2:4]
    expect[0][0, 50:54] = blocks[3][0, 4:]
    # ([1, 1], [0, 0], [1, 1])

    expect[0][0, 4:8] = reshape(perm_axes(blocks[3][1:, :2], [1, 0]), (1, 4)) * -1
    expect[0][0, 14:18] = reshape(perm_axes(blocks[3][1:, 2:4], [1, 0]), (1, 4)) * -1
    expect[0][0, 70:78] = reshape(perm_axes(blocks[3][1:, 4:], [1, 0]), (1, 8)) * -1
    # ([1, 1], [1, 2], [1, 1])

    expect[0][0, 26:34] = reshape(perm_axes(blocks[4][:, :], [1, 0]), (1, 8))
    expect[0][0, 26:34] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 2]))
    # ([1, 1], [1, 1], [0, 2])

    expect[0][0, 8:12] = reshape(perm_axes(blocks[5][:, :2], [1, 0]), (1, 4))
    expect[0][0, 42:50] = reshape(perm_axes(blocks[5][:, 2:], [1, 0]), (1, 8))
    expect[0][0, np.r_[8:12, 42:50]] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 2]))
    # ([1, 1], [1, 1], [1, 2])

    expect[0][0, 62:70] = reshape(perm_axes(blocks[6][:, :], [1, 0]), (1, 8))
    expect[0][0, 62:70] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 3]))
    # ([1, 1], [1, 2], [0, 3])

    expect[0][0, 78:] = reshape(perm_axes(blocks[7][:, :], [1, 0]), (1, 8))
    expect[0][0, 78:] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 3]))
    # ([1, 1], [1, 2], [1, 3])

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s2, s3, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0], [3, 2, 1], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # similar tensor, replace one sector with its dual (Frobenius-Schur is now relevant); bend up
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2, s3.dual])

    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [3, 1]])
    expect = [zero_block([12, 1], Dtype.complex128), zero_block([37, 2], Dtype.complex128)]

    expect[0][2:4, 0] = blocks[0][:, 0]  # ([0, 0], [0, 0], [0, 0])

    expect[1][3:7, :] = reshape(perm_axes(reshape(blocks[0][:, 1:], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][3:7, :] *= sym.inv_sqrt_qdim(np.array([1, 1]))  # ([1, 1], [1, 1], [0, 0])

    expect[1][11:15, :] = reshape(perm_axes(reshape(blocks[1][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][11:15, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 0]))
    # ([1, 1], [1, 1], [1, 0])

    expect[1][21:25, :] = reshape(perm_axes(reshape(blocks[2][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][21:25, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 1])) * -1
    # ([1, 1], [1, 2], [0, 1])

    expect[0][:2, 0] = blocks[3][0, :2]
    expect[0][8:, :] = reshape(blocks[3][1:, :2], (4, 1))
    expect[0][np.r_[:2, 8:12], 0] *= sym.sqrt_qdim(np.array([1, 1])) * -1
    # ([0, 0], [1, 1], [1, 1])

    expect[1][0, :] = blocks[3][0, 2:4]
    expect[1][19:21, :] = blocks[3][1:, 2:4]
    # ([1, 1], [0, 0], [1, 1])

    expect[1][1:3, :] = perm_axes(reshape(blocks[3][0, 4:], (2, 2)), [1, 0]) * -1
    expect[1][29:33, :] = reshape(perm_axes(reshape(blocks[3][1:, 4:], (2, 2, 2)), [0, 2, 1]), (4, 2)) * -1
    # ([1, 1], [1, 2], [1, 1])

    expect[1][7:11, :] = reshape(perm_axes(reshape(blocks[4][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][7:11, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 2])) * -1
    # ([1, 1], [1, 1], [0, 2])

    expect[0][4:8, :] = reshape(blocks[5][:, :2], (4, 1))
    expect[0][4:8, :] *= sym.sqrt_qdim(np.array([1, 2]))
    # ([0, 0], [1, 2], [1, 2])

    expect[1][15:19, :] = reshape(perm_axes(reshape(blocks[5][:, 2:], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][15:19, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 2])) * -1
    # ([1, 1], [1, 1], [1, 2])

    expect[1][25:29, :] = reshape(perm_axes(reshape(blocks[6][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][25:29, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 3]))
    # ([1, 1], [1, 2], [0, 3])

    expect[1][33:, :] = reshape(perm_axes(reshape(blocks[7][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][33:, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 3]))
    # ([1, 1], [1, 2], [1, 3])

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1, s3, s3])
    expect_domain = TensorProduct([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [3], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [
        TensorProduct([], symmetry=sym),
        TensorProduct([s2]),
        TensorProduct([s3.dual]),
        TensorProduct([s1, s3]),
        TensorProduct([s2, s3.dual]),
        TensorProduct([s1, s3, s2.dual]),
    ]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    perm_axes = backend.block_backend.permute_axes
    reshape = backend.block_backend.reshape
    zero_block = backend.block_backend.zeros
    eps = 1.0e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    s3 = ElementarySpace(sym, [[0], [1], [3]], [1, 2, 3])  # 1, 8, 10-
    qdim8 = sym.sqrt_qdim(np.array([1]))  # sqrt of qdim of charge 8
    # when multiplying with qdims (from the b symbols), only 8 is relevant since all other qdim are 1
    # the b symbols are diagonal in the multiplicity index

    # tensor with two legs in codomain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((5, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [zero_block([1, 2], Dtype.complex128), zero_block([1, 3], Dtype.complex128)]

    expect[0][0, :] = blocks[0][:2, 0] / qdim8  # (8, 8, 1) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][0, :] = blocks[0][2:, 0]  # (10, 10-, 1)

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0], [1], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with two legs in codomain, one leg in domain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2])

    block_inds = np.array([[1, 0]])
    blocks = [backend.block_backend.random_uniform((10, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [zero_block([1, 16], Dtype.complex128), zero_block([1, 4], Dtype.complex128)]

    expect[0][0, :2] = blocks[0][0, :]  # (8, 1, 8)
    expect[0][0, 2:6] = reshape(perm_axes(blocks[0][1:3, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 6:10] = reshape(perm_axes(blocks[0][3:5, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 10:] = reshape(perm_axes(blocks[0][5:8, :], [1, 0]), (1, 6)) * -1  # (8, 10-, 8)

    expect[1][0, :] = reshape(perm_axes(blocks[0][8:, :], [1, 0]), (1, 4)) * qdim8 * -1  # (10, 8, 8)

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s2, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0], [2, 1], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # same tensor, bend up
    expect_block_inds = np.array([[0, 0]])
    expect = [zero_block([20, 1], Dtype.complex128)]

    expect[0][:2, 0] = blocks[0][0, :] * qdim8  # (1, 8, 8)
    expect[0][2:6, :] = reshape(blocks[0][1:3, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][6:10, :] = reshape(blocks[0][3:5, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][10:16, :] = reshape(blocks[0][5:8, :], (6, 1)) * qdim8  # (1, 8, 8)
    expect[0][16:, :] = reshape(blocks[0][8:, :], (4, 1)) * qdim8  # (1, 8, 8)

    expect_data = backends.FusionTreeData(
        expect_block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1, s3, s2.dual])
    expect_domain = TensorProduct([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1, 2], [], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # more complicated tensor, bend down
    codomain = TensorProduct([s1, s2, s2])
    domain = TensorProduct([s2, s3])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(12, 4), (36, 16), (12, 4), (12, 4)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128, device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_shapes = [(2, 32), (6, 88), (2, 32), (2, 32)]
    expect = [zero_block(shp, Dtype.complex128) for shp in expect_shapes]

    expect[0][:, :4] = reshape(perm_axes(reshape(blocks[1][:4, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[0][:, 4:12] = reshape(perm_axes(reshape(blocks[1][:4, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[0][:, 12:20] = reshape(perm_axes(reshape(blocks[1][:4, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[0][:, 20:32] = reshape(perm_axes(reshape(blocks[1][:4, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[0][:, :] *= qdim8  # (1, 8, 8)

    expect[1][:2, :4] = reshape(perm_axes(reshape(blocks[1][4:8, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][:2, 4:8] = reshape(perm_axes(reshape(blocks[1][12:16, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][2:4, :4] = reshape(perm_axes(reshape(blocks[1][8:12, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][2:4, 4:8] = reshape(perm_axes(reshape(blocks[1][16:20, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][4:, :4] = reshape(perm_axes(reshape(blocks[1][28:32, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][4:, 4:8] = reshape(perm_axes(reshape(blocks[1][32:, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    # (8, 8, 8)

    expect[1][:2, 8:16] = reshape(perm_axes(reshape(blocks[0][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 8:16] = reshape(perm_axes(reshape(blocks[0][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 8:16] = reshape(perm_axes(reshape(blocks[0][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 8:16] /= qdim8  # (1, 8, 1)

    expect[1][:2, 16:24] = reshape(perm_axes(reshape(blocks[1][4:8, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 16:24] = reshape(perm_axes(reshape(blocks[1][8:12, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 16:24] = reshape(perm_axes(reshape(blocks[1][28:32, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 24:32] = reshape(perm_axes(reshape(blocks[1][4:8, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 24:32] = reshape(perm_axes(reshape(blocks[1][8:12, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 24:32] = reshape(perm_axes(reshape(blocks[1][28:32, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 32:40] = reshape(perm_axes(reshape(blocks[1][12:16, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 32:40] = reshape(perm_axes(reshape(blocks[1][16:20, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 32:40] = reshape(perm_axes(reshape(blocks[1][32:, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 40:48] = reshape(perm_axes(reshape(blocks[1][12:16, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 40:48] = reshape(perm_axes(reshape(blocks[1][16:20, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 40:48] = reshape(perm_axes(reshape(blocks[1][32:, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 48:56] = reshape(perm_axes(reshape(blocks[2][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 48:56] = reshape(perm_axes(reshape(blocks[2][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 48:56] = reshape(perm_axes(reshape(blocks[2][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 48:56] *= -1 / qdim8  # (8, 8, 10)

    expect[1][:2, 56:64] = reshape(perm_axes(reshape(blocks[3][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 56:64] = reshape(perm_axes(reshape(blocks[3][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 56:64] = reshape(perm_axes(reshape(blocks[3][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 56:64] *= -1 / qdim8  # (8, 8, 10-)

    expect[1][:2, 64:76] = reshape(perm_axes(reshape(blocks[1][4:8, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][:2, 76:] = reshape(perm_axes(reshape(blocks[1][12:16, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][2:4, 64:76] = reshape(perm_axes(reshape(blocks[1][8:12, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][2:4, 76:] = reshape(perm_axes(reshape(blocks[1][16:20, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][4:, 64:76] = reshape(perm_axes(reshape(blocks[1][28:32, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][4:, 76:] = reshape(perm_axes(reshape(blocks[1][32:, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    # (8, 8, 8)

    expect[2][:, :4] = reshape(perm_axes(reshape(blocks[1][20:24, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[2][:, 4:12] = reshape(perm_axes(reshape(blocks[1][20:24, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[2][:, 12:20] = reshape(perm_axes(reshape(blocks[1][20:24, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[2][:, 20:32] = reshape(perm_axes(reshape(blocks[1][20:24, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[2][:, :] *= qdim8 * -1  # (10, 8, 8)

    expect[3][:, :4] = reshape(perm_axes(reshape(blocks[1][24:28, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[3][:, 4:12] = reshape(perm_axes(reshape(blocks[1][24:28, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[3][:, 12:20] = reshape(perm_axes(reshape(blocks[1][24:28, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[3][:, 20:32] = reshape(perm_axes(reshape(blocks[1][24:28, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[3][:, :] *= qdim8 * -1  # (10-, 8, 8)

    expect_data = backends.FusionTreeData(
        block_inds, expect, Dtype.complex128, device=backend.block_backend.default_device
    )
    expect_codomain = TensorProduct([s1, s2])
    expect_domain = TensorProduct([s2, s3, s2.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    new_tens = permute_legs(tens, [0, 1], [4, 3, 2], None, bend_right=True)
    assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [
        TensorProduct([], symmetry=sym),
        TensorProduct([s2]),
        TensorProduct([s3.dual]),
        TensorProduct([s1, s3]),
        TensorProduct([s2, s3.dual]),
        TensorProduct([s1, s3, s2.dual]),
    ]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.parametrize(
    'symmetry',
    [
        fibonacci_anyon_category,
        ising_anyon_category,
        SU2_kAnyonCategory(4),
        SU2_kAnyonCategory(5) * u1_symmetry,
        SU2Symmetry() * ising_anyon_category,
        SU3_3AnyonCategory() * u1_symmetry,
        fibonacci_anyon_category * z5_symmetry,
    ],
)
def test_ftb_transpose(symmetry: Symmetry, block_backend: str, np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    num_codom_legs, num_dom_legs = np_random.integers(low=2, high=4, size=2)
    tens = random_tensor(
        symmetry=symmetry,
        codomain=int(num_codom_legs),
        domain=int(num_dom_legs),
        backend=backend,
        max_multiplicity=1,
        cls=SymmetricTensor,
        np_random=np_random,
    )

    res = transpose(tens)
    res.test_sanity()
    for over in [True, False]:
        for twist_codomain in [True, False]:
            print(f'\n\n{over=}  {twist_codomain=}')
            other = cross_check_transpose(tens, over=over, twist_codomain=twist_codomain)
            assert_tensors_almost_equal(res, other, rtol=1e-13, atol=1e-13)

    double_transp = transpose(res)
    double_transp.test_sanity()
    assert_tensors_almost_equal(double_transp, tens)


def test_permute_legs_instructions():
    # Note that there is no single unique correct sequence of instructions.
    # Depending on the implementation of the algorithm that comes up with instructions,
    # the instructions might change.
    # In this test, we guaranteed that the expected instruction lists that are hard coded
    # are correct (by manually verifying).
    # If there is a mismatch in the future, this does not necessarily mean that either one
    # is incorrect!

    # Note that we also verify that the legs are permuted to the correct places
    # during `permute_legs_instructions`, but not if braid chiralities are correct.
    # That we verify manually only.

    codomain = 6
    domain = 6

    num_legs = codomain + domain
    original_codomain_idcs = np.arange(codomain)
    original_domain_idcs = num_legs - 1 - np.arange(domain)
    unspecified_levels = [None] * (codomain + domain)

    # =============================================
    # 0) do nothing
    # =============================================
    instructions0 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=original_codomain_idcs,
        domain_idcs=original_domain_idcs,
        levels=unspecified_levels,
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    assert list(instructions0) == []

    # =============================================
    # 1) codomain permutation only
    # =============================================
    codomain_idcs1 = np.array([1, 0, 4, 3, 5, 2])
    levels1 = [1, 0, 3, 2, 5, 4, *range(codomain, num_legs)]
    #
    instructions1 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=codomain_idcs1,
        domain_idcs=original_domain_idcs,
        levels=levels1,
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    instructions1 = list(instructions1)
    # to derive the expected braids, you need to draw the braiding and derive them manually
    # note that the order of braids is in general not unique, and we need to make the same conventional
    # choice as the implementation: first braid the leg that ends up at codomain[0] and so on
    expect_instructions1 = [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in [(0, True), (3, False), (2, False), (3, True), (4, False)]
    ]
    assert instructions1 == expect_instructions1

    # =============================================
    # 2) domain permutation only
    # =============================================
    domain_idcs2 = [10, 8, 9, 11, 7, 6]
    levels2 = [*range(6), 7, 9, 8, 6, 10, 11]
    #
    instructions2 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=original_codomain_idcs,
        domain_idcs=domain_idcs2,
        levels=levels2,
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    instructions2 = list(instructions2)
    #
    expect_instructions2 = [
        fusion_tree_backend.BraidInstruction(codomain=False, idx=j, overbraid=overbraid)
        for j, overbraid in [(0, False), (1, False), (2, False), (1, True)]
    ]
    assert instructions2 == expect_instructions2

    # =============================================
    # 3) domain and codomain permutations, but no bends
    # =============================================
    # just do the braids of the two cases above
    levels3 = levels1[:6] + levels2[6:]
    instructions3 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=codomain_idcs1,
        domain_idcs=domain_idcs2,
        levels=levels3,
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    instructions3 = list(instructions3)
    expect_instructions3 = expect_instructions2 + expect_instructions1
    assert instructions3 == expect_instructions3

    # =============================================
    # 4) up bends only (levels=None)
    # =============================================
    num = 3
    instructions4 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[*range(6 + num)],
        domain_idcs=[*reversed(range(6 + num, num_legs))],
        levels=unspecified_levels,
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    assert list(instructions4) == [fusion_tree_backend.BendInstruction(bend_down=True)] * num

    # =============================================
    # 5) down bends only
    # =============================================
    num = 2
    instructions5 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[*range(6 - num)],
        domain_idcs=[*reversed(range(6 - num, num_legs))],
        levels=[*range(12)],
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    assert list(instructions5) == [fusion_tree_backend.BendInstruction(bend_down=False)] * num

    # =============================================
    # 6) codomain perm and bends
    # =============================================
    instructions6 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[6, 0, 1, 5, 3, 7, 4, 2],
        domain_idcs=[11, 10, 9, 8],
        levels=[8, 2, 11, 7, 1, 4, 6, 9, 0, 5, 10, 3],
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    expect_instructions6 = [fusion_tree_backend.BendInstruction(bend_down=True)] * 2
    expect_instructions6 += [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in [
            (5, False),
            (4, False),
            (3, True),
            (2, True),
            (1, False),
            (0, True),
            (5, False),
            (4, True),
            (3, True),
            (4, True),
            (6, False),
            (5, True),
            (6, True),
        ]
    ]
    assert list(instructions6) == expect_instructions6

    # =============================================
    # domain perm and bends
    # =============================================
    instructions7 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[0, 1, 2, 3],
        domain_idcs=[7, 9, 5, 11, 10, 4, 8, 6],
        levels=[5, 8, 7, 6, 4, 3, 0, 1, 2, 10, 9, 11],
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    instructions7 = list(instructions7)
    expect_instructions7 = [fusion_tree_backend.BendInstruction(bend_down=False)] * 2
    expect_instructions7 += [
        fusion_tree_backend.BraidInstruction(codomain=False, idx=j, overbraid=overbraid)
        for j, overbraid in zip(
            [5, 6, 3, 4, 5, 1, 2, 3, 0, 1, 2, 0],
            [True, True, False, True, True, True, False, False, False, False, False, False],
        )
    ]
    assert list(instructions7) == expect_instructions7

    # =============================================
    # general case with right bends
    # =============================================
    instructions8 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[6, 3, 9, 5, 10, 4, 2],
        domain_idcs=[8, 7, 1, 0, 11],
        levels=[1, 2, 0, 4, 10, 3, 8, 11, 5, 7, 6, 9],
        has_symmetric_braid=False,
        bend_right=[True] * (codomain + domain),
    )
    instructions8 = list(instructions8)
    expect_instructions8 = [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in zip([1, 2, 3, 4, 0, 1, 2, 3], [True, False, False, False, True, False, False, False])
    ]
    expect_instructions8 += [fusion_tree_backend.BendInstruction(bend_down=False)] * 2
    expect_instructions8 += [
        fusion_tree_backend.BraidInstruction(codomain=False, idx=j, overbraid=overbraid)
        for j, overbraid in zip(
            [5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3],
            [False, False, False, True, False, False, False, True, False, False, False, True, False, False],
        )
    ]
    expect_instructions8 += [fusion_tree_backend.BendInstruction(bend_down=True)] * 3
    expect_instructions8 += [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in zip(
            [3, 2, 1, 0, 1, 4, 3, 2, 4, 3, 5, 4, 5],
            [False, True, False, False, False, False, True, False, True, False, True, False, False],
        )
    ]
    assert instructions8 == expect_instructions8

    # =============================================
    # general case with left and right bends
    # =============================================
    instructions9 = permute_legs_instructions(
        codomain,
        domain,
        codomain_idcs=[6, 3, 9, 5, 10, 4, 2],
        domain_idcs=[8, 7, 1, 0, 11],
        levels=[1, 2, 0, 4, 10, 3, 8, 11, 5, 7, 6, 9],
        has_symmetric_braid=False,
        bend_right=[True, False, None, None, None, None, True, None, None, False, True, None],
    )
    instructions9 = list(instructions9)
    expect_instructions9 = [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in zip([0, 1, 2, 3, 4], [False, True, False, False, False])
    ]
    expect_instructions9 += [fusion_tree_backend.BendInstruction(bend_down=False)]
    expect_instructions9 += [fusion_tree_backend.TwistInstruction(codomain=True, idcs=[0], overtwist=True)]
    expect_instructions9 += [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in zip([0, 1, 2, 3], [True, True, True, True])
    ]
    expect_instructions9 += [fusion_tree_backend.BendInstruction(bend_down=False)]
    expect_instructions9 += [
        fusion_tree_backend.BraidInstruction(codomain=False, idx=j, overbraid=overbraid)
        for j, overbraid in zip(
            [6, 5, 4, 3, 2, 1, 0, 6, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
            ],
        )
    ]
    expect_instructions9 += [fusion_tree_backend.BendInstruction(bend_down=True)] * 2
    expect_instructions9 += [fusion_tree_backend.TwistInstruction(codomain=False, idcs=[0], overtwist=False)]
    expect_instructions9 += [
        fusion_tree_backend.BraidInstruction(codomain=False, idx=j, overbraid=overbraid)
        for j, overbraid in zip([0, 1, 2, 3, 4], [False] * 5)
    ]
    expect_instructions9 += [fusion_tree_backend.BendInstruction(bend_down=True)]
    expect_instructions9 += [
        fusion_tree_backend.BraidInstruction(codomain=True, idx=j, overbraid=overbraid)
        for j, overbraid in zip(
            [5, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 2, 1, 4, 3, 5, 4, 5],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
            ],
        )
    ]
    assert instructions9 == expect_instructions9


# TODO make sure that there is a case that fails when not complex conjugating
# the b symbols for domain trees
# (all tests currently seem to pass irrespective of complex conjugation)
@pytest.mark.parametrize(
    'symmetry',
    [
        fibonacci_anyon_category,
        ising_anyon_category,
        SU2_kAnyonCategory(4),
        SU2_kAnyonCategory(5) * u1_symmetry,
        SU2Symmetry() * ising_anyon_category,
        SU3_3AnyonCategory() * u1_symmetry,
        fibonacci_anyon_category * z5_symmetry,
    ],
)
def test_ftb_partial_trace(symmetry: Symmetry, block_backend: str, np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    num_codom_legs, num_dom_legs = np_random.integers(low=2, high=4, size=2)
    num_legs = num_codom_legs + num_dom_legs
    pairs = [[[0, 1]], [[0, 3]], [[1, 3]], [[0, 2]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]]
    if num_legs >= 5:
        pairs.extend([[[0, 4]], [[1, 4], [0, 2]], [[0, 4], [1, 3]]])
    pairs = pairs[np_random.choice(len(pairs))]

    spaces = []
    idcs1 = [p[0] for p in pairs]
    idcs2 = [p[1] for p in pairs]
    for i in range(num_legs):
        if i in idcs2:
            idx = idcs2.index(i)
            if i >= num_codom_legs and idcs1[idx] < num_codom_legs:
                spaces.append(spaces[idcs1[idx]])
            else:
                spaces.append(spaces[idcs1[idx]].dual)
        else:
            spaces.append(random_ElementarySpace(symmetry=symmetry, np_random=np_random))

    levels = list(np_random.permutation(num_legs - len(pairs)))
    for idx in np.argsort(idcs2):
        level = levels[idcs1[idx]]
        levels = [l + 1 if l > level else l for l in levels]
        levels.insert(idcs2[idx], level + 1)

    codomain = spaces[:num_codom_legs]
    domain = spaces[num_codom_legs:][::-1]
    tens = random_tensor(
        symmetry=symmetry,
        codomain=codomain,
        domain=domain,
        backend=backend,
        max_multiplicity=3,
        cls=SymmetricTensor,
        np_random=np_random,
    )

    data1, codom1, dom1 = tens.backend.partial_trace(tens, pairs=pairs, levels=levels)
    data2, codom2, dom2 = cross_check_partial_trace(tens, pairs=pairs, levels=levels)
    tens1 = SymmetricTensor(data1, codom1, dom1, backend=tens.backend)
    tens2 = SymmetricTensor(data2, codom2, dom2, backend=tens.backend)
    assert_tensors_almost_equal(tens1, tens2, 1e-13, 1e-13)


# HELPER FUNCTIONS FOR THE TESTS


def permute_legs_instructions(
    num_codomain_legs: int,
    num_domain_legs: int,
    codomain_idcs: list[int],
    domain_idcs: list[int],
    levels: list[int | None],
    bend_right: list[bool | None],
    has_symmetric_braid: bool,
) -> list[fusion_tree_backend.Instruction]:
    h = fusion_tree_backend.PermuteLegsInstructionEngine(
        num_codomain_legs=num_codomain_legs,
        num_domain_legs=num_domain_legs,
        codomain_idcs=codomain_idcs,
        domain_idcs=domain_idcs,
        levels=levels,
        bend_right=bend_right,
        has_symmetric_braid=has_symmetric_braid,
    )
    instructions = h.evaluate_instructions()
    h.verify(num_codomain_legs, num_domain_legs, codomain_idcs, domain_idcs)
    return instructions


def apply_single_b_symbol(
    ten: SymmetricTensor, bend_down: bool
) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Use the implementation of b symbols using `TreeMappingDicts` to return the action
    of a single b symbol in the same format (input and output) as the cross check
    implementation. This is of course inefficient usage of this implementation but a
    necessity in order to use the structure of the already implemented tests.
    """
    instruction = fusion_tree_backend.BendInstruction(bend_down=bend_down)
    mapping = fusion_tree_backend.TreePairMapping.from_instructions(
        [instruction], codomain=ten.codomain, domain=ten.domain
    )
    if bend_down:
        codomain_idcs = [*range(ten.num_codomain_legs + 1)]
        domain_idcs = [*reversed(range(ten.num_codomain_legs + 1, ten.num_legs))]
        codomain_factors = [*ten.codomain.factors, ten.domain[-1].dual]
        domain_factors = ten.domain.factors[:-1]
    else:
        codomain_idcs = [*range(ten.num_codomain_legs - 1)]
        domain_idcs = [*reversed(range(ten.num_codomain_legs - 1, ten.num_legs))]
        codomain_factors = ten.codomain.factors[:-1]
        domain_factors = [*ten.domain.factors, ten.codomain.factors[-1].dual]
    new_codomain = TensorProduct(codomain_factors, symmetry=ten.symmetry)
    new_domain = TensorProduct(domain_factors, symmetry=ten.symmetry)
    data = mapping.transform_tensor(
        data=ten.data,
        codomain=ten.codomain,
        domain=ten.domain,
        new_codomain=new_codomain,
        new_domain=new_domain,
        codomain_idcs=codomain_idcs,
        domain_idcs=domain_idcs,
        block_backend=ten.backend.block_backend,
    )
    return data, new_codomain, new_domain


def apply_single_c_symbol(
    ten: SymmetricTensor, leg: int | str, levels: list[int]
) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Use the implementation of c symbols using `TreeMappingDicts` to return the action
    of a single c symbol in the same format (input and output) as the cross check
    implementations. This is of course inefficient usage of this implementation but a
    necessity in order to use the structure of the already implemented tests.
    """
    assert isinstance(ten.backend, fusion_tree_backend.FusionTreeBackend)
    in_domain, idx, leg = ten._parse_leg_idx(leg)
    if in_domain:
        idx -= 1
    overbraid = levels[leg] > levels[leg + 1]
    instruction = fusion_tree_backend.BraidInstruction(codomain=not in_domain, idx=idx, overbraid=overbraid)
    mapping = fusion_tree_backend.FactorizedTreeMapping.from_instructions(
        [instruction], codomain=ten.codomain, domain=ten.domain, block_inds=ten.data.block_inds
    )
    if in_domain:
        new_codomain = ten.codomain
        factors = ten.domain.factors[:]
        factors[idx], factors[idx + 1] = factors[idx + 1], factors[idx]
        new_domain = TensorProduct(factors)
        codomain_idcs = [*range(ten.num_codomain_legs)]
        domain_idcs = [*reversed(range(ten.num_codomain_legs, ten.num_legs))]
        domain_idcs[idx], domain_idcs[idx + 1] = domain_idcs[idx + 1], domain_idcs[idx]
    else:
        factors = ten.codomain.factors[:]
        factors[idx], factors[idx + 1] = factors[idx + 1], factors[idx]
        new_codomain = TensorProduct(factors)
        new_domain = ten.domain
        codomain_idcs = [*range(idx), idx + 1, idx, *range(idx + 2, ten.num_codomain_legs)]
        domain_idcs = [*reversed(range(ten.num_codomain_legs, ten.num_legs))]
    data = mapping.transform_tensor(
        ten.data,
        codomain=ten.codomain,
        domain=ten.domain,
        new_codomain=new_codomain,
        new_domain=new_domain,
        codomain_idcs=codomain_idcs,
        domain_idcs=domain_idcs,
        block_backend=ten.backend.block_backend,
    )
    return data, new_codomain, new_domain


def assert_bending_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable], eps: float):
    """Check that when rescaling and bending legs, it does not matter whether one first
    performs the rescaling and then the bending process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the function directly acting on tensors;
    this function is tested elsewhere.
    """
    bends = [True, False]
    for bend_down in bends:
        if a.num_codomain_legs == 0 and not bend_down:
            continue
        elif a.num_domain_legs == 0 and bend_down:
            continue

        for func in funcs:
            if bend_down:
                num_leg = a.num_codomain_legs - 1
                leg = a.legs[num_leg]
            else:
                num_leg = a.num_codomain_legs
                leg = a.legs[num_leg].dual

            diag = DiagonalTensor.from_random_uniform(leg, backend=a.backend, dtype=a.dtype)
            new_a = a.copy()
            new_a2 = a.copy()

            # apply scale_axis first
            new_a.data = new_a.backend.scale_axis(new_a, diag, num_leg)
            new_data, new_codomain, new_domain = func(new_a, bend_down)
            new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a.backend)

            # bend first
            new_data, new_codomain, new_domain = func(new_a2, bend_down)
            new_a2 = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a2.backend)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag, num_leg)

            assert_tensors_almost_equal(new_a, new_a2, eps, eps)


def assert_bending_up_and_down_trivial(
    codomains: list[TensorProduct],
    domains: list[TensorProduct],
    funcs: list[Callable],
    backend: backends.TensorBackend,
    multiple: bool,
    eps: float,
):
    """Check that bending a leg up and down (or down and up) is trivial. All given codomains are combined with all
    given domains to construct random tensors for which the identities are checked. All codomains and domains must
    have the same symmetry; this is not explicitly checked.
    If `multiple == True`, the identity is also checked for bending multiple (i.e. more than one) legs up and down
    up to the maximum number possible. Otherwise, the identity is only checked for a single leg.
    """
    for codomain in codomains:
        for domain in domains:
            tens = SymmetricTensor.from_random_uniform(codomain, domain, backend=backend)
            if len(tens.data.blocks) == 0:  # trivial tensors
                continue

            bending = []
            for i in range(tens.num_domain_legs):
                bends = [True] * (i + 1) + [False] * (i + 1)
                bending.append(bends)
                if not multiple:
                    break
            for i in range(tens.num_codomain_legs):
                bends = [False] * (i + 1) + [True] * (i + 1)
                bending.append(bends)
                if not multiple:
                    break

            for func in funcs:
                for bends in bending:
                    new_tens = tens.copy()
                    for bend in bends:
                        new_data, new_codomain, new_domain = func(new_tens, bend)
                        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

                    assert_tensors_almost_equal(new_tens, tens, eps, eps)


def assert_braiding_and_scale_axis_commutation(
    a: SymmetricTensor, funcs: list[Callable], levels: list[int], eps: float
):
    """Check that when rescaling and exchanging legs, it does not matter whether one first
    performs the rescaling and then the exchange process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the function directly acting on tensors;
    this function is tested elsewhere.
    """
    for func in funcs:
        for leg in range(a.num_legs - 1):
            if leg == a.num_codomain_legs - 1:
                continue

            legs = [a.legs[leg], a.legs[leg + 1]]
            if leg > a.num_codomain_legs - 1:  # in domain
                legs = [leg_.dual for leg_ in legs]
            diag_left = DiagonalTensor.from_random_uniform(legs[0], backend=a.backend, dtype=a.dtype)
            diag_right = DiagonalTensor.from_random_uniform(legs[1], backend=a.backend, dtype=a.dtype)
            new_a = a.copy()
            new_a2 = a.copy()

            # apply scale_axis first
            new_a.data = new_a.backend.scale_axis(new_a, diag_left, leg)
            new_a.data = new_a.backend.scale_axis(new_a, diag_right, leg + 1)
            new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
            new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a.backend)

            # exchange first
            new_data, new_codomain, new_domain = func(new_a2, leg=leg, levels=levels)
            new_a2 = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a2.backend)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag_left, leg + 1)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag_right, leg)

            assert_tensors_almost_equal(new_a, new_a2, eps, eps)


def assert_clockwise_counterclockwise_trivial(a: SymmetricTensor, funcs: list[Callable], levels: list[int], eps: float):
    """Check that braiding a pair of neighboring legs clockwise and then anti-clockwise
    (or vice versa) leaves the tensor invariant. `levels` specifies whether the first
    exchange is clockwise or anti-clockwise (the second is the opposite).
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
    for func in funcs:
        for leg in range(a.num_legs - 1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            new_levels = levels[:]
            for _ in range(2):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=new_levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)
                new_levels[leg : leg + 2] = new_levels[leg : leg + 2][::-1]

            assert_tensors_almost_equal(new_a, a, eps, eps)


def assert_clockwise_counterclockwise_trivial_long_range(
    a: SymmetricTensor, eps: float, np_random: np.random.Generator
):
    """Same as `assert_clockwise_counterclockwise_trivial` with the difference that a random
    sequence of exchanges and bends is chosen.
    """
    levels = list(np_random.permutation(a.num_legs))
    permutation = list(np_random.permutation(a.num_legs))
    inv_permutation = [permutation.index(i) for i in range(a.num_legs)]
    inv_levels = [levels[i] for i in permutation]
    num_codomain = np.random.randint(a.num_legs + 1)

    new_a = permute_legs(a, permutation[:num_codomain], permutation[num_codomain:][::-1], levels, bend_right=True)
    new_a = permute_legs(
        new_a,
        inv_permutation[: a.num_codomain_legs],
        inv_permutation[a.num_codomain_legs :][::-1],
        inv_levels,
        bend_right=True,
    )
    assert_tensors_almost_equal(new_a, a, eps, eps)


def assert_repeated_braids_trivial(
    a: SymmetricTensor, funcs: list[Callable], levels: list[int], repeat: int, eps: float
):
    """Check that repeatedly braiding two neighboring legs often enough leaves the tensor
    invariant. The number of required repetitions must be chosen to be even (legs must be
    at their initial positions again) and such that all (relevant) r symbols to the power
    of the repetitions is identity. `levels` specifies whether the repeated exchange is
    always clockwise or anti-clockwise.
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
    for func in funcs:
        for leg in range(a.num_legs - 1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            for _ in range(repeat):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

            assert_tensors_almost_equal(new_a, a, eps, eps)


# FUNCTIONS FOR CROSS CHECKING THE COMPUTATION OF THE ACTION OF B AND C SYMBOLS


def cross_check_partial_trace(ten: SymmetricTensor, pairs: list[tuple[int, int]], levels: list[int]):
    """There are different ways to compute partial traces. One particular
    choice is implemented in `partial_trace` in the `FusionTreeBackend` by
    choosing a certain way of braiding the paired legs such that they are
    adjacent to each other.

    Here we choose a different way to achieve this adjacency before performing
    the partial trace itself (which is again done by calling this very method
    `partial_trace`).
    """
    # we do not need to check that the levels are consistent since the result
    # of this function is compared to the result obtained using partial_trace
    # in the fusion_tree_backend, where the levels are checked.
    pairs = np.asarray([pair if pair[0] < pair[1] else (pair[1], pair[0]) for pair in pairs], dtype=int)
    # sort w.r.t. the second entry first
    pairs = pairs[np.lexsort(pairs.T)]
    idcs1 = []
    idcs2 = []
    for i1, i2 in pairs:
        idcs1.append(i1)
        idcs2.append(i2)
    remaining = [n for n in range(ten.num_legs) if n not in idcs1 and n not in idcs2]

    insert_idcs = [np.searchsorted(remaining, pair[1]) + 2 * i for i, pair in enumerate(pairs)]
    # permute legs such that the ones with the larger index do not move
    num_codom_legs = ten.num_codomain_legs
    idcs = remaining[:]
    for idx, pair in zip(insert_idcs, pairs):
        idcs[idx:idx] = list(pair)
        if pair[0] < ten.num_codomain_legs and pair[1] >= ten.num_codomain_legs:
            num_codom_legs -= 1  # leg at pair[0] is bent down

    ten = permute_legs(
        ten, codomain=idcs[:num_codom_legs], domain=idcs[num_codom_legs:][::-1], levels=levels, bend_right=True
    )
    tr_idcs1 = [i for i, idx in enumerate(idcs) if idx in idcs1]
    tr_idcs2 = [i for i, idx in enumerate(idcs) if idx in idcs2]
    new_pairs = list(zip(tr_idcs1, tr_idcs2))
    return ten.backend.partial_trace(ten, pairs=new_pairs, levels=[None] * ten.num_legs)


def cross_check_single_c_symbol_tree_blocks(
    ten: SymmetricTensor, leg: int | str, levels: list[int]
) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Naive implementation of a single C symbol for test purposes on the level
    of the tree blocks. `ten.legs[leg]` is exchanged with `ten.legs[leg + 1]`.

    NOTE this function may be deleted at a later stage
    """
    # NOTE the case of braiding in domain and codomain are treated separately despite being similar
    # This way, it is easier to understand the more compact and more efficient function
    ftb = fusion_tree_backend
    index = ten.get_leg_idcs(leg)[0]
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry

    assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
    assert index < ten.num_legs - 1

    in_domain = index > ten.num_codomain_legs - 1

    if in_domain:
        domain_index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
        spaces = (
            ten.domain[:domain_index]
            + [ten.domain[domain_index + 1]]
            + [ten.domain[domain_index]]
            + ten.domain[domain_index + 2 :]
        )
        new_domain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use: _sectors=ten.domain.sector_decomposition, _multiplicities=ten.domain.multiplicities)
        new_codomain = ten.codomain
    else:
        spaces = ten.codomain[:index] + [ten.codomain[index + 1]] + [ten.codomain[index]] + ten.codomain[index + 2 :]
        new_codomain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use: ten.codomain.sector_decomposition, ten.codomain.multiplicities
        new_domain = ten.domain

    zero_blocks = [
        block_backend.zeros(block_backend.get_shape(block), dtype=Dtype.complex128) for block in ten.data.blocks
    ]
    new_data = ftb.FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype, device=block_backend.default_device)
    shape_perm = np.arange(ten.num_legs)  # for permuting the shape of the tree blocks
    shifted_index = ten.num_codomain_legs + domain_index if in_domain else index
    shape_perm[shifted_index : shifted_index + 2] = shape_perm[shifted_index : shifted_index + 2][::-1]
    shape_perm = list(shape_perm)  # torch does not like np.arrays

    for alpha_tree, beta_tree, tree_block in ftb._tree_block_iter(ten):
        block_charge = ten.domain.sector_decomposition_where(alpha_tree.coupled)
        block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

        initial_shape = block_backend.get_shape(tree_block)
        modified_shape = [ten.codomain[i].sector_multiplicity(sec) for i, sec in enumerate(alpha_tree.uncoupled)]
        modified_shape += [ten.domain[i].sector_multiplicity(sec) for i, sec in enumerate(beta_tree.uncoupled)]

        tree_block = block_backend.reshape(tree_block, tuple(modified_shape))
        tree_block = block_backend.permute_axes(tree_block, shape_perm)
        tree_block = block_backend.reshape(tree_block, initial_shape)

        if index == 0 or index == ten.num_legs - 2:
            if in_domain:
                alpha_slice = new_codomain.tree_block_slice(alpha_tree)
                b = beta_tree.copy(True)
                b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities
                f = b.coupled if len(b_in) == 0 else b_in[0]
                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    r = symmetry.r_symbol(b_unc[0], b_unc[1], f)[b_mul[0]]
                else:
                    r = symmetry.r_symbol(b_unc[1], b_unc[0], f)[b_mul[0]].conj()
                b_unc[domain_index : domain_index + 2] = b_unc[domain_index : domain_index + 2][::-1]
                beta_slice = new_domain.tree_block_slice(b)
            else:
                beta_slice = new_domain.tree_block_slice(beta_tree)
                a = alpha_tree.copy(True)
                a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities
                f = a.coupled if len(a_in) == 0 else a_in[0]
                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    r = symmetry.r_symbol(a_unc[1], a_unc[0], f)[a_mul[0]]
                else:
                    r = symmetry.r_symbol(a_unc[0], a_unc[1], f)[a_mul[0]].conj()
                a_unc[index : index + 2] = a_unc[index : index + 2][::-1]
                alpha_slice = new_codomain.tree_block_slice(a)

            new_data.blocks[block_charge][alpha_slice, beta_slice] += r * tree_block
        else:
            if in_domain:
                alpha_slice = new_codomain.tree_block_slice(alpha_tree)

                beta_unc, beta_in, beta_mul = (beta_tree.uncoupled, beta_tree.inner_sectors, beta_tree.multiplicities)

                if domain_index == 1:
                    left_charge = beta_unc[domain_index - 1]
                else:
                    left_charge = beta_in[domain_index - 2]
                if domain_index == ten.num_domain_legs - 2:
                    right_charge = beta_tree.coupled
                else:
                    right_charge = beta_in[domain_index]

                for f in symmetry.fusion_outcomes(left_charge, beta_unc[domain_index + 1]):
                    if not symmetry.can_fuse_to(f, beta_unc[domain_index], right_charge):
                        continue

                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        cs = symmetry.c_symbol(
                            left_charge,
                            beta_unc[domain_index],
                            beta_unc[domain_index + 1],
                            right_charge,
                            beta_in[domain_index - 1],
                            f,
                        )[beta_mul[domain_index - 1], beta_mul[domain_index], :, :]
                    else:
                        cs = symmetry.c_symbol(
                            left_charge,
                            beta_unc[domain_index + 1],
                            beta_unc[domain_index],
                            right_charge,
                            f,
                            beta_in[domain_index - 1],
                        )[:, :, beta_mul[domain_index - 1], beta_mul[domain_index]].conj()

                    b = beta_tree.copy(True)
                    b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities

                    b_unc[domain_index : domain_index + 2] = b_unc[domain_index : domain_index + 2][::-1]
                    b_in[domain_index - 1] = f
                    for (kap, lam), c in np.ndenumerate(cs):
                        if abs(c) < backend.eps:
                            continue
                        b_mul[domain_index - 1] = kap
                        b_mul[domain_index] = lam

                        beta_slice = new_domain.tree_block_slice(b)
                        new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block
            else:
                beta_slice = new_domain.tree_block_slice(beta_tree)
                alpha_unc, alpha_in, alpha_mul = (
                    alpha_tree.uncoupled,
                    alpha_tree.inner_sectors,
                    alpha_tree.multiplicities,
                )

                left_charge = alpha_unc[0] if index == 1 else alpha_in[index - 2]
                right_charge = alpha_tree.coupled if index == ten.num_codomain_legs - 2 else alpha_in[index]

                for f in symmetry.fusion_outcomes(left_charge, alpha_unc[index + 1]):
                    if not symmetry.can_fuse_to(f, alpha_unc[index], right_charge):
                        continue

                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        cs = symmetry.c_symbol(
                            left_charge, alpha_unc[index + 1], alpha_unc[index], right_charge, f, alpha_in[index - 1]
                        )[:, :, alpha_mul[index - 1], alpha_mul[index]]
                    else:
                        cs = symmetry.c_symbol(
                            left_charge, alpha_unc[index], alpha_unc[index + 1], right_charge, alpha_in[index - 1], f
                        )[alpha_mul[index - 1], alpha_mul[index], :, :].conj()

                    a = alpha_tree.copy(True)
                    a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities

                    a_unc[index : index + 2] = a_unc[index : index + 2][::-1]
                    a_in[index - 1] = f
                    for (kap, lam), c in np.ndenumerate(cs):
                        if abs(c) < backend.eps:
                            continue
                        a_mul[index - 1] = kap
                        a_mul[index] = lam

                        alpha_slice = new_codomain.tree_block_slice(a)
                        new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_single_c_symbol_tree_cols(
    ten: SymmetricTensor, leg: int | str, levels: list[int]
) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Naive implementation of a single C symbol for test purposes on the level
    of the tree columns (= tree slices of codomain xor domain). `ten.legs[leg]`
    is exchanged with `ten.legs[leg + 1]`.

    NOTE this function may be deleted at a later stage
    """
    ftb = fusion_tree_backend
    index = ten.get_leg_idcs(leg)[0]
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry

    # NOTE do these checks in permute_legs for the actual (efficient) function
    assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
    assert index < ten.num_legs - 1

    in_domain = index > ten.num_codomain_legs - 1

    if in_domain:
        index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
        levels = levels[::-1]
        spaces = ten.domain[:index] + ten.domain[index : index + 2][::-1] + ten.domain[index + 2 :]
        new_domain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use:_sectors=ten.domain.sector_decomposition, _multiplicities=ten.domain.multiplicities)
        new_codomain = ten.codomain
        # for permuting the shape of the tree blocks
        shape_perm = np.append([0], np.arange(1, ten.num_domain_legs + 1))
        shape_perm[index + 1 : index + 3] = shape_perm[index + 1 : index + 3][::-1]
    else:
        spaces = ten.codomain[:index] + ten.codomain[index : index + 2][::-1] + ten.codomain[index + 2 :]
        new_codomain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use:
        # _sectors=ten.codomain.sector_decomposition,
        # _multiplicities=ten.codomain.multiplicities)
        new_domain = ten.domain
        shape_perm = np.append(np.arange(ten.num_codomain_legs), [ten.num_codomain_legs])
        shape_perm[index : index + 2] = shape_perm[index : index + 2][::-1]

    shape_perm = list(shape_perm)  # torch does not like np.arrays
    zero_blocks = [
        block_backend.zeros(block_backend.get_shape(block), dtype=Dtype.complex128) for block in ten.data.blocks
    ]
    new_data = ftb.FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype, device=block_backend.default_device)
    iter_space = [ten.codomain, ten.domain][in_domain]
    iter_coupled = [ten.codomain.sector_decomposition[ind[0]] for ind in ten.data.block_inds]

    for tree, slc, _, _ in iter_space.iter_tree_blocks(iter_coupled):
        block_charge = ten.domain.sector_decomposition_where(tree.coupled)
        block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

        tree_block = ten.data.blocks[block_charge][:, slc] if in_domain else ten.data.blocks[block_charge][slc, :]

        initial_shape = block_backend.get_shape(tree_block)

        modified_shape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(tree.uncoupled)]
        modified_shape.insert((not in_domain) * len(modified_shape), initial_shape[not in_domain])

        tree_block = block_backend.reshape(tree_block, tuple(modified_shape))
        tree_block = block_backend.permute_axes(tree_block, shape_perm)
        tree_block = block_backend.reshape(tree_block, initial_shape)

        if index == 0 or index == ten.num_legs - 2:
            new_tree = tree.copy(deep=True)
            _unc, _in, _mul = new_tree.uncoupled, new_tree.inner_sectors, new_tree.multiplicities
            f = new_tree.coupled if len(_in) == 0 else _in[0]

            if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                r = symmetry._r_symbol(_unc[1], _unc[0], f)[_mul[0]]
            else:
                r = symmetry._r_symbol(_unc[0], _unc[1], f)[_mul[0]].conj()

            if in_domain:
                r = r.conj()

            _unc[index : index + 2] = _unc[index : index + 2][::-1]
            if in_domain:
                new_slc = new_domain.tree_block_slice(new_tree)
            else:
                new_slc = new_codomain.tree_block_slice(new_tree)

            if in_domain:
                new_data.blocks[block_charge][:, new_slc] += r * tree_block
            else:
                new_data.blocks[block_charge][new_slc, :] += r * tree_block
        else:
            _unc, _in, _mul = tree.uncoupled, tree.inner_sectors, tree.multiplicities

            left_charge = _unc[0] if index == 1 else _in[index - 2]
            right_charge = tree.coupled if index == _in.shape[0] else _in[index]

            for f in symmetry.fusion_outcomes(left_charge, _unc[index + 1]):
                if not symmetry.can_fuse_to(f, _unc[index], right_charge):
                    continue

                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    cs = symmetry._c_symbol(left_charge, _unc[index + 1], _unc[index], right_charge, f, _in[index - 1])[
                        :, :, _mul[index - 1], _mul[index]
                    ]
                else:
                    cs = symmetry._c_symbol(left_charge, _unc[index], _unc[index + 1], right_charge, _in[index - 1], f)[
                        _mul[index - 1], _mul[index], :, :
                    ].conj()

                if in_domain:
                    cs = cs.conj()

                new_tree = tree.copy(deep=True)
                new_tree.uncoupled[index : index + 2] = new_tree.uncoupled[index : index + 2][::-1]
                new_tree.inner_sectors[index - 1] = f
                for (kap, lam), c in np.ndenumerate(cs):
                    if abs(c) < backend.eps:
                        continue
                    new_tree.multiplicities[index - 1] = kap
                    new_tree.multiplicities[index] = lam

                    if in_domain:
                        new_slc = new_domain.tree_block_slice(new_tree)
                    else:
                        new_slc = new_codomain.tree_block_slice(new_tree)

                    if in_domain:
                        new_data.blocks[block_charge][:, new_slc] += c * tree_block
                    else:
                        new_data.blocks[block_charge][new_slc, :] += c * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_single_b_symbol(
    ten: SymmetricTensor, bend_down: bool
) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Naive implementation of a single B symbol for test purposes.
    If `bend_down == True`, the right-most leg in the domain is bent down,
    otherwise the right-most leg in the codomain is bent up.

    NOTE this function may be deleted at a later stage
    """
    ftb = fusion_tree_backend
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry

    # NOTE do these checks in permute_legs for the actual (efficient) function
    if bend_down:
        assert ten.num_domain_legs > 0, 'There is no leg to bend in the domain!'
    else:
        assert ten.num_codomain_legs > 0, 'There is no leg to bend in the codomain!'
    assert len(ten.data.blocks) > 0, 'The given tensor has no blocks to act on!'

    spaces = [ten.codomain, ten.domain]
    space1, space2 = spaces[bend_down], spaces[not bend_down]
    new_space1 = TensorProduct(space1.factors[:-1], symmetry=symmetry)
    new_space2 = TensorProduct(space2.factors + [space1.factors[-1].dual], symmetry=symmetry)

    new_codomain = [new_space1, new_space2][bend_down]
    new_domain = [new_space1, new_space2][not bend_down]

    new_data = ten.backend.zero_data(
        new_codomain, new_domain, dtype=Dtype.complex128, device=ten.data.device, all_blocks=True
    )

    for alpha_tree, beta_tree, tree_block in ftb._tree_block_iter(ten):
        modified_shape = [ten.codomain[i].sector_multiplicity(sec) for i, sec in enumerate(alpha_tree.uncoupled)]
        modified_shape += [ten.domain[i].sector_multiplicity(sec) for i, sec in enumerate(beta_tree.uncoupled)]

        if bend_down:
            if beta_tree.uncoupled.shape[0] == 1:
                coupled = symmetry.trivial_sector
            else:
                coupled = (
                    beta_tree.inner_sectors[-1] if beta_tree.inner_sectors.shape[0] > 0 else beta_tree.uncoupled[0]
                )
            modified_shape = (
                prod(modified_shape[: ten.num_codomain_legs]),
                prod(modified_shape[ten.num_codomain_legs : ten.num_legs - 1]),
                modified_shape[ten.num_legs - 1],
            )
            sec_mul = ten.domain[-1].sector_multiplicity(beta_tree.uncoupled[-1])
            final_shape = (
                block_backend.get_shape(tree_block)[0] * sec_mul,
                block_backend.get_shape(tree_block)[1] // sec_mul,
            )
        else:
            if alpha_tree.uncoupled.shape[0] == 1:
                coupled = symmetry.trivial_sector
            else:
                coupled = (
                    alpha_tree.inner_sectors[-1] if alpha_tree.inner_sectors.shape[0] > 0 else alpha_tree.uncoupled[0]
                )
            modified_shape = (
                prod(modified_shape[: ten.num_codomain_legs - 1]),
                modified_shape[ten.num_codomain_legs - 1],
                prod(modified_shape[ten.num_codomain_legs : ten.num_legs]),
            )
            sec_mul = ten.codomain[-1].sector_multiplicity(alpha_tree.uncoupled[-1])
            final_shape = (
                block_backend.get_shape(tree_block)[0] // sec_mul,
                block_backend.get_shape(tree_block)[1] * sec_mul,
            )
        block_ind = new_domain.sector_decomposition_where(coupled)
        block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

        tree_block = block_backend.reshape(tree_block, modified_shape)
        tree_block = block_backend.permute_axes(tree_block, [0, 2, 1])
        tree_block = block_backend.reshape(tree_block, final_shape)

        trees = [alpha_tree, beta_tree]
        tree1, tree2 = trees[bend_down], trees[not bend_down]

        if tree1.uncoupled.shape[0] == 1:
            tree1_in_1 = symmetry.trivial_sector
        else:
            tree1_in_1 = tree1.inner_sectors[-1] if tree1.inner_sectors.shape[0] > 0 else tree1.uncoupled[0]

        new_tree1 = FusionTree(
            symmetry,
            tree1.uncoupled[:-1],
            tree1_in_1,
            tree1.are_dual[:-1],
            tree1.inner_sectors[:-1],
            tree1.multiplicities[:-1],
        )
        new_tree2 = FusionTree(
            symmetry,
            np.append(tree2.uncoupled, [symmetry.dual_sector(tree1.uncoupled[-1])], axis=0),
            tree1_in_1,
            np.append(tree2.are_dual, [not tree1.are_dual[-1]]),
            np.append(tree2.inner_sectors, [tree2.coupled], axis=0),
            np.append(tree2.multiplicities, [0]),
        )

        b_sym = symmetry._b_symbol(tree1_in_1, tree1.uncoupled[-1], tree1.coupled)
        if not bend_down:
            b_sym = b_sym.conj()
        if tree1.are_dual[-1]:
            b_sym *= symmetry.frobenius_schur(tree1.uncoupled[-1])
        mu = tree1.multiplicities[-1] if tree1.multiplicities.shape[0] > 0 else 0
        for nu in range(b_sym.shape[1]):
            if abs(b_sym[mu, nu]) < backend.eps:
                continue
            new_tree2.multiplicities[-1] = nu

            if bend_down:
                alpha_slice = new_codomain.tree_block_slice(new_tree2)
                if new_tree1.uncoupled.shape[0] == 0:
                    beta_slice = slice(0, 1)
                else:
                    beta_slice = new_domain.tree_block_slice(new_tree1)
            else:
                if new_tree1.uncoupled.shape[0] == 0:
                    alpha_slice = slice(0, 1)
                else:
                    alpha_slice = new_codomain.tree_block_slice(new_tree1)
                beta_slice = new_domain.tree_block_slice(new_tree2)

            new_data.blocks[block_ind][alpha_slice, beta_slice] += b_sym[mu, nu] * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_transpose(ten: SymmetricTensor, over: bool, twist_codomain: bool) -> SymmetricTensor:
    """Alternative implementation of transpose.

    There are four ways we can realize a transpose by using permute legs.
    This function implements all of them, so that we can compare.
    One of the four is implemented in the actual FusionTreeBackend.

    Firstly, consider the definition where the legs from the original codomain are bent to the left
    and domain to the right. (``twist_codomain=True``)
    To realize this via `permute_legs`, we need to twist the legs from the original codomain
    such that they also bend to the right.
    We have a choice to move them either over the legs from the domain or under, and need to
    consistently use twists with the right chirality and braids with matching chirality (levels).

    Secondly, we can consider the other equal definition of the transpose, where the legs from
    the original codomain are bent to the right. (``twist_codomain=False``)
    Then, we need to twist the legs from the domain, and again have a binary choice for the
    chiralities.
    """
    if twist_codomain:
        #                │            ╭─╮     │
        #    ╭─────╮     │            ╰─│─────│──╮    <- chirality of both crossings depends on
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │       `over`. ``over=False`` is drawn.
        #    │  ┃  Y  ┃  │           ┃  Y  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    │     │     │     =        │     │  │
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │
        #    │  ┃  X  ┃  │           ┃  X  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    │     ╰─────╯              ╰─────╯  │
        overtwist = over
        # over: codomain goes on top with the high levels & vice versa
        levels = [*reversed(range(ten.num_legs))] if over else [*range(ten.num_legs)]
        twist_idcs = [*range(ten.num_codomain_legs)]
    else:
        #    │     ╭─────╮              ╭─────╮  │
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │
        #    │  ┃  Y  ┃  │           ┃  Y  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    │     │     │     =        │     │  │
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │
        #    │  ┃  X  ┃  │           ┃  X  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    ╰─────╯     │            ╭─│─────│──╯    <- chirality of both crossings depends on
        #                │            ╰─╯     │          `over`. ``over=False`` is drawn.
        overtwist = not over
        # over: domain goes on top with the high levels & vice versa
        levels = [*range(ten.num_legs)] if over else [*reversed(range(ten.num_legs))]
        twist_idcs = [*range(ten.num_domain_legs)]

    twist_instruction = fusion_tree_backend.TwistInstruction(
        codomain=twist_codomain, idcs=twist_idcs, overtwist=overtwist
    )
    codomain_idcs = list(range(ten.num_codomain_legs, ten.num_legs))
    domain_idcs = list(reversed(range(ten.num_codomain_legs)))
    # we take care of the twist manually here, so we use only right bends in the permute_legs part
    permute_instructions = permute_legs_instructions(
        num_codomain_legs=ten.num_codomain_legs,
        num_domain_legs=ten.num_domain_legs,
        codomain_idcs=codomain_idcs,
        domain_idcs=domain_idcs,
        levels=levels,
        has_symmetric_braid=ten.symmetry.has_symmetric_braid,
        bend_right=[True] * ten.num_legs,
    )
    instructions = [twist_instruction, *permute_instructions]
    new_codomain = ten.domain.dual
    new_domain = ten.codomain.dual
    data = ten.backend.apply_instructions(
        ten,
        instructions=instructions,
        codomain_idcs=codomain_idcs,
        domain_idcs=domain_idcs,
        new_codomain=new_codomain,
        new_domain=new_domain,
        mixes_codomain_domain=True,
    )
    return SymmetricTensor(data, new_codomain, new_domain, backend=ten.backend)
