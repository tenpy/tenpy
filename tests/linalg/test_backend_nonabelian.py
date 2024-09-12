"""A collection of tests for tenpy.linalg.backends.fusion_tree_backend"""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import Callable
import pytest
import numpy as np

from tenpy.linalg import backends
from tenpy.linalg.backends import fusion_tree_backend, get_backend, TensorBackend
from tenpy.linalg.spaces import ElementarySpace, ProductSpace
from tenpy.linalg.tensors import DiagonalTensor, SymmetricTensor, move_leg
from tenpy.linalg.symmetries import ProductSymmetry, fibonacci_anyon_category, SU2Symmetry, SU3_3AnyonCategory
from tenpy.linalg.dtypes import Dtype


def apply_single_b_symbol_efficient(ten: SymmetricTensor, bend_up: bool
                                    ) -> tuple[fusion_tree_backend.FusionTreeData, ProductSpace, ProductSpace]:
    """Use the efficient implementation of b symbols to return the action of a single
    b symbol in the same format (input and output) as the inefficient implementation.
    This is of course inefficient usage of this implementation but a necessity  in order
    to use the structure of the already implemented tests.
    """
    func = ten.backend._find_approproiate_mapping_dict
    index = ten.num_codomain_legs - 1
    coupled = [ten.domain.sectors[ind[1]] for ind in ten.data.block_inds]

    if bend_up:
        axes_perm = list(range(ten.num_codomain_legs)) + [ten.num_legs - 1]
        axes_perm += [ten.num_codomain_legs + i for i in range(ten.num_domain_legs - 1)]
    else:
        axes_perm = list(range(ten.num_codomain_legs - 1))
        axes_perm += [ten.num_codomain_legs + i for i in range(ten.num_domain_legs)]
        axes_perm += [ten.num_codomain_legs - 1]

    mapp, new_codomain, new_domain, _ = func(ten.codomain, ten.domain, index, coupled,
                                             None, bend_up)
    new_data = ten.backend._apply_mapping_dict(ten, new_codomain, new_domain, axes_perm,
                                               mapp, in_domain=None)
    return new_data, new_codomain, new_domain


def apply_single_c_symbol_efficient(ten: SymmetricTensor, leg: int | str, levels: list[int]
                                    ) -> tuple[fusion_tree_backend.FusionTreeData, ProductSpace, ProductSpace]:
    """Use the efficient implementation of c symbols to return the action of a single
    c symbol in the same format (input and output) as the inefficient implementation.
    This is of course inefficient usage of this implementation but a necessity  in order
    to use the structure of the already implemented tests.
    """
    func = ten.backend._find_approproiate_mapping_dict
    index = ten.get_leg_idcs(leg)[0]
    in_domain = index > ten.num_codomain_legs - 1
    overbraid = levels[index] > levels[index + 1]
    coupled = [ten.domain.sectors[ind[1]] for ind in ten.data.block_inds]

    if not in_domain:
        axes_perm = list(range(ten.num_codomain_legs))
        index_ = index
    else:
        axes_perm = list(range(ten.num_domain_legs))
        index_ = ten.num_legs - 1 - (index + 1)
    axes_perm[index_:index_ + 2] = axes_perm[index_:index_ + 2][::-1]

    mapp, new_codomain, new_domain, _ = func(ten.codomain, ten.domain, index, coupled,
                                             overbraid, None)
    new_data = ten.backend._apply_mapping_dict(ten, new_codomain, new_domain, axes_perm,
                                               mapp, in_domain)
    return new_data, new_codomain, new_domain


def assert_tensors_almost_equal(a: SymmetricTensor, expect: SymmetricTensor, eps: float):
    assert a.codomain == expect.codomain
    assert a.domain == expect.domain
    assert a.backend.almost_equal(a, expect, rtol=eps, atol=eps)


def assert_repeated_braids_trivial(a: SymmetricTensor, funcs: list[Callable], levels: list[int],
                                   repeat: int, eps: float):
    """Check that repeatedly braiding two neighboring legs often enough leaves the tensor
    invariant. The number of required repetitions must be chosen to be even (legs must be
    at their initial positions again) and such that all (relevant) r symbols to the power
    of the repetitions is identity. `levels` specifies whether the repeated exchange is
    always clockwise or anti-clockwise.
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            for _ in range(repeat):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

            assert_tensors_almost_equal(new_a, a, eps)


def assert_clockwise_counterclockwise_trivial(a: SymmetricTensor, funcs: list[Callable],
                                              levels: list[int], eps: float):
    """Check that braiding a pair of neighboring legs clockwise and then anti-clockwise
    (or vice versa) leaves the tensor invariant. `levels` specifies whether the first
    exchange is clockwise or anti-clockwise (the second is the opposite).
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
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


def assert_clockwise_counterclockwise_trivial_long_range(a: SymmetricTensor, move_leg_or_permute_leg: str,
                                                         eps: float, np_random: np.random.Generator):
    """Same as `assert_clockwise_counterclockwise_trivial` with the difference that a random
    sequence of exchanges and bends is chosen. The identity is checked using `permute_legs`
    of the tensor backend or using `move_leg` depending on `move_leg_or_permute_leg`.
    """
    levels = list(np_random.permutation(a.num_legs))
    if move_leg_or_permute_leg == 'permute_leg':
        # more general case; needs more input
        permutation = list(np_random.permutation(a.num_legs))
        inv_permutation = [permutation.index(i) for i in range(a.num_legs)]
        inv_levels = [levels[i] for i in permutation]
        num_codomain = np.random.randint(a.num_legs + 1)

        new_data, new_codomain, new_domain = a.backend.permute_legs(a, permutation[:num_codomain],
                                                                    permutation[num_codomain:][::-1], levels)
        new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)
        new_data, new_codomain, new_domain = a.backend.permute_legs(new_a, inv_permutation[:a.num_codomain_legs],
                                                                    inv_permutation[a.num_codomain_legs:][::-1], inv_levels)
        new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

    elif move_leg_or_permute_leg == 'move_leg':
        leg = np_random.integers(a.num_legs)
        co_dom_pos = np_random.integers(a.num_legs)

        if co_dom_pos >= a.num_codomain_legs:
            new_a = move_leg(a, leg, domain_pos=a.num_legs - 1 - co_dom_pos, levels=levels)
        else:
            new_a = move_leg(a, leg, codomain_pos=co_dom_pos, levels=levels)

        tmp = levels[leg]
        levels = [levels[i] for i in range(a.num_legs) if i != leg]
        levels.insert(co_dom_pos, tmp)
        if leg >= a.num_codomain_legs:
            new_a = move_leg(new_a, co_dom_pos, domain_pos=a.num_legs - 1 - leg, levels=levels)
        else:
            new_a = move_leg(new_a, co_dom_pos, codomain_pos=leg, levels=levels)

    assert_tensors_almost_equal(new_a, a, eps)


def assert_braiding_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable],
                                               levels: list[int], eps: float):
    """Check that when rescaling and exchanging legs, it does not matter whether one first
    performs the rescaling and then the exchange process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the funtion directly acting on tensors;
    this function is tested elsewhere.
    """
    for func in funcs:
        for leg in range(a.num_legs-1):
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

            assert_tensors_almost_equal(new_a, new_a2, eps)


def assert_bending_up_and_down_trivial(codomains: list[ProductSpace], domains: list[ProductSpace],
                                       funcs: list[Callable], backend: TensorBackend, multiple: bool, eps: float):
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
                bends = [True] * (i+1) + [False] * (i+1)
                bending.append(bends)
                if not multiple:
                    break
            for i in range(tens.num_codomain_legs):
                bends = [False] * (i+1) + [True] * (i+1)
                bending.append(bends)
                if not multiple:
                    break

            for func in funcs:
                for bends in bending:
                    new_tens = tens.copy()
                    for bend in bends:
                        new_data, new_codomain, new_domain = func(new_tens, bend)
                        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

                    assert_tensors_almost_equal(new_tens, tens, eps)


def assert_bending_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable], eps: float):
    """Check that when rescaling and bending legs, it does not matter whether one first
    performs the rescaling and then the bending process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the funtion directly acting on tensors;
    this function is tested elsewhere.
    """
    bends = [True, False]
    for bend_up in bends:
        if a.num_codomain_legs == 0 and not bend_up:
            continue
        elif a.num_domain_legs == 0 and bend_up:
            continue

        for func in funcs:
            if bend_up:
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
            new_data, new_codomain, new_domain = func(new_a, bend_up)
            new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a.backend)

            # bend first
            new_data, new_codomain, new_domain = func(new_a2, bend_up)
            new_a2 = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a2.backend)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag, num_leg)

            assert_tensors_almost_equal(new_a, new_a2, eps)


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


def test_c_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_c_symbol_inefficient,
             backend._apply_single_c_symbol_more_efficient,
             apply_single_c_symbol_efficient]
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2, 3], [6, 5, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 5 and 6 (in domain)
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [5, 6, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 6, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 2 and 3 (in codomain)
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 3, 2], [6, 5, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 3, codomain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 4 and 5 (in domain)
    expect = [np.zeros((8, 3), dtype=complex), np.zeros((13, 5), dtype=complex)]

    expect[0][:, 0] = blocks[0][:, 0] * r1
    expect[0][:, 1] = blocks[0][:, 1]
    expect[0][:, 2] = blocks[0][:, 2] * rtau

    expect[1][:, 0] = blocks[1][:, 0]
    expect[1][:, 1] = blocks[1][:, 1] * rtau
    expect[1][:, 2] = blocks[1][:, 2]
    expect[1][:, 3] = blocks[1][:, 3] * ctttt11 + blocks[1][:, 4] * cttttt1
    expect[1][:, 4] = blocks[1][:, 3] * ctttt1t + blocks[1][:, 4] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_domain = ProductSpace([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [6, 4, 5], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


def test_c_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_c_symbol_inefficient,
             backend._apply_single_c_symbol_more_efficient,
             apply_single_c_symbol_efficient]
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=0, levels=levels)
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [4, 5, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 5, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 3 and 4 (in domain)
    phi = (1 + 5**0.5) / 2
    ctttt11 = phi**-1 * r1.conj()  # C symbols
    cttttt1 = phi**-0.5 * rtau * r1.conj()
    ctttt1t = phi**-0.5 * rtau.conj()
    ctttttt = -1*phi**-1
    exc4 = [0, 2, 1, 3, 4, 6, 5, 7]
    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * r1 * -1
    expect[0][:, 4:] = blocks[0][:, 4:]

    expect[1][:, :4] = blocks[1][:, exc] * rtau * -1
    expect[1][:, 4:] = blocks[1][:, 4:]

    # f-symbols for su(2) [e -> f]: 0 -> 0: -1/2, 2 -> 2: 1/2, 0 -> 2 and 2 -> 0: 3**.5/2
    expect[2][:, :8] = (blocks[2][:, exc4] * rtau * (-1/4 + 3/4)
                       + blocks[2][:, [8 + i for i in exc4]] * rtau * (3**0.5/4 + 3**0.5/4))
    expect[2][:, 8:] = (blocks[2][:, exc4] * rtau * (3**0.5/4 + 3**0.5/4)
                       + blocks[2][:, [8 + i for i in exc4]] * rtau * (1/4 - 3/4))

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = (blocks[3][:, [2 + i for i in exc4]] * ctttt11 * (-1/4 + 3/4)
                         + blocks[3][:, [10 + i for i in exc4]] * ctttt11 * (3**0.5/4 + 3**0.5/4)
                         + blocks[3][:, [18 + i for i in exc4]] * cttttt1 * (-1/4 + 3/4)
                         + blocks[3][:, [26 + i for i in exc4]] * cttttt1 * (3**0.5/4 + 3**0.5/4))
    expect[3][:, 10:18] = (blocks[3][:, [2 + i for i in exc4]] * ctttt11 * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt11 * (1/4 - 3/4)
                          + blocks[3][:, [18 + i for i in exc4]] * cttttt1 * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [26 + i for i in exc4]] * cttttt1 * (1/4 - 3/4))
    expect[3][:, 18:26] = (blocks[3][:, [2 + i for i in exc4]] * ctttt1t * (-1/4 + 3/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt1t * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [18 + i for i in exc4]] * ctttttt * (-1/4 + 3/4)
                          + blocks[3][:, [26 + i for i in exc4]] * ctttttt * (3**0.5/4 + 3**0.5/4))
    expect[3][:, 26:34] = (blocks[3][:, [2 + i for i in exc4]] * ctttt1t * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt1t * (1/4 - 3/4)
                          + blocks[3][:, [18 + i for i in exc4]] * ctttttt * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [26 + i for i in exc4]] * ctttttt * (1/4 - 3/4))

    expect[4][:, :4] = blocks[4][:, exc] * r1
    expect[4][:, 4:] = blocks[4][:, 4:]

    expect[5][:, :4] = blocks[5][:, exc] * rtau
    expect[5][:, 4:] = blocks[5][:, 4:]

    expect[6][:, :] = blocks[6][:, exc4] * rtau

    expect[7][:, :8] = blocks[7][:, exc4] * ctttt11 + blocks[7][:, [8 + i for i in exc4]] * cttttt1
    expect[7][:, 8:] = blocks[7][:, exc4] * ctttt1t + blocks[7][:, [8 + i for i in exc4]] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_domain = ProductSpace([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [5, 3, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


def test_c_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_c_symbol_inefficient,
             backend._apply_single_c_symbol_more_efficient,
             apply_single_c_symbol_efficient]
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=0, levels=levels)
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [4, 5, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 5, domain_pos=1, levels=levels)
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

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 2, 1], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # exchange legs 3 and 4 (in domain)
    exc = [0, 2, 1, 3]
    exc4, exc8 = [4 + i for i in exc], [8 + i for i in exc]
    exc28, exc32 = [28 + i for i in exc], [32 + i for i in exc]
    expect = [np.zeros(shp, dtype=complex) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * r8[0]
    expect[0][:, 4:8] = blocks[0][:, exc4] * r8[1]
    expect[0][:, 8:] = blocks[0][:, exc8] * -1

    v = [blocks[1][:, [4*i + j for j in exc]] for i in range(7)]
    for i in range(7):
        w = [csym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        expect[1][:, 4*i:4*(i+1)] = np.sum([v[j] * w[j] for j in range(7)], axis=0)

    expect[1][:, 28:32] = (blocks[1][:, exc28] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                          + blocks[1][:, exc32] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[1][:, 32:] = (blocks[1][:, exc28] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[1][:, exc32] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))

    expect[2][:, :4] = (blocks[2][:, exc] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                       + blocks[2][:, exc4] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[2][:, 4:8] = (blocks[2][:, exc] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[2][:, exc4] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[2][:, 8:] = blocks[2][:, exc8] * -1

    expect[3][:, :4] = (blocks[3][:, exc] * (f2[0,0]*r8[0]*f1[0,0] + f2[0,1]*r8[1]*f1[1,0])
                       + blocks[3][:, exc4] * (f2[1,0]*r8[0]*f1[0,0] + f2[1,1]*r8[1]*f1[1,0]))
    expect[3][:, 4:8] = (blocks[3][:, exc] * (f2[0,0]*r8[0]*f1[0,1] + f2[0,1]*r8[1]*f1[1,1])
                        + blocks[3][:, exc4] * (f2[1,0]*r8[0]*f1[0,1] + f2[1,1]*r8[1]*f1[1,1]))
    expect[3][:, 8:] = blocks[3][:, exc8] * -1

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [5, 3, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # braid 4 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=4, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


def test_b_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_b_symbol, apply_single_b_symbol_efficient]
    eps = 1.e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    s3 = ElementarySpace(sym, [[0], [1]], [2, 3])  # 1 and tau

    # tensor with single leg in codomain; bend down
    codomain = ProductSpace([s2])
    domain = ProductSpace([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.block_random_uniform((1, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_codomain = ProductSpace([], symmetry=sym)
    expect_domain = ProductSpace([s2.dual])
    expect_tens = SymmetricTensor(data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [], [0], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, domain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # tensor with single leg in domain; bend up
    codomain = ProductSpace([], symmetry=sym)
    domain = ProductSpace([s3])

    block_inds = np.array([[0,0]])
    blocks = [backend.block_backend.block_random_uniform((1, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect = [backend.block_backend.block_reshape(blocks[0], (2, 1))]
    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s3.dual])
    expect_domain = ProductSpace([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # more complicated tensor
    codomain = ProductSpace([s2, s1, s1])
    domain = ProductSpace([s2, s1, s2])

    block_inds = np.array([[0,0], [1,1]])
    blocks = [backend.block_backend.block_random_uniform((2, 3), Dtype.complex128),
              backend.block_backend.block_random_uniform((3, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    # bend up
    phi = (1 + 5**0.5) / 2
    expect = [np.zeros((5, 1), dtype=complex), np.zeros((8, 2), dtype=complex)]

    expect[0][0, 0] = blocks[0][0, 1] # (0, 0, 0) = (a, b, c) as in _b_symbol(a, b, c)
    expect[0][1, 0] = blocks[1][0, 3] * phi**0.5 # (0, 1, 1)
    expect[0][2, 0] = blocks[0][1, 1] # (0, 0, 0)
    expect[0][3, 0] = blocks[1][1, 3] * phi**0.5 # (0, 1, 1)
    expect[0][4, 0] = blocks[1][2, 3] * phi**0.5 # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0] # (1, 0, 1)
    expect[1][1, 0] = blocks[0][0, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][2, 0] = blocks[1][0, 1] # (1, 1, 1)
    expect[1][3, 0] = blocks[1][1, 0] # (1, 0, 1)
    expect[1][4, 0] = blocks[1][2, 0] # (1, 0, 1)
    expect[1][5, 0] = blocks[1][1, 1] # (1, 1, 1)
    expect[1][6, 0] = blocks[0][1, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][7, 0] = blocks[1][2, 1] # (1, 1, 1)

    expect[1][0, 1] = blocks[1][0, 2] # (1, 0, 1)
    expect[1][1, 1] = blocks[0][0, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][2, 1] = blocks[1][0, 4] # (1, 1, 1)
    expect[1][3, 1] = blocks[1][1, 2] # (1, 0, 1)
    expect[1][4, 1] = blocks[1][2, 2] # (1, 0, 1)
    expect[1][5, 1] = blocks[1][1, 4] # (1, 1, 1)
    expect[1][6, 1] = blocks[0][1, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][7, 1] = blocks[1][2, 4] # (1, 1, 1)

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s2, s1, s1, s2.dual])
    expect_domain = ProductSpace([s2, s1])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [5, 4], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 3, codomain_pos=3, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # bend down
    expect = [np.zeros((1, 5), dtype=complex), np.zeros((2, 8), dtype=complex)]

    expect[0][0, 0] = blocks[1][1, 0] * phi**0.5 # (0, 1, 1)
    expect[0][0, 1] = blocks[1][1, 1] * phi**0.5 # (0, 1, 1)
    expect[0][0, 2] = blocks[1][1, 2] * phi**0.5 # (0, 1, 1)
    expect[0][0, 3] = blocks[1][1, 3] * phi**0.5 # (0, 1, 1)
    expect[0][0, 4] = blocks[1][1, 4] * phi**0.5 # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0] # (1, 1, 1)
    expect[1][0, 1] = blocks[0][0, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 2] = blocks[1][0, 1] # (1, 1, 1)
    expect[1][0, 3] = blocks[0][0, 1] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 4] = blocks[1][0, 2] # (1, 1, 1)
    expect[1][0, 5] = blocks[1][0, 3] # (1, 1, 1)
    expect[1][0, 6] = blocks[0][0, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 7] = blocks[1][0, 4] # (1, 1, 1)

    expect[1][1, 0] = blocks[1][2, 0] # (1, 1, 1)
    expect[1][1, 1] = blocks[0][1, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 2] = blocks[1][2, 1] # (1, 1, 1)
    expect[1][1, 3] = blocks[0][1, 1] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 4] = blocks[1][2, 2] # (1, 1, 1)
    expect[1][1, 5] = blocks[1][2, 3] # (1, 1, 1)
    expect[1][1, 6] = blocks[0][1, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 7] = blocks[1][2, 4] # (1, 1, 1)

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s2, s1])
    expect_domain = ProductSpace([s2, s1, s2, s1.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1], [5, 4, 3, 2], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, domain_pos=3, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    spaces = [ProductSpace([], symmetry=sym), ProductSpace([s2]), ProductSpace([s3]),
              ProductSpace([s1, s3]), ProductSpace([s2, s3]), ProductSpace([s3, s1, s3, s2])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


def test_b_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_b_symbol, apply_single_b_symbol_efficient]
    perm_axes = backend.block_backend.block_permute_axes
    reshape = backend.block_backend.block_reshape
    eps = 1.e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [1])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    s3 = ElementarySpace(sym, [[0, 0], [1, 1], [1, 2]], [1, 2, 2])  # (1, spin-0), (tau, spin-1/2) and (tau, spin-1)

    # tensor with two legs in domain; bend up
    codomain = ProductSpace([], symmetry=sym)
    domain = ProductSpace([s2, s3])

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.block_random_uniform((1, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [1, 1]])
    expect = [np.zeros((1, 1), dtype=complex), np.zeros((2, 2), dtype=complex)]

    expect[0][0, 0] = blocks[0][0, 0]  # ([0, 0], [0, 0], [0, 0]) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][:, :] = perm_axes(reshape(blocks[0][0, 1:], (2, 2)), [1, 0])
    expect[1][:, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * -1  # ([1, 1], [1, 1], [0, 0])

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s3.dual])
    expect_domain = ProductSpace([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # tensor with two legs in codomain, two leg in domain; bend down
    codomain = ProductSpace([s1, s3])
    domain = ProductSpace([s2, s3])

    # charges [0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(2, 5), (2, 4), (2, 4), (3, 8), (2, 4), (2, 6), (2, 4), (2, 4)]
    blocks = [backend.block_backend.block_random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 2]])
    expect = [np.zeros((1, 86), dtype=complex)]

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

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1])
    expect_domain = ProductSpace([s2, s3, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [3, 2, 1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # similar tensor, replace one sector with its dual (Frobenius-Schur is now relevant); bend up
    codomain = ProductSpace([s1, s3])
    domain = ProductSpace([s2, s3.dual])

    blocks = [backend.block_backend.block_random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [3, 1]])
    expect = [np.zeros((12, 1), dtype=complex), np.zeros((37, 2), dtype=complex)]

    expect[0][2:4, 0] = blocks[0][:, 0] # ([0, 0], [0, 0], [0, 0])

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

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1, s3, s3])
    expect_domain = ProductSpace([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [3], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, codomain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    spaces = [ProductSpace([], symmetry=sym), ProductSpace([s2]), ProductSpace([s3.dual]),
              ProductSpace([s1, s3]), ProductSpace([s2, s3.dual]), ProductSpace([s1, s3, s2.dual])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


def test_b_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [backend._apply_single_b_symbol, apply_single_b_symbol_efficient]
    perm_axes = backend.block_backend.block_permute_axes
    reshape = backend.block_backend.block_reshape
    eps = 1.e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    s3 = ElementarySpace(sym, [[0], [1], [3]], [1, 2, 3])  # 1, 8, 10-
    qdim8 = sym.sqrt_qdim(np.array([1]))  # sqrt of qdim of charge 8
    # when multiplying with qdims (from the b symbols), only 8 is relevant since all other qdim are 1
    # the b symbols are diagonal in the multiplicity index

    # tensor with two legs in codomain; bend down
    codomain = ProductSpace([s1, s3])
    domain = ProductSpace([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.block_random_uniform((5, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [np.zeros((1, 2), dtype=complex), np.zeros((1, 3), dtype=complex)]

    expect[0][0, :] = blocks[0][:2, 0] / qdim8  # (8, 8, 1) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][0, :] = blocks[0][2:, 0]  # (10, 10-, 1)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1])
    expect_domain = ProductSpace([s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # tensor with two legs in codomain, one leg in domain; bend down
    codomain = ProductSpace([s1, s3])
    domain = ProductSpace([s2])

    block_inds = np.array([[1, 0]])
    blocks = [backend.block_backend.block_random_uniform((10, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [np.zeros((1, 16), dtype=complex), np.zeros((1, 4), dtype=complex)]

    expect[0][0, :2] = blocks[0][0, :]  # (8, 1, 8)
    expect[0][0, 2:6] = reshape(perm_axes(blocks[0][1:3, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 6:10] = reshape(perm_axes(blocks[0][3:5, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 10:] = reshape(perm_axes(blocks[0][5:8, :], [1, 0]), (1, 6)) * -1 # (8, 10-, 8)

    expect[1][0, :] = reshape(perm_axes(blocks[0][8:, :], [1, 0]), (1, 4)) * qdim8 * -1  # (10, 8, 8)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1])
    expect_domain = ProductSpace([s2, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [2, 1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=1, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # same tensor, bend up
    expect_block_inds = np.array([[0, 0]])
    expect = [np.zeros((20, 1), dtype=complex)]

    expect[0][:2, 0] = blocks[0][0, :] * qdim8  # (1, 8, 8)
    expect[0][2:6, :] = reshape(blocks[0][1:3, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][6:10, :] = reshape(blocks[0][3:5, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][10:16, :] = reshape(blocks[0][5:8, :], (6, 1)) * qdim8  # (1, 8, 8)
    expect[0][16:, :] = reshape(blocks[0][8:, :], (4, 1)) * qdim8  # (1, 8, 8)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1, s3, s2.dual])
    expect_domain = ProductSpace([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, codomain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    # more complicated tensor, bend down
    codomain = ProductSpace([s1, s2, s2])
    domain = ProductSpace([s2, s3])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(12, 4), (36, 16), (12, 4), (12, 4)]
    blocks = [backend.block_backend.block_random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_shapes = [(2, 32), (6, 88), (2, 32), (2, 32)]
    expect = [np.zeros(shp, dtype=complex) for shp in expect_shapes]

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

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128)
    expect_codomain = ProductSpace([s1, s2])
    expect_domain = ProductSpace([s2, s3, s2.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1], [4, 3, 2], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, domain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps)


    spaces = [ProductSpace([], symmetry=sym), ProductSpace([s2]), ProductSpace([s3.dual]),
              ProductSpace([s1, s3]), ProductSpace([s2, s3.dual]), ProductSpace([s1, s3, s2.dual])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)
