"""A collection of tests for :module:`tenpy.networks.terms`."""

# Copyright (C) TeNPy Developers, Apache license
import copy

import lasyten as lasy
import numpy as np
import pytest

from tenpy.models.spins import SpinChain

# from tenpy.models.spins_nnn import SpinChainNNN2
from tenpy.networks import mpo
from tenpy.networks.terms import *

spin_half = lasy.models.SpinSite(0.5, conserve='Sz')


def test_TermList():
    pytest.skip('TermList not adapted yet')
    terms = [
        [('N', 3), ('N', 2), ('N', 3)],
        [('C', 0), ('N', 2), ('N', 4), ('Cd', 3), ('C', 2)],
        [('C', 0), ('N', 2), ('N', 4), ('Cd', 3), ('Cd', 0), ('C', 2)],
    ]
    strength = [1.0, 2.0, 3.0]
    terms_copy = copy.deepcopy(terms)
    terms_ordered = [
        [('N', 2), ('N N', 3)],
        [('C', 0), ('N C', 2), ('Cd', 3), ('N', 4)],
        [('C Cd', 0), ('N C', 2), ('Cd', 3), ('N', 4)],
    ]
    tl = TermList(terms, strength)
    print(tl)
    tl.order_combine([dummy] * 7)
    print(tl)
    assert terms == terms_copy
    assert tl.terms == terms_ordered
    assert np.all(tl.strength == np.array(strength))  # no sites -> just permute
    tl = TermList(terms, strength)
    tl.order_combine([fermion] * 3)  # should anti-commute
    assert tl.terms == terms_ordered
    assert np.all(tl.strength == np.array([1.0, -2.0, 3.0]))


def test_onsite_terms():
    L = 6
    strength1 = np.arange(1.0, 1.0 + L * 0.25, 0.25)
    o1 = OnsiteTerms(L)
    for i in [1, 0, 3]:
        o1.add_onsite_term(strength1[i], i, f'X_{i:d}')
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"X_1": strength1[1]},
                               {},
                               {"X_3": strength1[3]},
                               {},
                               {}]  # fmt: skip
    strength2 = np.arange(2.0, 2.0 + L * 0.25, 0.25)
    o2 = OnsiteTerms(L)
    for i in [1, 4, 3, 5]:
        o2.add_onsite_term(strength2[i], i, f'Y_{i:d}')
    o2.add_onsite_term(strength2[3], 3, 'X_3')  # add to previous part
    o2.add_onsite_term(-strength1[1], 1, 'X_1')  # remove previous part
    o1 += o2
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"X_1": 0., "Y_1": strength2[1]},
                               {},
                               {"X_3": strength1[3] + strength2[3], "Y_3": strength2[3]},
                               {"Y_4": strength2[4]},
                               {"Y_5": strength2[5]}]  # fmt: skip
    o1.remove_zeros()
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"Y_1": strength2[1]},
                               {},
                               {"X_3": strength1[3]+ strength2[3], "Y_3": strength2[3]},
                               {"Y_4": strength2[4]},
                               {"Y_5": strength2[5]}]  # fmt: skip
    # TODO: skipped to/from TermList


def test_coupling_terms(backend=None):
    pytest.skip('Test needs to change')
    spin_half = lasy.models.SpinSite(0.5, conserve='Sz', backend=backend)
    L = 4
    sites = []
    for i in range(L):
        s = lasy.Site(spin_half.leg)
        s.add_op(f'X_{i:d}', 2.0 * np.eye(2))
        s.add_op(f'Y_{i:d}', 3.0 * np.eye(2))
        s.add_op('S1', 4.0 * np.eye(2))
        s.add_op('S2', 5.0 * np.eye(2))
        sites.append(s)
    strength1 = np.arange(0.0, 5)[:, np.newaxis] + np.arange(0.0, 0.625, 0.125)[np.newaxis, :]
    c1 = CouplingTerms(L)
    for i, j in [(2, 3)]:
        c1.add_coupling_term(strength1[i, j], i, j, f'X_{i:d}', f'Y_{j:d}')
    assert c1.max_range() == 3 - 2
    for i, j in [(0, 1), (0, 3), (0, 2)]:
        c1.add_coupling_term(strength1[i, j], i, j, f'X_{i:d}', f'Y_{j:d}')
    c1_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.125},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}}  # fmt: skip
    assert c1.coupling_terms == c1_des
    c1._test_terms(sites)
    assert c1.max_range() == 3 - 0
    tl1 = c1.to_TermList()
    term_list_des = [
        [('X_0', 0), ('Y_1', 1)],
        [('X_0', 0), ('Y_2', 2)],
        [('X_0', 0), ('Y_3', 3)],
        [('X_2', 2), ('Y_3', 3)],
    ]
    assert tl1.terms == term_list_des
    assert np.all(tl1.strength == [0.125, 0.25, 0.375, 2.375])
    ot1, ct1_conv = tl1.to_OnsiteTerms_CouplingTerms(sites)
    assert ot1.onsite_terms == [{}] * L
    assert ct1_conv.coupling_terms == c1_des

    mc = MultiCouplingTerms(L)
    for i, j in [(2, 3)]:  # exact same terms as c1
        mc.add_coupling_term(strength1[i, j], i, j, f'X_{i:d}', f'Y_{j:d}')
    assert mc.max_range() == 3 - 2
    for i, j in [(0, 1), (0, 3), (0, 2)]:  # exact same terms as c1
        mc.add_coupling_term(strength1[i, j], i, j, f'X_{i:d}', f'Y_{j:d}')
    # couplings now in left/ right structure from MultiCoupling

    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4]}}}  # fmt: skip
    t_des_R = {5: [1, 2],
               3: {('Y_3', 'Id'): {5: [3]}},
               2: {('Y_2', 'Id'): {5: [4]}}}  # fmt: skip
    c_des = [None,
             (3, 'Y_3', 0, 2.375),
             (1, 'Y_1', 0, 0.125),
             (2, 'Id', 0, 0.375),
             (1, 'Id', 0, 0.25),
            ]  # fmt: skip
    assert mc.terms_left == t_des_L
    assert mc.terms_right == t_des_R
    assert mc.connections == c_des
    assert mc.max_range() == 3 - 0
    mc.add_multi_coupling_term(20.0, [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['Id', 'Id'])
    mc.add_multi_coupling_term(30.0, [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['S1', 'S2'])
    mc.add_multi_coupling_term(40.0, [1, 2, 3], ['X_1', 'Y_2', 'Y_3'], ['Id', 'Id'])

    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4],
                                   1: {('Y_1', 'Id'): {-1: [5]}}},
                   ('X_0', 'S1'): {1: {('Y_1', 'S2'): {-1: [6]}}}},
               1: {('X_1', 'Id'): {-1: [7]}}}  # fmt: skip
    t_des_R = {5: [1, 2],
               3: {('Y_3', 'Id'): {5: [3, 5, 7]},
                   ('Y_3', 'S2'): {5: [6]}},
               2: {('Y_2', 'Id'): {5: [4]}}}  # fmt: skip
    c_des = [
        None,
        (3, 'Y_3', 0, 2.375),
        (1, 'Y_1', 0, 0.125),
        (2, 'Id', 0, 0.375),
        (1, 'Id', 0, 0.25),
        (2, 'Id', 0, 20.0),
        (2, 'S2', 0, 30.0),
        (2, 'Y_2', 0, 40.0),
    ]

    assert mc.terms_left == t_des_L
    assert mc.terms_right == t_des_R
    assert mc.connections == c_des
    mc._test_terms(sites)
    # convert to TermList
    tl_mc = mc.to_TermList()
    term_list_des = [
        [('X_2', 2), ('Y_3', 3)],
        [('X_0', 0), ('Y_1', 1)],
        [('X_0', 0), ('Y_3', 3)],
        [('X_0', 0), ('Y_2', 2)],
        [('X_0', 0), ('Y_1', 1), ('Y_3', 3)],
        [('X_0', 0), ('Y_1', 1), ('Y_3', 3)],  # (!) dropped S1, S2 (!)
        [('X_1', 1), ('Y_2', 2), ('Y_3', 3)],
    ]

    assert tl_mc.terms == term_list_des
    assert np.all(tl_mc.strength == [2.375, 0.125, 0.375, 0.25, 20.0, 30.0, 40.0])
    ot, mc_conv = tl_mc.to_OnsiteTerms_CouplingTerms(sites)
    assert ot1.onsite_terms == [{}] * L
    # conversion dropped the opstring names -> need to join previous counter 5/6
    del t_des_L[0][('X_0', 'S1')]
    del t_des_R[3][('Y_3', 'S2')]
    del c_des[6]
    c_des[5] = (2, 'Id', 0, 20.0 + 30.0)
    t_des_L[1][('X_1', 'Id')][-1] = [6]  # dropped previous counter 6, change counter of higher terms
    t_des_R[3][('Y_3', 'Id')][5][-1] = 6
    assert mc_conv.terms_left == t_des_L
    assert mc_conv.terms_right == t_des_R
    assert mc_conv.connections == c_des

    # addition
    c2 = CouplingTerms(L)
    for i, j in [(0, 1), (1, 2)]:
        c2.add_coupling_term(strength1[i, j], i, j, f'X_{i:d}', f'Y_{j:d}')
    c1 += c2
    c1_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.25},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}}},
              1: {('X_1', 'Id'): {2: {'Y_2': 1.25}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}}  # fmt: skip
    assert c1.coupling_terms == c1_des
    c1._test_terms(sites)
    mc += c2
    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4],
                                   1: {('Y_1', 'Id'): {-1: [5]}}},
                   ('X_0', 'S1'): {1: {('Y_1', 'S2'): {-1: [6]}}}},
               1: {('X_1', 'Id'): {-1: [7, 8]}}}  # fmt: skip

    t_des_R = {5: [1, 2, 8],
               3: {('Y_3', 'Id'): {5: [3, 5, 7]},
                   ('Y_3', 'S2'): {5: [6]}},
               2: {('Y_2', 'Id'): {5: [4]}}}  # fmt: skip
    c_des = [None,
             (3, 'Y_3', 0, 2.375),
             (1, 'Y_1', 0, 0.25),
             (2, 'Id', 0, 0.375),
             (1, 'Id', 0, 0.25),
             (2, 'Id', 0, 20.0),
             (2, 'S2', 0, 30.0),
             (2, 'Y_2', 0, 40.0),
             (2, 'Y_2', 0, 1.25)]  # fmt: skip
    assert mc.terms_left == t_des_L
    assert mc.terms_right == t_des_R
    assert mc.connections == c_des
    # coupling accross mps boundary
    mc.add_multi_coupling_term(0.05, [1, 3, 5], ['X_1', 'Y_3', 'Y_1'], ['STR', 'JW'])
    assert mc.max_range() == 5 - 1
    mc._test_terms(sites)
    # remove the last coupling again
    mc.remove_zeros(tol_zero=0.1)
    assert mc.terms_left == t_des_L
    assert mc.terms_right == t_des_R
    assert mc.connections == c_des + [None]
    assert mc.max_range() == 3 - 0


def test_coupling_terms_handle_JW():
    pytest.skip('coupling_term_handle_JW no longer exists. Instead do some fermion consistency checks')


@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_exp_decaying_terms(bc_MPS, cutoff=0.01):
    L = 8
    p, l = 3.0, 0.5
    cutoff_range = 8
    assert p * l**cutoff_range > cutoff > p * l ** (cutoff_range + 1)

    model = SpinChain(dict(L=L, bc_MPS=bc_MPS, conserve='Sz'))
    sites = model.lattice.mps_sites()

    # ======================================================================
    print('without subsites_start')
    # ======================================================================
    edt = ExponentiallyDecayingTerms(L)
    edt.add_exponentially_decaying_coupling(p, l, 'Sp.Sm', subsites=[0, 2, 4, 6])
    edt.test_terms(sites, model)

    # recreate manually, for comparison
    ct = CouplingTerms(L)
    if bc_MPS == 'finite':
        ct.add_coupling_term(p * l**1, 0, 2, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 0, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l**3, 0, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 2, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 2, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 4, 6, 'Sp.Sm')
    else:
        for i in [0, 2, 4, 6]:
            for k in range(1, cutoff_range + 1):
                ct.add_coupling_term(p * l**k, i, i + 2 * k, 'Sp.Sm')
    assert edt.to_CouplingTerms(model, cutoff=cutoff, bc=bc_MPS) == ct

    # check whether the MPO construction works by comparing MPOs
    H_expect = mpo.MPOGraph.from_terms([ct], model, bc=bc_MPS, unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc=bc_MPS, unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert H.is_equal(H_expect, eps=1e-10 if bc_MPS == 'finite' else cutoff)

    # ======================================================================
    print('with subsites_start (-1)')
    # ======================================================================
    edt = ExponentiallyDecayingTerms(L)
    edt.add_exponentially_decaying_coupling(p, l, 'Sp.Sm', subsites=[1, 3, 5, 7], subsites_start=[0, 2, 4, 6])
    edt.test_terms(sites, model)

    # recreate manually, for comparison
    ct = CouplingTerms(L)
    if bc_MPS == 'finite':
        ct.add_coupling_term(p * l**1, 0, 1, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 0, 3, 'Sp.Sm')
        ct.add_coupling_term(p * l**3, 0, 5, 'Sp.Sm')
        ct.add_coupling_term(p * l**4, 0, 7, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 2, 3, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 2, 5, 'Sp.Sm')
        ct.add_coupling_term(p * l**3, 2, 7, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 4, 5, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 4, 7, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 6, 7, 'Sp.Sm')
    else:
        for i in [0, 2, 4, 6]:
            for k in range(1, cutoff_range + 1):
                ct.add_coupling_term(p * l**k, i, i - 1 + 2 * k, 'Sp.Sm')
    assert edt.to_CouplingTerms(model, cutoff=cutoff, bc=bc_MPS) == ct

    # check whether the MPO construction works by comparing MPOs
    H_expect = mpo.MPOGraph.from_terms([ct], model, bc=bc_MPS, unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc=bc_MPS, unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert H.is_equal(H_expect, eps=1e-10 if bc_MPS == 'finite' else cutoff)

    # ======================================================================
    print('with subsites_start (+1)')
    # ======================================================================
    edt = ExponentiallyDecayingTerms(L)
    edt.add_exponentially_decaying_coupling(p, l, 'Sp.Sm', subsites=[0, 2, 4, 6], subsites_start=[1, 3, 5, 7])
    edt.test_terms(sites, model)

    # recreate manually, for comparison
    ct = CouplingTerms(L)
    if bc_MPS == 'finite':
        ct.add_coupling_term(p * l**1, 1, 2, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 1, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l**3, 1, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 3, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l**2, 3, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l**1, 5, 6, 'Sp.Sm')
    else:
        for i in [0, 2, 4, 6]:
            for k in range(1, cutoff_range + 1):
                ct.add_coupling_term(p * l**k, i + 1, i + 2 * k, 'Sp.Sm')
    assert edt.to_CouplingTerms(model, cutoff=cutoff, bc=bc_MPS) == ct

    # check whether the MPO construction works by comparing MPOs
    H_expect = mpo.MPOGraph.from_terms([ct], model, bc=bc_MPS, unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc=bc_MPS, unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert max(H.chi) == 3
    assert H.is_equal(H_expect, eps=1e-10 if bc_MPS == 'finite' else cutoff)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_exp_non_uniform_decaying_terms(bc):
    L = 8
    subsite_step = 2
    subsites = np.arange(0, L, subsite_step)
    cutoff = 1e-2
    model = SpinChain(dict(L=L, bc_MPS=bc, conserve='Sz'))
    sites = model.lattice.mps_sites()

    edt = ExponentiallyDecayingTerms(L)
    p = 3.0
    l = 1.0 / (1 + np.arange(L))
    edt.add_exponentially_decaying_coupling(p, l, 'Sp.Sm', subsites=subsites)
    edt.test_terms(sites, model)

    ct = CouplingTerms(L)
    if bc == 'finite':
        ct.add_coupling_term(p * l[0], 0, 2, 'Sp.Sm')
        ct.add_coupling_term(p * l[0] * l[2], 0, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l[0] * l[2] * l[4], 0, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l[2], 2, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l[2] * l[4], 2, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l[4], 4, 6, 'Sp.Sm')
    else:
        for n, i in enumerate(subsites):
            for m in range(n + 1, 1000):
                j = subsites[m % len(subsites)] + (m // len(subsites)) * L
                subsite_idcs = np.arange(n, m) % len(subsites)
                strength = p * np.prod(l[subsites[subsite_idcs]])
                if strength < cutoff:
                    break
                ct.add_coupling_term(strength, i, j, 'Sp.Sm')
    assert edt.to_CouplingTerms(model, cutoff=cutoff, bc=bc) == ct

    H_expect = mpo.MPOGraph.from_terms([ct], model, bc=bc, unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc=bc, unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert max(H.chi) == 3
    assert H.is_equal(H_expect, eps=1e-10 if bc == 'finite' else cutoff)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_exp_non_uniform_decaying_terms_subsites_start(bc):
    L = 8
    subsite_step = 2
    subsites = np.arange(0, L, subsite_step)
    subsites_start = np.arange(1, L, subsite_step)
    cutoff = 1e-2
    model = SpinChain(dict(L=L, bc_MPS=bc, conserve='Sz'))
    sites = model.lattice.mps_sites()

    edt = ExponentiallyDecayingTerms(L)
    p = 3.0
    l = 1.0 / (1 + np.arange(L))
    edt.add_exponentially_decaying_coupling(p, l, 'Sp.Sm', subsites=subsites, subsites_start=subsites_start)
    edt.test_terms(sites, model)

    ct = CouplingTerms(L)
    if bc == 'finite':
        ct.add_coupling_term(p * l[1], 1, 2, 'Sp.Sm')
        ct.add_coupling_term(p * l[1] * l[2], 1, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l[1] * l[2] * l[4], 1, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l[3], 3, 4, 'Sp.Sm')
        ct.add_coupling_term(p * l[3] * l[4], 3, 6, 'Sp.Sm')
        ct.add_coupling_term(p * l[5], 5, 6, 'Sp.Sm')
    else:
        for n, i in enumerate(subsites_start):
            for m in range(n + 1, 1000):
                j = subsites[m % len(subsites)] + (m // len(subsites)) * L
                subsite_idcs = np.arange(n + 1, m) % len(subsites)
                strength = p * np.prod([l[i]] + list(l[subsites[subsite_idcs]]))
                if strength < cutoff:
                    break
                ct.add_coupling_term(strength, i, j, 'Sp.Sm')
    assert edt.to_CouplingTerms(model, cutoff=cutoff, bc=bc) == ct

    H_expect = mpo.MPOGraph.from_terms([ct], model, bc=bc, unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc=bc, unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert max(H.chi) == 3
    assert H.is_equal(H_expect, eps=1e-10 if bc == 'finite' else cutoff)


@pytest.mark.parametrize('i', [0, 3, -1])
@pytest.mark.parametrize('subsites', [None, np.array([0, 3, 4, 6, 7])])
@pytest.mark.parametrize('uniform', [True, False])
def test_exponentially_decaying_centered_terms(i, subsites, uniform):
    L = 8
    model = SpinChain(dict(L=L, bc_MPS='finite', conserve='Sz'))
    sites = model.lattice.mps_sites()

    edt = ExponentiallyDecayingTerms(L)
    p = 3.0
    if uniform:
        l_arg = 0.5
        l_compare = np.full((L,), l_arg)
    else:
        l_arg = l_compare = 1.0 / (1 + np.arange(L))
    edt.add_centered_exponentially_decaying_term(p, l_arg, 'Sp.Sm', i, subsites=subsites)
    edt.test_terms(sites, model)

    if i < 0:
        i = i + L
    if subsites is None:
        subsites = np.arange(L)
    ct = CouplingTerms(L)
    for j in subsites:
        if j == i:
            continue
        if j < i:
            prefactor_sites = subsites[(subsites > j) & (subsites <= i)]
            strength = p * np.prod(l_compare[prefactor_sites])
            ct.add_coupling_term(strength, j, i, 'Sm.Sp')  # Note: always act with Sp on site i
        else:
            prefactor_sites = subsites[(subsites >= i) & (subsites < j)]
            strength = p * np.prod(l_compare[prefactor_sites])
            ct.add_coupling_term(strength, i, j, 'Sp.Sm')

    assert edt.to_CouplingTerms(model, cutoff=1e-6, bc='finite') == ct

    H_expect = mpo.MPOGraph.from_terms([ct], model, bc='finite', unit_cell_width=L).build_MPO()
    graph = mpo.MPOGraph.from_terms([edt], model, bc='finite', unit_cell_width=L)
    graph.test_sanity()
    H = graph.build_MPO()
    assert max(H.chi) == 3
    assert H.is_equal(H_expect, eps=1e-10)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_mpo_to_term_list(bc):
    pytest.skip('MPO.to_TermList not adapted yet.')
    # Addresses PR 477 / 479
    L = 6
    model = SpinChainNNN2(dict(bc_MPS=bc, Jx=0, Jy=0, Jz=1, Jxp=0, Jyp=0, Jzp=1, L=6, S=0.5))
    ts = model.H_MPO.to_TermList(['Id', 'Sp', 'Sm', 'Sz'])
    ts_expected = [[('Sz', i), ('Sz', i + 1)] for i in range(L - 1)]
    ts_expected += [[('Sz', i), ('Sz', i + 2)] for i in range(L - 2)]
    if bc == 'infinite':
        ts_expected += [[('Sz', L - 1), ('Sz', L)]]
        ts_expected += [[('Sz', L - 2), ('Sz', L)], [('Sz', L - 1), ('Sz', L + 1)]]
    # do not have correct order
    print(sorted(ts_expected))
    print()
    print(sorted(ts.terms))
    assert len(ts.terms) == len(ts_expected)
    assert sorted(ts.terms) == sorted(ts_expected)
