"""A collection of tests for :module:`tenpy.networks.terms`."""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import copy

from tenpy.networks.terms import *
from tenpy.networks import site
from tenpy.networks import mpo
from tenpy.linalg.np_conserved import LegCharge

spin_half = site.SpinHalfSite(conserve='Sz', sort_charge=True)
fermion = site.FermionSite(conserve='N')
dummy = site.Site(spin_half.leg)


def test_TermList():
    terms = [[('N', 3), ('N', 2), ('N', 3)], [('C', 0), ('N', 2), ('N', 4), ('Cd', 3), ('C', 2)],
             [('C', 0), ('N', 2), ('N', 4), ('Cd', 3), ('Cd', 0), ('C', 2)]]
    strength = [1., 2., 3.]
    terms_copy = copy.deepcopy(terms)
    terms_ordered = [[('N', 2), ('N N', 3)], [('C', 0), ('N C', 2), ('Cd', 3), ('N', 4)],
                     [('C Cd', 0), ('N C', 2), ('Cd', 3), ('N', 4)]]
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
    assert np.all(tl.strength == np.array([1., -2., 3.]))


def test_onsite_terms():
    L = 6
    strength1 = np.arange(1., 1. + L * 0.25, 0.25)
    o1 = OnsiteTerms(L)
    for i in [1, 0, 3]:
        o1.add_onsite_term(strength1[i], i, "X_{i:d}".format(i=i))
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"X_1": strength1[1]},
                               {},
                               {"X_3": strength1[3]},
                               {},
                               {}] # yapf: disable
    strength2 = np.arange(2., 2. + L * 0.25, 0.25)
    o2 = OnsiteTerms(L)
    for i in [1, 4, 3, 5]:
        o2.add_onsite_term(strength2[i], i, "Y_{i:d}".format(i=i))
    o2.add_onsite_term(strength2[3], 3, "X_3")  # add to previous part
    o2.add_onsite_term(-strength1[1], 1, "X_1")  # remove previous part
    o1 += o2
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"X_1": 0., "Y_1": strength2[1]},
                               {},
                               {"X_3": strength1[3] + strength2[3], "Y_3": strength2[3]},
                               {"Y_4": strength2[4]},
                               {"Y_5": strength2[5]}] # yapf: disable
    o1.remove_zeros()
    assert o1.onsite_terms == [{"X_0": strength1[0]},
                               {"Y_1": strength2[1]},
                               {},
                               {"X_3": strength1[3]+ strength2[3], "Y_3": strength2[3]},
                               {"Y_4": strength2[4]},
                               {"Y_5": strength2[5]}] # yapf: disable
    # convert to term_list
    tl = o1.to_TermList()
    assert tl.terms == [[("X_0", 0)], [("Y_1", 1)], [("X_3", 3)], [("Y_3", 3)], [("Y_4", 4)],
                        [("Y_5", 5)]]
    o3, c3 = tl.to_OnsiteTerms_CouplingTerms([dummy] * L)
    assert o3.onsite_terms == o1.onsite_terms


def test_coupling_terms():
    L = 4
    sites = []
    for i in range(L):
        s = site.Site(spin_half.leg)
        s.add_op("X_{i:d}".format(i=i), 2. * np.eye(2))
        s.add_op("Y_{i:d}".format(i=i), 3. * np.eye(2))
        s.add_op("S1", 4. * np.eye(2))
        s.add_op("S2", 5. * np.eye(2))
        sites.append(s)
    strength1 = np.arange(0., 5)[:, np.newaxis] + np.arange(0., 0.625, 0.125)[np.newaxis, :]
    c1 = CouplingTerms(L)
    for i, j in [(2, 3)]:
        c1.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    assert c1.max_range() == 3 - 2
    for i, j in [(0, 1), (0, 3), (0, 2)]:
        c1.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    c1_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.125},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}} # yapf: disable
    assert c1.coupling_terms == c1_des
    c1._test_terms(sites)
    assert c1.max_range() == 3 - 0
    tl1 = c1.to_TermList()
    term_list_des = [[('X_0', 0), ('Y_1', 1)], [('X_0', 0), ('Y_2', 2)], [('X_0', 0), ('Y_3', 3)],
                     [('X_2', 2), ('Y_3', 3)]]
    assert tl1.terms == term_list_des
    assert np.all(tl1.strength == [0.125, 0.25, 0.375, 2.375])
    ot1, ct1_conv = tl1.to_OnsiteTerms_CouplingTerms(sites)
    assert ot1.onsite_terms == [{}] * L
    assert ct1_conv.coupling_terms == c1_des

    mc = MultiCouplingTerms(L)
    for i, j in [(2, 3)]:  # exact same terms as c1
        mc.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    assert mc.max_range() == 3 - 2
    for i, j in [(0, 1), (0, 3), (0, 2)]:  # exact same terms as c1
        mc.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    #couplings now in left/ right structure from MultiCoupling

    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4]}}} # yapf: disable
    t_des_R = {5: [1, 2],
               3: {('Y_3', 'Id'): {5: [3]}},
               2: {('Y_2', 'Id'): {5: [4]}}} # yapf: disable
    c_des = [None,
             (3, 'Y_3', 0, 2.375),
             (1, 'Y_1', 0, 0.125),
             (2, 'Id', 0, 0.375),
             (1, 'Id', 0, 0.25),
            ]  # yapf: disable
    assert mc.terms_left == t_des_L
    assert mc.terms_right == t_des_R
    assert mc.connections == c_des
    assert mc.max_range() == 3 - 0
    mc.add_multi_coupling_term(20., [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['Id', 'Id'])
    mc.add_multi_coupling_term(30., [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['S1', 'S2'])
    mc.add_multi_coupling_term(40., [1, 2, 3], ['X_1', 'Y_2', 'Y_3'], ['Id', 'Id'])

    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4],
                                   1: {('Y_1', 'Id'): {-1: [5]}}},
                   ('X_0', 'S1'): {1: {('Y_1', 'S2'): {-1: [6]}}}},
               1: {('X_1', 'Id'): {-1: [7]}}}  # yapf: disable
    t_des_R = {5: [1, 2],
               3: {('Y_3', 'Id'): {5: [3, 5, 7]},
                   ('Y_3', 'S2'): {5: [6]}},
               2: {('Y_2', 'Id'): {5: [4]}}}  # yapf: disable
    c_des = [None,
             (3, 'Y_3', 0, 2.375),
             (1, 'Y_1', 0, 0.125),
             (2, 'Id', 0, 0.375),
             (1, 'Id', 0, 0.25),
             (2, 'Id', 0, 20.0),
             (2, 'S2', 0, 30.0),
             (2, 'Y_2', 0, 40.0)]

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
        [('X_1', 1), ('Y_2', 2), ('Y_3', 3)]
    ]

    assert tl_mc.terms == term_list_des
    assert np.all(tl_mc.strength == [2.375, 0.125, 0.375, 0.25, 20., 30., 40.])
    ot, mc_conv = tl_mc.to_OnsiteTerms_CouplingTerms(sites)
    assert ot1.onsite_terms == [{}] * L
    # conversion dropped the opstring names -> need to join previous counter 5/6
    del t_des_L[0][('X_0', 'S1')]
    del t_des_R[3][('Y_3', 'S2')]
    del c_des[6]
    c_des[5] = (2, 'Id', 0, 20.0 + 30.0)
    t_des_L[1][('X_1', 'Id')][-1] = [6] # dropped previous counter 6, change counter of higher terms
    t_des_R[3][('Y_3', 'Id')][5][-1] = 6
    assert mc_conv.terms_left == t_des_L
    assert mc_conv.terms_right == t_des_R
    assert mc_conv.connections == c_des

    # addition
    c2 = CouplingTerms(L)
    for i, j in [(0, 1), (1, 2)]:
        c2.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    c1 += c2
    c1_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.25},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}}},
              1: {('X_1', 'Id'): {2: {'Y_2': 1.25}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}} # yapf: disable
    assert c1.coupling_terms == c1_des
    c1._test_terms(sites)
    mc += c2
    t_des_L = {2: {('X_2', 'Id'): {-1: [1]}},
               0: {('X_0', 'Id'): {-1: [2, 3, 4],
                                   1: {('Y_1', 'Id'): {-1: [5]}}},
                   ('X_0', 'S1'): {1: {('Y_1', 'S2'): {-1: [6]}}}},
               1: {('X_1', 'Id'): {-1: [7, 8]}}}  # yapf: disable

    t_des_R = {5: [1, 2, 8],
               3: {('Y_3', 'Id'): {5: [3, 5, 7]},
                   ('Y_3', 'S2'): {5: [6]}},
               2: {('Y_2', 'Id'): {5: [4]}}}  # yapf: disable
    c_des = [None,
             (3, 'Y_3', 0, 2.375),
             (1, 'Y_1', 0, 0.25),
             (2, 'Id', 0, 0.375),
             (1, 'Id', 0, 0.25),
             (2, 'Id', 0, 20.0),
             (2, 'S2', 0, 30.0),
             (2, 'Y_2', 0, 40.0),
             (2, 'Y_2', 0, 1.25)]  # yapf:disable
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
    strength = 0.25
    sites = []
    L = 4
    for i in range(L):
        s = site.Site(spin_half.leg)
        s.add_op("X_{i:d}".format(i=i), 2. * np.eye(2))
        s.add_op("Y_{i:d}".format(i=i), 3. * np.eye(2), need_JW=True)
        sites.append(s)
    mc = MultiCouplingTerms(L)
    # two-site terms
    term = [("X_1", 1), ("X_0", 4)]
    args = mc.coupling_term_handle_JW(strength, term, sites)
    # args = i, j, op_i, op_j, op_str
    assert args == (strength, 1, 4, "X_1", "X_0", "Id")
    term = [("Y_1", 1), ("Y_0", 4)]
    args = mc.coupling_term_handle_JW(strength, term, sites)
    assert args == (strength, 1, 4, "Y_1 JW", "Y_0", "JW")

    # switch order
    term = [("Y_0", 4), ("Y_1", 1)]
    term, sign = order_combine_term(term, sites)
    assert term == [("Y_1", 1), ("Y_0", 4)]
    args = mc.coupling_term_handle_JW(strength * sign, term, sites)
    assert args == (-strength, 1, 4, "Y_1 JW", "Y_0", "JW")

    # multi coupling
    term = [("X_0", 0), ("X_1", 1), ("X_3", 3)]
    args = mc.multi_coupling_term_handle_JW(strength, term, sites)
    assert args == (strength, [0, 1, 3], ["X_0", "X_1", "X_3"], ["Id", "Id"])
    term = [("X_0", 0), ("Y_1", 1), ("Y_3", 3)]
    args = mc.multi_coupling_term_handle_JW(strength, term, sites)
    assert args == (strength, [0, 1, 3], ["X_0", "Y_1 JW", "Y_3"], ["Id", "JW"])

    term = [("Y_0", 0), ("X_1", 1), ("Y_3", 3), ("X_0", 4), ("Y_2", 6), ("Y_3", 7)]
    args = mc.multi_coupling_term_handle_JW(strength, term, [dummy] * 4)
    assert args == (strength, [0, 1, 3, 4, 6, 7], [op[0] for op in term], ["Id"] * (len(term) - 1))
    args = mc.multi_coupling_term_handle_JW(strength, term, sites)
    print(args)
    assert args == (strength, [0, 1, 3, 4, 6,
                               7], ["Y_0 JW", "X_1 JW", "Y_3", "X_0", "Y_2 JW",
                                    "Y_3"], ["JW", "JW", "Id", "Id", "JW"])

    term = [
        ("Y_3", 7),
        ("X_1", 1),
        ("Y_0", 0),
        ("X_0", 4),
        ("Y_2", 6),
        ("Y_3", 3),
    ]
    term, sign = order_combine_term(term, sites)
    args = mc.multi_coupling_term_handle_JW(strength * sign, term, sites)
    print(args)
    assert args == (strength, [0, 1, 3, 4, 6,
                               7], ["Y_0 JW", "X_1 JW", "Y_3", "X_0", "Y_2 JW",
                                    "Y_3"], ["JW", "JW", "Id", "Id", "JW"])


def test_exp_decaying_terms():
    L = 8
    spin = site.Site(spin_half.leg)
    spin.add_op("X", 2. * np.eye(2))
    spin.add_op("Y", 3. * np.eye(2))
    sites = [spin] * L
    edt = ExponentiallyDecayingTerms(L)
    p, l = 3., 0.5
    edt.add_exponentially_decaying_coupling(p, l, 'X', 'Y', subsites=[0, 2, 4, 6])
    edt._test_terms(sites)
    ts = edt.to_TermList(bc='finite', cutoff=0.01)
    ts_desired = [
        [("X", 0), ("Y", 2)],
        [("X", 0), ("Y", 4)],
        [("X", 0), ("Y", 6)],
        [("X", 2), ("Y", 4)],
        [("X", 2), ("Y", 6)],
        [("X", 4), ("Y", 6)],
    ]
    assert ts.terms == ts_desired
    assert np.all(ts.strength == p * np.array([l, l**2, l**3, l, l**2, l]))

    # check whether the MPO construction works by comparing MPOs
    # constructed from ts vs. directly
    H1 = mpo.MPOGraph.from_term_list(ts, sites, bc='finite').build_MPO()
    G = mpo.MPOGraph(sites, bc='finite')
    edt.add_to_graph(G)
    G.test_sanity()
    G.add_missing_IdL_IdR()
    H2 = G.build_MPO()
    assert H1.is_equal(H2)

    # check infinite versions
    cutoff = 0.01
    cutoff_range = 8
    assert p * l**cutoff_range > cutoff > p * l**(cutoff_range + 1)
    ts = edt.to_TermList(bc='infinite', cutoff=cutoff)
    ts_desired = ([[("X", 0), ("Y", 0 + 2 * i)] for i in range(1, cutoff_range + 1)] +
                  [[("X", 2), ("Y", 2 + 2 * i)] for i in range(1, cutoff_range + 1)] +
                  [[("X", 4), ("Y", 4 + 2 * i)] for i in range(1, cutoff_range + 1)] +
                  [[("X", 6), ("Y", 6 + 2 * i)] for i in range(1, cutoff_range + 1)])  # yapf: disable
    assert ts.terms == ts_desired
    strength_desired = np.tile(l**np.arange(1, cutoff_range + 1) * p, 4)
    assert np.all(ts.strength == strength_desired)
    G = mpo.MPOGraph(sites, bc='infinite')
    edt.add_to_graph(G)
    G.test_sanity()
    G.add_missing_IdL_IdR()
    H2 = G.build_MPO()
    H1 = mpo.MPOGraph.from_term_list(ts, sites, bc='infinite').build_MPO()
    assert H1.is_equal(H2, cutoff)
