"""A collection of tests for :module:`tenpy.networks.terms`.

"""
# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.networks.terms import OnsiteTerms, CouplingTerms, MultiCouplingTerms
from tenpy.networks import site

spin_half = site.SpinHalfSite(conserve='Sz')


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
    o3, c3 = tl.to_OnsiteTerms_CouplingTerms([None] * L)
    assert o3.onsite_terms == o1.onsite_terms


def test_coupling_terms():
    L = 4
    sites = []
    for i in range(L):
        s = site.Site(spin_half.leg)
        s.add_op("X_{i:d}".format(i=i), 2. * np.eye(2))
        s.add_op("Y_{i:d}".format(i=i), 3. * np.eye(2))
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
    assert mc.coupling_terms == c1_des
    assert mc.max_range() == 3 - 0
    mc.add_multi_coupling_term(20., [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['Id', 'Id'])
    mc.add_multi_coupling_term(30., [0, 1, 3], ['X_0', 'Y_1', 'Y_3'], ['S1', 'S2'])
    mc.add_multi_coupling_term(40., [1, 2, 3], ['X_1', 'Y_2', 'Y_3'], ['Id', 'Id'])
    mc_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.125,
                                      ('Y_1', 'Id'): {3: {'Y_3': 20.0}}},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}},
                  ('X_0', 'S1'): {1: {('Y_1', 'S2'): {3: {'Y_3': 30.0}}}}},
              1: {('X_1', 'Id'): {2: {('Y_2', 'Id'): {3: {'Y_3': 40.0}}}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}} # yapf: disable
    assert mc.coupling_terms == mc_des
    mc._test_terms(sites)
    # convert to TermList
    tl_mc = mc.to_TermList()
    term_list_des = [
        [('X_0', 0), ('Y_1', 1)],
        [('X_0', 0), ('Y_1', 1), ('Y_3', 3)],
        [('X_0', 0), ('Y_2', 2)],
        [('X_0', 0), ('Y_3', 3)],
        [('X_0', 0), ('Y_1', 1), ('Y_3', 3)],  # (!) droppend S1, S2 (!)
        [('X_1', 1), ('Y_2', 2), ('Y_3', 3)],
        [('X_2', 2), ('Y_3', 3)]
    ]
    assert tl_mc.terms == term_list_des
    assert np.all(tl_mc.strength == [0.125, 20., 0.25, 0.375, 30., 40., 2.375])
    ot, mc_conv = tl_mc.to_OnsiteTerms_CouplingTerms(sites)
    assert ot1.onsite_terms == [{}] * L
    del (mc_des[0])[('X_0', 'S1')]  # conversion dropped the opstring names
    mc_des[0][('X_0', 'Id')][1][('Y_1', 'Id')][3]['Y_3'] += 30.  # add it to other term
    assert mc_conv.coupling_terms == mc_des

    # addition
    c2 = CouplingTerms(L)
    for i, j in [(0, 1), (1, 2)]:
        c1.add_coupling_term(strength1[i, j], i, j, "X_{i:d}".format(i=i), "Y_{j:d}".format(j=j))
    c1 += c2
    c1_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.25},
                                  2: {'Y_2': 0.25},
                                  3: {'Y_3': 0.375}}},
              1: {('X_1', 'Id'): {2: {'Y_2': 1.25}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 2.375}}}} # yapf: disable
    assert c1.coupling_terms == c1_des
    c1._test_terms(sites)
    mc += c1
    mc_des = {0: {('X_0', 'Id'): {1: {'Y_1': 0.375,
                                      ('Y_1', 'Id'): {3: {'Y_3': 20.0}}},
                                  2: {'Y_2': 0.5},
                                  3: {'Y_3': 0.75}},
                  ('X_0', 'S1'): {1: {('Y_1', 'S2'): {3: {'Y_3': 30.0}}}}},
              1: {('X_1', 'Id'): {2: {'Y_2': 1.25,
                                      ('Y_2', 'Id'): {3: {'Y_3': 40.0}}}}},
              2: {('X_2', 'Id'): {3: {'Y_3': 4.75}}}} # yapf: disable
    assert mc.coupling_terms == mc_des
    # coupling accross mps boundary
    mc.add_multi_coupling_term(50., [1, 3, 5], ['X_1', 'Y_3', 'Y_1'], ['STR', 'JW'])
    assert mc.max_range() == 5 - 1
    mc._test_terms(sites)
    # remove the last coupling again
    mc.add_multi_coupling_term(-50., [1, 3, 5], ['X_1', 'Y_3', 'Y_1'], ['STR', 'JW'])
    mc.remove_zeros()
    assert mc.coupling_terms == mc_des
    assert mc.max_range() == 3 - 0


def test_coupling_terms_handle_JW():
    mc = MultiCouplingTerms(4)
    # two-site terms
    term = [("X_1", 1), ("X_0", 4)]
    args = mc.coupling_term_handle_JW(term, [False, False])
    # args = i, j, op_i, op_j, op_str
    assert args == (1, 4, "X_1", "X_0", "Id")
    args = mc.coupling_term_handle_JW(term, [True, True])
    assert args == (1, 4, "X_1 JW", "X_0", "JW")

    # switch order
    term = [("X_0", 4), ("X_1", 1)]
    args = mc.coupling_term_handle_JW(term, [False, False])
    assert args == (1, 4, "X_1", "X_0", "Id")
    args = mc.coupling_term_handle_JW(term, [True, True])
    assert args == (1, 4, "JW X_1", "X_0", "JW")

    # multi coupling
    term = [("X_0", 0), ("Y_1", 1), ("Y_3", 3)]
    args = mc.multi_coupling_term_handle_JW(term, [False, False, False])
    assert args == ([0, 1, 3], ["X_0", "Y_1", "Y_3"], ["Id", "Id"])
    args = mc.multi_coupling_term_handle_JW(term, [False, True, True])
    assert args == ([0, 1, 3], ["X_0", "Y_1 JW", "Y_3"], ["Id", "JW"])

    term = [("Y_0", 0), ("X_1", 1), ("Y_3", 3), ("X_4", 4), ("Y_6", 6), ("Y_7", 7)]
    args = mc.multi_coupling_term_handle_JW(term, [False] * 6)
    assert args == ([0, 1, 3, 4, 6, 7], [op[0] for op in term], ["Id"] * (len(term) - 1))
    args = mc.multi_coupling_term_handle_JW(term, [True, False, True, False, True, True])
    print(args)
    assert args == ([0, 1, 3, 4, 6, 7], ["Y_0 JW", "X_1 JW", "Y_3", "X_4", "Y_6 JW",
                                         "Y_7"], ["JW", "JW", "Id", "Id", "JW"])

    term = [
        ("Y_7", 7),
        ("X_1", 1),
        ("Y_0", 0),
        ("X_4", 4),
        ("Y_6", 6),
        ("Y_3", 3),
    ]
    args = mc.multi_coupling_term_handle_JW(term, [True, False, True, False, True, True])
    print(args)
    assert args == ([0, 1, 3, 4, 6, 7], ["JW Y_0", "JW X_1", "Y_3", "JW X_4 JW", "JW Y_6",
                                         "Y_7"], ["JW", "JW", "Id", "Id", "JW"])
