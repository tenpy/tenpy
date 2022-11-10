"""test of :class:`tenpy.models.XXZChain`."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import pprint

import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain, XXZChain2
from test_model import check_general_model


def test_XXZChain():
    pars = dict(L=4, Jxx=1., Jz=1., hz=0., bc_MPS='finite', sort_charge=True)
    chain = XXZChain(pars)
    chain.test_sanity()
    for Hb in chain.H_bond[1:]:  # check bond eigenvalues
        Hb2 = Hb.combine_legs([['p0', 'p1'], ['p0*', 'p1*']], qconj=[+1, -1])
        print(Hb2.to_ndarray())
        W = npc.eigvalsh(Hb2)
        npt.assert_array_almost_equal_nulp(np.sort(W), np.sort([-0.75, 0.25, 0.25, 0.25]), 16**3)
    # now check with non-trivial onsite terms
    pars['hz'] = 0.2
    print("hz =", pars['hz'])
    chain = XXZChain(pars)
    chain.test_sanity()
    Hb = chain.H_bond[2]  # the only central bonds: boundaries have different hz.
    Hb2 = Hb.combine_legs([['p0', 'p1'], ['p0*', 'p1*']], qconj=[+1, -1])
    print(Hb2.to_ndarray())
    W = npc.eigvalsh(Hb2)
    print(W)
    npt.assert_array_almost_equal_nulp(
        np.sort(W),
        np.sort(
            [-0.75, 0.25 - 2 * 0.5 * 0.5 * pars['hz'], 0.25, 0.25 + 2. * 0.5 * 0.5 * pars['hz']]),
        16**3)
    pars['bc_MPS'] = 'infinite'

    for L in [2, 3, 4, 5, 6]:
        print("L =", L)
        pars['L'] = L
        chain = XXZChain(pars)
        pprint.pprint(chain.coupling_terms)
        assert len(chain.H_bond) == L
        Hb0 = chain.H_bond[0]
        for Hb in chain.H_bond[1:]:
            assert (npc.norm(Hb - Hb0, np.inf) == 0.)  # exactly equal
    pars['Jxx'] = 0.
    chain = XXZChain(pars)
    chain.test_sanity()


def test_XXZChain_general(tol=1.e-14):
    check_general_model(XXZChain, dict(L=4, Jxx=1., hz=0., bc_MPS='finite', sort_charge=True), {
        'Jz': [0., 1., 2.],
        'hz': [0., 0.2]
    })
    check_general_model(XXZChain2, dict(L=4, Jxx=1., hz=0., bc_MPS='finite', sort_charge=True), {
        'Jz': [0., 1., 2.],
        'hz': [0., 0.2]
    })
    model_param = dict(L=3, Jxx=1., Jz=1.5, hz=0.25, bc_MPS='finite', sort_charge=True)
    m1 = XXZChain(model_param)
    m2 = XXZChain2(model_param)
    for Hb1, Hb2 in zip(m1.H_bond, m2.H_bond):
        if Hb1 is None:
            assert Hb2 is None
            continue
        assert npc.norm(Hb1 - Hb2) < tol
