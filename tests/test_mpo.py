"""A collection of tests for (classes in) :module:`tenpy.networks.mpo`.

"""
# Copyright 2018 TeNPy Developers

import numpy as np
import numpy.testing as npt
import nose.tools as nst
from tenpy.models.xxz_chain import XXZChain

from tenpy.linalg import np_conserved as npc

from tenpy.networks import mps, mpo, site

spin_half = site.SpinHalfSite(conserve='Sz')


def check_hermitian(H):
    """Check if `H` is a hermitian MPO."""
    if not H.finite:
        # include once over the boundary: double the unit cell
        # a general MPO might have terms going over multiple unit cells, but we ignore that...
        Ws = H._W * 2
    else:
        Ws = H._W
    #check trace(H.H) = trace(H.H^dagger):  equivalent to H=H^dagger
    W = Ws[0].take_slice([H.get_IdL(0)], ['wL'])

    trHH = npc.tensordot(W, W.replace_label('wR', 'wR*'), axes=[['p', 'p*'], ['p*', 'p']])
    trHHd = npc.tensordot(W, W.conj(), axes=[['p', 'p*'], ['p*', 'p']])
    for W in Ws[1:]:
        trHH = npc.tensordot(trHH, W, axes=['wR', 'wL'])
        trHHd = npc.tensordot(trHHd, W, axes=['wR', 'wL'])
        trHH = npc.tensordot(
            trHH, W.replace_label('wR', 'wR*'), axes=[['wR*', 'p', 'p*'], ['wL', 'p*', 'p']])
        trHHd = npc.tensordot(trHHd, W.conj(), axes=[['wR*', 'p', 'p*'], ['wL*', 'p*', 'p']])
    i = H.get_IdR(H.L - 1)
    trHH = trHH[i, i]
    trHHd = trHHd[i, i]
    print("check_hermitian: ", trHH, trHHd)
    npt.assert_array_almost_equal_nulp(trHH, trHHd, H.L * 20)


def test_MPO():
    s = spin_half
    for bc in mpo.MPO._valid_bc:
        for L in [4, 2, 1]:
            print(bc, ", L =", L)
            grid = [[s.Id, s.Sp, s.Sm, s.Sz],
                    [None, None, None, s.Sm],
                    [None, None, None, s.Sp],
                    [None, None, None, s.Id]]
            legW = npc.LegCharge.from_qflat(s.leg.chinfo, [[0], s.Sp.qtotal, s.Sm.qtotal, [0]])
            W = npc.grid_outer(grid, [legW, legW.conj()])
            W.iset_leg_labels(['wL', 'wR', 'p', 'p*'])
            Ws = [W] * L
            if bc == 'finite':
                Ws[0] = Ws[0][0:1, :, :, :]
                Ws[-1] = Ws[-1][:, 3:4, :, :]
            H = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L))
            H.test_sanity()
            print(H.dim)
            print(H.chi)
            check_hermitian(H)
        if L == 4:
            H2 = H.group_sites(n=2)
            H2.test_sanity()
            assert H2.L == 2


def test_MPOGraph():
    for bc in ['finite', 'infinite']:
        for L in [2, 4]:
            print("L =", L)
            g = mpo.MPOGraph([spin_half] * L, bc)
            g.add(0, 'IdL', 'IdR', 'Sz', 0.1)
            g.add(0, 'IdL', 'Sz0', 'Sz', 1.)
            g.add(1, 'Sz0', 'IdR', 'Sz', 0.5)
            g.add(0, 'IdL', (0, 'Sp'), 'Sp', 0.3)
            g.add(1, (0, 'Sp'), 'IdR', 'Sm', 0.2)
            if L > 2:
                g.add_string(0, 3, (0, 'Sp'), 'Id')
                g.add(3, (0, 'Sp'), 'IdR', 'Sm', 0.1)
            g.add_missing_IdL_IdR()
            g.test_sanity()
            print(repr(g))
            print(str(g))
            print("build MPO")
            g_mpo = g.build_MPO()
            g_mpo.test_sanity()


def test_MPOGraph_term_conversion():
    L = 4

    g1= mpo.MPOGraph([spin_half] * L, 'infinite')
    g1.test_sanity()
    for i in range(L):
        g1.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g1.add(i, 'IdL', (i, 'Sp', 'Id'), 'Sp', 1.)
        g1.add(i+1, (i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g1.add_missing_IdL_IdR()
    terms = [[("Sz", i)] for i in range(L)]
    terms += [[("Sp", i), ("Sm", i+1)] for i in range(L)]
    prefactors = [0.5] * L + [1.5] * L
    g2, ot, ct = mpo.MPOGraph.from_term_list(terms, prefactors, [spin_half] * L, 'infinite')
    g2.test_sanity()
    assert g1.graph == g2.graph
    terms[3:3] = [[("Sm", 2), ("Sp", 0), ("Sz", 1)]]
    prefactors[3:3] = [3.]
    g3, ot2, ct2 = mpo.MPOGraph.from_term_list(terms, prefactors, [spin_half] * L, 'infinite')
    g1.add(1, (0, 'Sp', 'Id'), (0, 'Sp', 'Id', 1, 'Sz', 'Id'), 'Sz', 1.)
    g1.add(2, (0, 'Sp', 'Id', 1, 'Sz', 'Id'), 'IdR', 'Sm', 3.)
    assert g1.graph == g3.graph


def test_MPOEnvironment():
    xxz_pars = dict(L=4, Jxx=1., Jz=1.1, hz=0.1, bc_MPS='finite')
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1] * L)[:L]  # Neel
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc='finite')
    env = mpo.MPOEnvironment(psi, M.H_MPO, psi)
    env.get_LP(3, True)
    env.get_RP(0, True)
    env.test_sanity()
    E_old = None
    for i in range(4):
        E = env.full_contraction(i)  # should be one
        print("total energy for contraction at site ", i, ": E =", E)
        if E_old is not None:
            assert (abs(E - E_old) < 1.e-14)
        E_old = E


def test_MPO_expectation_value():
    s = spin_half
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='infinite')
    psi1.test_sanity()
    ot = mps.OnsiteTerms(4)
    ot.add_onsite_term(0.1, 0, 'Sz') # -> 0.5
    ot.add_onsite_term(0.2, 3, 'Sz') # -> 0.
    ct = mps.CouplingTerms(4) # note: ct.L != psi1.L
    ct.add_coupling_term(1., 2, 3, 'Sz', 'Sz') # -> 0.
    ct.add_coupling_term(1.5, 1, 3, 'Sz', 'Sz') # -> 1.5*(-0.25)
    ct.add_coupling_term(2.5, 0, 6, 'Sz', 'Sz') # -> 2.5*0.25
    H = mpo.MPOGraph.from_terms(ot, ct, [s]*4, 'infinite').build_MPO()
    ev = H.expectation_value(psi1)
    desired_ev = (0.1*0.5 + 0.2*0. + 1.*0. + 1.5*-0.25 + 2.5 * 0.25) / H.L
    assert abs(ev - desired_ev) < 1.e-8
    grid = [[s.Id, s.Sz, 3*s.Sz],
            [None, 0.1*s.Id, s.Sz],
            [None, None, s.Id]]
    L = 1
    exp_dec_H = mpo.MPO.from_grids([s]*L, [grid]*L, bc='infinite', IdL=0, IdR=2)
    ev = exp_dec_H.expectation_value(psi1)
    desired_ev = 3*0.5 + 0.25*0.1**(4-1) + 0.25*0.1**(6-1) + 0.25 * 0.1**(10-1)  # values > 1.e-15
    assert abs(ev - desired_ev) < 1.e-15
    L = 3
    exp_dec_H = mpo.MPO.from_grids([s]*L, [grid]*L, bc='infinite', IdL=0, IdR=2)
    ev = exp_dec_H.expectation_value(psi1)
    desired_ev = (desired_ev + # first site
                  3*0. - 0.25 * 0.1**(3-1-1)  +  # second site
                  3*0. - 0.25 * 0.1**(5-2-1)) / 3.
    print("ev = ", ev, "desired", desired_ev)
    assert abs(ev - desired_ev) < 1.e-14

if __name__ == "__main__":
    test_MPOGraph_term_conversion()
