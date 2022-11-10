"""A collection of tests for (classes in) :module:`tenpy.networks.mpo`."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import pytest
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.spins import SpinChain
from tenpy.linalg import np_conserved as npc
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks import mps, mpo, site
from tenpy.networks.terms import OnsiteTerms, CouplingTerms, MultiCouplingTerms, TermList

from random_test import random_MPS

spin_half = site.SpinHalfSite(conserve='Sz', sort_charge=False)


def test_MPO():
    s = spin_half
    for bc in mpo.MPO._valid_bc:
        for L in [4, 2, 1]:
            print(bc, ", L =", L)
            grid = [[s.Id, s.Sp, s.Sm, s.Sz], [None, None, None, s.Sm], [None, None, None, s.Sp],
                    [None, None, None, s.Id]]
            legW = npc.LegCharge.from_qflat(s.leg.chinfo, [[0], s.Sp.qtotal, s.Sm.qtotal, [0]])
            W = npc.grid_outer(grid, [legW, legW.conj()], grid_labels=['wL', 'wR'])
            Ws = [W] * L
            if bc == 'finite':
                Ws[0] = Ws[0][0:1, :, :, :]
                Ws[-1] = Ws[-1][:, 3:4, :, :]
            H = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L))
            H_copy = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L))
            H.test_sanity()
            print(H.dim)
            print(H.chi)
            assert H.is_equal(H)  # everything should be equal to itself
            assert H.is_hermitian()
            H.sort_legcharges()
            H.test_sanity()
            assert H.is_equal(H_copy)
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
            if L > 2 or bc == 'infinite':
                keyR = g.add_string_left_to_right(0, 3, (0, 'Sp'), 'Id')
                g.add(3, keyR, 'IdR', 'Sm', 0.1)
            g.add_missing_IdL_IdR()
            g.test_sanity()
            print(repr(g))
            print(str(g))
            print("build MPO")
            g_mpo = g.build_MPO()
            g_mpo.test_sanity()


def test_MPOGraph_term_conversion():
    L = 4

    g1 = mpo.MPOGraph([spin_half] * L, 'infinite')
    g1.test_sanity()
    for i in range(L):
        g1.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g1.add(i, 'IdL', ("left", i, 'Sp', 'Id'), 'Sp', 1.)
        g1.add(i + 1, ("left", i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g1.add_missing_IdL_IdR()
    terms = [[("Sz", i)] for i in range(L)]
    terms += [[("Sp", i), ("Sm", i + 1)] for i in range(L)]
    prefactors = [0.5] * L + [1.5] * L
    term_list = TermList(terms, prefactors)
    g2 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite')
    g2.test_sanity()
    assert g1.graph == g2.graph
    terms[3:3] = [[("Sm", 2), ("Sp", 0), ("Sz", 1)]]
    prefactors[3:3] = [3.]
    term_list = TermList(terms, prefactors)
    g3 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite')
    g4 = mpo.MPOGraph([spin_half] * L, 'infinite')
    g4.test_sanity()
    for i in range(L):
        g4.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g4.add(i, 'IdL', ("left", i, 'Sp', 'Id'), 'Sp', 1.)
        g4.add(i + 1, ("left", i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g4.add_missing_IdL_IdR()
    g4.add(1, ("left", 0, 'Sp', 'Id'), ("right", 2, 'Sm', 'Id'), 'Sz', 3.)
    g4.add(2, ("right", 2, 'Sm', 'Id'), 'IdR', 'Sm', 1.)
    assert g4.graph == g3.graph


def test_MPO_conversion():
    L = 8
    sites = []
    for i in range(L):
        s = site.Site(spin_half.leg)
        s.add_op("X_{i:d}".format(i=i), np.diag([2., 1.]))
        s.add_op("Y_{i:d}".format(i=i), np.diag([1., 2.]))
        sites.append(s)
    terms = [
        [("X_0", 0)],
        [("X_0", 0), ("X_1", 1)],
        [("X_0", 0), ("X_3", 3)],
        [("X_4", 4), ("Y_5", 5), ("Y_7", 7)],
        [("X_4", 4), ("Y_5", 5), ("X_7", 7)],
        [("X_4", 4), ("Y_6", 6), ("Y_7", 7)],
    ]
    prefactors = [0.25, 10., 11., 101., 102., 103.]
    term_list = TermList(terms, prefactors)
    g1 = mpo.MPOGraph.from_term_list(term_list, sites, bc='finite', insert_all_id=False)
    ct_add = MultiCouplingTerms(L)
    ct_add.add_coupling_term(12., 4, 5, "X_4", "X_5")
    ct_add.add_multi_coupling_term(0.5, [4, 5, 7], ["X_4", "Y_5", "X_7"], "Id")
    ct_add.add_to_graph(g1)
    H1 = g1.build_MPO()
    grids = [
        [['Id', 'X_0', [('X_0', 0.25)]]  # site 0
         ],
        [
            ['Id', None, None],  # site 1
            [None, 'Id', [('X_1', 10.0)]],
            [None, None, 'Id']
        ],
        [
            ['Id', None, None],  # site 2
            [None, [('Id', 11.0)], None],
            [None, None, 'Id']
        ],
        [
            ['Id', None],  # site 3
            [None, 'X_3'],
            [None, 'Id']
        ],
        [
            ['X_4', None],  #site 4
            [None, 'Id']
        ],
        [
            ['Id', 'Y_5', [('X_5', 12.0)]],  # site 5
            [None, None, 'Id']
        ],
        [
            # site 6
            [None, [('Y_6', 103.0)], None],
            [[('Id', 102.5)], [('Id', 101.0)], None],
            [None, None, 'Id']
        ],
        [
            ['X_7'],  # site 7
            ['Y_7'],
            ['Id']
        ]
    ]
    H2 = mpo.MPO.from_grids(sites, grids, 'finite', [0] * 4 + [None] * 5, [None] * 5 + [-1] * 4)
    for w1, w2 in zip(H1._W, H2._W):
        assert npc.norm(w1 - w2, np.inf) == 0.
    pass


def test_MPOEnvironment():
    xxz_pars = dict(L=4, Jxx=1., Jz=1.1, hz=0.1, bc_MPS='finite', sort_charge=True)
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1] * L)[:L]  # Neel
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc='finite')
    env = mpo.MPOEnvironment(psi, M.H_MPO, psi)
    env.get_LP(3, True)
    env.get_RP(0, True)
    env.test_sanity()
    E_exact = -0.825
    for i in range(4):
        E = env.full_contraction(i)  # should be one
        print("total energy for contraction at site ", i, ": E =", E)
        assert (abs(E - E_exact) < 1.e-14)


def test_MPO_hermitian():
    s = spin_half
    ot = OnsiteTerms(4)
    ct = CouplingTerms(4)
    ct.add_coupling_term(1., 2, 3, 'Sm', 'Sp')
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'infinite').build_MPO()
    assert not H.is_hermitian()
    assert H.is_equal(H)
    ct.add_coupling_term(1., 2, 3, 'Sp', 'Sm')
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'infinite').build_MPO()
    assert H.is_hermitian()
    assert H.is_equal(H)

    ct.add_coupling_term(1., 3, 18, 'Sm', 'Sp')
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'infinite').build_MPO()
    assert not H.is_hermitian()
    assert H.is_equal(H)
    ct.add_coupling_term(1., 3, 18, 'Sp', 'Sm')
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'infinite').build_MPO()
    assert H.is_hermitian()
    assert H.is_equal(H)


def test_MPO_addition():
    for bc in ['infinite', 'finite']:
        print('bc = ', bc, '-' * 40)
        s = spin_half
        ot1 = OnsiteTerms(4)
        ct1 = CouplingTerms(4)
        ct1.add_coupling_term(2., 2, 3, 'Sm', 'Sp')
        ct1.add_coupling_term(2., 2, 3, 'Sp', 'Sm')
        ct1.add_coupling_term(2., 1, 2, 'Sz', 'Sz')
        ot1.add_onsite_term(3., 1, 'Sz')
        H1 = mpo.MPOGraph.from_terms((ot1, ct1), [s] * 4, bc).build_MPO()
        ot2 = OnsiteTerms(4)
        ct2 = CouplingTerms(4)
        ct2.add_coupling_term(4., 0, 2, 'Sz', 'Sz')
        ct2.add_coupling_term(4., 1, 2, 'Sz', 'Sz')
        ot2.add_onsite_term(5., 1, 'Sz')
        H2 = mpo.MPOGraph.from_terms((ot2, ct2), [s] * 4, bc).build_MPO()
        H12_sum = H1 + H2
        ot12 = OnsiteTerms(4)
        ot12 += ot1
        ot12 += ot2
        ct12 = CouplingTerms(4)
        ct12 += ct1
        ct12 += ct2
        H12 = mpo.MPOGraph.from_terms((ot12, ct12), [s] * 4, bc).build_MPO()
        assert H12.is_equal(H12_sum)


def test_MPO_expectation_value(tol=1.e-15):
    s = spin_half
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='infinite')
    psi1.test_sanity()
    ot = OnsiteTerms(4) # H.L != psi.L, consider L = lcm(4, 6) = 12
    ot.add_onsite_term(0.1, 0, 'Sz')  # -> 0.5 * 2 (sites 0, 4, not 8)
    ot.add_onsite_term(0.2, 3, 'Sz')  # -> 0.  (not sites 4, 7, 11)
    ct = CouplingTerms(4)  # note: ct.L != psi1.L
    ct.add_coupling_term(1., 2, 3, 'Sz', 'Sz')  # -> 0. (not 2-3, 6-7, 10-11)
    ct.add_coupling_term(1.5, 1, 3, 'Sz', 'Sz')  # -> 1.5*(-0.25) (1-3, not 5-7, not 9-11)
    ct.add_coupling_term(2.5, 0, 6, 'Sz', 'Sz')  # -> 2.5*0.25*3 (0-6, 4-10, not 8-14)
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'infinite').build_MPO()
    desired_ev = (0.1 * 0.5 * 2 + 0.2 * 0. + 1. * 0. + 1.5 * -0.25 + 2.5 * 0.25 * 2) / 12
    ev_power = H.expectation_value_power(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    ev_TM = H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_TM - desired_ev) < tol
    ev = H.expectation_value(psi1, tol)
    assert abs(ev - desired_ev) < tol

    # now with exponentially decaying term
    grid = [
        [s.Id, s.Sz, 3 * s.Sz],
        [None, 0.1 * s.Id, s.Sz],
        [None, None, s.Id],
    ]
    L = 1
    exp_dec_H = mpo.MPO.from_grids([s] * L, [grid] * L, bc='infinite', IdL=0, IdR=2)
    desired_ev = (3 * 0.5 * 2 + # Sz onsite
                  0.25 * (0.1**3 + 0.1**5 + 0.1**9 + 0.1**11 + 0.1**15 +  # Z_0 Z_i
                          0.1**1 + 0.1**5 + 0.1**7 + 0.1**11 + 0.1**13) +  # Z_4 Z_i
                  -0.25 * (0.1**1 +  # Z_1 Z_3
                           0.1**2)  # Z_2 Z_4
                  + 0.  # other sites
                  ) / 6.  # values > 1.e-15
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol
    L = 3  # should give exactly the same answer!
    exp_dec_H = mpo.MPO.from_grids([s] * L, [grid] * L, bc='infinite', IdL=0, IdR=2)
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol


def test_MPO_var(L=8, tol=1.e-13):
    xxz_pars = dict(L=L, Jx=1., Jy=1., Jz=1.1, hz=0.1, bc_MPS='finite', conserve=None)
    M = SpinChain(xxz_pars)
    psi = random_MPS(L, 2, 10)
    exp_val = M.H_MPO.expectation_value(psi)

    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    psi_full = ED.mps_to_full(psi)
    exp_val_full = npc.inner(psi_full,
                             npc.tensordot(ED.full_H, psi_full, axes=1),
                             axes='range',
                             do_conj=True)
    assert abs(exp_val - exp_val_full) / abs(exp_val_full) < tol

    Hsquared = M.H_MPO.variance(psi, 0.)

    Hsquared_full = npc.inner(psi_full,
                              npc.tensordot(ED.full_H,
                                            npc.tensordot(ED.full_H, psi_full, axes=1),
                                            axes=1),
                              axes='range',
                              do_conj=True)
    assert abs(Hsquared - Hsquared_full) / abs(Hsquared_full) < tol
    var = M.H_MPO.variance(psi)
    var_full = Hsquared_full - exp_val_full**2
    assert abs(var - var_full) / abs(var_full) < tol


@pytest.mark.parametrize('method', ['SVD', 'variational', 'zip_up'])
def test_apply_mpo(method):
    bc_MPS = "finite"
    # NOTE: overlap doesn't work for calculating the energy (density) in infinite systems!
    # energy is extensive, overlap exponential....
    L = 5
    g = 0.5
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L)  # pointing in (-x)-direction
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    H = M.H_MPO
    Eexp = H.expectation_value(psi)
    psi2 = psi.copy()
    options = {'compression_method': method, 'trunc_params': {'chi_max': 50}}
    H.apply(psi2, options)
    Eapply = psi2.overlap(psi)
    assert abs(Eexp - Eapply) < 1e-5
    psi3 = psi.copy()
    H.apply(psi3, options)
    Eapply3 = psi3.overlap(psi)
    assert abs(Eexp - Eapply3) < 1e-5


def test_MPOTransferMatrix(eps=1.e-13):
    s = spin_half
    # exponential decay in Sz term to make it harder
    gamma = 0.5
    Jxy, Jz = 4., 1.
    hz = 0.
    grid = [[s.Id, s.Sp, s.Sm, s.Sz, hz * s.Sz],
            [None, None, None, None, Jxy*0.5*s.Sm],
            [None, None, None, None, Jxy*0.5*s.Sp],
            [None, None, None, gamma*s.Id, Jz*s.Sz],
            [None, None, None, None, s.Id]]  # yapf: disable
    H = mpo.MPO.from_grids([s] * 3, [grid] * 3, 'infinite', 0, 4, max_range=np.inf)
    psi = mps.MPS.from_singlets(s, 3, [(0, 1)], lonely=[2], bc='infinite')
    psi.roll_mps_unit_cell(-1)  # -> nontrivial chi at the cut between unit cells
    exact_E = ((-0.25 - 0.25) * Jxy  # singlet <0.5*Sp_i Sm_{i+1}> = 0.25 = <0.5*Sm_i Sp_{i+1}>
               - 0.25 * Jz   # singlet <Sz_i Sz_{i+1}>
               + 1.*(0.25 * gamma**2 / (1. - gamma**3)))  # exponentially decaying "lonely" states
    # lonely: <Sz_{i} gamma gamma Sz_{i+2}
    exact_E = exact_E / psi.L  # energy per site
    for transpose in [False, True]:
        print(f"transpose={transpose!s}")
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        TM.matvec(TM.guess, project=False)
        TM.matvec(TM.guess, project=True)
        val, vec = TM.dominant_eigenvector()
        assert abs(val - 1.) < eps
        E0 = TM.energy(vec)
        print(E0, exact_E)
        assert abs(E0 - exact_E) < eps
        if not transpose:
            vec.itranspose(['vL', 'wL', 'vL*'])
            assert (vec[:, 4, :] - npc.eye_like(vec, 0)).norm() < eps
        else:
            vec.itranspose(['vR*', 'wR', 'vR'])
            assert (vec[:, 0, :] - npc.eye_like(vec, 0)).norm() < eps
    # get non-trivial psi
    psi.perturb({'N_steps': 10, 'trunc_params': {'chi_max': 3 , 'svd_min': 1.e-14}},
                close_1=False,
                canonicalize=False)
    psi.canonical_form_infinite2()
    # now find eigenvector again
    # and check TM |vec> = |vec> + val * |v0>
    # with |v0> = eye(chi) * IdL/IdR
    for transpose in [False, True]:
        print(f"transpose={transpose!s}")
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        val, vec = TM.dominant_eigenvector()
        E0 = TM.energy(vec)
        assert abs(val - 1.) < eps
        if not transpose:
            vec.itranspose(['vL', 'wL', 'vL*'])
            assert (vec[:, 4, :] - npc.eye_like(vec, 0)).norm() < eps
            v0 = npc.eye_like(vec, 0, labels=['vL', 'vL*']).add_leg(vec.get_leg('wL'), 0, 1, 'wL')
        else:
            vec.itranspose(['vR*', 'wR', 'vR'])
            assert (vec[:, 0, :] - npc.eye_like(vec, 0)).norm() < eps
            v0 = npc.eye_like(vec, 0, labels=['vR*', 'vR']).add_leg(vec.get_leg('wR'), 4, 1, 'wR')
        TM_vec = TM.matvec(vec, project=False)
        TM_vec.itranspose(vec.get_leg_labels())
        assert (TM_vec  - (vec + E0 * 3 * v0)).norm() < eps


def test_MPO_from_wavepacket(L=10):
    k0, x0, sigma, = np.pi/8., 2., 2.
    x = np.arange(L)
    coeff = np.exp(-1.j * k0 * x) * np.exp(- 0.5 * (x - x0)**2 / sigma**2)
    coeff /= np.linalg.norm(coeff)
    s = site.FermionSite(conserve='N')
    wp = mpo.MPO.from_wavepacket([s] * L, coeff, 'Cd')
    psi = mps.MPS.from_product_state([s] * L, ['empty'] * L)
    wp.apply(psi, dict(compression_method='SVD'))
    C = psi.correlation_function('Cd', 'C')
    C_expexcted = np.conj(coeff)[:, np.newaxis] * coeff[np.newaxis, :]
    assert np.max(np.abs(C - C_expexcted)) < 1.e-10
    print(C)
    print(C_expexcted)
