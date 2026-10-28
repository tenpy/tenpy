"""A collection of tests for (classes in) :module:`tenpy.networks.mpo`."""
# Copyright (C) TeNPy Developers, Apache license

import lasyten as lasy
import numpy as np
import pytest
from lasyten.models.couplings import Coupling
from lasyten.models.sites import SpinSite
from numpy import testing as npt
from scipy import linalg as spla

from tenpy.algorithms import ExactDiag
from tenpy.models import lattice, model
from tenpy.models.spins import SpinChain
from tenpy.networks import mpo
from tenpy.networks.terms import CouplingTerms, OnsiteTerms

# spin_half = site.SpinHalfSite(conserve='Sz', sort_charge=False)
spin_half = SpinSite(S=0.5, conserve=None)


def test_MPO():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half
    for bc in mpo.MPO._valid_bc:
        for L in [4, 2, 1]:
            print(bc, ', L =', L)
            grid = [[s.Id, s.Sp, s.Sm, s.Sz],
                    [None, None, None, s.Sm],
                    [None, None, None, s.Sp],
                    [None, None, None, s.Id]]  # fmt: skip
            legW = npc.LegCharge.from_qflat(s.leg.chinfo, [[0], s.Sp.qtotal, s.Sm.qtotal, [0]])
            W = npc.grid_outer(grid, [legW, legW.conj()], grid_labels=['wL', 'wR'])
            Ws = [W] * L
            if bc == 'finite':
                Ws[0] = Ws[0][0:1, :, :, :]
                Ws[-1] = Ws[-1][:, 3:4, :, :]
            H = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L), mps_unit_cell_width=L)
            H_copy = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L), mps_unit_cell_width=L)
            H.test_sanity()
            print(H.dim)
            print(H.chi)
            assert H.is_equal(H)  # everything should be equal to itself
            assert H.is_hermitian()
            H.test_sanity()
            assert H.is_equal(H_copy)
            assert H.prefactor(0, ['Sz']) == 1.0
            for i in range(L - 1):
                assert H.prefactor(i, ['Sp', 'Sm']) == 1.0
                assert H.prefactor(i, ['Sz', 'Sz']) == 0.0
        if L == 4:
            H2 = H.group_sites(n=2)
            H2.test_sanity()
            assert H2.L == 2


def test_MPOGraph_term_conversion():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L = 4

    g1 = mpo.MPOGraph([spin_half] * L, 'infinite', unit_cell_width=L)
    g1.test_sanity()
    for i in range(L):
        g1.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g1.add(i, 'IdL', ('left', i, 'Sp', 'Id'), 'Sp', 1.0)
        g1.add(i + 1, ('left', i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g1.add_missing_IdL_IdR()
    terms = [[('Sz', i)] for i in range(L)]
    terms += [[('Sp', i), ('Sm', i + 1)] for i in range(L)]
    prefactors = [0.5] * L + [1.5] * L
    term_list = TermList(terms, prefactors)
    g2 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite', unit_cell_width=L)
    g2.test_sanity()
    assert g1.graph == g2.graph
    terms[3:3] = [[('Sm', 2), ('Sp', 0), ('Sz', 1)]]
    prefactors[3:3] = [3.0]
    term_list = TermList(terms, prefactors)
    g3 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite', unit_cell_width=L)
    g4 = mpo.MPOGraph([spin_half] * L, 'infinite', unit_cell_width=L)
    g4.test_sanity()
    for i in range(L):
        g4.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g4.add(i, 'IdL', ('left', i, 'Sp', 'Id'), 'Sp', 1.0)
        g4.add(i + 1, ('left', i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g4.add_missing_IdL_IdR()
    g4.add(1, ('left', 0, 'Sp', 'Id'), ('right', 2, 'Sm', 'Id'), 'Sz', 3.0)
    g4.add(2, ('right', 2, 'Sm', 'Id'), 'IdR', 'Sm', 1.0)
    assert g4.graph == g3.graph


def test_MPO_conversion():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L = 8
    sites = []
    for i in range(L):
        s = site.Site(npc.LegCharge.from_trivial(2))
        s.add_op(f'X_{i:d}', np.array([[0.0, 1.0], [1.0, 0.0]]))
        s.add_op(f'Y_{i:d}', np.array([[0.0, 1.0], [-1.0, 0.0]]))
        s.add_op(f'Z_{i:d}', np.array([[1.0, 0.0], [0.0, -1.0]]))
        sites.append(s)

    terms = [
        [('X_0', 0)],
        [('X_0', 0), ('X_1', 1)],
        [('X_0', 0), ('X_3', 3)],
        [('X_4', 4), ('Y_5', 5), ('Y_7', 7)],
        [('X_4', 4), ('Y_5', 5), ('X_7', 7)],
        [('X_4', 4), ('Y_6', 6), ('Y_7', 7)],
    ]
    prefactors = [0.25, 10.0, 11.0, 101.0, 102.0, 103.0]
    term_list = TermList(terms, prefactors)
    g1 = mpo.MPOGraph.from_term_list(term_list, sites, bc='finite', insert_all_id=False, unit_cell_width=L)
    ct_add = MultiCouplingTerms(L)
    ct_add.add_coupling_term(12.0, 4, 5, 'X_4', 'X_5')
    ct_add.add_multi_coupling_term(0.5, [4, 5, 7], ['X_4', 'Y_5', 'X_7'], 'Id')
    prefactors[-2] += 0.5
    terms.append([('X_4', 4), ('X_5', 5)])
    prefactors.append(12.0)
    ct_add.add_to_graph(g1)
    H1 = g1.build_MPO()
    grids = [
        [
            ['Id', 'X_0', [('X_0', 0.25)]]  # site 0
        ],
        [
            ['Id', None, None],  # site 1
            [None, 'Id', [('X_1', 10.0)]],
            [None, None, 'Id'],
        ],
        [
            ['Id', None, None],  # site 2
            [None, [('Id', 11.0)], None],
            [None, None, 'Id'],
        ],
        [
            ['Id', None],  # site 3
            [None, 'X_3'],
            [None, 'Id'],
        ],
        [
            ['X_4', None],  # site 4
            [None, 'Id'],
        ],
        [
            ['Id', 'Y_5', [('X_5', 12.0)]],  # site 5
            [None, None, 'Id'],
        ],
        [
            # site 6
            [None, [('Y_6', 103.0)], None],
            [[('Id', 102.5)], [('Id', 101.0)], None],
            [None, None, 'Id'],
        ],
        [
            ['X_7'],  # site 7
            ['Y_7'],
            ['Id'],
        ],
    ]
    H2 = mpo.MPO.from_grids(
        sites, grids, 'finite', [0] * 4 + [None] * 5, [None] * 5 + [-1] * 4, mps_unit_cell_width=len(sites)
    )
    for w1, w2 in zip(H1._W, H2._W):
        assert npc.norm(w1 - w2, np.inf) == 0.0
    assert H2.is_equal(H1)
    back_to_terms = H1.to_TermList([['Id', f'X_{i:d}', f'Y_{i:d}', f'Z_{i:d}'] for i in range(L)], max_range=8)
    assert len(back_to_terms.terms) == len(terms)
    for term, pref in zip(back_to_terms.terms, back_to_terms.strength):
        assert term in terms
        i = terms.index(term)
        assert abs(prefactors[i] - pref) < 1.0e-13


def test_MPOEnvironment():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    xxz_pars = dict(L=4, Jxx=1.0, Jz=1.1, hz=0.1, bc_MPS='finite', sort_charge=True)
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([[0], [1]] * L)[:L]  # Neel
    psi = mps.MPS.from_lat_product_state(M.lattice, state)
    env = mpo.MPOEnvironment(psi, M.H_MPO, psi)
    env.get_LP(3, True)
    env.get_RP(0, True)
    env.test_sanity()
    E_exact = -0.825
    for i in range(4):
        E = env.full_contraction(i)  # should be one
        print('total energy for contraction at site ', i, ': E =', E)
        assert abs(E - E_exact) < 1.0e-14


def _hopping_coupling(site, op_i, op_j):
    """Build a two-site hopping Coupling ``op_i`` on site 0, ``op_j`` on site 1."""
    # dense block axes [p0, p1, p1*, p0*], see Coupling.from_dense_block / spin_spin_coupling
    h = np.tensordot(op_i, op_j, axes=0)  # [p0, p0*, p1, p1*]
    h = np.transpose(h, [0, 2, 3, 1])  # -> [p0, p1, p1*, p0*]
    return Coupling.from_dense_block(h, [site, site], understood_braiding=True)


@pytest.mark.parametrize('bc', ['infinite', 'finite'])
@pytest.mark.parametrize('explicit_plus_hc', [True, False])
def test_MPO_addition(bc, explicit_plus_hc):
    L = 4
    print('bc = ', bc, '-' * 40)
    s = spin_half

    def new_model():
        lat_bc = 'open' if bc == 'finite' else 'periodic'
        lat = lattice.Chain(L, s, bc=lat_bc, bc_MPS=bc)
        M = model.CouplingModel(lat)
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jmp=1, backend=s.backend, name='Sm.Sp'),
            hc='Sp.Sm',
            permute='swap_hc',
        )
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jx=1, backend=s.backend, name='Sx.Sx'),
            hc='Sx.Sx',
            permute='invariant',
        )
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jz=1, backend=s.backend, name='Sz.Sz'),
            hc='Sz.Sz',
            permute='invariant',
        )
        return M

    def graph_from_model(M):
        return mpo.MPOGraph.from_terms(M.get_nonzero_terms(), M, bc, unit_cell_width=M.lattice.mps_unit_cell_width)

    M1 = new_model()
    M2 = new_model()
    M12 = new_model()

    str_A = [0, 0, 2.0, 0] if bc == 'infinite' else [0, 0, 2.0]  # dx=1
    str_B = [0, 2.0, 0, 0] if bc == 'infinite' else [0, 2.0, 0]  # dx=1
    str_C = [0, 4.0, 7.0, 0] if bc == 'infinite' else [4, 7.0]  # dx=2
    for M in [M1, M12]:
        M.add_two_site_coupling(str_A, 'Sm.Sp', [1], x1='sum_over', u1=0, u2=0, plus_hc=True)
        M.add_two_site_coupling(str_B, 'Sz.Sz', [1], x1='sum_over', u1=0, u2=0)
        M.add_onsite_operator([0, 3.0, 0, 0], 'Sz', x='sum_over', u=0)
    cpl_slc = slice(0, 4) if bc == 'infinite' else slice(0, 3)
    for M in [M2, M12]:
        M.add_two_site_coupling(str_C, 'Sz.Sz', [2], x1='sum_over', u1=0, u2=0)
        M.add_onsite_operator([0, 5.0, 0, 0], 'Sz', x='sum_over', u=0)

    H1 = graph_from_model(M1).build_MPO()
    H2 = graph_from_model(M2).build_MPO()
    H12 = graph_from_model(M12).build_MPO()
    H12_sum = H1 + H2
    assert H12.is_equal(H12_sum)


@pytest.mark.parametrize('sites', [None, [0], [1], [0, 1, 2, 3], [2, 3], [3, 2]])
def test_MPO_plus_identity(sites, alpha=0.7, beta=0.42):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half

    ot = OnsiteTerms(4)
    ct = CouplingTerms(4)
    ct.add_coupling_term(2.0, 2, 3, 'Sm', 'Sp')
    ct.add_coupling_term(2.0, 2, 3, 'Sp', 'Sm')
    ct.add_coupling_term(2.0, 1, 2, 'Sz', 'Sz')
    ot.add_onsite_term(3.0, 1, 'Sz')
    H1 = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'finite', unit_cell_width=4).build_MPO()

    if sites is None:
        H2 = H1.plus_identity(alpha=alpha, beta=beta)
    else:
        H2 = H1.plus_identity(alpha=alpha, beta=beta, sites=sites)

    ot_expect = OnsiteTerms(4)
    ct_expect = CouplingTerms(4)
    ct_expect.add_coupling_term(beta * 2.0, 2, 3, 'Sm', 'Sp')
    ct_expect.add_coupling_term(beta * 2.0, 2, 3, 'Sp', 'Sm')
    ct_expect.add_coupling_term(beta * 2.0, 1, 2, 'Sz', 'Sz')
    ot_expect.add_onsite_term(beta * 3.0, 1, 'Sz')
    ot_expect.add_onsite_term(alpha, 1, 'Id')
    H2_expect = mpo.MPOGraph.from_terms((ot_expect, ct_expect), [s] * 4, 'finite', unit_cell_width=4).build_MPO()

    assert H2.is_equal(H2_expect)


def test_MPO_expectation_value(tol=1.0e-15):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L_mpo = 4
    s = spin_half
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='infinite', unit_cell_width=6)
    psi1.test_sanity()
    ot = OnsiteTerms(L_mpo)  # H.L != psi.L, consider L = lcm(4, 6) = 12
    ot.add_onsite_term(0.1, 0, 'Sz')  # -> 0.5 * 2 (sites 0, 4, not 8)
    ot.add_onsite_term(0.2, 3, 'Sz')  # -> 0.  (not sites 4, 7, 11)
    ct = CouplingTerms(L_mpo)  # note: ct.L != psi1.L
    ct.add_coupling_term(1.0, 2, 3, 'Sz', 'Sz')  # -> 0. (not 2-3, 6-7, 10-11)
    ct.add_coupling_term(1.5, 1, 3, 'Sz', 'Sz')  # -> 1.5*(-0.25) (1-3, not 5-7, not 9-11)
    ct.add_coupling_term(2.5, 0, 6, 'Sz', 'Sz')  # -> 2.5*0.25*3 (0-6, 4-10, not 8-14)
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * L_mpo, 'infinite', unit_cell_width=L_mpo).build_MPO()
    desired_ev = (0.1 * 0.5 * 2 + 0.2 * 0.0 + 1.0 * 0.0 + 1.5 * -0.25 + 2.5 * 0.25 * 2) / 12
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
    L_mpo = 1
    exp_dec_H = mpo.MPO.from_grids([s] * L_mpo, [grid] * L_mpo, bc='infinite', IdL=0, IdR=2, mps_unit_cell_width=L_mpo)
    desired_ev = (
        3 * 0.5 * 2  # Sz onsite
        + 0.25
        * (
            0.1**3
            + 0.1**5
            + 0.1**9
            + 0.1**11
            + 0.1**15  # Z_0 Z_i
            + 0.1**1
            + 0.1**5
            + 0.1**7
            + 0.1**11
            + 0.1**13
        )  # Z_4 Z_i
        + -0.25
        * (
            0.1**1  # Z_1 Z_3
            + 0.1**2
        )  # Z_2 Z_4
        + 0.0  # other sites
    ) / 6.0  # values > 1.e-15
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol
    L_mpo = 3  # should give exactly the same answer!
    exp_dec_H = mpo.MPO.from_grids([s] * L_mpo, [grid] * L_mpo, bc='infinite', IdL=0, IdR=2, mps_unit_cell_width=L_mpo)
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol


def test_MPO_var(L=8, tol=1.0e-13):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    xxz_pars = dict(L=L, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.1, bc_MPS='finite', conserve=None)
    M = SpinChain(xxz_pars)
    psi = random_MPS(L, 2, 10)
    exp_val = M.H_MPO.expectation_value(psi)

    ED = ExactDiag.from_model(M)
    ED.build_full_H_from_mpo()
    psi_full = ED.mps_to_full(psi)
    exp_val_full = npc.inner(psi_full, npc.tensordot(ED.full_H, psi_full, axes=1), axes='range', do_conj=True)
    assert abs(exp_val - exp_val_full) / abs(exp_val_full) < tol

    Hsquared = M.H_MPO.variance(psi, 0.0)

    Hsquared_full = npc.inner(
        psi_full,
        npc.tensordot(ED.full_H, npc.tensordot(ED.full_H, psi_full, axes=1), axes=1),
        axes='range',
        do_conj=True,
    )
    assert abs(Hsquared - Hsquared_full) / abs(Hsquared_full) < tol
    var = M.H_MPO.variance(psi)
    var_full = Hsquared_full - exp_val_full**2
    assert abs(var - var_full) / abs(var_full) < tol


@pytest.mark.parametrize('method', ['SVD', 'variational', 'zip_up'])
def test_apply_mpo(method):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    bc_MPS = 'finite'
    # NOTE: overlap doesn't work for calculating the energy (density) in infinite systems!
    # energy is extensive, overlap exponential....
    L = 5
    g = 0.5
    model_pars = dict(L=L, Jx=0.0, Jy=0.0, Jz=-4.0, hx=2.0 * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = [[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L  # pointing in (-x)-direction
    psi = mps.MPS.from_product_state(
        M.lattice.mps_sites(), state, bc=bc_MPS, unit_cell_width=M.lattice.mps_unit_cell_width
    )
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


def test_MPOTransferMatrix(eps=1.0e-13):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half
    # exponential decay in Sz term to make it harder
    gamma = 0.5
    Jxy, Jz = 4.0, 1.0
    hz = 0.0
    grid = [[s.Id, s.Sp, s.Sm, s.Sz, hz * s.Sz],
            [None, None, None, None, Jxy*0.5*s.Sm],
            [None, None, None, None, Jxy*0.5*s.Sp],
            [None, None, None, gamma*s.Id, Jz*s.Sz],
            [None, None, None, None, s.Id]]  # fmt: skip
    H = mpo.MPO.from_grids([s] * 3, [grid] * 3, 'infinite', 0, 4, max_range=np.inf, mps_unit_cell_width=3)
    psi = mps.MPS.from_singlets(s, 3, [(0, 1)], lonely=[2], bc='infinite', unit_cell_width=3)
    psi.roll_mps_unit_cell(-1)  # -> nontrivial chi at the cut between unit cells
    exact_E = (
        (-0.25 - 0.25) * Jxy  # singlet <0.5*Sp_i Sm_{i+1}> = 0.25 = <0.5*Sm_i Sp_{i+1}>
        - 0.25 * Jz  # singlet <Sz_i Sz_{i+1}>
        + 1.0 * (0.25 * gamma**2 / (1.0 - gamma**3))
    )  # exponentially decaying "lonely" states
    # lonely: <Sz_{i} gamma gamma Sz_{i+2}
    exact_E = exact_E / psi.L  # energy per site
    for transpose in [False, True]:
        print(f'transpose={transpose!s}')
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        TM.matvec(TM.guess, project=False)
        TM.matvec(TM.guess, project=True)
        val, vec = TM.dominant_eigenvector()
        assert abs(val - 1.0) < eps
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
    psi.perturb({'N_steps': 10, 'trunc_params': {'chi_max': 3, 'svd_min': 1.0e-14}}, close_1=False, canonicalize=False)
    psi.canonical_form_infinite2()
    # now find eigenvector again
    # and check TM |vec> = |vec> + val * |v0>
    # with |v0> = eye(chi) * IdL/IdR
    for transpose in [False, True]:
        print(f'transpose={transpose!s}')
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        val, vec = TM.dominant_eigenvector()
        E0 = TM.energy(vec)
        assert abs(val - 1.0) < eps
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
        assert (TM_vec - (vec + E0 * 3 * v0)).norm() < eps


def test_MPO_from_wavepacket(L=10):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    (
        k0,
        x0,
        sigma,
    ) = np.pi / 8.0, 2.0, 2.0
    x = np.arange(L)
    coeff = np.exp(-1.0j * k0 * x) * np.exp(-0.5 * (x - x0) ** 2 / sigma**2)
    coeff /= np.linalg.norm(coeff)
    s = site.FermionSite(conserve='N')
    wp = mpo.MPO.from_wavepacket([s] * L, coeff, 'Cd', unit_cell_width=L)
    psi = mps.MPS.from_product_state([s] * L, ['empty'] * L, unit_cell_width=L)
    wp.apply(psi, dict(compression_method='SVD'))
    C = psi.correlation_function('Cd', 'C')
    C_expexcted = np.conj(coeff)[:, np.newaxis] * coeff[np.newaxis, :]
    assert np.max(np.abs(C - C_expexcted)) < 1.0e-10
    print(C)
    print(C_expexcted)


# Will remove once this works in exact_diag
def _dense_Hamiltonian(site, L, terms):
    """Reference dense Hamiltonian built directly from `site`'s own operators.

    `terms` is a list of ``(strength, {position: opname})`` entries, e.g.
    ``(0.5, {0: 'Sz', 1: 'Sz'})`` for a two-site term.
    """
    ops = {name: site.get_op(name).to_numpy() for name in ('Sx', 'Sy', 'Sz')}
    Id = np.eye(site.dim)
    dim = site.dim**L
    H = np.zeros((dim, dim), dtype=complex)
    for strength, positions_ops in terms:
        factors = [Id] * L
        for pos, opname in positions_ops.items():
            factors[pos] = ops[opname]
        term = np.eye(1)
        for factor in factors:
            term = np.kron(term, factor)
        H += strength * term
    return H


# Will remove once this works in exact_diag
def _coupling_to_dense(coupling, L):
    """Contract a Coupling's factorization into a dense (dim**L, dim**L) matrix."""
    labels = [f'p{i}' for i in range(L)] + [f'p{i}*' for i in range(L)]
    dim = coupling.sites[0].dim
    return coupling.to_tensor().to_numpy(labels).reshape(dim**L, dim**L)


def make_WI_np(dt, A, B, C, D):
    assert all(T.ndim == 4 for T in [A, B, C, D])
    Id = np.eye(2)[None, None, :, :]
    W = np.zeros((1 + A.shape[0], 1 + A.shape[1], 2, 2), complex)
    W[0:1, 0:1] = Id + dt * D
    W[0:1, 1:] = C
    W[1:, 0:1] = dt * B
    W[1:, 1:] = A
    return W


def make_WII_np(dt, A, B, C, D):
    assert all(T.ndim == 4 for T in [A, B, C, D])
    B = B[:, 0]
    C = C[0, :]
    D = D[0, 0]

    # Implementation from v1:

    tC = np.sqrt(np.abs(dt))  # spread time step across B, C
    tB = dt / tC
    d = D.shape[0]

    # The virtual size of W is  (1+Nr, 1+Nc)
    Nr = A.shape[0]
    Nc = A.shape[1]
    W = np.zeros((1 + Nr, 1 + Nc, d, d), dtype=np.result_type(D, dt))

    Id_ = np.array([[1, 0], [0, 1]])  # 2x2 operators in a hard-core boson space
    b = np.array([[0, 0], [1, 0]])

    Id = np.kron(Id_, Id_)  # 4x4 operators in the 2x hard core boson space
    Br = np.kron(b, Id_)
    Bc = np.kron(Id_, b)
    Brc = np.kron(b, b)
    for r in range(Nr):  # double loop over row / column of A
        for c in range(Nc):
            # Select relevant part of virtual space and extend by hardcore bosons
            h = (
                np.kron(Brc, A[r, c, :, :])
                + np.kron(Br, tB * B[r, :, :])
                + np.kron(Bc, tC * C[c, :, :])
                + dt * np.kron(Id, D)
            )
            w = spla.expm(h)  # Exponentiate in the extended Hilbert space
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 1 + c, :, :] = w[1, 1]  # extracts relevant parts according to Eqn 11
            if c == 0:
                W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 1 + c] = w[0, 1]
                if c == 0:
                    W[0, 0] = w[0, 0]
        if Nc == 0:  # technically only need one boson
            h = np.kron(Br, tB * B[r, :, :]) + t * np.kron(Id, D)
            w = spla.expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 0] = w[0, 0]
    if Nr == 0:
        for c in range(Nc):
            h = np.kron(Bc, tC * C[c, :, :]) + t * np.kron(Id, D)
            w = spla.expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[0, 1 + c] = w[0, 1]
            if c == 0:
                W[0, 0] = w[0, 0]
        if Nc == 0:
            W = spla.expm(dt * D).reshape([1, 1, d, d])
    return W


def make_W_np_v1(dt, H_W: list[np.ndarray], approximation: str, finite: bool) -> list[np.ndarray]:
    """make WI or WII from W of H_MPO, like in v1"""

    # ensure IdL / IdR as expected
    eye = np.eye(2)

    if finite:
        assert H_W[0].shape[0] == 1
        assert H_W[-1].shape[1] == 1

    W_U = []
    for i, W in enumerate(H_W):  # W[wL, wR, p, p*]
        if finite and i == 0:
            # W = [[1 C D]]
            assert np.allclose(W[0, 0], eye)
            C = W[0, 1:-1][None, :, :, :]
            D = W[0, -1][None, None, :, :]
            A = np.zeros((0, C.shape[1], 2, 2))
            B = np.zeros((0, D.shape[1], 2, 2))
        elif finite and i == len(H_W) - 1:
            # W =[[D] [B] [1]]
            D = W[0, -1][None, None, :, :]
            B = W[1:-1, -1][:, None, :, :]
            assert np.allclose(W[-1, -1], eye)
            A = np.zeros((B.shape[0], 0, 2, 2))
            C = np.zeros((D.shape[0], 0, 2, 2))
        else:
            # extract blocks such that
            # W = [[1 C D]
            #      [0 A B]
            #      [0 0 1]]
            assert np.allclose(W[0, 0], eye)
            C = W[0, 1:-1][None, :, :, :]
            D = W[0, -1][None, None, :, :]
            assert np.allclose(W[1:-1, 0], 0)
            A = W[1:-1, 1:-1]
            B = W[1:-1, -1][:, None, :, :]
            assert np.allclose(W[-1, :-1], 0)
            assert np.allclose(W[-1, -1], eye)

        if approximation == 'I':
            W_U.append(make_WI_np(dt, A, B, C, D))
        elif approximation == 'II':
            W_U.append(make_WII_np(dt, A, B, C, D))
        else:
            raise NotImplementedError
    return W_U


def _roll_IdR_position(W, wL=True, wR=True):
    """Map between conventions [IdL, IdR, *other] -> [IdL, *others, IdR] for order of W-blocks"""
    dL, dR, _, _ = W.shape
    # [IdL, IdR, *other] -> [IdL, *others, IdR]
    if wL and wR:
        res = W[*np.ix_([0, *range(2, dL), 1], [0, *range(2, dR), 1]), :, :]
    elif wL:
        res = W[[0, *range(2, dL), 1], :, :, :]
    elif wR:
        res = W[:, [0, *range(2, dR), 1], :, :]
    else:
        res = W
    assert res.shape == W.shape
    return res


def make_W_np(dt, H: mpo.MPO, approximation: str, finite: bool) -> np.ndarray:
    """Make pure-numpy WI / WII tensors from the Hamiltonian MPO"""
    # transpose to v1 leg order
    assert all(W.labels == ['wL', 'p', 'wR', 'p*'] for W in H._W)
    H_W = [W.to_numpy().transpose([0, 2, 1, 3]) for W in H._W]  # transpose to tenpy v1 leg order

    # ensure row/column ordering within MPO tensors
    has_v1_state_ordering = True
    has_lexsorted_state_ordering = True
    for i, W in enumerate(H._W):
        labelsL = W.get_leg('wL').summand_labels
        if i == 0 and finite:
            has_v1_state_ordering &= labelsL == ['IdL']
            has_lexsorted_state_ordering &= labelsL == ['IdL']
        else:
            has_v1_state_ordering &= labelsL[0] == 'IdL' and labelsL[-1] == 'IdR'
            has_lexsorted_state_ordering &= labelsL[0] == 'IdL' and labelsL[1] == 'IdR'

        labelsR = W.get_leg('wR').summand_labels
        if i == H.L - 1 and finite:
            has_v1_state_ordering &= labelsR == ['IdR']
            has_lexsorted_state_ordering &= labelsR == ['IdR']
        else:
            has_v1_state_ordering &= labelsR[0] == 'IdL' and labelsR[-1] == 'IdR'
            has_lexsorted_state_ordering &= labelsR[0] == 'IdL' and labelsR[1] == 'IdR'

    if has_v1_state_ordering:
        pass
    elif has_lexsorted_state_ordering:
        H_W = [_roll_IdR_position(W, wL=not finite or i > 0, wR=not finite or i < H.L - 1) for i, W in enumerate(H_W)]
    else:
        raise NotImplementedError

    op_norms = [np.linalg.norm(W, axis=(2, 3)) for W in H_W]  # for debugging
    U_W = make_W_np_v1(dt, H_W, approximation=approximation, finite=finite)

    # transpose back to v2 leg order
    U_W = [W.transpose([0, 2, 1, 3]) for W in U_W]

    return U_W


@pytest.mark.parametrize('conserve', [None, 'best'])
@pytest.mark.parametrize('backend', ['fusion_tree', 'abelian', 'no_symmetry'])
@pytest.mark.parametrize('approximation', ['I', 'II'])
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_MPO_make_U(conserve, backend, approximation, bc_MPS, L=6, dt=0.01):
    """Compare MPO.make_U vs a pure-numpy implementation

    This pure-numpy implementation is also run in the v1 tests, and thus trusted
    """
    M = SpinChain(dict(L=L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.2, bc_MPS=bc_MPS, conserve=conserve))
    if M.H_MPO._W[0].backend != lasy.get_backend(backend):
        pytest.skip('Can not control model backend yet')

    if approximation == 'II':
        with pytest.raises(NotImplementedError):
            _ = M.H_MPO.make_U(-1j * dt, approximation=approximation)
        pytest.skip('WII not ported yet')

    U_MPO: mpo.MPO = M.H_MPO.make_U(-1j * dt, approximation=approximation)
    U_MPO.test_sanity()
    assert U_MPO.L == L

    # compare against pure-numpy construction of the W tensors
    W_U_expect_np = make_W_np(-1j * dt, M.H_MPO, approximation=approximation, finite=bc_MPS == 'finite')
    for W, expect_np in zip(U_MPO._W, W_U_expect_np):
        W_np = W.to_numpy()
        npt.assert_almost_equal(W_np, expect_np)

    # check error bound
    if bc_MPS == 'finite':
        ED_H = ExactDiag.from_H_mpo(M.H_MPO)
        ED_H.build_full_H_from_mpo()
        ED_H.full_diagonalization()
        U_exact = ED_H.exp_H(dt).to_numpy()

        ED_U = ExactDiag.from_H_mpo(U_MPO)
        ED_U.build_full_H_from_mpo()
        U_np = ED_U.full_H.to_numpy()

        err = np.linalg.norm(U_np - U_exact)
        assert err < L * (dt**2)
