"""A collection of tests for (classes in) :mod:`tenpy.models.model`."""
# Copyright (C) TeNPy Developers, Apache license

import itertools

import numpy as np
import numpy.testing as npt
import pytest

import tenpy.linalg.np_conserved as npc
import tenpy.networks.site
from tenpy.algorithms.exact_diag import ExactDiag, get_numpy_Hamiltonian
from tenpy.models import lattice, model
from tenpy.models.spins import DipolarSpinChain
from tenpy.models.xxz_chain import XXZChain

spin_half_site = tenpy.networks.site.SpinHalfSite('Sz', sort_charge=False)

fermion_site = tenpy.networks.site.FermionSite('N')

__all__ = ['check_model_sanity', 'check_general_model']


def check_model_sanity(M, hermitian=True):
    """call M.test_sanity() for all different subclasses of M."""
    if isinstance(M, model.CouplingModel):
        model.CouplingModel.test_sanity(M)
    if isinstance(M, model.NearestNeighborModel):
        model.NearestNeighborModel.test_sanity(M)
        if hermitian:
            for i, H in enumerate(M.H_bond):
                if H is not None:
                    err = npc.norm(H - H.conj().transpose(H.get_leg_labels()))
                    if err > 1.0e-14:
                        print(H)
                        raise ValueError(f'H on bond {i:d} not hermitian')
    if isinstance(M, model.MPOModel):
        model.MPOModel.test_sanity(M)
        if hermitian:
            assert M.H_MPO.is_hermitian()


def check_general_model(ModelClass, model_pars={}, check_pars={}, hermitian=True):
    """Create a model for different sets of parameters and check its sanity.

    Parameters
    ----------
    ModelClass :
        We generate models of this class
    model_pars : dict
        Model parameters used.
    check_pars : dict
        pairs (`key`, `list of values`); we update ``model_pars[key]`` with any values of
        ``check_pars[key]`` (in each possible combination!) and create a model for it.
    hermitian : bool
        If True, check that the Hamiltonian is Hermitian.
    """
    if len(check_pars) == 0:
        M = ModelClass(model_pars.copy())
        check_model_sanity(M, hermitian)
        return
    for vals in itertools.product(*list(check_pars.values())):
        print('-' * 40)
        params = model_pars.copy()
        for k, v in zip(list(check_pars.keys()), vals):
            params[k] = v
        print('check_model_sanity with following parameters:')
        print(params)
        M = ModelClass(params)
        check_model_sanity(M, hermitian)


def test_CouplingModel():
    for bc in ['open', 'periodic']:
        spin_half_lat = lattice.Chain(5, spin_half_site, bc=bc, bc_MPS='finite')
        M = model.CouplingModel(spin_half_lat)
        M.add_coupling(1.2, 0, 'Sz', 0, 'Sz', 1)
        M.test_sanity()
        M.calc_H_MPO()
        if bc == 'periodic':
            with pytest.raises(ValueError, match='initialize H_bond for a NearestNeighborModel'):
                M.calc_H_bond()  # should raise a ValueError
                # periodic bc but finite bc_MPS leads to a long-range coupling
        else:
            M.calc_H_bond()


def test_ext_flux():
    Lx, Ly = 3, 4
    lat = lattice.Square(Lx, Ly, fermion_site, bc=['periodic', 'periodic'], bc_MPS='infinite')
    M = model.CouplingModel(lat)
    strength = 1.23
    strength_array = np.ones((Lx, Ly)) * strength
    for phi in [0, 2 * np.pi]:  # flux shouldn't do anything
        print('phi = ', phi)
        for dx in [1, 0], [0, 1], [0, 2], [1, -1], [-2, 2]:
            print('dx = ', dx)
            strength_flux = M.coupling_strength_add_ext_flux(strength, [1, 0], [0, phi])
            npt.assert_array_almost_equal_nulp(strength_flux, strength_array, 10)
    for phi in [np.pi / 2, 0.123]:
        print('phi = ', phi)
        strength_hop_x = M.coupling_strength_add_ext_flux(strength, [1, 0], [0, phi])
        npt.assert_array_almost_equal_nulp(strength_hop_x, strength_array, 10)
        expect_y_1 = np.array(strength_array, dtype=np.complex128)
        expect_y_1[:, -1:] = strength * np.exp(1.0j * phi)
        for dx in [[0, 1], [0, -1], [1, -1], [1, 1]]:
            print('dx = ', dx)
            strength_hop_y_1 = M.coupling_strength_add_ext_flux(strength, dx, [0, phi])
            if dx[1] < 0:
                npt.assert_array_almost_equal_nulp(strength_hop_y_1, expect_y_1, 10)
            else:
                npt.assert_array_almost_equal_nulp(strength_hop_y_1, np.conj(expect_y_1), 10)
        expect_y_2 = np.array(strength_array, dtype=np.complex128)
        expect_y_2[:, -2:] = strength * np.exp(1.0j * phi)
        for dx in [[0, 2], [0, -2], [1, 2], [3, 2]]:
            print('dx = ', dx)
            strength_hop_y_2 = M.coupling_strength_add_ext_flux(strength, dx, [0, phi])
            if dx[1] < 0:
                npt.assert_array_almost_equal_nulp(strength_hop_y_2, expect_y_2, 10)
            else:
                npt.assert_array_almost_equal_nulp(strength_hop_y_2, np.conj(expect_y_2), 10)


def test_CouplingModel_shift(Lx=3, Ly=3, shift=1):
    bc = ['periodic', shift]
    spin_half_square = lattice.Square(Lx, Ly, spin_half_site, bc=bc, bc_MPS='infinite')
    M = model.CouplingModel(spin_half_square)
    M.add_coupling(1.2, 0, 'Sz', 0, 'Sz', [1, 0])
    M.add_multi_coupling(0.8, [('Sz', [0, 0], 0), ('Sz', [0, 1], 0), ('Sz', [1, 0], 0)])
    M.test_sanity()
    H = M.calc_H_MPO()
    dims = [W.shape[0] for W in H._W]
    # check translation invariance of the MPO: at least the dimensions should fit
    # (the states are differently ordered, so the matrices differ!)
    for i in range(1, Lx):
        assert dims[:Ly] == dims[i * Ly : (i + 1) * Ly]


def test_CouplingModel_fermions():
    for bc, bc_MPS in zip(['open', 'periodic'], ['finite', 'infinite']):
        fermion_lat = lattice.Chain(5, fermion_site, bc=bc, bc_MPS=bc_MPS)
        M = model.CouplingModel(fermion_lat)
        M.add_coupling(1.2, 0, 'Cd', 0, 'C', 1, 'JW')
        M.add_coupling(1.2, 0, 'Cd', 0, 'C', -1, 'JW')
        M.test_sanity()
        M.calc_H_MPO()
        M.calc_H_bond()


def test_CouplingModel_explicit():
    fermion_lat_cyl = lattice.Square(1, 2, fermion_site, bc='periodic', bc_MPS='infinite')
    M = model.CouplingModel(fermion_lat_cyl)
    M.add_onsite(0.125, 0, 'N')
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, 1), None)  # auto-determine JW-string!
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, -1), None)
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (1, 0), None)
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (-1, 0), None)
    M.add_coupling(4.0, 0, 'N', 0, 'N', (-2, -1), None)  # a full unit cell inbetween!
    H_mpo = M.calc_H_MPO()
    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)
    Id, JW, N = fermion_site.Id, fermion_site.JW, fermion_site.N
    Cd, C = fermion_site.Cd, fermion_site.C
    CdJW = Cd.matvec(JW)  # = Cd
    JWC = JW.matvec(C)  # = C
    #  H_MPO_graph = tenpy.networks.mpo.MPOGraph.from_terms((M.all_onsite_terms(),
    #                                                        M.all_coupling_terms(),
    #                                                        M.exp_decaying_terms),
    #                                                       M.lat.mps_sites(),
    #                                                       M.lat.bc_MPS,
    #                                                       unit_cell_width=M.lat.mps_unit_cell_width)
    #  H_MPO_graph._set_ordered_states()
    #  from pprint import pprint
    #  pprint(H_MPO_graph._ordered_states)
    #  print(M.all_coupling_terms().to_TermList())
    #  [{'IdL': 0,
    #    ('left', 0, 'Cd JW', 'JW'): 1,
    #    ('left', 0, 'JW C', 'JW'): 2,
    #    ('left', 0, 'N', 'Id'): 3,
    #    ('left', 1, 'Cd JW', 'JW'): 4,
    #    ('left', 1, 'JW C', 'JW'): 5,
    #    ('left', 1, 'N', 'Id'): 6,
    #    ('left', 0, 'N', 'Id', 2, 'Id', 'Id'): 7,
    #    ('left', 1, 'N', 'Id', 3, 'Id', 'Id'): 8},
    #    'IdR': 9,
    #   {'IdL': 0,
    #    ('left', 0, 'Cd JW', 'JW'): 1,
    #    ('left', 0, 'JW C', 'JW'): 2,
    #    ('left', 0, 'N', 'Id'): 3,
    #    ('left', 1, 'Cd JW', 'JW'): 4,
    #    ('left', 1, 'JW C', 'JW'): 5,
    #    ('left', 1, 'N', 'Id'): 6},
    #    ('left', 0, 'N', 'Id', 2, 'Id', 'Id'): 7,
    #    ('left', 0, 'N', 'Id', 2, 'Id', 'Id', 4, 'Id', 'Id'): 8,
    #    'IdR': 9,
    #  0.50000 * Cd JW_0 C_1 +
    #  1.50000 * Cd JW_0 C_2 +
    #  0.50000 * JW C_0 Cd_1 +
    #  1.50000 * JW C_0 Cd_2 +
    #  4.00000 * N_0 N_5 +
    #  1.50000 * Cd JW_1 C_3 +
    #  1.50000 * JW C_1 Cd_3 +
    #  4.00000 * N_1 N_4

    # fmt: off
    W0_ex = [[Id,   CdJW, JWC,  N,    None, None, None, None, None, N*0.125],
             [None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, None, None, None, None, None, None, Id,   None, None],
             [None, None, None, None, JW,   None, None, None, None, None],
             [None, None, None, None, None, JW,   None, None, None, None],
             [None, None, None, None, None, None, Id,   None, None, None],
             [None, None, None, None, None, None, None, None, Id,   None],
             [None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, None, None, None, None, None, None, Id]]
    W1_ex = [[Id,   None, None, None, CdJW, JWC,  N,    None, None, N*0.125],
             [None, JW,   None, None, None, None, None, None, None, C*0.5],
             [None, None, JW,   None, None, None, None, None, None, Cd*0.5],
             [None, None, None, Id,   None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, None, None, None, None, None, None, None, Id,   None],
             [None, None, None, None, None, None, None, Id,   None, None],
             [None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, None, None, None, None, None, None, Id]]
    # fmt: on

    W0_ex = npc.grid_outer(W0_ex, W0_new.legs[:2])
    W1_ex = npc.grid_outer(W1_ex, W1_new.legs[:2])
    assert npc.norm(W0_new - W0_ex) ** 2 == 0.0  # coupling constants: no rounding errors
    assert npc.norm(W1_new - W1_ex) ** 2 == 0.0  # coupling constants: no rounding errors


@pytest.mark.parametrize('use_plus_hc, JW', [(False, 'JW'), (False, None), (True, None)])
def test_CouplingModel_multi_couplings_explicit(use_plus_hc, JW):
    fermion_lat_cyl = lattice.Square(1, 2, fermion_site, bc='periodic', bc_MPS='infinite')
    M = model.CouplingModel(fermion_lat_cyl)
    # create a weird fermionic model with 3-body interactions
    M.add_onsite(0.125, 0, 'N')
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, 1), plus_hc=use_plus_hc)
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (1, 0), JW, plus_hc=use_plus_hc)
    if not use_plus_hc:
        M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, -1), JW)
        M.add_coupling(1.5, 0, 'Cd', 0, 'C', (-1, 0), JW)
    # multi_coupling with a full unit cell inbetween the operators!
    M.add_multi_coupling(4.0, [('N', (0, 0), 0), ('N', (-2, -1), 0)])
    # some weird mediated hopping along the diagonal
    M.add_multi_coupling(1.125, [('N', (0, 0), 0), ('Cd', (0, 1), 0), ('C', (1, 0), 0)])
    H_mpo = M.calc_H_MPO()
    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)
    Id, JW, N = fermion_site.Id, fermion_site.JW, fermion_site.N
    Cd, C = fermion_site.Cd, fermion_site.C
    CdJW = Cd.matvec(JW)  # = Cd
    JWC = JW.matvec(C)  # = C
    NJW = N.matvec(JW)
    # fmt: off
    H_MPO_graph = tenpy.networks.mpo.MPOGraph.from_terms((M.all_onsite_terms(),
                                                          M.all_coupling_terms(),
                                                          M.exp_decaying_terms),
                                                         M.lat.mps_sites(),
                                                         M.lat.bc_MPS,
                                                         unit_cell_width=M.lat.mps_unit_cell_width)
    H_MPO_graph._set_ordered_states()
    from pprint import pprint
    pprint(H_MPO_graph._ordered_states)
    pprint(H_MPO_graph._build_grids())
    print(M.all_coupling_terms().to_TermList())
    #  0.50000 * Cd JW_0 C_1 +
    #  1.50000 * Cd JW_0 C_2 +
    #  0.50000 * JW C_0 Cd_1 +
    #  1.50000 * JW C_0 Cd_2 +
    #  1.50000 * Cd JW_1 C_3 +
    #  1.50000 * JW C_1 Cd_3 +
    #  4.00000 * N_0 N_5 +
    #  4.00000 * N_1 N_4 +
    #  1.12500 * N_0 Cd JW_1 C_2 +
    #  1.12500 * Cd JW_0 N JW_1 C_3

    W0_ex = [[Id,   CdJW, JWC,  N,    None, None, None, None, None, N*0.125],
             [None, None, None, None, None, Id,   None, None, None, None],
             [None, None, None, None, None, None, JW*1.5,   None, None, None],
             [None, None, None, None, None, None, None, JW*1.5,   None, None],
             [None, None, None, None, Id,   None, None, None, None, None],
             [None, None, None, None, None, None, JW*1.125, None, None, None],
             [None, None, None, None, None, None, None, None, None, C],
             [None, None, None, None, None, None, None, None, None, Cd],
             [None, None, None, None, None, None, None, None, None, N],
             [None, None, None, None, None, None, None, None, Id,   None],
             [None, None, None, None, None, None, None, None, None, Id]]

    W1_ex = [[Id,   None, CdJW, JWC,  N,    None, None, None, None, None, N*0.125],
             [None, None, None, None, None, NJW,  JW*1.5, None, None, None, C*0.5],
             [None, None, None, None, None, None, None, JW*1.5, None, None, Cd*0.5],
             [None, Id,   None, None, None, None, CdJW*1.125, None, None, None, None],
             [None, None, None, None, None, None, None, None, Id*4., None, None],
             [None, None, None, None, None, None, None, None, None, Id*4., None],
             [None, None, None, None, None, None, None, None, None, None, C],
             [None, None, None, None, None, None, None, None, None, None, Cd],
             [None, None, None, None, None, None, None, None, None, None, N],
             [None, None, None, None, None, None, None, None, None, None, Id]]
    # fmt: on
    W0_ex = npc.grid_outer(W0_ex, W0_new.legs[:2])
    assert npc.norm(W0_new - W0_ex) == 0.0  # coupling constants: no rounding errors
    W1_ex = npc.grid_outer(W1_ex, W1_new.legs[:2])

    assert npc.norm(W1_new - W1_ex) == 0.0  # coupling constants: no rounding errors


@pytest.mark.parametrize('use_fermions', [False, True], ids=['spins', 'fermions'])
@pytest.mark.parametrize('add_hc', [False, 'manually', 'flag'], ids=['no_hc', 'manual_hc', 'plus_hc'])
@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_CouplingModel_exponentially_decaying_coupling(use_fermions, add_hc, bc, L=6):
    """test add_exponentially_decaying coupling by comparing with manual couplings"""
    if use_fermions:
        s = tenpy.networks.site.FermionSite(None)
        op_i = 'Cd'
        op_j = 'C'
        anti_commute_sign = -1
    else:
        s = tenpy.networks.site.SpinHalfSite(None)
        op_i = 'Sp'
        op_j = 'Sm'
        anti_commute_sign = +1

    lat = lattice.Chain(L=L, site=s, bc=['open' if bc == 'finite' else 'periodic'], bc_MPS=bc)
    m_exp = model.CouplingModel(lat)
    m_manual = model.CouplingModel(lat)

    print('standard case: no subsites')
    a = 3.0 + 0.42j
    l = 0.2
    m_exp.add_exponentially_decaying_coupling(a, l, op_i, op_j, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        # interface does not allow us to change the order of operators, so we need to manually
        # include the sign for anti-commuting them
        m_exp.add_exponentially_decaying_coupling(
            anti_commute_sign * np.conj(a), np.conj(l), s.hc_ops[op_i], s.hc_ops[op_j], plus_hc=(add_hc == 'flag')
        )
    max_range = L if bc == 'finite' else int(np.ceil(np.log(1e-10) / np.log(l)))
    for k in range(1, max_range):
        m_manual.add_coupling(a * (l**k), 0, op_i, 0, op_j, dx=k, plus_hc=add_hc is not False)
    assert m_exp.calc_H_MPO().is_equal(m_manual.calc_H_MPO())

    print('non-uniform decay')
    m_exp = model.CouplingModel(lat)
    m_manual = model.CouplingModel(lat)
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    m_exp.add_exponentially_decaying_coupling(a, l, op_i, op_j, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        # interface does not allow us to change the order of operators, so we need to manually
        # include the sign for anti-commuting them
        m_exp.add_exponentially_decaying_coupling(
            anti_commute_sign * np.conj(a), np.conj(l), s.hc_ops[op_i], s.hc_ops[op_j]
        )
    # use the max_range from before. since l <= .2, this can only decay faster
    for k in range(1, max_range):
        strength = a
        for j in range(k):
            if bc == 'finite':
                strength = strength * l[j : -k + j]
            else:
                strength = strength * np.roll(l, -j)
        if np.all(strength < 1e-10):
            continue
        m_manual.add_coupling(strength, 0, op_i, 0, op_j, dx=k, plus_hc=add_hc is not False)
    assert m_exp.calc_H_MPO().is_equal(m_manual.calc_H_MPO())

    print('with subsites')
    m_exp = model.CouplingModel(lat)
    m_manual = model.CouplingModel(lat)
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    subsites = [1, 3, 5]
    m_exp.add_exponentially_decaying_coupling(a, l, op_i, op_j, subsites, plus_hc=(add_hc == 'flag'))
    if add_hc == 'manually':
        # interface does not allow us to change the order of operators, so we need to manually
        # include the sign for anti-commuting them
        m_exp.add_exponentially_decaying_coupling(
            anti_commute_sign * np.conj(a), np.conj(l), s.hc_ops[op_i], s.hc_ops[op_j], subsites
        )
    # use the max_range from before. since l <= .2, this can only decay faster
    # except now we interpret it as a distance *within* subsites
    for n_i, i in enumerate(subsites):
        for k in range(1, max_range):
            n_j = n_i + k
            if bc == 'finite' and n_j >= len(subsites):
                continue
            j = subsites[n_j % len(subsites)] + L * (n_j // len(subsites))
            strength = a
            for m in range(n_i, n_j):
                strength *= l[subsites[m % len(subsites)]]
            m_manual.add_coupling_term(
                strength, i, j, op_i, op_j, plus_hc=add_hc is not False, op_string='JW' if use_fermions else 'Id'
            )
    assert m_exp.calc_H_MPO().is_equal(m_manual.calc_H_MPO())

    print('with subsites and subsites_start')
    m_exp = model.CouplingModel(lat)
    m_manual = model.CouplingModel(lat)
    a = 3.0 + 0.42j
    l = np.random.uniform(0.01, 0.2, size=L)
    subsites_start = [0, 2]
    subsites = [1, 3, 5]
    m_exp.add_exponentially_decaying_coupling(
        a, l, op_i, op_j, subsites, subsites_start=subsites_start, plus_hc=(add_hc == 'flag')
    )
    if add_hc == 'manually':
        # interface does not allow us to change the order of operators, so we need to manually
        # include the sign for anti-commuting them
        m_exp.add_exponentially_decaying_coupling(
            anti_commute_sign * np.conj(a),
            np.conj(l),
            s.hc_ops[op_i],
            s.hc_ops[op_j],
            subsites,
            subsites_start=subsites_start,
        )
    # use the max_range from before. since l <= .2, this can only decay faster
    # except now we interpret it as a distance *within* subsites
    for n_i, i in enumerate(subsites_start):
        min_n_j = min(n_j for n_j, j in enumerate(subsites) if j > i)
        for n_j in range(min_n_j, min_n_j + max_range):
            if bc == 'finite' and n_j >= len(subsites):
                continue
            j = subsites[n_j % len(subsites)] + L * (n_j // len(subsites))
            strength = a * l[i]
            for m in range(min_n_j, n_j):
                strength *= l[subsites[m % len(subsites)]]
            m_manual.add_coupling_term(
                strength, i, j, op_i, op_j, plus_hc=add_hc is not False, op_string='JW' if use_fermions else 'Id'
            )
    assert m_exp.calc_H_MPO().is_equal(m_manual.calc_H_MPO())


class MyMod(model.CouplingMPOModel, model.NearestNeighborModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        return tenpy.networks.site.SpinHalfSite(conserve, True)

    def init_terms(self, model_params):
        x = model_params.get('x', 1.0)
        y = model_params.get('y', 0.25)
        self.add_onsite_term(y, 0, 'Sz')
        self.add_local_term(y, [('Sz', [4, 0])])
        self.add_coupling_term(x, 0, 1, 'Sx', 'Sx')
        self.add_coupling_term(2.0 * x, 1, 2, 'Sy', 'Sy')
        self.add_local_term(3.0 * x, [('Sy', [3, 0]), ('Sy', [4, 0])])


def test_CouplingMPOModel_group():
    m1 = MyMod(dict(x=0.5, L=5, bc_MPS='finite'))
    model_params = {'L': 6, 'hz': np.random.random([6]), 'bc_MPS': 'finite', 'sort_charge': True}
    m2 = XXZChain(model_params)
    for m in [m1, m2]:
        print('model = ', m)
        assert m.H_MPO.max_range == 1
        # test grouping sites
        ED = ExactDiag(m)
        #  ED.build_full_H_from_mpo()
        ED.build_full_H_from_bonds()
        m.group_sites(n=2)
        assert m.H_MPO.max_range == 1
        ED_gr = ExactDiag(m)
        ED_gr.build_full_H_from_mpo()
        H = ED.full_H.split_legs().to_ndarray()
        Hgr = ED_gr.full_H.split_legs()
        Hgr.idrop_labels()
        Hgr = Hgr.split_legs().to_ndarray()
        assert np.linalg.norm(H - Hgr) < 1.0e-14
        ED_gr.full_H = None
        ED_gr.build_full_H_from_bonds()
        Hgr = ED_gr.full_H.split_legs()
        Hgr.idrop_labels()
        Hgr = Hgr.split_legs().to_ndarray()
        assert np.linalg.norm(H - Hgr) < 1.0e-14


def test_model_H_conversion(L=6):
    bc = 'finite'
    model_params = {'L': L, 'hz': np.random.random([L]), 'bc_MPS': bc, 'sort_charge': True}
    m = XXZChain(model_params)

    # can we run the conversion?
    # conversion from bond to MPO in NearestNeighborModel
    H_MPO = m.calc_H_MPO_from_bond()
    # conversion from MPO to bond in MPOModel
    H_bond = m.calc_H_bond_from_MPO()
    # compare: did we get the correct result?
    ED = ExactDiag(m)
    ED.build_full_H_from_bonds()
    H0 = ED.full_H  # this should be correct
    ED.full_H = None
    m.H_MPO = H_MPO
    ED.build_full_H_from_mpo()
    full_H_mpo = ED.full_H  # the one generated by NearstNeighborModel.calc_H_MPO_from_bond()
    print('npc.norm(H0 - full_H_mpo) = ', npc.norm(H0 - full_H_mpo))
    assert npc.norm(H0 - full_H_mpo) < 1.0e-14  # round off errors on order of 1.e-15
    m.H_bond = H_bond
    ED.full_H = None
    ED.build_full_H_from_bonds()
    full_H_bond = ED.full_H  # the one generated by NearstNeighborModel.calc_H_MPO_from_bond()
    print('npc.norm(H0 - full_H_bond) = ', npc.norm(H0 - full_H_bond))
    assert npc.norm(H0 - full_H_bond) < 1.0e-14  # round off errors on order of 1.e-15


def test_model_H_conversion_dipolar(L=6):
    model_params = dict(L=L, S=1, J3=1.0, J4=0.5, bc_MPS='finite', sort_charge=True)

    # build full hamiltonian from MPO, assume that to be correct
    m = DipolarSpinChain(model_params)
    m.group_sites(3)
    ED = ExactDiag(m)
    ED.build_full_H_from_mpo()
    H0 = ED.full_H
    H0.test_sanity()
    assert np.all(H0.qtotal == 0)

    # convert H_MPO -> H_bond (calc_H_bond_from_MPO called by from_MPOModel)
    m_nn = model.NearestNeighborModel.from_MPOModel(m)
    ED = ExactDiag(m_nn)
    ED.build_full_H_from_bonds()
    H1 = ED.full_H
    H1.test_sanity()
    assert np.all(H1.qtotal == 0)

    # convert H_bond -> H_MPO
    m_nn.H_MPO = m_nn.calc_H_MPO_from_bond()
    ED.full_H = None
    ED.build_full_H_from_mpo()
    H2 = ED.full_H
    H2.test_sanity()
    assert np.all(H2.qtotal == 0)

    assert npc.norm(H0 - H1) < 1e-13
    assert npc.norm(H0 - H2) < 1e-13


def compare_models_plus_hc(
    m_manual: model.CouplingModel,
    m_plus_hc: model.CouplingModel,
    m_explicit: model.CouplingModel,
    expect_non_hermitian_mpo: bool = False,  # if MPO without hc is non-hermitian
):
    # helper for test_model_plus_hc; check if the models are equivalent
    for m in [m_manual, m_plus_hc, m_explicit]:
        m.H_MPO = m.calc_H_MPO()

    assert m_manual.H_MPO.is_hermitian(), 'm_manual not hermitian'
    assert m_plus_hc.H_MPO.is_hermitian(), 'm_plus_hc not hermitian'
    assert m_explicit.H_MPO.is_hermitian(), 'm_explicit not hermitian'

    # for m_explicit, look at the "bare" MPO, without its hc
    m_explicit_bare_MPO = m_explicit.H_MPO.copy()
    m_explicit_bare_MPO.explicit_plus_hc = False  # now this is an MPO without the +hc part
    if expect_non_hermitian_mpo:
        assert not m_explicit_bare_MPO.is_hermitian()

    if expect_non_hermitian_mpo and m_explicit.H_MPO.chi[3] > 2:
        # check for smaller MPO bond dimension
        assert m_explicit.H_MPO.chi[3] < m_plus_hc.H_MPO.chi[3]

    assert m_plus_hc.H_MPO.is_equal(m_manual.H_MPO)
    assert m_explicit.H_MPO.is_equal(m_manual.H_MPO)


@pytest.mark.parametrize(
    'which_site, which_ops, op_string',
    [
        ('spin', 'Sp-Sm', None),
        ('fermion', 'Cd-C', None),
        ('spin-fermion', 'Sp-Sm', None),
        ('spin-fermion', 'Cd-C', None),
    ],
)
def test_model_plus_hc(which_site, which_ops, op_string, L=6):
    """Same as `test_model_plus_hc`, but uses fermions and default JW behavior"""
    if which_site == 'spin':
        lat = lattice.Chain(L=L, site=tenpy.networks.site.SpinHalfSite(None))
    elif which_site == 'fermion':
        lat = lattice.Chain(L=L, site=tenpy.networks.site.FermionSite(None))
    elif which_site == 'spin-fermion':
        lat = lattice.Chain(L=L, site=tenpy.networks.site.SpinHalfFermionSite(None, None))
    else:
        raise ValueError

    if which_ops == 'Sp-Sm':
        hconj_map = dict(Sp='Sm', Sm='Sp', Sz='Sz')
        onsite_op = 'Sp'
        Sp = 'Sp'
        Sm = 'Sm'
        Sz = 'Sz'
        exp_A = 'Sp'
        exp_B = 'Sz'
    elif which_ops == 'Cd-C' and which_site == 'spin-fermion':
        hconj_map = dict(dN='dN', Cdu='Cu', Cd='Cdd', Ntot='Ntot', Cu='Cdu', Cdd='Cd')
        onsite_op = 'dN'
        Sp = 'Cdu'
        Sm = 'Cd'
        Sz = 'Ntot'
        exp_A = 'Cdu'
        exp_B = 'Cu'
    elif which_ops == 'Cd-C' and which_site == 'fermion':
        hconj_map = dict(dN='dN', Cd='C', C='Cd', N='N')
        onsite_op = 'dN'
        Sp = 'Cd'
        Sm = 'C'
        Sz = 'N'
        exp_A = 'Cd'
        exp_B = 'C'
    else:
        raise ValueError

    if op_string is None:
        op_str_kw = {}  # TODO use, more cases
    else:
        raise ValueError

    m_manual = model.CouplingModel(lat)
    m_plus_hc = model.CouplingModel(lat)
    m_explicit = model.CouplingModel(lat, explicit_plus_hc=True)

    print('onsite')
    hx = np.random.random(L)
    m_manual.add_onsite(hx, 0, onsite_op)
    m_manual.add_onsite(hx, 0, hconj_map[onsite_op])
    m_plus_hc.add_onsite(hx, 0, onsite_op, plus_hc=True)
    m_explicit.add_onsite(hx, 0, onsite_op, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit, expect_non_hermitian_mpo=(which_ops != 'Cd-C'))

    print('coupling')
    t = np.random.random(L - 1) + 1.0j * np.random.random(L - 1)
    m_manual.add_coupling(t, 0, Sp, 0, Sm, 1)
    m_manual.add_coupling(np.conj(t), 0, hconj_map[Sm], 0, hconj_map[Sp], -1)
    m_plus_hc.add_coupling(t, 0, Sp, 0, Sm, 1, plus_hc=True)
    m_explicit.add_coupling(t, 0, Sp, 0, Sm, 1, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('multi coupling (2-site)')
    t2 = np.random.random(L - 1)
    ops = [(Sp, [+1], 0), (Sm, [0], 0), (Sz, [0], 0)]
    ops_hc = [(hconj_map[Sz], [0], 0), (hconj_map[Sm], [0], 0), (hconj_map[Sp], [+1], 0)]
    m_manual.add_multi_coupling(t2, ops)
    m_manual.add_multi_coupling(t2, ops_hc)
    m_plus_hc.add_multi_coupling(t2, ops, plus_hc=True)
    m_explicit.add_multi_coupling(t2, ops, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('multi coupling (3-site)')
    t3 = np.random.random(L - 2)
    ops = [(Sp, [+2], 0), (Sm, [+1], 0), (Sz, [0], 0)]
    ops_hc = [(hconj_map[Sz], [0], 0), (hconj_map[Sm], [+1], 0), (hconj_map[Sp], [+2], 0)]
    m_manual.add_multi_coupling(t3, ops)
    m_manual.add_multi_coupling(t3, ops_hc)
    m_plus_hc.add_multi_coupling(t3, ops, plus_hc=True)
    m_explicit.add_multi_coupling(t3, ops, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    if which_ops != 'Cd-C':
        a = -1.5j
        print('1-body local term')
        m_manual.add_local_term(a, [(Sp, [1, 0])])
        m_manual.add_local_term(np.conj(a), [(hconj_map[Sp], [1, 0])])
        m_plus_hc.add_local_term(a, [(Sp, [1, 0])], plus_hc=True)
        m_explicit.add_local_term(a, [(Sp, [1, 0])], plus_hc=True)
        compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('2-body local term')
    b = 2.5
    m_manual.add_local_term(b, [(Sp, [0, 0]), (Sm, [2, 0])])
    m_manual.add_local_term(np.conj(b), [(hconj_map[Sm], [2, 0]), (hconj_map[Sp], [0, 0])])
    m_plus_hc.add_local_term(b, [(Sp, [0, 0]), (Sm, [2, 0])], plus_hc=True)
    m_explicit.add_local_term(b, [(Sp, [0, 0]), (Sm, [2, 0])], plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('3-body local term')
    c = 0.5j
    m_manual.add_local_term(c, [(Sp, [4, 0]), (Sz, [3, 0]), (Sm, [5, 0])])
    m_manual.add_local_term(np.conj(c), [(hconj_map[Sm], [5, 0]), (hconj_map[Sz], [3, 0]), (hconj_map[Sp], [4, 0])])
    m_plus_hc.add_local_term(c, [(Sp, [4, 0]), (Sz, [3, 0]), (Sm, [5, 0])], plus_hc=True)
    m_explicit.add_local_term(c, [(Sp, [4, 0]), (Sz, [3, 0]), (Sm, [5, 0])], plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('exponentially decaying coupling')
    d = 0.25
    l = 0.2
    hc_coeff = np.conj(d)
    if which_ops == 'Cd-C':
        # the interface doesnt allow us to control the order of the two operators, so we need
        # to take care of the anti-commutation manually...
        hc_coeff = -hc_coeff
    m_manual.add_exponentially_decaying_coupling(d, l, exp_A, exp_B)
    m_manual.add_exponentially_decaying_coupling(hc_coeff, np.conj(l), hconj_map[exp_A], hconj_map[exp_B])
    m_plus_hc.add_exponentially_decaying_coupling(d, l, exp_A, exp_B, plus_hc=True)
    m_explicit.add_exponentially_decaying_coupling(d, l, exp_A, exp_B, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('exponentially decaying coupling with subsites')
    subsite_kwargs = dict(subsites=[1, 3, 5], subsites_start=[0, 2])
    e = 3 + 0.42j
    hc_coeff = np.conj(e)
    if which_ops == 'Cd-C':
        # the interface doesnt allow us to control the order of the two operators, so we need
        # to take care of the anti-commutation manually...
        hc_coeff = -hc_coeff
    m_manual.add_exponentially_decaying_coupling(e, l, exp_A, exp_B, **subsite_kwargs)
    m_manual.add_exponentially_decaying_coupling(hc_coeff, l, hconj_map[exp_A], hconj_map[exp_B], **subsite_kwargs)
    m_plus_hc.add_exponentially_decaying_coupling(e, l, exp_A, exp_B, **subsite_kwargs, plus_hc=True)
    m_explicit.add_exponentially_decaying_coupling(e, l, exp_A, exp_B, **subsite_kwargs, plus_hc=True)
    compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)

    print('exponentially decaying centered terms')
    subsites = [1, 3, 5]
    f = 0.42j
    if which_ops == 'Cd-C':
        with pytest.raises(NotImplementedError):
            m_manual.add_exponentially_decaying_centered_terms(f, l, exp_A, exp_B, 3, subsites=subsites)
    else:
        m_manual.add_exponentially_decaying_centered_terms(f, l, exp_A, exp_B, 3, subsites=subsites)
        m_manual.add_exponentially_decaying_centered_terms(
            np.conj(f), l, hconj_map[exp_A], hconj_map[exp_B], 3, subsites=subsites
        )
        m_plus_hc.add_exponentially_decaying_centered_terms(f, l, exp_A, exp_B, 3, subsites=subsites, plus_hc=True)
        m_explicit.add_exponentially_decaying_centered_terms(f, l, exp_A, exp_B, 3, subsites=subsites, plus_hc=True)
        compare_models_plus_hc(m_manual, m_plus_hc, m_explicit)


class DisorderedLatticeModel(model.CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        return tenpy.networks.site.SpinHalfSite(conserve, sort_charge=True)

    def init_lattice(self, model_params):
        lat = super().init_lattice(model_params)
        sigma = model_params.get('disorder_sigma', 0.1)
        shape = lat.shape + (lat.basis.shape[-1],)
        lat.position_disorder = np.random.normal(scale=sigma, size=shape)
        return lat

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            dist = self.lat.distance(u1, u2, dx)
            self.add_coupling(J / dist, u1, 'Sz', u2, 'Sz', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            dist = self.lat.distance(u1, u2, dx)
            self.add_coupling(J / dist, u1, 'Sx', u2, 'Sx', dx)


@pytest.mark.parametrize('bc', ['open', 'periodic'])
def test_disordered_lattice_model(bc, J=2.0):
    model_params = {
        'lattice': 'Kagome',
        'Lx': 2,
        'Ly': 3,
        'bc_y': bc,
        'bc_x': bc,
        'bc_MPS': 'finite' if bc == 'open' else 'infinite',
        'disorder_sigma': 0.1,
        'J': J,
    }
    M = DisorderedLatticeModel(model_params)
    terms = M.all_coupling_terms().to_TermList()
    for i, j, op, need_pbc in [
        ([0, 0, 0], [0, 0, 1], 'Sz', False),
        ([1, 0, 0], [0, 0, 1], 'Sz', False),
        ([1, 0, 2], [0, 1, 1], 'Sz', False),
        ([0, 0, 1], [0, 1, 0], 'Sx', False),
        ([1, 1, 2], [0, 2, 0], 'Sx', False),
        ([0, 2, 2], [1, 2, 0], 'Sx', False),
        ([0, 2, 2], [1, 2, 0], 'Sx', False),
        ([1, 0, 1], [2, 0, 0], 'Sz', True),
        ([1, 1, 1], [2, 0, 2], 'Sz', True),
        ([1, 2, 2], [1, 3, 0], 'Sz', True),
    ]:
        if need_pbc and bc == 'open':
            continue
        ij = np.array([i, j])
        mps_i, mps_j = M.lat.lat2mps_idx(ij)
        pos_i, pos_j = M.lat.position(ij)
        dist = np.linalg.norm(pos_i - pos_j)
        if need_pbc:
            dist = min(dist, np.linalg.norm(pos_i - pos_j + M.lat.basis[1] * M.lat.Ls[1]))
        try:
            idx = terms.terms.index([(op, mps_i), (op, mps_j)])
        except ValueError:
            idx = terms.terms.index([(op, mps_j), (op, mps_i)])
        assert abs(terms.strength[idx] - J / dist) < 1.0e-14


def test_fixes_511(L=6, t=1.234, tp=2.54):
    # https://github.com/tenpy/tenpy/issues/511

    class TTprimeSpinfulChain(model.CouplingMPOModel):
        """Spin-1/2 fermions on a 1D chain with NN hopping t and NNN hopping t'."""

        def init_sites(self, p):
            # conserve total N and Sz; degenerate spin DOF
            return tenpy.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None)

        # CouplingMPOModel already defaults to a Chain lattice with length p['L'].

        def init_terms(self, p):
            t = float(p.get('t', 1.0))  # NN hopping
            tp = float(p.get('tp', 0.0))  # NNN hopping
            mu = float(p.get('mu', 0.0))  # chemical potential
            U = float(p.get('U', 0.0))  # onsite Hubbard U (optional)

            # onsite: -mu * (n_up + n_down) + U * n_up n_down
            self.add_onsite(-mu, 0, 'Ntot')
            if abs(U) > 0:
                self.add_onsite(U, 0, 'NuNd')

            # NN hopping: -t * (c†_{iσ} c_{i+1,σ} + h.c.)
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)  # spin ↑
                self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)  # spin ↓

            # NNN hopping: -t' * (c†_{iσ} c_{i+2,σ} + h.c.)
            for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
                self.add_coupling(-tp, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(-tp, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

    m_NN = TTprimeSpinfulChain(dict(L=L, t=t, tp=0, mu=0, bc_MPS='finite'))
    m_NNN = TTprimeSpinfulChain(dict(L=L, t=0, tp=tp, mu=0, bc_MPS='finite'))
    m_full = TTprimeSpinfulChain(dict(L=L, t=t, tp=tp, mu=0, bc_MPS='finite'))

    H_NN = get_numpy_Hamiltonian(m_NN)
    H_NNN = get_numpy_Hamiltonian(m_NNN)
    H_full = get_numpy_Hamiltonian(m_full)

    assert np.allclose(H_full, H_NN + H_NNN)
