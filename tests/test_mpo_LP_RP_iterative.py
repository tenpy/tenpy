"""test of graph structure in :class:`tenpy.networks.mpo.MPO`
and method `init_LP_RP_iterative` in :class:`tenpy.netoworks.mpo.MPOEnvironment`."""
# Copyright (C) TeNPy Developers, Apache license

import pytest

from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as dmrg_eng
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Square
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mpo import MPO, MPOEnvironment, MPOEnvironmentBuilder, MPOTransferMatrix
from tenpy.networks.mps import MPS

# networks
from tenpy.networks.site import SpinHalfSite

# ----- TEST FUNCTIONS -----


def helper_test_graph(H, name):
    # check that graph is a 1 to 1 mapping of H
    chis = H.chi
    for j, W in enumerate(H._W):
        for jL in range(chis[j]):
            for jR in range(chis[j + 1]):
                op = W[jL, jR]
                if npc.norm(op) < 1e-12:
                    assert (jL, jR) not in H._graph[j], name + ': entry of norm zero found in graph'
                else:
                    assert npc.norm(op - H._graph[j][(jL, jR)]) < 1e-12, name + f': _graph[{j}][({jL},{jR})] wrong'


def helper_test_init_env(psi, E, H, name, tol=1e-10):
    env_base, E_base, _ = MPOTransferMatrix.find_init_LP_RP(H, psi, 0, psi.L - 1, calc_E=True)
    env = MPOEnvironmentBuilder(H, psi)
    init_env, _, E_iter = env.init_LP_RP_iterative(calc_E=True, which='both')
    env_base['init_RP'].itranspose(['wL', 'vL', 'vL*'])
    env_base['init_LP'].itranspose(['wR', 'vR', 'vR*'])
    assert npc.norm(env_base['init_LP'] - init_env['init_LP']) < 1e-8, name + ': LP_iterative not converged'
    assert npc.norm(env_base['init_RP'] - init_env['init_RP']) < 1e-8, name + ': RP_iterative not converged'
    assert abs(E_iter[0] - E_base[0]) < tol
    assert abs(E_iter[1] - E_base[1]) < tol, name + ": Energies of iterative environment intialization don't match"


def helper_test_H_square(psi, square_of_H, H, name, state):
    e_iter1 = MPOEnvironmentBuilder(square_of_H, psi)
    e_iter2 = MPOEnvironmentBuilder(H, psi)
    ee_iter2 = MPOEnvironment(psi, H, psi)
    _, env1 = e_iter1.init_LP_RP_iterative(which='both')
    _, env2 = e_iter2.init_LP_RP_iterative(which='both')

    # compute variance from 0-site Heff and test values along the way
    def var(e1, e2, rhoL, rhoR):
        def scale_rho(e, rho, ax='vL'):
            res = e.scale_axis(rho, axis=ax)
            res = res.iscale_axis(rho, axis=ax + '*')
            return res

        def check_res(res, max_pow, env_name='env1'):
            combs = [(j, jj) for j in range(1, max_pow + 1) for jj in range(j)] + [(j, j) for j in range(max_pow + 1)]
            for c in combs:
                entries = [x for x in res[c]]
                if c[0] != c[1]:
                    entries += res[(c[1], c[0])]
                max_diff = max([abs(entries[0] - x) for x in entries[1:]])
                assert max_diff < 1e-10, name + ': Heff does not match for: ' + env_name
                if c[0] + c[1] > max_pow:
                    assert abs(entries[0]) < 1e-10, name + ': Higher power environments have overlap for: ' + env_name

        Rs1 = [scale_rho(e, rhoR) for e in e1['init_RP']]
        Ls1 = [scale_rho(e, rhoL, 'vR') for e in e1['init_LP']]
        res_1 = {}
        for jL, eL in enumerate(Ls1):
            for jR, eR in enumerate(Rs1):
                res_1[(jL, jR)] = [
                    npc.tensordot(eL, e1['init_RP'][jR], axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']]),
                    npc.tensordot(e1['init_LP'][jL], eR, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']]),
                ]
        check_res(res_1, 1)
        Rs = [scale_rho(e, rhoR) for e in e2['init_RP']]
        Ls = [scale_rho(e, rhoL, 'vR') for e in e2['init_LP']]
        res_2 = {}
        for jL, eL in enumerate(Ls):
            for jR, eR in enumerate(Rs):
                res_2[(jL, jR)] = [
                    npc.tensordot(eL, e2['init_RP'][jR], axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']]),
                    npc.tensordot(e2['init_LP'][jL], eR, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']]),
                ]
        check_res(res_2, 2, 'env2')
        return (
            2 * res_2[(0, 0)][0] + res_1[(0, 0)][0] ** 2,
            2 * res_2[(1, 0)][0] - 2 * res_1[(0, 0)][0] * res_1[(1, 0)][0],
            2 * res_2[(2, 0)][0] - res_1[(1, 0)][0] ** 2,
        )

    # tests
    rhoR = psi.get_SR(3)
    rhoL = psi.get_SL(0)
    v0, v1, v2 = var(env1, env2, rhoL, rhoR)
    if state == 'up':
        assert abs(v0 - 0) < 1e-15, (
            name + ': variance obtained from environements wrong (part ~n==0) for state: ' + state
        )
        assert abs(v1 - 4) < 1e-15, (
            name + ': variance obtained from environements wrong (part ~n==1) for state: ' + state
        )
        assert abs(v2 - 0) < 1e-15, (
            name + ': variance obtained from environements wrong (part ~n==2) for state: ' + state
        )
    else:
        assert abs(v1) < 1e-12, name + ': variance obtained from environements wrong (part ~n==1) for state: ' + state
        assert abs(v2) < 1e-12, name + ': variance obtained from environements wrong (part ~n==2) for state: ' + state
    # check left eigenvector explicitly
    for j in range(1, ee_iter2.L):
        ee_iter2.del_LP(j)
    ee_iter2.set_LP(0, env2['init_LP'][0], 0)
    LP_prelast = ee_iter2.get_LP(ee_iter2.L - 1, False)
    LP = ee_iter2._contract_LP(ee_iter2.L - 1, LP_prelast)
    for _e in [LP] + env2['init_LP']:
        _e.itranspose(['wR', 'vR', 'vR*'])
    LP_diff = LP - env2['init_LP'][1] - env2['init_LP'][2] - env2['init_LP'][0]
    assert (npc.norm(LP_diff)) < 1e-8, name + 'Left environment of H**2 is not the correct eigenvector'


def helper_test_enlarge_unit_cell(H, name):
    H.enlarge_mps_unit_cell(2)
    chis = H.chi
    for j, W in enumerate(H._W):
        for jL in range(chis[j]):
            for jR in range(chis[j + 1]):
                op = W[jL, jR]
                if npc.norm(op) < 1e-12:
                    assert (jL, jR) not in H._graph[j], (
                        name + ': entry of norm zero found in graph after enlarge_unit_cell()'
                    )
                else:
                    assert npc.norm(op - H._graph[j][(jL, jR)]) < 1e-12, (
                        name + f': _graph[{j}][({jL},{jR})] wrong after enlarge_unit_cell()'
                    )


def helper_test_grid(psi, H, name):
    # ----- Test Grid from MPOEnvironmentBuilder -----
    env = MPOEnvironment(psi, H, psi)
    env2 = MPOEnvironmentBuilder(H, psi)
    # TransferMatrix environments with one unit cell contracted
    LP_base = env.get_LP(0)
    RP_base = env.get_RP(3)
    LP_last = env.get_LP(3)  # TransferMatrix envs
    RP_last = env.get_RP(0)
    LP_last = env._contract_LP(3, LP_last)
    RP_last = env._contract_RP(0, RP_last)
    LP_base.itranspose(['wR', 'vR', 'vR*'])
    RP_base.itranspose(['wL', 'vL', 'vL*'])
    LP_last.itranspose(['wR', 'vR', 'vR*'])
    RP_last.itranspose(['wL', 'vL', 'vL*'])
    # check grids by contracting LP via grids
    grid_L = env2._left_grid()
    for j in range(3):
        env2._contract_left_grid(grid_L, LP_base[j], j)
    for j in range(3):
        assert npc.norm(LP_last[j] - grid_L[3][j][0]) < 1e-12, (
            name + ': left grid contraction does not agree with expected LP'
        )
    # RP
    grid_R = env2._right_grid()
    for j in range(3):
        env2._contract_right_grid(grid_R, RP_base[j], j)
    for j in range(3):
        assert npc.norm(grid_R[0][j][0] - RP_last[j]) < 1e-12, (
            name + ': left grid contraction does not agree with expected LP'
        )


def setup(test_case: int):
    Lx, Ly = 4, 3
    if test_case == 0 or test_case == 2:
        # model setup
        m_tfi = TFIChain({'conserve': 'parity', 'g': 1.5, 'J': 1.0, 'L': Lx, 'bc_MPS': 'infinite'})
        H_tfi = m_tfi.calc_H_MPO()
        psi_tfi = MPS.from_lat_product_state(m_tfi.lat, p_state=[['up']] * Lx, bc='infinite')
        eng = dmrg_eng(psi_tfi, m_tfi, {'min_sweeps': 19, 'max_sweeps': 29, 'trunc_params': {'chi_max': 32}})
        E_tfi, psi_tfi = eng.run()
    if test_case == 1:
        # general coupling model on square lattice
        site_z = SpinHalfSite(conserve='Sz')
        lat = Square(Lx, Ly, site_z, bc='periodic', bc_MPS='infinite')
        m_couple = CouplingModel(lat)
        # exp_decaying
        m_couple.add_exponentially_decaying_coupling(1.2, 0.42, 'Sigmaz', 'Sigmaz', subsites=[1, 4, 7, 10])
        m_couple.add_exponentially_decaying_coupling(0.32, 0.42, 'Sigmaz', 'Sigmaz', subsites=[0, 3, 6, 9])
        # onsite
        for site in range(11):
            m_couple.add_onsite_term(-0.2, site, 'Sigmaz')
        # y-direction
        for s0 in range(0, 12, 3):
            for offset in range(3):
                i = min(s0 + offset, s0 + (offset + 1) % 3)
                j = max(s0 + offset, s0 + (offset + 1) % 3)
                m_couple.add_coupling_term(1.77, i, j, 'Sp', 'Sm')
                m_couple.add_coupling_term(1.77, i, j, 'Sm', 'Sp')
        for s0 in range(0, 12, 4):
            for offset in range(3):
                m_couple.add_multi_coupling_term(
                    1.16,
                    [s0 + offset, s0 + offset + 3, s0 + offset + 6, s0 + offset + 9],
                    ['Sm', 'Sigmaz', 'Sigmaz', 'Sp'],
                    op_string=['Id'] * 3,
                )
                m_couple.add_multi_coupling_term(
                    1.16,
                    [s0 + offset, s0 + offset + 3, s0 + offset + 6, s0 + offset + 9],
                    ['Sp', 'Sigmaz', 'Sigmaz', 'Sm'],
                    op_string=['Id'] * 3,
                )
        H_couple = m_couple.calc_H_MPO()
        psi_couple = MPS.from_lat_product_state(m_couple.lat, [[['up']] * Ly] * Lx, bc='infinite')
        m_couple_MPO = MPOModel(m_couple.lat, H_couple)
        eng = dmrg_eng(psi_couple, m_couple_MPO, {'min_sweeps': 19, 'max_sweeps': 29, 'trunc_params': {'chi_max': 8}})
        E_couple, psi_couple = eng.run()
    if test_case == 2:
        # MPO of H^2 for TFI model
        # pauli_op
        s = SpinHalfSite(conserve='parity')
        Sig_x = s.get_op('Sigmax')
        Sig_y = s.get_op('Sigmay')
        Sig_z = s.get_op('Sigmaz')
        Sig_id = s.get_op('Id')
        # TFI
        grid_tfi = [
            [
                [Sig_id, Sig_x, -1.5 * Sig_z, None, 1.625 * Sig_id],
                [None, None, -Sig_x, -Sig_id, None],
                [None, None, Sig_id, Sig_x, -1.5 * Sig_z],
                [None, None, None, None, -Sig_x],
                [None, None, None, None, Sig_id],
            ]
            for _ in range(Lx)
        ]
        H_tfi_square = MPO.from_grids(
            m_tfi.lat.mps_sites(),
            grid_tfi,
            bc='infinite',
            IdL=0,
            IdR=4,
            mps_unit_cell_width=m_tfi.lat.mps_unit_cell_width,
        )
        psi_up = MPS.from_lat_product_state(m_tfi.lat, [['up']] * Lx, bc='infinite')

    # return H, psi, E, name, N_cycles, cycle_indices, square_of_H, square_of_psi
    if test_case == 0:
        return H_tfi, psi_tfi, E_tfi, 'TFIChain', 2, [0, 2], None, None
    elif test_case == 1:
        return H_couple, psi_couple, E_couple, 'CouplingModel', 4, None, None, None
    elif test_case == 2:
        return H_tfi_square, psi_up, E_tfi, 'square(TFI)', 3, [0, 2, 4], H_tfi, psi_tfi
    else:
        raise ValueError('invalid test_case')


@pytest.mark.parametrize('test_case', [0, 1, 2])
@pytest.mark.slow
def test_init_LP_RP_iterative(test_case):
    H, psi, E, name, N_cycles, cycle_indices, square_of_H, square_of_psi = setup(test_case)
    # check with and without sorted legcharges
    for sort_charges in [0, 1, 2]:
        if sort_charges == 1:
            tperms = H.sort_legcharges()
        if sort_charges == 2:
            H._reset_graph()
        if sort_charges != 1:
            H._make_graph()  # implicit itranspose of W matrices for tests later
            H._order_graph()

        # check that graph is a 1 to 1 mapping of H
        helper_test_graph(H, name)

        # test contract grid for TFI model
        if test_case == 0 and sort_charges == 0:
            helper_test_grid(psi, H, name)

        # environment checks
        if sort_charges != 2 and test_case < 2:
            helper_test_init_env(psi, E, H, name)

        # H^2 test
        if sort_charges == 0 and test_case == 2:
            helper_test_H_square(psi, square_of_H, H, name, 'up')
            helper_test_H_square(square_of_psi, square_of_H, H, name, 'gs')

        # enlarge unit cell
        if sort_charges == 2 and test_case < 2:
            helper_test_enlarge_unit_cell(H, name)

        # ---- Additional checks on graph -----
        assert len(H._cycles) == N_cycles, name + ': wrong number of cycles'
        # outer permutation
        assert H._outer_permutation[0] == H.IdL[0], name + ': wrong IdL index in _outer_permutation'
        assert H._outer_permutation[-1] == H.IdR[-1], name + ': wrong IdR index in _outer_permutation'
        # IdL, IdR cycle
        for val1, val2 in zip(H._cycles[H.IdL[0]], H.IdL):
            assert val1 == val2, name + ': H.IdL different from H._cycles[H.IdL[0]]'
        for val1, val2 in zip(H._cycles[H.IdR[-1]], H.IdR):
            assert val1 == val2, name + ': H.IdR different from H._cycles[H.IdR[-1]]'
        # cycles
        for i0 in H._cycles:
            c = H._cycles[i0]
            assert len(c) == H.L + 1, name + ': wrong cycle length'
            assert c[0] == c[-1], name + ': invalid cycle encountered'
        if sort_charges == 0 and test_case != 1:  # explicit cycle check
            for j_cycle in cycle_indices:
                assert H._cycles[j_cycle] == [j_cycle] * (H.L + 1), name + f': _cycles[{j_cycle}] not as expected'
