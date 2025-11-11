"""A collection of tests to check the functionality of `tenpy.dmrg`"""
# Copyright (C) TeNPy Developers, Apache license

import warnings

import numpy as np
import pytest
from scipy import integrate

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg, dmrg_parallel
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.models.spins import DipolarSpinChain, SpinChain
from tenpy.models.tf_ising import TFIChain
from tenpy.networks import mps
from tenpy.networks.site import SpinHalfSite


def e0_transverse_ising(g=0.5):
    """Exact groundstate energy of transverse field Ising.

    H = - J sigma_z sigma_z + g sigma_x
    Can be obtained by mapping to free fermions.
    """
    return integrate.quad(_f_tfi, 0, np.pi, args=(g,))[0]


def _f_tfi(k, g):
    return -2 * np.sqrt(1 + g**2 - 2 * g * np.cos(k)) / np.pi / 2.0


params = [
    # bc     combine  mixer n
    ('finite', True, False, 2),  # simplest case
    ('finite', True, True, 1),
    # 1-site DMRG without mixer is expected to fail!
    ('finite', True, 'DensityMatrixMixer', 2),
    ('finite', True, 'SubspaceExpansion', 2),
    ('finite', False, True, 2),
    #  ('finite', False, False, 2),
    ('infinite', True, False, 2),  # simplest case infinite
    ('infinite', True, 'DensityMatrixMixer', 2),  # with mixer
    ('infinite', True, 'SubspaceExpansion', 2),
    ('infinite', True, True, 1),
    #  ('infinite', True, True, 2),
    ('infinite', False, True, 1),
    #  ('infinite', False, True, 2),
    #  ('infinite', False, False, 2)
]


@pytest.mark.parametrize('bc_MPS, combine, mixer, n', params)
@pytest.mark.slow
def test_dmrg(bc_MPS, combine, mixer, n, g=1.2):
    L = 2 if bc_MPS == 'infinite' else 8
    model_params = dict(L=L, J=1.0, g=g, bc_MPS=bc_MPS, conserve=None)
    M = TFIChain(model_params)
    state = [0] * L  # Ferromagnetic Ising
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width)
    dmrg_pars = {
        'combine': combine,
        'mixer': mixer,
        'chi_list': {0: 10, 5: 30},
        'max_E_err': 1.0e-12,
        'max_S_err': 1.0e-8,
        'N_sweeps_check': 4 if bc_MPS == 'infinite' else 1,
        'mixer_params': {
            'disable_after': 10,
            'amplitude': 1.0e-7,
        },
        'trunc_params': {
            'svd_min': 1.0e-10,
        },
        'max_N_for_ED': 20,  # small enough that we test both diag_method=lanczos and ED_block!
        'max_sweeps': 40,
        'active_sites': n,
    }
    if not mixer:
        del dmrg_pars['mixer_params']  # avoid warning of unused parameter
    if bc_MPS == 'infinite':
        # if mixer is not None:
        #     dmrg_pars['mixer_params']['amplitude'] = 1.e-12  # don't actually contribute...
        dmrg_pars['start_env'] = 1
    res = dmrg.run(psi, M, dmrg_pars)
    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        E_ED, psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), 'range', do_conj=True)
        print('E_DMRG={Edmrg:.14f} vs E_exact={Eex:.14f}'.format(Edmrg=res['E'], Eex=E_ED))
        print('compare with ED: overlap = ', abs(ov) ** 2)
        assert abs(abs(ov) - 1.0) < 1.0e-8  # unique groundstate: finite size gap!
        var = M.H_MPO.variance(psi)
        assert var < 1.0e-8
    else:
        # compare exact solution for transverse field Ising model
        Edmrg = res['E']
        Eexact = e0_transverse_ising(g)
        print(f'E_DMRG={Edmrg:.12f} vs E_exact={Eexact:.12f}')
        print(f'relative energy error: {abs((Edmrg - Eexact) / Eexact):.2e}')
        print('norm err:', psi.norm_test())
        Edmrg2 = np.mean(psi.expectation_value(M.H_bond))
        Edmrg3 = M.H_MPO.expectation_value(psi)
        assert abs((Edmrg - Eexact) / Eexact) < 1.0e-10
        assert abs((Edmrg - Edmrg2) / Edmrg2) < max(1.0e-10, np.max(psi.norm_test()))
        assert abs((Edmrg - Edmrg3) / Edmrg3) < max(1.0e-10, np.max(psi.norm_test()))


@pytest.mark.slow
def test_dmrg_rerun(L=2):
    bc_MPS = 'infinite'
    model_params = dict(L=L, J=1.0, g=1.5, bc_MPS=bc_MPS, conserve=None)
    M = TFIChain(model_params)
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), [0] * L, bc=bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width)
    dmrg_pars = {'chi_list': {0: 5, 5: 10}, 'N_sweeps_check': 4, 'combine': True}
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_pars)
    E1, _ = eng.run()
    assert abs(E1 - -1.67192622) < 1.0e-6
    model_params['g'] = 1.3
    M = TFIChain(model_params)
    del eng.options['chi_list']
    new_chi = 15
    eng.options['trunc_params']['chi_max'] = new_chi
    eng.init_env(M)
    E2, psi = eng.run()
    assert max(psi.chi) == new_chi
    assert abs(E2 - -1.50082324) < 1.0e-6


params = [
    ('TwoSiteDMRGEngine', 'lanczos'),
    ('TwoSiteDMRGEngine', 'arpack'),
    ('TwoSiteDMRGEngine', 'ED_block'),
    ('TwoSiteDMRGEngine', 'ED_all'),
    ('SingleSiteDMRGEngine', 'ED_block'),
]


@pytest.mark.slow
@pytest.mark.parametrize('engine, diag_method', params)
def test_dmrg_diag_method(engine, diag_method, tol=1.0e-6):
    bc_MPS = 'finite'
    model_params = dict(L=6, S=0.5, bc_MPS=bc_MPS, conserve='Sz')
    M = SpinChain(model_params)
    # chose total Sz= 4, not 3=6/2, i.e. not the sector with lowest energy!
    # make sure below that we stay in that sector, if we're supposed to.
    init_Sz_4 = ['up', 'down', 'up', 'up', 'up', 'down']
    psi_Sz_4 = mps.MPS.from_product_state(
        M.lat.mps_sites(), init_Sz_4, bc=bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_pars = {
        'N_sweeps_check': 1,
        'combine': True,
        'max_sweeps': 5,
        'diag_method': diag_method,
        'mixer': True,
    }
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization()
    if diag_method == 'ED_all':
        charge_sector = None  # allow to change the sector
    else:
        charge_sector = [2]  # don't allow to change the sector
    E_ED, psi_ED = ED.groundstate(charge_sector=charge_sector)

    DMRGEng = dmrg.__dict__.get(engine)
    print('DMRGEng = ', DMRGEng)
    print('setting diag_method = ', dmrg_pars['diag_method'])
    eng = DMRGEng(psi_Sz_4.copy(), M, dmrg_pars)
    E0, psi0 = eng.run()
    eng.options['lanczos_params'].touch('P_tol')
    print(f'E0 = {E0:.15f}')
    assert abs(E_ED - E0) < tol
    ov = npc.inner(psi_ED, ED.mps_to_full(psi0), 'range', do_conj=True)
    assert abs(abs(ov) - 1) < tol


@pytest.mark.slow
def test_dmrg_excited(eps=1.0e-12):
    # checks ground state and 2 excited states (in same symmetry sector) for a small system
    # (without truncation)
    L, g = 8, 1.3
    bc = 'finite'
    model_params = dict(L=L, J=1.0, g=g, bc_MPS=bc, conserve='parity', sort_charge=True)
    M = TFIChain(model_params)
    # compare to exact solution
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization()
    # Note: energies sorted by charge sector (first 0), then ascending -> perfect for comparison
    print('Exact diag: E[:5] = ', ED.E[:5])
    print('Exact diag: (smallest E)[:10] = ', np.sort(ED.E)[:10])

    psi_ED = [ED.V.take_slice(i, 'ps*') for i in range(5)]
    print('charges : ', [psi.qtotal for psi in psi_ED])

    # first DMRG run
    psi0 = mps.MPS.from_product_state(M.lat.mps_sites(), [0] * L, bc=bc, unit_cell_width=M.lat.mps_unit_cell_width)
    dmrg_pars = {'N_sweeps_check': 1, 'lanczos_params': {'reortho': False}, 'diag_method': 'lanczos', 'combine': True}
    eng0 = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_pars)
    E0, psi0 = eng0.run()
    assert abs((E0 - ED.E[0]) / ED.E[0]) < eps
    ov = npc.inner(psi_ED[0], ED.mps_to_full(psi0), 'range', do_conj=True)
    assert abs(abs(ov) - 1.0) < eps  # unique groundstate: finite size gap!
    # second DMRG run for first excited state
    psi1 = mps.MPS.from_product_state(M.lat.mps_sites(), [0] * L, bc=bc, unit_cell_width=M.lat.mps_unit_cell_width)
    eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_pars, orthogonal_to=[psi0])
    E1, psi1 = eng1.run()
    assert abs((E1 - ED.E[1]) / ED.E[1]) < eps
    ov = npc.inner(psi_ED[1], ED.mps_to_full(psi1), 'range', do_conj=True)
    assert abs(abs(ov) - 1.0) < eps  # unique groundstate: finite size gap!
    # and a third one to check with 2 eigenstates
    # note: different initial state necessary, otherwise H is 0
    psi2 = mps.MPS.from_singlets(
        psi0.sites[0], L, [(0, 1), (2, 3), (4, 5), (6, 7)], bc=bc, unit_cell_width=M.lat.mps_unit_cell_width
    )
    eng2 = dmrg.TwoSiteDMRGEngine(psi2, M, dmrg_pars, orthogonal_to=[psi0, psi1])
    E2, psi2 = eng2.run()
    print(E2)
    assert abs((E2 - ED.E[2]) / ED.E[2]) < eps
    ov = npc.inner(psi_ED[2], ED.mps_to_full(psi2), 'range', do_conj=True)
    assert abs(abs(ov) - 1.0) < eps  # unique groundstate: finite size gap!


@pytest.mark.slow
def test_enlarge_mps_unit_cell():
    g = 1.3  # deep in the paramagnetic phase
    bc_MPS = 'infinite'
    model_params = dict(L=2, J=1.0, g=g, bc_MPS=bc_MPS, conserve=None)
    M_2 = TFIChain(model_params)
    M_4 = TFIChain(model_params)
    M_4.enlarge_mps_unit_cell(2)
    psi_2 = mps.MPS.from_product_state(
        M_2.lat.mps_sites(), ['up', 'up'], bc=bc_MPS, unit_cell_width=M_2.lat.mps_unit_cell_width
    )
    psi_4 = mps.MPS.from_product_state(
        M_2.lat.mps_sites(), ['up', 'up'], bc=bc_MPS, unit_cell_width=M_2.lat.mps_unit_cell_width
    )
    psi_4.enlarge_mps_unit_cell(2)
    dmrg_params = {
        'combine': True,
        'max_sweeps': 30,
        'update_env': 0,
        'mixer': False,  # not needed in this case
        'trunc_params': {'svd_min': 1.0e-10, 'chi_max': 50},
    }
    E_2, _ = dmrg.TwoSiteDMRGEngine(psi_2, M_2, dmrg_params).run()
    E_4, _ = dmrg.TwoSiteDMRGEngine(psi_4, M_4, dmrg_params).run()
    assert abs(E_2 - E_4) < 1.0e-12
    psi_2.enlarge_mps_unit_cell(2)
    ov = abs(psi_2.overlap(psi_4, understood_infinite=True))
    print('ov = ', ov)
    assert abs(ov - 1.0) < 1.0e-12


def test_chi_list():
    assert dmrg.chi_list(3) == {0: 3}
    assert dmrg.chi_list(12, 12, 5) == {0: 12}
    assert dmrg.chi_list(24, 12, 5) == {0: 12, 5: 24}
    assert dmrg.chi_list(27, 12, 5) == {0: 12, 5: 24, 10: 27}


@pytest.mark.slow
@pytest.mark.parametrize('N, bc_MPS', [(6, 'finite'), (2, 'infinite')])
def test_dmrg_explicit_plus_hc(N, bc_MPS, tol=1.0e-13, bc='finite'):
    model_params = dict(L=2 * N, Jx=1.0, Jy=1.0, Jz=2.5, hz=5.125, bc_MPS=bc_MPS)
    dmrg_params = dict(N_sweeps_check=2, mixer=True, trunc_params={'chi_max': 50})
    M1 = SpinChain(model_params)
    model_params['explicit_plus_hc'] = True
    M2 = SpinChain(model_params)
    assert M2.H_MPO.explicit_plus_hc
    psi1 = mps.MPS.from_product_state(
        M1.lat.mps_sites(), ['up', 'down'] * N, bc=bc_MPS, unit_cell_width=M1.lat.mps_unit_cell_width
    )
    E1, psi1 = dmrg.TwoSiteDMRGEngine(psi1, M1, dmrg_params).run()
    psi2 = mps.MPS.from_product_state(
        M2.lat.mps_sites(), ['up', 'down'] * N, bc=bc_MPS, unit_cell_width=M2.lat.mps_unit_cell_width
    )
    E2, psi2 = dmrg.TwoSiteDMRGEngine(psi2, M2, dmrg_params).run()
    print(E1, E2, abs(E1 - E2))
    assert abs(E1 - E2) < tol
    ov = abs(psi1.overlap(psi2, understood_infinite=True))
    print('ov =', ov)
    assert abs(ov - 1) < tol
    dmrg_params['combine'] = True
    psi3 = mps.MPS.from_product_state(
        M2.lat.mps_sites(), ['up', 'down'] * N, bc=bc_MPS, unit_cell_width=M2.lat.mps_unit_cell_width
    )
    E3, psi3 = dmrg_parallel.DMRGThreadPlusHC(psi3, M2, dmrg_params).run()
    print(E1, E3, abs(E1 - E3))
    assert abs(E1 - E3) < tol
    ov = abs(psi1.overlap(psi3, understood_infinite=True))
    print('ov =', ov)
    assert abs(ov - 1) < tol


@pytest.mark.parametrize('N, bc_MPS', [(6, 'finite'), (2, 'infinite')])
def test_dmrg_dipole_conservation(N, bc_MPS, S=1, tol=1.0e-13, J4=0.0):
    dmrg_params = dict(N_sweeps_check=2, mixer=True, trunc_params={'chi_max': 50}, max_sweeps=20)

    # initial_state = ['up', 'down'] * (N // 2) + ['down', 'up'] * (N // 2)
    initial_state = ['up', 'down'] * N
    # initial_state = [1, 1] * N

    # finite passes for J4=0
    with warnings.catch_warnings():  # may issue warning that H is zero in sector. thats ok since we use a mixer.
        warnings.simplefilter('ignore')
        M_dip = DipolarSpinChain(dict(L=2 * N, S=S, J3=1.0, J4=J4, bc_MPS=bc_MPS, conserve='dipole'))
        psi_dip = mps.MPS.from_product_state(
            M_dip.lat.mps_sites(), initial_state, bc=bc_MPS, unit_cell_width=M_dip.lat.mps_unit_cell_width
        )
        E_dip, psi_dip = dmrg.TwoSiteDMRGEngine(psi_dip, M_dip, dmrg_params).run()

    # run without dipole conservation for comparison
    with warnings.catch_warnings():  # may issue warning that H is zero in sector. thats ok since we use a mixer.
        warnings.simplefilter('ignore')
        M = DipolarSpinChain(dict(L=2 * N, S=S, J3=1.0, J4=J4, bc_MPS=bc_MPS, conserve='Sz'))
        psi = mps.MPS.from_product_state(
            M.lat.mps_sites(), initial_state, bc=bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
        )
        E, psi = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params).run()

    # can not compute overlap easily due to different chinfo...
    print(f'E={E}')
    print(f'E_dip={E_dip}')
    print(f'diff : {abs(E - E_dip)}')

    if bc_MPS == 'infinite':
        # DMRG runs, which is reassuring, but energy is above Sz-dmrg by ~1e-5
        # Takes quite long too to fully converge (here we set max_sweeps=20).
        # looks like energy is oscillating (if we would let it run longer).
        # not clear that we selected the correct charge sector here, i.e. the GS within this
        # dipole sector might be higher in energy than the GS in the Sz sector
        tol = 2e-5

    assert abs(E - E_dip) < tol


@pytest.mark.parametrize('L, bc_MPS', [(12, 'finite'), (4, 'infinite')])
def test_dmrg_mixer_cleanup(L, bc_MPS):
    model_params = dict(L=L, Jx=1.0, Jy=1.0, Jz=2.5, hz=5.125, bc_MPS=bc_MPS, conserve='parity')
    dmrg_params = dict(N_sweeps_check=2, mixer=True, trunc_params={'chi_max': 50})
    model = SpinChain(model_params)
    psi = mps.MPS.from_lat_product_state(model.lat, [['up'], ['down']])
    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    # do a few steps of engine.run()
    engine.shelve = False
    engine.pre_run_initialize()
    for _ in range(3):
        engine.run_iteration()
    assert engine.mixer is not None
    old_psi = engine.psi.copy()
    old_LP = [engine.env.get_LP(i) for i in range(psi.L)]
    old_RP = [engine.env.get_RP(i) for i in range(psi.L)]

    print(f'Checking consistency of old environments...')
    old_contractions = [engine.env.full_contraction(i) for i in range(L)]

    print('Calling mixer_cleanup()...')
    engine.mixer_deactivate()
    engine.mixer_cleanup()

    print('Checking sanity...')
    engine.psi.test_sanity()

    print('Make sure envs were updated...')
    new_LP = [engine.env.get_LP(i) for i in range(psi.L)]
    new_RP = [engine.env.get_RP(i) for i in range(psi.L)]
    for i in range(L):
        if not (bc_MPS == 'finite' and i == 0):
            assert new_LP[i] is not old_LP[i]
        if not (bc_MPS == 'finite' and i == L - 1):
            assert new_RP[i] is not old_RP[i]

    print(f'Checking consistency of new environments...')
    for i in range(L):
        assert np.allclose(engine.env.full_contraction(i), old_contractions[i])

    print(f'Check that expectation values have not changed...')
    for op in ['Sx', 'Sz']:
        assert np.allclose(engine.psi.expectation_value(op), old_psi.expectation_value(op))


class _TransverseClusterModel(CouplingModel, MPOModel):
    def __init__(self, model_params):
        L = model_params.get('L', 2)
        B = model_params.get('B', 0)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        site = SpinHalfSite(conserve=None)
        lat = Chain(L, site, bc='periodic', bc_MPS=bc_MPS)
        CouplingModel.__init__(self, lat)
        self.add_onsite(-B, 0, 'Sigmax')
        self.add_multi_coupling(-1, [('Sigmaz', -1, 0), ('Sigmax', 0, 0), ('Sigmaz', 1, 0)])
        MPOModel.__init__(self, lat, self.calc_H_MPO())


@pytest.mark.parametrize('model', ['tfi', 'cluster'])
def test_segment_dmrg(model):
    if model == 'tfi':
        model = TFIChain(dict(J=1, g=1.5, L=2, bc_MPS='infinite'))
    elif model == 'cluster':
        # model from https://tenpy.johannes-hauschild.de/viewtopic.php?t=691
        model = _TransverseClusterModel({})

    # first dmrg run for *infinite* lattice
    psi0_infinite = mps.MPS.from_lat_product_state(model.lat, [['up']])
    trunc_params = dict(chi_max=100, svd_min=1e-10)
    dmrg_params = dict(mixer=True, max_E_err=1e-10, trunc_params=trunc_params)
    eng0 = dmrg.TwoSiteDMRGEngine(psi0_infinite, model, dmrg_params)
    eng0.run()

    model_segment = model.extract_segment(enlarge=10)
    psi0_segment = psi0_infinite.extract_segment(*model_segment.lat.segment_first_last)
    init_env_data = eng0.env.get_initialization_data(*model_segment.lat.segment_first_last)

    psi1_segment = psi0_segment.copy()
    psi1_segment.perturb()
    eng1 = dmrg.TwoSiteDMRGEngine(
        psi1_segment, model_segment, dmrg_params, resume_data={'init_env_data': init_env_data}
    )
    eng1.run()

    assert np.allclose(psi1_segment.entanglement_entropy(), np.mean(psi0_infinite.entanglement_entropy()))
    assert np.allclose(psi0_infinite.expectation_value('Sz', [0]), psi1_segment.expectation_value('Sz', [0]))
    assert np.allclose(psi0_infinite.expectation_value('Sx', [0]), psi1_segment.expectation_value('Sx', [0]))

    with pytest.warns():
        eng0.options.warn_unused()
        eng1.options.warn_unused()
