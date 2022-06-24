#!/usr/bin/python2
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3
import numpy as np
import pytest
from tenpy.models.spins import SpinChain
from tenpy.algorithms import tdvp
from tenpy.algorithms import tebd
from tenpy.networks.mps import MPS


@pytest.mark.slow
def test_tdvp(eps=1.e-5):
    """compare overlap from TDVP with TEBD """
    L = 8
    chi = 20  # no truncation necessary!
    delta_t = 0.01
    parameters = {
        'L': L,
        'S': 0.5,
        'conserve': None,
        'Jx': 1.0,
        'Jy': 1.0,
        'Jz': 1.0,
        'hx': np.random.random(L),
        'hz': np.random.random(L),
        'bc_MPS': 'finite',
    }

    M = SpinChain(parameters)
    # prepare system in product state
    product_state = ["up", "down"] * (L // 2)
    psi_tebd = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    N_steps = 2
    tebd_params = {
        'order': 4,
        'dt': delta_t,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }

    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }

    # start by comparing TEBD and 2-site TDVP (increasing bond dimension)
    psi_tdvp = psi_tebd.copy()
    tebd_engine = tebd.TEBDEngine(psi_tebd, M, tebd_params)
    tdvp2_engine = tdvp.TwoSiteTDVPEngine(psi_tdvp, M, tdvp_params)
    for _ in range(3):
        tebd_engine.run()
        tdvp2_engine.run()
        ov = psi_tebd.overlap(psi_tdvp)
        print(tdvp2_engine.evolved_time, "ov = 1. - ", ov - 1.)
        assert np.abs(1 - ov) < eps
        Sz_tebd = psi_tebd.expectation_value('Sz')
        Sz_tdvp = psi_tdvp.expectation_value('Sz')
        assert np.max(np.abs(Sz_tebd - Sz_tdvp)) < eps

    # now compare TEBD and 1-site TDVP (constant bond dimension)
    tdvp_params['start_time'] = tdvp2_engine.evolved_time
    tdvp1_engine = tdvp.SingleSiteTDVPEngine(psi_tdvp, M, tdvp_params)
    print('single-site tdvp')
    for _ in range(3):
        tebd_engine.run()
        tdvp1_engine.run()
        ov = psi_tebd.overlap(psi_tdvp)
        print(tdvp1_engine.evolved_time, "ov = 1. - ", ov - 1.)
        assert np.abs(1 - np.abs(ov)) < 1e-5
        Sz_tebd = psi_tebd.expectation_value('Sz')
        Sz_tdvp = psi_tdvp.expectation_value('Sz')
        assert np.max(np.abs(Sz_tebd - Sz_tdvp)) < eps
