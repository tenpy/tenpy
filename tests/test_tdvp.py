#!/usr/bin/python2
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3
import numpy as np
import copy
import pickle
import tenpy.linalg.np_conserved as npc
import tenpy.models.spins
import tenpy.networks.mps as mps
import tenpy.networks.site as site
from tenpy.algorithms import tdvp
from tenpy.algorithms import tebd
import sys
import tdvp_numpy
import tenpy.networks.mpo
import tenpy.models.model as model
import tenpy.models.lattice
from tenpy.networks.mps import MPS
from tenpy.tools.misc import inverse_permutation
import pytest


# TODO: no need to convert everything to numpy...
# just compare the np_conserved TEBD with np_conserved TDVP, using mps.overlap()
# or even better: directly compare to ED for small system
def overlap(mps1, mps2):
    """Calculate overlap <self|mps2>.

    Performs conjugation of self!
    """
    X = np.ones((1, 1))
    L = len(mps1)
    for i in range(0, L):
        tmp = np.tensordot(X, np.conj(mps1[i]), axes=[[0], [1]])
        X = np.tensordot(tmp, mps2[i], axes=[[0, 1], [1, 0]])
    overlap = X.reshape(())
    return overlap


@pytest.mark.slow
def test_tdvp():
    L = 10
    J = 1
    chi = 20
    delta_t = 0.01
    parameters = {
        'L': L,
        'S': 0.5,
        'conserve': 'Sz',
        'Jz': 1.0,
        'Jy': 1.0,
        'Jx': 1.0,
        'hx': 0.0,
        'hy': 0.0,
        'hz': 0.0,
        'muJ': 0.0,
        'bc_MPS': 'finite',
    }

    heisenberg = tenpy.models.spins.SpinChain(parameters)
    H_MPO = heisenberg.H_MPO
    h_test = []
    for i_sites in range(H_MPO.L):
        h_test.append(H_MPO.get_W(i_sites).transpose(['wL', 'wR', 'p*', 'p']).to_ndarray())

    def random_prod_state_tenpy(L, a_model):
        product_state = []
        #the numpy mps used to compare
        psi_compare = []
        sz = 2. * np.random.randint(0, 2, size=L) - 1.0
        for i in range(L):
            psi_compare.append(np.zeros((2, 1, 1)))
            if sz[i] > 0:
                product_state += ["up"]
                psi_compare[-1][0, 0, 0] = 1
            else:
                product_state += ["down"]
                psi_compare[-1][1, 0, 0] = 1

        psi = MPS.from_product_state(a_model.lat.mps_sites(),
                                     product_state,
                                     bc=a_model.lat.bc_MPS,
                                     form='B')
        psi_converted = []
        for i in range(L):
            site = psi.sites[i]
            perm = site.perm
            B_tmp = psi.get_B(i).transpose(['p', 'vL', 'vR']).to_ndarray()
            B = B_tmp[inverse_permutation(perm), :, :]
            B = B[::-1, :, :]
            psi_converted.append(B)

        return psi

    psi = random_prod_state_tenpy(heisenberg.lat.N_sites, heisenberg)
    N_steps = 50
    tebd_params = {
        'order': 2,
        'dt': delta_t,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }

    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }

    psi_tdvp2 = copy.deepcopy(psi)
    engine = tebd.TEBDEngine(psi, heisenberg, tebd_params)
    tdvp_engine = tdvp.TDVPEngine(psi_tdvp2, heisenberg, tdvp_params)
    engine.run()
    tdvp_engine.run_two_sites(N_steps)
    ov = psi.overlap(psi_tdvp2)
    print("overlap TDVP and TEBD")
    psi = engine.psi
    print("difference")
    print(np.abs(1 - np.abs(ov)))
    assert np.abs(1 - np.abs(ov)) < 1e-10
    print("two sites tdvp works")

    # test that the initial conditions are the same

    tdvp_engine = tdvp.TDVPEngine(psi, heisenberg, tdvp_params)
    psit_compare = []
    for i in range(L):
        B_tmp = psi.get_B(i).transpose(['p', 'vL', 'vR']).to_ndarray()
        B = B_tmp[::-1, :, :]
        psit_compare.append(B)
    #**********************************************************************************************************
    #Initialize TDVP
    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'N_steps': 1,
    }
    tdvp_engine = tdvp.TDVPEngine(psi, heisenberg, tdvp_params)
    for t in range(10):
        tdvp_engine.run_one_site(N_steps=1)
        psit_compare, Rp_list, spectrum = tdvp_numpy.tdvp(psit_compare,
                                                          h_test,
                                                          0.5 * 1j * delta_t,
                                                          Rp_list=None)
        psit_ = []
    for i in range(L):
        B = psi.get_B(i).transpose(['p', 'vL', 'vR']).to_ndarray()
        B = B[::-1, :, :]
        psit_.append(B)
    assert np.abs(np.abs(overlap(psit_, psit_compare)) - 1.0) < 1e-13
    print("one site TDVP works")
