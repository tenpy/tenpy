"""A collection of tests to check the functionality of `tenpy.vumps`"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import vumps
from tenpy.networks import mps
import pytest
import numpy as np
from scipy import integrate

def e0_tranverse_ising(g=0.5):
    """Exact groundstate energy of transverse field Ising.

    H = - J sigma_z sigma_z + g sigma_x
    Can be obtained by mapping to free fermions.
    """
    return integrate.quad(_f_tfi, 0, np.pi, args=(g, ))[0]


def _f_tfi(k, g):
    return -2 * np.sqrt(1 + g**2 - 2 * g * np.cos(k)) / np.pi / 2.


params = [
    # L     engine  mixer
    (1, vumps.SingleSiteVUMPSEngine, False),
    (2, vumps.SingleSiteVUMPSEngine, False), 
    (2, vumps.TwoSiteVUMPSEngine, False),
    (2, vumps.TwoSiteVUMPSEngine, "SubspaceExpansion"),
    (3, vumps.TwoSiteVUMPSEngine, "DensityMatrixMixer"),
]


@pytest.mark.parametrize("L, engine, mixer", params)
@pytest.mark.slow
def test_vumps(L, engine, mixer, g=1.2):
    model_params = dict(L=L, J=1., g=g, bc_MPS='infinite', conserve=None)
    M = TFIChain(model_params)
    if engine == vumps.SingleSiteVUMPSEngine:
        psi = mps.MPS.from_desired_bond_dimension(M.lat.mps_sites(), 32, bc='infinite')
    else:
        state = [0] * L  # Ferromagnetic Ising
        psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc='infinite')
    vumps_pars = {
        'combine': False,
        'mixer': mixer,
        'chi_list': {
            0: 10,
            5: 30
        },
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-8,
        'N_sweeps_check': 1,
        'mixer_params': {
            'disable_after': 5,
            'amplitude': 1.e-5,
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'max_sweeps': 20,
    }
    if not mixer:
        del vumps_pars['mixer_params']  # avoid warning of unused parameter
    eng = engine(psi, M, vumps_pars)
    Edmrg, psi = eng.run()
    
    # compare exact solution for transverse field Ising model
    Eexact = e0_tranverse_ising(g)
    print("E_DMRG={Edmrg:.12f} vs E_exact={Eex:.12f}".format(Edmrg=Edmrg, Eex=Eexact))
    print("relative energy error: {err:.2e}".format(err=abs((Edmrg - Eexact) / Eexact)))
    print("norm err:", psi.norm_test())
    Edmrg2 = np.mean(psi.expectation_value(M.H_bond))
    Edmrg3 = M.H_MPO.expectation_value(psi)
    assert abs((Edmrg - Eexact) / Eexact) < 1.e-10
    assert abs((Edmrg - Edmrg2) / Edmrg2) < max(1.e-10, np.max(psi.norm_test()))
    assert abs((Edmrg - Edmrg3) / Edmrg3) < max(1.e-10, np.max(psi.norm_test()))