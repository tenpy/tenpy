"""A collection of tests to check the functionality of `tenpy.vumps`"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import vumps
from tenpy.networks import mps
import pytest
import numpy as np
from scipy import integrate


def e0_transverse_ising(g=0.5):
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
    eng = engine(psi, M, vumps_pars)
    E_vumps, psi = eng.run()
    eng.options.touch('mixer_params')  # not used if no mixer
    eng.options.touch('trunc_params')
    eng.options['trunc_params'].touch('svd_min', 'chi_max')  # not used for single-site VUMPS

    # compare exact solution for transverse field Ising model
    Eexact = e0_transverse_ising(g)
    print(f"E_DMRG={E_vumps:.12f} vs E_exact={Eexact:.12f}")
    print(f"relative energy error: {abs((E_vumps - Eexact) / Eexact):.2e}")
    print("norm err:", psi.norm_test())
    E_vumps2 = np.mean(psi.expectation_value(M.H_bond))
    E_vumps3 = M.H_MPO.expectation_value(psi)
    assert abs((E_vumps - Eexact) / Eexact) < 1.e-10
    assert abs((E_vumps - E_vumps2) / E_vumps2) < max(1.e-10, np.max(psi.norm_test()))
    assert abs((E_vumps - E_vumps3) / E_vumps3) < max(1.e-10, np.max(psi.norm_test()))
