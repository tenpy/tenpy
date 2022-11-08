# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from tenpy.models.toric_code import ToricCode
from test_model import check_general_model
import pytest
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import numpy as np
import warnings


@pytest.mark.slow
def test_ToricCode_general():
    check_general_model(ToricCode, dict(Lx=2, Ly=3, bc_MPS='infinite', sort_charge=True), {
        'conserve': [None, 'parity'],
    })


@pytest.mark.slow
def test_ToricCode(Lx=1, Ly=2):
    model_params = {'Lx': Lx, 'Ly': Ly, 'bc_MPS': 'infinite', 'sort_charge': True}
    M = ToricCode(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0] * M.lat.N_sites, bc='infinite')
    dmrg_params = {
        'mixer': True,
        'trunc_params': {
            'chi_max': 10,
            'svd_min': 1.e-10
        },
        'P_tol_to_trunc': None,  # avoid warning about unused "P_tol" lanczos
        'max_E_err': 1.e-10,
        'N_sweeps_check': 4,
    }
    result = dmrg.run(psi, M, dmrg_params)
    E = result['E']
    print("E =", E)
    psi.canonical_form()
    # energy per "cell"=2 -> energy per site in the dual lattice = 1
    assert abs(E - (-1.)) < dmrg_params['max_E_err']
    print("chi=", psi.chi)
    if Ly == 2:
        assert tuple(psi.chi[:4]) == (2, 2, 2, 2)
    assert abs(psi.entanglement_entropy(bonds=[0])[0] - np.log(2) * (Ly - 1)) < 1.e-5
