# Copyright 2021 TeNPy Developers, GNU GPLv3
from tenpy.models import aklt
from test_model import check_general_model
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.networks.mps import MPS
import pytest


def test_AKLT_finite():
    L = 4
    check_general_model(aklt.AKLTChain, {'L': L}, {})
    M = aklt.AKLTChain({'L': L, 'bc_MPS': 'finite'})
    psi = MPS.from_lat_product_state(M.lat, [['up'], ['down']])
    eng = TwoSiteDMRGEngine(psi, M, {'trunc_params': {'svd_min': 1.e-10, 'chi_max': 10}})
    E0, psi0 = eng.run()
    assert abs(E0 - (-2 / 3.) * (L - 1)) < 1.e-10
    # note: if we view the system as spin-1/2 projected to spin-1, the first and last spin
    # are arbitrary.
    # if they are in a superposition, this will contribute at most another factor of 2 to chi.
    assert all([chi <= 4 for chi in psi0.chi])


@pytest.mark.slow
def test_AKLT_infinite():
    # switch to infinite
    L = 4
    M = aklt.AKLTChain({'L': L, 'bc_MPS': 'infinite'})
    psi = MPS.from_lat_product_state(M.lat, [['up'], ['down']])
    eng = TwoSiteDMRGEngine(psi, M, {'trunc_params': {'svd_min': 1.e-10, 'chi_max': 10},
                                     'N_sweeps_check': 1,
                                     'mixer': False})
    E0, psi0 = eng.run()
    assert abs(E0 - (-2 / 3.)) < 1.e-10
    psi_aklt = M.psi_AKLT()
    assert abs(1. - abs(psi0.overlap(psi_aklt, understood_infinite=True))) < 1.e-10
