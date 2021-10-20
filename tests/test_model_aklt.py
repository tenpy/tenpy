# Copyright 2021 TeNPy Developers, GNU GPLv3
from tenpy.models import aklt
from test_model import check_general_model
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.networks.mps import MPS


def test_AKLT(L=4):
    check_general_model(aklt.AKLTChain, {'L': L}, {})
    M = aklt.AKLTChain({'L': L})
    psi = MPS.from_lat_product_state(M.lat, [['up'], ['down']])
    eng = TwoSiteDMRGEngine(psi, M, {'trunc_params': {'svd_min': 1.e-6}})
    E0, psi0 = eng.run()
    assert abs(E0 - (-2 / 3.) * (L - 1)) < 1.e-10
    # note: if we view the system as spin-1/2 projected to spin-1, the first and last spin
    # are arbitrary.
    # if they are in a superposition, this will contribute at most another factor of 2 to chi.
    assert all([chi <= 4 for chi in psi0.chi])
