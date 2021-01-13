"""Call of (infinite) TEBD."""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd

M = TFIChain({"L": 2, "J": 1., "g": 1.5, "bc_MPS": "infinite"})
psi = MPS.from_product_state(M.lat.mps_sites(), [0] * 2, "infinite")
tebd_params = {
    "order": 2,
    "delta_tau_list": [0.1, 0.001, 1.e-5],
    "max_error_E": 1.e-6,
    "trunc_params": {
        "chi_max": 30,
        "svd_min": 1.e-10
    }
}
eng = tebd.TEBDEngine(psi, M, tebd_params)
eng.run_GS()  # imaginary time evolution with TEBD
print("E =", sum(psi.expectation_value(M.H_bond)) / psi.L)
print("final bond dimensions: ", psi.chi)
