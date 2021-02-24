"""Example illustrating the use of TDVP in tenpy.

As of now, we have TDVP only for finite systems. The call structure is quite similar to TEBD. A
difference is that we can run one-site TDVP or two-site TDVP. In the former, the bond dimension can
not grow; the latter allows to grow the bond dimension and hence requires a truncation.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3
import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy.models.spins
import tenpy.networks.mps as mps
import tenpy.networks.site as site
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
import copy


def run_out_of_equilibrium():
    L = 10
    chi = 5
    delta_t = 0.1
    model_params = {
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

    heisenberg = tenpy.models.spins.SpinChain(model_params)
    product_state = ["up"] * (L // 2) + ["down"] * (L - L // 2)
    # starting from a domain-wall product state which is not an eigenstate of the Heisenberg model
    psi = MPS.from_product_state(heisenberg.lat.mps_sites(),
                                 product_state,
                                 bc=heisenberg.lat.bc_MPS,
                                 form='B')

    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }
    tdvp_engine = tdvp.TDVPEngine(psi, heisenberg, tdvp_params)
    times = []
    S_mid = []
    for i in range(30):
        tdvp_engine.run_two_sites(N_steps=1)
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L // 2])[0])
    for i in range(30):
        tdvp_engine.run_one_site(N_steps=1)
        #psi_2=copy.deepcopy(psi)
        #psi_2.canonical_form()
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L // 2])[0])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(times, S_mid)
    plt.xlabel('t')
    plt.ylabel('S')
    plt.axvline(x=3.1, color='red')
    plt.text(0.0, 0.0000015, "Two sites update")
    plt.text(3.1, 0.0000015, "One site update")
    plt.show()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_out_of_equilibrium()
