"""Example illustrating the use of TDVP in tenpy.

As of now, we have TDVP only for finite systems. The call structure is quite similar to TEBD. A
difference is that we can run one-site TDVP or two-site TDVP. In the former, the bond dimension can
not grow; the latter allows to grow the bond dimension and hence requires a truncation.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3
import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy
from tenpy.networks import mps
from tenpy.networks import site
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS


def example_TDVP():
    L = 14
    chi = 20  # exemplary strong truncation!
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
    product_state = ["up", "down"] * (L // 2)
    # starting from a Neel product state which is not an eigenstate of the Heisenberg model
    psi = MPS.from_product_state(heisenberg.lat.mps_sites(),
                                 product_state,
                                 bc=heisenberg.lat.bc_MPS,
                                 form='B')

    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'N_steps': 1,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }
    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, heisenberg, tdvp_params)
    times = []
    S_mid = []
    Es = []
    def measure():
        times.append(tdvp_engine.evolved_time)
        S_mid.append(psi.entanglement_entropy(bonds=[L // 2])[0])
        Es.append(heisenberg.H_MPO.expectation_value(psi))

    measure()
    for i in range(30):
        tdvp_engine.run()
        measure()

    tdvp_engine = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine)
    for i in range(30):
        tdvp_engine.run()
        measure()

    return times, S_mid, Es


def plot_example_TDVP(times, S_mid, Es):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7,7))
    axes[0].plot(times, S_mid)
    axes[0].set_ylabel('entroy $S$ at center bond')
    axes[1].plot(times, np.array(Es) - Es[0])
    axes[1].set_ylabel('energy $E - E(t=0)$')
    axes[1].set_xlabel('time $t$')
    for ax in axes:
        ax.axvline(x=3.01, color='red')
    axes[0].text(2.9, 0.0000015, "Two-site update", ha='right')
    axes[0].text(3.1, 0.0000015, "One-site update, strict TDVP")
    plt.show()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    measure = example_TDVP()
    plot_example_TDVP(*measure)
