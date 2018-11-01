"""Example illustrating the use of TEBD in tenpy.

The example functions in this class do the same as the ones in `toycodes/c_tebd.py`,
but make use of the classes defined in tenpy.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
import tfi_exact


def example_TEBD_gs_finite(L, g):
    print("finite TEBD, imaginary time evolution, L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    tebd_params = {
        'order': 2,
        'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5],
        'N_steps': 10,
        'max_error_E': 1.e-6,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': 1
    }
    eng = tebd.Engine(psi, M, tebd_params)
    eng.run_GS()  # the main work...
    E = np.sum(psi.expectation_value(M.H_bond[1:]))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    if L < 20:  # compare to exact result
        E_exact = tfi_exact.finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_TEBD_gs_infinite(g):
    print("infinite TEBD, imaginary time evolution, g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    tebd_params = {
        'order': 2,
        'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5],
        'N_steps': 10,
        'max_error_E': 1.e-8,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': 1
    }
    eng = tebd.Engine(psi, M, tebd_params)
    eng.run_GS()  # the main work...
    E = np.mean(psi.expectation_value(M.H_bond))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    E_exact = tfi_exact.infinite_gs_energy(1., g)
    print("Analytic result: E/L = {E:.13f}".format(E=E_exact))
    print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_TEBD_lightcone(L, g, tmax, dt):
    print("finite TEBD, real time evolution, L={L:d}, g={g:.2f}".format(L=L, g=g))
    # find ground state with TEBD or DMRG
    #  E, psi, M = example_TEBD_gs_finite(L, g)
    from d_dmrg import example_DMRG_finite
    E, psi, M = example_DMRG_finite(L, g)
    i0 = L // 2
    # apply sigmaz on site i0
    psi.apply_local_op(i0, 'Sigmaz', unitary=True)
    dt_measure = 0.05
    # tebd.Engine makes 'N_steps' steps of `dt` at once; for second order this is more efficient.
    tebd_params = {
        'order': 2,
        'dt': dt,
        'N_steps': int(dt_measure // dt),
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }
    eng = tebd.Engine(psi, M, tebd_params)
    S = [psi.entanglement_entropy()]
    for n in range(int(tmax / dt_measure + 0.5)):
        eng.run()
        S.append(psi.entanglement_entropy())
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(
        S[::-1],
        vmin=0.,
        aspect='auto',
        interpolation='nearest',
        extent=(0, L - 1., -0.5 * dt_measure, eng.evolved_time + 0.5 * dt_measure))
    plt.xlabel('site $i$')
    plt.ylabel('time $t/J$')
    plt.ylim(0., tmax)
    plt.colorbar().set_label('entropy $S$')
    plt.savefig('c_tebd_lightcone.pdf')


if __name__ == "__main__":
    example_TEBD_gs_finite(L=10, g=1.)
    print("-" * 100)
    example_TEBD_gs_infinite(g=1.5)
    print("-" * 100)
    example_TEBD_lightcone(L=20, g=1.5, tmax=3., dt=0.01)
