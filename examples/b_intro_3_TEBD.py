"""Example illustrating the use of TEBD in tenpy.

The example functions in this class do the same as the ones in `a_simple_3_TEBD.py`,
but make use of the classes defined in tenpy.
"""

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd
from a_simple_2_model import TFIModel as SimpleTFIModel


def example_TEBD_gs_finite(L, g):
    print("finite TEBD, imaginary time evolution, L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0]*L, bc='finite')
    tebd_params = {'order': 2, 'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5], 'N_steps': 10,
                   'max_error_E': 1.e-6, 'trunc_params': {'chi_max': 30, 'svd_min': 1.e-10},
                   'verbose': 1}
    eng = tebd.Engine(psi, M, tebd_params)
    eng.run_GS()  # the main work...
    E = np.sum(psi.expectation_value(M.H_bond[1:]))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    if L < 20:
        M2 = SimpleTFIModel(L=L, J=1., g=g, bc='finite')
        E_ed = M2.exact_finite_gs_energy()
        print("Exact diagonalization: E = {E:.13f}".format(E=E_ed))
        print("relative error: ", abs((E - E_ed) / E_ed))
    return E, psi, M


def example_TEBD_gs_infinite(g):
    print("infinite TEBD, imaginary time evolution, g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0]*2, bc='infinite')
    tebd_params = {'order': 2, 'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5], 'N_steps': 10,
                   'max_error_E': 1.e-8, 'trunc_params': {'chi_max': 30, 'svd_min': 1.e-10},
                   'verbose': 1}
    eng = tebd.Engine(psi, M, tebd_params)
    eng.run_GS()  # the main work...
    E = np.mean(psi.expectation_value(M.H_bond))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    print("correlation length:", psi.correlation_length())
    M2 = SimpleTFIModel(L=2, J=1., g=g, bc='infinite')
    E_ex = M2.exact_infinite_gs_energy()
    print("Analytic result: E/L = {E:.13f}".format(E=E_ex))
    print("relative error: ", abs((E - E_ex) / E_ex))
    return E, psi, M


def example_TEBD_lightcone(L, g, tmax, dt):
    print("finite TEBD, real time evolution, L={L:d}, g={g:.2f}".format(L=L, g=g))
    # find ground state with TEBD or DMRG
    #  E, psi, M = example_TEBD_gs_finite(L, g)
    from b_intro_4_DMRG import example_DMRG_finite
    E, psi, M = example_DMRG_finite(L, g)
    i0 = L // 2
    # apply sigmaz on site i0
    psi.apply_local_op(i0, 'Sigmaz', unitary=True)
    dt_measure = 0.05
    # tebd.Engine makes 'N_steps' steps of `dt` at once; for second order this is more efficient.
    tebd_params = {'order': 2, 'dt': dt, 'N_steps': int(dt_measure//dt),
                   'trunc_params': {'chi_max': 50, 'svd_min': 1.e-10, 'trunc_cut': None}}
    eng = tebd.Engine(psi, M, tebd_params)
    S = [psi.entanglement_entropy()]
    for n in range(int(tmax / dt_measure + 0.5)):
        eng.run()
        S.append(psi.entanglement_entropy())
    import pylab as pl
    pl.figure()
    pl.imshow(S[::-1], vmin=0., aspect='auto', interpolation='nearest',
              extent=(0, L - 1., -0.5*dt_measure, eng.evolved_time+0.5*dt_measure))  # yapf:disable
    pl.xlabel('site $i$')
    pl.ylabel('time $t/J$')
    pl.ylim(0., tmax)
    pl.colorbar().set_label('entropy $S$')
    pl.show()


if __name__ == "__main__":
    example_TEBD_gs_finite(L=10, g=1.)
    print("-" * 100)
    example_TEBD_gs_infinite(g=1.5)
    print("-" * 100)
    example_TEBD_lightcone(L=20, g=1.5, tmax=3., dt=0.01)
