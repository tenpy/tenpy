"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `a_simple_4_DMRG.py`,
but make use of the classes defined in tenpy.
"""

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg
from a_simple_2_model import TFIModel as SimpleTFIModel


def example_DMRG_finite(L, g):
    print("finite DMRG, L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0] * L, bc='finite')
    dmrg_params = {'mixer': None, 'trunc_params': {'chi_max': 30, 'svd_min': 1.e-10}, 'verbose': 1}
    dmrg.run(psi, M, dmrg_params)
    E = np.sum(psi.expectation_value(M.H_bond[1:]))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    if L < 20:
        M2 = SimpleTFIModel(L=L, J=1., g=g, bc='finite')
        E_ed = M2.exact_finite_gs_energy()
        print("Exact diagonalization: E = {E:.13f}".format(E=E_ed))
        print("relative error: ", abs((E - E_ed) / E_ed))
    return E, psi, M


def example_DMRG_infinite(g):
    print("infinite DMRG, g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=0)
    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0] * 2, bc='infinite')
    dmrg_params = {'mixer': None, 'trunc_params': {'chi_max': 30, 'svd_min': 1.e-10}, 'verbose': 1}
    dmrg.run(psi, M, dmrg_params)
    E = np.mean(psi.expectation_value(M.H_bond))
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    print("correlation length:", psi.correlation_length())
    M2 = SimpleTFIModel(L=2, J=1., g=g, bc='infinite')
    E_ex = M2.exact_infinite_gs_energy()
    print("Analytic result: E/L = {E:.13f}".format(E=E_ex))
    print("relative error: ", abs((E - E_ex) / E_ex))
    return E, psi, M


if __name__ == "__main__":
    example_DMRG_finite(L=10, g=1.)
    print("-" * 100)
    example_DMRG_infinite(g=1.5)
