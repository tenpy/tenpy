"""Example to extract the central charge from the entranglement scaling.

This example code evaluate the central charge of the transverse field Ising model using IDMRG.
The expected value for the central charge c = 1/2. The code always recycle the environment from
the previous simulation, which can be seen at the "age".

For the theoretical background why :math:`S = c/6 log(xi)`, see :cite:`pollmann2009`.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import tenpy
import time
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg


def example_DMRG_tf_ising_infinite_S_xi_scaling(g):
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve='best', verbose=0)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'start_env': 10,
        'mixer': False,
        #  'mixer_params': {'amplitude': 1.e-3, 'decay': 5., 'disable_after': 50},
        'trunc_params': {
            'chi_max': 5,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-6,
        'update_env': 0,
        'verbose': 0
    }

    chi_list = np.arange(7, 31, 2)
    s_list = []
    xi_list = []
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    for chi in chi_list:

        t0 = time.time()
        eng.reset_stats(
        )  # necessary if you for example have a fixed numer of sweeps, if you don't set this you option your simulation stops after initial number of sweeps!
        eng.trunc_params['chi_max'] = chi
        ##   DMRG Calculation    ##
        print("Start IDMRG CALCULATION")
        eng.run()
        eng.options['mixer'] = None
        psi.canonical_form()

        ##   Calculating bond entropy and correlation length  ##
        s_list.append(psi.entanglement_entropy()[0])
        # the bond 0 is between MPS unit cells and hence sensible even for 2D lattices.
        xi_list.append(psi.correlation_length())

        print(chi,
              time.time() - t0,
              np.mean(psi.expectation_value(M.H_bond)),
              s_list[-1],
              xi_list[-1],
              flush=True)
        tenpy.tools.optimization.optimize(3)  # quite some speedup for small chi

        print("SETTING NEW BOND DIMENSION")

    return s_list, xi_list


def fit_plot_central_charge(s_list, xi_list, filename):
    """Plot routine in order to determine the cental charge."""
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def fitFunc(Xi, c, a):
        return (c / 6) * np.log(Xi) + a

    Xi = np.array(xi_list)
    S = np.array(s_list)
    LXi = np.log(Xi)  # Logarithm of the correlation length xi

    fitParams, fitCovariances = curve_fit(fitFunc, Xi, S)

    # Plot fitting parameter and covariances
    print('c =', fitParams[0], 'a =', fitParams[1])
    print('Covariance Matrix', fitCovariances)

    # plot the data as blue circles
    plt.errorbar(LXi,
                 S,
                 fmt='o',
                 c='blue',
                 ms=5.5,
                 markerfacecolor='white',
                 markeredgecolor='blue',
                 markeredgewidth=1.4)
    # plot the fitted line
    plt.plot(LXi,
             fitFunc(Xi, fitParams[0], fitParams[1]),
             linewidth=1.5,
             c='black',
             label='fit c={c:.2f}'.format(c=fitParams[0]))

    plt.xlabel(r'$\log{\,}\xi_{\chi}$', fontsize=16)
    plt.ylabel(r'$S$', fontsize=16)
    plt.legend(loc='lower right', borderaxespad=0., fancybox=True, shadow=True, fontsize=16)
    plt.savefig(filename)


if __name__ == "__main__":
    s_list, xi_list = example_DMRG_tf_ising_infinite_S_xi_scaling(g=1)
    fit_plot_central_charge(s_list, xi_list, "central_charge_ising.pdf")
