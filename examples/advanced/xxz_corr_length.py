"""Calculate the correleation legnth of the transferse field Ising model for various h_z.

This example uses DMRG to find the ground state of the transverse field Ising model when tuning
through the phase transition by changing the field `hz`. It uses
:meth:`~tenpy.networks.mps.MPS.correlation_length` to extract the correlation length of the ground
state, and plots it vs. hz in the end.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt


def run(Jzs):
    L = 2
    model_params = dict(L=L, Jx=1., Jy=1., Jz=Jzs[0], bc_MPS='infinite', conserve='Sz')
    chi = 300
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'update_env': 20,
        'start_env': 20,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'mixer': False
    }

    M = SpinChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L], M.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    np.set_printoptions(linewidth=120)
    corr_length = []
    for Jz in Jzs:
        print("-" * 80)
        print("Jz = {Jz:.4f}".format(Jz))
        print("-" * 80)
        model_params['Jz'] = Jz
        M = SpinChain(model_params)
        engine.init_env(model=M)  # (re)initialize DMRG environment with new model
        # this uses the result from the previous DMRG as first initial guess
        engine.run()
        # psi is modified by engine.run() and now represents the ground state for the current `Jz`.
        corr_length.append(psi.correlation_length(tol_ev0=1.e-3))
        print("corr. length", corr_length[-1])
        print("<Sz>", psi.expectation_value('Sz'))
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
    corr_length = np.array(corr_length)
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'Jzs': Jzs,
        'corr_length': corr_length,
        'eval_transfermatrix': np.exp(-1. / corr_length)
    }
    return results


def plot(results, filename):
    corr_length = results['corr_length']
    Jzs = results['Jzs']
    plt.plot(Jzs, np.exp(-1. / corr_length))
    plt.xlabel(r'$J_z/J_x$')
    plt.ylabel(r'$t = \exp(-\frac{1}{\xi})$')
    plt.savefig(filename)
    print("saved to " + filename)


if __name__ == "__main__":
    filename = 'xxz_corrlength.pkl'
    import pickle
    import os.path
    if not os.path.exists(filename):
        results = run(list(np.arange(4.0, 1.5, -0.25)) + list(np.arange(1.5, 0.8, -0.05)))
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    else:
        print("just load the data")
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    plot(results, filename[:-4] + '.pdf')
