"""Simple example to find the ground state of the XXZ model with DMRG

.. todo :
    document...
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import time

from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import run as run_DMRG


def run_groundstate_xxz(L=30, Jz=1., hz=0., conserve='best', chi_max=50, Jz_init=None):
    model_params = dict(
        L=L, Jx=1., Jy=1., Jz=Jz, hz=hz, bc_MPS='finite', conserve='best', verbose=1)
    result = {}
    M = SpinChain(model_params)
    result['model'] = 'SpinChain'
    result['model_params'] = model_params

    result['sites'] = np.arange(L)
    result['bonds'] = np.arange(L - 1) + 0.5

    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L])
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'mixer': True,
        'verbose': 1
    }
    if Jz_init:
        # run first time with small hx to break the symmetry
        model_params_Jz = model_params.copy()
        model_params_Jz['Jz'] = Jz_init
        M_Jz = SpinChain(model_params_Jz)
        dmrg_params['start_env'] = 10,
        run_DMRG(psi, M_Jz, dmrg_params)

    # run simulation
    t0 = time.time()

    info = run_DMRG(psi, M, dmrg_params)
    print("DMRG finished after", time.time() - t0, "s")

    # save results in output file
    result['chi'] = np.array(psi.chi)
    result['S'] = np.array(psi.entanglement_entropy())
    result['E'] = info['E']
    result['sweeps'] = info['sweep_statistics']['sweep']
    for key in ['E', 'S', 'max_trunc_err', 'max_E_trunc', 'max_chi']:
        result_key = 'sweep_' + key
        result[result_key] = np.array(info['sweep_statistics'][key])
    return result


if __name__ == "__main__":
    data = run_groundstate_xxz()
    # here you can do what you need to do with the data - save it, plot it, ...
