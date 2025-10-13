"""Excitations of the transverse field Ising model.

This example uses VUMPS to find the ground state of the transverse field Ising model and uses the plane wave excitation ansatz to compute the first excited states.
"""
# Copyright (C) TeNPy Developers, Apache license

from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import vumps, plane_wave_excitation
from tenpy.networks import mps
import numpy as np


def tfi_vumps(g=1.5):
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve='parity')
    M = TFIChain(model_params)
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), [0, 0], bc='infinite')
    vumps_pars = {
        'combine': False,
        'N_sweeps_check': 1,
        'mixer': False,
        'N_sweeps_check': 1,
        'trunc_params': {
            'chi_max': 32,
            'svd_min': 1.e-14,
        },
        'min_sweeps': 30,
        'max_sweeps': 50,
        'max_split_err': 1e-8,  # different criteria than DMRG
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-8,
    }
   
    
    eng = vumps.TwoSiteVUMPSEngine(psi, M, vumps_pars)
    E, psi = eng.run()
    print(f"ground state energy: {E:.5f}")

    # to directly continue with plane wave excitations, return the uniform MPS
    uniform_psi = eng.psi

    return E, uniform_psi, M

def tfi_excitations(psi_gs, M):
    pw_params = {
        'lanczos_params': {
            'N_max': 50,
        },
    }
    eng_pw = plane_wave_excitation.PlaneWaveExcitationEngine(psi_gs, M, pw_params)

    momenta = np.arange(0, np.pi, np.pi/8)  # compute for some momenta
    qtotal_change = [1]  # look for excitations in other parity sector
    num_ev = 1  # we only compute the lowest dispersion mode

    dispersions = []
    for p in momenta:
        Es, _, _ = eng_pw.run(p, qtotal_change, num_ev=num_ev)
        dispersions.append(Es)
        print(f"excitation energy for momentum {p/np.pi:.2f} Pi: {Es[0]:.5f}")
    return momenta, np.array(dispersions)

def tfi_dispersion(k, g):
   # exact dispersion for two site unit cell
   return np.min([2*np.sqrt(g**2-2*g*np.cos(k)+1), 2*np.sqrt(g**2-2*g*np.cos(k+np.pi)+1)], axis=0)


if __name__ == "__main__":
    E, psi, M = tfi_vumps()
    momenta, dispersions = tfi_excitations(psi, M)

    # plot and compare to exact results
    import matplotlib.pyplot as plt 

    plt.plot(momenta, dispersions, 'x', label='plane wave ansatz')
    plt.plot(np.arange(0, np.pi, 0.1), tfi_dispersion(np.arange(0, np.pi, 0.1), 1.5), ':', label='exact', c='black')
    plt.legend()
    plt.xlabel('momentum')
    plt.ylabel('excitation energy')
    plt.show()
