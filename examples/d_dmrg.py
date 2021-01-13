"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `toycodes/d_dmrg.py`,
but make use of the classes defined in tenpy.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg


def example_DMRG_tf_ising_finite(L, g, verbose=True):
    print("finite DMRG, transverse field Ising model")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=verbose)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
        'combine': True
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_1site_DMRG_tf_ising_finite(L, g, verbose=True):
    print("single-site finite DMRG, transverse field Ising model")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=verbose)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
        'combine': False,
        'active_sites': 1  # specifies single-site
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_DMRG_tf_ising_infinite(g, verbose=True):
    print("infinite DMRG, transverse field Ising model")
    print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=verbose)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
    }
    # Sometimes, we want to call a 'DMRG engine' explicitly
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_1site_DMRG_tf_ising_infinite(g, verbose=True):
    print("single-site infinite DMRG, transverse field Ising model")
    print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=verbose)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
        'combine': True
    }
    eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    print("relative error: ", abs((E - E_exact) / E_exact))


def example_DMRG_heisenberg_xxz_infinite(Jz, conserve='best', verbose=True):
    print("infinite DMRG, Heisenberg XXZ chain")
    print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
    model_params = dict(
        L=2,
        S=0.5,  # spin 1/2
        Jx=1.,
        Jy=1.,
        Jz=Jz,  # couplings
        bc_MPS='infinite',
        conserve=conserve,
        verbose=verbose)
    M = SpinModel(model_params)
    product_state = ["up", "down"]  # initial Neel state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                     Sz1=Sz[1],
                                                                     mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    print("correlation length:", psi.correlation_length())
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)
    return E, psi, M


if __name__ == "__main__":
    example_DMRG_tf_ising_finite(L=10, g=1., verbose=True)
    print("-" * 100)
    example_1site_DMRG_tf_ising_finite(L=10, g=1., verbose=True)
    print("-" * 100)
    example_DMRG_tf_ising_infinite(g=1.5, verbose=True)
    print("-" * 100)
    example_1site_DMRG_tf_ising_infinite(g=1.5, verbose=True)
    print("-" * 100)
    example_DMRG_heisenberg_xxz_infinite(Jz=1.5)
