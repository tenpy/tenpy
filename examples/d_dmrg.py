"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `toycodes/d_dmrg.py`,
but make use of the classes defined in tenpy.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg


def example_DMRG_tf_ising_finite(L, g):
    print("finite DMRG, transverse field Ising model")
    print(f"L={L:d}, g={g:.2f}")
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(
        M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'combine': True
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    print(f"E = {E:.13f}")
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print(f"magnetization in X = {mag_x:.5f}")
    print(f"magnetization in Z = {mag_z:.5f}")
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print(f"Exact diagonalization: E = {E_exact:.13f}")
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_1site_DMRG_tf_ising_finite(L, g):
    print("single-site finite DMRG, transverse field Ising model")
    print(f"L={L:d}, g={g:.2f}")
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(
        M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'combine': False,
        'active_sites': 1  # specifies single-site
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print(f"E = {E:.13f}")
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print(f"magnetization in X = {mag_x:.5f}")
    print(f"magnetization in Z = {mag_z:.5f}")
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print(f"Exact diagonalization: E = {E_exact:.13f}")
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_DMRG_tf_ising_infinite(g):
    print("infinite DMRG, transverse field Ising model")
    print(f"g={g:.2f}")
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(
        M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
    }
    # Sometimes, we want to call a 'DMRG engine' explicitly
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    print(f"E = {E:.13f}")
    print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    print(f"<sigma_x> = {mag_x:.5f}")
    print(f"<sigma_z> = {mag_z:.5f}")
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print(f"Analytic result: E (per site) = {E_exact:.13f}")
    print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_1site_DMRG_tf_ising_infinite(g):
    print("single-site infinite DMRG, transverse field Ising model")
    print(f"g={g:.2f}")
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(
        M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'combine': True
    }
    eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    print(f"E = {E:.13f}")
    print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    print(f"<sigma_x> = {mag_x:.5f}")
    print(f"<sigma_z> = {mag_z:.5f}")
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print(f"Analytic result: E (per site) = {E_exact:.13f}")
    print("relative error: ", abs((E - E_exact) / E_exact))


def example_DMRG_heisenberg_xxz_infinite(Jz, conserve='best'):
    print("infinite DMRG, Heisenberg XXZ chain")
    print(f"Jz={Jz:.2f}, conserve={conserve!r}")
    model_params = dict(
        L=2,
        S=0.5,  # spin 1/2
        Jx=1.,
        Jy=1.,
        Jz=Jz,  # couplings
        bc_MPS='infinite',
        conserve=conserve)
    M = SpinModel(model_params)
    product_state = ["up", "down"]  # initial Neel state
    psi = MPS.from_product_state(
        M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS, unit_cell_width=M.lat.mps_unit_cell_width
    )
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print(f"E = {E:.13f}")
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print(f"<S_z> = [{Sz[0]:.5f}, {Sz[1]:.5f}]; mean ={mag_z:.5f}")
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    print("correlation length:", psi.correlation_length())
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)
    return E, psi, M


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    example_DMRG_tf_ising_finite(L=10, g=1.)
    print("-" * 100)
    example_1site_DMRG_tf_ising_finite(L=10, g=1.)
    print("-" * 100)
    example_DMRG_tf_ising_infinite(g=1.5)
    print("-" * 100)
    example_1site_DMRG_tf_ising_infinite(g=1.5)
    print("-" * 100)
    example_DMRG_heisenberg_xxz_infinite(Jz=1.5)
