"""A test for :meth:`~tenpy.networks.mps.MPS.project_onto_charge_sector`"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from tenpy.networks.site import SpinSite, FermionSite, SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.models.hubbard import FermiHubbardModel


def test_cs_projection():
    # Test for SpinHalfSite
    # define a product state (as used in MPS.from_product_state)
    product_state_list = np.array([[1, 1],
                                   [-1, 1],
                                   [1, 1]]) / np.sqrt(2)

    # define a list of sites (with conservation of the desired charge)
    sites = [SpinSite(S=0.5, conserve='Sz')]*len(product_state_list)

    charge_sectors = [(-1,), (1,), (-3,), (3,)]
    for charge_sector in charge_sectors:
        # :meth:`MPS.project_onto_charge_sector` calls
        # :meth:`MPS.get_charge_tree_for_given_charge_sector` and
        # :meth:`MPS._project_onto_sector_from_charge_tree`
        mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=charge_sector)
        expval_Sz = mps_projected.expectation_value('Sz').sum()
        assert np.isclose(expval_Sz, charge_sector[0] / 2)

    # test for SpinSite with S=1
    product_state_list = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]])

    sites = [SpinSite(S=1, conserve='Sz')]*len(product_state_list)

    # try different charge sectors
    charge_sectors = [(-6,), (-2,), (0,), (2,)]
    for charge_sector in charge_sectors:
        mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=charge_sector)
        expval_Sz = mps_projected.expectation_value('Sz').sum()
        assert np.isclose(expval_Sz, charge_sector[0]/2)

    # test conservation of ``'N'`` for FermionSite
    product_state_list = np.array([[1, 1],
                                   [1, 1],
                                   [1, 1]])

    sites = [FermionSite()]*len(product_state_list)
    # try different charge sectors
    charge_sectors = [(0,), (1,), (2,)]
    for charge_sector in charge_sectors:
        mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=charge_sector)
        expval_N = mps_projected.expectation_value('N').sum()

        assert np.isclose(expval_N, charge_sector[0])


def test_cs_projection_with_two_charges():
    # Test SpinHalfFermionSite (conserving N and Sz9
    product_state_list = np.array([[1, 1, 1, 1],
                                   [1, 1, 1, 1],
                                   [1, 1, 1, 1]])

    sites = [SpinHalfFermionSite()] * len(product_state_list)

    # try different charge sectors
    charge_sectors = [(2, 0), (4, -2), (2, 2), (6, 0), (0, 0)]
    for charge_sector in charge_sectors:
        mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=charge_sector)
        expval_Sz = mps_projected.expectation_value('Sz').sum()
        expval_N = mps_projected.expectation_value('Ntot').sum()
        assert np.isclose(expval_Sz, charge_sector[1]/2)
        assert np.isclose(expval_N, charge_sector[0])


def test_from_infiniteT_canonical():
    M = FermiHubbardModel(dict(L=10, cons_Sz='Sz', cons_N='N'))
    psi = PurificationMPS.from_infiniteT_canonical(M.lat.mps_sites(), (M.lat.N_sites, 0))
    assert np.isclose(psi.expectation_value('Ntot').sum(), M.lat.N_sites)
    assert np.isclose(psi.expectation_value('Sz').sum(), 0)
