"""A test for :meth:`~tenpy.networks.mps.MPS.project_onto_charge_sector`"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
from tenpy.networks.site import SpinSite, FermionSite
from tenpy.networks.mps import MPS


def test_cs_projection():
    # define a product state (as used in MPS.from_product_state)
    product_state_list = np.array([[1, 1],
                                   [-1, 1],
                                   [1, 1]]) / np.sqrt(2)

    # define a list of sites (with conservation of the desired charge)
    sites = [SpinSite(S=0.5, conserve='Sz')]*len(product_state_list) 
    
    # :meth:`MPS.project_onto_charge_sector` calls
    # :meth:`MPS.get_charge_tree_for_given_charge_sector` and 
    # :meth:`MPS._project_onto_sector_from_charge_tree`
    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(1,))
    
    expval = mps_projected.expectation_value('Sz').sum()
    # check that we are in the given charge-sector (note that ints are used for half-integer spins)
    assert np.isclose(expval, 0.5)

    # test for different charge sector
    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(-1,))
    expval = mps_projected.expectation_value('Sz').sum()

    assert np.isclose(expval, -0.5)

    # test for spin 1
    product_state_list = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]])/np.sqrt(3)

    sites = [SpinSite(S=1, conserve='Sz')]*len(product_state_list)
 
    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(2,))
    expval = mps_projected.expectation_value('Sz').sum()

    assert np.isclose(expval, 1)

    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(-2,))
    expval = mps_projected.expectation_value('Sz').sum()

    assert np.isclose(expval, -1)

    # test conservation of ``'N'`` for fermions
    product_state_list = np.array([[1, 1, 1, 1], 
                                   [1, 1, 1, 1],
                                   [1, 1, 1, 1]]) / 2

    sites = [FermionSite()]*len(product_state_list)
    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(2,))

    assert np.isclose(mps_projected.expectation_value('N').sum(), 2)

    mps_projected = MPS.project_onto_charge_sector(sites, product_state_list, charge_sector=(0,))

    assert np.isclose(mps_projected.expectation_value('N').sum(), 0)
