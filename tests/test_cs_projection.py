"""A test for tenpy.networks.mps._project_onto_sector_from_charge_tree"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import pytest
#from tenpy.networks.site import SpinHalfSite
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS


def test_cs_projection():


    spin_site = SpinSite(S=0.5, conserve=None)
    product_state_list = [[1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]]
    MPS_from_prodstate = MPS.from_product_state([spin_site] * 3, product_state_list)

    sites = MPS_from_prodstate.sites

    new_sites = []
    for site in sites:
        site_new = site.__class__(S=site.S, conserve='Sz')
        new_sites.append(site_new)

    tree = MPS.get_charge_tree_for_given_charge_sector(new_sites, charge_sector=(1,))
    res = MPS._project_onto_sector_from_charge_tree(new_sites,product_state_list, tree)
    expval=res.expectation_value('Sz').sum()

    assert np.abs(expval -1/2)< 10**14

    tree = MPS.get_charge_tree_for_given_charge_sector(new_sites, charge_sector=(-1,))
    res = MPS._project_onto_sector_from_charge_tree(new_sites,product_state_list, tree)
    expval=res.expectation_value('Sz').sum()

    assert np.abs(expval +1/2)< 10**14



    spin_site = SpinSite(S=1, conserve=None)
    product_state_list = [[1/np.sqrt(3),1/np.sqrt(3), 1/np.sqrt(3)], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [1/np.sqrt(3),1/np.sqrt(3), 1/np.sqrt(3)]]
    MPS_from_prodstate = MPS.from_product_state([spin_site] * 3, product_state_list)

    sites = MPS_from_prodstate.sites

    new_sites = []
    for site in sites:
        site_new = site.__class__(S=site.S, conserve='Sz')
        new_sites.append(site_new)

    tree = MPS.get_charge_tree_for_given_charge_sector(new_sites, charge_sector=(2,))
    res = MPS._project_onto_sector_from_charge_tree(new_sites, product_state_list, tree)
    expval = res.expectation_value('Sz').sum()

    assert np.abs(expval -1) < 10 ** 14


    tree = MPS.get_charge_tree_for_given_charge_sector(new_sites, charge_sector=(-2,))
    res = MPS._project_onto_sector_from_charge_tree(new_sites, product_state_list, tree)
    expval = res.expectation_value('Sz').sum()

    assert np.abs(expval +1) < 10 ** 14





