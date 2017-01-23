"""A collection of tests to check the functionality of `tenpy.dmrg`"""
from __future__ import division

import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg
from tenpy.networks import mps

import numpy as np


def test_dmrg():
    xxz_pars = dict(L=4, Jxx=1., Jz=1.1, hz=0.1, bc_MPS='finite')
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1]*L)[:L]  # Neel
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc='finite')
    dmrg_pars = {}
    dmrg.run(psi, M, dmrg_pars)
    # TODO
