"""A collection of tests to check the functionality of `tenpy.dmrg`"""
from __future__ import division

import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg


def test_dmrg():
    pars = dict(L=4, Jxx=1., Jz=1.1, hz=0.1, bc_MPS='finite')
    L = pars['L']
    M = XXZChain(pars)
    state = ([0, 1]*L)[:L]  # Neel
    psi = mps.MPS.from_product_state([site_spin_half]*L, state, bc='finite')
    dmrg.run(M, psi)


