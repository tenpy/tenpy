"""A collection of tests to check the functionality of algorithms.exact_diagonalization"""
from __future__ import division

import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms.exact_diag import ExactDiag


def test_ED():
    # just quickly check that it runs without errors for a small system
    xxz_pars = dict(L=4, Jxx=1., Jz=1., hz=0.0, bc_MPS='finite')
    M = XXZChain(xxz_pars)
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    H, ED.full_H = ED.full_H, None
    ED.build_full_H_from_bonds()
    H2 = ED.full_H
    assert(npc.norm(H-H2, np.inf) < 1.e-14)
    ED.full_diagonalization()
    psi = ED.groundstate()
    print "select charge_sector =", psi.qtotal
    ED2 = ExactDiag(M, psi.qtotal)
    ED2.build_full_H_from_mpo()
    ED2.full_diagonalization()
    psi2 = ED2.groundstate()
    full_psi2 = psi.zeros_like()
    full_psi2[ED2._mask] = psi2
    ov = npc.inner(psi, full_psi2, do_conj=True)
    print "overlab <psi | psi2> = 1. -", 1.-ov
    assert(abs(abs(ov)-1) < 1.e-15)
