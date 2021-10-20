"""A collection of tests to check the functionality of algorithms.exact_diagonalization."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.linalg.lanczos import lanczos


def test_ED():
    # just quickly check that it runs without errors for a small system
    xxz_pars = dict(L=4, Jxx=1., Jz=1., hz=0.0, bc_MPS='finite')
    M = XXZChain(xxz_pars)
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    H, ED.full_H = ED.full_H, None
    ED.build_full_H_from_bonds()
    H2 = ED.full_H
    assert (npc.norm(H - H2, np.inf) < 1.e-14)
    ED.full_diagonalization()
    E, psi = ED.groundstate()
    print("select charge_sector =", psi.qtotal)
    assert np.all(psi.qtotal == [0])
    E_sec2, psi_sec2 = ED.groundstate([2])
    assert np.all(psi_sec2.qtotal == [2])
    ED2 = ExactDiag(M, psi.qtotal)
    ED2.build_full_H_from_mpo()
    ED2.full_diagonalization()
    E2, psi2 = ED2.groundstate()
    full_psi2 = psi.zeros_like()
    full_psi2[ED2._mask] = psi2
    ov = npc.inner(psi, full_psi2, 'range', do_conj=True)
    print("overlab <psi | psi2> = 1. -", 1. - ov)
    assert (abs(abs(ov) - 1.) < 1.e-15)
    # starting from a random guess in the correct charge sector,
    # check if we can also do lanczos.
    np.random.seed(12345)
    psi3 = npc.Array.from_func(np.random.random, psi2.legs, qtotal=psi2.qtotal, shape_kw='size')
    E0, psi3, N = lanczos(ED2, psi3)
    print("Lanczos E0 =", E0)
    ov = npc.inner(psi3, psi2, 'range', do_conj=True)
    print("overlab <psi2 | psi3> = 1. -", 1. - ov)
    assert (abs(abs(ov) - 1.) < 1.e-15)

    ED3 = ExactDiag.from_H_mpo(M.H_MPO)
    ED3.build_full_H_from_mpo()
    assert npc.norm(ED3.full_H - H, np.inf) < 1.e-14
