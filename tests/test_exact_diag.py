"""A collection of tests to check the functionality of algorithms.exact_diagonalization."""
# Copyright (C) TeNPy Developers, Apache license

import pytest
import numpy as np
from functools import reduce
import copy

import tenpy.linalg.np_conserved as npc
from tenpy.networks import MPS, SpinHalfSite
from tenpy.models import TFIChain, XXZChain
from tenpy.algorithms import exact_diag
from tenpy.linalg.krylov_based import LanczosGroundState


def test_ED():
    # just quickly check that it runs without errors for a small system
    xxz_pars = dict(L=4, Jxx=1., Jz=1., hz=0.1, bc_MPS='finite', sort_charge=True)
    M = XXZChain(xxz_pars)
    ED = exact_diag.ExactDiag(M)
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
    ED2 = exact_diag.ExactDiag(M, psi.qtotal)
    ED2.build_full_H_from_mpo()
    ED2.full_diagonalization()
    E2, psi2 = ED2.groundstate()
    full_psi2 = psi.zeros_like()
    full_psi2[ED2._mask] = psi2
    ov = npc.inner(psi, full_psi2, 'range', do_conj=True)
    print("overlap <psi | psi2> = 1. -", 1. - ov)
    assert (abs(abs(ov) - 1.) < 1.e-15)
    # starting from a random guess in the correct charge sector,
    # check if we can also do lanczos.
    np.random.seed(12345)
    psi3 = npc.Array.from_func(np.random.random, psi2.legs, qtotal=psi2.qtotal, shape_kw='size')
    E0, psi3, N = LanczosGroundState(ED2, psi3, {}).run()
    print("Lanczos E0 =", E0)
    ov = npc.inner(psi3, psi2, 'range', do_conj=True)
    print("overlap <psi2 | psi3> = 1. -", 1. - ov)
    assert (abs(abs(ov) - 1.) < 1.e-15)

    ED3 = exact_diag.ExactDiag.from_H_mpo(M.H_MPO)
    ED3.build_full_H_from_mpo()
    assert npc.norm(ED3.full_H - H, np.inf) < 1.e-14

    xxz_pars_inf = copy.copy(xxz_pars)
    xxz_pars_inf['bc_MPS'] = 'infinite'
    xxz_pars_inf['L'] = 2
    M_inf = XXZChain(xxz_pars_inf)
    ED4 = exact_diag.ExactDiag.from_infinite_model(M_inf, enlarge=2)
    ED4.build_full_H_from_mpo()
    assert npc.norm(ED4.full_H - H, np.inf) < 1.e-14


def get_tfi_Hamiltonian(L, J, g, up_down_basis=True):
    if up_down_basis:
        sz = np.array([[1, 0], [0, -1]], float)
    else:
        sz = np.array([[-1, 0], [0, 1]], float)
    sx = np.array([[0, 1], [1, 0]], float)
    eye = np.eye(2, dtype=float)
    ops = [eye] * L
    H_expect = 0
    for i in range(L):
        ops = [eye] * L
        ops[i] = sz
        H_expect = H_expect - g * reduce(np.kron, ops)
    for i in range(L - 1):
        ops = [eye] * L
        ops[i] = ops[i + 1] = sx
        H_expect = H_expect - J * reduce(np.kron, ops)
    return H_expect


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_full_wavefunction(undo_sort_charge, conserve, L=10):
    # check with a singlet covering
    # sign convention of singlet = (|up,down> - |down,up>) / sqrt(2)
    assert L % 2 == 0
    assert L % 4 == 2  # only for an odd number of singlets do we detect mixing up the basis order

    # build wavefunction exactly
    up_down_basis = undo_sort_charge or conserve == 'None'
    singlet = np.zeros((2, 2))
    if up_down_basis:
        singlet[0, 1] = +1
        singlet[1, 0] = -1
    else:
        singlet[1, 0] = +1
        singlet[0, 1] = -1
    singlet = np.reshape(singlet, -1) / np.sqrt(2)
    expect = reduce(np.kron, [singlet] * (L // 2))

    # use get_full_wavefunction
    site = SpinHalfSite(conserve='Sz' if conserve == 'best' else conserve)
    psi = MPS.from_singlets(site=site, L=L, pairs=[[i, i + 1] for i in range(0, L, 2)])
    res = exact_diag.get_full_wavefunction(psi, undo_sort_charge=undo_sort_charge)

    # compare
    assert np.allclose(res, expect)


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_scipy_sparse_Hamiltonian(undo_sort_charge, conserve, L=10, J=1, g=4.3291):
    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=undo_sort_charge or conserve == 'None')
    H_res = exact_diag.get_scipy_sparse_Hamiltonian(model, undo_sort_charge=undo_sort_charge)
    assert np.allclose(H_res.toarray(), H_expect)


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
@pytest.mark.parametrize('use_ED', [True, False])
def test_get_numpy_Hamiltonian(undo_sort_charge, conserve, use_ED, J=1, g=4.3291):
    L = 6 if use_ED else 10  # using ED is a bit slow (overhead from combine? or is np.ix_ slow?)
    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=undo_sort_charge or conserve == 'None')
    if use_ED:
        # default behavior for this model is from couplings. test from full_H explicitly.
        H_res = exact_diag._get_numpy_Hamiltonian_ExactDiag_full_H(
            model, from_mpo=True, undo_sort_charge=undo_sort_charge
        )
    else:
        H_res = exact_diag.get_numpy_Hamiltonian(model, undo_sort_charge=undo_sort_charge)
    assert np.allclose(H_res, H_expect)
