# Copyright 2018 TeNPy Developers
import tenpy.linalg.np_conserved as npc
import numpy as np

from random_test import gen_random_legcharge
from tenpy.linalg import sparse
import tenpy.linalg.random_matrix as rmat
import scipy.sparse.linalg
import scipy.linalg

ch = npc.ChargeInfo([2])


def test_FlatLinearOperator(n=30, k=5, tol=5.e-15):
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg)
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eigh(H_flat)
    E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    qtotal = npc.detect_qtotal(psi0_flat, [leg])

    H_sparse = sparse.FlatLinearOperator.from_NpcArray(H, charge_sector=qtotal)
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    psi_init /= npc.norm(psi_init)
    psi_init_flat = H_sparse.npc_to_flat(psi_init)

    # check diagonalization
    E, psi = scipy.sparse.linalg.eigsh(H_sparse, k, v0=psi_init_flat, which='SA')
    E0, psi0 = E[0], psi[:, 0]
    print("full spectrum:", E_flat)
    print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    assert (abs((E0 - E0_flat) / E0_flat) < tol)
    psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)


def test_FlatHermitianOperator(n=30, k=5, tol=5.e-15):
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg)
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eigh(H_flat)
    E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    qtotal = npc.detect_qtotal(psi0_flat, [leg])

    H_sparse = sparse.FlatHermitianOperator.from_NpcArray(H, charge_sector=qtotal)
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    psi_init /= npc.norm(psi_init)
    psi_init_flat = H_sparse.npc_to_flat(psi_init)

    # check diagonalization
    E, psi = scipy.sparse.linalg.eigsh(H_sparse, k, v0=psi_init_flat, which='SA')
    E0, psi0 = E[0], psi[:, 0]
    print("full spectrum:", E_flat)
    print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    assert (abs((E0 - E0_flat) / E0_flat) < tol)
    psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
