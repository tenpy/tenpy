# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
import tenpy.linalg.np_conserved as npc
import numpy as np

from random_test import gen_random_legcharge
from tenpy.linalg import sparse
import tenpy.linalg.random_matrix as rmat
import scipy.sparse.linalg
import scipy.linalg

ch = npc.ChargeInfo([2])


def test_FlatLinearOperator(n=30, k=5, tol=1.e-14):
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


def test_FlatHermitianOperator(n=30, k=5, tol=1.e-14):
    leg = gen_random_legcharge(ch, n // 2)
    leg2 = gen_random_legcharge(ch, 2)
    pipe = npc.LegPipe([leg, leg2], qconj=+1)
    H = npc.Array.from_func_square(rmat.GUE, pipe, labels=["(a.b)", "(a*.b*)"])
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eigh(H_flat)
    E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    qtotal = npc.detect_qtotal(psi0_flat, [pipe])

    H_sparse = sparse.FlatHermitianOperator.from_NpcArray(H, charge_sector=qtotal)
    psi_init = npc.Array.from_func(np.random.random, [pipe], qtotal=qtotal, labels=["(a.b)"])
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

    # split H to check `FlatHermitianOperator.from_guess_with_pipe`.
    print("=========")
    print("split legs and define separate matvec")
    assert psi_init.legs[0] is pipe
    psi_init_split = psi_init.split_legs([0])
    H_split = H.split_legs()

    def H_split_matvec(vec):
        vec = npc.tensordot(H_split, vec, [["a*", "b*"], ["a", "b"]])
        # TODO as additional challenge, transpose the resulting vector
        return vec

    H_sparse_split, psi_init_split_flat = sparse.FlatLinearOperator.from_guess_with_pipe(
        H_split_matvec, psi_init_split, dtype=H_split.dtype)

    # diagonalize
    E, psi = scipy.sparse.linalg.eigsh(H_sparse_split, k, v0=psi_init_split_flat, which='SA')
    E0, psi0 = E[0], psi[:, 0]
    print("full spectrum:", E_flat)
    print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    assert (abs((E0 - E0_flat) / E0_flat) < tol)
    psi0_H_psi0 = np.inner(psi0.conj(), H_sparse.matvec(psi0)).item()
    print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
