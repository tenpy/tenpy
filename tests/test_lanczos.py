"""A collection of tests for tenpy.linalg.lanczos"""
# Copyright 2018 TeNPy Developers

import tenpy.linalg.np_conserved as npc
import numpy as np

from random_test import gen_random_legcharge
from tenpy.linalg import lanczos, sparse
import tenpy.linalg.random_matrix as rmat
from scipy.linalg import expm
import pytest

ch = npc.ChargeInfo([2])


def test_gramschmidt(n=30, k=5, tol=1.e-15):
    leg = gen_random_legcharge(ch, n)
    vecs_old = [npc.Array.from_func(np.random.random, [leg], shape_kw='size') for i in range(k)]
    vecs_new, _ = lanczos.gram_schmidt(vecs_old, rcond=0., verbose=1)
    assert all([v == w for v, w in zip(vecs_new, vecs_old)])
    vecs_new, _ = lanczos.gram_schmidt(vecs_old, rcond=tol, verbose=1)
    vecs = [v.to_ndarray() for v in vecs_new]
    ovs = np.zeros((k, k))
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            ovs[i, j] = np.inner(v.conj(), w)
    print(ovs)
    assert (np.linalg.norm(ovs - np.eye(k)) < 2 * n * k * k * tol)


@pytest.mark.parametrize('n, N_cache', [(10, 20)] + [(n, 6) for n in [1, 2, 4, 20]])
def test_lanczos_gs(n, N_cache, tol=5.e-15):
    # generate Hermitian test array
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg)
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eigh(H_flat)
    E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    qtotal = npc.detect_qtotal(psi0_flat, [leg])

    H_Op = H  # use `matvec` of the array
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)

    E0, psi0, N = lanczos.lanczos(H_Op, psi_init, {'verbose': 1, 'N_cache': N_cache})
    print("full spectrum:", E_flat)
    print("E0 = {E0:.14f} vs exact {E0_flat:.14f}".format(E0=E0, E0_flat=E0_flat))
    print("|E0-E0_flat| / |E0_flat| =", abs((E0 - E0_flat) / E0_flat))
    psi0_H_psi0 = npc.inner(psi0, npc.tensordot(H, psi0, axes=[1, 0]), do_conj=True)
    print("<psi0|H|psi0> / E0 = 1. + ", psi0_H_psi0 / E0 - 1.)
    assert (abs(psi0_H_psi0 / E0 - 1.) < tol)
    print("<psi0_flat|H_flat|psi0_flat> / E0_flat = ", end=' ')
    print(np.inner(psi0_flat.conj(), np.dot(H_flat, psi0_flat)) / E0_flat)
    ov = np.inner(psi0.to_ndarray().conj(), psi0_flat)
    print("|<psi0|psi0_flat>|=", abs(ov))
    assert (abs(1. - abs(ov)) < tol)

    # now repeat, but keep orthogonal to original ground state
    # -> should give second eigenvector psi1 in the same charge sector
    for i in range(1, len(E_flat)):
        E1_flat, psi1_flat = E_flat[i], psi_flat[:, i]
        qtotal = npc.detect_qtotal(psi1_flat, psi0.legs)
        if np.all(qtotal == psi0.qtotal) and E1_flat < -0.01:
            break  # found psi1 in same charge sector
    else:
        print("warning: test didn't find a second eigenvector in the same charge sector!")
        return  # just ignore the rest....
    E1, psi1, N = lanczos.lanczos(H_Op,
                                  psi_init, {
                                      'verbose': 1,
                                      'reortho': True
                                  },
                                  orthogonal_to=[psi0])
    print("E1 = {E1:.14f} vs exact {E1_flat:.14f}".format(E1=E1, E1_flat=E1_flat))
    print("|E1-E1_flat| / |E1_flat| =", abs((E1 - E1_flat) / E1_flat))
    psi1_H_psi1 = npc.inner(psi1, npc.tensordot(H, psi1, axes=[1, 0]), do_conj=True)
    print("<psi1|H|psi1> / E1 = 1 + ", psi1_H_psi1 / E1 - 1.)
    assert (abs(psi1_H_psi1 / E1 - 1.) < tol)
    print("<psi1_flat|H_flat|psi1_flat> / E1_flat = ", end=' ')
    print(np.inner(psi1_flat.conj(), np.dot(H_flat, psi1_flat)) / E1_flat)
    ov = np.inner(psi1.to_ndarray().conj(), psi1_flat)
    print("|<psi1|psi1_flat>|=", abs(ov))
    assert (abs(1. - abs(ov)) < tol)
    # and finnally check also orthogonality to psi0
    ov = npc.inner(psi0, psi1, do_conj=True)
    print("|<psi0|psi1>| =", abs(ov))
    assert (abs(ov) < tol**0.5)


@pytest.mark.parametrize('n, N_cache', [(10, 20)] + [(n, 6) for n in [1, 2, 4, 20]])
def test_lanczos_evolve(n, N_cache, tol=5.e-15):
    # generate Hermitian test array
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg) - npc.diag(1., leg)
    H_flat = H.to_ndarray()
    H_Op = H  # use `matvec` of the array
    qtotal = leg.to_qflat()[0]
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    psi_init /= npc.norm(psi_init)
    psi_init_flat = psi_init.to_ndarray()
    lanc = lanczos.LanczosEvolution(H_Op, psi_init, {'verbose': 1, 'N_cache': N_cache})
    for delta in [-0.1j, 0.1j, 1.j]:  #, 0.1, 1.]:
        psi_final_flat = expm(H_flat * delta).dot(psi_init_flat)
        psi_final, N = lanc.run(delta)
        ov = np.inner(psi_final.to_ndarray().conj(), psi_final_flat)
        ov /= np.linalg.norm(psi_final_flat)
        print("<psi1|psi1_flat>/norm=", ov)
        assert (abs(1. - abs(ov)) < tol)
