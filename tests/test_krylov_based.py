"""A collection of tests for tenpy.linalg.krylov_based."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import pytest
from random_test import gen_random_legcharge
from scipy.linalg import expm

import tenpy.linalg.np_conserved as npc
import tenpy.linalg.random_matrix as rmat
from tenpy.linalg import krylov_based, sparse

ch = npc.ChargeInfo([2])


def test_gramschmidt(n=30, k=5, tol=1.0e-15):
    leg = gen_random_legcharge(ch, n)
    vecs_old = [npc.Array.from_func(rmat.standard_normal_complex, [leg], shape_kw='size') for i in range(k)]
    vecs_new = krylov_based.gram_schmidt(vecs_old, rcond=0.0)
    assert all([v is w for v, w in zip(vecs_new, vecs_old)])
    vecs_new = krylov_based.gram_schmidt(vecs_old, rcond=tol)
    vecs = [v.to_ndarray() for v in vecs_new]
    ovs = np.zeros((k, k), dtype=np.complex128)
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            ovs[i, j] = np.inner(v.conj(), w)
    print(ovs)
    assert np.linalg.norm(ovs - np.eye(k)) < 2 * n * k * k * tol


@pytest.mark.parametrize('n, N_cache', [(10, 20)] + [(n, 6) for n in [1, 2, 4, 20]])
def test_lanczos_gs(n, N_cache, tol=5.0e-14):
    # generate Hermitian test array
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg)
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eigh(H_flat)
    E0_flat, psi0_flat = E_flat[0], psi_flat[:, 0]
    qtotal = npc.detect_qtotal(psi0_flat, [leg])

    H_Op = H  # use `matvec` of the array
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)

    E0, psi0, N = krylov_based.LanczosGroundState(H_Op, psi_init, {'N_cache': N_cache}).run()
    print('full spectrum:', E_flat)
    print(f'E0 = {E0:.14f} vs exact {E0_flat:.14f}')
    print('|E0-E0_flat| / |E0_flat| =', abs((E0 - E0_flat) / E0_flat))
    psi0_H_psi0 = npc.inner(psi0, npc.tensordot(H, psi0, axes=[1, 0]), 'range', do_conj=True)
    print('<psi0|H|psi0> / E0 = 1. + ', psi0_H_psi0 / E0 - 1.0)
    assert abs(psi0_H_psi0 / E0 - 1.0) < tol
    print('<psi0_flat|H_flat|psi0_flat> / E0_flat = ', end=' ')
    print(np.inner(psi0_flat.conj(), np.dot(H_flat, psi0_flat)) / E0_flat)
    ov = np.inner(psi0.to_ndarray().conj(), psi0_flat)
    print('|<psi0|psi0_flat>|=', abs(ov))
    assert abs(1.0 - abs(ov)) < tol

    print('version with arpack')
    E0a, psi0a = krylov_based.lanczos_arpack(H_Op, psi_init, {})
    print(f'E0a = {E0a:.14f} vs exact {E0_flat:.14f}')
    print('|E0a-E0_flat| / |E0_flat| =', abs((E0a - E0_flat) / E0_flat))
    psi0a_H_psi0a = npc.inner(psi0a, npc.tensordot(H, psi0a, axes=[1, 0]), 'range', do_conj=True)
    print('<psi0a|H|psi0a> / E0a = 1. + ', psi0a_H_psi0a / E0a - 1.0)
    assert abs(psi0a_H_psi0a / E0a - 1.0) < tol
    ov = np.inner(psi0a.to_ndarray().conj(), psi0_flat)
    print('|<psi0a|psi0_flat>|=', abs(ov))
    assert abs(1.0 - abs(ov)) < tol

    # now repeat, but keep orthogonal to original ground state
    orthogonal_to = [psi0]
    # -> should give second eigenvector psi1 in the same charge sector
    for i in range(1, len(E_flat)):
        E1_flat, psi1_flat = E_flat[i], psi_flat[:, i]
        qtotal = npc.detect_qtotal(psi1_flat, psi0.legs, cutoff=1.0e-10)
        if np.any(qtotal != psi0.qtotal):
            continue  # not in same charge sector
        print('--- excited state #', len(orthogonal_to))
        ortho_to = [psi.copy() for psi in orthogonal_to]  # (gets modified inplace)
        lanczos_params = {'reortho': True}
        if E1_flat > -0.01:
            lanczos_params['E_shift'] = -2.0 * E1_flat - 0.2
        E1, psi1, N = krylov_based.LanczosGroundState(
            sparse.OrthogonalNpcLinearOperator(H_Op, ortho_to), psi_init, lanczos_params
        ).run()
        print(f'E1 = {E1:.14f} vs exact {E1_flat:.14f}')
        print('|E1-E1_flat| / |E1_flat| =', abs((E1 - E1_flat) / E1_flat))
        psi1_H_psi1 = npc.inner(psi1, npc.tensordot(H, psi1, axes=[1, 0]), 'range', do_conj=True)
        print('<psi1|H|psi1> / E1 = 1 + ', psi1_H_psi1 / E1 - 1.0)
        assert abs(psi1_H_psi1 / E1 - 1.0) < tol
        print('<psi1_flat|H_flat|psi1_flat> / E1_flat = ', end='')
        print(np.inner(psi1_flat.conj(), np.dot(H_flat, psi1_flat)) / E1_flat)
        ov = np.inner(psi1.to_ndarray().conj(), psi1_flat)
        print('|<psi1|psi1_flat>|=', abs(ov))
        assert abs(1.0 - abs(ov)) < tol
        # and finnally check also orthogonality to previous states
        for psi_prev in orthogonal_to:
            ov = npc.inner(psi_prev, psi1, 'range', do_conj=True)
            assert abs(ov) < tol
        orthogonal_to.append(psi1)
    if len(orthogonal_to) == 1:
        print("warning: test didn't find a second eigenvector in the same charge sector!")


@pytest.mark.parametrize('n, N_cache', [(10, 20)] + [(n, 6) for n in [1, 2, 4, 20]])
def test_lanczos_evolve(n, N_cache, tol=5.0e-14):
    # generate Hermitian test array
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.GUE, leg) - npc.diag(1.0, leg)
    H_flat = H.to_ndarray()
    H_Op = H  # use `matvec` of the array
    qtotal = leg.to_qflat()[0]
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    # psi_init /= npc.norm(psi_init) # not necessary
    psi_init_flat = psi_init.to_ndarray()
    lanc = krylov_based.LanczosEvolution(H_Op, psi_init, {'N_cache': N_cache})
    for delta in [-0.1j, 0.1j, 1.0j, 0.1, 1.0]:
        psi_final_flat = expm(H_flat * delta).dot(psi_init_flat)
        norm = np.linalg.norm(psi_final_flat)
        psi_final, N = lanc.run(delta, normalize=False)
        diff = np.linalg.norm(psi_final.to_ndarray() - psi_final_flat)
        print('norm(|psi_final> - |psi_final_flat>)/norm = ', diff / norm)  # should be 1.
        assert diff / norm < tol
        psi_final2, N = lanc.run(delta, normalize=True)
        assert npc.norm(psi_final / norm - psi_final2) < tol


@pytest.mark.parametrize('n, which', [(10, 'LM'), (30, 'LM'), (30, 'SR'), (30, 'LR')])
def test_arnoldi(n, which):
    tol = 5.0e-14 if n <= 20 else 1.0e-10
    # generate Hermitian test array
    leg = gen_random_legcharge(ch, n)
    # if looking for small/large real part, ensure hermitian H
    func = rmat.GUE if which[-1] == 'R' else rmat.standard_normal_complex
    H = npc.Array.from_func_square(func, leg)
    H_flat = H.to_ndarray()
    E_flat, psi_flat = np.linalg.eig(H_flat)
    if which == 'LM':
        i = np.argmax(np.abs(E_flat))
    elif which == 'LR':
        i = np.argmax(np.real(E_flat))
    elif which == 'SR':
        i = np.argmin(np.real(E_flat))
    E0_flat, psi0_flat = E_flat[i], psi_flat[:, i]
    qtotal = npc.detect_qtotal(psi0_flat, [leg])

    H_Op = H  # use `matvec` of the array
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)

    eng = krylov_based.Arnoldi(H_Op, psi_init, {'which': which, 'num_ev': 1, 'N_max': 20})
    (E0,), (psi0,), N = eng.run()
    print('full spectrum:', E_flat)
    print(f'E0 = {E0:.14f} vs exact {E0_flat:.14f}')
    print('|E0-E0_flat| / |E0_flat| =', abs((E0 - E0_flat) / E0_flat))
    assert abs((E0 - E0_flat) / E0_flat) < tol
    psi0_H_psi0 = npc.inner(psi0, npc.tensordot(H, psi0, axes=[1, 0]), 'range', do_conj=True)
    print('<psi0|H|psi0> / E0 = 1. + ', psi0_H_psi0 / E0 - 1.0)
    assert abs(psi0_H_psi0 / E0 - 1.0) < tol
    print('<psi0_flat|H_flat|psi0_flat> / E0_flat = ', end=' ')
    print(np.inner(psi0_flat.conj(), np.dot(H_flat, psi0_flat)) / E0_flat)
    ov = np.inner(psi0.to_ndarray().conj(), psi0_flat)
    print('|<psi0|psi0_flat>|=', abs(ov))
    assert abs(1.0 - abs(ov)) < tol


@pytest.mark.parametrize('n', [1, 2, 4, 10, 20])
def test_arnoldi_evolve(n, tol=5.0e-13):
    # non-Hermitian matrix (standard_normal_complex gives an arbitrary complex matrix)
    leg = gen_random_legcharge(ch, n)
    H = npc.Array.from_func_square(rmat.standard_normal_complex, leg)
    H_flat = H.to_ndarray()
    H_Op = H  # use `matvec` of the array
    qtotal = leg.to_qflat()[0]
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    psi_init_flat = psi_init.to_ndarray()
    eng = krylov_based.ArnoldiEvolution(H_Op, psi_init, {'N_max': 20})
    for delta in [-0.1j, 0.1j, 0.5j, 0.1, -0.05 - 0.1j]:
        psi_final_flat = expm(H_flat * delta).dot(psi_init_flat)
        norm = np.linalg.norm(psi_final_flat)
        psi_final, N = eng.run(delta, normalize=False)
        diff = np.linalg.norm(psi_final.to_ndarray() - psi_final_flat)
        print(f'delta={delta}, N={N}, norm(diff)/norm = {diff / norm}')
        assert diff / norm < tol
        psi_final2, N = eng.run(delta, normalize=True)
        assert npc.norm(psi_final / norm - psi_final2) < tol
        # Default normalize=None should behave like normalize=False
        psi_final_default, N = eng.run(delta)
        assert npc.norm(psi_final_default - psi_final) < tol * npc.norm(psi_final)


def test_arnoldi_evolve_dense(tol=5.0e-13):
    """ArnoldiEvolution on a dense (trivial-charge) matrix, larger Krylov space."""
    n = 30
    H_np = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = npc.Array.from_ndarray_trivial(H_np, dtype=complex, labels=['p', 'p*'])
    leg = H.legs[0]
    psi_init_np = np.random.randn(n) + 1j * np.random.randn(n)
    psi_init = npc.Array.from_ndarray_trivial(psi_init_np, dtype=complex, labels=['p'])
    psi_init_flat = psi_init.to_ndarray()
    eng = krylov_based.ArnoldiEvolution(H, psi_init, {'N_max': 30})
    for delta in [-0.1j, 1.0j, 0.1, -0.05 - 0.1j]:
        psi_final_flat = expm(H_np * delta).dot(psi_init_flat)
        norm = np.linalg.norm(psi_final_flat)
        psi_final, N = eng.run(delta, normalize=False)
        diff = np.linalg.norm(psi_final.to_ndarray() - psi_final_flat)
        print(f'dense n={n}, delta={delta}, N={N}, norm(diff)/norm = {diff / norm}')
        assert diff / norm < tol
    # For large delta (delta=1.0j), expect N > 1 (non-trivial Krylov expansion needed)
    _, N_large = eng.run(1.0j, normalize=False)
    assert N_large > 1


def test_arnoldi_vs_lanczos_nonhermitian(tol_arnoldi=1.0e-10, tol_lanczos_wrong=1.0e-2):
    """ArnoldiEvolution is accurate for non-Hermitian H; LanczosEvolution is not."""
    n = 20
    leg = gen_random_legcharge(ch, n)
    # Anti-Hermitian H = 1j * G with G from GUE: H† = -H, so exp(delta*H) is unitary for real delta.
    # This is non-Hermitian (imaginary eigenvalues), so LanczosEvolution's eigh is wrong.
    G = npc.Array.from_func_square(rmat.GUE, leg)
    H = 1j * G
    H_flat = H.to_ndarray()
    qtotal = leg.to_qflat()[0]
    psi_init = npc.Array.from_func(np.random.random, [leg], qtotal=qtotal)
    psi_init_flat = psi_init.to_ndarray()

    delta = 1.0  # real delta; exp(1.0 * 1j * G) is unitary
    psi_ref_flat = expm(H_flat * delta).dot(psi_init_flat)
    norm_ref = np.linalg.norm(psi_ref_flat)

    psi_arnoldi, _ = krylov_based.ArnoldiEvolution(H, psi_init, {'N_max': 20}).run(delta, normalize=False)
    diff_arnoldi = np.linalg.norm(psi_arnoldi.to_ndarray() - psi_ref_flat)
    print(f'ArnoldiEvolution diff/norm = {diff_arnoldi / norm_ref}')
    assert diff_arnoldi / norm_ref < tol_arnoldi

    psi_lanczos, _ = krylov_based.LanczosEvolution(H, psi_init, {}).run(delta, normalize=False)
    diff_lanczos = np.linalg.norm(psi_lanczos.to_ndarray() - psi_ref_flat)
    print(f'LanczosEvolution diff/norm = {diff_lanczos / norm_ref}  (expected to be WRONG)')
    assert diff_lanczos / norm_ref > tol_lanczos_wrong
