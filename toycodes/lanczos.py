"""Simple Lanczos diagonalization routines.

These functions can be used in d_dmrg.py and e_tdvp to replace
the eigsh and expm_mulitply calls, as commented out in the code.
"""
# Copyright 2021-2022 TeNPy Developers, GNU GPLv3

from scipy.sparse.linalg import expm
import numpy as np
import warnings

default_k = 20

def lanczos_ground_state(H, psi0, k=None):
    """Use lanczos to calculated the ground state of a hermitian H.

    If you don't know Lanczos, you can view this as a black box algorithm.
    The idea is to built an orthogonal basis in the Krylov space spanned by
    ``{ H^i |psi0> for i < k}`` and find the ground state in this sub space.
    It only requires an `H` which defines a matrix-vector multiplication.
    """
    T, vecs = lanczos_iterations(H, psi0, k)
    E, v = np.linalg.eigh(T)
    result = vecs @ v[:, 0]
    if abs(np.linalg.norm(result) - 1.) > 1.e-5:
        warnings.warn("poorly conditioned lanczos. Maybe a non-hermitian H?")
    return E[0], result

def lanczos_expm_multiply(H, psi0, dt, k=None):
    """Use lanczos to calculated ``expm(-i H dt)|psi0>`` for sufficiently small dt and hermitian H.

    If you don't know Lanczos, you can view this as a black box algorithm.
    The idea is to built an orthogonal basis in the Krylov space spanned by
    ``{ H^i |psi0> for i < k}`` and evolve only in this subspace.
    It only requires an `H` which defines a matrix-vector multiplication.
    """
    T, vecs = lanczos_iterations(H, psi0, k)
    v0 = np.zeros(T.shape[0])
    v0[0] = 1.
    vt = expm(-1.j*dt*T) @ v0
    result = vecs @ vt
    if abs(np.linalg.norm(result) - 1.) > 1.e-5:
        warnings.warn("poorly conditioned lanczos. Maybe a non-hermitian H?")
    return result

def lanczos_iterations(H, psi0, k):
    """Perform `k` Lanczos iterations building tridiagonal matrix T and ONB of the Krylov space."""
    if k is None:
        k = default_k
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")
    psi0 = psi0/np.linalg.norm(psi0)
    vecs = [psi0]
    T = np.zeros((k, k))
    psi = H @ psi0
    alpha = T[0, 0] = np.inner(psi0.conj(), psi).real
    psi = psi - alpha* vecs[-1]
    for i in range(1, k):
        beta = np.linalg.norm(psi)
        if beta  < 1.e-13:
            #  print(f"Lanczos terminated early after i={i:d} steps:"
            #        "full Krylov space built")
            T = T[:i, :i]
            break
        psi /= beta
        vecs.append(psi)
        psi = H @ psi - beta * vecs[-2]
        alpha = np.inner(vecs[-1].conj(), psi).real
        psi = psi - alpha * vecs[-1]
        T[i, i] = alpha
        T[i-1, i] = T[i, i-1] = beta
    return T, np.array(vecs).T
