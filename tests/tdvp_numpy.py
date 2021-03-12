"""TDVP based purely on numpy.

.. todo ::     Make this a nice toycode! For the test, use some pre-defined values.
"""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import numpy as np
from scipy.sparse.linalg import onenormest


def tdvp(Psi, W, dt, Rp_list=None, k=5, O=None):
    """Applies TDVP on an MPS."""
    L = len(Psi)

    def sweep(Psi, W, dt, Lp_list, Rp_list):
        s = np.ones([1, 1])

        spectrum = []
        spectrum.append(np.array([1]))
        expectation_O = []
        for j in range(L):
            # Get theta
            theta = np.tensordot(s, Psi[j], axes=[1, 1])  # a,i,b
            theta = theta.transpose(1, 0, 2)  # i,a,b
            # Apply expm (-dt H) for 1-site
            d, chia, chib = theta.shape
            H = H1_mixed(Lp_list[-1], Rp_list[L - j - 1], W[j])
            theta = evolve_lanczos(H, theta.reshape(d * chia * chib), -dt,
                                   np.min([d * chia * chib - 1, k]))
            #theta = expm_multiply(H, theta.reshape(d*chia*chib), -dt, np.min([d*chia*chib-1,k]))
            theta = theta.reshape(d * chia, chib) / np.linalg.norm(theta)
            # SVD and update environment
            U, s, V = np.linalg.svd(theta, full_matrices=0)
            spectrum.append(s / np.linalg.norm(s))
            U = U.reshape(d, chia, chib)
            s = np.diag(s)
            V = V.reshape(chib, chib)
            Psi[j] = U

            Lp = np.tensordot(Lp_list[-1], U, axes=(0, 1))  # ap,m,i,b
            Lp = np.tensordot(Lp, W[j], axes=([1, 2], [0, 2]))  # ap,b,n,ip
            Lp = np.tensordot(Lp, np.conj(U), axes=([0, 3], [1, 0]))  # ap,n,b
            Lp = np.transpose(Lp, (0, 2, 1))  # ap,b,n
            Lp_list.append(Lp)

            if j < L - 1:
                # Apply expm (-dt H) for 0-site

                Psi[j + 1] = np.tensordot(V, Psi[j + 1], axes=(1, 1)).transpose(1, 0, 2)

                Rp = np.tensordot(np.conj(V), Rp_list[L - j - 1], axes=(1, 1))
                Rp = np.tensordot(V, Rp, axes=(1, 1))

                H = H0_mixed(Lp_list[-1], Rp)
                s = evolve_lanczos(H, s.reshape(chib * chib), dt, np.min([chib * chib - 1, k]))
                #s = expm_multiply(H, s.reshape(chib*chib), dt, np.min([chib*chib-1,k]))

                s = s.reshape(chib, chib) / np.linalg.norm(s)

        return Psi, Lp_list, spectrum

    D = W[0].shape[0]

    if Rp_list == None:
        Rp_list = [np.zeros([1, 1, D])]
        Rp_list[0][0, 0, D - 1] = 1
        for i in np.arange(L - 1, -1, -1):
            Rp = np.tensordot(Psi[i], Rp_list[-1], axes=(2, 0))  # i a b m
            Rp = np.tensordot(W[i], Rp, axes=([1, 2], [3, 0]))  # m ip a b
            Rp = np.tensordot(np.conj(Psi[i]), Rp, axes=([0, 2], [1, 3]))  # b m a
            Rp = np.transpose(Rp, (2, 0, 1))
            Rp_list.append(Rp)

    Lp_list = [np.zeros([1, 1, D])]
    Lp_list[0][0, 0, 0] = 1
    Psi, Rp_list, spectrum = sweep(Psi, W, dt, Lp_list, Rp_list)
    Lp_mid = Rp_list[int(L / 2)]
    Psi = mps_invert(Psi)
    W = mpo_invert(W)

    Lp_list = [np.zeros([1, 1, D])]
    Lp_list[0][0, 0, D - 1] = 1
    Psi, Rp_list, spectrum = sweep(Psi, W, dt, Lp_list, Rp_list)
    Rp_mid = Rp_list[int(L / 2)]

    Psi = mps_invert(Psi)
    W = mpo_invert(W)

    return Psi, Rp_list, spectrum


#def tdvp_robust(Psi, W, dt,chi,chi0,U,Rp_list=None, k=5,O=None):
#    if chi0<chi


class H0_mixed(object):
    def __init__(self, Lp, Rp, dtype=float):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.chi1 = Lp.shape[0]
        self.chi2 = Rp.shape[0]
        self.shape = np.array([self.chi1 * self.chi2, self.chi1 * self.chi2])
        self.dtype = dtype

    def matvec(self, x):
        x = np.reshape(x, (self.chi1, self.chi2))  # a,b
        x = np.tensordot(self.Lp, x, axes=(0, 0))  # ap,m,b
        x = np.tensordot(x, self.Rp, axes=([1, 2], [2, 0]))  # ap,bp
        x = np.reshape(x, self.chi1 * self.chi2)
        return (x)


class H1_mixed(object):
    def __init__(self, Lp, Rp, M, dtype=float):
        self.Lp = Lp  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.M = M  # m,n,i,ip
        self.d = M.shape[3]
        self.chi1 = Lp.shape[0]
        self.chi2 = Rp.shape[0]
        self.shape = np.array([self.d * self.chi1 * self.chi2, self.d * self.chi1 * self.chi2])
        self.dtype = dtype

    def matvec(self, x):
        x = np.reshape(x, (self.d, self.chi1, self.chi2))  # i,a,b
        x = np.tensordot(self.Lp, x, axes=(0, 1))
        x = np.tensordot(x, self.M, axes=([1, 2], [0, 2]))  # ap,b,n,ip
        x = np.tensordot(x, self.Rp, axes=([1, 2], [0, 2]))  # ap,ip,bp
        x = np.transpose(x, (1, 0, 2))
        x = np.reshape(x, self.d * self.chi1 * self.chi2)
        return (x)


def evolve_lanczos(H, psiI, dt, krylovDim):
    Dim = psiI.shape[0]
    if Dim > 4:
        try:
            Vmatrix = np.zeros((Dim, krylovDim), dtype=np.complex128)

            psiI = psiI / np.linalg.norm(psiI)
            Vmatrix[:, 0] = psiI

            alpha = np.zeros(krylovDim, dtype=np.complex128)
            beta = np.zeros(krylovDim, dtype=np.complex128)

            w = H.matvec(psiI)

            alpha[0] = np.inner(np.conjugate(w), psiI)
            w = w - alpha[0] * Vmatrix[:, 0]
            beta[1] = np.linalg.norm(w)
            Vmatrix[:, 1] = w / beta[1]

            for jj in range(1, krylovDim - 1):
                w = H.matvec(Vmatrix[:, jj]) - beta[jj] * Vmatrix[:, jj - 1]
                alpha[jj] = np.real(np.inner(np.conjugate(w), Vmatrix[:, jj]))
                w = w - alpha[jj] * Vmatrix[:, jj]
                beta[jj + 1] = np.linalg.norm(w)
                Vmatrix[:, jj + 1] = w / beta[jj + 1]

            w = H.matvec(
                Vmatrix[:, krylovDim - 1]) - beta[krylovDim - 1] * Vmatrix[:, krylovDim - 2]
            alpha[krylovDim - 1] = np.real(np.inner(np.conjugate(w), Vmatrix[:, krylovDim - 1]))
            Tmatrix = np.diag(alpha, 0) + np.diag(beta[1:krylovDim], 1) + np.diag(
                beta[1:krylovDim], -1)

            unitVector = np.zeros(krylovDim, dtype=complex)
            unitVector[0] = 1.
            subspaceFinal = np.dot(expm(dt * Tmatrix), unitVector)

            psiF = np.dot(Vmatrix, subspaceFinal)
        except:
            M = np.zeros([Dim, Dim], dtype=complex)
            for i in range(Dim):
                for j in range(Dim):
                    vj = np.zeros(Dim)
                    vj[j] = 1.
                    M[i, j] = H.matvec(vj)[i]
            psiF = np.dot(expm(dt * M), psiI)

    else:
        M = np.zeros([Dim, Dim], dtype=complex)
        for i in range(Dim):
            for j in range(Dim):
                vj = np.zeros(Dim)
                vj[j] = 1.
                M[i, j] = H.matvec(vj)[i]
        psiF = np.dot(expm(dt * M), psiI)

    return psiF


def expm_multiply(A, v, time, m):
    iflag = np.array([1])
    tol = 0.0
    n = A.shape[0]
    anorm = 1
    wsp = np.zeros(7 + n * (m + 2) + 5 * (m + 2) * (m + 2), dtype=complex)
    iwsp = np.zeros(m + 2, dtype=int)

    output_vec, tol0, iflag0 = zgexpv(m, time, v, tol, anorm, wsp, iwsp, A.matvec, 0)
    return output_vec


def MPO_TFI(Jx, Jz, hx, hz):
    d = 2
    Id = np.eye(2, dtype=float)
    Sx = np.array([[0., 1.], [1., 0.]])
    Sz = np.array([[1., 0.], [0., -1.]])

    chi = 4
    W = np.zeros((chi, chi, 2, 2))
    W[0, 0] += Id
    W[0, 1] += Sz
    W[0, 2] += Sx
    W[0, 3] += hz * Sz + hx * Sx

    W[1, 3] += Jz * Sz
    W[2, 3] += Jx * Sx
    W[3, 3] += Id

    return W


def MPO_TFI_general(Jx, Jz, hx, hz, d):
    S = 0.5 * (d - 1)
    Id = np.eye(d, dtype=float)
    d = int(d)
    Sz_diag = -S + np.arange(d)
    Sz = np.diag(Sz_diag)
    Sp = np.zeros([d, d])
    for n in np.arange(d - 1):
        # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
        m = n - S
        Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
    Sm = np.transpose(Sp)
    # Sp = Sx + i Sy, Sm = Sx - i Sy
    Sx = (Sp + Sm) * 0.5
    Sy = (Sm - Sp) * 0.5j
    chi = 4
    W = np.zeros((chi, chi, d, d))
    W[0, 0] += Id
    W[0, 1] += 2 * Sz
    W[0, 2] += 2 * Sx
    W[0, 3] += 2 * hz * Sz + 2 * hx * Sx

    W[1, 3] += 2 * Jz * Sz
    W[2, 3] += 2 * Jx * Sx
    W[3, 3] += Id

    return W


def MPO_Heisenberg(J, d):
    S = 0.5 * (d - 1)
    Id = np.eye(d, dtype=float)
    d = int(d)
    Sz_diag = -S + np.arange(d)
    Sz = np.diag(Sz_diag)
    Sp = np.zeros([d, d])
    for n in np.arange(d - 1):
        # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
        m = n - S
        Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
    Sm = np.transpose(Sp)
    # Sp = Sx + i Sy, Sm = Sx - i Sy
    Sx = (Sp + Sm) * 0.5
    Sy = (Sm - Sp) * 0.5j
    chi = 5
    W = np.zeros((chi, chi, d, d))
    W[0, 0] += Id
    W[0, 1] += 2 * Sp
    W[0, 2] += 2 * Sm
    W[0, 3] += 2 * Sz
    W[0, 4] += 0

    W[1, 4] += J * Sm
    W[2, 4] += J * Sp
    W[3, 4] += 2 * J * Sz
    W[4, 4] += Id

    return W


def MPO_XXZ(Jp, Jz, hz):
    s0 = np.eye(2)
    sp = np.array([[0., 1.], [0., 0.]])
    sm = np.array([[0., 0.], [1., 0.]])
    sz = np.array([[0.5, 0.], [0., -0.5]])
    w_list = []

    w = np.zeros((5, 5, 2, 2), dtype=np.float64)
    w[0, :4] = [s0, sp, sm, sz]
    w[0:, 4] = [hz * sz, Jp / 2. * sm, Jp / 2. * sp, Jz * sz, s0]
    return w


def middle_bond_hamiltonian(Jx, Jz, hx, hz, L):
    """" Returns the spin operators sigma_x and sigma_z for L sites."""
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.], [0., -1.]])

    H_bond = Jx * np.kron(sx, sx) + Jz * np.kron(sz, sz)
    H_bond = H_bond + hx / 2 * np.kron(sx, np.eye(2)) + hx / 2 * np.kron(np.eye(2), sx)
    H_bond = H_bond + hz / 2 * np.kron(sz, np.eye(2)) + hz / 2 * np.kron(np.eye(2), sz)
    H_bond = H_bond.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)  #i1 i2 i1' i2' -->
    U, s, V = np.linalg.svd(H_bond)

    M1 = np.dot(U, np.diag(s)).reshape(2, 2, 1, 4).transpose(2, 3, 0, 1)
    M2 = V.reshape(4, 1, 2, 2)
    M0 = np.tensordot(np.tensordot([1], [1], axes=0), np.eye(2), axes=0)
    W = []

    for i in range(L):
        if i == L / 2 - 1:
            W.append(M1)
        elif i == L / 2:
            W.append(M2)
        else:
            W.append(M0)
    return W


def middle_site_hamiltonian(Jx, Jz, hx, hz, L):
    M0 = np.tensordot(np.tensordot([1], [1], axes=0), np.eye(2), axes=0)
    M1 = MPO_TFI(0, 0, 0, 0)[0:1, :, :, :]
    M2 = MPO_TFI(Jx / 2., Jz / 2., hx, hz)
    M3 = MPO_TFI(Jx / 2., Jz / 2., 0, 0)[:, 3:4, :, :]

    W = []
    for i in range(L):
        if i == L / 2 - 1:
            W.append(M1)
        elif i == L / 2:
            W.append(M2)
        elif i == L / 2 + 1:
            W.append(M3)
        else:
            W.append(M0)
    return (W)


def mps_invert(Psi):
    np = Psi[0].ndim - 2
    return [b.transpose(list(range(np)) + [-1, -2]) for b in Psi[::-1]]


def mpo_invert(Psi):
    np = Psi[0].ndim
    return [b.transpose([1, 0] + list(range(2, np))) for b in Psi[::-1]]
