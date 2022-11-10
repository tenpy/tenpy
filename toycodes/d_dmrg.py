"""Toy code implementing the density-matrix renormalization group (DMRG)."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from a_mps import split_truncate_theta
import scipy.sparse
from scipy.sparse.linalg import eigsh


class SimpleDMRGEngine:
    """DMRG algorithm, implemented as class holding the necessary data.

    DMRG sweeps left-right-left through the system, moving the orthogonality center along.
    Here, we still just save right-canonical `B` tensors in `psi`, which requires taking inverses
    of the Schmidt values - this is bad practice, but it keeps two things simpler:
    - We don't need book keeping in the MPS class to keep track of the canonical form, and all the
    MPS methods (expectation values etc) can just *assume* that the MPS is in right-canonical form.
    - The generalization to the infinite case is straight forward.
    Note, however, that we only use the A and B tensors directly from the SVD (without taking
    inverses) to update the environments. The effective Hamiltonian does thus not suffer
    from taking the inverses of Schmidt values.

    Parameters
    ----------
    psi, model, chi_max, eps:
        See attributes

    Attributes
    ----------
    psi : SimpleMPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated. Needs to have an `H_mpo`.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
    LPs, RPs : list of np.Array[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPs[i]`` has legs ``vR* wR* vR``
    """
    def __init__(self, psi, model, chi_max, eps):
        assert psi.L == model.L and psi.bc == model.bc  # ensure compatibility
        self.H_mpo = model.H_mpo
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        # initialize left and right environment
        D = self.H_mpo[0].shape[0]
        chi = psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype=float)  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype=float)  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i, psi.Bs[i])

    def sweep(self):
        # sweep from left to right
        for i in range(self.psi.nbonds - 1):
            self.update_bond(i)
        # sweep from right to left
        for i in range(self.psi.nbonds - 1, 0, -1):
            E0 = self.update_bond(i)
        return E0

    def update_bond(self, i):
        j = (i + 1) % self.psi.L
        # get effective Hamiltonian
        Heff = SimpleHeff2(self.LPs[i], self.RPs[j], self.H_mpo[i], self.H_mpo[j])
        # Diagonalize Heff, find ground state `theta`
        theta_guess = self.psi.get_theta2(i)
        E0, theta = self.diag(Heff, theta_guess)
        # split and truncate
        Ai, Sj, Bj = split_truncate_theta(theta, self.chi_max, self.eps)
        # put back into MPS
        Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), Ai, axes=(1, 0))  # vL [vL*], [vL] i vC
        self.psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))  # vL i [vC], [vC*] vC
        self.psi.Ss[j] = Sj  # vC
        self.psi.Bs[j] = Bj  # vC j vR
        self.update_LP(i, Ai)
        self.update_RP(j, Bj)
        return E0

    def diag(self, Heff, guess):
        """Diagonalize the effective hamiltonian with an initial guess."""
        guess = np.reshape(guess, [Heff.shape[1]])
        E, V = eigsh(Heff, k=1, which='SA', return_eigenvectors=True, v0=guess)
        return E, np.reshape(V[:, 0], Heff.theta_shape)
        #  # alternatively, use custom lanczos implementation
        #  from lanczos import lanczos_ground_state
        #  return lanczos_ground_state(H, guess)

    def update_RP(self, i, B):
        """Calculate RP environment right of site `i-1`.

        Uses RP right of `i` and the given, right-canonical `B` on site `i`."""
        j = (i - 1) % self.psi.L
        RP = self.RPs[i]  # vR* wR* vR
        # B has legs     vL i vR
        Bc = B.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, axes=(2, 0))  # vL i [vR], [vR*] wR* vR
        RP = np.tensordot(RP, W, axes=([1, 2], [3, 1]))  # vL [i] [wR*] vR, wL [wR] i [i*]
        RP = np.tensordot(RP, Bc, axes=([1, 3], [2, 1]))  # vL [vR] wL [i], vL* [i*] [vR*]
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)

    def update_LP(self, i, A):
        """Calculate LP environment left of site `i+1`.

        Uses the LP left of site `i` and the given, left-canonical `A` on site `i`."""
        j = (i + 1) % self.psi.L
        LP = self.LPs[i]  # vL wL vL*
        # A has legs    vL i vR
        Ac = A.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        LP = np.tensordot(LP, A, axes=(2, 0))  # vL wL* [vL*], [vL] i vR
        LP = np.tensordot(W, LP, axes=([0, 3], [1, 2]))  # [wL] wR i [i*], vL [wL*] [i] vR
        LP = np.tensordot(Ac, LP, axes=([0, 1], [2, 1]))  # [vL*] [i*] vR*, wR [i] [vL] vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)


class SimpleHeff2(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian on two sites.

    To be diagonalized in `SimpleDMRGEnginge.diag` during the bond update. Looks like this::

        .--vL*           vR*--.
        |       i*    j*      |
        |       |     |       |
       (LP)----(W1)--(W2)----(RP)
        |       |     |       |
        |       i     j       |
        .--vL             vR--.
    """
    def __init__(self, LP, RP, W1, W2):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chi1, d1, d2, chi2)  # vL i j vR
        self.shape = (chi1 * d1 * d2 * chi2, chi1 * d1 * d2 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        """Calculate the matrix-vecotr product |theta'> = H_eff |theta>.

        This function is used by :func:`scipy.sparse.linalg.eigsh` to diagonalize
        the effective Hamiltonian with a Lanczos method, withouth generating the full matrix.
        """
        x = np.reshape(theta, self.theta_shape)  # vL i j vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i j vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] j vR, [wL] wC i [i*]
        x = np.tensordot(x, self.W2, axes=([3, 1], [0, 3]))  # vL [j] vR [wC] i, [wC] wR j [j*]
        x = np.tensordot(x, self.RP, axes=([1, 3], [0, 1]))  # vL [vR] i [wR] j, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x


def example_DMRG_tf_ising_finite(L, g, chi_max=20):
    print("finite DMRG, transverse field Ising")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    import a_mps
    import b_model
    model = b_model.TFIModel(L=L, J=1., g=g, bc='finite')
    psi = a_mps.init_FM_MPS(model.L, model.d, model.bc)
    eng = SimpleDMRGEngine(psi, model, chi_max=chi_max, eps=1.e-10)
    for i in range(10):
        eng.sweep()
        E = np.sum(psi.bond_expectation_value(model.H_bonds))
        print("sweep {i:2d}: E = {E:.13f}".format(i=i + 1, E=E))
    print("final bond dimensions: ", psi.get_chi())
    mag_x = np.sum(psi.site_expectation_value(model.sigmax))
    mag_z = np.sum(psi.site_expectation_value(model.sigmaz))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, model


def example_DMRG_tf_ising_infinite(g, chi_max=30):
    print("infinite DMRG, transverse field Ising")
    print("g={g:.2f}".format(g=g))
    import a_mps
    import b_model
    model = b_model.TFIModel(L=2, J=1., g=g, bc='infinite')
    psi = a_mps.init_FM_MPS(model.L, model.d, model.bc)
    eng = SimpleDMRGEngine(psi, model, chi_max=chi_max, eps=1.e-14)
    for i in range(20):
        eng.sweep()
        E = np.mean(psi.bond_expectation_value(model.H_bonds))
        print("sweep {i:2d}: E (per site) = {E:.13f}".format(i=i + 1, E=E))
    print("final bond dimensions: ", psi.get_chi())
    mag_x = np.mean(psi.site_expectation_value(model.sigmax))
    mag_z = np.mean(psi.site_expectation_value(model.sigmaz))
    print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, model


if __name__ == "__main__":
    example_DMRG_tf_ising_finite(L=10, g=1.)
    print("-" * 100)
    example_DMRG_tf_ising_infinite(g=1.5)
