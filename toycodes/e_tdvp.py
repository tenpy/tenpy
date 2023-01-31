"""Toy code implementing the TDVP for *finite* MPS."""
# Copyright 2021-2022 TeNPy Developers, GNU GPLv3

import numpy as np
import scipy.sparse.linalg
from scipy.sparse.linalg import expm

from d_dmrg import SimpleHeff2
import a_mps


class SimpleTDVPEngine:
    """TDVP algorithm for finite systems, implemented as class holding the necessary data.

    Note that this class is very similar to `d_dmrg.SimpleDMRGEngine`.
    We could use a common base class; but to keep things maximally simple and readable,
    we rather duplicate the code for the `__init__`, `update_LP`, and `update_RP` methods.

    Also, here we generalize the sweep to temporarily change the MPS to a mixed canonical form
    and directly save `A` tensors in it. This means that the SimpleMPS methods (which *assume*
    that the tensors are all right-canonical) would give wrong results *during* the sweep; yet
    we recover the all-right-canonical B form on each site at the end of the sweep.

    Parameters
    ----------
    psi, chi_max, eps:
        See attributes below.
    model :
        The model with the Hamiltonian for time evolution as `model.H_mpo`.

    Attributes
    ----------
    psi : SimpleMPS
        The current state to be evolved.
    H_mpo : list of W tensors with legs ``wL wR i i*``
        The Hamiltonian as an MPO.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
        Only used when we evolve two-site wave functions!
    LPs, RPs : list of np.Array[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPs[i]`` has legs ``vR* wR* vR``
    """
    def __init__(self, psi, model, chi_max, eps):
        assert psi.L == model.L and psi.bc == model.bc  # ensure compatibility
        if psi.bc != 'finite':
            raise ValueError("This TDVP implementation works only for finite MPS.")
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
        for i in range(psi.L - 1, 0, -1):
            self.update_RP(i, psi.Bs[i])

    def sweep_one_site(self, dt):
        """Perform one one-site TDVP sweep to evolve |psi> -> exp(-i H_mpo dt) |psi>.

        This does *not* grow the bond dimension of the MPS, but is strictly TDVP.
        """
        psi = self.psi
        L = self.psi.L
        # sweep from left to right
        theta = self.psi.get_theta1(0)
        for i in range(L - 1):
            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            Ai, theta = self.split_one_site_theta(i, theta, move_right=True)
            # here theta is zero-site between site i and i+1
            psi.Bs[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.update_LP(i, Ai)
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            j = i + 1
            Bj = self.psi.Bs[j]
            theta = np.tensordot(theta, Bj, axes=(1, 0))  # vL [vL'], [vL] j vR
            # here theta is one-site on site j = i + 1
        # right boundary
        i = L - 1
        theta = self.evolve_one_site(i, dt, theta)  # forward
        theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
        self.psi.Bs[i] = Bi
        self.update_RP(i, Bi)
        # sweep from right to left
        for i in reversed(range(L - 1)):
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            Ai = self.psi.Bs[i]  # still in left-canonical A form from the above right-sweep!
            theta = np.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
            self.psi.Bs[i] = Bi
            self.update_RP(i, Bi)
        # The last `evolve_one_site` brought the tensor on site 0 in right-canonical B form,
        # recovering the right-canonical form on each MPS tensor (as the SimpleMPS assumes).
        # It splitted the very left, trivial leg off theta,
        # which should only have an arbitrary phase for the left, trivial singular vector,
        # and a singular value 1 (if the state is normalized).
        assert theta.shape == (1, 1)
        assert abs(abs(theta[0]) - 1.) < 1.e-10
        # To keep track of the phase, we put it back into the tensor.
        self.psi.Bs[0] *= theta[0, 0]

    def sweep_two_site(self, dt):
        """Perform one two-site TDVP sweep to evolve |psi> -> exp(-i H_mpo dt) |psi>.

        This can grow the bond dimension, but is *not* stricly TDVP.
        """
        psi = self.psi
        L = self.psi.L
        # sweep from left to right
        theta = self.psi.get_theta2(0)
        for i in range(L - 2):
            j = i + 1
            k = i + 2
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            psi.Bs[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.update_LP(i, Ai)
            theta = np.tensordot(np.diag(S), Bj, axes=(1, 0))  # vL [vL'], [vL] j vC
            # here theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Bk = self.psi.Bs[k]
            theta = np.tensordot(theta, Bk, axes=(2, 0))  # vL j [vC], [vC] k vR
            # here theta is two-site on sites j, k = i + 1, i + 2
        # right boundary
        i = L - 2
        j = L - 1
        Ai, S, Bj = self.evolve_split_two_site(i, dt, theta)  # forward
        theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))  # vL i [vC], [vC'] vC
        self.psi.Bs[j] = Bj
        self.update_RP(j, Bj)
        # sweep from right to left
        for i in reversed(range(L - 2)):
            j = i + 1
            # here, theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Ai = self.psi.Bs[i]  # still in left-canonical A form from the above right-sweep!
            theta = np.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            # here, theta is two-site on sites i, j = i, i + 1
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            self.psi.Bs[j] = Bj
            self.update_RP(j, Bj)
            theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))  # vL i vC, [vC'] vC
        self.psi.Bs[0] = theta  # this is right-canonical, because for a finite system
        # the left-most virtual bond is trivial, so `theta` and `B` are the same on site 0.
        # So we recovered the right-canonical form on each MPS tensor (as the SimpleMPS assumes).

    def evolve_zero_site(self, i, dt, theta):
        """Evolve zero-site `theta` with SimpleHeff0 right of site `i`."""
        Heff = SimpleHeff0(self.LPs[i + 1], self.RPs[i])
        theta = np.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        # no truncation necessary!
        return np.reshape(theta, Heff.theta_shape)

    def evolve_one_site(self, i, dt, theta):
        """Evolve one-site `theta` with SimpleHeff1 on site i."""
        # get effective Hamiltonian
        Heff = SimpleHeff1(self.LPs[i], self.RPs[i], self.H_mpo[i])
        theta = np.reshape(theta, [Heff.shape[0]])
        theta = self.expm_multiply(Heff, theta, dt)
        # no truncation necessary!
        return np.reshape(theta, Heff.theta_shape)

    def evolve_split_two_site(self, i, dt, theta):
        """Evolve two-site `theta` with SimpleHeff2 on sites i and i + 1."""
        j = i + 1
        # get effective Hamiltonian
        Heff = SimpleHeff2(self.LPs[i], self.RPs[j], self.H_mpo[i], self.H_mpo[j])
        theta = np.reshape(theta, [Heff.shape[0]])  # group legs
        theta = self.expm_multiply(Heff, theta, dt)
        theta = np.reshape(theta, Heff.theta_shape)  # split legs
        # truncation necessary!
        Ai, S, Bj = a_mps.split_truncate_theta(theta, self.chi_max, self.eps)
        self.psi.Ss[j] = S
        return Ai, S, Bj

    def split_one_site_theta(self, i, theta, move_right=True):
        """Split a one-site theta into `Ai, theta` (right move) or ``theta, Bi`` (left move)."""
        chivL, d, chivR = theta.shape
        if move_right:
            # group i to the left
            theta = np.reshape(theta, [chivL * d, chivR])
            A, S, V = a_mps.svd(theta, full_matrices=False)  # vL vC, vC, vC i vR
            S /= np.linalg.norm(S)
            self.psi.Ss[i + 1] = S
            chivC = len(S)  # no truncation necessary!
            A = np.reshape(A, [chivL, d, chivC])
            theta = np.tensordot(np.diag(S), V, axes=(1, 0)) # vC [vC'], [vC] vR
            return A, theta
        else:
            # group i to the right
            theta = np.reshape(theta, [chivL, d * chivR])
            U, S, B = a_mps.svd(theta, full_matrices=False)  #  vL i vC, vC, vC vR
            S /= np.linalg.norm(S)
            self.psi.Ss[i] = S
            chivC = len(S)  # no truncation necessary!
            B = np.reshape(B, [chivC, d, chivR])
            theta = np.tensordot(U, np.diag(S), axes=(1, 0)) # vL [vC], [vC'] vC
            return theta, B

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


    def expm_multiply(self, H, psi0, dt):
        from scipy.sparse.linalg import expm_multiply
        return expm_multiply((-1.j*dt) * H, psi0)
        #  # alternatively, use custom lanczos implementation
        #  from lanczos import lanczos_expm_multiply
        #  return lanczos_expm_multiply(H, psi0, dt)


class SimpleHeff1(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian on 1 site.

    Basically the same as d_dmrg.SimpleHeff2, but acts on a single site::

        .--vL*     vR*--.
        |       i*      |
        |       |       |
       (LP)----(W1)----(RP)
        |       |       |
        |       i       |
        .--vL       vR--.
    """
    def __init__(self, LP, RP, W1, prefactor=1.):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wR i i*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1 = W1.shape[2]
        self.theta_shape = (chi1, d1, chi2)  # vL i vR
        self.shape = (chi1 * d1 * chi2, chi1 * d1 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = np.reshape(theta, self.theta_shape)  # vL i vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] vR, [wL] wR i [i*]
        x = np.tensordot(x, self.RP, axes=([1, 2], [0, 1]))  # vL [vR] [wR] i, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x


class SimpleHeff0(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian.

    Basically the same as d_dmrg.SimpleHeff1, but acts on the zero-site wave function::

        .--vL*   vR*--.
        |             |
        |             |
       (LP)----------(RP)
        |             |
        |             |
        .--vL     vR--.
    """
    def __init__(self, LP, RP, prefactor=1.):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        chi1, chi2 = LP.shape[0], RP.shape[2]
        self.theta_shape = (chi1, chi2)  # vL vR
        self.shape = (chi1 * chi2, chi1 * chi2)
        self.dtype = LP.dtype

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = np.reshape(theta, self.theta_shape)  # vL vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] vR
        x = np.tensordot(x, self.RP, axes=([1, 2], [1, 0]))  # vL [wL*] [vL*] , [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x


def example_TDVP_tf_ising_lightcone(L, g, tmax, dt, one_site=True, chi_max=50):
    # compare this code to c_tebd.example_TEBD_tf_ising_lightcone - it's almost the same.
    print("finite TEBD, real time evolution, transverse field Ising")
    print("L={L:d}, g={g:.2f}, tmax={tmax:.2f}, dt={dt:.3f}".format(L=L, g=g, tmax=tmax, dt=dt))
    # find ground state with TEBD or DMRG
    #  E, psi, model = example_TEBD_gs_tf_ising_finite(L, g)
    from d_dmrg import example_DMRG_tf_ising_finite
    E, psi, model = example_DMRG_tf_ising_finite(L, g)
    i0 = L // 2
    # apply sigmax on site i0
    SxB = np.tensordot(model.sigmaz, psi.Bs[i0], axes=(1, 1))  # i [i*], vL [i] vR
    psi.Bs[i0] = np.transpose(SxB, [1, 0, 2])  # vL i vR
    E = np.sum(psi.bond_expectation_value(model.H_bonds))
    print("E after applying Sz = {E:.13f}".format(E=E))
    eng = SimpleTDVPEngine(psi, model, chi_max=chi_max, eps=1.e-7)
    S = [psi.entanglement_entropy()]
    Nsteps = int(tmax / dt + 0.5)
    for n in range(Nsteps):
        if abs((n * dt + 0.1) % 0.2 - 0.1) < 1.e-10:
            print("t = {t:.2f}, chi =".format(t=n * dt), psi.get_chi())
        if one_site:
            eng.sweep_one_site(dt)
        else:
            eng.sweep_two_site(dt)
        S.append(psi.entanglement_entropy())
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(S[::-1],
               vmin=0.,
               aspect='auto',
               interpolation='nearest',
               extent=(0, L - 1., -0.5 * dt, (Nsteps + 0.5) * dt))
    plt.xlabel('site $i$')
    plt.ylabel('time $t/J$')
    plt.ylim(0., tmax)
    plt.colorbar().set_label('entropy $S$')
    E = np.sum(psi.bond_expectation_value(model.H_bonds))
    print("final E = {E:.13f}".format(E=E))


if __name__ == "__main__":
    example_TDVP_tf_ising_lightcone(L=20, g=1.5, tmax=3., dt=0.05, one_site=True)
