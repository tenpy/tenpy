"""Toy code implementing the transverse-field ising model."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np


class TFIModel:
    """Simple class generating the Hamiltonian of the transverse-field Ising model.

    The Hamiltonian reads
    .. math ::
        H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i

    Parameters
    ----------
    L : int
        Number of sites.
    J, g : float
        Coupling parameters of the above defined Hamiltonian.
    bc : 'infinite', 'finite'
        Boundary conditions.

    Attributes
    ----------
    L : int
        Number of sites.
    bc : 'infinite', 'finite'
        Boundary conditions.
    sigmax, sigmay, sigmaz, id :
        Local operators, namely the Pauli matrices and identity.
    H_bonds : list of np.Array[ndim=4]
        The Hamiltonian written in terms of local 2-site operators, ``H = sum_i H_bonds[i]``.
        Each ``H_bonds[i]`` has (physical) legs (i out, (i+1) out, i in, (i+1) in),
        in short ``i j i* j*``.
    H_mpo : lit of np.Array[ndim=4]
        The Hamiltonian written as an MPO.
        Each ``H_mpo[i]`` has legs (virtual left, virtual right, physical out, physical in),
        in short ``wL wR i i*``.
    """
    def __init__(self, L, J, g, bc='finite'):
        assert bc in ['finite', 'infinite']
        self.L, self.d, self.bc = L, 2, bc
        self.J, self.g = J, g
        self.sigmax = np.array([[0., 1.], [1., 0.]])
        self.sigmay = np.array([[0., -1j], [1j, 0.]])
        self.sigmaz = np.array([[1., 0.], [0., -1.]])
        self.id = np.eye(2)
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        sx, sz, id = self.sigmax, self.sigmaz, self.id
        d = self.d
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        H_list = []
        for i in range(nbonds):
            gL = gR = 0.5 * self.g
            if self.bc == 'finite':
                if i == 0:
                    gL = self.g
                if i + 1 == self.L - 1:
                    gR = self.g
            H_bond = -self.J * np.kron(sz, sz) - gL * np.kron(sx, id) - gR * np.kron(id, sx)
            # H_bond has legs ``i, j, i*, j*``
            H_list.append(np.reshape(H_bond, [d, d, d, d]))
        self.H_bonds = H_list

    # (note: not required for TEBD)
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        w_list = []
        for i in range(self.L):
            w = np.zeros((3, 3, self.d, self.d), dtype=float)
            w[0, 0] = w[2, 2] = self.id
            w[0, 1] = self.sigmaz
            w[0, 2] = -self.g * self.sigmax
            w[1, 2] = -self.J * self.sigmaz
            w_list.append(w)
        self.H_mpo = w_list
