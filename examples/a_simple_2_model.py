"""Toy code implementing the transverse-field ising model."""

import numpy as np


class TFIModel(object):
    r"""Simple class generating the Hamiltonian of the transverse-field Ising model.

    The Hamiltonian reads
    .. math ::
        H = - J \sum_{i} \sigma^z_i \sigma^z_{i+1} - g \sum_{i} \sigma^x_i

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
        Each ``H_mpo[i]`` has legs (virutal left, virtual right, physical out, physical in),
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
        """Initialize `H_bonds` hamiltonian. Called by __init__()."""
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
        """Initialize `H_mpo` Hamiltonian. Called by __init__()."""
        w_list = []
        for i in range(self.L):
            w = np.zeros((3, 3, self.d, self.d), dtype=np.float)
            w[0, 0] = w[2, 2] = self.id
            w[0, 1] = self.sigmaz
            w[0, 2] = -self.g * self.sigmax
            w[1, 2] = -self.J * self.sigmaz
            w_list.append(w)
        self.H_mpo = w_list

    def exact_finite_gs_energy(self):
        """For comparison: obtain ground state energy from exact diagonalization.

        Exponentially expensive in L, only works for small enough `L` <~ 20"""
        import scipy.sparse as sparse
        import scipy.sparse.linalg.eigen.arpack as arp
        import warnings
        assert self.bc == 'finite'
        L = self.L
        if L >= 20:
            warnings.warn("Large L: Exact diagonalization might take a long time!")
        # get single site operaors
        sx = sparse.csr_matrix(self.sigmax)
        sz = sparse.csr_matrix(self.sigmaz)
        id = sparse.csr_matrix(self.id)
        sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
        sz_list = []
        for i_site in range(L):
            x_ops = [id] * L
            z_ops = [id] * L
            x_ops[i_site] = sx
            z_ops[i_site] = sz
            X = x_ops[0]
            Z = z_ops[0]
            for j in range(1, L):
                X = sparse.kron(X, x_ops[j], 'csr')
                Z = sparse.kron(Z, z_ops[j], 'csr')
            sx_list.append(X)
            sz_list.append(Z)
        H_zz = sparse.csr_matrix((2**L, 2**L))
        H_x = sparse.csr_matrix((2**L, 2**L))
        for i in range(L - 1):
            H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
        for i in range(L):
            H_x = H_x + sx_list[i]
        H = -self.J * H_zz - self.g * H_x
        E, V = arp.eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
        return E[0]

    def exact_infinite_gs_energy(self):
        """For comparison: Calculate groundstate energy from analytic formula."""
        import scipy.integrate

        def f(k, g):
            return -2 * np.sqrt(1 + g**2 - 2 * g * np.cos(k)) / np.pi / 2.

        E0_exact = scipy.integrate.quad(f, 0, np.pi, args=(self.g / self.J, ))[0]
        return E0_exact
