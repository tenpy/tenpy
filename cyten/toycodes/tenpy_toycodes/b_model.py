"""Toy code implementing the transverse-field ising model."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import scipy.sparse as scisp
from scipy.sparse.linalg import eigsh

import cyten as ct


class TFIModel:
    """TFI: -J XX - g Z"""

    def __init__(
        self,
        L: int,
        J: float,
        g: float,
        bc: str = 'finite',
        conserve: str = 'none',
        backend: ct.backends.TensorBackend | None = None,
    ):
        assert bc in ['finite', 'infinite']
        self.site = ct.sites.SpinSite(S=0.5, conserve=conserve, backend=backend)
        self.backend = backend
        self.symmetry = self.site.symmetry
        self.L = L
        self.bc = bc
        self.J = J
        self.g = g
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        p = self.site

        # OPTIMIZE this currently constructs the tensors, splits them for the coupling
        # and then recomputes the tensor again in order to form a superposition...
        # factors 2 and 4 due to difference between spin and Pauli matrices
        XX = ct.couplings.spin_spin_coupling(sites=[p, p], Jx=4).to_tensor()
        Z = ct.couplings.spin_field_coupling(sites=[p], hz=2).to_tensor()
        I = ct.SymmetricTensor.from_eye([p.leg], labels=['p'], backend=self.backend)
        IZ = ct.outer(I, Z, {'p': 'p0', 'p*': 'p0*'}, {'p0': 'p1', 'p0*': 'p1*'})
        ZI = ct.outer(Z, I, None, {'p': 'p1', 'p*': 'p1*'})

        H_list = []
        for i in range(nbonds):
            gL = gR = 0.5 * self.g
            if self.bc == 'finite':
                if i == 0:
                    gL = self.g
                if i + 1 == self.L - 1:
                    gR = self.g
            H_list.append(-self.J * XX - gL * ZI - gR * IZ)
        self.H_bonds = H_list

    # (note: not required for TEBD)
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        p = self.site
        XX = ct.couplings.spin_spin_coupling(sites=[p, p], Jx=4)
        Z = ct.couplings.spin_field_coupling(sites=[p], hz=2)
        I = ct.SymmetricTensor.from_eye([p.leg], labels=['p0'], backend=self.backend)
        I = ct.Coupling.from_tensor(I, [p])

        grid = [
            [I.factorization[0], -self.J * XX.factorization[0], -self.g * Z.factorization[0]],
            [None, None, XX.factorization[1]],
            [None, None, I.factorization[0]],
        ]
        W = ct.tensors.tensor_from_grid(grid, labels=['wL', 'p', 'wR', 'p*'])
        self.H_mpo = [W] * self.L


class HeisenbergModel:
    """J (XX + YY + ZZ)"""

    def __init__(
        self,
        L: int,
        J: float,
        bc: str = 'finite',
        conserve: str = 'none',
        backend: ct.backends.TensorBackend | None = None,
    ):
        assert bc in ['finite', 'infinite']
        self.site = ct.sites.SpinSite(S=0.5, conserve=conserve, backend=backend)
        self.backend = backend
        self.symmetry = self.site.symmetry
        self.L = L
        self.bc = bc
        self.J = J
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        p = self.site
        SdotS = ct.couplings.spin_spin_coupling(sites=[p, p], Jx=4, Jy=4, Jz=4).to_tensor()
        self.H_bonds = [self.J * SdotS] * nbonds

    # (note: not required for TEBD)
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        p = self.site
        SdotS = ct.couplings.spin_spin_coupling(sites=[p, p], Jx=4, Jy=4, Jz=4)
        I = ct.SymmetricTensor.from_eye([p.leg], labels=['p0'], backend=self.backend)
        I = ct.Coupling.from_tensor(I, [p])
        grid = [
            [I.factorization[0], self.J * SdotS.factorization[0], None],
            [None, None, SdotS.factorization[1]],
            [None, None, I.factorization[0]],
        ]
        W = ct.tensors.tensor_from_grid(grid, labels=['wL', 'p', 'wR', 'p*'])
        self.H_mpo = [W] * self.L


class GoldenChainModel:
    r"""-J P^{\tau \tau}_1 (projector of two neighboring Fibonacci anyons onto their trivial fusion channel)"""

    def __init__(self, L: int, J: float, bc: str = 'finite', backend: ct.backends.TensorBackend | None = None):
        assert bc in ['finite', 'infinite']
        self.site = ct.sites.GoldenSite('left', backend=backend)
        self.backend = backend
        self.symmetry = self.site.symmetry
        self.L = L
        self.bc = bc
        self.J = J
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        p = self.site
        P1 = ct.couplings.gold_coupling([p, p]).to_tensor()
        self.H_bonds = [self.J * P1] * nbonds

    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        p = self.site
        P1 = ct.couplings.gold_coupling([p, p])
        I = ct.SymmetricTensor.from_eye([p.leg], labels=['p0'], backend=self.backend)
        I = ct.Coupling.from_tensor(I, [p])
        grid = [
            [I.factorization[0], self.J * P1.factorization[0], None],
            [None, None, P1.factorization[1]],
            [None, None, I.factorization[0]],
        ]
        W = ct.tensors.tensor_from_grid(grid, labels=['wL', 'p', 'wR', 'p*'])
        self.H_mpo = [W] * self.L


def tfi_finite_gs_energy(L: int, J: float, g: float) -> float:
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    # get single site operaors
    sx = scisp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sz = scisp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    id = scisp.csr_matrix(np.eye(2))
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
            X = scisp.kron(X, x_ops[j], 'csr')
            Z = scisp.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_xx = scisp.csr_matrix((2**L, 2**L))
    H_z = scisp.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
    for i in range(L):
        H_z = H_z + sz_list[i]
    H = -J * H_xx - g * H_z
    E, V = eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    return E[0]


def heisenberg_finite_gs_energy(L: int, J: float) -> float:
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    # get single site operaors
    sx = scisp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sy = scisp.csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]]))
    sz = scisp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    id = scisp.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sy_list = []
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        y_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        y_ops[i_site] = sy
        z_ops[i_site] = sz
        X = x_ops[0]
        Y = y_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = scisp.kron(X, x_ops[j], 'csr')
            Y = scisp.kron(Y, y_ops[j], 'csr')
            Z = scisp.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sy_list.append(Y)
        sz_list.append(Z)
    H = scisp.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H += sx_list[i] * sx_list[(i + 1) % L]
        H += sy_list[i] * sy_list[(i + 1) % L]
        H += sz_list[i] * sz_list[(i + 1) % L]
    H *= J
    E, V = eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    return E[0]
