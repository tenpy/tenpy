"""Toy code implementing a matrix product state."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np

import cyten as ct


class SimpleMPS:
    """Simple class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We *assume* that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss, bc:
        Same as attributes.

    Attributes
    ----------
    Bs : list of SymmetricTensor
        MPS tensors in B form. Labels ``vL, p, vR``
    Ss : list of DiagonalTensor
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``. Labels ``vL, vR``.
    bc : 'infinite', 'finite'
        Boundary conditions.
    L : int
        Number of sites (in the unit-cell for an infinite MPS).
    nbonds : int
        Number of (non-trivial) bonds: L-1 for 'finite' boundary conditions, L for 'infinite'.

    """

    def __init__(self, Bs: list[ct.SymmetricTensor], Ss: list[ct.DiagonalTensor], bc='finite'):
        assert bc in ['finite', 'infinite']
        self.symmetry = Bs[0].symmetry
        self.Bs = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.backend = ct.backends.get_same_backend(*Bs, *Ss)
        self.nbonds = self.L - 1 if self.bc == 'finite' else self.L

    def copy(self):
        return SimpleMPS(self.Bs[:], self.Ss[:], self.bc)

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs labels ``vL, p, vR``
        """
        return ct.scale_axis(self.Bs[i], self.Ss[i], 'vL')

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        Bj = ct.permute_legs(self.Bs[j], codomain=['vL'], bend_right=True)
        return ct.tdot(self.get_theta1(i), Bj, 'vR', 'vL', relabel1=dict(p='p0'), relabel2=dict(p='p1'))

    def get_chi(self):
        """Return bond dimensions."""
        return [sum(self.Bs[i].get_leg_co_domain('vR').multiplicities) for i in range(self.nbonds)]

    def site_expectation_value(self, op: ct.Tensor):
        """Calculate expectation values of a local operator at each site.

        op has labels ``p, p*`` in any order.
        """
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = ct.tdot(op, theta, 'p*', 'p')
            result.append(ct.tdot(theta.hc, op_theta, ['vL*', 'p*', 'vR*'], ['vL', 'p', 'vR']))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op: list[ct.Tensor]):
        """Calculate expectation values of a local operator at each bond.

        op has labels ``p0, p1, p0*, p1*`` in any order.
        """
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = ct.tdot(op[i], theta, ['p0*', 'p1*'], ['p0', 'p1'])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(ct.tdot(theta.hc, op_theta, ['vL*', 'p0*', 'p1*', 'vR*'], ['vL', 'p0', 'p1', 'vR']))
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        return [ct.entropy(self.Ss[i]) for i in bonds]

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        raise NotImplementedError  # TODO

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i)
        C = ct.tdot(op_i, theta, 'p*', 'p')
        C = ct.tdot(theta.dagger, C, ['vL*', 'p*'], ['vL', 'p'])
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]
            C = ct.tdot(C, B, 'vR', 'vL')
            C = ct.tdot(B.hc, C, ['vL*', 'p*'], ['vL', 'p'])
        B = self.Bs[j % self.L]
        C = ct.tdot(C, B, 'vR', 'vL')
        C = ct.tdot(op_j, C, 'p*', 'p')
        C = ct.tdot(B.hc, C, ['vL*', 'p*', 'vR*'], ['vL', 'p', 'vR'])
        return ct.item(C)


def init_FM_MPS(L, d=2, bc='finite', backend='abelian', conserve='none'):
    """Return a ferromagnetic MPS (= product state with all spins up)"""
    if backend in ['no_symmetry', 'abelian', 'fusion_tree']:
        backend = ct.get_backend(backend, 'numpy')
    B = np.array([1] + [0] * (d - 1), float)[None, :, None]
    if conserve == 'none':
        p = ct.ElementarySpace.from_trivial_sector(d)
        v = ct.ElementarySpace.from_trivial_sector(1)
        B = ct.tensor(B, [v, p], [v], labels=['vL', 'p', 'vR'], backend=backend)
        S = ct.eye(v, backend=backend, labels=['vL', 'vR'])
    elif conserve == 'Z2':
        sym = ct.ZNSymmetry(2, 'Sz_parity')
        p = ct.ElementarySpace.from_basis(sym, np.arange(d)[:, None] % 2)
        v = ct.ElementarySpace.from_trivial_sector(1, sym)
        B = ct.SymmetricTensor.from_dense_block(B, [v, p], [v], labels=['vL', 'p', 'vR'], backend=backend)
        S = ct.eye(v, backend=backend, labels=['vL', 'vR'])
    else:
        raise ValueError
    return SimpleMPS([B] * L, [S] * L, bc=bc)


def init_Neel_MPS(L, d=2, bc='finite', backend='abelian', conserve='none'):
    """Return a Neel state MPS (= product state with alternating spins up  down up down... )"""
    assert bc == 'finite' or L % 2 == 0
    if backend in ['no_symmetry', 'abelian', 'fusion_tree']:
        backend = ct.get_backend(backend, 'numpy')
    B1 = np.array([1] + [0] * (d - 1), float)[None, :, None]
    B2 = np.array([0] * (d - 1) + [1], float)[None, :, None]
    if conserve == 'none':
        p = ct.ElementarySpace.from_trivial_sector(d)
        v = ct.ElementarySpace.from_trivial_sector(1)
        B1 = ct.SymmetricTensor.from_dense_block(B1, [v, p], [v], labels=['vL', 'p', 'vR'], backend=backend)
        B2 = ct.SymmetricTensor.from_dense_block(B2, [v, p], [v], labels=['vL', 'p', 'vR'], backend=backend)
        S = ct.DiagonalTensor.from_eye(v, backend=backend, labels=['vL', 'vR'])
        B_list = [B1, B2] * (L // 2) + [B1] * (L % 2)
        S_list = [S] * L
    elif conserve == 'Z2':
        sym = ct.ZNSymmetry(2, 'Sz_parity')
        p = ct.ElementarySpace.from_basis(sym, np.arange(d)[:, None] % 2)
        v1 = ct.ElementarySpace.from_trivial_sector(1, sym)
        v2 = ct.ElementarySpace.from_defining_sectors(sym, [[1]])
        v2.test_sanity()
        B11 = ct.SymmetricTensor.from_dense_block(B1, [v1, p], [v1], labels=['vL', 'p', 'vR'], backend=backend)
        B12 = ct.SymmetricTensor.from_dense_block(B1, [v2, p], [v2], labels=['vL', 'p', 'vR'], backend=backend)
        B21 = ct.SymmetricTensor.from_dense_block(B2, [v1, p], [v2], labels=['vL', 'p', 'vR'], backend=backend)
        B22 = ct.SymmetricTensor.from_dense_block(B2, [v2, p], [v1], labels=['vL', 'p', 'vR'], backend=backend)
        S1 = ct.DiagonalTensor.from_eye(v1, backend=backend, labels=['vL', 'vR'])
        S2 = ct.DiagonalTensor.from_eye(v2, backend=backend, labels=['vL', 'vR'])
        B_list = [B11, B21, B12, B22]
        B_list = B_list * (L // 4) + B_list[: L % 4]
        S_list = [S1, S1, S2, S2]
        S_list = S_list * (L // 4) + S_list[: L % 4]
    else:
        raise ValueError
    return SimpleMPS(B_list, S_list, bc=bc)


def init_SU2_sym_MPS(L, d=2, bc='finite', backend=None):
    """Return the simplest SU(2) symmetric MPS (neighboring spins form singlets)"""
    assert L % 2 == 0
    if backend is None:
        backend = ct.get_backend('fusion_tree', 'numpy')
    sym = ct.SU2Symmetry('spin')
    p = ct.ElementarySpace.from_defining_sectors(sym, [[d - 1]])
    v1 = ct.ElementarySpace.from_trivial_sector(1, sym)
    v2 = p
    B1 = ct.SymmetricTensor.from_block_func(
        lambda x: np.ones(x), [v1, p], [v2], labels=['vL', 'p', 'vR'], backend=backend
    )
    B2 = ct.SymmetricTensor.from_block_func(
        lambda x: np.ones(x), [v2, p], [v1], labels=['vL', 'p', 'vR'], backend=backend
    )
    S1 = ct.DiagonalTensor.from_eye(v1, backend=backend, labels=['vL', 'vR'])
    S2 = ct.DiagonalTensor.from_eye(v2, backend=backend, labels=['vL', 'vR'])
    return SimpleMPS([B1, B2] * (L // 2), [S1, S2] * (L // 2), bc=bc)


def init_Fib_anyon_MPS(L, bc='finite', backend=None):
    """Return the Fib anyon symmetric MPS with tau charges on all bonds."""
    if backend is None:
        backend = ct.get_backend('fusion_tree', 'numpy')
    sym = ct.fibonacci_anyon_category
    p = ct.ElementarySpace.from_defining_sectors(sym, [[1]])
    v = p
    B = ct.SymmetricTensor.from_block_func(
        lambda x: np.ones(x, dtype=complex), [v, p], [v], labels=['vL', 'p', 'vR'], backend=backend
    )
    S = ct.DiagonalTensor.from_eye(v, backend=backend, labels=['vL', 'vR'])
    return SimpleMPS([B] * L, [S] * L, bc=bc)


def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : SymmetricTensor
        Two-site wave function in mixed canonical form, with legs ``vL, p0, p1, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : SymmetricTensor
        Left-canonical matrix on site i, with legs ``vL, p, vR``
    S : DiagonalTensor
        Singular/Schmidt values with legs ``vL, vR``.
    B : SymmetricTensor
        Right-canonical matrix on site j, with legs ``vL, p, vR``

    """
    A, S, B, _, _ = ct.truncated_svd(theta, ['vR', 'vL'], chi_max=chi_max, svd_min=eps)
    B = ct.permute_legs(B, codomain=['vL', 'p1'], bend_right=True)
    A.relabel({'p0': 'p'})
    B.relabel({'p1': 'p'})
    return A, S, B
