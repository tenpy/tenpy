r"""This module contains a base class for Projected Entangled Pair States (PEPS) in two dimensions

A PEPS looks roughly like this::

    |        ...        ...        ...
    |         |          |          |
    |   -- T[0][2] -- T[1][2] -- T[2][2] -- ...
    |         |          |          |
    |   -- T[0][1] -- T[1][1] -- T[2][1] -- ...
    |         |          |          |
    |   -- T[0][0] -- T[1][0] -- T[2][0] -- ...
    |         |          |          |

where each T also has a physical leg which is not shown (e.g. pointing "downward" into the screen)

We use the following label convention for the `T` (where arrows indicate `qconj`)::

    |         vU
    |         ^
    |         |
    |  vL ->- T ->- vR
    |       / |
    |      ^  ^
    |     p   vD

We store one 5-leg tensor `_T[x][y]` with labels ``'p', 'vU', 'vL', 'vD', 'vR'`` for each of the
`lx * ly` lattice sites where ``0 <= x < lx`` and ``0 <= y < ly``.

TODO explain more, e.g.
- boundary conditions (infinite, finite, TODOD: segment)
- symmetries, e.g. C4v
- normalization
    
"""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np

from ..linalg import np_conserved as npc

__all__ = []  # TODO


class PEPS:
    r"""A projected entangled pair state (PEPS), either finite (fPEPS) or infinite (iPEPS).

    Parameters
    ----------
    sites : list of list of :class:`~tenpy.networks.site.Site`
        Defines the local Hilbert space for each site
    Ts : list of list of :class:`~tenpy.linalg.np_conserved.Array`
        The tensors of the PEPS, labels are ``p, vU, vL, vD, vR`` (in any order)
    bc : ``'finite' | 'infinite'``
        Boundary conditions as descrided in the module doc-string.

    Attributes
    ----------
    TODO

    
    """

    _valid_bc = ['finite', 'infinite']
    _p_label = ['p']  # labels of phyical leg(s)
    _T_labels = ['p', 'vU', 'vL', 'vD', 'vR']  # all labels of a _T tensor (order is used!)

    def __init__(self, sites, Ts, bc='finite'):
        # TODO store a norm attribute...?
        self.sites = [list(col) for col in sites]
        self.chinfo = self.sites[0][0].leg.chinfo
        self.dtype = dtype = np.find_common_type([T.dtype for col in Ts for T in col], [])
        self.bc = bc
        self.lx = len(Ts)
        self.ly = len(Ts[0])
        self._T = [[T.astype(dtype, copy=True).itranspose(self._T_labels) for T in col] for col in Ts]
        self.test_sanity()

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        if self.bc not in self._valid_bc:
            raise ValueError(f'invalid boundary condition: {self.bc}')

        # check self.sites
        if len(self.sites) == self.lx:
            raise ValueError('wrong len of self.sites')
        wrong_length_columns = [i for i, col in enumerate(self.sites) if len(col) != self.ly]
        if wrong_length_columns:
            raise ValueError(f'wrong len of self.sites[i] for i in {wrong_length_columns}.')

        # check self._T
        #  correct list lengths
        if len(self._T) != self.lx:
            raise ValueError('wrong len of self._T')
        wrong_length_columns = [i for i, col in enumerate(self._T) if len(col) != self.ly]
        if wrong_length_columns:
            raise ValueError(f'wrong len of self._T[i] for i in {wrong_length_columns}.')
        #  correct labels and non-trivial legs
        for x, col in enumerate(self._T):
            for y, T in enumerate(col):
                if T.get_leg_labels() != self._T_labels:
                    msg = f'T at site {(x, y)} has wrong labels {T.get_leg_labels()}. Expected {self._T_labels}'
                    raise ValueError(msg)

                if self.bc == 'infinite' or x + 1 < self.lx:
                    T2 = self._T[(x + 1) % self.lx][y]
                    T.get_leg('vR').test_contractible(T2.get_leg('vL'))
                if self.bc == 'infinite' or y + 1 < self.ly:
                    T2 = self._T[x][(y + 1) % self.ly]
                    T.get_leg('vU').test_contractivle(T2.get_leg('vD'))
        #  correct trivial legs
        if self.bc == 'finite':
            for T in self._T[0]:
                if T.get_leg('vL').ind_len != 1:
                    raise ValueError('Non-trivial leg at left boundary')
            for T in self._T[-1]:
                if T.get_leg('vR').ind_len != 1:
                    raise ValueError('Non-trivial leg at right boundary')
            for col in self._T:
                if col[0].get_leg('vD').ind_len != 1:
                    raise ValueError('Non-trivial leg at bottom boundary')
            for col in self._T:
                if col[-1].get_leg('vU').ind_len != 1:
                    raise ValueError('Non-trivial leg at bottom boundary')

    def copy(self):
        """Returns a copy of `self`.

        The copy still shares the sites, chinfo, and LegCharges of the T tensors, but the values of
        T are deeply copied.
        """
        # __init__ makes deep copies of T
        cp = self.__class__(sites=self.sites, Ts=self._T, bc=self.bc)
        # TODO need to set any other attributes?
        return cp

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        raise NotImplementedError  # TODO
                    
    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        raise NotImplementedError  # TODO

    @classmethod
    def from_product_state(cls, sites, p_state, bc='finite', dtype=np.complex128, permute=True,
                           chargesL=None, chargesD=None, chargesR=None, chargesU=None):
        """Construct a PEPS from a given product state

        Parameters
        ----------
        sites : list of list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
        p_state : list of list of {int | str | 1D array}
            Defines the product state to be represented.
            An entry of `str` type is translated to an `int` via :meth:`~tenpy.networks.site.Site.state_labels`.
            An entry of `int` type represents the physical index of the state to be used.
            An entry which is a 1D array defines the complete wavefunction on that site; this
            allows to make a (local) superposition.
        bc : str, optional
            PEPS boundary conditions, see module docstring.
        dtype :
            The data type of the array entries.
        permute : bool, optional
            The :class:`~tenpy.networks.Site` might permute the local basis states if charge
            conservation gets enabled.
            If `permute` is True (default), we permute the given `p_state` locally according to
            each site's :attr:`~tenpy.networks.Site.perm`.
            The `p_state` entries should then always be given as if `conserve=None` in the Site.
        chargesL : list of list of charges, optional
            Leg charges for the left virtual legs.
        chargesD : list of list of charges, optional
            Leg charges for the right virtual legs.
        chargesR : list of charges, optional
            Only for finite boundary conditions; the charges on the right legs for the rightmost column
        chargesU : list of charges, optional
            Only for finite boundary conditions; the charges on the up legs for the uppermost column

        Returns
        -------
        product_peps : :class:`PEPS`
            A PEPS representing the specified product state.

        TODO example, doctest
        """
        sites = [list(col) for col in sites]
        lx = len(sites)
        ly = len(sites[0])
        assert all(len(col) == ly for col in sites[1:])
        chinfo = sites[0][0].leg.chinfo
        if chargesL is None:
            chargesL = [[None] * ly] * lx
        legsL = [[npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid(ch)]) for ch in col] for col in chargesL]
        if chargesD is None:
            chargesD = [[None] * ly] * lx
        legsD = [[npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid(ch)]) for ch in col] for col in chargesD]
        if bc == 'finite':
            if chargesR is None:
                chargesR = [None] * ly
            right_col = [npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid(ch)]) for ch in chargesR]
            legsR = [[l.conj() for l in col] for col in legsL[1:]] + [right_col]
            if chargesU is None:
                chargesU = [None] * lx
            top_row = [npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid(ch)]) for ch in chargesU]
            legsU = [[l.conj() for l in col[1:]] + [top_leg] for col, top_leg in zip(legsD, top_row)]
        else:
            assert chargesR is None
            assert chargesU is None
            legsR = [[l.conj() for l in col] for col in legsL[1:]] + [[l.conj() for l in legsL[0]]]
            legsU = [[l.conj() for l in col[1:]] + [col[0].conj()] for col in legsD]

        Ts = []
        for cols in zip(p_state, sites, legsU, legsL, legsD, legsR):
            T_col = []
            for p_st, site, legU, legL, legD, legR in zip(*cols):
                perm = permute
                if isinstance(p_st, str):
                    p_st = site.state_labels[p_st]
                    perm = False
                try:
                    iter(p_st)
                except TypeError:
                    is_iterable = False
                else:
                    is_iterable = True
                if is_iterable:
                    if len(p_st) != site.dim:
                        raise ValueError('p_state incompatible with local dim.')
                    T = np.array(p_st, dtype).reshape((site.dim, 1, 1, 1, 1))
                else:
                    T = np.zeros((site.dim, 1, 1, 1, 1), dtype)
                    T[p_st, 0, 0, 0, 0] = 1.0
                if perm:
                    T = T[site.perm, :, :, :, :]
                T = npc.Array.from_ndarray(T, [site.leg, legU, legL, legD, legR])
                T_col.append(T)
            Ts.append(T_col)
        return cls(sites=sites, Ts=Ts, bc=bc)

    @property
    def bc_is_infinite(self):
        assert self.bc in self._valid_bc
        return self.bc == 'infinite'

    @property
    def hor_D(self) -> np.ndarray:
        """Dimensions of the (nontrivial) horizontal virtual bonds"""
        if self.bc == 'finite':
            x_slice = slice(1, self.lx)
        else:
            x_slice = slice(0, self.lx)
        dims = [[T.shape[T.get_leg_index('vL')] for T in col] for col in self._Ts[x_slice]]
        return np.array(dims, dtype=int)

    @property
    def vert_D(self) -> np.ndarray:
        """Dimensions of the (nontrivial) vertical virtual bonds"""
        if self.bc == 'finite':
            y_slice = slice(1, self.ly)
        else:
            y_slice = slice(0, self.ly)
        dims = [[T.shape[T.get_leg_index('vD')] for T in col[y_slice]] for col in self._Ts]
        return np.array(dims, dtype=int)

    @property
    def max_D(self) -> int:
        """maximum of virtual bond dimensions"""
        return max(np.max(self.hor_D), np.max(self.vert_D))

    def get_T(self, x: int, y: int) -> npc.Array:
        return self._Ts[x][y]

    def set_T(self, x: int, y: int, T: npc.Array):
        self._Ts[x][y] = T
 