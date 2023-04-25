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
from ..linalg.np_conserved import Array
from ..networks.site import Site

__all__ = []  # TODO


class PEPSLikeIndexable:
    _valid_bc = ['finite', 'infinite']
    _p_labels = ['p']  # labels of phyical leg(s)
    _tensor_labels = ['p', 'vU', 'vL', 'vD', 'vR']  # all labels of a _tensor (order is used!)
    
    def __init__(self, sites: list[Site], tensors: list[Array], lx: int, ly: int, bc: str = 'finite'):
        # TODO doc: flat lists: tensors[x * ly + y]
        
        self.sites = sites
        self.chinfo = self.sites[0].leg.chinfo  # TODO check others?
        self.dtype = dtype = np.find_common_type([t.dtype for t in tensors], [])
        self.bc = bc
        self.lx = lx
        self.ly = ly
        self._tensors = [T.astype(dtype, copy=True).itranspose(self._tensor_labels) for T in tensors]
        self.test_sanity()

    @classmethod
    def from_list_of_lists(cls, sites: list[list[Site]], tensors: list[list[Array]], bc: str = 'finite'):
        lx = len(sites)
        ly = len(sites[0])
        assert all(len(col) == ly for col in sites)
        assert len(tensors) == lx
        assert all(len(col) == ly for col in tensors)
        return cls(
            sites=[s for col in sites for s in col],
            tensors=[t for col in tensors for t in col],
            lx=lx, ly=ly, bc=bc
        )

    @property
    def num_sites(self):
        return self.lx * self.ly

    def _parse_x(self, x: int) -> int:
        if x < 0:
            x = x + self.lx
        if not 0 <= x < self.lx:
            raise ValueError(f'x index out of bounds: {x}')
        return x

    def _parse_y(self, y: int) -> int:
        if y < 0:
            y = y + self.ly
        if not 0 <= y < self.ly:
            raise ValueError(f'y index out of bound: {y}')
        return y

    def _coords_to_idx(self, x: int, y: int) -> int:
        return self._parse_x(x) * self.ly + self._parse_y(y)

    def _idx_to_coords(self, i: int) -> tuple[int, int]:
        if i < 0:
            i = i + self.num_sites
        assert 0 <= i < self.num_sites
        return divmod(i, self.ly)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        if self.bc not in self._valid_bc:
            raise ValueError(f'invalid boundary condition: {self.bc}')

        if len(self.sites) != self.num_sites:
            msg = f'Wrong len of sites. Expected {self.lx} * {self.ly} = {self.num_sites}. Got {len(self.sites)}'
            raise ValueError(msg)
        if len(self.tensors) != self.num_sites:
            msg = f'Wrong len of tensors. Expected {self.lx} * {self.ly} = {self.num_sites}. Got {len(self.tensors)}'
            raise ValueError(msg)

        for i, tens in enumerate(self._tensors):
            x, y = self._idx_to_coords(i)

            if tens.get_leg_labels() != self._tensors_labels:
                msg = f'tensor at site {(x, y)} has wrong labels {T.get_leg_labels()}. Expected {self._tensors_labels}'
                raise ValueError(msg)

            if self.bc == 'infinite' or x > 0:  # check the bonds between unit cells (x==0) only for infinite
                tens.get_leg('vL').test_contractible(self[x - 1, y].get_leg('vR'))
            if self.bc == 'infinite' or y > 0:
                tens.get_leg('vD').test_contractible(self[x, y - 1].get_leg('vU'))

        #  for finite system; check boundary legs are trivial
        if self.bc == 'finite':
            for boundary, leg in zip(self[0, :], self[-1, :], self[:, 0], self[:, -1],
                                     ['vL', 'vR', 'vD', 'vU']):
                for tens in boundary:
                    if tens.get_leg(leg).ind_len != 1:
                        raise ValueError(f'Non-trivial {leg} leg at boundary')

    def copy(self):
        """Returns a copy of `self`.

        The copy still shares the sites, chinfo, and LegCharges of the T tensors, but the values of
        T are deeply copied.
        """
        # __init__ makes deep copies of tensors
        cp = self.__class__(sites=self.sites, tensors=self._tensors, bc=self.bc)
        return cp

    @property
    def bc_is_infinite(self):
        assert self.bc in self._valid_bc
        return self.bc == 'infinite'

    @property
    def hor_bond_dims(self) -> np.ndarray:
        """Dimensions of the (nontrivial) horizontal virtual bonds"""
        if self.bc == 'finite':
            # omit x == 0 sites, i.e. indices which are multiples of ly
            dims = [t.shape[t.get_leg_index['vL']] for i, t in enumerate(self._tensors) if i % self.ly != 0]
        else:
            dims = [t.shape[t.get_leg_index['vL']] for t in self._tensors]
        return np.array(dims, dtype=int)

    @property
    def vert_bond_dims(self) -> np.ndarray:
        """Dimensions of the (nontrivial) vertical virtual bonds"""
        if self.bc == 'finite':
            # omit y == 0 sites, i.e. indices < ly
            dims = [t.shape[t.get_leg_index['vL']] for i, t in enumerate(self._tensors) if i >= self.ly]
        else:
            dims = [t.shape[t.get_leg_index['vL']] for t in self._tensors]
        return np.array(dims, dtype=int)

    @property
    def bond_dim(self) -> int:
        """maximum of virtual bond dimensions"""
        return max(np.max(self.hor_D), np.max(self.vert_D))

    def _parse_item(self, x: int | slice, y: int | slice = None, *rest) -> int | slice:
        if rest:
            raise IndexError('too many indices for PEPS')
        
        if y is None:
            y = slice(0, self.ly, 1)

        if isinstance(x, int):
            x = self._parse_x(x)
            if isinstance(y, int):
                return x * self.ly + self._parse_y(y)
            if isinstance(y, slice):
                return slice(x * self.ly + self._parse_y(y.start),
                             x * self.ly + self._parse_y(y.stop),
                             y.step)
        elif isinstance(x, slice):
            if isinstance(y, int):
                y = self._parse_y(y)
                return slice(self._parse_x(x.start) + y,
                             self._parse_x(x.stop) + y,
                             self.ly * x.step)

        # all valid cases were covered above and have already returned
        raise TypeError('PEPS indices must be int, (int, slice) or (slice, int)')
                
    def __getitem__(self, item):
        idcs = self._parse_item(*item)
        return self._tensors[idcs]

    def __setitem__(self, item, value):
        idcs = self._parse_item(*item)
        if isinstance(idcs, slice):
            try:
                valid = all(t.get_leg_labels() == self._tensor_labels for t in value)
            except (TypeError, AttributeError):  # iteration failed or elements dont have get_leg_labels()
                raise TypeError(f'Expected iterable of Array. Got {type(value)}')
        else:
            try:
                valid = value.get_leg_labels() == self._tensor_labels
            except AttributeError:
                raise TypeError(f'Expected iterable of Array. Got {type(value)}')
        if not valid:
            raise ValueError('Invalid legs.')
        self._tensors[idcs] = value
        

class PEPS(PEPSLikeIndexable):
    r"""A projected entangled pair state (PEPS), either finite (fPEPS) or infinite (iPEPS).

    TODO (JU) : normalization of tensors? store norm seperately?

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
    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        raise NotImplementedError  # TODO (JU) can implement in PEPSLikeIndexable?
                    
    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        raise NotImplementedError  # TODO (JU) can implement in PEPSLikeIndexable?

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
        raise NotImplementedError  # FIXME update to flat list
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


class PEPO(PEPSLikeIndexable):
    _p_labels = ['p', 'p*']
    _tensor_labels = ['p', 'p*', 'vU', 'vL', 'vD', 'vR']
    