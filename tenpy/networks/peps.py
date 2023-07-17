r"""This module contains a base class for Projected Entangled Pair States (PEPS) in two dimensions

A PEPS looks roughly like this, here for a ``3 x 3`` system or infinite system with ``3 x 3``
unit cell::

    |        |       |       |
    |   -- T[2] -- T[5] -- T[8] --
    |        |       |       |
    |   -- T[1] -- T[4] -- T[7] --
    |        |       |       |
    |   -- T[0] -- T[3] -- T[6] --
    |        |       |       |

where each T also has a physical leg which is not shown (e.g. pointing "downward" into the screen)

We use the following label convention for the `T` (where arrows indicate `qconj`)::

    |         vU
    |         ^
    |         |
    |  vL ->- T ->- vR
    |       / |
    |      ^  ^
    |     p   vD

We store one 5-leg tensor with labels ``'p', 'vU', 'vL', 'vD', 'vR'`` for each of the `lx * ly` lattice sites.

TODO explain more, e.g.
- boundary conditions (infinite, finite, TODO: segment?)
- symmetries, e.g. C4v
- normalization
    
"""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np
import copy

from ..linalg import np_conserved as npc
from ..linalg.np_conserved import Array
from ..networks.site import Site
from .mpo import MPO

__all__ = ['PEPSLike', 'PEPS', 'PEPO']


class PEPSLike:
    """Class with common features of PEPS-like tensor networks, e.g. PEPS, PEPO"""
    _valid_bc = ['finite', 'infinite']
    _p_labels = ['p']  # labels of phyical leg(s)
    _up_label = 'vU'
    _left_label = 'vL'
    _down_label = 'vD'
    _right_label = 'vR'
    _tensor_labels = ['p', 'vU', 'vL', 'vD', 'vR']  # all labels of a _tensor (order is used!)]
    
    def __init__(self, sites: list[Site], tensors: list[Array], lx: int, ly: int, bc: str = 'finite'):
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
                msg = f'tensor at site {(x, y)} has wrong labels {tens.get_leg_labels()}. '\
                      f'Expected {self._tensors_labels}.'
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
        # helper for __getitem__ and __setitem__
        # using `peps[item]` results in a call `_parse_item(*item)`.
        # the return is an int or slice, which is used to index the flat list self._tensors
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

    def to_dense(self) -> npc.Array:
        if self.bc != 'finite':
            raise RuntimeError
        if self.lx * self.ly > 49:
            raise RuntimeError

        # bottom left corner
        res = self._tensors[0].squeeze([self._down_label, self._left_label])
        res.ireplace_labels(self._p_labels, [f'{p}_00' for p in self._p_labels])

        # left column
        for y in range(1, self.ly):
            tens = self._tensors[y].squeeze(self._left_label)
            res = npc.tensordot(res, tens, (self._up_label, self._down_label))
            res.ireplace_labels(self._p_labels + [self._right_label], 
                                [f'{p}_0{y}' for p in self._p_labels] + [f'{self._right_label}_{y}'])
        res = res.squeeze(self._up_labels)

        # other columns
        for x in range(1, self.lx):
            tens = self._tensors[x * self.ly].squeez(self._down_label)
            res = npc.tensordot(res, tens, (f'{self._right_label}_0', self._left_label))
            res.ireplace_labels(self._p_labels + [self._right_label], 
                                [f'{p}_{x}0' for p in self._p_labels] + [f'{self._right_label}_{y}'])

            for y in range(1, self.ly):
                tens = self._tensors[x * self.ly + y]
                res = npc.tensordot(res, tens, ([self._up_label, f'{self._right_label}_{y}'], 
                                                [self._down_label, self._left_label]))
                res.ireplace_labels(self._p_labels + [self._right_label], 
                                [f'{p}_{x}{y}' for p in self._p_labels] + [f'{self._right_label}_{y}'])
            res = res.squeeze(self._up_labels)

        res = res.squeeze([f'{self._right_label}_{y}' for y in range(self.ly)])
        return res

    def copy(self):
        return copy.copy(self)

    def rotate90(self, clockwise: bool = True):
        old_labels = [self._up_label, self._left_label, self._down_label, self._right_label]
        if clockwise:
            new_labels = [self._right_label, self._up_label, self._left_label, self._down_label]
        else:
            new_labels = [self._left_label, self._down_label, self._right_label, self._up_label]
        new_lx = self.ly
        new_ly = self.lx
        tensors = []
        sites = []
        for new_x in range(new_lx):
            for new_y in range(new_ly):
                if clockwise:
                    # (old_x, old_y) = (old_lx - new_y, new_x)
                    old_idx = (self.lx - new_y) * self.ly + new_x
                else:
                    # (old_x, old_y) = (new_y, old_ly - new_x)
                    old_idx = new_y * self.ly + self.ly - new_x
                tens = self._tensors[old_idx].replace_labels(old_labels, new_labels)
                tensors.append(tens)
                sites.append(self.sites[old_idx])
        cp = self.copy()
        cp.tensors = tensors
        cp.sites = sites
        cp.lx = new_lx
        cp.ly = new_ly
        return cp

    def rotate180(self):
        old_labels = [self._up_label, self._left_label, self._down_label, self._right_label]
        new_labels = [self._down_label, self._right_label, self._up_label, self._left_label]
        tensors = []
        sites = []
        for new_x in range(self.lx):
            for new_y in range(self.ly):
                # (old_x, old_y) = (lx - new_x, ly - new_y)
                old_idx = (self.lx - new_x) * self.ly + self.ly - new_y
                tens = self._tensors[old_idx].replace_labels(old_labels, new_labels)
                tensors.append(tens)
                sites.append(self.sites[old_idx])
        cp = self.copy()
        cp.tensors = tensors
        cp.sites = sites
        return cp
        

class PEPS(PEPSLike):
    r"""A projected entangled pair state (PEPS), either finite (fPEPS) or infinite (iPEPS).

    TODO (JU) : normalization of tensors? store norm seperately?

    Parameters
    ----------
    sites : list of :class:`~tenpy.networks.site.Site`
        Defines the local Hilbert space for each site
    tensors : list of :class:`~tenpy.linalg.np_conserved.Array`
        The tensors of the PEPS, labels are ``p, vU, vL, vD, vR``.
        If the legs are not in the above order, the tensors are transposed to match it.
        The tensor at site ``(x, y)`` is ``tensors[x * ly + y]``.
    lx : int
        Horizontal size of the system (for finite bc) or unit cell (for infinite bc)
    ly : int
        Vertival size
    bc : ``'finite' | 'infinite'``
        Boundary conditions as descrided in the module doc-string.

    Attributes
    ----------
    sites
    chinfo
    dtype
    bc
    lx
    ly
    _tensors
    _factor : float
        A factor. A PEPS represents the state given by its tensors, scaled by _factor.
    
    """
    def __init__(self, sites: list[Site], tensors: list[Array], lx: int, ly: int, bc: str = 'finite'):
        self._factor = 1.
        PEPSLike.__init__(self, sites, tensors, lx, ly, bc)
    
    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        raise NotImplementedError  # TODO (JU) can implement in PEPSLike?
                    
    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        raise NotImplementedError  # TODO (JU) can implement in PEPSLike?

    @classmethod
    def from_product_state(cls, sites: list[Site], p_state: list[int | str | np.ndarray], 
                           lx: int, ly: int, bc: str = 'finite', dtype=np.complex128, 
                           permute: bool = True):
        """Construct a PEPS from a given product state

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
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

        Returns
        -------
        product_peps : :class:`PEPS`
            A PEPS representing the specified product state.

        TODO example, doctest
        """
        chinfo = sites[0].leg.chinfo
        leg_L = leg_D = npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid(None)])
        leg_R = leg_U = leg_L.conj()
        tensors = []
        for site, state in zip(sites, p_state):
            do_permute = permute
            if isinstance(state, str):
                state = site.state_labels[state]
                do_permute = False

            try:  # TODO make this a function in tools.misc?
                iter(state)
            except TypeError:
                is_iterable = False
            else:
                is_iterable = True

            if is_iterable:
                if len(state) != site.dim:
                    raise ValueError('p_state incompatible with local dim.')
                T = np.array(state, dtype).reshape((site.dim, 1, 1, 1, 1))
            else:
                T = np.zeros((site.dim, 1, 1, 1, 1), dtype)
                T[state, 0, 0, 0, 0] = 1.0

            if do_permute:
                T = T[site.perm, :, :, :, :]
            T = npc.Array.from_ndarray(T, [site.leg, leg_U, leg_L, leg_D, leg_R], 
                                       labels=['p', 'vU', 'vL', 'vD', 'vR'])
            tensors.append(T)
        return cls(sites=sites, tensors=tensors, lx=lx, ly=ly, bc=bc)

    def normalize_tensors(self, norm_ord=2, preserve_norm: bool = False):
        """

        Parameters
        ----------
        norm_ord : float | ``'inf'`` | ``'transfer'``
            Normalizes the tensors to `npc.norm(T, norm_ord) == 1`.
            ``transfer`` instead normalizes the dominant singular value of T @ T* as a matrix
            from [(vL.vL*.vD.vD*), (vR.vR*.vU.vU*) to 1.
        preserve_norm : bool
            If self._factor should be updated accordingly, to preserve the norm of self
        """
        factors = []
        for n, tens in enumerate(self._tensors):
            if norm_ord == 'transfer':
                raise NotImplementedError
            else:
                norm = npc.norm(tens, ord=norm_ord)
            self._tensors[n] = tens / norm
            factors.append(norm)
        if preserve_norm:
            self._factor = self._factor * np.prod(factors)


class PEPO(PEPSLike):
    _p_labels = ['p', 'p*']
    _up_label = 'wU'
    _left_label = 'wL'
    _down_label = 'wD'
    _right_label = 'wR'
    _tensor_labels = ['p', 'p*', 'wU', 'wL', 'wD', 'wR']

    @classmethod
    def from_nearest_neighbors(cls, site: Site, lx: int, ly: int, 
                               horizontal: bool, bond_pos: int,
                               a: float = 1., A: str | npc.Array = None, 
                               b: float = 1., B: str | npc.Array = None,
                               ):
        """
        Generate all horizontal (or all vertical) terms of a nearest neighbor model with finite bc.

        TODO can generalize this quite a bit, can form sums of MPOs, *if* they have the usual grid.
        Could e.g. add MPOGraphs.

        The resulting operator is

        .. math ::

            a \sum_{<i,j>_hor} A_j A_j + b/2 \sum_i B_i

        where ``<i, j>_hor`` are all pairs of horizontal nearest neighbors.
        The factor 1/2 in front of b is conventional because a NN operator is usually split into
        a horizontal and a vertical part
        """
        if horizontal:
            vert = cls.from_nearest_neighbors(
                site=site, 
                lx=ly, ly=lx, horizontal=False, bond_pos=bond_pos,  # note rotation (lx, ly) <-> (ly, lx)
                a=a, A=A, b=b, B=B
            )
            return vert.rotate90()
        
        # can assume vertical from now on
        assert 0 <= bond_pos < ly

        if A is None:
            site.get_op('Id').zeros_like()
        elif isinstance(A, str):
            A = site.get_op(A)
        if B is None:
            B = site.get_op('Id').zeros_like()
        elif isinstance(B, str):
            B = site.get_op(B)
        Id = site.get_op('Id')

        trivial_leg = npc.LegCharge.from_qflat(site.leg.chinfo, [site.leg.chinfo.make_valid()])
        trivial_leg_conj = trivial_leg.conj()
        leg_C_wU = npc.LegCharge.from_qflat(site.leg.chinfo, [Id.qtotal, A.qtotal, Id.qtotal], qconj=-1)
        leg_D_wL = npc.LegCharge.from_qflat(site.leg.chinfo, [site.leg.chinfo.make_valid()] * 2)

        _C = [[Id, A, .5 * b * B], [None, None, a * A], [None, None, Id]]
        _I = [[None, None, Id], [None, None, None], [None, None, None]]
        _zero = [[None, None, None], [None, None, None], [None, None, None]]
        grid_labels = ['wL', 'wR', 'wD', 'wU']
        C_grid = np.array([[_C]], dtype=object)
        D_grid = np.array([[_I, _C], [_zero, _I]], dtype=object)

        C_bottom = npc.grid_outer(C_grid[:, :, :1, :],  # contract wD with [1, 0, 0]
                                  [trivial_leg, trivial_leg_conj, trivial_leg, leg_C_wU],
                                  grid_labels=grid_labels)
        C_mid = npc.grid_outer(C_grid,
                               [trivial_leg, trivial_leg_conj, leg_C_wU.conj(), leg_C_wU],
                               grid_labels=grid_labels)
        C_top = npc.grid_outer(C_grid[:, :, :, -1:],  # contract wU with [0, 0, 1]
                               [trivial_leg, trivial_leg_conj, leg_C_wU.conj(), trivial_leg_conj],
                               grid_labels=grid_labels)
        C_col = [C_bottom] + [C_mid] * (ly - 2) + [C_top]

        if bond_pos == 0:
            D_left = npc.grid_outer(D_grid[:1, :, :1, :],  # contract wL with [1, 0] and wD with [1, 0, 0]
                                    [trivial_leg, leg_D_wL.conj(), trivial_leg, leg_C_wU],
                                    grid_labels=grid_labels)
            D_mid = npc.grid_outer(D_grid[:, :, :1, :],  # contract wD with [1, 0, 0]
                                   [leg_D_wL, leg_D_wL.conj(), trivial_leg, leg_C_wU],
                                   grid_labels=grid_labels)
            D_right = npc.grid_outer(D_grid[:, -1:, :1, :],  # contract wR with [0, 1] and wD with [1, 0, 0]
                                     [leg_D_wL, trivial_leg_conj, trivial_leg, leg_C_wU],
                                     grid_labels=grid_labels)
        elif bond_pos == ly - 1:
            D_left = npc.grid_outer(D_grid[:1, :, :, -1:],  # contract wL with [1, 0] and wU with [0, 0, 1]
                                    [trivial_leg, leg_D_wL.conj(), leg_C_wU.conj(), trivial_leg_conj],
                                    grid_labels=grid_labels)
            D_mid = npc.grid_outer(D_grid[:, :, :, -1:],  # contract wU with [0, 0, 1]
                                   [leg_D_wL, leg_D_wL.conj(), leg_C_wU.conj(), trivial_leg_conj],
                                   grid_labels=grid_labels)
            D_right = npc.grid_outer(D_grid[:, -1:, :, -1:],  # contract wR with [0, 1] and wU with [0, 0, 1]
                                     [leg_D_wL, trivial_leg_conj, leg_C_wU.conj(), trivial_leg_conj],
                                     grid_labels=grid_labels)
        else:
            D_left = npc.grid_outer(D_grid[:1, :, :, :],  # contract wL with [1, 0]
                                    [trivial_leg, leg_D_wL.conj(), leg_C_wU.conj(), leg_C_wU],
                                    grid_labels=grid_labels)
            D_mid = npc.grid_outer(D_grid,
                                [leg_D_wL, leg_D_wL.conj(), leg_C_wU.conj(), leg_C_wU],
                                grid_labels=grid_labels)
            D_right = npc.grid_outer(D_grid[:, -1:, :, :],  # contract wR with [0, 1]
                                    [leg_D_wL, trivial_leg_conj, leg_C_wU.conj(), leg_C_wU],
                                    grid_labels=grid_labels)
        
        left_col = C_col[:bond_pos] + [D_left] + C_col[bond_pos + 1:]
        mid_col = C_col[:bond_pos] + [D_mid] + C_col[bond_pos + 1:]
        right_col = C_col[:bond_pos] + [D_right] + C_col[bond_pos + 1:]
        tensors = left_col + mid_col * (lx - 2) + right_col

        return cls(sites=[site] * (lx * ly), tensors=tensors, lx=lx, ly=ly, bc='finite')

    @classmethod
    def from_mpo_sum(cls, mpos: list[MPO], horizontal: bool, bond_pos: int = None):
        """Create a PEPO for the sum of MPOs. Only for finite systems.

        The MPOs are embedded "left to right" for horizontal and "bottom to top" for vertical
        and act trivially on sites outside of their row / column.
        
        Parameters
        ----------
        mpos
        horizontal : bool
            If the MPOs are embedded horizontally (``True``) or vertically (``False``)
        bond_pos : int
            The index (x or y coordinate) of the additional two-dimensional bond is inserted
        """
        raise NotImplementedError

    def dagger(self):
        """Return hermition conjugate copy of self."""
        raise NotImplementedError  # need to think about charges and qconj in detail
        # [p, p*, vU, ...] -conj-> [p*, p, vU*, ...] -transp-> [p, p*, vU*, ...]
        tensors = [t.conj().itranspose(['p', 'p*', 'vU*', 'vL*', 'vD*', 'vR*']) for t in self._tensors]
        # [p, p*, vU*, ...] -relabel-> [p, p*, vU, ...]
        for t in tensors:
            t.ireplace_labels(['vU*', 'vL*', 'vD*', 'vR*'], ['vU', 'vL', 'vD', 'vR'])
