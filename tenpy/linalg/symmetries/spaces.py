# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
from numpy import ndarray
import copy
import warnings
import bisect
from typing import TYPE_CHECKING, Sequence

from tenpy.linalg.dummy_config import printoptions

from .groups import Sector, SectorArray, Symmetry, no_symmetry
from ...tools.misc import inverse_permutation
from ...tools.string import format_like_list

if TYPE_CHECKING:
    from ..backends.abstract_backend import AbstractBackend


__all__ = ['VectorSpace', 'ProductSpace']


class VectorSpace:
    r"""A vector space, which decomposes into sectors of a given symmetry.

    A vector space is characterized by a basis, and in particular a defined order of basis elements.
    Each basis vector lies in one of the sectors of the :attr:`symmetry`.

    For efficiency we want to use a modified order of the basis, such that::

        - The basis elements that belong to the same sector appear contigously.
          This allows us to directly read-off the blocks that contain the free parameters.

        - The sectors are sorted. This makes look-ups more efficient.
          For a ket-space (``is_dual is False``), we sort according to ``np.lexsort(sectors.T)``.
          For a bra-space (``is_dual is True``), we sort according
          to ``np.lexsort(symmetry.dual_sectors(sectors).T)``.
          This means that a space and its dual have compatible sector orders.
     
    A VectorSpace instance captures this information, i.e. the permutation of the basis that
    achieves the above properties, which sectors appear and how often.

    .. note ::
        It is best to think of ``VectorSpace``s as immutable objects.
        In practice they are mutable, i.e. you could change their attributes.
        This may lead to unexpected behavior, however, since it might make the cached metadata inconsistent.

    Attributes
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sectors : 2D numpy array of int
        The sectors that compose this space. A 2D array of integers with axes [s, q] where s goes
        over different sectors and q over the (one or more) numbers needed to label a sector.
        The sectors (to be precise, the rows `sectors[i, :]`) are unique. The order is an
        implementation detail and not physical, see :attr:`_non_dual_sectors`
    multiplicities : 1D numpy array of int
        How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
        ``sectors[i, :]`` appears ``multiplicities[i]`` times.
    slices : 2D numpy array of int
        For every sector ``sectors[n]``, the start ``slices[n, 0]`` and stop ``slices[n, 1]`` of
        indices (in the *internal* basis order) that belong to this sector.
        Conversely, ``basis_perm[slices[n, 0]:slices[n, 1]]`` are the elements of the public
        basis that live in ``sectors[n]``.
    is_dual : bool
        Whether this is the dual (a.k.a. bra) space or the regular (a.k.a. ket) space.
    basis_perm : ndarray
        The permutation of basis elements such that ``basis_vectors[basis_perm]`` is grouped by
        sector, i.e. such that ``basis_vectors[basis_perm][slices[n, 0]:slices[n, 1]]`` lie in
        ``sectors[n]``. ``dense_data[basis_perm] == internal_data``.
        All other attributes implicitly or explicitly assume this basis order.
    sectors_of_basis : 2D numpy array of int
        The sectors of every element of the "public" computational basis.
    _non_dual_sectors : 2D numpy array of int
        Internally stored version of :attr:`sectors`. For ket spaces (``is_dual=True``),
        these are the same as :attr:`sectors`, for bra spaces these are their duals.
        They are sorted such that ``np.lexsort(_non_dual_sectors.T)`` is trivial.

    Parameters
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sectors: 2D array_like of int
        For a ket-space (``_is_dual=False``), the sectors of the symmetry that compose this space.
        For a bra-space, the duals of that.
        Must be sorted such that ``np.lexsort(_non_dual_sectors.T)`` is trivial.
        Sectors may not contain duplicates. Multiplicity is specified via the separate arg below.
    multiplicities: 1D array_like of int, optional
        How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
        ``sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
        If not given, a multiplicity ``1`` is assumed for all `sectors`.
    basis_perm : ndarray, optional
        See the attribute :attr:`basis_perm`.
        Per default the trivial permutation ``[0, 1, 2, ...]`` is used.
    is_real : bool
        Whether the space is over the real numbers. Otherwise it is over the complex numbers (default).
    _is_dual : bool
        Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.

        .. warning :
            For ``_is_dual is True``, the passed `sectors` are interpreted as the sectors of the
            ("non-dual") ket-space isomorphic to self.
    """
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 basis_perm: ndarray = None, is_real: bool = False, _is_dual: bool = False):
        self.symmetry = symmetry
        self.is_real = is_real
        self.is_dual = _is_dual
        self._non_dual_sectors = sectors = np.asarray(sectors, dtype=int)
        self.num_sectors = num_sectors = len(sectors)
        if multiplicities is None:
            self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int)
        else:
            self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int)
        self.slices = _calc_slices(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities)
        self.dim = dim = np.sum(symmetry.batch_sector_dim(sectors) * multiplicities)
        if basis_perm is None:
            # OPTIMIZE special case this, where no permutation is needed?
            self.basis_perm = basis_perm = np.arange(dim)
        else:
            self.basis_perm = basis_perm = np.asarray(basis_perm, dtype=int)
        self._inverse_basis_perm = inverse_permutation(basis_perm)

    @classmethod
    def from_unsorted_sectors(cls, symmetry: Symmetry, sectors: SectorArray,
                              multiplicities: ndarray = None, basis_perm: ndarray = None,
                              is_real: bool = False):
        """Like constructor, but allows `sectors` in any order and only creates ket-spaces."""
        # OPTIMIZE : doing this in a cumbersome way right now...
        #   The difficult part is figuring out the new basis_perm.
        #   I think doing sth like basis_perm[slice(*s) for s in slices[sort]] might work...
        sectors = np.asarray(sectors, dtype=int)
        assert len(sectors.shape) == 2 and sectors.shape[1] == symmetry.sector_ind_len
        if multiplicities is None:
            multiplicities = np.ones((len(sectors),), dtype=int)
        slices = _calc_slices(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities)
        dim = np.sum(symmetry.batch_sector_dim(sectors) * multiplicities)
        if basis_perm is None:
            basis_perm = np.arange(dim)
        sectors_of_basis = np.zeros((dim, symmetry.sector_ind_len), dtype=int)
        for sect, slc in zip(sectors, slices):
            sectors_of_basis[basis_perm[slice(*slc)]] = sect[None, :]
        return cls.from_basis(symmetry=symmetry, sectors_of_basis=sectors_of_basis, is_real=is_real)

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors_of_basis: Sequence[Sequence[int]],
                   is_real: bool = False):
        """Create a VectorSpace by specifying the sector of every basis element.

        This classmethod always creates a ket-space (i.e. ``res.is_dual is False``).
        Use ``VectorSpace.from_basis(...)

        Parameters
        ----------
        sectors_of_basis : iterable of iterable of int
            Specifies the basis. ``sectors[n]`` is the sector of the ``n``-th basis element.
        symmetry, is_real
            Same as parameters for :class:`VectorSpace`
        """
        sectors_of_basis = np.asarray(sectors_of_basis, dtype=int)
        assert sectors_of_basis.shape[1] == symmetry.sector_ind_len
        # unfortunately, np.unique has the opposite sorting convention ("first entry first")
        # from np.lexsort ("last entry first").
        # We thus reverse the order of every sector before calling unique.
        unique, inv_idcs, mults = np.unique(sectors_of_basis[:, ::-1], axis=0, return_inverse=True,
                                            return_counts=True)
        sectors = unique[:, ::-1]
        assert np.all(np.lexsort(sectors.T) == np.arange(len(sectors)))  # TODO remove check
        basis_perm = np.argsort(inv_idcs)
        return cls(symmetry=symmetry, sectors=sectors, multiplicities=mults, basis_perm=basis_perm,
                   is_real=is_real, _is_dual=False)

    def test_sanity(self):
        # sectors
        assert all(self.symmetry.is_valid_sector(s) for s in self._non_dual_sectors)
        assert len(self._non_dual_sectors) == self.num_sectors
        assert len(np.unique(self._non_dual_sectors, axis=0)) == self.num_sectors
        assert self._non_dual_sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len)
        assert np.all(np.lexsort(self._non_dual_sectors.T) == np.arange(self.num_sectors))
        # multiplicities
        assert np.all(self.multiplicities > 0)
        assert self.multiplicities.shape == (self.num_sectors,)
        # slices
        assert self.slices.shape == (self.num_sectors, 2)
        slice_diffs = self.slices[:, 1] - self.slices[:, 0]
        expect_diffs = self.symmetry.batch_sector_dim(self._non_dual_sectors) * self.multiplicities
        assert np.all(slice_diffs == expect_diffs)
        # slices should be consecutive
        if len(self.slices) > 0:
            assert self.slices[0, 0] == 0
            assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
            assert self.slices[-1, 1] == self.dim
        # basis_perm
        assert self.basis_perm.shape == (self.dim,)
        assert len(np.unique(self.basis_perm)) == self.dim
        assert np.all(self.basis_perm[self._inverse_basis_perm] == np.arange(self.dim))

    @classmethod
    def without_symmetry(cls, dim: int, is_real: bool = False, is_dual: bool = False):
        """Initialize a VectorSpace with no symmetry of a given dimension"""
        return cls(symmetry=no_symmetry, sectors=no_symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_real=is_real, _is_dual=is_dual)

    @property
    def sectors(self):
        # OPTIMIZE cachedproperty?
        if self.is_dual:
            return self.symmetry.dual_sectors(self._non_dual_sectors)
        else:
            return self._non_dual_sectors

    @property
    def sectors_of_basis(self):
        # build in internal basis, then permute
        res = np.zeros((self.dim, self.symmetry.sector_ind_len), dtype=int)
        for sect, slc in zip(self.sectors, self.slices):
            res[slice(*slc)] = sect[None, :]
        return res[self._inverse_basis_perm]

    def sector(self, i: int) -> Sector:
        """Return the `i`-th sector. Equivalent to ``self.sectors[i]``."""
        sector = self._non_dual_sectors[i, :]
        if self.is_dual:
            return self.symmetry.dual_sector(sector)
        return sector

    def parse_index(self, idx: int) -> tuple[int, int]:
        """Utility function to translate an index for this VectorSpace.

        Parameters
        ----------
        idx : int
            An index of the leg, labelling an element of the public computational basis of self.

        Returns
        -------
        sector_idx : int
            The index of the correspinding sector,
            indicating that the `idx`-th basis element lives in ``self.sectors[sector_idx]``.
        multiplicity_idx : int
            The index "within the sector", in ``range(self.multiplicities[sector_index])``.
        """
        print(f'parsing index. input {idx}')
        idx = self._inverse_basis_perm[idx]
        sector_idx = bisect.bisect(self.slices[:, 0], idx) - 1
        multiplicity_idx = idx - self.slices[sector_idx, 0]
        return sector_idx, multiplicity_idx

    def idx_to_sector(self, idx: int) -> Sector:
        """Returns the sector associated with an index.
        
        This is the sector that the `idx`-th element of the public computational basis of self lives in.
        """
        return self.sector(self.parse_index(idx)[0])

    def sectors_where(self, sector: Sector) -> int | None:
        """Find the index `i` s.t. ``self.sectors[i] == sector``, or ``None`` if no such ``i`` exists."""
        where = np.where(np.all(self.sectors == sector, axis=1))[0]
        if len(where) == 0:
            return None
        if len(where) == 1:
            return where[0]
        raise RuntimeError  # sectors should have unique entries, so this should not happen

    def _non_dual_sectors_where(self, sector: Sector) -> int | None:
        """Find the index `i` s.t. ``self._non_dual_sectors[i] == sector``.

        Or ``None`` if no such ``i`` exists.
        """
        # OPTIMIZE use that _non_dual_sectors is sorted to speed up lookup?
        where = np.where(np.all(self._non_dual_sectors == sector, axis=1))[0]
        if len(where) == 0:
            return None
        if len(where) == 1:
            return where[0]
        raise RuntimeError  # sectors should have unique entries, so this should not happen

    def sector_multiplicity(self, sector: Sector) -> int:
        """The multiplicitiy of the given sector.

        Returns 0 if self does not have that sector.
        """
        idx = self.sectors_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]

    def _non_dual_sector_multiplicity(self, sector: Sector) -> int:
        """The multiplicitiy of the given _non_dual_sector.

        Returns 0 if self does not have that sector.
        """
        idx = self._non_dual_sectors_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]

    def __repr__(self):
        # TODO include basis_perm?
        # number of *entries* in sectors (num_sectors * num_charges) that triggers multiple lines
        multiline_threshold = 4
        
        is_dual_str = '.dual' if self.is_dual else ''

        if self.sectors.size < multiline_threshold:
            elements = [
                f'VectorSpace({self.symmetry!r}',
                *(['is_real=True'] if self.is_real else []),
                f'sectors={format_like_list(self.symmetry.sector_str(s) for s in self.sectors)}',
                f'multiplicities={format_like_list(self.multiplicities)}){is_dual_str}',
            ]
            res = ', '.join(elements)
            if len(res) <= printoptions.linewidth:
                return res

        indent = printoptions.indent * ' '
        
        # check if printing all sectors fits within linewidth
        if 3 * self.sectors.size < printoptions.linewidth:  # otherwise there is no chance anyway
            lines = [
                f'VectorSpace({self.symmetry!r},{" is_real=True," if self.is_real else ""}',
                f'{indent}sectors={format_like_list(self.symmetry.sector_str(s) for s in self.sectors)},',
                f'{indent}multiplicities={format_like_list(self.multiplicities)}',
                f'){is_dual_str}'
            ]
            if all(len(l) <= printoptions.linewidth for l in lines):
                return '\n'.join(lines)

        # add as many sectors as possible before linewidth is reached
        # save most recent suggestion in variabel res. if new suggestion is too long, return res.
        res = f'VectorSpace({self.symmetry!r}, ...)'
        if len(res) > printoptions.linewidth:
            return 'VectorSpace(...)'
        prio = self._sector_print_priorities(use_private_sectors=False)
        lines = [
            f'VectorSpace({self.symmetry!r},{" is_real=True," if self.is_real else ""}',
            f'{indent}sectors=[...],',
            f'{indent}multiplicities=[...]',
            f'){is_dual_str}'
        ]
        for n in range(1, len(prio)):
            # OPTIMIZE could optimize the search grid...
            if any(len(l) > printoptions.linewidth for l in lines):
                # this is the first time that printing all lines would be too long
                return res
            which = np.sort(prio[:n])
            sectors = [self.symmetry.sector_str(s) for s in self.sectors[which]]
            mults = list(self.multiplicities[which])
            # insert '...' between non-consecutive sectors
            jumps = np.where((which[1:] - which[:-1]) > 1)[0]
            for j in reversed(jumps):
                sectors[j + 1:j + 1] = ['...']
                mults[j + 1:j + 1] = ['...']
            lines[1] = f'{indent}sectors={format_like_list(sectors)},'
            lines[2] = f'{indent}multiplicities={format_like_list(mults)}'
        raise RuntimeError  # the above return should always trigger

    def __str__(self):
        return self._debugging_str(use_private_sectors=False)

    def _debugging_str(self, use_private_sectors: bool = True):
        """Version of ``str(self)`` intended for debugging the internals of ``tenpy.linalg``.

        Instead of the :attr:`sectors`, it shows the "private" :attr:`_non_dual_sectors`.
        """
        return '\n'.join(self._debugging_str_lines(use_private_sectors=use_private_sectors))

    def _debugging_str_lines(self, use_private_sectors: bool = True):
        """Part of :meth:`_debugging_str`"""
        if use_private_sectors:
            sectors = self._non_dual_sectors
            mults = self.multiplicities
        else:
            sectors = self.sectors
            mults = self.multiplicities
        indent = printoptions.indent * ' '
        lines = [
            'VectorSpace(',
            *([f'{indent}is_real=True'] if self.is_real else []),
            f'{indent}symmetry: {self.symmetry!s}',
            f'{indent}dim: {self.dim}',
            f'{indent}is_dual: {self.is_dual}',
            f'{indent}basis_perm: {self.basis_perm}',
            f'{indent}num sectors: {self.num_sectors}',
        ]
        # determine sectors: list[str] and mults: list[str]
        if len(lines) + self.num_sectors <= printoptions.maxlines_spaces:
            sectors = [self.symmetry.sector_str(s) for s in sectors]
            mults = [str(m) for m in mults]
        else:
            # not all sectors are shown. add how many there are
            lines[4:4]= []
            prio = self._sector_print_priorities(use_private_sectors=use_private_sectors)
            which = []
            jumps = []
            for n in range(len(prio)):
                _which = np.sort(prio[:n])
                _jumps = np.where((_which[1:] - _which[:-1]) > 1)[0]
                header = 1
                if len(lines) + header + n + len(_jumps) > printoptions.maxlines_spaces:
                    sectors = [self.symmetry.sector_str(s) for s in sectors[which]]
                    mults = [str(m) for m in mults[which]]
                    for j in reversed(jumps):
                        sectors[j + 1:j + 1] = ['...']
                        mults[j + 1:j + 1] = ['...']
                    break
                which = _which
                jumps = _jumps
            else:
                raise RuntimeError  # break should have been reached
        # done with sectors: list[str] and mults: list[str]
        sector_col_width = max(len("sectors"), max(len(s) for s in sectors))
        lines.append(f'{indent}{"sectors".ljust(sector_col_width)} | multiplicities')
        lines.extend(f'{indent}{s.ljust(sector_col_width)} | {m}' for s, m in zip(sectors, mults))
        lines.append(')')
        return lines

    def __eq__(self, other):
        if not isinstance(other, VectorSpace):
            return NotImplemented
        if isinstance(other, ProductSpace):
            # ProductSpace overrides __eq__, so self is not a ProductSpace
            return False
        return self.is_dual == other.is_dual and self.is_equal_or_dual(other)

    @property
    def dual(self):
        res = copy.copy(self)  # shallow copy, works for subclasses as well and preserves metadata
        res.is_dual = not self.is_dual
        return res

    def flip_is_dual(self) -> VectorSpace:
        """Return copy of `self` with same :attr:`sectors`, but opposite :attr:`is_dual` flag.

        Note that this leg can not be contracted with `self`.
        """
        # TODO test coverage
        non_dual_sectors = self.symmetry.dual_sectors(self._non_dual_sectors)
        sort = np.lexsort(non_dual_sectors.T)
        raise NotImplementedError
        # TODO need to figure out basis_perm...
        #  I think one needs to do sth like basis_perm[slice(*s) for s in slices[sort]]
        return VectorSpace(symmetry=self.symmetry,
                           sectors=non_dual_sectors[sort],
                           multiplicities=self.multiplicities[sort],
                           basis_perm=self.basis_perm,
                           is_real=self.is_real,
                           _is_dual=not self.is_dual)

    def is_equal_or_dual(self, other: VectorSpace) -> bool:
        """If another VectorSpace is equal to *or* dual of `self`.

        Assumes without checking that other is a VectorSpace.
        Does not check for ProductSpace.
        """
        if self.is_real != other.is_real:
            return False
        if self.symmetry != other.symmetry:
            return False
        if self.num_sectors != other.num_sectors:
            # now we may assume that checking all multiplicities of self is enough.
            return False
        if not np.all(self._non_dual_sectors == other._non_dual_sectors):
            return False
        if not np.all(self.multiplicities == other.multiplicities):
            return False
        if not np.all(self.basis_perm == other.basis_perm):
            return False
        return True

    def can_contract_with(self, other):
        """If self can be contracted with other.

        Equivalent to ``self == other.dual`` if `other` is a :class:`VectorSpace`.
        """
        if not isinstance(other, VectorSpace):
            return False
        if isinstance(other, ProductSpace):
            # ProductSpace overrides can_contract_with, so self is not a ProductSpace
            return False
        return self.is_dual != other.is_dual and self.is_equal_or_dual(other)

    @property
    def is_trivial(self) -> bool:
        """Whether self is the trivial space.

        The trivial space is the one-dimensional space which consists only of the trivial sector,
        appearing exactly once. In a mathematical sense, the trivial sector _is_ the trivial space.
        """
        if self._non_dual_sectors.shape[0] != 1:
            return False
        # have already checked if there is more than 1 sector, so can assume self.multiplicities.shape == (1,)
        if self.multiplicities[0] != 1:
            return False
        if not np.all(self._non_dual_sectors[0] == self.symmetry.trivial_sector):
            return False
        return True

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the number of linearly independent symmetric tensors in this space."""
        # the trivial sector is by definition self-dual, so we can search self._non_dual_sectors,
        # even if self.is_dual is True.
        # OPTIMIZE use that _non_dual_sectors are lexsorted to shorten the loop.
        for s, m in zip(self._non_dual_sectors, self.multiplicities):
            if np.all(s == self.symmetry.trivial_sector):
                return m
        return 0

    def is_subspace_of(self, other: VectorSpace) -> bool:
        """Whether self is a subspace of other.

        This function considers both spaces purely as `VectorSpace`s and ignores a possible
        `ProductSpace` structure.
        Per convention, self is never a subspace of other, if the :attr:`is_dual` are different
        """
        if self.is_dual != other.is_dual:
            return False
        if self.symmetry != other.symmetry:
            return False

        # the _non_dual_sectors are lexsorted, so we can just iterate over both of them
        n_self = 0
        for other_sector, other_mult in zip(other._non_dual_sectors, other.multiplicities):
            if np.all(self._non_dual_sectors[n_self] == other_sector):
                if self.multiplicities[n_self] > other_mult:
                    return False
                n_self += 1
            if n_self == self.num_sectors:
                # have checked all sectors of self
                return True
        # reaching this line means self has sectors which other does not have
        return False

    def as_VectorSpace(self):
        return self

    def _sector_print_priorities(self, use_private_sectors: bool):
        """How to prioritize sectors if not all can be printed.

        Used in `__repr__` and `__str__`.
        Returns indices of either ``self._non_dual_sectors`` if `use_private_sectors`
        or of ``self.sectors`` in order of priority"""
        first = 0
        last = self.num_sectors - 1
        largest = np.argmax(self.multiplicities)
        special = [first, last, largest, first + 1, last - 1, largest - 1, largest + 1]
        which = []
        for i in special:
            if i not in which and 0 <= i < self.num_sectors:
                which.append(i)
        which.extend(i for i in range(self.num_sectors) if i not in which)
        return np.array(which)


def _calc_slices(symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray) -> ndarray:
    """Calculate the slices given sectors and multiplicities *in the dense order*, i.e. not sorted."""
    slices = np.zeros((len(sectors), 2), dtype=np.intp)
    # OPTIMIZE should we special case abelian symmetries to skip some multiplications by 1?
    slices[:, 1] = slice_ends = np.cumsum(multiplicities * symmetry.batch_sector_dim(sectors))
    slices[1:, 0] = slice_ends[:-1]  # slices[0, 0] remains 0, which is correct
    return slices


class ProductSpace(VectorSpace):
    r"""The product of multiple spaces.

    Since the product of graded spaces may not be associative, we fix a convention for the order
    of pairwise fusions: "left to right".
    This generates a fusion tree looking like this (or it's dual flipped upside down)::

    |    spaces[0]
    |         \   spaces[1]
    |          \ /
    |           Y   spaces[2]
    |            \ /
    |             Y
    |              \
    |             ...  spaces[-1]
    |               \ /
    |                Y
    |                 \
    |                 self

    It is the product space of the individual `spaces`,
    but with an associated basis change implied to allow preserving the symmetry.

    Parameters
    ----------
    spaces:
        The factor spaces that multiply to this space.
        The resulting product space can always be split back into these.
    backend : AbstractBackend | None
        If a backend is given, the backend-specific metadata will be set via ``backend._fuse_spaces``.
    _is_dual : bool | None
        Flag indicating wether the fusion space represents a dual (bra) space or a non-dual (ket) space.
        Per default (``_is_dual=None``), ``spaces[0].is_dual`` is used, i.e. the ``ProductSpace``
        will be a bra space if and only if its first factor is a bra space.
        See notes on duality below.
    _sectors, _multiplicities:
        These inputs to VectorSpace.__init__ can optionally be passed to avoid recomputation.
        _sectors need to be _non_dual_sectors and in particular are assumed to be sorted
        (this is not checked!).

    Notes
    -----
    While mathematically the dual of the product :math:`(V \otimes W)^*` is the same as the
    product of the duals :math:`V^* \otimes W^*`, we distinguish these two objects in the
    implementation. This allows us to fulfill all of the following constraints
    a) Have the same order of :attr:`_non_dual_sectors` for a space and its dual,
        to make contractions easier.
    b) Consistently view every ProductSpace as a VectorSpace, i.e. have proper subclass behavior
        and in particular a well-behaved `is_dual` attribute.
    c) A ProductSpace can always be split into its :attr:`spaces`.

    As an example, consider two VectorSpaces ``V`` and ``W`` and the following four possible
    products::

        ==== ============================ ================== ========= ==================================
             Mathematical Expression      .spaces            .is_dual  ._non_dual_sectors
        ==== ============================ ================== ========= ==================================
        P1   :math:`V \otimes W`          [V, W]             False     P1._non_dual_sectors
        P2   :math:`(V \otimes W)^*`      [V.dual, W.dual]   True      P1._non_dual_sectors
        P3   :math:`V^* \otimes W^*`      [V.dual, W.dual]   False     dual(P1._non_dual_sectors)
        P4   :math:`(V^* \otimes W^*)^*`  [V, W]             True      dual(P1._non_dual_sectors)
        ==== ============================ ================== ========= ==================================

    They can be related to each other via the :attr:`dual` property or via :meth:`flip_is_dual`.
    In this example we have `P1.dual == P2`` and ``P3.dual == P4``, as well as
    ``P1.flip_is_dual() == P4`` and ``P2.flip_is_dual() == P3``.

    The mutually dual spaces, e.g. ``P1`` and ``P2``, can be contracted with each other, as they
    have opposite :attr:`is_dual` and matching :attr:`_non_dual_sectors`.
    The spaces related by :meth:`flip_is_dual()`, e.g. ``P2`` and ``P3``, would be considered
    the same space mathematically, but in this implementation we have ``P2 != P3`` due to the
    different :attr:`is_dual` attribute. Since they represent the same space, they have the same
    :attr:`sectors`.
    This also means that ``P1.can_contract_with(P3) is False``.
    The contraction can be done, however, by first converting ``P3.flip_is_dual() == P2``,
    since then ``P1.can_contract_with(P2) is True``.
    # TODO (JU) is there a corresponding function that does this on a tensor? -> reference it.

    This convention has the downside that the mathematical notation :math:`P_2 = (V \otimes W)^*`
    does not transcribe trivially into a single call of ``ProductSpace.__init__``, since
    ``P2 = ProductSpace([V.dual, W.dual], _is_dual=True)``.

    Note that the default behavior of the `_is_dual` argument guarantees that
    `ProductSpace(some_space)` is contractible with `ProductSpace([s.dual for s in some_spaces])`.
    """
    def __init__(self, spaces: list[VectorSpace], backend: AbstractBackend = None, _is_dual: bool = None,
                 _sectors: SectorArray = None, _multiplicities: ndarray = None):
        if _is_dual is None:
            _is_dual = spaces[0].is_dual
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)

        if _sectors is None:
            assert _multiplicities is None
            if backend is None:
                _sectors, _multiplicities, metadata = _fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            else:
                _sectors, _multiplicities, metadata = backend._fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            for key, val in metadata.items():
                setattr(self, key, val)
        else:
            assert _multiplicities is not None
        # TODO we could fix the basis_perm, but we probably dont need to ...
        VectorSpace.__init__(self, symmetry=symmetry, sectors=_sectors, multiplicities=_multiplicities,
                             is_real=is_real, _is_dual=_is_dual)

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors: Sequence[Sequence[int]], is_real: bool = False):
        raise NotImplementedError('from_basis can not create ProductSpaces')

    def test_sanity(self):
        for s in self.spaces:
            assert s.symmetry == self.symmetry
            s.test_sanity()
        return super().test_sanity()

    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self._non_dual_sectors,
                           multiplicities=self.multiplicities,
                           is_real=self.is_real,
                           basis_perm=None,
                           _is_dual=self.is_dual,)

    def flip_is_dual(self) -> ProductSpace:
        """Return a ProductSpace isomorphic to self, which has the opposite is_dual attribute.

        This realizes the isomorphism between ``V.dual * W.dual`` and ``(V * W).dual``
        for `VectorSpace` ``V`` and ``W``.
        Note that the returned space is equal to neither ``self`` nor ``self.dual``.
        See docstring of :class:`ProductSpace` for details.

        Backend-specific metadata may be lost.
        TODO (JU) can we figure out how to keep it?
                  alternatively: should we call backend.add_leg_metadata to add it again?
        """
        sectors = self.symmetry.dual_sectors(self._non_dual_sectors)
        sort = np.lexsort(sectors.T)
        return ProductSpace(spaces=self.spaces,
                            _is_dual=not self.is_dual,
                            _sectors=sectors[sort],
                            _multiplicities=self.multiplicities[sort])

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return iter(self.spaces)

    def __repr__(self):
        lines = [f'ProductSpace([']
        indent = printoptions.indent * ' '
        num_lines = 2  # first and last line
        for s in self.spaces:
            next_space = indent + repr(s).replace('\n', '\n' + indent) + ','
            additional_lines = 1 + next_space.count('\n')
            if num_lines + additional_lines > printoptions.maxlines_spaces:
                break
            lines.append(next_space)
            num_lines += additional_lines
        if self.is_dual:
            lines.append(']).dual')
        else:
            lines.append(f'])')
        return '\n'.join(lines)

    def _debugging_str_lines(self, use_private_sectors: bool = True):
        indent = printoptions.indent * ' '
        lines = VectorSpace._debugging_str_lines(self, use_private_sectors=use_private_sectors)
        lines[0] = 'ProductSpace('
        offset = 1 if self.is_real else 0
        lines[2 + offset] = f'{indent}dim: [{", ".join(str(s.dim) for s in self.spaces)}] -> {self.dim}'
        lines[3 + offset] = f'{indent}is_dual: [{", ".join(str(s.is_dual) for s in self.spaces)}] -> {self.is_dual}'
        lines[4 + offset] = f'{indent}num sectors: [{", ".join(str(s.num_sectors) for s in self.spaces)}] -> {self.num_sectors}'
        return lines

    def __eq__(self, other):
        if not isinstance(other, VectorSpace):
            return NotImplemented
        if not isinstance(other, ProductSpace):
            return False
        if other.is_dual != self.is_dual:
            return False
        if len(other.spaces) != len(self.spaces):
            return False
        return all(s1 == s2 for s1, s2 in zip(self.spaces, other.spaces))

    @property
    def dual(self):
        res = copy.copy(self)  # shallow copy, works for subclasses as well
        res.is_dual = not self.is_dual
        res.spaces = [s.dual for s in self.spaces]
        return res

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    def can_contract_with(self, other):
        if not isinstance(other, ProductSpace):
            return False
        if self.is_dual == other.is_dual:
            return False
        if len(self.spaces) != len(other.spaces):
            return False
        return all(s1.can_contract_with(s2) for s1, s2 in zip(self.spaces, other.spaces))


def _fuse_spaces(symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool
                 ) -> tuple[SectorArray, ndarray, ndarray, dict]:
    """This function is called as part of ProductSpace.__init__.
    
    It determines the sectors and multiplicities of the ProductSpace.
    There is also a verison of this function in the backends, i.e.
    :meth:`~tenpy.linalg.backends.abstract_backend.AbstractBackend._fuse_spaces:, which may
    customize this behavior and in particulat may return metadata, i.e. attributes to be added to
    the ProductSpace.
    This default implementation returns empty metadata ``{}``.

    Returns
    -------
    sectors : 2D array of int
        The :attr:`VectorSpace._non_dual_sectors`.
    mutliplicities : 1D array of int
        the :attr:`VectorSpace.multiplicities`.
    metadata : dict
        A dictionary with string keys and arbitrary values.
        These will be added as attributes of the ProductSpace
    """
    # TODO (JU) should we special case symmetry.fusion_style == FusionStyle.single ?
    if _is_dual:
        spaces = [s.dual for s in spaces] # directly fuse sectors of dual spaces.
        # This yields overall dual `sectors` to return, which we directly save in
        # self._non_dual_sectors, such that `self.sectors` (which takes a dual!) yields correct sectors
        # Overall, this ensures consistent sorting/order of sectors between dual ProductSpace!
    fusion = {tuple(s): m for s, m in zip(spaces[0].sectors, spaces[0].multiplicities)}
    for space in spaces[1:]:
        new_fusion = {}
        for t_a, m_a in fusion.items():
            s_a = np.array(t_a)
            for s_b, m_b in zip(space.sectors, space.multiplicities):
                for s_c in symmetry.fusion_outcomes(s_a, s_b):
                    t_c = tuple(s_c)
                    n = symmetry._n_symbol(s_a, s_b, s_c)
                    new_fusion[t_c] = new_fusion.get(t_c, 0) + m_a * m_b * n
        fusion = new_fusion
        # by convention fuse spaces left to right, i.e. (...((0,1), 2), ..., N)
    non_dual_sectors = np.asarray(list(fusion.keys()))
    multiplicities = np.asarray(list(fusion.values()))
    sort = np.lexsort(non_dual_sectors.T)
    metadata = {}
    return non_dual_sectors[sort], multiplicities[sort], metadata
