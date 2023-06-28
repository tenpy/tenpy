# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
from numpy import ndarray
import copy
from functools import cached_property
import warnings
import bisect
from typing import TYPE_CHECKING

from .groups import Sector, SectorArray, Symmetry, no_symmetry
from ...tools.misc import inverse_permutation
from ...tools.string import vert_join

if TYPE_CHECKING:
    from ..backends.abstract_backend import AbstractBackend


__all__ = ['VectorSpace', 'ProductSpace']


class VectorSpace:
    r"""A vector space, which decomposes into sectors of a given symmetry.

    .. note ::
        It is best to think of ``VectorSpace``s as immutable objects.
        In practice they are mutable, i.e. you could change their attributes.
        This may lead to unexpected behavior, however, since it might make the cached metadata inconsistent.

    Attributes
    ----------
    sectors : 2D numpy array of int
        The sectors that compose this space, in order of appearance in the space.
        A 2D array of integers with axes [s, q] where s goes over different sectors and q over the
        (one or more) numbers needed to label a sector.
        The sectors (to be precise, the rows `sectors[i, :]`) are unique, i.e. the computational
        basis needs to be ordered such that basis vectors that belong to the same sector appear
        consecutively.
    multiplicities : 1D numpy array of int
        How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
        ``sectors[i, :]`` appears ``multiplicities[i]`` times.
    slices : 2D numpy array of int
        For every sector ``sectors[n, :]``, this specifies the start (``slices[n, 0]``) and stop
        (``slices[n, 1]``) of indices on a dense array into which the sector is embedded.
        This means e.g. that
        ``slices[n, 1] - slices[n, 0] == symmetry.sector_dim(sectors[n]) * multiplicities[n]``
    is_dual : bool
        Whether this is the dual (a.k.a. bra) space or the regular (a.k.a. ket) space.

    _non_dual_sorted_sectors : 2D numpy array of int
        Internally stored version of :attr:`sectors`.
        To simplify the internal functionality, we store the information given by `sectors` in a
        different way.
        Firstly, for bra spaces (`is_dual=True`), we store the duals of the sectors, i.e. the
        sectors of the ket space that is isomorphic to self.
        This means that mutually dual VectorSpaces have the same `_non_dual_sorted_sectors`, which
        simplifies bookkeeping in contractions etc.
        Secondly, we sort the sectors such that ``np.lexsort(_non_dual_sorted_sectors.T)`` is trivial.
        This simplifies lookups.
        Overall we have ``_non_dual_sorted_sectors == sectors[_sector_perm]`` for a ket space and
        ``_non_dual_sorted_sectors == symmetry.dual_sectors(sectors[_sector_perm])`` for a bra space.
    _sorted_sectors : 2D numpy array of int
        Property. Same duality as :attr:`sectors` but in the order of :attr:`_sector_perm`,
        such that ``space._sorted_sectors == space.sectors[space._sector_perm]``.
    _sector_perm : 1D numpy array of int
        The permutation of sectors induced by the above sorting
    _sorted_multiplicities : 1D numpy array of int
        The multiplicities ordered like `_non_dual_sorted_sectors`, i.e. `_multiplicities == multiplicities[sector_perm]`.
    _sorted_slices : 2D numpy array of int
        The slices ordered like `_sectors`, i.e. `_sorted_slices == slices[sector_perm]`.
        
    Parameters
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sectors: 2D array_like of int
        The sectors of the symmetry that compose this space.
        A 2D array of integers with axes [s, q] where s goes over different sectors
        and q over the different quantities needed to describe a sector.
        E.g. if the symmetry is :math:`U(1) \times Z_2`, then ``sectors[s, 0]`` gives the :math:`U(1)`
        charge for the `s`-th sector, and ``sectors[s, 1]`` the respective :math:`Z_2` charge.
        Sectors may not contain duplicates. Multiplicity is specified via the separate arg below.
    multiplicities: 1D array_like of int, optional
        How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
        ``sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
        If not given, a multiplicity ``1`` is assumed for all `sectors`.
        Before storing, the multiplicities are permuted along with the `sectors` to sort the latter.
    is_real : bool
        Whether the space is over the real numbers. Otherwise it is over the complex numbers (default).

    _is_dual : bool
        Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.

        .. warning :
            For ``_is_dual is True``, the passed `sectors` are interpreted as the sectors of the
            ("non-dual") ket-space isomorphic to self.
            These are the :attr:`_non_dual_sorted_sectors`, except they dont need to be sorted.
            This means that to construct the dual of ``VectorSpace(..., some_sectors)``,
            we need to call ``VectorSpace(..., some_sectors, _is_dual=True)`` and in particular
            pass the *same* sectors.
            Consider using ``VectorSpace(..., some_sectors).dual`` instead for more readable code.

    _sector_perm : 1D array-like of int, optional
        Allows to skip sorting of sectors, if the _sector_perm is already known.
        If `_sector_perm` is given, the `sectors` and `multiplicities` are assumed to be in sorted
        order, such that the following two calls are equivalent:
        ``v1 = VectorSpace(..., sectors, mults, _sector_perm=None)``
        ``v2 = VectorSpace(..., sectors[perm], mults[perm], _sector_perm=perm)``
        where ``perm = np.lexsort(sectors.T)``.
        TODO (JU) should test exactly this setup in pytest.
    _sorted_slices : 2D array-like
        Allows to skip recomputing the slices.
        Note: These are the :attr:`_sorted_slices`, not :attr:`slices`!
    """

    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 is_real: bool = False, _is_dual: bool = False, _sector_perm: ndarray = None,
                 _sorted_slices: ndarray = None):
        self.symmetry = symmetry
        self.is_real = is_real
        self.is_dual = _is_dual
        sectors = np.asarray(sectors, dtype=int)
        num_sectors = len(sectors)
        if multiplicities is None:
            multiplicities = np.ones((num_sectors,), dtype=int)
        else:
            multiplicities = np.asarray(multiplicities, dtype=int)

        if _sector_perm is None:
            # sort the sectors
            self._sector_perm = perm = np.lexsort(sectors.T)
            self._non_dual_sorted_sectors = sectors[perm]
            self._sorted_multiplicities = multiplicities[perm]
        else:
            perm = None
            self._sector_perm = np.asarray(_sector_perm, dtype=np.intp)
            self._non_dual_sorted_sectors = sectors
            self._sorted_multiplicities = multiplicities

        if _sorted_slices is None:
            # note: need to use the sectors, multiplicities *before sorting* here
            _sorted_slices = _calc_slices(symmetry, sectors, multiplicities)
            if perm is not None:
                # if sectors have been sorted above, we need to permute the slices accordingly
                _sorted_slices = _sorted_slices[perm]
        self._sorted_slices = _sorted_slices

    def test_sanity(self):
        assert len(np.unique(self._non_dual_sorted_sectors, axis=0)) == self.num_sectors
        assert self._non_dual_sorted_sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len)
        assert self._sector_perm.shape == (self.num_sectors,)
        assert np.all(np.sum(self._sector_perm == np.arange(self.num_sectors)[None, :], axis=0) == 1)
        assert np.all(self._sorted_multiplicities > 0)
        assert self._sorted_multiplicities.shape == (self.num_sectors,)
 
    # TODO when should these caches be reset?
    @cached_property
    def _inverse_sector_perm(self):
        """The inverse permutation of :attr:`_sector_perm`."""
        return inverse_permutation(self._sector_perm)

    @property
    def dim(self):
        return np.sum(self.symmetry.batch_sector_dim(self._non_dual_sorted_sectors) * self._sorted_multiplicities)

    @property
    def num_sectors(self):
        return len(self._non_dual_sorted_sectors)

    @classmethod
    def non_symmetric(cls, dim: int, is_real: bool = False, _is_dual: bool = False):
        return cls(symmetry=no_symmetry, sectors=no_symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_real=is_real, _is_dual=_is_dual)

    @property
    def multiplicities(self):
        # OPTIMIZE cache?
        return self._sorted_multiplicities[self._inverse_sector_perm]

    @property
    def sectors(self):
        # OPTIMIZE cachedproperty?
        if self.is_dual:
            res = self.symmetry.dual_sectors(self._non_dual_sorted_sectors)
        else:
            res = self._non_dual_sorted_sectors
        return res[self._inverse_sector_perm]

    @property
    def _sorted_sectors(self):
        # OPTIMIZE cached?
        if self.is_dual:
            return self.symmetry.dual_sectors(self._non_dual_sorted_sectors)
        return self._non_dual_sorted_sectors

    @property
    def slices(self):
        # OPTIMIZE cache?
        return self._sorted_slices[self._inverse_sector_perm]

    def sector(self, i: int) -> Sector:
        """Return the `i`-th sector.

        Equivalent to ``self.sectors[i]``.
        """
        sector = self._non_dual_sorted_sectors[self._inverse_sector_perm[i], :]
        if self.is_dual:
            return self.symmetry.dual_sector(sector)
        return sector

    def parse_index(self, idx: int) -> tuple[int, int]:
        """Utility function to translate an index for this VectorSpace.

        Checks that an index is in the appropriate range, translates negative indices
        and splits it into a sector-part and a part within that sector.

        Parameters
        ----------
        idx : int
            An index of the leg, labelling an element of the computational basis of self.

        Returns
        -------
        sector_idx : int
            The index of the correspinding sector,
            indicating that the `idx`-th basis element lives in `self.sector(n)`.
        multiplicity_idx : int
            The index "within the sector", in `range(self.multiplicities[sector_index])`.
        """
        _sector_idx, multiplicity_idx = self._parse_index(idx)
        sector_idx = self._sector_perm[_sector_idx]
        return sector_idx, multiplicity_idx

    def _parse_index(self, idx: int) -> tuple[int, int]:
        """Like :meth:`parse_index`, but the `sector_idx` is w.r.t. the private :attr:`_sectors`.

        In particular, this means that the `idx`-th basis element lives
        in ``self._non_dual_sorted_sectors[self._parse_index(idx)[0]]`` (note the underscores!).
        """
        if not -self.dim <= idx < self.dim:
            raise IndexError(f'flat index {idx} out of bounds for space of dimension {self.dim}')
        if idx < 0:
            idx = idx + self.dim
        _sector_idx = bisect.bisect(self._sorted_slices, idx) - 1
        multiplicity_idx = idx - self._sorted_slices[_sector_idx]
        return _sector_idx, multiplicity_idx

    def idx_to_sector(self, idx: int) -> Sector:
        """Returns the sector associated with an index.
        
        This is the sector that the `idx`-th basis element of self lives in.
        """
        _sector_idx, _ = self._parse_index(idx)
        if self.is_dual:
            return self.symmetry.dual_sector(self._non_dual_sorted_sectors[_sector_idx])
        return self._non_dual_sorted_sectors[_sector_idx]

    def sectors_str(self, separator=', ', max_len=70) -> str:
        """short str describing the self_non_dual_sorted_sectors (note the underscore!) and their multiplicities"""
        full = separator.join(f'{self.symmetry.sector_str(a)}: {mult}'
                              for a, mult in zip(self._non_dual_sorted_sectors, self._sorted_multiplicities))
        if len(full) <= max_len:
            return full

        res = ''
        end = '[...]'

        for idx in np.argsort(self._sorted_multiplicities):
            new = f'{self.symmetry.sector_str(self._non_dual_sorted_sectors[idx])}: {self._sorted_multiplicities[idx]}'
            if len(res) + len(new) + len(end) + 2 * len(separator) > max_len:
                return res + separator + end
            res = res + separator + new
        raise RuntimeError  # a return should be triggered from within the for loop!

    def __repr__(self):
        # TODO include sector_perm?
        is_real_str = 'is_real=True, ' if self.is_real else ''
        summarization_threshold = 20
        sectors = np.array2string(self._non_dual_sorted_sectors[self._inverse_sector_perm],
                                  threshold=summarization_threshold, separator=', '
                                  ).replace('\n', '')
        mults = np.array2string(self._sorted_multiplicities[self._inverse_sector_perm],
                                threshold=summarization_threshold, separator=', '
                                ).replace('\n', '')
        
        sep = '\n   ' if self._non_dual_sorted_sectors.size > 3 else ''
        return f'VectorSpace({self.symmetry!r}, is_dual={self.is_dual},{sep} sectors={sectors},{sep} multiplicities={mults}{is_real_str}{sep[:1]})'

    def __str__(self):
        sectors = '_sector\n' + '\n'.join(self.symmetry.sector_str(s) for s in self._non_dual_sorted_sectors)
        mults = 'multiplicity\n' + '\n'.join(map(str, self._sorted_multiplicities))
        return f'symmetry: {self.symmetry!s}\nis_dual: {self.is_dual}\n' + vert_join([sectors, mults], delim=' | ')

    def __eq__(self, other):
        # TODO should we compare the perm_sectors?
        #  I.e. is VectorSpace(sym, sectors=[s1, s2]) == VectorSpace(sym, sectors=[s2, s1]) ?
        if not isinstance(other, VectorSpace):
            return NotImplemented
        if self.is_real != other.is_real:
            return False
        if self.is_dual != other.is_dual:
            return False
        if self.num_sectors != other.num_sectors:
            # now we may assume that checking all multiplicities of self is enough.
            return False
        if self.symmetry != other.symmetry:
            return False
        # _sectors are sorted, so we can just compare them elementwise.
        return np.all(self._non_dual_sorted_sectors == other._non_dual_sorted_sectors) and np.all(self._sorted_multiplicities == other._sorted_multiplicities)

    @property
    def dual(self):
        res = copy.copy(self)  # shallow copy, works for subclasses as well
        res.is_dual = not self.is_dual
        return res

    def flip_is_dual(self) -> VectorSpace:
        """Return copy of `self` with same :attr:`sectors`, but opposite :attr:`is_dual` flag.

        Note that this leg can often not be contracted with `self` since the order of the
        :attr:`sectors` might have changed.
        """
        # note: yields dual sectors so can have different sorting of _non_dual_sorted_sectors!
        # sectors can get sorted in VectorSpace.__init__() (e.g. in AbelianBackendVectorSpace)
        return self.__class__(self.symmetry,
                              self.symmetry.dual_sectors(self._non_dual_sorted_sectors),
                              self._sorted_multiplicities,
                              self.is_real,
                              not self.is_dual)

    def can_contract_with(self, other):
        """If self can be contracted with other.

        Equivalent to ``self == other.dual``"""
        # TODO should we compare the perm_sectors ?
        #  I.e. can VectorSpace(sym, [s1, s2]) be contracted with VectorSpace(sym, [s2, s1]).dual ?
        #  they have compatible _sectors and multiplicities, but different sector_perm
        if not isinstance(other, VectorSpace):
            return False
        if self.is_real != other.is_real:
            return False
        if self.is_dual == other.is_dual:
            return False
        if self.num_sectors != other.num_sectors:
            return False
        if self.symmetry != other.symmetry:
            return False
        # the _sectors (note the underscore!) of the dual space are the same as those of the
        # original space, while the other.sectors would be different.
        return np.all(self._non_dual_sorted_sectors == other._non_dual_sorted_sectors) and np.all(self._sorted_multiplicities == other._sorted_multiplicities)

    @property
    def is_trivial(self) -> bool:
        """Whether self is the trivial space.

        The trivial space is the one-dimensional space which consists only of the trivial sector,
        appearing exactly once. In a mathematical sense, the trivial sector _is_ the trivial space.
        """
        if self._non_dual_sorted_sectors.shape[0] != 1:
            return False
        # have already checked if there is more than 1 sector, so can assume self.multiplicities.shape == (1,)
        if self._sorted_multiplicities[0] != 1:
            return False
        if not np.all(self._non_dual_sorted_sectors[0] == self.symmetry.trivial_sector):
            return False
        return True

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the number of linearly independent symmetric tensors in this space."""
        # the trivial sector is by definition self-dual, so we can search self._non_dual_sorted_sectors,
        # even if self.is_dual is True.
        # OPTIMIZE use that _non_dual_sorted_sectors are lexsorted to shorten the loop if trivial sector is not in _non_dual_sorted_sectors.
        for s, m in zip(self._non_dual_sorted_sectors, self._sorted_multiplicities):
            if np.all(s == self.symmetry.trivial_sector):
                return m
        return 0

    def project(self, mask: ndarray):
        """Return a copy, keeping only the indices specified by `mask`.

        Parameters
        ----------
        mask : 1D array(bool)
            Whether to keep each of the indices in the *dense* array.

        Returns
        -------
        sector_idx_map : 1D array
            Map of sector indices.
            A non-negative entry ``m = sector_idx_map[n]`` indicates that ``sectors_after[m] == sectors_before[n]``
            and ``sector_idx_map[n] == -1`` indicates that ``sectors_before[n]`` is projected out entirely.
        sector_masks: list of 1D array
            For every *remaining* sector, the respective mask of length `sector_dim * old_multiplicity`.
        projected : :class:`VectorSpace`
            Copy of self after the projection, i.e. with ``projected.dim == np.sum(mask)``.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        projected = copy.copy(self)
        sector_masks = [mask[start:stop] for start, stop in self.slices]
        new_multiplicities = np.array([np.sum(sm) for sm in sector_masks])
        keep = np.nonzero(new_multiplicities)[0]
        sector_masks = [sector_masks[i] for i in keep]
        new_sector_number = len(sector_masks)
        sector_idx_map = np.full((new_sector_number,), -1, dtype=int)
        sector_idx_map[keep] = np.arange(new_sector_number)
        projected._sector_perm = perm = sector_idx_map[projected._sector_perm]
        projected._non_dual_sorted_sectors = projected._non_dual_sorted_sectors[keep]
        projected._sorted_multiplicities = new_multiplicities[keep]
        inv_perm = inverse_permutation(perm)
        unpermuted_slices = _calc_slices(self.symmetry, projected.sectors[inv_perm], projected._sorted_multiplicities[inv_perm])
        projected.slices = unpermuted_slices[perm]
        return sector_idx_map, sector_masks, projected

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

        # the _sectors are lexsorted, so we can just iterate over both of them
        n_self = 0
        for other_sector, other_mult in zip(other._non_dual_sorted_sectors, other._sorted_multiplicities):
            if np.all(self._non_dual_sorted_sectors[n_self] == other_sector):
                if self._sorted_multiplicities[n_self] > other_mult:
                    return False
                n_self += 1
            if n_self == self.num_sectors:
                # have checked all sectors of self
                return True
        # reaching this line means self has sectors which other does not have
        return False


def _calc_slices(symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray) -> ndarray:
    """Calculate the slices given sectors and multiplicities *in the dense order*, i.e. not sorted."""
    slices = np.zeros((len(sectors), 2), dtype=np.intp)
    # OPTIMIZE should we special case abelian symmetries to skip some multiplications by 1?
    slices[:, 1] = slice_ends = np.cumsum(multiplicities * symmetry.batch_sector_dim(sectors))
    slices[1:, 0] = slice_ends[:-1]
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
        If a backend is given, the backend-specific metadata will be set
        via `backend._fuse_spaces` and `backend.add_leg_metadata`
    _is_dual : bool | None
        Flag indicating wether the fusion space represents a dual (bra) space or a non-dual (ket) space.
        Per default (``_is_dual=None``), ``spaces[0].is_dual`` is used, i.e. the ``ProductSpace``
        will be a bra space if and only if its first factor is a bra space.

        .. warning ::
            When setting `_is_dual=True`, consider the notes below!

    _sectors, _multiplicities, _sector_perm:
        These inputs to VectorSpace.__init__ can optionally be passed to avoid recomputation.
        Specify either all or None of them.
    _sorted_slices:
        inputs to VectorSpace.__init__ can optionally be passed to avoid recomputation

    Notes
    -----
    While mathematically the dual of the product :math:`(V \otimes W)^*` is the same as the
    product of the duals :math:`V^* \otimes W^*`, we distinguish these two objects in the
    implementation. This allows us to fulfill all of the following constraints
    a) Have the same order of `_sectors` for a space and its dual, to make contractions easier.
    b) Consistently view every ProductSpace as a VectorSpace, i.e. have proper subclass behavior
        and in particular a well-behaved `is_dual` attribute.
    c) A ProductSpace can always be split into its :attr:`spaces`.

    As an example, consider two VectorSpaces ``V`` and ``W`` and the following four possible
    products::

        ==== ============================ ================== ========= ==================================
             Mathematical Expression      .spaces            .is_dual  ._non_dual_sorted_sectors
        ==== ============================ ================== ========= ==================================
        P1   :math:`V \otimes W`          [V, W]             False     P1._non_dual_sorted_sectors
        P2   :math:`(V \otimes W)^*`      [V.dual, W.dual]   True      P1._non_dual_sorted_sectors
        P3   :math:`V^* \otimes W^*`      [V.dual, W.dual]   False     dual(P1._non_dual_sorted_sectors)
        P4   :math:`(V^* \otimes W^*)^*`  [V, W]             True      dual(P1._non_dual_sorted_sectors)
        ==== ============================ ================== ========= ==================================

    They can be related to each other via the :attr:`dual` property or via :meth:`flip_is_dual`.
    In this example we have `P1.dual == P2`` and ``P3.dual == P4``, as well as
    ``P1.flip_is_dual() == P4`` and ``P2.flip_is_dual() == P3``.

    The mutually dual spaces, e.g. ``P1`` and ``P2``, can be contracted with each other, as they
    have opposite :attr:`is_dual` and matching :attr:`_non_dual_sorted_sectors`.
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
                 _sectors: SectorArray = None, _multiplicities: ndarray = None, _sector_perm: ndarray = None,
                 _sorted_slices: ndarray = None):
        if _is_dual is None:
            _is_dual = spaces[0].is_dual
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)

        if _sectors is None:
            assert all(arg is None for arg in [_multiplicities, _sector_perm])
            if backend is None:
                _sectors, _multiplicities, _sector_perm, metadata = _fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            else:
                _sectors, _multiplicities, _sector_perm, metadata = backend._fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            for key, val in metadata.items():
                setattr(self, key, val)
        else:
            assert _multiplicities is not None
        VectorSpace.__init__(self, symmetry=symmetry, sectors=_sectors, multiplicities=_multiplicities,
                             is_real=is_real, _is_dual=_is_dual, _sector_perm=_sector_perm, _sorted_slices=_sorted_slices)
        if backend is not None:
            backend.add_leg_metadata(self)

    def test_sanity(self):
        assert all(s.symmetry == self.symmetry for s in self.spaces)
        return super().test_sanity()

    # TODO python naming convention is snake case: as_vector_space
    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self._non_dual_sorted_sectors,  # underscore is important!
                           multiplicities=self._sorted_multiplicities,
                           is_real=self.is_real,
                           _is_dual=self.is_dual)

    def flip_is_dual(self) -> ProductSpace:
        """Return a ProductSpace isomorphic to self, which has the opposite is_dual attribute.

        This realizes the isomorphism between ``V.dual * W.dual`` and ``(V * W).dual``
        for `VectorSpace` ``V`` and ``W``.
        However, note that the returned space can often not be contracted with `self`
        since the order of the :attr:`sectors` might have changed.

        Backend-specific metadata may be lost.
        """
        # TODO test this
        _sectors = self.symmetry.dual_sectors(self._non_dual_sorted_sectors)
        
        return ProductSpace(spaces=self.spaces, _is_dual=not self.is_dual,
                            _sectors=self.symmetry.dual_sectors(self._non_dual_sorted_sectors),
                            _multiplicities=self._sorted_multiplicities,
                            _sector_perm=None, _sorted_slices=self._sorted_slices)

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return iter(self.spaces)

    def __repr__(self):
        lines = [f'{self.__class__.__name__}([']
        indent = '    '
        for s in self.spaces:
            lines.append(indent + repr(s).replace('\n', '\n' + indent))
        if self.is_dual:
            lines.append(']).dual')
        else:
            lines.append(f'])')
        return '\n'.join(lines)

    def __str__(self):
        res = f'ProductSpace(' \
              f'shape: [{", ".join(str(s.dim) for s in self.spaces)}]->{self.dim}, ' \
              f'is_dual: [{", ".join(str(s.is_dual) for s in self.spaces)}]->{self.is_dual}, ' \
              f'# sectors: [{", ".join(str(s.num_sectors) for s in self.spaces)}]->{self.num_sectors})'
        return res

    def __eq__(self, other):
        if not isinstance(other, ProductSpace):
            return NotImplemented
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

    def project(self, *args, **kwargs):
        """Convert self to VectorSpace and call :meth:`VectorSpace.project`.

        In general, this could be implemented for a ProductSpace, but would make
        `split_legs` more complicated, thus we keep it simple.
        If you really want to project and split afterwards, use the following work-around,
        which is for example used in :class:`~tenpy.algorithms.exact_diagonalization`:

        1) Create the full pipe and save it separately.
        2) Convert the Pipe to a Leg & project the array with it.
        3) [... do calculations ...]
        4) To split the 'projected pipe' of `A`, create an empty array `B` with the legs of A,
           but replace the projected leg by the full pipe. Set `A` as a slice of `B`.
           Finally split the pipe.
        """
        warnings.warn("Converting ProductSpace to VectorSpace for `project`", stacklevel=2)
        res = self.as_VectorSpace()
        return res.project(*args, **kwargs)


def _fuse_spaces(symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool
                 ) -> tuple[SectorArray, ndarray, ndarray, dict]:
    """This function is called as part of ProductSpace.__init__.
    The backend-specific metadata for the ProductSpace can be computed in parallel with the
    sectors and multiplicities of the product space, which is why this function lives in the backend.

    This default implementation does not add any additional metadata.
    Backends may override this.

    Returns
    -------
    _non_dual_sorted_sectors : 2D array of int
        The sectors to be stored as self._non_dual_sorted_sectors. May or may no be sorted, see `sector_perm`
    multiplicities : 1D array of int
        The multiplicities, in the order matching _sectors
    sector_perm : None | 1D array of int
        If the returned sectors are sorted, the :attr:`sector_perm` for self.
        If _sectors are not sorted, this is None.
    metadata : dict
        A dictionary with string keys and arbitrary values.
        These will be added as attributes of the ProductSpace
    """
    # TODO (JU) should we special case symmetry.fusion_style == FusionStyle.single ?
    if _is_dual:
        spaces = [s.dual for s in spaces] # directly fuse sectors of dual spaces.
        # This yields overall dual `sectors` to return, which we directly save in
        # self._non_dual_sorted_sectors, such that `self.sectors` (which takes a dual!) yields correct sectors
        # Overall, this ensures consistent sorting/order of sectors between dual ProductSpace!
    fusion = {tuple(s): m for s, m in zip(spaces[0].sectors, spaces[0]._sorted_multiplicities)}
    for space in spaces[1:]:
        new_fusion = {}
        for t_a, m_a in fusion.items():
            s_a = np.array(t_a)
            for s_b, m_b in zip(space.sectors, space._sorted_multiplicities):
                for s_c in symmetry.fusion_outcomes(s_a, s_b):
                    t_c = tuple(s_c)
                    n = symmetry._n_symbol(s_a, s_b, s_c)
                    new_fusion[t_c] = new_fusion.get(t_c, 0) + m_a * m_b * n
        fusion = new_fusion
        # by convention fuse spaces left to right, i.e. (...((0,1), 2), ..., N)
    _non_dual_sorted_sectors = np.asarray(list(fusion.keys()))
    multiplicities = np.asarray(list(fusion.values()))
    sector_perm = None  # note: in this implementation, this is always None, while in backend
                        #       specific implementation it might be given
    metadata = {}
    return _non_dual_sorted_sectors, multiplicities, sector_perm, metadata
