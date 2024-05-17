# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
from numpy import ndarray
import copy
import bisect
import itertools as it
from typing import TYPE_CHECKING, Sequence, Iterator

from tenpy.linalg.dummy_config import printoptions

from .symmetries import (Sector, SectorArray, Symmetry, ProductSymmetry, NoSymmetry, no_symmetry,
                         FusionStyle, SymmetryError)
from .misc import make_stride, find_row_differences, unstridify, join_as_many_as_possible
from ..tools.misc import inverse_permutation, rank_data, to_iterable
from ..tools.string import format_like_list

if TYPE_CHECKING:
    from .backends.abstract_backend import Backend, Block


__all__ = ['VectorSpace', 'ProductSpace']


class VectorSpace:
    r"""A vector space, which decomposes into sectors of a given symmetry.

    A vector space is characterized by a basis, and in particular a defined order of basis elements.
    Each basis vector lies in one of the sectors of the :attr:`symmetry`.

    For efficiency of the internal data manipulations, we want to use an internal basis,
    which is ordered such that::

        - The basis elements that belong to the same sector appear contiguously.
          This allows us to directly read-off the blocks that contain the free parameters.

        - The sectors are sorted. This makes look-ups more efficient.
          For a ket-space (``is_dual is False``), we sort according to ``np.lexsort(sectors.T)``.
          Regarding bra-spaces, see the notes on duality below.
    
    A VectorSpace instance captures this information, i.e. the permutation of the basis that
    achieves the above properties, which sectors appear and how often.

    We give special treatment to the dual ``V.dual`` of a given space ``V`` (with ``V.is_dual is False``).
    It is most convenient to facilitate contractions etc. to only switch a single boolean flag when
    going to the dual space and keeping all other attributes the same. Thus, both ``V.dual`` and
    ``V`` have the same :attr:`_non_dual_sectors`, :attr:`multiplicities` etc. and in particular
    the same basis order. As a consequence, the actual :attr:`sectors`, which are a property,
    computed from the :attr:`_non_dual_sectors` only when needed, that `V.dual` decomposes into
    are not sorted.

    .. note ::

        The notion of a basis and its associated attributes and functions is only a useful
        concept if ``symmetry.can_be_dropped``, i.e. if we can think of the sectors as 
        complex vector spaces. This is not the case, e.g. anyonic symmetries, or fermion grading.
        In these cases, there is are no substitute non-symmetric tensors that have the same behavior,
        e.g. because of non-trivial braiding which only the symmetric tensors can exhibit.
        In particular :attr:`sector_dims`, :attr:`slices` and :attr:`dim`, :attr:`sectors_of_basis`,
        :meth:`from_basis` etc. may not be defined and either be set to None or raise errors.

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
    sector_dims : 1D array of int | None
        If ``symmetry.can_be_dropped``, the integer dimension of each of the :attr:`sectors`.
    sector_qdims : 1D array of float
        The (quantum) dimension of each of the sectors. Unlike :attr:`sector_dims` this is always
        defined, but may not always be integer.
    dim : int | float
        The total dimension. Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
    slices : 2D numpy array of int | None
        For every sector ``sectors[n]``, the start ``slices[n, 0]`` and stop ``slices[n, 1]`` of
        indices (in the *internal* basis order) that belong to this sector.
        Conversely, ``basis_perm[slices[n, 0]:slices[n, 1]]`` are the elements of the public
        basis that live in ``sectors[n]``. Only available if ``symmetry.can_be_dropped``.
    is_real : bool
        If the space is over the real or complex numbers.
    is_dual : bool
        Whether this is the dual (a.k.a. bra) space or the regular (a.k.a. ket) space.
    basis_perm : ndarray | None
        The tensor manipulations of `tenpy.linalg` benefit from choosing a canonical order for the
        basis of vector spaces. This attribute translates between the "public" order of the basis,
        in which e.g. the inputs to :meth:`from_dense_block` are interpreted to this internal order,
        such that ``public_basis[basis_perm] == internal_basis``.
        This internal order is such that the basis vectors are grouped by sector and such that
        sectors occur in the canonical sorted order, see :attr:`_non_dual_sectors`.
        We store the inverse as `_inverse_basis_perm`.
        For `ProductSpace`s we always set a trivial permutation.
        We can translate indices as ``public_idx == basis_perm[internal_idx]``.
        Only available if ``symmetry.can_be_dropped``.
        ``_basis_perm`` is the internal version which may be ``None`` if the permutation is trivial.
    inverse_basis_perm : ndarray | None
        Inverse of :attr:`basis_perm`. ``_inverse_basis_perm`` is the internal version.
        Only available if ``symmetry.can_be_dropped``.
    sectors_of_basis : 2D numpy array of int | None
        The sectors of every element of the "public" computational basis.
        Multi-dimensional sectors (such as e.g. the spin-1/2 sector of SU(2)) are listed multiple
        times (once for each basis state in the multiplet), such that the length is the total
        dimension. Only available if ``symmetry.can_be_dropped``.
    _non_dual_sectors : 2D numpy array of int
        Internally stored version of :attr:`sectors`. For ket spaces (``is_dual=True``),
        these are the same as :attr:`sectors`, for bra spaces these are their duals.
        They are sorted such that ``np.lexsort(_non_dual_sectors.T)`` is trivial.
    """
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 basis_perm: ndarray = None, is_real: bool = False, _is_dual: bool = False):
        """Initialize a VectorSpace

        `VectorSpace.__init__` is not very user-friendly. Use :meth:`from_sectors` instead.
        
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
            If the space is over the real or complex (default) numbers.
        _is_dual : bool
            Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.

            .. warning :
                For ``_is_dual is True``, the passed `sectors` are interpreted as the sectors of the
                ("non-dual") ket-space isomorphic to self.
        """
        self.symmetry = symmetry
        self.is_real = is_real
        self.is_dual = _is_dual
        self._non_dual_sectors = sectors = np.asarray(sectors, dtype=int)
        if sectors.ndim != 2 or sectors.shape[1] != symmetry.sector_ind_len:
            msg = (f'Wrong sectors.shape: Expected (*, {symmetry.sector_ind_len}), '
                   f'got {sectors.shape}.')
            raise ValueError(msg)
        assert sectors.ndim == 2 and sectors.shape[1] == symmetry.sector_ind_len
        self.num_sectors = num_sectors = len(sectors)
        if multiplicities is None:
            self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int)
        else:
            self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int)
        if symmetry.can_be_dropped:
            self.sector_dims = sector_dims = symmetry.batch_sector_dim(sectors)
            self.sector_qdims = sector_dims
            slices = np.zeros((len(sectors), 2), dtype=np.intp)
            slices[:, 1] = slice_ends = np.cumsum(multiplicities * sector_dims)
            slices[1:, 0] = slice_ends[:-1]  # slices[0, 0] remains 0, which is correct
            self.slices = slices
            self.dim = np.sum(sector_dims * multiplicities)
        else:
            self.sector_dims = None
            self.sector_qdims = sector_qdims = symmetry.batch_qdim(sectors)
            self.slices = None
            self.dim = np.sum(sector_qdims * multiplicities)
        if basis_perm is None:
            self._basis_perm = None
            self._inverse_basis_perm = None
        else:
            if not symmetry.can_be_dropped:
                msg = f'basis_perm is meaningless for {symmetry}.'
                raise SymmetryError(msg)
            # OPTIMIZE set to None if trivial but explicit?
            self._basis_perm = basis_perm = np.asarray(basis_perm, dtype=int)
            self._inverse_basis_perm = inverse_permutation(basis_perm)

    def test_sanity(self):
        # sectors
        assert all(self.symmetry.is_valid_sector(s) for s in self._non_dual_sectors)
        assert len(self._non_dual_sectors) == self.num_sectors
        assert len(np.unique(self._non_dual_sectors, axis=0)) == self.num_sectors, 'duplicate sectors'
        assert self._non_dual_sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len)
        assert np.all(np.lexsort(self._non_dual_sectors.T) == np.arange(self.num_sectors)), 'wrong order'
        # multiplicities
        assert np.all(self.multiplicities > 0)
        assert self.multiplicities.shape == (self.num_sectors,)
        if self.symmetry.can_be_dropped:
            # slices
            assert self.slices.shape == (self.num_sectors, 2)
            slice_diffs = self.slices[:, 1] - self.slices[:, 0]
            assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self._non_dual_sectors))
            expect_diffs = self.sector_dims * self.multiplicities
            assert np.all(slice_diffs == expect_diffs)
            # slices should be consecutive
            if len(self.slices) > 0:
                assert self.slices[0, 0] == 0
                assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
                assert self.slices[-1, 1] == self.dim
            # basis_perm
            if self._basis_perm is None:
                assert self._inverse_basis_perm is None
            else:
                assert self._basis_perm.shape == (self.dim,)
                assert len(np.unique(self._basis_perm)) == self.dim
                assert np.all(self._basis_perm[self._inverse_basis_perm] == np.arange(self.dim))
        else:
            assert self.slices is None
            assert self._basis_perm is None
            assert self._inverse_basis_perm is None
        assert self.dim >= 0

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors_of_basis: Sequence[Sequence[int]],
                   is_real: bool = False):
        """Create a VectorSpace by specifying the sector of every basis element.

        This classmethod always creates a ket-space (i.e. ``res.is_dual is False``).
        Use ``VectorSpace.from_basis(...).dual`` for a bra-space.

        .. note ::
            Unlike :meth:`from_sectors`, this method expects the same sector to be listed
            multiple times, if the sector is multi-dimensional. The Hilbert Space of a spin-one-half
            D.O.F. can e.g. be created as ``VectorSpace.from_basis(su2, [[spin_half], [spin_half]])``
            or as ``VectorSpace.from_sectors(su2, [[1]])``. In the former case we need to list the
            same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        sectors_of_basis : iterable of iterable of int
            Specifies the basis. ``sectors_of_basis[n]`` is the sector of the ``n``-th basis element.
            In particular, for a ``d`` dimensional sector, we expect an integer multiple of ``d``
            occurrences. They need not be contiguous though. They will be grouped by order of
            appearance, such that they ``m``-th time a sector appears, that basis state is interpreted
            as the ``(m % d)``-th state of the multiplet.
        is_real : bool
            If the space is over the real or complex (default) numbers.

        See Also
        --------
        :attr:`sectors_of_basis`
            Reproduces the `sectors_of_basis` parameter.
        """
        if not symmetry.can_be_dropped:
            msg = f'from_basis is meaningless for {symmetry}.'
            raise SymmetryError(msg)
        sectors_of_basis = np.asarray(sectors_of_basis, dtype=int)
        assert sectors_of_basis.shape[1] == symmetry.sector_ind_len
        # note: numpy.lexsort is stable, i.e. it preserves the order of equal keys.
        basis_perm = np.lexsort(sectors_of_basis.T)
        sectors = sectors_of_basis[basis_perm]
        diffs = find_row_differences(sectors, include_len=True)
        sectors = sectors[diffs[:-1]]  # [:-1] to exclude len
        dims = symmetry.batch_sector_dim(sectors)
        num_occurrences = diffs[1:] - diffs[:-1]  # how often each appears in the input sectors_of_basis
        multiplicities, remainders = np.divmod(num_occurrences, dims)
        if np.any(remainders > 0):
            msg = ('Sectors must appear in whole multiplets, i.e. a number of times that is an '
                   'integer multiple of their dimension.')
            raise ValueError(msg)
        return cls(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                   basis_perm=basis_perm, is_real=is_real, _is_dual=False)

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions: list[VectorSpace],
                                    symmetry: Symmetry = None):
        """Create a VectorSpace with multiple independent symmetries.

        Parameters
        ----------
        independent_descriptions : list of :class:`VectorSpace`
            Each entry describes the resulting :class:`VectorSpace` in terms of *one* of
            the independent symmetries. Spaces with a :class:`NoSymmetry` are ignored.
        symmetry: :class:`~tenpy.linalg.groups.Symmetry`, optional
            The resulting symmetry can optionally be passed. We assume without checking that
            it :meth:`~tenpy.linalg.groups.Symmetry.is_same_symmetry` as the default
            ``ProductSymmetry.from_nested_factors([s.symmetry for s in independent_descriptions])``.

        Returns
        -------
        :class:`VectorSpace`
            A space with the overall `symmetry`.
        """
        if not all(sp.symmetry.can_be_dropped for sp in independent_descriptions):
            msg = f'from_independent_symmetries is not supported for {symmetry}.'
            # TODO is there a way to define this?
            #      the straight-forward picture works only if we have a vectorspace and can identify states.
            raise SymmetryError(msg)
        assert len(independent_descriptions) > 0
        dim = independent_descriptions[0].dim
        assert all(s.dim == dim for s in independent_descriptions)
        # ignore those with np_symmetry
        independent_descriptions = [s for s in independent_descriptions if s.symmetry != no_symmetry]
        if len(independent_descriptions) == 0:
            # all descriptions had no_symmetry
            return cls.from_trivial_sector(dim=dim)
        if symmetry is None:
            symmetry = ProductSymmetry.from_nested_factors(
                [s.symmetry for s in independent_descriptions]
            )
        sectors_of_basis = np.concatenate([s.sectors_of_basis for s in independent_descriptions], axis=1)
        if (is_real := any(s.is_real for s in independent_descriptions)):
            assert all(s.is_real for s in independent_descriptions)
        return cls.from_basis(symmetry=symmetry, sectors_of_basis=sectors_of_basis, is_real=is_real)

    @classmethod
    def from_trivial_sector(cls, dim: int, symmetry: Symmetry = no_symmetry, is_real: bool = False,
                            is_dual: bool = False):
        """Create a VectorSpace that lives in the trivial sector (i.e. it is symmetric).

        Parameters
        ----------
        dim : int
            The dimension of the space.
        symmetry : :class:`~tenpy.linalg.groups.Symmetry`
            The symmetry of the space. By default, we use `no_symmetry`.
        is_real, is_dual : bool
            If the space should be real / dual.
        """
        return cls(symmetry=symmetry, sectors=[symmetry.trivial_sector], multiplicities=[dim],
                   is_real=is_real, _is_dual=is_dual)

    @classmethod
    def from_sectors(cls, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                     basis_perm: ndarray = None, is_real: bool = False, return_perm: bool = False,
                     _is_dual: bool = False):
        """Like constructor, but fewer requirements on `sectors`.

        .. note ::
            Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
            only once to mean its entire multiplet of basis states. The Hilbert Space of a spin-1/2
            D.O.F. can e.g. be created as ``VectorSpace.from_basis(su2, [spin_half, spin_half])``
            or as ``VectorSpace.from_sectors(su2, [spin_half])``. In the former case we need to
            list the same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        sectors: 2D array_like of int
            The sectors of the symmetry that compose this space.
            Can be in any order and may contain duplicates.
        multiplicities: 1D array_like of int, optional
            How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
            ``sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
            If not given, a multiplicity ``1`` is assumed for all `sectors`.
        basis_perm : ndarray, optional
            The permutation from the desired public basis to the basis described by `sectors`
            and `multiplicities`. Per default the trivial permutation ``[0, 1, 2, ...]`` is used.
        is_real : bool
            If the space is over the real or complex numbers.
        _is_dual : bool
            Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.

            .. warning :
                For ``_is_dual is True``, the passed `sectors` are interpreted as the sectors of the
                ("non-dual") ket-space isomorphic to self.

        Returns
        -------
        space : VectorSpace
        sector_sort : bool, optional
            Only returned `if return_perm`. The permutation that sorts the `sectors`.
        """
        sectors = np.asarray(sectors, dtype=int)
        assert sectors.ndim == 2 and sectors.shape[1] == symmetry.sector_ind_len
        if multiplicities is None:
            multiplicities = np.ones((len(sectors),), dtype=int)
        else:
            multiplicities = np.asarray(multiplicities, dtype=int)
            assert multiplicities.shape == ((len(sectors),))
        if basis_perm is None:
            basis_perm = np.arange(np.sum(symmetry.batch_sector_dim(sectors) * multiplicities))
        num_states = symmetry.batch_sector_dim(sectors) * multiplicities
        basis_slices = np.concatenate([[0], np.cumsum(num_states)], axis=0)
        # sort sectors
        sort = np.lexsort(sectors.T)
        sectors = sectors[sort]
        multiplicities = multiplicities[sort]
        mult_slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
        basis_perm = np.concatenate([basis_perm[basis_slices[i]: basis_slices[i + 1]] for i in sort])
        # merge duplicate sectors (does not affect basis_perm)
        diffs = find_row_differences(sectors, include_len=True)
        multiplicities = mult_slices[diffs[1:]] - mult_slices[diffs[:-1]]
        sectors = sectors[diffs[:-1]]  # [:-1] to exclude len
        res = cls(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                  basis_perm=basis_perm, is_real=is_real, _is_dual=_is_dual)
        if return_perm:
            return res, sort
        return res

    @classmethod
    def null_space(cls, symmetry: Symmetry, is_real: bool = False, is_dual: bool = False
                   ) -> VectorSpace:
        """The zero-dimensional space, i.e. the span of the empty set."""
        sectors = np.zeros((0, symmetry.sector_ind_len), dtype=symmetry.trivial_sector.dtype)
        multiplicities = np.zeros(0, int)
        return cls(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                   is_real=is_real, _is_dual=is_dual)

    @property
    def sectors(self):
        # OPTIMIZE cachedproperty?
        if self.is_dual:
            return self.symmetry.dual_sectors(self._non_dual_sectors)
        else:
            return self._non_dual_sectors

    @property
    def sectors_of_basis(self):
        """The sector for each basis vector, like the input of :meth:`from_basis`."""
        if not self.symmetry.can_be_dropped:
            msg = f'sectors_of_basis is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        # build in internal basis, then permute
        res = np.zeros((self.dim, self.symmetry.sector_ind_len), dtype=int)
        # multi-dimensional sectors are captured by compatible slices.
        for sect, slc in zip(self.sectors, self.slices):
            res[slice(*slc), :] = sect[None, :]
        if self._inverse_basis_perm is not None:
            res = res[self._inverse_basis_perm]
        return res

    def as_VectorSpace(self):
        """Convert to a :class:`VectorSpace` which is *not* a :class:`ProductSpace`."""
        # already a VectorSpace. nothing to do.
        return self

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable,
                        backend: Backend = None) -> VectorSpace:
        """Change the symmetry by specifying how the sectors change.

        Parameters
        ----------
        symmetry : :class:`~tenpy.linalg.groups.Symmetry`
            The symmetry of the new space
        sector_map : function (SectorArray,) -> (SectorArray,)
            A mapping of sectors (2D ndarrays of int), such
            that ``new_sectors = sector_map(old_sectors)``.
            The map is assumed to cooperate with duality, i.e. we assume without checking that
            ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
            ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
            TODO do we need to assume more, i.e. compatibility with fusion?
        backend : :class: `~tenpy.linalg.backends.abstract_backend.Backend`
            This parameter is ignored. We only include it to have matching signatures
            with :meth:`ProductSpace.change_symmetry`.

        Returns
        -------
        :class:`VectorSpace`
            A space with the new symmetry. The order of the basis is preserved, but every
            basis element lives in a new sector, according to `sector_map`.
        """
        # OPTIMIZE could we directly map the _non_dual sectors ?
        res = VectorSpace.from_sectors(
            symmetry=symmetry, sectors=sector_map(self.sectors), multiplicities=self.multiplicities,
            basis_perm=self._basis_perm, is_real=self.is_real
        )
        if self.is_dual:
            res = res.dual
        return res

    def drop_symmetry(self, which: int | list[int] = None, remaining_symmetry: Symmetry = None):
        """Drop some or all symmetries.

        Parameters
        ----------
        which : None | (list of) int
            If ``None`` (default) the entire symmetry is dropped and the result has ``no_symmetry``.
            An integer or list of integers assume that ``self.symmetry`` is a ``ProductSymmetry``
            and indicates which of its factors to drop.
        remaining_symmetry : :class:`~tenpy.linalg.groups.Symmetry`, optional
            The resulting symmetry can optionally be passed, e.g. to control its name.
            Should be a :class:`~tenpy.linalg.groups.NoSymmetry` if all symmetries are
            dropped or :class:`~tenpy.linalg.groups.ProductSymmetry` otherwise.
            Is not checked for correctness (TODO or should we?).
    
        Returns
        -------
        A new VectorSpace instance with reduced `symmetry`.
        """
        if which is None:
            pass
        elif which == []:
            return self
        elif isinstance(self.symmetry, ProductSymmetry):
            which = to_iterable(which)
            num_factors = len(self.symmetry.factors)
            # normalize negative indices to be in range(num_factors)
            for i, w in enumerate(which):
                if not -num_factors <= w < num_factors:
                    raise ValueError(f'which entry {w} out of bounds for {num_factors} symmetries.')
                if w < 0:
                    which[i] += num_factors
            if len(which) == num_factors:
                which = None
        elif which == 0 or which == [0]:
            which = None
        else:
            msg = f'Can not drop which={which} for a single (non-ProductSymmetry) symmetry.'
            raise ValueError(msg)
        
        if which is None:
            return VectorSpace(no_symmetry, sectors=[no_symmetry.trivial_sector],
                               multiplicities=[self.dim], basis_perm=self._basis_perm,
                               is_real=self.is_real, _is_dual=self.is_dual)

        if remaining_symmetry is None:
            factors = [f for i, f in enumerate(self.symmetry.factors) if i not in which]
            if len(factors) == 1:
                remaining_symmetry = factors[0]
            else:
                remaining_symmetry = ProductSymmetry(factors)
        # TODO check compatible otherwise?

        mask = np.ones((self.symmetry.sector_ind_len,), dtype=bool)
        for i in which:
            start, stop = self.symmetry.sector_slices[i:i + 2]
            mask[start:stop] = False

        return self.change_symmetry(symmetry=remaining_symmetry,
                                    sector_map=lambda sectors: sectors[:, mask])

    def sector(self, i: int) -> Sector:
        """Return the `i`-th sector. Equivalent to ``self.sectors[i]``."""
        sector = self._non_dual_sectors[i, :]
        if self.is_dual:
            return self.symmetry.dual_sector(sector)
        return sector

    def take_slice(self, blockmask) -> VectorSpace:
        """Take a "slice" of the leg, keeping only some of the basis states.

        Any ProductSpace structure is lost.

        Parameters
        ----------
        blockmask : 1D array-like of bool
            For every basis state of self, if it should be kept (``True``) or discarded (``False``).
        """
        if not self.symmetry.can_be_dropped:
            msg = f'take_slice is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        blockmask = np.asarray(blockmask, dtype=bool)
        if self._basis_perm is not None:
            blockmask = blockmask[self._basis_perm]
        sectors = []
        mults = []
        dims = self.symmetry.batch_sector_dim
        for a, d_a, slc in zip(self._non_dual_sectors, self.sector_dims, self.slices):
            sector_mask = blockmask[slice(*slc)]
            per_basis_state = np.reshape(sector_mask, (-1, d_a))
            if not np.all(per_basis_state == per_basis_state[:, 0, None]):
                msg = 'Multiplets need to be kept or discarded as a whole.'
                raise ValueError(msg)
            num_kept = np.sum(sector_mask)
            assert num_kept % d_a == 0  # should be guaranteed by check above already, but to be sure...
            mult = num_kept // d_a
            if mult > 0:
                sectors.append(a)
                mults.append(mult)
        if len(sectors) == 0:
            sectors = np.zeros(shape=(0, self.symmetry.sector_ind_len),
                               dtype=self.symmetry.trivial_sector.dtype)
            mults = np.zeros(0, int)
        # build basis_perm for small leg.
        # it is determined by demanding
        #    a) that the following diagram commutes
        #
        #        (self, public) ---- self.basis_perm ---->  (self, internal)
        #         |                                           |
        #         v public_blockmask                          v projection_internal
        #         |                                           |
        #        (res, public) ----- small_leg_perm ----->  (res, internal)
        #
        #    b) that projection_internal is also just a mask (i.e it preserves ordering)
        #       which is given by public_blockmask[self.basis_perm]
        #
        # this allows us to internally (e.g. in the abelian backend) store only 1D boolean masks
        # as blocks.
        #
        # note that we have already converted blockmask to public_blockmask[self.basis_perm] above
        basis_perm = rank_data(self.basis_perm[blockmask])
        return VectorSpace(symmetry=self.symmetry, sectors=sectors, multiplicities=mults,
                           basis_perm=basis_perm, is_real=self.is_real, _is_dual=self.is_dual)

    def parse_index(self, idx: int) -> tuple[int, int]:
        """Utility function to translate an index for this VectorSpace.

        Parameters
        ----------
        idx : int
            An index of the leg, labelling an element of the public computational basis of self.

        Returns
        -------
        sector_idx : int
            The index of the corresponding sector,
            indicating that the `idx`-th basis element lives in ``self.sectors[sector_idx]``.
        multiplicity_idx : int
            The index "within the sector", in ``range(sector_dim * self.multiplicities[sector_index])``.
        """
        if not self.symmetry.can_be_dropped:
            msg = f'parse_index is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is not None:
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
        if self.is_dual:
            # lookup dual(sector) in _non_dual_sectors instead.
            # that lookup is (or will be?) optimized, since the _non_dual_sectors are sorted.
            # plus, we do not need to form the self.sectors (taking *all* duals).
            sector = self.symmetry.dual_sector(sector)
        return self._non_dual_sectors_where(sector)

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
        """The multiplicity of the given sector.

        Returns 0 if self does not have that sector.
        """
        idx = self.sectors_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]

    def _non_dual_sector_multiplicity(self, sector: Sector) -> int:
        """The multiplicity of the given _non_dual_sector.

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
        sectors = self._non_dual_sectors  # printing the non-dual space, eventually followed by .dual

        if self.sectors.size < multiline_threshold:
            elements = [
                f'VectorSpace({self.symmetry!r}',
                *(['is_real=True'] if self.is_real else []),
                f'sectors={format_like_list(self.symmetry.sector_str(s) for s in sectors)}',
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
                f'{indent}sectors={format_like_list(self.symmetry.sector_str(s) for s in sectors)},',
                f'{indent}multiplicities={format_like_list(self.multiplicities)}',
                f'){is_dual_str}'
            ]
            if all(len(l) <= printoptions.linewidth for l in lines):
                return '\n'.join(lines)

        # add as many sectors as possible before linewidth is reached
        # save most recent suggestion in variable res. if new suggestion is too long, return res.
        res = f'VectorSpace({self.symmetry!r}, ...)'
        if len(res) > printoptions.linewidth:
            return 'VectorSpace(...)'
        prio = self._sector_print_priorities()
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
            res = '\n'.join(lines)
        return res

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

        basis_perm_header = 'basis_perm: '
        if self._basis_perm is None:
            basis_perm_str = 'None (no permutation)'
        else:
            basis_perm_str = join_as_many_as_possible(
                [str(x) for x in self.basis_perm],
                separator=' ',
                max_len=printoptions.linewidth - len(basis_perm_header) - len(indent) - 2,  # -2 for the brackets
            )
            basis_perm_str = '[' + basis_perm_str + ']'
        lines = [
            'VectorSpace(',
            *([f'{indent}is_real=True'] if self.is_real else []),
            f'{indent}symmetry: {self.symmetry!s}',
            f'{indent}dim: {self.dim}',
            f'{indent}is_dual: {self.is_dual}',
            f'{indent}{basis_perm_header}{basis_perm_str}',
            f'{indent}num sectors: {self.num_sectors}',
        ]
        # determine sectors: list[str] and mults: list[str]
        if len(lines) + self.num_sectors <= printoptions.maxlines_spaces:
            sectors = [self.symmetry.sector_str(s) for s in sectors]
            mults = [str(m) for m in mults]
        else:
            # not all sectors are shown. add how many there are
            lines[4:4]= []
            prio = self._sector_print_priorities()
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

    def flip_is_dual(self, return_perm: bool = False) -> VectorSpace:
        """Return a copy with opposite :attr:`is_dual` flag.

        The result has the same :attr:`sectors` and :attr:`multiplicities` *up to permutation*,
        such that they have the exact same :attr:`sectors_of_basis`.
        Therefore, the result can replace `self` on a tensor without affecting the charge rule.
        Note that this leg is neither equal to `self` nor can it be contracted with `self`.

        Returns
        -------
        flipped : VectorSpace
            The flipped space.
        sector_sort : bool, optional
            Only returned `if return_perm`.
            The permutation that sorts the sectors after taking the duals.

        """
        return VectorSpace.from_sectors(
            symmetry=self.symmetry, sectors=self.symmetry.dual_sectors(self._non_dual_sectors),
            multiplicities=self.multiplicities, basis_perm=self._basis_perm, is_real=self.is_real,
            return_perm=return_perm, _is_dual=not self.is_dual
        )

    def is_equal_or_dual(self, other: VectorSpace) -> bool:
        """If another VectorSpace is equal to *or* dual of `self`."""
        if not isinstance(other, VectorSpace):
            return False
        if isinstance(self, ProductSpace) != isinstance(other, ProductSpace):
            return False
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
        if (self._basis_perm is not None) or (other._basis_perm is not None):
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
    def basis_perm(self):
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._basis_perm is None:
            return np.arange(self.dim)
        return self._basis_perm

    @property
    def inverse_basis_perm(self):
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is None:
            return np.arange(self.dim)
        return self._inverse_basis_perm

    @property
    def is_trivial(self) -> bool:
        """Whether self is the trivial space.

        The trivial space is the one-dimensional space which consists only of the trivial sector,
        appearing exactly once. In a mathematical sense, the trivial sector _is_ the trivial space.

        TODO name is maybe not ideal... the VectorSpace.null_space could also be called "trivial"
             this space is the unit of fusion
        """
        if len(self._non_dual_sectors) != 1:
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
        # even if self.is_dual is True. Also, its dimension must be 1.
        # OPTIMIZE cache?
        return self.sector_multiplicity(self.symmetry.trivial_sector)

    def is_subspace_of(self, other: VectorSpace) -> bool:
        """Whether self is a subspace of other.

        This function considers both spaces purely as `VectorSpace`s and ignores a possible
        `ProductSpace` structure.
        Per convention, self is never a subspace of other, if the :attr:`is_dual` or the
        :attr:`symmetry` are different.
        The :attr:`basis_perm`s are not considered.
        """
        if self.is_dual != other.is_dual:
            return False
        if self.symmetry != other.symmetry:
            return False
        if self.num_sectors == 0:
            return True
        
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

    def _sector_print_priorities(self):
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

    def direct_sum(self, *others: VectorSpace) -> VectorSpace:
        """Form the direct sum (i.e. stacking).

        The basis of the new space results from concatenating the individual bases.
        
        Spaces must have the same symmetry, is_dual and is_real.
        The result is a space with the same symmetry, is_dual and is_real, whose sectors are those
        that appear in any of the spaces and multiplicities are the sum of the multiplicities
        in each of the spaces. Any ProductSpace structure is lost.
        """
        if not others:
            return self.as_VectorSpace()

        symmetry = self.symmetry
        assert all(o.symmetry == symmetry for o in others)
        is_real = self.is_real
        assert all(o.is_real == is_real for o in others)
        is_dual = self.is_dual
        assert all(o.is_dual == is_dual for o in others)


        if symmetry.can_be_dropped:
            offsets = np.cumsum([self.dim, *(o.dim for o in others)])
            basis_perm = np.concatenate(
                [self.basis_perm] + [o.basis_perm + n for o, n in zip(others, offsets)]
            )
        else:
            basis_perm = None
        sectors = np.concatenate([self._non_dual_sectors, *(o._non_dual_sectors for o in others)])
        multiplicities = np.concatenate([self.multiplicities, *(o.multiplicities for o in others)])
        res = VectorSpace.from_sectors(symmetry=symmetry, sectors=sectors,
                                       multiplicities=multiplicities,basis_perm=basis_perm,
                                       is_real=is_real)
        res.is_dual = is_dual
        return res


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
    TODO (JU) actually the product of graded spaces *is* associative up to a canonical isomorphism,
         which is typically ignored. I think we can sweep this under the rug at this relatively
         public level of docs...
         As an implementation detail, it is useful to work with the trees, because you can then
         build the basis transformation of an N-space fusion from the basis transformations of
         successive pairwise fusions. If you want to use this strategy, again as an internal
         implementation detail, not as a property of the product space, you should have a
         canonical order of pairwise fusions, i.e. a tree.
         I think the only thing people should be told at this level is that the order
         of :attr:`spaces` has meaning and you should not mess with it. But since that determines
         the :attr:`basis_perm` of the ProductSpace, this should be clear anyway.

    TODO elaborate on basis transformation from uncoupled to coupled basis.

    Attributes
    ----------
    basis_perm : ndarray
        For `ProductSpace`s, this is always the trivial permutation ``[0, 1, 2, 3, ...]``.
    _fusion_outcomes_sort : 1D array | None
        Only available for abelian symmetries.
        The permutation that ``np.lexsort( .T)``s the list of all possible fusion outcomes.
        Note that that list contains duplicates.
        Shape is ``(np.prod([sp.num_sectors for sp in self.spaces]))``.  (TODO this true for FusionTree?)
    _fusion_outcomes_inverse_sort : 1D ndarray | None
        Only available for abelian symmetries.
        Inverse permutation of :attr:`_fusion_outcomes_sort`.

    Parameters
    ----------
    spaces:
        The factor spaces that multiply to this space.
        The resulting product space can always be split back into these.
    backend : Backend | None
        If a backend is given, the backend-specific metadata will be set via ``backend._fuse_spaces``.
    symmetry, is_real : optional
        Like arguments to :meth:`VectorSpace.__init__`.
        Required if ``len(spaces) == 0``. Ignored otherwise.
    _is_dual : bool | None
        Flag indicating wether the fusion space represents a dual (bra) space or a non-dual (ket) space.
        Per default (``_is_dual=None``), ``spaces[0].is_dual`` is used, i.e. the ``ProductSpace``
        will be a bra space if and only if its first factor is a bra space.
        An empty product is not dual by default.
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
    You can write ``P2 == ProductSpace([V, W]).dual``, though.

    Note that the default behavior of the `_is_dual` argument guarantees that
    `ProductSpace(some_space)` is contractible with `ProductSpace([s.dual for s in some_spaces])`.
    """
    def __init__(self, spaces: list[VectorSpace], backend: Backend = None,
                 symmetry: Symmetry = None, is_real: bool = False, _is_dual: bool = None,
                 _sectors: SectorArray = None, _multiplicities: ndarray = None):
        if _is_dual is None:
            if len(spaces) > 0:
                _is_dual = spaces[0].is_dual
            else:
                _is_dual = False  # not dual by default.
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        if len(spaces) == 0:
            assert symmetry is not None, 'symmetry arg is required if spaces is empty'
            assert is_real is not None, 'is_real arg is requires if spaces is empty'
            # the empty product is the monoidal unit, i.e. the trivial sector.
            _sectors = symmetry.trivial_sector[None, :]
            _multiplicities = np.array([1])
        else:
            symmetry = spaces[0].symmetry
            is_real = spaces[0].is_real
            assert all(s.symmetry == symmetry for s in spaces)
            assert all(space.is_real == is_real for space in spaces)

        if _sectors is None:
            assert _multiplicities is None
            if backend is None:
                _sectors, _multiplicities, fusion_outcomes_sort, metadata = _fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            else:
                _sectors, _multiplicities, fusion_outcomes_sort, metadata = backend._fuse_spaces(
                    symmetry=spaces[0].symmetry, spaces=spaces, _is_dual=_is_dual
                )
            self._fusion_outcomes_sort = fusion_outcomes_sort
            if fusion_outcomes_sort is None:
                self._fusion_outcomes_inverse_sort = None
            else:
                self._fusion_outcomes_inverse_sort = inverse_permutation(fusion_outcomes_sort)
            for key, val in metadata.items():
                setattr(self, key, val)
        else:
            assert _multiplicities is not None
        VectorSpace.__init__(self, symmetry=symmetry, sectors=_sectors, multiplicities=_multiplicities,
                             is_real=is_real, _is_dual=_is_dual, basis_perm=None)

    def test_sanity(self):
        for s in self.spaces:
            assert s.symmetry == self.symmetry
            s.test_sanity()
        assert self._basis_perm is None
        return super().test_sanity()

    @classmethod
    def from_basis(cls, *a, **kw):
        raise NotImplementedError('from_basis can not create ProductSpaces')

    @classmethod
    def from_independent_symmetries(cls, *a, **kw):
        raise NotImplementedError('from_independent_symmetries can not create ProductSpaces')

    @classmethod
    def from_trivial_sector(cls, *a, **kw):
        raise NotImplementedError('from_trivial_sector can not create ProductSpaces')

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

    def get_flat_spaces(self) -> list[VectorSpace]:
        """Flatten the potential nesting of product structures.

        Returns
        -------
        factor_spaces : list of :class:`VectorSpace`
            A list of spaces which are not :class:`ProductSpace`.
            Those are all the spaces that appear in ``self.spaces``, either directly or nested,
            e.g. either as ``self.spaces[i]`` or ``self.spaces[j].spaces[k]`` etc.
        """
        factors_spaces = []
        for space in self.spaces:
            if isinstance(space, ProductSpace):
                factors_spaces.extend(space.get_flat_spaces())
            else:
                factors_spaces.append(space)
        return factors_spaces
        
    def as_flat_product(self) -> ProductSpace:
        """Create a flat ProductSpace from potentially nested ProductSpace.

        Transform a tree-like structure, where the :attr:`spaces` may themselves be ProductSpaces
        to a flat structure where they are not.
        """
        return ProductSpace(self.get_flat_spaces(), _is_dual=self.is_dual,
                            _sectors=self._non_dual_sectors, _multiplicities=self.multiplicities)
    
    def get_basis_transformation(self) -> np.ndarray:
        r"""Get the basis transformation from uncoupled to coupled basis.

        The uncoupled basis of the ProductSpace :math:`P = V \otimes W \otimes \dots \otimes Z`
        is given by products of the individual ("uncoupled") basis elements, i.e. by elements of
        the form :math:`v_{i_1} \otimes w_{i_2} \otimes \dots \otimes z_{i_n}`.
        In particular, the order for the uncoupled basis does *not*
        consider :attr:`VectorSpace.basis_perm`, i.e. it is in general not grouped by sectors.
        The coupled basis is organized by sectors. See the :class:`ProductSpace` docstring for
        details.

        Returns
        -------
        trafo : ndarray
            A numpy array with shape ``(*space.dim for space in self.spaces, self.dim)``.
            The first axes go over the basis for each of the :attr:`spaces` (in public order).
            The last axis goes over the coupled basis of self.
            The entries are coefficients of the basis transformation such that

            .. math ::
                \ket{c} = \sum_{i_1, \dots, i_N} \texttt{trafo[i1, ..., iN]}
                          \ket{i_1} \otimes \dots \otimes \ket{i_n}

        See Also
        --------
        get_basis_transformation_perm
            For abelian symmetries, the basis transformation, when reshaped to a square matrix,
            is just a permutation matrix. This method directly returns the permutation.

        Examples
        --------
        See :meth:`get_basis_transformation_perm` for an example with an abelian symmetry.

        Consider two spin-1/2 sites with SU(2) conservation.
        For both sites, we choose the basis :math:`\set{\ket{\uparrow}, \ket{\downarrow}}`.
        The uncoupled basis for the product is

        .. math ::
            \set{\ket{\uparrow;\uparrow}, \ket{\uparrow;\downarrow}, \ket{\downarrow;\uparrow},
                 \ket{\downarrow;\downarrow}}

        But the coupled basis is given by a basis transformation

        .. math ::
            \ket{s=0, m=0} = \frac{1}{\sqrt{2}}\left( \ket{\uparrow;\downarrow} - \ket{\downarrow;\uparrow} \right)
            \ket{s=1, m=-1} = \ket{\downarrow;\downarrow}
            \ket{s=1, m=0} = \frac{1}{\sqrt{2}}\left( \ket{\uparrow;\downarrow} + \ket{\downarrow;\uparrow} \right)
            \ket{s=1, m=1} = \ket{\uparrow;\uparrow}

        Such that we get

        TODO unskip the doctest

        .. testsetup :: get_basis_transformation
            from tenpy.linalg import ProductSpace, VectorSpace, su2_symmetry

        .. doctest :: get_basis_transformation
            :options: +SKIP

            >>> spin_one_half = [1]  # sectors are labelled by 2*S
            >>> site = VectorSpace(su2_symmetry, [spin_one_half])
            >>> prod_space = ProductSpace([site, site])
            >>> trafo = prod_space.get_basis_transformation()
            >>> trafo[:, :, 0]  # | s=0, m=0 >
            array([[ 0.        ,  0.70710678],
                   [-0.70710678,  0.        ]])
            >>> trafo[:, :, 1]  # |s=0, m=-1 >
            array([[0., 0.],
                   [0., 1.]])
            >>> trafo[:, :, 2]  # | s=1, m=0 >
            array([[0.        , 0.70710678],
                   [0.70710678, 0.        ]])
            >>> trafo[:, :, 3]  # | s=1, m=1 >
            array([[1., 0.],
                   [0., 0.]])
            
        """
        if self.symmetry.is_abelian:
            transform = np.zeros((self.dim, self.dim), dtype=np.intp)
            perm = self.get_basis_transformation_perm()
            transform[np.ix_(perm, range(self.dim))] = 1.
            return np.reshape(transform, (*(s.dim for s in self.spaces), self.dim))
        raise NotImplementedError  # TODO for FusionTree this is just fusion_tree.__array__, right?

    def get_basis_transformation_perm(self):
        r"""Get the permutation equivalent to :meth:`get_basis_transformation`.

        This is only defined for abelian symmetries, since then :meth:`get_basis_transformation`
        gives (up to reshaping) a permutation matrix::

            permutation_matrix = self.get_basis_transformation().reshape((self.dim, self.dim))

        which only has nonzero entries at ``permutation_matrix[perm[i], i] for i in range(self.dim)``.
        This method returns the permutation ``perm``.

        Examples
        --------
        Consider two spin-1 sites with Sz_parity conservation.
        For both sites, we choose the basis :math:`\set{\ket{+}, \ket{0}, \ket{-}}`.
        Now the uncoupled basis for the product is

        .. math ::
            \set{\ket{+;+}, \ket{+;0}, \ket{+;-}, \ket{0;+}, \ket{0;0}, \ket{0;-},
                 \ket{-;+}, \ket{-;0}, \ket{-;-}}

        Which becomes the following after grouping and sorting by sector

        .. math ::
            \set{\ket{+;+}, \ket{+;-}, \ket{0;0}, \ket{-;+}, \ket{-;-},
                 \ket{+;0}, \ket{0;+}, \ket{0;-}, \ket{-;0}}

        Such that we get

        .. testsetup :: get_basis_transformation_perm
            import numpy as np
            from tenpy.linalg import ProductSpace, VectorSpace, z2_symmetry

        .. doctest :: get_basis_transformation_perm

            >>> even, odd = [0], [1]
            >>> spin1 = VectorSpace.from_basis(z2_symmetry, [even, odd, even])
            >>> product_space = ProductSpace([spin1, spin1])
            >>> perm = product_space.get_basis_transformation_perm()
            >>> perm
            array([0, 2, 4, 6, 8, 1, 3, 5, 7])
            >>> transform = np.zeros((9, 9))
            >>> transform[np.ix_(perm, range(9))] = 1.
            >>> np.all(product_space.get_basis_transformation() == transform.reshape((3, 3, 9)))
            True
        """
        # TODO expand testing
        if not self.symmetry.is_abelian:
            raise SymmetryError('For non-abelian symmetries use get_basis_transformation instead.')
        # C-style for compatibility with e.g. numpy.reshape
        strides = make_stride(shape=[space.dim for space in self.spaces], cstyle=True)
        order = unstridify(self._get_fusion_outcomes_perm(), strides).T  # indices of the internal bases
        return sum(stride * space.inverse_basis_perm[p]
                   for stride, space, p in zip(strides, self.spaces, order))

    def _get_fusion_outcomes_perm(self):
        r"""Get the permutation introduced by the fusion.

        This permutation arises as follows:
        For each of the :attr:`spaces` consider all sectors by order of appearance in the internal
        order, i.e. in :attr:`VectorSpace.sectors``. Take all combinations of sectors from all the
        spaces in C-style order, i.e. varying those from the last space the fastest.
        For each combination, take all of its fusion outcomes (TODO define order for FusionTree).
        The target permutation np.lexsort( .T)s the resulting list of sectors.
        """
        # OPTIMIZE this probably not the most efficient way to do this, but it hurts my brain
        #  and i need to get this work, if only in an ugly way...
        
        # j : multi-index into the uncoupled private basis, i.e. into the C-style product of internal bases of the spaces
        # i : index of self.spaces
        # s : index of the list of all fusion outcomes / fusion channels
        dim_strides = make_stride([sp.dim for sp in self.spaces])  # (num_spaces,)
        sector_strides = make_stride([sp.num_sectors for sp in self.spaces])  # (num_spaces,)
        num_sector_combinations = np.prod([space.num_sectors for space in self.spaces])
        
        # [i, j] :: position of the part of j in spaces[i] within its private basis
        idcs = unstridify(np.arange(self.dim), dim_strides).T
        
        # [i, j] :: sector of the part of j in spaces[i] is spaces[i].sectors[sector_idcs[i, j]]
        #           sector_idcs[i, j] = bisect.bisect(spaces[i].slices[:, 0], idcs[i, j]) - 1
        sector_idcs = np.array(
            [[bisect.bisect(sp.slices[:, 0], idx) - 1 for idx in idx_col]
             for sp, idx_col in zip(self.spaces, idcs)]
        )  # OPTIMIZE can bisect.bisect be broadcast somehow? is there a numpy alternative?
        
        # [i, j] :: the part of j in spaces[i] is the degeneracy_idcs[i, j]-th state within that sector
        #           degeneracy_idcs[i, j] = idcs[i, j] - spaces[i].slices[sector_idcs[i, j], 0]
        degeneracy_idcs = idcs - np.stack(
            [sp.slices[si_col, 0] for sp, si_col in zip(self.spaces, sector_idcs)]
        )
        
        # [i, j] :: strides for combining degeneracy indices.
        #           degeneracy_strides[:, j] = make_stride([... mults with sector_idcs[:, j]])
        degeneracy_strides = np.array(
            [make_stride([sp.multiplicities[si] for sp, si in zip(self.spaces, si_row)])
             for si_row in sector_idcs.T]
        ).T  # OPTIMIZE make make_stride broadcast?
        
        # [j] :: position of j in the unsorted list of fusion outcomes
        fusion_outcome = np.sum(sector_idcs * sector_strides[:, None], axis=0)

        # [i, s] :: sector combination s has spaces[i].sectors[all_sector_idcs[i, s]]
        all_sector_idcs = unstridify(np.arange(num_sector_combinations), sector_strides).T

        # [i, s] :: all_mults[i, s] = spaces[i].multiplicities[all_sector_idcs[i, s]]
        all_mults = np.array([sp.multiplicities[comb] for sp, comb in zip(self.spaces, all_sector_idcs)])

        # [s] : total multiplicity of the fusion channel
        fusion_outcome_multiplicities = np.prod(all_mults, axis=0)

        # [s] : !!shape == (L_s + 1,)!!  ; starts ([s]) and stops ([s + 1]) of fusion channels in the sorted list
        fusion_outcome_slices = np.concatenate(
            [[0], np.cumsum(fusion_outcome_multiplicities[self._fusion_outcomes_sort])]
        )

        # [j] : position of fusion channel after sorting
        sorted_pos = self._fusion_outcomes_inverse_sort[fusion_outcome]

        # [j] :: contribution from the sector, i.e. start of all the js of the same fusion channel
        sector_part = fusion_outcome_slices[sorted_pos]
        
        # [j] :: contribution from the multiplicities, i.e. position with all js of the same fusion channel
        degeneracy_part = np.sum(degeneracy_idcs * degeneracy_strides, axis=0)

        return inverse_permutation(sector_part + degeneracy_part)

    def fuse_states(self, states: list[Block], backend: Backend) -> Block:
        """TODO write docs"""
        # if abelian first kron then use get_basis_transformation_perm()
        # other wise contract get_basis_transformation() with the states
        raise NotImplementedError  # TODO

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable,
                         backend: Backend = None) -> ProductSpace:
        spaces = [s.change_symmetry(symmetry=symmetry, sector_map=sector_map) for s in self.spaces]
        return ProductSpace(spaces, backend=backend, _is_dual=self.is_dual)

    def drop_symmetry(self, which: int | list[int] = None, remaining_symmetry: Symmetry = None):
        # TODO do we need the backend arg of ProductSpace.__init__?
        if len(self.spaces) == 0:
            return ProductSpace([], _is_dual=self.is_dual)
        first = self.spaces[0].drop_symmetry(which, remaining_symmetry)
        if remaining_symmetry is None:
            remaining_symmetry = first.symmetry
        rest = [space.drop_symmetry(which, remaining_symmetry) for space in self.spaces[1:]]
        return ProductSpace([first] + rest, _is_dual=self.is_dual)

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

    def is_equal_or_dual(self, other: ProductSpace) -> bool:
        """If another ProductSpace is equal to *or* dual of `self`."""
        if not isinstance(other, ProductSpace):
            return False
        if len(other.spaces) != len(self.spaces):
            return False
        return all(s1.is_equal_or_dual(s2) for s1, s2 in zip(self.spaces, other.spaces))

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

    def iter_uncoupled(self) -> Iterator[tuple[Sector]]:
        """Iterate over all combinations of sectors"""
        return it.product(*(s.sectors for s in self.spaces))
        

def _fuse_spaces(symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool
                 ) -> tuple[SectorArray, ndarray, ndarray, dict]:
    """This function is called as part of ProductSpace.__init__.
    
    It determines the sectors and multiplicities of the ProductSpace.
    There is also a version of this function in the backends, i.e.
    :meth:`~tenpy.linalg.backends.abstract_backend.Backend._fuse_spaces:, which may
    customize this behavior and in particular may return metadata, i.e. attributes to be added to
    the ProductSpace.
    This default implementation returns empty metadata ``{}``.

    Returns
    -------
    sectors : 2D array of int
        The :attr:`VectorSpace._non_dual_sectors`.
    multiplicities : 1D array of int
        the :attr:`VectorSpace.multiplicities`.
    fusion_outcomes_sort
        the :attr:`ProductSpace._fusion_outcomes_sort`.
    metadata : dict
        A dictionary with string keys and arbitrary values.
        These will be added as attributes of the ProductSpace
    """
    if isinstance(symmetry, NoSymmetry):
        sectors = symmetry.trivial_sector[None, :]
        multiplicities = [np.prod([sp.dim for sp in spaces])]
        return sectors, multiplicities, np.arange(1), {}
    
    if _is_dual:
        spaces = [s.dual for s in spaces] # directly fuse sectors of dual spaces.
        # This yields overall dual `sectors` to return, which we directly save in
        # self._non_dual_sectors, such that `self.sectors` (which takes a dual!) yields correct sectors
        # Overall, this ensures consistent sorting/order of sectors between dual ProductSpace!

    if symmetry.is_abelian:
        # copying parts from AbelianBackend._fuse_spaces here...
        grid = np.indices(tuple(space.num_sectors for space in spaces), np.intp)
        grid = grid.T.reshape(-1, len(spaces))
        sectors = symmetry.multiple_fusion_broadcast(
            *(sp.sectors[gr] for sp, gr in zip(spaces, grid.T))
        )
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(spaces, grid.T)],
                                  axis=0)
        sectors, multiplicities, fusion_outcomes_sort = _unique_sorted_sectors(sectors, multiplicities)
        return sectors, multiplicities, fusion_outcomes_sort, {}

    # define recursively. base cases:
    if len(spaces) == 0:
        return symmetry.empty_sector_array, [], None, {}

    if len(spaces) == 1:
        sectors = spaces[0].sectors
        mults = spaces[0].multiplicities
        order = np.lexsort(sectors.T)
        return sectors[order, :], mults[order], None, {}

    # _is_dual is already accounted for.
    sectors_1, mults_1, _, _ = _fuse_spaces(symmetry, spaces[:-1], _is_dual=False)
    sectors_2 = spaces[-1].sectors
    mults_2 = spaces[-1].multiplicities

    sector_contributions = []
    mult_contributions = []
    for s2, m2 in zip(sectors_2, mults_2):
        for s1, m1 in zip(sectors_1, mults_1):
            sects = symmetry.fusion_outcomes(s1, s2)
            sector_contributions.append(sects)
            if symmetry.fusion_style is FusionStyle.general:
                mult_contributions.append(
                    m1 * m2 * np.array([symmetry._n_symbol(s1, s2, c) for c in sects], dtype=int)
                )
            else:
                mult_contributions.append(m1 * m2 * np.ones((len(sects),), dtype=int))
                
    sectors, multiplicities, _ = _unique_sorted_sectors(
        np.concatenate(sector_contributions, axis=0),
        np.concatenate(mult_contributions, axis=0)
    )
    return sectors, multiplicities, None, {}


def _unique_sorted_sectors(unsorted_sectors: SectorArray, unsorted_multiplicities: np.ndarray):
    """Helper function for _fuse_spaces

    Given unsorted sectors which may contain duplicates,
    return a sorted list of unique sectors and corresponding *aggregate* multiplicities

    Returns
    -------
    sectors
        The unique entries of the `unsorted_sectors`, sorted according to ``np.lexsort( .T)``.
    multiplicities
        The corresponding aggregate multiplicities, i.e. the sum of all entries in
        `unsorted_multiplicities` which correspond to the given sector
    perm
        The permutation that sorts the input, i.e. ``np.lexsort(unsorted_sectors.T)``.
    """
    perm = np.lexsort(unsorted_sectors.T)
    sectors = unsorted_sectors[perm]
    multiplicities = unsorted_multiplicities[perm]
    slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
    diffs = find_row_differences(sectors, include_len=True)
    slices = slices[diffs]
    multiplicities = slices[1:] - slices[:-1]
    sectors = sectors[diffs[:-1]]
    return sectors, multiplicities, perm
