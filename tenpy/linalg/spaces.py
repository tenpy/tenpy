# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import ndarray
import bisect
import itertools as it
from typing import TYPE_CHECKING, Sequence, Iterator

from .dummy_config import printoptions
from .symmetries import (Sector, SectorArray, Symmetry, ProductSymmetry, no_symmetry, FusionStyle,
                         SymmetryError)
from ..tools.misc import (inverse_permutation, rank_data, to_iterable, UNSPECIFIED, make_stride,
                          find_row_differences, unstridify, iter_common_sorted_arrays)
from ..tools.string import format_like_list

if TYPE_CHECKING:
    from .backends.abstract_backend import TensorBackend, Block

__all__ = ['Space', 'ElementarySpace', 'ProductSpace']


class Space(metaclass=ABCMeta):
    """A space, which decomposes into sectors of a given symmetry.

    This is a base classes, the concrete subclasses are :class:`ElementarySpace`
    and :class:`ProductSpace`.

    Attributes
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sectors : 2D numpy array of int
        The sectors that compose this space. A 2D array of integers with axes [s, q] where s goes
        over different sectors and q over the (one or more) numbers needed to label a sector.
        The sectors (to be precise, the rows ``sectors[i, :]``) are unique and sorted, such that
        ``np.lexsort(sectors.T)`` is trivial. We use :attr:`multiplicities` for duplicates.
    multiplicities : 1D numpy array of int
        How often each of the :attr:`sectors` appears. A 1D array of positive integers with axis [s].
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
    is_bra_space : bool
        Whether this is a bra space. For :class:`ElementarySpace`, this is the same is
        :attr:`ElementarySpace.is_dual`. For :class:`ProductSpace`, it is always ``False``.
    """
    
    def __init__(self, symmetry: Symmetry, sectors: SectorArray | Sequence[Sequence[int]],
                 multiplicities: Sequence[int] | None, is_bra_space: bool):
        self.symmetry = symmetry
        self.sectors = sectors = np.asarray(sectors, dtype=int)
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
            self.dim = np.sum(sector_dims * multiplicities).item()
        else:
            self.sector_dims = None
            self.sector_qdims = sector_qdims = symmetry.batch_qdim(sectors)
            self.slices = None
            self.dim = np.sum(sector_qdims * multiplicities).item()
        self.is_bra_space = is_bra_space

    def test_sanity(self):
        assert self.dim >= 0
        # sectors
        assert self.sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len), 'wrong sectors.shape'
        assert all(self.symmetry.is_valid_sector(s) for s in self.sectors), 'invalid sectors'
        assert len(np.unique(self.sectors, axis=0)) == self.num_sectors, 'duplicate sectors'
        assert np.all(np.lexsort(self.sectors.T) == np.arange(self.num_sectors)), 'wrong sector order'
        # multiplicities
        assert np.all(self.multiplicities > 0)
        assert self.multiplicities.shape == (self.num_sectors,)
        if self.symmetry.can_be_dropped:
            # slices
            assert self.slices.shape == (self.num_sectors, 2)
            slice_diffs = self.slices[:, 1] - self.slices[:, 0]
            assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self.sectors))
            expect_diffs = self.sector_dims * self.multiplicities
            assert np.all(slice_diffs == expect_diffs)
            # slices should be consecutive
            if self.num_sectors > 0:
                assert self.slices[0, 0] == 0
                assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
                assert self.slices[-1, 1] == self.dim

    # ABSTRACT

    @property
    def dual(self):
        return self._dual_space(return_perm=False)

    @property
    @abstractmethod
    def is_trivial(self):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def as_ElementarySpace(self, is_dual: bool = None) -> ElementarySpace:
        ...

    @abstractmethod
    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, backend: TensorBackend = None,
                        injective: bool = False) -> ElementarySpace:
        """Change the symmetry by specifying how the sectors change.

        Parameters
        ----------
        symmetry : :class:`~tenpy.linalg.groups.Symmetry`
            The symmetry of the new space
        sector_map : function (SectorArray,) -> (SectorArray,)
            A map of sectors (2D int arrays), such that ``new_sectors = sector_map(old_sectors)``.
            The map is assumed to cooperate with duality, i.e. we assume without checking that
            ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
            ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
            TODO do we need to assume more, i.e. compatibility with fusion?
        backend : :class: `~tenpy.linalg.backends.abstract_backend.Backend`
            This parameter is ignored. We only include it to have matching signatures
            with :meth:`ProductSpace.change_symmetry`.
        injective: bool
            If ``True``, the `sector_map` is assumed to be injective, i.e. produce a list of
            unique outputs, if the inputs are unique.

        Returns
        -------
        A space with the new symmetry. The order of the basis is preserved, but every
        basis element lives in a new sector, according to `sector_map`.
        """
        ...

    @abstractmethod
    def drop_symmetry(self, which: int | list[int] = None):
        """Drop some or all symmetries.

        Parameters
        ----------
        which : None | (list of) int
            If ``None`` (default) the entire symmetry is dropped and the result has ``no_symmetry``.
            An integer or list of integers assume that ``self.symmetry`` is a ``ProductSymmetry``
            and indicates which of its factors to drop.
        """
        ...

    @abstractmethod
    def _dual_space(self, return_perm: bool = False) -> Space | tuple[Space, np.ndarray]:
        """Compute the dual space. Optionally return the induced permutation of sectors.

        Returns
        -------
        dual: Space
            The dual space
        perm: 1D ndarray, optional
            The permutation such that ``dual.sectors[n]`` is the dual of ``self.sectors[perm[n]]``
        """
        ...

    @abstractmethod
    def _repr(self, show_symmetry: bool):
        ...

    # CONCRETE IMPLEMENTATIONS

    def __repr__(self):
        res = self._repr(show_symmetry=True)
        if res is None:
            return f'<{self.__class__.__name__}>'
        return res

    @property
    def num_parameters(self) -> int:
        """The number of linearly independent *symmetric* tensors in this space."""
        return self.sector_multiplicity(self.symmetry.trivial_sector)

    def largest_common_subspace(self, other: Space, is_dual: bool = False) -> ElementarySpace:
        """The largest common subspace."""
        assert self.symmetry == other.symmetry
        sectors = []
        mults = []
        for i, j in iter_common_sorted_arrays(self.sectors, other.sectors):
            sectors.append(self.sectors[i])
            mults.append(min(self.multiplicities[i], other.multiplicities[j]))
        return ElementarySpace(self.symmetry, sectors, mults, is_dual=is_dual)

    def sectors_where(self, sector: Sector) -> int | None:
        # TODO / OPTIMIZE : use that sectors are sorted to speed up the lookup
        where = np.where(np.all(self.sectors == sector, axis=1))[0]
        if len(where) == 0:
            return None
        if len(where) == 1:
            return where[0]
        raise RuntimeError  # sectors should have unique entries, so this should not happen

    def sector_multiplicity(self, sector: Sector) -> int:
        idx = self.sectors_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]


class ElementarySpace(Space):
    r"""A space which is graded by a symmetry, but has no further structure.

    We distinguish ket spaces :math:`V_k := a_1 \oplus a_2 \oplus \dots \plus a_N` with
    ``is_dual=False`` and bra spaces :math:`V_b := [b_1 \oplus b_2 \oplus \dots \plus b_N]^*`
    with ``is_dual=True``. The bra space also decomposes into sectors, as
    :math:`V_b \cong \bar{b}_1 \oplus \bar{b}_2 \oplus \dots \plus \bar{b}_N`,
    where :math:`\bar{b}` is the :meth:`Symmetry.dual_sector` of :math:`b`.
    The :attr:`sectors` of a space then describe the :math:`\{a_n\}` for the ket space
    :math:`V_k` and the :math:`\{\bar{b}_n\}` for the bra space :math:`V_b`.

    If the symmetry :attr:`Symmetry.can_be_dropped`, there is a notion of a basis for the
    spaces. We demand the basis to be compatible with the symmetry, i.e. each basis vector
    needs to lie in one of the sectors of the symmetry. The *internal* basis order that results
    from demanding that the sectors are contiguous and sorted may, however, not be the desired
    basis order, e.g. for matrix representations. For example, the standard basis of a spin-1
    degree of freedom with ``'Sz_parity'`` conservation has sectors ``[[1], [0], [1]]`` and is
    neither sorted by sector nor contiguous. We allow these different *public* basis orders
    and store the relevant perturbation as :attr:`basis_perm`.
    See also :attr:`sectors_of_basis` and :meth:`from_basis`.

    Parameters
    ----------
    symmetry, sectors, multiplicities, is_dual, basis_perm
        Like attributes of the same name, and nested lists are allowed in place of arrays.

    Attributes
    ----------
    is_dual: bool
        If this is a bra or a ket space, such that ``ElementarySpace(sym, sec, is_dual=True)``
        is equivalent to ``ElementarySpace(sym, sec, is_dual=False).flip_is_dual()`` and equal to
        ``ElementarySpace(sym, dual_sectors(sec), is_dual=False).dual``.
    """
    
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 is_dual: bool = False, basis_perm: ndarray | None = None):
        Space.__init__(self, symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                       is_bra_space=is_dual)
        self.is_dual = is_dual
        if basis_perm is None:
            self._basis_perm = self._inverse_basis_perm = None
        else:
            if not symmetry.can_be_dropped:
                msg = f'basis_perm is meaningless for {symmetry}.'
                raise SymmetryError(msg)
            self._basis_perm = basis_perm = np.asarray(basis_perm, dtype=int)
            self._inverse_basis_perm = inverse_permutation(basis_perm)

    def test_sanity(self):
        assert self.is_bra_space == self.is_dual
        if not self.symmetry.can_be_dropped:
            assert self._basis_perm is None
        if self._basis_perm is None:
            assert self._inverse_basis_perm is None
        else:
            assert self._inverse_basis_perm is not None
            assert self._basis_perm.shape == self._inverse_basis_perm.shape == (self.dim,)
            assert len(np.unique(self._basis_perm)) == self.dim  # is a permutation
            assert len(np.unique(self._inverse_basis_perm)) == self.dim  # is a permutation
            assert np.all(self._basis_perm[self._inverse_basis_perm] == np.arange(self.dim))
        super().test_sanity()

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors_of_basis: Sequence[Sequence[int]],
                   is_dual: bool = False) -> ElementarySpace:
        """Create an ElementarySpace by specifying the sector of every basis element.

        .. note ::
            Unlike :meth:`from_sectors`, this method expects the same sector to be listed
            multiple times, if the sector is multi-dimensional. The Hilbert Space of a spin-one-half
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_sectors(su2, [[spin_half]])``. In the former case we need to
            list the same sector both for the spin up and spin down state.

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
        is_dual : bool
            If the space is a bra space of a ket space. Either way, it decomposes into the given
            sectors.

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
                   is_dual=is_dual, basis_perm=basis_perm)

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions: list[ElementarySpace]
                                    ) -> ElementarySpace:
        """Create an ElementarySpace with multiple independent symmetries.

        Parameters
        ----------
        independent_descriptions : list of :class:`ElementarySpace`
            Each entry describes the resulting :class:`ElementarySpace` in terms of *one* of
            the independent symmetries. Spaces with a :class:`NoSymmetry` are ignored.
        """
        # OPTIMIZE this can be implemented better. if many consecutive basis elements have the same
        #          resulting sector, we can skip over all of them.
        assert len(independent_descriptions) > 0
        dim = independent_descriptions[0].dim
        assert all(s.dim == dim for s in independent_descriptions)
        # ignore those with no_symmetry
        independent_descriptions = [s for s in independent_descriptions if s.symmetry != no_symmetry]
        if len(independent_descriptions) == 0:
            # all descriptions had no_symmetry
            return cls.from_trivial_sector(dim=dim)
        symmetry = ProductSymmetry.from_nested_factors(
            [s.symmetry for s in independent_descriptions]
        )
        if not symmetry.can_be_dropped:
            msg = f'from_independent_symmetries is not supported for {symmetry}.'
            # TODO is there a way to define this?
            #      the straight-forward picture works only if we have a vector space and can identify states.
            raise SymmetryError(msg)
        sectors_of_basis = np.concatenate([s.sectors_of_basis for s in independent_descriptions],
                                          axis=1)
        return cls.from_basis(symmetry, sectors_of_basis)

    @classmethod
    def from_null_space(cls, symmetry: Symmetry, is_dual: bool = False) -> ElementarySpace:
        """The zero-dimensional space, i.e. the span of the empty set."""
        return cls(symmetry=symmetry, sectors=symmetry.empty_sector_array,
                   multiplicities=np.zeros(0, int), is_dual=is_dual)

    @classmethod
    def from_sectors(cls, symmetry: Symmetry, sectors: SectorArray,
                     multiplicities: Sequence[int] = None, is_dual: bool = False,
                     basis_perm: ndarray = None, unique_sectors: bool = False,
                     return_sorting_perm: bool = False
                     ) -> ElementarySpace | tuple[ElementarySpace, ndarray]:
        """Similar to the constructor, but with fewer requirements.

        .. note ::
            Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
            only once to mean its entire multiplet of basis states. The Hilbert Space of a spin-1/2
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_sectors(su2, [spin_half])``. In the former case we need to
            list the same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        sectors: 2D array_like of int
            The sectors of the symmetry that compose this space.
            Can be in any order and may contain duplicates (see `unique_sectors`).
        multiplicities: 1D array_like of int, optional
            How often each of the `sectors` appears. A 1D array of positive integers with axis [s].
            ``sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
            If not given, a multiplicity ``1`` is assumed for all `sectors`.
        is_dual: bool
            If the result is a bra- or a ket space.
        basis_perm: ndarray, optional
            The permutation from the desired public basis to the basis described by `sectors`
            and `multiplicities`.
        unique_sectors: bool
            If ``True``, the `sectors` are assumed to be duplicate-free.
        return_sorting_perm: bool
            If ``True``, the permutation ``np.lexsort(sectors.T)`` is returned too.

        Returns
        -------
        space: ElementarySpace
        sector_sort: 1D array, optional
            Only returned ``if return_sorting_perm``. The permutation that sorts the `sectors`.
        """
        sectors = np.asarray(sectors, dtype=int)
        assert sectors.ndim == 2 and sectors.shape[1] == symmetry.sector_ind_len
        if multiplicities is None:
            multiplicities = np.ones((len(sectors),), dtype=int)
        else:
            multiplicities = np.asarray(multiplicities, dtype=int)
            assert multiplicities.shape == ((len(sectors),))
        
        # sort sectors
        if symmetry.can_be_dropped:
            num_states = symmetry.batch_sector_dim(sectors) * multiplicities
            basis_slices = np.concatenate([[0], np.cumsum(num_states)], axis=0)
            sectors, multiplicities, sort = _sort_sectors(sectors, multiplicities)
            if len(sectors) == 0:
                basis_perm = np.zeros(0, int)
            else:
                if basis_perm is None:
                    basis_perm = np.arange(np.sum(num_states))
                basis_perm = np.concatenate([basis_perm[basis_slices[i]: basis_slices[i + 1]]
                                            for i in sort])
        else:
            sectors, multiplicities, sort = _sort_sectors(sectors, multiplicities)
            assert basis_perm is None
        # combine duplicate sectors (does not affect basis_perm)
        if not unique_sectors:
            mult_slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
            diffs = find_row_differences(sectors, include_len=True)
            multiplicities = mult_slices[diffs[1:]] - mult_slices[diffs[:-1]]
            sectors = sectors[diffs[:-1]]  # [:-1] to exclude len
        res = cls(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                  is_dual=is_dual, basis_perm=basis_perm)
        if return_sorting_perm:
            return res, sort
        return res

    @classmethod
    def from_trivial_sector(cls, dim: int, symmetry: Symmetry = no_symmetry, is_dual: bool = False,
                            basis_perm: ndarray = None) -> ElementarySpace:
        """Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).

        Parameters
        ----------
        dim : int
            The dimension of the space.
        symmetry : :class:`~tenpy.linalg.groups.Symmetry`
            The symmetry of the space. By default, we use `no_symmetry`.
        is_real, is_dual : bool
            If the space should be real / dual.
        """
        if dim == 0:
            return cls.from_null_space(symmetry=symmetry, is_dual=is_dual)
        return cls(symmetry=symmetry, sectors=symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_dual=is_dual, basis_perm=basis_perm)

    @property
    def basis_perm(self) -> ndarray:
        """Permutation that translates between public and internal basis order.

        For the inverse permutation, see :attr:`inverse_basis_perm`.

        The tensor manipulations of ``tenpy.linalg`` benefit from choosing a canonical order for the
        basis of vector spaces. This attribute translates between the "public" order of the basis,
        in which e.g. the inputs to :meth:`from_dense_block` are interpreted to this internal order,
        such that ``public_basis[basis_perm] == internal_basis``.
        The internal order is such that the basis vectors are grouped and sorted by sector.
        We can translate indices as ``public_idx == basis_perm[internal_idx]``.
        Only available if ``symmetry.can_be_dropped``, as otherwise there is no well-defined
        notion of a basis.
        
        ``_basis_perm`` is the internal version which may be ``None`` if the permutation is trivial.
        """
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._basis_perm is None:
            return np.arange(self.dim)
        return self._basis_perm

    @property
    def inverse_basis_perm(self) -> ndarray:
        """Inverse permutation of :attr:`basis_perm`."""
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is None:
            return np.arange(self.dim)
        return self._inverse_basis_perm

    @property
    def is_trivial(self) -> bool:
        """Whether self is the trivial space.

        The trivial space is a one-dimensional space which consists only of the trivial sector,
        appearing exactly once. In a mathematical sense, the trivial sector _is_ the trivial space.
        We count both the bra space and the ket space with these attributes as trivial.

        TODO name is maybe not ideal... the ElementarySpace.from_null_space could also be called "trivial"
             this space is the unit of fusion
        """
        if self.num_sectors != 1:
            return False
        if self.multiplicities[0] != 1:
            return False
        if not np.all(self.sectors[0] == self.symmetry.trivial_sector):
            return False
        return True

    @property
    def sectors_of_basis(self):
        """The sector for each basis vector, like the input of :meth:`from_basis`."""
        if not self.symmetry.can_be_dropped:
            msg = f'sectors_of_basis is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        # build in internal basis, then permute
        res = np.zeros((self.dim, self.symmetry.sector_ind_len), dtype=int)
        for sect, slc in zip(self.sectors, self.slices):
            res[slice(*slc), :] = sect[None, :]
        if self._inverse_basis_perm is not None:
            res = res[self._inverse_basis_perm]
        return res

    def _repr(self, show_symmetry: bool):
        # used by Space.__repr__
        indent = printoptions.indent * ' '
        # 1) Try showing all data
        if 3 * self.sectors.size < printoptions.linewidth:
            # otherwise there is no chance to print all sectors in one line anyway
            if self._basis_perm is None:
                basis_perm = 'None'
            else:
                basis_perm = format_like_list(self._basis_perm)
            elements = [f'ElementarySpace(']
            if show_symmetry:
                elements.append(f'{self.symmetry!r}')
            elements.extend([
                f'sectors={format_like_list(self.symmetry.sector_str(a) for a in self.sectors)}',
                f'multiplicities={format_like_list(self.multiplicities)}',
                f'basis_perm={basis_perm}',
                f'is_dual={self.is_dual}',
                ')'
            ])
            one_line = ', '.join(elements)
            if len(one_line) <= printoptions.linewidth:
                return one_line
            if all(len(l) <= printoptions.linewidth for l in elements) and len(elements) <= printoptions.maxlines_spaces:
                elements[1:-1] = [f'{indent}{line},' for line in elements[1:-1]]
                return '\n'.join(elements)
        # 2) Try showing summarized data
        elements = [f'<ElementarySpace:']
        if show_symmetry:
            elements.append(f'{self.symmetry!s}')
        elements.extend([
            f'{self.num_sectors} sectors',
            f'basis_perm={"None" if self._basis_perm is None else "[...]"}',
            f'is_dual={self.is_dual}',
            '>',
        ])
        one_line = ' '.join(elements)
        if len(one_line) < printoptions.linewidth:
            return one_line
        if all(len(l) <= printoptions.linewidth for l in elements) and len(elements) <= printoptions.maxlines_spaces:
            elements[1:-1] = [f'{indent}{line},' for line in elements[1:-1]]
            return '\n'.join(elements)
        # 3) Try showing only symmetry
        if show_symmetry:
            elements[2:-1] = []
            one_line = ' '.join(elements)
            if len(one_line) < printoptions.linewidth:
                return one_line
            if all(len(l) <= printoptions.linewidth for l in elements) and len(elements) <= printoptions.maxlines_spaces:
                elements[1:-1] = [f'{indent}{line},' for line in elements[1:-1]]
                return '\n'.join(elements)
        # 4) Show no data at all
        return None

    def __eq__(self, other):
        if not isinstance(other, ElementarySpace):
            return NotImplemented
        if self.is_dual != other.is_dual:
            return False
        if self.symmetry != other.symmetry:
            return False
        if self.num_sectors != other.num_sectors:  # check this first to safely compare later
            return False
        if not np.all(self.multiplicities == other.multiplicities):
            return False
        if not np.all(self.sectors == other.sectors):
            return False
        if (self._basis_perm is not None) or (other._basis_perm is not None):
            # otherwise both are trivial and this match
            if not np.all(self.basis_perm == other.basis_perm):
                return False
        return True

    def as_ElementarySpace(self, is_dual: bool = None) -> ElementarySpace:
        if (is_dual is None) or (is_dual == self.is_dual):
            return self
        return self.with_opposite_duality()

    def as_ket_space(self):
        """The ket space (``is_dual=False``) isomorphic or equal to self."""
        if not self.is_dual:
            return self
        return self.with_opposite_duality()

    def as_bra_space(self):
        """The bra space (``is_dual=False``) isomorphic or equal to self."""
        if self.is_dual:
            return self
        return self.with_opposite_duality()

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, backend: TensorBackend = None,
                        injective: bool = False
                        ) -> ElementarySpace:
        # backend is just there to have the same signature as ProductSpace.change_symmetry
        # TODO / OPTIMIZE can avoid some computation if the map is injective.
        #                 then we just need to sort the new sectors, no need to combine
        return ElementarySpace.from_sectors(
            symmetry=symmetry, sectors=sector_map(self.sectors), multiplicities=self.multiplicities,
            is_dual=self.is_dual, basis_perm=self._basis_perm, unique_sectors=injective
        )

    def direct_sum(self, *others: ElementarySpace) -> ElementarySpace:
        """Form the direct sum (i.e. stacking).

        The basis of the new space results from concatenating the individual bases.
        
        Spaces must have the same symmetry and is_dual.
        The result is a space with the same symmetry and is_dual, whose sectors are those
        that appear in any of the spaces and multiplicities are the sum of the multiplicities
        in each of the spaces.
        """
        if not others:
            return self
        assert all(o.symmetry == self.symmetry for o in others)
        assert all(o.is_dual == self.is_dual for o in others)
        if self.symmetry.can_be_dropped:
            offsets = np.cumsum([self.dim, *(o.dim for o in others)])
            basis_perm = np.concatenate(
                [self.basis_perm] + [o.basis_perm + n for o, n in zip(others, offsets)]
            )
        else:
            basis_perm = None
        return ElementarySpace.from_sectors(
            symmetry=self.symmetry,
            sectors=np.concatenate([self.sectors, *(o.sectors for o in others)]),
            multiplicities=np.concatenate([self.multiplicities, *(o.multiplicities for o in others)]),
            is_dual=self.is_dual, basis_perm=basis_perm
        )

    def drop_symmetry(self, which: int | list[int] = None):
        which, remaining_symmetry = _parse_inputs_drop_symmetry(which, self.symmetry)
        if which is None:
            return ElementarySpace.from_trivial_sector(
                dim=self.dim, symmetry=remaining_symmetry, is_dual=self.is_dual,
                basis_perm=self._basis_perm
            )
        mask = np.ones((self.symmetry.sector_ind_len,), dtype=bool)
        for i in which:
            start, stop = self.symmetry.sector_slices[i:i + 2]
            mask[start:stop] = False
        return self.change_symmetry(symmetry=remaining_symmetry,
                                    sector_map=lambda sectors: sectors[:, mask])

    def _dual_space(self, return_perm: bool = False
                    ) -> ElementarySpace | tuple[ElementarySpace, np.ndarray]:
        return ElementarySpace.from_sectors(
            symmetry=self.symmetry, sectors=self.symmetry.dual_sectors(self.sectors),
            multiplicities=self.multiplicities, is_dual=not self.is_dual,
            basis_perm=self._basis_perm, unique_sectors=True, return_sorting_perm=return_perm
        )

    def is_subspace_of(self, other: ElementarySpace) -> bool:
        """Whether self is a subspace of other.
        
        Per convention, self is never a subspace of other, if the :attr:`is_dual` or the
        :attr:`symmetry` are different.
        The :attr:`basis_perm`s are not considered.
        """
        if self.is_dual != other.is_dual:
            return False
        if not self.symmetry.is_same_symmetry(other.symmetry):
            return False
        if self.num_sectors == 0:
            return True
        # sectors are sorted, so we can just iterate over both of them
        n_self = 0
        for other_sector, other_mult in zip(other.sectors, other.multiplicities):
            if np.all(self.sectors[n_self] == other_sector):
                if self.multiplicities[n_self] > other_mult:
                    return False
                n_self += 1
            if n_self == self.num_sectors:
                # have checked all sectors of self
                return True
        # reaching this line means self has sectors which other does not have
        return False

    def parse_index(self, idx: int) -> tuple[int, int]:
        """Utility function to translate an index.

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
        sector_idx, _ = self.parse_index(idx)
        return self.sectors[sector_idx]

    def take_slice(self, blockmask: Block) -> ElementarySpace:
        """Take a "slice" of the leg, keeping only some of the basis states.

        Parameters
        ----------
        blockmask : 1D array-like of bool
            For every basis state of self, in the public basis order,
            if it should be kept (``True``) or discarded (``False``).
        """
        if not self.symmetry.can_be_dropped:
            msg = f'take_slice is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        blockmask = np.asarray(blockmask, dtype=bool)
        if self._basis_perm is not None:
            blockmask = blockmask[self._basis_perm]
        #
        sectors = []
        mults = []
        for a, d_a, slc in zip(self.sectors, self.sector_dims, self.slices):
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
            sectors = self.symmetry.empty_sector_array
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
        # note blockmask is in the private basis order.
        basis_perm = rank_data(self.basis_perm[blockmask])
        return ElementarySpace(symmetry=self.symmetry, sectors=sectors, multiplicities=mults,
                               is_dual=self.is_dual, basis_perm=basis_perm)

    def with_opposite_duality(self):
        """A space isomorphic to self with opposite ``is_dual`` attribute."""
        return ElementarySpace(symmetry=self.symmetry, sectors=self.sectors,
                               multiplicities=self.multiplicities, is_dual=not self.is_dual,
                               basis_perm=self._basis_perm)

    def with_is_dual(self, is_dual: bool) -> ElementarySpace:
        """A space isomorphic to self with given ``is_dual`` attribute."""
        if is_dual == self.is_dual:
            return self  # TODO copy?
        return self.with_opposite_duality()


class ProductSpace(Space):
    r"""The tensor product of multiple spaces, which is itself a space.

    Unlike for :class:`ElementarySpace`, we do not distinguish between bra and ket spaces.
    This is indeed what makes the :class:`ElementarySpace` "elementary".

    The :class:`ProductSpace` introduces a basis transformation the *uncoupled* ("public") basis
    is given by products of the individual (thus "uncoupled") basis elements.
    E.g. for a product space :math:`P = V \otimes W \otimes \dots \otimes Z`, this basis consists
    of elements of the form :math:`v_{i_1} \otimes w_{i_2} \otimes \dots \otimes z_{i_n}`.
    The order is given in C-style (varying the last index, here :math:`i_n` the fastest) combination
    of the *public* basis order of the factor :attr:`spaces`.
    The *coupled* basis is given by the fusion outcomes, sorted and grouped by sector.
    See :meth:`get_basis_transformation` for the explicit transformation.
    Thus, a product space does not have a ``basis_perm`` attribute, unlike an
    :class:`ElementarySpace`.

    Backends may add :attr:`metadata` to ProductSpaces.

    Parameters
    ----------
    spaces, symmetry
        Like the attributes of the same name
    backend: TensorBackend | None
        The backend, used in :meth:`_fuse_spaces`, to add backend-specific :attr:`metadata`.
    _sectors, _multiplicities, _metadata
        Can optionally be passed to avoid recomputation.

    Attributes
    ----------
    metadata: dict
        Backend-specific additional data, added by :meth:`TensorBackend._fuse_spaces`.
        Metadata is considered optional and can be computed on-demand via :meth:`get_metadata`.
        A common entry is accessible via the property :attr:`fusion_outcomes_sort`.
    """

    def __init__(self, spaces: list[Space], symmetry: Symmetry = None, backend: TensorBackend = None,
                 _sectors: SectorArray = UNSPECIFIED, _multiplicities: ndarray = UNSPECIFIED,
                 _metadata: dict = UNSPECIFIED):
        self.spaces = spaces[:]
        self.num_spaces = len(spaces)
        if symmetry is None:
            if len(spaces) == 0:
                raise ValueError('If spaces is empty, the symmetry arg is required.')
            symmetry = spaces[0].symmetry
        if not all(sp.symmetry == symmetry for sp in spaces):
            raise SymmetryError('Incompatible symmetries.')
        self.symmetry = symmetry
        if (_sectors is UNSPECIFIED) or (_multiplicities is UNSPECIFIED):
            _sectors, _multiplicities, _metadata = _fuse_spaces(
                symmetry=symmetry, spaces=spaces, backend=backend
            )
        Space.__init__(self, symmetry=symmetry, sectors=_sectors,
                       multiplicities=_multiplicities, is_bra_space=False)
        if _metadata is UNSPECIFIED:
            if backend is None:
                _metadata = {}
            else:
                _metadata = backend.get_leg_metadata(self)
        self.metadata = _metadata
        self._basis_perm = None
        self._inverse_basis_perm = None

    def test_sanity(self):
        assert isinstance(self.metadata, dict)
        assert len(self.spaces) == self.num_spaces
        for sp in self.spaces:
            sp.test_sanity()
        Space.test_sanity(self)

    @classmethod
    def from_partial_products(cls, *factors: ProductSpace, backend: TensorBackend | None = None
                              ) -> ProductSpace:
        """Given multiple product spaces, create the flat product of all their :attr:`spaces`.

        This is equivalent to ``ProductSpace([p_space.spaces for p_space in factors])``,
        but avoids some of the computation of sectors.
        """
        isomorphic = ProductSpace(factors, backend=backend)
        return ProductSpace(
            spaces=[sp for pr in factors for sp in pr.spaces], backend=backend,
            _sectors=isomorphic.sectors, _multiplicities=isomorphic.multiplicities
        )

    def _dual_space(self, return_perm: bool = False
                    ) -> ProductSpace | tuple[ProductSpace, np.ndarray]:
        sectors, mults, perm = _sort_sectors(self.symmetry.dual_sectors(self.sectors), self.multiplicities)
        dual = ProductSpace([sp.dual for sp in reversed(self.spaces)], symmetry=self.symmetry,
                            _sectors=sectors, _multiplicities=mults)
        if return_perm:
            return dual, perm
        return dual
            
    @property
    def fusion_outcomes_sort(self):
        fusion_outcomes_sort = self.metadata.get('fusion_outcomes_sort', None)
        if fusion_outcomes_sort is None:
            grid = np.indices(tuple(space.num_sectors for space in self.spaces), np.intp)
            grid = grid.T.reshape(-1, len(self.spaces))
            sectors = self.symmetry.multiple_fusion_broadcast(
                *(sp.sectors[gr] for sp, gr in zip(self.spaces, grid.T))
            )
            multiplicities = np.prod([space.multiplicities[gr]
                                      for space, gr in zip(self.spaces, grid.T)], axis=0)
            _, _, fusion_outcomes_sort = _unique_sorted_sectors(sectors, multiplicities)
            self.metadata['fusion_outcomes_sort'] = fusion_outcomes_sort
        return fusion_outcomes_sort

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    def get_metadata(self, key: str, backend: TensorBackend = None):
        if key not in self.metadata:
            _, _, metadata = _fuse_spaces(self.symmetry, self.spaces, backend)
            self.metadata.update(metadata)
            if key not in self.metadata:
                msg = f'Unable to find key or generate it using _fuse_spaces: {key}'
                raise KeyError(msg)
        return self.metadata[key]

    def __getitem__(self, idx):
        return self.spaces[idx]

    def __iter__(self):
        return iter(self.spaces)
    
    def __len__(self):
        return self.num_spaces

    def _repr(self, show_symmetry: bool):
        indent = printoptions.indent * ' '
        lines = [f'ProductSpace(']
        if show_symmetry:
            lines.append(f'{indent}symmetry={self.symmetry!r},')
        num_lines = len(lines) + 1  # already consider final line ')'
        summarize = len(self.spaces) == 0  # if there are no spaces, auto-summarize
        for sp in self.spaces:
            sp_repr = sp._repr(show_symmetry=False)
            if sp_repr is None:
                summarize = True
                break
            next_space = indent + sp_repr.replace('\n', '\n' + indent) + ','
            additional_lines = 1 + next_space.count('\n')
            if num_lines + additional_lines > printoptions.maxlines_spaces:
                summarize = True
                break
            lines.append(next_space)
            num_lines += additional_lines
        lines.append(')')
        if not summarize:
            return '\n'.join(lines)
        # need to summarize
        elements = [f'<ProductSpace']
        if show_symmetry:
            elements.append(f'symmetry={self.symmetry!r}')
        elements.extend([
            f'{self.num_spaces} spaces',
            '>'
        ])
        one_line = ' '.join(elements)
        if len(one_line) <= printoptions.linewidth:
            return one_line
        elements[1:-1] = [f'{indent}{line}' for line in elements]
        if all(len(l) < printoptions.linewidth for l in elements) and len(elements) <= printoptions.maxlines_spaces:
            return '\n'.join(elements)
        return None
    
    def __eq__(self, other):
        if not isinstance(other, ProductSpace):
            return NotImplemented
        if self.num_spaces != other.num_spaces:
            return False
        return all(s1 == s2 for s1, s2 in zip(self.spaces, other.spaces))

    def as_ElementarySpace(self, is_dual: bool = None) -> ElementarySpace:
        res = ElementarySpace(symmetry=self.symmetry, sectors=self.sectors,
                              multiplicities=self.multiplicities)
        if is_dual is True:
            res = res.as_bra_space()
        return res

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, backend: TensorBackend = None
                        ) -> ProductSpace:
        sectors, multiplicities = _unique_sorted_sectors(
            sector_map(self.sectors), self.multiplicities
        )
        # OPTIMIZE can we preserve the metadata?
        return ProductSpace(
            spaces=[sp.change_symmetry(symmetry, sector_map, backend) for sp in self.spaces],
            symmetry=self.symmetry, backend=backend,
            _sectors=sectors, _multiplicities=multiplicities
        )

    def drop_symmetry(self, which: int | list[int] = None):
        which, remaining_symmetry = _parse_inputs_drop_symmetry(which, self.symmetry)
        return ProductSpace(spaces=[sp.drop_symmetry(which) for sp in self.spaces],
                            symmetry=remaining_symmetry)

    def fuse_states(self, states: list[Block], backend: TensorBackend) -> Block:
        """TODO"""
        if not self.symmetry.can_be_dropped:
            raise SymmetryError
        if self.symmetry.is_abelian:
            # first kron then use get_basis_transformation_perm()
            raise NotImplementedError  # TODO
        # other wise contract get_basis_transformation() with the states
        raise NotImplementedError

    def get_basis_transformation(self) -> np.ndarray:
        r"""Get the basis transformation from uncoupled to coupled basis.

        The uncoupled basis of the ProductSpace :math:`P = V \otimes W \otimes \dots \otimes Z`
        is given by products of the individual ("uncoupled") basis elements, i.e. by elements of
        the form :math:`v_{i_1} \otimes w_{i_2} \otimes \dots \otimes z_{i_n}`.
        In particular, the order for the uncoupled basis does *not*
        consider :attr:`ElementarySpace.basis_perm`, i.e. it is in general not grouped by sectors.
        The coupled basis is grouped and sorted organized by sectors.
        
        For abelian groups, this is achieved simply by permuting basis elements, see
        :meth:`get_basis_transformation_perm` for that permutation.
        For general groups, this is a more general unitary basis transformation.
        For non-group symmetries, this is not well defined.

        Returns
        -------
        trafo : ndarray
            A numpy array with shape ``(*space.dim for space in self.spaces, self.dim)``.
            The first axes go over the basis for each of the :attr:`spaces` (in public order).
            The last axis goes over the coupled basis of self.
            The entries are coefficients of the basis transformation such that

            .. math ::
                \ket{c} = \sum_{i_1, \dots, i_N} \texttt{trafo[i1, ..., iN, c]}
                          \ket{i_1} \otimes \dots \otimes \ket{i_n}

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

        .. testsetup :: get_basis_transformation
            from tenpy.linalg import ProductSpace, ElementarySpace, su2_symmetry

        .. doctest :: get_basis_transformation

            >>> spin_one_half = [1]  # sectors are labelled by 2*S
            >>> site = ElementarySpace(su2_symmetry, [spin_one_half])
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
        if not self.symmetry.can_be_dropped:
            raise SymmetryError
        if self.symmetry.is_abelian:
            transform = np.zeros((self.dim, self.dim), dtype=np.intp)
            perm = self.get_basis_transformation_perm()
            transform[np.ix_(perm, range(self.dim))] = 1.
            return np.reshape(transform, (*(s.dim for s in self.spaces), self.dim))
        raise NotImplementedError  # TODO
        
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
            from tenpy.linalg import ProductSpace, ElementarySpace, z2_symmetry

        .. doctest :: get_basis_transformation_perm

            >>> even, odd = [0], [1]
            >>> spin1 = ElementarySpace.from_basis(z2_symmetry, [even, odd, even])
            >>> product_space = ProductSpace([spin1, spin1])
            >>> perm = product_space.get_basis_transformation_perm()
            >>> perm
            array([0, 2, 4, 6, 8, 1, 3, 5, 7])
            >>> transform = np.zeros((9, 9))
            >>> transform[np.ix_(perm, range(9))] = 1.
            >>> np.all(product_space.get_basis_transformation() == transform.reshape((3, 3, 9)))
            True
        """
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
        order, i.e. in :attr:`ElementarySpace.sectors``. Take all combinations of sectors from all the
        spaces in C-style order, i.e. varying those from the last space the fastest.
        For each combination, take all of its fusion outcomes (TODO define order for FusionTree).
        The target permutation np.lexsort( .T)s the resulting list of sectors.
        """
        # OPTIMIZE this probably not the most efficient way to do this, but it hurts my brain
        #  and i need to get this work, if only in an ugly way...
        fusion_outcomes_inverse_sort = inverse_permutation(self.fusion_outcomes_sort)
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
            [[0], np.cumsum(fusion_outcome_multiplicities[self.fusion_outcomes_sort])]
        )
        # [j] : position of fusion channel after sorting
        sorted_pos = fusion_outcomes_inverse_sort[fusion_outcome]
        # [j] :: contribution from the sector, i.e. start of all the js of the same fusion channel
        sector_part = fusion_outcome_slices[sorted_pos]
        # [j] :: contribution from the multiplicities, i.e. position with all js of the same fusion channel
        degeneracy_part = np.sum(degeneracy_idcs * degeneracy_strides, axis=0)
        return inverse_permutation(sector_part + degeneracy_part)

    def insert_multiply(self, other: Space, pos: int, backend: TensorBackend | None = None
                        ) -> ProductSpace:
        """Insert an additional factor at given position.

        Parameters
        ----------
        other: Space
            The new factor to insert into :attr:`spaces`.
        pos: int
            The position of the new factor in the *result* :attr:`spaces`.

        Returns
        -------
        A new product space, consisting of ``spaces[:pos] + [other] + spaces[pos:]``.
        """
        new_num_spaces = self.num_spaces + 1
        assert -new_num_spaces <= pos < new_num_spaces
        if pos < 0:
            pos += new_num_spaces
        isomorphic = ProductSpace([self, other])  # this space has the same sectors and mults
        return ProductSpace(
            spaces=self.spaces[:pos] + [other] + self.spaces[pos:],
            symmetry=self.symmetry, backend=backend,
            _sectors=isomorphic.sectors, _multiplicities=isomorphic.multiplicities
        )
        
    def iter_uncoupled(self) -> Iterator[tuple[Sector]]:
        """Iterate over all combinations of sectors"""
        return it.product(*(s.sectors for s in self.spaces))

    def left_multiply(self, other: Space, backend: TensorBackend | None = None) -> ProductSpace:
        """Add a new factor at the left / beginning of the spaces"""
        return self.insert_multiply(other, 0, backend=backend)

    def right_multiply(self, other: Space, backend: TensorBackend | None = None) -> ProductSpace:
        """Add a new factor at the right / end of the spaces"""
        return self.insert_multiply(other, -1, backend=backend)


def _fuse_spaces(symmetry: Symmetry, spaces: list[Space], backend: TensorBackend | None = None):
    """Helper function, called as part of ``ProductSpace.__init__``.
    
    It determines the sectors and multiplicities of the ProductSpace.
    There is also a version of this function in the backends, i.e.
    :meth:`~tenpy.linalg.backends.abstract_backend.TensorBackend._fuse_spaces`, which may
    customize this behavior and in particular may return metadata, i.e. attributes to be added to
    the ProductSpace.
    This default implementation returns default metadata, with only ``fusion_outcomes_sort``
    if the symmetry is abelian and empty metadata otherwise.

    Returns
    -------
    sectors : 2D array of int
        The :attr:`ElementarySpace.sectors`.
    multiplicities : 1D array of int
        the :attr:`ElementarySpace.multiplicities`.
    metadata : dict
        A dictionary with string keys and arbitrary values.
        These will be added as attributes of the ProductSpace
    """
    if symmetry is None:
        if len(spaces) == 0:
            raise ValueError('If spaces is empty, the symmetry arg is required.')
        symmetry = spaces[0].symmetry
    
    if backend is not None:
        try:
            return backend._fuse_spaces(symmetry=symmetry, spaces=spaces)
        except NotImplementedError:
            pass

    if symmetry.is_abelian:
        if len(spaces) == 0:
            metadata = dict(fusion_outcomes_sort=np.array([0], dtype=int))
            return symmetry.trivial_sector[None, :], [1], metadata
        grid = np.indices(tuple(space.num_sectors for space in spaces), np.intp)
        grid = grid.T.reshape(-1, len(spaces))
        sectors = symmetry.multiple_fusion_broadcast(
            *(sp.sectors[gr] for sp, gr in zip(spaces, grid.T))
        )
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(spaces, grid.T)],
                                  axis=0)
        sectors, multiplicities, fusion_outcomes_sort = _unique_sorted_sectors(sectors, multiplicities)
        metadata = dict(fusion_outcomes_sort=fusion_outcomes_sort)
        return sectors, multiplicities, metadata

    # define recursively. base cases:
    if len(spaces) == 0:
        return symmetry.trivial_sector[None, :], [1], {}

    if len(spaces) == 1:
        return spaces[0].sectors, spaces[0].multiplicities, {}

    sectors_1, mults_1, _ = _fuse_spaces(symmetry, spaces[:-1])

    sector_arrays = []
    mult_arrays = []
    for s2, m2 in zip(spaces[-1].sectors, spaces[-1].multiplicities):
        for s1, m1 in zip(sectors_1, mults_1):
            new_sects = symmetry.fusion_outcomes(s1, s2)
            sector_arrays.append(new_sects)
            if symmetry.fusion_style <= FusionStyle.multiple_unique:
                new_mults = m1 * m2 * np.ones(len(new_sects), dtype=int)
            else:
                # OPTIMIZE support batched N symbol?
                new_mults = m1 * m2 * np.array([symmetry._n_symbol(s1, s2, c) for c in new_sects], dtype=int)
            mult_arrays.append(new_mults)
    sectors, multiplicities, _ = _unique_sorted_sectors(
        np.concatenate(sector_arrays, axis=0),
        np.concatenate(mult_arrays, axis=0)
    )
    return sectors, multiplicities, {}


def _unique_sorted_sectors(unsorted_sectors: SectorArray, unsorted_multiplicities: np.ndarray):
    """Sort sectors and merge duplicates.

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
    sectors, multiplicities, perm = _sort_sectors(unsorted_sectors, unsorted_multiplicities)
    slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
    diffs = find_row_differences(sectors, include_len=True)
    slices = slices[diffs]
    multiplicities = slices[1:] - slices[:-1]
    sectors = sectors[diffs[:-1]]
    return sectors, multiplicities, perm


def _sort_sectors(sectors: SectorArray, multiplicities: np.ndarray):
    perm = np.lexsort(sectors.T)
    return sectors[perm], multiplicities[perm], perm


def _parse_inputs_drop_symmetry(which: int | list[int] | None, symmetry: Symmetry
                                ) -> tuple[list[int] | None, Symmetry]:
    """Input parsing for :meth:`Space.drop_symmetry`.

    Returns
    -------
    which : None | list of int
        Which symmetries to drop, as integers in ``range(len(symmetries.factors))``.
        ``None`` indicates to drop all.
    remaining_symmetry : Symmetry
        The symmetry that remains.
    """
    if which is None or which == []:
        pass
    elif isinstance(symmetry, ProductSymmetry):
        which = to_iterable(which)
        num_factors = len(symmetry.factors)
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
        remaining_symmetry = no_symmetry
    else:
        factors = [f for i, f in enumerate(symmetry.factors) if i not in which]
        if len(factors) == 1:
            remaining_symmetry = factors[0]
        else:
            remaining_symmetry = ProductSymmetry(factors)

    return which, remaining_symmetry
    
