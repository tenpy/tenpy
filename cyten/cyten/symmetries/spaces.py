"""The spaces, i.e. the legs of a tensor."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import bisect
import itertools as it
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Sequence
from math import prod
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import ndarray

from ..dummy_config import printoptions
from ..tools.misc import (
    UNSPECIFIED,
    combine_permutations,
    find_row_differences,
    inverse_permutation,
    iter_common_sorted_arrays,
    make_grid,
    make_stride,
    rank_data,
    to_iterable,
    to_valid_idx,
)
from ..tools.string import format_like_list
from ._symmetries import FusionStyle, ProductSymmetry, Sector, SectorArray, Symmetry, SymmetryError, no_symmetry
from .trees import FusionTree, fusion_trees

if TYPE_CHECKING:
    from ..block_backends import Block


class Leg(metaclass=ABCMeta):
    """Common base class for a single leg of a tensor.

    A single leg on a tensor can either be an :class:`ElementarySpace` or, e.g. as the result
    of combining legs, a :class:`LegPipe`.

    Attributes
    ----------
    symmetry : Symmetry
        The symmetry associated with this leg.
    dim : int or float
        The (quantum-)dimension of this leg.
        Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
    is_dual : bool
        A boolean flag that changes when the :attr:`dual` is taken. May or may not have additional
        meaning and implications, depending on the concrete subclass of :class:`Leg`.

    """

    def __init__(self, symmetry: Symmetry, dim: int | float, is_dual: bool, basis_perm: ndarray | None):
        self.symmetry = symmetry
        self.dim = dim
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
        """Perform sanity checks."""
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

    @abstractmethod
    def as_Space(self) -> Space:
        """Convert to (an appropriate subclass of) :class:`Space`."""
        ...

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        """Convert to an isomorphic :class:`ElementarySpace`"""
        # can be overridden for performance
        return self.as_Space().as_ElementarySpace(is_dual=is_dual)

    @property
    @abstractmethod
    def dual(self) -> Leg:
        """The dual leg, that is obtained when bending this leg."""
        ...

    @property
    @abstractmethod
    def is_trivial(self) -> bool: ...

    @property
    def basis_perm(self) -> ndarray:
        """Permutation that translates between public and internal basis order.

        For the inverse permutation, see :attr:`inverse_basis_perm`.

        The tensor manipulations of ``cyten`` benefit from choosing a canonical order for the
        basis of vector spaces. This attribute translates between the "public" order of the basis,
        in which e.g. the inputs to :meth:`from_dense_block` are interpreted to this internal order,
        such that ``public_basis[basis_perm] == internal_basis``.
        The internal order is such that the basis vectors are grouped and sorted by sector.
        We can translate indices as ``public_idx == basis_perm[internal_idx]``.
        Only available if ``symmetry.can_be_dropped``, as otherwise there is no well-defined
        notion of a basis.

        ``_basis_perm`` is the internal version which may be ``None`` if the permutation is trivial.
        See also :meth:`apply_basis_perm`.
        """
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._basis_perm is None:
            return np.arange(self.dim)
        return self._basis_perm

    @basis_perm.setter
    def basis_perm(self, basis_perm):
        self.set_basis_perm(basis_perm=basis_perm)

    @property
    def flat_legs(self) -> list[ElementarySpace]:
        """Flatten until there are no more pipes.

        See Also
        --------
        flat_spaces : Keeps :class:`AbelianLegPipes` nested.

        """
        return [self]

    @property
    def flat_spaces(self) -> list[ElementarySpace]:
        """Flatten until we get spaces.

        See Also
        --------
        flat_legs : Also flattens :class:`AbelianLegPipes`.

        """
        return [self]

    @property
    def inverse_basis_perm(self) -> ndarray:
        """Inverse permutation of :attr:`basis_perm`."""
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is None:
            return np.arange(self.dim)
        return self._inverse_basis_perm

    @inverse_basis_perm.setter
    def inverse_basis_perm(self, inverse_basis_perm):
        self.set_basis_perm(inverse_basis_perm=inverse_basis_perm)

    @property
    def num_flat_legs(self) -> int:
        """The number of :attr:`flat_legs`."""
        return 1

    @property
    def ascii_arrow(self) -> str:
        """A single character arrow, for use in tensor diagrams

        Indicates (a) if the leg is a pipe and (b) for ElementarySpaces, the duality
        """
        is_pipe = isinstance(self, LegPipe)
        if isinstance(self, ElementarySpace):
            return {
                (False, False): 'v',
                (False, True): '▼',
                (True, False): '^',
                (True, True): '▲',
            }[self.is_dual, is_pipe]
        if is_pipe:
            return '║'
        raise RuntimeError  # should have already covered all cases

    @abstractmethod
    def __eq__(self, other): ...

    def apply_basis_perm(self, arr, axis: int = 0, inverse: bool = False, pre_compose: bool = False):
        """Apply the basis_perm, i.e. form ``arr[self.basis_perm]``.

        This is the preferred method of accessing the permutation, since we may skip applying
        trivial permutations.

        Parameters
        ----------
        arr : numpy array
            The data to act on.
        axis : int
            Which axis of ``arr`` to act on. We use ``numpy.take(arr, perm, axis)``.
        inverse : bool
            If we should apply the inverse permutation :attr:`inverse_basis_perm` instead.
        pre_compose : bool
            If we should pre-compose instead, i.e. form ``basis_perm[arr]``.
            Note that in that case, `axis` is ignored.

        """
        # this implementation assumes _basis_perm. AbelianLegPipe overrides this method.
        perm = self._inverse_basis_perm if inverse else self._basis_perm
        if perm is None:
            # perm is identity permutation
            return arr
        if pre_compose:
            assert axis == 0
            return perm[arr]
        return np.take(arr, perm, axis=axis)

    def set_basis_perm(
        self, basis_perm: Sequence[int] | None = UNSPECIFIED, inverse_basis_perm: Sequence[int] | None = UNSPECIFIED
    ):
        """Common setter for :attr:`basis_perm` and :attr:`inverse_basis_perm`."""
        if basis_perm is UNSPECIFIED and inverse_basis_perm is UNSPECIFIED:
            raise ValueError('Must specify at least one of the arguments')
        if basis_perm is UNSPECIFIED:
            if inverse_basis_perm is None:
                basis_perm = None
            else:
                inverse_basis_perm = np.asarray(inverse_basis_perm, int)
                assert inverse_basis_perm.shape == (self.dim,)
                basis_perm = inverse_permutation(inverse_basis_perm)
        elif inverse_basis_perm is UNSPECIFIED:
            if basis_perm is None:
                inverse_basis_perm = None
            else:
                basis_perm = np.asarray(basis_perm, int)
                assert basis_perm.shape == (self.dim,)
                inverse_basis_perm = inverse_permutation(basis_perm)
        elif basis_perm is None and inverse_basis_perm is None:
            pass
        elif basis_perm is None or inverse_basis_perm is None:
            raise ValueError('Can not mix None with an explicit permutation')
        else:
            basis_perm = np.asarray(basis_perm, int)
            assert basis_perm.shape == (self.dim,)
            inverse_basis_perm = np.asarray(inverse_basis_perm, int)
            assert inverse_basis_perm.shape == (self.dim,)
            if not np.all(basis_perm[inverse_basis_perm] == np.arange(self.dim)):
                raise ValueError('The given permutations are not mutually inverse!')
        self._basis_perm = basis_perm
        self._inverse_basis_perm = inverse_basis_perm


class LegPipe(Leg):
    """A group of legs, i.e. resulting from :func:`~cyten.tensors.combine_legs`.

    Note that the abelian backend defines a custom subclass.

    The :attr:`dual` of a pipe is given by another :class:`LegPipe`, which consists of the
    dual of each of the :attr:`legs`, *in reverse order*. We also flip the :attr:`is_dual`
    attribute to keep track of that (but the attribute has no further meaning).

    Attributes
    ----------
    legs
        The legs that were grouped, and that this pipe can be split into.
    combine_cstyle : bool
        The leg pipe defines an order in which multi-indices (one per leg) are combined into
        a single index. This can either be C-style (where the index for the last leg is varied the
        fastest) or F-style (where the first index is varied the fastest). For compatibility with
        the default behavior of ``np.reshape``, we favor C-style. However, if the `legs` were in
        the domain (at the top) of a tensor before combining, the conventional leg order implies
        a reversal of their order in ``Tensor.legs``. Thus, pipes in the domain should have F-style
        combine. Consistent with this expectation, the style is flipped on taking the :attr:`dual`

    See Also
    --------
    TensorProduct

    """

    def __init__(self, legs: Sequence[Leg], is_dual: bool = False, combine_cstyle: bool = True):
        self.legs = legs[:]
        self.num_legs = num_legs = len(legs)
        assert num_legs > 0
        self.combine_cstyle = combine_cstyle

        if all(l._basis_perm is None for l in legs):
            basis_perm = None
        else:
            basis_perm = combine_permutations([l.basis_perm for l in self.legs], cstyle=combine_cstyle)
        Leg.__init__(
            self, symmetry=legs[0].symmetry, dim=prod(l.dim for l in legs), is_dual=is_dual, basis_perm=basis_perm
        )

    def test_sanity(self):
        """Perform sanity checks."""
        assert all(l.symmetry == self.symmetry for l in self.legs)
        for l in self.legs:
            l.test_sanity()
        Leg.test_sanity(self)

    def as_Space(self):
        return TensorProduct([l.as_Space() for l in self.legs], symmetry=self.symmetry)

    @property
    def dual(self) -> LegPipe:
        return LegPipe(
            [l.dual for l in reversed(self.legs)], is_dual=not self.is_dual, combine_cstyle=not self.combine_cstyle
        )

    @property
    def is_trivial(self) -> bool:
        return all(l.is_trivial for l in self.legs)

    @property
    def flat_legs(self) -> list[ElementarySpace]:
        return list(it.chain.from_iterable(l.flat_legs for l in self.legs))

    @property
    def flat_spaces(self) -> list[ElementarySpace]:
        return list(it.chain.from_iterable(l.flat_spaces for l in self.legs))

    @property
    def num_flat_legs(self) -> int:
        return sum(l.num_flat_legs for l in self.legs)

    def set_basis_perm(
        self, basis_perm: Sequence[int] | None = UNSPECIFIED, inverse_basis_perm: Sequence[int] | None = UNSPECIFIED
    ):
        msg = f'Can not set basis_perm for {type(self).__name__}.'
        raise TypeError(msg)

    def __eq__(self, other):
        if not isinstance(other, LegPipe):
            return NotImplemented
        if isinstance(self, AbelianLegPipe) != isinstance(other, AbelianLegPipe):
            return False
        if self.is_dual != other.is_dual:
            return False
        if self.combine_cstyle != other.combine_cstyle:
            return False
        if self.num_legs != other.num_legs:
            return False
        if not all(l1 == l2 for l1, l2 in zip(self.legs, other.legs)):
            return False
        return True

    def __getitem__(self, idx):
        return self.legs[idx]

    def __iter__(self):
        return iter(self.legs)

    def __len__(self):
        return self.num_legs

    def __repr__(self, show_symmetry: bool = True, one_line=False):
        ClsName = type(self).__name__

        if one_line:
            if show_symmetry:
                res = (
                    f'{ClsName}(num_legs={self.num_legs}, is_dual={self.is_dual}, '
                    f'symmetry={self.symmetry!r}, combine_cstyle={self.combine_cstyle})'
                )
                if len(res) <= printoptions.linewidth:
                    return res
                return self.__repr__(show_symmetry=False, one_line=True)
            else:
                res = (
                    f'{ClsName}(num_legs={self.num_legs}, is_dual={self.is_dual}, combine_cstyle={self.combine_cstyle})'
                )
                if len(res) <= printoptions.linewidth:
                    return res
                raise RuntimeError  # the above should always fit in linewidth ...

        lines = [f'{ClsName}([']
        indent = printoptions.indent * ' '

        for force_children_one_line in [False, True]:
            for leg in self.legs:
                rep = leg.__repr__(show_symmetry=False, one_line=force_children_one_line)
                for new_line in rep.split('\n'):
                    lines.append(indent + new_line)
            if show_symmetry:
                lines.append(f'], is_dual={self.is_dual}, symmetry={self.symmetry!r})')
            else:
                lines.append(f'], is_dual={self.is_dual})')
            maxlines_ok = len(lines) <= printoptions.maxlines_spaces
            linewidth_ok = all(len(l) < printoptions.linewidth for l in lines)
            if maxlines_ok and linewidth_ok:
                return '\n'.join(lines)

        # fallback
        return self.__repr__(show_symmetry=show_symmetry, one_line=True)


class Space(metaclass=ABCMeta):
    r"""Base class for symmetry spaces, see :class:`ElementarySpace` for the standard case.

    A symmetry space is e.g. a vector space with a representation of a symmetry group.

    Each symmetry space is equivalent to a direct sum of sectors, that
    is :math:`V \cong \bigoplus_a \bigoplus_{\mu=1}{N_a} a`.
    This is e.g. because the representation of the symmetry group is equivalent to a direct sum of
    irreducible representations. From a different perspective, the vector space decomposes into
    different charge sectors of the conserved charge. The unique sectors :math:`a` that appear in
    the decomposition at least once, e.g. with `N_a > 0`, are stored in :attr:`sector_decomposition`
    in a canonical order, while their multiplicities :math:`N_a` are stored in :attr:`multiplicities`.

    Attributes
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sector_decomposition : 2D numpy array of int
        The unique sectors that appear in the sector decomposition. A 2D array of integers with
        axes [s, q] where s goes over different sectors and q over the (one or more) numbers needed
        to label a sector. The sectors (to be precise, the rows ``sector_decomposition[i, :]``) are
        unique. We use :attr:`multiplicities` to  account for duplicates.
    sector_order : 'sorted' | 'dual_sorted' | None
        Indicates if (and how) the :attr:`sector_decomposition` is sorted.
        If ``'sorted'``, indicates that they are sorted by sector, i.e. such that
        ``np.lexsort(sector_decomposition.T) == np.arange(num_sectors)``.
        If ``'dual_sorted'``, indicated that the duals are sorted, i.e. such that
        ``np.lexsort(dual_sectors(sector_decomposition).T) == np.arange(num_sectors)``.
        If ``None``, no particular order is guaranteed.
    multiplicities : 1D numpy array of int | None
        How often each of the sectors in :attr:`sector_decomposition` appears. A 1D array of positive
        integers with axis [s]. ``sector_decomposition[i, :]`` appears ``multiplicities[i]`` times.
        ``None`` is equivalent to a sequence of ``1`` of appropriate length.
    num_sectors : int
        The number of sectors in the :attr:`sector_decomposition`.
        This is the number of *unique* sectors, regardless of their multiplicity, and different
        from the total number of sectors ``sum(multiplicities)``.
    sector_dims : 1D array of int | None
        If ``symmetry.can_be_dropped``, the integer dimension of each sector of the
        :attr:`sector_decomposition`. Otherwise, not defined and set to ``None``.
    sector_qdims : 1D array of float
        The (quantum) dimension of each of the sectors. Unlike :attr:`sector_dims` this is always
        defined, but may not always be integer.
    dim : int | float
        The total dimension. Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
    slices : 2D numpy array of int | None
        For every sector ``sector_decomposition[n]``, the start ``slices[n, 0]`` and stop
        ``slices[n, 1]`` of indices (in the *internal* basis order) that belong to this sector.
        Conversely, ``basis_perm[slices[n, 0]:slices[n, 1]]`` are the elements of the public
        basis that live in ``sector_decomposition[n]``. Only available if ``symmetry.can_be_dropped``.

    """

    def __init__(
        self,
        symmetry: Symmetry,
        sector_decomposition: SectorArray | Sequence[Sequence[int]],
        multiplicities: Sequence[int] | None = None,
        sector_order: Literal['sorted'] | Literal['dual_sorted'] | None = None,
    ):
        self.symmetry = symmetry
        self.sector_decomposition = sector_decomposition = np.asarray(sector_decomposition, dtype=int)
        self.sector_order = sector_order
        if sector_decomposition.ndim != 2 or sector_decomposition.shape[1] != symmetry.sector_ind_len:
            msg = f'Wrong sectors.shape: Expected (*, {symmetry.sector_ind_len}), got {sector_decomposition.shape}.'
            raise ValueError(msg)
        assert sector_decomposition.ndim == 2 and sector_decomposition.shape[1] == symmetry.sector_ind_len
        self.num_sectors = num_sectors = len(sector_decomposition)
        if multiplicities is None:
            self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int)
        else:
            self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int)
            assert multiplicities.shape == (num_sectors,)
        if symmetry.can_be_dropped:
            self.sector_dims = sector_dims = symmetry.batch_sector_dim(sector_decomposition)
            self.sector_qdims = sector_dims
            slices = np.zeros((len(sector_decomposition), 2), dtype=np.intp)
            slices[:, 1] = slice_ends = np.cumsum(multiplicities * sector_dims)
            slices[1:, 0] = slice_ends[:-1]  # slices[0, 0] remains 0, which is correct
            self.slices = slices
            self.dim = np.sum(sector_dims * multiplicities).item()
        else:
            self.sector_dims = None
            self.sector_qdims = sector_qdims = symmetry.batch_qdim(sector_decomposition)
            self.slices = None
            self.dim = np.sum(sector_qdims * multiplicities).item()

    def test_sanity(self):
        """Perform sanity checks."""
        assert self.dim >= 0
        # sectors
        if self.sector_decomposition.shape != (self.num_sectors, self.symmetry.sector_ind_len):
            raise AssertionError('wrong sectors.shape')
        assert self.symmetry.are_valid_sectors(self.sector_decomposition), 'invalid sectors'
        assert len(np.unique(self.sector_decomposition, axis=0)) == self.num_sectors, 'duplicate sectors'
        if self.sector_order == 'sorted':
            assert np.all(np.lexsort(self.sector_decomposition.T) == np.arange(self.num_sectors)), 'wrong sector order'
        elif self.sector_order == 'dual_sorted':
            expect_sorted = self.symmetry.dual_sectors(self.sector_decomposition)
            assert np.all(np.lexsort(expect_sorted.T) == np.arange(self.num_sectors)), 'wrong sector order'
        elif self.sector_order is None:
            pass  # nothing to check
        else:
            raise AssertionError(f'Invalid sector_order: {self.sector_order}')
        # multiplicities
        assert np.all(self.multiplicities > 0)
        assert self.multiplicities.shape == (self.num_sectors,)
        if self.symmetry.can_be_dropped:
            # slices
            assert self.slices.shape == (self.num_sectors, 2)
            slice_diffs = self.slices[:, 1] - self.slices[:, 0]
            assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self.sector_decomposition))
            expect_diffs = self.sector_dims * self.multiplicities
            assert np.all(slice_diffs == expect_diffs)
            # slices should be consecutive
            if self.num_sectors > 0:
                assert self.slices[0, 0] == 0
                assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
                assert self.slices[-1, 1] == self.dim

    # ABSTRACT

    @property
    @abstractmethod
    def dual(self) -> Space:
        """The dual space of the same type.

        A dual space necessarily has a :attr:`sector_decomposition` which consists of the
        :meth:`Symmetry.dual_sectors` of the original (though not necessarily in order).

        Strictly speaking, this only guarantees to give one possible choice for a dual space and
        might differ from *the* dual space by an irrelevant isomorphism.
        """
        ...

    @property
    def is_trivial(self) -> bool:
        """If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.

        A trivial space is one-dimensional and transforms trivially under a symmetry group.
        In category speak, it is (isomorphic to) the monoidal unit.
        """
        if self.num_sectors > 1:
            return False
        if self.multiplicities[0] > 1:
            return False
        return np.all(self.sector_decomposition[0] == self.symmetry.trivial_sector)

    @abstractmethod
    def __eq__(self, other):
        msg = f'{self.__class__.__name__} does not support "==" comparison. Use `is_isomorphic_to` instead.'
        raise TypeError(msg)

    def is_isomorphic_to(self, other: Space) -> bool:
        """If the two spaces are isomorphic, i.e. have the same :attr:`sector_decomposition`."""
        if self.symmetry != other.symmetry:
            raise SymmetryError('Incompatible symmetries')
        if self.num_sectors != other.num_sectors:
            return False

        # find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and ``other.sector_decomposition[perm2]``
        # have the same sorting convention and can be directly compared
        if self.sector_order is None:
            if other.sector_order == 'sorted':
                perm1 = np.lexsort(self.sector_decomposition.T)
                perm2 = slice(None, None, None)
            elif other.sector_order == 'dual_sorted':
                perm1 = np.lexsort(self.symmetry.dual_sectors(self.sector_decomposition).T)
                perm2 = slice(None, None, None)
            else:
                perm1 = np.lexsort(self.sector_decomposition.T)
                perm2 = np.lexsort(other.sector_decomposition.T)
        elif other.sector_order is None:
            if self.sector_order == 'sorted':
                perm1 = slice(None, None, None)
                perm2 = np.lexsort(other.sector_decomposition.T)
            elif self.sector_order == 'dual_sorted':
                perm1 = slice(None, None, None)
                perm2 = np.lexsort(self.symmetry.dual_sectors(other.sector_decomposition).T)
            else:
                raise RuntimeError  # case should have been covered above
        elif self.sector_order == other.sector_order:
            perm1 = perm2 = slice(None, None, None)
        elif self.sector_order == 'sorted':
            perm1 = slice(None, None, None)
            perm2 = np.lexsort(other.sector_decomposition.T)
        elif other.sector_order == 'sorted':
            perm1 = np.lexsort(self.sector_decomposition.T)
            perm2 = slice(None, None, None)
        else:
            raise RuntimeError  # all cases should have been covered.

        if not np.all(self.multiplicities[perm1] == other.multiplicities[perm2]):
            return False
        return np.all(self.sector_decomposition[perm1] == other.sector_decomposition[perm2])

    def is_subspace_of(self, other: Space) -> bool:
        """Whether self is (isomorphic to) a subspace of other.

        Per convention, self is never a subspace of other, if the :attr:`symmetry` are different.

        See Also
        --------
        ElementarySpace.from_largest_common_subspace

        """
        if not self.symmetry.is_same_symmetry(other.symmetry):
            return False
        if self.num_sectors == 0:
            return True
        if self.sector_order == 'sorted' == other.sector_order:
            # sectors are sorted, so we can just iterate over both of them
            n_self = 0
            for other_sector, other_mult in zip(other.sector_decomposition, other.multiplicities):
                if np.all(self.sector_decomposition[n_self] == other_sector):
                    if self.multiplicities[n_self] > other_mult:
                        return False
                    n_self += 1
                if n_self == self.num_sectors:
                    # have checked all sectors of self
                    return True
            # reaching this line means self has sectors which other does not have
            return False

        # OPTIMIZE sort once instead of looking up each time
        num_sectors_checked = 0
        for sector, mult in zip(other.sector_decomposition, other.multiplicities):
            m = self.sector_multiplicity(sector)
            if m == 0:
                continue
            if m > mult:
                return False
            num_sectors_checked += 1
        if num_sectors_checked < self.num_sectors:
            # this means self has some sectors that other doesn't have
            return False
        return True

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        """Convert to an isomorphic :class:`ElementarySpace`."""
        if is_dual:
            defining_sectors = self.symmetry.dual_sectors(self.sector_decomposition)
            is_sorted = self.sector_order == 'dual_sorted'
        else:
            defining_sectors = self.sector_decomposition
            is_sorted = self.sector_order == 'sorted'

        if is_sorted:
            return ElementarySpace(
                symmetry=self.symmetry,
                defining_sectors=defining_sectors,
                multiplicities=self.multiplicities,
                is_dual=is_dual,
            )
        return ElementarySpace.from_defining_sectors(
            symmetry=self.symmetry,
            defining_sectors=defining_sectors,
            multiplicities=self.multiplicities,
            is_dual=is_dual,
            unique_sectors=True,
        )

    @abstractmethod
    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, injective: bool = False) -> ElementarySpace:
        """Change the symmetry by specifying how the sectors change.

        .. note ::
            This interface assumes that a single sector of the old symmetry is mapped to a single
            sector of the new symmetry, i.e. that the functor that we realize here preserves
            simple objects. This does e.g. not cover the case of relaxing SU(2) to its U(1)
            subgroup.

        Parameters
        ----------
        symmetry : :class:`~cyten.groups.Symmetry`
            The symmetry of the new space
        sector_map : function (SectorArray,) -> (SectorArray,)
            A map of sectors (2D int arrays), such that ``new_sectors = sector_map(old_sectors)``.
            The map is assumed to cooperate with duality, i.e. we assume without checking that
            ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
            ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
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

    # CONCRETE IMPLEMENTATIONS

    def as_Space(self):
        return self

    def sector_decomposition_where(self, sector: Sector) -> int | None:
        """Find the index of a given sector in the :attr:`sector_decomposition`.

        Returns
        -------
        idx : int | None
            If the `sector` is found the :attr:`sector_decomposition`, its index there such
            that ``sector_decomposition[idx] == sector``. Otherwise ``None``.

        """
        # OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
        where = np.where(np.all(self.sector_decomposition == sector, axis=1))[0]
        if len(where) == 0:
            return None
        if len(where) == 1:
            return int(where[0])
        # sector_decomposition should be unique, so one of the above if statements should trigger.
        # If we get here, something is wrong / inconsistent.
        self.test_sanity()  # this should raise an informative error
        raise RuntimeError('This should not happen. Please report this bug on github.')

    def sector_multiplicity(self, sector: Sector) -> int:
        """The multiplicity of a given sector in the :attr:`sector_decomposition`."""
        idx = self.sector_decomposition_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]


class ElementarySpace(Space, Leg):
    r"""A :class:`Space` that is defined as (the dual of) a direct sum of sectors.

    While every :class:`Space` is isomorphic to a direct sum of sectors, an :class:`ElementarySpace`
    is by definition *equal* to such a direct sum, or to the dual of such a sum. We distinguish
    "ket" spaces :math:`V_k := a_1 \oplus a_2 \oplus \dots \plus a_N` with ``is_dual=False`` and
    "bra" spaces :math:`V_b := [b_1 \oplus b_2 \oplus \dots \plus b_N]^*` with ``is_dual=True``.
    The listed sectors, :math:`\{a_n\}` for the ket space :math:`V_k` and the :math:`\{b_n\}`
    for the bra space, are the :attr:`defining_sectors` of the space. For a ket space, they coincide
    with the :attr:`sector_decomposition`, while for a bra space they are mutually dual, since
    we have :math:`V_b \cong \bar{b}_1 \oplus \bar{b}_2 \oplus \dots \plus \bar{b}_N`.

    We impose a canonical order of sectors, such that the :attr:`defining_sectors` are sorted.
    This in turn means that the :attr:`sector_order` is ``'sorted'`` for ket spaces and
    ``'dual_sorted'`` for bra spaces.

    If the symmetry :attr:`Symmetry.can_be_dropped`, there is a notion of a basis for the
    spaces. We demand the basis to be compatible with the symmetry, i.e. each basis vector
    needs to lie in one of the sectors of the symmetry. The *internal* basis order that results
    from demanding that the sectors are contiguous and sorted may, however, not be the desired
    basis order, e.g. for matrix representations. For example, the standard basis of a spin-1
    degree of freedom with ``'Sz_parity'`` conservation has sectors ``[[1], [0], [1]]`` and is
    neither sorted by sector nor contiguous. We allow these different *public* basis orders
    and store the relevant permutations as :attr:`basis_perm` and :attr:`inverse_basis_perm`.
    See also :attr:`sectors_of_basis` and :meth:`from_basis`.

    Parameters
    ----------
    symmetry, sectors, multiplicities, is_dual, basis_perm
        Like attributes of the same name, except nested sequences are allowed in place of arrays.

    Attributes
    ----------
    is_dual: bool
        If this is a ket space (``False``) or a bra space (``True``).
    defining_sectors: 2D array of int
        The defining sectors, see class docstring of :class:`ElementarySpace`.
        Is ``np.lexsort( .T)``-ed.
        The :attr:`sector_decomposition` is equal for ket spaces (``is_dual=False``) or given by
        the respective :meth:`~cyten.symmetries.Symmetry.dual_sectors` for bra spaces.

    """

    def __init__(
        self,
        symmetry: Symmetry,
        defining_sectors: SectorArray,
        multiplicities: ndarray = None,
        is_dual: bool = False,
        basis_perm: ndarray | None = None,
    ):
        defining_sectors = np.asarray(defining_sectors, dtype=int)
        assert symmetry.are_valid_sectors(defining_sectors), 'invalid sectors'
        if is_dual:
            sector_decomposition = symmetry.dual_sectors(defining_sectors)
            sector_order = 'dual_sorted'
        else:
            sector_decomposition = defining_sectors
            sector_order = 'sorted'
        Space.__init__(
            self,
            symmetry=symmetry,
            sector_decomposition=sector_decomposition,
            multiplicities=multiplicities,
            sector_order=sector_order,
        )
        Leg.__init__(self, symmetry=symmetry, dim=self.dim, is_dual=is_dual, basis_perm=basis_perm)
        self.defining_sectors = defining_sectors

    def test_sanity(self):
        """Perform sanity checks."""
        assert self.defining_sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len)
        if self.is_dual:
            assert self.sector_order == 'dual_sorted'
        else:
            assert self.sector_order == 'sorted'
        Space.test_sanity(self)
        Leg.test_sanity(self)

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors_of_basis: Sequence[Sequence[int]]) -> ElementarySpace:
        """Create an ElementarySpace by specifying the sector of every basis element.

        This requires that the symmetry :attr:`~cyten.symmetries.Symmetry.can_be_dropped`, such
        that there is a useful notion of a basis.

        .. note ::
            Unlike :meth:`from_defining_sectors`, this method expects the same sector to be listed
            multiple times, if the sector is multi-dimensional. The Hilbert Space of a spin-one-half
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_defining_sectors(su2, [spin_half])``. In the former case
            we need to list the same sector both for the spin up and spin down state.

        .. note ::
            This classmethod always creates ket-spaces with ``is_dual=False``. This is to make
            it unambiguous if `sectors_of_basis` refers to the :attr:`sector_decomposition` or the
            :attr:`defining_sectors`, since they coincide for ket spaces.
            Use :attr:`dual` or :meth:`as_bra_space` to create bra spaces.

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

        See Also
        --------
        :attr:`sectors_of_basis`
            Reproduces the `sectors_of_basis` parameter.
        from_defining_sectors
            Similar to the constructor, but with fewer requirements.

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
            msg = (
                'Sectors must appear in whole multiplets, i.e. a number of times that is an '
                'integer multiple of their dimension.'
            )
            raise ValueError(msg)
        return cls(
            symmetry=symmetry,
            defining_sectors=sectors,
            multiplicities=multiplicities,
            is_dual=False,
            basis_perm=basis_perm,
        )

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions: list[ElementarySpace]) -> ElementarySpace:
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
        symmetry = ProductSymmetry.from_nested_factors([s.symmetry for s in independent_descriptions])
        if not symmetry.can_be_dropped:
            msg = f'from_independent_symmetries is not supported for {symmetry}.'
            # TODO is there a way to define this? the straight-forward picture works only if we have
            #      a vector space and can identify states.
            #      note: this interface is more general than it needs to be. The use case in
            #            GroupedSite would allow us to specialize, if that is easier. A given state
            #            is in the trivial sector for all but one of the independent_descriptions.
            raise SymmetryError(msg)
        sectors_of_basis = np.concatenate([s.sectors_of_basis for s in independent_descriptions], axis=1)
        return cls.from_basis(symmetry, sectors_of_basis)

    @classmethod
    def from_largest_common_subspace(cls, *spaces: Space, is_dual: bool = False) -> ElementarySpace:
        """The largest common subspace of a list of spaces.

        The largest :class:`ElementarySpace` that :meth:`is_subspace_of` all of the `spaces`.
        I.e. the :attr:`sector_decomposition` is given by the "sector-wise minimum" of all
        multiplicities of the `spaces`.

        See Also
        --------
        is_subspace_of

        """
        if len(spaces) == 0:
            raise ValueError('Need at least one space')
        if len(spaces) == 1:
            return spaces[0].as_ElementarySpace(is_dual=is_dual)
        sp1, sp2, *more = spaces
        if more:
            # OPTIMIZE directly implement for many
            sp = ElementarySpace.from_largest_common_subspace(sp1, sp2)
            return ElementarySpace.from_largest_common_subspace(sp, *more, is_dual=is_dual)
        sectors = []
        mults = []
        if sp1.sector_order == 'sorted' == sp2.sector_order:
            for i, j in iter_common_sorted_arrays(sp1.sector_decomposition, sp2.sector_decomposition):
                sectors.append(sp1.sector_decomposition[i])
                mults.append(min(sp1.multiplicities[i], sp2.multiplicities[j]))
        else:
            # OPTIMIZE implementation for mixed orders? or just override this in ElementarySpace?
            for i, sector in enumerate(sp1.sector_decomposition):
                j = sp2.sector_decomposition_where(sector)
                if j is None:
                    continue
                sectors.append(sector)
                mults.append(min(sp1.multiplicities[i], sp2.multiplicities[j]))

        res = ElementarySpace.from_sector_decomposition(
            sp1.symmetry, sectors, mults, is_dual=is_dual, unique_sectors=True
        )
        # from_sector_decomposition potentially introduces a meaningless basis_perm,
        # which we want to ignore here.
        # OPTIMIZE (JU) then dont compute it in the first place?
        res._basis_perm = None
        res._inverse_basis_perm = None
        return res

    @classmethod
    def from_null_space(cls, symmetry: Symmetry, is_dual: bool = False) -> ElementarySpace:
        """The zero-dimensional space, i.e. the span of the empty set."""
        return cls(
            symmetry=symmetry,
            defining_sectors=symmetry.empty_sector_array,
            multiplicities=np.zeros(0, int),
            is_dual=is_dual,
        )

    @classmethod
    def from_defining_sectors(
        cls,
        symmetry: Symmetry,
        defining_sectors: SectorArray,
        multiplicities: Sequence[int] = None,
        is_dual: bool = False,
        basis_perm: ndarray = None,
        unique_sectors: bool = False,
        return_sorting_perm: bool = False,
    ) -> ElementarySpace | tuple[ElementarySpace, ndarray]:
        """Similar to the constructor, but with fewer requirements.

        .. note ::
            Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
            only once to mean its entire multiplet of basis states. The Hilbert Space of a spin-1/2
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_defining_sectors(su2, [spin_half])``. In the former case
            we need to list the same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        defining_sectors: 2D array_like of int
            Like the :attr:`defining_sectors` attribute, but can be in any order and may contain
            duplicates (see `unique_sectors`).
        multiplicities: 1D array_like of int, optional
            How often each of the `defining_sectors` appears. A 1D array of positive integers with
            axis [s]. ``defining_sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
            If not given, a multiplicity ``1`` is assumed for all `defining_sectors`.
        is_dual: bool
            If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
            Note that this changes the meaning of the `defining_sectors`.
        basis_perm: ndarray, optional
            The permutation from the desired public basis to the basis described by
            `defining_sectors` and `multiplicities`.
        unique_sectors: bool
            If ``True``, the `sectors` are assumed to be duplicate-free.
        return_sorting_perm: bool
            If ``True``, the permutation ``np.lexsort(sectors.T)`` is returned too.

        Returns
        -------
        space: ElementarySpace
            The new space
        sector_sort: 1D array, optional
            Only ``if return_sorting_perm``. The permutation that sorts the `defining_sectors`.

        """
        defining_sectors = np.asarray(defining_sectors, dtype=int)
        assert defining_sectors.ndim == 2 and defining_sectors.shape[1] == symmetry.sector_ind_len
        if multiplicities is None:
            multiplicities = np.ones((len(defining_sectors),), dtype=int)
        else:
            multiplicities = np.asarray(multiplicities, dtype=int)
            assert multiplicities.shape == ((len(defining_sectors),))

        # sort sectors
        if symmetry.can_be_dropped:
            num_states = symmetry.batch_sector_dim(defining_sectors) * multiplicities
            basis_slices = np.concatenate([[0], np.cumsum(num_states)], axis=0)
            defining_sectors, multiplicities, sort = _sort_sectors(defining_sectors, multiplicities)
            if len(defining_sectors) == 0:
                basis_perm = np.zeros(0, int)
            else:
                if basis_perm is None:
                    basis_perm = np.arange(np.sum(num_states))
                basis_perm = np.concatenate([basis_perm[basis_slices[i] : basis_slices[i + 1]] for i in sort])
        else:
            defining_sectors, multiplicities, sort = _sort_sectors(defining_sectors, multiplicities)
            assert basis_perm is None
        # combine duplicate sectors (does not affect basis_perm)
        if not unique_sectors:
            mult_slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
            diffs = find_row_differences(defining_sectors, include_len=True)
            # the convention is that for sectors with dim > 1, all copies of the first
            # state appear, then all copies of the second state, etc. At this point,
            # this order is not yet fully respected
            if basis_perm is not None and not symmetry.is_abelian:
                # updated basis_slices after sorting defining_sectors
                num_states = symmetry.batch_sector_dim(defining_sectors) * multiplicities
                basis_slices = np.concatenate([[0], np.cumsum(num_states)], axis=0)
                for i in range(len(diffs) - 1):
                    sector_dim = symmetry.sector_dim(defining_sectors[diffs[i]])
                    if sector_dim == 1:
                        continue
                    mults = multiplicities[diffs[i] : diffs[i + 1]]
                    offsets = np.concatenate([[0], np.cumsum(mults * sector_dim)])
                    sector_basis_perm = basis_perm[basis_slices[diffs[i]] : basis_slices[diffs[i + 1]]]
                    # take the basis_perm associated with the first states and make them contiguous,
                    # then go to the second state, etc.
                    new_perm = [
                        sector_basis_perm[offsets[j] + k * mult : offsets[j] + (k + 1) * mult]
                        for k in range(sector_dim)
                        for j, mult in enumerate(mults)
                    ]
                    new_perm = np.concatenate(new_perm)
                    basis_perm[basis_slices[diffs[i]] : basis_slices[diffs[i + 1]]] = new_perm

            multiplicities = mult_slices[diffs[1:]] - mult_slices[diffs[:-1]]
            defining_sectors = defining_sectors[diffs[:-1]]  # [:-1] to exclude len
        res = cls(
            symmetry=symmetry,
            defining_sectors=defining_sectors,
            multiplicities=multiplicities,
            is_dual=is_dual,
            basis_perm=basis_perm,
        )
        if return_sorting_perm:
            return res, sort
        return res

    @classmethod
    def from_sector_decomposition(
        cls,
        symmetry: Symmetry,
        sector_decomposition: SectorArray,
        multiplicities: Sequence[int] = None,
        is_dual: bool = False,
        basis_perm: ndarray = None,
        unique_sectors: bool = False,
    ) -> ElementarySpace:
        """Create a :class:`ElementarySpace` that has a given :attr:`sector_decomposition`.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        sector_decomposition: 2D array_like of int
            Like the :attr:`sector_decomposition` attribute, but can be in any order and may contain
            duplicates (see `unique_sectors`).
        multiplicities: 1D array_like of int, optional
            How often each of the `sector_decomposition` appears. A 1D array of positive integers
            with axis [s]. ``sector_decomposition[i_s, :]`` appears ``multiplicities[i_s]`` times.
            If not given, a multiplicity ``1`` is assumed for all `sector_decomposition`.
        is_dual: bool
            If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
        basis_perm: ndarray, optional
            The permutation from the desired public basis to the basis described by
            `sector_decomposition` and `multiplicities`.
        unique_sectors: bool
            If ``True``, the `sectors` are assumed to be duplicate-free.

        See Also
        --------
        from_defining_sectors

        """
        sector_decomposition = np.asarray(sector_decomposition, int)
        assert sector_decomposition.ndim == 2 and sector_decomposition.shape[1] == symmetry.sector_ind_len
        if is_dual:
            defining_sectors = symmetry.dual_sectors(sector_decomposition)
        else:
            defining_sectors = sector_decomposition
        return cls.from_defining_sectors(
            symmetry=symmetry,
            defining_sectors=defining_sectors,
            multiplicities=multiplicities,
            is_dual=is_dual,
            basis_perm=basis_perm,
            unique_sectors=unique_sectors,
        )

    @classmethod
    def from_trivial_sector(
        cls, dim: int = 1, symmetry: Symmetry = no_symmetry, is_dual: bool = False, basis_perm: ndarray = None
    ) -> ElementarySpace:
        """Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).

        Parameters
        ----------
        dim : int
            The dimension of the space.
        symmetry : :class:`~cyten.groups.Symmetry`
            The symmetry of the space.
        is_dual : bool
            If the space should be bra or a ket space.

        """
        if dim == 0:
            return cls.from_null_space(symmetry=symmetry, is_dual=is_dual)
        return cls(
            symmetry=symmetry,
            defining_sectors=symmetry.trivial_sector[None, :],
            multiplicities=[dim],
            is_dual=is_dual,
            basis_perm=basis_perm,
        )

    @property
    def sectors_of_basis(self):
        """The sector (from the :attr:`sector_decomposition`) of each basis vector."""
        if not self.symmetry.can_be_dropped:
            msg = f'sectors_of_basis is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        # build in internal basis, then permute
        res = np.zeros((self.dim, self.symmetry.sector_ind_len), dtype=int)
        for sect, slc in zip(self.sector_decomposition, self.slices):
            res[slice(*slc), :] = sect[None, :]
        return self.apply_basis_perm(res, inverse=True)

    def __repr__(self, show_symmetry: bool = True, one_line=False):
        ClsName = type(self).__name__
        indent = printoptions.indent * ' '

        # try to show everything, then less and less
        for full_sectors, summarized_sectors, symmetry in [
            (True, False, show_symmetry),
            (False, True, show_symmetry),
            (False, False, show_symmetry),
            (False, False, False),
        ]:
            if full_sectors and (3 * self.defining_sectors.size > printoptions.linewidth):
                # there is no chance to print all sectors in one line
                continue

            items = []

            if symmetry:
                items.append(f'symmetry={self.symmetry!r}')
            if full_sectors:
                def_sector_strs = [self.symmetry.sector_str(a) for a in self.defining_sectors]
                sector_dec_strs = [self.symmetry.sector_str(a) for a in self.sector_decomposition]
                items.append(f'defining_sectors={format_like_list(def_sector_strs)}')
                items.append(f'sector_decomposition={format_like_list(sector_dec_strs)}')
                items.append(f'multiplicities={format_like_list(self.multiplicities)}')
                if self._basis_perm is not None:
                    items.append(f'basis_perm={format_like_list(self._basis_perm)}')
            if summarized_sectors:
                items.append(f'num_sectors={self.num_sectors}')
                if self._basis_perm is not None:
                    items.append(f'basis_perm=[...]')
            items.append(f'is_dual={self.is_dual}')

            # try one line
            res = ClsName + '(' + ', '.join(items) + ')'
            if len(res) <= printoptions.linewidth:
                return res

            if not one_line:
                # try multi line
                items = [indent + i + ',' for i in items]
                maxlines_ok = len(items) + 2 <= printoptions.maxlines_spaces
                linewidth_ok = all(len(l) < printoptions.linewidth for l in items)
                if maxlines_ok and linewidth_ok:
                    return ClsName + '(\n' + '\n'.join(indent + i for i in items) + '\n)'

        raise RuntimeError  # one of the above returns should have triggered

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
        if not np.all(self.defining_sectors == other.defining_sectors):
            return False
        if (self._basis_perm is not None) or (other._basis_perm is not None):
            if not np.all(self.basis_perm == other.basis_perm):
                return False
        else:
            pass  # both permutations are trivial, thus equal
        return True

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        if bool(is_dual) == self.is_dual:
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

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, injective: bool = False) -> ElementarySpace:
        return ElementarySpace.from_defining_sectors(
            symmetry=symmetry,
            defining_sectors=sector_map(self.defining_sectors),
            multiplicities=self.multiplicities,
            is_dual=self.is_dual,
            basis_perm=self._basis_perm,
            unique_sectors=injective,
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
            basis_perm = np.concatenate([self.basis_perm] + [o.basis_perm + n for o, n in zip(others, offsets)])
        else:
            basis_perm = None
        return ElementarySpace.from_defining_sectors(
            symmetry=self.symmetry,
            defining_sectors=np.concatenate([self.defining_sectors, *(o.defining_sectors for o in others)]),
            multiplicities=np.concatenate([self.multiplicities, *(o.multiplicities for o in others)]),
            is_dual=self.is_dual,
            basis_perm=basis_perm,
        )

    def drop_symmetry(self, which: int | list[int] = None):
        which, remaining_symmetry = _parse_inputs_drop_symmetry(which, self.symmetry)
        if which is None:
            return ElementarySpace.from_trivial_sector(
                dim=self.dim, symmetry=remaining_symmetry, is_dual=self.is_dual, basis_perm=self._basis_perm
            )
        mask = np.ones((self.symmetry.sector_ind_len,), dtype=bool)
        for i in which:
            start, stop = self.symmetry.sector_slices[i : i + 2]
            mask[start:stop] = False
        return self.change_symmetry(symmetry=remaining_symmetry, sector_map=lambda sectors: sectors[:, mask])

    @property
    def dual(self) -> ElementarySpace:
        return ElementarySpace(
            self.symmetry,
            defining_sectors=self.defining_sectors,
            multiplicities=self.multiplicities,
            is_dual=not self.is_dual,
            basis_perm=self._basis_perm,
        )

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
            indicating that the `idx`-th basis element lives in ``self.sector_decomposition[sector_idx]``.
        multiplicity_idx : int
            The index "within the sector", in ``range(sector_dim * self.multiplicities[sector_index])``.

        """
        if not self.symmetry.can_be_dropped:
            msg = f'parse_index is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        idx = self.apply_basis_perm(idx, inverse=True, pre_compose=True)
        sector_idx = bisect.bisect(self.slices[:, 0], idx) - 1
        multiplicity_idx = idx - self.slices[sector_idx, 0]
        return sector_idx, multiplicity_idx

    def idx_to_sector(self, idx: int) -> Sector:
        sector_idx, _ = self.parse_index(idx)
        return self.sector_decomposition[sector_idx]

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
        blockmask = self.apply_basis_perm(blockmask)
        sectors = []
        mults = []
        for a, d_a, slc in zip(self.defining_sectors, self.sector_dims, self.slices):
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
        return ElementarySpace(
            symmetry=self.symmetry,
            defining_sectors=sectors,
            multiplicities=mults,
            is_dual=self.is_dual,
            basis_perm=basis_perm,
        )

    def with_opposite_duality(self):
        """A space isomorphic to self with opposite ``is_dual`` attribute."""
        if self.is_dual:
            # already have the self.symmetry.dual_sectors(self.defining_sectors)
            dual_defining_sectors = self.sector_decomposition
        else:
            dual_defining_sectors = self.symmetry.dual_sectors(self.defining_sectors)
        # note: dual_defining_sectors are not sorted, but they are unique.
        return ElementarySpace.from_defining_sectors(
            symmetry=self.symmetry,
            defining_sectors=dual_defining_sectors,
            multiplicities=self.multiplicities,
            is_dual=not self.is_dual,
            basis_perm=self._basis_perm,
            unique_sectors=True,
        )

    def with_is_dual(self, is_dual: bool) -> ElementarySpace:
        """A space isomorphic to self with given ``is_dual`` attribute."""
        if is_dual == self.is_dual:
            return self
        return self.with_opposite_duality()

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.defining_sectors, subpath + 'defining_sectors')
        hdf5_saver.save(self.sector_decomposition, subpath + 'sector_decomposition')
        hdf5_saver.save(self.sector_order, subpath + 'sector_order')
        hdf5_saver.save(self._basis_perm, subpath + '_basis_perm')
        hdf5_saver.save(self._inverse_basis_perm, subpath + '_inverse_basis_perm')
        hdf5_saver.save(self.multiplicities, subpath + 'multiplicities')
        hdf5_saver.save(self.symmetry, subpath + 'symmetry')
        hdf5_saver.save(self.dim, subpath + 'dim')
        hdf5_saver.save(self.num_sectors, subpath + 'num_sectors')
        hdf5_saver.save(self.slices, subpath + 'slices')
        hdf5_saver.save(self.sector_dims, subpath + 'sector_dims')

        h5gr.attrs['is_dual'] = self.is_dual

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.defining_sectors = hdf5_loader.load(subpath + 'defining_sectors')
        obj.sector_decomposition = hdf5_loader.load(subpath + 'sector_decomposition')
        obj.sector_order = hdf5_loader.load(subpath + 'sector_order')
        obj._basis_perm = hdf5_loader.load(subpath + '_basis_perm')
        obj._inverse_basis_perm = hdf5_loader.load(subpath + '_inverse_basis_perm')
        obj.multiplicities = hdf5_loader.load(subpath + 'multiplicities')
        obj.symmetry = hdf5_loader.load(subpath + 'symmetry')
        obj.dim = hdf5_loader.load(subpath + 'dim')
        obj.num_sectors = hdf5_loader.load(subpath + 'num_sectors')
        obj.slices = hdf5_loader.load(subpath + 'slices')
        obj.sector_dims = hdf5_loader.load(subpath + 'sector_dims')
        obj.is_dual = hdf5_loader.get_attr(h5gr, 'is_dual')

        return obj


class TensorProduct(Space):
    r"""Represents a tensor product of :class:`Spaces`\ s, e.g. the (co-)domain of a tensor.

    Attributes
    ----------
    factors : list[Space | LegPipe]
        The factors in the tensor product, e.g. some of the legs of a tensor.
    num_factors : int
        The number of :attr:`factors`.
    _sector_decomposition, _multiplicities
        If the sectors, multiplicities are already known, recomputation can be skipped.
        Warning: If given, they are not checked for correctness!

    See Also
    --------
    LegPipe
        A :class:`LegPipe` has the same mathematical idea as the :class:`TensorProduct`.
        There are two main differences:
        Firstly, for a :class:`TensorProduct`, we compute the :attr:`sector_decomposition`, which
        we do not do for a :class`LegPipe`. This is reflected in the fact that only
        :class:`TensorProduct`s are :class:`Space`s, while :class:`LegPipe`s are not.
        Secondly, we only keep track of duality with an explicit flag for :class:`Leg`s, to have
        arrows on our tensor legs. A :class:`TensorProduct` has no ``is_dual`` attribute.

    """

    def __init__(
        self,
        factors: list[Space | LegPipe],
        symmetry: Symmetry = None,
        _sector_decomposition: SectorArray = None,
        _multiplicities: SectorArray = None,
    ):
        self.num_factors = num_factors = len(factors)
        if symmetry is None:
            if num_factors == 0:
                raise ValueError('If spaces is empty, the symmetry arg is required.')
            symmetry = factors[0].symmetry
        if not all(sp.symmetry == symmetry for sp in factors):
            raise SymmetryError('Incompatible symmetries.')
        self.symmetry = symmetry  # need to set this early, for use in _calc_sectors
        self.factors = factors[:]
        if _sector_decomposition is None or _multiplicities is None:
            if _sector_decomposition is not None or _multiplicities is not None:
                msg = 'Need both _sectors and _multiplicities to skip recomputation. Got just one.'
                warnings.warn(msg)
            _sector_decomposition, _multiplicities = self._calc_sectors(factors)
        Space.__init__(
            self,
            symmetry=symmetry,
            sector_decomposition=_sector_decomposition,
            multiplicities=_multiplicities,
            sector_order='sorted',
        )

    def test_sanity(self):
        """Perform sanity checks."""
        assert len(self.factors) == self.num_factors
        for sp in self.factors:
            sp.test_sanity()
        Space.test_sanity(self)

    # CLASSMETHODS

    @classmethod
    def from_partial_products(cls, *factors: TensorProduct) -> TensorProduct:
        r"""Form the :class:`TensorProduct` of all :attr:`spaces` from partial products.

        The result has as :attr:`spaces` all those spaces that appear on the `factors`.
        I.e. we form :math:`V_1 \otimes V_2 \otimes W_1 \otimes W_2 \dots` from
        :math:`V_1 \otimes V_2` and :math:`W_1 \otimes W_2 \dots`.
        """
        spaces = factors[0].factors[:]
        symmetry = factors[0].symmetry
        for f in factors[1:]:
            spaces.extend(f.factors)
            assert f.symmetry == symmetry, 'Mismatched symmetries'
        isomorphic = TensorProduct(factors=factors, symmetry=symmetry)
        # forming isomorphic performs the fusion more efficiently, since it uses the partially
        # fused [f.sectors for f in factors] instead of the flat [s.factors for f in factors for s in f.factors]
        return TensorProduct(
            factors=spaces,
            symmetry=symmetry,
            _sector_decomposition=isomorphic.sector_decomposition,
            _multiplicities=isomorphic.multiplicities,
        )

    # PROPERTIES

    @property
    def dual(self):
        sectors = self.symmetry.dual_sectors(self.sector_decomposition)
        sectors, mults, _ = _sort_sectors(sectors, self.multiplicities)
        return TensorProduct(
            [sp.dual for sp in reversed(self.factors)],
            symmetry=self.symmetry,
            _sector_decomposition=sectors,
            _multiplicities=mults,
        )

    # METHODS

    def block_size(self, coupled: Sector | int) -> int:
        """The size of a block.

        Parameters
        ----------
        coupled : Sector or int
            Specify the coupled sector, either directly as a sector or as an integer, which
            is interpreted as an index, i.e. is equivalent to the sector
            ``self.sector_decomposition[coupled]``.

        """
        if isinstance(coupled, int):
            return self.multiplicities[coupled]
        return self.sector_multiplicity(coupled)

    def change_symmetry(self, symmetry, sector_map, injective=False):
        sectors = sector_map(self.sector_decomposition)
        multiplicities = self.multiplicities
        if not injective:
            sectors, multiplicities, _ = _unique_sorted_sectors(sectors, multiplicities)
        else:
            sectors, multiplicities, _ = _sort_sectors(sectors, multiplicities)
        return TensorProduct(
            [space.change_symmetry(symmetry, sector_map, injective) for space in self.factors],
            symmetry=self.symmetry,
            _sector_decomposition=sectors,
            _multiplicities=multiplicities,
        )

    def drop_symmetry(self, which=None):
        which, remaining_symmetry = _parse_inputs_drop_symmetry(which, self.symmetry)
        if which is None:
            sectors = self.symmetry.trivial_sector[None, :]
            multiplicities = [self.dim]
        else:
            mask = np.ones((self.symmetry.sector_ind_len,), dtype=bool)
            for i in which:
                start, stop = self.symmetry.sector_slices[i : i + 2]
                mask[start:stop] = False
            sectors = self.sector_decomposition[mask, :]
            multiplicities = self.multiplicities
            sectors, multiplicities, _ = _unique_sorted_sectors(sectors, multiplicities)
        return TensorProduct(
            [space.drop_symmetry(which) for space in self.factors],
            symmetry=remaining_symmetry,
            _sector_decomposition=sectors,
            _multiplicities=multiplicities,
        )

    @property
    def has_pipes(self) -> bool:
        """Is any of the :attr:`factors` a pipe?"""
        return any(isinstance(l, LegPipe) for l in self.factors)

    @property
    def flat_legs(self) -> list[ElementarySpace]:
        """Flatten until there are no more pipes.

        See Also
        --------
        flat_spaces : Keeps :class:`AbelianLegPipes` nested.

        """
        return list(it.chain.from_iterable(l.flat_legs for l in self.factors))

    @property
    def flat_spaces(self) -> list[ElementarySpace]:
        """Flatten until we get spaces.

        See Also
        --------
        flat_legs : Also flattens :class:`AbelianLegPipes`.

        """
        return list(it.chain.from_iterable(l.flat_spaces for l in self.factors))

    @property
    def num_flat_legs(self) -> int:
        """The number of :attr:`flat_legs`."""
        return sum(l.num_flat_legs for l in self.factors)

    def flat_legs_nesting(self) -> list[list[int]]:
        """The indices into :attr:`flat_legs`, that combine to each :attr:`factor`."""
        i = 0
        res = []
        for l in self.factors:
            num = l.num_flat_legs
            res.append([*range(i, i + num)])
            i += num
        return res

    def flat_leg_idcs(self, i: int) -> list[int]:
        """All indices into the :meth:`flat_legs` that the leg ``factors[i]`` flattens to."""
        i = to_valid_idx(i, self.num_factors)
        start = sum(l.num_flat_legs for l in self.factors[:i])
        num = self.factors[i].num_flat_legs
        return list(range(start, start + num))

    def forest_block_size(self, uncoupled: tuple[Sector], coupled: Sector) -> int:
        """The size of a forest-block"""
        # OPTIMIZE ?
        num_trees = len(fusion_trees(self.symmetry, uncoupled, coupled))
        return num_trees * self.tree_block_size(uncoupled)

    def forest_block_slice(self, uncoupled: tuple[Sector], coupled: Sector) -> slice:
        """The range of indices of a forest-block within its block, as a slice."""
        offset = 0
        for unc, mults in self.iter_uncoupled():
            if all(np.all(a == b) for a, b in zip(unc, uncoupled)):
                break
            tree_block_size = np.prod(mults)
            forest_block_size = len(fusion_trees(self.symmetry, unc, coupled)) * tree_block_size
            offset += forest_block_size
        else:  # no break occurred
            raise ValueError('Uncoupled sectors incompatible')
        size = self.forest_block_size(uncoupled, coupled)
        return slice(offset, offset + size)

    def insert_multiply(self, other: Space, pos: int) -> TensorProduct:
        """Insert a new space into the product at position `pos`."""
        isomorphic = TensorProduct([self, other])
        return TensorProduct(
            self.factors[:pos] + [other] + self.factors[pos:],
            symmetry=self.symmetry,
            _sector_decomposition=isomorphic.sector_decomposition,
            _multiplicities=isomorphic.multiplicities,
        )

    def iter_tree_blocks(
        self, coupled: Sequence[Sector]
    ) -> Generator[tuple[FusionTree, slice, np.ndarray, int], None, None]:
        """Iterate over tree blocks. Helper function for :class:`FusionTreeBackend`.

        See :ref:`fusion_tree_backend__blocks` for definitions of blocks and tree blocks.

        Yields
        ------
        tree : FusionTree
            A fusion tree whose uncoupled sectors are consistent with `self` and whose
            coupled sector is ``coupled[i]``
        slc : slice
            The slice of the tree-block associated with `tree` in its block.
        mults : 1D array of int
            The multiplicities of the uncoupled sectors of `tree` within their ``self.factor``.
        i : int
            The index of the current coupled sector in `coupled`

        See Also
        --------
        iter_forest_blocks
        iter_uncoupled

        """
        # OPTIMIZE some users in FTBackend ignore some of the yielded values.
        #          is that ok performance wise or should we have special case iterators?
        are_dual = [sp.is_dual for sp in self.flat_legs]
        for i, c in enumerate(coupled):
            start = 0  # start index of the current tree block within the block
            for uncoupled, mults in self.iter_uncoupled():
                tree_block_size = prod(mults)
                for tree in fusion_trees(self.symmetry, uncoupled, c, are_dual):
                    yield tree, slice(start, start + tree_block_size), mults, i
                    start += tree_block_size

    def iter_forest_blocks(self, coupled: Sequence[Sector]) -> Generator[tuple[tuple[Sector], slice, int], None, None]:
        """Iterate over forest blocks. Helper function for :class:`FusionTreeBackend`.

        See :ref:`fusion_tree_backend__blocks` for definitions of blocks and forest blocks.

        Yields
        ------
        uncoupled : tuple of Sector
            A tuple of uncoupled sectors that can fuse to a coupled sector ``coupled[i]``
        slc : slice
            The slice of the tree-block associated with `tree` in its block.
        i : int
            The index of the current coupled sector in `coupled`

        See Also
        --------
        iter_tree_blocks
        iter_uncoupled

        """
        for i, c in enumerate(coupled):
            start = 0
            for uncoupled, mults in self.iter_uncoupled():
                tree_block_size = np.prod(mults)
                num_trees = len(fusion_trees(self.symmetry, uncoupled, c))
                forest_block_width = num_trees * tree_block_size
                if forest_block_width == 0:
                    continue
                slc = slice(start, start + forest_block_width)
                yield uncoupled, slc, i
                start += forest_block_width

    def iter_uncoupled(
        self, yield_slices: bool = False
    ) -> Generator[tuple[SectorArray, np.ndarray] | tuple[SectorArray, np.ndarray, list[slice]], None, None]:
        """Iterate over all combinations of sectors from the :attr:`flat_legs`.

        Yields
        ------
        uncoupled : 2D array of int
            A combination of uncoupled sectors, where
            ``uncoupled[i] == self.flat_legs[i].sector_decomposition[some_idx]``.
        multiplicities : 1D array of int
            The corresponding multiplicities
            ``multiplicities[i] == self.flat_legs[i].multiplicities[some_idx]``.
        slices : list of slice, optional
            Only if ``yield_slices``, the corresponding entry of :attr:`Space.slices`, as a slice.
            I.e. ``slices[i] == slice(*self.flat_legs[i].slices[some_idx])``.

        Notes
        -----
        For a TensorProduct of zero spaces, i.e. with ``num_factors == 0``,
        we *do* yield once, where the yielded arrays are empty (e.g. ``len(uncoupled) == 0``).

        """
        flat_legs = self.flat_legs
        for idcs in it.product(*(range(s.num_sectors) for s in flat_legs)):
            a = np.array([flat_legs[n].sector_decomposition[i] for n, i in enumerate(idcs)], int)
            m = np.array([flat_legs[n].multiplicities[i] for n, i in enumerate(idcs)], int)
            if yield_slices:
                slcs = [slice(*flat_legs[n].slices[i]) for n, i in enumerate(idcs)]
                yield a, m, slcs
            else:
                yield a, m

    def left_multiply(self, other: Space) -> TensorProduct:
        """Add a new factor at the left / beginning of the spaces"""
        return self.insert_multiply(other, 0)

    def permuted(self, perm: Sequence[int]) -> TensorProduct:
        """A product of the same :attr:`factors` in a different order."""
        assert len(perm) == self.num_factors
        assert set(perm) == set(range(self.num_factors))
        return TensorProduct(
            factors=[self.factors[i] for i in perm],
            symmetry=self.symmetry,
            _sector_decomposition=self.sector_decomposition,
            _multiplicities=self.multiplicities,
        )

    def right_multiply(self, other: Space) -> TensorProduct:
        """Add a new factor at the right / end of the spaces"""
        return self.insert_multiply(other, -1)

    def tree_block_size(space: TensorProduct, uncoupled: tuple[Sector]) -> int:
        """The size of a tree-block"""
        # OPTIMIZE ?
        return prod(s.sector_multiplicity(a) for s, a in zip(space.flat_legs, uncoupled))

    def tree_block_slice(self, tree: FusionTree) -> slice:
        """The range of indices of a tree-block within its block, as a slice."""
        # OPTIMIZE ?
        start = 0
        for unc, mults in self.iter_uncoupled():
            tree_block_size = np.prod(mults)
            if all(np.all(a == b) for a, b in zip(unc, tree.uncoupled)):
                break
            num_trees = len(fusion_trees(self.symmetry, unc, tree.coupled))
            start += num_trees * tree_block_size
        else:  # no break occurred
            raise ValueError('Uncoupled sectors incompatible')
        tree_idx = fusion_trees(self.symmetry, tree.uncoupled, tree.coupled, tree.are_dual).index(tree)
        start += tree_block_size * tree_idx
        return slice(start, start + tree_block_size)

    # DUNDERS AND INTERNAL HELPERS

    def __eq__(self, other):
        if not isinstance(other, TensorProduct):
            return NotImplemented
        if self.num_factors != other.num_factors:
            return False
        if self.symmetry != other.symmetry:
            return False
        return all(s1 == s2 for s1, s2 in zip(self.factors, other.factors))

    def __getitem__(self, idx):
        return self.factors[idx]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return self.num_factors

    def __repr__(self, show_symmetry: bool = True, one_line=False):
        ClsName = type(self).__name__
        indent = printoptions.indent * ' '

        for mode in [
            (True, False, True, show_symmetry),
            (False, True, True, show_symmetry),
            (True, False, False, show_symmetry),
            (False, True, False, show_symmetry),
            (False, False, False, show_symmetry),
            (False, False, False, False),
        ]:
            full_sectors, summarized_sectors, show_all_factors, symmetry = mode

            if full_sectors and (3 * self.sector_decomposition.size > printoptions.linewidth):
                # there is no chance to print all sectors in one line
                continue

            # populate two lists; one intended for single line, one for multiline
            one_line_items = []
            lines = [f'{ClsName}(']
            if symmetry:
                one_line_items.append(f'symmetry={self.symmetry!r}')
                lines.append(f'{indent}symmetry={self.symmetry!r},')
            if show_all_factors:
                reprs = [f.__repr__(show_symmetry=False, one_line=True) for f in self.factors]
                one_line_items.append(f'factors=[{", ".join(reprs)}]')
                lines.append(f'{indent}factors=[')
                for r in reprs:
                    lines.append(f'{indent}{indent}{r},')
                lines.append(f'{indent}],')
            else:
                one_line_items.append(f'num_factors={self.num_factors}')
                lines.append(f'{indent}num_factors={self.num_factors},')
            if full_sectors:
                sector_strs = [self.symmetry.sector_str(a) for a in self.sector_decomposition]
                new_items = [
                    f'sector_decomposition={format_like_list(sector_strs)}',
                    f'multiplicities={format_like_list(self.multiplicities)}',
                ]
                one_line_items.extend(new_items)
                lines.extend(indent + i + ',' for i in new_items)
            if summarized_sectors:
                one_line_items.append(f'num_sectors={self.num_sectors}')
                lines.append(f'{indent}num_sectors={self.num_sectors},')
            lines.append(')')

            # try one line
            res = ClsName + '(' + ', '.join(one_line_items) + ')'
            if len(res) <= printoptions.linewidth:
                return res

            if not one_line:
                # try multi line
                maxlines_ok = len(lines) <= printoptions.maxlines_spaces
                linewidth_ok = all(len(l) < printoptions.linewidth for l in lines)
                if maxlines_ok and linewidth_ok:
                    return '\n'.join(lines)

        raise RuntimeError  # one of the above returns should have triggered

    def _calc_sectors(self, factors: list[Space | Leg]) -> tuple[SectorArray, ndarray]:
        """Helper function for :meth:`__init__`"""
        # LegPipes do not have sectors -> flatten them for the purpose of calculating sectors
        factors = list(it.chain.from_iterable(l.flat_spaces for l in factors))
        if len(factors) == 0:
            return self.symmetry.trivial_sector[None, :], np.ones([1], int)

        # need the sector decomposition of each factor. easiest way: convert to Space
        # OPTIMIZE is this optimal? should we store the f.as_Space() for later use?
        factors = [f.as_Space() for f in factors]

        if len(factors) == 1:
            sectors = factors[0].sector_decomposition
            mults = factors[0].multiplicities
            if factors[0].sector_order == 'sorted':
                return sectors, mults
            perm = np.lexsort(sectors.T)
            return sectors[perm], mults[perm]

        if self.symmetry.is_abelian:
            grid = make_grid([space.num_sectors for space in factors], cstyle=False)
            sectors = self.symmetry.multiple_fusion_broadcast(
                *(sp.sector_decomposition[gr] for sp, gr in zip(factors, grid.T))
            )
            multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(factors, grid.T)], axis=0)
            sectors, multiplicities, _ = _unique_sorted_sectors(sectors, multiplicities)
            return sectors, multiplicities

        # define recursively
        sectors, mults = self._calc_sectors(factors[:-1])
        sector_arrays = []
        mult_arrays = []
        for s2, m2 in zip(factors[-1].sector_decomposition, factors[-1].multiplicities):
            for s1, m1 in zip(sectors, mults):
                new_sects = self.symmetry.fusion_outcomes(s1, s2)
                sector_arrays.append(new_sects)
                if self.symmetry.fusion_style <= FusionStyle.multiple_unique:
                    new_mults = m1 * m2 * np.ones(len(new_sects), dtype=int)
                else:
                    # OPTIMIZE support batched N symbol?
                    new_mults = m1 * m2 * np.array([self.symmetry._n_symbol(s1, s2, c) for c in new_sects], dtype=int)
                mult_arrays.append(new_mults)
        sectors, multiplicities, _ = _unique_sorted_sectors(
            np.concatenate(sector_arrays, axis=0), np.concatenate(mult_arrays, axis=0)
        )
        return sectors, multiplicities

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.factors, subpath + 'factors')
        hdf5_saver.save(self.slices, subpath + 'slices')
        hdf5_saver.save(self.symmetry, subpath + 'symmetry')
        hdf5_saver.save(self.num_sectors, subpath + 'num_sectors')
        hdf5_saver.save(self.num_factors, subpath + 'num_factors')
        hdf5_saver.save(self.sector_decomposition, subpath + 'sector_decomposition')
        hdf5_saver.save(self.sector_order, subpath + 'sector_order')
        hdf5_saver.save(self.dim, subpath + 'dim')
        hdf5_saver.save(self.multiplicities, subpath + 'multiplicities')
        hdf5_saver.save(self.sector_dims, subpath + 'sector_dims')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)

        hdf5_loader.memorize_load(h5gr, obj)

        obj.factors = hdf5_loader.load(subpath + 'factors')
        obj.slices = hdf5_loader.load(subpath + 'slices')
        obj.symmetry = hdf5_loader.load(subpath + 'symmetry')
        obj.num_sectors = hdf5_loader.load(subpath + 'num_sectors')
        obj.num_factors = hdf5_loader.load(subpath + 'num_factors')
        obj.sector_decomposition = hdf5_loader.load(subpath + 'sector_decomposition')
        obj.sector_order = hdf5_loader.load(subpath + 'sector_order')
        obj.dim = hdf5_loader.load(subpath + 'dim')
        obj.multiplicities = hdf5_loader.load(subpath + 'multiplicities')
        obj.sector_dims = hdf5_loader.load(subpath + 'sector_dims')

        return obj


class AbelianLegPipe(LegPipe, ElementarySpace):
    r"""Special case of a :class:`LegPipe` for abelian group symmetries.

    This class essentially exists to allow specialized handling of combined legs in the
    :class:`AbelianBackend`. For this backend, we want to treat combined legs, i.e. pipes, exactly
    the same as regular legs. This is why this class also inherits from :class:`ElementarySpace`,
    which are the "uncombined" legs. Crucially, this allows the pipe to have
    :attr:`defining_sectors` for the :attr:`cyten.backends.abelian.AbelianBackendData.block_inds`
    to point to, to have a well-behaved :attr:`is_dual` attribute and to have a :attr:`basis_perm`,
    which can account for the basis permutation that is induced by going from sectors of the
    individual legs to a sorted list of coupled sectors on the pipe.

    Attributes
    ----------
    legs:
        The individual legs that form this pipe, and that the pipe can be split into.
        In particular, these are such that the pipe, as an :class:`ElementarySpace`, is isomorphic
        to their tensor product ``TensorProduct(legs)``, i.e. has the same :attr:`sector_decomposition`.
    sector_strides : 1D numpy array of int
        Strides for the shape ``[leg.num_sectors for leg in self.legs]``. Is either C-style or
        F-style, depending on `combine_cstyle`. This allows one-to-one mapping between multi-indices
        (one block_ind per space) to a single index. Used in :meth:`AbelianBackend.combine_legs`.
    fusion_outcomes_sort : 1D numpy array of int
        The permutation that sorts the list of fusion outcomes.
        To calculate the :attr:`sector_decomposition` of the pipe, we go through all combinations
        of sectors from the :attr:`legs` in F-style order, i.e. varying sectors from the first leg
        the fastest. For each combination of sectors, we perform their fusion, which yields a single
        sector in the abelian case assumed here. The resulting list of fused sectors is in general
        neither sorted nor unique. This permutation (stable) sorts the resulting list.
        We use F-style to match the sorting convention of :attr:`block_ind_map`.
    block_ind_map_slices : 1D numpy array of int
        Slices for embedding the unique fused sectors in the sorted list of all fusion outcomes.
        Shape is ``(K,)`` where ``K == pipe.num_sectors + 1``.
        Fusing all sectors from the :attr:`sector_decomposition` of all legs and sorting the
        outcomes gives a list which contains (in general) duplicates.
        The slice ``block_ind_map_slices[n]:block_ind_map_slices[n + 1]`` within this sorted list
        contains the same entry, namely ``pipe.sector_decomposition[n]``.
        Used in :math:`AbelianBackend.split_legs`.
    block_ind_map : 2D numpy array of int
        Map for the embedding of uncoupled to coupled indices, see notes below.
        Shape is ``(M, N)`` where ``M`` is the number of combinations of sectors,
        i.e. ``M == prod(leg.num_sectors for leg in legs)`` and ``N == 3 + len(legs)``.

    Notes
    -----
    In ``numpy``, combining legs is usually done via ``np.reshape``.
    There, mapping indices :math:`i,j,... \rightarrow k` amounted to
    :math:`k = s_1*i + s_2*j + ...` for appropriate strides :math:`s_1,s_2`.

    In the symmetric case, however, we want to group and sort the :math:`k` by sector, so we must
    implicitly permute as well. The details of this grouping depend on if the pipe is in the domain
    or codomain. Let us assume that all legs are in the codomain first.
    The reordering is encoded in :attr:`block_ind_map` as follows.

    Each block index combination :math:`(i_1, ..., i_{nlegs})` of the ``nlegs=len(legs)``
    input :class:`ElementarySpace`s will end up getting placed in some slice :math:`a_j:a_{j+1}`
    of the resulting :class:`AbelianLegPipe`. Within this slice, the data is simply reshaped in
    usual row-major fashion ('C'-order), i.e., with strides :math:`s_1 > s_2 > ...` given by the
    block size, if the legs are in the codomain (for domain, see below).

    It will be a subslice of a new total block in the `AbelianLegPipe` labelled by block index
    :math:`J`. We fuse sectors according to the rule::

        pipe.sector_decomposition[J] == pipe.symmetry.multiple_fusion(
            *[l.sector_decomposition[i_l] for i_l, l in zip(block_index_combination, pipe.legs)]
        )

    Since many sector combinations can fuse to the same coupled sector,
    in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
    block :math:`J` in the `AbelianLegPipe`.

    The rows of :attr:`block_ind_map` are precisely the collections of
    ``[b_{J,k}, b_{J,k+1}, i_1, . . . , i_{nlegs}, J]``.
    Here, :math:`b_k:b_{k+1}` denotes the slice of this block index combination *within*
    the total block `J`, i.e., ``b_{J,k} = a_j - self.slices[J]``.

    The rows of :attr:`block_ind_map` are sorted first by ``J``, then the ``i``.
    If the legs are in the codomain (for domain, see below) we sort by the ``i`` *in C-style order*,
    i.e. second by ``i_1``, then ``i_2``, ... and finally by ``i_{nlegs}``.
    In particular, that means that ``block_ind_map[:, [-2, -3, ..., 3, 2, -1]]``
    is ``np.lexsort( .T)``ed.
    Each ``J`` will have multiple rows, and the order in which they are stored in :attr:`block_inds`
    is the order the data is stored in the actual tensor.
    Thus, ``block_ind_map`` might look like ::

        [ ...,
        [ b_{J,k},   b_{J,k+1},  i_1,    ..., i_{nlegs}   , J,   ],
        [ b_{J,k+1}, b_{J,k+2},  i'_1,   ..., i'_{nlegs}  , J,   ],
        [ 0,         b_{J+1,1},  i''_1,  ..., i''_{nlegs} , J + 1],
        [ b_{J+1,1}, b_{J+1,2},  i'''_1, ..., i'''_{nlegs}, J + 1],
        ...]

    Now for pipes in the domain, the order of the :attr:`legs` is reversed compared to their
    appearance in ``pre_combine_tensor.legs``. Therefore, to stay consistent with the C-style order
    of combinations that pipes in the codomain have, we flip the style and sort by F-style instead.
    For reshaping the actual data, this has no effect, since data is relative to the
    ``tensor.legs`` and unaffected by this reversed leg order. For the sector combinations, however,
    we now take F-style combinations, such that e.g. the ``i`` in :attr:`block_ind_map` are
    sorted by F-style, such that the whole :attr:`block_ind_map` is ``np.lexsort( .T)``\ -ed.
    This also affects the :attr:`basis_perm`, since we need to use F-style combinations when arguing
    about the order of public or internal basis of the pipe.

    """

    def __init__(self, legs: Sequence[ElementarySpace], is_dual: bool = False, combine_cstyle: bool = True):
        LegPipe.__init__(self, legs=legs, is_dual=is_dual, combine_cstyle=combine_cstyle)
        assert self.symmetry.is_abelian and self.symmetry.can_be_dropped
        sectors, mults = self._calc_sectors()  # also sets some attributes
        basis_perm = self._calc_basis_perm(mults)
        ElementarySpace.__init__(
            self,
            symmetry=self.symmetry,
            defining_sectors=sectors,
            multiplicities=mults,
            is_dual=is_dual,
            basis_perm=basis_perm,
        )

    def test_sanity(self):
        """Perform sanity checks."""
        for l in self.legs:
            assert isinstance(l, ElementarySpace)
            if isinstance(l, LegPipe):
                assert isinstance(l, AbelianLegPipe)
            l.test_sanity()
        # check self.sector_strides
        assert self.sector_strides.shape == (self.num_legs,)
        expect = make_stride([leg.num_sectors for leg in self.legs], cstyle=self.combine_cstyle)
        assert np.all(self.sector_strides == expect)
        # check block_ind_map_slices
        # note: we do not check for full correctness, just for consistency as slices
        assert self.block_ind_map_slices.shape == (self.num_sectors + 1,)
        assert self.block_ind_map_slices[0] == 0
        assert self.block_ind_map_slices[-1] == np.prod([l.num_sectors for l in self.legs])
        assert np.all(self.block_ind_map_slices[1:] >= self.block_ind_map_slices[:-1])
        # check block_ind_map
        M, N = self.block_ind_map.shape
        assert M == np.prod([leg.num_sectors for leg in self.legs])
        assert N == 3 + self.num_legs
        if self.combine_cstyle:
            # C style grid -> lexsorted after reversing column order (see notes)
            should_be_sorted = self.block_ind_map[:, [*reversed(range(2, N - 1)), -1]]
        else:
            # F style grid -> is lexsorted
            should_be_sorted = self.block_ind_map[:, 2:]
        assert np.all(np.lexsort(should_be_sorted.T) == np.arange(len(should_be_sorted)))
        for i, (b1, b2, *idcs, J) in enumerate(self.block_ind_map):
            if i > 0 and J == self.block_ind_map[i - 1][-1]:
                assert b1 == self.block_ind_map[i - 1][1]
            else:
                assert b1 == 0
            sectors = (leg.sector_decomposition[i] for i, leg in zip(idcs, self.legs))
            fused = self.symmetry.multiple_fusion(*sectors)
            assert np.all(fused == self.sector_decomposition[J])
        # call to super class(es)
        LegPipe.test_sanity(self)
        ElementarySpace.test_sanity(self)

    def as_Space(self):
        return self

    def as_ElementarySpace(self, is_dual: bool = False):
        return self.with_is_dual(is_dual=is_dual)

    @property
    def dual(self) -> AbelianLegPipe:
        return AbelianLegPipe(
            [l.dual for l in reversed(self.legs)], is_dual=not self.is_dual, combine_cstyle=not self.combine_cstyle
        )

    @property
    def is_trivial(self) -> bool:
        return ElementarySpace.is_trivial.fget(self)

    @property
    def flat_spaces(self) -> list[ElementarySpace]:
        # Unlike the plain LegPipe, we do not need to flatten AbelianLegPipes, if we just
        # want to flatten until we get spaces
        return [self]

    @classmethod
    def from_basis(cls, *a, **kw):
        raise TypeError('from_basis is not supported for AbelianLegPipe')

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions):
        assert all(isinstance(i, AbelianLegPipe) for i in independent_descriptions)
        is_dual = independent_descriptions[0].is_dual
        assert all(i.is_dual == is_dual for i in independent_descriptions[1:])
        num_legs = independent_descriptions[0].num_legs
        assert all(i.num_legs == num_legs for i in independent_descriptions[1:])
        legs = [
            i_legs[0].from_independent_symmetries(i_legs) for i_legs in zip(*(i.legs for i in independent_descriptions))
        ]
        return cls(legs, is_dual=is_dual)

    @classmethod
    def from_null_space(cls, symmetry, is_dual=False):
        raise TypeError('from_null_space is not supported for AbelianLegPipe')

    @classmethod
    def from_defining_sectors(cls, *a, **kw):
        raise TypeError('from_defining_sectors is not supported for AbelianLegPipe')

    @classmethod
    def from_trivial_sector(cls, *a, **kw):
        raise TypeError('from_trivial_sector is not supported for AbelianLegPipe')

    def change_symmetry(self, symmetry, sector_map, injective=False):
        legs = [l.change_symmetry(symmetry, sector_map, injective) for l in self.legs]
        return AbelianLegPipe(legs, is_dual=self.is_dual, combine_cstyle=self.combine_cstyle)

    def drop_symmetry(self, which: int | list[int] = None):
        # OPTIMIZE can we avoid recomputation of fusion?
        legs = [l.drop_symmetry(which) for l in self.legs]
        return AbelianLegPipe(legs, is_dual=self.is_dual, combine_cstyle=self.combine_cstyle)

    def set_basis_perm(
        self, basis_perm: Sequence[int] | None = UNSPECIFIED, inverse_basis_perm: Sequence[int] | None = UNSPECIFIED
    ):
        msg = f'Can not set basis_perm for {type(self).__name__}.'
        raise TypeError(msg)

    def take_slice(self, blockmask):
        msg = (
            'Using `AbelianLegPipe.take_slice` loses the product (pipe) structure and results in '
            'a plain ElementarySpace. Explicitly convert using `as_ElementarySpace` to suppress '
            'this warning.'
        )
        warnings.warn(msg, stacklevel=2)
        return self.as_ElementarySpace(is_dual=self.is_dual).take_slice(blockmask)

    def with_opposite_duality(self):
        return AbelianLegPipe(legs=self.legs, is_dual=not self.is_dual, combine_cstyle=self.combine_cstyle)

    def __eq__(self, other):
        res = LegPipe.__eq__(self, other)
        if res is NotImplemented:
            return res
        if not res:
            return False
        if self.combine_cstyle != other.combine_cstyle:
            return False
        return True

    def __repr__(self, show_symmetry: bool = True, one_line=False):
        ClsName = type(self).__name__
        indent = printoptions.indent * ' '

        for mode in [
            (0, 0, False, show_symmetry),
            (0, 0, True, show_symmetry),
            (0, 1, True, show_symmetry),
            (0, 2, True, show_symmetry),
            (1, 2, True, show_symmetry),
            (2, 2, True, show_symmetry),
            (2, 2, True, False),
        ]:
            sector_mode, child_mode, summarize_basis_perm, symmetry = mode
            # sector_mode:  0=show full arrays , 1=show only nums, 2=dont show
            # child_mode: 0=show full , 1=force one-line each, 2=show only num
            # summarize_basis_perm: bool

            if (sector_mode == 0) and (3 * self.sector_decomposition.size > printoptions.linewidth):
                # there is no chance to print all sectors in one line
                continue

            # populate two lists; one intended for single line, one for multiline
            # this is because lines behaves differently when dealing with the children / self.legs
            one_line_items = []
            lines = [f'{ClsName}(']

            if symmetry:
                one_line_items.append(f'symmetry={self.symmetry!r}')
                lines.append(f'{indent}symmetry={self.symmetry!r},')

            if child_mode < 2:
                reprs = [f.__repr__(show_symmetry=False, one_line=child_mode > 0) for f in self.legs]
                one_line_items.append(f'factors=[{", ".join(reprs)}]')
                lines.append(f'{indent}factors=[')
                for r in reprs:
                    lines.append(f'{indent}{indent}{r},')
                lines.append(f'{indent}],')
            elif child_mode == 2:
                one_line_items.append(f'num_legs={self.num_legs}')
                lines.append(f'{indent}num_legs={self.num_legs},')
            else:
                raise RuntimeError

            if sector_mode == 0:
                sector_dec_strs = [self.symmetry.sector_str(a) for a in self.sector_decomposition]
                def_sector_strs = [self.symmetry.sector_str(a) for a in self.defining_sectors]
                new_items = [
                    f'sector_decomposition={format_like_list(sector_dec_strs)}',
                    f'defining_sectors={format_like_list(def_sector_strs)}',
                    f'multiplicities={format_like_list(self.multiplicities)}',
                ]
                one_line_items.extend(new_items)
                lines.extend(indent + i + ',' for i in new_items)
            elif sector_mode == 1:
                one_line_items.append(f'num_sectors={self.num_sectors}')
                lines.append(f'{indent}num_sectors={self.num_sectors},')
            elif sector_mode == 2:
                pass  # dont add anything
            else:
                raise RuntimeError

            if self._basis_perm is not None:
                if summarize_basis_perm:
                    one_line_items.append('basis_perm=[...]')
                    lines.append(f'{indent}basis_perm=[...],')
                else:
                    one_line_items.append(f'basis_perm={format_like_list(self._basis_perm)}')
                    lines.append(f'{indent}basis_perm={format_like_list(self._basis_perm)},')

            one_line_items.append(f'is_dual={self.is_dual}')
            lines.append(f'{indent}is_dual={self.is_dual},')

            lines.append(')')

            # try one line
            res = ClsName + '(' + ', '.join(one_line_items) + ')'
            if len(res) <= printoptions.linewidth:
                return res

            if not one_line:
                # try multi line
                maxlines_ok = len(lines) <= printoptions.maxlines_spaces
                linewidth_ok = all(len(l) < printoptions.linewidth for l in lines)
                if maxlines_ok and linewidth_ok:
                    return '\n'.join(lines)

        raise RuntimeError  # one of the above returns should have triggered

    def _calc_sectors(self):
        """Helper function for :meth:`__init__`. Assumes ``LegPipe.__init__`` was called.

        Returns the defining_sectors and related multiplicities. Also sets the some attributes.
        """
        legs_num_sectors = tuple(l.num_sectors for l in self.legs)
        self.sector_strides = make_stride(legs_num_sectors, cstyle=self.combine_cstyle)

        grid = make_grid([leg.num_sectors for leg in self.legs], cstyle=self.combine_cstyle)

        nblocks = grid.shape[0]  # number of blocks in pipe = np.product(legs_num_sectors)
        # this is different from num_sectors

        # determine block_ind_map -- it's essentially the grid.
        block_ind_map = np.zeros((nblocks, 3 + self.num_legs), dtype=np.intp)
        block_ind_map[:, 2:-1] = grid  # possible combinations of indices
        # block_ind_map[:, :2] and [:, -1] are set later.

        # the multiplicity for given (i1, i2, ...) is the product of ``multiplicities[il]``
        # advanced indexing:
        # ``grid.T[li]`` is a 1D array containing the block_indices `b_li` of leg ``li`` for all blocks
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(self.legs, grid.T)], axis=0)

        # calculate new defining_sectors
        # at this point, they have duplicates and are not sorted
        sectors = self.symmetry.multiple_fusion_broadcast(
            *(s.sector_decomposition[gr] for s, gr in zip(self.legs, grid.T))
        )
        if self.is_dual:
            # the above are the future self.sector_decomposition
            # but we want to compute (and in particular sort according to) the defining_sectors
            sectors = self.symmetry.dual_sectors(sectors)

        # sort sectors
        self.fusion_outcomes_sort = fusion_outcomes_sort = np.lexsort(sectors.T)
        block_ind_map = block_ind_map[fusion_outcomes_sort]
        sectors = sectors[fusion_outcomes_sort]
        multiplicities = multiplicities[fusion_outcomes_sort]

        # compute slices in the whole internal basis (we subtract the start of each block later)
        slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
        block_ind_map[:, 0] = slices[:-1]  # start with 0
        block_ind_map[:, 1] = slices[1:]

        # bunch sectors with equal sectors together
        diffs = find_row_differences(sectors, include_len=True)  # include len, to index slices
        self.block_ind_map_slices = diffs
        slices = slices[diffs]
        multiplicities = slices[1:] - slices[:-1]
        diffs = diffs[:-1]  # now exclude len, to index sectors by diffs

        sectors = sectors[diffs]

        new_block_ind = np.zeros(len(block_ind_map), dtype=np.intp)  # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        block_ind_map[:, :2] -= slices[new_block_ind][:, np.newaxis]
        self.block_ind_map = block_ind_map

        return sectors, multiplicities

    def _calc_basis_perm(self, multiplicities):
        """Calculate the :attr:`basis_perm`.

        Helper function for :meth:`__init__`.
        Assumes ``LegPipe.__init__`` and :meth:`_calc_sectors` were called. Returns the basis_perm.

        The :attr:`basis_perm` of an :class:`AbelianLegPipe` should be such that
        ``tensor.combined_legs(...).to_numpy() == tensor.to_numpy().reshape(...)``

        This is achieved by demanding that the following two paths, as operations on
        ordered bases, are equivalent::

            public      --------------------->     internal     ----------->    fusion
            uncoupled     legs `basis_perm`s       uncoupled      fusion        outcomes
               |                                                                   |
               | fusion                                                            | sort
               v                                                                   v
            public      --------------------------------------------------->    internal
            coupled                     pipe.basis_perm                         coupled


        Here, ``fusion`` stands for first forming combinations, either in C-style or F-style order,
        depending on :attr:`combine_cstyle`, then performing the fusion of sectors, e.g. via
        :meth:`Symmetry.fusion_outcomes_broadcast` of the :attr:`sector_decomposition`.
        ``sort`` on the other hand stands for stable-sorting the resulting basis elements by sector.
        Depending on :attr:`is_dual`, we either sort by ``np.lexsort(_.T)`` (if ``is_dual=False``)
        or by ``np.lexsort(dual_sectors(_).T)`` (if ``is_dual=True``), i.e. such that the resulting
        :attr:`defining_sectors` are sorted.

        OPTIMIZE (JU) should we make this on-demand only? i.e. make ``_basis_perm`` a cached property?
        """
        # see diagram in docstring, we follow the path parallel to ``pipe.basis_perm``.
        # inverse of fusion
        order = 'C' if self.combine_cstyle else 'F'
        res2 = np.reshape(np.arange(self.dim), [leg.dim for leg in self.legs], order=order)
        # apply basis perm of each leg
        res2 = res2[np.ix_(*(leg.basis_perm for leg in self.legs))]
        # fusion
        res2 = np.reshape(res2, (self.dim,), order=order)
        # apply fusion_outcomes_perm (``sort`` in the diagram)
        res2 = res2[self._get_fusion_outcomes_perm(multiplicities)]
        return res2

    def _get_fusion_outcomes_perm(self, multiplicities):
        r"""Get the permutation of basis elements that is introduced by the fusion.

        Helper function for :meth:`_calc_basis_perm`.

        This permutation arises as follows:
        For each of the :attr:`legs` consider all basis elements by order of appearance in the
        internal order, i.e. in :attr:`ElementarySpace.sector_decomposition``. Take all combinations
        of basis elements from all the legs. Use C-style or F-style order of the combinations,
        according to :attr:`combine_cstyle`. For each combination, perform the fusion (for abelian
        symmetries this yields a single sector each). This yields a list of basis elements of the
        combined space (pipe). The target permutation stable-sorts this list by sector.
        Depending on :attr:`is_dual`, we either sort by ``np.lexsort(_.T)`` (if ``is_dual=False``)
        or by ``np.lexsort(dual_sectors(_).T)`` (if ``is_dual=True``), i.e. such that the resulting
        :attr:`defining_sectors` are sorted.
        """
        dim_strides = make_stride([leg.dim for leg in self.legs], cstyle=self.combine_cstyle)
        perm = np.empty(self.dim, int)

        # slices_starts is slices[:, 0], but we need to compute it here,
        # since ElementarySpace.__init__ was not called yet at this point
        slices_starts = np.concatenate([[0], np.cumsum(multiplicities)[:-1]])

        for start, stop, *idcs, J in self.block_ind_map:
            # shift the slice start:stop from within the block back to within the whole internal basis
            offset = slices_starts[J]
            start = start + offset
            stop = stop + offset

            # Now for each basis element in start:stop, we construct where it was before sorting

            # multiplicity_grid :: each row stands for a combination of uncoupled basis elements.
            #                      they are the indices of that basis element *within* the sector.
            multiplicity_grid = make_grid(
                [leg.multiplicities[idx] for leg, idx in zip(self.legs, idcs)], cstyle=self.combine_cstyle
            )

            # sector_starts[n] is the index of the first basis vector for legs[n] that is in the
            # current sector, namely legs[n].sector_decomposition[idcs[n]]
            sector_starts = np.array([leg.slices[idx, 0] for leg, idx in zip(self.legs, idcs)])

            # basis_grid :: each row stands for a combination of uncoupled basis elements.
            #               they are the indices of that basis element within its legs internal basis
            basis_grid = multiplicity_grid + sector_starts

            # now we need to map the multi-indices (rows of basis_grid) to single indices into
            # the unsorted list of fusion outcomes. Note that the relevant strides are ``dim_strides``,
            # and that these strides come from a *different* shape than the multiplicity_grid.
            # That is, we want to do ``perm[start + n] = np.sum(basis_grid[n] * dim_strides)``.
            # Turns out we can do it batched:
            perm[start:stop] = np.sum(basis_grid * dim_strides, axis=1)

        return perm


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


def _parse_inputs_drop_symmetry(which: int | list[int] | None, symmetry: Symmetry) -> tuple[list[int] | None, Symmetry]:
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
