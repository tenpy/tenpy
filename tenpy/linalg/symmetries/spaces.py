# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from itertools import product
import numpy as np
from numpy import ndarray
import copy

from .groups import Sector, SectorArray, Symmetry, no_symmetry


__all__ = ['VectorSpace', 'ProductSpace']


class VectorSpace:
    """A vector space, which decomposes into sectors of a given symmetry.

    .. note ::
        It is best to think of ``VectorSpace``s as immutable objects.
        In practice they are mutable, i.e. you could change their attributes. but this may lead to
        unexpected behavior, since it might make the cached metadata inconsistent.

    Attributes
    ----------
    sectors : 2D np array of int
        The sectors that compose this space.
    _sectors: : 2D np array of int
        Internally stored version of :attr:`sectors`. These are the sectors of the "non-dual" ket-space,
        that is either equal (if `self.is_dual is False`) or isomorphic (if `self.is_dual is True`) to
        `self`. This allows us to bring the sectors in a canonical order that is the same for a space
        and its dual, by sorting `_sectors` instead of `sectors`.
    is_dual : bool
        Whether this is the dual (a.k.a. bra) space, composed of `sectors == symmtery.dual_sectors(_sectors)`
        or the regular (a.k.a. ket) space composed of `sectors == _sectors`.

    Parameters
    ----------
    symmetry:
        The symmetry associated with this space.
    sectors:
        The sectors of the symmetry that compose this space.
        A 2D array of integers with axes [s, q] where s goes over different sectors
        and q over the different quantities needed to describe a sector.
        If sectors appear multiple times in the space, they may either be described through
        repetition in sectors or via multiplicities.
    multiplicities:
        How often each of the `sectors` appears.
        A 1D array of positive integers with axis [s].
        ``sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
        If not given, a multiplicity ``1`` is assumed for all `sectors`.
    is_real : bool
        Whether the space is over the real numbers.
        Otherwise it is over the complex numbers (default).
    _is_dual : bool
        Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.

        .. warning :
            For ``_is_dual is True``, the passed `sectors` are interpreted as the sectors of the
            ("non-dual") ket-space isomorphic to self.
            They are stored as ``self._sectors``, while ``self.sectors``, accessed via the property,
            are the duals of `sectors == self._sectors` if ``_is_dual is True``.
            This means that to construct the dual of ``VectorSpace(..., some_sectors)``,
            we need to call ``VectorSpace(..., some_sectors, _is_dual=True)`` and in particular
            pass the *same* sectors.
            Consider using ``VectorSpace(..., some_sectors).dual`` instead for more readable code.
    """
    ProductSpace = None  # we set this to the ProductSpace class below
    # for subclasses, it's the corresponding ProductSpace subclass, e.g.
    # AbelianBackendVectorSpace.ProductSpace = AbelianBackendProductSpace
    # This allows combine_legs() etc to generate appropriate sublcasses

    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 is_real: bool = False, _is_dual: bool = False):
        self.symmetry = symmetry
        self._sectors = np.asarray(sectors, dtype=int)
        self.num_sectors = num_sectors = len(sectors)

        if multiplicities is None:
            multiplicities = np.ones((num_sectors,), dtype=int)
        multiplicities = np.asarray(multiplicities, dtype=int)
        assert np.all(multiplicities > 0)
        assert multiplicities.shape == (num_sectors,)
        self.multiplicities = multiplicities
        self.dim = np.sum(symmetry.batch_sector_dim(sectors) * multiplicities)
        self.is_dual = _is_dual
        self.is_real = is_real

    @classmethod
    def non_symmetric(cls, dim: int, is_real: bool = False, _is_dual: bool = False):
        return cls(symmetry=no_symmetry, sectors=no_symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_real=is_real, _is_dual=_is_dual)

    @property
    def sectors(self):
        if self.is_dual:
            return self.symmetry.dual_sectors(self._sectors)
        return self._sectors

    def sector(self, i: int) -> Sector:
        """Return a single sector for a given index `i`.
        This is equivalent to ``self.sectors[i]``, but faster if ``self.is_dual``.
        In particular, it ignores the ``self.sector_perm``.
        """
        sector = self._sectors[i, :]
        if self.is_dual:
            return self.symmetry.dual_sector(sector)
        return sector

    def sectors_str(self, separator=', ', max_len=70) -> str:
        """short str describing the self._sectors (note the underscore!) and their multiplicities"""
        full = separator.join(f'{self.symmetry.sector_str(a)}: {mult}'
                              for a, mult in zip(self._sectors, self.multiplicities))
        if len(full) <= max_len:
            return full

        res = ''
        end = '[...]'

        for idx in np.argsort(self.multiplicities):
            new = f'{self.symmetry.sector_str(self._sectors[idx])}: {self.multiplicities[idx]}'
            if len(res) + len(new) + len(end) + 2 * len(separator) > max_len:
                return res + separator + end
            res = res + separator + new
        raise RuntimeError  # a return should be triggered from within the for loop!

    def __repr__(self):
        dual_str = '.dual' if self.is_dual else ''
        is_real_str = ', is_real=True' if self.is_real else ''
        sectors_str = repr(self._sectors)
        if len(sectors_str) > 50:
            sectors_str = '[...]'
        return f'{self.__class__.__name__}({self.symmetry!r}, sectors={sectors_str}, ' \
               f'multiplicities={self.multiplicities!s}{is_real_str}){dual_str}'

    def __str__(self):
        field = 'ℝ' if self.is_real else 'ℂ'
        if self.symmetry == no_symmetry:
            symm_details = ''
        else:
            symm_details = f'[{self.symmetry}, {self.sectors_str()}]'
        res = f'{field}^{self.dim}{symm_details}'
        return f'dual({res})' if self.is_dual else res

    def __eq__(self, other):
        if not isinstance(other, VectorSpace):
            return NotImplemented

        if self.is_real != other.is_real:
            return False

        if self.is_dual != other.is_dual:
            return False

        if self.num_sectors != other.num_sectors:
            # now we may assume that checking all multiplicities of self is enough.
            return False

        # TODO: (JH) we should by convention always sort self._sectors...
        self_order = np.argsort(self.sectors, axis=0)
        other_order = np.argsort(other.sectors, axis=0)
        return np.all(self.sectors[self_order] == other.sectors[other_order]) \
            and np.all(self.multiplicities[self_order] == other.multiplicities[other_order])

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
        # note: yields dual self._sectors so can have different sorting of _sectors!
        # sectors can get sorted in VectorSpace.__init__() (e.g. in AbelianBackendVectorSpace)
        return self.__class__(self.symmetry,
                              self.symmetry.dual_sectors(self._sectors),
                              self.multiplicities,
                              self.is_real,
                              not self.is_dual)

    def can_contract_with(self, other):
        """If self can be contracted with other.

        Equivalent to ``self == other.dual``"""
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
        return np.all(self._sectors == other._sectors) and np.all(self.multiplicities == other.multiplicities)

    @property
    def is_trivial(self) -> bool:
        """Whether self is the trivial space.

        The trivial space is the one-dimensional space which consists only of the trivial sector,
        appearing exactly once. In a mathematical sense, the trivial sector _is_ the trivial space.
        """
        if self._sectors.shape[0] != 1:
            return False
        # have already checked if there is more than 1 sector, so can assume self.multiplicities.shape == (1,)
        if self.multiplicities[0] != 1:
            return False
        if not np.all(self._sectors[0] == self.symmetry.trivial_sector):
            return False
        return True

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the number of linearly independent symmetric tensors in this space."""
        # TODO isnt this just the multiplicity of the trivial sector?
        raise NotImplementedError  # TODO


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

    .. note ::
        While mathematically the dual of the product :math:`(V \otimes W)^*` is the same as the
        product of the duals :math:`V^* \otimes W^*`, we distinguish these two objects in the
        implementation. This allows us to fulfill all of the following constraints
        a) Have the same order of `_sectors` for a space and its dual, to make contractions easier.
        b) Consistently view every ProductSpace as a VectorSpace, i.e. have proper subclass behavior
           and in particular a well-behaved `is_dual` attribute.
        c) A ProductSpace can always be split into its :attr:`spaces`.

        As an example, consider two VectorSpaces ``V`` and ``W`` and the following four possible
        products::

            ==== ============================ ================== ========= =================
                 Mathematical Expression      .spaces            .is_dual  ._sectors
            ==== ============================ ================== ========= =================
            P1   :math:`V \otimes W`          [V, W]             False     P1._sectors
            P2   :math:`(V \otimes W)^*`      [V.dual, W.dual]   True      P1._sectors
            P3   :math:`V^* \otimes W^*`      [V.dual, W.dual]   False     dual(P1._sectors)
            P4   :math:`(V^* \otimes W^*)^*`  [V, W]             True      dual(P1._sectors)
            ==== ============================ ================== ========= =================

        They can be related to each other via the :attr:`dual` property or via :meth:`flip_is_dual`.
        In this example we have `P1.dual == P2`` and ``P3.dual == P4``, as well as
        ``P1.flip_is_dual() == P4`` and ``P2.flip_is_dual() == P3``.

        The mutually dual spaces, e.g. ``P1`` and ``P2``, can be contracted with each other, as they
        have opposite :attr:`is_dual` and matching :attr:`._sectors`.
        The spaces related by :meth:`flip_is_dual()`, e.g. ``P2`` and ``P3``, would be considered
        the same space mathematically, but in this implementation we have ``P2 != P3`` due to the
        different :attr:`is_dual` attribute.
        Since they represent the same space, they have the same entries in :attr:`sectors` (no
        underscore!), but not necessarily in the same order; due to the different :attr:`is_dual`,
        their :attr:`_sectors` are different and we sort by :attr:`_sectors`, not :attr:`sectors`.
        This also means that ``P1.can_contract_with(P3) is False``.
        The contraction can be done, however, by first converting ``P3.flip_is_dual() == P2``,
        since then ``P1.can_contract_with(P2) is True``.
        # TODO (JU) is there a corresponding function that does this on a tensor? -> reference it.

        This convention has the downside that the mathematical notation :math:`P_2 = (V \otimes W)^*`
        does not transcribe trivially into a single call of ``ProductSpace.__init__``, since
        ``P2 = ProductSpace([V.dual, W.dual], is_dual=True)``.
        Consider writing ``P2 = ProductSpace([V, W]).dual`` instead for more readable code.

    Parameters
    ----------
    spaces:
        The factor spaces that multiply to this space.
        The resulting product space can always be split back into these.
    _is_dual : bool
        Flag indicating wether the fusion space represents a dual (bra) space or a non-dual (ket) space.

        .. warning ::
            When setting `_is_dual=True`, consider the note above!

    _sectors, _multiplicities:
        Can optionally be passed to avoid recomputation.
        These are the inputs to VectorSpace.__init__, as computed by _fuse_spaces.
    """

    def __init__(self, spaces: list[VectorSpace], _is_dual: bool = False,
                 _sectors: SectorArray = None, _multiplicities: ndarray = None):
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)
        if _sectors is None or _multiplicities is None:
            _sectors, _multiplicities = self._fuse_spaces(symmetry, spaces, _is_dual)
        super().__init__(symmetry=symmetry,
                         sectors=_sectors,
                         multiplicities=_multiplicities,
                         is_real=is_real,
                         _is_dual=_is_dual)

    # TODO python naming convention is snake case: as_vector_space
    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self._sectors,  # underscore is important!
                           multiplicities=self.multiplicities,
                           is_real=self.is_real,
                           _is_dual=self.is_dual)

    def flip_is_dual(self) -> ProductSpace:
        """Return a ProductSpace isomorphic to self, which has the opposite is_dual attribute.

        This realizes the isomorphism between ``V.dual * W.dual`` and ``(V * W).dual``
        for `VectorSpace` ``V`` and ``W``.
        However, note that the returned space can often not be contracted with `self`
        since the order of the :attr:`sectors` might have changed.
        """
        # note: yields dual self._sectors so can have different sorting of _sectors!
        # so can't just pass self._sectors and self._multiplicities
        # TODO (JU) we can pass self.symmetry.dual_sectors(self._sectors) and self.multiplicities.
        #           we just need to be careful if we need to sort them here or if __init__ takes care of it.
        # note: Using self.__class__ makes this work for the backend-specific subclasses as well.
        return self.__class__(spaces=self.spaces, _is_dual=not self.is_dual)

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return iter(self.spaces)

    def __repr__(self):
        lines = [f'{self.__class__.__name__}([']
        for s in self.spaces:
            lines.append(f'  {repr(s)},')
        if self.is_dual:
            lines.append(']).dual')
        else:
            lines.append(f'])')
        return '\n'.join(lines)

    def __str__(self):
        res = ' ⊗ '.join(map(str, self.spaces))
        if self.is_dual:
            res = f'dual({res})'
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

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool,
                     ) -> tuple[SectorArray, ndarray]:
        """Calculate sectors and multiplicities in the fusion of spaces."""
        if _is_dual:
            spaces = [s.dual for s in spaces] # directly fuse sectors of dual spaces.
            # This yields overall dual `sectors` to return, which we directly save in
            # self._sectors, such that `self.sectors` (which takes a dual!) yields correct sectors
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
        _sectors = np.asarray(list(fusion.keys()))
        multiplicities = np.asarray(list(fusion.values()))
        return _sectors, multiplicities


VectorSpace.ProductSpace = ProductSpace
