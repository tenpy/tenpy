# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from itertools import product
import numpy as np
from numpy import ndarray
import copy

from .groups import SectorArray, Symmetry, no_symmetry


__all__ = ['VectorSpace', 'ProductSpace']


class VectorSpace:
    """A vector space, which decomposes into sectors of a given symmetry.

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
        To construct a dual space, consider using `space.dual` instead.

        .. warning :
            For ``_is_dual==True``, the passed `sectors` are interpreted as the sectors of
            the ("non-dual") ket-space isomorphic to self.
            They are stored as ``self._sectors``, while ``self.sectors``, accessed via the property,
            are the duals of `sectors == self._sectors` if ``_is_dual==True``.
            This means that to construct the dual of ``VectorSpace(..., some_sectors)``,
            we need to call ``VectorSpace(..., some_sectors, _is_dual=True)`` and in particular
            pass the _same_ sectors.
    """

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
        # TODO (JU) if we have a version of sector_dim that works on SectorArray, we could use
        #  numpy __mul__ and np.sum here...
        self.dim = sum(symmetry.sector_dim(s) * m for s, m in zip(sectors, self.multiplicities))
        self.is_dual = _is_dual

        if is_real:
            # TODO (JU): pretty sure some parts of linalg.symmetries.groups relies on
            #  the assumption of complex vector spaces. not sure though, need to check.
            raise NotImplementedError
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

    def sectors_str(self) -> str:
        """short str describing the self._sectors and their multiplicities"""
        # TODO (JU) what if there are a lot of sectors?
        # (JH) maybe print up to 5 or 10 with largest multiplicities?
        return ', '.join(f'{self.symmetry.sector_str(a)}: {mult}'
                         for a, mult in zip(self._sectors, self.multiplicities))

    # TODO (JU) this product is not associative; a * (b * c) and (a * b) * c have different nestings.
    #  should we even define __mul__ ...?
    def __mul__(self, other):
        if isinstance(other, VectorSpace):
            return ProductSpace([self, other])
        return NotImplemented

    def __repr__(self):
        # TODO (JU) what if there are a lot of sectors?
        dual_str = '.dual' if self.is_dual else ''
        is_real_str = ', is_real=True' if self.is_real else ''
        return f'VectorSpace({repr(self.symmetry)}, sectors={self.sectors}, ' \
               f'multiplicities={self.multiplicities}{is_real_str}){dual_str}'

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

        # TODO: this is probably inefficient. eventually this should all be C(++) anyway...
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

    def can_contract_with(self, other):
        if self.is_real:
            # TODO (JU) is this actually true...?
            #  it is if we ignore symmetries, but with symmetries we should take care of the sectors?
            # (JH) just by defining the `is_dual` flag to distinguish bra vs ket,
            #      I'd say we should still check self == other.dual for is_real=True.
            return self == other
        else:
            return self == other.dual

    # TODO (JU) deprecate this in favor of can_contract_with ?
    def is_dual_of(self, other):
        return self == other.dual

    @property
    def is_trivial(self) -> bool:
        if self._sectors.shape[0] != 1:
            return False
        if not np.all(self._sectors[0] == self.symmetry.trivial_sector):
            return False
        # have already checked if there is more than 1 sector, so can assume self.multiplicities.shape == (1,)
        if self.multiplicities[0] != 1:
            return False
        return True

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
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
        While mathematically
        ``ProductSpace([s.dual for s in spaces]).dual == ProductSpace(spaces)``,
        in this implementation we explicitly distinguish those spaces, and consider them as
        non-equal due to a different :attr:`is_dual` flag!
        This flag is somewhat artifical, but necessary to allows us to
        a) sort by charge sectors on the non-dual spaces to ensure matching order between legs that
        can be contracted, and further
        b) consistently view every `ProductSpace` as a :class:`VectorSpace`, i.e. have proper
        subclass behavior.

    Parameters
    ----------
    spaces:
        The factor spaces that multiply to this space.
        For `is_dual=True`, they are the factors of the dual space.
    _is_dual : bool
        Flag indicating wether the fusion space represents a dual (bra) space or a non-dual (ket)
        space. See note above.
    _sectors, _multiplicities:
        Can optionally be passed to avoid recomputation.
        These are the inputs to VectorSpace.__init__, so they are unchanged by flipping is_dual.
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
        VectorSpace.__init__(self,
                             symmetry=symmetry,
                             sectors=_sectors,
                             multiplicities=_multiplicities,
                             is_real=is_real,
                             _is_dual=_is_dual)

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
        for `VectorSpace`s ``V`` and ``W``.
        """
        # note: yields dual self._sector so can have different sorting of _sectors!
        # so can't just pass self._sectors and self._multiplicities
        # TODO: should we return the associated permutation of block_indices?
        return ProductSpace(spaces=self.spaces, _is_dual=not self.is_dual)

    def gauge_is_dual(self, is_dual: bool) -> ProductSpace:
        """Return a ProductSpace isomorphic (or equal) to self with the given is_dual attribute."""
        if is_dual == self.is_dual:
            return self
        else:
            return self.flip_is_dual()

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return iter(self.spaces)

    def __repr__(self):
        lines = ['ProductSpace([']
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
            res = f'dual(res)'
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
        fusion = dict((tuple(s), m) for s, m in zip(spaces[0].sectors, spaces[0].multiplicities))
        for space in spaces[1:]:
            new_fusion = {}
            for t_a, m_a in fusion.items():
                s_a = np.array(t_a)
                for s_b, m_b in zip(space.sectors, space.multiplicities):
                    for s_c in symmetry.fusion_outcomes(s_a, s_b):
                        t_c = tuple(s_c)
                        # TODO do we need to take symmetry.sector_dim into account here?
                        #  JU: no. the multiplicity of a sector in a space does not include the sector_dim.
                        #      the dimension of the space is (roughly)
                        #      sum(multiplicities[i] * sector_dim(sectors[i]))
                        n = symmetry.n_symbol(s_a, s_b, s_c)
                        new_fusion[t_c] = new_fusion.get(t_c, 0) + m_a * m_b * n
            fusion = new_fusion
            # by convention fuse spaces left to right, i.e. (...((0,1), 2), ..., N)
        _sectors = np.asarray(list(fusion.keys()))
        multiplicities = np.asarray(list(fusion.values()))
        return _sectors, multiplicities
