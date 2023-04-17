# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from itertools import product
import numpy as np
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
    
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: np.ndarray = None,
                 is_real: bool = False, _is_dual: bool = False):
        self.symmetry = symmetry
        self._sectors = sectors

        # TODO (JU): call it num_sectors for PEP8s sake (i.e. uppercase only for classes)?
        self.N_sectors = N_sectors = len(sectors)

        # TODO (JU) make multiplicities a numpy array?
        if multiplicities is None:
            self.multiplicities = [1] * N_sectors
        else:
            assert len(multiplicities) == N_sectors
            self.multiplicities = multiplicities
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
        return f'VectorSpace(symmetry={self.symmetry}, sectors={self.sectors}, ' \
               f'multiplicities={self.multiplicities}, is_real={self.is_real}){dual_str}'

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
            return False

        if self.is_real != other.is_real:
            return False

        if self.is_dual != other.is_dual:
            return False

        if len(self.sectors) != len(other.sectors):
            # now we may assume that checking all multiplicities of self is enough.
            return False

        # TODO: this is probably inefficient. eventually this should all be C(++) anyway...
        # it might be enough to check sectors in order, if we fix the order through a convention
        # then we don't need to generate the lookup dict here
        other_multiplicities = {sector: mult for sector, mult in zip(other.sectors, other.multiplicities)}

        return all(mult == other_multiplicities.get(sector, -1)
                   for mult, sector in zip(self.multiplicities, self.sectors))

    @property
    def dual(self):
        res = copy.copy(self)  # shallow copy, works for subclasses as well
        res.is_dual = not self.is_dual
        return res

    def can_contract_with(self, other):
        if self.is_real:
            # TODO (JU) is this actually true...?
            #  it is if we ignore symmetries, but with symmetries we should take care of the sectors?
            return self == other
        else:
            return self == other.dual

    # TODO (JU) deprecate this in favor of can_contract_with ?
    def is_dual_of(self, other):
        # FIXME think about duality in more detail.
        #  i.e. is a
        # `Vectorspace(a.symmetry, [sector.dual for sector in a.sectors], a.multiplicities, not a.is_dual, a.is_real) == a` ?
        # JH: no, it's not, the `is_dual` indicates whether it's a "bra" or "ket" space.
        return self == other.dual

    @property
    def is_trivial(self) -> bool:
        if self._sectors.shape[0] != 1:
            return False
        if not np.all(self._sectors[0] == self.symmetry.trivial_sector):
            return False
        if self.multiplicities != [1]:
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

    Parameters
    ----------
    spaces:
        The factor spaces that multiply to this space.
    _is_dual : bool
        Whether this is the "normal" product of ``spaces`` or the dual of the product of ``spaces``.
        To construct duals, consider using `space.dual` instead.

        .. warning :
            For ``_is_dual==True``, the passed `spaces` are interpreted as the factors of 
            the "non-dual" space.
            This means that to construct the dual of ``ProducSpace(some_spaces)``,
            wen need to call ``ProductSpace(some_spaces, _is_dual=True)`` and in particular
            pass the _same_ spaces.
            In particular, this is not the same as ``ProductSpace([s.dual for s in some_spaces])``.
            
    _sectors:
        Can optionally pass the sectors of self, to avoid recomputation.
        These are the inputs to VectorSpace.__init__, so they are unchanged by flipping is_dual.
    _multiplicities:
        Can optionally pass the multiplicities of self, to avoid recomputation.
    """
    
    def __init__(self, spaces: list[VectorSpace], is_dual: bool = False, 
                 _sectors: SectorArray = None, _multiplicities: np.ndarray = None):
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)
        if _sectors is None or _multiplicities is None:
            _sectors, _multiplicities = self._fuse_spaces(spaces)
        VectorSpace.__init__(self,
                             symmetry=symmetry,
                             sectors=_sectors,
                             multiplicities=_multiplicities,
                             is_real=is_real,
                             is_dual=is_dual)
        
    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self._sectors,  # underscore is important!
                           multiplicities=self.multiplicities,
                           is_dual=self.is_dual,
                           is_real=self.is_real)

    def flip_is_dual(self) -> ProductSpace:
        """Return a ProductSpace isomrophic to self with opposite is_dual attribute.

        This realizes the isomorphism between ``dual(V) * dual(W)`` and ``dual(V * W)``.
        """
        return ProductSpace(spaces=[s.dual for s in self.spaces], is_dual=not self.is_dual,
                            _sectors=self._sectors, _multiplicities=self.multiplicities)
        
    def gauge_is_dual(self, is_dual: bool) -> ProductSpace:
        """Return a ProductSpace isomrophic to self with the given is_dual attribute."""
        if is_dual == self.is_dual:
            return self
        else:
            return self.flip_is_dual()

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return iter(self.spaces)

    def __repr__(self):
        lines = ['ProductSpace([', *map(repr, self.spaces), '])']
        if self.is_dual:
            lines[-1:-1] = [f'is_dual={self.is_dual}']
        return '\n'.join(lines)

    def __str__(self):
        res = ' ⊗ '.join(map(str, self.spaces))
        if self.is_dual:
            res = f'dual(res)'
        return res

    def __eq__(self, other):
        if not isinstance(other, ProductSpace):
            return False
        if other.is_dual != self.is_dual:
            return False
        if len(other.spaces) != len(self.spaces):
            return False
        return all(s1 == s2 for s1, s2 in zip(self.spaces, other.spaces))

    @property
    def dual(self):
        return ProductSpace(spaces=self.spaces, is_dual=not self.is_dual, _sectors=self._sectors,
                            _multiplicities=self.multiplicities)

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    def _fuse_spaces(self, spaces: list[VectorSpace], symmetry: Symmetry
                     ) -> tuple[SectorArray, np.ndarray]:
        """Calculate sectors and multiplicities in the fusion of spaces."""
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
        sectors = np.asarray(fusion.keys())
        multiplicities = fusion.values()
        
        # note: sectors are not sorted here; need `is_dual` to allow correct sorting.
        # TODO FIXME (JU): no, we can sort them here. Those are the "non-dual" sectors. right?
        
        return sectors, multiplicities
