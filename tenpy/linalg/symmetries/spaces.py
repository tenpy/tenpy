# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from itertools import product
import numpy as np
import copy

from .groups import SectorArray, Symmetry, no_symmetry


__all__ = ['VectorSpace', 'ProductSpace', 'FusionSpace']


class VectorSpace:
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: list[int] = None,
                 is_dual: bool = False, is_real: bool = False):
        """A vector space, which decomposes into sectors of a given symmetry.
        Parameters
        ----------
        is_dual:
            Whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.
            For ``is_dual=True`` the stored `self._sectors` are the dual of the passed `sectors`,
            but `self.sectors` still returns the original (dual of the dual) sectors.
        is_real:
            Whether the space is over the real numbers (otherwise over the complex numbers)
        """
        self.symmetry = symmetry
        if is_dual:
            # by convention, we store non-dual sectors in self._sectors
            self._sectors = symmetry.dual_sectors(sectors)
        else:
            self._sectors = sectors
        self.N_sectors = N_sectors = len(sectors)

        # TODO (JU) make multiplicities a numpy array?
        if multiplicities is None:
            self.multiplicities = [1] * N_sectors
        else:
            assert len(multiplicities) == N_sectors
            self.multiplicities = multiplicities
        self.dim = sum(symmetry.sector_dim(s) * m for s, m in zip(sectors, self.multiplicities))
        self.is_dual = is_dual
        self.is_real = is_real

    @classmethod
    def non_symmetric(cls, dim: int, is_dual: bool = False, is_real: bool = False):
        return cls(symmetry=no_symmetry, sectors=no_symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_dual=is_dual, is_real=is_real)

    @property
    def sectors(self):
        if self.is_dual:
            return self.symmetry.dual_sectors(self._sectors)
        return self._sectors

    def sectors_str(self) -> str:
        """short str describing the (possibly dual) sectors and their multiplicities"""
        # FIXME variable `dual` not defined
        return ', '.join(f'{self.symmetry.sector_str(a)}{dual}: {mult}'
                         for a, mult in zip(self._sectors, self.multiplicities))

    # TODO (JU) define mul for ProductSpace?
    def __mul__(self, other):
        if isinstance(other, VectorSpace):
            return ProductSpace([self, other])
        return NotImplemented

    def __repr__(self):
        return f'VectorSpace(symmetry={self.symmetry}, sectors={self.sectors}, multiplicities={self.multiplicities}, ' \
               f'is_dual={self.is_dual}, is_real={self.is_real})'

    def __str__(self):
        field = 'ℝ' if self.is_real else 'ℂ'
        if self.symmetry == no_symmetry:
            symm_details = ''
        else:
            symm_details = f'[{self.symmetry}, {self.sectors_str()}]'
        res = f'{field}^{self.dim}{symm_details}'
        # TODO does duality of sectors make sense like this (as defined in sectors_str)
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
            return self == other
        else:
            return self == other.dual

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


# TODO: does the distinction between ProductSpace and FusionSpace make sense?
#  JU: FusionSpace looks good. I dont think we need the current ProductSpace.
#      If we keep only FusionSpace, we might name it ProductSpace again.
#      If we keep both, ProductSpace should not be a subclass of VectorSpace, since we dont
#       evaluate what its sectors as a symmetry-graded VectorSpace are. Then, a FusionSpace
#       would be a VectorSpace and a ProductSpace.
class ProductSpace(VectorSpace):
    def __init__(self, spaces: list[VectorSpace], is_dual: bool = False):
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        # TODO FIXME sectors are lists of previous sectors and hence not valid sector for the given symmetry!?
        sectors = [list(combination) for combination in product(*(space.sectors for space in spaces))]
        multiplicities = [np.prod(combination) for combination in product(*(space.multiplicities for space in spaces))]
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)

        VectorSpace.__init__(self, symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                             is_dual=is_dual, is_real=is_real)

    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self.sectors,
                           multiplicities=self.multiplicities,
                           is_dual=self.is_dual,
                           is_real=self.is_real)

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        return self.spaces.__iter__()

    def __repr__(self):
        return '\n'.join(('ProductSpace([', *map(repr, self.spaces), '])'))

    def __str__(self):
        return ' ⊗ '.join(map(str, self.spaces))

    def __eq__(self, other):
        if isinstance(other, ProductSpace):
            # FIXME need to be more careful about is_dual flags!
            return self.spaces == other.spaces
        else:
            return False

    @property
    def dual(self):
        # need to flip both self.is_dual and self.spaces[:].is_dual to keep it consistent!
        res = copy.copy(self)  # works for subclasses as well
        res.is_dual = not self.is_dual
        res.spaces = [s.dual() for s in self.spaces]
        return res

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
        raise NotImplementedError  # TODO


class FusionSpace(VectorSpace):
    r"""Take the product of multiple spaces and fuse them left-to-right.
    This generates a fusion tree looking like this (or it's dual flipped upside down)::
        spaces[0]
             \   spaces[1]
              \ /
               Y   spaces[2]
                \ /
                 Y
                  \
                   ....
    It is the product space of the individual `spaces`,
    but with an associated basis change implied to allow preserving the symmetry.
    """
    def __init__(self, spaces: list[VectorSpace], is_dual: bool = False):
        assert len(spaces) > 0
        symmetry = spaces[0].symmetry
        self.spaces = spaces  # spaces can themselves be ProductSpaces

        fused_sectors, fused_multiplicities = self._fuse_spaces(spaces, is_dual)
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)

        # for `is_dual=True` VectorSpace.__init__(...) just saves dual self._sectors internally
        # TODO think through non-abelian case where switching is_dual compared to spaces
        #      implicitly contracts a cap/cup.
        VectorSpace.__init__(self,
                             symmetry=symmetry,
                             sectors=fused_sectors,
                             multiplicities=fused_multiplicities,
                             is_dual=is_dual,
                             is_real=is_real)

    def _fuse_spaces(self, spaces: list[VectorSpace], is_dual: bool):
        """Calculate sectors and multiplicities of possible fusion results from merging spaces."""
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)

        # use t_ = tuple(s_) as dict keys with a bit ugly conversion between tuple and ndarray
        fusion = dict((tuple(s), m) for s, m in zip(spaces[0].sectors, spaces[0].multiplicities))
        for space in spaces[1:]:
            new_fusion = {}
            for t_a, m_a in fusion.items():
                s_a = np.array(t_a)
                for s_b, m_b in zip(space.sectors, space.multiplicities):
                    for s_c in symmetry.fusion_outcomes(s_a, s_b):
                        t_c = tuple(s_c)
                        # TODO FIXME do we need to take symmetry.sector_dim into account here?
                        n = symmetry.n_symbol(s_a, s_b, s_c)
                        new_fusion[t_c] = new_fusion.get(t_c, 0) + m_a * m_b * n
            fusion = new_fusion
            # by convention fuse spaces left to right, i.e. (...((0,1), 2), ..., N)
        sectors = np.asarray(fusion.keys())
        multiplicities = fusion.values()
        # note: sectors are not sorted here; need `is_dual` to allow correct sorting.
        return sectors, multiplicities

    def as_VectorSpace(self):
        """Forget about the substructure of the ProductSpace but view only as VectorSpace.
        This is necessary before truncation, after which the product-space structure is no
        longer necessarily given.
        """
        return VectorSpace(symmetry=self.symmetry,
                           sectors=self.sectors,
                           multiplicities=self.multiplicities,
                           is_dual=self.is_dual,
                           is_real=self.is_real)

    def __repr__(self):
        return '\n'.join(('FusionSpace([', *map(repr, self.spaces), '])'))

    def __str__(self):
        return f"FusionSpace([{', '.join(map(str, self.spaces))}])"

    def __eq__(self, other):
        if isinstance(other, FusionSpace):
            return self.is_dual == other.dual and self.spaces == other.spaces
        # else
        return False

    @property
    def dual(self):
        # need to flip both self.is_dual and self.spaces[:].is_dual to keep it consistent!
        res = copy.copy(self)  # works for subclasses as well
        res.is_dual = not self.is_dual
        res.spaces = [s.dual() for s in self.spaces]
        # TODO double-check/write test that the fusion of the dual goes through like this...
        return res

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
        raise NotImplementedError  # TODO
