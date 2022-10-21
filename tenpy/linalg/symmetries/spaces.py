from __future__ import annotations
from abc import ABC
from math import prod

from tenpy.linalg.symmetries.groups import AbstractSymmetry, Sector, no_symmetry


class AbstractSpace(ABC):
    def __init__(self, symmetry: AbstractSymmetry, dim: int):
        self.symmetry = symmetry
        self.dim = dim


class VectorSpace(AbstractSpace):
    def __init__(self, symmetry: AbstractSymmetry, sector_list: list[Sector],
                 multiplicity_list: list[int] = None, is_dual: bool = False,
                 is_real: bool = False):
        """
        A vector space, which decomposes into sectors of given symmetry.
        conj: whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.
        is_real: whether the space is real or complex
        """
        self.sector_list = sector_list
        if multiplicity_list is None:
            self.multiplicity_list = [1 for s in sector_list]
        else:
            assert len(multiplicity_list) == len(sector_list)
            self.multiplicity_list = multiplicity_list
        self.is_dual = is_dual
        self.is_real = is_real
        dim = sum(symmetry.sector_dim(s) * m for s, m in zip(sector_list, self.multiplicity_list))
        super().__init__(symmetry=symmetry, dim=dim)

    @classmethod
    def non_symmetric(cls, dim: int, is_dual: bool = False, is_real: bool = False):
        return cls(symmetry=no_symmetry, sector_list=[None], multiplicity_list=[dim], is_dual=is_dual, is_real=is_real)

    def __mul__(self, other):
        if isinstance(other, VectorSpace):
            return ProductSpace([self, other])
        if isinstance(other, ProductSpace):
            return ProductSpace([self, *other.spaces])
        return NotImplemented


class ProductSpace(AbstractSpace):
    def __init__(self, spaces: list[AbstractSpace]):
        self.spaces = spaces
        super().__init__(symmetry=spaces[0].symmetry, dim=prod(s.dim for s in spaces))

    def __mul__(self, other):
        if isinstance(other, VectorSpace):
            return ProductSpace([*self.spaces, other])
        if isinstance(other, ProductSpace):
            return ProductSpace([*self.spaces, *other.spaces])
        return NotImplemented
