from __future__ import annotations
from abc import ABC, abstractmethod
from math import prod

from tenpy.linalg.symmetries.groups import Symmetry, Sector, no_symmetry


class AbstractSpace(ABC):
    def __init__(self, symmetry: Symmetry, dim: int):
        self.symmetry = symmetry
        self.dim = dim

    def __mul__(self, other):
        if isinstance(other, AbstractSpace):
            return ProductSpace([self, other])
        return NotImplemented

    @property
    @abstractmethod
    def dual(self):
        ...

    @property
    @abstractmethod
    def is_trivial(self) -> bool:
        """If the space is trivial, i.e. one-dimensional with trivial grading,
        i.e. the identity object of the monoidal category"""
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    def can_contract_with(self, other):
        return self == other.dual


class VectorSpace(AbstractSpace):
    def __init__(self, symmetry: Symmetry, sectors: list[Sector], multiplicities: list[int] = None,
                 is_dual: bool = False, is_real: bool = False):
        """
        A vector space, which decomposes into sectors of a given symmetry.
        is_dual: whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.
        is_real: whether the space is over the real numbers (otherwise over the complex numbers)
        """
        self.sectors = sectors
        if multiplicities is None:
            self.multiplicities = [1 for s in sectors]
        else:
            assert len(multiplicities) == len(sectors)
            self.multiplicities = multiplicities
        self.is_dual = is_dual
        self.is_real = is_real
        dim = sum(symmetry.sector_dim(s) * m for s, m in zip(sectors, self.multiplicities))
        super().__init__(symmetry=symmetry, dim=dim)

    @classmethod
    def non_symmetric(cls, dim: int, is_dual: bool = False, is_real: bool = False):
        return cls(symmetry=no_symmetry, sectors=[None], multiplicities=[dim], is_dual=is_dual, is_real=is_real)

    def sectors_str(self) -> str:
        """short str describing the sectors and their multiplicities"""
        return ', '.join(f'{self.symmetry.sector_str(a)}: {mult}' for a, mult in zip(self.sectors, self.multiplicities))

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
        # TODO make duality shorter?
        return f'dual({res})' if self.is_dual else res

    def __eq__(self, other):
        if isinstance(other, VectorSpace):
            return self.sectors == other.sectors and self.multiplicities == other.multiplicities \
                   and self.is_dual == other.is_dual and self.is_real == other.is_real
        else:
            return False

    @property
    def dual(self):
        return VectorSpace(symmetry=self.symmetry, sectors=self.sectors, multiplicities=self.multiplicities,
                           is_dual=not self.is_dual, is_real=self.is_real)

    @property
    def is_trivial(self) -> bool:
        return len(self.sectors) == 1


class ProductSpace(AbstractSpace):
    def __init__(self, spaces: list[AbstractSpace]):
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        super().__init__(symmetry=symmetry, dim=prod(s.dim for s in spaces))

    def __len__(self):
        return len(self.spaces)

    def __iter__(self):
        yield from self.spaces

    def __repr__(self):
        return '\n'.join(('ProductSpace([', *map(repr, self.spaces), '])'))

    def __str__(self):
        return ' ⊗ '.join(map(str, self.spaces))

    def __eq__(self, other):
        if isinstance(other, ProductSpace):
            return self.spaces == other.spaces
        else:
            return False

    @property
    def dual(self):
        return ProductSpace([s.dual() for s in self.spaces])

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
        raise NotImplementedError  # TODO
