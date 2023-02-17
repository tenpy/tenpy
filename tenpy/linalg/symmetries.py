from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from math import prod
from typing import TypeVar, Final


Sector = TypeVar('Sector')  # place-holder for the type of a sector. must support comparison (for sorting) and hashing


class FusionStyle(Enum):
    single = 0  # only one resulting sector, a ⊗ b = c, eg abelian symmetry groups
    multiple_unique = 10  # every sector appears at most once in pairwise fusion, N^{ab}_c \in {0,1}
    general = 20  # no assumptions N^{ab}_c = 0, 1, 2, ...


class BraidingStyle(Enum):
    bosonic = 0  # symmetric braiding with trivial twist; v ⊗ w ↦ w ⊗ v
    fermionic = 10  # symmetric braiding with non-trivial twist; v ⊗ w ↦ (-1)^p(v,w) w ⊗ v
    anyonic = 20  # non-symmetric braiding
    no_braiding = 30  # braiding is not defined


class Symmetry(ABC):
    """Base class for symmetries that impose a block-structure on tensors"""
    def __init__(self, fusion_style: FusionStyle, braiding_style: BraidingStyle, trivial_sector: Sector,
                 group_name: str, descriptive_name: str | None = None):
        self.fusion_style = fusion_style
        self.braiding_style = braiding_style
        self.trivial_sector = trivial_sector
        self.group_name = group_name
        self.descriptive_name = descriptive_name

    @property
    def is_abelian(self) -> bool:
        if isinstance(self, ProductSymmetry):
            return all(factor.is_abelian for factor in self.factors)
        else:
            return isinstance(self, AbelianGroup)

    @abstractmethod
    def is_valid_sector(self, a: Sector) -> bool:
        """Whether `a` is a valid sector of this symmetry"""
        ...

    @abstractmethod
    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        """Returns all outcomes for the fusion of sectors.
        Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion"""
        ...

    @abstractmethod
    def sector_dim(self, a: Sector) -> int:
        """The dimension of a sector as a subspace of the hilbert space"""
        ...

    def sector_str(self, a: Sector) -> str:
        """Short and readable string for the sector. Is used in __str__ of symmetry-related objects."""
        return str(a)

    @abstractmethod
    def __repr__(self):
        # Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
        ...

    def __str__(self):
        res = self.group_name
        if self.descriptive_name is not None:
            res = res + f'("{self.descriptive_name}")'
        return res

    def __mul__(self, other):
        if isinstance(self, ProductSymmetry):
            factors = self.factors
        elif isinstance(self, Symmetry):
            factors = [self]
        else:
            return NotImplemented

        if isinstance(other, ProductSymmetry):
            factors = factors + other.factors
        elif isinstance(other, Symmetry):
            factors = factors + [other]
        else:
            return NotImplemented

        return ProductSymmetry(factors=factors)

    @abstractmethod
    def __eq__(self, other) -> bool:
        ...

    @abstractmethod
    def dual_sector(self, a: Sector) -> Sector:
        """
        The sector dual to a, such that N^{a,dual(a)}_u = 1.
        TODO: define precisely what the dual sector is.
        we want the canonical representative of its equivalence class
        """
        ...

    # TODO a bunch of methods, such as n-symbol etc which (i think) only matter for the non-abelian implementation


class NoSymmetry(Symmetry):
    """Trivial symmetry group that doesn't do anything. the only allowed sector is `None`"""

    def __init__(self):
        Symmetry.__init__(self, fusion_style=FusionStyle.single, braiding_style=BraidingStyle.bosonic,
                          trivial_sector=None, group_name='NoSymmetry', descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return a is None

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [None]

    def sector_dim(self, a: Sector) -> int:
        return 1

    def sector_str(self, a: Sector) -> int:
        return '.'

    def __repr__(self):
        return 'NoSymmetry()'

    def __eq__(self, other) -> bool:
        return isinstance(other, NoSymmetry)

    def dual_sector(self, a: Sector) -> Sector:
        return None


class ProductSymmetry(Symmetry):
    """Multiple symmetry groups. The allowed sectors are lists of sectors of the factor symmetries."""

    def __init__(self, factors: list[Symmetry]):
        self.factors = factors
        if all(f.descriptive_name is not None for f in factors):
            descriptive_name = f'[{", ".join(f.descriptive_name for f in factors)}]'
        else:
            descriptive_name = None
        Symmetry.__init__(
            self,
            fusion_style=max((f.fusion_style for f in factors), key=lambda style: style.value),
            braiding_style=max((f.braiding_style for f in factors), key=lambda style: style.value),
            trivial_sector=[f.trivial_sector for f in factors],
            group_name=' ⨉ '.join(f.group_name for f in factors),
            descriptive_name=descriptive_name
        )

    def is_valid_sector(self, a: Sector) -> bool:
        try:
            len_a = len(a)
        except TypeError:
            return False
        if len_a != len(self.factors):
            return False
        try:
            return all(f.is_valid_sector(b) for f, b in zip(self.factors, a))
        except TypeError:
            # if a is not iterable
            return False

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        # this can probably be optimized. could also special-case FusionStyle.single
        all_outcomes = (f.fusion_outcomes(a_f, b_f) for f, a_f, b_f in zip(self.factors, a, b))
        return [list(combination) for combination in product(*all_outcomes)]

    def sector_dim(self, a: Sector) -> int:
        return prod(f.sector_dim(a_f) for f, a_f in zip(self.factors, a))

    def sector_str(self, a: Sector) -> str:
        return f'[{", ".join(f.sector_str(a_f) for f, a_f in zip(self.factors, a))}]'

    def __repr__(self):
        return ' * '.join(repr(f) for f in self.factors)

    def __str__(self):
        return ' ⨉ '.join(str(f) for f in self.factors)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProductSymmetry):
            return False
        if len(self.factors) != len(other.factors):
            return False
        return all(f1 == f2 for f1, f2 in zip(self.factors, other.factors))

    def dual_sector(self, a: Sector) -> Sector:
        return [f.dual_sector(a_f) for f, a_f in zip(self.factors, a)]


# TODO: call it GroupSymmetry instead?
class Group(Symmetry, ABC):
    """
    Base-class for symmetries that are described by a group via a faithful representation on the Hilbert space.
    Noteable counter-examples are fermionic parity or anyonic grading.
    """
    def __init__(self, fusion_style: FusionStyle, trivial_sector: Sector, group_name: str, descriptive_name: str | None = None):
        Symmetry.__init__(self, fusion_style=fusion_style, braiding_style=BraidingStyle.bosonic,
                          trivial_sector=trivial_sector, group_name=group_name,
                          descriptive_name=descriptive_name)


class AbelianGroup(Group, ABC):
    """
    Base-class for abelian symmetry groups.
    Note that a product of several abelian groups is also an abelian group, but represented by a ProductSymmetry,
    which is not a subclass of AbelianGroup.
    """
    is_abelian = True

    def __init__(self, trivial_sector: Sector, group_name: str, descriptive_name: str | None = None):
        Group.__init__(self, fusion_style=FusionStyle.single, trivial_sector=trivial_sector,
                       group_name=group_name, descriptive_name=descriptive_name)

    def sector_dim(self, a: Sector) -> int:
        return 1

# TODO group_names U(1) and SU(2) or U₁ and SU₂ ?
# JH: at least consistent: if Z_N, then also U_1 and SU_2.
# Challenge: when you want to compare, you need to copy-paste

class U1Symmetry(AbelianGroup):
    """U(1) symmetry. Sectors are integers ..., `-2`, `-1`, `0`, `1`, `2`, ..."""
    def __init__(self, descriptive_name: str | None = None):
        AbelianGroup.__init__(self, trivial_sector=0, group_name='U(1)',
                              descriptive_name=descriptive_name)

    def is_valid_sector(self, a: Sector) -> bool:
        return isinstance(a, int)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [(a + b)]

    def dual_sector(self, a: Sector) -> Sector:
        return -a

    def __repr__(self):
        return 'U1Symmetry()'

    def __eq__(self, other) -> bool:
        return isinstance(other, U1Symmetry)


class ZNSymmetry(AbelianGroup):
    """Z_N symmetry. Sectors are integers `0`, `1`, ..., `N-1`"""
    def __init__(self, N: int, descriptive_name: str | None = None):
        assert isinstance(N, int)
        if not isinstance(N, int) and N > 1:
            raise ValueError(f"invalid ZNSymmetry(N={N!r},{descriptive_name!s})")
        self.N = N
        subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
                         '7': '₇', '8': '₈', '9': '₉'}
        subscript_N = ''.join(subscript_map[char] for char in str(N))
        group_name = f'ℤ{subscript_N}'
        AbelianGroup.__init__(self, trivial_sector=0, group_name=group_name,
                              descriptive_name=descriptive_name)

    def __repr__(self):
        return f'ZNSymmetry(N={self.N})'  # TODO include descriptive_name?

    def __eq__(self, other) -> bool:
        return isinstance(other, ZNSymmetry) and other.N == self.N

    def is_valid_sector(self, a: Sector) -> bool:
        return isinstance(a, int) and (0 <= a < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [(a + b) % self.N]

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N


class SU2Symmetry(Group):
    """SU(2) symmetry. Sectors are positive integers `jj` = `0`, `1`, `2`, ...
    which label the spin `jj/2` irrep of SU(2).
    This is for convenience so that we can work with `int` objects.
    E.g. a spin-1/2 degree of freedom is represented by the sector `1`.
    """
    def __init__(self, descriptive_name: str | None = None):
        Group.__init__(self, fusion_style=FusionStyle.multiple_unique, trivial_sector=0,
                       group_name='SU(2)', descriptive_name=descriptive_name)

    def is_valid_sector(self, a: Sector) -> bool:
        return isinstance(a, int) and a >= 0

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        # J_tot = |J1 - J2|, ..., J1 + J2
        return list(range(abs(a - b), a + b + 2, 2))

    def sector_dim(self, a: Sector) -> int:
        # dim = 2 * J + 1 = jj + 1
        return a + 1

    def sector_str(self, a: Sector) -> str:
        j_str = str(a // 2) if a % 2 == 0 else f'{a}/2'
        return f'J={j_str}'

    def __repr__(self):
        return 'SU2Symmetry()'

    def __eq__(self, other) -> bool:
        return isinstance(other, SU2Symmetry)

    def dual_sector(self, a: Sector) -> Sector:
        return a


# TODO: shouldn't this be a subclass of ZNsymmetry?
class FermionParity(Symmetry):
    """Fermionic Parity. Sectors are `0` (even parity) and `1` (odd parity)"""

    def __init__(self):
        Symmetry.__init__(self, fusion_style=FusionStyle.single, braiding_style=BraidingStyle.fermionic,
                          trivial_sector=0, group_name='FermionParity', descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return a in [0, 1]

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        # equal sectors fuse to even parity, i.e. to `0 == int(False)`
        # unequal sectors fuse to odd parity i.e. to `1 == int(True)`
        return int(a != b)

    def sector_dim(self, a: Sector) -> int:
        return 1

    def sector_str(self, a: Sector) -> str:
        return 'even' if a == 0 else 'odd'

    def __repr__(self):
        return 'FermionParity()'

    def __eq__(self, other) -> bool:
        return isinstance(other, FermionParity)

    def dual_sector(self, a: Sector) -> Sector:
        return a


no_symmetry: Final = NoSymmetry()
z2_symmetry: Final = ZNSymmetry(N=2)
z3_symmetry: Final = ZNSymmetry(N=3)
z4_symmetry: Final = ZNSymmetry(N=4)
z5_symmetry: Final = ZNSymmetry(N=5)
z6_symmetry: Final = ZNSymmetry(N=6)
z7_symmetry: Final = ZNSymmetry(N=7)
z8_symmetry: Final = ZNSymmetry(N=8)
z9_symmetry: Final = ZNSymmetry(N=9)
u1_symmetry: Final = U1Symmetry()
su2_symmetry: Final = SU2Symmetry()
fermion_parity: Final = FermionParity()


# TODO fibonacci anyons ...


class VectorSpace:

    def __init__(self, symmetry: Symmetry, sectors: list[Sector], multiplicities: list[int] = None,
                 is_dual: bool = False, is_real: bool = False):
        """
        A vector space, which decomposes into sectors of a given symmetry.
        is_dual: whether this is the "normal" (i.e. ket) or dual (i.e. bra) space.
        is_real: whether the space is over the real numbers (otherwise over the complex numbers)
        """
        self.symmetry = symmetry
        self.sectors = sectors
        if multiplicities is None:
            self.multiplicities = [1 for s in sectors]
        else:
            assert len(multiplicities) == len(sectors)
            self.multiplicities = multiplicities
        self.dim = sum(symmetry.sector_dim(s) * m for s, m in zip(sectors, self.multiplicities))
        self.is_dual = is_dual
        self.is_real = is_real

        # backends may write these attributes to cache metadata
        self._abelian_data = None

    @classmethod
    def non_symmetric(cls, dim: int, is_dual: bool = False, is_real: bool = False):
        return cls(symmetry=no_symmetry, sectors=[None], multiplicities=[dim], is_dual=is_dual, is_real=is_real)

    def sectors_str(self) -> str:
        """short str describing the sectors and their multiplicities"""
        return ', '.join(f'{self.symmetry.sector_str(a)}: {mult}' for a, mult in zip(self.sectors, self.multiplicities))

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
        # TODO make duality shorter?
        return f'dual({res})' if self.is_dual else res

    def __eq__(self, other):
        if isinstance(other, VectorSpace):
            # FIXME need to be more careful with is_dual flag!
            return self.sectors == other.sectors and self.multiplicities == other.multiplicities \
                   and self.is_dual == other.is_dual and self.is_real == other.is_real
        else:
            return False

    @property
    def dual(self):
        return VectorSpace(symmetry=self.symmetry, sectors=self.sectors, multiplicities=self.multiplicities,
                           is_dual=not self.is_dual, is_real=self.is_real)

    def is_dual_of(self, other):
        # FIXME think about duality in more detail.
        #  i.e. is a
        # `Vectorspace(a.symmetry, [sector.dual for sector in a.sectors], a.multiplicities, not a.is_dual, a.is_real) == a` ?
        return self == other.dual

    @property
    def is_trivial(self) -> bool:
        return len(self.sectors) == 1

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
        raise NotImplementedError  # TODO


class ProductSpace(VectorSpace):
    def __init__(self, spaces: list[VectorSpace], is_dual: bool = False):
        self.spaces = spaces  # spaces can be themselves ProductSpaces
        symmetry = spaces[0].symmetry
        assert all(s.symmetry == symmetry for s in spaces)
        sectors = [list(combination) for combination in product(*(space.sectors for space in spaces))]
        multiplicities = [prod(combination) for combination in product(*(space.multiplicities for space in spaces))]
        is_real = spaces[0].is_real
        assert all(space.is_real == is_real for space in spaces)

        VectorSpace.__init__(self, symmetry=symmetry, sectors=sectors, multiplicities=multiplicities,
                             is_dual=is_dual, is_real=is_real)

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
            # FIXME need to be more careful about is_dual flags!
            return self.spaces == other.spaces
        else:
            return False

    @property
    def dual(self):
        # TODO should this just change self.is_dual instead...?
        return ProductSpace([s.dual() for s in self.spaces])

    @property
    def is_trivial(self) -> bool:
        return all(s.is_trivial for s in self.spaces)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors within this space"""
        raise NotImplementedError  # TODO
