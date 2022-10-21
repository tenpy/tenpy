"""
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Final

Sector = TypeVar('Sector')  # place-holder for the type of a sector. must support comparison (for sorting) and hashing


class FusionStyle(Enum):
    single = 0  # only one resulting sector, a ⊗ b = c, eg abelian symmetry groups
    multiple_unique = 1  # every sector appears at most once in pairwise fusion, N^{ab}_c \in {0,1}
    general = 2


class BraidingStyle(Enum):
    bosonic = 0  # symmetric braiding with trivial twist; v ⊗ w ↦ w ⊗ v
    fermionic = 1  # symmetric braiding with non-trivial twist; v ⊗ w ↦ (-1)^p(v,w) w ⊗ v
    anyonic = 2  # non-symmetric braiding
    no_braiding = 3  # braiding is not defined


class AbstractSymmetry(ABC):
    fusion_style: FusionStyle
    braiding_style: BraidingStyle
    trivial_sector: Sector
    group_ascii: str

    def __init__(self, name: str | None = None):
        """name: A descriptive name for the (quantity conserved by the) symmetry, e.g. 'Sz', 'ky', 'S_tot'"""
        self.name = name

    @abstractmethod
    def short_str(self):
        """short string representation of the symmetry, without self.name"""
        ...

    @abstractmethod
    def is_valid_sector(self, a) -> bool:
        """Whether `a` is a valid sector of this symmetry"""
        ...

    @abstractmethod
    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        """Returns all outcomes for the fusion of sectors.
        Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion"""
        ...

    @abstractmethod
    def sector_dim(self, a: Sector) -> int:
        """The dimension of a given sector"""
        ...

    @abstractmethod
    def sector_str(self, a: Sector) -> str:
        """Short and readable string for the sector. To be used only in the context of self."""
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def __repr__(self):
        # Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
        ...

    def __str__(self):
        descriptor = '' if self.name is None else f'("{self.name}")'
        return f'{self.short_str()}{descriptor}'

    def __mul__(self, other):
        return ProductSymmetry([self, other])

    # TODO a bunch of methods, such as n-symbol etc which (i think) only matter for the non-abelian implementation


class NoSymmetry(AbstractSymmetry):
    """Trivial symmetry group that doesn't do anything. the only allowed sector is `None`"""
    fusion_style = FusionStyle.single
    braiding_style = BraidingStyle.bosonic
    trivial_sector = None

    def __init__(self):
        super().__init__(name=None)

    def short_str(self):
        return 'NoSymmetry'

    def sector_str(self, a: Sector) -> str:
        return '.'

    def is_valid_sector(self, a) -> bool:
        return a is None

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [None]

    def sector_dim(self, a: Sector) -> int:
        return 1

    def __eq__(self, other):
        return isinstance(other, NoSymmetry)

    def __repr__(self):
        return 'NoSymmetry()'


class SymmetryGroup(AbstractSymmetry, ABC):
    """Base-class that defines common features of a symmetry given
    by a faithful representation of a group"""
    braiding_style = BraidingStyle.bosonic


class AbelianSymmetryGroup(SymmetryGroup, ABC):
    """Base-class that defines common features of abelian symmetry groups"""
    fusion_style = FusionStyle.single

    def sector_dim(self, a: Sector) -> int:
        return 1


class U1Symmetry(AbelianSymmetryGroup):
    """U(1) symmetry. Sectors are integers ..., -1, 0, 1, ... ."""
    trivial_sector = 0

    def short_str(self):
        return 'U₁'

    def sector_str(self, a: Sector) -> str:
        return str(a)

    def is_valid_sector(self, a) -> bool:
        return isinstance(a, int)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [a + b]

    def __eq__(self, other):
        return isinstance(other, U1Symmetry)

    def __repr__(self):
        name_arg = '' if self.name is None else f'name="{self.name}"'
        return f'U1Symmetry({name_arg})'


class ZNSymmetry(AbelianSymmetryGroup):
    """Z_N symmetry. Sectors are integers 0 <= m < N"""
    trivial_sector = 0

    def __init__(self, N: int, name: str = None):
        assert N > 1
        self.N = N
        super().__init__(name=name)

    def short_str(self):
        subscript_map = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
                         "7": "₇", "8": "₈", "9": "₉"}
        N_subscript = ''.join(subscript_map[char] for char in str(self.N))
        return f'Z{N_subscript}'

    def sector_str(self, a: Sector) -> str:
        return str(a)

    def is_valid_sector(self, a) -> bool:
        return isinstance(a, int) and 0 <= a < self.N

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [(a + b) % self.N]

    def __eq__(self, other):
        return isinstance(other, ZNSymmetry) and self.N == other.N

    def __repr__(self):
        name_arg = '' if self.name is None else f', "{self.name}"'
        return f'ZNSymmetry({self.N}{name_arg})'


class SU2Symmetry(SymmetryGroup):
    """SU(2) symmetry. Allowed sectors are non-negative integers j = 0, 1, 2... labelling the spin j/2 irrep."""
    fusion_style = FusionStyle.multiple_unique
    trivial_sector = 0

    def short_str(self):
        return 'SU₂'

    def sector_str(self, a: Sector) -> str:
        return str(a // 2) if a % 2 == 0 else f'{a}/2'

    def is_valid_sector(self, a) -> bool:
        # Sector values represent twice the spin quantum number, e.g. the spin one-half sector has a=1
        return isinstance(a, int) and a >= 0

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return list(range(a, b + 2, 2))

    def sector_dim(self, a: Sector) -> int:
        return a + 1

    def __eq__(self, other):
        return isinstance(other, SU2Symmetry)

    def __repr__(self):
        name_arg = '' if self.name is None else f'name="{self.name}"'
        return f'SU2Symmetry({name_arg})'


class FermionParity(AbstractSymmetry):
    """Z2 grading induced by fermionic parity. Allowed sectors are False (even parity) and True (odd parity)"""
    fusion_style = FusionStyle.single
    braiding_style = BraidingStyle.fermionic
    trivial_sector = False

    def __init__(self):
        super().__init__(name=None)  # dont need a name, its always parity of fermion occupation

    def short_str(self):
        return 'FermionParity'

    def sector_str(self, a: Sector) -> str:
        return 'odd' if a else 'even'

    def is_valid_sector(self, a) -> bool:
        # sectors label fermionic parity, i.e. False is the even/bosonic sector, True the fermionic/odd one
        return isinstance(a, bool)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        # equal sectors fuse to bosonic sector, i.e. False
        return [a != b]

    def sector_dim(self, a: Sector) -> int:
        return 1

    def __eq__(self, other):
        return isinstance(other, FermionParity)

    def __repr__(self):
        return 'FermionParity()'


# TODO which other symmetries could we need?


class ProductSymmetry(AbstractSymmetry):
    """
    Product of several symmetries. Allowed sectors are lists, whose entries are sectors of the factor symmetries.
    ProductSymmetry instances can also be obtained via the multiplication `*` of AbstractSymmetry instances.
    """

    def __init__(self, symmetries: list[AbstractSymmetry]):
        # enforce associativity. i.e product of products is a single product
        while any(isinstance(sym, ProductSymmetry) for sym in symmetries):
            _symmetries = []
            for sym in symmetries:
                if isinstance(sym, ProductSymmetry):
                    _symmetries.extend(sym.symmetries)
                else:
                    _symmetries.append(sym)
            symmetries = _symmetries
        self.symmetries = symmetries
        self.num_factors = len(symmetries)
        self.fusion_style = FusionStyle[max(s.fusion_style.value for s in symmetries)]
        self.braiding_style = BraidingStyle[max(s.braiding_style.value for s in symmetries)]
        self.trivial_sector = [s.trivial_sector for s in symmetries]
        super().__init__(name=None)  # (the individual symmetries have names)

    def short_str(self):
        return ' × '.join(s.short_str() for s in self.symmetries)

    def sector_str(self, a: Sector) -> str:
        return str([s.sector_str(a_s) for s, a_s in zip(self.symmetries, a)])

    def is_valid_sector(self, a) -> bool:
        # charges are lists, entries are charges of the factor-symmetries
        return isinstance(a, list) \
               and len(a) == self.num_factors \
               and all(s.is_valid_sector(a_s) for a_s, s in zip(a, self.symmetries))

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        raise NotImplementedError  # TODO think in detail what the sectors are!
        # results = product(s.fusion_outcomes(a_s, b_s) for s, a_s, b_s in zip(self.symmetries, a, b))
        # # results has the relevant data, but it is a generator of tuples, want a list of lists
        # return list(map(list, results))

    def sector_dim(self, a: Sector) -> int:
        raise NotImplemented  # TODO think in detail what the sectors are!

    def __eq__(self, other):
        return isinstance(other, ProductSymmetry) and self.symmetries == other.symmetries

    def __str__(self):
        return ' × '.join(map(str, self.symmetries))

    def __repr__(self):
        return f'ProductSymmetry([{", ".join(map(repr, self.symmetries))}])'

    def __len__(self):
        return self.num_factors

    def __iter__(self):
        yield from self.symmetries


no_symmetry: Final = NoSymmetry()
u1_symmetry: Final = U1Symmetry()
z2_symmetry: Final = ZNSymmetry(2)
z3_symmetry: Final = ZNSymmetry(3)
z4_symmetry: Final = ZNSymmetry(4)
su2_symmetry: Final = SU2Symmetry()
fermion_parity: Final = FermionParity()
