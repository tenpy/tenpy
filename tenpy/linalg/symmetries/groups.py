from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from typing import TypeVar


Sector = TypeVar('Sector')  # place-holder for the type of a sector.


class FusionStyle(Enum):
    single = 0  # only one resulting sector, a ⊗ b = c
    multiple_unique = 1  # every sector appears at most once in pairwise fusion
    general = 2


class BraidingStyle(Enum):
    bosonic = 0  # symmetric braiding with trivial twist
    fermionic = 1  # symmetric braiding with non-trivial twist
    anyonic = 2  # non-symmetric braiding
    no_braiding = 3  # braiding is not defined


class AbstractSymmetry(ABC):
    fusion_style: FusionStyle
    braiding_style: BraidingStyle
    trivial_sector: Sector

    def __init__(self, name: str | None):
        """
        A descriptive name for the symmetry, e.g. 'Sz', 'ky', 'S_tot'
        """
        self.name = name

    @abstractmethod
    def is_valid_charge(self, a) -> bool:
        """Whether `a` is a valid sector of this symmetry"""
        ...

    @abstractmethod
    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        """Returns all outcomes for the fusion of sectors.
        Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion"""
        ...

    # TODO a bunch of methods, such as n-symbol etc which (i think) only matter for the non-abelian implementation


class NoSymmetry(AbstractSymmetry):
    """Trivial symmetry group that doesn't do anything. the only allowed sector is `None`"""
    fusion_style = FusionStyle.single
    braiding_style = BraidingStyle.bosonic
    trivial_sector = None

    def __init__(self):
        super().__init__(name=None)

    def is_valid_charge(self, a) -> bool:
        return a is None

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [None]


class SymmetryGroup(AbstractSymmetry, ABC):
    """Base-class that defines common features of a symmetry with group structure"""
    braiding_style = BraidingStyle.bosonic


class AbelianSymmetryGroup(SymmetryGroup, ABC):
    """Base-class that defines common features of abelian symmetry groups"""
    fusion_style = FusionStyle.single


class U1(AbelianSymmetryGroup):
    """U(1) symmetry. Sectors are integers ..., -1, 0, 1, ... ."""
    trivial_sector = 0

    def is_valid_charge(self, a) -> bool:
        return isinstance(a, int)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [a + b]


class ZN(AbelianSymmetryGroup):
    """Z_N symmetry. Sectors are integers 0 <= m < N"""
    trivial_sector = 0

    def __init__(self, N: int, name: str = None):
        self.N = N
        super().__init__(name=name)

    def is_valid_charge(self, a) -> bool:
        return isinstance(a, int) and 0 <= a < self.N

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return [(a + b) % self.N]


class SU2(SymmetryGroup):
    """SU(2) symmetry. Allowed sectors are non-negative integers j = 0, 1, 2... labelling the spin j/2 irrep."""
    fusion_style = FusionStyle.multiple_unique
    trivial_sector = 0

    def is_valid_charge(self, a) -> bool:
        # charges represent twice the spin quantum number, e.g. the spin one-half sector has charge a=1
        return isinstance(a, int) and a >= 0

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        return list(range(a, b + 2, 2))


class FermionicGrading(AbstractSymmetry):
    """Z2 grading induced by fermionic parity. Allowed sectors are False (even parity) and True (odd parity)"""
    fusion_style = FusionStyle.single
    braiding_style = BraidingStyle.fermionic
    trivial_sector = False

    def __init__(self):
        super().__init__(name=None)  # dont need a name, its always parity of fermion occupation

    def is_valid_charge(self, a) -> bool:
        # sectors label fermionic parity, i.e. False is the bosonic sector, True the fermionic one
        return isinstance(a, bool)

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        # equal sectors fuse to bosonic sector, i.e. False
        return [a != b]


class ProductSymmetry(AbstractSymmetry):
    """Product of several symmetries. Allowed sectors are lists, whose entries are sectors of the factor symmetries."""
    def __init__(self, symmetries: list[AbstractSymmetry]):
        self.symmetries = symmetries
        self.num_factors = len(symmetries)
        self.fusion_style = FusionStyle[max(s.fusion_style.value for s in symmetries)]
        self.braiding_style = BraidingStyle[max(s.braiding_style.value for s in symmetries)]
        self.trivial_sector = [s.trivial_sector for s in symmetries]
        super().__init__(name=None)  # (the individual symmetries have names)

    def is_valid_charge(self, a) -> bool:
        # charges are lists, entries are charges of the factor-symmetries
        return isinstance(a, list) \
               and len(a) == self.num_factors \
               and all(s.is_valid_charge(a_s) for a_s, s in zip(a, self.symmetries))

    def fusion_outcomes(self, a: Sector, b: Sector) -> list[Sector]:
        results = product(s.fusion_outcomes(a_s, b_s) for s, a_s, b_s in zip(self.symmetries, a, b))
        # results has the relevant data, but it is a generator of tuples, want a list of lists
        return list(map(list, results))
