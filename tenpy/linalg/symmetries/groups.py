# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import TypeVar, Iterator
from itertools import product, count
from numpy import typing as npt
import numpy as np


__all__ = ['Sector', 'FusionStyle', 'BraidingStyle', 'Symmetry', 'NoSymmetry', 'ProductSymmetry',
           'GroupSymmetry', 'AbelianGroup', 'U1Symmetry', 'ZNSymmetry', 'SU2Symmetry', 'FermionParity',
           'no_symmetry', 'z2_symmetry', 'z3_symmetry', 'z4_symmetry', 'z5_symmetry', 'z6_symmetry',
           'z7_symmetry', 'z8_symmetry', 'z9_symmetry', 'u1_symmetry', 'su2_symmetry', 'fermion_parity',
           ]


Sector = npt.NDArray[np.int_] # 1D array, axis [q], containing the an integer representation of the charges (e.g. one per "conservation law")
SectorArray = npt.NDArray[np.int_]  # 2D array, axes [s, q], where s goes over different sectors


class FusionStyle(Enum):
    single = 0  # only one resulting sector, a ⊗ b = c, e.g. abelian symmetry groups
    multiple_unique = 10  # every sector appears at most once in pairwise fusion, N^{ab}_c \in {0,1}
    general = 20  # no assumptions N^{ab}_c = 0, 1, 2, ...


class BraidingStyle(Enum):
    bosonic = 0  # symmetric braiding with trivial twist; v ⊗ w ↦ w ⊗ v
    fermionic = 10  # symmetric braiding with non-trivial twist; v ⊗ w ↦ (-1)^p(v,w) w ⊗ v
    anyonic = 20  # non-symmetric braiding
    no_braiding = 30  # braiding is not defined


class Symmetry(metaclass=ABCMeta):
    """Base class for symmetries that impose a block-structure on tensors"""
    def __init__(self, fusion_style: FusionStyle, braiding_style: BraidingStyle, trivial_sector: Sector,
                 group_name: str, num_sectors: int | float, descriptive_name: str | None = None):
        self.fusion_style = fusion_style
        self.braiding_style = braiding_style
        self.trivial_sector = trivial_sector
        self.group_name = group_name
        self.num_sectors = num_sectors
        self.descriptive_name = descriptive_name
        self.sector_ind_len = len(trivial_sector)  # how many entries are needed to describe a Sector
        self.is_abelian = (fusion_style == FusionStyle.single)

    @abstractmethod
    def is_valid_sector(self, a: Sector) -> bool:
        """Whether `a` is a valid sector of this symmetry"""
        ...

    @abstractmethod
    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        """Returns all outcomes for the fusion of sectors

        Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion
        """
        ...

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        """This method allows optimized fusion in the case of FusionStyle.single.

        For two SectorArrays, return the element-wise fusion outcome of each pair of Sectors,
        which is a single unique Sector, as a new SectorArray.
        Subclasses may override this with more efficient implementations.
        """
        assert self.fusion_style == FusionStyle.single
        # self.fusion_outcomes(s_a, s_b) is a 2D array with with shape [1, num_q]
        # stack the outcomes along the trivial first axis
        return np.concatenate([self.fusion_outcomes(s_a, s_b) for s_a, s_b in zip(a, b)], axis=0)

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        """Whether c is a valid fusion outcome, i.e. if it appears in ``self.fusion_outcomes(a, b)``"""
        return np.any(np.all(self.fusion_outcomes(a, b) == c[None, :], axis=1))

    @abstractmethod
    def sector_dim(self, a: Sector) -> int:
        """The dimension of a sector as a subspace of the hilbert space"""
        ...

    def batch_sector_dim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        """sector_dim of every sector (row) in a"""
        if self.fusion_style == FusionStyle.single:
            return np.ones([a.shape[0]], dtype=int)
        return np.array([self.sector_dim(s) for s in a])

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

    def __eq__(self, other):
        if not isinstance(other, Symmetry):
            return False

        if self.descriptive_name != other.descriptive_name:
            return False

        return self.is_same_symmetry(other)

    @abstractmethod
    def is_same_symmetry(self, other) -> bool:
        """whether self and other describe the same mathematical structure.
        descriptive_name is ignored.
        """
        ...

    @abstractmethod
    def dual_sector(self, a: Sector) -> Sector:
        """The sector dual to a, such that N^{a,dual(a)}_u = 1."""
        ...

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        """dual_sector for multiple sectors

        subclasses my override this.
        """
        return np.stack([self.dual_sector(s) for s in sectors])

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """The N-symbol N^{ab}_c, i.e. how often c appears in the fusion of a and b."""
        if not self.can_fuse_to(a, b, c):
            return 0
        return self._n_symbol(a, b, c)

    @abstractmethod
    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.
        If it is not, the results (which should be 0), may be nonsensical.
        We do this for optimization purposes
        """
        ...

    def all_sectors(self) -> SectorArray:
        """If there are finitely many sectors, return all of them. Else raise a ValueError."""
        if self.num_sectors == np.inf:
            msg = f'{type(self)} has infinitely many sectors.'
            raise ValueError(msg)

        raise NotImplementedError


class ProductSymmetry(Symmetry):
    """Multiple symmetries.

    The allowed sectors are "stacks" (using e.g. :func:`numpy.concatenate`) of sectors for the
    individual symmetries. For recoveering the individual sectors see :attr:`sector_slices`.


    Attributes
    ----------
    factors : list of `Symmetry`
        The individual symmetries. We do not allow nesting, i.e. the `factors` can not
        be ``ProductSymmetry``s themselves.
    sector_slices : 1D ndarray
        Describes how the sectors of the `factors` are embedded in a sector of the product.
        Indicates that the slice ``sector_slices[i]:sector_slices[i + 1]`` of a sector of the
        product symmetry contains the entries of a sector of `factors[i]`.

    Parameters
    ----------
    factors : list of `Symmetry`
        The factors that comprise this symmetry. If any are `ProductSymmetries`, the
        nesting is flattened, i.e. ``[*others, psymm]`` is equivalent to ``[*others, psymm.factors]``
        for a `ProductSymmetry` ``psymm``.
    """
    def __init__(self, factors: list[Symmetry]):
        syms = []
        for f in factors:
            if isinstance(f, ProductSymmetry):
                syms.extend(f.factors)
            else:
                syms.append(f)
        self.factors = syms
        for f in syms:
            assert not isinstance(f, ProductSymmetry)  # avoid unnecesary nesting
        if all(f.descriptive_name is not None for f in syms):
            descriptive_name = f'[{", ".join(f.descriptive_name for f in syms)}]'
        else:
            descriptive_name = None
        self.sector_slices = np.cumsum([0] + [f.sector_ind_len for f in syms])
        Symmetry.__init__(
            self,
            fusion_style=max((f.fusion_style for f in factors), key=lambda style: style.value),
            braiding_style=max((f.braiding_style for f in factors), key=lambda style: style.value),
            trivial_sector=np.concatenate([f.trivial_sector for f in factors]),
            group_name=' ⨉ '.join(f.group_name for f in factors),
            num_sectors=np.prod([symm.num_sectors for symm in factors]),
            descriptive_name=descriptive_name
        )

    def is_valid_sector(self, a: Sector) -> bool:
        if getattr(a, 'shape', ()) != (self.sector_ind_len,):
            return False
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            if not f_i.is_valid_sector(a_i):
                return False
        return True

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        colon = slice(None, None, None)
        all_outcomes = []
        num_possibilities = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            b_i = b[self.sector_slices[i]:self.sector_slices[i + 1]]
            c_i = f_i.fusion_outcomes(a_i, b_i)
            all_outcomes.append(c_i)
            num_possibilities.append(c_i.shape[0])

        # form an array of all combinations of the c_i
        # e.g. if we have 3 factors, we want
        # result[n1, n2, n3, :] = np.concatenate([c_1[n1, :], c_2[n2, :], c_3[n3, :]], axis=-1)
        # we set the following elements:
        #
        # |                                                       i-th axis
        # |                                                       v
        # | results[:, :, ..., :, slice_i] = c_i[None, None, ..., :, ..., None, :]
        #
        c_dtype = np.find_common_type([], [a.dtype, b.dtype])
        result = np.zeros(num_possibilities + [self.sector_ind_len], dtype=c_dtype)
        for i, c_i in enumerate(all_outcomes):
            res_idx = (colon,) * len(self.factors) + (slice(self.sector_slices[i], self.sector_slices[i + 1], None),)
            c_i_idx = (None,) * i + (colon,) + (None,) * (len(self.factors) - i - 1) + (colon,)
            result[res_idx] = c_i[c_i_idx]

        # now reshape so that we get a 2D array where the first index (axis=0) runs over all those
        # combinations
        result = np.reshape(result, (np.prod(num_possibilities), self.sector_ind_len))
        return result

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        assert self.fusion_style == FusionStyle.single
        components = []
        for i, f_i in enumerate(self.factors):
            a_i = a[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            b_i = b[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            c_i = f_i.fusion_outcomes_broadcast(a_i, b_i)
            components.append(c_i)
        # the c_i have the same first axis as a and b.
        # it remains to concatenate them along the last axis
        return np.concatenate(components, axis=-1)

    def sector_dim(self, a: Sector) -> int:
        if self.fusion_style == FusionStyle.single:
            return 1

        dims = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            dims.append(f_i.sector_dim(a_i))
        return np.prod(dims)

    def batch_sector_dim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        if self.fusion_style == FusionStyle.single:
            return np.ones([a.shape[0]], dtype=int)
        dims = []
        for i, f_i in enumerate(self.factors):
            a_i = a[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            dims.append(f_i.batch_sector_dim(a_i))
        return np.prod(dims, axis=0)

    def sector_str(self, a: Sector) -> str:
        strs = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            strs.append(f_i.sector_str(a_i))
        return f'[{", ".join(strs)}]'

    def __repr__(self):
        return ' * '.join(repr(f) for f in self.factors)

    def __str__(self):
        return ' ⨉ '.join(str(f) for f in self.factors)

    def __eq__(self, other):
        if not isinstance(other, ProductSymmetry):
            return False

        if len(self.factors) != len(other.factors):
            return False

        return all(f1 == f2 for f1, f2 in zip(self.factors, other.factors))

    def is_same_symmetry(self, other) -> bool:
        if not isinstance(other, ProductSymmetry):
            return False
        if len(self.factors) != len(other.factors):
            return False
        return all(f1.is_same_symmetry(f2) for f1, f2 in zip(self.factors, other.factors))

    def dual_sector(self, a: Sector) -> Sector:
        components = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            components.append(f_i.dual_sector(a_i))
        return np.concatenate(components)

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        components = []
        for i, f_i in enumerate(self.factors):
            sectors_i = sectors[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            components.append(f_i.dual_sectors(sectors_i))
        return np.concatenate(components, axis=-1)

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        if self.fusion_style in [FusionStyle.single, FusionStyle.multiple_unique]:
            return 1

        contributions = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            b_i = b[self.sector_slices[i]:self.sector_slices[i + 1]]
            c_i = c[self.sector_slices[i]:self.sector_slices[i + 1]]
            contributions.append(f_i._n_symbol(a_i, b_i, c_i))
        return np.prod(contributions)

    def all_sectors(self) -> SectorArray:
        if self.num_sectors == np.inf:
            msg = f'{self} has infinitely many sectors.'
            raise ValueError(msg)

        # construct like in fusion_outcomes
        colon = slice(None, None, None)
        results_shape = [f.num_sectors for f in self.factors] + [self.sector_ind_len]
        results = np.zeros(results_shape, dtype=self.trivial_sector.dtype)
        for i, f_i in enumerate(self.factors):
            lhs_idx = (colon,) * len(self.factors) + (slice(self.sector_slices[i], self.sector_slices[i + 1], None),)
            rhs_idx = (None,) * i + (colon,) + (None,) * (len(self.factors) - i - 1) + (colon,)
            results[lhs_idx] = f_i.all_sectors()[rhs_idx]
        return np.reshape(results, (np.prod(results_shape[:-1]), results_shape[-1]))


class _ABCFactorSymmetryMeta(ABCMeta):
    """Metaclass for the AbstractBaseClasses which can be factors of a ProductSymmetry.

    For concreteness let FactorSymmetry be such a class.
    This metaclass, in addition to having the same effects as making the class an AbstractBaseClass
    modifies instancecheck, such that products of FactorSymmetry instances, which are instances
    of ProductSymmetry, not of FactorSymmetry, do appear like instances of FactorSymmetry

    E.g. a ProductSymmetry instance whose factors are all instances of AbelianGroup
    then appears to also be an instance of AbelianGroup
    """

    def __instancecheck__(cls, instance) -> bool:
        if (cls == GroupSymmetry or cls == AbelianGroup) and \
                type.__instancecheck__(ProductSymmetry, instance):
            return all(type.__instancecheck__(cls, factor) for factor in instance.factors)
        return type.__instancecheck__(cls, instance)


class GroupSymmetry(Symmetry, metaclass=_ABCFactorSymmetryMeta):
    """
    Base-class for symmetries that are described by a group via a faithful representation on the Hilbert space.
    Noteable counter-examples are fermionic parity or anyonic grading.
    """
    def __init__(self, fusion_style: FusionStyle, trivial_sector: Sector, group_name: str,
                 num_sectors: int | float, descriptive_name: str | None = None):
        Symmetry.__init__(self, fusion_style=fusion_style, braiding_style=BraidingStyle.bosonic,
                          trivial_sector=trivial_sector, group_name=group_name, num_sectors=num_sectors,
                          descriptive_name=descriptive_name)


class AbelianGroup(GroupSymmetry, metaclass=_ABCFactorSymmetryMeta):
    """
    Base-class for abelian symmetry groups.
    Note that a product of several abelian groups is also an abelian group, but represented by a ProductSymmetry,
    which is not a subclass of AbelianGroup.
    """

    def __init__(self, trivial_sector: Sector, group_name: str, num_sectors: int | float,
                 descriptive_name: str | None = None):
        GroupSymmetry.__init__(self, fusion_style=FusionStyle.single, trivial_sector=trivial_sector,
                       group_name=group_name, num_sectors=num_sectors, descriptive_name=descriptive_name)

    def sector_str(self, a: Sector) -> str:
        # we know sectors are labelled by a single number
        return str(a.item())
    
    def sector_dim(self, a: Sector) -> int:
        return 1

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1


class NoSymmetry(AbelianGroup):
    """Trivial symmetry group that doesn't do anything.

    The only allowed sector is `[0]` of integer dtype.
    """

    def __init__(self):
        AbelianGroup.__init__(self, trivial_sector=np.array([0], dtype=int), group_name='NoSymmetry',
                              num_sectors=1, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and a[0] == 0

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return a[np.newaxis, :]

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return a

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def sector_str(self, a: Sector) -> str:
        return '0'

    def __repr__(self):
        return 'NoSymmetry()'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, NoSymmetry)

    def all_sectors(self) -> SectorArray:
        return self.trivial_sector[np.newaxis, :]


class U1Symmetry(AbelianGroup):
    """U(1) symmetry.

    Allowed sectors are 1D arrays with a single integer entry.
    ..., `[-2]`, `[-1]`, `[0]`, `[1]`, `[2]`, ...
    """
    def __init__(self, descriptive_name: str | None = None):
        AbelianGroup.__init__(self, trivial_sector=np.array([0], dtype=int), group_name='U(1)',
                              num_sectors=np.inf, descriptive_name=descriptive_name)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return a + b

    def dual_sector(self, a: Sector) -> Sector:
        return -a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return -sectors

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'U1Symmetry({name_str})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, U1Symmetry)


class ZNSymmetry(AbelianGroup):
    """Z_N symmetry.

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`
    """
    def __init__(self, N: int, descriptive_name: str | None = None):
        assert isinstance(N, int)
        if not isinstance(N, int) and N > 1:
            raise ValueError(f"invalid ZNSymmetry(N={N!r},{descriptive_name!s})")
        self.N = N
        subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
                         '7': '₇', '8': '₈', '9': '₉'}
        subscript_N = ''.join(subscript_map[char] for char in str(N))
        group_name = f'ℤ{subscript_N}'
        AbelianGroup.__init__(self, trivial_sector=np.array([0], dtype=int), group_name=group_name,
                              num_sectors=N, descriptive_name=descriptive_name)

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f', "{self.descriptive_name}"'
        return f'ZNSymmetry({self.N}{name_str})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, ZNSymmetry) and other.N == self.N

    def is_valid_sector(self, a: Sector) -> bool:
        return (getattr(a, 'shape', ()) == (1,)) and (0 <= a[0] < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class SU2Symmetry(GroupSymmetry):
    """SU(2) symmetry.

    Allowed sectors are 1D arrays ``[jj]`` of positive integers `jj` = `0`, `1`, `2`, ...
    which label the spin `jj/2` irrep of SU(2).
    This is for convenience so that we can work with `int` objects.
    E.g. a spin-1/2 degree of freedom is represented by the sector `[1]`.
    """

    def __init__(self, descriptive_name: str | None = None):
        GroupSymmetry.__init__(self, fusion_style=FusionStyle.multiple_unique, trivial_sector=np.array([0], dtype=int),
                       group_name='SU(2)', num_sectors=np.inf, descriptive_name=descriptive_name)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and a[0] >= 0

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        # J_tot = |J1 - J2|, ..., J1 + J2
        return np.arange(np.abs(a - b), a + b + 2, 2)[:, np.newaxis]

    def sector_dim(self, a: Sector) -> int:
        # dim = 2 * J + 1 = jj + 1
        return a[0] + 1

    def batch_sector_dim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        # dim = 2 * J + 1 = jj + 1
        return a[:, 0] + 1

    def sector_str(self, a: Sector) -> str:
        jj = a[0]
        j_str = str(jj // 2) if jj % 2 == 0 else f'{jj}/2'
        return f'{jj} (J={j_str})'

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'SU2Symmetry({name_str})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, SU2Symmetry)

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        raise NotImplementedError  # TODO port su2calc


class FermionParity(Symmetry):
    """Fermionic Parity.

    Allowed sectors are 1D arrays with a single entry of either `0` (even parity) or `1` (odd parity).
    `[0]`, `[1]`
    """

    def __init__(self):
        Symmetry.__init__(self, fusion_style=FusionStyle.single, braiding_style=BraidingStyle.fermionic,
                          trivial_sector=np.array([0], dtype=int), group_name='FermionParity',
                          num_sectors=2, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and (a[0] in [0, 1])

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        # equal sectors fuse to even parity, i.e. to `0 == (0 + 0) % 2 == (1 + 1) % 2`
        # unequal sectors fuse to odd parity i.e. to `1 == (0 + 1) % 2 == (1 + 0) % 2`
        return (a + b) % 2

    def sector_dim(self, a: Sector) -> int:
        return 1

    def sector_str(self, a: Sector) -> str:
        return 'even' if a[0] == 0 else 'odd'

    def __repr__(self):
        return 'FermionParity()'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, FermionParity)

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]


class FibonacciGrading(Symmetry):
    """Grading of Fibonacci anyons

    Allowed sectors are 1D arrays with a single entry of either `0` ("vacuum") or `1` ("tau anyon").
    `[0]`, `[1]`
    """

    _fusion_map = {  # key: number of tau in fusion input
        0: np.array([[0]]),  # 1 x 1 = 1
        1: np.array([[1]]),  # 1 x t = t = t x 1
        2: np.array([[0], [1]]),  # t x t = 1 + t
    }
    _phi = .5 * (1 + np.sqrt(5))  # the golden ratio

    def __init__(self):
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.multiple_unique,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='FibonacciGrading',
                          num_sectors=2, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and (a[0] in [0, 1])

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0] + b[0]]

    def sector_dim(self, a: Sector) -> int:
        return 1 if a[0] == 0 else self._phi

    def sector_str(self, a: Sector) -> str:
        return 'vac' if a[0] == 0 else 'tau'

    def __repr__(self):
        return 'FibonacciGrading()'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, FibonacciGrading)

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]


no_symmetry = NoSymmetry()
z2_symmetry = ZNSymmetry(N=2)
z3_symmetry = ZNSymmetry(N=3)
z4_symmetry = ZNSymmetry(N=4)
z5_symmetry = ZNSymmetry(N=5)
z6_symmetry = ZNSymmetry(N=6)
z7_symmetry = ZNSymmetry(N=7)
z8_symmetry = ZNSymmetry(N=8)
z9_symmetry = ZNSymmetry(N=9)
u1_symmetry = U1Symmetry()
su2_symmetry = SU2Symmetry()
fermion_parity = FermionParity()
