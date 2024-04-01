# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import TypeVar, Iterator
from itertools import product, count
from functools import reduce
from numpy import typing as npt
import numpy as np
from numpy._typing import NDArray


__all__ = ['Sector', 'SectorArray', 'FusionStyle', 'BraidingStyle', 'Symmetry', 'ProductSymmetry',
           'GroupSymmetry', 'AbelianGroup', 'NoSymmetry', 'U1Symmetry', 'ZNSymmetry', 'SU2Symmetry',
           'FermionParity', 'FibonacciGrading', 'no_symmetry', 'z2_symmetry', 'z3_symmetry',
           'z4_symmetry', 'z5_symmetry', 'z6_symmetry', 'z7_symmetry', 'z8_symmetry', 'z9_symmetry',
           'u1_symmetry', 'su2_symmetry', 'fermion_parity', 'IsingGrading',
           'QuantumDoubleZNAnyonModel', 'SU2_kGrading', 'ZNAnyonModel', 'ZNAnyonModel2',
           'double_semion_model', 'fibonacci_grading', 'ising_grading', 'semion_model', 'toric_code'
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
    """Base class for symmetries that impose a block-structure on tensors

    Attributes
    ----------
    fusion_style, braiding_style, trivial_sector, group_name, num_sectors, descriptive_name
        TODO
    sector_ind_len : int
        Valid sectors are numpy arrays with shape ``(sector_ind_len,)``.
    empty_sector_array : 2D ndarray
        A SectorArray with no sectors, shape ``(0, sector_ind_len)``.
    is_abelian : bool
        If the symmetry is abelian.
    """
    def __init__(self, fusion_style: FusionStyle, braiding_style: BraidingStyle, trivial_sector: Sector,
                 group_name: str, num_sectors: int | float, descriptive_name: str | None = None):
        self.fusion_style = fusion_style
        self.braiding_style = braiding_style
        self.trivial_sector = trivial_sector
        self.group_name = group_name
        self.num_sectors = num_sectors
        self.descriptive_name = descriptive_name
        self.sector_ind_len = sector_ind_len = len(trivial_sector)
        self.empty_sector_array = np.zeros((0, sector_ind_len), dtype=int)
        self.is_abelian = (fusion_style == FusionStyle.single)  # TODO doc that this does not imply group!

    @abstractmethod
    def is_valid_sector(self, a: Sector) -> bool:
        """Whether `a` is a valid sector of this symmetry"""
        ...

    def are_valid_sectors(self, sectors: SectorArray) -> bool:
        return all(self.is_valid_sector(a) for a in sectors)

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
        assert self.is_abelian
        # self.fusion_outcomes(s_a, s_b) is a 2D array with with shape [1, num_q]
        # stack the outcomes along the trivial first axis
        return np.concatenate([self.fusion_outcomes(s_a, s_b) for s_a, s_b in zip(a, b)], axis=0)

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        """This method allows optimized fusion in the case of FusionStyle.single.

        It generalizes :meth:`fusion_outcomes_broadcast` to more than two fusion inputs.
        """
        return reduce(self.fusion_outcomes_broadcast, sectors)

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        """Whether c is a valid fusion outcome, i.e. if it appears in ``self.fusion_outcomes(a, b)``"""
        return np.any(np.all(self.fusion_outcomes(a, b) == c[None, :], axis=1))

    def sector_dim(self, a: Sector) -> int:
        """The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).

        For group symmetries, this coincides with the quantum dimension computed by :meth:`qdim`.

        Note that this concept does not make sense for some anyonic symmetries.
        TODO actually, does it make sense for *any* anyonic symmetry ...?
        We raise in that case.
        """
        # TODO should we have some custom error class for "you cant do this because of symmetry stuff"
        raise ValueError(f'sector_dim is not supported for {self.__class__.__name__}')

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        """sector_dim of every sector (row) in a"""
        if self.is_abelian:
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
        r"""The sector dual to a, such that N^{a,dual(a)}_u = 1.

        Note that the dual space :math:`a^\star` to a sector :math:`a` may not itself be one of
        the sectors, but it must be isomorphic to one of the sectors. This method returns that
        representative :math:`\bar{a}` of the equivalence class.
        """
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
        If it is not, the results may be nonsensical. We do this for optimization purposes
        """
        ...

    @abstractmethod
    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        r"""Coefficients :math:`[F^{abc}_d]^e_f` related to recoupling of fusion.

        The F symbol relates the following two maps::

            m1 := [a ⊗ b ⊗ c] --(1 ⊗ X_μ)--> [a ⊗ e] --(X_ν)--> d
            m2 := [a ⊗ b ⊗ c] --(X_κ ⊗ 1)--> [f ⊗ c] --(X_λ)--> d

        Such that :math:`m_1 = \sum_{f\kappa\lambda} [F^{abc}_d]^{e\mu\nu}_{f\kappa\lambda} m_2`.

        The F symbol is unitary as a matrix from indices :math:`(f\kappa\lambda)`
        to :math:`(e\mu\nu)`.

        Parameters
        ----------
        a, b, c, d, e, f
            Sectors. Must be compatible with the fusion described above. This is not checked!

        Returns
        -------
        F : 4D array
            The F symbol as an array of the multiplicity indices [μ,ν,κ,λ]
        """
        ...

    def frobenius_schur(self, a: Sector) -> int:
        """The Frobenius Schur indicator of a sector."""
        F = self._f_symbol(a, self.dual_sector(a), a, a, self.trivial_sector, self.trivial_sector)
        return np.sign(np.real(F[0, 0, 0, 0]))

    def qdim(self, a: Sector) -> float:
        """The quantum dimension ``Tr(id_a)`` of a sector"""
        F = self._f_symbol(a, self.dual_sector(a), a, a, self.trivial_sector, self.trivial_sector)
        return 1. / np.abs(F[0, 0, 0, 0])

    def sqrt_qdim(self, a: Sector) -> float:
        """The square root of the quantum dimension."""
        return np.sqrt(self.qdim(a))

    def inv_sqrt_qdim(self, a: Sector) -> float:
        """The inverse square root of the quantum dimension."""
        return 1. / self.sqrt_qdim(a)

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Coefficients :math:`B^{ab}_c` related to bending the right leg on a fusion tensor.

        The B symbol relates the following two maps::

            m1 := a --(1 ⊗ η_b)--> [a ⊗ b ⊗ b^*] --(X_μ ⊗ 1)--> [c ⊗ b^*]
            m2 := a --(Y_ν)--> [c ⊗ \bar{b}] --(1 ⊗ Z_b^†)--> [c ⊗ b^*]

        such that :math:`m_1 = \sum_{\nu} [B^{ab}_c]^\mu_\nu m_2`.

        The related A-symbol for bending left legs is not needed, since we always
        work with fusion trees in form

        Parameters
        ----------
        a, b, c
            Sectors. Must be compatible with the fusion described above. This is not checked!

        Returns
        -------
        B : 2D array
            The B symbol as an array of the multiplicity indices [μ,ν]
        """
        F = self._f_symbol(a, b, self.dual_sector(b), a, self.trivial_sector, c)
        return self.sqrt_qdim(a) * F[0, 0, :, :]

    @abstractmethod
    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Coefficients :math:`R^{ab}_c` related to braiding on a single fusion tensor.

        The R symbol relates the following two maps::

            m1 := [b ⊗ a] --τ--> [a ⊗ b] --X_μ--> c
            m2 := [b ⊗ a] --X_ν--> c

        such that :math:`m_1 = \sum_{\nu} [R^{ab}_c]^\mu_\nu m_2`.

        .. todo ::
            Nico said (and Jakob agrees) that it should be possible to gauge the fusion tensors
            such that the R symbols are diagonal in the multiplicity index.

        Parameters
        ----------
        a, b, c
            Sectors. Must be compatible with the fusion described above. This is not checked!

        Returns
        -------
        R : 2D array
            The R symbol as an array of the multiplicity indices [μ,ν]
        """
        ...

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        r"""Coefficients :math:`[C^{abc}_d]^e_f` related to braiding on a pair of fusion tensors.

        The C symbol relates the following two maps::

            m1 := [a ⊗ c ⊗ b] --(1 ⊗ τ)--> [a ⊗ b ⊗ c] --(X_μ ⊗ 1)--> [e ⊗ c] --X_ν--> d
            m2 := [a ⊗ c ⊗ b] --(X_κ ⊗ 1)--> [f ⊗ b] --X_λ--> d

        such that :math:`m_1 = \sum_{f\kappa\lambda} C^{e\mu\nu}_{f\kappa\lambda} m_2`.

        Parameters
        ----------
        a, b, c, d, e, f
            Sectors. Must be compatible with the fusion described above. This is not checked!

        Returns
        -------
        C : 4D array
            The C symbol as an array of the multiplicity indices [μ,ν,κ,λ]
        """
        R1 = self._r_symbol(e, c, d)
        F = self._f_symbol(c, a, b, d, e, f)
        R2 = self._r_symbol(a, c, f)
        # [nu, (al)] & [mu, (al), bet, lam] -> [nu, mu, bet, lam]
        res = np.tensordot(R1, F, (1, 1))
        # [nu, mu, (bet), lam] & [kap, (bet)] -> [nu, mu, lam, kap]
        res = np.tensordot(res, np.conj(R2), (2, 1))
        # [nu, mu, lam, kap] -> [mu, nu, kap, lam]
        return np.transpose(res, [1, 0, 3, 2])

    def fusion_tensor(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Matrix elements of the fusion tensor :math:`X^{ab}_{c,\mu}` for all :math:`\mu`.

        May not be well defined for anyons.

        Returns
        -------
        X : 4D ndarray
            Axis [μ, m_a, m_b, m_c] where μ is the multiplicity index of the fusion tensor and
            m_a goes over a basis for sector a, etc.
        """
        if not self.can_fuse_to(a, b, c):
            raise ValueError('Incompatible sectors')
        return self._fusion_tensor(a, b, c)

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Internal implementation of :meth:`fusion_tensor` without the input checks."""
        msg = f'fusion_tensor is not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def all_sectors(self) -> SectorArray:
        """If there are finitely many sectors, return all of them. Else raise a ValueError."""
        if self.num_sectors == np.inf:
            msg = f'{type(self)} has infinitely many sectors.'
            raise ValueError(msg)

        raise NotImplementedError


class ProductSymmetry(Symmetry):
    """Multiple symmetries.

    The allowed sectors are "stacks" (using e.g. :func:`numpy.concatenate`) of sectors for the
    individual symmetries. For recovering the individual sectors see :attr:`sector_slices`.

    If all factors are `AbelianGroup` instances, instances of this class will masquerade as
    instances of `AbelianGroup` too, meaning they fulfill ``isinstance(s, AbelianGroup)``.
    Same for `GroupSymmetry`.

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
        nesting is flattened, i.e. ``[*others, psymm]`` is translated to
        ``[*others, *psymm.factors]`` for a :class:`ProductSymmetry` ``psymm``.
    """
    def __init__(self, factors: list[Symmetry]):
        flat_factors = []
        for f in factors:
            if isinstance(f, ProductSymmetry):
                flat_factors.extend(f.factors)
            else:
                flat_factors.append(f)
        self.factors = flat_factors
        for f in flat_factors:
            assert not isinstance(f, ProductSymmetry)  # avoid unnecessary nesting
        if all(f.descriptive_name is not None for f in flat_factors):
            descriptive_name = f'[{", ".join(f.descriptive_name for f in flat_factors)}]'
        else:
            descriptive_name = None
        self.sector_slices = np.cumsum([0] + [f.sector_ind_len for f in flat_factors])
        Symmetry.__init__(
            self,
            fusion_style=max((f.fusion_style for f in flat_factors), key=lambda style: style.value),
            braiding_style=max((f.braiding_style for f in flat_factors), key=lambda style: style.value),
            trivial_sector=np.concatenate([f.trivial_sector for f in flat_factors]),
            group_name=' ⨉ '.join(f.group_name for f in flat_factors),
            num_sectors=np.prod([symm.num_sectors for symm in flat_factors]),
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

    def are_valid_sectors(self, sectors: SectorArray) -> bool:
        shape = getattr(sectors, 'shape', ())
        if len(shape) != 2 or shape[1] != self.sector_ind_len:
            return False
        for i, f_i in enumerate(self.factors):
            sectors_i = sectors[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            if not f_i.are_valid_sectors(sectors_i):
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
        c_dtype = np.promote_types(a.dtype, b.dtype)
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

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        components = []
        for i, f_i in enumerate(self.factors):
            sectors_i = (s[:, self.sector_slices[i]:self.sector_slices[i + 1]] for s in sectors)
            c_i = f_i.multiple_fusion_broadcast(*sectors_i)
            components.append(c_i)
        return np.concatenate(components, axis=-1)

    def sector_dim(self, a: Sector) -> int:
        if self.is_abelian:
            return 1

        dims = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            dims.append(f_i.sector_dim(a_i))
        return np.prod(dims)

    def batch_sector_dim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        if self.is_abelian:
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
        if len(self.factors) == 0:
            return f'ProductSymmetry([])'
        if len(self.factors) == 1:
            return f'ProductSymmetry({self.factors[0]!r})'
        return ' * '.join(repr(f) for f in self.factors)

    def __str__(self):
        if len(self.factors) == 0:
            return f'ProductSymmetry([])'
        if len(self.factors) == 1:
            return f'ProductSymmetry({self.factors[0]!s})'
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

    def factor_where(self, descriptive_name: str) -> int:
        """Return the index of the first factor with that name. Raises if not found."""
        for i, factor in enumerate(self.factors):
            if factor.descriptive_name == descriptive_name:
                return i
        raise ValueError(f'Name not found: {descriptive_name}')

    def qdim(self, a: Sector) -> int:
        if self.is_abelian:
            return 1

        dims = []
        for i, f_i in enumerate(self.factors):
            a_i = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            dims.append(f_i.qdim(a_i))
        return np.prod(dims)

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        raise NotImplementedError  # TODO

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        raise NotImplementedError  # TODO


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
    """Base-class for symmetries that are described by a group.

    The symmetry is given via a faithful representation on the Hilbert space.
    Notable counter-examples are fermionic parity or anyonic grading.

    Products of of `GroupSymmetry`s are instances described by the `ProductSymmetry` class, which
    is not a sub- or superclass of `GroupSymmetry`. Nevertheless, instancechecks can be used to
    check if a given `ProductSymmetry` *instance* is a group-symmetry.
    See examples in :class:`AbelianGroup`.
    """
    def __init__(self, fusion_style: FusionStyle, trivial_sector: Sector, group_name: str,
                 num_sectors: int | float, descriptive_name: str | None = None):
        Symmetry.__init__(self, fusion_style=fusion_style, braiding_style=BraidingStyle.bosonic,
                          trivial_sector=trivial_sector, group_name=group_name, num_sectors=num_sectors,
                          descriptive_name=descriptive_name)

    def qdim(self, a: Sector) -> float:
        return self.sector_dim(a)


class AbelianGroup(GroupSymmetry, metaclass=_ABCFactorSymmetryMeta):
    """Base-class for abelian symmetry groups.

    Notes
    -----
    
    A product of several abelian groups is also an abelian group, but represented by a
    ProductSymmetry, which is not a subclass of AbelianGroup.
    We have adjusted instancechecks accordingly, i.e. we have

    .. doctest ::
    
        >>> s = ProductSymmetry([z3_symmetry, z5_symmetry])  # product of abelian groups
        >>> isinstance(s, AbelianGroup)
        True
        >>> issubclass(type(s), AbelianGroup)
        False
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, trivial_sector: Sector, group_name: str, num_sectors: int | float,
                 descriptive_name: str | None = None):
        GroupSymmetry.__init__(self, fusion_style=FusionStyle.single, trivial_sector=trivial_sector,
                       group_name=group_name, num_sectors=num_sectors, descriptive_name=descriptive_name)

    def sector_str(self, a: Sector) -> str:
        # we know sectors are labelled by a single number
        return str(a.item())
    
    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def sqrt_qdim(self, a: Sector) -> float:
        return 1

    def inv_sqrt_qdim(self, a: Sector) -> float:
        return 1

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._one_2D

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # For abelian groups, the R symbol is always 1.
        return self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._one_4D

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._one_4D


class NoSymmetry(AbelianGroup):
    """Trivial symmetry group that doesn't do anything.

    The only allowed sector is `[0]` of integer dtype.
    """

    def __init__(self):
        AbelianGroup.__init__(self, trivial_sector=np.array([0], dtype=int),
                              group_name='no_symmetry', num_sectors=1, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and a == 0

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(sectors == 0)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return a[np.newaxis, :]

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return a

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sectors[0]

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

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return a + b

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors)

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
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < self.N

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(0 < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

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
        return getattr(a, 'shape', ()) == (1,) and (a >= 0)

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(sectors >= 0)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        # J_tot = |J1 - J2|, ..., J1 + J2
        JJ_min = np.abs(a - b).item()
        JJ_max = (a + b).item()
        return np.arange(JJ_min, JJ_max + 2, 2)[:, np.newaxis]

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

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # R symbol is +1 if ``j_sum = (j_a + j_b - j_c)`` is even, -1 otherwise.
        # Note that (j_a + j_b - j_c) is integer by fusion rule and that e.g. ``a == j_a``.
        # For even (odd) j_sum, we get that ``(a + b - c) % 4`` is 0 (2),
        # such that ``1 - (a + b - c) % 4`` is 1 (-1). It has shape ``(1,)``.
        R = 1 - (a + b - c) % 4
        return R[:, None]
    
    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        raise NotImplementedError  # TODO


class FermionParity(Symmetry):
    """Fermionic Parity.

    Allowed sectors are 1D arrays with a single entry of either `0` (even parity) or `1` (odd parity).
    `[0]`, `[1]`
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self):
        Symmetry.__init__(self, fusion_style=FusionStyle.single, braiding_style=BraidingStyle.fermionic,
                          trivial_sector=np.array([0], dtype=int), group_name='FermionParity',
                          num_sectors=2, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 2

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 2)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        # equal sectors fuse to even parity, i.e. to `0 == (0 + 0) % 2 == (1 + 1) % 2`
        # unequal sectors fuse to odd parity i.e. to `1 == (0 + 1) % 2 == (1 + 0) % 2`
        return (a + b) % 2

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % 2

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

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

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def sqrt_qdim(self, a: Sector) -> float:
        return 1

    def inv_sqrt_qdim(self, a: Sector) -> float:
        return 1

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # sqrt(d_a) [F^{a b dual(b)}_a]^{111}_{c,mu,nu} = sqrt(1) * 1 = 1
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # if a and b are fermionic -1, otherwise +1
        return (1 - 2 * a * b)[None, :]

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        # R^{ec}_d conj(R)^{ca}_f
        C = (1 - 2 * e * c) * (1 - 2 * c * a)
        return C[None, None, None, :]

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._one_4D


class ZNAnyonModel(Symmetry):
    """Abelian anyon model with fusion rules corresponding to the Z_N group;
    also written as :math:`Z_N^{(n)}`.

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon model corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, N: int, n: int):
        assert type(N) == int
        assert type(n) == int
        self.N = N
        self.n = n % N
        self._phase = np.exp(2j * np.pi * self.n / self.N)
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.single,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='ZNAnyonModel',
                          num_sectors=N, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a[0] < self.N

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'ZNAnyonModel(N={self.N}, n={self.n})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, ZNAnyonModel) and other.N == self.N and other.n == self.n

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase**(a[0] * b[0]) * self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase**(b[0] * c[0]) * self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class ZNAnyonModel2(Symmetry):
    """Abelian anyon model with fusion rules corresponding to the Z_N group;
    also written as :math:`Z_N^{(n+1/2)}`. `N` must be even.

    .. todo ::
        find better name or include in `ZNAnyonModel`?

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon model corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, N: int, n: int):
        assert type(N) == int
        assert N % 2 == 0
        assert type(n) == int
        self.N = N
        self.n = n % N
        self._phase = np.exp(2j * np.pi * (self.n + .5) / self.N)
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.single,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='ZNAnyonModel2',
                          num_sectors=N, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < self.N

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'ZNAnyonModel2(N={self.N}, n={self.n})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, ZNAnyonModel2) and other.N == self.N and other.n == self.n

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        return (-1)**(a[0] * ((b[0] + c[0])//self.N)) * self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return (-1)**a[0]

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase**(a[0] * b[0]) * self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return (self._phase**(b[0] * c[0])) * self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class QuantumDoubleZNAnyonModel(Symmetry):
    """Doubled abelian anyon model with fusion rules corresponding to the Z_N group;
    also written as :math:`D(Z_N)`.

    Allowed sectors are 1D arrays with two integers between `0` and `N-1`.
    `[0, 0]`, `[0, 1]`, ..., `[N-1, N-1]`

    This is not a simple product for two `ZNAnyonModel`s; there are nontrivial R-symbols.
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, N: int):
        assert type(N) == int
        self.N = N
        self._phase = np.exp(2j * np.pi / self.N)
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.single,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='QuantumDoubleZNAnyonModel',
                          num_sectors=N**2, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (2,) and np.all(0 <= a) and np.all(a < self.N)

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 2 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'QuantumDoubleZNAnyonModel(N={self.N})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, QuantumDoubleZNAnyonModel) and other.N == self.N

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase**(a[0] * b[1]) * self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase**(b[0] * c[1]) * self._one_4D

    def all_sectors(self) -> SectorArray:
        x = np.arange(self.N, dtype=int)
        return np.dstack(np.meshgrid(x, x)).reshape(-1, 2)


class FibonacciGrading(Symmetry):
    """Grading of Fibonacci anyons

    .. todo ::
        Is "grading" a sensible name here?

    Allowed sectors are 1D arrays with a single entry of either `0` ("vacuum") or `1` ("tau anyon").
    `[0]`, `[1]`

    `handedness`: ``'left' | 'right'``
        Specifies the chirality / handedness of the anyons. Changing the handedness corresponds to
        complex conjugating the R-symbols, which also affects, e.g., the braid-symbols.
        Considering anyons of different handedness is necessary for doubled models like,
        e.g., the anyons realized in the Levin-Wen string-net models.
    """

    _fusion_map = {  # key: number of tau in fusion input
        0: np.array([[0]]),  # 1 x 1 = 1
        1: np.array([[1]]),  # 1 x t = t = t x 1
        2: np.array([[0], [1]]),  # t x t = 1 + t
    }
    _phi = .5 * (1 + np.sqrt(5))  # the golden ratio
    _f = np.expand_dims([_phi**-1, _phi**-0.5, -_phi**-1], axis=(1,2,3,4))  # nontrivial F-symbols
    _r = np.expand_dims([np.exp(-4j*np.pi/5), np.exp(3j*np.pi/5)], axis=(1,2))  # nontrivial R-symbols
    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, handedness = 'left'):
        assert handedness in ['left', 'right']
        self.handedness = handedness
        if handedness == 'right':
            self._r = self._r.conj()
        self._c = [super()._c_symbol([0], [1], [1], [0], [1], [1]), 0, 0,  # nontrivial C-symbols
                   super()._c_symbol([0], [1], [1], [1], [1], [1]), 0, 0,
                   super()._c_symbol([1], [1], [1], [0], [1], [1]),
                   super()._c_symbol([1], [1], [1], [1], [0], [0]),
                   super()._c_symbol([1], [1], [1], [1], [1], [0]),
                   super()._c_symbol([1], [1], [1], [1], [1], [1])]
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.multiple_unique,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='FibonacciGrading',
                          num_sectors=2, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 2
    
    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 2)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0] + b[0]]

    def sector_str(self, a: Sector) -> str:
        return 'vac' if a[0] == 0 else 'tau'

    def __repr__(self):
        return f'FibonacciGrading(handedness={self.handedness})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, FibonacciGrading) and other.handedness == self.handedness

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        if np.all(np.concatenate([a, b, c, d])):
            return self._f[e[0] + f[0]]
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1 if a[0] == 0 else self._phi

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[c[0], :, :]
        return self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            return self._c[6 * a[0] + 3 * d[0] + e[0] + f[0] - 2]
        return self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]


class IsingGrading(Symmetry):
    """Grading of Ising anyons

    .. todo ::
        Is "grading" a sensible name here?

    Allowed sectors are 1D arrays with a single entry of either `0` ("vacuum"), `1` ("Ising anyon")
    or `2` ("fermion").
    `[0]`, `[1]`, `[2]`

    `nu`: odd `int`
        In total, there are 8 distinct Ising models, i.e., `nu` and `nu + 16` describe the same
        anyon model. Different `nu` correspond to different topological twists of the Ising anyons.
        The Ising anyon model of opposite handedness is obtained for `-nu`.
    """

    _fusion_map = {  # 1: vacuum, σ: Ising anyon, ψ: fermion
        0: np.array([[0]]),  # 1 x 1 = 1
        1: np.array([[1]]),  # 1 x σ = σ = σ x 1
        2: np.array([[0], [2]]),  # σ x σ = 1 + ψ
        4: np.array([[2]]),  # 1 x ψ = ψ = 1 x ψ
        5: np.array([[1]]),  # σ x ψ = σ = σ x ψ
        8: np.array([[0]])  # ψ x ψ = 1
    }
    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, nu: int = 1):
        assert nu % 2 == 1
        self.nu = nu % 16
        self.frobenius = [1, int((-1)**((self.nu**2-1)/8)), 1]
        self._f = (np.expand_dims([1, 0, 1, 0, -1], axis=(1,2,3,4))
                            * self.frobenius[1] / np.sqrt(2))  # nontrivial F-symbols
        self._r = np.expand_dims([(-1j)**self.nu, -1, np.exp(3j*self.nu*np.pi/8) * self.frobenius[1],
                    np.exp(-1j*self.nu*np.pi/8) * self.frobenius[1], 0], axis=(1,2))  # nontrivial R-symbols
        self._c = [(-1j)**self.nu * self._one_4D, -1 * (-1j)**self.nu * self._one_4D,
                   super()._c_symbol([0], [1], [1], [0], [1], [1]),  # nontrivial C-symbols
                   super()._c_symbol([0], [1], [1], [2], [1], [1]),
                   super()._c_symbol([1], [1], [1], [1], [0], [0]),
                   super()._c_symbol([1], [1], [1], [1], [0], [2]),
                   super()._c_symbol([1], [1], [1], [1], [2], [2]), 0,
                   super()._c_symbol([2], [1], [1], [0], [1], [1]),
                   super()._c_symbol([2], [1], [1], [2], [1], [1]), -1 * self._one_4D]
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.multiple_unique,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='IsingGrading',
                          num_sectors=3, descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 3

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 3)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0]**2 + b[0]**2]

    def sector_str(self, a: Sector) -> str:
        if a[0] == 1:
            return 'sigma'
        return 'vac' if a[0] == 0 else 'psi'

    def __repr__(self):
        return f'IsingGrading(nu={self.nu})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, IsingGrading) and other.nu == self.nu

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        if not np.any(np.concatenate([a, b, c, d]) - [1, 1, 1, 1]):
            return self._f[e[0] + f[0]]
        elif (not np.any(np.concatenate([a, b, c, d]) - [2, 1, 2, 1])
                or not np.any(np.concatenate([a, b, c, d]) - [1, 2, 1, 2])):
            return -1 * self._one_4D
        return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return self.frobenius[a[0]]

    def qdim(self, a: Sector) -> float:
        return np.sqrt(2) if a[0] == 1 else 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[(a[0] + b[0]) * (c[0] - 1), :, :]
        return self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            factor = -1 * (b[0] - c[0] - 1) * (b[0] - c[0] + 1)  # = 0 if σ and ψ or σ and ψ, 1 otherwise
            factor *= ( 1 - a[0]//2 - d[0]//2 + 9 * (b[0] - 1) + (2 - b[0]) * ((e[0] + f[0])//2 + d[0]//2 + 3 * a[0]) )
            return self._c[factor + a[0]//2 + d[0]//2]
        return self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(3, dtype=int)[:, None]


class SU2_kGrading(Symmetry):
    """:math:`SU(2)_k` anyons.

    .. todo ::
        Implement C-symbols without fallback? -> need to save them
        We probably want to introduce the option to save and load R-symbols, F-symbols, etc.
        Otherwise, for "large" k, constructing the data takes too much time

    The anyons can be associated with the spins `0`, `1/2`, `1`, ..., `k/2`.
    Unlike regular SU(2), there is a cutoff at `k/2`.

    Allowed sectors are 1D arrays ``[jj]`` of positive integers `jj` = `0`, `1`, `2`, ..., `k`
    corresponding to `jj/2` listed above.

    `handedness`: ``'left' | 'right'``
        Specifies the chirality / handedness of the anyons. Changing the handedness corresponds to
        complex conjugating the R-symbols, which also affects, e.g., the braid-symbols.
        Considering anyons of different handedness is necessary for doubled models like,
        e.g., the anyons realized in the Levin-Wen string-net models.
    """

    _one_2D = np.ones((1, 1), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, k: int, handedness = 'left'):
        assert type(k) == int
        assert handedness in ['left', 'right']
        self.k = k
        self.handedness = handedness
        self._q = np.exp(2j * np.pi / (k + 2))

        self._r = {}
        for i in range((self.k + 1)**3):
            jj1 = i % (self.k + 1)
            jj2 = i // (self.k + 1) % (self.k + 1)
            jj = i // (self.k + 1)**2 % (self.k + 1)
            if jj > jj1 + jj2 or jj < abs(jj1 - jj2) or jj1 * jj2 == 0:
                continue  # do not save trivial R-symbols
            factor = (-1)**((jj - jj1 - jj2) / 2)
            factor *= self._q**(( jj*(jj+2) - jj1*(jj1+2) - jj2*(jj2+2) ) / 8)
            self._r[i] = factor * self._one_2D

        self._f = {}
        self._convert_to_key = np.array([(self.k + 1)**i for i in range(6)])
        for i in range((self.k + 1)**6):
            jj1 = i % (self.k + 1)
            jj2 = i // (self.k + 1) % (self.k + 1)
            jj3 = i // (self.k + 1)**2 % (self.k + 1)
            jj12 = i // (self.k + 1)**3 % (self.k + 1)
            jj23 = i // (self.k + 1)**4 % (self.k + 1)
            jj = i // (self.k + 1)**5 % (self.k + 1)
            if jj1 * jj2 * jj3 == 0:  # do not save trivial F-symbols
                continue
            jsymbol = self._j_symbol(jj1, jj2, jj12, jj3, jj, jj23)
            if jsymbol != 0:
                prefactor = (-1)**((jj + jj1 + jj2 + jj3) / 2)
                prefactor *= np.sqrt(self._n_q(jj12 + 1) * self._n_q(jj23 + 1))
                self._f[i] = prefactor * jsymbol * self._one_4D

        Symmetry.__init__(self,
                          fusion_style=FusionStyle.multiple_unique,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='SU2_kGrading',
                          num_sectors=self.k+1, descriptive_name=None)

    def _n_q(self, n: int) -> float:
        return (self._q**(.5*n) - self._q**(-.5*n)) / (self._q**.5 - self._q**-.5)

    def _n_q_fac(self, n: int) -> float:
        fac = 1
        for i in range(n):
            fac *= self._n_q(i + 1)
        return fac

    def _delta(self, jj1: int, jj2: int, jj3: int) -> float:
        res = self._n_q_fac( round(-1*jj1/2 + jj2/2 + jj3/2) ) * self._n_q_fac( round(jj1/2 - jj2/2 + jj3/2) )
        res *= self._n_q_fac( round(jj1/2 + jj2/2 - jj3/2) ) / self._n_q_fac( round(jj1/2 + jj2/2 + jj3/2 + 1) )
        return np.sqrt(res)

    def _j_symbol(self, jj1: int, jj2: int, jj12: int, jj3: int, jj: int, jj23: int) -> float:
        for triad in [[jj1, jj2, jj12], [jj1, jj, jj23], [jj3, jj2, jj23], [jj3, jj, jj12]]:
            if triad[0] > triad[1] + triad[2] or triad[0] < abs(triad[1] - triad[2]):
                return 0
        start = max([jj1 + jj2 + jj12, jj12 + jj3 + jj, jj2 + jj3 + jj23, jj1 + jj23 + jj]) // 2
        stop = min([jj1 + jj2 + jj3 + jj, jj1 + jj12 + jj3 + jj23, jj2 + jj12 + jj + jj23]) // 2
        res = 0
        for z in range(start, stop + 1):  # runs over all integers for which the factorials have non-negative arguments
            factor = self._n_q_fac( round(z - jj1/2 - jj2/2 - jj12/2) ) * self._n_q_fac( round(z - jj12/2 - jj3/2 - jj/2) )
            factor *= self._n_q_fac( round(z - jj2/2 - jj3/2 - jj23/2) ) * self._n_q_fac( round(z - jj1/2 - jj23/2 - jj/2) )
            factor *= self._n_q_fac( round(jj1/2 + jj2/2 + jj3/2 + jj/2 - z) )
            factor *= self._n_q_fac( round(jj1/2 + jj12/2 + jj3/2 + jj23/2 - z) )
            factor *= self._n_q_fac( round(jj2/2 + jj12/2 + jj/2 + jj23/2 - z) )
            res += (-1)**z * self._n_q_fac(z + 1) / factor
        return res * (self._delta(jj1, jj2, jj12) * self._delta(jj12, jj3, jj)
                      * self._delta(jj2, jj3, jj23) * self._delta(jj1, jj23, jj))

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a <= self.k

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors <= self.k)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        upper_limit = min(a[0] + b[0], 2 * self.k - a[0] - b[0])
        return np.arange(abs(a[0] - b[0]), upper_limit + 2, 2)[:, np.newaxis]

    def sector_str(self, a: Sector) -> str:
        jj = a[0]
        j_str = str(jj // 2) if jj % 2 == 0 else f'{jj}/2'
        return f'{jj} (j={j_str})'

    def __repr__(self):
        return f'SU2_kGrading({self.k}, {self.handedness})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, SU2_kGrading) and other.k == self.k and other.handedness == self.handedness

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        try:  # nontrivial F-symbols
            return self._f[np.sum(self._convert_to_key * np.concatenate([a, b, c, e, f, d]))]
        except KeyError:
            return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return -1 if a[0] % 2 == 1 else 1

    def qdim(self, a: Sector) -> float:
        return np.sin((a[0] + 1) * np.pi / (self.k + 2)) / np.sin(np.pi / (self.k + 2))

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        try:  # nontrivial R-symbols
            return self._r[np.sum(self._convert_to_key[:3] * np.concatenate([a, b, c]))]
        except KeyError:
            return self._one_2D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return super()._c_symbol(a, b, c, d, e, f)

    def all_sectors(self) -> SectorArray:
        return np.arange(self.k + 1, dtype=int)[:, None]


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
semion_model = ZNAnyonModel2(2, 0)
toric_code = QuantumDoubleZNAnyonModel(2)
double_semion_model = ProductSymmetry([ZNAnyonModel2(2, 0), ZNAnyonModel2(2, 1)])
fibonacci_grading = FibonacciGrading(handedness='left')
ising_grading = IsingGrading(nu=1)
