""""""
# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from enum import Enum
from functools import reduce, total_ordering
from itertools import product

from numpy import typing as npt
import numpy as np
import math

from .dtypes import Dtype

__all__ = ['SymmetryError', 'Sector', 'SectorArray', 'FusionStyle', 'BraidingStyle',
           # symmetry base-classes
           'Symmetry', 'ProductSymmetry', 'GroupSymmetry', 'AbelianGroup',
           # abelian groups
           'NoSymmetry', 'U1Symmetry', 'ZNSymmetry',
           # non-abelian groups
           'SU2Symmetry', 'SUNSymmetry',
           # anyons
           'FermionParity', 'FibonacciAnyonCategory', 'IsingAnyonCategory', 'SU3_3AnyonCategory',
           'QuantumDoubleZNAnyonCategory', 'SU2_kAnyonCategory', 'ZNAnyonCategory',
           'ZNAnyonCategory2',
           # concrete instances
           'no_symmetry', 'u1_symmetry', 'z2_symmetry', 'z3_symmetry', 'z4_symmetry', 'z5_symmetry', 'z6_symmetry',
           'z7_symmetry', 'z8_symmetry', 'z9_symmetry',
           'fermion_parity', 'fibonacci_anyon_category', 'ising_anyon_category',
           'double_semion_category', 'semion_category', 'toric_code_category',
           ]


class SymmetryError(Exception):
    """An exception that is raised whenever something is not possible or not allowed due to symmetry"""
    pass


Sector = npt.NDArray[np.int_]
"""Type hint for a sector. A 1D array of integers with axis [q] and shape ``(sector_ind_len,)``."""

SectorArray = npt.NDArray[np.int_]
"""Type hint for an array of multiple sectors.

A 2D array of int with axis [s, q] and shape ``(num_sectors, sector_ind_len)``.
"""


# TODO eventually decide if we want to do them or not or optionally.
#      but we should definitely have these checks while developing.
_DO_FUSION_INPUT_CHECKS = True


@total_ordering
class FusionStyle(Enum):
    """Describes properties of fusion, i.e. of the tensor product.

    =================  =============================================================================
    Value              Meaning
    =================  =============================================================================
    single             Fusing sectors results in a single sector ``a ⊗ b = c``, e.g. abelian groups.
    -----------------  -----------------------------------------------------------------------------
    multiple_unique    Every sector appears at most once in pairwise fusion, ``N_symbol in [0, 1]``.
    -----------------  -----------------------------------------------------------------------------
    general            No assumptions, ``N_symbol in [0, 1, 2, 3, ...]``.
    =================  =============================================================================

    """
    single = 0  # only one resulting sector, a ⊗ b = c, e.g. abelian symmetry groups
    multiple_unique = 10  # every sector appears at most once in pairwise fusion, N^{ab}_c \in {0,1}
    general = 20  # no assumptions N^{ab}_c = 0, 1, 2, ...

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


@total_ordering
class BraidingStyle(Enum):
    """Describes properties of braiding.

    =============  ===========================================
    Value
    =============  ===========================================
    bosonic        Symmetric braiding with trivial twist
    -------------  -------------------------------------------
    fermionic      Symmetric braiding with non-trivial twist
    -------------  -------------------------------------------
    anyonic        General, non-symmetric braiding
    -------------  -------------------------------------------
    no_braiding    Braiding is not defined
    =============  ===========================================
    """

    bosonic = 0  # symmetric braiding with trivial twist; v ⊗ w ↦ w ⊗ v
    fermionic = 10  # symmetric braiding with non-trivial twist; v ⊗ w ↦ (-1)^p(v,w) w ⊗ v
    anyonic = 20  # non-symmetric braiding
    no_braiding = 30  # braiding is not defined

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


class Symmetry(metaclass=ABCMeta):
    r"""Base class for symmetries that impose a block-structure on tensors

    Attributes
    ----------
    fusion_style: :class:`FusionStyle`
    braiding_style: :class:`BraidingStyle`
    can_be_dropped: bool
        If the symmetry could be dropped to :class:`NoSymmetry` while preserving the structure.
        This is e.g. the case for :class:`GroupSymmetry` subclasses.
        This means that there is a well-defined notion of a basis of graded vector spaces and of
        dense array representations of symmetric Tensor. See notes below.
    trivial_sector: Sector
        The trivial sector of the symmetry.
        For a group this is the "symmetric" sector, where the group acts trivially.
        For a general category, this is the monoidal unit.
    group_name: str
        A readable name for the symmetry, purely as a mathematical structure, e.g. ``'U(1)'``.
    descriptive_name: str | None
        Optionally, an additional name for the group, indicating e.g. how it arises.
        Could be e.g. ``'Sz'`` for the U(1) symmetry that conserves magnetization.
    num_sectors: int | float
        The number of sectors of the symmetry. An integer if finite, otherwise ``float('inf')``.
    sector_ind_len : int
        Valid sectors are numpy arrays with shape ``(sector_ind_len,)``.
    empty_sector_array : 2D ndarray
        A SectorArray with no sectors, shape ``(0, sector_ind_len)``.
    is_abelian : bool
        If the symmetry is abelian.  An abelian symmetry is characterized by ``FusionStyle.single``,
        which implies that all sectors are one-dimensional.
        Note that this does *not* imply that it is a group, as the braiding may not be bosonic!

    Notes
    -----
    Some symmetries, can in principle be dropped to :class:`NoSymmetry`.
    We call this property :attr:`can_be_dropped`. Currently, only :class:`GroupSymmetry` subclasses
    and their products have this property.
    It implies that all operations that may be carried out on symmetric objects have a corresponding
    operation on a non-symmetric counterpart. For example, a symmetric space :math:`A` has a
    corresponding space :math:`\mathbb{C}^n_A`, without further structure.
    It "corresponds" to :math:`A` in the sense that it has the same properties, e.g. same dimension,
    and that there are compatible operations (tensor product, direct sum, ...) such that::

        symmetric :math:`A`  -------- (operation) --->   symmetric :math:`B`
                |                                                 |
             (drop symm)                                       (drop symm)
                |                                                 |
                v                                                 v
        :math:`\mathbb{C}^{n_A}`  --- (operation) --->   :math:`\mathbb{C}^{n_B}`

    commutes.
    The same goes for tensors, i.e. for symmetric tensors there are corresponding non-symmetric
    tensors which we may manipulate instead. This means that if *and only if* the symmetry has this
    property does it make sense to between symmetric tensors and e.g. numpy arrays, which we can
    think of as tensors with :class:`NoSymmetry`. Additionally, the concept of a basis only makes
    sense in exactly these cases.
    """

    can_be_dropped: bool = False  # set to False by default. must override for ``True``.

    fusion_tensor_dtype = None
    """The dtype of fusion tensors, if available. Set to ``None`` otherwise."""

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
        self.is_abelian = (fusion_style == FusionStyle.single)

    # ABSTRACT METHODS

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

    @abstractmethod
    def __repr__(self):
        # Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
        ...

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

    @abstractmethod
    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.
        If it is not, the results may be nonsensical. We do this for optimization purposes
        """
        ...

    @abstractmethod
    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        """Internal implementation of :meth:`f_symbol`. Can assume that inputs are valid."""
        ...

    @abstractmethod
    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Internal implementation of :meth:`r_symbol`. Can assume that inputs are valid."""
        ...

    @property
    def has_symmetric_braid(self) -> bool:
        return self.braiding_style <= BraidingStyle.fermionic

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> np.ndarray:
        """Internal implementation of :meth:`fusion_tensor`. Can assume that inputs are valid."""
        if not self.can_be_dropped:
            raise SymmetryError(f'fusion tensor can not be written as array for {self}')
        raise NotImplementedError('should be implemented by subclass')

    def Z_iso(self, a: Sector) -> np.ndarray:
        r"""The Z isomorphism :math:`Z_{\bar{a}} : \bar{a}^* \to a`.

        The dual :math:`a^*` of a sector :math:`a` is another irreducible space.
        However, it may not be itself a sector. It must be isomorphic to one of the sector
        representatives though, which we call :math:`\bar{a}`.
        The Z isomorphism :math:`Z_a : a^* \to \bar{a}` is that isomorphism.

        We return the matrix elements
        .. math ::
            (Z_{\bar{a}})_{mn} = \langle m \vert Z_{\bar{a}}(\langle n \vert)

        where :math:`m` goes over a (dual) basis of :math:`\bar{a}` and :math:`n` over a basis of
        :math:`a`.

        Parameters
        ----------
        a : Sector
            Note that this is the target sector of the map, not its subscript!

        Returns
        -------
        The matrix elements as a [d_a, d_a] numpy array.
        """
        if not self.can_be_dropped:
            raise SymmetryError(f'Z iso can not be written as array for {self}')
        raise NotImplementedError('should be implemented by subclass')

    def all_sectors(self) -> SectorArray:
        """If there are finitely many sectors, return all of them. Else raise a ValueError.

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.
        """
        if self.num_sectors == np.inf:
            msg = f'{type(self)} has infinitely many sectors.'
            raise SymmetryError(msg)

        raise NotImplementedError

    # FALLBACK IMPLEMENTATIONS (might want to override)

    def are_valid_sectors(self, sectors: SectorArray) -> bool:
        return all(self.is_valid_sector(a) for a in sectors)

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

    def multiple_fusion(self, *sectors: Sector) -> Sector:
        # OPTIMIZE ?
        return self.multiple_fusion_broadcast(*(a[None, :] for a in sectors))[0, :]

    def multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        """This method allows optimized fusion in the case of FusionStyle.single.

        It generalizes :meth:`fusion_outcomes_broadcast` to more than two fusion inputs.
        """
        if len(sectors) == 0:
            return self.trivial_sector[None, :]
        if len(sectors) == 1:
            return sectors[0]
        return self._multiple_fusion_broadcast(*sectors)

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        """Internal version of :meth:`multiple_fusion_broadcast`. May assume ``len(sectors) >= 2``."""
        return reduce(self.fusion_outcomes_broadcast, sectors)

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        """Whether c is a valid fusion outcome, i.e. if it appears in ``self.fusion_outcomes(a, b)``"""
        return np.any(np.all(self.fusion_outcomes(a, b) == c[None, :], axis=1))

    def sector_dim(self, a: Sector) -> int:
        """The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).

        For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum
        dimension computed by :meth:`qdim`.
        For other braiding styles,
        """
        if not self.can_be_dropped:
            raise SymmetryError(f'sector_dim is not supported for {self}.')
        return int(np.round(self.qdim()))

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        """sector_dim of every sector (row) in a"""
        if self.is_abelian:
            return np.ones([a.shape[0]], dtype=int)
        return np.array([self.sector_dim(s) for s in a])

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        """qdim of every sector (row) in a"""
        if self.is_abelian:
            return np.ones([a.shape[0]], dtype=int)
        return np.array([self.qdim(s) for s in a])

    def sector_str(self, a: Sector) -> str:
        """Short and readable string for the sector. Is used in __str__ of symmetry-related objects."""
        return str(a)

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        """dual_sector for multiple sectors

        subclasses my override this.
        """
        return np.stack([self.dual_sector(s) for s in sectors])

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

    def total_qdim(self) -> float:
        r"""Total quantum dimension, :math:`D = \sqrt{\sum_a d_a^2}`."""
        D = np.sum([self.qdim(a)**2 for a in self.all_sectors()])
        return np.sqrt(D)

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Internal implementation of :meth:`b_symbol`. Can assume that inputs are valid."""
        F = self._f_symbol(a, b, self.dual_sector(b), a, self.trivial_sector, c)
        return self.sqrt_qdim(b) * F[0, 0, :, :]

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        """Internal implementation of :meth:`c_symbol`. Can assume that inputs are valid."""
        R1 = self._r_symbol(e, c, d)
        F = self._f_symbol(c, a, b, d, e, f)
        R2 = self._r_symbol(a, c, f)
        # axis [mu, nu, kap, lam] ; R symbols are diagonal
        return R1[None, :, None, None] * F * np.conj(R2)[None, None, :, None]

    def topological_twist(self, a: Sector) -> complex:
        """Phase acquired when charges fully rotate around their own axis once."""
        return self.frobenius_schur(a) * self._r_symbol(self.dual_sector(a), a, self.trivial_sector)[0].conj()

    def s_matrix_element(self, a: Sector, b: Sector) -> complex:
        """Element of the S-matrix for anyon categories. The S-matrix is unitary for modular anyon categories."""
        S = 0
        for c in self.fusion_outcomes(a, b):
            S += self._n_symbol(a, b, c) * self.qdim(c) * self.topological_twist(c)
        S /= self.topological_twist(a) * self.topological_twist(b) * self.total_qdim()
        return np.real_if_close(S)

    def s_matrix(self) -> np.ndarray:
        """S-matrix for anyon categories. The S-matrix is unitary for modular anyon categories."""
        sectors = self.all_sectors()
        S = np.zeros((self.num_sectors, self.num_sectors), dtype=complex)
        normalization = np.array([1/self.topological_twist(a) for a in sectors])
        normalization = np.outer(normalization, normalization) / self.total_qdim()
        for a in range(sectors.shape[0]):
            for b in range(sectors.shape[0]):
                for c in self.fusion_outcomes(sectors[a], sectors[b]):
                    S[a,b] += self._n_symbol(sectors[a], sectors[b], c) * self.qdim(c) * self.topological_twist(c)
        return np.real_if_close(S * normalization)

    # CONCRETE IMPLEMENTATIONS

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

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """The N-symbol N^{ab}_c, i.e. how often c appears in the fusion of a and b."""
        if not self.can_fuse_to(a, b, c):
            return 0
        return self._n_symbol(a, b, c)

    def f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                 ) -> np.ndarray:
        r"""Coefficients :math:`[F^{abc}_d]^e_f` related to recoupling of fusion.

        The F symbol relates the following two maps::

            m1 := [a ⊗ b ⊗ c] --(1 ⊗ X_μ)--> [a ⊗ e] --(X_ν)--> d
            m2 := [a ⊗ b ⊗ c] --(X_κ ⊗ 1)--> [f ⊗ c] --(X_λ)--> d

        Such that :math:`m_1 = \sum_{f\kappa\lambda} [F^{abc}_d]^{e\mu\nu}_{f\kappa\lambda} m_2`.

        The F symbol is unitary as a matrix from indices :math:`(f\kappa\lambda)`
        to :math:`(e\mu\nu)`.

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.

        Parameters
        ----------
        a, b, c, d, e, f
            Sectors. Must be compatible with the fusion described above.

        Returns
        -------
        F : 4D array
            The F symbol as an array of the multiplicity indices [μ,ν,κ,λ]
        """
        if _DO_FUSION_INPUT_CHECKS:
            is_correct = all([
                self.can_fuse_to(b, c, e),
                self.can_fuse_to(a, e, d),
                self.can_fuse_to(a, b, f),
                self.can_fuse_to(f, c, d)
            ])
            if not is_correct:
                raise SymmetryError('Sectors are not consistent with fusion rules.')
        return self._f_symbol(a, b, c, d, e, f)

    def b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Coefficients :math:`B^{ab}_c` related to bending the right leg on a fusion tensor.

        The B symbol relates the following two maps::

            m1 := a --(1 ⊗ η_b)--> [a ⊗ b ⊗ b^*] --(X_μ ⊗ 1)--> [c ⊗ b^*]
            m2 := a --(Y_ν)--> [c ⊗ \bar{b}] --(1 ⊗ Z_b^†)--> [c ⊗ b^*]

        such that :math:`m_1 = \sum_{\nu} [B^{ab}_c]^\mu_\nu m_2`.

        The related A-symbol for bending left legs is not needed, since we always
        work with fusion trees in form

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.

        Parameters
        ----------
        a, b, c
            Sectors. Must be compatible with the fusion described above.

        Returns
        -------
        B : 2D array
            The B symbol as an array of the multiplicity indices [μ,ν]
        """
        if _DO_FUSION_INPUT_CHECKS:
            is_correct = self.can_fuse_to(a, b, c)
            if not is_correct:
                raise SymmetryError('Sectors are not consistent with fusion rules.')
        return self._b_symbol(a, b, c)

    def r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        r"""Coefficients :math:`R^{ab}_c` related to braiding on a single fusion tensor.

        The R symbol relates the following two maps::

            m1 := [b ⊗ a] --τ--> [a ⊗ b] --X_μ--> c
            m2 := [b ⊗ a] --X_ν--> c

        such that :math:`m_1 = \sum_{\nu} [R^{ab}_c]^\mu_\nu m_2`.

        We can use the unitary gauge freedom of the fusion tensors
        .. math ::

            X_μ \mapsto \sum_ν U_{μ,ν} X_ν

        to enforce that the R symbol is diagonal.

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.

        Parameters
        ----------
        a, b, c
            Sectors. Must be compatible with the fusion described above.

        Returns
        -------
        R : 1D array
            The diagonal entries of the R symbol as an array of the multiplicity index [μ].
        """
        if _DO_FUSION_INPUT_CHECKS:
            is_correct = self.can_fuse_to(a, b, c)
            if not is_correct:
                raise SymmetryError('Sectors are not consistent with fusion rules.')
        return self._r_symbol(a, b, c)

    def c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        r"""Coefficients :math:`[C^{abc}_d]^e_f` related to braiding on a pair of fusion tensors.

        The C symbol relates the following two maps::

            m1 := [a ⊗ c ⊗ b] --(1 ⊗ τ)--> [a ⊗ b ⊗ c] --(X_μ ⊗ 1)--> [e ⊗ c] --X_ν--> d
            m2 := [a ⊗ c ⊗ b] --(X_κ ⊗ 1)--> [f ⊗ b] --X_λ--> d

        such that :math:`m_1 = \sum_{f\kappa\lambda} C^{e\mu\nu}_{f\kappa\lambda} m_2`.

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.

        Parameters
        ----------
        a, b, c, d, e, f
            Sectors. Must be compatible with the fusion described above.

        Returns
        -------
        C : 4D array
            The C symbol as an array of the multiplicity indices [μ,ν,κ,λ]
        """
        if _DO_FUSION_INPUT_CHECKS:
            is_correct = all([
                self.can_fuse_to(a, b, e),
                self.can_fuse_to(e, c, d),
                self.can_fuse_to(a, c, f),
                self.can_fuse_to(f, b, d)
            ])
            if not is_correct:
                raise SymmetryError('Sectors are not consistent with fusion rules.')
        return self._c_symbol(a, b, c, d, e, f)

    def fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool = False, Z_b: bool = False
                      ) -> np.ndarray:
        r"""Matrix elements of the fusion tensor :math:`X^{ab}_{c,\mu}` for all :math:`\mu`.

        May not be well defined for anyons.

        .. warning ::
            Do not perform inplace operations on the output. That may invalidate caches.

        Parameters
        ----------
        a, b, c
            Sectors. Must be compatible with the fusion described above.
        Z_a, Z_b : bool
            If we should include a Z isomorphism below the sector a.
            If so, the composite is a map from :math:`\bar{a}^* \otimes b \to c`.

        Returns
        -------
        X : 4D ndarray
            Axis [μ, m_a, m_b, m_c] where μ is the multiplicity index of the fusion tensor and
            m_a goes over a basis for sector a, etc.
        """
        if _DO_FUSION_INPUT_CHECKS:
            is_correct = self.can_fuse_to(a, b, c)
            if not is_correct:
                raise SymmetryError('Sectors are not consistent with fusion rules.')
        return self._fusion_tensor(a, b, c, Z_a, Z_b)


class ProductSymmetry(Symmetry):
    """Multiple symmetries.

    The allowed sectors are "stacks" (using e.g. :func:`numpy.concatenate`) of sectors for the
    individual symmetries. For recovering the individual sectors see :attr:`sector_slices`.

    If all factors are :class:`AbelianGroup` instances, instances of this class will masquerade as
    instances of :class:`AbelianGroup` too, meaning they fulfill ``isinstance(s, AbelianGroup)``.
    Same for :class:`GroupSymmetry`.

    Attributes
    ----------
    factors : list of :class:`Symmetry`
        The individual symmetries. We do not allow nesting, i.e. the `factors` can not
        be :class:`ProductSymmetry`s themselves.
    sector_slices : 1D ndarray
        Describes how the sectors of the `factors` are embedded in a sector of the product.
        Indicates that the slice ``sector_slices[i]:sector_slices[i + 1]`` of a sector of the
        product symmetry contains the entries of a sector of ``factors[i]``.

    Parameters
    ----------
    factors : list of :class:`Symmetry`
        The factors that comprise this symmetry. If any are :class:`ProductSymmetry`s, the
        nesting is flattened, i.e. ``[*others, psymm]`` is translated to
        ``[*others, *psymm.factors]`` for a :class:`ProductSymmetry` ``psymm``.
    """
    can_be_dropped = None  # set by __init__

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
        if all(f.descriptive_name is None for f in flat_factors):
            descriptive_name = None
        else:
            descriptive_name = ', '.join(f.descriptive_name or '-' for f in flat_factors)
        self.sector_slices = np.cumsum([0] + [f.sector_ind_len for f in flat_factors])
        Symmetry.__init__(
            self,
            fusion_style=max((f.fusion_style for f in flat_factors), key=lambda style: style.value),
            braiding_style=max((f.braiding_style for f in flat_factors), key=lambda style: style.value),
            trivial_sector=np.concatenate([f.trivial_sector for f in flat_factors]),
            group_name=' ⨉ '.join(f.group_name for f in flat_factors),
            num_sectors=math.prod([symm.num_sectors for symm in flat_factors]),
            descriptive_name=descriptive_name
        )
        self.can_be_dropped = all(f.can_be_dropped for f in flat_factors)
        dtypes = [f.fusion_tensor_dtype for f in flat_factors]
        if None in dtypes:
            self.fusion_tensor_dtype = None
        else:
            self.fusion_tensor_dtype = Dtype.common(*dtypes)

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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        components = []
        for i, f_i in enumerate(self.factors):
            sectors_i = tuple(s[:, self.sector_slices[i]:self.sector_slices[i + 1]] for s in sectors)
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

    def batch_qdim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        if self.is_abelian:
            return np.ones([a.shape[0]], dtype=int)
        dims = []
        for i, f_i in enumerate(self.factors):
            a_i = a[:, self.sector_slices[i]:self.sector_slices[i + 1]]
            dims.append(f_i.batch_qdim(a_i))
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
            raise SymmetryError(msg)

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

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector ) -> np.ndarray:
        #print(self.factors)

        contributions = []
        for i, f_i in enumerate(self.factors):      #self =Fibonacci_L x Fibonacci_R Product symmetry i=index f_i= FibonacciGrading left, FibonacciGrading right

            a_k = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            b_k = b[self.sector_slices[i]:self.sector_slices[i + 1]]
            c_k = c[self.sector_slices[i]:self.sector_slices[i + 1]]
            d_k = d[self.sector_slices[i]:self.sector_slices[i + 1]]
            e_k = e[self.sector_slices[i]:self.sector_slices[i + 1]]
            f_k = f[self.sector_slices[i]:self.sector_slices[i + 1]]


            contributions.append(f_i._f_symbol(a_k, b_k, c_k, d_k, e_k, f_k)) #[μ,ν,κ,λ]


        Fs=np.multiply(contributions[0],contributions[1])
        for i in contributions[2:]:
            Fs = np.kron(Fs,i)

        return Fs

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:

        contributions = []
        for i, f_i in enumerate(self.factors):
            a_k = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            b_k = b[self.sector_slices[i]:self.sector_slices[i + 1]]
            c_k = c[self.sector_slices[i]:self.sector_slices[i + 1]]

            contributions.append(f_i._r_symbol(a_k, b_k, c_k))

        rsym=np.kron(contributions[0],contributions[1])
        for i in contributions[2:]:
            rsym=np.kron(rsym,i)

        return rsym

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool = False, Z_b: bool = False
                       ) -> np.ndarray:
        if not self.can_be_dropped:
            raise SymmetryError(f'fusion tensor can not be written as array for {self}')

        contributions = []
        for i, f_i in enumerate(self.factors):
            a_k = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            b_k = b[self.sector_slices[i]:self.sector_slices[i + 1]]
            c_k = c[self.sector_slices[i]:self.sector_slices[i + 1]]
            contributions.append(f_i._fusion_tensor(a_k, b_k, c_k, Z_a, Z_b))

        ftensor = np.kron(contributions[0], contributions[1])
        for i in contributions[2:]:
            ftensor = np.kron(ftensor, i)

        return ftensor

    def Z_iso(self, a: Sector) -> np.ndarray:
        if not self.can_be_dropped:
            raise SymmetryError(f'Z iso can not be written as array for {self}')

        contributions = []
        for i, f_i in enumerate(self.factors):
            a_k = a[self.sector_slices[i]:self.sector_slices[i + 1]]
            contributions.append(f_i.Z_iso(a_k))

        res = np.kron(contributions[0], contributions[1])
        for i in contributions[2:]:
            res = np.kron(res, i)

        return res


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

    Notes
    -----

    Products of :class:`GroupSymmetry`s are instances described by the :class:`ProductSymmetry`
    class, which is not a sub- or superclass of `GroupSymmetry`. Nevertheless, instancechecks can
    be used to check if a given `ProductSymmetry` *instance* is a group-symmetry.
    See examples in docstring of :class:`AbelianGroup`.
    """
    can_be_dropped = True

    def __init__(self, fusion_style: FusionStyle, trivial_sector: Sector, group_name: str,
                 num_sectors: int | float, descriptive_name: str | None = None):
        Symmetry.__init__(self, fusion_style=fusion_style, braiding_style=BraidingStyle.bosonic,
                          trivial_sector=trivial_sector, group_name=group_name, num_sectors=num_sectors,
                          descriptive_name=descriptive_name)

    @abstractmethod
    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> npt.NDArray:
        # subclasses must implement. for groups it is always possible.
        ...

    @abstractmethod
    def Z_iso(self, a: Sector) -> npt.NDArray:
        # subclasses must implement. for groups it is always possible.
        ...

    def qdim(self, a: Sector) -> float:
        return self.sector_dim(a)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return self.batch_sector_dim(a)


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

    fusion_tensor_dtype = Dtype.float64

    # TODO should we just have this in the module?
    _one_1D = np.ones((1), dtype=int)
    _one_2D = np.ones((1, 1), dtype=int)
    _one_2D_float = np.ones((1, 1), dtype=float)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)
    _one_4D_float = np.ones((1, 1, 1, 1), dtype=float)

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
        return self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._one_4D

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> np.ndarray:
        return self._one_4D_float

    def Z_iso(self, a: Sector) -> np.ndarray:
        return self._one_2D_float


class NoSymmetry(AbelianGroup):
    """Trivial symmetry group that doesn't do anything.

    The only allowed sector is ``[0]``.
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
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

    fusion_tensor_dtype = Dtype.float64

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

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        return (c <= a + b) and (a <= b + c) and (b <= c + a) and ((a + b + c) % 2 == 0)

    def sector_dim(self, a: Sector) -> int:
        # dim = 2 * J + 1 = jj + 1
        return a[0] + 1

    def batch_sector_dim(self, a: SectorArray) -> npt.NDArray[np.int_]:
        # dim = 2 * J + 1 = jj + 1
        if len(a) == 0:
            return np.zeros([0], dtype=int)
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
        # all sectors are self-dual
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        # OPTIMIZE: jutho has a special case if all sectors are trivial ...?
        from . import _su2data
        return _su2data.f_symbol(a[0], b[0], c[0], d[0], e[0], f[0])

    def frobenius_schur(self, a: Sector):
        # +1 for integer spin (i.e. even `a`), -1 for half integer
        return 1 - 2 * (a[0] % 2)

    def qdim(self, a: Sector) -> float:
        return a[0] + 1

    # OPTIMIZE implement b symbol? cache it?

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # R symbol is +1 if ``j_sum = (j_a + j_b - j_c)`` is even, -1 otherwise.
        # Note that (j_a + j_b - j_c) is integer by fusion rule and that e.g. ``a == 2 * j_a``.
        # For even (odd) j_sum, we get that ``(a + b - c) % 4`` is 0 (2),
        # such that ``1 - (a + b - c) % 4`` is 1 (-1). It has shape ``(1,)``.
        return 1 - (a + b - c) % 4

    # OPTIMIZE implement c symbol? cache it?

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> np.ndarray:
        from . import _su2data
        X = _su2data.fusion_tensor(a[0], b[0], c[0])
        if Z_a:
            X = np.tensordot(self.Z_iso(a), X, (1, 1))  # [m_a, mu, m_b, m_c]
            X = np.transpose(X, [1, 0, 2, 3])
        if Z_b:
            X = np.tensordot(self.Z_iso(b), X, (1, 2))  # [m_b, mu, m_a, m_c]
            X = np.transpose(X, [1, 2, 0, 3])
        return X

    def Z_iso(self, a: Sector) -> np.ndarray:
        # TODO double check that the formula is correct!
        # Claim : Z_a : <m| \mapsto (-1)^(j + m) |-m>
        # Note to extend *linearly*, not anti-linearly.
        from . import _su2data
        return _su2data.Z_iso(a[0])


class SUNSymmetry(GroupSymmetry):
    """SU(N) group symmetry

    The sectors are lists of length N which correspond to first rows of normalized Gelfand-Tsetlin patterns
     (see https://arxiv.org/pdf/1009.0437 ).
     E.g. for SU(3) the 8 dimensional irreducible representation is labeled by [2,1,0]

     Clebsch Gordan coefficients and F/R symbols need to be calculated within the clebsch_gordan_coefficients package and exported as hdf5 file.

     CGfile: hdf5 file containing the clebsch gordan coefficients
     Ffile: hdf5 file containing the F symbols
     Rfile: hdf5 file containing the R Symbols

    """

    def __init__(self, N:int, CGfile, Ffile, Rfile):

        assert isinstance(N, int)
        if not isinstance(N, int) and N > 1:
            raise ValueError("Invalid N!")

        if not N == CGfile.attrs['N'] or not N == Ffile.attrs['N'] or not N == Rfile.attrs['N']:
            raise ValueError("Files must contain data for same N!")
        self.N = N
        self.CGfile = CGfile
        self.Ffile = Ffile
        self.Rfile = Rfile

        GroupSymmetry.__init__(self,
                               fusion_style=FusionStyle.general,
                               trivial_sector=np.array([0]*N, dtype=int),
                               group_name=f'SU({N})',
                               num_sectors=np.inf,
                               descriptive_name=None)

    def is_valid_sector(self, a: Sector) -> bool:
        l = (type(a) == Sector)
        m = (len(a) == 3)
        n = (a[-1] == 0)
        return l and m and n


    def sector_dim(self, a: Sector) -> int:
        '''
        dimension of irrep given as first row of GT pattern
        '''

        N = len(a)

        dim = 1

        for kp in range(2, N + 1):
            for k in range(1, kp):
                dim *= (1 + ((a[k - 1] - a[kp - 1]) / (kp - k)))

        return int(dim)


    def dual_sector(self, a: Sector) -> Sector:
        '''
        Finds the dual irrep for a given input irrep. if the irrep is self dual, then the input irrep is returned.
        Dual irreps have the same highest weight and dimension.

        :param a: irrep i.e. first row of a GT pattern
        :return: dual irrep
        '''

        def gen_irreps(N, k):
            '''generates a list of all possible irreps for given N and highest weight k'''

            if N <= 0:
                return [[]]
            r = []
            for i in range(k, -1, -1):
                for comb in gen_irreps(N - 1, i):
                    a = [i] + comb
                    if a[-1] == 0:
                        r.append(a[:])
            return r

        hweight = a[0]
        dimA = self.sector_dim(a)

        irreps = gen_irreps(len(a), hweight)
        irreps = [k for k in irreps if a[0] == hweight]

        for i in irreps:
            dimI = self.sector_dim(i)
            if dimI == dimA and not i == a:
                return i

        return a

    # def N_from_CG(self) -> int:
    #     """Returns the N in SU(N) from a given hdf5 file containing the CG coefficients"""
    #
    #     return int(self.CGfile.attrs['N'])

    def hweight_from_CG_hdf5(self) -> int:
        return int(self.CGfile.attrs['Highest_Weight'])

    def hweight_from_F_hdf5(self) -> int:
        return int(self.Ffile.attrs['Highest_Weight'])

    def hweight_from_R_hdf5(self) -> int:
        return int(self.Rfile.attrs['Highest_Weight'])

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        """Returns True if c appears at least once in the decomposition of a x b and False otherwise

            Parameters:
            -----------
            a,b: lists, labeling an irrep i.e. first row of GT pattern e.g. [1,0]
            c: list, labeling an irrep i.e. first row of GT pattern e.g. [1,0]
            file: hdf5 file containing the Clebsch Gordan coefficients
            """

        hmax = self.hweight_from_CG_hdf5()
        if a[0] > hmax or b[0] > hmax:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        if c[0] > a[0] + b[0]:
            return False

        N = '/N_' + str(self.N) + '/'

        astr = "/".join(tuple(map(str, a))) + '/'
        bstr = "/".join(tuple(map(str, b))) + '/'

        key = N + astr + bstr

        if not key in self.CGfile:
            key = N + bstr + astr

        dec = []

        for i in list(self.CGfile[key]):
            dec.append(list(self.CGfile[key][str(i)].attrs['Irreplabel']))

        if c in dec:
            return True

        return False

    def fusion_multiplicity(self, a: Sector, b: Sector, c: Sector) -> int:
        """returns the fusion multiplicity of an irrep c in the decomposition of a x b specified in the file

        Parameters:
        -----------
        a,b: lists, labeling an irrep i.e. first row of GT pattern e.g. [1,0]
        c: list, labeling an irrep i.e. first row of GT pattern e.g. [1,0]
        file: hdf5 file containing the Clebsch Gordan coefficients

        Returns:
        --------
        Fusion multiplicity (value of N^{ab}_c symbol) as integer
        """

        N = '/N_' + str(self.N) + '/'

        a = "/".join(tuple(map(str, a))) + '/'
        b = "/".join(tuple(map(str, b))) + '/'

        c = 'Irrep' + "".join(map(str, c)) + 'a1'

        key = N + a + b

        if not key in self.CGfile:
            key = N + b + a

        if c not in list(self.CGfile[key]):
            return 0

        return self.CGfile[key][c].attrs['Outer Multiplicity']

    def S_index_irrep_weight(self, a: Sector) -> int:
        """ to every su(N) irrep, labeled by the first row of a GT pattern, we can assign an integer number S
        """

        N = self.N
        S = 0

        for k in range(1, N):
            S += math.comb(N - k + a[k - 1] - 1, N - k)

        return int(S)

    def highest_irrep_in_decomp(self, a: Sector, b: Sector) -> Sector:
        '''
        Returns the highest irrep (i.e. first row of GT pattern) which appears in the decomposition of irrep1 x irrep2.
        '''

        #assert len(a) == len(b)
        return np.array(a) + np.array(b)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        """Returns a SectorArray of all irreps appearing in the decomposition of  a x b
        The irreps in this list are again Sectors of the form [2,1,0]. i.e. first rows of a GT pattern
         """

        hmax = self.hweight_from_CG_hdf5()
        if a[0] > hmax or b[0] > hmax:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        N = '/N_' + str(self.N) + '/'

        a = "/".join(tuple(map(str, a))) + '/'
        b = "/".join(tuple(map(str, b))) + '/'

        key = N + a + b

        if not key in self.CGfile:
            key = N + b + a

        dec = np.array([])
        for i in list(self.CGfile[key]):
            np.append(dec,np.array(self.CGfile[key][str(i)].attrs['Irreplabel']), axis=1)

        return dec

    def dims_of_irreps(self, a: Sector, b: Sector) -> dict:
        """Returns a dictionary with irreps as keys and their dimension as values.
        The irreps are the ones appearing in the decomposition of a x b
        Does not contain multiplicities!
        """

        dec = self.fusion_outcomes(a, b)
        N = '/N_' + str(self.N) + '/'

        a = "/".join(tuple(map(str, a))) + '/'
        b = "/".join(tuple(map(str, b))) + '/'

        key = N + a + b

        C = {}
        keys = []

        for i in dec:
            keys.append(tuple(i))

        for k in keys:
            obj = 'Irrep' + ''.join(map(str, k)) + 'a1'
            C[k] = int(self.CGfile[key][obj].attrs['Dimension'])

        return C

    def outer_multiplicity_from_CG(self, a: Sector, b: Sector) -> dict:
        '''

        returns a dictionary with the outer multiplicities for the corresp. irrep (as key) in the decomp of a x b
        '''

        dec = self.fusion_outcomes(a, b)
        N = '/N_' + str(self.N) + '/'

        a = "/".join(tuple(map(str, a))) + '/'
        b = "/".join(tuple(map(str, b))) + '/'

        key = N + a + b

        C = {}
        keys = []

        for i in dec:
            keys.append(tuple(i))

        for k in keys:
            obj = 'Irrep' + ''.join(map(str, k)) + 'a1'
            C[k] = int(self.CGfile[key][obj].attrs['Outer Multiplicity'])

        return C

    def clebschgordan(self, a: Sector, q_a: int, b: Sector, q_b: int, c: Sector, q_c: int, mu: int) -> float:
        '''

        :param a:       irrep a
        :param q_a:     index of the Gelfand Tsetlin pattern
        :param b:       irrep b
        :param q_b:     index of the Gelfand Tsetlin pattern
        :param c:       irrep c
        :param q_c:     index of the Gelfand Tsetlin pattern
        :param mu:      multiplicity index 1 <= mu
        :param file:    the file that contains the CG coefficients
        :return:        the CG coefficient for the given input
        '''


        hw = self.hweight_from_CG_hdf5()

        if a[0] > hw or b[0] > hw or c[0] > hw:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')


        N = '/N_' + str(self.N) + '/'

        a = "/".join(tuple(map(str, a))) + '/'
        b = "/".join(tuple(map(str, b))) + '/'

        c = "".join(map(str, c))

        ms = [float(q_a), float(q_b), float(q_c)]

        key1 = N + a + b
        key2 = 'Irrep' + c + 'a' + str(mu)

        if key1 in self.CGfile:
            arr = np.array(self.CGfile[key1][key2])[0]
        else:
            key1 = N + b + a  # we only save a x b  and not also b x a since the clebsch gordan coefficients are the same in both cases
            arr = np.array(self.CGfile[key1][key2])[0]

            ms = [float(q_b), float(q_a), float(q_c)]

        for i in range(len(arr)):
            if list(arr[i][0:3]) == ms:
                return arr[i][3]

        return 0.


    def fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool = False, Z_b: bool = False
                      ) -> np.ndarray:
        '''
        a,b,c are irreps (first rows of GT patterns)
        CG_coeffs is the np.array from CG code for decomposition a x b = mu * c

        '''


        hw = self.hweight_from_CG_hdf5()

        if a[0] > hw or b[0] > hw or c[0] > hw:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')


        dim_Sa = self.sector_dim(a)
        dim_Sb = self.sector_dim(b)
        dim_Sc = self.sector_dim(c)
        dim_mu = self.fusion_multiplicity(a, b, c)

        if dim_mu == 0:
            return np.zeros((dim_Sa, dim_Sb, dim_Sc, 1), dtype=np.float64)

        X = np.zeros((dim_Sa, dim_Sb, dim_Sc, dim_mu), dtype=np.float64)

        for m_a in range(1, dim_Sa + 1):
            for m_b in range(1, dim_Sb + 1):
                for m_c in range(1, dim_Sc + 1):
                    for mu in range(1, dim_mu + 1):
                        # print(m_a,m_b,m_c)
                        # print(clebschgordan(a, m_a, b, m_b, c, m_c, mu, file))
                        rr = self.clebschgordan(a, m_a, b, m_b, c, m_c, mu)
                        X[m_a - 1, m_b - 1, m_c - 1, mu - 1] = rr

        return X

    def _f_symbol_from_CG(self,a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector):
        """a,b,c,d,e,f are irrep labels, i.e. first rows of GT patterns
        output is the conjugated F symbol [F^{abc}_{def}]^*_{mu,nu,kappa, lambda}
        where a x b = mu c, c x d =nu e, b x d= kappa f and a x f =lambda e """


        hw = self.hweight_from_CG_hdf5()

        if a[0] > hw or b[0] > hw or c[0] > hw or d[0] > hw or e[0] > hw or f[0] > hw:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        X1 = self.fusion_tensor(a, b, f)  # [a,b,f, kappa]
        X2 = self.fusion_tensor(f, c, d)  # [f,c,d, lambda]

        X3 = self.fusion_tensor(b, c, e)  # [b,c,e, mu]
        X4 = self.fusion_tensor(a, e, d)  # [a,e,d, nu]

        if not X1.any() or not X2.any() or not X3.any() or not X4.any():
            return np.zeros((1, 1, 1, 1), dtype=complex)

        X12 = np.tensordot(X1, X2, axes=[[2], [0]])  # [a,b,[f], kappa] ; [[f],c,d, lambda] --> [a,b,kappa,c,d, lambda]
        X12 = X12.transpose([0, 1, 3, 4, 2, 5])  # [a,b,c,d,kappa,lambda]

        X34 = np.tensordot(X3, X4, axes=[[2], [1]])  # [b,c,[e], mu] ; [a,[e],d, nu] --> [b,c,mu,a,d,nu]
        X34 = X34.transpose([3, 0, 1, 4, 2, 5])  # [a,b,c,d,mu,nu]

        F = np.tensordot(X12, np.conj(X34), axes=[[0, 1, 2, 3], [0, 1, 2, 3]])  # [a,b,c,d,kappa,lambda] ; [a,b,c,d,mu,nu] --> [kappa,lambda,mu,nu]

        F = F.transpose([2, 3, 0, 1])  # [mu, nu, kappa, lambda]

        F[np.abs(F) < (10 ** -12)] = 0

        return F / ((self.sector_dim(d) + 0.j))

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                 ) -> np.ndarray:
        '''
        Returns the F-symbol F^{abc mu nu}_{def kappa lambda} from the hdf5 file
        '''

        hmax = self.hweight_from_F_hdf5()

        if a[0] > hmax or b[0] > hmax or c[0] > hmax or d[0] > hmax or e[0] > hmax or f[0] > hmax:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        abar = self.dual_sector(a)
        bbar = self.dual_sector(b)
        cbar = self.dual_sector(c)
        dbar = self.dual_sector(d)
        ebar = self.dual_sector(e)
        fbar = self.dual_sector(f)

        key = 'F' + str(a) + str(b) + str(c) + str(d) + str(e) + str(f)
        keybar = 'F' + str(abar) + str(bbar) + str(cbar) + str(dbar) + str(ebar) + str(fbar)

        if key in self.Ffile['/F_sym/']:
            return np.array(self.Ffile['/F_sym/'][key])

        elif keybar in self.Ffile['/F_sym/']:
            return np.array(self.Ffile['/F_sym/'][keybar])

        return np.zeros((1, 1, 1, 1), dtype=complex)

    def _r_symbol_from_CG(self, a: Sector ,b: Sector ,c: Sector):
        """a,b,c are irrep labels, i.e. first rows of GT patterns
        output is the R symbol [R^{ab}_{c}]^*_{mu,nu}
        where a x b = mu c, c x d =nu e, b x d= kappa f and a x f =lambda e """

        hw = self.hweight_from_CG_hdf5()

        mult = self.fusion_multiplicity(a, b, c)

        if a[0] > hw or b[0] > hw or c[0] > hw:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        X1 = self.fusion_tensor(a, b, c)  # [a,b,c, nu]
        Y1 = self.fusion_tensor(b, a, c).conj()  # [b,a,c,mu]

        if not X1.any() or not Y1.any():
            return np.zeros((1), dtype=complex)

        R = np.tensordot(X1, Y1, axes=[[0, 1, 2], [1, 0, 2]])  # [[a],(b),{c}, nu] , [(b),[a],{c},mu] --> [nu,mu]

        R = R.transpose([1, 0]) / ((self.sector_dim(c) + 0.j))

        return np.diag(R)

    def _r_symbol(self, a: Sector, b: Sector, c: Sector):
        '''
        Returns the R-symbol R^{ab}_{c mu} from the hdf5 file
        '''

        hmax = self.hweight_from_R_hdf5()

        if a[0] > hmax or b[0] > hmax or c[0] > hmax:
            raise ValueError('Input irreps have higher weight than highest weight irrep in HDF5-file')

        key = 'F' + str(a) + str(b) + str(c)

        if key in self.Rfile['/R_sym/']:
            return np.array(self.Rfile['/R_sym/'][key])

        return np.zeros((1,), dtype=complex)


class FermionParity(Symmetry):
    """Fermionic Parity.

    Allowed sectors are arrays with a single entry; either ``[0]`` (even) or ``1`` (odd).
    The parity is the number of fermions in a given state modulo 2.
    """
    fusion_tensor_dtype = Dtype.float64
    _one_2D = np.ones((1, 1), dtype=int)
    _one_2D_float = np.ones((1, 1), dtype=float)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)
    _one_4D_float = np.ones((1, 1, 1, 1), dtype=float)

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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % 2

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
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
        # sqrt(d_b) [F^{a b dual(b)}_a]^{111}_{c,mu,nu} = sqrt(1) * 1 = 1
        return self._one_2D

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # if a and b are fermionic -1, otherwise +1
        # in the first (second) case above, we have ``a * b`` equal to 1 (0).
        return 1 - 2 * a * b

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        # R^{ec}_d conj(R)^{ca}_f
        C = (1 - 2 * e * c) * (1 - 2 * c * a)
        return C[None, None, None, :]

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]

    def Z_iso(self, a: Sector) -> np.ndarray:
        return self._one_2D_float


class ZNAnyonCategory(Symmetry):
    r"""Abelian anyon category with fusion rules corresponding to the Z_N group;
    also written as :math:`Z_N^{(n)}`.

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon category corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """
    _one_1D = np.ones((1,), dtype=int)
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
                          group_name='ZNAnyonCategory',
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'ZNAnyonCategory(N={self.N}, n={self.n})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, ZNAnyonCategory) and other.N == self.N and other.n == self.n

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
        return self._phase ** (a * b)

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase**(b[0] * c[0]) * self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class ZNAnyonCategory2(Symmetry):
    r"""Abelian anyon category with fusion rules corresponding to the Z_N group;
    also written as :math:`Z_N^{(n+1/2)}`. `N` must be even.

    .. todo ::
        include in `ZNAnyonCategory`?

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon category corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """
    _one_1D = np.ones((1,), dtype=int)
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
                          group_name='ZNAnyonCategory2',
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'ZNAnyonCategory2(N={self.N}, n={self.n})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, ZNAnyonCategory2) and other.N == self.N and other.n == self.n

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
        return self._phase ** (a * b) * self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return (self._phase**(b[0] * c[0])) * self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class QuantumDoubleZNAnyonCategory(Symmetry):
    """Doubled abelian anyon category with fusion rules corresponding to the Z_N group;
    also written as :math:`D(Z_N)`.

    Allowed sectors are 1D arrays with two integers between `0` and `N-1`.
    `[0, 0]`, `[0, 1]`, ..., `[N-1, N-1]`

    This is not a simple product for two `ZNAnyonCategory`s; there are nontrivial R-symbols.
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
                          trivial_sector=np.array([0, 0], dtype=int),
                          group_name='QuantumDoubleZNAnyonCategory',
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

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        return f'QuantumDoubleZNAnyonCategory(N={self.N})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, QuantumDoubleZNAnyonCategory) and other.N == self.N

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
        return self._phase ** (a[0:1] * b[1:2])

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase**(b[0] * c[1]) * self._one_4D

    def all_sectors(self) -> SectorArray:
        x = np.arange(self.N, dtype=int)
        return np.dstack(np.meshgrid(x, x)).reshape(-1, 2)


class FibonacciAnyonCategory(Symmetry):
    """Category describing Fibonacci anyons.

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
    _r = np.expand_dims([np.exp(-4j*np.pi/5), np.exp(3j*np.pi/5)], axis=1)  # nontrivial R-symbols
    _one_1D = np.ones((1,), dtype=int)
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
                          group_name='FibonacciAnyonCategory',
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
        return f'FibonacciAnyonCategory(handedness={self.handedness})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, FibonacciAnyonCategory) and other.handedness == self.handedness

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

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, self._phi, 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[c[0], :]
        return self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            return self._c[6 * a[0] + 3 * d[0] + e[0] + f[0] - 2]
        return self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]


class IsingAnyonCategory(Symmetry):
    """Category describing Ising anyons.

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
    _one_1D = np.ones((1,), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, nu: int = 1):
        assert nu % 2 == 1
        self.nu = nu % 16
        self.frobenius = [1, int((-1)**((self.nu**2-1)/8)), 1]
        self._f = (np.expand_dims([1, 0, 1, 0, -1], axis=(1,2,3,4))
                            * self.frobenius[1] / np.sqrt(2))  # nontrivial F-symbols
        self._r = np.expand_dims([(-1j)**self.nu, -1, np.exp(3j*self.nu*np.pi/8) * self.frobenius[1],
                    np.exp(-1j*self.nu*np.pi/8) * self.frobenius[1], 0], axis=1)  # nontrivial R-symbols
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
                          group_name='IsingAnyonCategory',
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
        return f'IsingAnyonCategory(nu={self.nu})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, IsingAnyonCategory) and other.nu == self.nu

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

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, np.sqrt(2), 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[(a[0] + b[0]) * (c[0] - 1), :]
        return self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            factor = -1 * (b[0] - c[0] - 1) * (b[0] - c[0] + 1)  # = 0 if σ and ψ or σ and ψ, 1 otherwise
            factor *= ( 1 - a[0]//2 - d[0]//2 + 9 * (b[0] - 1) + (2 - b[0]) * ((e[0] + f[0])//2 + d[0]//2 + 3 * a[0]) )
            return self._c[factor + a[0]//2 + d[0]//2]
        return self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(3, dtype=int)[:, None]


class SU2_kAnyonCategory(Symmetry):
    """:math:`SU(2)_k` anyon category.

    .. todo ::
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
    _one_1D = np.ones((1,), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)

    def __init__(self, k: int, handedness = 'left'):
        assert type(k) == int
        assert handedness in ['left', 'right']
        self.k = k
        self.handedness = handedness
        self._q = np.exp(2j * np.pi / (k + 2))

        Symmetry.__init__(self,
                          fusion_style=FusionStyle.multiple_unique,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='SU2_kAnyonCategory',
                          num_sectors=self.k+1, descriptive_name=None)

        self._r = {}
        for jj1, jj2, jj in product(range(self.k + 1), repeat=3):
            if jj > jj1 + jj2 or jj < abs(jj1 - jj2) or jj1 * jj2 == 0 or jj1 < jj2:
                continue  # do not save trivial R-symbols and use symmetry jj1 <-> jj2
            factor = (-1)**((jj - jj1 - jj2) / 2)
            factor *= self._q**(( jj*(jj+2) - jj1*(jj1+2) - jj2*(jj2+2) ) / 8)
            if self.handedness == 'right':
                factor = factor.conj()
            self._r[(jj1, jj2, jj)] = factor * self._one_1D

        self._6j = {}
        for jj1, jj2, jj3, jj, jj12, jj23 in product(range(self.k + 1), repeat=6):
            if not (jj1 == np.max([jj1, jj2, jj3, jj, jj12, jj23]) and
                                    jj2 == np.max([jj2, jj, jj12, jj23])):
                continue
            jsymbol = self._j_symbol(jj1, jj2, jj12, jj3, jj, jj23)
            if jsymbol != 0:
                self._6j[(jj1, jj2, jj12, jj3, jj, jj23)] = jsymbol

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
        return f'SU2_kAnyonCategory({self.k}, {self.handedness})'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, SU2_kAnyonCategory) and other.k == self.k and other.handedness == self.handedness

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        # The q-deformed 6j symbols have the same symmetries as the usual SU(2) 6j symbols.
        # We can get all f symbols from the cases 6j symbols for
        # a == np.max([a, b, c, d, e, f]) and b == np.max([b, c, e, f]).
        # I.e., we need to exchange the charges accordingly

        # need to compute before exchanging charges
        factor = np.sqrt( self._n_q(e[0]+1) * self._n_q(f[0]+1) )
        factor *= (-1)**((a[0] + b[0] + c[0] + d[0]) / 2)

        argm = np.argmax([a, c, b, d, f, e])
        if argm > 1:
            if argm // 2 == 1:
                a, c, b, d = b, d, a, c
            else:
                a, c, f, e = f, e, a, c

        argm_ = np.argmax([b, d, f, e])
        if argm_ > 1:
            b, d, f, e = f, e, b, d

        if argm % 2 == 1 and argm_  % 2 == 1:
            a, c, b, d = c, a, d, b
        elif argm % 2 == 1:
            a, c, f, e = c, a, e, f
        elif argm_ % 2 == 1:
            b, d, f, e = d, b, e, f

        try:  # nontrivial F-symbols
            return factor * self._6j[(a[0], b[0], f[0], c[0], d[0], e[0])] * self._one_4D
        except KeyError:
            return self._one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return -1 if a[0] % 2 == 1 else 1

    def qdim(self, a: Sector) -> float:
        return np.sin((a[0] + 1) * np.pi / (self.k + 2)) / np.sin(np.pi / (self.k + 2))

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.sin((a.flatten() + 1) * np.pi / (self.k + 2)) / np.sin(np.pi / (self.k + 2))

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if a[0] < b[0]:
            a, b = b, a
        try:  # nontrivial R-symbols
            return self._r[(a[0], b[0], c[0])]
        except KeyError:
            return self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return super()._c_symbol(a, b, c, d, e, f)

    def all_sectors(self) -> SectorArray:
        return np.arange(self.k + 1, dtype=int)[:, None]


class SU3_3AnyonCategory(Symmetry):
    r""":math:`SU(3)_3` anyon category

    Can be used as a good first check for categories with higher fusion multiplicities.

    The anyons are denoted by `1`, `8`, `10` and `\bar{10}` with the fusion rule
    `8 x 8 = 1 + 8 + 8 + 10 + 10-`. (For convenience, we denote `\bar{10}` as `10-`)
    The anyons correspond to the allowed sectors (1D arrays) ``[j]`` with `j = 0,1,2,3`.

    The notion of handedness does not make sense for this specific anyon model since it
    only exchanges the two fusion multiplicities of anyon `8`.
    """
    _one_1D = np.ones((1,), dtype=int)
    _one_4D = np.ones((1, 1, 1, 1), dtype=int)
    _fusion_map = {  # notation: 10- = \bar{10}
        0: np.array([[0]]),  # 1 x 1 = 1
        1: np.array([[1]]),  # 1 x 8 = 8 = 8 x 1
        4: np.array([[2]]),  # 1 x 10 = 10 = 1 x 10
        9: np.array([[3]]),  # 1 x 10- = 10- = 1 x 10-
        2: np.array([[0], [1], [2], [3]]),  # 8 x 8 = 1 + 8 + 8 + 10 + 10-
        5: np.array([[1]]),  # 8 x 10 = 8 = 10 x 8
        10: np.array([[1]]), # 8 x 10- = 8 = 10- x 8
        8: np.array([[3]]),  # 10 x 10 = 10-
        13: np.array([[0]]), # 10 x 10- = 1 = 10- x 10
        18: np.array([[2]])  # 10- x 10- = 10
    }
    _dual_map = {0: np.array([0]),
                 1: np.array([1]),
                 2: np.array([3]),
                 3: np.array([2])}
    _f1 = np.identity(2)
    _f2 = np.array([[-0.5, -3**0.5/2],
                    [3**0.5/2, -0.5]])
    _f3 = _f2.T
    _f4 = np.zeros((7,7))
    _f4[0,0] = _f4[5,5] = _f4[6,5] = _f4[5,6] = _f4[6,6] = 1/3
    _f4[0,5] = _f4[0,6] = _f4[5,0] = _f4[6,0] = -1/3
    _f4[0,1] = _f4[1,0] = _f4[0,4] = _f4[4,0] = 3**-0.5
    _f4[2,2] = _f4[3,2] = _f4[2,3] = _f4[3,3] = _f4[1,4] = _f4[4,1] = 0.5
    _f4[2,6] = _f4[6,3] = _f4[3,5] = _f4[5,2] = 0.5
    _f4[2,5] = _f4[5,3] = _f4[3,6] = _f4[6,2] = -0.5
    _f4[1,1] = _f4[4,4] = -0.5
    _f4[1,5] = _f4[1,6] = _f4[5,1] = _f4[6,1] = 12**-0.5
    _f4[4,5] = _f4[4,6] = _f4[5,4] = _f4[6,4] = 12**-0.5
    _fsym_map = {}

    def __init__(self):
        self._c = {}
        Symmetry.__init__(self,
                          fusion_style=FusionStyle.general,
                          braiding_style=BraidingStyle.anyonic,
                          trivial_sector=np.array([0], dtype=int),
                          group_name='SU3_3AnyonCategory',
                          num_sectors=4, descriptive_name=None)

        for charges in product(range(4), repeat=6):
            a, b, c, d, e, f = [np.array([i]) for i in charges]
            self._fsym_map[(a[0], b[0], c[0], d[0], e[0], f[0])] = self._compute_f_symbol(a, b, c, d, e, f)

        for charges in product(range(4), repeat=6):
            a, b, c, d, e, f = [np.array([i]) for i in charges]
            if (self.can_fuse_to(a, b, e) and self.can_fuse_to(e, c, d) and
                    self.can_fuse_to(a, c, f) and self.can_fuse_to(f, b, d)):
                self._c[(a[0], b[0], c[0], d[0], e[0], f[0])] = super()._c_symbol(a, b, c, d, e, f)

    def _compute_f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        if not np.all([self.can_fuse_to(b, c, e), self.can_fuse_to(a, e, d),
                       self.can_fuse_to(a, b, f), self.can_fuse_to(f, c, d)]):
            return self._one_4D

        abcd = [a, b, c, d]
        check_8 = [charge == np.array([1]) for charge in abcd]
        shape = (self._n_symbol(b, c, e), self._n_symbol(a, e, d),
                 self._n_symbol(a, b, f), self._n_symbol(f, c, d))

        if check_8.count(True) == 4:
            slices = []
            for charge in [e, f]:
                if charge ==  np.array([0]):
                    slices.append(slice(0,1))
                elif charge ==  np.array([1]):
                    slices.append(slice(1,5))
                elif charge ==  np.array([2]):
                    slices.append(slice(5,6))
                else:
                    slices.append(slice(6,7))
            return self._f4[slices[1], slices[0]].reshape(shape)

        elif check_8.count(True) == 3:
            index = check_8.index(False)
            not_8 = abcd[index]
            if not_8 == self.trivial_sector:
                return self._f1.reshape(shape)
            elif (not_8 == np.array([2]) and index != 1) or (not_8 == np.array([3]) and index == 1):
                return self._f2.reshape(shape)
            else:
                return self._f3.reshape(shape)

        elif check_8.count(True) == 2 and np.all(abcd):  # two 8 and no 1
            index1 = check_8.index(True)
            check_8[index1] = False
            index2 = check_8.index(True)
            if (index2 == index1 + 1) or (index1 == 0 and index2 == 3):
                return -1 * self._one_4D

        elif check_8.count(True) == 0 and np.all(abcd):
            check_10 = [charge == np.array([2]) for charge in abcd]
            index = 1
            if check_10.count(True) == 3:
                index = check_10.index(False)
            elif check_10.count(True) == 1:
                index = check_10.index(True)
            if index == 0 or index == 2:
                return -1 * self._one_4D
        return self._one_4D

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 4

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 4)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0]**2 + b[0]**2]

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def sector_str(self, a: Sector) -> str:
        if a[0] == 1:
            return '8'
        elif a[0] == 2:
            return '10'
        return 'vac' if a[0] == 0 else '10-'

    def __repr__(self):
        return f'SU3_3AnyonCategory()'

    def is_same_symmetry(self, other) -> bool:
        return isinstance(other, SU3_3AnyonCategory)

    def dual_sector(self, a: Sector) -> Sector:
        return self._dual_map[a[0]]

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return np.where(sectors >= 2, -sectors % 5, sectors)

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 2 if np.all(np.concatenate([a, b, c]) == np.array([[1]*3])) else 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
                  ) -> np.ndarray:
        return self._fsym_map[(a[0], b[0], c[0], d[0], e[0], f[0])]

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 3 if a[0] == 1 else 1

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, 3, 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b]) == np.array([[1], [1]])):
            if c == np.array([1]):
                return np.array([-1j, 1j])
            return -1 * self._one_1D
        return self._one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        try:
            return self._c[(a[0], b[0], c[0], d[0], e[0], f[0])]
        except KeyError:  # inconsistent fusion
            return self._one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(4, dtype=int)[:, None]


# Note : some symmetries have expensive __init__ ! Do not initialize those.
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
fermion_parity = FermionParity()
semion_category = ZNAnyonCategory2(2, 0)
toric_code_category = QuantumDoubleZNAnyonCategory(2)
double_semion_category = ProductSymmetry([ZNAnyonCategory2(2, 0), ZNAnyonCategory2(2, 1)])
fibonacci_anyon_category = FibonacciAnyonCategory(handedness='left')
ising_anyon_category = IsingAnyonCategory(nu=1)
