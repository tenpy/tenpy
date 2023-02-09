from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeVar, Any

from tenpy.linalg.symmetries import AbstractSymmetry, ProductSpace, VectorSpace, AbstractSpace


# TODO make Dtype.float32
class Precision(Enum):
    half = auto()  # 16 bit per float
    single = auto()  # 32 bit per float
    double = auto()  # 64 bit per float
    long_double = auto()  # C standard `long double`, may be 96 or 128 bit  TODO make 96 AND 128, raise if unavailable
    quadruple = auto()  # 128 bit per float


@dataclass
class Dtype:
    precision: Precision
    is_real: bool

    def __repr__(self):
        return f'Dtype(Precision.{self.precision.name}, is_real={self.is_real})'

    def as_real(self):
        return Dtype(self.precision, is_real=True)

    def as_complex(self):
        return Dtype(self.precision, is_real=False)


BackendDtype = TypeVar('BackendDtype')
BackendArray = TypeVar('BackendArray')  # placeholder for a backend-specific type that represents (symmetric) tensors
Block = TypeVar('Block')  # placeholder for a backend-specific type that represents the blocks of symmetric tensors


class AbstractBackend(ABC):
    """
    Inheritance structure:

            AbstractBackend           AbstractBlockBackend
                  |                            |
          AbstractXxxBackend            YyyBlockBackend
                  |                            |
                  ------------------------------
                                |
                          XxxYyyBackend

    Where Xxx describes the symmetry, e.g. NoSymmetry, Abelian, Nonabelian
    and Yyy describes the numerical routines that handle the blocks, e.g. numpy, torch, ...
    """
    default_precision: Precision

    def __init__(self, symmetry: AbstractSymmetry):
        self.symmetry = symmetry

    def __repr__(self):
        return f'{type(self).__name__}(symmetry={repr(self.symmetry)})'

    def __str__(self):
        return f'{type(self).__name__}({self.symmetry.short_str()})'

    @abstractmethod
    def parse_data(self, obj, legs: list[VectorSpace], dtype: BackendDtype = None
                   ) -> tuple[BackendArray, tuple[int]]:
        """Extract backend-specific data structure from arbitrary python object, if possible.
        Return this data and its shape.
        Raise TypeError or ValueError if obj can not be interpreted as an array."""
        ...

    @abstractmethod
    def parse_dtype(self, dtype: Dtype | None) -> BackendDtype:
        """Translate Dtype instance to a backend-specific format"""
        ...

    @abstractmethod
    def infer_dtype(self, data: BackendArray) -> Dtype:
        ...

    @abstractmethod
    def to_dtype(self, data: BackendArray, dtype: Dtype) -> BackendArray:
        ...

    @abstractmethod
    def is_real(self, data: BackendArray) -> bool:
        """If the data is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        ...

    @abstractmethod
    def legs_are_compatible(self, data: BackendArray, legs: list[VectorSpace]) -> bool:
        """Whether a given list of vector spaces is compatible with the data"""
        ...

    @abstractmethod
    def tdot(self, a: BackendArray, b: BackendArray, a_axes: list[int], b_axes: list[int]
             ) -> BackendArray:
        ...

    @abstractmethod
    def item(self, data: BackendArray) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def to_dense_block(self, data: BackendArray) -> Block:
        """Forget about symmetry structure and convert to a single block."""
        ...

    @abstractmethod
    def reduce_symmetry(self, data: BackendArray, new_symm: AbstractSymmetry) -> BackendArray:
        """Convert to lower symmetry group. TODO what additional info do we need?"""
        ...

    @abstractmethod
    def increase_symmetry(self, data: BackendArray, new_symm: AbstractSymmetry, atol=1e-8, rtol=1e-5
                          ) -> BackendArray:
        """Convert to higher symmetry, if data is symmetric under it.
        If data is not symmetric under the higher symmetry i.e. if
        norm(old - projected) >= atol + rtol * norm(old), raise a ValueError"""
        ...

    @abstractmethod
    def copy_data(self, data: BackendArray) -> BackendArray:
        """Return a copy, such that future in-place operations on the output data do not affect the input data"""
        ...

    @abstractmethod
    def _data_repr_lines(self, data: BackendArray, indent: str, max_width: int, max_lines: int):
        ...

    @abstractmethod
    def svd(self, a: BackendArray, idcs1: list[int], idcs2: list[int]
            ) -> tuple[BackendArray, BackendArray, BackendArray, VectorSpace]:
        """

        Returns
        -------
        u, s, vh, new_space
        """
        ...

    @abstractmethod
    def outer(self, a: BackendArray, b: BackendArray) -> BackendArray:
        ...

    @abstractmethod
    def inner(self, a: BackendArray, b: BackendArray, axs2: list[int] | None) -> complex:
        # inner product of <a|b>, both of which are given as ket-like vectors
        # (i.e. in C^N, the entries of a would need to be conjugated before multiplying with entries of b)
        # axs2, if not None, gives the order of the axes of b
        ...

    @abstractmethod
    def transpose(self, a: BackendArray, permutation: list[int]) -> BackendArray:
        ...

    @abstractmethod
    def trace(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        ...

    @abstractmethod
    def conj(self, a: BackendArray) -> BackendArray:
        ...

    @abstractmethod
    def combine_legs(self, a: BackendArray, legs: list[int], old_legs: list[VectorSpace], 
                     new_leg: ProductSpace) -> BackendArray:
        """combine legs of a. resulting leg takes position of legs[0]"""
        ...

    @abstractmethod
    def split_leg(self, a: BackendArray, leg_idx: int, leg: ProductSpace) -> BackendArray:
        """split a leg. resulting legs all take place of leg"""
        ...

    @abstractmethod
    def allclose(self, a: BackendArray, b: BackendArray, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def squeeze_legs(self, a: BackendArray, idcs: list[int]) -> BackendArray:
        ...

    @abstractmethod
    def norm(self, a: BackendArray) -> float:
        ...

    @abstractmethod
    def exp(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        ...

    @abstractmethod
    def log(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        ...

    @abstractmethod
    def random_uniform(self, legs: list[AbstractSpace], dtype: Dtype) -> BackendArray:
        ...

    @abstractmethod
    def random_gaussian(self, legs: list[AbstractSpace], dtype: Dtype, sigma: float) -> BackendArray:
        ...

    @abstractmethod
    def add(self, a: BackendArray, b: BackendArray) -> BackendArray:
        ...

    @abstractmethod
    def mul(self, a: float | complex, b: BackendArray) -> BackendArray:
        ...


class AbstractBlockBackend(ABC):
    svd_algorithms: list[str]  # first is default

    def __init__(self, default_precision: Precision):
        self.default_precision = default_precision

    @abstractmethod
    def parse_block(self, obj, dtype: BackendDtype = None) -> Block:
        """Extract a block from arbitrary python object, if possible.
        Raise TypeError or ValueError if not."""
        ...

    @abstractmethod
    def block_is_real(self, a: Block):
        """If the block is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        ...

    @abstractmethod
    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]
                   ) -> Block:
        ...

    @abstractmethod
    def block_shape(self, a: Block) -> tuple[int]:
        ...

    @abstractmethod
    def block_item(self, a: Block):
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def block_dtype(self, a: Block) -> Dtype:
        ...

    @abstractmethod
    def block_to_dtype(self, a: Block, dtype: BackendDtype) -> Block:
        ...

    @abstractmethod
    def block_copy(self, a: Block) -> Block:
        ...

    @abstractmethod
    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        ...

    @abstractmethod
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """SVD of a 2D block"""
        ...

    @abstractmethod
    def block_outer(self, a: Block, b: Block) -> Block:
        ...

    @abstractmethod
    def block_inner(self, a: Block, b: Block, axs2: list[int] | None) -> complex:
        ...

    @abstractmethod
    def block_transpose(self, a: Block, permutation: list[int]) -> Block:
        ...

    @abstractmethod
    def block_trace(self, a: Block, idcs1: list[int], idcs2: list[int]) -> Block:
        ...

    @abstractmethod
    def block_conj(self, a: Block) -> Block:
        """complex conjugate of a block"""
        ...

    @abstractmethod
    def block_combine_legs(self, a: Block, legs: list[int]) -> Block:
        ...

    @abstractmethod
    def block_split_leg(self, a: Block, leg: int, dims: list[int]) -> Block:
        ...

    @abstractmethod
    def block_allclose(self, a: Block, b: Block, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        ...

    @abstractmethod
    def block_norm(self, a: Block) -> float:
        ...

    @abstractmethod
    def block_matrixify(self, a: Block, idcs1: list[int], idcs2: list[int]) -> tuple[Block, Any]:
        """reshape to a matrix. return that matrix and data necessary to revert it"""
        ...

    @abstractmethod
    def block_dematrixify(self, matrix: Block, aux: Any) -> Block:
        ...

    @abstractmethod
    def matrix_exp(self, matrix: Block) -> Block:
        ...

    @abstractmethod
    def matrix_log(self, matrix: Block) -> Block:
        ...

    @abstractmethod
    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def block_random_gaussian(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        ...

    def block_add(self, a: Block, b: Block) -> Block:
        return a + b

    def block_mul(self, a: float | complex, b: Block) -> Block:
        return a * b
