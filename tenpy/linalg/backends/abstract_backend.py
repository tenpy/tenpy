from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeVar

from tenpy.linalg.symmetries import AbstractSymmetry, VectorSpace, AbstractSpace


class Precision(Enum):
    half = auto()  # 16 bit per float
    single = auto()  # 32 bit per float
    double = auto()  # 64 bit per float
    long_double = auto()  # C standard `long double`, may be 96 or 128 bit
    quadruple = auto()  # 128 bit per float


@dataclass
class Dtype:
    precision: Precision
    is_real: bool

    def __repr__(self):
        return f'Dtype(Precision.{self.precision.name}, is_real={self.is_real})'


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
    def parse_data(self, obj, dtype: BackendDtype = None) -> BackendArray:
        """Extract backend-specific data structure from arbitrary python object, if possible.
#         Raise TypeError or ValueError if not."""
        ...

    @abstractmethod
    def parse_dtype(self, dtype: Dtype) -> BackendDtype:
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
    def infer_legs(self, data: BackendArray) -> list[AbstractSpace]:
        """Infer the vector spaces, if possible"""
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
    def _data_repr_lines(self, indent: str, max_width: int, max_lines: int):
        ...


class AbstractBlockBackend(ABC):
    _dtype_map: dict[Dtype, BackendDtype]

    def __init__(self, default_precision: Precision):
        self.default_precision = default_precision

    def parse_dtype(self, dtype: Dtype) -> BackendDtype:
        try:
            return self._dtype_map[dtype]
        except KeyError:
            raise ValueError(f'dtype {dtype} not supported for {self}.') from None

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
    def block_tdot(self, a: Block, b: Block, a_axes: list[int], b_axes: list[int]
                   ) -> Block:
        ...

    @abstractmethod
    def block_shape(self, a: Block) -> tuple[int]:
        ...

    @abstractmethod
    def block_item(self, a: Block):
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...
