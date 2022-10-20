from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeVar

from tenpy.linalg.symmetries import AbstractSymmetry
from tenpy.linalg.tensors import VectorSpace


class Precision(Enum):
    default = auto()  # use the backends favorite type, usually single
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
Data = TypeVar('Data')


class AbstractBackend(ABC):

    def __init__(self, symmetry: AbstractSymmetry):
        self.symmetry = symmetry

    @abstractmethod
    def parse_dtype(self, dtype: Dtype) -> BackendDtype:
        """Translate Dtype instance to a backend-specific format"""
        ...

    @abstractmethod
    def get_dtype(self, data: Data) -> Dtype:
        """Get the dtype of the data in tenpy format, i.e. as a Dtype instance"""
        ...

    @abstractmethod
    def parse_data(self, obj) -> Data:
        """Extract backend-specific data structure from arbitrary python object, if possible.
        Raise TypeError or ValueError if not."""
        ...

    @abstractmethod
    def infer_legs(self, data: Data) -> list[VectorSpace]:
        """Infer the vector spaces, if possible"""
        ...

    @abstractmethod
    def legs_are_compatible(self, data: Data, legs: list[VectorSpace]) -> bool:
        """Whether a given list of legs is compatible with the data"""
        ...




