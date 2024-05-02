"""Utility function concerning dtypes and in particular the Dtype class."""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations

from enum import Enum
from numbers import Number
import numpy as np


__all__ = ['Dtype']


class Dtype(Enum):
    # TODO expose those in some high-level init, maybe even as tenpy.float32 ?
    # value = num_bytes * 2 + int(not is_real)
    bool = 2
    float32 = 8
    complex64 = 9
    float64 = 16
    complex128 = 17

    @property
    def is_real(dtype):
        return dtype.value % 2 == 0

    @property
    def is_complex(dtype):
        return dtype.value % 2 == 1

    @property
    def to_complex(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to complex')
        if dtype.value % 2 == 1:
            return dtype
        return Dtype(dtype.value + 1)

    @property
    def to_real(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to real')
        if dtype.value % 2 == 0:
            return dtype
        return Dtype(dtype.value - 1)

    @property
    def python_type(dtype):
        if dtype.value == 2:
            return bool
        if dtype.is_real:
            return float
        return complex

    @property
    def zero_scalar(dtype):
        return dtype.python_type(0)

    @property
    def eps(dtype):
        # difference between 1.0 and the next representable floating point number at the given precision
        if dtype.value == 2:
            raise ValueError(f'{dtype} is not inexact')
        n_bits = 8 * (dtype.value // 2)
        if n_bits == 32:
            return 2 ** -52
        if n_bits == 64:
            return 2 ** -23
        raise NotImplementedError(f'Dtype.eps not implemented for n_bits={n_bits}')

    def __repr__(self) -> str:
        return f'Dtype.{self.name}'

    def common(*dtypes):
        res = Dtype(max(t.value for t in dtypes))
        if res.is_real:
            if not all(t.is_real for t in dtypes):
                return Dtype(res.value + 1)  # = res.to_complex
        return res

    def convert_python_scalar(dtype, value) -> complex | float | bool:
        if dtype.value == 2:  # Dtype.bool
            if value in [True, False, 0, 1]:
                return bool(value)
        elif dtype.is_real:
            if isinstance(value, (int, float)):
                return float(value)
            # TODO what should we do for complex values?
        else:
            if isinstance(value, Number):
                return complex(value)
        raise TypeError(f'Type {type(value)} is incompatible with dtype {dtype}')

    def to_numpy_dtype(dtype):
        return _tenpy_dtype_to_numpy[dtype]

    @classmethod
    def from_numpy_dtype(cls, dtype):
        return _numpy_dtype_to_tenpy[dtype]


_numpy_dtype_to_tenpy = {
    np.float32: Dtype.float32,
    np.float64: Dtype.float64,
    np.complex64: Dtype.complex64,
    np.complex128: Dtype.complex128,
    np.bool_: Dtype.bool,
    np.dtype('float32'): Dtype.float32,
    np.dtype('float64'): Dtype.float64,
    np.dtype('complex64'): Dtype.complex64,
    np.dtype('complex128'): Dtype.complex128,
    np.dtype('bool'): Dtype.bool,
    None: None,
}


_tenpy_dtype_to_numpy = {
    Dtype.float32: np.float32,
    Dtype.float64: np.float64,
    Dtype.complex64: np.complex64,
    Dtype.complex128: np.complex128,
    Dtype.bool: np.bool_,
    None: None,
}
