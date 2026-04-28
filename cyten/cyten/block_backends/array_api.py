"""Implements a generic BlockBackend that works with any library which follows the Array API.

The API standard is documented at https://data-apis.org/array-api/latest/purpose_and_scope.html
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import numpy as np

from ._block_backend import Block, BlockBackend
from .dtypes import Dtype


class ArrayApiBlockBackend(BlockBackend):
    """A block-backend based on a generic Array API compliant library"""

    svd_algorithms = ['default']  # can not specify algorithms through the array API

    def __init__(self, api_namespace, default_device: str = 'cpu'):
        self._api = api_namespace
        self.BlockCls = type(api_namespace.zero(1))

        self.cyten_dtype_map = {
            api_namespace.float32: Dtype.float32,
            api_namespace.float64: Dtype.float64,
            api_namespace.complex64: Dtype.complex64,
            api_namespace.complex128: Dtype.complex128,
            api_namespace.bool: Dtype.bool,
            None: None,
        }
        self.backend_dtype_map = {
            Dtype.float32: api_namespace.float32,
            Dtype.float64: api_namespace.float64,
            Dtype.complex64: api_namespace.complex64,
            Dtype.complex128: api_namespace.complex128,
            Dtype.bool: api_namespace.bool,
            None: None,
        }
        super().__init__(default_device=default_device)

    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False, device: str = None) -> Block:
        if device is None and not hasattr(a, 'device'):
            device = self.default_device
        block = self._api.asarray(a, dtype=self.backend_dtype_map[dtype], device=device)
        if dtype != Dtype.bool:
            # force float or complex dtype without multiplying
            block = 1.0 * block
        if return_dtype:
            return block, self.cyten_dtype_map[block.dtype]
        return block

    def as_device(self, device: str | None) -> str:
        if device is None:
            device = self.default_device
        # need to do this hack, since the API does not provide a unified way to instantiate
        # device objects. making this block also guarantees that device is valid and available.
        return str(self.ones_block([1], device=device).device)

    def block_all(self, a) -> bool:
        return self._api.all(a)

    def block_any(self, a) -> bool:
        return self._api.any(a)

    def tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return self._api.tensordot(a, b, (idcs_a, idcs_b))

    def get_shape(self, a: Block) -> tuple[int]:
        shape = a.shape
        if None in shape:
            raise RuntimeError  # array API allows "unknown" dimensions. we do not.
        return shape

    def item(self, a: Block) -> float | complex:
        if self.is_real(a):
            return float(a)
        return complex(a)

    def get_dtype(self, a: Block) -> Dtype:
        return self.cyten_dtype_map[a.dtype]

    def to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return self._api.astype(a, self.backend_dtype_map[dtype])

    def copy_block(self, a: Block, device: str = None) -> Block:
        return self._api.asarray(a, copy=True, device=device)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        lines = [f'{indent}{line}' for line in str(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def outer(self, a: Block, b: Block) -> Block:
        return self._api.tensordot(a, b, 0)

    def permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return self._api.permute_dims(a, permutation)

    def trace_full(self, a: Block) -> float | complex:
        shape = a.shape
        num_trace = len(shape) // 2
        trace_dim = np.prod(shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = self._api.reshape(self._api.permute_dims(a, perm), (trace_dim, trace_dim))
        res = self._api.linalg.trace(a)  # performs trace along last two axes
        return self.item(res)

    def trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = self._api.permute_dims(a, remaining + idcs1 + idcs2)
        trace_dim = np.prod(a.shape[len(remaining) : len(remaining) + len(idcs1)])
        a = self._api.reshape(a, (-1, trace_dim, trace_dim))
        return self._api.linalg.trace(a)

    def conj(self, a: Block) -> Block:
        return self._api.conj(a)

    def angle(self, a: Block) -> Block:
        raise NotImplementedError

    def real(self, a: Block) -> Block:
        return self._api.real(a)

    def real_if_close(self, a: Block, tol: float) -> Block:
        raise NotImplementedError

    def sqrt(self, a: Block) -> Block:
        raise NotImplementedError

    def imag(self, a: Block) -> Block:
        return self._api.imag(a)

    def exp(self, a: Block) -> Block:
        return self._api.exp(a)

    def log(self, a: Block) -> Block:
        return self._api.log(a)

    def allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        res = self._api.all(self._api.abs(a - b) <= (atol + rtol * self._api.abs(b)))
        return self.item(res)

    def squeeze_axes(self, a: Block, idcs: list[int]) -> Block:
        return self._api.squeeze(a, tuple(idcs))

    def add_axis(self, a: Block, pos: int) -> Block:
        return self._api.expand_dims(a, axis=pos)

    def norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        res = self._api.linalg.vector_norm(a, axis=axis, ord=order)
        if axis is None:
            res = self.item(res)
        return res

    def max(self, a: Block) -> float | complex:
        return self.item(self._api.max(a))

    def max_abs(self, a: Block) -> float:
        return self.item(self._api.max(self._api.abs(a)))

    def min(self, a: Block) -> float | complex:
        return self.item(self._api.min(a))

    def reshape(self, a: Block, shape: tuple[int]) -> Block:
        return self._api.reshape(a, shape)

    def matrix_dot(self, a: Block, b: Block) -> Block:
        return self._api.matmul(a, b)

    # noinspection PyTypeChecker
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        if algorithm is None:
            algorithm = 'default'

        if algorithm != 'default':
            raise ValueError(f'SVD algorithm not supported: {algorithm}')

        self._api.linalg.svd(a, full_matrices=False)

    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        return self._api.linalg.qr(a, mode='complete' if full else 'reduced')

    def matrix_exp(self, matrix: Block) -> Block:
        raise NotImplementedError(f'{self} does not support matrix_exp.')

    def matrix_log(self, matrix: Block) -> Block:
        raise NotImplementedError(f'{self} does not support matrix_log.')

    def random_uniform(self, dims: list[int], dtype: Dtype, device: str = None) -> Block:
        # API does not specify random generation, so we generate in numpy and convert
        res = np.random.uniform(-1, 1, size=dims)
        if not dtype.is_real:
            res += 1.0j * np.random.uniform(-1, 1, size=dims)
        return self._api.asarray(res, device=device)

    def random_normal(self, dims: list[int], dtype: Dtype, sigma: float, device: str = None) -> Block:
        res = np.random.normal(loc=0, scale=sigma, size=dims)
        if not dtype.is_real:
            res += 1.0j * np.random.normal(loc=0, scale=sigma, size=dims)
        return self._api.asarray(res, device=device)

    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None, device: str = None) -> Block:
        return self._api.asarray(a, dtype=self.backend_dtype_map[dtype], device=device)

    def zeros(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        return self._api.zeros(shape, dtype=self.backend_dtype_map[dtype], device=device)

    def ones_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        return self._api.ones(shape, dtype=self.backend_dtype_map[dtype], device=device)

    def eye_matrix(self, dim: int, dtype: Dtype, device: str = None) -> Block:
        return self._api.eye(dim, dtype=self.backend_dtype_map[dtype], device=device)

    def kron(self, a: Block, b: Block) -> Block:
        # is this really not in the API...?
        # should be able to do this via mul and reshape?
        raise NotImplementedError

    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        return self.item(a[tuple(idcs)])

    def get_device(self, a: Block) -> str:
        return a.device

    def get_diagonal(self, a: Block, tol: float | None) -> Block:
        assert a.ndim == 2
        res = self._api.diagonal(a)
        if tol is not None:
            if not self.allclose(a, self.block_from_diagonal(res), rtol=0, atol=tol):
                raise ValueError('Not a diagonal block.')
        return res

    def block_from_diagonal(self, diag: Block) -> Block:
        raise NotImplementedError

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        raise NotImplementedError

    def sum(self, a: Block, ax: int) -> Block:
        return self._api.sum(a, axis=ax)

    def sum_all(self, a: Block) -> float | complex:
        return self.item(self._api.sum(a))

    def eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = self._api.linalg.eigh(block)
        if sort is not None:
            perm = self.argsort(w, sort)
            w = w[perm]
            v = v[:, perm]
        return w, v

    def eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = self._api.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.argsort(w, sort)
            w = w[perm]
        return w

    def abs_argmax(self, block: Block) -> list[int]:
        flat_idx = self._api.argmax(self._api.abs(block))
        # OPTIMIZE numpy has np.unravel_indices. no analogue here?
        idcs = []
        for dim in reversed(block.shape):
            flat_idx, idx = divmod(flat_idx, dim)
            idcs.append(idx)
        return idcs

    def _argsort(self, block: Block, axis: int) -> Block:
        return self._api.argsort(block, axis=axis)

    def enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = self._api.zeros(shape, dtype=block.dtype)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[idcs] = self.copy_block(block)  # OPTIMIZE copy needed?
        return res

    def stable_log(self, block: Block, cutoff: float) -> Block:
        return self._api.where(block > cutoff, self._api.log(block), 0.0)
