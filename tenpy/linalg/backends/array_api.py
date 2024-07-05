"""Implements a generic BlockBackend that works with any library which follows the Array API standard
https://data-apis.org/array-api/latest/purpose_and_scope.html
"""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations

from .abstract_backend import BlockBackend, Block
from .no_symmetry import NoSymmetryBackend
from .fusion_tree_backend import FusionTreeBackend
from .abelian import AbelianBackend
from ..dtypes import Dtype

import numpy as np


__all__ = ['ArrayApiBlockBackend', 'NoSymmetryArrayApiBackend', 'AbelianArrayApiBackend',
           'FusionTreeArrayApiBackend']


# TODO carry through device args?
# TODO provide an example...


class ArrayApiBlockBackend(BlockBackend):

    svd_algorithms = ['default']  # can not specify algorithms through the array API

    def __init__(self, api_namespace):
        self._api = api_namespace
        self.BlockCls = type(api_namespace.zero(1))

        self.tenpy_dtype_map = {
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
        
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False) -> Block:
        block = self._api.asarray(a, dtype=self.backend_dtype_map[dtype])
        if dtype != Dtype.bool:
            # force float or complex dtype without multiplying
            block = 1. * block
        if return_dtype:
            return block, self.tenpy_dtype_map[block.dtype]
        return block

    def block_all(self, a) -> bool:
        return self._api.all(a)
        
    def block_any(self, a) -> bool:
        return self._api.any(a)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return self._api.tensordot(a, b, (idcs_a, idcs_b))

    def block_shape(self, a: Block) -> tuple[int]:
        shape = a.shape
        if None in shape:
            raise RuntimeError  # array API allows "unknown" dimensions. we do not.
        return shape

    def block_item(self, a: Block) -> float | complex:
        # TODO this is not part of the API spec and may fail...
        #  note that a lot of methods here depend on this...
        return a.item()

    def block_dtype(self, a: Block) -> Dtype:
        return self.tenpy_dtype_map[a.dtype]

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return self._api.astype(a, self.backend_dtype_map[dtype])

    def block_copy(self, a: Block) -> Block:
        return self._api.asarray(a, copy=True)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        # TODO i like julia style much better actually, especially for many legs
        lines = [f'{indent}{line}' for line in str(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def block_outer(self, a: Block, b: Block) -> Block:
        return self._api.tensordot(a, b, 0)

    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return self._api.permute_dims(a, permutation)

    def block_trace_full(self, a: Block) -> float | complex:
        shape = a.shape
        num_trace = len(shape) // 2
        trace_dim = np.prod(shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = self._api.reshape(self._api.permute_dims(a, perm), (trace_dim, trace_dim))
        res = self._api.linalg.trace(a)  # performs trace along last two axes
        return self.block_item(res)

    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = self._api.permute_dims(a, remaining + idcs1 + idcs2)
        trace_dim = np.prod(a.shape[len(remaining):len(remaining)+len(idcs1)])
        a = self._api.reshape(a, (-1, trace_dim, trace_dim))
        return self._api.linalg.trace(a)

    def block_conj(self, a: Block) -> Block:
        return self._api.conj(a)

    def block_angle(self, a: Block) -> Block:
        raise NotImplementedError  # TODO

    def block_real(self, a: Block) -> Block:
        return self._api.real(a)

    def block_real_if_close(self, a: Block, tol: float) -> Block:
        raise NotImplementedError  # TODO
    
    def block_sqrt(self, a: Block) -> Block:
        raise NotImplementedError  # TODO

    def block_imag(self, a: Block) -> Block:
        return self._api.imag(a)

    def block_exp(self, a: Block) -> Block:
        return self._api.exp(a)

    def block_log(self, a: Block) -> Block:
        return self._api.log(a)

    def block_allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        res = self._api.all(self._api.abs(a - b) <= (atol + rtol * self._api.abs(b)))
        return self.block_item(res)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        return self._api.squeeze(a, tuple(idcs))

    def block_add_axis(self, a: Block, pos: int) -> Block:
        return self._api.expand_dims(a, axis=pos)

    def block_norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        res = self._api.linalg.vector_norm(a, axis=axis, ord=order)
        if axis is None:
            res = self.block_item(res)
        return res

    def block_max(self, a: Block) -> float | complex:
        return self._api.max(a)
    
    def block_max_abs(self, a: Block) -> float:
        return self._api.max(self._api.abs(a))

    def block_min(self, a: Block) -> float | complex:
        return self._api.min(a)
    
    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
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

    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        # API does not specify random generation, so we generate in numpy and convert
        res = np.random.uniform(-1, 1, size=dims)
        if not dtype.is_real:
            res += 1.j * np.random.uniform(-1, 1, size=dims)
        return self._api.asarray(res)

    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        res = np.random.normal(loc=0, scale=sigma, size=dims)
        if not dtype.is_real:
            res += 1.j * np.random.normal(loc=0, scale=sigma, size=dims)
        return self._api.asarray(res)

    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None) -> Block:
        return self._api.asarray(a, dtype=self.backend_dtype_map[dtype])

    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        return self._api.zeros(shape, dtype=self.backend_dtype_map[dtype])

    def ones_block(self, shape: list[int], dtype: Dtype) -> Block:
        return self._api.ones(shape, dtype=self.backend_dtype_map[dtype])

    def eye_matrix(self, dim: int, dtype: Dtype) -> Block:
        return self._api.eye(dim, dtype=self.backend_dtype_map[dtype])

    def block_kron(self, a: Block, b: Block) -> Block:
        raise NotImplementedError  # TODO not in API...?

    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        return a[tuple(idcs)].item()

    def block_get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        raise NotImplementedError  # TODO
        # res = np.diagonal(a)
        # if check_offdiagonal:
        #     if not np.allclose(a, np.diag(res)):
        #         raise ValueError('Not a diagonal block.')
        # return res

    def block_from_diagonal(self, diag: Block) -> Block:
        raise NotImplementedError  # TODO
        # return np.diag(diag)

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        raise NotImplementedError  # TODO

    def block_sum(self, a: Block, ax: int) -> Block:
        return self._api.sum(a, axis=ax)

    def block_sum_all(self, a: Block) -> float | complex:
        return self._api.sum(a)

    def block_eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = self._api.linalg.eigh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
            v = v[:, perm]
        return w, v

    def block_eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = self._api.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
        return w
            
    def block_abs_argmax(self, block: Block) -> list[int]:
        flat_idx = self._api.argmax(self._api.abs(block))
        # OPTIMIZE numpy has np.unravel_indices. no analogue here?
        idcs = []
        for dim in reversed(block.shape):
            flat_idx, idx = divmod(flat_idx, dim)
            idcs.append(idx)
        return idcs

    def _block_argsort(self, block: Block, axis: int) -> Block:
        return self._api.argsort(block, axis=axis)

    def block_enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = self._api.zeros(shape, dtype=block.dtype)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[idcs] = block
        return res

    def block_stable_log(self, block: Block, cutoff: float) -> Block:
        return self._api.where(block > cutoff, self._api.log(block), 0.)


class NoSymmetryArrayApiBackend(NoSymmetryBackend):
    def __init__(self, api_namespace):
        block_backend = ArrayApiBlockBackend(api_namespace)
        NoSymmetryBackend.__init__(self, block_backend=block_backend)


class AbelianArrayApiBackend(AbelianBackend):
    def __init__(self, api_namespace):
        block_backend = ArrayApiBlockBackend(api_namespace)
        AbelianBackend.__init__(self, block_backend=block_backend)


class FusionTreeArrayApiBackend(FusionTreeBackend):
    def __init__(self, api_namespace):
        block_backend = ArrayApiBlockBackend(api_namespace)
        FusionTreeBackend.__init__(self, block_backend=block_backend)
