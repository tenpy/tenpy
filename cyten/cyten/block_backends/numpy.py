"""A block backend using numpy."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import numpy as np
import scipy

from ._block_backend import Block, BlockBackend
from .dtypes import Dtype, _cyten_dtype_to_numpy, _numpy_dtype_to_cyten


class NumpyBlockBackend(BlockBackend):
    """A block backend using numpy."""

    BlockCls = np.ndarray
    svd_algorithms = ['gesdd', 'gesvd', 'robust', 'robust_silent']

    cyten_dtype_map = _numpy_dtype_to_cyten
    backend_dtype_map = _cyten_dtype_to_numpy

    def __init__(self):
        super().__init__(default_device='cpu')

    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        block = np.asarray(a, dtype=self.backend_dtype_map[dtype])
        if np.issubdtype(block.dtype, np.integer):
            block = block.astype(np.float64, copy=False)
        if return_dtype:
            return block, self.cyten_dtype_map[block.dtype]
        return block

    def as_device(self, device: str | None) -> str:
        if device is None:
            return self.default_device
        if device != self.default_device:
            msg = f'{self.__class__.__name__} does not support device {device}.'
            raise ValueError(msg)
        return device

    def add_axis(self, a: Block, pos: int) -> Block:
        return np.expand_dims(a, pos)

    def abs_argmax(self, block: Block) -> list[int]:
        return np.unravel_index(np.argmax(np.abs(block)), block.shape)

    def block_all(self, a) -> bool:
        return np.all(a)

    def allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return np.allclose(a, b, rtol=rtol, atol=atol)

    def angle(self, a: Block) -> Block:
        return np.angle(a)

    def block_any(self, a) -> bool:
        return np.any(a)

    def apply_mask(self, block: Block, mask: Block, ax: int) -> Block:
        return np.compress(mask, block, ax)

    def _argsort(self, block: Block, axis: int) -> Block:
        return np.argsort(block, axis=axis)

    def conj(self, a: Block) -> Block:
        return np.conj(a)

    def copy_block(self, a: Block, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        return np.copy(a)

    def cutoff_inverse(self, a: Block, cutoff: float) -> Block:
        """The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``."""
        return 1 / np.where(np.abs(a) < cutoff, np.inf, a)

    def get_dtype(self, a: Block) -> Dtype:
        return self.cyten_dtype_map[a.dtype]

    def eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = np.linalg.eigh(block)
        if sort is not None:
            perm = self.argsort(w, sort)
            w = np.take(w, perm)
            v = np.take(v, perm, axis=1)
        return w, v

    def eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = np.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.argsort(w, sort)
            w = np.take(w, perm)
        return w

    def enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        # OPTIMIZE is there a numpy builtin function that does this? or at least part of this?
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = np.zeros(shape, dtype=block.dtype)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[tuple(idcs)] = block.copy()  # OPTIMIZE is the copy needed
        return res

    def exp(self, a: Block) -> Block:
        return np.exp(a)

    def block_from_diagonal(self, diag: Block) -> Block:
        return np.diag(diag)

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        (M,) = mask.shape
        N = np.sum(mask)
        res = np.zeros((N, M), dtype=self.backend_dtype_map[dtype])
        res[np.arange(N), mask] = 1
        return res

    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        if dtype is None:
            return a
        return np.asarray(a, self.backend_dtype_map[dtype])

    def get_device(self, a: Block) -> str:
        return self.default_device

    def get_diagonal(self, a: Block, tol: float | None) -> Block:
        res = np.diagonal(a)
        if tol is not None:
            if not np.allclose(a, np.diag(res), atol=tol):
                raise ValueError('Not a diagonal block.')
        return res

    def imag(self, a: Block) -> Block:
        return np.imag(a)

    def inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        # OPTIMIZE use np.sum(a * b) instead?
        if do_dagger:
            return np.tensordot(np.conj(a), b, a.ndim).item()
        return np.tensordot(a, b, [list(range(a.ndim)), list(reversed(range(a.ndim)))]).item()

    def item(self, a: Block) -> float | complex:
        return a.item()

    def kron(self, a: Block, b: Block) -> Block:
        return np.kron(a, b)

    def log(self, a: Block) -> Block:
        return np.log(a)

    def max(self, a: Block) -> float | complex:
        return np.max(a).item()

    def max_abs(self, a: Block) -> float:
        return np.max(np.abs(a)).item()

    def min(self, a: Block) -> float | complex:
        return np.min(a).item()

    def norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        if axis is None:
            return np.linalg.norm(a.ravel(), ord=order).item()
        return np.linalg.norm(a, ord=order, axis=axis)

    def outer(self, a: Block, b: Block) -> Block:
        return np.tensordot(a, b, ((), ()))

    def permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return np.transpose(a, permutation)

    def random_normal(self, dims: list[int], dtype: Dtype, sigma: float, device: str = None) -> Block:
        # if sigma is standard deviation for complex numbers, need to divide by sqrt(2)
        # to get standard deviation in real and imag parts
        if not dtype.is_real:
            sigma /= np.sqrt(2)
        _ = self.as_device(device)  # for input check only
        res = np.random.normal(loc=0, scale=sigma, size=dims)
        if not dtype.is_real:
            res = res + 1.0j * np.random.normal(loc=0, scale=sigma, size=dims)
        return res

    def random_uniform(self, dims: list[int], dtype: Dtype, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        res = np.random.uniform(-1, 1, size=dims)
        if not dtype.is_real:
            res = res + 1.0j * np.random.uniform(-1, 1, size=dims)
        return res

    def real(self, a: Block) -> Block:
        return np.real(a)

    def real_if_close(self, a: Block, tol: float) -> Block:
        return np.real_if_close(a, tol=tol)

    def tile(self, a: Block, repeats: int, axis: int | None = None) -> Block:
        return np.tile(a, repeats)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        with np.printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in str(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def reshape(self, a: Block, shape: tuple[int]) -> Block:
        return np.reshape(a, shape)

    def get_shape(self, a: Block) -> tuple[int]:
        return np.shape(a)

    def sqrt(self, a: Block) -> Block:
        return np.sqrt(a)

    def squeeze_axes(self, a: Block, idcs: list[int]) -> Block:
        return np.squeeze(a, tuple(idcs))

    def stable_log(self, block: Block, cutoff: float) -> Block:
        return np.where(block > cutoff, np.log(block), 0.0)

    def sum(self, a: Block, ax: int) -> Block:
        return np.sum(a, axis=ax)

    def sum_all(self, a: Block) -> float | complex:
        return np.sum(a).item()

    def tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return np.tensordot(a, b, (idcs_a, idcs_b))

    def to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return np.asarray(a, dtype=self.backend_dtype_map[dtype])

    def trace_full(self, a: Block) -> float | complex:
        num_trace = a.ndim // 2
        trace_dim = np.prod(a.shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = np.reshape(np.transpose(a, perm), (trace_dim, trace_dim))
        return np.trace(a, axis1=0, axis2=1).item()

    def trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = np.transpose(a, remaining + idcs1 + idcs2)
        trace_dim = np.prod(a.shape[len(remaining) : len(remaining) + len(idcs1)], dtype=int)
        a = np.reshape(a, a.shape[: len(remaining)] + (trace_dim, trace_dim))
        return np.trace(a, axis1=-2, axis2=-1)

    def eye_matrix(self, dim: int, dtype: Dtype, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        return np.eye(dim, dtype=self.backend_dtype_map[dtype])

    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        return a[tuple(idcs)].item()

    def matrix_dot(self, a: Block, b: Block) -> Block:
        return np.dot(a, b)

    def matrix_exp(self, matrix: Block) -> Block:
        return scipy.linalg.expm(matrix)

    def matrix_log(self, matrix: Block) -> Block:
        return scipy.linalg.logm(matrix)

    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        return scipy.linalg.qr(a, mode='full' if full else 'economic')

    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        if algorithm is None:
            algorithm = 'gesdd'

        if algorithm == 'gesdd':
            return scipy.linalg.svd(a, full_matrices=False)

        elif algorithm in ['robust', 'robust_silent']:
            silent = algorithm == 'robust_silent'
            try:
                return scipy.linalg.svd(a, full_matrices=False)
            except np.linalg.LinAlgError:
                if not silent:
                    raise NotImplementedError  # log warning
            return _svd_gesvd(a)

        elif algorithm == 'gesvd':
            return _svd_gesvd(a)

        else:
            raise ValueError(f'SVD algorithm not supported: {algorithm}')

    def ones_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        return np.ones(shape, dtype=self.backend_dtype_map[dtype])

    def zeros(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        _ = self.as_device(device)  # for input check only
        return np.zeros(shape, dtype=self.backend_dtype_map[dtype])


def _svd_gesvd(a):
    raise NotImplementedError
