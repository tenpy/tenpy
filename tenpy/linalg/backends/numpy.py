# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations

import numpy as np
import scipy

from .abelian import AbelianBackend
from .abstract_backend import BlockBackend, Block
from .no_symmetry import NoSymmetryBackend
from .fusion_tree_backend import FusionTreeBackend
from ..dtypes import Dtype, _numpy_dtype_to_tenpy, _tenpy_dtype_to_numpy

__all__ = ['NumpyBlockBackend', 'NoSymmetryNumpyBackend', 'AbelianNumpyBackend',
           'FusionTreeNumpyBackend']


class NumpyBlockBackend(BlockBackend):
    BlockCls = np.ndarray
    svd_algorithms = ['gesdd', 'gesvd', 'robust', 'robust_silent']

    tenpy_dtype_map = _numpy_dtype_to_tenpy
    backend_dtype_map = _tenpy_dtype_to_numpy
    
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False) -> Block:
        block = np.asarray(a, dtype=self.backend_dtype_map[dtype])
        if np.issubdtype(block.dtype, np.integer):
            block = block.astype(np.float64, copy=False)
        if return_dtype:
            return block, self.tenpy_dtype_map[block.dtype]
        return block

    def block_add_axis(self, a: Block, pos: int) -> Block:
        return np.expand_dims(a, pos)

    def block_abs_argmax(self, block: Block) -> list[int]:
        return np.unravel_index(np.argmax(np.abs(block)), block.shape)

    def block_all(self, a) -> bool:
        return np.all(a)
        
    def block_allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return np.allclose(a, b, rtol=rtol, atol=atol)

    def block_angle(self, a: Block) -> Block:
        return np.angle(a)

    def block_any(self, a) -> bool:
        return np.any(a)

    def block_apply_mask(self, block: Block, mask: Block, ax: int) -> Block:
        return np.compress(mask, block, ax)

    def _block_argsort(self, block: Block, axis: int) -> Block:
        return np.argsort(block, axis=axis)

    def block_conj(self, a: Block) -> Block:
        return np.conj(a)

    def block_copy(self, a: Block) -> Block:
        return np.copy(a)

    def block_dtype(self, a: Block) -> Dtype:
        return self.tenpy_dtype_map[a.dtype]

    def block_eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = np.linalg.eigh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = np.take(w, perm)
            v = np.take(v, perm, axis=1)
        return w, v

    def block_eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = np.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = np.take(w, perm)
        return w

    def block_enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        # OPTIMIZE is there a numpy builtin function that does this? or at least part of this?
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = np.zeros(shape, dtype=block.dtype)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[tuple(idcs)] = block  # TODO should we worry about mutability?
        return res

    def block_exp(self, a: Block) -> Block:
        return np.exp(a)

    def block_from_diagonal(self, diag: Block) -> Block:
        return np.diag(diag)

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        M, = mask.shape
        N = np.sum(mask)
        res = np.zeros((N, M), dtype=self.backend_dtype_map[dtype])
        res[np.arange(N), mask] = 1
        return res

    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None) -> Block:
        if dtype is None:
            return a
        return np.asarray(a, self.backend_dtype_map[dtype])
    
    def block_get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        res = np.diagonal(a)
        if check_offdiagonal:
            if not np.allclose(a, np.diag(res)):
                raise ValueError('Not a diagonal block.')
        return res

    def block_imag(self, a: Block) -> Block:
        return np.imag(a)

    def block_inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        # TODO use np.sum(a * b) instead?
        if do_dagger:
            return np.tensordot(np.conj(a), b, a.ndim).item()
        return np.tensordot(a, b, [list(range(a.ndim)), list(reversed(range(a.ndim)))]).item()
        
    def block_item(self, a: Block) -> float | complex:
        return a.item()

    def block_kron(self, a: Block, b: Block) -> Block:
        return np.kron(a, b)

    def block_log(self, a: Block) -> Block:
        return np.log(a)

    def block_max(self, a: Block) -> float | complex:
        return np.max(a)

    def block_max_abs(self, a: Block) -> float:
        return np.max(np.abs(a))

    def block_min(self, a: Block) -> float | complex:
        return np.min(a)
    
    def block_norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        if axis is None:
            return np.linalg.norm(a.ravel(), ord=order)
        return np.linalg.norm(a, ord=order, axis=axis)

    def block_outer(self, a: Block, b: Block) -> Block:
        return np.tensordot(a, b, ((), ()))

    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return np.transpose(a, permutation)

    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        res = np.random.normal(loc=0, scale=sigma, size=dims)
        if not dtype.is_real:
            res = res + 1.j * np.random.normal(loc=0, scale=sigma, size=dims)
        return res

    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        res = np.random.uniform(-1, 1, size=dims)
        if not dtype.is_real:
            res = res + 1.j * np.random.uniform(-1, 1, size=dims)
        return res

    def block_real(self, a: Block) -> Block:
        return np.real(a)

    def block_real_if_close(self, a: Block, tol: float) -> Block:
        return np.real_if_close(a, tol=tol)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        # TODO i like julia style much better actually, especially for many legs
        with np.printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in str(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        return np.reshape(a, shape)

    def block_shape(self, a: Block) -> tuple[int]:
        return np.shape(a)

    def block_sqrt(self, a: Block) -> Block:
        return np.sqrt(a)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        return np.squeeze(a, tuple(idcs))

    def block_stable_log(self, block: Block, cutoff: float) -> Block:
        return np.where(block > cutoff, np.log(block), 0.)
    
    def block_sum(self, a: Block, ax: int) -> Block:
        return np.sum(a, axis=ax)

    def block_sum_all(self, a: Block) -> float | complex:
        return np.sum(a)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return np.tensordot(a, b, (idcs_a, idcs_b))

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return np.asarray(a, dtype=self.backend_dtype_map[dtype])

    def block_trace_full(self, a: Block) -> float | complex:
        num_trace = a.ndim // 2
        trace_dim = np.prod(a.shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = np.reshape(np.transpose(a, perm), (trace_dim, trace_dim))
        return np.trace(a, axis1=0, axis2=1).item()

    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = np.transpose(a, remaining + idcs1 + idcs2)
        trace_dim = np.prod(a.shape[len(remaining):len(remaining)+len(idcs1)], dtype=int)
        a = np.reshape(a, a.shape[:len(remaining)] + (trace_dim, trace_dim))
        return np.trace(a, axis1=-2, axis2=-1)

    def eye_matrix(self, dim: int, dtype: Dtype) -> Block:
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
                    raise NotImplementedError  # TODO log warning
            return _svd_gesvd(a)

        elif algorithm == 'gesvd':
            return _svd_gesvd(a)

        else:
            raise ValueError(f'SVD algorithm not supported: {algorithm}')

    def ones_block(self, shape: list[int], dtype: Dtype) -> Block:
        return np.ones(shape, dtype=self.backend_dtype_map[dtype])

    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        return np.zeros(shape, dtype=self.backend_dtype_map[dtype])


class NoSymmetryNumpyBackend(NoSymmetryBackend):
    def __init__(self):
        NoSymmetryBackend.__init__(self, block_backend=NumpyBlockBackend())


class AbelianNumpyBackend(AbelianBackend):
    def __init__(self):
        AbelianBackend.__init__(self, block_backend=NumpyBlockBackend())


class FusionTreeNumpyBackend(FusionTreeBackend):
    def __init__(self):
        FusionTreeBackend.__init__(self, block_backend=NumpyBlockBackend())


def _svd_gesvd(a):
    raise NotImplementedError  # TODO
