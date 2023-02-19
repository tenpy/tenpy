# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations

from math import prod
from typing import Any

import numpy as np
import scipy

from .abelian import AbstractAbelianBackend
from .abstract_backend import AbstractBlockBackend, Block, Data
from .no_symmtery import AbstractNoSymmetryBackend
from .nonabelian import AbstractNonabelianBackend
from ..misc import inverse_permutation
from ..symmetries import VectorSpace
from ..tensors import Tensor, Dtype


class NumpyBlockBackend(AbstractBlockBackend):
    svd_algorithms = ['gesdd', 'gesvd', 'robust', 'robust_silent']

    def block_is_real(self, a: Block):
        return not np.iscomplexobj(a)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return np.tensordot(a, b, (idcs_a, idcs_b))

    def block_shape(self, a: Block) -> tuple[int]:
        return np.shape(a)

    def block_item(self, a: Block) -> float | complex:
        return a.item()

    def block_dtype(self, a: Block) -> Dtype:
        return a.dtype

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return np.asarray(a, dtype=dtype)

    def block_copy(self, a: Block) -> Block:
        return np.copy(a)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        # TODO i like julia style much better actually, especially for many legs
        with np.printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    # noinspection PyTypeChecker
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

    def block_outer(self, a: Block, b: Block) -> Block:
        return np.tensordot(a, b, ((), ()))

    def block_inner(self, a: Block, b: Block, axs2: list[int] | None) -> complex:
        dim = max(a.ndim, b.ndim)
        axs2 = list(range(dim)) if axs2 is None else axs2
        return np.tensordot(np.conj(a), b, (list(range(dim)), axs2)).item()

    def block_transpose(self, a: Block, permutation: list[int]) -> Block:
        return np.transpose(a, permutation)

    def block_trace(self, a: Block, idcs1: list[int], idcs2: list[int]) -> Block:
        remaining = [n for n in range(len(a.shape)) if n not in idcs1 and n not in idcs2]
        a = np.transpose(a, remaining, idcs1, idcs2)
        trace_dim = prod(a.shape[len(remaining):len(remaining)+len(idcs1)])
        a = np.reshape(a, (a.shape[:len(remaining)], trace_dim, trace_dim))
        return np.trace(a, axis1=-2, axis2=-1)

    def block_conj(self, a: Block) -> Block:
        return np.conj(a)

    def block_combine_legs(self, a: Block, legs: list[int]) -> Block:
        # TODO optimize this?
        legs_before_new_leg = [n for n in range(legs[0]) if n not in legs]
        legs_after_new_leg = [n for n in range(legs[0] + 1, len(a.shape)) if n not in legs]
        permutation = legs_before_new_leg + legs + legs_after_new_leg
        new_shape = [a.shape[n] for n in permutation]
        a = np.transpose(a, permutation)
        return np.reshape(a, new_shape)

    def block_split_leg(self, a: Block, leg: int, dims: list[int]) -> Block:
        return np.reshape(a, a.shape[:leg] + dims + a.shape[leg + 1:])

    def block_allclose(self, a: Block, b: Block, rtol: float, atol: float) -> bool:
        return np.allclose(a, b, rtol=rtol, atol=atol)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        return np.squeeze(a, idcs)

    def block_norm(self, a: Block) -> float:
        return np.linalg.norm(a, ord=2)

    def block_matrixify(self, a: Block, idcs1: list[int], idcs2: list[int]) -> tuple[Block, Any]:
        permutation = idcs1 + idcs2
        a = np.transpose(a, permutation)
        a_shape = np.shape(a)
        matrix_shape = prod(a_shape[:len(idcs1)]), prod(a_shape[len(idcs1):])
        matrix = np.reshape(a, matrix_shape)
        aux = (permutation, a_shape)
        return matrix, aux

    def block_dematrixify(self, matrix: Block, aux: Any) -> Block:
        permutation, a_shape = aux
        res = np.reshape(matrix, a_shape)
        return np.transpose(res, inverse_permutation(permutation))

    def matrix_exp(self, matrix: Block) -> Block:
        return scipy.linalg.expm(matrix)

    def matrix_log(self, matrix: Block) -> Block:
        return scipy.linalg.logm(matrix)

    # TODO is this useful...?
    # def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
    #     if dtype.is_real:
    #         res = np.random.uniform(low=-1, high=1., size=dims)
    #     else:
    #         # z = r * e^{i pi}; PDF of r is 2r on [0, 1], CDF is r^2 ; inverse CDF sampling
    #         r = np.sqrt(np.random.uniform(low=0, high=1, size=dims))
    #         phi = np.random.uniform(low=0, high=2 * np.pi, size=dims)
    #         res = r * np.exp(1.j * phi)
    #     return np.asarray(res, dtype=_dtype_map[dtype])

    def block_random_gaussian(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        res = np.random.normal(loc=0, scale=sigma, size=dims)
        if not dtype.is_real:
            res += 1.j * np.random.normal(loc=0, scale=sigma, size=dims)
        return res

    def block_from_numpy(self, a) -> Block:
        return a

    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        return np.zeros(shape, dtype=dtype)

    def eye_block(self, legs: list[int], dtype: Dtype) -> Data:
        matrix_dim = prod(legs)
        eye = np.eye(matrix_dim, dtype=dtype)
        eye = np.reshape(eye, legs + legs)
        return eye
    

class NoSymmetryNumpyBackend(NumpyBlockBackend, AbstractNoSymmetryBackend):
    def svd(self, a: Tensor, axs1: list[int], axs2: list[int], new_leg: VectorSpace | None
            ) -> tuple[Data, Data, Data, VectorSpace]:
        a = np.transpose(a.data, axs1 + axs2)
        a_shape1 = np.shape(a)[:len(axs1)]
        a_shape2 = np.shape(a)[len(axs1):]
        a = np.reshape(a, (prod(a_shape1), prod(a_shape2)))
        u, s, vh = self.matrix_svd(a)
        u = np.reshape(u, (*a_shape1, len(s)))
        vh = np.reshape(vh, (len(s), *a_shape2))
        if new_leg is None:
            new_leg = VectorSpace.non_symmetric(len(s), is_dual=False, is_real=False)
        return u, s, vh, new_leg


class AbelianNumpyBackend(NumpyBlockBackend, AbstractAbelianBackend):
    pass


class NonabelianNumpyBackend(NumpyBlockBackend, AbstractNonabelianBackend):
    pass


def _svd_gesvd(a):
    raise NotImplementedError  # TODO
