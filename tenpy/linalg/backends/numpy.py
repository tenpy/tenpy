from __future__ import annotations

from math import prod

import numpy as np
import scipy

from tenpy.linalg.backends.abelian import AbstractAbelianBackend
from tenpy.linalg.backends.abstract_backend import AbstractBlockBackend, Dtype, Precision, \
    BackendDtype, Block
from tenpy.linalg.backends.no_symmtery import AbstractNoSymmetryBackend
from tenpy.linalg.backends.nonabelian import AbstractNonabelianBackend
from tenpy.linalg.symmetries import AbstractSymmetry, VectorSpace

_dtype_map = {
    Dtype(Precision.half, True): np.float16,
    Dtype(Precision.single, True): np.float32,
    Dtype(Precision.double, True): np.float64,
    Dtype(Precision.long_double, True): np.longdouble,
    Dtype(Precision.half, False): np.complex32,
    Dtype(Precision.single, False): np.complex64,
    Dtype(Precision.double, False): np.complex128,
    Dtype(Precision.long_double, False): np.clongdouble,
}

_inverse_dtype_map = {
    np.float16: Dtype(Precision.half, True),
    np.float32: Dtype(Precision.single, True),
    np.float64: Dtype(Precision.double, True),
    np.longdouble: Dtype(Precision.long_double, True),
    np.complex32: Dtype(Precision.half, False),
    np.complex64: Dtype(Precision.single, False),
    np.complex128: Dtype(Precision.double, False),
    np.clongdouble: Dtype(Precision.long_double, False),
}


class NumpyBlockBackend(AbstractBlockBackend):
    svd_algorithms = ['gesdd', 'gesvd', 'robust', 'robust_silent']

    def __init__(self):
        AbstractBlockBackend.__init__(self, default_precision=Precision.single)

    def parse_dtype(self, dtype: Dtype) -> BackendDtype:
        try:
            return _dtype_map[dtype]
        except KeyError:
            raise ValueError(f'dtype {dtype} not supported for {self}.') from None

    def parse_block(self, obj, dtype: BackendDtype = None) -> Block:
        return np.asarray(obj, dtype)

    def block_is_real(self, a: Block):
        return not np.iscomplexobj(a)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return np.tensordot(a, b, (idcs_a, idcs_b))

    def block_shape(self, a: Block) -> tuple[int]:
        return np.shape(a)

    def block_item(self, a: Block):
        return a.item()

    def block_dtype(self, a: Block) -> Dtype:
        return _inverse_dtype_map[np.dtype(a)]

    def block_to_dtype(self, a: Block, dtype: BackendDtype) -> Block:
        return np.asarray(a, dtype=dtype)

    def block_copy(self, a: Block) -> Block:
        return np.copy(a)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        # TODO i like julia style much better actually, especially for many legs
        with np.set_printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

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

    def block_inner(self, a: Block, b: Block) -> complex:
        dim = max(a.ndim, b.ndim)
        return np.tensordot(np.conj(a), b, (list(range(dim)),) * 2).item()

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


class NoSymmetryNumpyBackend(NumpyBlockBackend, AbstractNoSymmetryBackend):
    def __init__(self):
        NumpyBlockBackend.__init__(self)
        AbstractNoSymmetryBackend.__init__(self)

    def svd(self, a, idcs1: list[int], idcs2: list[int], max_singular_values: int,
            threshold: float, max_err: float, algorithm: str = None):
        a = np.transpose(a, idcs1 + idcs2)
        a_shape1 = np.shape(a)[:len(idcs1)]
        a_shape2 = np.shape(a)[len(idcs1):]
        a = np.reshape(a, (prod(a_shape1), prod(a_shape2)))
        u, s, vh = self.matrix_svd(a, algorithm=algorithm)

        # determine number of singular values that fulfills truncation constraints
        chi = min(max_singular_values, len(s))
        if threshold > 0:
            chi = min(chi, np.sum(s > threshold))
        full_norm = np.linalg.norm(s)
        if max_err > 0:
            # cum_rel_err[n] is the relative error we would get if we kept only s[:n-1]
            cum_rel_err = np.cumsum(s[::-1] ** 2)[::-1] / (full_norm ** 2)
            n = np.flatnonzero(cum_rel_err <= max_err)[0]
            # now, cum_rel_err[n] is the first entry below max_err
            chi = min(chi, n - 1)

        if chi < len(s):
            trunc_err = np.linalg.norm(s[chi:]) / full_norm
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi]
        else:
            trunc_err = 0

        u = np.reshape(u, (*a_shape1, chi))
        vh = np.reshape(vh, (chi, *a_shape2))
        new_space = VectorSpace.non_symmetric(chi, is_dual=False, is_real=False)
        return u, s, vh, trunc_err, new_space


class AbelianNumpyBackend(NumpyBlockBackend, AbstractAbelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        NumpyBlockBackend.__init__(self)
        AbstractAbelianBackend.__init__(self, symmetry=symmetry)


class NonabelianNumpyBackend(NumpyBlockBackend, AbstractNonabelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        NumpyBlockBackend.__init__(self)
        AbstractNonabelianBackend.__init__(self, symmetry=symmetry)


def _svd_gesvd(a):
    raise NotImplementedError  # TODO
