from __future__ import annotations
from abc import ABC, abstractmethod
from math import prod

from tenpy.linalg.backends.abstract_backend import AbstractBackend, AbstractBlockBackend, BackendArray, BackendDtype, \
    Block, Dtype
from tenpy.linalg.symmetries import VectorSpace, no_symmetry, AbstractSymmetry, AbstractSpace
from tenpy.linalg.tensors import Leg


# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class AbstractNoSymmetryBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """
    Backend with no symmetries.
    BackendArrays are just the Blocks of the AbstractBlockBackend.
    Thus most functions just delegate to the respective functions on blocks.
    """

    def __init__(self):
        AbstractBackend.__init__(self, no_symmetry)

    def parse_data(self, obj, dtype: BackendDtype = None) -> BackendArray:
        return self.parse_block(obj, dtype)

    def is_real(self, data: BackendArray) -> bool:
        return self.block_is_real(data)

    def infer_legs(self, data: BackendArray, labels: list[str] | None) -> list[Leg]:
        is_real = self.is_real(data)
        shape = self.block_shape(data)
        if labels is None:
            labels = [None] * len(shape)
        else:
            assert len(labels) == len(shape)
        return [Leg.trivial(dim=d, ax=n, label=l, is_real=is_real) 
                for n, (d, l) in enumerate(zip(shape, labels))]

    def tdot(self, a: BackendArray, b: BackendArray, a_axes: list[int], b_axes: list[int]
             ) -> BackendArray:
        return self.block_tdot(a, b, a_axes, b_axes)

    def legs_are_compatible(self, data: BackendArray, legs: list[Leg]) -> bool:
        shape = self.block_shape(data)
        for n, leg in enumerate(legs):
            if not (leg.space.symmetry == no_symmetry and leg.space.dim == shape[n]):
                return False
        return True

    def item(self, data: BackendArray) -> float | complex:
        return self.block_item(data)

    def to_dense_block(self, data: BackendArray) -> Block:
        return data

    def reduce_symmetry(self, data: BackendArray, new_symm: AbstractSymmetry) -> BackendArray:
        if new_symm == no_symmetry:
            return data
        else:
            raise ValueError(f'Can not decrease {no_symmetry} to {new_symm}.')

    def increase_symmetry(self, data: BackendArray, new_symm: AbstractSymmetry, atol=1e-8, rtol=1e-5
                          ) -> BackendArray:
        if new_symm == no_symmetry:
            raise UserWarning  # TODO logging
            return data
        else:
            raise ValueError(f'{self} can not represent {new_symm}.')

    def infer_dtype(self, data: BackendArray) -> Dtype:
        return self.block_dtype(data)

    def to_dtype(self, data: BackendArray, dtype: Dtype) -> BackendArray:
        return self.block_to_dtype(data, self.parse_dtype(dtype))

    def copy_data(self, data: BackendArray) -> BackendArray:
        return self.block_copy(data)

    def _data_repr_lines(self, data: BackendArray, indent: str, max_width: int, max_lines: int):
        return [f'{indent}* Data:'] + self._block_repr_lines(data, indent=indent + '  ', max_width=max_width,
                                                            max_lines=max_lines - 1)

    @abstractmethod
    def svd(self, a: BackendArray, idcs1: list[int], idcs2: list[int], max_singular_values: int,
            threshold: float, max_err: float, algorithm: str):
        # reshaping, slicing etc is so specific to the BlockBackend that I dont bother unifying anything here.
        # that might change though...
        ...

    def outer(self, a: BackendArray, b: BackendArray) -> BackendArray:
        return self.block_outer(a, b)

    def inner(self, a: BackendArray, b: BackendArray, axs2: list[int] | None) -> complex:
        return self.block_inner(a, b, axs2)

    def transpose(self, a: BackendArray, permutation: list[int]) -> BackendArray:
        return self.block_transpose(a, permutation)

    def trace(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        return self.block_trace(a, idcs1, idcs2)

    def conj(self, a: BackendArray) -> BackendArray:
        return self.block_conj(a)

    def combine_legs(self, a: BackendArray, legs: list[int]) -> BackendArray:
        # TODO we could lazy-evaluate this...
        return self.block_combine_legs(a, legs)

    def split_leg(self, a: BackendArray, leg: int, orig_spaces: list[AbstractSpace]) -> BackendArray:
        # TODO we could lazy-evaluate this...
        return self.block_split_leg(a, leg, dims=[s.dim for s in orig_spaces])

    def allclose(self, a: BackendArray, b: BackendArray, rtol: float, atol: float) -> bool:
        return self.block_allclose(a, b, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: BackendArray, idcs: list[int]) -> BackendArray:
        return self.block_squeeze_legs(a, idcs)

    def norm(self, a: BackendArray) -> float:
        return self.block_norm(a)

    def exp(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        matrix, aux = self.block_matrixify(a, idcs1, idcs2)
        return self.block_dematrixify(self.matrix_exp(matrix), aux)

    def log(self, a: BackendArray, idcs1: list[int], idcs2: list[int]) -> BackendArray:
        matrix, aux = self.block_matrixify(a, idcs1, idcs2)
        return self.block_dematrixify(self.matrix_log(matrix), aux)

    def random_uniform(self, legs: list[AbstractSpace], dtype: Dtype) -> BackendArray:
        return self.block_random_uniform([l.dim for l in legs], dtype=dtype)

    def random_gaussian(self, legs: list[AbstractSpace], dtype: Dtype, sigma: float) -> BackendArray:
        return self.block_random_gaussian([l.dim for l in legs], dtype=dtype, sigma=sigma)
