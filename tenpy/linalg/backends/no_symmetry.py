# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from numpy import prod

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, Block, Dtype
from ..symmetries.groups import no_symmetry, Symmetry
from ..symmetries.spaces import VectorSpace, ProductSpace

__all__ = ['AbstractNoSymmetryBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor


# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class AbstractNoSymmetryBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """
    Backend with no symmetries.

    Attributes
    ----------
    data : Block
        A single block of the AbstractBlockBackend, e.g. a numpy.ndarray for NumpyBlockBackend.
    """

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_dtype(a)

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        return self.block_to_dtype(a.data, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def is_real(self, a: Tensor) -> bool:
        # TODO(JU): this should be checkable via the dtype and can be removed, no?
        return self.block_is_real(a.data)

    def data_item(self, a: Data) -> float | complex:
        return self.block_item(a)

    def to_dense_block(self, a: Tensor) -> Block:
        return a.data

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Data:
        assert all(leg.symmetry == no_symmetry for leg in legs)
        return a  # TODO could this cause mutability issues?

    def from_block_func(self, func, legs: list[VectorSpace]):
        return func(tuple(l.dim for l in legs))

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype):
        return self.zero_block(shape=[l.dim for l in legs], dtype=dtype)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        return self.eye_block(legs=[l.dim for l in legs], dtype=dtype)

    def copy_data(self, a: Tensor) -> Data:
        return self.block_copy(a.data)

    def _data_repr_lines(self, data: Data, indent: str, max_width: int, max_lines: int):
        return [f'{indent}* Data:'] + self._block_repr_lines(data, indent=indent + '  ', max_width=max_width,
                                                            max_lines=max_lines - 1)

    def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]) -> Data:
        return self.block_tdot(a.data, b.data, axs_a, axs_b)

    def svd(self, a: Tensor, axs1: list[int], axs2: list[int], new_leg: VectorSpace | None
            ) -> tuple[Data, Data, Data, VectorSpace]:
        a = self.block_transpose(a.data, axs1 + axs2)
        a_shape = self.block_shape(a)
        a_shape1 = a_shape[:len(axs1)]
        a_shape2 = a_shape[len(axs1):]
        a = self.block_reshape(a, (prod(a_shape1), prod(a_shape2)))
        u, s, vh = self.matrix_svd(a)
        u = self.block_reshape(u, a_shape1 + (len(s), ))
        vh = self.block_reshape(vh, (len(s), ) + a_shape2)
        if new_leg is None:
            new_leg = VectorSpace.non_symmetric(len(s), is_dual=False, is_real=a.legs[0].is_real)
        return u, s, vh, new_leg

    def outer(self, a: Tensor, b: Tensor) -> Data:
        return self.block_outer(a.data, b.data)

    def inner(self, a: Tensor, b: Tensor, axs2: list[int] | None) -> complex:
        return self.block_inner(a.data, b.data, axs2)

    def transpose(self, a: Tensor, permutation: list[int]) -> Data:
        return self.block_transpose(a.data, permutation)

    def trace(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        return self.block_trace(a.data, idcs1, idcs2)

    def conj(self, a: Tensor) -> Data:
        return self.block_conj(a.data)

    def combine_legs(self, a: Tensor, combine_slices: list[int, int], product_spaces: list[ProductSpace], new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        return self.block_combine_legs(a.data, combine_slices)

    def split_legs(self, a: Tensor, leg_idcs: list[int]) -> Data:
        return self.block_split_legs(a.data, leg_idcs, [[s.dim for s in a.legs[i].spaces] for i in leg_idcs])

    def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        return self.block_squeeze_legs(a.data, idcs)

    def norm(self, a: Tensor) -> float:
        return self.block_norm(a.data)

    def exp(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        matrix, aux = self.block_matrixify(a.data, idcs1, idcs2)
        return self.block_dematrixify(self.matrix_exp(matrix), aux)

    def log(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        matrix, aux = self.block_matrixify(a.data, idcs1, idcs2)
        return self.block_dematrixify(self.matrix_log(matrix), aux)

    def random_normal(self, legs: list[VectorSpace], dtype: Dtype, sigma: float) -> Data:
        return self.block_random_normal([l.dim for l in legs], dtype=dtype, sigma=sigma)

    def add(self, a: Tensor, b: Tensor) -> Data:
        return self.block_add(a.data, b.data)

    def mul(self, a: float | complex, b: Tensor) -> Data:
        return self.block_mul(a, b.data)

