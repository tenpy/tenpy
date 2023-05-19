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

    """
    VectorSpaceCls = VectorSpace
    ProductSpaceCls = ProductSpace
    DataCls = "Block of AbstractBlockBackend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DataCls = self.BlockCls

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_dtype(a)

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        return self.block_to_dtype(a.data, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

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

    def svd(self, a: Tensor, new_vh_leg_dual: bool) -> tuple[Data, Data, Data, VectorSpace]:
        u, s, vh = self.matrix_svd(a)
        new_leg = VectorSpace.non_symmetric(len(s), is_dual=new_vh_leg_dual, is_real=a.legs[0].is_real)
        return u, s, vh, new_leg

    def qr(self, a: Tensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, VectorSpace]:
        q, r = self.matrix_qr(a, full=full)
        new_leg_dim = self.block_shape(r)[0]
        new_leg = VectorSpace.non_symmetric(new_leg_dim, is_dual=new_r_leg_dual, is_real=a.legs[0].is_real)
        return q, r, new_leg

    def outer(self, a: Tensor, b: Tensor) -> Data:
        return self.block_outer(a.data, b.data)

    def inner(self, a: Tensor, b: Tensor, do_conj: bool, axs2: list[int] | None) -> complex:
        return self.block_inner(a.data, b.data, do_conj=do_conj, axs2=axs2)

    def permute_legs(self, a: Tensor, permutation: list[int]) -> Data:
        return self.block_permute_axes(a.data, permutation)

    def trace_full(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        return self.block_trace_full(a.data, idcs1, idcs2)

    def trace_partial(self, a: Tensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        return self.block_trace_partial(a.data, idcs1, idcs2, remaining_idcs)

    def conj(self, a: Tensor) -> Data:
        return self.block_conj(a.data)

    def combine_legs(self, a: Tensor, combine_slices: list[int, int], product_spaces: list[ProductSpace], new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        return self.block_combine_legs(a.data, combine_slices)

    def split_legs(self, a: Tensor, leg_idcs: list[int], final_legs: list[VectorSpace]) -> Data:
        return self.block_split_legs(a.data, leg_idcs, [[s.dim for s in a.legs[i].spaces] for i in leg_idcs])

    def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        return self.block_squeeze_legs(a.data, idcs)

    def norm(self, a: Tensor) -> float:
        return self.block_norm(a.data)

    def act_block_diagonal_square_matrix(self, a: Tensor, block_method: str) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`."""
        block_method = getattr(self, block_method)
        return block_method(a.data)

    def add(self, a: Tensor, b: Tensor) -> Data:
        return self.block_add(a.data, b.data)

    def mul(self, a: float | complex, b: Tensor) -> Data:
        return self.block_mul(a, b.data)
