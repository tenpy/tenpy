# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from numpy import prod

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, DiagonalData, Block, Dtype
from ..symmetries.groups import no_symmetry, Symmetry
from ..symmetries.spaces import VectorSpace, ProductSpace

__all__ = ['AbstractNoSymmetryBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor, DiagonalTensor, Mask


# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class AbstractNoSymmetryBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """
    Backend with no symmetries.

    Notes
    -----
    The data stored for the various tensor classes defined in ``tenpy.linalg.tensors`` is::

        - ``Tensor``:
            A single Block with as many axes as there a legs on the tensor.

        - ``DiagonalTensor`` :
            A single 1D Block. The diagonal of the corresponding 2D block of a ``Tensor``.

        - ``Mask``:
            These bool values indicate which indices of the large leg are kept for the small leg.

    """
    DataCls = "Block of AbstractBlockBackend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DataCls = self.BlockCls

    def test_data_sanity(self, a: Tensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        if is_diagonal:
            assert self.block_shape(a.data) == (a.legs[0].dim,), f'{self.block_shape(a)} != {(a.legs[0].dim,)}'
        else:
            assert self.block_shape(a.data) == tuple(a.shape), f'{self.block_shape(a)} != {tuple(a.shape.dims)}'

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_dtype(a)

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        return self.block_to_dtype(a.data, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        return self.block_item(a)

    def to_dense_block(self, a: Tensor) -> Block:
        return a.data

    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        return a.data

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Data:
        assert all(leg.symmetry == no_symmetry for leg in legs)
        return a  # TODO could this cause mutability issues?

    def diagonal_from_block(self, a: Block, leg: VectorSpace) -> DiagonalData:
        return a

    def from_block_func(self, func, legs: list[VectorSpace], func_kwargs={}):
        return func(tuple(l.dim for l in legs), **func_kwargs)

    def diagonal_from_block_func(self, func, leg: VectorSpace, func_kwargs={}) -> DiagonalData:
        return func((leg.dim,), **func_kwargs)

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype):
        return self.zero_block(shape=[l.dim for l in legs], dtype=dtype)

    def zero_diagonal_data(self, leg: VectorSpace, dtype: Dtype) -> DiagonalData:
        return self.zero_block(shape=[leg.dim], dtype=dtype)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        return self.eye_block(legs=[l.dim for l in legs], dtype=dtype)

    def copy_data(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_copy(a.data)

    def _data_repr_lines(self, a: Tensor, indent: str, max_width: int, max_lines: int):
        return [f'{indent}* Data:'] + self._block_repr_lines(a.data, indent=indent + '  ', max_width=max_width,
                                                            max_lines=max_lines - 1)

    def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]) -> Data:
        return self.block_tdot(a.data, b.data, axs_a, axs_b)

    def svd(self, a: Tensor, new_vh_leg_dual: bool, algorithm: str | None) -> tuple[Data, DiagonalData, Data, VectorSpace]:
        u, s, vh = self.matrix_svd(a.data, algorithm=algorithm)
        new_leg = VectorSpace.non_symmetric(len(s), is_real=a.legs[0].is_real, _is_dual=new_vh_leg_dual)
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

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return self.block_sum_all(a.data)

    def conj(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_conj(a.data)

    def combine_legs(self, a: Tensor, combine_slices: list[int, int], product_spaces: list[ProductSpace], new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        return self.block_combine_legs(a.data, combine_slices)

    def split_legs(self, a: Tensor, leg_idcs: list[int], final_legs: list[VectorSpace]) -> Data:
        return self.block_split_legs(a.data, leg_idcs, [[s.dim for s in a.legs[i].spaces] for i in leg_idcs])

    def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        return self.block_squeeze_legs(a.data, idcs)

    def norm(self, a: Tensor | DiagonalTensor, order: int | float = None) -> float:
        return self.block_norm(a.data, order=order)

    def act_block_diagonal_square_matrix(self, a: Tensor, block_method: str) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`."""
        block_method = getattr(self, block_method)
        return block_method(a.data)

    def add(self, a: Tensor, b: Tensor) -> Data:
        return self.block_add(a.data, b.data)

    def mul(self, a: float | complex, b: Tensor) -> Data:
        return self.block_mul(a, b.data)

    def infer_leg(self, block: Block, legs: list[VectorSpace | None], is_dual: bool = False,
                  is_real: bool = False) -> VectorSpace:
        idx, *more = [n for n, leg in enumerate(legs) if leg is None]
        if more:
            raise ValueError('Can only infer one leg')
        dim = self.block_shape(block)[idx]
        return VectorSpace.non_symmetric(dim, _is_dual=is_dual, is_real=is_real)

    def get_element(self, a: Tensor, idcs: list[int]) -> complex | float | bool:
        return self.get_block_element(a.data, idcs)

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        # a.data is a single 1D block
        return self.get_block_element(a.data, [idx])

    def set_element(self, a: Tensor, idcs: list[int], value: complex | float) -> Data:
        return self.set_block_element(a.data, idcs, value)

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        return self.set_block_element(a.data, [idx], value)
    
    def diagonal_data_from_full_tensor(self, a: Tensor, check_offdiagonal: bool) -> DiagonalData:
        return self.block_get_diagonal(a.data, check_offdiagonal=check_offdiagonal)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        return self.block_from_diagonal(a.data)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        return self.block_from_block_mask(a.data, dtype=dtype)

    def scale_axis(self, a: Tensor, b: DiagonalTensor, leg: int) -> Data:
        return self.block_scale_axis(a.data, b.data, leg)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        return func(a.data, **func_kwargs)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        return func(a.data, b.data, **func_kwargs)

    def fuse_states(self, state1: Block | None, state2: Block | None, space1: VectorSpace,
                    space2: VectorSpace, product_space: ProductSpace = None) -> Block | None:
        if state1 is None:
            return state2
        if state2 is None:
            return state1
        return self.block_kron(state1, state2)

    def mask_infer_small_leg(self, mask_data: Data, large_leg: VectorSpace) -> VectorSpace:
        return VectorSpace.non_symmetric(dim=self.block_sum_all(mask_data),
                                         is_real=large_leg.is_real,
                                         _is_dual=large_leg.is_dual)

    def apply_mask_to_Tensor(self, tensor: Tensor, mask: Mask, leg_idx: int) -> Data:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=leg_idx)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=0)
