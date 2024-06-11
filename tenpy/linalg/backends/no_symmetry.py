# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable
import numpy as np

from .abstract_backend import Backend, BlockBackend, Data, DiagonalData, Block
from ..dtypes import Dtype
from ..symmetries import no_symmetry, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace

__all__ = ['NoSymmetryBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import BlockDiagonalTensor, DiagonalTensor, Mask


# TODO eventually remove BlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class NoSymmetryBackend(Backend, BlockBackend, metaclass=ABCMeta):
    """Abstract base class for backends that do not enforce any symmetry.

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
    DataCls = "Block of BlockBackend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DataCls = self.BlockCls

    def test_data_sanity(self, a: BlockDiagonalTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        if is_diagonal:
            assert self.block_shape(a.data) == (a.legs[0].dim,), f'{self.block_shape(a)} != {(a.legs[0].dim,)}'
        else:
            assert self.block_shape(a.data) == tuple(a.shape), f'{self.block_shape(a)} != {tuple(a.shape.dims)}'

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        assert self.block_shape(a.data) == (a.legs[0].dim,)
        assert self.block_sum_all(a.data) == a.legs[1].dim

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_dtype(a)

    def to_dtype(self, a: BlockDiagonalTensor, dtype: Dtype) -> Data:
        return self.block_to_dtype(a.data, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        return self.block_item(a)

    def to_dense_block(self, a: BlockDiagonalTensor) -> Block:
        return self.apply_basis_perm(a.data, a.legs, inv=True)

    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        return self.apply_basis_perm(a.data, [a.legs[0]], inv=True)

    def from_dense_block(self, a: Block, legs: list[Space], num_domain_legs: int,
                         tol: float = 1e-8) -> Data:
        assert all(leg.symmetry == no_symmetry for leg in legs)
        return self.apply_basis_perm(a, legs)

    def diagonal_from_block(self, a: Block, leg: Space) -> DiagonalData:
        return self.apply_basis_perm(a, [leg])

    def mask_from_block(self, a: Block, large_leg: Space, small_leg: ElementarySpace
                        ) -> DiagonalData:
        data = self.block_to_dtype(a, Dtype.bool)
        data = self.apply_basis_perm(data, [large_leg])
        return data

    def from_block_func(self, func, legs: list[Space], num_domain_legs: int, func_kwargs={}):
        return func(tuple(l.dim for l in legs), **func_kwargs)

    def diagonal_from_block_func(self, func, leg: Space, func_kwargs={}) -> DiagonalData:
        return func((leg.dim,), **func_kwargs)

    def zero_data(self, legs: list[Space], dtype: Dtype, num_domain_legs: int):
        return self.zero_block(shape=[l.dim for l in legs], dtype=dtype)

    def zero_diagonal_data(self, leg: Space, dtype: Dtype) -> DiagonalData:
        return self.zero_block(shape=[leg.dim], dtype=dtype)

    def eye_data(self, legs: list[Space], dtype: Dtype) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        return self.eye_block(legs=[l.dim for l in legs], dtype=dtype)

    def copy_data(self, a: BlockDiagonalTensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_copy(a.data)

    def _data_repr_lines(self, a: BlockDiagonalTensor, indent: str, max_width: int, max_lines: int):
        return [f'{indent}* Data:'] + self._block_repr_lines(a.data, indent=indent + '  ', max_width=max_width,
                                                            max_lines=max_lines - 1)

    def tdot(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, axs_a: list[int], axs_b: list[int]) -> Data:
        return self.block_tdot(a.data, b.data, axs_a, axs_b)

    def svd(self, a: BlockDiagonalTensor, new_vh_leg_dual: bool, algorithm: str | None, compute_u: bool,
            compute_vh: bool) -> tuple[Data, DiagonalData, Data, ElementarySpace]:
        u, s, vh = self.matrix_svd(a.data, algorithm=algorithm, compute_u=compute_u, compute_vh=compute_vh)
        new_leg = ElementarySpace.from_trivial_sector(len(s), is_dual=new_vh_leg_dual)
        return u, s, vh, new_leg

    def qr(self, a: BlockDiagonalTensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, ElementarySpace]:
        q, r = self.matrix_qr(a.data, full=full)
        new_leg_dim = self.block_shape(r)[0]
        new_leg = ElementarySpace.from_trivial_sector(new_leg_dim, is_dual=new_r_leg_dual)
        return q, r, new_leg

    def outer(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        return self.block_outer(a.data, b.data)

    def inner(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, do_conj: bool, axs2: list[int] | None) -> complex:
        return self.block_inner(a.data, b.data, do_conj=do_conj, axs2=axs2)

    def permute_legs(self, a: BlockDiagonalTensor, permutation: list[int] | None, num_domain_legs: int) -> Data:
        if permutation is None:
            return a.data
        return self.block_permute_axes(a.data, permutation)

    def trace_full(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        return self.block_trace_full(a.data, idcs1, idcs2)

    def trace_partial(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        return self.block_trace_partial(a.data, idcs1, idcs2, remaining_idcs)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return self.block_sum_all(a.data)

    def conj(self, a: BlockDiagonalTensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_conj(a.data)

    def combine_legs(self, a: BlockDiagonalTensor, combine_slices: list[int, int],
                     product_spaces: list[ProductSpace], new_axes: list[int],
                     final_legs: list[Space]) -> Data:
        return self.block_combine_legs(a.data, combine_slices)

    def split_legs(self, a: BlockDiagonalTensor, leg_idcs: list[int],
                   final_legs: list[Space]) -> Data:
        return self.block_split_legs(a.data, leg_idcs, [[s.dim for s in a.legs[i].spaces]
                                                        for i in leg_idcs])

    def add_trivial_leg(self, a: BlockDiagonalTensor, pos: int, to_domain: bool) -> Data:
        return self.block_add_axis(a.data, pos)

    def almost_equal(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, rtol: float, atol: float) -> bool:
        return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: BlockDiagonalTensor, idcs: list[int]) -> Data:
        return self.block_squeeze_legs(a.data, idcs)

    def norm(self, a: BlockDiagonalTensor | DiagonalTensor, order: int | float = 2) -> float:
        return self.block_norm(a.data, order=order)

    def act_block_diagonal_square_matrix(self, a: BlockDiagonalTensor, block_method: Callable[[Block], Block]
                                         ) -> Data:
        return block_method(a.data)

    def add(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        return self.block_add(a.data, b.data)

    def mul(self, a: float | complex, b: BlockDiagonalTensor) -> Data:
        return self.block_mul(a, b.data)

    def infer_leg(self, block: Block, legs: list[Space | None], is_dual: bool = False,
                  is_real: bool = False) -> ElementarySpace:
        idx, *more = [n for n, leg in enumerate(legs) if leg is None]
        if more:
            raise ValueError('Can only infer one leg')
        dim = self.block_shape(block)[idx]
        return ElementarySpace.from_trivial_sector(dim, is_dual=is_dual)

    def get_element(self, a: BlockDiagonalTensor, idcs: list[int]) -> complex | float | bool:
        return self.get_block_element(a.data, idcs)

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        # a.data is a single 1D block
        return self.get_block_element(a.data, [idx])

    def set_element(self, a: BlockDiagonalTensor, idcs: list[int], value: complex | float) -> Data:
        return self.set_block_element(a.data, idcs, value)

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        return self.set_block_element(a.data, [idx], value)
    
    def diagonal_data_from_full_tensor(self, a: BlockDiagonalTensor, check_offdiagonal: bool) -> DiagonalData:
        return self.block_get_diagonal(a.data, check_offdiagonal=check_offdiagonal)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        return self.block_from_diagonal(a.data)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        return self.block_from_mask(a.data, dtype=dtype)

    def scale_axis(self, a: BlockDiagonalTensor, b: DiagonalTensor, leg: int) -> Data:
        return self.block_scale_axis(a.data, b.data, leg)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        return func(a.data, **func_kwargs)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        return func(a.data, b.data, **func_kwargs)

    def apply_mask_to_Tensor(self, tensor: BlockDiagonalTensor, mask: Mask, leg_idx: int) -> Data:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=leg_idx)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=0)

    def eigh(self, a: BlockDiagonalTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        return self.block_eigh(a.data, sort=sort)

    def from_flat_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        assert self.block_shape(block) == (leg.dim,)
        return self.apply_basis_perm(block, [leg])

    def to_flat_block_trivial_sector(self, tensor: BlockDiagonalTensor) -> Block:
        return self.apply_basis_perm(tensor.data, tensor.legs, inv=True)

    def inv_part_from_flat_block_single_sector(self, block: Block, leg: Space,
                                               dummy_leg: ElementarySpace) -> Data:
        block = self.apply_basis_perm(block, [leg])
        return self.block_add_axis(block, pos=1)

    def inv_part_to_flat_block_single_sector(self, tensor: BlockDiagonalTensor) -> Block:
        return self.apply_basis_perm(tensor.data[:, 0], [tensor.legs[0]], inv=True)

    def flip_leg_duality(self, tensor: BlockDiagonalTensor, which_legs: list[int],
                         flipped_legs: list[Space], perms: list[np.ndarray]) -> Data:
        return tensor.data
