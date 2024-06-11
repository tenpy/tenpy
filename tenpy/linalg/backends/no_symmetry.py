# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable
import numpy as np

from .abstract_backend import Backend, BlockBackend, Data, DiagonalData, Block, conventional_leg_order
from ..dtypes import Dtype
from ..symmetries import no_symmetry, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace

__all__ = ['NoSymmetryBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


# TODO eventually remove BlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class NoSymmetryBackend(Backend, BlockBackend, metaclass=ABCMeta):
    """Abstract base class for backends that do not enforce any symmetry.

    Notes
    -----
    The data stored for the various tensor classes defined in ``tenpy.linalg.tensors`` is::

        - ``SymmetricTensor``:
            A single Block with as many axes as there a legs on the tensor.
            Same leg order as ``Tensor.legs``, i.e. ``[*codomain, *reversed(domain)]``.

        - ``DiagonalTensor`` :
            A single 1D Block. The diagonal of the corresponding 2D block of a ``Tensor``.

        - ``Mask``:
            These bool values indicate which indices of the large leg are kept for the small leg.

    """
    DataCls = "Block of BlockBackend"  # is dynamically set by __init__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DataCls = self.BlockCls

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        if is_diagonal:
            assert self.block_shape(a.data) == (a.legs[0].dim,), f'{self.block_shape(a)} != {(a.legs[0].dim,)}'
        else:
            assert self.block_shape(a.data) == a.shape, f'{self.block_shape(a)} != {a.shape}'

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        assert self.block_shape(a.data) == (a.legs[0].dim,)
        assert self.block_sum_all(a.data) == a.legs[1].dim

    # ABSTRACT METHODS:

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor, block_method: Callable[[Block], Block]
                                         ) -> Data:
        return block_method(a.data)

    def add(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        return self.block_add(a.data, b.data)

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        return self.block_add_axis(a.data, legs_pos)

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=0)

    def apply_mask_to_Tensor(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int) -> Data:
        return self.apply_mask_to_block(tensor.data, mask.data, ax=leg_idx)

    def combine_legs(self, a: SymmetricTensor, combine_slices: list[int, int],
                     product_spaces: list[ProductSpace], new_axes: list[int],
                     final_legs: list[Space]) -> Data:
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_combine_legs(a.data, combine_slices)

    def conj(self, a: SymmetricTensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_conj(a.data)

    def copy_data(self, a: SymmetricTensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_copy(a.data)

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        return self.block_item(a)

    def _data_repr_lines(self, a: SymmetricTensor, indent: str, max_width: int, max_lines: int):
        block_lines = self._block_repr_lines(a.data, indent=indent + '  ', max_width=max_width,
                                             max_lines=max_lines - 1)
        return [f'{indent}* Data:', *block_lines]

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        return self.block_all(a.data)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return self.block_any(a.data)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        return func(a.data, b.data, **func_kwargs)
    
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        return func(a.data, **func_kwargs)
    
    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        return self.apply_basis_perm(a, co_domain.spaces)

    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        coupled = co_domain.symmetry.trivial_sector
        shape = (co_domain.dim,)
        return func(shape, coupled)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool) -> DiagonalData:
        return self.block_get_diagonal(a.data, check_offdiagonal=check_offdiagonal)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return self.block_sum_all(a.data)

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        return self.apply_basis_perm(a.data, [a.leg], inv=True)

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[DiagonalData, ElementarySpace]:
        raise NotImplementedError  # TODO

    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        return self.block_eigh(a.data, sort=sort)

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        return self.eye_block(legs=[l.dim for l in co_domain.spaces], dtype=dtype)

    def flip_leg_duality(self, tensor: SymmetricTensor, which_legs: list[int],
                         flipped_legs: list[Space], perms: list[np.ndarray]) -> Data:
        return tensor.data

    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> Data:
        return self.apply_basis_perm(a, conventional_leg_order(codomain, domain))

    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        # there are no other sectors, so this is just the unmodified block.
        assert self.block_shape(block) == (leg.dim,)
        return self.apply_basis_perm(block, [leg])

    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        shape = [leg.dim for leg in conventional_leg_order(codomain, domain)]
        return self.block_random_normal(shape, dtype=dtype, sigma=sigma)

    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        coupled = codomain.symmetry.trivial_sector
        shape = tuple(l.dim for l in conventional_leg_order(codomain, domain))
        return func(shape, coupled)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        return self.block_from_diagonal(a.data)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        return self.block_from_mask(a.data, dtype=dtype)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_dtype(a)

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        return self.get_block_element(a.data, idcs)

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        # a.data is a single 1D block
        return self.get_block_element(a.data, [idx])

    def infer_leg(self, block: Block, legs: list[Space | None], is_dual: bool = False
                  ) -> ElementarySpace:
        idx, *more = [n for n, leg in enumerate(legs) if leg is None]
        if more:
            raise ValueError('Can only infer one leg')
        dim = self.block_shape(block)[idx]
        return ElementarySpace.from_trivial_sector(dim, is_dual=is_dual)

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_conj: bool, axs2: list[int] | None) -> complex:
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_inner(a.data, b.data, do_conj=do_conj, axs2=axs2)

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                                charge_leg: ElementarySpace) -> Data:
        block = self.apply_basis_perm(vector, [space])
        return self.block_add_axis(block, pos=1)

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        return self.apply_basis_perm(tensor.data[:, 0], [tensor.legs[0]], inv=True)

    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        data = func(mask1.data, mask2.data)
        raise NotImplementedError
    
    def mask_from_block(self, a: Block, large_leg: Space, small_leg: ElementarySpace
                        ) -> DiagonalData:
        data = self.apply_basis_perm(data, [large_leg])
        return data

    def mask_unary_operand(self, mask: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        data = func(mask.data)
        raise NotImplementedError
        
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        return self.block_mul(a, b.data)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        return self.block_norm(a.data)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        raise NotImplementedError  # TODO not yet reviewed. careful with leg order!
        return self.block_outer(a.data, b.data)

    def permute_legs(self, a: SymmetricTensor, **kw) -> Data:
        # TODO decide signature
        raise NotImplementedError  # TODO not yet reviewed
        if permutation is None:
            return a.data
        return self.block_permute_axes(a.data, permutation)

    def qr(self, a: SymmetricTensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, ElementarySpace]:
        raise NotImplementedError  # TODO not yet reviewed
        q, r = self.matrix_qr(a.data, full=full)
        new_leg_dim = self.block_shape(r)[0]
        new_leg = ElementarySpace.from_trivial_sector(new_leg_dim, is_dual=new_r_leg_dual)
        return q, r, new_leg

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        return self.block_scale_axis(a.data, b.data, leg)

    def set_element(self, a: SymmetricTensor, idcs: list[int], value: complex | float) -> Data:
        return self.set_block_element(a.data, idcs, value)

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        return self.set_block_element(a.data, [idx], value)
    
    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int],
                   final_legs: list[Space]) -> Data:
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_split_legs(a.data, leg_idcs, [[s.dim for s in a.legs[i].spaces]
                                                        for i in leg_idcs])

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        return self.block_squeeze_legs(a.data, idcs)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def svd(self, a: SymmetricTensor, new_vh_leg_dual: bool, algorithm: str | None, compute_u: bool,
            compute_vh: bool) -> tuple[Data, DiagonalData, Data, ElementarySpace]:
        # TODO interface may change
        raise NotImplementedError  # TODO not yet reviewed
        u, s, vh = self.matrix_svd(a.data, algorithm=algorithm, compute_u=compute_u, compute_vh=compute_vh)
        new_leg = ElementarySpace.from_trivial_sector(len(s), is_dual=new_vh_leg_dual)
        return u, s, vh, new_leg

    def tdot(self, a: SymmetricTensor, b: SymmetricTensor, axs_a: list[int], axs_b: list[int]) -> Data:
        # TODO interface may change
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_tdot(a.data, b.data, axs_a, axs_b)

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        return self.apply_basis_perm(a.data, a.legs, inv=True)

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        # there are no other sectors, so this is essentially the same as to_dense_block.
        return self.apply_basis_perm(tensor.data, tensor.legs, inv=True)

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        return self.block_to_dtype(a.data, dtype)

    def trace_full(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_trace_full(a.data, idcs1, idcs2)

    def trace_partial(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        raise NotImplementedError  # TODO not yet reviewed
        return self.block_trace_partial(a.data, idcs1, idcs2, remaining_idcs)

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype):
        return self.zero_block(shape=[l.dim for l in conventional_leg_order(codomain, domain)],
                               dtype=dtype)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return self.zero_block(shape=[co_domain.dim], dtype=dtype)
