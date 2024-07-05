# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from math import prod

from .abstract_backend import (TensorBackend, BlockBackend, Data, DiagonalData, MaskData, Block,
                               conventional_leg_order)
from ..dtypes import Dtype
from ..symmetries import no_symmetry, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace
from ...tools.misc import rank_data


__all__ = ['NoSymmetryBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


# TODO eventually remove BlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy


class NoSymmetryBackend(TensorBackend):
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
            The bool values indicate which indices of the large leg are kept for the small leg.

    """
    DataCls = "Block of BlockBackend"  # is dynamically set by __init__
    can_decompose_tensors = True

    def __init__(self, block_backend: BlockBackend):
        super().__init__(block_backend=block_backend)
        self.DataCls = block_backend.BlockCls

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        if is_diagonal:
            assert self.block_backend.block_shape(a.data) == (a.legs[0].dim,), \
                f'{self.block_backend.block_shape(a)} != {(a.legs[0].dim,)}'
        else:
            assert self.block_backend.block_shape(a.data) == a.shape, \
                f'{self.block_backend.block_shape(a.data)} != {a.shape}'

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        assert self.block_backend.block_shape(a.data) == (a.large_leg.dim,)
        assert self.block_backend.block_sum_all(a.data) == a.small_leg.dim

    # ABSTRACT METHODS:

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block],
                                         dtype_map: Callable[[Dtype], Dtype] | None) -> Data:
        return block_method(a.data)

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        return self.block_backend.block_add_axis(a.data, legs_pos)

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        return self.block_backend.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        return self.block_backend.block_apply_mask(tensor.data, mask.data, ax=0)

    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     product_spaces: list[ProductSpace],
                     new_codomain: ProductSpace,
                     new_domain: ProductSpace,
                     ) -> Data:
         return self.block_backend.block_combine_legs(tensor.data, leg_idcs_combine)

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        a_domain = list(reversed(range(a.num_codomain_legs, a.num_legs)))
        b_codomain = list(range(b.num_codomain_legs))
        return self.block_backend.block_tdot(a.data, b.data, a_domain, b_codomain)

    def copy_data(self, a: SymmetricTensor | DiagonalTensor) -> Data | DiagonalData:
        return self.block_backend.block_copy(a.data)

    def dagger(self, a: SymmetricTensor) -> Data:
        return self.block_backend.block_dagger(a.data)

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        return self.block_backend.block_item(a)
    
    def diagonal_all(self, a: DiagonalTensor) -> bool:
        return self.block_backend.block_all(a.data)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return self.block_backend.block_any(a.data)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        return func(a.data, b.data, **func_kwargs)
    
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        return func(a.data, **func_kwargs)
    
    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        return a

    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        coupled = co_domain.symmetry.trivial_sector
        shape = (co_domain.dim,)
        return func(shape, coupled)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool) -> DiagonalData:
        return self.block_backend.block_get_diagonal(a.data, check_offdiagonal=check_offdiagonal)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return self.block_backend.block_sum_all(a.data)

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        return a.data

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[MaskData, ElementarySpace]:
        large_leg = tens.leg
        basis_perm = large_leg._basis_perm
        data = tens.data
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[data])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.block_sum_all(data), symmetry=large_leg.symmetry, is_dual=large_leg.is_dual,
            basis_perm=basis_perm
        )
        return data, small_leg

    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        return tens.leg.dual, tens.data

    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        J = a.num_codomain_legs
        N = 2 * J
        mat = self.block_backend.block_permute_axes(a.data, [*range(J), *reversed(range(J, N))])
        k = a.domain.dim
        mat = self.block_backend.block_reshape(mat, (k,) * 2)
        w, v = self.block_backend.block_eigh(mat, sort=sort)
        v = self.block_backend.block_reshape(v, a.shape[:J] + (k,))
        return w, v

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        return self.block_backend.eye_block(legs=[l.dim for l in co_domain.spaces], dtype=dtype)

    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> Data:
        return a

    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        # there are no other sectors, so this is just the unmodified block.
        assert self.block_backend.block_shape(block) == (leg.dim,)

    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        shape = [leg.dim for leg in conventional_leg_order(codomain, domain)]
        return self.block_backend.block_random_normal(shape, dtype=dtype, sigma=sigma)

    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        coupled = codomain.symmetry.trivial_sector
        shape = tuple(l.dim for l in conventional_leg_order(codomain, domain))
        return func(shape, coupled)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        return self.block_backend.block_from_diagonal(a.data)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        return self.block_backend.block_from_mask(a.data, dtype=dtype)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_backend.block_dtype(a)

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        idcs = [l.parse_index(idx)[1] for l, idx in zip(conventional_leg_order(a), idcs)]
        return self.block_backend.get_block_element(a.data, idcs)

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        # a.data is a single 1D block
        _, idx = a.leg.parse_index(idx)
        return self.block_backend.get_block_element(a.data, [idx])

    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        idcs = [l.parse_index(idx)[1] for l, idx in zip(conventional_leg_order(a), idcs)]
        if a.is_projection:
            small, large = idcs
        else:
            large, small = idcs
        return self.block_backend.get_block_mask_element(a.data, large, small)

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        return self.block_backend.block_inner(a.data, b.data, do_dagger=do_dagger)

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                                charge_leg: ElementarySpace) -> Data:
        return self.block_backend.block_add_axis(vector, pos=1)

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        return tensor.data[:, 0]

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        return self.block_backend.block_linear_combination(a, v.data, b, w.data)

    def lq(self, tensor: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        l_dims = tensor.shape[:tensor.num_codomain_legs]
        q_dims = tensor.shape[tensor.num_codomain_legs:]
        mat = self.block_backend.block_reshape(tensor.data, (prod(l_dims), prod(q_dims)))
        l, q = self.block_backend.matrix_lq(mat, full=False)
        k = self.block_backend.block_shape(q)[0]
        l = self.block_backend.block_reshape(l, l_dims + (k,))
        q = self.block_backend.block_reshape(q, (k,) + q_dims)
        return l, q
    
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func
                            ) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = mask1.large_leg
        basis_perm = large_leg._basis_perm
        data = func(mask1.data, mask2.data)
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[data])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.block_sum_all(data), symmetry=large_leg.symmetry, is_dual=large_leg.is_dual,
            basis_perm=basis_perm
        )
        return data, small_leg

    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        data = self.block_backend.block_apply_mask(tensor.data, mask.data, leg_idx)
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.spaces[:]
            spaces[co_domain_idx] = mask.small_leg
            domain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.spaces[:]
            spaces[co_domain_idx] = mask.small_leg
            codomain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        return data, codomain, domain
    
    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        data = self.block_backend.block_enlarge_leg(tensor.data, mask.data, leg_idx)
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.spaces[:]
            spaces[co_domain_idx] = mask.large_leg
            domain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.spaces[:]
            spaces[co_domain_idx] = mask.large_leg
            codomain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        return data, codomain, domain

    def mask_dagger(self, mask: Mask) -> MaskData:
        return mask.data
    
    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        basis_perm = large_leg._basis_perm
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[a])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.block_sum_all(a), symmetry=large_leg.symmetry, is_dual=large_leg.is_dual,
            basis_perm=basis_perm
        )
        return a, small_leg

    def mask_to_block(self, a: Mask) -> Block:
        return a.data

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        return self.block_backend.block_to_dtype(a.data, dtype)

    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        space_in = tens.codomain[0].dual
        space_out = tens.domain[0].dual
        return space_in, space_out, tens.data

    def mask_unary_operand(self, mask: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = mask.large_leg
        basis_perm = large_leg._basis_perm
        data = func(mask.data)
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[data])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.block_sum_all(data), symmetry=large_leg.symmetry, is_dual=large_leg.is_dual,
            basis_perm=basis_perm
        )
        return data, small_leg
        
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        return self.block_backend.block_mul(a, b.data)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        return self.block_backend.block_norm(a.data)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        return self.block_backend.block_tensor_outer(a.data, b.data, K=a.num_codomain_legs)

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, ProductSpace, ProductSpace]:
        N = tensor.num_legs
        idcs1 = []
        idcs2 = []
        for i1, i2 in pairs:
            idcs1.append(i1)
            idcs2.append(i2)
        remaining = [n for n in range(N) if n not in idcs1 and n not in idcs2]
        data = self.block_backend.block_trace_partial(tensor.data, idcs1, idcs2, remaining)
        if len(remaining) == 0:
            return self.block_backend.block_item(data), None, None
        codomain = ProductSpace(
            [leg for n, leg in enumerate(tensor.codomain) if n in remaining],
            symmetry=tensor.symmetry, backend=self
        )
        domain = ProductSpace(
            [leg for n, leg in enumerate(tensor.domain) if N - 1 - n in remaining],
            symmetry=tensor.symmetry, backend=self
        )
        return data, codomain, domain

    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, ProductSpace, ProductSpace]:
        codomain = ProductSpace([a._as_codomain_leg(i) for i in codomain_idcs],
                                symmetry=a.symmetry, backend=self)
        domain = ProductSpace([a._as_domain_leg(i) for i in domain_idcs],
                              symmetry=a.symmetry, backend=self)
        data = self.block_backend.block_permute_axes(a.data, [*codomain_idcs, *reversed(domain_idcs)])
        return data, codomain, domain

    def qr(self, a: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        q_dims = a.shape[:a.num_codomain_legs]
        r_dims = a.shape[a.num_codomain_legs:]
        mat = self.block_backend.block_reshape(a.data, (prod(q_dims), prod(r_dims)))
        q, r = self.block_backend.matrix_qr(mat, full=False)
        k = self.block_backend.block_shape(r)[0]
        q = self.block_backend.block_reshape(q, q_dims + (k,))
        r = self.block_backend.block_reshape(r, (k,) + r_dims)
        return q, r

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        return block_func(tensor.data)

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        return self.block_backend.block_scale_axis(a.data, b.data, leg)

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], codomain_split: list[int],
                   domain_split: list[int], new_codomain: ProductSpace, new_domain: ProductSpace
                   ) -> Data:
        dims = []
        for n in leg_idcs:
            in_domain, co_domain_idx, _ = a._parse_leg_idx(n)
            if in_domain:
                dims.append([s.dim for s in reversed(a.domain[co_domain_idx].spaces)])
            else:
                dims.append([s.dim for s in a.codomain[co_domain_idx].spaces])
        # determine the dims to split to
        # new_legs = list(conventional_leg_order(new_codomain, new_domain))
        # dims = [[s.dim for s in l.spaces]
        #         for n, l in enumerate(conventional_leg_order(a)) if n in leg_idcs]
        return self.block_backend.block_split_legs(a.data, leg_idcs, dims)

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        return self.block_backend.block_squeeze_legs(a.data, idcs)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def svd(self, a: SymmetricTensor, new_leg: ElementarySpace, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        u_dims = a.shape[:a.num_codomain_legs]
        vh_dims = a.shape[a.num_codomain_legs:]
        mat = self.block_backend.block_reshape(a.data, (prod(u_dims), prod(vh_dims)))
        u, s, vh = self.block_backend.matrix_svd(mat, algorithm=algorithm)
        k = self.block_backend.block_shape(s)[0]
        u = self.block_backend.block_reshape(u, u_dims + (k,))
        vh = self.block_backend.block_reshape(vh, (k,) + vh_dims)
        return u, s, vh
      
    def state_tensor_product(self, state1: Block, state2: Block, prod_space: ProductSpace):
        #TODO clearly define what this should do in tensors.py first!
        raise NotImplementedError('state_tensor_product not implemented')

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        return a.data

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        # there are no other sectors, so this is essentially the same as to_dense_block.
        return tensor.data

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        return self.block_backend.block_to_dtype(a.data, dtype)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        return self.block_backend.block_trace_full(a.data)

    def transpose(self, a: SymmetricTensor) -> tuple[Data, ProductSpace, ProductSpace]:
        perm = [*range(a.num_codomain_legs, a.num_legs), *range(a.num_codomain_legs)]
        data = self.block_backend.block_permute_axes(a.data, perm)
        codomain = a.domain.dual
        domain = a.codomain.dual
        return data, codomain, domain

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype):
        return self.block_backend.zero_block(shape=[l.dim for l in conventional_leg_order(codomain, domain)],
                               dtype=dtype)
    
    def zero_mask_data(self, large_leg: Space) -> MaskData:
        return self.block_backend.zero_block(shape=[large_leg.dim], dtype=Dtype.bool)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return self.block_backend.zero_block(shape=[co_domain.dim], dtype=dtype)
