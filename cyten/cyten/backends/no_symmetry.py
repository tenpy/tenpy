"""Implements a 'dummy' tensor backend that does not exploit symmetries."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import TYPE_CHECKING

from ..block_backends import Block, BlockBackend
from ..block_backends.dtypes import Dtype
from ..symmetries import ElementarySpace, FusionTree, LegPipe, Space, Symmetry, TensorProduct, no_symmetry
from ..tools.misc import rank_data
from ._backend import Data, DiagonalData, MaskData, TensorBackend, conventional_leg_order

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import DiagonalTensor, Mask, SymmetricTensor


class NoSymmetryBackend(TensorBackend):
    """Abstract base class for backends that do not enforce any symmetry.

    Notes
    -----
    The data stored for the various tensor classes defined in ``cyten.tensors`` is::

        - ``SymmetricTensor``:
            A single Block with as many axes as there a legs on the tensor.
            Same leg order as ``Tensor.legs``, i.e. ``[*codomain, *reversed(domain)]``.

        - ``DiagonalTensor`` :
            A single 1D Block. The diagonal of the corresponding 2D block of a ``Tensor``.

        - ``Mask``:
            The bool values indicate which indices of the large leg are kept for the small leg.

    """

    DataCls = 'Block of BlockBackend'  # is dynamically set by __init__
    can_decompose_tensors = True

    def __init__(self, block_backend: BlockBackend):
        super().__init__(block_backend=block_backend)
        self.DataCls = block_backend.BlockCls

    # OVERRIDES

    def test_tensor_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_tensor_sanity(a, is_diagonal=is_diagonal)
        if is_diagonal:
            expect_shape = (a.legs[0].dim,)
        else:
            expect_shape = a.shape
        self.block_backend.test_block_sanity(
            a.data, expect_shape=expect_shape, expect_dtype=a.dtype, expect_device=a.device
        )

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        self.block_backend.test_block_sanity(
            a.data, expect_shape=(a.large_leg.dim,), expect_dtype=Dtype.bool, expect_device=a.device
        )
        assert self.block_backend.sum_all(a.data) == a.small_leg.dim

    # ABSTRACT METHODS:

    def act_block_diagonal_square_matrix(
        self, a: SymmetricTensor, block_method: Callable[[Block], Block], dtype_map: Callable[[Dtype], Dtype] | None
    ) -> Data:
        return block_method(a.data)

    def add_trivial_leg(
        self,
        a: SymmetricTensor,
        legs_pos: int,
        add_to_domain: bool,
        co_domain_pos: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        return self.block_backend.add_axis(a.data, legs_pos)

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        return self.block_backend.allclose(a.data, b.data, rtol=rtol, atol=atol)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        return self.block_backend.apply_mask(tensor.data, mask.data, ax=0)

    def combine_legs(
        self,
        tensor: SymmetricTensor,
        leg_idcs_combine: list[list[int]],
        pipes: list[LegPipe],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        in_domain = [group[0] >= tensor.num_codomain_legs for group in leg_idcs_combine]
        cstyles = [pipe.combine_cstyle != in_dom for pipe, in_dom in zip(pipes, in_domain)]
        return self.block_backend.combine_legs(tensor.data, leg_idcs_combine, cstyles)

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        a_domain = list(reversed(range(a.num_codomain_legs, a.num_legs)))
        b_codomain = list(range(b.num_codomain_legs))
        return self.block_backend.tdot(a.data, b.data, a_domain, b_codomain)

    def copy_data(self, a: SymmetricTensor | DiagonalTensor, device: str = None) -> Data | DiagonalData:
        return self.block_backend.copy_block(a.data, device=device)

    def dagger(self, a: SymmetricTensor) -> Data:
        return self.block_backend.dagger(a.data)

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        return self.block_backend.item(a)

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        return self.block_backend.block_all(a.data)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return self.block_backend.block_any(a.data)

    def diagonal_elementwise_binary(
        self, a: DiagonalTensor, b: DiagonalTensor, func, func_kwargs, partial_zero_is_zero: bool
    ) -> DiagonalData:
        return func(a.data, b.data, **func_kwargs)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool) -> DiagonalData:
        return func(a.data, **func_kwargs)

    def diagonal_from_block(self, a: Block, co_domain: TensorProduct, tol: float) -> DiagonalData:
        return a

    def diagonal_from_sector_block_func(self, func, co_domain: TensorProduct) -> DiagonalData:
        coupled = co_domain.symmetry.trivial_sector
        shape = (co_domain.dim,)
        return func(shape, coupled)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, tol: float | None) -> DiagonalData:
        return self.block_backend.get_diagonal(a.data, tol=tol)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return self.block_backend.sum_all(a.data)

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        return a.data

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[MaskData, ElementarySpace]:
        large_leg = tens.leg
        basis_perm = large_leg._basis_perm
        data = tens.data
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[self.block_backend.to_numpy(data)])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.sum_all(data),
            symmetry=large_leg.symmetry,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        return tens.leg.dual, tens.data

    def eigh(
        self, a: SymmetricTensor, new_leg_dual: bool, sort: str = None
    ) -> tuple[DiagonalData, Data, ElementarySpace]:
        new_leg = a.domain.as_ElementarySpace(is_dual=new_leg_dual)
        J = a.num_codomain_legs
        N = 2 * J
        mat = self.block_backend.permute_axes(a.data, [*range(J), *reversed(range(J, N))])
        k = a.domain.dim
        mat = self.block_backend.reshape(mat, (k,) * 2)
        w, v = self.block_backend.eigh(mat, sort=sort)
        v = self.block_backend.reshape(v, a.shape[:J] + (k,))
        return w, v, new_leg

    def eye_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        return self.block_backend.eye_block(legs=[l.dim for l in co_domain.factors], dtype=dtype, device=device)

    def from_dense_block(self, a: Block, codomain: TensorProduct, domain: TensorProduct, tol: float) -> Data:
        return a

    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        # there are no other sectors, so this is just the unmodified block.
        assert self.block_backend.get_shape(block) == (leg.dim,)

    def from_grid(
        self,
        grid: list[list[SymmetricTensor | None]],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
        left_mult_slices: list[list[int]],
        right_mult_slices: list[list[int]],
        dtype: Dtype,
        device: str,
    ) -> Data:
        heights = left_mult_slices[0]
        widths = right_mult_slices[0]
        data = self.zero_data(codomain=new_codomain, domain=new_domain, dtype=dtype, device=device)
        codom_slcs = [slice(None)] * (len(new_codomain) - 1)
        dom_slcs = [slice(None)] * (len(new_domain) - 1)
        for i, row in enumerate(grid):
            for j, op in enumerate(row):
                if op is None:
                    continue
                row_slc = slice(widths[j], widths[j + 1])
                col_slc = slice(heights[i], heights[i + 1])
                slcs = (col_slc, *codom_slcs, row_slc, *dom_slcs)
                data[slcs] += op.data
        return data

    def from_random_normal(
        self, codomain: TensorProduct, domain: TensorProduct, sigma: float, dtype: Dtype, device: str
    ) -> Data:
        shape = [leg.dim for leg in conventional_leg_order(codomain, domain)]
        return self.block_backend.random_normal(shape, dtype=dtype, sigma=sigma, device=device)

    def from_sector_block_func(self, func, codomain: TensorProduct, domain: TensorProduct) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        coupled = codomain.symmetry.trivial_sector
        shape = tuple(l.dim for l in conventional_leg_order(codomain, domain))
        return func(shape, coupled)

    def from_tree_pairs(
        self,
        trees: dict[tuple[FusionTree, FusionTree], Block],
        codomain: TensorProduct,
        domain: TensorProduct,
        dtype: Dtype,
        device: str,
    ) -> Data:
        assert len(trees) == 1
        (block,) = trees.values()
        expect_shape = tuple(l.dim for l in conventional_leg_order(codomain, domain))
        assert self.block_backend.get_shape(block) == expect_shape
        return block

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        return self.block_backend.block_from_diagonal(a.data)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        return self.block_backend.block_from_mask(a.data, dtype=dtype)

    def get_device_from_data(self, a: Data) -> str:
        return self.block_backend.get_device(a)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return self.block_backend.get_dtype(a)

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        idcs = [
            l.apply_basis_perm(idx, inverse=True, pre_compose=True) for l, idx in zip(conventional_leg_order(a), idcs)
        ]
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
        return self.block_backend.inner(a.data, b.data, do_dagger=do_dagger)

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space, charge_leg: ElementarySpace) -> Data:
        return self.block_backend.add_axis(vector, pos=1)

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        return tensor.data[:, 0]

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        return self.block_backend.linear_combination(a, v.data, b, w.data)

    def lq(self, tensor: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        l_dims = tensor.shape[: tensor.num_codomain_legs]
        q_dims = tensor.shape[tensor.num_codomain_legs :]
        mat = self.block_backend.reshape(tensor.data, (prod(l_dims), prod(q_dims)))
        l, q = self.block_backend.matrix_lq(mat, full=False)
        k = self.block_backend.get_shape(q)[0]
        l = self.block_backend.reshape(l, l_dims + (k,))
        q = self.block_backend.reshape(q, (k,) + q_dims)
        return l, q

    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = mask1.large_leg
        basis_perm = large_leg._basis_perm
        data = func(mask1.data, mask2.data)
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[data])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.sum_all(data),
            symmetry=large_leg.symmetry,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def mask_contract_large_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        data = self.block_backend.apply_mask(tensor.data, mask.data, leg_idx)
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.factors[:]
            spaces[co_domain_idx] = mask.small_leg
            domain = TensorProduct(spaces, symmetry=tensor.symmetry)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.factors[:]
            spaces[co_domain_idx] = mask.small_leg
            codomain = TensorProduct(spaces, symmetry=tensor.symmetry)
        return data, codomain, domain

    def mask_contract_small_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        data = self.block_backend.enlarge_leg(tensor.data, mask.data, leg_idx)
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.factors[:]
            spaces[co_domain_idx] = mask.large_leg
            domain = TensorProduct(spaces, symmetry=tensor.symmetry)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.factors[:]
            spaces[co_domain_idx] = mask.large_leg
            codomain = TensorProduct(spaces, symmetry=tensor.symmetry)
        return data, codomain, domain

    def mask_dagger(self, mask: Mask) -> MaskData:
        return mask.data

    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        basis_perm = large_leg._basis_perm
        if basis_perm is not None:
            basis_perm = rank_data(basis_perm[a])
        small_leg = ElementarySpace.from_trivial_sector(
            dim=self.block_backend.sum_all(a),
            symmetry=large_leg.symmetry,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return a, small_leg

    def mask_to_block(self, a: Mask) -> Block:
        return a.data

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        return self.block_backend.to_dtype(a.data, dtype)

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
            dim=self.block_backend.sum_all(data),
            symmetry=large_leg.symmetry,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def move_to_device(self, a: SymmetricTensor | DiagonalTensor | Mask, device: str) -> Data:
        return self.block_backend.as_block(a.data, device=device)

    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        return self.block_backend.mul(a, b.data)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        return self.block_backend.norm(a.data)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        return self.block_backend.tensor_outer(a.data, b.data, K=a.num_codomain_legs)

    def partial_compose(
        self,
        a: SymmetricTensor,
        b: SymmetricTensor,
        a_first_leg: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        if a_first_leg < a.num_codomain_legs:
            num_contr_legs = b.num_domain_legs
            num_add_legs = b.num_codomain_legs
            idcs_b = list(reversed(range(b.num_codomain_legs, b.num_legs)))
        else:
            num_contr_legs = b.num_codomain_legs
            num_add_legs = b.num_domain_legs
            idcs_b = list(reversed(range(b.num_codomain_legs)))
        idcs_a = list(range(a_first_leg, a_first_leg + num_contr_legs))
        res = self.block_backend.tdot(a.data, b.data, idcs_a, idcs_b)
        perm = (
            list(range(a_first_leg))
            + list(range(a.num_legs - num_contr_legs, a.num_legs - num_contr_legs + num_add_legs))
            + list(range(a_first_leg, a.num_legs - num_contr_legs))
        )
        return self.block_backend.permute_axes(res, perm)

    def partial_trace(
        self, tensor: SymmetricTensor, pairs: list[tuple[int, int]], levels: list[int] | None
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        N = tensor.num_legs
        idcs1 = []
        idcs2 = []
        for i1, i2 in pairs:
            idcs1.append(i1)
            idcs2.append(i2)
        remaining = [n for n in range(N) if n not in idcs1 and n not in idcs2]
        data = self.block_backend.trace_partial(tensor.data, idcs1, idcs2, remaining)
        if len(remaining) == 0:
            return self.block_backend.item(data), None, None
        codomain = TensorProduct(
            [leg for n, leg in enumerate(tensor.codomain) if n in remaining], symmetry=tensor.symmetry
        )
        domain = TensorProduct(
            [leg for n, leg in enumerate(tensor.domain) if N - 1 - n in remaining], symmetry=tensor.symmetry
        )
        return data, codomain, domain

    def permute_legs(
        self,
        a: SymmetricTensor,
        codomain_idcs: list[int],
        domain_idcs: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
        mixes_codomain_domain: bool,
        levels: list[int] | None,
        bend_right: list[bool | None],
    ) -> Data:
        return self.block_backend.permute_axes(a.data, [*codomain_idcs, *reversed(domain_idcs)])

    def qr(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        q_dims = a.shape[: a.num_codomain_legs]
        r_dims = a.shape[a.num_codomain_legs :]
        mat = self.block_backend.reshape(a.data, (prod(q_dims), prod(r_dims)))
        q, r = self.block_backend.matrix_qr(mat, full=False)
        k = self.block_backend.get_shape(r)[0]
        q = self.block_backend.reshape(q, q_dims + (k,))
        r = self.block_backend.reshape(r, (k,) + r_dims)
        return q, r

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        return block_func(tensor.data)

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        return self.block_backend.scale_axis(a.data, b.data, leg)

    def split_legs(
        self,
        a: SymmetricTensor,
        leg_idcs: list[int],
        codomain_split: list[int],
        domain_split: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        dims = []
        cstyles = []
        for n in leg_idcs:
            in_domain, co_domain_idx, _ = a._parse_leg_idx(n)
            if in_domain:
                dims.append([s.dim for s in reversed(a.domain[co_domain_idx].legs)])
                cstyles.append(not a.domain[co_domain_idx].combine_cstyle)
            else:
                dims.append([s.dim for s in a.codomain[co_domain_idx].legs])
                cstyles.append(a.codomain[co_domain_idx].combine_cstyle)
        return self.block_backend.split_legs(a.data, leg_idcs, dims, cstyles)

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        return self.block_backend.squeeze_axes(a.data, idcs)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry == no_symmetry

    def svd(
        self, a: SymmetricTensor, new_co_domain: TensorProduct, algorithm: str | None
    ) -> tuple[Data, DiagonalData, Data]:
        u_dims = a.shape[: a.num_codomain_legs]
        vh_dims = a.shape[a.num_codomain_legs :]
        mat = self.block_backend.reshape(a.data, (prod(u_dims), prod(vh_dims)))
        u, s, vh = self.block_backend.matrix_svd(mat, algorithm=algorithm)
        k = self.block_backend.get_shape(s)[0]
        u = self.block_backend.reshape(u, u_dims + (k,))
        vh = self.block_backend.reshape(vh, (k,) + vh_dims)
        return u, s, vh

    def state_tensor_product(self, state1: Block, state2: Block, pipe: LegPipe):
        # TODO clearly define what this should do in tensors.py first!
        raise NotImplementedError('state_tensor_product not implemented')

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        return a.data

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        # there are no other sectors, so this is essentially the same as to_dense_block.
        return tensor.data

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        return self.block_backend.to_dtype(a.data, dtype)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        return self.block_backend.trace_full(a.data)

    def truncate_singular_values(
        self,
        S: DiagonalTensor,
        chi_max: int | None,
        chi_min: int,
        degeneracy_tol: float,
        trunc_cut: float,
        svd_min: float,
        minimize_error: bool = True,
    ) -> tuple[MaskData, ElementarySpace, float, float]:
        S_np = self.block_backend.to_numpy(S.data)
        keep, err, new_norm = self._truncate_singular_values_selection(
            S=S_np,
            qdims=None,
            chi_max=chi_max,
            chi_min=chi_min,
            degeneracy_tol=degeneracy_tol,
            trunc_cut=trunc_cut,
            svd_min=svd_min,
            minimize_error=minimize_error,
        )
        mask_data = self.block_backend.block_from_numpy(keep, dtype=Dtype.bool)
        if isinstance(S.leg, ElementarySpace):
            is_dual = S.leg.is_dual
        else:
            is_dual = True
        new_leg = ElementarySpace.from_trivial_sector(dim=keep.sum(), symmetry=S.symmetry, is_dual=is_dual)
        return mask_data, new_leg, err, new_norm

    def zero_data(
        self, codomain: TensorProduct, domain: TensorProduct, dtype: Dtype, device: str, all_blocks: bool = False
    ) -> Data:
        return self.block_backend.zeros(
            shape=[l.dim for l in conventional_leg_order(codomain, domain)], dtype=dtype, device=device
        )

    def zero_mask_data(self, large_leg: Space, device: str) -> MaskData:
        return self.block_backend.zeros(shape=[large_leg.dim], dtype=Dtype.bool, device=device)

    def zero_diagonal_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> DiagonalData:
        return self.block_backend.zeros(shape=[co_domain.dim], dtype=dtype, device=device)
