# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Callable
from math import prod
import numpy as np

from .abstract_backend import (
    Backend, BlockBackend, Block, Data, DiagonalData, _iter_common_sorted_arrays
)
from ..dtypes import Dtype
from ..symmetries import Sector, SectorArray, Symmetry
from ..spaces import VectorSpace, ProductSpace
from ..trees import FusionTree, fusion_trees

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import BlockDiagonalTensor, DiagonalTensor, Mask


__all__ = ['block_size', 'forest_block_size', 'tree_block_size', 'forest_block_slice',
           'tree_block_slice', 'FusionTreeBackend', 'FusionTreeData']


def block_size(space: ProductSpace, coupled: Sector) -> int:
    """The size of a block"""
    return space.sector_multiplicity(coupled)


def forest_block_size(space: ProductSpace, uncoupled: tuple[Sector], coupled: Sector) -> int:
    """The size of a forest-block"""
    return len(fusion_trees(space.symmetry, uncoupled, coupled)) * tree_block_size(space, uncoupled)


def tree_block_size(space: ProductSpace, uncoupled: tuple[Sector]) -> int:
    """The size of a tree-block"""
    return prod(s.sector_multiplicity(a) for s, a in zip(space.spaces, uncoupled))


def forest_block_slice(space: ProductSpace, uncoupled: tuple[Sector], coupled: Sector) -> slice:
    """The range of indices of a forest-block within its block, as a slice."""
    # OPTIMIZE ?
    offset = 0
    for _unc in space.iter_uncoupled():
        if all(np.all(a == b) for a, b in zip(_unc, uncoupled)):
            break
        offset += forest_block_size(space, _unc)
    else:  # no break ocurred
        raise ValueError('Uncoupled sectors incompatible with `space`')
    size = forest_block_size(space, uncoupled, coupled)
    return slice(offset, offset + size)


def tree_block_slice(space: ProductSpace, tree: FusionTree) -> slice:
    """The range of indices of a tree-block within its block, as a slice."""
    # OPTIMIZE ?
    offset = 0
    for _unc in space.iter_uncoupled():
        if all(np.all(a == b) for a, b in zip(_unc, tree.uncoupled)):
            break
        offset += forest_block_size(space, _unc)
    else:  # no break ocurred
        raise ValueError('Uncoupled sectors incompatible with `space`')
    offset += fusion_trees(space.symmetry, tree.uncoupled, tree.coupled).index(tree)
    size = tree_block_size(space, tree.uncoupled)
    return slice(offset, offset + size)


def _make_domain_codomain(legs: list[VectorSpace], num_domain_legs: int = 0, backend=None
                          ) -> tuple[ProductSpace, ProductSpace]:
    assert 0 <= num_domain_legs < len(legs)
    # need to pass symmetry and is_real, since codomain or domain might be the empty product.
    symmetry = legs[0].symmetry
    is_real = legs[0].is_real
    domain = ProductSpace([l.dual for l in legs[:num_domain_legs]], backend=backend,
                          symmetry=symmetry, is_real=is_real)
    codomain = ProductSpace(legs[num_domain_legs:][::-1], backend=backend, symmetry=symmetry,
                            is_real=is_real)
    return domain, codomain


class FusionTreeData:
    r"""Data stored in a Tensor for :class:`FusionTreeBackend`.

    TODO describe/define what blocks are

    Attributes
    ----------
    coupled_sectors : 2D array
        The coupled sectors :math:`c_n` for which there are non-zero blocks.
        Must be ``lexsort( .T)``-ed (this is not checked!).
        OPTIMIZE force 2D array or allow list of 1D array?
    blocks : list of 2D Block
        The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
    domain, codomain : ProductSpace
        The domain and codomain of the tensor ``T : domain -> codomain``.
        Their "outer" :attr:`ProductSpace.is_dual` must be ``False``.
        Must not be nested, i.e. their :attr:`ProductSpace.spaces` can not themselves be
        ProductSpaces.
        TODO can we support nesting or do they *need* to be flat?? -> adjust test_data_sanity
        They comprise the legs of the tensor as::

            T.legs == [W.dual for W in domain.spaces] + codomain.spaces[::-1]
        
        OPTIMIZE Should we use list[VectorSpace] instead of ProductSpace to save some
                 potential overhead from ProductSpace.__init__ computing its sectors?
                 Having these coupled sectors is *sometimes* useful.
                 Not clear right now if it always is.
    """
    def __init__(self, coupled_sectors: SectorArray, blocks: list[Block], domain: ProductSpace,
                 codomain: ProductSpace, dtype: Dtype):
        assert coupled_sectors.ndim == 2
        self.coupled_sectors = coupled_sectors
        self.coupled_dims = domain.symmetry.batch_sector_dim(coupled_sectors)  # TODO is this used?
        self.blocks = blocks
        self.domain = domain
        self.codomain = codomain
        self.num_domain_legs = K = len(domain.spaces)
        self.num_codomain_legs = J = len(codomain.spaces)
        self.num_legs = K + J
        self.dtype = dtype

    @classmethod
    def from_unsorted(cls, coupled_sectors: SectorArray, blocks: list[Block], domain: ProductSpace,
                      codomain: ProductSpace, dtype: Dtype):
        """Like __init__, but coupled_sectors does not need to be sorted"""
        perm = np.lexsort(coupled_sectors.T)
        coupled_sectors = coupled_sectors[perm, :]
        blocks = [blocks[n] for n in perm]
        return cls(coupled_sectors, blocks, domain, codomain, dtype)

    def get_block(self, coupled_sector: Sector) -> Block | None:
        """Get the block for a given coupled sector.

        Returns ``None`` if the block is not set, even if the coupled sector is not allowed.
        """
        match = np.argwhere(np.all(self.coupled_sectors == coupled_sector[None, :], axis=1))[:, 0]
        if len(match) == 0:
            return None
        return self.blocks[match[0]]

    def get_forest_block(self, coupled: Sector, uncoupled_in: SectorArray,
                         uncoupled_out: SectorArray) -> Block | None:
        """Get the slice of :meth:`get_block` that corresponds to fixed coupled sectors.

        Returns ``None`` if the block is not set, even if the coupled sector is not allowed.
        """
        block = self.get_block(coupled)
        if block is None:
            return None
        idx1 = forest_block_slice(self.codomain, uncoupled_in)
        idx2 = forest_block_slice(self.domain, uncoupled_out)
        return block[idx1, idx2]

    def get_tree_block(self, splitting_tree: FusionTree, fusion_tree: FusionTree):
        """Get the slice of :meth:`get_block` that corresponds to fixed fusion and splitting trees.

        Returns ``None`` if the block is not set, even if the coupled sector is not allowed.
        """
        assert np.all(splitting_tree.coupled == fusion_tree.coupled)
        block = self.get_block(fusion_tree.coupled)
        if block is None:
            return None
        idx1 = tree_block_slice(self.codomain, splitting_tree)
        idx2 = tree_block_slice(self.domain, fusion_tree)
        return block[idx1, idx2]


# TODO do we need to inherit from ABC again?? (same in abelian and no_symmetry)
# TODO eventually remove BlockBackend inheritance, it is not needed,
#      jakob only keeps it around to make his IDE happy  (same in abelian and no_symmetry)
class FusionTreeBackend(Backend, BlockBackend, ABC):
    
    DataCls = FusionTreeData

    def test_data_sanity(self, a: BlockDiagonalTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # check domain and codomain
        assert a.data.num_domain_legs == a.num_domain_legs
        for n in range(a.num_domain_legs):
            # domain: duals of legs[:K]
            assert a.legs[n].can_contract_with(a.data.domain.spaces[n])
        for n in range(a.num_codomain_legs):
            # codomain: legs[K:] in reverse order
            assert a.legs[a.num_domain_legs + n] == a.data.codomain.spaces[-n]
        assert a.data.domain.is_dual is False
        assert a.data.codomain.is_dual is False
        assert all(not isinstance(s, ProductSpace) for s in a.data.codomain.spaces)
        assert all(not isinstance(s, ProductSpace) for s in a.data.domain.spaces)
        # coupled sectors must be lexsorted
        perm = np.lexsort(a.data.coupled_sectors.T)
        assert np.all(perm == np.arange(len(perm)))
        # blocks
        assert len(a.data.coupled_sectors) == len(a.data.blocks)
        for c, block in zip(a.data.coupled_sectors, a.data.blocks):
            assert a.symmetry.is_valid_sector(c)
            # shape correct?
            expect_shape = (block_size(a.data.codomain, c), block_size(a.data.domain, c))
            if is_diagonal:
                assert expect_shape[0] == expect_shape[1]
                expect_shape = (expect_shape[0],)
            assert all(dim > 0 for dim in expect_shape), 'should skip forbidden block'
            assert self.block_shape(block) == expect_shape
            # check matching dtype
            assert self.block_dtype(block) == a.data.dtype

    def test_mask_sanity(self, a: Mask):
        raise NotImplementedError  # TODO

    # TODO do we need leg metadata?
    #  related methods:
    #   - test_leg_sanity
    #   - _fuse_spaces
    #   - add_leg_metadata

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def to_dtype(self, a: BlockDiagonalTensor, dtype: Dtype) -> FusionTreeData:
        blocks = [self.block_to_dtype(block, dtype) for block in a.data.blocks]
        return FusionTreeData(a.data.coupled_sectors, blocks, a.data.domain, a.data.codomain, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def data_item(self, a: FusionTreeData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_item(a.blocks[0])

    def to_dense_block(self, a: BlockDiagonalTensor) -> Block:
        raise NotImplementedError  # TODO use self.apply_basis_perm
        res = self.zero_block(a.shape, a.dtype)
        codomain = [a.legs[n] for n in a.data.codomain_legs]
        domain = [a.legs[n] for n in a.data.domain_legs]
        for coupled in a.data.blocks.keys():
            for splitting_tree in all_fusion_trees(codomain, coupled):
                for fusion_tree in all_fusion_trees(domain, coupled):
                    X = fusion_tree.as_block(backend=self)  # [b1,...,bN,c]
                    degeneracy_data = a.data.get_sub_block(fusion_tree, splitting_tree)  # [j1,...,jM, k1,...,kN]
                    Y = self.block_conj(splitting_tree.as_block(backend=self))  # [a1,...,aM,c]
                    # symmetric tensors are the identity on the coupled sectors; so we can contract
                    symmetry_data = self.block_tdot(Y, X, [-1], [-1])  # [a1,...,aM , b1,...,bN]
                    # kron into combined indices [(a1,j1), ..., (aM,jM) , (b1,k1),...,(bN,kN)]
                    contribution = self.block_kron(symmetry_data, degeneracy_data)

                    # TODO: get_slice implementation?
                    # TODO: want a map from sector to its index, so we dont have to sort always
                    # TODO: would be better to iterate over uncoupled sectors, together with their slices, then we dont need to look them up.
                    idcs = (*(leg.get_slice(sector) for leg, sector in zip(a.data.codomain, splitting_tree.uncoupled)),
                            *(leg.get_slice(sector) for leg, sector in zip(a.data.domain, fusion_tree.uncoupled)))

                    res[idcs] += contribution
        return res

    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        raise NotImplementedError  # TODO

    def from_dense_block(self, a: Block, legs: list[VectorSpace], domain_num_legs: int,
                         tol: float = 1e-8) -> FusionTreeData:
        raise NotImplementedError  # TODO

    def diagonal_from_block(self, a: Block, leg: VectorSpace) -> DiagonalData:
        raise NotImplementedError  # TODO

    def mask_from_block(self, a: Block, large_leg: VectorSpace, small_leg: VectorSpace) -> DiagonalData:
        raise NotImplementedError  # TODO

    def from_block_func(self, func, legs: list[VectorSpace], num_domain_legs: int, func_kwargs={}
                        ) -> FusionTreeData:
        domain, codomain = _make_domain_codomain(legs, num_domain_legs=num_domain_legs, backend=self)
        coupled_sectors = []
        blocks = []
        for i, _ in _iter_common_sorted_arrays(domain._non_dual_sectors, codomain._non_dual_sectors):
            coupled = domain._non_dual_sectors[i]
            shape = (block_size(codomain, coupled), block_size(domain, coupled))
            coupled_sectors.append(coupled)
            blocks.append(func(shape, **func_kwargs))
        if len(blocks) > 0:
            sample_block = blocks[0]
        else:
            sample_block = func((1,) * len(legs), **func_kwargs)
        dtype = self.block_dtype(sample_block)
        coupled_sectors = np.asarray(coupled_sectors, int)
        return FusionTreeData(coupled_sectors, blocks, domain, codomain, dtype)

    def diagonal_from_block_func(self, func, leg: VectorSpace, func_kwargs={}) -> DiagonalData:
        raise NotImplementedError  # TODO

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype, num_domain_legs: int) -> FusionTreeData:
        domain, codomain = _make_domain_codomain(legs, num_domain_legs=num_domain_legs, backend=self)
        return FusionTreeData(coupled_sectors=codomain.symmetry.empty_sector_array, blocks=[],
                              domain=domain, codomain=codomain, dtype=dtype)

    def zero_diagonal_data(self, leg: VectorSpace, dtype: Dtype) -> DiagonalData:
        raise NotImplementedError  # TODO

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype, domain_num_legs: int) -> FusionTreeData:
        domain, codomain = _make_domain_codomain(legs, num_codomain=domain_num_legs, backend=self)
        raise NotImplementedError  # TODO use _iter_common_... instead of allowed_coupled_sectors
        coupled = allowed_coupled_sectors(codomain, domain)
        blocks = [self.eye_block((block_size(codomain, c), block_size(domain, c)), dtype=dtype)
                  for c in coupled]
        return FusionTreeData(coupled_sectors=coupled, blocks=blocks, domain=domain,
                              codomain=codomain, dtype=dtype)

    def copy_data(self, a: BlockDiagonalTensor) -> FusionTreeData:
        return FusionTreeData(
            coupled_sectors=a.data.coupled_sectors.copy(),  # OPTIMIZE do we need to copy these?
            blocks=[self.block_copy(block) for block in a.data.blocks],
            codomain=a.data.codomain, domain=a.data.domain
        )

    def _data_repr_lines(self, a: BlockDiagonalTensor, indent: str, max_width: int,
                         max_lines: int) -> list[str]:
        raise NotImplementedError  # TODO

    def tdot(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, axs_a: list[int],
             axs_b: list[int]) -> Data:
        # TODO need to be able to specify levels of braiding in general case!
        # TODO offer separate planar version? or just high
        raise NotImplementedError  # TODO

    def svd(self, a: BlockDiagonalTensor, new_vh_leg_dual: bool, algorithm: str | None,
            compute_u: bool, compute_vh: bool) -> tuple[Data, DiagonalData, Data, VectorSpace]:
        # TODO need to redesign Backend.svd specification! need to allow more than two legs!
        # TODO need to be able to specify levels of braiding in general case!
        raise NotImplementedError  # TODO

    def qr(self, a: BlockDiagonalTensor, new_r_leg_dual: bool, full: bool
           ) -> tuple[Data, Data, VectorSpace]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError  # TODO

    def outer(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        # TODO what target leg order is easiest? does it match the one specified in Backend.outer?
        raise NotImplementedError  # TODO

    def inner(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, do_conj: bool,
              axs2: list[int] | None) -> complex:
        raise NotImplementedError  # TODO
    
    def permute_legs(self, a: BlockDiagonalTensor, permutation: list[int] | None, num_domain_legs: int
                     ) -> Data:
        # TODO need to specify levels for braiding and partitioning into domain / codomain
        raise NotImplementedError  # TODO

    def trace_full(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int]
                   ) -> float | complex:
        raise NotImplementedError  # TODO

    def trace_partial(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int],
                      remaining_idcs: list[int]) -> Data:
        raise NotImplementedError  # TODO

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        raise NotImplementedError  # TODO

    def conj(self, a: BlockDiagonalTensor | DiagonalTensor) -> Data | DiagonalData:
        # TODO what does this even mean? transpose of dagger?
        # TODO should we offer transpose and dagger too?
        raise NotImplementedError  # TODO

    def combine_legs(self, a: BlockDiagonalTensor, combine_slices: list[int, int],
                     product_spaces: list[ProductSpace], new_axes: list[int],
                     final_legs: list[VectorSpace]) -> Data:
        raise NotImplementedError  # TODO
        
    def split_legs(self, a: BlockDiagonalTensor, leg_idcs: list[int],
                   final_legs: list[VectorSpace]) -> Data:
        # TODO do we need metadata to split, like in abelian?
        raise NotImplementedError  # TODO

    def add_trivial_leg(self, a: BlockDiagonalTensor, pos: int, to_domain: bool) -> Data:
        raise NotImplementedError  # TODO

    def almost_equal(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, rtol: float, atol: float
                     ) -> bool:
        raise NotImplementedError  # TODO

    def squeeze_legs(self, a: BlockDiagonalTensor, idcs: list[int]) -> Data:
        raise NotImplementedError  # TODO

    def norm(self, a: BlockDiagonalTensor | DiagonalTensor, order: int | float = None) -> float:
        # TODO be careful about weight with quantum dimensions! probably only support 2 norm.
        raise NotImplementedError  # TODO

    def act_block_diagonal_square_matrix(self, a: BlockDiagonalTensor,
                                         block_method: Callable[[Block], Block]) -> Data:
        raise NotImplementedError  # TODO

    def add(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        raise NotImplementedError  # TODO

    def mul(self, a: float | complex, b: BlockDiagonalTensor) -> Data:
        raise NotImplementedError  # TODO

    def infer_leg(self, block: Block, legs: list[VectorSpace | None], is_dual: bool = False,
                  is_real: bool = False) -> VectorSpace:
        raise NotImplementedError  # TODO

    def get_element(self, a: BlockDiagonalTensor, idcs: list[int]) -> complex | float | bool:
        raise NotImplementedError  # TODO

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        raise NotImplementedError  # TODO

    def set_element(self, a: BlockDiagonalTensor, idcs: list[int], value: complex | float) -> Data:
        # TODO not sure this can even be done sensibly, one entry of the dense block
        #      affects in general many entries of the blocks.
        raise NotImplementedError  # TODO

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        # TODO not sure this can even be done sensibly, one entry of the dense block
        #      affects in general many entries of the blocks.
        raise NotImplementedError  # TODO

    def diagonal_data_from_full_tensor(self, a: BlockDiagonalTensor, check_offdiagonal: bool
                                       ) -> DiagonalData:
        raise NotImplementedError  # TODO

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        raise NotImplementedError  # TODO

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        raise NotImplementedError  # TODO

    def scale_axis(self, a: BlockDiagonalTensor, b: DiagonalTensor, leg: int) -> Data:
        raise NotImplementedError  # TODO

    # TODO how does entropy work here, should it consider quantum dimensions?

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs,
                                   maps_zero_to_zero: bool) -> DiagonalData:
        raise NotImplementedError  # TODO

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool) -> DiagonalData:
        raise NotImplementedError  # TODO

    def apply_mask_to_Tensor(self, tensor: BlockDiagonalTensor, mask: Mask, leg_idx: int) -> Data:
        raise NotImplementedError  # TODO

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        raise NotImplementedError  # TODO

    def eigh(self, a: BlockDiagonalTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError  # TODO

    def from_flat_block_trivial_sector(self, block: Block, leg: VectorSpace) -> Data:
        raise NotImplementedError  # TODO

    def to_flat_block_trivial_sector(self, tensor: BlockDiagonalTensor) -> Block:
        raise NotImplementedError  # TODO

    def inv_part_from_flat_block_single_sector(self, block: Block, leg: VectorSpace,
                                               dummy_leg: VectorSpace) -> Data:
        raise NotImplementedError  # TODO

    def inv_part_to_flat_block_single_sector(self, tensor: BlockDiagonalTensor) -> Block:
        raise NotImplementedError  # TODO

    def flip_leg_duality(self, tensor: BlockDiagonalTensor, which_legs: list[int],
                         flipped_legs: list[VectorSpace], perms: list[np.ndarray]) -> Data:
        # TODO think carefully about what this means.
        raise NotImplementedError  # TODO
