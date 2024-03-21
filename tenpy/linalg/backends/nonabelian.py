# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING
from math import prod
import numpy as np

from .abstract_backend import Backend, BlockBackend, Block, Dtype, Data, DiagonalData
from ..groups import Sector, SectorArray, Symmetry
from ..spaces import VectorSpace, ProductSpace
from ..trees import FusionTree, fusion_trees, allowed_coupled_sectors

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import BlockDiagonalTensor, DiagonalTensor, Mask


__all__ = ['block_size', 'subblock_size', 'subblock_slice', 'NonabelianBackend', 'NonAbelianData']


def block_size(space: ProductSpace, coupled: Sector) -> int:
    """The size of a block"""
    # OPTIMIZE : this is not super efficient. Make sure its not used in tdot / svd etc
    return sum(
        len(fusion_trees(space.symmetry, uncoupled, coupled)) * subblock_size(space, uncoupled)
        for uncoupled in space.iter_uncoupled_sectors()
    )


def subblock_size(space: ProductSpace, uncoupled: tuple[Sector]) -> int:
    """The size of a subblock"""
    # OPTIMIZE : this is not super efficient. Make sure its not used in tdot / svd etc
    return prod(s.sector_multiplicity(a) for s, a in zip(space.spaces, uncoupled))


def subblock_slice(space: ProductSpace, tree: FusionTree) -> slice:
    """The range of indices of a subblock within its block, as a slice."""
    # OPTIMIZE : this is not super efficient. Make sure its not used in tdot / svd etc
    offset = 0
    for uncoupled in space.iter_uncoupled_sectors():
        if all(np.all(a_unc == a_tree) for a_unc, a_tree in zip(uncoupled, tree.uncoupled)):
            break
        num_trees = len(fusion_trees(space.symmetry, uncoupled, tree.coupled))
        offset += num_trees * subblock_size(space, uncoupled)
    else:
        # no break ocurred
        raise ValueError('Uncoupled sectors of `tree` incompatible with `space`')
    offset += fusion_trees(space.symmetry, tree.uncoupled, tree.coupled).index(tree)
    size = subblock_size(space, uncoupled)
    return slice(offset, offset + size)


class NonAbelianData:
    r"""Data stored in a Tensor for :class:`NonabelianBackend`.

    TODO describe/define what blocks are

    Attributes
    ----------
    coupled_sectors : 2D array
        The coupled sectors :math:`c_n` for which there are non-zero blocks.
        Must be ``lexsort( .T)``-ed (this is not checked!).
    blocks : list of 2D Block
        The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
    domain, codomain : ProductSpace
        The domain and codomain of the tensor ``T : domain -> codomain``.
        Must not be nested, i.e. their :attr:`ProductSpace.spaces` can not themselves be
        ProductSpaces.
        TODO can we support nesting or do they *need* to be flat??
        TODO how to encode Tensor.legs -> [domain, codomain] ?? allow a perm here?
    """
    def __init__(self, coupled_sectors: SectorArray, blocks: list[Block], domain: ProductSpace,
                 codomain: ProductSpace, dtype: Dtype):
        self.coupled_sectors = coupled_sectors
        self.blocks = blocks
        self.domain = domain
        self.codomain = codomain
        self.domain_num_legs = K = len(domain.spaces)
        self.codomain_num_legs = J = len(codomain.spaces)
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
        
    def get_subblock(self, splitting_tree: FusionTree, fusion_tree: FusionTree):
        assert np.all(splitting_tree.coupled == fusion_tree.coupled)
        block = self.get_block(fusion_tree.coupled)
        if block is None:
            return None
        idx1 = subblock_slice(self.codomain, splitting_tree)
        idx2 = subblock_slice(self.domain, fusion_tree)
        return block[idx1, idx2]


# TODO do we need to inherit from ABC again??
# TODO eventually remove BlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy
class NonabelianBackend(Backend, BlockBackend, ABC):
    
    DataCls = NonAbelianData

    def test_data_sanity(self, a: BlockDiagonalTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # check domain and codomain
        assert all(not isinstance(s, ProductSpace) for s in a.data.codomain.spaces)
        assert all(not isinstance(s, ProductSpace) for s in a.data.domain.spaces)
        # TODO still need to decide how the a.legs correspond to domain and codomain, and what to do
        #      if a has combined legs.
        #      when that is implemented, check its consistency here
        # coupled sectors must be lexsorted
        perm = np.lexsort(a.data.coupled_sectors.T)
        assert np.all(perm == np.arange(len(perm)))
        # blocks
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

    def get_dtype_from_data(self, a: NonAbelianData) -> Dtype:
        return a.dtype

    def to_dtype(self, a: BlockDiagonalTensor, dtype: Dtype) -> NonAbelianData:
        blocks = [self.block_to_dtype(block, dtype) for block in a.data.blocks]
        return NonAbelianData(a.data.coupled_sectors, blocks, a.data.domain, a.data.codomain, dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def is_real(self, a: BlockDiagonalTensor) -> bool:
        # TODO (JU) this may not even be well-defined if a.symmetry is not a group ...
        #  I think we should take the POV that we are always working over the complex numbers,
        #  even if the dtype is real. in that case, imaginary parts just happen to be 0.
        raise NotImplementedError

    def data_item(self, a: NonAbelianData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError('Not a scalar.')
        return self.block_item(a.blocks[0])

    def to_dense_block(self, a: BlockDiagonalTensor) -> Block:
        raise NotImplementedError  # TODO use self.apply_basis_perm
        res = self.zero_block(a.shape, a.dtype)
        codomain = [a.legs[n] for n in a.data.codomain_legs]
        domain = [a.legs[n] for n in a.data.domain_legs]
        for coupled in a.data.blocks.keys():
            for splitting_tree in all_fusion_trees(codomain, coupled):
                for fusion_tree in all_fusion_trees(domain, coupled):
                    X = self.fusion_tree_to_block(fusion_tree)  # [b1,...,bN,c]
                    degeneracy_data = a.data.get_sub_block(fusion_tree, splitting_tree)  # [j1,...,jM, k1,...,kN]
                    Y = self.block_conj(self.fusion_tree_to_block(splitting_tree))  # [a1,...,aM,c]
                    # symmetric tensors are the identity on the coupled sectors; so we can contract
                    symmetry_data = self.block_tdot(Y, X, [-1], [-1])  # [a1,...,aM , b1,...,bN]
                    # kron into combined indices [(a1,j1), ..., (aM,jM) , (b1,k1),...,(bN,kN)]
                    contribution = self.block_kron(symmetry_data, degeneracy_data)

                    # TODO: get_slice implementation?
                    # TODO: nonabelian (at least) want a map from sector to its index, so we dont have to sort always
                    # TODO: would be better to iterate over uncoupled sectors, together with their slices, then we dont need to look them up.
                    idcs = (*(leg.get_slice(sector) for leg, sector in zip(a.data.codomain, splitting_tree.uncoupled)),
                            *(leg.get_slice(sector) for leg, sector in zip(a.data.domain, fusion_tree.uncoupled)))

                    res[idcs] += contribution
        return res

    def fusion_tree_to_block(self, tree: FusionTree) -> Block:
        """convert a FusionTree to a block, containing its "matrix" representation."""
        raise NotImplementedError  # TODO

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8,
                         rtol: float = 0.00001) -> NonAbelianData:
        # TODO add arg to specify (co-)domain?
        raise NotImplementedError

    def from_block_func(self, func, legs: list[VectorSpace], func_kwargs={}) -> NonAbelianData:
        # TODO add arg to specify (co-)domain?
        codomain = ProductSpace(legs, backend=self).as_flat_product()
        domain = ProductSpace([], backend=self)
        coupled = allowed_coupled_sectors(codomain, domain)
        blocks = [func((block_size(codomain, c), block_size(domain, c)), **func_kwargs)
                  for c in coupled]
        if len(blocks) > 0:
            sample_block = blocks[0]
        else:
            sample_block = func((1,) * len(legs), **func_kwargs)
        dtype = self.block_dtype(sample_block)
        return NonAbelianData(coupled, blocks, domain, codomain, dtype)

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> NonAbelianData:
        # TODO add arg to specify (co-)domain?
        codomain = ProductSpace(legs, backend=self).as_flat_product()
        domain = ProductSpace([], backend=self)
        return NonAbelianData(coupled_sectors=codomain.symmetry.empty_sector_array, blocks=[],
                              domain=domain, codomain=codomain, dtype=dtype)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> NonAbelianData:
        # TODO add arg to specify (co-)domain?
        codomain = ProductSpace(legs, backend=self).as_flat_product()
        domain = ProductSpace([], backend=self)
        coupled = allowed_coupled_sectors(codomain, domain)
        blocks = [self.eye_block((block_size(codomain, c), block_size(domain, c)), dtype=dtype)
                  for c in coupled]
        return NonAbelianData(coupled_sectors=coupled, blocks=blocks, domain=domain,
                              codomain=codomain, dtype=dtype)

    def copy_data(self, a: BlockDiagonalTensor) -> NonAbelianData:
        return NonAbelianData(
            coupled_sectors=a.data.coupled_sectors.copy(),  # OPTIMIZE do we need to copy these?
            blocks=[self.block_copy(block) for block in a.data.blocks],
            codomain=a.data.codomain, domain=a.data.domain
        )

    def _data_repr_lines(self, a: BlockDiagonalTensor, indent: str, max_width: int,
                         max_lines: int) -> list[str]:
        raise NotImplementedError  # TODO

    # TODO implement all backend methods
