# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Block
from ..symmetries import Sector, Symmetry, VectorSpace, ProductSpace
from ..tensors import Dtype, Tensor

__all__ = ['AbstractNonabelianBackend']


@dataclass
class NonAbelianData:
    # NOTE: assumes the first num_in_legs legs to be "ingoing" i.e. part of the domain, the rest
    # as outgoing, i.e. part of the codomain
    # per default, all legs are outgoing
    blocks: dict[Sector, Block]
    codomain: ProductSpace
    domain: ProductSpace
    dtype: Dtype
    

# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy
class AbstractNonabelianBackend(AbstractBackend, AbstractBlockBackend, ABC):

    def finalize_Tensor_init(self, a: Tensor):
        # TODO do we need to do stuff?
        return super().finalize_Tensor_init(a)

    def get_dtype_from_data(self, a: NonAbelianData) -> Dtype:
        return a.dtype

    def to_dtype(self, a: Tensor, dtype: Dtype) -> NonAbelianData:
        blocks = {sector: self.block_to_dtype(block, dtype) 
                  for sector, block in a.data.blocks.items()}
        return NonAbelianData(blocks=blocks, codomain=a.data.codomain, domain=a.data.domain, 
                                    dtype=a.dtype)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return True

    def is_real(self, a: Tensor) -> bool:
        return all(self.block_is_real(block) for block in a.data.blocks.values())

    def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]
             ) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def data_item(self, a: NonAbelianData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError('Not a scalar.')
        if len(a.blocks) == 0:
            raise RuntimeError('This should not happen')
        return self.block_item(next(iter(a.blocks.values())))

    def to_dense_block(self, a: Tensor) -> Block:
        # FIXME not done here
        res = self.zero_block(a.shape, a.dtype)
        for coupled in a.data.blocks.keys():
            for splitting_tree in all_fusion_trees(a.data.codomain, coupled):
                for fusion_tree in all_fusion_trees(a.data.domain, coupled):
                    # [b1,...,bN,c]
                    X = self.fusion_tree_to_block(fusion_tree)  # FIXME implement

                    # [j1,...,jM, k1,...,kN]
                    degeneracy_data = a.data.get_sub_block(fusion_tree, splitting_tree)

                    # [a1,...,aM,c]
                    Y = self.block_conj(self.fusion_tree_to_block(splitting_tree))

                    # symmetric tensors are the identity on the coupled sectors; so we can contract
                    # [a1,...,aM , b1,...,bN]
                    symmetry_data = self.block_tdot(Y, X, [-1], [-1])

                    # kron into combined indeces [(a1,j1), ..., (aM,jM) , (b1,k1),...,(bN,kN)]
                    contribution = self.block_kron(symmetry_data, degeneracy_data)  # FIXME implmnt

                    # FIXME get_slice implementation?
                    # TODO nonabelian (at least) want a map from sector to its index, so we dont have to sort always
                    idcs = (*(leg.get_slice(sector) for leg, sector in zip(a.data.codomain, splitting_tree.uncoupled)),
                            *(leg.get_slice(sector) for leg, sector in zip(a.data.domain, fusion_tree.uncoupled)))

                    res[idcs] += contribution
        return res  
                    

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, 
                         rtol: float = 0.00001) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> NonAbelianData:
        codomain = ProductSpace(legs)
        domain = ProductSpace([])
        # FIXME implement block_size
        # FIXME coupled sectors: second argument optional!
        blocks = {c: self.zero_block((codomain.block_size(c), domain.block_size(c)), dtype) 
                  for c in coupled_sectors(codomain, domain)}
        return NonAbelianData(blocks, codomain=codomain, domain=domain, dtype=dtype)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> NonAbelianData:
        codomain = ProductSpace(legs)
        # TODO think about if ProductSpace([l.dual for l in legs]) is ProductSpace(legs).dual
        domain = codomain.dual
        blocks = {c: self.eye_block([codomain.block_size(c)], dtype) 
                  for c in coupled_sectors(codomain)}
        return NonAbelianData(blocks, codomain=codomain, domain=domain, dtype=dtype)

    def copy_data(self, a: Tensor) -> NonAbelianData:
        # TODO define more clearly what this should even do
        # TODO should we have Tensor.codomain and Tensor.domain properties?
        #      how else would we expose details about the grouping?
        return NonAbelianData(
            blocks={sector: self.block_copy(block) for sector, block in a.data.blocks.values()},
            codomain=a.data.codomain, domain=a.data.domain, dtype=a.dtype
        )

    def _data_repr_lines(self, data: NonAbelianData, indent: str, max_width: int, 
                         max_lines: int) -> list[str]:
        raise NotImplementedError  # FIXME

    def svd(self, a: Tensor, axs1: list[int], axs2: list[int], new_leg: VectorSpace | None
            ) -> tuple[NonAbelianData, NonAbelianData, NonAbelianData, VectorSpace]:
        # can use self.matrix_svd
        raise NotImplementedError  # FIXME

    def outer(self, a: Tensor, b: Tensor) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def inner(self, a: Tensor, b: Tensor, axs2: list[int] | None) -> complex:
        raise NotImplementedError  # FIXME

    def transpose(self, a: Tensor, permutation: list[int]) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def trace(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def conj(self, a: Tensor) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def combine_legs(self, a: Tensor, idcs: list[int], new_leg: ProductSpace) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def split_leg(self, a: Tensor, leg_idx: int) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def allclose(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        raise NotImplementedError  # FIXME

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def norm(self, a: Tensor) -> float:
        raise NotImplementedError  # FIXME

    def exp(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def log(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def random_gaussian(self, legs: list[VectorSpace], dtype: Dtype, sigma: float) -> NonAbelianData:
        raise NotImplementedError  # FIXME

    def add(self, a: Tensor, b: Tensor) -> NonAbelianData:
        # TODO: not checking leg compatibility. ok?
        blocks = {coupled: block_a + block_b 
                  for coupled, block_a, block_b in _block_pairs(a.data, b.data)}
        return NonAbelianData(blocks, codomain=a.data.codomain, domain=a.data.domain, dtype=a.dtype)

    def mul(self, a: float | complex, b: Tensor) -> NonAbelianData:
        blocks = {coupled: a * block for coupled, block in b.data.blocks.items()}
        return NonAbelianData(blocks, codomain=b.data.codomain, domain=b.data.domain, dtype=b.dtype)
