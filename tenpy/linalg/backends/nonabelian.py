# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Block
from ..symmetries import FusionStyle, Sector, Symmetry, VectorSpace, ProductSpace
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
        res = self.zero_block(a.shape, a.dtype)
        for coupled in a.data.blocks.keys():
            for splitting_tree in all_fusion_trees(a.data.codomain, coupled):
                for fusion_tree in all_fusion_trees(a.data.domain, coupled):
                    # [b1,...,bN,c]
                    X = self.fusion_tree_to_block(fusion_tree)

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

    def fusion_tree_to_block(self, tree: FusionTree) -> Block:
        """convert a FusionTree to a block, containing its "matrix" representation."""
        if tree.num_vertices == 0:
            # tree ist just c <- c for a single sector c == tree.coupled
            if tree.are_dual[0]:
                ...  # FIXME stopped here. should probably write down clear definitions...
                    
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


def _block_pairs(a: NonAbelianData, b: NonAbelianData) -> Iterator[tuple[Sector, Block, Block]]:
    """yield all block pairs, if a coupled sector appears as a key in the blocks dictionary
    of one, but not both inputs, the corresponding other block defaults to zeros like the existing block
    """
    assert a.codomain == b.codomain
    assert a.domain == b.domain

    for coupled, block_a in a.blocks.items():
        block_b = b.blocks.get(coupled, None)
        if block_b is None:
            block_b = 0 * block_a
        yield coupled, block_a, block_b

    for coupled, block_b in b.blocks.items():
        if coupled in a.blocks:
            continue  # have already yielded for that coupled sector
        block_a = a.blocks.get(coupled, None)
        if block_a is None:
            block_a = 0 * block_b
        yield coupled, block_a, block_b


# TODO dedicated fusion_trees module?


class FusionTree:
    """
    A fusion tree, which represents the map from uncoupled to coupled sectors

        example fusion tree with
            uncoupled = [a, b, c, d]
            are_dual = [False, True, True, False]
            inner_sectors = [x, y]
            multiplicities = [m0, m1, m2]

        |
        coupled
        |
        m2
        |  \ 
        x   \ 
        |    \ 
        m1    \ 
        |  \   \ 
        y   \   \ 
        |    \   \ 
        m0    \   \ 
        |  \   \   \ 
        a   b   c   d
        |   |   |   |
        |   Z   Z   |
        |   |   |   |
    
    """

    def __init__(self, symmetry: Symmetry,
                 uncoupled: list[Sector],  # N uncoupled sectors
                 coupled: Sector,
                 are_dual: list[bool],  # N flags: is there a Z isomorphism below the uncoupled sector
                 inner_sectors: list[Sector],  # N - 2 internal sectors
                 multiplicities: list[int] = None,  # N - 1 multiplicity labels; all 0 per default
                 ):
        self.symmetry = symmetry
        self.uncoupled = uncoupled
        self.coupled = coupled
        self.are_dual = are_dual
        self.inner_sectors = inner_sectors
        self.multiplicities = [0] * (len(uncoupled) - 1) if multiplicities is None else multiplicities

        self.fusion_style = symmetry.fusion_style
        self.braiding_style = symmetry.braiding_style
        self.num_uncoupled = len(uncoupled)
        self.num_vertices = max(len(uncoupled) - 1, 0)
        self.num_inner_edges = max(len(uncoupled) - 2, 0)

    def check_sanity(self):
        assert all(self.symmetry.is_valid_sector(a) for a in self.uncoupled)
        assert len(self.uncoupled) == self.num_uncoupled
        assert self.symmetry.is_valid_sector(self.coupled)
        assert len(self.are_dual) == self.num_uncoupled
        assert len(self.inner_sectors) == self.num_inner_edges
        assert all(self.symmetry.is_valid_sector(x) for x in self.inner_sectors)
        assert len(self.multiplicities) == self.num_vertices

        # check that inner_sectors and multiplicities are consistent with the fusion rules
        for vertex in range(self.num_vertices):
            # the two sectors below this vertex
            a = self.uncoupled[0] if vertex == 0 else self.inner_sectors[vertex - 1]
            b = self.uncoupled[vertex + 1]
            # the sector above this vertex
            c = self.inner_sectors[vertex] if vertex < self.num_inner_edges else self.uncoupled
            N = self.symmetry.n_symbol(a, b, c)
            assert N > 0  # if N==0 then a and b can not fuse to c
            assert 0 <= self.multiplicities[vertex] < N

    def __hash__(self) -> int:
        if self.fusion_style == FusionStyle.single:
            # inner sectors are completely determined by uncoupled, all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled)
        elif self.fusion_style == FusionStyle.multiple_unique:
            # all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors)
        else:
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors, self.multiplicities)

        return hash(unique_identifier)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FusionTree):
            return False

        return all(
            self.coupled == other.coupled,
            self.uncoupled == other.uncoupled,
            self.inner_sectors == other.inner_sectors,
            self.multiplicities == other.multiplicities
        )

    def __str__(self) -> str:
        # TODO this ignores are_dual !!
        uncoupled_str = '(' + ', '.join(self.symmetry.sector_str(a) for a in self.uncoupled) + ')'
        entries = [f'{self.symmetry.sector_str(self.coupled)} ⟵ {uncoupled_str}']
        if self.fusion_style in [FusionStyle.multiple_unique, FusionStyle.general] and self.num_inner_edges > 0:
            inner_sectors_str = ', '.join(self.symmetry.sector_str(x) for x in self.inner_sectors)
            entries.append(f'({inner_sectors_str})')
        if self.fusion_style == FusionStyle.general and self.num_vertices > 0:
            mults_str = ', '.join(self.multiplicities)
            entries.append(f'({mults_str})')
        entries = ', '.join(entries)
        return f'FusionTree[{str(self.symmetry)}]({entries})'

    def __repr__(self) -> str:
        return f'FusionTree({self.uncoupled}, {self.coupled}, {self.are_dual}, {self.inner_sectors}, {self.multiplicities})'        


class fusion_trees:
    """
    custom iterator for `FusionTree`s.
    Reason to do this is that we can conveniently access length of the fusion_trees without 
    generating all the actual trees and have a more efficient lookup for the index of a tree.
    """
    def __init__(self, symmetry: Symmetry, uncoupled: list[Sector], coupled: Sector | None = None):
        # DOC: coupled = None means trivial sector
        self.symmetry = symmetry
        assert len(uncoupled) > 0
        self.uncoupled = uncoupled
        self.coupled = symmetry.trivial_sector if coupled is None else coupled

    def __iter__(self) -> Iterator[FusionTree]:
        if len(self.uncoupled) == 0:
            raise RuntimeError
        elif len(self.uncoupled) == 1:
            yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [False], [], [])
        elif len(self.uncoupled) == 2:
            for mu in range(self.symmetry.n_symbol(*self.uncoupled, self.coupled)):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [False, False], [], [mu])
        else:
            a1, a2, *a_rest = self.uncoupled
            for b in self.symmetry.fusion_outcomes(a1, a2):
                for rest_tree in fusion_trees(symmetry=self.symmetry, uncoupled=[b] + a_rest, coupled=self.coupled):
                    for mu in range(self.symmetry.n_symbol(a1, a2, b)):
                        yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [False] * len(self.uncoupled), 
                                         [b] + rest_tree.inner_sectors, [mu] + rest_tree.multiplicities)

    def __len__(self) -> int:
        if len(self.uncoupled) == 1:
            return 1

        if len(self.uncoupled) == 2:
            return self.symmetry.n_symbol(*self.uncoupled, self.coupled)

        else:
            a1, a2, *a_rest = self.uncoupled
            # TODO if this is used a lot, could cache those lengths of the subtrees
            return sum(
                self.symmetry.n_symbol(a1, a2, b) * len(fusion_trees([b] + a_rest, self.coupled))
                for b in self.symmetry.fusion_outcomes(a1, a2)
            )

    def index(self, tree: FusionTree) -> int:
        # TODO inefficient dummy implementation, can exploit __len__ of iterator over subtrees
        # to know how many we need to skip.
        for n, t in enumerate(self):
            if t == tree:
                return n
        raise ValueError(f'tree not in {self}: {tree}')


def all_fusion_trees(space: VectorSpace, coupled: Sector = None) -> Iterator[FusionTree]:
    """Yield all fusion trees from the uncoupled sectors of space to the given coupled sector 
    (if not None) or to all possible coupled sectors (default)"""
    if coupled is None:
        for coupled in coupled_sectors(space):
            yield from all_fusion_trees(space, coupled=coupled)
    else:
        for uncoupled in space.sectors:
            yield from fusion_trees(uncoupled, coupled)


def coupled_sectors(space: VectorSpace) -> Iterable[Sector]:
    """All possible coupled sectors"""
    if isinstance(space, ProductSpace):
        # TODO this can probably be optimized...
        # set is important to exclude duplicates
        return set(
            coupled for uncoupled in space.sectors 
            for coupled in _iter_coupled(space.symmetry, uncoupled)
        )
    else:
        return space.sectors


def _iter_coupled(symmetry: Symmetry, uncoupled: list[Sector]) -> Iterator[Sector]:
    """Iterate all possible coupled sectors of given uncoupled sectors"""
    if len(uncoupled) == 0:
        raise RuntimeError

    if len(uncoupled) == 1:
        yield uncoupled[0]
    elif len(uncoupled) == 2:
        a1, a2, *rest = uncoupled
        for b in symmetry.fusion_outcomes(a1, a2):
            yield from _iter_coupled(symmetry, [b, *rest])
