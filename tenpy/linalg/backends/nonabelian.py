# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, TYPE_CHECKING
from math import prod
import numpy as np

from .abstract_backend import Backend, BlockBackend, Block, Dtype, Data, DiagonalData
from ..groups import FusionStyle, Sector, SectorArray, Symmetry
from ..spaces import VectorSpace, ProductSpace

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import BlockDiagonalTensor, DiagonalTensor, Mask


__all__ = ['block_size', 'subblock_size', 'subblock_slice', 'NonabelianBackend', 'NonAbelianData',
           'FusionTree', 'fusion_trees', 'all_fusion_trees', 'coupled_sectors']


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
        coupled = coupled_sectors(codomain, domain)
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
        coupled = coupled_sectors(codomain, domain)
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


# TODO dedicated fusion_trees module?


class FusionTree:
    r"""
    A fusion tree, which represents the map from uncoupled to coupled sectors

    Example fusion tree with
        uncoupled = [a, b, c, d]
        are_dual = [False, True, True, False]
        inner_sectors = [x, y]
        multiplicities = [m0, m1, m2]

    |    |
    |    coupled
    |    |
    |    m2
    |    |  \
    |    x   \
    |    |    \
    |    m1    \
    |    |  \   \
    |    y   \   \
    |    |    \   \
    |    m0    \   \
    |    |  \   \   \
    |    a   b   c   d
    |    |   |   |   |
    |    |   Z   Z   |
    |    |   |   |   |

    

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
        self.is_abelian = symmetry.is_abelian
        self.braiding_style = symmetry.braiding_style
        self.num_uncoupled = len(uncoupled)
        self.num_vertices = max(len(uncoupled) - 1, 0)
        self.num_inner_edges = max(len(uncoupled) - 2, 0)

    def test_sanity(self):
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
            N = self.symmetry._n_symbol(a, b, c)
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
            for mu in range(self.symmetry._n_symbol(*self.uncoupled, self.coupled)):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [False, False], [], [mu])
        else:
            a1, a2, *a_rest = self.uncoupled
            for b in self.symmetry.fusion_outcomes(a1, a2):
                for rest_tree in fusion_trees(symmetry=self.symmetry, uncoupled=[b] + a_rest, coupled=self.coupled):
                    for mu in range(self.symmetry._n_symbol(a1, a2, b)):
                        yield FusionTree(self.symmetry, self.uncoupled, self.coupled, [False] * len(self.uncoupled),
                                         [b] + rest_tree.inner_sectors, [mu] + rest_tree.multiplicities)

    def __len__(self) -> int:
        if len(self.uncoupled) == 1:
            return 1

        if len(self.uncoupled) == 2:
            return self.symmetry._n_symbol(*self.uncoupled, self.coupled)

        else:
            a1, a2, *a_rest = self.uncoupled
            # TODO if this is used a lot, could cache those lengths of the subtrees
            return sum(
                self.symmetry._n_symbol(a1, a2, b) * len(fusion_trees([b] + a_rest, self.coupled))
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
        # TODO double check this is the right spaces attribute!
        for uncoupled in space.sectors:
            yield from fusion_trees(uncoupled, coupled)


def coupled_sectors(codomain: ProductSpace, domain: ProductSpace) -> SectorArray:
    """The coupled sectors which are admitted by both codomain and domain"""
    # TODO think about duality!
    codomain_coupled = codomain._non_dual_sectors
    domain_coupled = domain._non_dual_sectors
    # OPTIMIZE: find the sectors which appear in both codomain_coupled and domain_coupled
    #  can probably be done much more efficiently, in particular since they are sorted.
    #  look at np.intersect1d for inspiration?
    are_equal = codomain_coupled[:, None, :] == domain_coupled[None, :, :]  # [c_codom, c_dom, q]
    mask = np.any(np.all(are_equal, axis=2), axis=0)  # [c_dom]
    return domain_coupled[mask]
