# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np
from itertools import product

from .abstract_backend import (
    TensorBackend, BlockBackend, Block, Data, DiagonalData, MaskData
)
from ..dtypes import Dtype
from ..symmetries import Sector, SectorArray, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace
from ..trees import FusionTree, fusion_trees
from ...tools.misc import (
    inverse_permutation, iter_common_sorted_arrays, iter_common_noncommon_sorted,
    iter_common_sorted
)

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


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
        offset += forest_block_size(space, _unc, coupled)
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
        offset += forest_block_size(space, _unc, tree.coupled)
    else:  # no break ocurred
        raise ValueError('Uncoupled sectors incompatible with `space`')
    tree_block_sizes = tree_block_size(space, tree.uncoupled)
    offset += fusion_trees(space.symmetry, tree.uncoupled, tree.coupled).index(tree) * tree_block_sizes
    size = tree_block_sizes
    return slice(offset, offset + size)


def _tree_block_iter(a: SymmetricTensor):
    sym = a.symmetry
    domain_are_dual = [sp.is_dual for sp in a.domain.spaces]
    codomain_are_dual = [sp.is_dual for sp in a.codomain.spaces]
    for (bi, _), block in zip(a.data.block_inds, a.data.blocks):
        coupled = a.codomain.sectors[bi]
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors in _iter_sectors(a.domain.spaces, sym):
            tree_block_width = tree_block_size(a.domain, b_sectors)
            forest_block_width = 0
            for a_sectors in _iter_sectors(a.codomain.spaces, sym):
                tree_block_height = tree_block_size(a.codomain, a_sectors)
                i1 = i1_forest  # start row index of the current tree block
                i2 = i2_forest  # start column index of the current tree block
                for alpha_tree in fusion_trees(sym, a_sectors, coupled, codomain_are_dual):
                    i2 = i2_forest  # reset to the left of the current forest block
                    for beta_tree in fusion_trees(sym, b_sectors, coupled, domain_are_dual):
                        idx1 = slice(i1, i1 + tree_block_height)
                        idx2 = slice(i2, i2 + tree_block_width)
                        entries = block[idx1, idx2]
                        yield alpha_tree, beta_tree, entries
                        i2 += tree_block_width  # move right by one tree block
                    i1 += tree_block_height  # move down by one tree block
                forest_block_height = i1 - i1_forest
                forest_block_width = max(forest_block_width, i2 - i2_forest)
                i1_forest += forest_block_height
            i1_forest = 0  # reset to the top of the block
            i2_forest += forest_block_width


def _tree_block_iter_product_space(space: ProductSpace, coupled: SectorArray | list[Sector],
                                   symmetry: Symmetry) -> Iterator[tuple[FusionTree, slice, int]]:
    """Iterator over all trees in `space` with total charge in `coupled`.
    Yields the `FusionTree`s consistent with the input together with the corresponding slices and
    the index of the total charge within `coupled`. This index coincides with the index enumerating
    the blocks in `FusionTreeData` if `coupled` is lexsorted.
    This function can be used to iterate over domain OR codomain rather than both, as done in
    `_tree_block_iter`.
    """
    are_dual = [sp.is_dual for sp in space.spaces]
    for ind, c in enumerate(coupled):
        i = 0
        for sectors in _iter_sectors(space.spaces, symmetry):
            tree_block_width = tree_block_size(space, sectors)
            for tree in fusion_trees(symmetry, sectors, c, are_dual):
                slc = slice(i, i + tree_block_width)
                yield tree, slc, ind
                i += tree_block_width


def _forest_block_iter_product_space(space: ProductSpace, coupled: SectorArray | list[Sector],
                                     symmetry: Symmetry) -> Iterator[tuple[SectorArray, slice, int]]:
    """Iterator over all forests in `space` with total charge in `coupled`.
    Yields the `SectorArray`s consistent with the input together with the corresponding slices and
    the index of the total charge within `coupled`. This index coincides with the index enumerating
    the blocks in `FusionTreeData` if `coupled` is lexsorted.
    See also `_tree_block_iter_product_space`.
    """
    for ind, c in enumerate(coupled):
        i = 0
        for sectors in _iter_sectors(space.spaces, symmetry):
            forest_block_width = forest_block_size(space, sectors, c)
            slc = slice(i, i + forest_block_width)
            yield sectors, slc, ind
            i += forest_block_width


def _iter_sectors(spaces: list[Space], symmetry: Symmetry) -> Iterator[SectorArray]:
    """Helper iterator over all combinations of sectors.
    Simplified version of `_iter_sectors_mults_slices`.

    Yields
    ------
    uncoupled : list of 1D array of int
        A combination ``[spaces[0].sectors[i0], spaces[1].sectors[i1], ...]``
        of uncoupled sectors
    """
    if len(spaces) == 0:
        yield symmetry.empty_sector_array
        return

    for charges in product(*[space.sectors for space in spaces]):
        yield np.array(charges)

        
def _iter_sectors_mults_slices(spaces: list[Space], symmetry: Symmetry
                               ) -> Iterator[tuple[SectorArray, list[int], list[slice]]]:
    """Helper iterator over all combinations of sectors and respective mults and slices.
    
    Yields
    ------
    uncoupled : list of 1D array of int
        A combination ``[spaces[0].sectors[i0], spaces[1].sectors[i1], ...]``
        of uncoupled sectors
    mults : list of int
        The corresponding ``[spaces[0].multiplicities[i0], spaces[1].multiplicities[i1], ...]``.
    slices : list of slice
        The corresponding ``[slice(*spaces[0].slices[i0]), slice(*spaces[1].slices[i1]), ...]``.
    """
    if len(spaces) == 0:
        yield symmetry.empty_sector_array, [], []
        return
    
    if len(spaces) == 1:
        for a, m, slc in zip(spaces[0].sectors, spaces[0].multiplicities, spaces[0].slices):
            yield a[None, :], [m], [slice(*slc)]
        return
    
    # OPTIMIZE there is probably some itertools magic that does this better?
    # OPTIMIZE or build a grid of indices?
    for a_0, m_0, slc_0 in zip(spaces[0].sectors, spaces[0].multiplicities, spaces[0].slices):
        for a_rest, m_rest, slc_rest in _iter_sectors_mults_slices(spaces[1:], symmetry):
            yield np.concatenate([a_0[None, :], a_rest]), [m_0, *m_rest], [slice(*slc_0), *slc_rest]


class FusionTreeData:
    r"""Data stored in a Tensor for :class:`FusionTreeBackend`.

    TODO describe/define what blocks are

    Attributes
    ----------
    block_inds : 2D array
        Indices that specify the coupled sectors of the non-zero blocks.
        ``block_inds[n] == [i, j]`` indicates that the coupled sector for ``blocks[n]`` is given by
        ``tensor.codomain.sectors[i] == coupled == tensor.domain.sectors[j]``.
    blocks : list of 2D Block
        The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
    is_sorted : bool
        If ``False`` (default), we permute `blocks` and `block_inds` according to
        ``np.lexsort(block_inds.T)``.
        If ``True``, we assume they are sorted *without* checking.
    """
    def __init__(self, block_inds: np.ndarray, blocks: list[Block], dtype: Dtype,
                 is_sorted: bool = False):
        if not is_sorted:
            perm = np.lexsort(block_inds.T)
            block_inds = block_inds[perm, :]
            blocks = [blocks[n] for n in perm]
        self.block_inds = block_inds
        self.blocks = blocks
        self.dtype = dtype

    def block_ind_from_domain_sector_ind(self, domain_sector_ind: int) -> int | None:
        "Return `ind` such that ``block_inds[ind][1] == domain_sector_ind``"
        ind = np.searchsorted(self.block_inds[:,1], domain_sector_ind)
        if ind >= len(self.block_inds) or self.block_inds[ind, 1] != domain_sector_ind:
            return None
        if ind + 1 < self.block_inds.shape[0] and self.block_inds[ind + 1, 1] == domain_sector_ind:
            raise RuntimeError
        return ind

    def discard_zero_blocks(self, backend: BlockBackend, eps: float) -> None:
        "Discard blocks whose norm is below the threshold `eps`"
        keep = []
        for i, block in enumerate(self.blocks):
            if backend.block_norm(block) >= eps:
                keep.append(i)
        self.blocks = [self.blocks[i] for i in keep]
        self.block_inds = self.block_inds[keep]

    @classmethod
    def _zero_data(cls, codomain: ProductSpace, domain: ProductSpace, backend: BlockBackend,
                  dtype: Dtype) -> FusionTreeData:
        """Return `FusionTreeData` consistent with `codomain` and `domain`, where all blocks
        (corresponding to all allowed coupled sectors) are zero. These zero blocks are stored
        such that new values can be assigned step by step. Note however that zero blocks are
        generally not stored in tensors.
        """
        # TODO leave this here? would be more consistet in FusionTreeBackend
        block_shapes = []
        block_inds = []
        for j, coupled in enumerate(domain.sectors):
            i = codomain.sectors_where(coupled)
            if i == None:
                continue
            shp = (block_size(codomain, coupled), block_size(domain, coupled))
            block_shapes.append(shp)
            block_inds.append([i, j])
        block_inds = np.array(block_inds)

        zero_blocks = [backend.zero_block(block_shape, dtype=dtype) for block_shape in block_shapes]
        return cls(block_inds, zero_blocks, dtype=dtype, is_sorted=True)


# TODO do we need to inherit from ABC again?? (same in abelian and no_symmetry)
# TODO eventually remove BlockBackend inheritance, it is not needed,
#      jakob only keeps it around to make his IDE happy  (same in abelian and no_symmetry)
class FusionTreeBackend(TensorBackend):
    
    DataCls = FusionTreeData
    can_decompose_tensors = True

    def __init__(self, block_backend: BlockBackend, eps: float = 1.e-14):
        self.eps = eps
        super().__init__(block_backend)

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # coupled sectors must be lexsorted
        perm = np.lexsort(a.data.block_inds.T)
        assert np.all(perm == np.arange(len(perm)))
        # blocks
        for (i, j), block in zip(a.data.block_inds, a.data.blocks):
            assert 0 <= i < a.codomain.num_sectors
            assert 0 <= j < a.domain.num_sectors
            expect_shape = (a.codomain.multiplicities[i], a.domain.multiplicities[j])
            if is_diagonal:
                assert expect_shape[0] == expect_shape[1]
                expect_shape = (expect_shape[0],)
            assert all(dim > 0 for dim in expect_shape), 'should skip forbidden block'
            self.block_backend.test_block_sanity(block, expect_shape=expect_shape, expect_dtype=a.dtype)

    def test_mask_sanity(self, a: Mask):
        raise NotImplementedError  # TODO

    # TODO do we need leg metadata?
    #  related methods:
    #   - test_leg_sanity
    #   - _fuse_spaces

    # ABSTRACT METHODS

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block],
                                         dtype_map: Callable[[Dtype], Dtype] | None) -> Data:
        block_inds = a.data.block_inds
        res_blocks = []
        # square matrix => codomain == domain have the same sectors
        n = 0
        bi = -1 if n >= len(block_inds) else block_inds[n, 0]
        for i in range(a.codomain.num_sectors):
            if bi == i:
                block = a.data.blocks[n]
                n += 1
                bi = -1 if n >= len(block_inds) else block_inds[n, 0]
            else:
                mult = a.codomain.multiplicities[i]
                block = self.block_backend.zero_block(shape=[mult, mult], dtype=a.dtype)
            res_blocks.append(block_method(block))
        if dtype_map is None:
            dtype = a.dtype
        else:
            dtype = dtype_map(a.dtype)
        res_block_inds = np.repeat(np.arange(a.domain.num_sectors)[:, None], 2, axis=1)
        return FusionTreeData(res_block_inds, res_blocks, dtype)

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        # does not change blocks or coupled sectors at all.
        return a.data

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float
                     ) -> bool:
        # since the coupled sector must agree, it is enough to compare block_inds[:, 0]
        for i, j in iter_common_noncommon_sorted(a.data.block_inds[:, 0], b.data.block_inds[:, 0]):
            if j is None:
                if self.block_backend.block_max_abs(a.data.blocks[i]) > atol:
                    return False
            if i is None:
                if self.block_backend.block_max_abs(b.data.blocks[j]) > atol:
                    return False
            else:
                if not self.block_backend.block_allclose(a.data.blocks[i], b.data.blocks[j], rtol=rtol, atol=atol):
                    return False
        return True

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        raise NotImplementedError('apply_mask_to_DiagonalTensor not implemented')  # TODO
    
    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     product_spaces: list[ProductSpace],
                     new_codomain: ProductSpace,
                     new_domain: ProductSpace,
                     ) -> Data:
        # TODO depending on how this implementation will actually look like, we may want to adjust
        #      the signature.
        #      - new_codomain_combine and new_domain_combine are not used by the other backends.
        #        We have them available from tensors.combine_legs and I (Jakob) expect that they
        #        may be useful here. If they are not, may as well remove them from the args.
        #      - Should clearly design the metadata for productspaces first, then consider;
        #        During the manipulations here, we may accumulate all info to form the new
        #        codomain (in particular the metadata!). Then we should not compute it in
        #        tensors.combine_legs, but rather here. Note that sectors and mults are known (unchanged)
        raise NotImplementedError('combine_legs not implemented')  # TODO

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        res_dtype = Dtype.common(a.dtype, b.dtype)
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        b_blocks = b.data.blocks
        b_block_inds = b.data.block_inds
        if a.dtype != res_dtype:
            a_blocks = [self.block_backend.block_to_dtype(bl, res_dtype) for bl in a_blocks]
        if b.dtype != res_dtype:
            b_blocks = [self.block_backend.block_to_dtype(bl, res_dtype) for bl in b_blocks]
        blocks = []
        block_inds = []
        if len(a.data.block_inds) > 0 and  len(b.data.block_inds) > 0:
            for i, j in iter_common_sorted(a.data.block_inds[:, 1], b.data.block_inds[:, 0]):
                blocks.append(self.block_backend.matrix_dot(a_blocks[i], b_blocks[j]))
                block_inds.append([a_block_inds[i, 0], b_block_inds[j, 1]])
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, res_dtype)

    def copy_data(self, a: SymmetricTensor) -> FusionTreeData:
        return FusionTreeData(
            block_inds=a.data.block_inds.copy(),  # OPTIMIZE do we need to copy these?
            blocks=[self.block_backend.block_copy(block) for block in a.data.blocks],
            dtype=a.data.dtype
        )

    def dagger(self, a: SymmetricTensor) -> Data:
        return FusionTreeData(
            block_inds=a.data.block_inds[:, ::-1],  # domain and codomain have swapped
            blocks=[self.block_backend.block_dagger(b) for b in a.data.blocks],
            dtype=a.dtype
        )

    def data_item(self, a: FusionTreeData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_backend.block_item(a.blocks[0])

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        if len(a.data.blocks) < a.domain.num_sectors:
            # there are missing blocks. -> they contain False -> all(a) == False
            return False
        # now it is enough to check the existing blocks
        return all(self.block_backend.block_all(b) for b in a.data.blocks)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return any(self.block_backend.block_any(b) for b in a.data.blocks)
    
    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool) -> DiagonalData:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        if partial_zero_is_zero:
            blocks = []
            block_inds = []
            for i, j in iter_common_sorted(a_block_inds[:, 0], b_block_inds[:, 0]):
                block_inds.append(a_block_inds[i])
                blocks.append(func(a_blocks[i], b_blocks[j], **func_kwargs))
            if len(block_inds) == 0:
                block_inds = np.zeros((0, 2), int)
            else:
                block_inds = np.array(block_inds, int)
        else:
            n_a = 0  # a_block_inds[:n_a] already visited
            bi_a = -1 if n_a >= len(a_block_inds) else a_block_inds[n_a, 0]
            n_b = 0  # b_block_inds[:n_b] already visited
            bi_b = -1 if n_b >= len(b_block_inds) else b_block_inds[n_b, 0]
            blocks = []
            for i in range(a.codomain.num_sectors):
                if i == bi_a:
                    a_block = a_blocks[n_a]
                    n_a += 1
                    bi_a = -1 if n_a >= len(a_block_inds) else a_block_inds[n_a, 0]
                else:
                    a_block = self.block_backend.zero_block([a.domain.multiplicities[i]], dtype=a.dtype)
                if i == bi_b:
                    b_block = b_blocks[n_b]
                    n_b += 1
                    bi_b = -1 if n_b >= len(b_block_inds) else b_block_inds[n_b, 0]
                else:
                    b_block = self.block_backend.zero_block([a.domain.multiplicities[i]], dtype=b.dtype)
                blocks.append(func(a_block, b_block, **func_kwargs))
            block_inds = np.repeat(np.arange(a.domain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            dtype = self.block_backend.block_dtype(blocks[0])
        else:
            a_block = self.block_backend.ones_block([1], dtype=a.dtype)
            b_block = self.block_backend.ones_block([1], dtype=b.dtype)
            example_block = func(a_block, b_block, **func_kwargs)
            dtype = self.block_backend.block_dtype(example_block)
        return FusionTreeData(block_inds=block_inds, blocks=blocks, dtype=dtype)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs,
                                   maps_zero_to_zero: bool) -> DiagonalData:
        if maps_zero_to_zero:
            blocks = [func(b, **func_kwargs) for b in a.data.blocks]
            block_inds = a.data.block_inds
        else:
            a_blocks = a.data.blocks
            a_block_inds = a.data.block_inds
            n = 0
            bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            blocks = []
            for i in range(a.codomain.num_sectors):
                if i == bi:
                    block = a_blocks[n]
                    n += 1
                    bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
                else:
                    mult = a.codomain.multiplicities[i]
                    block = self.block_backend.zero_block([mult], dtype=a.dtype)
                blocks.append(func(block, **func_kwargs))
            block_inds = np.repeat(np.arange(a.codomain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            dtype = self.block_backend.block_dtype(blocks[0])
        else:
            example_block = func(self.block_backend.ones_block([1], dtype=a.dtype), **func_kwargs)
            dtype = self.block_backend.block_dtype(example_block)
        return FusionTreeData(block_inds=block_inds, blocks=blocks, dtype=dtype)

    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        dtype = self.block_backend.block_dtype(a)
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        blocks = []
        for coupled, mult, slc in zip(co_domain.sectors, co_domain.multiplicities, co_domain.slices):
            dim_c = co_domain.symmetry.sector_dim(coupled)
            entries = self.block_backend.block_reshape(a[slice(*slc)], (dim_c, mult))
            # project onto the identity on the coupled sector
            block = self.block_backend.block_sum(entries, 0) / dim_c
            projected = self.block_backend.block_outer(
                self.block_backend.ones_block([dim_c], dtype=dtype), block
            )
            if self.block_backend.block_norm(entries - projected) > tol * self.block_backend.block_norm(entries):
                raise ValueError('Block is not symmetric up to tolerance.')
            blocks.append(block)
        return FusionTreeData(block_inds, blocks, dtype)

    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        blocks = [func((block_size(co_domain, coupled),), coupled) for coupled in co_domain.sectors]
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            sample_block = blocks[0]
        else:
            sample_block = func((1,), co_domain.symmetry.trivial_sector)
        dtype = self.block_backend.block_dtype(sample_block)
        return FusionTreeData(block_inds, blocks, dtype)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                       ) -> DiagonalData:
        raise NotImplementedError('diagonal_tensor_from_full_tensor not implemented')  # TODO

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return sum(
            (a.domain.sector_qdims[bi] * self.block_backend.block_sum_all(block)
             for bi, block in zip(a.data.block_inds[:, 0], a.data.blocks)),
            a.dtype.zero_scalar
        )

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        assert a.symmetry.can_be_dropped
        block_inds = a.data.block_inds
        res = self.block_backend.zero_block([a.leg.dim], a.dtype)
        for n, i in enumerate(a.data.block_inds[:, 0]):
            dim_c = a.codomain.sector_dims[i]
            symmetry_data = self.block_backend.ones_block([dim_c], dtype=a.dtype)
            degeneracy_data = a.data.blocks[n]
            entries = self.block_backend.block_outer(symmetry_data, degeneracy_data)
            entries = self.block_backend.block_reshape(entries, (-1,))
            res[slice(*a.leg.slices[i])] = entries
        return res

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[DiagonalData, ElementarySpace]:
        raise NotImplementedError('diagonal_to_mask not implemented')
    
    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        dual_leg, perm = tens.leg._dual_space(return_perm=True)
        data = FusionTreeData(block_inds=inverse_permutation(perm)[tens.data.block_inds],
                              blocks=tens.data.blocks, dtype=tens.dtype)
        return dual_leg, data
        
    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        v_blocks = []
        w_blocks = []
        n = 0
        bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        for i in range(a.codomain.num_sectors):
            if i == bi:
                vals, vects = self.block_backend.block_eigh(a_blocks[n], sort=sort)
                v_blocks.append(vects)
                w_blocks.append(vals)
                n += 1
                bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is not block for that sector. => eigenvalues are 0.
                # choose eigenvectors as standard basis vectors (eye matrix)
                block_size = a.codomain.multiplicities[i]
                v_blocks.append(self.block_backend.eye_matrix(block_size, a.dtype))
        #
        v_block_inds = np.repeat(np.arange(a.codomain.num_sectors)[:, None], 2, axis=1)
        v_data = FusionTreeData(v_block_inds, v_blocks, a.dtype)
        w_data = FusionTreeData(a_block_inds, w_blocks, a.dtype.to_real)
        return w_data, v_data

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> FusionTreeData:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        blocks = [self.block_backend.eye_matrix(block_size(co_domain, c), dtype)
                  for c in co_domain.sectors]
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        return FusionTreeData(block_inds, blocks, dtype)

    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> FusionTreeData:
        sym = codomain.symmetry
        assert sym.can_be_dropped
        # convert to internal basis order, where the sectors are sorted and contiguous
        J = len(codomain.spaces)
        K = len(domain.spaces)
        num_legs = J + K
        # [i1,...,iJ,jK,...,j1] -> [i1,...,iJ,j1,...,jK]
        a = self.block_backend.block_permute_axes(a, [*range(J), *reversed(range(J, num_legs))])
        dtype = Dtype.common(self.block_backend.block_dtype(a), sym.fusion_tensor_dtype)
        # main loop: iterate over coupled sectors and construct the respective block.
        block_inds = []
        blocks = []
        norm_sq_projected = 0
        for i, j in iter_common_sorted_arrays(codomain.sectors, domain.sectors):
            coupled = codomain.sectors[i]
            dim_c = codomain.sector_dims[i]
            block_size = [codomain.multiplicities[i], domain.multiplicities[j]]
            # OPTIMIZE could be sth like np.empty
            block = self.block_backend.zero_block(block_size, dtype)
            # iterate over uncoupled sectors / forest-blocks within the block
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(domain.spaces, sym):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = tree_block_size(domain, b_sectors)
                for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(codomain.spaces, sym):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = tree_block_size(codomain, a_sectors)
                    entries = a[(*j1, *j2)]  # [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    # reshape to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                    shape = [0] * (2 * num_legs)
                    shape[::2] = [*a_dims, *b_dims]
                    shape[1::2] = m_dims + n_dims
                    entries = self.block_backend.block_reshape(entries, shape)
                    # permute to [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...nK]
                    perm = [*range(0, 2 * num_legs, 2), *range(1, 2 * num_legs, 2)]
                    entries = self.block_backend.block_permute_axes(entries, perm)
                    num_alpha_trees, num_beta_trees = self._add_forest_block_entries(
                        block, entries, sym, codomain, domain, coupled, dim_c, a_sectors, b_sectors,
                        tree_block_width, tree_block_height, i1, i2
                    )
                    forest_block_height = num_alpha_trees * tree_block_height
                    forest_block_width = num_beta_trees * tree_block_width
                    i1 += forest_block_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_block_width  # move right by one forest-block
            block_norm = self.block_backend.block_norm(block, order=2)
            if block_norm <= 0.:  # TODO small finite tolerance instead?
                continue
            block_inds.append([i, j])
            blocks.append(block)
            contribution = dim_c * block_norm ** 2
            norm_sq_projected += contribution

        # since the symmetric and non-symmetric components of ``a = a_sym + a_rest`` are mutually
        # orthogonal, we have  ``norm(a) ** 2 = norm(a_sym) ** 2 + norm(a_rest) ** 2``.
        # thus ``abs_err = norm(a - a_sym) = norm(a_rest) = sqrt(norm(a) ** 2 - norm(a_sym) ** 2)``
        if tol is not None:
            a_norm_sq = self.block_backend.block_norm(a, order=2) ** 2
            norm_diff_sq = a_norm_sq - norm_sq_projected
            abs_tol_sq = tol * tol * a_norm_sq
            if norm_diff_sq > abs_tol_sq > 0:
                msg = (f'Block is not symmetric up to tolerance. '
                       f'Original norm: {np.sqrt(a_norm_sq)}. '
                       f'Norm after projection: {np.sqrt(norm_sq_projected)}.')
                raise ValueError(msg)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype)
    
    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        raise NotImplementedError('from_dense_block_trivial_sector not implemented')  # TODO

    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        raise NotImplementedError  # TODO

    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> FusionTreeData:
        blocks = []
        block_inds = []
        for i, j in iter_common_sorted_arrays(codomain.sectors, domain.sectors):
            coupled = codomain.sectors[i]
            shape = (block_size(codomain, coupled), block_size(domain, coupled))
            block_inds.append([i, j])
            blocks.append(func(shape, coupled))
        if len(blocks) > 0:
            sample_block = blocks[0]
            block_inds = np.asarray(block_inds, int)
        else:
            sample_block = func((1, 1), codomain.symmetry.trivial_sector)
            block_inds = np.zeros((0, 2), int)
        dtype = self.block_backend.block_dtype(sample_block)
        return FusionTreeData(block_inds, blocks, dtype)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_backend.block_from_diagonal(block) for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, dtype=a.dtype)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        raise NotImplementedError('full_data_from_mask not implemented')  # TODO

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        raise NotImplementedError('get_element not implemented')  # TODO

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        raise NotImplementedError('get_element_diagonal not implemented')  # TODO
    
    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        raise NotImplementedError('get_element_mask not implemented')  # TODO

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        a_blocks = a.data.blocks
        a_codomain_qdims = a.codomain.sector_qdims
        b_blocks = b.data.blocks
        a_codomain_block_inds = a.data.block_inds[:, 0]
        if do_dagger:
            # need to match a.codomain == b.codomain
            b_block_inds = b.data.block_inds[:, 0]
        else:
            # need to math a.codomain == b.domain
            b_block_inds = b.data.block_inds[:, 1]
        res = a.dtype.zero_scalar * b.dtype.zero_scalar
        for i, j in iter_common_sorted(a_codomain_block_inds, b_block_inds):
            inn = self.block_backend.block_inner(a_blocks[i], b_blocks[j], do_dagger=do_dagger)
            res += a_codomain_qdims[a_codomain_block_inds[i]] * inn
        return res
    
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                               charge_leg: ElementarySpace) -> Data:
        raise NotImplementedError('inv_part_from_dense_block_single_sector not implemented')  # TODO

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('inv_part_to_dense_block_single_sector not implemented')  # TODO

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        dtype = v.data.dtype.common(w.data.dtype)
        v_blocks = [self.block_backend.block_to_dtype(_a, dtype) for _a in v.data.blocks]
        w_blocks = [self.block_backend.block_to_dtype(_b, dtype) for _b in w.data.blocks]
        v_block_inds = v.data.block_inds
        w_block_inds = w.data.block_inds
        blocks = []
        block_inds = []
        for i, j in iter_common_noncommon_sorted(v_block_inds[:, 0], w_block_inds[:, 0]):
            if i is None:
                blocks.append(self.block_backend.block_mul(b, w_blocks[j]))
                block_inds.append(w_block_inds[j])
            elif j is None:
                blocks.append(self.block_backend.block_mul(a, v_blocks[i]))
                block_inds.append(v_block_inds[i])
            else:
                blocks.append(
                    self.block_backend.block_linear_combination(a, v_blocks[i], b, w_blocks[j])
                )
                block_inds.append(v_block_inds[i])
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype)

    def lq(self, a: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        l_blocks = []
        l_block_inds = []
        q_blocks = []
        q_block_inds = []
        n = 0
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        for i_new, (i_cod, i_dom) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            q_block_inds.append([i_new, i_dom])
            if bi_cod == i_cod:
                l, q = self.block_backend.matrix_lq(a_blocks[n], full=False)
                l_blocks.append(l)
                q_blocks.append(q)
                l_block_inds.append([i_cod, i_new])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                B_dom = a.domain.multiplicities[i_dom]
                B_new = new_leg.multiplicities[i_new]
                q_blocks.append(self.block_backend.eye_matrix(B_dom, a.dtype)[:B_new, :])
        if len(l_block_inds) == 0:
            l_block_inds = np.zeros((0, 2), int)
        else:
            l_block_inds = np.array(l_block_inds)
        if len(q_block_inds) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds)
        l_data = FusionTreeData(l_block_inds, l_blocks, a.dtype)
        q_data = FusionTreeData(q_block_inds, q_blocks, a.dtype)
        return l_data, q_data
    
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[MaskData, ElementarySpace]:
        raise NotImplementedError('mask_binary_operand not implemented')

    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        raise NotImplementedError('mask_contract_large_leg not implemented')
    
    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        raise NotImplementedError('mask_contract_small_leg not implemented')
    
    def mask_dagger(self, mask: Mask) -> MaskData:
        raise NotImplementedError('mask_dagger not implemented')

    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        raise NotImplementedError('mask_from_block not implemented')  # TODO

    def mask_to_block(self, a: Mask) -> Block:
        raise NotImplementedError

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        raise NotImplementedError

    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        raise NotImplementedError('mask_transpose not implemented')
    
    def mask_unary_operand(self, mask: Mask, func) -> tuple[MaskData, ElementarySpace]:
        raise NotImplementedError
        
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        if a == 0.:
            return self.zero_data(b.codomain, b.domain, b.dtype)
        blocks = [self.block_backend.block_mul(a, T) for T in b.data.blocks]
        if len(blocks) == 0:
            if isinstance(a, float):
                dtype = b.data.dtype
            else:
                dtype = b.data.dtype.to_complex()
        else:
            dtype = self.block_backend.block_dtype(blocks[0])
        return FusionTreeData(b.data.block_inds, blocks, dtype)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        # OPTIMIZE should we offer the square-norm instead?
        norm_sq = 0
        for i, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            norm_sq += a.codomain.sector_qdims[i] * (self.block_backend.block_norm(block) ** 2)
        return self.block_backend.block_sqrt(norm_sq)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        raise NotImplementedError('outer not implemented')  # TODO

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, ProductSpace, ProductSpace]:
        raise NotImplementedError('partial_trace not implemented')  # TODO

    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, ProductSpace, ProductSpace]:
        def exchanges_within_productspace(initial_perm: list, final_perm: list) -> list:
            exchanges = []
            while final_perm != initial_perm:
                for i in range(len(final_perm)):
                    if final_perm[i] != initial_perm[i]:
                        ind = initial_perm.index(final_perm[i])
                        initial_perm[ind-1:ind+1] = initial_perm[ind-1:ind+1][::-1]
                        exchanges.append(ind - 1)
                        break
            return exchanges
        # TODO special cases without bends

        # legs that need to be bent up or down
        bend_up = sorted([i for i in codomain_idcs if i >= a.num_codomain_legs])
        bend_down = sorted([i for i in domain_idcs if i < a.num_codomain_legs])
        num_bend_up = len(bend_up)
        num_bend_down = len(bend_down)
        all_exchanges, all_bend_ups = [], []
        num_operations = []
        levels_None = levels == None
        if not levels_None:
            levels = levels[:]

        # exchanges such that the legs to be bent down are on the right in the codomain
        exchanges = []
        for i in range(len(bend_down)):
            for j in range(bend_down[-1 - i], a.num_codomain_legs - 1 - i):
                exchanges.append(j)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # bend down
        all_exchanges += list(range(a.num_codomain_legs - 1, a.num_codomain_legs - 1 - num_bend_down, -1))
        all_bend_ups += [False] * num_bend_down
        num_operations.append(num_bend_down)

        # exchanges in the domain such that the legs to be bent up are on the right
        exchanges = []
        for i in range(len(bend_up)):
            for j in range(a.num_legs - bend_up[i] - 1, a.num_domain_legs + num_bend_down - 1 - i):
                exchanges.append(a.num_legs - 2 - j)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # exchanges within the domain such that the legs agree with domain_idcs
        inter_domain_idcs = ([i for i in range(a.num_legs-1, a.num_codomain_legs-1, -1) if not i in bend_up]
                             + bend_down[::-1])
        exchanges = exchanges_within_productspace(inter_domain_idcs, domain_idcs)
        exchanges = [a.num_legs - 2 - i for i in exchanges]
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # bend up
        all_exchanges += list(range(a.num_codomain_legs - 1 - num_bend_down,
                                    a.num_codomain_legs - 1 - num_bend_down + num_bend_up))
        all_bend_ups += [True] * num_bend_up
        num_operations.append(num_bend_up)

        # exchanges within the codomain such that the legs agree with codomain_idcs
        inter_codomain_idcs = [i for i in range(a.num_codomain_legs) if not i in bend_down] + bend_up
        exchanges = exchanges_within_productspace(inter_codomain_idcs, codomain_idcs)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # no legs are permuted
        if len(all_exchanges) == 0:
            return a.data, a.codomain, a.domain
        # c symbols are involved
        elif len(all_exchanges) - num_bend_down - num_bend_up > 0 and a.symmetry.braiding_style.value >= 20:
            assert not levels_None, 'Levels must be specified when braiding with anyonic exchange statistics'

        codomain = a.codomain
        domain = a.domain
        coupled = np.array([domain.sectors[i[1]] for i in a.data.block_inds])
        mappings = []
        offset = [0] + list(np.cumsum(num_operations))
        for i in range(len(num_operations)):
            mappings_step = []
            for j in range(num_operations[i]):
                ind = offset[i] + j
                exchange_ind = all_exchanges[ind]
                if exchange_ind != codomain.num_spaces - 1 and not levels_None:
                    overbraid = levels[exchange_ind] > levels[exchange_ind + 1]
                    levels[exchange_ind:exchange_ind + 2] = levels[exchange_ind:exchange_ind + 2][::-1]
                else:
                    overbraid = None

                mapp, codomain, domain, coupled = self._find_approproiate_mapping_dict(codomain, domain,
                                                                                       exchange_ind, coupled,
                                                                                       overbraid, all_bend_ups[ind])
                mappings_step.append(mapp)

            if len(mappings_step) > 0:
                mappings_step = self._combine_multiple_mapping_dicts(mappings_step)
                if i == 0 or i == 5:
                    mappings_step = self._add_prodspace_to_mapping_dict(mappings_step, domain, coupled, 1)
                elif i == 2 or i == 3:
                    mappings_step = self._add_prodspace_to_mapping_dict(mappings_step, codomain, coupled, 0)
                mappings.append(mappings_step)

        mappings = self._combine_multiple_mapping_dicts(mappings)
        axes_perm = codomain_idcs + domain_idcs
        axes_perm = [i if i < a.num_codomain_legs else a.num_legs - 1 - i + a.num_codomain_legs for i in axes_perm]
        data = self._apply_mapping_dict(a, codomain, domain, axes_perm, mappings, None)
        return data, codomain, domain

    def qr(self, a: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        q_blocks = []
        q_block_inds = []
        r_blocks = []
        r_block_inds = []
        n = 0  # running index, indicating we have already processed a_blocks[:n]
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        for i_new, (i_cod, i_dom) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            q_block_inds.append([i_cod, i_new])
            if bi_cod == i_cod:
                q, r = self.block_backend.matrix_qr(a_blocks[n], full=False)
                q_blocks.append(q)
                r_blocks.append(r)
                r_block_inds.append([i_new, i_dom])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is no block for that sector. => r=0, no need to set it.
                # choose basis vectors for q as standard basis vectors (cols/rows of eye)
                B_cod = a.codomain.multiplicities[i_cod]
                B_new = new_leg.multiplicities[i_new]
                q_blocks.append(self.block_backend.eye_matrix(B_cod, a.dtype)[:, :B_new])
        if len(q_block_inds) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds)
        if len(r_block_inds) == 0:
            r_block_inds = np.zeros((0, 2), int)
        else:
            r_block_inds = np.array(r_block_inds)
        q_data = FusionTreeData(q_block_inds, q_blocks, a.dtype)
        r_data = FusionTreeData(r_block_inds, r_blocks, a.dtype)
        return q_data, r_data

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        numbers = []
        blocks = tensor.data.blocks
        block_inds = tensor.data.block_inds
        n = 0
        bi = -1 if n >= len(block_inds) else block_inds[n, 0]
        for i in range(tensor.codomain.num_sectors):
            if i == bi:
                block = blocks[n]
                n += 1
                bi = -1 if n >= len(block_inds) else block_inds[n, 0]
            else:
                block = self.block_backend.zero_block([tensor.codomain.multiplicities[n]],
                                                      dtype=tensor.dtype)
            numbers.append(block_func(block))
        return func(numbers)
        
    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        in_domain, co_domain_idx, leg_idx = a._parse_leg_idx(leg)
        ax_a = int(in_domain)  # 1 if in_domain, 0 else
        
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds

        if (in_domain and a.domain.num_spaces == 1) or (not in_domain and a.codomain.num_spaces == 1):
            # special case where it is essentially compose.
            
            blocks = []
            block_inds = []

            if len(a_block_inds) > 0 and  len(b_block_inds) > 0:
                for n_a, n_b in iter_common_sorted(a_block_inds[:, ax_a], b_block_inds[:, 1 - ax_a]):
                    blocks.append(self.block_backend.block_scale_axis(a_blocks[n_a], b_blocks[n_b], axis=ax_a))
                    if in_domain:
                        block_inds.append([a_block_inds[n_a, 0], b_block_inds[n_b, 1]])
                    else:
                        block_inds.append([b_block_inds[n_b, 0], a_block_inds[n_a, 0]])
            if len(block_inds) == 0:
                block_inds = np.zeros((0, 2), int)
            else:
                block_inds = np.array(block_inds, int)
            return FusionTreeData(block_inds, blocks, a.dtype)

        blocks = []
        block_inds = np.zeros((0, 2), int)
        # potential coupled sectors
        coupled_sectors = np.array([a.codomain.sectors[ind[0]] for ind in a_block_inds])
        ind_mapping = {}  # mapping between index in coupled sectors and index in blocks
        iter_space = [a.codomain, a.domain][ax_a]
        for uncoupled, slc, coupled_ind in _forest_block_iter_product_space(iter_space, coupled_sectors, a.symmetry):
            ind = a.domain.sectors_where(coupled_sectors[coupled_ind])
            ind_b = b.data.block_ind_from_domain_sector_ind(b.domain.sectors_where(uncoupled[co_domain_idx]))
            if ind_b == None:  # zero block
                continue

            if not ind in block_inds[:, 1]:
                ind_mapping[coupled_ind] = len(blocks)
                block_inds = np.append(block_inds, np.array([[a.codomain.sectors_where(a.domain.sectors[ind]), ind]]), axis=0)
                shape = self.block_backend.block_shape(a_blocks[coupled_ind])
                blocks.append(self.block_backend.zero_block(shape, a.dtype))

            reshape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(uncoupled)]
            if in_domain:
                forest = a_blocks[coupled_ind][:, slc]
                initial_shape = self.block_backend.block_shape(forest)
                # add -1 for reshaping to take care of multiple trees within the same forest
                forest = self.block_backend.block_reshape(forest, (initial_shape[0], -1, *reshape))
                slcs = [slice(initial_shape[0]), slc]
            else:
                forest = a_blocks[coupled_ind][slc, :]
                initial_shape = self.block_backend.block_shape(forest)
                forest = self.block_backend.block_reshape(forest, (-1, *reshape, initial_shape[1]))
                slcs = [slc, slice(initial_shape[1])]
                
            # + 1 for axis comes from adding -1 to the reshaping
            forest = self.block_backend.block_scale_axis(forest, b_blocks[ind_b], axis=ax_a+co_domain_idx+1)
            forest = self.block_backend.block_reshape(forest, initial_shape)
            blocks[ind_mapping[coupled_ind]][slcs[0], slcs[1]] = forest
        return FusionTreeData(block_inds, blocks, a.dtype)

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int],
                   final_legs: list[Space]) -> Data:
        # TODO do we need metadata to split, like in abelian?
        raise NotImplementedError('split_legs not implemented')  # TODO

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        return a.data

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def svd(self, a: SymmetricTensor, new_leg: ElementarySpace, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        u_blocks = []
        s_blocks = []
        vh_blocks = []
        u_block_inds = []
        s_block_inds = []
        vh_block_inds = []
        #
        n = 0
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        for i_new, (i_cod, i_dom) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            u_block_inds.append([i_cod, i_new])
            vh_block_inds.append([i_new, i_dom])
            if bi_cod == i_cod:
                u, s, vh = self.block_backend.matrix_svd(a_blocks[n], algorithm=algorithm)
                u_blocks.append(u)
                s_blocks.append(s)
                vh_blocks.append(vh)
                s_block_inds.append([i_new, i_new])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is no block for that sector. => s=0, no need to set it.
                # choose basis vectors for u/vh as standard basis vectors (cols/rows of eye)
                B_cod = a.codomain.multiplicities[i_cod]
                B_dom = a.domain.multiplicities[i_dom]
                B_new = new_leg.multiplicities[i_new]
                u_blocks.append(self.block_backend.eye_matrix(B_cod, a.dtype)[:, :B_new])
                vh_blocks.append(self.block_backend.eye_matrix(B_dom, a.dtype)[:B_new, :])
        if len(u_block_inds) == 0:
            u_block_inds = np.zeros((0, 2), int)
        else:
            u_block_inds = np.array(u_block_inds, int)
        if len(s_block_inds) == 0:
            s_block_inds = np.zeros((0, 2), int)
        else:
            s_block_inds = np.array(s_block_inds, int)
        if len(vh_block_inds) == 0:
            vh_block_inds = np.zeros((0, 2), int)
        else:
            vh_block_inds = np.array(vh_block_inds, int)
        u_data = FusionTreeData(u_block_inds, u_blocks, a.dtype)
        s_data = FusionTreeData(s_block_inds, s_blocks, a.dtype.to_real)
        vh_data = FusionTreeData(vh_block_inds, vh_blocks, a.dtype)
        return u_data, s_data, vh_data

    def state_tensor_product(self, state1: Block, state2: Block, prod_space: ProductSpace):
        #TODO clearly define what this should do in tensors.py first!
        raise NotImplementedError

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        assert a.symmetry.can_be_dropped
        J = len(a.codomain.spaces)
        K = len(a.domain.spaces)
        num_legs = J + K
        dtype = Dtype.common(a.data.dtype, a.symmetry.fusion_tensor_dtype)
        sym = a.symmetry
        # build in internal basis order first, then apply permutations in the end
        # build in codomain/domain leg order first, then permute legs in the end
        # [i1,...,iJ,j1,...,jK]
        shape = [leg.dim for leg in a.codomain.spaces] + [leg.dim for leg in a.domain.spaces]
        res = self.block_backend.zero_block(shape, dtype)
        for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            coupled = a.codomain.sectors[bi_cod]
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(a.domain.spaces, sym):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = tree_block_size(a.domain, b_sectors)
                for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(a.codomain.spaces, sym):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = tree_block_size(a.codomain, a_sectors)
                    entries, num_alpha_trees, num_beta_trees = self._get_forest_block_contribution(
                        block, sym, a.codomain, a.domain, coupled, a_sectors, b_sectors,
                        a_dims, b_dims, tree_block_width, tree_block_height, i1, i2, m_dims, n_dims,
                        dtype
                    )
                    forest_b_height = num_alpha_trees * tree_block_height
                    forest_b_width = num_beta_trees * tree_block_width
                    if forest_b_height == 0 or forest_b_width == 0:
                        continue
                    # entries : [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
                    # permute to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                    perm = [i + offset for i in range(num_legs) for offset in [0, num_legs]]
                    entries = self.block_backend.block_permute_axes(entries, perm)
                    # reshape to [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    shape = [d_a * m for d_a, m in zip(a_dims, m_dims)] \
                            + [d_b * n for d_b, n in zip(b_dims, n_dims)]
                    entries = self.block_backend.block_reshape(entries, shape)
                    res[(*j1, *j2)] += entries
                    i1 += forest_b_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_b_width  # move right by one forest-block
        # permute leg order [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1]
        res = self.block_backend.block_permute_axes(res, [*range(J), *reversed(range(J, J + K))])
        return res

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('to_dense_block_trivial_sector not implemented')  # TODO

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> FusionTreeData:
        blocks = [self.block_backend.block_to_dtype(block, dtype) for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, dtype)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        return sum(
            (a.codomain.sector_qdims[bi_cod] * self.block_backend.block_trace_full(block)
             for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks)),
            a.dtype.zero_scalar
        )

    def transpose(self, a: SymmetricTensor) -> tuple[Data, ProductSpace, ProductSpace]:
        # Juthos implementation:
        # tensors: https://github.com/Jutho/TensorKit.jl/blob/b026cf2c1d470c6df1788a8f742c20acca67db83/src/tensors/indexmanipulations.jl#L143
        # trees: https://github.com/Jutho/TensorKit.jl/blob/b026cf2c1d470c6df1788a8f742c20acca67db83/src/fusiontrees/manipulations.jl#L524
        raise NotImplementedError('transpose not implemented')  # TODO

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype
                  ) -> FusionTreeData:
        return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=dtype)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=dtype)

    def zero_mask_data(self, large_leg: Space) -> MaskData:
        return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=Dtype.bool)

    # INTERNAL FUNCTIONS

    def _get_forest_block_contribution(self, block, sym: Symmetry, codomain, domain, coupled,
                                       a_sectors, b_sectors, a_dims, b_dims, tree_block_width,
                                       tree_block_height, i1_init, i2_init, m_dims, n_dims,
                                       dtype):
        """Helper function for :meth:`to_dense_block`.

        Obtain the contributions from a given forest block

        Parameters:
            block: The current block
            sym: The symmetry
            codomain, domain: The codomain and domain of the new tensor
            coupled, dim_c: The coupled sector of the current block and its quantum dimension
            a_sectors: The codomain uncoupled sectors [a1, a2, ..., aJ]
            b_sectors: The domain uncoupled sectors [b1, b2, ..., bK]
            tree_block_width: Equal to ``tree_block_size(domain, b_sectors)``
            tree_block_height: Equal to ``tree_block_size(codomain, a_sectors)``
            i1_init, i2_init: The start indices of the current forest block within the block

        Returns:
            entries: The entries of the dense block corresponding to the given uncoupled sectors.
                     Legs [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
            num_alpha_trees: The number of fusion trees from ``a_sectors`` to ``coupled``
            num_beta_trees : The number of fusion trees from ``b_sectors`` to ``coupled``
        """
        # OPTIMIZE do one loop per vertex in the tree instead.
        i1 = i1_init  # i1: start row index of the current tree block within the block
        i2 = i2_init  # i2: start column index of the current tree block within the block
        alpha_tree_iter = fusion_trees(sym, a_sectors, coupled, [sp.is_dual for sp in codomain.spaces])
        beta_tree_iter = fusion_trees(sym, b_sectors, coupled, [sp.is_dual for sp in domain.spaces])
        entries = self.block_backend.zero_block([*a_dims, *b_dims, *m_dims, *n_dims], dtype)
        for alpha_tree in alpha_tree_iter:
            Y = self.block_backend.block_conj(alpha_tree.as_block(backend=self))  # [a1,...,aJ,c]
            for beta_tree in beta_tree_iter:
                X = beta_tree.as_block(backend=self)  # [b1,...,bK,c]
                symmetry_data = self.block_backend.block_tdot(Y, X, -1, -1)  # [a1,...,aJ,b1,...,bK]
                idx1 = slice(i1, i1 + tree_block_height)
                idx2 = slice(i2, i2 + tree_block_width)
                degeneracy_data = block[idx1, idx2]  # [M, N]
                # [M, N] -> [m1,...,mJ,n1,...,nK]
                degeneracy_data = self.block_backend.block_reshape(degeneracy_data, m_dims + n_dims)
                entries += self.block_backend.block_outer(symmetry_data, degeneracy_data)  # [{aj} {bk} {mj} {nk}]
                i2 += tree_block_width
            i2 = i2_init  # reset to the left of the current forest-block
            i1 += tree_block_height
        num_alpha_trees = len(alpha_tree_iter)  # OPTIMIZE count loop iterations above instead?
                                                #          (same in _add_forest_block_entries)
        num_beta_trees = len(beta_tree_iter)
        return entries, num_alpha_trees, num_beta_trees

    def _add_forest_block_entries(self, block, entries, sym: Symmetry, codomain, domain, coupled,
                                dim_c, a_sectors, b_sectors, tree_block_width, tree_block_height,
                                i1_init, i2_init):
        """Helper function for :meth:`from_dense_block`.

        Adds the entries from a single forest-block to the current `block`, in place.

        Parameters:
            block: The block to modify
            entries: The entries of the dense block corresponding to the given uncoupled sectors.
                     Legs [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
            sym: The symmetry
            codomain, domain: The codomain and domain of the new tensor
            coupled, dim_c: The coupled sector of the current block and its quantum dimension
            a_sectors: The codomain uncoupled sectors [a1, a2, ..., aJ]
            b_sectors: The domain uncoupled sectors [b1, b2, ..., bK]
            tree_block_width: Equal to ``tree_block_size(domain, b_sectors)``
            tree_block_height: Equal to ``tree_block_size(codomain, a_sectors)``
            i1_init, i2_init: The start indices of the current forest block within the block

        Returns:
            num_alpha_trees: The number of fusion trees from ``a_sectors`` to ``coupled``
            num_beta_trees : The number of fusion trees from ``b_sectors`` to ``coupled``
        """
        # OPTIMIZE do one loop per vertex in the tree instead.
        i1 = i1_init  # i1: start row index of the current tree block within the block
        i2 = i2_init  # i2: start column index of the current tree block within the block
        domain_are_dual = [sp.is_dual for sp in domain.spaces]
        codomain_are_dual = [sp.is_dual for sp in codomain.spaces]
        J = len(codomain.spaces)
        K = len(domain.spaces)
        range_J = list(range(J))  # used in tdot calls below
        range_K = list(range(K))  # used in tdot calls below
        range_JK = list(range(J + K))
        alpha_tree_iter = fusion_trees(sym, a_sectors, coupled, codomain_are_dual)
        beta_tree_iter = fusion_trees(sym, b_sectors, coupled, domain_are_dual)
        for alpha_tree in alpha_tree_iter:
            X = alpha_tree.as_block(backend=self)
            # entries: [a1,...,aJ,b1,...,bK,m1,...,mJ,n1,...,nK]
            projected = self.block_backend.block_tdot(entries, X, range_J, range_J)  # [{bk}, {mj}, {nk}, c]
            for beta_tree in beta_tree_iter:
                Y = self.block_backend.block_conj(beta_tree.as_block(backend=self))
                projected = self.block_backend.block_tdot(projected, Y, range_K, range_K)  # [{mj}, {nk}, c, c']
                # projected onto the identity on [c, c']
                tree_block = self.block_backend.block_trace_partial(projected, [-2], [-1], range_JK) / dim_c
                # [m1,...,mJ,n1,...,nK] -> [M, N]
                ms_ns = self.block_backend.block_shape(tree_block)
                tree_block = np.reshape(tree_block, (prod(ms_ns[:J]), prod(ms_ns[J:])))
                idx1 = slice(i1, i1 + tree_block_height)
                idx2 = slice(i2, i2 + tree_block_width)
                # make sure we set in-range elements! otherwise item assignment silently does nothing.
                assert 0 <= idx1.start < idx1.stop <= block.shape[0]
                assert 0 <= idx2.start < idx2.stop <= block.shape[1]
                block[idx1, idx2] = tree_block
                i2 += tree_block_width  # move right by one tree-block
            i2 = i2_init  # reset to the left of the current forest-block
            i1 += tree_block_height  # move down by one tree-block (we reset to the left at start of the loop)
        num_alpha_trees = len(alpha_tree_iter)  # OPTIMIZE count loop iterations above instead?
        num_beta_trees = len(beta_tree_iter)
        return num_alpha_trees, num_beta_trees

    # METHODS FOR COMPUTING THE ACTION OF B AND C SYMBOLS

    def _add_to_mapping_dict(self, mapping: dict, trees_i: tuple[FusionTree],
                             trees_f: tuple[FusionTree], amplitude: float | complex) -> None:
        """Add the contribution that maps `tree_i` to `tree_f` with amplitude `amplitude`
        to the mapping dict `mapping`.
        """
        if trees_i in mapping:
            if trees_f in mapping[trees_i]:
                mapping[trees_i][trees_f] += amplitude
            else:
                mapping[trees_i][trees_f] = amplitude
        else:
            mapping[trees_i] = {trees_f: amplitude}

    def _add_prodspace_to_mapping_dict(self, mapping: dict, prodspace: ProductSpace,
                                       coupled: SectorArray, index: int) -> dict:
        """Add the product space `prodspace` to the mapping dict `mapping`. The new
        mapping dict's key are now tuples with one additional entry corresponding to
        fusion trees in `prodspace` with coupled sector in `coupled`. The new product
        space does not affect the amplitudes in the mapping dict. `index` specifies
        the position of the new trees within the new keys.

        This function can be used to translate mapping dicts obtained for c symbols
        to the level of both codomain and domain such that it can be combined with
        b symbols.
        """
        def new_key(key, tree):
            newkey = list(key)
            newkey.insert(index, tree)
            return tuple(newkey)

        new_mapping = {}
        for tree, _, _ in _tree_block_iter_product_space(prodspace, coupled, prodspace.symmetry):
            for key in mapping:
                if not np.all(tree.coupled == key[0].coupled):
                    continue
                new_value = {new_key(key2, tree): value for (key2, value) in mapping[key].items()}
                new_mapping[new_key(key, tree)] = new_value
        return new_mapping

    def _combine_mapping_dicts(self, dict1: dict, dict2: dict) -> dict:
        """Return a dict corresponding to the mapping associated with first applying
        dict1 and then dict2.
        """
        comb = {}
        for key1 in dict1:
            comb[key1] = {}
            for key2 in dict1[key1]:
                for key3 in dict2[key2]:
                    if key3 in comb[key1]:
                        comb[key1][key3] += dict1[key1][key2] * dict2[key2][key3]
                    else:
                        comb[key1][key3] = dict1[key1][key2] * dict2[key2][key3]
        return comb

    def _combine_multiple_mapping_dicts(self, dicts: list[dict]) -> dict:
        """Return a dict corresponding to the mapping associated with applying all
        dicts in the order given by the list. It is assumed that the all keys have
        the same format (i.e., all either specified by a single or by two fusion
        diagrams).
        """
        comb = dicts[0]
        for mapping in dicts[1:]:
            comb = self._combine_mapping_dicts(comb, mapping)
        return comb

    def _single_c_symbol_as_mapping(self, prodspace: ProductSpace, coupled: SectorArray,
                                    index: int, overbraid: bool, in_domain: bool) -> dict:
        """Return a dictionary including the details on how to combine the old fusion trees
        in `prodspace` in order to obtain the new ones after braiding. The braided spaces
        correspond to `index` and `index+1`;  the counting is from left to right (standard)
        in the codomain and from right to left (reverse) in the domain (if `in_domain ==
        True`). If `overbraid == True`, the space corresponding to `index` is above the one
        corresponding to `index+1`. The coupled sectors `coupled` correspond to the coupled
        sectors of interest (= sectors with non-zero blocks of the tensor). Contributions
        smaller than `self.eps` are discarded.
        """
        symmetry = prodspace.symmetry
        if in_domain:
            index = prodspace.num_spaces - 2 - index
            overbraid = not overbraid

        mapping = {}
        for tree, _, _ in _tree_block_iter_product_space(prodspace, coupled, symmetry):
            unc, inn, mul = tree.uncoupled, tree.inner_sectors, tree.multiplicities
            if index == 0:
                f = tree.coupled if len(inn) == 0 else inn[0]
                if overbraid:
                    factor = symmetry._r_symbol(unc[1], unc[0], f)[mul[0]]
                else:
                    factor = symmetry._r_symbol(unc[0], unc[1], f)[mul[0]].conj()
                if in_domain:
                    factor = factor.conj()

                new_tree = tree.copy(deep=True)
                new_tree.uncoupled[:2] = new_tree.uncoupled[:2][::-1]
                new_tree.are_dual[:2] = new_tree.are_dual[:2][::-1]
                self._add_to_mapping_dict(mapping, (tree, ), (new_tree, ), factor)
            else:
                left_charge = unc[0] if index == 1 else inn[index-2]
                right_charge = tree.coupled if index == inn.shape[0] else inn[index]

                for f in symmetry.fusion_outcomes(left_charge, unc[index+1]):
                    if not symmetry.can_fuse_to(f, unc[index], right_charge):
                        continue

                    new_tree = tree.copy(deep=True)
                    new_tree.inner_sectors[index-1] = f
                    new_tree.uncoupled[index:index+2] = new_tree.uncoupled[index:index+2][::-1]
                    new_tree.are_dual[index:index+2] = new_tree.are_dual[index:index+2][::-1]

                    if overbraid:
                        factors = symmetry._c_symbol(left_charge, unc[index+1], unc[index], right_charge,
                                                     f, inn[index-1])[:, :, mul[index-1], mul[index]]
                    else:
                        factors = symmetry._c_symbol(left_charge, unc[index], unc[index+1], right_charge,
                                                     inn[index-1], f)[mul[index-1], mul[index], :, :].conj()
                    if in_domain:
                        factors = factors.conj()

                    for (kap, lam), factor in np.ndenumerate(factors):
                        if abs(factor) < self.eps:
                            continue

                        new_tree.multiplicities[index-1] = kap
                        new_tree.multiplicities[index] = lam
                        self._add_to_mapping_dict(mapping, (tree, ), (new_tree.copy(deep=True), ), factor)
        return mapping

    def _single_b_symbol_as_mapping(self, codomain: ProductSpace, domain: ProductSpace,
                                    coupled: SectorArray, bend_up: bool
                                    ) -> tuple[dict, SectorArray]:
        """Return a dictionary including the details on how to combine the old fusion
        trees in `codomain` and `domain` in order to obtain the new ones after bending
        the final leg in the codomain down (`bend_up == False`) / domain up (`bend_up
        == True`). The coupled sectors `coupled` correspond to the coupled sectors of
        interest (= sectors with non-zero blocks of the tensor). Contributions smaller
        than `self.eps` are discarded.
        """
        symmetry = codomain.symmetry
        mapping = {}
        new_coupled = []
        spaces = [codomain, domain]
        for tree1, _, _ in _tree_block_iter_product_space(spaces[bend_up], coupled, symmetry):
            if tree1.uncoupled.shape[0] == 1:
                new_trees_coupled = symmetry.trivial_sector
            else:
                new_trees_coupled = tree1.inner_sectors[-1] if tree1.inner_sectors.shape[0] > 0 else tree1.uncoupled[0]

            new_tree1 = FusionTree(symmetry, tree1.uncoupled[:-1], new_trees_coupled, tree1.are_dual[:-1],
                                   tree1.inner_sectors[:-1], tree1.multiplicities[:-1])

            if len(new_coupled) == 0 or not np.any( np.all(new_trees_coupled == new_coupled, axis=1) ):
                new_coupled.append(new_trees_coupled)

            b_sym = symmetry._b_symbol(new_trees_coupled, tree1.uncoupled[-1], tree1.coupled)
            if not bend_up:
                b_sym = b_sym.conj()
            if tree1.are_dual[-1]:
                b_sym *= symmetry.frobenius_schur(tree1.uncoupled[-1])
            mu = tree1.multiplicities[-1] if tree1.multiplicities.shape[0] > 0 else 0

            for tree2, _, _ in _tree_block_iter_product_space(spaces[not bend_up], [tree1.coupled], symmetry):
                if len(tree2.uncoupled) == 0:
                    new_unc = np.array([symmetry.dual_sector(tree1.uncoupled[-1])])
                    new_dual = np.array([not tree1.are_dual[-1]])
                    new_mul = np.array([], dtype=int)
                else:
                    new_unc = np.append(tree2.uncoupled, [symmetry.dual_sector(tree1.uncoupled[-1])], axis=0)
                    new_dual = np.append(tree2.are_dual, [not tree1.are_dual[-1]])
                    new_mul = np.append(tree2.multiplicities, [0]) if len(new_unc) > 2 else np.array([0])
                new_in = np.append(tree2.inner_sectors, [tree2.coupled], axis=0) if len(new_unc) > 2 else []
                new_tree2 = FusionTree(symmetry, new_unc, new_trees_coupled, new_dual, new_in, new_mul)

                for nu in range(b_sym.shape[1]):
                    if abs(b_sym[mu, nu]) < self.eps:
                        continue

                    # assign it only if new_tree2 has a multiplicity, i.e., more than 1 uncoupled charge
                    if len(new_tree2.uncoupled) > 1:
                        new_tree2.multiplicities[-1] = nu

                    old_trees = ([tree1, tree2][bend_up], [tree1, tree2][not bend_up])
                    new_trees = ([new_tree1, new_tree2][bend_up], [new_tree1, new_tree2][not bend_up])
                    self._add_to_mapping_dict(mapping, old_trees, new_trees, b_sym[mu, nu])
        # TODO should we lexsort new_coupled? I dont see a reason right now
        return mapping, np.array(new_coupled)

    def _find_approproiate_mapping_dict(self, codomain: ProductSpace, domain: ProductSpace,
                                        index: int, coupled: SectorArray,
                                        overbraid: bool | None, bend_up: bool | None
                                        ) -> tuple[dict, ProductSpace, ProductSpace, SectorArray]:
        """Return a mapping dict together with the new codomain, new domain and new
        coupled charges that result from a specific operation on tensors over `codomain`
        and `domain` with coupled charges `coupled`. This specific operation is the
        application of a b symbol if `index == codomain.num_spaces - 1`, in which case
        `bend_up` specifies if the final leg in the codomain is bent down or the final
        leg in the domain is bent up. For all other values of `index`, the c symbol
        exchanging the legs corresponding to `index` and `index + 1` is applied. In this
        case, `overbraid == True` means that the leg corresponding to `index` is above
        the other leg. Contributions smaller than `self.eps` are discarded.

        The outputs are designed such that they can be used again as input to compute
        the next step in a sequence of operations. Such sequences are then essentially
        specified by lists containing the corresponding values for `index`, `overbraid`
        and `bend_up`.
        """
        symmetry = codomain.symmetry
        # b symbol
        if index == codomain.num_spaces - 1:
            if bend_up:
                new_domain = ProductSpace(domain.spaces[:-1], symmetry, self)
                new_codomain = ProductSpace(codomain.spaces + [domain.spaces[-1].dual], symmetry, self)
            else:
                new_codomain = ProductSpace(codomain.spaces[:-1], symmetry, self)
                new_domain = ProductSpace(domain.spaces + [codomain.spaces[-1].dual], symmetry, self)

            mapping, new_coupled = self._single_b_symbol_as_mapping(codomain, domain, coupled, bend_up)

        # c symbol
        else:
            new_coupled = coupled
            if index > codomain.num_spaces - 1:
                in_domain = True
                new_codomain = codomain
                index_ = codomain.num_spaces + domain.num_spaces - 1 - (index + 1)
                spaces = domain.spaces[:]
                spaces[index_:index_ + 2] = spaces[index_:index_ + 2][::-1]
                new_domain = ProductSpace(spaces, symmetry, self, domain.sectors, domain.multiplicities)
                index -= codomain.num_spaces
            else:
                in_domain = False
                new_domain = domain
                spaces = codomain.spaces[:]
                spaces[index:index + 2] = spaces[index:index + 2][::-1]
                new_codomain = ProductSpace(spaces, symmetry, self, codomain.sectors, codomain.multiplicities)

            prodspace = [codomain, domain][in_domain]
            mapping = self._single_c_symbol_as_mapping(prodspace, coupled, index, overbraid, in_domain)

        return mapping, new_codomain, new_domain, new_coupled

    def _apply_mapping_dict(self, ten: SymmetricTensor, new_codomain: ProductSpace,
                            new_domain: ProductSpace, block_axes_permutation: list[int],
                            mapping: dict, in_domain: bool | None) -> FusionTreeData:
        """Apply the mapping dict `mapping` to the tensor `ten` and return the resulting
        `FusionTreeData`. `new_codomain` and `new_domain` are the codomain and domain of
        the final tensor, `block_axes_permutation` gives the permutation of the axes
        after reshaping the tree blocks such that each leg corresponds to its own axis.

        If the keys in `mapping` contain fusion trees for both codomain and domain, all
        operatrions are performed on this level (as described above); if the keys only
        contain a single fusion tree, it is assumed (but not checked) that either the
        codomain XOR domain is relevant for the mapping, the other one can be ignored.
        That is, all operations are performed on the level of rows or columns of the
        blocks rather than on the level of tree blocks. This is reflected in the inputs:
        `block_axes_permutation` should then only feature the indices for the codomain
        XOR domain and `in_domain` specifies whether the mapping is to be applied to the
        domain or codomain; there is no use for it in the other case.
        """
        backend = ten.backend.block_backend
        new_data = FusionTreeData._zero_data(new_codomain, new_domain, backend, Dtype.complex128)

        # we allow this case for more efficient treatment when only c symbols in the (co)domain are involved
        single_fusion_tree_in_keys = False
        for key in mapping:
            if len(key) == 1:
                single_fusion_tree_in_keys = True
            break

        if single_fusion_tree_in_keys:
            iter_space = [ten.codomain, ten.domain][in_domain]
            new_space = [new_codomain, new_domain][in_domain]
            old_coupled = [sec for i, sec in enumerate(ten.domain.sectors) if ten.data.block_ind_from_domain_sector_ind(i) != None]

            if in_domain:
                block_axes_permutation = [0] + [i+1 for i in block_axes_permutation]
            else:
                block_axes_permutation.append(len(block_axes_permutation))

            for tree, slc, ind in _tree_block_iter_product_space(iter_space, old_coupled, ten.symmetry):
                modified_shape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(tree.uncoupled)]
                if in_domain:
                    block_slice = ten.data.blocks[ind][:, slc]
                    modified_shape.insert(0, -1)
                else:
                    block_slice = ten.data.blocks[ind][slc, :]
                    modified_shape.append(-1)

                final_shape = backend.block_shape(block_slice)

                block_slice = backend.block_reshape(block_slice, tuple(modified_shape))
                block_slice = backend.block_permute_axes(block_slice, block_axes_permutation)
                block_slice = backend.block_reshape(block_slice, final_shape)

                contributions = mapping[(tree, )]
                for ((new_tree, ), amplitude) in contributions.items():
                    new_slc = tree_block_slice(new_space, new_tree)
                    coupled = new_tree.coupled
                    block_ind = new_domain.sectors_where(coupled)
                    block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

                    if in_domain:
                        new_data.blocks[block_ind][:, new_slc] += amplitude * block_slice
                    else:
                        new_data.blocks[block_ind][new_slc, :] += amplitude * block_slice
        
        else:
            for alpha_tree, beta_tree, tree_block in _tree_block_iter(ten):
                contributions = mapping[(alpha_tree, beta_tree)]

                # reshape tree_block
                modified_shape = [ten.codomain[i].sector_multiplicity(sec) for i, sec in enumerate(alpha_tree.uncoupled)]
                modified_shape += [ten.domain[i].sector_multiplicity(sec) for i, sec in enumerate(beta_tree.uncoupled)]
                final_shape = [modified_shape[i] for i in block_axes_permutation]
                final_shape = (prod(final_shape[:new_codomain.num_spaces]), prod(final_shape[new_codomain.num_spaces:]))

                tree_block = backend.block_reshape(tree_block, tuple(modified_shape))
                tree_block = backend.block_permute_axes(tree_block, block_axes_permutation)
                tree_block = backend.block_reshape(tree_block, final_shape)

                for ((new_alpha_tree, new_beta_tree), amplitude) in contributions.items():
                    # TODO do we want to cache the slices and / or block_inds?
                    alpha_slice = tree_block_slice(new_codomain, new_alpha_tree)
                    beta_slice = tree_block_slice(new_domain, new_beta_tree)

                    coupled = new_alpha_tree.coupled
                    block_ind = new_domain.sectors_where(coupled)
                    block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

                    new_data.blocks[block_ind][alpha_slice, beta_slice] += amplitude * tree_block
        
        new_data.discard_zero_blocks(backend, ten.backend.eps)
        return new_data

    # INEFFICIENT METHODS FOR COMPUTING THE ACTION OF B AND C SYMBOLS

    def _apply_single_c_symbol_inefficient(self, ten: SymmetricTensor, leg: int | str, levels: list[int],
                                           ) -> tuple[FusionTreeData, ProductSpace, ProductSpace]:
        """Naive and inefficient implementation of a single C symbol for test purposes.
        `ten.legs[leg]` is exchanged with `ten.legs[leg + 1]`

        NOTE this function may be deleted at a later stage
        """
        # NOTE the case of braiding in domain and codomain are treated separately despite being similar
        # This way, it is easier to understand the more compact and more efficient function
        index = ten.get_leg_idcs(leg)[0]
        backend = self.block_backend
        symmetry = ten.symmetry

        # NOTE do these checks in permute_legs for the actual (efficient) function
        # we do it here because we directly call this function in the tests
        assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
        assert index < ten.num_legs - 1

        in_domain = index > ten.num_codomain_legs - 1

        if in_domain:
            domain_index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
            new_domain = ProductSpace(ten.domain[:domain_index] + [ten.domain[domain_index + 1]]
                                      + [ten.domain[domain_index]] + ten.domain[domain_index + 2:],
                                      symmetry=symmetry, backend=self, _sectors=ten.domain.sectors,
                                      _multiplicities=ten.domain.multiplicities)
            new_codomain = ten.codomain
        else:
            new_codomain = ProductSpace(ten.codomain[:index] + [ten.codomain[index + 1]]
                                        + [ten.codomain[index]] + ten.codomain[index + 2:],
                                        symmetry=symmetry, backend=self, _sectors=ten.codomain.sectors,
                                        _multiplicities=ten.codomain.multiplicities)
            new_domain = ten.domain

        zero_blocks = [backend.zero_block(backend.block_shape(block), dtype=Dtype.complex128) for block in ten.data.blocks]
        new_data = FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype)
        shape_perm = np.arange(ten.num_legs)  # for permuting the shape of the tree blocks
        shifted_index = ten.num_codomain_legs + domain_index if in_domain else index
        shape_perm[shifted_index:shifted_index+2] = shape_perm[shifted_index:shifted_index+2][::-1]

        for alpha_tree, beta_tree, tree_block in _tree_block_iter(ten):
            block_charge = ten.domain.sectors_where(alpha_tree.coupled)
            block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

            initial_shape = backend.block_shape(tree_block)
            modified_shape = [ten.codomain[i].sector_multiplicity(sec) for i, sec in enumerate(alpha_tree.uncoupled)]
            modified_shape += [ten.domain[i].sector_multiplicity(sec) for i, sec in enumerate(beta_tree.uncoupled)]

            tree_block = backend.block_reshape(tree_block, tuple(modified_shape))
            tree_block = backend.block_permute_axes(tree_block, shape_perm)
            tree_block = backend.block_reshape(tree_block, initial_shape)

            if index == 0 or index == ten.num_legs - 2:
                if in_domain:
                    alpha_slice = tree_block_slice(new_codomain, alpha_tree)
                    b = beta_tree.copy(True)
                    b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities
                    f = b.coupled if len(b_in) == 0 else b_in[0]
                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        r = symmetry.r_symbol(b_unc[0], b_unc[1], f)[b_mul[0]]
                    else:
                        r = symmetry.r_symbol(b_unc[1], b_unc[0], f)[b_mul[0]].conj()
                    b_unc[domain_index:domain_index+2] = b_unc[domain_index:domain_index+2][::-1]
                    beta_slice = tree_block_slice(new_domain, b)
                else:
                    beta_slice = tree_block_slice(new_domain, beta_tree)
                    a = alpha_tree.copy(True)
                    a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities
                    f = a.coupled if len(a_in) == 0 else a_in[0]
                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        r = symmetry.r_symbol(a_unc[1], a_unc[0], f)[a_mul[0]]
                    else:
                        r = symmetry.r_symbol(a_unc[0], a_unc[1], f)[a_mul[0]].conj()
                    a_unc[index:index+2] = a_unc[index:index+2][::-1]
                    alpha_slice = tree_block_slice(new_codomain, a)

                new_data.blocks[block_charge][alpha_slice, beta_slice] += r * tree_block
            else:
                if in_domain:
                    alpha_slice = tree_block_slice(new_codomain, alpha_tree)

                    beta_unc, beta_in, beta_mul = beta_tree.uncoupled, beta_tree.inner_sectors, beta_tree.multiplicities

                    if domain_index == 1:
                        left_charge = beta_unc[domain_index-1]
                    else:
                        left_charge = beta_in[domain_index-2]
                    if domain_index == ten.num_domain_legs - 2:
                        right_charge = beta_tree.coupled
                    else:
                        right_charge = beta_in[domain_index]

                    for f in symmetry.fusion_outcomes(left_charge, beta_unc[domain_index+1]):
                        if not symmetry.can_fuse_to(f, beta_unc[domain_index], right_charge):
                            continue

                        if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                            cs = symmetry.c_symbol(left_charge, beta_unc[domain_index], beta_unc[domain_index+1],
                                                   right_charge, beta_in[domain_index-1], f)[
                                                   beta_mul[domain_index-1], beta_mul[domain_index], :, :]
                        else:
                            cs = symmetry.c_symbol(left_charge, beta_unc[domain_index+1], beta_unc[domain_index],
                                                   right_charge, f, beta_in[domain_index-1])[:, :,
                                                   beta_mul[domain_index-1], beta_mul[domain_index]].conj()

                        b = beta_tree.copy(True)
                        b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities

                        b_unc[domain_index:domain_index+2] = b_unc[domain_index:domain_index+2][::-1]
                        b_in[domain_index-1] = f
                        for (kap, lam), c in np.ndenumerate(cs):
                            if abs(c) < self.eps:
                                continue
                            b_mul[domain_index-1] = kap
                            b_mul[domain_index] = lam

                            beta_slice = tree_block_slice(new_domain, b)
                            new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block
                else:
                    beta_slice = tree_block_slice(new_domain, beta_tree)
                    alpha_unc, alpha_in, alpha_mul = alpha_tree.uncoupled, alpha_tree.inner_sectors, alpha_tree.multiplicities

                    left_charge = alpha_unc[0] if index == 1 else alpha_in[index-2]
                    right_charge = alpha_tree.coupled if index == ten.num_codomain_legs - 2 else alpha_in[index]

                    for f in symmetry.fusion_outcomes(left_charge, alpha_unc[index+1]):
                        if not symmetry.can_fuse_to(f, alpha_unc[index], right_charge):
                            continue

                        if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                            cs = symmetry.c_symbol(left_charge, alpha_unc[index+1], alpha_unc[index],
                                                   right_charge, f, alpha_in[index-1])[:, :,
                                                   alpha_mul[index-1], alpha_mul[index]]
                        else:
                            cs = symmetry.c_symbol(left_charge, alpha_unc[index], alpha_unc[index+1],
                                                   right_charge, alpha_in[index-1], f)[
                                                   alpha_mul[index-1], alpha_mul[index], :, :].conj()

                        a = alpha_tree.copy(True)
                        a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities
                        
                        a_unc[index:index+2] = a_unc[index:index+2][::-1]
                        a_in[index-1] = f
                        for (kap, lam), c in np.ndenumerate(cs):
                            if abs(c) < self.eps:
                                continue
                            a_mul[index-1] = kap
                            a_mul[index] = lam

                            alpha_slice = tree_block_slice(new_codomain, a)
                            new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block

        new_data.discard_zero_blocks(backend, self.eps)
        return new_data, new_codomain, new_domain

    def _apply_single_c_symbol_more_efficient(self, ten: SymmetricTensor, leg: int | str, levels: list[int]
                                              ) -> tuple[FusionTreeData, ProductSpace, ProductSpace]:
        """More efficient implementation of a single C symbol compared to `_apply_single_C_symbol_inefficient`.
        Still not the final version.
        `ten.legs[leg]` is exchanged with `ten.legs[leg + 1]`

        NOTE this function may be deleted at a later stage
        """
        index = ten.get_leg_idcs(leg)[0]
        backend = self.block_backend
        symmetry = ten.symmetry

        # NOTE do these checks in permute_legs for the actual (efficient) function
        assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
        assert index < ten.num_legs - 1

        in_domain = index > ten.num_codomain_legs - 1

        if in_domain:
            index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
            levels = levels[::-1]
            # TODO simply take old metadata?
            new_domain = ProductSpace(ten.domain[:index] + ten.domain[index:index+2][::-1] + ten.domain[index+2:],
                                      symmetry=symmetry, backend=self, _sectors=ten.domain.sectors,
                                      _multiplicities=ten.domain.multiplicities)
            new_codomain = ten.codomain
            shape_perm = np.append([0], np.arange(1, ten.num_domain_legs+1))  # for permuting the shape of the tree blocks
            shape_perm[index+1:index+3] = shape_perm[index+1:index+3][::-1]
        else:
            new_codomain = ProductSpace(ten.codomain[:index] + ten.codomain[index:index+2][::-1] + ten.codomain[index+2:],
                                        symmetry=symmetry, backend=self, _sectors=ten.codomain.sectors,
                                        _multiplicities=ten.codomain.multiplicities)
            new_domain = ten.domain
            shape_perm = np.append(np.arange(ten.num_codomain_legs), [ten.num_codomain_legs])
            shape_perm[index:index+2] = shape_perm[index:index+2][::-1]

        # TODO dtype? cast to float if initial tensor is float and result is real?
        zero_blocks = [backend.zero_block(backend.block_shape(block), dtype=Dtype.complex128) for block in ten.data.blocks]
        new_data = FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype)
        iter_space = [ten.codomain, ten.domain][in_domain]
        iter_coupled = [ten.codomain.sectors[ind[0]] for ind in ten.data.block_inds]

        for tree, slc, _ in _tree_block_iter_product_space(iter_space, iter_coupled, symmetry):
            block_charge = ten.domain.sectors_where(tree.coupled)
            block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

            tree_block = ten.data.blocks[block_charge][:,slc] if in_domain else ten.data.blocks[block_charge][slc,:]

            initial_shape = backend.block_shape(tree_block)

            modified_shape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(tree.uncoupled)]
            modified_shape.insert((not in_domain) * len(modified_shape), initial_shape[not in_domain])

            tree_block = backend.block_reshape(tree_block, tuple(modified_shape))
            tree_block = backend.block_permute_axes(tree_block, shape_perm)
            tree_block = backend.block_reshape(tree_block, initial_shape)

            if index == 0 or index == ten.num_legs - 2:
                new_tree = tree.copy(deep=True)
                _unc, _in, _mul = new_tree.uncoupled, new_tree.inner_sectors, new_tree.multiplicities
                f = new_tree.coupled if len(_in) == 0 else _in[0]

                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    r = symmetry._r_symbol(_unc[1], _unc[0], f)[_mul[0]]
                else:
                    r = symmetry._r_symbol(_unc[0], _unc[1], f)[_mul[0]].conj()

                if in_domain:
                    r = r.conj()

                _unc[index:index+2] = _unc[index:index+2][::-1]
                new_slc = tree_block_slice([new_codomain, new_domain][in_domain], new_tree)

                if in_domain:
                    new_data.blocks[block_charge][:, new_slc] += r * tree_block
                else:
                    new_data.blocks[block_charge][new_slc, :] += r * tree_block
            else:
                _unc, _in, _mul = tree.uncoupled, tree.inner_sectors, tree.multiplicities

                left_charge = _unc[0] if index == 1 else _in[index-2]
                right_charge = tree.coupled if index == _in.shape[0] else _in[index]

                for f in symmetry.fusion_outcomes(left_charge, _unc[index+1]):
                    if not symmetry.can_fuse_to(f, _unc[index], right_charge):
                        continue

                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index+1]:
                        cs = symmetry._c_symbol(left_charge, _unc[index+1], _unc[index], right_charge,
                                                f, _in[index-1])[:, :, _mul[index-1], _mul[index]]
                    else:
                        cs = symmetry._c_symbol(left_charge, _unc[index], _unc[index+1], right_charge,
                                                _in[index-1], f)[_mul[index-1], _mul[index], :, :].conj()

                    if in_domain:
                        cs = cs.conj()

                    new_tree = tree.copy(deep=True)
                    new_tree.uncoupled[index:index+2] = new_tree.uncoupled[index:index+2][::-1]
                    new_tree.inner_sectors[index-1] = f
                    for (kap, lam), c in np.ndenumerate(cs):
                        if abs(c) < self.eps:
                            continue
                        new_tree.multiplicities[index-1] = kap
                        new_tree.multiplicities[index] = lam

                        new_slc = tree_block_slice([new_codomain, new_domain][in_domain], new_tree)

                        if in_domain:
                            new_data.blocks[block_charge][:, new_slc] += c * tree_block
                        else:
                            new_data.blocks[block_charge][new_slc, :] += c * tree_block

        new_data.discard_zero_blocks(backend, self.eps)
        return new_data, new_codomain, new_domain

    def _apply_single_b_symbol(self, ten: SymmetricTensor, bend_up: bool
                               ) -> tuple[FusionTreeData, ProductSpace, ProductSpace]:
        """Naive implementation of a single B symbol for test purposes.
        If `bend_up == True`, the right-most leg in the domain is bent up, otherwise
        the right-most leg in the codomain is bent down.

        NOTE this function may be deleted at a later stage
        """
        backend = self.block_backend
        symmetry = ten.symmetry

        # NOTE do these checks in permute_legs for the actual (efficient) function
        if bend_up:
            assert ten.num_domain_legs > 0, 'There is no leg to bend in the domain!'
        else:
            assert ten.num_codomain_legs > 0, 'There is no leg to bend in the codomain!'
        assert len(ten.data.blocks) > 0, 'The given tensor has no blocks to act on!'

        spaces = [ten.codomain, ten.domain]
        space1, space2 = spaces[bend_up], spaces[not bend_up]
        new_space1 = ProductSpace(space1.spaces[:-1], symmetry, self)
        new_space2 = ProductSpace(space2.spaces + [space1.spaces[-1].dual], symmetry, self)

        new_codomain = [new_space1, new_space2][bend_up]
        new_domain = [new_space1, new_space2][not bend_up]

        new_data = FusionTreeData._zero_data(new_codomain, new_domain, backend, Dtype.complex128)

        for alpha_tree, beta_tree, tree_block in _tree_block_iter(ten):
            modified_shape = [ten.codomain[i].sector_multiplicity(sec) for i, sec in enumerate(alpha_tree.uncoupled)]
            modified_shape += [ten.domain[i].sector_multiplicity(sec) for i, sec in enumerate(beta_tree.uncoupled)]

            if bend_up:
                if beta_tree.uncoupled.shape[0] == 1:
                    coupled = symmetry.trivial_sector
                else:
                    coupled = beta_tree.inner_sectors[-1] if beta_tree.inner_sectors.shape[0] > 0 else beta_tree.uncoupled[0]
                modified_shape = (prod(modified_shape[:ten.num_codomain_legs]),
                                  prod(modified_shape[ten.num_codomain_legs:ten.num_legs-1]),
                                  modified_shape[ten.num_legs-1])
                sec_mul = ten.domain[-1].sector_multiplicity(beta_tree.uncoupled[-1])
                final_shape = (backend.block_shape(tree_block)[0] * sec_mul, backend.block_shape(tree_block)[1] // sec_mul)
            else:
                if alpha_tree.uncoupled.shape[0] == 1:
                    coupled = symmetry.trivial_sector
                else:
                    coupled = alpha_tree.inner_sectors[-1] if alpha_tree.inner_sectors.shape[0] > 0 else alpha_tree.uncoupled[0]
                modified_shape = (prod(modified_shape[:ten.num_codomain_legs-1]),
                                  modified_shape[ten.num_codomain_legs-1],
                                  prod(modified_shape[ten.num_codomain_legs:ten.num_legs]))
                sec_mul = ten.codomain[-1].sector_multiplicity(alpha_tree.uncoupled[-1])
                final_shape = (backend.block_shape(tree_block)[0] // sec_mul, backend.block_shape(tree_block)[1] * sec_mul)
            block_ind = new_domain.sectors_where(coupled)
            block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

            tree_block = backend.block_reshape(tree_block, modified_shape)
            tree_block = backend.block_permute_axes(tree_block, [0, 2, 1])
            tree_block = backend.block_reshape(tree_block, final_shape)

            trees = [alpha_tree, beta_tree]
            tree1, tree2 = trees[bend_up], trees[not bend_up]

            if tree1.uncoupled.shape[0] == 1:
                tree1_in_1 = symmetry.trivial_sector
            else:
                tree1_in_1 = tree1.inner_sectors[-1] if tree1.inner_sectors.shape[0] > 0 else tree1.uncoupled[0]

            new_tree1 = FusionTree(symmetry, tree1.uncoupled[:-1], tree1_in_1, tree1.are_dual[:-1],
                                   tree1.inner_sectors[:-1], tree1.multiplicities[:-1])
            new_tree2 = FusionTree(symmetry, np.append(tree2.uncoupled, [symmetry.dual_sector(tree1.uncoupled[-1])], axis=0),
                                   tree1_in_1, np.append(tree2.are_dual, [not tree1.are_dual[-1]]),
                                   np.append(tree2.inner_sectors, [tree2.coupled], axis=0), np.append(tree2.multiplicities, [0]))

            b_sym = symmetry._b_symbol(tree1_in_1, tree1.uncoupled[-1], tree1.coupled)
            if not bend_up:
                b_sym = b_sym.conj()
            if tree1.are_dual[-1]:
                b_sym *= symmetry.frobenius_schur(tree1.uncoupled[-1])
            mu = tree1.multiplicities[-1] if tree1.multiplicities.shape[0] > 0 else 0
            for nu in range(b_sym.shape[1]):
                if abs(b_sym[mu, nu]) < self.eps:
                    continue
                new_tree2.multiplicities[-1] = nu

                if bend_up:
                    alpha_slice = tree_block_slice(new_codomain, new_tree2)
                    if new_tree1.uncoupled.shape[0] == 0:
                        beta_slice = slice(0, 1)
                    else:
                        beta_slice = tree_block_slice(new_domain, new_tree1)
                else:
                    if new_tree1.uncoupled.shape[0] == 0:
                        alpha_slice = slice(0, 1)
                    else:
                        alpha_slice = tree_block_slice(new_codomain, new_tree1)
                    beta_slice = tree_block_slice(new_domain, new_tree2)

                new_data.blocks[block_ind][alpha_slice, beta_slice] += b_sym[mu, nu] * tree_block

        # TODO dtype? cast to float if initial tensor is float and result is real?
        new_data.discard_zero_blocks(backend, self.eps)
        return new_data, new_codomain, new_domain