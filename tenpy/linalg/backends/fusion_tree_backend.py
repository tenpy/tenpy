# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np
from itertools import product

from .abstract_backend import (
    TensorBackend, Block, Data, DiagonalData, MaskData
)
from ..dtypes import Dtype
from ..symmetries import Sector, SectorArray, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace
from ..trees import FusionTree, fusion_trees
from ...tools.misc import iter_common_sorted_arrays, iter_common_noncommon_sorted, iter_common_sorted

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
    offset += fusion_trees(space.symmetry, tree.uncoupled, tree.coupled).index(tree)
    size = tree_block_size(space, tree.uncoupled)
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
                forest_block_width = i2 - i2_forest
                i1_forest += forest_block_height
            i1_forest = 0  # reset to the top of the block
            i2_forest += forest_block_width


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


# TODO do we need to inherit from ABC again?? (same in abelian and no_symmetry)
# TODO eventually remove BlockBackend inheritance, it is not needed,
#      jakob only keeps it around to make his IDE happy  (same in abelian and no_symmetry)
class FusionTreeBackend(TensorBackend):
    
    DataCls = FusionTreeData
    can_decompose_tensors = True

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
        raise NotImplementedError('diagonal_transpose not implemented')
        
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
        raise NotImplementedError('permute_legs not implemented')  # TODO

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
        in_domain, co_codomain_idx, leg_idx = a._parse_leg_idx(leg)
        ax_a = int(in_domain)  # 1 if in_domain, 0 else
        
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds

        if (in_domain and a.domain.num_spaces == 1) or (not in_domain and a.codomain.num_spaces == 1):
            # special case where it is essentially compose.
            
            blocks = []
            block_inds = []

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

        raise NotImplementedError('scale_axis not implemented')  # TODO

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
