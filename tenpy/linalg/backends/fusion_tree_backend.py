# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np

from ..misc import iter_common_noncommon_sorted_arrays, iter_common_sorted_arrays

from .abstract_backend import (
    Backend, BlockBackend, Block, Data, DiagonalData, MaskData
)
from ..dtypes import Dtype
from ..symmetries import Sector, SectorArray, Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace
from ..trees import FusionTree, fusion_trees

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


def _tree_block_iter(a: SymmetricTensor, backend: BlockBackend):
    sym = a.symmetry
    domain_are_dual = [sp.is_dual for sp in a.domain.spaces]
    codomain_are_dual = [sp.is_dual for sp in a.codomain.spaces]
    for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(a.domain.spaces, sym):
            tree_block_width = tree_block_size(a.domain, b_sectors)
            for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(a.codomain.spaces, sym):
                tree_block_height = tree_block_size(a.codomain, a_sectors)
                i1 = i1_forest  # start row index of the current tree block
                i2 = i2_forest  # start column index of the current tree block
                for alpha_tree in fusion_trees(sym, a_sectors, coupled, codomain_are_dual):
                    i2 = i2_forest  # reset to the left of the current forest block
                    for beta_tree in fusion_trees(sym, b_sectors, coupled, domain_are_dual):
                        idx1 = slice(i1, i1 + tree_block_height)
                        idx2 = slice(i2, i2 + tree_block_width)
                        entries = block[idx1, idx2]
                        entries = backend.block_reshape(entries, m_dims + n_dims)
                        yield alpha_tree, beta_tree, entries
                        i2 += tree_block_width  # move right by one tree block
                    i1 += tree_block_height  # move down by one tree block
                forest_block_height = i1 - i1_forest
                forest_block_width = i2 - i2_forest
                i1_forest += forest_block_height
            i1_forest = 0  # reset to the top of the block
            i2_forest += forest_block_width


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
    coupled_sectors : 2D array
        The coupled sectors :math:`c_n` for which there are non-zero blocks.
        Must be ``lexsort( .T)``-ed (this is not checked!).
        OPTIMIZE force 2D array or allow list of 1D array?
    blocks : list of 2D Block
        The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
    """
    def __init__(self, coupled_sectors: SectorArray, blocks: list[Block], dtype: Dtype):
        assert coupled_sectors.ndim == 2
        self.coupled_sectors = coupled_sectors
        self.blocks = blocks
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


# TODO do we need to inherit from ABC again?? (same in abelian and no_symmetry)
# TODO eventually remove BlockBackend inheritance, it is not needed,
#      jakob only keeps it around to make his IDE happy  (same in abelian and no_symmetry)
class FusionTreeBackend(Backend, BlockBackend, metaclass=ABCMeta):
    
    DataCls = FusionTreeData
    can_decompose_tensors = True

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # coupled sectors must be lexsorted
        perm = np.lexsort(a.data.coupled_sectors.T)
        assert np.all(perm == np.arange(len(perm)))
        # blocks
        assert len(a.data.coupled_sectors) == len(a.data.blocks)
        for c, block in zip(a.data.coupled_sectors, a.data.blocks):
            assert a.symmetry.is_valid_sector(c)
            expect_shape = (block_size(a.codomain, c), block_size(a.domain, c))
            if is_diagonal:
                assert expect_shape[0] == expect_shape[1]
                expect_shape = (expect_shape[0],)
            assert all(dim > 0 for dim in expect_shape), 'should skip forbidden block'
            self.test_block_sanity(block, expect_shape=expect_shape, expect_dtype=a.dtype)

    def test_mask_sanity(self, a: Mask):
        raise NotImplementedError  # TODO

    # TODO do we need leg metadata?
    #  related methods:
    #   - test_leg_sanity
    #   - _fuse_spaces

    # ABSTRACT METHODS

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block]) -> Data:
        raise NotImplementedError('act_block_diagonal_square_matrix not implemented')  # TODO

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        raise NotImplementedError('add_trivial_leg not implemented')  # TODO

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float
                     ) -> bool:
        for i, j in iter_common_noncommon_sorted_arrays(a.data.coupled_sectors, b.data.coupled_sectors):
            if j is None:
                if self.block_max_abs(a.data.blocks[i]) > atol:
                    return False
            if i is None:
                if self.block_max_abs(b.data.blocks[j]) > atol:
                    return False
            else:
                if not self.block_allclose(a.data.blocks[i], b.data.blocks[j], rtol=rtol, atol=atol):
                    return False
        return True

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        raise NotImplementedError('apply_mask_to_DiagonalTensor not implemented')  # TODO
    
    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     product_spaces: list[ProductSpace],
                     new_codomain_combine: list[tuple[list[int], ProductSpace]],
                     new_domain_combine: list[tuple[list[int], ProductSpace]],
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
        b_blocks = b.data.blocks
        if a.dtype != res_dtype:
            a_blocks = [self.block_to_dtype(bl, res_dtype) for bl in a_blocks]
        if b.dtype != res_dtype:
            b_blocks = [self.block_to_dtype(bl, res_dtype) for bl in b_blocks]
        blocks = []
        coupled = []
        a_coupled = a.data.coupled_sectors
        for i, j in iter_common_sorted_arrays(a_coupled, b.data.coupled_sectors):
            blocks.append(self.matrix_dot(a_blocks[i], b_blocks[j]))
            coupled.append(a_coupled[i])
        if len(coupled) == 0:
            coupled = a.symmetry.empty_sector_array
        else:
            coupled = np.array(coupled, int)
        return FusionTreeData(coupled, blocks, res_dtype)

    def copy_data(self, a: SymmetricTensor) -> FusionTreeData:
        return FusionTreeData(
            coupled_sectors=a.data.coupled_sectors.copy(),  # OPTIMIZE do we need to copy these?
            blocks=[self.block_copy(block) for block in a.data.blocks],
            dtype=a.data.dtype
        )

    def dagger(self, a: SymmetricTensor) -> Data:
        return FusionTreeData(
            coupled_sectors=a.data.coupled_sectors,
            blocks=[self.block_dagger(b) for b in a.data.blocks],
            dtype=a.dtype
        )

    def data_item(self, a: FusionTreeData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_item(a.blocks[0])

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        if len(a.data.blocks) < a.domain.num_sectors:
            # there are missing blocks. -> they contain False -> all(a) == False
            return False
        # now it is enough to check the existing blocks
        return all(self.block_all(b) for b in a.data.blocks)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return any(self.block_any(b) for b in a.data.blocks)
    
    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool) -> DiagonalData:
        a_coupled = a.data.coupled_sectors
        b_coupled = b.data.coupled_sectors
        if partial_zero_is_zero:
            blocks = []
            coupled_sectors = []
            for i, j in iter_common_sorted_arrays(a_coupled, b_coupled):
                coupled = a_coupled[i]
                coupled_sectors.append(coupled)
                blocks.append(func(a.data.blocks[i], b.data.blocks[j], **func_kwargs))
        else:
            i_a = 0  # during the loop: a_coupled[:i_a] was already visited
            ac_ia = None if len(a_coupled) == 0 else a_coupled[0]
            i_b = 0  # same for b_coupled
            bc_ib = None if len(b_coupled) == 0 else b_coupled[0]
            coupled_sectors = a.domain.sectors
            blocks = []
            for coupled in coupled_sectors:
                if ac_ia is not None and np.all(coupled == ac_ia):
                    a_block = a.data.blocks[i_a]
                    i_a += 1
                    ac_ia = None if i_a >= len(a_coupled) else a_coupled[i_a]
                else:
                    a_block = self.zero_block([block_size(a.domain, coupled)], dtype=a.dtype)
                if bc_ib is not None and np.all(coupled == b_coupled[i_b]):
                    b_block = b.data.blocks[i_b]
                    i_b += 1
                    bc_ib = None if i_b >= len(b_coupled) else b_coupled[i_b]
                else:
                    b_block = self.zero_block([block_size(a.domain, coupled)], dtype=b.dtype)
                blocks.append(func(a_block, b_block, **func_kwargs))
        if len(blocks) > 0:
            dtype = self.block_dtype(blocks[0])
        else:
            a_block = self.ones_block([1], dtype=a.dtype)
            b_block = self.ones_block([1], dtype=b.dtype)
            example_block = func(a_block, b_block, **func_kwargs)
            dtype = self.block_dtype(example_block)
        return FusionTreeData(coupled_sectors=coupled_sectors, blocks=blocks, dtype=dtype)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs,
                                   maps_zero_to_zero: bool) -> DiagonalData:
        if maps_zero_to_zero:
            blocks = [func(b, **func_kwargs) for b in a.data.blocks]
            coupled_sectors = a.data.coupled_sectors
        else:
            coupled_sectors = a.domain.sectors
            blocks = []
            for i, j in iter_common_noncommon_sorted_arrays(coupled_sectors, a.data.coupled_sectors):
                if j is None:
                    block = self.zero_block([block_size(a.domain, coupled_sectors[i])], dtype=a.dtype)
                else:
                    block = a.data.blocks[j]
                blocks.append(func(block, **func_kwargs))
        if len(blocks) > 0:
            dtype = self.block_dtype(blocks[0])
        else:
            dtype = self.block_dtype(func(self.ones_block([1], dtype=a.dtype), **func_kwargs))
        return FusionTreeData(coupled_sectors=coupled_sectors, blocks=blocks, dtype=dtype)

    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        dtype = self.block_dtype(a)
        coupled_sectors = co_domain.sectors
        blocks = []
        for coupled, mult, slc in zip(co_domain.sectors, co_domain.multiplicities, co_domain.slices):
            dim_c = co_domain.symmetry.sector_dim(coupled)
            entries = self.block_reshape(a[slice(*slc)], (dim_c, mult))
            # project onto the identity on the coupled sector
            block = self.block_sum(entries, 0) / dim_c
            projected = self.block_outer(self.ones_block([dim_c], dtype=dtype), block)
            if self.block_norm(entries - projected) > tol * self.block_norm(entries):
                raise ValueError('Block is not symmetric up to tolerance.')
            blocks.append(block)
        return FusionTreeData(coupled_sectors, blocks, dtype)

    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        coupled_sectors = co_domain.sectors
        blocks = [func((block_size(co_domain, coupled),), coupled) for coupled in coupled_sectors]
        if len(blocks) > 0:
            sample_block = blocks[0]
            coupled_sectors = np.asarray(coupled_sectors, int)
        else:
            sample_block = func((1,), co_domain.symmetry.trivial_sector)
            coupled_sectors = co_domain.symmetry.empty_sector_array
        dtype = self.block_dtype(sample_block)
        return FusionTreeData(coupled_sectors, blocks, dtype)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                       ) -> DiagonalData:
        raise NotImplementedError('diagonal_tensor_from_full_tensor not implemented')  # TODO

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        res = a.dtype.zero_scalar
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
            d_c = a.symmetry.qdim(coupled)
            res += d_c * self.block_sum_all(block)
        return res

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        assert a.symmetry.can_be_dropped
        res = self.zero_block([a.leg.dim], a.dtype)
        for i, j in iter_common_sorted_arrays(a.leg.sectors, a.data.coupled_sectors):
            coupled = a.leg.sectors[i]
            symmetry_data = self.ones_block([a.symmetry.sector_dim(coupled)], dtype=a.dtype)
            degeneracy_data = a.data.blocks[j]
            entries = self.block_outer(symmetry_data, degeneracy_data)
            entries = self.block_reshape(entries, (-1,))
            res[slice(*a.leg.slices[i])] = entries
        return res

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[DiagonalData, ElementarySpace]:
        raise NotImplementedError('diagonal_to_mask not implemented')
    
    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        raise NotImplementedError('diagonal_transpose not implemented')
        
    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError('eigh not implemented')  # TODO

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> FusionTreeData:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        coupled_sectors = co_domain.sectors
        blocks = [self.eye_matrix(block_size(co_domain, c), dtype) for c in coupled_sectors]
        return FusionTreeData(coupled_sectors, blocks, dtype)

    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> FusionTreeData:
        sym = codomain.symmetry
        assert sym.can_be_dropped
        # convert to internal basis order, where the sectors are sorted and contiguous
        J = len(codomain.spaces)
        K = len(domain.spaces)
        num_legs = J + K
        # [i1,...,iJ,jK,...,j1] -> [i1,...,iJ,j1,...,jK]
        a = self.block_permute_axes(a, [*range(J), *reversed(range(J, num_legs))])
        dtype = Dtype.common(self.block_dtype(a), sym.fusion_tensor_dtype)
        # main loop: iterate over coupled sectors and construct the respective block.
        coupled_sectors = []
        blocks = []
        norm_sq_projected = 0
        for i, _ in iter_common_sorted_arrays(domain.sectors, codomain.sectors):
            coupled = domain.sectors[i]
            dim_c = sym.sector_dim(coupled)
            # OPTIMIZE could be sth like np.empty
            block = self.zero_block([block_size(codomain, coupled), block_size(domain, coupled)], dtype)
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
                    entries = self.block_reshape(entries, shape)
                    # permute to [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...nK]
                    perm = [*range(0, 2 * num_legs, 2), *range(1, 2 * num_legs, 2)]
                    entries = self.block_permute_axes(entries, perm)
                    num_alpha_trees, num_beta_trees = self._add_forest_block_entries(
                        block, entries, sym, codomain, domain, coupled, dim_c, a_sectors, b_sectors,
                        tree_block_width, tree_block_height, i1, i2
                    )
                    forest_block_height = num_alpha_trees * tree_block_height
                    forest_block_width = num_beta_trees * tree_block_width
                    i1 += forest_block_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_block_width  # move right by one forest-block
            block_norm = self.block_norm(block, order=2)
            if block_norm <= 0.:  # TODO small finite tolerance instead?
                continue
            coupled_sectors.append(coupled)
            blocks.append(block)
            contribution = dim_c * block_norm ** 2
            norm_sq_projected += contribution

        # since the symmetric and non-symmetric components of ``a = a_sym + a_rest`` are mutually
        # orthogonal, we have  ``norm(a) ** 2 = norm(a_sym) ** 2 + norm(a_rest) ** 2``.
        # thus ``abs_err = norm(a - a_sym) = norm(a_rest) = sqrt(norm(a) ** 2 - norm(a_sym) ** 2)``
        if tol is not None:
            a_norm_sq = self.block_norm(a, order=2) ** 2
            norm_diff_sq = a_norm_sq - norm_sq_projected
            abs_tol_sq = tol * tol * a_norm_sq
            if norm_diff_sq > abs_tol_sq > 0:
                msg = (f'Block is not symmetric up to tolerance. '
                       f'Original norm: {np.sqrt(a_norm_sq)}. '
                       f'Norm after projection: {np.sqrt(norm_sq_projected)}.')
                raise ValueError(msg)
        coupled_sectors = np.asarray(coupled_sectors, int)
        return FusionTreeData(coupled_sectors, blocks, dtype)
    
    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        raise NotImplementedError('from_dense_block_trivial_sector not implemented')  # TODO

    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        raise NotImplementedError  # TODO

    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> FusionTreeData:
        coupled_sectors = []
        blocks = []
        for i, _ in iter_common_sorted_arrays(domain.sectors, codomain.sectors):
            coupled = domain.sectors[i]
            shape = (block_size(codomain, coupled), block_size(domain, coupled))
            coupled_sectors.append(coupled)
            blocks.append(func(shape, coupled))
        if len(blocks) > 0:
            sample_block = blocks[0]
            coupled_sectors = np.asarray(coupled_sectors, int)
        else:
            sample_block = func((1, 1), codomain.symmetry.trivial_sector)
            coupled_sectors = domain.symmetry.empty_sector_array
        dtype = self.block_dtype(sample_block)
        return FusionTreeData(coupled_sectors, blocks, dtype)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_from_diagonal(block) for block in a.data.blocks]
        return FusionTreeData(a.data.coupled_sectors, blocks, dtype=a.dtype)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        raise NotImplementedError('full_data_from_mask not implemented')  # TODO

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        raise NotImplementedError('get_element not implemented')  # TODO

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        raise NotImplementedError('get_element_diagonal not implemented')  # TODO

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        res = 0.
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_coupled = a.data.coupled_sectors
        for i, j in iter_common_sorted_arrays(a_coupled, b.data.coupled_sectors):
            d_c = a.symmetry.qdim(a_coupled[i])
            res += d_c * self.block_inner(a_blocks[i], b_blocks[j], do_dagger=do_dagger)
        return res
    
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                               charge_leg: ElementarySpace) -> Data:
        raise NotImplementedError('inv_part_from_dense_block_single_sector not implemented')  # TODO

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('inv_part_to_dense_block_single_sector not implemented')  # TODO

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        dtype = v.data.dtype.common(w.data.dtype)
        v_blocks = [self.block_to_dtype(_a, dtype) for _a in v.data.blocks]
        w_blocks = [self.block_to_dtype(_b, dtype) for _b in w.data.blocks]
        blocks = []
        coupled_sectors = []
        for i, j in iter_common_noncommon_sorted_arrays(v.data.coupled_sectors, w.data.coupled_sectors):
            if i is None:
                blocks.append(self.block_mul(b, w_blocks[j]))
                coupled_sectors.append(w.data.coupled_sectors[j])
            elif j is None:
                blocks.append(self.block_mul(a, v_blocks[i]))
                coupled_sectors.append(v.data.coupled_sectors[i])
            else:
                blocks.append(self.block_linear_combination(a, v_blocks[i], b, w_blocks[j]))
                coupled_sectors.append(v.data.coupled_sectors[i])
        if len(blocks) == 0:
            coupled_sectors = v.symmetry.empty_sector_array
        else:
            coupled_sectors = np.array(coupled_sectors)
        return FusionTreeData(coupled_sectors, blocks, dtype)
        
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
        blocks = [self.block_mul(a, T) for T in b.data.blocks]
        if len(blocks) == 0:
            if isinstance(a, float):
                dtype = b.data.dtype
            else:
                dtype = b.data.dtype.to_complex()
        else:
            dtype = self.block_dtype(blocks[0])
        return FusionTreeData(b.data.coupled_sectors, blocks, dtype)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        # OPTIMIZE should we offer the square-norm instead?
        norm_sq = 0
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
            norm_sq += a.symmetry.sector_dim(coupled) * (self.block_norm(block) ** 2)
        return self.block_sqrt(norm_sq)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        raise NotImplementedError('outer not implemented')  # TODO

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, ProductSpace, ProductSpace]:
        raise NotImplementedError('partial_trace not implemented')  # TODO

    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, ProductSpace, ProductSpace]:
        raise NotImplementedError('permute_legs not implemented')  # TODO

    def qr(self, a: SymmetricTensor, new_r_leg_dual: bool, full: bool
           ) -> tuple[Data, Data, ElementarySpace]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError('qr not implemented')  # TODO

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        numbers = []
        coupled = tensor.data.coupled_sectors
        blocks = tensor.data.blocks
        i = 0
        for c, m in zip(tensor.leg.sectors, tensor.leg.multiplicities):
            if np.all(c == coupled[i]):
                block = blocks[i]
                i += 1
            else:
                block = self.zero_block([m], dtype=tensor.dtype)
            numbers.append(block_func(block))
        return func(numbers)
        
    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        in_domain, co_codomain_idx, leg_idx = a._parse_leg_idx(leg)
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_coupled = a.data.coupled_sectors
        
        if (in_domain and a.domain.num_spaces == 1) or (not in_domain and a.codomain.num_spaces == 1):
            blocks = []
            coupled = []
            for i, j in iter_common_sorted_arrays(a_coupled, b.data.coupled_sectors):
                blocks.append(self.block_scale_axis(a_blocks[i], b_blocks[j], axis=1))
                coupled.append(a_coupled[i])
            if len(coupled) == 0:
                coupled = a.symmetry.empty_sector_array
            else:
                coupled = np.array(coupled, int)
            return FusionTreeData(coupled, blocks, a.dtype)

        raise NotImplementedError('scale_axis not implemented')  # TODO

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int],
                   final_legs: list[Space]) -> Data:
        # TODO do we need metadata to split, like in abelian?
        raise NotImplementedError('split_legs not implemented')  # TODO

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        raise NotImplementedError('squeeze_legs not implemented')  # TODO

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def svd(self, a: SymmetricTensor, new_leg: ElementarySpace, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        a_blocks = a.data.blocks
        a_coupled = a.data.coupled_sectors
        #
        u_blocks = []
        s_blocks = []
        vh_blocks = []
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            # due to the loop setup we have:
            #   a.codomain.sectors[j] == new_leg.sectors[n]
            #   a.domain.sectors[k] == new_leg.sectors[n]
            if i < len(a_coupled) and np.all(new_leg.sectors[n] == a_coupled[i]):
                # we have a block for that sector
                u, s, vh = self.matrix_svd(a_blocks[i], algorithm=algorithm)
                u_blocks.append(u)
                s_blocks.append(s)
                vh_blocks.append(vh)
                i += 1
            else: 
                # there is no block for that sector. => s=0, no need to set it.
                # choose basis vectors for u/vh as standard basis vectors (cols/rows of eye)
                codomain_block_size = a.codomain.multiplicities[j]
                domain_block_size = a.domain.multiplicities[k]
                new_leg_block_size = new_leg.multiplicities[n]
                u_blocks.append(
                    self.eye_matrix(codomain_block_size, a.dtype)[:, :new_leg_block_size]
                )
                vh_blocks.append(
                    self.eye_matrix(domain_block_size, a.dtype)[:new_leg_block_size, :]
                )
                
        u_data = FusionTreeData(new_leg.sectors, u_blocks, a.dtype)
        s_data = FusionTreeData(a_coupled, s_blocks, a.dtype.to_real)
        vh_data = FusionTreeData(new_leg.sectors, vh_blocks, a.dtype)
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
        res = self.zero_block(shape, dtype)
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
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
                    entries = self.block_permute_axes(entries, perm)
                    # reshape to [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    shape = [d_a * m for d_a, m in zip(a_dims, m_dims)] \
                            + [d_b * n for d_b, n in zip(b_dims, n_dims)]
                    entries = self.block_reshape(entries, shape)
                    res[(*j1, *j2)] += entries
                    i1 += forest_b_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_b_width  # move right by one forest-block
        # permute leg order [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1]
        res = self.block_permute_axes(res, [*range(J), *reversed(range(J, J + K))])
        return res

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('to_dense_block_trivial_sector not implemented')  # TODO

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> FusionTreeData:
        blocks = [self.block_to_dtype(block, dtype) for block in a.data.blocks]
        return FusionTreeData(a.data.coupled_sectors, blocks, dtype)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        res = a.dtype.zero_scalar
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
            d_c = a.symmetry.qdim(coupled)
            res += d_c * self.block_trace_full(block)
        return res

    def transpose(self, a: SymmetricTensor) -> tuple[Data, ProductSpace, ProductSpace]:
        # Juthos implementation:
        # tensors: https://github.com/Jutho/TensorKit.jl/blob/b026cf2c1d470c6df1788a8f742c20acca67db83/src/tensors/indexmanipulations.jl#L143
        # trees: https://github.com/Jutho/TensorKit.jl/blob/b026cf2c1d470c6df1788a8f742c20acca67db83/src/fusiontrees/manipulations.jl#L524
        raise NotImplementedError('transpose not implemented')  # TODO

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype
                  ) -> FusionTreeData:
        return FusionTreeData(coupled_sectors=codomain.symmetry.empty_sector_array, blocks=[],
                              dtype=dtype)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return FusionTreeData(coupled_sectors=co_domain.symmetry.empty_sector_array, blocks=[],
                              dtype=dtype)

    def zero_mask_data(self, large_leg: Space) -> MaskData:
        return FusionTreeData(coupled_sectors=large_leg.symmetry.empty_sector_array, blocks=[],
                              dtype=Dtype.bool)

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
        entries = self.zero_block([*a_dims, *b_dims, *m_dims, *n_dims], dtype)
        for alpha_tree in alpha_tree_iter:
            Y = self.block_conj(alpha_tree.as_block(backend=self))  # [a1,...,aJ,c]
            for beta_tree in beta_tree_iter:
                X = beta_tree.as_block(backend=self)  # [b1,...,bK,c]
                symmetry_data = self.block_tdot(Y, X, -1, -1)  # [a1,...,aJ,b1,...,bK]
                idx1 = slice(i1, i1 + tree_block_height)
                idx2 = slice(i2, i2 + tree_block_width)
                degeneracy_data = block[idx1, idx2]  # [M, N]
                # [M, N] -> [m1,...,mJ,n1,...,nK]
                degeneracy_data = self.block_reshape(degeneracy_data, m_dims + n_dims)
                entries += self.block_outer(symmetry_data, degeneracy_data)  # [{aj} {bk} {mj} {nk}]
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
            projected = self.block_tdot(entries, X, range_J, range_J)  # [{bk}, {mj}, {nk}, c]
            for beta_tree in beta_tree_iter:
                Y = self.block_conj(beta_tree.as_block(backend=self))
                projected = self.block_tdot(projected, Y, range_K, range_K)  # [{mj}, {nk}, c, c']
                # projected onto the identity on [c, c']
                tree_block = self.block_trace_partial(projected, [-2], [-1], range_JK) / dim_c
                # [m1,...,mJ,n1,...,nK] -> [M, N]
                ms_ns = self.block_shape(tree_block)
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
