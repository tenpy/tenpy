# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np

from .abstract_backend import (
    Backend, BlockBackend, Block, Data, DiagonalData, iter_common_sorted_arrays,
    iter_common_noncommon_sorted_arrays, conventional_leg_order
)
from ..dtypes import Dtype
from ..symmetries import Sector, SectorArray, Symmetry, FusionStyle
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


def _tree_block_iter(data: FusionTreeData, backend: BlockBackend):
    sym = data.domain.symmetry
    domain_are_dual = [sp.is_dual for sp in data.domain.spaces]
    codomain_are_dual = [sp.is_dual for sp in data.codomain.spaces]
    for coupled, block in zip(data.coupled_sectors, data.blocks):
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(data.domain.spaces, sym):
            tree_block_width = tree_block_size(data.domain, b_sectors)
            for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(data.codomain.spaces, sym):
                tree_block_height = tree_block_size(data.codomain, a_sectors)
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
    domain, codomain : ProductSpace
        The domain and codomain of the tensor ``T : domain -> codomain``.
        Their "outer" :attr:`ProductSpace.is_dual` must be ``False``.
        Must not be nested, i.e. their :attr:`ProductSpace.spaces` can not themselves be
        ProductSpaces.
        TODO can we support nesting or do they *need* to be flat?? -> adjust test_data_sanity
        They comprise the legs of the tensor as::

            T.legs == [W.dual for W in domain.spaces] + codomain.spaces[::-1]
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
class FusionTreeBackend(Backend, BlockBackend, metaclass=ABCMeta):
    
    DataCls = FusionTreeData

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # check domain and codomain
        assert a.data.codomain == a.codomain
        assert a.data.domain == a.domain
        # coupled sectors must be lexsorted
        perm = np.lexsort(a.data.coupled_sectors.T)
        assert np.all(perm == np.arange(len(perm)))
        # blocks
        assert len(a.data.coupled_sectors) == len(a.data.blocks)
        for c, block in zip(a.data.coupled_sectors, a.data.blocks):
            assert a.symmetry.is_valid_sector(c)
            expect_shape = (block_size(a.data.codomain, c), block_size(a.data.domain, c))
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

    def add(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        assert a.data.num_domain_legs == b.data.num_domain_legs
        dtype = a.data.dtype.common(b.data.dtype)
        a_blocks = [self.block_to_dtype(_a, dtype) for _a in a.data.blocks]
        b_blocks = [self.block_to_dtype(_b, dtype) for _b in b.data.blocks]
        blocks = []
        coupled_sectors = []
        for i, j in iter_common_noncommon_sorted_arrays(a.data.coupled_sectors, b.data.coupled_sectors):
            if i is None:
                blocks.append(b_blocks[j])
                coupled_sectors.append(b.data.coupled_sectors[j])
            elif j is None:
                blocks.append(a_blocks[i])
                coupled_sectors.append(a.data.coupled_sectors[i])
            else:
                blocks.append(self.block_add(a_blocks[i], b_blocks[j]))
                coupled_sectors.append(a.data.coupled_sectors[i])
        if len(blocks) == 0:
            coupled_sectors = a.symmetry.empty_sector_array
        else:
            coupled_sectors = np.array(coupled_sectors)
        return FusionTreeData(coupled_sectors, blocks, a.data.domain, a.data.codomain, dtype)

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

    def apply_mask_to_Tensor(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int) -> Data:
        raise NotImplementedError('apply_mask_to_Tensor not implemented')  # TODO

    def combine_legs(self, a: SymmetricTensor, combine_slices: list[int, int],
                     product_spaces: list[ProductSpace], new_axes: list[int],
                     final_legs: list[Space]) -> Data:
        raise NotImplementedError('combine_legs not implemented')  # TODO
        
    def conj(self, a: SymmetricTensor | DiagonalTensor) -> Data | DiagonalData:
        # TODO what does this even mean? transpose of dagger?
        # TODO should we offer transpose and dagger too?
        raise NotImplementedError('conj not implemented')  # TODO

    def copy_data(self, a: SymmetricTensor) -> FusionTreeData:
        return FusionTreeData(
            coupled_sectors=a.data.coupled_sectors.copy(),  # OPTIMIZE do we need to copy these?
            blocks=[self.block_copy(block) for block in a.data.blocks],
            codomain=a.data.codomain, domain=a.data.domain
        )

    def data_item(self, a: FusionTreeData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_item(a.blocks[0])

    def _data_repr_lines(self, a: SymmetricTensor, indent: str, max_width: int,
                         max_lines: int) -> list[str]:
        raise NotImplementedError  # TODO not yet reviewed
        from ..dummy_config import printoptions
        if len(a.data.blocks) == 0:
            return [f'{indent}* Data : no non-zero blocks']

        lines = []
        for alpha_tree, beta_tree, entries in _tree_block_iter(a.data, backend=self):
            # build (a_before_Z) <- (a_after_Z) <- coupled <- (b_after_Z) <- (b_before_Z)
            a_before_Z, a_after_Z, coupled = alpha_tree._str_uncoupled_coupled(
                a.symmetry, alpha_tree.uncoupled, alpha_tree.coupled, alpha_tree.are_dual
            ).split(' -> ')
            b_before_Z, b_after_Z, coupled = beta_tree._str_uncoupled_coupled(
                a.symmetry, beta_tree.uncoupled, beta_tree.coupled, beta_tree.are_dual
            ).split(' -> ')
            sectors = f'{a_after_Z} <- {coupled} <- {b_after_Z}'
            if a_before_Z != a_after_Z:
                sectors = a_before_Z + ' <- ' + sectors
            if b_before_Z != b_after_Z:
                sectors = sectors + ' <- ' + b_before_Z

            if a.symmetry.fusion_style is FusionStyle.single:
                lines.append(f'{indent}* Data for sectors {sectors}')
            elif a.symmetry.fusion_style is FusionStyle.multiple_unique:
                # dont need multiplicity labels
                a_inner = ', '.join(a.symmetry.sector_str(i) for i in alpha_tree.inner_sectors)
                b_inner = ', '.join(a.symmetry.sector_str(i) for i in beta_tree.inner_sectors)
                lines.append(f'{indent}* Data for trees {sectors}')
                lines.append(f'{indent}  with inner sectors ({a_inner}) and ({b_inner})')
            else:
                a_inner = ', '.join(a.symmetry.sector_str(i) for i in alpha_tree.inner_sectors)
                b_inner = ', '.join(a.symmetry.sector_str(i) for i in beta_tree.inner_sectors)
                a_mults = ', '.join(map(str, alpha_tree.multiplicities))
                b_mults = ', '.join(map(str, beta_tree.multiplicities))
                lines.append(f'{indent}* Data for trees {sectors}')
                lines.append(f'{indent}  with inner sectors ({a_inner}) and ({b_inner})')
                lines.append(f'{indent}  with multiplicities ({a_mults}) <- ({b_mults})')
            lines.extend(self._block_repr_lines(entries, indent=indent + printoptions.indent * ' ',
                                                max_width=max_width, max_lines=max_lines))

            if (len(lines) > max_lines) or any(len(line) > max_width for line in lines):
                # fallback to just stating number of blocks.
                num_entries = sum(prod(self.block_shape(b)) for b in a.data.blocks)
                return [
                    f'{indent}* Data: {num_entries} entries in {len(a.data.blocks)} blocks.'
                ]
        return lines

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
            i_b = 0  # same for b_coupled
            coupled_sectors = a.domain.sectors
            blocks = []
            for coupled  in coupled_sectors:
                if np.all(coupled == a_coupled[i_a]):
                    a_block = a.data.block[i_a]
                    i_a += 1
                else:
                    a_block = self.zero_block([block_size(a.domain, coupled)], dtype=a.dtype)
                if np.all(coupled == b_coupled[i_b]):
                    b_block = b.data.block[i_b]
                    i_b += 1
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
        return FusionTreeData(coupled_sectors=coupled_sectors, blocks=blocks, domain=a.data.domain,
                              codomain=a.data.codomain, dtype=dtype)

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
        return FusionTreeData(coupled_sectors=coupled_sectors, blocks=blocks,
                              domain=a.data.domain, codomain=a.data.codomain, dtype=dtype)

    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        dtype = self.block_dtype(a)
        a = self.apply_basis_perm(a, co_domain.spaces)
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
        return FusionTreeData(coupled_sectors, blocks, co_domain, co_domain, dtype)

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
        return FusionTreeData(coupled_sectors, blocks, co_domain, co_domain, dtype)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                       ) -> DiagonalData:
        raise NotImplementedError('diagonal_tensor_from_full_tensor not implemented')  # TODO

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        raise NotImplementedError('diagonal_tensor_trace_full not implemented')  # TODO

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
        res = self.apply_basis_perm(res, [a.leg], inv=True)
        return res

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[DiagonalData, ElementarySpace]:
        raise NotImplementedError

    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError('eigh not implemented')  # TODO

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> FusionTreeData:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        coupled_sectors = co_domain.sectors
        blocks = [self.eye_matrix(block_size(co_domain, c), dtype) for c in coupled_sectors]
        return FusionTreeData(coupled_sectors, blocks, co_domain, co_domain, dtype)

    def flip_leg_duality(self, tensor: SymmetricTensor, which_legs: list[int],
                         flipped_legs: list[Space], perms: list[np.ndarray]) -> Data:
        # TODO think carefully about what this means.
        raise NotImplementedError('flip_leg_duality not implemented')  # TODO
    
    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> FusionTreeData:
        sym = codomain.symmetry
        assert sym.can_be_dropped
        # convert to internal basis order, where the sectors are sorted and contiguous
        a = self.apply_basis_perm(a, conventional_leg_order(codomain, domain))
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
                    entries = a[*j1, *j2]  # [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
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
        return FusionTreeData(coupled_sectors, blocks, domain, codomain, dtype)
    
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
        return FusionTreeData(coupled_sectors, blocks, domain, codomain, dtype)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        raise NotImplementedError('full_data_from_diagonal_tensor not implemented')  # TODO

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        raise NotImplementedError('full_data_from_mask not implemented')  # TODO

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        raise NotImplementedError('get_element not implemented')  # TODO

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        raise NotImplementedError('get_element_diagonal not implemented')  # TODO

    def infer_leg(self, block: Block, legs: list[Space | None], is_dual: bool = False,
                  ) -> ElementarySpace:
        raise NotImplementedError('infer_leg not implemented')  # TODO

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_conj: bool,
              axs2: list[int] | None) -> complex:
        raise NotImplementedError('inner not implemented')  # TODO
    
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                               charge_leg: ElementarySpace) -> Data:
        raise NotImplementedError('inv_part_from_dense_block_single_sector not implemented')  # TODO

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('inv_part_to_dense_block_single_sector not implemented')  # TODO

    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        raise NotImplementedError

    def mask_from_block(self, a: Block, large_leg: Space, small_leg: ElementarySpace) -> DiagonalData:
        raise NotImplementedError('mask_from_block not implemented')  # TODO

    def mask_unary_operand(self, mask: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
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
        return FusionTreeData(b.data.coupled_sectors, blocks, b.data.domain, b.data.codomain, dtype)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        # OPTIMIZE should we offer the square-norm instead?
        norm_sq = 0
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
            norm_sq += a.symmetry.sector_dim(coupled) * (self.block_norm(block) ** 2)
        return self.block_sqrt(norm_sq)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        # TODO what target leg order is easiest? does it match the one specified in Backend.outer?
        raise NotImplementedError('outer not implemented')  # TODO

    def permute_legs(self, a: SymmetricTensor, **kw) -> Data:
        # TODO decide signature
        raise NotImplementedError('permute_legs not implemented')  # TODO

    def qr(self, a: SymmetricTensor, new_r_leg_dual: bool, full: bool
           ) -> tuple[Data, Data, ElementarySpace]:
        # TODO do SVD first, comments there apply.
        raise NotImplementedError('qr not implemented')  # TODO

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        raise NotImplementedError('scale_axis not implemented')  # TODO

    def set_element(self, a: SymmetricTensor, idcs: list[int], value: complex | float) -> Data:
        # TODO not sure this can even be done sensibly, one entry of the dense block
        #      affects in general many entries of the blocks.
        raise NotImplementedError('set_element not implemented')  # TODO

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        # TODO not sure this can even be done sensibly, one entry of the dense block
        #      affects in general many entries of the blocks.
        raise NotImplementedError('set_element_diagonal not implemented')  # TODO

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int],
                   final_legs: list[Space]) -> Data:
        # TODO do we need metadata to split, like in abelian?
        raise NotImplementedError('split_legs not implemented')  # TODO

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        raise NotImplementedError('squeeze_legs not implemented')  # TODO

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def svd(self, a: SymmetricTensor, new_vh_leg_dual: bool, algorithm: str | None,
            compute_u: bool, compute_vh: bool) -> tuple[Data, DiagonalData, Data, ElementarySpace]:
        # TODO need to redesign Backend.svd specification! need to allow more than two legs!
        # TODO need to be able to specify levels of braiding in general case!
        raise NotImplementedError('svd not implemented')  # TODO

    def tdot(self, a: SymmetricTensor, b: SymmetricTensor, axs_a: list[int],
             axs_b: list[int]) -> Data:
        # TODO need to be able to specify levels of braiding in general case!
        # TODO offer separate planar version? or just high
        raise NotImplementedError('tdot not implemented')  # TODO

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        assert a.symmetry.can_be_dropped
        J = len(a.data.codomain.spaces)
        K = len(a.data.domain.spaces)
        num_legs = J + K
        dtype = Dtype.common(a.data.dtype, a.symmetry.fusion_tensor_dtype)
        sym = a.symmetry
        # build in internal basis order first, then apply permutations in the end
        # build in codomain/domain leg order first, then permute legs in the end
        # [i1,...,iJ,j1,...,jK]
        shape = [leg.dim for leg in a.data.codomain.spaces] + [leg.dim for leg in a.data.domain.spaces]
        res = self.zero_block(shape, dtype)
        for coupled, block in zip(a.data.coupled_sectors, a.data.blocks):
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(a.data.domain.spaces, sym):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = tree_block_size(a.data.domain, b_sectors)
                for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(a.data.codomain.spaces, sym):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = tree_block_size(a.data.codomain, a_sectors)
                    entries, num_alpha_trees, num_beta_trees = self._get_forest_block_contribution(
                        block, sym, a.data.codomain, a.data.domain, coupled, a_sectors, b_sectors,
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
                    res[*j1, *j2] += entries
                    i1 += forest_b_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_b_width  # move right by one forest-block
        # permute leg order [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1]
        res = self.block_permute_axes(res, [*range(J), *reversed(range(J, J + K))])
        # apply permutation to public basis order
        res = self.apply_basis_perm(res, conventional_leg_order(a), inv=True)
        return res

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('to_dense_block_trivial_sector not implemented')  # TODO

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> FusionTreeData:
        blocks = [self.block_to_dtype(block, dtype) for block in a.data.blocks]
        return FusionTreeData(a.data.coupled_sectors, blocks, a.data.domain, a.data.codomain, dtype)

    def trace_full(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int]
                   ) -> float | complex:
        raise NotImplementedError('trace_full not implemented')  # TODO

    def trace_partial(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int],
                      remaining_idcs: list[int]) -> Data:
        raise NotImplementedError('trace_partial not implemented')  # TODO

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype
                  ) -> FusionTreeData:
        return FusionTreeData(coupled_sectors=codomain.symmetry.empty_sector_array, blocks=[],
                              domain=domain, codomain=codomain, dtype=dtype)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return FusionTreeData(coupled_sectors=co_domain.symmetry.empty_sector_array, blocks=[],
                              domain=co_domain, codomain=co_domain, dtype=dtype)

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
