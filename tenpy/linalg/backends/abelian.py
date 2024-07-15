"""Backends for abelian group symmetries.

Changes compared to old np_conserved:

- replace `ChargeInfo` by subclasses of `AbelianGroup` (or `ProductSymmetry`)
- replace `LegCharge` by `ElementarySpace` and `LegPipe` by `ProductSpace`. Changed class hierarchy!
- standard `Tensor` have qtotal=0, only ChargedTensor can have non-zero qtotal
- relabeling:
    - `Array.qdata`, "qind" and "qindices" to `AbelianBackendData.block_inds` and "block indices"
    - `LegPipe.qmap` to `ProductSpace.metadata['_block_ind_map']` (with changed column order!!!)
    - `LegPipe._perm` to `ProductSpace._perm_block_inds_map`  TODO (JU) this is outdated np?
    - `LegCharge.get_block_sizes()` is just `Space.multiplicities`
- TODO point towards Space attributes
- keep spaces "sorted" and "bunched",
  i.e. do not support legs with smaller blocks to effectively allow block-sparse tensors with
  smaller blocks than dictated by symmetries (which we actually have in H_MPO on the virtual legs...)
  In turn, ElementarySpace saves a `_perm` used to sort the originally passed `sectors`.
- keep `block_inds` sorted (i.e. no arbitrary gauge permutation in block indices)

"""
# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy import ndarray

from .abstract_backend import (
    TensorBackend, Data, DiagonalData, MaskData, Block, conventional_leg_order
)
from ..dtypes import Dtype
from ..symmetries import BraidingStyle, Symmetry, SectorArray
from ..spaces import Space, ElementarySpace, ProductSpace
from ...tools.misc import (
    inverse_permutation, list_to_dict_list, rank_data, iter_common_noncommon_sorted_arrays,
    iter_common_sorted, iter_common_sorted_arrays, make_stride, find_row_differences
)

__all__ = ['AbelianBackendData', 'AbelianBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


def _valid_block_inds(codomain: ProductSpace, domain: ProductSpace):
    # OPTIMIZE: this is brute-force going through all possible combinations of block indices
    # spaces are sorted, so we can probably reduce that search space quite a bit...
    M = codomain.num_spaces
    N = domain.num_spaces
    symmetry = codomain.symmetry
    grid = np.indices((s.num_sectors for s in conventional_leg_order(codomain, domain)), dtype=int)
    grid = grid.T.reshape((-1, M + N))
    codomain_coupled = symmetry.multiple_fusion_broadcast(
        *(space.sectors[i] for space, i in zip(codomain.spaces, grid.T))
    )
    domain_coupled = symmetry.multiple_fusion_broadcast(
        *(space.sectors[i] for space, i in zip(domain.spaces, grid.T[::-1]))
    )
    valid = np.all(codomain_coupled == domain_coupled, axis=1)
    block_inds = grid[valid, :]
    perm = np.lexsort(block_inds.T)
    return block_inds[perm]


class AbelianBackendData:
    """Data stored in a Tensor for :class:`AbelianBackend`.

    The :attr:`block_inds` can be visualized as follows::

        |           ---- codomain ---->  <--- domain ----
        |
        |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    b |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    l |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    o |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    c |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    k |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |    s |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
        |      v

    Attributes
    ----------
    dtype : Dtype
        The dtype of the data
    blocks : list of block
        A list of blocks containing the actual entries of the tensor.
        Leg order is ``[*codomain, *reversed(domain()]``, like ``Tensor.legs``.
    block_inds : 2D ndarray
        A 2D array of positive integers with shape (len(blocks), num_legs).
        The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of ``leg``,
        that is to ``leg.sectors[block_inds[n, m]]``,
        where ``leg == (codomain.spaces[m] if m < len(codomain) else domain.spaces[-1 - m]``.
        Thus, the columns of `block_inds` follow the same ordering convention as :attr:`Tensor.legs`.
        By convention, we store `blocks` and `block_inds` such that ``np.lexsort(block_inds.T)``
        is sorted.

    Parameters
    ----------
    dtype, blocks, block_inds
        like attributes above, but not necessarily sorted
    is_sorted : bool
        If ``False`` (default), we permute `blocks` and `block_inds` according to
        ``np.lexsort(block_inds.T)``.
        If ``True``, we assume they are sorted *without* checking.
    """
    def __init__(self, dtype: Dtype, blocks: list[Block], block_inds: ndarray,
                 is_sorted: bool = False):
        if not is_sorted:
            perm = np.lexsort(block_inds.T)
            block_inds = block_inds[perm, :]
            blocks = [blocks[n] for n in perm]
        self.dtype = dtype
        self.blocks = blocks
        self.block_inds = block_inds

    def get_block_num(self, block_inds: ndarray) -> Block | None:
        """Return the index ``n`` of the block which matches the block_inds.

        I.e. such that ``all(self.block_inds[n, :] == block_inds)``.
        Return None if no such ``n`` exists.
        """
        # OPTIMIZE use sorted for lookup?
        match = np.argwhere(np.all(self.block_inds == block_inds, axis=1))[:, 0]
        if len(match) == 0:
            return None
        return match[0]

    def get_block(self, block_inds: ndarray) -> Block | None:
        """Return the block in :attr:`blocks` matching the given block_inds,
        i.e. `self.blocks[n]` such that `all(self.block_inds[n, :] == blocks_inds)`
        or None if no such block exists
        """
        block_num = self.get_block_num(block_inds)
        return None if block_num is None else self.blocks[block_num]


class AbelianBackend(TensorBackend):
    """Backend for Abelian group symmetries.

    Notes
    -----
    The data stored for the various tensor classes defined in ``tenpy.linalg.tensors`` is::

        - ``SymmetricTensor``:
            An ``AbelianBackendData`` instance whose blocks have as many axes as the tensor has legs.

        - ``DiagonalTensor`` :
            An ``AbelianBackendData`` instance whose blocks have only a single axis.
            This is the diagonal of the corresponding 2D block in a ``Tensor``.

        - ``Mask`` :
            An ``AbelianBackendData`` instance whose blocks have only a single axis and bool values.
            These bool values indicate which indices of the large leg are kept for the small leg.
            The block_inds refer to the two legs of the mask, as they would for SymmetricTensor, in
            the usual order. Note that the position of the larger leg depends on ``Mask.is_projection``!

    """
    DataCls = AbelianBackendData

    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        # check block_inds
        assert not np.any(a.data.block_inds < 0)
        sector_nums = [leg.num_sectors for leg in conventional_leg_order(a)]
        assert not np.any(a.data.block_inds >= np.array([sector_nums]))
        assert np.all(np.lexsort(a.data.block_inds.T) == np.arange(len(a.data.blocks)))
        if a.data.block_inds.shape != (len(a.data.blocks), a.num_legs):
            msg = f'Wrong blocks_inds shape. ' \
                  f'Expected {(len(a.data.blocks), a.num_legs)}, got {a.data.block_inds.shape}.'
            raise ValueError(msg)
        if is_diagonal:
            assert np.all(a.data.block_inds[:, 0] == a.data.block_inds[:, 1])
        # check block_inds fulfill charge rule
        for inds in a.data.block_inds:
            # OPTIMIZE can do this with one multiple_fusion_broadcast call?
            codomain_coupled = a.symmetry.multiple_fusion(
                *(leg.sectors[i] for leg, i in zip(a.codomain.spaces, inds))
            )
            domain_coupled = a.symmetry.multiple_fusion(
                *(leg.sectors[i] for leg, i in zip(a.domain.spaces, inds[::-1]))
            )
            assert np.all(codomain_coupled == domain_coupled)
        # check expected tensor dimensions
        for block, b_i in zip(a.data.blocks, a.data.block_inds):
            if is_diagonal:
                shape = (a.leg.multiplicities[b_i[0]],)
            else:
                shape = tuple(leg.multiplicities[i]
                              for leg, i in zip(conventional_leg_order(a), b_i))
            self.block_backend.test_block_sanity(block, expect_shape=shape, expect_dtype=a.dtype)
        # check matching dtypes
        assert all(self.block_backend.block_dtype(block) == a.data.dtype for block in a.data.blocks)
        
    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        if a.data.block_inds.shape != (len(a.data.blocks), 2):
            msg = f'Wrong blocks_inds shape. ' \
                  f'Expected {(len(a.data.blocks), 2)}, got {a.data.block_inds.shape}.'
            raise ValueError(msg)
        assert a.dtype == a.data.dtype == Dtype.bool
        large_leg = a.large_leg
        small_leg = a.small_leg
        for block, block_inds in zip(a.data.blocks, a.data.block_inds):
            if a.is_projection:
                bi_small, bi_large = block_inds
            else:
                bi_large, bi_small = block_inds
            assert 0 <= bi_large < large_leg.num_sectors
            assert 0 <= bi_small < small_leg.num_sectors
            assert bi_large >= bi_small
            assert np.all(large_leg.sectors[bi_large] == small_leg.sectors[bi_small])
            assert self.block_backend.block_shape(block) == (large_leg.multiplicities[bi_large],)
            assert self.block_backend.block_sum_all(block) == small_leg.multiplicities[bi_small]
            assert self.block_backend.block_dtype(block) == Dtype.bool

    # ABSTRACT METHODS

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block],
                                         dtype_map: Callable[[Dtype], Dtype] | None) -> Data:
        leg = a.domain.spaces[0]
        a_block_inds = a.data.block_inds
        all_block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        res_blocks = []
        for i, j in iter_common_noncommon_sorted_arrays(a_block_inds, all_block_inds):
            if i is None:
                # use that all_block_inds is just ascending -> all_block_inds[j, 0] == j
                block = self.block_backend.zero_block(shape=[leg.multiplicities[j]] * 2, dtype=a.dtype)
            else:
                block = a.data.blocks[i]
            res_blocks.append(block_method(block))
        if dtype_map is None:
            dtype = a.dtype
        else:
            dtype = dtype_map(a.dtype)
        res_blocks = [self.block_backend.block_to_dtype(block, dtype) for block in res_blocks]
        return AbelianBackendData(dtype, res_blocks, all_block_inds, is_sorted=True)

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        blocks = [self.block_backend.block_add_axis(block, legs_pos) for block in a.data.blocks]
        block_inds = np.insert(a.data.block_inds, legs_pos, 0, axis=1)
        # since the new column is constant, block_inds are still sorted.
        return AbelianBackendData(a.data.dtype, blocks, block_inds, is_sorted=True)

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        for i, j in iter_common_noncommon_sorted_arrays(a.data.block_inds, b.data.block_inds):
            if j is None:
                if self.block_backend.block_max_abs(a_blocks[i]) > atol:
                    return False
            elif i is None:
                if self.block_backend.block_max_abs(b_blocks[j]) > atol:
                    return False
            else:
                if not self.block_backend.block_allclose(a_blocks[i], b_blocks[j], rtol=rtol, atol=atol):
                    return False
        return True

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        tensor_blocks = tensor.data.blocks
        tensor_block_inds_contr = tensor.data.block_inds[:, :1]  # is sorted
        mask_blocks = mask.data.blocks
        mask_block_inds = mask.data.block_inds
        mask_block_inds_contr = mask_block_inds[:, 1]  # is sorted
        res_blocks = []
        res_block_inds = []  # append only for one leg, repeat later
        for i, j in iter_common_sorted(tensor_block_inds_contr, mask_block_inds_contr):
            block = self.block_backend.block_apply_mask(tensor_blocks[i], mask_blocks[j], ax=0)
            res_blocks.append(block)
            res_block_inds.append(mask_block_inds[j, 0])
        if len(res_block_inds) > 0:
            res_block_inds = np.repeat(np.array(res_block_inds)[:, None], 2, axis=1)
        else:
            res_block_inds = np.zeros((0, 2), int)
        return AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=True)

    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     product_spaces: list[ProductSpace],
                     new_codomain: ProductSpace,
                     new_domain: ProductSpace,
                     ) -> Data:
        raise NotImplementedError('currently bugged')  # TODO
        num_result_legs = tensor.num_legs - sum(len(group) - 1 for group in leg_idcs_combine)
        old_block_inds = tensor.data.block_inds
        old_blocks = tensor.data.blocks
        #
        # first, find block indices of the final array to which we map
        map_inds = [
            self.product_space_map_incoming_block_inds(p_space, old_block_inds[:, first:last + 1])
            for p_space, (first, *_, last) in zip(product_spaces, leg_idcs_combine)
        ]
        #
        # build result block_inds
        res_block_inds = np.empty((len(old_block_inds), num_result_legs), int)
        i = 0  # res_block_inds[:, :i] is already set
        j = 0  # old_block_inds[:, :j] are already considered
        for group, p_space, map_ind in zip(leg_idcs_combine, product_spaces, map_inds):
            # fill in block_inds for the uncombined legs that came up since the last group
            num_uncombined = group[0] - j
            res_block_inds[:, i:i + num_uncombined] = old_block_inds[:, j:j + num_uncombined]
            i += num_uncombined
            j += num_uncombined
            # fill in block_inds for the current combined group
            block_ind_map = p_space.get_metadata('_block_ind_map', backend=self)
            res_block_inds[:, i] = block_ind_map[map_ind, -1]
            i += 1
            j += len(group)
        # trailing uncombined legs:
        res_block_inds[:, i:] = old_block_inds[:, j:]
        #
        # now we have probably many duplicate rows in res_block_inds, since many combinations of
        # non-combined block indices map to the same block index in product space
        # -> find unique entries by sorting res_block_inds
        sort = np.lexsort(res_block_inds.T)
        res_block_inds = res_block_inds[sort]
        old_blocks = [old_blocks[i] for i in sort]
        map_inds = [map_[sort] for map_ in map_inds]
        #
        # determine slices in the new blocks
        block_slices = np.zeros((len(old_blocks), num_result_legs, 2), int)
        block_shape = np.empty((len(old_blocks), num_result_legs), int)
        i = 0  # have already set info for new_legs[:i]
        j = 0  # have already considered old_legs[:j]
        for group, p_space, map_ind in zip(leg_idcs_combine, product_spaces, map_inds):
            # uncombined legs since previous group
            num_uncombined = group[0] - j
            for n in range(num_uncombined):
                # block_slices[:, i + n, 0] = 0 is already set, since we initialized zeros
                i2 = i + n
                leg_idx = j + n
                mult = tensor.get_leg_co_domain(leg_idx).multiplicities[res_block_inds[:, i2]]
                block_slices[:, i2, 1] = mult
                block_shape[:, i2] = mult
            i += num_uncombined
            j += num_uncombined
            # current combined group
            block_ind_map = p_space.get_metadata('_block_ind_map', backend=self)
            slices = block_ind_map[map_ind, :2]
            block_slices[:, i, :] = slices
            block_shape[:, i] = slices[:, 1] - slices[:, 0]
            i += 1
            j += len(group)
        # trailing uncombined legs
        for n in range(tensor.num_legs - j):
            i2 = i + n
            leg_idx = j + n
            mult = tensor.get_leg_co_domain(leg_idx).multiplicities[res_block_inds[:, i2]]
            block_slices[:, i2, 1] = mult
            block_shape[:, i2] = mult
        #
        # split res_block_inds into parts, which give a unique new blocks
        diffs = find_row_differences(res_block_inds, include_len=True)  # including 0 and len to have slices later
        res_num_blocks = len(diffs) - 1
        res_block_inds = res_block_inds[diffs[:-1], :]
        res_block_shapes = np.empty((res_num_blocks, num_result_legs), int)
        for i, leg in enumerate(conventional_leg_order(new_codomain, new_domain)):
            res_block_shapes[:, i] = leg.multiplicities[res_block_inds[:, i]]
        #
        # map the data
        res_blocks = []
        for shape, start, stop in zip(res_block_shapes, diffs[:-1], diffs[1:]):
            new_block = self.block_backend.zero_block(shape, dtype=tensor.dtype)
            for old_row in range(start, stop):  # copy blocks
                old_block = self.block_backend.block_reshape(old_blocks[old_row], block_shape[old_row])
                new_slices = tuple(slice(b, e) for (b, e) in block_slices[old_row])
                new_block[new_slices] = old_block
            res_blocks.append(new_block)
        
        # we lexsort( .T)-ed res_block_inds while it still had duplicates, and then indexed by diffs,
        # which is sorted and thus preserves lexsort( .T)-ing of res_block_inds
        return AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=True)

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """

        Notes
        -----
        Looking at the source of numpy's tensordot (which is just 62 lines of python code),
        you will find that it has the following strategy:

            1. Transpose `a` and `b` such that the axes to sum over are in the end of `a` and front of `b`.
            2. Combine the legs `axes`-legs and other legs with a `np.reshape`,
               such that `a` and `b` are matrices.
            3. Perform a matrix product with `np.dot`.
            4. Split the remaining axes with another `np.reshape` to obtain the correct shape.

        The main work is done by `np.dot`, which calls LAPACK to perform the simple matrix product.
        (This matrix multiplication of a ``NxK`` times ``KxM`` matrix is actually faster
        than the O(N*K*M) needed by a naive implementation looping over the indices.)

        We follow the same overall strategy;

        Step 1) is implemented in ``tenpy.tdot`` and is already done when we call ``tenpy.compose``;
        this function gets input with legs sorted accordingly. Note the different leg order the.
        We need to contract the last leg of `a` with the first legs of `b` and so on.

        The steps 2) and 4) could be implemented with `combine_legs` and `split_legs`.
        However, that would actually be an overkill: we're not interested
        in the full charge data of the combined legs (which would be generated in the LegPipes).
        Instead, we just need to track the block_inds of a and b carefully.

        For step 2), we split `a.data.block_inds` into `a_block_inds_keep` and `a_block_inds_contr`,
        and similar for `b`. Then, view `a` is a matrix :math:`A_{i,k1}` and `b` as :math:`B_{k2,j}`
        where `i` can be any row of `a_block_inds_keep`, `j` can be any row of `b_block_inds_keep`.
        The `k1` and `k2` are rows/columns of `a/b_block_inds_contr`, which come from compatible legs.
        In our storage scheme, `a.data.blocks[s]` then contains the block :math:`A_{i,k1}` for
        ``j = a_block_inds_keep[s]`` and ``k1 = a_block_inds_contr[s]``.
        To identify the different indices `i` and `j`, it is easiest to lexsort in the `s`.
        Note that we give priority to the `{a,b}_block_inds_keep` over the `_contr`, such that
        equal rows of `i` are contiguous in `a_block_inds_keep`.
        Then, they are identified with :func:`find_row_differences`.

        Now, the goal is to calculate the sums :math:`C_{i,j} = sum_k A_{i,k} B_{k,j}`,
        analogous to step 3) above. This is implemented directly in this function.
        It is done 'naively' by explicit loops over ``i``, ``j`` and ``k``.
        However, this is not as bad as it sounds:
        First, we loop only over existent ``i`` and ``j``
        (in the sense that there is at least some non-zero block with these ``i`` and ``j``).
        Second, if the ``i`` and ``j`` are not compatible with the new total charge,
        we know that ``C_{i,j}`` will be zero.
        Third, given ``i`` and ``j``, the sum over ``k`` runs only over
        ``k1`` with nonzero :math:`A_{i,k1}`, and ``k2` with nonzero :math:`B_{k2,j}`.

         How many multiplications :math:`A_{i,k} B_{k,j}` we actually have to perform
         depends on the sparseness. If ``k`` comes from a single leg, it is completely sorted
         by charges, so the 'sum' over ``k`` will contain at most one term!
        """
        if a.num_codomain_legs == 0 and b.num_domain_legs == 0:
            return self.inner(a, b, do_dagger=False)
        if a.num_domain_legs == 0:
            return self._compose_no_contraction(a, b)
        res_dtype = Dtype.common(a.dtype, b.dtype)

        if len(a.data.blocks) == 0 or len(b.data.blocks) == 0:
            # if there are no actual blocks to contract, we can directly return 0
            return self.zero_data(a.codomain, b.domain, res_dtype)

        # convert blocks to common dtype
        a_blocks = a.data.blocks
        if a.dtype != res_dtype:
            a_blocks = [self.block_backend.block_to_dtype(B, res_dtype) for B in a_blocks]
        b_blocks = b.data.blocks
        if b.dtype != res_dtype:
            b_blocks = [self.block_backend.block_to_dtype(B, res_dtype) for B in b_blocks]

        
        # need to contract the domain legs of a with the codomain legs of b.
        # due to the leg ordering

        # Deal with the columns of the block inds that are kept/contracted separately
        a_block_inds_keep, a_block_inds_contr = np.hsplit(a.data.block_inds, [a.num_codomain_legs])
        b_block_inds_contr, b_block_inds_keep = np.hsplit(b.data.block_inds, [b.num_codomain_legs])
        # Merge the block_inds on the contracted legs to a single column, using strides.
        # Note: The order in a.data.block_inds is opposite from the order in b.data.block_inds!
        #       I.e. a.data.block_inds[-1-n] and b.data.block_inds[n] describe one leg to contract
        # We choose F-style strides, by appearance in b.data.block_inds.
        # This guarantees that the b.data.block_inds sorting is preserved.
        # We do not care about the sorting of the a.data.block_inds, since we need to re-sort anyway,
        # to group by a_block_inds_keep.
        strides = make_stride([l.num_sectors for l in b.codomain], cstyle=False)
        a_block_inds_contr = np.sum(a_block_inds_contr * strides[::-1], axis=1)  # 1D array
        b_block_inds_contr = np.sum(b_block_inds_contr * strides, axis=1)  # 1D array
        
        # sort the a.data.block_inds *first* by the _keep, *then* by the _contr columns
        a_sort = np.lexsort(np.hstack([a_block_inds_contr[:, None], a_block_inds_keep]).T)
        a_block_inds_keep = a_block_inds_keep[a_sort, :]
        a_block_inds_contr = a_block_inds_contr[a_sort]
        a_blocks = [a_blocks[i] for i in a_sort]
        # The b_block_inds_* and b_blocks are already sorted like that.

        # now group everything that has matching *_block_inds_keep
        a_slices = find_row_differences(a_block_inds_keep, include_len=True)
        b_slices = find_row_differences(b_block_inds_keep, include_len=True)
        a_blocks = [a_blocks[i:i2] for i, i2 in zip(a_slices, a_slices[1:])]
        b_blocks = [b_blocks[j:j2] for j, j2 in zip(b_slices, b_slices[1:])]
        a_block_inds_contr = [a_block_inds_contr[i:i2] for i, i2 in zip(a_slices, a_slices[1:])]
        b_block_inds_contr = [b_block_inds_contr[j:j2] for j, j2 in zip(b_slices, b_slices[1:])]
        a_block_inds_keep = a_block_inds_keep[a_slices[:-1]]
        b_block_inds_keep = b_block_inds_keep[b_slices[:-1]]

        # Reshape blocks to matrices.
        # Reason: We could use block_tdot to do the pairwise block contractions.
        #         This would then internally reshape to matrices, to use e.g. GEMM.
        #         One of the a_blocks may be contracted with many different b_blocks, and require
        #         the same reshape every time. Instead, we do it once at this point.
        # All blocks in a_blocks[n] have the same kept legs -> same kept shape
        a_shape_keep = [self.block_backend.block_shape(blocks[0])[:a.num_codomain_legs]
                        for blocks in a_blocks]
        b_shape_keep = [self.block_backend.block_shape(blocks[0])[b.num_codomain_legs:]
                        for blocks in b_blocks]
        if a.num_codomain_legs == 0:
            # special case: reshape to vector.
            a_blocks = [[self.block_backend.block_reshape(B, (-1,)) for B in blocks]
                        for blocks in a_blocks]
        else:
            a_blocks = [[self.block_backend.block_reshape(B, (np.prod(shape_keep), -1))
                         for B in blocks] for blocks, shape_keep in zip(a_blocks, a_shape_keep)]
        # need to permute the leg order of one group of permuted legs.
        # OPTIMIZE does it matter, which?
        # choose to permute the legs of the b-blocks
        if b.num_domain_legs == 0:
            # special case: reshape to vector
            perm = list(reversed(range(b.num_legs)))
            b_blocks = [[
                self.block_backend.block_reshape(self.block_backend.block_permute_axes(B, perm), (-1,))
                for B in blocks
            ] for blocks in b_blocks]
        else:
            perm = [*reversed(range(b.num_codomain_legs)), *range(b.num_codomain_legs, b.num_legs)]
            b_blocks = [[
                self.block_backend.block_reshape(self.block_backend.block_permute_axes(B, perm),
                                                 (-1, np.prod(shape_keep))) 
                for B in blocks]
                for blocks, shape_keep in zip(b_blocks, b_shape_keep)
            ]

        # compute coupled sectors for all rows of the block inds // for all blocks
        if a.num_codomain_legs > 0:
            a_charges = a.symmetry.multiple_fusion_broadcast(
                *(leg.sectors[bi] for leg, bi in zip(a.codomain, a_block_inds_keep.T))
            )
        else:
            a_charges = np.repeat(a.symmetry.trivial_sector[None, :], len(a_block_inds_keep), axis=1)
        if b.num_domain_legs > 0:
            b_charges = a.symmetry.multiple_fusion_broadcast(
                *(leg.sectors[bi] for leg, bi in zip(b.domain, b_block_inds_keep[:, ::-1].T))
            )
        else:
            b_charges = np.repeat(a.symmetry.trivial_sector[None, :], len(b_block_inds_keep), axis=1)
        a_charge_lookup = list_to_dict_list(a_charges)  # lookup table ``tuple(sector) -> idcs_in_a_charges``

        # rows_a changes faster than cols_b, such that the resulting block_inds are lex-sorted
        res_blocks = []
        res_block_inds_a = []
        res_block_inds_b = []
        for col_b, coupled in enumerate(b_charges):
            b_blocks_in_col = b_blocks[col_b]
            rows_a = a_charge_lookup.get(tuple(coupled), [])  # empty list if no match
            for row_a in rows_a:
                common_inds_iter = iter_common_sorted(a_block_inds_contr[row_a], b_block_inds_contr[col_b])
                # Use first pair of common indices to initialize a block.
                try:
                    k1, k2 = next(common_inds_iter)
                except StopIteration:
                    continue
                a_blocks_in_row = a_blocks[row_a]
                block = self.block_backend.matrix_dot(a_blocks_in_row[k1], b_blocks_in_col[k2])
                # for further pairs of common indices, add the result onto the existing block
                for k1, k2 in common_inds_iter:
                    block += self.block_backend.matrix_dot(a_blocks_in_row[k1], b_blocks_in_col[k2])
                block = self.block_backend.block_reshape(
                    block, a_shape_keep[row_a] + b_shape_keep[col_b]
                )
                res_blocks.append(block)
                res_block_inds_a.append(a_block_inds_keep[row_a])
                res_block_inds_b.append(b_block_inds_keep[col_b])

        # finish up:
        if len(res_blocks) == 0:
            block_inds = np.zeros((0, a.num_codomain_legs + b.num_domain_legs), dtype=int)
        else:
            block_inds = np.hstack((res_block_inds_a, res_block_inds_b))
        return AbelianBackendData(res_dtype, blocks=res_blocks, block_inds=block_inds,
                                  is_sorted=True)

    def _compose_no_contraction(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """special case of :meth:`compose` where no legs are actually contracted.
        Note that this is not the same as :meth:`outer`, the resulting leg order is different.
        """
        res_dtype = a.data.dtype.common(b.data.dtype)
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        if a.data.dtype != res_dtype:
            a_blocks = [self.block_backend.block_to_dtype(T, res_dtype) for T in a_blocks]
        if b.data.dtype != res_dtype:
            b_blocks = [self.block_backend.block_to_dtype(T, res_dtype) for T in b_blocks]
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        l_a, num_legs_a = a_block_inds.shape
        l_b, num_legs_b = b_block_inds.shape
        grid = np.indices([len(a_block_inds), len(b_block_inds)]).T.reshape(-1, 2)
        # grid is lexsorted, with rows as all combinations of a/b block indices.
        res_block_inds = np.empty((l_a * l_b, num_legs_a + num_legs_b), dtype=int)
        res_block_inds[:, :num_legs_a] = a_block_inds[grid[:, 0]]
        res_block_inds[:, num_legs_a:] = b_block_inds[grid[:, 1]]
        res_blocks = [self.block_backend.block_outer(a_blocks[i], b_blocks[j]) for i, j in grid]
        # TODO (JU) are the block_inds actually sorted?
        #  if yes: add comment explaining why, adjust argument below
        return AbelianBackendData(res_dtype, res_blocks, res_block_inds, is_sorted=False)

    def copy_data(self, a: SymmetricTensor | DiagonalTensor) -> Data | DiagonalData:
        blocks = [self.block_backend.block_copy(b) for b in a.data.blocks]
        return AbelianBackendData(a.data.dtype, blocks, a.data.block_inds.copy(), is_sorted=True)

    def dagger(self, a: SymmetricTensor) -> Data:
        blocks = [self.block_backend.block_dagger(b) for b in a.data.blocks]
        block_inds = a.data.block_inds[:, ::-1]
        return AbelianBackendData(a.dtype, blocks=blocks, block_inds=block_inds)

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_backend.block_item(a.blocks[0])

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        if len(a.data.block_inds) < a.leg.num_sectors:
            # missing blocks are filled with False
            return False
        # now it is enough to check that all existing blocks are all-True
        return all(self.block_backend.block_all(b) for b in a.data.blocks)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return any(self.block_backend.block_any(b) for b in a.data.blocks)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        # TODO could further distinguish cases for what is zero and drop respective blocks:
        #  - only left:: func(0, b) == 0
        #  - only right:: func(a, 0) == 0
        #  - only both:: func(0, 0) == 0
        leg = a.leg
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        
        blocks = []
        block_inds = []

        ia = 0  # next block of a to process
        bi_a = -1 if len(a_block_inds) == 0 else a_block_inds[ia, 0]  # block_ind of that block => it belongs to leg.sectors[bi_a]
        ib = 0  # next block of b to process
        bi_b = -1 if len(b_block_inds) == 0 else b_block_inds[ib, 0]  # block_ind of that block => it belongs to leg.sectors[bi_b]
        #
        for i, mult in enumerate(leg.multiplicities):
            if i == bi_a:
                block_a = a_blocks[ia]
                ia += 1
                if ia >= len(a_block_inds):
                    bi_a = -1  # a has no further blocks
                else:
                    bi_a = a_block_inds[ia, 0]
            elif partial_zero_is_zero:
                continue
            else:
                block_a = self.block_backend.zero_block([mult], a.dtype)

            if i == bi_b:
                block_b = b_blocks[ib]
                ib += 1
                if ib >= len(b_block_inds):
                    bi_b = -1  # b has no further blocks
                else:
                    bi_b = b_block_inds[ib, 0]
            elif partial_zero_is_zero:
                continue
            else:
                block_b = self.block_backend.zero_block([mult], a.dtype)
            blocks.append(func(block_a, block_b, **func_kwargs))
            block_inds.append(i)

        if len(blocks) == 0:
            block_inds = np.zeros((0, 2), int)
            dtype = self.block_backend.block_dtype(
                func(self.block_backend.ones_block([1], dtype=a.dtype),
                     self.block_backend.ones_block([1], dtype=b.dtype))
            )
        else:
            block_inds = np.repeat(np.array(block_inds)[:, None], 2, axis=1)
            dtype = self.block_backend.block_dtype(blocks[0])
            
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        a_blocks = a.data.blocks
        if maps_zero_to_zero:
            blocks = [func(block, **func_kwargs) for block in a.data.blocks]
            block_inds = a.data.block_inds
        else:
            a_block_inds = a.data.block_inds
            block_inds = np.repeat(np.arange(a.leg.num_sectors)[:, None], 2, axis=1)
            blocks = []
            for i, j in iter_common_noncommon_sorted_arrays(block_inds, a_block_inds):
                if j is None:
                    # use that block_inds is just arange -> block_inds[i, 0] == i
                    block = self.block_backend.zero_block([a.leg.multiplicities[i]], dtype=a.dtype)
                else:
                    block = a_blocks[j]
                blocks.append(func(block, **func_kwargs))
        if len(blocks) == 0:
            example_block = func(self.block_backend.zero_block([1], dtype=a.dtype))
            dtype = self.block_backend.block_dtype(example_block)
        else:
            dtype = self.block_backend.block_dtype(blocks[0])
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        leg = co_domain.spaces[0]
        dtype = self.block_backend.block_dtype(a)
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        blocks = [a[slice(*leg.slices[i])] for i in block_inds[:, 0]]
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        leg = co_domain.spaces[0]
        block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        blocks = [func((mult,), coupled)
                  for coupled, mult in zip(leg.sectors, leg.multiplicities)]
        if len(blocks) == 0:
            sample_block = func((1,), co_domain.symmetry.trivial_sector)
        else:
            sample_block = blocks[0]
        dtype = self.block_backend.block_dtype(sample_block)
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool) -> DiagonalData:
        blocks = [self.block_backend.block_get_diagonal(block, check_offdiagonal)
                  for block in a.data.blocks]
        return AbelianBackendData(a.dtype, blocks, a.data.block_inds, is_sorted=True)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        total_sum = a.data.dtype.zero_scalar
        for block in a.data.blocks:
            total_sum += self.block_backend.block_sum_all(block)
        return total_sum

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        res = self.block_backend.zero_block([a.leg.dim], a.dtype)
        for block, b_i_0 in zip(a.data.blocks, a.data.block_inds[:, 0]):
            res[slice(*a.leg.slices[b_i_0])] = block
        return res

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[MaskData, ElementarySpace]:
        large_leg = tens.leg
        basis_perm = large_leg._basis_perm
        blocks = []
        large_leg_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        for diag_block, diag_bi in zip(tens.data.blocks, tens.data.block_inds):
            if not self.block_backend.block_any(diag_block):
                continue
            bi, _ = diag_bi
            #
            blocks.append(diag_block)
            large_leg_block_inds.append(bi)
            sectors.append(large_leg.sectors[bi])
            multiplicities.append(self.block_backend.block_sum_all(diag_block))
            if basis_perm is not None:
                basis_perm_ranks.append(basis_perm[slice(*large_leg.slices[bi])][diag_block])

        if len(blocks) == 0:
            sectors = tens.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), large_leg_block_inds])
        
        data = AbelianBackendData(
            dtype=Dtype.bool, blocks=blocks, block_inds=np.array(block_inds, int), is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=tens.symmetry, sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg
    
    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        dual_leg, perm = tens.leg._dual_space(return_perm=True)
        data = AbelianBackendData(
            tens.dtype, blocks=tens.data.blocks,
            block_inds=inverse_permutation(perm)[tens.data.block_inds]
        )
        return dual_leg, data
    
    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        a_block_inds = a.data.block_inds
        # for missing blocks, i.e. a zero block, the eigenvalues are zero, so we can just skip
        # adding that block to the eigenvalues.
        # for the eigenvectors, we choose the computational basis vectors, i.e. the matrix
        # representation within that block is the identity matrix.
        # we initialize all blocks to eye and override those where `a` has blocks.
        v_data = self.eye_data(a.domain, a.dtype)
        w_blocks = []
        for block, bi in zip(a.data.blocks, a_block_inds):
            vals, vects = self.block_backend.block_eigh(block, sort=sort)
            w_blocks.append(vals)
            v_data.blocks[bi[0]] = vects
        w_data = AbelianBackendData(dtype=a.dtype.to_real, blocks=w_blocks, block_inds=a_block_inds,
                                    is_sorted=True)
        return w_data, v_data

    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        # results[i1,...im,jm,...,j1] = delta_{i1,j1} ... delta{im,jm}
        # need exactly the "diagonal" blocks, where sector of i1 matches the one of j1 etc.
        # to guarantee sorting later, it is easier to generate the block inds of the domain
        domain_dims = [leg.num_sectors for leg in reversed(co_domain.spaces)]
        domain_block_inds = np.indices(domain_dims).T.reshape(-1, co_domain.num_spaces)
        block_inds = np.hstack([domain_block_inds[:, ::-1], domain_block_inds])
        # domain_block_inds is by construction np.lexsort( .T)-ed.
        # since the last co_domain.num_spaces columns of block_inds are already unique, the first
        # columns are not relevant to np.lexsort( .T)-ing, thus the block_inds above is sorted.
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i] for leg, i in zip(co_domain.spaces, b_i)]
            blocks.append(self.block_backend.eye_block(shape, dtype))
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> AbelianBackendData:
        dtype = self.block_backend.block_dtype(a)
        projected = self.block_backend.zero_block(self.block_backend.block_shape(a), dtype=dtype)
        block_inds = _valid_block_inds(codomain, domain)
        blocks = []
        for b_i in block_inds:
            slices = tuple(slice(*leg.slices[i])
                           for i, leg in zip(b_i, conventional_leg_order(codomain, domain)))
            block = a[slices]
            blocks.append(block)
            projected[slices] = block
        if tol is not None:
            if self.block_backend.block_norm(a - projected) > tol * self.block_backend.block_norm(a):
                raise ValueError('Block is not symmetric up to tolerance.')
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)
 
    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        bi = leg.sectors_where(leg.symmetry.trivial_sector)
        return AbelianBackendData(
            dtype=self.block_backend.block_dtype(block), blocks=[block],
            block_inds=np.array([[bi]]),
            is_sorted=True
        )

    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        return self.from_sector_block_func(
            func=lambda shape, coupled: self.block_backend.block_random_normal(shape, dtype, sigma),
            codomain=codomain, domain=domain
        )

    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        block_inds = _valid_block_inds(codomain=codomain, domain=domain)
        M = codomain.num_spaces
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i]
                     for i, leg in zip(b_i, conventional_leg_order(codomain, domain))]
            coupled = codomain.symmetry.multiple_fusion(
                *(leg.sectors[i] for i, leg in zip(b_i, codomain.spaces))
            )
            blocks.append(func(shape, coupled))
        if len(blocks) == 0:
            sample_block = func((1,) * (M + domain.num_spaces), codomain.symmetry.trivial_sector)
        else:
            sample_block = blocks[0]
        dtype = self.block_backend.block_dtype(sample_block)
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_backend.block_from_diagonal(block) for block in a.data.blocks]
        return AbelianBackendData(a.dtype, blocks, a.data.block_inds, is_sorted=True)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        blocks = [self.block_backend.block_from_mask(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, blocks, a.data.block_inds, is_sorted=True)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        pos = np.array([l.parse_index(idx) for l, idx in zip(conventional_leg_order(a), idcs)])
        block = a.data.get_block(pos[:, 0])
        if block is None:
            return a.dtype.zero_scalar
        return self.block_backend.get_block_element(block, pos[:, 1])

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        block_idx, idx_within = a.leg.parse_index(idx)
        block = a.data.get_block(np.array([block_idx, block_idx]))
        if block is None:
            return a.dtype.zero_scalar
        return self.block_backend.get_block_element(block, [idx_within])

    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        pos = np.array([l.parse_index(idx) for l, idx in zip(conventional_leg_order(a), idcs)])
        block = a.data.get_block(pos[:, 0])
        if block is None:
            return False
        if a.is_projection:
            small, large = pos[:, 1]
        else:
            large, small = pos[:, 1]
        return self.block_backend.get_block_mask_element(block, large, small)
        
    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        # F-style strides for block_inds -> preserve sorting
        strides = make_stride([l.num_sectors for l in a.legs], cstyle=False)
        a_block_inds = np.sum(a.data.block_inds * strides, axis=1)
        if do_dagger:
            b_block_inds = np.sum(b.data.block_inds * strides, axis=1)
        else:
            b_block_inds = np.sum(b.data.block_inds * strides[::-1], axis=1)
            # these are not sorted:
            sort = np.argsort(b_block_inds)
            b_block_inds = b_block_inds[sort]
            b_blocks = [b_blocks[i] for i in sort]
        res = 0.
        for i, j in iter_common_sorted(a_block_inds, b_block_inds):
            res += self.block_backend.block_inner(a_blocks[i], b_blocks[j], do_dagger=do_dagger)
        return res

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                                charge_leg: ElementarySpace) -> Data:
        assert charge_leg.num_sectors == 1
        bi = space.sectors_where(charge_leg.sectors[0])
        assert bi is not None
        assert self.block_backend.block_shape(vector) == (space.multiplicities[bi])
        return AbelianBackendData(
            dtype=self.block_backend.block_dtype(vector),
            blocks=[self.block_backend.block_add_axis(vector, pos=1)],
            block_inds=np.array([[bi, 0]])
        )

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        num_blocks = len(tensor.data.blocks)
        assert num_blocks <= 1
        if num_blocks == 1:
            return tensor.data.blocks[0][:, 0]
        sector = tensor.domain[0].sectors[0]
        dim = tensor.codomain[0].sector_multiplicity(sector)
        return self.block_backend.zero_block([dim], dtype=tensor.data.dtype)

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        v_blocks = v.data.blocks
        w_blocks = w.data.blocks
        v_block_inds = v.data.block_inds
        w_block_inds = w.data.block_inds
        # ensure common dtypes
        common_dtype = v.dtype.common(w.dtype)
        if v.data.dtype != common_dtype:
            v_blocks = [self.block_backend.block_to_dtype(T, common_dtype) for T in v_blocks]
        if w.data.dtype != common_dtype:
            w_blocks = [self.block_backend.block_to_dtype(T, common_dtype) for T in w_blocks]
        res_blocks = []
        res_block_inds = []
        for i, j in iter_common_noncommon_sorted_arrays(v_block_inds, w_block_inds):
            if j is None:
                res_blocks.append(self.block_backend.block_mul(a, v_blocks[i]))
                res_block_inds.append(v_block_inds[i])
            elif i is None:
                res_blocks.append(self.block_backend.block_mul(b, w_blocks[j]))
                res_block_inds.append(w_block_inds[j])
            else:
                res_blocks.append(
                    self.block_backend.block_linear_combination(a, v_blocks[i], b, w_blocks[j])
                )
                res_block_inds.append(v_block_inds[i])
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, v.num_legs), int)
        return AbelianBackendData(common_dtype, res_blocks, res_block_inds, is_sorted=True)

    def lq(self, a: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        assert a.num_codomain_legs == 1 == a.num_domain_legs  # since self.can_decompose_tensors is False
        l_blocks = []
        q_blocks = []
        l_block_inds = []
        q_block_inds = []
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            # due to the loop setup we have:
            #   a.codomain.sectors[j] == new_leg.sectors[n]
            #   a.domain.sectors[k] == new_leg.sectors[n]
            if i < len(a_block_inds) and a_block_inds[i, 0] == j:
                # we have a block for that sector -> decompose it
                l, q = self.block_backend.matrix_lq(a_blocks[i], full=False)
                l_blocks.append(l)
                q_blocks.append(q)
                l_block_inds.append([j, n])
                i += 1
            else:
                # we do not have a block for that sector
                # => L_block == 0 and we dont even set it.
                # can choose arbitrary blocks for q, as long as they are isometric
                new_leg_dim = new_leg.multiplicities[n]
                q_blocks.append(
                    self.block_backend.eye_matrix(a.domain.multiplicities[k], a.dtype)[:new_leg_dim, :]
                )
            q_block_inds.append([n, k])
        if len(l_blocks) == 0:
            l_block_inds = np.zeros((0, 2), int)
        else:
            l_block_inds = np.array(l_block_inds, int)
        if len(q_blocks) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds, int)
        #
        l_data = AbelianBackendData(a.dtype, l_blocks, l_block_inds, is_sorted=True)
        q_data = AbelianBackendData(a.dtype, q_blocks, q_block_inds, is_sorted=True)
        return l_data, q_data
    
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = mask1.large_leg
        basis_perm = large_leg._basis_perm
        mask1_block_inds = mask1.data.block_inds
        mask1_blocks = mask1.data.blocks
        mask2_block_inds = mask2.data.block_inds
        mask2_blocks = mask2.data.blocks
        #
        blocks = []
        large_leg_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        #
        i1 = 0  # next block of mask1 to process
        b1_i1 = -1 if len(mask1_block_inds) == 0 else mask1_block_inds[i1, 1]  # its block_ind for the large leg.
        i2 = 0
        b2_i2 = -1 if len(mask2_block_inds) == 0 else mask2_block_inds[i2, 1]
        #
        for sector_idx, (sector, slc) in enumerate(zip(large_leg.sectors, large_leg.slices)):
            if sector_idx == b1_i1:
                block1 = mask1_blocks[i1]
                i1 += 1
                if i1 >= len(mask1_block_inds):
                    b1_i1 = -1  # mask1 has no further blocks
                else:
                    b1_i1 = mask1_block_inds[i1, 1]
            else:
                block1 = self.block_backend.zero_block([large_leg.multiplicities[sector_idx]], Dtype.bool)
            if sector_idx == b2_i2:
                block2 = mask2_blocks[i2]
                i2 += 1
                if i2 >= len(mask2_block_inds):
                    b2_i2 = -1  # mask2 has no further blocks
                else:
                    b2_i2 = mask1_block_inds[i2, 1]
            else:
                block2 = self.block_backend.zero_block([large_leg.multiplicities[sector_idx]], Dtype.bool)
            new_block = func(block1, block2)
            mult = self.block_backend.block_sum_all(new_block)
            if mult == 0:
                continue
            blocks.append(new_block)
            large_leg_block_inds.append(sector_idx)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                basis_perm_ranks.append(basis_perm[slice(*slc)][new_block])
        block_inds = np.column_stack([np.arange(len(sectors)), large_leg_block_inds])
        data = AbelianBackendData(
            dtype=Dtype.bool, blocks=blocks, block_inds=block_inds, is_sorted=True
        )
        if len(sector) == 0:
            sectors = mask1.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
        small_leg = ElementarySpace(
            symmetry=mask1.symmetry, sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg
    
    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=True)
    
    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=False)

    def _mask_contract(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int, large_leg: bool
                       ) -> tuple[Data, ProductSpace, ProductSpace]:
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        if in_domain:
            assert mask.is_projection != large_leg
            mask_contr = 0
        else:
            assert mask.is_projection == large_leg
            mask_contr = 1
        #
        tensor_blocks = tensor.data.blocks
        tensor_block_inds = tensor.data.block_inds
        tensor_block_inds_contr = tensor_block_inds[:, leg_idx:leg_idx + 1]
        #
        mask_blocks = mask.data.blocks
        mask_block_inds = mask.data.block_inds
        mask_block_inds_contr = mask_block_inds[:, mask_contr:mask_contr + 1]
        #
        # sort by the contracted rows
        if leg_idx != tensor.num_legs - 1:  # otherwise, if leg_idx == -1, the tensor_block_inds_contr are sorted
            sort = np.lexsort(tensor_block_inds_contr.T)
            tensor_blocks = [tensor_blocks[i] for i in sort]
            tensor_block_inds = tensor_block_inds[sort]
            tensor_block_inds_contr = tensor_block_inds_contr[sort]
        if mask_contr == 0:  # otherwise it is already sorted
            sort = np.lexsort(mask_block_inds_contr.T)
            mask_blocks = [mask_blocks[i] for i in sort]
            mask_block_inds = mask_block_inds[sort]
            mask_block_inds_contr = mask_block_inds_contr[sort]
        #
        res_blocks = []
        res_block_inds = []
        # need to iterate only over the "common" blocks. If either block is zero, so is the result
        for i, j in iter_common_sorted_arrays(tensor_block_inds_contr, mask_block_inds_contr, a_strict=False):
            if large_leg:
                block = self.block_backend.block_apply_mask(tensor_blocks[i], mask_blocks[j], ax=leg_idx)
            else:
                block = self.block_backend.block_enlarge_leg(tensor_blocks[i], mask_blocks[j], axis=leg_idx)
            block_inds = tensor_block_inds[i].copy()
            block_inds[leg_idx] = mask_block_inds[j, 1 - mask_contr]
            res_blocks.append(block)
            res_block_inds.append(block_inds)
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, tensor.num_legs), int)
        # OPTIMIZE (JU) block_inds might actually be sorted but i am not sure right now
        data = AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=False)
        #
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.spaces[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            domain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.spaces[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            codomain = ProductSpace(spaces, symmetry=tensor.symmetry, backend=self)
        return data, codomain, domain
    
    def mask_dagger(self, mask: Mask) -> MaskData:
        # the legs swap between domain and codomain. need to swap the two columns of block_inds.
        # since both columns are unique and ascending, the resulting block_inds are still sorted.
        block_inds = mask.data.block_inds[:, ::-1]
        return AbelianBackendData(dtype=mask.dtype, blocks=mask.data.blocks, block_inds=block_inds,
                                  is_sorted=True)
            
    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        basis_perm = large_leg._basis_perm
        blocks = []
        large_leg_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        for bi_large, (slc, sector) in enumerate(zip(large_leg.slices, large_leg.sectors)):
            block = a[slice(*slc)]
            mult = self.block_backend.block_sum_all(block)
            if mult == 0:
                continue
            blocks.append(block)
            large_leg_block_inds.append(bi_large)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                basis_perm_ranks.append(large_leg.basis_perm[slice(*slc)][block])
        
        if len(blocks) == 0:
            sectors = large_leg.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros(shape=(0, 2), dtype=int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), large_leg_block_inds])

        data = AbelianBackendData(dtype=Dtype.bool, blocks=blocks, block_inds=block_inds,
                                  is_sorted=True)
        small_leg = ElementarySpace(
            symmetry=large_leg.symmetry, sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg

    def mask_to_block(self, a: Mask) -> Block:
        large_leg = a.large_leg
        res = self.block_backend.zero_block([large_leg.dim], Dtype.bool)
        for block, b_i in zip(a.data.blocks, a.data.block_inds):
            if a.is_projection:
                bi_small, bi_large = b_i
            else:
                bi_large, bi_small = b_i
            res[slice(*large_leg.slices[bi_large])] = block
        return res

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        blocks = [self.block_backend.block_to_dtype(b, dtype) for b in a.data.blocks]
        large_leg_bi = a.data.block_inds[:, 1] if a.is_projection else a.data.block_inds[:, 0]
        block_inds = np.repeat(large_leg_bi[:, None], 2, axis=1)
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds)
    
    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        leg_in, perm1 = tens.codomain[0]._dual_space(return_perm=True)
        leg_out, perm2 = tens.domain[0]._dual_space(return_perm=True)
        # block_inds[:, 0] refers to tens.codomain[0]. Thus it should be permuted with perm1.
        # It ends up being result.domain[0] -> second column of block_inds
        block_inds = np.column_stack([
            inverse_permutation(perm2)[tens.data.block_inds[:, 1]],
            inverse_permutation(perm1)[tens.data.block_inds[:, 0]]
        ])
        data = AbelianBackendData(dtype=tens.dtype, blocks=tens.data.blocks, block_inds=block_inds,
                                  is_sorted=False)
        return leg_in, leg_out, data
        
    def mask_unary_operand(self, mask: Mask, func) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = mask.large_leg
        basis_perm = large_leg._basis_perm
        mask_blocks_inds = mask.data.blocks_inds
        mask_blocks = mask.data.blocks
        #
        blocks = []
        large_leg_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        #
        i = 0
        b_i = -1 if len(mask_blocks_inds) == 0 else mask_blocks_inds[i, 1]
        for sector_idx, (sector, slc) in enumerate(zip(large_leg.sectors, large_leg.slices)):
            if sector_idx == b_i:
                block = mask_blocks[i]
                i += 1
                if i >= len(mask_blocks_inds):
                    b_i = -1  # mask has no further blocks
                else:
                    b_i = mask_blocks_inds[i, 1]
            else:
                block = self.block_backend.zero_block([large_leg.multiplicities[sector_idx]], Dtype.bool)
            new_block = func(block)
            mult = self.block_backend.block_sum_all(new_block)
            if mult == 0:
                continue
            blocks.append(new_block)
            large_leg_block_inds.append(sector_idx)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                basis_perm_ranks.append(large_leg.basis_perm[slice(*slc)][new_block])
        if len(blocks) == 0:
            sectors = mask.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), large_leg_block_inds])
        data = AbelianBackendData(
            dtype=Dtype.bool, blocks=blocks, block_inds=block_inds, is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=mask.symmetry, sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg
    
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        if a == 0.:
            return self.zero_data(b.codomain, b.domain, b.dtype)
        blocks = [self.block_backend.block_mul(a, T) for T in b.data.blocks]
        if len(blocks) == 0:
            if isinstance(a, float):
                dtype = b.data.dtype
            else:
                dtype = b.data.dtype.to_complex
        else:
            dtype = self.block_backend.block_dtype(blocks[0])
        return AbelianBackendData(dtype, blocks, b.data.block_inds, is_sorted=True)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        block_norms = [self.block_backend.block_norm(b, order=2) for b in a.data.blocks]
        return np.linalg.norm(block_norms, ord=2)

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        l_a, N_a = a_block_inds.shape
        l_b, N_b = b_block_inds.shape
        K_a = a.num_codomain_legs
        # convert to common dtype
        res_dtype = Dtype.common(a.dtype, b.dtype)
        if a.dtype != res_dtype:
            a_blocks = [self.block_backend.block_to_dtype(T, res_dtype) for T in a_blocks]
        if b.dtype != res_dtype:
            b_blocks = [self.block_backend.block_to_dtype(T, res_dtype) for T in b_blocks]
        #
        grid = np.indices([l_a, l_b]).T.reshape(-1, 2)
        # grid is lexsorted, with rows as all combinations of a/b block indices.
        #
        res_block_inds = np.empty((l_a * l_b, N_a + N_b), dtype=int)
        res_block_inds[:, :K_a] =  a_block_inds[grid[:, 0], :K_a]
        res_block_inds[:, K_a:K_a+N_b] = b_block_inds[grid[:, 1]]
        res_block_inds[:, K_a+N_b:] = a_block_inds[grid[:, 0], K_a:]
        res_blocks = [self.block_backend.block_tensor_outer(a_blocks[i], b_blocks[j], K_a)
                      for i, j in grid]
        # res_block_inds are in general not sorted.
        #
        return AbelianBackendData(res_dtype, res_blocks, res_block_inds, is_sorted=False)

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, ProductSpace, ProductSpace]:
        N = tensor.num_legs
        K = tensor.num_codomain_legs
        idcs1 = []
        idcs2 = []
        opposite_sides = []  # if pairs[n] has one leg each in codomain and domain or if they are both on the same side
        for i1, i2 in pairs:
            idcs1.append(i1)
            idcs2.append(i2)
            opposite_sides.append((i1 < K) != (i2 < K))
        remaining = [n for n in range(N) if n not in idcs1 and n not in idcs2]
        #
        blocks = tensor.data.blocks
        block_inds_1 = tensor.data.block_inds[:, idcs1]
        block_inds_2 = tensor.data.block_inds[:, idcs2]
        block_inds_rem = tensor.data.block_inds[:, remaining]
        #
        # OPTIMIZE
        #   - avoid python loops / function calls
        #   - spaces could store (or cache!) the sector permutation between itself and its dual.
        #     this permutation could be applied to the block_inds and then we can compare on block_inds
        #     level again, without resorting to the sectors.
        #
        def on_diagonal(bi1, bi2):
            # given bi1==block_inds_1[n] and bi2==block_inds_2[n], return if blocks[n] is on the
            # diagonal and thus contributes to the trace, or not.
            for n, (i1, i2) in enumerate(zip(bi1, bi2)):
                if opposite_sides[n]:
                    # legs are the same -> can compare block_inds
                    if i1 != i2:
                        return False
                else:
                    # legs have opposite duality. need to compare sectors explicitly
                    sector1 = tensor.get_leg_co_domain(idcs1[n]).sectors[i1]
                    sector2 = tensor.get_leg_co_domain(idcs2[n]).sectors[i2]
                    if not np.all(sector1 == tensor.symmetry.dual_sector(sector2)):
                        return False
            return True
            
        #
        res_data = {}  # dictionary res_block_inds_row -> Block
        for block, i1, i2, bi_rem in zip(blocks, block_inds_1, block_inds_2, block_inds_rem):
            if not on_diagonal(i1, i2):
                continue
            bi_rem = tuple(bi_rem)
            block = self.block_backend.block_trace_partial(block, idcs1, idcs2, remaining)
            add_block = res_data.get(bi_rem, None)
            if add_block is not None:
                block = block + add_block
            res_data[bi_rem] = block
        res_blocks = list(res_data.values())

        if len(remaining) == 0:
            # scalar result
            if len(res_blocks) == 0:
                return tensor.dtype.zero_scalar, None, None
            elif len(res_blocks) == 1:
                return self.block_backend.block_item(res_blocks[0]), None, None
            raise RuntimeError  # by charge rule, should be impossible to get multiple blocks.
        
        if len(res_blocks) == 0:
            res_block_inds = np.zeros((0, len(remaining)), int)
        else:
            res_block_inds = np.array(list(res_data.keys()), int)
        data = AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=False)
        codomain = ProductSpace(
            [leg for n, leg in enumerate(tensor.codomain) if n in remaining],
            symmetry=tensor.symmetry, backend=self
        )
        domain = ProductSpace(
            [leg for n, leg in enumerate(tensor.domain) if N - 1 - n in remaining],
            symmetry=tensor.symmetry, backend=self
        )
        return data, codomain, domain
        

    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, ProductSpace, ProductSpace]:
        codomain_legs = []
        codomain_sector_perms = []
        for i in codomain_idcs:
            in_domain, co_domain_idx, _ = a._parse_leg_idx(i)
            if in_domain:
                # leg.sectors == duals(old_leg.sectors)[perm]
                # block with given sectors which had bi_old belonged to
                # old_leg.sectors[bi_old]
                # it now belongs to 
                # dual(old_leg.sectors[bi_old]) == dual(old_leg.sectors[perm[bi_new]]) == leg.sectors[bi_new]
                #  -> bi_old == perm[bi_new]
                leg, perm = a.domain[co_domain_idx]._dual_space(return_perm=True)
                perm = inverse_permutation(perm)
            else:
                leg = a.codomain[co_domain_idx]
                perm = None
            codomain_legs.append(leg)
            codomain_sector_perms.append(perm)
        codomain = ProductSpace(codomain_legs, symmetry=a.symmetry, backend=self)
        #
        domain_legs = []
        domain_sector_perms = []
        for i in domain_idcs:
            in_domain, co_domain_idx, _ = a._parse_leg_idx(i)
            if in_domain:
                leg = a.domain[co_domain_idx]
                perm = None
            else:
                leg, perm = a.codomain[co_domain_idx]._dual_space(return_perm=True)
                perm = inverse_permutation(perm)
            domain_legs.append(leg)
            domain_sector_perms.append(perm)
        domain = ProductSpace(domain_legs, symmetry=a.symmetry, backend=self)
        #
        axes_perm = [*codomain_idcs, *reversed(domain_idcs)]
        sector_perms = [*codomain_sector_perms, *reversed(domain_sector_perms)]
        blocks = [self.block_backend.block_permute_axes(block, axes_perm) for block in a.data.blocks]
        block_inds = a.data.block_inds[:, axes_perm]
        for ax, sector_perm in enumerate(sector_perms):
            if sector_perm is None:
                continue
            block_inds[:, ax] = sector_perm[block_inds[:, ax]]
        data = AbelianBackendData(a.dtype, blocks=blocks, block_inds=block_inds, is_sorted=False)
        return data, codomain, domain

    def qr(self, a: SymmetricTensor, new_leg: ElementarySpace) -> tuple[Data, Data]:
        assert a.num_codomain_legs == 1 == a.num_domain_legs  # since self.can_decompose_tensors is False
        q_blocks = []
        r_blocks = []
        q_block_inds = []
        r_block_inds = []
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            # due to the loop setup we have:
            #   a.codomain.sectors[j] == new_leg.sectors[n]
            #   a.domain.sectors[k] == new_leg.sectors[n]
            if i < len(a_block_inds) and a_block_inds[i, 0] == j:
                # we have a block for that sector -> decompose it
                q, r = self.block_backend.matrix_qr(a_blocks[i], full=False)
                q_blocks.append(q)
                r_blocks.append(r)
                r_block_inds.append([n, k])
                i += 1
            else:
                # we do not have a block for that sector
                # => R_block == 0 and we dont even set it.
                # can choose arbitrary blocks for q, as long as they are isometric
                new_leg_dim = new_leg.multiplicities[n]
                eye = self.block_backend.eye_matrix(a.codomain.multiplicities[j], a.dtype)
                q_blocks.append(eye[:, :new_leg_dim])
            q_block_inds.append([j, n])
        #
        if len(q_blocks) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds, int)
        if len(r_blocks) == 0:
            r_block_inds = np.zeros((0, 2), int)
        else:
            r_block_inds = np.array(r_block_inds, int)
        #
        q_data = AbelianBackendData(a.dtype, q_blocks, q_block_inds, is_sorted=True)
        r_data = AbelianBackendData(a.dtype, r_blocks, r_block_inds, is_sorted=True)
        return q_data, r_data

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        numbers = []
        block_inds = tensor.data.block_inds
        blocks = tensor.data.blocks
        i = 0
        for j, m in enumerate(tensor.leg.multiplicities):
            if j == block_inds[i, 0]:
                block = blocks[i]
                i += 1
            else:
                block = self.block_backend.zero_block(m, dtype=tensor.dtype)
            numbers.append(block_func(block))
        return func(numbers)

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        a_block_inds_cont = a_block_inds[:, leg:leg+1]
        if leg == a.num_legs - 1:
            # due to lexsort(a_block_inds.T), a_block_inds_cont is already sorted in this case
            pass
        else:
            sort = np.lexsort(a_block_inds_cont.T)
            a_blocks = [a_blocks[i] for i in sort]
            a_block_inds = a_block_inds[sort, :]
            a_block_inds_cont = a_block_inds_cont[sort, :]
        b_block_inds = b.data.block_inds
        #
        # ensure common dtype
        common_dtype = a.dtype.common(b.dtype)
        if a.data.dtype != common_dtype:
            a_blocks = [self.block_backend.block_to_dtype(block, common_dtype) for block in a_blocks]
        if b.data.dtype != common_dtype:
            b_blocks = [self.block_backend.block_to_dtype(block, common_dtype) for block in b_blocks]
        #
        # only need to iterate over common blocks, the non-common multiply to 0.
        # note: unlike the tdot implementation, we do not combine and reshape here.
        #       this is because we know the result will have the same block-structure as `a`, and
        #       we only need to scale the blocks on one axis, not perform a general tensordot.
        #       but this also means that we may encounter duplicates in a_block_inds_cont,
        #       i.e. multiple blocks of `a` which have the same sector on the leg to be scaled.
        res_blocks = []
        res_block_inds = []
        for i, j in iter_common_sorted_arrays(a_block_inds_cont, b_block_inds[:, :1], a_strict=False):
            res_blocks.append(self.block_backend.block_scale_axis(a_blocks[i], b_blocks[j], axis=leg))
            res_block_inds.append(a_block_inds[i])
        #
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, a.num_legs), int)
        #
        return AbelianBackendData(common_dtype, res_blocks, res_block_inds, is_sorted=False)
   
    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], codomain_split: list[int],
                   domain_split: list[int], new_codomain: ProductSpace, new_domain: ProductSpace
                   ) -> Data:
        raise NotImplementedError  # TODO not yet reviewed
        # TODO (JH) below, we implement it by first generating the block_inds of the splitted tensor and
        # then extract subblocks from the original one.
        # Why not go the other way around and implement
        # block_views = self.block_backend.block_split(block, block_sizes, axis) similar as np.array_split()
        # and call that for each axis to be split for each block?
        # we do a lot of numpy index tricks below, but does that really save time for splitting?
        # block_split should just be views anyways, not data copies?

        if len(a.data.blocks) == 0:
            return self.zero_data(new_codomain, new_domain, a.data.dtype)
        n_split = len(leg_idcs)
        # TODO dont use legs! use conventional_leg_order / domain / codomain
        product_spaces = [a.legs[i] for i in leg_idcs]
        res_num_legs = new_codomain.num_spaces + new_domain.num_spaces

        old_blocks = a.data.blocks
        old_block_inds = a.data.block_inds

        map_slices_beg = np.zeros((len(old_blocks), n_split), int)
        map_slices_shape = np.zeros((len(old_blocks), n_split), int)  # = end - beg
        for j, product_space in enumerate(product_spaces):
            block_inds_j = old_block_inds[:, leg_idcs[j]]
            block_ind_map_slices = product_space.get_metadata('_block_ind_map_slices', backend=self)
            map_slices_beg[:, j] = block_ind_map_slices[block_inds_j]
            sizes = block_ind_map_slices[1:] - block_ind_map_slices[:-1]
            map_slices_shape[:, j] = sizes[block_inds_j]
        new_data_blocks_per_old_block = np.prod(map_slices_shape, axis=1)

        old_rows = np.concatenate([np.full((s,), i, int) for i, s in enumerate(new_data_blocks_per_old_block)])
        res_num_blocks = len(old_rows)

        map_rows = []
        for beg, shape in zip(map_slices_beg, map_slices_shape):
            map_rows.append(np.indices(shape, int).reshape(n_split, -1).T + beg[np.newaxis, :])
        map_rows = np.concatenate(map_rows, axis=0)  # shape (res_num_blocks, n_split)

        # generate new block_inds and figure out slices within old blocks to be extracted
        new_block_inds = np.empty((res_num_blocks, res_num_legs), dtype=int)
        old_block_beg = np.zeros((res_num_blocks, a.num_legs), dtype=int)
        old_block_shapes = np.empty((res_num_blocks, a.num_legs), dtype=int)
        shift = 0  #  = i - k for indices below
        j = 0  # index within product_spaces
        for i in range(a.num_legs):  # i = index in old tensor
            if i in leg_idcs:
                product_space = product_spaces[j]  # = a.legs[i]
                k = i + shift  # = index where split legs begin in new tensor
                k2 = k + len(product_space.spaces)  # = until where spaces go in new tensor
                _block_ind_map = product_space.get_metadata('_block_ind_map', backend=self)[map_rows[:, j], :]
                new_block_inds[:, k:k2] = _block_ind_map[:, 2:-1]
                old_block_beg[:, i] = _block_ind_map[:, 0]
                old_block_shapes[:, i] = _block_ind_map[:, 1] - _block_ind_map[:, 0]
                shift += len(product_space.spaces) - 1
                j += 1
            else:
                new_block_inds[:, i + shift] = nbi = old_block_inds[old_rows, i]
                # TODO dont use legs! use conventional_leg_order / domain / codomain
                old_block_shapes[:, i] = a.legs[i].multiplicities[nbi]
        # sort new_block_inds
        # OPTIMIZE (JU) could also skip sorting here and put is_sorted=False in AbelianBackendData(..) below?
        sort = np.lexsort(new_block_inds.T)
        new_block_inds = new_block_inds[sort, :]
        old_block_beg = old_block_beg[sort]
        old_block_shapes = old_block_shapes[sort]
        old_rows = old_rows[sort]

        new_block_shapes = np.empty((res_num_blocks, res_num_legs), dtype=int)
        for i, leg in conventional_leg_order(new_codomain, new_domain):
            new_block_shapes[:, i] = leg.multiplicities[new_block_inds[:, i]]

        # the actual loop to split the blocks
        new_blocks = []
        for i in range(res_num_blocks):
            old_block = old_blocks[old_rows[i]]
            slices = tuple(slice(b, b + s) for b, s in zip(old_block_beg[i], old_block_shapes[i]))
            new_block = old_block[slices]
            new_blocks.append(self.block_backend.block_reshape(new_block, new_block_shapes[i]))

        return AbelianBackendData(a.data.dtype, new_blocks, new_block_inds, is_sorted=True)

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        n_legs = a.num_legs
        if len(a.data.blocks) == 0:
            block_inds = np.zeros([0, n_legs - len(idcs)], dtype=int)
            return AbelianBackendData(a.data.dtype, [], block_inds, is_sorted=True)
        blocks = [self.block_backend.block_squeeze_legs(b, idcs) for b in a.data.blocks]
        keep = np.ones(n_legs, dtype=bool)
        keep[idcs] = False
        block_inds = a.data.block_inds[:, keep]
        return AbelianBackendData(a.data.dtype, blocks, block_inds, is_sorted=True)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry.is_abelian and symmetry.braiding_style == BraidingStyle.bosonic

    def svd(self, a: SymmetricTensor, new_leg: ElementarySpace, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        assert a.num_codomain_legs == 1 == a.num_domain_legs  # since self.can_decompose_tensors is False
        u_blocks = []
        s_blocks = []
        vh_blocks = []
        s_block_inds = []
        u_block_inds = []
        vh_block_inds = []
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(iter_common_sorted_arrays(a.codomain.sectors, a.domain.sectors)):
            # due to the loop setup we have:
            #   a.codomain.sectors[j] == new_leg.sectors[n]
            #   a.domain.sectors[k] == new_leg.sectors[n]
            if i < len(a_block_inds) and a_block_inds[i, 0] == j:
                # we have a block for that sector -> decompose it
                u, s, vh = self.block_backend.matrix_svd(a_blocks[i], algorithm=algorithm)
                u_blocks.append(u)
                s_blocks.append(s)
                vh_blocks.append(vh)
                s_block_inds.append(n)
                i += 1
            else:
                # we do not have a block for that sector.
                #  => S_block == 0, dont even set it.
                #  can choose arbitrary blocks for u and vh, as long as they are isometric / orthogonal
                new_leg_dim = new_leg.multiplicities[n]
                eye_u = self.block_backend.eye_matrix(a.codomain.multiplicities[j], a.dtype)
                u_blocks.append(eye_u[:, :new_leg_dim])
                eye_v = self.block_backend.eye_matrix(a.domain.multiplicities[j], a.dtype)
                vh_blocks.append(eye_v[:new_leg_dim, :])
            u_block_inds.append([j, n])
            vh_block_inds.append([n, k])

        if len(s_blocks) == 0:
            # TODO warn or error??
            s_block_inds = np.zeros([0, 2], int)
        else:
            s_block_inds = np.repeat(np.array(s_block_inds, int)[:, None], 2, axis=1)
        if len(u_blocks) == 0:
            u_block_inds = vh_block_inds = np.zeros([0, 2], int)
        else:
            u_block_inds = np.array(u_block_inds, int)
            vh_block_inds = np.array(vh_block_inds, int)

        # for all block_inds, the last column is sorted and duplicate-free,
        # thus the block_inds are np.lexsort( .T)-ed
        u_data = AbelianBackendData(a.dtype, u_blocks, u_block_inds, is_sorted=True)
        s_data = AbelianBackendData(a.dtype.to_real, s_blocks, s_block_inds, is_sorted=True)
        vh_data = AbelianBackendData(a.dtype, vh_blocks, vh_block_inds, is_sorted=True)
        return u_data, s_data, vh_data

    def state_tensor_product(self, state1: Block, state2: Block, prod_space: ProductSpace):
        #TODO clearly define what this should do in tensors.py first!
        raise NotImplementedError('state_tensor_product not implemented')

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        res = self.block_backend.zero_block(a.shape, a.data.dtype)
        for block, b_i in zip(a.data.blocks, a.data.block_inds):
            slices = tuple(slice(*leg.slices[i]) for i, leg in zip(b_i, conventional_leg_order(a)))
            res[slices] = block
        return res

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError  # TODO not yet reviewed
        num_blocks = len(tensor.data.blocks)
        if num_blocks == 1:
            return tensor.data.blocks[0]
        elif num_blocks == 0:
            dim = tensor.legs[0]._non_dual_sector_multiplicity(tensor.symmetry.trivial_sector)
            return self.block_backend.zero_block(shape=[dim], dtype=tensor.data.dtype)
        raise ValueError  # this should not happen for single-leg tensors

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        # shallow copy if dtype stays same
        blocks = [self.block_backend.block_to_dtype(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, blocks, a.data.block_inds, is_sorted=True)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        K = a.num_codomain_legs
        res = a.data.dtype.zero_scalar
        for block, bi in zip(a_blocks, a_block_inds):
            bi_cod = bi[:K]
            bi_dom = bi[K:]
            if np.all(bi_cod == bi_dom[::-1]):
                res += self.block_backend.block_trace_full(block)
            # else: block is entirely off-diagonal and does not contribute to the trace
        return res

    def transpose(self, a: SymmetricTensor) -> tuple[Data, ProductSpace, ProductSpace]:
        return self.permute_legs(a,
                                 codomain_idcs=list(range(a.num_codomain_legs, a.num_legs)),
                                 domain_idcs=list(reversed(range(a.num_codomain_legs))),
                                 levels=None)

    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype
                  ) -> AbelianBackendData:
        block_inds = np.zeros((0, codomain.num_spaces + domain.num_spaces), dtype=int)
        return AbelianBackendData(dtype, blocks=[], block_inds=block_inds, is_sorted=True)

    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        return AbelianBackendData(dtype, blocks=[], block_inds=np.zeros((0, 2), dtype=int),
                                  is_sorted=True)
    
    def zero_mask_data(self, large_leg: Space) -> MaskData:
        return AbelianBackendData(Dtype.bool, blocks=[], block_inds=np.zeros((0, 2), dtype=int),
                                  is_sorted=True)

    # OPTIONAL OVERRIDES
    
    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[Space]
                     ) -> tuple[SectorArray, ndarray, dict]:
        r"""
        The abelian backend adds the following metadata:
            _strides : 1D numpy array of int
                F-style strides for the shape ``tuple(space.num_sectors for space in spaces)``.
                This allows one-to-one mapping between multi-indices (one block_ind per space) to a single index.
            _block_ind_map_slices : 1D numpy array of int
                Slices for embedding the unique fused sectors in the sorted list of all fusion outcomes.
                Shape is ``(K,)`` where ``K == product_space.num_sectors + 1``.
                Fusing all sectors of all spaces and sorting the outcomes gives a list which
                contains (in general) duplicates.
                The slice ``_block_ind_map_slices[n]:_block_ind_map_slices[n + 1]`` within this
                sorted list contains the same entry, namely ``product_space.sectors[n]``.
            _block_ind_map : 2D numpy array of int
                Map for the embedding of uncoupled to coupled indices, see notes below.
                Shape is ``(M, N)`` where ``M`` is the number of combinations of sectors,
                i.e. ``M == prod(s.num_sectors for s in spaces)`` and ``N == 3 + len(spaces)``.

        Notes
        -----
        For ``np.reshape``, taking, for example,  :math:`i,j,... \rightarrow k` amounted to
        :math:`k = s_1*i + s_2*j + ...` for appropriate strides :math:`s_1,s_2`.
        
        In the charged case, however, we want to block :math:`k` by charge, so we must
        implicitly permute as well.  This reordering is encoded in `_block_ind_map` as follows.
        
        Each block index combination :math:`(i_1, ..., i_{nlegs})` of the `nlegs=len(spaces)`
        input `Space`s will end up getting placed in some slice :math:`a_j:a_{j+1}` of the
        resulting `ProductSpace`. Within this slice, the data is simply reshaped in usual row-major
        fashion ('C'-order), i.e., with strides :math:`s_1 > s_2 > ...` given by the block size.
        
        It will be a subslice of a new total block in the `ProductSpace` labelled by block index
        :math:`J`. We fuse charges according to the rule::
        
            ProductSpace.sectors[J] = fusion_outcomes(*[l.sectors[i_l]
                for l, i_l, l in zip(incoming_block_inds, spaces)])
                
        Since many charge combinations can fuse to the same total charge,
        in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
        charge block :math:`J` in the `ProductSpace`.
        
        The rows of `_block_ind_map` are precisely the collections of
        ``[b_{J,k}, b_{J,k+1}, i_1, . . . , i_{nlegs}, J]``.
        Here, :math:`b_k:b_{k+1}` denotes the slice of this block index combination *within*
        the total block `J`, i.e., ``b_{J,k} = a_j - self.slices[J]``.
        
        The rows of `_block_ind_map` are lex-sorted first by ``J``, then the ``i``.
        Each ``J`` will have multiple rows, and the order in which they are stored in `block_inds`
        is the order the data is stored in the actual tensor.
        Thus, ``_block_ind_map`` might look like ::
        
            [ ...,
            [ b_{J,k},   b_{J,k+1},  i_1,    ..., i_{nlegs}   , J,   ],
            [ b_{J,k+1}, b_{J,k+2},  i'_1,   ..., i'_{nlegs}  , J,   ],
            [ 0,         b_{J,1},    i''_1,  ..., i''_{nlegs} , J + 1],
            [ b_{J,1},   b_{J,2},    i'''_1, ..., i'''_{nlegs}, J + 1],
            ...]

        """
        
        # this function heavily uses numpys advanced indexing, for details see
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

        if len(spaces) == 0:
            metadata = dict(
                _strides=np.ones((0,), int),
                fusion_outcomes_sort=np.array([0], dtype=int),
                _block_ind_map_slices=np.array([0, 1], int),
                _block_ind_map=np.ones((0, 3), int),
            )
            sectors = symmetry.trivial_sector[None, :]
            multiplicities = [1]
            return sectors, multiplicities, metadata

        metadata = {}
        
        num_spaces = len(spaces)
        spaces_num_sectors = tuple(space.num_sectors for space in spaces)
        metadata['_strides'] = make_stride(spaces_num_sectors, cstyle=False)
        # (save strides for :meth:`product_space_map_incoming_block_inds`)

        # create a grid to select the multi-index sector
        grid = np.indices(spaces_num_sectors, np.intp)
        # grid is an array with shape ``(num_spaces, *spaces_num_sectors)``,
        # with grid[li, ...] = {np.arange(space_block_numbers[li]) increasing in li-th direction}
        # collapse the different directions into one.
        grid = grid.T.reshape(-1, num_spaces)  # *this* is the actual `reshaping`
        # *rows* of grid are now all possible combinations of qindices.
        # transpose before reshape ensures that grid.T is np.lexsort()-ed

        nblocks = grid.shape[0]  # number of blocks in ProductSpace = np.product(spaces_num_sectors)
        # this is different from num_sectors
        
        # determine _block_ind_map -- it's essentially the grid.
        _block_ind_map = np.zeros((nblocks, 3 + num_spaces), dtype=np.intp)
        _block_ind_map[:, 2:-1] = grid  # possible combinations of indices

        # the block size for given (i1, i2, ...) is the product of ``multiplicities[il]``
        # advanced indexing:
        # ``grid.T[li]`` is a 1D array containing the qindex `q_li` of leg ``li`` for all blocks
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(spaces, grid.T)],
                                 axis=0)
        # _block_ind_map[:, :2] and [:, -1] is initialized after sort/bunch.

        # calculate new non-dual sectors
        sectors = symmetry.multiple_fusion_broadcast(
            *(s.sectors[gr] for s, gr in zip(spaces, grid.T))
        )

        # sort (non-dual) charge sectors.
        fusion_outcomes_sort = np.lexsort(sectors.T)
        _block_ind_map = _block_ind_map[fusion_outcomes_sort]
        sectors = sectors[fusion_outcomes_sort]
        multiplicities = multiplicities[fusion_outcomes_sort]
        metadata['fusion_outcomes_sort'] = fusion_outcomes_sort

        slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
        _block_ind_map[:, 0] = slices[:-1]  # start with 0
        _block_ind_map[:, 1] = slices[1:]

        # bunch sectors with equal charges together
        diffs = find_row_differences(sectors, include_len=True)
        metadata['_block_ind_map_slices'] = diffs
        slices = slices[diffs]
        multiplicities = slices[1:] - slices[:-1]
        diffs = diffs[:-1]

        sectors = sectors[diffs]

        new_block_ind = np.zeros(len(_block_ind_map), dtype=np.intp) # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        _block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        _block_ind_map[:, :2] -= slices[new_block_ind][:, np.newaxis]
        metadata['_block_ind_map'] = _block_ind_map
        
        return sectors, multiplicities, metadata

    def get_leg_metadata(self, leg: Space) -> dict:
        if isinstance(leg, ProductSpace):
            # TODO / OPTIMIZE write a version that just calculates the metadata?
            _, _, metadata = self._fuse_spaces(symmetry=leg.symmetry, spaces=leg.spaces)
            return metadata
        return {}
        
    # INTERNAL HELPERS
    
    def product_space_map_incoming_block_inds(self, space: ProductSpace, incoming_block_inds):
        """Map incoming block indices to indices of :attr:`_block_ind_map`.

        Needed for `combine_legs`.

        Parameters
        ----------
        space : ProductSpace
            The ProductSpace which indices are to be mapped
        incoming_block_inds : 2D array
            Rows are block indices :math:`(i_1, i_2, ... i_{nlegs})` for incoming legs.

        Returns
        -------
        block_inds: 1D array
            For each row j of `incoming_block_inds` an index `J` such that
            ``self.metadata['_block_ind_map'][J, 2:-1] == block_inds[j]``.
        """
        assert incoming_block_inds.shape[1] == len(space.spaces)
        # calculate indices of _block_ind_map by using the appropriate strides
        strides = space.get_metadata('_strides', backend=self)
        inds_before_perm = np.sum(incoming_block_inds * strides[np.newaxis, :], axis=1)
        # now permute them to indices in _block_ind_map
        return inverse_permutation(space.fusion_outcomes_sort)[inds_before_perm]
