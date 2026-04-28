"""Backends for abelian group symmetries.

Changes compared to old np_conserved:

- replace `ChargeInfo` by subclasses of `AbelianGroup` (or `ProductSymmetry`)
- replace `LegCharge` by `ElementarySpace` and `LegPipe` by `AbelianLegPipe`. Changed class hierarchy!
- standard `Tensor` have qtotal=0, only ChargedTensor can have non-zero qtotal
- relabeling:
    - `Array.qdata`, "qind" and "qindices" to `AbelianBackendData.block_inds` and "block indices"
    - `LegPipe.qmap` to `AbelianLegPipe.block_ind_map` (with changed column order, and no longer lexsorted!!!)
    - `LegPipe` now forms the sector combinations in either C-style or F-style order
       depending on `AbelianLegPipe.combine_cstyle`. Before it was F-style always.
    - `LegPipe._perm` now is roughly covered by `AbelianLegPipe.fusion_outcomes_sort`
    - `AbelianLegPipe` now has a consistent `basis_perm`.
    - `LegCharge.get_block_sizes()` is just `Space.multiplicities`
- keep spaces "sorted" and "bunched",
  i.e. do not support legs with smaller blocks to effectively allow block-sparse tensors with
  smaller blocks than dictated by symmetries (which we actually have in H_MPO on the virtual legs...)
  In turn, ElementarySpace saves a `_perm` used to sort the originally passed `sectors`.
- keep `block_inds` sorted (i.e. no arbitrary gauge permutation in block indices)

.. _abelian_backend__blocks:

Blocks
------
TODO elaborate about blocks, block_inds etc
TODO link to this section from appropriate places

"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ..block_backends import Block
from ..block_backends.dtypes import Dtype
from ..symmetries import (
    AbelianLegPipe,
    BraidingStyle,
    ElementarySpace,
    FusionTree,
    Leg,
    LegPipe,
    Space,
    Symmetry,
    TensorProduct,
)
from ..tools.misc import (
    find_row_differences,
    inverse_permutation,
    iter_common_noncommon_sorted_arrays,
    iter_common_sorted,
    iter_common_sorted_arrays,
    list_to_dict_list,
    make_grid,
    make_stride,
    rank_data,
)
from ._backend import Data, DiagonalData, MaskData, TensorBackend, conventional_leg_order

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import DiagonalTensor, Mask, SymmetricTensor


def _valid_block_inds(codomain: TensorProduct, domain: TensorProduct):
    # OPTIMIZE: this is brute-force going through all possible combinations of block indices
    #           spaces are sorted, so we can probably reduce that search space quite a bit...
    symmetry = codomain.symmetry
    grid = make_grid([s.num_sectors for s in conventional_leg_order(codomain, domain)], cstyle=False)
    codomain_coupled = symmetry.multiple_fusion_broadcast(
        *(space.sector_decomposition[i] for space, i in zip(codomain.factors, grid.T))
    )
    domain_coupled = symmetry.multiple_fusion_broadcast(
        *(space.sector_decomposition[i] for space, i in zip(domain.factors, grid.T[::-1]))
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
    device : str
        The device on which the blocks are currently stored.
        We currently only support tensors which have all blocks on a single device.
        Should be the device returned by :func:`BlockBackend.as_device`.
    blocks : list of block
        A list of blocks containing the actual entries of the tensor.
        Leg order is ``[*codomain, *reversed(domain()]``, like ``Tensor.legs``.
    block_inds : 2D ndarray
        A 2D array of positive integers with shape (len(blocks), num_legs).
        The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of ``leg``,
        that is to ``leg.sector_decomposition[block_inds[n, m]]``, where::

            leg == (codomain.spaces[m] if m < len(codomain) else domain.spaces[-1 - m])
                == tensor.get_leg_co_domain(m)

        Thus, the columns of `block_inds` follow the same ordering convention as :attr:`Tensor.legs`.
        By convention, we store `blocks` and `block_inds` such that ``np.lexsort(block_inds.T)``
        is sorted.

    Parameters
    ----------
    dtype, device, blocks, block_inds
        like attributes above, but not necessarily sorted
    is_sorted : bool
        If ``False`` (default), we permute `blocks` and `block_inds` according to
        ``np.lexsort(block_inds.T)``.
        If ``True``, we assume they are sorted *without* checking.

    """

    def __init__(self, dtype: Dtype, device: str, blocks: list[Block], block_inds: ndarray, is_sorted: bool = False):
        if not is_sorted:
            perm = np.lexsort(block_inds.T)
            block_inds = block_inds[perm, :]
            blocks = [blocks[n] for n in perm]
        self.dtype = dtype
        self.device = device
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
        """Get the block at given block indices.

        Return the block in :attr:`blocks` matching the given block_inds,
        i.e. `self.blocks[n]` such that `all(self.block_inds[n, :] == blocks_inds)`
        or None if no such block exists
        """
        block_num = self.get_block_num(block_inds)
        return None if block_num is None else self.blocks[block_num]

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.block_inds, subpath + 'block_inds')
        hdf5_saver.save(self.blocks, subpath + 'blocks')
        hdf5_saver.save(self.dtype.to_numpy_dtype(), subpath + 'dtype')
        hdf5_saver.save(self.device, subpath + 'device')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.block_inds = hdf5_loader.load(subpath + 'block_inds')
        obj.blocks = hdf5_loader.load(subpath + 'blocks')
        obj.device = hdf5_loader.load(subpath + 'device')
        dt = hdf5_loader.load(subpath + 'dtype')
        obj.dtype = Dtype.from_numpy_dtype(dt)

        return obj


class AbelianBackend(TensorBackend):
    """Backend for Abelian group symmetries.

    Notes
    -----
    The data stored for the various tensor classes defined in ``cyten.tensors`` is::

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

    def test_tensor_sanity(self, a: SymmetricTensor | DiagonalTensor, is_diagonal: bool):
        super().test_tensor_sanity(a, is_diagonal=is_diagonal)
        data: AbelianBackendData = a.data
        # check device and dtype
        assert a.device == data.device == self.block_backend.as_device(data.device)
        assert a.dtype == data.dtype
        # check leg types
        for l in (a.get_leg_co_domain(n) for n in range(a.num_legs)):
            assert isinstance(l, ElementarySpace), 'legs must be ElementarySpace'
            if isinstance(l, LegPipe):
                assert isinstance(l, AbelianLegPipe), 'pipes must be AbelianLegPipe'
            # recursion into nested pipes is handled via AbelianLegPipe.test_sanity(), which
            # is called via (co)domain.test_sanity() during Tensor.test_sanity()
        # check block_inds
        assert data.block_inds.shape == (len(data.blocks), a.num_legs)
        assert np.all(data.block_inds >= 0)
        assert np.all(data.block_inds < np.array([[leg.num_sectors for leg in conventional_leg_order(a)]]))
        assert np.all(np.lexsort(data.block_inds.T) == np.arange(len(data.blocks)))
        # check block_inds fulfill charge rule
        if is_diagonal:
            assert np.all(data.block_inds[:, 0] == data.block_inds[:, 1])
        for inds in data.block_inds:
            codomain_coupled = a.symmetry.multiple_fusion(
                *(leg.sector_decomposition[i] for leg, i in zip(a.codomain.factors, inds))
            )
            domain_coupled = a.symmetry.multiple_fusion(
                *(leg.sector_decomposition[i] for leg, i in zip(a.domain.factors, inds[::-1]))
            )
            assert np.all(codomain_coupled == domain_coupled)
        # check blocks and charge rule
        for block, b_i in zip(data.blocks, data.block_inds):
            if is_diagonal:
                shape = (a.leg.multiplicities[b_i[0]],)
            else:
                shape = tuple(leg.multiplicities[i] for leg, i in zip(conventional_leg_order(a), b_i))
            self.block_backend.test_block_sanity(
                block, expect_shape=shape, expect_dtype=a.dtype, expect_device=a.device
            )

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        data: AbelianBackendData = a.data
        # check device and dtype
        assert a.device == data.device == self.block_backend.as_device(data.device)
        assert a.dtype == data.dtype == Dtype.bool
        # check leg types
        assert all(isinstance(l, ElementarySpace) for l in [a.large_leg, a.small_leg])
        # check block_inds
        assert data.block_inds.shape == (len(data.blocks), a.num_legs)
        assert np.all(data.block_inds >= 0)
        assert np.all(data.block_inds < np.array([[leg.num_sectors for leg in conventional_leg_order(a)]]))
        assert np.all(np.lexsort(data.block_inds.T) == np.arange(len(data.blocks)))
        # check blocks and charge rule
        for block, block_inds in zip(data.blocks, data.block_inds):
            if a.is_projection:
                bi_small, bi_large = block_inds
            else:
                bi_large, bi_small = block_inds
            assert bi_large >= bi_small
            # check charge rule
            assert np.all(a.large_leg.sector_decomposition[bi_large] == a.small_leg.sector_decomposition[bi_small])
            # check blocks
            expect_len = a.large_leg.multiplicities[bi_large]
            expect_sum = a.small_leg.multiplicities[bi_small]
            assert expect_len > 0
            assert expect_sum > 0
            self.block_backend.test_block_sanity(
                block, expect_shape=(expect_len,), expect_dtype=Dtype.bool, expect_device=data.device
            )
            assert self.block_backend.sum_all(block) == expect_sum

    # OVERRIDES

    def make_pipe(self, legs: list[Leg], is_dual: bool, in_domain: bool, pipe: LegPipe | None = None) -> LegPipe:
        assert all(isinstance(l, ElementarySpace) for l in legs)  # OPTIMIZE rm check
        if isinstance(pipe, AbelianLegPipe):
            assert pipe.combine_cstyle == (not is_dual)
            assert pipe.is_dual == is_dual
            assert pipe.legs == legs
            return pipe
        return AbelianLegPipe(legs, is_dual=is_dual, combine_cstyle=not is_dual)

    # ABSTRACT METHODS

    def act_block_diagonal_square_matrix(
        self, a: SymmetricTensor, block_method: Callable[[Block], Block], dtype_map: Callable[[Dtype], Dtype] | None
    ) -> Data:
        leg = a.domain.factors[0]
        a_block_inds = a.data.block_inds
        all_block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        res_blocks = []
        for i, j in iter_common_noncommon_sorted_arrays(a_block_inds, all_block_inds):
            if i is None:
                # use that all_block_inds is just ascending -> all_block_inds[j, 0] == j
                block = self.block_backend.zeros(shape=[leg.multiplicities[j]] * 2, dtype=a.dtype)
            else:
                block = a.data.blocks[i]
            res_blocks.append(block_method(block))
        if dtype_map is None:
            dtype = a.dtype
        else:
            dtype = dtype_map(a.dtype)
        res_blocks = [self.block_backend.to_dtype(block, dtype) for block in res_blocks]
        return AbelianBackendData(dtype, a.data.device, res_blocks, all_block_inds, is_sorted=True)

    def add_trivial_leg(
        self,
        a: SymmetricTensor,
        legs_pos: int,
        add_to_domain: bool,
        co_domain_pos: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        blocks = [self.block_backend.add_axis(block, legs_pos) for block in a.data.blocks]
        block_inds = np.insert(a.data.block_inds, legs_pos, 0, axis=1)
        # since the new column is constant, block_inds are still sorted.
        return AbelianBackendData(a.data.dtype, a.data.device, blocks, block_inds, is_sorted=True)

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        for i, j in iter_common_noncommon_sorted_arrays(a.data.block_inds, b.data.block_inds):
            if j is None:
                if self.block_backend.max_abs(a_blocks[i]) > atol:
                    return False
            elif i is None:
                if self.block_backend.max_abs(b_blocks[j]) > atol:
                    return False
            else:
                if not self.block_backend.allclose(a_blocks[i], b_blocks[j], rtol=rtol, atol=atol):
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
            block = self.block_backend.apply_mask(tensor_blocks[i], mask_blocks[j], ax=0)
            res_blocks.append(block)
            res_block_inds.append(mask_block_inds[j, 0])
        if len(res_block_inds) > 0:
            res_block_inds = np.repeat(np.array(res_block_inds)[:, None], 2, axis=1)
        else:
            res_block_inds = np.zeros((0, 2), int)
        return AbelianBackendData(tensor.dtype, tensor.data.device, res_blocks, res_block_inds, is_sorted=True)

    def combine_legs(
        self,
        tensor: SymmetricTensor,
        leg_idcs_combine: list[list[int]],
        pipes: list[LegPipe],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        assert all(isinstance(p, AbelianLegPipe) for p in pipes), 'abelian backend requires ``AbelianLegPipe``s'
        num_result_legs = tensor.num_legs - sum(len(group) - 1 for group in leg_idcs_combine)
        old_blocks = tensor.data.blocks
        cstyles = []  # which combined legs are formed in C and F style

        # build new block_inds, compatible with old_blocks, but contain duplicates and are not sorted
        res_block_inds = np.empty((len(tensor.data.block_inds), num_result_legs), int)
        i = 0  # res_block_inds[:, :i] is already set
        j = 0  # old_block_inds[:, :j] are already considered
        map_inds = []
        for group, pipe in zip(leg_idcs_combine, pipes):
            # uncombined legs since last group: block_inds are simply unchanged
            num_uncombined = group[0] - j
            res_block_inds[:, i : i + num_uncombined] = tensor.data.block_inds[:, j : j + num_uncombined]
            i += num_uncombined
            j += num_uncombined
            # current combined group
            in_domain = group[0] >= tensor.num_codomain_legs
            cstyles.append(pipe.combine_cstyle != in_domain)
            block_inds = tensor.data.block_inds[:, group[0] : group[-1] + 1]
            if in_domain:
                # product space in the domain has opposite order of its spaces compared to the
                # convention in block_inds
                block_inds = block_inds[:, ::-1]
            # for each row of block_inds, find the corresponding row of pipe.block_ind_map
            multi_indices = np.sum(block_inds * pipe.sector_strides[None, :], axis=1)
            block_ind_map_rows = inverse_permutation(pipe.fusion_outcomes_sort)[multi_indices]
            map_inds.append(block_ind_map_rows)
            res_block_inds[:, i] = pipe.block_ind_map[block_ind_map_rows, -1]
            i += 1
            j += len(group)
        # trailing uncombined legs:
        res_block_inds[:, i:] = tensor.data.block_inds[:, j:]

        # sort the new block_inds
        sort = np.lexsort(res_block_inds.T)
        res_block_inds = res_block_inds[sort]
        old_blocks = [old_blocks[i] for i in sort]
        map_inds = [rows[sort] for rows in map_inds]

        # determine, for each old block, which slices of the new block it should occupy
        i = 0  # have already set info for new_legs[:i]
        j = 0  # have already considered old_legs[:j]
        block_slices = np.zeros((len(old_blocks), num_result_legs, 2), int)
        for group, pipe, block_ind_map_rows in zip(leg_idcs_combine, pipes, map_inds):
            # uncombined legs since last group: slice is all of 0:mult
            num_uncombined = group[0] - j
            for _ in range(num_uncombined):
                # block_slices[:, i, 0] = 0 is already set
                block_slices[:, i, 1] = tensor.get_leg_co_domain(j).multiplicities[res_block_inds[:, i]]
                i += 1
                j += 1
            # current combined group
            block_slices[:, i, :] = pipe.block_ind_map[block_ind_map_rows, :2]
            i += 1
            j += len(group)
        # trailing uncombined legs
        for _ in range(tensor.num_legs - j):
            # block_slices[:, i, 0] = 0 is already set
            block_slices[:, i, 1] = tensor.get_leg_co_domain(j).multiplicities[res_block_inds[:, i]]
            i += 1
            j += 1

        # identify the duplicates in res_block_inds
        # all those old_blocks are embedded into a single new block
        diffs = find_row_differences(res_block_inds, include_len=True)  # includes both 0 and len, to have slices later

        # build the new blocks
        res_num_blocks = len(diffs) - 1
        res_block_inds = res_block_inds[diffs[:-1], :]
        res_block_shapes = np.zeros((res_num_blocks, num_result_legs), int)
        for i, leg in enumerate(conventional_leg_order(new_codomain, new_domain)):
            res_block_shapes[:, i] = leg.multiplicities[res_block_inds[:, i]]
        res_blocks = []
        for shape, start, stop in zip(res_block_shapes, diffs[:-1], diffs[1:]):
            new_block = self.block_backend.zeros(shape, dtype=tensor.dtype, device=tensor.device)
            for row in range(start, stop):
                slices = tuple(slice(b, e) for (b, e) in block_slices[row])
                new_block[slices] = self.block_backend.combine_legs(old_blocks[row], leg_idcs_combine, cstyles)
            res_blocks.append(new_block)

        # we lexsort( .T)-ed res_block_inds while it still had duplicates, and then indexed by diffs,
        # which is sorted and thus preserves lexsort( .T)-ing of res_block_inds
        return AbelianBackendData(tensor.dtype, tensor.data.device, res_blocks, res_block_inds, is_sorted=True)

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        if a.num_codomain_legs == 0 and b.num_domain_legs == 0:
            return self.inner(a, b, do_dagger=False)
        if a.num_domain_legs == 0:
            return self._compose_no_contraction(a, b)
        return self._compose_worker(a.data, b.data, a.codomain, b.codomain, b.domain)

    def _compose_worker(
        self,
        a_data: Data,
        b_data: Data,
        new_codomain: TensorProduct,
        contr_spaces: list[Space] | TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        """See compose docstring.

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

        Step 1) is implemented in ``cyten.tdot`` and is already done when we call ``cyten.compose``;
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
        ``i = a_block_inds_keep[s]`` and ``k1 = a_block_inds_contr[s]``.
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
        ``k1`` with nonzero :math:`A_{i,k1}`, and ``k2`` with nonzero :math:`B_{k2,j}`.

        How many multiplications :math:`A_{i,k} B_{k,j}` we actually have to perform
        depends on the sparseness. If ``k`` comes from a single leg, it is completely sorted
        by charges, so the 'sum' over ``k`` will contain at most one term!

        """
        symmetry = new_codomain.symmetry
        a_dtype = self.get_dtype_from_data(a_data)
        b_dtype = self.get_dtype_from_data(b_data)
        res_dtype = Dtype.common(a_dtype, b_dtype)
        if len(a_data.blocks) == 0 or len(b_data.blocks) == 0:
            # if there are no actual blocks to contract, we can directly return 0
            return self.zero_data(new_codomain, new_domain, res_dtype, device=a_data.device)

        # convert blocks to common dtype
        a_blocks = a_data.blocks
        if a_dtype != res_dtype:
            a_blocks = [self.block_backend.to_dtype(B, res_dtype) for B in a_blocks]
        b_blocks = b_data.blocks
        if b_dtype != res_dtype:
            b_blocks = [self.block_backend.to_dtype(B, res_dtype) for B in b_blocks]

        # need to contract the domain legs of a with the codomain legs of b.
        # due to the leg ordering

        # Deal with the columns of the block inds that are kept/contracted separately
        num_contr = len(contr_spaces)
        a_block_inds_keep, a_block_inds_contr = np.hsplit(a_data.block_inds, [new_codomain.num_factors])
        b_block_inds_contr, b_block_inds_keep = np.hsplit(b_data.block_inds, [num_contr])
        # Merge the block_inds on the contracted legs to a single column, using strides.
        # Note: The order in a.data.block_inds is opposite from the order in b.data.block_inds!
        #       I.e. a.data.block_inds[-1-n] and b.data.block_inds[n] describe one leg to contract
        # We choose F-style strides, by appearance in b.data.block_inds.
        # This guarantees that the b.data.block_inds sorting is preserved.
        # We do not care about the sorting of the a.data.block_inds, since we need to re-sort anyway,
        # to group by a_block_inds_keep.
        strides = make_stride([l.num_sectors for l in contr_spaces], cstyle=False)
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
        a_shape_keep = [self.block_backend.get_shape(blocks[0])[: new_codomain.num_factors] for blocks in a_blocks]
        b_shape_keep = [self.block_backend.get_shape(blocks[0])[num_contr:] for blocks in b_blocks]
        if new_codomain.num_factors == 0:
            # special case: reshape to vector.
            a_blocks = [[self.block_backend.reshape(B, (-1,)) for B in blocks] for blocks in a_blocks]
        else:
            a_blocks = [
                [self.block_backend.reshape(B, (np.prod(shape_keep), -1)) for B in blocks]
                for blocks, shape_keep in zip(a_blocks, a_shape_keep)
            ]
        # need to permute the leg order of one group of permuted legs.
        # OPTIMIZE does it matter, which?
        # choose to permute the legs of the b-blocks
        if new_domain.num_factors == 0:
            # special case: reshape to vector
            perm = list(reversed(range(new_domain.num_factors + num_contr)))
            b_blocks = [
                [self.block_backend.reshape(self.block_backend.permute_axes(B, perm), (-1,)) for B in blocks]
                for blocks in b_blocks
            ]
        else:
            perm = [*reversed(range(num_contr)), *range(num_contr, new_domain.num_factors + num_contr)]
            b_blocks = [
                [
                    self.block_backend.reshape(self.block_backend.permute_axes(B, perm), (-1, np.prod(shape_keep)))
                    for B in blocks
                ]
                for blocks, shape_keep in zip(b_blocks, b_shape_keep)
            ]

        # compute coupled sectors for all rows of the block inds // for all blocks
        if new_codomain.num_factors > 0:
            a_charges = symmetry.multiple_fusion_broadcast(
                *(leg.sector_decomposition[bi] for leg, bi in zip(new_codomain, a_block_inds_keep.T))
            )
        else:
            a_charges = np.repeat(symmetry.trivial_sector[None, :], len(a_block_inds_keep), axis=1)
        if new_domain.num_factors > 0:
            b_charges = symmetry.multiple_fusion_broadcast(
                *(leg.sector_decomposition[bi] for leg, bi in zip(new_domain, b_block_inds_keep[:, ::-1].T))
            )
        else:
            b_charges = np.repeat(symmetry.trivial_sector[None, :], len(b_block_inds_keep), axis=1)
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
                block = self.block_backend.reshape(block, a_shape_keep[row_a] + b_shape_keep[col_b])
                res_blocks.append(block)
                res_block_inds_a.append(a_block_inds_keep[row_a])
                res_block_inds_b.append(b_block_inds_keep[col_b])

        # finish up:
        if len(res_blocks) == 0:
            block_inds = np.zeros((0, new_codomain.num_factors + new_domain.num_factors), dtype=int)
        else:
            block_inds = np.hstack((res_block_inds_a, res_block_inds_b))
        return AbelianBackendData(res_dtype, a_data.device, blocks=res_blocks, block_inds=block_inds, is_sorted=True)

    def _compose_no_contraction(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Special case of :meth:`compose` where no legs are actually contracted.

        Note that this is not the same as :meth:`outer`, the resulting leg order is different.
        """
        res_dtype = a.data.dtype.common(b.data.dtype)
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        if a.data.dtype != res_dtype:
            a_blocks = [self.block_backend.to_dtype(T, res_dtype) for T in a_blocks]
        if b.data.dtype != res_dtype:
            b_blocks = [self.block_backend.to_dtype(T, res_dtype) for T in b_blocks]
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        l_a, num_legs_a = a_block_inds.shape
        l_b, num_legs_b = b_block_inds.shape
        grid = make_grid([len(a_block_inds), len(b_block_inds)], cstyle=False)
        # grid is lexsorted, with rows as all combinations of a/b block indices.
        res_block_inds = np.empty((l_a * l_b, num_legs_a + num_legs_b), dtype=int)
        res_block_inds[:, :num_legs_a] = a_block_inds[grid[:, 0]]
        res_block_inds[:, num_legs_a:] = b_block_inds[grid[:, 1]]
        res_blocks = [self.block_backend.outer(a_blocks[i], b_blocks[j]) for i, j in grid]

        # Since the grid was in F-style, and the a_block_inds, b_block_inds are sorted,
        # the res_block_inds are sorted.
        return AbelianBackendData(res_dtype, a.data.device, res_blocks, res_block_inds, is_sorted=True)

    def copy_data(self, a: SymmetricTensor | DiagonalTensor, device: str = None) -> Data | DiagonalData:
        blocks = [self.block_backend.copy_block(b, device=device) for b in a.data.blocks]
        if device is None:
            device = a.data.device
        else:
            device = self.block_backend.as_device(device)
        # OPTIMIZE do we need to copy the block_inds ??
        return AbelianBackendData(a.data.dtype, device, blocks, a.data.block_inds.copy(), is_sorted=True)

    def dagger(self, a: SymmetricTensor) -> Data:
        blocks = [self.block_backend.dagger(b) for b in a.data.blocks]
        block_inds = a.data.block_inds[:, ::-1]
        return AbelianBackendData(a.dtype, a.data.device, blocks=blocks, block_inds=block_inds)

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError('More than 1 block!')
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_backend.item(a.blocks[0])

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        if len(a.data.block_inds) < a.leg.num_sectors:
            # missing blocks are filled with False
            return False
        # now it is enough to check that all existing blocks are all-True
        return all(self.block_backend.block_all(b) for b in a.data.blocks)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return any(self.block_backend.block_any(b) for b in a.data.blocks)

    def diagonal_elementwise_binary(
        self, a: DiagonalTensor, b: DiagonalTensor, func, func_kwargs, partial_zero_is_zero: bool
    ) -> DiagonalData:
        # OPTIMIZE should we drop zero blocks after?
        leg = a.leg
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds

        blocks = []
        block_inds = []

        ia = 0  # next block of a to process
        # block_ind of that block => it belongs to leg.sector_decomposition[bi_a]
        bi_a = -1 if len(a_block_inds) == 0 else a_block_inds[ia, 0]
        ib = 0  # next block of b to process
        # block_ind of that block => it belongs to leg.sector_decomposition[bi_b]
        bi_b = -1 if len(b_block_inds) == 0 else b_block_inds[ib, 0]
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
                block_a = self.block_backend.zeros([mult], a.dtype)

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
                block_b = self.block_backend.zeros([mult], a.dtype)
            blocks.append(func(block_a, block_b, **func_kwargs))
            block_inds.append(i)

        if len(blocks) == 0:
            block_inds = np.zeros((0, 2), int)
            dtype = self.block_backend.get_dtype(
                func(
                    self.block_backend.ones_block([1], dtype=a.dtype), self.block_backend.ones_block([1], dtype=b.dtype)
                )
            )
        else:
            block_inds = np.repeat(np.array(block_inds)[:, None], 2, axis=1)
            dtype = self.block_backend.get_dtype(blocks[0])

        return AbelianBackendData(
            dtype=dtype, device=a.data.device, blocks=blocks, block_inds=block_inds, is_sorted=True
        )

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool) -> DiagonalData:
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
                    block = self.block_backend.zeros([a.leg.multiplicities[i]], dtype=a.dtype)
                else:
                    block = a_blocks[j]
                blocks.append(func(block, **func_kwargs))
        if len(blocks) == 0:
            example_block = func(self.block_backend.zeros([1], dtype=a.dtype))
            dtype = self.block_backend.get_dtype(example_block)
        else:
            dtype = self.block_backend.get_dtype(blocks[0])
        return AbelianBackendData(
            dtype=dtype, device=a.data.device, blocks=blocks, block_inds=block_inds, is_sorted=True
        )

    def diagonal_from_block(self, a: Block, co_domain: TensorProduct, tol: float) -> DiagonalData:
        leg = co_domain.factors[0]
        dtype = self.block_backend.get_dtype(a)
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        blocks = [a[slice(*leg.slices[i])] for i in block_inds[:, 0]]
        device = self.block_backend.get_device(a)
        return AbelianBackendData(dtype, device, blocks, block_inds, is_sorted=True)

    def diagonal_from_sector_block_func(self, func, co_domain: TensorProduct) -> DiagonalData:
        leg = co_domain.factors[0]
        block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        blocks = [func((mult,), coupled) for coupled, mult in zip(leg.sector_decomposition, leg.multiplicities)]
        if len(blocks) == 0:
            sample_block = func((1,), co_domain.symmetry.trivial_sector)
        else:
            sample_block = blocks[0]
        dtype = self.block_backend.get_dtype(sample_block)
        device = self.block_backend.get_device(sample_block)
        return AbelianBackendData(dtype=dtype, device=device, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, tol: float | None) -> DiagonalData:
        blocks = [self.block_backend.get_diagonal(block, tol) for block in a.data.blocks]
        return AbelianBackendData(a.dtype, a.data.device, blocks, a.data.block_inds, is_sorted=True)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        total_sum = a.data.dtype.zero_scalar
        for block in a.data.blocks:
            total_sum += self.block_backend.sum_all(block)
        return total_sum

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        res = self.block_backend.zeros([a.leg.dim], a.dtype)
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
            sectors.append(large_leg.defining_sectors[bi])
            multiplicities.append(self.block_backend.sum_all(diag_block))
            if basis_perm is not None:
                mask = self.block_backend.to_numpy(diag_block, bool)
                basis_perm_ranks.append(basis_perm[slice(*large_leg.slices[bi])][mask])

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
            dtype=Dtype.bool,
            device=tens.data.device,
            blocks=blocks,
            block_inds=np.array(block_inds, int),
            is_sorted=True,
        )
        small_leg = ElementarySpace(
            symmetry=tens.symmetry,
            defining_sectors=sectors,
            multiplicities=multiplicities,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        # OPTIMIZE copy needed?
        return tens.leg.dual, self.copy_data(tens)

    def eigh(
        self, a: SymmetricTensor, new_leg_dual: bool, sort: str = None
    ) -> tuple[DiagonalData, Data, ElementarySpace]:
        # in tensors.py, we do pre-processing such that the following holds:
        assert a.num_codomain_legs == 1 == a.num_domain_legs
        assert new_leg_dual == a.domain[0].is_dual  # such that we can use the same block_inds

        new_leg = a.domain.as_ElementarySpace(is_dual=new_leg_dual)

        a_block_inds = a.data.block_inds
        # for missing blocks, i.e. a zero block, the eigenvalues are zero, so we can just skip
        # adding that block to the eigenvalues.
        # for the eigenvectors, we choose the computational basis vectors, i.e. the matrix
        # representation within that block is the identity matrix.
        # we initialize all blocks to eye and override those where `a` has blocks.
        v_data = self.eye_data(a.domain, a.dtype, device=a.data.device)
        w_blocks = []
        for block, bi in zip(a.data.blocks, a_block_inds):
            vals, vects = self.block_backend.eigh(block, sort=sort)
            w_blocks.append(vals)
            v_data.blocks[bi[0]] = vects
        w_data = AbelianBackendData(
            dtype=a.dtype.to_real, device=a.data.device, blocks=w_blocks, block_inds=a_block_inds, is_sorted=True
        )
        return w_data, v_data, new_leg

    def eye_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> Data:
        # Note: the identity has the same matrix elements in all ONB, so ne need to consider
        #       the basis perms.
        # results[i1,...im,jm,...,j1] = delta_{i1,j1} ... delta{im,jm}
        # need exactly the "diagonal" blocks, where sector of i1 matches the one of j1 etc.
        # to guarantee sorting later, it is easier to generate the block inds of the domain
        domain_dims = [leg.num_sectors for leg in reversed(co_domain.factors)]
        domain_block_inds = np.indices(domain_dims).T.reshape(-1, co_domain.num_factors)
        block_inds = np.hstack([domain_block_inds[:, ::-1], domain_block_inds])
        # domain_block_inds is by construction np.lexsort( .T)-ed.
        # since the last co_domain.num_spaces columns of block_inds are already unique, the first
        # columns are not relevant to np.lexsort( .T)-ing, thus the block_inds above is sorted.
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i] for leg, i in zip(co_domain.factors, b_i)]
            blocks.append(self.block_backend.eye_block(shape, dtype, device=device))
        return AbelianBackendData(dtype, device, blocks, block_inds, is_sorted=True)

    def from_dense_block(
        self, a: Block, codomain: TensorProduct, domain: TensorProduct, tol: float
    ) -> AbelianBackendData:
        dtype = self.block_backend.get_dtype(a)
        device = self.block_backend.get_device(a)
        projected = self.block_backend.zeros(self.block_backend.get_shape(a), dtype=dtype)
        block_inds = _valid_block_inds(codomain, domain)
        blocks = []
        for b_i in block_inds:
            slices = tuple(slice(*leg.slices[i]) for i, leg in zip(b_i, conventional_leg_order(codomain, domain)))
            block = a[slices]
            blocks.append(block)
            projected[slices] = block
        if tol is not None:
            if self.block_backend.norm(a - projected) > tol * self.block_backend.norm(a):
                raise ValueError('Block is not symmetric up to tolerance.')
        return AbelianBackendData(dtype, device, blocks, block_inds, is_sorted=True)

    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        bi = leg.sector_decomposition_where(leg.symmetry.trivial_sector)
        return AbelianBackendData(
            dtype=self.block_backend.get_dtype(block),
            device=self.block_backend.get_device(block),
            blocks=[block],
            block_inds=np.array([[bi]]),
            is_sorted=True,
        )

    def from_grid(
        self,
        grid: list[list[SymmetricTensor | None]],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
        left_mult_slices: list[list[int]],
        right_mult_slices: list[list[int]],
        dtype: Dtype,
        device: str,
    ) -> Data:
        blocks = []
        block_inds = np.zeros((0, len(new_codomain) + len(new_domain)), dtype=int)
        codom_slcs = [slice(None)] * (len(new_codomain) - 1)
        dom_slcs = [slice(None)] * (len(new_domain) - 1)
        for i, row in enumerate(grid):
            for j, op in enumerate(row):
                if op is None:
                    continue
                for op_bi, op_block in zip(op.data.block_inds, op.data.blocks):
                    # all block inds apart from the ones for the row and col
                    # must be identical to the ones of op
                    left_sector = op.codomain[0].sector_decomposition[op_bi[0]]
                    left_ind = new_codomain[0].sector_decomposition_where(left_sector)
                    right_sector = op.domain[-1].sector_decomposition[op_bi[len(new_codomain)]]
                    right_ind = new_domain[-1].sector_decomposition_where(right_sector)
                    new_bi = [left_ind, *op_bi[1 : len(new_codomain)], right_ind, *op_bi[len(new_codomain) + 1 :]]
                    new_bi = np.array(new_bi, dtype=int)

                    # find block or create it if it does not exist yet
                    block_idx = np.argwhere(np.all(block_inds == new_bi, axis=1))[:, 0]
                    if len(block_idx) == 0:
                        block_idx = len(blocks)
                        block_inds = np.vstack((block_inds, new_bi))
                        shape = [
                            leg.multiplicities[i]
                            for i, leg in zip(new_bi, conventional_leg_order(new_codomain, new_domain))
                        ]
                        blocks.append(self.block_backend.zeros(shape, dtype=dtype, device=device))
                    else:
                        block_idx = block_idx[0]

                    row_slc = slice(right_mult_slices[right_ind][j], right_mult_slices[right_ind][j + 1])
                    col_slc = slice(left_mult_slices[left_ind][i], left_mult_slices[left_ind][i + 1])
                    block_slcs = (col_slc, *codom_slcs, row_slc, *dom_slcs)
                    blocks[block_idx][block_slcs] += op_block
        return AbelianBackendData(dtype=dtype, device=device, blocks=blocks, block_inds=block_inds, is_sorted=False)

    def from_random_normal(
        self, codomain: TensorProduct, domain: TensorProduct, sigma: float, dtype: Dtype, device: str
    ) -> Data:
        def func(shape, coupled):
            return self.block_backend.random_normal(shape, dtype, sigma, device=device)

        return self.from_sector_block_func(func, codomain=codomain, domain=domain)

    def from_sector_block_func(self, func, codomain: TensorProduct, domain: TensorProduct) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        block_inds = _valid_block_inds(codomain=codomain, domain=domain)
        M = codomain.num_factors
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i] for i, leg in zip(b_i, conventional_leg_order(codomain, domain))]
            coupled = codomain.symmetry.multiple_fusion(
                *(leg.sector_decomposition[i] for i, leg in zip(b_i, codomain.factors))
            )
            blocks.append(func(shape, coupled))
        if len(blocks) == 0:
            sample_block = func((1,) * (M + domain.num_factors), codomain.symmetry.trivial_sector)
        else:
            sample_block = blocks[0]
        dtype = self.block_backend.get_dtype(sample_block)
        device = self.block_backend.get_device(sample_block)
        return AbelianBackendData(dtype=dtype, device=device, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def from_tree_pairs(
        self,
        trees: dict[tuple[FusionTree, FusionTree], Block],
        codomain: TensorProduct,
        domain: TensorProduct,
        dtype: Dtype,
        device: str,
    ) -> Data:
        block_inds = []
        blocks = []
        pairs_done = set()
        for bi in _valid_block_inds(codomain, domain):
            X = FusionTree.from_abelian_symmetry(
                symmetry=codomain.symmetry,
                uncoupled=[f.sector_decomposition[bi[n]] for n, f in enumerate(codomain)],
                are_dual=[f.is_dual for f in codomain],
            )
            Y = FusionTree.from_abelian_symmetry(
                symmetry=domain.symmetry,
                uncoupled=[f.sector_decomposition[bi[-1 - n]] for n, f in enumerate(domain)],
                are_dual=[f.is_dual for f in domain],
            )
            pair = (X, Y)
            pairs_done.add(pair)
            block = trees.get(pair, None)
            if block is None:
                continue
            block_inds.append(bi)
            blocks.append(block)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, codomain.num_factors + domain.num_factors), int)
        else:
            block_inds = np.array(block_inds)
        # check if we covered all keys in the dict
        for pair in trees.keys():
            if pair not in pairs_done:
                # SymmetricTensor.from_tree_pairs should have done enough input checks to prevent this
                # OPTIMIZE if the code works, we could remove this check
                raise RuntimeError
        return AbelianBackendData(dtype=dtype, device=device, blocks=blocks, block_inds=block_inds, is_sorted=False)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_backend.block_from_diagonal(block) for block in a.data.blocks]
        return AbelianBackendData(a.dtype, a.data.device, blocks, a.data.block_inds, is_sorted=True)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        blocks = [self.block_backend.block_from_mask(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, a.data.device, blocks, a.data.block_inds, is_sorted=True)

    def get_device_from_data(self, a: AbelianBackendData) -> str:
        return a.device

    def get_dtype_from_data(self, a: AbelianBackendData) -> Dtype:
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
        res = 0.0
        for i, j in iter_common_sorted(a_block_inds, b_block_inds):
            res += self.block_backend.inner(a_blocks[i], b_blocks[j], do_dagger=do_dagger)
        return res

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space, charge_leg: ElementarySpace) -> Data:
        assert charge_leg.num_sectors == 1
        bi = space.sector_decomposition_where(charge_leg.sector_decomposition[0])
        assert bi is not None
        assert self.block_backend.get_shape(vector) == (space.multiplicities[bi])
        return AbelianBackendData(
            dtype=self.block_backend.get_dtype(vector),
            device=self.block_backend.get_device(vector),
            blocks=[self.block_backend.add_axis(vector, pos=1)],
            block_inds=np.array([[bi, 0]]),
            is_sorted=True,
        )

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        num_blocks = len(tensor.data.blocks)
        assert num_blocks <= 1
        if num_blocks == 1:
            return tensor.data.blocks[0][:, 0]
        sector = tensor.domain[0].sector_decomposition[0]
        dim = tensor.codomain[0].sector_multiplicity(sector)
        return self.block_backend.zeros([dim], dtype=tensor.data.dtype)

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        v_blocks = v.data.blocks
        w_blocks = w.data.blocks
        v_block_inds = v.data.block_inds
        w_block_inds = w.data.block_inds
        # ensure common dtypes
        common_dtype = v.dtype.common(w.dtype)
        if v.data.dtype != common_dtype:
            v_blocks = [self.block_backend.to_dtype(T, common_dtype) for T in v_blocks]
        if w.data.dtype != common_dtype:
            w_blocks = [self.block_backend.to_dtype(T, common_dtype) for T in w_blocks]
        res_blocks = []
        res_block_inds = []
        for i, j in iter_common_noncommon_sorted_arrays(v_block_inds, w_block_inds):
            if j is None:
                res_blocks.append(self.block_backend.mul(a, v_blocks[i]))
                res_block_inds.append(v_block_inds[i])
            elif i is None:
                res_blocks.append(self.block_backend.mul(b, w_blocks[j]))
                res_block_inds.append(w_block_inds[j])
            else:
                res_blocks.append(self.block_backend.linear_combination(a, v_blocks[i], b, w_blocks[j]))
                res_block_inds.append(v_block_inds[i])
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, v.num_legs), int)
        return AbelianBackendData(common_dtype, v.data.device, res_blocks, res_block_inds, is_sorted=True)

    def lq(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        new_leg = new_co_domain[0]
        assert a.num_codomain_legs == 1 == a.num_domain_legs  # since self.can_decompose_tensors is False
        l_blocks = []
        q_blocks = []
        l_block_inds = []
        q_block_inds = []
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(
            iter_common_sorted_arrays(a.codomain.sector_decomposition, a.domain.sector_decomposition)
        ):
            # due to the loop setup we have:
            #   a.codomain.sector_decomposition[j] == new_leg.sector_decomposition[n]
            #   a.domain.sector_decomposition[k] == new_leg.sector_decomposition[n]
            # but we still need the leg indices (which may differ depending on the sector_order)
            sector = a.codomain.sector_decomposition[j]
            if a.codomain[0].sector_order != 'sorted':
                j = a.codomain[0].sector_decomposition_where(sector)
            if a.domain[0].sector_order != 'sorted':
                k = a.domain[0].sector_decomposition_where(sector)
                # block_inds is lexsorted and in this case duplicate-free
                # -> running index i is correct iff domain is correctly sorted
                i = np.searchsorted(a_block_inds[:, 1], k)
            if new_leg.sector_order != 'sorted':
                n = new_leg.sector_decomposition_where(sector)

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
                q_blocks.append(self.block_backend.eye_matrix(a.domain[0].multiplicities[k], a.dtype)[:new_leg_dim, :])
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
        l_sorted = new_leg.sector_order == 'sorted'
        q_sorted = a.domain[0].sector_order == 'sorted'
        l_data = AbelianBackendData(a.dtype, a.data.device, l_blocks, l_block_inds, is_sorted=l_sorted)
        q_data = AbelianBackendData(a.dtype, a.data.device, q_blocks, q_block_inds, is_sorted=q_sorted)
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
        for sector_idx, (sector, slc) in enumerate(zip(large_leg.defining_sectors, large_leg.slices)):
            if sector_idx == b1_i1:
                block1 = mask1_blocks[i1]
                i1 += 1
                if i1 >= len(mask1_block_inds):
                    b1_i1 = -1  # mask1 has no further blocks
                else:
                    b1_i1 = mask1_block_inds[i1, 1]
            else:
                block1 = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)
            if sector_idx == b2_i2:
                block2 = mask2_blocks[i2]
                i2 += 1
                if i2 >= len(mask2_block_inds):
                    b2_i2 = -1  # mask2 has no further blocks
                else:
                    b2_i2 = mask1_block_inds[i2, 1]
            else:
                block2 = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)
            new_block = func(block1, block2)
            mult = self.block_backend.sum_all(new_block)
            if mult == 0:
                continue
            blocks.append(new_block)
            large_leg_block_inds.append(sector_idx)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                mask = self.block_backend.to_numpy(new_block)
                basis_perm_ranks.append(basis_perm[slice(*slc)][mask])
        block_inds = np.column_stack([np.arange(len(sectors)), large_leg_block_inds])
        data = AbelianBackendData(
            dtype=Dtype.bool, device=mask1.device, blocks=blocks, block_inds=block_inds, is_sorted=True
        )
        if len(sectors) == 0:
            sectors = mask1.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
        small_leg = ElementarySpace(
            symmetry=mask1.symmetry,
            defining_sectors=sectors,
            multiplicities=multiplicities,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def mask_contract_large_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=True)

    def mask_contract_small_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=False)

    def _mask_contract(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int, large_leg: bool
    ) -> tuple[Data, TensorProduct, TensorProduct]:
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
        tensor_block_inds_contr = tensor_block_inds[:, leg_idx : leg_idx + 1]
        #
        mask_blocks = mask.data.blocks
        mask_block_inds = mask.data.block_inds
        mask_block_inds_contr = mask_block_inds[:, mask_contr : mask_contr + 1]
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
                block = self.block_backend.apply_mask(tensor_blocks[i], mask_blocks[j], ax=leg_idx)
            else:
                block = self.block_backend.enlarge_leg(tensor_blocks[i], mask_blocks[j], axis=leg_idx)
            block_inds = tensor_block_inds[i].copy()
            block_inds[leg_idx] = mask_block_inds[j, 1 - mask_contr]
            res_blocks.append(block)
            res_block_inds.append(block_inds)
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, tensor.num_legs), int)
        # OPTIMIZE (JU) block_inds might actually be sorted but i am not sure right now
        data = AbelianBackendData(tensor.dtype, tensor.device, res_blocks, res_block_inds, is_sorted=False)
        #
        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.factors[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            domain = TensorProduct(spaces, symmetry=tensor.symmetry)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.factors[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            codomain = TensorProduct(spaces, symmetry=tensor.symmetry)
        return data, codomain, domain

    def mask_dagger(self, mask: Mask) -> MaskData:
        # the legs swap between domain and codomain. need to swap the two columns of block_inds.
        # since both columns are unique and ascending, the resulting block_inds are still sorted.
        block_inds = mask.data.block_inds[:, ::-1]
        return AbelianBackendData(
            dtype=mask.dtype, device=mask.device, blocks=mask.data.blocks, block_inds=block_inds, is_sorted=True
        )

    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        basis_perm = large_leg._basis_perm
        blocks = []
        large_leg_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        for bi_large, (slc, sector) in enumerate(zip(large_leg.slices, large_leg.defining_sectors)):
            block = a[slice(*slc)]
            mult = self.block_backend.sum_all(block)
            if mult == 0:
                continue
            blocks.append(block)
            large_leg_block_inds.append(bi_large)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                mask = self.block_backend.to_numpy(block)
                basis_perm_ranks.append(large_leg.basis_perm[slice(*slc)][mask])

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

        data = AbelianBackendData(
            dtype=Dtype.bool,
            device=self.block_backend.get_device(a),
            blocks=blocks,
            block_inds=block_inds,
            is_sorted=True,
        )
        small_leg = ElementarySpace(
            symmetry=large_leg.symmetry,
            defining_sectors=sectors,
            multiplicities=multiplicities,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def mask_to_block(self, a: Mask) -> Block:
        large_leg = a.large_leg
        res = self.block_backend.zeros([large_leg.dim], Dtype.bool)
        for block, b_i in zip(a.data.blocks, a.data.block_inds):
            if a.is_projection:
                bi_small, bi_large = b_i
            else:
                bi_large, bi_small = b_i
            res[slice(*large_leg.slices[bi_large])] = block
        return res

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        blocks = [self.block_backend.to_dtype(b, dtype) for b in a.data.blocks]
        large_leg_bi = a.data.block_inds[:, 1] if a.is_projection else a.data.block_inds[:, 0]
        block_inds = np.repeat(large_leg_bi[:, None], 2, axis=1)
        return AbelianBackendData(dtype=dtype, device=a.data.device, blocks=blocks, block_inds=block_inds)

    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        block_inds = tens.data.block_inds[:, ::-1]
        data = AbelianBackendData(
            dtype=tens.dtype, device=tens.data.device, blocks=tens.data.blocks, block_inds=block_inds, is_sorted=False
        )
        return tens.codomain[0].dual, tens.domain[0].dual, data

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
        for sector_idx, (sector, slc) in enumerate(zip(large_leg.defining_sectors, large_leg.slices)):
            if sector_idx == b_i:
                block = mask_blocks[i]
                i += 1
                if i >= len(mask_blocks_inds):
                    b_i = -1  # mask has no further blocks
                else:
                    b_i = mask_blocks_inds[i, 1]
            else:
                block = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)
            new_block = func(block)
            mult = self.block_backend.sum_all(new_block)
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
            dtype=Dtype.bool, device=mask.data.device, blocks=blocks, block_inds=block_inds, is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=mask.symmetry,
            defining_sectors=sectors,
            multiplicities=multiplicities,
            is_dual=large_leg.is_dual,
            basis_perm=basis_perm,
        )
        return data, small_leg

    def move_to_device(self, a: SymmetricTensor | DiagonalTensor | Mask, device: str) -> Data:
        for i in range(len(a.data.blocks)):
            a.data.blocks[i] = self.block_backend.as_block(a.data.blocks[i], device=device)
        a.data.device = self.block_backend.as_device(device)
        return a.data

    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        if a == 0.0:
            return self.zero_data(b.codomain, b.domain, b.dtype, device=b.data.device)
        blocks = [self.block_backend.mul(a, T) for T in b.data.blocks]
        if len(blocks) == 0:
            if isinstance(a, float):
                dtype = b.data.dtype
            else:
                dtype = b.data.dtype.to_complex
        else:
            dtype = self.block_backend.get_dtype(blocks[0])
        return AbelianBackendData(dtype, b.data.device, blocks, b.data.block_inds, is_sorted=True)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        block_norms = [self.block_backend.norm(b, order=2) for b in a.data.blocks]
        return float(np.linalg.norm(block_norms, ord=2))

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
            a_blocks = [self.block_backend.to_dtype(T, res_dtype) for T in a_blocks]
        if b.dtype != res_dtype:
            b_blocks = [self.block_backend.to_dtype(T, res_dtype) for T in b_blocks]
        #
        grid = make_grid([l_a, l_b], cstyle=False)
        #
        res_block_inds = np.empty((l_a * l_b, N_a + N_b), dtype=int)
        res_block_inds[:, :K_a] = a_block_inds[grid[:, 0], :K_a]
        res_block_inds[:, K_a : K_a + N_b] = b_block_inds[grid[:, 1]]
        res_block_inds[:, K_a + N_b :] = a_block_inds[grid[:, 0], K_a:]
        res_blocks = [self.block_backend.tensor_outer(a_blocks[i], b_blocks[j], K_a) for i, j in grid]
        # res_block_inds are in general not sorted.
        #
        return AbelianBackendData(res_dtype, a.data.device, res_blocks, res_block_inds, is_sorted=False)

    def partial_compose(
        self,
        a: SymmetricTensor,
        b: SymmetricTensor,
        a_first_leg: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        # construct new data and spaces with the legs to be contracted at the end of a and the beginning of b
        if a_first_leg < a.num_codomain_legs:
            num_contr_legs = b.num_domain_legs
            num_add_legs = b.num_codomain_legs
            perm_b = list(range(b.num_codomain_legs, b.num_legs)) + list(range(b.num_codomain_legs))
            b_blocks = [self.block_backend.permute_axes(block, perm_b) for block in b.data.blocks]
            b_data = AbelianBackendData(
                b.data.dtype, b.data.device, b_blocks, b.data.block_inds[:, perm_b], is_sorted=False
            )
        else:
            num_contr_legs = b.num_codomain_legs
            num_add_legs = b.num_domain_legs
            perm_b = list(range(b.num_legs))
            b_data = b.data

        perm_a = (
            list(range(a_first_leg))
            + list(range(a_first_leg + num_contr_legs, a.num_legs))
            + list(range(a_first_leg, a_first_leg + num_contr_legs))
        )
        a_blocks = [self.block_backend.permute_axes(block, perm_a) for block in a.data.blocks]
        a_data = AbelianBackendData(
            a.data.dtype, a.data.device, a_blocks, a.data.block_inds[:, perm_a], is_sorted=False
        )

        # the computation of these modified tensorproducts cannot be avoided
        # since they may differ from the ones computed in _tensors.py by bending
        mod_codomain = [a._as_codomain_leg(idx) for i, idx in enumerate(perm_a) if i < a.num_legs - num_contr_legs]
        mod_codomain = TensorProduct(mod_codomain, a.symmetry)
        mod_domain = [b._as_domain_leg(idx) for i, idx in enumerate(perm_b) if i >= num_contr_legs][::-1]
        mod_domain = TensorProduct(mod_domain)
        contr_spaces = [b.get_leg_co_domain(idx) for i, idx in enumerate(perm_b) if i < num_contr_legs]

        res_data = self._compose_worker(a_data, b_data, mod_codomain, contr_spaces, mod_domain)
        perm_res = (
            list(range(a_first_leg))
            + list(range(a.num_legs - num_contr_legs, a.num_legs - num_contr_legs + num_add_legs))
            + list(range(a_first_leg, a.num_legs - num_contr_legs))
        )
        res_blocks = [self.block_backend.permute_axes(block, perm_res) for block in res_data.blocks]
        return AbelianBackendData(
            res_data.dtype, res_data.device, res_blocks, res_data.block_inds[:, perm_res], is_sorted=False
        )

    def partial_trace(
        self, tensor: SymmetricTensor, pairs: list[tuple[int, int]], levels: list[int] | None
    ) -> tuple[Data, TensorProduct, TensorProduct]:
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

        # only blocks "on the diagonal" of the trace contribute.
        # figure out which blocks are on the diagonal.
        on_diagonal = np.ones(len(blocks), bool)  # we do logical_and, so we start with all true
        for n, opposite in enumerate(opposite_sides):
            if opposite:
                # legs are the same -> can compare the block_inds
                on_diagonal &= block_inds_1[:, n] == block_inds_2[:, n]
            else:
                # legs have opposite duality. need to compare sectors explicitly
                # OPTIMIZE (JU) spaces could store (or cache!) the sector permutation between
                #               itself and its dual, then we could compare on the level of block_inds
                s1 = tensor.get_leg_co_domain(idcs1[n]).sector_decomposition[block_inds_1[:, n], :]
                s2 = tensor.get_leg_co_domain(idcs2[n]).sector_decomposition[block_inds_2[:, n], :]
                on_diagonal &= np.all(s1 == tensor.symmetry.dual_sectors(s2), axis=1)

        res_data = {}  # dictionary res_block_inds_row -> Block
        for block, contributes, bi_rem in zip(blocks, on_diagonal, block_inds_rem):
            if not contributes:
                continue
            bi_rem = tuple(bi_rem)
            block = self.block_backend.trace_partial(block, idcs1, idcs2, remaining)
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
                return self.block_backend.item(res_blocks[0]), None, None
            raise RuntimeError  # by charge rule, should be impossible to get multiple blocks.

        if len(res_blocks) == 0:
            res_block_inds = np.zeros((0, len(remaining)), int)
        else:
            res_block_inds = np.array(list(res_data.keys()), int)
        data = AbelianBackendData(tensor.dtype, tensor.data.device, res_blocks, res_block_inds, is_sorted=False)
        codomain = TensorProduct(
            [leg for n, leg in enumerate(tensor.codomain) if n in remaining], symmetry=tensor.symmetry
        )
        domain = TensorProduct(
            [leg for n, leg in enumerate(tensor.domain) if N - 1 - n in remaining], symmetry=tensor.symmetry
        )
        return data, codomain, domain

    def permute_legs(
        self,
        a: SymmetricTensor,
        codomain_idcs: list[int],
        domain_idcs: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
        mixes_codomain_domain: bool,
        levels: list[int] | None,
        bend_right: list[bool | None],
    ) -> AbelianBackendData:
        axes_perm = [*codomain_idcs, *reversed(domain_idcs)]
        blocks = [self.block_backend.permute_axes(block, axes_perm) for block in a.data.blocks]
        block_inds = a.data.block_inds[:, axes_perm]
        data = AbelianBackendData(a.dtype, a.data.device, blocks=blocks, block_inds=block_inds, is_sorted=False)
        return data

    def qr(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        new_leg = new_co_domain[0]
        assert a.num_codomain_legs == 1 == a.num_domain_legs  # since self.can_decompose_tensors is False
        q_blocks = []
        r_blocks = []
        q_block_inds = []
        r_block_inds = []
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        i = 0  # running index, indicating we have already processed a_blocks[:i]
        for n, (j, k) in enumerate(
            iter_common_sorted_arrays(a.codomain.sector_decomposition, a.domain.sector_decomposition)
        ):
            # due to the loop setup we have:
            #   a.codomain.sector_decomposition[j] == new_leg.sector_decomposition[n]
            #   a.domain.sector_decomposition[k] == new_leg.sector_decomposition[n]
            # but we still need the leg indices (which may differ depending on the sector_order)
            sector = a.codomain.sector_decomposition[j]
            if a.codomain[0].sector_order != 'sorted':
                j = a.codomain[0].sector_decomposition_where(sector)
            if a.domain[0].sector_order != 'sorted':
                k = a.domain[0].sector_decomposition_where(sector)
                # block_inds is lexsorted and in this case duplicate-free
                # -> running index i is correct iff domain is correctly sorted
                i = np.searchsorted(a_block_inds[:, 1], k)
            if new_leg.sector_order != 'sorted':
                n = new_leg.sector_decomposition_where(sector)

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
                eye = self.block_backend.eye_matrix(a.codomain[0].multiplicities[j], a.dtype)
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
        q_sorted = new_leg.sector_order == 'sorted'
        r_sorted = a.domain[0].sector_order == 'sorted'
        q_data = AbelianBackendData(a.dtype, a.data.device, q_blocks, q_block_inds, is_sorted=q_sorted)
        r_data = AbelianBackendData(a.dtype, a.data.device, r_blocks, r_block_inds, is_sorted=r_sorted)
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
                block = self.block_backend.zeros(m, dtype=tensor.dtype)
            numbers.append(block_func(block))
        return func(numbers)

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        a_block_inds_cont = a_block_inds[:, leg : leg + 1]
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
            a_blocks = [self.block_backend.to_dtype(block, common_dtype) for block in a_blocks]
        if b.data.dtype != common_dtype:
            b_blocks = [self.block_backend.to_dtype(block, common_dtype) for block in b_blocks]
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
            res_blocks.append(self.block_backend.scale_axis(a_blocks[i], b_blocks[j], axis=leg))
            res_block_inds.append(a_block_inds[i])
        #
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, a.num_legs), int)
        #
        return AbelianBackendData(common_dtype, a.data.device, res_blocks, res_block_inds, is_sorted=False)

    def split_legs(
        self,
        a: SymmetricTensor,
        leg_idcs: list[int],
        codomain_split: list[int],
        domain_split: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        if len(a.data.blocks) == 0:
            return self.zero_data(new_codomain, new_domain, a.data.dtype, device=a.data.device)

        n_split = len(leg_idcs)
        pipes = [a.get_leg_co_domain(i) for i in leg_idcs]
        res_num_legs = new_codomain.num_factors + new_domain.num_factors

        old_blocks = a.data.blocks
        old_block_inds = a.data.block_inds

        map_slices_beg = np.zeros((len(old_blocks), n_split), int)
        map_slices_shape = np.zeros((len(old_blocks), n_split), int)  # = end - beg
        for j, pipe in enumerate(pipes):
            block_inds_j = old_block_inds[:, leg_idcs[j]]
            map_slices_beg[:, j] = pipe.block_ind_map_slices[block_inds_j]
            sizes = pipe.block_ind_map_slices[1:] - pipe.block_ind_map_slices[:-1]
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
        # splitting pipes in F style is done by splitting them in C style and permuting the axes
        axes_perm = list(range(res_num_legs))
        shift = 0  # = i - k for indices below
        j = 0  # index within pipes
        for i in range(a.num_legs):  # i = index in old tensor
            if i in leg_idcs:
                in_domain = i >= a.num_codomain_legs
                pipe = pipes[j]  # = a.legs[i]
                k = i + shift  # = index where split legs begin in new tensor
                k2 = k + pipe.num_legs  # = until where spaces go in new tensor
                if pipe.combine_cstyle == in_domain:
                    axes_perm[k:k2] = axes_perm[k:k2][::-1]
                block_ind_map = pipe.block_ind_map[map_rows[:, j], :]
                if in_domain:
                    # if the leg to be split is in the domain, the order of block_inds and of its
                    # block_ind_map are opposite -> need to reverse
                    new_block_inds[:, k:k2] = block_ind_map[:, -2:1:-1]
                else:
                    new_block_inds[:, k:k2] = block_ind_map[:, 2:-1]
                old_block_beg[:, i] = block_ind_map[:, 0]
                old_block_shapes[:, i] = block_ind_map[:, 1] - block_ind_map[:, 0]
                shift += pipe.num_legs - 1
                j += 1
            else:
                new_block_inds[:, i + shift] = nbi = old_block_inds[old_rows, i]
                old_block_shapes[:, i] = a.get_leg_co_domain(i).multiplicities[nbi]

        new_block_shapes = np.empty((res_num_blocks, res_num_legs), dtype=int)
        for i, leg in enumerate(conventional_leg_order(new_codomain, new_domain)):
            new_block_shapes[:, i] = leg.multiplicities[new_block_inds[:, i]]

        # the actual loop to split the blocks
        new_blocks = []
        for i in range(res_num_blocks):
            old_block = old_blocks[old_rows[i]]
            slices = tuple(slice(b, b + s) for b, s in zip(old_block_beg[i], old_block_shapes[i]))
            new_block = old_block[slices]
            new_blocks.append(self.block_backend.reshape(new_block, new_block_shapes[i]))
        new_blocks = [self.block_backend.permute_axes(block, axes_perm) for block in new_blocks]

        return AbelianBackendData(a.data.dtype, a.data.device, new_blocks, new_block_inds, is_sorted=False)

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        n_legs = a.num_legs
        if len(a.data.blocks) == 0:
            block_inds = np.zeros([0, n_legs - len(idcs)], dtype=int)
            return AbelianBackendData(a.data.dtype, a.data.device, [], block_inds, is_sorted=True)
        blocks = [self.block_backend.squeeze_axes(b, idcs) for b in a.data.blocks]
        keep = np.ones(n_legs, dtype=bool)
        keep[idcs] = False
        block_inds = a.data.block_inds[:, keep]
        return AbelianBackendData(a.data.dtype, a.data.device, blocks, block_inds, is_sorted=True)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry.is_abelian and symmetry.braiding_style == BraidingStyle.bosonic

    def svd(
        self, a: SymmetricTensor, new_co_domain: TensorProduct, algorithm: str | None
    ) -> tuple[Data, DiagonalData, Data]:
        # The issue here is that sector_decomposition of the (co)domain is sorted, but may be
        # dual_sorted for the single leg in the (co)domain. The block_inds do contain the indices
        # of the legs, i.e., either we (generically) cannot iterate over sorted arrays (= iterate
        # over legs) or we iterate over sorted arrays (= iterate over (co)domain) and then need an
        # additional step to find the correct indices.
        # We do the latter, i.e., assuming that sector_decomposition_where is efficient.
        # Additionally, the block_inds of u, s, vh are in general no longer lexsorted.

        # In the special case in which the sector_decomposition of all legs is sorted, it reduces
        # to the previous case, where we do not need to find any indices and the block_inds are
        # constructed in a lexsorted way.
        new_leg = new_co_domain[0]
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
        for n, (j, k) in enumerate(
            iter_common_sorted_arrays(a.codomain.sector_decomposition, a.domain.sector_decomposition)
        ):
            # due to the loop setup we have:
            #   a.codomain.sector_decomposition[j] == new_leg.sector_decomposition[n]
            #   a.domain.sector_decomposition[k] == new_leg.sector_decomposition[n]
            # but we still need the leg indices (which may differ depending on the sector_order)
            sector = a.codomain.sector_decomposition[j]
            if a.codomain[0].sector_order != 'sorted':
                j = a.codomain[0].sector_decomposition_where(sector)
            if a.domain[0].sector_order != 'sorted':
                k = a.domain[0].sector_decomposition_where(sector)
                # block_inds is lexsorted and in this case duplicate-free
                # -> running index i is correct iff domain is correctly sorted
                i = np.searchsorted(a_block_inds[:, 1], k)
            if new_leg.sector_order != 'sorted':
                n = new_leg.sector_decomposition_where(sector)

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
                eye_u = self.block_backend.eye_matrix(a.codomain[0].multiplicities[j], a.dtype)
                u_blocks.append(eye_u[:, :new_leg_dim])
                eye_v = self.block_backend.eye_matrix(a.domain[0].multiplicities[k], a.dtype)
                vh_blocks.append(eye_v[:new_leg_dim, :])
            u_block_inds.append([j, n])
            vh_block_inds.append([n, k])

        if len(s_blocks) == 0:
            s_block_inds = np.zeros([0, 2], int)
        else:
            s_block_inds = np.repeat(np.array(s_block_inds, int)[:, None], 2, axis=1)
        if len(u_blocks) == 0:
            u_block_inds = vh_block_inds = np.zeros([0, 2], int)
        else:
            u_block_inds = np.array(u_block_inds, int)
            vh_block_inds = np.array(vh_block_inds, int)

        # for all block_inds, the last column is sorted and duplicate-free,
        # thus the block_inds are np.lexsort( .T)-ed if the sector_order of
        # the corresponding leg is sorted
        u_sorted = s_sorted = new_leg.sector_order == 'sorted'
        vh_sorted = a.domain[0].sector_order == 'sorted'

        u_data = AbelianBackendData(a.dtype, a.data.device, u_blocks, u_block_inds, is_sorted=u_sorted)
        s_data = AbelianBackendData(a.dtype.to_real, a.data.device, s_blocks, s_block_inds, is_sorted=s_sorted)
        vh_data = AbelianBackendData(a.dtype, a.data.device, vh_blocks, vh_block_inds, is_sorted=vh_sorted)
        return u_data, s_data, vh_data

    def state_tensor_product(self, state1: Block, state2: Block, pipe: AbelianLegPipe):
        # clearly define what this should do in tensors.py first!
        raise NotImplementedError('state_tensor_product not implemented')

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        res = self.block_backend.zeros(a.shape, a.data.dtype)
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
            return self.block_backend.zeros(shape=[dim], dtype=tensor.data.dtype)
        raise ValueError  # this should not happen for single-leg tensors

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        # shallow copy if dtype stays same
        blocks = [self.block_backend.to_dtype(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, a.data.device, blocks, a.data.block_inds, is_sorted=True)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        K = a.num_codomain_legs
        res = a.data.dtype.zero_scalar
        for block, bi in zip(a_blocks, a_block_inds):
            bi_cod = bi[:K]
            bi_dom = bi[K:]
            if np.all(bi_cod == bi_dom[::-1]):
                res += self.block_backend.trace_full(block)
            # else: block is entirely off-diagonal and does not contribute to the trace
        return res

    def truncate_singular_values(
        self,
        S: DiagonalTensor,
        chi_max: int | None,
        chi_min: int,
        degeneracy_tol: float,
        trunc_cut: float,
        svd_min: float,
        minimize_error: bool = True,
    ) -> tuple[MaskData, ElementarySpace, float, float]:
        S_np = self.block_backend.to_numpy(self.diagonal_tensor_to_block(S))
        keep, err, new_norm = self._truncate_singular_values_selection(
            S=S_np,
            qdims=None,
            chi_max=chi_max,
            chi_min=chi_min,
            degeneracy_tol=degeneracy_tol,
            trunc_cut=trunc_cut,
            svd_min=svd_min,
            minimize_error=minimize_error,
        )
        keep = self.block_backend.as_block(keep, Dtype.bool)
        mask_data, small_leg = self.mask_from_block(keep, large_leg=S.leg)
        return mask_data, small_leg, err, new_norm

    def zero_data(
        self, codomain: TensorProduct, domain: TensorProduct, dtype: Dtype, device: str, all_blocks: bool = False
    ) -> AbelianBackendData:
        if not all_blocks:
            block_inds = np.zeros((0, codomain.num_factors + domain.num_factors), dtype=int)
            return AbelianBackendData(dtype, device, blocks=[], block_inds=block_inds, is_sorted=True)

        block_inds = _valid_block_inds(codomain=codomain, domain=domain)
        zero_blocks = []
        for idcs in block_inds:
            shape = [leg.multiplicities[i] for i, leg in zip(idcs, conventional_leg_order(codomain, domain))]
            zero_blocks.append(self.block_backend.zeros(shape, dtype=dtype, device=device))
        return AbelianBackendData(dtype, device, zero_blocks, block_inds, is_sorted=True)

    def zero_diagonal_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> DiagonalData:
        return AbelianBackendData(dtype, device, blocks=[], block_inds=np.zeros((0, 2), dtype=int), is_sorted=True)

    def zero_mask_data(self, large_leg: Space, device: str) -> MaskData:
        return AbelianBackendData(Dtype.bool, device, blocks=[], block_inds=np.zeros((0, 2), dtype=int), is_sorted=True)

    # INTERNAL HELPERS

    def leg_pipe_map_incoming_block_inds(self, pipe: AbelianLegPipe, incoming_block_inds):
        """Map incoming block indices to indices of :attr:`block_ind_map`.

        Needed for `combine_legs`.

        Parameters
        ----------
        pipe : AbelianLegPipe
            The pipe which indices are to be mapped
        incoming_block_inds : 2D array
            Rows are block indices :math:`(i_1, i_2, ... i_{nlegs})` for incoming legs.

        Returns
        -------
        block_inds: 1D array
            For each row j of `incoming_block_inds` an index `J` such that
            ``pipe.block_ind_map[J, 2:-1] == block_inds[j]``.

        """
        assert incoming_block_inds.shape[1] == pipe.num_legs
        # calculate indices of _block_ind_map by using the appropriate strides
        inds_before_perm = np.sum(incoming_block_inds * pipe.sector_strides[np.newaxis, :], axis=1)
        # now permute them to indices in _block_ind_map
        return inverse_permutation(pipe.fusion_outcomes_sort)[inds_before_perm]

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.DataCls, subpath + 'DataCls')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.DataCls = hdf5_loader.load(subpath + 'DataCls')

        return obj
