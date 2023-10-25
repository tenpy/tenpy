"""Backends for abelian group symmetries.

Changes compared to old np_conserved:

- replace `ChargeInfo` by subclasses of `AbelianGroup` (or `ProductSymmetry`)
- replace `LegCharge` by `AbelianBackendVectorSpace` and `LegPipe` by `AbelianBackendProductSpace`
- standard `Tensor` have qtotal=0, only ChargedTensor can have non-zero qtotal
- relabeling:
    - `Array.qdata`, "qind" and "qindices" to `AbelianBackendData.block_inds` and "block indices"
    - `LegPipe.qmap` to `ProductSpace._block_ind_map` (with changed column order!!!)
    - `LegPipe._perm` to `ProductSpace._perm_block_inds_map`  TODO (JU) this is outdated np?
    - `LegCharge.get_block_sizes()` is just `VectorSpace.multiplicities`
- TODO point towards VectorSpace attributes
- keep VectorSpace and ProductSpace "sorted" and "bunched",
  i.e. do not support legs with smaller blocks to effectively allow block-sparse tensors with
  smaller blocks than dictated by symmetries (which we actually have in H_MPO on the virtual legs...)
  In turn, VectorSpace saves a `_perm` used to sort the originally passed `sectors`.
- keep `block_inds` sorted (i.e. no arbitrary gauge permutation in block indices)

"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import numpy as np
import copy
import warnings

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, DiagonalData, Block, Dtype
from ..misc import make_stride, find_row_differences
from ..groups import FusionStyle, BraidingStyle, Symmetry, Sector, SectorArray, AbelianGroup
from numpy import ndarray
from ..spaces import VectorSpace, ProductSpace
from ...tools.misc import inverse_permutation, list_to_dict_list

__all__ = ['AbelianBackendData', 'AbstractAbelianBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor, ChargedTensor, DiagonalTensor, Mask


def _valid_block_indices(spaces: list[VectorSpace]):
    """Find block_inds where the charges of the `spaces` fuse to `symmetry.trivial_sector`.
    The resulting block_inds are lexsort( .T)-ed."""
    assert len(spaces) > 0
    symmetry = spaces[0].symmetry
    # OPTIMIZE: this is brute-force going through all possible combinations of block indices
    # spaces are sorted, so we can probably reduce that search space quite a bit...
    # similar to `grid` in ProductSpace._fuse_spaces()
    grid = np.indices((s.num_sectors for s in spaces), dtype=int)
    grid = grid.T.reshape((-1, len(spaces)))
    total_sectors = symmetry.multiple_fusion_broadcast(
        *(space.sectors[gr] for space, gr in zip(spaces, grid.T))
    )
    valid = np.all(total_sectors == symmetry.trivial_sector[np.newaxis, :], axis=1)
    block_inds = grid[valid, :]
    perm = np.lexsort(block_inds.T)
    return block_inds[perm, :]


def _iter_common_sorted(a, b):
    """Yield indices ``i, j`` for which ``a[i] == b[j]``.

    *Assumes* that `a` and `b` are strictly ascending 1D arrays.
    Given that, it is equivalent to (but faster than)
    ``[(i, j) for j, i in itertools.product(range(len(b)), range(len(a)) if a[i] == b[j]]``
    """
    # when we call this function, we basically wanted _iter_common_sorted_arrays,
    # but used strides to merge multiple columns to avoid too much python loops
    # for C-implementation, this is definitely no longer necessary.
    l_a = len(a)
    l_b = len(b)
    i, j = 0, 0
    res = []
    while i < l_a and j < l_b:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            res.append((i, j))
            i += 1
            j += 1
    return res


def _iter_common_sorted_arrays(a, b):
    """Yield indices ``i, j`` for which ``a[i, :] == b[j, :]``.

    *Assumes* that `a` and `b` are strictly lex-sorted (according to ``np.lexsort(a.T)``).
    Given that, it is equivalent to (but faster than)
    ``[(i, j) for j, i in itertools.product(range(len(b)), range(len(a)) if all(a[i,:] == b[j,:]]``
    """
    l_a, d_a = a.shape
    l_b, d_b = b.shape
    assert d_a == d_b
    i, j = 0, 0
    while i < l_a and j < l_b:
        for k in reversed(range(d_a)):
            if a[i, k] < b[j, k]:
                i += 1
                break
            elif b[j, k] < a[i, k]:
                j += 1
                break
        else:
            yield (i, j)
            i += 1
            j += 1
    # done


def _iter_common_nonstrict_sorted_arrays(a, b):
    """Yield indices ``i, j`` for which ``a[i, :] == b[j, :]``.

    Like _iter_common_sorted_arrays, but allows duplicate rows in `a`.
    I.e. `a.T` is lex-sorted, but not strictly. `b.T` is still assumed to be strictly lexsorted.
    """
    l_a, d_a = a.shape
    l_b, d_b = b.shape
    assert d_a == d_b
    i, j = 0, 0
    while i < l_a and j < l_b:
        for k in reversed(range(d_a)):
            if a[i, k] < b[j, k]:
                i += 1
                break
            elif b[j, k] < a[i, k]:
                j += 1
                break
        else:  # (no break)
            yield (i, j)
            # difference to _iter_common_sorted_arrays:
            # dont increase j because a[i + 1] might also match b[j]
            i += 1


def _iter_common_noncommon_sorted_arrays(a, b):
    """Yield the following pairs ``i, j`` of indices:

    - Matching entries, i.e. ``(i, j)`` such that ``all(a[i, :] == b[j, :])``
    - Entries only in `a`, i.e. ``(i, None)`` such that ``a[i, :]`` is not in `b`
    - Entries only in `b`, i.e. ``(None, j)`` such that ``b[j, :]`` is not in `a`

    *Assumes* that `a` and `b` are strictly lex-sorted (according to ``np.lexsort(a.T)``).
    """
    l_a, d_a = a.shape
    l_b, d_b = b.shape
    assert d_a == d_b
    i, j = 0, 0
    both = []  # TODO (JU) @jhauschild : this variable is unused? did something get lost while copying from old tenpy?
    while i < l_a and j < l_b:
        for k in reversed(range(d_a)):
            if a[i, k] < b[j, k]:
                yield i, None
                i += 1
                break
            elif a[i, k] > b[j, k]:
                yield None, j
                j += 1
                break
        else:
            yield i, j
            i += 1
            j += 1
    # can still have i < l_a or j < l_b, but not both
    for i2 in range(i, l_a):
        yield i2, None
    for j2 in range(j, l_b):
        yield None, j2
    # done


class AbelianBackendData:
    """Data stored in a Tensor for :class:`AbstractAbelianBackend`.

    Attributes
    ----------
    dtype : Dtype
        The dtype of the data
    blocks : list of block
        A list of blocks containing the actual entries of the tensor.
        Formerly known as Array._data.
    block_inds : 2D ndarray
        A 2D array of positive integers with shape (len(blocks), num_legs).
        The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of the `m`-th leg,
        that is to ``tensor.legs[m].sectors[block_inds[n, m]]``.
        By convention, we store `blocks` and `block_inds` such that ``np.lexsort(block_inds.T)``
        is sorted.

    Parameters
    ----------
    dtype, blocks, block_inds
        like attributes above, but not necessarily sorted
    is_sorted : bool
        If ``False`` (default), we permute `blocks` and `block_inds` according to ``np.lexsort(block_inds.T)``.
        If ``True``, we assume they are sorted *without* checking.
    """
    def __init__(self, dtype: Dtype, blocks: list[Block], block_inds: ndarray, is_sorted: bool = False):
        
        self.dtype = dtype
        self.blocks = blocks
        self.block_inds = block_inds
        if not is_sorted:
            self._sort_block_inds()

    def _sort_block_inds(self):
        """Bring `block_inds` (back) into the conventional sorted order.

        To speed up functions as tensordot, we always keep the blocks in a well-defined order
        where ``np.lexsort(block_inds.T)`` is trivial."""
        perm = np.lexsort(self.block_inds.T)
        self.block_inds = self.block_inds[perm, :]
        self.blocks = [self.blocks[p] for p in perm]

    def get_block_num(self, block_inds: ndarray) -> Block | None:
        """Return the index ``n`` of the block which matches the block_inds.

        I.e. such that ``all(self.block_inds[n, :] == block_inds)``.
        Return None if no such ``n`` exists.
        """
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
        return None if block_num is None else self.block[block_num]
        

class AbstractAbelianBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """Backend for Abelian group symmetries.

    Notes
    -----
    The data stored for the various tensor classes defined in ``tenpy.linalg.tensors`` is::

        - ``Tensor``:
            An ``AbelianBackendData`` instance whose blocks have as many axes as the tensor has legs.

        - ``DiagonalTensor`` :
            An ``AbelianBackendData`` instance whose blocks have only a single axis.
            This is the diagonal of the corresponding 2D block in a ``Tensor``.

        - ``Mask`` :
            An ``AbelianBackendData`` instance whose blocks have only a single axis and bool values.
            These bool values indicate which indices of the large leg are kept for the small leg.

    """
    DataCls = AbelianBackendData

    def test_data_sanity(self, a: Tensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_data_sanity(a, is_diagonal=is_diagonal)
        if a.data.block_inds.shape != (len(a.data.blocks), a.num_legs):
            msg = f'Wrong blocks_inds shape. ' \
                  f'Expected {(len(a.data.blocks), a.num_legs)}, got {a.data.block_inds.shape}.'
            raise ValueError(msg)
        if is_diagonal:
            assert np.all(a.data.block_inds[:, 0] == a.data.block_inds[:, 1])
        # check expected tensor dimensions
        block_shapes = np.array([leg.multiplicities[i] for leg, i in zip(a.legs, a.data.block_inds.T)]).T
        for block, shape in zip(a.data.blocks, block_shapes):
            expect_shape = (shape[0],) if is_diagonal else tuple(shape)
            assert self.block_shape(block) == expect_shape
        # check matching dtypes
        assert all(self.block_dtype(block) == a.data.dtype for block in a.data.blocks)
        assert not np.any(a.data.block_inds < 0)
        assert not np.any(a.data.block_inds >= np.array([[leg.num_sectors for leg in a.legs]]))

    def test_leg_sanity(self, leg: VectorSpace):
        assert self._leg_has_metadata(leg)
        # TODO should we do some consistency checks on the metadata shapes / values?
        super().test_leg_sanity(leg)

    def _leg_has_metadata(self, leg: VectorSpace) -> bool:
        if isinstance(leg, ProductSpace):
            return all(hasattr(leg, attr) for attr in ['_strides', '_block_ind_map_slices', '_block_ind_map', '_fusion_outcomes_inverse_sort'])
        else:
            return True

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool
                     ) -> tuple[SectorArray, ndarray, ndarray, dict]:
        r"""
        The abelian backend adds the following metadata:
            _strides : 1D numpy array of int
                F-style strides for the shape ``tuple(space.num_sectors for space in spaces)``.
                This allows one-to-one mapping between multi-indices (one block_ind per space) to a single index.
            _block_ind_map_slices : 1D numpy array of int
                Slices for embedding the unique fused sectors in the sorted list of all fusion outcomes.
                Shape is ``(K,)`` where ``K == product_space.num_sectors + 1``.
                Fusing all (non-dual) sectors of all spaces and sorting the outcomes gives a list
                which contains (in general) duplicates.
                The slice ``_block_ind_map_slices[n]:_block_ind_map_slices[n + 1]`` within this sorted
                list contains the same entry, namely ``product_space._non_dual_sectors[n]``.
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
        input `VectorSpace`s will end up getting placed in some slice :math:`a_j:a_{j+1}` of the
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
        if _is_dual:
            # overall fusion of sectors is equivalent to taking dual of each sector
            # in standard use cases, this can often avoid explicit
            # symmetry.dual_sector() calls in VectorSpace.sectors()
            fuse_sectors = [s.dual.sectors for s in spaces]
        else:
            fuse_sectors = [s.sectors for s in spaces]

        _non_dual_sectors = symmetry.multiple_fusion_broadcast(
            *(sectors[gr] for sectors, gr in zip(fuse_sectors, grid.T))
        )

        # sort (non-dual) charge sectors. Similar code as in VectorSpace.__init__
        fusion_outcomes_sort = np.lexsort(_non_dual_sectors.T)
        _block_ind_map = _block_ind_map[fusion_outcomes_sort]
        _non_dual_sectors = _non_dual_sectors[fusion_outcomes_sort]
        multiplicities = multiplicities[fusion_outcomes_sort]

        slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
        _block_ind_map[:, 0] = slices[:-1]  # start with 0
        _block_ind_map[:, 1] = slices[1:]

        # bunch sectors with equal charges together
        diffs = find_row_differences(_non_dual_sectors, include_len=True)
        metadata['_block_ind_map_slices'] = diffs
        slices = slices[diffs]
        multiplicities = slices[1:] - slices[:-1]
        diffs = diffs[:-1]

        _non_dual_sectors = _non_dual_sectors[diffs]

        new_block_ind = np.zeros(len(_block_ind_map), dtype=np.intp) # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        _block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        _block_ind_map[:, :2] -= slices[new_block_ind][:, np.newaxis]
        metadata['_block_ind_map'] = _block_ind_map  # finished
        
        return _non_dual_sectors, multiplicities, fusion_outcomes_sort, metadata

    def add_leg_metadata(self, leg: VectorSpace) -> VectorSpace:
        if isinstance(leg, ProductSpace):
            if not self._leg_has_metadata(leg):
                # OPTIMIZE write version that just calculates the metadata, without sectors?
                _, _, _, metadata = self._fuse_spaces(symmetry=leg.symmetry, spaces=leg.spaces, _is_dual=leg.is_dual)
                for key, val in metadata.items():
                    setattr(leg, key, val)
        # for non-ProductSpace: no metadata to add
        return leg
            
    def get_dtype_from_data(self, a: Data) -> Dtype:
        return a.dtype

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        # shallow copy if dtype stays same
        blocks = [self.block_to_dtype(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, blocks, a.data.block_inds, is_sorted=True)

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry.is_abelian and symmetry.braiding_style == BraidingStyle.bosonic

    def data_item(self, a: Data | DiagonalData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_item(a.blocks[0])

    def to_dense_block(self, a: Tensor) -> Block:
        res = self.zero_block([leg.dim for leg in a.legs], a.data.dtype)
        for block, block_inds in zip(a.data.blocks, a.data.block_inds):
            slices = [slice(*leg.slices[i]) for i, leg in zip(block_inds, a.legs)]
            res[tuple(slices)] = block
        return self.apply_basis_perm(res, a.legs, inv=True)

    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        res = self.zero_block([a.legs[0].dim], a.dtype)
        for block, block_idx in zip(a.data.blocks, a.data.block_inds[:, 0]):
            res[slice(*a.legs[0].slices[block_idx])] = block
        return self.apply_basis_perm(res, [a.legs[0]], inv=True)

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5) -> AbelianBackendData:
        a = self.apply_basis_perm(a, legs)
        projected = self.zero_block(self.block_shape(a), dtype=self.block_dtype(a))
        dtype = self.block_dtype(a)
        block_inds = _valid_block_indices(legs)
        blocks = []
        for b_i in block_inds:
            slices = tuple(slice(*leg.slices[i]) for i, leg in zip(b_i, legs))
            blocks.append(a[slices])
            projected[slices] = a[slices]
        if not self.block_allclose(a, projected, atol=atol, rtol=rtol):
            raise ValueError('Block is not symmetric up to tolerance.')
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def diagonal_from_block(self, a: Block, leg: VectorSpace) -> DiagonalData:
        a = self.apply_basis_perm(a, [leg])
        dtype = self.block_dtype(a)
        block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        blocks = [a[slice(*leg.slices[i])] for i in block_inds[:, 0]]
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def mask_from_block(self, a: Block, large_leg: VectorSpace) -> tuple[DiagonalData, VectorSpace]:
        # TODO thoroughly test this!
        a = self.block_to_dtype(a, Dtype.bool)
        a = self.apply_basis_perm(a, [large_leg])
        block_inds_large = []
        block_inds_small = []
        blocks = []
        small_leg_sectors = []
        small_leg_mults = []
        block_counter = 0
        for i in range(large_leg.num_sectors):
            block = a[slice(*large_leg.slices[i])]
            multiplicity = self.block_sum_all(block)
            if multiplicity == 0:
                continue
            blocks.append(block)
            block_inds_large.append(i)
            block_inds_small.append(block_counter)
            small_leg_sectors.append(large_leg._non_dual_sectors[i])
            small_leg_mults.append(multiplicity)
            block_counter += 1
        if len(blocks) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array([block_inds_large, block_inds_small]).T
        # OPTIMIZE (JU) block_inds might be sorted but i am not sure right now
        data = AbelianBackendData(dtype=Dtype.bool, blocks=blocks, block_inds=block_inds, is_sorted=False)
        small_leg = VectorSpace(
            symmetry=large_leg.symmetry, sectors=small_leg_sectors, multiplicities=small_leg_mults,
            is_real=large_leg.is_real, _is_dual=large_leg.is_dual
        )
        return data, small_leg

    def from_block_func(self, func, legs: list[VectorSpace], func_kwargs={}) -> AbelianBackendData:
        block_inds = _valid_block_indices(legs)
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i] for i, leg in zip(b_i, legs)]
            blocks.append(func(tuple(shape), **func_kwargs))
        if len(blocks) == 0:
            dtype = self.block_dtype(func((1,) * len(legs), **func_kwargs))
        else:
            dtype = self.block_dtype(blocks[0])
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def diagonal_from_block_func(self, func, leg: VectorSpace, func_kwargs={}) -> DiagonalData:
        block_inds = np.repeat(np.arange(leg.num_sectors)[:, None], 2, axis=1)
        blocks = [func((leg.multiplicities[i],), **func_kwargs) for i in block_inds[:, 0]]
        if len(blocks) == 0:
            dtype = self.block_dtype(func((1,), **func_kwargs))
        else:
            dtype = self.block_dtype(blocks[0])
        return AbelianBackendData(dtype, blocks, block_inds, is_sorted=True)

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> AbelianBackendData:
        return AbelianBackendData(dtype, blocks=[], block_inds=np.zeros((0, len(legs)), dtype=int), is_sorted=True)

    def zero_diagonal_data(self, leg: VectorSpace, dtype: Dtype) -> DiagonalData:
        return AbelianBackendData(dtype, blocks=[], block_inds=np.zeros((0, 2), dtype=int), is_sorted=True)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        block_inds = np.indices((leg.num_sectors for leg in legs)).T.reshape(-1, len(legs))
        # block_inds is by construction np.lexsort()-ed
        dims = [leg.multiplicities[bi] for leg, bi in zip(legs, block_inds.T)]
        blocks = [self.eye_block(shape, dtype) for shape in zip(*dims)]
        return AbelianBackendData(dtype, blocks, np.hstack([block_inds, block_inds]), is_sorted=True)

    def copy_data(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
        blocks = [self.block_copy(b) for b in self.blocks]
        return AbelianBackendData(a.data.dtype, blocks, a.data.block_inds.copy(), is_sorted=True)

    def _data_repr_lines(self, a: Tensor, indent: str, max_width: int, max_lines: int):
        from ..dummy_config import printoptions
        from ..misc import join_as_many_as_possible
        
        data = a.data
        if len(data.blocks) == 0:
            return [f'{indent}* Data : no non-zero blocks']
        if max_lines <= 1:
            return [f'{indent}* Data : Showing none of {len(data.blocks):d} blocks']

        line_start = f'{indent}* Data for sectors '

        if not printoptions.summarize_blocks:
            # try showing all blocks
            lines = []
            for block, block_inds in zip(data.blocks, data.block_inds):
                sectors = join_as_many_as_possible(
                    [a.symmetry.sector_str(leg.sectors[i]) for leg, i in zip(a.legs, block_inds)],
                    separator=', ', max_len=printoptions.linewidth - len(line_start) - 1
                )
                lines.append(f'{line_start}[{sectors}]:')
                lines.extend(self._block_repr_lines(block,
                                                    indent=indent + printoptions.indent * ' ',
                                                    max_width=max_width,
                                                    max_lines=max_lines))
                if len(lines) > max_lines:
                    break
            else:  # (no break ocurred)
                return lines
            

        # try showing shapes of all blocks
        lines = []
        for block, block_inds in zip(data.blocks, data.block_inds):
            sectors = join_as_many_as_possible(
                [a.symmetry.sector_str(leg.sectors[i]) for leg, i in zip(a.legs, block_inds)],
                separator=', ', max_len=printoptions.linewidth - len(line_start) - 1
            )
            shape = str(self.block_shape(block))
            if len(line_start) + len(sectors) + 10 + len(shape) <= printoptions.linewidth:
                lines.append(f'{line_start}[{sectors}]: shape {shape}')
            else:
                lines.append(f'{line_start}[{sectors}]: shape {shape}')
                lines.append(f'{indent}    shape {shape}')
            if len(lines) > max_lines:
                break
        else:  # (no break ocurred)
            return lines

        # only show shapes of largest blocks
        lines = []
        sizes = np.prod([self.block_shape(block) for block in data.blocks], axis=1)
        missing_blocks = len(a.data.blocks)
        for j in np.argsort(sizes):
            sectors = join_as_many_as_possible(
                [a.symmetry.sector_str(leg.sectors[i])
                 for leg, i in zip(a.legs, a.data.block_inds[j])],
                separator=', ', max_len=printoptions.linewidth - len(line_start) - 1
            )
            shape = str(self.block_shape(a.data.blocks[j]))
            if len(line_start) + len(sectors) + 10 + len(shape) <= printoptions.linewidth:
                new_lines = [f'{line_start}[{sectors}]: shape {shape}']
            else:
                new_lines = [f'{line_start}[{sectors}]: shape {shape}',
                             f'{indent}    shape {shape}']
            if len(lines) + len(new_lines) >= max_lines:
                lines.append(f'{indent}* Data for {missing_blocks} smaller blocks not shown')
                return lines
            lines.extend(new_lines)
            missing_blocks -= 1

        raise ValueError  # the above return should have triggered

    def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]) -> Data:
        #  Looking at the source of numpy's tensordot (which is just 62 lines of python code),
        #  you will find that it has the following strategy:

        #  1. Transpose `a` and `b` such that the axes to sum over are in the end of `a` and front of `b`.
        #  2. Combine the legs `axes`-legs and other legs with a `np.reshape`,
        #  such that `a` and `b` are matrices.
        #  3. Perform a matrix product with `np.dot`.
        #  4. Split the remaining axes with another `np.reshape` to obtain the correct shape.

        #  The main work is done by `np.dot`, which calls LAPACK to perform the simple matrix product.
        #  [This matrix multiplication of a ``NxK`` times ``KxM`` matrix is actually faster
        #  than the O(N*K*M) needed by a naive implementation looping over the indices.]

        #  We follow the same overall strategy data block entries.
        #  Step 1) is performed by :meth:`_tdot_transpose_axes`

        #  The steps 2) and 4) could be implemented with `combine_legs` and `split_legs`.
        #  However, that would actually be an overkill: we're not interested
        #  in the full charge data of the combined legs (which would be generated in the LegPipes).
        #  Instead, we just need to track the block_inds of a and b carefully.

        #  Our step 2) is implemented in :meth:`_tdot_pre_worker`:
        #  We split `a.data.block_inds` into `a_block_inds_keep` and `a_block_inds_contr`, and similar for `b`.
        #  Then, view `a` is a matrix :math:`A_{i,k1}` and `b` as :math:`B_{k2,j}`, where
        #  `i` can be any row of `a_block_inds_keep`, `j` can be any row of `b_block_inds_keep`.
        #  The `k1` and `k2` are rows/columns of `a/b_block_inds_contr`, which come from compatible dual legs.
        #  In our storage scheme, `a.data.blocks[s]` then contains the block :math:`A_{i,k1}` for
        #  ``j = a_block_inds_keep[s]`` and ``k1 = a_block_inds_contr[s]``.
        #  To identify the different indices `i` and `j`, it is easiest to lexsort in the `s`.
        #  Note that we give priority to the `{a,b}_block_inds_keep` over the `_contr`, such that
        #  equal rows of `i` are contiguous in `a_block_inds_keep`.
        #  Then, they are identified with :func:`find_row_differences`.

        #  Now, the goal is to calculate the sums :math:`C_{i,j} = sum_k A_{i,k} B_{k,j}`,
        #  analogous to step 3) above. This is implemented directly in this function.
        #  It is done 'naively' by explicit loops over ``i``, ``j`` and ``k``.
        #  However, this is not as bad as it sounds:
        #  First, we loop only over existent ``i`` and ``j``
        #  (in the sense that there is at least some non-zero block with these ``i`` and ``j``).
        #  Second, if the ``i`` and ``j`` are not compatible with the new total charge,
        #  we know that ``C_{i,j}`` will be zero.
        #  Third, given ``i`` and ``j``, the sum over ``k`` runs only over
        #  ``k1`` with nonzero :math:`A_{i,k1}`, and ``k2` with nonzero :math:`B_{k2,j}`.

        #  How many multiplications :math:`A_{i,k} B_{k,j}` we actually have to perform
        #  depends on the sparseness. If ``k`` comes from a single leg, it is completely sorted
        #  by charges, so the 'sum' over ``k`` will contain at most one term!

        # note: tensor.tdot() checks special-cases inner and outer, so it's at least a mat-vec
        open_axs_a = [idx for idx in range(a.num_legs) if idx not in axs_a]
        open_axs_b = [idx for idx in range(b.num_legs) if idx not in axs_b]
        assert len(open_axs_a) > 0 or len(open_axs_b) > 0, "special case inner() in tensor.tdot()"
        assert len(axs_a) > 0, "special case outer() in tensor.tdot()"

        if len(a.data.blocks) == 0 or len(b.data.blocks) == 0:
            dtype = a.data.dtype.common(b.data.dtype)
            return self.zero_data([a.legs[i] for i in open_axs_a] + [b.legs[i] for i in open_axs_b], dtype)

        # for details on the implementation, see _tensordot_worker.
        # Step 1: transpose if necessary:
        a, b, contr_axes = self._tdot_transpose_axes(a, b, open_axs_a, axs_a, axs_b, open_axs_b)
        # now we need to contract the last `contr_axes` of a with the first `contr_axes` of b

        # Step 2:
        cut_a = a.num_legs - contr_axes
        cut_b = contr_axes
        a_pre_result, b_pre_result, res_dtype = self._tdot_pre_worker(a, b, cut_a, cut_b)
        a_blocks, a_block_inds_contr, a_block_inds_keep, a_shape_keep = a_pre_result
        b_blocks, b_block_inds_contr, b_block_inds_keep, b_shape_keep = b_pre_result

        # Step 3) loop over column/row of the result
        sym = a.legs[0].symmetry
        if cut_a > 0:
            a_charges_keep = sym.multiple_fusion_broadcast(
                *(leg.sectors[i] for leg, i in zip(a.legs[:cut_a], a_block_inds_keep.T))
            )
        else:
            a_charges_keep = np.zeros((len(a_block_inds_keep), sym.sector_ind_len), int)
        if cut_b < b.num_legs:
            b_charges_keep_dual = sym.multiple_fusion_broadcast(
                *(leg.dual.sectors[i] for leg, i in zip(b.legs[cut_b:], b_block_inds_keep.T))
            )
        else:
            b_charges_keep_dual = np.zeros((len(b_block_inds_keep), sym.sector_ind_len), int)
        # dual such that b_charges_keep_dual must match a_charges_keep
        a_lookup_charges = list_to_dict_list(a_charges_keep)  # lookup table ``charge -> [row_a]``

        # (rows_a changes faster than cols_b, such that the resulting array is qdata lex-sorted)
        # determine output qdata
        res_blocks = []
        res_block_inds_a = []
        res_block_inds_b = []
        for col_b, charge_match in enumerate(b_charges_keep_dual):
            b_blocks_in_col = b_blocks[col_b]
            rows_a = a_lookup_charges.get(tuple(charge_match), [])  # empty list if no match
            for row_a in rows_a:
                ks = _iter_common_sorted(a_block_inds_contr[row_a], b_block_inds_contr[col_b])
                if len(ks) == 0:
                    continue
                a_blocks_in_row = a_blocks[row_a]
                k1, k2 = ks[0]
                block_contr = self.matrix_dot(a_blocks_in_row[k1], b_blocks_in_col[k2])
                for k1, k2 in ks[1:]:
                    block_contr = block_contr + self.matrix_dot(a_blocks_in_row[k1],
                                                                b_blocks_in_col[k2])

                # Step 4) reshape back to tensors
                block_contr = self.block_reshape(block_contr, a_shape_keep[row_a] + b_shape_keep[col_b])
                res_blocks.append(block_contr)
                res_block_inds_a.append(a_block_inds_keep[row_a])
                res_block_inds_b.append(b_block_inds_keep[col_b])
        if len(res_blocks) == 0:
            return self.zero_data(a.legs[:cut_a] + b.legs[cut_b:], res_dtype)
        block_inds = np.hstack((res_block_inds_a, res_block_inds_b))
        return AbelianBackendData(res_dtype, res_blocks, block_inds, is_sorted=True)

    def _tdot_transpose_axes(self, a: Tensor, b: Tensor, open_axs_a, axs_a, axs_b, open_axs_b):
        contr_axes = len(axs_a)
        open_a = len(open_axs_a)
        # try to be smart and avoid unnecessary transposes
        last_axes_a = all(i >= open_a for i in axs_a)
        first_axes_b = all(i < contr_axes for i in axs_b)
        if last_axes_a and first_axes_b:
            # we contract only last axes of a and first axes of b
            axs_a_order = [i - open_a for i in axs_a]
            if all(i == j for i, j in zip(axs_a_order, axs_b)):
                return a, b, contr_axes  # no transpose necessary
                # (doesn't matter if axs_b is not ordered)
            # it's enough to transpose one of the arrays!
            # let's sort axs_a and only transpose axs_b  # TODO optimization: choose depending on size of a/b?
            axs_b = [axs_b[i] for i in np.argsort(axs_a)]
            b = b.permute_legs(axs_b + open_axs_b)
            return a, b, contr_axes
        if last_axes_a:
            # no need to transpose a
            axs_b = [axs_b[i] for i in np.argsort(axs_a)]
            b = b.permute_legs(axs_b + open_axs_b)
            return a, b, contr_axes
        elif first_axes_b:
            # no need to transpose b
            axs_a = [axs_a[i] for i in np.argsort(axs_b)]
            a = a.permute_legs(open_axs_a + axs_a)
            return a, b, contr_axes
        # no special case to avoid transpose -> transpose both
        a = a.permute_legs(open_axs_a + axs_a)
        b = b.permute_legs(axs_b + open_axs_b)
        return a, b, contr_axes

    def _tdot_pre_worker(self, a: Tensor, b: Tensor, cut_a:int, cut_b: int):
        """Pre-calculations before the actual matrix product of tdot.

        Called by :meth:`_tensordot_worker`.
        See doc-string of :meth:`tdot` for details on the implementation.

        Returns
        -------
        a_pre_result, b_pre_result : tuple
            In the following order, it contains for `a`, and `b` respectively:
            a_blocks : list of list of reshaped tensors
            a_block_inds_contr : 2D array with block indices of `a` which we need to sum over
            a_block_inds_keep : 2D array of the block indices of `a` which will appear in the final result
            a_slices : partition to map the indices of a_*_keep to a_data
        res_dtype : np.dtype
            The data type which should be chosen for the result.
            (The `dtype` of the ``s`` above might differ from `res_dtype`!).
        """
        # convert block_inds_contr over which we sum to a 1D array for faster lookup/iteration
        # F-style strides to preserve sorting
        stride = make_stride([l.num_sectors for l in a.legs[cut_a:]], cstyle=False)
        a_block_inds_contr = np.sum(a.data.block_inds[:, cut_a:] * stride, axis=1)
        # lex-sort a.data.block_inds, dominated by the axes kept, then the axes summed over.
        a_sort = np.lexsort(np.hstack([a_block_inds_contr[:, np.newaxis], a.data.block_inds[:, :cut_a]]).T)
        a_block_inds_keep = a.data.block_inds[a_sort, :cut_a]
        a_block_inds_contr = a_block_inds_contr[a_sort]
        a_blocks = a.data.blocks
        a_blocks = [a_blocks[i] for i in a_sort]
        # combine all b_block_inds[:cut_b] into one column (with the same stride as before)
        b_block_inds_contr = np.sum(b.data.block_inds[:, :cut_b] * stride, axis=1)
        # b_block_inds is already lex-sorted, dominated by the axes kept, then the axes summed over
        b_blocks = b.data.blocks
        b_block_inds_keep = b.data.block_inds[:, cut_b:]
        # find blocks where block_inds_a[not_axes_a] and block_inds_b[not_axes_b] change
        a_slices = find_row_differences(a_block_inds_keep, include_len=True)
        b_slices = find_row_differences(b_block_inds_keep, include_len=True)
        # the slices divide a_blocks and b_blocks into rows and columns of the final result
        a_blocks = [a_blocks[i:i2] for i, i2 in zip(a_slices[:-1], a_slices[1:])]
        b_blocks = [b_blocks[j:j2] for j, j2 in zip(b_slices[:-1], b_slices[1:])]
        a_block_inds_contr = [a_block_inds_contr[i:i2] for i, i2 in zip(a_slices[:-1], a_slices[1:])]
        b_block_inds_contr = [b_block_inds_contr[i:i2] for i, i2 in zip(b_slices[:-1], b_slices[1:])]
        a_block_inds_keep = a_block_inds_keep[a_slices[:-1]]
        b_block_inds_keep = b_block_inds_keep[b_slices[:-1]]
        a_shape_keep = [blocks[0].shape[:cut_a] for blocks in a_blocks]
        b_shape_keep = [blocks[0].shape[cut_b:] for blocks in b_blocks]

        res_dtype = a.data.dtype.common(b.data.dtype)
        if a.data.dtype != res_dtype:
            a_blocks = [[self.block_to_dtype(T, res_dtype) for T in blocks] for blocks in a_blocks]
        if b.data.dtype != res_dtype:
            b_blocks = [[self.block_to_dtype(T, res_dtype) for T in blocks] for blocks in b_blocks]
        # reshape a_blocks and b_blocks to matrix/vector
        a_blocks = self._tdot_pre_reshape(a_blocks, cut_a, a.num_legs)
        b_blocks = self._tdot_pre_reshape(b_blocks, cut_b, b.num_legs)

        # collect and return the results
        a_pre_result = a_blocks, a_block_inds_contr, a_block_inds_keep, a_shape_keep
        b_pre_result = b_blocks, b_block_inds_contr, b_block_inds_keep, b_shape_keep
        return a_pre_result, b_pre_result, res_dtype

    def _tdot_pre_reshape(self, blocks_list, cut, num_legs):
        """Reshape blocks to (fortran) matrix/vector (depending on `cut`)"""
        if cut == 0 or cut == num_legs:
            # special case: reshape to 1D vectors
            return [[self.block_reshape(T, (-1,)) for T in blocks]
                    for blocks in blocks_list]
        res = [[self.block_reshape(T, (np.prod(self.block_shape(T)[:cut]), -1)) for T in blocks]
                for blocks in blocks_list]
        return res

    def svd(self, a: Tensor, new_vh_leg_dual: bool, algorithm: str | None) -> tuple[Data, DiagonalData, Data, VectorSpace]:
        u_blocks = []
        s_blocks = []
        vh_blocks = []
        for block in a.data.blocks:
            u, s, vh = self.matrix_svd(block, algorithm=algorithm)
            u_blocks.append(u)
            s_blocks.append(s)
            assert len(s) > 0
            vh_blocks.append(vh)
        
        leg_L, leg_R = a.legs
        symmetry = a.legs[0].symmetry
        block_inds_L, block_inds_R = a.data.block_inds.T  # columns of block_inds
        # due to lexsort(a.data.block_inds.T), block_inds_R is sorted, but block_inds_L not.

        # build new leg: add sectors in the order given by block_inds_R, which is sorted
        leg_C_sectors = leg_R._non_dual_sectors[block_inds_R]
        # economic SVD (aka full_matrices=False) : len(s) = min(block.shape)
        leg_C_mults = np.minimum(leg_L.multiplicities[block_inds_L], leg_R.multiplicities[block_inds_R])
        block_inds_C = np.arange(len(s_blocks), dtype=int)
        if new_vh_leg_dual != leg_R.is_dual:
            # opposite dual flag in legs of vH => same _sectors
            new_leg = VectorSpace(symmetry, leg_C_sectors, leg_C_mults, is_real=leg_R.is_real, _is_dual=new_vh_leg_dual)
        else:  # new_vh_leg_dual == leg_R.is_dual
            # same dual flag in legs of vH => opposite _sectors => opposite sorting!!!
            leg_C_sectors = symmetry.dual_sectors(leg_C_sectors)  # not sorted
            sort = np.lexsort(leg_C_sectors.T)
            block_inds_C = block_inds_C[sort]
            new_leg = VectorSpace(symmetry, leg_C_sectors[sort], leg_C_mults[sort], is_real=leg_R.is_real, _is_dual=new_vh_leg_dual)
            
        u_block_inds = np.column_stack([block_inds_L, block_inds_C])
        s_block_inds = np.repeat(block_inds_C[:, None], 2, axis=1)
        vh_block_inds = np.column_stack([block_inds_C, block_inds_R])
        if new_vh_leg_dual == leg_R.is_dual:
            # need to sort u_block_inds and s_block_inds
            # since we lexsort with last column changing slowest, we need to sort block_inds_C only
            sort = np.argsort(block_inds_C)
            u_block_inds = u_block_inds[sort, :]
            s_block_inds = s_block_inds[sort, :]
            u_blocks = [u_blocks[i] for i in sort]
            s_blocks = [s_blocks[i] for i in sort]

        dtype = a.data.dtype
        return (AbelianBackendData(dtype, u_blocks, u_block_inds, is_sorted=True),
                AbelianBackendData(dtype.to_real, s_blocks, s_block_inds, is_sorted=True),
                AbelianBackendData(dtype, vh_blocks, vh_block_inds, is_sorted=True),
                new_leg)

    def qr(self, a: Tensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, VectorSpace]:
        q_leg_0, r_leg_1 = a.legs
        q_blocks = []
        r_blocks = []
        for block in a.data.blocks:
            q, r = self.matrix_qr(block, full=full)
            q_blocks.append(q)
            r_blocks.append(r)
        sym = a.symmetry
        if full:
            new_leg = q_leg_0.as_VectorSpace()
            if new_leg.is_dual != new_r_leg_dual:
                # taking the dual leaves _non_sorted_dual_sectors unaffected and
                # thus we dont need to adjust anything else
                new_leg = new_leg.dual
            # sort q_blocks
            q_blocks_full = [None] * q_leg_0.num_sectors
            for i, q in zip(a.data.block_inds[:, 0], q_blocks):
                q_blocks_full[i] = q
            if len(q_blocks) < q_leg_0.num_sectors:
                # there is a block-column in `a` that is completely 0 and not in a.data.blocks
                # so we need to add corresponding identity blocks in q to ensure q is unitary!
                dtype = a.data.dtype
                for i, q in enumerate(q_blocks_full):
                    if q is None:
                        q_blocks_full[i] = self.eye_block([q_leg_0.multiplicities[i]], dtype)
            q_block_inds = np.repeat(np.arange(q_leg_0.num_sectors, dtype=int)[:, None], 2, axis=1)  # sorted
            r_block_inds = a.data.block_inds.copy() # is already sorted...

            q_data = AbelianBackendData(a.data.dtype, q_blocks_full, q_block_inds, is_sorted=True)
            r_data = AbelianBackendData(a.data.dtype, r_blocks, r_block_inds, is_sorted=True)
        else:
            keep = a.data.block_inds[:, 0]

            # fix the order of sectors on the new leg: by order of appearance in q_leg_0
            keep_perm = np.argsort(keep)
            keep_sorted = keep[keep_perm]  # this fixes the order! "by order of appearance in q_leg_0"

            new_leg_sectors = q_leg_0._non_dual_sectors[keep_sorted, :]  # this is lexsort(x.T)-ed
            new_leg_mults = np.array([self.block_shape(q)[1] for q in q_blocks], int)[keep_perm]
            new_leg = VectorSpace(sym, new_leg_sectors, new_leg_mults, is_real=q_leg_0.is_real,
                                  _is_dual=new_r_leg_dual)

            # determine block_inds for the new leg:
            # for q_blocks[i], the relevant sector is the same as the sector on the 0 leg:
            # q_leg_0._non_dual_sectors[a.data.block_inds[i, 0]]
            #  == q_leg_0._non_dual_sectors[keep[i]]
            #  == q_leg_0._non_dual_sectors[keep_sorted[inv_keep_perm[i]]]
            #  == new_leg_sectors[inv_keep_perm[i]]
            #  == new_leg._non_dual_sectors[inv_keep_perm[i]]
            # Thus we have
            new_block_inds = inverse_permutation(keep_perm)
            
            q_block_inds = np.hstack([keep[:, None], new_block_inds[:, None]])  # not sorted.
            r_block_inds = np.hstack([new_block_inds[:, None], a.data.block_inds[:, 1:2]])  # lexsorted.
            assert np.all(np.lexsort(r_block_inds.T) == np.arange(len(r_block_inds)))  # TODO remove test
            q_data = AbelianBackendData(a.dtype, q_blocks, q_block_inds, is_sorted=False)
            r_data = AbelianBackendData(a.dtype, r_blocks, r_block_inds, is_sorted=True)

        return q_data, r_data, new_leg

    def outer(self, a: Tensor, b: Tensor) -> Data:
        res_dtype = a.data.dtype.common(b.data.dtype)
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        if a.data.dtype != res_dtype:
            a_blocks = [self.block_to_dtype(T, res_dtype) for T in a_blocks]
        if b.data.dtype != res_dtype:
            b_blocks = [self.block_to_dtype(T, res_dtype) for T in b_blocks]
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        l_a, num_legs_a = a_block_inds.shape
        l_b, num_legs_b = b_block_inds.shape
        grid = np.indices([len(a_block_inds), len(b_block_inds)]).T.reshape(-1, 2)
        # grid is lexsorted, with rows as all combinations of a/b block indices.
        res_block_inds = np.empty((l_a * l_b, num_legs_a + num_legs_b), dtype=int)
        res_block_inds[:, :num_legs_a] = a_block_inds[grid[:, 0]]
        res_block_inds[:, num_legs_a:] = b_block_inds[grid[:, 1]]

        res_blocks = [self.block_outer(a_blocks[i], b_blocks[j]) for i, j in grid]

        # TODO (JU) are the block_inds actually sorted?
        #  if yes: add comment explaining why, adjust argument below
        return AbelianBackendData(res_dtype, res_blocks, res_block_inds, is_sorted=False)

    def inner(self, a: Tensor, b: Tensor, do_conj: bool, axs2: list[int] | None) -> complex:
        # a.legs[i] to be contracted with b.legs[axs2[i]]
        a_blocks = a.data.blocks
        stride = make_stride([l.num_sectors for l in a.legs], cstyle=False)
        a_block_inds = np.sum(a.data.block_inds * stride, axis=1)
        if axs2 is not None:
            # permute strides to match the label order on b:
            # strides_for_a[i] == strides_for_b[axs2[i]]
            stride[axs2] = stride.copy()
        b_blocks = b.data.blocks
        b_block_inds = np.sum(b.data.block_inds * stride, axis=1)
        if axs2 is not None:
            # we permuted the strides, so b_block_inds is no longer sorted
            sort = np.argsort(b_block_inds)
            b_block_inds = b_block_inds[sort]
            b_blocks = [b_blocks[i] for i in sort]
        res = [self.block_inner(a_blocks[i], b_blocks[j], do_conj=do_conj, axs2=axs2)
               for i, j in _iter_common_sorted(a_block_inds, b_block_inds)]
        return np.sum(res)

    def permute_legs(self, a: Tensor, permutation: list[int]) -> Data:
        blocks = a.data.blocks
        blocks = [self.block_permute_axes(block, permutation) for block in a.data.blocks]
        block_inds = a.data.block_inds[:, permutation]
        data = AbelianBackendData(a.data.dtype, blocks, block_inds, is_sorted=False)
        return data

    def trace_full(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        a_blocks = a.data.blocks
        a_block_inds_1 = a.data.block_inds[:, idcs1]
        a_block_inds_2 = a.data.block_inds[:, idcs2]
        total_sum = a.data.dtype.zero_scalar
        for block, i1, i2 in zip(a_blocks, a_block_inds_1, a_block_inds_2):
            # if len(idcs1) == 1, i1==i2 due to charge conservation,
            # but for multi-dimensional indices not clear
            if np.all(i1 == i2):
                total_sum += self.block_trace_full(block, idcs1, idcs2)
        return total_sum

    def trace_partial(self, a: Tensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        a_blocks = a.data.blocks
        a_block_inds_1 = a.data.block_inds[:, idcs1]
        a_block_inds_2 = a.data.block_inds[:, idcs2]
        a_block_inds_rem = a.data.block_inds[:, remaining_idcs]
        res_data = {}  # dictionary res_block_inds_row -> Block
        for block, i1, i2, ir in zip(a_blocks, a_block_inds_1, a_block_inds_2, a_block_inds_rem):
            if not np.all(i1 == i2):
                continue
            ir = tuple(ir)
            block = self.block_trace_partial(block, idcs1, idcs2, remaining_idcs)
            add_block = res_data.get(ir, None)
            if add_block is not None:
                block = block + add_block
            res_data[ir] = block
        res_blocks = list(res_data.values())
        if len(res_blocks) == 0:
            return self.zero_data([a.legs[i] for i in remaining_idcs], a.data.dtype)
        res_block_inds = np.array(list(res_data.keys()), dtype=int)
        return AbelianBackendData(a.data.dtype, res_blocks, res_block_inds, is_sorted=False)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        a_blocks = a.data.blocks
        total_sum = a.data.dtype.zero_scalar
        for block in a_blocks:
            total_sum += self.block_sum_all(block)
        return total_sum

    def conj(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
        blocks = [self.block_conj(b) for b in a.data.blocks]
        return AbelianBackendData(a.data.dtype, blocks, a.data.block_inds, is_sorted=True)

    def combine_legs(self, a: Tensor, combine_slices: list[int, int], product_spaces: list[ProductSpace],
                     new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        res_dtype = a.data.dtype
        old_block_inds = a.data.block_inds
        # first, find block indices of the final array to which we map
        map_inds = [product_space_map_incoming_block_inds(product_space, old_block_inds[:, b:e])
                    for product_space, (b,e) in zip(product_spaces, combine_slices)]
        old_block_inds = a.data.block_inds
        old_blocks = a.data.blocks
        res_block_inds = np.empty((len(old_block_inds), len(final_legs)), dtype=int)
        last_e = 0
        last_i = -1
        for i, (b, e), product_space, map_ind in zip(new_axes, combine_slices, product_spaces, map_inds):
            res_block_inds[:, last_i + 1:i] = old_block_inds[:, last_e:b]
            res_block_inds[:, i] = product_space._block_ind_map[map_ind, -1]
            last_e = e
            last_i = i
        res_block_inds[:, last_i + 1:] = old_block_inds[:, last_e:]

        # now we have probably many duplicate rows in res_block_inds, since many combinations of
        # non-combined block indices map to the same block index in product space
        # -> find unique entries by sorting res_block_inds
        sort = np.lexsort(res_block_inds.T)
        res_block_inds = res_block_inds[sort]
        old_blocks = [old_blocks[i] for i in sort]
        map_inds = [map_[sort] for map_ in map_inds]

        # determine slices in the new blocks
        block_slices = np.zeros((len(old_blocks), len(final_legs), 2), int)
        block_shape = np.empty((len(old_blocks), len(final_legs)), int)
        for i, leg in enumerate(final_legs):  # legs not in new_axes
            if i not in new_axes:
                # block_slices[:, i, 0] = 0
                block_slices[:, i, 1] = block_shape[:, i] = leg.multiplicities[res_block_inds[:, i]]
        for i, product_space, map_ind in zip(new_axes, product_spaces, map_inds):  # legs in new_axes
            slices = product_space._block_ind_map[map_ind, :2]
            block_slices[:, i, :] = slices
            block_shape[:, i] = slices[:, 1] - slices[:, 0]

        # split res_block_inds into parts, which give a unique new blocks
        diffs = find_row_differences(res_block_inds, include_len=True)  # including 0 and len to have slices later
        res_num_blocks = len(diffs) - 1
        res_block_inds = res_block_inds[diffs[:res_num_blocks], :]
        res_block_shapes = np.empty((res_num_blocks, len(final_legs)), int)
        for i, leg in enumerate(final_legs):
            res_block_shapes[:, i] = leg.multiplicities[res_block_inds[:, i]]

        # now the hard part: map data
        res_blocks = []
        # iterate over ranges of equal qindices in qdata
        for res_block_shape, beg, end in zip(res_block_shapes, diffs[:-1], diffs[1:]):
            new_block = self.zero_block(res_block_shape, dtype=res_dtype)
            for old_row in range(beg, end):  # copy blocks
                shape = block_shape[old_row]  # this has multiplied dimensions for combined legs
                old_block = self.block_reshape(old_blocks[old_row], shape)
                new_slices = tuple(slice(b, e) for b, e in block_slices[old_row])

                new_block[new_slices] = old_block  # actual data copy

            res_blocks.append(new_block)

        # we lexsort( .T)-ed res_block_inds while it still had duplicates, and then indexed by diffs,
        # which is sorted and thus preserves lexsort( .T)-ing of res_block_inds
        return AbelianBackendData(res_dtype, res_blocks, res_block_inds, is_sorted=True)

    def split_legs(self, a: Tensor, leg_idcs: list[int], final_legs: list[VectorSpace]) -> Data:
        # TODO (JH) below, we implement it by first generating the block_inds of the splitted tensor and
        # then extract subblocks from the original one.
        # Why not go the other way around and implement
        # block_views = self.block_split(block, block_sizes, axis) similar as np.array_split()
        # and call that for each axis to be split for each block?
        # we do a lot of numpy index tricks below, but does that really save time for splitting?
        # block_split should just be views anyways, not data copies?

        if len(a.data.blocks) == 0:
            return self.zero_data(final_legs, a.data.dtype)
        n_split = len(leg_idcs)
        product_spaces = [a.legs[i] for i in leg_idcs]
        res_num_legs = len(final_legs)

        old_blocks = a.data.blocks
        old_block_inds = a.data.block_inds

        map_slices_beg = np.zeros((len(old_blocks), n_split), int)
        map_slices_shape = np.zeros((len(old_blocks), n_split), int)  # = end - beg
        for j, product_space in enumerate(product_spaces):
            block_inds_j = old_block_inds[:, leg_idcs[j]]
            map_slices_beg[:, j] = product_space._block_ind_map_slices[block_inds_j]
            sizes = product_space._block_ind_map_slices[1:] - product_space._block_ind_map_slices[:-1]
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
                _block_ind_map = product_space._block_ind_map[map_rows[:, j], :]
                new_block_inds[:, k:k2] = _block_ind_map[:, 2:-1]
                old_block_beg[:, i] = _block_ind_map[:, 0]
                old_block_shapes[:, i] = _block_ind_map[:, 1] - _block_ind_map[:, 0]
                shift += len(product_space.spaces) - 1
                j += 1
            else:
                new_block_inds[:, i + shift] = nbi = old_block_inds[old_rows, i]
                old_block_shapes[:, i] = a.legs[i].multiplicities[nbi]
        # sort new_block_inds
        # OPTIMIZE (JU) could also skip sorting here and put is_sorted=False in AbelianBackendData(..) below?
        sort = np.lexsort(new_block_inds.T)
        new_block_inds = new_block_inds[sort, :]
        old_block_beg = old_block_beg[sort]
        old_block_shapes = old_block_shapes[sort]
        old_rows = old_rows[sort]

        new_block_shapes = np.empty((res_num_blocks, res_num_legs), dtype=int)
        for i, leg in enumerate(final_legs):
            new_block_shapes[:, i] = leg.multiplicities[new_block_inds[:, i]]

        # the actual loop to split the blocks
        new_blocks = []
        for i in range(res_num_blocks):
            old_block = old_blocks[old_rows[i]]
            slices = tuple(slice(b, b + s) for b, s in zip(old_block_beg[i], old_block_shapes[i]))
            new_block = old_block[slices]
            new_blocks.append(self.block_reshape(new_block, new_block_shapes[i]))

        return AbelianBackendData(a.data.dtype, new_blocks, new_block_inds, is_sorted=True)

    def add_trivial_leg(self, a: Tensor, pos: int) -> Data:
        blocks = [self.block_add_axis(block, pos) for block in a.data.blocks]
        block_inds = np.insert(a.data.block_inds, pos, 0, axis=1)
        # since the new column is constant, block_inds are still sorted.
        return AbelianBackendData(a.data.dtype, blocks, block_inds, is_sorted=True)

    def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        for i, j in _iter_common_noncommon_sorted_arrays(a.data.block_inds, b.data.block_inds):
            if j is None:
                if self.block_max_abs(a_blocks[i]) > atol:
                    return False
            elif i is None:
                if self.block_max_abs(b_blocks[j]) > atol:
                    return False
            else:
                if not self.block_allclose(a_blocks[i], b_blocks[j], rtol=rtol, atol=atol):
                    return False
        return True

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        n_legs = a.num_legs
        if len(a.data.blocks) == 0:
            block_inds = np.zeros([0, n_legs - len(idcs)], dtype=int)
            return AbelianBackendData(a.data.dtype, [], block_inds, is_sorted=True)
        blocks = [self.block_squeeze_legs(b, idcs) for b in a.data.blocks]
        block_inds = a.data.block_inds
        symmetry = a.legs[0].symmetry
        sector = symmetry.trivial_sector
        for i in idcs:
            bi = block_inds[0, i]
            assert np.all(block_inds[:, i] == bi)
            sector = symmetry.fusion_outcomes(sector, a.legs[i].sector(bi))[0]
        if not np.all(sector == symmetry.trivial_sector):
            # TODO return corresponding ChargedTensor instead in this case?
            raise ValueError("Squeezing legs drops non-trivial charges, would give ChargedTensor.")
        keep = np.ones(n_legs, dtype=bool)
        keep[idcs] = False
        block_inds = block_inds[:, keep]
        return AbelianBackendData(a.data.dtype, blocks, block_inds, is_sorted=True)

    def norm(self, a: Tensor | DiagonalTensor, order: int | float = None) -> float:
        block_norms = [self.block_norm(b, order=order) for b in a.data.blocks]
        return np.linalg.norm(block_norms, ord=order)

    def act_block_diagonal_square_matrix(self, a: Tensor, block_method: Callable[[Block], Block]
                                         ) -> Data:
        a_block_inds = a.data.block_inds
        all_block_inds = np.repeat(np.arange(a.legs[0].num_sectors)[:, None], 2, axis=1)  # [[0, 0], [1, 1], ...]
        res_blocks = []
        for i, j in _iter_common_noncommon_sorted_arrays(a_block_inds, all_block_inds):
            if i is None:
                # use that all_block_inds is just ascending -> all_block_inds[j, 0] == j
                block = self.zero_block(shape=[a.legs[0].multiplicities[j]] * 2, dtype=a.dtype)
            else:
                block = a.data.blocks[i]
            res_blocks.append(block_method(block))
        dtype = Dtype.common(*(self.block_dtype(block) for block in res_blocks))
        res_blocks = [self.block_to_dtype(block, dtype) for block in res_blocks]
        return AbelianBackendData(dtype, res_blocks, all_block_inds, is_sorted=True)

    def add(self, a: Tensor, b: Tensor) -> Data:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        # ensure common dtypes
        common_dtype = a.dtype.common(b.dtype)
        if a.data.dtype != common_dtype:
            a_blocks = [self.block_to_dtype(T, common_dtype) for T in a_blocks]
        if b.data.dtype != common_dtype:
            b_blocks = [self.block_to_dtype(T, common_dtype) for T in b_blocks]
        res_blocks = []
        res_block_inds = []
        for i, j in _iter_common_noncommon_sorted_arrays(a_block_inds, b_block_inds):
            if j is None:
                res_blocks.append(a_blocks[i])
                res_block_inds.append(a_block_inds[i])
            elif i is None:
                res_blocks.append(b_blocks[j])
                res_block_inds.append(b_block_inds[j])
            else:
                res_blocks.append(self.block_add(a_blocks[i], b_blocks[j]))
                res_block_inds.append(a_block_inds[i])
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, a.num_legs), int)
        return AbelianBackendData(common_dtype, res_blocks, res_block_inds, is_sorted=True)

    def mul(self, a: float | complex, b: Tensor) -> Data:
        if a == 0.:
            return self.zero_data(b.legs, b.data.dtype)
        res_blocks = [self.block_mul(a, T) for T in b.data.blocks]
        res_dtype = b.data.dtype if len(res_blocks) == 0 else self.block_dtype(res_blocks[0])
        return AbelianBackendData(res_dtype, res_blocks, b.data.block_inds, is_sorted=True)

    def infer_leg(self, block: Block, legs: list[VectorSpace | None], is_dual: bool = False,
                  is_real: bool = False) -> VectorSpace:
        raise NotImplementedError  # TODO
        # TODO how to handle ChargedTensor vs Tensor?
        #  JU: dont need to consider ChargedTensor here.
        #      When e.g. a ChargedTensor classmethod wants to infer the charge, it should add a
        #      one-dimensional axis to the block. then, this block is "uncharged", i.e. it will
        #      become the ``invariant_part: Tensor`` of the ChargedTensor.
        #      The "qtotal" is then the charge of the "artificial"/"dummy" leg.
        
        #  def detect_qtotal(flat_array, legcharges):
        #      inds_max = np.unravel_index(np.argmax(np.abs(flat_array)), flat_array.shape)
        #      val_max = abs(flat_array[inds_max])

        #      test_array = zeros(legcharges)  # Array prototype with correct charges
        #      qindices = [leg.get_qindex(i)[0] for leg, i in zip(legcharges, inds_max)]
        #      q = np.sum([l.get_charge(qi) for l, qi in zip(self.legs, qindices)], axis=0)
        #      return make_valid(q)  # TODO: leg.get_qindex, leg.get_charge

    def get_element(self, a: Tensor, idcs: list[int]) -> complex | float | bool:
        pos = np.array([l.parse_index(idx) for l, idx in zip(a.legs, idcs)])
        block = a.data.get_block(pos[:, 0])
        if block is None:
            return a.dtype.zero_scalar
        return self.get_block_element(block, pos[:, 1])

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        block_idx, idx_within = a.legs[0].parse_index(idx)
        block = a.data.get_block(np.array([block_idx]))
        if block is None:
            return a.dtype.zero_scalar
        return self.get_block_element(block, [idx_within])
            
    def set_element(self, a: Tensor, idcs: list[int], value: complex | float) -> Data:
        pos = np.array([l.parse_index(idx) for l, idx in zip(a.legs, idcs)])
        n = a.data.get_block_num(pos[:, 0])
        if n is None:
            shape = [leg.multiplicities[sector_idx] for leg, sector_idx in zip(a.legs, pos[:, 0])]
            block = self.zero_block(shape, dtype=a.dtype)
        else:
            block = a.data.blocks[n]
        blocks = a.data.blocks[:]
        blocks[n] = self.set_block_element(block, pos[:, 1], value)
        return AbelianBackendData(dtype=a.data.dtype, blocks=blocks, block_inds=a.data.blocks_inds,
                                  is_sorted=True)

    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        block_idx, idx_within = a.legs[0].parse_index(idx)
        n = a.data.get_block_num(np.array([block_idx]))
        if n is None:
            block = self.zero_block(shape=[a.legs[0].multiplicities[block_idx]], dtype=a.dtype)
        else:
            block = a.data.blocks[n]
        blocks = a.data.blocks[:]
        blocks[n] = self.set_block_element(block, [idx_within], value)
        return AbelianBackendData(dtype=a.data.dtype, blocks=blocks, block_inds=a.data.blocks_inds,
                                  is_sorted=True)

    def diagonal_data_from_full_tensor(self, a: Tensor, check_offdiagonal: bool) -> DiagonalData:
        blocks = [self.block_get_diagonal(block, check_offdiagonal) for block in a.data.blocks]
        return AbelianBackendData(a.dtype, blocks, a.data.block_inds, is_sorted=True)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_from_diagonal(block) for block in a.data.blocks]
        return AbelianBackendData(a.dtype, blocks, a.data.block_inds, is_sorted=True)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        blocks = [self.block_from_mask(block, dtype) for block in a.data.blocks]
        return AbelianBackendData(dtype, blocks, a.data.block_inds, is_sorted=True)

    def scale_axis(self, a: Tensor, b: DiagonalTensor, leg: int) -> Data:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks

        a_block_inds = a.data.block_inds
        a_block_inds_cont = a_block_inds[:, leg:leg+1]
        if leg == a.num_legs - 1:
            # due to lexsort(a_block_inds.T), a_block_inds_cont is sorted in this case
            pass
        else:
            sort = np.lexsort(a_block_inds_cont.T)
            a_blocks = [a_blocks[i] for i in sort]
            a_block_inds = a_block_inds[sort, :]
            a_block_inds_cont = a_block_inds_cont[sort, :]
        b_block_inds = b.data.block_inds
        
        # ensure common dtypes
        common_dtype = a.dtype.common(b.dtype)
        if a.data.dtype != common_dtype:
            a_blocks = [self.block_to_dtype(block, common_dtype) for block in a_blocks]
        if b.data.dtype != common_dtype:
            b_blocks = [self.block_to_dtype(block, common_dtype) for block in b_blocks]
        
        res_blocks = []
        res_block_inds = []
        # can assume that a.legs[leg] and b.legs[0] have same _sectors.
        # only need to iterate over common blocks, the non-common multiply to 0.
        # note: unlike the tdot implementation, we do not combine and reshape here.
        #       this is because we know the result will have the same block-structure as `a`, and
        #       we only need to scale the blocks on one axis, not perform a general tensordot.
        #       but this also means that we may encounter duplicates in a_block_inds_cont,
        #       i.e. multiple blocks of `a` which have the same sector on the leg to be scaled.
        #       -> use _iter_common_nonstrict_sorted_arrays instead of _iter_common_sorted_arrays
        for i, j in _iter_common_nonstrict_sorted_arrays(a_block_inds_cont, b_block_inds[:, :1]):
            res_blocks.append(self.block_scale_axis(a_blocks[i], b_blocks[j], axis=leg))
            res_block_inds.append(a_block_inds[i])
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, a.num_legs), int)
        
        return AbelianBackendData(common_dtype, res_blocks, res_block_inds, is_sorted=True)
    
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        a_blocks = a.data.blocks
        if maps_zero_to_zero:
            blocks = [func(block, **func_kwargs) for block in a.data.blocks]
            block_inds = a.data.block_inds
        else:
            a_block_inds = a.data.block_inds
            block_inds = np.repeat(np.arange(a.legs[0].num_sectors)[:, None], 2, axis=1)
            blocks = []
            for i, j in _iter_common_noncommon_sorted_arrays(block_inds, a_block_inds):
                if j is None:
                    # use that block_inds is just arange -> block_inds[i, 0] == i
                    block = self.zero_block([a.legs[0].multiplicities[i]], dtype=a.dtype)
                else:
                    block = a_blocks[j]
                blocks.append(func(block, **func_kwargs))
        if len(blocks) == 0:
            dtype = self.block_dtype(func(self.zero_block([1], dtype=a.dtype)))
        else:
            dtype = self.block_dtype(blocks[0])
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        a_mults = a.legs[0].multiplicities
        b_mults = b.legs[0].multiplicities
        
        blocks = []
        block_inds = []
        if partial_zero_is_zero:
            for i, j in _iter_common_sorted_arrays(a_block_inds, b_block_inds):
                blocks.append(func(a_blocks[i], b_blocks[j], **func_kwargs))
                block_inds.append(a_block_inds[i])
        else:
            for i, j in _iter_common_noncommon_sorted_arrays(a_block_inds, b_block_inds):
                if i is None:
                    a_block = self.zero_block([b_mults[b_block_inds[j, 0]]], dtype=a.dtype)
                    b_block = b_blocks[j]
                    block_inds.append(b_block_inds[j])
                elif j is None:
                    a_block = a_blocks[i]
                    b_block = self.zero_block([a_mults[a_block_inds[i, 0]]], dtype=b.dtype)
                    block_inds.append(a_block_inds[i])
                else:
                    a_block = a_blocks[i]
                    b_block = b_blocks[j]
                    block_inds.append(a_block_inds[i])
                blocks.append(func(a_block, b_block, **func_kwargs))
        block_inds = np.array(block_inds)

        if len(blocks) == 0:
            block = func(self.ones_block([1], dtype=a.dtype), self.ones_block([1], dtype=b.dtype))
            dtype = self.block_dtype(block)
        else:
            dtype = self.block_dtype(blocks[0])
            
        return AbelianBackendData(dtype=dtype, blocks=blocks, block_inds=block_inds, is_sorted=True)

    def apply_mask_to_Tensor(self, tensor: Tensor, mask: Mask, leg_idx: int) -> Data:
        # implementation similar to scale_axis, see notes there
        tensor_blocks = tensor.data.blocks
        mask_blocks = mask.data.blocks
        tensor_block_inds = tensor.data.block_inds
        tensor_block_inds_cont = tensor_block_inds[:, leg_idx:leg_idx + 1]
        if leg_idx != tensor.num_legs - 1:
            # due to np.lexsort(tensor_block_inds.T), only tensor_block_inds[:, -1] is sorted
            sort = np.lexsort(tensor_block_inds_cont.T)
            tensor_blocks = [tensor_blocks[i] for i in sort]
            tensor_block_inds = tensor_block_inds[sort]
            tensor_block_inds_cont = tensor_block_inds_cont[sort]
        mask_block_inds = mask.data.block_inds
        mask_block_inds_cont = mask_block_inds[:, 0:1]
        sort = np.lexsort(mask_block_inds_cont.T)
        mask_blocks = [mask_blocks[i] for i in sort]
        mask_block_inds = mask_block_inds[sort]
        mask_block_inds_cont = mask_block_inds_cont[sort]

        res_blocks = []
        res_block_inds = []
        # need only common blocks : zeros masks to zero, and a missing mask block means all False
        for i, j in _iter_common_nonstrict_sorted_arrays(tensor_block_inds_cont, mask_block_inds_cont):
            res_blocks.append(self.apply_mask_to_block(block=tensor_blocks[i], mask=mask_blocks[j], ax=leg_idx))
            block_inds = tensor_block_inds[i].copy()
            # tensor_block_inds[i] refer to mask.legs[0]._sectors
            # need to adjust to refer to mask.legs[1]._sectors
            block_inds[leg_idx] = mask_block_inds[j, 1]
            res_block_inds.append(block_inds)
        if len(res_block_inds) > 0:
            res_block_inds = np.array(res_block_inds)
        else:
            res_block_inds = np.zeros((0, tensor.num_legs), int)
        # OPTIMIZE (JU) block_inds might actually be sorted but i am not sure right now
        return AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=False)

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        tensor_blocks = tensor.data.blocks
        mask_blocks = mask.data.blocks
        tensor_block_inds_cont = tensor.data.block_inds[:, :1]  # since tensor is Diagonal, this is sorted
        mask_block_inds = mask.data.block_inds
        mask_block_inds_cont = mask_block_inds[:, :1]
        sort = np.lexsort(mask_block_inds_cont.T)
        mask_blocks = [mask_blocks[i] for i in sort]
        mask_block_inds = mask_block_inds[sort]
        mask_block_inds_cont = mask_block_inds_cont[sort]
        
        res_blocks = []
        res_block_inds = []  # gather only the entries of the first column in this list, repeat later
        for i, j in _iter_common_sorted_arrays(tensor_block_inds_cont, mask_block_inds_cont):
            res_blocks.append(self.apply_mask_to_block(block=tensor_blocks[i], mask=mask_blocks[j], ax=0))
            res_block_inds.append(mask_block_inds[j, 1])
        if len(res_block_inds) > 0:
            res_block_inds = np.repeat(np.array(res_block_inds)[:, None], 2, axis=1)
        else:
            res_block_inds = np.zeros((0, 2), int)
        return AbelianBackendData(tensor.dtype, res_blocks, res_block_inds, is_sorted=True)

    def eigh(self, a: Tensor, sort: str = None) -> tuple[DiagonalData, Data]:
        # for missing blocks, i.e. a zero block, the eigenvalues are zero, so we can just skip adding
        # that block to the eigenvalues.
        # for the eigenvectors, we choose the computational basis vectors, i.e. the matrix
        # representation within that block is the identity matrix.
        # we initialize all blocks to eye and override those where a has blocks.
        eigvects_data = self.eye_data(legs=a.legs[0:1], dtype=a.dtype)
        eigvals_blocks = []
        for block, bi in zip(a.data.blocks, a.data.block_inds):
            vals, vects = self.block_eigh(block, sort=sort)
            eigvals_blocks.append(vals)
            assert bi[0] == bi[1]  # TODO remove this check
            eigvects_data.blocks[bi[0]] = vects
        eigvals_data = AbelianBackendData(dtype=a.dtype.to_real, blocks=eigvals_blocks,
                                          block_inds=a.data.block_inds, is_sorted=True)
        return eigvals_data, eigvects_data
    
    def from_flat_block_trivial_sector(self, block: Block, leg: VectorSpace) -> Data:
        # need to consider basis_perm. see comment in to_flat_block_trivial_sector.
        # here we need the inverse though
        bi = leg._non_dual_sectors_where(leg.symmetry.trivial_sector)
        if leg.basis_perm is not None:
            perm = np.argsort(leg.basis_perm[slice(*leg.slices[bi])])
            block = self.apply_leg_permutations(block, [inverse_permutation(perm)])
        return AbelianBackendData(
            dtype=self.block_dtype(block), blocks=[block],
            block_inds=np.array([[bi]]),
            is_sorted=True
        )

    def to_flat_block_trivial_sector(self, tensor: Tensor) -> Block:
        num_blocks = len(tensor.data.blocks)
        if num_blocks == 1:
            res = tensor.data.blocks[0]
            if tensor.legs[0].basis_perm is not None:
                # we need to find the permutation perm such that res[perm] == dense_data[some_mask]
                # so far we have
                # res == internal_data[slice] == dense_data[basis_perm][slice] == dense_data[basis_perm[slice]]
                # thus dense_data[some_mask] == res[perm] == dense_data[basis_perm[slice][perm]]
                # i.e. perm needs to sort basis_perm[slice]
                bi = tensor.legs[0]._non_dual_sectors_where(tensor.legs[0].symmetry.trivial_sector)
                perm = np.argsort(tensor.legs[0].basis_perm[slice(*tensor.legs[0].slices[bi])])
                res = self.apply_leg_permutations(res, [perm])
            return res
        elif num_blocks == 0:
            dim = tensor.legs[0]._non_dual_sector_multiplicity(tensor.symmetry.trivial_sector)
            # no need to consider basis_perm, since its all 0 anyway
            return self.zero_block(shape=[dim], dtype=tensor.data.dtype)
        raise ValueError  # this should not happen for single-leg tensors

    def inv_part_from_flat_block_single_sector(self, block: Block, leg: VectorSpace, dummy_leg: VectorSpace) -> Data:
        assert dummy_leg.num_sectors == 1
        bi = leg.sectors_where(leg.symmetry.dual_sector(dummy_leg.sectors[0]))
        assert bi is not None
        assert self.block_shape(block) == (leg.multiplicities[bi],)
        if leg.basis_perm is not None:
            # see comment in to_flat_block_trivial_sector. here we need the inverse of that.
            perm = np.argsort(leg.basis_perm[slice(*leg.slices[bi])])
            block = self.apply_leg_permutations(block, [inverse_permutation(perm)])
        return AbelianBackendData(
            dtype=self.block_dtype(block),
            blocks=[self.block_add_axis(block, pos=1)],
            block_inds=np.array([[bi, 0]])
        )

    def inv_part_to_flat_block_single_sector(self, tensor: Tensor) -> Block:
        num_blocks = len(tensor.data.blocks)
        assert tensor.legs[1].num_sectors == 1
        # find the block-index that the single allowed block has (or would have)
        dummy_leg_sector = tensor.legs[1].sectors[0]
        bi = tensor.legs[0].sectors_where(tensor.symmetry.dual_sector(dummy_leg_sector))
        if num_blocks == 1:
            res = tensor.data.blocks[0][:, 0]
            if tensor.legs[0].basis_perm is not None:
                # see comment in to_flat_block_trivial_sector
                perm = np.argsort(tensor.legs[0].basis_perm[slice(*tensor.legs[0].slices[bi])])
                res = self.apply_leg_permutations(res, [perm])
            return res
        elif num_blocks == 0:
            dim = tensor.legs[0].multiplicities[bi]
            # no need to consider basis_perm, since its all 0 anyway
            return self.zero_block(shape=[dim], dtype=tensor.data.dtype)
        raise ValueError  # should have been caught by input checks in ChargedTensor.to_flat_block_single_sector

    def flip_leg_duality(self, tensor: Tensor, which_legs: list[int],
                         flipped_legs: list[VectorSpace], perms: list[np.ndarray]) -> Data:
        block_inds = np.copy(tensor.data.block_inds)
        for i, perm in zip(which_legs, perms):
            # old_sector_idx = perm[new_sector_idx]
            block_inds[:, i] = inverse_permutation(perm)[block_inds[:, i]]
        return AbelianBackendData(dtype=tensor.data.dtype, blocks=tensor.data.blocks,
                                  block_inds=block_inds, is_sorted=False)


def product_space_map_incoming_block_inds(space: ProductSpace, incoming_block_inds):
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
        ``self._block_ind_map[J, 2:-1] == block_inds[j]``.
    """
    # FIXME move this back to ProductSpace?
    assert incoming_block_inds.shape[1] == len(space.spaces)
    # calculate indices of _block_ind_map by using the appropriate strides
    inds_before_perm = np.sum(incoming_block_inds * space._strides[np.newaxis, :], axis=1)
    # now permute them to indices in _block_ind_map
    return space._fusion_outcomes_inverse_sort[inds_before_perm]
