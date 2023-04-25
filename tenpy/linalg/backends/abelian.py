"""Backends for abelian group symmetries.

Changes compared to old np_conserved:

- replace `ChargeInfo` by subclasses of `AbelianGroup` (or `ProductSymmetry`)
- replace `LegCharge` by `AbelianBackendVectorSpace` and `LegPipe` by `AbelianBackendProductSpace`
- standard `Tensor` have qtotal=0, only ChargedTensor can have non-zero qtotal
- relabeling:
    - `Array.qdata`, "qind" and "qindices" to `AbelianBackendData.block_inds` and "block indices"
    - `LegPipe.qmap` to `AbelianBackendProductSpace.block_ind_map` (witch changed column order!!!)
    - `LegPipe._perm` to `ProductSpace._perm_block_inds_map`
    - `LetCharge.get_block_sizes()` is just `VectorSpace.multiplicities`
- keep VectorSpace and ProductSpace "sorted" and "bunched",
  i.e. do not support legs with smaller blocks to effectively allow block-sparse tensors with
  smaller blocks than dictated by symmetries (which we actually have in H_MPO on the virtual legs...)
  In turn, VectorSpace saves a `_perm` used to sort the originally passed `sectors`.
- keep `Tensor.block_ins` sorted (i.e. no arbitrary gauge permutation in block indices)

"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, TypeVar, List, TYPE_CHECKING
import numpy as np
import copy
import warnings

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, Block, Dtype
from ..symmetries.groups import FusionStyle, BraidingStyle, Symmetry, Sector
from numpy import ndarray
from ..symmetries.spaces import VectorSpace, ProductSpace
from ...tools.misc import inverse_permutation
from ...tools.optimization import use_cython

__all__ = ['AbelianBackendData', 'AbelianBackendVectorSpace', 'AbelianBackendProductSpace',
           'AbstractAbelianBackend', 'detect_qtotal']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor, ChargedTensor


class AbelianBackendVectorSpace(VectorSpace):
    """Subclass of VectorSpace with additonal data and restrictions for AbstractAbelianBackend.


    Attributes
    ----------
    perm_block_inds : ndarray[int]
        Permutation from the original order of sectors to the sorted one in :attr:`sectors`.
    slices : ndarray[(int, int)]
        For each sector the begin and end when projecting to/from a "flat" ndarray
        without symmetries. Note that this is not sorted when perm_block_inds is non-trivial.

    """
    def __init__(self, symmetry: Symmetry, sectors: SectorArray, multiplicities: ndarray = None,
                 is_real: bool = False, _is_dual: bool = False,
                 perm_block_inds=None, slices=None):
        VectorSpace.__init__(self, symmetry, sectors, multiplicities, is_real, _is_dual)
        num_sectors = sectors.shape[0]
        self.sector_ndim = sectors.ndim
        if perm_block_inds is None:
            # sort by slices
            assert slices is None
            self.slices = _slices_from_multiplicities(self.multiplicities)
            self._sort_sectors()
        else:
            # TODO: do we need this case?
            assert slices is not None
            self.perm_block_inds = perm_block_inds
            self.slices = slices

    def _sort_sectors(self):
            # sort sectors
            assert not hasattr(self, 'perm_block_inds')
            perm_block_inds = np.lexsort(self._sectors.T)
            self.perm_block_inds = perm_block_inds
            self._sectors = self._sectors[perm_block_inds]
            self.multiplicities = self.multiplicities[perm_block_inds]
            self.slices = self.slices[perm_block_inds]

    # TODO: do we need get_qindex?
    # Since slices is no longer sorted, it would be O(L) rather than O(log(L))

    def project(self, mask: ndarray):
        """Return copy keeping only the indices specified by `mask`.

        Parameters
        ----------
        mask : 1D array(bool)
            Whether to keep each of the indices in the dense array.

        Returns
        -------
        map_qind : 1D array
            Map of qindices, such that ``qind_new = map_qind[qind_old]``,
            and ``map_qind[qind_old] = -1`` for qindices projected out.
        block_masks : 1D array
            The bool mask for each of the *remaining* blocks.
        projected_copy : :class:`LegCharge`
            Copy of self with the qind projected by `mask`.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        cp = copy.copy()
        block_masks = [mask[b:e] for b, e in self.slices]
        new_multiplicities = np.array([np.sum(bm) for bm in block_masks])
        keep = np.nonzero(new_multiplicities)[0]
        block_masks = [block_masks[i] for i in keep]
        new_block_number = len(block_masks)
        cp._sectors = cp._sectors[keep]
        cp.multiplicities = new_multiplicities[keep]
        cp.slices = _slices_from_multiplicities(cp.multiplicities)
        map_qind = np.full((new_block_number,), -1, np.intp)
        map_qind[keep] = cp.perm_block_inds = np.arange(new_block_number)
        return map_qind, block_masks, cp

    def __mul__(self, other):
        if isinstance(other, AbelianBackendVectorSpace):
            return AbelianBackendProductSpace([self, other])
        return NotImplemented


def _slices_from_multiplicities(multiplicities: ndarray):
    slices = np.zeros((len(multiplicities), 2), np.intp)
    slices[:, 1] = slice_ends = np.cumsum(multiplicities)
    slices[1:, 0] = slice_ends[:-1]
    return slices


# TODO: is the diamond-structure inheritance okay?
class AbelianBackendProductSpace(ProductSpace, AbelianBackendVectorSpace):
    r"""

    Attributes
    ----------
    block_ind_map :
        See below. (In old np_conserved: `qmap`)


    Notes
    -----
    For ``np.reshape``, taking, for example,  :math:`i,j,... \rightarrow k` amounted to
    :math:`k = s_1*i + s_2*j + ...` for appropriate strides :math:`s_1,s_2`.

    In the charged case, however, we want to block :math:`k` by charge, so we must
    implicitly permute as well.  This reordering is encoded in `block_ind_map` as follows.

    Each block index combination :math:`(i_1, ..., i_{nlegs})` of the `nlegs=len(spaces)`
    input VectorSpaces will end up getting placed in some slice :math:`a_j:a_{j+1}` of the
    resulting `ProductSpace`. Within this slice, the data is simply reshaped in usual row-major
    fashion ('C'-order), i.e., with strides :math:`s_1 > s_2 > ...` given by the block size.

    It will be a subslice of a new total block in the ProductSpace labeled by block index
    :mah:`J`. We fuse charges according to the rule::

        ProductSpace.sectors[J] = fusion_outcomes(*[l.sectors[i_l]
            for l, i_l,l in zip(incoming_block_inds, spaces)])

    Since many charge combinations can fuse to the same total charge,
    in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
    charge block :math:`J` in the `ProductSpace`.

    The rows of `block_ind_map` are precisely the collections of
    ``[b_{J,k}, b_{J,k+1}, i_1, . . . , i_{nlegs}, J]``.
    Here, :math:`b_k:b_{k+1}` denotes the slice of this block index combination *within*
    the total block `J`, i.e., ``b_{J,k} = a_j - self.slices[J]``.

    The rows of `block_ind_map` are lex-sorted first by ``J``, then the ``i``.
    Each ``J`` will have multiple rows,
    and the order in which they are stored in `block_inds` is the order the data is stored
    in the actual tensor, i.e., it might look like ::

        [ ...,
         [ b_{J,k},   b_{J,k+1},  i_1,    ..., i_{nlegs}   , J,   ],
         [ b_{J,k+1}, b_{J,k+2},  i'_1,   ..., i'_{nlegs}  , J,   ],
         [ 0,         b_{J,1},    i''_1,  ..., i''_{nlegs} , J + 1],
         [ b_{J,1},   b_{J,2},    i'''_1, ..., i'''_{nlegs}, J + 1],
         ...]

    """
    # formerly known as LegPipe

    # TODO: AbelienBackendVectorSpace doesn't have access to backend,
    # so can't call convert_vector_space().
    #  def __init__(self, spaces: list[VectorSpace], _is_dual: bool = False):
    #      backend = spaces[0].backend
    #      spaces = [backend.convert_vector_space(s) for s in spaces]
    #      ProductSpace.__init__(self, spaces, is_dual)

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool,
                     ) -> tuple[SectorArray, ndarray]:
        # this function heavily uses numpys advanced indexing, for details see
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        num_spaces = len(spaces)

        spaces_num_sectors = tuple(space.num_sectors for space in spaces)
        self._strides = _make_stride(spaces_num_sectors, cstyle=False)
        # (save strides for :meth:`_map_incoming_block_inds`)

        # create a grid to select the multi-index sector
        grid = np.indices(spaces_num_sectors, np.intp)
        # grid is an array with shape ``(num_spaces, *spaces_num_sectors)``,
        # with grid[li, ...] = {np.arange(space_block_numbers[li]) increasing in li-th direcion}
        # collapse the different directions into one.
        grid = grid.T.reshape(-1, num_spaces)  # *this* is the actual `reshaping`
        # *rows* of grid are now all possible cominations of qindices.
        # transpose before reshape ensures that grid.T is np.lexsort()-ed

        nblocks = grid.shape[0]  # number of blocks in ProductSpace = np.product(spaces_num_sectors)
        # this is different from num_sectors

        # determine block_ind_map -- it's essentially the grid.
        block_ind_map = np.zeros((nblocks, 3 + num_spaces), dtype=np.intp)
        block_ind_map[:, 2:-1] = grid  # possible combinations of indices

        # the block size for given (i1, i2, ...) is the product of ``multiplicities[il]``
        # andvanced indexing:
        # ``grid.T[li]`` is a 1D array containing the qindex `q_li` of leg ``li`` for all blocks
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(spaces, grid.T)],
                                 axis=0)
        # block_ind_map[:, :2] and [:, -1] is initialized after sort/bunch.

        # calculate new non-dual sectors
        if _is_dual:
            # overall fusion of sectors is equivalent to taking dual of each sector
            # in standard use cases, this can often avoid explicit
            # symmetry.dual_sector() calls in VectorSpace.sectors()
            fuse_sectors = [s.dual.sectors for s in spaces]
        else:
            fuse_sectors = [s.sectors for s in spaces]
        # _sectors are the ones saved in self._sectors, not the property self.sectors
        _sectors = _fuse_abelian_charges(symmetry,
            *(sectors[gr] for sectors, gr in zip(fuse_sectors, grid.T)))

        # sort (non-dual) charge sectors. Similar code as in :meth:`LegCharge.sort`
        perm_block_inds = np.lexsort(_sectors.T)
        block_ind_map = block_ind_map[perm_block_inds]
        _sectors = _sectors[perm_block_inds]
        multiplicities = multiplicities[perm_block_inds]
        # inverse permutation is needed in _map_incoming_block_inds
        self._inv_perm_block_inds = inverse_permutation(perm_block_inds)

        slices = np.cumsum(multiplicities)
        block_ind_map[1:, 0] = slices[:-1]  # start with 0
        block_ind_map[:, 1] = slices

        # bunch sectors with equal charges together
        diffs = _find_row_differences(_sectors)
        _sectors = _sectors[diffs]
        multiplicities = slices[diffs]
        multiplicities[1:] -= multiplicities[:-1]

        new_block_ind = np.zeros(len(block_ind_map), dtype=np.intp) # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        block_ind_map[:, :2] -= multiplicities[new_block_ind][:, np.newaxis]

        self.block_ind_map = block_ind_map  # finished
        # self.q_map_slices = diffs  # reminder: differs from old npc by missing last index.
        # TODO: do we need this? I think it was used in split_legs()...

        return _sectors, multiplicities

    def _map_incoming_block_inds(self, incoming_block_inds):
        """Map incoming qindices to indices of :attr:`block_ind_map`.

        Needed for `combine_legs`.

        Parameters
        ----------
        incoming_block_inds : 2D array
            Rows are block indices :math:`(i_1, i_2, ... i_{nlegs})` for incoming legs.

        Returns
        -------
        block_inds: 1D array
            For each row of `incoming_block_inds` an index `J` such that
            ``self.block_ind_map[J, 2:-1] == block_inds[j]``.
        """
        assert incoming_block_inds.shape[1] == len(self.spaces)
        # calculate indices of block_ind_map[_inv_perm_block_inds],
        # which is sorted by :math:`i_1, i_2, ...`,
        # by using the appropriate strides
        inds_before_perm = np.sum(incoming_block_inds * self._strides[np.newaxis, :], axis=1)
        # now permute them to indices in block_ind_map
        return self._inv_perm_block_inds[inds_before_perm]

    def as_VectorSpace(self):
        return AbelianBackendVectorSpace(symmetry=self.symmetry,
                                         sectors=self._sectors,
                                         multiplicities=self.multiplicities,
                                         is_real=self.is_real,
                                         _is_dual=self.is_dual)

    def project(self, *args, **kwargs):
        """Convert self to VectorSpace and call :meth:`AbelianBackendVectorSpace.project`.

        In general, this could be implemented for a ProductSpace, but would make
        `split_legs` more complicated, thus we keep it simple.
        If you really want to project and split afterwards, use the following work-around,
        which is for example used in :class:`~tenpy.algorithms.exact_diagonalization`:

        1) Create the full pipe and save it separetely.
        2) Convert the Pipe to a Leg & project the array with it.
        3) [... do calculations ...]
        4) To split the 'projected pipe' of `A`, create an empty array `B` with the legs of A,
           but replace the projected leg by the full pipe. Set `A` as a slice of `B`.
           Finally split the pipe.
        """
        # TODO: this should be ProductSpace.project()
        # is method resolution order correct to choose that over AbelianBackendVectorSpace.project()?
        warnings.warn("Converting ProductSpace to VectorSpace for `project`", stacklevel=2)
        res = self.as_VectorSpace()
        return res.project(*args, **kwargs)


def _make_stride(shape, cstyle=True):
    """Create the strides for C- (or F-style) arrays with a given shape.

    Equivalent to ``x = np.zeros(shape); return np.array(x.strides, np.intp) // x.itemsize``.
    """
    L = len(shape)
    stride = 1
    res = np.empty([L], np.intp)
    if cstyle:
        res[L - 1] = 1
        for a in range(L - 1, 0, -1):
            stride *= shape[a]
            res[a - 1] = stride
    else:
        res[0] = 1
        for a in range(0, L - 1):
            stride *= shape[a]
            res[a + 1] = stride
    return res

def _find_row_differences(sectors: SectorArray):
    """Return indices where the rows of the 2D array `qflat` change.

    Parameters
    ----------
    sectors : 2D array
        The rows of this array are compared.

    Returns
    -------
    diffs: 1D array
        The indices where rows change, including the first and last. Equivalent to:
        ``[0] + [i for i in range(1, len(qflat)) if np.any(qflat[i-1] != qflat[i])]``
    """
    # note: remove last entry [len(sectors)] compared to old.charges
    diff = np.ones(sectors.shape[0], dtype=np.bool_)
    diff[1:] = np.any(sectors[1:] != sectors[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values


def _fuse_abelian_charges(symmetry: AbelianSymmetry, *sector_arrays: SectorArray) -> SectorArray:
    assert symmetry.fusion_style == FusionStyle.single
    fusion = sector_arrays[0]
    for sectors in sector_arrays[1:]:
        fusion = symmetry.fusion_outcomes_broadcast(fusion, sectors)
        # == fusion + space.sector, but mod N for ZN
    return np.asarray(fusion)

def _valid_block_indices(spaces: list[AbelianBackendVectorSpace]):
    """Find block_inds where the charges of the `spaces` fuse to `symmetry.trivial_sector`"""
    symmetry = spaces[0].symmetry
    # TODO: this is brute-force going through all possible combinations of block indices
    # spaces are sorted, so we can probably reduce that search space quite a bit...
    # similar to `grid` in ProductSpace._fuse_spaces()
    grid = np.indices((s.num_sectors for s in spaces), dtype=int)
    grid = grid.T.reshape((-1, len(spaces)))
    total_sectors = _fuse_abelian_charges(symmetry,
                                          (space.sector[gr] for space, gr in zip(spaces, grid.T)))
    valid = np.all(total_sectors == symmetry.trivial_sector[np.newaxis, :], axis=1)
    block_inds = grid[valid, :]
    perm = np.lexsort(block_inds.T)
    return block_inds[perm, :]


@dataclass
class AbelianBackendData:
    """Data stored in a Tensor for :class:`AbstractAbelianBackend`."""
    dtype : Dtype
    blocks : List[Block]  # The actual entries of the tensor. Formerly known as Array._data
    block_inds : ndarray  # For each of the blocks entries the block indices specifying to which
    # sector of the different legs it belongs

    def copy(self, deep=True):
        if deep:
            return AbelianBackendData(self.dtype,
                                      [self.block_copy(b) for b in self.blocks],
                                      self.block_inds.copy())
        return AbelianBackendData(self.dtype, self.blocks, self.block_inds)

    def _sort_block_inds(self):
        """Bring `block_inds` (back) into the conventional sorted order.

        To speed up functions as tensordot, we always keep the blocks in a well-defined order
        where ``np.lexsort(block_inds.T)`` is trivial."""
        perm = np.lexsort(self.block_inds.T)
        self.block_inds = self.block_inds[perm, :]
        self.blocks = [self.blocks[p] for p in perm]


class AbstractAbelianBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """Backend for Abelian group symmetries.

    """
    def convert_vector_space(self, leg: VectorSpace) -> AbelianBackendVectorSpace:
        if isinstance(leg, (AbelianBackendVectorSpace, AbelianBackendProductSpace)):
            return leg
        elif isinstance(leg, ProductSpace):
            return AbelianBackendProductSpace(leg.spaces, leg.is_dual)
        else:
            return AbelianBackendVectorSpace(leg.symmetry, leg.sectors, leg.multiplicities,
                                             leg.is_real, leg.is_dual)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return a.dtype

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        data = a.data.copy()  # TODO: should this make a copy?
        data.blocks = [self.block_to_dtype(block, dtype) for block in data.blocks]
        data.dtype = dtype
        return data

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry.is_abelian and symmetry.braiding_style == BraidingStyle.bosonic

    def is_real(self, a: Tensor) -> bool:
        return a.dtype.is_real

    def data_item(self, a: Data) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            assert all(leg.dim == 1 for leg in a.legs)
            return 0.
        return self.block_item(a.blocks[0])

    def to_dense_block(self, a: Tensor) -> Block:
        res = self.zero_block([leg.dim for leg in a.legs])
        for block, block_inds in zip(a.data.blocks, a.data.block_inds):
            slices = [slice(*leg.slices[qi]) for i, leg in zip(block_inds, a.legs)]
            res[tuple(slices)] = block
        return res

    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5) -> AbelianBackendData:
        legs = [self.convert_vector_space(leg) for leg in legs]
        # TODO (JH) should we convert legs in tensors.py or here per backend?
        # -> same for other functions like zeros() etc
        dtype = self.block_dtype(a)
        block_inds = _valid_block_indices(legs)
        blocks = []
        for b_i in block_inds:
            slices = [slice(*leg.slices[i]) for i, leg in zip(b_i, a.legs)]
            blocks.append(a[tuple(slices)])
        return AbelianBackendData(dtype, blocks, block_inds)

    def from_block_func(self, func, legs: list[VectorSpace]) -> AbelianBackendData:
        dtype = self.block_dtype(a)
        block_inds = _valid_block_indices(legs)
        blocks = []
        for b_i in block_inds:
            shape = [leg.multiplicities[i] for i, leg in zip(b_i, a.legs)]
            blocks.append(func(tuple(shape)))
        return AbelianBackendData(dtype, blocks, block_inds)

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> AbelianBackendData:
        block_inds = np.zeros((0, len(legs)), dtype=int)
        return AbelianBackendData(dtype, [], block_inds)

    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        block_inds = np.indices((leg.num_sectors for leg in legs)).T.reshape(-1, len(legs))
        # block_inds is by construction np.lexsort()-ed
        dims = [leg.multiplicities[bi] for leg, bi in zip(legs, block_inds.T)]
        blocks = [self.eye_block(shape, dtype) for shape in zip(dims)]
        return AbelianBackendData(dtype, blocks, np.hstack([block_inds, block_inds]))

    def copy_data(self, a: Tensor) -> Data:
        return a.data.copy(deep=True)

    def _data_repr_lines(self, data: Data, indent: str, max_width: int, max_lines: int):
        if len(data.blocks) == 0:
            return [f'{indent}* Data : no non-zero block']
        if max_lines <= 1:
            return [f'{indent}* Data : {len(data.blocks):d} blocks']
        # show largest blocks first until we hit max_lines
        sizes = np.prod([self.block_shape(block) for block in data.blocks], axis=1)
        all_lines = [None]
        shown_blocks = 0
        for i in np.argsort(sizes):
            bi = data.block_inds[i, :]
            # TODO : would make sense to show sectors rather than block_inds
            # but don't have access to legs
            # sector = [symmetry.sector_str(leg.sectors[i]) for i, leg in zip(bi, legs)]
            sector_line = f'{indent}  * block_inds {bi!s}'
            all_lines.append(sector_line)
            all_lines.extend(self._block_repr_lines(data.blocks[i],
                                                    indent=indent + '    ',
                                                    max_width=max_width,
                                                    max_lines=max_lines - len(all_lines)))
            shown_blocks += 1
            if len(all_lines) + 1 >= max_lines:
                break
        if shown_blocks == len(data.blocks):
            shown = f'all {shown_blocks:d}'
        else:
            shown = f'largest {shown_blocks:d} of {len(data.blocks):d}'
        all_lines[0] = f'{indent}* Data : {shown} blocks'
        return all_lines

    #  def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]) -> Data:
    #      return self.block_tdot(a.data, b.data, axs_a, axs_b)

    #  @abstractmethod
    #  def svd(self, a: Tensor, axs1: list[int], axs2: list[int], new_leg: VectorSpace | None
    #          ) -> tuple[Data, Data, Data, VectorSpace]:
    #      # reshaping, slicing etc is so specific to the BlockBackend that I dont bother unifying anything here.
    #      # that might change though...
    #      ...

    #  def outer(self, a: Tensor, b: Tensor) -> Data:
    #      return self.block_outer(a.data, b.data)

    #  def inner(self, a: Tensor, b: Tensor, axs2: list[int] | None) -> complex:
    #      return self.block_inner(a.data, b.data, axs2)

    def transpose(self, a: Tensor, permutation: list[int]) -> Data:
        blocks = a.data.blocks
        blocks = [self.block_transpose(block, permutation) for block in a.data.blocks]
        block_inds = a.data.block_inds[:, permutation]
        data = AbelianBackendData(a.data.dtype, blocks, block_inds)
        data._sort_block_inds()
        return data

    #  def trace(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
    #      return self.block_trace(a.data, idcs1, idcs2)

    def conj(self, a: Tensor) -> Data:
        blocks = [self.block_conj(b) for b in a.data.blocks]
        return AbelianBackendData(a.data.dtype, blocks, a.data.block_inds)

    #  def combine_legs(self, a: Tensor, idcs: list[int], new_leg: ProductSpace) -> Data:
    #      return self.block_combine_legs(a.data, idcs)

    #  def split_leg(self, a: Tensor, leg_idx: int) -> Data:
    #      return self.block_split_leg(a, leg_idx, dims=[s.dim for s in a.legs[leg_idx]])

    #  def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
    #      return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        n_legs = a.num_legs
        if len(a.data.blocks) == 0:
            return AbelianBackendData(a.data.dtype, [],
                                      np.zeros([0, n_legs - len(idcs)], dtype=int))
        blocks = [self.block_squeeze_legs(b, idcs) for b in self.blocks]
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
        return AbelianBackendData(a.data.dtype, blocks, block_inds)

    def norm(self, a: Tensor) -> float:
        # TODO: argument for different p-norms?
        block_norms = [self.block_norm(b) for b in a.data]
        return np.linalg.norm(block_norms)

    #  def exp(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
    #      matrix, aux = self.block_matrixify(a.data, idcs1, idcs2)
    #      return self.block_dematrixify(self.matrix_exp(matrix), aux)

    #  def log(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
    #      matrix, aux = self.block_matrixify(a.data, idcs1, idcs2)
    #      return self.block_dematrixify(self.matrix_log(matrix), aux)

    #  def random_normal(self, legs: list[VectorSpace], dtype: Dtype, sigma: float) -> Data:
    #      return self.block_random_normal([l.dim for l in legs], dtype=dtype, sigma=sigma)

    #  def add(self, a: Tensor, b: Tensor) -> Data:
    #      return self.block_add(a.data, b.data)

    #  def mul(self, a: float | complex, b: Tensor) -> Data:
    #      return self.block_mul(a, b.data)


# TODO FIXME how to handle ChargedTensor vs Tensor?
#  def detect_qtotal(flat_array, legcharges):
#      inds_max = np.unravel_index(np.argmax(np.abs(flat_array)), flat_array.shape)
#      val_max = abs(flat_array[inds_max])

#      test_array = zeros(legcharges)  # Array prototype with correct charges
#      qindices = [leg.get_qindex(i)[0] for leg, i in zip(legcharges, inds_max)]
#      q = np.sum([l.get_charge(qi) for l, qi in zip(self.legs, qindices)], axis=0)
#      return make_valid(q)  # TODO: leg.get_qindex, leg.get_charge
