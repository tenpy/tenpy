"""Backends for abelian group symmetries.

Changes compared to old np_conserved:

- replace `ChargeInfo` by subclasses of `AbelianGroup` (or `ProductSymmetry`)
- replace `LegCharge` by `AbelianBackendVectorSpace` and `LegPipe` by `AbelianBackendFusionSpace`
- standard `Tensor` have qtotal=0, only ChargedTensor can have non-zero qtotal
- relabeling:
    - `Array.qdata`, "qind" and "qindices" to `AbelianBackendData.block_inds` and "block indices"
    - `LegPipe.qmap` to `AbelianBackendFusionSpace.block_ind_map` (witch changed column order!!!)
    - `LegPipe._perm` to `FusionSpace._perm_block_inds_map`
    - `LetCharge.get_block_sizes()` is just `VectorSpace.multiplicities`
- keep VectorSpace and FusionSpace "sorted" and "bunched",
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

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, Block, Dtype
from ..symmetries import Symmetry, ProductSymmetry, AbelianGroup, VectorSpace, FusionSpace, Sector
from ...tools.optimization import use_cython

__all__ = ['AbelianBackendData', 'AbelianBackendVectorSpace', 'AbelianBackendFusionSpace',
           'AbstractAbelianBackend', 'detect_qtotal']

Charge = np.int_
#Sector = ndarray[Charge] with length=number of symmetries (>1 only for ProductSymmetry) internally
# properties return a single Charge or Tuple of Charges
Qindex = np.int_  # index of a given sector within an (AbelianBackend)VectorSpace


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor, ChargedTensor


class AbelianBackendVectorSpace(VectorSpace):
    """Subclass of VectorSpace with additonal data and restrictions.

    A `Sector` consists of a single integer of charge values (for U1Symmetry and ZNSymmetry)
    or a tuple of integers for ProductSpace.


    Attributes
    ----------
    perm_block_inds : ndarray[int]
        Permutation from the original order of sectors to the sorted one in :attr:`sectors`.
    slices : ndarray[(int, int)]
        For each sector the begin and end when projecting to/from a "flat" ndarray
        without symmetries. Note that this is not sorted when perm_block_inds is non-trivial.

    """
    def __init__(self, symmetry: Symmetry, sectors: list[Sector], multiplicities: list[int],
                 is_dual: bool, is_real: bool, perm_block_inds=None, slices=None):
        # possibly originally unsorted sectors
        sectors = np.asarray(sectors, dtype=Charge) # TODO: typecheck probably complains?
        multiplicities = np.asarray(multiplicities)
        VectorSpace.__init__(self, symmetry, sectors, multiplicities, is_dual, is_real)
        N_sectors = sectors.shape[0]
        self.sector_ndim = sectors.ndim
        if sectors.ndim == 1:
            assert not isinstance(symmetry, ProductSymmetry)
        if perm_block_inds is None:
            # sort by slices
            assert slices is None
            slices = np.zeros((N_sectors, 2), np.intp)
            slices[:, 1] = slice_ends = np.cumsum(self.multiplicities)
            slices[1:, 0] = slice_ends[:-1]
            self.slices = slices
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

    def project(self, mask):
        raise NotImplementedError("TODO")



# TODO: is the diamond-structure inheritance okay?
class AbelianBackendFusionSpace(FusionSpace, AbelianBackendVectorSpace):
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
    resulting `FusionSpace`. Within this slice, the data is simply reshaped in usual row-major
    fashion ('C'-order), i.e., with strides :math:`s_1 > s_2 > ...` given by the block size.

    It will be a subslice of a new total block in the FusionSpace labeled by block index
    :mah:`J`. We fuse charges according to the rule::

        FusionSpace.sectors[J] = fusion_outcomes(*[l.sectors[i_l]
            for l, i_l,l in zip(incoming_block_inds, spaces)])

    Since many charge combinations can fuse to the same total charge,
    in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
    charge block :math:`J` in the `FusionSpace`.

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
    def __init__(self, spaces: list[VectorSpace], is_dual: bool = False):
        backend = spaces[0].backend
        spaces = [backend.convert_vector_space(s) for s in spaces]
        FusionSpace.__init__(self, spaces, is_dual)

    def _fuse_spaces(self, spaces: list[AbelianBackendVectorSpace], is_dual: bool):
        # this function heavily uses numpys advanced indexing, for details see
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        N_spaces = len(spaces)

        spaces_N_sectors = tuple(space.N_sectors for space in spaces)
        self._strides = _make_stride(spaces_N_sectors, Cstyle=True)
        # (save strides for :meth:`_map_incoming_block_inds`)

        # create a grid to select the multi-index sector
        grid = np.indices(spaces_N_sectors, np.intp)
        # grid is an array with shape ``(N_spaces, *spaces_N_sectors)``,
        # with grid[li, ...] = {np.arange(space_block_numbers[li]) increasing in li-th direcion}
        # save the strides of grid, which is needed for :meth:`_map_incoming_block_inds`
        # collapse the different directions into one.
        grid = grid.reshape(N_spaces, -1)  # *this* is the actual `reshaping`
        # *columns* of grid are now all possible cominations of qindices.

        nblocks = grid.shape[1]  # number of blocks in FusionSpace = np.product(spaces_N_sectors)
        # this is different from N_sectors

        # determine block_ind_map -- it's essentially the grid.
        block_ind_map = np.zeros((nblocks, 3 + N_spaces), dtype=np.intp)
        block_ind_map[:, 2:-1] = grid.T  # transpose -> rows are possible combinations.

        # the block size for given (i1, i2, ...) is the product of ``multiplicities[il]``
        # andvanced indexing:
        # ``grid[li]`` is a 1D array containing the qindex `q_li` of leg ``li`` for all blocks
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(spaces, grid)], axis=0)
        # block_ind_map[:, :3] is initialized after sort/bunch.

        # calculate new non-dual sectors
        non_dual_sectors = _fuse_abelian_charges(spaces[0].symmetry,
                                                 *(space.sectors for space in spaces))
        if is_dual:
            non_dual_sectors = spaces[0].symmetry.dual_sectors(non_dual_sectors)

        # sort (non-dual) charge sectors. Similar code as in :meth:`LegCharge.sort`
        perm_block_inds = np.lexsort(charges.T)
        block_ind_map = block_ind_map[perm_block_inds]
        non_dual_sectors = non_dual_sectors[perm_block_inds]
        multiplicities = multiplicities[perm_block_inds]
        # inverse permutation is needed in _map_incoming_block_inds
        self._inv_perm_block_inds = inverse_permutation(perm_block_inds)


        slices = np.cumsum(multiplicities)
        block_ind_map[1:, 0] = slices[:-1]  # start with 0
        block_ind_map[:, 1] = slices

        # bunch sectors with equal charges together
        diffs = _find_row_differences(non_dual_sectors)
        non_dual_sectors = non_dual_sectors[diffs]
        multiplicities = slices[diffs]
        multiplicities[1:] -= multiplicities[:-1]

        new_block_ind = np.zeros(len(block_ind_map), dtype=np.intp) # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        block_ind_map[:, :2] -= multiplicities[new_block_ind][:, np.newaxis]

        self.block_ind_map = block_ind_map  # finished
        # self.q_map_slices = diffs  # up to last index.
        # TODO: do we need this? I think it was used in split_legs()...

        if is_dual:
            sectors = spaces[0].symmetry.dual_sectors(non_dual_sectors)
        else:
            sectors = non_dual_sectors
        return sectors, multiplicities

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
        res = AbelianBackendVectorSpace.__new__(AbelianBackendVectorSpace)
        return res


    #  def sectors(self):
    #      # In AbelianBackend, we save the _sectors in a 2D numpy array, no matter the nesting
    #      # convert to nested list of list for nested FusionSpace
    #      returns nested lists for nested FusionSpace, but we
    #      if self.is_dual:


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
    # NOTE: remove last entry [len(sectors)] compared to old.charges
    diff = np.ones(sectors.shape[0], dtype=np.bool_)
    diff[1:] = np.any(sectors[1:] != sectors[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values


def _fuse_abelian_charges(symmetry: AbelianSymmetry, *sector_arrays: ndarray[Sector]):
    from ..symmetries import FusionStyle
    assert symmetry.fusion_style == FusionStyle.single
    # sectors[i] can be 1d numpy arrays, or list of 1D arrays if using a ProductSymmetry
    fusion = sector_arrays[0]
    for sector in sector_arrays[1:]:
        fusion = symmetry.fusion_outcomes_broadcast(fusion, sector_array)
        # == fusion + space.sector, but mod N for ZN
    return np.asarray(fusion)




@dataclass
class AbelianBackendData:
    """Data stored in a Tensor for :class:`AbstractAbelianBackend`."""
    dtype : Dtype
    np_dtype : np.dtype
    blocks : List[Block]  # The actual entries of the tensor. Formerly known as Array._data
    block_inds : np.ndarray  # For each of the blocks entries the qindices of the different legs.

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
        if isinstance(leg, (AbelianBackendVectorSpace, AbelianBackendFusionSpace)):
            return leg
        elif isinstance(leg, FusionSpace):
            return AbelianBackendFusionSpace(leg.spaces, leg.is_dual)
        else:
            return AbelianBackendVectorSpace(leg.symmetry, leg.sectors,
                                             leg.multiplicities, leg.is_dual, leg.is_real)

    def get_dtype_from_data(self, a: Data) -> Dtype:
        return a.dtype

    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        data = a.data.copy()  # TODO: should this make a copy?
        data.blocks = [self.block_to_dtype(block, dtype) for block in data.blocks]
        data.dtype = dtype
        return data

    # TODO: unclear what this should do instance-specific? Shouldn't this be class-specific?
    #  JU: yes, this can be a classmethod
    # AbstractNoSymmetryBackend checks == no_symmetry, but we might not have a unique instance?
    #  JU: have implement Symmetry.__eq__ to to sensible things
    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        return symmetry.is_abelian

    def is_real(self, a: Tensor) -> bool:
        return a.data.dtype.is_real

    def data_item(self, a: Data) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            assert all(leg.dim == 1 for leg in a.legs)
            return 0
        return self.block_item(a.blocks[0])

    def to_dense_block(self, a: Tensor) -> Block:
        res = np.zeros([leg.dim for leg in a.legs])
        for block, qindices in zip(a.blocks, a.qdata):
            slices = []
            for (qi, leg) in zip(qindices, a.legs):
                sl = leg._abelian_data.slices
                slices.append(slice(sl[qi], sl[qi+1]))
            res[tuple(slices)] = block
        return res

    # TODO: is it okay to have extra kwargs qtotal?
    #  JU: probably
    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5,
                         *, qtotal = None) -> AbelianBackendData:
        # JU: res is not defined
        if qtotal is None:
            res.qtotal = qtotal = detect_qtotal(a, legs, atol, rtol)

        for block, qindices in zip(a.blocks, a.qdata):
            slices = []
            for (qi, leg) in zip(qindices, a.legs):
                sl = leg._abelian_data.slices
                slices.append(slice(sl[qi], sl[qi+1]))
            res[tuple(slices)] = block

        return res

    def from_block_func(self, func, legs: list[VectorSpace]) -> AbelianBackendData:
        raise NotImplementedError  # TODO

    def zero_data(self, legs: list[VectorSpace], dtype: Dtype):
        qtotal = legs[0].symmetry.trivial_sector
        return

    #  def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
    #      return self.eye_block(legs=[l.dim for l in legs], dtype=dtype)

    #  def copy_data(self, a: Tensor) -> Data:
    #      return self.block_copy(a.data)

    #  def _data_repr_lines(self, data: Data, indent: str, max_width: int, max_lines: int):
    #      return [f'{indent}* Data:'] + self._block_repr_lines(data, indent=indent + '  ', max_width=max_width,
    #                                                          max_lines=max_lines - 1)

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

    #  def transpose(self, a: Tensor, permutation: list[int]) -> Data:
    #      return self.block_transpose(a.data, permutation)

    #  def trace(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
    #      return self.block_trace(a.data, idcs1, idcs2)

    #  def conj(self, a: Tensor) -> Data:
    #      return self.block_conj(a.data)

    #  def combine_legs(self, a: Tensor, idcs: list[int], new_leg: FusionSpace) -> Data:
    #      return self.block_combine_legs(a.data, idcs)

    #  def split_leg(self, a: Tensor, leg_idx: int) -> Data:
    #      return self.block_split_leg(a, leg_idx, dims=[s.dim for s in a.legs[leg_idx]])

    #  def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
    #      return self.block_allclose(a.data, b.data, rtol=rtol, atol=atol)

    #  def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
    #      return self.block_squeeze_legs(a, idcs)

    #  def norm(self, a: Tensor) -> float:
    #      return self.block_norm(a.data)

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
