"""Backends for abelian group symmetries.

Changes compared to old np_conserved:
    - keep legs "sorted" and "bunched" at all times
    - keep qdata sorted (i.e. not arbitrary permutation in qinds)
    -

"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, TypeVar, List

import numpy as np

from .abstract_backend import AbstractBackend, AbstractBlockBackend, Data, Block
from ..symmetries import Symmetry, AbelianGroup, VectorSpace, ProductSpace
from ..tensors import Tensor, Dtype
from ...tools.optimization import use_cython


@dataclass
class AbelianBlockData:
    """Data stored in a tensor for AbelianBlockData"""
    dtype : Dtype  # numpy dtype (more than just np.float64)  # TODO: revert to using custom dtypes?
    qtotal : np.ndarray  # total charge
    blocks : List[Block]  # The actual entries of the tensor. Formerly known as Array._data
    qdata : np.ndarray  # For each of the blocks entries the qindices of the different legs.
    qdata_sorted : bool # Whether qdata is lexsorted. Defaults to `True`, but *must* be set to `False` by algorithms changing qdata.

    def copy(self, deep=True):
        if deep:
            return AbelianBlockData(self.dtype,
                                    self.qtotal.copy(),
                                    self.blocks.copy(),
                                    self.qdata.copy(),
                                    self.qdata_sorted)
        return AbelianBlockData(self.dtype,
                                self.qtotal,
                                self.blocks,
                                self.qdata,
                                self.qdata_sorted)

# JU: also dataclass?
class AbelianVectorSpaceData:
    def __init__(self, leg: VectorSpace):
        self.slices = np.cumsum(leg.multiplicities)

# JU: also dataclass?
class AbelianProductSpaceData:
    def __init__(self, pipe: ProductSpace):
        self.qmap = ...
        raise NotImplementedError("TODO")  # TODO



class AbstractAbelianBackend(AbstractBackend, AbstractBlockBackend, ABC):
    """Backend for Abelian group symmetries.

    Attributes
    ----------

    """
    def finalize_Tensor_init(self, a: Tensor) -> None:
        for leg in a.legs:
            if leg._abelian_data is None:
                if isinstance(leg, ProductSpace):  # TODO: hasattr(leg, 'spaces') is more robust
                    leg._abelian_data = AbelianProductSpaceData(leg)
                else:
                    leg._abelian_data = AbelianVectorSpaceData(leg)


    def get_dtype(self, a: Tensor) -> Dtype:
        return a.data.dtype  # TODO type vs numpy dtype?

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

    def item(self, a: Tensor) -> float | complex:
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
                         *, qtotal = None) -> Data:
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

    #  def combine_legs(self, a: Tensor, idcs: list[int], new_leg: ProductSpace) -> Data:
    #      return self.block_combine_legs(a.data, idcs)

    #  def split_leg(self, a: Tensor, leg_idx: int) -> Data:
    #      return self.block_split_leg(a, leg_idx, dims=[s.dim for s in a.legs[leg_idx]])

    #  def allclose(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
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

    #  def random_gaussian(self, legs: list[VectorSpace], dtype: Dtype, sigma: float) -> Data:
    #      return self.block_random_gaussian([l.dim for l in legs], dtype=dtype, sigma=sigma)

    #  def add(self, a: Tensor, b: Tensor) -> Data:
    #      return self.block_add(a.data, b.data)

    #  def mul(self, a: float | complex, b: Tensor) -> Data:
    #      return self.block_mul(a, b.data)


def detect_qtotal(flat_array, legcharges):
    inds_max = np.unravel_index(np.argmax(np.abs(flat_array)), flat_array.shape)
    val_max = abs(flat_array[inds_max])

    test_array = zeros(legcharges)  # Array prototype with correct charges
    qindices = [leg.get_qindex(i)[0] for leg, i in zip(legcharges, inds_max)]
    q = np.sum([l.get_charge(qi) for l, qi in zip(self.legs, qindices)], axis=0)
    return make_valid(q)  # TODO: leg.get_qindex, leg.get_charge
