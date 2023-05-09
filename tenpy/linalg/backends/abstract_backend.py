# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeVar, Any, TYPE_CHECKING
import numpy as np

from ..symmetries.groups import Symmetry
from ..symmetries.spaces import VectorSpace, ProductSpace

__all__ = ['Data', 'Block', 'AbstractBackend', 'AbstractBlockBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')

# placeholder for a backend-specific type that represents the blocks of symmetric tensors
Block = TypeVar('Block')

class Dtype(Enum):
    # TODO expose those in some high-level init, maybe even as tenpy.float32 ?
    # value = num_bytes * 2 + int(not is_real)
    float32 = 8
    complex64 = 9
    float64 = 10
    complex128 = 11

    @property
    def is_real(dtype):
        return dtype.value % 2 == 0

    @property
    def to_complex(dtype):
        if dtype.value % 2 == 1:
            return dtype
        return MyDtype(dtype.value + 1)

    def common(*dtypes):
        res = MyDtype(max(*(t.value for t in dtypes)))
        if res.is_real:
            if not all(t.is_real for t in dtypes):
                return MyDtype(res.value + 1)  # = res.to_complex
        return res


class AbstractBackend(ABC):
    """
    Inheritance structure:

            AbstractBackend           AbstractBlockBackend
                  |                            |
          AbstractXxxBackend            YyyBlockBackend
                  |                            |
                  ------------------------------
                                |
                          XxxYyyBackend

    Where Xxx describes the symmetry, e.g. NoSymmetry, Abelian, Nonabelian
    and Yyy describes the numerical routines that handle the blocks, e.g. numpy, torch, ...
    """

    def __repr__(self):
        return f'{type(self).__name__}'

    def __str__(self):
        return f'{type(self).__name__}'

    def convert_vector_space(self, leg: VectorSpace) -> VectorSpace:
        """convert a VectorSpace (or ProductSpace) instance to a backend-specific subclass"""
        return leg

    @abstractmethod
    def get_dtype_from_data(self, a: Data) -> Dtype:
        ...

    @abstractmethod
    def to_dtype(self, a: Tensor, dtype: Dtype) -> Data:
        """cast to given dtype. No copy if already has dtype."""
        ...

    @abstractmethod
    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        ...

    @abstractmethod
    def is_real(self, a: Tensor) -> bool:
        """If the Tensor is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        ...

    def item(self, a: Tensor) -> float | complex:
        """Assumes that tensor is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        return self.data_item(a.data)

    @abstractmethod
    def data_item(self, a: Data) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def to_dense_block(self, a: Tensor) -> Block:
        """Forget about symmetry structure and convert to a single block."""
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Data:
        """Convert a dense block to the data for a symmetric tensor.
        If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
        where ``projected`` is `a` projected to the space of symmetric tensors
        """
        ...

    @abstractmethod
    def from_block_func(self, func, legs: list[VectorSpace]) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int]) -> block``."""
        ...

    @abstractmethod
    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        """Data for a zero tensor"""
        ...

    @abstractmethod
    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        """Data for an identity map from legs to their duals. In particular, the resulting tensor
        has twice as many legs"""
        ...

    @abstractmethod
    def copy_data(self, a: Tensor) -> Data:
        """Return a copy, such that future in-place operations on the output data do not affect the input data"""
        ...

    @abstractmethod
    def _data_repr_lines(self, data: Data, indent: str, max_width: int, max_lines: int) -> list[str]:
        """helper function for Tensor.__repr__ ; return a list of strs which are the lines
        comprising the ``"* Data:"``section.
        indent is to be placed in front of every line"""
        ...

    @abstractmethod
    def tdot(self, a: Tensor, b: Tensor, axs_a: list[int], axs_b: list[int]) -> Data:
        """Tensordot i.e. pairwise contraction"""
        ...

    @abstractmethod
    def svd(self, a: Tensor, axs1: list[int], axs2: list[int], new_leg: VectorSpace | None
            ) -> tuple[Data, Data, Data, VectorSpace]:
        """
        SVD of a tensor, interpreted as a linear map / matrix from axs1 to axs2.

        Development Notes
        -----------------
        - abelian backend: if len(axs1) > 1 or len(axs2) > 1, call combine legs and warn that this may
        be inefficient.

        Returns
        -------
        u, s, vh, new_leg
        """
        ...

    @abstractmethod
    def outer(self, a: Tensor, b: Tensor) -> Data:
        ...

    @abstractmethod
    def inner(self, a: Tensor, b: Tensor, axs2: list[int] | None) -> complex:
        """
        inner product of <a|b>, both of which are given as ket-like vectors
        (i.e. in C^N, the entries of a would need to be conjugated before multiplying with entries of b)
        axs2, if not None, gives the order of the axes of b
        """
        ...

    @abstractmethod
    def transpose(self, a: Tensor, permutation: list[int]) -> Data:
        ...

    @abstractmethod
    def trace(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        ...

    @abstractmethod
    def conj(self, a: Tensor) -> Data:
        ...

    @abstractmethod
    def combine_legs(self, a: Tensor, combine_slices: list[int, int], product_spaces: list[ProductSpace], new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        """combine legs of `a` (without transpose).

        ``combine_slices[i]=(begin, end)`` sorted in ascending order of `begin` indicates that
        ``a.legs[begin:end]`` is to be combined to `product_spaces[i]`, yielding `final_legs`.
        `new_axes[i]` is the index of `product_spaces[i]` in `final_legs` (also fixed by `combine_slices`).
        """
        ...

    @abstractmethod
    def split_legs(self, a: Tensor, leg_idcs: list[int], final_legs: list[VectorSpace]) -> Data:
        """split multiple product space legs."""
        ...

    @abstractmethod
    def almost_equal(self, a: Tensor, b: Tensor, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def squeeze_legs(self, a: Tensor, idcs: list[int]) -> Data:
        """Assume the legs at given indices are trivial and get rid of them"""
        ...

    @abstractmethod
    def norm(self, a: Tensor) -> float:
        ...

    @abstractmethod
    def exp(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        ...

    @abstractmethod
    def log(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> Data:
        ...

    @abstractmethod
    def random_normal(self, legs: list[VectorSpace], dtype: Dtype, sigma: float) -> Data:
        """generate the data for a tensor drawn randomly from the normal distribution with zero mean
        and standard deviation sigma"""
        ...

    @abstractmethod
    def add(self, a: Tensor, b: Tensor) -> Data:
        ...

    @abstractmethod
    def mul(self, a: float | complex, b: Tensor) -> Data:
        ...


class AbstractBlockBackend(ABC):
    svd_algorithms: list[str]  # first is default

    @abstractmethod
    def block_from_numpy(self, a) -> Block:
        ...

    @abstractmethod
    def block_is_real(self, a: Block) -> bool:
        """If the block is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        ...

    @abstractmethod
    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]
                   ) -> Block:
        ...

    @abstractmethod
    def block_shape(self, a: Block) -> tuple[int]:
        ...

    @abstractmethod
    def block_item(self, a: Block) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def block_dtype(self, a: Block) -> Dtype:
        ...

    @abstractmethod
    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        ...

    def block_to_numpy(self, a: Block, numpy_dtype=None) -> np.ndarray:
        # BlockBackends may override, if this implementation is not valid
        return np.asarray(a, dtype=numpy_dtype)

    @abstractmethod
    def block_copy(self, a: Block) -> Block:
        ...

    @abstractmethod
    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        ...

    @abstractmethod
    def block_outer(self, a: Block, b: Block) -> Block:
        ...

    @abstractmethod
    def block_inner(self, a: Block, b: Block, axs2: list[int] | None) -> complex:
        ...

    @abstractmethod
    def block_transpose(self, a: Block, permutation: list[int]) -> Block:
        ...

    @abstractmethod
    def block_trace(self, a: Block, idcs1: list[int], idcs2: list[int]) -> Block:
        ...

    @abstractmethod
    def block_conj(self, a: Block) -> Block:
        """complex conjugate of a block"""
        ...

    def block_combine_legs(self, a: Block, legs_slices: list[tuple[int]]) -> Block:
        """no transpose, only reshape ``legs[b:e] for b,e in legs_slicse`` to single legs"""
        old_shape = self.block_shape(a)
        new_shape = []
        last_e = 0
        for b, e in legs_slices:  # ascending!
            new_shape.extend(old_shape[last_e:b])
            new_shape.append(np.product(old_shape[b:e]))
            last_e = e
        new_shape.extend(old_shape[last_e:])
        return self.block_reshape(a, tuple(new_shape))

    def block_split_legs(self, a: Block, idcs: list[int], dims: list[list[int]]) -> Block:
        old_shape = self.block_shape(a)
        new_shape = []
        start = 0
        for i, i_dims in zip(idcs, dims):
            new_shape.extend(old_shape[start:i])
            new_shape.extend(i_dims)
            start = i + 1
        new_shape.extend(old_shape[start:])
        return self.block_reshape(a, tuple(new_shape))

    @abstractmethod
    def block_allclose(self, a: Block, b: Block, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        ...

    @abstractmethod
    def block_norm(self, a: Block) -> float:
        ...

    @abstractmethod
    def block_max_abs(self, a: Block) -> float:
        ...

    @abstractmethod
    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        ...

    @abstractmethod
    def block_matrixify(self, a: Block, idcs1: list[int], idcs2: list[int]) -> tuple[Block, Any]:
        """reshape to a matrix. return that matrix and data necessary to revert it"""
        ...

    @abstractmethod
    def block_dematrixify(self, matrix: Block, aux: Any) -> Block:
        ...

    @abstractmethod
    def matrix_dot(self, a: Block, b: Block) -> Block:
        """As in numpy.dot, both a and b might be matrix or vector."""
        ...

    @abstractmethod
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """SVD of a 2D block"""
        ...

    @abstractmethod
    def matrix_exp(self, matrix: Block) -> Block:
        ...

    @abstractmethod
    def matrix_log(self, matrix: Block) -> Block:
        ...

    @abstractmethod
    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        ...

    def block_add(self, a: Block, b: Block) -> Block:
        return a + b

    def block_mul(self, a: float | complex, b: Block) -> Block:
        return a * b

    @abstractmethod
    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def eye_block(self, legs: list[int], dtype: Dtype) -> Data:
        """eye from legs to dual of legs (result has ``2 * len(legs)`` axes!!)"""
        ...
