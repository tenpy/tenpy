# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeVar, Any, TYPE_CHECKING, Type
from numbers import Number
import numpy as np

from ..symmetries.groups import Symmetry
from ..symmetries.spaces import VectorSpace, ProductSpace, _fuse_spaces

__all__ = ['Data', 'Block', 'AbstractBackend', 'AbstractBlockBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor, DiagonalTensor, Mask

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')

# placeholder for a backend-specific type that represents the blocks of symmetric tensors
Block = TypeVar('Block')

class Dtype(Enum):
    # TODO expose those in some high-level init, maybe even as tenpy.float32 ?
    # value = num_bytes * 2 + int(not is_real)
    bool = 2
    float32 = 8
    complex64 = 9
    float64 = 10
    complex128 = 11

    @property
    def is_real(dtype):
        return dtype.value % 2 == 0

    @property
    def to_complex(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to complex')
        if dtype.value % 2 == 1:
            return dtype
        return Dtype(dtype.value + 1)

    @property
    def to_real(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to real')
        if dtype.value % 2 == 0:
            return dtype
        return Dtype(dtype.value - 1)

    @property
    def python_type(dtype):
        if dtype.value == 2:
            return bool
        if dtype.is_real:
            return float
        return complex

    @property
    def zero_scalar(dtype):
        return dtype.python_type(0)

    def common(*dtypes):
        res = Dtype(max(t.value for t in dtypes))
        if res.is_real:
            if not all(t.is_real for t in dtypes):
                return Dtype(res.value + 1)  # = res.to_complex
        return res

    def convert_python_scalar(dtype, value) -> complex | float | bool:
        if dtype.value == 2:  # Dtype.bool
            if value in [True, False, 0, 1]:
                return bool(value)
        elif dtype.is_real:
            if isinstance(value, (int, float)):
                return float(value)
            # TODO what should we do for complex values?
        else:
            if isinstance(value, Number):
                return complex(value)
        raise TypeError(f'Type {type(value)} is incompatible with dtype {dtype}')


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
    DataCls = Block

    def test_data_sanity(self, a: Tensor | DiagonalTensor | Mask, is_diagonal: bool):
        assert isinstance(a.data, self.DataCls)
        # note: no super(), this is the top you reach!
        # subclasses will typically call super().test_data_sanity(a)

    def test_leg_sanity(self, leg: VectorSpace):
        assert isinstance(leg, VectorSpace)

    def __repr__(self):
        return f'{type(self).__name__}'

    def __str__(self):
        return f'{type(self).__name__}'

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool):
        """Backends may override the behavior of linalg.spaces._fuse_spaces in order to compute
        their backend-specfic metadata alongside the sectors"""
        return _fuse_spaces(symmetry=symmetry, spaces=spaces, _is_dual=_is_dual)

    def add_leg_metadata(self, leg: VectorSpace) -> VectorSpace:
        """Add backend-specifc metadata to a leg (modifying it in-place) and returning it"""
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

    def is_real(self, a: Tensor) -> bool:
        """If the Tensor is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        # NonAbelian backend might implement this differently.
        return a.dtype.is_real

    def item(self, a: Tensor | DiagonalTensor) -> float | complex:
        """Assumes that tensor is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        return self.data_item(a.data)

    @abstractmethod
    def data_item(self, a: Data | DiagonalData) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def to_dense_block(self, a: Tensor) -> Block:
        """Forget about symmetry structure and convert to a single block.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see VectorSpace.sector_perm or VectorSpace.index_perm).
        """
        ...

    @abstractmethod
    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        """Forget about symmetry structure and convert the diagonals of the blocks
        to a single 1D block.
        This is the diagonal of the respective non-symmetric 2D tensor.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see VectorSpace.sector_perm or VectorSpace.index_perm).

        Equivalent to self.block_get_diagonal(a.to_full_tensor().to_dense_block())
        """
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, legs: list[VectorSpace], atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Data:
        """Convert a dense block to the data for a symmetric tensor.
        If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
        where ``projected`` is `a` projected to the space of symmetric tensors
        This includes a permutation of the basis, specified by the legs of `a`.
        (see VectorSpace.sector_perm or VectorSpace.index_perm).
        """
        ...

    @abstractmethod
    def diagonal_from_block(self, a: Block, leg: VectorSpace) -> DiagonalData:
        """DiagonalData from a 1D block.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see VectorSpace.sector_perm or VectorSpace.index_perm).
        """
        ...

    @abstractmethod
    def from_block_func(self, func, legs: list[VectorSpace], func_kwargs={}) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int]) -> Block``."""
        ...

    @abstractmethod
    def diagonal_from_block_func(self, func, leg: VectorSpace, func_kwargs={}) -> DiagonalData:
        ...

    @abstractmethod
    def zero_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        """Data for a zero tensor"""
        ...

    @abstractmethod
    def zero_diagonal_data(self, leg: VectorSpace, dtype: Dtype) -> DiagonalData:
        ...

    @abstractmethod
    def eye_data(self, legs: list[VectorSpace], dtype: Dtype) -> Data:
        """Data for an identity map from legs to their duals. In particular, the resulting tensor
        has twice as many legs"""
        ...

    @abstractmethod
    def copy_data(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
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
    def svd(self, a: Tensor, new_vh_leg_dual: bool) -> tuple[Data, DiagonalData, Data, VectorSpace]:  # TODO: Data -> DiagonalData for S
        """
        SVD of a Matrix, `a` has only two legs (often ProductSpace).

        Returns
        -------
        u, s, vh :
            Data of corresponding tensors.
        new_leg :
            (Backend-specific) VectorSpace the new leg of vh.
        """
        ...

    @abstractmethod
    def qr(self, a: Tensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, VectorSpace]:
        """QR decomposition of a Matrix `a` with two legs (which may be ProductSpace)

        Returns
        -------
        q, r:
            Data of corresponding tensors.
        new_leg :
            (Backend-specific) VectorSpace the new leg of r.
        """
        ...

    @abstractmethod
    def outer(self, a: Tensor, b: Tensor) -> Data:
        ...

    @abstractmethod
    def inner(self, a: Tensor, b: Tensor, do_conj: bool, axs2: list[int] | None) -> float | complex:
        """
        inner product of <a|b>, both of which are given as ket-like vectors
        (i.e. in C^N, the entries of a would need to be conjugated before multiplying with entries of b)
        axs2, if not None, gives the order of the axes of b.
        If do_conj, a is assumed as a "ket vector", in the same space as b, which will need to be conjugated.
        Otherwise, a is assumed as a "bra vector", in the dual space, s.t. no conj is needed.
        """
        ...

    @abstractmethod
    def permute_legs(self, a: Tensor, permutation: list[int]) -> Data:
        ...

    @abstractmethod
    def trace_full(self, a: Tensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        ...

    @abstractmethod
    def trace_partial(self, a: Tensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        ...

    @abstractmethod
    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        ...

    @abstractmethod
    def conj(self, a: Tensor | DiagonalTensor) -> Data | DiagonalData:
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
    def norm(self, a: Tensor | DiagonalTensor) -> float:
        ...

    @abstractmethod
    def act_block_diagonal_square_matrix(self, a: Tensor, block_method: str) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`.

        block_method :
            Name of a BlockBackend method with signature ``block_method(a: Block) -> Block``.
        """
        ...

    @abstractmethod
    def add(self, a: Tensor, b: Tensor) -> Data:
        ...

    @abstractmethod
    def mul(self, a: float | complex, b: Tensor) -> Data:
        ...

    @abstractmethod
    def infer_leg(self, block: Block, legs: list[VectorSpace | None], is_dual: bool = False,
                  is_real: bool = False) -> VectorSpace:
        """Infer a missing leg from the dense block"""
        # TODO make it poss
        ...

    @abstractmethod
    def get_element(self, a: Tensor, idcs: list[int]) -> complex | float | bool:
        """Get a single scalar element from a tensor.

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs
            - 0 <= idx < leg.dim
            - the indices reference an allowed (by the charge rule) entry.
            The indices are w.r.t. the internal (sorted) order.
        """
        ...

    @abstractmethod
    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        """Get a single scalar element from a diagonal tensor.

        Parameters
        ----------
        idx
            The index for both legs. Checks have already been performed, i.e. we may assume that
            - 0 <= idx < leg.dim
            The index are w.r.t. the internal (sorted) order.
        """
        ...

    @abstractmethod
    def set_element(self, a: Tensor, idcs: list[int], value: complex | float) -> Data:
        """Return a copy of the data of a tensor, with a single element changed.

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs
            - 0 <= idx < leg.dim
            - the indices reference an allowed (by the charge rule) entry.
            The indices are w.r.t. the internal (sorted) order.
        value
            A value of the appropriate type ``a.dtype.python_type``.
        """
        ...

    @abstractmethod
    def set_element_diagonal(self, a: DiagonalTensor, idx: int, value: complex | float | bool
                             ) -> DiagonalData:
        """Return a copy of the data of a diagonal tensor, with a single element changed.

        Parameters
        ----------
        idx
            The index for both legs. Checks have already been performed, i.e. we may assume that
            - 0 <= idx < leg.dim
            The index are w.r.t. the internal (sorted) order.
        value
            A value of the appropriate type ``a.dtype.python_type``.
        """
        ...

    @abstractmethod
    def diagonal_data_from_full_tensor(self, a: Tensor, check_offdiagonal: bool) -> DiagonalData:
        """Get the DiagonalData corresponding to a tensor with two legs"""
        ...

    @abstractmethod
    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        ...

    @abstractmethod
    def full_data_from_mask(self, a: Mask) -> Data:
        ...

    @abstractmethod
    def scale_axis(self, a: Tensor, b: DiagonalTensor, leg: int) -> Data:
        """Scale axis ``leg`` of ``a`` with ``b``, then permute legs to move the scaled leg to given position"""
        ...

    @abstractmethod
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        """Apply a function ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
        ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
        """
        ...

    @abstractmethod
    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_identity: bool, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        """Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
        pairs of elements.
        Input tensors are both DiagonalTensor and have equal legs.
        ``partial_zero_is_identity=True`` promises that ``func(any_block, zero_block) == any_block``,
        and similarly for the second argument.
        ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
        and similarly for the second argument.
        """
        ...

    @abstractmethod
    def fuse_states(self, state1: Block | None, state2: Block | None, space1: VectorSpace,
                    space2: VectorSpace, product_space: ProductSpace = None) -> Block | None:
        """Given states in two VectorSpaces, compute the respective state in the product space.

        States can be specified as 1D blocks or as ``None``, which represents ``[1.]``.
        """
        ...

    @abstractmethod
    def mask_infer_small_leg(self, mask_data: Data, large_leg: VectorSpace) -> VectorSpace:
        """Infer the smaller leg that a mask with the given data projects to."""
        ...

    @abstractmethod
    def apply_mask_to_Tensor(self, tensor: Tensor, mask: Mask, leg_idx: int) -> Data:
        ...


class AbstractBlockBackend(ABC):
    svd_algorithms: list[str]  # first is default

    @abstractmethod
    def block_from_numpy(self, a: np.ndarray) -> Block:
        ...

    def block_is_real(self, a: Block) -> bool:
        """If the block is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        return self.tenpy_dtype_map[self.block_dtype(a)].is_real

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
    def block_inner(self, a: Block, b: Block, do_conj: bool, axs2: list[int] | None) -> float | complex:
        ...

    @abstractmethod
    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        ...

    @abstractmethod
    def block_trace_full(self, a: Block, idcs1: list[int], idcs2: list[int]) -> float | complex:
        ...

    @abstractmethod
    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Block:
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
        # TODO rename to squeeze_axes ?
        ...

    @abstractmethod
    def block_add_axis(self, a: Block, pos: int) -> Block:
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
    def matrix_dot(self, a: Block, b: Block) -> Block:
        """As in numpy.dot, both a and b might be matrix or vector."""
        ...

    @abstractmethod
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """SVD of a 2D block.

        With full_matrices=False, i.e. shape ``(n,m) -> (n,k), (k,) (k,m)`` where
        ``k <= min(n,m)``.
        """
        ...

    @abstractmethod
    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        """QR decomposition of a 2D block"""
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
    def ones_block(self, shape: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def eye_block(self, legs: list[int], dtype: Dtype) -> Data:
        """eye from legs to dual of legs (result has ``2 * len(legs)`` axes!!)"""
        ...

    @abstractmethod
    def block_kron(self, a: Block, b: Block) -> Block:
        """The kronecker product, like numpy.kron"""
        ...

    @abstractmethod
    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        ...

    @abstractmethod
    def set_block_element(self, a: Block, idcs: list[int], value: complex | float | bool) -> Block:
        """Return a modified copy, with the entry at `idcs` set to `value`"""
        ...

    @abstractmethod
    def block_get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        """Get the diagonal of a 2D block as a 1D block"""
        ...

    @abstractmethod
    def block_from_diagonal(self, diag: Block) -> Block:
        """Return a 2D square block that has the 1D ``diag`` on the diagonal"""
        ...

    @abstractmethod
    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        """Return a (M, N) of numbers (floa or complex dtype) from a 1D bool-valued block shape (M,)
        where N is the number of True entries. The result is the coefficient matrix of the projection map."""
        ...

    def block_scale_axis(self, block: Block, factors: Block, axis: int) -> Block:
        """multiply block with the factors (a 1D block), along a given axis.
        E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
        ``block * factors[None, None, :, None]``.
        """
        idx = [None] * len(self.block_shape(block))
        idx[axis] = slice(None, None,  None)
        return block * factors[idx]

    @abstractmethod
    def block_sum_all(self, a: Block) -> float | complex:
        """The sum of all entries of the block.
        If the block contains boolean values, this should return the number of ``True`` entries.
        """
        ...

    def apply_mask_to_block(block: Block, mask: Block, ax: int) -> Block:
        """Apply a mask (1D boolean block) to a block, slicing/projecting that axis"""
        idx = (slice(None, None, None),) * (ax - 1) + (mask,)
        return block[idx]
    