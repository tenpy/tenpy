"""TODO
Also contains some private utility function used by multiple backend modules.

"""

# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TypeVar, TYPE_CHECKING, Callable
import numpy as np

from ..symmetries import Symmetry
from ..spaces import VectorSpace, ProductSpace, _fuse_spaces
from ..dtypes import Dtype

__all__ = ['Data', 'DiagonalData', 'Block', 'Backend', 'BlockBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import BlockDiagonalTensor, DiagonalTensor, Mask

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')

# placeholder for a backend-specific type that represents the blocks of symmetric tensors
Block = TypeVar('Block')


class Backend(metaclass=ABCMeta):
    """Abstract base class for backends.

    A backend implements functions that acts on tensors.
    We abstract two separate concepts for a backend.
    There is a block backend, that abstracts what the numerical data format (numpy array,
    torch Tensor, CUDA tensor, ...) and a SymmetryBackend that abstracts how block-sparse
    structures that arise from symmetries are accounted for.

    The implementation strategy is then to implement a BlockBackend subclass for every type of
    block we want to support. Similarly, we implement a direct subclass of Backend for every
    class of symmetry (no symmetry, abelian symmetry, nonabelian symmetry, more general grading)
    that uses the methods provided by the BlockBackend.
    In the simplest case, a concrete backend with a specific block type and symmetry class can
    then by implemented simply by inheriting from both of those, e.g. ::
    
        |           Backend                    BlockBackend
        |              |                            |
        |          XxxBackend                 YyyBlockBackend
        |              |                            |
        |              ------------------------------
        |                            |
        |                      XxxYyyBackend

    Where Xxx describes the symmetry backend, e.g. NoSymmetry, Abelian, FusionTree
    and Yyy describes the numerical routines that handle the blocks, e.g. numpy, torch, ...

    However, the ``XxxYyyBackend`` class may also override any of the methods, if needed.
    """
    DataCls = None  # to be set by subclasses

    def test_data_sanity(self, a: BlockDiagonalTensor | DiagonalTensor, is_diagonal: bool):
        # subclasses will typically call super().test_data_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def test_mask_sanity(self, a: Mask):
        # subclasses will typically call super().test_mask_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def test_leg_sanity(self, leg: VectorSpace):
        # subclasses will typically call super().test_leg_sanity(a)
        assert isinstance(leg, VectorSpace)
        leg.test_sanity()

    def __repr__(self):
        return f'{type(self).__name__}'

    def __str__(self):
        return f'{type(self).__name__}'

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[VectorSpace], _is_dual: bool):
        """Backends may override the behavior of linalg.spaces._fuse_spaces in order to compute
        their backend-specific metadata alongside the sectors.

        Note that the implementation of ``VectorSpace.dual`` assumes that the metadata of the
        resulting dual space is the same as for the original space.
        """
        return _fuse_spaces(symmetry=symmetry, spaces=spaces, _is_dual=_is_dual)

    def add_leg_metadata(self, leg: VectorSpace) -> VectorSpace:
        """Add backend-specific metadata to a leg (modifying it in-place) and returning it.

        Note that the implementation of ``VectorSpace.dual`` assumes that the metadata of the
        resulting dual space is the same as for the original space.
        """
        return leg

    @abstractmethod
    def get_dtype_from_data(self, a: Data) -> Dtype:
        ...

    @abstractmethod
    def to_dtype(self, a: BlockDiagonalTensor, dtype: Dtype) -> Data:
        """cast to given dtype. No copy if already has dtype."""
        ...

    @abstractmethod
    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        ...

    def is_real(self, a: BlockDiagonalTensor) -> bool:
        """If the Tensor is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        # FusionTree backend might implement this differently.
        return a.dtype.is_real

    def item(self, a: BlockDiagonalTensor | DiagonalTensor) -> float | complex:
        """Assumes that tensor is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        return self.data_item(a.data)

    @abstractmethod
    def data_item(self, a: Data | DiagonalData) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def to_dense_block(self, a: BlockDiagonalTensor) -> Block:
        """Forget about symmetry structure and convert to a single block.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see e.g. VectorSpace.basis_perm).
        """
        ...

    @abstractmethod
    def diagonal_to_block(self, a: DiagonalTensor) -> Block:
        """Forget about symmetry structure and convert the diagonals of the blocks
        to a single 1D block.
        This is the diagonal of the respective non-symmetric 2D tensor.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see e.g. VectorSpace.basis_perm).

        Equivalent to self.block_get_diagonal(a.to_full_tensor().to_dense_block())
        """
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, legs: list[VectorSpace], num_domain_legs: int,
                         tol: float = 1e-8) -> Data:
        """Convert a dense block to the data for a symmetric tensor.
        
        If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
        where ``projected`` is `a` projected to the space of symmetric tensors, raise a ``ValueError``.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see e.g. VectorSpace.basis_perm).
        """
        ...

    @abstractmethod
    def diagonal_from_block(self, a: Block, leg: VectorSpace) -> DiagonalData:
        """DiagonalData from a 1D block.
        This includes a permutation of the basis, specified by the legs of `a`.
        (see e.g. VectorSpace.basis_perm).
        """
        ...

    @abstractmethod
    def mask_from_block(self, a: Block, large_leg: VectorSpace, small_leg: VectorSpace
                        ) -> DiagonalData:
        """DiagonalData for a Mask from a 1D block.
        
        This includes a permutation of the basis, specified by the legs of `a`.
        (see e.g. VectorSpace.basis_perm).
        """
        ...

    @abstractmethod
    def from_block_func(self, func, legs: list[VectorSpace], num_domain_legs: int, func_kwargs={}
                        ) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int]) -> Block``."""
        ...

    @abstractmethod
    def diagonal_from_block_func(self, func, leg: VectorSpace, func_kwargs={}) -> DiagonalData:
        ...

    @abstractmethod
    def zero_data(self, legs: list[VectorSpace], dtype: Dtype, num_domain_legs: int) -> Data:
        """Data for a zero tensor"""
        ...

    @abstractmethod
    def zero_diagonal_data(self, leg: VectorSpace, dtype: Dtype) -> DiagonalData:
        ...

    @abstractmethod
    def eye_data(self, legs: list[VectorSpace], dtype: Dtype, num_domain_legs: int) -> Data:
        """Data for an identity map from legs to their duals. In particular, the resulting tensor
        has twice as many legs"""
        ...

    @abstractmethod
    def copy_data(self, a: BlockDiagonalTensor | DiagonalTensor) -> Data | DiagonalData:
        """Return a copy, such that future in-place operations on the output data do not affect the input data"""
        ...

    @abstractmethod
    def _data_repr_lines(self, a: BlockDiagonalTensor, indent: str, max_width: int, max_lines: int) -> list[str]:
        """helper function for Tensor.__repr__ ; return a list of strs which are the lines
        comprising the ``"* Data:"``section.
        indent is to be placed in front of every line"""
        ...

    @abstractmethod
    def tdot(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, axs_a: list[int], axs_b: list[int]) -> Data:
        """Tensordot i.e. pairwise contraction"""
        ...

    @abstractmethod
    def svd(self, a: BlockDiagonalTensor, new_vh_leg_dual: bool, algorithm: str | None, compute_u: bool,
            compute_vh: bool) -> tuple[Data, DiagonalData, Data, VectorSpace]:
        """SVD of a Matrix, `a` has only two legs (often ProductSpace).
        
        Parameters
        ----------
        algorithm : str
            (Backend-specific) algorithm to use for computing the SVD.
            See e.g. the :attr:`~BlockBackend.svd_algorithms` attribute.
            We also implement ``'eigh'`` for all backends.
        compute_u, compute_vh : bool
            Only for ``algorithm='eigh'``.
        
        Returns
        -------
        u, s, vh :
            Data of corresponding tensors.
        new_leg :
            (Backend-specific) VectorSpace the new leg of vh.
        """
        ...

    @abstractmethod
    def qr(self, a: BlockDiagonalTensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, VectorSpace]:
        """QR decomposition of a Tensor `a` with two legs.

        The legs of `a` may be :class:`~tenpy.linalg.spaces.ProductSpace`

        Returns
        -------
        q, r:
            Data of corresponding tensors.
        new_leg : VectorSpace
            the new leg of r.
        """
        ...

    @abstractmethod
    def outer(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        ...

    @abstractmethod
    def inner(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, do_conj: bool, axs2: list[int] | None) -> float | complex:
        """
        inner product of <a|b>, both of which are given as ket-like vectors
        (i.e. in C^N, the entries of a would need to be conjugated before multiplying with entries of b)
        axs2, if not None, gives the order of the axes of b.
        If do_conj, a is assumed as a "ket vector", in the same space as b, which will need to be conjugated.
        Otherwise, a is assumed as a "bra vector", in the dual space, s.t. no conj is needed.
        """
        ...

    @abstractmethod
    def permute_legs(self, a: BlockDiagonalTensor, permutation: list[int] | None,
                     num_domain_legs: int) -> Data:
        ...

    @abstractmethod
    def trace_full(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        ...

    @abstractmethod
    def trace_partial(self, a: BlockDiagonalTensor, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Data:
        ...

    @abstractmethod
    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        ...

    @abstractmethod
    def conj(self, a: BlockDiagonalTensor | DiagonalTensor) -> Data | DiagonalData:
        ...

    @abstractmethod
    def combine_legs(self, a: BlockDiagonalTensor, combine_slices: list[int, int], product_spaces: list[ProductSpace], new_axes: list[int], final_legs: list[VectorSpace]) -> Data:
        """combine legs of `a` (without transpose).

        ``combine_slices[i]=(begin, end)`` sorted in ascending order of `begin` indicates that
        ``a.legs[begin:end]`` is to be combined to `product_spaces[i]`, yielding `final_legs`.
        `new_axes[i]` is the index of `product_spaces[i]` in `final_legs` (also fixed by `combine_slices`).
        """
        ...

    @abstractmethod
    def split_legs(self, a: BlockDiagonalTensor, leg_idcs: list[int], final_legs: list[VectorSpace]) -> Data:
        """split multiple product space legs."""
        ...

    @abstractmethod
    def add_trivial_leg(self, a: BlockDiagonalTensor, pos: int, to_domain: bool) -> Data:
        ...

    @abstractmethod
    def almost_equal(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor, rtol: float, atol: float) -> bool:
        ...
        
    def almost_equal_diagonal(self, a: DiagonalTensor, b: DiagonalTensor, rtol: float, atol: float
                              ) -> bool:
        # for most backends, almost_equal will just work, but if not, backends may override this.
        return self.almost_equal(a, b, rtol, atol)

    @abstractmethod
    def squeeze_legs(self, a: BlockDiagonalTensor, idcs: list[int]) -> Data:
        """Assume the legs at given indices are trivial and get rid of them"""
        ...

    @abstractmethod
    def norm(self, a: BlockDiagonalTensor | DiagonalTensor, order: int | float = None) -> float:
        """Norm of a tensor. order has already been parsed and is a number"""
        ...

    @abstractmethod
    def act_block_diagonal_square_matrix(self, a: BlockDiagonalTensor, block_method: Callable[[Block], Block]
                                         ) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`.

        Parameters
        ----------
        a : Tensor
            The tensor to act on
        block_method : function
            A function with signature ``block_method(a: Block) -> Block`` acting on backend-blocks.
        """
        ...

    @abstractmethod
    def add(self, a: BlockDiagonalTensor, b: BlockDiagonalTensor) -> Data:
        ...

    @abstractmethod
    def mul(self, a: float | complex, b: BlockDiagonalTensor) -> Data:
        ...

    @abstractmethod
    def infer_leg(self, block: Block, legs: list[VectorSpace | None], is_dual: bool = False,
                  is_real: bool = False) -> VectorSpace:
        """Infer a missing leg from the dense block"""
        # TODO make it poss
        ...

    @abstractmethod
    def get_element(self, a: BlockDiagonalTensor, idcs: list[int]) -> complex | float | bool:
        """Get a single scalar element from a tensor.

        TODO we might have a bit of redundancy in checking / parsing the indices

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs
            - 0 <= idx < leg.dim
            - the indices reference an allowed (by the charge rule) entry.
        """
        ...

    @abstractmethod
    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        """Get a single scalar element from a diagonal tensor.

        Parameters
        ----------
        idx
            The index for both legs. Checks have already been performed, i.e. we may assume that
            ``0 <= idx < leg.dim``
        """
        ...

    @abstractmethod
    def set_element(self, a: BlockDiagonalTensor, idcs: list[int], value: complex | float) -> Data:
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
    def diagonal_data_from_full_tensor(self, a: BlockDiagonalTensor, check_offdiagonal: bool) -> DiagonalData:
        """Get the DiagonalData corresponding to a tensor with two legs.
        Can assume that the two legs are either equal or dual, such that their ._non_dual_sectors match"""
        ...

    @abstractmethod
    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        ...

    @abstractmethod
    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        ...

    @abstractmethod
    def scale_axis(self, a: BlockDiagonalTensor, b: DiagonalTensor, leg: int) -> Data:
        """Scale axis ``leg`` of ``a`` with ``b``, then permute legs to move the scaled leg to given position"""
        ...

    @abstractmethod
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
        ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
        """
        ...

    @abstractmethod
    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
        pairs of elements.
        Input tensors are both DiagonalTensor and have equal legs.
        ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
        and similarly for the second argument.
        """
        ...

    @abstractmethod
    def apply_mask_to_Tensor(self, tensor: BlockDiagonalTensor, mask: Mask, leg_idx: int) -> Data:
        ...

    @abstractmethod
    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        ...

    @abstractmethod
    def eigh(self, a: BlockDiagonalTensor, sort: str = None) -> tuple[DiagonalData, Data]:
        """Eigenvalue decomposition of a 2-leg hermitian tensor

        Parameters
        ----------
        a
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted *within* each charge block.
            See :func:`argsort` for details.
        """
        ...

    @abstractmethod
    def from_flat_block_trivial_sector(self, block: Block, leg: VectorSpace) -> Data:
        """Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector."""
        ...

    @abstractmethod
    def to_flat_block_trivial_sector(self, tensor: BlockDiagonalTensor) -> Block:
        """Single-leg tensor to the *part of* the coefficients in the trivial sector."""
        ...

    @abstractmethod
    def inv_part_from_flat_block_single_sector(self, block: Block, leg: VectorSpace, dummy_leg: VectorSpace) -> Data:
        """Data for the invariant part used in ChargedTensor.from_flat_block_single_sector"""
        ...

    @abstractmethod
    def inv_part_to_flat_block_single_sector(self, tensor: BlockDiagonalTensor) -> Block:
        """Inverse of inv_part_from_flat_block_single_sector"""
        ...

    @abstractmethod
    def flip_leg_duality(self, tensor: BlockDiagonalTensor, which_legs: list[int],
                         flipped_legs: list[VectorSpace], perms: list[np.ndarray]) -> Data:
        ...


class BlockBackend(metaclass=ABCMeta):
    """Abstract base class that defines the operation on dense blocks."""
    svd_algorithms: list[str]  # first is default

    @abstractmethod
    def as_block(self, a) -> Block:
        """Convert objects to blocks. Should support blocks, numpy arrays, nested python containers.
        May support more."""
        ...
    
    @abstractmethod
    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None) -> Block:
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

    @abstractmethod
    def block_angle(self, a: Block) -> Block:
        """The angle of a complex number such that ``a == exp(1.j * angle(a))``. Elementwise."""
        ...

    @abstractmethod
    def block_real(self, a: Block) -> Block:
        """The real part of a complex number, elementwise."""
        ...

    @abstractmethod
    def block_real_if_close(self, a: Block, tol: float) -> Block:
        """If a block is close to its real part, return the real part. Otherwise the original block.
        Elementwise."""
        ...

    @abstractmethod
    def block_sqrt(self, a: Block) -> Block:
        """The elementwise square root"""
        ...

    @abstractmethod
    def block_imag(self, a: Block) -> Block:
        """The imaginary part of a complex number, elementwise."""
        ...

    @abstractmethod
    def block_exp(self, a: Block) -> Block:
        """The *elementwise* exponential. Not to be confused with :meth:`matrix_exp`, the *matrix*
        exponential."""
        ...

    @abstractmethod
    def block_log(self, a: Block) -> Block:
        """The *elementwise* natural logarithm. Not to be confused with :meth:`matrix_log`, the
        *matrix* logarithm."""
        ...

    def block_combine_legs(self, a: Block, legs_slices: list[tuple[int]]) -> Block:
        """no transpose, only reshape ``legs[b:e] for b,e in legs_slice`` to single legs"""
        old_shape = self.block_shape(a)
        new_shape = []
        last_e = 0
        for b, e in legs_slices:  # ascending!
            new_shape.extend(old_shape[last_e:b])
            new_shape.append(np.prod(old_shape[b:e]))
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
    def block_norm(self, a: Block, order: int | float = None, axis: int | None = None) -> float:
        """
        axis=None means "all axes", i.e. norm of the flattened block.
        axis: int means to broadcast the norm over all other axes.
        """
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

    def matrix_svd(self, a: Block, algorithm: str | None, compute_u: bool, compute_vh: bool
                   ) -> tuple[Block, Block, Block]:
        """SVD of a 2D block.

        With full_matrices=False, i.e. shape ``(n,m) -> (n,k), (k,) (k,m)`` where
        ``k = min(n,m)``.
        
        Assumes that U and Vh have the same dtype as a, while S has a matching real dtype.
        """
        if algorithm == 'eigh':
            return self.matrix_eig_based_svd(a, compute_u=compute_u, compute_vh=compute_vh)
        return self._matrix_svd(a, algorithm)

    @abstractmethod
    def _matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """Internal version of :meth:`matrix_svd`, to be implemented by subclasses."""
        ...

    def matrix_eig_based_svd(self, a: Block, compute_u: bool, compute_vh: bool
                             ) -> tuple[Block, Block, Block]:
        """Eig-based SVD of a 2D block.

        With full_matrices=False, i.e. shape ``(n,m) -> (n,k), (k,) (k,m)`` where
        ``k = min(n,m)``.
        
        Assumes that U and Vh have the same dtype as a, while S has a matching real dtype.
        """
        # TODO should we actually contract the full square a.hc @ a or can we work with the
        #      factored form?
        #      consider discussion in https://www.math.wsu.edu/math/faculty/watkins/pdfiles/1-44311.pdf
        m, n = self.block_shape(a)
        k = min(m, n)
        if compute_u and compute_vh:
            raise ValueError('Can not compute both U and Vh.')
        if (not compute_u) and (not compute_vh):
            if m > n:  # a.hc @ a is n x n, thus its cheaper to compute its eigenvalues
                square = self.block_tdot(self.block_conj(a), a, [0], [0])
            else:
                square = self.block_tdot(a, self.block_conj(a), [1], [1])
            S_sq = self.block_eigvalsh(square)
            U = Vh = None
        if compute_u:  # decompose a @ a.hc = U @ S**2 @ U.hc
            a_ahc = self.block_tdot(a, self.block_conj(a), [1], [1])
            S_sq, U = self.block_eigh(a_ahc, sort='>')
            U = U[:, :k]
            Vh = None
        else:  # decompose a.hc @ a = V @ S**2 @ V.hc  (note that we want V.hc !)
            ahc_a = self.block_tdot(self.block_conj(a), a, [0], [0])
            S_sq, V = self.block_eigh(ahc_a, sort='>')
            Vh = self.block_permute_axes(self.block_conj(V), [1, 0])[:k, :]
            U = None
        # economic SVD: only k=min(m, n) singular values
        S = self.block_sqrt(abs(S_sq[:k]))
        return U, S, Vh

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
        """The kronecker product.

        Parameters
        ----------
        a, b
            Twp blocks with the same number of dimensions.

        Notes
        -----
        The elements are products of elements from `a` and `b`::
            kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

        where::
            kt = it * st + jt,  t = 0,...,N

        (Taken from numpy docs)
        """
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
        """Return a (M, N) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
        where N is the number of True entries. The result is the coefficient matrix of the projection map."""
        ...

    def block_scale_axis(self, block: Block, factors: Block, axis: int) -> Block:
        """multiply block with the factors (a 1D block), along a given axis.
        E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
        ``block * factors[None, None, :, None]``.
        """
        idx = [None] * len(self.block_shape(block))
        idx[axis] = slice(None, None,  None)
        return block * factors[tuple(idx)]

    @abstractmethod
    def block_sum_all(self, a: Block) -> float | complex:
        """The sum of all entries of the block.
        If the block contains boolean values, this should return the number of ``True`` entries.
        """
        ...

    def apply_mask_to_block(self, block: Block, mask: Block, ax: int) -> Block:
        """Apply a mask (1D boolean block) to a block, slicing/projecting that axis"""
        idx = (slice(None, None, None),) * (ax - 1) + (mask,)
        return block[idx]

    @abstractmethod
    def block_eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        """Eigenvalue decomposition of a 2D hermitian block.

        Return a 1D block of eigenvalues and a 2D block of eigenvectors
        
        Parameters
        ----------
        block : Block
            The block to decompose
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted
        """
        ...

    @abstractmethod
    def block_eigvalsh(self, block: Block, sort: str = None) -> Block:
        """Eigenvalues of a 2D hermitian block.

        Return a 1D block of eigenvalues
        
        Parameters
        ----------
        block : Block
            The block to decompose
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted
        """
        ...

    @abstractmethod
    def block_abs_argmax(self, block: Block) -> list[int]:
        """Return the indices (one per axis) of the largest entry (by magnitude) of the block"""
        ...

    def synchronize(self):
        """Wait for asynchronous processes (if any) to finish"""
        pass

    def apply_leg_permutations(self, block: Block, perms: list[np.ndarray]) -> Block:
        """Apply permutations to every axis of a dense block"""
        return block[np.ix_(*perms)]

    def apply_basis_perm(self, block: Block, legs: list[VectorSpace], inv: bool = False) -> Block:
        """Apply basis_perm of a VectorSpace (or its inverse) on every axis of a dense block"""
        # OPTIMIZE should we special-case None for "no permutation to do"?
        if inv:
            perms = [leg._inverse_basis_perm for leg in legs]
        else:
            perms = [leg.basis_perm for leg in legs]
        return self.apply_leg_permutations(block, perms)

    def block_argsort(self, block: Block, sort: str = None, axis: int = 0) -> Block:
        """Return the permutation that would sort a block along one axis.

        Parameters
        ----------
        block : Block
            The block to sort.
        sort : str
            Specify how the arguments should be sorted.

            ==================== =============================
            `sort`               order
            ==================== =============================
            ``'m>', 'LM'``       Largest magnitude first
            -------------------- -----------------------------
            ``'m<', 'SM'``       Smallest magnitude first
            -------------------- -----------------------------
            ``'>', 'LR', 'LA'``  Largest real part first
            -------------------- -----------------------------
            ``'<', 'SR', 'SA'``  Smallest real part first
            -------------------- -----------------------------
            ``'LI'``             Largest imaginary part first
            -------------------- -----------------------------
            ``'SI'``             Smallest imaginary part first
            ==================== =============================

        axis : int
            The axis along which to sort

        Returns
        -------
        1D block of int
            The indices that would sort the block
        """
        if sort == 'm<' or sort == 'SM':
            block = np.abs(block)
        elif sort == 'm>' or sort == 'LM':
            block = -np.abs(block)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            block = np.real(block)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            block = -np.real(block)
        elif sort == 'SI':
            block = np.imag(block)
        elif sort == 'LI':
            block = -np.imag(block)
        else:
            raise ValueError("unknown sort option " + repr(sort))
        return self._block_argsort(block, axis=axis)

    @abstractmethod
    def _block_argsort(self, block: Block, axis: int) -> Block:
        """Like :meth:`block_argsort` but can assume real valued block, and sort ascending"""
        ...


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
