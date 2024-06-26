"""TODO
Also contains some private utility function used by multiple backend modules.

"""

# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TypeVar, TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np

from ..symmetries import Symmetry
from ..spaces import Space, ElementarySpace, ProductSpace
from ..dtypes import Dtype

__all__ = ['Data', 'DiagonalData', 'MaskData', 'Block', 'Backend', 'BlockBackend',
           'conventional_leg_order', 'iter_common_sorted', 'iter_common_sorted_arrays',
           'iter_common_nonstrict_sorted_arrays', 'iter_common_noncommon_sorted_arrays']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')
MaskData = TypeVar('MaskData')

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

    def __repr__(self):
        return f'{type(self).__name__}'

    def __str__(self):
        return f'{type(self).__name__}'

    def item(self, a: SymmetricTensor | DiagonalTensor) -> float | complex:
        """Assumes that tensor is a scalar (i.e. has only one entry).
        Returns that scalar as python float or complex"""
        return self.data_item(a.data)
    
    def test_data_sanity(self, a: SymmetricTensor | DiagonalTensor, is_diagonal: bool):
        # subclasses will typically call super().test_data_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def test_leg_sanity(self, leg: Space):
        # subclasses will typically call super().test_leg_sanity(a)
        assert isinstance(leg, Space)
        leg.test_sanity()

    def test_mask_sanity(self, a: Mask):
        # subclasses will typically call super().test_mask_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    # ABSTRACT METHODS
    
    @abstractmethod
    def act_block_diagonal_square_matrix(self, a: SymmetricTensor, block_method: Callable[[Block], Block]
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
    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: ProductSpace, new_domain: ProductSpace
                        ) -> Data:
        ...

    @abstractmethod
    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        ...

    @abstractmethod
    def combine_legs(self, a: SymmetricTensor, combine_slices: list[int, int],
                     product_spaces: list[ProductSpace], new_axes: list[int],
                     final_legs: list[Space]) -> Data:
        """combine legs of `a` (without transpose).

        ``combine_slices[i]=(begin, end)`` sorted in ascending order of `begin` indicates that
        ``a.legs[begin:end]`` is to be combined to `product_spaces[i]`, yielding `final_legs`.
        `new_axes[i]` is the index of `product_spaces[i]` in `final_legs` (also fixed by `combine_slices`).
        """
        ...

    @abstractmethod
    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Assumes ``a.domain == b.codomain`` and performs map composition,
        i.e. tensor contraction over those shared legs.

        Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
        not both empty.
        """
        ...

    @abstractmethod
    def copy_data(self, a: SymmetricTensor | DiagonalTensor | MaskData
                  ) -> Data | DiagonalData | MaskData:
        """Return a copy, such that future in-place operations on the output data do not affect the input data"""
        ...

    @abstractmethod
    def dagger(self, a: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def data_item(self, a: Data | DiagonalData | MaskData) -> float | complex:
        """Assumes that data is a scalar (as defined in tensors.is_scalar).
        
        Return that scalar as python float or complex
        """
        ...

    @abstractmethod
    def diagonal_all(self, a: DiagonalTensor) -> bool:
        """Assumes a boolean DiagonalTensor. If all entries are True."""
        ...

    @abstractmethod
    def diagonal_any(self, a: DiagonalTensor) -> bool:
        """Assumes a boolean DiagonalTensor. If any entry is True."""
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
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
        ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
        """
        ...

    @abstractmethod
    def diagonal_from_block(self, a: Block, co_domain: ProductSpace, tol: float) -> DiagonalData:
        """DiagonalData from a 1D block in *internal* basis order."""
        ...

    @abstractmethod
    def diagonal_from_sector_block_func(self, func, co_domain: ProductSpace) -> DiagonalData:
        """Generate diagonal data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        ...
       
    @abstractmethod
    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                       ) -> DiagonalData:
        """Get the DiagonalData corresponding to a tensor with two legs.

        Can assume that domain and codomain consist of the same single leg.
        """
        ...

    @abstractmethod
    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        ...

    @abstractmethod
    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        """Forget about symmetry structure and convert to a single 1D block.
        
        This is the diagonal of the respective non-symmetric 2D tensor.
        In the *internal* basis order of the leg.
        """
        ...

    @abstractmethod
    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[MaskData, ElementarySpace]:
        """Convert a DiagonalTensor to a Mask.

        May assume that dtype is bool.
        Returns ``mask_data, small_leg``.
        """
        ...

    @abstractmethod
    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        """Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``"""
        ...

    @abstractmethod
    def eigh(self, a: SymmetricTensor, sort: str = None) -> tuple[DiagonalData, Data]:
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
    def eye_data(self, co_domain: ProductSpace, dtype: Dtype) -> Data:
        """Data for :meth:``SymmetricTensor.eye``.

        The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
        """
        ...

    @abstractmethod
    def flip_leg_duality(self, tensor: SymmetricTensor, which_legs: list[int],
                         flipped_legs: list[Space], perms: list[np.ndarray]) -> Data:
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, codomain: ProductSpace, domain: ProductSpace, tol: float
                         ) -> Data:
        """Convert a dense block to the data for a symmetric tensor.

        Block is in the *internal* basis order of the respective legs and the leg order is
        ``[*codomain, *reversed(domain)]``.
        
        If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
        where ``projected`` is `a` projected to the space of symmetric tensors, raise a ``ValueError``.
        """
        ...

    @abstractmethod
    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        """Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.

        Is given in the *internal* basis order.
        """
        ...

    @abstractmethod
    def from_random_normal(self, codomain: ProductSpace, domain: ProductSpace, sigma: float,
                           dtype: Dtype) -> Data:
        ...

    @abstractmethod
    def from_sector_block_func(self, func, codomain: ProductSpace, domain: ProductSpace) -> Data:
        """Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``."""
        ...

    @abstractmethod
    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        ...

    @abstractmethod
    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        """May assume that the mask is a projection."""
        ...

    @abstractmethod
    def get_dtype_from_data(self, a: Data) -> Dtype:
        ...

    @abstractmethod
    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
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
    def infer_leg(self, block: Block, legs: list[Space | None], is_dual: bool = False,
                  is_real: bool = False) -> ElementarySpace:
        """Infer a missing leg from the dense block"""
        # TODO make it poss
        ...

    @abstractmethod
    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        """tensors.inner on SymmetricTensors"""
        ...

    @abstractmethod
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                               charge_leg: ElementarySpace) -> Data:
        """Data for the invariant part used in ChargedTensor.from_dense_block_single_sector

        The vector is given in the *internal* basis order of `spaces`.
        """
        ...

    @abstractmethod
    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        """Inverse of inv_part_from_dense_block_single_sector

        In the *internal* basis order of `spaces`.
        """
        ...

    @abstractmethod
    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[MaskData, ElementarySpace]:
        """Elementwise binary function acting on two masks.

        May assume that both masks are a projection (from large to small leg)
        and that the large legs match.
        returns ``mask_data, new_small_leg``
        """
        ...

    @abstractmethod
    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        """Implementation of :func:`tenpy.linalg.tensors._compose_with_Mask` in the case where
        the large leg of the mask is contracted.
        Note that the mask may be a projection to be applied to the codomain or an inclusion
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, ProductSpace, ProductSpace]:
        """Implementation of :func:`tenpy.linalg.tensors._compose_with_Mask` in the case where
        the small leg of the mask is contracted.
        Note that the mask may be an inclusion to be applied to the codomain or a projection
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_dagger(self, mask: Mask) -> MaskData:
        ...

    @abstractmethod
    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        """Data for a *projection* Mask, and the resulting small leg, from a 1D block.

        a: 1D block, the Mask in *internal* basis order of `large_leg`.
        """
        ...

    @abstractmethod
    def mask_to_block(self, a: Mask) -> Block:
        """As a block of the large_leg, in *internal* basis order."""
        ...

    @abstractmethod
    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> MaskData:
        ...

    @abstractmethod
    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        """Transpose a mask. Also return the new ``space_in`` and ``space_out``.

        Those spaces are the duals of the respective other in the old mask.
        """
        ...

    @abstractmethod
    def mask_unary_operand(self, mask: Mask, func) -> tuple[MaskData, ElementarySpace]:
        """Elementwise function acting on a mask.

        May assume that mask is a projection (from large to small leg).
        Returns ``mask_data, new_small_leg``
        """
        ...
        
    @abstractmethod
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        """Norm of a tensor. order has already been parsed and is a number"""
        ...

    @abstractmethod
    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, ProductSpace, ProductSpace]:
        """Perform an arbitrary number of traces. Pairs are converted to leg idcs.
        Returns ``data, codomain, domain``.
        """
        ...

    @abstractmethod
    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, ProductSpace, ProductSpace]:
        """Permute legs on the tensors.

        codomain_idcs, domain_idcs:
            Which of the legs should end up in the (co-)domain.
            All are leg indices (``0 <= i < a.num_legs``)
        levels:
            The levels. Can assume they are unique, support comparison and are non-negative.
            ``None`` means unspecified.

        Returns
        -------
        data:
            The data for the permuted tensor, of ``None`` if `levels` are required were not specified.
        codomain, domain
            The (co-)domain of the new tensor.
        """
        ...

    @abstractmethod
    def qr(self, a: SymmetricTensor, new_r_leg_dual: bool, full: bool) -> tuple[Data, Data, ElementarySpace]:
        """QR decomposition of a Tensor `a` with two legs.

        The legs of `a` may be :class:`~tenpy.linalg.spaces.ProductSpace`

        Returns
        -------
        q, r:
            Data of corresponding tensors.
        new_leg : ElementarySpace
            the new leg of r.
        """
        ...

    @abstractmethod
    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        """Reduce a diagonal tensor to a single number.

        Used e.g. to implement ``DiagonalTensor.max``.
        ``block_func(block: Block) -> float | complex`` realizes that reduction on blocks,
        ``func(numbers: Sequence[float | complex]) -> float | complex`` for python numbers.
        """
        ...

    @abstractmethod
    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        """Scale axis ``leg`` of ``a`` with ``b``.

        Can assume ``a.get_leg_co_domain(leg) == b.leg``.
        """
        ...

    @abstractmethod
    def set_element(self, a: SymmetricTensor, idcs: list[int], value: complex | float) -> Data:
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
    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], final_legs: list[Space]) -> Data:
        """split multiple product space legs."""
        ...

    @abstractmethod
    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        """Assume the legs at given indices are trivial and get rid of them"""
        ...

    @abstractmethod
    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        ...

    @abstractmethod
    def svd(self, a: SymmetricTensor, new_vh_leg_dual: bool, algorithm: str | None, compute_u: bool,
            compute_vh: bool) -> tuple[Data, DiagonalData, Data, ElementarySpace]:
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
            ElementarySpace the new leg of vh.
        """
        ...

    @abstractmethod
    def tdot(self, a: SymmetricTensor, b: SymmetricTensor, axs_a: list[int], axs_b: list[int]) -> Data:
        """Tensordot i.e. pairwise contraction"""
        ...

    @abstractmethod
    def state_tensor_product(self, state1: Block, state2: Block, prod_space: ProductSpace):
        """TODO clearly define what this should do in tensors.py first!

        In particular regarding basis orders.
        """
        ...

    @abstractmethod
    def to_dense_block(self, a: SymmetricTensor) -> Block:
        """Forget about symmetry structure and convert to a single block.
        
        Return a block in the *internal* basis order of the respective legs,
        with leg order ``[*codomain, *reversed(domain)]``.
        """
        ...

    @abstractmethod
    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        """Single-leg tensor to the *part of* the coefficients in the trivial sector.

        In *internal* basis order.
        """
        ...

    @abstractmethod
    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        """cast to given dtype. No copy if already has dtype."""
        ...

    @abstractmethod
    def trace_full(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        ...

    @abstractmethod
    def transpose(self, a: SymmetricTensor) -> tuple[Data, ProductSpace, ProductSpace]:
        """Returns ``data, new_codomain, new_domain``.
        Note that ``new_codomain == a.domain.dual`` and ``new_domain == a.codomain.dual``.
        """
        ...

    @abstractmethod
    def zero_data(self, codomain: ProductSpace, domain: ProductSpace, dtype: Dtype) -> Data:
        """Data for a zero tensor"""
        ...

    @abstractmethod
    def zero_diagonal_data(self, co_domain: ProductSpace, dtype: Dtype) -> DiagonalData:
        ...

    @abstractmethod
    def zero_mask_data(self, large_leg: Space) -> MaskData:
        ...

    # OPTIONALLY OVERRIDE THESE

    def _fuse_spaces(self, symmetry: Symmetry, spaces: list[Space]):
        """Backends may override the behavior of linalg.spaces._fuse_spaces in order to compute
        their backend-specific metadata alongside the sectors.
        """
        raise NotImplementedError

    def get_leg_metadata(self, leg: Space) -> dict:
        """Get just the metadata returned by :meth:`_fuse_spaces`, without the sectors."""
        return {}

    def is_real(self, a: SymmetricTensor) -> bool:
        """If the Tensor is comprised of real numbers.
        Complex numbers with small or zero imaginary part still cause a `False` return."""
        # FusionTree backend might implement this differently.
        return a.dtype.is_real


class BlockBackend(metaclass=ABCMeta):
    """Abstract base class that defines the operation on dense blocks."""
    svd_algorithms: list[str]  # first is default
    BlockCls = None  # to be set by subclass

    @abstractmethod
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False
                 ) -> Block | tuple[Block, Dtype]:
        """Convert objects to blocks.

        Should support blocks, numpy arrays, nested python containers. May support more.
        Convert to `dtype`, if given.

        TODO make sure to emit warning on complex -> float!

        Returns
        -------
        block: Block
            The new block
        dtype: Dtype, optional
            The new dtype of the block. Only returned if `return_dtype`.
        """
        ...

    @abstractmethod
    def block_all(self, a) -> bool:
        """Require a boolean block. If all of its entries are True"""
        ...
        
    @abstractmethod
    def block_any(self, a) -> bool:
        """Require a boolean block. If any of its entries are True"""
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

    def block_tensor_outer(self, a: Block, b: Block, K: int) -> Block:
        """Version of ``tensors.outer`` on blocks.

        Note the different leg order to usual outer products::

            res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]

        intended to be used with ``K == a_num_codomain_legs``.
        """
        res = self.block_outer(a, b)  # [i1,...,iN,j1,...,jM]
        N = len(self.block_shape(a))
        M = len(self.block_shape(b))
        return self.block_permute_axes(res, [*range(K), *range(N, N + M), *range(K, N)])

    @abstractmethod
    def block_shape(self, a: Block) -> tuple[int]:
        ...

    @abstractmethod
    def block_item(self, a: Block) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    def block_dagger(self, a: Block) -> Block:
        """Permute axes to reverse order and elementwise conj."""
        num_legs = len(self.block_shape(a))
        res = self.block_permute_axes(a, list(reversed(range(num_legs))))
        return self.block_conj(res)

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
        """Outer product of blocks.

        ``res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM]``
        """
        ...

    def block_inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        """Dense block version of tensors.inner.

        If do dagger, ``sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])``
        otherwise, ``sum(a[i1, ..., iN] * b[iN, ..., i2, i1])``.
        """
        if do_dagger:
            a = self.block_conj(a)
        else:
            a = self.block_permute_axes(a, list(reversed(range(a.ndim))))
        return self.block_sum_all(a * b)  # TODO or do tensordot?

    @abstractmethod
    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        ...

    @abstractmethod
    def block_trace_full(self, a: Block) -> float | complex:
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
    def block_allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        ...

    @abstractmethod
    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        # TODO rename to squeeze_axes ?
        ...

    @abstractmethod
    def block_add_axis(self, a: Block, pos: int) -> Block:
        ...

    @abstractmethod
    def block_norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        r"""The p-norm vector-norm of a block.

        Parameters
        ----------
        order : float
            The order :math:`p` of the norm.
            Unlike numpy, we always compute vector norms, never matrix norms.
            We only support p-norms :math:`\Vert x \Vert = \sqrt[p]{\sum_i \abs{x_i}^p}`.
        axis : int | None
            ``axis=None`` means "all axes", i.e. norm of the flattened block.
            An integer means to broadcast the norm over all other axes.
        """
        ...

    @abstractmethod
    def block_max(self, a: Block) -> float:
        ...

    @abstractmethod
    def block_max_abs(self, a: Block) -> float:
        ...

    @abstractmethod
    def block_min(self, a: Block) -> float:
        ...
        
    @abstractmethod
    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        ...

    @abstractmethod
    def matrix_dot(self, a: Block, b: Block) -> Block:
        """As in numpy.dot, both a and b might be matrix or vector."""
        # TODO can probably remove this? was only used in an old version of tdot.
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

    def block_linear_combination(self, a, v: Block, b, w: Block) -> Block:
        return a * v + b * w

    def block_mul(self, a: float | complex, b: Block) -> Block:
        return a * b

    @abstractmethod
    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def ones_block(self, shape: list[int], dtype: Dtype) -> Block:
        ...

    @abstractmethod
    def eye_matrix(self, dim: int, dtype: Dtype) -> Block:
        """The ``dim x dim`` identity matrix"""
        ...

    def eye_block(self, legs: list[int], dtype: Dtype) -> Data:
        """The identity matrix, reshaped to a block.

        Note the unusual leg order ``[m1,...,mJ,mJ*,...,m1*]``,
        which is chosen to match :meth:`eye_data`.

        Note also that the ``legs`` only specify the dimensions of the first half,
        namely ``m1,...,mJ``.
        """
        J = len(legs)
        eye = self.eye_matrix(prod(legs), dtype)
        # [M, M*] -> [m1,...,mJ,m1*,...,mJ*]
        eye = self.block_reshape(eye, legs * 2)
        # [m1,...,mJ,mJ*,...,m1*]
        return self.block_permute_axes(eye, [*range(J), *reversed(range(J, 2 * J))])

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
        """Return a (N, M) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
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
    def block_sum(self, a: Block, ax: int) -> Block:
        """The sum over a single axis."""
        ...

    @abstractmethod
    def block_sum_all(self, a: Block) -> float | complex:
        """The sum of all entries of the block.
        If the block contains boolean values, this should return the number of ``True`` entries.
        """
        ...

    def block_apply_mask(self, block: Block, mask: Block, ax: int) -> Block:
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

    def apply_basis_perm(self, block: Block, legs: list[Space], inv: bool = False) -> Block:
        """Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block"""
        perms = []
        for leg in legs:
            p = leg._inverse_basis_perm if inv else leg._basis_perm
            if p is None:
                # OPTIMIZE support None in apply_leg_permutations, to skip permuting that leg? 
                p = np.arange(leg.dim)
            perms.append(p)
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

    def test_block_sanity(self, block, expect_shape: tuple[int, ...] | None = None,
                          expect_dtype: Dtype | None = None):
        assert isinstance(block, self.BlockCls), 'wrong block type'
        if expect_shape is not None:
            assert self.block_shape(block) == expect_shape, 'wrong block shape'
        if expect_dtype is not None:
            assert self.block_dtype(block) == expect_dtype, 'wrong block dtype'

    def block_enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(self.block_shape(block))
        shape[axis] = self.block_shape(mask)[0]
        res = self.zero_block(shape, dtype=self.block_dtype(block))
        idcs = (slice(None, None, None),) * axis + (mask,)
        res[idcs] = block  # TODO mutability?
        return res

    @abstractmethod
    def block_stable_log(self, block: Block, cutoff: float) -> Block:
        """Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0."""
        ...


def conventional_leg_order(tensor_or_codomain: SymmetricTensor | ProductSpace,
                           domain: ProductSpace = None) -> Iterator[Space]:
    if domain is None:
        codomain = tensor_or_codomain.codomain
        domain = tensor_or_codomain.domain
    else:
        codomain = tensor_or_codomain
    yield from codomain.spaces
    yield from reversed(domain.spaces)


def iter_common_sorted(a, b):
    """Yield indices ``i, j`` for which ``a[i] == b[j]``.

    *Assumes* that `a` and `b` are strictly ascending 1D arrays.
    Given that, it is equivalent to (but faster than)
    ``[(i, j) for j, i in itertools.product(range(len(b)), range(len(a)) if a[i] == b[j]]``
    """
    # when we call this function, we basically wanted iter_common_sorted_arrays,
    # but used strides to merge multiple columns to avoid too much python loops
    # for C-implementation, this is definitely no longer necessary.
    l_a = len(a)
    l_b = len(b)
    i, j = 0, 0
    while i < l_a and j < l_b:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            yield i, j
            i += 1
            j += 1


def iter_common_sorted_arrays(a, b):
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


def iter_common_nonstrict_sorted_arrays(a, b):
    """Yield indices ``i, j`` for which ``a[i, :] == b[j, :]``.

    Like iter_common_sorted_arrays, but allows duplicate rows in `a`.
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
            # difference to iter_common_sorted_arrays:
            # dont increase j because a[i + 1] might also match b[j]
            i += 1


def iter_common_noncommon_sorted_arrays(a, b):
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
