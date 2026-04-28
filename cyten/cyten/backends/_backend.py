"""TODO summary

Also contains some private utility function used by multiple backend modules.
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from ..block_backends import Block, BlockBackend
from ..block_backends.dtypes import Dtype
from ..symmetries import ElementarySpace, FusionTree, Leg, LegPipe, Space, Symmetry, TensorProduct
from ..tools.misc import combine_constraints

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import DiagonalTensor, Mask, SymmetricTensor

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')
MaskData = TypeVar('MaskData')


class TensorBackend(metaclass=ABCMeta):
    """Abstract base class for tensor-backends.

    A backends implements functions that act on tensors.
    We abstract two separate concepts for a backend.
    There is a block backend, that abstracts what the numerical data format (numpy array,
    torch Tensor, CUDA tensor, ...) is and a tensor-backend that abstracts how block-sparse
    structures that arise from symmetries are accounted for.

    A tensor backend has a the :attr:`block_backend` as an attribute and can call its functions
    to operate on blocks. This allows the tensor backend to be agnostic of the details of these
    blocks.
    """

    DataCls = None  # to be set by subclasses

    can_decompose_tensors = False
    """If the decompositions (SVD, QR, EIGH, ...) can operate on many-leg tensors,
    or require legs to be combined first."""

    def __init__(self, block_backend: BlockBackend):
        self.block_backend = block_backend

    def __repr__(self):
        return f'{type(self).__name__}({self.block_backend!r})'

    def __str__(self):
        return f'{type(self).__name__}({self.block_backend!r})'

    def item(self, a: SymmetricTensor | DiagonalTensor) -> float | complex:
        """Convert tensor to a python scalar.

        Assumes that tensor is a scalar (i.e. has only one entry).
        """
        return self.data_item(a.data)

    def test_tensor_sanity(self, a: SymmetricTensor | DiagonalTensor, is_diagonal: bool):
        """Called as part of :meth:`cyten.Tensor.test_sanity`.

        Perform sanity checks on the ``a.data``, and possibly additional backend-specific checks
        of the tensor.
        """
        # subclasses will typically call super().test_tensor_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def test_mask_sanity(self, a: Mask):
        # subclasses will typically call super().test_mask_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def make_pipe(self, legs: list[Leg], is_dual: bool, in_domain: bool, pipe: LegPipe | None = None) -> LegPipe:
        """Make a pipe *of the appropriate type* for :meth:`combine_legs`.

        If `pipe` is given, try to return it if suitable.
        """
        if isinstance(pipe, LegPipe):
            assert pipe.combine_cstyle == (not is_dual)
            assert pipe.is_dual == is_dual
            assert pipe.legs == legs
            return pipe
        return LegPipe(legs, is_dual=is_dual, combine_cstyle=not is_dual)

    # ABSTRACT METHODS

    @abstractmethod
    def act_block_diagonal_square_matrix(
        self, a: SymmetricTensor, block_method: Callable[[Block], Block], dtype_map: Callable[[Dtype], Dtype] | None
    ) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`.

        Assumes the block_method returns blocks on the same device.

        Parameters
        ----------
        a : Tensor
            The tensor to act on. Can assume ``a.codomain == a.domain``.
        block_method : function
            A function with signature ``block_method(a: Block) -> Block`` acting on backend-blocks.
        dtype_map : function or None
            Specify how the result dtype depends on the input dtype. ``None`` means unchanged.
            This is needed in abelian and fusion-tree backends, in case there are 0 blocks.

        """
        ...

    @abstractmethod
    def add_trivial_leg(
        self,
        a: SymmetricTensor,
        legs_pos: int,
        add_to_domain: bool,
        co_domain_pos: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data: ...

    @abstractmethod
    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool: ...

    @abstractmethod
    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData: ...

    @abstractmethod
    def combine_legs(
        self,
        tensor: SymmetricTensor,
        leg_idcs_combine: list[list[int]],
        pipes: list[LegPipe],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        """Implementation of :func:`cyten.tensors.combine_legs`.

        Assumptions:

        - Legs have been permuted, such that each group of legs to be combined appears contiguously
          and either entirely in the codomain or entirely in the domain

        Parameters
        ----------
        tensor: SymmetricTensor
            The tensor to modify
        leg_idcs_combine: list of list of int
            A list of groups. Each group a list of integer leg indices, to be combined.
        pipes: list of LegPipe
            The resulting pipes. Same length and order as `leg_idcs_combine`.
            In the domain, this is the product space as it will appear in the domain, not in legs.
        new_codomain_combine:
            A list of tuples ``(positions, combined)``, where positions are all the codomain-indices
            which should be combined and ``combined`` is the resulting :class:`LegPipe`,
            i.e. ``combined == LegPipe([tensor.codomain[n] for n in positions])``
        new_domain_combine:
            Similar as `new_codomain_combine` but for the domain. Note that ``positions`` are
            domain-indices, i.e ``n = positions[i]`` refers to ``tensor.domain[n]``, *not*
            ``tensor.legs[n]`` !
        new_codomain, new_domain: TensorProduct
            The codomain and domain of the resulting tensor

        """
        ...

    @abstractmethod
    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Assumes ``a.domain == b.codomain`` and performs contraction over those legs.

        Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
        not both empty. Assumes both input tensors are on the same device.
        """
        ...

    @abstractmethod
    def copy_data(
        self, a: SymmetricTensor | DiagonalTensor | MaskData, device: str = None
    ) -> Data | DiagonalData | MaskData:
        """Return a copy.

        The main requirement is that future in-place operations on the output data do not affect
        the input data

        Parameters
        ----------
        a : Tensor
            The tensor to copy
        device : str, optional
            The device for the result. Per default (or if ``None``), use the same device as `a`.

        See Also
        --------
        move_to_device

        """
        ...

    @abstractmethod
    def dagger(self, a: SymmetricTensor) -> Data: ...

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
    def diagonal_elementwise_binary(
        self, a: DiagonalTensor, b: DiagonalTensor, func, func_kwargs, partial_zero_is_zero: bool
    ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
        pairs of elements.
        Input tensors are both DiagonalTensor and have equal legs.
        ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
        and similarly for the second argument.

        Assumes both tensors are on the same device.
        """
        ...

    @abstractmethod
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
        ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
        """
        ...

    @abstractmethod
    def diagonal_from_block(self, a: Block, co_domain: TensorProduct, tol: float) -> DiagonalData:
        """The DiagonalData from a 1D block in *internal* basis order."""
        ...

    @abstractmethod
    def diagonal_from_sector_block_func(self, func, co_domain: TensorProduct) -> DiagonalData:
        """Generate diagonal data from a function.

        Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
        Assumes all generated blocks are on the same device.
        """
        ...

    @abstractmethod
    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, tol: float | None = 1e-12) -> DiagonalData:
        """Get the DiagonalData corresponding to a tensor with two legs.

        Can assume that domain and codomain consist of the same single leg.
        """
        ...

    @abstractmethod
    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex: ...

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
    def eigh(
        self, a: SymmetricTensor, new_leg_dual: bool, sort: str = None
    ) -> tuple[DiagonalData, Data, ElementarySpace]:
        """Eigenvalue decomposition of a hermitian tensor

        Note that this does *not* guarantee to return the duality given by `new_leg_dual`.
        In particular, for the abelian backend, the duality is fixed.

        Parameters
        ----------
        a
            The input tensor. Assumed to be hermitian without checking!
        new_leg_dual : bool
            If the new leg should be dual or not.
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted *within* each charge block.
            See :func:`argsort` for details.

        Returns
        -------
        w_data
            Data for the :class:`DiagonalTensor` of eigenvalues
        v_data
            Data for the :class:`Tensor` of eigenvectors
        new_leg
            The new leg.

        """
        ...

    @abstractmethod
    def eye_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> Data:
        """Data for :meth:``SymmetricTensor.eye``.

        The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
        """
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, codomain: TensorProduct, domain: TensorProduct, tol: float) -> Data:
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
        """Data from a grid of tensors.

        Parameters
        ----------
        grid: list[list[SymmetricTensor | None]]
            Contains the tensors from which a single tensor is constructed. `None` entries are
            interpreted as tensors with all blocks equal to zero.
        new_codomain: TensorProduct
            Codomain of the resulting tensor after stacking the tensors in the grid.
        new_domain: TensorProduct
            Domain of the resulting tensor after stacking the tensors in the grid.
        left_mult_slices: list[list[int]]
            Multiplicity slices for each sector for the stacking in the codomain. That is,
            ``slice(left_mult_slices[sector_idx][i], left_mult_slices[sector_idx][i + 1])`` is the
            slice that is contributed from the tensors in the `i`th column to the sector
            ``new_codomain[0].sector_decomposition[sector_idx]`` of the leg ``new_codomain[0]``.
        right_mult_slices: list[list[int]]
            Multiplicity slices for each sector for the stacking in the domain. That is,
            ``slice(right_mult_slices[sector_idx][i], right_mult_slices[sector_idx][i + 1])`` is
            the slice that is contributed from the tensors in the `i`th row to the sector
            ``new_domain[-1].sector_decomposition[sector_idx]`` of the leg ``new_domain[-1]``.
        dtype: Dtype
            The new dtype of the block.
        device: str
            The device for the block.

        """
        ...

    @abstractmethod
    def from_random_normal(
        self, codomain: TensorProduct, domain: TensorProduct, sigma: float, dtype: Dtype, device: str
    ) -> Data: ...

    @abstractmethod
    def from_sector_block_func(self, func, codomain: TensorProduct, domain: TensorProduct) -> Data:
        """Generate tensor data from a function-

        Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
        Assumes all generated blocks are on the same device.
        """
        ...

    @abstractmethod
    def from_tree_pairs(
        self,
        trees: dict[tuple[FusionTree, FusionTree], Block],
        codomain: TensorProduct,
        domain: TensorProduct,
        dtype: Dtype,
        device: str,
    ) -> Data:
        """Compute the data for :meth:`SymmetricTensor.from_tree_pairs`."""
        ...

    @abstractmethod
    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data: ...

    @abstractmethod
    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        """May assume that the mask is a projection."""
        ...

    @abstractmethod
    def get_device_from_data(self, a: Data) -> str:
        """Extract the device from the data object"""
        ...

    @abstractmethod
    def get_dtype_from_data(self, a: Data) -> Dtype: ...

    @abstractmethod
    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        """Get a single scalar element from a tensor.

        Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs
            - 0 <= idx < leg.dim

        """
        ...

    @abstractmethod
    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        """Get a single scalar element from a diagonal tensor.

        Should be equivalent to ``a.to_numpy()[idx, idx].item()`` or ``a.diagonal_as_numpy()[idx].item()``.

        Parameters
        ----------
        idx
            The index for both legs. Checks have already been performed, i.e. we may assume that
            ``0 <= idx < leg.dim``

        """
        ...

    @abstractmethod
    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        """Get a single scalar element from a diagonal tensor.

        Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs == 2
            - 0 <= idx < leg.dim

        """
        ...

    @abstractmethod
    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        """tensors.inner on SymmetricTensors"""
        ...

    @abstractmethod
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space, charge_leg: ElementarySpace) -> Data:
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
        """Form the linear combinations ``a * v + b * w``.

        Assumes `v` and `w` are on the same device.
        """
        ...

    @abstractmethod
    def lq(self, tensor: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]: ...

    @abstractmethod
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[MaskData, ElementarySpace]:
        """Elementwise binary function acting on two masks.

        May assume that both masks are a projection (from large to small leg)
        and that the large legs match.

        Assumes that `mask1` and `mask2` are on the same device.

        returns ``mask_data, new_small_leg``
        """
        ...

    @abstractmethod
    def mask_contract_large_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        """Contraction with the large leg of a Mask.

        Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
        the large leg of the mask is contracted.
        Note that the mask may be a projection to be applied to the codomain or an inclusion
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_contract_small_leg(
        self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        """Contraction with the small leg of a Mask.

        Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
        the small leg of the mask is contracted.
        Note that the mask may be an inclusion to be applied to the codomain or a projection
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_dagger(self, mask: Mask) -> MaskData: ...

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
    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> MaskData: ...

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
    def move_to_device(self, a: SymmetricTensor | DiagonalTensor | Mask, device: str) -> Data:
        """Move tensor to a given device.

        The result is *not* guaranteed to be a copy. In particular, if `a` already is on the
        target device, it is returned without modification.

        See Also
        --------
        copy_data

        """

    @abstractmethod
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data: ...

    @abstractmethod
    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        """Norm of a tensor. order has already been parsed and is a number"""
        ...

    @abstractmethod
    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Form the outer product, or tensor product of maps.

        Assumes that `a` and `b` are on the same device.
        """
        ...

    @abstractmethod
    def partial_compose(
        self,
        a: SymmetricTensor,
        b: SymmetricTensor,
        a_first_leg: int,
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        """Contract the codomain (domain) of `b` with the a part of the domain (codomain) of `a`.

        Assumes that there is at least one open leg in the domain (codomain) of the resulting
        tensor. Assumes both input tensors are on the same device.
        """
        ...

    @abstractmethod
    def partial_trace(
        self, tensor: SymmetricTensor, pairs: list[tuple[int, int]], levels: list[int] | None
    ) -> tuple[Data, TensorProduct, TensorProduct]:
        """Perform an arbitrary number of traces. Pairs are converted to leg idcs.

        Returns ``data, codomain, domain``.
        """
        ...

    @abstractmethod
    def permute_legs(
        self,
        a: SymmetricTensor,
        codomain_idcs: list[int],
        domain_idcs: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
        mixes_codomain_domain: bool,
        levels: list[int | None],
        bend_right: list[bool | None],
    ) -> Data:
        """Permute legs on the tensors.

        Parameters
        ----------
        a : SymmetricTensor
            The tensor to act on.
        codomain_idcs, domain_idcs:
            Which of the legs should end up in the (co-)domain.
            All are leg indices (``0 <= i < a.num_legs``).
        new_codomain, new_domain : TensorProduct
            The (co)domain of the result.
        mixes_codomain_domain : bool
            If any leg moves from the codomain to the domain or vv during the permutation.
        levels:
            The levels. Must support comparison with ``<`` or be ``None``, meaning unspecified.
        bend_right:
            For each leg, whether it bends to the left or right of the tensor.
            ``None`` is allowed as a placeholder, only if that leg does not bend at all.
            Note that non-bending legs do not necessarily have a ``None`` entry, however.

        Returns
        -------
        data:
            The data for the permuted tensor, or ``None`` if `levels` are required but were not
            specified.
        codomain, domain
            The (co-)domain of the new tensor.

        """
        ...

    @abstractmethod
    def qr(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        """Perform a QR decomposition.

        With ``a == Q @ R``
        ``Q.domain == a.domain``, ``Q.codomain == new_codomain``
        ``R.domain == new_codomain``, ``R.codomain == a.codomain``
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
        Assumes that `a` and `b` are on the same device.
        """
        ...

    @abstractmethod
    def split_legs(
        self,
        a: SymmetricTensor,
        leg_idcs: list[int],
        codomain_split: list[int],
        domain_split: list[int],
        new_codomain: TensorProduct,
        new_domain: TensorProduct,
    ) -> Data:
        """Split (multiple) product space legs.

        Parameters
        ----------
        a
            The tensor to split legs on.
        leg_idcs:
            List of leg-indices, fulfilling ``0 <= i < a.num_legs``, to split.
        codomain_split, domain_split
            Contains the same information as `leg_idcs`. Which legs to split is indices for the
            (co)domain.
        new_codomain, new_domain
            The new (co-)domain, after splitting. Has same sectors and multiplicities.

        """
        ...

    @abstractmethod
    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        """Assume the legs at given indices are trivial and get rid of them"""
        ...

    @abstractmethod
    def supports_symmetry(self, symmetry: Symmetry) -> bool: ...

    @abstractmethod
    def svd(
        self, a: SymmetricTensor, new_co_domain: TensorProduct, algorithm: str | None
    ) -> tuple[Data, DiagonalData, Data]: ...

    @abstractmethod
    def state_tensor_product(self, state1: Block, state2: Block, pipe: LegPipe):
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
        """Cast to given dtype. No copy if already has dtype."""
        ...

    @abstractmethod
    def trace_full(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int]) -> float | complex: ...

    @abstractmethod
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
        """Implementation of :func:`cyten.tensors.truncate_singular_values`.

        Returns
        -------
        mask_data
            Data for the mask
        new_leg : ElementarySpace
            The new leg after truncation, i.e. the small leg of the mask
        err : float
            The truncation error ``norm(S_discard) == norm(S - S_keep)``.
        new_norm
            The norm ``norm(S_keep)`` of the approximation.

        """
        ...

    def _truncate_singular_values_selection(
        self,
        S: np.ndarray,
        qdims: np.ndarray | None,
        chi_max: int | None,
        chi_min: int,
        degeneracy_tol: float,
        trunc_cut: float,
        svd_min: float,
        minimize_error: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Helper function for :meth:`truncate_singular_values`.

        Parameters
        ----------
        S_np : 1D numpy array of float
            A numpy array of singular values S[i]
        qdims : 1D numpy array of float
            A numpy array of the quantum dimensions. ``None`` means all qdims are one.
        chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error
            Constraints for truncation. See :func:`cyten.tensors.truncate_singular_values`.

        Returns
        -------
        mask : 1D numpy array of bool
            A boolean mask, indicating that ``S_np[mask]`` should be kept
        err : float
            The truncation error ``norm(S_discard) == norm(S - S_keep)``.
        new_norm
            The norm ``norm(S_keep)`` of the approximation.

        """
        # contributions ``err[i] = d[i] * S[i] ** 2`` to the error, if S[i] would be truncated.
        if qdims is None:
            marginal_errs = S**2
        else:
            marginal_errs = qdims * (S**2)

        # sort *ascending* by marginal errors (smallest first, should be truncated first)
        piv = np.argsort(marginal_errs)
        S = S[piv]
        # qdims = qdims[piv]  # not needed again.
        marginal_errs = marginal_errs[piv]

        # take safe logarithm, clipping small values to log(1e-100).
        # this is only used for degeneracy tol.
        logS = np.log(np.choose(S <= 1.0e-100, [S, 1.0e-100 * np.ones(len(S))]))

        # goal: find an index 'cut' such that we keep piv[cut:], i.e. cut between `cut-1` and `cut`.
        # build an array good, where ``good[cut] = (is `cut` an allowed choice)``.
        # we then choose the smallest good cut, i.e. we keep as many singular values as possible
        good = np.ones(len(S), dtype=bool)

        if (chi_max is not None) and (chi_max < len(S)):
            # keep at most chi_max values
            good2 = np.zeros(len(piv), dtype=np.bool_)
            good2[-chi_max:] = True
            good = combine_constraints(good, good2, 'chi_max')

        if (chi_min is not None) and (chi_min > 1):
            # keep at least chi_min values
            good2 = np.ones(len(piv), dtype=np.bool_)
            good2[-chi_min + 1 :] = False
            good = combine_constraints(good, good2, 'chi_min')

        if (degeneracy_tol is not None) and (degeneracy_tol > 0):
            # don't cut between values (cut-1, cut) with ``log(S[cut]/S[cut-1]) < deg_tol``
            # this is equivalent to
            # ``(S[cut] - S[cut-1])/S[cut-1] < exp(deg_tol) - 1 = deg_tol + O(deg_tol^2)``
            good2 = np.empty(len(piv), np.bool_)
            good2[0] = True
            good2[1:] = np.greater_equal(logS[1:] - logS[:-1], degeneracy_tol)
            good = combine_constraints(good, good2, 'degeneracy_tol')

        if svd_min is not None:
            # keep only values S[i] >= svd_min
            good2 = np.greater_equal(S, svd_min)
            good = combine_constraints(good, good2, 'svd_min')

        if trunc_cut is not None:
            good2 = np.cumsum(marginal_errs) > trunc_cut * trunc_cut
            good = combine_constraints(good, good2, 'trunc_cut')

        if minimize_error:
            cut = np.nonzero(good)[0][0]  # smallest cut for which good[cut] is True
        else:
            cut = np.nonzero(good)[0][-1]  # largest cut for which good[cut] is True
        err = np.sum(marginal_errs[:cut])
        new_norm = np.sum(marginal_errs[cut:])
        # build mask in the original order, before sorting
        mask = np.zeros(len(S), dtype=bool)
        np.put(mask, piv[cut:], True)
        return mask, err, new_norm

    @abstractmethod
    def zero_data(
        self, codomain: TensorProduct, domain: TensorProduct, dtype: Dtype, device: str, all_blocks: bool = False
    ) -> Data:
        """Data for a zero tensor.

        Parameters
        ----------
        all_blocks: bool
            Some specific backends can omit zero blocks ("sparsity").
            By default (``False``), omit them if possible.
            If ``True``, force all blocks to be created, with zero entries.

        """
        ...

    @abstractmethod
    def zero_diagonal_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> DiagonalData: ...

    @abstractmethod
    def zero_mask_data(self, large_leg: Space, device: str) -> MaskData: ...

    def is_real(self, a: SymmetricTensor) -> bool:
        """If the Tensor is comprised of real numbers.

        Complex numbers with small or zero imaginary part still cause a `False` return.
        """
        # FusionTree backend might implement this differently.
        return a.dtype.is_real

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.block_backend, subpath + 'block_backend')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.block_backend = hdf5_loader.load(subpath + 'block_backend')


def conventional_leg_order(
    tensor_or_codomain: SymmetricTensor | TensorProduct, domain: TensorProduct = None
) -> Generator[Space, None, None]:
    """The conventional order of legs."""
    if domain is None:
        codomain = tensor_or_codomain.codomain
        domain = tensor_or_codomain.domain
    else:
        codomain = tensor_or_codomain
    yield from codomain.factors
    yield from reversed(domain.factors)


class HasBackend(Protocol):  # noqa D101
    backend: TensorBackend


def get_same_backend(*objs: HasBackend, error_msg: str = 'Incompatible backends.') -> TensorBackend:
    """If the given object have the same backend, return it. Raise otherwise."""
    if len(objs) == 0:
        raise ValueError('Need at least one tensor')
    backend = objs[0].backend
    if not all(o.backend == backend for o in objs[1:]):
        raise ValueError(error_msg)
    return backend
