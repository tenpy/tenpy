"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from ..tools.misc import to_iterable
from .dtypes import Dtype

# placeholder for a backend-specific type that represents the blocks of symmetric tensors
Block = TypeVar('Block')

if TYPE_CHECKING:
    from ..symmetries.spaces import Space


class BlockBackend(metaclass=ABCMeta):
    """Abstract base class that defines the operation on dense blocks."""

    svd_algorithms: list[str]  # first is default
    BlockCls = None  # to be set by subclass

    def __init__(self, default_device: str):
        self.default_device = default_device

    def __repr__(self):
        return f'{type(self).__name__}()'

    def __str__(self):
        return f'{type(self).__name__}()'

    def apply_basis_perm(self, block: Block, legs: list[Space], inv: bool = False) -> Block:
        """Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block"""
        # OPTIMIZE avoid applying permutations that we know are trivial (_basis_perm = None)
        if inv:
            perms = [leg.inverse_basis_perm for leg in legs]
        else:
            perms = [leg.basis_perm for leg in legs]
        return self.apply_leg_permutations(block, perms)

    def apply_leg_permutations(self, block: Block, perms: list[np.ndarray]) -> Block:
        """Apply permutations to every axis of a dense block"""
        assert len(block.shape) == len(perms)
        return block[np.ix_(*perms)]

    @abstractmethod
    def as_block(
        self, a, dtype: Dtype = None, return_dtype: bool = False, device: str = None
    ) -> Block | tuple[Block, Dtype]:
        """Convert objects to blocks.

        Should support blocks, numpy arrays, nested python containers. May support more.
        If `a` is already a block of correct dtype on the correct device, it may be returned
        un-modified.

        Returns
        -------
        block: Block
            The new block
        dtype: Dtype, optional
            The new dtype of the block. Only returned if `return_dtype`.
        device: str, optional
            The device for the block. Default behavior (if ``None``) is to leave `a` on its
            current device if it already is a block, and to use :attr:`default_device` if a new
            block needs to be created (e.g. if `a` is a list).

        See Also
        --------
        block_copy
            Guarantees an independent copy.

        """
        ...

    @abstractmethod
    def as_device(self, device: str | None) -> str:
        """Convert input string to unambiguous device name.

        In particular, this should map any possible aliases to one unique name, e.g.
        for PyTorch, map ``'cuda'`` to ``'cuda:0'``.
        Also checks if that device is valid and available.
        """
        ...

    @abstractmethod
    def abs_argmax(self, block: Block) -> list[int]:
        """Return the indices (one per axis) of the largest entry (by magnitude) of the block"""
        ...

    @abstractmethod
    def add_axis(self, a: Block, pos: int) -> Block: ...

    @abstractmethod
    def block_all(self, a) -> bool:
        """Require a boolean block. If all of its entries are True"""
        ...

    @abstractmethod
    def allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool: ...

    @abstractmethod
    def angle(self, a: Block) -> Block:
        """The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise."""
        ...

    @abstractmethod
    def block_any(self, a) -> bool:
        """Require a boolean block. If any of its entries are True"""
        ...

    def apply_mask(self, block: Block, mask: Block, ax: int) -> Block:
        """Apply a mask (1D boolean block) to a block, slicing/projecting that axis"""
        idx = (slice(None, None, None),) * ax + (mask,)
        return block[idx]

    def argsort(self, block: Block, sort: str = None, axis: int = 0) -> Block:
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
            raise ValueError('unknown sort option ' + repr(sort))
        return self._argsort(block, axis=axis)

    @abstractmethod
    def _argsort(self, block: Block, axis: int) -> Block:
        """Like :meth:`block_argsort` but can assume real valued block, and sort ascending"""
        ...

    def combine_legs(self, a: Block, leg_idcs_combine: list[list[int]], cstyles: bool | list[bool] = True) -> Block:
        """Combine each group of legs in `leg_idcs_combine` into a single leg.

        The group of legs in each entry of `leg_idcs_combine` must be contiguous.
        The legs can be combined in C style (default) or F style; the style can
        be specified for each group of legs independently.
        """
        cstyles = to_iterable(cstyles)  # single bool to list
        if len(cstyles) == 1:
            cstyles *= len(leg_idcs_combine)
        old_shape = self.get_shape(a)
        axes_perm = list(range(len(old_shape)))
        new_shape = []
        last_stop = 0
        for group, cstyle in zip(leg_idcs_combine, cstyles):
            start = group[0]
            stop = group[-1] + 1
            if group != [*range(start, stop)]:
                raise ValueError('Each group in leg_idcs_combine must be contiguous and ascending')
            if start < last_stop:
                raise ValueError('The groups in leg_idcs_combine must not overlap')
            new_shape.extend(old_shape[last_stop:start])  # all leg *not* to be combined
            new_shape.append(np.prod(old_shape[start:stop]))
            if not cstyle:
                axes_perm[start:stop] = axes_perm[start:stop][::-1]
            last_stop = stop
        new_shape.extend(old_shape[last_stop:])
        return self.reshape(self.permute_axes(a, axes_perm), tuple(new_shape))

    @abstractmethod
    def conj(self, a: Block) -> Block:
        """Complex conjugate of a block"""
        ...

    @abstractmethod
    def copy_block(self, a: Block, device: str = None) -> Block:
        """Create a new, independent block with the same data

        Parameters
        ----------
        a
            The block to copy
        device
            The device for the new block. Per default, use the same device as the old block.

        See Also
        --------
        as_block
            Function to guarantee dtype and device, without forcing copies.

        """
        ...

    def cutoff_inverse(self, a: Block, cutoff: float) -> Block:
        """The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``."""
        res = 1.0 * self.copy_block(a)
        res[abs(a) < cutoff] = float('inf')
        return 1 / res

    def dagger(self, a: Block) -> Block:
        """Permute axes to reverse order and elementwise conj."""
        num_legs = len(self.get_shape(a))
        res = self.permute_axes(a, list(reversed(range(num_legs))))
        return self.conj(res)

    @abstractmethod
    def get_dtype(self, a: Block) -> Dtype: ...

    @abstractmethod
    def eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
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
    def eigvalsh(self, block: Block, sort: str = None) -> Block:
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

    def enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(self.get_shape(block))
        shape[axis] = self.get_shape(mask)[0]
        res = self.zeros(shape, dtype=self.get_dtype(block))
        idcs = (slice(None, None, None),) * axis + (mask,)
        res[idcs] = self.copy_block(block)  # OPTIMIZE copy needed?
        return res

    @abstractmethod
    def exp(self, a: Block) -> Block:
        """The *elementwise* exponential.

        Not to be confused with :meth:`matrix_exp`, the *matrix* exponential.
        """
        ...

    @abstractmethod
    def block_from_diagonal(self, diag: Block) -> Block:
        """Return a 2D square block that has the 1D ``diag`` on the diagonal"""
        ...

    @abstractmethod
    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        """Convert a mask to a full block.

        Return a (N, M) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
        where N is the number of True entries. The result is the coefficient matrix of the projection map.
        """
        ...

    @abstractmethod
    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None, device: str = None) -> Block: ...

    @abstractmethod
    def get_device(self, a: Block) -> str: ...

    @abstractmethod
    def get_diagonal(self, a: Block, tol: float | None) -> Block:
        """Get the diagonal of a 2D block as a 1D block"""
        ...

    @abstractmethod
    def imag(self, a: Block) -> Block:
        """The imaginary part of a complex number, elementwise."""
        ...

    def inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        """Dense block version of tensors.inner.

        If do dagger, ``sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])``
        otherwise, ``sum(a[i1, ..., iN] * b[iN, ..., i2, i1])``.
        """
        if do_dagger:
            a = self.conj(a)
        else:
            a = self.permute_axes(a, list(reversed(range(a.ndim))))
        return self.sum_all(a * b)

    def is_real(self, a: Block) -> bool:
        """If the block is comprised of real numbers.

        Complex numbers with small or zero imaginary part still cause a `False` return.
        """
        return self.cyten_dtype_map[self.get_dtype(a)].is_real

    @abstractmethod
    def item(self, a: Block) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def kron(self, a: Block, b: Block) -> Block:
        """The kronecker product.

        Parameters
        ----------
        a, b
            Two blocks with the same number of dimensions.

        Notes
        -----
        The elements are products of elements from `a` and `b`::
            kron(a, b)[k0, k1, ..., kN] = a[i0, i1, ..., iN] * b[j0, j1, ..., jN]

        where::
            kt = it * st + jt,  t = 0,...,N

        (Taken from numpy docs)

        """
        ...

    def linear_combination(self, a, v: Block, b, w: Block) -> Block:
        return a * v + b * w

    @abstractmethod
    def log(self, a: Block) -> Block:
        """The *elementwise* natural logarithm.

        Not to be confused with :meth:`matrix_log`, the *matrix* logarithm.
        """
        ...

    @abstractmethod
    def max(self, a: Block) -> float: ...

    @abstractmethod
    def max_abs(self, a: Block) -> float: ...

    @abstractmethod
    def min(self, a: Block) -> float: ...

    def mul(self, a: float | complex, b: Block) -> Block:
        return a * b

    @abstractmethod
    def norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
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
    def outer(self, a: Block, b: Block) -> Block:
        """Outer product of blocks.

        ``res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM]``
        """
        ...

    @abstractmethod
    def permute_axes(self, a: Block, permutation: list[int]) -> Block: ...

    def permute_combined_matrix(
        self, block: Block, dims1: Sequence[int], idcs1: Sequence[int], dims2: Sequence[int], idcs2: Sequence[int]
    ) -> Block:
        """For a matrix `a` with two combined multi-indices, permute the sub-indices.

        Parameters
        ----------
        a : 2D Block
            A matrix with combined axes ``[(m1.m2...mJ), (n1.n2...nK)]``.
        dims1 : list or 1D array of int
            The dimensions of the subindices ``[m1, m2, ..., mJ]``.
        idcs1 : list or 1D array of int
            Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the first
            multi-index of the result.
        dims2 : list or 1D array of int
            The dimensions of the subindices ``[n1, n2, ..., nK]``.
        idcs2 : list or 1D array of int
            Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the second
            multi-index of the result.

        Returns
        -------
        2D block
            A matrix with the same entries as `a`, but rearranged to the new axis order,
            e.g. ``[M, N]``, where ``M == combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs1])``
            and ``N == combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs2])``.

        See Also
        --------
        permute_combined_idx

        """
        block = self.reshape(block, [*dims1, *dims2])
        block = self.permute_axes(block, [*idcs1, *idcs2])
        shape = self.get_shape(block)
        M_new = prod(shape[: len(idcs1)])
        N_new = prod(shape[len(idcs1) :])
        return self.reshape(block, (M_new, N_new))

    def permute_combined_idx(self, block: Block, axis: int, dims: Sequence[int], idcs: Sequence[int]) -> Block:
        """For a matrix `a` with a single combined multi-index, permute sub-indices.

        Parameters
        ----------
        a : 2D Block
            A matrix with axes ``[M, N]``, where either ``M = (m1.m2...mJ)`` or ``N = (n1.n2...nK)``
            is a multi-index *but not both*.
        axis : int
            Which of the two axes has the multi-indices
        dims : list or 1D array of int
            The dimensions of the sub-indices, e.g. ``[m1, m2, ..., mJ]``.
        idcs : list of 1D array of int
            The order of the sub-indices in the results, such that the result has
            axes ``[[m1, m2, ..., mJ][i] for i in idcs]``.

        Returns
        -------
        2D Block
            A matrix with the same entries as `a`, but rearranged to the new axis order,
            i.e. ``[M_new, N_new]`` where e.g. ``M_new = combined([m1, m2, ..., mJ][idcs])``.

        See Also
        --------
        permute_combined_matrix

        """
        M, N = self.get_shape(block)
        assert -2 <= axis < 2
        if axis < 0:
            axis = axis + 2
        if axis == 0:
            block = self.reshape(block, [*dims, N])
            block = self.permute_axes(block, [*idcs, len(idcs)])
            return self.reshape(block, (M, N))
        if axis == 1:
            block = self.reshape(block, [M, *dims])
            block = self.permute_axes(block, [0, *(1 + i for i in idcs)])
            return self.reshape(block, (M, N))
        raise RuntimeError

    @abstractmethod
    def random_normal(self, dims: list[int], dtype: Dtype, sigma: float, device: str = None) -> Block: ...

    @abstractmethod
    def random_uniform(self, dims: list[int], dtype: Dtype, device: str = None) -> Block: ...

    @abstractmethod
    def real(self, a: Block) -> Block:
        """The real part of a complex number, elementwise."""
        ...

    @abstractmethod
    def real_if_close(self, a: Block, tol: float) -> Block:
        """If a block is close to its real part, return the real part.

        Otherwise the original block. Elementwise.
        """
        ...

    @abstractmethod
    def tile(self, a: Block, repeats: int) -> Block:
        """Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat."""
        ...

    @abstractmethod
    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]: ...

    @abstractmethod
    def reshape(self, a: Block, shape: tuple[int]) -> Block: ...

    def scale_axis(self, block: Block, factors: Block, axis: int) -> Block:
        """Multiply block with the factors (a 1D block), along a given axis.

        E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
        ``block * factors[None, None, :, None]``.
        """
        idx = [None] * len(self.get_shape(block))
        idx[axis] = slice(None, None, None)
        return block * factors[tuple(idx)]

    @abstractmethod
    def get_shape(self, a: Block) -> tuple[int]: ...

    def split_legs(self, a: Block, idcs: list[int], dims: list[list[int]], cstyles: bool | list[bool] = True) -> Block:
        """Split legs into groups of legs with specified dimensions.

        The splitting of a leg can be in C style (default) or F style. In the
        latter case, the specified dimensions of the resulting group of legs
        *are reversed*. The style can be specified for each group of legs
        independently.
        """
        cstyles = to_iterable(cstyles)  # single bool to list
        if len(cstyles) == 1:
            cstyles *= len(idcs)
        axes_perm = []
        old_shape = self.get_shape(a)
        new_shape = []
        start = 0
        for i, i_dims, cstyle in zip(idcs, dims, cstyles):
            new_shape.extend(old_shape[start:i])
            new_shape.extend(i_dims)
            axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + i - start)))
            if cstyle:
                axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(i_dims))))
            else:
                axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(i_dims)))[::-1])
            start = i + 1
        new_shape.extend(old_shape[start:])
        axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(old_shape) - start)))
        return self.permute_axes(self.reshape(a, tuple(new_shape)), axes_perm)

    @abstractmethod
    def sqrt(self, a: Block) -> Block:
        """The elementwise square root"""
        ...

    @abstractmethod
    def squeeze_axes(self, a: Block, idcs: list[int]) -> Block: ...

    @abstractmethod
    def stable_log(self, block: Block, cutoff: float) -> Block:
        """Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0."""
        ...

    @abstractmethod
    def sum(self, a: Block, ax: int) -> Block:
        """The sum over a single axis."""
        ...

    @abstractmethod
    def sum_all(self, a: Block) -> float | complex:
        """The sum of all entries of the block.

        If the block contains boolean values, this should return the number of ``True`` entries.
        """
        ...

    @abstractmethod
    def tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block: ...

    def tensor_outer(self, a: Block, b: Block, K: int) -> Block:
        """Version of ``tensors.outer`` on blocks.

        Note the different leg order to usual outer products::

            res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]

        intended to be used with ``K == a_num_codomain_legs``.
        """
        res = self.outer(a, b)  # [i1,...,iN,j1,...,jM]
        N = len(self.get_shape(a))
        M = len(self.get_shape(b))
        return self.permute_axes(res, [*range(K), *range(N, N + M), *range(K, N)])

    @abstractmethod
    def to_dtype(self, a: Block, dtype: Dtype) -> Block: ...

    def to_numpy(self, a: Block, numpy_dtype=None) -> np.ndarray:
        # BlockBackends may override, if this implementation is not valid
        return np.asarray(a, dtype=numpy_dtype)

    @abstractmethod
    def trace_full(self, a: Block) -> float | complex: ...

    @abstractmethod
    def trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Block: ...

    def eye_block(self, legs: list[int], dtype: Dtype, device: str = None) -> Block:
        """The identity matrix, reshaped to a block.

        Note the unusual leg order ``[m1,...,mJ,mJ*,...,m1*]``,
        which is chosen to match :meth:`eye_data`.

        Note also that the ``legs`` only specify the dimensions of the first half,
        namely ``m1,...,mJ``.
        """
        J = len(legs)
        eye = self.eye_matrix(prod(legs), dtype, device)
        # [M, M*] -> [m1,...,mJ,m1*,...,mJ*]
        eye = self.reshape(eye, legs * 2)
        # [m1,...,mJ,mJ*,...,m1*]
        return self.permute_axes(eye, [*range(J), *reversed(range(J, 2 * J))])

    @abstractmethod
    def eye_matrix(self, dim: int, dtype: Dtype, device: str = None) -> Block:
        """The ``dim x dim`` identity matrix"""
        ...

    @abstractmethod
    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool: ...

    def get_block_mask_element(self, a: Block, large_leg_idx: int, small_leg_idx: int, sum_block: int = 0) -> bool:
        """Get an element of a mask.

        Mask elements are `True` if the entry `a[large_leg_idx]` is the `small_leg_idx`-th `True`
        in the block.

        Parameters
        ----------
        a
            The mask block
        large_leg_idx, small_leg_idx
            The block indices
        sum_block
            Number of `True` entries in the block, i.e., ``sum_block == self.sum_all(a)``. Agrees
            with the sector multiplicity of the small leg.
            (Only important if the sector dimension is larger than 1.)

        """
        offset = (large_leg_idx // self.get_shape(a)[0]) * sum_block
        large_leg_idx = large_leg_idx % self.get_shape(a)[0]
        # if this does not work, need to override.
        if not a[large_leg_idx]:
            # if the block has a False entry, the matrix has only False in that column
            return False
        # otherwise, there is exactly one True in that column, at index sum(a[:large_leg_idx])
        return bool(small_leg_idx == offset + self.sum_all(a[:large_leg_idx]))

    @abstractmethod
    def matrix_dot(self, a: Block, b: Block) -> Block:
        """As in numpy.dot, both a and b might be matrix or vector."""
        ...

    @abstractmethod
    def matrix_exp(self, matrix: Block) -> Block: ...

    @abstractmethod
    def matrix_log(self, matrix: Block) -> Block: ...

    def matrix_lq(self, a: Block, full: bool) -> tuple[Block, Block]:
        q, r = self.matrix_qr(self.permute_axes(a, [1, 0]), full=full)
        return self.permute_axes(r, [1, 0]), self.permute_axes(q, [1, 0])

    @abstractmethod
    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        """QR decomposition of a 2D block"""
        ...

    @abstractmethod
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """Internal version of :meth:`matrix_svd`, to be implemented by subclasses."""
        ...

    @abstractmethod
    def ones_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block: ...

    def synchronize(self):
        """Wait for asynchronous processes (if any) to finish"""
        pass

    def test_block_sanity(
        self,
        block,
        expect_shape: tuple[int, ...] | None = None,
        expect_dtype: Dtype | None = None,
        expect_device: str | None = None,
    ):
        assert isinstance(block, self.BlockCls), 'wrong block type'
        if expect_shape is not None:
            if self.get_shape(block) != expect_shape:
                msg = f'wrong block shape {self.get_shape(block)} != {expect_shape}'
                raise AssertionError(msg)
        if expect_dtype is not None:
            assert self.get_dtype(block) == expect_dtype, 'wrong block dtype'
        if expect_device is not None:
            assert self.get_device(block) == expect_device, 'wrong block device'

    @abstractmethod
    def zeros(self, shape: list[int], dtype: Dtype, device: str = None) -> Block: ...

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.BlockCls, subpath + 'BlockCls')
        hdf5_saver.save(self.svd_algorithms, subpath + 'svd_algorithms')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.BlockCls = hdf5_loader.load(subpath + 'BlockCls')
        obj.svd_algorithms = hdf5_loader.load(subpath + 'svd_algorithms')

        return obj
