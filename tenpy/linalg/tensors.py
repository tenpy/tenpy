"""

TODO put this in the proper place:
The following characters have special meaning in labels and should be avoided:
`(`, `.`, `)`, `?`, `!` and `*`.
"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABC, abstractmethod
import operator
from typing import TypeVar, Sequence, NoReturn
from numbers import Integral, Number
import numpy as np
import warnings
from functools import cached_property

from .misc import duplicate_entries, force_str_len, join_as_many_as_possible
from .dummy_config import config
from .symmetries.groups import AbelianGroup
from .symmetries.spaces import VectorSpace, ProductSpace
from .backends.backend_factory import get_default_backend
from .backends.abstract_backend import Dtype, Block, AbstractBackend
from ..tools.misc import to_iterable, to_iterable_of_len

__all__ = ['Shape', 'AbstractTensor', 'Tensor', 'ChargedTensor', 'DiagonalTensor', 'Mask',
           'almost_equal', 'combine_legs', 'conj', 'inner', 'is_scalar', 'norm', 'outer',
           'permute_legs', 'split_legs', 'squeeze_legs', 'tdot', 'trace', 'zero_like',
           'get_same_backend', 'match_leg_order']

# svd, qr, eigen, exp, log, ... are implemented in matrix_operations.py

class Shape:
    """An object storing the legs and labels of a tensor.
    When iterated or indexed, it behaves like a sequence of integers, the dimension of the legs.
    Can be indexed by integer (leg position) or string (leg label).
    """

    def __init__(self, legs: list[VectorSpace], labels: list[str | None] = None):
        self.legs = legs
        if labels is None:
            labels = [None] * len(legs)
        else:
            labels = labels[:]
        self._labels = labels
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}
        self.num_legs = len(legs)
        self.dims = [l.dim for l in legs]

    def test_sanity(self):
        assert not duplicate_entries(self._labels, ignore=[None])

    def __iter__(self):
        return iter(self.dims)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = self.label_to_legnum(key)
            except ValueError:
                raise IndexError(f'No leg with label {key}.') from None
        return self.dims[key]

    def set_labels(self, labels: list[str | None]):
        assert not duplicate_entries(labels, ignore=[None])
        assert len(labels) == self.num_legs
        self._labels = labels[:]
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}

    @property
    def labels(self) -> list[str | None]:
        return self._labels[:]

    @labels.setter
    def labels(self, value):
        self.set_labels(value)

    @property
    def is_fully_labelled(self) -> bool:
        return None not in self._labels

    def label_to_legnum(self, label: str) -> int:
        num = self._labelmap.get(label, None)
        if num is None:
            raise ValueError(f'No leg with label {label}')
        return num

    def __str__(self):
        dims = ','.join((f"{lbl}:{d:d}" if lbl is not None else str(d))
                        for lbl, d in zip(self._labels, self.dims))
        return f"({dims})"


# ##################################
# tensor classes
# ##################################


class AbstractTensor(ABC):
    """
    Common base class for tensors.

    .. note ::
        TODO write clean text about VectorSpace.sector_perm and how it affects internal storage

    Parameters
    ----------
    legs : list[VectorSpace]
        The legs of the Tensor
    backend: :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    #  backend.get_dtype_from_data(self.data)
    def __init__(self, legs: list[VectorSpace], backend, labels: list[str | None] | None, dtype: Dtype):
        if backend is None:
            self.backend = get_default_backend()
        else:
            self.backend = backend
        #  self.legs = [self.backend.convert_vector_space(leg) for leg in legs]  # TODO: this causes issues since data.block_inds don't fit any more!!!!!
        self.legs = list(legs)
        for leg in self.legs:
            assert isinstance(leg, (self.backend.VectorSpaceCls, self.backend.ProductSpaceCls))  # TODO: remove this test
        self.shape = Shape(legs=self.legs, labels=labels)
        self.num_legs = len(legs)
        self.symmetry = legs[0].symmetry
        self.dtype = dtype

    def test_sanity(self) -> None:
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.symmetry == self.symmetry for l in self.legs)
        assert len(self.legs) == self.shape.num_legs == self.num_legs > 0
        for leg in self.legs:
            assert isinstance(leg, (self.backend.VectorSpaceCls, self.backend.ProductSpaceCls))
        self.shape.test_sanity()

    # ----------------------------------
    # Concrete Implementations
    # ----------------------------------

    @property
    def is_fully_labelled(self) -> bool:
        return self.shape.is_fully_labelled

    @property
    def labels(self) -> list[str | None]:
        return self.shape.labels

    @labels.setter
    def labels(self, value):
        self.shape.set_labels(value)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors with the same legs"""
        return self.parent_space.num_parameters

    @cached_property
    def parent_space(self) -> ProductSpace:
        """The space that the tensor lives in"""
        return self.legs[0].ProductSpace(self.legs)

    @property
    def size(self) -> int:
        """The total number of entries, i.e. the dimension of the space of tensors on the same space
        if symmetries were ignored"""
        return self.parent_space.dim

    def index(self, *legs: int | str) -> _TensorIndexHelper:
        """This method allows indexing a tensor "by label".

        It returns a helper object, that can be indexed instead of self.
        For example, if we have a tensor with labels 'a', 'b' and 'c', but we are not sure about
        their order, we can call ``tensor.index('a', 'b')[0, 1]``.
        If ``tensors.labels == ['a', 'b', 'c']`` in alphabetic order, we get ``tensor[0, 1]``.
        However if the order of labels happens to be different, e.g.
        ``tensor.labels == ['b', 'c', 'a']`` we get ``tensor[1, :, 0]``.
        """
        return _TensorIndexHelper(self, legs)

    def has_label(self, label: str, *more: str) -> bool:
        return label in self.shape._labels and all(l in self.shape._labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        if not self.is_fully_labelled:
            return False
        if len(labels) != len(self.shape._labels):
            return False
        return set(labels) == set(self.shape._labels)

    def set_labels(self, labels: list[str | None]) -> None:
        self.shape.set_labels(labels)

    def get_leg_idx(self, which_leg: int | str) -> int:
        if isinstance(which_leg, str):
            which_leg = self.shape.label_to_legnum(which_leg)
        if isinstance(which_leg, (int, np.int32, np.int64)):
            if which_leg < 0:
                which_leg = which_leg + self.num_legs
            if not 0 <= which_leg < self.num_legs:
                raise ValueError(f'Leg index out of bounds: {which_leg}.') from None
            return which_leg
        else:
            raise TypeError

    def get_leg_idcs(self, which_legs: int | str | list[int | str]) -> list[int]:
        if isinstance(which_legs, (int, str)):
            return [self.get_leg_idx(which_legs)]
        else:
            return list(map(self.get_leg_idx, which_legs))

    def get_legs(self, which_legs: int | str | list[int | str]) -> list[VectorSpace]:
        return [self.legs[idx] for idx in self.get_leg_idcs(which_legs)]

    def to_numpy_ndarray(self, leg_order: list[int | str] = None, numpy_dtype=None) -> np.ndarray:
        """Convert to a numpy array"""
        block = self.to_dense_block(leg_order=leg_order)
        return self.backend.block_to_numpy(block, numpy_dtype=numpy_dtype)

    # ----------------------------------
    # Private and Dunder methods
    # ----------------------------------

    def _repr_leg_components(self, max_len: int) -> list[str]:
        """A summary of the components of legs, used in repr"""
        components_strs = []
        for leg, label in zip(self.legs, self.labels):
            if isinstance(leg, ProductSpace):
                sublabels = [f'?{n}' if l is None else l for n, l in enumerate(_split_leg_label(label))]
                prefix = 'ProductSpace: '
                components = prefix + join_as_many_as_possible(
                    [f'({l}: {s.dim})' for l, s in zip(sublabels, leg.spaces)],
                    separator=' ⊗ ',
                    priorities=[s.dim for s in leg.spaces],
                    max_len=max_len - len(prefix)
                )
            else:
                components = join_as_many_as_possible(
                    [f'({mult} * {leg.symmetry.sector_str(sect)})'
                     for mult, sect in zip(leg.multiplicities, leg.sectors)],
                    separator=' ⊕ ',
                    priorities=leg.multiplicities,
                    max_len=max_len
                )
            components_strs.append(components)
        return components_strs

    def _repr_header_lines(self, indent: str) -> list[str]:
        num_cols_label = min(10, max(5, *(len(str(l)) for l in self.labels)))
        num_cols_dim = min(5, max(3, *(len(str(leg.dim)) for leg in self.legs)))

        label_strs = [force_str_len(label, num_cols_label, rjust=False) for label in self.labels]
        dim_strs = [force_str_len(leg.dim, num_cols_dim) for leg in self.legs]
        dual_strs = ['dual' if leg.is_dual else '   /' for leg in self.legs]
        components_strs = self._repr_leg_components(max_len=50)

        lines = [
            f'{indent}* Backend: {self.backend}',
            f'{indent}* Symmetry: {self.symmetry}',
            f'{indent}* Legs:  label{" " * (num_cols_label - 5)}  {" " * (num_cols_dim - 3)}dim  dual  components',
            f'{indent}         {"=" * (10 + num_cols_label + num_cols_dim + max(10, *(len(c) for c in components_strs)))}',
        ]
        for entries in zip(label_strs, dim_strs, dual_strs, components_strs):
            lines.append(f'{indent}         {"  ".join(entries)}')

        return lines

    def _getitem_apply_masks(self, masks: list[Mask], legs: list[int]) -> AbstractTensor:
        """Helper function for __getitem__ to index self with masks.
        Subclasses may override this implementation."""
        res = self
        for mask, leg in zip(masks, legs):
            res = res.apply_mask(mask, leg)
        return res

    def _input_checks_inner(self, other: AbstractTensor, do_conj: bool, legs1: list[int | str] | None,
                            legs2: list[int | str] | None) -> list[int] | None:
        """Check if inputs to inner are valid.

        Returns
        -------
        leg_order_2 : (list of int) or None
            The order of legs on other, such that they match the legs of self.
            None is returned instead of the trivial order ``[0, 1, 2, ...]``.
        """
        if self.num_legs != other.num_legs:
            raise ValueError('Tensors need to have the same number of legs')
        # determine leg_order_2
        if legs1 is None and legs2 is None:
            leg_order_2 = match_leg_order(self, other)
        else:
            if legs1 is None:
                legs1 = np.arange(self.num_legs, dtype=int)
            else:
                legs1 = np.array(self.get_leg_idcs(legs1), dtype=int)
            if legs2 is None:
                legs2 = np.arange(self.num_legs, dtype=int)
            else:
                legs2 = np.array(other.get_leg_idcs(legs2), dtype=int)
            if np.all(legs1 == legs2):
                leg_order_2 = None
            else:
                leg_order_2 = np.argsort(legs1)[legs2]
        if leg_order_2 is None:
            other_legs_ordered = other.legs
        else:
            other_legs_ordered = (other.legs[leg_order_2[n]] for n in range(self.num_legs))
        if do_conj:
            are_compatible = all(l1 == l2 for l1, l2 in zip(self.legs, other_legs_ordered))
        else:
            are_compatible = all(l1.can_contract_with(l2) for l1, l2 in zip(self.legs, other_legs_ordered))
        if not are_compatible:
            raise ValueError('Incompatible legs')
        return leg_order_2

    def _input_checks_add_tensor(self, other: AbstractTensor) -> list[int] | None:
        """Check if inputs to _add_tensor are valid.

        Returns
        -------
        other_order : (list of int) or None
            The order of legs on other, such that they match the legs of self.
            None is equivalent to ``list(range(other.num_legs)`` and indicates that no permutation is needed.
        """
        other_order = match_leg_order(self, other)  # TODO double check and cover in tests that this works
        for n, (leg_self, leg_other) in enumerate(zip(self.legs, other.legs)):
            if leg_self != leg_other:
                self_label = self.shape._labels[n]
                self_label = '' if self_label is None else self_label + ': '
                other_label = other.shape._label[n]
                other_label = '' if other_label is None else other_label + ': '
                msg = '\n'.join([
                    'Incompatible legs for +:',
                    self_label + str(leg_self),
                    other_label + str(leg_other)
                ])
                raise ValueError(msg)
        return other_order

    def __repr__(self):
        indent = '  '
        lines = [f'{self.__class__.__name__}(']
        lines.extend(self._repr_header_lines(indent=indent))
        lines.extend(self.backend._data_repr_lines(self.data, indent=indent, max_width=70, max_lines=20))
        lines.append(')')
        return "\n".join(lines)

    def __getitem__(self, idcs):
        # TODO tests
        """We support two modes of indexing tensors for __getitem__:
        - Getting single entries, i.e. giving one integer per leg
        - Getting a "masked" Tensor, i.e. giving a Mask for some or all legs.
          Legs not to be masked can be indicated via ``slice(None, None, None)``, i.e. typing ``:``,
          or ``Ellipsis``, i.e. typing ``...``.
        """
        idcs = _parse_idcs(idcs, length=self.num_legs)
        if isinstance(idcs[0], int):
            if not all(isinstance(idx, int) for idx in idcs[1:]):
                msg = 'Invalid index type. If tensors are indexed by integer, all legs need to be indexed by an intger.'
                raise IndexError(msg)
            for leg_num, idx in enumerate(idcs):
                if not -self.legs[leg_num].dim <= idx < self.legs[leg_num].dim:
                    msg = f'Index {idx} is out of bounds for leg {leg_num} with label {self.labels[leg_num]} ' \
                          f'and dimension {self.legs[leg_num].dim}'
                    raise IndexError(msg)
                if idx < 0:
                    idcs[leg_num] = idx + self.legs[leg_num].dim
            return self._get_element(idcs)
        else:
            mask_legs = []
            masks = []
            for leg_num, idx in enumerate(idcs):
                if isinstance(idx, Mask):
                    masks.append(idx)
                    mask_legs.append(leg_num)
                elif isinstance(idx, slice):
                    if idx != slice(None, None, None):
                        raise IndexError('Non-trivial slices are not supported.')
                else:
                    raise IndexError(f'Invalid index type: {type(idx)}')
            return self._getitem_apply_masks(masks=masks, legs=mask_legs)

    def __setitem__(self, idcs, value):
        # TODO tests
        idcs = _parse_idcs(idcs, length=self.num_legs)
        for leg_num, idx in enumerate(idcs):
            if not isinstance(idx, int):
                raise IndexError('Can only set single entries')
            if not -self.legs[leg_num].dim <= idx < self.legs[leg_num].dim:
                msg = f'Index {idx} is out of bounds for leg {leg_num} with label {self.labels[leg_num]} ' \
                        f'and dimension {self.legs[leg_num].dim}'
                raise IndexError(msg)
            if idx < 0:
                idcs[leg_num] = idx + self.legs[leg_num].dim
        value = self.dtype.convert_python_scalar(value)
        self._set_element(idcs, value)

    def __neg__(self):
        return self._mul_scalar(-1)

    def __pos__(self):
        return self

    def __eq__(self, other):
        msg = f'{type(self)} does not support == comparison. Use tenpy.almost_equal instead.'
        raise TypeError(msg)

    def __add__(self, other):
        if isinstance(other, AbstractTensor):
            return self._add_tensor(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, AbstractTensor):
            return self.__add__(other._mul_scalar(-1))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self._mul_scalar(other)
        raise TypeError(f'Tensors can only be multiplied with scalars, not {type(other)}.') from None

    def __rmul__(self, other):
        # all allowed multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise TypeError(f'Tensors can only be divived by scalars, not {type(other)}.') from None
        try:
            other_inv = 1. / other
        except Exception:
            raise ValueError(f'Tensors can only be divided by invertible scalars.') from None
        return self._mul_scalar(1. / other)

    def __float__(self):
        if not self.dtype.is_real:
            warnings.warn("converting complex to real, only return real part!", stacklevel=2)
        return self.item().real

    def __complex__(self):
        return complex(self.item())

    # ----------------------------------
    # Abstract methods
    # ----------------------------------

    @classmethod
    @abstractmethod
    def zero(cls, legs: VectorSpace | list[VectorSpace], backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.float64) -> AbstractTensor:
        """A zero tensor"""
        ...

    @abstractmethod
    def apply_mask(self, mask: Mask, leg: int | str) -> AbstractTensor:
        ...

    @abstractmethod
    def combine_legs(self,
                     *legs: list[int | str],
                     new_legs: list[ProductSpace]=None,
                     product_spaces_dual: list[bool]=None,
                     new_axes: list[int]=None) -> AbstractTensor:
        """See tensors.combine_legs"""
        ...

    @abstractmethod
    def conj(self) -> AbstractTensor:
        """See tensors.conj"""
        ...

    @abstractmethod
    def copy(self, deep=True) -> AbstractTensor:
        ...

    @abstractmethod
    def item(self) -> bool | float | complex:
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        ...

    @abstractmethod
    def norm(self) -> float:
        """See tensors.norm"""
        ...

    @abstractmethod
    def permute_legs(self, permutation: list[int]) -> AbstractTensor:
        """See tensors.transpose"""
        ...

    @abstractmethod
    def split_legs(self, legs: list[int | str] = None) -> AbstractTensor:
        """See tensors.split_legs"""
        ...

    @abstractmethod
    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> AbstractTensor:
        """See tensors.squeeze_legs"""
        ...

    @abstractmethod
    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        """Convert tensor to a dense (i.e. no longer exploiting the symmetry structure) block,
        i.e. a numpy ndarray if the backend is a NumpyBlockBackend, or a torch Tensor of
        the backend is a TorchBlockBackend"""
        ...

    @abstractmethod
    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> AbstractTensor | float | complex:
        """See tensors.trace"""
        ...

    @abstractmethod
    def _get_element(self, idcs: list[int]) -> bool | float | complex:
        """Helper function for __getitem__ to index self with integers

        Parameters
        ----------
        idcs
            Indices which are partially pre-processed, i.e. there is one integer index per leg
            and it is in the correct range, i.e. `0 <= idx < leg.dim`.
            Indices are w.r.t. dense arrays, *not* the internal (sorted) order.
        """
        ...

    @abstractmethod
    def _mul_scalar(self, other: complex) -> AbstractTensor:
        ...

    @abstractmethod
    def _set_element(self, idcs: list[int], value: bool | float | complex) -> None:
        """Helper function for __setitem__ after arguments were parsed.
        Can assume that idcs has correct length and entries are valid & non-negative (0 <= idx < dim).
        Modifies self in-place with a modified copy of the underlying data
        """
        ...

    # ----------------------------------
    # Abstract binary tensor methods
    # ----------------------------------
    #  -> concrete implementations need to distinguish type of `other`

    @abstractmethod
    def almost_equal(self, other: AbstractTensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
        """See tensors.almost_equal"""
        ...

    @abstractmethod
    def inner(self, other: AbstractTensor, do_conj: bool = True,
              legs1: list[int | str] = None, legs2: list[int | str]  = None) -> float | complex:
        """See tensors.inner"""
        ...

    @abstractmethod
    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> AbstractTensor:
        """See tensors.outer"""
        ...

    @abstractmethod
    def tdot(self, other: AbstractTensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> AbstractTensor | float | complex:
        """See tensors.tdot"""
        ...

    @abstractmethod
    def _add_tensor(self, other: AbstractTensor) -> AbstractTensor:
        ...


class Tensor(AbstractTensor):
    """

    Attributes
    ----------
    data
        backend-specific data structure that contains the numerical data, i.e. the free parameters
        of tensors with the given symmetry.
        data about the symmetry is contained in the legs.
    backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
    legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`
        These may be instances of a backend-specifc subclass of :class:`~tenpy.linalg.symmetries.VectorSpace`
    labels : list of {``None``, str}
    """

    def __init__(self, data, legs: list[VectorSpace], backend=None, labels: list[str | None] | None = None):
        """
        This constructor is not user-friendly.
        Use as_tensor instead.
        Inputs are not checked for consistency.

        Parameters
        ----------
        data
            The numerical data ("free parameters") comprising the tensor. type is backend-specific
        backend: :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
            The backend for the Tensor
        legs : list[VectorSpace]
            The legs of the Tensor
        labels : list[str | None] | None
            Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
        """
        if backend is None:
            backend = get_default_backend()
        dtype = backend.get_dtype_from_data(data)
        AbstractTensor.__init__(self, backend=backend, legs=legs, labels=labels, dtype=dtype)
        self.data = data
        assert isinstance(data, self.backend.DataCls)

    def test_sanity(self) -> None:
        super().test_sanity()
        self.backend.test_data_sanity(self, is_diagonal=False)

    # --------------------------------------------
    # Additional methods (not in AbstractTensor)
    # --------------------------------------------

    @classmethod
    def eye(cls, legs: VectorSpace | list[VectorSpace], backend=None,
            labels: list[str | None] = None, dtype: Dtype = Dtype.float64) -> Tensor:
        """The identity map from one group of legs to their duals.

        Parameters
        ----------
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs : (list of) VectorSpace
            *Half* of the legs of the result. The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
            Can either give one label for each of the `legs`, and the second half will be the respective
            dual labels, or give twice as many and specify them all.
        dtype : Dtype, optional
            The data type of the Tensor entries.
        """
        if backend is None:
            backend = get_default_backend()
        legs = to_iterable(legs)
        legs = [backend.convert_vector_space(leg) for leg in legs]
        data = backend.eye_data(legs=legs, dtype=dtype)
        legs = legs + [leg.dual for leg in legs]
        if labels is not None:
            if len(labels) == len(legs):
                labels = labels + [_dual_leg_label(l) for l in labels]
            elif len(labels) != 2 * len(legs):
                msg = f'Wrong number of labels. Expected {len(legs)} or {2 * len(legs)}. Got {len(labels)}.'
                raise ValueError(msg)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def from_block_func(cls, func, legs: list[VectorSpace], backend=None,
                        labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> Tensor:
        """Create a Tensor from a block function.

        This function ceates a tensor by filling the blocks, i.e. the free paramaters of the tensors
        using `func`, which is a function returning backend-specific blocks.

        Parameters
        ----------
        func : callable
            A callable object which is called to generate the blocks.
            We expect that `func` returns a backend-specific block of the given `shape`.
            If no `shape_kw` is given, it is called as ``func(shape, **func_kwargs)``,
            otherwise as ``func(**{shape_kw: shape}, **func_kwargs)``,
            where `shape` is a tuple of int.
        legs : list of VectorSpace
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype

        See Also
        --------
        from_numpy_func
        """
        if backend is None:
            backend = get_default_backend()
        legs = [backend.convert_vector_space(leg) for leg in legs]

        if shape_kw is not None:
            def block_func(shape):
                block = func(**{shape_kw: shape}, **func_kwargs)
                return backend.block_to_dtype(block, dtype)
        else:
            def block_func(shape):
                block = func(shape, **func_kwargs)
                return backend.block_to_dtype(block, dtype)

        data = backend.from_block_func(block_func, legs)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def from_dense_block(cls, block, legs: list[VectorSpace], backend=None, dtype: Dtype=None,
                         labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Tensor:
        """Convert a dense block of the backend to a Tensor.

        If the block is not symmetric under the symmetry (specified by the legs), i.e. if
        ``not allclose(block, projected, atol, rtol)``, a ValueError is raised.

        Parameters
        ----------
        block : Block
            The data to be converted to a Tensor as a backend-specific block.
        legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`, optional
            The vectorspaces associated with legs of the tensors. This specifies the symmetry.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
            The backend for the Tensor
        dtype : ``np.dtype``, optional
            The data type of the Tensor entries. Defaults to dtype of `block`
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
        atol, rtol : float
            Absolute and relative tolerance for checking if the block is symmetric.

        See Also
        --------
        from_numpy
        """
        if backend is None:
            backend = get_default_backend()
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        data = backend.from_dense_block(block, legs=legs, atol=atol, rtol=rtol)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def from_numpy(cls, array: np.ndarray, legs: list[VectorSpace], backend=None, dtype: Dtype = None,
                   labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5) -> Tensor:
        """Like from_dense_block but `array` is a numpy array(-like)"""
        if backend is None:
            backend = get_default_backend()
        block = backend.block_from_numpy(np.asarray(array))
        return cls.from_dense_block(block=block, legs=legs, backend=backend, labels=labels, atol=atol,
                                    rtol=rtol, dtype=dtype)

    @classmethod
    def from_numpy_func(cls, func, legs: list[VectorSpace], backend=None,
                        labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> Tensor:
        """Create a Tensor from a numpy function.

        This function ceates a tensor by filling the blocks, i.e. the free paramaters of the tensors
        using `func`, which is a function returning numpy arrays, e.g. ``np.ones`` or
        ``np.random.standard_normal``

        Parameters
        ----------
        func : callable
            A callable object which is called to generate the blocks.
            We expect that `func` returns a numpy ndarray of the given `shape`.
            If no `shape_kw` is given, it is called as ``func(shape, **func_kwargs)``,
            otherwise as ``func(**{shape_kw: shape}, **func_kwargs)``,
            where `shape` is a tuple of int.
        legs : list of VectorSpace
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype

        See Also
        --------
        from_block_func
        """
        if backend is None:
            backend = get_default_backend()
        legs = [backend.convert_vector_space(leg) for leg in legs]

        if shape_kw is not None:
            def block_func(shape):
                arr = func(**{shape_kw: shape}, **func_kwargs)
                block = backend.block_from_numpy(arr)
                return backend.block_to_dtype(block, dtype)
        else:
            def block_func(shape):
                arr = func(shape, **func_kwargs)
                block = backend.block_from_numpy(arr)
                return backend.block_to_dtype(block, dtype)
        data = backend.from_block_func(block_func, legs)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def random_normal(cls, legs: VectorSpace | list[VectorSpace] = None,
                      mean: Tensor = None, sigma: float = 1.,
                      backend=None, labels: list[str | None] = None, dtype: Dtype = None) -> Tensor:
        r"""Generate a tensor from the normal distribution.

        The probability density is

        .. math ::
            p(T) \propto \mathrm{exp}\left[ \frac{(T - \mathtt{mean})^2}{2 \mathtt{sigma}^2} \right]

        .. note ::
            The tensors legs, backend and labels can be specified either via the `mean` or via
            explicit parameters, but not both.
            If `mean` is given, the explicit parameters are ignored.

        Parameters
        ----------
        legs : (list of) VectorSpace
            If `mean` is given, this argument is ignored and legs are the same as those of `mean`.
            Otherwise, the legs of the result.
        mean : Tensor | None
            The mean of the distribution. `mean=None` means a mean of zero and makes the
            `legs` argument required.
        sigma : float
            The standard deviation of the distribution
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            If `mean` is given, this argument is ignored and the backend is the same as of `mean`.
            Otherwise, the backend for the tensor
        labels : list[str | None], optional
            If `mean` is given, this argument is ignored and labels are the same as those of `mean`.
            Otherwise, labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype
            The dtype for the tensor. If not given, use the dtype of `mean`. If `mean` is not given,
            default to `Dtype.float64`.
        """
        if mean is not None:
            for name, val in zip(['legs', 'backend', 'labels'], [legs, backend, labels]):
                if val is not None:
                    warnings.warn(f'{name} argument to Tensor.random_normal was ignored, because mean was given.')

            if dtype is None:
                dtype = mean.dtype
            return mean + cls.random_normal(legs=mean.legs, mean=None, sigma=sigma, backend=mean.backend,
                                            labels=mean.labels, dtype=dtype)

        if backend is None:
            backend = get_default_backend()
        if dtype is None:
            dtype = Dtype.float64
        legs = [backend.convert_vector_space(leg) for leg in legs]

        return cls(data=backend.from_block_func(backend.block_random_normal, legs, dtype=dtype, sigma=sigma),
                   backend=backend, legs=legs, labels=labels)

    @classmethod
    def random_uniform(cls, legs: VectorSpace | list[VectorSpace], backend=None,
                       labels: list[str | None] = None, dtype: Dtype = Dtype.float64) -> Tensor:
        """Generate a tensor whose block-entries (i.e. the free parameters of tensors compatible with
        the symmetry) are drawn independently and uniformly.
        If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and imaginary part
        are drawn independently from [-1, 1].

        .. note ::
            This is not a well defined probability distribution on the space of symmetric tensors,
            since it depends on a choice of basis that defines what the individual uniformly drawn
            numbers mean.

        Parameters
        ----------
        legs : (list of) VectorSpace
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype
            The dtype for the tensor
        """
        if backend is None:
            backend = get_default_backend()
        legs = [backend.convert_vector_space(leg) for leg in legs]
        return cls(data=backend.from_block_func(backend.block_random_uniform, legs, dtype=dtype),
                   backend=backend, legs=legs, labels=labels)

    def diagonal(self) -> DiagonalTensor:
        return DiagonalTensor.from_tensor(self, check_offdiagonal=False)

    def idcs_fulfill_charge_rule(self, idcs: list[int]) -> bool:
        """Whether the given indices fulfill the charge rule.

        This is equivalent to asking if `self.to_dense_block()[idcs]` can *in principle*
        be non-zero, i.e. if the symmetry allows a non-zero entry at this position.

        Parameters
        ----------
        idcs : list of int
            One index per leg, each in the range ``0 <= idx < leg.dim``.
        """
        sectors = [leg.idx_to_sector(idx) for idx, leg in zip(idcs, self.legs)]
        if len(idcs) == 0:
            return True

        if len(idcs) == 1:
            coupled = sectors
        elif isinstance(self.symmetry, AbelianGroup):
            coupled = self.symmetry.fusion_outcomes(*sectors[2:])
            for s in sectors[2:]:
                # note: AbelianGroup allows us to assume that len(coupled) == 1, and continue with
                #       the unique fusion channel coupled[0]
                coupled = self.symmetry.fusion_outcomes(coupled[0], s)
        else:
            # will do this when there is a general implementation of fusion trees
            raise NotImplementedError

        return np.any(np.all(coupled == self.symmetry.trivial_sector[None, :], axis=1))

    # --------------------------------------------
    # Overriding methods from AbstractTensor
    # --------------------------------------------

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, legs: VectorSpace | list[VectorSpace],
             backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.float64) -> Tensor:
        """Empty Tensor with zero entries (not stored explicitly in most backends).

        Parameters
        ----------
        legs : (list of) VectorSpace
            The legs of the Tensor.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype, optional
            The data type of the Tensor entries.

        """
        legs = [backend.convert_vector_space(leg) for leg in legs]
        if backend is None:
            backend = get_default_backend()
        data = backend.zero_data(legs=legs, dtype=dtype)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    def apply_mask(self, mask: Mask, leg: int | str) -> Tensor:
        raise NotImplementedError  # TODO

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace]=None,
                     product_spaces_dual: list[bool]=None,
                     new_axes: list[int]=None) -> Tensor:
        """See tensors.combine_legs"""
        combine_leg_idcs = [self.get_leg_idcs(ll) for ll in legs]
        for leg_idcs in combine_leg_idcs:
            assert len(leg_idcs) > 0, "empty `legs` entry"

        product_spaces = self._combine_legs_make_ProductSpace(combine_leg_idcs, product_spaces, product_spaces_dual)
        combine_slices, new_axes, transp, perm_args = self._combine_legs_new_axes(combine_leg_idcs, new_axes)
        product_spaces = [product_spaces[p] for p in perm_args]  # permuted args such that new_axes is ascending

        if transp != tuple(range(len(transp))):
            res = self.permute_legs(transp)
        else:
            res = self.copy(deep=False)

        res_labels = list(res.labels)
        res_legs = list(res.legs)
        for (b, e), new_leg in zip(reversed(combine_slices), reversed(product_spaces)):  # descending b:e!
            res_labels[b:e] = [_combine_leg_labels(res_labels[b:e])]
            res_legs[b:e] = [new_leg]
        res_data = self.backend.combine_legs(res, combine_slices, product_spaces, new_axes, res_legs)
        return Tensor(res_data, backend=self.backend, legs=res_legs, labels=res_labels)

    def conj(self) -> Tensor:
        """See tensors.conj"""
        return Tensor(self.backend.conj(self), backend=self.backend, legs=[l.dual for l in self.legs],
                      labels=[_dual_leg_label(l) for l in self.shape._labels])

    def copy(self, deep=True) -> Tensor:
        if deep:
            return Tensor(data=self.backend.copy_data(self.data),
                          backend=self.backend,
                          legs=self.legs[:],
                          labels=self.labels[:])
        return Tensor(data=self.data,
                      backend=self.backend,
                      legs=self.legs,
                      labels=self.labels)

    def item(self) -> float | complex:
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')

    def norm(self) -> float:
        """See tensors.norm"""
        return self.backend.norm(self)

    def permute_legs(self, permutation: list[int | str]) -> Tensor:
        permutation = self.get_leg_idcs(permutation)
        assert len(permutation) == self.num_legs
        assert set(permutation) == set(range(self.num_legs))
        res_data = self.backend.permute_legs(self, permutation)
        return Tensor(res_data, backend=self.backend, legs=[self.legs[n] for n in permutation],
                    labels=[self.shape._labels[n] for n in permutation])

    def split_legs(self, *legs: int | str) -> Tensor:
        """See tensors.split_legs"""
        if len(legs) == 0:
            leg_idcs = [i for i, leg in enumerate(self.legs) if isinstance(leg, ProductSpace)]
        else:
            leg_idcs = sorted(self.get_leg_idcs(legs))
            for i in leg_idcs:
                if not isinstance(self.legs[i], ProductSpace):
                    raise ValueError(f'Leg {i} is not a ProductSpace.')
        old_legs = self.legs
        old_labels = self.labels
        new_legs = []
        new_labels = []
        start = 0
        for i in leg_idcs:
            new_legs.extend(old_legs[start:i])
            new_legs.extend(old_legs[i].spaces)
            new_labels.extend(old_labels[start:i])
            new_labels.extend(_split_leg_label(old_labels[i]))
            start = i + 1
        new_legs.extend(old_legs[start:])
        new_labels.extend(old_labels[start:])
        res_data = self.backend.split_legs(self, leg_idcs, new_legs)
        return Tensor(res_data, backend=self.backend, legs=new_legs, labels=new_labels)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> Tensor:
        """See tensors.squeeze_legs"""
        if legs is None:
            leg_idcs = [n for n, l in enumerate(self.legs) if l.is_trivial]
        else:
            leg_idcs = self.get_leg_idcs(legs)
            if not all(self.legs[idx].is_trivial for idx in leg_idcs):
                raise ValueError('Tried to squeeze non-trivial legs.')
        res_legs = [l for idx, l in enumerate(self.legs) if idx not in leg_idcs]
        if len(res_legs) == 0:
            raise ValueError("squeeze_legs() with no leg left. Use item instead.")
        res_labels = [label for idx, label in enumerate(self.labels) if idx not in leg_idcs]
        res_data = self.backend.squeeze_legs(self, leg_idcs)
        return Tensor(res_data, backend=self.backend, legs=res_legs, labels=res_labels)

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        block = self.backend.to_dense_block(self)
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> Tensor | float | complex:
        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = self.get_leg_idcs(legs2)
        if len(leg_idcs1) != len(leg_idcs2):
            raise ValueError('Must specify same number of legs')
        remaining_leg_idcs = [n for n in range(self.num_legs) if n not in leg_idcs1 and n not in leg_idcs2]
        if len(remaining_leg_idcs) == 0:
            return self.backend.trace_full(self, leg_idcs1, leg_idcs2)
        else:
            res_data = self.backend.trace_partial(self, leg_idcs1, leg_idcs2, remaining_leg_idcs)
            return Tensor(res_data, backend=self.backend,
                          legs=[self.legs[n] for n in remaining_leg_idcs],
                          labels=[self.labels[n] for n in remaining_leg_idcs])

    def _get_element(self, idcs: list[int]) -> float | complex:
        if not self.idcs_fulfill_charge_rule(idcs):
            return self.dtype.zero_scalar
        data_idcs = [leg.inverse_index_perm[idx] for leg, idx in zip(self.legs, idcs)]
        return self.backend.get_element(self, data_idcs)

    def _mul_scalar(self, other: complex) -> Tensor:
        return Tensor(self.backend.mul(other, self), backend=self.backend, legs=self.legs,
                      labels=self.labels)

    def _set_element(self, idcs: list[int], value: float | complex) -> None:
        if not self.idcs_fulfill_charge_rule(idcs):
            msg = f'Can not set element at indices {idcs}. They do not fulfill the charge rule.'
            raise ValueError(msg)
        data_idcs = [leg.inverse_index_perm[idx] for leg, idx in zip(self.legs, idcs)]
        self.data = self.backend.set_element(self, idcs=data_idcs, value=value)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    def almost_equal(self, other: Tensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
        if isinstance(other, DiagonalTensor):
            other = other.to_full_tensor()
        if not isinstance(other, Tensor):
            raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')
        if self.legs != other.legs:
            raise ValueError('Mismatching shapes')
        return get_same_backend(self, other).almost_equal(self, other, atol=atol, rtol=rtol)

    def inner(self, other: AbstractTensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, Tensor):
            return get_same_backend(self, other).inner(self, other, do_conj=do_conj, axs2=leg_order_2)
        if isinstance(other, ChargedTensor):
            # self is not charged and thus lives in the trivial sector of the parent space.
            # thus, only the components of other in the trivial sector contribute to the overlap.
            other = other._project_to_invariant()
            if other is None:
                # other has no part in the trivial sector
                return Dtype.common(self.dtype, other.dtype).zero_scalar
            # other is now a Tensor -> redirect to isinstance(other, Tensor) case
        if isinstance(other, DiagonalTensor):
            t1 = self.conj() if do_conj else self
            return t1.tdot(other, legs1=[0, 1], legs2=leg_order_2)
        raise TypeError(f'inner not supported for {type(self)} and {type(other)}')

    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> AbstractTensor:
        if isinstance(other, DiagonalTensor):
            # OPTIMIZE this could be done more efficiently in the backend...
            other = other.to_full_tensor()
        if isinstance(other, Tensor):
            backend = get_same_backend(self, other)
            return Tensor(data=backend.outer(self, other),
                          legs=self.legs + other.legs,
                          backend=backend,
                          labels=_get_result_labels(self.labels, other.labels, relabel1, relabel2))
        if isinstance(other, ChargedTensor):
            assert other.invariant_part.labels[-1] not in relabel2
            return ChargedTensor(
                invariant_part=self.outer(other.invariant_part, relabel1=relabel1, relabel2=relabel2),
                dummy_leg_state=other.dummy_leg_state
            )
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: AbstractTensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> AbstractTensor | float | complex:
        if isinstance(other, ChargedTensor):
            legs2 = other.get_leg_idcs(legs2)  # make sure we reference w.r.t. other, not other.invariant_part
            assert other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                       relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = other.get_leg_idcs(legs2)
        open_legs1 = [leg for idx, leg in enumerate(self.legs) if idx not in leg_idcs1]
        open_legs2 = [leg for idx, leg in enumerate(other.legs) if idx not in leg_idcs2]
        open_labels1 = [leg for idx, leg in enumerate(self.labels) if idx not in leg_idcs1]
        open_labels2 = [leg for idx, leg in enumerate(other.labels) if idx not in leg_idcs2]
        if len(leg_idcs1) != len(leg_idcs2):
            # checking this for leg_idcs* instead of legs* allows us to assume that they are both lists
            raise ValueError('Must specify the same number of legs for both tensors')
        if not all(self.legs[idx1].can_contract_with(other.legs[idx2]) for idx1, idx2 in zip(leg_idcs1, leg_idcs2)):
            raise ValueError('Incompatible legs.')
        # special case: outer()
        if len(leg_idcs1) == 0:
            return self.outer(other, relabel1, relabel2)
        backend = get_same_backend(self, other)
        if isinstance(other, DiagonalTensor):
            if len(leg_idcs1) == 2:
                # first contract leg that appears later in self.legs
                #  -> legs_idcs1[which_second] stays where it is
                which_first, which_second = (1, 0) if leg_idcs1[0] < leg_idcs1[1] else (0, 1)
                res = self.tdot(other, leg_idcs1[which_first], leg_idcs2[which_first],
                                relabel1=relabel1, relabel2=relabel2)
                return res.trace(leg_idcs1[which_second], -1)
            assert len(leg_idcs1) == 1 # have already excluded all other possibilities
            res = Tensor(
                data=backend.scale_axis(self, other, leg=leg_idcs2[0]),
                legs=self.legs[:leg_idcs1[0]] + other.legs[1 - leg_idcs2[0]] + self.legs[leg_idcs1[0] + 1:],
                backend=backend,
                labels=self.labels[:leg_idcs1[0]] + other.labels[1 - leg_idcs2[0]] + self.labels[leg_idcs1[0] + 1:]
            )
            # move scaled leg to the back
            perm = list(range(leg_idcs1[0])) + list(range(leg_idcs1[0] + 1, res.num_legs)) + [leg_idcs1[0]]
            return res.permute_legs(perm)
        if isinstance(other, Tensor):
            # special case: inner()
            if len(open_legs1) == 0 and len(open_legs2) == 0:
                return self.inner(other, do_conj=False, legs1=leg_idcs1, legs2=leg_idcs2)
            # have already checked special cases outer() and inner(), so backend does not have to do that.
            # remaining case: actual tensordot with non-trivial contraction and with open indices
            res_labels = _get_result_labels(open_labels1, open_labels2, relabel1, relabel2)
            res_data = backend.tdot(self, other, leg_idcs1, leg_idcs2)  # most of the work
            res_legs = open_legs1 + open_legs2
            if len(res_legs) == 0:
                return backend.data_item(res_data)
            else:
                return Tensor(res_data, backend=backend, legs=res_legs, labels=res_labels)
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')

    def _add_tensor(self, other: AbstractTensor) -> AbstractTensor:
        if isinstance(other, DiagonalTensor):
            # OPTIMIZE ?
            other = other.to_full_tensor()
        if isinstance(other, Tensor):
            backend = get_same_backend(self, other)
            other_order = self._input_checks_add_tensor(other)
            if other_order is not None:
                other = permute_legs(other, other_order)
            res_data = backend.add(self, other)
            return Tensor(res_data, backend=backend, legs=self.legs, labels=self.labels)
        raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")

    # --------------------------------------------
    # Internal utility methods
    # --------------------------------------------

    # TODO (JU): should this be implemented in AbstractTensor?
    # TODO: (JU) should we name it make_product_space ?
    #  make_ProductSpace to me suggests that i get (a subclass of) ProductSpace, not an instance.
    def make_ProductSpace(self, legs, **kwargs) -> ProductSpace:
        legs = self.get_legs(legs)
        # TODO: this should be something like class-attribute self.backend.ProductSpace
        #  JU: I think the attribute which has the same name as an existing class is confusing...
        #      At least i would call if ProductSpaceCls or similar.
        #  JU: What do you think about a make_product_space(cls, spaces, **kwargs) classmethod in VectorSpace?
        #      We could then call it here as ``legs[0].make_product_space(legs, **kwargs)``.
        #      By assigning the ProductSpace class-attribute (which is callable), you are effectively
        #      adding exactly such a method to the namespace of any VectorSpace instance, just
        #      with less clear names and docs.
        return legs[0].ProductSpace(legs, **kwargs)

    def _combine_legs_make_ProductSpace(self, combine_leg_idcs, product_spaces, product_spaces_dual):
        """Argument parsing for :meth:`combine_legs`: make missing ProductSpace legs.

        """
        n_comb = len(combine_leg_idcs)
        if product_spaces is None:
            product_spaces = [None] * n_comb
        elif len(product_spaces) != n_comb:
            raise ValueError("wrong len of `product_spaces`")
        else:
            product_spaces = list(product_spaces)
        product_spaces_dual = to_iterable_of_len(product_spaces_dual, n_comb)
        # make pipes as necessary
        for i, (leg_idcs, product_space) in enumerate(zip(combine_leg_idcs, product_spaces)):
            if product_space is None:
                product_spaces[i] = self.make_ProductSpace(leg_idcs, _is_dual=product_spaces_dual[i])
            else:
                # test for compatibility
                legs = [self.legs[a] for a in leg_idcs]
                if len(legs) != len(product_space.spaces):
                    raise ValueError("passed ProductSpace has wrong number of legs")
                if legs[0].is_dual != product_space.spaces[0].is_dual:
                    # TODO: should we just implicitly flip?
                    raise ValueError("Wrong `is_dual` flag of ProductSpace")
                for self_leg, given_space in zip(legs, product_space.spaces):
                    assert self_leg == given_space, f"Incompatible `self.legs` and product_spaces[{i:d}].spaces"
        return product_spaces

    def _combine_legs_new_axes(self, combine_leg_idcs, new_axes):
        """Figure out new_axes and how legs have to be transposed."""
        all_combine_leg_idcs = np.concatenate(combine_leg_idcs)
        non_combined_legs = np.array([a for a in range(self.num_legs) if a not in all_combine_leg_idcs])
        if new_axes is None:  # figure out default product_spaces
            first_cl = np.array([cl[0] for cl in combine_leg_idcs])
            new_axes = [(np.sum(non_combined_legs < a) + np.sum(first_cl < a)) for a in first_cl]
        else:  # test compatibility
            if len(new_axes) != len(combine_leg_idcs):
                raise ValueError("wrong len of `new_axes`")
            new_axes = list(new_axes)
            new_rank = len(combine_leg_idcs) + len(non_combined_legs)
            for i, a in enumerate(new_axes):
                if a < 0:
                    new_axes[i] = a + new_rank
                elif a >= new_rank:
                    raise ValueError("new_axis larger than the new number of legs")
        # construct transpose
        transpose = [[a] for a in non_combined_legs]
        perm_args = np.argsort(new_axes)
        cumsum = 0  # TODO @jhauschild this is not used?
        for s in perm_args:
            transpose.insert(new_axes[s], list(combine_leg_idcs[s]))
        new_axes = [new_axes[s] for s in perm_args]
        transposed_slices = [0] + list(np.cumsum([len(c) for c in transpose]))
        combine_slices = [(transposed_slices[a], transposed_slices[a+1]) for a in new_axes]
        transpose = sum(transpose, [])  # flatten: [a] + [b] = [a, b]
        return combine_slices, new_axes, tuple(transpose), perm_args


class ChargedTensor(AbstractTensor):
    """
    Tensors which transform under a given symmetry in a more general manner than the Tensor class,
    which represent tensors which are invariant under the symmetry.

    The ChargedTensor is a composite object consisting of an invariant Tensor with a
    designated dummy leg and a state that this leg is to be contracted with.
    This contraction is kept track of only symbolically.

    If the dummy leg has any non-trivial sectors, this composite object is not invariant under the
    symmetry.
    The decomposition into a symmetric tensor and an explicit state, however, allows us
    to still apply the machinery for symmetric tensors and exploit the sparsity.

    Parameters
    ----------
    invariant_part:
        The symmetry-invariant part. the dummy leg is the last of its legs.
    dummy_leg_state: block | None
        The state that the dummy leg is contracted with.
        Either a backend-specific block of shape ``(dummy_leg.dim,)``, or `None`,
        which is interpreted ``[1]`` if `dummmy_leg.dim == 1` and raises a `ValueError` otherwise.
        This state is in the "public" basis of the dummy_leg, i.e. not considering its `index_perm`.
    """
    _DUMMY_LABEL = '!'  # canonical label for the dummy leg

    def __init__(self, invariant_part: Tensor, dummy_leg_state=None):
        if dummy_leg_state is None:
            dtype = invariant_part.dtype
        else:
            dtype = Dtype.common(invariant_part.dtype,
                                 invariant_part.backend.get_dtype_from_data(dummy_leg_state))
        AbstractTensor.__init__(self, backend=invariant_part.backend, legs=invariant_part.legs[:-1],
                                labels=invariant_part.labels[:-1], dtype=dtype)
        self.invariant_part = invariant_part
        self.dummy_leg = invariant_part.legs[-1]
        if dummy_leg_state is None:
            if self.dummy_leg.dim != 1:
                raise ValueError('Can not infer state for a dummy leg with dim > 1')
        self.dummy_leg_state = dummy_leg_state

    def test_sanity(self):
        self.invariant_part.test_sanity()
        if self.dummy_leg_state is not None:
            assert self.backend.block_shape(self.dummy_leg_state) == (self.dummy_leg.dim,)
        AbstractTensor.test_sanity(self)

    # --------------------------------------------
    # Additional methods (not in AbstractTensor)
    # --------------------------------------------

    @classmethod
    def from_block_func(cls, func, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> ChargedTensor:
        inv = Tensor.from_block_func(func=func, legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], func_kwargs=func_kwargs,
                                     shape_kw=shape_kw, dtype=dtype)
        shape = (dummy_leg.dim,)
        if shape_kw is not None:
            block = func(**{shape_kw: shape}, **func_kwargs)
        else:
            block = func(shape, **func_kwargs)
        block = inv.backend.block_to_dtype(block, dtype)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=block)

    @classmethod
    def from_dense_block(cls, block, legs: list[VectorSpace], backend=None, dtype: Dtype=None,
                         labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5,
                         dummy_leg: VectorSpace = None, dummy_leg_state=None
                         ) -> ChargedTensor:
        """Convert a dense block of the backend to a ChargedTensor, if possible.

        TODO doc how and when it could fail

        Parameters
        ----------
        block :
            The data to be converted, a backend-specific block.
        legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`, optional
            The vectorspaces associated with legs of the tensors. Contains symmetry data.
            Does not contain the dummy leg.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
            The backend for the ChargedTensor.
        dtype : Dtype, optional
            The data type for the ChargedTensor. By default, this is inferred from the block.
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
            Does not contain a label for the dummy leg.
        atol, rtol : float
            Absolute and relative tolerance for checking if the block is actually symmetric.
        dummy_leg : VectorSpace
            The dummy leg. If not given, it is inferred from the block.
        dummy_leg_state : block
            The state on the dummy leg. Defaults to ``None``, which represents the state ``[1.]``.
        """
        if backend is None:
            backend = get_default_backend()
        if labels is None:
            labels = [None] * len(legs)
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        # add 1-dim axis for the dummy leg
        block = backend.block_add_axis(block, -1)
        if dummy_leg is None:
            dummy_leg = backend.infer_leg(block, legs + [None])
        if dummy_leg_state is not None and backend.block_shape(dummy_leg_state) != (1,):
            msg = f'Wrong shape of dummy_leg_state. Expected (1,). Got {backend.block_shape(dummy_leg_state)}'
            raise ValueError(msg)
        invariant_part = Tensor.from_dense_block(block, legs=legs + [dummy_leg], backend=backend,
                                                 dtype=dtype, labels=labels + [cls._DUMMY_LABEL],
                                                 atol=atol, rtol=rtol)
        return cls(invariant_part, dummy_leg_state=dummy_leg_state)

    @classmethod
    def from_numpy(cls, array: np.ndarray, legs: list[VectorSpace], backend=None, dtype: Dtype=None,
                   labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5,
                   dummy_leg: VectorSpace = None, dummy_leg_state=None
                   ) -> ChargedTensor:
        """
        Like from_dense_block but `array` and `dummy_leg_state` are numpy arrays.
        """
        if backend is None:
            backend = get_default_backend()
        block = backend.block_from_numpy(np.asarray(array))
        if dummy_leg_state is not None:
            dummy_leg_state = backend.block_from_numpy(np.asarray(dummy_leg_state))
        return cls.from_dense_block(block, legs=legs, backend=backend, dtype=dtype, labels=labels,
                                    atol=atol, rtol=rtol, dummy_leg=dummy_leg, dummy_leg_state=dummy_leg_state)

    @classmethod
    def from_numpy_func(cls, func, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> ChargedTensor:
        inv = Tensor.from_numpy_func(func=func, legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], func_kwargs=func_kwargs,
                                     shape_kw=shape_kw, dtype=dtype)
        shape = (dummy_leg.dim,)
        if shape_kw is not None:
            arr = func(**{shape_kw: shape}, **func_kwargs)
        else:
            arr = func(shape, **func_kwargs)
        block = inv.backend.block_from_numpy(arr)
        block = inv.backend.block_to_dtype(block, dtype)  # TODO (JU) add dtype arg directly to block_from_numpy?
        return ChargedTensor(invariant_part=inv, dummy_leg_state=block)

    @classmethod
    def from_tensor(cls, tens: Tensor) -> ChargedTensor:
        return ChargedTensor(invariant_part=add_trivial_leg(tens, pos=-1))

    @classmethod
    def from_two_dummy_legs(cls, invariant_part: Tensor, leg1: int, state1: Block, leg2: int,
                            state2: Block, convert_to_tensor_if_possible: bool = False
                            ) -> ChargedTensor | Tensor:
        leg1 = invariant_part.get_leg_idx(leg1)
        leg2 = invariant_part.get_leg_idx(leg2)
        first = min(leg1, leg2)
        second = max(leg1, leg2)
        other_legs = list(range(first)) \
                     + list(range(first + 1, second)) \
                     + list(range(second + 1, invariant_part.num_legs))
        invariant_part = invariant_part.permute_legs(other_legs + [leg1, leg2])
        invariant_part = invariant_part.combine_legs(-2, -1)
        product_space: ProductSpace = invariant_part.legs[-1]
        state = invariant_part.backend.fuse_states(
            state1=state1, state2=state2, space1=product_space.spaces[0], space2=product_space[1],
            product_space=product_space
        )
        res = ChargedTensor(invariant_part=invariant_part, dummy_leg_state=state)
        if convert_to_tensor_if_possible:
            try:
                res = res.convert_to_tensor()
            except ValueError:
                pass  # if its not possible to convert to Tensor, just leave it as ChargedTensor
        return res
    
    @classmethod
    def random_uniform(cls, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                       backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.float64,
                       dummy_leg_state=None) -> ChargedTensor:
        inv = Tensor.random_uniform(legs=legs + [dummy_leg], backend=backend, labels=labels + [cls._DUMMY_LABEL],
                                    dtype=dtype)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=dummy_leg_state)

    @classmethod
    def random_normal(cls) -> ChargedTensor:
        raise NotImplementedError  # TODO

    def project_to_invariant(self) -> Tensor:
        """Project self into the invariant subspace of the parent space.

        These are the contributions from trivial sectors in the dummy leg.

        See Also
        --------
        :meth:`_project_to_invariant`
        """
        res = self._project_to_invariant()
        if res is None:
            res = Tensor.zero(legs=self.legs, backend=self.backend, labels=self.labels, dtype=self.dtype)
        return res

    def _project_to_invariant(self) -> Tensor | None:
        """Internal version of :meth:`project_to_invariant`.

        If the dummy leg does not contain the trivial sector of the symmetry,
        the result is known to vanish exactly and ``None`` is returned.
        The "public" version :meth:`project_to_invariant` (no underscore!) instead returns a
        zero `Tensor` in this case.
        """
        raise NotImplementedError  # TODO

    def convert_to_tensor(self) -> Tensor:
        """If possible, convert self to a Tensor. Otherwise raise a ValueError.

        It is possible to convert a ChargedTensor to a Tensor if and only if the dummy leg only
        contains the trivial sector.

        See Also
        --------
        project_to_invariant
        """
        if not np.all(self.dummy_leg._sectors[:] == self.symmetry.trivial_sector[None, :]):
            raise ValueError('ChargedTensor with non-trivial charge could not be converted to Tensor.')
        if self.dummy_leg.dim == 1:
            return self._dummy_leg_state_item() * self.invariant_part.squeeze_legs(-1)
        state = Tensor.from_dense_block(self.dummy_leg_state, legs=[self.dummy_leg], backend=self.backend)
        return self.invariant_part.tdot(state, -1, 0)

    # --------------------------------------------
    # Overriding methods from AbstractTensor
    # --------------------------------------------
    
    def _repr_header_lines(self, indent: str) -> list[str]:
        lines = AbstractTensor._repr_header_lines(self, indent=indent)
        lines.append(f'{indent}* Dummy Leg: {self.dummy_leg}')
        lines.append(f'{indent}* Dummy Leg state:')
        if self.dummy_leg_state is None:
            lines.append(f'{indent}  [1.]')
        else:
            lines.extend(self.backend._block_repr_lines(self.dummy_leg_state, indent=indent + '  '),
                        max_width=70, max_lines=3)
        return lines

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
             backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.float64,
             dummy_leg_state=None) -> ChargedTensor:
        if isinstance(legs, VectorSpace):
            legs = [legs]
        if labels is None:
            labels = [None] * len(legs)
        invariant_part = Tensor.zero(legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], dtype=dtype)
        return cls(invariant_part=invariant_part, dummy_leg_state=dummy_leg_state)

    def apply_mask(self, mask: Mask, leg: int | str) -> ChargedTensor:
        raise NotImplementedError  # TODO

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace]=None,
                     product_spaces_dual: list[bool]=None,
                     new_axes: list[int]=None) -> ChargedTensor:
        legs = [self.get_leg_idcs(group) for group in legs]  # needed, since invariant_part does not have the same legs
        inv = self.invariant_part.combine_legs(*legs, product_spaces=product_spaces,
                                               product_spaces_dual=product_spaces_dual, new_axes=new_axes)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=self.dummy_leg_state)

    def conj(self) -> ChargedTensor:
        if self.dummy_leg_state is None:
            dummy_leg_state = None  # conj([1]) == [1]
        else:
            dummy_leg_state = self.backend.block_conj(self.dummy_leg_state)
        return ChargedTensor(invariant_part=self.invariant_part.conj(), dummy_leg_state=dummy_leg_state)

    def copy(self, deep=True) -> ChargedTensor:
        if deep:
            if self.dummy_leg_state is None:
                dummy_leg_state = None
            else:
                dummy_leg_state = self.backend.block_copy(self.dummy_leg_state)
            return ChargedTensor(invariant_part=self.invariant_part.copy(deep=True),
                                 dummy_leg_state=dummy_leg_state)
        return ChargedTensor(invariant_part=self.invariant_part, dummy_leg_state=self.dummy_leg_state)

    def item(self) -> float | complex:
        if not all(leg.dim == 1 for leg in self.invariant_part.legs[:-1]):
            raise ValueError('Not a scalar')
        return self.backend.block_item(self.to_dense_block())

    def norm(self) -> float:
        if self.dummy_leg.dim == 1:
            return self._dummy_leg_state_item() * self.invariant_part.norm()
        else:
            warnings.warn('Converting ChargedTensor to dense block for `norm`', stacklevel=2)
            return self.backend.block_norm(self.to_dense_block())

    def permute_legs(self, permutation: list[int | str]) -> ChargedTensor:
        permutation = self.get_leg_idcs(permutation)  # needed, since invariant_part does not have the same legs
        permutation = permutation + [-1]  # keep dummy leg at its position
        return ChargedTensor(invariant_part=self.invariant_part.permute_legs(permutation),
                             dummy_leg_state=self.dummy_leg_state)

    def split_legs(self, *legs: int | str) -> ChargedTensor:
        legs = [self.get_leg_idcs(group) for group in legs]  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.split_legs(*legs),
                             dummy_leg_state=self.dummy_leg_state)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> ChargedTensor:
        legs = self.get_leg_idcs(legs)  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.squeeze_legs(legs),
                             dummy_leg_state=self.dummy_leg_state)

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        invariant_block = self.backend.to_dense_block(self.invariant_part)
        if self.dummy_leg_state is None:
            block = self.backend.block_squeeze_legs(invariant_block, -1)
        else:
            block = self.backend.block_tdot(invariant_block, self.dummy_leg_state, [-1], [0])
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> ChargedTensor | float | complex:
        legs1 = self.get_leg_idcs(legs1)  # needed, since invariant_part does not have the same legs
        legs2 = self.get_leg_idcs(legs2)
        return ChargedTensor(invariant_part=self.invariant_part.trace(legs1, legs2),
                             dummy_leg_state=self.dummy_leg_state)

    def _get_element(self, idcs: list[int]) -> float | complex:
        if self.dummy_leg.dim == 1:
            return self._dummy_leg_state_item() * self.invariant_part._get_element(idcs + [0])
        masks = [Mask.from_indices([idx], leg) for idx, leg in zip(idcs, self.legs)]
        return self._getitem_apply_masks(masks, legs=list(range(self.num_legs))).item()

    def _mul_scalar(self, other: complex) -> ChargedTensor:
        # can choose to either "scale" the invariant part or the dummy_leg_state.
        # we might want to keep dummy_leg_state=None, so we scale the invariant part
        return ChargedTensor(
            invariant_part=self.invariant_part._mul_scalar(other),
            dummy_leg_state=self.dummy_leg_state
        )

    def _set_element(self, idcs: list[int], value: float | complex) -> None:
        try:
            factor = self._dummy_leg_state_item()
        except ValueError:
            msg = 'Can not set elements of ChargedTensor with non-trivial dummy leg'
            raise ValueError(msg) from None
        self.invariant_part._set_element(idcs + [0], value / factor)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    def almost_equal(self, other: AbstractTensor, atol: float = 0.00001, rtol: float = 1e-8) -> bool:
        if not isinstance(other, ChargedTensor):
            raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, ChargedTensor)

        if self.legs != other.legs:
            raise ValueError('Mismatching shapes')
        if self.dummy_leg != other.dummy_leg:
            return False

        if self.dummy_leg.dim == 1:
            factor = self._dummy_leg_state_item() / other._dummy_leg_state_item()
            return self.invariant_part.almost_equal(factor * other, atol=atol, rtol=rtol)
        else:
            # The decomposition into invariant part and non-invariant state is not unique,
            # so we cant just compare them individually.
            # OPTIMIZE (JU) is there a more efficient way?
            warnings.warn('Converting ChargedTensor to dense block for `almost_equal`', stacklevel=2)
            backend = get_same_backend(self, other)
            self_block = self.to_dense_block()
            other_block = other.to_dense_block(leg_order=match_leg_order(self, other))
            return backend.block_allclose(self_block, other_block)

    def inner(self, other: AbstractTensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, (DiagonalTensor, Tensor)):
            # other is not charged and thus lives in the trivial sector of the parent space.
            # thus, only the components of self in the trivial sector contribute to the overlap
            self_projected = self._project_to_invariant()
            if self_projected is None:
                return Dtype.common(self.dtype, other.dtype).zero_scalar
            return self_projected.inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, ChargedTensor):
            # OPTIMIZE could directly return 0 if the two tensors have completely different charges
            backend = get_same_backend(self, other)
            # contract the invariant parts with each other and convert do dense block
            inv1 = self.invariant_part.conj() if do_conj else self.invariant_part
            if leg_order_2 is None:
                leg_order_2 = list(range(other.num_legs))
            res = inv1.tdot(other.invariant_part, legs1=list(range(self.num_legs)), legs2=leg_order_2)
            res = res.to_dense_block()
            # contract with state on dummy leg of self
            if self.dummy_leg_state is None:
                res = backend.block_squeeze_legs(res, 0)
            else:
                state = backend.block_conj(self.dummy_leg_state) if do_conj else self.dummy_leg_state
                res = backend.block_tdot(state, res, 0, 0)
            # contract with state on dummy leg of other
            if other.dummy_leg_state is None:
                res = backend.block_squeeze_legs(res, 0)
            else:
                res = backend.block_tdot(res, other.dumm_leg_state, 0, 0)
            return backend.block_item(res)
        raise TypeError(f'inner not supported for {type(self)} and {type(other)}')
    
    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> AbstractTensor:
        assert self.invariant_part.labels[-1] not in relabel1
        if isinstance(other, DiagonalTensor):
            other = other.to_full_tensor()
        if isinstance(other, Tensor):
            inv_part = self.invariant_part.outer(other, relabel1=relabel1, relabel2=relabel2)
            # permute dummy leg to the back
            self_normal = list(range(self.num_legs))
            self_dummy = [self.num_legs]
            other_normal = list(range(self.num_legs + 1, self.num_legs + 1 + other.num_legs))
            inv_part = inv_part.permute_legs(self_normal + other_normal + self_dummy)
            return ChargedTensor(invariant_part=inv_part, dummy_leg_state=self.dummy_leg_state)
        if isinstance(other, ChargedTensor):
            assert other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.invariant_part.outer(other.invariant_part, relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor.from_two_dummy_legs(
                invariant_part, leg1=self.num_legs, state1=self.dummy_leg_state, leg2=-1,
                state2=other.dummy_leg_state, convert_to_tensor_if_possible=True
            )
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: AbstractTensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> AbstractTensor | float | complex:
        legs1 = self.get_leg_idcs(legs1)  # make sure we reference w.r.t. self, not self.invariant_part
        assert self.invariant_part.labels[-1] not in relabel1
        if isinstance(other, (Tensor, DiagonalTensor)):
            # In both of these cases, the main work is done by tdot(self.invariant_part, other, ...)
            invariant_part = self.invariant_part.tdot(other, legs1=legs1, legs2=legs2,
                                                      relabel1=relabel1, relabel2=relabel2)
            # move dummy leg to the back
            num_legs_from_self = self.num_legs - len(legs1)
            permutation = list(range(num_legs_from_self)) \
                          + list(range(num_legs_from_self + 1, invariant_part.num_legs)) \
                          + [num_legs_from_self]
            invariant_part = invariant_part.permute_legs(permutation)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=self.dummy_leg_state)
        if isinstance(other, ChargedTensor):
            legs2 = other.get_leg_idcs(legs2)  # make sure we referecne w.r.t. other
            assert other.invariant_part.labels[-1] not in relabel2
            invariant = self.invariant_part.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                                 relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor.from_two_dummy_legs(
                invariant, leg1=self.num_legs - len(legs1), state1=self.dummy_leg_state,
                leg2=-1, state2=other.dummy_leg_state, convert_to_tensor_if_possible=True
            )
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')

    def _add_tensor(self, other: AbstractTensor) -> ChargedTensor:
        if not isinstance(other, ChargedTensor):
            raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")
        if self.dummy_leg != other.dummy_leg:
            raise ValueError('Can not add ChargedTensors with different dummy legs')
        try:
            factor = self._dummy_leg_state_item()
        except ValueError:
            msg = 'Can not add ChargedTensors unless dummy_leg is one-dimensional'
            raise ValueError(msg) from None
        return ChargedTensor(invariant_part=self.invariant_part + factor * other.invariant_part,
                             dummy_leg_state=self.dummy_leg_state)

    # --------------------------------------------
    # Internal utility methods
    # --------------------------------------------

    def _dummy_leg_state_item(self) -> float | complex:
        """If the dummy leg is one-dimensonal, return the single item of the dummy_leg_state.
        Otherwise raise a ValueError"""
        if self.dummy_leg.dim != 1:
            raise ValueError('Leg is not one-dimensional')
        if self.dummy_leg_state is None:
            return 1.
        else:
            return self.backend.block_item(self.dummy_leg_state)


# TODO where to put this? where to doc it?
BOTH = object()  # sentinel value for the default behavior of DiagonalTensor.apply_mask


class DiagonalTensor(AbstractTensor):
    r"""Special case of a tensor with two legs that is diagonal in the computational basis.

    TODO more "elementwise" methods (exp, log, sqrt, ...?)

    Parameters
    ----------
    data
        The numerical data ("free parameters") comprising the tensor. type is backend-specific
    first_leg : VectorSpace
        The first leg of this tensor.
    second_leg_dual : bool
        Wether the second leg is the dual of the first (default) or the same.
        If ``True``, the result is a tensor :math:`\sum_n c_n \vert n \rangle\langle n \vert`.
        Otherwise it is :math:`\sum_n c_n \vert n \rangle \otimes \vert n \rangle`.
    backend: :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    def __init__(self, data, first_leg: VectorSpace, second_leg_dual: bool = True, backend=None,
                 labels: list[str | None] = None):
        self.data = data
        self.second_leg_dual = second_leg_dual
        second_leg = first_leg.dual if second_leg_dual else first_leg
        if backend is None:
            backend = get_default_backend()
        dtype = backend.get_dtype_from_data(data)
        AbstractTensor.__init__(self, legs=[first_leg, second_leg], backend=backend, labels=labels,
                                dtype=dtype)

    def test_sanity(self) -> None:
        super().test_sanity()
        self.backend.test_data_sanity(self, is_diagonal=True)

    # --------------------------------------------
    # Additional methods (not in AbstractTensor)
    # --------------------------------------------
    
    # TODO (JU) clearly define when legs need to be converted to backend-specific spaces
    #      if possible (?), AbstractTensor.__init__ would be nice.
    #      Tensor classmethods do it.
    #      If that ends up being the place of choice, the classmethods below need to do it too.
    
    @cached_property
    def diag_block(self) -> Block:
        return self.backend.diagonal_to_block(self)

    @cached_property
    def diag_numpy(self) -> np.ndarray:
        block = self.diag_block
        return self.backend.block_to_numpy(block)

    @classmethod
    def eye(cls, first_leg: VectorSpace, backend=None, labels: list[str | None] = None,
            dtype: Dtype = Dtype.float64) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        return cls.from_block_func(
            func=backend.ones_block,
            first_leg=first_leg, second_leg_dual=True, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype),
        )

    @classmethod
    def from_block_func(cls, func, first_leg: VectorSpace, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        if shape_kw is not None:
            def block_func(shape):
                block = func(**{shape_kw: shape}, **func_kwargs)
                return backend.block_to_dtype(block, dtype)
        else:
            def block_func(shape):
                block = func(shape, **func_kwargs)
                return backend.block_to_dtype(block, dtype)

        data = backend.diagonal_from_block_func(block_func, leg=first_leg)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def from_diag_block(cls, diag: Block, first_leg: VectorSpace, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        assert backend.block_shape(diag) == (first_leg.dim,)
        data = backend.diagonal_from_block(diag, leg=first_leg)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def from_diag_numpy(cls, diag: np.ndarray, first_leg: VectorSpace, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        return cls.from_diag_block(diag=backend.block_from_numpy(diag), first_leg=first_leg,
                                   second_leg_dual=second_leg_dual, backend=backend, labels=labels)

    @classmethod
    def from_numpy_func(cls, func, first_leg: VectorSpace, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        if shape_kw is not None:
            def block_func(shape):
                arr = func(**{shape_kw: shape}, **func_kwargs)
                block = backend.block_from_numpy(arr)
                return backend.block_to_dtype(block, dtype)
        else:
            def block_func(shape):
                arr = func(shape, **func_kwargs)
                block = backend.block_from_numpy(arr)
                return backend.block_to_dtype(block, dtype)
        data = backend.diagonal_from_block_func(block_func, leg=first_leg)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def from_tensor(cls, tens: Tensor, check_offdiagonal: bool = True) -> DiagonalTensor:
        if tens.num_legs != 2:
            raise ValueError
        if tens.legs[1] == tens.legs[0]:
            second_leg_dual = False
        elif tens.legs[1].can_contract_with(tens.legs[0]):
            second_leg_dual=True
        else:
            raise ValueError('Second leg must be equal to or dual of first leg')
        data = tens.backend.diagonal_data_from_full_tensor(tens, check_offdiagonal=check_offdiagonal)
        return cls(data=data, first_leg=tens.legs[0], second_leg_dual=second_leg_dual,
                   backend=tens.backend, labels=tens.labels)

    @classmethod
    def random_normal(cls, first_leg: VectorSpace = None, second_leg_dual: bool = None,
                      mean: DiagonalTensor = None, sigma: float = 1., backend=None,
                      labels: list[str | None] = None, dtype: Dtype = None
                      ) -> DiagonalTensor:
        r"""Generate a tensor from the normal distribution.

        Like :meth:`Tensor.random_normal`."""
        if mean is not None:
            for name, val in zip(['first_leg', 'second_leg_dual', 'backend', 'labels'],
                                 [first_leg, second_leg_dual, backend, labels]):
                if val is not None:
                    msg = f'{name} argument to Tensor.random_normal was ignored, because mean was given.'
                    warnings.warn(msg)
            if dtype is None:
                dtype = mean.dtype
            return mean + cls.random_normal(
                first_leg=mean.legs[0], second_leg_dual=mean.second_leg_dual, mean=None,
                sigma=sigma, backend=mean.backend, labels=mean.labels, dtype=dtype
            )

        if backend is None:
            backend = get_default_backend()
        if dtype is None:
            dtype = Dtype.float64
        data = backend.diagonal_from_block_func(backend.block_random_normal, leg=first_leg,
                                                func_kwargs=dict(dtype=dtype))
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def random_uniform(cls, first_leg: VectorSpace, second_leg_dual: bool = True, backend=None,
                       labels: list[str | None] = None, dtype: Dtype = Dtype.float64
                       ) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        data = backend.diagonal_from_block_func(backend.block_random_uniform, leg=first_leg,
                                                func_kwargs=dict(dtype=dtype))
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    def to_full_tensor(self) -> Tensor:
        """Forget about diagonal structure and convert to a Tensor"""
        return Tensor(
            data=self.backend.full_data_from_diagonal_tensor(self),
            legs=self.legs, backend=self.backend, labels=self.labels
        )

    def _apply_mask_both_legs(self, mask: Mask) -> DiagonalTensor:
        """Apply the same mask to both legs."""
        raise NotImplementedError  # TODO

    def _binary_operand(self, other: bool | Number | DiagonalTensor, func, operand: str, bool_inputs: bool):
        """Utility function for a shared implementation of binary functions, whose second argument
        may be a scalar ("to be broadcast") or a DiagonalTensor,
        e.g. the dunder methods, ``__mul__, __eq__, __lt__, __pow__, ...``.

        Parameters
        ----------
        other
            If ``bool_inputs is True``, either a bool or a DiagonalTensor of dtype bool.
            If ``bool_inputs is False``, either a number or a DiagonalTensor of non-bool dtype.
            If ``bool_inputs is None``, any of the above.
        func
            The function as with signature
            ``func(self_block: Block, other_block_or_other: bool | Number | Block) -> Block``
        operand
            A string representation of the operand, used in error messages
        bool_inputs
            If the inputs need to be bools, see `other`
        """
        if isinstance(other, bool):
            other_is_scalar = True
            other_type_ok = (bool_inputs is not False)
        elif isinstance(other, Number):
            other_is_scalar = True
            other_type_ok = (bool_inputs is not True)
        elif isinstance(other, DiagonalTensor):
            if self.legs != other.legs:
                raise ValueError('Incompatible legs')
            other_is_scalar = False
            is_bool = other.dtype == Dtype.bool
            other_type_ok = (bool_inputs is not False) if is_bool else (bool_inputs is not True)
        else:
            other_type_ok = False
        self_is_bool = self.dtype == Dtype.bool
        self_type_ok = (bool_inputs is not False) if self_is_bool else (bool_inputs is not True)

        if not (other_type_ok and self_type_ok):
            self_type = f'DiagonalTensor[dtype={self.dtype}]'
            if isinstance(other, DiagonalTensor):
                other_type = f'DiagonalTensor[dtype={other.dtype}]'
            else:
                other_type = type(other)
            msg = f'Invalid types for operand "{operand}": {self_type} and {other_type}'
            raise TypeError(msg)

        if other_is_scalar:
            return self._elementwise_unary(func=lambda block: func(block, other))
        return self._elementwise_binary(other, func=func)
            

    def _elementwise_unary(self, func, func_kwargs={}, maps_zero_to_zero: bool = False) -> DiagonalTensor:
        """Wrap backend.diagonal_elementwise_unary

        func(a: Block, **kwargs) -> Block"""
        data = self.backend.diagonal_elementwise_unary(
            self, func, func_kwargs=func_kwargs, maps_zero_to_zero=maps_zero_to_zero
        )
        return DiagonalTensor(data=data, first_leg=self.legs[0], second_leg_dual=self.second_leg_dual,
                              backend=self.backend, labels=self.labels)

    def _elementwise_binary(self, other: DiagonalTensor, func, func_kwargs={},
                            partial_zero_is_identity: bool = False,
                            partial_zero_is_zero: bool = False) -> DiagonalTensor:
        """Wrap backend.diagonal_elementwise_binary

        func(a: Block, b: Block, **kwargs) -> Block"""
        assert isinstance(other, DiagonalTensor)
        if self.legs[0] != other.legs[0] or self.second_leg_dual != other.second_leg_dual:
            raise ValueError('Incompatible legs!')
        backend = get_same_backend(self, other)
        data = backend.diagonal_elementwise_binary(
            self, other, func=func, func_kwargs=func_kwargs,
            partial_zero_is_identity=partial_zero_is_identity,
            partial_zero_is_zero=partial_zero_is_zero
        )
        return DiagonalTensor(data, first_leg=self.legs[0], second_leg_dual=self.second_leg_dual,
                              backend=backend, labels=self.labels)

    def __abs__(self):
        return self._elementwise_unary(func=operator.abs, maps_zero_to_zero=True)

    def __bool__(self):
        if self.dtype == Dtype.bool:
            return bool(self.item())
        raise TypeError(f'DiagonalTensor of dtype {self.dtype} can not be cast to bool.')

    def __and__(self, other):
        return self._binary_operand(self, other, func=operator.and_, operand='&', bool_inputs=True)

    def __ge__(self, other):
        return self._binary_operand(self, other, func=operator.ge, operand='>=', bool_inputs=False)

    def __gt__(self, other):
        return self._binary_operand(self, other, func=operator.gt, operand='>', bool_inputs=False)

    def __le__(self, other):
        return self._binary_operand(self, other, func=operator.le, operand='<=', bool_inputs=False)

    def __lt__(self, other):
        return self._binary_operand(self, other, func=operator.lt, operand='<', bool_inputs=False)

    def __ne__(self, other):
        return self._binary_operand(self, other, func=operator.ne, operand='!=', bool_inputs=None)

    def __or__(self, other):
        return self._binary_operand(self, other, func=operator.or_, operand='|', bool_inputs=True)

    def __pow__(self, other):
        if isinstance(other, DiagonalTensor):
            return self._elementwise_binary(other, func=operator.pow)
        elif isinstance(other, Number):
            return self._elementwise_unary(func=lambda block: block ** other, maps_zero_to_zero=True)
        raise TypeError(f'Invalid types for operand "**": {type(self)} and {type(other)}')

    def __rand__(self, other):
        # other & self  ==  self & other
        return self.__and__(other)
    
    def __ror__(self, other):
        # other | self  ==  self | other
        return self.__or__(other)

    def __rpow__(self, other):
        if isinstance(other, Number):
            return self._elementwise_unary(func=lambda block: other ** block)
        raise TypeError(f'Invalid types for operand "**": {type(other)} and {type(self)}')

    def __rxor__(self, other):
        return self.__xor__(other)
    
    def __xor__(self, other):
        return self._binary_operand(self, other, func=operator.xor, operand='^', bool_inputs=True)

    # --------------------------------------------
    # Overriding methods from AbstractTensor
    # --------------------------------------------

    def _getitem_apply_masks(self, masks: list[Mask], legs: list[int]) -> Tensor | DiagonalTensor:
        if len(masks) == 2:
            if masks[0].same_mask_action(masks[1]):
                return self._apply_mask_both_legs(masks[0])
        warnings.warn('Converting DiagonalTensor to Tensor in order to apply mask', stacklevel=2)
        return self.to_full_tensor()._getitem_apply_masks(masks, legs)

    def __eq__(self, other):
        return self._binary_operand(self, other, func=operator.eq, operand='==', bool_inputs=None)

    def __mul__(self, other):
        if isinstance(other, DiagonalTensor):
            # _elementwise_binary performs input checks.
            return self._elementwise_binary(other, func=operator.mul, partial_zero_is_zero=True)
        return AbstractTensor.__mul__(self, other)

    def __truediv__(self, other):
        if isinstance(other, DiagonalTensor):
            # _elementwise_binary performs input checks.
            return self._elementwise_binary(other, func=operator.truediv)
        return AbstractTensor.__mul__truediv____(self, other)

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, first_leg: VectorSpace, second_leg_dual: bool = True, backend=None,
             labels: list[str | None] = None, dtype: Dtype = Dtype.float64) -> DiagonalTensor:
        if backend is None:
            backend = get_default_backend()
        data = backend.zero_diagonal_data(leg=first_leg, dtype=dtype)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    def apply_mask(self, mask: Mask, leg: int | str = BOTH) -> DiagonalTensor | Tensor:
        # TODO amend docstring of superclass
        if leg is BOTH:
            return self._apply_mask_both_legs(mask)
        
        raise NotImplementedError  # TODO

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace]=None,
                     product_spaces_dual: list[bool]=None,
                     new_axes: list[int]=None) -> Tensor:
        warnings.warn('Converting DiagonalTensor to Tensor in order to combine legs', stacklevel=2)
        return self.to_full_tensor().combine_legs(
            *legs, product_spaces=product_spaces, product_spaces_dual=product_spaces_dual,
            new_axes=new_axes
        )

    def conj(self) -> DiagonalTensor:
        return DiagonalTensor(data=self.backend.conj(self), first_leg=self.legs[0].dual,
                              second_leg_dual=self.second_leg_dual, backend=self.backend,
                              labels=[_dual_leg_label(l) for l in self.shape._labels])

    def copy(self, deep=True) -> DiagonalTensor:
        if deep:
            return DiagonalTensor(data=self.backend.copy_data(self.data),
                                  first_leg=self.legs[0],
                                  second_leg_dual=(self.legs[1].is_dual != self.legs[0].is_dual),
                                  backend=self.backend,
                                  labels=self.labels[:])
        return DiagonalTensor(data=self.data,
                              first_leg=self.legs[0],
                              second_leg_dual=(self.legs[1].is_dual != self.legs[0].is_dual),
                              backend=self.backend,
                              labels=self.labels)

    def item(self) -> bool | float | complex:
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')
        
    def norm(self) -> float:
        return self.backend.norm(self)

    def permute_legs(self, permutation: list[int | str]) -> DiagonalTensor:
        permutation = self.get_leg_idcs(permutation)
        return DiagonalTensor(data=self.data, first_leg=self.legs[permutation[0]],
                              second_leg_dual=self.second_leg_dual, backend=self.backend,
                              labels=[self.shape._labels[n] for n in permutation])

    def split_legs(self, *legs: int | str) -> Tensor:
        warnings.warn('Converting DiagonalTensor to Tensor in order to split legs', stacklevel=2)
        return self.to_full_tensor().split_legs(*legs)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> NoReturn:
        raise TypeError(f'{type(self)} does not support squeeze_legs')

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        # need to fill in the off-diagonal zeros anyway, so we may as well use to_full_tensor first.
        return self.to_full_tensor().to_dense_block(leg_order)

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> float | complex:
        # TODO should we check for invalid inputs? -> I think yes...
        return self.backend.diagonal_tensor_trace_full(self)

    def _get_element(self, idcs: list[int]) -> bool | float | complex:
        if idcs[0] != idcs[1]:
            return self.dtype.zero_scalar
        # TODO in tests, do the consistency check that it really doesnt matter which leg is used here
        data_idx = self.legs[0].index_perm[idcs[0]]
        return self.backend.get_element_diagonal(self, data_idx)

    def _set_element(self, idcs: list[int], value: bool | float | complex) -> None:
        if idcs[0] != idcs[1]:
            raise IndexError('Off-diagonal entry can not be set for DiagonalTensor')
        self.data = self.backend.set_element_diagonal(self, idcs[0], value)

    def _mul_scalar(self, other: complex) -> DiagonalTensor:
        return self._elementwise_unary(lambda block: self.backend.block_mul(other, block),
                                       maps_zero_to_zero=True)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------
    
    def almost_equal(self, other: Tensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
        if isinstance(other, DiagonalTensor):
            raise NotImplementedError  # TODO
        if isinstance(other, Tensor):
            return self.to_full_tensor().almost_equal(other, atol=atol, rtol=rtol)
        raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')

    def inner(self, other: AbstractTensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        t1 = self.conj() if do_conj else self
        return t1.tdot(other, legs1=[0, 1], legs2=leg_order_2)

    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> AbstractTensor:
        if isinstance(other, DiagonalTensor):
            warnings.warn('Converting DiagonalTensors to Tensors for outer', stacklevel=2)
            other = other.to_full_tensor()
        if isinstance(other, (Tensor, ChargedTensor)):
            return self.to_full_tensor().outer(other, relabel1=relabel1, relabel2=relabel2)
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: AbstractTensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> AbstractTensor | float | complex:
        if isinstance(other, ChargedTensor):
            legs2 = other.get_leg_idcs(legs2)  # make sure we reference w.r.t. other, not other.invariant_part
            assert other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                       relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        legs1 = self.get_leg_idcs(legs1)
        legs2 = self.get_leg_idcs(legs2)
        # deal with special cases
        if len(legs1) == 0:
            return self.outer(other, relabel1=relabel1, relabel2=relabel2)
        if len(legs1) == 2:
            which_first, which_second = (0, 1) if legs2[0] < legs2[1] else (1, 0)
            res = self.tdot(other, legs1[which_first], legs2[which_first],
                            relabel1=relabel1, relabel2=relabel2)
            # legs1[which_second] is not at position 0, legs2[which_second] has not moved
            return res.trace(0, legs2[which_second])
        # now we know that exactly one leg should be contracted
        if not self.legs[legs1[0]].can_contract_with(other.legs[legs2[0]]):
            raise ValueError('Incompatible legs.')
        backend = get_same_backend(self, other)
        if isinstance(other, DiagonalTensor):
            # note that contractible legs guarantee that self.legs[0] and other.legs[0] are either
            # equal or mutually dual and we can safely use elementwise_binary, no matter which of
            # the legs are actually contracted.
            return DiagonalTensor(
                data=backend.diagonal_elementwise_binary(self, other, operator.mul, partial_zero_is_zero=True),
                first_leg=self.legs[1 - legs1[0]],
                second_leg_dual=(self.second_leg_dual == other.second_leg_dual),
                backend=backend,
                labels=[self.labels[1 - legs1[0]], other.labels[1 - legs2[0]]]
            )
        if isinstance(other, Tensor):
            res = Tensor(
                data=backend.scale_axis(other, self, leg=legs2[0]),
                legs=other.legs[:legs2[0]] + [self.legs[1 - legs1[0]]] + other.legs[legs2[0] + 1:],
                backend=backend,
                labels=other.labels[:legs2[0]] + [self.labels[1 - legs1[0]]] + other.labels[legs2[0] + 1:]
            )
            # move scaled leg to the front
            return res.permute_legs([legs2[0]] + list(range(legs2[0])) + list(range(legs2[0] + 1, res.num_legs)))
            
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')


class Mask(AbstractTensor):
    r"""A boolean mask that can be used to project a leg.

    As an AbstractTensor, the first leg is the larger leg and the second is a "slice" of it.

    TODO put this piece of doc in the right place:
    Via `tdot`, the mask can be applied only to the *dual* of `large_leg`.
    With the  `apply_*` methods however, a mask can be applied to both `large_leg` and its dual.

    Parameters
    ----------
    data
        The numerical data (i.e. boolean flags) comprising the mask. type is backend-specific
    large_leg : VectorSpace
        The larger leg, the source/domain of the projection.
    small_leg : VectorSpace
        The smaller leg, the target/codomain of the projection.
        Should have the same :attr:`is_dual` as `large_leg`.
    backend: :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    def __init__(self, data, small_leg: VectorSpace, large_leg: VectorSpace, backend=None,
                 labels: list[str | None] = None):
        self.data = data
        AbstractTensor.__init__(self, legs=[small_leg, large_leg], backend=backend, labels=labels, dtype=Dtype.bool)

    def test_sanity(self) -> None:
        super().test_sanity()
        assert isinstance(self.data, self.backend.DataCls)
        # self.backend.test_data_sanity(self)  # TODO modify this
        assert self.legs[0].is_dual != self.legs[1].is_dual
        # TODO check if legs[0] is a slice of legs[1]

    # --------------------------------------------
    # Additional methods (not in AbstractTensor)
    # --------------------------------------------

    @property
    def large_leg(self) -> VectorSpace:
        return self.spaces[1]

    @property
    def small_leg(self) -> VectorSpace:
        return self.spaces[0]

    @classmethod
    def from_diagonal_tensor(cls, d: DiagonalTensor) -> Mask:
        """"""
        assert d.dtype == Dtype.bool
        ...  # TODO

    @classmethod
    def from_flat_block(cls, mask: np.ndarray, large_leg: VectorSpace) -> Mask:
        # TODO better name
        # TODO remember to use large_leg.index_perm!
        ...  # TODO
        
    @classmethod
    def from_flat_numpy(cls, mask: np.ndaray, large_leg: VectorSpace) -> Mask:
        # TODO better name
        # TODO remember to use large_leg.index_perm!
        ...  # TODO

    @classmethod
    def from_indices(cls, indices: list[int] | np.ndarray, large_leg: VectorSpace) -> Mask:
        # TODO remember to use large_leg.index_perm!
        ...  # TODO

    @classmethod
    def from_slice(cls, s: slice, large_leg: VectorSpace) -> Mask:
        # TODO remember to use large_leg.index_perm!
        ...  # TODO

    def same_mask_action(self, other: Mask) -> bool:
        """A mask can act on both the large_leg or its dual.
        This function determines if this action is the same."""
        raise NotImplementedError  # TODO

    def __eq__(self, other) -> bool:
        if isinstance(other, Mask):
            return self.large_leg == other.large_leg and self.same_mask_action(other)
        raise TypeError(f'{type(self)} does not support == comparison with {type(other)}')

    # --------------------------------------------
    # Overriding methods from AbstractTensor
    # --------------------------------------------

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------

    # TODO implement all
    
    def apply_mask(self, mask: Mask, leg: int | str) -> Mask:
        # TODO what if masking makes the "large_leg" smaller than the small leg?
        #  -> think about _take_mask_indices as well
        raise NotImplementedError

    def copy(self, deep=True) -> Mask:
        if deep:
            return Mask(data=self.backend.copy_data(self.data),
                        small_legs=self.legs[0],
                        large_leg=self.legs[1],
                        backend=self.backend,
                        labels=self.labels[:])
        return Mask(data=self.data,
                    small_legs=self.legs[0],
                    large_leg=self.legs[1],
                    backend=self.backend,
                    labels=self.labels)

    def _get_element(self, idcs: list[int]) -> bool:
        raise NotImplementedError  # TODO

    def _set_element(self, idcs: list[int], value: bool) -> None:
        raise NotImplementedError  # TODO

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    # TODO implement them


# ##################################
# API functions
# ##################################

# TODO there should be an operation that converts only one or some of the legs to dual
#  i.e. vectorization of density matrices
#  formally, this is contraction with the (co-)evaluation map, aka cup or cap

# TODO (JU) find a good way to write type hints for these, having in mind the possible combinations
#           of AbstractTensor-subtypes.

def add_trivial_leg(tens, pos: int = -1):
    raise NotImplementedError  # TODO


def almost_equal(t1: AbstractTensor, t2: AbstractTensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
    """Checks if two tensors are equal up to numerical tolerance.

    The blocks of the two tensors are compared.
    The tensors count as almost equal if all block-entries, i.e. all their free parameters
    individually fulfill `abs(a1 - a2) <= atol + rtol * abs(a1)`.

    In the non-symmetric case, this is equivalent to e.g. ``numpy.allclose``.
    In the symmetric case, it is a close analogue.

    .. note ::
        The definition is not symmetric, so there may be edge-cases where
        ``almost_equal(t1, t2) != almost_equal(t2, t1)``
    """
    return t1.almost_equal(t2, atol=atol, rtol=rtol)


def combine_legs(t: AbstractTensor,
                 *legs: list[int | str],
                 product_spaces: list[ProductSpace]=None,
                 product_spaces_dual: list[bool]=None,
                 new_axes: list[int]=None):
    """
    Combine (multiple) groups of legs on a tensor to (multiple) ProductSpaces.

    Parameters
    ----------
    t :
        The tensor whose legs should be combined.
    *legs : tuple of list of {int | str}
        One or more groups of legs.
    product_spaces : list of ProductSpace
        For optimization, the resulting ProductSpace instances per leg group can be passed. keyword only.
        By default, they are recomputed.
    product_spaces_dual : list of bool
        For each group, whether the resulting ProductSpace should be dual. keyword only.
        Per default, it has the same is_dual as the first (by appearance in `legs`) of the combined legs.

    Result
    ------
    combined :
        A tensor with combined legs.
        The order of legs is the same as for `t`, except that ``legs[n][0]`` is replaced by the
        ``n``-th ``ProductSpace`` and the other ``legs[n][1:]`` are left out.

    See Also
    --------
    split_legs
    """
    return t.combine_legs(*legs, product_spaces=product_spaces, product_spaces_dual=product_spaces_dual,
                          new_axes=new_axes)


def conj(t: AbstractTensor) -> AbstractTensor:
    """
    The conjugate of t, living in the dual space.
    Labels are adjuste as `'p'` -> `'p*'` and `'p*'` -> `'p'`
    """
    return t.conj()


def inner(t1: AbstractTensor, t2: AbstractTensor, do_conj: bool = True,
          legs1: list[int | str] = None, legs2: list[int | str]  = None) -> complex:
    """
    Inner product ``<t1|t2>`` of two tensors.

    Parameters
    ----------
    t1, t2 :
        The two tensors
    do_conj : bool
        If true (default), `t1` lives in the same space as `t2` and will be conjugated to form ``<t1|t2>``.
        Otherwise, `t1` lives in the dual space of `t2` and will not be conjugated.
    legs1, legs2 : list of int or str, optional
        Specify which legs are to be contracted with which ones.
        If both are ``None`` (default), legs are identified "by label" in strict label mode or "by order"
        in lax label mode, like in :meth:`match_leg_order`.
        Otherwise, ``legs1[n]`` of ``t1`` is contracted with ``legs2[n]`` of ``t2``, where
        a single ``None``/unspecified list is interpreted as ``list(range(num_legs))``.
    """
    return t1.inner(t2, do_conj=do_conj, legs1=legs1, legs2=legs2)


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs"""
    if isinstance(obj, AbstractTensor):
        return all(d == 1 for d in obj.shape)
    # TODO use ``isinstance(obj, Number)`` instead?
    if isinstance(obj, (int, float, complex, Integral)):  # Integral for np.int64()
        return True
    # TODO (JU): should we just return False here?
    raise TypeError(f'Type not supported for is_scalar: {type(obj)}')


def norm(t: AbstractTensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.norm()


def outer(t1: AbstractTensor, t2: AbstractTensor, relabel1: dict[str, str] = None,
          relabel2: dict[str, str] = None) -> AbstractTensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    return t1.outer(t2, relabel1=relabel1, relabel2=relabel2)


def permute_legs(t: AbstractTensor, permutation: list[int]) -> AbstractTensor:
    """Change the order of legs of a Tensor.
    """
    # TODO: also have an inplace version?
    return t.permute_legs(permutation)


def split_legs(t: AbstractTensor, *legs: int | str) -> Tensor:
    """
    Split legs that were previously combined.

    Up to a possible ``permute_legs``, this is the inverse operation of ``combine_legs``.
    It is the exact inverse if the original non-fused legs were contiguous and ordered in ``t_before_combine.legs``.

    Parameters
    ----------
    t :
        The tensor whose legs should be split.
    legs : tuple of {int | str}, or None
        Which legs should be split. If none are specified, all ``ProductSpace``s are split.

    See Also
    --------
    combine_legs
    """
    # TODO inplace version?
    return t.split_legs(*legs)


def squeeze_legs(t: AbstractTensor, legs: int | str | list[int | str] = None) -> AbstractTensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If legs is None (default), all trivial legs are squeezed
    """
    return t.squeeze_legs(legs=legs)


def tdot(t1: AbstractTensor, t2: AbstractTensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None
         ) -> AbstractTensor | float | complex:
    """
    Contraction of two tensors.

    TODO more details, e.g. that legs need to match, legorder of the result (like numpy)

    Parameters
    ----------
    t1 : AbstractTensor
    t2 : AbstractTensor
    legs1 : int or str or list of int or list of str
        the leg(s) on t1 to be contracted, referenced either by index or by label
    legs2 : int or str of list of int or list of str
        the leg(s) on t2 to be contracted, referenced either by index or by label
    relabel1 : dict
        labels of the result are determined as if t1 had been relabelled by this mapping before contraction
    relabel2
        labels of the result are determined as if t2 had been relabelled by this mapping before contraction

    Returns
    -------

    """
    return t1.tdot(t2, legs1=legs1, legs2=legs2, relabel1=relabel1, relabel2=relabel2)


def trace(t: AbstractTensor, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
          ) -> AbstractTensor | float | complex:
    """
    Trace over one or more pairs of legs, that is contract these pairs.
    """
    return t.trace(legs1=legs1, legs2=legs2)


def zero_like(tens: AbstractTensor) -> AbstractTensor:
    return tens.zero(backend=tens.backend, legs=tens.legs, labels=tens.labels, dtype=tens.dtype)


def eye_like(tens: AbstractTensor) -> Tensor | DiagonalTensor:
    if not isinstance(tens, (Tensor, DiagonalTensor)):
        raise TypeError(f'eye is not defined for type {type(tens)}')
    if tens.num_legs % 2 != 0:
        raise ValueError('eye is not defined for an odd number of legs')
    legs_1 = tens.legs[:tens.num_legs // 2]
    legs_2 = tens.legs[tens.num_legs // 2:]
    for l1, l2 in zip(legs_1, legs_2):
        if not l1.can_contract_with(l2):
            if len(legs_1) == 1:
                msg = 'Second leg must be the dual of the first leg'
            else:
                msg = 'Second half of legs must be the dual of the first half'
            raise ValueError(msg)
    return tens.eye(legs=legs_1, backend=tens.backend, labels=tens.labels, dtype=tens.dtype)


# ##################################
# utility functions
# ##################################


def get_same_backend(*tensors: AbstractTensor, error_msg: str = 'Incompatible backends.'
                     ) -> AbstractBackend:
    """If all tensors have the same backend, return it. Otherwise raise a ValueError"""
    try:
        backend = tensors[0].backend
    except IndexError:
        raise ValueError('expected at least one tensor') from None
    if not all(tens.backend == backend for tens in tensors):
        raise ValueError(error_msg)
    return backend


def match_leg_order(t1: AbstractTensor, t2: AbstractTensor) -> list[int] | None:
    """Utility function to determine how to match legs of two tensors.

    In strict label mode, returns the permutation ``perm`` that matches the legs "by label",
    i.e. such that ``t2.labels[perm[n]] == t1.labels[n]``.
    In other words, it is the order of legs on ``t2``, such that they match those on ``t1``.
    None is returned instead of a trivial permutation.
    In lax label mode, we want to match the legs "by order" and hence always return None.
    """
    if config.strict_labels:
        if not (t1.is_fully_labelled and t2.is_fully_labelled):
            raise ValueError('Fully labelled tensors are required in strict label mode')
        if t1.shape._labels == t2.shape._labels:
            return None
        return t2.get_leg_idcs(t1.labels)
    else:
        return None


# ##################################
# "private" helper functions
# ##################################


T = TypeVar('T')


def _parse_idcs(idcs: T | Sequence[T | Ellipsis], length: int, fill: T = slice(None, None, None)
                ) -> list[T]:
    """Parse a single index or sequence of indices to a list of given length by replacing Ellipsis
    and missing entries  at the back with `fill`.

    For invalid input, an IndexError is raised instead of ValueError, since this is a helper
    function for __getitem__ and __setitem__.
    """
    idcs = list(to_iterable(idcs))
    if Ellipsis in idcs:
        where = idcs.index(Ellipsis)
        first = idcs[:where]
        last = idcs[where + 1:]
        if Ellipsis in last:
            raise IndexError("Ellipsis ('...') may not appear multiple times.")
        num_fill = length - len(first) - len(last)
        if num_fill < 0:
            got = len(idcs) - 1  # dont count the ellipsis
            raise IndexError(f'Too many indices. Expected {length}. Got {got}.')
        return first + [fill] * num_fill + last
    else:
        num_fill = length - len(idcs)
        if num_fill < 0:
            raise IndexError(f'Too many indices. Expected {length}. Got {len(idcs)}.')
        return idcs + [fill] * num_fill


class _TensorIndexHelper:
    """A helper class that redirects __getitem__ and __setitem__ to an AbstractTensor.

    See :meth:`~tenpy.linalg.tensors.AbstractTensor.index`.
    """
    def __init__(self, tensor: AbstractTensor, which_legs: list[int | str]):
        self.tensor = tensor
        self.which_legs = which_legs

    def transform_idcs(self, idcs):
        idcs = _parse_idcs(idcs, length=len(self.which_legs))
        res = [slice(None, None, None) for _ in range(self.tensor.num_legs)]
        for which_leg, idx in zip(self.which_legs, idcs):
            res[self.tensor.get_leg_idx(which_leg)] = idx
        return res

    def __getitem__(self, idcs):
        return self.tensor.__getitem__(self.transform_idcs(idcs))

    def __setitem__(self, idcs, value):
        return self.tensor.__setitem__(self.transform_idcs(idcs), value)


def _dual_leg_label(label: str) -> str:
    """the label that a leg should have after conjugation"""
    if label is None:
        return None
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def _combine_leg_labels(labels: list[str | None]) -> str:
    """the label that a combined leg should have"""
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


def _split_leg_label(label: str) -> list[str | None]:
    """undo _combine_leg_labels, i.e. recover the original labels"""
    if label.startswith('(') and label.endswith(')'):
        labels = label[1:-1].split('.')
        return [None if l.startswith('?') else l for l in labels]
    else:
        raise ValueError('Invalid format for a combined label')


def _get_result_labels(legs1: list[str | None], legs2: list[str | None],
                       relabel1: dict[str, str] | None, relabel2: dict[str, str] | None) -> list[str]:
    """
    Utility function to combine two lists of leg labels, such that they can appear on the same tensor.
    Labels are changed by the mappings relabel1 and relabel2.
    Any conflicting labels (after relabelling) are dropped
    """
    if relabel1 is None:
        labels1 = legs1
    else:
        labels1 = [relabel1.get(leg.label, leg.label) for leg in legs1]
    if relabel2 is None:
        labels2 = legs2
    else:
        labels2 = [relabel2.get(leg.label, leg.label) for leg in legs2]
    conflicting = [label for label in labels1 if label in labels2]
    labels = labels1 + labels2
    if conflicting:
        # TODO issue warning?
        #  JU: maybe logger.debug?
        labels = [None if label in conflicting else label for label in labels]
    return labels
