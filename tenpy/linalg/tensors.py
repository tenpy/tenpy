"""

TODO put this in the proper place:
The labels should not include the characters `.`, `?` or `!`.
"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
from numbers import Integral
import numpy as np
import warnings
from functools import cached_property

from .misc import duplicate_entries, force_str_len, join_as_many_as_possible
from .dummy_config import config
from .symmetries.spaces import VectorSpace, ProductSpace
from .backends.backend_factory import get_default_backend
from .backends.abstract_backend import Dtype
from ..tools.misc import to_iterable

__all__ = ['AbstractTensor', 'Tensor', 'ChargedTensor', 'DiagonalTensor', 'tdot', 'outer', 'inner',
           'permute_legs', 'trace', 'conj', 'combine_legs', 'combine_leg', 'split_legs', 'split_leg',
           'is_scalar', 'squeeze_legs', 'norm', 'get_same_backend', 'Dtype', 'zero_like',
           ]

# svd, qr, eigen, exp, log, ... are implemented in matrix_operations.py


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

    def __init__(self, legs: list[VectorSpace], backend, labels: list[str | None] | None):
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

    def test_sanity(self):
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.symmetry == self.symmetry for l in self.legs)
        assert len(self.legs) == self.shape.num_legs == self.num_legs > 0
        for leg in self.legs:
            assert isinstance(leg, (self.backend.VectorSpaceCls, self.backend.ProductSpaceCls))
        self.shape.test_sanity()

    @cached_property
    def parent_space(self) -> VectorSpace:
        if self.num_legs == 1:
            return self.legs[0]
        else:
            return self.legs[0].ProductSpace(self.legs)

    @property
    def size(self) -> int:
        """The total number of entries, i.e. the dimension of the space of tensors on the same space
        if symmetries were ignored"""
        return self.parent_space.dim

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors with the same legs"""
        return self.parent_space.num_parameters

    @property
    def is_fully_labelled(self) -> bool:
        return self.shape.is_fully_labelled

    def has_label(self, label: str, *more: str) -> bool:
        return label in self.shape._labels and all(l in self.shape._labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        if not self.is_fully_labelled:
            return False
        if len(labels) != len(self.shape._labels):
            return False
        return set(labels) == set(self.shape._labels)

    def set_labels(self, labels: list[str | None]):
        self.shape.set_labels(labels)

    @property
    def labels(self) -> list[str | None]:
        return self.shape.labels

    @labels.setter
    def labels(self, value):
        self.shape.set_labels(value)

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

    @abstractmethod
    def copy(self, deep=True):
        ...

    @abstractmethod
    def item(self):
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        ...

    def _repr_leg_components(self, max_len: int) -> list:
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

    def __repr__(self):
        indent = '  '
        lines = [f'{self.__class__.__name__}(']
        lines.extend(self._repr_header_lines(indent=indent))
        lines.extend(self.backend._data_repr_lines(self.data, indent=indent, max_width=70, max_lines=20))
        lines.append(')')
        return "\n".join(lines)

    @abstractmethod
    def _mul_scalar(self, other: complex):
        ...

    def __getitem__(self, item):
        raise TypeError('Tensor object is not subscriptable')

    def __neg__(self):
        return self._mul_scalar(-1)

    def __pos__(self):
        return self

    def __eq__(self, other):
        msg = f'{type(self)} does not support == comparison. Use tenpy.almost_equal instead.'
        raise TypeError(msg)

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __sub__(self, other):
        ...

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

    @abstractmethod
    def is_real(self) -> bool:
        ...

    def __float__(self):
        if not self.is_real():
            warnings.warn("converting complex to real, only return real part!", stacklevel=2)
        return self.item().real

    def __complex__(self):
        return complex(self.item())

    @abstractmethod
    def to_dense_block(self, leg_order: list[int | str] = None):
        """Convert tensor to a dense (i.e. no longer exploiting the symmetry structure) block,
        i.e. a numpy ndarray if the backend is a NumpyBlockBackend, or a torch Tensor of
        the backend is a TorchBlockBackend"""
        ...

    def to_numpy_ndarray(self, leg_order: list[int | str] = None, numpy_dtype=None) -> np.ndarray:
        """Convert to a numpy array"""
        block = self.to_dense_block(leg_order=leg_order)
        return self.backend.block_to_numpy(block, numpy_dtype=numpy_dtype)

    @classmethod
    @abstractmethod
    def zero(cls, legs: VectorSpace | list[VectorSpace], backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.complex128) -> Tensor:
        """A zero tensor"""
        ...

    @abstractmethod
    def tdot(self, other: AbstractTensor,
             legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
             relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        """See tensors.tdot"""
        ...

    @abstractmethod
    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> AbstractTensor:
        """See tensors.outer"""
        ...

    @abstractmethod
    def inner(self, other: AbstractTensor, do_conj: bool = True,
              legs1: list[int | str] = None, legs2: list[int | str]  = None) -> complex:
        """See tensors.inner"""
        ...

    @abstractmethod
    def permute_legs(self, permutation: list[int]) -> AbstractTensor:
        """See tensors.transpose"""
        ...

    @abstractmethod
    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> AbstractTensor | float | complex:
        """See tensors.trace"""
        ...

    @abstractmethod
    def conj(self) -> AbstractTensor:
        """See tensors.conj"""
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
    def split_legs(self, legs: list[int | str] = None) -> AbstractTensor:
        """See tensors.split_legs"""
        ...

    @abstractmethod
    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> AbstractTensor:
        """See tensors.squeeze_legs"""
        ...

    @abstractmethod
    def norm(self) -> float:
        """See tensors.norm"""
        ...

    @abstractmethod
    def almost_equal(self, other: AbstractTensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
        """See tensors.almost_equal"""
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
        AbstractTensor.__init__(self, backend=backend, legs=legs, labels=labels)
        self.data = data
        assert isinstance(data, self.backend.DataCls)

    @property
    def dtype(self) -> Dtype:
        return self.backend.get_dtype_from_data(self.data)

    def test_sanity(self):
        super().test_sanity()
        assert isinstance(self.data, self.backend.DataCls)
        self.backend.test_data_sanity(self)

    def copy(self, deep=True):
        if deep:
            return Tensor(data=self.backend.copy_data(self.data),
                          backend=self.backend,
                          legs=self.legs[:],
                          labels=self.labels[:])
        return Tensor(data=self.data,
                      backend=self.backend,
                      legs=self.legs,
                      labels=self.labels)

    def item(self):
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')

    def _mul_scalar(self, other: complex):
        return Tensor(self.backend.mul(other, self), backend=self.backend, legs=self.legs,
                      labels=self.labels)

    def __add__(self, other):
        if isinstance(other, Tensor):
            backend = get_same_backend(self, other)
            other_order = match_legs(self, other)
            if other_order is not None:
                other = permute_legs(other, other_order)
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
            res_data = backend.add(self, other)
            return Tensor(res_data, backend=backend, legs=self.legs, labels=self.labels)
        return NotImplemented

    def __sub__(self, other):
        # TODO make this the default implementation at AbstractTensor level?
        if isinstance(other, AbstractTensor):
            return self.__add__(other._mul_scalar(-1))
        return NotImplemented

    def is_real(self):
        return self.backend.is_real(self)

    def to_dense_block(self, leg_order: list[int | str] = None):
        block = self.backend.to_dense_block(self)
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

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
    def from_dense_block(cls, block, legs: list[VectorSpace], backend=None, dtype: Dtype=None,
                         labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Tensor:
        """Convert a dense block of the backend to a Tensor.

        If the block is not symmetric under the symmetry (specified by the legs), i.e. if
        ``not allclose(block, projected, atol, rtol)``, a ValueError is raised.

        TODO document how the sectors are expected to be embedded, i.e. which slices correspond to which charge.
        TODO support non-canonical embedding?

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
        """
        if backend is None:
            backend = get_default_backend()
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        data = backend.from_dense_block(block, legs=legs, atol=atol, rtol=rtol)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def zero(cls, legs: VectorSpace | list[VectorSpace],
             backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.complex128) -> Tensor:
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

    @classmethod
    def eye(cls, legs: VectorSpace | list[VectorSpace], backend=None,
            labels: list[str | None] = None, dtype: Dtype = Dtype.complex128) -> Tensor:
        """The identity map from one group of legs to their duals.

        Parameters
        ----------
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs : (list of) VectorSpace
            *Half* of the legs of the result. The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype, optional
            The data type of the Tensor entries.

        """
        if backend is None:
            backend = get_default_backend()
        legs = to_iterable(legs)
        legs = [backend.convert_vector_space(leg) for leg in legs]
        data = backend.eye_data(legs=legs, dtype=dtype)
        legs = legs + [leg.dual for leg in legs]
        return cls(data=data, backend=backend, legs=legs, labels=labels)

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
    def random_uniform(cls, legs: VectorSpace | list[VectorSpace], backend=None,
                       labels: list[str | None] = None, dtype: Dtype = Dtype.complex128) -> Tensor:
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

        return cls(data=backend.from_block_func(backend.block_random_uniform, legs, dtype=Dtype),
                   backend=backend, legs=legs, labels=labels)

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
            default to `Dtype.complex128`.
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
            dtype = Dtype.complex128
        legs = [backend.convert_vector_space(leg) for leg in legs]

        return cls(data=backend.from_block_func(backend.block_random_normal, legs, dtype=dtype, sigma=sigma),
                   backend=backend, legs=legs, labels=labels)

    def tdot(self, other: AbstractTensor,
             legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
             relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        if isinstance(other, ChargedTensor):
            # make sure that legs2 are interpreted w.r.t. other, not other.invariant_part
            legs2 = other.get_leg_idcs(legs2)
            assert other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                       relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        elif isinstance(other, DiagonalTensor):
            raise NotImplementedError  # TODO
        elif not isinstance(other, Tensor):
            raise TypeError(f'tdot not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, Tensor)

        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = other.get_leg_idcs(legs2)
        if len(leg_idcs1) != len(leg_idcs2):
            # checking this for leg_idcs* instead of legs* allows us to assume that they are both lists
            raise ValueError('Must specify the same number of legs for both tensors')
        if not all(self.legs[idx1].can_contract_with(other.legs[idx2]) for idx1, idx2 in zip(leg_idcs1, leg_idcs2)):
            raise ValueError('Incompatible legs.')
        backend = get_same_backend(self, other)
        open_legs1 = [leg for idx, leg in enumerate(self.legs) if idx not in leg_idcs1]
        open_legs2 = [leg for idx, leg in enumerate(other.legs) if idx not in leg_idcs2]
        open_labels1 = [leg for idx, leg in enumerate(self.labels) if idx not in leg_idcs1]
        open_labels2 = [leg for idx, leg in enumerate(other.labels) if idx not in leg_idcs2]
        # check for special cases, such that backend.tdot doesn't have to do that
        # special case: inner()
        if len(open_legs1) == 0 and len(open_legs2) == 0:
            # TODO add args to inner() with implicit transpose and specify that it shouldn't use conj(a)
            a = self.permute_legs(leg_idcs1)
            b = other.permute_legs(leg_idcs2)
            return a.conj().inner(b)
        # special case: outer()
        if len(leg_idcs1) == 0:
            return self.outer(other, relabel1, relabel2)
        res_labels = _get_result_labels(open_labels1, open_labels2, relabel1, relabel2)

        res_data = backend.tdot(self, other, leg_idcs1, leg_idcs2)  # most of the work

        res_legs = open_legs1 + open_legs2
        if len(res_legs) == 0:
            return backend.data_item(res_data)
        else:
            return Tensor(res_data, backend=backend, legs=res_legs, labels=res_labels)

    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        if isinstance(other, ChargedTensor):
            assert other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.outer(other.invariant_part, relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        elif isinstance(other, DiagonalTensor):
            raise NotImplementedError  # TODO
        elif not isinstance(other, Tensor):
            raise TypeError(f'outer not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, Tensor)

        backend = get_same_backend(self, other)
        res_labels = _get_result_labels(self.labels, other.labels, relabel1, relabel2)
        res_data = backend.outer(self, other)
        return Tensor(res_data, backend=backend, legs=self.legs + other.legs, labels=res_labels)

    def inner(self, other: AbstractTensor, do_conj: bool = True,
              legs1: list[int | str] = None, legs2: list[int | str]  = None) -> complex:
        if self.num_legs != other.num_legs:
            raise ValueError('Tensors need to have the same number of legs')
        # TODO: if co_conj=False, should we not match e.g. 'p*' with 'p'?
        leg_order_2 = match_legs(self, other, legs1=legs1, legs2=legs2)
        if leg_order_2 is None:
            leg_order_2 = list(range(other.num_legs))
        if do_conj:
            are_compatible = all(self.legs[n1] == other.legs[n2] for n1, n2 in enumerate(leg_order_2))
        else:
            are_compatible = all(self.legs[n1].can_contract_with(other.legs[n2]) for n1, n2 in enumerate(leg_order_2))
        if not are_compatible:
            raise ValueError('Incompatible legs')
        backend = get_same_backend(self, other)

        if isinstance(other, ChargedTensor):
            res_invariant = self.tdot(other.invariant_part, legs1=list(range(self.num_legs)), legs2=leg_order_2)
            res = ChargedTensor(res_invariant, dummy_leg_state=other.dummy_leg_state)
            return res.item()
        elif isinstance(other, DiagonalTensor):
            raise NotImplementedError  # TODO
        elif not isinstance(other, Tensor):
            raise TypeError(f'inner not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, Tensor)

        res = backend.inner(self, other, do_conj=do_conj, axs2=leg_order_2)
        return res

    def permute_legs(self, permutation: list[int | str]) -> AbstractTensor:
        permutation = self.get_leg_idcs(permutation)
        assert len(permutation) == self.num_legs
        assert set(permutation) == set(range(self.num_legs))
        res_data = self.backend.permute_legs(self, permutation)
        return Tensor(res_data, backend=self.backend, legs=[self.legs[n] for n in permutation],
                    labels=[self.shape._labels[n] for n in permutation])

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> AbstractTensor | float | complex:
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

    def conj(self) -> AbstractTensor:
        """See tensors.conj"""
        return Tensor(self.backend.conj(self), backend=self.backend, legs=[l.dual for l in self.legs],
                      labels=[_dual_leg_label(l) for l in self.shape._labels])

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

    # TODO (JU): should this be implemented in AbstractTensor?
    # TODO: (JU) should we name it make_product_space ?
    #  make_ProductSpace to me suggests that i get (a subclass of) ProductSpace, not an instance.
    def make_ProductSpace(self, legs, **kwargs):
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
        # default arguments for product_spaces_dual
        if product_spaces_dual is None:
            product_spaces_dual = [self.legs[leg_idcs[0]].is_dual for leg_idcs in combine_leg_idcs]
        elif product_spaces_dual is True:
            product_spaces_dual = [True] * n_comb
        elif product_spaces_dual is False:
            product_spaces_dual = [False] * n_comb
        elif len(product_spaces_dual) != n_comb:
            raise ValueError("wrong len of `product_spaces_dual`")
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

    def split_legs(self, *legs: int | str) -> AbstractTensor:
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

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> AbstractTensor:
        """See tensors.squeeze_legs"""
        if legs is None:
            leg_idcs = [n for n, l in enumerate(self.legs) if l.is_trivial]
        else:
            leg_idcs = self.get_leg_idcs(legs)
            if not all(self.legs[idx].is_trivial for idx in leg_idcs):
                raise ValueError('Tried to squeeze non-trivial legs.')
        res_legs = [l for idx, l in enumerate(self.legs) if idx not in leg_idcs]
        if len(res_legs) == 0:
            raise ValueError("squeeze_legs() with no leg left.")
            # TODO: should we return self.item() in this case?
            #  JU: I dont think so, because code might rely on getting a AbstractTensor.
            #      There might be a case for a Scalar(AbstractTensor) type...
            #      the ArrayAPI does this too, any "scalar" output needs to be a 0D array.
        res_labels = [label for idx, label in enumerate(self.labels) if idx not in leg_idcs]
        res_data = self.backend.squeeze_legs(self, leg_idcs)
        return Tensor(res_data, backend=self.backend, legs=res_legs, labels=res_labels)

    def norm(self) -> float:
        """See tensors.norm"""
        return self.backend.norm(self)

    def almost_equal(self, other: AbstractTensor, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
        if not isinstance(other, Tensor):
            raise TypeError(f'tdot not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, Tensor)

        if self.legs != other.legs:
            raise ValueError('Mismatching shapes')
        backend = get_same_backend(self, other)
        return backend.almost_equal(self, other, atol=atol, rtol=rtol)


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
    """
    _DUMMY_LABEL = '!'  # canonical label for the dummy leg

    def __init__(self, invariant_part: Tensor, dummy_leg_state=None):
        AbstractTensor.__init__(self, backend=invariant_part.backend, legs=invariant_part.legs[:-1],
                                labels=invariant_part.labels[:-1])
        self.invariant_part = invariant_part
        self.dummy_leg = invariant_part.legs[-1]
        if dummy_leg_state is None:
            if self.dummy_leg.dim != 1:
                raise ValueError('Can not infer state for a dummy leg with dim > 1')
        self.dummy_leg_state = dummy_leg_state

    @property
    def dtype(self) -> Dtype:
        return self.invariant_part.dtype

    def test_sanity(self):
        self.invariant_part.test_sanity()
        if self.dummy_leg_state is not None:
            assert self.backend.block_shape(self.dummy_leg_state) == (self.dummy_leg.dim,)
        AbstractTensor.test_sanity(self)

    def copy(self, deep=True):
        if deep:
            if self.dummy_leg_state is None:
                dummy_leg_state = None
            else:
                dummy_leg_state = self.backend.block_copy(self.dummy_leg_state)
            return ChargedTensor(invariant_part=self.invariant_part.copy(deep=True),
                                 dummy_leg_state=dummy_leg_state)
        return ChargedTensor(invariant_part=self.invariant_part, dummy_leg_state=self.dummy_leg_state)

    def item(self):
        if not all(leg.dim == 1 for leg in self.invariant_part.legs[:-1]):
            raise ValueError('Not a scalar')
        return self.backend.block_item(self.to_dense_block())

    def _mul_scalar(self, other: complex):
        # can chosse to either "scale" the invariant part or the dummy_leg_state.
        # we might want to keep dummy_leg_state=None, so we scale the invariant part
        return ChargedTensor(
            invariant_part=self.invariant_part._mul_scalar(other),
            dummy_leg_state=self.dummy_leg_state
        )

    def __add__(self, other):
        if isinstance(other, ChargedTensor):
            if self.dummy_leg != other.dummy_leg:
                raise ValueError('can not add ChargedTensors with different dummy legs')

            if self.dummy_leg.dim == 1:
                factor = self._dummy_leg_state_item() / other._dummy_leg_state_item()
            elif self.backend.block_allclose(self.dummy_leg_state, other.dummy_legs_state):
                factor = 1.
            else:
                msg = 'Can not add ChargedTensors with different dummy_leg_states unless dummy_leg is one-dimensional'
                raise ValueError(msg)
            return ChargedTensor(invariant_part=self.invariant_part + factor * other.invariant_part,
                                 dummy_leg_state=self.dummy_leg_state)

        elif isinstance(other, AbstractTensor):
            raise TypeError(f"unsupported operand type(s) for +: 'ChargedTensor' and '{type(other)}'")
        return NotImplemented

    def __sub__(self, other):
        # TODO make this the default implementation at AbstractTensor level?
        if isinstance(other, AbstractTensor):
            return self.__add__(other._mul_scalar(-1))
        return NotImplemented

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

    def is_real(self):
        dummy_leg_state_real = self.dummy_leg_state is None or self.backend.block_is_real(self.dummy_leg_state)
        return self.backend.is_real(self.invariant_part) and dummy_leg_state_real

    def to_dense_block(self, leg_order: list[int | str] = None):
        invariant_block = self.backend.to_dense_block(self.invariant_part)
        if self.dummy_leg_state is None:
            block = self.backend.block_squeeze_legs(invariant_block, -1)
        else:
            block = self.backend.block_tdot(invariant_block, self.dummy_leg_state, [-1], [0])
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

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
    def zero(cls, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
             backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.complex128,
             dummy_leg_state=None):
        if isinstance(legs, VectorSpace):
            legs = [legs]
        if labels is None:
            labels = [None] * len(legs)
        invariant_part = Tensor.zero(legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], dtype=dtype)
        return cls(invariant_part=invariant_part, dummy_leg_state=dummy_leg_state)

    @classmethod
    def from_numpy_func(cls, func, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None):
        inv = Tensor.from_numpy_func(func=func, legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], func_kwargs=func_kwargs,
                                     shape_kw=shape_kw, dtype=dtype)
        shape = (dummy_leg.dim,)
        if shape_kw is not None:
            arr = func(**{shape_kw: shape}, **func_kwargs)
        else:
            arr = func(shape, **func_kwargs)
        block = inv.backend.block_from_numpy(block)
        block = inv.backend.block_to_dtype(block, dtype)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=block)

    @classmethod
    def from_block_func(cls, func, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None):
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
    def random_uniform(cls, legs: VectorSpace | list[VectorSpace], dummy_leg: VectorSpace,
                       backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.complex128,
                       dummy_leg_state=None):
        inv = Tensor.random_uniform(legs=legs + [dummy_leg], backend=backend, labels=labels + [cls._DUMMY_LABEL],
                                    dtype=dtype)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=dummy_leg_state)

    @classmethod
    def random_normal(cls):
        raise NotImplementedError  # TODO

    def _dummy_leg_state_item(self) -> float:
        """If the dummy leg is one-dimensonal, return the single item of the dummy_leg_state.
        Otherwise raise a ValueError"""
        if self.dummy_leg.dim != 1:
            raise ValueError('Leg is not one-dimensional')
        if self.dummy_leg_state is None:
            return 1.
        else:
            return self.backend.block_item(self.dummy_leg_state)

    def almost_equal(self, other: AbstractTensor, atol: float = 0.00001, rtol: float = 1e-8) -> bool:
        if not isinstance(other, ChargedTensor):
            raise TypeError(f'tdot not supported for types {type(self)} and {type(other)}.')
        # can now assume that isinstance(other, ChargedTensor)

        if self.legs != other.legs:
            raise ValueError('Mismatching shapes')
        if self.dummy_leg != other.dummy_leg:
            return False

        if self.dummy_leg.dim == 1:
            factor = self._dummy_leg_state_item() / other._dummy_leg_state_item()
            return self.invariant_part.almost_equal(factor * other, atol=atol, rtol=rtol)
        elif self.backend.block_allclose(self.dummy_leg_state, other.dummy_leg_state, atol, rtol):
            return self.invariant_part.almost_equal(other, atol=atol, rtol=rtol)
        else:
            # The decomposition into invariant part and non-invariant state is not unique,
            # so we cant just compare them individually.
            # OPTIMIZE (JU) is there a more efficient way?
            # TODO should we have an extra argument that changes the behavior to return False
            #  (or None, which is falsy) in this case?
            warnings.warn('Converting ChargedTensor to dense block for `almost_equal`', stacklevel=2)
            backend = get_same_backend(self, other)
            self_block = self.to_dense_block()
            other_block = other.to_dense_block(leg_order=match_legs(self, other))
            return backend.block_allclose(self_block, other_block)

    # TODO i dont really liek this method-based implementation anymore...
    def tdot(self, other: AbstractTensor,
             legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
             relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        raise NotImplementedError  # TODO

    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        raise NotImplementedError  # TODO

    def inner(self, other: AbstractTensor, do_conj: bool = True,
              legs1: list[int | str] = None, legs2: list[int | str]  = None) -> complex:
        raise NotImplementedError  # TODO

    def permute_legs(self, permutation: list[int | str]) -> AbstractTensor:
        permutation = self.get_leg_idcs(permutation)  # needed, since invariant_part does not have the same legs
        permutation = permutation + [-1]  # keep dummy leg at its position
        return ChargedTensor(invariant_part=self.invariant_part.permute_legs(permutation),
                             dummy_leg_state=self.dummy_leg_state)

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> AbstractTensor | float | complex:
        legs1 = self.get_leg_idcs(legs1)  # needed, since invariant_part does not have the same legs
        legs2 = self.get_leg_idcs(legs2)
        return ChargedTensor(invariant_part=self.invariant_part.trace(legs1, legs2),
                             dummy_leg_state=self.dummy_leg_state)

    def conj(self) -> AbstractTensor:
        if self.dummy_leg_state is None:
            dummy_leg_state = None  # conj([1]) == [1]
        else:
            dummy_leg_state = self.backend.block_conj(self.dummy_leg_state)
        return ChargedTensor(invariant_part=self.invariant_part.conj(), dummy_leg_state=dummy_leg_state)

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace]=None,
                     product_spaces_dual: list[bool]=None,
                     new_axes: list[int]=None) -> Tensor:
        legs = [self.get_leg_idcs(group) for group in legs]  # needed, since invariant_part does not have the same legs
        inv = self.invariant_part.combine_legs(*legs, product_spaces=product_spaces,
                                               product_spaces_dual=product_spaces_dual, new_axes=new_axes)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=self.dummy_leg_state)

    def split_legs(self, *legs: int | str) -> AbstractTensor:
        legs = [self.get_leg_idcs(group) for group in legs]  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.split_legs(*legs),
                             dummy_leg_state=self.dummy_leg_state)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> AbstractTensor:
        legs = self.get_leg_idcs(legs)  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.squeeze_legs(legs),
                             dummy_leg_state=self.dummy_leg_state)

    def norm(self) -> float:
        if self.dummy_leg.dim == 1:
            return self._dummy_leg_state_item() * self.invariant_part.norm()
        else:
            # TODO could do sth like
            # sqrt(sum(slice_norms ** 2))
            #  where slice_norms = [abs(s) * inv[..., idx] for idx, s in enumerate(dummy_leg_state)]
            warnings.warn('Converting ChargedTensor to dense block for `norm`', stacklevel=2)
            return self.backend.block_norm(self.to_dense_block())


class DiagonalTensor(AbstractTensor):

    # special case where incoming and outgoing legs are equal and the
    # tensor is "diagonal" (yet to precisely formulate what this means in a basis-independent way...)
    # this would be the natural type for the singular values of an SVD
    #  > no_symmetry: tensor, reshaped to matrix is diagonal
    #  > abelian: blocks, reshaped to matrices are diagonal
    #  > nonabelian: not only diagonal in coupled irrep, but also in its multiplicity, i.e. blocks are diagonal matrices
    # TODO revisit this when Tensor class and specification for data-structure of backend is "finished"
    # TODO this could implement element-wise operations such as __mul__ and __pow__, it would be well defined

    # TODO should this allow that the incoming and outgoing legs are non-dual of each other?

    def __init__(self) -> None:
        raise NotImplementedError

    def to_full_tensor(self) -> Tensor:
        raise NotImplementedError


def zero_like(tens: AbstractTensor, labels: list[str | None] = None) -> Tensor:
    if labels is None:
        labels = tens.labels
    return type(tens).zero(backend=tens.backend, legs=tens.legs, labels=labels, dtype=tens.dtype)


# TODO is there a use for a special Scalar(AbstractTensor) class?


def match_legs(t1: AbstractTensor, t2: AbstractTensor,
               legs1: list[int | str] = None, legs2: list[int | str] = None):
    """Utility function that determines the permutation necessary to match the legs
    of the second tensor `t2` to those of the first tensor `t1`.

    Parameters
    ----------
    t1, t2 :
        The two tensors
    legs1, legs2 : list of int or str, optional
        Two lists that specify which leg of `t1` should be matched with which of `t2`;
        If both are `None` (default) in strict label mode, legs with the same label will be matched
        and an ValueError is raised if that is not possible.
        If both are `None` (default) in lax label mode, legs will be matched by order.
        If one is `None`, it is equivalent to `range(tn.num_legs)`.
        If both are given, they specify that ``legs1[n]`` of `t1` will be contracted with ``legs2[n]`` of `t2`.

    Returns
    -------
    permutation : (list of int) or None
        The permutation required, the ``permutation[n]``-th leg of `t2` is matched with the ``n``-th
        leg of `t1`. A result ``None`` is equivalent to ``range(t1.num_legs)`` and signals that
        no permutation is needed
    """
    if legs1 is None and legs2 is None:
        if config.strict_labels:
            if not (t1.is_fully_labelled and t2.is_fully_labelled):
                raise ValueError('Fully labelled tensors are required in strict label mode')
            if t1.shape._labels == t2.shape._labels:
                return None
            return t2.get_leg_idcs(t1.labels)
        else:
            return None

    elif legs1 is None:
        return t2.get_leg_idcs(legs2)

    elif legs2 is None:
        return np.argsort(t1.get_leg_idcs(legs1))

    else:
        order1 = np.asarray(t1.get_leg_idcs(legs1), dtype=np.intp)
        order2 = np.asarray(t2.get_leg_idcs(legs2), dtype=np.intp)
        perm = np.argsort(order2)[order1]
        return perm


def tdot(t1: AbstractTensor, t2: AbstractTensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
    """
    Contraction of two tensors.

    TODO more details, e.g. that legs need to match

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


def outer(t1: AbstractTensor, t2: AbstractTensor, relabel1: dict[str, str] = None,
          relabel2: dict[str, str] = None) -> AbstractTensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    return t1.outer(t2, relabel1=relabel1, relabel2=relabel2)


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
        Specify which leg belongs to which, see `match_legs`.
    """
    return t1.inner(t2, do_conj=do_conj, legs1=legs1, legs2=legs2)


def permute_legs(t: AbstractTensor, permutation: list[int]) -> AbstractTensor:
    """Change the order of legs of a Tensor.
    """
    # TODO: also have an inplace version?
    # TODO: name it permute_legs or sth instead?
    return t.permute_legs(permutation)


def trace(t: AbstractTensor, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
          ) -> AbstractTensor | float | complex:
    """
    Trace over one or more pairs of legs, that is contract these pairs.
    """
    return t.trace(legs1=legs1, legs2=legs2)


def conj(t: AbstractTensor) -> AbstractTensor:
    """
    The conjugate of t, living in the dual space.
    Labels are adjuste as `'p'` -> `'p*'` and `'p*'` -> `'p'`
    """
    return t.conj()


# TODO there should be an operation that converts only one or some of the legs to dual
#  i.e. vectorization of density matrices
#  formally, this is contraction with the (co-)evaluation map, aka cup or cap

def combine_legs(t: AbstractTensor,
                 *legs: list[int | str],
                 product_spaces: list[ProductSpace]=None,
                 product_spaces_dual: list[bool]=None,
                 new_axes: list[int]=None) -> AbstractTensor:
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
    # TODO inplace version
    return t.split_legs(*legs)


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs"""
    if isinstance(obj, AbstractTensor):
        return all(d == 1 for d in obj.shape)
        # TODO is above check wrong?
        # It was checking all(leg.is_trivial for leg in obj.legs) before, but that returned False
        # for Tensor([[1.]], [leg, leg.dual]) with leg = VectorSpace(sectors=[[1]], multiplicities=[1])
    if isinstance(obj, (int, float, complex, Integral)):  # Integral for np.int64()
        return True
    raise TypeError(f'Type not supported for is_scalar: {type(obj)}')


def squeeze_legs(t: AbstractTensor, legs: int | str | list[int | str] = None) -> Tensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If legs is None (default), all trivial legs are squeezed
    """
    return t.squeeze_legs(legs=legs)


def norm(t: AbstractTensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.norm()


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


def get_same_backend(*tensors: AbstractTensor, error_msg: str = 'Incompatible backends.'):
    """If all tensors have the same backend, return it. Otherwise raise a ValueError"""
    try:
        backend = tensors[0].backend
    except IndexError:
        raise ValueError('expected at least one tensor') from None
    if not all(tens.backend == backend for tens in tensors):
        raise ValueError(error_msg)
    return backend
