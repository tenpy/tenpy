# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Iterable
import numpy as np
from enum import Enum, auto

from .misc import duplicate_entries, force_str_len, join_as_many_as_possible
from .dummy_config import config
from .symmetries import VectorSpace, ProductSpace
from .backends.backend_factory import get_default_backend

__all__ = ['AbstractTensor', 'Tensor', 'ChargedTensor', 'DiagonalTensor', 'tdot', 'outer', 'inner', 
           'transpose', 'trace', 'conj', 'combine_legs', 'split_leg', 'is_scalar', 'allclose', 
           'squeeze_legs', 'norm', 'get_same_backend', 'Dtype', 'zero_like']


class Dtype(Enum):
    # TODO expose those in some high-level init, maybe even as tenpy.float32 ?
    float32 = auto()
    float64 = auto()
    complex64 = auto()
    complex128 = auto()
    

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
        labels = label.lstrip('(').rstrip(')').split('.')
        return [None if l.startswith('?') else l for l in labels]
    else:
        raise ValueError('Invalid format for a combined label')

_DUMMY_LABEL = '!'


ALL_TRIVIAL_LEGS = object()  # TODO use None instead ...


class AbstractTensor(ABC):

    def __init__(self, legs: list[VectorSpace], backend, labels: list[str | None] | None):
        """
        Parameters
        ----------
        legs : list[VectorSpace]
            The legs of the Tensor
        backend: :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
            The backend for the Tensor
        labels : list[str | None] | None
            Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
        """
        if backend is None:
            self.backend = get_default_backend()
        else:
            self.backend = backend
        self.legs = [backend.convert_vector_space(leg) for leg in legs]
        if labels is None:
            self._labels = [None] * len(legs)
        else:
            self._labels = labels
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}
        self.num_legs = len(legs)
        self.symmetry = legs[0].symmetry
        
        if self.num_legs == 1:
            self.parent_space = self.legs[0]
        else:
            self.parent_space = backend.convert_vector_space(ProductSpace(self.legs))

    def check_sanity(self):
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.symmetry == self.symmetry for l in self.legs)
        assert len(self.legs) == len(self._labels) == self.num_legs > 0
        assert not duplicate_entries(self._labels, ignore=[None])

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

    # TODO implement a shape property. 
    # should probably be a custom class that can be indexed by label (str) or index (int)
    #  -> include in tests

    @property
    def is_fully_labelled(self) -> bool:
        return None not in self._labels

    def has_label(self, label: str, *more: str) -> bool:
        return label in self._labels and all(l in self._labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        return self.is_fully_labelled and len(labels) == len(self._labels) and set(labels) == set(self._labels)

    def set_labels(self, labels: list[str | None]):
        assert not duplicate_entries(labels, ignore=[None])
        assert len(labels) == self.num_legs
        self._labels = labels[:]
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}

    @property
    def labels(self) -> list[str]:
        return self._labels[:]

    @labels.setter
    def labels(self, value):
        self.set_labels(value)

    def get_leg_idx(self, which_leg: int | str) -> int:
        if isinstance(which_leg, str):
            try:
                which_leg = self._labelmap[which_leg]
            except KeyError:
                raise ValueError(f'No leg with label {which_leg}.') from None
        if isinstance(which_leg, int):
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
        """A summary of the components of legs, used in Tensor.__repr__"""
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

    @abstractmethod
    def _mul_scalar(self, other: complex):
        ...

    def __getitem__(self, item):
        # TODO point towards a "as flat" option
        raise TypeError('Tensor object is not subscriptable')

    def __neg__(self):
        return self._mul_scalar(-1)

    def __pos__(self):
        return self

    def __eq__(self, other):
        # TODO make sure the pointer is correct.
        raise TypeError('Tensor does not support == comparison. Use tenpy.allclose instead.')

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __sub__(self, other):
        ...

    def __mul__(self, other):
        if isinstance(other, AbstractTensor):
            if all(leg.dim == 1 for leg in other.legs):
                return self._mul_scalar(other.item())
            if all(leg.dim == 1 for leg in self.legs):
                return other._mul_scalar(self.item())
            raise ValueError('Tensors can only be multiplied with scalars') from None
        if isinstance(other, (int, float, complex)):
            return self._mul_scalar(other)
        return NotImplemented

    def __rmul__(self, other):
        # all allowed multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            try:
                other = other.item()
            except ValueError:
                raise ValueError('Tensors can only be divived by scalars') from None
        return self._mul_scalar(1. / other)

    @abstractmethod
    def is_real(self) -> bool:
        ...

    def __float__(self):
        if not self.is_real():
            pass  # FIXME issue warning
        return self.item().real

    def __complex__(self):
        return complex(self.item())

    @abstractmethod
    def to_dense_block(self, leg_order: list[int | str] = None):
        """Convert tensor to a dense (i.e. no longer exploiting the symmetry structure) block,
        i.e. a numpy ndarray if the backend is a NumpyBlockBackend, or a torch Tensor of
        the backend is a TorchBlockBackend"""
        ...

    def to_numpy_ndarray(self, leg_order: list[int | str] = None, dtype=None) -> np.ndarray:
        """Convert to a numpy array"""
        # TODO (JU) this assumes that the blocks are valid inputs to np.asarray.
        #  are there cases where they are not?
        # TODO document what dtype is! give it a clearer name, eg numpy_dtype ?
        return np.asarray(self.to_dense_block(leg_order=leg_order), dtype=dtype)

    def __array__(self, dtype=None):
        return self.to_numpy_ndarray(dtype=dtype)

    @classmethod
    @abstractmethod
    def zero(cls, legs: list[VectorSpace] | list[int], backend=None, labels: list[str | None] = None,
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
    def inner(self, other: AbstractTensor) -> complex: 
        """See tensors.inner"""
        ...

    @abstractmethod
    def transpose(self, permutation: list[int]) -> AbstractTensor:
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
    def combine_legs(self, legs: list[int | str], new_leg: ProductSpace = None) -> AbstractTensor:
        """See tensors.combine_legs"""
        ...

    @abstractmethod
    def split_leg(self, leg: int | str) -> AbstractTensor:
        """See tensors.split_leg"""
        ...

    @abstractmethod
    def squeeze_legs(self, legs: int | str | list[int | str] = ALL_TRIVIAL_LEGS) -> AbstractTensor:
        """See tensors.squeeze_legs"""
        ...

    @abstractmethod
    def norm(self) -> float:
        """See tensors.norm"""
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
        Use as_tensor instead.  TODO point to which methods here?
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

    @property
    def dtype(self) -> Dtype:
        return self.backend.get_dtype_from_data(self.data)

    def check_sanity(self):
        super().check_sanity()

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
            other_order = _match_label_order(self, other)
            if other_order is not None:
                other = transpose(other, other_order)
            res_data = backend.add(self, other)
            return Tensor(res_data, backend=backend, legs=self.legs, labels=self.labels)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self.__add__(other._mul_scalar(-1))
        return NotImplemented

    def __repr__(self):
        indent = '  '
        label_strs = [force_str_len(label, 5) for label in self.labels]
        dim_strs = [force_str_len(leg.dim, 5) for leg in self.legs]
        components_strs = self._repr_leg_components(max_len=50)  # TODO picked an arbitrary length

        lines = [
            f'Tensor(',
            f'{indent}* Backend: {type(self.backend).__name__}',
            f'{indent}* Symmetry: {self.symmetry}',
            f'{indent}* Legs:  label    dim  components',
            f'{indent}         =============={"=" * max(10, *(len(c) for c in components_strs))}',
        ]
        for l, d, c in zip(label_strs, dim_strs, components_strs):
            lines.append(f'{indent}         {l}  {d}  {c}')
        lines.extend(self.backend._data_repr_lines(self.data, indent=indent, max_width=70, max_lines=20))
        lines.append(')')
        return "\n".join(lines)

    def is_real(self):
        return self.backend.is_real(self)

    def to_dense_block(self, leg_order: list[int | str] = None):
        block = self.backend.to_dense_block(self)
        if leg_order is not None:
            block = self.backend.block_transpose(block, self.get_leg_idcs(leg_order))
        return block

    @classmethod
    def from_numpy(cls, array: np.ndarray, backend=None, legs: list[VectorSpace]=None, dtype: Dtype = None,
                   labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5) -> Tensor:
        """
        Like from_dense_block but `array` is a numpy array
        """
        if backend is None:
            backend = get_default_backend()
        block = backend.block_from_numpy(np.asarray(array))
        return cls.from_dense_block(block=block, backend=backend, legs=legs, labels=labels, atol=atol,
                                    rtol=rtol, dtype=dtype)

    @classmethod
    def from_dense_block(cls, block, backend=None, legs: list[VectorSpace]=None, dtype: Dtype=None,
                         labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Tensor:
        """Convert a dense block of the backend to a Tensor with given symmetry (implied by the `legs`), 
        if the block is symmetric under it.
        If data is not symmetric under the symmetry i.e. if
        ``not allclose(array, projected, atol, rtol)``, raise a ValueError.

        TODO document how the sectors are expected to be embedded, i.e. which slices correspond to which charge.
        TODO support non-canonical embedding?
        TODO make backend optional? let get_backend with no args return a globally configurable default

        Parameters
        ----------
        array : array_like
            The data to be converted to a Tensor.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`, optional
            The backend for the Tensor
        legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`, optional
            The vectorspaces associated with legs of the tensors. Contains symmetry data.
            If ``None`` (default), trivial legs of appropriate dimension are assumed.
        dtype : ``np.dtype``, optional
            The data type of the Tensor entries. Defaults to dtype of `block`
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
        """
        is_real = False  # FIXME dummy
        if backend is None:
            backend = get_default_backend()
        if legs is None:
            legs = [VectorSpace.non_symmetric(d, is_real=is_real) for d in backend.block_shape(block)]
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        data = backend.from_dense_block(block, legs=legs, atol=atol, rtol=rtol)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def zero(cls, legs: list[VectorSpace] | list[int], backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.complex128) -> Tensor:
        if any(isinstance(l, int) for l in legs):
            assert all(isinstance(l, int) for l in legs)
            legs = [VectorSpace.non_symmetric(d) for d in legs]
        if backend is None:
            backend = get_default_backend()
        data = backend.zero_data(legs=legs, dtype=dtype)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def eye(cls, legs_or_dims: int | VectorSpace | list[int | VectorSpace], backend=None, 
            labels: list[str | None] = None, dtype: Dtype = Dtype.complex128) -> Tensor:
        """The identity map from one group of legs to their duals.

        Parameters
        ----------
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs_or_dims : int | VectorSpace | list[int | VectorSpace]
            Description of *half* of the legs of the result, either via their vectorspace
            or via an integer, which means a trivial VectorSpace of that dimension.
            The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype, optional
            The data type of the Tensor entries.

        """
        if backend is None:
            backend = get_default_backend()
        legs = _parse_legs_or_dims(legs_or_dims)
        data = backend.eye_data(legs=legs, dtype=dtype)
        legs = legs + [leg.dual for leg in legs]
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def from_numpy_func(cls, func, legs_or_dims: int | VectorSpace | list[int | VectorSpace], backend=None,
                        labels: list[str | None] = None, func_args=(), func_kwargs={}, 
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
            If no `shape_kw` is given, it is called as ``func(shape, *func_args, **func_kwargs)``,
            otherwise as ``func(*func_args, **{shape_kw: shape}, **func_kwargs)``,
            where `shape` is a tuple of int.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the tensor
        legs_or_dims : int | VectorSpace | list[int | VectorSpace]
            Description of *half* of the legs of the result, either via their vectorspace
            or via an integer, which means a trivial VectorSpace of that dimension.
            The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        unc_args : iterable
            Additional arguments given to `func`.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype
        """
        if backend is None:
            backend = get_default_backend()
            
        def block_func(shape):
            if shape_kw is None:
                arr = func(shape, *func_args, **func_kwargs)
            else:
                arr = func(*func_args, **{shape_kw: shape}, **func_kwargs)
            return backend.block_from_numpy(arr)
        
        legs = _parse_legs_or_dims(legs_or_dims)
        return cls(data=backend.from_block_func(block_func, legs), backend=backend, legs=legs,
                   labels=labels)

    @classmethod
    def from_block_func(cls, func, legs_or_dims: int | VectorSpace | list[int | VectorSpace], backend=None,
                        labels: list[str | None] = None, func_args=(), func_kwargs={}, 
                        shape_kw: str = None, dtype: Dtype = None) -> Tensor:
        """Create a Tensor from a block function.

        This function ceates a tensor by filling the blocks, i.e. the free paramaters of the tensors
        using `func`, which is a function returning backend-specific blocks.

        Parameters
        ----------
        func : callable
            A callable object which is called to generate the blocks.
            We expect that `func` returns a backend-specific block of the given `shape`.
            If no `shape_kw` is given, it is called as ``func(shape, *func_args, **func_kwargs)``,
            otherwise as ``func(*func_args, **{shape_kw: shape}, **func_kwargs)``,
            where `shape` is a tuple of int.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the tensor
        legs_or_dims : int | VectorSpace | list[int | VectorSpace]
            Description of *half* of the legs of the result, either via their vectorspace
            or via an integer, which means a trivial VectorSpace of that dimension.
            The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        unc_args : iterable
            Additional arguments given to `func`.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype
        """
        if backend is None:
            backend = get_default_backend()
            
        def block_func(shape):
            if shape_kw is None:
                return func(shape, *func_args, **func_kwargs)
            else:
                return func(*func_args, **{shape_kw: shape}, **func_kwargs)
        legs = _parse_legs_or_dims(legs_or_dims)
        return cls(data=backend.from_block_func(block_func, legs), backend=backend, legs=legs,
                   labels=labels)

    def tdot(self, other: AbstractTensor, 
             legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0, 
             relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        if not isinstance(other, Tensor):
            raise NotImplementedError  # TODO

        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = other.get_leg_idcs(legs2)
        if len(leg_idcs1) != len(leg_idcs2):
            # checking this for leg_idcs* instead of legs* allows us to assume that they are both lists
            raise ValueError('Must specify the same number of legs for both tensors')
        if not all(self.legs[idx1].is_dual_of(other.legs[idx2]) for idx1, idx2 in zip(leg_idcs1, leg_idcs2)):
            raise ValueError('Incompatible legs.')  # TODO show which
        backend = get_same_backend(self, other)
        open_legs1 = [leg for idx, leg in enumerate(self.legs) if idx not in leg_idcs1]
        open_legs2 = [leg for idx, leg in enumerate(other.legs) if idx not in leg_idcs2]
        open_labels1 = [leg for idx, leg in enumerate(self.labels) if idx not in leg_idcs1]
        open_labels2 = [leg for idx, leg in enumerate(other.labels) if idx not in leg_idcs2]
        res_labels = _get_result_labels(open_labels1, open_labels2, relabel1, relabel2)
        res_data = backend.tdot(self, other, leg_idcs1, leg_idcs2)
        res_legs = open_legs1 + open_legs2
        if len(res_legs) == 0:
            # TODO make sure this is a python scalar not some weird backend-structure
            return res_data
        else:
            return Tensor(res_data, backend=backend, legs=res_legs, labels=res_labels)

    def outer(self, other: AbstractTensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
        if not isinstance(other, Tensor):
            raise NotImplementedError  # TODO

        backend = get_same_backend(self, other)
        res_labels = _get_result_labels(self.labels, other.labels, relabel1, relabel2)
        res_data = backend.outer(self, other)
        return Tensor(res_data, backend=backend, legs=self.legs + other.legs, labels=res_labels)

    def inner(self, other: AbstractTensor) -> complex:
        if not isinstance(other, Tensor):
            raise NotImplementedError  # TODO

        if self.num_legs != other.num_legs:
            raise ValueError('Tensors need to have the same number of legs')
        leg_order_2 = _match_label_order(self, other)
        if leg_order_2 is None:
            leg_order_2 = list(range(other.num_legs))
        if not all(self.legs[n1] == other.legs[n2] for n1, n2 in enumerate(leg_order_2)):
            raise ValueError('Incompatible legs')
        backend = get_same_backend(self, other)
        res = backend.inner(self, other, axs2=leg_order_2)
        # TODO: Scalar(Tensor) class...?
        return res

    def transpose(self, permutation: list[int]) -> AbstractTensor:
        # TODO also support labels (i.e. str) as permutation entries?
        if config.strict_labels:
            # TODO: proper warning:
            # strict labels means position of legs should be irrelevant, there is no need to transpose.
            print('dummy warning!')
        assert len(permutation) == self.num_legs
        assert set(permutation) == set(range(self.num_legs))
        res_data = self.backend.transpose(self, permutation)
        return Tensor(res_data, backend=self.backend, legs=[self.legs[n] for n in permutation],
                    labels=[self._labels[n] for n in permutation])
        
    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> AbstractTensor | float | complex:
        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = self.get_leg_idcs(legs2)
        if len(leg_idcs1) != len(leg_idcs2):
            raise ValueError('Must specify same number of legs')
        remaining_leg_idcs = [n for n in range(self.num_legs) if n not in leg_idcs1 and n not in leg_idcs2]
        res_data = self.backend.trace(self, leg_idcs1, leg_idcs2)
        if len(remaining_leg_idcs) == 0:
            # result is a scalar
            return self.backend.data_item(res_data)
        else:
            return Tensor(res_data, backend=self.backend, legs=[self.legs[n] for n in remaining_leg_idcs],
                          labels=[self.labels[n] for n in remaining_leg_idcs])

    def conj(self) -> AbstractTensor:
        """See tensors.conj"""
        # TODO (Jakob) think about this in the context of pivotal category with duals
        return Tensor(self.backend.conj(self), backend=self.backend, legs=[l.dual for l in self.legs],
                      labels=[_dual_leg_label(l) for l in self._labels])

    def combine_legs(self, legs: list[int | str], new_leg: ProductSpace = None) -> AbstractTensor:
        """See tensors.combine_legs"""
        if len(legs) < 2:
            raise ValueError('expected at least two legs')

        leg_idcs = self.get_leg_idcs(legs)
        if new_leg is None:
            new_leg = ProductSpace([self.legs[idx] for idx in leg_idcs])
        res_legs = [new_leg if idx == leg_idcs[0] else leg for idx, leg in enumerate(self.legs)
                    if idx not in leg_idcs[1:]]
        new_label = _combine_leg_labels([self._labels[idx] for idx in leg_idcs])
        res_labels = [new_label if idx == leg_idcs[0] else label for idx, label in enumerate(self._labels)
                      if idx not in leg_idcs[1:]]
        res_data = self.backend.combine_legs(self, idcs=leg_idcs, new_leg=new_leg)
        return Tensor(res_data, backend=self.backend, legs=res_legs, labels=res_labels)

    def split_leg(self, leg: int | str) -> AbstractTensor:
        """See tensors.split_leg"""
        leg_idx = self.get_leg_idx(leg)
        if not isinstance(self.legs[leg_idx], ProductSpace):
            raise ValueError(f'Leg {leg} is not a ProductSpace.')
        legs = self.legs[:leg_idx] + self.legs[leg_idx].spaces + self.legs[leg_idx + 1:]
        labels = self.labels[:leg_idx] + _split_leg_label(self.labels[leg_idx]) + self.labels[leg_idx + 1:]
        res_data = self.backend.split_leg(self, leg_idx=leg_idx)
        return Tensor(res_data, backend=self.backend, legs=legs, labels=labels)

    def squeeze_legs(self, legs: int | str | list[int | str] = ALL_TRIVIAL_LEGS) -> AbstractTensor:
        """See tensors.squeeze_legs"""
        if legs is ALL_TRIVIAL_LEGS:
            leg_idcs = [n for n, l in enumerate(self.legs) if l.is_trivial]
        else:
            leg_idcs = self.get_leg_idcs(legs)
            if not all(self.legs[idx].is_trivial for idx in leg_idcs):
                raise ValueError('Tried to squeeze non-trivial legs.')
        res_legs = [l for idx, l in enumerate(self.legs) if idx not in leg_idcs]
        res_labels = [label for idx, label in enumerate(self.labels) if idx not in leg_idcs]
        res_data = self.backend.squeeze_legs(self, leg_idcs)
        return Tensor(res_data, backend=self.backend, legs=res_legs, labels=res_labels)

    def norm(self) -> float:
        """See tensors.norm"""
        return self.backend.norm(self)


# TODO is this a good name?
class ChargedTensor(AbstractTensor):
    # formerly, this was covered by npc Arrays with qtotal != 0.
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
    
    def __init__(self, invariant_part: Tensor, dummy_leg_state=None):
        AbstractTensor.__init__(self, backend=invariant_part.backend, legs=invariant_part.legs[:-1], 
                                labels=invariant_part.labels[:-1])
        self.invariant_part = invariant_part
        self.dummy_leg = invariant_part.leg[-1]
        if dummy_leg_state is None:
            if self.dummy_leg.dim != 1:
                raise ValueError('Can not infer state for a dummy leg with dim > 1')
            dummy_leg_state = self.backend.block_from_numpy(np.array([1.]))
        self.dummy_leg_state = dummy_leg_state

    @property
    def dtype(self) -> Dtype:
        return self.invariant_part.dtype

    def check_sanity(self):
        self.invariant_part.check_sanity()
        assert self.backend.block_shape(self.dummy_leg_state) == (self.dummy_leg.dim,)
        super().check_sanity()

    def copy(self, deep=True):
        if deep:
            return ChargedTensor(invariant_part=self.invariant_part.copy(deep=True),
                                 dummy_leg_state=self.backend.block_copy(self.dummy_leg_state))
        return ChargedTensor(invariant_part=self.invariant_part, dummy_leg_state=self.dummy_leg_state)

    def item(self):
        if not all(leg.dim == 1 for leg in self.invariant_part.legs[:-1]):
            raise ValueError('Not a scalar')
        return self.backend.block_item(self.to_dense_block())

    def __repr__(self):
        return 'ChargedTensor(... [this is a stub __repr__])'  # TODO

    def is_real(self):
        return self.backend.is_real(self.invariant_part) and self.backend.block_is_real(self.dummy_leg_state)

    def to_dense_block(self, leg_order: list[int | str] = None):
        invariant_block = self.backend.to_dense_block(self.invariant_part)
        block = self.backend.block_tdot(invariant_block, self.dummy_leg_state, [-1], [0])
        if leg_order is not None:
            block = self.backend.block_transpose(block, self.get_leg_idcs(leg_order))
        return block

    # TODO "detect qtotal"-like classmethod

    @classmethod
    def from_numpy(cls, **todo_args):
        ...  # FIXME stub

    @classmethod
    def from_dense_block(cls, **todo_args):
        ...  # FIXME stub

    @classmethod
    def zero(cls, **todo_args):
        ...


class DiagonalTensor(AbstractTensor):

    # special case where incoming and outgoing legs are equal and the
    # tensor is "diagonal" (yet to precisely formulate what this means in a basis-independent way...)
    # this would be the natural type for the singular values of an SVD
    #  > no_symmetry: tensor, reshaped to matrix is diagonal
    #  > abelian: blocks, reshaped to matrices are diagonal
    #  > nonabelian: not only diagonal in coupled irrep, but also in its multiplicity, i.e. blocks are diagonal matrices
    # TODO revisit this when Tensor class and specification for data-structure of backend is "finished"
    # TODO this could implement element-wise operations such as __mul__ and __pow__, it would be well defined

    def __init__(self) -> None:
        raise NotImplementedError  # TODO

    def to_full_tensor(self) -> Tensor:
        raise NotImplementedError


def zero_like(tens: AbstractTensor, labels: list[str | None] = None) -> Tensor:
    if labels is None:
        labels = tens.labels
    return type(tens).zero(backend=tens.backend, legs=tens.legs, labels=labels, dtype=tens.dtype)


# TODO is there a use for a special Scalar(AbstractTensor) class?


def _match_label_order(a: Tensor, b: Tensor) -> Iterable[int] | None:
    """Determine the order of legs of b, such that they match the legs of a.
    If config.stric_labels, this is a permutation determined by the labels, otherwise it is None.
    A None return indicates range(b.num_legs), i.e. that no trasnpose is needed.
    """
    if config.strict_labels:
        if a.is_fully_labelled and b.is_fully_labelled:
            match_by_labels = True
        else:
            match_by_labels = False
            # TODO issue warning?
    else:
        match_by_labels = False

    if not match_by_labels:
        return None

    if a._labels == b._labels:
        return None
    
    return b.get_leg_idcs(a.labels)


def tdot(t1: AbstractTensor, t2: AbstractTensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> AbstractTensor:
    """
    TODO: decide name, eg from tensordot, tdot, contract

    Contraction of two tensors

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


def inner(t1: AbstractTensor, t2: AbstractTensor) -> complex:
    """
    Inner product of two tensors with the same legs.
    t1 and t2 live in the same space, the inner product is the contraction of the dual ("conjugate") of t1 with t2

    If config.strict_labels, legs with matching labels are contracted.
    Otherwise the n-th leg of t1 is contracted with the n-th leg of t2
    """
    return t1.inner(t2)


def transpose(t: AbstractTensor, permutation: list[int]) -> AbstractTensor:
    """Change the order of legs of a Tensor.
    TODO: also have an inplace version?
    TODO: name it permute_legs or sth instead?
    """
    return t.transpose(permutation)


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


def combine_legs(t: AbstractTensor, legs: list[int | str], new_leg: ProductSpace = None
                 ) -> AbstractTensor:
    """
    Combine a group of legs of a tensor. Resulting leg (of type ProductSpace) is at the
    previous position of legs[0].
    # TODO support multiple combines in one function call? what would the signature be
    # TODO inplace version
    """
    return t.combine_legs(legs=legs, new_leg=new_leg)


def split_leg(t: AbstractTensor, leg: int | str) -> Tensor:
    """
    Split a leg that was previously combined.
    If the legs were contiguous in t.legs before combining, this is the inverse operation of combine_legs,
    otherwise it is the inverse up to a permute_legs
    # TODO support multiple splits? -> make consistent with combine
    # TODO inplace version
    """
    return t.split_leg(leg=leg)


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs"""
    if isinstance(obj, (int, float, complex)):
        return True
    if isinstance(obj, Tensor):
        return all(l.is_trivial for l in obj.legs)
    if isinstance(obj, AbstractTensor):
        raise NotImplementedError  # TODO
    else:
        raise TypeError(f'Type not supported for is_scalar: {type(obj)}')


def allclose(a, b, rtol=1e-05, atol=1e-08) -> bool:
    """
    If a and b are equal up to numerical tolerance, that is if `norm(a - b) <= atol + rtol * norm(a)`.
    Note that the definition is not symmetric under exchanging `a` and `b`.

    TODO "all" isnt really reflecting what going on. different name?
    TODO this should scale with the number of entries somehow, no?
    """
    assert rtol >= 0
    assert atol >= 0
    if isinstance(a, AbstractTensor) and isinstance(b, AbstractTensor):
        diff = norm(a - b)
        a_norm = norm(a)
    else:
        if isinstance(a, AbstractTensor):
            try:
                a = a.item()
            except ValueError:
                raise ValueError('Can not compare non-scalar Tensor and scalar') from None
        if isinstance(b, AbstractTensor):
            try:
                b = b.item()
            except ValueError:
                raise ValueError('Can not compare scalar and non-scalar Tensor') from None
        diff = abs(a - b)
        a_norm = abs(a)
    return diff <= atol + rtol * a_norm


def squeeze_legs(t: AbstractTensor, legs: int | str | list[int | str] = ALL_TRIVIAL_LEGS) -> Tensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If no legs are specified, all trivial legs are squeezed
    """
    return t.squeeze_legs(legs=legs)


def norm(t: AbstractTensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.norm()


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
        # TODO issue warning
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


def _parse_legs_or_dims(legs_or_dims: int | VectorSpace | list[int | VectorSpace]) -> list[VectorSpace]:
    if isinstance(legs_or_dims, int):
        return [VectorSpace.non_symmetric(legs_or_dims)]
    elif isinstance(legs_or_dims, VectorSpace):
        return [legs_or_dims]
    else:
        return [ele if isinstance(ele, VectorSpace) else VectorSpace.non_symmetric(ele)
                for ele in legs_or_dims]
