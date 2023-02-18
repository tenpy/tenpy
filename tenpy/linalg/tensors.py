from __future__ import annotations

from typing import Iterable, TypeVar
import numpy as np

from .misc import duplicate_entries, force_str_len
from .dummy_config import config
from .symmetries import VectorSpace, ProductSpace


float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128
SUPPORTED_DTYPES = [float32, float64, complex64, complex128]
Dtype = TypeVar('Dtype', bound=type)
# TODO need more dtypes than that?


def dual_leg_label(label: str) -> str:
    """return the label that a leg should have after conjugation"""
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def combine_leg_labels(labels: list[str | None]) -> str:
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


def split_leg_label(label: str) -> list[str | None]:
    if label.startswith('(') and label.endswith(')'):
        labels = label.lstrip('(').rstrip(')').split('.')
        return [None if l.startswith('?') else l for l in labels]
    else:
        raise ValueError('Invalid format for a combined label')


class Tensor:
    """

    Attributes
    ----------
    data
        backend-specific data structure that contains the numerical data, i.e. the free parameters
        of tensors with the given symmetry.
        data about the symmetry is contained in the legs.
    backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
    legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`
    labels : list of {``None``, str}
    """

    def __init__(self, data, backend, legs: list[VectorSpace], labels: list[str | None] = None):
        """
        This constructor is not user-friendly. Use as_tensor instead.
        Inputs are not checked for consistency.
        """
        self.data = data
        self.backend = backend
        self.legs = legs
        self._labels = labels or [None] * len(legs)
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}
        self.num_legs = len(legs)
        self.symmetry = legs[0].symmetry
        self.backend.finalize_Tensor_init(self)

    @property
    def dtype(self):
        return self.backend.get_dtype(self)

    @property
    def parent_space(self) -> VectorSpace:
        if self.num_legs == 1:
            return self.legs[0]
        else:
            return ProductSpace(spaces=self.legs)

    def check_sanity(self):
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.symmetry == self.symmetry for l in self.legs)
        assert len(self.legs) == self.num_legs > 0

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
        return None not in self.labels

    def has_label(self, label: str, *more: str) -> bool:
        return label in self.labels and all(l in self.labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        return self.is_fully_labelled and len(labels) == len(self.labels) and set(labels) == set(self.labels)

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
        # TODO which type of error should be raised?
        if isinstance(which_leg, str):
            try:
                which_leg = self._labelmap[which_leg]
            except KeyError:
                raise KeyError(f'No leg with label {which_leg}.') from None
        if isinstance(which_leg, int):
            if which_leg < 0:
                which_leg = which_leg + self.num_legs
            if not 0 <= which_leg < self.num_legs:
                raise KeyError(f'Leg index out of bounds: {which_leg}.') from None
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

    def copy(self):
        """return a Tensor object equal to self, such that in-place operations on self.copy() do not affect self"""
        return Tensor(data=self.backend.copy_data(self.data), backend=self.backend, legs=self.legs[:], labels=self.labels[:])

    def item(self):
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')

    def __repr__(self):
        indent = '  '
        COMPONENT_LEN = 50

        label_strs = [force_str_len(label, 5) for label in self.labels]
        dim_strs = [force_str_len(leg.dim, 5) for leg in self.legs]
        components_strs = []
        for leg, label in zip(self.legs, self.labels):
            if isinstance(leg, ProductSpace):
                sublabels = split_leg_label(label)
                components = ' ⊗ '.join(f'({l}: {s.dim})' for l, s in zip(sublabels, leg.spaces))
            else:
                components = ' ⊕ '.join(f'({mult} * {leg.symmetry.sector_str(sect)})'
                                        for mult, sect in zip(leg.multiplicities, leg.sectors))
            if len(components) > COMPONENT_LEN:
                components = components[:COMPONENT_LEN - 6] + ' [...]'
            components_strs.append(components)

        lines = [
            f'Tensor(',
            f'{indent}* Backend: {type(self.backend).__name__}'
            f'{indent}* Symmetry: {self.symmetry}',
            # TODO if we end up supporting qtotal, it should go here
            f'{indent}* Legs:  label    dim  components',
            f'{indent}         =============={"=" * max(10, *(len(c) for c in components_strs))}',
        ]
        for l, d, c in zip(label_strs, dim_strs, components_strs):
            lines.append(f'{indent}         {l}  {d}  {c}')
        lines.extend(self.backend._data_repr_lines(self.data, indent=indent, max_width=70, max_lines=20))
        lines.append(')')
        return "\n".join(lines)

    def __getitem__(self, item):
        # TODO point towards a "as flat" option
        raise TypeError('Tensor object is not subscriptable')

    def __neg__(self):
        return mul(-1, self)

    def __pos__(self):
        return self

    def __eq__(self, other):
        # TODO make sure the pointer is correct.
        raise TypeError('Tensor does not support == comparison. Use tenpy.allclose instead.')

    def __add__(self, other):
        if isinstance(other, Tensor):
            return add(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            # TODO is this efficient enough?
            return add(self, mul(-1, other))
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if all(leg.dim == 1 for leg in self.legs):
                return mul(self.item(), other)
            if all(leg.dim == 1 for leg in other.legs):
                return mul(other.item(), self)
            raise ValueError('Tensors can only be multiplied with scalars') from None
        if isinstance(other, (int, float, complex)):
            return mul(other, self)
        return NotImplemented

    __rmul__ = __mul__  # all allowed multiplications are commutative

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            try:
                other = other.item()
            except ValueError:
                raise ValueError('Tensors can only be divived by scalars') from None
        try:
            factor = 1. / other
        except TypeError:
            return NotImplemented
        return self.__mul__(factor)

    def __float__(self):
        if not self.dtype.is_real:
            raise UserWarning  # TODO logging system
        return self.item().real

    def __complex__(self):
        return complex(self.item())

    def __array__(self, dtype):
        # TODO this assumes that the blocks are valid inputs to np.asarray.
        #  are there cases where they are not?
        return np.asarray(self.backend.to_dense_block(self.data), dtype)

    @classmethod
    def from_numpy(cls, array: np.ndarray, backend, legs: list[VectorSpace]=None, dtype=None,
                   labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5) -> Tensor:
        """Convert a numpy array to a Tensor with given symmetry, if the array is symmetric under it.
        If data is not symmetric under the symmetry i.e. if
        ``not allclose(array, projected, atol, rtol)``, raise a ValueError.

        TODO document how the sectors are expected to be embedded, i.e. which slices correspond to which charge.
        TODO support non-canonical embedding?
        TODO make backend optional? let get_backend with no args return a globally configurable default

        Parameters
        ----------
        array : array_like
            The data to be converted to a Tensor.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`, optional
            The vectorspaces associated with legs of the tensors. Contains symmetry data.
            If ``None`` (default), trivial legs of appropriate dimension are assumed.
        dtype : ``np.dtype``, optional
            The data type of the Tensor entries. Defaults to dtype of `array`
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
        """
        block = backend.block_from_numpy(np.asarray(array, dtype=dtype))
        return cls.from_dense_block(block=block, backend=backend, legs=legs, labels=labels, atol=atol,
                                    rtol=rtol)

    @classmethod
    def from_dense_block(cls, block, backend, legs: list[VectorSpace]=None,
                         labels: list[str | None] = None, atol: float = 1e-8, rtol: float = 1e-5
                         ) -> Tensor:
        """Convert a dense block of the backend to a Tensor with given symmetry, if the block is
        symmetric under it.
        If data is not symmetric under the symmetry i.e. if
        ``not allclose(array, projected, atol, rtol)``, raise a ValueError.

        TODO document how the sectors are expected to be embedded, i.e. which slices correspond to which charge.
        TODO support non-canonical embedding?
        TODO make backend optional? let get_backend with no args return a globally configurable default

        Parameters
        ----------
        array : array_like
            The data to be converted to a Tensor.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs : list of :class:`~tenpy.linalg.symmetries.VectorSpace`, optional
            The vectorspaces associated with legs of the tensors. Contains symmetry data.
            If ``None`` (default), trivial legs of appropriate dimension are assumed.
        dtype : ``np.dtype``, optional
            The data type of the Tensor entries. Defaults to dtype of `array`
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
        """
        is_real = False  # FIXME dummy
        if legs is None:
            legs = [VectorSpace.non_symmetric(d, is_real=is_real) for d in backend.block_shape(block)]
        data = backend.from_dense_block(block, legs=legs, atol=atol, rtol=rtol)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def zero(cls, backend, legs: list[VectorSpace] | list[int], labels: list[str | None] = None,
             dtype: Dtype = complex128) -> Tensor:
        """A zero tensor"""
        if any(isinstance(l, int) for l in legs):
            assert all(isinstance(l, int) for l in legs)
            legs = [VectorSpace.non_symmetric(d) for d in legs]
        data = backend.zero_data(legs=legs, dtype=dtype)
        return cls(data=data, backend=backend, legs=legs, labels=labels)

    @classmethod
    def eye(cls, backend, legs_or_dim: int | list[VectorSpace], labels: list[str | None] = None,
            dtype: Dtype = complex128) -> Tensor:
        """The identity map from one group of legs to their duals.

        Parameters
        ----------
        backend : :class:`~tenpy.linalg.backends.abstract_backend.AbstractBackend`
            The backend for the Tensor
        legs_or_dim : int | list[VectorSpace]
            Either an integer, equivalent to a trivial leg of this dimension; the result has two legs.
            Or a list of spaces, then the identity map is from those spaces to their dual.
            The resulting tensor has twice as many legs.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype, optional
            The data type of the Tensor entries. Defaults to dtype of `array`.

        """
        if isinstance(legs_or_dim, int):
            legs_or_dim = [VectorSpace.non_symmetric(legs_or_dim)]
        data = backend.eye_data(legs=legs_or_dim, dtype=dtype)
        legs = legs_or_dim + [leg.dual for leg in legs_or_dim]
        return cls(data=data, backend=backend, legs=legs, labels=labels)



class DiagonalTensor(Tensor):

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


# TODO is there a use for a special Scalar(DiagonalTensor) class?


# TODO provide function with more narrowly defined input, "from_numpy" or sth,
# def as_tensor(obj, backend: AbstractBackend, legs: list[VectorSpace] = None, labels: list[str] = None,
#               dtype: Dtype = None) -> Tensor:
#     # TODO use a default backend from global config?
#     if isinstance(obj, Tensor):
#         obj = obj.copy()

#         if legs is not None:
#             raise NotImplementedError  # TODO what to do here?

#         if backend is not None:
#             raise NotImplementedError  # TODO

#         if labels is not None:
#             obj.set_labels(labels)

#         if dtype is not None:
#             obj.data = obj.backend.to_dtype(obj, dtype)

#         obj.check_sanity()
#         return obj

#     else:
#         obj, shape = backend.parse_data(obj, legs, dtype=backend.parse_dtype(dtype))
#         if legs is None:
#             legs = [VectorSpace.non_symmetric(d) for d in shape]
#         else:
#             assert backend.legs_are_compatible(obj, legs)
#         return Tensor(obj, backend, legs=legs, labels=labels)


def match_label_order(a: Tensor, b: Tensor) -> Iterable[int] | None:
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

    if a._labels == b._lables:
        return None
    
    return b.get_leg_idcs(a.leg_labels)


def add(a: Tensor, b: Tensor) -> Tensor:
    # TODO if one but not both is a DiagonalTensor, we need to convert it to Tensor
    backend = get_same_backend(a, b)
    b_order = match_label_order(a, b)
    if b_order is not None:
        b = transpose(b, b_order)
    res_data = backend.add(a, b)
    return Tensor(res_data, backend=backend, legs=a.legs, labels=a.labels)


def mul(a: float | complex, b: Tensor) -> Tensor:
    res_data = b.backend.mul(a, b)
    return Tensor(res_data, backend=b.backend, legs=b.legs, labels=b.labels)


def tdot(t1: Tensor, t2: Tensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """
    TODO: decide name, eg from tensordot, tdot, contract

    Contraction of two tensors

    Parameters
    ----------
    t1 : Tensor
    t2 : Tensor
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
    leg_idcs1 = t1.get_leg_idcs(legs1)
    leg_idcs2 = t2.get_leg_idcs(legs2)
    if len(leg_idcs1) != len(leg_idcs2):
        # checking this for leg_idcs* instead of legs* allows us to assume that they are both lists
        raise ValueError('Must specify the same number of legs for both tensors')
    if not all(t1.legs[idx1].can_contract_with(t2.legs[idx2]) for idx1, idx2 in zip(leg_idcs1, leg_idcs2)):
        raise ValueError('Incompatible legs.')  # TODO show which
    backend = get_same_backend(t1, t2)
    open_legs1 = [leg for idx, leg in enumerate(t1.legs) if idx not in leg_idcs1]
    open_legs2 = [leg for idx, leg in enumerate(t2.legs) if idx not in leg_idcs2]
    open_labels1 = [leg for idx, leg in enumerate(t1.labels) if idx not in leg_idcs1]
    open_labels2 = [leg for idx, leg in enumerate(t2.labels) if idx not in leg_idcs2]
    res_labels = get_result_labels(open_labels1, open_labels2, relabel1, relabel2)
    res_data = backend.tdot(t1, t2, leg_idcs1, leg_idcs2)
    return Tensor(res_data, backend=backend, legs=open_legs1 + open_legs2, labels=res_labels)


def outer(t1: Tensor, t2: Tensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    backend = get_same_backend(t1, t2)
    res_labels = get_result_labels(t1.labels, t2.labels, relabel1, relabel2)
    res_data = backend.outer(t1, t2)
    return Tensor(res_data, backend=backend, legs=t1.legs + t2.legs, labels=res_labels)


def inner(t1: Tensor, t2: Tensor) -> complex:
    """
    Inner product of two tensors with the same legs.
    t1 and t2 live in the same space, the inner product is the contraction of the dual ("conjugate") of t1 with t2

    If config.strict_labels, legs with matching labels are contracted.
    Otherwise the n-th leg of t1 is contracted with the n-th leg of t2
    """
    if t1.num_legs != t2.num_legs:
        raise ValueError('Tensors need to have the same number of legs')
    leg_order_2 = match_label_order(t1, t2)
    if not all(t1.legs[n1].space == t2.legs[n2].space for n1, n2 in enumerate(leg_order_2)):
        raise ValueError('Incompatible legs')
    backend = get_same_backend(t1, t2)
    res = backend.inner(t1, t2, axs2=leg_order_2)
    # TODO: Scalar(Tensor) class...?
    return res


def transpose(t: Tensor, permutation: list[int]) -> Tensor:
    """Change the order of legs of a Tensor.
    TODO: also have an inplace version?
    TODO: name it permute_legs or sth instead?
    """
    if config.strict_labels:
        # TODO: proper warning:
        # strict labels means position of legs should be irrelevant, there is no need to transpose.
        print('dummy warning!')
    assert len(permutation) == t.num_legs
    assert set(permutation) == set(range(t.num_legs))
    res_data = t.backend.transpose(t)
    return Tensor(res_data, backend=t.backend, legs=[t.legs[n] for n in permutation],
                  labels=[t.labels[n] for n in permutation])


def trace(t: Tensor, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
          ) -> Tensor | float | complex:
    """
    Trace over one or more pairs of legs, that is contract these pairs.
    """
    leg_idcs1 = t.get_leg_idcs(legs1)
    leg_idcs2 = t.get_leg_idcs(legs2)
    if len(leg_idcs1) != len(leg_idcs2):
        raise ValueError('Must specify same number of legs')
    remaining_leg_idcs = [n for n in range(t.num_legs) if n not in leg_idcs1 and n not in leg_idcs2]
    res_data = t.backend.trace(t, leg_idcs1, leg_idcs2)
    if len(remaining_leg_idcs) == 0:
        # result is a scalar
        return t.backend.item(res_data)
    else:
        return Tensor(res_data, backend=t.backend, legs=[t.legs[n] for n in remaining_leg_idcs],
                      labels=[t.labels[n] for n in remaining_leg_idcs])


def conj(t: Tensor) -> Tensor:
    """
    The conjugate of t, living in the dual space.
    Labels are adjuste as `'p'` -> `'p*'` and `'p*'` -> `'p'`
    """
    # TODO (Jakob) think about this in the context of pivotal category with duals
    return Tensor(t.backend.conj(t), backend=t.backend, legs=[l.dual for l in t.legs])


# TODO there should be an operation that converts only one or some of the legs to dual
#  i.e. vectorization of density matrices
#  formally, this is contraction with the (co-)evaluation map, aka cup or cap


def combine_legs(t: Tensor, legs: list[int | str], new_leg: ProductSpace = None) -> Tensor:
    """
    Combine a group of legs of a tensor. Resulting leg (of type ProductSpace) is at the
    previous position of legs[0].
    # TODO support multiple combines in one function call? what would the signature be
    # TODO inplace version
    """
    if len(legs) < 2:
        raise ValueError('expected at least two legs')

    leg_idcs = t.get_leg_idcs(legs)
    if new_leg is None:
        new_leg = ProductSpace([t.legs[idx] for idx in leg_idcs])
    old_legs = [t.legs[idx] for idx in leg_idcs]
    res_legs = [new_leg if idx == leg_idcs[0] else leg for idx, leg in enumerate(t.legs)
            if idx not in leg_idcs[1:]]
    new_label = combine_leg_labels(t.leg_labels)
    res_labels = [new_label if idx == leg_idcs[0] else label for idx, label in enumerate(t.leg_labels)
              if idx not in leg_idcs[1:]]
    res_data = t.backend.combine_legs(t, leg_idcs=leg_idcs, new_leg=new_leg)
    return Tensor(res_data, backend=t.backend, legs=res_legs, labels=res_labels)


def split_leg(t: Tensor, leg: int | str) -> Tensor:
    """
    Split a leg that was previously combined.
    If the legs were contiguous in t.legs before combining, this is the inverse operation of combine_legs,
    otherwise it is the inverse up to a permute_legs
    # TODO support multiple splits? -> make consistent with combine
    # TODO inplace version
    """
    leg_idx = t.get_leg_idx(leg)
    if not isinstance(t.legs[leg_idx]):
        raise ValueError(f'Leg {leg} is not a ProductSpace.')
    legs = t.legs[:leg_idx] + t.legs[leg_idx].spaces + t.legs[leg_idx + 1:]
    labels = t.labels[:leg_idx] + split_leg_label(t.labels[leg_idx]) + t.labels[leg_idx + 1:]
    res_data = t.backend.split_leg(t, leg_idx=leg_idx)
    return Tensor(res_data, backend=t.backend, legs=legs, labels=labels)


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs"""
    if isinstance(obj, (int, float, complex)):
        return True
    if isinstance(obj, Tensor):
        return all(l.is_trivial for l in obj.legs)
    else:
        raise TypeError(f'Type not supported for is_scalar: {type(obj)}')


def allclose(a: Tensor, b: Tensor, rtol=1e-05, atol=1e-08) -> bool:
    """
    If a and b are equal up to numerical tolerance, that is if `norm(a - b) <= atol + rtol * norm(a)`.
    Note that the definition is not symmetric under exchanging `a` and `b`.
    """
    assert rtol >= 0
    assert atol >= 0
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        diff = norm(a - b)
        a_norm = norm(a)
    else:
        if isinstance(a, Tensor):
            try:
                a = a.item()
            except ValueError:
                raise ValueError('Can not compare non-scalar Tensor and scalar') from None
        if isinstance(b, Tensor):
            try:
                b = b.item()
            except ValueError:
                raise ValueError('Can not compare scalar and non-scalar Tensor') from None
        diff = abs(a - b)
        a_norm = abs(a)
    return diff <= atol + rtol * a_norm


ALL_TRIVIAL_LEGS = object()


def squeeze_legs(t: Tensor, legs: int | str | list[int | str] = ALL_TRIVIAL_LEGS) -> Tensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If no legs are specified, all trivial legs are squeezed
    """
    if legs is ALL_TRIVIAL_LEGS:
        leg_idcs = [n for n, l in enumerate(t.legs) if l.is_trivial]
    else:
        leg_idcs = t.get_leg_idcs(legs)
        if not all(t.legs[idx].is_trivial for idx in leg_idcs):
            raise ValueError('Tried to squeeze non-trivial legs.')
    res_legs = [l for idx, l in enumerate(t.legs) if idx not in leg_idcs]
    res_labels = [label for idx, label in enumerate(t.labels) if idx not in leg_idcs]
    res_data = t.backend.squeeze_legs(t, leg_idcs)
    return Tensor(res_data, backend=t.backend, legs=res_legs, labels=res_labels)


def norm(t: Tensor) -> float:
    """2-norm of a tensor, i.e. sqrt(inner(t, t))"""
    return t.backend.norm(t)


def get_result_labels(legs1: list[str | None], legs2: list[str | None],
                      relabel1: dict[str, str] | None, relabel2: dict[str, str] | None) -> list[str]:
    """
    Utility function to combine two lists of leg labels, such that they can appear on the same tensor.
    Labels are changed by the mappings relabel1 and relabel2.
    Any conflicting labels (after relabelling) are dropped
    """
    relabel1 = relabel1 or {}
    relabel2 = relabel2 or {}

    labels1 = [relabel1.get(leg.label, leg.label) for leg in legs1]
    labels2 = [relabel2.get(leg.label, leg.label) for leg in legs2]
    conflicting = [label for label in labels1 if label in labels2]
    labels = labels1 + labels2
    if conflicting:
        # TODO issue warning
        labels = [None if label in conflicting else label for label in labels]
    return labels


def get_same_backend(*tensors: Tensor, error_msg: str = 'Incompatible backends.'):
    """If all tensors have the same backend, return it. Otherwise raise a ValueError"""
    try:
        backend = tensors[0].backend
    except IndexError:
        raise ValueError('expected at least one tensor') from None
    if not all(tens.backend == backend for tens in tensors):
        raise ValueError(error_msg)
    return backend
