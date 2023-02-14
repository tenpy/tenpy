from __future__ import annotations

from typing import Iterable
import numpy as np

from .misc import duplicate_entries, force_str_len
from .dummy_config import config
from .symmetries import VectorSpace, ProductSpace


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
        self.labels = labels or [None] * len(legs)
        self.num_legs = len(legs)
        self.symmetry = legs[0].space.symmetry

    @property
    def dtype(self):
        return self.backend.infer_dtype(self.data)

    @property
    def parent_space(self) -> VectorSpace:
        if self.num_legs == 1:
            return self.legs[0]
        else:
            return ProductSpace(spaces=self.legs)

    def check_sanity(self):
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.space.symmetry == self.symmetry for l in self.legs)
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

    @property
    def is_fully_labelled(self) -> bool:
        return None not in self.labels

    def has_label(self, label: str, *more: str) -> bool:
        return label in self.labels and all(l in self.labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        return set(self.labels) == set(labels)

    def set_labels(self, labels: list[str | None]):
        assert not duplicate_entries(labels, ignore=[None])
        assert len(labels) == self.num_legs
        self.labels = labels[:]

    def get_leg_idx(self, which_leg: int | str) -> int:
        if isinstance(which_leg, str):
            which_leg = self.label_map[which_leg]
        if isinstance(which_leg, int):
            if which_leg < 0:
                which_leg = which_leg + self.num_legs
            assert 0 <= which_leg < self.num_legs
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
            return self.backend.item(self.data)
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

    def __getitem__(self, item):
        # TODO point towards a "as flat" option
        raise TypeError('Tensor object is not subscriptable')

    def __neg__(self):
        return self.__mul__(-1)

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
            try:
                other = other.item()
            except ValueError:
                raise ValueError('Tensors can only be multiplied with scalars') from None
        if isinstance(other, (float, complex)):
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
#             obj.data = obj.backend.to_dtype(obj.data, dtype)

#         obj.check_sanity()
#         return obj

#     else:
#         obj, shape = backend.parse_data(obj, legs, dtype=backend.parse_dtype(dtype))
#         if legs is None:
#             legs = [VectorSpace.non_symmetric(d) for d in shape]
#         else:
#             assert backend.legs_are_compatible(obj, legs)
#         return Tensor(obj, backend, legs=legs, labels=labels)


def match_label_order(a: Tensor, b: Tensor) -> Iterable[int]:
    """Determine the order of legs of b, such that they match the legs of a.
    If config.stric_labels, this is a permutation determined by the labels, otherwise it is range(num_legs).
    """
    if config.strict_labels:
        if a.is_fully_labelled and b.is_fully_labelled:
            match_by_labels = True
        else:
            match_by_labels = False
            # TODO issue warning?
    else:
        match_by_labels = False

    if match_by_labels:
        leg_order = b.get_leg_idcs(a.leg_labels)
    else:
        leg_order = range(b.num_legs)


def add(a: Tensor, b: Tensor) -> Tensor:
    # TODO moved import here to avoid import cycle
    from .numpy import get_same_backend
    backend = get_same_backend(a, b)
    res_data = backend.add(a.data, b.data, b_perm=match_label_order(a, b))
    return Tensor(res_data, backend=backend, legs=a.legs, labels=a.labels)


def mul(a: float | complex, b: Tensor) -> Tensor:
    res_data = b.backend.mul(a, b.data)
    return Tensor(res_data, backend=b.backend, legs=b.legs, labels=b.labels)




