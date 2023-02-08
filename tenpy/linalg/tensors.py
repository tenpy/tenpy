from __future__ import annotations
import enum

from math import prod

import numpy as np

from .backends.abstract_backend import AbstractBackend, Dtype
from .misc import duplicate_entries, force_str_len
from .symmetries.spaces import VectorSpace, ProductSpace


class Leg:
    def __init__(self, space: VectorSpace | ProductSpace, axs: list[int], label: str = None, 
                 fused_from: list[Leg] = None):
        """Data associated with the leg of a tensor

        Args:
            space (VectorSpace | ProductSpace): The space that this leg represents. Includes symmetry data
            axs (list[int]): which of the axes of the underlying backend-specific data this leg refers to
            label (str, optional): a string label used to identify this leg.
            fused_from (list[Leg], optional): if this leg is the result of fusing legs, a list of those legs.
                this data allows fusing to be reversed, and lazily evaluated.
        """
        self.space = space
        self.axs = axs
        self.num_axs = len(axs)
        self.label = label
        self.fused_from = fused_from

    @classmethod
    def combine(cls, legs: list[Leg], label: str = None) -> Leg:
        return cls(space=ProductSpace([l.space for l in legs]), axs=[ax for l in legs for ax in l.axs],
                   label=label, fused_from=legs)

    def split(self) -> list[Leg]:
        if self.fused_from is None:
            raise ValueError(f'Leg "{self.label}" can not be split')
        return self.fused_from

    @classmethod
    def trivial(cls, dim: int, ax: int, label: str = None, is_real: bool = True):
        return cls(space=VectorSpace.non_symmetric(dim=dim, is_real=is_real), axs=[ax], label=label)

    @property
    def is_trivial(self) -> bool:
        return self.space.dim == 1

    def dual(self) -> Leg:
        """The dual leg, that can be contracted with self"""
        return Leg(space=self.space.dual, axs=self.axs, label=dual_leg_label(self.label))

    def can_contract_with(self, other: Leg) -> bool:
        return self.space.is_dual_of(other.space)

    def components_str(self, max_len: int) -> str:
        if self.fused_from is not None:
            res = ' ⊗ '.join(f'({leg.label}: {leg.space.dim})' for leg in self.fused_from)
        elif isinstance(self.space, ProductSpace):
            res = ' ⊗ '.join(str(leg.space.dim) for leg in self.fused_from)
        else:
            res = ' ⊕ '.join(f'({mult} * {self.space.symmetry.sector_str(sector)})' 
                             for mult, sector in zip(self.space.multiplicities, self.space.sectors))

        if len(res) > max_len:
            res = res[:max_len - 6] + ' [...]'

        return res

    def copy(self):
        return Leg(space=self.space, axs=self.axs, label=self.label, fused_from=self.fused_from)

    def relabelled(self, label: str | None):
        """return a copy with a new label"""
        return Leg(space=self.space, axs=self.axs, label=label, fused_from=self.fused_from)


def dual_leg_label(label: str) -> str:
    """return the label that a leg should have after conjugation"""
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def combine_leg_labels(labels: list[str | None]) -> str:
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


class Tensor:

    def __init__(self, data, backend: AbstractBackend, legs: list[Leg]):
        """
        This constructor is not user-friendly. Use as_tensor instead.
        Inputs are not checked for consistency.
        """
        self.data = data
        self.backend = backend
        self.legs = legs
        self.num_legs = len(legs)
        self.symmetry = legs[0].space.symmetry
        self.label_map = {l.label: n for n, l in enumerate(legs) if l.label is not None}
        self.parent_space = ProductSpace(spaces=[l.space for l in legs])

    @property
    def dtype(self):
        return self.backend.infer_dtype(self.data)

    def check_sanity(self):
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.space.symmetry == self.symmetry for l in self.legs)
        all_axs = [ax for leg in self.legs for ax in leg.axs]
        expect_num_axs = self.backend.num_axs(self.data)
        assert not duplicate_entries(all_axs) 
        assert len(all_axs) == expect_num_axs
        assert all(0 <= ax < expect_num_axs for ax in all_axs)

    @property
    def size(self) -> int:
        """The total number of entries, i.e. the dimension of the tensorproduct of the legs,
        *not* considering symmetry"""
        return prod(l.dim for l in self.legs)

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors with the same legs"""
        return self.parent_space.num_parameters

    @property
    def leg_labels(self) -> list[str]:
        return [l.label for l in self.legs]

    @leg_labels.setter
    def leg_labels(self, leg_labels):
        self.set_labels(leg_labels)

    @property
    def is_fully_labelled(self) -> bool:
        return None not in self.leg_labels

    def set_labels(self, leg_labels: list[str]):
        assert not duplicate_entries(leg_labels, ignore=[None])
        assert len(leg_labels) == self.num_legs
        for leg, label in zip(self.legs, leg_labels):
            leg.label = label

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

    def get_legs(self, which_legs: int | str | list[int | str]) -> list[Leg]:
        return [self.legs[idx] for idx in self.get_leg_idcs(which_legs)]

    def get_all_axs(self, which_legs: int | str | list[int | str]) -> list[int]:
        """Get a list of all axes of the underlying data that are described by the given leg(s)"""
        if isinstance(which_legs, (int, str)):
            return self.legs[self.get_leg_idx(which_legs)].axs
        else:
            return [ax for which_leg in which_legs for ax in self.legs[self.get_leg_idx(which_leg)].axs]

    def copy(self):
        """return a Tensor object equal to self, such that in-place operations on self.copy() do not affect self"""
        return Tensor(data=self.backend.copy_data(self.data), backend=self.backend, legs=self.legs[:])

    def item(self):
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self.data)
        else:
            raise ValueError('Not a scalar')

    def __repr__(self):
        indent = '  '

        label_strs = [force_str_len(l.label, 5) for l in self.legs]
        dim_strs = [force_str_len(l.dim, 5) for l in self.legs]
        components_strs = [l.components_str(max_len=50) for l in self.legs]

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
        # TODO worth it to write specialized code here?
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
            return sub(self, other)
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


def as_tensor(obj, backend: AbstractBackend, legs: list[Leg] = None, labels: list[str] = None,
              dtype: Dtype = None) -> Tensor:
    if isinstance(obj, Tensor):
        obj = obj.copy()

        if legs is not None:
            assert labels is None
            raise NotImplementedError  # TODO what to do here?

        if backend is not None:
            raise NotImplementedError  # TODO

        if labels is not None:
            obj.set_labels(labels)

        if dtype is not None:
            obj.data = obj.backend.to_dtype(obj.data, dtype)

        obj.check_sanity()
        return obj

    else:
        obj = backend.parse_data(obj, dtype=None if dtype is None else backend.parse_dtype(dtype))
        if legs is None:
            legs = backend.infer_legs(obj, labels=labels)
        else:
            assert labels is None
            assert backend.legs_are_compatible(obj, legs)
        return Tensor(obj, backend, legs=legs)


# FIXME stubs below


def sub(a, b):
    ...


def add(a, b):
    ...


def mul(a, b):
    ...
