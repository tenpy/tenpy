# """TODO
#
# >>> sym = U1("momentum")
# >>> backend = get_backend(sym, 'numpy')
# >>> leg_p = VectorSpace(backend, [1, -1], conj=False)
# >>> Sz = Tensor.from_dense([[1., 0.], [0, -1.]], [leg_p, leg_p.conj()], ['p', 'p*'])
# >>> Sz_2 = tensordot(Sz, Sz, 'p', 'p*')
# >>> Sz_2 = tensordot(Sz, Sz, ['p'], ['p*'])
# >>> Id = eye(leg_p, ['p'], ['p*'])
# >>> assert all_close(Sz_2, Id , eps=1.e-10)
# >>> Sz_2 + 5 * Id
#
#
# """
from __future__ import annotations

from math import prod

import numpy as np

from .backends.abstract_backend import AbstractBackend, Dtype
from .misc import force_str_len
from .symmetries.spaces import AbstractSpace, VectorSpace, ProductSpace
from .dummy_config import config


class LegPipe(ProductSpace):
    spaces: list[VectorSpace | LegPipe]

    def __init__(self, legs: list[VectorSpace | LegPipe], old_labels: list[str | None]):
        self.old_labels = old_labels[:]
        super().__init__(spaces=legs)

# TODO could have a global default_backend or sth and make all backend arguments optional


class Tensor:

    # TODO reminders:
    #  - I/O support (HDF5 and/or pickle support)
    #  - label utilities
    #    > get_leg_idx, get_leg_idcs, has_labels, labels_are
    #    > change labels, e.g. set_labels, update_labels, drop_labels

    # TODO: are tensors iterable? -> no! i.e. define __len__ and __iter__?
    #  Jakob: I think they shouldn't, it is not obvious what that means for non-abelian
    #         We could define it still and raise some error when it makes no sense to iterate

    # TODO: decide broadcasting rules. for now only same shapes are supported -> no!

    # TODO: implement a shape attribute / property? in principle it is determined by legs,
    #  but we could want a convenience object Shape which has legs as an attribute and several
    #  possible constructors / classmethods, e.g. we could allow Shape([2, 3, 1]) (implying no_symmetry)
    #  as well as Shape([vL_leg, p_leg, vR_leg]) with isinstance(p_leg, VectorSpace).
    #  the Shape type could then also work as a parameter to e.g. zeros, eye and random_uniform
    #  // rather do zeros_like, eye_like, ...

    # TODO make __eq__ be allclose with a default low tolerance

    # TODO jakob decided not to include the following dunders, revisit that decision
    #  - comparison: __lt__ etc, element-wise comparison is basis-dependent,
    #    also result would need to be a bool-valued Tensor which i dont want to support
    #  - __mul__, __pow__ where isinstance(other, Tensor).
    #    numpy conventions suggest elementwise operations here, which is basis-dependent and thus ill-defined
    #  - __floordiv__ and __mod__ since those would be integer-valued
    #  - boolean and bitwise operations e.g. __or__, __lshift__ etc
    #  - conversions to types other than float or complex, e.g. __int__

    def __init__(self, data, backend: AbstractBackend, legs: list[VectorSpace | LegPipe],
                 leg_labels: list[str | None], dtype: Dtype):
        """
        This constructor is not user-friendly. Use as_tensor instead.
        Inputs are not checked for consistency.
        """
        self.data = data
        self.backend = backend
        self.legs = legs
        self.num_legs = len(legs)
        self.dtype = dtype
        self.symmetry = backend.symmetry
        self._leg_labels = leg_labels[:]

    def check_sanity(self):
        assert all(leg.symmetry == self.backend.symmetry for leg in self.legs)
        assert all(isinstance(leg, (LegPipe, VectorSpace)) for leg in self.legs)
        assert self.num_legs == len(self.legs)
        assert self.dtype == self.backend.infer_dtype(self.data)
        #
        # checks on labels
        assert len(self._leg_labels) == self.num_legs
        assert all(l is None or isinstance(l, str) for l in self._leg_labels)
        str_labels = [l for l in self._leg_labels if l is not None]
        assert len(str_labels) == len(set(str_labels))  # checks that str_labels are unique
        if config.strict_labels:
            # check if labels are unique
            assert None not in self._leg_labels

    @property
    def size(self):
        """The total number of entries, i.e. the dimension of the tensorproduct of the legs,
        not considering symmetry"""
        return prod(l.dim for l in self.legs)

    @property
    def num_parameters(self):
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors with the same legs"""
        return self.backend.num_parameters(self.legs)

    @property
    def leg_labels(self):
        return self._leg_labels[:]  # return a copy, so that the private attribute can not be mutated

    @leg_labels.setter
    def leg_labels(self, leg_labels):
        raise AttributeError('Can not set Tensor.leg_labels. Use tenpy.linalg.set_labels() instead.')

    def get_leg_idx(self, leg: int | str) -> int:
        if isinstance(leg, int):
            assert 0 <= leg < self.num_legs
            return leg
        if isinstance(leg, str):
            return self._leg_labels.index(leg)
        raise TypeError

    def get_leg_idcs(self, legs: int | str | list[int | str]) -> list[int]:
        if isinstance(legs, (int, str)):
            return [self.get_leg_idx(legs)]
        else:
            return list(map(self.get_leg_idx, legs))

    def copy(self):
        """return a Tensor object equal to self, such that in-place operations on self.copy() do not affect self"""
        return Tensor(data=self.backend.copy_data(self.data), backend=self.backend, legs=self.legs[:],
                      leg_labels=self._leg_labels[:], dtype=self.dtype)

    def item(self):
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self.data)
        else:
            raise ValueError

    def __repr__(self):
        indent = '  '
        lines = [
            f'Tensor(',
            f'{indent}* Backend: {type(self.backend).__name__}'
            f'{indent}* Symmetry: {self.symmetry}',
            # TODO if we end up supporting qtotal, it should go here
            # TODO what is the right header in place of (?)
            f'{indent}* Legs:  label    dim  dual  components',
            f'{indent}         =================================================',  # TODO how long should this be?
        ]
        for leg, label in zip(self.legs, self._leg_labels):
            if isinstance(leg, LegPipe):
                dual = '   -'
                comps = f'LegPipe: {", ".join(leg.old_labels)}'
                comps = ', '.join(f'{sub_label}: {factor_space.dim}' for sub_label, factor_space
                                       in zip(leg.old_labels, leg.spaces))
            else:
                dual = ' yes' if leg.is_dual else '  no'
                comps = ', '.join(f'{self.symmetry.sector_str(sector)}: {mult}' for sector, mult in
                                       zip(leg.sectors, leg.multiplicities))
            lines.append(f'{indent}         {force_str_len(label, 5)}  {force_str_len(leg.dim, 5)}  {dual}  {comps}')
        lines.extend(self.backend._data_repr_lines(self.data, indent=indent, max_width=70, max_lines=20))
        lines.append(')')

    def __getitem__(self, item):
        # TODO indexing is another big topic with design choices to make
        #  - e.g. even forming T[2:7,:,0] for a non-abelian tensor cab involve expensive or even impossible
        #    (for anyonic spaces) computations and the result is (in general) not a symmetric tensor anymore,
        #    if the slice mis-aligns with the sectors.
        #  - should T[2:7,:,0] be a copy or a "view"?, e.g. does
        #    ```x = T[2:7, :, 0]; x[0, 0, 0] = 42``` change T?
        # TODO once the above choices are made, implement __setitem__.
        #  also for things like x[0, 0, 0] += 42, the result of __getitem__ should have compatible __iadd__ etc
        # TODO there are constraints to the format of item, maybe it makes sense to write a function
        #  get_slice or similar that supports a more general form, e.g. with labels
        # TODO maybe it would be nice to have some sort of label-indexer
        #  e.g. T.index_legs('a', 'c')[0, 0] would translate to T[0, :, 0] if the labels are ['a', 'b', 'c']
        raise NotImplementedError

    def __neg__(self):
        # TODO worth it to write specialized code here?
        return self.__mul__(-1)

    def __pos__(self):
        return self

    def __eq__(self, other):
        raise TypeError  # use all_close instead

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
            other = other.item()  # TODO handle errors
        if isinstance(other, (float, complex)):
            return mul(other, self)
        return NotImplemented

    __rmul__ = __mul__  # all allowed multiplications are commutative

    def __truediv__(self, other):
        # TODO worth it to write specialized code here?
        return self.__mul__(1. / other)

    def __float__(self):
        if not self.dtype.is_real:
            raise UserWarning  # TODO logging system
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __array__(self, dtype):
        return np.asarray(self.backend.to_dense_block(self.data), dtype)


class DiagonalTensor:

    # special case where incoming and outgoing legs are equal and the
    # tensor is "diagonal" (yet to precisely formulate what this means in a basis-independent way...)
    # this would be the natural type for the singular values of an SVD
    #  > no_symmetry: tensor, reshaped to matrix is diagonal
    #  > abelian: blocks, reshaped to matrices are diagonal
    #  > nonabelian: not only diagonal in coupled irrep, but also in its multiplicity, i.e. blocks are diagonal matrices
    # TODO revisit this when Tensor class and specification for data-structure of backend is "finished"
    # TODO this could implement element-wise operations such as __mul__ and __pow__, it would be well defined
    pass


def as_tensor(obj, backend: AbstractBackend, legs: list[VectorSpace] = None, labels: list[str] = None,
              dtype: Dtype = None) -> Tensor:
    if isinstance(obj, Tensor):
        assert (legs is None) or obj.legs == legs  # see eg transpose
        assert (backend is None) or obj.backend == backend  # TODO support switching backends somewhere else
        if (dtype is not None) and (obj.dtype != dtype):
            return Tensor(obj.backend.to_dtype(obj.data, dtype), obj.backend, legs=obj.legs,
                          leg_labels=obj.leg_labels if labels is None else labels, dtype=dtype)
        else:
            return obj

    else:
        obj = backend.parse_data(obj, dtype=None if dtype is None else backend.parse_dtype(dtype))
        if legs is None:
            legs = backend.infer_legs(obj)
        else:
            assert backend.legs_are_compatible(obj, legs)
        return Tensor(obj, backend, legs=legs, leg_labels=labels, dtype=dtype)


# FIXME stubs below


def sub(a, b):
    ...


def add(a, b):
    ...


def mul(a, b):
    ...

#
# def fuse_legs(..., allow_lazy: bool = True):
#     # if allow_lazy, the underlying data is unaffected.
#     # otherwise, the underlying data is reshaped.
#     ...
#
#
# def zeros(self, legs, ....):
#     ...
#
#
# def random_uniform_tensor(self, legs, ...)
#     return Tensor(...)
#
# def random_unitary_tensor(self, legs, ....):
#     return Tensor(...)
#
#
#
# def tensordot(A: Tensor, B: Tensor, contract_A, contract_B):
#     check_A_B_diagonal(A, B)
#     contract_A = _parse_indices(A, contract_A)
#     contract_B = _parse_indices(B, contract_B)
#     assert A.backend == B.backend
#     new_labels = _tensordot_new_labels(A, B, contract_A, contract_B)
#     new_data = backend.tensordot(A, B, contract_A, contract_B, new_labels)
#     return Tensor(...)
#
#
#
# def svd(A: Tensor, labels_A, label_B, new_label, new_conj=False,
#         trunc_params=None, ...):
#
#
# def eigh(A: Tensor, labels_left, labels_right):
#     ...
