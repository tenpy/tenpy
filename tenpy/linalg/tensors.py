r"""


.. _tensor_leg_labels:

Leg Labels
----------

TODO elaborate

The following characters have special meaning in labels and should be avoided:
`(`, `.`, `)`, `?`, `!` and `*`.


.. _tensors_as_maps:

Tensors as Maps
---------------

We can view :math:`m \times n` matrices either as linear maps
:math:`\mathbb{C}^n \to \mathbb{C}^m` or as elements of the space
:math:`\mathbb{C}^n \otimes \mathbb{C}^m^*`, which is known in the context of mixed state
density matrices as "vectorization".

Similarly, we can view any tensor, i.e. elements of tensor product spaces as linear maps.
TODO elaborate.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABCMeta, abstractmethod
import operator
from types import EllipsisType
from typing import TypeVar, Sequence, NoReturn
from numbers import Number, Integral
import numpy as np
import warnings
import functools
import logging
logger = logging.getLogger(__name__)

from .dummy_config import printoptions
from .misc import duplicate_entries, join_as_many_as_possible
from .dummy_config import config
from .symmetries import AbelianGroup, Symmetry, SymmetryError
from .spaces import Space, ElementarySpace, ProductSpace, Sector, SectorArray
from .backends.backend_factory import get_backend
from .backends.abstract_backend import Block, Backend
from .dtypes import Dtype
from ..tools.misc import to_iterable, to_iterable_of_len
from ..tools.docs import amend_parent_docstring

__all__ = ['Shape', 'Tensor', 'SymmetricTensor', 'BlockDiagonalTensor', 'ChargedTensor',
           'DiagonalTensor', 'Mask', 'add_trivial_leg', 'almost_equal', 'combine_legs', 'conj',
           'detect_sectors_from_block', 'entropy', 'flip_leg_duality', 'hconj', 'inner',
           'is_scalar', 'norm', 'outer', 'permute_legs', 'split_legs', 'squeeze_legs', 'tdot',
           'trace', 'zero_like', 'eye_like', 'angle', 'real', 'imag', 'real_if_close', 'sqrt',
           'get_same_backend', 'tensor_from_block']

# svd, qr, eigen, exp, log, ... are implemented in matrix_operations.py


# TODO where to put this? where to doc it?
BOTH = object()  # sentinel value for the default behavior of DiagonalTensor.apply_mask
KEEP_OLD_LABEL = object()  # sentinel value, e.g. for apply_mask


class Shape:
    """An object storing the legs and labels of a tensor.
    When iterated or indexed, it behaves like a sequence of integers, the dimension of the legs.
    Can be indexed by integer (leg position) or string (leg label).
    """

    def __init__(self, legs: list[Space], num_domain_legs: int, labels: list[str | None] = None):
        self.legs = legs
        if labels is None:
            labels = [None] * len(legs)
        else:
            labels = labels[:]
        self._labels = labels
        self._labelmap = {label: leg_num for leg_num, label in enumerate(self.labels) if label is not None}
        self.num_legs = len(legs)
        self.dims = [l.dim for l in legs]
        self.num_domain_legs = num_domain_legs

    def test_sanity(self):
        assert not duplicate_entries(self._labels, ignore=[None])
        assert 0 <= self.num_domain_legs <= self.num_legs

    def __iter__(self):
        return iter(self.dims)

    def __len__(self):
        return self.num_legs

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = self.label_to_legnum(key)
            except ValueError:
                raise IndexError(f'No leg with label {key}.') from None
        return self.dims[key]

    def __eq__(self, other):
        if isinstance(other, Shape):
            return self.legs == other.legs and self._labels == other._labels
        return False

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
            raise ValueError(f'No leg with label {label}. Labels are {self.labels}')
        return num

    def relabel(self, mapping: dict[str, str]) -> None:
        """Apply mapping to labels. In-place."""
        self.set_labels([mapping.get(l, l) for l in self._labels])

    def __str__(self):
        entries = [str(dim) if label is None else f'{label}:{dim:d}'
                   for label, dim in zip(self._labels, self.dims)]
        return '(' + ', '.join(entries[:self.num_domain_legs]) + ' ; ' \
            + ', '.join(entries[self.num_domain_legs:]) + ')'

    def __repr__(self) -> str:
        dims = ', '.join((f"{lbl}:{d:d}" if lbl is not None else str(d))
                         for lbl, d in zip(self._labels, self.dims))
        return f'Shape({dims}, num_domain_legs={self.num_domain_legs})'


# ##################################
# tensor classes
# ##################################


class Tensor(metaclass=ABCMeta):
    """Common abstract base class for tensors.

    .. note ::
        TODO write clean text about ElementarySpace.basis_perm and how it affects internal storage.

    Parameters
    ----------
    legs : list[Space]
        The legs of the Tensor
    backend: :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    def __init__(self, legs: list[Space], backend: Backend, labels: list[str | None] | None,
                 dtype: Dtype, num_domain_legs: int):
        if backend is None:
            self.backend = backend = get_backend(legs[0].symmetry)
        else:
            self.backend = backend
        self.legs = legs = legs[:]  # FIXME rm ok?
        self.shape = Shape(legs=legs, num_domain_legs=num_domain_legs, labels=labels)
        self.num_legs = len(legs)
        self.symmetry = legs[0].symmetry
        self.dtype = dtype

    def test_sanity(self) -> None:
        assert self.backend.supports_symmetry(self.symmetry)
        assert all(l.symmetry == self.symmetry for l in self.legs)
        assert len(self.legs) == self.shape.num_legs == self.num_legs > 0
        for leg in self.legs:
            self.backend.test_leg_sanity(leg)
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
    def num_domain_legs(self) -> int:
        """How many of the legs are in the domain. See :ref:`tensors_as_maps`."""
        return self.shape.num_domain_legs

    @property
    def num_codomain_legs(self) -> int:
        """How many of the legs are in the codomain. See :ref:`tensors_as_maps`."""
        return self.num_legs - self.num_domain_legs

    @property
    def num_parameters(self) -> int:
        """The number of free parameters, i.e. the dimension of the space of symmetry-preserving
        tensors with the same legs"""
        return self.parent_space.num_parameters

    @functools.cached_property
    def parent_space(self) -> ProductSpace:
        """The space that the tensor lives in"""
        return ProductSpace(self.legs, backend=self.backend)

    @property
    def size(self) -> int:
        """The total number of entries, i.e. the dimension of the space of tensors on the same space
        if symmetries were ignored"""
        return self.parent_space.dim

    def flip_leg_duality(self, which_leg: int | str, *more: int | str) -> Tensor:
        """See :func:`tensors.flip_leg_duality`"""
        which_legs = self.get_leg_idcs([which_leg, *more])
        flipped_legs = []
        perms = []
        for i in which_legs:
            flipped, perm = self.legs[i].flip_is_dual(return_perm=True)
            flipped_legs.append(flipped)
            perms.append(perm)
        res = self.copy(deep=False)
        res.data = self.backend.flip_leg_duality(self, which_legs=which_legs,
                                                 flipped_legs=flipped_legs, perms=perms)
        for i, leg in zip(which_legs, flipped_legs):
            res.legs[i] = leg
        return res

    def with_legs(self, *legs: int | str) -> _TensorIndexHelper:
        """This method allows indexing a tensor "by label".

        It returns a helper object, that can be indexed instead of self.
        For example, if we have a tensor with labels 'a', 'b' and 'c', but we are not sure about
        their order, we can call ``tensor.with_legs('a', 'b')[0, 1]``.
        If ``tensors.labels == ['a', 'b', 'c']`` in alphabetic order, we get ``tensor[0, 1]``.
        However if the order of labels happens to be different, e.g.
        ``tensor.labels == ['b', 'c', 'a']`` we get ``tensor[1, :, 0]``.
        """
        return _TensorIndexHelper(self, legs)

    def hconj(self, legs1: int | str | list[int | str] = -1,
              legs2: int | str | list[int | str] = None) -> Tensor:
        """See :func:`tensors.hconj`"""
        legs1 = np.array(self.get_leg_idcs(legs1))
        if legs2 is None:
            # TODO match by labels instead?
            legs2 = [n for n in range(self.num_legs) if n not in legs1]
        else:
            legs2 = np.array(self.get_leg_idcs(legs2))
        assert not duplicate_entries(legs1 + legs2)
        assert len(legs1) == len(legs2) == self.num_legs // 2
        labels = self.labels
        permutation = np.zeros((self.num_legs,), dtype=int)
        for i1, i2 in zip(legs1, legs2):
            assert self.legs[i1] == self.legs[i2].dual
            assert labels[i1] == _dual_leg_label(labels[i2])  # accepts None as well
            permutation[i1] = i2
            permutation[i2] = i1
        return self.conj().permute_legs(permutation)

    def has_label(self, label: str, *more: str) -> bool:
        return label in self.shape._labels and all(l in self.shape._labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        if not self.is_fully_labelled:
            return False
        if len(labels) != len(self.shape._labels):
            return False
        return set(labels) == set(self.shape._labels)

    def relabel(self, mapping: dict[str, str | None], inplace=True) -> Tensor:
        """Re-label by applying a mapping to the labels."""
        if not inplace:
            return self.copy(deep=False).relabel(mapping, inplace=True)
        self.shape.relabel(mapping)
        return self

    def set_labels(self, labels: list[str | None]) -> None:
        self.shape.set_labels(labels)

    def get_leg_idx(self, which_leg: int | str) -> int:
        if isinstance(which_leg, str):
            which_leg = self.shape.label_to_legnum(which_leg)
        if isinstance(which_leg, Integral):
            if which_leg < 0:
                which_leg = which_leg + self.num_legs
            if not 0 <= which_leg < self.num_legs:
                raise ValueError(f'Leg index out of bounds: {which_leg}.') from None
            return which_leg
        else:
            raise TypeError(f'Expected int or str. Got {type(which_leg)}')

    def get_leg_idcs(self, which_legs: int | str | list[int | str]) -> list[int]:
        if isinstance(which_legs, (Integral, str)):
            return [self.get_leg_idx(which_legs)]
        else:
            return list(map(self.get_leg_idx, which_legs))

    def get_leg(self, which_leg: int | str) -> Space:
        """Get a single leg of the tensor

        Parameters
        ----------
        which_leg : int | str
            Either the position (``int``) or label (``str``) of the desired leg.

        See Also
        --------
        get_legs
        """
        return self.legs[self.get_leg_idx(which_leg)]

    def get_legs(self, which_legs: int | str | list[int | str]) -> list[Space]:
        """Get a number of legs of the tensor

        Parameters
        ----------
        which_legs : int | str | list of int | list of str
            For every desired leg, either the position (``int``) or label (``str``).

        See Also
        --------
        get_leg
        """
        return [self.legs[idx] for idx in self.get_leg_idcs(which_legs)]

    def to_numpy(self, leg_order: list[int | str] = None, numpy_dtype=None) -> np.ndarray:
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
                sublabels = [f'?{n}' if l is None else l
                             for n, l in enumerate(_split_leg_label(label, num=len(leg.spaces)))]
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
        # vertical table for the legs
        labels = ['label'] + [str(l) for l in self.labels]
        dims = ['    dim'] + [str(leg.dim) for leg in self.legs]
        sector_nums = ['is_dual'] + [{True: 'yes', False: ' no'}[leg.is_dual] for leg in self.legs]
        col_widths = [max(len(l), len(d), len(n)) for l, d, n in zip(labels, dims, sector_nums)]

        # put '|' between legs and '||' between domain and codomain
        n = self.num_domain_legs + 1  # +1 since there is a first header column
        labels_row = ' | '.join(x.rjust(w) for x, w in zip(labels[:n], col_widths[:n]))
        labels_row += ' || ' + ' | '.join(x.rjust(w) for x, w in zip(labels[n:], col_widths[n:]))
        dims_row = ' | '.join(x.rjust(w) for x, w in zip(dims[:n], col_widths[:n]))
        dims_row += ' || ' + ' | '.join(x.rjust(w) for x, w in zip(dims[n:], col_widths[n:]))
        sectors_nums_row = ' | '.join(x.rjust(w) for x, w in zip(sector_nums[:n], col_widths[:n]))
        sectors_nums_row += ' || ' + ' | '.join(x.rjust(w) for x, w in zip(sector_nums[n:], col_widths[n:]))
        if self.num_domain_legs == self.num_legs:
            labels_row += ' ||'
            dims_row += ' ||'
            sectors_nums_row += ' ||'
        
        lines = [
            f'{indent}* Backend: {self.backend}',
            f'{indent}* Symmetry: {self.symmetry}',
            f'{indent}* Legs: {labels_row}',
            f'{indent}        {dims_row}',
            f'{indent}        {sectors_nums_row}',
        ]
        return lines

    def _getitem_apply_masks(self, masks: list[Mask], legs: list[int]) -> Tensor:
        """Helper function for __getitem__ to index self with masks.
        Subclasses may override this implementation."""
        res = self
        for mask, leg in zip(masks, legs):
            res = res.apply_mask(mask, leg)
        return res

    def _input_checks_inner(self, other: Tensor, do_conj: bool, legs1: list[int | str] | None,
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
            leg_order_2 = _match_leg_order(self, other)
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

    def _input_checks_same_legs(self, other: Tensor) -> list[int] | None:
        """Common input checks for functions that expect two tensors with the same set of legs.

        Returns
        -------
        other_order : (list of int) or None
            The order of legs on other, such that they match the legs of self.
            None is equivalent to ``list(range(other.num_legs)`` and indicates that no permutation is needed.
        """
        other_order = _match_leg_order(self, other)
        for n in range(self.num_legs):
            leg_self = self.legs[n]
            leg_other = other.legs[n] if other_order is None else other.legs[other_order[n]]
            if leg_self != leg_other:
                self_label = self.shape._labels[n]
                self_label = '' if self_label is None else self_label + ': '
                other_label = other.shape._labels[n]
                other_label = '' if other_label is None else other_label + ': '
                msg = '\n'.join([
                    'Incompatible legs:',
                    self_label + str(leg_self),
                    other_label + str(leg_other)
                ])
                raise ValueError(msg)
        return other_order

    def _input_checks_tdot(self, other: Tensor, legs1: int | str | list[int | str], legs2: int | str | list[int | str]
                           ) -> tuple[list[int], list[int]]:
        _legs1 = to_iterable(legs1)
        _legs2 = to_iterable(legs2)
        legs1 = list(map(self.get_leg_idx, _legs1))
        legs2 = list(map(other.get_leg_idx, _legs2))
        if duplicate_entries(legs1) or duplicate_entries(legs2):
            raise ValueError('legs may not contain duplicates')
        if len(legs1) != len(legs2):
            raise ValueError('Mismatching number of legs')
        incompatible = []
        for _l1, _l2, l1, l2 in zip(_legs1, _legs2, legs1, legs2):
            if not self.legs[l1].can_contract_with(other.legs[l2]):
                incompatible.append((_l1, _l2))
        if incompatible:
            msg = f'{len(incompatible)} incompatible leg pairs: {", ".join(map(str, incompatible))}'
            raise ValueError(msg)
        return legs1, legs2

    def __repr__(self):
        indent = printoptions.indent * ' '
        lines = [f'{self.__class__.__name__}(']
        lines.extend(self._repr_header_lines(indent=indent))
        if not printoptions.skip_data:
            lines.extend(self._data_repr_lines(indent=indent, max_lines=printoptions.maxlines_tensors - len(lines) - 1))
        lines.append(')')
        return "\n".join(lines)

    def _data_repr_lines(self, indent: str, max_lines: int) -> list[str]:
        return self.backend._data_repr_lines(
            self, indent=indent, max_width=printoptions.linewidth, max_lines=max_lines
        )

    def _parse_integer_indices(self, idcs):
        if not all(isinstance(idx, int) for idx in idcs[1:]):
            msg = 'Invalid index type. If tensors are indexed by integer, all legs need to be indexed by an integer.'
            raise IndexError(msg)
        for leg_num, idx in enumerate(idcs):
            leg = self.legs[leg_num]
            if not -leg.dim <= idx < leg.dim:
                msg = f'Index {idx} is out of bounds for leg {leg_num} with label {self.labels[leg_num]} ' \
                        f'and dimension {leg.dim}'
                raise IndexError(msg)
            if idx < 0:
                idx = idx + leg.dim
            if leg._inverse_basis_perm is not None:
                idx = leg._inverse_basis_perm[idx]
            idcs[leg_num] = idx
        return idcs
        
    def __getitem__(self, idcs):
        """
        TODO eventually we should document this at some high-level place, e.g. in one of the rst files
        Collecting snippets here for now

        We support two modes of indexing tensors for __getitem__:
        - Getting single entries, i.e. giving one integer per leg
        - Getting a "masked" Tensor, i.e. giving a Mask for some or all legs.
          Legs not to be masked can be indicated via ``slice(None, None, None)``, i.e. typing ``:``,
          or ``Ellipsis``, i.e. typing ``...``.

        For ``DiagonalTensor`` and ``Mask`` we additionally support indexing by a single integer `i`.
        For ``DiagonalTensor``, this returns the diagonal element, i.e. ``diag[i] == diag[i, i]``.
        For ``Mask``, this is the boolean entry that indicates if the ``i-th`` index is preserved
        or projected out by the mask, i.e. ``mask[i] == mask[i, j_i]`` where loosely ``j_i = sum(mask[:i])``.
        """
        idcs = _parse_idcs(idcs, length=self.num_legs)
        if isinstance(idcs[0], int):
            idcs = self._parse_integer_indices(idcs)
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
        idcs = _parse_idcs(idcs, length=self.num_legs)
        if not isinstance(idcs[0], int):
            raise IndexError('Can only set single entries')
        idcs = self._parse_integer_indices(idcs)
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
        if isinstance(other, Tensor):
            return self._add_tensor(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self.__add__(other._mul_scalar(-1))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._mul_scalar(other)
        raise TypeError(f'Tensors can only be multiplied with scalars, not {type(other)}.') from None

    def __rmul__(self, other):
        # all allowed multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f'Tensors can only be divided by scalars, not {type(other)}.') from None
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
    def zero(cls, legs: Space | list[Space], backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.float64) -> Tensor:
        """A zero tensor"""
        ...

    @abstractmethod
    def add_trivial_leg(self, label: str = None, is_dual: bool = False, pos: int = -1,
                        to_domain: bool = None) -> Tensor:
        """See tensors.add_trivial_leg"""
        ...

    @abstractmethod
    def apply_mask(self, mask: Mask, leg: int | str) -> Tensor:
        """Apply a mask to one of the legs, projecting to a smaller leg.

        If the masked leg is a :class:`ProductSpace`, the product structure is dropped while masking
        and the masked leg will be only a :class:`ElementarySpace`, not a :class:`ProductSpace`.
        See notes below.

        Parameters
        ==========
        mask : Mask
            The mask to be applied
        leg : int | str
            Which leg to apply to

        Notes
        =====
        It would be possible to implement Mask-application in a way that keeps the product structure.
        TODO (JU) is this even true for non-abelian?
        That would, however, make `split_legs` more complicated, thus we keep it simple.
        If you really want to project and split afterwards, use the following work-around,
        which is for example used in :class:`~tenpy.algorithms.exact_diagonalization`:

        1) Save the unprojected ProductSpace separately.
        2) Apply the mask, yielding a tensor with a non-ProductSpace leg
        3) [... do calculations ...]
        4) To split the 'projected ProductSpace' of `A`, create an zero Tensor `B` with the legs of A,
           but replace the projected leg by the full ProductSpace. Set `A` as a slice of `B`.
           Finally split the ProductSpace leg.
        """
        ...

    @abstractmethod
    def as_Tensor(self) -> BlockDiagonalTensor:
        """Convert to Tensor."""
        ...

    @abstractmethod
    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace] = None,
                     product_spaces_dual: list[bool] = None,
                     new_axes: list[int] = None,
                     new_labels: list[str | None] = None) -> Tensor:
        """See :func:`tenpy.linalg.tensors.combine_legs`."""
        ...

    @abstractmethod
    def conj(self) -> Tensor:
        """See :func:`tensors.conj`"""
        ...

    @abstractmethod
    def copy(self, deep=True) -> Tensor:
        ...

    @abstractmethod
    def item(self) -> bool | float | complex:
        """If the tensor is a scalar (i.e. has only one entry), return that scalar as a float or complex.
        Otherwise raise a ValueError"""
        ...

    @abstractmethod
    def norm(self, order=None) -> float:
        """See tensors.norm"""
        ...

    @abstractmethod
    def permute_legs(self, permutation: list[int] = None, num_domain_legs: int = None) -> Tensor:
        """See tensors.permute_legs"""
        ...

    @abstractmethod
    def split_legs(self, legs: list[int | str] = None) -> Tensor:
        """See tensors.split_legs"""
        ...

    @abstractmethod
    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> Tensor:
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
              ) -> Tensor | float | complex:
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
            Indices are w.r.t. to the internal basis order, i.e. `basis_perm` is already accounted
            for, but `sector_perm` is not.
        """
        ...

    @abstractmethod
    def _mul_scalar(self, other: complex) -> Tensor:
        ...

    @abstractmethod
    def _set_element(self, idcs: list[int], value: bool | float | complex) -> None:
        """Helper function for __setitem__ after arguments were parsed.
        Can assume that idcs has correct length and entries are valid & non-negative (0 <= idx < dim).
        Indices are w.r.t. to the internal basis order, i.e. `basis_perm` is already accounted
        for, but `sector_perm` is not.
        Modifies self in-place with a modified copy of the underlying data.
        """
        ...

    # ----------------------------------
    # Abstract binary tensor methods
    # ----------------------------------
    #  -> concrete implementations need to distinguish type of `other`

    @abstractmethod
    def almost_equal(self, other: Tensor, atol: float = 1e-5, rtol: float = 1e-8,
                     allow_different_types: bool = False) -> bool:
        """See tensors.almost_equal"""
        ...

    @abstractmethod
    def inner(self, other: Tensor, do_conj: bool = True,
              legs1: list[int | str] = None, legs2: list[int | str]  = None) -> float | complex:
        """See tensors.inner"""
        ...

    @abstractmethod
    def outer(self, other: Tensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> Tensor:
        """See tensors.outer"""
        ...

    @abstractmethod
    def tdot(self, other: Tensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> Tensor | float | complex:
        """See tensors.tdot"""
        ...

    @abstractmethod
    def _add_tensor(self, other: Tensor) -> Tensor:
        ...


class SymmetricTensor(Tensor):
    """Common base class for tensors which are symmetric (i.e. not charged)"""
    pass


class BlockDiagonalTensor(SymmetricTensor):
    """

    Attributes
    ----------
    data
        backend-specific data structure that contains the numerical data, i.e. the free parameters
        of tensors with the given symmetry.
        data about the symmetry is contained in the legs.
    backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
    legs : list of :class:`~tenpy.linalg.spaces.Space`
    labels : list of {``None``, str}
    """

    def __init__(self, data, legs: list[Space], num_domain_legs: int, backend=None,
                 labels: list[str | None] | None = None):
        """
        This constructor is not user-friendly.
        Use as_tensor instead.
        Inputs are not checked for consistency.

        Parameters
        ----------
        data
            The numerical data ("free parameters") comprising the tensor. type is backend-specific
        legs : list[Space]
            The legs of the Tensor
        num_domain_legs : int
            How many of the legs are in the domain. See :ref:`tensors_as_maps`.
        backend: :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
            The backend for the Tensor
        labels : list[str | None] | None
            Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        dtype = backend.get_dtype_from_data(data)
        Tensor.__init__(self, backend=backend, legs=legs, labels=labels, dtype=dtype,
                        num_domain_legs=num_domain_legs)
        self.data = data
        assert isinstance(data, self.backend.DataCls)

    def test_sanity(self) -> None:
        super().test_sanity()
        self.backend.test_data_sanity(self, is_diagonal=False)
        assert self.dtype != Dtype.bool

    # --------------------------------------------
    # Additional methods (not in Tensor)
    # --------------------------------------------

    @classmethod
    def eye(cls, legs: Space | list[Space], backend=None,
            labels: list[str | None] = None, dtype: Dtype = Dtype.float64) -> BlockDiagonalTensor:
        """The identity map on a group of legs, as a tensor with twice as many legs.

        For a single leg, this is the identity matrix.
        Note the leg order. E.g. for two legs ``p, q`` we have
        ``eye([p, q]).legs == [p, q, q.dual, p.dual]``, where the duals are in *reverse order*.

        Parameters
        ----------
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend for the Tensor
        legs : (list of) Space
            *Half* of the legs of the result.
            The result has legs ``[*legs, *(l.dual for l in reversed(legs))]``,
            the given `legs` followed by their respective duals in reverse order.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
            Can either give one label for each of the `legs`, and the second half will be the
            respective dual labels, or give twice as many and specify them all.
        dtype : Dtype, optional
            The data type of the Tensor entries.
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        legs = to_iterable(legs)
        half_leg_num = len(legs)
        # legs = [backend.add_leg_metadata(leg) for leg in legs]  # FIXME
        data = backend.eye_data(legs=legs, dtype=dtype)
        legs = legs + [leg.dual for leg in reversed(legs)]
        if labels is None:
            labels = [None] * (2 * half_leg_num)
        if len(labels) == half_leg_num:
            labels = labels + [_dual_leg_label(l) for l in reversed(labels)]
        if len(labels) != 2 * half_leg_num:
            msg = f'Wrong number of labels. Expected {half_leg_num} or {2 * half_leg_num}. Got {len(labels)}.'
            raise ValueError(msg)
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=half_leg_num, labels=labels)

    @classmethod
    def from_block_func(cls, func, legs: list[Space], backend=None,
                        labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None, num_domain_legs: int = 0
                        ) -> BlockDiagonalTensor:
        """Create a Tensor from a block function.

        This function creates a tensor by filling the blocks, i.e. the free parameters of the tensors
        using `func`, which is a function returning backend-specific blocks.

        Parameters
        ----------
        func : callable
            A callable object which is called to generate the blocks.
            We expect that `func` returns a backend-specific block of the given `shape`.
            If no `shape_kw` is given, it is called as ``func(shape, **func_kwargs)``,
            otherwise as ``func(**{shape_kw: shape}, **func_kwargs)``,
            where `shape` is a tuple of int.
        legs : list of Space
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.

        See Also
        --------
        from_numpy_func
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        # legs = [backend.add_leg_metadata(leg) for leg in legs]  # FIXME

        if shape_kw is not None:
            def block_func(shape):
                block = func(**{shape_kw: shape}, **func_kwargs)
                if dtype is not None:
                    block = backend.block_to_dtype(block, dtype)
                return block
        else:
            def block_func(shape):
                block = func(shape, **func_kwargs)
                if dtype is not None:
                    block = backend.block_to_dtype(block, dtype)
                return block

        data = backend.from_block_func(block_func, legs, num_domain_legs)
        res = cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs, labels=labels)
        res.test_sanity()  # this catches e.g. errors where the block_func returns blocks of wrong shape
        return res

    @classmethod
    def from_dense_block(cls, block, legs: list[Space], backend=None, dtype: Dtype=None,
                         labels: list[str | None] = None, tol: float = 1e-8,
                         num_domain_legs: int = 0) -> BlockDiagonalTensor:
        """Convert a dense block of the backend to a Tensor.

        Parameters
        ----------
        block : Block-like
            The data to be converted to a Tensor as a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`.
            This includes e.g. nested python iterables or numpy arrays.
            The block should be given in the "public" basis order of the `legs`, e.g.
            according to :attr:`ElementarySpace.sectors_of_basis`.
        legs : list of :class:`~tenpy.linalg.spaces.Space`, optional
            The vectorspaces associated with legs of the tensors. This specifies the symmetry.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
            The backend for the Tensor
        dtype : ``np.dtype``, optional
            The data type of the Tensor entries. Defaults to dtype of `block`
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
        tol : float or None
            By default, we check if the block is symmetric according to
            ``norm(block - symmetric_components) < tol * norm(block)``.
            This can be disabled by setting ``tol=None``, such that we always just return the
            symmetric components of ``block``.
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        block = backend.as_block(block)
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        data = backend.from_dense_block(block, legs=legs, tol=tol, num_domain_legs=num_domain_legs)
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs, labels=labels)

    @classmethod
    def from_flat_block_trivial_sector(cls, leg: Space, block: Block, backend: Backend,
                                       label: str = None) -> BlockDiagonalTensor:
        """Create a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.

        Parameters
        ----------
        leg : :class:`~tenpy.linalg.spaces.Space`
            The single leg of the resulting tensor
        block : backend-specific Block
            The block of shape ``(M,)`` where ``M`` is the multiplicity of the trivial sector in `leg`.
            This is a slice of ``result.to_dense_block()``.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend of the resulting tensor
        label : str | None
            The label for the single leg

        See Also
        --------
        to_flat_block_trivial_sector
        """
        return cls(data=backend.from_flat_block_trivial_sector(block, leg=leg), backend=backend,
                   legs=[leg], num_domain_legs=0, labels=[label])

    @classmethod
    def from_numpy_func(cls, func, legs: list[Space], backend=None,
                        labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None, num_domain_legs: int = 0
                        ) -> BlockDiagonalTensor:
        """Create a Tensor from a numpy function.

        This function creates a tensor by filling the blocks, i.e. the free parameters of the tensors
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
        legs : list of Space
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        func_kwargs : dict
            Additional keyword arguments given to `func`.
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.
        dtype : None | Dtype
            If given, the results of `func` are converted to this dtype
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.

        See Also
        --------
        from_block_func
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        # FIXME have removed add_metadata. that ok?

        if shape_kw is not None:
            def block_func(shape):
                arr = func(**{shape_kw: shape}, **func_kwargs)
                block = backend.block_from_numpy(arr, dtype)
                return block
        else:
            def block_func(shape):
                arr = func(shape, **func_kwargs)
                block = backend.block_from_numpy(arr, dtype)
                return block

        data = backend.from_block_func(block_func, legs, num_domain_legs)
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs,
                   labels=labels)

    @classmethod
    def random_normal(cls, legs: Space | list[Space] = None,
                      mean: BlockDiagonalTensor | None = None, sigma: float = 1.,
                      backend=None, labels: list[str | None] = None, dtype: Dtype = None,
                      num_domain_legs: int = 0) -> BlockDiagonalTensor:
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
        legs : (list of) Space
            If `mean` is given, this argument is ignored and legs are the same as those of `mean`.
            Otherwise, the legs of the result.
        mean : Tensor | None
            The mean of the distribution. By default (`mean=None`) the mean is zero.
        sigma : float
            The standard deviation of the distribution
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            If `mean` is given, this argument is ignored and the backend is the same as of `mean`.
            Otherwise, the backend for the tensor
        labels : list[str | None], optional
            If `mean` is given, this argument is ignored and labels are the same as those of `mean`.
            Otherwise, labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype
            The dtype for the tensor. If not given, use the dtype of `mean`. If `mean` is not given,
            default to `Dtype.float64`.
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.
            If `mean` is given, use its :attr:`Tensor.num_domain_legs` attribute instead.
        """
        if mean is not None:
            for name, val in zip(['legs', 'backend', 'labels'], [legs, backend, labels]):
                if val is not None:
                    warnings.warn(f'{name} argument to Tensor.random_normal was ignored, because mean was given.')

            if dtype is None:
                dtype = mean.dtype
            return mean + cls.random_normal(legs=mean.legs, mean=None, sigma=sigma, backend=mean.backend,
                                            labels=mean.labels, dtype=dtype, num_domain_legs=mean.num_domain_legs)

        if backend is None:
            backend = get_backend(legs[0].symmetry)
        if dtype is None:
            dtype = Dtype.float64
        # legs = [backend.add_leg_metadata(leg) for leg in legs]  # FIXME
        kwargs = dict(dtype=dtype, sigma=sigma)
        data = backend.from_block_func(backend.block_random_normal, legs, num_domain_legs, kwargs)
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs,
                   labels=labels)

    @classmethod
    def random_uniform(cls, legs: Space | list[Space], backend=None,
                       labels: list[str | None] = None, dtype: Dtype = Dtype.float64,
                       num_domain_legs: int = 0) -> BlockDiagonalTensor:
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
        legs : (list of) Space
            The legs of the result.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend for the tensor
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype
            The dtype for the tensor
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.
        """
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        # legs = [backend.add_leg_metadata(leg) for leg in legs]  # FIXME
        data = backend.from_block_func(backend.block_random_uniform, legs, num_domain_legs,
                                       func_kwargs=dict(dtype=dtype))
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs, labels=labels)

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
        if not self.symmetry.can_be_dropped:
            msg = f'Tensors with {self.symmetry} do not allow access to individual tensor entries'
            raise SymmetryError(msg)

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

    def to_flat_block_trivial_sector(self) -> Block:
        """Assumes self is a single-leg tensor and returns its components in the trivial sector.

        See Also
        --------
        from_flat_block_trivial_sector
        """
        assert self.num_legs == 1
        return self.backend.to_flat_block_trivial_sector(self)

    # --------------------------------------------
    # Overriding methods from Tensor
    # --------------------------------------------

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, legs: Space | list[Space],
             backend=None, labels: list[str | None] = None,
             dtype: Dtype = Dtype.float64, num_domain_legs: int = 0) -> BlockDiagonalTensor:
        """Empty Tensor with zero entries (not stored explicitly in most backends).

        Parameters
        ----------
        legs : (list of) Space
            The legs of the Tensor.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend for the Tensor.
        labels : list[str | None], optional
            Labels associated with each leg, ``None`` for unnamed legs.
        dtype : Dtype, optional
            The data type of the Tensor entries.
        num_domain_legs : int
            How many of the legs should be in the domain. See :ref:`tensors_as_maps`.

        """
        # legs = [backend.add_leg_metadata(leg) for leg in legs]    # FIXME
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        data = backend.zero_data(legs=legs, dtype=dtype, num_domain_legs=num_domain_legs)
        return cls(data=data, backend=backend, legs=legs, num_domain_legs=num_domain_legs, labels=labels)


    def add_trivial_leg(self, label: str = None, is_dual: bool = False, pos: int = -1,
                        to_domain: bool = None) -> BlockDiagonalTensor:
        res_num_legs = self.num_legs + 1
        if pos < 0:
            pos = pos + res_num_legs
        assert 0 <= pos < res_num_legs
        if pos < self.num_domain_legs:
            assert to_domain is not False
            to_domain = True
        if pos == self.num_domain_legs and to_domain is None:
            to_domain = False  # add to codomain by default
        if pos > self.num_domain_legs:
            assert to_domain is not True
            to_domain = False
        data = self.backend.add_trivial_leg(self, pos=pos, to_domain=to_domain)
        new_leg = ElementarySpace.from_trivial_sector(1, symmetry=self.symmetry, is_dual=is_dual)
        legs = self.legs[:pos] + [new_leg] + self.legs[pos:]
        labels = self.labels[:pos] + [label] + self.labels[pos:]
        res = BlockDiagonalTensor(
            data=data, legs=legs, num_domain_legs=self.num_domain_legs + int(to_domain),
            backend=self.backend, labels=labels
        )
        return res

    def apply_mask(self, mask: Mask, leg: int | str, new_label: str = KEEP_OLD_LABEL) -> BlockDiagonalTensor:
        leg_idx = self.get_leg_idx(leg)
        assert self.legs[leg_idx].is_equal_or_dual(mask.large_leg)
        projected_leg = mask.small_leg
        if self.legs[leg_idx].is_dual != projected_leg.is_dual:
            projected_leg = projected_leg.dual
        legs = self.legs[:]
        legs[leg_idx] = projected_leg
        labels = self.labels[:]
        if new_label is not KEEP_OLD_LABEL:
            labels[leg_idx] = new_label
        return BlockDiagonalTensor(
            data=self.backend.apply_mask_to_Tensor(self, mask, leg_idx),
            legs=legs, num_domain_legs=self.num_domain_legs, backend=self.backend, labels=labels
        )

    def as_Tensor(self) -> BlockDiagonalTensor:
        return self

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace] = None,
                     product_spaces_dual: list[bool] = None,
                     new_axes: list[int] = None,
                     new_labels: list[str | None] = None) -> BlockDiagonalTensor:
        """See :func:`tenpy.linalg.tensors.combine_legs`."""
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
        for n in reversed(range(len(product_spaces))):
            b, e = combine_slices[n]
            if new_labels is None:
                res_labels[b:e] = [_combine_leg_labels(res_labels[b:e])]
            else:
                res_labels[b:e] = [new_labels[n]]
            res_legs[b:e] = [product_spaces[n]]
        res_data = self.backend.combine_legs(res, combine_slices, product_spaces, new_axes, res_legs)
        # TODO num_domain_legs is a dummy!
        return BlockDiagonalTensor(res_data, backend=self.backend, legs=res_legs, 
                                   num_domain_legs=0, labels=res_labels)

    def conj(self) -> BlockDiagonalTensor:
        """See tensors.conj"""
        return BlockDiagonalTensor(
            self.backend.conj(self),
            legs=[l.dual for l in self.legs], num_domain_legs=self.num_domain_legs,
            backend=self.backend, labels=[_dual_leg_label(l) for l in self.shape._labels]
        )

    def copy(self, deep=True) -> BlockDiagonalTensor:
        if deep:
            return BlockDiagonalTensor(data=self.backend.copy_data(self.data),
                                       backend=self.backend,
                                       legs=self.legs[:],
                                       num_domain_legs=self.num_domain_legs,
                                       labels=self.labels[:])
        return BlockDiagonalTensor(data=self.data,
                                   backend=self.backend,
                                   legs=self.legs,
                                   num_domain_legs=self.num_domain_legs,
                                   labels=self.labels)

    def item(self) -> float | complex:
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')

    def norm(self, order=None) -> float:
        """See tensors.norm"""
        if order is None:
            order = 2
        return self.backend.norm(self, order=order)

    def permute_legs(self, permutation: list[int | str] = None, num_domain_legs: int = None
                     ) -> BlockDiagonalTensor:
        if permutation is None:
            legs = self.legs[:]
            labels = self.shape._labels[:]
        if permutation is not None:
            permutation = self.get_leg_idcs(permutation)
            assert len(permutation) == self.num_legs
            assert set(permutation) == set(range(self.num_legs))
            legs = [self.legs[n] for n in permutation]
            labels = [self.shape._labels[n] for n in permutation]
        res_data = self.backend.permute_legs(self, permutation, num_domain_legs)
        if num_domain_legs is None:
            num_domain_legs = self.num_domain_legs
        return BlockDiagonalTensor(res_data, legs=legs, num_domain_legs=num_domain_legs,
                                   backend=self.backend, labels=labels)

    def split_legs(self, *legs: int | str) -> BlockDiagonalTensor:
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
        num_domain_legs = self.num_domain_legs
        for i in leg_idcs:
            if i < self.num_domain_legs:
                # -1, since the old productspace was counted already
                num_domain_legs += len(old_legs[i].spaces) - 1
            new_legs.extend(old_legs[start:i])
            new_legs.extend(old_legs[i].spaces)
            new_labels.extend(old_labels[start:i])
            new_labels.extend(_split_leg_label(old_labels[i], num=len(old_legs[i].spaces)))
            start = i + 1
        new_legs.extend(old_legs[start:])
        new_labels.extend(old_labels[start:])
        res_data = self.backend.split_legs(self, leg_idcs, new_legs)
        return BlockDiagonalTensor(res_data, backend=self.backend, legs=new_legs,
                                   num_domain_legs=num_domain_legs, labels=new_labels)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> BlockDiagonalTensor:
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
        num_domain_legs = self.num_domain_legs - sum(squeezed < self.num_domain_legs for squeezed in leg_idcs)
        return BlockDiagonalTensor(res_data, backend=self.backend, legs=res_legs,
                                   num_domain_legs=num_domain_legs, labels=res_labels)

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        block = self.backend.to_dense_block(self)
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> BlockDiagonalTensor | float | complex:
        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = self.get_leg_idcs(legs2)
        if len(leg_idcs1) != len(leg_idcs2):
            raise ValueError('Must specify same number of legs')
        remaining_leg_idcs = [n for n in range(self.num_legs) if n not in leg_idcs1 and n not in leg_idcs2]
        if len(remaining_leg_idcs) == 0:
            return self.backend.trace_full(self, leg_idcs1, leg_idcs2)
        else:
            return BlockDiagonalTensor(
                self.backend.trace_partial(self, leg_idcs1, leg_idcs2, remaining_leg_idcs),
                legs=[self.legs[n] for n in remaining_leg_idcs],
                num_domain_legs=sum(n < self.num_domain_legs for n in remaining_leg_idcs),
                backend=self.backend,
                labels=[self.labels[n] for n in remaining_leg_idcs]
            )

    def _get_element(self, idcs: list[int]) -> float | complex:
        if not self.idcs_fulfill_charge_rule(idcs):
            return self.dtype.zero_scalar
        return self.backend.get_element(self, idcs)

    def _mul_scalar(self, other: complex) -> BlockDiagonalTensor:
        return BlockDiagonalTensor(
            self.backend.mul(other, self), legs=self.legs, num_domain_legs=self.num_domain_legs,
            backend=self.backend, labels=self.labels
        )

    def _set_element(self, idcs: list[int], value: float | complex) -> None:
        if not self.idcs_fulfill_charge_rule(idcs):
            msg = f'Can not set element at indices {idcs}. They do not fulfill the charge rule.'
            raise ValueError(msg)
        self.data = self.backend.set_element(self, idcs=idcs, value=value)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    def almost_equal(self, other: Tensor, atol: float = 1e-5, rtol: float = 1e-8,
                     allow_different_types: bool = False) -> bool:
        if not isinstance(other, BlockDiagonalTensor):
            if not allow_different_types:
                raise TypeError(f'Different types: {type(self)} and {type(other)}.')
            if isinstance(other, DiagonalTensor):
                other = other.as_Tensor()
            elif isinstance(other, ChargedTensor):
                try:
                    other = other.as_Tensor()
                except ValueError:
                    other, original_other = other.project_to_invariant(), other
                    if not other.almost_equal(original_other, atol, rtol):
                        return False
                # remains to check self.almost_equal(other, ...), which is done below
            else:
                raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')
        # can now assume that other is a Tensor
        other_order = self._input_checks_same_legs(other)
        if other_order is not None:
            other = other.permute_legs(other_order)
        return get_same_backend(self, other).almost_equal(self, other, atol=atol, rtol=rtol)

    def inner(self, other: Tensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, BlockDiagonalTensor):
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
        if isinstance(other, Mask):
            # use that leg_order_2 is either [0, 1] or [1, 0]
            # -> the leg to mask n is the one where leg_order_2[n] == 0, i.e. leg_order_2[0]
            return self.apply_mask(other, leg=leg_order_2[0]).trace()
        raise TypeError(f'inner not supported for {type(self)} and {type(other)}')

    def outer(self, other: Tensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> Tensor:
        if isinstance(other, DiagonalTensor):
            # OPTIMIZE this could be done more efficiently in the backend...
            other = other.as_Tensor()
        if isinstance(other, BlockDiagonalTensor):
            backend = get_same_backend(self, other)
            return BlockDiagonalTensor(
                backend.outer(self, other),
                legs=self.legs + other.legs,
                num_domain_legs=self.num_domain_legs + other.num_domain_legs,
                backend=backend,
                labels=_get_result_labels(self.labels, other.labels, relabel1, relabel2)
            )
        if isinstance(other, ChargedTensor):
            assert relabel2 is None or other.invariant_part.labels[-1] not in relabel2
            return ChargedTensor(
                invariant_part=self.outer(other.invariant_part, relabel1=relabel1, relabel2=relabel2),
                dummy_leg_state=other.dummy_leg_state
            )
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: Tensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> Tensor | float | complex:
        if isinstance(other, ChargedTensor):
            legs2 = other.get_leg_idcs(legs2)  # make sure we reference w.r.t. other, not other.invariant_part
            assert relabel2 is None or other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                       relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        leg_idcs1, leg_idcs2 = self._input_checks_tdot(other, legs1, legs2)
        open_legs1 = [leg for idx, leg in enumerate(self.legs) if idx not in leg_idcs1]
        open_legs2 = [leg for idx, leg in enumerate(other.legs) if idx not in leg_idcs2]
        open_labels1 = [leg for idx, leg in enumerate(self.labels) if idx not in leg_idcs1]
        open_labels2 = [leg for idx, leg in enumerate(other.labels) if idx not in leg_idcs2]
        # special case: outer()
        if len(leg_idcs1) == 0:
            return self.outer(other, relabel1, relabel2)
        backend = get_same_backend(self, other)
        if isinstance(other, Mask):
            if len(leg_idcs1) == 1:
                new_label = other.labels[1]
                if relabel2 is not None:
                    new_label = relabel2.get(new_label, new_label)
                res = self.apply_mask(other, leg_idcs1[0], new_label=new_label)
                res = res.permute_legs([n for n in range(self.num_legs) if n != leg_idcs1[0]] + leg_idcs1)
                if relabel2 is not None:
                    res.set_labels([relabel2.get(l, l) for l in res.labels[:-1]] + [res.labels[-1]])
                return res
            if len(leg_idcs1) == 2:
                # leg_idcs2 is [0, 1] or [1, 0]. we determine leg_idcs2[large_leg_idx] == 0
                large_leg_idx = leg_idcs2[0]
                res = self.apply_mask(other, leg_idcs1[large_leg_idx])
                res = res.trace(*leg_idcs1)
                if relabel2 is not None:
                    res.set_labels([relabel2.get(l, l) for l in res.labels])
                return res
            raise RuntimeError  # should have caught all other cases already
        if isinstance(other, DiagonalTensor):
            if len(leg_idcs1) == 2:
                # first contract leg that appears later in self.legs
                #  -> legs_idcs1[which_second] stays where it is
                which_first, which_second = (1, 0) if leg_idcs1[0] < leg_idcs1[1] else (0, 1)
                res = self.tdot(other, leg_idcs1[which_first], leg_idcs2[which_first],
                                relabel1=relabel1, relabel2=relabel2)
                return res.trace(leg_idcs1[which_second], -1)
            assert len(leg_idcs1) == 1 # have already excluded all other possibilities
            res = BlockDiagonalTensor(
                data=backend.scale_axis(self, other, leg=leg_idcs1[0]),
                legs=self.legs[:leg_idcs1[0]] + [other.legs[1 - leg_idcs2[0]]] + self.legs[leg_idcs1[0] + 1:],
                num_domain_legs=self.num_domain_legs,  # TODO this correct?
                backend=backend,
                labels=self.labels[:leg_idcs1[0]] + [other.labels[1 - leg_idcs2[0]]] + self.labels[leg_idcs1[0] + 1:]
            )
            # move scaled leg to the back
            perm = list(range(leg_idcs1[0])) + list(range(leg_idcs1[0] + 1, res.num_legs)) + [leg_idcs1[0]]
            return res.permute_legs(perm)
        if isinstance(other, BlockDiagonalTensor):
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
                num_domain_legs = \
                    self.num_domain_legs \
                    + other.num_domain_legs \
                    - sum(contr < self.num_domain_legs for contr in leg_idcs1) \
                    - sum(contr < other.num_domain_legs for contr in leg_idcs2)  # TODO this correct?
                return BlockDiagonalTensor(res_data, backend=backend, legs=res_legs,
                                           num_domain_legs=num_domain_legs, labels=res_labels)
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')

    def _add_tensor(self, other: Tensor) -> Tensor:
        if isinstance(other, DiagonalTensor):
            # OPTIMIZE ?
            other = other.as_Tensor()
        if isinstance(other, BlockDiagonalTensor):
            backend = get_same_backend(self, other)
            other_order = self._input_checks_same_legs(other)
            if other_order is not None:
                other = permute_legs(other, other_order)
            res_data = backend.add(self, other)
            return BlockDiagonalTensor(res_data, backend=backend, legs=self.legs,
                                       num_domain_legs=self.num_domain_legs, labels=self.labels)
        raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")

    # --------------------------------------------
    # Internal utility methods
    # --------------------------------------------

    def make_ProductSpace(self, legs: int | str | list[int | str], _is_dual=None) -> ProductSpace:
        # FIXME is_dual ???
        # FIXME is this really needed??
        return ProductSpace(self.get_legs(legs), backend=self.backend)

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
                # add metadata in-place if it is missing. if it already exists, this does nothing.
                self.backend.add_leg_metadata(product_spaces[i])
                # test for compatibility
                legs = [self.legs[a] for a in leg_idcs]
                if len(legs) != len(product_space.spaces):
                    raise ValueError("passed ProductSpace has wrong number of legs")
                # FIXME tmp rmed check
                # if legs[0].is_dual != product_space.spaces[0].is_dual:
                #     # TODO: should we just implicitly flip?
                #     raise ValueError("Wrong `is_dual` flag of ProductSpace")
                for self_leg, given_space in zip(legs, product_space.factors):
                    if self_leg != given_space:
                        msg = f'`product_spaces[{i:d}]` is incompatible with the legs to be combined'
                        raise ValueError(msg)
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


class ChargedTensor(Tensor):
    r"""Tensors which transform non-trivially under a symmetry.

    This class represents two related but distinct concepts, which functionally behave
    the same and thus share an implementation.
    The first is the concept of a "charged tensor", which only makes sense if the symmetry
    :attr:`~tenpy.linalg.symmetries.Symmetry.can_be_dropped`.
    The second concept is that of a "hidden leg", which can be later retrieved.

    The concept of a "charged tensor" arises in the context of group symmetries. A charged tensor,
    unlike :class:`SymmetricTensor`s, transforms non-trivially under the symmetry group.
    For concrete examples, consider on-site operators that change the charge sector of a state when
    applied, such as e.g. a boson creation operator if boson number is conserved or a spin raising
    operator if magnetization is conserved.
    We represent such charged tensors as a composite object consisting of an invariant
    :class:`SymmetricTensor` with a designated dummy leg and a state that this leg is to be
    contracted with. The interface is designed such that this implementation detail is hidden from
    the user and the `ChargedTensor` is a valid :class:`Tensor` (which does *not* have the dummy
    leg in its :attr:`legs`). The contraction with the :attr:`dummy_leg_state` (which is not
    symmetric and hence not a tensor!) is kept track of only symbolically, i.e.
    the :attr:`invariant_part` is kept separately until e.g. :meth:`item` is called.

    Contracting two `ChargedTensor`s via :meth:`tdot` can result in a `Tensor` if the dummy legs
    allow it. This would occur e.g. when forming the boson occupation number :math:`b^\dagger b`,
    which is a `SymmetricTensor` from `ChargedTensor`s.
    TODO revise if this should happen automatically, or if an explicit as_Tensor call should be made

    The second concept of a "hidden leg" also uses an invariant :class:`SymmetricTensor` with an
    additional "dummy" leg, but does not specify a state on that leg.
    Thus, it is a well-defined concept for all symmetries.
    In essence, it allows the dummy leg to be hidden from functions/algorithms and be retrieved
    later. This allows, for example, to evaluate correlation functions of more general operators,
    such as e.g. simulating :math:`\langle S_i^x(t) S_j^x(0) \rangle` with :math:`S^z` conservation.
    The :math:`S^x` operator, when using :math:`S^z` conservation, is a `ChargedTensor` with a
    two-dimensional dummy leg. But, for the correlation function, we do not actually need a state
    for that leg, we just need to contract it with the dummy leg of the other :math:`S^x`, after
    having time-evolved :math:`S_j^x(0) \ket{\psi_0}`.
    TODO revisit this paragraph, do we actually support doing that?

    Parameters
    ----------
    invariant_part:
        The symmetry-invariant part. the dummy leg is the last of its legs.
    dummy_leg_state: block | None
        Either ``None`` to signal that the dummy leg is just a hidden leg.
        Or a backend-specific block of shape ``(dummy_leg.dim,)``, defining a charged tensor
        which transforms non-trivially under the symmetry (see notes above); only allowed for
        group symmetries.

    Notes
    -----
    We always put the dummy leg as the last leg, and we always put it in the codomain.
    """
    _DUMMY_LABEL = '!'  # canonical label for the dummy leg

    def __init__(self, invariant_part: BlockDiagonalTensor, dummy_leg_state: Block | None):
        assert invariant_part.num_codomain_legs > 0  # dummy leg must be in codomain
        if dummy_leg_state is None:
            dtype = invariant_part.dtype
        else:
            assert len(dummy_leg_state) == invariant_part.legs[-1].dim
            if not invariant_part.symmetry.can_be_dropped:
                msg = f'ChargedTensor with dummy_leg_state is not well-defined for this symmetry.'
                raise SymmetryError(msg)
            dtype = Dtype.common(invariant_part.dtype,
                                 invariant_part.backend.block_dtype(dummy_leg_state))
        if invariant_part.labels[-1] is None:
            invariant_part = invariant_part.copy(deep=False)
            invariant_part.set_labels(invariant_part.labels[:-1] + [self._DUMMY_LABEL])
        else:
            assert invariant_part.labels[-1] == self._DUMMY_LABEL
        Tensor.__init__(self, backend=invariant_part.backend, legs=invariant_part.legs[:-1],
                        labels=invariant_part.labels[:-1], dtype=dtype,
                        num_domain_legs=invariant_part.num_domain_legs)
        self.invariant_part = invariant_part
        self.dummy_leg = invariant_part.legs[-1]
        self.dummy_leg_state = dummy_leg_state

    def test_sanity(self):
        super().test_sanity()
        self.invariant_part.test_sanity()
        if self.dummy_leg_state is not None:
            assert self.backend.block_shape(self.dummy_leg_state) == (self.dummy_leg.dim,)
        assert self.dtype != Dtype.bool

    # --------------------------------------------
    # Additional methods (not in Tensor)
    # --------------------------------------------

    @classmethod
    def from_block_func(cls, func, legs: Space | list[Space], charge: Space | Sector,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None, num_domain_legs: int = 0,
                        dummy_leg_state=None
                        ) -> ChargedTensor:
        assert 0 <= num_domain_legs <= len(legs)
        dummy_leg = cls._dummy_leg_from_charge(charge, symmetry=legs[0].symmetry)
        inv = BlockDiagonalTensor.from_block_func(func=func, legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], func_kwargs=func_kwargs,
                                     shape_kw=shape_kw, dtype=dtype, num_domain_legs=num_domain_legs)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=dummy_leg_state)

    @classmethod
    def from_dense_block(cls, block, legs: list[Space], backend=None, dtype: Dtype=None,
                         labels: list[str | None] = None, tol: float = 1e-8,
                         charge: Space | Sector = None, dummy_leg_state=None,
                         num_domain_legs: int = 0) -> ChargedTensor:
        """Convert a dense block of the backend to a ChargedTensor, if possible.

        TODO doc how and when it could fail

        Parameters
        ----------
        block :
            The data to be converted, a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`. This includes e.g.
            nested python iterables or numpy arrays.
        legs : list of :class:`~tenpy.linalg.spaces.Space`, optional
            The vectorspaces associated with legs of the tensors. Contains symmetry data.
            Does not contain the dummy leg.
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
            The backend for the ChargedTensor.
        dtype : Dtype, optional
            The data type for the ChargedTensor. By default, this is inferred from the block.
        labels : list of {str | None}, optional
            Labels associated with each leg, ``None`` for unnamed legs.
            Does not contain a label for the dummy leg.
        tol : float
            Tolerance for checking if a block is symmetric. We check if
            ``norm(block - symmetric_components) < tol * norm(block)``.
        charge : Space or Sector, optional
            The charge, specified either via the dummy leg of the :class:`ChargedTensor` or
            via the sector that the tensor should live in.
            As such, a sector is equivalent to `ElementarySpace(symmetry, sectors=[charge]).dual`.
            Note the `.dual`! The charge-rule for the invariant part then forces the composite
            ChargedTensor to be in the specified sector.
            If not given, it is inferred from the largest (by magnitude) entry of the block.
        dummy_leg_state : block
            The state on the dummy leg. May be ``None`` ("unspecified").
        num_domain_legs : int
            Only meaningful for FusionTree backend.
        """
        assert 0 <= num_domain_legs <= len(legs)
        if backend is None:
            backend = get_backend(legs[0].symmetry)
        if labels is None:
            labels = [None] * len(legs)
        block = backend.as_block(block)
        if dtype is not None:
            block = backend.block_to_dtype(block, dtype)
        # add 1-dim axis for the dummy leg
        block = backend.block_add_axis(block, -1)
        if charge is None:
            dummy_leg = backend.infer_leg(block, legs + [None])
        else:
            dummy_leg = cls._dummy_leg_from_charge(charge, symmetry=legs[0].symmetry)
        invariant_part = BlockDiagonalTensor.from_dense_block(
            block, legs=legs + [dummy_leg], backend=backend, dtype=dtype,
            labels=labels + [cls._DUMMY_LABEL], tol=tol, num_domain_legs=num_domain_legs
        )
        return cls(invariant_part, dummy_leg_state=dummy_leg_state)

    @classmethod
    def from_flat_block_single_sector(cls, leg: Space, block: Block, sector: Sector,
                                      backend: Backend, label: str = None) -> ChargedTensor:
        """Create a single-leg `ChargedTensor` from the *part of* the coefficients in the given sector.

        The resulting dummy leg will have the (dual of the) given sector with multiplicity 1 and
        be in the codomain.

        Parameters
        ----------
        leg : :class:`~tenpy.linalg.spaces.Space`
            The single leg of the resulting tensor.
        block : backend-specific Block
            The block of shape ``(D * M,)`` where ``M`` is the multiplicity of the given `sector`
            in `leg` and ``D`` is its dimension.
            This is a slice of ``result.to_dense_block()``.
        sector : Sector
            The charge of the resulting ChargedTensor, i.e. the sector it lives in
        backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
            The backend of the resulting tensor
        label : str | None
            The label for the single leg

        See Also
        --------
        to_flat_block_single_sector
        """
        if leg.symmetry.sector_dim(sector) > 1:
            # TODO how to handle multi-dim sectors? which dummy leg state to give?
            raise NotImplementedError
        dummy_leg = cls._dummy_leg_from_charge(sector, symmetry=leg.symmetry)
        inv_part = BlockDiagonalTensor(
            data=backend.inv_part_from_flat_block_single_sector(block=block, leg=leg, dummy_leg=dummy_leg),
            legs=[leg, dummy_leg], num_domain_legs=0, backend=backend, labels=[label, cls._DUMMY_LABEL]
        )
        return cls(inv_part, dummy_leg_state=inv_part.backend.as_block([1]))

    @classmethod
    def from_numpy_func(cls, func, legs: Space | list[Space], charge: Space | Sector,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None, num_domain_legs: int = 0
                        ) -> ChargedTensor:
        assert 0 <= num_domain_legs <= len(legs)
        legs = to_iterable(legs)
        dummy_leg = cls._dummy_leg_from_charge(charge, symmetry=legs[0].symmetry)
        inv = BlockDiagonalTensor.from_numpy_func(func=func, legs=legs + [dummy_leg], backend=backend,
                                     labels=labels + [cls._DUMMY_LABEL], func_kwargs=func_kwargs,
                                     shape_kw=shape_kw, dtype=dtype, num_domain_legs=num_domain_legs)
        shape = (dummy_leg.dim,)
        if shape_kw is not None:
            arr = func(**{shape_kw: shape}, **func_kwargs)
        else:
            arr = func(shape, **func_kwargs)
        block = inv.backend.block_from_numpy(arr, dtype)
        # TODO allow to specify dummy_leg_state?
        return ChargedTensor(invariant_part=inv, dummy_leg_state=block)

    @classmethod
    def from_tensor(cls, tens: BlockDiagonalTensor) -> ChargedTensor:
        return ChargedTensor(
            invariant_part=add_trivial_leg(tens, pos=-1),
            dummy_leg_state=tens.backend.ones_block((1,), dtype=tens.dtype)
        )

    @classmethod
    def from_two_dummy_legs(cls, invariant_part: BlockDiagonalTensor, leg1: int, state1: Block | None, leg2: int,
                            state2: Block | None, convert_to_tensor_if_possible: bool = False
                            ) -> ChargedTensor | BlockDiagonalTensor:
        leg1 = invariant_part.get_leg_idx(leg1)
        leg2 = invariant_part.get_leg_idx(leg2)
        first = min(leg1, leg2)
        second = max(leg1, leg2)
        other_legs = list(range(first)) \
                     + list(range(first + 1, second)) \
                     + list(range(second + 1, invariant_part.num_legs))
        invariant_part = invariant_part.permute_legs(other_legs + [leg1, leg2])  # TODO make sure they end up in the codomain
        invariant_part = invariant_part.combine_legs(-2, -1)
        product_space: ProductSpace = invariant_part.legs[-1]
        if state1 is not None and state2 is not None:
            state = product_space.fuse_states([state1, state2], backend=invariant_part.backend)
        elif state1 is None and state2 is None:
            state = None
        elif state1 is None and leg1.dim == 1:  # state1 ~= [1.]
            state = state2
        elif state2 is None and leg2.dim == 1:  # state2 ~= [1.]
            state = state1
        else:
            raise ValueError('Can not fuse a specified state with an unspecified state')
        res = ChargedTensor(invariant_part=invariant_part, dummy_leg_state=state)
        if convert_to_tensor_if_possible:
            try:
                res = res.as_Tensor()
            except ValueError:
                pass  # if its not possible to convert to Tensor, just leave it as ChargedTensor
        return res
    
    @classmethod
    def random_uniform(cls, legs: Space | list[Space], charge: Space | Sector,
                       backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.float64,
                       dummy_leg_state=None, num_domain_legs: int = 0) -> ChargedTensor:
        legs = to_iterable(legs)
        dummy_leg = cls._dummy_leg_from_charge(charge, symmetry=legs[0].symmetry)
        if labels is None:
            labels = [None] * len(legs)
        inv = BlockDiagonalTensor.random_uniform(
            legs=legs + [dummy_leg], backend=backend, labels=labels + [cls._DUMMY_LABEL],
            dtype=dtype, num_domain_legs=num_domain_legs
        )
        if dummy_leg_state is not None:
            dummy_leg_state = inv.backend.as_block(dummy_leg_state)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=dummy_leg_state)

    @classmethod
    def random_normal(cls) -> ChargedTensor:
        raise NotImplementedError  # TODO

    def flip_dummy_leg_duality(self) -> ChargedTensor:
        """Like :func:`flip_leg_duality` but for the dummy leg"""
        # TODO does this conj the dummy_leg_state ?
        inv = self.invariant_part.flip_leg_duality(-1).relabel({'!*': '!'})
        return ChargedTensor(invariant_part=inv, dummy_leg_state=self.dummy_leg_state)

    def project_to_invariant(self) -> BlockDiagonalTensor:
        """Project self into the invariant subspace of the parent space.

        These are the contributions from trivial sectors in the dummy leg.

        See Also
        --------
        :meth:`_project_to_invariant`
        """
        res = self._project_to_invariant()
        if res is None:
            res = BlockDiagonalTensor.zero(legs=self.legs, backend=self.backend, labels=self.labels, dtype=self.dtype)
        return res

    def _project_to_invariant(self) -> BlockDiagonalTensor | None:
        """Internal version of :meth:`project_to_invariant`.

        If the dummy leg does not contain the trivial sector of the symmetry,
        the result is known to vanish exactly and ``None`` is returned.
        The "public" version :meth:`project_to_invariant` (no underscore!) instead returns a
        zero `Tensor` in this case.
        """
        raise NotImplementedError  # TODO

    def as_Tensor(self) -> BlockDiagonalTensor:
        """If possible, convert self to a Tensor. Otherwise raise a SymmetryError.

        It is possible to convert a ChargedTensor to a Tensor if and only if the dummy leg only
        contains the trivial sector.

        See Also
        --------
        project_to_invariant
        """
        if not np.all(self.dummy_leg.sectors[:] == self.symmetry.trivial_sector[None, :]):
            raise ValueError('ChargedTensor with non-trivial charge could not be converted to Tensor.')
        if self.dummy_leg_state is None:
            msg = 'Can not convert to Tensor. dummy_leg_state is unspecified.'
            raise SymmetryError(msg)
        if self.dummy_leg.dim == 1:
            factor = self.backend.block_item(self.dummy_leg_state)
            return factor * self.invariant_part.squeeze_legs(-1)
        state = BlockDiagonalTensor.from_dense_block(self.dummy_leg_state, legs=[self.dummy_leg], backend=self.backend)
        return self.invariant_part.tdot(state, -1, 0)

    def to_flat_block_single_sector(self) -> Block:
        """Assumes a single-leg tensor living in a single sector and returns its components within
        that sector.

        See Also
        --------
        from_flat_block_single_sector
        """
        if self.dummy_leg_state is None:
            raise ValueError('Unspecified dummy_leg_state')
        if self.num_legs > 1:
            raise ValueError('Expected a single leg')
        if self.dummy_leg.num_sectors != 1 or self.dummy_leg.multiplicities[0] != 1:
            raise ValueError('Not a single sector')
        if self.symmetry.sector_dim(self.dummy_leg.sectors[0]) > 1:
            # TODO how to handle multi-dim sectors? should cooperate with from_flat_block_single_sector
            raise NotImplementedError
        block = self.backend.inv_part_to_flat_block_single_sector(self.invariant_part)
        return self.backend.block_item(self.dummy_leg_state) * block

    # --------------------------------------------
    # Overriding methods from Tensor
    # --------------------------------------------

    def relabel(self, mapping: dict[str, str | None], inplace=True) -> Tensor:
        assert self._DUMMY_LABEL not in mapping
        if inplace:
            self.shape.relabel(mapping)
            self.invariant_part.relabel(mapping, inplace=True)
            return self
        return ChargedTensor(invariant_part=self.invariant_part.relabel(mapping, inplace=False),
                             dummy_leg_state=self.dummy_leg_state)

    def _data_repr_lines(self, indent: str, max_lines: int) -> list[str]:
        return self.backend._data_repr_lines(
            self.invariant_part, indent=indent, max_width=printoptions.linewidth, max_lines=max_lines
        )
    
    def _repr_header_lines(self, indent: str) -> list[str]:
        lines = Tensor._repr_header_lines(self, indent=indent)
        lines.append(f'{indent}* Dummy Leg: dim={self.dummy_leg.dim}, '
                     f'dual={self.dummy_leg.is_dual}, sectors={self.dummy_leg.sectors}')
        lines.append(f'{indent}* Dummy Leg state:')
        if self.dummy_leg_state is None:
            lines.append(f'{indent}  unspecified')
        else:
            lines.extend(self.backend._block_repr_lines(self.dummy_leg_state, indent=indent + '  '),
                        max_width=70, max_lines=3)
        return lines

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, legs: Space | list[Space], dummy_leg: Space,
             backend=None, labels: list[str | None] = None, dtype: Dtype = Dtype.float64,
             dummy_leg_state=None, num_domain_legs: int = 0) -> ChargedTensor:
        if isinstance(legs, Space):
            legs = [legs]
        if labels is None:
            labels = [None] * len(legs)
        invariant_part = BlockDiagonalTensor.zero(
            legs=legs + [dummy_leg], backend=backend, labels=labels + [cls._DUMMY_LABEL],
            dtype=dtype, num_domain_legs=num_domain_legs
        )
        return cls(invariant_part=invariant_part, dummy_leg_state=dummy_leg_state)

    def add_trivial_leg(self, label: str = None, is_dual: bool = False, pos: int = -1,
                        to_domain: bool = None) -> ChargedTensor:
        # parse pos now, since it is w.r.t. ``self.num_legs``, not ``self.invariant_part.num_legs``.
        res_num_legs = self.num_legs + 1
        if not (-res_num_legs <= pos < res_num_legs):
            raise ValueError(f'`pos` out of bounds: {pos}')
        if pos < 0:
            pos = pos + res_num_legs
        inv = self.invariant_part.add_trivial_leg(label=label, is_dual=is_dual, pos=pos,
                                                  to_domain=to_domain)
        return ChargedTensor(inv, self.dummy_leg_state)

    def apply_mask(self, mask: Mask, leg: int | str) -> ChargedTensor:
        return ChargedTensor(
            invariant_part=self.invariant_part.apply_mask(mask, self.get_leg_idx(leg)),
            dummy_leg_state=self.dummy_leg_state
        )

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace] = None,
                     product_spaces_dual: list[bool] = None,
                     new_axes: list[int] = None,
                     new_labels: list[str | None] = None) -> ChargedTensor:
        """See :func:`tenpy.linalg.tensors.combine_legs`."""
        legs = [self.get_leg_idcs(group) for group in legs]  # needed, since invariant_part does not have the same legs
        inv = self.invariant_part.combine_legs(*legs, product_spaces=product_spaces,
                                               product_spaces_dual=product_spaces_dual,
                                               new_axes=new_axes, new_labels=new_labels)
        return ChargedTensor(invariant_part=inv, dummy_leg_state=self.dummy_leg_state)

    def conj(self) -> ChargedTensor:
        if self.dummy_leg_state is None:
            # TODO think about this case carefully. We could just also do dummy_leg_state = None
            # i.e. the state is still unspecified. Could it cause problems that we then did not
            # keep track of the conj? Probably not, if the state is unspecified, you cant expect
            # tenpy to keep track of what happens to it...
            raise NotImplementedError
        else:
            dummy_leg_state = self.backend.block_conj(self.dummy_leg_state)
        inv = self.invariant_part.conj().relabel({'!*': '!'})
        return ChargedTensor(invariant_part=inv, dummy_leg_state=dummy_leg_state)

    def hconj(self) -> ChargedTensor:
        # TODO is this expected behavior? i.e. that conj() has opposite dummy_leg.is_dual
        #      but hconj gives the same?
        return super().hconj().flip_dummy_leg_duality()

    def copy(self, deep=True) -> ChargedTensor:
        if deep:
            if self.dummy_leg_state is None:
                dummy_leg_state = None
            else:
                dummy_leg_state = self.backend.block_copy(self.dummy_leg_state)
            return ChargedTensor(invariant_part=self.invariant_part.copy(deep=True),
                                 dummy_leg_state=dummy_leg_state)
        return ChargedTensor(invariant_part=self.invariant_part.copy(deep=False),
                             dummy_leg_state=self.dummy_leg_state)

    def item(self) -> float | complex:
        if not all(leg.dim == 1 for leg in self.invariant_part.legs[:-1]):
            raise ValueError('Not a scalar')
        return self.backend.block_item(self.to_dense_block())

    def norm(self, order=None) -> float:
        if order is None:
            order = 2
        if self.dummy_leg_state is None:
            raise ValueError('dummy_leg_state unspecified.')
        if self.dummy_leg.dim == 1:
            factor = self.backend.block_item(self.dummy_leg_state)
            return factor * self.invariant_part.norm(order=order)
        else:
            warnings.warn('Converting ChargedTensor to dense block for `norm`', stacklevel=2)
            return self.backend.block_norm(self.to_dense_block(), order=order)

    def permute_legs(self, permutation: list[int | str] = None, num_domain_legs: int = None
                     ) -> ChargedTensor:
        permutation = self.get_leg_idcs(permutation)  # needed, since invariant_part does not have the same legs
        permutation = permutation + [-1]  # keep dummy leg at its position
        inv_part = self.invariant_part.permute_legs(permutation, num_domain_legs)
        return ChargedTensor(inv_part, self.dummy_leg_state)

    def split_legs(self, *legs: int | str) -> ChargedTensor:
        legs = self.get_leg_idcs(legs)  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.split_legs(*legs),
                             dummy_leg_state=self.dummy_leg_state)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> ChargedTensor:
        legs = self.get_leg_idcs(legs)  # needed, since invariant_part does not have the same legs
        return ChargedTensor(invariant_part=self.invariant_part.squeeze_legs(legs),
                             dummy_leg_state=self.dummy_leg_state)

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        invariant_block = self.backend.to_dense_block(self.invariant_part)
        if self.dummy_leg_state is None:
            raise ValueError('dummy_leg_state not specified')
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
        if self.dummy_leg_state is None:
            raise ValueError('dummy_leg_state unspecified.')
        if self.dummy_leg.dim == 1:
            factor = self.backend.block_item(self.dummy_leg_state)
            return factor * self.invariant_part._get_element(idcs + [0])
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
        if self.dummy_leg_state is None:
            raise ValueError('dummy_leg_state unspecified.')
        if self.dummy_leg.dim > 1:
            msg = 'Can not set elements of ChargedTensor with non-trivial dummy leg'
            raise ValueError(msg)
        factor = self.backend.block_item(self.dummy_leg_state)
        self.invariant_part._set_element(idcs + [0], value / factor)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    def almost_equal(self, other: Tensor, atol: float = 0.00001, rtol: float = 1e-8,
                     allow_different_types: bool = False) -> bool:
        if not isinstance(other, ChargedTensor):
            if not allow_different_types:
                raise TypeError(f'Different types: {type(self)} and {type(other)}.')
            if isinstance(other, DiagonalTensor):
                other = other.to_full()
            if isinstance(other, BlockDiagonalTensor):
                try:
                    s = self.as_Tensor()
                except ValueError:
                    s_proj = self.project_to_invariant()
                    return self.almost_equal(s_proj, atol, rtol) and s_proj.almost_equal(other, atol, rtol)
                return s.almost_equal(other, atol, rtol)
            else:
                raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')

        other_order = self._input_checks_same_legs(other)
        if other_order is not None:
            other = other.permute_legs(other_order)

        # can now assume that isinstance(other, ChargedTensor)
        if self.legs != other.legs:
            raise ValueError('Mismatching shapes')
        if self.dummy_leg.is_dual != other.dummy_leg.is_dual:
            other = other.flip_dummy_leg_duality()
        if self.dummy_leg != other.dummy_leg:
            return False
        if (self.dummy_leg_state is None) != (other.dummy_leg_state is None):
            raise ValueError('Mismatch: defined and undefined dummy_leg_state')
        if self.dummy_leg_state is None:
            return self.invariant_part.almost_equal(other.invariant_part, atol=atol, rtol=rtol)
        backend = get_same_backend(self, other)
        if self.dummy_leg.dim == 1:
            factor = backend.block_item(self.dummy_leg_state) / backend.block_item(other.dummy_leg_state)
            return self.invariant_part.almost_equal(factor * other.invariant_part, atol=atol, rtol=rtol)
        else:
            # The decomposition into invariant part and non-invariant state is not unique,
            # so we cant just compare them individually.
            # OPTIMIZE (JU) is there a more efficient way?
            warnings.warn('Converting ChargedTensor to dense block for `almost_equal`')
            other_order = self._input_checks_same_legs(other)
            backend = get_same_backend(self, other)
            self_block = self.to_dense_block()
            other_block = other.to_dense_block(leg_order=other_order)
            return backend.block_allclose(self_block, other_block)

    def inner(self, other: Tensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        if self.dummy_leg_state is None:
            raise ValueError('dummy_leg_state unspecified')
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, (DiagonalTensor, BlockDiagonalTensor)):
            # other is not charged and thus lives in the trivial sector of the parent space.
            # thus, only the components of self in the trivial sector contribute to the overlap
            self_projected = self._project_to_invariant()
            if self_projected is None:
                return Dtype.common(self.dtype, other.dtype).zero_scalar
            return self_projected.inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, ChargedTensor):
            if other.dummy_leg_state is None:
                raise ValueError('dummy_leg_state unspecified')
            # OPTIMIZE could directly return 0 if the two tensors have completely different charges
            backend = get_same_backend(self, other)
            # contract the invariant parts with each other and convert do dense block
            inv1 = self.invariant_part.conj() if do_conj else self.invariant_part
            if leg_order_2 is None:
                leg_order_2 = list(range(other.num_legs))
            res = inv1.tdot(other.invariant_part, legs1=list(range(self.num_legs)), legs2=leg_order_2)
            res = res.to_dense_block()
            # contract with state on dummy leg of self
            state = backend.block_conj(self.dummy_leg_state) if do_conj else self.dummy_leg_state
            res = backend.block_tdot(state, res, 0, 0)
            # contract with state on dummy leg of other
            res = backend.block_tdot(res, other.dummy_leg_state, 0, 0)
            return backend.block_item(res)
        if isinstance(other, Mask):
            # use that leg_order_2 is either [0, 1] or [1, 0]
            # -> the leg n we need to mask is the one where leg_order_2[n] == 0, i.e. n == leg_order_2[0]
            return self.apply_mask(other, leg=leg_order_2[0]).trace()
        raise TypeError(f'inner not supported for {type(self)} and {type(other)}')
    
    def outer(self, other: Tensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> Tensor:
        assert relabel1 is None or self.invariant_part.labels[-1] not in relabel1
        if isinstance(other, DiagonalTensor):
            other = other.as_Tensor()
        if isinstance(other, BlockDiagonalTensor):
            inv_part = self.invariant_part.outer(other, relabel1=relabel1, relabel2=relabel2)
            # permute dummy leg to the back
            self_normal = list(range(self.num_legs))
            self_dummy = [self.num_legs]
            other_normal = list(range(self.num_legs + 1, self.num_legs + 1 + other.num_legs))
            inv_part = inv_part.permute_legs(self_normal + other_normal + self_dummy)
            return ChargedTensor(invariant_part=inv_part, dummy_leg_state=self.dummy_leg_state)
        if isinstance(other, ChargedTensor):
            assert relabel2 is None or other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.invariant_part.outer(other.invariant_part, relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor.from_two_dummy_legs(
                invariant_part, leg1=self.num_legs, state1=self.dummy_leg_state, leg2=-1,
                state2=other.dummy_leg_state, convert_to_tensor_if_possible=True
            )
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: Tensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> Tensor | float | complex:
        # no need to do input checks, since we reduce to Tensor.tdot, which checks
        legs1 = self.get_leg_idcs(legs1)  # make sure we reference w.r.t. self, not self.invariant_part
        assert relabel1 is None or self.invariant_part.labels[-1] not in relabel1
        if isinstance(other, (BlockDiagonalTensor, DiagonalTensor, Mask)):
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
            legs2 = other.get_leg_idcs(legs2)  # make sure we reference w.r.t. other
            assert relabel2 is None or other.invariant_part.labels[-1] not in relabel2
            invariant = self.invariant_part.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                                 relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor.from_two_dummy_legs(
                invariant, leg1=self.num_legs - len(legs1), state1=self.dummy_leg_state,
                leg2=-1, state2=other.dummy_leg_state, convert_to_tensor_if_possible=True
            )
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')

    def _add_tensor(self, other: Tensor) -> ChargedTensor:
        if not isinstance(other, ChargedTensor):
            raise TypeError(f"unsupported operand type(s) for +: 'ChargedTensor' and '{type(other)}'")
        if self.dummy_leg != other.dummy_leg:
            raise ValueError('Can not add ChargedTensors with different dummy legs')
        if (self.dummy_leg_state is None) != (other.dummy_leg_state is None):
            raise ValueError('Can not add ChargedTensors with specified and unspecified dummy_leg_state')
        if self.dummy_leg_state is None:
            return ChargedTensor(self.invariant_part + other.invariant_part, dummy_leg_state=None)
        if self.dummy_leg.dim == 1:
            factor = self.backend.block_item(self.dummy_leg_state) / self.backend.block_item(other.dummy_leg_state)
            return ChargedTensor(self.invariant_part + factor * other.invariant_part,
                                 dummy_leg_state=self.dummy_leg_state)
        raise NotImplementedError  # TODO can we cover the remaining case somehow? might be impossible.

    # --------------------------------------------
    # Internal utility methods
    # --------------------------------------------

    @staticmethod
    def _dummy_leg_from_charge(charge: Space | Sector, symmetry: Symmetry):
        if isinstance(charge, Space):
            return charge.as_ElementarySpace()
        assert symmetry.is_valid_sector(charge)
        return ElementarySpace(symmetry, sectors=[charge], multiplicities=[1]).dual


class DiagonalTensor(SymmetricTensor):
    r"""Special case of a tensor with two legs that is diagonal in the computational basis.

    TODO more "elementwise" methods (exp, log, sqrt, ...?)

    Parameters
    ----------
    data
        The numerical data ("free parameters") comprising the tensor. type is backend-specific
    first_leg : Space
        The first leg of this tensor.
    second_leg_dual : bool
        Wether the second leg is the dual of the first (default) or the same.
        If ``True``, the result is a tensor :math:`\sum_n c_n \vert n \rangle\langle n \vert`.
        Otherwise it is :math:`\sum_n c_n \vert n \rangle \otimes \vert n \rangle`.
    backend: :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    def __init__(self, data, first_leg: Space, second_leg_dual: bool = True, backend=None,
                 labels: list[str | None] = None):
        self.data = data
        self.second_leg_dual = second_leg_dual
        second_leg = first_leg.dual if second_leg_dual else first_leg
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        dtype = backend.get_dtype_from_data(data)
        Tensor.__init__(self, legs=[first_leg, second_leg], backend=backend, labels=labels,
                        dtype=dtype, num_domain_legs=1)

    def test_sanity(self) -> None:
        super().test_sanity()
        self.backend.test_data_sanity(self, is_diagonal=True)
        assert self.dtype != Dtype.bool

    # --------------------------------------------
    # Additional methods (not in Tensor)
    # --------------------------------------------
    
    @functools.cached_property
    def diag_block(self) -> Block:
        return self.backend.diagonal_to_block(self)

    @functools.cached_property
    def diag_numpy(self) -> np.ndarray:
        block = self.diag_block
        return self.backend.block_to_numpy(block)

    @classmethod
    def eye(cls, first_leg: Space, backend=None, labels: list[str | None] = None,
            dtype: Dtype = Dtype.float64) -> DiagonalTensor:
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        if len(labels) == 1:
            labels = [labels[0], _dual_leg_label(labels[0])]
        assert len(labels) == 2
        return cls.from_block_func(
            func=backend.ones_block,
            first_leg=first_leg, second_leg_dual=True, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype),
        )

    @classmethod
    def from_block_func(cls, func, first_leg: Space, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> DiagonalTensor:
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        if shape_kw is not None:
            def block_func(shape):
                block = func(**{shape_kw: shape}, **func_kwargs)
                if dtype is not None:
                    block = backend.block_to_dtype(block, dtype)
                return block
        else:
            def block_func(shape):
                block = func(shape, **func_kwargs)
                if dtype is not None:
                    block = backend.block_to_dtype(block, dtype)
                return block

        data = backend.diagonal_from_block_func(block_func, leg=first_leg)
        res = cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_diag(cls, diag: Block, first_leg: Space, second_leg_dual: bool = True,
                  backend=None, labels: list[str | None] = None) -> DiagonalTensor:
        """Convert a dense 1D block containing the diagonal entries to a DiagonalTensor.

        Parameters
        ----------
        diag : Block-like
            The diagonal entries as a backend-specific block or some data that can be converted
            using :meth:`AbstractBlockBackend.as_block`. This includes e.g. nested python iterables
            or numpy arrays.
        first_leg, second_leg_dual, backend, labels
            Same as in constructor of :class:`DiagonalTensor`.
        """
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        diag = backend.as_block(diag)
        assert backend.block_shape(diag) == (first_leg.dim,)
        data = backend.diagonal_from_block(diag, leg=first_leg)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def from_numpy_func(cls, func, first_leg: Space, second_leg_dual: bool = True,
                        backend=None, labels: list[str | None] = None, func_kwargs={},
                        shape_kw: str = None, dtype: Dtype = None) -> DiagonalTensor:
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        if shape_kw is not None:
            def block_func(shape):
                arr = func(**{shape_kw: shape}, **func_kwargs)
                block = backend.block_from_numpy(arr, dtype)
                return block

        else:
            def block_func(shape):
                arr = func(shape, **func_kwargs)
                block = backend.block_from_numpy(arr, dtype)
                return block

        data = backend.diagonal_from_block_func(block_func, leg=first_leg)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def from_tensor(cls, tens: BlockDiagonalTensor, check_offdiagonal: bool = True) -> DiagonalTensor:
        """Create DiagonalTensor from a Tensor.

        Parameters
        ----------
        tens : :class:`Tensor`
            Must have two legs. Its diagonal entries ``tens[i, i]`` are used.
        check_offdiagonal : bool
            If the off-diagonal entries of `tens` should be checked.
        
        Raises
        ------
        ValueError
            If `check_offdiagonal` and any off-diagonal element is non-zero.
            TODO should there be a tolerance?
        """
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
    def random_normal(cls, first_leg: Space = None, second_leg_dual: bool = None,
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
            backend = get_backend(first_leg.symmetry)
        if dtype is None:
            dtype = Dtype.float64
        data = backend.diagonal_from_block_func(backend.block_random_normal, leg=first_leg,
                                                func_kwargs=dict(dtype=dtype))
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    @classmethod
    def random_uniform(cls, first_leg: Space, second_leg_dual: bool = True, backend=None,
                       labels: list[str | None] = None, dtype: Dtype = Dtype.float64
                       ) -> DiagonalTensor:
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        data = backend.diagonal_from_block_func(backend.block_random_uniform, leg=first_leg,
                                                func_kwargs=dict(dtype=dtype))
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    def _apply_mask_both_legs(self, mask: Mask) -> DiagonalTensor:
        """Apply the same mask to both legs."""
        assert self.legs[0].is_equal_or_dual(mask.large_leg)
        res_leg = mask.small_leg
        if self.legs[0].is_dual != res_leg.is_dual:
            res_leg = res_leg.dual
        return DiagonalTensor(
            data=self.backend.apply_mask_to_DiagonalTensor(self, mask),
            first_leg=res_leg,
            second_leg_dual=self.second_leg_dual, backend=self.backend, labels=self.labels,
        )

    def _binary_operand(self, other: Number | DiagonalTensor, func, operand: str,
                        is_bool_valued: bool, return_NotImplemented: bool = True
                        ) -> DiagonalTensor | Mask:
        """Utility function for a shared implementation of binary functions, whose second argument
        may be a scalar ("to be broadcast") or a DiagonalTensor,
        e.g. the dunder methods, ``__mul__, __eq__, __lt__, __pow__, ...``.

        Parameters
        ----------
        other
            Either a number or a DiagonalTensor.
        func
            The function with signature
            ``func(self_block: Block, other_or_other_block: Number | Block) -> Block``
        operand
            A string representation of the operand, used in error messages
        is_bool_valued
            Whether the output is boolean-valued.
            If so, a `Mask` is returned, otherwise a `DiagonalTensor`.
        return_NotImplemented
            Whether `NotImplemented` should be returned on a non-scalar and non-`Tensor` other.
        """
        if isinstance(other, Number):
            backend = self.backend
            data = backend.diagonal_elementwise_unary(
                self, func=lambda block: func(block, other), func_kwargs={}, maps_zero_to_zero=False
            )
            labels = self.labels
        elif isinstance(other, DiagonalTensor):
            backend = get_same_backend(self, other)
            if self.legs[0] != other.legs[0] or self.second_leg_dual != other.second_leg_dual:
                raise ValueError('Incompatible legs!')
            data = backend.diagonal_elementwise_binary(self, other, func=func)
            labels = _get_same_labels(self.labels, other.labels)
            
        elif return_NotImplemented and not isinstance(other, Tensor):
            return NotImplemented
        else:
            msg = f'Invalid types for operand "{operand}": {type(self)} and {type(other)}'
            raise TypeError(msg)

        res = DiagonalTensor(data, first_leg=self.legs[0], second_leg_dual=self.second_leg_dual,
                             backend=backend, labels=labels)
        if is_bool_valued:
            res = Mask.from_DiagonalTensor(res)
        return res

    def _elementwise_unary(self, func, func_kwargs={}, maps_zero_to_zero: bool = False) -> DiagonalTensor:
        """Wrap backend.diagonal_elementwise_unary

        func(a: Block, **kwargs) -> Block"""
        data = self.backend.diagonal_elementwise_unary(
            self, func, func_kwargs=func_kwargs, maps_zero_to_zero=maps_zero_to_zero
        )
        return DiagonalTensor(data=data, first_leg=self.legs[0], second_leg_dual=self.second_leg_dual,
                              backend=self.backend, labels=self.labels)

    def _elementwise_binary(self, other: DiagonalTensor, func, func_kwargs={},
                            partial_zero_is_zero: bool = False) -> DiagonalTensor:
        """Wrap backend.diagonal_elementwise_binary

        func(a: Block, b: Block, **kwargs) -> Block"""
        assert isinstance(other, DiagonalTensor)
        if self.legs[0] != other.legs[0] or self.second_leg_dual != other.second_leg_dual:
            raise ValueError('Incompatible legs!')
        backend = get_same_backend(self, other)
        data = backend.diagonal_elementwise_binary(
            self, other, func=func, func_kwargs=func_kwargs,
            partial_zero_is_zero=partial_zero_is_zero
        )
        return DiagonalTensor(data, first_leg=self.legs[0], second_leg_dual=self.second_leg_dual,
                              backend=backend, labels=self.labels)

    def __abs__(self):
        return self._elementwise_unary(func=operator.abs, maps_zero_to_zero=True)

    def __ge__(self, other):
        return self._binary_operand(other, func=operator.ge, operand='>=', is_bool_valued=True)

    def __gt__(self, other):
        return self._binary_operand(other, func=operator.gt, operand='>', is_bool_valued=True)

    def __le__(self, other):
        return self._binary_operand(other, func=operator.le, operand='<=', is_bool_valued=True)

    def __lt__(self, other):
        return self._binary_operand(other, func=operator.lt, operand='<', is_bool_valued=True)

    def __pow__(self, other):
        return self._binary_operand(other, func=operator.pow, operand='**', is_bool_valued=False)

    def __rpow__(self, other):
        if isinstance(other, Number):
            return self._elementwise_unary(func=lambda block: other ** block)
        raise TypeError(f'Invalid types for operand "**": {type(other)} and {type(self)}')

    # --------------------------------------------
    # Overriding methods from Tensor
    # --------------------------------------------

    def _getitem_apply_masks(self, masks: list[Mask], legs: list[int]) -> BlockDiagonalTensor | DiagonalTensor:
        if len(masks) == 2:
            if masks[0].same_mask(masks[1]):
                return self._apply_mask_both_legs(masks[0])
        warnings.warn('Converting DiagonalTensor to Tensor in order to apply mask', stacklevel=2)
        return self.as_Tensor()._getitem_apply_masks(masks, legs)

    def __getitem__(self, idcs):
        # allow indexing by a single integer -> applied to both axes
        _idcs = to_iterable(idcs)
        if len(_idcs) == 1 and isinstance(_idcs[0], int):
            idcs = (_idcs[0], _idcs[0])
        return Tensor.__getitem__(self, idcs)

    def __mul__(self, other):
        if isinstance(other, DiagonalTensor):
            # _elementwise_binary performs input checks.
            return self._elementwise_binary(other, func=operator.mul, partial_zero_is_zero=True)
        return Tensor.__mul__(self, other)

    def __setitem__(self, idcs, value):
        _idcs = to_iterable(idcs)
        if len(_idcs) == 1 and isinstance(_idcs[0], int):
            idcs = (_idcs[0], _idcs[0])
        return Tensor.__setitem__(self, idcs, value)

    def __truediv__(self, other):
        if isinstance(other, DiagonalTensor):
            # _elementwise_binary performs input checks.
            return self._elementwise_binary(other, func=operator.truediv)
        return Tensor.__truediv__(self, other)

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, first_leg: Space, second_leg_dual: bool = True, backend=None,
             labels: list[str | None] = None, dtype: Dtype = Dtype.float64) -> DiagonalTensor:
        if backend is None:
            backend = get_backend(first_leg.symmetry)
        data = backend.zero_diagonal_data(leg=first_leg, dtype=dtype)
        return cls(data=data, first_leg=first_leg, second_leg_dual=second_leg_dual, backend=backend,
                   labels=labels)

    def add_trivial_leg(self, label: str = None, is_dual: bool = False, pos: int = -1,
                        to_domain: bool = None) -> DiagonalTensor:
        warnings.warn('Converting DiagonalTensor to Tensor for add_trivial_leg', stacklevel=2)
        return self.as_Tensor.add_trivial_leg(label=label, is_dual=is_dual, pos=pos, to_domain=to_domain)

    @amend_parent_docstring(parent=Tensor.apply_mask)
    def apply_mask(self, mask: Mask, leg: int | str = BOTH) -> DiagonalTensor | BlockDiagonalTensor:
        """.. note ::
            For ``DiagonalTensor``s, :meth:`apply_mask` has a default argument for `legs` which
            causes the mask to be applied to *both* legs.
            If it is only applied to a single leg, the result will be a `Tensor` instead if `DiagonalTensor`.
        """
        if leg is BOTH:
            return self._apply_mask_both_legs(mask)
        return self.as_Tensor().apply_mask(mask, leg)

    def as_Tensor(self) -> BlockDiagonalTensor:
        """Forget about diagonal structure and convert to a Tensor"""
        return BlockDiagonalTensor(
            data=self.backend.full_data_from_diagonal_tensor(self),
            legs=self.legs, num_domain_legs=1, backend=self.backend, labels=self.labels
        )

    def combine_legs(self,
                     *legs: list[int | str],
                     product_spaces: list[ProductSpace] = None,
                     product_spaces_dual: list[bool] = None,
                     new_axes: list[int] = None,
                     new_labels: list[str | None] = None) -> BlockDiagonalTensor:
        """See :func:`tenpy.linalg.tensors.combine_legs`."""
        warnings.warn('Converting DiagonalTensor to Tensor in order to combine legs', stacklevel=2)
        return self.as_Tensor().combine_legs(
            *legs, product_spaces=product_spaces, product_spaces_dual=product_spaces_dual,
            new_axes=new_axes, new_labels=new_labels
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

    def item(self) -> float | complex:
        if all(leg.dim == 1 for leg in self.legs):
            return self.backend.item(self)
        else:
            raise ValueError('Not a scalar')
        
    def norm(self, order=None) -> float:
        if order is None:
            order = 2
        return self.backend.norm(self, order=order)

    def permute_legs(self, permutation: list[int | str] = None, num_domain_legs: int = None
                     ) -> DiagonalTensor:
        assert num_domain_legs in [None, 1], 'DiagonalTensor must have one leg in the domain.'
        permutation = self.get_leg_idcs(permutation)
        return DiagonalTensor(data=self.data, first_leg=self.legs[permutation[0]],
                              second_leg_dual=self.second_leg_dual, backend=self.backend,
                              labels=[self.shape._labels[n] for n in permutation])

    def split_legs(self, *legs: int | str) -> BlockDiagonalTensor:
        warnings.warn('Converting DiagonalTensor to Tensor in order to split legs', stacklevel=2)
        return self.as_Tensor().split_legs(*legs)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> NoReturn:
        raise TypeError(f'{type(self)} does not support squeeze_legs')

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        # need to fill in the off-diagonal zeros anyway, so we may as well use as_Tensor first.
        # OPTIMIZE a specialized implementation could be slightly more efficient...
        return self.as_Tensor().to_dense_block(leg_order)

    def trace(self, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
              ) -> float | complex:
        leg_idcs1 = self.get_leg_idcs(legs1)
        leg_idcs2 = self.get_leg_idcs(legs2)
        if not len(leg_idcs1) == 1 == len(leg_idcs2):
            raise ValueError('Wrong number of legs.')
        if leg_idcs1[0] == leg_idcs2[0]:
            raise ValueError('Duplicate leg')
        return self.backend.diagonal_tensor_trace_full(self)

    def _get_element(self, idcs: list[int]) -> float | complex:
        if idcs[0] != idcs[1]:
            return self.dtype.zero_scalar
        return self.backend.get_element_diagonal(self, idcs[0])

    def _set_element(self, idcs: list[int], value: float | complex) -> None:
        if idcs[0] != idcs[1]:
            raise IndexError('Off-diagonal entry can not be set for DiagonalTensor')
        self.data = self.backend.set_element_diagonal(self, idcs[0], value)

    def _mul_scalar(self, other: complex) -> DiagonalTensor:
        return self._elementwise_unary(lambda block: self.backend.block_mul(other, block),
                                       maps_zero_to_zero=True)

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------
    
    def almost_equal(self, other: DiagonalTensor, atol: float = 1e-5, rtol: float = 1e-8,
                     allow_different_types: bool = False) -> bool:
        if not isinstance(other, DiagonalTensor):
            if not allow_different_types:
                raise TypeError(f'Different types: {type(self)} and {type(other)}.')
            if isinstance(other, (BlockDiagonalTensor, ChargedTensor)):
                return self.as_Tensor().almost_equal(other, atol=atol, rtol=rtol, allow_different_types=True)
            else:
                raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')

        # can now assume that other is a DiagonalTensor
        _ = self._input_checks_same_legs(other)
        # no need to permute legs, diagonal data is the same either way.
        return get_same_backend(self, other).almost_equal_diagonal(self, other, atol, rtol)

    def inner(self, other: Tensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        t1 = self.conj() if do_conj else self
        return t1.tdot(other, legs1=[0, 1], legs2=leg_order_2)

    def outer(self, other: Tensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> Tensor:
        if isinstance(other, DiagonalTensor):
            warnings.warn('Converting DiagonalTensors to Tensors for outer', stacklevel=2)
            other = other.as_Tensor()
        if isinstance(other, (BlockDiagonalTensor, ChargedTensor)):
            return self.as_Tensor().outer(other, relabel1=relabel1, relabel2=relabel2)
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: Tensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> Tensor | float | complex:
        if isinstance(other, ChargedTensor):
            legs2 = other.get_leg_idcs(legs2)  # make sure we reference w.r.t. other, not other.invariant_part
            assert relabel2 is None or other.invariant_part.labels[-1] not in relabel2
            invariant_part = self.tdot(other.invariant_part, legs1=legs1, legs2=legs2,
                                       relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(invariant_part=invariant_part, dummy_leg_state=other.dummy_leg_state)
        legs1, legs2 = self._input_checks_tdot(other, legs1, legs2)
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
        backend = get_same_backend(self, other)
        if isinstance(other, Mask):
            if legs2[0] == 1:
                # OPTIMIZE ? dont need to convert to full but its easier for now
                return self.tdot(other.as_Tensor(), legs1=legs1, legs2=legs2, relabel1=relabel1, relabel2=relabel2)
            # else: legs2[0] == 0, i.e. we contract the large leg of the Mask
            new_label = other.labels[1]
            if relabel1 is not None:
                new_label = relabel1.get(new_label, new_label)
            res = self.apply_mask(other, leg=legs1[0], new_label=new_label)
            res = res.permute_legs(legs1)
            if relabel2 is not None:
                res.set_labels([relabel2.get(res.labels[0], res.labels[0]), res.labels[-1]])
            return res
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
        if isinstance(other, BlockDiagonalTensor):
            res = BlockDiagonalTensor(
                data=backend.scale_axis(other, self, leg=legs2[0]),
                legs=other.legs[:legs2[0]] + [self.legs[1 - legs1[0]]] + other.legs[legs2[0] + 1:],
                num_domain_legs=other.num_domain_legs,  # TODO this correct?
                backend=backend,
                labels=other.labels[:legs2[0]] + [self.labels[1 - legs1[0]]] + other.labels[legs2[0] + 1:]
            )
            # move scaled leg to the front
            return res.permute_legs([legs2[0]] + list(range(legs2[0])) + list(range(legs2[0] + 1, res.num_legs)))
            
        raise TypeError(f'tdot not supported for {type(self)} and {type(other)}')

    def _add_tensor(self, other: Tensor) -> Tensor:
        if isinstance(other, BlockDiagonalTensor):
            return self.as_Tensor()._add_tensor(other)
        if isinstance(other, DiagonalTensor):
            other_order = self._input_checks_same_legs(other)
            # by definition, permuting the legs does nothing to a DiagonalTensors data
            backend = get_same_backend(self, other)
            return DiagonalTensor(backend.add(self, other), first_leg=self.legs[0],
                                  second_leg_dual=self.second_leg_dual, backend=backend,
                                  labels=self.labels)
        raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")


class Mask(Tensor):
    r"""A boolean mask that can be used to project a leg.

    Projecting a leg can be done via `tdot`, which views the projection, which is a linear map,
    as an :class:`Tensor`. Then, it can only project the :attr:`large_leg`.
    Using :meth:`Tensor.apply_mask` on the other hand allows a mask to be applied either
    to :attr:`large_leg` or its dual.

    Parameters
    ----------
    data
        The numerical data (i.e. boolean flags) comprising the mask. type is backend-specific.
        Should have boolean dtype.
    large_leg : Space
        The larger leg that can be projected with this mask.
        The resulting mask has `mask.legs[0] == large_leg.dual`, which can be contracted with
        the `large_leg`.
    small_leg : ElementarySpace
        FIXME condition for duality
        The small leg, i.e. the masked / projected version of `large_leg`.
        Should have same :attr:`ElementarySpace.is_dual` as `large_leg`.
    backend: :class:`~tenpy.linalg.backends.abstract_backend.Backend`, optional
        The backend for the Tensor
    labels : list[str | None] | None
        Labels for the legs. If None, translates to ``[None, None, ...]`` of appropriate length
    """
    def __init__(self, data, large_leg: Space, small_leg: ElementarySpace, backend=None,
                 labels: list[str | None] = None):
        assert small_leg.is_subspace_of(large_leg.as_ElementarySpace())
        self.data = data
        if backend is None:
            backend = get_backend(large_leg.symmetry)
        Tensor.__init__(self, legs=[large_leg.dual, small_leg], backend=backend, labels=labels,
                        dtype=Dtype.bool, num_domain_legs=1)

    def test_sanity(self) -> None:
        super().test_sanity()
        self.backend.test_mask_sanity(self)
        assert self.large_leg.is_dual == self.small_leg.is_dual
        assert self.small_leg.is_subspace_of(self.large_leg)
        assert self.dtype == Dtype.bool

    # --------------------------------------------
    # Additional methods (not in Tensor)
    # --------------------------------------------

    @property
    def blockmask(self) -> Block:
        """1D block of bools, indicating which basis elements are kept."""
        return self.backend.diagonal_to_block(self)
    
    @property
    def large_leg(self) -> Space:
        """The large leg that can be projected using this mask."""
        return self.legs[0].dual

    @property
    def numpymask(self) -> np.ndarray:
        """1D numpy array of bools, indicating which basis elements are kept."""
        return self.backend.block_to_numpy(self.blockmask)

    @property
    def small_leg(self) -> ElementarySpace:
        """The slice of :attr:`large_leg` that the mask projects to."""
        return self.legs[1]

    @classmethod
    def eye(cls, large_leg: Space, backend=None, labels: list[str | None] = None) -> Mask:
        """The identity mask, that keeps all basis elements."""
        # OPTIMIZE
        return cls.from_blockmask(np.ones(large_leg.dim, dtype=bool), large_leg=large_leg,
                                  backend=backend, labels=labels)

    @classmethod
    def from_blockmask(cls, blockmask: Block, large_leg: Space, backend: Backend = None,
                   labels: list[str | None] = None) -> Mask:
        """Create a Mask from a 1D boolean block.

        TODO find a better word than blockmask. Use it everywhere else too.
             We need to distinguish the concepts of the 2D rectangular block and this 1D blockmask.
             Consider renaming numpymask property too.

        Parameters
        ----------
        blockmask : 1D Block-like of bool
            A block with boolean entries, where ``blockmask[i]`` indicates if the ``i``-th element
            of the computational basis of `large_leg` should be kept.
        large_leg, backend, labels
            Same as for :meth:`Mask.__init__`.
        """
        small_leg = large_leg.take_slice(np.asarray(blockmask, bool))
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)
        blockmask = backend.block_to_dtype(backend.as_block(blockmask), Dtype.bool)
        return cls(backend.mask_from_block(blockmask, large_leg=large_leg, small_leg=small_leg),
                   large_leg=large_leg, small_leg=small_leg, backend=backend, labels=labels)

    @classmethod
    def from_DiagonalTensor(cls, diag: DiagonalTensor) -> Mask:
        """Create a Mask from a bool-valued DiagonalTensor"""
        # OPTIMIZE could do this "directly", without converting to block, but its easier for now...
        return cls.from_blockmask(diag.backend.block_to_dtype(diag.diag_block, Dtype.bool),
                                  large_leg=diag.legs[0], backend=diag.backend, labels=diag.labels)

    @classmethod
    def from_indices(cls, indices: int | list[int] | np.ndarray | slice, large_leg: Space,
                     backend: Backend = None, labels: list[str | None] = None) -> Mask:
        """Create a Mask from the indices that are kept.

        Parameters
        ----------
        indices
            Valid index/indices for a 1D numpy array. The elements of the computational basis of
            `large_leg` with these indices are kept by the projection.
        large_leg, backend, labels
            Same as for :meth:`Mask.__init__`.
        """
        mask = np.zeros(large_leg.dim, dtype=bool)
        mask[indices] = True
        return cls.from_blockmask(mask, large_leg=large_leg, backend=backend, labels=labels)

    def all(self) -> bool:
        """If the mask keeps all basis elements"""
        return self.small_leg.dim == self.large_leg.dim

    def any(self) -> bool:
        """If the mask keeps any basis elements"""
        return self.small_leg.dim > 0

    def as_Tensor(self, dtype=Dtype.float64) -> BlockDiagonalTensor:
        return BlockDiagonalTensor(
            data=self.backend.full_data_from_mask(self, dtype=dtype),
            legs=self.legs, num_domain_legs=1, backend=self.backend, labels=self.labels
        )

    def same_mask(self, other: Mask) -> bool:
        return (self.large_leg == other.large_leg) and Mask.all(self == other)

    def logical_not(self) -> Mask:
        """The opposite mask that keeps exactly what self discard.

        Viewing masks as projectors, this is the orthogonal complement.
        """
        # OPTIMIZE
        return Mask.from_blockmask(~self.blockmask, large_leg=self.large_leg,
                                   backend=self.backend, labels=self.labels)

    def _binary_operand(self, other: bool | Mask, func, operand: str, return_NotImplemented: bool = True
                        ) -> Mask:
        """Utility function for a shared implementation of binary functions, whose second argument
        may be a scalar ("to be broadcast") or a Mask.

        Parameters
        ----------
        other
            Either a bool or a Mask.
        func
            The function with signature
            ``func(self_block: Block, other_or_other_block: bool | Block) -> Block``
        operand
            A string representation of the operand, used in error messages
        return_NotImplemented
            Whether `NotImplemented` should be returned on a non-scalar and non-`Tensor` other.
        """
        if isinstance(other, bool):
            backend = backend
            data = self.backend.diagonal_elementwise_unary(self, lambda block: func(block, other),
                                                           func_kwargs={}, maps_zero_to_zero=False)
            return Mask(data, large_leg=self.large_leg, small_leg=None, backend=backend,
                        labels=self.labels)
        elif isinstance(other, Mask):
            backend = get_same_backend(self, other)
            if self.legs[0] != other.legs[0]:
                raise ValueError('Incompatible legs!')
            # OPTIMIZE
            return Mask.from_blockmask(func(self.blockmask, other.blockmask),
                                       large_leg=self.large_leg, backend=backend,
                                       labels=_get_same_labels(self.labels, other.labels))
        elif return_NotImplemented and not isinstance(other, (Tensor, Number)):
            return NotImplemented
        msg = f'Invalid types for operand "{operand}": {type(self)} and {type(other)}'
        raise TypeError(msg)

    def __and__(self, other) -> bool:
        return self._binary_operand(other, func=operator.and_, operand='&')

    def __bool__(self):
        raise ValueError('The truth value of a mask is undefined. Use Mask.any() or Mask.all().')

    def __eq__(self, other) -> bool:
        return self._binary_operand(other, func=operator.eq, operand='==')

    def __ne__(self, other) -> bool:
        return self._binary_operand(other, func=operator.ne, operand='!=')

    def __rand__(self, other) -> bool:
        return self._binary_operand(other, func=operator.and_, operand='&')

    def __ror__(self, other) -> bool:
        return self._binary_operand(other, func=operator.or_, operand='|')

    def __rxor__(self, other) -> bool:
        return self._binary_operand(other, func=operator.xor, operand='^')

    def __or__(self, other) -> bool:
        return self._binary_operand(other, func=operator.or_, operand='|')

    def __xor__(self, other) -> bool:
        return self._binary_operand(other, func=operator.xor, operand='^')

    # --------------------------------------------
    # Overriding methods from Tensor
    # --------------------------------------------

    def __getitem__(self, idcs):
        # TODO what even is the expected behavior here?
        raise NotImplementedError

    def __setitem__(self, idcs, value):
        # TODO what even is the expected behavior here?
        raise NotImplementedError

    # --------------------------------------------
    # Implementing abstractmethods
    # --------------------------------------------
    
    @classmethod
    def zero(cls, large_leg: Space, backend=None, labels: list[str | None] = None) -> Mask:
        """The zero mask, that discards all basis elements."""
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)
        data = backend.zero_diagonal_data(leg=large_leg, dtype=Dtype.bool)
        
        small_leg = ElementarySpace.from_null_space(symmetry=large_leg.symmetry, is_dual=large_leg.is_bra_space)
        return cls(data=data, large_leg=large_leg, small_leg=small_leg, backend=backend)

    def add_trivial_leg(self, label: str = None, is_dual: bool = False, pos: int = -1,
                        to_domain: bool = None) -> DiagonalTensor:
        raise TypeError(f'add_trivial_leg not supported for types {type(self)}.')

    def apply_mask(self, mask: Mask, leg: int | str) -> Mask:
        # in principle, this is possible. it is cumbersome though, and i dont think we need it.
        raise TypeError('apply_mask() is not supported for Mask')

    def combine_legs(self, *legs: list[int | str],
                     product_spaces: list[ProductSpace] = None,
                     product_spaces_dual: list[bool] = None,
                     new_axes: list[int] = None,
                     new_labels: list[str | None] = None) -> BlockDiagonalTensor:
        """See :func:`tenpy.linalg.tensors.combine_legs`."""
        msg = 'Converting Mask to full Tensor for `combine_legs`. If this is what you wanted, ' \
              'explicitly convert via Mask.as_Tensor() first to suppress the warning.'
        warnings.warn(msg, stacklevel=2)
        return self.as_Tensor().combine_legs(
            *legs, product_spaces=product_spaces, product_spaces_dual=product_spaces_dual,
            new_axes=new_axes, new_labels=new_labels
        )

    def conj(self) -> Mask:
        return self

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

    def item(self) -> bool:
        if self.large_leg.dim == 1:
            return self.backend.item(self)
        raise ValueError('Not a scalar')

    def norm(self, order=None) -> float:
        num_true_entries = self.small_leg.dim
        if order is None:
            return float(np.sqrt(num_true_entries))
        if order >= np.inf:
            return 1. if num_true_entries > 0 else 0.
        if order <= -np.inf:
            return 0. if num_true_entries < self.large_leg.dim else 1.
        if order == 0:
            return num_true_entries
        return num_true_entries ** (1. / order)

    def permute_legs(self, permutation: list[int] = None, num_domain_legs: int = None) -> BlockDiagonalTensor:
        msg = 'Converting Mask to full Tensor for `permute_legs`. If this is what you wanted, ' \
              'explicitly convert via Mask.as_Tensor() first to suppress the warning.'
        warnings.warn(msg, stacklevel=2)
        return self.as_Tensor().permute_legs(permutation, num_domain_legs=num_domain_legs)

    def split_legs(self, legs: list[int | str] = None) -> NoReturn:
        msg = 'Converting Mask to full Tensor for `split_legs`. If this is what you wanted, ' \
              'explicitly convert via Mask.as_Tensor() first to suppress the warning.'
        warnings.warn(msg, stacklevel=2)
        return self.as_Tensor().permute_legs(legs)

    def squeeze_legs(self, legs: int | str | list[int | str] = None) -> BlockDiagonalTensor:
        msg = 'Converting Mask to full Tensor for `squeeze_legs`. If this is what you wanted, ' \
              'explicitly convert via Mask.as_Tensor() first to suppress the warning.'
        warnings.warn(msg, stacklevel=2)
        return self.as_Tensor().squeeze_legs(legs)

    def to_dense_block(self, leg_order: list[int | str] = None) -> Block:
        # OPTIMIZE a dedicated implementation could be slightly more efficient
        return self.as_Tensor().to_dense_block(leg_order)

    def trace(self, *a, **k) -> NoReturn:
        raise TypeError('Can not perform trace of a Mask, they are not square.')

    def _get_element(self, idcs: list[int]) -> bool:
        raise NotImplementedError  # TODO

    def _set_element(self, idcs: list[int], value: bool) -> None:
        assert isinstance(value, bool)
        raise NotImplementedError  # TODO

    def _mul_scalar(self, other: Number) -> Mask:
        if isinstance(other, bool):
            return self.__and__(other)
        raise TypeError(f'Can not multiply Mask with {type(other)}.')

    # --------------------------------------------
    # Implementing binary tensor methods
    # --------------------------------------------

    def almost_equal(self, other: Tensor, atol: float = 1e-5, rtol: float = 1e-8,
                     allow_different_types: bool = False) -> bool:
        if isinstance(other, Mask):
            return self.same_mask(other)
        raise TypeError(f'almost_equal not supported for types {type(self)} and {type(other)}.')

    def inner(self, other: Tensor, do_conj: bool = True, legs1: list[int | str] = None,
              legs2: list[int | str]  = None) -> float | complex:
        leg_order_2 = self._input_checks_inner(other, do_conj=do_conj, legs1=legs1, legs2=legs2)
        if isinstance(other, Mask):
            return self.__and__(other).small_leg.dim
        else:
            return other.apply_mask(self, leg=leg_order_2[0]).trace()

    def outer(self, other: Tensor, relabel1: dict[str, str] = None,
              relabel2: dict[str, str] = None) -> Tensor:
        raise TypeError(f'outer not supported for {type(self)} and {type(other)}')

    def tdot(self, other: Tensor, legs1: int | str | list[int | str] = -1,
             legs2: int | str | list[int | str] = 0, relabel1: dict[str, str] = None,
             relabel2: dict[str, str] = None) -> Tensor | float | complex:
        legs1, legs2 = self._input_checks_tdot(other, legs1, legs2)
        if len(legs1) == 1:
            if legs1[0] == 0:  # contracting the large leg
                new_label = self.labels[1]
                if relabel1 is not None:
                    new_label = relabel1.get(new_label, new_label)
                res = other.apply_mask(self, legs2[0], new_label=new_label)
                res = res.permute_legs(legs2 + [n for n in range(res.num_legs) if n not in legs2])
                if relabel2 is not None:
                    res.set_labels([res.labels[0]] + [relabel2.get(l, l) for l in res.labels[1:]])
                return res
            
            # OPTIMIZE the remaining case could be done more efficiently, by inserting zero-slice wherever the mask is False
            return self.as_Tensor().tdot(other, legs1, legs2, relabel1, relabel2)
        if len(legs1) == 0:
            raise TypeError(f'tdot with no contracted legs (i.e. outer) not supported for {type(self)} and {type(other)}')
        if len(legs1) == 2:
            large_leg_idx = legs1[0]  # since legs1 is [0, 1] or [1, 0], legs1[legs1[0]] == 0 is the large leg
            res = other.apply_mask(self, legs2[large_leg_idx])
            res = res.trace(*legs2)
            if relabel2 is not None:
                res.set_labels([relabel2.get(l, l) for l in res.labels])
            return res
        raise ValueError  # should have been caught by input checks

    def _add_tensor(self, other: Tensor) -> Tensor:
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

# ##################################
# API functions
# ##################################

# TODO (JU) find a good way to write type hints for these, having in mind the possible combinations
#           of Tensor-subtypes.

def add_trivial_leg(tens: Tensor, label: str = None, is_dual: bool = False, pos: int = -1,
                    to_domain: bool = None):
    """Add a trivial leg to a tensor.

    A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.

    Parameters
    ----------
    tens : Tensor
        The tensor to add a leg to.
    label : str | None
        The label for the new leg
    is_dual : bool
        If the new leg should be dual
    pos : int
        The position to insert the new leg at.
        The result has the new trivial leg at ``result.legs[pos]``.
    to_domain : int
        If the new leg should be added to domain or codomain. See :ref:`tensors_as_maps`.
        If ``pos < tens.num_domain_legs`` we always add to the domain (or raise) and
        if ``pos > tens.num_domain_legs`` we always add to the codomain (or raise).
        There is only a choice if ``pos == tens.num_domain_legs``.
        Then, we get ``result.num_domain_legs == tens.num_domain_legs + int(to_domain)``.
        Per default, we add to the codomain.
    """
    return tens.add_trivial_leg(label=label, is_dual=is_dual, pos=pos, to_domain=to_domain)


def almost_equal(t1: Tensor, t2: Tensor, atol: float = 1e-5, rtol: float = 1e-8,
                 allow_different_types: bool = False) -> bool:
    """Checks if two tensors are equal up to numerical tolerance.

    The blocks of the two tensors are compared.
    The tensors count as almost equal if all block-entries, i.e. all their free parameters
    individually fulfill `abs(a1 - a2) <= atol + rtol * abs(a1)`.

    In the non-symmetric case, this is equivalent to e.g. ``numpy.allclose``.
    In the symmetric case, it is a close analogue.

    .. note ::
        The definition is not symmetric, so there may be edge-cases where
        ``almost_equal(t1, t2) != almost_equal(t2, t1)``

    Parameters
    ----------
    t1, t2
        The tensors to compare
    atol, rtol
        Absolute and relative tolerance, see above.
    allow_different_types : bool
        If ``False`` (default), comparing tensors of different types raises a ``TypeError``.
        If ``True``, types are converted, *if possible*.
        
    """
    return t1.almost_equal(t2, atol=atol, rtol=rtol, allow_different_types=allow_different_types)


def combine_legs(t: Tensor,
                 *legs: list[int | str],
                 product_spaces: list[ProductSpace] = None,
                 product_spaces_dual: list[bool] = None,
                 new_axes: list[int] = None,
                 new_labels: list[str | None] = None):
    """
    Combine (multiple) groups of legs on a tensor to (multiple) ProductSpaces.

    .. warning ::
        Combining legs introduces a basis-transformation. This is important to consider if
        you convert to a dense block (e.g. via :meth:`Tensor.to_dense_block`). In
        particular ``some_tens.combine_legs(...).to_dense_block()`` is not equivalent
        to ``some_tens.to_dense_block().reshape(...)``.
        See :meth:`ProductSpace.get_basis_transformation`.

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
    TODO doc or remove new_axes
    new_labels : list of str
        A new label for each group of legs.
        By default, the label is given by joining the original labels with dots ``'.'`` and
        surrounding with parentheses. For example, combining legs ``'vL', 'p0', 'p1'`` results in
        a leg with label ``'(vL.p0.p1)'``.
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
                          new_axes=new_axes, new_labels=new_labels)


def conj(t: Tensor) -> Tensor:
    """
    The conjugate of t, living in the dual space.
    Labels are adjusted as `'p'` -> `'p*'` and `'p*'` -> `'p'`
    """
    return t.conj()


def detect_sectors_from_block(block: Block, legs: list[Space], backend: Backend
                              ) -> SectorArray:
    """Detect the symmetry sectors of a dense block.

    Given a `block` that represents a symmetric tensor with the given `legs`, return
    the sectors (one sector per leg) of the largest (by magnitude) entry of the `block`.
    """
    assert backend.block_shape(block) == tuple(l.dim for l in legs)
    idcs = backend.block_abs_argmax(block)
    sectors = [leg.sectors[leg.parse_index(i)[0]] for leg, i in zip(legs, idcs)]
    return np.stack(sectors, axis=0)


def entropy(p: DiagonalTensor | Sequence[float], n: float = 1):
    r"""Entropy of a probability distribution.

    Assumes that `p` is a normalized (``sum(p) == 1``) probability distribution (real, non-negative).

    Parameters
    ----------
    p : :class:`DiagonalTensor` | iterable of float
        A normalized distribution.
    n : 1 | float | np.inf
        Selects the entropy, see below.

    Returns
    -------
    entropy : float
        Shannon-entropy :math:`-\sum_i p_i \log(p_i)` (n=1) or
        Renyi-entropy :math:`\frac{1}{1-n} \log(\sum_i p_i^n)` (n != 1)
        of the distribution `p`.
    """
    if isinstance(p, DiagonalTensor):
        p = p.diag_numpy
    else:
        p = np.array(p)
    assert len(p.shape) == 1
    p = p[p > 1e-30]  # for stability reasons / to avoid NaN in log
    if n == 1:
        return -np.inner(np.log(p), p)
    elif n == np.inf:
        return -np.log(np.max(p))
    else:  # general n != 1, inf
        return np.log(np.sum(p**n)) / (1. - n)


def flip_leg_duality(t: Tensor, which_leg: int | str, *more: int | str) -> Tensor:
    r"""Flip the duality of one or more legs.

    E.g. flipping the second leg of a three-leg tensor (assuming the first leg is ket-like) means

    .. math ::
        t = \sum_{i,j,k} c_{i, j, k} \ket{i} \otimes \ket{j} \otimes \bra{k}
        \mapsto \sum_{i,j} c_{i, j} \ket{i} \otimes \bra{j} \otimes \bra{k}

    Notes
    -----
    This is a basis-dependent notion. It depends on the choice of (co-)evaluation maps.
    Here we choose the canonical maps

    .. math ::
        \varepsilon_V\colon V^* \otimes V \to \mathbb{C} , \bra{m} \otimes \bra{n} \mapsto \delta_{m, n}

        \eta_V\colon \mathbb{C} \to V \otimes V^* , \alpha \mapsto \alpha \sum_n \ket{n} \otimes \bra{n}

    which are defined by linear extension and depend on the basis :math:`\set{n}`.
    """
    return t.flip_leg_duality(which_leg, *more)


def hconj(t: Tensor, legs1: int | str | list[int | str] = -1,
          legs2: int | str | list[int | str] = None) -> Tensor:
    """Hermitian conjugate.

    This is basically :func:`conj` with additional input checks and a :func:`permute_legs`, such
    that the result has the same legs as `t`, and can e.g. be added to `t`.
    """
    return t.hconj(legs1=legs1, legs2=legs2)


def inner(t1: Tensor, t2: Tensor, do_conj: bool = True,
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
        in lax label mode, like in :meth:`_match_leg_order`.
        Otherwise, ``legs1[n]`` of ``t1`` is contracted with ``legs2[n]`` of ``t2``, where
        a single ``None``/unspecified list is interpreted as ``list(range(num_legs))``.
    """
    return t1.inner(t2, do_conj=do_conj, legs1=legs1, legs2=legs2)


def is_scalar(obj) -> bool:
    """If obj is a scalar, meaning either a python scalar like float or complex, or a Tensor
    which has only one-dimensional legs"""
    if isinstance(obj, Tensor):
        return all(d == 1 for d in obj.shape)
    # checking for Number includes int, float, complex, but also e.g. np.int64()
    if isinstance(obj, Number):
        return True
    return False


def norm(t: Tensor, order=None) -> float:
    """Norm of a tensor.

    Equivalent to ``np.linalg.norm(a.to_numpy().flatten(), order)``.
    TODO is this statement true for general nonabelian symmetries?

    In contrast to numpy, we don't distinguish between matrices and vectors, but rather
    compute the `order`-norm of the "flattened" coefficients `x` of `t` in the computational basis.

    ==========  ======================================
    ord         norm
    ==========  ======================================
    None        Frobenius norm (same as 2-norm)
    np.inf      ``max(abs(x))``
    -np.inf     ``min(abs(x))``
    0           ``sum(x != 0) == np.count_nonzero(x)``
    other       usual p-norm with p=`order`
    ==========  ======================================

    Parameters
    ----------
    t : :class:`Tensor`
        The tensor of which the norm should be calculated
    order :
        The order of the norm. See table above.
    """
    return t.norm(order=order)


def outer(t1: Tensor, t2: Tensor, relabel1: dict[str, str] = None,
          relabel2: dict[str, str] = None) -> Tensor:
    """outer product, aka tensor product, aka direct product of two tensors"""
    return t1.outer(t2, relabel1=relabel1, relabel2=relabel2)


def permute_legs(t: Tensor, permutation: list[int] = None, num_domain_legs: int = None) -> Tensor:
    """Change the order of legs of a Tensor.

    TODO elaborate what this does to the underlying data for the respective backends.

    Parameters
    ----------
    t : Tensor
        The tensor to permute
    permutation : list of int, optional
        The permutation. The new legs will be ``[T.legs[n] for n in permutation]``.
    num_domain_legs : int or None
        The number of legs that should be in the domain. Per default (``None``), it is not changed.

    Returns
    -------
    Tensor
        A copy with modified leg order.
    """
    # TODO: also have an inplace version?
    return t.permute_legs(permutation, num_domain_legs=num_domain_legs)


def split_legs(t: Tensor, *legs: int | str) -> BlockDiagonalTensor:
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


def squeeze_legs(t: Tensor, legs: int | str | list[int | str] = None) -> Tensor:
    """
    Remove trivial leg from tensor.
    If legs are specified, they are squeezed if they are trivial and a ValueError is raised if not.
    If legs is None (default), all trivial legs are squeezed
    """
    return t.squeeze_legs(legs=legs)


def tdot(t1: Tensor, t2: Tensor,
         legs1: int | str | list[int | str] = -1, legs2: int | str | list[int | str] = 0,
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None
         ) -> Tensor | float | complex:
    """
    Contraction of two tensors.

    A number of legs of `t1`, indicated by `legs1` are contracted with the *same* number of legs
    of `t2`, indicated by `legs2`. The pairs of legs need to be mutually dual.

    The legs of the resulting tensor are in "numpy-style" order, i.e. first the uncontracted legs
    of `t1`, then those of `t2`.

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
    return t1.tdot(t2, legs1=legs1, legs2=legs2, relabel1=relabel1, relabel2=relabel2)


def trace(t: Tensor, legs1: int | str | list[int | str] = -2, legs2: int | str | list[int | str] = -1
          ) -> Tensor | float | complex:
    """
    Trace over one or more pairs of legs, that is contract these pairs.
    """
    return t.trace(legs1=legs1, legs2=legs2)


def zero_like(tens: Tensor) -> Tensor:
    if isinstance(tens, BlockDiagonalTensor):
        return BlockDiagonalTensor.zero(legs=tens.legs, backend=tens.backend, labels=tens.labels,
                                        dtype=tens.dtype, num_domain_legs=tens.num_domain_legs)
    if isinstance(tens, DiagonalTensor):
        return DiagonalTensor.zero(first_leg=tens.first_leg, second_leg_dual=tens.second_leg_dual,
                                   backend=tens.backend, labels=tens.labels, dtype=tens.dtype)
    if isinstance(tens, Mask):
        return Mask.zero(large_leg=tens.large_leg, backend=tens.backend, labels=tens.labels)
    raise TypeError(f'zero is not defined for type {type(tens)}')


def eye_like(tens: Tensor) -> BlockDiagonalTensor | DiagonalTensor:
    """The identity map, as a tensor of the same type, with the same legs as `tens`.

    The legs of `tens` must be compatible with the identity map.
    There must be an even number, where the second half are the duals of the first half
    in reverse order, i.e. ``tens.legs[i] == tens.legs[-1 - i].dual``.
    """
    if isinstance(tens, BlockDiagonalTensor):
        if tens.num_legs % 2 != 0:
            raise ValueError('eye is not defined for an odd number of legs')
        legs_1 = tens.legs[:tens.num_legs // 2]
        legs_2 = tens.legs[tens.num_legs // 2:]
        for l1, l2 in zip(legs_1, legs_2[::-1]):
            if not l1.can_contract_with(l2):
                if len(legs_1) == 1:
                    msg = 'Second leg must be the dual of the first leg'
                else:
                    msg = 'Second half of legs must be the dual of the first half'
                raise ValueError(msg)
        return BlockDiagonalTensor.eye(legs=legs_1, backend=tens.backend, labels=tens.labels,
                                       dtype=tens.dtype)
    if isinstance(tens, DiagonalTensor):
        if tens.legs[0].is_dual == tens.legs[1].is_dual:
            raise ValueError('Second leg must be the dual of the first leg')
        return DiagonalTensor.eye(tens.legs[0], backend=tens.backend, labels=tens.labels,
                                  dtype=tens.dtype)
    if isinstance(tens, Mask):
        return Mask.eye(tens.large_leg, backend=tens.backend, labels=tens.labels)
    raise TypeError(f'eye is not defined for type {type(tens)}')


# ##################################
# element-wise function for DiagonalTensor
# ##################################

ElementwiseData = TypeVar('ElementwiseData', Number, DiagonalTensor)


def elementwise_function(block_func: str, func_kwargs={}, maps_zero_to_zero=False):
    """Decorator factory used to define elementwise functions.

    The resulting decorator can take a ``function(x: Number, *a, **kw) -> Number`` that is defined
    on numbers and turns it into a function ``wrapped`` that is roughly equivalent to::

        |   def wrapped(x, *a, **kw):
        |       if isinstance(x, DiagonalTensor):
        |           return DiagonalTensor(...)  # uses ``block_func(old_block, *a, **kw)``
        |       return function(x, *a, **kw)

    Parameters
    ----------
    block_func : str
        The name of a :class:`BlockBackend` method that implements the elementwise function on
        the level of backend-specific blocks, e.g. ``'block_real'``
        for :meth:`BlockBackend.block_real`.
    func_kwargs : dict
        Additional kwargs for the `block_func`, in addition to any kwargs given to the
        decorated function itself. The explicit kwargs, i.e. ``kw`` in the above summary,
        take priority.
    maps_zero_to_zero : bool
        If the function maps zero entries to zero.

    Returns
    -------
    decorator
        A function, to be used as a decorator, see summary above.
    """
    def decorator(function):
        @functools.wraps(function)
        def wrapped(x: ElementwiseData, *args, **kwargs) -> ElementwiseData:
            if isinstance(x, DiagonalTensor):
                return x._elementwise_unary(
                    functools.partial(getattr(x.backend, block_func), *args),
                    func_kwargs={**kwargs, **func_kwargs}, maps_zero_to_zero=maps_zero_to_zero
                )
            elif is_scalar(x):
                return function(x, *args, **kwargs)
            raise TypeError(f'Expected DiagonalTensor or scalar. Got {type(x)}')
        return wrapped
    return decorator


@elementwise_function('block_angle', maps_zero_to_zero=True)
def angle(x: ElementwiseData) -> ElementwiseData:
    """The angle of a complex number, elementwise.

    The counterclockwise angle from the positive real axis on the complex plane in the
    range (-pi, pi] with a real dtype. The angle of `0.` is `0.`.
    """
    return np.angle(x)
    

@elementwise_function('block_real', maps_zero_to_zero=True)
def real(x: ElementwiseData) -> ElementwiseData:
    """The real part of a complex number, elementwise."""
    return np.real(x)


@elementwise_function('block_real_if_close', func_kwargs=dict(tol=100), maps_zero_to_zero=True)
def real_if_close(x: ElementwiseData, tol: float = 100) -> ElementwiseData:
    """If the :func:`imag` part is close to 0, return the :func:`real` part. Elementwise.

    Parameters
    ----------
    x : :class:`DiagonalTensor` | Number
        The input complex number(s)
    tol : float
        The precision for considering the imaginary part "close to zero".
        Multiples of machine epsilon for the dtype of `x`.

    Returns
    -------
    If `x` is close to real, the real part of `x`. Otherwise the original complex `x`.
    """
    return np.real_if_close(x, tol=tol)


@elementwise_function('block_sqrt', maps_zero_to_zero=True)
def sqrt(x: ElementwiseData) -> ElementwiseData:
    """The square root of a number, elementwise."""
    return np.sqrt(x)


@elementwise_function('block_imag', maps_zero_to_zero=True)
def imag(x: ElementwiseData) -> ElementwiseData:
    """The imaginary part of a complex number, elementwise."""
    return np.imag(x)


# we define abs via __abs__ on diagonal tensors
    

# ##################################
# utility functions
# ##################################


def get_same_backend(*tensors: Tensor, error_msg: str = 'Incompatible backends.'
                     ) -> Backend:
    """If all tensors have the same backend, return it. Otherwise raise a ValueError"""
    try:
        backend = tensors[0].backend
    except IndexError:
        raise ValueError('expected at least one tensor') from None
    if not all(tens.backend == backend for tens in tensors):
        raise ValueError(error_msg)
    return backend


def tensor_from_block(block: Block, legs: list[Space], backend: Backend,
                      labels: list[str] = None, num_domain_legs: int = 0
                      ) -> ChargedTensor | BlockDiagonalTensor:
    """Assume the block lives in a single symmetry sector and convert to tensor.

    Parameters
    ----------
    block : backend-specific block
        The data to convert.
    legs : list of :class:`Space`
        The legs of the resulting tensor. Length must match number of axes of `block`.
    backend : :class:`Backend`
        The backend for the resulting tensor.
    labels : list of str, optional
        The labels for the resulting tensor.
    num_domain_legs : int
        How many of the legs should be in the domain. See :ref:`tensors_as_maps`.

    Returns
    -------
    If the block is symmetric, i.e. lives in the trivial sector, returns a :class:`Tensor`.
    Otherwise, returns a :class:`ChargedTensor`.
    """
    # TODO test
    block = backend.as_block(block)
    sectors = detect_sectors_from_block(block=block, legs=legs, backend=backend)
    symmetry = legs[0].symmetry
    if not symmetry.is_abelian:
        # TODO in the non-abelian case we can not infer a single sector from the largest entry.
        #  multiple sectors could have entries at that position.
        #  The code below identifies all of these sectors and packages them in the dummy_leg.
        #  We should either:
        #   a) Assume the block is only in one of those sectors and identify which be calculating
        #      overlaps.
        #   b) leave that as is (allow multi-dimensional dummy_leg) and adjust docs accordingly
        raise NotImplementedError
    assert all(leg.symmetry == symmetry for leg in legs[1:])
    dummy_leg = ProductSpace([ElementarySpace(symmetry, [s]) for s in sectors]).as_ElementarySpace().dual
    if np.all(dummy_leg.sectors == symmetry.trivial_sector[None, :]):
        return BlockDiagonalTensor.from_dense_block(block, legs=legs, backend=backend, labels=labels,
                                                    num_domain_legs=num_domain_legs)
    else:
        assert dummy_leg.dim == 1
        return ChargedTensor.from_dense_block(block, legs=legs, backend=backend, labels=labels,
                                              charge=dummy_leg, num_domain_legs=num_domain_legs,
                                              dummy_leg_state=backend.as_block([1]))


# ##################################
# "private" helper functions
# ##################################


def _match_leg_order(t1: Tensor, t2: Tensor) -> list[int] | None:
    """Utility function to determine how to match legs of two tensors.

    In strict label mode, returns the permutation ``perm`` that matches the legs "by label",
    i.e. such that ``t2.labels[perm[n]] == t1.labels[n]``.
    In other words, it is the order of legs on ``t2``, such that they match those on ``t1``.
    None is returned instead of a trivial permutation.
    In lax label mode, we want to match the legs "by order" and hence always return None.
    """
    if config.strict_labels:
        if _no_userdefined_labels(t1) and _no_userdefined_labels(t2):
            return None
        if not (t1.is_fully_labelled and t2.is_fully_labelled):
            raise ValueError('In strict label mode, labelled tensors must be *fully* labelled.')
        if t1.shape._labels == t2.shape._labels:
            return None
        return t2.get_leg_idcs(t1.labels)
    else:
        return None


T = TypeVar('T')


def _parse_idcs(idcs: T | Sequence[T | EllipsisType], length: int, fill: T = slice(None, None, None)
                ) -> list[T]:
    """Parse a single index or sequence of indices to a list of given length by replacing Ellipsis
    (``...``) and missing entries at the back with `fill`.

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
    """A helper class that redirects __getitem__ and __setitem__ to an Tensor.

    See :meth:`~tenpy.linalg.tensors.Tensor.with_legs`.
    """
    def __init__(self, tensor: Tensor, which_legs: list[int | str]):
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
    if label.startswith('(') and label.endswith(')'):
        return _combine_leg_labels([_dual_leg_label(l) for l in _split_leg_label(label)])
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def _combine_leg_labels(labels: list[str | None]) -> str:
    """the label that a combined leg should have"""
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


def _split_leg_label(label: str | None, num: int = None) -> list[str | None]:
    """undo _combine_leg_labels, i.e. recover the original labels"""
    if label is None:
        assert num is not None
        return [None] * num
    if label.startswith('(') and label.endswith(')'):
        labels = label[1:-1].split('.')
        assert num is None or len(labels) == num
        return [None if l.startswith('?') else l for l in labels]
    else:
        raise ValueError('Invalid format for a combined label')


def _no_userdefined_labels(t: Tensor) -> bool:
    """If the labels of t are consistent with a user who never specifies labels.

    This includes
    - The label ``None``
    - The labels '!' and '!*' (arising on the invariant part of a ChargedTensor)
    - Results of combining legs that have consistent labels
    """
    if all(l is None for l in t.labels):
        # the most common case
        return True
    to_check = list(zip(t.legs, t.labels))  # tuples of (leg, label) pairs to check
    while len(to_check) > 0:
        leg, label = to_check.pop(0)
        if label is None or label in ['!', '!*']:
            continue
        if isinstance(leg, ProductSpace):
            try:
                split_labels = _split_leg_label(label, len(leg.spaces))
            except (ValueError, AssertionError):
                return False
            to_check.extend(zip(leg.spaces, split_labels))
            continue
        return False
    return True


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
        labels1 = [relabel1.get(leg, leg) for leg in legs1]
    if relabel2 is None:
        labels2 = legs2
    else:
        labels2 = [relabel2.get(leg, leg) for leg in legs2]
    conflicting = [label for label in labels1 if (label is not None) and (label in labels2)]
    labels = labels1 + labels2
    if conflicting:
        loglevel = logging.WARNING if config.strict_labels else logging.DEBUG
        msg = f'Conflicting labels {", ".join(conflicting)} are dropped.'
        # stacklevel 1 is this function, 2 is the API function using it, 3 could be from the user.
        logger.log(loglevel, msg, stacklevel=3)
        labels = [None if label in conflicting else label for label in labels]
    return labels


def _get_same_labels(labels1: list[str | None], labels2: list[str | None]) -> list[str | None]:
    """Utility function that compares labels. Per pair of labels: If one is None, the other is chosen.
    If both are not None and equal, they are chosen. If both are not None, but unequal, None is chosen
    and a warning is emitted to logger.debug"""
    labels = []
    for l1, l2 in zip(labels1, labels2):
        if l1 is None:
            labels.append(l2)
        elif l2 is None or l1 == l2:
            labels.append(l1)
        else:
            logger.debug(f'Conflicting labels {l1} vs. {l2} are dropped.', stacklevel=4)
            labels.append(None)
    return labels
