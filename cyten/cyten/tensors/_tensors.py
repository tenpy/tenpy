"""See :mod:`cyten.tensors`."""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import functools
import logging
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from math import exp as math_exp
from numbers import Integral, Number
from typing import TypeVar

import numpy as np

from ..backends import TensorBackend, conventional_leg_order, get_backend, get_same_backend
from ..block_backends import Block, Dtype
from ..dummy_config import printoptions
from ..symmetries import (
    BraidingStyle,
    ElementarySpace,
    FusionTree,
    Leg,
    LegPipe,
    Sector,
    Space,
    Symmetry,
    SymmetryError,
    TensorProduct,
)
from ..tools.misc import (
    duplicate_entries,
    inverse_permutation,
    is_iterable,
    iter_common_sorted_arrays,
    rank_data,
    to_iterable,
    to_valid_idx,
)

logger = logging.getLogger(__name__)
_USE_PERMUTE_LEGS_ERR_MSG = 'Legs can not be permuted automatically. Explicitly use permute_legs()'


CONTRACT_SYMBOL = '@'
"""Reserved character to indicate contractions in :mod:`~cyten.planar` diagrams."""

LEG_SELECT_SYMBOL = ':'
"""Reserved character to select a leg of a tensor in :mod:`~cyten.planar` diagrams."""

OPEN_LEG_SYMBOL = '->'
"""Reserved character to indicate an open leg in :mod:`~cyten.planar` diagrams."""

FORBIDDEN_LEG_LABEL_CHARS = [
    ' ',
    '\t',
    '\n',  # whitespace
    CONTRACT_SYMBOL,
    LEG_SELECT_SYMBOL,
    *OPEN_LEG_SYMBOL,
]
"""List of characters that are forbidden in leg labels"""


# TENSOR CLASSES


class LabelledLegs:
    """Base class that implements handling of labelled legs."""

    def __init__(self, labels: list[str | None]):
        dup = duplicate_entries(labels, ignore=[None])
        if len(dup) > 0:
            raise ValueError(f'Duplicate leg labels: {", ".join(dup)}')
        self._labels = labels
        self.num_legs = len(labels)
        self._labelmap = {label: legnum for legnum, label in enumerate(labels) if label is not None}

    def test_sanity(self):
        """Perform sanity checks."""
        assert all(is_valid_leg_label(l) for l in self._labels)
        assert not duplicate_entries(self._labels, ignore=[None])
        assert not duplicate_entries(list(self._labelmap.values()))

    @property
    def is_fully_labelled(self) -> bool:
        return None not in self._labels

    @property
    def labels(self) -> list[str | None]:
        """The labels that refer to the :attr:`legs`.

        Thus, ``labels[:K]`` are the ``codomain_labels`` and ``labels[K:][::-1]`` are the
        ``domain_labels`` where ``K == num_codomain_legs``.
        """
        return self._labels[:]

    @labels.setter
    def labels(self, labels):
        self.set_labels(labels)

    def get_leg_idcs(self, idcs: int | str | Sequence[int | str]) -> list[int]:
        """Parse leg-idcs of leg-labels to leg-idcs (i.e. indices of :attr:`legs`)."""
        res = []
        for idx in to_iterable(idcs):
            if isinstance(idx, str):
                try:
                    idx = self._labelmap[idx]
                except KeyError:
                    msg = f'No leg with label {idx}. Labels are {self._labels}'
                    raise ValueError(msg) from None
            else:
                idx = to_valid_idx(idx, self.num_legs)
            res.append(idx)
        return res

    def has_label(self, label: str, *more: str) -> bool:
        return (label in self._labels) and all(l in self._labels for l in more)

    def labels_are(self, *labels: str) -> bool:
        """If the given labels and the :attr:`labels` are the same, up to permutation."""
        if not self.is_fully_labelled:
            return False
        if len(labels) != self.num_legs:
            return False
        # have checked same length, so comparing the unique labels via set is enough.
        return set(labels) == set(self._labels)

    def relabel(self, mapping: dict[str, str]) -> None:
        """Apply mapping to labels. In-place."""
        self.set_labels([mapping.get(l, l) for l in self._labels])
        return self

    def set_label(self, pos: int, label: str | None):
        """Set a single label at given position, in-place. Return the modified instance."""
        if label in self._labels[:pos] or label in self._labels[pos + 1 :]:
            raise ValueError('Duplicate label')
        self._labelmap.pop(self._labels[pos], None)
        self._labels[pos] = label
        self._labelmap[label] = pos
        return self

    def set_labels(self, labels: list[str | None]):
        """Set the given labels, in-place. Return the modified instance."""
        assert not duplicate_entries(labels, ignore=[None])
        assert len(labels) == self.num_legs
        self._labels = labels
        self._labelmap = {label: legnum for legnum, label in enumerate(labels) if label is not None}
        return self


class Tensor(LabelledLegs, metaclass=ABCMeta):
    """Common base class for tensors.

    TODO elaborate

    The legs of the tensor (spaces of the domain or codomain) can be referred to either via
    string labels (see :ref:`tensor_leg_labels` and the :attr:`labels` attribute) or via integer
    positional indices. Both allow you to be ignorant of the distinction between domain and codomain
    (see :ref:`tensors_as_maps`). For the integer indices, we refer to the position of a given legs
    in the :attr:`Tensor.legs`. E.g. if ``codomain == [V, W, Z]`` and ``domain == [X, Y]``,
    we have ``legs == [V, W, Z, Y.dual, X.dual]`` and indices ``1`` and ``-4`` both refer to the
    ``W`` leg in the codomain, while indices ``3`` and ``-2`` both refer to the ``X`` leg in the
    domain. Graphically, the leg indices are arranged as follows::

    |      11  10   9   8   7   6
    |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
    |      ┃          T          ┃
    |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
    |       0   1   2   3   4   5

    A similar graphical representation is available as :attr:`Tensor.ascii_diagram` and can be
    printed to stdout using :meth:`Tensor.dbg`.

    Attributes
    ----------
    codomain, domain : TensorProduct
        The domain and codomain of the tensor. See also :attr:`legs` and :ref:`tensors_as_maps`.
    backend : TensorBackend
        The backend of the tensor.
    symmetry : Symmetry
        The symmetry of the tensor.
    num_legs : int
        The total number of legs in the domain and codomain.
    dtype : Dtype
        The dtype of tensor entries. Note that a real dtype does not necessarily imply that
        the result of :meth:`to_dense_block` is real.
    shape: tuple of int
        The dimension of each of the :attr:`legs`.

    Parameters
    ----------
    codomain : TensorProduct | list[Space]
        The codomain.
    domain : TensorProduct | list[Space] | None
        The domain. ``None`` is equivalent to ``[]``, i.e. no legs in the domain.
    backend : TensorBackend
        The backend of the tensor.
    labels: list[list[str | None]] | list[str | None] | None
        Specify the labels for the legs.
        Can either give two lists, one for the codomain, one for the domain.
        Or a single flat list for all legs in the order of the :attr:`legs`,
        such that ``[codomain_labels, domain_labels]`` is equivalent
        to ``[*codomain_legs, *reversed(domain_legs)]``.
    dtype : Dtype
        The dtype of tensor entries.

    """

    _forbidden_dtypes = [Dtype.bool]

    def __init__(
        self,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None,
        backend: TensorBackend | None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None,
        dtype: Dtype,
        device: str,
    ):
        codomain, domain, backend, symmetry = self._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        #
        self.codomain = codomain
        self.domain = domain
        self.backend = backend
        self.symmetry = symmetry
        self.dtype = dtype
        self.device = device
        self.shape = tuple(sp.dim for sp in codomain.factors) + tuple(sp.dim for sp in reversed(domain.factors))
        labels = self._init_parse_labels(labels, codomain=codomain, domain=domain)
        assert len(labels) == codomain.num_factors + domain.num_factors
        LabelledLegs.__init__(self, labels=labels)

    @staticmethod
    def _init_parse_args(
        codomain: TensorProduct | list[Space], domain: TensorProduct | list[Space] | None, backend: TensorBackend | None
    ):
        """Common input parsing for ``__init__`` methods of tensor classes.

        Also checks if they are compatible.

        Returns
        -------
        codomain, domain: TensorProduct
            The codomain and domain, converted to :class:`TensorProduct` if needed.
        backend: TensorBackend
            The given backend, or the default backend compatible with `symmetry`.
        symmetry: Symmetry
            The symmetry of the domain and codomain

        """
        # Extract the symmetry from codomain or domain. Note that either may be empty, but not both.
        if isinstance(codomain, TensorProduct):
            symmetry = codomain.symmetry
        elif len(codomain) > 0:
            symmetry = codomain[0].symmetry
        elif isinstance(domain, TensorProduct):
            symmetry = domain.symmetry
        elif len(domain) > 0:
            symmetry = domain[0].symmetry
        else:
            raise ValueError('domain and codomain can not both be empty')

        # Make sure backend is compatible with symmetry
        if backend is None:
            backend = get_backend(symmetry=symmetry)
        else:
            assert backend.supports_symmetry(symmetry)

        # Bring (co-)domain to TensorProduct form
        if not isinstance(codomain, TensorProduct):
            codomain = TensorProduct(codomain, symmetry=symmetry)
        assert codomain.symmetry == symmetry
        if domain is None:
            domain = []
        if not isinstance(domain, TensorProduct):
            domain = TensorProduct(domain, symmetry=symmetry)
        assert domain.symmetry == symmetry
        return codomain, domain, backend, symmetry

    @staticmethod
    def _init_parse_labels(
        labels: Sequence[list[str | None] | None] | list[str | None] | None,
        codomain: TensorProduct,
        domain: TensorProduct,
        is_endomorphism: bool = False,
    ):
        """Parse the various allowed input formats for labels to the format of :attr:`labels`.

        Also supports a special case for input formats of endomorphisms (maps where domain
        and codomain coincide), where a flat list of labels for the codomain can be given,
        and the domain labels are auto-filled with the respective dual labels.
        """
        num_legs = codomain.num_factors + domain.num_factors
        if is_endomorphism:
            assert codomain.num_factors == domain.num_factors

        # case 1: None
        if labels is None:
            return [None] * num_legs

        # case 2: two lists, one each for codomain and domain
        if not (isinstance(labels[0], str) or labels[0] is None):
            # expect nested lists
            codomain_labels, domain_labels = labels
            if codomain_labels is None:
                if is_endomorphism and domain_labels is not None:
                    codomain_labels = [_dual_leg_label(l) for l in domain_labels]
                else:
                    codomain_labels = [None] * codomain.num_factors
            assert len(codomain_labels) == codomain.num_factors
            if domain_labels is None:
                if is_endomorphism:
                    domain_labels = [_dual_leg_label(l) for l in codomain_labels]
                else:
                    domain_labels = [None] * domain.num_factors
            assert len(domain_labels) == domain.num_factors
            return [*codomain_labels, *reversed(domain_labels)]

        # case 3a: (only if is_endomorphism) a flat list for the codomain
        if is_endomorphism and len(labels) == codomain.num_factors:
            return [*labels, *(_dual_leg_label(l) for l in reversed(labels))]

        # case 3: a flat list for the legs
        assert len(labels) == num_legs
        return labels[:]

    def test_sanity(self):
        """Perform sanity checks."""
        self.domain.test_sanity()  # this checks all legs, and recursively through pipes
        self.codomain.test_sanity()  # this checks all legs, and recursively through pipes
        assert self.dtype not in self._forbidden_dtypes
        assert all(isinstance(leg, Leg) for leg in self.domain.factors)
        assert all(isinstance(leg, Leg) for leg in self.codomain.factors)
        super().test_sanity()

    @property
    def ascii_diagram(self) -> str:
        """An ascii representation of the tensor.

        It shows the type, leg labels, leg dimensions and leg arrows.

        Examples
        --------
        Consider the following example::

            |     123   123   132   123
            |       ^     v     v     ^
            |       a     b     c     d
            |   ┏━━━┷━━━━━┷━━━━━┷━━━━━┷━━━┓
            |   ┃          TEXT           ┃
            |   ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛
            |    i     h     g     f     e
            |    ^     v     ^     ^     v
            |   42   777    11     2     3

        """
        text = {SymmetricTensor: 'Symm', ChargedTensor: 'Charged', DiagonalTensor: 'Diag', Mask: 'Mask'}.get(
            type(self), '???'
        )
        #

        DISTANCE = 5  # distance between legs in chars, i.e. number of '━' between the '┯'

        huge_dim = f'>1e{DISTANCE + 1}'  # for numbers that can not fit in DISTANCE digits
        assert len(huge_dim) <= DISTANCE
        huge_dim = huge_dim.rjust(DISTANCE)
        huge_dim_value = 10**DISTANCE
        assert len(str(huge_dim_value)) > DISTANCE
        assert len(str(huge_dim_value - 1)) <= DISTANCE

        dims = []
        for l in self.legs:
            if len(str(l.dim)) <= DISTANCE:
                dims.append(str(l.dim).rjust(DISTANCE))
                continue
            if l.dim >= huge_dim_value:
                dims.append(huge_dim)
                continue
            s = f'{l.dim:.1f}'
            if len(s) <= DISTANCE:
                dims.append(s.rjust(DISTANCE))
                continue
            s = str(int(round(l.dim, 0)))
            if len(s) <= DISTANCE:
                dims.append(s.rjust(DISTANCE))
                continue
            raise RuntimeError  # this should not happen
        codomain_dims = dims[: self.num_codomain_legs]
        domain_dims = dims[self.num_codomain_legs :][::-1]
        codomain_arrows = [l.ascii_arrow.rjust(DISTANCE) for l in self.codomain]
        domain_arrows = [l.ascii_arrow.rjust(DISTANCE) for l in self.domain]
        codomain_labels = [
            str(l).rjust(DISTANCE) if len(str(l)) <= DISTANCE else '...'.rjust(DISTANCE) for l in self.codomain_labels
        ]
        domain_labels = [
            str(l).rjust(DISTANCE) if len(str(l)) <= DISTANCE else '...'.rjust(DISTANCE) for l in self.domain_labels
        ]
        start = ' ' * (DISTANCE - 2)  # such that f'{start}┗┯' has length DISTANCE
        #
        assert DISTANCE % 2 == 1
        if self.num_codomain_legs > self.num_domain_legs:
            codomain_extra = 0
            domain_extra = ((DISTANCE + 1) // 2) * (self.num_codomain_legs - self.num_domain_legs)
        else:
            codomain_extra = ((DISTANCE + 1) // 2) * (self.num_domain_legs - self.num_codomain_legs)
            domain_extra = 0
        #
        if self.num_codomain_legs < 2 and self.num_domain_legs < 2:
            # make room for the text
            codomain_extra += 3
            domain_extra += 3
        # top border:
        if self.num_domain_legs > 0:
            top_border = ''.join(
                [
                    start,
                    '┏',
                    '━' * domain_extra,
                    (DISTANCE * '━').join(['┷'] * self.num_domain_legs),
                    '━' * domain_extra,
                    '┓',
                ]
            )
        else:
            top_border = ''.join([start, '┏', '━' * ((DISTANCE + 1) * (self.num_codomain_legs - 1) + 1), '┓'])
        # body:
        chars_in_box = len(top_border) - len(start) - 2
        front_pad = ' ' * ((chars_in_box - len(text)) // 2)
        back_pad = ' ' * (chars_in_box - len(text) - len(front_pad))
        body = ''.join([start, '┃', front_pad, text, back_pad, '┃'])
        # bottom border:
        if self.num_codomain_legs > 0:
            bottom_border = ''.join(
                [
                    start,
                    '┗',
                    '━' * codomain_extra,
                    (DISTANCE * '━').join(['┯'] * self.num_codomain_legs),
                    '━' * codomain_extra,
                    '┛',
                ]
            )
        else:
            bottom_border = ''.join([start, '┗', '━' * ((DISTANCE + 1) * (self.num_domain_legs - 1) + 1), '┛'])
        # stitch together
        return '\n'.join(
            [
                ' ' * domain_extra + ' '.join(domain_dims),
                ' ' * domain_extra + ' '.join(domain_arrows),
                ' ' * domain_extra + ' '.join(domain_labels),
                top_border,
                body,
                bottom_border,
                ' ' * codomain_extra + ' '.join(codomain_labels),
                ' ' * codomain_extra + ' '.join(codomain_arrows),
                ' ' * codomain_extra + ' '.join(codomain_dims),
            ]
        )

    @abstractmethod
    def as_SymmetricTensor(self, guarantee_copy: bool = False, warning: str = None) -> SymmetricTensor:
        """Convert to a :class:`SymmetricTensor`, if possible.

        Parameters
        ----------
        guarantee_copy : bool
            If already a SymmetricTensor, we do *not* make a copy by default.
            Set this flag to ``True`` to guarantee a copy.
        warning : str, optional
            If given, and if the conversion is non-trivial (i.e. if it was not already a
            SymmetricTensor to begin with), a warning with this text is issued.

        """
        ...

    @abstractmethod
    def copy(self, deep=True, device: str = None) -> Tensor:
        """Copy the tensor.

        Parameters
        ----------
        deep: bool
            If the copy should be deep. A shallow copy is a new instance with the same data.
        device: str, optional
            The device for the result. Per default, use the same device as `self`.

        """
        ...

    @abstractmethod
    def to_dense_block(
        self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False
    ) -> Block:
        """Convert to a dense block of the backend, if possible.

        This corresponds to "forgetting" the symmetry structure and is only possible if the
        symmetry :attr:`Symmetry.can_be_dropped`.
        The result is a backend-specific block, e.g. a numpy array if the block backend is a
        :class:`NumpyBlockBackend` or a torch Tensor if the backend is a :class:`TorchBlockBackend`.

        Parameters
        ----------
        leg_order: list of (int | str), optional
            If given, the leg of the resulting block are permuted to match this leg order.
        dtype: Dtype, optional
            If given, the result is converted to this dtype. Per default it has the :attr:`dtype`
            of the tensor.
        understood_braiding : bool
            For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the resulting
            dense block does no longer capture the braiding statistics correctly. This means that
            :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
            the dense block representation. Permuting its legs would require e.g. explicit swap
            gates. When using the result, special care needs to be taken regarding the leg order.
            To avoid this pitfall, we raise an error by default. Set this flag to ``True`` to
            disable the error. It is then your responsibility to take care of leg orders and braids.

        """
        ...

    @property
    def codomain_labels(self) -> list[str | None]:
        """The labels that refer to legs in the codomain."""
        return self._labels[: self.num_codomain_legs]

    @property
    def dagger(self) -> Tensor:
        return dagger(self)

    @property
    def domain_labels(self) -> list[str | None]:
        """The labels that refer to legs in the domain."""
        return self._labels[self.num_codomain_legs :][::-1]

    @property
    def has_pipes(self) -> bool:
        """If any of the legs is a pipe"""
        if any(isinstance(l, LegPipe) for l in self.codomain):
            return True
        if any(isinstance(l, LegPipe) for l in self.domain):
            return True
        return False

    @property
    def hc(self) -> Tensor:
        """The :func:`dagger`"""
        return dagger(self)

    @functools.cached_property
    def legs(self) -> list[Space]:
        """All legs of the tensor.

        These the spaces of the codomain, followed by the duals of the domain spaces
        *in reverse order*.
        If we permute all legs to the codomain, we would get these spaces, i.e.::

            tensor.legs == tensor.permute_legs(codomain=range(tensor.num_legs)).codomain.spaces

        See :ref:`tensors_as_maps`.
        """
        return [*self.codomain.factors, *(sp.dual for sp in reversed(self.domain.factors))]

    @abstractmethod
    def move_to_device(self, device: str):
        """Move tensor to a given device, *in place*."""
        ...

    @property
    def num_codomain_legs(self) -> int:
        """How many of the legs are in the codomain. See :ref:`tensors_as_maps`."""
        return self.codomain.num_factors

    @property
    def num_domain_legs(self) -> int:
        """How many of the legs are in the domain. See :ref:`tensors_as_maps`."""
        return self.domain.num_factors

    @property
    def num_codomain_flat_legs(self) -> int:
        """Number of flat legs in the codomain."""
        return self.codomain.num_flat_legs

    @property
    def num_domain_flat_legs(self) -> int:
        """Number of flat legs in the domain."""
        return self.domain.num_flat_legs

    @property
    def num_flat_legs(self) -> int:
        """Total number of flat legs of self."""
        return self.num_domain_flat_legs + self.num_codomain_flat_legs

    @property
    def num_parameters(self) -> int:
        """The number of free parameters for the given legs.

        This is the dimension of the space of symmetry-preserving tensors with the given legs.
        """
        assert self.domain.sector_order == 'sorted' == self.codomain.sector_order
        res = 0
        for i, j in iter_common_sorted_arrays(self.codomain.sector_decomposition, self.domain.sector_decomposition):
            res += self.codomain.multiplicities[i] * self.domain.multiplicities[j]
        return res

    @property
    def size(self) -> int:
        """The number of entries of a dense block representation of self.

        This is only defined if ``self.symmetry.can_be_dropped``.
        In that case, it is the number of entries of :func:`to_dense_block`.
        """
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(f'Tensor.size is not defined for symmetry {self.symmetry}')
        return int(self.domain.dim * self.codomain.dim)

    @property
    def T(self) -> Tensor:
        """The :func:`transpose`."""
        return transpose(self)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return linear_combination(+1, self, +1, other)
        return NotImplemented

    def __complex__(self):
        raise TypeError('complex() of a tensor is not defined. Use cyten.item() instead.')

    def __eq__(self, other):
        msg = f'{self.__class__.__name__} does not support == comparison. Use cyten.almost_equal instead.'
        raise TypeError(msg)

    def __float__(self):
        raise TypeError('float() of a tensor is not defined. Use cyten.item() instead.')

    def __getitem__(self, idx):
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(f'Can not access elements for tensor with symmetry {self.symmetry}')
        idx = to_iterable(idx)
        if len(idx) != self.num_legs:
            msg = f'Expected {self.num_legs} indices (one per leg). Got {len(idx)}'
            raise IndexError(msg)
        try:
            idx = [int(i) for i in idx]
        except TypeError:
            raise IndexError('Indices must be integers.') from None
        idx = [to_valid_idx(i, d) for i, d in zip(idx, self.shape)]
        return self._get_item(idx)

    def __matmul__(self, other):
        return compose(self, other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return scalar_multiply(other, self)
        return NotImplemented

    def __neg__(self):
        return scalar_multiply(-1, self)

    def __pos__(self):
        return self

    def __repr__(self):
        indent = printoptions.indent * ' '
        lines = [f'<{self.__class__.__name__}']
        lines.extend(self._repr_header_lines(indent=indent))
        # skipped showing data. see commit 4bdaa5c for an old implementation of showing data.
        lines.append('>')
        return '\n'.join(lines)

    def __rmul__(self, other):
        if isinstance(other, Number):
            return scalar_multiply(other, self)
        return NotImplemented

    def __setitem__(self, idx, value):
        raise TypeError('Tensors do not support item assignment.')

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return linear_combination(+1, self, -1, other)
        return NotImplemented

    def __truediv__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        try:
            inverse_other = 1.0 / other
        except Exception:
            raise ValueError('Tensor can only be divided by invertible scalars.') from None
        return scalar_multiply(inverse_other, self)

    def _as_codomain_leg(self, idx: int | str) -> Space:
        """Return the leg, as if it was moved to the codomain."""
        in_domain, co_domain_idx, _ = self._parse_leg_idx(idx)
        if in_domain:
            return self.domain[co_domain_idx].dual
        return self.codomain[co_domain_idx]

    def _as_domain_leg(self, idx: int | str) -> Space:
        """Return the leg, as if it was moved to the domain."""
        in_domain, co_domain_idx, _ = self._parse_leg_idx(idx)
        if in_domain:
            return self.domain[co_domain_idx]
        return self.codomain[co_domain_idx].dual

    def dbg(self):
        print(self.ascii_diagram)

    @abstractmethod
    def _get_item(self, idx: list[int]) -> bool | float | complex:
        """Implementation of :meth:`__getitem__`.

        Can assume we have one non-negative integer index per leg.
        """
        ...

    def _parse_leg_idx(self, which_leg: int | str) -> tuple[bool, int, int]:
        """Parse a leg index or a leg label.

        Parameters
        ----------
        idx: int | str
            An index referring to one of the :attr:`legs` *or* a label.

        Returns
        -------
        in_domain: bool
            If the leg is in the domain.
        co_domain_idx: int
            The index of the leg in the (co-)domain
        legs_idx: int
            The index of the leg in :attr:`legs`. Same as input ``idx``, except
            it is guaranteed to be in ``range(num_legs)``.

        """
        if isinstance(which_leg, str):
            idx = self._labelmap.get(which_leg, None)
            if idx is None:
                msg = f'No leg with label {which_leg}. Labels are {self._labels}'
                raise ValueError(msg)
        else:
            idx = to_valid_idx(which_leg, self.num_legs)
        in_domain = idx >= len(self.codomain)
        if in_domain:
            co_domain_idx = self.num_legs - 1 - idx
        else:
            co_domain_idx = idx
        return in_domain, co_domain_idx, idx

    def _repr_header_lines(self, indent: str) -> list[str]:
        if all(l is None for l in self._labels):
            labels_str = 'None'
        else:
            labels_str = f'{self._labels}   ;   {self.codomain_labels} <- {self.domain_labels}'
        lines = [
            f'{indent}* Device: {self.device}',
            f'{indent}* Backend: {self.backend!s}',
            f'{indent}* Symmetry: {self.symmetry!s}',
            f'{indent}* Labels: {labels_str}',
        ]
        if self.symmetry.can_be_dropped:
            codomain_dims = self.shape[: self.num_codomain_legs]
            domain_dims = tuple(reversed(self.shape[self.num_codomain_legs :]))
            lines.append(f'{indent}* Shape: {self.shape}   ;   {codomain_dims} <- {domain_dims}')
        if (not self.symmetry.can_be_dropped) or (not self.symmetry.is_abelian):
            if self.has_pipes:
                pass  # TODO should we put some info still ...?
            else:
                codomain_nums = []
                codomain_nums = tuple(np.sum(leg.multiplicities).item() for leg in self.codomain)
                domain_nums = tuple(np.sum(leg.multiplicities).item() for leg in self.domain)
                all_nums = tuple((*codomain_nums, *reversed(domain_nums)))
                lines.append(f'{indent}* Num Sectors: {all_nums}   ;   {codomain_nums} <- {domain_nums}')
        return lines

    def get_leg(self, which_leg: int | str | list[int | str]) -> Space | list[Space]:
        """Basically ``self.legs[which_leg]``, but allows labels and multiple indices."""
        if not isinstance(which_leg, (Integral, str)):
            # which_leg is a list
            return list(map(self.get_leg, which_leg))
        in_domain, co_domain_idx, _ = self._parse_leg_idx(which_leg)
        if in_domain:
            return self.domain.factors[co_domain_idx].dual
        return self.codomain.factors[co_domain_idx]

    def get_leg_co_domain(self, which_leg: int | str) -> Space:
        """Get the specified leg from the domain or codomain.

        This is the same as :meth:`get_leg` if the leg is in the codomain, and the respective
        dual if the leg is in the domain.
        """
        if not isinstance(which_leg, (Integral, str)):
            # which_leg is a list
            return list(map(self.get_leg, which_leg))
        in_domain, co_domain_idx, _ = self._parse_leg_idx(which_leg)
        if in_domain:
            return self.domain.factors[co_domain_idx]
        return self.codomain.factors[co_domain_idx]

    def set_labels(self, labels: Sequence[list[str | None] | None] | list[str | None] | None):
        """Set the given labels, in-place. Return the modified instance."""
        labels = self._init_parse_labels(labels, codomain=self.codomain, domain=self.domain)
        return LabelledLegs.set_labels(self, labels)

    def to_numpy(
        self, leg_order: list[int | str] = None, numpy_dtype=None, understood_braiding: bool = False
    ) -> np.ndarray:
        """Convert to a numpy array"""
        block = self.to_dense_block(leg_order=leg_order, understood_braiding=understood_braiding)
        return self.backend.block_backend.to_numpy(block, numpy_dtype=numpy_dtype)


class SymmetricTensor(Tensor):
    """A tensor that is symmetric, i.e. invariant under the symmetry.

    .. note ::
        The constructor is not particularly user friendly.
        Consider using the various classmethods instead.

    Parameters
    ----------
    codomain : TensorProduct | list[Space]
        The codomain.
    domain : TensorProduct | list[Space] | None
        The domain. ``None`` (the default) is equivalent to ``[]``, i.e. no legs in the domain.
    backend : TensorBackend
        The backend of the tensor.
    labels: list[list[str | None]] | list[str | None] | None
        Specify the labels for the legs.
        Can either give two lists, one for the codomain, one for the domain.
        Or a single flat list for all legs in the order of the :attr:`legs`,
        such that ``[codomain_labels, domain_labels]`` is equivalent
        to ``[*codomain_legs, *reversed(domain_legs)]``.
    dtype : Dtype
        The dtype of tensor entries.

    Attributes
    ----------
    data:
        Backend-specific data structure that contains the numerical data, i.e. the free parameters
        of tensors with the given symmetry.

    """

    def __init__(
        self,
        data,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
    ):
        codomain, domain, backend, _ = self._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        Tensor.__init__(
            self,
            codomain=codomain,
            domain=domain,
            backend=backend,
            labels=labels,
            dtype=backend.get_dtype_from_data(data),
            device=backend.get_device_from_data(data),
        )
        assert isinstance(data, self.backend.DataCls)
        self.data = data
        self.verify_dtype()

    def test_sanity(self):
        """Perform sanity checks."""
        super().test_sanity()
        assert self.dtype == self.backend.get_dtype_from_data(self.data)
        assert self.device == self.backend.get_device_from_data(self.data)
        self.backend.test_tensor_sanity(self, is_diagonal=isinstance(self, DiagonalTensor))
        self.verify_dtype()

    def verify_dtype(self):
        if self.symmetry.has_complex_topological_data and self.dtype.is_real:
            raise ValueError(f'SymmetricTensor with {self.symmetry} must have complex dtype')

    @classmethod
    def from_block_func(
        cls,
        func,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        func_kwargs: dict = None,
        shape_kw: str = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        """Initialize a :class:`SymmetricTensor` by generating its blocks from a function.

        Here "the blocks of a tensor" are the backend-specific blocks that contain the free
        parameters of the tensor in the :attr:`data`. The concrete meaning of these blocks depends
        on the backend.

        Parameters
        ----------
        func: callable
            A function with two possible signatures. If `shape_kw` is given, we expect::

                ``func(*, shape_kw: tuple[int, ...], **kwargs) -> BlockLike``

            Otherwise::

                ``func(shape: tuple[int, ...], **kwargs) -> BlockLike``

            Where ``shape`` is the shape of the block to be generate and `func_kwargs` are passed
            as ``kwargs``. The output is converted to backend-specific blocks
            via ``backend.as_block``. In particular, it may be modified in-place after that.
        codomain, domain, backend, labels
            Arguments for constructor of :class:`SymmetricTensor`.
        func_kwargs: dict, optional
            Additional keyword arguments to be passed to ``func``.
        shape_kw: str
            If given, the shape is passed to `func` as a kwarg with this keyword.
        dtype: Dtype, None
            If given, the resulting blocks from `func` are converted to this dtype.
        device: str, optional
            If given, the resulting blocks are moved to that device.
            Per default, if `func` returns backend-specific blocks, their device is used and
            otherwise the default device of the backend.

        See Also
        --------
        from_sector_block_func
            Allows the `func` to take the current coupled sectors as an argument.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)

        # wrap func to consider func_kwargs, shape_kw, dtype, device
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            # use same backend function as from_sector_block_func, so we include the coupled arg
            # but just ignore it.
            if shape_kw is None:
                block = func(shape, **func_kwargs)
            else:
                block = func(**{shape_kw: shape}, **func_kwargs)
            return backend.block_backend.as_block(block, dtype, device=device)

        data = backend.from_sector_block_func(block_func, codomain=codomain, domain=domain)
        res = cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)
        res.test_sanity()  # OPTIMIZE remove?
        return res

    @classmethod
    def from_dense_block(
        cls,
        block,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        device: str = None,
        tol: float = 1e-6,
        understood_braiding: bool = False,
    ):
        """Convert a dense block of the backend to a Tensor.

        Parameters
        ----------
        block : Block-like
            The data to be converted to a Tensor as a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`.
            This includes e.g. nested python iterables or numpy arrays.
            The order of axes should match the :attr:`Tensor.legs`, i.e. first the codomain legs,
            then the domain leg *in reverse order*.
            The block should be given in the "public" basis order of the `legs`, e.g.
            according to :attr:`ElementarySpace.sectors_of_basis`.
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
        dtype: Dtype, optional
            If given, the block is converted to that dtype and the resulting tensor will have that
            dtype. By default, we detect the dtype from the block.
        device: str, optional
            If given, the block is moved to that device. Per default, try to use the device of
            the `block`, if it is a backend-specific block, or fall back to the backends default
            device.
        understood_braiding : bool
            For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the input
            dense block does not capture the braiding statistics correctly. This means e.g. that
            :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
            the dense block representation. This means that the input dense block needs to be
            constructed in the correct leg order. To avoid this pitfall, we raise an error by
            default. Set this flag to ``True`` to disable the error. It is then your responsibility
            to take care of leg orders and braids.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)
        if not symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {symmetry}'
            raise SymmetryError(msg)
        if not symmetry.has_trivial_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of from_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        block = backend.block_backend.as_block(block, dtype=dtype, device=device)
        assert len(backend.block_backend.get_shape(block)) == codomain.num_factors + domain.num_factors
        block = backend.block_backend.apply_basis_perm(block, conventional_leg_order(codomain, domain))
        data = backend.from_dense_block(block, codomain=codomain, domain=domain, tol=tol)
        return cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)

    @classmethod
    def from_dense_block_trivial_sector(
        cls,
        vector: Block,
        space: Space,
        backend: TensorBackend | None = None,
        device: str = None,
        label: str | None = None,
    ) -> SymmetricTensor:
        """Inverse of to_dense_block_trivial_sector."""
        if backend is None:
            backend = get_backend(symmetry=space.symmetry)
        vector = backend.block_backend.as_block(vector, device=device)
        if space._basis_perm is not None:
            i = space.sector_decomposition_where(space.symmetry.trivial_sector)
            perm = rank_data(space.basis_perm[slice(*space.slices[i])])
            vector = backend.block_backend.apply_leg_permutations(vector, [perm])
        raise NotImplementedError

    @classmethod
    def from_eye(
        cls,
        co_domain: list[Space] | TensorProduct,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ) -> SymmetricTensor:
        """The identity map as a SymmetricTensor.

        Parameters
        ----------
        co_domain
            The domain *and* codomain of the resulting tensor.
        labels
            Can either specify the labels for all legs of the resulting tensor, like
            in the constructor of :class:`SymmetricTensor`.
            Alternatively, can give labels only for the codomain (one list), and the domain labels
            are constructed as their dual labels i.e. ``'p' <-> 'p*'``.
        backend: TensorBackend, optional
            The backend of the tensor.
        dtype: Dtype
            The dtype of the tensor.
        device: str
            The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
            the block backend.

        """
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=co_domain, domain=co_domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)
        labels = cls._init_parse_labels(labels, codomain=co_domain, domain=co_domain, is_endomorphism=True)
        device = backend.block_backend.as_device(device)
        data = backend.eye_data(co_domain=co_domain, dtype=dtype, device=device)
        return cls(data, codomain=co_domain, domain=co_domain, backend=backend, labels=labels)

    @classmethod
    def from_random_normal(
        cls,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        mean: SymmetricTensor | None = None,
        sigma: float = 1.0,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        r"""Generate a sample from the normal distribution.

        The probability density is

        .. math ::
            p(T) \propto \mathrm{exp}\left[
                \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
            \right]

        .. note ::
            For a complex `dtype`, the samples are taken from the complex normal distribution,
            which corresponds to sampling the real and imaginary parts independently from (real)
            normal distributions with half the variance of the complex normal distribution.

        Parameters
        ----------
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
            If `mean` is given, all of them are optional and the respective attributes of
            `mean` are used.
        dtype: Dtype
            The dtype.
        mean: SymmetricTensor, optional
            The mean of the distribution. ``None`` is equivalent to zero mean.
        sigma: float
            The standard deviation of the distribution

        """
        assert sigma > 0.0
        if mean is not None:
            if codomain is None:
                codomain = mean.codomain
            else:
                assert mean.codomain == codomain
            if domain is None:
                domain = mean.domain
            else:
                assert mean.domain == domain
            if backend is None:
                backend = mean.backend
            else:
                assert mean.backend == backend
            if labels is None:
                labels = mean.labels
            else:
                assert mean.labels == cls._init_parse_labels(labels, codomain=codomain, domain=domain)
            if dtype is None:
                dtype = mean.dtype
            else:
                assert mean.dtype == dtype
            symmetry = mean.symmetry
        else:
            if codomain is None:
                raise ValueError('Must specify the codomain if mean is not given.')
            codomain, domain, backend, symmetry = cls._init_parse_args(
                codomain=codomain, domain=domain, backend=backend
            )

        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)

        if device is None:
            if mean is None:
                device = backend.block_backend.default_device
            else:
                device = mean.backend

        data = backend.from_random_normal(codomain, domain, sigma=sigma, dtype=dtype, device=device)
        with_zero_mean = cls(data=data, codomain=codomain, domain=domain, backend=backend, labels=labels)

        if mean is not None:
            return mean + with_zero_mean
        return with_zero_mean

    @classmethod
    def from_random_uniform(
        cls,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        """Generate a tensor with uniformly random block-entries.

        The block entries, i.e. the free parameters of the tensor are drawn independently and
        uniformly. If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and
        imaginary part are drawn independently from [-1, 1].

        .. note ::
            This is not a well defined probability distribution on the space of symmetric tensors,
            since the meaning of the uniformly drawn numbers depends on both the choice of the
            basis and on the backend.

        Parameters
        ----------
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
        dtype: Dtype
            The dtype for the tensor.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)
        return cls.from_block_func(
            func=backend.block_backend.random_uniform,
            codomain=codomain,
            domain=domain,
            backend=backend,
            labels=labels,
            func_kwargs=dict(dtype=dtype, device=device),
            dtype=dtype,
        )

    @classmethod
    def from_sector_block_func(
        cls,
        func,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        func_kwargs: dict = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        """Initialize a :class:`SymmetricTensor` by generating its blocks from a function.

        Here "the blocks of a tensor" are the backend-specific blocks that contain the free
        parameters of the tensor in the :attr:`data`. The concrete meaning of these blocks depends
        on the backend.

        Unlike :meth:`from_block_func`, this classmethod supports a `func` that takes the current
        coupled sector as an argument. The tensor, as a map from its domain to its codomain is
        block-diagonal in the coupled sectors, i.e. in the ``domain.sector_decomposition``.
        Thus, the free parameters of a tensor are associated with one block of this structure,
        and thus with a given coupled sector. A value of ``coupled`` indicates that the generated
        block is (part of) the components that maps from ``coupled`` in the domain to ``coupled``
        in the codomain.

        Parameters
        ----------
        func: callable
            A function with the following signature::

                ``func(shape: tuple[int, ...], coupled: Sector, **kwargs) -> BlockLike``

            Where ``shape`` is the shape of the block to be generated, ``coupled`` is the current
            coupled sector and `func_kwargs` are passed as ``kwargs``.
            The output is converted to backend-specific blocks via ``backend.block_backend.as_block``.
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
        func_kwargs: dict, optional
            Additional keyword arguments to be passed to ``func``.
        shape_kw: str
            If given, the shape is passed to `func` as a kwarg with this keyword.
        dtype: Dtype, None
            If given, the resulting blocks from `func` are converted to this dtype.
        device: str, optional
            If given, the resulting blocks are moved to that device.
            Per default, if `func` returns backend-specific blocks, their device is used and
            otherwise the default device of the backend.

        See Also
        --------
        from_block_func

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)

        # wrap func to consider func_kwargs and dtype
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            block = func(shape, coupled, **func_kwargs)
            return backend.block_backend.as_block(block, dtype, device=device)

        data = backend.from_sector_block_func(block_func, codomain=codomain, domain=domain)
        res = cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_sector_projection(
        cls,
        co_domain: list[Space] | TensorProduct,
        sector: Sector,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        device: str = None,
    ) -> SymmetricTensor:
        """A tensor that projects onto a given coupled sector of it domain."""
        if not isinstance(co_domain, TensorProduct):
            co_domain = TensorProduct(co_domain)
        assert co_domain.symmetry.is_valid_sector(sector)
        if co_domain.sector_multiplicity(sector) == 0:
            warnings.warn('Sector does not appear. from_sector_projection yields zero')
        if backend is None:
            backend = get_backend(symmetry=co_domain.symmetry)
        dtype = cls._parse_default_dtype(dtype, symmetry=co_domain.symmetry)

        def func(shape: tuple[int, ...], coupled: Sector):
            if np.all(coupled == sector):
                return backend.block_backend.eye_block([*shape[: len(shape) // 2]], dtype=dtype, device=device)
            return backend.block_backend.zeros(shape, dtype=dtype, device=device)

        data = backend.from_sector_block_func(func, codomain=co_domain, domain=co_domain)
        res = cls(data, codomain=co_domain, domain=co_domain, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_tree_pairs(
        cls,
        trees: dict[tuple[FusionTree, FusionTree], Block],
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        """Create a tensor from a linear combination of fusion-tree splitting-tree pairs.

        Parameters
        ----------
        trees : {(FusionTree, FusionTree): (J+K)-D Block}
            Specifies the linear combination that defines the resulting tensor.
            Each entry of the dict, ``{(X, Y): coeffs}`` represents several contributions to the
            linear combination, one per entry of the block ``coeffs``.
            The contribution with prefactor ``coeffs[n1, ..., nJ, mK, ..., m1]`` (note the axis order!)
            consists of the following steps as a map from domain to codomain::

                1. Project each leg ``k`` of the domain to a single sector, where the sector is
                   given by ``Y.uncoupled[k]`` and the degeneracy index by ``mk`` (an index to
                   the array ``coeffs``).

                2. Apply the fusion tree ``Y``.

                3. Apply the splitting tree ``X``.

                4. Apply inclusions on each leg ``j`` of the codomain, where the sector is given by
                   ``X.uncoupled[j]`` and the degeneracy index by ``nj`` (an index to the array
                   ``coeffs``).
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.

        """
        if len(trees) == 0:
            if dtype is None:
                raise ValueError('Can not infer Dtype')
            if device is None:
                raise ValueError('Can not infer device')
            return cls.from_zero(
                codomain=codomain, domain=domain, backend=backend, labels=labels, dtype=dtype, device=device
            )
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        if codomain.has_pipes or domain.has_pipes:
            raise NotImplementedError('from_tree_pairs does not support pipes (yet?)')
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)
        if device is None:
            some_block = backend.block_backend.as_block(next(iter(trees.values())))
            device = backend.block_backend.get_device(some_block)
        X_are_dual = np.array([l.is_dual for l in codomain], bool)
        Y_are_dual = np.array([l.is_dual for l in domain])
        for X, Y in trees.keys():
            assert np.all(X.coupled == Y.coupled)
            assert np.all(X.are_dual == X_are_dual)
            assert np.all(Y.are_dual == Y_are_dual)
            block = trees[X, Y]
            block = backend.block_backend.as_block(block, dtype=dtype, device=device)
            assert backend.block_backend.get_device(block) == device
            trees[X, Y] = block
        if dtype is None:
            dtype = Dtype.common(*(backend.block_backend.get_dtype(b) for b in trees.values()))
        data = backend.from_tree_pairs(trees, codomain=codomain, domain=domain, dtype=dtype, device=device)
        return cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)

    @classmethod
    def from_zero(
        cls,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str | None = None,
    ):
        """A zero tensor.

        Parameters
        ----------
        codomain, domain, backend, labels:
            Arguments, like for constructor of :class:`SymmetricTensor`.
        dtype: Dtype
            The dtype for the entries.
        device: str
            The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
            the block backend.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain=codomain, domain=domain, backend=backend)
        dtype = cls._parse_default_dtype(dtype, symmetry=symmetry)
        device = backend.block_backend.as_device(device)
        return cls(
            data=backend.zero_data(codomain=codomain, domain=domain, dtype=dtype, device=device),
            codomain=codomain,
            domain=domain,
            backend=backend,
            labels=labels,
        )

    @staticmethod
    def _parse_default_dtype(dtype: Dtype | None, symmetry: Symmetry):
        if symmetry.has_complex_topological_data:
            if dtype is None:
                dtype = Dtype.complex128
            if dtype.is_real:
                raise ValueError(f'SymmetricTensor with {symmetry} must have complex dtype')
        return dtype

    def as_SymmetricTensor(self, guarantee_copy: bool = False, warning: str = None) -> SymmetricTensor:
        if guarantee_copy:
            return self.copy()
        return self

    def copy(self, deep=True, device: str = None) -> SymmetricTensor:
        if deep:
            data = self.backend.copy_data(self, device=device)
        elif device is not None:
            data = self.backend.move_to_device(self, device=device)
        else:
            data = self.data
        return SymmetricTensor(
            data=data, codomain=self.codomain, domain=self.domain, backend=self.backend, labels=self.labels[:]
        )

    def diagonal(self, check_offdiagonal=False) -> DiagonalTensor:
        """The diagonal part as a :class:`DiagonalTensor`.

        Parameters
        ----------
        check_offdiagonal: bool
            If we should check that the off-diagonal parts vanish.

        """
        return DiagonalTensor.from_tensor(self, check_offdiagonal=check_offdiagonal)

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        return self.backend.get_element(self, idx)

    def move_to_device(self, device: str):
        self.data = self.backend.move_to_device(self, device=device)
        self.device = self.backend.block_backend.as_device(device)

    def to_dense_block(
        self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False
    ) -> Block:
        if not self.symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {self.symmetry}'
            raise SymmetryError(msg)
        if not self.symmetry.has_trivial_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of to_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        block = self.backend.to_dense_block(self)
        block = self.backend.block_backend.apply_basis_perm(block, conventional_leg_order(self), inv=True)
        if dtype is not None:
            block = self.backend.block_backend.to_dtype(block, dtype)
        if leg_order is not None:
            block = self.backend.block_backend.permute_axes(block, self.get_leg_idcs(leg_order))
        return block

    def to_dense_block_trivial_sector(self) -> Block:
        """Assumes self is a single-leg tensor and returns its components in the trivial sector.

        See Also
        --------
        from_dense_block_trivial_sector

        """
        assert self.num_legs == 1
        block = self.backend.to_dense_block_trivial_sector(self)
        assert self.num_codomain_legs == 1  # TODO assuming this for now to construct the perm. should we keep that?
        leg = self.codomain[0]
        if leg._basis_perm is not None:
            i = leg.sector_decomposition_where(self.symmetry.trivial_sector)
            perm = np.argsort(leg.basis_perm[slice(*leg.slices[i])])
            block = self.backend.block_backend.apply_leg_permutations(block, [perm])
        return block

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export SymmetricTensor to hdf5 such that it can be re-imported with from_hdf5"""
        hdf5_saver.save(self.domain, subpath + 'domain')
        hdf5_saver.save(self.codomain, subpath + 'codomain')
        hdf5_saver.save(self.backend, subpath + 'backend')
        hdf5_saver.save(self.data, subpath + 'data')
        hdf5_saver.save(self.symmetry, subpath + 'symmetry')
        hdf5_saver.save(self.dtype.to_numpy_dtype(), subpath + 'dtype')
        hdf5_saver.save(self.device, subpath + 'device')
        h5gr.attrs['num_legs'] = self.num_legs
        h5gr.attrs['shape'] = np.array(self.shape, np.intp)

        if all(i is None for i in self.labels):
            h5gr.attrs['labels'] = []

        else:
            h5gr.attrs['labels'] = self.labels

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Import SymmetricTensor from hdf5"""
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.domain = hdf5_loader.load(subpath + 'domain')
        obj.codomain = hdf5_loader.load(subpath + 'codomain')
        obj.symmetry = hdf5_loader.load(subpath + 'symmetry')
        obj.backend = get_backend(obj.symmetry, 'numpy')
        obj.data = hdf5_loader.load(subpath + 'data')
        obj.device = hdf5_loader.load(subpath + 'device')
        dt = hdf5_loader.load(subpath + 'dtype')
        obj.dtype = Dtype.from_numpy_dtype(dt)
        obj.num_legs = hdf5_loader.get_attr(h5gr, 'num_legs')
        obj.shape = hdf5_loader.get_attr(h5gr, 'shape')
        labels = hdf5_loader.get_attr(h5gr, 'labels')
        obj._labels = labels
        obj._labelmap = {label: legnum for legnum, label in enumerate(labels) if label is not None}

        return obj


class DiagonalTensor(SymmetricTensor):
    r"""Special case of a :class:`SymmetricTensor` that is diagonal in the computational basis.

    The domain and codomain of a diagonal tensor are the same and consist of a single leg::

    |        │
    |      ┏━┷━┓
    |      ┃ D ┃
    |      ┗━┯━┛
    |        │

    A diagonal tensor then is a map that is a multiple of the identity on each sector of the leg,
    i.e. it is given by :math:`\bigoplus_a \lambda_a \eye_a`, where the sum goes over sectors
    :math:`a` of the `leg` :math:`V = \bigoplus_a a`.

    This is the natural type e.g. for singular values or eigenvalue and allows
    :ref:`elementwise <diagonal_elementwise>` operations.

    Parameters
    ----------
    data
        The numerical data ("free parameters") comprising the tensor. type is backend-specific
    leg: Space
        The single leg in both the domain and codomain
    backend : TensorBackend
        The backend of the tensor.
    labels: list[list[str | None]] | list[str | None] | None
        Specify the labels for the legs.
        Can either give two lists, one for the codomain, one for the domain.
        Or a single flat list for all legs in the order of the :attr:`legs`,
        such that ``[codomain_labels, domain_labels]`` is equivalent
        to ``[*codomain_legs, *reversed(domain_legs)]``.

    .. _diagonal_elementwise:

    Elementwise Functions
    ---------------------
    A bunch of "elementwise" functions can be defined for diagonal tensors.
    If a function can be defined as a power series in ``D`` and ``D.hc``, its action can be achieved
    by applying that power series to the diagonal elements individually.
    E.g. :func:`complex_conj`, :func:`sqrt`, :func:`exp` etc.

    """

    _forbidden_dtypes = []

    def __init__(
        self,
        data,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
    ):
        if isinstance(leg, LegPipe):
            raise ValueError('DiagonalTensor is not defined on LegPipes.')
        SymmetricTensor.__init__(self, data, codomain=[leg], domain=[leg], backend=backend, labels=labels)

    def test_sanity(self):
        """Perform sanity checks."""
        super().test_sanity()
        assert self.domain == self.codomain
        assert self.domain.num_factors == 1

    def verify_dtype(self):
        # for diagonal tensors, we always allow real dtypes
        pass

    @classmethod
    def from_block_func(
        cls,
        func,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        func_kwargs: dict = None,
        shape_kw: str = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        # wrap func to consider func_kwargs, shape_kw, dtype
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            # use same backend function as from_sector_block_func, so we include the coupled arg
            # but just ignore it.
            if shape_kw is None:
                block = func(shape, **func_kwargs)
            else:
                block = func(**{shape_kw: shape}, **func_kwargs)
            return backend.block_backend.as_block(block, dtype, device=device)

        data = backend.diagonal_from_sector_block_func(block_func, co_domain=co_domain)
        res = cls(data, leg=leg, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_dense_block(
        cls,
        block: Block,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        tol: float = 1e-6,
        device: str = None,
        understood_braiding: bool = False,
    ):
        if not leg.symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {leg.symmetry}'
            raise SymmetryError(msg)
        if not leg.symmetry.has_symmetric_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of from_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        block = backend.block_backend.as_block(block, dtype=dtype, device=device)
        diag = backend.block_backend.get_diagonal(block, tol=1e-10)
        return cls.from_diag_block(diag, leg=leg, backend=backend, labels=labels, dtype=dtype, tol=tol)

    @classmethod
    def from_diag_block(
        cls,
        diag: Block,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        device: str = None,
        tol: float = 1e-6,
    ):
        """Convert a dense 1D block containing the diagonal entries to a DiagonalTensor.

        Parameters
        ----------
        diag: Block-like
            The diagonal entries as a backend-specific block or some data that can be converted
            using :meth:`BlockBackend.as_block`. This includes e.g. nested python iterables
            or numpy arrays.
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            If given, `diag` is converted to this dtype.

        See Also
        --------
        diagonal_as_block, diagonal_as_numpy
            Inverse methods that recover the `diag` entries.

        """
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        diag = backend.block_backend.as_block(diag, dtype=dtype, device=device)
        diag = backend.block_backend.apply_basis_perm(diag, [leg])
        return cls(
            data=backend.diagonal_from_block(diag, co_domain=co_domain, tol=tol),
            leg=leg,
            backend=backend,
            labels=labels,
        )

    @classmethod
    def from_eye(
        cls,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.float64,
        device: str = None,
    ):
        """The identity map as a DiagonalTensor.

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            The dtype for the entries.

        """
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        return cls.from_block_func(
            backend.block_backend.ones_block,
            leg=leg,
            backend=backend,
            labels=labels,
            func_kwargs=dict(dtype=dtype, device=device),
            dtype=dtype,
        )

    @classmethod
    def from_random_normal(
        cls,
        leg: Space,
        mean: DiagonalTensor | None = None,
        sigma: float = 1.0,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        r"""Generate a sample from the complex normal distribution.

        The probability density is

        .. math ::
            p(T) \propto \mathrm{exp}\left[
                \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
            \right]

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        mean: DiagonalTensor, optional
            The mean of the distribution. ``None`` is equivalent to zero mean.
        sigma: float
            The standard deviation of the distribution
        dtype: Dtype
            The dtype for the entries.

        """
        assert dtype.is_complex
        assert sigma > 0.0
        if mean is not None:
            assert isinstance(mean, DiagonalTensor)
            if leg is None:
                leg = mean.leg
            else:
                assert mean.leg == leg
            if backend is None:
                backend = mean.backend
            else:
                assert mean.backend == backend
            if labels is None:
                labels = mean.labels
            if dtype is None:
                dtype = mean.dtype
            else:
                assert mean.dtype == dtype
        else:
            if leg is None:
                raise ValueError('Must specify the lef if mean is not given.')
            co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)

        if device is None:
            if mean is None:
                device = backend.block_backend.default_device
            else:
                device = mean.device

        with_zero_mean = cls.from_block_func(
            backend.block_backend.random_normal,
            leg=leg,
            backend=backend,
            labels=labels,
            func_kwargs=dict(dtype=dtype, sigma=sigma),
            dtype=dtype,
        )

        if mean is not None:
            return mean + with_zero_mean
        return with_zero_mean

    @classmethod
    def from_random_uniform(
        cls,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        """Generate a tensor with uniformly random block-entries.

        The block entries, i.e. the free parameters of the tensor are drawn independently and
        uniformly. If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and
        imaginary part are drawn independently from [-1, 1].

        .. note ::
            This is not a well defined probability distribution on the space of symmetric tensors,
            since the meaning of the uniformly drawn numbers depends on both the choice of the
            basis and on the backend.

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            The dtype for the entries.

        """
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        return cls.from_block_func(
            func=backend.block_backend.random_uniform,
            leg=leg,
            backend=backend,
            labels=labels,
            func_kwargs=dict(dtype=dtype, device=device),
            dtype=dtype,
        )

    @classmethod
    def from_sector_block_func(
        cls,
        func,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        func_kwargs: dict = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        co_domain, _, backend, _ = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        # wrap func to consider func_kwargs and dtype
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            block = func(shape, coupled, **func_kwargs)
            return backend.block_backend.as_block(block, dtype, device=device)

        data = backend.diagonal_from_sector_block_func(block_func, co_domain=co_domain)
        res = cls(data, leg=leg, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_tensor(cls, tens: SymmetricTensor, tol: float | None = 1e-12) -> DiagonalTensor:
        """Create DiagonalTensor from a Tensor.

        Parameters
        ----------
        tens : :class:`Tensor`
            Must have exactly two legs. Its diagonal entries ``tens[i, i]`` are used.
        tol : float | None
            Tolerance for checking if the `tens` is actually diagonal, in the sense that any
            "off-diagonal" free parameters that should vanish are smaller than this by magnitude.
            Set to ``None`` to disable the check.

        """
        assert tens.num_legs == 2
        assert tens.domain == tens.codomain
        data = tens.backend.diagonal_tensor_from_full_tensor(tens, tol=tol)
        return cls(data=data, leg=tens.codomain.factors[0], backend=tens.backend, labels=tens.labels)

    @classmethod
    def from_zero(
        cls,
        leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        """A zero tensor.

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            The dtype for the entries.
        device: str
            The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
            the block backend.

        """
        co_domain, _, backend, symmetry = cls._init_parse_args(codomain=[leg], domain=[leg], backend=backend)
        device = backend.block_backend.as_device(device)
        return cls(
            data=backend.zero_diagonal_data(co_domain=co_domain, dtype=dtype, device=device),
            leg=leg,
            backend=backend,
            labels=labels,
        )

    @property
    def leg(self) -> Space:
        """Return the single space that makes up to domain and codomain."""
        return self.codomain.factors[0]

    def __abs__(self):
        return self._elementwise_unary(func=operator.abs, maps_zero_to_zero=True)

    def __bool__(self):
        if self.dtype == Dtype.bool and is_scalar(self):
            return bool(item(self))
        msg = 'The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()'
        raise ValueError(msg)

    def __add__(self, other):
        return self._binary_operand(other, func=operator.add, operand='+')

    def __ge__(self, other):
        return self._binary_operand(other, func=operator.ge, operand='>=')

    def __gt__(self, other):
        return self._binary_operand(other, func=operator.gt, operand='>')

    def __le__(self, other):
        return self._binary_operand(other, func=operator.le, operand='<=')

    def __lt__(self, other):
        return self._binary_operand(other, func=operator.lt, operand='<')

    def __mul__(self, other):
        return self._binary_operand(other, func=operator.mul, operand='*')

    def __pow__(self, other):
        return self._binary_operand(other, func=operator.pow, operand='**')

    def __radd__(self, other):
        return self._binary_operand(other, func=operator.add, operand='+', right=True)

    def __rmul__(self, other):
        return self._binary_operand(other, func=operator.mul, operand='*', right=True)

    def __rpow__(self, other):
        return self._binary_operand(other, func=operator.pow, operand='**', right=True)

    def __rsub__(self, other):
        return self._binary_operand(other, func=operator.sub, operand='-', right=True)

    def __rtruediv__(self, other):
        return self._binary_operand(other, func=operator.truediv, operand='/', right=True)

    def __sub__(self, other):
        return self._binary_operand(other, func=operator.sub, operand='-')

    def __truediv__(self, other):
        return self._binary_operand(other, func=operator.truediv, operand='/')

    def all(self) -> bool:
        """For a bool dtype, if all values are True. Raises for other dtypes."""
        if self.dtype != Dtype.bool:
            raise ValueError(f'all is not defined for dtype {self.dtype}')
        return self.backend.diagonal_all(self)

    def any(self) -> bool:
        """For a bool dtype, if any value is True. Raises for other dtypes."""
        if self.dtype != Dtype.bool:
            raise ValueError(f'all is not defined for dtype {self.dtype}')
        return self.backend.diagonal_any(self)

    def as_SymmetricTensor(self, guarantee_copy: bool = False, warning: str = None) -> SymmetricTensor:
        if warning is not None:
            warnings.warn(warning, UserWarning, stacklevel=2)
        return SymmetricTensor(
            data=self.backend.full_data_from_diagonal_tensor(self),
            codomain=self.codomain,
            domain=self.domain,
            backend=self.backend,
            labels=self.labels,
        )

    def _binary_operand(
        self,
        other: Number | DiagonalTensor,
        func,
        operand: str,
        return_NotImplemented: bool = False,
        right: bool = False,
    ):
        """Common implementation for the binary dunder methods ``__mul__`` etc.

        Parameters
        ----------
        other
            Either a number or a DiagonalTensor.
        func
            The function with signature
            ``func(self_block: Block, other_or_other_block: Number | Block) -> Block``
        operand
            A string representation of the operand, used in error messages
        return_NotImplemented
            Whether `NotImplemented` should be returned on a non-scalar and non-`Tensor` other.
        right
            If this is the "right" version, i.e. ``func(other, self)``.

        """
        if isinstance(other, Number):
            backend = self.backend
            if right:
                data = backend.diagonal_elementwise_unary(
                    self, func=lambda block: func(other, block), func_kwargs={}, maps_zero_to_zero=False
                )
            else:
                data = backend.diagonal_elementwise_unary(
                    self, func=lambda block: func(block, other), func_kwargs={}, maps_zero_to_zero=False
                )
            labels = self.labels
        elif isinstance(other, DiagonalTensor):
            backend = get_same_backend(self, other)
            if self.leg != other.leg:
                raise ValueError('Incompatible legs!')
            if right:
                data = backend.diagonal_elementwise_binary(
                    other, self, func=func, func_kwargs={}, partial_zero_is_zero=False
                )
            else:
                data = backend.diagonal_elementwise_binary(
                    self, other, func=func, func_kwargs={}, partial_zero_is_zero=False
                )
            labels = _get_matching_labels(self.labels, other.labels)
        elif return_NotImplemented and not isinstance(other, Tensor):
            return NotImplemented
        else:
            if right:
                msg = f'Invalid types for operand "{operand}": {type(other)} and {type(self)}'
            else:
                msg = f'Invalid types for operand "{operand}": {type(self)} and {type(other)}'
            raise TypeError(msg)
        return DiagonalTensor(data, leg=self.leg, backend=self.backend, labels=labels)

    def copy(self, deep=True, device: str = None) -> SymmetricTensor:
        if deep:
            data = self.backend.copy_data(self)
        elif device is not None:
            data = self.backend.move_to_device(self, device=device)
        else:
            data = self.data
        return DiagonalTensor(data, leg=self.leg, backend=self.backend, labels=self.labels)

    def diagonal(self) -> DiagonalTensor:
        return self

    def diagonal_as_block(self, dtype: Dtype = None) -> Block:
        if not self.symmetry.can_be_dropped:
            raise SymmetryError
        res = self.backend.diagonal_tensor_to_block(self)
        res = self.backend.block_backend.apply_basis_perm(res, [self.leg], inv=True)
        if dtype is not None:
            res = self.backend.block_backend.to_dtype(res, dtype)
        return res

    def diagonal_as_numpy(self, numpy_dtype=None) -> np.ndarray:
        block = self.diagonal_as_block(dtype=Dtype.from_numpy_dtype(numpy_dtype))
        return self.backend.block_backend.to_numpy(block, numpy_dtype=numpy_dtype)

    def elementwise_almost_equal(self, other: DiagonalTensor, rtol: float = 1e-5, atol=1e-8) -> DiagonalTensor:
        return abs(self - other) <= (atol + rtol * abs(self))

    def _elementwise_binary(
        self, other: DiagonalTensor, func, func_kwargs: dict = None, partial_zero_is_zero: bool = False
    ) -> DiagonalTensor:
        """An elementwise function acting on two diagonal tensors.

        Applies ``func(self_block: Block, other_block: Block, **func_kwargs) -> Block`` elementwise.
        Set ``partial_zero_is_zero=True`` to promise that ``func(0, any) == 0 == func(any, 0)``.
        """
        if not isinstance(other, DiagonalTensor):
            raise TypeError('Expected a DiagonalTensor')
        if not self.leg == other.leg:
            raise ValueError('Incompatible legs')
        backend = get_same_backend(self, other)
        data = backend.diagonal_elementwise_binary(
            self, other, func=func, func_kwargs=func_kwargs or {}, partial_zero_is_zero=partial_zero_is_zero
        )
        labels = _get_matching_labels(self._labels, other._labels)
        return DiagonalTensor(data, self.leg, backend=backend, labels=labels)

    def _elementwise_unary(self, func, func_kwargs: dict = None, maps_zero_to_zero: bool = False) -> DiagonalTensor:
        """An elementwise function acting on a diagonal tensor.

        Applies ``func(self_block: Block, **func_kwargs) -> Block`` elementwise.
        Set ``maps_zero_to_zero=True`` to promise that ``func(0) == 0``.
        """
        data = self.backend.diagonal_elementwise_unary(
            self, func, func_kwargs=func_kwargs or {}, maps_zero_to_zero=maps_zero_to_zero
        )
        return DiagonalTensor(data, self.leg, backend=self.backend, labels=self.labels)

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        i1, i2 = idx
        if i1 != i2:
            return self.dtype.zero_scalar
        return self.backend.get_element_diagonal(self, i1)

    def max(self):
        assert self.dtype.is_real
        return self.backend.reduce_DiagonalTensor(self, block_func=self.backend.block_backend.max, func=max)

    def min(self):
        assert self.dtype.is_real
        return self.backend.reduce_DiagonalTensor(self, block_func=self.backend.block_backend.min, func=min)

    def move_to_device(self, device: str):
        self.data = self.backend.move_to_device(self, device=device)
        self.device = self.backend.block_backend.as_device(device)

    def to_dense_block(
        self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False
    ) -> Block:
        if not self.symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {self.symmetry}'
            raise SymmetryError(msg)
        if not self.symmetry.has_trivial_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of to_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        diag = self.diagonal_as_block(dtype=dtype)
        res = self.backend.block_backend.block_from_diagonal(diag)
        if leg_order is not None:
            res = self.backend.block_backend.permute_axes(res, self.get_leg_idcs(leg_order))
        return res

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export DiagonalTensor to hdf5 such that it can be re-imported with from_hdf5"""
        super().save_hdf5(hdf5_saver, h5gr, subpath)

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Import DiagonalTensor from hdf5"""
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj = super().from_hdf5(hdf5_loader, h5gr, subpath)

        return obj


class Mask(Tensor):
    r"""A boolean mask that can be used to project or enlarge a leg.

    Masks come in two versions: projections and inclusions. A projection Mask has a single leg, the
    :attr:`large_leg` in its domain and maps it to a single leg, the :attr:`small_leg` in the
    codomain. An inclusion Mask is the dagger of this projection Mask and maps from the small leg
    in the domain to the large leg in the codomain::

        |         ║                 │
        |      ┏━━┷━━┓           ┏━━┷━━┓
        |      ┃ M_p ┃    OR     ┃ M_i ┃
        |      ┗━━┯━━┛           ┗━━┯━━┛
        |         │                 ║

    A Mask places restrictions on the basis order of the respective legs. For a projection Mask,
    the kept basis elements from the large leg need to appear in their original order in the small
    leg. Analogously, for an inclusion, the basis elements from the small leg need to be embedded
    into the large leg in their original order. This restricts
    the :attr:`~cyten.linalg.ElementarySpace.basis_perm` of the legs, see notes below.
    Most classmethods that are used to build Masks take care of this for you.

    Attributes
    ----------
    is_projection: bool
        If the Mask is a projection or inclusion map (see class docstring above).

    Parameters
    ----------
    data
        The numerical data (i.e. boolean flags) comprising the mask. type is backend-specific.
        Should have boolean dtype.
    space_in: Space
        The single space of the domain.
        This is the large leg for projections or the small leg for inclusions.
    space_out: Space
        The single space of the codomain
        This is the small leg for projections or the large leg for inclusions.
    is_projection: bool, optional
        If this Mask is a projection (from large to small) map.
        Otherwise it is in inclusion map (from small to large).
        Required if ``space_in == space_out``, since it is ambiguous in that case.
    backend: TensorBackend, optional
        The backend of the tensor.
    labels: list[list[str | None]] | list[str | None] | None
        Specify the labels for the legs.
        Can either give two lists, one for the codomain, one for the domain.
        Or a single flat list for all legs in the order of the :attr:`legs`,
        such that ``[codomain_labels, domain_labels]`` is equivalent
        to ``[*codomain_labels, *reversed(domain_labels)]``.

    Notes
    -----
    The :attr:`~cyten.linalg.ElementarySpace.basis_perm` of the legs is constrained by the
    requirements of the Mask, and in particular *depending on the data* as follows;
    The following explanation is intuitive only for a projection Mask but also applies to inclusions.
    Taking the ordered set of basis elements, permuting it by the large legs basis perm, then
    discarding some of them according to the mask data, and finally permuting the remaining
    elements back by the (inverse) small leg perm should result in a basis of the small leg,
    where the relative ordering of elements is preserved.

    In code, this means ::

        ranks = self.large_leg.basis_perm[mask_in_internal_basis][self.small_leg.inverse_basis_perm]

    In particular, the basis permutation of the small leg is uniquely determined by the
    permutation of the large leg and the mask data.

    Consider the following valid example, assuming for simplicity only one one-dim. sector ::

        large_leg_perm = [2, 4, 0, 1, 3]
        mask_in_internal_basis = [True, True, False, True, False]
        # mask_in_public_basis = [False, True, True, False, True]
        small_leg_perm = [1, 2, 0]
        small_leg_perm_inv = [2, 0, 1]

    Which maps an ordered basis as follows ::
        {e0, e1, e2, e3, e4}
        ---large_leg_perm--> {e2, e4, e0, e1, e3}
        ---mask_in_internal_basis--> {e2, e4, e1}
        ---small_leg_perm_inv--> {e1, e2, e4}

    Such that the result is ordered.

    """

    _forbidden_dtypes = [Dtype.float32, Dtype.float64, Dtype.complex64, Dtype.complex128]

    def __init__(
        self,
        data,
        space_in: ElementarySpace,
        space_out: ElementarySpace,
        is_projection: bool = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
    ):
        if isinstance(space_in, LegPipe) or isinstance(space_out, LegPipe):
            raise ValueError('Mask is not defined on LegPipes.')
        if is_projection is None:
            if space_in.dim == space_out.dim:
                raise ValueError('Need to specify is_projection for equal spaces.')
            is_projection = space_in.dim > space_out.dim
        elif is_projection is True:
            assert space_in.dim >= space_out.dim
        elif is_projection is False:
            assert space_in.dim <= space_out.dim
        else:
            raise TypeError('Invalid is_projection. Expected None or bool.')
        self.is_projection = is_projection
        if is_projection:
            assert space_out.is_subspace_of(space_in)
        else:
            assert space_in.is_subspace_of(space_out)
        assert space_out.is_dual == space_in.is_dual
        assert isinstance(space_in, ElementarySpace)
        assert isinstance(space_out, ElementarySpace)
        Tensor.__init__(
            self,
            codomain=[space_out],
            domain=[space_in],
            backend=backend,
            labels=labels,
            dtype=Dtype.bool,
            device=backend.get_device_from_data(data),
        )
        self.data = data

    def test_sanity(self):
        """Perform sanity checks."""
        super().test_sanity()
        self.backend.test_mask_sanity(self)
        assert self.codomain.num_factors == 1 == self.domain.num_factors
        assert isinstance(self.codomain.factors[0], ElementarySpace)
        assert isinstance(self.domain.factors[0], ElementarySpace)
        assert self.large_leg.is_dual == self.small_leg.is_dual
        assert self.small_leg.is_subspace_of(self.large_leg)
        assert self.dtype == Dtype.bool
        assert self.device == self.backend.get_device_from_data(self.data)

        # check consistency of the basis perm of the small leg.
        if self.large_leg._basis_perm is None:
            if self.small_leg._basis_perm is None:
                pass  # this is consistent.
            else:
                assert np.all(self.small_leg.basis_perm == np.arange(self.small_leg.dim))
        else:
            mask_in_internal_basis = self.backend.block_backend.to_numpy(self.backend.mask_to_block(self), bool)
            pi_1 = self.large_leg.basis_perm
            pi_2_inv = self.small_leg.inverse_basis_perm
            ranks = pi_1[mask_in_internal_basis][pi_2_inv]
            # check if ranks is sorted
            assert np.all(ranks[:-1] < ranks[1:])

    @property
    def large_leg(self) -> ElementarySpace:
        if self.is_projection:
            return self.domain.factors[0]
        else:
            return self.codomain.factors[0]

    @property
    def small_leg(self) -> ElementarySpace:
        if self.is_projection:
            return self.codomain.factors[0]
        else:
            return self.domain.factors[0]

    @classmethod
    def from_eye(
        cls,
        leg: ElementarySpace,
        is_projection: bool = True,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        device: str = None,
    ):
        """The identity map as a Mask, i.e. the mask that keeps all states and discards none.

        Parameters
        ----------
        leg : ElementarySpace
            The single leg for the Mask, equal to both its small and large leg.
        is_projection, backend, labels
            Arguments, like for constructor of :class:`Mask`.

        See Also
        --------
        from_zero
            The projection Mask that discards all states and keeps none.

        """
        diag = DiagonalTensor.from_eye(leg=leg, backend=backend, labels=labels, dtype=Dtype.bool, device=device)
        res = cls.from_DiagonalTensor(diag)
        if not is_projection:
            return dagger(res)
        return res

    @classmethod
    def from_block_mask(
        cls,
        block_mask: Block,
        large_leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        device: str = None,
    ):
        """Create a projection Mask from a boolean block.

        To get the related inclusion Mask, use :func:`dagger`.

        The small leg of the projection is fully determined by the large leg and by the boolean
        data. In particular, its basis permutation is such that the kept basis elements from the large
        leg appear in order.

        Parameters
        ----------
        block_mask: Block
            A boolean Block indicating for each basis element of the public basis, if it is kept.
        large_leg: Space
            The large leg, in the domain of the projection
        backend, labels
            Arguments, like for the constructor

        """
        if not large_leg.symmetry.can_be_dropped:
            raise SymmetryError
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)
        block_mask = backend.block_backend.as_block(block_mask, Dtype.bool, device=device)
        block_mask = backend.block_backend.apply_basis_perm(block_mask, [large_leg])
        data, small_leg = backend.mask_from_block(block_mask, large_leg=large_leg)
        return cls(
            data=data, space_in=large_leg, space_out=small_leg, is_projection=True, backend=backend, labels=labels
        )

    @classmethod
    def from_DiagonalTensor(cls, diag: DiagonalTensor):
        """Create a projection Mask from a boolean DiagonalTensor.

        The resulting mask keeps exactly those basis elements for which the entry of `diag` is
        ``True``. To get the related inclusion Mask, use the :func:`dagger`.

        The small leg of the projection is fully determined by the large leg and by `diag`.
        In particular, its basis permutation is such that those basis elements from the large leg
        that are kept appear in order.
        """
        assert diag.dtype == Dtype.bool
        data, small_leg = diag.backend.diagonal_to_mask(diag)
        return cls(
            data=data,
            space_in=diag.domain.factors[0],
            space_out=small_leg,
            is_projection=True,
            backend=diag.backend,
            labels=diag.labels,
        )

    @classmethod
    def from_indices(
        cls,
        indices: int | Sequence[int] | slice,
        large_leg: Space,
        backend: TensorBackend = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        device: str = None,
    ):
        """Create a projection Mask from the indices that are kept.

        To get the related inclusion Mask, use :func:`dagger`.

        The small leg of the projection is fully determined by the large leg and by the `indices`.
        In particular, its basis permutation is such that those basis elements from the large leg
        that are kept appear in order.

        Parameters
        ----------
        indices
            Valid index/indices for a 1D numpy array. The elements of the public basis of
            `large_leg` with these indices are kept by the projection.
        large_leg, backend, labels
            Same as for :meth:`Mask.__init__`.

        """
        block_mask = np.zeros(large_leg.dim, bool)
        block_mask[indices] = True
        return cls.from_block_mask(block_mask, large_leg=large_leg, backend=backend, labels=labels, device=device)

    @classmethod
    def from_random(
        cls,
        large_leg: Space,
        small_leg: Space | None = None,
        backend: TensorBackend | None = None,
        p_keep: float = 0.5,
        min_keep: int = 0,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        device: str = None,
        np_random: np.random.Generator = np.random.default_rng(),
    ):
        """Create a random projection Mask.

        To get the related inclusion Mask, use :func:`dagger`.

        Parameters
        ----------
        large_leg: Space
            The large leg, in the domain of the projection
        small_leg: Space, optional
            The small leg. If given, must be a subspace of the `large_leg` with compatible basis
            order (see notes in class docstring of :class:`Mask`).
            If ``None``, a small leg is randomly generated, according to `p_keep` and `min_keep`.
        backend, labels
            Arguments, like for the constructor
        p_keep: float, optional
            If `small_leg` is not given, the probability that any single sector is kept.
            Is ignored if `small_leg` is given, since it determines the number of kept sectors.
        min_keep: int, optional
            If `small_leg` is not given, the minimum number of sectors kept.
            Is ignored of `small_leg` is given.

        """
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)

        if not isinstance(large_leg, ElementarySpace):
            raise ValueError('large_leg must be ElementarySpace.')

        if small_leg is None:
            assert 0 <= p_keep <= 1
            diag = DiagonalTensor.from_random_uniform(
                large_leg, backend=backend, labels=labels, dtype=Dtype.float32, device=device
            )
            cutoff = 2 * p_keep - 1  # diagonal entries are uniform in [-1, 1].
            res = cls.from_DiagonalTensor(diag < cutoff)

            if np.sum(res.small_leg.multiplicities) >= min_keep:
                return res

            large_leg_sector_num = np.sum(large_leg.multiplicities)
            assert min_keep <= large_leg_sector_num, 'min_keep can not be fulfilled'
            if min_keep == large_leg_sector_num:
                return Mask.from_eye(large_leg, is_projection=True, backend=backend, labels=labels)
            # explicitly constructing the small_leg with exactly min_keep sectors kept is
            # quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
            # first, try a heuristic
            p_keep = np.ceil(1.05 * min_keep / np.sum(large_leg.multiplicities))
            res = cls.from_DiagonalTensor(diag < (2 * p_keep - 1))
            for _ in range(20):
                if np.sum(res.small_leg.multiplicities) >= min_keep:
                    return res
                p_keep = 0.5 * (p_keep + 1)  # step halfway towards 100%
                res = cls.from_DiagonalTensor(diag < (2 * p_keep - 1))
            raise RuntimeError('Could not fulfill min_keep')

        if not small_leg.is_subspace_of(large_leg):
            raise ValueError('small_leg must be a subspace of the large leg.')
        if not isinstance(small_leg, ElementarySpace):
            raise ValueError('small_leg must be ElementarySpace.')

        large_perm_trivial = large_leg._basis_perm is None or np.all(
            large_leg._basis_perm == np.arange(len(large_leg._basis_perm))
        )
        small_perm_trivial = small_leg._basis_perm is None or np.all(
            small_leg._basis_perm == np.arange(len(small_leg._basis_perm))
        )

        if (not large_perm_trivial) or (not small_perm_trivial):
            msg = 'Generating random Masks with non-trivial, fixed basis_perm is hard and hopefully never needed.'
            raise NotImplementedError(msg)

        def func(shape, coupled):
            num_keep = small_leg.sector_multiplicity(coupled)
            block = np.zeros(shape, bool)
            which = np_random.choice(shape[0], size=num_keep, replace=False)
            block[which] = True
            return block

        diag = DiagonalTensor.from_sector_block_func(
            func, leg=large_leg, backend=backend, labels=labels, dtype=Dtype.bool, device=device
        )
        res = cls.from_DiagonalTensor(diag)
        assert res.small_leg == small_leg
        return res

    @classmethod
    def from_zero(
        cls,
        large_leg: Space,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        device: str = None,
    ):
        """The zero projection Mask, that discards all states and keeps none.

        To get the related inclusion Mask, use :func:`dagger`.

        Parameters
        ----------
        large_leg: Space
            The large leg, in the domain of the projection
        backend, labels
            Arguments, like for the constructor
        device: str
            The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
            the block backend.

        See Also
        --------
        from_eye
            The projection (or inclusion) Mask that keeps all states

        """
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)
        device = backend.block_backend.as_device(device)
        data = backend.zero_mask_data(large_leg=large_leg, device=device)
        if isinstance(large_leg, ElementarySpace):
            is_dual = large_leg.is_dual
        else:
            is_dual = False
        small_leg = ElementarySpace.from_null_space(symmetry=large_leg.symmetry, is_dual=is_dual)
        return cls(data, space_in=large_leg, space_out=small_leg, is_projection=True, backend=backend, labels=labels)

    def __and__(self, other):  # ``self & other``
        return self._binary_operand(other, operator.and_, '&')

    def __bool__(self):
        msg = 'The truth value of a Mask is ambiguous. Use a.any() or a.all()'
        raise TypeError(msg)

    def __eq__(self, other):  # ``self == other``
        return self._binary_operand(other, func=operator.eq, operand='==')

    def __invert__(self):  # ``~self``
        return self._unary_operand(operator.invert)

    def __ne__(self, other):  # ``self != other``
        return self._binary_operand(other, func=operator.ne, operand='!=')

    def __or__(self, other):  # ``self | other``
        return self._binary_operand(other, func=operator.or_, operand='|')

    def __rand__(self, other):  # ``other & self``
        return self._binary_operand(other, func=operator.and_, operand='&')

    def __ror__(self, other):  # ``other | self``
        return self._binary_operand(other, func=operator.or_, operand='|')

    def __rxor__(self, other):  # ``other ^ self``
        return self._binary_operand(other, func=operator.xor, operand='^')

    def __xor__(self, other):  # ``self ^ other``
        return self._binary_operand(other, func=operator.xor, operand='^')

    def all(self) -> bool:
        """If the mask keeps all basis elements"""
        # assuming subspace, it is enough to check that the total sector number is the same.
        return np.sum(self.small_leg.multiplicities) == np.sum(self.large_leg.multiplicities)

    def any(self) -> bool:
        """If the mask keeps any basis elements"""
        return self.small_leg.dim > 0

    def as_block_mask(self) -> Block:
        res = self.backend.mask_to_block(self)
        return self.backend.block_backend.apply_basis_perm(res, [self.large_leg], inv=True)

    def as_numpy_mask(self) -> np.ndarray:
        res = self.as_block_mask()
        return self.backend.block_backend.to_numpy(res, numpy_dtype=bool)

    def as_DiagonalTensor(self, dtype=Dtype.complex128) -> DiagonalTensor:
        return DiagonalTensor(
            data=self.backend.mask_to_diagonal(self, dtype=dtype),
            leg=self.large_leg,
            backend=self.backend,
            labels=self.labels,
        )

    def as_SymmetricTensor(
        self, guarantee_copy: bool = False, warning: str = None, dtype=Dtype.complex128
    ) -> SymmetricTensor:
        if warning is not None:
            warnings.warn(warning, UserWarning, stacklevel=2)
        if not self.is_projection:
            # OPTIMIZE how hard is it to deal with inclusions in the backend?
            return dagger(dagger(self).as_SymmetricTensor())
        data = self.backend.full_data_from_mask(self, dtype)
        return SymmetricTensor(
            data, codomain=self.codomain, domain=self.domain, backend=self.backend, labels=self.labels
        )

    def _binary_operand(self, other: bool | Mask, func, operand: str, return_NotImplemented: bool = True) -> Mask:
        """Utility function for a shared implementation of binary functions.

        Parameters
        ----------
        other
            Either a bool or a Mask. If a Mask, must have same :attr:`is_projection`.
        func
            The function with signature
            ``func(self_block: Block, other_or_other_block: bool | Block) -> Block``
        operand
            A string representation of the operand, used in error messages
        return_NotImplemented
            Whether `NotImplemented` should be returned on a non-scalar and non-`Tensor` other.

        """
        # deal with non-Mask types
        if isinstance(other, bool):
            return self._unary_operand(lambda block: func(block, other))
        elif isinstance(other, Mask):
            pass
        elif return_NotImplemented and not isinstance(other, (Tensor, Number)):
            return NotImplemented
        else:
            msg = f'Invalid types for operand "{operand}": {type(self)} and {type(other)}'
            raise TypeError(msg)
        # remaining case: other is Mask
        if self.is_projection != other.is_projection:
            raise ValueError('Mismatching is_projection.')
        if not self.is_projection:
            # OPTIMIZE how hard is it to deal with inclusions in the backend?
            res_projection = dagger(self)._binary_operand(
                dagger(other), func=func, operand=operand, return_NotImplemented=return_NotImplemented
            )
            return dagger(res_projection)
        backend = get_same_backend(self, other)
        if self.domain != other.domain:
            raise ValueError('Incompatible domain.')
        data, small_leg = backend.mask_binary_operand(self, other, func)
        return Mask(
            data,
            space_in=self.large_leg,
            space_out=small_leg,
            is_projection=self.is_projection,
            backend=backend,
            labels=_get_matching_labels(self.labels, other.labels),
        )

    def copy(self, deep=True, device: str = None) -> Mask:
        if deep:
            data = self.backend.copy_data(self)
        elif device is not None:
            data = self.backend.move_to_device(self, device=device)
        else:
            data = self.data
        return Mask(
            data,
            space_in=self.large_leg,
            space_out=self.small_leg,
            is_projection=self.is_projection,
            backend=self.backend,
            labels=self.labels,
        )

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        return self.backend.get_element_mask(self, idx)

    def logical_not(self):
        """Alias for :meth:`orthogonal_complement`"""
        return self._unary_operand(operator.invert)

    def move_to_device(self, device: str):
        self.data = self.backend.move_to_device(self, device=device)
        self.device = self.backend.block_backend.as_device(device)

    def orthogonal_complement(self):
        """The "opposite" Mask, that keeps exactly what self discards and vv."""
        return self._unary_operand(operator.invert)

    def to_dense_block(
        self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False
    ) -> Block:
        if not self.symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {self.symmetry}'
            raise SymmetryError(msg)
        if not self.symmetry.has_trivial_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of to_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        # for Mask, defining via numpy is actually easier, to use numpy indexing
        numpy_dtype = None if dtype is None else dtype.to_numpy_dtype()
        as_numpy = self.to_numpy(leg_order=leg_order, numpy_dtype=numpy_dtype)
        return self.backend.block_backend.as_block(as_numpy, dtype=dtype)

    def to_numpy(
        self, leg_order: list[int | str] = None, numpy_dtype=None, understood_braiding: bool = False
    ) -> np.ndarray:
        if not self.symmetry.can_be_dropped:
            msg = f'Dense block representation is not supported for symmetry {self.symmetry}'
            raise SymmetryError(msg)
        if not self.symmetry.has_trivial_braid and not understood_braiding:
            msg = (
                'If the symmetry has non-trivial braids, dense block representations do not '
                'consistently reproduce the braiding statistics. Make sure you understand what '
                'that means (read the docstring of to_dense_block). Then you can disable '
                'this error by setting ``understood_braiding=True``.'
            )
            raise SymmetryError(msg)
        assert self.symmetry.can_be_dropped
        mask = self.as_numpy_mask()
        res = np.zeros(self.shape, numpy_dtype or bool)
        m, n = self.shape
        if self.is_projection:
            res[np.arange(m), mask] = 1  # sets the appropriate dtype. e.g. sets ``True`` for bool.
        else:
            res[mask, np.arange(n)] = 1
        if leg_order is not None:
            res = np.transpose(res, self.get_leg_idcs(leg_order))
        return res

    def _unary_operand(self, func) -> Mask:
        # operate on the respective projection
        if not self.is_projection:
            # OPTIMIZE: how hard is it to deal with inclusion Masks in the backends?
            return dagger(dagger(self)._unary_operand(func))

        data, small_leg = self.backend.mask_unary_operand(self, func)
        return Mask(
            data,
            space_in=self.large_leg,
            space_out=small_leg,
            is_projection=True,
            backend=self.backend,
            labels=self.labels,
        )

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export Mask to hdf5 such that it can be re-imported with from_hdf5"""
        hdf5_saver.save(self.domain, subpath + 'domain')
        hdf5_saver.save(self.codomain, subpath + 'codomain')
        hdf5_saver.save(self.backend, subpath + 'backend')
        hdf5_saver.save(self.data, subpath + 'data')
        hdf5_saver.save(self.symmetry, subpath + 'symmetry')
        h5gr.attrs['dtype'] = self.dtype.name
        h5gr.attrs['num_legs'] = self.num_legs
        h5gr.attrs['shape'] = np.array(self.shape, np.intp)

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Import Mask from hdf5"""
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.domain = hdf5_loader.load(subpath + 'domain')
        obj.codomain = hdf5_loader.load(subpath + 'codomain')
        obj.symmetry = hdf5_loader.load(subpath + 'symmetry')
        obj.backend = hdf5_loader.load(subpath + 'backend')
        obj.data = hdf5_loader.load(subpath + 'data')
        obj.dtype = hdf5_loader.get_attr(h5gr, 'dtype')
        obj.num_legs = hdf5_loader.get_attr(h5gr, 'num_legs')
        obj.shape = hdf5_loader.get_attr(h5gr, 'shape')
        obj.dtype = hdf5_loader.get_attr(h5gr, 'dtype')
        obj.num_legs = hdf5_loader.get_attr(h5gr, 'num_legs')
        obj.shape = hdf5_loader.get_attr(h5gr, 'shape')


class ChargedTensor(Tensor):
    r"""Tensors which are not symmetric, but carry a well defined charge.

    This captures two related but slightly different concepts.
    In both cases, the main component of a symmetric tensor is an invariant part, which
    is a :class:`SymmetricTensor`, that has an additional hidden leg, which carries the charge.
    See notes below.

    If the symmetry is a group symmetry, a particular state (i.e. a vector) on the extra leg may be
    specified. It is (generally) not symmetric, and thus this state is not a "tensor".
    The composite object of invariant part and this `charged_state` then has a well-defined
    transformation behavior under the action of the symmetry group; unlike a :class:`SymmetricTensor`,
    which is invariant under the action, it transforms under the group representation associated
    with the sectors of the additional leg.

    Alternatively, if the symmetry has symmetric braiding (which includes all group symmetries),
    we can leave the charged state unspecified and use the :class:`ChargedTensor` as a way to hide
    an additional leg from algorithms.
    We require the braiding to be symmetric, since otherwise the braiding behavior of the hidden
    leg is ambiguous.

    Parameters
    ----------
    invariant_part:
        The symmetry-invariant part. the charge leg is the its ``domain.spaces[0]``.
    charged_state: block | None
        Either ``None``, or a backend-specific block of shape ``(charge_leg.dim,)``, which specifies
        a state on the charge leg.

    Notes
    -----
    The non-trivial transformation is achieved by adding a "charged" extra leg.
    As discussed in :ref:`tensors_as_map`, we  can view symmetric tensors :math:`T` in a tensor
    space :math:`V_1 \otimes V_2 \otimes V_3` with ``legs == [V1, V2, V3]`` as a map
    :math:`\Cbb \to V_1 \otimes V_2 \otimes V_3, \alpha \mapsto \alpha T`, we define charged tensors
    with ``legs == [V1, V2, V3]`` as maps :math:`C \to V_1 \otimes V_2 \otimes V_3`.
    Note that the :attr:`charge_leg` :math:`C` is not one of the :attr:`legs`.

    The above examples had empty :attr:`domain`s. We can also consider
    ``domain == [V3, V4]`` and ``codomain == [V1, V2]``, i.e. with ``legs == [V1, V2, V4.dual, V3.dual]``.
    For a symmetric tensor, this is a map :math:`V_3 \otimes V_4 \to V_1 \otimes V_2`.
    We understand a charged tensors with these attributes to be a map
    :math:`C \otimes V_3 \otimes V_4 \to V_1 \otimes V_2`.

    We represent charged tensors as a composite object, consisting of an :attr:`invariant_part`
    and an optional :attr:`charge_state`. The invariant part is a symmetric tensor, which
    includes the :attr:`charge_leg` :math:`C` as its ``invariant_part.domain.spaces[0]``,
    and therefore has ``invariant_part.legs[-1] == C.dual``.
    Such that in the above examples we have ``invariant_part1.legs == [V1, V2, V3, C.dual]``
    and ``invariant_part2.legs == [V1, V2, V4.dual, V3.dual, C.dual]``::

    |      ┏━┓
    |      ┗┯┛
    |      C┆   6   5   4
    |      ┏┷━━━┷━━━┷━━━┷┓
    |      ┃  invariant  ┃
    |      ┗┯━━━┯━━━┯━━━┯┛
    |       0   1   2   3

    If the symmetry :attr:`Symmetry.can_be_dropped`, a specific state on the charge leg can be
    specified as a dense block. For example, consider an SU(2) symmetry and a charge leg :math:`C`,
    which is a spin-1 space. Then, a block of length ``(3,)`` can be specified selecting a state
    from the three-dimensional space. The contraction with the :attr:`charged_state` (which is not
    symmetric and hence not a tensor!) is kept track of only symbolically, i.e. all operations
    are performed on the symmetric :attr:`invariant_part` until e.g. :meth:`item` is called.

    Typical examples for charged tensors are local operators that do not conserve the charge,
    but have a well defined mapping, i.e. they map from one sector to a single other sector.
    Consider for example a U(1) symmetry which conserves the boson particle number and the
    integer sectors label that particle number. Then, the boson creation operator :math:`b^\dagger`
    can be written as a charged tensor with charge leg
    ``C == ElementarySpace(u1_sym, defining_sectors=[[+1]])``.
    Similarly, if the Sz magnetization of a spin system is conserved, the spin raising operator
    can be written only as a charged tensor.

    Charged tensors with unspecified charged state can also be used to "hide" an extra leg
    from functions and algorithms and be retrieved later. This allows, for example, to evaluate
    correlation functions of more general operators, such as e.g.
    simulating :math:`\langle S_i^x(t) S_j^x(0) \rangle` with :math:`S^z` conservation.
    The :math:`S^x` operator, when using :math:`S^z` conservation, is a `ChargedTensor` with a
    two-dimensional charge leg. But, for the correlation function, we do not actually need a state
    for that leg, we just need to contract it with the charge leg of the other :math:`S^x`, after
    having time-evolved :math:`S_j^x(0) \ket{\psi_0}`.

    """

    _CHARGE_LEG_LABEL = '!'  # canonical label for the charge leg

    def __init__(self, invariant_part: SymmetricTensor, charged_state: Block | None):
        assert invariant_part.domain.num_factors > 0, 'domain must contain at least the charge leg'
        self.charge_leg = invariant_part.domain.factors[0]
        assert invariant_part._labels[-1] == self._CHARGE_LEG_LABEL, 'incorrect label on charge leg'
        if not self.supports_symmetry(invariant_part.symmetry):
            msg = f'ChargedTensor is not well-defined for symmetry {invariant_part.symmetry}.'
            raise SymmetryError(msg)
        if charged_state is not None:
            if not invariant_part.symmetry.can_be_dropped:
                msg = f'charged_state can not be specified for symmetry {invariant_part.symmetry}'
                raise SymmetryError(msg)
            charged_state = invariant_part.backend.block_backend.as_block(
                charged_state, invariant_part.dtype, device=invariant_part.device
            )
        self.charged_state = charged_state
        self.invariant_part = invariant_part
        Tensor.__init__(
            self,
            codomain=invariant_part.codomain,
            domain=TensorProduct(
                invariant_part.domain.factors[1:],
                symmetry=invariant_part.symmetry,
            ),
            backend=invariant_part.backend,
            labels=invariant_part._labels[:-1],
            dtype=invariant_part.dtype,
            device=invariant_part.device,
        )

    def test_sanity(self):
        """Perform sanity checks."""
        super().test_sanity()
        assert self.labels == self.invariant_part.labels[:-1]
        self.invariant_part.test_sanity()
        assert self.invariant_part.device == self.device
        if self.charged_state is not None:
            self.backend.block_backend.test_block_sanity(
                self.charged_state, expect_shape=(self.charge_leg.dim,), expect_device=self.device
            )

    @staticmethod
    def _parse_inv_domain(domain: TensorProduct, charge: Space | Sector | Sequence[int]) -> tuple[TensorProduct, Space]:
        """Helper function to build the domain of the invariant part.

        Parameters
        ----------
        domain: TensorProduct
            The domain of the ChargedTensor
        charge: Space | SectorLike
            Specification for the charge_leg, either as a space or a single sector

        Returns
        -------
        inv_domain: TensorProduct
            The domain of the invariant part
        charge_leg: Space
            The charge_leg of the resulting ChargedTensor

        """
        assert isinstance(domain, TensorProduct), 'call _init_parse_args first?'
        if isinstance(charge, ElementarySpace):
            pass
        elif isinstance(charge, (Space, Leg)):
            raise TypeError
        else:
            charge = ElementarySpace(domain.symmetry, np.asarray(charge, int)[None, :])
        return domain.left_multiply(charge), charge

    @staticmethod
    def _parse_inv_labels(
        labels: Sequence[list[str | None] | None] | list[str | None] | None,
        codomain: TensorProduct,
        domain: TensorProduct,
    ):
        """Utility like :meth:`_init_parse_labels`, but also returns invariant part labels."""
        labels = ChargedTensor._init_parse_labels(labels, codomain, domain)
        inv_labels = labels + [ChargedTensor._CHARGE_LEG_LABEL]
        return labels, inv_labels

    @classmethod
    def from_block_func(
        cls,
        func,
        charge: Space | Sector,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        charged_state: Block | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        func_kwargs: dict = None,
        shape_kw: str = None,
        dtype: Dtype = None,
        device: str = None,
    ):
        """Create a charged tensor with inv_part from :meth:`SymmetricTensor.from_block_func`.

        Parameters
        ----------
        func, codomain, backend, labels, func_kwargs, shape_kw, dtype
            Like the arguments for :meth:`SymmetricTensor.from_block_func`.
        domain
            The domain of the resulting charged tensor. Its invariant part has the additional
            charged leg in the domain.
        charge: Space | Sector-like
            The charged leg. A single sector is equivalent to a space consisting of only that sector.
        charged_state: Block-like | None
            Argument for constructor of :class:`ChargedTensor`.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain, domain, backend)
        if device is None:
            if charged_state is None:
                device = backend.block_backend.default_device
            else:
                device = backend.block_backend.get_device(charged_state)
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge)
        inv = SymmetricTensor.from_block_func(
            func=func,
            codomain=codomain,
            domain=inv_domain,
            backend=backend,
            labels=labels,
            func_kwargs=func_kwargs,
            shape_kw=shape_kw,
            dtype=dtype,
            device=device,
        )
        return ChargedTensor(inv, charged_state)

    @classmethod
    def from_dense_block(
        cls,
        block: Block,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        charge: Space | Sector | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = None,
        device: str = None,
        tol: float = 1e-6,
        understood_braiding: bool = False,
    ):
        """Convert a dense block of to a ChargedTensor, if possible.

        Parameters
        ----------
        block : Block-like
            The data to be converted to a ChargedTensor as a backend-specific block or some data
            that can be converted using :meth:`BlockBackend.as_block`.
            This includes e.g. nested python iterables or numpy arrays.
            The order of axes should match the :attr:`Tensor.legs`, i.e. first the codomain legs,
            then the domain leg *in reverse order*.
            The block should be given in the "public" basis order of the `legs`, e.g.
            according to :attr:`ElementarySpace.sectors_of_basis`.
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
        dtype: Dtype, optional
            If given, the block is converted to that dtype and the resulting tensor will have that
            dtype. By default, we detect the dtype from the block.
        understood_braiding : bool
            See the same argument in :meth:`SymmetricTensor.from_dense_block`.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain, domain, backend)
        labels, inv_labels = cls._parse_inv_labels(labels, codomain, domain)
        if not symmetry.can_be_dropped:
            raise SymmetryError
        block = backend.block_backend.as_block(block, dtype, device=device)
        if charge is None:
            raise NotImplementedError
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge)
        if charge_leg.dim != 1:
            raise NotImplementedError
        inv_part = SymmetricTensor.from_dense_block(
            block=backend.block_backend.add_axis(block, -1),
            codomain=codomain,
            domain=inv_domain,
            backend=backend,
            labels=inv_labels,
            tol=tol,
            understood_braiding=understood_braiding,
        )
        return cls(inv_part, charged_state=[1])

    @classmethod
    def from_dense_block_single_sector(
        cls,
        vector: Block,
        space: Space,
        sector: Sector,
        backend: TensorBackend | None = None,
        label: str | None = None,
        device: str = None,
    ) -> ChargedTensor:
        """Given a `vector` in single `space`, represent the components in a single given `sector`.

        The resulting charged tensor has a charge lector which has the `sector`.

        See Also
        --------
        to_dense_block_single_sector

        """
        raise NotImplementedError
        if backend is None:
            backend = get_backend(symmetry=space.symmetry)
        if space.symmetry.sector_dim(sector) > 1:
            # how to handle multi-dim sectors? which dummy leg state to give?
            raise NotImplementedError
        charge_leg = ElementarySpace(space.symmetry, [sector])
        vector = backend.block_backend.as_block(vector, device=device)
        if space._basis_perm is not None:
            i = space.sector_decomposition_where(sector)
            perm = rank_data(space.basis_perm[slice(*space.slices[i])])
            vector = backend.block_backend.apply_leg_permutations(vector, [perm])
        inv_data = backend.inv_part_from_dense_block_single_sector(vector=vector, space=space, charge_leg=charge_leg)
        inv_part = SymmetricTensor(
            inv_data, codomain=[space], domain=[charge_leg], backend=backend, labels=[label, cls._CHARGE_LEG_LABEL]
        )
        return cls(inv_part, [1])

    @classmethod
    def from_invariant_part(
        cls, invariant_part: SymmetricTensor, charged_state: Block | None
    ) -> ChargedTensor | complex:
        """Like constructor, but deals with the case where invariant_part has only one leg.

        In that case, we return a scalar if the charged_state is specified and raise otherwise.
        """
        if invariant_part.num_legs == 1:
            if charged_state is None:
                raise ValueError('Can not instantiate ChargedTensor with no legs and unspecified charged_states.')
            # OPTIMIZE ?
            inv_block = invariant_part.to_dense_block(understood_braiding=True)
            return invariant_part.backend.block_backend.inner(inv_block, charged_state, do_dagger=False)
        return cls(invariant_part, charged_state)

    @classmethod
    def from_two_charge_legs(
        cls, invariant_part: SymmetricTensor, state1: Block | None, state2: Block | None
    ) -> ChargedTensor | complex:
        """Create a charged tensor from an invariant part with two charged legs."""
        assert invariant_part._labels[-1].startswith(ChargedTensor._CHARGE_LEG_LABEL)
        assert invariant_part._labels[-2].startswith(ChargedTensor._CHARGE_LEG_LABEL)
        inv_part = combine_legs(invariant_part, [-2, -1])
        inv_part.set_label(-1, cls._CHARGE_LEG_LABEL)
        if state1 is None and state2 is None:
            state = None
        elif state1 is None or state2 is None:
            raise ValueError('Must specify either both or none of the states')
        else:
            state = invariant_part.backend.state_tensor_product(state1, state2, inv_part.domain[0])
        return cls.from_invariant_part(inv_part, state)

    @classmethod
    def from_zero(
        cls,
        codomain: TensorProduct | list[Space],
        domain: TensorProduct | list[Space] | None = None,
        charge: Space | Sector | None = None,
        charged_state: Block | None = None,
        backend: TensorBackend | None = None,
        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
        dtype: Dtype = Dtype.complex128,
        device: str = None,
    ):
        """A zero tensor.

        Parameters
        ----------
        device: str, optional
            The device for the tensor. Per default, we try to use the device of the `charged_state`.
            If not available, use the default device for the backend.

        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain, domain, backend)
        if device is None:
            if charged_state is None:
                device = backend.block_backend.default_device
            else:
                device = backend.block_backend.get_device(charged_state)
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge)
        labels, inv_labels = cls._parse_inv_labels(labels, codomain, domain)
        inv_part = SymmetricTensor.from_zero(
            codomain=codomain, domain=inv_domain, backend=backend, labels=inv_labels, dtype=dtype, device=device
        )
        return ChargedTensor(inv_part, charged_state)

    @classmethod
    def supports_symmetry(cls, symmetry: Symmetry) -> bool:
        """If the :class:`ChargedTensor` concept is well defined for the `symmetry`."""
        return symmetry.has_symmetric_braid

    def as_SymmetricTensor(self, guarantee_copy: bool = False, warning: str = None) -> SymmetricTensor:
        """Convert to symmetric tensor, if possible."""
        if warning is not None:
            warnings.warn(warning, UserWarning, stacklevel=2)
        if not np.all(self.charge_leg.sector_decomposition == self.symmetry.trivial_sector[None, :]):
            raise SymmetryError('Not a symmetric tensor')
        if self.charge_leg.dim == 1:
            res = squeeze_legs(self.invariant_part, -1)
            if self.charged_state is not None:
                res = self.backend.block_backend.item(self.charged_state) * res
            return res
        if self.charged_state is None:
            raise ValueError('Can not convert to SymmetricTensor. charged_state is not defined.')
        state = SymmetricTensor.from_dense_block(
            self.charged_state,
            codomain=[self.charged_state.dual],
            backend=self.backend,
            labels=[_dual_leg_label(self._CHARGE_LEG_LABEL)],
            dtype=self.dtype,
            understood_braiding=True,
        )
        res = tdot(state, self.invariant_part, 0, -1)
        return bend_legs(res, num_codomain_legs=self.num_codomain_legs)

    def copy(self, deep=True, device: str = None) -> ChargedTensor:
        inv_part = self.invariant_part.copy(deep=deep, device=device)
        charged_state = self.charged_state
        if charged_state is not None:
            if deep:
                charged_state = self.backend.block_backend.copy_block(charged_state, device=device)
            elif device is not None:
                charged_state = self.backend.block_backend.as_block(charged_state, device=device)
        return ChargedTensor(inv_part, charged_state)

    def _get_item(self, idx: list[int]) -> bool | float | complex:
        if self.charged_state is None:
            raise IndexError('Can not index a ChargedTensor with unspecified charged_state.')
        if len(self.charged_state) > 10:
            raise NotImplementedError  # should do sth smarter...
        return sum(
            (
                self.backend.block_backend.item(a) * self.invariant_part._get_item([*idx, n])
                for n, a in enumerate(self.charged_state)
            ),
            start=self.dtype.zero_scalar,
        )

    def move_to_device(self, device: str):
        self.invariant_part.move_to_device(device)
        self.device = self.invariant_part.device
        if self.charged_state is not None:
            self.charged_state = self.backend.block_backend.as_block(self.charged_state, device=device)

    def _repr_header_lines(self, indent: str) -> list[str]:
        lines = Tensor._repr_header_lines(self, indent=indent)
        lines.append(
            f'{indent}* Charge Leg: dim={round(self.charge_leg.dim, 3)} sectors={self.charge_leg.sector_decomposition}'
        )
        start = f'{indent}* Charged State: '
        if self.charged_state is None:
            lines.append(f'{start}unspecified')
        else:
            state_lines = self.backend.block_backend._block_repr_lines(
                self.charged_state, indent=indent + '  ', max_width=printoptions.linewidth - len(start), max_lines=1
            )
            lines.append(start + state_lines[0])
        return lines

    def set_label(self, pos: int, label: str | None):
        pos = to_valid_idx(pos, self.num_legs)
        self.invariant_part.set_label(pos, label)
        return super().set_label(pos, label)

    def set_labels(self, labels: Sequence[list[str | None] | None] | list[str | None] | None):
        super().set_labels(labels)
        self.invariant_part.set_labels([*self._labels, *self._CHARGE_LEG_LABEL])
        return self

    def to_dense_block(
        self, leg_order: list[int | str] = None, dtype: Dtype = None, understood_braiding: bool = False
    ) -> Block:
        if self.charged_state is None:
            raise ValueError('charged_state not specified.')
        inv_block = self.invariant_part.to_dense_block(
            leg_order=None, dtype=dtype, understood_braiding=understood_braiding
        )
        block = self.backend.block_backend.tdot(inv_block, self.charged_state, [-1], [0])
        if dtype is not None:
            block = self.backend.block_backend.to_dtype(block, dtype)
        if leg_order is not None:
            block = self.backend.block_backend.permute_axes(block, self._as_leg_idcs(leg_order))
        return block

    def to_dense_block_single_sector(self) -> Block:
        """Return the components associated with a single sector.

        Assumes a single-leg tensor living in a single sector and returns its components within
        that sector.

        See Also
        --------
        from_dense_block_single_sector

        """
        if self.charged_state is None:
            raise ValueError('Unspecified charged_state')
        if self.num_legs > 1:
            raise ValueError('Expected a single leg')
        if self.charge_leg.num_sectors != 1 or self.charge_leg.multiplicities[0] != 1:
            raise ValueError('Not a single sector.')
        if self.charge_leg.sector_dims[0] > 1:
            raise NotImplementedError
        block = self.backend.inv_part_to_dense_block_single_sector(self.invariant_part)
        return self.backend.block_backend.item(self.charged_state) * block

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export ChargedTensor to hdf5 such that it can be re-imported with from_hdf5"""
        hdf5_saver.save(self.domain, subpath + 'domain')
        hdf5_saver.save(self.codomain, subpath + 'codomain')
        hdf5_saver.save(self.backend, subpath + 'backend')
        hdf5_saver.save(self.data, subpath + 'data')
        hdf5_saver.save(self.symmetry, subpath + 'symmetry')
        h5gr.attrs['dtype'] = self.dtype.name
        h5gr.attrs['num_legs'] = self.num_legs
        h5gr.attrs['shape'] = np.array(self.shape, np.intp)

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Import ChargedTensor from hdf5"""
        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.domain = hdf5_loader.load(subpath + 'domain')
        obj.codomain = hdf5_loader.load(subpath + 'codomain')
        obj.symmetry = hdf5_loader.load(subpath + 'symmetry')
        obj.backend = hdf5_loader.load(subpath + 'backend')
        obj.data = hdf5_loader.load(subpath + 'data')
        obj.dtype = hdf5_loader.get_attr(h5gr, 'dtype')
        obj.num_legs = hdf5_loader.get_attr(h5gr, 'num_legs')
        obj.shape = hdf5_loader.get_attr(h5gr, 'shape')
        obj.dtype = hdf5_loader.get_attr(h5gr, 'dtype')
        obj.num_legs = hdf5_loader.get_attr(h5gr, 'num_legs')
        obj.shape = hdf5_loader.get_attr(h5gr, 'shape')

        return obj


_ElementwiseType = TypeVar('_ElementwiseType', Number, DiagonalTensor)


def _elementwise_function(block_func: str, func_kwargs={}, maps_zero_to_zero=False):
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

    Notes
    -----
    Take care if the function you are defining/decorating has optional kwargs with default value,
    which are mandatory for the `block_func`. In that case, you should pass the default value
    in ``func_kwargs``, since the default value is not accessible to the wrapped function!
    See e.g. the implementation of :func:`real_if_close`.

    """

    def decorator(function):
        @functools.wraps(function)
        def wrapped(x, *args, **kwargs):
            if isinstance(x, DiagonalTensor):
                kwargs = {**func_kwargs, **kwargs}  # kwargs take precedence over func_kwargs
                func = getattr(x.backend.block_backend, block_func)
                return x._elementwise_unary(
                    lambda block: func(block, *args, **kwargs), maps_zero_to_zero=maps_zero_to_zero
                )
            elif is_scalar(x):
                return function(x, *args, **kwargs)
            raise TypeError(f'Expected DiagonalTensor or scalar. Got {type(x)}')

        return wrapped

    return decorator


# HELPERS FOR TENSOR CREATION


def eye(
    leg: ElementarySpace,
    backend: TensorBackend = None,
    labels: list[str | None] = None,
    dtype: Dtype = Dtype.float64,
    device: str = None,
    diagonal: bool = True,
) -> DiagonalTensor | SymmetricTensor:
    """The identity tensor on a given leg."""
    res = DiagonalTensor.from_eye(leg=leg, backend=backend, labels=labels, dtype=dtype, device=device)
    if diagonal:
        return res
    return res.as_SymmetricTensor()


def tensor(
    obj,
    codomain: Sequence[Leg],
    domain: Sequence[Leg] = None,
    backend: TensorBackend = None,
    labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
    dtype: Dtype = None,
    device: str = None,
    understood_braiding: bool = False,
) -> SymmetricTensor:
    """Convert object to tensor if possible."""
    if isinstance(obj, Tensor):
        copied = False
        if codomain != obj.codomain:
            raise ValueError('Mismatching codomain')
        if domain is not None and domain != obj.domain:
            raise ValueError('Mismatching domain')
        if backend is not None and backend != obj.backend:
            raise ValueError('Mismatching backend')
        if labels is not None and labels != obj._labels:
            if not copied:
                obj = obj.copy()
                copied = True
            obj.labels = labels
        if dtype is not None:
            raise ValueError('Mismatching dtype')
        if device is not None:
            raise ValueError('Mismatching device')
        return obj.as_SymmetricTensor()
    return SymmetricTensor.from_dense_block(
        obj,
        codomain,
        domain,
        backend=backend,
        labels=labels,
        dtype=dtype,
        device=device,
        understood_braiding=understood_braiding,
    )


# FUNCTIONS ON TENSORS


def add_trivial_leg(
    tens: Tensor,
    legs_pos: int = None,
    *,
    codomain_pos: int = None,
    domain_pos: int = None,
    label: str = None,
    is_dual: bool = False,
):
    """Add a trivial leg to a tensor.

    A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.

    Parameters
    ----------
    tens: Tensor
        The tensor to add a leg to. Since :class:`DiagonalTensor` and :class:`Mask` do not
        support adding legs, they will be converted to :class:`SymmetricTensor` first.
    legs_pos, codomain_pos, domain_pos: int
        The position of the new leg can be specified in three mutually exclusive ways.
        If the positional argument `leg_pos` is used, ``result.legs[leg_pos]`` will be the trivial
        leg. In most cases that unambiguously assigns it to either the domain or the codomain.
        If ambiguous (``if legs_pos == num_codomain_legs``), it is added to the codomain.
        Alternatively, it can be added to the codomain at ``codomain[codomain_pos]``
        or to the domain at ``domain_pos``.
        Note the implications for the ``is_dual`` argument!
        Per default, we use ``0``, i.e. add at ``legs[0]`` / ``codomain[0]``.
    label: str
        The label for the new leg.
    is_dual: bool
        If we add a dual (bra-like) or ket-like leg.
        Note that if `leg_pos` is given, we have ``result.legs[leg_pos].is_dual == is_dual``,
        but if `domain_pos` is given, we have ``result.domain[domain_pos].is_dual == is_dual``,
        which are mutually opposite.

    """
    res_num_legs = tens.num_legs + 1
    # parse position to format:
    #  - leg_pos: int,  0 <= leg_pos < res_num_legs
    #  - add_to_domain: bool
    #  - co_domain_pos: int, 0 <= co_domain_pos < num_[co]domain_legs
    #  - is_dual: bool, if the leg in the [co]domain should be dual
    if legs_pos is not None:
        assert codomain_pos is None and domain_pos is None
        legs_pos = to_valid_idx(legs_pos, res_num_legs)
        add_to_domain = legs_pos > tens.num_codomain_legs
        if add_to_domain:
            co_domain_pos = res_num_legs - 1 - legs_pos
        else:
            co_domain_pos = legs_pos
    elif codomain_pos is not None:
        assert legs_pos is None and domain_pos is None
        res_codomain_legs = tens.num_codomain_legs + 1
        codomain_pos = to_valid_idx(codomain_pos, res_codomain_legs)
        add_to_domain = False
        co_domain_pos = codomain_pos
        legs_pos = codomain_pos
    elif domain_pos is not None:
        assert legs_pos is None and codomain_pos is None
        res_domain_legs = tens.num_domain_legs + 1
        domain_pos = to_valid_idx(domain_pos, res_domain_legs)
        add_to_domain = True
        co_domain_pos = domain_pos
        legs_pos = res_num_legs - 1 - domain_pos
    else:
        add_to_domain = False
        co_domain_pos = 0
        legs_pos = 0

    if isinstance(tens, (DiagonalTensor, Mask)):
        msg = (
            'Converting to SymmetricTensor for add_trivial_leg. '
            'Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        tens = tens.as_SymmetricTensor(warning=msg)
    if isinstance(tens, ChargedTensor):
        if add_to_domain:
            # domain[0] is the charge leg, so we need to add 1
            inv_part = add_trivial_leg(tens.invariant_part, domain_pos=co_domain_pos + 1, label=label, is_dual=is_dual)
        else:
            inv_part = add_trivial_leg(tens.invariant_part, codomain_pos=co_domain_pos, label=label, is_dual=is_dual)
        return ChargedTensor(inv_part, charged_state=tens.charged_state)
    if not isinstance(tens, SymmetricTensor):
        raise TypeError

    new_leg = ElementarySpace.from_trivial_sector(1, symmetry=tens.symmetry, is_dual=is_dual)
    if add_to_domain:
        domain = tens.domain.insert_multiply(new_leg, pos=co_domain_pos)
        codomain = tens.codomain
    else:
        domain = tens.domain
        codomain = tens.codomain.insert_multiply(new_leg, pos=co_domain_pos)
    data = tens.backend.add_trivial_leg(
        tens,
        legs_pos=legs_pos,
        add_to_domain=add_to_domain,
        co_domain_pos=co_domain_pos,
        new_codomain=codomain,
        new_domain=domain,
    )
    return SymmetricTensor(
        data,
        codomain=codomain,
        domain=domain,
        backend=tens.backend,
        labels=[*tens.labels[:legs_pos], label, *tens.labels[legs_pos:]],
    )


@_elementwise_function(block_func='angle', maps_zero_to_zero=True)
def angle(x: _ElementwiseType) -> _ElementwiseType:
    """The angle of a complex number, :ref:`elementwise <diagonal_elementwise>`.

    The counterclockwise angle from the positive real axis on the complex plane in the
    range (-pi, pi] with a real dtype. The angle of `0.` is `0.`.
    """
    return np.angle(x)


def almost_equal(
    tensor_1: Tensor, tensor_2: Tensor, rtol: float = 1e-5, atol=1e-8, allow_different_types: bool = False
) -> bool:
    """Checks if two tensors are equal up to numerical tolerance.

    We compare the blocks, i.e. the free parameters of the tensors.
    The tensors count as almost equal if all block-entries, i.e. all their free parameters
    individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
    Note that this is a basis-dependent and backend-dependent notion of distance, which does
    not come from a norm in the strict mathematical sense.

    Parameters
    ----------
    t1, t2
        The tensors to compare
    atol, rtol
        Absolute and relative tolerance, see above.
    allow_different_types: bool
        If ``True``, we convert types, e.g. via :meth:`DiagonalTensor.as_SymmetricTensor`
        to allow comparison. If ``False``, we raise on mismatching types.

    Notes
    -----
    Unlike numpy, our definition is symmetric under exchanging

    """
    check_same_legs(tensor_1, tensor_2)
    _ = get_same_device(tensor_1, tensor_2)

    if isinstance(tensor_1, Mask):
        if isinstance(tensor_2, Mask):
            return Mask.all(tensor_1 == tensor_2)
        if isinstance(tensor_2, DiagonalTensor) and allow_different_types:
            return almost_equal(tensor_1.as_DiagonalTensor(), tensor_2, rtol=rtol, atol=atol)
        if isinstance(tensor_2, (SymmetricTensor, ChargedTensor)) and allow_different_types:
            return almost_equal(tensor_1.as_SymmetricTensor(), tensor_2, rtol=rtol, atol=atol)

    if isinstance(tensor_1, DiagonalTensor):
        if isinstance(tensor_2, Mask) and allow_different_types:
            return almost_equal(tensor_1, tensor_2.as_DiagonalTensor(), rtol=rtol, atol=atol)
        if isinstance(tensor_2, DiagonalTensor):
            return tensor_1.elementwise_almost_equal(tensor_2, rtol=rtol, atol=atol).all()
        if isinstance(tensor_2, (SymmetricTensor, ChargedTensor)) and allow_different_types:
            return almost_equal(tensor_1.as_SymmetricTensor(), tensor_2, rtol=rtol, atol=atol)

    if isinstance(tensor_1, SymmetricTensor):
        if isinstance(tensor_2, (Mask, DiagonalTensor)) and allow_different_types:
            return almost_equal(tensor_1, tensor_2.as_SymmetricTensor(), rtol=rtol, atol=atol)
        if isinstance(tensor_2, SymmetricTensor):
            return get_same_backend(tensor_1, tensor_2).almost_equal(tensor_1, tensor_2, rtol=rtol, atol=atol)
        if isinstance(tensor_2, ChargedTensor) and allow_different_types:
            try:
                t2_symm = tensor_2.as_SymmetricTensor()
                return almost_equal(tensor_1, t2_symm, rtol=rtol, atol=atol)
            except SymmetryError:
                pass
            raise NotImplementedError

    if isinstance(tensor_1, ChargedTensor):
        if isinstance(tensor_2, (Mask, DiagonalTensor)) and allow_different_types:
            return almost_equal(tensor_1, tensor_2.as_SymmetricTensor(), rtol=rtol, atol=atol)
        if isinstance(tensor_2, SymmetricTensor):
            # TODO this is not strictly correct, since definition is not symmetric...
            # we implement the mixed type comparison SymmetricTensor and ChargedTensor only once.
            # to swap the arguments we need to adjust the definition, to use abs(a2)
            return almost_equal(tensor_2, tensor_1, rtol=rtol, atol=atol)
        if isinstance(tensor_2, ChargedTensor):
            if tensor_1.charge_leg != tensor_2.charge_leg:
                raise ValueError('Mismatched charge_leg')
            if (tensor_1.charged_state is None) != (tensor_2.charged_state is None):
                raise ValueError('Mismatch: defined and undefined dummy_leg_state')
            if tensor_1.charged_state is None:
                return almost_equal(tensor_1.invariant_part, tensor_2.invariant_part, rtol=rtol, atol=atol)
            backend = get_same_backend(tensor_1, tensor_2)
            if tensor_1.charge_leg.dim == 1:
                return almost_equal(
                    backend.block_backend.item(tensor_2.charged_state) * tensor_1.invariant_part,
                    backend.block_backend.item(tensor_1.charged_state) * tensor_2.invariant_part,
                    rtol=rtol,
                    atol=atol,
                )
            raise NotImplementedError

    msg = f'Incompatible types: {tensor_1.__class__.__name__} and {tensor_2.__class__.__name__}'
    raise TypeError(msg)


def apply_mask(tensor: Tensor, mask: Mask, leg: int | str) -> Tensor:
    """Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.

    The mask must be a projection, i.e. its large leg is in its domain, at the top.
    We apply the mask via map composition::

        |                                │   │   ╭───╮   │
        |      │   │   │                 │   │  ┏┷┓  │   │           │ ┏━┷━┓ │
        |     ┏┷━━━┷━━━┷┓                │   │  ┃M┃  │   │           │ ┃M.T┃ │
        |     ┃ tensor  ┃                │   │  ┗┯┛  │   │           │ ┗━┯━┛ │
        |     ┗┯━━━┯━━━┯┛       OR       │   ╰───╯   │   │    ==    ┏┷━━━┷━━━┷┓
        |      │   │  ┏┷┓               ┏┷━━━━━━━━━━━┷━━━┷┓         ┃ tensor  ┃
        |      │   │  ┃M┃               ┃ tensor          ┃         ┗┯━━━┯━━━┯┛
        |      │   │  ┗┯┛               ┗┯━━━┯━━━┯━━━┯━━━┯┛          │   │   │
        |                                │   │   │   │   │

    where ``M.T == transpose(M)``.

    Parameters
    ----------
    tensor: Tensor
        The tensor to project
    mask: Mask
        A *projection* mask. Its large leg must be equal to the respective :attr:`Tensor.legs`.
        Note that if the leg is in the domain this means ``mask.large_leg == domain[n].dual == legs[-n]``!
    leg: int | str
        Which leg of the tensor to project

    Returns
    -------
    A masked tensor of the same type as `tensor` (exception: `DiagonalTensor`s are converted to
    `SymmetricTensor`s before masking). The leg order and labels are the same as on `tensor`.
    The masked leg is *smaller* (or equal) than before.

    See Also
    --------
    enlarge_leg, compose, partial_compose, tdot, scale_axis, apply_mask_DiagonalTensor

    """
    _ = get_same_device(tensor, mask)
    in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg)
    assert mask.is_projection
    if in_domain:
        mask = transpose(mask)
    return _compose_with_Mask(tensor, mask, leg_idx)


def apply_mask_DiagonalTensor(tensor: DiagonalTensor, mask: Mask) -> DiagonalTensor:
    """Apply a mask to *both* legs of a diagonal tensor.

    The mask must be a projection, i.e. its large leg is in the domain, at the top.
    we apply the mask via map composition::

        |     ┏━━┷━━┓
        |     ┃ M.hc┃
        |     ┗━━┯━━┛
        |     ┏━━┷━━┓
        |     ┃  D  ┃
        |     ┗━━┯━━┛
        |     ┏━━┷━━┓
        |     ┃  M  ┃
        |     ┗━━┯━━┛

    where ``M.hc == dagger(M)``

    Parameters
    ----------
    tensor: DiagonalTensor
        The diagonal tensor to project
    mask: Mask
        A *projection* mask. Its large leg must be equal to the :attr:`DiagonalTensor.leg`
        of `tensor`.

    Returns
    -------
    A masked :class:`DiagonalTensor`. Its :attr:`DiagonalTensor.leg` is the :attr:`Mask.small_leg`
    of the `mask`. Its labels are the same as those of `tensor`.

    See Also
    --------
    apply_mask

    """
    _ = get_same_device(tensor, mask)
    assert mask.is_projection
    assert mask.large_leg == tensor.leg
    backend = get_same_backend(tensor, mask)
    return DiagonalTensor(
        data=backend.apply_mask_to_DiagonalTensor(tensor, mask),
        leg=mask.small_leg,
        backend=backend,
        labels=tensor.labels,
    )


def bend_legs(tensor: Tensor, num_codomain_legs: int = None, num_domain_legs: int = None) -> Tensor:
    """Move legs between codomain and domain without changing the order of ``tensor.legs``.

    Note that legs are always bent to the right side of the tensor.
    For more general manipulations involving bends to the left side, use :func:`permute_legs`.

    For example::

        |        │   ╭───────────╮
        |        │   │   ╭───╮   │    ==    bend_legs(T, num_domain_legs=1)
        |       ┏┷━━━┷━━━┷┓  │   │
        |       ┃    T    ┃  │   │    ==    bend_legs(T, num_codomain_legs=5)
        |       ┗┯━━━┯━━━┯┛  │   │
        |        │   │   │   │   │

    or::

        |        │   │   │   │
        |       ┏┷━━━┷━━━┷┓  │
        |       ┃    T    ┃  │        ==    bend_legs(T, num_domain_legs=4)
        |       ┗┯━━━┯━━━┯┛  │
        |        │   │   ╰───╯        ==    bend_legs(T, num_codomain_legs=2)

    Parameters
    ----------
    tensor:
        The tensor to modify
    num_codomain_legs, num_domain_legs: int, optional
        The desired number of legs in the (co-)domain after the bending. Only one is required.

    See Also
    --------
    permute_legs
        More general permutations, including braids

    """
    if num_codomain_legs is None and num_domain_legs is None:
        raise ValueError
    elif num_domain_legs is None:
        num_domain_legs = tensor.num_legs - num_codomain_legs
    elif num_codomain_legs is None:
        num_codomain_legs = tensor.num_legs - num_domain_legs
    else:
        assert num_codomain_legs + num_domain_legs == tensor.num_legs
    return permute_legs(
        tensor,
        codomain=range(num_codomain_legs),
        domain=reversed(range(num_codomain_legs, tensor.num_legs)),
        bend_right=True,
    )


def check_same_legs(t1: Tensor, t2: Tensor) -> tuple[list[int], list[int]] | None:
    """Check if two tensors have the same legs.

    If there are matching labels in mismatched positions (which indicates that the leg order
    is mixed up by accident), the error message is amended accordingly on mismatched legs.
    If the legs still match regardless, a warning is issued.
    """
    if not t1.symmetry.is_same_symmetry(t2.symmetry):
        raise ValueError('Incompatible symmetries')
    incompatible_labels = False
    for n1, l1 in enumerate(t1._labels):
        n2 = t2._labelmap.get(l1, None)
        if n2 is None:
            # either l1 is None or l1 not in l2.labels
            continue
        if n2 != n1:
            incompatible_labels = True
            break
    same_legs = t1.domain == t2.domain and t1.codomain == t1.codomain
    if not same_legs:
        msg = 'Incompatible legs. '
        if incompatible_labels:
            msg += f'Should you permute_legs first? {t1.labels=}  {t2.labels=}'
        raise ValueError(msg)
    if incompatible_labels:
        logger.warning('Compatible legs with permuted labels detected. Double check your leg order!', stacklevel=3)
    # done


def combine_legs(
    tensor: Tensor,
    *which_legs: list[int | str],
    pipe_dualities: bool | list[bool] = False,
    pipes: list[LegPipe | None] = None,
    levels: list[int] | dict[str | int, int] = None,
) -> Tensor:
    """Combine (multiple) groups of legs, each to a :class:`LegPipe`.

    If the legs to be combined are contiguous to begin with (and ordered within each group),
    the combine is just a grouping of the legs::

        |       │   │          ║    │
        |       │   │   ╭───┬──╨╮   │
        |      11  10   9   8   7   6
        |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
        |      ┃          T          ┃    ==   combine_legs(T, [0, 1, 2], [4, 5], [7, 8, 9])
        |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
        |       0   1   2   3   4   5
        |       ╰╥──┴───╯   │   ╰╥──╯
        |        ║          │    ║

    Note that the conventional leg order in the domain goes right to left, such that the first
    element in the group, ``7``, is the *right*-most leg in the product, but we still have
    ``result.domain[2] == LegPipe([T.domain[2], T.domain[3], T.domain[4]])`` in left-to-right
    order.
    This is needed to make :func:`combine_legs` cooperate seamlessly with :func:`bend_legs`,
    i.e. you get the same result if you bend legs 6-9 to the codomain first and combine ``[7, 8, 9]``
    there or if you combine them in the domain and then bend leg 6 and the newly combined leg.
    Another way to see this is that we perform the product of spaces in the ``T.legs`` first,
    and then take the dual if we need the combined leg in the domain::

        result.domain[2] == result.legs[4].dual
                         == LegPipe([T.legs[7], T.legs[8], T.legs[9]]).dual
                         == LegPipe([T.domain[4].dual, T.domain[3].dual, T.domain[2].dual]).dual
                         == LegPipe([T.domain[2], T.domain[3], T.domain[4]])

    In the general case, the legs are permuted first, to match that leg order.
    The combined leg takes the position of the first of its original legs on the tensor.
    If the symmetry does not have symmetric braids, the `levels` are required to specify the
    chirality of the braids, like in :func:`permute_legs`. For example::

        |       │           │          ║
        |       │           │     ╭──┬─╨╮
        |       │     ╭─────│─────│──╯  │     ╭───╮
        |      11    10     9     8     7     6   │
        |      ┏┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷┓  │
        |      ┃               T               ┃  │    ==   combine_legs(T, [2, 6, 0], [7, 10, 8])
        |      ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛  │
        |       0     1     2     3     4     5   │
        |       ╰─────│─────│───╮ │     │     │   │
        |             │     │ ╭─│─│─────│─────│───╯
        |             │     ╰╥┴─╯ │     │     │
        |             │      ║    │     │     │

    Parameters
    ----------
    tensor:
        The tensor whose legs should be combined.
    *which_legs : list of {int | str}
        One or more groups of legs to combine.
    pipe_dualities : list of bool, optional
        Can optionally specify the :attr:`LegPipe.is_dual` attribute of each resulting pipe.
        This is an arbitrary choice for each pipe.
        The pipes are formed such that ``result.legs.[pipe_idx].is_dual == pipe_dualities[i]``.
        Defaults to all ``False``.
    pipes: list of {LegPipe | None}, optional
        For each ``group = which_legs[i]`` of legs, the resulting pipe can be passed to
        avoid recomputation. If we group to the codomain (``group[0] < tensor.num_codomain_legs``),
        we expect ``LegPipe([tensor._as_codomain_leg(i) for i in group])``.
        Otherwise we expect ``LegPipe([tensor._as_domain_leg(i) for i in reversed(group)])``.
        Note the reverse order in the latter case!
        In the intended use case, when another tensor with the same legs has already been combined,
        obtain those pipes simply via :meth:`Tensor.get_leg_co_domain`.
        It is possible to pass only some of the pipes, use ``None`` as filler.
    levels: optional
        Is ignored if the symmetry has symmetric braids. Otherwise, these levels specify the
        chirality of any possible braids induced by permuting the legs. See :func:`permute_legs`.

    Returns
    -------
    A tensor with combined legs, such that up to a :meth:`permute_legs`, the original tensor
    can be recovered with :meth:`split_legs`.
    The leg order of the result arises as follows;
    In both the domain and the codomain, the first leg of each group is replaced by the entire group,
    in order of appearance in `which_legs`. This may involve moving some legs from codomain to
    domain or vice versa. Naturally, those other legs are removed from their previous positions,
    such that the ordering of non-participating legs is preserved.
    Then, each group is replaced by the appropriate product space, either in the domain or the
    codomain.

    """
    # 1) Deal with different tensor types. Reduce everything to SymmetricTensor.
    # ==============================================================================================
    if isinstance(tensor, (DiagonalTensor, Mask)):
        msg = (
            'Converting to SymmetricTensor for combine_legs. '
            'Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        tensor = tensor.as_SymmetricTensor(warning=msg)

    which_legs = [tensor.get_leg_idcs(group) for group in which_legs]
    #
    if isinstance(tensor, ChargedTensor):
        # note: its important to parse negative integers before via tensor.get_leg_idcs, since
        #       the invariant part has an additional leg.
        if levels is not None:
            # charge leg is not combined with anything and thus does not braid.
            # so its level is irrelevant. just make sure its not a duplicate
            levels = [*levels, min(levels) - 1]
        inv_part = combine_legs(tensor.invariant_part, *which_legs, pipe_dualities=pipe_dualities, pipes=pipes)
        return ChargedTensor(inv_part, charged_state=tensor.charged_state)
    #
    # 2) permute legs such that the groups are contiguous and fully in codomain or fully in domain
    # ==============================================================================================
    N = tensor.num_legs
    J = tensor.num_codomain_legs
    to_combine = [idx for group in which_legs for idx in group]
    if duplicate_entries(to_combine):
        raise ValueError('Groups may not contain duplicates.')
    #
    # build indices for permute_legs
    codomain_groups = {group[0]: group for group in which_legs if group[0] < J}
    domain_groups = {group[0]: group for group in which_legs if group[0] >= J}
    codomain_idcs = []
    domain_idcs_reversed = []  # easier to build right-to-left.
    for n in range(N):
        if n in codomain_groups:
            codomain_idcs.extend(codomain_groups[n])
        elif n in domain_groups:
            # note: the group is given in right-to-left convention, but this is what we expect.
            domain_idcs_reversed.extend(domain_groups[n])
        elif n in to_combine:
            # n is one of the legs to be combined, but it is not the first of its group.
            pass
        elif n < J:
            codomain_idcs.append(n)
        else:
            domain_idcs_reversed.append(n)
    #
    tensor = permute_legs(tensor, codomain_idcs, domain_idcs_reversed[::-1], levels=levels)
    # leg positions have changed, so we need to update the following lists/dicts:
    inv_perm = inverse_permutation([*codomain_idcs, *domain_idcs_reversed])
    which_legs = [[inv_perm[l] for l in group] for group in which_legs]
    to_combine = [idx for group in which_legs for idx in group]
    J = tensor.num_codomain_legs
    codomain_groups = {group[0]: group for group in which_legs if group[0] < J}
    domain_groups = {group[0]: group for group in which_legs if group[0] >= J}
    #
    # 3) build new domain and codomain, labels
    # ==============================================================================================
    if pipes is None:
        pipes = [None] * len(which_legs)
    if is_iterable(pipe_dualities):
        assert len(pipe_dualities) == len(which_legs)
    else:
        pipe_dualities = [pipe_dualities] * len(which_legs)
    codomain_spaces = []
    codomain_labels = []
    domain_labels_reversed = []
    domain_spaces_reversed = []
    i = 0  # have already used pipes[:i]
    for n in range(N):
        if n in codomain_groups:
            group = codomain_groups[n]
            spaces_to_combine = tensor.codomain[group[0] : group[-1] + 1]
            combined = tensor.backend.make_pipe(
                spaces_to_combine, is_dual=pipe_dualities[i], in_domain=False, pipe=pipes[i]
            )
            pipes[i] = combined
            codomain_spaces.append(combined)
            codomain_labels.append(_combine_leg_labels(tensor.labels[group[0] : group[-1] + 1]))
            i += 1
        elif n in domain_groups:
            group = domain_groups[n]
            domain_idx1 = N - 1 - group[0]
            codomain_idx2 = N - 1 - group[-1]
            spaces_to_combine = tensor.domain[codomain_idx2 : domain_idx1 + 1]
            # Note: this is the result.domain[some_idx],  which has opposite duality from
            #       result.legs[-1-some_idx], so we need to invert pipe_dualities[i]
            combined = tensor.backend.make_pipe(
                spaces_to_combine, is_dual=not pipe_dualities[i], in_domain=True, pipe=pipes[i]
            )
            pipes[i] = combined
            domain_spaces_reversed.append(combined)
            domain_labels_reversed.append(_combine_leg_labels(tensor.labels[group[0] : group[-1] + 1]))
        elif n in to_combine:
            # n is part of a group, but not the *first* of its group
            pass
        elif n < J:
            codomain_spaces.append(tensor.codomain[n])
            codomain_labels.append(tensor.labels[n])
        else:
            domain_spaces_reversed.append(tensor.domain[N - 1 - n])
            domain_labels_reversed.append(tensor.labels[n])

    # OPTIMIZE if no bending happened, we can re-use the (co)domain.sector_decomposition.
    codomain = TensorProduct(codomain_spaces, symmetry=tensor.symmetry)
    domain = TensorProduct(domain_spaces_reversed[::-1], symmetry=tensor.symmetry)
    #
    # 4) Build the data / finish up
    # ==============================================================================================
    data = tensor.backend.combine_legs(
        tensor, leg_idcs_combine=which_legs, pipes=pipes, new_codomain=codomain, new_domain=domain
    )
    return SymmetricTensor(
        data,
        codomain=codomain,
        domain=domain,
        backend=tensor.backend,
        labels=[*codomain_labels, *domain_labels_reversed],
    )


def combine_to_matrix(
    tensor: Tensor,
    codomain: int | str | list[int | str] | None = None,
    domain: int | str | list[int | str] | None = None,
    levels: list[int] | dict[str | int, int] = None,
) -> Tensor:
    """Combine legs of a tensor into two combined LegPipes.

    The resulting tensor can be interpreted as a matrix, i.e. has two legs::



    |                    ║
    |             ╭─┬─┬──╨────╮
    |             │ ╰─│─────╮ │
    |         ╭───│───│─────│─│─╮
    |         6   5   4     │ │ │
    |      ┏━━┷━━━┷━━━┷━━┓  │ │ │
    |      ┃      T      ┃  │ │ │   =    combine_to_matrix(T, [1, 3, -1], [5, 2, 4, 0])
    |      ┗┯━━━┯━━━┯━━━┯┛  │ │ │
    |       0   1   2   3   │ │ │
    |       │   │   ╰───│───╯ │ │
    |       ╰───│───────│─────╯ │
    |           ╰──╥────┴───────╯
    |              ║

    Parameters
    ----------
    tensor: Tensor
        The tensor to act on
    codomain, domain: (list of) {int | str}, or None
        Two groups of legs. Each can be specified via leg index or leg label.
        Together, they must comprise all legs of `tensor` without duplicates.
        Only one of the two is required; the other one is determined by using "the rest" of
        the legs of `tensor`.
    levels: optional
        Is ignored if the symmetry has symmetric braids. Otherwise, these levels specify the
        chirality of any possible braids induced by permuting the legs. See :func:`permute_legs`.

    See Also
    --------
    permute_legs
        Move leg to domain / codomain without combining them there
    combine_legs
        Combine an arbitrary number of legs. Since the number of groups is arbitrary, this
        does not have the interpretation of the matrix, with one group each in domain and codomain.

    """
    res = permute_legs(tensor, codomain=codomain, domain=domain, levels=levels)
    return combine_legs(res, range(res.num_codomain_legs), range(res.num_codomain_legs, res.num_legs))


@_elementwise_function(block_func='cutoff_inverse', maps_zero_to_zero=True)
def cutoff_inverse(x: _ElementwiseType, cutoff: float = 1e-15) -> _ElementwiseType:
    """The :ref:`elementwise <diagonal_elementwise>` cutoff inverse.

    The cutoff-inverse for a number ``x`` is ``1 / x`` if ``abs(x) >= cutoff``, otherwise ``0``.
    """
    if abs(x) < cutoff:
        return 0
    return 1.0 / x


@_elementwise_function(block_func='conj', maps_zero_to_zero=True)
def complex_conj(x: _ElementwiseType) -> _ElementwiseType:
    """Complex conjugation, :ref:`elementwise <diagonal_elementwise>`."""
    return np.conj(x)


def dagger(tensor: Tensor) -> Tensor:
    r"""The hermitian conjugate tensor, a.k.a the dagger of a tensor.

    For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
    the hermitian conjugate matrix :math:`(M^\dagger)_{i,j} = \bar{M}_{j, i}` .
    For a tensor ``A: W -> V`` the dagger is a map ``dagger(A): V -> W``.
    Graphically::

        |          e   d             a   b   c
        |          │   │             │   │   │
        |       ┏━━┷━━━┷━━┓         ┏┷━━━┷━━━┷┓
        |       ┃    A    ┃         ┃dagger(A)┃
        |       ┗┯━━━┯━━━┯┛         ┗━━┯━━━┯━━┛
        |        │   │   │             │   │
        |        a   b   c             e   d

    Where ``a, b, c, d, e`` denote the legs in to (co-)domain.

    Returns
    -------
    The hermitian conjugate tensor. Its legs and labels are::

        dagger(A).codomain == A.domain
        dagger(A).domain == A.codomain
        dagger(A).legs == [leg.dual for leg in reversed(A.legs)]
        dagger(A).labels == [_dual_leg_label(l) for l in reversed(A.labels)]

    Note that the resulting :attr:`Tensor.legs` only depend on the input :attr:`Tensor.legs`, not
    on their bipartition into domain and codomain.
    For labels, we toggle a duality marker, i.e. if ``A.labels == ['a', 'b', 'c', 'd*', 'e*']``,
    then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*']``.

    """
    if isinstance(tensor, Mask):
        return Mask(
            data=tensor.backend.mask_dagger(tensor),
            space_in=tensor.codomain[0],
            space_out=tensor.domain[0],
            is_projection=not tensor.is_projection,
            backend=tensor.backend,
            labels=[_dual_leg_label(l) for l in reversed(tensor._labels)],
        )
    if isinstance(tensor, DiagonalTensor):
        if tensor.dtype == Dtype.bool:
            res = tensor.copy(deep=False)
            res.set_labels([_dual_leg_label(l) for l in reversed(tensor._labels)])
            return res
        res = complex_conj(tensor)
        res.set_labels([_dual_leg_label(l) for l in reversed(tensor._labels)])
        return res
    if isinstance(tensor, SymmetricTensor):
        return SymmetricTensor(
            data=tensor.backend.dagger(tensor),
            codomain=tensor.domain,
            domain=tensor.codomain,
            backend=tensor.backend,
            labels=[_dual_leg_label(l) for l in reversed(tensor._labels)],
        )
    if isinstance(tensor, ChargedTensor):
        inv_part = dagger(tensor.invariant_part)  # charge_leg ends up as codomain[0] and is dual.
        inv_part.set_label(0, ChargedTensor._CHARGE_LEG_LABEL)
        inv_part = move_leg(inv_part, 0, domain_pos=0, bend_right=True)
        charged_state = tensor.charged_state
        if charged_state is not None:
            charged_state = tensor.backend.block_backend.conj(charged_state)
        return ChargedTensor(inv_part, charged_state)
    raise TypeError


def compose(
    tensor1: Tensor, tensor2: Tensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None
) -> Tensor:
    r"""Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.

    Graphically::

        |        │   │   │   │
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃   tensor2   ┃
        |       ┗━━━━┯━━━┯━━━━┛
        |            │   │
        |       ┏━━━━┷━━━┷━━━━┓
        |       ┃   tensor1   ┃
        |       ┗━━┯━━━┯━━━┯━━┛
        |          │   │   │

    Returns
    -------
    The composite map :math:`T_1 \circ T_2` from ``tensor2.domain`` to ``tensor1.codomain``.

    See Also
    --------
    partial_compose, tdot, apply_mask, scale_axis

    """
    _ = get_same_device(tensor1, tensor2)
    _check_compatible_legs([tensor1.domain], [tensor2.codomain])

    if relabel1 is None:
        codomain_labels = tensor1.codomain_labels
    else:
        codomain_labels = [relabel1.get(l, l) for l in tensor1.codomain_labels]
    if relabel2 is None:
        domain_labels = tensor2.domain_labels
    else:
        domain_labels = [relabel2.get(l, l) for l in tensor2.domain_labels]
    res_labels = [codomain_labels, domain_labels]

    if isinstance(tensor1, Mask):
        return _compose_with_Mask(tensor2, tensor1, 0).set_label(0, tensor1.labels[0])
    if isinstance(tensor2, Mask):
        return _compose_with_Mask(tensor1, tensor2, -1).set_label(-1, tensor2.labels[1])

    if isinstance(tensor1, DiagonalTensor):
        return scale_axis(tensor2, tensor1, 0).set_labels(res_labels)
    if isinstance(tensor2, DiagonalTensor):
        return scale_axis(tensor1, tensor2, -1).set_labels(res_labels)

    if isinstance(tensor1, ChargedTensor) or isinstance(tensor2, ChargedTensor):
        # OPTIMIZE dedicated implementation?
        return tdot(
            tensor1,
            tensor2,
            list(reversed(range(tensor1.num_codomain_legs, tensor1.num_legs))),
            list(range(tensor2.num_codomain_legs)),
            relabel1=relabel1,
            relabel2=relabel2,
        )

    return _compose_SymmetricTensors(tensor1, tensor2, relabel1=relabel1, relabel2=relabel2)


def _compose_with_Mask(tensor: Tensor, mask: Mask, leg_idx: int) -> Tensor:
    """Compose `tensor` with a mask, preserving the leg order of `tensor`

    We expect ``tensor.codomain[leg_idx] == mask.domain[0]`` if `leg_idx` is in the codomain, or
    ``tensor.domain[co_domain_idx] == mask.codomain[0]`` otherwise.

    That is we have::

        |      │   │   │            │   │  ┏┷┓
        |     ┏┷━━━┷━━━┷┓           │   │  ┃M┃
        |     ┃ tensor  ┃           │   │  ┗┯┛
        |     ┗┯━━━┯━━━┯┛   OR     ┏┷━━━┷━━━┷┓
        |      │  ┏┷┓  │           ┃ tensor  ┃
        |      │  ┃M┃  │           ┗┯━━━┯━━━┯┛
        |      │  ┗┯┛  │            │   │   │

    Note that the resulting leg may be smaller than before (for a projection mask in the codomain
    or an inclusion mask in the domain) or larger (otherwise).

    The result hast the same leg order and labels as `tensor`.
    """
    in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
    if in_domain:
        _check_compatible_legs([tensor.domain[co_domain_idx]], [mask.codomain[0]])
    else:
        _check_compatible_legs([tensor.codomain[co_domain_idx]], [mask.domain[0]])

    # deal with other tensor types
    if isinstance(tensor, ChargedTensor):
        invariant_part = _compose_with_Mask(tensor.invariant_part, mask, leg_idx)
        return ChargedTensor(invariant_part, tensor.charged_state)
    if isinstance(tensor, Mask):
        raise NotImplementedError('tensors._compose_with_Mask not implemented for Mask')
    tensor = tensor.as_SymmetricTensor(warning='Converting to SymmetricTensor.')

    backend = get_same_backend(tensor, mask)
    if in_domain == mask.is_projection:
        data, codomain, domain = backend.mask_contract_small_leg(tensor, mask, leg_idx)
    else:
        data, codomain, domain = backend.mask_contract_large_leg(tensor, mask, leg_idx)
    return SymmetricTensor(data, codomain, domain, backend=backend, labels=tensor.labels)


def _compose_SymmetricTensors(
    tensor1: SymmetricTensor, tensor2: SymmetricTensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None
) -> SymmetricTensor:
    """Restricted case of :func:`compose` where we assume that both tensors are SymmetricTensor.

    Is used by both compose and tdot.
    """
    if tensor1.num_codomain_legs == 0 == tensor2.num_domain_legs:  # no remaining open legs
        return inner(tensor1, tensor2, do_dagger=False)

    if relabel1 is None:
        labels_codomain = tensor1.codomain_labels
    else:
        labels_codomain = [relabel1.get(l, l) for l in tensor1.codomain_labels]
    if relabel2 is None:
        labels_domain = tensor2.domain_labels
    else:
        labels_domain = [relabel2.get(l, l) for l in tensor2.domain_labels]

    # drop duplicate labels
    labels = [*labels_codomain, *reversed(labels_domain)]
    dup_counter = 0
    duplicates = duplicate_entries(labels, ignore=[None])
    for n in range(len(labels)):
        if labels[n] in duplicates:
            labels[n] = f'?{dup_counter}'
            dup_counter += 1

    backend = get_same_backend(tensor1, tensor2)
    data = backend.compose(tensor1, tensor2)
    return SymmetricTensor(data=data, codomain=tensor1.codomain, domain=tensor2.domain, backend=backend, labels=labels)


def eigh(
    tensor: Tensor,
    new_labels: str | list[str] | None,
    new_leg_dual: bool,
    sort=None,
) -> tuple[DiagonalTensor, Tensor]:
    """The eigen-decomposition of a hermitian tensor.

    A :ref:`tensor decomposition <decompositions>` ``tensor ~ V @ W @ dagger(V)`` with the following
    properties:

    - ``V`` is unitary: ``dagger(V) @ V ~ eye`` and ``V @ dagger(V) ~ eye``.
    - ``W`` is a :class:`DiagonalTensor` with the real eigenvalues of ``tensor``.

    *Assumes* that `tensor` is hermitian: ``dagger(tensor) ~ tensor``, which requires in particular
    that ``tensor.domain == tensor.codomain``. Graphically::

        |                                 │   │   │   │
        |                                ┏┷━━━┷━━━┷━━━┷┓
        |                                ┃  dagger(V)  ┃
        |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
        |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
        |       ┃   tensor    ┃    ==         ┃ W ┃
        |       ┗┯━━━┯━━━┯━━━┯┛               ┗━┯━┛
        |        │   │   │   │           ┏━━━━━━┷━━━━━━┓
        |                                ┃      V      ┃
        |                                ┗┯━━━┯━━━┯━━━┯┛
        |                                 │   │   │   │

    Parameters
    ----------
    tensor: :class:`Tensor`
        The hermitian tensor to decompose.
    new_labels: (list of) str, optional
        The labels for the new legs can be specified in the following three ways;
        Three labels ``[a, b, c]`` result in ``V.labels[-1] == a`` and ``W.labels == [b, c]``.
        Two labels ``[a, b]`` are equivalent to ``[a, b, a]``.
        A single label ``a`` is equivalent to ``[a, a*, a]``.
        The new legs are unlabelled by default.
    new_leg_dual: bool
        If the new leg should be a ket space (``False``) or bra space (``True``)
    sort: {'m>', 'm<', '>', '<', ``None``}
        How the eigenvalues should are sorted *within* each charge block.
        Defaults to ``None``, which is same as '<'. See :func:`argsort` for details.

    Returns
    -------
    W: :class:`DiagonalTensor`
        The real eigenvalues.
    V: :class:`SymmetricTensor`
        The orthonormal eigenvectors.

    """
    new_labels = to_iterable(new_labels)
    if len(new_labels) == 1:
        a = c = new_labels[0]
        b = _dual_leg_label(a)
    elif len(new_labels) == 2:
        a = c = new_labels[0]
        b = new_labels[1]
    elif len(new_labels) == 3:
        a, b, c = new_labels
    else:
        raise ValueError(f'Expected 1, 2 or 3 new_labels. Got {len(new_labels)}.')
    #
    assert tensor.domain == tensor.codomain
    if isinstance(tensor, ChargedTensor):
        # do not define decompositions for ChargedTensors.
        raise NotImplementedError
    if isinstance(tensor, DiagonalTensor):
        V = SymmetricTensor.from_eye(
            [tensor.leg],
            backend=tensor.backend,
            labels=[tensor.codomain_labels[0], a],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        W = tensor.copy().set_labels([b, c])
        return W, V
    tensor = tensor.as_SymmetricTensor()

    # If the backend requires it, combine legs first
    if not tensor.backend.can_decompose_tensors:
        tensor = combine_legs(
            tensor,
            range(tensor.num_codomain_legs),
            range(tensor.num_codomain_legs, tensor.num_legs),
            pipe_dualities=[new_leg_dual, not new_leg_dual],
        )

    # first, compute a decomposition where the new leg is a ket space
    w_data, v_data, new_leg = tensor.backend.eigh(tensor, new_leg_dual, sort=sort)
    W = DiagonalTensor(w_data, new_leg, tensor.backend, [b, c])
    V = SymmetricTensor(
        v_data, codomain=tensor.codomain, domain=[new_leg], backend=tensor.backend, labels=[tensor.codomain_labels, [a]]
    )

    # undo the combine
    if not tensor.backend.can_decompose_tensors:
        V = split_legs(V, 0)

    # if required, flip the leg duality
    if new_leg_dual != new_leg.is_dual:
        raise NotImplementedError

    return W, V


def enlarge_leg(tensor: Tensor, mask: Mask, leg: int | str) -> Tensor:
    """Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.

    The mask must be an inclusion, i.e. its large leg is in its codomain, at the top.
    We apply the mask via map composition::

        |                                │   │   ╭───╮   │
        |      │   │   │                 │   │  ┏┷┓  │   │           │ ┏━┷━┓ │
        |     ┏┷━━━┷━━━┷┓                │   │  ┃M┃  │   │           │ ┃M.T┃ │
        |     ┃ tensor  ┃                │   │  ┗┯┛  │   │           │ ┗━┯━┛ │
        |     ┗┯━━━┯━━━┯┛       OR       │   ╰───╯   │   │    ==    ┏┷━━━┷━━━┷┓
        |      │   │  ┏┷┓               ┏┷━━━━━━━━━━━┷━━━┷┓         ┃ tensor  ┃
        |      │   │  ┃M┃               ┃ tensor          ┃         ┗┯━━━┯━━━┯┛
        |      │   │  ┗┯┛               ┗┯━━━┯━━━┯━━━┯━━━┯┛          │   │   │
        |                                │   │   │   │   │

    where ``M.T == transpose(M)``.

    Parameters
    ----------
    tensor: Tensor
        The tensor to enlarge
    mask: Mask
        An *inclusion* mask. Its small leg must be equal to the respective :attr:`Tensor.legs`.
        Note that if the leg is in the domain this means ``mask.small_leg == domain[n].dual == legs[-n]``!
    leg: int | str
        Which leg of the tensor to enlarge

    Returns
    -------
    An embedded tensor of the same type as `tensor` (exception: `DiagonalTensor`s are converted to
    `SymmetricTensor`s before enlarging). The leg order and labels are the same as on `tensor`.
    The new leg is *larger* (or equal) than before.

    See Also
    --------
    apply_mask, compose, tdot, scale_axis

    """
    _ = get_same_device(tensor, mask)
    # parse inputs
    in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg)
    assert not mask.is_projection
    if in_domain:
        mask = transpose(mask)
    return _compose_with_Mask(tensor, mask, leg_idx)


def entropy(p: DiagonalTensor | Sequence[float], n=1):
    r"""The entropy of a probability distribution.

    Assumes that `p` is a probability distribution, i.e. real, non-negative and normalized to
    ``p.sum() == 1.``.

    For ``n==1``, we compute the von-Neumann entropy
    :math:`S_\text{vN} = -\mathrm{Tr}[p \mathrm{log} p]`.
    Otherwise, we compute the Renyi entropy
    :math:`S_n = \frac{1}{1 - n} \mathrm{log} \mathrm{Tr}[p^n]`

    Notes
    -----
    For non-abelian symmetries and anyonic gradings we have
    :math:`p = \bigotimes_a \rho_a \mathbb{1}_a` with :math:`\rho_a \ge 0`
    and :math:`\sum_a d_a \rho_a = 1`. The entropy is then obtained as
    :math:`S_\text{vN} = \sum_a d_a \rho_a \mathrm{log} \rho_a` or
    :math:`S_n = \frac{1}{1 - n} \mathrm{log} \sum_a d_a \rho_a^n` where :math:`d_a`
    is the quantum dimension of sector :math:`a`. (See :meth:`Symmetry.qdim`.)

    """
    if isinstance(p, DiagonalTensor):
        assert p.dtype.is_real
        if n == 1:
            return -trace(p * stable_log(p, cutoff=1e-30))
        if n == np.inf:
            return -np.log(p.max())
        return np.log(trace(p**n)) / (1.0 - n)
    else:
        p = np.asarray(p)
        p = np.real_if_close(p)
    p = p[p > 1e-30]  # for stability of log
    if n == 1:
        return -np.inner(np.log(p), p)
    if n == np.inf:
        return -np.log(np.max(p))
    return np.log(np.sum(p**n)) / (1.0 - n)


def exp(obj: Tensor | complex | float) -> Tensor | complex | float:
    """The exponential function.

    For a tensor, viewed as a linear map from its domain to its codomain, the exponential
    function is defined via its power series. For a diagonal tensor, this is equivalent to
    the :ref:`elementwise <diagonal_elementwise>` exponential function.
    """
    if isinstance(obj, DiagonalTensor):
        return obj._elementwise_unary(obj.backend.block_backend.exp)
    if isinstance(obj, ChargedTensor):
        raise TypeError('ChargedTensor can not be exponentiated.')
    if isinstance(obj, SymmetricTensor):
        _check_compatible_legs([obj.domain], [obj.codomain])

        combine = (not obj.backend.can_decompose_tensors) and obj.num_domain_legs > 1
        if combine:
            # OPTIMIZE have the same pipe in domain and codomain. could avoid recomputing?
            obj = combine_legs(obj, range(obj.num_codomain_legs), range(obj.num_codomain_legs, obj.num_legs))
        data = obj.backend.act_block_diagonal_square_matrix(obj, obj.backend.block_backend.matrix_exp, dtype_map=None)
        res = SymmetricTensor(data, codomain=obj.codomain, domain=obj.domain, backend=obj.backend, labels=obj.labels)
        if combine:
            res = split_legs(res, [0, 1])
        return res
    if isinstance(obj, Tensor):
        raise NotImplementedError  # should have considered all tensor types above
    return math_exp(obj)


def get_same_device(*tensors: Tensor, error_msg: str = 'Incompatible devices.') -> str:
    """If the given tensors have the same device, return it. Raise otherwise."""
    if len(tensors) == 0:
        raise ValueError('Need at least one tensor')
    device = tensors[0].device
    if not all(tens.device == device for tens in tensors[1:]):
        raise ValueError(error_msg)
    return device


def horizontal_factorization(
    tensor: Tensor,
    codomain_cut: int,
    domain_cut: int,
    new_labels: str | Sequence[str] = None,
    cutoff_singular_values: float = None,
) -> tuple[Tensor, Tensor]:
    """Factorize a tensor into left and right parts.

    Graphically, here with ``codomain_cut=3, domain_cut=1``::

        |      │   │   │               │           │   │             │   ╭──────╮    │   │
        |   ┏━━┷━━━┷━━━┷━━┓         ┏━━┷━━━━━━┓   ┏┷━━━┷┓         ┏━━┷━━━┷━━┓   │   ┏┷━━━┷┓
        |   ┃   tensor    ┃    =    ┃    A    ┠───┨  B  ┃   :=    ┃    A    ┃   │   ┃  B  ┃
        |   ┗┯━━━┯━━━┯━━━┯┛         ┗┯━━━┯━━━┯┛   ┗━━━━┯┛         ┗┯━━━┯━━━┯┛   │   ┗┯━━━┯┛
        |    │   │   │   │           │   │   │         │           │   │   │    ╰────╯   │

    Parameters
    ----------
    tensor: Tensor
        The tensor to factorize
    codomain_cut: int
        The first `codomain_cut` legs from the codomain end up in the codomain of `A`, the rest
        of the codomain ends up in the codomain of `B`.
    domain_cut: int
        The first `domain_cut` legs from the domain end up in the domain of `A`, the rest
        of the domain ends up in the domain of `B`.
    new_labels: (list of) str
        The labels for the new legs.
        Two entries ``[a, b]`` result in ``A.labels[-1 - domain_cut] == a`` and ``B.labels[0] == b``
        and a single entry ``a`` is equivalent to ``[a, a*]``.
    cutoff_singular_values: float, optional
        If ``None`` (default), we factorize using :func:`qr` without truncation. If given, we use a
        truncated SVD and truncate by discarding singular values below this threshold.

    Returns
    -------
    A, B: Tensor
        A factorization of the `tensor`, such that ``tdot(A, B, -1 - domain_cut, 1)`` reproduces
        the `tensor`, up to bending and possibly up to truncation if `cutoff_singular_values` is
        given.

    Notes
    -----
    This is achieved by bending legs such that we can do the factorization as a QR or SVD,
    then bend back, that is for the example case depicted above::

        |                                             │    │   │   ╭────╮         │   │   │
        |             │           │   │    ╭──╮       │ ┏━━┷━━━┷━━━┷━━┓ │         │  ┏┷━━━┷┓
        |             │  ╭────╮   │   │    │  │       │ ┃      B'     ┃ │         │  ┃  B  ┃
        |             │  │ ┏━━┷━━━┷━━━┷━━┓ │  │       │ ┗━━━━━━┯━━━━━━┛ │         │  ┗┯━━━┯┛
        |   LHS   =   │  │ ┃   tensor    ┃ │  │   =   │        │        │   =     │   │   │   =  RHS
        |             │  │ ┗┯━━━┯━━━┯━━━┯┛ │  │       │ ┏━━━━━━┷━━━━━━┓ │      ┏━━┷━━━┷━━┓│
        |             │  │  │   │   │   ╰──╯  │       │ ┃      A'     ┃ │      ┃    A    ┃│
        |             ╰──╯  │   │   │         │       │ ┗┯━━━┯━━━┯━━━┯┛ │      ┗┯━━━┯━━━┯┛│
        |                                             ╰──╯   │   │   │  │       │   │   │ │


    Note how we bend some legs to the left, to avoid any braids, such that the operation does not
    need to specify any braid chiralities.

    """
    # OPTIMIZE for fusion tree backend, can probably work something better out with explicit trees?
    assert 0 <= codomain_cut <= tensor.num_codomain_legs
    assert 0 <= domain_cut <= tensor.num_domain_legs
    if codomain_cut == 0 and domain_cut == 0:
        raise ValueError('Nothing to do')
    if codomain_cut == tensor.num_codomain_legs and domain_cut == tensor.num_domain_legs:
        raise ValueError('Nothing to do')

    J = tensor.num_codomain_legs
    J1 = codomain_cut
    J2 = J - J1
    K = tensor.num_domain_legs
    K1 = domain_cut
    K2 = K - K1

    to_decompose = permute_legs(
        tensor,
        codomain=[*range(J + K2, J + K), *range(J1)],
        domain=[*reversed(range(J1, J + K2))],
        bend_right=[True] * J + [False] * K,
    )

    if cutoff_singular_values is None:
        A, B = qr(to_decompose, new_labels=new_labels)
    else:
        A, S, Vh, _, _ = truncated_svd(to_decompose, new_labels=new_labels, svd_min=cutoff_singular_values)
        B = compose(S, Vh)

    A = permute_legs(A, codomain=[*range(K1, K1 + J1)], domain=[*reversed(range(K1)), -1], bend_right=False)
    B = permute_legs(B, codomain=[*range(1 + J2)], domain=[*reversed(range(1 + J2, 1 + J2 + K2))], bend_right=True)
    return A, B


@_elementwise_function(block_func='imag', maps_zero_to_zero=True)
def imag(x: _ElementwiseType) -> _ElementwiseType:
    """The imaginary part of a complex number, :ref:`elementwise <diagonal_elementwise>`."""
    return np.imag(x)


def inner(A: Tensor, B: Tensor, do_dagger: bool = True) -> float | complex:
    r"""The Frobenius inner product :math:`\langle A \vert B \rangle_\text{F}` of two tensors.

    Graphically::

        |          ╭───────────╮
        |          │   ╭─────╮ │
        |       ┏━━┷━━━┷━━┓  │ │
        |       ┃    B    ┃  │ │
        |       ┗┯━━━┯━━━┯┛  │ │
        |       ┏┷━━━┷━━━┷┓  │ │
        |       ┃dagger(A)┃  │ │
        |       ┗━━┯━━━┯━━┛  │ │
        |          │   ╰─────╯ │
        |          ╰───────────╯

    Assumes that the two tensors have the same (co-)domains.
    The inner product is defined as :math:`\mathrm{Tr}[ A^\dagger \circ B]`.
    It is thus equivalent to, but more efficient than ``trace(dot(A.hc, B))``.

    Parameters
    ----------
    A, B
        The two tensors. Must have matching (co-)domains, except if ``do_dagger=False``.
    do_dagger: bool
        If ``True``, the standard inner product as above is computed.
        If ``False``, we assume that the dagger has already been performed on one of the tensors.
        Thus we require ``tensor_1.domain == tensor_2.codomain`` and vice versa and just perform
        the contraction and trace.

    See Also
    --------
    norm
        The Frobenius norm, induced by this inner product.

    """
    _ = get_same_device(A, B)

    if do_dagger:
        _check_compatible_legs([A.codomain, A.domain], [B.codomain, B.domain])
    else:
        _check_compatible_legs([A.codomain, A.domain], [B.domain, B.codomain])

    if isinstance(A, (DiagonalTensor, Mask)):
        # in this case, there is no benefit to having a dedicated backend function,
        # as the dot is cheap
        if do_dagger:
            return trace(compose(dagger(A), B))
        return trace(compose(A, B))
    if isinstance(B, (DiagonalTensor, Mask)):
        # same argument as above.
        if do_dagger:
            return np.conj(trace(compose(dagger(B), A)))
        return trace(compose(A, B))

    # remaining cases: both are either SymmetricTensor or ChargedTensor
    backend = get_same_backend(A, B)

    if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
        if A.charged_state is None or B.charged_state is None:
            raise ValueError('charged_state must be specified for inner()')
        if do_dagger:
            inv_part = _compose_SymmetricTensors(
                bend_legs(dagger(A.invariant_part), num_codomain_legs=1),  # ['!*'] <- [*a_legs]
                bend_legs(B.invariant_part, num_domain_legs=1),  # [*b_legs] <- ['!']
            )  # ['!*', '!']
            # OPTIMIZE: like GEMM, should we offer an interface where dagger is implicitly done during tdot?
            inv_block = inv_part.to_dense_block(understood_braiding=True)
            res = backend.block_backend.tdot(
                backend.block_backend.conj(A.charged_state),
                backend.block_backend.tdot(inv_block, B.charged_state, [1], [0]),
                [0],
                [0],
            )
        else:
            A_inv = permute_legs(
                A.invariant_part, [-1], [*reversed(range(A.num_legs))], bend_right=[True] * A.num_legs + [False]
            )
            B_inv = permute_legs(B.invariant_part, [*range(A.num_legs)], [-1], bend_right=True)
            inv_part = _compose_SymmetricTensors(A_inv, B_inv, relabel1={'!': '!A'}, relabel2={'!': '!B'})
            assert inv_part.labels == ['!A', '!B']
            inv_block = inv_part.to_dense_block(understood_braiding=True)
            # [!A, !B] @ [!B*] -> [!A]
            res = backend.block_backend.tdot(inv_block, B.charged_state, [1], [0])
            # [!A] @ [!A*] -> []
            res = backend.block_backend.tdot(A.charged_state, res, [0], [0])
        return backend.block_backend.item(res)

    if isinstance(A, ChargedTensor):  # and B is a SymmetricTensor
        # reduce to the case where B is charged and A is not  # OPTIMIZE write it out instead...
        if do_dagger:
            return np.conj(inner(B, A, do_dagger=True))
        return inner(B, A, do_dagger=False)

    if isinstance(B, ChargedTensor):
        if B.charged_state is None:
            raise ValueError('charged_state must be specified for inner()')
        if B.charge_leg.sector_multiplicity(B.symmetry.trivial_sector) == 0:
            return Dtype.common(A.dtype, B.dtype).zero_scalar
        # OPTIMIZE: by charge rule, only components in the trivial sector of the charge_leg contribute
        #           could exploit by projecting to those components first.
        if do_dagger:
            inv_part = tdot(dagger(A), B.invariant_part, [*range(A.num_legs)], [*reversed(range(A.num_legs))])
            B_state = backend.block_backend.conj(B.charged_state)
            res = backend.block_backend.tdot(inv_part.to_dense_block(), B_state, [0], [0])
        else:
            inv_part = tdot(A, B.invariant_part, [*range(A.num_legs)], [*reversed(range(A.num_legs))])
            res = backend.block_backend.tdot(inv_part.to_dense_block(), B.charged_state, [0], [0])
        return backend.block_backend.item(res)

    # remaining case: both are SymmetricTensor
    return backend.inner(A, B, do_dagger=do_dagger)


def is_scalar(obj):
    """If an object is a scalar.

    We count numbers as scalars, if they fulfill ``isinstance(obj, numbers.Number)``.
    This is the case e.g. for builtin numbers

    A tensor counts as a scalar if both its domain and its codomain consist of the *same* single
    sector. For abelian group symmetries, this is equivalent to saying that ``tensor.to_numpy()``
    is an array with a single entry.
    In the general case, the consequence is that ``tensor.num_parameters == 1``.
    The tensor, as a map from codomain to domain must be a multiple of the identity on that single
    sector and the prefactor is the single free parameter.
    We can thus think of the tensor as equivalent to that single parameter, i.e. a scalar.
    """
    if isinstance(obj, Tensor):
        if obj.domain.num_sectors != 1:
            return False
        if obj.codomain.num_sectors != 1:
            return False
        if not np.all(obj.domain.sector_decomposition == obj.codomain.sector_decomposition):
            return False
        if not np.all(obj.domain.multiplicities == 1):
            return False
        if not np.all(obj.codomain.multiplicities == 1):
            return False
        return True
    return isinstance(obj, Number)


def item(tensor: Tensor) -> float | complex | bool:
    """If the tensor is a scalar (with only trivial legs), convert to python scalar."""
    if not is_scalar(tensor):
        raise ValueError('Not a scalar')
    if isinstance(tensor, Mask):
        return Mask.any(tensor)
    if isinstance(tensor, (DiagonalTensor, SymmetricTensor)):
        return tensor.backend.item(tensor)
    if isinstance(tensor, ChargedTensor):
        if tensor.charged_state is None:
            raise ValueError('Can not compute .item of ChargedTensor with unspecified charged_state.')
        inv_block = tensor.invariant_part.to_dense_block(understood_braiding=True)
        res = tensor.backend.block_backend.tdot(tensor.charged_state, inv_block, 0, -1)
        return tensor.backend.block_backend.item(res)
    raise TypeError


def linear_combination(a: Number, v: Tensor, b: Number, w: Tensor):
    """The linear combination ``a * v + b * w``"""
    _ = get_same_device(v, w)
    _check_compatible_legs([v.codomain, v.domain], [w.codomain, w.domain])
    # Note: We implement Tensor.__add__ and Tensor.__sub__ in terms of this function, so we cant
    #       use them (or the ``+`` and ``-`` operations) here.
    if (not isinstance(a, Number)) or (not isinstance(b, Number)):
        msg = f'unsupported scalar types: {type(a).__name__}, {type(b).__name__}'
        raise TypeError(msg)
    if isinstance(v, DiagonalTensor) and isinstance(w, DiagonalTensor):
        return DiagonalTensor._binary_operand(v, w, func=lambda _v, _w: a * _v + b * _w, operand='linear_combination')
    if isinstance(v, ChargedTensor) and isinstance(w, ChargedTensor):
        if v.charge_leg != w.charge_leg:
            raise ValueError('Can not add ChargedTensors with different dummy legs')
        if (v.charged_state is None) != (w.charged_state is None):
            raise ValueError('Can not add ChargedTensors with unspecified and specified charged_state')
        if v.charged_state is None:
            inv_part = linear_combination(a, v.invariant_part, b, w.invariant_part)
            return ChargedTensor(inv_part, None)
        if v.charge_leg.dim == 1:
            factor = v.backend.block_backend.item(w.charged_state) / v.backend.block_backend.item(v.charged_state)
            inv_part = linear_combination(a, v.invariant_part, factor * b, w.invariant_part)
            return ChargedTensor(inv_part, v.charged_state)
        raise NotImplementedError
    if isinstance(v, ChargedTensor) or isinstance(w, ChargedTensor):
        raise TypeError('Can not add ChargedTensor and non-charged tensor.')

    # Remaining case: Mask, DiagonalTensor (but not both), SymmetricTensor
    if isinstance(v, (DiagonalTensor, Mask)) or isinstance(w, (DiagonalTensor, Mask)):
        msg = (
            f'Converting types ({type(v).__name__, type(w).__name__}) to '
            f'(SymmetricTensor, SymmetricTensor) for  linear_combination. '
            f'Use tensor.as_SymmetricTensor() explicitly to suppress this warning.'
        )
        warnings.warn(msg, stacklevel=2)
    v = v.as_SymmetricTensor()
    w = w.as_SymmetricTensor()

    backend = get_same_backend(v, w)
    return SymmetricTensor(
        backend.linear_combination(a, v, b, w),
        codomain=v.codomain,
        domain=v.domain,
        backend=backend,
        labels=_get_matching_labels(v._labels, w._labels),
    )


def move_leg(
    tensor: Tensor,
    which_leg: int | str,
    codomain_pos: int | None = None,
    *,
    domain_pos: int | None = None,
    levels: list[int] | dict[str | int, int] | None = None,
    bend_right: bool = None,
) -> Tensor:
    """Move one leg of a tensor to a specified position.

    Graphically::

        |        │   ╭───│─╯ │
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃      T      ┃       ==    move_leg(T, 6, domain_pos=-2)
        |       ┗┯━━━┯━━━┯━━━┯┛
        |        │   │   │   │

    Or::

        |        │   │   ╭───│───╮
        |       ┏┷━━━┷━━━┷━━━┷┓  │
        |       ┃      T      ┃  │    ==    move_leg(T, 5, codomain_pos=1, bend_right=True)
        |       ┗┯━━━┯━━━┯━━━┯┛  │
        |        │ ╭─│───│───│───╯

    Parameters
    ----------
    tensor: Tensor
        The tensor to act on
    which_leg: int | str
        Which leg of the `tensor` to move, by index or by label.
    codomain_pos: int, optional, keyword only
        If given, move the leg to that position of the resulting codomain.
    domain_pos: int, optional, keyword only
        If given, move the lef to that position of the resulting domain.
    levels: optional
        Is ignored if the symmetry has symmetric braids. Otherwise, these levels specify the
        chirality of any possible braids induced by permuting the legs. See :func:`permute_legs`.
    bend_right: bool
        If the moving leg should bend to the right of the tensor (as shown above) or to the left.
        If either the leg does not bend at all or if the symmetry has symmetric braids, the argument
        is ignored since it either does not apply or both options are equivalent anyway.

    """
    from_domain, _, leg_idx = tensor._parse_leg_idx(which_leg)
    if from_domain:
        new_codomain = list(range(tensor.num_codomain_legs))
        new_domain = [n for n in reversed(range(tensor.num_codomain_legs, tensor.num_legs)) if n != leg_idx]
    else:
        new_codomain = [n for n in range(tensor.num_codomain_legs) if n != leg_idx]
        new_domain = list(reversed(range(tensor.num_codomain_legs, tensor.num_legs)))
    #
    if codomain_pos is not None:
        if domain_pos is not None:
            raise ValueError('Can not specify both codomain_pos and domain_pos.')
        pos = to_valid_idx(codomain_pos, len(new_codomain) + 1)
        new_codomain[pos:pos] = [leg_idx]
    elif domain_pos is not None:
        pos = to_valid_idx(domain_pos, len(new_domain) + 1)
        new_domain[pos:pos] = [leg_idx]
    else:
        raise ValueError('Need to specify either codomain_pos or domain_pos.')
    #
    return permute_legs(tensor, new_codomain, new_domain, levels=levels, bend_right=bend_right)


def norm(tensor: Tensor) -> float:
    r"""The Frobenius norm of a Tensor.

    The norm is given by :math:`\Vert A \Vert_\text{F} = \sqrt{\langle A \vert A \rangle_\text{F}}`,
    where :math:`\langle {-} \vert {-} \rangle_\text{F}` is the Frobenius inner product, implemented
    in :func:`inner`.
    """
    if isinstance(tensor, Mask):
        # norm ** 2 = Tr(m^\dagger . m) = Tr(id_{small_leg}) = dim(small_leg)
        return np.sqrt(tensor.small_leg.dim)
    if isinstance(tensor, (DiagonalTensor, SymmetricTensor)):
        return tensor.backend.norm(tensor)
    if isinstance(tensor, ChargedTensor):
        if tensor.charged_state is None:
            msg = (
                'norm of a ChargedTensor with unspecified charged_state is ambiguous. '
                'Use e.g. norm(tensor.invariant_part).'
            )
            raise ValueError(msg)
        if tensor.charge_leg.dim == 1:
            factor = abs(tensor.backend.block_backend.item(tensor.charged_state))
            return factor * tensor.backend.norm(tensor.invariant_part)
        else:
            # OPTIMIZE
            warnings.warn('Converting ChargedTensor to dense block for `norm`', stacklevel=2)
            block = tensor.to_dense_block(understood_braiding=True)
            return tensor.backend.block_backend.norm(block, order=2)
    raise TypeError


def on_device(tensor: Tensor, device: str, copy: bool = True) -> Tensor:
    """An equivalent tensor (with the same entries) on another device.

    Parameters
    ----------
    tensor: Tensor
        The tensor to move
    device: str
        The device to move to
    copy: bool
        If a copy should be made. Otherwise, operate *in-place*.

    Returns
    -------
    If `copy` (default), a new instance, on `device`.
    Otherwise, the instance `tensor` is modified in-place, and then returned.

    """
    if copy:
        return tensor.copy(device=device)
    tensor.move_to_device(device)
    return tensor


def outer(tensor1: Tensor, tensor2: Tensor, relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    r"""The outer product, or tensor product.

    The outer product of two maps :math:`A : W_A \to V_A` and :math:`B : W_B \to V_B` is
    a map :math:`A \otimes B : W_A \otimes W_B \to V_A \otimes V_B`.

    |        │   │   │   │            │   │     │   │
    |       ┏┷━━━┷━━━┷━━━┷┓          ┏┷━━━┷┓   ┏┷━━━┷┓
    |       ┃ outer(A, B) ┃    ==    ┃  A  ┃   ┃  B  ┃
    |       ┗━━┯━━━┯━━━┯━━┛          ┗┯━━━┯┛   ┗━━┯━━┛
    |          │   │   │              │   │       │

    Returns
    -------
    The outer product :math:`A \otimes B`, with domain ``[*A.domain, *B.domain]`` and codomain
    ``[*A.codomain, *B.codomain]``. Thus, the :attr:`Tensor.legs` are, *up to a permutation*,
    the :attr:`Tensor.legs` of `A` plus the :attr:`Tensor.legs` of `B`.

    See Also
    --------
    tdot
        tdot with no actual contractions gives a similar product, with different leg order.
    relabel1, relabel2: dict[str, str], optional
        A mapping of labels for each of the tensors. The result has labels, as if the
        input tensors were relabelled accordingly before contraction.

    """
    _ = get_same_device(tensor1, tensor2)
    assert tensor1.symmetry.is_same_symmetry(tensor2.symmetry)

    if isinstance(tensor1, (Mask, DiagonalTensor)):
        msg = 'Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to suppress the warning.'
        tensor1 = tensor1.as_SymmetricTensor(warning=msg)
    if isinstance(tensor2, (Mask, DiagonalTensor)):
        msg = 'Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to suppress the warning.'
        tensor2 = tensor2.as_SymmetricTensor(warning=msg)
    if isinstance(tensor1, ChargedTensor):
        if isinstance(tensor2, ChargedTensor):
            bang = ChargedTensor._CHARGE_LEG_LABEL
            inv_part = outer(
                tensor1.invariant_part,
                tensor2.invariant_part,
                relabel1={**relabel1, bang: f'{bang}1'},
                relabel2={**relabel2, bang: f'{bang}2'},
            )
            inv_part = move_leg(inv_part, f'{bang}2', domain_pos=1)
            return ChargedTensor.from_two_charge_legs(inv_part, tensor1.charged_state, tensor2.charged_state)
        else:
            inv_part = outer(tensor1.invariant_part, tensor2, relabel1=relabel1, relabel2=relabel2)
            return ChargedTensor(inv_part, tensor1.charged_state)
    if isinstance(tensor2, ChargedTensor):
        inv_part = outer(tensor1, tensor2.invariant_part, relabel1=relabel1, relabel2=relabel2)
        inv_part = move_leg(inv_part, tensor1.num_codomain_legs + tensor2.num_legs, domain_pos=0)
        return ChargedTensor(inv_part, tensor2.charged_state)
    backend = get_same_backend(tensor1, tensor2)
    data = backend.outer(tensor1, tensor2)
    codomain = TensorProduct.from_partial_products(tensor1.codomain, tensor2.codomain)
    domain = TensorProduct.from_partial_products(tensor1.domain, tensor2.domain)
    # construct new labels
    codomain_labels = []
    domain_labels = []
    if relabel1 is None:
        codomain_labels.extend(tensor1.codomain_labels)
        domain_labels.extend(tensor1.domain_labels)
    else:
        codomain_labels.extend(relabel1.get(l, l) for l in tensor1.codomain_labels)
        domain_labels.extend(relabel1.get(l, l) for l in tensor1.domain_labels)
    if relabel2 is None:
        codomain_labels.extend(tensor2.codomain_labels)
        domain_labels.extend(tensor2.domain_labels)
    else:
        codomain_labels.extend(relabel2.get(l, l) for l in tensor2.codomain_labels)
        domain_labels.extend(relabel2.get(l, l) for l in tensor2.domain_labels)
    #
    return SymmetricTensor(data, codomain, domain, backend, [codomain_labels, domain_labels])


def partial_compose(
    tensor1: Tensor,
    tensor2: Tensor,
    tensor1_first_leg: str | int,
    relabel1: dict[str, str] = None,
    relabel2: dict[str, str] = None,
) -> Tensor:
    r"""Tensor contraction / composition involving only a part of the full (co)domain.

    Requires that all codomain (domain) legs of `tensor2` are consistent with the respective domain
    (codomain) legs of `tensor1`; all legs to be contracted must be either in the codomain or in
    the domain and `tensor1` must have at least one leg in the domain (codomain) that is not
    contracted.

    Graphically::

        |        │   │   │   │
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃      A      ┃ == partial_compose(A, B, 2)
        |       ┗┯━━━┯━━━┯━━━┯┛
        |        │   │  ┏┷━━━┷┓
        |        │   │  ┃  B  ┃
        |        │   │  ┗┯━━━┯┛

    Or::

        |        │   │  ┏┷━━━┷┓
        |        │   │  ┃  B  ┃
        |        │   │  ┗┯━━━┯┛
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃      A      ┃ == partial_compose(A, B, 4)
        |       ┗┯━━━┯━━━┯━━━┯┛
        |        │   │   │   │

    Parameters
    ----------
    tensor1, tensor2: Tensor
        The two tensors to partially compose.
    tensor1_first_leg: str | int
        Which leg of `tensor1` is the first to be contracted with the first leg of `tensor2`.
        In particular, if ``tensor1_first_leg < tensor1.num_codomain_legs``, part of the codomain
        of `tensor1` is contracted with the full domain of `tensor2`, where
        ``tensor1.codomain[tensor1_first_leg] == tensor2.domain[0]``.
        Otherwise (``tensor1_first_leg >= tensor1.num_codomain_legs``), part of the domain of
        `tensor1` is contracted with the full codomain of `tensor2`, where
        ``tensor1.domain[tensor1.num_legs - 1 - tensor1_first_leg] == tensor2.codomain[-1]``.
    relabel1, relabel2: dict[str, str], optional
        A mapping of labels for each of the tensors. The result has labels as if the input tensors
        were relabelled accordingly before contraction.

    Returns
    -------
    The partially composed tensor. The resulting legs correspond to the legs of `tensor1` after
    replacing the legs to be contracted by the open legs of `tensor2`.

    See Also
    --------
    compose, tdot, apply_mask, scale_axis

    """
    _ = get_same_device(tensor1, tensor2)
    tensor1_first_leg = tensor1.get_leg_idcs(tensor1_first_leg)[0]

    if relabel1 is None:
        codomain_labels = tensor1.codomain_labels
        domain_labels = tensor1.domain_labels
    else:
        codomain_labels = [relabel1.get(l, l) for l in tensor1.codomain_labels]
        domain_labels = [relabel1.get(l, l) for l in tensor1.domain_labels]

    leg_msg = 'Not all legs to be contracted are in the (co)domain'
    compose_msg = 'Use compose for contracting the full (co)domain'
    if tensor1_first_leg < tensor1.num_codomain_legs:
        num_legs = tensor2.num_domain_legs
        tensor1_last_leg = tensor1_first_leg + num_legs - 1
        assert tensor1_last_leg < tensor1.num_codomain_legs, leg_msg
        assert num_legs < tensor1.num_codomain_legs, compose_msg
        _check_compatible_legs(
            tensor1.codomain.factors[tensor1_first_leg : tensor1_last_leg + 1], tensor2.domain.factors
        )
        if relabel2 is None:
            tensor2_labels = tensor2.codomain_labels
        else:
            tensor2_labels = [relabel2.get(l, l) for l in tensor2.codomain_labels]
        codomain_labels[tensor1_first_leg : tensor1_last_leg + 1] = tensor2_labels

        new_codomain = tensor1.codomain.factors[:]
        new_codomain[tensor1_first_leg : tensor1_last_leg + 1] = tensor2.codomain
        new_codomain = TensorProduct(new_codomain, tensor1.symmetry)
        new_domain = tensor1.domain
    else:
        num_legs = tensor2.num_codomain_legs
        tensor1_last_leg = tensor1_first_leg + num_legs - 1
        assert tensor1_last_leg < tensor1.num_legs, leg_msg
        assert num_legs < tensor1.num_domain_legs, compose_msg
        domain_first_leg = tensor1.num_legs - 1 - tensor1_last_leg
        domain_last_leg = tensor1.num_legs - 1 - tensor1_first_leg
        _check_compatible_legs(tensor1.domain.factors[domain_first_leg : domain_last_leg + 1], tensor2.codomain.factors)
        if relabel2 is None:
            tensor2_labels = tensor2.domain_labels
        else:
            tensor2_labels = [relabel2.get(l, l) for l in tensor2.domain_labels]
        domain_labels[domain_first_leg : domain_last_leg + 1] = tensor2_labels

        new_codomain = tensor1.codomain
        new_domain = tensor1.domain[:]
        new_domain[domain_first_leg : domain_last_leg + 1] = tensor2.domain
        new_domain = TensorProduct(new_domain, tensor1.symmetry)

    res_labels = [*codomain_labels, *reversed(domain_labels)]
    assert not duplicate_entries(res_labels, ignore=[None]), 'duplicate labels'

    # tensor1 cannot be Mask or DiagonalTensor due to num_legs constraint
    if isinstance(tensor2, Mask):
        return _compose_with_Mask(tensor1, tensor2, tensor1_first_leg).set_labels(res_labels)

    if isinstance(tensor2, DiagonalTensor):
        return scale_axis(tensor1, tensor2, tensor1_first_leg).set_labels(res_labels)

    if isinstance(tensor1, ChargedTensor) and isinstance(tensor2, ChargedTensor):
        if (tensor1.charged_state is None) != (tensor2.charged_state is None):
            raise ValueError('Mismatched: specified and unspecified ChargedTensor.charged_state')
        c = ChargedTensor._CHARGE_LEG_LABEL
        c1 = c + '1'
        c2 = c + '2'
        relabel1 = {c: c1} if relabel1 is None else {**relabel1, c: c1}
        relabel2 = {c: c2} if relabel2 is None else {**relabel2, c: c2}
        inv_part = tensor2.invariant_part
        if tensor1_first_leg < tensor1.num_codomain_legs:
            # need to bend down charge leg first
            inv_part = move_leg(inv_part, c, codomain_pos=tensor2.num_codomain_legs - 1, bend_right=True)
        inv_part = partial_compose(tensor1.invariant_part, inv_part, tensor1_first_leg, relabel1, relabel2)
        # domain_pos 1 since domain_pos 0 would mean braiding with c1
        inv_part = move_leg(inv_part, c2, domain_pos=1, bend_right=True)
        return ChargedTensor.from_two_charge_legs(inv_part, state1=tensor1.charged_state, state2=tensor2.charged_state)
    if isinstance(tensor1, ChargedTensor):
        inv_part = partial_compose(tensor1.invariant_part, tensor2, tensor1_first_leg, relabel1, relabel2)
        return ChargedTensor.from_invariant_part(inv_part, tensor1.charged_state)
    if isinstance(tensor2, ChargedTensor):
        inv_part = tensor2.invariant_part
        if tensor1_first_leg < tensor1.num_codomain_legs:
            # need to bend down charge leg first
            inv_part = move_leg(
                inv_part, ChargedTensor._CHARGE_LEG_LABEL, codomain_pos=tensor2.num_codomain_legs - 1, bend_right=True
            )
        inv_part = partial_compose(tensor1, inv_part, tensor1_first_leg, relabel1, relabel2)
        inv_part = move_leg(inv_part, ChargedTensor._CHARGE_LEG_LABEL, domain_pos=0, bend_right=True)
        return ChargedTensor.from_invariant_part(inv_part, tensor2.charged_state)

    backend = get_same_backend(tensor1, tensor2)
    data = backend.partial_compose(tensor1, tensor2, tensor1_first_leg, new_codomain, new_domain)
    return SymmetricTensor(data=data, codomain=new_codomain, domain=new_domain, backend=backend, labels=res_labels)


def partial_trace(
    tensor: Tensor, *pairs: Sequence[int | str], levels: list[int] | dict[str | int, int] | None = None
) -> Tensor:
    """Perform a partial trace.

    An arbitrary number of pairs can be traced over::

        |    │       ╭───────╮
        |    │   ╭───│───╮   │
        |    7   6   5   4   │
        |   ┏┷━━━┷━━━┷━━━┷┓  │
        |   ┃      A      ┃  │    ==   trace(A, (0, 2), (3, 5), (-2, 4))
        |   ┗┯━━━┯━━━┯━━━┯┛  │
        |    0   1   2   3   │
        |    ╰───│───╯   ╰───╯


    Note that despite its name, a "full" trace with a scalar result *can* be realized.

    Parameters
    ----------
    tensor: Tensor
        The tensor to act on
    *pairs:
        A number of pairs, each describing two legs via index or via label.
        Each pair is connected, realizing a partial trace.
        By definition, we create loops between legs on opposite sides to the right side of the
        tensor (this is not necessarily equivalent to a left closing, if there are braids).
        Must be compatible ``tensor.get_leg(pair[0]) == tensor.get_leg(pair[1]).dual``.
    levels:
        The connectivity of the partial trace may induce braids.
        For symmetries with non-symmetric braiding, these levels are used to determine the
        chirality of those braids, like in :func:`permute_legs`.

    Returns
    -------
    If all legs are traced, a python scalar.
    If legs are left open, a tensor with the same type as `tensor`.

    See Also
    --------
    trace

    """
    # check legs are compatible
    pairs = [tensor.get_leg_idcs(pair) for pair in pairs]
    traced_idcs = [l for pair in pairs for l in pair]
    duplicates = duplicate_entries(traced_idcs)
    if duplicates:
        raise ValueError('Pairs may not contain duplicates.')
    _check_compatible_legs(
        [tensor._as_codomain_leg(i1) for i1, _ in pairs], [tensor._as_domain_leg(i2) for _, i2 in pairs]
    )

    if len(pairs) == 0:
        return tensor
    # deal with other tensor types
    if isinstance(tensor, (DiagonalTensor, Mask)):
        # only remaining option after input checks if the full trace.
        return trace(tensor)
    if isinstance(tensor, ChargedTensor):
        if levels is not None:
            # charge leg is not traced and thus does not braid.
            # so its level is irrelevant. just make sure its not a duplicate
            levels = [*levels, min(levels) - 1]
        invariant_part = partial_trace(tensor.invariant_part, *pairs, levels=levels)
        if invariant_part.num_legs == 1:
            # scalar result
            if tensor.charged_state is None:
                raise ValueError('Need to specify charged_state for full trace of ChargedTensor')
            inv_block = invariant_part.to_dense_block(understood_braiding=True)
            res = tensor.backend.block_backend.tdot(inv_block, tensor.charged_state, [0], [0])
            return tensor.backend.block_backend.item(res)
        return ChargedTensor(invariant_part, tensor.charged_state)
    if not isinstance(tensor, SymmetricTensor):
        raise TypeError(f'Unexpected tensor type: {type(tensor).__name__}')
    if levels is None:
        levels = [None] * tensor.num_legs
    else:
        levels = list(levels)  # ensure copy

    try:
        data, codomain, domain = tensor.backend.partial_trace(tensor, pairs, levels)
    except SymmetryError:
        raise SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG) from None

    if tensor.num_legs == len(traced_idcs):
        # should be a scalar
        return data
    labels = [l for n, l in enumerate(tensor._labels) if n not in traced_idcs]
    return SymmetricTensor(data=data, codomain=codomain, domain=domain, backend=tensor.backend, labels=labels)


def permute_legs(
    tensor: Tensor,
    codomain: list[int | str] = None,
    domain: list[int | str] = None,
    levels: list[int] | dict[str | int, int] = None,
    bend_right: bool | Sequence[bool | None] | dict[str | int, bool] = None,
):
    """Permute the legs of a tensor by braiding legs and bending lines.

    Graphically (note that we ignore the `levels` graphically and do not draw braid chiralities)::


    |             │ ╰─│─────╮ │
    |         ╭───│───│─────│─│─╮
    |         6   5   4     │ │ │
    |      ┏━━┷━━━┷━━━┷━━┓  │ │ │
    |      ┃      T      ┃  │ │ │   =    permute_legs(T, [1, 3, -1], [5, 2, 4, 0])
    |      ┗┯━━━┯━━━┯━━━┯┛  │ │ │
    |       0   1   2   3   │ │ │
    |       │   │   ╰───│───╯ │ │
    |       ╰───│───────│─────╯


    |        │ ╭─────────│─╯ │
    |      ╭─│─│─────╮   │   │
    |      │ │ │     6   5   4
    |      │ │ │  ┏━━┷━━━┷━━━┷━━┓
    |      │ │ │  ┃      T      ┃   =   permute_legs(T, [6, 1, 3], [0, 5, 2, 4], bend_right=False)
    |      │ │ │  ┗┯━━━┯━━━┯━━━┯┛
    |      │ │ │   0   1   2   3
    |      │ │ ╰───│───│───╯   │
    |      │ ╰─────╯   │       │

    .. note ::
        We expect that there are only two cases where you should do explicit leg permutations:
        Firstly, if you need to specify the `levels` explicitly in the case of an anyonic symmetry.
        Secondly, if you are optimizing for performance and know what you are doing.
        In most other cases, you should be able to refer to legs by label and let the API functions
        do implicit leg rearrangements as needed.

    .. warning ::
        It is inefficient (especially when using the fusiontree backend) to do a series of leg
        rearrangements as multiple function calls. For performance, they should be done in a
        single call.

    Parameters
    ----------
    tensor: Tensor
        The tensor to permute
    codomain, domain: list of {int | str}
        Which of the legs of `tensor`, specified by their position in ``tensor.legs`` or by
        string label, should end up in the (co)domain of the result.
        Only one of the two is required; the other one is determined by using "the rest" of
        the legs of `tensor`, such that their order in ``tensor.legs`` is unchanged.
        Together, `codomain` and `domain` must comprise all legs of the original `tensor` without
        duplicates.
    levels
        If the symmetry has symmetric braiding (e.g. for group symmetries, or fermions, see
        :attr:`Symmetry.braiding_style`), this argument is ignored.
        For non-symmetric braiding, this argument specifies if a crossing of legs is an over-
        or under-crossing. It assigns a "level" or height to each leg.
        Either as a list ``levels[leg_num]`` or as a dictionary ``levels[leg_num_or_label]``.
        If two legs are crossed at some point, the one with the higher level goes over the other.
    bend_right
        For each leg that bends up or down, whether it bends to right of the tensor (as shown
        above) or to the left. If the symmetry has a symmetric braid (e.g. group symmetries or
        fermions), this makes no difference and this argument is ignored. For anyonic symmetries,
        the two options are not equivalent and an explicit choice is required for all legs that
        do bend. Allowed formats are::

            - A single boolean is applied to all legs.
            - A list of bools specifies for each leg by leg index.
              ``None`` is allowed as a placeholder for legs that do not bend.
            - A dictionary with keys that are either leg indices or leg labels, and bool values.

    """
    # Parse domain and codomain to list[int]. Get rid of duplicates.
    if codomain is None and domain is None:
        raise ValueError('Need to specify either domain or codomain.')
    elif codomain is None:
        domain = tensor.get_leg_idcs(domain)
        codomain = [n for n in range(tensor.num_legs) if n not in domain]
    elif domain is None:
        codomain = tensor.get_leg_idcs(codomain)
        # to preserve order of Tensor.legs, need to put domain legs in descending order of their leg_idx
        domain = [n for n in reversed(range(tensor.num_legs)) if n not in codomain]
    else:
        domain = tensor.get_leg_idcs(domain)
        codomain = tensor.get_leg_idcs(codomain)
        specified_legs = [*domain, *codomain]
        duplicates = duplicate_entries(specified_legs)
        missing = [n for n in range(tensor.num_legs) if n not in specified_legs]
        if duplicates:
            raise ValueError(f'Duplicate entries. By leg index: {", ".join(map(str, duplicates))}')
        if missing:
            raise ValueError(f'Missing legs. By leg index: {", ".join(map(str, missing))}')
    # Special case: if no legs move
    if codomain == list(range(tensor.num_codomain_legs)) and domain == list(
        reversed(range(tensor.num_codomain_legs, tensor.num_legs))
    ):
        return tensor

    # parse levels to format list[int | None]
    if levels is None:
        levels = [None] * tensor.num_legs
    elif isinstance(levels, dict):
        tmp = [None] * tensor.num_legs
        for leg, level in levels.items():
            idx = tensor.get_leg_idcs(leg)[0]
            if tmp[idx] is not None:
                raise ValueError(f'Level for leg {leg} defined multiple times.')
            tmp[idx] = level
        levels = tmp
    else:
        levels = list(levels)
        assert len(levels) == tensor.num_legs

    # parse bend_right to format list[bool | None]
    legs_bending_down = [i for i in domain if i < tensor.num_codomain_legs]
    legs_bending_up = [i for i in codomain if i >= tensor.num_codomain_legs]
    bending_legs = legs_bending_down + legs_bending_up
    if isinstance(bend_right, dict):
        tmp = [None] * tensor.num_legs
        for leg, b in bend_right.items():
            tmp[tensor.get_leg_idcs(leg)[0]] = b
        bend_right = tmp
    elif is_iterable(bend_right):
        assert len(bend_right) == tensor.num_legs
    elif bend_right is None:  # default -> all undefined
        bend_right = [None] * tensor.num_legs
    elif bend_right in [True, False]:  # single bool applies to all legs
        bend_right = [bend_right] * tensor.num_legs
    else:
        raise ValueError
    # check if those that need to be specified are
    if tensor.symmetry.has_trivial_braid:
        # it doesnt matter which way. choose all right
        bend_right = [True] * tensor.num_legs
    else:
        if any(bend_right[l] is None for l in bending_legs):
            raise SymmetryError('Need to specify bend_right!')

    # Deal with other tensor types
    if isinstance(tensor, (DiagonalTensor, Mask)):
        if codomain == [0] and domain == [1]:
            return tensor
        if codomain == [1] and domain == [0]:
            return transpose(tensor)
        # other cases involve two legs either in the domain or codomain.
        # Cant be done with Mask / DiagonalTensor
        msg = (
            'Converting to SymmetricTensor for permuting legs. '
            'Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        tensor = tensor.as_SymmetricTensor(warning=msg)
    if isinstance(tensor, ChargedTensor):
        # assign level `None` to the charge leg. it does not braid, so we dont need to define it.
        inv_part = permute_legs(
            tensor.invariant_part,
            codomain=codomain,
            domain=[-1, *domain],
            levels=[*levels, None],
            bend_right=[*bend_right, None],
        )
        return ChargedTensor(inv_part, charged_state=tensor.charged_state)

    # Build new codomain and domain
    if len(bending_legs) > 0:
        new_codomain = TensorProduct([tensor._as_codomain_leg(i) for i in codomain], symmetry=tensor.symmetry)
        new_domain = TensorProduct([tensor._as_domain_leg(i) for i in domain], symmetry=tensor.symmetry)
    else:
        # (co)domain has the same factor as before, only permuted -> can re-use sectors!
        new_codomain = tensor.codomain.permuted(codomain)
        new_domain = tensor.domain.permuted([tensor.num_legs - 1 - i for i in domain])

    data = tensor.backend.permute_legs(
        tensor,
        codomain_idcs=codomain,
        domain_idcs=domain,
        new_codomain=new_codomain,
        new_domain=new_domain,
        mixes_codomain_domain=len(bending_legs) > 0,
        levels=levels,
        bend_right=bend_right,
    )

    labels = [[tensor._labels[n] for n in codomain], [tensor._labels[n] for n in domain]]
    return SymmetricTensor(data, new_codomain, new_domain, backend=tensor.backend, labels=labels)


def pinv(tensor: Tensor, cutoff=1e-15) -> Tensor:
    """The Moore-Penrose pseudo-inverse of a tensor."""
    if isinstance(tensor, DiagonalTensor):
        return cutoff_inverse(tensor, cutoff=cutoff)
    U, S, Vh = truncated_svd(tensor, options=dict(svd_min=cutoff))
    return dagger(U @ cutoff_inverse(S, cutoff=cutoff) @ Vh)


def qr(tensor: Tensor, new_labels: str | list[str] = None, new_leg_dual: bool = False) -> tuple[Tensor, Tensor]:
    """The QR decomposition of a tensor.

    A :ref:`tensor decomposition <decompositions>` ``tensor ~ Q @ R`` with the following
    properties:

    - ``Q`` is an isometry: ``dagger(Q) @ Q ~ eye``.
    - ``R`` has an upper triangular structure *in the coupled basis*.

    Graphically::

        |                                 │   │   │   │
        |                                ┏┷━━━┷━━━┷━━━┷┓
        |        │   │   │   │           ┃      R      ┃
        |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
        |       ┃   tensor    ┃    ==           │
        |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
        |          │   │   │             ┃      Q      ┃
        |                                ┗━━┯━━━┯━━━┯━━┛
        |                                   │   │   │

    We always compute the "reduced", a.k.a. "economic" version.
    To group the legs differently, use :func:`permute_legs` or `combine_to_matrix` first.

    Parameters
    ----------
    tensor: :class:`Tensor`
        The tensor to decompose.
    new_labels: (list of) str
        Labels for the new legs. Either two legs ``[a, b]`` s.t. ``Q.labels[-1] == a``
        and ``R.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
    new_leg_dual: bool
        If the new leg should be a ket space (``False``) or bra space (``True``)

    """
    a, b = _decomposition_labels(new_labels)
    tensor, new_co_domain, combine_codomain, combine_domain = _decomposition_prepare(tensor, new_leg_dual)
    q_data, r_data = tensor.backend.qr(tensor, new_co_domain=new_co_domain)
    Q = SymmetricTensor(
        q_data,
        codomain=tensor.codomain,
        domain=new_co_domain,
        backend=tensor.backend,
        labels=[tensor.codomain_labels, [a]],
    )
    R = SymmetricTensor(
        r_data, codomain=new_co_domain, domain=tensor.domain, backend=tensor.backend, labels=[[b], tensor.domain_labels]
    )
    if combine_codomain:
        Q = split_legs(Q, 0)
    if combine_domain:
        R = split_legs(R, -1)
    return Q, R


@_elementwise_function(block_func='real', maps_zero_to_zero=True)
def real(x: _ElementwiseType) -> _ElementwiseType:
    """The real part of a complex number, :ref:`elementwise <diagonal_elementwise>`."""
    return np.real(x)


@_elementwise_function(block_func='real_if_close', func_kwargs=dict(tol=100), maps_zero_to_zero=True)
def real_if_close(x: _ElementwiseType, tol: float = 100) -> _ElementwiseType:
    """If close to real, return the :func:`real` part, :ref:`elementwise <diagonal_elementwise>`.

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


def lq(tensor: Tensor, new_labels: str | list[str] = None, new_leg_dual: bool = False) -> tuple[Tensor, Tensor]:
    """The LQ decomposition of a tensor.

    A :ref:`tensor decomposition <decompositions>` ``tensor ~ Q @ R`` with the following
    properties:

    - ``L`` has a lower triangular structure *in the coupled basis*.
    - ``Q`` is an isometry: ``dagger(Q) @ Q ~ eye``.

    Graphically::

        |                                 │   │   │   │
        |                                ┏┷━━━┷━━━┷━━━┷┓
        |        │   │   │   │           ┃      Q      ┃
        |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
        |       ┃   tensor    ┃    ==           │
        |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
        |          │   │   │             ┃      L      ┃
        |                                ┗━━┯━━━┯━━━┯━━┛
        |                                   │   │   │

    We always compute the "reduced", a.k.a. "economic" version.
    To group the legs differently, use :func:`permute_legs` or `combine_to_matrix` first.

    Parameters
    ----------
    tensor: :class:`Tensor`
        The tensor to decompose.
    new_labels: (list of) str
        Labels for the new legs. Either two legs ``[a, b]`` s.t. ``L.labels[-1] == a``
        and ``Q.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
    new_leg_dual: bool
        If the new leg should be a ket space (``False``) or bra space (``True``)

    """
    a, b = _decomposition_labels(new_labels)
    tensor, new_co_domain, combine_codomain, combine_domain = _decomposition_prepare(tensor, new_leg_dual)
    l_data, q_data = tensor.backend.lq(tensor, new_co_domain=new_co_domain)
    L = SymmetricTensor(
        l_data,
        codomain=tensor.codomain,
        domain=new_co_domain,
        backend=tensor.backend,
        labels=[tensor.codomain_labels, [a]],
    )
    Q = SymmetricTensor(
        q_data, codomain=new_co_domain, domain=tensor.domain, backend=tensor.backend, labels=[[b], tensor.domain_labels]
    )
    if combine_codomain:
        L = split_legs(L, 0)
    if combine_domain:
        Q = split_legs(Q, -1)
    return L, Q


def scalar_multiply(a: Number, v: Tensor) -> Tensor:
    """The scalar multiplication ``a * v``"""
    if not isinstance(a, Number):
        msg = f'unsupported scalar type: {type(a).__name__}'
        raise TypeError(msg)
    if isinstance(v, DiagonalTensor):
        return DiagonalTensor._elementwise_unary(v, func=lambda _v: a * _v, maps_zero_to_zero=True)
    if isinstance(v, Mask):
        msg = (
            'Converting to SymmetricTensor for scalar multiplication. '
            'Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        v = v.as_SymmetricTensor(warning=msg)
    if isinstance(v, ChargedTensor):
        if v.charged_state is None:
            inv_part = scalar_multiply(a, v.invariant_part)
            charged_state = None
        else:
            inv_part = v.invariant_part
            charged_state = v.backend.block_backend.mul(a, v.charged_state)
        return ChargedTensor(inv_part, charged_state)
    # remaining case: SymmetricTensor
    return SymmetricTensor(
        v.backend.mul(a, v), codomain=v.codomain, domain=v.domain, backend=v.backend, labels=v._labels
    )


def scale_axis(tensor: Tensor, diag: DiagonalTensor, leg: int | str) -> Tensor:
    """Contract one `leg` of  `tensor` with a diagonal tensor.

    Leg order, labels and legs of `tensor` are not changed.
    The diagonal tensors leg ``diag.leg`` must be the same or the dual of the leg on the tensor,
    if mismatched, the `diag` is automatically transposed, as needed.

    Graphically::

        |        │   │   │            │   │  ┏┷┓
        |       ┏┷━━━┷━━━┷┓           │   │  ┃D┃
        |       ┃ tensor  ┃           │   │  ┗┯┛
        |       ┗┯━━━┯━━━┯┛    OR    ┏┷━━━┷━━━┷┓
        |        │  ┏┷┓  │           ┃ tensor  ┃
        |        │  ┃D┃  │           ┗┯━━━┯━━━┯┛
        |        │  ┗┯┛  │            │   │   │

    Or transpose as needed:

        |        │   │   │   │   │
        |       ┏┷━━━┷━━━┷━━━┷━━━┷┓            │   │   │
        |       ┃ tensor          ┃           ┏┷━━━┷━━━┷┓
        |       ┗┯━━━┯━━━━━━━━━━━┯┛           ┃ tensor  ┃
        |        │   │   ╭───╮   │      ==    ┗┯━━━┯━━━┯┛
        |        │   │  ┏┷┓  │   │             │ ┏━┷━┓ │
        |        │   │  ┃D┃  │   │             │ ┃D.T┃ │
        |        │   │  ┗┯┛  │   │             │ ┗━┯━┛ │
        |        │   ╰───╯   │   │

    where ``D.T == transpose(D)``.

    See Also
    --------
    dot, tdot, apply_mask, partial_compose

    """
    _ = get_same_device(tensor, diag)

    # transpose if needed
    in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg)
    if in_domain:
        leg = tensor.domain[co_domain_idx]
    else:
        leg = tensor.codomain[co_domain_idx]
    assert tensor.symmetry.is_same_symmetry(diag.symmetry)
    if leg == diag.leg:
        pass
    elif leg == diag.leg.dual:
        diag = transpose(diag)
    else:
        raise ValueError('Incompatible legs')

    if isinstance(tensor, DiagonalTensor):
        return (tensor * diag).set_labels(tensor.labels)
    if isinstance(tensor, Mask):
        if leg_idx == 0:
            return compose(diag, tensor).set_labels(tensor.labels)
        return compose(tensor, diag).set_labels(tensor.labels)
    if isinstance(tensor, ChargedTensor):
        inv_part = scale_axis(tensor.invariant_part, diag, leg_idx)
        return ChargedTensor(inv_part, tensor.charged_state)
    backend = get_same_backend(tensor, diag)
    data = backend.scale_axis(tensor, diag, leg_idx)
    return SymmetricTensor(data, codomain=tensor.codomain, domain=tensor.domain, backend=backend, labels=tensor._labels)


def split_legs(tensor: Tensor, legs: int | str | list[int | str] | None = None):
    r"""Split legs that were previously combined using :func:`combine_legs`.

    |       │   │   │   │   │   │
    |       ╰──┴───╥╯   │   ╰──╥╯
    |      ┏━━━━━━━┷━━━━┷━━━━━━┷━┓
    |      ┃          T          ┃    ==    split_legs(T, [2, 4, 6])
    |      ┗┯━━━┯━━━━┯━━━━━━━━━━┯┛
    |       │   │   ╭╨───┬──╮   │
    |       │   │   │   │   │   │

    This is the inverse of :func:`combine_legs`, up to a possible :func:`permute_legs`.

    Parameters
    ----------
    tensor
        The tensor to act on.
    legs: list of int | str
        Which legs to split. If ``None`` (default), all those legs that are :class:`LegPipe`\ s
        are split.

    """
    # Deal with different tensor types. Reduce everything to SymmetricTensor.
    if isinstance(tensor, (DiagonalTensor, Mask)):
        msg = (
            'Converting to SymmetricTensor for split_legs. Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        tensor = tensor.as_SymmetricTensor(warning=msg)
    if isinstance(tensor, ChargedTensor):
        if legs is not None:
            legs = tensor.get_leg_idcs(legs)
        return ChargedTensor(split_legs(tensor.invariant_part, legs), tensor.charged_state)
    #
    # parse indices
    if legs is None:
        codomain_split = [n for n, l in enumerate(tensor.codomain) if isinstance(l, LegPipe)]
        domain_split = [n for n, l in enumerate(tensor.domain) if isinstance(l, LegPipe)]
        leg_idcs = [*codomain_split, *(tensor.num_legs - 1 - n for n in domain_split)]
    else:
        leg_idcs = []
        codomain_split = []
        domain_split = []
        for l in to_iterable(legs):
            in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(l)
            leg_idcs.append(leg_idx)
            if in_domain:
                domain_split.append(co_domain_idx)
            else:
                codomain_split.append(co_domain_idx)
            if not isinstance(tensor.get_leg_co_domain(leg_idx), LegPipe):
                raise ValueError('Not a LegPipe.')
    #
    # build new (co)domain
    codomain_spaces = []
    for n, l in enumerate(tensor.codomain):
        if n in codomain_split:
            codomain_spaces.extend(l.legs)
        else:
            codomain_spaces.append(l)
    domain_spaces = []
    for n, l in enumerate(tensor.domain):
        if n in domain_split:
            domain_spaces.extend(l.legs)
        else:
            domain_spaces.append(l)

    # we only split, i.e. remove parentheses in tensor products, so sectors dont change
    codomain = TensorProduct(
        codomain_spaces,
        symmetry=tensor.symmetry,
        _sector_decomposition=tensor.codomain.sector_decomposition,
        _multiplicities=tensor.codomain.multiplicities,
    )
    domain = TensorProduct(
        domain_spaces,
        symmetry=tensor.symmetry,
        _sector_decomposition=tensor.domain.sector_decomposition,
        _multiplicities=tensor.domain.multiplicities,
    )

    #
    # build labels
    labels = []
    for n, l in enumerate(tensor.labels):
        if n in leg_idcs:
            labels.extend(_split_leg_label(l, num=tensor.get_leg_co_domain(n).num_legs))
        else:
            labels.append(l)
    #
    data = tensor.backend.split_legs(tensor, leg_idcs, codomain_split, domain_split, codomain, domain)
    return SymmetricTensor(data, codomain, domain, backend=tensor.backend, labels=labels)


@_elementwise_function(block_func='sqrt', maps_zero_to_zero=True)
def sqrt(x: _ElementwiseType) -> _ElementwiseType:
    """The square root of a number, :ref:`elementwise <diagonal_elementwise>`."""
    return np.sqrt(x)


def squeeze_legs(tensor: Tensor, legs: int | str | list[int | str] = None) -> Tensor:
    """Remove trivial legs.

    A leg counts as trivial according to :attr:`Space.is_trivial`, i.e. if it consists of a single
    copy of the trivial sector.

    Parameters
    ----------
    tensor:
        The tensor to act on.
    legs: (list of) {int | str}
        Which legs to squeeze. Squeezed legs must be trivial.
        If ``None`` (default) all trivial legs are squeezed.

    """
    if legs is None:
        legs = [n for n, l in enumerate(conventional_leg_order(tensor)) if l.is_trivial]
    else:
        legs = tensor.get_leg_idcs(legs)
        if not all(tensor.get_leg_co_domain(n).is_trivial for n in legs):
            raise ValueError('Can only squeeze trivial legs')
    if len(legs) == 0:
        return tensor
    if isinstance(tensor, (DiagonalTensor, Mask)):
        msg = (
            'Converting to SymmetricTensor for squeeze_legs. '
            'Use as_SymmetricTensor() explicitly to suppress the warning.'
        )
        tensor = tensor.as_SymmetricTensor(warning=msg)
    if isinstance(tensor, ChargedTensor):
        return ChargedTensor(squeeze_legs(tensor.invariant_part, legs=legs), tensor.charged_state)
    # Remaining case: SymmetricTensor
    remaining = [n for n in range(tensor.num_legs) if n not in legs]
    data = tensor.backend.squeeze_legs(tensor, legs)

    # the fusion with the trivial legs was trivial, so removing it doesnt change the sectors
    codomain = TensorProduct(
        [tensor.codomain[n] for n in range(tensor.num_codomain_legs) if n not in legs],
        symmetry=tensor.symmetry,
        _sector_decomposition=tensor.codomain.sector_decomposition,
        _multiplicities=tensor.codomain.multiplicities,
    )
    domain = TensorProduct(
        [tensor.domain[n] for n in range(tensor.num_domain_legs) if (tensor.num_legs - 1 - n) not in legs],
        symmetry=tensor.symmetry,
        _sector_decomposition=tensor.domain.sector_decomposition,
        _multiplicities=tensor.domain.multiplicities,
    )
    return SymmetricTensor(
        data, codomain, domain, backend=tensor.backend, labels=[tensor._labels[n] for n in remaining]
    )


@_elementwise_function(block_func='stable_log', func_kwargs=dict(cutoff=1e-30), maps_zero_to_zero=True)
def stable_log(x: _ElementwiseType, cutoff=1e-30) -> _ElementwiseType:
    """Stabilized logarithm, :ref:`elementwise <diagonal_elementwise>`.

    For values ``> cutoff``, this is the standard natural logarithm. For values smaller than the
    cutoff, return 0.
    """
    assert cutoff > 0
    return np.where(x > cutoff, np.log(x), 0.0)


def svd(
    tensor: Tensor,
    new_labels: str | list[str] | None = None,
    new_leg_dual: bool = False,
    algorithm: str | None = None,
) -> tuple[Tensor, DiagonalTensor, Tensor]:
    """The singular value decomposition (SVD) of a tensor.

    A :ref:`tensor decomposition <decompositions>` ``tensor ~ U @ S @ Vh`` with the following
    properties:

    - ``Vh`` and ``U``are isometries: ``dagger(U) @ U ~ eye ~ Vh @ dagger(Vh)``.
    - ``S`` is a :class:`DiagonalTensor` with real, non-negative entries.
    - If `tensor` is a matrix (i.e. if it has exactly one leg each in domain and codomain), it
      reproduces the usual matrix SVD.

    .. note ::
        The basis for the newly generated leg is chosen arbitrarily, and in particular unlike
        e.g. :func:`numpy.linalg.svd` it is not guaranteed that ``S.diag_numpy`` is sorted.

    Graphically::

        |                                 │   │   │   │
        |                                ┏┷━━━┷━━━┷━━━┷┓
        |                                ┃      Vh     ┃
        |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
        |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
        |       ┃   tensor    ┃    ==         ┃ S ┃
        |       ┗━━┯━━━┯━━━┯━━┛               ┗━┯━┛
        |          │   │   │             ┏━━━━━━┷━━━━━━┓
        |                                ┃      U      ┃
        |                                ┗━━┯━━━┯━━━┯━━┛
        |                                   │   │   │

    We always compute the "reduced", a.k.a. "economic" version of SVD, where the isometries are
    (in general) not full unitaries.

    To group the legs differently, use :func:`permute_legs` or `combine_to_matrix` first.

    Parameters
    ----------
    tensor: :class:`Tensor`
        The tensor to decompose.
    new_labels: (list of) str, optional
        The labels for the new legs can be specified in the following three ways;
        Four labels ``[a, b, c, d]`` result in ``U.labels[-1] == a``, ``S.labels == [b, c]`` and
        ``Vh.labels[0] == d``.
        Two labels ``[a, b]`` are equivalent to ``[a, b, a, b]``.
        A single label ``a`` is equivalent to ``[a, a*, a, a*]``.
        The new legs are unlabelled by default.
    new_leg_dual: bool
        If the new leg should be a ket space (``False``) or bra space (``True``)
    algorithm: str, optional
        The algorithm (a.k.a. "driver") for the block-wise svd. Choices are backend-specific.
        See the :attr:`~cyten.backends.BlockBackend.svd_algorithms` attribute of the
        ``tensor.backend.block_backend``.

    Returns
    -------
    U: SymmetricTensor
    S: DiagonalTensor
    Vh: SymmetricTensor

    """
    a, b, c, d = _svd_new_labels(new_labels)
    tensor, new_co_domain, combine_codomain, combine_domain = _decomposition_prepare(tensor, new_leg_dual)
    u_data, s_data, vh_data = tensor.backend.svd(tensor, new_co_domain=new_co_domain, algorithm=algorithm)
    U = SymmetricTensor(
        u_data,
        codomain=tensor.codomain,
        domain=new_co_domain,
        backend=tensor.backend,
        labels=[tensor.codomain_labels, [a]],
    )
    S = DiagonalTensor(s_data, leg=new_co_domain[0], backend=tensor.backend, labels=[b, c])
    Vh = SymmetricTensor(
        vh_data,
        codomain=new_co_domain,
        domain=tensor.domain,
        backend=tensor.backend,
        labels=[[d], tensor.domain_labels],
    )
    # split legs, if they were previously combined
    if combine_codomain:
        U = split_legs(U, 0)
    if combine_domain:
        Vh = split_legs(Vh, -1)
    return U, S, Vh


def svd_apply_mask(
    U: SymmetricTensor, S: DiagonalTensor, Vh: SymmetricTensor, mask: Mask
) -> tuple[SymmetricTensor, DiagonalTensor, SymmetricTensor]:
    """Truncate an existing SVD"""
    assert mask.is_projection
    assert mask.domain[0] == S.domain[0]

    U = _compose_with_Mask(U, dagger(mask), -1)
    S = apply_mask_DiagonalTensor(S, mask)
    Vh = _compose_with_Mask(Vh, mask, 0)
    return U, S, Vh


def tensor_from_grid(
    grid: list[list[SymmetricTensor | None]],
    labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
    dtype: Dtype | None = None,
) -> SymmetricTensor:
    r"""Stack a grid of tensors along existing legs.

    The tensors are stacked along the first leg in their codomain and the final leg in their
    domain. The resulting legs are :math:`result.codomain[0] = V = \bigoplus_m V_m` and
    :math:`result.domain[-1] = W = \bigoplus_n W_n`, where :math:`V_m` is the first codomain leg
    of all tensors in the ``m``-th row ``grid[m]``, and :math:`W_n` is the last domain leg of all
    tensors in the ``n``-th column, i.e. for the tensors ``[row[n] for row in grid]``.

    Graphically::

        |                                                      W
        |                                              │   │ ┏━┷━┓
        |                                              │   │ ┃p_n┃
        |                  W                           │   │ ┗━┯━┛
        |          │   │   │                           │   │   │ W_n
        |       ┏━━┷━━━┷━━━┷━━┓                     ┏━━┷━━━┷━━━┷━━┓
        |       ┃     res     ┃    ==   sum_{m,n}   ┃  grid[m][n] ┃
        |       ┗┯━━━┯━━━┯━━━┯┛                     ┗┯━━━┯━━━┯━━━┯┛
        |        │   │   │   │                   V_m │   │   │   │
        |        V                                 ┏━┷━┓ │   │   │
        |                                          ┃i_m┃ │   │   │
        |                                          ┗━┯━┛ │   │   │
        |

    where :math:`p_n : W = \bigoplus_{n'} W_{n'} \to W_n` is the projection map of the direct sum
    and :math:`i_m : V_m \to \bigoplus_{m'} V_{m'}` the inclusion.

    Parameters
    ----------
    grid: list[list[SymmetricTensor | None]]
        Contains the tensors from which a single tensor is constructed by stacking. `None` entries
        are interpreted as tensors with all blocks equal to zero. All legs except the ones along
        which the stacking happens must be identical across all tensors in the grid. For
        consistency, tensors within the same row must have identical left spaces (first leg in the
        codomain), and tensors within the same column must have identical right spaces (final leg
        in the domain).
    labels
        Leg labels of the resulting tensor.
    dtype: Dtype | None
        The dtype of the tensor. Uses the common dtype across all tensors in the grid if `None`.

    """
    op_list = [op for row in grid for op in row if op is not None]
    backend = get_same_backend(*op_list)
    device = get_same_device(*op_list)
    if dtype is None:
        dtype = op_list[0].dtype.common(*[op_list[i].dtype for i in range(1, len(op_list))])
    # check input
    for op in op_list:
        assert op.num_codomain_legs == op_list[0].num_codomain_legs
        assert op.num_domain_legs == op_list[0].num_domain_legs
        assert op.codomain[1:] == op_list[0].codomain[1:]
        assert op.domain[:-1] == op_list[0].domain[:-1]
        # only ElementarySpaces have direct_sum
        assert isinstance(op.codomain[0], ElementarySpace)
        assert isinstance(op.domain[-1], ElementarySpace)

    transposed_grid = list(map(list, zip(*grid)))
    right_ops = grid[0][:]
    for i, op in enumerate(right_ops):
        if op is not None:
            continue
        # find op from same column
        for new_op in transposed_grid[i]:
            if new_op is None:
                continue
            right_ops[i] = new_op
            break
    if any(op is None for op in right_ops):
        raise ValueError('Must have at least one nonzero entry in each column.')
    right_spaces = [op.domain[-1] for op in right_ops]

    left_ops = transposed_grid[0]
    for i, op in enumerate(left_ops):
        if op is not None:
            continue
        # find op from same row
        for new_op in grid[i]:
            if new_op is None:
                continue
            left_ops[i] = new_op
            break
    if any(op is None for op in left_ops):
        raise ValueError('Must have at least one nonzero entry in each row.')
    left_spaces = [op.codomain[0] for op in left_ops]

    left_space = left_spaces[0].direct_sum(*left_spaces[1:])
    right_space = right_spaces[0].direct_sum(*right_spaces[1:])

    # for each sector in the direct sum, find which multiplicities come from which space
    left_mult_slices = []
    for sector in left_space.sector_decomposition:
        mults = []
        for space in left_spaces:
            idx = space.sector_decomposition_where(sector)
            mult = 0 if idx is None else space.multiplicities[idx]
            mults.append(mult)
        left_mult_slices.append(np.concatenate([[0], np.cumsum(mults)], axis=0))
    right_mult_slices = []
    for sector in right_space.sector_decomposition:
        mults = []
        for space in right_spaces:
            idx = space.sector_decomposition_where(sector)
            mult = 0 if idx is None else space.multiplicities[idx]
            mults.append(mult)
        right_mult_slices.append(np.concatenate([[0], np.cumsum(mults)], axis=0))

    codomain = TensorProduct([left_space, *op_list[0].codomain[1:]])
    domain = TensorProduct([*op_list[0].domain[:-1], right_space])
    data = backend.from_grid(
        grid=grid,
        new_codomain=codomain,
        new_domain=domain,
        left_mult_slices=left_mult_slices,
        right_mult_slices=right_mult_slices,
        dtype=dtype,
        device=device,
    )
    return SymmetricTensor(data, codomain=codomain, domain=domain, backend=backend, labels=labels)


def tdot(
    tensor1: Tensor,
    tensor2: Tensor,
    legs1: int | str | list[int | str],
    legs2: int | str | list[int | str],
    relabel1: dict[str, str] = None,
    relabel2: dict[str, str] = None,
):
    """General tensor contraction, connecting arbitrary pairs of (matching!) legs.

    For example::

        |    ╭───╮   ╭───│───│──╮
        |    │   4   3   2   │  │
        |    │  ┏┷━━━┷━━━┷┓  │  │
        |    │  ┃    B    ┃  │  │
        |    │  ┗━━┯━━━┯━━┛  │  │
        |    │     0   1     │  │
        |    │     │   ╰─────╯  │    ==    tdot(A, B, [1, 4, 5], [3, 0, 4])
        |    ╰───╮ ╰─╮   ╭───╮  │
        |        5   4   3   │  │
        |       ┏┷━━━┷━━━┷┓  │  │
        |       ┃    A    ┃  │  │
        |       ┗┯━━━┯━━━┯┛  │  │
        |        0   1   2   │  │
        |        │   ╰───│───│──╯

    Parameters
    ----------
    tensor1, tensor2: Tensor
        The two tensors to contract.
    legs1, legs2
        Which legs to contract: ``legs1[n]`` on `tensor1` is contracted with ``legs2[n]`` on
        `tensor2`.
    relabel1, relabel2: dict[str, str], optional
        A mapping of labels for each of the tensors. The result has labels, as if the
        input tensors were relabelled accordingly before contraction.

    Returns
    -------
    A tensor given by the contraction.
    Its domain is formed by the uncontracted legs of `tensor2`, in *inverse* order and with
    *opposite* duality compared to ``tensor2.legs``, i.e. like they were all in ``tensor2.domain``.
    Its codomain, conversely, is given by the uncontracted legs of `tensor1`, in the same order
    and with the same duality as in ``tensor1.legs``, i.e. like they were all in ``tensor1.codomain``.
    Therefore, the ``result.legs`` are the uncontracted from ``tensor1.legs``, followed by the
    uncontracted ``tensor2.legs``.

    See Also
    --------
    compose, partial_compose, apply_mask, scale_axis

    """
    _ = get_same_device(tensor1, tensor2)

    # parse legs to list[int] and check they are valid
    legs1 = tensor1.get_leg_idcs(legs1)
    legs2 = tensor2.get_leg_idcs(legs2)
    if duplicate_entries(legs1) or duplicate_entries(legs2):
        raise ValueError(f'Duplicate leg entries.')
    num_contr = len(legs1)
    assert len(legs2) == num_contr
    _check_compatible_legs(
        [tensor1._as_domain_leg(i1) for i1 in legs1],
        [tensor2._as_codomain_leg(i2) for i2 in legs2],
    )

    # Deal with Masks: either return or reduce to SymmetricTensor
    if isinstance(tensor1, Mask):
        if num_contr == 0:
            tensor1 = tensor1.as_SymmetricTensor()
        if num_contr == 1:
            t1_in_domain = legs1[0] == 1
            t2_in_domain = legs2[0] >= tensor2.num_codomain_legs
            if t2_in_domain == t1_in_domain:
                res = _compose_with_Mask(tensor2, transpose(tensor1), legs2[0])
            else:
                res = _compose_with_Mask(tensor2, tensor1, legs2[0])
            res.set_label(legs2[0], tensor1.labels[1 - legs1[0]])
            # move legs to tdot convention
            return permute_legs(res, codomain=legs1)
        if num_contr == 2:
            # contract the large leg first
            which_is_large = legs1.index(1 if tensor1.is_projection else 0)
            t1_in_domain = tensor1.is_projection
            t2_in_domain = legs2[which_is_large] >= tensor2.num_codomain_legs
            if t1_in_domain == t2_in_domain:
                res = _compose_with_Mask(tensor2, transpose(tensor1), legs2[which_is_large])
            else:
                res = _compose_with_Mask(tensor2, tensor1, legs2[which_is_large])
            # then trace over the small leg
            res = partial_trace(res, legs2)
            # move legs to tdot convention
            if tensor2.num_legs == 2:  # scalar result
                return res
            return bend_legs(res, num_codomain_legs=0)
    if isinstance(tensor2, Mask):
        if num_contr == 0:
            tensor2 = tensor2.as_SymmetricTensor()
        if num_contr == 1:
            t1_in_domain = legs1[0] >= tensor1.num_codomain_legs
            t2_in_domain = legs2[0] == 1
            if t1_in_domain == t2_in_domain:
                res = _compose_with_Mask(tensor1, transpose(tensor2), legs1[0])
            else:
                res = _compose_with_Mask(tensor1, tensor2, legs1[0])
            res.set_label(legs1[0], tensor2.labels[1 - legs2[0]])
            # move legs to tdot convention
            try:
                return permute_legs(res, domain=legs2, bend_right=None)
            except SymmetryError:
                raise SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG) from None
        if num_contr == 2:
            # contract the large leg first
            which_is_large = legs2.index(1 if tensor2.is_projection else 0)
            t1_in_domain = legs1[which_is_large] >= tensor1.num_codomain_legs
            t2_in_domain = tensor2.is_projection
            if t1_in_domain == t2_in_domain:
                res = _compose_with_Mask(tensor1, transpose(tensor2), legs1[which_is_large])
            else:
                res = _compose_with_Mask(tensor1, tensor2, legs1[which_is_large])
            # then trace over the small leg
            res = partial_trace(res, legs1)
            # move legs to tdot convention
            if tensor1.num_legs == 2:  # scalar result
                return res
            return bend_legs(res, num_domain_legs=0)

    # Deal with DiagonalTensor: either return or reduce to SymmetricTensor
    if isinstance(tensor1, DiagonalTensor):
        if num_contr == 0:
            tensor1 = tensor1.as_SymmetricTensor()
        if num_contr == 1:
            res = scale_axis(tensor2, tensor1, legs2[0])
            res.set_label(legs2[0], tensor1.labels[1 - legs1[0]])
            try:
                return permute_legs(res, codomain=legs1)
            except SymmetryError:
                raise SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG) from None
        if num_contr == 2:
            res = scale_axis(tensor2, tensor1, legs2[0])
            res = partial_trace(res, legs2)
            if tensor2.num_legs == 2:  # scalar result
                return res
            return bend_legs(res, num_codomain_legs=0)
    if isinstance(tensor2, DiagonalTensor):
        if num_contr == 0:
            tensor2 = tensor2.as_SymmetricTensor()
        if num_contr == 1:
            res = scale_axis(tensor1, tensor2, legs1[0])
            res.set_label(legs1[0], tensor2.labels[1 - legs2[0]])
            try:
                return permute_legs(res, domain=legs1)
            except SymmetryError:
                raise SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG) from None
        if num_contr == 2:
            res = scale_axis(tensor1, tensor2, legs1[0])
            res = partial_trace(res, legs1)
            if tensor1.num_legs == 2:  # scalar result
                return res
            return bend_legs(res, num_domain_legs=0)

    # Deal with ChargedTensor
    if isinstance(tensor1, ChargedTensor) and isinstance(tensor2, ChargedTensor):
        # note: its important that we have already used get_leg_idcs
        if (tensor1.charged_state is None) != (tensor2.charged_state is None):
            raise ValueError('Mismatched: specified and unspecified ChargedTensor.charged_state')
        c = ChargedTensor._CHARGE_LEG_LABEL
        c1 = c + '1'
        c2 = c + '2'
        relabel1 = {c: c1} if relabel1 is None else {**relabel1, c: c1}
        relabel2 = {c: c2} if relabel2 is None else {**relabel2, c: c2}
        inv_part = tdot(
            tensor1.invariant_part,
            tensor2.invariant_part,
            legs1=legs1,
            legs2=legs2,
            relabel1=relabel1,
            relabel2=relabel2,
        )
        inv_part = move_leg(inv_part, c1, domain_pos=0)
        return ChargedTensor.from_two_charge_legs(
            inv_part,
            state1=tensor1.charged_state,
            state2=tensor2.charged_state,
        )
    if isinstance(tensor1, ChargedTensor):
        inv_part = tdot(tensor1.invariant_part, tensor2, legs1=legs1, legs2=legs2, relabel1=relabel1, relabel2=relabel2)
        inv_part = move_leg(inv_part, ChargedTensor._CHARGE_LEG_LABEL, domain_pos=0)
        return ChargedTensor.from_invariant_part(inv_part, tensor1.charged_state)
    if isinstance(tensor2, ChargedTensor):
        inv_part = tdot(tensor1, tensor2.invariant_part, legs1=legs1, legs2=legs2, relabel1=relabel1, relabel2=relabel2)
        return ChargedTensor.from_invariant_part(inv_part, tensor2.charged_state)

    # Remaining case: both are SymmetricTenor

    # OPTIMIZE actually, we only need to permute legs to *any* matching order.
    #          could use ``legs1[perm]`` and ``legs2[perm]`` instead, if that means fewer braids.
    try:
        tensor1 = permute_legs(tensor1, domain=legs1, bend_right=None)
        tensor2 = permute_legs(tensor2, codomain=legs2, bend_right=None)
    except SymmetryError as e:
        raise SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG) from e
    return _compose_SymmetricTensors(tensor1, tensor2, relabel1=relabel1, relabel2=relabel2)


def trace(tensor: Tensor):
    """Perform the trace.

    Requires that ``tensor.domain == tensor.codomain`` and perform the full trace::

        |    ╭───────────────╮
        |    │   ╭─────────╮ │
        |    │   │   ╭───╮ │ │
        |   ┏┷━━━┷━━━┷┓  │ │ │
        |   ┃    A    ┃  │ │ │    ==    trace(A)
        |   ┗┯━━━┯━━━┯┛  │ │ │
        |    │   │   ╰───╯ │ │
        |    │   ╰─────────╯ │
        |    ╰───────────────╯

    Parameters
    ----------
    tensor: Tensor
        The tensor to trace on

    Returns
    -------
    A python scalar, the trace.

    See Also
    --------
    partial_trace
        Trace only some legs, or trace all legs with a different connectivity.

    """
    _check_compatible_legs([tensor.domain], [tensor.codomain])
    if isinstance(tensor, DiagonalTensor):
        return tensor.backend.diagonal_tensor_trace_full(tensor)
    if isinstance(tensor, ChargedTensor):
        if tensor.charged_state is None:
            raise ValueError('Need to specify charged_state for full trace of ChargedTensor')
        # OPTIMIZE can project to trivial sector on charge leg first
        N = tensor.num_legs
        pairs = [[n, N - 1 - n] for n in range(tensor.num_codomain_legs)]
        inv_block = partial_trace(tensor.invariant_part, *pairs)
        inv_block = inv_block.to_dense_block(understood_braiding=True)
        res = tensor.backend.block_backend.tdot(inv_block, tensor.charged_state, [0], [0])
        return tensor.backend.block_backend.item(res)
    return tensor.backend.trace_full(tensor)


def transpose(tensor: Tensor) -> Tensor:
    r"""The transpose of a tensor.

    For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
    the transpose matrix :math:`(M^\text{T})_{i,j} = M_{j, i}` .
    For a map :math:`f: V \to W`, the transpose is a map :math:`f: W^* \to V^*`::

    |          │   │   │             ╭───────────╮
    |          │   │   │             │ ╭─────╮   │     │ │ │
    |       ┏━━┷━━━┷━━━┷━━┓          │ │  ┏━━┷━━━┷━━┓  │ │ │
    |       ┃transpose(A) ┃    ==    │ │  ┃    A    ┃  │ │ │
    |       ┗━━━━┯━━━┯━━━━┛          │ │  ┗┯━━━┯━━━┯┛  │ │ │
    |            │   │               │ │   │   │   ╰───╯ │ │
    |            │   │               │ │   │   ╰─────────╯ │
    |            │   │               │ │   ╰───────────────╯

    Returns
    -------
    The transposed tensor. Its legs and labels fulfill e.g.::

        transpose(A).codomain == A.domain.dual == [W2.dual, W1.dual]  # if A.domain == [W1, W2]
        transpose(A).domain == A.codomain.dual == [V2.dual, V1.dual]  # if A.codomain == [V1, V2]
        transpose(A).legs == [W2.dual, W1.dual, V1, V2]  # compared to A.legs == [V1, V2, W2.dual, W1.dual]
        transpose(A).labels == [*reversed(A.domain_labels), *A.codomain_labels]

    Note that the resulting :attr:`Tensor.legs` depend not only on the input :attr:`Tensor.legs`,
    but also on how they are partitioned into domain and codomain.
    We use the "same" labels, up to the permutation.

    """
    labels = [*reversed(tensor.domain_labels), *tensor.codomain_labels]
    if isinstance(tensor, Mask):
        space_in, space_out, data = tensor.backend.mask_transpose(tensor)
        return Mask(
            data,
            space_in=space_in,
            space_out=space_out,
            is_projection=not tensor.is_projection,
            backend=tensor.backend,
            labels=labels,
        )
    if isinstance(tensor, DiagonalTensor):
        dual_leg, data = tensor.backend.diagonal_transpose(tensor)
        return DiagonalTensor(data=data, leg=dual_leg, backend=tensor.backend, labels=labels)
    if isinstance(tensor, SymmetricTensor):
        return permute_legs(
            tensor,
            codomain=[*range(tensor.num_codomain_legs, tensor.num_legs)],
            domain=[*reversed(range(tensor.num_codomain_legs))],
            bend_right=[False] * tensor.num_codomain_legs + [True] * tensor.num_domain_legs,
        )
    if isinstance(tensor, ChargedTensor):
        if tensor.symmetry.braiding_style > BraidingStyle.bosonic:
            msg = (
                f'transpose is not defined for ChargedTensors with fermionic symmetries. '
                f'This is because there is no way to recover the ChargedTensor format in such a '
                f'way that transposing twice gives back the original tensor. '
                f'Use permute_legs instead'
            )
            raise SymmetryError(msg)
        inv_part = transpose(tensor.invariant_part)
        inv_part = move_leg(inv_part, ChargedTensor._CHARGE_LEG_LABEL, domain_pos=0)
        return ChargedTensor(inv_part, tensor.charged_state)
    raise TypeError


def truncate_singular_values(
    S: DiagonalTensor,
    chi_max: int = None,
    chi_min: int = 1,
    degeneracy_tol: float = 0,
    trunc_cut: float = 0,
    svd_min: float = 0,
    minimize_error: bool = True,
    mask_labels: list[str] = None,
) -> tuple[Mask, float, float]:
    r"""Given *normalized* singular values, determine which to keep.

    Parameters
    ----------
    S : DiagonalTensor
        Singular values, normalized to ``S.norm() == 1.``.
    chi_max : int, optional
        Keep at most this many singular values. ``None`` means no constraint.
    chi_min : int, optional
        Keep at least this many singular values. Comments for `chi_max` apply.
    degeneracy_tol : float
        Do not split (nearly) degenerate singular values.
        We count ``S[i]`` and ``S[j]`` as nearly degenerate
        if ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
        if ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``.
        In that case, either both are kept or both are truncated.
    trunc_cut : float
        A *lower* bound on the incurred truncation error, meaning as long as the error remains
        below this threshold, singular values will be truncated.
        In particular, this means as long as ``sum_{i discarded} d[i] S[i] ** 2 <= trunc_cut ** 2``,
        where ``d[i]`` is the quantum dimension (always one for abelian symmetries).
    svd_min : float
        Discard all singular values below this threshold ``S[i] < svd_min``.
        This is intended to exclude singular values that can not be distinguished from zero at the
        given precision. It does *not* have a direct implication on the resulting truncation error.
        Use `trunc_cut` instead for setting a tolerable error. See notes below for details.
    minimize_error : bool
        If we should minimize the resulting truncation error by keeping as many singular values
        as allowed by the other constraints. Otherwise we keep as few as possible.
    mask_labels : list of str, optional
        The labels for the `mask`. Either a list of two string labels or ``None`` (default).
        By default, the `mask` has labels ``[S.labels[0], dual_label(S.labels[0])]``.

    Returns
    -------
    mask : Mask
        A mask, indicating that ``S[mask]`` are the singular values to keep.
    err : float
        The truncation error ``trunc_err == norm(S_trunc) == norm(S[mask.logical_not()])``.
        This is the distance ``norm(S - S[mask])`` between the original singular values and the
        *un-normalized* approximation.
    new_norm : float
        The norm ``norm(S[mask])`` of the approximation.

    Notes
    -----
    In the case of non-Abelian group symmetries, the quantum dimensions need to be considered when
    truncating. Each independent "entry" :math:`S_i` in `S` is associated with a sector :math:`a`
    of the symmetry, e.g. the spin 1 sector of an SU(2) symmetry. It represents an entire multiplet
    (e.g. a triplet in the spin 1 case) of degenerate singular values in that charge sector, with
    the degeneracy given by the quantum dimension :math:`d_a` of the sector. When converting to a
    non-symmetric representation, e.g. via ``S.diagonal_to_numpy()``, that value :math:`S_i` will
    appear :math:`d_a` times. In particular, the error that we get by truncating some of the
    :math:`S_i` is given by :math:`\epsilon = \sum_{i discarded} d_{a_i} S_i^2`, such that the
    quantum dimensions need to be considered when choosing which singular values to keep for an
    optimal truncation error.

    This is why the singular values are prioritized by largest :math:`d_{a_i} S_i^2`, and why
    the quantum dimensions appear as a part

    For anyonic symmetries we lose the interpretation as a multiplet, since :math:`d_a` is in
    general not integer, but the formula for the error holds, and the considerations for selecting
    which singular values to keep applies just the same.
    For abelian groups or for fermions these considerations become trivial, since all sectors are
    one-dimensional.

    """
    assert S.dtype.is_real
    mask_data, new_leg, err, new_norm = S.backend.truncate_singular_values(
        S,
        chi_max=chi_max,
        chi_min=chi_min,
        degeneracy_tol=degeneracy_tol,
        trunc_cut=trunc_cut,
        svd_min=svd_min,
        minimize_error=minimize_error,
    )
    if mask_labels is None:
        mask_labels = [S.labels[0], _dual_leg_label(S.labels[0])]
    mask = Mask(mask_data, space_in=S.leg, space_out=new_leg, is_projection=True, backend=S.backend, labels=mask_labels)
    return mask, err, new_norm


def truncated_svd(
    tensor: Tensor,
    new_labels: str | list[str] | None = None,
    new_leg_dual: bool = False,
    algorithm: str | None = None,
    normalize_to: float = None,
    chi_max: int = None,
    chi_min: int = 1,
    degeneracy_tol: float = 0,
    trunc_cut: float = 0,
    svd_min: float = 0,
) -> tuple[Tensor, DiagonalTensor, Tensor, float, float]:
    """Truncated version of :func:`svd`.

    Parameters
    ----------
    tensor, new_labels, new_leg_dual, algorithm
        Same as for the non-truncated :func:`svd`.
    normalize_to: float or None
        If ``None`` (default), the resulting singular values are not renormalized,
        resulting in an approximation in terms of ``U, S, Vh`` which has smaller norm than `a`.
        If a ``float``, the singular values are scaled such that ``norm(S) == normalize_to``.
    chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min
        Options for truncations, see documentation of :func:`truncate_singular_values`.

    Returns
    -------
    U, S, Vh
        The tensors U, S, Vh that form the truncated SVD, such that
        ``tdot(U, tdot(S, Vh, 1, 0), -1, 0)`` is *approximately* equal to `a`.
    err : float
        The relative 2-norm truncation error ``norm(a - U_S_Vh) / norm(a)``.
        This is the (relative) 2-norm weight of the discarded singular values.
    renormalize : float
        Factor, by which `S` was renormalized, i.e. ``norm(S) / norm(a)``, such that
        ``U @ S @ Vh / renormalize`` has the same norm as `a`.

    See Also
    --------
    svd

    """
    U, S, Vh = svd(tensor, new_labels=new_labels, new_leg_dual=new_leg_dual, algorithm=algorithm)
    S_norm = norm(S)
    mask, err, new_norm = truncate_singular_values(
        S / S_norm,
        chi_max=chi_max,
        chi_min=chi_min,
        degeneracy_tol=degeneracy_tol,
        trunc_cut=trunc_cut,
        svd_min=svd_min,
    )
    U, S, Vh = svd_apply_mask(U, S, Vh, mask)
    if normalize_to is None:
        renormalize = 1
    else:
        # norm(S[mask]) == S_norm * new_norm
        renormalize = normalize_to / S_norm / new_norm
        S = renormalize * S
    return U, S, Vh, err, renormalize


def zero_like(tensor: Tensor) -> Tensor:
    """Return a zero tensor with same type, dtype, legs, backend and labels."""
    if isinstance(tensor, Mask):
        return Mask.from_zero(
            large_leg=tensor.large_leg, backend=tensor.backend, labels=tensor.labels, device=tensor.device
        )
    if isinstance(tensor, DiagonalTensor):
        return DiagonalTensor.from_zero(
            leg=tensor.leg, backend=tensor.backend, labels=tensor.labels, dtype=tensor.dtype, device=tensor.device
        )
    if isinstance(tensor, SymmetricTensor):
        return SymmetricTensor.from_zero(
            codomain=tensor.codomain,
            domain=tensor.domain,
            backend=tensor.backend,
            labels=tensor.labels,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    if isinstance(tensor, ChargedTensor):
        return ChargedTensor.from_zero(
            codomain=tensor.codomain,
            domain=tensor.domain,
            charge=tensor.charge_leg,
            charged_state=tensor.charged_state,
            backend=tensor.backend,
            labels=tensor.labels,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    raise TypeError


# INTERNAL HELPER FUNCTIONS


T = TypeVar('T')


def _check_compatible_legs(legs1: Sequence[Leg], legs2: Sequence[Leg], expect_equal: bool = True):
    """Check if legs are compatible (equal if `expect_equal`, otherwise mutually dual)."""
    if len(legs1) != len(legs2):
        raise ValueError('Different number of legs')
    for l1, l2 in zip(legs1, legs2):
        if not l1.symmetry.is_same_symmetry(l2.symmetry):
            raise ValueError('Different symmetries')
        compatible = l1 == (l2 if expect_equal else l2.dual)
        if not compatible:
            raise ValueError('Incompatible legs.')


def _combine_leg_labels(labels: list[str | None]) -> str:
    """The label that a combined leg should have"""
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


def _decomposition_prepare(tensor: Tensor, new_leg_dual: bool) -> tuple[SymmetricTensor, ElementarySpace, bool, bool]:
    """Common steps to prepare a SymmetricTensor before a decomposition"""
    assert tensor.num_codomain_legs > 0, 'empty codomain'
    assert tensor.num_domain_legs > 0, 'empty domain'

    if isinstance(tensor, ChargedTensor):
        # do not define decompositions for ChargedTensors.
        raise NotImplementedError
    tensor = tensor.as_SymmetricTensor()

    new_leg = ElementarySpace.from_largest_common_subspace(tensor.codomain, tensor.domain, is_dual=new_leg_dual)
    new_co_domain = TensorProduct([new_leg])
    if tensor.backend.can_decompose_tensors:
        combine_codomain = combine_domain = False
    else:
        combine_codomain = tensor.num_codomain_legs > 1
        combine_domain = tensor.num_domain_legs > 1
        if combine_codomain and combine_domain:
            tensor = combine_legs(
                tensor, range(tensor.num_codomain_legs), range(tensor.num_codomain_legs, tensor.num_legs)
            )
        elif combine_codomain:
            tensor = combine_legs(tensor, range(tensor.num_codomain_legs))
        elif combine_domain:
            tensor = combine_legs(tensor, range(tensor.num_codomain_legs, tensor.num_legs))
    return tensor, new_co_domain, combine_codomain, combine_domain


def _decomposition_labels(new_labels: str | None | list[str]) -> tuple[str, str]:
    new_labels = to_iterable(new_labels)
    if len(new_labels) == 1:
        a = new_labels[0]
        b = _dual_leg_label(a)
    elif len(new_labels) == 2:
        a, b = new_labels
    else:
        raise ValueError(f'Expected 1 or 2 labels. Got {len(new_labels)}')
    return a, b


def _dual_label_list(labels: list[str | None]) -> list[str | None]:
    return [_dual_leg_label(l) for l in reversed(labels)]


def _dual_leg_label(label: str | None) -> str | None:
    """The label that a leg should have after conjugation"""
    if label is None:
        return None
    if label.startswith('(') and label.endswith(')'):
        return _combine_leg_labels(_dual_label_list(_split_leg_label(label)))
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def _get_matching_labels(labels1: list[str | None], labels2: list[str | None], stacklevel: int = 1) -> list[str | None]:
    """Utility function to combine two lists of labels that should match.

    Per pair of labels::
        - If one is ``None``, use the other.
        - If they are equal, use that label.
        - If they are different, emit DEBUG message to the logger and choose ``None``.
          ``stacklevel=1`` refers to the line that calls this function. Increment to skip to
          higher frames.
    """
    labels = []
    conflicts = []
    for n, (l1, l2) in enumerate(zip(labels1, labels2)):
        if l1 is None:
            labels.append(l2)
        elif (l2 is None) or (l1 == l2):
            labels.append(l1)
        else:
            conflicts.append(n)
            labels.append(None)
    if conflicts:
        msg = f'Conflicting labels at positions {", ".join(map(str, conflicts))} are dropped. {labels1=}, {labels2=}.'
        logger.debug(msg, stacklevel=stacklevel + 1)
    return labels


def is_valid_leg_label(label) -> bool:
    """If the given string is a valid leg label."""
    if label is None:
        return True
    if not isinstance(label, str):
        return False
    # TODO extend: check for valid syntax of combined / conjugated labels?
    if any(f in label for f in FORBIDDEN_LEG_LABEL_CHARS):
        return False
    return True


def _parse_idcs(idcs: T | Sequence[T], length: int, fill: T = slice(None, None, None)) -> list[T]:
    """Parse a single index or sequence of indices to a list of given length.

    Ellipsis (``...``) and missing entries at the back are filled in using `fill`.

    For invalid input, an IndexError is raised instead of ValueError, since this is a helper
    function for __getitem__ and __setitem__.
    """
    idcs = list(to_iterable(idcs))
    if Ellipsis in idcs:
        where = idcs.index(Ellipsis)
        first = idcs[:where]
        last = idcs[where + 1 :]
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


def _split_leg_label(label: str | None, num: int = None) -> list[str | None]:
    """Undo _combine_leg_labels, i.e. recover the original labels"""
    if label is None:
        assert num is not None
        return [None] * num
    if label.startswith('(') and label.endswith(')'):
        labels = label[1:-1].split('.')
        assert num is None or len(labels) == num
        return [None if l.startswith('?') else l for l in labels]
    else:
        raise ValueError('Invalid format for a combined label')


def _svd_new_labels(new_labels: str | Sequence[str]) -> tuple[str, str, str, str]:
    """Parse label for :func:`svd`."""
    if new_labels is None:
        a = b = c = d = None
    else:
        new_labels = to_iterable(new_labels)
        if len(new_labels) == 1:
            a = c = new_labels[0]
            b = d = _dual_leg_label(new_labels[0])
        elif len(new_labels) == 2:
            a = c = new_labels[0]
            b = d = new_labels[1]
        elif len(new_labels) == 4:
            a, b, c, d = new_labels
        else:
            raise ValueError(f'Expected 1, 2 or 4 new_labels. Got {len(new_labels)}')
        assert (b is None) or b != c
    return a, b, c, d
