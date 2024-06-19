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


.. _conj_and_transpose

Conjugation and Transposition
-----------------------------

TODO should this be here or in the docstrings of the respective functions?

TODO elaborate on the differences between dagger and conj etc.

Note that only dagger is independent of partition of the legs into (co)domain.

    ==============  ====================  ====================  ============================
    tensor          domain                codomain              legs
    ==============  ====================  ====================  ============================
    A               [V1, V2]              [W1, W2]              [W1, W2, V2.dual, V1.dual]
    --------------  --------------------  --------------------  ----------------------------
    dagger(A)       [W1, W2]              [V1, V2]              [V1, V2, W2.dual, W1.dual]
    --------------  --------------------  --------------------  ----------------------------
    transpose(A)    [W2.dual, W1.dual]    [V2.dual, V1.dual]    [V2.dual, V1.dual, W1, W2]
    --------------  --------------------  --------------------  ----------------------------
    conj(A)         [V2.dual, V1.dual]    [W2.dual, W1.dual]    [W2.dual, W1.dual, V1, V2]
    ==============  ====================  ====================  ============================

Consider now a matrix ``A`` with signature ``[V] -> [W]``, i.e. with legs ``[W, V.dual]``.
The dagger ``dagger(A)`` has legs signature ``[W] -> [V]`` and legs ``[V, W.dual]``, i.e.
it can be directly contracted with ``A``.



"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABCMeta, abstractmethod
import operator
from typing import TypeVar, Sequence
from numbers import Number, Integral
import numpy as np
import warnings
import functools
import logging
logger = logging.getLogger(__name__)

from .misc import duplicate_entries
from .dummy_config import printoptions
from .symmetries import SymmetryError
from .spaces import Space, ElementarySpace, ProductSpace, Sector
from .backends.backend_factory import get_backend
from .backends.abstract_backend import Block, Backend, conventional_leg_order
from .dtypes import Dtype
from ..tools.misc import to_iterable, rank_data


__all__ = ['Tensor', 'SymmetricTensor', 'DiagonalTensor', 'ChargedTensor', 'Mask',
           'add_trivial_leg', 'almost_equal', 'angle', 'apply_mask', 'apply_mask_DiagonalTensor',
           'bend_legs', 'combine_legs', 'combine_to_matrix', 'conj', 'dagger', 'compose', 'entropy',
           'imag', 'inner', 'is_scalar', 'item', 'linear_combination', 'move_leg', 'norm', 'outer',
           'permute_legs', 'real', 'real_if_close', 'scalar_multiply', 'scale_axis', 'split_legs',
           'sqrt', 'squeeze_legs', 'tdot', 'trace', 'transpose', 'zero_like', 'get_same_backend',
           'check_same_legs']


# TENSOR CLASSES


class Tensor(metaclass=ABCMeta):
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

    |       0   1   2   3   4   5
    |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
    |      ┃          T          ┃
    |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
    |      11  10   9   8   7   6

    Attributes
    ----------
    codomain, domain : ProductSpace
        The domain and codomain of the tensor. See also :attr:`legs` and :ref:`tensors_as_maps`.
    backend : Backend
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
    codomain : ProductSpace | list[Space]
        The codomain.
    domain : ProductSpace | list[Space] | None
        The domain. ``None`` is equivalent to ``[]``, i.e. no legs in the domain.
    backend : Backend
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
    
    def __init__(self,
                 codomain: ProductSpace | list[Space],
                 domain: ProductSpace | list[Space] | None,
                 backend: Backend | None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None,
                 dtype: Dtype):
        codomain, domain, backend, symmetry = self._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        #
        self.codomain = codomain
        self.domain = domain
        self.backend = backend
        self.symmetry = symmetry
        codomain_num_legs = codomain.num_spaces
        domain_num_legs = domain.num_spaces
        self.num_legs = codomain_num_legs + domain_num_legs
        self.dtype = dtype
        self.shape = tuple(sp.dim for sp in codomain.spaces) + tuple(sp.dim for sp in reversed(domain.spaces))
        self._labels = labels = self._init_parse_labels(labels, codomain=codomain, domain=domain)
        self._labelmap = {label: legnum
                          for legnum, label in enumerate(labels)
                          if label is not None}

    @staticmethod
    def _init_parse_args(codomain: ProductSpace | list[Space],
                         domain: ProductSpace | list[Space] | None,
                         backend: Backend | None):
        """Common input parsing for ``__init__`` methods of tensor classes.

        Also checks if they are compatible.

        Returns
        -------
        codomain, domain: ProductSpace
            The codomain and domain, converted to ProductSpace if needed.
        backend: Backend
            The given backend, or the default backend compatible with `symmetry`.
        symmetry: Symmetry
            The symmetry of the domain and codomain
        """
        # Extract the symmetry from codomain or domain. Note that either may be empty, but not both.
        if isinstance(codomain, ProductSpace):
            symmetry = codomain.symmetry
        elif len(codomain) > 0:
            symmetry = codomain[0].symmetry
        elif isinstance(domain, ProductSpace):
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

        # Bring (co-)domain to ProductSpace form
        if not isinstance(codomain, ProductSpace):
            codomain = ProductSpace(codomain, symmetry=symmetry, backend=backend)
        assert codomain.symmetry == symmetry
        if domain is None:
            domain = []
        if not isinstance(domain, ProductSpace):
            domain = ProductSpace(domain, symmetry=symmetry, backend=backend)
        assert domain.symmetry == symmetry
        return codomain, domain, backend, symmetry

    @staticmethod
    def _init_parse_labels(labels: Sequence[list[str | None] | None] | list[str | None] | None,
                           codomain: ProductSpace, domain: ProductSpace,
                           is_endomorphism: bool = False):
        """Parse the various allowed input formats for labels to the format of :attr:`labels`.

        Also supports a special case for input formats of endomorphisms (maps where domain
        and codomain coincide), where a flat list of labels for the codomain can be given,
        and the domain labels are auto-filled with the respective dual labels.
        """
        # TODO improve errors on illegal formats
        num_legs = codomain.num_spaces + domain.num_spaces
        if is_endomorphism:
            assert codomain.num_spaces == domain.num_spaces

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
                    codomain_labels = [None] * codomain.num_spaces
            assert len(codomain_labels) == codomain.num_spaces
            if domain_labels is None:
                if is_endomorphism:
                    domain_labels = [_dual_leg_label(l) for l in codomain_labels]
                else:
                    domain_labels = [None] * domain.num_spaces
            assert len(domain_labels) == domain.num_spaces
            return [*codomain_labels, *reversed(domain_labels)]

        # case 3a: (only if is_endomorphism) a flat list for the codomain
        if is_endomorphism and len(labels) == codomain.num_spaces:
            return [*labels, *(_dual_leg_label(l) for l in reversed(labels))]

        # case 3: a flat list for the legs
        assert len(labels) == num_legs
        return labels[:]

    def test_sanity(self):
        assert all(_is_valid_leg_label(l) for l in self._labels)
        assert not duplicate_entries(self._labels, ignore=[None])
        assert not duplicate_entries(list(self._labelmap.values()))
        self.backend.test_leg_sanity(self.domain)
        self.backend.test_leg_sanity(self.codomain)
        assert self.dtype not in self._forbidden_dtypes

    @abstractmethod
    def as_SymmetricTensor(self) -> SymmetricTensor:
        ...

    @abstractmethod
    def copy(self, deep=True) -> Tensor:
        ...

    @abstractmethod
    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None) -> Block:
        """Convert to a dense block of the backend, if possible.

        This corresponds to "forgetting" the symmetry structure and is only possible if the
        symmetry :attr:`Symmetry.can_be_dropped`.
        The result is a backend-specific block, e.g. a numpy array if the backend is a
        :class:`NumpyBlockBackend` or a torch Tensor if the backend is a :class:`TorchBlockBackend`.

        Parameters
        ----------
        leg_order: list of (int | str), optional
            If given, the leg of the resulting block are permuted to match this leg order.
        dtype: Dtype, optional
            If given, the result is converted to this dtype. Per default it has the :attr:`dtype`
            of the tensor.
        """
        ...

    @property
    def codomain_labels(self) -> list[str | None]:
        """The labels that refer to legs in the codomain."""
        return self._labels[:self.num_codomain_legs]

    @property
    def dagger(self) -> Tensor:
        return dagger(self)

    @property
    def domain_labels(self) -> list[str | None]:
        """The labels that refer to legs in the domain."""
        return self._labels[self.num_codomain_legs:][::-1]

    @property
    def hc(self) -> Tensor:
        return dagger(self)
    
    @property
    def is_fully_labelled(self) -> bool:
        return (None not in self._labels)

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

    @functools.cached_property
    def legs(self) -> list[Space]:
        """All legs of the tensor.
        
        These the spaces of the codomain, followed by the duals of the domain spaces
        *in reverse order*.
        If we permute all legs to the codomain, we would get these spaces, i.e.::

            tensor.legs == tensor.permute_legs(codomain=range(tensor.num_legs)).codomain.spaces

        See :ref:`tensors_as_maps`.
        """
        return [*self.codomain.spaces, *(sp.dual for sp in reversed(self.domain.spaces))]

    @property
    def num_codomain_legs(self) -> int:
        """How many of the legs are in the codomain. See :ref:`tensors_as_maps`."""
        return self.codomain.num_spaces

    @property
    def num_domain_legs(self) -> int:
        """How many of the legs are in the domain. See :ref:`tensors_as_maps`."""
        return self.domain.num_spaces

    @property
    def num_parameters(self) -> int:
        """The number of free parameters for the given legs.

        This is the dimension of the space of symmetry-preserving tensors with the given legs.
        """
        # TODO / OPTIMIZE could also compute this from codomain.sectors and domain.sectors.
        #           get min(codomain.sector_multiplicity(c), domain.sector_multiplicity(c))
        #           free parameters from each coupled sector c.
        return self.parent_space.num_parameters

    @functools.cached_property
    def parent_space(self) -> ProductSpace:
        """The space that the tensor lives in. This is the product of the :attr:`legs`."""
        return ProductSpace.from_partial_products(self.codomain, self.domain.dual, backend=self.backend)

    @property
    def size(self) -> int:
        """The number of entries of a dense block representation of self.

        This is only defined if ``self.symmetry.can_be_dropped``.
        In that case, it is the number of entries of :func:`to_dense_block`.
        """
        if not self.symmetry.can_be_dropped:
            raise SymmetryError(f'Tensor.size is not defined for symmetry {self.symmetry}')
        return self.parent_space.dim

    @property
    def T(self) -> Tensor:
        return transpose(self)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return linear_combination(+1, self, +1, other)
        return NotImplemented

    def __complex__(self):
        raise TypeError('complex() of a tensor is not defined. Use tenpy.item() instead.')

    def __eq__(self, other):
        msg = f'{self.__class__.__name__} does not support == comparison. Use tenpy.almost_equal instead.'
        raise TypeError(msg)

    def __float__(self):
        raise TypeError('float() of a tensor is not defined. Use tenpy.item() instead.')

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
        # TODO skipped showing data. see commit 4bdaa5c for an old implementation of showing data.
        lines.append('>')
        return "\n".join(lines)

    def __rmul__(self, other):
        if isinstance(other, Number):
            return scalar_multiply(other, self)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return linear_combination(+1, self, -1, other)
        return NotImplemented
    
    def __truediv__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        try:
            inverse_other = 1. / other
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
            idx = _normalize_idx(which_leg, self.num_legs)
        in_domain = (idx >= len(self.codomain))
        if in_domain:
            co_domain_idx = self.num_legs - 1 - idx
        else:
            co_domain_idx = idx
        return in_domain, co_domain_idx, idx

    def _repr_header_lines(self, indent: str) -> list[str]:
        codomain_labels = []
        for n in range(self.num_codomain_legs):
            l = self._labels[n]
            if l is None:
                codomain_labels.append(f'?{n}')
            else:
                codomain_labels.append(f'"{l}"')
        codomain_labels = '[' + ', '.join(codomain_labels) + ']'
        #
        domain_labels = []
        for n in reversed(range(self.num_codomain_legs, self.num_legs)):
            l = self._labels[n]
            if l is None:
                domain_labels.append(f'?{n}')
            else:
                domain_labels.append(f'"{l}"')
        domain_labels = '[' + ', '.join(domain_labels) + ']'
        lines = [
            f'{indent}* Backend: {self.backend!s}',
            f'{indent}* Symmetry: {self.symmetry!s}',
            f'{indent}* Labels: {self._labels}',
            f'{indent}* Shape: {self.shape}',
            f'{indent}* Domain -> Codomain: {domain_labels} -> {codomain_labels}'
        ]
        return lines

    def get_leg(self, which_leg: int | str | list[int | str]) -> Space | list[Space]:
        """Basically ``self.legs[which_leg]``, but allows labels and multiple indices.

        TODO elaborate
        """
        if not isinstance(which_leg, (Integral, str)):
            # which_leg is a list
            return list(map(self.get_leg, which_leg))
        in_domain, co_domain_idx, _ = self._parse_leg_idx(which_leg)
        if in_domain:
            return self.domain.spaces[co_domain_idx].dual
        return self.codomain.spaces[co_domain_idx]

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
            return self.domain.spaces[co_domain_idx].dual
        return self.codomain.spaces[co_domain_idx]

    def get_leg_idcs(self, idcs: int | str | Sequence[int | str]) -> list[int]:
        """Parse leg-idcs of leg-labels to leg-idcs (i.e. indices of :attr:`legs`)."""
        res = []
        for idx in to_iterable(idcs):
            if isinstance(idx, str):
                idx = self._labelmap.get(idx, None)
                if idx is None:
                    msg = f'No leg with label {idx}. Labels are {self._labels}'
                    raise ValueError(msg)
            else:
                idx = _normalize_idx(idx, self.num_legs)
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

    def set_label(self, pos: int, label: str | None):
        """Set a single label at given position, in-place. Return the modified instance."""
        if label in self._labels[:pos] or label in self._labels[pos + 1:]:
            raise ValueError('Duplicate label')
        self._labels[pos] = label
        return self

    def set_labels(self, labels: Sequence[list[str | None] | None] | list[str | None] | None):
        """Set the given labels, in-place. Return the modified instance."""
        labels = self._init_parse_labels(labels, codomain=self.codomain, domain=self.domain)
        assert not duplicate_entries(labels, ignore=[None])
        assert len(labels) == self.num_legs
        self._labels = labels
        self._labelmap = {label: legnum for legnum, label in enumerate(labels) if label is not None}
        return self

    def to_numpy(self, leg_order: list[int | str] = None, numpy_dtype=None) -> np.ndarray:
        """Convert to a numpy array"""
        block = self.to_dense_block(leg_order=leg_order)
        return self.backend.block_to_numpy(block, numpy_dtype=numpy_dtype)

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


class SymmetricTensor(Tensor):
    """A tensor that is symmetric, i.e. invariant under the symmetry.

    .. note ::
        The constructor is not particularly user friendly.
        Consider using the various classmethods instead.

    TODO elaborate

    Parameters
    ----------
    codomain : ProductSpace | list[Space]
        The codomain.
    domain : ProductSpace | list[Space] | None
        The domain. ``None`` (the default) is equivalent to ``[]``, i.e. no legs in the domain.
    backend : Backend
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
    def __init__(self,
                 data,
                 codomain: ProductSpace | list[Space],
                 domain: ProductSpace | list[Space] | None = None,
                 backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        codomain, domain, backend, _ = self._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        Tensor.__init__(self, codomain=codomain, domain=domain, backend=backend, labels=labels,
                        dtype=backend.get_dtype_from_data(data))
        assert isinstance(data, self.backend.DataCls)
        self.data = data

    def test_sanity(self):
        super().test_sanity()
        self.backend.test_data_sanity(self, is_diagonal=isinstance(self, DiagonalTensor))

    @classmethod
    def from_block_func(cls, func,
                        codomain: ProductSpace | list[Space],
                        domain: ProductSpace | list[Space] | None = None,
                        backend: Backend | None = None,
                        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                        func_kwargs: dict = None,
                        shape_kw: str = None,
                        dtype: Dtype = None,
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
            via ``backend.as_block``.
        codomain, domain, backend, labels
            Arguments for constructor of :class:`SymmetricTensor`.
        func_kwargs: dict, optional
            Additional keyword arguments to be passed to ``func``.
        shape_kw: str
            If given, the shape is passed to `func` as a kwarg with this keyword.
        dtype: Dtype, None
            If given, the resulting blocks from `func` are converted to this dtype.

        See Also
        --------
        from_sector_block_func
            Allows the `func` to take the current coupled sectors as an argument.
        """
        codomain, domain, backend, symmetry = cls._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
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
            return backend.as_block(block, dtype)

        data = backend.from_sector_block_func(block_func, codomain=codomain, domain=domain)
        res = cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)
        res.test_sanity()  # OPTIMIZE remove?
        return res

    @classmethod
    def from_dense_block(cls, block,
                         codomain: ProductSpace | list[Space],
                         domain: ProductSpace | list[Space] | None = None,
                         backend: Backend | None = None,
                         labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                         dtype: Dtype = None,
                         tol: float = 1e-6):
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
        """
        codomain, domain, backend, symmetry = cls._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        block, dtype = backend.as_block(block, dtype, return_dtype=True)
        block = backend.apply_basis_perm(block, conventional_leg_order(codomain, domain))
        data = backend.from_dense_block(block, codomain=codomain, domain=domain, tol=tol)
        return cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)

    @classmethod
    def from_dense_block_trivial_sector(cls, vector: Block, space: Space,
                                        backend: Backend | None = None, label: str | None = None
                                        ) -> SymmetricTensor:
        """Inverse of to_dense_block_trivial_sector. TODO"""
        if backend is None:
            backend = get_backend(symmetry=space.symmetry)
        vector = backend.as_block(vector)
        if space._basis_perm is not None:
            i = space.sectors_where(space.symmetry.trivial_sector)
            perm = rank_data(space.basis_perm[slice(*space.slices[i])])
            vector = backend.apply_leg_permutations(vector, [perm])
        raise NotImplementedError  # TODO
    
    @classmethod
    def from_eye(cls, co_domain: list[Space] | ProductSpace,
                 backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                 dtype: Dtype = Dtype.complex128
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
        backend: Backend, optional
            The backend of the tensor.
        dtype: Dtype
            The dtype of the tensor.
        """
        co_domain, _, backend, _ = cls._init_parse_args(
            codomain=co_domain, domain=co_domain, backend=backend
        )
        labels = cls._init_parse_labels(labels, codomain=co_domain, domain=co_domain,
                                        is_endomorphism=True)
        data = backend.eye_data(co_domain=co_domain, dtype=dtype)
        return cls(data, codomain=co_domain, domain=co_domain, backend=backend, labels=labels)

    @classmethod
    def from_random_normal(cls, codomain: ProductSpace | list[Space],
                           domain: ProductSpace | list[Space] | None = None,
                           mean: SymmetricTensor | None = None,
                           sigma: float = 1.,
                           backend: Backend | None = None,
                           labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                           dtype: Dtype = Dtype.complex128):
        r"""Generate a sample from the complex normal distribution.

        The probability density is

        .. math ::
            p(T) \propto \mathrm{exp}\left[
                \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
            \right]

        TODO make sure we actually generate from that distribution in the non-abelian or anyonic case!

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
        assert dtype.is_complex
        assert sigma > 0.
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
        else:
            if codomain is None:
                raise ValueError('Must specify the codomain if mean is not given.')
            codomain, domain, backend, _ = cls._init_parse_args(
                codomain=codomain, domain=domain, backend=backend
            )
        with_zero_mean = cls(
            data=backend.from_random_normal(codomain, domain, sigma=sigma, dtype=dtype),
            codomain=codomain, domain=domain, backend=backend, labels=labels
        )
        if mean is not None:
            return mean + with_zero_mean
        return with_zero_mean

    @classmethod
    def from_random_uniform(cls, codomain: ProductSpace | list[Space],
                            domain: ProductSpace | list[Space] | None = None,
                            backend: Backend | None = None,
                            labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                            dtype: Dtype = Dtype.complex128):
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
        codomain, domain, backend, symmetry = cls._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        return cls.from_block_func(
            func=backend.block_random_uniform,
            codomain=codomain, domain=domain, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype), dtype=dtype
        )

    @classmethod
    def from_sector_block_func(cls, func,
                               codomain: ProductSpace | list[Space],
                               domain: ProductSpace | list[Space] | None = None,
                               backend: Backend | None = None,
                               labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                               func_kwargs: dict = None,
                               dtype: Dtype = None,
                               ):
        """Initialize a :class:`SymmetricTensor` by generating its blocks from a function.

        Here "the blocks of a tensor" are the backend-specific blocks that contain the free
        parameters of the tensor in the :attr:`data`. The concrete meaning of these blocks depends
        on the backend.

        Unlike :meth:`from_block_func`, this classmethod supports a `func` that takes the current
        coupled sector as an argument. The tensor, as a map from its domain to its codomain is
        block-diagonal in the coupled sectors, i.e. in the ``domain.sectors``.
        Thus, the free parameters of a tensor are associated with one of block of this structure,
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
            The output is converted to backend-specific blocks via ``backend.as_block``.
        codomain, domain, backend, labels
            Arguments, like for constructor of :class:`SymmetricTensor`.
        func_kwargs: dict, optional
            Additional keyword arguments to be passed to ``func``.
        shape_kw: str
            If given, the shape is passed to `func` as a kwarg with this keyword.
        dtype: Dtype, None
            If given, the resulting blocks from `func` are converted to this dtype.

        See Also
        --------
        from_block_func
        """
        codomain, domain, backend, symmetry = cls._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        # wrap func to consider func_kwargs and dtype
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            block = func(shape, coupled, **func_kwargs)
            return backend.as_block(block, dtype)

        data = backend.from_sector_block_func(block_func, codomain=codomain, domain=domain)
        res = cls(data, codomain=codomain, domain=domain, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_zero(cls, codomain: ProductSpace | list[Space],
                  domain: ProductSpace | list[Space] | None = None,
                  backend: Backend | None = None,
                  labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                  dtype: Dtype = Dtype.complex128):
        """A zero tensor.

        Parameters
        ----------
        codomain, domain, backend, labels:
            Arguments, like for constructor of :class:`SymmetricTensor`.
        dtype: Dtype
            The dtype for the entries.
        """
        codomain, domain, backend, symmetry = cls._init_parse_args(
            codomain=codomain, domain=domain, backend=backend
        )
        return cls(
            data=backend.zero_data(codomain=codomain, domain=domain, dtype=dtype),
            codomain=codomain, domain=domain, backend=backend, labels=labels
        )

    def as_SymmetricTensor(self) -> SymmetricTensor:
        return self.copy()

    def copy(self, deep=True) -> SymmetricTensor:
        if deep:
            data = self.backend.copy_data(self)
        else:
            data = self.data
        return SymmetricTensor(data=data, codomain=self.codomain, domain=self.domain,
                               backend=self.backend, labels=self.labels)

    def diagonal(self, check_offdiagonal=False) -> DiagonalTensor:
        """The diagonal part as a :class:`DiagonalTensor`.

        Parameters
        ----------
        check_offdiagonal: bool
            If we should check that the off-diagonal parts vanish.
        """
        return DiagonalTensor.from_tensor(self, check_offdiagonal=check_offdiagonal)

    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None) -> Block:
        block = self.backend.to_dense_block(self)
        block = self.backend.apply_basis_perm(block, conventional_leg_order(self), inv=True)
        if dtype is not None:
            block = self.backend.block_to_dtype(block, dtype)
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self.get_leg_idcs(leg_order))
        return block

    def to_dense_block_trivial_sector(self) -> Block:
        """Assumes self is a single-leg tensor and returns its components in the trivial sector.

        TODO elaborate.
        TODO better name?

        See Also
        --------
        from_dense_block_trivial_sector
        """
        assert self.num_legs == 1
        block = self.backend.to_dense_block_trivial_sector(self)
        assert self.num_codomain_legs == 1  # TODO assuming this for now to construct the perm. should we keep that?
        leg = self.codomain[0]
        if leg._basis_perm is not None:
            i = leg.sectors_where(self.symmetry.trivial_sector)
            perm = np.argsort(leg.basis_perm[slice(*leg.slices[i])])
            block = self.backend.apply_leg_permutations(block, [perm])
        return block


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
    
    This is the natural type e.g. for singular values or eigenvalue and allows elementwise methods
    such as TODO.

    Parameters
    ----------
    data
        The numerical data ("free parameters") comprising the tensor. type is backend-specific
    leg: Space
        The single leg in both the domain and codomain
    backend : Backend
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
    TODO elaborate
    """
    _forbidden_dtypes = []
    
    def __init__(self, data, leg: Space, backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        SymmetricTensor.__init__(self, data, codomain=[leg], domain=[leg], backend=backend,
                                 labels=labels)

    def test_sanity(self):
        super().test_sanity()
        assert self.domain == self.codomain
        assert self.domain.num_spaces == 1

    @classmethod
    def from_block_func(cls, func, leg: Space, backend: Backend | None = None,
                        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                        func_kwargs: dict = None,
                        shape_kw: str = None,
                        dtype: Dtype = None):
        co_domain, _, backend, symmetry = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
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
            return backend.as_block(block, dtype)

        data = backend.diagonal_from_sector_block_func(block_func, co_domain=co_domain)
        res = cls(data, leg=leg, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_dense_block(cls, block: Block, leg: Space, backend: Backend | None = None,
                         labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                         dtype: Dtype = None, tol: float = 1e-6):
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        diag = backend.block_get_diagonal(backend.as_block(block, dtype=dtype), check_offdiagonal=True)
        return cls.from_diag_block(diag, leg=leg, backend=backend, labels=labels, dtype=dtype,
                                   tol=tol)

    @classmethod
    def from_diag_block(cls, diag: Block, leg: Space, backend: Backend | None = None,
                        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                        dtype: Dtype = None, tol: float = 1e-6):
        """Convert a dense 1D block containing the diagonal entries to a DiagonalTensor.

        Parameters
        ----------
        diag: Block-like
            The diagonal entries as a backend-specific block or some data that can be converted
            using :meth:`AbstractBlockBackend.as_block`. This includes e.g. nested python iterables
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
        co_domain, _, backend, symmetry = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
        diag = backend.as_block(diag, dtype=dtype)
        diag = backend.apply_basis_perm(diag, [leg])
        return cls(
            data=backend.diagonal_from_block(diag, co_domain=co_domain, tol=tol),
            leg=leg, backend=backend, labels=labels
        )

    @classmethod
    def from_eye(cls, leg: Space, backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                 dtype: Dtype = Dtype.float64):
        """The identity map as a DiagonalTensor.

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            The dtype for the entries.
        """
        co_domain, _, backend, symmetry = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
        return cls.from_block_func(
            backend.ones_block, leg=leg, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype), dtype=dtype
        )

    @classmethod
    def from_random_normal(cls, leg: Space,
                           mean: DiagonalTensor | None = None,
                           sigma: float = 1.,
                           backend: Backend | None = None,
                           labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                           dtype: Dtype = Dtype.complex128):
        r"""Generate a sample from the complex normal distribution.

        The probability density is

        .. math ::
            p(T) \propto \mathrm{exp}\left[
                \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
            \right]

        TODO make sure we actually generate from that distribution in the non-abelian or anyonic case!

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
        assert sigma > 0.
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
            co_domain, _, backend, symmetry = cls._init_parse_args(
                codomain=[leg], domain=[leg], backend=backend
            )
        with_zero_mean = cls.from_block_func(
            backend.block_random_normal, leg=leg, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype), dtype=dtype
        )
        if mean is not None:
            return mean + with_zero_mean
        return with_zero_mean

    @classmethod
    def from_random_uniform(cls, leg: Space,
                           backend: Backend | None = None,
                           labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                           dtype: Dtype = Dtype.complex128):
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
        co_domain, _, backend, symmetry = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
        return cls.from_block_func(
            func=backend.block_random_uniform, leg=leg, backend=backend, labels=labels,
            func_kwargs=dict(dtype=dtype), dtype=dtype
        )

    @classmethod
    def from_sector_block_func(cls, func, leg: Space, backend: Backend | None = None,
                               labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                               func_kwargs: dict = None,
                               dtype: Dtype = None):
        co_domain, _, backend, _ = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
        # wrap func to consider func_kwargs and dtype
        if func_kwargs is None:
            func_kwargs = {}

        def block_func(shape, coupled):
            block = func(shape, coupled, **func_kwargs)
            return backend.as_block(block, dtype)

        data = backend.diagonal_from_sector_block_func(block_func, co_domain=co_domain)
        res = cls(data, leg=leg, backend=backend, labels=labels)
        res.test_sanity()
        return res

    @classmethod
    def from_tensor(cls, tens: SymmetricTensor, check_offdiagonal: bool = True) -> DiagonalTensor:
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
        assert tens.num_legs == 2
        assert tens.domain == tens.codomain
        data = tens.backend.diagonal_tensor_from_full_tensor(tens, check_offdiagonal=check_offdiagonal)
        return cls(data=data, leg=tens.codomain.spaces[0], backend=tens.backend, labels=tens.labels)

    @classmethod
    def from_zero(cls, leg: Space,
                  backend: Backend | None = None,
                  labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                  dtype: Dtype = Dtype.complex128):
        """A zero tensor.

        Parameters
        ----------
        leg, backend, labels
            Arguments for constructor of :class:`DiagonalTensor`.
        dtype: Dtype
            The dtype for the entries.
        """
        co_domain, _, backend, symmetry = cls._init_parse_args(
            codomain=[leg], domain=[leg], backend=backend
        )
        return cls(
            data=backend.zero_diagonal_data(co_domain=co_domain, dtype=dtype),
            leg=leg, backend=backend, labels=labels
        )

    @property
    def leg(self) -> Space:
        """Return the single space that makes up to domain and codomain."""
        return self.codomain.spaces[0]

    def __abs__(self):
        return self._elementwise_unary(func=operator.abs, maps_zero_to_zero=True)

    def __bool__(self):
        if self.dtype == Dtype.bool and is_scalar(self):
            return bool(item(self))
        msg = 'The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()'
        raise ValueError(msg)

    # TODO offer elementwise boolean operations: and, or, xor, eq

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

    def as_SymmetricTensor(self) -> SymmetricTensor:
        return SymmetricTensor(
            data=self.backend.full_data_from_diagonal_tensor(self),
            codomain=self.codomain, domain=self.domain, backend=self.backend, labels=self.labels
        )

    def _binary_operand(self, other: Number | DiagonalTensor, func, operand: str,
                        return_NotImplemented: bool = False, right: bool = False):
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
                    self, func=lambda block: func(other, block), func_kwargs={},
                    maps_zero_to_zero=False
                )
            else:
                data = backend.diagonal_elementwise_unary(
                    self, func=lambda block: func(block, other), func_kwargs={},
                    maps_zero_to_zero=False
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

    def copy(self, deep=True) -> SymmetricTensor:
        if deep:
            data = self.backend.copy_data(self)
        else:
            data = self.data
        return DiagonalTensor(data, leg=self.leg, backend=self.backend, labels=self.labels)

    def diagonal(self) -> DiagonalTensor:
        return self

    def diagonal_as_block(self, dtype: Dtype = None) -> Block:
        res = self.backend.diagonal_tensor_to_block(self)
        res = self.backend.apply_basis_perm(res, [self.leg], inv=True)
        if dtype is not None:
            res = self.backend.block_to_dtype(res, dtype)
        return res

    def diagonal_as_numpy(self, numpy_dtype=None) -> np.ndarray:
        block = self.diagonal_as_block(dtype=Dtype.from_numpy_dtype(numpy_dtype))
        return self.backend.block_to_numpy(block, numpy_dtype=numpy_dtype)

    def elementwise_almost_equal(self, other: DiagonalTensor, rtol: float = 1e-5, atol=1e-8
                                 ) -> DiagonalTensor:
        return abs(self - other) <= (atol + rtol * abs(self))

    def _elementwise_binary(self, other: DiagonalTensor, func, func_kwargs: dict = None,
                            partial_zero_is_zero: bool = False) -> DiagonalTensor:
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
            self, other, func=func, func_kwargs=func_kwargs or {},
            partial_zero_is_zero=partial_zero_is_zero
        )
        labels = _get_matching_labels(self._labels, other._labels)
        return DiagonalTensor(data, self.leg, backend=backend, labels=labels)

    def _elementwise_unary(self, func, func_kwargs: dict = None, maps_zero_to_zero: bool = False
                           ) -> DiagonalTensor:
        """An elementwise function acting on a diagonal tensor.

        Applies ``func(self_block: Block, **func_kwargs) -> Block`` elementwise.
        Set ``maps_zero_to_zero=True`` to promise that ``func(0) == 0``.
        """
        data = self.backend.diagonal_elementwise_unary(
            self, func, func_kwargs=func_kwargs or {}, maps_zero_to_zero=maps_zero_to_zero
        )
        return DiagonalTensor(data, self.leg, backend=self.backend, labels=self.labels)
    
    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None) -> Block:
        diag = self.diagonal_as_block(dtype=dtype)
        res = self.backend.block_from_diagonal(diag)
        if leg_order is not None:
            res = self.backend.block_permute_axes(res, self.get_leg_idcs(leg_order))
        return res


class Mask(Tensor):
    r"""A boolean mask that can be used to project or enlarge a leg.

    Masks come in two versions: projections and inclusions.
    A projection Mask is a special kind of projection map, that either keeps or discards any given
    sector. It has a single leg, the :attr:`large_leg` in its domain and maps it to a single leg,
    the :attr:`small_leg` in the codomain.
    An inclusion Mask is the dagger of this projection Mask and maps from the small leg in the
    domain to the large leg in the codomain::

    
    |         │                 ║
    |      ┏━━┷━━┓           ┏━━┷━━┓
    |      ┃ M_p ┃    OR     ┃ M_i ┃
    |      ┗━━┯━━┛           ┗━━┯━━┛
    |         ║                 │

    TODO think in detail about the basis_perm and how it interacts with masking...

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
    backend: Backend, optional
        The backend of the tensor.
    labels: list[list[str | None]] | list[str | None] | None
        Specify the labels for the legs.
        Can either give two lists, one for the codomain, one for the domain.
        Or a single flat list for all legs in the order of the :attr:`legs`,
        such that ``[codomain_labels, domain_labels]`` is equivalent
        to ``[*codomain_legs, *reversed(domain_legs)]``.
    """
    _forbidden_dtypes = [Dtype.float32, Dtype.float64, Dtype.complex64, Dtype.complex128]

    def __init__(self, data, space_in: ElementarySpace, space_out: ElementarySpace,
                 is_projection: bool = None, backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        if is_projection is None:
            if space_in.dim == space_out.dim:
                raise ValueError('Need to specify is_projection for equal spaces.')
            is_projection = (space_in.dim > space_out.dim)
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
        Tensor.__init__(self, codomain=[space_out], domain=[space_in], backend=backend,
                        labels=labels, dtype=Dtype.bool)
        assert isinstance(data, self.backend.DataCls)  # TODO rm check after testing?
        self.data = data

    def test_sanity(self):
        super().test_sanity()
        self.backend.test_mask_sanity(self)
        assert self.codomain.num_spaces == 1 == self.domain.num_spaces
        assert self.large_leg.is_dual == self.small_leg.is_dual
        assert self.small_leg.is_subspace_of(self.large_leg)
        assert self.dtype == Dtype.bool

    @property
    def large_leg(self) -> ElementarySpace:
        if self.is_projection:
            return self.domain.spaces[0]
        else:
            return self.codomain.spaces[0]

    @property
    def small_leg(self) -> ElementarySpace:
        if self.is_projection:
            return self.codomain.spaces[0]
        else:
            return self.domain.spaces[0]

    @classmethod
    def from_eye(cls, leg: ElementarySpace, is_projection: bool = True,
                 backend: Backend | None = None,
                 labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        """The identity map as a Mask, i.e. the mask that keeps all states and discards none.

        Parameters
        ----------
        large_leg, backend, labels:
            Arguments, like for constructor of :class:`Mask`.

        See Also
        --------
        from_zero
            The projection Mask, that discards all states and keeps none.
        """
        diag = DiagonalTensor.from_eye(leg=leg, backend=backend, labels=labels, dtype=Dtype.bool)
        res = cls.from_DiagonalTensor(diag)
        if not is_projection:
            return dagger(res)
        return res

    @classmethod
    def from_block_mask(cls, block_mask: Block, large_leg: Space, backend: Backend | None = None,
                        labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        """Create a projection Mask from a boolean block.

        To get the related inclusion Mask, use :func:`dagger`.

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
        block_mask = backend.as_block(block_mask, Dtype.bool)
        block_mask = backend.apply_basis_perm(block_mask, [large_leg])
        data, small_leg = backend.mask_from_block(block_mask, large_leg=large_leg)
        return cls(data=data, space_in=large_leg, space_out=small_leg, is_projection=True,
                   backend=backend, labels=labels)

    @classmethod
    def from_DiagonalTensor(cls, diag: DiagonalTensor):
        """Create a projection Mask from a boolean DiagonalTensor.

        The resulting mask keeps exactly those basis elements for which the entry of `diag` is ``True``.
        To get the related inclusion Mask, use the :func:`dagger`.
        """
        assert diag.dtype == Dtype.bool
        data, small_leg = diag.backend.diagonal_to_mask(diag)
        return cls(
            data=data, space_in=diag.domain.spaces[0], space_out=small_leg, is_projection=True,
            backend=diag.backend, labels=diag.labels
        )

    @classmethod
    def from_indices(cls, indices: int | Sequence[int] | slice, large_leg: Space,
                     backend: Backend = None,
                     labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        """Create a projection Mask from the indices that are kept.

        To get the related inclusion Mask, use :func:`dagger`.

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
        return cls.from_block_mask(block_mask, large_leg=large_leg, backend=backend, labels=labels)

    @classmethod
    def from_random(cls, large_leg: Space, small_leg: Space | None = None,
                    backend: Backend | None = None, p_keep: float = .5, min_keep: int = 0,
                    labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                    np_random: np.random.Generator = np.random.default_rng()):
        """Create a random projection Mask.

        To get the related inclusion Mask, use :func:`dagger`.

        Parameters
        ----------
        large_leg: Space
            The large leg, in the domain of the projection
        small_leg: Space, optional
            The small leg. If given, must be a subspace of the `large_leg`.
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
            
        if small_leg is None:
            assert 0 <= p_keep <= 1
            diag = DiagonalTensor.from_random_uniform(large_leg, backend=backend, labels=labels,
                                                      dtype=Dtype.float32)
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
            # first, try just a bit higher
            p_keep = p_keep + 0.05 * (1 - p_keep)
            res = cls.from_DiagonalTensor(diag < (2 * p_keep - 1))
            for _ in range(20):
                if np.sum(res.small_leg.multiplicities) >= min_keep:
                    return res
                p_keep = .5 * (p_keep + 1)  # step halfway towards 100%
                res = cls.from_DiagonalTensor(diag < (2 * p_keep - 1))
            raise RuntimeError('Could not fulfill min_keep')

        assert small_leg.is_subspace_of(large_leg)
        

        def func(shape, coupled):
            num_keep = small_leg.sector_multiplicity(coupled)
            block = np.zeros(shape, bool)
            which = np_random.choice(shape[0], size=num_keep, replace=False)
            block[which] = True
            return block

        diag = DiagonalTensor.from_sector_block_func(
            func, leg=large_leg, backend=backend, labels=labels, dtype=Dtype.bool
        )
        res = cls.from_DiagonalTensor(diag)
        res.small_leg._basis_perm = small_leg._basis_perm
        res.small_leg._inverse_basis_perm = small_leg._inverse_basis_perm
        return res

    @classmethod
    def from_zero(cls, large_leg: Space, backend: Backend | None = None,
                  labels: Sequence[list[str | None] | None] | list[str | None] | None = None):
        """The zero projection Mask, that discards all states and keeps none.

        To get the related inclusion Mask, use :func:`dagger`.

        Parameters
        ----------
        large_leg: Space
            The large leg, in the domain of the projection
        backend, labels
            Arguments, like for the constructor

        See Also
        --------
        from_eye
            The projection (or inclusion) Mask that keeps all states
        """
        if backend is None:
            backend = get_backend(symmetry=large_leg.symmetry)
        data = backend.zero_mask_data(large_leg=large_leg)
        small_leg = ElementarySpace.from_null_space(symmetry=large_leg.symmetry,
                                                    is_dual=large_leg.is_bra_space)
        return cls(data, space_in=large_leg, space_out=small_leg, is_projection=True,
                   backend=backend, labels=labels)

    def __and__(self, other):  # ``self & other``
        return self._binary_operand(other, operator.and_, '==')

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
        return self.backend.apply_basis_perm(res, [self.large_leg], inv=True)

    def as_numpy_mask(self) -> np.ndarray:
        return self.backend.block_to_numpy(self.as_block_mask(), numpy_dtype=bool)

    def as_DiagonalTensor(self, dtype=Dtype.complex128) -> DiagonalTensor:
        return DiagonalTensor(data=self.backend.mask_to_diagonal(self, dtype=dtype),
                              leg=self.large_leg, backend=self.backend, labels=self.labels)

    def as_SymmetricTensor(self, dtype=Dtype.complex128) -> SymmetricTensor:
        if not self.is_projection:
            # OPTIMIZE how hard is it to deal with inclusions in the backend?
            return dagger(dagger(self).as_SymmetricTensor())
        data = self.backend.full_data_from_mask(self, dtype)
        return SymmetricTensor(data, codomain=self.codomain, domain=self.domain,
                               backend=self.backend, labels=self.labels)

    def _binary_operand(self, other: bool | Mask, func, operand: str,
                        return_NotImplemented: bool = True) -> Mask:
        """Utility function for a shared implementation of binary functions, whose second argument
        may be a scalar ("to be broadcast") or a Mask.

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
                dagger(other), func=func, operand=operand,
                return_NotImplemented=return_NotImplemented
            )
            return dagger(res_projection)
        backend = get_same_backend(self, other)
        if self.domain != other.domain:
            raise ValueError('Incompatible domain.')
        data, small_leg = backend.mask_binary_operand(self, other, func)
        return Mask(data, space_in=self.large_leg, space_out=small_leg,
                    is_projection=self.is_projection, backend=backend,
                    labels=_get_matching_labels(self.labels, other.labels))

    def copy(self, deep=True) -> Mask:
        if deep:
            data = self.backend.copy_data(self)
        else:
            data = self.data
        return Mask(data, space_in=self.large_leg, space_out=self.small_leg,
                    is_projection=self.is_projection, backend=self.backend, labels=self.labels)
    
    def logical_not(self):
        """Alias for :meth:`orthogonal_complement`"""
        return self._unary_operand(operator.invert)

    def orthogonal_complement(self):
        """The "opposite" Mask, that keeps exactly what self discards and vv."""
        return self._unary_operand(operator.invert)

    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None) -> Block:
        # for Mask, defining via numpy is actually easier, to use numpy indexing
        numpy_dtype = None if dtype is None else dtype.to_numpy_dtype()
        as_numpy = self.to_numpy(leg_order=leg_order, numpy_dtype=numpy_dtype)
        return self.backend.as_block(as_numpy, dtype=dtype)

    def to_numpy(self, leg_order: list[int | str] = None, numpy_dtype=None) -> np.ndarray:
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
        return Mask(data, space_in=self.large_leg, space_out=small_leg,
                    is_projection=True, backend=self.backend, labels=self.labels)


class ChargedTensor(Tensor):
    r"""Tensors which transform non-trivially under a symmetry.

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

    |       0   1   2   3
    |      ┏┷━━━┷━━━┷━━━┷┓
    |      ┃  invariant  ┃
    |      ┗┯━━━┯━━━┯━━━┯┛
    |      C┆   6   5   4
    |      ┏┷┓
    |      ┗━┛
    
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
    ``C == ElementarySpace(u1_sym, sectors=[[+1]])``.
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
    TODO revisit this paragraph, do we actually support doing that? 
    
    Parameters
    ----------
    invariant_part:
        The symmetry-invariant part. the charge leg is the its ``domain.spaces[0]``.
    charged_state: block | None
        Either ``None``, or a backend-specific block of shape ``(charge_leg.dim,)``, which specifies
        a state on the charge leg (see notes above).
    """
    
    _CHARGE_LEG_LABEL = '!'  # canonical label for the charge leg
    def __init__(self, invariant_part: SymmetricTensor, charged_state: Block | None):
        assert invariant_part.domain.num_spaces > 0, 'domain must contain at least the charge leg'
        self.charge_leg = invariant_part.domain.spaces[0]
        assert invariant_part._labels[-1] == self._CHARGE_LEG_LABEL, 'incorrect label on charge leg'
        if charged_state is not None:
            if not invariant_part.symmetry.can_be_dropped:
                msg = f'charged_state can not be specified for symmetry {invariant_part.symmetry}'
                raise SymmetryError(msg)
            charged_state = invariant_part.backend.as_block(
                charged_state, invariant_part.dtype
            )
        self.charged_state = charged_state
        self.invariant_part = invariant_part
        Tensor.__init__(
            self,
            codomain=invariant_part.codomain,
            domain=ProductSpace(
                invariant_part.domain.spaces[1:], symmetry=invariant_part.symmetry,
                backend=invariant_part.backend
            ),
            backend=invariant_part.backend,
            labels=invariant_part._labels[:-1],
            dtype=invariant_part.dtype
        )

    def test_sanity(self):
        super().test_sanity()
        self.invariant_part.test_sanity()
        if self.charged_state is not None:
            assert self.backend.block_shape(self.charged_state) == (self.charge_leg.dim,)

    @staticmethod
    def _parse_inv_domain(domain: ProductSpace, charge: Space | Sector | Sequence[int],
                          backend: Backend) -> tuple[ProductSpace, Space]:
        """Helper function to build the domain of the invariant part.

        Parameters
        ----------
        domain: ProductSpace
            The domain of the ChargedTensor
        charge: Space | SectorLike
            Specification for the charge_leg, either as a space or a single sector
        backend: Backend
            The backend, used for building the output ProductSpace.

        Returns
        -------
        inv_domain: ProductSpace
            The domain of the invariant part
        charge_leg: Space
            The charge_leg of the resulting ChargedTensor
        """
        assert isinstance(domain, ProductSpace), 'call _init_parse_args first?'
        if not isinstance(charge, Space):
            sectors = np.asarray(charge, int)[None, :]
            charge = Space(domain.symmetry, sectors)
        return domain.left_multiply(charge, backend=backend), charge

    @staticmethod
    def _parse_inv_labels(labels: Sequence[list[str | None] | None] | list[str | None] | None,
                          codomain: ProductSpace, domain: ProductSpace):
        """Utility like :meth:`_init_parse_labels`, but also returns the labels for the invariant
        part."""
        labels = ChargedTensor._init_parse_labels(labels, codomain, domain)
        inv_labels = labels + [ChargedTensor._CHARGE_LEG_LABEL]
        return labels, inv_labels

    @classmethod
    def from_block_func(cls, func,
                        charge: Space | Sector,
                        codomain: ProductSpace | list[Space],
                        domain: ProductSpace | list[Space] | None = None,
                        charged_state: Block | None = None,
                        backend: Backend | None = None,
                        labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                        func_kwargs: dict = None,
                        shape_kw: str = None,
                        dtype: Dtype = None,):
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
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge, backend=backend)
        inv = SymmetricTensor.from_block_func(
            func=func, codomain=codomain, domain=inv_domain, backend=backend, labels=labels,
            func_kwargs=func_kwargs, shape_kw=shape_kw, dtype=dtype
        )
        return ChargedTensor(inv, charged_state)

    @classmethod
    def from_dense_block(cls, block: Block,
                         codomain: ProductSpace | list[Space],
                         domain: ProductSpace | list[Space] | None = None,
                         charge: Space | Sector | None = None,
                         backend: Backend | None = None,
                         labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                         dtype: Dtype = None,
                         tol: float = 1e-6
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
        """
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain, domain, backend)
        labels, inv_labels = cls._parse_inv_labels(labels, codomain, domain)
        if not symmetry.can_be_dropped:
            raise SymmetryError
        block, dtype = backend.as_block(block, dtype, return_dtype=True)
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge, backend=backend)
        if charge_leg.dim != 1:
            raise NotImplementedError  # TODO
        inv_part = SymmetricTensor.from_dense_block(
            block=backend.add_axis(block, -1),
            codomain=codomain, domain=inv_domain, backend=backend, labels=inv_labels, dtype=dtype,
            tol=tol
        )
        return cls(inv_part, charged_state=[1])

    @classmethod
    def from_dense_block_single_sector(cls, vector: Block, space: Space, sector: Sector,
                                       backend: Backend | None = None, label: str | None = None
                                       ) -> ChargedTensor:
        """Given a `vector` in single `space`, represent the components in a single given `sector`.

        The resulting charged tensor has a charge lector which has the `sector`.

        See Also
        --------
        to_dense_block_single_sector
        """
        if backend is None:
            backend = get_backend(symmetry=space.symmetry)
        if space.symmetry.sector_dim(sector) > 1:
            # TODO how to handle multi-dim sectors? which dummy leg state to give?
            raise NotImplementedError
        charge_leg = ElementarySpace(space.symmetry, [sector])
        vector = backend.as_block(vector)
        if space._basis_perm is not None:
            i = space.sectors_where(sector)
            perm = rank_data(space.basis_perm[slice(*space.slices[i])])
            vector = backend.apply_leg_permutations(vector, [perm])
        inv_data = backend.inv_part_from_dense_block_single_sector(
            vector=vector, space=space, charge_leg=charge_leg
        )
        inv_part = SymmetricTensor(inv_data, codomain=[space], domain=[charge_leg], backend=backend,
                                   labels=[label, cls._CHARGE_LEG_LABEL])
        return cls(inv_part, [1])

    @classmethod
    def from_invariant_part(cls, invariant_part: SymmetricTensor, charged_state: Block | None
                            ) -> ChargedTensor | complex:
        """Like constructor, but deals with the case where invariant_part has only one leg.

        In that case, we return a scalar if the charged_state is specified and raise otherwise.
        """
        if invariant_part.num_legs == 1:
            if charged_state is None:
                raise ValueError('Can not instantiate ChargedTensor with no legs and unspecified charged_states.')
            # OPTIMIZE ?
            inv_block = invariant_part.to_dense_block()
            return invariant_part.backend.block_inner(inv_block, charged_state, do_dagger=False)
        return cls(invariant_part, charged_state)

    @classmethod
    def from_two_charge_legs(cls, invariant_part: SymmetricTensor, state1: Block | None,
                             state2: Block | None) -> ChargedTensor | complex:
        """Create a charged tensor from an invariant part with two charged legs.

        Parameters
        -
        """
        inv_part = combine_legs(invariant_part, -1, -2)
        inv_part.set_label(-1, cls._CHARGE_LEG_LABEL)
        if state1 is None and state2 is None:
            state = None
        elif state1 is None or state2 is None:
            raise ValueError('Must specify either both or none of the states')
        else:
            state = invariant_part.backend.state_tensor_product(state1, state2, inv_part.domain[0])
        return cls.from_invariant_part(inv_part, state)

    @classmethod
    def from_zero(cls, codomain: ProductSpace | list[Space],
                  domain: ProductSpace | list[Space] | None = None,
                  charge: Space | Sector | None = None,
                  charged_state: Block | None = None,
                  backend: Backend | None = None,
                  labels: Sequence[list[str | None] | None] | list[str | None] | None = None,
                  dtype: Dtype = Dtype.complex128,
                  ):
        """A zero tensor."""
        codomain, domain, backend, symmetry = cls._init_parse_args(codomain, domain, backend)
        inv_domain, charge_leg = cls._parse_inv_domain(domain=domain, charge=charge, backend=backend)
        labels, inv_labels = cls._parse_inv_labels(labels, codomain, domain)
        inv_part = SymmetricTensor.from_zero(codomain=codomain, domain=inv_domain, backend=backend,
                                             labels=inv_labels, dtype=dtype)
        return ChargedTensor(inv_part, charged_state)

    def as_SymmetricTensor(self) -> SymmetricTensor:
        """Convert to symmetric tensor, if possible."""
        if not np.all(self.charge_leg.sectors == self.symmetry.trivial_sector[None, :]):
            raise SymmetryError('Not a symmetric tensor')
        if self.charge_leg.dim == 1:
            res = squeeze_legs(self.invariant_part, -1)
            if self.charged_state is not None:
                res = self.backend.block_item(self.charged_state) * res
            return res
        if self.charged_state is None:
            raise ValueError('Can not convert to SymmetricTensor. charged_state is not defined.')
        state = SymmetricTensor.from_dense_block(
            self.charged_state, codomain=[self.charged_state.dual], backend=self.backend,
            labels=[_dual_leg_label(self._CHARGE_LEG_LABEL)], dtype=self.dtype
        )
        res = tdot(state, self.invariant_part, 0, -1)
        return bend_legs(res, num_codomain_legs=self.num_codomain_legs)
        
    def copy(self, deep=True) -> ChargedTensor:
        inv_part = self.invariant_part.copy(deep=deep)  # this extra layer is cheap...
        charged_state = self.charged_state
        if deep and self.charged_state is not None:
            charged_state = self.backend.block_copy(charged_state)
        return ChargedTensor(inv_part, charged_state)
    
    def _repr_header_lines(self, indent: str) -> list[str]:
        lines = Tensor._repr_header_lines(self, indent=indent)
        lines.append(f'{indent}* Charge Leg: dim={self.charge_leg.dim} sectors={self.charge_leg.sectors}')
        start = f'{indent}* Charged State: '
        if self.charged_state is None:
            lines.append(f'{start}unspecified')
        else:
            state_lines = self.backend._block_repr_lines(
                self.charged_state, indent=indent + '  ',
                max_width=printoptions.linewidth - len(start), max_lines=1
            )
            lines.append(start + state_lines[0])
        return lines

    def set_label(self, pos: int, label: str | None):
        pos = _normalize_idx(pos, self.num_legs)
        self.invariant_part.set_label(pos, label)
        return super().set_label(pos, label)

    def set_labels(self, labels: Sequence[list[str | None] | None] | list[str | None] | None):
        super().set_labels(labels)
        self.invariant_part.set_labels([*self._labels, *self._CHARGE_LEG_LABEL])
        return self
    
    def to_dense_block(self, leg_order: list[int | str] = None, dtype: Dtype = None) -> Block:
        if self.charged_state is None:
            raise ValueError('charged_state not specified.')
        inv_block = self.invariant_part.to_dense_block(leg_order=None, dtype=dtype)
        block = self.backend.block_tdot(inv_block, self.charged_state, [-1], [0])
        if dtype is not None:
            block = self.backend.block_to_dtype(block, dtype)
        if leg_order is not None:
            block = self.backend.block_permute_axes(block, self._as_leg_idcs(leg_order))
        return block

    def to_dense_block_single_sector(self) -> Block:
        """Assumes a single-leg tensor living in a single sector and returns its components within
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
        return self.backend.block_item(self.charged_state) * block


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
                return x._elementwise_unary(
                    lambda block: getattr(x.backend, block_func)(block, *args, **kwargs),
                    maps_zero_to_zero=maps_zero_to_zero
                )
            elif is_scalar(x):
                return function(x, *args, **kwargs)
            raise TypeError(f'Expected DiagonalTensor or scalar. Got {type(x)}')
        return wrapped
    return decorator


# FUNCTIONS ON TENSORS


def add_trivial_leg(tens: Tensor,
                    legs_pos: int = None, *, codomain_pos: int = None, domain_pos: int = None,
                    label: str = None, is_dual: bool = False):
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
    if isinstance(tens, (DiagonalTensor, Mask)):
        return add_trivial_leg(tens.as_SymmetricTensor(), legs_pos, codomain_pos=codomain_pos,
                               domain_pos=domain_pos, label=label, is_dual=is_dual)

    res_num_legs = tens.num_legs + 1
    # parse position to format:
    #  - leg_pos: int,  0 <= leg_pos < res_num_legs
    #  - add_to_domain: bool
    #  - co_domain_pos: int, 0 <= co_domain_pos < num_[co]domain_legs
    #  - is_dual: bool, if the leg in the [co]domain should be dual
    if legs_pos is not None:
        assert codomain_pos is None and domain_pos is None
        legs_pos = _normalize_idx(legs_pos, res_num_legs)
        add_to_domain = (legs_pos > tens.num_codomain_legs)
        if add_to_domain:
            co_domain_pos = res_num_legs - 1 - legs_pos
        else:
            co_domain_pos = legs_pos
    elif codomain_pos is not None:
        assert legs_pos is None and domain_pos is None
        res_codomain_legs = tens.num_codomain_legs + 1
        codomain_pos = _normalize_idx(codomain_pos, res_codomain_legs)
        add_to_domain = False
        co_domain_pos = codomain_pos
        legs_pos = codomain_pos
    elif domain_pos is not None:
        assert legs_pos is None and codomain_pos is None
        res_domain_legs = tens.num_domain_legs + 1
        domain_pos = _normalize_idx(domain_pos, res_domain_legs)
        add_to_domain = True
        co_domain_pos = domain_pos
        legs_pos = res_num_legs - 1 - domain_pos
    else:
        add_to_domain = False
        co_domain_pos = 0
        legs_pos = 0

    if isinstance(tens, ChargedTensor):
        if add_to_domain:
            # domain[0] is the charge leg, so we need to add 1
            inv_part = add_trivial_leg(tens.invariant_part, domain_pos=co_domain_pos + 1, label=label,
                                       is_dual=is_dual)
        else:
            inv_part = add_trivial_leg(tens.invariant_part, codomain_pos=co_domain_pos, label=label,
                                       is_dual=is_dual)
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
        tens, legs_pos=legs_pos, add_to_domain=add_to_domain, co_domain_pos=co_domain_pos,
        new_codomain=codomain, new_domain=domain
    )
    return SymmetricTensor(
        data, codomain=codomain, domain=domain, backend=tens.backend,
        labels=[*tens.labels[:legs_pos], label, *tens.labels[legs_pos:]],
    )


@_elementwise_function(block_func='block_angle', maps_zero_to_zero=True)
def angle(x: _ElementwiseType) -> _ElementwiseType:
    """The angle of a complex number, :ref:`elementwise <diagonal_elementwise>`.

    The counterclockwise angle from the positive real axis on the complex plane in the
    range (-pi, pi] with a real dtype. The angle of `0.` is `0.`.
    """
    return np.angle(x)


def almost_equal(tensor_1: Tensor, tensor_2: Tensor, rtol: float = 1e-5, atol=1e-8,
                 allow_different_types: bool = False) -> bool:
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
                    backend.block_item(tensor_2.charged_state) * tensor_1.invariant_part,
                    backend.block_item(tensor_1.charged_state) * tensor_2.invariant_part,
                    rtol=rtol, atol=atol
                )
            raise NotImplementedError  # TODO

    msg = f'Incompatible types: {tensor_1.__class__.__name__} and {tensor_2.__class__.__name__}'
    raise TypeError(msg)


def apply_mask(tensor: Tensor, mask: Mask, leg: int | str) -> Tensor:
    """Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.

    The mask must be a projection, i.e. its large leg is at the bottom.
    We apply the mask via map composition::

        |                                │   │   │   │   │
        |      │   │  ┏┷┓               ┏┷━━━┷━━━┷━━━┷━━━┷┓          │   │   │
        |      │   │  ┃M┃               ┃ tensor          ┃         ┏┷━━━┷━━━┷┓
        |      │   │  ┗┯┛               ┗┯━━━┯━━━━━━━━━━━┯┛         ┃ tensor  ┃
        |     ┏┷━━━┷━━━┷┓       OR       │   │   ╭───╮   │    ==    ┗┯━━━┯━━━┯┛
        |     ┃ tensor  ┃                │   │  ┏┷┓  │   │           │ ┏━┷━┓ │
        |     ┗┯━━━┯━━━┯┛                │   │  ┃M┃  │   │           │ ┃M.T┃ │
        |      │   │   │                 │   │  ┗┯┛  │   │           │ ┗━┯━┛ │
        |                                │   ╰───╯   │   │

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
    TODO add arg to specify the label? Now we just use the same as from `tensor`.

    Returns
    -------
    A masked tensor of the same type as `tensor` (exception: `DiagonalTensor`s are converted to
    `SymmetricTensor`s before masking). The leg order and labels are the same as on `tensor`.
    The masked leg is *smaller* (or equal) than before.

    See Also
    --------
    compose, tdot, scale_axis, apply_mask_DiagonalTensor
    """
    in_domain, _, leg_idx = tensor._parse_leg_idx(leg)

    if isinstance(tensor, ChargedTensor):
        inv_part = apply_mask(tensor.invariant_part, mask, leg_idx)
        return ChargedTensor(inv_part, tensor.charged_state)
    
    assert mask.large_leg == tensor.get_leg(leg_idx)
    assert mask.is_projection

    if isinstance(tensor, DiagonalTensor):
        tensor = tensor.as_SymmetricTensor()
    if isinstance(tensor, Mask):
        raise NotImplementedError  # TODO
    if not isinstance(tensor, SymmetricTensor):
        raise TypeError(f'Invalid tensor type: {type(tensor).__name__}')
    
    if in_domain:
        mask = transpose(mask)
    backend = get_same_backend(tensor, mask)
    data, codomain, domain = backend.apply_mask_to_SymmetricTensor(tensor, mask, leg_idx)
    return SymmetricTensor(data=data, codomain=codomain, domain=domain, backend=backend,
                           labels=tensor.labels)


def apply_mask_DiagonalTensor(tensor: DiagonalTensor, mask: Mask) -> DiagonalTensor:
    """Apply a mask to *both* legs of a diagonal tensor.

    The mask must be a projection, i.e. its large leg is in the domain, at the bottom.
    we apply the mask via map composition::

        |     ┏━━┷━━┓
        |     ┃  M  ┃
        |     ┗━━┯━━┛
        |     ┏━━┷━━┓
        |     ┃  D  ┃
        |     ┗━━┯━━┛
        |     ┏━━┷━━┓
        |     ┃ M.hc┃
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
    assert mask.is_projection
    assert mask.large_leg == tensor.leg
    backend = get_same_backend(tensor, mask)
    return DiagonalTensor(
        data=backend.apply_mask_to_DiagonalTensor(tensor, mask),
        leg=mask.small_leg, backend=backend, labels=tensor.labels
    )


def bend_legs(tensor: Tensor, num_codomain_legs: int = None, num_domain_legs: int = None) -> Tensor:
    """Move legs between codomain and domain without changing the order of ``tensor.legs``.

    For example::

        |        │   ╭───────────╮
        |        │   │   ╭───╮   │    ==    bend_legs(T, num_codomain_legs=1)
        |       ┏┷━━━┷━━━┷┓  │   │    
        |       ┃    T    ┃  │   │    ==    bend_legs(T, num_domain_legs=5)
        |       ┗┯━━━┯━━━┯┛  │   │
        |        │   │   │   │   │

    or::

        |        │   │   │   │
        |       ┏┷━━━┷━━━┷┓  │
        |       ┃    T    ┃  │        ==    bend_legs(T, num_codomain_legs=4)
        |       ┗┯━━━┯━━━┯┛  │
        |        │   │   ╰───╯        ==    bend_legs(T, num_codomain_legs=2)

    Parameters
    ----------
    tensor:
        The tensor to modify
    num_codomain_legs, num_domain_legs: int, optional
        The desired number of legs in the (co-)domain. Only one is required.

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
    return _permute_legs(tensor,
                         codomain=range(num_codomain_legs),
                         domain=reversed(range(num_codomain_legs, tensor.num_legs)),
                         err_msg='This should never raise.')


def check_same_legs(t1: Tensor, t2: Tensor) -> tuple[list[int], list[int]] | None:
    """Check if two tensors have the same legs.

    If there are matching labels in mismatched positions (which indicates that the leg order
    is mixed up by accident), the error message is amended accordingly on mismatched legs.
    If the legs still match regardless, a warning is issued.
    """
    incompatible_labels = False
    for n1, l1 in enumerate(t1._labels):
        n2 = t2._labelmap.get(l1, None)
        if n2 is None:
            # either l1 is None or l1 not in l2.labels
            continue
        if n2 != n1:
            incompatible_labels = True
            break
    same_legs = (t1.domain == t2.domain and t1.codomain == t1.codomain)
    if not same_legs:
        msg = 'Incompatible legs. '
        if incompatible_labels:
            msg += f'Should you permute_legs first? {t1.labels=}  {t2.labels=}'
        raise ValueError(msg)
    if incompatible_labels:
        logger.warning('Compatible legs with permuted labels detected. Double check your leg order!',
                       stacklevel=3)
    # done


def combine_legs(tensor: Tensor,
                 *which_legs: list[int | str],
                 combined_spaces: list[ProductSpace | None] = None,
                 levels: list[int] | dict[str | int, int] = None,
                 ) -> Tensor:
    """Combine (multiple) groups of legs, each to a :class:`ProductSpace`.

    If the legs to be combined are contiguous to begin with, the combination is just a grouping
    of the legs::

        |        ║          │    ║ 
        |       ╭╨──┬───╮   │   ╭╨──╮
        |       0   1   2   3   4   5
        |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
        |      ┃          T          ┃    ==   combine_legs(T, [0, 1, 2], [4, 5], [7, 8, 9])
        |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
        |      11  10   9   8   7   6
        |       │   │   ╰───┴──╥╯   │
        |       │   │          ║    │

    In the general case, the legs are permuted first, to match that leg order.
    If the symmetry does not have symmetric braids, the `levels` are required to specify the
    chirality of the braids, like in :func:`permute_legs`. For example::

        |             │      ║    │     │     │
        |             │     ╭╨┬─╮ │     │     │
        |             │     │ ╰─│─│─────│─────│───╮
        |       ╭─────│─────│───╯ │     │     │   │
        |       0     1     2     3     4     5   │
        |      ┏┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷┓  │
        |      ┃               T               ┃  │    ==   combine_legs(T, [2, 6, 0], [7, 10, 8])
        |      ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛  │
        |      11    10     9     8     7     6   │
        |       │     ╰─────│─────│──╮  │     ╰───╯
        |       │           │     ╰──┴─╥╯
        |       │           │          ║ 

    TODO think again about the order of the product on the domain.
         i.e. is the picture above given by ``[7, 10, 8]`` or ``[8, 10, 7]``?

    .. warning ::
        Combining legs introduces a basis-transformation. This is important to consider if
        you convert to a dense block (e.g. via :meth:`Tensor.to_dense_block`).
        In particular it is (in general) impossible to obtain
        ``some_tens.combine_legs(...).to_numpy()`` via
        ``some_tens.to_numpy().transpose(transp).reshape(new_shape)``.
        See :meth:`ProductSpace.get_basis_transformation`.

    Parameters
    ----------
    tensor:
        The tensor whose legs should be combined.
    *which_legs : list of {int | str}
        One or more groups of legs to combine.
    combined_spaces: list of {ProductSpace | None}, optional
        For each group of `legs`, the resulting ProductSpace can be passed to avoid recomputation.
        Must be the same legs as on the result described below. In particular, choosing the right
        duality depends on if the resulting combined leg ends up in domain or codomain.
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
        msg = (f'Converting {tensor.__class__.__name__} to SymmetricTensor for combine_legs. '
               f'To suppress this warning, explicitly use as_SymmetricTensor() first.')
        warnings.warn(msg, stacklevel=2)
        tensor = tensor.as_SymmetricTensor()

    which_legs = [tensor.get_leg_idcs(group) for group in which_legs]

    if isinstance(tensor, ChargedTensor):
        # note: its important to parse negative integers before via tensor.get_leg_idcs, since
        #       the invariant part has an additional leg.
        inv_part = combine_legs(
            tensor.invariant_part, *which_legs, combined_spaces=combined_spaces
        )
        return ChargedTensor(inv_part, charged_state=tensor.charged_state)

    # 2) permute legs such that the groups are contiguous and fully in codomain or fully in domain
    # ==============================================================================================

    if combined_spaces is None:
        combined_spaces = [None] * len(which_legs)

    # 2a) populate the following data structures:
    domain_groups = {}  # {domain_pos: (leg_idcs_to_combine, combined_ProductSpace)}
    domain_skip = []  # [domain_pos to skip, bc they are part of a group[1:]]
    codomain_groups = {}  # same as above, but for codomain
    codomain_skip = []  # same as above, but for codomain
    
    for n, group in enumerate(which_legs):
        combine_to_domain, combined_pos, first_leg = tensor._parse_leg_idx(group[0])
        group_idcs = [first_leg]
        to_combine = []  # the legs we need to combine. duality adjusted if they need to be bent.
        if combine_to_domain:
            to_combine.append(tensor.domain.spaces[combined_pos])
        else:
            to_combine.append(tensor.codomain.spaces[combined_pos])
        for l in group[1:]:
            in_domain, pos, leg_idx = tensor._parse_leg_idx(l)
            group_idcs.append(leg_idx)
            if in_domain:
                domain_skip.append(pos)
                leg = tensor.domain.spaces[pos]
            else:
                codomain_skip.append(pos)
                leg = tensor.codomain.spaces[pos]
            if in_domain == combine_to_domain:
                to_combine.append(leg)
            else:
                to_combine.append(leg.dual)
        if combined_spaces[n] is None:
            combined = ProductSpace(to_combine, backend=tensor.backend)
        else:
            combined = combined_spaces[n]
            assert combined.spaces == to_combine
        if combine_to_domain:
            domain_groups[combined_pos] = (group_idcs, combined)
        else:
            codomain_groups[combined_pos] = (group_idcs, combined)

    # 2b) populate the following data structures:
    new_domain_combine = []  # list[tuple[list[domain_pos], ProductSpace]], instructions for after permuting
    new_codomain_combine = []  # same as above, but for codomain
    codomain_idcs = []  # [leg_idx], input for permute_legs
    domain_idcs = []  # [leg_idx], input for permute_legs
    
    for n in range(tensor.num_codomain_legs):
        # for codomain, n==leg_idx
        if n in codomain_skip:
            continue
        group = codomain_groups.get(n, None)
        if group is None:
            codomain_idcs.append(n)
            continue
        leg_idcs, combined = group
        start = len(codomain_idcs)
        num = len(leg_idcs)
        new_codomain_combine.append((range(start, start + num), combined))
        codomain_idcs.extend(leg_idcs)
    for n in range(tensor.num_domain_legs):
        if n in domain_skip:
            continue
        group = domain_groups.get(n, None)
        if group is None:
            leg_idx = tensor.num_legs - 1 - n
            domain_idcs.append(leg_idx)
            continue
        leg_idcs, combined = group
        start = len(domain_idcs)
        num = len(leg_idcs)
        new_domain_combine.append((range(start, start + num), combined))
        domain_idcs.extend(leg_idcs)

    # 2c) finally, do the permute
    tensor = _permute_legs(tensor, codomain_idcs, domain_idcs, levels=levels)

    # 3) build new domain and codomain, labels
    # ==============================================================================================

    # strategy:
    #   a) preserve list lengths to not invalidate the positions, fill with None
    #   b) remove the Nones
    # [a, b, c, d, e, f, g]
    #  -> [a, None, None, (b.c.d), e, None, (f.g)]
    #  -> [a, (b.c.d), e, (f.g)]
    
    domain_spaces = tensor.domain.spaces[:]
    codomain_spaces = tensor.codomain.spaces[:]
    domain_labels = tensor.domain_labels
    codomain_labels = tensor.codomain_labels
    for positions, combined in new_domain_combine:
        first, *_, last = positions
        label = _combine_leg_labels(domain_labels[first:last + 1])
        domain_spaces[first:last] = [None] * (last - first)
        domain_spaces[last] = combined
        domain_labels[first:last] = [None] * (last - first)
        domain_labels[last] = label
    for positions, combined in new_codomain_combine:
        first, *_, last = positions
        label = _combine_leg_labels(codomain_labels[first:last + 1])
        codomain_spaces[first:last] = [None] * (last - first)
        codomain_spaces[last] = combined
        codomain_labels[first:last] = [None] * (last - first)
        codomain_labels[last] = label
    domain_spaces = [s for s in domain_spaces if s is not None]
    codomain_spaces = [s for s in codomain_spaces if s is not None]
    domain_labels = [l for l in domain_labels if l is not None]
    codomain_labels = [l for l in codomain_labels if l is not None]
    
    domain = ProductSpace(domain_spaces, backend=tensor.backend,
                          _sectors=tensor.domain.sectors,
                          _multiplicities=tensor.domain.multiplicities)
    codomain = ProductSpace(codomain_spaces, backend=tensor.backend,
                            _sectors=tensor.codomain.sectors,
                            _multiplicities=tensor.codomain.multiplicities)

    # 4) Build the data / finish up
    # ==============================================================================================
    raise NotImplementedError  # TODO probably need to rework the backend method.
    data = tensor.backend.combine_legs(tensor, ...)
    return SymmetricTensor(data, codomain=codomain, domain=domain, backend=tensor.backend,
                           labels=[codomain_labels, domain_labels])


def combine_to_matrix(tensor: Tensor,
                      codomain: int | str | list[int | str] | None = None,
                      domain: int | str | list[int | str] | None = None,
                      levels: list[int] | dict[str | int, int] = None,
                      ) -> Tensor:
    """Combine legs of a tensor into two combined ProductSpaces.

    The resulting tensor can be interpreted as a matrix, i.e. has two legs::

    |              ║
    |           ╭──╨────┬───────╮
    |       ╭───│───────│─────╮ │
    |       │   │   ╭───│───╮ │ │
    |       0   1   2   3   │ │ │
    |      ┏┷━━━┷━━━┷━━━┷┓  │ │ │
    |      ┃      T      ┃  │ │ │   =    combine_to_matrix(T, [1, 3, -1], [5, 2, 4, 0])
    |      ┗━━┯━━━┯━━━┯━━┛  │ │ │
    |         6   5   4     │ │ │
    |         ╰───│───│─────│─│─╯
    |             │ ╭─│─────╯ │
    |             ╰─┴─┴──╥────╯
    |                    ║

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
    res = _permute_legs(tensor, codomain=codomain, domain=domain, levels=levels)
    return combine_legs(res, range(res.num_codomain_legs), range(res.num_codomain_legs, res.num_legs))


def conj(tensor: Tensor):
    """TODO doc this.
    TODO do we even need this?
    """
    return dagger(transpose(tensor))  # OPTIMIZE


def dagger(tensor: Tensor) -> Tensor:
    r"""The hermitian conjugate tensor, a.k.a the dagger of a tensor.

    For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
    the hermitian conjugate matrix :math:`(M^\dagger)_{i,j} = \bar{M}_{j, i}` .
    For a tensor ``A: X -> Y`` the dagger is a map ``dagger(A): Y -> X``.
    Graphically::

        |        a   b   c             e   d
        |        │   │   │             │   │
        |       ┏┷━━━┷━━━┷┓         ┏━━┷━━━┷━━┓
        |       ┃    A    ┃         ┃dagger(A)┃
        |       ┗━━┯━━━┯━━┛         ┗┯━━━┯━━━┯┛
        |          │   │             │   │   │
        |          e   d             a   b   c

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
    then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*'].
    """
    if isinstance(tensor, Mask):
        return Mask(
            data=tensor.backend.mask_dagger(tensor),
            space_in=tensor.codomain[0], space_out=tensor.domain[0],
            is_projection=not tensor.is_projection, backend=tensor.backend,
            labels=[_dual_leg_label(l) for l in reversed(tensor._labels)]
        )
    if isinstance(tensor, DiagonalTensor):
        res = tensor._elementwise_unary(tensor.backend.block_conj, maps_zero_to_zero=True)
        res.set_labels([_dual_leg_label(l) for l in reversed(tensor._labels)])
        return res
    if isinstance(tensor, SymmetricTensor):
        return SymmetricTensor(
            data=tensor.backend.dagger(tensor),
            codomain=tensor.domain, domain=tensor.codomain,
            backend=tensor.backend,
            labels=[_dual_leg_label(l) for l in reversed(tensor._labels)]
        )
    if isinstance(tensor, ChargedTensor):
        inv_part = dagger(tensor.invariant_part)  # charge_leg ends up as codomain[0] and is dual.
        inv_part.set_label(0, ChargedTensor._CHARGE_LEG_LABEL)
        inv_part = move_leg(inv_part, 0, domain_pos=0)
        charged_state = tensor.charged_state
        if charged_state is not None:
            charged_state = tensor.backend.block_conj(charged_state)
        return ChargedTensor(inv_part, charged_state)
    raise TypeError


def compose(tensor1: Tensor, tensor2: Tensor, relabel1: dict[str, str] = None,
            relabel2: dict[str, str] = None) -> Tensor:
    r"""Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.

    Graphically::

        |        │   │   │   │
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃   tensor1   ┃
        |       ┗━━━━┯━━━┯━━━━┛
        |            │   │
        |       ┏━━━━┷━━━┷━━━━┓
        |       ┃   tensor2   ┃
        |       ┗━━┯━━━┯━━━┯━━┛
        |          │   │   │

    Returns
    -------
    The composite map :math:`T_1 \circ T_2` from ``tensor2.domain`` to ``tensor1.codomain``.

    See Also
    --------
    tdot, apply_mask, scale_axis
    """
    if tensor1.domain != tensor2.codomain:
        raise ValueError('Incompatible legs')
    
    res_labels = [[relabel1.get(l, l) for l in tensor1.codomain_labels],
                  [relabel2.get(l, l) for l in tensor2.domain_labels]]
    if isinstance(tensor1, Mask):
        return apply_mask(tensor2, tensor1, 0).set_labels(res_labels)
    if isinstance(tensor2, Mask):
        return apply_mask(tensor1, tensor2, -1).set_labels(res_labels)

    if isinstance(tensor1, DiagonalTensor):
        return scale_axis(tensor2, tensor1, 0).set_labels(res_labels)
    if isinstance(tensor2, DiagonalTensor):
        return scale_axis(tensor1, tensor2, -1).set_labels(res_labels)

    if isinstance(tensor1, ChargedTensor) or isinstance(tensor2, ChargedTensor):
        # OPTIMIZE dedicated implementation?
        return tdot(tensor1, tensor2,
                        list(reversed(range(tensor1.num_codomain_legs, tensor1.num_legs))),
                        list(range(tensor2.num_codomain_legs)),
                        relabel1=relabel1, relabel2=relabel2)

    return _compose_SymmetricTensors(tensor1, tensor2, relabel1=relabel1, relabel2=relabel2)


def _compose_SymmetricTensors(tensor1: SymmetricTensor, tensor2: SymmetricTensor,
                              relabel1: dict[str, str] = None, relabel2: dict[str, str] = None
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
        
    backend = get_same_backend(tensor1, tensor2)
    return SymmetricTensor(
        data=backend.compose(tensor1, tensor2),  # TODO impl, rename
        codomain=tensor1.codomain, domain=tensor2.domain, backend=backend,
        labels=[labels_codomain, labels_domain]
    )


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
        # TODO assumes symmetry can be dropped!
        p = p.to_numpy()  # OPTIMIZE
    else:
        p = np.asarray(p)
        p = np.real_if_close(p)
    p = p[p > 1e-30]  # for stability of log
    if n == 1:
        return -np.inner(np.log(p), p)
    if n == np.inf:
        return -np.log(np.max(p))
    return np.log(np.sum(p ** n)) / (1. - n)


def get_same_backend(*tensors: Tensor, error_msg: str = 'Incompatible backends.') -> Backend:
    """If the given tensors have the same backend, return it. Raise otherwise."""
    if len(tensors) == 0:
        raise ValueError('Need at least one tensor')
    backend = tensors[0].backend
    if not all(tens.backend == backend for tens in tensors[1:]):
        raise ValueError(error_msg)
    return backend


@_elementwise_function(block_func='block_imag', maps_zero_to_zero=True)
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
    It is thus equivalent to, but more efficient than ``trace(dot(A, B))``.

    TODO should we allow an arg here to permute the legs before inner?

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
    if do_dagger:
        assert A.codomain == B.codomain
        assert A.domain == B.domain
    else:
        assert A.domain == B.codomain
        assert A.codomain == B.domain
    
    if isinstance(A, (DiagonalTensor, Mask)):
        # in this case, there is no benefit to having a dedicated backend function,
        # as the dot is cheap
        if do_dagger:
            return trace(compose(dagger(A), B))
        return trace(compose(A, B))
    if isinstance(B, (DiagonalTensor, Mask)):
        # same argument as above.
        if do_dagger:
            return conj(trace(compose(dagger(B), A)))
        return trace(compose(A, B))

    # remaining cases: both are either SymmetricTensor or ChargedTensor
    
    if isinstance(A, ChargedTensor) and isinstance(B, ChargedTensor):
        raise NotImplementedError  # TODO
    
    if isinstance(B, ChargedTensor):
        # reduce to the case where A is charged and B is not
        if do_dagger:
            return conj(inner(B, A))
        return inner(B, A, do_dagger=False)

    if isinstance(A, ChargedTensor):  # and B is a SymmetricTensor
        raise NotImplementedError   # TODO

    # remaining case: both are SymmetricTensor
    return get_same_backend(A, B).inner(A, B, do_dagger=do_dagger)


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
        if not np.all(obj.domain.sectors == obj.codomain.sectors):
            return False
        if not np.all(obj.domain.multiplicities == 1):
            return False
        if not np.all(obj.codomain.multiplicities == 1):
            return False
        return True
    return isinstance(obj, Number)


def item(tensor: Tensor) -> float | complex | bool:
    if not is_scalar(tensor):
        raise ValueError('Not a scalar')
    if isinstance(tensor, Mask):
        return Mask.any(tensor)
    if isinstance(tensor, (DiagonalTensor, SymmetricTensor)):
        return tensor.backend.item(tensor)
    if isinstance(tensor, ChargedTensor):
        if tensor.charged_state is None:
            raise ValueError('Can not compute .item of ChargedTensor with unspecified charged_state.')
        inv_block = tensor.invariant_part.to_dense_block()
        res = tensor.backend.block_tdot(tensor.charged_state, inv_block, 0, -1)
        return tensor.backend.block_item(res)
    raise TypeError


def linear_combination(a: Number, v: Tensor, b: Number, w: Tensor):
    """The linear combination ``a * v + b * w``"""
    # Note: We implement Tensor.__add__ and Tensor.__sub__ in terms of this function, so we cant
    #       use them (or the ``+`` and ``-`` operations) here.
    if (not isinstance(a, Number)) or (not isinstance(b, Number)):
        msg = f'unsupported scalar types: {type(a).__name__}, {type(b).__name__}'
        raise TypeError(msg)
    if isinstance(v, DiagonalTensor) and isinstance(w, DiagonalTensor):
        return DiagonalTensor._binary_operand(
            v, w, func=lambda _v, _w: a * _v + b * _w, operand='linear_combination'
        )
    if isinstance(v, ChargedTensor) and isinstance(w, ChargedTensor):
        if v.charge_leg != w.charge_leg:
            raise ValueError('Can not add ChargedTensors with different dummy legs')
        if (v.charged_state is None) != (w.charged_state is None):
            raise ValueError('Can not add ChargedTensors with unspecified and specified charged_state')
        if v.charged_state is None:
            inv_part = linear_combination(a, v.invariant_part, b, w.invariant_part)
            return ChargedTensor(inv_part, None)
        if v.charge_leg.dim == 1:
            factor = v.backend.block_item(v.charged_state) / v.backend.block_item(w.charged_state)
            inv_part = linear_combination(a, v.invariant_part, factor * b, w.invariant_part)
            return ChargedTensor(inv_part, v.charged_state)
        raise NotImplementedError
    if isinstance(v, ChargedTensor) or isinstance(w, ChargedTensor):
        raise TypeError('Can not add ChargedTensor and non-charged tensor.')
    # Remaining case: Mask, DiagonalTensor (but not both), SymmetricTensor
    v = v.as_SymmetricTensor()
    w = w.as_SymmetricTensor()
    backend = get_same_backend(v, w)
    assert v.codomain == w.codomain, 'Mismatched domain'
    assert v.domain == w.domain, 'Mismatched domain'
    return SymmetricTensor(
        backend.linear_combination(a, v, b, w),
        codomain=v.codomain, domain=v.domain, backend=backend,
        labels=_get_matching_labels(v._labels, w._labels)
    )


def move_leg(tensor: Tensor, which_leg: int | str, codomain_pos: int = None, *,
             domain_pos: int = None, levels: list[int] | dict[str | int, int] | None = None
             ) -> Tensor:
    """Move one leg of a tensor to a specified position.

    Graphically::

        |        │   ╭───│─╯ │
        |       ┏┷━━━┷━━━┷━━━┷┓
        |       ┃      T      ┃       ==    move_leg(T, 1, codomain_pos=-2)
        |       ┗┯━━━┯━━━┯━━━┯┛
        |        │   │   │   │

    Or::
    
        |        │   │   ╭───│───╮
        |       ┏┷━━━┷━━━┷━━━┷┓  │
        |       ┃      T      ┃  │    ==    move_leg(T, 2, domain_pos=1)
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
    """
    # TODO make this a separate backend function? Easier to determine move order for fusion tree.

    from_domain, co_domain_pos_from, leg_idx = tensor._parse_leg_idx(which_leg)
    if from_domain:
        new_codomain = list(range(tensor.num_codomain_legs))
        new_domain = [n for n in reversed(range(tensor.num_codomain_legs, tensor.num_legs))
                      if n != leg_idx]
    else:
        new_codomain = [n for n in range(tensor.num_codomain_legs) if n != leg_idx]
        new_domain = list(reversed(range(tensor.num_codomain_legs, tensor.num_legs)))
    #
    if codomain_pos is not None:
        if domain_pos is not None:
            raise ValueError('Can not specify both codomain_pos and domain_pos.')
        pos = _normalize_idx(codomain_pos, len(new_codomain) + 1)
        new_codomain[pos:pos] = [leg_idx]
    elif domain_pos is not None:
        pos = _normalize_idx(domain_pos, len(new_domain) + 1)
        new_domain[pos:pos] = [leg_idx]
        to_domain = True
    else:
        raise ValueError('Need to specify either codomain_pos or domain_pos.')
    #
    return permute_legs(tensor, new_codomain, new_domain, levels=levels)


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
            msg = ('norm of a ChargedTensor with unspecified charged_state is ambiguous. '
                   'Use e.g. norm(tensor.invariant_part).')
            raise ValueError(msg)
        if tensor.charge_leg.dim == 1:
            factor = tensor.backend.block_item(tensor.charged_state)
            return factor * tensor.backend.norm(tensor.invariant_part)
        else:
            # OPTIMIZE
            warnings.warn('Converting ChargedTensor to dense block for `norm`', stacklevel=2)
            return tensor.backend.block_norm(tensor.to_dense_block(), order=2)
    raise TypeError


def outer(tensor1: Tensor, tensor2: Tensor,
          relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
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
    raise NotImplementedError  # TODO


def _permute_legs(tensor: Tensor,
                  codomain: list[int | str] | None = None,
                  domain: list[int | str] | None = None,
                  levels: list[int] | dict[str | int, int] | None = None,
                  err_msg: str = None
                  ) -> Tensor:
    """Internal implementation of :func:`permute_legs` that allows to specify the error msg.

    Except for the additional `err_msg` arg, this has the same in-/outputs as :func:`permute_legs`.
    If the `levels` are needed, but not given, an error with the specified message is raised.
    This allows easier error handling when using ``_permute_legs`` as part of other functions.

    The default error message is appropriate to *other* contexts, other than :func:`permute_legs`.
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
            raise ValueError(f'Duplicate entries. By leg index: {", ".join(duplicates)}')
        if missing:
            raise ValueError(f'Missing legs. By leg index: {", ".join(missing)}')
    # Default error message
    if err_msg is None:
        err_msg = ('Legs can not be permuted automatically. '
                   'Explicitly use permute_legs() with specified levels first.')
    # Special case: if no legs move
    if codomain == list(range(tensor.num_codomain_legs)) and domain == list(range(tensor.num_codomain_legs, tensor.num_legs)):
        return tensor
        
    # Deal with other tensor types
    if isinstance(tensor, (DiagonalTensor, Mask)):
        if codomain == [0] and domain == [1]:
            return tensor
        if codomain == [1] and domain == [0]:
            return transpose(tensor)
        # other cases involve two legs either in the domain or codomain.
        # Cant be done with Mask / DiagonalTensor
        # TODO should we warn? is this expected?
        tensor = tensor.as_SymmetricTensor()
    if isinstance(tensor, ChargedTensor):
        if levels is not None:
            # assign the highest level to the charge leg. since it does not move, it should not matter.
            highest = max(levels) + 1
            levels = [*levels, highest]
        inv_part = _permute_legs(tensor.invariant_part, codomain=codomain, domain=[-1, *domain],
                                 levels=levels, err_msg=err_msg)
        return ChargedTensor(inv_part, charged_state=tensor.charged_state)
    # Remaining case: SymmetricTensor
    if levels is not None:
        if duplicate_entries(levels):
            raise ValueError('Levels must be unique.')
        if any(l < 0 for l in levels):
            raise ValueError('Levels must be non-negative.')
    data, new_codomain, new_domain = tensor.backend.permute_legs(
        tensor, codomain_idcs=codomain, domain_idcs=domain, levels=levels
    )
    if data is None:
        raise SymmetryError(err_msg)
    labels = [[tensor._labels[n] for n in codomain], [tensor._labels[n] for n in domain]]
    return SymmetricTensor(data, new_codomain, new_domain, backend=tensor.backend, labels=labels)


def permute_legs(tensor: Tensor, codomain: list[int | str] = None, domain: list[int | str] = None,
                 levels: list[int] | dict[str | int, int] = None):
    """Permute the legs of a tensor by braiding legs and bending lines.

    |       ╭───│───────│─────╮ │
    |       │   │   ╭───│───╮ │ │
    |       0   1   2   3   │ │ │
    |      ┏┷━━━┷━━━┷━━━┷┓  │ │ │
    |      ┃      T      ┃  │ │ │   =    permute_legs(T, [1, 3, -1], [5, 2, 4, 0])
    |      ┗━━┯━━━┯━━━┯━━┛  │ │ │
    |         6   5   4     │ │ │
    |         ╰───│───│─────│─│─╯
    |             │ ╭─│─────╯ │
        
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
    """
    return _permute_legs(tensor, codomain=codomain, domain=domain, levels=levels,
                         err_msg='The given permutation requires levels, but none were given.')


@_elementwise_function(block_func='block_real', maps_zero_to_zero=True)
def real(x: _ElementwiseType) -> _ElementwiseType:
    """The real part of a complex number, :ref:`elementwise <diagonal_elementwise>`."""
    return np.real(x)


@_elementwise_function(block_func='block_real_if_close', func_kwargs=dict(tol=100),
                       maps_zero_to_zero=True)
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


def scalar_multiply(a: Number, v: Tensor) -> Tensor:
    """The scalar multiplication ``a * v``"""
    if not isinstance(a, Number):
        msg = f'unsupported scalar type: {type(a).__name__}'
        raise TypeError(msg)
    if isinstance(v, DiagonalTensor):
        return DiagonalTensor._elementwise_unary(v, func=lambda _v: a * _v, maps_zero_to_zero=True)
    if isinstance(v, Mask):
        v = v.as_SymmetricTensor()
    if isinstance(v, ChargedTensor):
        if v.charged_state is None:
            inv_part = scalar_multiply(a, v.invariant_part)
            charged_state = None
        else:
            inv_part = v.invariant_part
            charged_state = v.backend.block_mul(a, v.charged_state)
        return ChargedTensor(inv_part, charged_state)
    # remaining case: SymmetricTensor
    return SymmetricTensor(
        v.backend.mul(a, v), codomain=v.codomain, domain=v.domain, backend=v.backend,
        labels=v._labels
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
    dot, tdot, apply_mask
    """
    # transpose if needed
    in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg)
    if in_domain:
        leg = tensor.domain[co_domain_idx]
    else:
        leg = tensor.codomain[co_domain_idx]
    if leg == diag.leg:
        pass
    elif leg == diag.leg.dual:
        diag = transpose(diag)
    else:
        raise ValueError('Incompatible legs')
    
    if isinstance(tensor, DiagonalTensor):
        return (tensor * diag).set_labels(tensor.labels)
    if isinstance(tensor, Mask):
        # leg == 0 -> mask is on leg 1 of diagonal and vice versa
        return apply_mask(diag.as_SymmetricTensor(), tensor, 1 - leg_idx)
    if isinstance(tensor, ChargedTensor):
        inv_part = scale_axis(tensor.invariant_part, diag, leg_idx)
        return ChargedTensor(inv_part, tensor.charged_state)
    backend = get_same_backend(tensor, diag)
    return SymmetricTensor(backend.scale_axis(tensor, diag, leg_idx), codomain=tensor.codomain,
                           domain=tensor.domain, backend=backend, labels=tensor._labels)


def split_legs(tensor: Tensor, legs: list[int | str] = None):
    """Split legs that were previously combined using :func:`combine_legs`.

    |       │   │   │   │   │   │
    |       ╰╥──┴───╯   │   ╰╥──╯
    |      ┏━┷━━━━━━━━━━┷━━━━┷━━━┓
    |      ┃          T          ┃    ==    split_legs(T, [0, 2, 5])
    |      ┗┯━━━┯━━━━━━━━━━┯━━━━┯┛
    |       │   │   ╭───┬──╨╮   │
    |       │   │   │   │   │   │

    This is the inverse of :func:`combine_legs`, *only up to possible :func:`permute_legs`*!

    Parameters
    ----------
    tensor
        The tensor to act on.
    legs: list of int | str
        Which legs to split. If ``None`` (default), all those legs that are :class:`ProductSpace`s
        are split.
    """
    if isinstance(tensor, (DiagonalTensor, Mask)):
        tensor = tensor.as_SymmetricTensor()
    if isinstance(tensor, ChargedTensor):
        if legs is not None:
            legs = tensor.get_leg_idcs(legs)
        return ChargedTensor(split_legs(tensor.invariant_part, legs), tensor.charged_state)
    # remaining case: SymmetricTensor.
    raise NotImplementedError  # TODO


@_elementwise_function(block_func='block_sqrt', maps_zero_to_zero=True)
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
        tensor = tensor.as_SymmetricTensor()
    if isinstance(tensor, ChargedTensor):
        return ChargedTensor(
            squeeze_legs(tensor.invariant_part, legs=legs),
            tensor.charged_state
        )
    # Remaining case: SymmetricTensor
    raise NotImplementedError  # TODO


def tdot(tensor1: Tensor, tensor2: Tensor,
         legs1: int | str | list[int | str], legs2: int | str | list[int | str],
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None):
    """General tensor contraction, connecting arbitrary pairs of (matching!) legs.

    For example::

        |        │   ╭───│───│──╮
        |        0   1   2   │  │
        |       ┏┷━━━┷━━━┷┓  │  │
        |       ┃    A    ┃  │  │
        |       ┗┯━━━┯━━━┯┛  │  │
        |        5   4   3   │  │
        |    ╭───╯ ╭─╯   ╰───╯  │
        |    │     │   ╭─────╮  │    ==    tdot(A, B, [1, 4, 5], [3, 0, 4])
        |    │     0   1     │  │
        |    │  ┏━━┷━━━┷━━┓  │  │
        |    │  ┃    B    ┃  │  │
        |    │  ┗┯━━━┯━━━┯┛  │  │
        |    │   4   3   2   │  │
        |    ╰───╯   ╰───│───│──╯

    Parameters
    ----------
    tensor1, tensor2: Tensor
        The two tensor to contract
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
    compose, apply_mask, scale_axis
    """
    # parse legs to list[int] and check they are valid
    legs1 = tensor1.get_leg_idcs(to_iterable(legs1))
    legs2 = tensor2.get_leg_idcs(to_iterable(legs2))
    if duplicate_entries(legs1) or duplicate_entries(legs2):
        raise ValueError(f'Duplicate leg entries.')
    num_contr = len(legs1)
    assert len(legs2) == num_contr

    # Deal with Masks: either return or reduce to SymmetricTensor
    if isinstance(tensor1, Mask):
        if num_contr == 0:
            warnings.warn('Converting Mask to SymmetricTensor for non-contracting contract()')
            tensor1 = tensor1.as_SymmetricTensor()
        if num_contr == 1:
            res = apply_mask(tensor2, tensor1, legs2[0])
            res.set_label(legs2[0], tensor1.labels[1 - legs1[0]])
            return permute_legs(res, codomain=legs1)
        if num_contr == 2:
            large_leg_contr = legs1.index(1) if tensor1.is_projection else legs1.index(0)
            res = apply_mask(tensor2, tensor1, legs2[large_leg_contr])
            res = trace(res, legs2)
            return bend_legs(res, num_codomain_legs=0)
    if isinstance(tensor2, Mask):
        if num_contr == 0:
            warnings.warn('Converting Mask to SymmetricTensor for non-contracting contract()')
            tensor2 = tensor2.as_SymmetricTensor()
        if num_contr == 1:
            large_leg_contr = legs2.index(1) if tensor2.is_projection else legs2.index(0)
            res = apply_mask(tensor1, tensor2, legs1[large_leg_contr])
            res.set_label(legs1[0], tensor2.labels[1 - legs2[0]])  # set to the uncontracted labels
            return permute_legs(res, domain=legs2)
        if num_contr == 2:
            res = apply_mask(tensor1, tensor2, legs1[0])
            res = trace(res, legs1)
            return bend_legs(res, num_domain_legs=0)

    # Deal with DiagonalTensor: either return or reduce to SymmetricTensor
    if isinstance(tensor1, DiagonalTensor):
        if num_contr == 0:
            warnings.warn('Converting DiagonalTensor to SymmetricTensor for non-contracting contract()')
            tensor1 = tensor1.as_SymmetricTensor()
        if num_contr == 1:
            res = scale_axis(tensor2, tensor1, legs2[0])
            res.set_label(legs2[0], tensor1.labels[1 - legs1[0]])
            return permute_legs(res, codomain=legs1)
        if num_contr == 2:
            res = scale_axis(tensor2, tensor1, legs2[0])
            res = trace(res, legs2)
            return bend_legs(res, num_codomain_legs=0)
    if isinstance(tensor2, DiagonalTensor):
        if num_contr == 0:
            warnings.warn('Converting DiagonalTensor to SymmetricTensor for non-contracting contract()')
            tensor2 = tensor2.as_SymmetricTensor()
        if num_contr == 1:
            res = scale_axis(tensor1, tensor2, legs1[0])
            res.set_label(legs1[0], tensor2.labels[1 - legs2[0]])
            return permute_legs(res, domain=legs2)
        if num_contr == 2:
            res = scale_axis(tensor1, tensor2, legs1[0])
            res = trace(res, legs1)
            return bend_legs(res, num_domain_legs=0)

    # Deal with ChargedTensor
    if isinstance(tensor1, ChargedTensor) and isinstance(tensor2, ChargedTensor):
        # note: its important that we have already used get_leg_idcs
        if (tensor1.charged_state is None) != (tensor2.charged_state is None):
            raise ValueError('Mismatched: specified and unspecified ChargedTensor.charged_state')
        c = ChargedTensor._CHARGE_LEG_LABEL
        c1 = c + '1'
        c2 = c + '2'
        inv_part = tdot(tensor1.invariant_part, tensor2.invariant_part, legs1=legs1, legs2=legs2,
                        relabel1={**relabel1, c: c1}, relabel2={**relabel2, c: c2})
        inv_part = move_leg(inv_part, c1, domain_pos=0)
        return ChargedTensor.from_two_charge_legs(
            inv_part, state1=tensor1.charged_state, state2=tensor2.charged_state,
        )
    if isinstance(tensor1, ChargedTensor):
        inv_part = tdot(tensor1.invariant_part, tensor2, legs1=legs1, legs2=legs2,
                            relabel1=relabel1, relabel2=relabel2)
        inv_part = move_leg(inv_part, ChargedTensor._CHARGE_LEG_LABEL, domain_pos=0)
        return ChargedTensor.from_invariant_part(inv_part, tensor1.charged_state)
    if isinstance(tensor2, ChargedTensor):
        inv_part = tdot(tensor1, tensor2.invariant_part, legs1=legs1, legs2=legs2,
                            relabel1=relabel1, relabel2=relabel2)
        return ChargedTensor.from_invariant_part(inv_part, tensor2.charged_state)

    # Remaining case: both are SymmetricTenor

    # OPTIMIZE actually, we only need to permute legs to *any* matching order.
    #          could use ``legs1[perm]`` and ``legs2[perm]`` instead, if that means fewer braids.
    tensor1 = _permute_legs(tensor1, domain=legs1)
    tensor2 = _permute_legs(tensor2, codomain=legs2)
    return _compose_SymmetricTensors(tensor1, tensor2, relabel1=relabel1, relabel2=relabel2)


def trace(tensor: Tensor,
          *pairs: Sequence[int | str],
          levels: list[int] | dict[str | int, int] | None = None):
    """Perform a (partial) trace.

    By default, require that ``tensor.domain == tensor.codomain`` and perform the full trace::

        |    ╭───────────────╮
        |    │   ╭─────────╮ │
        |    │   │   ╭───╮ │ │
        |   ┏┷━━━┷━━━┷┓  │ │ │
        |   ┃    A    ┃  │ │ │    ==    trace(A)
        |   ┗┯━━━┯━━━┯┛  │ │ │
        |    │   │   ╰───╯ │ │
        |    │   ╰─────────╯ │
        |    ╰───────────────╯

    If `pairs` are specified, a number of partial traces are performed::

        |    ╭───│───╮   ╭───╮
        |    0   1   2   3   │
        |   ┏┷━━━┷━━━┷━━━┷┓  │
        |   ┃      A      ┃  │    ==   trace(A, (0, 2), (3, 5), (-2, 4))
        |   ┗┯━━━┯━━━┯━━━┯┛  │
        |    7   6   5   4   │
        |    │   ╰───│───╯   │
        |    │       ╰───────╯

    Parameters
    ----------
    tensor: Tensor
        The tensor to act on
    *pairs:
        A number of pairs, each describing two legs via index or via label.
        Each pair is connected, realizing a partial trace
    levels:
        If `pairs` is given, the connectivity of the partial trace may induce braids.
        For symmetries with non-symmetric braiding, these levels are used to determine the
        chirality of those braids, like in :func:`permute_legs`.

    Returns
    -------
    If all legs are traced, a python scalar.
    If legs are left open, a tensor, whose legs are the untraced legs.
    """
    raise NotImplementedError('tensors.trace not implemented')  # TODO


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

        transpose(A).codomain == A.domain.dual == [V2.dual, V1.dual]  # if A.codomain == [V1, V2]
        transpose(A).domain == A.codomain.dual == [W2.dual, W1.dual]  # if A.domain == [W1, W2]
        transpose(A).legs == [V2.dual, V1.dual, W1, W2]  # compared to A.legs == [V1, V2, W2.dual, W1.dual]
        transpose(A).labels == [*reversed(A.domain_labels), *A.codomain_labels]

    Note that the resulting :attr:`Tensor.legs` depend not only on the input :attr:`Tensor.legs`,
    but also on how they are partitioned into domain and codomain.
    We use the "same" labels, up to the permutation.
    """
    labels = [*reversed(tensor.domain_labels), *tensor.codomain_labels]
    if isinstance(tensor, Mask):
        space_in, space_out, data = tensor.backend.mask_transpose(tensor)
        return Mask(data, space_in=space_in, space_out=space_out,
                    is_projection=not tensor.is_projection, backend=tensor.backend,
                    labels=labels)
    if isinstance(tensor, DiagonalTensor):
        # TODO implement this backend method.
        #      the result has dual leg, which means a permutation of sectors.
        dual_leg, data = tensor.backend.diagonal_transpose(tensor)
        return DiagonalTensor(data=data, leg=dual_leg, backend=tensor.backend, labels=labels)
    if isinstance(tensor, SymmetricTensor):
        data, codomain, domain = tensor.backend.transpose(tensor)
        return SymmetricTensor(data=data, codomain=codomain, domain=domain, backend=tensor.backend,
                               labels=labels)
    if isinstance(tensor, ChargedTensor):
        inv_part = transpose(tensor.invariant_part)
        inv_part = move_leg(inv_part, ChargedTensor._CHARGE_LEG_LABEL, domain_pos=0)
        return ChargedTensor(inv_part, tensor.charged_state)
    raise TypeError


def zero_like(tensor: Tensor) -> Tensor:
    """Return a zero tensor with same type, dtype, legs, backend and labels."""
    if isinstance(tensor, Mask):
        return Mask.from_zero(large_leg=tensor.large_leg, backend=tensor.backend, labels=tensor.labels)
    if isinstance(tensor, DiagonalTensor):
        return DiagonalTensor.from_zero(leg=tensor.leg, backend=tensor.backend, labels=tensor.labels,
                                        dtype=tensor.dtype)
    if isinstance(tensor, SymmetricTensor):
        return SymmetricTensor.from_zero(codomain=tensor.codomain, domain=tensor.domain,
                                         backend=tensor.backend, labels=tensor.labels,
                                         dtype=tensor.dtype)
    if isinstance(tensor, ChargedTensor):
        return ChargedTensor.from_zero(
            codomain=tensor.codomain, domain=tensor.domain, charge=tensor.charge_leg,
            charged_state=tensor.charged_state, backend=tensor.backend, labels=tensor.labels,
            dtype=tensor.dtype
        )
    raise TypeError


# INTERNAL HELPER FUNCTIONS


T = TypeVar('T')


def _combine_leg_labels(labels: list[str | None]) -> str:
    """the label that a combined leg should have"""
    return '(' + '.'.join(f'?{n}' if l is None else l for n, l in enumerate(labels)) + ')'


def _dual_label_list(labels: list[str | None]) -> list[str | None]:
    return [_dual_leg_label(l) for l in reversed(labels)]


def _dual_leg_label(label: str | None) -> str | None:
    """the label that a leg should have after conjugation"""
    if label is None:
        return None
    if label.startswith('(') and label.endswith(')'):
        return _combine_leg_labels(_dual_label_list(_split_leg_label(label)))
    if label.endswith('*'):
        return label[:-1]
    else:
        return label + '*'


def _get_matching_labels(labels1: list[str | None], labels2: list[str | None],
                         stacklevel: int = 1) -> list[str | None]:
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
        msg = (f'Conflicting labels at positions {", ".join(map(str, conflicts))} are dropped. '
               f'{labels1=}, {labels2=}.')
        logger.debug(msg, stacklevel=stacklevel + 1)
    return labels


def _is_valid_leg_label(label) -> bool:
    # TODO this correct?
    return label is None or isinstance(label, str)


def _normalize_idx(idx: int, length: int) -> int:
    assert -length <= idx < length, 'index out of bounds'
    if idx < 0:
        idx += length
    return idx


def _parse_idcs(idcs: T | Sequence[T], length: int, fill: T = slice(None, None, None)
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


class _TensorIndexHelper:
    """A helper class that redirects __getitem__ and __setitem__ to a Tensor.

    See :meth:`~tenpy.linalg.tensors.Tensor.with_legs`.
    """
    def __init__(self, tensor: Tensor, which_legs: list[int | str]):
        self.tensor = tensor
        self.which_legs = [tensor._parse_leg_idx(i)[2] for i in which_legs]

    def transform_idc(self, idcs):
        idcs = _parse_idcs(idcs, length=len(self.which_legs))
        res = [slice(None, None, None) for _ in range(self.tensor.num_legs)]
        for which_leg, idx in zip(self.which_legs, idcs):
            res[which_leg] = idx
        return res

    def __getitem__(self, idcs):
        return self.tensor.__getitem__(self.transform_idcs(idcs))

    def __setitem__(self, idcs, value):
        return self.tensor.__setitem__(self.transform_idcs(idcs), value)

