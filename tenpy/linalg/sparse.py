"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of a linear
operator, but only the action on a vector, i.e., a matrix-vector product `matvec`. Here we define
the strucuture of such a general operator, :class:`TenpyLinearOperator`, as it is used in our own
implementations of these algorithms (e.g., :mod:`~tenpy.linalg.krylov_based`). Moreover, the
:class:`FlatLinearOperator` allows to use all the scipy sparse methods by providing functionality
to convert flat numpy arrays to and from tenpy tensors.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABC, abstractmethod
from numbers import Number
import warnings
import numpy as np

from tenpy.linalg.tensors import AbstractTensor

from .tensors import AbstractTensor, Shape, Tensor, tdot, eye_like, zero_like
from .backends.abstract_backend import Dtype, AbstractBackend


__all__ = ['LinearOperator', 'LinearOperatorWrapper', 'SumLinearOperator',
           'ShiftedLinearOperator', 'ProjectedLinearOperator']


class LinearOperator(ABC):
    """Base class for a linear operator acting on tenpy tensors.

    Attributes
    ----------
    vector_shape : Shape
        The shape of tensors that this operator can act on
    dtype : Dtype
        The dtype of a full representation of the operator
    """
    def __init__(self, vector_shape: Shape, dtype: Dtype):
        self.vector_shape = vector_shape
        self.dtype = dtype

    @abstractmethod
    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        """Apply the linear operator to a "vector".

        We consider as vectors all tensors of the shape given by :attr:`vector_shape`,
        and in particular allow multi-leg tensors as "vectors".
        The result of `matvec` must be a tensor of the same shape.
        """
        ...

    @abstractmethod
    def to_tensor(self, **kw) -> AbstractTensor:
        """Compute a full tensor representation of the linear operator.
        
        Returns
        -------
        A tensor `t` with ``2 * N`` legs where ``N == self.vector_shape.num_legs``, such that
        ``self.matvec(vec)`` is equivalent to ``tdot(t, vec, list(range(N, 2 * N)), list(range(N)))``.
        """
        ...

    def to_matrix(self, backend: AbstractBackend = None) -> AbstractTensor:
        """The tensor representation of self, reshaped to a matrix."""
        # OPTIMIZE could find a way to store the ProductSpace and use it here
        N = self.vector_shape.num_legs
        return self.to_tensor(backend=backend).combine_legs(list(range(N)), list(range(N, 2 * N)))

    def adjoint(self) -> LinearOperator:
        """Return the hermitian conjugate operator.

        If `self` is hermitian, subclasses *can* choose to implement this to define
        the adjoint operator of `self` to be `self`.
        """
        raise NotImplementedError("No adjoint defined")


class TensorLinearOperator(LinearOperator):
    """Linear operator defined by a two-leg tensor with contractible legs.

    The matvec is defined by contracting one of the two legs of this tensor with the vector.
    This class is effectively a thin wrapper around tensors that allows them to be used as inputs
    for sparse linear algebra routines, such as lanczos.

    Parameter
    ---------
    tensor :
        The tensor that is contracted with the vector on matvec
    which_legs : int or str
        Which leg of `tensor` is to be contracted on matvec.
    """
    def __init__(self, tensor: Tensor, which_leg: int | str = -1):
        if tensor.num_legs > 2:
            raise ValueError('Expected a two-leg tensor')
        if not tensor.legs[0].can_contract_with(tensor.legs[1]):
            raise ValueError('Expected contractible legs')
        self.which_leg = which_leg = tensor.get_leg_idx(which_leg)
        self.other_leg = other_leg = 1 - which_leg
        self.tensor = tensor
        vector_shape = Shape(legs=[tensor.legs[other_leg]], labels=tensor.labels[other_leg])
        super().__init__(vector_shape=vector_shape, dtype=tensor.dtype)

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        assert vec.num_legs == 1
        return self.tensor.tdot(vec, self.which_leg, 0)

    def to_tensor(self, **kw) -> AbstractTensor:
        if self.tensor.which_leg == 1:
            return self.tensor
        return self.tensor.permute_legs([1, 0])

    def adjoint(self) -> TensorLinearOperator:
        return TensorLinearOperator(tensor=self.tensor.conj(), which_leg=self.other_leg)


def as_linear_operator(obj: LinearOperator | AbstractTensor) -> LinearOperator:
    """Converts an object to a :class:`LinearOperator`.

    The following objects can be converted::
        - :class:`LinearOperator` trivially
        - :class:`~tenpy.linalg.tensors.AbstractTensor` if they have exactly two legs which are contractible,
           by wrapping them in :class:`TensorLinearOperator`.
    """
    if isinstance(obj, LinearOperator):
        return obj
    if isinstance(obj, AbstractTensor):
        return TensorLinearOperator(tensor=obj, which_leg=-1)
    raise TypeError(f'Could not convert {type(obj)} to linear operator')


class LinearOperatorWrapper(LinearOperator, ABC):
    """Base class for wrapping around another :class:`LinearOperator`.

    Attributes which are not explicitly set, e.g. via `self.attribute = value` or by
    defining methods default to the attributes of the `original_operator`.

    This behavior is particularly useful when wrapping some concrete subclass of :class:`LinearOperator`,
    which defines additional attributes.
    Using this base class, we can define the wrappers below without considering those extra attributes.

    .. warning ::
        If there are multiple levels of wrapping operators, the order might be critical to get
        correct results; e.g. :class:`ProjectedLinearOperator` needs to be the outer-most
        wrapper to produce correct results and/or be efficient.

    Parameters
    ----------
    original_operator : :class:`LinearOperator`
        The original operator implementing the `matvec`.
    """
    def __init__(self, original_operator: LinearOperator):
        self.original_operator = original_operator
        # TODO (JU) should we call LinearOperator.__init__ or super().__init__ here?
        #      Its current implementation only sets attributes, which we dont need because
        #      we hack into __getattr__

    def __getattr__(self, name):
        # note: __getattr__ (unlike __getattribute__) is only called if the attribute is not
        #       found in the __dict__, so it is the fallback for attributes that are not explicitly set.
        return getattr(self.original_operator, name)

    def unwrapped(self, recursive: bool = True) -> LinearOperator:
        """Return the original :class:`LinearOperator`

        By default, unwrapping is done recursively, such that the result is *not* a `LinearOperatorWrapper`.
        """
        parent = self.original_operator
        if not recursive:
            return parent
        for _ in range(10000):
            try:
                parent = parent.unwrapped()
            except AttributeError:
                # parent has no :meth:`unwrapped`, so we can stop unwrapping
                return parent
        raise ValueError('maximum recursion depth for unwrapping reached')


class SumLinearOperator(LinearOperatorWrapper):
    """The sum of multiple operators"""
    def __init__(self, original_operator: LinearOperator, *more_operators: LinearOperator):
        super().__init__(original_operator=original_operator)
        assert all(op.vector_shape == original_operator.vector_shape for op in more_operators)
        self.more_operators = more_operators
        self.dtype = Dtype.common(original_operator.dtype, *(op.dtype for op in more_operators))

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        return sum((op.matvec(vec) for op in self.more_operators), self.original_operator.matvec(vec))

    def to_tensor(self, **kw) -> AbstractTensor:
        return sum((op.to_tensor(**kw) for op in self.more_operators),
                   self.original_operator.to_tensor(**kw))

    def adjoint(self) -> LinearOperator:
        return SumLinearOperator(self.original_operator.adjoint(),
                                 *(op.adjoint() for op in self.more_operators))


class ShiftedLinearOperator(LinearOperatorWrapper):
    """A shifted operator, i.e. ``original_operator + shift * identity``.

    This can be useful e.g. for better Lanczos convergence.
    """
    def __init__(self, original_operator: LinearOperator, shift: Number):
        if shift in [0, 0.]:
            warnings.warn('shift=0: no need for ShiftedLinearOperator', stacklevel=2)
        super().__init__(original_operator=original_operator)
        self.shift = shift
        if np.iscomplexobj(shift):
            self.dtype = original_operator.dtype.to_complex

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        return self.original_operator.matvec(vec) + self.shift * vec

    def to_tensor(self, **kw) -> AbstractTensor:
        res = self.original_operator.to_tensor(**kw)
        return res + self.shift * eye_like(res)

    def adjoint(self):
        return ShiftedLinearOperator(original_operator=self.original_operator.adjoint(),
                                          shift=np.conj(self.shift))


class ProjectedLinearOperator(LinearOperatorWrapper):
    """Projected version ``P H P + penalty * (1 - P)`` of an original operator ``H``.

    The projector ``P = 1 - sum_o |o> <o|`` is given in terms of a set :attr:`ortho_vecs` of vectors
    ``|o>``.
    
    The result is that all vectors from the subspace spanned by the :attr:`ortho_vecs` are eigenvectors
    with eigenvalue `penalty`, while the eigensystem in the "rest" (i.e. in the orthogonal complement
    to that subspace) remains unchanged.
    This can be used to exclude the :attr:`ortho_vecs` from extremal eigensolvers, i.e. to find
    the extremal eigenvectors among those that are orthogonal to the :attr:`ortho_vecs`.
    In previous versions of tenpy, this behavior was achieved by an argument called `orthogonal_to`.

    Parameters
    ----------
    original_operator : :class:`LinearOperator`-like
        The original operator, denoted ``H`` in the summary above.
    ortho_vecs : list of :class:`~tenpy.linalg.tensors.AbstractTensor`
        The list of vectors spanning the projected space.
        They need not be orthonormal, as Gram-Schmidt is performed on them explicitly.
    penalty : complex, optional
        See summary above. Defaults to ``None``, which is equivalent to ``0.``.
    """
    def __init__(self, original_operator: LinearOperator, ortho_vecs: list[AbstractTensor],
                 penalty: Number = None):
        if len(ortho_vecs) == 0:
            warnings.warn('empty ortho_vecs: no need for ProjectedLinearOperator', stacklevel=2)
        original_operator = as_linear_operator(original_operator)
        super().__init__(original_operator=original_operator)
        assert all(v.shape == original_operator.vector_shape for v in ortho_vecs)
        self.ortho_vecs = gram_schmidt(ortho_vecs)
        self.penalty = penalty

    def matvec(self, vec: AbstractTensor) -> AbstractTensor:
        res = vec
        # form ``P vec`` and keep coefficients for later use in the penalty term
        coefficients = []
        for o in self.ortho_vecs:
            c = o.inner(res)
            coefficients.append(c)
            res = res - c * o
        # ``H P vec``
        res = self.original_operator.matvec(res)
        # ``P H P vec``
        for o in self.ortho_vecs:
            res = res - o.inner(res) * o
        if self.penalty is not None:
            for c, o in zip(coefficients, self.ortho_vecs):
                res = res + self.penalty * c * o
        return res

    def to_tensor(self, **kw) -> AbstractTensor:
        res = self.original_operator.to_tensor(**kw)
        P_ortho = zero_like(res)
        for o in self.ortho_vecs:
            P_ortho += o.outer(o.conj())
        P = eye_like(res) - P_ortho
        N = self.vector_shape.num_legs
        first = list(range(N))
        last = list(range(N, 2 * N))
        res = tdot(res, P, last, first)  # should we offer tdot(res, P, N) for this use case?
        res = tdot(P, res, last, first)
        if self.penalty is not None:
            res = res + self.penalty * P_ortho
        return res
        
    def adjoint(self) -> LinearOperator:
        return ProjectedLinearOperator(
            original_operator=self.original_operator.adjoint(),
            ortho_vecs=self.ortho_vecs,  # hc(|o> <o|) = |o> <o|  ->  can use same ortho_vecs
            penalty=None if self.penalty is None else np.conj(self.penalty)
        )


# TODO (JU) port FlatLinearOperator from old



def gram_schmidt(vecs: list[AbstractTensor], rcond=1.e-14) -> list[AbstractTensor]:
    """Gram-Schmidt orthonormalization of a list of tensors.

    Parameters
    ----------
    vecs : list of :class:`~tenpy.linalg.tensors.AbstractTensor`
        The list of vectors to be orthogonalized. All with the same legs.
    rcond : _type_, optional
        Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.

    Returns
    -------
    list of :class:`~tenpy.linalg.tensors.AbstractTensor`
        A list of orthonormal vectors which span the same space as `vecs`.
    """
    res = []
    for vec in vecs:
        for other in res:
            ov = other.inner(vec)
            vec = vec - ov * other
        n = vec.norm()
        if n > rcond:
            res.append(vec._mul_scalar(1. / n))
    return res
