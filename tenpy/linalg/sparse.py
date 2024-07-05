"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of a linear
operator, but only the action on a vector, i.e., a matrix-vector product `matvec`. Here we define
the structure of such a general operator, :class:`TenpyLinearOperator`, as it is used in our own
implementations of these algorithms (e.g., :mod:`~tenpy.linalg.krylov_based`). Moreover, the
:class:`FlatLinearOperator` allows to use all the scipy sparse methods by providing functionality
to convert flat numpy arrays to and from tenpy tensors.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from numbers import Number
import warnings
from typing import Literal
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, ArpackNoConvergence

from .spaces import Space, ProductSpace, Sector
from .tensors import Tensor, SymmetricTensor, ChargedTensor
from .backends.abstract_backend import TensorBackend
from .dtypes import Dtype
from ..tools.math import speigs, speigsh
from ..tools.misc import argsort


__all__ = ['LinearOperator', 'TensorLinearOperator', 'LinearOperatorWrapper', 'SumLinearOperator',
           'ShiftedLinearOperator', 'ProjectedLinearOperator', 'NumpyArrayLinearOperator',
           'HermitianNumpyArrayLinearOperator', 'gram_schmidt']


class LinearOperator(metaclass=ABCMeta):
    """Base class for a linear operator acting on tenpy tensors.

    Attributes
    ----------
    vector_shape : Shape
        The shape of tensors that this operator can act on
    dtype : Dtype
        The dtype of a full representation of the operator
    acts_on : list of str
        Labels of the state on which the operator can act. NB: Class attribute.
    """
    acts_on = None  # Derived classes should set this as a class attribute
    
    def __init__(self, vector_shape, dtype: Dtype):  # TODO Shape removed
        self.vector_shape = vector_shape
        self.dtype = dtype

    @abstractmethod
    def matvec(self, vec: Tensor) -> Tensor:
        """Apply the linear operator to a "vector".

        We consider as vectors all tensors of the shape given by :attr:`vector_shape`,
        and in particular allow multi-leg tensors as "vectors".
        The result of `matvec` must be a tensor of the same shape.
        """
        ...

    @abstractmethod
    def to_tensor(self, **kw) -> Tensor:
        """Compute a full tensor representation of the linear operator.
        
        Returns
        -------
        A tensor `t` with ``2 * N`` legs ``[a1, a2, ..., aN, aN*, ..., a2*, a1*]``, where
        ``[a1, a2, ..., aN]`` are the legs of the vectors this operator acts on.
        S.t. ``self.matvec(vec)`` is equivalent to ``tdot(t, vec, [N, ..., 2*N-1], [N-1,...,0])``.
        """
        ...

    def to_matrix(self, backend: TensorBackend = None) -> Tensor:
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
    def __init__(self, tensor: SymmetricTensor, which_leg: int | str = -1):
        if tensor.num_legs > 2:
            raise ValueError('Expected a two-leg tensor')
        raise NotImplementedError  # TODO
        # TODO can_contract_with was removed. should probably check codomain == domain after permuting?
        # if not tensor.legs[0].can_contract_with(tensor.legs[1]):
        #     raise ValueError('Expected contractible legs')
        # self.which_leg = which_leg = tensor.get_leg_idx(which_leg)
        # self.other_leg = other_leg = 1 - which_leg
        # self.tensor = tensor
        # vector_shape = Shape(legs=[tensor.legs[other_leg]], num_domain_legs=0, labels=tensor.labels[other_leg])
        # super().__init__(vector_shape=vector_shape, dtype=tensor.dtype)

    def matvec(self, vec: Tensor) -> Tensor:
        assert vec.num_legs == 1
        return self.tensor.tdot(vec, self.which_leg, 0)

    def to_tensor(self, **kw) -> Tensor:
        if self.tensor.which_leg == 1:
            return self.tensor
        return self.tensor.permute_legs([1, 0])

    def adjoint(self) -> TensorLinearOperator:
        return TensorLinearOperator(tensor=self.tensor.conj(), which_leg=self.other_leg)


class LinearOperatorWrapper(LinearOperator):
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

    def matvec(self, vec: Tensor) -> Tensor:
        return sum((op.matvec(vec) for op in self.more_operators), self.original_operator.matvec(vec))

    def to_tensor(self, **kw) -> Tensor:
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

    def matvec(self, vec: Tensor) -> Tensor:
        return self.original_operator.matvec(vec) + self.shift * vec

    # def to_tensor(self, **kw) -> Tensor:
    #     res = self.original_operator.to_tensor(**kw)
    #     return res + self.shift * eye_like(res)

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
    If this is done, at least for krylov-based eigensolvers such as lanczos, the penalty should be chosen
    such that the `ortho_vecs` are somewhere in the bulk of the spectrum.
    This is because lanczos has best convergence for the extremal eigenvalues and we want to converge
    the solutions well, not the `ortho_vecs`.
    E.g. for a typical Hamiltonian with a spectrum symmetric around zero, ``project_operator=True``
    and ``penalty=None`` shifts the `ortho_vecs` to eigenvalue zero, thus fulfilling this criterion.
    However, for operators with e.g. strictly positive spectrum, this prescription might fail.

    Parameters
    ----------
    original_operator : :class:`LinearOperator`-like
        The original operator, denoted ``H`` in the summary above.
    ortho_vecs : list of :class:`~tenpy.linalg.tensors.Tensor`
        The list of vectors spanning the projected space.
        They need not be orthonormal, as Gram-Schmidt is performed on them explicitly.
    project_operator: bool
        If False (True per default), the projection of the operator ``H -> P H P`` is skipped
        and ``H + penalty * (1 - P)`` is represented instead.
    penalty : complex, optional
        See summary above. Defaults to ``None``, which is equivalent to ``0.``.
    """
    def __init__(self, original_operator: LinearOperator, ortho_vecs: list[Tensor],
                 project_operator: bool = True, penalty: Number = None):
        if len(ortho_vecs) == 0:
            warnings.warn('empty ortho_vecs: no need for ProjectedLinearOperator', stacklevel=2)
        if not project_operator and penalty is None:
            warnings.warn('project_operator=False and penalty=None means ' \
                          'ProjectedLinearOperator does not do anything')
        super().__init__(original_operator=original_operator)
        assert all(v.shape == original_operator.vector_shape for v in ortho_vecs)
        self.ortho_vecs = gram_schmidt(ortho_vecs)
        self.project_operator = project_operator
        self.penalty = penalty

    def matvec(self, vec: Tensor) -> Tensor:
        res = vec
        # 1: res = P vec
        if self.project_operator:
            # form ``P vec`` and keep coefficients for later use in the penalty term
            coefficients = []
            for o in self.ortho_vecs:
                c = o.inner(res)
                coefficients.append(c)
                res = res - c * o
        else:
            coefficients = [o.inner(res) for o in self.ortho_vecs]
        # 2: res = H P vec
        res = self.original_operator.matvec(res)
        # 3: res = P H P vec
        if self.project_operator:
            for o in self.ortho_vecs:
                res = res - o.inner(res) * o
        # 4: res = P H P vec + (1 - P) vec
        if self.penalty is not None:
            for c, o in zip(coefficients, self.ortho_vecs):
                res = res + self.penalty * c * o
        # done
        return res

    def to_tensor(self, **kw) -> Tensor:
        raise NotImplementedError 
        # TODO adjust to changed leg convention (change convention of outer to match this?)
        #      or change conj accordingly? or implement a projector function |a><a|
        # res = self.original_operator.to_tensor(**kw)
        # P_ortho = zero_like(res)
        # for o in self.ortho_vecs:
        #     P_ortho += o.outer(o.conj())
        # if self.project_operator:
        #     P = eye_like(res) - P_ortho
        #     N = self.vector_shape.num_legs
        #     first = list(range(N))
        #     last = list(range(N, 2 * N))
        #     # TODO should we offer tdot(res, P, N) with N: int for this use case?
        #     res = tdot(res, P, last, first)
        #     res = tdot(P, res, last, first)
        # if self.penalty is not None:
        #     res = res + self.penalty * P_ortho
        # return res
        
    def adjoint(self) -> LinearOperator:
        return ProjectedLinearOperator(
            original_operator=self.original_operator.adjoint(),
            ortho_vecs=self.ortho_vecs,  # hc(|o> <o|) = |o> <o|  ->  can use same ortho_vecs
            penalty=None if self.penalty is None else np.conj(self.penalty)
        )


class NumpyArrayLinearOperator(ScipyLinearOperator):
    """Square Linear operator acting on numpy arrays based on a matvec acting on tenpy tensors.

    Note that this class represents a square linear operator.

    Parameters
    ----------
    tenpy_matvec : callable
        Function with signature ``tenpy_matvec(vec: Tensor) -> Tensor`.
        Has to return a tensor with the same legs and has to be linear.
        Unless `labels` are given, the leg order of the output must be the same as for the input.
    legs : list of :class:`~tenpy.linalg.spaces.ElementarySpace`
        The legs of a Tensor that `tenpy_matvec` can act on.
    backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `tenpy_matvec`.
    charge_sector : None | Sector | 'trivial'
        If given, only the specified charge sector is considered.
        Per default, or if the string ``'trivial'`` is given, the trivial sector of the symmetry is used.
        ``None`` stands for *all* sectors.

    Attributes
    ----------
    tenpy_matvec : callable
        Function with signature ``tenpy_matvec(vec: Tensor) -> Tensor`.
    legs : list of :class:`~tenpy.linalg.spaces.Space`
        The legs of a Tensor that `tenpy_matvec` can act on.
    backend : :class:`~tenpy.linalg.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `tenpy_matvec`.
    charge_sector : None | Sector | 'trivial'
        If given, only the specified charge sector is considered.
        If ``'trivial'`` is given, the trivial sector of the symmetry is used.
        ``None`` stands for *all* sectors.
    matvec_count : int
        The number of times `tenpy_matvec` was called.
    N : int
        The length of the numpy vectors that this operator acts on
    domain : :class:`~tenpy.linalg.spaces.ProductSpace`
        The product of the :attr:`legs`. Self is an operator on either this entire space,
        or one of its sectors, as specified by :attr:`charge_sector`.
    symmetry
        The symmetry of all involved spaces
    shape : (int, int)
        The shape of self as an operator on 1D numpy arrays
    """
    def __init__(self, tenpy_matvec, legs: list[Space], backend: TensorBackend, dtype,
                 labels: list[str] = None,
                 charge_sector: None | Sector | Literal['trivial'] = 'trivial'):
        self.tenpy_matvec = tenpy_matvec
        self.legs = legs
        self.backend = backend
        # even if there is just one leg, we form the ProductSpace anyway, so we dont have to distinguish
        #  cases and use combine_legs / split_legs in np_to_tensor and tensor_to_np
        self.domain = ProductSpace(legs, backend=backend)
        self.symmetry = legs[0].symmetry
        self.matvec_count = 0
        self.labels = labels
        
        self.shape = None  # set by charge_sector.setter
        self._charge_sector = None  # set by charge_sector.setter
        self.charge_sector = charge_sector  # uses setter with its input checks and conversions
        
        ScipyLinearOperator.__init__(self, dtype=dtype, shape=self.shape)

    @classmethod
    def from_Tensor(cls, tensor: SymmetricTensor, legs1: list[int | str], legs2: list[int | str],
                    charge_sector: None | Sector | Literal['trivial'] = 'trivial'
                    ) -> NumpyArrayLinearOperator:
        """Create a :class:`NumpyArrayLinearOperator` from a tensor that acts via contraction (`tdot`).

        The `tenpy_matvec` acting on ``vec`` is given by ``tdot(tensor, vec, legs1, legs2)``.

        Parameters
        ----------
        tensor : Tensor
            A tensors whose legs specified by `legs1` are contractible with the remaining legs.
        legs1 : list of {int | str}
            Which legs of `tensor` should be contracted on matvec
        legs2 : list of {int | str}
            Which legs of the "vector" should be contracted on `matvec`
        charge_sector : None | Sector | 'trivial'
            If given, only the specified charge sector is considered.
            If ``'trivial'`` is given, the trivial sector of the symmetry is used.
            ``None`` stands for *all* sectors.
        """
        idcs1 = tensor.get_leg_idcs(legs1)
        tensor_contr_legs = [tensor.legs[idx] for idx in idcs1]
        res_legs = [tensor.legs[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        res_labels = [tensor.labels[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        if None in res_labels:
            res_labels = None
        vec_contr_legs = []
        for l in legs2:
            if isinstance(l, int):
                vec_contr_legs.append(res_legs[l])
            else:
                vec_contr_legs.extend(tensor.get_legs(l))
        raise NotImplementedError  # TODO
        # TODO can_contract_with was removed. should probably check codomain == domain after permuting?
        if not all(l_t.can_contract_with(l_v) for l_t, l_v in zip(tensor_contr_legs, vec_contr_legs)):
            raise ValueError('Expected contractible legs')

        def tenpy_matvec(vec):
            return tensor.tdot(vec, legs1, legs2)

        return cls(tenpy_matvec, legs=vec_contr_legs, backend=tensor.backend,
                   dtype=tensor.dtype.to_numpy_dtype(), labels=res_labels,
                   charge_sector=charge_sector)

    @classmethod
    def from_matvec_and_vector(cls, tenpy_matvec, vector: Tensor, dtype=None
                               ) -> tuple[NumpyArrayLinearOperator, np.ndarray]:
        """Create a :class:`NumpyArrayLinearOperator` from a matvec and a vector that it can act on.

        This is a convenience wrapper around the constructor where arguments are inferred
        from the example `vector` that is given.
        Additionally, the `vector` is converted via :meth:`tensor_to_np`.
        The resulting `NumpyArrayLinearOperator` has a `charge_sector` set to be the sector of
        `vector`.

        Parameters
        ----------
        tenpy_matvec : callable
            Function with signature ``tenpy_matvec(vec: Tensor) -> Tensor`.
            Has to return a tensor with the same leg and has to be linear.
        vector : :class:`~tenpy.linalg.tensors.Tensor` | :class:`~tenpy.linalg.tensors.ChargedTensor`
            A tensor that `tenpy_matvec` can act on.
            If a ChargedTensor, expect a single sector on the dummy leg, which is used as the
            :attr:`charge_sector`.
            TODO revise this. purge the "dummy" language, its now "charged"
        dtype
            The *numpy* dtype of the operator. Per default, the dtype of `vector` is used.

        Returns
        -------
        op : :class:`NumpyArrayLinearOperator`
            The resulting operator
        vec_flat : 1D ndarray
            Flat numpy vector representing `vector` within its charge sector.
        """
        if isinstance(vector, ChargedTensor):
            assert vector.dummy_leg.num_sectors == 1 and vector.dummy_leg.multiplicities[0] == 1
            sector = vector.dummy_leg.sectors[0]
        else:
            sector = 'trivial'
        if dtype is None:
            dtype = vector.dtype.to_numpy_dtype()
        op = cls(tenpy_matvec, legs=vector.legs, backend=vector.backend, dtype=dtype, charge_sector=sector)
        vec_flat = op.tensor_to_flat_array(vector)
        return op, vec_flat
        
    @property
    def charge_sector(self):
        return self._charge_sector

    @charge_sector.setter
    def charge_sector(self, value):
        if isinstance(value, str) and value == 'trivial':
            sector = self.symmetry.trivial_sector
        elif value is None:
            sector = None
        else:
            assert self.symmetry.is_valid_sector(value)
            sector = value
        self._charge_sector = value
        if sector is None:
            size = self.domain.dim
        else:
            sector_idx = self.domain.sectors_where(sector)
            if sector_idx is None:
                raise ValueError('Domain of linear operator does not have this sector')
            size = (self.symmetry.sector_dim(sector) * self.domain.multiplicities[sector_idx]).item()
        self.shape = (size, size)

    def _matvec(self, vec):
        """Matvec operation acting on a numpy ndarray of the selected charge sector.

        Parameters
        ----------
        vec : np.ndarray
            A length ``N`` vector (or ``N`` x 1 matrix) where ``N`` is the total dimension
            of the selected charge sector in the parent space, or the total dimension of the
            parent space if "all" charge sectors are selected.

        Returns
        -------
        matvec_vec : 1D ndarray
            The result of the linear operation as a length ``N`` vector
        """
        vec = np.asarray(vec)
        if vec.ndim != 1:  # convert Nx1 matrix to vector
            vec = np.squeeze(vec, axis=1)
            assert vec.ndim == 1
        tens = self.flat_array_to_tensor(vec)
        tens = self.tenpy_matvec(tens)
        self.matvec_count += 1
        return self.tensor_to_flat_array(tens)

    def flat_array_to_tensor(self, vec: np.ndarray) -> Tensor:
        """Convert flat numpy data to a tensor in the selected charge sector."""
        assert vec.shape == (self.shape[1],)
        if self._charge_sector is None:
            # TODO this is a bit difficult.
            #  We need to work with tensors which do not fulfill the charge rule.
            #  I.e. they are not confined to live in the trivial sector of their parent space
            #  but can have components in all of its sectors.
            #  We could emulate this behavior by using a ChargedTensor that has as a dummy leg
            #  all sectors of the self.domain, with multiplicities all 1 and a state [1, 1, ..., 1]
            #  One way to make conversion flat_array <-> such ChargedTensor work would be to
            #  "stack" ChargedTensors?
            raise NotImplementedError
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            tens = SymmetricTensor.from_dense_block_trivial_sector(
                leg=self.domain, block=self.backend.block_from_numpy(vec), backend=self.backend
            )
            res = tens.split_legs(0)
        else:
            tens = ChargedTensor.from_dense_block_single_sector(
                leg=self.domain, block=self.backend.block_from_numpy(vec), sector=self._charge_sector,
                backend=self.backend
            )
            res = tens.split_legs(0)
        if self.labels is not None:
            res.set_labels(self.labels)
        return res

    def tensor_to_flat_array(self, tens: Tensor) -> np.ndarray:
        """Convert a tensor in the selected charge sector to a flat numpy array."""
        if self.labels is not None:
            tens = tens.permute_legs(tens.get_leg_idcs(self.labels))
        if self._charge_sector is None:
            # TODO undo the conversion from flat_array_to_tensor
            raise NotImplementedError
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            res = tens.combine_legs(list(range(tens.num_legs)), product_spaces=[self.domain])
            res = res.to_dense_block_trivial_sector()
        else:
            res = tens.combine_legs(list(range(tens.num_legs)), product_spaces=[self.domain])
            res = res.to_dense_block_single_sector()
        res = self.backend.block_to_numpy(res)
        assert res.shape == (self.shape[0],)
        return res

    def eigenvectors(self, num_ev: int = 1, max_num_ev: int = None, max_tol: float = 1.e-12,
                     which: str = 'LM', v0_np: np.ndarray = None, v0_tensor: Tensor = None,
                     cutoff: float = 1.e-10, hermitian: bool = False, **kwargs):
        """Find the (dominant) eigenvector(s) of self using :func:`scipy.sparse.linalg.eigs`.

        If a charge_sector was specified, these are the dominant eigenvectors *within that sector*.
        Otherwise, we look in all charge sectors.

        Parameters
        ----------
        num_ev : int
            Number of eigenvalues/vectors to look for.
        max_num_ev : int
            :func:`scipy.sparse.linalg.speigs` sometimes raises a NoConvergenceError for small
            `num_ev`, which might be avoided by increasing `num_ev`. As a work-around,
            we try it again in the case of an error, just with larger `num_ev` up to `max_num_ev`.
            ``None`` defaults to ``num_ev + 2``.
        max_tol : float
            After the first `NoConvergenceError` we increase the `tol` argument to that value.
        which : str
            Which eigenvalues to look for, see :func:`scipy.sparse.linalg.eigs`.
            More details also in :func:`~tenpy.tools.misc.argsort`.
        v0_np : 1D ndarray
            Initial guess as a flat numpy array, i.e. a suitable input to :meth:`_matvec`.
        v0_tensor : :class:`~tenpy.linalg.tensors.Tensor` | :class:`~tenpy.linalg.tensors.ChargedTensor`
            Initial guess as a tensor, i.e. a suitable input to :meth:`tensor_to_np`.
        cutoff : float
            Only used if ``self.charge_sector is None``; in that case it determines when entries in
            a given charge-block are considered nonzero, and what counts as degenerate.
        hermitian : bool
            If False (default), use :func:`scipy.sparse.linalg.eigs`
            If True, assume that self is hermitian and use :func:`scipy.sparse.linalg.eigsh`.
        **kwargs :
            Further keyword arguments given to :func:`scipy.sparse.linalg.eigsh` or
            :func:`scipy.sparse.linalg.eigs`, respectively.

        Returns
        -------
        eta : 1D ndarray
            The eigenvalues, sorted according to `which`.
        w : list of :class:`~tenpy.linalg.tensors.Tensor` or :class:`~tenpy.linalg.tensors.ChargedTensor`
            The corresponding eigenvectors as tensors.
        """
        if max_num_ev is None:
            max_num_ev = num_ev + 2
        if v0_tensor is not None:
            assert v0_np is None
            v0_np = self.tensor_to_flat_array(v0_tensor)
        if v0_np is not None:
            kwargs['v0'] = v0_np
            
        for k in range(num_ev, max_num_ev + 1):
            if k > num_ev:
                warnings.warn(f'Increasing `num_ev` to {k}')
            try:
                if hermitian:
                    eta, A = speigsh(self, k=k, which=which, **kwargs)
                else:
                    eta, A = speigs(self, k=k, which=which, **kwargs)
                break
            except ArpackNoConvergence:
                if k == max_num_ev:
                    raise
            kwargs['tol'] = max(max_tol, kwargs.get('tol', 0))
        cutoff = max(cutoff, 10 * kwargs.get('tol', 1.e-16))
        A = np.real_if_close(A)

        if self._charge_sector is not None:
            vecs = [self.flat_array_to_tensor(A[:, j]) for j in range(A.shape[1])]
        else:
            # TODO again this is complicated. can and should select a single charge sector here.
            raise NotImplementedError

        perm = argsort(eta, which)
        return np.array(eta)[perm], [vecs[j] for j in perm]


class HermitianNumpyArrayLinearOperator(NumpyArrayLinearOperator):
    """Hermitian variant of :class:`NumpyArrayLinearOperator`.

    Note that we don't check hermicity of :meth:`matvec`.
    """
    def _adjoint(self):
        return self

    def eigenvectors(self, *args, **kwargs):
        """Same as NumpyArrayLinearOperator.eigenvectors(..., hermitian=True)"""
        kwargs['hermitian'] = True
        return NumpyArrayLinearOperator.eigenvectors(self, *args, **kwargs)
        
        
def gram_schmidt(vecs: list[Tensor], rcond=1.e-14) -> list[Tensor]:
    """Gram-Schmidt orthonormalization of a list of tensors.

    Parameters
    ----------
    vecs : list of :class:`~tenpy.linalg.tensors.Tensor`
        The list of vectors to be orthogonalized. All with the same legs.
    rcond : _type_, optional
        Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.

    Returns
    -------
    list of :class:`~tenpy.linalg.tensors.Tensor`
        A list of orthonormal vectors which span the same space as `vecs`.
    """
    res = []
    for vec in vecs:
        for other in res:
            ov = other.inner(vec)
            vec = vec - ov * other
        n = vec.norm()
        if n > rcond:
            res.append(vec.multiply_scalar(1. / n))
    return res
