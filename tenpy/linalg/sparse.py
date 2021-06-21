"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of a linear
operator, but only the action on a vector, i.e., a matrix-vector product `matvec`. Here we define
the strucuture of such a general operator, :class:`NpcLinearOperator`, as it is used in our own
implementations of these algorithms (e.g., :mod:`~tenpy.linalg.lanczos`). Moreover, the
:class:`FlatLinearOperator` allows to use all the scipy sparse methods by providing functionality
to convert flat numpy arrays to and from np_conserved arrays.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from . import np_conserved as npc
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from ..tools.math import speigs, speigsh
from ..tools.misc import argsort, group_by_degeneracy
import warnings

__all__ = [
    'NpcLinearOperator',
    'NpcLinearOperatorWrapper',
    'SumNpcLinearOperator',
    'ShiftNpcLinearOperator',
    'OrthogonalNpcLinearOperator',
    'FlatLinearOperator',
    'FlatHermitianOperator',
]


class NpcLinearOperator:
    """Prototype for a Linear Operator acting on :class:`~tenpy.linalg.np_conserved.Array`.

    Note that an :class:`~tenpy.linalg.np_conserved.Array` implements a matvec function. Thus you
    can use any (square) npc Array as an NpcLinearOperator.

    Attributes
    ----------
    dtype : np.type
        The data type of its action.
    acts_on : list of str
        Labels of the state on which the operator can act. NB: Class attribute.
    """
    acts_on = None  # Derived classes should set this as a class attribute

    def matvec(self, vec):
        """Calculate the action of the operator on a vector `vec`.

        Note that we don't require `vec` to be one-dimensional. However, for square operators we
        require that the result of `matvec` has the same legs (in the same order) as `vec` such
        that they can be added. Note that this excludes a non-trivial `qtotal` for square
        operators.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def to_matrix(self):
        """Contract `self` to a matrix.

        If `self` represents an operator with very small shape,
        e.g. because the MPS bond dimension is very small,
        an algorithm might choose to contract `self` to a single tensor.

        Returns
        -------
        matrix : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of the represented operator.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def adjoint(self):
        """Return the hermitian conjugate of `self`

        If `self` is hermitian, subclasses *can* choose to implement this to define
        the adjoint operator of `self`."""
        raise NotImplementedError("No adjoint defined")


class NpcLinearOperatorWrapper:
    """Base class for wrapping around another :class:`NpcLinearOperator`.

    Attributes not explicitly set with `self.attribute = value` (or by defining methods)
    default to the attributes of the wrapped `orig_operator`.

    .. warning ::
        If there are multiple levels of wrapping operators, the order might be critical to get
        correct results; e.g. :class:`OrthogonalNpcLinearOperator` needs to be the outer-most
        wrapper to produce correct results and/or be efficient.

    Parameters
    ----------
    orig_operator : NpcLinearOperator
        The original operator implementing the `matvec`.

    Attributes
    ----------
    orig_operator : NpcLinearOperator
        The original operator implementing the `matvec`.
    """
    def __init__(self, orig_operator):
        self.orig_operator = orig_operator

    def __getattr__(self, name):
        # default to un-wrapped attributes
        return getattr(self.orig_operator, name)

    def unwrapped(self):
        """Return to the original NpcLinearOperator.

        If multiple levels of wrapping were used, this returns the most unwrapped one.
        """
        parent = self.orig_operator
        for _ in range(10000):
            if hasattr(parent, "unwrapped"):
                parent = parent.unwrapped()
            else:
                break
        else:
            raise ValueError("maximum recursion depth for unwrapping reached")
        return parent

    def to_matrix(self):
        """Contract `self` to a matrix."""
        raise NotImplementedError("This function should be implemented in derived classes")

    def adjoint(self):
        """Return the hermitian conjugate of `self`.

        If `self` is hermitian, subclasses *can* choose to implement this to define
        the adjoint operator of `self`."""
        raise NotImplementedError("This function should be implemented in derived classes")


class SumNpcLinearOperator(NpcLinearOperatorWrapper):
    """Sum of two linear operators."""
    def __init__(self, orig_operator, other_operator):
        super().__init__(orig_operator)
        self.other_operator = other_operator

    def matvec(self, vec):
        return self.orig_operator.matvec(vec) + self.other_operator.matvec(vec)

    def to_matrix(self):
        return self.orig_operator.to_matrix() + self.other_operator.to_matrix()

    def adjoint(self):
        return SumNpcLinearOperator(self.orig_operator.adjoint(), self.other_operator.adjoint())


class ShiftNpcLinearOperator(NpcLinearOperatorWrapper):
    """Representes ``original_operator + shift * identity``.

    This can be useful e.g. for better Lanczos convergence.
    """
    def __init__(self, orig_operator, shift):
        if shift == 0.:
            warnings.warn("shift=0: no need for ShiftNpcLinearOperator", stacklevel=2)
        super().__init__(orig_operator)
        self.shift = shift

    def matvec(self, vec):
        return self.orig_operator.matvec(vec) + self.shift * vec

    def to_matrix(self):
        mat = self.orig_operator.to_matrix()
        return mat + self.shift * npc.eye_like(mat)

    def adjoint(self):
        return ShiftNpcLinearOperator(self.orig_operator.adjoint(), np.conj(self.shift))


class OrthogonalNpcLinearOperator(NpcLinearOperatorWrapper):
    """Replace ``H -> P H P`` with the projector ``P = 1 - sum_o |o> <o|``.

    Here, ``|o>`` are the vectors from :attr:`ortho_vecs`.

    Parameters
    ----------
    orig_operator : :class:`EffectiveH`
        The original `EffectiveH` instance to wrap around.
    ortho_vecs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The vectors to orthogonalize against.
    """
    def __init__(self, orig_operator, ortho_vecs):
        if len(ortho_vecs) == 0:
            warnings.warn("Empty `ortho_vecs`: no need to patch `OrthogonalNpcLinearOperator`",
                          stacklevel=2)
        super().__init__(orig_operator)
        from .lanczos import gram_schmidt
        ortho_vecs, _ = gram_schmidt(ortho_vecs)
        self.ortho_vecs = ortho_vecs

    def matvec(self, vec):
        # equivalent to using H' = P H P where P is the projector (1-sum_o |o><o|)
        vec = vec.copy()
        for o in self.ortho_vecs:  # Project out
            vec.iadd_prefactor_other(-npc.inner(o, vec, 'range', do_conj=True), o)
        vec = self.orig_operator.matvec(vec)
        for o in self.ortho_vecs[::-1]:  # reverse: more obviously Hermitian.
            vec.iadd_prefactor_other(-npc.inner(o, vec, 'range', do_conj=True), o)
        return vec

    def to_matrix(self):
        matrix = self.orig_operator.to_matrix()
        labels = matrix.get_leg_labels()
        proj = npc.eye_like(matrix, 0)
        for o in self.ortho_vecs:
            o = o.combine_legs(o.get_leg_labels())
            proj -= npc.outer(o, o.conj())
        matrix = npc.tensordot(matrix, proj, len(labels) // 2)
        matrix = npc.tensordot(proj, matrix, len(labels) // 2)
        matrix.iset_leg_labels(labels)
        return matrix

    def adjoint(self):
        return OrthogonalNpcLinearOperator(self.orig_operator.adjoint(), self.ortho_vecs)


class FlatLinearOperator(ScipyLinearOperator):
    """Square Linear operator acting on numpy arrays based on a `matvec` acting on npc Arrays.

    Note that this class represents a square linear operator.  In terms of charges,
    this means it has legs ``[self.leg.conj(), self.leg]`` and trivial (zero) ``qtotal``.

    Parameters
    ----------
    npc_matvec : function
        Function to calculate the action of the linear operator on an npc vector
        (with the specified `leg`). Has to return an npc vector with the same leg.
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Leg of the vector on which `npc_matvec` can act on.
    dtype : np.dtype
        The data type of the arrays.
    charge_sector : None | charges | ``0``
        Selects the charge sector of the vector onto which the Linear operator acts.
        ``None`` stands for *all* sectors, ``0`` stands for the zero-charge sector.
        Defaults to ``0``, i.e., *assumes* the dominant eigenvector is in charge sector 0.
    vec_label : None | str
        Label to be set to the npc vector before acting on it with `npc_matvec`. Ignored if `None`.

    Attributes
    ----------
    possible_charge_sectors : ndarray[QTYPE, ndim=2]
        Each row corresponds to one possible choice for `charge_sector`.
    shape : (int, int)
        The dimensions for the selected charge sector.
    dtype : np.dtype
        The data type of the arrays.
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Leg of the vector on which `npc_matvec` can act on.
    vec_label : None | str
        Label to be set to the npc vector before acting on it with `npc_matvec`. Ignored if `None`.
    npc_matvec : function
        Function to calculate the action of the linear operator on an npc vector (with one `leg`).
    matvec_count : int
        The number of times `npc_matvec` was called.
    _mask : ndarray[ndim=1, bool]
        The indices of `leg` corresponding to the `charge_sector` to be diagonalized.
    _npc_matvec_multileg : function | None
        Only set if initalized with :meth:`from_guess_with_pipe`.
        The `npc_matvec` function to be wrapped around. Takes the npc Array in multidimensional
        form and returns it that way.
    _labels_split : list of str
        Only set if initalized with :meth:`from_guess_with_pipe`.
        Labels of the guess before combining them into a pipe (stored as `leg`).
    """
    def __init__(self, npc_matvec, leg, dtype, charge_sector=0, vec_label=None):
        self.npc_matvec = npc_matvec
        self.leg = leg
        self.possible_charge_sectors = leg.charge_sectors()
        self.shape = (leg.ind_len, leg.ind_len)
        self.dtype = dtype
        self.vec_label = vec_label
        self.matvec_count = 0
        self._charge_sector = None
        self._mask = None
        self.charge_sector = charge_sector  # uses the setter
        self._npc_matvec_multileg = None
        self._labels_split = None
        ScipyLinearOperator.__init__(self, self.dtype, self.shape)

    @classmethod
    def from_NpcArray(cls, mat, charge_sector=0):
        """Create a `FlatLinearOperator` from a square :class:`~tenpy.linalg.np_conserved.Array`.

        Parameters
        ----------
        mat : :class:`~tenpy.linalg.np_conserved.Array`
            A square matrix, with contractable legs.
        charge_sector : None | charges | ``0``
            Selects the charge sector of the vector onto which the Linear operator acts.
            ``None`` stands for *all* sectors, ``0`` stands for the zero-charge sector.
            Defaults to ``0``, i.e., *assumes* the dominant eigenvector is in charge sector 0.
        """
        if mat.rank != 2:
            raise ValueError("Works only for square matrices")
        mat.legs[1].test_contractible(mat.legs[0])
        return cls(mat.matvec, mat.legs[0], mat.dtype, charge_sector)

    @classmethod
    def from_guess_with_pipe(cls, npc_matvec, v0_guess, labels_split=None, dtype=None):
        """Create a `FlatLinearOperator`` from a `matvec` function acting on multiple legs.

        This function creates a wrapper `matvec` function to allow acting on a "vector" with
        multiple legs. The wrapper combines the legs into a :class:`~tenpy.linalg.charges.LegPipe`
        before calling the actual `matvec` function, and splits them again in the end.

        Parameters
        ----------
        npc_matvec : function
            Function to calculate the action of the linear operator on an npc vector
            with the given split labels `labels_split`.
            Has to return an npc vector with the same legs.
        v0_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess/starting vector which can be applied to `npc_matvec`.
        labels_split : None | list of str
            Labels of v0_guess in the order in which they are to be combined into a
            :class:`~tenpy.linalg.charges.LegPipe`. ``None`` defaults to
            ``v0_guess.get_leg_labels()``.
        dtype : np.dtype | None
            The data type of the arrays. ``None`` defaults to dtype of `v0_guess` (!).

        Returns
        -------
        lin_op : cls
            Instance of the class to be used as linear operator
        guess_flat : np.ndarray
            Numpy vector representing the guess `v0_guess`.
        """
        if dtype is None:
            dtype = v0_guess.dtype
        if labels_split is None:
            labels_split = v0_guess.get_leg_labels()
        v0_combined = v0_guess.combine_legs(labels_split, qconj=+1)
        if v0_combined.rank != 1:
            raise ValueError("`labels_split` must contain all the legs of `v0_guess`")
        pipe = v0_combined.legs[0]
        pipe_label = v0_combined.get_leg_labels()[0]
        res = cls(npc_matvec, pipe, dtype, v0_combined.qtotal, pipe_label)
        res._labels_split = labels_split
        res._npc_matvec_multileg = npc_matvec
        res.npc_matvec = res._npc_matvec_wrapper  # activate the wrapper
        guess_flat = res.npc_to_flat(v0_combined)
        return res, guess_flat

    @property
    def charge_sector(self):
        """Charge sector of the vector which is acted on."""
        return self._charge_sector

    @charge_sector.setter
    def charge_sector(self, value):
        if type(value) == int and value == 0:
            value = self.leg.chinfo.make_valid()  # zero charges
        elif value is not None:
            value = self.leg.chinfo.make_valid(value)
        self._charge_sector = value
        if value is not None:
            self._mask = np.all(self.leg.to_qflat() == value[np.newaxis, :], axis=1)
            self.shape = tuple([np.sum(self._mask)] * 2)
        else:
            chi2 = self.leg.ind_len
            self.shape = (chi2, chi2)
            self._mask = np.ones([chi2], dtype=np.bool_)

    def _matvec(self, vec):
        """Matvec operation acting on a numpy ndarray of the selected charge sector.

        Parameters
        ----------
        vec : np.ndarray
            Vector (or N x 1 matrix) to be acted on by self.

        Returns
        -------
        matvec_vec : np.ndarray
            The result of acting the represented LinearOperator (`self`) on `vec`,
            i.e., the result of applying `npc_matvec` to an npc Array generated from `vec`.
        """
        vec = np.asarray(vec)
        if vec.ndim != 1:
            vec = np.squeeze(vec, axis=1)  # need a vector, not a Nx1 matrix
            assert vec.ndim == 1
        npc_vec = self.flat_to_npc(vec)  # convert to npc Array
        npc_vec = self.npc_matvec(npc_vec)  # the expensive part, wrapped matvec function
        self.matvec_count += 1
        vec = self.npc_to_flat(npc_vec)  # convert back to numpy Array
        return vec

    def flat_to_npc(self, vec):
        """Convert flat numpy vector of selected charge sector into npc Array.

        If :attr:`charge_sector` is not None, convert to a 1D npc vector with leg `self.leg`.
        Otherwise convert `vec`, which can be non-zero in *all* charge sectors, to a npc matrix
        with an additional ``'charge'`` leg to allow representing the full vector at once.

        Parameters
        ----------
        vec : 1D ndarray
            Numpy vector to be converted. Should have the entries according to self.charge_sector.

        Returns
        -------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Same as `vec`, but converted into a npc array.
        """
        if self._charge_sector is not None:
            res = npc.zeros([self.leg], vec.dtype, self._charge_sector, labels=[self.vec_label])
            res[self._mask] = vec
            return res
        else:
            leg = self.leg
            ch_leg = npc.LegCharge.from_qflat(leg.chinfo,
                                              self.possible_charge_sectors,
                                              qconj=-leg.qconj)
            res = npc.zeros([self.leg, ch_leg], vec.dtype, labels=[self.vec_label, 'charge'])
            res._qdata = np.repeat(np.arange(leg.block_number, dtype=np.intp),
                                   2).reshape(leg.block_number, 2)
            for qi in range(leg.block_number):
                res._data.append(vec[leg.get_slice(qi)].reshape((-1, 1)))
            res.test_sanity()
            return res

    def npc_to_flat(self, npc_vec):
        """Convert npc Array into a 1D ndarray, inverse of :meth:`flat_to_npc`.

        Parameters
        ----------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Npc Array to be converted.
            If `self.charge_sector` is not None, this should be a 1D array with that `qtotal`.
            If `self.charge_sector` is not None, it should have an additional ``"charge"`` leg,
            (as returned by :meth:`flat_to_npc`).

        Returns
        -------
        vec : 1D ndarray
            Same entries as `npc_vec`, but converted into a flat Numpy array.
        """
        if self._charge_sector is not None:
            assert npc_vec.rank == 1
            if np.any(npc_vec.qtotal != self._charge_sector):
                raise ValueError("npc_vec.qtotal and charge sector don't match!")
            if isinstance(npc_vec.legs[0], npc.LegPipe):
                npc_vec = npc_vec.copy(deep=False)
                npc_vec.legs[0] = npc_vec.legs[0].to_LegCharge()
            return npc_vec[self._mask].to_ndarray()
        else:
            npc_vec.itranspose([self.vec_label, 'charge'])
            res = np.zeros([self.leg.ind_len], npc_vec.dtype)
            leg = self.leg
            for qinds, data in zip(npc_vec._qdata, npc_vec._data):
                qi = qinds[0]
                assert qi == qinds[1]
                res[leg.get_slice(qi)] = data.reshape((-1, ))
            return res

    def flat_to_npc_all_sectors(self, vec):
        """Convert flat vector of *all* charge sectors into npc Array with extra "charge" leg.

        .. deprecated :: 0.7.3
            This is merged into :meth:`flat_to_npc` with ``self.charge_sector = None``.

        Parameters
        ----------
        vec : 1D ndarray
            Numpy vector to be converted.

        Returns
        -------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Same as `vec`, but converted into a npc array.
        """
        assert self._charge_sector is None
        warnings.warn(
            "Deprecated access of FlatLinearOperator.flat_to_npc_all_sectors.\n"
            "directly use flat_to_npc instead!",
            category=FutureWarning,
            stacklevel=2)
        return self.flat_to_npc(vec)

    def flat_to_npc_None_sector(self, vec, cutoff=1.e-10):
        """Convert flat vector of undetermined charge sectors into npc Array.

        The charge sector to be used is chosen as the block with the maximal norm,
        not by `self.charge_sector` (which might be `None`).

        Parameters
        ----------
        vec : 1D ndarray
            Numpy vector to be converted.

        Returns
        -------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Same as `vec`, but converted into a npc array.
        """
        assert self._charge_sector is None
        return npc.Array.from_ndarray(vec, [self.leg], cutoff=cutoff, labels=[self.vec_label])

    def npc_to_flat_all_sectors(self, npc_vec):
        """Convert npc Array with qtotal = self.charge_sector into ndarray.

        .. deprecated :: 0.7.3
            This is merged into :meth:`npc_to_flat` with ``self.charge_sector = None``.

        Parameters
        ----------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Npc Array to be converted. Should only have entries in `self.charge_sector`.

        Returns
        -------
        vec : 1D ndarray
            Same as `npc_vec`, but converted into a flat Numpy array.
        """
        assert self._charge_sector is None
        warnings.warn(
            "Deprecated access of FlatLinearOperator.npc_to_flat_all_sectors.\n"
            "directly use npc_to_flat instead!",
            category=FutureWarning,
            stacklevel=2)
        return self.npc_to_flat(npc_vec)

    def _npc_matvec_wrapper(self, vec):
        """Wrapper around ``self._npc_matvec_multileg`` acting on a LegPipe.

        Used when the class was generated with :meth:`from_guess_with_pipe`.

        ``self._npc_matvec_multileg`` is a function which can act on a multi-dimensional npc Array
        and returns it with the same legs (with labels ``self._labels_split``).
        This function can act on a vector where these legs are combined into a LegPipe
        (the pipe is stored as ``self.leg``).

        Parameters
        ----------
        vec : :class:`~tenpy.linalg.np_conserved.Array`
            Npc Array to act on. Can have multiple legs (as necessary for
            ``self._npc_matvec_multileg``), or have the legs combined in the LegPipe stored as
            ``self.leg``.

        Returns
        -------
        matvec_vec : np.ndarray
            The result of acting the represented LinearOperator (`self`) on `vec`,
            i.e., the result of applying `npc_matvec` to an npc Array generated from `vec`.
            Has the same leg structure as `vec`.
        """
        legs_initially_combined = (vec.rank == 1)
        if legs_initially_combined:
            vec = vec.split_legs(0)
        vec.itranspose(self._labels_split)  # ensure correct leg/label structure
        vec = self._npc_matvec_multileg(vec)  # apply matvec acting on multi-leg Array
        vec.itranspose(self._labels_split)
        if legs_initially_combined:
            vec = vec.combine_legs(self._labels_split, pipes=self.leg)
        return vec

    def eigenvectors(self,
                     num_ev=1,
                     max_num_ev=None,
                     max_tol=1.e-12,
                     which='LM',
                     v0=None,
                     v0_npc=None,
                     cutoff=1.e-12,
                     hermitian=False,
                     **kwargs):
        """Find (dominant) eigenvector(s) of self using :func:`scipy.sparse.linalg.eigs`.

        If no charge_sector was selected, we look in *all* charge sectors.

        Parameters
        ----------
        num_ev : int
            Number of eigenvalues/vectors to look for.
        max_num_ev : int
            :func:`scipy.sparse.linalg.speigs` somtimes raises a NoConvergenceError for small
            `num_ev`, which might be avoided by increasing `num_ev`. As a work-around,
            we try it again in the case of an error, just with larger `num_ev` up to `max_num_ev`.
            ``None`` defaults to ``num_ev + 2``.
        max_tol : float
            After the first `NoConvergenceError` we increase the `tol` argument to that value.
        which : str
            Which eigenvalues to look for, see :func:`scipy.sparse.linalg.eigs`.
            More details also in :func:`~tenpy.tools.misc.argsort`.
        v0 : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess as a "flat" numpy array.
        v0_npc : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess, to be converted by :meth:`npc_to_flat`.
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
        w : list of :class:`~tenpy.linalg.np_conserved.Array`
            The eigenvectors corresponding to `eta`, as npc.Array with LegPipe.
        """
        if max_num_ev is None:
            max_num_ev = num_ev + 2
        if v0_npc is not None:
            assert v0 is None
            v0 = self.npc_to_flat(v0_npc)
        if v0 is not None:
            kwargs['v0'] = v0
        # for given charge sector
        for k in range(num_ev, max_num_ev + 1):
            if k > num_ev:
                warnings.warn("TransferMatrix: increased `num_ev` to " + str(k + 1))
            try:
                if hermitian:
                    eta, A = speigsh(self, k=k, which=which, **kwargs)
                else:
                    eta, A = speigs(self, k=k, which=which, **kwargs)
                break
            except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
                if k == max_num_ev:
                    raise
            kwargs['tol'] = max(max_tol, kwargs.get('tol', 0))
        from_ndarray_args = dict(legcharges=[self.leg],
                                 cutoff=cutoff,
                                 labels=[self.vec_label],
                                 raise_wrong_sector=True)
        A = np.real_if_close(A)
        if self._charge_sector is not None:
            vecs = [self.flat_to_npc(A[:, j]) for j in range(A.shape[1])]
        else:
            k = A.shape[1]
            vecs = [None] * k
            multi_sectors = []
            for j in range(k):
                vec = A[:, j]
                try:
                    vecs[j] = npc.Array.from_ndarray(vec, **from_ndarray_args)
                except ValueError as e:
                    if not e.args[0].startswith('wrong sector'):
                        raise
                    multi_sectors.append(j)
            from_ndarray_args['raise_wrong_sector'] = False
            for degenerate in group_by_degeneracy(eta, cutoff=cutoff, subset=multi_sectors):
                # really, we would need to diagonalize the charges within the subspace of
                # degenerate eigenvectors of the transfermatrix.
                # However, we (might) only know a subset of the degenerate eigenvectors,
                # and diagonalizing the charges in that subspace might not yield real
                # charge eigenvectors (which is obvious if we have only one of them).
                # Instead we just project onto different blocks:
                # within a block it will for sure be an eigenvector of both the charge
                # and the transfer matrix!
                # This can fail, however, if there are less non-zero blocks than eigenvectors,
                # in which case we don't return an array with definite charge but with an
                # additional `charge` leg.
                if len(degenerate) == 1:
                    # degenerate eigenvalues but we didn't find enough eigenvectors to see that
                    j = degenerate[0]
                    qi = np.argmax([
                        np.linalg.norm(vec[self.leg.get_slice(qi)])
                        for qi in range(self.leg.block_number)
                    ])
                    qtotal = self.leg.get_charge(qi)
                    vecs[j] = npc.Array.from_ndarray(A[:, j], **from_ndarray_args, qtotal=qtotal)
                else:
                    used_blocks = set()
                    for j in degenerate:
                        vec = A[:, j]
                        block_norms = [
                            np.linalg.norm(vec[self.leg.get_slice(qi)])
                            for qi in range(self.leg.block_number)
                        ]
                        for qi in np.argsort(block_norms):
                            if qi in used_blocks or block_norms[qi] < cutoff:
                                continue
                            used_blocks.add(qi)
                            qtotal = self.leg.get_charge(qi)
                            vecs[j] = npc.Array.from_ndarray(vec,
                                                             qtotal=qtotal,
                                                             **from_ndarray_args)
                            vecs[j] /= npc.norm(vecs[j])
                            break
                        else:
                            # didn't break, so didn't set vecs[j]
                            # -> can't guarantee orthogonality to previous eigenvectors
                            msg = ("FlatLinearOperator.eigenvectors: can't project to definite "
                                   "charge block uniquely; would need to diagonalize again "
                                   "with given `charge_sector`.\n"
                                   "Will return eigenvector with extra `charge` leg instead, "
                                   "which might not be orthogonal to other eigenvectors.")
                            warnings.warn(msg, stacklevel=2)
                            vecs[j] = self.flat_to_npc(vec)
                            for qi in range(self.leg.used_block_number):
                                if block_norms[qi] > cutoff:
                                    used_blocks.add(qi)
        perm = argsort(eta, which)
        return np.array(eta)[perm], [vecs[j] for j in perm]


class FlatHermitianOperator(FlatLinearOperator):
    """Hermitian variant of :class:`FlatLinearOperator`.

    Note that we don't check :meth:`matvec` to return a hermitian result, we only define an adjoint
    to be `self`.
    """
    def _adjoint(self):
        return self

    def eigenvectors(self, *args, **kwargs):
        """Same as FlatLinearOperator(..., hermitian=True)."""
        kwargs['hermitian'] = True
        return FlatLinearOperator.eigenvectors(self, *args, **kwargs)
