"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of
a linear operator, but only the action on a vector, i.e., a matrix-vector product `matvec`.
Here we define the strucuture of such a general operator, :class:`NpcLinearOperator`,
as it is used in our own implementations of these algorithms (e.g., :mod:`~tenpy.linalg.lanczos`).
Moreover, the :class:`FlatLinearOperator` allows to use all the scipy sparse methods
by providing functionality to convert flat numpy arrays to and from np_conserved arrays.
"""
# Copyright 2018 TeNPy Developers

import numpy as np
from . import np_conserved as npc
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


class NpcLinearOperator:
    """Prototype for a Linear Operator acting on :class:`~tenpy.linalg.np_conserved.Array`.

    Note that an :class:`~tenpy.linalg.np_conserved.Array` implements a matvec function.
    Thus you can use any (square) npc Array as an NpcLinearOperator.
    """

    def matvec(self, vec):
        """Calculate the action of the operator on a vector `vec`.

        Note that we don't require `vec` to be one-dimensional.
        However, for square operators we require that the result of `matvec`
        has the same legs (in the same order) as `vec` such that they can be added.
        Note that this excludes a non-trivial `qtotal` for square operators.
        """
        raise NotImplementedError("Derived classes should implement this")


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
    charge_sector
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
        ScipyLinearOperator.__init__(self, self.dtype, self.shape)

    @classmethod
    def from_NpcArray(cls, mat, charge_sector=0):
        """Create an FlatLinearOperator from an square :class:`~tenpy.linalg.np_conserved.Array`.

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
            self._mask = np.ones([chi2], dtype=np.bool)

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
        if self._charge_sector is not None:
            npc_vec = self.flat_to_npc(vec)  # convert into npc Array
            npc_vec = self.npc_matvec(npc_vec)  # apply the transfer matrix
            self.matvec_count += 1
            return self.npc_to_flat(npc_vec)  # convert back into numpy ndarray.
        else:
            result = np.zeros(self.shape[0], self.dtype)
            # iterator over all charge sectors
            for sector in self.possible_charge_sectors:
                self.charge_sector = sector
                assert self._charges_sector is not None
                npc_vec = self.flat_to_npc(vec[self._mask])  # convert into npc Array
                npc_vec = self.npc_matvec(npc_vec)  # apply the transfer matrix
                self.matvec_count += 1
                result[self._mask] = self.npc_to_flat(npc_vec)  # convert back into numpy ndarray.
            return result

    def flat_to_npc(self, vec):
        """Convert flat vector of selected charge sector into npc Array.

        Parameters
        ----------
        vec : 1D ndarray
            Numpy vector to be converted. Should have the entries according to self.charge_sector.

        Returns
        -------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Same as `vec`, but converted into a flat array.
        """
        if self._charge_sector is None:
            raise ValueError("By definition, this can't work for all charges at once!")
        res = npc.zeros([self.leg], vec.dtype, self._charge_sector)
        res[self._mask] = vec
        if self.vec_label is not None:
            res.iset_leg_labels([self.vec_label])
        return res

    def npc_to_flat(self, npc_vec):
        """Convert npc Array with qtotal = self.charge_sector into ndarray.

        Parameters
        ----------
        npc_vec : :class:`~tenpy.linalg.np_conserved.Array`
            Npc Array to be converted. Should only have the entries only in self.charge_sector.

        Returns
        -------
        vec : 1D ndarray
            Same as `npc_vec`, but converted into a flat array.
        """
        if self._charge_sector is not None and np.any(npc_vec.qtotal != self._charge_sector):
            raise ValueError("npc_vec.qtotal and charge sector don't match!")
        if isinstance(npc_vec.legs[0], npc.LegPipe):
            npc_vec.legs[0] = npc_vec.legs[0].to_LegCharge()
        return npc_vec[self._mask].to_ndarray()


class FlatHermitianOperator(FlatLinearOperator):
    """Hermitian variant of :class:`FlatLinearOperator`.

    Note that we don't check :meth:`matvec` to return a hermitian result,
    we only define an adjoint to be `self`."""

    def _adjoint(self):
        return self
