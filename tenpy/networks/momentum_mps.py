r"""This module contains a base class for a Momentum Matrix Product State.

This is an extension of the uniform MPS :class:`~tenpy.networks.uniform_mps.UniformMPS`.

For each unit-cell site of the initial uniform MPS, we define an excitation tensor `B`.
The momentum state with momentum p is then constructed by the infinite sum, where one of the tensors of
the uniform MPS is replaced by the excitation tensor::

    |                        ...--AL[n-1] -- B[n] -- AR[n+1] -- ...
    | \sum_n  \exp{i p n}          |         |        |


The `B` tensors can possibly act on multiple neighboring sites to include larger excitations.
Furthermore, the `B` is decomposed into::

    |           -B- = - VL -- X -
    |            |      |

Here, `VL` is the orthogonal complement of the corresponding `AL` tensor, such that the state is
always orthogonal to the initial uniform MPS. `X` parametrizes the excited states.
"""

# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import logging
import warnings
from ..tools.misc import BetaWarning

logger = logging.getLogger(__name__)

__all__ = ['MomentumMPS']


class MomentumMPS:
    r"""A Matrix Product State, finite (MPS) or infinite (iMPS).

    Parameters
    ----------
    Xs : list of :class:`~tenpy.linalg.np_conserved.Array`
        Excitation tensors for each site of the unit cell.
    uMPS : :class:`~tenpy.networks.uniform_mps.UniformMPS`
        The uniform MPS on which the excitations are based.
    p : float
        The momentum of the state.
    n_sites : int
        Number of sites for each excitation.

    Attributes
    ----------
    dtype : type
        The data type of the ``_X``.
    _X : list of :class:`~tenpy.linalg.np_conserved.Array`
        The excitation matrices of the MPS. Labels are ``vL, p1, ..., p{n_sites-1}, vR``.
    uMPS_GS : :class:`~tenpy.networks.uniform_mps.UniformMPS`
        The uniform MPS, representing the ground state.
    p : float
        The momentum of the state.
    n_sites : int
        Number of sites for each excitation.
    """

    def __init__(self, Xs, uMPS, p, n_sites=1):
        warnings.warn('MomentumMPS is a new feature and not as well-tested as the '
                      'rest of the library', BetaWarning, stacklevel=2)
        assert len(Xs) == uMPS.L, "Need as many excitations as sites in unit cell."
        self.dtype = dtype = np.find_common_type([X.dtype for X in Xs], [])
        self._X = [X.astype(dtype, copy=True) for X in Xs]
        self.uMPS_GS = uMPS
        self.p = p
        self.n_sites = n_sites  # Number of sites of single excitation tensor.

    def copy(self):
        """Returns a copy of `self`.
        """
        # __init__ makes deep copies of B, S
        cp = self.__class__(self._X, self.uMPS_GS, self.p, self.n_sites)
        return cp

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        Specifically, it saves
        :attr:`_X` as ``"tensors"``,
        :attr:`uMPS_GS` as ``"GS_uMPS"``, and
        :attr:`p` as ``"momentum"``.
        Moreover, it saves :attr:`n_sites` as HDF5 attributes.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        hdf5_saver.save(self._X, subpath + "tensors")
        hdf5_saver.save(self.uMPS_GS, subpath + "GS_uMPS")
        hdf5_saver.save(self.p, subpath + "momentum")
        h5gr.attrs["n_sites"] = self.n_sites

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.hdf5_io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)

        obj._X = hdf5_loader.load(subpath + "tensors")
        obj.uMPS_GS = hdf5_loader.load(subpath + "GS_uMPS")
        obj.p = hdf5_loader.load(subpath + "momentum")
        obj.n_sites = hdf5_loader.get_attr(h5gr, "n_sites")
        obj.uMPS_GS.test_sanity()
        return obj

    def get_X(self, i, copy=False):
        """Return (view of) `X` at site `i`.

        Parameters
        ----------
        i : int
            Index choosing the site.
        copy : bool
            Whether to return a copy even if `form` matches the current form.

        Returns
        -------
        X : :class:`~tenpy.linalg.np_conserved.Array`
            The excitation 'matrix' `X` at site `i` with leg labels
            ``'vL', 'p1', ..., 'p{n_sites-1}', 'vR'``.
            May be a view of the matrix (if ``copy=False``) or a copy (if ``copy=True``).

        """
        i = self._to_valid_index(i)
        X = self._X[i]
        if copy:
            X = X.copy()
        return X

    def set_X(self, i, X):
        """Set `X` at site `i`.

        Parameters
        ----------
        i : int
            Index choosing the site.
        X : :class:`~tenpy.linalg.np_conserved.Array`
            The 'matrix' at site `i`. No copy is made!
            Should have leg labels ``'vL', 'p1', ..., 'p{n_sites-1}', 'vR'`` (not necessarily in that order).

        """
        i = self._to_valid_index(i)
        self.dtype = np.promote_types(self.dtype, X.dtype)
        self._X[i] = X

    def _to_valid_index(self, i):
        """Make sure `i` is a valid index."""
        return i % self.L
