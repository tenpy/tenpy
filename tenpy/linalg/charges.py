r"""Basic definitions of a charge: classes :class:`ChargeInfo` and :class:`LegCharge`.

.. note ::
    The contents of this module are imported in :mod:`~tenpy.linalg.np_conserved`,
    so you usually don't need to import this module in your application.

A detailed introduction to np_conserved can be found in :doc:`np_conserved`.
"""

from __future__ import division

import numpy as np


class ChargeInfo(object):
    r"""Meta-data about the charge of a tensor.

    Saves info about the charge of a tensor.
    Provides :meth:`make_valid`.

    Parameters
    ----------
    mod : iterable of int, default []
        For each charge one entry `m`: the charge is conserved modulo `m`.
        If the charge can be arbitrary,
        The len gives the number of charges. Default = trivial, i.e., no charge.
        The entries tell fo
    names : list of str, default [''] * self.num
        Descriptive names for the charges.

    Attributes
    ----------
    mod : 1D array_like of ints
        The periodicity of the charges. One entry for each charge.
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.

    Notes
    -----
    We restrict to abelian charges.
    A physical example for a `charge` could be the number of particles, or just it's parity.

    mod gives the

    """

    def __init__(self, mod=[], names=None):
        """see help(self)"""
        self.mod == np.asarray(mod, dtype=np.int_)
        self._mod_1 = np.equal(self.mod, 1)  # pre-convert for faster make_valid
        if names is None:
            names = [''] * self.number()
        self.names = [str(n) for n in names]
        self.test_sanity()  # checks for invalid arguments

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.mod.ndim != 1:
            raise ValueError("mod has wrong shape")
        assert self._mod_1 == np.equal(self.mod, 1)
        if np.any(self.mod <= 0):
            raise ValueError("mod should be > 0")
        if len(self.names) != self.number():
            raise ValueError("names has incompatible length with mod")

    def number(self):
        """Return the number of charges"""
        return len(self.mod)

    def make_valid(self, charges):
        r"""Take charges modulo self.mod.

        Parameters
        ----------
        charges: array_like
            1D or 2D array of charges, last dimension == `self.number()`

        Returns
        -------
        charges:
            `charges` taken module self.mod, but with x % 1 := x
        """
        charges = np.asarray(charges, dtype=np.int_)
        return np.where(self._mod_1, [charges, np.mod(charges, self.mod)])

    def check_valid(self, charges):
        r"""Check, if `charges` has all entries as expected from self.mod.

        Returns
        -------
        Bool
            True, if all 0 <= charges <= self.mod (whenever self.mod != 1)
        """
        charges = np.asarray(charges, dtype=np.int_)
        return np.all(np.logical_or(self._mod_1,  0 <= charges < self.mod))

    def __repr__(self):
        """string representation for debugging"""
        return "ChargeInfo({0:s}, {1:s})".format(list(self.mod), self.names)

    def __eq__(self, other):
        r"""compare self.mod and self.names for equality, ignoring missin names."""
        if self is other:
            return True
        if not self.mod == other.mod:
            return False
        for l, r in zip(self.names, other.names):
            if r != l and l != '' and r != '':
                return False
        return True

    def __ne__(self, other):
        r"""Define `self != other` as `not (self == other)`"""
        return not self.__eq__(other)


class LegCharge(object):

    pass  # TODO
