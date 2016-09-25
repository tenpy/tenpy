r"""Basic definitions of a charge: classes :class:`ChargeInfo` and :class:`LegCharge`.

.. note ::
    The contents of this module are imported in :mod:`~tenpy.linalg.np_conserved`,
    so you usually don't need to import this module in your application.

A detailed introduction to np_conserved can be found in :doc:`np_conserved`.
"""

from __future__ import division

import numpy as np
import copy

QDTYPE = np.int_  # the type of a single charge


class ChargeInfo(object):
    r"""Meta-data about the charge of a tensor.

    Saves info about the nature of the charge of a tensor.
    Provides :meth:`make_valid` for taking modulo `m`.

    Parameters
    ----------
    qmod : iterable of `QDTYPE`
        The len gives the number of charges, `qnumber`.
        For each charge one entry `m`: the charge is conserved modulo `m`.
        Defaults to trivial, i.e., no charge.
    names : list of str
        Descriptive names for the charges.  Defaults to ``['']*qnumber``.

    Attributes
    ----------
    qnumber
    qmod : 1D array_like of ints
        The periodicity of the charges. One entry for each charge.
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.

    Notes
    -----
    Instances of this class can (and should be) shared between different `LegCharge`s and even
    `npc.Array`s
    """

    def __init__(self, mod=[], names=None):
        """see help(self)"""
        self.mod = np.array(mod, dtype=QDTYPE)
        self._mod_1 = np.equal(self.mod, 1)  # pre-convert for faster make_valid
        if names is None:
            names = [''] * self.qnumber
        self.names = [str(n) for n in names]
        self.test_sanity()  # checks for invalid arguments

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.mod.ndim != 1:
            raise ValueError("mod has wrong shape")
        assert np.all(self._mod_1 == np.equal(self.mod, 1))
        if np.any(self.mod <= 0):
            raise ValueError("mod should be > 0")
        if len(self.names) != self.qnumber:
            raise ValueError("names has incompatible length with mod")

    @property
    def qnumber(self):
        """the number of charges, also refered to as qnumber"""
        return len(self.mod)

    def make_valid(self, charges):
        r"""Take charges modulo self.mod.

        Parameters
        ----------
        charges: array_like
            1D or 2D array of charges, last dimension `self.qnumber`

        Returns
        -------
        charges:
            `charges` taken module self.mod, but with x % 1 := x
        """
        charges = np.asarray(charges, dtype=QDTYPE)
        return np.where(self._mod_1, charges, np.mod(charges, self.mod))

    def check_valid(self, charges):
        r"""Check, if `charges` has all entries as expected from self.mod.

        Returns
        -------
        Bool
            True, if all 0 <= charges <= self.mod (whenever self.mod != 1)
        """
        charges = np.asarray(charges, dtype=QDTYPE)
        return np.all(np.logical_or(self._mod_1, np.logical_and(0 <= charges, charges < self.mod)))

    def __repr__(self):
        """full string representation"""
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
    r"""Save the charge data associated to a leg of a tensor.

    In the end, this is a wrapper around a 2D numpy array qind.

    Parameters
    ----------
    chargeinfo : :class:`ChargeInfo`
        the nature of the charge
    qind : 2D array_like
        first dimension = qindex,
        second dimension: first two entries define the block as `slice(beg, end)`,
        remaining entries = charge.
    qconj : {+1, -1}
        A flag telling whether the charge points inwards (+1, default) or outwards (-1).

    Attributes
    ----------
    ind_len
    block_number
    chinfo : :class:`ChargeInfo` instance
        the nature of the charge. Can be shared between LegCharges
    qind : np.array, shape(blocknumber, 2+qnumber)
        first dimension = qindex,
        second dimension: first two entries define the block as `slice(beg, end)`,
        remaining entries = charge.
    qconj : {-1, 1}
        A flag telling whether the charge points inwards (+1) or outwards (-1).
        When charges are added, they are multiplied with their qconj value.

    Notes
    -----
    Instances of this class can be shared between different `npc.Array`s.
    """

    def __init__(self, chargeinfo, qind, qconj=1):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.qind = np.array(qind, dtype=QDTYPE)
        self.qconj = qconj
        self.test_sanity()
        # TODO attributes blocked, sorted

    @classmethod
    def from_trivial(cls, ind_len, qconj=1):
        """create trivial (qnumber=0) LegCharge for given len of indices `ind_len`"""
        ci = ChargeInfo()
        return cls(ci, [[0, ind_len]], qconj)

    @classmethod
    def from_qflat(cls, chargeinfo, qflat, qconj=1):
        """create a LegCharge from qflat form.

        Parameters
        ----------
        chargeinfo : :class:`ChargeInfo`
            the nature of the charge
        qflat : list(charges)
            `qnumber` charges for each index of the leg on entry
        qconj : {-1, 1}
            A flag telling whether the charge points inwards (+1) or outwards (-1).
        """
        qflat = np.asarray(qflat, dtype=QDTYPE)
        indices = _find_row_differences(qflat).reshape(-1, 1)
        qind = np.hstack((indices[:-1], indices[1:], qflat[indices[:-1, 0]]))
        return cls(chargeinfo, qind, qconj)

    @classmethod
    def from_qind(cls, chargeinfo, qflat, qconj=1):
        """just a wrapper around self.__init__"""
        return cls(chargeinfo, qflat, qconj)

    @classmethod
    def from_qdict(cls, chargeinfo, qdict, qconj=1):
        """create a LegCharge from qdict form."""
        qind = [[sl.start, sl.stop] + list(ch) for (ch, sl) in qdict.iteritems()]
        qind = np.array(qind, dtype=QDTYPE)
        sort = np.argsort(qind[:, 0])
        qind = qind[sort, :]
        # TODO: this is blocked...
        return cls(chargeinfo, qind, qconj)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        qind = self.qind
        if qind.shape[1] != 2 + self.chinfo.qnumber:
            raise ValueError("shape of qind incompatible with qnumber")
        if np.any(qind[:, 0] >= qind[:, 1]):
            raise ValueError("Invalid slice in qind: beg >= end:\n" + str(self))
        if np.any(qind[:-1, 1] != qind[1:, 0]):
            raise ValueError("The slices of qind are not contiguous.\n" + str(self))
        if not self.chinfo.check_valid(qind[2:]):
            raise ValueError("qind charges invalid for " + str(self.chinfo) + "\n" + str(self))
        if self.qconj != -1 and self.qconj != 1:
            raise ValueError("qconj has invalid value != +-1 :" + str(self.qconj))

    @property
    def ind_len(self):
        """the number of indices for this leg"""
        return self.qind[-1, 1]

    @property
    def block_number(self):
        """the number of blocks, i.e., a qindex is in ``range(block_number)``."""
        return self.qind.shape[0]

    def conj(self):
        """return a shallow copy with opposite ``self.qconj``"""
        res = copy.copy(self)  # shallow
        res.qconj *= -1
        return res

    def to_qflat(self):
        """return `self.qind` in `qdict` form"""
        qflat = np.empty((self.ind_len, self.chinfo.qnumber), dtype=QDTYPE)
        for qsec in self.qind:
            qflat[slice(qsec[0], qsec[1])] = qsec[2:]
        return qflat

    def to_qind(self):
        """return `self.qind`"""
        return self.qind

    def to_qdict(self):
        """return `self.qind` in `qdict` form. Raises ValueError, if not blocked."""
        # TODO: needs self.blocked()
        # if not self.blocked:
        #     raise ValueError("can't convert qflat to qdict for non-blocked LegCharge")
        return dict([(tuple(qsec[2:]), slice(qsec[0], qsec[1])) for qsec in self.qind])

    def __str__(self):
        """return a string of qind"""
        return str(self.qind)

    def __repr__(self):
        """full string representation"""
        return "LegCharge({0:r},{1:s})".format(self.chinfo, self.qind)


def _find_row_differences(qflat):
    """Return indices where the rows of the 2D array `qflat` change.

    Parameters
    ----------
    qflat : 2D array
        the rows of this array are compared.

    Returns
    -------
    1D array:
        The indices where rows change, including the first and last. Equivalent to:
        ``[0]+[i for i in range(1, len(qflat)) if np.any(qflat[i-1] != qflat[i])] + [len(qflat)]``
    """
    if len(qflat) == 1:
        return
    diff = np.ones(qflat.shape[0] + 1, dtype=np.bool_)
    diff[1:-1] = np.any(qflat[1:] != qflat[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values
