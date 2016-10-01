r"""A module to handle charge conservation in tensor networks.

A detailed introduction (including notations) can be found in :doc:`../IntroNpc`.

This module `np_conserved` implements an class :class:`Array`
designed to make use of charge conservation in tensor networks.
The idea is that the `Array` class is used in a fashion very similar to
the `numpy.ndarray`, e.g you can call the functions :func:`tensordot` or :func:`svd`
(of this module) on them.
The structure of the algorithms (as DMRG) is thus the same as with basic numpy ndarrays.

Internally, an :class:`Array` saves charge meta data to keep track of blocks which are nonzero.
All possible operations (e.g. tensordot, svd, ...) on such arrays preserve the total charge
structure. In addition, these operations make use of the charges to figure out which of the blocks
it hase to use/combine - this is the basis for the speed-up.


See also
--------
tenpy.linalg.charges : Implementation of :class:`~tenpy.linalg.charge.ChargeInfo`
    and :class:`~tenpy.linalg.charge.LegCharge` with addiontal documentation.


.. todo ::
   usage and examples, Routine listing
   update ``from charges import``
"""
# Examples
# --------
# >>> import numpy as np
# >>> import tenpy.linalg.np_conserved as npc
# >>> Sz = np.array([[0., 1.], [1., 0.]])
# >>> Sz_c = npc.Array.from_flat(Sz)  # convert to npc array with trivial charge
# >>> Sz_c
# <npc.Array shape=(2, 2)>
# >>> sx = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>> b = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>>
# >>> print a[0, -1]
# >>> c = npc.tensordot(a, b, axes=([1], [0]))

from __future__ import division

import numpy as np
import copy

from .charges import (ChargeInfo, LegCharge)


class Array(object):
    r"""A multidimensional array (=tensor) for using charge conservation.

    Outside of np_conserved, a new array should be initialized with one of
    :meth:`from_ndarray` and :meth:`from_npfunc`, as :meth:`__init__` does not initialize any data.

    Parameters
    ----------
    chargeinfo: :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge, used as self.chinfo.
    legs: list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    dtype: type or string
        the data type of the array entries. Defaults to np.float64.


    Attributes
    ----------
    rank
    shape : tuple(int)
        the number of indices for each of the legs
    dtype : np.dtype
        the data type of the entries
    chinfo: :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge
    qtotal: charge values
        the total charge of the tensor.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    labels: dict (string -> int)
        labels for the different legs

    .. todo ::
        test everything
    """
    def __init__(self, chargeinfo, legcharges, dtype=np.float64, qtotal=None):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.legs = list(legcharges)
        self.shape = tuple([lc.ind_len() for lc in self.legs])
        self.dtype = np.dtype(dtype)
        self.qtotal = self.chinfo.make_valid(qtotal)
        self.labels = {}
        self._data = []
        self._qdata = []
        self.test_sanity()  # TODO: only in from_ndarray?

    @property
    def rank(self):
        """the number of legs"""
        return len(self.shape)

    def dtype_upcast(self, dtype, copy=False):
        """change the data type to dtype, upcasting all blocks in _data.

        """
        self.dtype = np.dtype(dtype)
        self._data = [d.astype(dtype, copy=copy) for d in self._data]

    def get_leg(self, index):
        """translate a leg-index or leg-label to a leg-index."""
        return self.labels.get(index, index)

    def get_legs(self, indices):
        """Translate a list of leg-indices or leg-labels to leg indices.

        See also
        --------
        get_leg: used to translate each of the single entries.
        """
        return [self.get_leg(i) for i in indices]

    def test_sanitiy(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.shape != tuple([lc.ind_len() for lc in self.legs]):
            raise ValueError("shape mismatch with LegCharges\n self.shape={0:s} != {1:s}".format(
                self.shape, tuple([lc.ind_len() for lc in self.legs])))
        if any([self.dtype != d.dtype for d in self._data]):
            raise ValueError("wrong dtype: {0:s} vs\n {1:s}".format(
                self.dtype, [self.dtype != d.dtype for d in self._data]))
        for l in self.legs:
            l.test_sanity()

    def gauge_total_charge(self, leg, newqtotal=None):
        """changes the total charge of an Array `A` inplace by adjusting the charge on a certain leg.

        The total charge is given by finding a nonzero entry [i1, i2, ...] and calculating::

            qtotal = sum([l.qconj[i] * l.qind[il] for il in zip([i1,i2,...], self.legs)])

        Thus, the total charge can be changed by redefining the leg charge of a given leg.
        This is exaclty what this function does.

        Parameters
        ----------
        leg: int or string
            the new leg (index or label), for which the charge is changed
        newqtotal: charge values, defaults to 0
            the new total charge
        """
        leg = self.get_leg(leg)
        newqtotal = self.chinfo.make_valid(newqtotal)  # converts to array, default zero
        chdiff = newqtotal - self.qtotal
        newleg = copy.copy(self.legs[leg])
        newleg.qind = newleg.qind.copy()
        newleg.qind[:, 2:] = self.chinfo.make_valid(newleg.qind[:, 2:] + newleg.qconj * chdiff)
        self.legs[leg] = newleg
        self.qtotal = newqtotal

    def __repr__(self):
        return "<npc.array charge={0:s} shape={0:s}>".format(self.charge, self.shape)

    def __str__(self):
        return self.to_ndarray().__str__()

    def __iter__(self):
        """Allow to iterate over the non-zero blocks, giving all data.

        Yields
        ------
        block: ndarray
            the actual entries of a charge block
        blockslices : list of slices
            a slice giving the range of the block in the original tensor for each of the legs
        charges : list of charges
            the charge value(s) for each of the legs
        qdat : ndarray
            the qind for each of the legs
        """
        for block, qdat in zip(self._data, self._qdata):
            qind = [l.qind[qind] for (qind, l) in zip(qdat, self.legs)]
            blockslices = [slice(qi[0], qi[1]) for qi in qind]
            charges = [qi[2:] for qi in qind]
            yield block, blockslices, charges, qdat


def zeros(shape, ):
    return
