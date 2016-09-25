r"""A module to handle charge conservation in tensor networks.

This module `np_conserved` implements an class :class:`Array`
designed make use of charge conservation in tensor networks.
The idea is that the `Array` class is used in a fashion very similar to
the `numpy.ndarray`, e.g you can call the functions :func:`tensordot` or :func:`svd`
(of this module) on them.
The structure of the algorithms (as DMRG) is thus the same as with basic numpy ndarrays.

Internally, an :class:`Array` saves charge meta data to keep track of blocks which are nonzero.
All possible operations (e.g. tensordot, svd, ...) on such arrays preserve the total charge
structure. In addition, these operations make use of the charges to figure out which of the blocks
it hase to use/combine - this is the basis for the speed-up.

A more detailed introduction (including notations) can be found in :doc:`../IntroNpc`.


See also
--------
tenpy.linalg.charges : Implementation of :class:`~tenpy.linalg.charge.ChargeInfo`
    and :class:`~tenpy.linalg.charge.LegCharge` with addiontal documentation.

Notes
-----

Details on the usage will follow soon. # TODO

"""
# # TODO usage and examples, Routine listing
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
from .charges import (ChargeInfo, LegCharge)


class Array(object):
    r"""A multidimensional array (=tensor) for using charge conservation.

    This class

    Attributes
    ----------
    shape
    charge : :class:`charge`
        The charge type of the tensor

    """

    def __init__(self, charge_info, leg_charges, dtype):
        r"""Initialize self

        Should only be called
        You
        See also
        --------
        Array.from_ndarray : initalize from a leg charge
        zeros : create a
        """
        self.charge = charge
        self.shape

        self._data = []
        self._data
        raise NotImplementedError()

    def test_sanitiy(self):
        raise NotImplementedError()

    def __repr__(self):
        return "<npc.array charge={0:s} shape={0:s}>".format(self.charge, self.shape)

    def __str__(self):
        return self.to_ndarray().__str__()

    pass  # TODO more stuff...


def zeros(shape, ):
    return
