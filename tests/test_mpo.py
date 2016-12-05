"""A collection of tests for (classes in) :module:`tenpy.networks.mpo`.

.. todo :
    A lot more to test...
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

from tenpy.networks import mpo

from test_model import site_spin_half  # TODO: example sites....

def test_MPOGraph(L=4):
    for L in [4, 2, 1]:
        print L
        g = mpo.MPOGraph([site_spin_half]*L, 'finite')
        g.test_sanity()
        print g
