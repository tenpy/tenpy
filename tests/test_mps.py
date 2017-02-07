"""A collection of tests for :module:`tenpy.networks.mps`.

.. todo ::
    A lot more to test...
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

from tenpy.networks import mps, site

site_spin_half = site.spin_half_site(conserve='Sz')


def test_mps():
    for L in [4, 2, 1]:
        print L
        state = ([0, 1] * L)[:L]
        psi = mps.MPS.from_product_state([site_spin_half] * L, state, bc='finite')
        psi.test_sanity()
        print repr(psi)
        print str(psi)
