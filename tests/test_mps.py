"""A collection of tests for :module:`tenpy.networks.mps`.

.. todo ::
    A lot more to test...
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst
from tenpy.models.xxz_chain import XXZChain

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
        psi2 = psi.copy()
        ov, env = psi.overlap(psi2)
        assert(abs(ov - 1.) < 1.e-15)
        psi.expectation_value('Sz')


def test_MPSEnvironment():
    xxz_pars = dict(L=4, Jxx=1., Jz=1.1, hz=0.1, bc_MPS='finite')
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1] * L)[:L]  # Neel state
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc='finite')
    env = mps.MPSEnvironment(psi, psi)
    env.get_LP(3, True)
    env.get_RP(0, True)
    env.test_sanity()
    for i in range(4):
        ov = env.full_contraction(i)  # should be one
        print "total contraction on site", i, ": ov = 1. - ", ov - 1.
        assert(abs(abs(ov)-1.) < 1.e-14)
    env.expectation_value('Sz')

if __name__ == "__main__":
    test_mps()
    test_MPSEnvironment()
