"""A collection of tests for :module:`tenpy.networks.purification_mps`."""

from __future__ import division

import numpy as np
import numpy.testing as npt
from tenpy.models.xxz_chain import XXZChain

from tenpy.networks import purification_mps, site

spin_half = site.SpinHalfSite(conserve='Sz')


def test_purification_mps():
    for L in [4, 2, 1]:
        print L
        psi = purification_mps.PurificationMPS.from_infinteT([spin_half] * L, bc='finite')
        psi.test_sanity()
        if L > 1:
            npt.assert_equal(psi.entanglement_entropy(), 0.)  # product state has no entanglement.
        E = psi.expectation_value('Sz')
        npt.assert_array_almost_equal_nulp(E, np.zeros([L]), 100)
        C = psi.correlation_function('Sz', 'Sz')
        npt.assert_array_almost_equal_nulp(C, 2*0.5*0.5*np.eye(L), 100)
