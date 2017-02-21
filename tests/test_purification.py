"""A collection of tests for :module:`tenpy.networks.purification_mps`."""

from __future__ import division

import numpy as np
import numpy.testing as npt
from tenpy.models.xxz_chain import XXZChain

from tenpy.networks import purification_mps, site
from tenpy.algorithms.purification_tebd import PurificationTEBD

spin_half = site.SpinHalfSite(conserve='Sz')


def test_purification_mps():
    for L in [4, 2, 1]:
        print L
        psi = purification_mps.PurificationMPS.from_infinteT([spin_half] * L, bc='finite')
        psi.test_sanity()
        if L > 1:
            npt.assert_equal(psi.entanglement_entropy(), 0.)  # product state has no entanglement.
        N = psi.expectation_value('Id')     # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
        E = psi.expectation_value('Sz')
        npt.assert_array_almost_equal_nulp(E, np.zeros([L]), 100)
        C = psi.correlation_function('Sz', 'Sz')
        npt.assert_array_almost_equal_nulp(C, 0.5 * 0.5 * np.eye(L), 100)


def test_purification_TEBD(L=4):
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    for disent in [None, 'backwards', 'renyi']:
        psi = purification_mps.PurificationMPS.from_infinteT(M.lat.mps_sites(), bc='finite')
        TEBD_params = {'chi_max': 32, 'svd_min': 1.e-13, 'disentangle': disent,
                       'verbose': 30, 'N_steps': 3}
        eng = PurificationTEBD(psi, M, TEBD_params)
        eng.run()
        print psi.get_B(0)
        N = psi.expectation_value('Id')     # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
        eng.run_imaginary(1.)
        print psi.get_B(0)
        N = psi.expectation_value('Id')     # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
