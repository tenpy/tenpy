"""A collection of tests for :module:`tenpy.networks.purification_mps`."""

from __future__ import division

import warnings
import numpy as np
import numpy.testing as npt
from tenpy.models.xxz_chain import XXZChain

from tenpy.networks import purification_mps, site
from tenpy.algorithms.purification_tebd import PurificationTEBD
import tenpy.linalg.np_conserved as npc

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
        TEBD_params = {'chi_max': 16, 'svd_min': 1.e-13, 'disentangle': disent, 'dt': 0.1,
                       'verbose': 30, 'N_steps': 2}
        eng = PurificationTEBD(psi, M, TEBD_params)
        eng.run()
        print psi.get_B(0)
        N = psi.expectation_value('Id')     # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
        eng.run_imaginary(0.2)
        print psi.get_B(0)
        N = psi.expectation_value('Id')     # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)


def test_disentangler(L=4, eps=1.e-15):
    np.random.seed(12345)  # TODO: this doesn't work for all seeds!?!
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    psi = purification_mps.PurificationMPS.from_infinteT(M.lat.mps_sites(), bc='finite')
    TEBD_params = {'chi_max': 32, 'svd_min': 1.e-13, 'disentangle': 'renyi',
                   'verbose': 30, 'N_steps': 3}
    eng = PurificationTEBD(psi, M, TEBD_params)
    theta = eng.psi.get_theta(1, 2)
    print theta[0, :, :, 0, :, :]
    # find random unitary: SVD of random matix
    pleg = psi.sites[0].leg
    legs = [pleg, pleg, pleg.conj(), pleg.conj()]
    A = npc.Array.from_func(np.random.random, legs, shape_kw='size')
    A.set_leg_labels(['p0', 'p1', 'p0*', 'p1*'])
    A = A.combine_legs([[0, 1], [2, 3]], qconj=[+1, -1])
    X, Y, Z = npc.svd(A)
    A = npc.tensordot(X, Z, axes=[1, 0]).split_legs()
    # Now we have unitary `A`, i.e. the optimal `U` should be `A^dagger`.
    theta = npc.tensordot(A, theta, axes=[['p0*', 'p1*'], ['p0', 'p1']])

    U0 = npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
                   npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*']))
    U = U0
    Sold = np.inf
    for i in xrange(20):
        S, U = eng.disentangle_renyi_iter(theta, U)
        if i == 0:
            S_0 = S
        print "iteration {i:d}: S={S:.5f}, DS={DS:.4e} ".format(i=i, S=S, DS=Sold-S)
        if abs(Sold - S) < eps:
            print "break: S converged down to {eps:.1e}".format(eps=eps)
            break
        Sold, S = S, Sold
    else:
        print "maximum number of iterations reached"
    theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
    print "new theta = "
    print theta.itranspose(['vL', 'vR', 'p0', 'q0', 'p1', 'q1'])
    print theta[0, 0]
    assert(S < S_0)   # this should always be true...
    if S > eps:
        warnings.warn("test of purification failed to find the optimum.")
        # This may happen for some random seeds! Why?
        # If the optimal U is 'too far away' from U0=eye?
