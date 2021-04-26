"""A collection of tests for :module:`tenpy.networks.purification_mps`."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import warnings
import numpy as np
import numpy.testing as npt
import scipy
from tenpy.models.xxz_chain import XXZChain

from tenpy.networks import purification_mps, site
from tenpy.networks.mps import MPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
import tenpy.linalg.random_matrix as rmat
import tenpy.linalg.np_conserved as npc
import pytest

spin_half = site.SpinHalfSite(conserve='Sz')
ferm = site.FermionSite(conserve='N')


def test_purification_mps():
    for L in [4, 2, 1]:
        print(L)
        psi = purification_mps.PurificationMPS.from_infiniteT([spin_half] * L, bc='finite')
        psi.test_sanity()
        if L > 1:
            npt.assert_equal(psi.entanglement_entropy(), 0.)  # product state has no entanglement.
        N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
        npt.assert_allclose(N, np.ones([L]), atol=1.e-13)
        E = psi.expectation_value('Sz')
        npt.assert_allclose(E, np.zeros([L]), atol=1.e-13)
        C = psi.correlation_function('Sz', 'Sz')
        npt.assert_allclose(C, 0.5 * 0.5 * np.eye(L), atol=1.e-13)
        coords, mutinf = psi.mutinf_two_site()
        for (i, j), Iij in zip(coords, mutinf):
            print(repr((i, j)), Iij)
        if L > 1:
            assert np.max(np.abs(mutinf)) < 1.e-14
        if L >= 2:
            # test that grouping sites works
            print('group & split sites')
            psi.group_sites(2)
            psi.test_sanity()
            psi.group_split()
            C = psi.correlation_function('Sz', 'Sz')
            npt.assert_allclose(C, 0.5 * 0.5 * np.eye(L), atol=1.e-13)


def test_canoncial_purification(L=6, charge_sector=0, eps=1.e-14):
    site = spin_half
    psi = purification_mps.PurificationMPS.from_infiniteT_canonical([site] * L, [charge_sector])
    psi.test_sanity()
    total_psi = psi.get_theta(0, L).take_slice(0, 'vL').take_slice(0, 'vR')
    total_psi.itranspose(['p' + str(i) for i in range(L)] + ['q' + str(i) for i in range(L)])
    # note: don't `combine_legs`: it will permute the p legs differently than q due to charges
    total_psi_dense = total_psi.to_ndarray().reshape(2**L, 2**L)
    # now it should be diagonal
    diag = np.diag(total_psi_dense)
    assert np.all(np.abs(total_psi_dense - np.diag(diag) < eps))  # is it diagonal?
    # and the diagonal should be sqrt(L choose L//2) for states with fitting numbers
    pref = 1. / scipy.special.comb(L, L // 2 + charge_sector)**0.5
    Q_p = site.leg.to_qflat()[:, 0]
    for i, entry in enumerate(diag):
        Q_i = sum([Q_p[int(b)] for b in format(i, 'b').zfill(L)])
        if Q_i == charge_sector:
            assert abs(entry - pref) < eps
        else:
            assert abs(entry) < eps

    # and one quick test of TEBD
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    TEBD_params = {
        'trunc_params': {
            'chi_max': 16,
            'svd_min': 1.e-8
        },
        'disentangle': None,  # 'renyi' should work as well, 'backwards' not.
        'dt': 0.1,
        'N_steps': 2
    }
    eng = PurificationTEBD(psi, M, TEBD_params)
    eng.run_imaginary(0.2)
    eng.run()
    N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
    npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)


@pytest.mark.slow
def test_purification_TEBD(L=3):
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    for disent in [
            None, 'backwards', 'min(None,last)-renyi', 'noise-norm', 'renyi-min(None,noise-renyi)'
    ]:
        psi = purification_mps.PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc='finite')
        TEBD_params = {
            'trunc_params': {
                'chi_max': 16,
                'svd_min': 1.e-8
            },
            'disentangle': disent,
            'dt': 0.1,
            'N_steps': 2
        }
        eng = PurificationTEBD(psi, M, TEBD_params)
        eng.run()
        N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
        eng.run_imaginary(0.2)
        N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
        npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
        if disent == 'last-renyi':
            eng.run_imaginary(0.3)
            eng.disentangle_global()


def test_purification_MPO(L=6):
    xxz_pars = dict(L=L, Jxx=1., Jz=2., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    psi = purification_mps.PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc='finite')
    options = {'trunc_params': {'chi_max': 50, 'svd_min': 1.e-8}}
    U = M.H_MPO.make_U_II(dt=0.1)
    PurificationApplyMPO(psi, U, options).run()
    N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
    npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)


def test_renyi_disentangler(L=4, eps=1.e-15):
    xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite')
    M = XXZChain(xxz_pars)
    psi = purification_mps.PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc='finite')
    eng = PurificationTEBD(psi, M, {'disentangle': 'renyi'})
    theta = eng.psi.get_theta(1, 2)
    print(theta[0, :, :, 0, :, :])
    # find random unitary: SVD of random matix
    pleg = psi.sites[0].leg
    pipe = npc.LegPipe([pleg, pleg])
    A = npc.Array.from_func_square(rmat.CUE, pipe).split_legs()
    A.iset_leg_labels(['p0', 'p1', 'p0*', 'p1*'])
    # Now we have unitary `A`, i.e. the optimal `U` should be `A^dagger`.
    theta = npc.tensordot(A, theta, axes=[['p0*', 'p1*'], ['p0', 'p1']])

    U0 = npc.outer(npc.eye_like(theta, 'q0', labels=['q0', 'q0*']),
                   npc.eye_like(theta, 'q1', labels=['q1', 'q1*']))
    U = U0
    Sold = np.inf
    for i in range(20):
        S, U = eng.used_disentangler.iter(theta, U)
        if i == 0:
            S_0 = S
        print("iteration {i:d}: S={S:.5f}, DS={DS:.4e} ".format(i=i, S=S, DS=Sold - S))
        if abs(Sold - S) < eps:
            print("break: S converged down to {eps:.1e}".format(eps=eps))
            break
        Sold, S = S, Sold
    else:
        print("maximum number of iterations reached")
    theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
    print("new theta = ")
    print(theta.itranspose(['vL', 'vR', 'p0', 'q0', 'p1', 'q1']))
    print(theta[0, 0])
    assert (S < S_0)  # this should always be true...
    if S > 100 * eps:
        print("final S =", S)
        assert False  # test of purification failed to find the optimum.
        # This may happen for some random seeds! Why?
        # If the optimal U is 'too far away' from U0=eye?


def gen_disentangler_psi_singlets(site_P, L, max_range=10, product_P=True):
    """generate an initial state of random singlets, identical in P and Q."""
    assert (L % 2 == 0)
    # generate pairs with given maximum range, for both P and Q
    pairs_PQ = [None, None]
    for i in range(2):
        pairs = pairs_PQ[i] = []
        have = list(range(L))
        while len(have) > 0:
            i = have.pop(0)
            js = [j for j in have[:max_range] if abs(i - j) <= max_range]
            j = have.pop(np.random.choice(len(js)))
            pairs.append((i, j))
    # generate singlet mps in P and Q
    if product_P:
        psiP = MPS.from_product_state([site_P] * L, [0, 1] * (L // 2))
    else:
        psiP = MPS.from_singlets(site_P, L, pairs_PQ[0])
    psiQ = MPS.from_singlets(site_P, L, pairs_PQ[1])
    # generate BS for PurificationMPS
    return gen_disentangler_psi_prod(psiP, psiQ), pairs_PQ


def gen_disentangler_psi_prod(psiP, psiQ):
    """generate a PurificationMPS as tensorproduct (psi_P x psi_Q).

    psiQ should have the same `sites` as psiP.
    """
    L = psiP.L
    Bs = []
    for i in range(L):
        BP = psiP.get_B(i)
        BQ = psiQ.get_B(i)
        B2 = npc.outer(BP, BQ.conj())
        B2 = B2.combine_legs([['vL', 'vL*'], ['vR', 'vR*']], qconj=[+1, -1])
        B2.ireplace_labels(['(vL.vL*)', '(vR.vR*)', 'p*'], ['vL', 'vR', 'q'])
        Bs.append(B2)
    Ss = [np.outer(S, S2).flatten() for S, S2 in zip(psiP._S, psiQ._S)]
    return purification_mps.PurificationMPS(psiP.sites, Bs, Ss)


@pytest.mark.slow
def gen_disentangler_psi_singlet_test(site_P=spin_half, L=6, max_range=4):
    psi0, pairs_PQ = gen_disentangler_psi_singlets(site_P, L, max_range)
    psi0.test_sanity()
    print("pairs: P", pairs_PQ[0])
    print("pairs: Q", pairs_PQ[1])
    print("entanglement entropy: ", psi0.entanglement_entropy() / np.log(2.))
    coords, mutinf_pq = psi0.mutinf_two_site(legs='pq')
    print("(i,j)=", [tuple(c) for c in coords])
    print("PQ:", np.round(mutinf_pq / np.log(2), 3))
    print("P: ", np.round(psi0.mutinf_two_site(legs='p')[1] / np.log(2), 3))
    print("Q: ", np.round(psi0.mutinf_two_site(legs='q')[1] / np.log(2), 3))
    M = XXZChain(dict(L=L))
    tebd_pars = dict(trunc_params={'trunc_cut': 1.e-10}, disentangle='diag')
    eng = PurificationTEBD(psi0, M, tebd_pars)
    for i in range(L):
        eng.disentangle_global()
    print(psi0.entanglement_entropy() / np.log(2))
    mutinf_Q = psi0.mutinf_two_site(legs='q')[1]
    print("P: ", np.round(psi0.mutinf_two_site(legs='p')[1] / np.log(2), 3))
    print("Q: ", np.round(mutinf_Q / np.log(2), 3))
    assert (np.all(mutinf_Q < 1.e-10))
