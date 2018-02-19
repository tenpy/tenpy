"""A collection of tests for :module:`tenpy.networks.mps`.

.. todo ::
    A lot more to test...
    Test for compute_K !
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import numpy.testing as npt
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.lattice import SquareLattice

from tenpy.networks import mps, site
from random_test import gen_random_legcharge, rand_permutation
import tenpy.linalg.np_conserved as npc

spin_half = site.SpinHalfSite(conserve='Sz')


def test_mps():
    site_triv = site.SpinHalfSite(conserve=None)
    psi = mps.MPS.from_product_state([site_triv] * 4, [0, 1, 0, 1], bc='finite')
    psi.test_sanity()
    for L in [4, 2, 1]:
        print(L)
        state = (spin_half.state_indices(['up', 'down']) * L)[:L]
        psi = mps.MPS.from_product_state([spin_half] * L, state, bc='finite')
        psi.test_sanity()
        print(repr(psi))
        print(str(psi))
        psi2 = psi.copy()
        ov, env = psi.overlap(psi2)
        assert (abs(ov - 1.) < 1.e-15)
        if L > 1:
            npt.assert_equal(psi.entanglement_entropy(), 0.)  # product state has no entanglement.
        E = psi.expectation_value('Sz')
        npt.assert_array_almost_equal_nulp(E, ([0.5, -0.5] * L)[:L], 100)
        C = psi.correlation_function('Sz', 'Sz')
        npt.assert_array_almost_equal_nulp(C, np.outer(E, E), 100)
        norm_err = psi.norm_test()
        assert (np.linalg.norm(norm_err) < 1.e-13)


def test_mps_add():
    s = site.SpinHalfSite(conserve='Sz')
    psi1 = mps.MPS.from_product_state([s] * 4, [0, 1, 0, 0], bc='finite')
    psi2 = mps.MPS.from_product_state([s] * 4, [0, 0, 1, 0], bc='finite')
    psi_sum = psi1.add(psi2, 0.5**0.5, -0.5**0.5)
    print(psi_sum)
    print(psi_sum._B[1])
    print(psi_sum._B[2])
    # check overlap with singlet state
    # TODO: doesn't work due to gauging of charges....
    psi = mps.MPS.from_singlets(s, 4, [(1, 2)], lonely=[0, 3], up=0, down=1, bc='finite')
    print(psi.expectation_value('Sz'))
    #  ov = psi.overlap(psi_sum)[0]
    #  print "ov = ", ov
    #  assert( abs(1.-ov) < 1.e-14)


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
        print("total contraction on site", i, ": ov = 1. - ", ov - 1.)
        assert (abs(abs(ov) - 1.) < 1.e-14)
    env.expectation_value('Sz')


def test_singlet_mps():
    pairs = [(0, 3), (1, 6), (2, 5)]
    bond_singlets = np.array([1, 2, 3, 2, 2, 1, 0])
    lonely = [4, 7]
    L = 2 * len(pairs) + len(lonely)
    print("singlet pairs: ", pairs)
    print("lonely sites: ", lonely)
    psi = mps.MPS.from_singlets(spin_half, L, pairs, lonely=lonely, up=0, down=1, bc='finite')
    psi.test_sanity()
    print('chi = ', psi.chi)
    assert (np.all(2**bond_singlets == np.array(psi.chi)))
    ent = psi.entanglement_entropy() / np.log(2)
    npt.assert_array_almost_equal_nulp(ent, bond_singlets, 5)
    psi.entanglement_spectrum(True)  # (just check that the function runs)
    print(psi.overlap(psi))
    print(psi.expectation_value('Id'))
    ent_segm = psi.entanglement_entropy_segment(list(range(4))) / np.log(2)
    print(ent_segm)
    npt.assert_array_almost_equal_nulp(ent_segm, [2, 3, 1, 3, 2], 5)
    coord, mutinf = psi.mutinf_two_site()
    coord = [(i, j) for i, j in coord]
    mutinf[np.abs(mutinf) < 1.e-14] = 0.
    mutinf /= np.log(2)
    print(mutinf)
    for (i, j) in pairs:
        k = coord.index((i, j))
        mutinf[k] -= 2.  # S(i)+S(j)-S(ij) = (1+1-0)*log(2)
    npt.assert_array_almost_equal_nulp(mutinf, 0., 10)
    # TODO: calculating overlap with product state
    # TODO: doesn't work yet: different qtotal.
    #  product_state = [None]*L
    #  for i, j in pairs:
    #      product_state[i] = 0
    #      product_state[j] = 1
    #  for k in lonely:
    #      product_state[k] = 0
    #  psi2 = mps.MPS.from_product_state([spin_half]*L, product_state, bc='finite')
    #  ov = psi.overlap(psi2)


def test_mps_swap():
    L = 6
    pairs = [(0, 3), (1, 5), (2, 4)]
    pairs_swap = [(0, 2), (1, 5), (3, 4)]
    print("singlet pairs: ", pairs)
    psi = mps.MPS.from_singlets(spin_half, L, pairs, bc='finite')
    psi_swap = mps.MPS.from_singlets(spin_half, L, pairs_swap, bc='finite')
    psi.swap_sites(2)
    assert abs(psi.overlap(psi_swap)[0] - 1.) < 1.e-15
    # now test permutation
    # recover original psi
    psi = mps.MPS.from_singlets(spin_half, L, pairs, bc='finite')
    perm = rand_permutation(L)
    pairs_perm = [(perm[i], perm[j]) for i, j in pairs]
    psi_perm = mps.MPS.from_singlets(spin_half, L, pairs_perm, bc='finite')
    psi.permute_sites(perm, verbose=2)
    print(psi.overlap(psi_perm)[0], psi.norm_test())
    assert abs(abs(psi.overlap(psi_perm)[0]) - 1.) < 1.e-10


def test_transfermatrix(chi=6, d=3):
    ch = npc.ChargeInfo([2])
    p = gen_random_legcharge(ch, d, qconj=1)
    vL = gen_random_legcharge(ch, chi, qconj=1)
    vR = gen_random_legcharge(ch, chi, qconj=-1)
    A = npc.Array.from_func(np.random.random, [vL, p, vR], shape_kw='size')
    B = npc.Array.from_func(np.random.random, [vR.conj(), p, vL.conj()], shape_kw='size')
    A.iset_leg_labels(['vL', 'p', 'vR'])
    B.iset_leg_labels(['vL', 'p', 'vR'])
    S = [np.ones(chi)] * 3
    psi = mps.MPS([site.Site(p)] * 2, [A, B], S, 'infinite', form=None)
    # now actually generate the transfermatrix
    TM = mps.TransferMatrix(psi, psi, charge_sector=0)
    eta, w = TM.eigenvectors(3)
    print("transfer matrix yields:")
    print(eta, w)


def test_compute_K():
    pairs = [(0, 1), (2, 3), (4, 5)]  # singlets on a 3x2 grid -> k_y = pi
    psi = mps.MPS.from_singlets(spin_half, 6, pairs, bc='infinite')
    psi.test_sanity()
    lat = SquareLattice(3, 2, spin_half, order='default', bc_MPS='infinite')
    U, W, q, ov, te = psi.compute_K(lat, verbose=100)
    assert (ov == -1.)
    npt.assert_array_equal(W, [1.])


if __name__ == "__main__":
    test_mps()
    test_mps_add()
    test_MPSEnvironment()
    test_singlet_mps()
    test_transfermatrix()
    test_compute_K()
