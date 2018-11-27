"""A collection of tests for :module:`tenpy.networks.mps`.

"""
# Copyright 2018 TeNPy Developers

import numpy as np
import numpy.testing as npt
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.lattice import Square

from tenpy.networks import mps, site
from random_test import rand_permutation, random_MPS
import tenpy.linalg.np_conserved as npc

from nose.plugins.attrib import attr

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
        ov = psi.overlap(psi2)
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
    #  ov = psi.overlap(psi_sum)
    #  print("ov = ", ov)
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
    assert abs(psi.overlap(psi_swap) - 1.) < 1.e-15
    # now test permutation
    # recover original psi
    psi = mps.MPS.from_singlets(spin_half, L, pairs, bc='finite')
    perm = rand_permutation(L)
    pairs_perm = [(perm[i], perm[j]) for i, j in pairs]
    psi_perm = mps.MPS.from_singlets(spin_half, L, pairs_perm, bc='finite')
    psi.permute_sites(perm, verbose=2)
    print(psi.overlap(psi_perm), psi.norm_test())
    assert abs(abs(psi.overlap(psi_perm)) - 1.) < 1.e-10


def test_TransferMatrix(chi=4, d=2):
    psi = random_MPS(2, d, chi, bc='infinite', form=None)
    full_TM = npc.tensordot(psi._B[0], psi._B[0].conj(), axes=['p', 'p*'])
    full_TM = npc.tensordot(full_TM, psi._B[1], axes=['vR', 'vL'])
    full_TM = npc.tensordot(full_TM, psi._B[1].conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
    full_TM = full_TM.combine_legs([['vL', 'vL*'], ['vR', 'vR*']], qconj=[+1, -1])
    full_TM_dense = full_TM.to_ndarray()
    eta_full, w_full = np.linalg.eig(full_TM_dense)
    sort = np.argsort(np.abs(eta_full))[::-1]
    eta_full = eta_full[sort]
    w_full = w_full[:, sort]
    TM = mps.TransferMatrix(psi, psi, charge_sector=0, form=None)
    eta, w = TM.eigenvectors(3)
    print("transfer matrix yields eigenvalues ", eta)
    print(eta.shape, eta_full.shape)
    print(psi.dtype)
    # note: second and third eigenvalue are complex conjugates
    if bool(eta[2].imag > 0.) == bool(eta_full[2].imag > 0.):
        npt.assert_allclose(eta[:3], eta_full[:3])
    else:
        npt.assert_allclose(eta[:3], eta_full[:3].conj())
    # compare largest eigenvector
    w0_full = w_full[:, 0]
    w0 = w[0].to_ndarray()
    assert (abs(np.sum(w0_full)) > 1.e-20)  # should be the case for random stuff
    w0_full /= np.sum(w0_full)  # fixes norm & phase
    w0 /= np.sum(w0)
    npt.assert_allclose(w0, w0_full)


def test_compute_K():
    pairs = [(0, 1), (2, 3), (4, 5)]  # singlets on a 3x2 grid -> k_y = pi
    psi = mps.MPS.from_singlets(spin_half, 6, pairs, bc='infinite')
    psi.test_sanity()
    lat = Square(3, 2, spin_half, order='default', bc_MPS='infinite')
    U, W, q, ov, te = psi.compute_K(lat, verbose=100)
    assert (ov == -1.)
    npt.assert_array_equal(W, [1.])


def check_canonical_form(bc):
    psi = random_MPS(8, 2, 6, form=None, bc=bc)
    psi2 = psi.copy()
    norm = np.sqrt(psi2.overlap(psi2, ignore_form=True))
    print("norm =", norm)
    psi2.norm /= norm  # normalize psi2
    norm2 = psi.overlap(psi2, ignore_form=True)
    print("norm2 =", norm2)
    assert abs(norm2 - norm) < 1.e-14 * norm
    psi.canonical_form(renormalize=False)
    psi.test_sanity()
    assert abs(psi.norm - norm) < 1.e-14 * norm
    psi.norm = 1.  # normalized psi
    ov = psi.overlap(psi2, ignore_form=True)
    print("normalized states: overlap <psi_canonical|psi> = 1.-", 1. - ov)
    assert abs(ov - 1.) < 1.e-14
    print("norm_test")
    print(psi.norm_test())
    assert np.max(psi.norm_test()) < 1.e-14


@attr('slow')
def test_canonical_form():
    yield check_canonical_form, 'finite'
    yield check_canonical_form, 'infinite'


def test_group():
    s = site.SpinHalfSite(conserve='parity')
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='finite')
    psi2 = psi1.copy()
    print("group n=2")
    psi2.group_sites(n=2)
    assert psi2.L == psi1.L//2
    psi2.test_sanity()
    psi2.group_split()
    psi2.test_sanity()
    ov = psi1.overlap(psi2)
    assert abs(1.-ov) < 1.e-14
    psi4 = psi1.copy()
    print("group n=4")
    psi4.group_sites(n=4)
    psi4.test_sanity()
    psi4.group_split()
    psi4.test_sanity()
    ov = psi1.overlap(psi4)
    assert abs(1.-ov) < 1.e-14


def test_expectation_value_term():
    s = spin_half
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='finite')
    ev = psi1.expectation_value_term([('Sz', 2), ('Sz', 3)])
    assert abs(0.-ev) < 1.e-14
    ev = psi1.expectation_value_term([('Sz', 1), ('Sz', 3)])
    assert abs(-0.25-ev) < 1.e-14
    ev = psi1.expectation_value_term([('Sz', 3), ('Sz', 1), ('Sz', 4)])
    assert abs(-0.25*0.5-ev) < 1.e-14
    fs = site.SpinHalfFermionSite()
    # check fermionic signs
    psi2 = mps.MPS.from_product_state([fs]*4, ['empty', 'up', 'down', 'full'], bc="infinite")
    ev = psi2.expectation_value_term([('Cu', 2), ('Nu', 1), ('Cdu', 2)])
    assert 1. == ev
    ev2 = psi2.expectation_value_term([('Cu', 2), ('Cd', 1), ('Cdd', 1), ('Cdu', 2)])
    assert ev2 == ev
    ev3 = psi2.expectation_value_term([('Cd', 1), ('Cu', 2), ('Cdd', 1), ('Cdu', 2)])
    assert ev3 == -ev2
    # over the infinite MPS boundary
    ev = psi2.expectation_value_term([('Nu', 1), ('Nd', 4)]) # should be zero
    assert abs(ev) == 0.
    ev = psi2.expectation_value_term([('Nu', 1), ('Nd', 6)]) # should be zero
    assert abs(ev) == 1.
    # terms_sum
    pref = np.random.random([5])
    evsum, _ = psi2.expectation_value_terms_sum([[('Nd', 0)],
                                                 [('Nu', 1), ('Nd', 2)],
                                                 [('Nd', 2), ('Nu', 5)],
                                                 [('Nu Nd', 3)],
                                                 [('Nu', 1), ('Nu', 5)]],
                                                pref) # should be zero
    assert abs(evsum) == sum(pref[1:])


if __name__ == "__main__":
    test_mps()
    test_mps_add()
    test_MPSEnvironment()
    test_singlet_mps()
    test_TransferMatrix()
    test_compute_K()
    check_canonical_form('finite')
    check_canonical_form('infinite')
    test_group()
    test_expectation_value_term()
