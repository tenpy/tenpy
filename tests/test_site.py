"""A collection of tests for :mod:`tenpy.models.site`.

.. todo ::
    More tests of commutators for the various special sites!
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst
import itertools as it

import tenpy.linalg.np_conserved as npc
from tenpy.networks import site

from random_test import gen_random_legcharge


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def anticommutator(A, B):
    return np.dot(A, B) + np.dot(B, A)


def test_site():
    chinfo = npc.ChargeInfo([1, 3])
    leg = gen_random_legcharge(chinfo, 8)
    op1 = npc.Array.from_func(np.random.random, [leg, leg.conj()], shape_kw='size')
    op2 = npc.Array.from_func(np.random.random, [leg, leg.conj()], shape_kw='size')
    labels = ['up'] + [None] * 6 + ['down']
    s = site.Site(leg, labels, silly_op=op1)
    nst.eq_(s.state_index('up'), 0)
    nst.eq_(s.state_index('down'), 8 - 1)
    nst.eq_(s.opnames, set(['silly_op', 'Id']))
    assert (s.silly_op is op1)
    s.add_op('op2', op2)
    assert (s.op2 is op2)
    assert (s.get_op('op2') is op2)
    assert (s.get_op('silly_op') is op1)
    npt.assert_equal(s.get_op('silly_op op2').to_ndarray(),
                     npc.tensordot(op1, op2, [1, 0]).to_ndarray())


def check_spin_site(S, SpSmSz=['Sp', 'Sm', 'Sz'], SxSy=['Sx', 'Sy']):
    """Test whether the spins operators behave as expected.

    `S` should be a :class:`site.Site`.
    Set `SxSy` to `None` to ignore Sx and Sy
    (if they don't exist as npc.Array due to conservation).
    """
    Sp, Sm, Sz = SpSmSz
    Sp, Sm, Sz = S.get_op(Sp).to_ndarray(), S.get_op(Sm).to_ndarray(), S.get_op(Sz).to_ndarray()
    npt.assert_almost_equal(commutator(Sz, Sp), Sp, 13)
    npt.assert_almost_equal(commutator(Sz, Sm), -Sm, 13)
    if SxSy is not None:
        Sx, Sy = SxSy
        Sx, Sy = S.get_op(Sx).to_ndarray(), S.get_op(Sy).to_ndarray()
        npt.assert_equal(Sx + 1.j * Sy, Sp)
        npt.assert_equal(Sx - 1.j * Sy, Sm)
        for i in range(3):
            Sa, Sb, Sc = ([Sx, Sy, Sz] * 2)[i:i + 3]
            # for pauli matrices ``sigma_a . sigma_b = 1.j * epsilon_{a,b,c} sigma_c``
            # with ``Sa = 0.5 sigma_a``, we get ``Sa . Sb = 0.5j epsilon_{a,b,c} Sc``.
            #  npt.assert_almost_equal(np.dot(Sa, Sb), 0.5j*Sc, 13) # holds only for S=1/2
            npt.assert_almost_equal(commutator(Sa, Sb), 1.j*Sc, 13)


def test_spin_half_site():
    for conserve in ['Sz', 'parity', None]:
        S = site.SpinHalfSite(conserve)
        S.test_sanity()
        if conserve != 'Sz':
            SxSy = ['Sx', 'Sy']
        else:
            SxSy = None
        check_spin_site(S, SxSy=SxSy)


def test_spin_site():
    for s in [0.5, 1, 1.5, 2, 5]:
        print 's = ', s
        for conserve in ['Sz', 'parity', None]:
            print "conserve = ", conserve
            S = site.SpinSite(s, conserve)
            S.test_sanity()
            if conserve != 'Sz':
                SxSy = ['Sx', 'Sy']
            else:
                SxSy = None
            check_spin_site(S, SxSy=SxSy)


def test_fermion_site():
    for conserve in ['N', 'parity', None]:
        S = site.FermionSite(conserve)
        S.test_sanity()
        C, Cd, N = S.C.to_ndarray(), S.Cd.to_ndarray(), S.N.to_ndarray()
        Id = S.Id.to_ndarray()
        JW = S.JW.to_ndarray()
        npt.assert_equal(np.dot(Cd, C), N)
        npt.assert_equal(anticommutator(Cd, C), Id)
        npt.assert_equal(np.dot(Cd, C), N)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(C, JW), -np.dot(JW, C))


def test_spin_half_fermion_site():
    for cons_N, cons_Sz in it.product(['N', 'parity', None], ['Sz', 'parity', None]):
        print "conserve ", repr(cons_N), repr(cons_Sz)
        S = site.SpinHalfFermionSite(cons_N, cons_Sz)
        S.test_sanity()
        Id = S.Id.to_ndarray()
        JW = S.JW.to_ndarray()
        Cu, Cd = S.Cu.to_ndarray(), S.Cd.to_ndarray()
        Cud, Cdd = S.Cud.to_ndarray(), S.Cdd.to_ndarray()
        Nu, Nd, Ntot = S.Nu.to_ndarray(), S.Nd.to_ndarray(), S.Ntot.to_ndarray()
        npt.assert_equal(np.dot(Cud, Cu), Nu)
        npt.assert_equal(np.dot(Cdd, Cd), Nd)
        npt.assert_equal(Nu + Nd, Ntot)
        npt.assert_equal(np.dot(Nu, Nd), S.NuNd.to_ndarray())
        npt.assert_equal(anticommutator(Cud, Cu), Id)
        npt.assert_equal(anticommutator(Cdd, Cd), Id)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cu, JW), -np.dot(JW, Cu))
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(Cud, JW), -np.dot(JW, Cud))
        npt.assert_equal(np.dot(Cdd, JW), -np.dot(JW, Cdd))
        # anti-commute Cu with Cd
        npt.assert_equal(np.dot(Cu, Cd), -np.dot(Cd, Cu))
        npt.assert_equal(np.dot(Cu, Cdd), -np.dot(Cdd, Cu))
        npt.assert_equal(np.dot(Cud, Cd), -np.dot(Cd, Cud))
        npt.assert_equal(np.dot(Cud, Cdd), -np.dot(Cdd, Cud))
        if cons_Sz != 'Sz':
            SxSy = ['Sx', 'Sy']
        else:
            SxSy = None
        check_spin_site(S, SxSy=SxSy)


def test_boson_site():
    for Nmax in [1, 2, 5, 10]:
        for conserve in ['N', 'parity', None]:
            S = site.BosonSite(Nmax, conserve=conserve)
            S.test_sanity()
        npt.assert_array_almost_equal_nulp(
            np.dot(S.Bd.to_ndarray(), S.B.to_ndarray()), S.N.to_ndarray(), 2)
