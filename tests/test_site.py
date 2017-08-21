"""A collection of tests for :mod:`tenpy.models.site`.

.. todo ::
    More tests of commutators for the various special sites!
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

import tenpy.linalg.np_conserved as npc
from tenpy.networks import site

from random_test import gen_random_legcharge


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


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


def test_spin_half_site():
    for conserve in ['Sz', 'parity', None]:
        S = site.SpinHalfSite(conserve)
        S.test_sanity()
    npt.assert_equal((S.Sx + 1.j * S.Sy).to_ndarray(), S.Sp.to_ndarray())
    npt.assert_equal((S.Sx - 1.j * S.Sy).to_ndarray(), S.Sm.to_ndarray())
    for i in range(3):
        Sa, Sb, Sc = ([S.Sx, S.Sy, S.Sz] * 2)[i:i + 3]
        # for pauli matrices ``sigma_a . sigma_b = 1.j * epsilon_{a,b,c} sigma_c``
        # with ``Sa = 0.5 sigma_a``, we get ``Sa . Sb = 0.5j epsilon_{a,b,c} Sc``.
        npt.assert_equal(np.dot(Sa.to_ndarray(), Sb.to_ndarray()), 0.5j * Sc.to_ndarray())


def test_spin_site():
    for s in [0.5, 1, 1.5, 2, 5]:
        print 's = ', s
        for conserve in ['Sz', 'parity', None]:
            S = site.SpinSite(s, conserve)
            S.test_sanity()
        npt.assert_equal((S.Sx + 1.j * S.Sy).to_ndarray(), S.Sp.to_ndarray())
        npt.assert_equal((S.Sx - 1.j * S.Sy).to_ndarray(), S.Sm.to_ndarray())
        Sx, Sy, Sz = S.Sx.to_ndarray(), S.Sy.to_ndarray(), S.Sz.to_ndarray()
        Sp, Sm = S.Sp.to_ndarray(), S.Sm.to_ndarray()
        tol = S.dim*S.dim
        for i in range(3):
            Sa, Sb, Sc = ([Sx, Sy, Sz] * 2)[i:i + 3]
            npt.assert_allclose(commutator(Sa, Sb), 1.j*Sc, tol, tol)
        npt.assert_array_almost_equal_nulp(commutator(Sz, Sp), Sp, tol)
        npt.assert_array_almost_equal_nulp(commutator(Sz, Sm), -Sm, tol)


def test_fermion_site():
    for conserve in ['N', 'parity', None]:
        S = site.FermionSite(conserve)
        S.test_sanity()
    npt.assert_equal(np.dot(S.Cd.to_ndarray(), S.C.to_ndarray()), S.N.to_ndarray())


def test_boson_site():
    for Nmax in [1, 2, 5, 10]:
        for conserve in ['N', 'parity', None]:
            S = site.BosonSite(Nmax, conserve=conserve)
            S.test_sanity()
        npt.assert_array_almost_equal_nulp(
            np.dot(S.Bd.to_ndarray(), S.B.to_ndarray()), S.N.to_ndarray(), 2)
