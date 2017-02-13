"""A collection of tests for :mod:`tenpy.models.site`."""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

import tenpy.linalg.np_conserved as npc
from tenpy.networks import site

from test_charges import gen_random_legcharge


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
        S = site.spin_half_site(conserve)
        S.test_sanity()
    npt.assert_equal((S.Sx + 1.j * S.Sy).to_ndarray(), S.Sp.to_ndarray())
    npt.assert_equal((S.Sx - 1.j * S.Sy).to_ndarray(), S.Sm.to_ndarray())
    for i in range(3):
        Sa, Sb, Sc = ([S.Sx, S.Sy, S.Sz] * 2)[i:i + 3]
        # for pauli matrices ``sigma_a . sigma_b = 1.j * epsilon_{a,b,c} sigma_c``
        # with ``Sa = 0.5 sigma_a``, we get ``Sa . Sb = 0.5 epsilon_{a,b,c} Sc``.
        npt.assert_equal(np.dot(Sa.to_ndarray(), Sb.to_ndarray()), 0.5j * Sc.to_ndarray())


def test_fermion_site():
    for conserve in ['N', 'parity', None]:
        S = site.fermion_site(conserve)
        S.test_sanity()
    npt.assert_equal(np.dot(S.Cd.to_ndarray(), S.C.to_ndarray()), S.N.to_ndarray())


def test_boson_site():
    for Nmax in [1, 2, 5, 10]:
        for conserve in ['N', 'parity', None]:
            S = site.boson_site(Nmax, conserve=conserve)
            S.test_sanity()
        npt.assert_array_almost_equal_nulp(
            np.dot(S.Bd.to_ndarray(), S.B.to_ndarray()), S.N.to_ndarray(), 2)
