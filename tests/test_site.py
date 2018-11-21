"""A collection of tests for :mod:`tenpy.models.site`.  """
# Copyright 2018 TeNPy Developers

import numpy as np
import numpy.testing as npt
import nose.tools as nst
import itertools as it
import copy

import tenpy.linalg.np_conserved as npc
from tenpy.networks import site
from tenpy.tools.misc import inverse_permutation

from random_test import gen_random_legcharge


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def anticommutator(A, B):
    return np.dot(A, B) + np.dot(B, A)


def get_site_op_flat(site, op):
    """Like ``site.get_op(op)``, but return a flat numpy array and revert permutation from charges.

    site.perm should store the permutation compared to "conserve=None", so we can use that
    to convert to the "standard" flat form with conserve=None.
    """
    op = site.get_op(op).to_ndarray()
    iperm = inverse_permutation(site.perm)
    return op[np.ix_(iperm, iperm)]


def test_site():
    chinfo = npc.ChargeInfo([1, 3])
    leg = gen_random_legcharge(chinfo, 8)
    op1 = npc.Array.from_func(np.random.random, [leg, leg.conj()], shape_kw='size')
    op2 = npc.Array.from_func(np.random.random, [leg, leg.conj()], shape_kw='size')
    labels = ['up'] + [None] * (leg.ind_len - 2) + ['down']
    s = site.Site(leg, labels, silly_op=op1)
    nst.eq_(s.state_index('up'), 0)
    nst.eq_(s.state_index('down'), leg.ind_len - 1)
    nst.eq_(s.opnames, set(['silly_op', 'Id', 'JW']))
    assert (s.silly_op is op1)
    s.add_op('op2', op2)
    assert (s.op2 is op2)
    assert (s.get_op('op2') is op2)
    assert (s.get_op('silly_op') is op1)
    npt.assert_equal(
        s.get_op('silly_op op2').to_ndarray(),
        npc.tensordot(op1, op2, [1, 0]).to_ndarray())
    leg2 = npc.LegCharge.from_drop_charge(leg, 1)
    leg2 = npc.LegCharge.from_change_charge(leg2, 0, 2, 'changed')
    s2 = copy.deepcopy(s)
    s2.change_charge(leg2)
    perm_qind, leg2s = leg2.sort()
    perm_flat = leg2.perm_flat_from_perm_qind(perm_qind)
    s2s = copy.deepcopy(s2)
    s2s.change_charge(leg2s, perm_flat)
    for site_check in [s2, s2s]:
        print("site_check.leg = ", site_check.leg)
        for opn in site_check.opnames:
            op1 = s.get_op(opn).to_ndarray()
            op2 = site_check.get_op(opn).to_ndarray()
            perm = site_check.perm
            npt.assert_equal(op1[np.ix_(perm, perm)], op2)
    # done


def test_double_site():
    for site0, site1 in [[site.SpinHalfSite(None)] * 2, [site.SpinHalfSite('Sz')] * 2]:
        for charges in ['same', 'drop', 'independent']:
            ds = site.GroupedSite([site0, site1], charges=charges)
            ds.test_sanity()
    fs = site.FermionSite('N')
    ds = site.GroupedSite([fs, fs], ['a', 'b'], charges='same')
    assert ds.need_JW_string == set(
        [op + 'a' for op in fs.need_JW_string] + [op + 'b' for op in fs.need_JW_string])
    ss = site.GroupedSite([fs])


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
            npt.assert_almost_equal(commutator(Sa, Sb), 1.j * Sc, 13)
            if S == 0.5:
                # for pauli matrices ``sigma_a . sigma_b = 1.j * epsilon_{a,b,c} sigma_c``
                # with ``Sa = 0.5 sigma_a``, we get ``Sa . Sb = 0.5j epsilon_{a,b,c} Sc``.
                npt.assert_almost_equal(np.dot(Sa, Sb), 0.5j*Sc, 13) # holds only for S=1/2

def check_same_operators(sites):
    """check that the given sites have the same onsite-operator using get_site_op_flat"""
    ops = {}
    for s in sites:
        for op_name in s.opnames:
            op = get_site_op_flat(s, op_name)
            if op_name in ops:  # only as far as defined before
                npt.assert_equal(op, ops[op_name])
            else:
                ops[op_name] = op
    # done


def test_spin_half_site_checks():
    for conserve in [None, 'Sz', 'parity']:
        S = site.SpinHalfSite(conserve)
        S.test_sanity()
        if conserve != 'Sz':
            SxSy = ['Sx', 'Sy']
        else:
            SxSy = None
        check_spin_site(S, SxSy=SxSy)


def test_spin_half_site():
    sites = []
    for conserve in [None, 'Sz', 'parity']:
        S = site.SpinHalfSite(conserve)
        sites.append(S)
    yield check_same_operators, sites


def test_spin_site():
    for s in [0.5, 1, 1.5, 2, 5]:
        print('s = ', s)
        for conserve in [None, 'Sz', 'parity']:
            print("conserve = ", conserve)
            S = site.SpinSite(s, conserve)
            S.test_sanity()
            if conserve != 'Sz':
                SxSy = ['Sx', 'Sy']
            else:
                SxSy = None
            check_spin_site(S, SxSy=SxSy)


def test_spin_site_ops():
    for s in [0.5, 1, 1.5, 2, 5]:
        sites = []
        for conserve in [None, 'Sz', 'parity']:
            S = site.SpinSite(s, conserve)
            sites.append(S)
        yield check_same_operators, sites


def test_fermion_site():
    for conserve in [None, 'N', 'parity']:
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
        assert S.need_JW_string == set(['Cd', 'C'])
        for op in ['C', 'Cd', 'C N', 'C Cd C']:
            assert S.op_needs_JW(op)
        for op in ['N', 'C Cd', 'JW']:
            assert not S.op_needs_JW(op)


def test_fermion_site_ops():
    sites = []
    for conserve in [None, 'N', 'parity']:
        S = site.FermionSite(conserve)
        sites.append(S)
    yield check_same_operators, sites


def test_spin_half_fermion_site():
    for cons_N, cons_Sz in it.product(['N', 'parity', None], ['Sz', 'parity', None]):
        print("conserve ", repr(cons_N), repr(cons_Sz))
        S = site.SpinHalfFermionSite(cons_N, cons_Sz)
        S.test_sanity()
        Id = S.Id.to_ndarray()
        JW = S.JW.to_ndarray()
        Cu, Cd = S.Cu.to_ndarray(), S.Cd.to_ndarray()
        Cdu, Cdd = S.Cdu.to_ndarray(), S.Cdd.to_ndarray()
        Nu, Nd, Ntot = S.Nu.to_ndarray(), S.Nd.to_ndarray(), S.Ntot.to_ndarray()
        npt.assert_equal(np.dot(Cdu, Cu), Nu)
        npt.assert_equal(np.dot(Cdd, Cd), Nd)
        npt.assert_equal(Nu + Nd, Ntot)
        npt.assert_equal(np.dot(Nu, Nd), S.NuNd.to_ndarray())
        npt.assert_equal(anticommutator(Cdu, Cu), Id)
        npt.assert_equal(anticommutator(Cdd, Cd), Id)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cu, JW), -np.dot(JW, Cu))
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(Cdu, JW), -np.dot(JW, Cdu))
        npt.assert_equal(np.dot(Cdd, JW), -np.dot(JW, Cdd))
        # anti-commute Cu with Cd
        npt.assert_equal(np.dot(Cu, Cd), -np.dot(Cd, Cu))
        npt.assert_equal(np.dot(Cu, Cdd), -np.dot(Cdd, Cu))
        npt.assert_equal(np.dot(Cdu, Cd), -np.dot(Cd, Cdu))
        npt.assert_equal(np.dot(Cdu, Cdd), -np.dot(Cdd, Cdu))
        if cons_Sz != 'Sz':
            SxSy = ['Sx', 'Sy']
        else:
            SxSy = None
        check_spin_site(S, SxSy=SxSy)


def test_spin_half_fermion_site_ops():
    sites = []
    for cons_n, cons_sz in it.product(['N', 'parity', None], ['Sz', 'parity', None]):
        S = site.SpinHalfFermionSite(cons_n, cons_sz)
        sites.append(S)
    yield check_same_operators, sites


def test_boson_site():
    for Nmax in [1, 2, 5, 10]:
        for conserve in ['N', 'parity', None]:
            S = site.BosonSite(Nmax, conserve=conserve)
            S.test_sanity()
        npt.assert_array_almost_equal_nulp(
            np.dot(S.Bd.to_ndarray(), S.B.to_ndarray()), S.N.to_ndarray(), 2)


def test_boson_site_ops():
    for Nmax in [1, 2, 5, 10]:
        sites = []
        for conserve in ['N', 'parity', None]:
            S = site.BosonSite(Nmax, conserve=conserve)
        yield check_same_operators, sites


def test_multi_sites_combine_charges():
   spin = site.SpinSite(0.5, 'Sz')
   spin1 = site.SpinSite(1, 'Sz')
   ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
   spin_ops = {op_name: get_site_op_flat(spin, op_name) for op_name in spin.opnames}
   spin1_ops = {op_name: get_site_op_flat(spin1, op_name) for op_name in spin1.opnames}
   ferm_ops = {op_name: get_site_op_flat(ferm, op_name) for op_name in ferm.opnames}
   site.multi_sites_combine_charges([spin, ferm])
   assert tuple(spin.leg.chinfo.names) == ('2*Sz', 'N', 'Sz')
   spin.test_sanity()
   ferm.test_sanity()
   for op_name, op_flat in spin_ops.items():
       op_flat2 = get_site_op_flat(spin, op_name)
       npt.assert_equal(op_flat, op_flat2)
   for op_name, op_flat in ferm_ops.items():
       op_flat2 = get_site_op_flat(ferm, op_name)
       npt.assert_equal(op_flat, op_flat2)

   spin = site.SpinSite(0.5, 'Sz')
   ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
   site.multi_sites_combine_charges([ferm, spin], same_charges=[[(0, 'Sz'), (1, '2*Sz')]])
   assert tuple(ferm.leg.chinfo.names) == ('N', 'Sz')
   spin.test_sanity()
   ferm.test_sanity()
   for op_name, op_flat in spin_ops.items():
       op_flat2 = get_site_op_flat(spin, op_name)
       npt.assert_equal(op_flat, op_flat2)
   for op_name, op_flat in ferm_ops.items():
       op_flat2 = get_site_op_flat(ferm, op_name)
       npt.assert_equal(op_flat, op_flat2)

   # and finally a few more, changing orders as well
   ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
   spin = site.SpinSite(0.5, 'Sz')
   spin1 = site.SpinSite(1, 'Sz')
   site.multi_sites_combine_charges([ferm, spin1, spin], same_charges=[[(0, 'Sz'), (2, '2*Sz')]])
   assert tuple(ferm.leg.chinfo.names) == ('N', 'Sz', '2*Sz')
   spin.test_sanity()
   ferm.test_sanity()
   spin1.test_sanity()
   for op_name, op_flat in spin_ops.items():
       op_flat2 = get_site_op_flat(spin, op_name)
       npt.assert_equal(op_flat, op_flat2)
   for op_name, op_flat in ferm_ops.items():
       op_flat2 = get_site_op_flat(ferm, op_name)
       npt.assert_equal(op_flat, op_flat2)
   for op_name, op_flat in spin1_ops.items():
       op_flat2 = get_site_op_flat(spin1, op_name)
       npt.assert_equal(op_flat, op_flat2)
