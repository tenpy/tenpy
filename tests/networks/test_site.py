"""A collection of tests for :mod:`tenpy.models.site`."""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import itertools as it
import copy
import pytest

from tenpy import linalg as la
from tenpy.networks import site
from tenpy.tools.misc import inverse_permutation

from conftest import random_symmetry_sectors


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def anticommutator(A, B):
    return np.dot(A, B) + np.dot(B, A)


@pytest.mark.parametrize('symmetry_backend, use_sym',
                         [('abelian', True), ('abelian', False), ('no_symmetry', False)])
def test_site(np_random, symmetry_backend, block_backend, use_sym):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if use_sym:
        sym = la.u1_symmetry * la.z3_symmetry
    else:
        sym = la.no_symmetry
    dim = 8
    some_sectors = random_symmetry_sectors(sym, np_random, len_=dim, sort=False)
    leg = la.VectorSpace.from_basis(sym, np_random.choice(some_sectors, size=dim, replace=True))
    assert leg.dim == dim
    op1 = la.Tensor.random_uniform([leg, leg.dual], backend, labels=['p', 'p*'])
    labels = [f'x{i:d}' for i in range(10, 10 + dim)]
    s = site.Site(leg, backend=backend, state_labels=labels, silly_op=op1)
    assert s.state_index('x10') == 0
    assert s.state_index('x17') == 7
    assert s.symmetric_ops_names == {'silly_op', 'Id', 'JW'}
    assert s.all_ops_names == {'silly_op', 'Id', 'JW'}
    assert s.silly_op is op1
    assert s.get_op('silly_op') is op1
    op2 = la.Tensor.random_uniform([leg, leg.dual], backend, labels=['p', 'p*'])
    s.add_op('op2', op2)
    assert s.op2 is op2
    assert s.get_op('op2') is op2
    op3_dense = np.diag(np.arange(10, 10 + dim))
    s.add_op('op3', op3_dense)
    npt.assert_equal(s.get_op('op3').to_numpy_ndarray(), op3_dense)

    # TODO reintroduce when mini-language is revised
    # npt.assert_equal(
    #     s.get_op('silly_op op2').to_ndarray(),
    #     npc.tensordot(op1, op2, [1, 0]).to_ndarray())
    
    if use_sym:
        leg2 = leg.drop_symmetry(1)
        # TODO for some reason changing to a Z4 symmetry fails, saying the operators are not symmetric...?
        # leg2 = leg2.change_symmetry(symmetry=la.z4_symmetry, sector_map=lambda s: s % 4)
        leg2 = leg2.change_symmetry(symmetry=la.z3_symmetry, sector_map=lambda s: s % 3)
    else:
        leg2 = leg
    leg2.test_sanity()
    s2 = copy.deepcopy(s)
    s2.change_leg(leg2)
    for name in ['silly_op', 'op2', 'op3']:
        s_op = s.get_op(name).to_numpy_ndarray()
        s2_op = s2.get_op(name).to_numpy_ndarray()
        npt.assert_equal(s_op, s2_op)


@pytest.mark.xfail()  # TODO
def test_double_site():
    for site0, site1 in [[site.SpinHalfSite(None)] * 2,
                         [site.SpinHalfSite('Sz', sort_charge=False)] * 2]:
        for charges in ['same', 'drop', 'independent']:
            ds = site.GroupedSite([site0, site1], symmetry_combine=charges)
            ds.test_sanity()
    fs = site.FermionSite('N')
    ds = site.GroupedSite([fs, fs], ['a', 'b'], symmetry_combine='same')
    assert ds.need_JW_string == set([op + 'a' for op in fs.need_JW_string] +
                                    [op + 'b' for op in fs.need_JW_string] + ['JW'])
    ss = site.GroupedSite([fs])


def check_spin_algebra(s, SpSmSz=['Sp', 'Sm', 'Sz'], SxSy=['Sx', 'Sy']):
    """Test whether the spins operators behave as expected.

    `S` should be a :class:`site.Site`. Set `SxSy` to `None` to ignore Sx and Sy (if they don't
    exist as npc.Array due to conservation).
    """
    Sp, Sm, Sz = [s.get_op(name).to_numpy_ndarray() for name in SpSmSz]
    npt.assert_almost_equal(commutator(Sz, Sp), Sp, 13)
    npt.assert_almost_equal(commutator(Sz, Sm), -Sm, 13)
    if SxSy is not None:
        Sx, Sy = [s.get_op(name).to_numpy_ndarray() for name in SxSy]
        npt.assert_equal(Sx + 1.j * Sy, Sp)
        npt.assert_equal(Sx - 1.j * Sy, Sm)
        for i in range(3):
            Sa, Sb, Sc = ([Sx, Sy, Sz] * 2)[i:i + 3]
            npt.assert_almost_equal(commutator(Sa, Sb), 1.j * Sc, 13)
            if len(Sz) == 2:  # spin 1/2
                # for pauli matrices ``sigma_a . sigma_b = 1.j * epsilon_{a,b,c} sigma_c``
                # with ``Sa = 0.5 sigma_a``, we get ``Sa . Sb = 0.5j epsilon_{a,b,c} Sc``.
                npt.assert_almost_equal(np.dot(Sa, Sb), 0.5j * Sc, 13)  # holds only for S=1/2


def check_same_operators(sites):
    """check that the given sites have equivalent onsite operators (possibly with different symmetries)"""
    ops = {}
    for s in sites:
        for name, op in s.all_ops.items():
            op = op.to_numpy_ndarray()
            if name in ops:  # only as far as defined before
                npt.assert_equal(op, ops[name])
            else:
                ops[name] = op


def test_spin_half_site(block_backend, symmetry_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    hcs = dict(Id='Id', JW='JW', Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp', Sigmax='Sigmax',
               Sigmay='Sigmay', Sigmaz='Sigmaz')
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
    elif symmetry_backend == 'abelian':
        all_conserve = ['Sz', 'parity', 'None']
    elif symmetry_backend == 'nonabelian':
        all_conserve = ['SU(2)', 'Sz', 'parity', 'None']
        pytest.xfail('Nonabelian backend not ready')
    else:
        raise ValueError
    
    sites = []
    for conserve in all_conserve:
        print(f'checking {conserve=}')
        s = site.SpinHalfSite(conserve, backend=backend)
        assert la.almost_equal(s.Sp, s.Sm.hconj())
        s.test_sanity()
        for op in s.all_ops_names:
            assert s.hc_ops[op] == hcs[op]
        SxSy = ['Sx', 'Sy'] if conserve in ['parity', 'None'] else None  # TODO include Sz when ready
        check_spin_algebra(s, SxSy=SxSy)
        sites.append(s)
    check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_spin_site():
    hcs = dict(Id='Id', JW='JW', Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')
    for s in [0.5, 1, 1.5, 2, 5]:
        print('s = ', s)
        sites = []
        for sort_charge in [True, False]:
            for conserve in [None, 'Sz', 'parity']:
                print("conserve = ", conserve)
                S = site.SpinSite(s, conserve, sort_charge=sort_charge)
                S.test_sanity()
                for op in S.onsite_ops:
                    assert S.hc_ops[op] == hcs[op]
                if conserve != 'Sz':
                    SxSy = ['Sx', 'Sy']
                else:
                    SxSy = None
                check_spin_algebra(S, SxSy=SxSy)
                sites.append(S)
        check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_fermion_site():
    hcs = dict(Id='Id', JW='JW', C='Cd', Cd='C', N='N', dN='dN', dNdN='dNdN')
    sites = []
    for conserve in [None, 'N', 'parity']:
        S = site.FermionSite(conserve)
        S.test_sanity()
        for op in S.onsite_ops:
            assert S.hc_ops[op] == hcs[op]
        C, Cd, N = S.C.to_ndarray(), S.Cd.to_ndarray(), S.N.to_ndarray()
        Id = S.Id.to_ndarray()
        JW = S.JW.to_ndarray()
        npt.assert_equal(np.dot(Cd, C), N)
        npt.assert_equal(anticommutator(Cd, C), Id)
        npt.assert_equal(np.dot(Cd, C), N)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(C, JW), -np.dot(JW, C))
        assert S.need_JW_string == set(['Cd', 'C', 'JW'])
        for op in ['C', 'Cd', 'C N', 'C Cd C', 'C JW Cd']:
            assert S.op_needs_JW(op)
        for op in ['N', 'C Cd', 'C JW', 'JW C']:
            assert not S.op_needs_JW(op)
        sites.append(S)
    check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_spin_half_fermion_site():
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', NuNd='NuNd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    sites = []
    for cons_N, cons_Sz in it.product(['N', 'parity', None], ['Sz', 'parity', None]):
        print("conserve ", repr(cons_N), repr(cons_Sz))
        S = site.SpinHalfFermionSite(cons_N, cons_Sz)
        S.test_sanity()
        for op in S.onsite_ops:
            assert S.hc_ops[op] == hcs[op]
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
        check_spin_algebra(S, SxSy=SxSy)
        sites.append(S)
    check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_spin_half_hole_site():
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    sites = []
    for cons_N, cons_Sz in it.product(['N', 'parity', None], ['Sz', 'parity', None]):
        print("conserve ", repr(cons_N), repr(cons_Sz))
        S = site.SpinHalfHoleSite(cons_N, cons_Sz)
        S.test_sanity()
        for op in S.onsite_ops:
            assert S.hc_ops[op] == hcs[op]
        Id = S.Id.to_ndarray()
        JW = S.JW.to_ndarray()
        Cu, Cd = S.Cu.to_ndarray(), S.Cd.to_ndarray()
        Cdu, Cdd = S.Cdu.to_ndarray(), S.Cdd.to_ndarray()
        Nu, Nd, Ntot = S.Nu.to_ndarray(), S.Nd.to_ndarray(), S.Ntot.to_ndarray()
        npt.assert_equal(np.dot(Cdu, Cu), Nu)
        npt.assert_equal(np.dot(Cdd, Cd), Nd)
        npt.assert_equal(Nu + Nd, Ntot)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cu, JW), -np.dot(JW, Cu))
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(Cdu, JW), -np.dot(JW, Cdu))
        npt.assert_equal(np.dot(Cdd, JW), -np.dot(JW, Cdd))
        # anti-commute Cu with Cd
        npt.assert_equal(np.dot(Cu, Cd), -np.dot(Cd, Cu))
        npt.assert_equal(np.dot(Cdu, Cdd), -np.dot(Cdd, Cdu))
        if cons_Sz != 'Sz':
            SxSy = ['Sx', 'Sy']
        else:
            SxSy = None
        check_spin_algebra(S, SxSy=SxSy)
        sites.append(S)
    check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_boson_site():
    hcs = dict(Id='Id', JW='JW', B='Bd', Bd='B', N='N', NN='NN', dN='dN', dNdN='dNdN', P='P')
    for Nmax in [1, 2, 5, 10]:
        sites = []
        for conserve in ['N', 'parity', None]:
            S = site.BosonSite(Nmax, conserve=conserve)
            S.test_sanity()
            for op in S.onsite_ops:
                assert S.hc_ops[op] == hcs[op]
            npt.assert_array_almost_equal_nulp(np.dot(S.Bd.to_ndarray(), S.B.to_ndarray()),
                                               S.N.to_ndarray(), 2)
            sites.append(S)
        check_same_operators(sites)


@pytest.mark.parametrize('q', [2, 3, 5, 10])
def test_clock_site(block_backend, symmetry_backend, q):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
    elif symmetry_backend == 'abelian':
        all_conserve = ['Z', 'None']
    elif symmetry_backend == 'nonabelian':
        all_conserve = ['Z', 'None']
    else:
        raise ValueError
    hcs = dict(Id='Id', JW='JW', Xphc='Xphc', Zphc='Zphc')
    if q == 2:
        # X and Z are hermitian for q == 2
        hcs.update(X='X', Z='Z', Xhc='Xhc', Zhc='Zhc')
    else:
        hcs.update(X='Xhc', Z='Zhc', Xhc='X', Zhc='Z')
    sites = []
    for conserve in all_conserve:
        s = site.ClockSite(q=q, conserve=conserve, backend=backend)
        s.test_sanity()
        for op in s.all_ops_names:
            assert s.hc_ops[op] == hcs[op]

        # clock algebra
        w = np.exp(2.j * np.pi / q)
        X = s.X.to_numpy_ndarray()
        Z = s.Z.to_numpy_ndarray()
        # compute q-th powers
        Z_pow_q = Z
        X_pow_q = X
        for _ in range(q - 1):
            Z_pow_q = np.dot(Z_pow_q, Z)
            X_pow_q = np.dot(X_pow_q, X)
            
        npt.assert_array_almost_equal_nulp(np.dot(X, Z), w * np.dot(Z, X), 3 * q)
        npt.assert_array_almost_equal_nulp(X_pow_q, np.eye(q), 3 * q)
        npt.assert_array_almost_equal_nulp(Z_pow_q, np.eye(q), 3 * q)

        sites.append(s)
    check_same_operators(sites)


@pytest.mark.xfail()  # TODO
def test_set_common_charges():
    raise NotImplementedError
    # spin = site.SpinSite(0.5, 'Sz')
    # spin1 = site.SpinSite(1, 'Sz')
    # ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
    # boson = site.BosonSite(2, 'N')
    # spin_ops = {op_name: get_site_op_flat(spin, op_name) for op_name in spin.opnames}
    # spin1_ops = {op_name: get_site_op_flat(spin1, op_name) for op_name in spin1.opnames}
    # ferm_ops = {op_name: get_site_op_flat(ferm, op_name) for op_name in ferm.opnames}
    # boson_ops = {op_name: get_site_op_flat(boson, op_name) for op_name in boson.opnames}
    # site.set_common_charges([spin, ferm])
    # assert tuple(spin.leg.chinfo.names) == ('2*Sz', 'N')
    # spin.test_sanity()
    # ferm.test_sanity()
    # for op_name, op_flat in spin_ops.items():
    #     op_flat2 = get_site_op_flat(spin, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
    # for op_name, op_flat in ferm_ops.items():
    #     op_flat2 = get_site_op_flat(ferm, op_name)
    #     npt.assert_equal(op_flat, op_flat2)

    # spin = site.SpinSite(0.5, 'Sz')
    # ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
    # site.set_common_charges([ferm, spin], new_charges=[[(1, 0, '2*Sz'), (1, 1, '2*Sz')]])
    # assert tuple(ferm.leg.chinfo.names) == ('2*Sz', )
    # spin.test_sanity()
    # ferm.test_sanity()
    # for op_name, op_flat in spin_ops.items():
    #     op_flat2 = get_site_op_flat(spin, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
    # for op_name, op_flat in ferm_ops.items():
    #     op_flat2 = get_site_op_flat(ferm, op_name)
    #     npt.assert_equal(op_flat, op_flat2)

    # # and finally a few more, changing orders as well
    # ferm = site.SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
    # spin = site.SpinSite(0.5, 'Sz')
    # spin1 = site.SpinSite(1, 'Sz')
    # boson = site.BosonSite(2, 'N')

    # site.set_common_charges([ferm, spin1, spin, boson],
    #                         new_charges=[[(1, 0, '2*Sz'), (1, 2, '2*Sz')],
    #                                      [(2, 0, 'N'), (1, 3, 'N')], [(0.5, 1, '2*Sz')]],
    #                         new_names=['2*(Sz_f + Sz_spin-half)', '2*N_f+N_b', 'Sz_spin-1'])
    # assert tuple(ferm.leg.chinfo.names) == ('2*(Sz_f + Sz_spin-half)', '2*N_f+N_b', 'Sz_spin-1')
    # spin.test_sanity()
    # ferm.test_sanity()
    # spin1.test_sanity()
    # boson.test_sanity()
    # for op_name, op_flat in spin_ops.items():
    #     op_flat2 = get_site_op_flat(spin, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
    # for op_name, op_flat in ferm_ops.items():
    #     op_flat2 = get_site_op_flat(ferm, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
    # for op_name, op_flat in spin1_ops.items():
    #     op_flat2 = get_site_op_flat(spin1, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
    # for op_name, op_flat in boson_ops.items():
    #     op_flat2 = get_site_op_flat(boson, op_name)
    #     npt.assert_equal(op_flat, op_flat2)
