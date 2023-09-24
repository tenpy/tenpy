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
    #     s.get_op('silly_op op2').to_numpy_ndarray(),
    #     npc.tensordot(op1, op2, [1, 0]).to_numpy_ndarray())
    
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


def test_double_site(symmetry_backend, block_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
        fs_conserve = 'None'
    elif symmetry_backend in ['abelian', 'nonabelian']:
        all_conserve = ['Sz', 'None']
        fs_conserve = 'N'
    else:
        raise ValueError
    
    for conserve in all_conserve:
        site0 = site1 = site.SpinHalfSite(conserve, backend=backend)
        for symmetry_combine in ['same', 'drop', 'independent']:
            print(f'{symmetry_combine=}')
            ds = site.GroupedSite([site0, site1], symmetry_combine=symmetry_combine)
            ds.test_sanity()
    fs = site.FermionSite(fs_conserve, backend=backend)
    ds = site.GroupedSite([fs, fs], ['a', 'b'], symmetry_combine='same')
    assert ds.need_JW_string == set([op + 'a' for op in fs.need_JW_string] +
                                    [op + 'b' for op in fs.need_JW_string] + ['JW'])
    _ = site.GroupedSite([fs])


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


@pytest.mark.parametrize('S', [0.5, 1, 1.5, 2, 5])
def test_spin_site(block_backend, symmetry_backend, S):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    hcs = dict(Id='Id', JW='JW', Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')
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
        print("conserve = ", conserve)
        s = site.SpinSite(S, conserve, backend=backend)
        s.test_sanity()
        for op in s.all_ops_names:
            assert s.hc_ops[op] == hcs[op]
        SxSy = ['Sx', 'Sy'] if conserve in ['parity', 'None'] else None  # TODO include Sz when ready
        check_spin_algebra(s, SxSy=SxSy)
        sites.append(s)
    check_same_operators(sites)


def test_fermion_site(block_backend, symmetry_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    hcs = dict(Id='Id', JW='JW', C='Cd', Cd='C', N='N', dN='dN', dNdN='dNdN')
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
    elif symmetry_backend == 'abelian':
        all_conserve = ['N', 'parity', 'None']
    elif symmetry_backend == 'nonabelian':
        # TODO check JW-free fermions when ready?
        pytest.xfail('Nonabelian backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve in all_conserve:
        s = site.FermionSite(conserve, backend=backend)
        s.test_sanity()
        for op in s.all_ops_names:
            assert s.hc_ops[op] == hcs[op]
        C, Cd, N = s.C.to_numpy_ndarray(), s.Cd.to_numpy_ndarray(), s.N.to_numpy_ndarray()
        Id, JW = s.Id.to_numpy_ndarray(), s.JW.to_numpy_ndarray()
        npt.assert_equal(np.dot(Cd, C), N)
        npt.assert_equal(anticommutator(Cd, C), Id)
        npt.assert_equal(np.dot(Cd, C), N)
        # anti-commutate with Jordan-Wigner
        npt.assert_equal(np.dot(Cd, JW), -np.dot(JW, Cd))
        npt.assert_equal(np.dot(C, JW), -np.dot(JW, C))
        assert s.need_JW_string == set(['Cd', 'C', 'JW'])
        for op in ['C', 'Cd']:  #  TODO reinstate with minilanguage: ['C', 'Cd', 'C N', 'C Cd C', 'C JW Cd']:
            assert s.op_needs_JW(op)
        for op in ['N']:  #  TODO reinstate with minilanguage: ['N', 'C Cd', 'C JW', 'JW C']:
            assert not s.op_needs_JW(op)
        sites.append(s)
    check_same_operators(sites)


def test_spin_half_fermion_site(block_backend, symmetry_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', NuNd='NuNd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    if symmetry_backend == 'no_symmetry':
        all_conserve_N = all_conserve_S = ['None']
    elif symmetry_backend == 'abelian':
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Sz', 'parity', 'None']
    elif symmetry_backend == 'nonabelian':
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Stot', 'Sz', 'parity', 'None']
        pytest.xfail('Nonabelian backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        print(f'{conserve_N=}, {conserve_S=}')
        S = site.SpinHalfFermionSite(conserve_N, conserve_S, backend=backend)
        S.test_sanity()
        for op in S.all_ops_names:
            assert S.hc_ops[op] == hcs[op]
        Id, JW = S.Id.to_numpy_ndarray(), S.JW.to_numpy_ndarray()
        Cu, Cd = S.Cu.to_numpy_ndarray(), S.Cd.to_numpy_ndarray()
        Cdu, Cdd = S.Cdu.to_numpy_ndarray(), S.Cdd.to_numpy_ndarray()
        Nu, Nd, Ntot = S.Nu.to_numpy_ndarray(), S.Nd.to_numpy_ndarray(), S.Ntot.to_numpy_ndarray()
        npt.assert_equal(np.dot(Cdu, Cu), Nu)
        npt.assert_equal(np.dot(Cdd, Cd), Nd)
        npt.assert_equal(Nu + Nd, Ntot)
        npt.assert_equal(np.dot(Nu, Nd), S.NuNd.to_numpy_ndarray())
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
        SxSy = ['Sx', 'Sy'] if conserve_S in ['parity', 'None'] else None  # TODO include Sz when ready
        check_spin_algebra(S, SxSy=SxSy)
        sites.append(S)
    check_same_operators(sites)


def test_spin_half_hole_site(block_backend, symmetry_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    if symmetry_backend == 'no_symmetry':
        all_conserve_N = all_conserve_S = ['None']
    elif symmetry_backend == 'abelian':
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Sz', 'parity', 'None']
    elif symmetry_backend == 'nonabelian':
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Stot', 'Sz', 'parity', 'None']
        pytest.xfail('Nonabelian backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        print(f'{conserve_N=}, {conserve_S=}')
        S = site.SpinHalfHoleSite(conserve_N, conserve_S, backend=backend)
        S.test_sanity()
        for op in S.all_ops_names:
            assert S.hc_ops[op] == hcs[op]
        JW = S.JW.to_numpy_ndarray()
        Cu, Cd = S.Cu.to_numpy_ndarray(), S.Cd.to_numpy_ndarray()
        Cdu, Cdd = S.Cdu.to_numpy_ndarray(), S.Cdd.to_numpy_ndarray()
        Nu, Nd, Ntot = S.Nu.to_numpy_ndarray(), S.Nd.to_numpy_ndarray(), S.Ntot.to_numpy_ndarray()
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
        SxSy = ['Sx', 'Sy'] if conserve_S in ['parity', 'None'] else None  # TODO include Sz when ready
        check_spin_algebra(S, SxSy=SxSy)
        sites.append(S)
    check_same_operators(sites)


@pytest.mark.parametrize('Nmax', [1, 2, 5, 10])
def test_boson_site(block_backend, symmetry_backend, Nmax):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
    elif symmetry_backend in ['abelian', 'nonabelian']:
        all_conserve = ['N', 'parity', 'None']
    else:
        raise ValueError
    hcs = dict(Id='Id', JW='JW', B='Bd', Bd='B', N='N', NN='NN', dN='dN', dNdN='dNdN', P='P')
    sites = []
    for conserve in all_conserve:
        s = site.BosonSite(Nmax, conserve=conserve, backend=backend)
        s.test_sanity()
        for op in s.all_ops_names:
            assert s.hc_ops[op] == hcs[op]
        B = s.B.to_numpy_ndarray()
        Bd = s.Bd.to_numpy_ndarray()
        N = s.N.to_numpy_ndarray()
        npt.assert_array_almost_equal_nulp(np.dot(Bd, B), N, 2)
        expect_commutator = np.eye(s.dim)
        expect_commutator[-1, -1] = -Nmax  # commutation relation violated due to truncated space
        npt.assert_array_almost_equal_nulp(commutator(B, Bd), expect_commutator, 20)
        sites.append(s)
    check_same_operators(sites)


@pytest.mark.parametrize('q', [2, 3, 5, 10])
def test_clock_site(block_backend, symmetry_backend, q):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if symmetry_backend == 'no_symmetry':
        all_conserve = ['None']
    elif symmetry_backend in ['abelian', 'nonabelian']:
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


def test_set_common_symmetry(block_backend, symmetry_backend):
    backend = la.get_backend(block_backend=block_backend, symmetry_backend=symmetry_backend)
    if symmetry_backend == 'no_symmetry':
        conserve_S = conserve_N = 'None'
        expect_symm_Sz = expect_symm_Sz_N = expect_symm_Sz_N_Sz = la.no_symmetry
    else:
        conserve_S = 'Sz'
        conserve_N = 'N'
        expect_symm_Sz = la.U1Symmetry('2*Sz')
        expect_symm_Sz_N = la.U1Symmetry('2*Sz') * la.U1Symmetry('N')
        expect_symm_Sz_N_Sz = la.U1Symmetry('2*Sz') * la.U1Symmetry('N') * la.U1Symmetry('2*Sz')
    
    conserve_S = 'None' if symmetry_backend == 'no_symmetry' else 'Sz'
    conserve_N = 'None' if symmetry_backend == 'no_symmetry' else 'N'
    
    spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=backend)
    spin1 = site.SpinSite(S=1, conserve=conserve_S, backend=backend)
    ferm = site.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S)
    boson = site.BosonSite(Nmax=2, conserve=conserve_N, backend=backend)
    spin_ops = {name: op.to_numpy_ndarray() for name, op in spin.all_ops.items()}
    spin1_ops = {name: op.to_numpy_ndarray() for name, op in spin1.all_ops.items()}
    ferm_ops = {name: op.to_numpy_ndarray() for name, op in ferm.all_ops.items()}
    boson_ops = {name: op.to_numpy_ndarray() for name, op in boson.all_ops.items()}

    for symmetry_combine, expect_symm in [('by_name', expect_symm_Sz_N),
                                          ('independent', expect_symm_Sz_N_Sz),
                                          ('drop', la.no_symmetry)]:
        print(f'{symmetry_combine=}')
        site.set_common_symmetry([spin, ferm], symmetry_combine=symmetry_combine)
        assert spin.leg.symmetry == expect_symm
        assert ferm.leg.symmetry == expect_symm
        spin.test_sanity()
        ferm.test_sanity()
        for op_name, op_np in spin_ops.items():
            op_np2 = spin.get_op(op_name).to_numpy_ndarray()
            npt.assert_equal(op_np, op_np2)
        for op_name, op_np in ferm_ops.items():
            op_np2 = ferm.get_op(op_name).to_numpy_ndarray()
            npt.assert_equal(op_np, op_np2)

        # reset the modified sites
        spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=backend)
        ferm = site.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S)

    print('symmetry_combine: function')
    # drop the ferm N symmetry and conserve the total 2*Sz
    site.set_common_symmetry([ferm, spin], symmetry_combine=lambda i, s: s[:, :1],
                             new_symmetry=spin.symmetry)
    assert ferm.symmetry == expect_symm_Sz
    assert spin.symmetry == expect_symm_Sz
    ferm.test_sanity()
    spin.test_sanity()
    for op_name, op_np in spin_ops.items():
        op_np2 = spin.get_op(op_name).to_numpy_ndarray()
        npt.assert_equal(op_np, op_np2, err_msg=f'{op_name=}')
    for op_name, op_np in ferm_ops.items():
        op_np2 = ferm.get_op(op_name).to_numpy_ndarray()
        npt.assert_equal(op_np, op_np2, err_msg=f'{op_name=}')

    # reset the modified sites
    spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=backend)
    ferm = site.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S)

    if symmetry_backend == 'no_symmetry':
        return  # the last test really only makes sense with non-trivial symmetries
    
    print('symmetry_combine: list[list[tuple]]')
    sites = [ferm, spin1, spin, boson]
    # [(prefactor, site_idx, sector_col_idx)]
    spin_half_Sz = [(1, 0, ferm.symmetry.factor_where('2*Sz')), (1, 2, 0)]
    weird_occupation = [(2, 0, ferm.symmetry.factor_where('N')), (1, 3, 0)]
    spin_one_Sz = [(0.5, 1, 0)]
    new_symmetry = (la.U1Symmetry('2*(Sz_f + Sz_spin-half)') * la.U1Symmetry('2*N_f+N_b')
                    * la.U1Symmetry('Sz_spin-1'))
    site.set_common_symmetry(sites, [spin_half_Sz, weird_occupation, spin_one_Sz], new_symmetry)
    for name, s, expect_ops in zip(['ferm', 'spin1', 'spin', 'boson'],
                                      [ferm, spin1, spin, boson],
                                      [ferm_ops, spin1_ops, spin_ops, boson_ops]):
        s.test_sanity()
        assert s.symmetry == new_symmetry
        for op_name, op_np in expect_ops.items():
            op_np2 = s.get_op(op_name).to_numpy_ndarray()
            npt.assert_equal(op_np, op_np2, err_msg=f'{name=} {op_name=}')
