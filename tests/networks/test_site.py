"""A collection of tests for :mod:`tenpy.models.site`."""
# Copyright (C) TeNPy Developers, GNU GPLv3


import numpy as np
import numpy.testing as npt
import itertools as it
import copy
import pytest

from tenpy import linalg as la, backends
from tenpy.networks import site

from conftest import random_symmetry_sectors



pytest.skip("site not yet revised", allow_module_level=True)  # TODO


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def anticommutator(A, B):
    return np.dot(A, B) + np.dot(B, A)


@pytest.mark.parametrize('symmetry_backend, use_sym',
                         [('abelian', True), ('abelian', False), ('no_symmetry', False)])
def test_site(np_random, block_backend, symmetry_backend, use_sym):
    backend = la.get_backend(block_backend=block_backend, symmetry=symmetry_backend)
    if use_sym:
        sym = la.u1_symmetry * la.z3_symmetry
    else:
        sym = la.no_symmetry
    dim = 8
    some_sectors = random_symmetry_sectors(sym, num=dim, sort=False, np_random=np_random)
    leg = la.ElementarySpace.from_basis(sym, np_random.choice(some_sectors, size=dim, replace=True))
    assert leg.dim == dim
    op1 = la.SymmetricTensor.random_uniform([leg, leg.dual], backend, labels=['p', 'p*'])
    labels = [f'x{i:d}' for i in range(10, 10 + dim)]
    s = site.Site(leg, backend=backend, state_labels=labels)
    s.add_symmetric_operator('silly_op', op1)
    assert s.state_index('x10') == 0
    assert s.state_index('x17') == 7
    assert s.symmetric_op_names == {'silly_op', 'Id', 'JW'}
    assert s.all_op_names == {'silly_op', 'Id', 'JW'}
    assert s['silly_op'] is op1
    assert s.get_op('silly_op') is op1
    op2 = la.SymmetricTensor.random_uniform([leg, leg.dual], backend, labels=['p', 'p*'])
    s.add_symmetric_operator('op2', op2)
    assert s['op2'] is op2
    assert s.get_op('op2') is op2
    op3_dense = np.diag(np.arange(10, 10 + dim))
    s.add_symmetric_operator('op3', op3_dense)
    assert isinstance(s.get_op('op3'), la.DiagonalTensor)
    npt.assert_equal(s.get_op('op3').to_numpy(), op3_dense)
    npt.assert_equal(s['op3'].to_numpy(), op3_dense)

    # TODO reintroduce when mini-language is revised
    # npt.assert_equal(
    #     s.get_op('silly_op op2').to_numpy(),
    #     npc.tensordot(op1, op2, [1, 0]).to_numpy())
    
    if use_sym:
        leg2 = leg.drop_symmetry(1)
        leg2 = leg2.change_symmetry(symmetry=la.z3_symmetry, sector_map=lambda s: s % 3)
    else:
        leg2 = leg
    leg2.test_sanity()
    s2 = copy.deepcopy(s)
    s2.change_leg(leg2)
    for name in ['silly_op', 'op2', 'op3']:
        s_op = s.get_op(name).to_numpy()
        s2_op = s2.get_op(name).to_numpy()
        npt.assert_equal(s_op, s2_op)


def test_double_site(any_backend):
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
        fs_conserve = 'None'
    elif isinstance(any_backend, (la.AbelianBackend, la.FusionTreeBackend)):
        all_conserve = ['Sz', 'None']
        fs_conserve = 'N'
    else:
        raise ValueError

    # the order of the basis is determined by forming the ProductSpace and is sorted by sector.
    # the two states with
    
    for conserve in all_conserve:
        print(f'{conserve=}')

        # forming the ProductSpace potentially re-orders the basis, sorting it by sector
        # TODO should we change this behavior? at least document it...
        if conserve == 'None':
            # all sectors are [0] -> same order [up, down] as for original sites, c-style
            expect_labels = ['up_0 up_1', 'up_0 down_1', 'down_0 up_1', 'down_0 down_1']
        elif conserve == 'Sz':
            # we have sorted sectors of site0.sectors == [-1, 1], i.e. [down, up].
            # The c-style product gives [(down, down), (down, up), (up, down), (up, up)]
            # This would then be sorted, but it is already.
            expect_labels = ['down_0 down_1', 'down_0 up_1', 'up_0 down_1', 'up_0 up_1']
        else:
            raise NotImplementedError

        if isinstance(any_backend, backends.FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
                site0 = site1 = site.SpinHalfSite(conserve, backend=any_backend)
            return  # TODO
        
        site0 = site1 = site.SpinHalfSite(conserve, backend=any_backend)
        for symmetry_combine in ['same', 'drop', 'independent']:
            print(f'  {symmetry_combine=}')
            ds = site.GroupedSite([site0, site1], symmetry_combine=symmetry_combine)
            ds.test_sanity()
            print([ds.state_labels[l] for l in expect_labels])
            for idx, label in enumerate(expect_labels):
                assert ds.state_labels[label] == idx
        
    fs = site.FermionSite(fs_conserve, backend=any_backend)
    ds = site.GroupedSite([fs, fs], ['a', 'b'], symmetry_combine='same')
    assert ds.need_JW_string == set([op + 'a' for op in fs.need_JW_string] +
                                    [op + 'b' for op in fs.need_JW_string] + ['JW'])
    _ = site.GroupedSite([fs])


def check_spin_algebra(s, SpSmSz=['Sp', 'Sm', 'Sz'], SxSy=['Sx', 'Sy']):
    """Test whether the spins operators behave as expected.

    `S` should be a :class:`site.Site`. Set `SxSy` to `None` to ignore Sx and Sy.
    """
    Sp, Sm, Sz = [s.get_op(name).to_numpy() for name in SpSmSz]
    npt.assert_almost_equal(commutator(Sz, Sp), Sp, 13)
    npt.assert_almost_equal(commutator(Sz, Sm), -Sm, 13)
    if SxSy is not None:
        Sx, Sy = [s.get_op(name).to_numpy() for name in SxSy]
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
    """Check that the given sites have equivalent operators.

    If operators with matching names exist, we check if they have the same dense array representation.
    If an operator is missing on some of the sites, that is ok.
    """
    symmetric_ops = {}
    charged_ops = {}
    for s in sites:
        for name, op in s.symmetric_ops.items():
            op = op.to_numpy()
            if name in symmetric_ops:  # only as far as defined before
                npt.assert_equal(op, symmetric_ops[name])
            else:
                symmetric_ops[name] = op
        for name, op in s.charged_ops.items():
            op_L = op.op_L.to_numpy()
            op_R = op.op_R.to_numpy()
            if name in charged_ops:
                L, R = charged_ops[name]
                npt.assert_equal(op_L, L)
                npt.assert_equal(op_R, R)
            else:
                charged_ops[name] = (op_L, op_R)


def check_operator_availability(s: site.Site, expect_symmetric_ops: dict[str, bool],
                                expect_charged_ops: dict[str, tuple[bool, int]]):
    """Check if the operators on a site are as expected.

    We check if the available operators

    Parameters
    """
    expected_symmetric_names = set(expect_symmetric_ops.keys()) | {'Id', 'JW'}
    assert s.symmetric_op_names == expected_symmetric_names
    for name, is_diag in expect_symmetric_ops.items():
        assert name in s.symmetric_ops
        assert isinstance(s.symmetric_ops[name], la.DiagonalTensor) == is_diag
    for name, (can_use_alone, dim) in expect_charged_ops.items():
        assert name in s.charged_ops
        op = s.charged_ops[name]
        assert op.can_use_alone == can_use_alone
        # TODO revise this. purge the "dummy" language, its now "charged"
        assert op.op_L.dummy_leg.dim == dim
        assert op.op_R.dummy_leg.dim == dim


def test_spin_half_site(any_backend):
    hcs = dict(Id='Id', JW='JW', Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp', Sigmax='Sigmax',
               Sigmay='Sigmay', Sigmaz='Sigmaz')
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, la.AbelianBackend):
        all_conserve = ['Sz', 'parity', 'None']
    elif isinstance(any_backend, la.FusionTreeBackend):
        all_conserve = ['Stot', 'Sz', 'parity', 'None']
        pytest.xfail('FusionTree backend not ready')
    else:
        raise ValueError
    
    sites = []
    for conserve in all_conserve:
        print(f'checking {conserve=}')
        s = site.SpinHalfSite(conserve, backend=any_backend)
        assert la.almost_equal(s['Sp'], s['Sm'].hconj())
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        # TODO when Sx and Sy are implemented also for conserve == 'Sz', include
        SxSy = ['Sx', 'Sy'] if conserve in ['parity', 'None'] else None
        check_spin_algebra(s, SxSy=SxSy)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True)
        expect_charged_ops = {}  # dict(Svec=3)  # TODO
        if conserve in ['Sz', 'parity', 'None']:
            expect_symmetric_ops.update(Sz=True, Sigmaz=True)
            expect_charged_ops.update(Sp=(1, True), Sm=(1, True))
        if conserve in ['parity', 'None']:
            expect_charged_ops.update(Sx=(1, True), Sy=(1, True), Sigmax=(1, True), Sigmay=(1, True))
        if conserve == 'None':
            expect_symmetric_ops.update(Sx=False, Sy=False, Sigmax=False, Sigmay=False, Sp=False,
                                        Sm=False)
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


@pytest.mark.parametrize('S', [0.5, 1, 1.5, 2, 5])
def test_spin_site(any_backend, S):
    hcs = dict(Id='Id', JW='JW', Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, la.AbelianBackend):
        all_conserve = ['Sz', 'parity', 'None']
    elif isinstance(any_backend, la.FusionTreeBackend):
        all_conserve = ['SU(2)', 'Sz', 'parity', 'None']
        pytest.xfail('FusionTree backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve in all_conserve:
        print("conserve = ", conserve)
        s = site.SpinSite(S, conserve, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        # TODO when Sx and Sy are implemented also for conserve in ['Sz', 'parity'], include
        SxSy = ['Sx', 'Sy'] if conserve == 'None' else None
        check_spin_algebra(s, SxSy=SxSy)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True)
        expect_charged_ops = {}  # dict(Svec=3)  # TODO
        if conserve in ['Sz', 'parity', 'None']:
            expect_symmetric_ops.update(Sz=True)
            expect_charged_ops.update(Sp=(1, True), Sm=(1, True))
        if conserve == 'None':
            expect_symmetric_ops.update(Sx=False, Sy=False, Sp=False, Sm=False)
            expect_charged_ops.update(Sx=(1, True), Sy=(1, True))
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


def test_fermion_site(any_backend):
    hcs = dict(Id='Id', JW='JW', C='Cd', Cd='C', N='N', dN='dN', dNdN='dNdN')
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, la.AbelianBackend):
        all_conserve = ['N', 'parity', 'None']
    elif isinstance(any_backend, la.FusionTreeBackend):
        # TODO check JW-free fermions when ready?
        pytest.xfail('FusionTree backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve in all_conserve:
        s = site.FermionSite(conserve, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        C, Cd, N = s['C'].to_numpy(), s['Cd'].to_numpy(), s['N'].to_numpy()
        Id, JW = s.Id.to_numpy(), s.JW.to_numpy()
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
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True, N=True, dN=True, dNdN=True)
        expect_charged_ops = dict(C=(1, True), Cd=(1, True))  # dict(Svec=3)  # TODO
        if conserve == 'None':
            expect_symmetric_ops.update(C=False, Cd=False)
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


def test_spin_half_fermion_site(any_backend):
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', NuNd='NuNd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve_N = all_conserve_S = ['None']
    elif isinstance(any_backend, la.AbelianBackend):
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Sz', 'parity', 'None']
    elif isinstance(any_backend, la.FusionTreeBackend):
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Stot', 'Sz', 'parity', 'None']
        pytest.xfail('FusionTree backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        print(f'{conserve_N=}, {conserve_S=}')
        s = site.SpinHalfFermionSite(conserve_N, conserve_S, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        Id, JW = s.Id.to_numpy(), s.JW.to_numpy()
        Cu, Cd = s['Cu'].to_numpy(), s['Cd'].to_numpy()
        Cdu, Cdd = s['Cdu'].to_numpy(), s['Cdd'].to_numpy()
        Nu, Nd = s['Nu'].to_numpy(), s['Nd'].to_numpy()
        Ntot, NuNd = s['Ntot'].to_numpy(), s['NuNd'].to_numpy()
        npt.assert_equal(np.dot(Cdu, Cu), Nu)
        npt.assert_equal(np.dot(Cdd, Cd), Nd)
        npt.assert_equal(Nu + Nd, Ntot)
        npt.assert_equal(np.dot(Nu, Nd), NuNd)
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
        check_spin_algebra(s, SxSy=SxSy)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True, NuNd=True, Ntot=True, dN=True)
        expect_charged_ops = dict()  # dict(Svec=3)  # TODO
        if conserve_S != 'Stot':  # (b), (c), (d), (e)
            expect_symmetric_ops.update(JWu=True, JWd=True, Nu=True, Nd=True, Sz=True)
            expect_charged_ops.update(Cu=(1, True), Cdu=(1, True), Cd=(1, True), Cdd=(1, True),
                                      Sp=(1, True), Sm=(1, True))
        if conserve_S in ['parity', 'None']:  # (c), (d), (e)
            expect_charged_ops.update(Sx=(1, True), Sy=(1, True))
        if conserve_S == 'None':  # (d), (e)
            expect_symmetric_ops.update(Sp=False, Sm=False, Sx=False, Sy=False)
        if conserve_S == 'None' == conserve_N:  # (e)
            expect_symmetric_ops.update(Cu=False, Cdu=False, Cd=False, Cdd=False)
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


def test_spin_half_hole_site(any_backend):
    hcs = dict(Id='Id', JW='JW', JWu='JWu', JWd='JWd',
               Cu='Cdu', Cdu='Cu', Cd='Cdd', Cdd='Cd',
               Nu='Nu', Nd='Nd', Ntot='Ntot', dN='dN',
               Sx='Sx', Sy='Sy', Sz='Sz', Sp='Sm', Sm='Sp')  # yapf: disable
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve_N = all_conserve_S = ['None']
    elif isinstance(any_backend, la.AbelianBackend):
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Sz', 'parity', 'None']
    elif isinstance(any_backend, la.FusionTreeBackend):
        all_conserve_N = ['N', 'parity', 'None']
        all_conserve_S = ['Stot', 'Sz', 'parity', 'None']
        pytest.xfail('FusionTree backend not ready')
    else:
        raise ValueError
    sites = []
    for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
        print(f'{conserve_N=}, {conserve_S=}')
        s = site.SpinHalfHoleSite(conserve_N, conserve_S, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        JW = s.JW.to_numpy()
        Cu, Cd = s['Cu'].to_numpy(), s['Cd'].to_numpy()
        Cdu, Cdd = s['Cdu'].to_numpy(), s['Cdd'].to_numpy()
        Nu, Nd = s['Nu'].to_numpy(), s['Nd'].to_numpy()
        Ntot = s['Ntot'].to_numpy()
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
        check_spin_algebra(s, SxSy=SxSy)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True, Ntot=True, dN=True)
        expect_charged_ops = dict()  # dict(Svec=3)  # TODO
        if conserve_S != 'Stot':  # (b), (c), (d), (e)
            expect_symmetric_ops.update(JWu=True, JWd=True, Nu=True, Nd=True, Sz=True)
            expect_charged_ops.update(Cu=(1, True), Cdu=(1, True), Cd=(1, True), Cdd=(1, True),
                                      Sp=(1, True), Sm=(1, True))
        if conserve_S in ['parity', 'None']:  # (c), (d), (e)
            expect_charged_ops.update(Sx=(1, True), Sy=(1, True))
        if conserve_S == 'None':  # (d), (e)
            expect_symmetric_ops.update(Sp=False, Sm=False, Sx=False, Sy=False)
        if conserve_S == 'None' == conserve_N:  # (e)
            expect_symmetric_ops.update(Cu=False, Cdu=False, Cd=False, Cdd=False)
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


@pytest.mark.parametrize('Nmax', [1, 2, 5, 10])
def test_boson_site(any_backend, Nmax):
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, (la.AbelianBackend, la.FusionTreeBackend)):
        all_conserve = ['N', 'parity', 'None']
    else:
        raise ValueError
    hcs = dict(Id='Id', JW='JW', B='Bd', Bd='B', N='N', NN='NN', dN='dN', dNdN='dNdN', P='P')
    sites = []
    for conserve in all_conserve:

        if isinstance(any_backend, backends.FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
                s = site.BosonSite(Nmax, conserve=conserve, backend=any_backend)
            return  # TODO
        
        s = site.BosonSite(Nmax, conserve=conserve, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        B = s['B'].to_numpy()
        Bd = s['Bd'].to_numpy()
        N = s['N'].to_numpy()
        npt.assert_array_almost_equal_nulp(np.dot(Bd, B), N, 2)
        expect_commutator = np.eye(s.dim)
        expect_commutator[-1, -1] = -Nmax  # commutation relation violated due to truncated space
        npt.assert_array_almost_equal_nulp(commutator(B, Bd), expect_commutator, 20)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True, N=True, NN=True, dN=True, dNdN=True, P=True)
        expect_charged_ops = dict(B=(1, True), Bd=(1, True))  # dict(Svec=3)  # TODO
        if conserve == 'None':
            expect_symmetric_ops.update(B=False, Bd=False)
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


@pytest.mark.parametrize('q', [2, 3, 5, 10])
def test_clock_site(any_backend, q):
    if isinstance(any_backend, la.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, (la.AbelianBackend, la.FusionTreeBackend)):
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

        if isinstance(any_backend, backends.FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
                s = site.ClockSite(q=q, conserve=conserve, backend=any_backend)
            return  # TODO
        
        s = site.ClockSite(q=q, conserve=conserve, backend=any_backend)
        s.test_sanity()
        for op in s.all_op_names:
            assert s.hc_ops[op] == hcs[op]
        # check clock algebra
        w = np.exp(2.j * np.pi / q)
        X = s['X'].to_numpy()
        Z = s['Z'].to_numpy()
        Z_pow_q = Z
        X_pow_q = X
        for _ in range(q - 1):
            Z_pow_q = np.dot(Z_pow_q, Z)
            X_pow_q = np.dot(X_pow_q, X)
        npt.assert_array_almost_equal_nulp(np.dot(X, Z), w * np.dot(Z, X), 3 * q)
        npt.assert_array_almost_equal_nulp(X_pow_q, np.eye(q), 3 * q)
        npt.assert_array_almost_equal_nulp(Z_pow_q, np.eye(q), 3 * q)
        # operator availability (check tables in docstring)
        expect_symmetric_ops = dict(Id=True, JW=True, Z=True, Zhc=True, Zphc=True)
        expect_charged_ops = dict(X=(1, True), Xhc=(1, True))  # dict(Svec=3)  # TODO
        if conserve == 'None':
            expect_symmetric_ops.update(X=False, Xhc=False, Xphc=False)
            expect_charged_ops.update(Xphc=(1, True))
        check_operator_availability(s, expect_symmetric_ops, expect_charged_ops)
        # done
        sites.append(s)
    check_same_operators(sites)


def test_set_common_symmetry(any_backend):
    if isinstance(any_backend, la.NoSymmetryBackend):
        conserve_S = conserve_N = 'None'
        expect_symm_Sz = expect_symm_Sz_N = expect_symm_Sz_N_Sz = la.no_symmetry
    else:
        conserve_S = 'Sz'
        conserve_N = 'N'
        expect_symm_Sz = la.U1Symmetry('2*Sz')
        expect_symm_Sz_N = la.U1Symmetry('2*Sz') * la.U1Symmetry('N')
        expect_symm_Sz_N_Sz = la.U1Symmetry('2*Sz') * la.U1Symmetry('N') * la.U1Symmetry('2*Sz')
    
    conserve_S = 'None' if isinstance(any_backend, la.NoSymmetryBackend) else 'Sz'
    conserve_N = 'None' if isinstance(any_backend, la.NoSymmetryBackend) else 'N'

    if isinstance(any_backend, backends.FusionTreeBackend):
        with pytest.raises(NotImplementedError, match='diagonal_from_block_func not implemented'):
            spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=any_backend)
        return  # TODO
    
    spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=any_backend)
    spin1 = site.SpinSite(S=1, conserve=conserve_S, backend=any_backend)
    ferm = site.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S)
    boson = site.BosonSite(Nmax=2, conserve=conserve_N, backend=any_backend)
    spin_ops = {name: op.to_numpy() for name, op in spin.symmetric_ops.items()}
    spin1_ops = {name: op.to_numpy() for name, op in spin1.symmetric_ops.items()}
    ferm_ops = {name: op.to_numpy() for name, op in ferm.symmetric_ops.items()}
    boson_ops = {name: op.to_numpy() for name, op in boson.symmetric_ops.items()}
    # TODO also checks on charged ops?

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
            op_np2 = spin.get_op(op_name).to_numpy()
            npt.assert_equal(op_np, op_np2)
        for op_name, op_np in ferm_ops.items():
            op_np2 = ferm.get_op(op_name).to_numpy()
            npt.assert_equal(op_np, op_np2)

        # reset the modified sites
        spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=any_backend)
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
        op_np2 = spin.get_op(op_name).to_numpy()
        npt.assert_equal(op_np, op_np2, err_msg=f'{op_name=}')
    for op_name, op_np in ferm_ops.items():
        op_np2 = ferm.get_op(op_name).to_numpy()
        npt.assert_equal(op_np, op_np2, err_msg=f'{op_name=}')

    # reset the modified sites
    spin = site.SpinSite(S=0.5, conserve=conserve_S, backend=any_backend)
    ferm = site.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S)

    if isinstance(any_backend, la.NoSymmetryBackend):
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
            op_np2 = s.get_op(op_name).to_numpy()
            npt.assert_equal(op_np, op_np2, err_msg=f'{name=} {op_name=}')
