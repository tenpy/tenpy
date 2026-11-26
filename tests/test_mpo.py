"""A collection of tests for (classes in) :module:`tenpy.networks.mpo`."""
# Copyright (C) TeNPy Developers, Apache license

import lasyten as lasy
import numpy as np
import pytest
from lasyten import ElementarySpace
from lasyten.models.couplings import Coupling
from lasyten.models.sites import AnyonSite, FibonacciAnyonSite, SpinHalfFermionSite, SpinlessFermionSite, SpinSite
from numpy import testing as npt
from scipy import linalg as spla

from tenpy.algorithms import ExactDiag
from tenpy.models import lattice, model
from tenpy.models.fermions_spinless import FermionChain
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.models.spins import DipolarSpinChain, SpinChain
from tenpy.networks import mpo, mps
from tenpy.networks.terms import CouplingTerms, OnsiteTerms
from tenpy.tools import asConfig

spin_half = SpinSite(S=0.5, conserve=None)


def test_MPO():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half
    for bc in mpo.MPO._valid_bc:
        for L in [4, 2, 1]:
            print(bc, ', L =', L)
            grid = [[s.Id, s.Sp, s.Sm, s.Sz],
                    [None, None, None, s.Sm],
                    [None, None, None, s.Sp],
                    [None, None, None, s.Id]]  # fmt: skip
            legW = npc.LegCharge.from_qflat(s.leg.chinfo, [[0], s.Sp.qtotal, s.Sm.qtotal, [0]])
            W = npc.grid_outer(grid, [legW, legW.conj()], grid_labels=['wL', 'wR'])
            Ws = [W] * L
            if bc == 'finite':
                Ws[0] = Ws[0][0:1, :, :, :]
                Ws[-1] = Ws[-1][:, 3:4, :, :]
            H = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L), mps_unit_cell_width=L)
            H_copy = mpo.MPO([s] * L, Ws, bc=bc, IdL=[0] * L + [None], IdR=[None] + [-1] * (L), mps_unit_cell_width=L)
            H.test_sanity()
            print(H.dim)
            print(H.chi)
            assert H.is_equal(H)  # everything should be equal to itself
            assert H.is_hermitian()
            H.test_sanity()
            assert H.is_equal(H_copy)
            assert H.prefactor(0, ['Sz']) == 1.0
            for i in range(L - 1):
                assert H.prefactor(i, ['Sp', 'Sm']) == 1.0
                assert H.prefactor(i, ['Sz', 'Sz']) == 0.0
        if L == 4:
            H2 = H.group_sites(n=2)
            H2.test_sanity()
            assert H2.L == 2


def test_MPOGraph_term_conversion():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L = 4

    g1 = mpo.MPOGraph([spin_half] * L, 'infinite', unit_cell_width=L)
    g1.test_sanity()
    for i in range(L):
        g1.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g1.add(i, 'IdL', ('left', i, 'Sp', 'Id'), 'Sp', 1.0)
        g1.add(i + 1, ('left', i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g1.add_missing_IdL_IdR()
    terms = [[('Sz', i)] for i in range(L)]
    terms += [[('Sp', i), ('Sm', i + 1)] for i in range(L)]
    prefactors = [0.5] * L + [1.5] * L
    term_list = TermList(terms, prefactors)
    g2 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite', unit_cell_width=L)
    g2.test_sanity()
    assert g1.graph == g2.graph
    terms[3:3] = [[('Sm', 2), ('Sp', 0), ('Sz', 1)]]
    prefactors[3:3] = [3.0]
    term_list = TermList(terms, prefactors)
    g3 = mpo.MPOGraph.from_term_list(term_list, [spin_half] * L, 'infinite', unit_cell_width=L)
    g4 = mpo.MPOGraph([spin_half] * L, 'infinite', unit_cell_width=L)
    g4.test_sanity()
    for i in range(L):
        g4.add(i, 'IdL', 'IdR', 'Sz', 0.5)
        g4.add(i, 'IdL', ('left', i, 'Sp', 'Id'), 'Sp', 1.0)
        g4.add(i + 1, ('left', i, 'Sp', 'Id'), 'IdR', 'Sm', 1.5)
    g4.add_missing_IdL_IdR()
    g4.add(1, ('left', 0, 'Sp', 'Id'), ('right', 2, 'Sm', 'Id'), 'Sz', 3.0)
    g4.add(2, ('right', 2, 'Sm', 'Id'), 'IdR', 'Sm', 1.0)
    assert g4.graph == g3.graph


def test_MPO_conversion():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L = 8
    sites = []
    for i in range(L):
        s = site.Site(npc.LegCharge.from_trivial(2))
        s.add_op(f'X_{i:d}', np.array([[0.0, 1.0], [1.0, 0.0]]))
        s.add_op(f'Y_{i:d}', np.array([[0.0, 1.0], [-1.0, 0.0]]))
        s.add_op(f'Z_{i:d}', np.array([[1.0, 0.0], [0.0, -1.0]]))
        sites.append(s)

    terms = [
        [('X_0', 0)],
        [('X_0', 0), ('X_1', 1)],
        [('X_0', 0), ('X_3', 3)],
        [('X_4', 4), ('Y_5', 5), ('Y_7', 7)],
        [('X_4', 4), ('Y_5', 5), ('X_7', 7)],
        [('X_4', 4), ('Y_6', 6), ('Y_7', 7)],
    ]
    prefactors = [0.25, 10.0, 11.0, 101.0, 102.0, 103.0]
    term_list = TermList(terms, prefactors)
    g1 = mpo.MPOGraph.from_term_list(term_list, sites, bc='finite', insert_all_id=False, unit_cell_width=L)
    ct_add = MultiCouplingTerms(L)
    ct_add.add_coupling_term(12.0, 4, 5, 'X_4', 'X_5')
    ct_add.add_multi_coupling_term(0.5, [4, 5, 7], ['X_4', 'Y_5', 'X_7'], 'Id')
    prefactors[-2] += 0.5
    terms.append([('X_4', 4), ('X_5', 5)])
    prefactors.append(12.0)
    ct_add.add_to_graph(g1)
    H1 = g1.build_MPO()
    grids = [
        [
            ['Id', 'X_0', [('X_0', 0.25)]]  # site 0
        ],
        [
            ['Id', None, None],  # site 1
            [None, 'Id', [('X_1', 10.0)]],
            [None, None, 'Id'],
        ],
        [
            ['Id', None, None],  # site 2
            [None, [('Id', 11.0)], None],
            [None, None, 'Id'],
        ],
        [
            ['Id', None],  # site 3
            [None, 'X_3'],
            [None, 'Id'],
        ],
        [
            ['X_4', None],  # site 4
            [None, 'Id'],
        ],
        [
            ['Id', 'Y_5', [('X_5', 12.0)]],  # site 5
            [None, None, 'Id'],
        ],
        [
            # site 6
            [None, [('Y_6', 103.0)], None],
            [[('Id', 102.5)], [('Id', 101.0)], None],
            [None, None, 'Id'],
        ],
        [
            ['X_7'],  # site 7
            ['Y_7'],
            ['Id'],
        ],
    ]
    H2 = mpo.MPO.from_grids(
        sites, grids, 'finite', [0] * 4 + [None] * 5, [None] * 5 + [-1] * 4, mps_unit_cell_width=len(sites)
    )
    for w1, w2 in zip(H1._W, H2._W):
        assert npc.norm(w1 - w2, np.inf) == 0.0
    assert H2.is_equal(H1)
    back_to_terms = H1.to_TermList([['Id', f'X_{i:d}', f'Y_{i:d}', f'Z_{i:d}'] for i in range(L)], max_range=8)
    assert len(back_to_terms.terms) == len(terms)
    for term, pref in zip(back_to_terms.terms, back_to_terms.strength):
        assert term in terms
        i = terms.index(term)
        assert abs(prefactors[i] - pref) < 1.0e-13


def test_MPOEnvironment():
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    xxz_pars = dict(L=4, Jxx=1.0, Jz=1.1, hz=0.1, bc_MPS='finite')
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([[0], [1]] * L)[:L]  # Neel
    psi = mps.MPS.from_lat_product_state(M.lattice, state)
    env = mpo.MPOEnvironment(psi, M.H_MPO, psi)
    env.get_LP(3, True)
    env.get_RP(0, True)
    env.test_sanity()
    E_exact = -0.825
    for i in range(4):
        E = env.full_contraction(i)  # should be one
        print('total energy for contraction at site ', i, ': E =', E)
        assert abs(E - E_exact) < 1.0e-14


def _hopping_coupling(site, op_i, op_j):
    """Build a two-site hopping Coupling ``op_i`` on site 0, ``op_j`` on site 1."""
    # dense block axes [p0, p1, p1*, p0*], see Coupling.from_dense_block / spin_spin_coupling
    h = np.tensordot(op_i, op_j, axes=0)  # [p0, p0*, p1, p1*]
    h = np.transpose(h, [0, 2, 3, 1])  # -> [p0, p1, p1*, p0*]
    return Coupling.from_dense_block(h, [site, site], understood_braiding=True)


def _mpo_to_dense(M, L):
    labels = [f'p{i}' for i in range(L)] + [f'p{i}*' for i in range(L)]
    dim = M.sites[0].dim
    return Coupling(list(M.sites), factorization=list(M._W)).to_tensor().to_numpy(labels).reshape(dim**L, dim**L)


def _nonhermitian_mpo(name, conserve, bc, L=4, dxs=(1, 2)):
    """The MPO of a chain with one-sided and/or complex couplings, which is not hermitian.

    The couplings respect the symmetry. `name` is one of

    - ``'spin'``: spin-1/2 with `conserve` in ``[None, 'Sz', 'SU2']``,
    - ``'fermion'``: spinless fermions with `conserve` in ``['N', 'parity']``,
    - ``'hubbard'``: spinful fermions with ``conserve_N='N'`` and ``conserve_S=conserve``.

    The `dxs` are the distances of the (at most two) couplings.
    """
    if name == 'spin':
        site = SpinSite(S=0.5, conserve=conserve)
        if conserve == 'SU2':
            # hermitian, but with complex prefactors
            to_define = [(lasy.models.heisenberg_coupling([site, site], name='S.S'), 'S.S', 'invariant')]
            terms = [('S.S', 1.0), ('S.S', 0.4j)]
        else:
            to_define = [
                (lasy.models.spin_spin_coupling([site, site], Jmp=1, name='Sm.Sp'), 'Sp.Sm', 'swap_hc'),
                (lasy.models.spin_spin_coupling([site, site], Jz=1, name='Sz.Sz'), 'Sz.Sz', 'invariant'),
            ]
            terms = [('Sm.Sp', 1.0), ('Sz.Sz', 0.3j)]
    else:
        if name == 'fermion':
            site = SpinlessFermionSite(conserve=conserve)
        else:
            site = SpinHalfFermionSite(conserve_N='N', conserve_S=conserve)
        # one-sided hopping with complex prefactors
        to_define = [(lasy.models.hopping([site, site], t_right=0, name='Cd.C'), 'C.Cd', 'swap_hc')]
        terms = [('Cd.C', 1j), ('Cd.C', 0.5 - 0.2j)]
    lat = lattice.Chain(L, site, bc='open' if bc == 'finite' else 'periodic', bc_MPS=bc)
    model_ = model.CouplingModel(lat)
    for coupling, hc, permute in to_define:
        model_.define_coupling(coupling, hc=hc, permute=permute)
    for dx, (coupling_name, strength) in zip(dxs, terms):
        model_.add_two_site_coupling(strength, coupling_name, dx=dx)
    return model_.get_H_MPO()


def _hermitian_mpo(name, conserve, bc, L=4):
    """The MPO of a hermitian model with the same `name` and `conserve` as in :func:`_nonhermitian_mpo`."""
    if name == 'spin':
        # couplings that respect the symmetry
        couplings = {
            None: dict(Jx=1.0, Jy=0.5, Jz=1.0, hz=0.2),
            'Sz': dict(Jx=1.0, Jy=1.0, Jz=1.5, hz=0.2),
            'SU2': dict(Jx=1.0, Jy=1.0, Jz=1.0),
        }
        return SpinChain(dict(L=L, bc_MPS=bc, conserve=conserve, **couplings[conserve])).get_H_MPO()
    if name == 'fermion':
        pars = dict(L=L, J=1.0, V=0.5, mu=0.2, conserve=conserve, bc_MPS=bc, symmetry_backend='fusion_tree')
        return FermionChain(pars).get_H_MPO()
    pars = dict(L=L, t=1.0, U=2.0, mu=0.3, bc_MPS=bc, conserve_N='N', conserve_S=conserve)
    return FermiHubbardModel(pars).get_H_MPO()


def test_MPO_dagger():
    """MPO.dagger agrees with the dense hermitian conjugate and keeps the IdL / IdR structure."""
    L = 4
    M = _nonhermitian_mpo('spin', None, 'finite', L, dxs=(3, 1))  # one-sided long range 'Sm.Sp', complex 'Sz.Sz'
    M_dag = M.dagger()
    M_dag.test_sanity()
    H_dense = _mpo_to_dense(M, L)
    H_dag_dense = _mpo_to_dense(M_dag, L)
    assert not np.allclose(H_dense, H_dense.conj().T)
    npt.assert_allclose(H_dag_dense, H_dense.conj().T, atol=1e-14)
    # the summand labels of the virtual legs are unchanged
    for W, W_dag in zip(M._W, M_dag._W):
        assert W_dag.labels == ['wL', 'p', 'wR', 'p*']
        assert W.get_leg('wL').summand_labels == W_dag.get_leg('wL').summand_labels
        assert W.get_leg('wR').summand_labels == W_dag.get_leg('wR').summand_labels
    # dagger is an involution
    npt.assert_allclose(_mpo_to_dense(M_dag.dagger(), L), H_dense, atol=1e-14)

    # symmetric, hermitian model
    M = SpinChain(dict(L=L, Jx=1.0, Jy=0.5, Jz=1.0, hz=0.2, bc_MPS='finite', conserve='best')).get_H_MPO()
    M_dag = M.dagger()
    M_dag.test_sanity()
    npt.assert_allclose(_mpo_to_dense(M_dag, L), _mpo_to_dense(M, L), atol=1e-14)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_MPO_dagger_virtual_legs(bc, conserve):
    """The virtual legs of `M.dagger()` have the same duality as those of `M`, as needed to combine them."""
    M = _nonhermitian_mpo('spin', conserve, bc)
    M_dag = M.dagger()
    M_dag.test_sanity()
    for W, W_dag in zip(M._W, M_dag._W):
        for leg in ['wL', 'wR']:
            assert W_dag.get_leg(leg).is_dual == W.get_leg(leg).is_dual
            assert W_dag.get_leg(leg).summand_labels == W.get_leg(leg).summand_labels
            assert W_dag.get_leg(leg).dim == W.get_leg(leg).dim
    # applying dagger twice restores the dualities of the virtual legs; the tensors themselves are only restored
    # up to a gauge on the virtual legs (e.g. for conserve='SU2'), so we do not compare them
    for W, W_dd in zip(M._W, M_dag.dagger()._W):
        assert W_dd.get_leg('wR').is_dual == W.get_leg('wR').is_dual


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
@pytest.mark.parametrize(
    'name, conserve, L',
    [
        ('spin', None, 4),
        ('spin', 'Sz', 4),
        ('spin', 'SU2', 4),
        ('fermion', 'N', 4),
        ('fermion', 'parity', 4),
        ('hubbard', 'SU2', 3),
    ],
)
def test_MPO_dagger_IdL_IdR_structure(bc, name, conserve, L):
    """`M.dagger()` keeps the summand labels and the block-upper-triangular `IdL` / `IdR` structure.

    ``W = [[1, C, D], [0, A, B], [0, 0, 1]]`` as a matrix of the summand labels ``[IdL, *others, IdR]``,
    see :func:`~tenpy.networks.mpo.partition_W`, which verifies this structure.
    """
    M = _nonhermitian_mpo(name, conserve, bc, L)
    M_dag = M.dagger()
    for i, (W, W_dag) in enumerate(zip(M._W, M_dag._W)):
        left, right = M.finite and i == 0, M.finite and i == M.L - 1
        # technically: the summand labels (and their order) survive
        for leg in ['wL', 'wR']:
            assert W_dag.get_leg(leg).summand_labels == W.get_leg(leg).summand_labels
        # definitionally: `partition_W` checks the identity cells at (IdL, IdL) and (IdR, IdR) and the
        # zeros below the diagonal blocks, it raises if W_dag is not of the right form.
        blocks = mpo.partition_W(W_dag, left_boundary=left, right_boundary=right)
        blocks_orig = mpo.partition_W(W, left_boundary=left, right_boundary=right)
        for block, block_orig in zip(blocks[:4], blocks_orig[:4]):  # A, B, C, D; Id is the same by construction
            assert (block is None) == (block_orig is None)
        # the hermitian conjugate acts on each cell separately: the zero pattern (even the norm) is the same
        for row in W.get_leg('wL').summand_labels:
            for col in W.get_leg('wR').summand_labels:
                norm = lasy.norm(lasy.tensor_grid_cell(W, row, col, 'wL', 'wR'))
                norm_dag = lasy.norm(lasy.tensor_grid_cell(W_dag, row, col, 'wL', 'wR'))
                assert norm_dag == pytest.approx(norm, abs=1e-12), f'cell ({row}, {col}) of W[{i}]'


@pytest.mark.parametrize('conserve', [None, 'Sz'])
def test_MPO_dagger_environment(conserve, L=4):
    """`IdL` / `IdR` still mean the same: ``<psi|H^dagger|psi> = conj(<psi|H|psi>)`` via `MPOEnvironment`."""
    site = SpinSite(S=0.5, conserve=conserve)
    M = _nonhermitian_mpo('spin', conserve, 'finite', L)
    M_dag = M.dagger()
    # a generic complex state; a fixed seed makes sure that the imaginary part of <psi|H|psi> checked below is
    # not accidentally small
    rng = np.random.default_rng(seed=0)
    vec = rng.normal(size=(site.dim,) * L) + 1j * rng.normal(size=(site.dim,) * L)
    if conserve == 'Sz':
        charges = [np.asarray(site.leg.idx_to_sector(k)).ravel()[0] for k in range(site.dim)]  # 2 * Sz per state
        vec = vec * (sum(np.meshgrid(*[charges] * L, indexing='ij')) == 0)  # total Sz = 0
    full = lasy.SymmetricTensor.from_dense_block(
        vec, codomain=[site.leg] * L, labels=[f'p{i}' for i in range(L)], understood_braiding=True
    )
    psi = mps.MPS.from_full([site] * L, full, unit_cell_width=L)
    env, env_dag = mpo.MPOEnvironment(psi, M, psi), mpo.MPOEnvironment(psi, M_dag, psi)
    for i in range(L):
        E = env.full_contraction(i)
        assert abs(E.imag) > 1e-3  # not hermitian, so the test is not trivial
        assert env_dag.full_contraction(i) == pytest.approx(np.conj(E), abs=1e-12)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
@pytest.mark.parametrize(
    'name, conserve, L',
    [
        ('spin', None, 4),
        ('spin', 'Sz', 4),
        ('spin', 'SU2', 4),
        ('fermion', 'N', 4),
        ('fermion', 'parity', 4),
        ('hubbard', 'SU2', 3),
    ],
)
def test_MPO_is_hermitian(bc, name, conserve, L):
    hermitian = _hermitian_mpo(name, conserve, bc, L)
    assert hermitian.is_hermitian()
    assert hermitian.is_equal(hermitian.dagger())
    non_hermitian = _nonhermitian_mpo(name, conserve, bc, L)
    assert not non_hermitian.is_hermitian()
    assert not non_hermitian.is_equal(non_hermitian.dagger())


def _check_dagger_tensor(M):
    """`M.dagger()` agrees with ``lasy.dagger`` of the contracted MPO and is an involution."""
    M_dag = M.dagger()
    M_dag.test_sanity()
    T = Coupling(list(M.sites), factorization=list(M._W)).to_tensor()
    T_dag = Coupling(list(M_dag.sites), factorization=list(M_dag._W)).to_tensor()
    assert T_dag.labels == lasy.dagger(T).labels
    assert lasy.almost_equal(T_dag, lasy.dagger(T))
    assert not lasy.almost_equal(T, lasy.dagger(T))  # the test case is not hermitian
    for W, W_dag in zip(M._W, M_dag._W):
        assert W_dag.labels == ['wL', 'p', 'wR', 'p*']
        for leg in ['wL', 'wR']:
            if hasattr(W.get_leg(leg), 'summand_labels'):  # e.g. not for random W tensors
                assert W.get_leg(leg).summand_labels == W_dag.get_leg(leg).summand_labels
    T_dag_dag = Coupling(list(M.sites), factorization=list(M_dag.dagger()._W)).to_tensor()
    assert lasy.almost_equal(T_dag_dag, T)


@pytest.mark.parametrize('conserve', ['N', 'parity'])
@pytest.mark.parametrize('dxs', [(1,), (1, 3)], ids=['nearest', 'long_range'])
def test_MPO_dagger_fermions(conserve, dxs):
    """MPO.dagger needs planar leg permutations (no braids), e.g. for the Jordan-Wigner strings."""
    _check_dagger_tensor(_nonhermitian_mpo('fermion', conserve, 'finite', 4, dxs=dxs))


def _random_anyonic_mpo(name, L, bc='finite'):
    """An MPO of anyons with random complex `W` tensors that contain all sectors on the virtual legs.

    `name` is one of ``'Fibonacci'``, ``'SU2_2'``, ``'Ising'``. The `W` do not have summand labels.
    The virtual legs at the boundary of a finite MPO are trivial.
    """
    if name == 'Fibonacci':
        site = FibonacciAnyonSite()
    elif name == 'SU2_2':
        site = AnyonSite(lasy.Symmetry([lasy.SU2_kAnyonCategory(2)]))
    else:
        site = AnyonSite(lasy.Symmetry([lasy.IsingAnyonCategory(3)]))
    sym = site.symmetry
    sectors = [np.array(s) for s in site.leg.sector_decomposition]
    virtual = ElementarySpace.from_defining_sectors(sym, sectors, [1] * len(sectors))
    spaces = [virtual] * (L + 1)
    if bc == 'finite':
        spaces[0] = spaces[-1] = ElementarySpace.from_trivial_sector(1, symmetry=sym)
    Ws = [
        lasy.SymmetricTensor.from_random_normal(
            [spaces[i], site.leg],
            [site.leg, spaces[i + 1]],
            labels=[['wL', 'p'], ['p*', 'wR']],
            dtype=lasy.Dtype.complex128,
        )
        for i in range(L)
    ]
    return mpo.MPO([site] * L, Ws, bc, mps_unit_cell_width=L)


@pytest.mark.parametrize(
    'name, L', [('Fibonacci', 3), ('Fibonacci', 4), ('SU2_2', 2), ('SU2_2', 3), ('Ising', 2), ('Ising', 3)]
)
def test_MPO_dagger_anyons(name, L):
    """MPO.dagger for anyons, with random complex W tensors containing all sectors on the virtual legs."""
    _check_dagger_tensor(_random_anyonic_mpo(name, L))


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
def test_MPO_dagger_golden_chain(bc, L=4):
    """The dagger of the MPO of a non-hermitian golden chain keeps the `IdL` / `IdR` structure; ``H + H.dagger()``.

    The MPO is built by hand, since `CouplingModel` / `MPOGraph` do not support anyons yet, see #730.
    ``H = J sum_i h_{i,i+1}`` with the golden chain coupling ``h = -P^vac`` (`gold_coupling`) and a complex `J`.
    """
    site = FibonacciAnyonSite()
    gold = lasy.models.gold_coupling([site, site], name='gold')
    Id = Coupling.from_tensor(lasy.SymmetricTensor.from_eye([site.leg], labels=['p0']), [site]).factorization[0]
    J = 1.0 + 0.5j
    grid = [[Id, J * gold.factorization[0], None], [None, None, gold.factorization[1]], [None, None, Id]]
    labels = ['IdL', 'gold', 'IdR']
    W = lasy.tensor_from_grid(grid, ['wL', 'wR'], stack_labels=(labels, labels))
    H = mpo.MPO([site] * 2, [W, W], 'infinite', mps_unit_cell_width=2)
    if bc == 'finite':
        H = _finite_from_infinite(H, L)
    H.test_sanity()
    assert not H.is_hermitian()
    H_dag = H.dagger()
    H_dag.test_sanity()
    for i, (W, W_dag) in enumerate(zip(H._W, H_dag._W)):
        for leg in ['wL', 'wR']:
            assert W_dag.get_leg(leg).summand_labels == W.get_leg(leg).summand_labels
        mpo.partition_W(W_dag, left_boundary=H.finite and i == 0, right_boundary=H.finite and i == H.L - 1)
    S = H + H_dag
    S.test_sanity()
    assert S.has_IdL_IdR_labels()
    assert S.is_hermitian()
    if bc == 'infinite':
        # no dense operator of an infinite MPO: compare the projections to a finite chain
        H, S = _finite_from_infinite(H, L), _finite_from_infinite(S, L)
    # no dense matrix for anyons: compare the contracted tensors
    T = Coupling(list(H.sites), factorization=list(H._W)).to_tensor()
    T_sum = Coupling(list(S.sites), factorization=list(S._W)).to_tensor()
    assert not lasy.almost_equal(T, lasy.dagger(T))  # the test case is not hermitian
    assert lasy.almost_equal(T_sum, T + lasy.dagger(T))


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
@pytest.mark.parametrize(
    'name, L', [('Fibonacci', 3), ('Fibonacci', 4), ('SU2_2', 2), ('SU2_2', 3), ('Ising', 2), ('Ising', 3)]
)
def test_MPO_add_dagger_anyons(bc, name, L):
    """``H + H.dagger()`` for anyons, with random `W` without summand labels (the block-diagonal sum)."""
    H = _random_anyonic_mpo(name, L, bc)
    H_dag = H.dagger()
    S = H + H_dag
    S.test_sanity()
    # the block-diagonal sum has the virtual legs of both operands (the boundary legs of a finite MPO are trivial)
    for i, (chi, chi_dag, chi_sum) in enumerate(zip(H.chi, H_dag.chi, S.chi)):
        trivial = bc == 'finite' and i in [0, L]
        assert chi_sum == pytest.approx(chi if trivial else chi + chi_dag)
    if bc == 'finite':
        # no dense matrix for anyons: compare the contracted tensors
        T = Coupling(list(H.sites), factorization=list(H._W)).to_tensor()
        T_sum = Coupling(list(S.sites), factorization=list(S._W)).to_tensor()
        assert not lasy.almost_equal(T, lasy.dagger(T))  # the test case is not hermitian
        assert lasy.almost_equal(T_sum, T + lasy.dagger(T))


def test_MPO_dagger_explicit_plus_hc():
    """For ``explicit_plus_hc=True`` the result does not share its list of ``W`` with the original."""
    M = SpinChain(dict(L=3, Jx=1.0, Jy=0.5, Jz=1.0, hz=0.2, bc_MPS='finite', conserve=None)).get_H_MPO()
    M.explicit_plus_hc = True
    W_orig = list(M._W)
    M_dag = M.dagger()
    assert M_dag.explicit_plus_hc
    assert M_dag._W is not M._W
    M_dag._W[0] = M_dag._W[1]
    M_dag._W.pop()
    assert len(M._W) == len(W_orig)
    assert all(a is b for a, b in zip(M._W, W_orig))


def _finite_from_infinite(M, L):
    """Project the infinite MPO `M` to a finite chain of length `L` (a multiple of ``M.L``).

    The first / last tensor are the ``IdL`` row / ``IdR`` column of the repeated unit cell.
    """
    Ws = [M._W[i % M.L] for i in range(L)]
    first = lasy.tensor_subgrid(Ws[0], ['IdL'], list(range(len(Ws[0].get_leg('wR').summand_labels))), 'wL', 'wR')
    last = lasy.tensor_subgrid(Ws[-1], list(range(len(Ws[-1].get_leg('wL').summand_labels))), ['IdR'], 'wL', 'wR')
    return mpo.MPO([M.sites[i % M.L] for i in range(L)], [first] + Ws[1:-1] + [last], 'finite', mps_unit_cell_width=L)


def test_MPO_dagger_infinite(L=4):
    """MPO.dagger of an infinite MPO: projecting to a finite chain gives the conjugate transpose."""
    site = SpinSite(S=0.5, conserve=None)

    def cell(name, pref=1.0):
        op = (site.get_op(name) * pref).as_SymmetricTensor()
        op = lasy.add_trivial_leg(op, codomain_pos=0, label='wL', is_dual=False)
        return lasy.add_trivial_leg(op, domain_pos=-1, label='wR', is_dual=False)

    Id = cell('Id')
    grid = [
        [Id, cell('Sp'), cell('Sz', 0.7j), cell('Sz', 0.3 + 0.2j)],
        [None, None, None, cell('Sm')],
        [None, None, None, cell('Sz')],
        [None, None, None, Id],
    ]
    labels = ['IdL', 'a', 'b', 'IdR']
    W = lasy.tensor_from_grid(grid, ['wL', 'wR'], stack_labels=(labels, labels))
    M = mpo.MPO([site], [W], 'infinite', mps_unit_cell_width=1)
    M.test_sanity()
    M_dag = M.dagger()
    M_dag.test_sanity()
    assert M_dag.bc == 'infinite'

    D, D_dag = (_mpo_to_dense(_finite_from_infinite(X, L), L) for X in [M, M_dag])
    assert not np.allclose(D, D.conj().T)  # the test case is not hermitian
    npt.assert_allclose(D_dag, D.conj().T, atol=1e-14)


@pytest.mark.parametrize('bc', ['finite', 'infinite'])
@pytest.mark.parametrize('hermitian', [True, False], ids=['hermitian', 'non_hermitian'])
@pytest.mark.parametrize(
    'name, conserve, L, L_cell',
    [
        ('spin', None, 4, 2),
        ('spin', None, 6, 3),
        ('spin', 'Sz', 4, 2),
        ('spin', 'SU2', 4, 2),
        ('spin', 'SU2', 6, 3),
        ('fermion', 'N', 4, 2),
        ('fermion', 'parity', 3, 3),
        ('hubbard', 'SU2', 3, 1),
    ],
)
def test_MPO_add_dagger(bc, hermitian, name, conserve, L, L_cell):
    """``H + H.dagger()`` is the dense ``M + M^dagger``, also for an infinite `H`, which is projected to length `L`.

    `L_cell` is the unit cell of the infinite MPO. This needs `MPO.__add__` to deal with the summand labels
    that `H` and ``H.dagger()`` share.
    """
    make = _hermitian_mpo if hermitian else _nonhermitian_mpo
    H = make(name, conserve, bc, L if bc == 'finite' else L_cell)
    H_dag = H.dagger()
    S = H + H_dag
    S.test_sanity()
    assert S.has_IdL_IdR_labels()
    assert S.is_equal(H_dag + H)
    assert S.is_hermitian()
    if bc == 'infinite':
        # IdL and IdR are shared, all other summands of the two MPOs are kept
        assert list(S.chi) == [chi + chi_dag - 2 for chi, chi_dag in zip(H.chi, H_dag.chi)]
        # no dense matrix of an infinite MPO: compare the projections to a finite chain
        H, S = _finite_from_infinite(H, L), _finite_from_infinite(S, L)
    M, S_dense = _dense_H(H), _dense_H(S)
    npt.assert_allclose(S_dense, M + M.conj().T, atol=1e-12)
    if hermitian:
        npt.assert_allclose(M, M.conj().T, atol=1e-12)
        npt.assert_allclose(S_dense, 2 * M, atol=1e-12)
    else:
        assert not np.allclose(M, M.conj().T, atol=1e-8)  # the test is not vacuous
        assert not np.allclose(S_dense, 2 * M, atol=1e-8)


@pytest.mark.parametrize('bc', ['infinite', 'finite'])
@pytest.mark.parametrize('explicit_plus_hc', [True, False])
def test_MPO_addition(bc, explicit_plus_hc):
    L = 4
    print('bc = ', bc, '-' * 40)
    s = spin_half

    def new_model():
        lat_bc = 'open' if bc == 'finite' else 'periodic'
        lat = lattice.Chain(L, s, bc=lat_bc, bc_MPS=bc)
        M = model.CouplingModel(lat)
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jmp=1, name='Sm.Sp'),
            hc='Sp.Sm',
            permute='swap_hc',
        )
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jx=1, name='Sx.Sx'),
            hc='Sx.Sx',
            permute='invariant',
        )
        M.define_coupling(
            lasy.models.spin_spin_coupling([s, s], Jz=1, name='Sz.Sz'),
            hc='Sz.Sz',
            permute='invariant',
        )
        return M

    def graph_from_model(M):
        return mpo.MPOGraph.from_terms(M.get_nonzero_terms(), M, bc, unit_cell_width=M.lattice.mps_unit_cell_width)

    M1 = new_model()
    M2 = new_model()
    M12 = new_model()

    str_A = [0, 0, 2.0, 0] if bc == 'infinite' else [0, 0, 2.0]  # dx=1
    str_B = [0, 2.0, 0, 0] if bc == 'infinite' else [0, 2.0, 0]  # dx=1
    str_C = [0, 4.0, 7.0, 0] if bc == 'infinite' else [4, 7.0]  # dx=2
    for M in [M1, M12]:
        M.add_two_site_coupling(str_A, 'Sm.Sp', [1], u1=0, u2=0, plus_hc=True)
        M.add_two_site_coupling(str_B, 'Sz.Sz', [1], u1=0, u2=0)
        M.add_onsite_operator([0, 3.0, 0, 0], 'Sz', x='sum_over', u=0)
    cpl_slc = slice(0, 4) if bc == 'infinite' else slice(0, 3)
    for M in [M2, M12]:
        M.add_two_site_coupling(str_C, 'Sz.Sz', [2], u1=0, u2=0)
        M.add_onsite_operator([0, 5.0, 0, 0], 'Sz', x='sum_over', u=0)

    H1 = graph_from_model(M1).build_MPO()
    H2 = graph_from_model(M2).build_MPO()
    H12 = graph_from_model(M12).build_MPO()
    H12_sum = H1 + H2
    assert H12.is_equal(H12_sum)


def _dense_H(H):
    ED = ExactDiag.from_H_mpo(H)
    ED.build_full_H_from_mpo()
    # `understood_braiding=True` is needed for fermions. Since we only compare matrices obtained in the
    # same way, the (basis) convention of the dense representation does not matter.
    return ED.full_H.to_numpy(understood_braiding=True)


@pytest.mark.parametrize('L', [2, 3, 4, 5])
def test_MPO_addition_spin_chain(L):
    """Adding two MPOs gives the sum of the dense operators and keeps the IdL / IdR labels."""
    M1 = SpinChain(dict(L=L, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2, bc_MPS='finite', conserve=None))
    M2 = SpinChain(dict(L=L, Jx=0.3, Jy=0.0, Jz=-0.7, hz=0.5, bc_MPS='finite', conserve=None))
    H1, H2 = M1.get_H_MPO(), M2.get_H_MPO()
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.L == L
    assert H12.bc == 'finite'

    # IdL / IdR are available as grid labels on the bonds
    for i, W in enumerate(H12._W):
        if i > 0:
            assert 'IdL' in W.get_leg('wL').summand_labels
            assert 'IdR' in W.get_leg('wL').summand_labels
        if i < L - 1:
            assert 'IdL' in W.get_leg('wR').summand_labels
            assert 'IdR' in W.get_leg('wR').summand_labels
    assert H12._W[0].get_leg('wL').summand_labels == ['IdL']
    assert H12._W[-1].get_leg('wR').summand_labels == ['IdR']

    npt.assert_allclose(_dense_H(H12), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H12), _dense_H(H2 + H1), atol=1e-12)
    # environments (which use IdL / IdR) work and give the sum of the energies
    sites = M1.lattice.mps_sites()
    psi = mps.MPS.from_product_state(sites, (['up', 'down'] * L)[:L], bc='finite', unit_cell_width=L)
    E = [mpo.MPOEnvironment(psi, H, psi).full_contraction(0) for H in (H1, H2, H12)]
    npt.assert_allclose(E[2], E[0] + E[1], atol=1e-12)

    # the sum can be added again, i.e. it is again in standard form
    npt.assert_allclose(_dense_H(H12 + H1), 2 * _dense_H(H1) + _dense_H(H2), atol=1e-12)


@pytest.mark.parametrize(
    'conserve, backend, couplings',
    [
        (None, 'no_symmetry', 'XXZ'),
        (None, 'abelian', 'XXZ'),
        (None, 'fusion_tree', 'XXZ'),
        ('parity', 'abelian', 'XXZ'),
        ('parity', 'fusion_tree', 'XXZ'),
        ('Sz', 'abelian', 'XXZ'),
        ('Sz', 'fusion_tree', 'XXZ'),
        ('SU2', 'fusion_tree', 'Heisenberg'),
    ],
)
def test_MPO_addition_symmetric(conserve, backend, couplings, L=4):
    """Adding MPOs with symmetries gives the sum of the dense operators and takes the fast path."""
    if couplings == 'XXZ':
        # Jx = Jy, such that Sz is conserved
        couplings1 = dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)
        couplings2 = dict(Jx=0.4, Jy=0.4, Jz=-0.7, hz=0.5)
    else:
        # SU(2) symmetric, i.e. isotropic couplings and no field
        couplings1 = dict(Jx=1.0, Jy=1.0, Jz=1.0, hz=0.0)
        couplings2 = dict(Jx=0.4, Jy=0.4, Jz=0.4, hz=0.0)
    pars = dict(L=L, bc_MPS='finite', conserve=conserve, symmetry_backend=backend)
    H1 = SpinChain(dict(**couplings1, **pars)).get_H_MPO()
    assert H1._W[0].backend == lasy.get_backend(backend)
    H2 = SpinChain(dict(**couplings2, **pars)).get_H_MPO()
    H12 = H1 + H2
    H12.test_sanity()
    npt.assert_allclose(_dense_H(H12), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    # fast path: the IdL / IdR entries are shared, the bond dimension is smaller than for the block diagonal sum
    assert H12.has_IdL_IdR_labels()
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi[1:-1], H1.chi[1:-1], H2.chi[1:-1]))


def test_MPO_addition_long_range(L=5):
    """Adding MPOs with three- and four-site (i.e. longer range) terms."""
    pars = dict(L=L, bc_MPS='finite', conserve=None)
    H1 = DipolarSpinChain(dict(J3=1.0, J4=0.3, **pars)).get_H_MPO()
    H2 = DipolarSpinChain(dict(J3=-0.4, J4=0.8, **pars)).get_H_MPO()
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.has_IdL_IdR_labels()
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi[1:-1], H1.chi[1:-1], H2.chi[1:-1]))  # fast path
    npt.assert_allclose(_dense_H(H12), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    # also mixed with a nearest-neighbor chain
    H3 = SpinChain(dict(Jx=1.0, Jy=0.5, Jz=0.2, hz=0.3, S=1, **pars)).get_H_MPO()
    npt.assert_allclose(_dense_H(H1 + H3), _dense_H(H1) + _dense_H(H3), atol=1e-12)


def test_MPO_addition_errors():
    pars = dict(Jx=1.0, Jy=1.0, Jz=1.0, bc_MPS='finite', conserve=None)
    H4 = SpinChain(dict(L=4, hz=0.2, **pars)).get_H_MPO()
    H3 = SpinChain(dict(L=3, hz=0.2, **pars)).get_H_MPO()
    with pytest.raises(ValueError, match='different length'):
        _ = H4 + H3
    H4_inf = H4.copy()
    H4_inf.bc = 'infinite'
    with pytest.raises(ValueError, match='different bc'):
        _ = H4 + H4_inf
    H4_hc = H4.copy()
    H4_hc.explicit_plus_hc = True
    with pytest.raises(ValueError, match='explicit_plus_hc'):
        _ = H4 + H4_hc
    H4_seg = H4.copy()
    H4_seg.bc = 'segment'
    # 'segment' bc is not supported: documented limitation of MPO.__add__ (#676), the error is the intended behavior
    with pytest.raises(NotImplementedError, match='segment'):
        _ = H4_seg + H4_seg
    assert H4.__add__(1.0) is NotImplemented


def _unlabelled_MPO(H):
    """Copy of the MPO `H` without any summand labels (not in the standard form)."""
    Ws = [lasy.tensor_from_grid(mpo._grid_cells(W), ['wL', 'wR']) for W in H._W]
    return mpo.MPO(H.sites, Ws, H.bc, H.max_range, mps_unit_cell_width=H.unit_cell_width)


def test_MPO_has_IdL_IdR_labels():
    pars = dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.0)
    H = SpinChain(dict(hz=0.2, **pars)).get_H_MPO()
    assert H.has_IdL_IdR_labels()
    assert (H + H).has_IdL_IdR_labels()  # the sum in standard form is again in standard form
    # MPOGraph puts 'IdL' and 'IdR' on all bulk bonds, also without a field
    assert SpinChain(dict(hz=0.0, **pars)).get_H_MPO().has_IdL_IdR_labels()
    # without summand labels
    H_unlabelled = _unlabelled_MPO(H)
    assert not H_unlabelled.has_IdL_IdR_labels()
    assert not (H + H_unlabelled).has_IdL_IdR_labels()  # the generic sum has no inner labels
    # only on-site terms: the labels are there
    assert SpinChain(dict(**{**pars, 'Jx': 0, 'Jy': 0, 'Jz': 0}, hz=0.2)).get_H_MPO().has_IdL_IdR_labels()
    # at the outer legs of a finite MPO, only the respective single summand is allowed
    H_infinite = SpinChain(dict(L=1, bc_MPS='infinite', conserve=None, Jx=0, Jy=0, Jz=0, hz=0.2)).get_H_MPO()
    assert H_infinite.has_IdL_IdR_labels()
    H_wrong_bc = H_infinite.copy()
    H_wrong_bc.bc = 'finite'
    assert not H_wrong_bc.has_IdL_IdR_labels()
    # for 'segment' bc, the outer legs are not restricted to single summands: both labels are fine
    H_segment = mpo.MPO(H.sites[1:3], H._W[1:3], 'segment', mps_unit_cell_width=2)
    assert H_segment.has_IdL_IdR_labels()


_BLOCK_DIAGONAL_CASES = {
    # name: ((kwargs, unlabelled) of the first, (kwargs, unlabelled) of the second MPO)
    'labels_and_unlabelled': ((dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2), False), (dict(Jx=0.4, Jy=0.4, Jz=-0.7), True)),
    'unlabelled_and_labels': ((dict(Jx=0.4, Jy=0.4, Jz=-0.7), True), (dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2), False)),
    'unlabelled_and_unlabelled': ((dict(Jx=1.0, Jy=1.0, Jz=1.1), True), (dict(Jx=0.4, Jy=0.4, Jz=-0.7), True)),
}
# these have the IdL / IdR labels, but on-site MPOs do not have summands besides them
_STANDARD_FORM_CASES = {
    'onsite_and_labels': (dict(Jx=0, Jy=0, Jz=0, hz=0.4), dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)),
    'labels_and_onsite': (dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2), dict(Jx=0, Jy=0, Jz=0, hz=0.4)),
    'onsite_and_onsite': (dict(Jx=0, Jy=0, Jz=0, hz=0.4), dict(Jx=0, Jy=0, Jz=0, hz=-0.1)),
}


@pytest.mark.parametrize('conserve', [None, 'Sz'])
@pytest.mark.parametrize('case', list(_BLOCK_DIAGONAL_CASES))
@pytest.mark.parametrize('L', [2, 3, 4])
def test_MPO_addition_block_diagonal(L, case, conserve):
    """MPOs without the standard IdL / IdR labels are added as a block diagonal MPO."""
    (kwargs1, unlabelled1), (kwargs2, unlabelled2) = _BLOCK_DIAGONAL_CASES[case]
    pars = dict(L=L, bc_MPS='finite', conserve=conserve)
    H1 = SpinChain(dict(**kwargs1, **pars)).get_H_MPO()
    H2 = SpinChain(dict(**kwargs2, **pars)).get_H_MPO()
    if unlabelled1:
        H1 = _unlabelled_MPO(H1)
    if unlabelled2:
        H2 = _unlabelled_MPO(H2)
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.L == L
    assert H12.bc == 'finite'
    assert not H12.has_IdL_IdR_labels()  # the inner bonds are not labelled
    # the outer legs have the labels, such that the sum can be used in environments
    assert H12._W[0].get_leg('wL').summand_labels == ['IdL']
    assert H12._W[-1].get_leg('wR').summand_labels == ['IdR']
    npt.assert_allclose(_dense_H(H12), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H2 + H1), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    # block diagonal: the bond dimensions add up
    chi1, chi2 = H1.chi, H2.chi
    assert H12.chi[0] == H12.chi[-1] == 1
    assert all(c12 == c1 + c2 for c12, c1, c2 in zip(H12.chi[1:-1], chi1[1:-1], chi2[1:-1]))
    # the sum can be added again
    npt.assert_allclose(_dense_H(H12 + H1), 2 * _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H1 + H12), 2 * _dense_H(H1) + _dense_H(H2), atol=1e-12)


def test_MPO_addition_block_diagonal_environment():
    """The block diagonal sum can be used in environments (the outer legs have IdL / IdR)."""
    L = 4
    pars = dict(L=L, bc_MPS='finite', conserve='Sz')
    H1 = SpinChain(dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2, **pars)).get_H_MPO()
    H2_labelled = SpinChain(dict(Jx=0.4, Jy=0.4, Jz=-0.7, hz=0.0, **pars)).get_H_MPO()
    H2 = _unlabelled_MPO(H2_labelled)
    assert not H2.has_IdL_IdR_labels()
    H12 = H1 + H2
    psi = mps.MPS.from_product_state(H1.sites, ['up', 'down'] * (L // 2), bc='finite', unit_cell_width=L)
    # H2 itself can not be used in an environment (no outer IdL / IdR), so take its labelled original
    E1, E2, E12 = [mpo.MPOEnvironment(psi, H, psi).full_contraction(0) for H in (H1, H2_labelled, H12)]
    npt.assert_allclose(E12, E1 + E2, atol=1e-12)
    npt.assert_allclose(H12.expectation_value(psi), E1 + E2, atol=1e-12)


@pytest.mark.parametrize('scaled_first', [True, False], ids=['scaled+H', 'H+scaled'])
@pytest.mark.parametrize('factor', [2.0, -1.0, 0.0])
def test_MPO_addition_rescaled(factor, scaled_first):
    """MPOs with the labels, but without the upper triangular structure are added block diagonally."""
    H = SpinChain(dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)).get_H_MPO()
    # MPO for ``factor * H`` obtained by rescaling the first tensor: keeps the labels, but breaks the structure
    H_scaled = _MPO_with_replaced_W(H, 0, factor * H._W[0])
    assert H_scaled.has_IdL_IdR_labels()
    assert not H_scaled._has_upper_triangular_structure()
    assert H._has_upper_triangular_structure()
    npt.assert_allclose(_dense_H(H_scaled), factor * _dense_H(H), atol=1e-12)
    H12 = H_scaled + H if scaled_first else H + H_scaled
    assert not H12.has_IdL_IdR_labels()
    npt.assert_allclose(_dense_H(H12), (factor + 1) * _dense_H(H), atol=1e-12)


@pytest.mark.parametrize('L', [2, 3, 4])
@pytest.mark.parametrize('case', list(_STANDARD_FORM_CASES))
@pytest.mark.parametrize('conserve', [None, 'Sz'])
def test_MPO_addition_standard_form_onsite(L, case, conserve):
    """MPOs with on-site terms only still have the shared IdL / IdR entries when added."""
    kwargs1, kwargs2 = _STANDARD_FORM_CASES[case]
    pars = dict(L=L, bc_MPS='finite', conserve=conserve)
    H1 = SpinChain(dict(**kwargs1, **pars)).get_H_MPO()
    H2 = SpinChain(dict(**kwargs2, **pars)).get_H_MPO()
    assert H1.has_IdL_IdR_labels() and H2.has_IdL_IdR_labels()
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.has_IdL_IdR_labels()
    npt.assert_allclose(_dense_H(H12), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H2 + H1), _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H12 + H1), 2 * _dense_H(H1) + _dense_H(H2), atol=1e-12)
    # the shared IdL / IdR entries make the bond dimension smaller than block diagonal
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi[1:-1], H1.chi[1:-1], H2.chi[1:-1]))
    if case == 'onsite_and_onsite':
        assert list(H12.chi) == [1] + [2] * (L - 1) + [1]  # only IdL and IdR on the bonds
    else:
        assert list(H12.chi) == list(max([H1, H2], key=lambda H: sum(H.chi)).chi)


def test_MPO_addition_standard_form_onsite_infinite():
    """On-site only infinite MPOs share IdL / IdR as well."""
    pars = dict(L=1, bc_MPS='infinite', conserve=None, Jx=0, Jy=0, Jz=0)
    H12 = SpinChain(dict(hz=0.2, **pars)).get_H_MPO() + SpinChain(dict(hz=-0.5, **pars)).get_H_MPO()
    H12.test_sanity()
    assert H12.has_IdL_IdR_labels()
    assert list(H12.chi) == [2, 2]  # only IdL and IdR


@pytest.mark.parametrize('order', ['12', '21', '22'])
def test_MPO_addition_block_diagonal_infinite(order):
    """For infinite bc, the generic sum is block diagonal on all tensors and has no labels."""
    pars = dict(L=1, bc_MPS='infinite', conserve=None, Jx=0, Jy=0, Jz=0)
    H1 = SpinChain(dict(hz=0.2, **pars)).get_H_MPO()
    H2 = _unlabelled_MPO(SpinChain(dict(hz=-0.5, **pars)).get_H_MPO())
    assert H1.has_IdL_IdR_labels()
    assert not H2.has_IdL_IdR_labels()
    A, B = [{'1': H1, '2': H2}[key] for key in order]
    H12 = A + B
    H12.test_sanity()
    assert H12.bc == 'infinite'
    assert not H12.has_IdL_IdR_labels()
    assert list(H12.chi) == [cA + cB for cA, cB in zip(A.chi, B.chi)]


@pytest.mark.parametrize(
    'conserve, backend',
    [
        (None, 'no_symmetry'),
        (None, 'abelian'),
        (None, 'fusion_tree'),
        ('parity', 'abelian'),
        ('parity', 'fusion_tree'),
        ('Sz', 'abelian'),
        ('Sz', 'fusion_tree'),
    ],
)
@pytest.mark.parametrize('L', [1, 2, 3])
def test_MPO_addition_spin_chain_infinite(L, conserve, backend):
    """Adding infinite MPOs of models gives the MPO of the model with the summed couplings (fast path)."""
    pars = dict(L=L, S=0.5, conserve=conserve, bc_MPS='infinite', symmetry_backend=backend)
    # Jx = Jy, such that Sz is conserved
    H1 = SpinChain(dict(Jx=1.0, Jy=1.0, Jz=0.3, hz=0.2, **pars)).get_H_MPO()
    H2 = SpinChain(dict(Jx=0.5, Jy=0.5, Jz=-1.0, hz=0.1, **pars)).get_H_MPO()
    H12_expected = SpinChain(dict(Jx=1.5, Jy=1.5, Jz=-0.7, hz=0.3, **pars)).get_H_MPO()
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.bc == 'infinite'
    assert H12.L == L
    assert H12.has_IdL_IdR_labels()
    # fast path: the IdL / IdR entries are shared, the bond dimension is smaller than for the block diagonal sum
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi, H1.chi, H2.chi))
    # is_equal returns a lasyten Scalar, which is converted to bool by the assert
    assert bool(H12.is_equal(H12_expected))
    assert bool((H2 + H1).is_equal(H12_expected))
    # negative controls: the sum differs from each of the summands
    assert not bool(H12.is_equal(H1))
    assert not bool(H12.is_equal(H2))
    assert not bool(H1.is_equal(H12_expected))


@pytest.mark.parametrize('conserve', ['N', 'parity'])
@pytest.mark.parametrize('L', [2, 3, 4])
def test_MPO_addition_fermions(L, conserve):
    """Adding MPOs of fermionic models gives the dense Hamiltonian of the model with the summed couplings."""
    # conserve=None is not supported for fermions
    pars = dict(L=L, conserve=conserve, bc_MPS='finite', symmetry_backend='fusion_tree')
    H1 = FermionChain(dict(J=1.0, V=0.5, mu=0.2, **pars)).get_H_MPO()
    H2 = FermionChain(dict(J=0.3, V=-1.0, mu=0.7, **pars)).get_H_MPO()
    H12_expected = FermionChain(dict(J=1.3, V=-0.5, mu=0.9, **pars)).get_H_MPO()
    H_wrong = FermionChain(dict(J=1.3, V=0.5, mu=0.9, **pars)).get_H_MPO()  # V is not summed
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.has_IdL_IdR_labels()
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi[1:-1], H1.chi[1:-1], H2.chi[1:-1]))
    dense_H12 = _dense_H(H12)
    npt.assert_allclose(dense_H12, _dense_H(H12_expected), atol=1e-12)
    npt.assert_allclose(dense_H12, _dense_H(H1) + _dense_H(H2), atol=1e-12)
    npt.assert_allclose(_dense_H(H2 + H1), dense_H12, atol=1e-12)
    # negative controls
    assert not np.allclose(dense_H12, _dense_H(H_wrong), atol=1e-8)
    assert not np.allclose(dense_H12, _dense_H(H1), atol=1e-8)


@pytest.mark.parametrize('conserve', ['N', 'parity'])
@pytest.mark.parametrize('L', [1, 2, 3])
def test_MPO_addition_fermions_infinite(L, conserve):
    """Adding infinite MPOs of fermionic models gives the MPO of the model with the summed couplings."""
    pars = dict(L=L, conserve=conserve, bc_MPS='infinite', symmetry_backend='fusion_tree')
    H1 = FermionChain(dict(J=1.0, V=0.5, mu=0.2, **pars)).get_H_MPO()
    H2 = FermionChain(dict(J=0.3, V=-1.0, mu=0.7, **pars)).get_H_MPO()
    H12_expected = FermionChain(dict(J=1.3, V=-0.5, mu=0.9, **pars)).get_H_MPO()
    H_wrong = FermionChain(dict(J=1.3, V=0.5, mu=0.9, **pars)).get_H_MPO()  # V is not summed
    H12 = H1 + H2
    H12.test_sanity()
    assert H12.has_IdL_IdR_labels()
    assert all(c12 < c1 + c2 for c12, c1, c2 in zip(H12.chi, H1.chi, H2.chi))
    assert bool(H12.is_equal(H12_expected))
    assert not bool(H12.is_equal(H_wrong))
    assert not bool(H12.is_equal(H1))


@pytest.mark.parametrize('dual_first', [True, False], ids=['dual+H', 'H+dual'])
def test_MPO_addition_mismatching_duality(dual_first):
    """Virtual legs with different ``is_dual`` flags can not be added."""
    H = SpinChain(dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)).get_H_MPO()
    Ws = []
    for W in H._W:
        # like the hermitian conjugate, which dualizes the virtual legs
        W_dag = lasy.planar_permute_legs(lasy.dagger(W), codomain=['wL*', 'p'], domain=['p*', 'wR*'])
        W_dag.labels.update({'wL*': 'wL', 'wR*': 'wR'})
        Ws.append(W_dag)
    H_dual = mpo.MPO(H.sites, Ws, 'finite', H.max_range, mps_unit_cell_width=H.unit_cell_width)
    A, B = (H_dual, H) if dual_first else (H, H_dual)
    with pytest.raises(ValueError, match='duality'):
        _ = A + B
    H_dual_sum = H_dual + H_dual  # same duality is fine
    H_dual_sum.test_sanity()
    assert H_dual_sum.has_IdL_IdR_labels()
    # numerically, compare the contracted MPOs as tensors
    tensors = [Coupling(list(M.sites), factorization=list(M._W)).to_tensor() for M in (H_dual, H_dual_sum)]
    assert lasy.almost_equal(tensors[1], 2 * tensors[0])


def test_MPO_make_U_I_needs_labels():
    H = SpinChain(dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)).get_H_MPO()
    H = _unlabelled_MPO(H)
    with pytest.raises(ValueError, match='IdL'):
        H.make_U_I(0.1)


def test_MPO_make_U_I_needs_structure():
    """``make_U_I`` raises a clear ValueError (not an assert) for a wrong structure."""
    pars = dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.1)
    H = SpinChain(dict(hz=0.2, **pars)).get_H_MPO()
    H.make_U_I(0.1)  # fine
    # rescaled first tensor: has the labels, but the IdL-IdL cell is not the identity
    H_scaled = _MPO_with_replaced_W(H, 0, 2.0 * H._W[0])
    assert H_scaled.has_IdL_IdR_labels()
    with pytest.raises(ValueError, match='IdL.*not the identity'):
        H_scaled.make_U_I(0.1)
    # nonzero block below the diagonal
    H_lower = _MPO_with_replaced_W(H, 1, _W_with_cell(H._W[1], 'IdR', 'IdL', 0.3))
    assert H_lower.has_IdL_IdR_labels()
    with pytest.raises(ValueError, match='IdR.*IdL.*nonzero'):
        H_lower.make_U_I(0.1)
    # only on-site terms: no summands besides IdL, IdR, the blocks A, B, C are None
    H_onsite = SpinChain(dict(**{**pars, 'Jx': 0, 'Jy': 0, 'Jz': 0}, hz=0.2)).get_H_MPO()
    assert H_onsite.has_IdL_IdR_labels()
    A, B, C, D, Id = mpo.partition_W(H_onsite._W[1])
    assert A is None and B is None and C is None and D is not None


def _MPO_with_replaced_W(H, i, W):
    return mpo.MPO(
        H.sites, H._W[:i] + [W] + H._W[i + 1 :], 'finite', H.max_range, mps_unit_cell_width=H.unit_cell_width
    )


def _W_with_cell(W, row_label, col_label, factor):
    """Copy of `W` where the cell (row_label, col_label) is replaced by `factor` times the identity."""
    row_labels, col_labels = W.get_leg('wL').summand_labels, W.get_leg('wR').summand_labels
    cells = mpo._grid_cells(W)
    cells[row_labels.index(row_label)][col_labels.index(col_label)] = factor * mpo._identity_cell(W)
    return lasy.tensor_from_grid(cells, ['wL', 'wR'], stack_labels=(row_labels, col_labels))


# modifications of an MPO of a spin chain, which do / do not keep the upper triangular structure
_STRUCTURE_VARIANTS = {
    'ok': lambda H: H,
    'rescaled first W': lambda H: _MPO_with_replaced_W(H, 0, 2 * H._W[0]),
    'rescaled bulk W': lambda H: _MPO_with_replaced_W(H, 1, 2 * H._W[1]),
    'rescaled last W': lambda H: _MPO_with_replaced_W(H, 3, 3 * H._W[3]),
    'lower-left IdR-IdL': lambda H: _MPO_with_replaced_W(H, 1, _W_with_cell(H._W[1], 'IdR', 'IdL', 0.3)),
    'lower-left other-IdL': lambda H: _MPO_with_replaced_W(
        H, 2, _W_with_cell(H._W[2], H._W[2].get_leg('wL').summand_labels[1], 'IdL', 0.3)
    ),
    'wrong identity IdL': lambda H: _MPO_with_replaced_W(H, 1, _W_with_cell(H._W[1], 'IdL', 'IdL', 0.5)),
    'wrong identity IdR': lambda H: _MPO_with_replaced_W(H, 2, _W_with_cell(H._W[2], 'IdR', 'IdR', -1.0)),
    # boundary tensors
    'left boundary wrong identity': lambda H: _MPO_with_replaced_W(H, 0, _W_with_cell(H._W[0], 'IdL', 'IdL', 2.0)),
    'right boundary wrong identity': lambda H: _MPO_with_replaced_W(H, 3, _W_with_cell(H._W[3], 'IdR', 'IdR', 2.0)),
    'no labels': _unlabelled_MPO,
}


@pytest.mark.parametrize('variant', list(_STRUCTURE_VARIANTS))
def test_MPO_upper_triangular_structure_consistent_with_partition_W(variant):
    """``_has_upper_triangular_structure`` is True exactly if ``partition_W`` accepts all tensors."""
    H = SpinChain(dict(L=4, bc_MPS='finite', conserve=None, Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2)).get_H_MPO()
    H = _STRUCTURE_VARIANTS[variant](H)
    accepted = True
    try:
        for i, W in enumerate(H._W):
            mpo.partition_W(W, left_boundary=i == 0, right_boundary=i == H.L - 1)
    except ValueError:
        accepted = False
    assert H._has_upper_triangular_structure() == accepted
    # only the unmodified MPO has the structure
    assert accepted == (variant == 'ok')


# labels of the non-Id summands, different / equal for the two operands
_SUMMAND_LABELS = {
    'equal': ('a', 'a'),
    'different': ('a', 'c'),
    'none': (None, None),
}


def _with_summand_labels(H, prefix):
    """Copy of `H` where the non-Id summand labels are ``prefix0, prefix1, ...`` (or ``''``, i.e. unlabelled)."""
    Ws = []
    for W in H._W:
        new_labels = []
        for leg in ('wL', 'wR'):
            labels = W.get_leg(leg).summand_labels
            new_labels.append(
                [l if l in ['IdL', 'IdR'] else ('' if prefix is None else f'{prefix}{k}') for k, l in enumerate(labels)]
            )
        Ws.append(
            lasy.tensor_from_grid(
                mpo._grid_cells(W),
                ['wL', 'wR'],
                stack_labels=(new_labels[0], new_labels[1]),
            )
        )
    return mpo.MPO(H.sites, Ws, 'finite', H.max_range, mps_unit_cell_width=H.unit_cell_width)


@pytest.mark.parametrize('operands', ['12', '11', '22'])
@pytest.mark.parametrize('case', list(_SUMMAND_LABELS))
@pytest.mark.parametrize('L', [2, 4])
def test_MPO_addition_duplicate_summand_labels(case, L, operands):
    """Operands may use equal (or no) labels for their non-Id summands; they are dropped in the sum."""
    prefix1, prefix2 = _SUMMAND_LABELS[case]
    pars = dict(L=L, bc_MPS='finite', conserve=None)
    H1 = SpinChain(dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.2, **pars)).get_H_MPO()
    H2 = SpinChain(dict(Jx=0.4, Jy=0.3, Jz=-0.7, hz=-0.5, **pars)).get_H_MPO()
    H1l, H2l = _with_summand_labels(H1, prefix1), _with_summand_labels(H2, prefix2)
    assert H1l.has_IdL_IdR_labels() and H2l.has_IdL_IdR_labels()
    if prefix1 is not None:
        assert any(str(l).startswith(prefix1) for l in H1l._W[0].get_leg('wR').summand_labels)
    A, B = [{'1': H1l, '2': H2l}[key] for key in operands]
    S = A + B
    S.test_sanity()
    assert S.has_IdL_IdR_labels()
    npt.assert_allclose(_dense_H(S), _dense_H(A) + _dense_H(B), atol=1e-12)
    assert all(c < ca + cb for c, ca, cb in zip(S.chi[1:-1], A.chi[1:-1], B.chi[1:-1]))  # fast path
    npt.assert_allclose(_dense_H(H1l + H2l + H1l), 2 * _dense_H(H1) + _dense_H(H2), atol=1e-12)


@pytest.mark.parametrize('operands', ['12', '21', '11'])
@pytest.mark.parametrize('case', list(_SUMMAND_LABELS))
def test_MPO_addition_duplicate_summand_labels_infinite(case, operands):
    """Same as above for a one-site infinite MPO from a hand-built grid."""
    site = SpinSite(S=0.5, conserve=None)

    def cell(name, pref=1.0):
        op = (site.get_op(name) * pref).as_SymmetricTensor()
        op = lasy.add_trivial_leg(op, codomain_pos=0, label='wL', is_dual=False)
        return lasy.add_trivial_leg(op, domain_pos=-1, label='wR', is_dual=False)

    Id = cell('Id')

    def imp(prefix, factor):
        labels = ['IdL'] + ['' if prefix is None else f'{prefix}{k}' for k in range(2)] + ['IdR']
        grid = [
            [Id, cell('Sp'), cell('Sz', 0.7j), cell('Sz', 0.3 * factor)],
            [None, None, None, cell('Sm')],
            [None, None, None, cell('Sz', factor)],
            [None, None, None, Id],
        ]
        W = lasy.tensor_from_grid(grid, ['wL', 'wR'], stack_labels=(labels, labels))
        return mpo.MPO([site], [W], 'infinite', mps_unit_cell_width=1)

    prefix1, prefix2 = _SUMMAND_LABELS[case]
    H1, H2 = imp(prefix1, 1.0), imp(prefix2, -2.0)
    A, B = [{'1': H1, '2': H2}[key] for key in operands]
    S = A + B
    S.test_sanity()
    assert S.has_IdL_IdR_labels()
    assert list(S.chi) == [6, 6]  # IdL, IdR shared, and two non-Id summands from each operand
    assert S._W[0].get_leg('wL').summand_labels == ['IdL', '', '', '', '', 'IdR']
    # check the numbers: all cells of the sum. Rows / columns of the sum: IdL, the two non-Id summands of A,
    # those of B, IdR; rows / columns of A and B: IdL, their two non-Id summands, IdR.
    W_sum, W_A, W_B = S._W[0], A._W[0], B._W[0]

    def grid_cell(W, row, col):
        return lasy.tensor_grid_cell(W, row, col, 'wL', 'wR')

    expected = {
        (0, 0): grid_cell(W_A, 0, 0),
        (5, 5): grid_cell(W_A, 3, 3),
        (0, 5): grid_cell(W_A, 0, 3) + grid_cell(W_B, 0, 3),
    }
    for k in (1, 2):
        expected[0, k], expected[0, k + 2] = grid_cell(W_A, 0, k), grid_cell(W_B, 0, k)
        expected[k, 5], expected[k + 2, 5] = grid_cell(W_A, k, 3), grid_cell(W_B, k, 3)
    for row in range(6):
        for col in range(6):
            if (row, col) in expected:
                assert lasy.almost_equal(grid_cell(W_sum, row, col), expected[row, col])
            else:
                assert lasy.norm(grid_cell(W_sum, row, col)) == pytest.approx(0)


@pytest.mark.parametrize('j', [3, 4, 7])
def test_MPOGraph_coupling_across_unit_cell_boundary(j):
    L = 3
    s = spin_half
    lat = lattice.Chain(L, s, bc='periodic', bc_MPS='infinite')
    M = model.CouplingModel(lat)
    M.define_coupling(
        lasy.models.spin_spin_coupling([s, s], Jz=1, name='Sz.Sz'),
        hc='Sz.Sz',
        permute='invariant',
    )
    ct = CouplingTerms(L)
    ct.add_coupling_term(1.0, L - 1, j, 'Sz.Sz')

    graph = mpo.MPOGraph.from_terms((ct,), M, bc='infinite', unit_cell_width=L)
    H = graph.build_MPO()
    H.test_sanity()

    expected_wraps = sum(k % L == 0 for k in range(L, j))
    wrap_edges = [
        (keyL, keyR)
        for keyL, edges in graph.graph[0].items()
        for keyR in edges
        if keyL not in ('IdL', 'IdR') and keyR not in ('IdL', 'IdR') and keyL != keyR
    ]
    assert len(wrap_edges) == expected_wraps


def test_MPOGraph_add_onsite_infers_virtual_leg_from_periodic_right_boundary():
    L = 3
    graph = mpo.MPOGraph([spin_half] * L, bc='infinite', unit_cell_width=L)
    graph.states[0]['right_state'] = graph.states[0]['IdL']

    graph.add_identity(L - 1, 'left_state', 'right_state', overbraid=True)

    assert graph.states[L]['right_state'] == graph.states[0]['right_state']
    graph.build_MPO().test_sanity()


@pytest.mark.parametrize('sites', [None, [0], [1], [0, 1, 2, 3], [2, 3], [3, 2]])
def test_MPO_plus_identity(sites, alpha=0.7, beta=0.42):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half

    ot = OnsiteTerms(4)
    ct = CouplingTerms(4)
    ct.add_coupling_term(2.0, 2, 3, 'Sm', 'Sp')
    ct.add_coupling_term(2.0, 2, 3, 'Sp', 'Sm')
    ct.add_coupling_term(2.0, 1, 2, 'Sz', 'Sz')
    ot.add_onsite_term(3.0, 1, 'Sz')
    H1 = mpo.MPOGraph.from_terms((ot, ct), [s] * 4, 'finite', unit_cell_width=4).build_MPO()

    if sites is None:
        H2 = H1.plus_identity(alpha=alpha, beta=beta)
    else:
        H2 = H1.plus_identity(alpha=alpha, beta=beta, sites=sites)

    ot_expect = OnsiteTerms(4)
    ct_expect = CouplingTerms(4)
    ct_expect.add_coupling_term(beta * 2.0, 2, 3, 'Sm', 'Sp')
    ct_expect.add_coupling_term(beta * 2.0, 2, 3, 'Sp', 'Sm')
    ct_expect.add_coupling_term(beta * 2.0, 1, 2, 'Sz', 'Sz')
    ot_expect.add_onsite_term(beta * 3.0, 1, 'Sz')
    ot_expect.add_onsite_term(alpha, 1, 'Id')
    H2_expect = mpo.MPOGraph.from_terms((ot_expect, ct_expect), [s] * 4, 'finite', unit_cell_width=4).build_MPO()

    assert H2.is_equal(H2_expect)


def test_MPO_expectation_value(tol=1.0e-15):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    L_mpo = 4
    s = spin_half
    psi1 = mps.MPS.from_singlets(s, 6, [(1, 3), (2, 5)], lonely=[0, 4], bc='infinite', unit_cell_width=6)
    psi1.test_sanity()
    ot = OnsiteTerms(L_mpo)  # H.L != psi.L, consider L = lcm(4, 6) = 12
    ot.add_onsite_term(0.1, 0, 'Sz')  # -> 0.5 * 2 (sites 0, 4, not 8)
    ot.add_onsite_term(0.2, 3, 'Sz')  # -> 0.  (not sites 4, 7, 11)
    ct = CouplingTerms(L_mpo)  # note: ct.L != psi1.L
    ct.add_coupling_term(1.0, 2, 3, 'Sz', 'Sz')  # -> 0. (not 2-3, 6-7, 10-11)
    ct.add_coupling_term(1.5, 1, 3, 'Sz', 'Sz')  # -> 1.5*(-0.25) (1-3, not 5-7, not 9-11)
    ct.add_coupling_term(2.5, 0, 6, 'Sz', 'Sz')  # -> 2.5*0.25*3 (0-6, 4-10, not 8-14)
    H = mpo.MPOGraph.from_terms((ot, ct), [s] * L_mpo, 'infinite', unit_cell_width=L_mpo).build_MPO()
    desired_ev = (0.1 * 0.5 * 2 + 0.2 * 0.0 + 1.0 * 0.0 + 1.5 * -0.25 + 2.5 * 0.25 * 2) / 12
    ev_power = H.expectation_value_power(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    ev_TM = H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_TM - desired_ev) < tol
    ev = H.expectation_value(psi1, tol)
    assert abs(ev - desired_ev) < tol

    # now with exponentially decaying term
    grid = [
        [s.Id, s.Sz, 3 * s.Sz],
        [None, 0.1 * s.Id, s.Sz],
        [None, None, s.Id],
    ]
    L_mpo = 1
    exp_dec_H = mpo.MPO.from_grids([s] * L_mpo, [grid] * L_mpo, bc='infinite', IdL=0, IdR=2, mps_unit_cell_width=L_mpo)
    desired_ev = (
        3 * 0.5 * 2  # Sz onsite
        + 0.25
        * (
            0.1**3
            + 0.1**5
            + 0.1**9
            + 0.1**11
            + 0.1**15  # Z_0 Z_i
            + 0.1**1
            + 0.1**5
            + 0.1**7
            + 0.1**11
            + 0.1**13
        )  # Z_4 Z_i
        + -0.25
        * (
            0.1**1  # Z_1 Z_3
            + 0.1**2
        )  # Z_2 Z_4
        + 0.0  # other sites
    ) / 6.0  # values > 1.e-15
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol
    L_mpo = 3  # should give exactly the same answer!
    exp_dec_H = mpo.MPO.from_grids([s] * L_mpo, [grid] * L_mpo, bc='infinite', IdL=0, IdR=2, mps_unit_cell_width=L_mpo)
    ev_power = exp_dec_H.expectation_value_power(psi1, tol=tol)
    ev_TM = exp_dec_H.expectation_value_TM(psi1, tol=tol)
    assert abs(ev_power - desired_ev) < tol
    assert abs(ev_TM - desired_ev) < tol
    ev = exp_dec_H.expectation_value(psi1, tol=tol)
    assert abs(ev - desired_ev) < tol


@pytest.mark.parametrize(
    'conserve, backend, couplings',
    [
        (None, 'no_symmetry', 'XXZ'),
        (None, 'abelian', 'XXZ'),
        (None, 'fusion_tree', 'XXZ'),
        ('best', 'abelian', 'XXZ'),  # U(1)
        ('best', 'fusion_tree', 'XXZ'),  # U(1)
        ('best', 'fusion_tree', 'Heisenberg'),  # SU(2)
    ],
)
def test_MPO_var(conserve, backend, couplings, L=6, tol=1.0e-12):
    if couplings == 'XXZ':
        couplings = dict(Jx=1.0, Jy=1.0, Jz=1.1, hz=0.1)
    else:
        couplings = dict(Jx=1.0, Jy=1.0, Jz=1.0, hz=0.0)
    xxz_pars = dict(L=L, bc_MPS='finite', conserve=conserve, symmetry_backend=backend, **couplings)
    M = SpinChain(xxz_pars)
    H = M.get_H_MPO()
    assert H._W[0].backend == lasy.get_backend(backend)
    psi = mps.MPS.from_desired_bond_dimension(M.lattice.mps_sites(), 6, bc='finite', unit_cell_width=L)
    exp_val = H.expectation_value(psi)

    ED = ExactDiag.from_model(M)
    ED.build_full_H_from_mpo()
    psi_full = lasy.split_legs(ED.mps_to_full(psi))
    psi_labels = [f'p{i}' for i in range(L)]
    H_psi = lasy.tdot(ED.full_H, psi_full, [f'p{i}*' for i in range(L)], psi_labels)
    exp_val_full = lasy.inner(psi_full, H_psi)
    assert abs(exp_val - exp_val_full) / abs(exp_val_full) < tol

    Hsquared = H.variance(psi, 0.0)
    Hsquared_full = lasy.inner(H_psi, H_psi)
    assert abs(Hsquared - Hsquared_full) / abs(Hsquared_full) < tol
    var = H.variance(psi)
    var_full = Hsquared_full - exp_val_full**2
    assert abs(var - var_full) / abs(var_full) < tol


@pytest.mark.parametrize('method', ['SVD', 'variational', 'zip_up'])
def test_apply_mpo(method):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    bc_MPS = 'finite'
    # NOTE: overlap doesn't work for calculating the energy (density) in infinite systems!
    # energy is extensive, overlap exponential....
    L = 5
    g = 0.5
    model_pars = dict(L=L, Jx=0.0, Jy=0.0, Jz=-4.0, hx=2.0 * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = [[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L  # pointing in (-x)-direction
    psi = mps.MPS.from_product_state(
        M.lattice.mps_sites(), state, bc=bc_MPS, unit_cell_width=M.lattice.mps_unit_cell_width
    )
    H = M.H_MPO
    Eexp = H.expectation_value(psi)
    psi2 = psi.copy()
    options = {'compression_method': method, 'trunc_params': {'chi_max': 50}}
    H.apply(psi2, options)
    Eapply = psi2.overlap(psi)
    assert abs(Eexp - Eapply) < 1e-5
    psi3 = psi.copy()
    H.apply(psi3, options)
    Eapply3 = psi3.overlap(psi)
    assert abs(Eexp - Eapply3) < 1e-5


def test_MPOTransferMatrix(eps=1.0e-13):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    s = spin_half
    # exponential decay in Sz term to make it harder
    gamma = 0.5
    Jxy, Jz = 4.0, 1.0
    hz = 0.0
    grid = [[s.Id, s.Sp, s.Sm, s.Sz, hz * s.Sz],
            [None, None, None, None, Jxy*0.5*s.Sm],
            [None, None, None, None, Jxy*0.5*s.Sp],
            [None, None, None, gamma*s.Id, Jz*s.Sz],
            [None, None, None, None, s.Id]]  # fmt: skip
    H = mpo.MPO.from_grids([s] * 3, [grid] * 3, 'infinite', 0, 4, max_range=np.inf, mps_unit_cell_width=3)
    psi = mps.MPS.from_singlets(s, 3, [(0, 1)], lonely=[2], bc='infinite', unit_cell_width=3)
    psi.roll_mps_unit_cell(-1)  # -> nontrivial chi at the cut between unit cells
    exact_E = (
        (-0.25 - 0.25) * Jxy  # singlet <0.5*Sp_i Sm_{i+1}> = 0.25 = <0.5*Sm_i Sp_{i+1}>
        - 0.25 * Jz  # singlet <Sz_i Sz_{i+1}>
        + 1.0 * (0.25 * gamma**2 / (1.0 - gamma**3))
    )  # exponentially decaying "lonely" states
    # lonely: <Sz_{i} gamma gamma Sz_{i+2}
    exact_E = exact_E / psi.L  # energy per site
    for transpose in [False, True]:
        print(f'transpose={transpose!s}')
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        TM.matvec(TM.guess, project=False)
        TM.matvec(TM.guess, project=True)
        val, vec = TM.dominant_eigenvector()
        assert abs(val - 1.0) < eps
        E0 = TM.energy(vec)
        print(E0, exact_E)
        assert abs(E0 - exact_E) < eps
        if not transpose:
            vec.itranspose(['vL', 'wL', 'vL*'])
            assert (vec[:, 4, :] - npc.eye_like(vec, 0)).norm() < eps
        else:
            vec.itranspose(['vR*', 'wR', 'vR'])
            assert (vec[:, 0, :] - npc.eye_like(vec, 0)).norm() < eps
    # get non-trivial psi
    psi.perturb({'N_steps': 10, 'trunc_params': {'chi_max': 3, 'svd_min': 1.0e-14}}, close_1=False, canonicalize=False)
    psi.canonical_form_infinite2()
    # now find eigenvector again
    # and check TM |vec> = |vec> + val * |v0>
    # with |v0> = eye(chi) * IdL/IdR
    for transpose in [False, True]:
        print(f'transpose={transpose!s}')
        TM = mpo.MPOTransferMatrix(H, psi, transpose=transpose)
        val, vec = TM.dominant_eigenvector()
        E0 = TM.energy(vec)
        assert abs(val - 1.0) < eps
        if not transpose:
            vec.itranspose(['vL', 'wL', 'vL*'])
            assert (vec[:, 4, :] - npc.eye_like(vec, 0)).norm() < eps
            v0 = npc.eye_like(vec, 0, labels=['vL', 'vL*']).add_leg(vec.get_leg('wL'), 0, 1, 'wL')
        else:
            vec.itranspose(['vR*', 'wR', 'vR'])
            assert (vec[:, 0, :] - npc.eye_like(vec, 0)).norm() < eps
            v0 = npc.eye_like(vec, 0, labels=['vR*', 'vR']).add_leg(vec.get_leg('wR'), 4, 1, 'wR')
        TM_vec = TM.matvec(vec, project=False)
        TM_vec.itranspose(vec.get_leg_labels())
        assert (TM_vec - (vec + E0 * 3 * v0)).norm() < eps


def test_MPO_from_wavepacket(L=10):
    pytest.skip('MPO test skipped for now, as it is not yet adapted to lasyten coupling framework')
    (
        k0,
        x0,
        sigma,
    ) = np.pi / 8.0, 2.0, 2.0
    x = np.arange(L)
    coeff = np.exp(-1.0j * k0 * x) * np.exp(-0.5 * (x - x0) ** 2 / sigma**2)
    coeff /= np.linalg.norm(coeff)
    s = site.FermionSite(conserve='N')
    wp = mpo.MPO.from_wavepacket([s] * L, coeff, 'Cd', unit_cell_width=L)
    psi = mps.MPS.from_product_state([s] * L, ['empty'] * L, unit_cell_width=L)
    wp.apply(psi, dict(compression_method='SVD'))
    C = psi.correlation_function('Cd', 'C')
    C_expexcted = np.conj(coeff)[:, np.newaxis] * coeff[np.newaxis, :]
    assert np.max(np.abs(C - C_expexcted)) < 1.0e-10
    print(C)
    print(C_expexcted)


# Will remove once this works in exact_diag
def _dense_Hamiltonian(site, L, terms):
    """Reference dense Hamiltonian built directly from `site`'s own operators.

    `terms` is a list of ``(strength, {position: opname})`` entries, e.g.
    ``(0.5, {0: 'Sz', 1: 'Sz'})`` for a two-site term.
    """
    ops = {name: site.get_op(name).to_numpy() for name in ('Sx', 'Sy', 'Sz')}
    Id = np.eye(site.dim)
    dim = site.dim**L
    H = np.zeros((dim, dim), dtype=complex)
    for strength, positions_ops in terms:
        factors = [Id] * L
        for pos, opname in positions_ops.items():
            factors[pos] = ops[opname]
        term = np.eye(1)
        for factor in factors:
            term = np.kron(term, factor)
        H += strength * term
    return H


# Will remove once this works in exact_diag
def _coupling_to_dense(coupling, L):
    """Contract a Coupling's factorization into a dense (dim**L, dim**L) matrix."""
    labels = [f'p{i}' for i in range(L)] + [f'p{i}*' for i in range(L)]
    dim = coupling.sites[0].dim
    return coupling.to_tensor().to_numpy(labels).reshape(dim**L, dim**L)


def make_WI_np(dt, A, B, C, D):
    assert all(T.ndim == 4 for T in [A, B, C, D])
    Id = np.eye(2)[None, None, :, :]
    W = np.zeros((1 + A.shape[0], 1 + A.shape[1], 2, 2), complex)
    W[0:1, 0:1] = Id + dt * D
    W[0:1, 1:] = C
    W[1:, 0:1] = dt * B
    W[1:, 1:] = A
    return W


def make_WII_np(dt, A, B, C, D):
    assert all(T.ndim == 4 for T in [A, B, C, D])
    B = B[:, 0]
    C = C[0, :]
    D = D[0, 0]

    # Implementation from v1:

    tC = np.sqrt(np.abs(dt))  # spread time step across B, C
    tB = dt / tC
    d = D.shape[0]

    # The virtual size of W is  (1+Nr, 1+Nc)
    Nr = A.shape[0]
    Nc = A.shape[1]
    W = np.zeros((1 + Nr, 1 + Nc, d, d), dtype=np.result_type(D, dt))

    Id_ = np.array([[1, 0], [0, 1]])  # 2x2 operators in a hard-core boson space
    b = np.array([[0, 0], [1, 0]])

    Id = np.kron(Id_, Id_)  # 4x4 operators in the 2x hard core boson space
    Br = np.kron(b, Id_)
    Bc = np.kron(Id_, b)
    Brc = np.kron(b, b)
    for r in range(Nr):  # double loop over row / column of A
        for c in range(Nc):
            # Select relevant part of virtual space and extend by hardcore bosons
            h = (
                np.kron(Brc, A[r, c, :, :])
                + np.kron(Br, tB * B[r, :, :])
                + np.kron(Bc, tC * C[c, :, :])
                + dt * np.kron(Id, D)
            )
            w = spla.expm(h)  # Exponentiate in the extended Hilbert space
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 1 + c, :, :] = w[1, 1]  # extracts relevant parts according to Eqn 11
            if c == 0:
                W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 1 + c] = w[0, 1]
                if c == 0:
                    W[0, 0] = w[0, 0]
        if Nc == 0:  # technically only need one boson
            h = np.kron(Br, tB * B[r, :, :]) + t * np.kron(Id, D)
            w = spla.expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 0] = w[0, 0]
    if Nr == 0:
        for c in range(Nc):
            h = np.kron(Bc, tC * C[c, :, :]) + t * np.kron(Id, D)
            w = spla.expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[0, 1 + c] = w[0, 1]
            if c == 0:
                W[0, 0] = w[0, 0]
        if Nc == 0:
            W = spla.expm(dt * D).reshape([1, 1, d, d])
    return W


def make_W_np_v1(dt, H_W: list[np.ndarray], approximation: str, finite: bool) -> list[np.ndarray]:
    """make WI or WII from W of H_MPO, like in v1"""

    # ensure IdL / IdR as expected
    eye = np.eye(2)

    if finite:
        assert H_W[0].shape[0] == 1
        assert H_W[-1].shape[1] == 1

    W_U = []
    for i, W in enumerate(H_W):  # W[wL, wR, p, p*]
        if finite and i == 0:
            # W = [[1 C D]]
            assert np.allclose(W[0, 0], eye)
            C = W[0, 1:-1][None, :, :, :]
            D = W[0, -1][None, None, :, :]
            A = np.zeros((0, C.shape[1], 2, 2))
            B = np.zeros((0, D.shape[1], 2, 2))
        elif finite and i == len(H_W) - 1:
            # W =[[D] [B] [1]]
            D = W[0, -1][None, None, :, :]
            B = W[1:-1, -1][:, None, :, :]
            assert np.allclose(W[-1, -1], eye)
            A = np.zeros((B.shape[0], 0, 2, 2))
            C = np.zeros((D.shape[0], 0, 2, 2))
        else:
            # extract blocks such that
            # W = [[1 C D]
            #      [0 A B]
            #      [0 0 1]]
            assert np.allclose(W[0, 0], eye)
            C = W[0, 1:-1][None, :, :, :]
            D = W[0, -1][None, None, :, :]
            assert np.allclose(W[1:-1, 0], 0)
            A = W[1:-1, 1:-1]
            B = W[1:-1, -1][:, None, :, :]
            assert np.allclose(W[-1, :-1], 0)
            assert np.allclose(W[-1, -1], eye)

        if approximation == 'I':
            W_U.append(make_WI_np(dt, A, B, C, D))
        elif approximation == 'II':
            W_U.append(make_WII_np(dt, A, B, C, D))
        else:
            raise NotImplementedError
    return W_U


def _roll_IdR_position(W, wL=True, wR=True):
    """Map between conventions [IdL, IdR, *other] -> [IdL, *others, IdR] for order of W-blocks"""
    dL, dR, _, _ = W.shape
    # [IdL, IdR, *other] -> [IdL, *others, IdR]
    if wL and wR:
        res = W[*np.ix_([0, *range(2, dL), 1], [0, *range(2, dR), 1]), :, :]
    elif wL:
        res = W[[0, *range(2, dL), 1], :, :, :]
    elif wR:
        res = W[:, [0, *range(2, dR), 1], :, :]
    else:
        res = W
    assert res.shape == W.shape
    return res


def make_W_np(dt, H: mpo.MPO, approximation: str, finite: bool) -> np.ndarray:
    """Make pure-numpy WI / WII tensors from the Hamiltonian MPO"""
    # transpose to v1 leg order
    assert all(W.labels == ['wL', 'p', 'wR', 'p*'] for W in H._W)
    H_W = [W.to_numpy().transpose([0, 2, 1, 3]) for W in H._W]  # transpose to tenpy v1 leg order

    # ensure row/column ordering within MPO tensors
    has_v1_state_ordering = True
    has_lexsorted_state_ordering = True
    for i, W in enumerate(H._W):
        labelsL = W.get_leg('wL').summand_labels
        if i == 0 and finite:
            has_v1_state_ordering &= labelsL == ['IdL']
            has_lexsorted_state_ordering &= labelsL == ['IdL']
        else:
            has_v1_state_ordering &= labelsL[0] == 'IdL' and labelsL[-1] == 'IdR'
            has_lexsorted_state_ordering &= labelsL[0] == 'IdL' and labelsL[1] == 'IdR'

        labelsR = W.get_leg('wR').summand_labels
        if i == H.L - 1 and finite:
            has_v1_state_ordering &= labelsR == ['IdR']
            has_lexsorted_state_ordering &= labelsR == ['IdR']
        else:
            has_v1_state_ordering &= labelsR[0] == 'IdL' and labelsR[-1] == 'IdR'
            has_lexsorted_state_ordering &= labelsR[0] == 'IdL' and labelsR[1] == 'IdR'

    if has_v1_state_ordering:
        pass
    elif has_lexsorted_state_ordering:
        H_W = [_roll_IdR_position(W, wL=not finite or i > 0, wR=not finite or i < H.L - 1) for i, W in enumerate(H_W)]
    else:
        raise NotImplementedError

    op_norms = [np.linalg.norm(W, axis=(2, 3)) for W in H_W]  # for debugging
    U_W = make_W_np_v1(dt, H_W, approximation=approximation, finite=finite)

    # transpose back to v2 leg order
    U_W = [W.transpose([0, 2, 1, 3]) for W in U_W]

    return U_W


@pytest.mark.parametrize('conserve', [None, 'best'])
@pytest.mark.parametrize('backend', ['fusion_tree', 'abelian', 'no_symmetry'])
@pytest.mark.parametrize('approximation', ['I', 'II'])
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_MPO_make_U(conserve, backend, approximation, bc_MPS, L=6, dt=0.01):
    """Compare MPO.make_U vs a pure-numpy implementation

    This pure-numpy implementation is also run in the v1 tests, and thus trusted
    """
    if conserve == 'best' and backend == 'no_symmetry':
        with pytest.raises(ValueError, match='backend does not support this symmetry'):
            model_params = asConfig(
                dict(L=L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.2, bc_MPS=bc_MPS, conserve=conserve, symmetry_backend=backend),
                'SpinChain',
            )
            model_params.touch_all()
            M = SpinChain(model_params)
        return  # error is expected behavior

    M = SpinChain(dict(L=L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.2, bc_MPS=bc_MPS, conserve=conserve, symmetry_backend=backend))

    if approximation == 'II':
        with pytest.raises(NotImplementedError):
            _ = M.H_MPO.make_U(-1j * dt, approximation=approximation)
        pytest.skip('WII not ported yet')

    U_MPO: mpo.MPO = M.H_MPO.make_U(-1j * dt, approximation=approximation)
    U_MPO.test_sanity()
    assert U_MPO.L == L

    # compare against pure-numpy construction of the W tensors
    W_U_expect_np = make_W_np(-1j * dt, M.H_MPO, approximation=approximation, finite=bc_MPS == 'finite')
    for W, expect_np in zip(U_MPO._W, W_U_expect_np):
        W_np = W.to_numpy()
        npt.assert_almost_equal(W_np, expect_np)

    # check error bound
    if bc_MPS == 'finite':
        ED_H = ExactDiag.from_H_mpo(M.H_MPO)
        ED_H.build_full_H_from_mpo()
        ED_H.full_diagonalization()
        U_exact = ED_H.exp_H(dt).to_numpy()

        ED_U = ExactDiag.from_H_mpo(U_MPO)
        ED_U.build_full_H_from_mpo()
        U_np = ED_U.full_H.to_numpy()

        err = np.linalg.norm(U_np - U_exact)
        assert err < L * (dt**2)
