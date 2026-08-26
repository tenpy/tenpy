"""A collection of tests to check the functionality of algorithms.exact_diagonalization."""

# Copyright (C) TeNPy Developers, Apache license
import copy
from functools import reduce

import numpy as np
import pytest
from cyten.models.sites import FibonacciAnyonSite, SpinSite
from cyten.symmetries import SymmetryError
from cyten.symmetries.spaces import AbelianLegPipe
from cyten.tensors import SymmetricTensor, compose, dagger, norm

from tenpy.algorithms import exact_diag

_LEGACY = 'legacy np_conserved test, not migrated to cyten yet'


def _dense_heisenberg(L):
    """Nearest-neighbour Heisenberg chain as a dense matrix, in the (down, up) Kronecker basis."""
    Sz = np.diag([-0.5, 0.5])
    Sp = np.array([[0.0, 0.0], [1.0, 0.0]])  # |up><down|
    Sm = Sp.T
    eye = np.eye(2)
    H = np.zeros((2**L, 2**L))
    for i in range(L - 1):
        for left, right in [(Sz, Sz), (0.5 * Sp, Sm), (0.5 * Sm, Sp)]:
            ops = [eye] * L
            ops[i], ops[i + 1] = left, right
            H = H + reduce(np.kron, ops)
    return H


def _heisenberg_tensor(site, L):
    """The Heisenberg chain as a cyten tensor, together with its dense reference."""
    assert site.state_labels['down'] == 0 and site.state_labels['up'] == 1  # basis of _dense_heisenberg
    H_ref = _dense_heisenberg(L)
    # the dense block wants the leg order [p0, ..., p{L-1}, p{L-1}*, ..., p0*]
    block = np.transpose(np.reshape(H_ref, [2] * (2 * L)), [*range(L), *reversed(range(L, 2 * L))])
    H = SymmetricTensor.from_dense_block(
        block,
        codomain=[site.leg] * L,
        domain=[site.leg] * L,
        labels=[[f'p{i}' for i in range(L)], [f'p{i}*' for i in range(L)]],
    )
    return H, H_ref


def _assert_eigenvector(H, E, psi_inv):
    """Check that ``psi_inv`` (e.g. a ``ChargedTensor.invariant_part``) solves ``H psi = E psi``."""
    scale = norm(psi_inv).as_float64()
    assert scale > 0
    resid = norm(compose(H, psi_inv) - psi_inv * E).as_float64()
    assert resid < 1e-8 * scale


def _assert_eigenvector_dense(H_ref, E, psi):
    """Check ``H_ref @ vec = E * vec`` for the dense vector represented by `psi`.

    Unlike :func:`_assert_eigenvector`, this works for a :class:`~cyten.tensors.ChargedTensor`
    with a `charged_state` set (as returned by :meth:`ExactDiag.sparse_diag`), where
    ``psi.invariant_part`` alone is *not* the eigenvector.
    """
    vec = psi.to_numpy().reshape(-1)
    scale = np.linalg.norm(vec)
    assert scale > 0
    resid = np.linalg.norm(H_ref @ vec - E * vec)
    assert resid < 1e-8 * scale


def _kron_charge_indices(site, L, sector):
    """Kronecker-basis indices whose total charge is `sector`.

    Computed by hand from the sites' basis charges, independent of :class:`ExactDiag`.
    """
    pipe = AbelianLegPipe([site.leg] * L)
    sectors_of_basis = np.asarray(pipe.sectors_of_basis)
    return np.flatnonzero(np.all(sectors_of_basis == np.asarray(sector)[np.newaxis, :], axis=1))


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None', 'SU(2)'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_full(conserve, L):
    site = SpinSite(0.5, conserve=conserve)
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    ED.full_diagonalization()
    np.testing.assert_allclose(np.sort(ED.E), np.sort(np.linalg.eigvalsh(H_ref)), atol=1e-10)

    E0, psi0 = ED.groundstate()
    np.testing.assert_allclose(E0, np.min(np.linalg.eigvalsh(H_ref)), atol=1e-10)
    sectors = ED.possible_charge_sectors()
    ground_sector = np.asarray(psi0.charge_leg.sector_decomposition)[0]
    assert np.any(np.all(sectors == ground_sector[np.newaxis, :], axis=1))
    _assert_eigenvector(H, E0, psi0.invariant_part)


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None', 'SU(2)'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_groundstate_per_sector(conserve, L):
    site = SpinSite(0.5, conserve=conserve)
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    ED.full_diagonalization()
    sectors_of_basis = np.asarray(ED.V.domain.factors[0].sectors_of_basis)

    E0s = []
    for sector in ED.possible_charge_sectors():
        E0, psi0 = ED.groundstate(sector)
        np.testing.assert_array_equal(np.asarray(psi0.charge_leg.sector_decomposition)[0], sector)
        _assert_eigenvector(H, E0, psi0.invariant_part)

        # cross-check the selection against the full spectrum, without calling groundstate() again
        mask = np.all(sectors_of_basis == sector[np.newaxis, :], axis=1)
        np.testing.assert_allclose(E0, np.min(ED.E[mask]), atol=1e-10)

        if site.leg.symmetry.is_abelian:
            # fully independent of cyten's eigh: a hand-picked dense sub-block of H_ref
            idx = _kron_charge_indices(site, L, sector)
            expected_E0 = np.min(np.linalg.eigvalsh(H_ref[np.ix_(idx, idx)]))
            np.testing.assert_allclose(E0, expected_E0, atol=1e-10)

        E0s.append(E0)

    np.testing.assert_allclose(np.min(E0s), np.min(np.linalg.eigvalsh(H_ref)), atol=1e-10)


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None', 'SU(2)'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_charge_sector(conserve, L):
    site = SpinSite(0.5, conserve=conserve)
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED_all = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    ED_all.full_diagonalization()

    for sector in ED_all.possible_charge_sectors():
        expected_E0, _ = ED_all.groundstate(sector)

        ED = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
        ED.full_diagonalization()
        E0, psi0 = ED.groundstate()
        np.testing.assert_allclose(E0, expected_E0, atol=1e-10)
        np.testing.assert_array_equal(np.asarray(psi0.charge_leg.sector_decomposition)[0], sector)

        # a charge_sector was already fixed at construction -- can't ask for another one
        with pytest.raises(ValueError, match='specified before'):
            ED.groundstate(sector)


@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_mask(conserve, L):
    site = SpinSite(0.5, conserve=conserve)
    H, _ = _heisenberg_tensor(site, L)
    sites = [site] * L

    for sector in exact_diag.ExactDiag.from_hamiltonian(H, sites).possible_charge_sectors():
        ED = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
        idx = _kron_charge_indices(site, L, sector)
        np.testing.assert_array_equal(np.flatnonzero(ED._mask), np.sort(idx))


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
def test_exact_diag_mask_su2(L=3):
    # a per-basis-state charge doesn't exist for a non-abelian symmetry, so _mask stays None
    site = SpinSite(0.5, conserve='SU(2)')
    H, _ = _heisenberg_tensor(site, L)
    sites = [site] * L
    sector = exact_diag.ExactDiag.from_hamiltonian(H, sites).possible_charge_sectors()[0]
    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
    assert ED._mask is None


def test_exact_diag_errors(L=3):
    site = SpinSite(0.5, conserve='Sz')
    H, _ = _heisenberg_tensor(site, L)
    sites = [site] * L

    with pytest.raises(ValueError, match='empty'):
        exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=[100])  # not a possible sector

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    with pytest.raises(ValueError, match='full_diagonalization'):
        ED.groundstate()

    ED.full_diagonalization()
    with pytest.raises(ValueError, match='empty'):
        ED.groundstate([100])


def test_exact_diag_anyons(L=3):
    # ExactDiag needs a dense basis (argmin, ChargedTensor, ...), which cyten does not define for
    # anyons -- so it refuses the symmetry outright, right at construction.
    site = FibonacciAnyonSite()
    leg = site.leg
    A = SymmetricTensor.from_random_uniform(
        [leg] * L, [leg] * L, labels=[[f'p{i}' for i in range(L)], [f'p{i}*' for i in range(L)]]
    )
    H = compose(A, dagger(A))  # Hermitian and positive semi-definite by construction
    sites = [site] * L

    with pytest.raises(SymmetryError):
        exact_diag.ExactDiag.from_hamiltonian(H, sites)


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_sparse_diag(conserve, L):
    site = SpinSite(0.5, conserve=conserve)
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    E, psi = ED.sparse_diag(1)
    np.testing.assert_allclose(E[0], np.min(np.linalg.eigvalsh(H_ref)), atol=1e-8)
    assert psi[0].labels == [f'p{i}' for i in range(L)]
    _assert_eigenvector_dense(H_ref, E[0], psi[0])

    for sector in ED.possible_charge_sectors():
        ED_sec = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
        E_sec, psi_sec = ED_sec.sparse_diag(1)
        idx = _kron_charge_indices(site, L, sector)
        expected_E0 = np.min(np.linalg.eigvalsh(H_ref[np.ix_(idx, idx)]))
        np.testing.assert_allclose(E_sec[0], expected_E0, atol=1e-8)
        assert psi_sec[0].labels == [f'p{i}' for i in range(L)]
        _assert_eigenvector_dense(H_ref, E_sec[0], psi_sec[0])


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
@pytest.mark.filterwarnings('ignore:bitcount function is deprecated:DeprecationWarning')
def test_exact_diag_sparse_diag_su2(L=3):
    site = SpinSite(0.5, conserve='SU(2)')
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    E, psi = ED.sparse_diag(1)
    np.testing.assert_allclose(E[0], np.min(np.linalg.eigvalsh(H_ref)), atol=1e-8)
    _assert_eigenvector_dense(H_ref, E[0], psi[0])

    # cyten does not yet support restricting to a single, higher-dimensional (non-abelian) sector
    sector = ED.possible_charge_sectors()[0]
    ED_sec = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
    with pytest.raises(NotImplementedError):
        ED_sec.sparse_diag(1)


def test_ED():
    pytest.skip(_LEGACY)
    import tenpy.linalg.np_conserved as npc
    from tenpy.linalg.krylov_based import LanczosGroundState

    from tenpy.models import XXZChain

    # just quickly check that it runs without errors for a small system
    xxz_pars = dict(L=4, Jxx=1.0, Jz=1.0, hz=0.1, bc_MPS='finite', sort_charge=True)
    M = XXZChain(xxz_pars)
    ED = exact_diag.ExactDiag(M)
    ED.build_full_H_from_mpo()
    H, ED.full_H = ED.full_H, None
    ED.build_full_H_from_bonds()
    H2 = ED.full_H
    assert npc.norm(H - H2, np.inf) < 1.0e-14
    ED.full_diagonalization()
    E, psi = ED.groundstate()
    print('select charge_sector =', psi.qtotal)
    assert np.all(psi.qtotal == [0])
    E_sec2, psi_sec2 = ED.groundstate([2])
    assert np.all(psi_sec2.qtotal == [2])
    ED2 = exact_diag.ExactDiag(M, psi.qtotal)
    ED2.build_full_H_from_mpo()
    ED2.full_diagonalization()
    E2, psi2 = ED2.groundstate()
    full_psi2 = psi.zeros_like()
    full_psi2[ED2._mask] = psi2
    ov = npc.inner(psi, full_psi2, 'range', do_conj=True)
    print('overlap <psi | psi2> = 1. -', 1.0 - ov)
    assert abs(abs(ov) - 1.0) < 1.0e-15
    # starting from a random guess in the correct charge sector,
    # check if we can also do lanczos.
    np.random.seed(12345)
    psi3 = npc.Array.from_func(np.random.random, psi2.legs, qtotal=psi2.qtotal, shape_kw='size')
    E0, psi3, N = LanczosGroundState(ED2, psi3, {}).run()
    print('Lanczos E0 =', E0)
    ov = npc.inner(psi3, psi2, 'range', do_conj=True)
    print('overlap <psi2 | psi3> = 1. -', 1.0 - ov)
    assert abs(abs(ov) - 1.0) < 1.0e-15

    ED3 = exact_diag.ExactDiag.from_H_mpo(M.H_MPO)
    ED3.build_full_H_from_mpo()
    assert npc.norm(ED3.full_H - H, np.inf) < 1.0e-14

    xxz_pars_inf = copy.copy(xxz_pars)
    xxz_pars_inf['bc_MPS'] = 'infinite'
    xxz_pars_inf['L'] = 2
    M_inf = XXZChain(xxz_pars_inf)
    ED4 = exact_diag.ExactDiag.from_infinite_model(M_inf, enlarge=2)
    ED4.build_full_H_from_mpo()
    assert npc.norm(ED4.full_H - H, np.inf) < 1.0e-14


def get_tfi_Hamiltonian(L, J, g, up_down_basis=True):
    if up_down_basis:
        sz = np.array([[1, 0], [0, -1]], float)
    else:
        sz = np.array([[-1, 0], [0, 1]], float)
    sx = np.array([[0, 1], [1, 0]], float)
    eye = np.eye(2, dtype=float)
    ops = [eye] * L
    H_expect = 0
    for i in range(L):
        ops = [eye] * L
        ops[i] = sz
        H_expect = H_expect - g * reduce(np.kron, ops)
    for i in range(L - 1):
        ops = [eye] * L
        ops[i] = ops[i + 1] = sx
        H_expect = H_expect - J * reduce(np.kron, ops)
    return H_expect


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_full_wavefunction(undo_sort_charge, conserve, L=10):
    pytest.skip(_LEGACY)
    from tenpy.networks import MPS, SpinHalfSite

    # check with a singlet covering
    # sign convention of singlet = (|up,down> - |down,up>) / sqrt(2)
    assert L % 2 == 0
    assert L % 4 == 2  # only for an odd number of singlets do we detect mixing up the basis order

    # build wavefunction exactly
    up_down_basis = undo_sort_charge or conserve == 'None'
    singlet = np.zeros((2, 2))
    if up_down_basis:
        singlet[0, 1] = +1
        singlet[1, 0] = -1
    else:
        singlet[1, 0] = +1
        singlet[0, 1] = -1
    singlet = np.reshape(singlet, -1) / np.sqrt(2)
    expect = reduce(np.kron, [singlet] * (L // 2))

    # use get_full_wavefunction
    site = SpinHalfSite(conserve='Sz' if conserve == 'best' else conserve)
    psi = MPS.from_singlets(site=site, L=L, pairs=[[i, i + 1] for i in range(0, L, 2)], unit_cell_width=L)
    res = exact_diag.get_full_wavefunction(psi, undo_sort_charge=undo_sort_charge)

    # compare
    assert np.allclose(res, expect)


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_scipy_sparse_Hamiltonian(undo_sort_charge, conserve, L=10, J=1, g=4.3291):
    pytest.skip(_LEGACY)
    from tenpy.models import TFIChain

    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=undo_sort_charge or conserve == 'None')
    H_res = exact_diag.get_scipy_sparse_Hamiltonian(model, undo_sort_charge=undo_sort_charge)
    assert np.allclose(H_res.toarray(), H_expect)


@pytest.mark.parametrize('undo_sort_charge', [True, False])
@pytest.mark.parametrize('conserve', ['best', 'None'])
@pytest.mark.parametrize('use_ED', [True, False])
def test_get_numpy_Hamiltonian(undo_sort_charge, conserve, use_ED, J=1, g=4.3291):
    pytest.skip(_LEGACY)
    from tenpy.models import TFIChain

    L = 6 if use_ED else 10  # using ED is a bit slow (overhead from combine? or is np.ix_ slow?)
    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=undo_sort_charge or conserve == 'None')
    if use_ED:
        # default behavior for this model is from couplings. test from full_H explicitly.
        H_res = exact_diag._get_numpy_Hamiltonian_ExactDiag_full_H(
            model, from_mpo=True, undo_sort_charge=undo_sort_charge
        )
    else:
        H_res = exact_diag.get_numpy_Hamiltonian(model, undo_sort_charge=undo_sort_charge)
    assert np.allclose(H_res, H_expect)
