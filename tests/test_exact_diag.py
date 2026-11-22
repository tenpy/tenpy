"""A collection of tests to check the functionality of algorithms.exact_diagonalization."""

# Copyright (C) TeNPy Developers, Apache license
from functools import reduce

import lasyten as lasy
import numpy as np
import pytest
from lasyten.models.sites import FibonacciAnyonSite, SpinSite
from lasyten.symmetries import SymmetryError
from lasyten.symmetries.spaces import AbelianLegPipe
from lasyten.tensors import SymmetricTensor, compose, dagger, norm
from scipy.linalg import expm

from tenpy.algorithms import exact_diag
from tenpy.models.spins import SpinChain
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS


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
    """The Heisenberg chain as a lasyten tensor, together with its dense reference."""
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

    Unlike :func:`_assert_eigenvector`, this works for a :class:`~lasyten.tensors.ChargedTensor`
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
    ground_sector = np.asarray(psi0.get_leg('!slice').dual.sector_decomposition)[0]
    assert np.any(np.all(sectors == ground_sector[np.newaxis, :], axis=1))
    _assert_eigenvector(H, E0, psi0)


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
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
        np.testing.assert_array_equal(np.asarray(psi0.get_leg('!slice').dual.sector_decomposition)[0], sector)
        _assert_eigenvector(H, E0, psi0)

        # cross-check the selection against the full spectrum, without calling groundstate() again
        mask = np.all(sectors_of_basis == sector[np.newaxis, :], axis=1)
        np.testing.assert_allclose(E0, np.min(ED.E[mask]), atol=1e-10)

        if site.leg.symmetry.is_abelian:
            # fully independent of lasyten's eigh: a hand-picked dense sub-block of H_ref
            idx = _kron_charge_indices(site, L, sector)
            expected_E0 = np.min(np.linalg.eigvalsh(H_ref[np.ix_(idx, idx)]))
            np.testing.assert_allclose(E0, expected_E0, atol=1e-10)

        E0s.append(E0)

    np.testing.assert_allclose(np.min(E0s), np.min(np.linalg.eigvalsh(H_ref)), atol=1e-10)


# the SU(2) Clebsch-Gordans go through sympy, which trips over a deprecation in mpmath
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
        np.testing.assert_array_equal(np.asarray(psi0.get_leg('!slice').dual.sector_decomposition)[0], sector)

        # a charge_sector was already fixed at construction -- can't ask for another one
        with pytest.raises(ValueError, match='specified before'):
            ED.groundstate(sector)


@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_mps_roundtrip(conserve, L):
    if conserve == 'Sz' and L == 3:
        pytest.skip('no charge-neutral sector for Sz, L=3; from_full needs neutral states')
    site = SpinSite(0.5, conserve=conserve)
    H, _ = _heisenberg_tensor(site, L)
    sites = [site] * L
    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    ED.full_diagonalization()

    # a charge-neutral random state in the p0..p{L-1} basis (nonzero only if a neutral sector exists)
    psi_full = SymmetricTensor.from_random_uniform(
        [site.leg] * L, [], labels=[[f'p{i}' for i in range(L)], []], dtype=lasy.Dtype.float64
    )
    psi_full = psi_full / norm(psi_full)
    mps = ED.full_to_mps(psi_full)
    assert mps.bc == 'finite'
    psi_back = ED.mps_to_full(mps)
    np.testing.assert_allclose(
        psi_back.to_numpy().reshape(-1),
        psi_full.to_numpy().reshape(-1),
        atol=1e-10,
    )


@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
@pytest.mark.parametrize('L', [3, 4])
def test_exact_diag_exp_H(conserve, L):

    site = SpinSite(0.5, conserve=conserve)
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L
    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    ED.full_diagonalization()
    dt = 0.1
    U = ED.exp_H(dt)
    # match lasyten's block convention: domain axes reversed (as in _heisenberg_tensor)
    expect_block = np.transpose(
        np.reshape(expm(-1.0j * dt * H_ref), [2] * (2 * L)), [*range(L), *reversed(range(L, 2 * L))]
    )
    np.testing.assert_allclose(U.to_numpy(), expect_block, atol=1e-10)


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
    # ExactDiag needs a dense basis (argmin, ChargedTensor, ...), which lasyten does not define for
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
def test_exact_diag_sparse_diag_su2(L=3):
    site = SpinSite(0.5, conserve='SU(2)')
    H, H_ref = _heisenberg_tensor(site, L)
    sites = [site] * L

    ED = exact_diag.ExactDiag.from_hamiltonian(H, sites)
    E, psi = ED.sparse_diag(1)
    np.testing.assert_allclose(E[0], np.min(np.linalg.eigvalsh(H_ref)), atol=1e-8)
    _assert_eigenvector_dense(H_ref, E[0], psi[0])

    # lasyten does not yet support restricting to a single, higher-dimensional (non-abelian) sector
    sector = ED.possible_charge_sectors()[0]
    ED_sec = exact_diag.ExactDiag.from_hamiltonian(H, sites, charge_sector=sector)
    with pytest.raises(NotImplementedError):
        ED_sec.sparse_diag(1)


@pytest.mark.parametrize(
    'conserve, symmetry_backend',
    [
        ('SU2', 'fusion_tree'),
        ('Sz', 'fusion_tree'),
        ('Sz', 'abelian'),
        (None, 'fusion_tree'),
        (None, 'abelian'),
        (None, 'no_symmetry'),
    ],
)
def test_ED(conserve, symmetry_backend):
    # just quickly check that it runs without errors for a small system
    base_params = dict(
        Jx=1.0, Jy=1.0, Jz=1.0, hz=0 if conserve == 'SU2' else 0.1, conserve=conserve, symmetry_backend=symmetry_backend
    )
    M = SpinChain(dict(**base_params, L=4, bc_MPS='finite'))

    ED = exact_diag.ExactDiag.from_model(M)
    ED.build_full_H_from_mpo()
    H = ED.full_H

    ED = exact_diag.ExactDiag.from_model(M)
    ED.build_full_H_from_bonds()
    H2 = ED.full_H
    assert lasy.almost_equal(H, H2)

    ED.full_diagonalization()
    E, psi = ED.groundstate()
    assert psi.get_leg('!slice').sector_decomposition == lasy.SectorArray([M.symmetry.trivial_sector])

    if conserve == 'Sz':
        target_sector = lasy.Sector([2])
        E_sec2, psi_sec2 = ED.groundstate(target_sector)
        charge = M.symmetry.dual_sectors(psi_sec2.get_leg('!slice').sector_decomposition)
        assert charge == lasy.SectorArray([target_sector])

    ED2 = exact_diag.ExactDiag.from_model(M, M.symmetry.trivial_sector)
    ED2.build_full_H_from_mpo()
    ED2.full_diagonalization()
    E2, psi2 = ED2.groundstate()
    ov = lasy.inner(psi2, psi)
    assert abs(ov) == pytest.approx(1)

    # compare with scipy sparse eigensolver
    np.random.seed(12345)
    psi3 = lasy.SymmetricTensor.from_random_normal(ED2.full_H.codomain, [], backend=H.backend)
    op = lasy.tensors.HermitianNumpyArrayLinearOperator.from_Tensor(
        ED2.full_H, legs1=ED2._labels_pconj, legs2=ED2._labels_p, charge_sector='trivial'
    )

    if symmetry_backend == 'fusion_tree':
        with pytest.raises(NotImplementedError, match='to_dense_block_trivial_sector not implemented'):
            _ = op.eigenvectors(num_ev=1, which='SA', v0_tensor=psi3)
        pytest.xfail('to_dense_block_trivial_sector not implemented')

    (E3,), (psi3,) = op.eigenvectors(num_ev=1, which='SA', v0_tensor=psi3)
    assert E3 == pytest.approx(E)
    ov = lasy.inner(psi3, lasy.squeeze_legs(psi2.as_SymmetricTensor()))
    assert abs(ov) == pytest.approx(1)

    ED3 = exact_diag.ExactDiag.from_H_mpo(M.get_H_MPO())
    ED3.build_full_H_from_mpo()
    assert lasy.almost_equal(ED3.full_H, H)

    M_inf = SpinChain(dict(**base_params, L=2, bc_MPS='infinite'))

    with pytest.raises(AttributeError, match="has no attribute 'extract_segment'"):
        _ = exact_diag.ExactDiag.from_infinite_model(M_inf, enlarge=2)
    # https://github.com/tenpy/lasyten/issues/339
    pytest.xfail('Extract segment not ported. #339')

    ED4 = exact_diag.ExactDiag.from_infinite_model(M_inf, enlarge=2)
    ED4.build_full_H_from_mpo()
    assert lasy.almost_equal(ED4.full_H, H)


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


@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_full_wavefunction(conserve, L=10):
    # check with a singlet covering
    # sign convention of singlet = (|up,down> - |down,up>) / sqrt(2)
    assert L % 2 == 0
    assert L % 4 == 2  # only for an odd number of singlets do we detect mixing up the basis order

    # build wavefunction exactly
    up_down_basis = False
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
    site = SpinSite(conserve='Sz' if conserve == 'best' else conserve)
    psi = MPS.from_singlets(site=site, L=L, pairs=[[i, i + 1] for i in range(0, L, 2)], unit_cell_width=L)
    res = exact_diag.get_full_wavefunction(psi)

    # compare
    assert np.allclose(res, expect)


@pytest.mark.parametrize('conserve', ['best', 'None'])
def test_get_scipy_sparse_Hamiltonian(conserve, L=10, J=1, g=4.3291):
    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=False)
    H_res = exact_diag.get_scipy_sparse_Hamiltonian(model)
    assert np.allclose(H_res.toarray(), H_expect)


@pytest.mark.parametrize('conserve', ['best', 'None'])
@pytest.mark.parametrize('use_ED', [True, False])
def test_get_numpy_Hamiltonian(conserve, use_ED, J=1, g=4.3291):
    L = 6 if use_ED else 10  # using ED is a bit slow (overhead from combine? or is np.ix_ slow?)
    model = TFIChain(dict(L=L, conserve=conserve, J=J, g=g))
    H_expect = get_tfi_Hamiltonian(L=L, J=J, g=g, up_down_basis=False)
    if use_ED:
        # default behavior for this model is from couplings. test from full_H explicitly.
        H_res = exact_diag._get_numpy_Hamiltonian_ExactDiag_full_H(model, from_mpo=True)
    else:
        H_res = exact_diag.get_numpy_Hamiltonian(model)
    assert np.allclose(H_res, H_expect)


def _dense_tfi(L, J, g):
    """Transverse field Ising chain ``-J sum sx sx - g sum sz`` as a dense matrix, in the (down, up) basis."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.diag([-1.0, 1.0])
    eye = np.eye(2)
    H = np.zeros((2**L, 2**L))
    for i in range(L):
        ops = [eye] * L
        ops[i] = sz
        H = H - g * reduce(np.kron, ops)
        if i < L - 1:
            ops = [eye] * L
            ops[i], ops[i + 1] = sx, sx
            H = H - J * reduce(np.kron, ops)
    return H


def _dense_to_tensor(site, H_ref, L):
    block = np.transpose(np.reshape(H_ref, [2] * (2 * L)), [*range(L), *reversed(range(L, 2 * L))])
    return SymmetricTensor.from_dense_block(
        block,
        codomain=[site.leg] * L,
        domain=[site.leg] * L,
        labels=[[f'p{i}' for i in range(L)], [f'p{i}*' for i in range(L)]],
    )


def _check_built_models(M, H_ref, H_expect):
    """Both builders reproduce `H_expect` and agree with `H_ref` after full diagonalization."""
    full = {}
    for method in ['build_full_H_from_mpo', 'build_full_H_from_bonds']:
        ED = exact_diag.ExactDiag.from_model(M)
        getattr(ED, method)()
        full[method] = ED.full_H
        assert ED.full_H.codomain == H_expect.codomain
        assert ED.full_H.domain == H_expect.domain
        assert ED.full_H.labels == H_expect.labels
        assert lasy.almost_equal(ED.full_H, H_expect)
        ED.full_diagonalization()
        np.testing.assert_allclose(np.sort(ED.E), np.sort(np.linalg.eigvalsh(H_ref)), atol=1e-10)
        np.testing.assert_allclose(np.min(ED.E), np.linalg.eigvalsh(H_ref)[0], atol=1e-10)
        E0, _ = ED.groundstate()
        np.testing.assert_allclose(E0, np.linalg.eigvalsh(H_ref)[0], atol=1e-10)
    assert lasy.almost_equal(full['build_full_H_from_mpo'], full['build_full_H_from_bonds'])


@pytest.mark.parametrize('conserve', [None, 'parity', 'Sz', 'SU2'])
@pytest.mark.parametrize('L', [3, 4])
def test_build_full_H_heisenberg(conserve, L):
    M = SpinChain(dict(L=L, bc_MPS='finite', conserve=conserve))
    H_expect, H_ref = _heisenberg_tensor(M.lattice.mps_sites()[0], L)
    _check_built_models(M, H_ref, H_expect)


@pytest.mark.parametrize('conserve', [None, 'parity'])
@pytest.mark.parametrize('L', [3, 4])
def test_build_full_H_tfi(conserve, L):
    M = TFIChain(dict(L=L, J=1.0, g=1.2, bc_MPS='finite', conserve=conserve))
    site = M.lattice.mps_sites()[0]
    assert site.state_labels['down'] == 0 and site.state_labels['up'] == 1
    H_ref = _dense_tfi(L, 1.0, 1.2)
    _check_built_models(M, H_ref, _dense_to_tensor(site, H_ref, L))


@pytest.mark.parametrize(
    'conserve, dense',
    [(None, False), ('parity', False), ('Sz', False), ('SU2', True)],
)
def test_assembly_route(conserve, dense):
    M = SpinChain(dict(L=3, bc_MPS='finite', conserve=conserve))
    ED = exact_diag.ExactDiag.from_model(M)
    assert ED._use_dense_assembly() == dense
    assert ED.symmetry.is_abelian == (not dense)


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_build_full_H_explicit_plus_hc(conserve, L=3):
    M = SpinChain(dict(L=L, bc_MPS='finite', conserve=conserve))
    H_expect, H_ref = _heisenberg_tensor(M.lattice.mps_sites()[0], L)
    M.H_MPO.explicit_plus_hc = True  # the Heisenberg MPO is hermitian, so H + h.c. = 2 H
    ED = exact_diag.ExactDiag.from_model(M)
    ED.build_full_H_from_mpo()
    assert lasy.almost_equal(ED.full_H, 2 * H_expect)


def test_dense_H_from_mpo_matches_reference():
    M = SpinChain(dict(L=4, bc_MPS='finite', conserve=None))
    H = exact_diag.ExactDiag._dense_H_from_mpo(M.H_MPO)
    np.testing.assert_allclose(H, _dense_heisenberg(4), atol=1e-12)
