"""Tests for infinite MPS: canonical form, transfer matrix, correlation length.

The MPS are constructed with random tensors and checked against exact properties, e.g. of the AKLT state.
"""

# Copyright (C) TeNPy Developers, Apache license
import lasyten as lasy
import numpy as np
import numpy.testing as npt
import pytest
from lasyten.models import sites

from tenpy.models.aklt import AKLTChain
from tenpy.networks import mps
from tenpy.networks.mps import TransferMatrix


@pytest.fixture(autouse=True)
def _seed_numpy_rng():
    """The numpy backend of lasyten draws random numbers (e.g. the noise for Arnoldi) from the global numpy RNG."""
    np.random.seed(12345)


def _virtual_space(symmetry, weights_mults):
    """Virtual space with the given SU(2) sectors, given as ``(2 * spin, multiplicity)`` pairs."""
    return lasy.ElementarySpace.from_defining_sectors(
        symmetry,
        np.array([[w, 0] for w, _ in weights_mults]),
        multiplicities=[m for _, m in weights_mults],
    )


def _random_mps(conserve, seed=0, L=2, canonicalize=True):
    """Random infinite MPS (unit cell of `L` spin-1 sites) with the given symmetry.

    If not `canonicalize`, the random tensors are used as they are, i.e. *without* calling ``canonical_form``.
    """
    np.random.seed(seed)
    site = sites.SpinSite(1, conserve=conserve)
    sym = site.leg.symmetry
    if conserve == 'SU2':
        # bonds with spin 1/2 and 3/2
        V = _virtual_space(sym, [(1, 3), (3, 2)])
    elif conserve == 'Sz':
        V = lasy.ElementarySpace.from_defining_sectors(sym, np.array([[-2], [0], [2]]), multiplicities=[3, 6, 3])
    else:
        V = lasy.ElementarySpace.from_trivial_sector(dim=12, symmetry=sym)
    if canonicalize:
        return mps.MPS.from_desired_virtual_spaces(
            [site] * L, [V] * L, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=L
        )
    backend = lasy.backends.get_same_backend(site)
    Bs = [
        lasy.SymmetricTensor.from_random_uniform(
            lasy.TensorProduct([V, site.leg], sym),
            lasy.TensorProduct([V], sym),
            backend,
            ['vL', 'p', 'vR'],
            dtype=lasy.Dtype.float64,
        )
        for _ in range(L)
    ]
    SVs = [lasy.DiagonalTensor.from_eye(V, backend, ['vL', 'vR'], lasy.Dtype.float64) for _ in range(L + 1)]
    psi = mps.MPS([site] * L, Bs, SVs, bc='infinite', form='B', unit_cell_width=L)
    psi.form = [None] * L  # the tensors are not in any canonical form
    return psi


def _tm_spectrum(psi, num=3):
    """Magnitudes of the dominant eigenvalues of the transfer matrix in the trivial sector (any form)."""
    TM = TransferMatrix(psi, psi, 0, 0, transpose=False, charge_sector=0, form=None)
    eta, _ = TM.eigenvectors(num, v0=TM.initial_guess(1.0), which='LM', tol=1e-10, N_max=80)
    return np.abs(eta)


def _extra_space(psi, conserve, mult):
    """Extra bond space with a single sector (trivial / ``Sz=0`` / spin 1/2) with `mult` states."""
    sym = psi.sites[0].leg.symmetry
    sector = {None: [0], 'Sz': [0], 'SU2': [1, 0]}[conserve]
    return lasy.ElementarySpace.from_defining_sectors(sym, np.array([sector]), multiplicities=[mult])


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
@pytest.mark.parametrize('seed', [0, 1])
def test_canonical_form_infinite(conserve, seed):
    psi = _random_mps(conserve, seed, canonicalize=False)
    spectrum_random = _tm_spectrum(psi)
    psi.canonical_form()
    psi.test_sanity()
    assert psi.form == [(0.0, 1.0)] * 2
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    # a gauge transformation (and bond-dimension preserving projection) does not change the transfer matrix spectrum
    spectrum_canonical = _tm_spectrum(psi)
    npt.assert_allclose(
        spectrum_canonical[1:] / spectrum_canonical[0], spectrum_random[1:] / spectrum_random[0], atol=1e-8
    )
    # the singular values are normalized; calling it again does not change anything
    S_canonical = [S.diagonal_as_numpy().copy() for S in psi._S]
    psi.canonical_form()
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    for S0, S1 in zip(S_canonical, [S.diagonal_as_numpy() for S in psi._S]):
        npt.assert_allclose(S0, S1, atol=1e-7)


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
@pytest.mark.parametrize('L', [3, 4])
def test_canonical_form_infinite_unit_cell_width(conserve, L):
    # regression test: the sweep over the bonds of the unit cell failed for more than two sites
    psi = _random_mps(conserve, L, L=L, canonicalize=False)
    psi.canonical_form()
    psi.test_sanity()
    assert np.max(np.abs(psi.norm_test())) < 1e-12


def test_canonical_form_infinite_singular_values_normalized():
    psi = _random_mps('SU2', 3)
    for i in range(psi.L):
        S = psi.get_SL(i)
        # the trace is weighted with the quantum dimensions of the sectors
        assert lasy.trace(S**2) == pytest.approx(1.0, abs=1e-12)
        assert np.all(S.diagonal_as_numpy() > 0)


def test_aklt_state():
    psi = AKLTChain(dict(L=2, bc_MPS='infinite', conserve='SU2')).psi_AKLT()
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    # entanglement entropy of the AKLT state is ln(2): a singlet between two spin-1/2 (one per bond)
    npt.assert_allclose(psi.entanglement_entropy(), np.log(2), atol=1e-12)
    # <S_i . S_{i+1}> = -4/3 and correlation length xi = 1 / ln(3), from the spin-1 sector of the transfer matrix
    xi, charge = psi.correlation_length2(charge_sector=None, return_charges=True)
    npt.assert_allclose(xi, 1 / np.log(3), atol=1e-12)
    spin_one = lasy.Sector([2, 0])  # TODO: use named sectors once https://github.com/tenpy/lasyten/issues/314 is done
    assert charge == spin_one
    xis, charges = psi.correlation_length2(target=2, charge_sector=None, return_charges=True)
    npt.assert_allclose(xis, [1 / np.log(3)], atol=1e-12)  # only a single sector with a second eigenvalue
    # in each sector, the dominant eigenvalue is the one of the transfer matrix
    npt.assert_allclose(psi.correlation_length2(charge_sector=spin_one), 1 / np.log(3), atol=1e-12)
    # expected behavior: the trivial sector only has the dominant eigenvalue, there is no second one
    with pytest.raises(ValueError, match='only a single eigenvalue'):
        psi.correlation_length2(charge_sector=0)


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_correlation_length_transfer_matrix(conserve):
    psi = _random_mps(conserve, 2)
    eta = _tm_spectrum(psi, 2)
    expected = -psi.L / np.log(eta[1] / eta[0])
    xi, charge = psi.correlation_length2(return_charges=True)
    npt.assert_allclose(xi, expected / psi.N_sites_per_hor_spacing, atol=1e-10)
    assert charge == psi.symmetry.trivial_sector
    # looking in all sectors can only give a longer correlation length
    xi_all, charge = psi.correlation_length2(charge_sector=None, return_charges=True)
    assert xi_all >= xi - 1e-10
    # all possible sectors
    sectors = psi.correlation_length_charge_sectors()
    xis = [psi.correlation_length2(charge_sector=sector) for sector in sectors]
    npt.assert_allclose(max(xis), xi_all, atol=1e-9)


def test_correlation_length_product_state():
    site = sites.SpinSite(1, conserve='Sz')
    psi = mps.MPS.from_product_state([site] * 2, [0, 2], bc='infinite', unit_cell_width=2)
    assert psi.correlation_length2() == 0.0
    xi, charge = psi.correlation_length2(return_charges=True)
    assert xi == 0.0
    assert charge == psi.symmetry.trivial_sector


@pytest.mark.parametrize('conserve', ['Sz', 'SU2'])
def test_transfer_matrix_charge_sector_none(conserve):
    # regression test: a sector with a single vector returned a 0d array of eigenvalues
    if conserve == 'SU2':
        psi = AKLTChain(dict(L=2, bc_MPS='infinite', conserve='SU2')).psi_AKLT()
    else:
        psi = _random_mps('Sz', 0)
    TM = TransferMatrix(psi, psi, charge_sector=None)
    eta, w = TM.eigenvectors(2)
    assert eta.ndim == 1
    assert len(eta) == len(w) == 2
    npt.assert_allclose(abs(eta[0]), 1.0, atol=1e-12)
    possible_sectors = list(psi.correlation_length_charge_sectors(drop_symmetric=False))
    for vec in w:
        assert any(TM.eigenvector_charge(vec) == sector for sector in possible_sectors)


def test_from_desired_bond_dimension_infinite_Sz():
    # regression test: the infinite branch called _truncate_virtual_space with a missing argument
    site = sites.SpinSite(1, conserve='Sz')
    psi = mps.MPS.from_desired_bond_dimension(
        [site] * 2, 16, bc='infinite', total_charge=site.leg.symmetry.trivial_sector, unit_cell_width=2
    )
    assert max(psi.chi) <= 16
    assert np.max(np.abs(psi.norm_test())) < 1e-12


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_canonical_form_infinite_complex(conserve):
    np.random.seed(5)
    site = sites.SpinSite(1, conserve=conserve)
    sym = site.leg.symmetry
    dtype = lasy.Dtype.complex128
    if conserve == 'SU2':
        V = _virtual_space(sym, [(1, 2), (3, 1)])
        psi = mps.MPS.from_desired_virtual_spaces(
            [site] * 2, [V] * 2, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2, dtype=dtype
        )
    else:
        psi = mps.MPS.from_desired_bond_dimension(
            [site] * 2, 10, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2, dtype=dtype
        )
    xi = psi.correlation_length2(charge_sector=None)
    assert psi.dtype == dtype
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    assert psi.get_SL(0).dtype == dtype.to_real  # the singular values stay real
    assert 0.0 < xi < np.inf


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_canonical_form_infinite_reduces_bond_dimension(conserve):
    # The extra bond states added by `enlarge_chi` are decoupled (zero singular values), so they are projected out.
    psi_ref = _random_mps(conserve, 7)
    psi = psi_ref.copy()
    psi.enlarge_chi([_extra_space(psi, conserve, 2), _extra_space(psi, conserve, 1)])
    psi.form = [None] * psi.L
    chi_ref = [B.get_leg('vL').dim for B in psi_ref._B]
    assert all(B.get_leg('vL').dim > chi for B, chi in zip(psi._B, chi_ref))
    psi.canonical_form_infinite1()
    psi.test_sanity()
    assert [B.get_leg('vL').dim for B in psi._B] == chi_ref
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    npt.assert_allclose(psi.get_SL(0).diagonal_as_numpy(), psi_ref.get_SL(0).diagonal_as_numpy(), atol=1e-8)
    assert 0.0 < psi.correlation_length2() < np.inf


@pytest.mark.parametrize('renormalize', [True, False])
def test_canonical_form_infinite_renormalize(renormalize):
    psi = _random_mps(None, 0)
    psi.form = [None] * psi.L
    psi._B[0] = psi._B[0] * 3.0  # the transfer matrix gets the dominant eigenvalue 9
    psi.canonical_form_infinite1(renormalize=renormalize)
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    npt.assert_allclose(psi.norm, 1.0 if renormalize else 3.0)


def test_canonical_form_infinite_degenerate_spectrum():
    psi = _random_mps(None, 0)
    psi._transfermatrix_keep = 2  # (the correlation length is only checked if we look at more than one eigenvalue)
    # expected behavior: the canonical form is not well defined if the correlation length is too large
    with pytest.raises(ValueError, match='Degenerate spectrum'):
        psi.canonical_form_infinite1(tol_xi=1e-3)


@pytest.mark.xfail(
    reason='canonical_form_infinite1 does not support a non-trivial total charge yet.',
    raises=NotImplementedError,
    strict=True,
)
def test_canonical_form_infinite_nontrivial_total_charge():
    # a non-trivial total charge is carried by a hidden leg
    site = sites.SpinSite(0.5, conserve='Sz')
    psi = mps.MPS.from_product_state([site] * 2, ['up', 'up'], bc='infinite', unit_cell_width=2)
    assert isinstance(psi._B[-1], lasy.HiddenLegTensor)
    psi.canonical_form_infinite1()
    assert np.max(np.abs(psi.norm_test())) < 1e-12


@pytest.mark.parametrize(
    'charge_sector',
    ['trivial', [0], np.int64(0), np.array([0]), lasy.Sector([0]), [lasy.Sector([0])], [0, 'trivial']],
    ids=['str', 'list', 'np_int', 'np_array', 'Sector', 'list_Sector', 'mixed_list'],
)
def test_correlation_length_charge_sector_trivial(charge_sector):
    # all of these select the trivial sector
    psi = _random_mps('Sz', 0)
    xi = psi.correlation_length2(charge_sector=0)
    npt.assert_allclose(psi.correlation_length2(charge_sector=charge_sector), xi, rtol=1e-8)


def test_correlation_length_charge_sector_arguments():
    psi = _random_mps('Sz', 0)
    xi = psi.correlation_length2(charge_sector=0)
    zero = lasy.Sector([0])
    # looking in all sectors can only give a larger correlation length
    assert psi.correlation_length2(charge_sector=None) >= xi - 1e-8
    # a sector given explicitly (and the dominant eigenvalue 1 from the trivial sector)
    for sector in psi.correlation_length_charge_sectors():
        xi_q, q = psi.correlation_length2(charge_sector=sector, return_charges=True)
        assert q == sector or sector == zero
    # expected behavior: only the trivial sector can be given as an integer or string
    with pytest.raises(ValueError, match='invalid charge_sector'):
        psi.correlation_length2(charge_sector=1)
    with pytest.raises(ValueError, match='invalid charge_sector'):
        psi.correlation_length2(charge_sector=[0, 'other'])


def test_correlation_length_dense_reference():
    # without symmetry, compare with the eigenvalues of the dense transfer matrix of the unit cell
    psi = _random_mps(None, 0)
    L = psi.L
    T = None
    for i in range(L):
        B = lasy.planar_permute_legs(psi.get_B(i, 'B'), codomain=['vL', 'p'], domain=['vR']).to_numpy()
        T_i = np.einsum('apb,cpd->acbd', B.conj(), B)
        T_i = T_i.reshape(B.shape[0] ** 2, B.shape[2] ** 2)
        T = T_i if T is None else T @ T_i
    eta = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    npt.assert_allclose(eta[0], 1.0, atol=1e-10)
    xi_dense = -L / np.log(eta[1:] / eta[0])
    npt.assert_allclose(psi.correlation_length2(charge_sector=None), xi_dense[0], rtol=1e-8)
    xis = psi.correlation_length2(target=3, charge_sector=None)
    for xi in xis:  # (exactly degenerate eigenvalues, e.g. complex conjugate pairs, can be missing)
        assert np.min(np.abs(xi_dense - xi)) < 1e-6 * xi


@pytest.mark.parametrize('conserve', [None, 'Sz'])
def test_overlap_infinite_different_states(conserve):
    psi_a, psi_b = _random_mps(conserve, 0), _random_mps(conserve, 1)
    npt.assert_allclose(psi_a.overlap(psi_a, understood_infinite=True), 1.0, atol=1e-10)
    ov_ab = psi_a.overlap(psi_b, understood_infinite=True)
    ov_ba = psi_b.overlap(psi_a, understood_infinite=True)
    assert abs(ov_ab) < 1.0
    # (for real tensors, the dominant eigenvalue can be one of a complex conjugate pair: only the modulus is fixed)
    npt.assert_allclose(abs(ov_ba), abs(ov_ab), atol=1e-10)


@pytest.mark.parametrize('conserve', [None, 'Sz', 'SU2'])
def test_enlarge_chi_infinite(conserve):
    psi = _random_mps(conserve, 0)
    chi = [B.get_leg('vL').dim for B in psi._B]
    S_old = [S.diagonal_as_numpy() for S in psi._S]
    psi_enl = psi.copy()
    psi_enl.enlarge_chi([_extra_space(psi, conserve, 2), 1])
    psi_enl.test_sanity()
    assert all(B.get_leg('vL').dim > c for B, c in zip(psi_enl._B, chi))
    assert np.max(np.abs(psi_enl.norm_test())) < 1e-12
    npt.assert_allclose(psi_enl.overlap(psi, understood_infinite=True), 1.0, atol=1e-12)
    # the new singular values are zero
    for S_new, S in zip(psi_enl._S, S_old):
        S_new = np.sort(S_new.diagonal_as_numpy())
        npt.assert_allclose(S_new[S_new.size - S.size :], np.sort(S), atol=1e-12)
        npt.assert_allclose(S_new[: S_new.size - S.size], 0.0, atol=1e-12)
    assert psi_enl.get_SL(0).dtype == psi.get_SL(0).dtype  # real singular values stay real


def test_enlarge_chi_errors():
    psi = _random_mps(None, 0)
    with pytest.raises(ValueError, match='wrong len of extra_legs'):
        psi.enlarge_chi([1])
    with pytest.raises(ValueError, match='overcomplete'):
        # a trivial site with chi=12 and d=3: there is only room for 3 * 12 - 12 = 24 extra rows
        psi.enlarge_chi([100, 0])
