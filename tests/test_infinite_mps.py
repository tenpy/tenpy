"""Tests for infinite MPS: canonical form, transfer matrix, correlation length.

The MPS are constructed with random tensors and checked against exact properties, e.g. of the AKLT state.
"""

# Copyright (C) TeNPy Developers, Apache license
import warnings

import lasyten as lasy
import numpy as np
import numpy.testing as npt
import pytest
from lasyten.models import sites

from tenpy.networks import mps
from tenpy.networks.mps import TransferMatrix


@pytest.fixture(autouse=True)
def _seed_numpy_rng():
    """The numpy backend of lasyten draws random numbers (e.g. the noise for Arnoldi) from the global numpy RNG."""
    np.random.seed(12345)


def _spin_sites(kind, S=1):
    conserve = {'SU2': 'SU2', 'Sz': 'Sz', 'none': None}[kind]
    return sites.SpinSite(S, conserve=conserve)


def _virtual_space(symmetry, weights_mults):
    """Virtual space with the given SU(2) sectors, given as ``(2 * spin, multiplicity)`` pairs."""
    return lasy.ElementarySpace.from_defining_sectors(
        symmetry,
        np.array([[w, 0] for w, _ in weights_mults]),
        multiplicities=[m for _, m in weights_mults],
    )


def _random_mps(kind, seed=0, canonicalize=True, monkeypatch=None):
    """Random infinite MPS (unit cell of two spin-1 sites) with the given symmetry.

    If not `canonicalize`, return the random tensors *without* calling ``canonical_form``.
    """
    np.random.seed(seed)
    site = _spin_sites(kind)
    sym = site.leg.symmetry
    if not canonicalize:
        monkeypatch.setattr(mps.MPS, 'canonical_form', lambda self, *a, **k: None)
    if kind == 'SU2':
        # bonds with spin 1/2 and 3/2
        V = _virtual_space(sym, [(1, 3), (3, 2)])
        return mps.MPS.from_desired_virtual_spaces(
            [site] * 2, [V] * 2, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2
        )
    return mps.MPS.from_desired_bond_dimension(
        [site] * 2, 12, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2
    )


def _tm_spectrum(psi, num=3):
    """Magnitudes of the dominant eigenvalues of the transfer matrix in the trivial sector (any form)."""
    TM = TransferMatrix(psi, psi, 0, 0, transpose=False, charge_sector=0, form=None)
    with warnings.catch_warnings():
        # for random (non-canonical) tensors, the transfer matrix is badly conditioned.
        warnings.simplefilter('ignore', UserWarning)
        eta, _ = TM.eigenvectors(num, v0=TM.initial_guess(1.0), which='LM', tol=1e-10, N_max=80)
    return np.abs(eta)


@pytest.mark.parametrize('kind', ['none', 'Sz', 'SU2'])
@pytest.mark.parametrize('seed', [0, 1])
def test_canonical_form_infinite(kind, seed, monkeypatch):
    psi = _random_mps(kind, seed, canonicalize=False, monkeypatch=monkeypatch)
    spectrum_before = _tm_spectrum(psi)
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        psi.canonical_form()
    psi.test_sanity()
    assert psi.form == [(0.0, 1.0)] * 2
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    # a gauge transformation (and bond-dimension preserving projection) does not change the transfer matrix spectrum
    spectrum_after = _tm_spectrum(psi)
    npt.assert_allclose(spectrum_after[1:] / spectrum_after[0], spectrum_before[1:] / spectrum_before[0], atol=1e-8)
    # the singular values are normalized; calling it again does not change anything
    S_before = [S.diagonal_as_numpy().copy() for S in psi._S]
    psi.canonical_form()
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    for S0, S1 in zip(S_before, [S.diagonal_as_numpy() for S in psi._S]):
        npt.assert_allclose(S0, S1, atol=1e-7)


@pytest.mark.parametrize('kind', ['none', 'Sz', 'SU2'])
@pytest.mark.parametrize('L', [3, 4])
def test_canonical_form_infinite_unit_cell_width(kind, L):
    # regression test: the sweep over the bonds of the unit cell failed for more than two sites
    np.random.seed(L)
    site = _spin_sites(kind)
    sym = site.leg.symmetry
    if kind == 'SU2':
        V = _virtual_space(sym, [(1, 2), (3, 1)])
        psi = mps.MPS.from_desired_virtual_spaces(
            [site] * L, [V] * L, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=L
        )
    else:
        psi = mps.MPS.from_desired_bond_dimension(
            [site] * L, 8, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=L
        )
    psi.test_sanity()
    assert np.max(np.abs(psi.norm_test())) < 1e-12


def test_canonical_form_infinite_singular_values_normalized():
    psi = _random_mps('SU2', 3)
    for i in range(psi.L):
        S = psi.get_SL(i)
        # the trace is weighted with the quantum dimensions of the sectors
        npt.assert_allclose(lasy.trace(S**2).to_numpy(), 1.0, atol=1e-12)
        assert np.all(S.diagonal_as_numpy() > 0)


def _aklt_mps():
    """The AKLT state as an SU(2) symmetric MPS: the symmetric tensor with a spin-1/2 bond is unique."""
    site = _spin_sites('SU2')
    sym = site.leg.symmetry
    V = _virtual_space(sym, [(1, 1)])
    return mps.MPS.from_desired_virtual_spaces(
        [site] * 2, [V] * 2, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2
    )


def test_aklt_state():
    psi = _aklt_mps()
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    # entanglement entropy of the AKLT state is ln(2): a singlet between two spin-1/2 (one per bond)
    npt.assert_allclose(psi.entanglement_entropy(), np.log(2), atol=1e-12)
    # <S_i . S_{i+1}> = -4/3 and correlation length xi = 1 / ln(3), from the spin-1 sector of the transfer matrix
    xi, charge = psi.correlation_length2(charge_sector=None, return_charges=True)
    npt.assert_allclose(xi, 1 / np.log(3), atol=1e-12)
    assert charge == lasy.Sector([2, 0])  # spin 1
    xis, charges = psi.correlation_length2(target=2, charge_sector=None, return_charges=True)
    npt.assert_allclose(xis, [1 / np.log(3)], atol=1e-12)  # only a single sector with a second eigenvalue
    # in each sector, the dominant eigenvalue is the one of the transfer matrix
    npt.assert_allclose(psi.correlation_length2(charge_sector=lasy.Sector([2, 0])), 1 / np.log(3), atol=1e-12)
    with pytest.raises(ValueError, match='only a single eigenvalue'):
        psi.correlation_length2(charge_sector=0)  # trivial sector only has the dominant eigenvalue


@pytest.mark.parametrize('kind', ['none', 'Sz', 'SU2'])
def test_correlation_length_transfer_matrix(kind):
    psi = _random_mps(kind, 2)
    eta = _tm_spectrum(psi, 2)
    expected = -psi.L / np.log(eta[1] / eta[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        xi = psi.correlation_length()
    npt.assert_allclose(xi, expected, atol=1e-10)
    xi2, charge = psi.correlation_length2(return_charges=True)
    npt.assert_allclose(xi2, expected / psi.N_sites_per_hor_spacing, atol=1e-10)
    assert charge == psi.symmetry.trivial_sector
    # looking in all sectors can only give a longer correlation length
    xi_all, charge = psi.correlation_length2(charge_sector=None, return_charges=True)
    assert xi_all >= xi2 - 1e-10
    # all possible sectors
    sectors = psi.correlation_length_charge_sectors()
    xis = [psi.correlation_length2(charge_sector=sector) for sector in sectors]
    npt.assert_allclose(max(xis), xi_all, atol=1e-9)


def test_correlation_length_product_state():
    site = _spin_sites('Sz')
    psi = mps.MPS.from_product_state([site] * 2, [0, 2], bc='infinite', unit_cell_width=2)
    assert psi.correlation_length2() == 0.0
    xi, charge = psi.correlation_length2(return_charges=True)
    assert xi == 0.0
    assert charge == psi.symmetry.trivial_sector


@pytest.mark.parametrize('kind', ['Sz', 'SU2'])
def test_transfer_matrix_charge_sector_none(kind):
    # regression test: a sector with a single vector returned a 0d array of eigenvalues
    psi = _aklt_mps() if kind == 'SU2' else _random_mps('Sz', 0)
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
    site = _spin_sites('Sz')
    psi = mps.MPS.from_desired_bond_dimension(
        [site] * 2, 16, bc='infinite', total_charge=site.leg.symmetry.trivial_sector, unit_cell_width=2
    )
    assert max(psi.chi) <= 16
    assert np.max(np.abs(psi.norm_test())) < 1e-12


@pytest.mark.parametrize('kind', ['none', 'Sz', 'SU2'])
def test_canonical_form_infinite_complex(kind):
    np.random.seed(5)
    site = _spin_sites(kind)
    sym = site.leg.symmetry
    dtype = lasy.Dtype.complex128
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        if kind == 'SU2':
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


def _zero_padded_mps(mode, pad=2, seed=7):
    """Infinite MPS without symmetry with a bond dimension `pad` larger than necessary.

    For ``mode='zero'``, the additional bond states are exactly decoupled (zero blocks), for ``mode='tiny'``
    they carry a tiny, decoupled block. Also returns the singular values of the (canonical) state without padding.
    """
    np.random.seed(seed)
    site = _spin_sites('none')
    sym = site.leg.symmetry
    psi0 = mps.MPS.from_desired_bond_dimension(
        [site] * 2, 4, bc='infinite', total_charge=sym.trivial_sector, unit_cell_width=2
    )
    Bs = []
    for i in range(psi0.L):
        B = lasy.planar_permute_legs(psi0.get_B(i, 'B'), codomain=['vL', 'p'], domain=['vR']).to_numpy()
        a, d, b = B.shape
        B_pad = np.zeros((a + pad, d, b + pad))
        B_pad[:a, :, :b] = B
        if mode == 'tiny':
            B_pad[a:, :, b:] = 1e-3 * np.random.rand(pad, d, pad)
        Bs.append(B_pad)
    V = lasy.ElementarySpace.from_trivial_sector(dim=4 + pad, symmetry=sym)
    Bs = [
        lasy.SymmetricTensor.from_dense_block(
            B, [V, site.leg], [V], psi0.backend, labels=['vL', 'p', 'vR'], dtype=lasy.Dtype.float64
        )
        for B in Bs
    ]
    SVs = [lasy.DiagonalTensor.from_eye(V, psi0.backend, ['vL', 'vR'], lasy.Dtype.float64) for _ in range(3)]
    psi = mps.MPS([site] * 2, Bs, SVs, bc='infinite', form='B', unit_cell_width=2)
    psi.form = [None] * psi.L
    return psi, psi0.get_SL(0).diagonal_as_numpy()


@pytest.mark.parametrize('mode', ['zero', 'tiny'])
def test_canonical_form_infinite_reduces_bond_dimension(mode):
    # the bond states that are (almost) decoupled are projected out, i.e., the bond dimension is reduced
    psi, S_ref = _zero_padded_mps(mode)
    assert [B.get_leg('vL').dim for B in psi._B] == [6, 6]
    psi.canonical_form_infinite1()
    psi.test_sanity()
    assert [B.get_leg('vL').dim for B in psi._B] == [4, 4]
    assert np.max(np.abs(psi.norm_test())) < 1e-12
    npt.assert_allclose(psi.get_SL(0).diagonal_as_numpy(), S_ref, atol=1e-8)
    assert 0.0 < psi.correlation_length2() < np.inf


def test_canonical_form_infinite_renormalize():
    for renormalize in [True, False]:
        psi = _random_mps('none', 0)
        psi.form = [None] * psi.L
        psi._B[0] = psi._B[0] * 3.0  # the transfer matrix gets the dominant eigenvalue 9
        psi.canonical_form_infinite1(renormalize=renormalize)
        assert np.max(np.abs(psi.norm_test())) < 1e-12
        npt.assert_allclose(psi.norm, 1.0 if renormalize else 3.0)


def test_canonical_form_infinite_errors():
    psi = _random_mps('none', 0)
    psi._transfermatrix_keep = 2  # (the correlation length is only checked if we look at more than one eigenvalue)
    with pytest.raises(ValueError, match='Degenerate spectrum'):
        psi.canonical_form_infinite1(tol_xi=1e-3)  # (the correlation length is larger)
    # a non-trivial total charge is carried by a hidden leg, which is not supported yet
    site = _spin_sites('Sz', S=0.5)
    psi = mps.MPS.from_product_state([site] * 2, ['up', 'up'], bc='infinite', unit_cell_width=2)
    assert isinstance(psi._B[-1], lasy.HiddenLegTensor)
    with pytest.raises(NotImplementedError, match='total charge'):
        psi.canonical_form_infinite1()


def test_correlation_length_charge_sector_arguments():
    psi = _random_mps('Sz', 0)
    xi = psi.correlation_length2(charge_sector=0)
    zero = lasy.Sector([0])
    for charge_sector in ['trivial', [0], np.int64(0), np.array([0]), zero, [zero], [0, 'trivial']]:
        npt.assert_allclose(psi.correlation_length2(charge_sector=charge_sector), xi, rtol=1e-8)
    # looking in all sectors can only give a larger correlation length
    assert psi.correlation_length2(charge_sector=None) >= xi - 1e-8
    # a sector given explicitly (and the dominant eigenvalue 1 from the trivial sector)
    for sector in psi.correlation_length_charge_sectors():
        xi_q, q = psi.correlation_length2(charge_sector=sector, return_charges=True)
        assert q == sector or sector == zero
    with pytest.raises(ValueError, match='invalid charge_sector'):
        psi.correlation_length2(charge_sector=1)
    with pytest.raises(ValueError, match='invalid charge_sector'):
        psi.correlation_length2(charge_sector=[0, 'other'])


def test_correlation_length_dense_reference():
    # without symmetry, compare with the eigenvalues of the dense transfer matrix of the unit cell
    psi = _random_mps('none', 0)
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


@pytest.mark.parametrize('kind', ['none', 'Sz'])
def test_overlap_infinite_different_states(kind):
    psi_a, psi_b = _random_mps(kind, 0), _random_mps(kind, 1)
    npt.assert_allclose(psi_a.overlap(psi_a, understood_infinite=True), 1.0, atol=1e-10)
    ov_ab = psi_a.overlap(psi_b, understood_infinite=True)
    ov_ba = psi_b.overlap(psi_a, understood_infinite=True)
    assert abs(ov_ab) < 1.0
    # (for real tensors, the dominant eigenvalue can be one of a complex conjugate pair: only the modulus is fixed)
    npt.assert_allclose(abs(ov_ba), abs(ov_ab), atol=1e-10)


def test_not_ported_raises_not_implemented():
    # these still contain np_conserved code; they should fail cleanly
    site = _spin_sites('Sz', S=0.5)
    psi_finite = mps.MPS.from_product_state([site] * 4, [0, 1, 0, 1], bc='finite', unit_cell_width=4)
    with pytest.raises(NotImplementedError):
        psi_finite.overlap(psi_finite, ignore_form=True)
    with pytest.raises(NotImplementedError):
        _random_mps('none', 0).compute_K(np.arange(2))
