"""A collection of tests for (classes in) :mod:`tenpy.algorithms.mps_common`."""

# Copyright (C) TeNPy Developers, Apache license
import cyten as ct
import numpy as np
import numpy.testing as npt
from cyten.models.sites import SpinSite

from tenpy.algorithms.mps_common import VariationalCompression
from tenpy.networks.mps import MPS
from tenpy.tools import TruncationError


def test_variational_compression():
    """Compress a random finite MPS to a smaller bond dimension.

    With no Hamiltonian involved, the exact optimum for compressing a state to a fixed bond
    dimension is given by an SVD truncation of the corresponding bipartition of the dense state.
    We pick a system size such that only a single bond needs truncating, so that this SVD
    truncation (done here by hand, directly on the dense array) is *the* global optimum that
    :class:`VariationalCompression` should reproduce.
    """
    rng = np.random.default_rng(0)
    L = 4
    site = SpinSite(S=0.5, conserve=None)
    sites = [site] * L
    d = site.dim
    backend = ct.get_backend('no_symmetry', 'numpy')

    block = rng.normal(size=(d,) * L) + 1.0j * rng.normal(size=(d,) * L)
    block /= np.linalg.norm(block)

    p_labels = [f'p{i}' for i in range(L)]
    psi_tensor = ct.SymmetricTensor.from_dense_block(block, codomain=[site.leg] * L, backend=backend, labels=p_labels)
    psi_large = MPS.from_full(sites, psi_tensor, form=None, unit_cell_width=L)
    assert max(psi_large.chi) == d**2  # bond (1, 2) is at its maximal, untruncated bond dimension

    chi_small = 2
    options = {
        'trunc_params': {'chi_max': chi_small},
        'min_sweeps': 5,
        'max_sweeps': 5,
        'tol_theta_diff': None,
        'max_trunc_err': 1.0,  # we *want* a large, known truncation error here
    }
    compressor = VariationalCompression(psi_large.copy(), options)
    trunc_err = compressor.run()
    psi_small = compressor.psi
    assert max(psi_small.chi) == chi_small

    # hard-coded reference: numpy SVD of the dense array at the only bond that needs truncating
    mat = block.reshape(d**2, d**2)
    S_ideal = np.linalg.svd(mat, compute_uv=False)
    err_ideal = TruncationError.from_S(S_ideal[chi_small:], norm_old=np.linalg.norm(S_ideal))

    npt.assert_allclose(trunc_err.eps, err_ideal.eps, atol=1e-10)

    # contract the compressed MPS back to a dense array and check its fidelity loss directly
    theta = psi_small.get_theta(0, n=L)
    got = theta.to_numpy(['vL'] + p_labels + ['vR']).reshape((d,) * L)
    overlap = np.vdot(got.ravel(), block.ravel())
    fidelity_loss = 1.0 - abs(overlap) ** 2 / (np.linalg.norm(got) ** 2 * np.linalg.norm(block) ** 2)
    npt.assert_allclose(fidelity_loss, err_ideal.eps, atol=1e-10)
