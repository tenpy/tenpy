"""Short test for vmem prediction."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

import tenpy.algorithms as algo
import tenpy.models as mods
import tenpy.networks.mps as mps


def test_bosonic_model_TEBD():
    """Test with a bosonic chain."""
    L = 15
    model = mods.hubbard.BoseHubbardChain({'conserve': None, 'U': 1, 't': 1, 'bc_MPS': 'finite', 'L': L, 'n_max': 4})
    # Test TEBD first
    psi = mps.MPS.from_product_state(model.lat.mps_sites(), [0] * L, unit_cell_width=model.lat.mps_unit_cell_width)
    engine = algo.tebd.TEBDEngine(psi, model, {'trunc_params': {'chi_max': 33}})
    # expected value:
    # bond | 0   | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
    # chi  | 5   | 25 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 25 | 5  |
    # add matrices:
    num_entries = (
        5 * 25
        + 25 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 33
        + 33 * 25
        + 25 * 5
    )
    num_entries *= 5  # physical leg
    # expected MPS vmem:
    exact = (num_entries * np.dtype('complex128').itemsize) / 1024**2  # in MB
    estimate = engine.estimate_RAM()
    if abs(estimate - exact) >= 1e-10:
        raise AssertionError(f'TEBD RAM did not match expectation (expected: {exact}, gotten:{estimate})')


def test_bosonic_model_DMRG():
    """Test with a bosonic chain."""
    L = 15
    model = mods.hubbard.BoseHubbardChain({'conserve': None, 'U': 1, 't': 1, 'bc_MPS': 'finite', 'L': L, 'n_max': 4})
    psi = mps.MPS.from_product_state(model.lat.mps_sites(), [0] * L, unit_cell_width=model.lat.mps_unit_cell_width)
    engine = algo.dmrg.TwoSiteDMRGEngine(psi, model, {'trunc_params': {'chi_max': 99}})
    # expected value:
    # bond | 0   | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
    # chi  | 5   | 25 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 25 | 5  |
    chis = np.array([5, 25, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 25])
    # add matrices:
    psi_entries = (
        5 * 25
        + 25 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 99
        + 99 * 25
        + 25 * 5
    )
    psi_entries *= 5  # physical leg
    # environment:
    MPS_env_entries = sum(chis**2 * 4)
    MPO_entries = 4**2 * 5**2 * (L - 2) + 2 * 4 * 5**2 * 2
    # add lanczos:
    lanczos_entries = 3 * 5**2 * (99**2 * 4) + 2 * 99**2 * 5**2
    total_entries = psi_entries + MPS_env_entries + MPO_entries + lanczos_entries
    exact = total_entries * np.dtype('float64').itemsize / 1024**2  # in MB
    estimate = engine.estimate_RAM()
    if abs(estimate - exact) >= 1.0e-10:
        raise AssertionError(f'DMRG RAM did not match expectation (expected: {exact}, gotten:{estimate})')
