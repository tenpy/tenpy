"""Tests for the option ``env_checkpoint_every`` of finite MPS sweep engines."""

import pytest

from tenpy.algorithms import dmrg
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS


def _run(k, conserve):
    M = XXZChain(dict(L=24, Jxx=1.0, Jz=1.0, hz=0.0, bc_MPS='finite', conserve=conserve))
    psi = MPS.from_product_state(
        M.lat.mps_sites(), ['up', 'down'] * 12, bc='finite', unit_cell_width=M.lat.mps_unit_cell_width
    )
    opts = dict(
        trunc_params=dict(chi_max=48, svd_min=1e-14),
        mixer=False,
        max_sweeps=6,
        min_sweeps=6,
        max_E_err=0.0,
        lanczos_params=dict(N_max=4, N_min=2),
    )
    if k:
        opts['env_checkpoint_every'] = k
    eng = dmrg.TwoSiteDMRGEngine(psi, M, opts)
    E, _ = eng.run()
    n_stored = sum(1 for key in eng.env.cache.keys() if key.startswith(('LP_', 'RP_')))
    return [float(e) for e in eng.sweep_stats['E']], E, n_stored


@pytest.mark.parametrize('conserve', ['Sz', None])
@pytest.mark.parametrize('k', [1, 2, 3, 5, 24, 100])
def test_env_checkpoint_exact(k, conserve):
    ref, E_ref, n_ref = _run(None, conserve)
    res, E, n = _run(k, conserve)
    assert res == ref  # identical contractions -> identical energies
    assert E == E_ref
    if 1 < k < 24:
        assert n < n_ref  # fewer environments stored (k=1 keeps all; k>=L keeps only the boundaries + segment)


def test_env_checkpoint_cache_bound():
    """During a sweep the number of stored environments stays ~ L/k + 2k (+ small constant)."""
    import math

    M = XXZChain(dict(L=48, Jxx=1.0, Jz=1.0, hz=0.0, bc_MPS='finite', conserve='Sz'))
    psi = MPS.from_product_state(
        M.lat.mps_sites(), ['up', 'down'] * 24, bc='finite', unit_cell_width=M.lat.mps_unit_cell_width
    )
    k = 7
    eng = dmrg.TwoSiteDMRGEngine(
        psi,
        M,
        dict(
            trunc_params=dict(chi_max=32, svd_min=1e-14),
            mixer=False,
            max_sweeps=3,
            min_sweeps=3,
            max_E_err=0.0,
            env_checkpoint_every=k,
        ),
    )
    worst = 0
    orig = eng.update_local

    def counting(*a, **kw):
        nonlocal worst
        worst = max(worst, sum(1 for key in eng.env.cache.keys() if key.startswith(('LP_', 'RP_'))))
        return orig(*a, **kw)

    eng.update_local = counting
    eng.run()
    assert worst <= 2 * (math.ceil(48 / k) + 2 * k) + 4, worst
