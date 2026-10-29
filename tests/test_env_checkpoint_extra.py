"""Extra checks for ``env_checkpoint_every``: mixers, TDVP, disk caching, infinite MPS."""

import glob
import os

import pytest

from tenpy.algorithms import dmrg, tdvp
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.tools.cache import CacheFile

L = 24


def model(bc='finite', Lm=L):
    return XXZChain(dict(L=Lm, Jxx=1.0, Jz=1.0, hz=0.0, bc_MPS=bc, conserve='Sz'))


def neel(M, bc='finite'):
    return MPS.from_product_state(
        M.lat.mps_sites(), ['up', 'down'] * (M.lat.N_sites // 2), bc=bc, unit_cell_width=M.lat.mps_unit_cell_width
    )


def dmrg_run(engine_cls, mixer, k, cache=None, extra=None):
    M = model()
    opts = dict(
        trunc_params=dict(chi_max=48, svd_min=1e-14),
        mixer=mixer,
        max_sweeps=6,
        min_sweeps=6,
        max_E_err=0.0,
        lanczos_params=dict(N_max=4, N_min=2),
    )
    if mixer:
        opts['mixer_params'] = dict(amplitude=1e-5, decay=2.0, disable_after=4)
    if k:
        opts['env_checkpoint_every'] = k
    opts.update(extra or {})
    kw = dict(cache=cache) if cache is not None else {}
    eng = engine_cls(neel(M), M, opts, **kw)
    eng.run()
    return [float(e) for e in eng.sweep_stats['E']]


@pytest.mark.parametrize(
    'engine_cls, mixer',
    [(dmrg.SingleSiteDMRGEngine, 'SubspaceExpansion'), (dmrg.TwoSiteDMRGEngine, 'DensityMatrixMixer')],
)
def test_mixers(engine_cls, mixer):
    assert dmrg_run(engine_cls, mixer, 5) == dmrg_run(engine_cls, mixer, None)


def test_tdvp():
    def run(k):
        psi = neel(model())
        opts = dict(dt=0.05, N_steps=10, trunc_params=dict(chi_max=48, svd_min=1e-12))
        if k:
            opts['env_checkpoint_every'] = k
        tdvp.TwoSiteTDVPEngine(psi, model(), opts).run()
        return psi

    assert abs(1 - abs(run(None).overlap(run(4)))) < 1e-12


def test_disk_cache(tmp_path):
    def run(k, d):
        with CacheFile.open(storage_class='PickleStorage', directory=str(d), delete=False) as cache:
            res = dmrg_run(dmrg.TwoSiteDMRGEngine, None, k, cache=cache, extra=dict(cache_threshold_chi=0))
            n = len([f for f in glob.glob(os.path.join(d, '**', '*'), recursive=True) if os.path.isfile(f)])
        return res, n

    r0, n0 = run(None, tmp_path / 'a')
    r1, n1 = run(5, tmp_path / 'b')
    assert r0 == r1
    assert n1 < n0


def test_infinite_noop():
    def run(k):
        M = model('infinite', Lm=2)
        opts = dict(
            trunc_params=dict(chi_max=32, svd_min=1e-12), mixer=False, max_sweeps=10, min_sweeps=10, max_E_err=0.0
        )
        if k:
            opts['env_checkpoint_every'] = k
        E, _ = dmrg.TwoSiteDMRGEngine(neel(M, 'infinite'), M, opts).run()
        return float(E)

    assert run(None) == run(3)
