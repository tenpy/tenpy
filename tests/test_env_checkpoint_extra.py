"""Extra checks for the TeNPy option ``env_checkpoint_every``.

1. DMRG with TeNPy's default-style mixer: single-site + SubspaceExpansion, two-site + DensityMatrixMixer.
2. TDVP (two-site, real time) from a Neel state.
3. Combination with disk offload: CacheFile + PickleStorage (cache_threshold_chi=0): exact, fewer files on disk.
4. Infinite DMRG: option must be a no-op.
"""
import os, glob, tempfile
import numpy as np
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tdvp
from tenpy.tools.cache import CacheFile

L = 24


def model(bc='finite', conserve='Sz', Lm=L):
    return XXZChain(dict(L=Lm, Jxx=1., Jz=1., hz=0., bc_MPS=bc, conserve=conserve))


def neel(M, bc='finite'):
    return MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (M.lat.N_sites // 2), bc=bc)


def dmrg_run(engine_cls, mixer, k, cache=None, extra=None):
    M = model()
    opts = dict(trunc_params=dict(chi_max=48, svd_min=1e-14), mixer=mixer, max_sweeps=6, min_sweeps=6, max_E_err=0.,
                lanczos_params=dict(N_max=4, N_min=2))
    if mixer:
        opts['mixer_params'] = dict(amplitude=1e-5, decay=2., disable_after=4)
    if k:
        opts['env_checkpoint_every'] = k
    opts.update(extra or {})
    kw = dict(cache=cache) if cache is not None else {}
    eng = engine_cls(neel(M), M, opts, **kw)
    E, _ = eng.run()
    return [float(e) for e in eng.sweep_stats['E']], float(E)


out = []
# 1. mixers
for name, cls, mixer in (("single-site + SubspaceExpansion", dmrg.SingleSiteDMRGEngine, 'SubspaceExpansion'),
                         ("two-site + DensityMatrixMixer", dmrg.TwoSiteDMRGEngine, 'DensityMatrixMixer')):
    ref, Er = dmrg_run(cls, mixer, None)
    for k in (3, 5):
        res, E = dmrg_run(cls, mixer, k)
        out.append(f"[1] {name} k={k}: max |dE| over sweeps = {max(abs(a - b) for a, b in zip(res, ref)):.3e}")

# 2. TDVP (two-site, real time)
def tdvp_run(k):
    M = model()
    psi = neel(M)
    opts = dict(dt=0.05, N_steps=10, trunc_params=dict(chi_max=48, svd_min=1e-12))
    if k:
        opts['env_checkpoint_every'] = k
    eng = tdvp.TwoSiteTDVPEngine(psi, M, opts)
    for _ in range(2):
        eng.run()
    return psi
a, b = tdvp_run(None), tdvp_run(4)
out.append(f"[2] TDVP two-site, 20 steps dt=0.05, k=4: |1 - |<a|b>|| = {abs(1 - abs(a.overlap(b))):.3e}")

# 3. with disk offload (PickleStorage): exactness and number of stored environment files
def disk_run(k):
    d = tempfile.mkdtemp(prefix="tenpy_ckpt_")
    with CacheFile.open(storage_class='PickleStorage', directory=os.path.join(d, 'c'), delete=False) as cache:
        res, E = dmrg_run(dmrg.TwoSiteDMRGEngine, None, k, cache=cache, extra=dict(cache_threshold_chi=0))
        files = [f for f in glob.glob(os.path.join(d, 'c', '**', '*'), recursive=True) if os.path.isfile(f)]
        size = sum(os.path.getsize(f) for f in files)
    return res, len(files), size
r0, n0, s0 = disk_run(None)
r1, n1, s1 = disk_run(5)
out.append(f"[3] + PickleStorage disk cache, k=5: max |dE| = {max(abs(a - b) for a, b in zip(r0, r1)):.3e}; "
           f"env files on disk {n0} -> {n1}, bytes {s0} -> {s1}")

# 4. infinite DMRG: no-op
def idmrg(k):
    M = model('infinite', Lm=2)
    opts = dict(trunc_params=dict(chi_max=32, svd_min=1e-12), mixer=False, max_sweeps=10, min_sweeps=10,
                max_E_err=0.)
    if k:
        opts['env_checkpoint_every'] = k
    E, _ = dmrg.TwoSiteDMRGEngine(neel(M, 'infinite'), M, opts).run()
    return float(E)
out.append(f"[4] infinite DMRG: E(no option) - E(k=3) = {idmrg(None) - idmrg(3):.3e}")
print("\n".join(out))
