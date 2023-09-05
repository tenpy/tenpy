"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3
from tenpy.tools.misc import to_iterable
from misc import get_qmod, parse_symmetry

try:
    import old_tenpy as otp  # type: ignore
except ModuleNotFoundError:
    print('This benchmark expects you to have a compiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy".')
    raise

def get_dmrg_engine(old_tenpy, mod_q=[1], legs=10, size=20, diag_method='lanczos', **kwargs):
    """Setup DMRG benchmark.

    Mapping of parameters:
        size -> chi
        legs -> L = number of sites
        mod_q -> conserve
    """
    L = legs  # number of sites: legs is abbreviated with `l`
    if len(mod_q) == 0:
        conserve = None
    elif mod_q == [2]:
        conserve = 'parity'
    elif mod_q == [1]:
        conserve = 'Sz'
    model_params = dict(L=L, S=2., D=0.3, bc_MPS='infinite', conserve=conserve)
    #  print("conserve =", repr(conserve))
    M = old_tenpy.models.spins.SpinChain(model_params)
    initial_state = (['up', 'down'] * L)[:L]
    psi = old_tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), initial_state, bc='infinite')
    dmrg_params = {
        'trunc_params': {
            'chi_max': size,
            'svd_min': 1.e-45,
        },
        'lanczos_params': {
            'N_min': 10,
            'N_max': 10
        },
        #  'mixer': None,
        #  'N_sweeps_check': 1,
        #  'min_sweeps': 10,
        #  'max_sweeps': 100,
        #  'max_E_err': 1.e-13,
    }
    eng = old_tenpy.algorithms.dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    eng.diag_method = diag_method
    for i in range(100):
        eng.sweep(meas_E_trunc=False)
        eng.sweep(optimize=False, meas_E_trunc=False)  # environment sweep
    eng.reset_stats()
    return eng


def setup_benchmark(symmetry='u1_symmetry', legs=10, size=20, diag_method='lanczos', **kwargs):
    symmetry = parse_symmetry(to_iterable(symmetry))
    return get_dmrg_engine(old_tenpy=otp, mod_q=get_qmod(symmetry), legs=legs, size=size,
                           diag_method=diag_method, **kwargs)


def benchmark(data):
    eng = data
    for i in range(10):  # 10 sweeps
        eng.sweep(meas_E_trunc=False)
        #  eng.sweep(optimize=False, meas_E_trunc=False)  # environment sweep
        #  if eng.verbose > 0.1:
        #      print(eng.psi.chi)
        #      print(eng.psi.entanglement_entropy())
    #  if eng.verbose > 0.01:
    #      print('final chi', eng.psi.chi)
    #      print("performed on average {0:.5f} Lanczos iterations".format(
    #          np.mean(eng.update_stats['N_lanczos'])))
