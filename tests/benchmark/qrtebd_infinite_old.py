"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3
import numpy as np
from tenpy.tools.misc import to_iterable
from misc import get_qmod, parse_symmetry

try:
    import old_tenpy as otp  # type: ignore
except ModuleNotFoundError:
    print('This benchmark expects you to have a compiled version of tenpy v0.10 in your '
          '$PYTHONPATH under the name "old_tenpy".')
    raise


def get_tebd_engine(old_tenpy, mod_q=[1], legs=10, size=20, **kwargs):
    """Setup TEBD benchmark.

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
    model_params = dict(L=L, S=2., D=0.3, bc_MPS='infinite', conserve=conserve, verbose=0)
    #  print("conserve =", repr(conserve))
    M = old_tenpy.models.spins.SpinChain(model_params)
    initial_state = (['up', 'down'] * L)[:L]
    psi = old_tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), initial_state, bc='infinite')
    local_dim = psi.sites[0].dim
    tebd_params = {
        'trunc_params': {
            'chi_max': size,
            'svd_min': 1.e-45,
        },
        'order': 2,
        'N_steps': 5,
        'dt': 0.1,
        'verbose': 0.,
        'use_eig_based_svd': True,
    }
    eng = old_tenpy.algorithms.tebd.QRBasedTEBDEngine(psi, M, tebd_params)
    eng.verbose = 0.02
    old_tenpy.tools.optimization.set_level(3)
    for i in range(5 + int(np.log(size) / np.log(local_dim))):
        eng.run()
        if eng.verbose > 0.1:
            print(eng.psi.chi)
            print(eng.psi.entanglement_entropy())
    assert min(eng.psi.chi) == size  # ensure full bond dimension
    if eng.verbose > 0.1:
        print("set up tebd for size", size)
    return eng


def setup_benchmark(symmetry='u1_symmetry', legs=10, size=20, **kwargs):
    symmetry = parse_symmetry(to_iterable(symmetry))
    return get_tebd_engine(old_tenpy=otp, mod_q=get_qmod(symmetry), legs=legs, size=size, **kwargs)


def benchmark(data):
    eng = data
    eng.run()
    if eng.verbose > 0.01:
        print('final chi', eng.psi.chi)
