"""A collection of tests to check the functionality of `tenpy.dmrg`"""
# Copyright 2018 TeNPy Developers

import itertools as it
import tenpy.linalg.np_conserved as npc
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks import mps
from nose.plugins.attrib import attr
import numpy as np
from scipy import integrate


def e0_tranverse_ising(g=0.5):
    """Exact groundstate energy of transverse field Ising

    H = - J sigma_z sigma_z + g sigma_x
    Can be obtained by mapping to free fermions."""
    return integrate.quad(_f_tfi, 0, np.pi, args=(g, ))[0]


def _f_tfi(k, g):
    return -2 * np.sqrt(1 + g**2 - 2 * g * np.cos(k)) / np.pi / 2.


def check_dmrg(L=4, bc_MPS='finite', engine='EngineCombine', mixer=None, g=1.5):
    model_params = dict(L=L, J=1., g=g, bc_MPS=bc_MPS, conserve=None, verbose=0)
    M = TFIChain(model_params)
    state = [0] * L  # Ferromagnetic Ising
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    dmrg_pars = {
        'verbose': 5,
        'engine': engine,
        'mixer': mixer,
        'chi_list': {
            0: 20,
            5: 40
        },
        'max_E_err': 1.e-12,
        'max_S_err': 1.e-8,
        'N_sweeps_check': 4,
        'mixer_params': {
            'disable_after': 6,
            'amplitude': 1.e-5
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'max_sweeps': 40,
    }
    if mixer is None:
        del dmrg_pars['mixer_params']  # avoid warning of unused parameter
    if bc_MPS == 'infinite':
        if mixer is not None:
            dmrg_pars['mixer_params']['amplitude'] = 1.e-12  # don't actually contribute...
        dmrg_pars['start_env'] = 1
    res = dmrg.run(psi, M, dmrg_pars)
    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), do_conj=True)
        print("E_DMRG={Edmrg:.14f} vs E_exact={Eex:.14f}".format(Edmrg=res['E'], Eex=np.min(ED.E)))
        print("compare with ED: overlap = ", abs(ov)**2)
        assert abs(abs(ov) - 1.) < 1.e-10  # unique groundstate: finite size gap!
    else:
        # compare exact solution for transverse field Ising model
        Edmrg = res['E']
        Eexact = e0_tranverse_ising(g)
        print("E_DMRG={Edmrg:.12f} vs E_exact={Eex:.12f}".format(Edmrg=Edmrg, Eex=Eexact))
        print("relative energy error: {err:.2e}".format(err=abs((Edmrg - Eexact) / Eexact)))
        print("norm err:", psi.norm_test())
        Edmrg2 = np.mean(psi.expectation_value(M.H_bond))
        assert abs((Edmrg - Eexact) / Eexact) < 1.e-10
        assert abs((Edmrg - Edmrg2) / Edmrg2) < max(1.e-10, np.max(psi.norm_test()))


@attr('slow')
def test_dmrg():
    for bc_MPS, engine, mixer in it.product(['finite', 'infinite'],
                                            ['EngineCombine', 'EngineFracture'], [None, 'Mixer']):
        L = 4 if bc_MPS == 'finite' else 2
        yield check_dmrg, L, bc_MPS, engine, mixer


if __name__ == "__main__":
    for f_args in test_dmrg():
        f = f_args[0]
        print("=" * 80)
        print(' '.join([str(a) for a in f_args]))
        print("=" * 80)
        f(*f_args[1:])
