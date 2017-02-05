"""A collection of tests to check the functionality of `tenpy.dmrg`"""
from __future__ import division

import itertools as it
import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks import mps


def check_dmrg(bc_MPS='finite', engine='EngineCombine', mixer=None):
    xxz_pars = dict(L=4, Jxx=1., Jz=3., hz=0.0, bc_MPS=bc_MPS)
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1]*L)[:L]  # Neel
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    dmrg_pars = {'verbose': 5, 'engine': engine, 'mixer': mixer,
                 'chi_list': {0: 20, 5: 40},
                 'N_sweeps_check': 4,
                 'mixer_params': {'disable_after': 6},
                 'max_sweeps': 40,
                 }
    dmrg.run(psi, M, dmrg_pars)
    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), do_conj=True)
        print "compare with ED: overlap = ", abs(ov)**2
        assert(abs(abs(ov) - 1.) < 1.e-10)

    # TODO: compare with known ground state (energy) / ED !


def test_dmrg():
    for bc_MPS, engine, mixer in it.product(['finite', 'infinite'],
                                            ['EngineCombine', 'EngineFracture'],
                                            [None, 'Mixer']
                                            ):
        yield check_dmrg, bc_MPS, engine, mixer

if __name__ == "__main__":
    for f_args in test_dmrg():
        f = f_args[0]
        print "="*80
        print ' '.join([str(a) for a in f_args])
        print "="*80
        f(*f_args[1:])
