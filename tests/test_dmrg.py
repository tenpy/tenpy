"""A collection of tests to check the functionality of `tenpy.dmrg`"""
from __future__ import division

import itertools as it
import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg
from tenpy.networks import mps

import warnings
warnings.simplefilter("error")


def check_dmrg(bc_MPS='finite', engine='EngineCombine', mixer=None):
    xxz_pars = dict(L=4, Jxx=1., Jz=3., hz=0.0, bc_MPS=bc_MPS)
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1]*L)[:L]  # Neel
    psi = mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    dmrg_pars = {'verbose': 50, 'engine': engine, 'mixer': mixer,
                 'chi_list': {0: 20, 10: 40},
                 'N_sweep_update': 3,
                 'max_sweeps': 20,
                 }
    dmrg.run(psi, M, dmrg_pars)
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
        f(*f_args[1:])
